"""S3 file profiler — DuckDB-backed profiling for S3-hosted CSV, Parquet, and JSON files."""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import UTC, date, datetime
from datetime import time as time_type
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

import duckdb

from core.api.v1.schemas.profiler import (
    BooleanColumnStats,
    ColumnMetadata,
    ColumnStatistics,
    DatabaseMetadata,
    NumericColumnStats,
    ProfilingConfig,
    ProfilingResponse,
    SchemaMetadata,
    StringColumnStats,
    TableMetadata,
    TemporalColumnStats,
    TopValueEntry,
)
from core.api.v1.schemas.s3 import S3ConnectionConfig, S3FileProfilingRequest, S3PathConfig
from core.exceptions import ExternalServiceError
from core.exceptions.base import ProfilingTimeoutError
from core.logging import get_logger
from core.observability import get_tracer
from core.services.base import BaseProfilerService
from core.services.column_stats import ColumnTypeCategory, classify_columns

if TYPE_CHECKING:
    from core.services.task_manager import ProgressReporter

logger = get_logger(__name__)
tracer = get_tracer()


# ── JSON serialisation helper ──────────────────────────────────────────────────


def _make_json_safe(value: Any) -> Any:
    """Convert a value returned by DuckDB to a JSON-serialisable Python primitive."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date, time_type)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex() if not isinstance(value, memoryview) else bytes(value).hex()
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


# ── Identifier / SQL helpers ──────────────────────────────────────────────────


def _esc(name: str) -> str:
    """Escape a DuckDB identifier for use in double-quoted SQL identifiers."""
    return name.replace('"', '""')


def _esc_sql_string(value: str) -> str:
    """Escape a SQL string literal value (single-quote safe)."""
    return value.replace("'", "''")


def _fetch_as_dicts(cur: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Convert a DuckDB cursor result into a list of row dicts."""
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return [dict(zip(cols, row)) for row in rows]  # noqa: B905


# ── S3 URI / expression builders ──────────────────────────────────────────────


def _build_s3_uri(path_cfg: S3PathConfig) -> str:
    """Build the s3:// URI for a path config."""
    key = path_cfg.key.lstrip("/")
    return f"s3://{path_cfg.bucket}/{key}"


def _build_read_expression(path_cfg: S3PathConfig) -> str:
    """Build a DuckDB table-function expression for reading a file or glob from S3.

    Returns a SQL fragment suitable for use in a FROM clause, e.g.
    ``read_csv_auto('s3://bucket/data/*.csv', header=true)``.
    """
    uri = _build_s3_uri(path_cfg)
    fmt = path_cfg.file_format

    if fmt == "auto":
        # Infer format from the terminal path segment (handles globs by checking the non-glob portion).
        terminal = path_cfg.key.rstrip("/").split("/")[-1]
        lc = terminal.lower()
        if lc.endswith((".parquet", ".parq")):
            fmt = "parquet"
        elif lc.endswith((".json", ".jsonl", ".ndjson")):
            fmt = "json"
        else:
            fmt = "csv"

    if fmt == "parquet":
        return f"read_parquet('{_esc_sql_string(uri)}')"
    if fmt == "json":
        return f"read_json_auto('{_esc_sql_string(uri)}')"

    # CSV (explicit or auto-inferred)
    opts: list[str] = []
    opts.append("header=true" if path_cfg.has_header else "header=false")
    if path_cfg.delimiter:
        opts.append(f"delim='{_esc_sql_string(path_cfg.delimiter)}'")
    opts_str = (", " + ", ".join(opts)) if opts else ""
    return f"read_csv_auto('{_esc_sql_string(uri)}'{opts_str})"


def _derive_table_name(path_cfg: S3PathConfig) -> str:
    """Derive a display table name from an S3PathConfig.

    Uses ``path_cfg.name`` if set; otherwise strips the last key segment of
    common file extensions and normalises to an identifier-safe string.
    """
    if path_cfg.name:
        return path_cfg.name
    segment = path_cfg.key.rstrip("/").split("/")[-1]
    for ext in (".csv", ".parquet", ".parq", ".json", ".jsonl", ".ndjson", ".gz", ".zst", ".bz2", ".lz4"):
        if segment.lower().endswith(ext):
            segment = segment[: -len(ext)]
    segment = re.sub(r"[^a-zA-Z0-9_]", "_", segment).strip("_")
    return segment or "s3_file"


# ── DuckDB session setup ──────────────────────────────────────────────────────


def _setup_duckdb_s3_session(conn: duckdb.DuckDBPyConnection, conn_cfg: S3ConnectionConfig) -> None:
    """Install and load httpfs, then configure S3 credentials for the session.

    When ``aws_access_key_id`` is ``None``, DuckDB's httpfs falls back to
    ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` environment variables and
    the EC2 instance metadata service (IAM role). Do not set empty strings —
    that would override the credential chain with blank values.
    """
    conn.execute("INSTALL httpfs")
    conn.execute("LOAD httpfs")

    region = _esc_sql_string(conn_cfg.aws_region)
    conn.execute(f"SET s3_region='{region}'")

    if conn_cfg.aws_access_key_id:
        key_id = _esc_sql_string(conn_cfg.aws_access_key_id)
        conn.execute(f"SET s3_access_key_id='{key_id}'")

    if conn_cfg.aws_secret_access_key:
        secret = _esc_sql_string(conn_cfg.aws_secret_access_key.get_secret_value())
        conn.execute(f"SET s3_secret_access_key='{secret}'")

    if conn_cfg.aws_session_token:
        token = _esc_sql_string(conn_cfg.aws_session_token.get_secret_value())
        conn.execute(f"SET s3_session_token='{token}'")

    if conn_cfg.endpoint_url:
        # DuckDB expects the endpoint without scheme (host:port only).
        endpoint = conn_cfg.endpoint_url
        if "://" in endpoint:
            endpoint = endpoint.split("://", 1)[1]
        conn.execute(f"SET s3_endpoint='{_esc_sql_string(endpoint)}'")
        # Custom endpoints (MinIO, LocalStack) normally use path-style addressing.
        conn.execute("SET s3_url_style='path'")
        if not conn_cfg.use_ssl:
            conn.execute("SET s3_use_ssl=false")


# ── Per-path synchronous profiling (runs inside asyncio.to_thread) ────────────


def _fetch_columns_sync(conn: duckdb.DuckDBPyConnection, read_expr: str) -> list[dict[str, Any]]:
    """Return column metadata by running DESCRIBE on the read expression."""
    cur = conn.execute(f"DESCRIBE SELECT * FROM {read_expr} LIMIT 0")
    rows = _fetch_as_dicts(cur)
    columns = []
    for i, row in enumerate(rows, start=1):
        columns.append(
            {
                "name": row["column_name"],
                "ordinal_position": i,
                "data_type": row["column_type"].lower(),
                "is_nullable": row.get("null", "YES") != "NO",
                "column_default": None,
                "character_maximum_length": None,
                "numeric_precision": None,
                "numeric_scale": None,
                "is_primary_key": False,
                "description": None,
                "enum_values": None,
                "sample_values": None,
                "statistics": None,
            }
        )
    return columns


def _fetch_row_count_sync(conn: duckdb.DuckDBPyConnection, read_expr: str) -> int:
    """Return the total row count for a file expression."""
    cur = conn.execute(f"SELECT COUNT(*) AS cnt FROM {read_expr}")
    row = cur.fetchone()
    return int(row[0]) if row else 0


def _fetch_sample_data_sync(
    conn: duckdb.DuckDBPyConnection,
    read_expr: str,
    sample_size: int,
) -> list[dict[str, Any]]:
    """Fetch sample rows using DuckDB's reservoir sampler."""
    cur = conn.execute(f"SELECT * FROM {read_expr} USING SAMPLE {sample_size} ROWS")
    return _fetch_as_dicts(cur)


# ── Column statistics (sync, 4-phase — mirrors PostgresProfiler) ──────────────


def _fetch_column_stats_sync(
    conn: duckdb.DuckDBPyConnection,
    read_expr: str,
    columns: list[dict[str, Any]],
    config: ProfilingConfig,
) -> dict[str, ColumnStatistics]:
    """Compute column-level statistics using DuckDB SQL.

    Follows the same 4-phase approach as PostgresProfiler:
    Phase 1 — common stats (single scan)
    Phase 2 — type-specific stats
    Phase 3 — top values for low-cardinality columns
    Phase 4 — outlier counts for numeric columns

    DuckDB supports PERCENTILE_CONT and COUNT(*) FILTER (WHERE ...) identically
    to PostgreSQL, so the query patterns are directly transferable.
    """
    try:
        return _do_fetch_column_stats_sync(conn, read_expr, columns, config)
    except Exception:
        logger.warning("Column statistics fetch failed; skipping", read_expr=read_expr[:80], exc_info=True)
        return {}


def _do_fetch_column_stats_sync(
    conn: duckdb.DuckDBPyConnection,
    read_expr: str,
    columns: list[dict[str, Any]],
    config: ProfilingConfig,
) -> dict[str, ColumnStatistics]:
    grouped = classify_columns(columns)

    # ── Phase 1: common stats (single table scan) ─────────────────────────
    count_parts: list[str] = []
    for col in columns:
        c = _esc(col["name"])
        count_parts.append(f'COUNT("{c}") AS "{c}__non_null"')
        count_parts.append(f'COUNT(DISTINCT "{c}") AS "{c}__distinct"')

    common_query = f"SELECT COUNT(*) AS total_count, {', '.join(count_parts)} FROM {read_expr}"
    common_cur = conn.execute(common_query)
    common_row = dict(zip([d[0] for d in common_cur.description], common_cur.fetchone() or []))  # noqa: B905
    total_count: int = common_row.get("total_count", 0) or 0

    common_by_col: dict[str, dict[str, Any]] = {}
    for col in columns:
        cn = col["name"]
        c = _esc(cn)
        non_null = common_row.get(f"{c}__non_null", 0) or 0
        distinct = common_row.get(f"{c}__distinct", 0) or 0
        null_count = total_count - non_null
        common_by_col[cn] = {
            "total_count": total_count,
            "null_count": null_count,
            "null_percentage": round(null_count / total_count * 100, 2) if total_count else 0.0,
            "distinct_count": distinct,
            "distinct_percentage": round(distinct / total_count * 100, 2) if total_count else 0.0,
        }

    # ── Phase 2: type-specific stats ──────────────────────────────────────
    numeric_stats_map: dict[str, NumericColumnStats] = {}
    string_stats_map: dict[str, StringColumnStats] = {}
    boolean_stats_map: dict[str, BooleanColumnStats] = {}
    temporal_stats_map: dict[str, TemporalColumnStats] = {}

    numeric_cols = grouped[ColumnTypeCategory.NUMERIC]
    string_cols = grouped[ColumnTypeCategory.STRING]
    boolean_cols = grouped[ColumnTypeCategory.BOOLEAN]
    temporal_cols = grouped[ColumnTypeCategory.TEMPORAL]

    if numeric_cols:
        numeric_stats_map = _fetch_numeric_stats_sync(conn, read_expr, numeric_cols)
    if string_cols:
        string_stats_map = _fetch_string_stats_sync(conn, read_expr, string_cols)
    if boolean_cols:
        boolean_stats_map = _fetch_boolean_stats_sync(conn, read_expr, boolean_cols)
    if temporal_cols:
        temporal_stats_map = _fetch_temporal_stats_sync(conn, read_expr, temporal_cols)

    # ── Phase 3: top values for low-cardinality columns ───────────────────
    top_values_by_col: dict[str, list[TopValueEntry]] = {}
    for col in boolean_cols:
        top_values_by_col[col["name"]] = _fetch_top_values_sync(
            conn, read_expr, col["name"], total_count, config.top_values_limit
        )
    for col in string_cols:
        cn = col["name"]
        if common_by_col[cn]["distinct_count"] <= config.top_values_cardinality_threshold:
            top_values_by_col[cn] = _fetch_top_values_sync(conn, read_expr, cn, total_count, config.top_values_limit)

    # ── Phase 4: IQR-based outlier counts for numeric columns ─────────────
    for cn, ns in numeric_stats_map.items():
        if ns.p25 is not None and ns.p75 is not None:
            oc = _fetch_outlier_count_sync(conn, read_expr, cn, ns.p25, ns.p75)
            numeric_stats_map[cn] = ns.model_copy(update={"outlier_count": oc})

    # ── Assemble ColumnStatistics per column ──────────────────────────────
    result: dict[str, ColumnStatistics] = {}
    for col in columns:
        cn = col["name"]
        result[cn] = ColumnStatistics(
            **common_by_col[cn],
            numeric=numeric_stats_map.get(cn),
            string=string_stats_map.get(cn),
            boolean=boolean_stats_map.get(cn),
            temporal=temporal_stats_map.get(cn),
            top_values=top_values_by_col.get(cn),
        )
    return result


def _fetch_numeric_stats_sync(
    conn: duckdb.DuckDBPyConnection,
    read_expr: str,
    columns: list[dict[str, Any]],
) -> dict[str, NumericColumnStats]:
    parts: list[str] = []
    for col in columns:
        c = _esc(col["name"])
        parts.extend(
            [
                f'MIN("{c}")::DOUBLE AS "{c}__min"',
                f'MAX("{c}")::DOUBLE AS "{c}__max"',
                f'AVG("{c}")::DOUBLE AS "{c}__mean"',
                f'STDDEV("{c}")::DOUBLE AS "{c}__stddev"',
                f'VARIANCE("{c}")::DOUBLE AS "{c}__variance"',
                f'SUM("{c}")::DOUBLE AS "{c}__sum"',
                f'PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "{c}") AS "{c}__p50"',
                f'PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY "{c}") AS "{c}__p5"',
                f'PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "{c}") AS "{c}__p25"',
                f'PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "{c}") AS "{c}__p75"',
                f'PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY "{c}") AS "{c}__p95"',
                f'COUNT(*) FILTER (WHERE "{c}" = 0) AS "{c}__zero_count"',
                f'COUNT(*) FILTER (WHERE "{c}" < 0) AS "{c}__negative_count"',
            ]
        )
    cur = conn.execute(f"SELECT {', '.join(parts)} FROM {read_expr}")
    row = dict(zip([d[0] for d in cur.description], cur.fetchone() or []))  # noqa: B905
    result: dict[str, NumericColumnStats] = {}
    for col in columns:
        cn = col["name"]
        c = _esc(cn)
        result[cn] = NumericColumnStats(
            min=_make_json_safe(row.get(f"{c}__min")),
            max=_make_json_safe(row.get(f"{c}__max")),
            mean=_make_json_safe(row.get(f"{c}__mean")),
            median=_make_json_safe(row.get(f"{c}__p50")),
            stddev=_make_json_safe(row.get(f"{c}__stddev")),
            variance=_make_json_safe(row.get(f"{c}__variance")),
            sum=_make_json_safe(row.get(f"{c}__sum")),
            p5=_make_json_safe(row.get(f"{c}__p5")),
            p25=_make_json_safe(row.get(f"{c}__p25")),
            p75=_make_json_safe(row.get(f"{c}__p75")),
            p95=_make_json_safe(row.get(f"{c}__p95")),
            zero_count=row.get(f"{c}__zero_count"),
            negative_count=row.get(f"{c}__negative_count"),
        )
    return result


def _fetch_string_stats_sync(
    conn: duckdb.DuckDBPyConnection,
    read_expr: str,
    columns: list[dict[str, Any]],
) -> dict[str, StringColumnStats]:
    parts: list[str] = []
    for col in columns:
        c = _esc(col["name"])
        parts.extend(
            [
                f'MIN(LENGTH("{c}")) AS "{c}__min_length"',
                f'MAX(LENGTH("{c}")) AS "{c}__max_length"',
                f'AVG(LENGTH("{c}"))::DOUBLE AS "{c}__avg_length"',
                f'COUNT(*) FILTER (WHERE "{c}" = \'\') AS "{c}__empty_count"',
            ]
        )
    cur = conn.execute(f"SELECT {', '.join(parts)} FROM {read_expr}")
    row = dict(zip([d[0] for d in cur.description], cur.fetchone() or []))  # noqa: B905
    result: dict[str, StringColumnStats] = {}
    for col in columns:
        cn = col["name"]
        c = _esc(cn)
        result[cn] = StringColumnStats(
            min_length=row.get(f"{c}__min_length"),
            max_length=row.get(f"{c}__max_length"),
            avg_length=_make_json_safe(row.get(f"{c}__avg_length")),
            empty_count=row.get(f"{c}__empty_count"),
        )
    return result


def _fetch_boolean_stats_sync(
    conn: duckdb.DuckDBPyConnection,
    read_expr: str,
    columns: list[dict[str, Any]],
) -> dict[str, BooleanColumnStats]:
    parts: list[str] = []
    for col in columns:
        c = _esc(col["name"])
        parts.extend(
            [
                f'COUNT(*) FILTER (WHERE "{c}" = TRUE) AS "{c}__true_count"',
                f'COUNT(*) FILTER (WHERE "{c}" = FALSE) AS "{c}__false_count"',
            ]
        )
    cur = conn.execute(f"SELECT {', '.join(parts)} FROM {read_expr}")
    row = dict(zip([d[0] for d in cur.description], cur.fetchone() or []))  # noqa:B905
    result: dict[str, BooleanColumnStats] = {}
    for col in columns:
        cn = col["name"]
        c = _esc(cn)
        tc = row.get(f"{c}__true_count", 0) or 0
        fc = row.get(f"{c}__false_count", 0) or 0
        total = tc + fc
        result[cn] = BooleanColumnStats(
            true_count=tc,
            false_count=fc,
            true_percentage=round(tc / total * 100, 2) if total else 0.0,
        )
    return result


def _fetch_temporal_stats_sync(
    conn: duckdb.DuckDBPyConnection,
    read_expr: str,
    columns: list[dict[str, Any]],
) -> dict[str, TemporalColumnStats]:
    parts: list[str] = []
    for col in columns:
        c = _esc(col["name"])
        parts.extend(
            [
                f'MIN("{c}")::TEXT AS "{c}__min"',
                f'MAX("{c}")::TEXT AS "{c}__max"',
            ]
        )
    cur = conn.execute(f"SELECT {', '.join(parts)} FROM {read_expr}")
    row = dict(zip([d[0] for d in cur.description], cur.fetchone() or []))  # noqa:B905
    result: dict[str, TemporalColumnStats] = {}
    for col in columns:
        cn = col["name"]
        c = _esc(cn)
        result[cn] = TemporalColumnStats(
            min=row.get(f"{c}__min"),
            max=row.get(f"{c}__max"),
        )
    return result


def _fetch_top_values_sync(
    conn: duckdb.DuckDBPyConnection,
    read_expr: str,
    col_name: str,
    total_count: int,
    limit: int,
) -> list[TopValueEntry]:
    c = _esc(col_name)
    cur = conn.execute(
        f'SELECT "{c}"::TEXT AS value, COUNT(*) AS cnt '
        f"FROM {read_expr} "
        f'WHERE "{c}" IS NOT NULL '
        f'GROUP BY "{c}" '
        "ORDER BY cnt DESC "
        "LIMIT ?",
        [limit],
    )
    rows = _fetch_as_dicts(cur)
    return [
        TopValueEntry(
            value=r["value"],
            count=r["cnt"],
            percentage=round(r["cnt"] / total_count * 100, 2) if total_count else 0.0,
        )
        for r in rows
    ]


def _fetch_outlier_count_sync(
    conn: duckdb.DuckDBPyConnection,
    read_expr: str,
    col_name: str,
    q1: float,
    q3: float,
) -> int:
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    c = _esc(col_name)
    cur = conn.execute(
        f'SELECT COUNT(*) AS outlier_count FROM {read_expr} WHERE "{c}" IS NOT NULL AND ("{c}" < ? OR "{c}" > ?)',
        [lower, upper],
    )
    row = cur.fetchone()
    return int(row[0]) if row else 0


# ── Per-path synchronous profiling entrypoint ────────────────────────────────


def _profile_one_path_sync(
    conn_cfg: S3ConnectionConfig,
    path_cfg: S3PathConfig,
    config: ProfilingConfig,
) -> dict[str, Any]:
    """Profile a single S3 path entirely within one synchronous DuckDB session.

    Creates its own DuckDB in-memory connection so that multiple paths can be
    profiled concurrently via asyncio.to_thread without thread-safety concerns.
    """
    conn = duckdb.connect(":memory:")
    try:
        _setup_duckdb_s3_session(conn, conn_cfg)
        read_expr = _build_read_expression(path_cfg)
        table_name = _derive_table_name(path_cfg)

        columns = _fetch_columns_sync(conn, read_expr)

        row_count: int | None = None
        if config.include_row_counts:
            row_count = _fetch_row_count_sync(conn, read_expr)

        sample_data: list[dict[str, Any]] | None = None
        if config.include_sample_data:
            sample_data = _fetch_sample_data_sync(conn, read_expr, config.sample_size)

        if sample_data is not None:
            for col in columns:
                col["sample_values"] = [_make_json_safe(row.get(col["name"])) for row in sample_data]

        col_stats: dict[str, ColumnStatistics] = {}
        if config.include_column_stats and columns:
            col_stats = _fetch_column_stats_sync(conn, read_expr, columns, config)
        for col in columns:
            col["statistics"] = col_stats.get(col["name"])

        return {
            "name": table_name,
            "schema": path_cfg.bucket,
            "owner": conn_cfg.endpoint_url or "aws_s3",
            "description": f"s3://{path_cfg.bucket}/{path_cfg.key}",
            "size_bytes": None,
            "total_size_bytes": None,
            "row_count": row_count,
            "columns": columns,
            "indexes": [],
            "relationships": [],
            "data_freshness": None,
        }
    finally:
        conn.close()


def _test_connection_sync(conn_cfg: S3ConnectionConfig, path_cfg: S3PathConfig) -> None:
    """Verify S3 connectivity by reading 1 row from the first configured path.

    Raises ``duckdb.Error`` on failure; the caller maps it to a service exception.
    """
    conn = duckdb.connect(":memory:")
    try:
        _setup_duckdb_s3_session(conn, conn_cfg)
        read_expr = _build_read_expression(path_cfg)
        conn.execute(f"SELECT * FROM {read_expr} LIMIT 1").fetchall()
    finally:
        conn.close()


def _map_duckdb_error(exc: Exception, path_cfg: S3PathConfig) -> ExternalServiceError:
    """Map a DuckDB error into a service-level ExternalServiceError."""
    uri = f"s3://{path_cfg.bucket}/{path_cfg.key}"
    msg = str(exc)
    if "403" in msg or "Access Denied" in msg or "AccessDenied" in msg or "Forbidden" in msg:
        return ExternalServiceError(
            message=f"AWS S3 access denied for {uri}. Check credentials and bucket permissions.",
            service_name="AWS S3",
            status_code=403,
        )
    if "404" in msg or "NoSuchKey" in msg or "NoSuchBucket" in msg or "Not Found" in msg:
        return ExternalServiceError(
            message=f"S3 path not found: {uri}",
            service_name="AWS S3",
            status_code=404,
        )
    return ExternalServiceError(
        message=f"DuckDB/S3 error for {uri}: {exc}",
        service_name="DuckDB/S3",
    )


# ── BaseProfilerService implementation ────────────────────────────────────────


class S3FileProfilerService(BaseProfilerService):
    """Profiler for S3-hosted structured files (CSV, Parquet, JSON) via DuckDB httpfs.

    Overrides :meth:`profile` to manage DuckDB connections directly, since S3
    access requires httpfs extension setup that cannot be routed through the
    generic ``create_connector()`` factory.  Each S3 path is profiled in its own
    DuckDB in-memory session running inside ``asyncio.to_thread``.
    """

    service_name = "AWS S3 (DuckDB)"
    span_name = "profiler.s3_file"

    def _span_attributes(self, conn: S3ConnectionConfig) -> dict[str, str | int]:
        attrs: dict[str, str | int] = {
            "db.system": "s3",
            "db.name": "s3_file",
            "s3.region": conn.aws_region,
        }
        if conn.endpoint_url:
            attrs["s3.endpoint"] = conn.endpoint_url
        return attrs

    def _log_context(self, conn: S3ConnectionConfig) -> dict[str, Any]:
        return {
            "host": conn.endpoint_url or f"s3.{conn.aws_region}.amazonaws.com",
            "database": "s3_file",
            "region": conn.aws_region,
            "has_explicit_credentials": conn.aws_access_key_id is not None,
        }

    async def _run(self, connector: Any, config: ProfilingConfig, progress: ProgressReporter | None = None) -> dict[str, Any]:
        """Not used — profile() is fully overridden for S3."""
        raise NotImplementedError("S3FileProfilerService manages DuckDB connections directly; call profile() instead.")

    def _assemble_response(self, raw: dict[str, Any], profiled_at: datetime) -> ProfilingResponse:
        """Map raw S3 profiling dict into a ProfilingResponse.

        Mapping:
          raw["database"]["name"]  →  DatabaseMetadata.name (bucket name)
          raw["schemas"][0]        →  single SchemaMetadata per request
          raw["schemas"][0]["tables"]  →  one TableMetadata per S3 path
        """
        db = raw["database"]
        database_meta = DatabaseMetadata(
            name=db["name"],
            version=db.get("version", "S3"),
            encoding=db.get("encoding", "UTF-8"),
            size_bytes=db.get("size_bytes") or 0,
        )

        schema_metas: list[SchemaMetadata] = []
        for schema in raw["schemas"]:
            tables: list[TableMetadata] = []
            for tbl in schema["tables"]:
                columns = [ColumnMetadata(**col) for col in tbl["columns"]]
                tables.append(
                    TableMetadata(
                        name=tbl["name"],
                        schema=tbl["schema"],
                        owner=tbl.get("owner", ""),
                        description=tbl.get("description"),
                        row_count=tbl.get("row_count"),
                        size_bytes=tbl.get("size_bytes"),
                        total_size_bytes=tbl.get("total_size_bytes"),
                        data_freshness=None,
                        columns=columns,
                        indexes=[],
                        relationships=[],
                    )
                )
            schema_metas.append(
                SchemaMetadata(
                    name=schema["name"],
                    owner=schema.get("owner", ""),
                    tables=tables,
                    views=[],
                )
            )

        return ProfilingResponse(
            profiled_at=profiled_at,
            database=database_meta,
            schemas=schema_metas,
        )

    # ── Override profile() ────────────────────────────────────────────────────

    async def profile(
        self,
        body: S3FileProfilingRequest,
        progress: ProgressReporter | None = None,
    ) -> ProfilingResponse:
        """Full S3 profiling flow: connection test → per-path profiling → LLM augmentation.

        Each path is profiled in its own ``asyncio.to_thread`` call with a fresh
        DuckDB in-memory connection, enabling concurrent path profiling without
        thread-safety concerns.
        """
        conn_cfg = body.connection
        raw_cfg = body.config
        cfg: ProfilingConfig = ProfilingConfig.model_validate(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
        ctx = self._log_context(conn_cfg)
        profiled_at = datetime.now(UTC)

        with tracer.start_as_current_span(self.span_name) as span:
            for key, value in self._span_attributes(conn_cfg).items():
                span.set_attribute(key, value)

            try:
                if progress:
                    progress.update(phase="connecting", percent=5)
                    await progress.flush()

                # Connection test — probe the first path.
                await asyncio.to_thread(_test_connection_sync, conn_cfg, body.paths[0])
                logger.info("S3 connection test passed", **ctx)

                if progress:
                    progress.update(phase="profiling", percent=10)
                    await progress.flush()

                # Profile all paths concurrently; each runs in its own thread with its own DuckDB conn.
                async def _profile_path_async(path_cfg: S3PathConfig) -> dict[str, Any]:
                    return await asyncio.to_thread(_profile_one_path_sync, conn_cfg, path_cfg, cfg)

                raw_tables = await asyncio.wait_for(
                    asyncio.gather(
                        *[_profile_path_async(p) for p in body.paths],
                        return_exceptions=True,
                    ),
                    timeout=cfg.timeout_seconds,
                )

            except TimeoutError:
                logger.warning("S3 profiling timed out", **ctx, timeout_seconds=cfg.timeout_seconds)
                raise ProfilingTimeoutError(
                    message=f"S3 profiling exceeded the configured timeout of {cfg.timeout_seconds}s",
                    timeout_seconds=cfg.timeout_seconds,
                ) from None

            except duckdb.Error as exc:
                raise _map_duckdb_error(exc, body.paths[0]) from exc

            except ExternalServiceError:
                raise

            except Exception as exc:
                logger.error("Unexpected error during S3 profiling", **ctx, error=str(exc), exc_info=True)
                raise ExternalServiceError(
                    message=f"S3 profiling failed: {exc}",
                    service_name=self.service_name,
                ) from exc

        # Separate successful path results from failures; log failures and continue.
        tables: list[dict[str, Any]] = []
        for path, result in zip(body.paths, raw_tables):  # noqa: B905
            if isinstance(result, BaseException):
                logger.warning(
                    "Failed to profile S3 path; skipping",
                    path=f"s3://{path.bucket}/{path.key}",
                    error=str(result),
                )
            else:
                tables.append(result)

        if not tables:
            raise ExternalServiceError(
                message="All configured S3 paths failed to profile. Check credentials and path validity.",
                service_name=self.service_name,
            )

        bucket_name = body.paths[0].bucket
        endpoint = conn_cfg.endpoint_url or f"s3.{conn_cfg.aws_region}.amazonaws.com"

        raw = {
            "database": {
                "name": bucket_name,
                "version": "S3",
                "encoding": "UTF-8",
                "size_bytes": 0,
            },
            "schemas": [
                {
                    "name": bucket_name,
                    "owner": endpoint,
                    "tables": tables,
                    "views": [],
                }
            ],
        }

        logger.info("S3 profiling complete", **ctx, path_count=len(body.paths), table_count=len(tables))
        response = self._assemble_response(raw, profiled_at)

        # LLM augmentation — same best-effort steps as other profilers.
        if cfg.augment_descriptions:
            if progress:
                progress.update(phase="augmenting", percent=70, detail={"augmentation_step": "table_descriptions"})
                await progress.flush()
            response = await self._augment_response(response, cfg)

        if cfg.augment_column_descriptions:
            if progress:
                progress.update(phase="augmenting", percent=80, detail={"augmentation_step": "column_descriptions"})
                await progress.flush()
            response = await self._augment_column_response(response, cfg)

        if cfg.augment_glossary:
            if progress:
                progress.update(phase="augmenting", percent=85, detail={"augmentation_step": "glossary"})
                await progress.flush()
            response = await self._augment_glossary_response(response, cfg)

        if cfg.infer_kpis:
            if progress:
                progress.update(phase="augmenting", percent=90, detail={"augmentation_step": "kpis"})
                await progress.flush()
            response = await self._augment_kpis_response(response, cfg)

        if progress:
            progress.update(phase="completed", percent=100)
            await progress.flush()

        return response

    # Keep _augment_* timing logs consistent with BaseProfilerService
    async def _augment_response(self, response: ProfilingResponse, cfg: ProfilingConfig) -> ProfilingResponse:
        t = time.monotonic()
        result = await super()._augment_response(response, cfg)
        logger.debug("S3 table description augmentation", duration_s=round(time.monotonic() - t, 3))
        return result
