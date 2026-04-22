"""PostgreSQL profiler — query service and BaseProfiler implementation."""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.services.task_manager import ProgressReporter
from uuid import UUID

from ignite_data_connectors import BaseConnector, PostgresConfig

from core.api.v1.schemas.profiler import (
    BooleanColumnStats,
    ColumnMetadata,
    ColumnStatistics,
    DatabaseMetadata,
    DataFreshnessInfo,
    IndexMetadata,
    NumericColumnStats,
    ProfilingConfig,
    ProfilingResponse,
    RelationshipMetadata,
    SchemaMetadata,
    StringColumnStats,
    TableMetadata,
    TemporalColumnStats,
    TopValueEntry,
    ViewMetadata,
)
from core.logging import get_logger
from core.observability import get_tracer
from core.services.base import BaseProfilerService
from core.services.column_stats import ColumnTypeCategory, classify_columns

from . import queries as q

logger = get_logger(__name__)
tracer = get_tracer()


# ── JSON serialisation helper ──────────────────────────────────────────────────


def _make_json_safe(value: Any) -> Any:
    """Convert a value to a JSON-serialisable Python primitive.

    asyncpg returns typed Python objects for many PostgreSQL types.
    This converts them to primitives that Pydantic and ``json.dumps``
    can handle without custom encoders.
    """
    if value is None:
        return None
    if isinstance(value, (str | int | float | bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime | date | time)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, memoryview):
        return bytes(value).hex()
    if isinstance(value, (bytes | bytearray)):
        return value.hex()
    if isinstance(value, (list | tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}
    # Fallback: only embed if already JSON-serialisable
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


# ── Query service ──────────────────────────────────────────────────────────────


class PostgresProfiler:
    """Orchestrates metadata extraction from a connected PostgreSQL database.

    The caller is responsible for opening and closing the connector; this
    service never calls ``connect()`` / ``disconnect()`` directly.

    Usage::

        async with create_connector(pg_config) as db:
            service = PostgresProfilerService(db, profiling_config)
            result = await service.profile()
    """

    def __init__(self, connector: BaseConnector, config: ProfilingConfig) -> None:
        self._db = connector
        self._config = config

    # ── Public orchestration ───────────────────────────────────────────────────

    async def profile(self) -> dict[str, Any]:
        """Run the full profiling pass and return a raw dict.

        Returns:
            dict with ``database`` and ``schemas`` keys.

        """
        with tracer.start_as_current_span("profiler.profile") as span:
            db_meta, schema_names = await asyncio.gather(
                self._fetch_database_metadata(),
                self._fetch_schema_names(),
            )
            span.set_attribute("profiler.schemas_discovered", len(schema_names))
            logger.info(
                "Profiling schemas",
                schema_count=len(schema_names),
                database=db_meta["name"],
            )

            schema_results = await asyncio.gather(
                *[self._profile_schema(s) for s in schema_names],
            )

            return {
                "database": db_meta,
                "schemas": list(schema_results),
            }

    # ── Database-level ─────────────────────────────────────────────────────────

    async def _fetch_database_metadata(self) -> dict[str, Any]:
        """Fetch database name, version, encoding, and size."""
        with tracer.start_as_current_span("profiler.fetch_database_metadata"):
            db_row = await self._db.fetch_one(q.DATABASE_METADATA)
            return {
                "name": db_row["db_name"] if db_row else "",
                "version": db_row["version"] if db_row else "",
                "encoding": db_row["encoding"] if db_row else "",
                "size_bytes": db_row["size_bytes"] if db_row else 0,
            }

    # ── Schema discovery ────────────────────────────────────────────────────────

    async def _fetch_schema_names(self) -> list[str]:
        """Return schema names after applying include/exclude filters."""
        with tracer.start_as_current_span("profiler.fetch_schema_names"):
            exclude = list(self._config.exclude_schemas)
            placeholders = ", ".join(f"${i + 1}" for i in range(len(exclude)))
            query = q.SCHEMAS.format(placeholders=placeholders)
            rows = await self._db.fetch_all(query, tuple(exclude))
            all_schemas = [r["schema_name"] for r in rows]

            if self._config.include_schemas is not None:
                include_set = set(self._config.include_schemas)
                all_schemas = [s for s in all_schemas if s in include_set]

            logger.debug("Discovered schemas", schemas=all_schemas)
            return all_schemas

    # ── Schema-level profiling ──────────────────────────────────────────────────

    async def _profile_schema(self, schema_name: str) -> dict[str, Any]:
        """Profile a single schema: fetch tables and views concurrently."""
        with tracer.start_as_current_span("profiler.profile_schema") as span:
            span.set_attribute("profiler.schema", schema_name)

            table_rows, view_rows = await asyncio.gather(
                self._fetch_tables(schema_name),
                self._fetch_views(schema_name),
            )

            filtered_tables = self._filter_tables(table_rows)

            sem = asyncio.Semaphore(self._config.max_concurrent_tables)

            async def _bounded_profile(row: dict[str, Any]) -> dict[str, Any]:
                async with sem:
                    return await self._profile_table(row, schema_name)

            table_results = await asyncio.gather(
                *[_bounded_profile(row) for row in filtered_tables],
            )

            # Derive schema owner from first table or view owner; fall back to empty string.
            owner = ""
            if table_rows:
                owner = table_rows[0].get("owner", "")
            elif view_rows:
                owner = view_rows[0].get("owner", "")

            return {
                "name": schema_name,
                "owner": owner,
                "tables": list(table_results),
                "views": view_rows,
            }

    def _filter_tables(self, table_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply ``include_tables`` and ``exclude_tables`` filters."""
        result = table_rows
        if self._config.include_tables is not None:
            include_set = set(self._config.include_tables)
            result = [r for r in result if r["table_name"] in include_set]
        if self._config.exclude_tables:
            exclude_set = set(self._config.exclude_tables)
            result = [r for r in result if r["table_name"] not in exclude_set]
        return result

    # ── Table-level ─────────────────────────────────────────────────────────────

    async def _fetch_tables(self, schema_name: str) -> list[dict[str, Any]]:
        """Fetch basic table metadata for all tables in a schema."""
        rows = await self._db.fetch_all(q.TABLES_IN_SCHEMA, (schema_name,))
        return [dict(r) for r in rows]

    async def _profile_table(self, table_row: dict[str, Any], schema_name: str) -> dict[str, Any]:
        """Profile a single table: columns, indexes, FK, row count, sample, freshness."""
        table_name = table_row["table_name"]
        with tracer.start_as_current_span("profiler.profile_table") as span:
            span.set_attribute("profiler.table", f"{schema_name}.{table_name}")

            # Build the task map based on config flags.
            tasks: dict[str, Any] = {
                "columns": self._fetch_columns(schema_name, table_name),
            }
            if self._config.include_row_counts:
                tasks["row_count"] = self._fetch_row_count(schema_name, table_name)
            if self._config.include_indexes:
                tasks["indexes"] = self._fetch_indexes(schema_name, table_name)
            if self._config.include_relationships:
                tasks["relationships"] = self._fetch_foreign_keys(schema_name, table_name)
            if self._config.include_data_freshness:
                tasks["data_freshness"] = self._fetch_data_freshness(schema_name, table_name)
            if self._config.include_sample_data:
                tasks["sample_data"] = self._fetch_sample_data(schema_name, table_name)

            keys = list(tasks.keys())
            results = await asyncio.gather(*tasks.values())
            resolved = dict(zip(keys, results, strict=True))

            columns: list[dict[str, Any]] = resolved.get("columns", [])
            sample_data: list[dict[str, Any]] | None = resolved.get("sample_data")
            if sample_data is not None:
                for col in columns:
                    col["sample_values"] = [row.get(col["name"]) for row in sample_data]

            # Fetch column statistics if enabled (runs after columns are
            # resolved because we need column names and data types).
            col_stats: dict[str, ColumnStatistics] = {}
            if self._config.include_column_stats and columns:
                col_stats = await self._fetch_column_stats(schema_name, table_name, columns)
            for col in columns:
                col["statistics"] = col_stats.get(col["name"])

            return {
                "name": table_name,
                "schema": schema_name,
                "owner": table_row.get("owner", ""),
                "description": table_row.get("description"),
                "size_bytes": table_row.get("size_bytes"),
                "total_size_bytes": table_row.get("total_size_bytes"),
                "row_count": resolved.get("row_count"),
                "columns": columns,
                "indexes": resolved.get("indexes"),
                "relationships": resolved.get("relationships"),
                "data_freshness": resolved.get("data_freshness"),
            }

    # ── Column-level ─────────────────────────────────────────────────────────────

    async def _fetch_columns(self, schema: str, table: str) -> list[dict[str, Any]]:
        """Fetch column metadata including primary key flag and enum values."""
        with tracer.start_as_current_span("profiler.fetch_columns"):
            col_rows, pk_rows = await asyncio.gather(
                self._db.fetch_all(q.COLUMNS_FOR_TABLE, (schema, table)),
                self._db.fetch_all(q.PRIMARY_KEY_COLUMNS, (schema, table)),
            )
            pk_columns = {r["column_name"] for r in pk_rows}
            columns = [
                {
                    "name": r["column_name"],
                    "ordinal_position": r["ordinal_position"],
                    "data_type": r["data_type"],
                    "is_nullable": r["is_nullable"],
                    "column_default": r["column_default"],
                    "character_maximum_length": r["character_maximum_length"],
                    "numeric_precision": r["numeric_precision"],
                    "numeric_scale": r["numeric_scale"],
                    "is_primary_key": r["column_name"] in pk_columns,
                    "description": r["description"],
                    "enum_values": None,
                    "sample_values": None,
                    "statistics": None,
                }
                for r in col_rows
            ]

            # Fetch enum values concurrently for all USER-DEFINED (enum) columns.
            enum_indices = [i for i, col in enumerate(columns) if col["data_type"] == "USER-DEFINED"]
            if enum_indices:
                regclass = f'"{schema}"."{table}"'
                enum_results = await asyncio.gather(
                    *[self._fetch_enum_values(regclass, columns[i]["name"]) for i in enum_indices]
                )
                for idx, values in zip(enum_indices, enum_results, strict=True):
                    columns[idx]["enum_values"] = values

            return columns

    async def _fetch_enum_values(self, regclass: str, column_name: str) -> list[str]:
        """Fetch allowed enum labels for a USER-DEFINED column."""
        rows = await self._db.fetch_all(q.ALLOWED_ENUM_VALUES, (regclass, column_name))
        return [r["enumlabel"] for r in rows]

    # ── Row count ────────────────────────────────────────────────────────────────

    async def _fetch_row_count(self, schema: str, table: str) -> int | None:
        """Return row count from pg_class statistics, falling back to exact COUNT(*).

        ``pg_class.reltuples`` is ``-1`` for tables that have never been
        analyzed.  In that case an exact ``COUNT(*)`` is issued instead.
        """
        safe_schema = schema.replace('"', '""')
        safe_table = table.replace('"', '""')
        exact_query = q.ROW_COUNT_EXACT.format(schema=safe_schema, table=safe_table)
        exact_row = await self._db.fetch_one(exact_query)
        return int(exact_row["row_count"]) if exact_row else None

    # ── Indexes ──────────────────────────────────────────────────────────────────

    async def _fetch_indexes(self, schema: str, table: str) -> list[dict[str, Any]]:
        """Fetch index metadata for a table."""
        rows = await self._db.fetch_all(q.INDEXES_FOR_TABLE, (table, schema))
        return [
            {
                "name": r["index_name"],
                "columns": list(r["columns"]),
                "is_unique": r["is_unique"],
                "is_primary": r["is_primary"],
                "index_type": r["index_type"],
            }
            for r in rows
        ]

    # ── Foreign keys ─────────────────────────────────────────────────────────────

    async def _fetch_foreign_keys(self, schema: str, table: str) -> list[dict[str, Any]]:
        """Fetch foreign key relationships for a table."""
        rows = await self._db.fetch_all(q.FOREIGN_KEYS_FOR_TABLE, (schema, table))
        return [
            {
                "constraint_name": r["constraint_name"],
                "from_column": r["column_name"],
                "to_schema": r["foreign_table_schema"],
                "to_table": r["foreign_table_name"],
                "to_column": r["foreign_column_name"],
                "on_update": r["update_rule"],
                "on_delete": r["delete_rule"],
            }
            for r in rows
        ]

    # ── Data freshness ───────────────────────────────────────────────────────────

    async def _fetch_data_freshness(self, schema: str, table: str) -> dict[str, Any] | None:
        """Fetch last analyze/vacuum timestamps from pg_stat_user_tables."""
        row = await self._db.fetch_one(q.DATA_FRESHNESS_FOR_TABLE, (schema, table))
        if row is None:
            return None
        return {
            "last_analyze": row["last_analyze"],
            "last_autoanalyze": row["last_autoanalyze"],
            "last_vacuum": row["last_vacuum"],
            "last_autovacuum": row["last_autovacuum"],
        }

    # ── Sample data ──────────────────────────────────────────────────────────────

    async def _fetch_sample_data(self, schema: str, table: str) -> list[dict[str, Any]]:
        """Fetch random sample rows and convert values to JSON-safe types.

        Schema and table identifiers are double-quoted to prevent SQL injection.
        They are **not** passed as parameters because PostgreSQL does not allow
        identifiers to be parameterised.
        """
        safe_schema = schema.replace('"', '""')
        safe_table = table.replace('"', '""')
        query = f'SELECT * FROM "{safe_schema}"."{safe_table}" ORDER BY RANDOM() LIMIT $1'
        rows = await self._db.fetch_all(query, (self._config.sample_size,))
        return [{k: _make_json_safe(v) for k, v in row.items()} for row in rows]

    # ── Column statistics ──────────────────────────────────────────────────────

    @staticmethod
    def _esc(name: str) -> str:
        """Escape a PostgreSQL identifier (double-quote safe)."""
        return name.replace('"', '""')

    async def _fetch_column_stats(
        self,
        schema: str,
        table: str,
        columns: list[dict[str, Any]],
    ) -> dict[str, ColumnStatistics]:
        """Compute column-level statistics for all columns in a table.

        Returns a mapping from column name to :class:`ColumnStatistics`.
        On error the method logs a warning and returns an empty dict.
        """
        with tracer.start_as_current_span("profiler.fetch_column_stats"):
            try:
                return await self._do_fetch_column_stats(schema, table, columns)
            except Exception:
                logger.warning(
                    "Column statistics fetch failed; skipping stats",
                    schema=schema,
                    table=table,
                    exc_info=True,
                )
                return {}

    async def _do_fetch_column_stats(
        self,
        schema: str,
        table: str,
        columns: list[dict[str, Any]],
    ) -> dict[str, ColumnStatistics]:
        grouped = classify_columns(columns)
        s_schema = self._esc(schema)
        s_table = self._esc(table)
        fqt = f'"{s_schema}"."{s_table}"'

        # ── Phase 1: common stats (single query, single table scan) ────────
        count_parts: list[str] = []
        for col in columns:
            c = self._esc(col["name"])
            count_parts.append(f'COUNT("{c}") AS "{c}__non_null"')
            # json type has no equality operator; cast to text for DISTINCT.
            col_expr = f'"{c}"::text' if col["data_type"].lower().strip() == "json" else f'"{c}"'
            count_parts.append(f'COUNT(DISTINCT {col_expr}) AS "{c}__distinct"')
        common_query = f"SELECT COUNT(*) AS total_count, {', '.join(count_parts)} FROM {fqt}"
        common_row = await self._db.fetch_one(common_query)
        total_count: int = common_row["total_count"] if common_row else 0

        # Build common stats dict per column.
        common_by_col: dict[str, dict[str, Any]] = {}
        for col in columns:
            cn = col["name"]
            non_null = common_row[f"{cn}__non_null"] if common_row else 0
            distinct = common_row[f"{cn}__distinct"] if common_row else 0
            null_count = total_count - non_null
            common_by_col[cn] = {
                "total_count": total_count,
                "null_count": null_count,
                "null_percentage": round(null_count / total_count * 100, 2) if total_count else 0.0,
                "distinct_count": distinct,
                "distinct_percentage": round(distinct / total_count * 100, 2) if total_count else 0.0,
            }

        # ── Phase 2: type-specific stats (concurrent) ─────────────────────
        numeric_cols = grouped[ColumnTypeCategory.NUMERIC]
        string_cols = grouped[ColumnTypeCategory.STRING]
        boolean_cols = grouped[ColumnTypeCategory.BOOLEAN]
        temporal_cols = grouped[ColumnTypeCategory.TEMPORAL]

        type_tasks: dict[str, Any] = {}
        if numeric_cols:
            type_tasks["numeric"] = self._fetch_numeric_stats(fqt, numeric_cols)
        if string_cols:
            type_tasks["string"] = self._fetch_string_stats(fqt, string_cols)
        if boolean_cols:
            type_tasks["boolean"] = self._fetch_boolean_stats(fqt, boolean_cols)
        if temporal_cols:
            type_tasks["temporal"] = self._fetch_temporal_stats(fqt, temporal_cols)

        type_keys = list(type_tasks.keys())
        type_results = await asyncio.gather(*type_tasks.values())
        type_resolved = dict(zip(type_keys, type_results, strict=True))

        # ── Phase 3: top values (concurrent per categorical column) ────────
        top_values_by_col: dict[str, list[TopValueEntry]] = {}
        tv_col_names: list[str] = []
        tv_tasks: list[Any] = []
        # Boolean columns always get top values.
        for col in boolean_cols:
            tv_col_names.append(col["name"])
            tv_tasks.append(self._fetch_top_values(fqt, col["name"], total_count))
        # Enum columns always get top values.
        for col in columns:
            if col.get("enum_values") is not None and col["name"] not in tv_col_names:
                tv_col_names.append(col["name"])
                tv_tasks.append(self._fetch_top_values(fqt, col["name"], total_count))
        # Low-cardinality string columns.
        for col in string_cols:
            cn = col["name"]
            if common_by_col[cn]["distinct_count"] <= self._config.top_values_cardinality_threshold:
                tv_col_names.append(cn)
                tv_tasks.append(self._fetch_top_values(fqt, cn, total_count))
        # Low-cardinality numeric columns.
        for col in numeric_cols:
            cn = col["name"]
            if common_by_col[cn]["distinct_count"] <= self._config.top_values_cardinality_threshold:
                tv_col_names.append(cn)
                tv_tasks.append(self._fetch_top_values(fqt, cn, total_count))
        if tv_tasks:
            tv_results = await asyncio.gather(*tv_tasks)
            for cn, tv in zip(tv_col_names, tv_results, strict=True):
                top_values_by_col[cn] = tv

        # ── Phase 4: outlier counts for numeric columns ───────────────────
        numeric_stats_map: dict[str, NumericColumnStats] = type_resolved.get("numeric", {})
        outlier_col_names: list[str] = []
        outlier_tasks: list[Any] = []
        for cn, ns in numeric_stats_map.items():
            if ns.p25 is not None and ns.p75 is not None:
                outlier_col_names.append(cn)
                outlier_tasks.append(self._fetch_outlier_count(fqt, cn, ns.p25, ns.p75))
        if outlier_tasks:
            oc_results = await asyncio.gather(*outlier_tasks)
            for cn, oc in zip(outlier_col_names, oc_results, strict=True):
                numeric_stats_map[cn] = numeric_stats_map[cn].model_copy(update={"outlier_count": oc})

        # ── Assemble ColumnStatistics per column ──────────────────────────
        result: dict[str, ColumnStatistics] = {}
        for col in columns:
            cn = col["name"]
            result[cn] = ColumnStatistics(
                **common_by_col[cn],
                numeric=numeric_stats_map.get(cn),
                string=type_resolved.get("string", {}).get(cn),
                boolean=type_resolved.get("boolean", {}).get(cn),
                temporal=type_resolved.get("temporal", {}).get(cn),
                top_values=top_values_by_col.get(cn),
            )
        return result

    async def _fetch_numeric_stats(self, fqt: str, columns: list[dict[str, Any]]) -> dict[str, NumericColumnStats]:
        parts: list[str] = []
        for col in columns:
            c = self._esc(col["name"])
            parts.extend(
                [
                    f'MIN("{c}")::float8 AS "{c}__min"',
                    f'MAX("{c}")::float8 AS "{c}__max"',
                    f'AVG("{c}")::float8 AS "{c}__mean"',
                    f'STDDEV("{c}")::float8 AS "{c}__stddev"',
                    f'VARIANCE("{c}")::float8 AS "{c}__variance"',
                    f'SUM("{c}")::float8 AS "{c}__sum"',
                    f'PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "{c}") AS "{c}__p50"',
                    f'PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY "{c}") AS "{c}__p5"',
                    f'PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "{c}") AS "{c}__p25"',
                    f'PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "{c}") AS "{c}__p75"',
                    f'PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY "{c}") AS "{c}__p95"',
                    f'COUNT(*) FILTER (WHERE "{c}" = 0) AS "{c}__zero_count"',
                    f'COUNT(*) FILTER (WHERE "{c}" < 0) AS "{c}__negative_count"',
                ]
            )
        query = f"SELECT {', '.join(parts)} FROM {fqt}"
        row = await self._db.fetch_one(query)
        result: dict[str, NumericColumnStats] = {}
        if row:
            for col in columns:
                cn = col["name"]
                result[cn] = NumericColumnStats(
                    min=_make_json_safe(row.get(f"{cn}__min")),
                    max=_make_json_safe(row.get(f"{cn}__max")),
                    mean=_make_json_safe(row.get(f"{cn}__mean")),
                    median=_make_json_safe(row.get(f"{cn}__p50")),
                    stddev=_make_json_safe(row.get(f"{cn}__stddev")),
                    variance=_make_json_safe(row.get(f"{cn}__variance")),
                    sum=_make_json_safe(row.get(f"{cn}__sum")),
                    p5=_make_json_safe(row.get(f"{cn}__p5")),
                    p25=_make_json_safe(row.get(f"{cn}__p25")),
                    p75=_make_json_safe(row.get(f"{cn}__p75")),
                    p95=_make_json_safe(row.get(f"{cn}__p95")),
                    zero_count=row.get(f"{cn}__zero_count"),
                    negative_count=row.get(f"{cn}__negative_count"),
                )
        return result

    async def _fetch_string_stats(self, fqt: str, columns: list[dict[str, Any]]) -> dict[str, StringColumnStats]:
        parts: list[str] = []
        for col in columns:
            c = self._esc(col["name"])
            parts.extend(
                [
                    f'MIN(LENGTH("{c}")) AS "{c}__min_length"',
                    f'MAX(LENGTH("{c}")) AS "{c}__max_length"',
                    f'AVG(LENGTH("{c}"))::float8 AS "{c}__avg_length"',
                    f'COUNT(*) FILTER (WHERE "{c}" = \'\') AS "{c}__empty_count"',
                ]
            )
        query = f"SELECT {', '.join(parts)} FROM {fqt}"
        row = await self._db.fetch_one(query)
        result: dict[str, StringColumnStats] = {}
        if row:
            for col in columns:
                cn = col["name"]
                result[cn] = StringColumnStats(
                    min_length=row.get(f"{cn}__min_length"),
                    max_length=row.get(f"{cn}__max_length"),
                    avg_length=_make_json_safe(row.get(f"{cn}__avg_length")),
                    empty_count=row.get(f"{cn}__empty_count"),
                )
        return result

    async def _fetch_boolean_stats(self, fqt: str, columns: list[dict[str, Any]]) -> dict[str, BooleanColumnStats]:
        parts: list[str] = []
        for col in columns:
            c = self._esc(col["name"])
            parts.extend(
                [
                    f'COUNT(*) FILTER (WHERE "{c}" = TRUE) AS "{c}__true_count"',
                    f'COUNT(*) FILTER (WHERE "{c}" = FALSE) AS "{c}__false_count"',
                ]
            )
        query = f"SELECT {', '.join(parts)} FROM {fqt}"
        row = await self._db.fetch_one(query)
        result: dict[str, BooleanColumnStats] = {}
        if row:
            for col in columns:
                cn = col["name"]
                tc = row.get(f"{cn}__true_count", 0) or 0
                fc = row.get(f"{cn}__false_count", 0) or 0
                total = tc + fc
                result[cn] = BooleanColumnStats(
                    true_count=tc,
                    false_count=fc,
                    true_percentage=round(tc / total * 100, 2) if total else 0.0,
                )
        return result

    async def _fetch_temporal_stats(self, fqt: str, columns: list[dict[str, Any]]) -> dict[str, TemporalColumnStats]:
        parts: list[str] = []
        for col in columns:
            c = self._esc(col["name"])
            parts.extend(
                [
                    f'MIN("{c}")::text AS "{c}__min"',
                    f'MAX("{c}")::text AS "{c}__max"',
                ]
            )
        query = f"SELECT {', '.join(parts)} FROM {fqt}"
        row = await self._db.fetch_one(query)
        result: dict[str, TemporalColumnStats] = {}
        if row:
            for col in columns:
                cn = col["name"]
                result[cn] = TemporalColumnStats(
                    min=row.get(f"{cn}__min"),
                    max=row.get(f"{cn}__max"),
                )
        return result

    async def _fetch_top_values(self, fqt: str, col_name: str, total_count: int) -> list[TopValueEntry]:
        c = self._esc(col_name)
        query = (
            f'SELECT "{c}"::text AS value, COUNT(*) AS count '
            f"FROM {fqt} "
            f'WHERE "{c}" IS NOT NULL '
            f'GROUP BY "{c}" '
            f"ORDER BY count DESC "
            f"LIMIT $1"
        )
        rows = await self._db.fetch_all(query, (self._config.top_values_limit,))
        return [
            TopValueEntry(
                value=r["value"],
                count=r["count"],
                percentage=round(r["count"] / total_count * 100, 2) if total_count else 0.0,
            )
            for r in rows
        ]

    async def _fetch_outlier_count(self, fqt: str, col_name: str, q1: float, q3: float) -> int:
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        c = self._esc(col_name)
        query = f'SELECT COUNT(*) AS outlier_count FROM {fqt} WHERE "{c}" IS NOT NULL AND ("{c}" < $1 OR "{c}" > $2)'
        row = await self._db.fetch_one(query, (lower, upper))
        return row["outlier_count"] if row else 0

    # ── Views ────────────────────────────────────────────────────────────────────

    async def _fetch_views(self, schema: str) -> list[dict[str, Any]]:
        """Fetch view metadata for a schema."""
        rows = await self._db.fetch_all(q.VIEWS_IN_SCHEMA, (schema,))
        return [
            {
                "name": r["table_name"],
                "schema": r["table_schema"],
                "owner": r["owner"],
                "definition": r["definition"],
            }
            for r in rows
        ]


# ── BaseProfiler implementation ────────────────────────────────────────────────


class PostgresProfilerService(BaseProfilerService):
    """Profiler for PostgreSQL databases.

    Delegates query execution to :class:`PostgresProfilerService` and
    assembles the typed response from its raw dict output.
    """

    service_name = "PostgreSQL"
    span_name = "profiler.postgres"
    _datasource_type = "postgres"

    def _span_attributes(self, conn: PostgresConfig) -> dict[str, str | int]:
        attrs: dict[str, str | int] = {
            "db.system": "postgresql",
            "db.name": conn.database,
            "net.peer.name": conn.host,
            "net.peer.port": conn.port,
        }
        if conn.ssh_tunnel is not None:
            attrs["net.ssh_tunnel.host"] = conn.ssh_tunnel.host
            attrs["net.ssh_tunnel.port"] = conn.ssh_tunnel.port
        return attrs

    def _log_context(self, conn: PostgresConfig) -> dict[str, Any]:
        ctx: dict[str, Any] = {
            "host": conn.host,
            "port": conn.port,
            "database": conn.database,
            "username": conn.username,
        }
        if conn.ssh_tunnel is not None:
            ctx["ssh_tunnel_host"] = conn.ssh_tunnel.host
            ctx["ssh_tunnel_port"] = conn.ssh_tunnel.port
            ctx["ssh_tunnel_username"] = conn.ssh_tunnel.username
        return ctx

    async def _run(
        self,
        connector: BaseConnector,
        config: ProfilingConfig,
        progress: ProgressReporter | None = None,
    ) -> dict[str, Any]:
        return await PostgresProfiler(connector, config).profile()

    def _assemble_response(
        self,
        raw: dict[str, Any],
        profiled_at: datetime,
    ) -> ProfilingResponse:
        db = raw["database"]
        database_meta = DatabaseMetadata(
            name=db["name"],
            version=db["version"],
            encoding=db["encoding"],
            size_bytes=db["size_bytes"],
        )

        schema_metas: list[SchemaMetadata] = []
        for schema in raw["schemas"]:
            tables: list[TableMetadata] = []
            for tbl in schema["tables"]:
                columns = [ColumnMetadata(**col) for col in tbl["columns"]]

                indexes: list[IndexMetadata] | None = None
                if tbl.get("indexes") is not None:
                    indexes = [IndexMetadata(**idx) for idx in tbl["indexes"]]

                relationships: list[RelationshipMetadata] | None = None
                if tbl.get("relationships") is not None:
                    relationships = [RelationshipMetadata(**rel) for rel in tbl["relationships"]]

                freshness: DataFreshnessInfo | None = None
                if tbl.get("data_freshness") is not None:
                    freshness = DataFreshnessInfo(**tbl["data_freshness"])

                tables.append(
                    TableMetadata(
                        name=tbl["name"],
                        schema=tbl["schema"],
                        owner=tbl["owner"],
                        description=tbl.get("description"),
                        row_count=tbl.get("row_count"),
                        size_bytes=tbl.get("size_bytes"),
                        total_size_bytes=tbl.get("total_size_bytes"),
                        data_freshness=freshness,
                        columns=columns,
                        indexes=indexes,
                        relationships=relationships,
                    )
                )

            views = [ViewMetadata(**v) for v in schema["views"]]
            schema_metas.append(
                SchemaMetadata(
                    name=schema["name"],
                    owner=schema["owner"],
                    tables=tables,
                    views=views,
                )
            )

        return ProfilingResponse(
            profiled_at=profiled_at,
            database=database_meta,
            schemas=schema_metas,
        )
