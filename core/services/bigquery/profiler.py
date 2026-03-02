"""BigQuery profiler — query service and BaseProfiler implementation."""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.services.task_manager import ProgressReporter
from uuid import UUID

from ignite_data_connectors import BaseConnector, BigQueryConfig

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

# BigQuery system schemas that are always excluded regardless of config.
_BIGQUERY_SYSTEM_SCHEMAS: frozenset[str] = frozenset({"INFORMATION_SCHEMA"})


# ── JSON serialisation helper ──────────────────────────────────────────────────


def _make_json_safe(value: Any) -> Any:
    """Convert a value to a JSON-serialisable Python primitive."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, memoryview):
        return bytes(value).hex()
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


# ── Query service ──────────────────────────────────────────────────────────────


class BigQueryProfiler:
    """Orchestrates metadata extraction from a connected BigQuery project.

    The caller is responsible for opening and closing the connector; this
    service never calls ``connect()`` / ``disconnect()`` directly.

    BigQuery hierarchy mapping:
    - **project** → ``DatabaseMetadata.name``
    - **dataset** → ``SchemaMetadata.name``
    - **table**   → ``TableMetadata.name``

    Usage::

        async with create_connector(bq_config) as db:
            result = await BigQueryProfiler(db, profiling_config, project="my-project").profile()
    """

    def __init__(self, connector: BaseConnector, config: ProfilingConfig, *, project: str) -> None:
        self._db = connector
        self._config = config
        self._project = project

    # ── Public orchestration ───────────────────────────────────────────────────

    async def profile(self) -> dict[str, Any]:
        """Run the full profiling pass and return a raw dict."""
        with tracer.start_as_current_span("profiler.profile") as span:
            db_meta, schema_names = await asyncio.gather(
                self._fetch_database_metadata(),
                self._fetch_schema_names(),
            )
            span.set_attribute("profiler.schemas_discovered", len(schema_names))
            logger.info(
                "Profiling datasets",
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
        """Return project-level metadata.

        BigQuery does not have a ``CURRENT_DATABASE()`` function.  The project
        name is provided via the constructor.
        """
        return {
            "name": self._project,
            "version": "",
            "encoding": "UTF-8",
            "size_bytes": 0,
        }

    # ── Schema (dataset) discovery ─────────────────────────────────────────────

    async def _fetch_schema_names(self) -> list[str]:
        """Return dataset names after applying include/exclude filters."""
        with tracer.start_as_current_span("profiler.fetch_schema_names"):
            exclude = list(_BIGQUERY_SYSTEM_SCHEMAS | set(self._config.exclude_schemas))
            placeholders = ", ".join("%s" for _ in exclude)
            query = q.SCHEMAS.format(placeholders=placeholders)
            rows = await self._db.fetch_all(query, tuple(exclude))
            all_schemas = [r["schema_name"] for r in rows]

            if self._config.include_schemas is not None:
                include_set = set(self._config.include_schemas)
                all_schemas = [s for s in all_schemas if s in include_set]

            logger.debug("Discovered datasets", schemas=all_schemas)
            return all_schemas

    # ── Schema-level profiling ──────────────────────────────────────────────────

    async def _profile_schema(self, dataset_name: str) -> dict[str, Any]:
        """Profile a single dataset: fetch tables and views concurrently."""
        with tracer.start_as_current_span("profiler.profile_schema") as span:
            span.set_attribute("profiler.schema", dataset_name)

            table_rows, view_rows = await asyncio.gather(
                self._fetch_tables(dataset_name),
                self._fetch_views(dataset_name),
            )

            filtered_tables = self._filter_tables(table_rows)

            table_results = await asyncio.gather(
                *[self._profile_table(row, dataset_name) for row in filtered_tables],
            )

            return {
                "name": dataset_name,
                "owner": "",
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

    async def _fetch_tables(self, dataset: str) -> list[dict[str, Any]]:
        """Fetch basic table metadata for all tables in a dataset."""
        safe_ds = self._esc(dataset)
        query = q.TABLES_IN_DATASET.format(dataset=safe_ds)
        rows = await self._db.fetch_all(query)
        return [dict(r) for r in rows]

    async def _profile_table(self, table_row: dict[str, Any], dataset: str) -> dict[str, Any]:
        """Profile a single table: columns, row count, sample, freshness."""
        table_name = table_row["table_name"]
        with tracer.start_as_current_span("profiler.profile_table") as span:
            span.set_attribute("profiler.table", f"{dataset}.{table_name}")

            tasks: dict[str, Any] = {
                "columns": self._fetch_columns(dataset, table_name),
            }
            if self._config.include_row_counts:
                tasks["row_count"] = self._fetch_row_count(dataset, table_name)
            if self._config.include_indexes:
                tasks["indexes"] = self._fetch_indexes()
            if self._config.include_relationships:
                tasks["relationships"] = self._fetch_foreign_keys()
            if self._config.include_data_freshness:
                tasks["data_freshness"] = self._fetch_data_freshness(dataset, table_name)
            if self._config.include_sample_data:
                tasks["sample_data"] = self._fetch_sample_data(dataset, table_name)

            keys = list(tasks.keys())
            results = await asyncio.gather(*tasks.values())
            resolved = dict(zip(keys, results, strict=True))

            columns: list[dict[str, Any]] = resolved.get("columns", [])
            sample_data: list[dict[str, Any]] | None = resolved.get("sample_data")
            if sample_data is not None:
                for col in columns:
                    col["sample_values"] = [row.get(col["name"]) for row in sample_data]

            col_stats: dict[str, ColumnStatistics] = {}
            if self._config.include_column_stats and columns:
                col_stats = await self._fetch_column_stats(dataset, table_name, columns)
            for col in columns:
                col["statistics"] = col_stats.get(col["name"])

            return {
                "name": table_name,
                "schema": dataset,
                "owner": table_row.get("owner", "") or "",
                "description": table_row.get("description") or None,
                "size_bytes": table_row.get("size_bytes"),
                "total_size_bytes": table_row.get("total_size_bytes"),
                "row_count": resolved.get("row_count"),
                "columns": columns,
                "indexes": resolved.get("indexes"),
                "relationships": resolved.get("relationships"),
                "data_freshness": resolved.get("data_freshness"),
            }

    # ── Column-level ─────────────────────────────────────────────────────────────

    async def _fetch_columns(self, dataset: str, table: str) -> list[dict[str, Any]]:
        """Fetch column metadata including descriptions from COLUMN_FIELD_PATHS.

        BigQuery does not have native enum types; ``enum_values`` is
        always ``None``.  There are no primary keys; ``is_primary_key``
        is always ``False``.
        """
        with tracer.start_as_current_span("profiler.fetch_columns"):
            safe_ds = self._esc(dataset)
            col_query = q.COLUMNS_FOR_TABLE.format(dataset=safe_ds)
            desc_query = q.COLUMN_DESCRIPTIONS.format(dataset=safe_ds)
            col_rows, desc_rows = await asyncio.gather(
                self._db.fetch_all(col_query, (dataset, table)),
                self._db.fetch_all(desc_query, (dataset, table)),
            )

            desc_map: dict[str, str | None] = {r["column_name"]: r["description"] for r in desc_rows}

            return [
                {
                    "name": r["column_name"],
                    "ordinal_position": r["ordinal_position"],
                    "data_type": r["data_type"],
                    "is_nullable": bool(r["is_nullable"]),
                    "column_default": r["column_default"],
                    "character_maximum_length": r["character_maximum_length"],
                    "numeric_precision": r["numeric_precision"],
                    "numeric_scale": r["numeric_scale"],
                    "is_primary_key": False,
                    "description": desc_map.get(r["column_name"]) or None,
                    "enum_values": None,
                    "sample_values": None,
                    "statistics": None,
                }
                for r in col_rows
            ]

    # ── Row count ────────────────────────────────────────────────────────────────

    async def _fetch_row_count(self, dataset: str, table: str) -> int | None:
        """Return row count from INFORMATION_SCHEMA, falling back to exact COUNT(1)."""
        safe_ds = self._esc(dataset)
        query = q.ROW_COUNT_FAST.format(dataset=safe_ds)
        row = await self._db.fetch_one(query, (dataset, table))
        if row is not None:
            estimated = row["row_count"]
            if estimated is not None and int(estimated) >= 0:
                return int(estimated)
        # NULL or missing — fall back to exact count.
        safe_table = self._esc(table)
        exact_query = q.ROW_COUNT_EXACT.format(dataset=safe_ds, table=safe_table)
        exact_row = await self._db.fetch_one(exact_query)
        return int(exact_row["row_count"]) if exact_row else None

    # ── Indexes ──────────────────────────────────────────────────────────────────

    async def _fetch_indexes(self) -> list[dict[str, Any]]:
        """BigQuery does not have traditional indexes.

        Always returns an empty list.
        """
        return []

    # ── Foreign keys ─────────────────────────────────────────────────────────────

    async def _fetch_foreign_keys(self) -> list[dict[str, Any]]:
        """BigQuery does not support foreign key constraints.

        Always returns an empty list.
        """
        return []

    # ── Data freshness ───────────────────────────────────────────────────────────

    async def _fetch_data_freshness(self, dataset: str, table: str) -> dict[str, Any] | None:
        """Fetch freshness timestamps from INFORMATION_SCHEMA.TABLES.

        BigQuery provides ``creation_time`` but not analyze/vacuum equivalents.
        All maintenance-related fields are set to ``None``.
        """
        safe_ds = self._esc(dataset)
        query = q.DATA_FRESHNESS_FOR_TABLE.format(dataset=safe_ds)
        row = await self._db.fetch_one(query, (dataset, table))
        if row is None:
            return None
        return {
            "last_analyze": row["created"],
            "last_autoanalyze": None,
            "last_vacuum": None,
            "last_autovacuum": None,
        }

    # ── Sample data ──────────────────────────────────────────────────────────────

    async def _fetch_sample_data(self, dataset: str, table: str) -> list[dict[str, Any]]:
        """Fetch random sample rows and convert values to JSON-safe types."""
        safe_ds = self._esc(dataset)
        safe_table = self._esc(table)
        query = f"SELECT * FROM `{safe_ds}`.`{safe_table}` ORDER BY RAND() LIMIT %s"
        rows = await self._db.fetch_all(query, (self._config.sample_size,))
        return [{k: _make_json_safe(v) for k, v in row.items()} for row in rows]

    # ── Column statistics ──────────────────────────────────────────────────────

    @staticmethod
    def _esc(name: str) -> str:
        """Escape a BigQuery identifier (backtick safe)."""
        return name.replace("`", "``")

    async def _fetch_column_stats(
        self,
        dataset: str,
        table: str,
        columns: list[dict[str, Any]],
    ) -> dict[str, ColumnStatistics]:
        """Compute column-level statistics for all columns in a table."""
        with tracer.start_as_current_span("profiler.fetch_column_stats"):
            try:
                return await self._do_fetch_column_stats(dataset, table, columns)
            except Exception:
                logger.warning(
                    "Column statistics fetch failed; skipping stats",
                    dataset=dataset,
                    table=table,
                    exc_info=True,
                )
                return {}

    async def _do_fetch_column_stats(
        self,
        dataset: str,
        table: str,
        columns: list[dict[str, Any]],
    ) -> dict[str, ColumnStatistics]:
        grouped = classify_columns(columns)
        s_dataset = self._esc(dataset)
        s_table = self._esc(table)
        fqt = f"`{s_dataset}`.`{s_table}`"

        # ── Phase 1: common stats (single query, single table scan) ────────
        count_parts: list[str] = []
        for col in columns:
            c = self._esc(col["name"])
            count_parts.append(f"COUNT(`{c}`) AS `{c}__non_null`")
            count_parts.append(f"COUNT(DISTINCT `{c}`) AS `{c}__distinct`")
        common_query = f"SELECT COUNT(*) AS total_count, {', '.join(count_parts)} FROM {fqt}"
        common_row = await self._db.fetch_one(common_query)
        total_count: int = int(common_row["total_count"]) if common_row else 0

        common_by_col: dict[str, dict[str, Any]] = {}
        for col in columns:
            cn = col["name"]
            non_null = int(common_row[f"{cn}__non_null"]) if common_row else 0
            distinct = int(common_row[f"{cn}__distinct"]) if common_row else 0
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
        for col in boolean_cols:
            tv_col_names.append(col["name"])
            tv_tasks.append(self._fetch_top_values(fqt, col["name"], total_count))
        for col in string_cols:
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
        # Phase 2a: basic aggregates (single query)
        agg_parts: list[str] = []
        for col in columns:
            c = self._esc(col["name"])
            agg_parts.extend(
                [
                    f"CAST(MIN(`{c}`) AS FLOAT64) AS `{c}__min`",
                    f"CAST(MAX(`{c}`) AS FLOAT64) AS `{c}__max`",
                    f"CAST(AVG(`{c}`) AS FLOAT64) AS `{c}__mean`",
                    f"CAST(STDDEV(`{c}`) AS FLOAT64) AS `{c}__stddev`",
                    f"CAST(VARIANCE(`{c}`) AS FLOAT64) AS `{c}__variance`",
                    f"CAST(SUM(`{c}`) AS FLOAT64) AS `{c}__sum`",
                    f"COUNTIF(`{c}` = 0) AS `{c}__zero_count`",
                    f"COUNTIF(`{c}` < 0) AS `{c}__negative_count`",
                ]
            )
        agg_query = f"SELECT {', '.join(agg_parts)} FROM {fqt}"

        # Phase 2b: percentiles via APPROX_QUANTILES (single query)
        pct_parts: list[str] = []
        for col in columns:
            c = self._esc(col["name"])
            pct_parts.extend(
                [
                    f"APPROX_QUANTILES(`{c}`, 20)[OFFSET(1)] AS `{c}__p5`",
                    f"APPROX_QUANTILES(`{c}`, 20)[OFFSET(5)] AS `{c}__p25`",
                    f"APPROX_QUANTILES(`{c}`, 20)[OFFSET(10)] AS `{c}__p50`",
                    f"APPROX_QUANTILES(`{c}`, 20)[OFFSET(15)] AS `{c}__p75`",
                    f"APPROX_QUANTILES(`{c}`, 20)[OFFSET(19)] AS `{c}__p95`",
                ]
            )
        pct_query = f"SELECT {', '.join(pct_parts)} FROM {fqt}"

        agg_row, pct_row = await asyncio.gather(
            self._db.fetch_one(agg_query),
            self._db.fetch_one(pct_query),
        )

        result: dict[str, NumericColumnStats] = {}
        for col in columns:
            cn = col["name"]
            result[cn] = NumericColumnStats(
                min=_make_json_safe(agg_row.get(f"{cn}__min")) if agg_row else None,
                max=_make_json_safe(agg_row.get(f"{cn}__max")) if agg_row else None,
                mean=_make_json_safe(agg_row.get(f"{cn}__mean")) if agg_row else None,
                median=_make_json_safe(pct_row.get(f"{cn}__p50")) if pct_row else None,
                stddev=_make_json_safe(agg_row.get(f"{cn}__stddev")) if agg_row else None,
                variance=_make_json_safe(agg_row.get(f"{cn}__variance")) if agg_row else None,
                sum=_make_json_safe(agg_row.get(f"{cn}__sum")) if agg_row else None,
                p5=_make_json_safe(pct_row.get(f"{cn}__p5")) if pct_row else None,
                p25=_make_json_safe(pct_row.get(f"{cn}__p25")) if pct_row else None,
                p75=_make_json_safe(pct_row.get(f"{cn}__p75")) if pct_row else None,
                p95=_make_json_safe(pct_row.get(f"{cn}__p95")) if pct_row else None,
                zero_count=int(agg_row[f"{cn}__zero_count"])
                if agg_row and agg_row.get(f"{cn}__zero_count") is not None
                else None,
                negative_count=(
                    int(agg_row[f"{cn}__negative_count"])
                    if agg_row and agg_row.get(f"{cn}__negative_count") is not None
                    else None
                ),
            )
        return result

    async def _fetch_string_stats(self, fqt: str, columns: list[dict[str, Any]]) -> dict[str, StringColumnStats]:
        parts: list[str] = []
        for col in columns:
            c = self._esc(col["name"])
            parts.extend(
                [
                    f"MIN(LENGTH(`{c}`)) AS `{c}__min_length`",
                    f"MAX(LENGTH(`{c}`)) AS `{c}__max_length`",
                    f"CAST(AVG(LENGTH(`{c}`)) AS FLOAT64) AS `{c}__avg_length`",
                    f"COUNTIF(`{c}` = '') AS `{c}__empty_count`",
                ]
            )
        query = f"SELECT {', '.join(parts)} FROM {fqt}"
        row = await self._db.fetch_one(query)
        result: dict[str, StringColumnStats] = {}
        if row:
            for col in columns:
                cn = col["name"]
                result[cn] = StringColumnStats(
                    min_length=int(row[f"{cn}__min_length"]) if row.get(f"{cn}__min_length") is not None else None,
                    max_length=int(row[f"{cn}__max_length"]) if row.get(f"{cn}__max_length") is not None else None,
                    avg_length=_make_json_safe(row.get(f"{cn}__avg_length")),
                    empty_count=(int(row[f"{cn}__empty_count"]) if row.get(f"{cn}__empty_count") is not None else None),
                )
        return result

    async def _fetch_boolean_stats(self, fqt: str, columns: list[dict[str, Any]]) -> dict[str, BooleanColumnStats]:
        parts: list[str] = []
        for col in columns:
            c = self._esc(col["name"])
            parts.extend(
                [
                    f"COUNTIF(`{c}` = TRUE) AS `{c}__true_count`",
                    f"COUNTIF(`{c}` = FALSE) AS `{c}__false_count`",
                ]
            )
        query = f"SELECT {', '.join(parts)} FROM {fqt}"
        row = await self._db.fetch_one(query)
        result: dict[str, BooleanColumnStats] = {}
        if row:
            for col in columns:
                cn = col["name"]
                tc = int(row.get(f"{cn}__true_count", 0) or 0)
                fc = int(row.get(f"{cn}__false_count", 0) or 0)
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
                    f"CAST(MIN(`{c}`) AS STRING) AS `{c}__min`",
                    f"CAST(MAX(`{c}`) AS STRING) AS `{c}__max`",
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
            f"SELECT CAST(`{c}` AS STRING) AS value, COUNT(*) AS count "
            f"FROM {fqt} "
            f"WHERE `{c}` IS NOT NULL "
            f"GROUP BY `{c}` "
            f"ORDER BY count DESC "
            f"LIMIT %s"
        )
        rows = await self._db.fetch_all(query, (self._config.top_values_limit,))
        return [
            TopValueEntry(
                value=r["value"],
                count=int(r["count"]),
                percentage=round(int(r["count"]) / total_count * 100, 2) if total_count else 0.0,
            )
            for r in rows
        ]

    async def _fetch_outlier_count(self, fqt: str, col_name: str, q1: float, q3: float) -> int:
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        c = self._esc(col_name)
        query = f"SELECT COUNT(*) AS outlier_count FROM {fqt} WHERE `{c}` IS NOT NULL AND (`{c}` < %s OR `{c}` > %s)"
        row = await self._db.fetch_one(query, (lower, upper))
        return int(row["outlier_count"]) if row else 0

    # ── Views ────────────────────────────────────────────────────────────────────

    async def _fetch_views(self, dataset: str) -> list[dict[str, Any]]:
        """Fetch view metadata for a dataset."""
        safe_ds = self._esc(dataset)
        query = q.VIEWS_IN_DATASET.format(dataset=safe_ds)
        rows = await self._db.fetch_all(query, (dataset,))
        return [
            {
                "name": r["table_name"],
                "schema": r["table_schema"],
                "owner": r.get("owner", "") or "",
                "definition": r["definition"],
            }
            for r in rows
        ]


# ── BaseProfiler implementation ────────────────────────────────────────────────


class BigQueryProfilerService(BaseProfilerService):
    """Profiler for BigQuery projects.

    Delegates query execution to :class:`BigQueryProfiler` and
    assembles the typed response from its raw dict output.
    """

    service_name = "BigQuery"
    span_name = "profiler.bigquery"

    def _span_attributes(self, conn: BigQueryConfig) -> dict[str, str | int]:
        attrs: dict[str, str | int] = {
            "db.system": "bigquery",
            "db.name": conn.project,
        }
        if conn.location:
            attrs["bigquery.location"] = conn.location
        if conn.dataset:
            attrs["bigquery.dataset"] = conn.dataset
        return attrs

    def _log_context(self, conn: BigQueryConfig) -> dict[str, Any]:
        return {
            "host": conn.project,
            "database": conn.project,
            "location": conn.location,
            "dataset": conn.dataset,
        }

    async def _run(
        self,
        connector: BaseConnector,
        config: ProfilingConfig,
        progress: ProgressReporter | None = None,
    ) -> dict[str, Any]:
        project = connector._config.project  # type: ignore[attr-defined]
        return await BigQueryProfiler(connector, config, project=project).profile()

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
