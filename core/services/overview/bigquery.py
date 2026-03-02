"""BigQuery overview service — metadata counts via dataset-scoped INFORMATION_SCHEMA."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from ignite_data_connectors import BaseConnector, BigQueryConfig

from core.api.v1.schemas.overview import DatabaseOverview, OverviewConfig, SchemaOverview
from core.services.overview.base import BaseOverviewService

# ── SQL queries (%s placeholders for project-level, .format for dataset) ───────

SCHEMAS = """
SELECT schema_name AS schema_name
FROM INFORMATION_SCHEMA.SCHEMATA
WHERE schema_name NOT IN ({placeholders})
ORDER BY schema_name
"""


def _bq_placeholders(n: int) -> str:
    return ", ".join(["%s"] * n)


def _safe_dataset(ds: str) -> str:
    """Escape backticks in dataset names for safe interpolation."""
    return ds.replace("`", "``")


class BigQueryOverviewService(BaseOverviewService):
    service_name = "BigQuery"
    span_name = "overview.bigquery"

    def _span_attributes(self, conn: BigQueryConfig) -> dict[str, str | int]:
        attrs: dict[str, str | int] = {
            "db.system": "bigquery",
            "db.name": conn.project,
        }
        if conn.location:
            attrs["bigquery.location"] = conn.location
        return attrs

    def _log_context(self, conn: BigQueryConfig) -> dict[str, Any]:
        return {
            "host": conn.project,
            "database": conn.project,
            "location": conn.location,
            "dataset": conn.dataset,
        }

    async def _count_dataset(self, db: BaseConnector, dataset: str) -> SchemaOverview:
        """Fetch counts for a single dataset using dataset-scoped INFORMATION_SCHEMA."""
        safe_ds = _safe_dataset(dataset)

        table_q = f"SELECT COUNT(*) AS cnt FROM `{safe_ds}`.INFORMATION_SCHEMA.TABLES WHERE table_type = 'BASE TABLE'"
        view_q = f"SELECT COUNT(*) AS cnt FROM `{safe_ds}`.INFORMATION_SCHEMA.TABLES WHERE table_type = 'VIEW'"
        col_q = f"SELECT COUNT(*) AS cnt FROM `{safe_ds}`.INFORMATION_SCHEMA.COLUMNS"

        t_row, v_row, c_row = await asyncio.gather(
            db.fetch_one(table_q),
            db.fetch_one(view_q),
            db.fetch_one(col_q),
        )

        return SchemaOverview(
            schema_name=dataset,
            table_count=t_row["cnt"] if t_row else 0,
            view_count=v_row["cnt"] if v_row else 0,
            index_count=0,  # BigQuery has no indexes
            relationship_count=0,  # BigQuery has no foreign keys
            column_count=c_row["cnt"] if c_row else 0,
        )

    async def _fetch_overview(
        self,
        db: BaseConnector,
        cfg: OverviewConfig,
        conn: BigQueryConfig,
    ) -> DatabaseOverview:
        exclude = list(cfg.exclude_schemas)
        ph = _bq_placeholders(len(exclude))
        params = tuple(exclude)

        schema_rows = await db.fetch_all(SCHEMAS.format(placeholders=ph), params)
        datasets = [r["schema_name"] for r in schema_rows]

        if cfg.include_schemas is not None:
            include = set(cfg.include_schemas)
            datasets = [d for d in datasets if d in include]

        # Count per-dataset concurrently
        schema_overviews = (
            await asyncio.gather(
                *[self._count_dataset(db, ds) for ds in datasets],
            )
            if datasets
            else []
        )

        return DatabaseOverview(
            database_name=conn.project,
            database_version=None,
            total_schemas=len(schema_overviews),
            total_tables=sum(s.table_count for s in schema_overviews),
            total_views=sum(s.view_count for s in schema_overviews),
            total_indexes=0,
            total_relationships=0,
            total_columns=sum(s.column_count for s in schema_overviews),
            schemas=list(schema_overviews),
            profiled_at=datetime.now(UTC),
            duration_ms=0,
        )
