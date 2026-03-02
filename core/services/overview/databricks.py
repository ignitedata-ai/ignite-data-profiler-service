"""Databricks overview service — metadata counts via INFORMATION_SCHEMA."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from ignite_data_connectors import BaseConnector, DatabricksConfig

from core.api.v1.schemas.overview import DatabaseOverview, OverviewConfig, SchemaOverview
from core.services.overview.base import BaseOverviewService

# ── SQL queries (? placeholders, backtick aliases) ─────────────────────────────

DATABASE_META = """
SELECT
    CURRENT_CATALOG() AS `db_name`,
    ''                AS `version`
"""

SCHEMAS = """
SELECT SCHEMA_NAME AS `schema_name`
FROM INFORMATION_SCHEMA.SCHEMATA
WHERE SCHEMA_NAME NOT IN ({placeholders})
ORDER BY SCHEMA_NAME
"""

TABLE_COUNTS = """
SELECT TABLE_SCHEMA AS `table_schema`, COUNT(*) AS `cnt`
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE'
  AND TABLE_SCHEMA NOT IN ({placeholders})
GROUP BY TABLE_SCHEMA
"""

VIEW_COUNTS = """
SELECT TABLE_SCHEMA AS `table_schema`, COUNT(*) AS `cnt`
FROM INFORMATION_SCHEMA.VIEWS
WHERE TABLE_SCHEMA NOT IN ({placeholders})
GROUP BY TABLE_SCHEMA
"""

FK_COUNTS = """
SELECT TABLE_SCHEMA AS `table_schema`, COUNT(*) AS `cnt`
FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
WHERE CONSTRAINT_TYPE = 'FOREIGN KEY'
  AND TABLE_SCHEMA NOT IN ({placeholders})
GROUP BY TABLE_SCHEMA
"""

COLUMN_COUNTS = """
SELECT TABLE_SCHEMA AS `table_schema`, COUNT(*) AS `cnt`
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA NOT IN ({placeholders})
GROUP BY TABLE_SCHEMA
"""


def _db_placeholders(n: int) -> str:
    return ", ".join(["?"] * n)


class DatabricksOverviewService(BaseOverviewService):
    service_name = "Databricks"
    span_name = "overview.databricks"

    def _span_attributes(self, conn: DatabricksConfig) -> dict[str, str | int]:
        attrs: dict[str, str | int] = {
            "db.system": "databricks",
            "db.name": conn.catalog or "",
            "databricks.server_hostname": conn.server_hostname,
        }
        if conn.http_path:
            attrs["databricks.http_path"] = conn.http_path
        return attrs

    def _log_context(self, conn: DatabricksConfig) -> dict[str, Any]:
        return {
            "host": conn.server_hostname,
            "database": conn.catalog or "",
            "http_path": conn.http_path,
        }

    async def _fetch_overview(
        self,
        db: BaseConnector,
        cfg: OverviewConfig,
        conn: DatabricksConfig,
    ) -> DatabaseOverview:
        exclude = list(cfg.exclude_schemas)
        ph = _db_placeholders(len(exclude))
        params = tuple(exclude)

        db_row, schema_rows, tables, views, fks, columns = await asyncio.gather(
            db.fetch_one(DATABASE_META),
            db.fetch_all(SCHEMAS.format(placeholders=ph), params),
            db.fetch_all(TABLE_COUNTS.format(placeholders=ph), params),
            db.fetch_all(VIEW_COUNTS.format(placeholders=ph), params),
            db.fetch_all(FK_COUNTS.format(placeholders=ph), params),
            db.fetch_all(COLUMN_COUNTS.format(placeholders=ph), params),
        )

        all_schemas = [r["schema_name"] for r in schema_rows]
        if cfg.include_schemas is not None:
            include = set(cfg.include_schemas)
            all_schemas = [s for s in all_schemas if s in include]

        table_map = {r["table_schema"]: r["cnt"] for r in tables}
        view_map = {r["table_schema"]: r["cnt"] for r in views}
        fk_map = {r["table_schema"]: r["cnt"] for r in fks}
        column_map = {r["table_schema"]: r["cnt"] for r in columns}

        schema_overviews = [
            SchemaOverview(
                schema_name=s,
                table_count=table_map.get(s, 0),
                view_count=view_map.get(s, 0),
                index_count=0,  # Databricks has no traditional indexes
                relationship_count=fk_map.get(s, 0),
                column_count=column_map.get(s, 0),
            )
            for s in all_schemas
        ]

        return DatabaseOverview(
            database_name=db_row["db_name"] if db_row else "",
            database_version=db_row["version"] if db_row else None,
            total_schemas=len(schema_overviews),
            total_tables=sum(s.table_count for s in schema_overviews),
            total_views=sum(s.view_count for s in schema_overviews),
            total_indexes=0,
            total_relationships=sum(s.relationship_count for s in schema_overviews),
            total_columns=sum(s.column_count for s in schema_overviews),
            schemas=schema_overviews,
            profiled_at=datetime.now(UTC),
            duration_ms=0,
        )
