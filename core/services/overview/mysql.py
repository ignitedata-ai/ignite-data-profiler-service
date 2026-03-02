"""MySQL overview service — metadata counts via information_schema."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from ignite_data_connectors import BaseConnector, MySQLConfig

from core.api.v1.schemas.overview import DatabaseOverview, OverviewConfig, SchemaOverview
from core.services.overview.base import BaseOverviewService

# ── SQL queries (aiomysql %s placeholders) ─────────────────────────────────────

DATABASE_META = """
SELECT
    DATABASE()  AS db_name,
    VERSION()   AS version
"""

TABLE_COUNTS = """
SELECT table_schema, COUNT(*) AS cnt
FROM information_schema.tables
WHERE table_type = 'BASE TABLE'
  AND table_schema IN ({placeholders})
GROUP BY table_schema
"""

VIEW_COUNTS = """
SELECT table_schema, COUNT(*) AS cnt
FROM information_schema.views
WHERE table_schema IN ({placeholders})
GROUP BY table_schema
"""

INDEX_COUNTS = """
SELECT table_schema, COUNT(DISTINCT index_name, table_name) AS cnt
FROM information_schema.statistics
WHERE table_schema IN ({placeholders})
GROUP BY table_schema
"""

FK_COUNTS = """
SELECT constraint_schema AS table_schema, COUNT(DISTINCT constraint_name) AS cnt
FROM information_schema.referential_constraints
WHERE constraint_schema IN ({placeholders})
GROUP BY constraint_schema
"""

COLUMN_COUNTS = """
SELECT table_schema, COUNT(*) AS cnt
FROM information_schema.columns
WHERE table_schema IN ({placeholders})
GROUP BY table_schema
"""

SYSTEM_SCHEMAS = {"information_schema", "mysql", "performance_schema", "sys"}


def _mysql_placeholders(n: int) -> str:
    return ", ".join(["%s"] * n)


class MySQLOverviewService(BaseOverviewService):
    service_name = "MySQL"
    span_name = "overview.mysql"

    def _span_attributes(self, conn: MySQLConfig) -> dict[str, str | int]:
        return {
            "db.system": "mysql",
            "db.name": conn.database,
            "net.peer.name": conn.host,
            "net.peer.port": conn.port,
        }

    def _log_context(self, conn: MySQLConfig) -> dict[str, Any]:
        return {
            "host": conn.host,
            "port": conn.port,
            "database": conn.database,
            "username": conn.username,
        }

    async def _fetch_overview(
        self,
        db: BaseConnector,
        cfg: OverviewConfig,
        conn: MySQLConfig,
    ) -> DatabaseOverview:
        # MySQL scopes to connected database by default
        db_row = await db.fetch_one(DATABASE_META)
        current_db = db_row["db_name"] if db_row else ""

        if cfg.include_schemas is not None:
            target_schemas = [s for s in cfg.include_schemas if s not in SYSTEM_SCHEMAS]
        else:
            target_schemas = [current_db]

        if not target_schemas:
            return DatabaseOverview(
                database_name=current_db,
                database_version=db_row["version"] if db_row else None,
                total_schemas=0,
                total_tables=0,
                total_views=0,
                total_indexes=0,
                total_relationships=0,
                total_columns=0,
                schemas=[],
                profiled_at=datetime.now(UTC),
                duration_ms=0,
            )

        ph = _mysql_placeholders(len(target_schemas))
        params = tuple(target_schemas)

        tables, views, indexes, fks, columns = await asyncio.gather(
            db.fetch_all(TABLE_COUNTS.format(placeholders=ph), params),
            db.fetch_all(VIEW_COUNTS.format(placeholders=ph), params),
            db.fetch_all(INDEX_COUNTS.format(placeholders=ph), params),
            db.fetch_all(FK_COUNTS.format(placeholders=ph), params),
            db.fetch_all(COLUMN_COUNTS.format(placeholders=ph), params),
        )

        table_map = {r["table_schema"]: r["cnt"] for r in tables}
        view_map = {r["table_schema"]: r["cnt"] for r in views}
        index_map = {r["table_schema"]: r["cnt"] for r in indexes}
        fk_map = {r["table_schema"]: r["cnt"] for r in fks}
        column_map = {r["table_schema"]: r["cnt"] for r in columns}

        schema_overviews = [
            SchemaOverview(
                schema_name=s,
                table_count=table_map.get(s, 0),
                view_count=view_map.get(s, 0),
                index_count=index_map.get(s, 0),
                relationship_count=fk_map.get(s, 0),
                column_count=column_map.get(s, 0),
            )
            for s in target_schemas
        ]

        return DatabaseOverview(
            database_name=current_db,
            database_version=db_row["version"] if db_row else None,
            total_schemas=len(schema_overviews),
            total_tables=sum(s.table_count for s in schema_overviews),
            total_views=sum(s.view_count for s in schema_overviews),
            total_indexes=sum(s.index_count for s in schema_overviews),
            total_relationships=sum(s.relationship_count for s in schema_overviews),
            total_columns=sum(s.column_count for s in schema_overviews),
            schemas=schema_overviews,
            profiled_at=datetime.now(UTC),
            duration_ms=0,
        )
