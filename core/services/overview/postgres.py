"""PostgreSQL overview service — metadata counts via information_schema."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from ignite_data_connectors import BaseConnector, PostgresConfig

from core.api.v1.schemas.overview import DatabaseOverview, OverviewConfig, SchemaOverview
from core.services.overview.base import BaseOverviewService

# ── SQL queries (asyncpg $1/$2 placeholders) ──────────────────────────────────

DATABASE_META = """
SELECT
    current_database()   AS db_name,
    version()            AS version
"""

SCHEMAS = """
SELECT schema_name
FROM information_schema.schemata
WHERE schema_name NOT IN ({placeholders})
ORDER BY schema_name
"""

TABLE_COUNTS = """
SELECT table_schema, COUNT(*) AS cnt
FROM information_schema.tables
WHERE table_type = 'BASE TABLE'
  AND table_schema NOT IN ({placeholders})
GROUP BY table_schema
"""

VIEW_COUNTS = """
SELECT table_schema, COUNT(*) AS cnt
FROM information_schema.views
WHERE table_schema NOT IN ({placeholders})
GROUP BY table_schema
"""

INDEX_COUNTS = """
SELECT n.nspname AS table_schema, COUNT(DISTINCT i.oid) AS cnt
FROM pg_index ix
JOIN pg_class i ON i.oid = ix.indexrelid
JOIN pg_class t ON t.oid = ix.indrelid
JOIN pg_namespace n ON n.oid = t.relnamespace
WHERE n.nspname NOT IN ({placeholders})
GROUP BY n.nspname
"""

FK_COUNTS = """
SELECT tc.table_schema, COUNT(*) AS cnt
FROM information_schema.table_constraints tc
WHERE tc.constraint_type = 'FOREIGN KEY'
  AND tc.table_schema NOT IN ({placeholders})
GROUP BY tc.table_schema
"""

COLUMN_COUNTS = """
SELECT table_schema, COUNT(*) AS cnt
FROM information_schema.columns
WHERE table_schema NOT IN ({placeholders})
GROUP BY table_schema
"""


def _pg_placeholders(n: int) -> str:
    """Build $1, $2, ... placeholder string for asyncpg."""
    return ", ".join(f"${i + 1}" for i in range(n))


class PostgresOverviewService(BaseOverviewService):
    service_name = "PostgreSQL"
    span_name = "overview.postgres"

    def _span_attributes(self, conn: PostgresConfig) -> dict[str, str | int]:
        return {
            "db.system": "postgresql",
            "db.name": conn.database,
            "net.peer.name": conn.host,
            "net.peer.port": conn.port,
        }

    def _log_context(self, conn: PostgresConfig) -> dict[str, Any]:
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
        conn: PostgresConfig,
    ) -> DatabaseOverview:
        exclude = list(cfg.exclude_schemas)
        ph = _pg_placeholders(len(exclude))
        params = tuple(exclude)

        # Discover schemas + database meta + all 5 counts concurrently
        db_row, schema_rows, tables, views, indexes, fks, columns = await asyncio.gather(
            db.fetch_one(DATABASE_META),
            db.fetch_all(SCHEMAS.format(placeholders=ph), params),
            db.fetch_all(TABLE_COUNTS.format(placeholders=ph), params),
            db.fetch_all(VIEW_COUNTS.format(placeholders=ph), params),
            db.fetch_all(INDEX_COUNTS.format(placeholders=ph), params),
            db.fetch_all(FK_COUNTS.format(placeholders=ph), params),
            db.fetch_all(COLUMN_COUNTS.format(placeholders=ph), params),
        )

        all_schemas = [r["schema_name"] for r in schema_rows]
        if cfg.include_schemas is not None:
            include = set(cfg.include_schemas)
            all_schemas = [s for s in all_schemas if s in include]

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
            for s in all_schemas
        ]

        return DatabaseOverview(
            database_name=db_row["db_name"] if db_row else "",
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
