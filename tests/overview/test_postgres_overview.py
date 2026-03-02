"""Unit tests for the PostgreSQL overview service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.api.v1.schemas.overview import (
    OverviewConfig,
)
from core.services.overview.postgres import PostgresOverviewService


@pytest.fixture
def service():
    return PostgresOverviewService()


@pytest.fixture
def pg_conn():
    conn = MagicMock()
    conn.host = "localhost"
    conn.port = 5432
    conn.database = "testdb"
    conn.username = "user"
    return conn


@pytest.fixture
def mock_db():
    """Mock BaseConnector with fetch_one / fetch_all."""
    db = AsyncMock()
    db.test_connection = AsyncMock()

    db.fetch_one = AsyncMock(return_value={"db_name": "testdb", "version": "PostgreSQL 15.4"})
    db.fetch_all = AsyncMock(side_effect=_default_fetch_all_responses())
    return db


def _default_fetch_all_responses():
    """Returns sequential side_effect values for the 6 fetch_all calls."""
    return [
        # schemas
        [{"schema_name": "public"}, {"schema_name": "app"}],
        # table counts
        [{"table_schema": "public", "cnt": 5}, {"table_schema": "app", "cnt": 3}],
        # view counts
        [{"table_schema": "public", "cnt": 2}],
        # index counts
        [{"table_schema": "public", "cnt": 10}, {"table_schema": "app", "cnt": 4}],
        # fk counts
        [{"table_schema": "public", "cnt": 3}],
        # column counts
        [{"table_schema": "public", "cnt": 30}, {"table_schema": "app", "cnt": 15}],
    ]


class TestPostgresOverviewService:
    def test_span_attributes(self, service, pg_conn):
        attrs = service._span_attributes(pg_conn)
        assert attrs["db.system"] == "postgresql"
        assert attrs["db.name"] == "testdb"
        assert attrs["net.peer.name"] == "localhost"

    def test_log_context(self, service, pg_conn):
        ctx = service._log_context(pg_conn)
        assert ctx["host"] == "localhost"
        assert ctx["database"] == "testdb"

    @pytest.mark.asyncio
    async def test_fetch_overview_assembles_counts(self, service, mock_db, pg_conn):
        cfg = OverviewConfig(exclude_schemas=["pg_catalog", "information_schema", "pg_toast"])
        result = await service._fetch_overview(mock_db, cfg, pg_conn)

        assert result.database_name == "testdb"
        assert result.database_version == "PostgreSQL 15.4"
        assert result.total_schemas == 2
        assert result.total_tables == 8
        assert result.total_views == 2
        assert result.total_indexes == 14
        assert result.total_relationships == 3
        assert result.total_columns == 45

        # Per-schema check
        public = next(s for s in result.schemas if s.schema_name == "public")
        assert public.table_count == 5
        assert public.view_count == 2
        assert public.index_count == 10

        app = next(s for s in result.schemas if s.schema_name == "app")
        assert app.table_count == 3
        assert app.view_count == 0  # not in view results
        assert app.relationship_count == 0  # not in fk results

    @pytest.mark.asyncio
    async def test_fetch_overview_with_include_schemas(self, service, mock_db, pg_conn):
        cfg = OverviewConfig(
            include_schemas=["public"],
            exclude_schemas=["pg_catalog", "information_schema", "pg_toast"],
        )
        result = await service._fetch_overview(mock_db, cfg, pg_conn)

        assert result.total_schemas == 1
        assert result.schemas[0].schema_name == "public"

    @pytest.mark.asyncio
    async def test_fetch_overview_empty_database(self, service, pg_conn):
        db = AsyncMock()
        db.fetch_one = AsyncMock(return_value={"db_name": "emptydb", "version": "15.4"})
        db.fetch_all = AsyncMock(return_value=[])

        cfg = OverviewConfig(exclude_schemas=["pg_catalog", "information_schema"])
        result = await service._fetch_overview(db, cfg, pg_conn)

        assert result.total_schemas == 0
        assert result.total_tables == 0
        assert result.schemas == []

    @pytest.mark.asyncio
    async def test_overview_full_flow(self, service, mock_db, pg_conn):
        """Test the full overview() flow with mocked connector."""
        body = MagicMock()
        body.connection = pg_conn
        body.config = OverviewConfig(
            exclude_schemas=["pg_catalog", "information_schema", "pg_toast"],
        )

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_db)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("core.services.overview.base.create_connector", return_value=mock_ctx):
            result = await service.overview(body)

        assert result.database_name == "testdb"
        assert result.total_schemas == 2
        assert result.duration_ms >= 0
        assert result.profiled_at is not None
