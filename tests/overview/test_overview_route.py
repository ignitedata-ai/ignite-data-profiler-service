"""Integration tests for the overview route."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from core.api.v1.schemas.overview import DatabaseOverview, SchemaOverview


@pytest.fixture
def mock_overview_result():
    return DatabaseOverview(
        database_name="testdb",
        database_version="PostgreSQL 15.4",
        total_schemas=1,
        total_tables=5,
        total_views=2,
        total_indexes=10,
        total_relationships=3,
        total_columns=30,
        schemas=[
            SchemaOverview(
                schema_name="public",
                table_count=5,
                view_count=2,
                index_count=10,
                relationship_count=3,
                column_count=30,
            )
        ],
        profiled_at=datetime.now(UTC),
        duration_ms=150,
    )


class TestOverviewRoute:
    @pytest.mark.asyncio
    async def test_overview_postgres_success(self, async_client, mock_overview_result):
        with patch(
            "core.api.v1.routes.overview.OVERVIEW_REGISTRY",
            {"postgres": AsyncMock(overview=AsyncMock(return_value=mock_overview_result))},
        ):
            resp = await async_client.post(
                "/api/v1/profile/overview",
                json={
                    "datasource_type": "postgres",
                    "connection": {
                        "host": "localhost",
                        "database": "testdb",
                        "username": "user",
                        "password": "pass",
                    },
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["database_name"] == "testdb"
        assert data["total_tables"] == 5
        assert data["total_schemas"] == 1
        assert len(data["schemas"]) == 1
        assert data["schemas"][0]["schema_name"] == "public"

    @pytest.mark.asyncio
    async def test_overview_invalid_datasource_type(self, async_client):
        resp = await async_client.post(
            "/api/v1/profile/overview",
            json={
                "datasource_type": "oracle",
                "connection": {
                    "host": "localhost",
                    "database": "db",
                    "username": "u",
                    "password": "p",
                },
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_overview_missing_connection(self, async_client):
        resp = await async_client.post(
            "/api/v1/profile/overview",
            json={"datasource_type": "postgres"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_overview_mysql_success(self, async_client, mock_overview_result):
        with patch(
            "core.api.v1.routes.overview.OVERVIEW_REGISTRY",
            {"mysql": AsyncMock(overview=AsyncMock(return_value=mock_overview_result))},
        ):
            resp = await async_client.post(
                "/api/v1/profile/overview",
                json={
                    "datasource_type": "mysql",
                    "connection": {
                        "host": "localhost",
                        "database": "testdb",
                        "username": "user",
                        "password": "pass",
                    },
                },
            )

        assert resp.status_code == 200
        assert resp.json()["total_tables"] == 5
