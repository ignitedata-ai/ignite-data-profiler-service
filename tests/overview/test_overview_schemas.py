"""Unit tests for overview Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from core.api.v1.schemas.overview import (
    DatabaseOverview,
    MySQLOverviewRequest,
    OverviewConfig,
    OverviewRequest,
    PostgresOverviewRequest,
    SchemaOverview,
    SnowflakeOverviewRequest,
)


class TestOverviewConfig:
    def test_defaults(self):
        cfg = OverviewConfig()
        assert cfg.include_schemas is None
        assert "pg_catalog" in cfg.exclude_schemas
        assert cfg.timeout_seconds == 60

    def test_timeout_min(self):
        with pytest.raises(PydanticValidationError):
            OverviewConfig(timeout_seconds=0)

    def test_timeout_max(self):
        with pytest.raises(PydanticValidationError):
            OverviewConfig(timeout_seconds=301)

    def test_timeout_boundary_values(self):
        assert OverviewConfig(timeout_seconds=1).timeout_seconds == 1
        assert OverviewConfig(timeout_seconds=300).timeout_seconds == 300

    def test_custom_exclude(self):
        cfg = OverviewConfig(exclude_schemas=["my_internal"])
        assert cfg.exclude_schemas == ["my_internal"]


class TestOverviewRequest:
    _pg_conn = dict(host="localhost", database="db", username="u", password="p")

    def test_discriminated_union_resolves_postgres(self):
        adapter = TypeAdapter(OverviewRequest)
        obj = adapter.validate_python({"datasource_type": "postgres", "connection": self._pg_conn})
        assert isinstance(obj, PostgresOverviewRequest)

    def test_discriminated_union_resolves_mysql(self):
        adapter = TypeAdapter(OverviewRequest)
        obj = adapter.validate_python({"datasource_type": "mysql", "connection": self._pg_conn})
        assert isinstance(obj, MySQLOverviewRequest)

    def test_unknown_datasource_type_rejected(self):
        adapter = TypeAdapter(OverviewRequest)
        with pytest.raises(PydanticValidationError):
            adapter.validate_python({"datasource_type": "oracle", "connection": self._pg_conn})

    def test_s3_file_not_supported(self):
        adapter = TypeAdapter(OverviewRequest)
        with pytest.raises(PydanticValidationError):
            adapter.validate_python({"datasource_type": "s3_file", "connection": {}})

    def test_mysql_default_exclude_schemas(self):
        adapter = TypeAdapter(OverviewRequest)
        obj = adapter.validate_python({"datasource_type": "mysql", "connection": self._pg_conn})
        assert "mysql" in obj.config.exclude_schemas
        assert "performance_schema" in obj.config.exclude_schemas

    def test_snowflake_default_exclude_schemas(self):
        adapter = TypeAdapter(OverviewRequest)
        obj = adapter.validate_python(
            {
                "datasource_type": "snowflake",
                "connection": {
                    "account": "acct",
                    "username": "u",
                    "password": "p",
                    "database": "db",
                },
            }
        )
        assert isinstance(obj, SnowflakeOverviewRequest)
        assert "INFORMATION_SCHEMA" in obj.config.exclude_schemas

    def test_config_defaults_applied(self):
        adapter = TypeAdapter(OverviewRequest)
        obj = adapter.validate_python({"datasource_type": "postgres", "connection": self._pg_conn})
        assert obj.config.timeout_seconds == 60


class TestSchemaOverview:
    def test_schema_overview_serialization(self):
        so = SchemaOverview(
            schema_name="public",
            table_count=10,
            view_count=3,
            index_count=15,
            relationship_count=5,
            column_count=80,
        )
        data = so.model_dump()
        assert data["schema_name"] == "public"
        assert data["table_count"] == 10
        assert data["column_count"] == 80


class TestDatabaseOverview:
    def test_database_overview_totals(self):
        from datetime import UTC, datetime

        overview = DatabaseOverview(
            database_name="testdb",
            database_version="15.1",
            total_schemas=2,
            total_tables=20,
            total_views=5,
            total_indexes=30,
            total_relationships=10,
            total_columns=150,
            schemas=[],
            profiled_at=datetime.now(UTC),
            duration_ms=123,
        )
        data = overview.model_dump()
        assert data["database_name"] == "testdb"
        assert data["total_tables"] == 20
        assert data["duration_ms"] == 123
