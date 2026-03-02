"""Unit tests for the BigQuery profiler module.

Covers:
- ``_make_json_safe`` — JSON serialisation helper
- ``BigQueryProfiler._filter_tables`` — include/exclude filtering
- ``BigQueryProfiler`` async fetch methods (mocked connector)
- ``BigQueryProfilerService._assemble_response`` — raw-dict → ProfilingResponse
- ``BigQueryProfilerService._span_attributes`` / ``_log_context``
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.api.v1.schemas.profiler import ProfilingConfig, ProfilingResponse
from core.services.bigquery.profiler import (
    BigQueryProfiler,
    BigQueryProfilerService,
    _make_json_safe,
)

# ── _make_json_safe ────────────────────────────────────────────────────────────


class TestMakeJsonSafe:
    def test_none_returns_none(self):
        assert _make_json_safe(None) is None

    def test_str_passes_through(self):
        assert _make_json_safe("hello") == "hello"

    def test_int_passes_through(self):
        assert _make_json_safe(42) == 42

    def test_decimal_to_float(self):
        assert _make_json_safe(Decimal("3.14")) == pytest.approx(3.14)

    def test_datetime_to_isoformat(self):
        dt = datetime(2024, 1, 15, 12, 0, 0)
        assert _make_json_safe(dt) == "2024-01-15T12:00:00"

    def test_date_to_isoformat(self):
        assert _make_json_safe(date(2024, 6, 1)) == "2024-06-01"

    def test_bytes_to_hex(self):
        assert _make_json_safe(b"\x00\xff") == "00ff"

    def test_nested_structure(self):
        result = _make_json_safe({"rows": [Decimal("1"), None, b"\xab"]})
        assert result == {"rows": [1.0, None, "ab"]}

    def test_non_serialisable_falls_back_to_str(self):
        class Unserializable:
            def __str__(self):
                return "custom"

        result = _make_json_safe(Unserializable())
        assert result == "custom"


# ── BigQueryProfiler — filter logic ─────────────────────────────────────────


class TestBigQueryProfilerFilters:
    def _make_profiler(self, config: ProfilingConfig | None = None) -> BigQueryProfiler:
        return BigQueryProfiler(MagicMock(), config or ProfilingConfig(), project="test-project")

    def test_no_filters_returns_all_tables(self):
        tables = [{"table_name": "a"}, {"table_name": "b"}, {"table_name": "c"}]
        profiler = self._make_profiler()
        assert profiler._filter_tables(tables) == tables

    def test_include_tables_keeps_only_specified(self):
        config = ProfilingConfig(include_tables=["a", "c"])
        tables = [{"table_name": "a"}, {"table_name": "b"}, {"table_name": "c"}]
        result = self._make_profiler(config)._filter_tables(tables)
        assert [t["table_name"] for t in result] == ["a", "c"]

    def test_exclude_tables_removes_specified(self):
        config = ProfilingConfig(exclude_tables=["b"])
        tables = [{"table_name": "a"}, {"table_name": "b"}, {"table_name": "c"}]
        result = self._make_profiler(config)._filter_tables(tables)
        assert [t["table_name"] for t in result] == ["a", "c"]

    def test_include_and_exclude_combined(self):
        config = ProfilingConfig(include_tables=["a", "b"], exclude_tables=["b"])
        tables = [{"table_name": "a"}, {"table_name": "b"}, {"table_name": "c"}]
        result = self._make_profiler(config)._filter_tables(tables)
        assert [t["table_name"] for t in result] == ["a"]

    def test_empty_table_list_returns_empty(self):
        assert self._make_profiler()._filter_tables([]) == []


# ── BigQueryProfiler — async fetch methods ─────────────────────────────────


class TestBigQueryProfilerSchemaDiscovery:
    @pytest.mark.asyncio
    async def test_information_schema_always_excluded(self):
        """INFORMATION_SCHEMA must be excluded even when exclude_schemas=[]."""
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [{"schema_name": "my_dataset"}]
        profiler = BigQueryProfiler(mock_db, ProfilingConfig(exclude_schemas=[]), project="proj")

        await profiler._fetch_schema_names()

        params = mock_db.fetch_all.call_args[0][1]
        assert "INFORMATION_SCHEMA" in params

    @pytest.mark.asyncio
    async def test_config_exclude_merged_with_system(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [{"schema_name": "my_dataset"}]
        config = ProfilingConfig(exclude_schemas=["staging", "dev"])
        profiler = BigQueryProfiler(mock_db, config, project="proj")

        await profiler._fetch_schema_names()

        params = mock_db.fetch_all.call_args[0][1]
        assert "staging" in params
        assert "dev" in params
        assert "INFORMATION_SCHEMA" in params

    @pytest.mark.asyncio
    async def test_include_schemas_filter_applied(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [
            {"schema_name": "analytics"},
            {"schema_name": "raw"},
            {"schema_name": "staging"},
        ]
        config = ProfilingConfig(include_schemas=["analytics", "raw"], exclude_schemas=[])
        profiler = BigQueryProfiler(mock_db, config, project="proj")

        result = await profiler._fetch_schema_names()

        assert result == ["analytics", "raw"]

    @pytest.mark.asyncio
    async def test_placeholder_count_matches_exclude_list_length(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        config = ProfilingConfig(exclude_schemas=["extra1", "extra2"])
        profiler = BigQueryProfiler(mock_db, config, project="proj")

        await profiler._fetch_schema_names()

        query: str = mock_db.fetch_all.call_args[0][0]
        params: tuple = mock_db.fetch_all.call_args[0][1]
        assert query.count("%s") == len(params)


class TestBigQueryProfilerColumnFetching:
    @pytest.mark.asyncio
    async def test_columns_with_descriptions_merged(self):
        """Column descriptions from COLUMN_FIELD_PATHS should be merged."""
        mock_db = AsyncMock()
        mock_db.fetch_all.side_effect = [
            [
                {
                    "column_name": "id",
                    "ordinal_position": 1,
                    "data_type": "INT64",
                    "is_nullable": False,
                    "column_default": None,
                    "character_maximum_length": None,
                    "numeric_precision": None,
                    "numeric_scale": None,
                }
            ],
            [{"column_name": "id", "description": "Primary identifier"}],
        ]
        profiler = BigQueryProfiler(mock_db, ProfilingConfig(), project="proj")

        columns = await profiler._fetch_columns("my_dataset", "users")

        assert len(columns) == 1
        assert columns[0]["description"] == "Primary identifier"
        assert columns[0]["is_primary_key"] is False
        assert columns[0]["enum_values"] is None

    @pytest.mark.asyncio
    async def test_columns_without_descriptions(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.side_effect = [
            [
                {
                    "column_name": "status",
                    "ordinal_position": 1,
                    "data_type": "STRING",
                    "is_nullable": True,
                    "column_default": None,
                    "character_maximum_length": None,
                    "numeric_precision": None,
                    "numeric_scale": None,
                }
            ],
            [],  # no descriptions
        ]
        profiler = BigQueryProfiler(mock_db, ProfilingConfig(), project="proj")

        columns = await profiler._fetch_columns("my_dataset", "users")

        assert columns[0]["description"] is None

    @pytest.mark.asyncio
    async def test_is_primary_key_always_false(self):
        """BigQuery does not have primary keys."""
        mock_db = AsyncMock()
        mock_db.fetch_all.side_effect = [
            [
                {
                    "column_name": "id",
                    "ordinal_position": 1,
                    "data_type": "INT64",
                    "is_nullable": False,
                    "column_default": None,
                    "character_maximum_length": None,
                    "numeric_precision": None,
                    "numeric_scale": None,
                }
            ],
            [],
        ]
        profiler = BigQueryProfiler(mock_db, ProfilingConfig(), project="proj")

        columns = await profiler._fetch_columns("my_dataset", "users")

        assert columns[0]["is_primary_key"] is False


class TestBigQueryProfilerIndexFetching:
    @pytest.mark.asyncio
    async def test_always_returns_empty_list(self):
        """BigQuery does not have traditional indexes."""
        profiler = BigQueryProfiler(MagicMock(), ProfilingConfig(), project="proj")
        indexes = await profiler._fetch_indexes()
        assert indexes == []


class TestBigQueryProfilerForeignKeys:
    @pytest.mark.asyncio
    async def test_always_returns_empty_list(self):
        """BigQuery does not support foreign key constraints."""
        profiler = BigQueryProfiler(MagicMock(), ProfilingConfig(), project="proj")
        fks = await profiler._fetch_foreign_keys()
        assert fks == []


class TestBigQueryProfilerRowCount:
    @pytest.mark.asyncio
    async def test_returns_estimate_from_information_schema(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {"row_count": 500}
        profiler = BigQueryProfiler(mock_db, ProfilingConfig(), project="proj")

        result = await profiler._fetch_row_count("my_dataset", "orders")

        assert result == 500

    @pytest.mark.asyncio
    async def test_falls_back_to_exact_count_when_null(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.side_effect = [
            {"row_count": None},
            {"row_count": 42},
        ]
        profiler = BigQueryProfiler(mock_db, ProfilingConfig(), project="proj")

        result = await profiler._fetch_row_count("my_dataset", "orders")

        assert result == 42
        assert mock_db.fetch_one.call_count == 2

    @pytest.mark.asyncio
    async def test_returns_none_when_table_not_found(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = None
        profiler = BigQueryProfiler(mock_db, ProfilingConfig(), project="proj")

        result = await profiler._fetch_row_count("my_dataset", "ghost")

        assert result is None


class TestBigQueryProfilerDataFreshness:
    @pytest.mark.asyncio
    async def test_maps_created_to_last_analyze(self):
        created_dt = datetime(2024, 1, 1, tzinfo=UTC)
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {"created": created_dt}
        profiler = BigQueryProfiler(mock_db, ProfilingConfig(), project="proj")

        result = await profiler._fetch_data_freshness("my_dataset", "orders")

        assert result["last_analyze"] == created_dt

    @pytest.mark.asyncio
    async def test_maintenance_fields_always_none(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {
            "created": datetime(2024, 1, 1, tzinfo=UTC),
        }
        profiler = BigQueryProfiler(mock_db, ProfilingConfig(), project="proj")

        result = await profiler._fetch_data_freshness("my_dataset", "t")

        assert result["last_autoanalyze"] is None
        assert result["last_vacuum"] is None
        assert result["last_autovacuum"] is None

    @pytest.mark.asyncio
    async def test_returns_none_when_row_missing(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = None
        profiler = BigQueryProfiler(mock_db, ProfilingConfig(), project="proj")

        result = await profiler._fetch_data_freshness("my_dataset", "ghost")

        assert result is None


class TestBigQueryProfilerSampleData:
    @pytest.mark.asyncio
    async def test_uses_backtick_quoting(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = BigQueryProfiler(mock_db, ProfilingConfig(sample_size=5), project="proj")

        await profiler._fetch_sample_data("my_dataset", "users")

        query: str = mock_db.fetch_all.call_args[0][0]
        assert "`my_dataset`" in query
        assert "`users`" in query
        assert "RAND()" in query
        assert "LIMIT" in query

    @pytest.mark.asyncio
    async def test_escapes_backticks_in_identifiers(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = BigQueryProfiler(mock_db, ProfilingConfig(sample_size=3), project="proj")

        await profiler._fetch_sample_data("my`dataset", "t`able")

        query: str = mock_db.fetch_all.call_args[0][0]
        assert "`my``dataset`" in query
        assert "`t``able`" in query

    @pytest.mark.asyncio
    async def test_sample_size_passed_as_param(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = BigQueryProfiler(mock_db, ProfilingConfig(sample_size=25), project="proj")

        await profiler._fetch_sample_data("my_dataset", "t")

        params = mock_db.fetch_all.call_args[0][1]
        assert params == (25,)


class TestBigQueryProfilerViews:
    @pytest.mark.asyncio
    async def test_view_fields_mapped_correctly(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [
            {
                "table_name": "active_users",
                "table_schema": "my_dataset",
                "owner": None,
                "definition": "SELECT * FROM users WHERE status = 'active'",
            }
        ]
        profiler = BigQueryProfiler(mock_db, ProfilingConfig(), project="proj")

        views = await profiler._fetch_views("my_dataset")

        assert len(views) == 1
        v = views[0]
        assert v["name"] == "active_users"
        assert v["schema"] == "my_dataset"
        assert v["owner"] == ""
        assert "SELECT" in v["definition"]

    @pytest.mark.asyncio
    async def test_no_views_returns_empty_list(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = BigQueryProfiler(mock_db, ProfilingConfig(), project="proj")

        assert await profiler._fetch_views("my_dataset") == []


class TestBigQueryProfilerDatabaseMetadata:
    @pytest.mark.asyncio
    async def test_returns_project_as_name(self):
        profiler = BigQueryProfiler(MagicMock(), ProfilingConfig(), project="my-gcp-project")

        meta = await profiler._fetch_database_metadata()

        assert meta["name"] == "my-gcp-project"
        assert meta["version"] == ""
        assert meta["encoding"] == "UTF-8"
        assert meta["size_bytes"] == 0


# ── BigQueryProfilerService._assemble_response ────────────────────────────


class TestBigQueryProfilerServiceAssemble:
    _RAW_EMPTY = {
        "database": {
            "name": "my-project",
            "version": "",
            "encoding": "UTF-8",
            "size_bytes": 0,
        },
        "schemas": [],
    }

    _RAW_FULL = {
        "database": {
            "name": "my-project",
            "version": "",
            "encoding": "UTF-8",
            "size_bytes": 0,
        },
        "schemas": [
            {
                "name": "my_dataset",
                "owner": "",
                "tables": [
                    {
                        "name": "users",
                        "schema": "my_dataset",
                        "owner": "",
                        "description": "User accounts",
                        "size_bytes": 4096,
                        "total_size_bytes": 4096,
                        "row_count": 100,
                        "columns": [
                            {
                                "name": "id",
                                "ordinal_position": 1,
                                "data_type": "INT64",
                                "is_nullable": False,
                                "column_default": None,
                                "character_maximum_length": None,
                                "numeric_precision": None,
                                "numeric_scale": None,
                                "is_primary_key": False,
                                "description": "User ID",
                                "enum_values": None,
                                "sample_values": None,
                            },
                        ],
                        "indexes": [],
                        "relationships": [],
                        "data_freshness": {
                            "last_analyze": datetime(2024, 1, 1, tzinfo=UTC),
                            "last_autoanalyze": None,
                            "last_vacuum": None,
                            "last_autovacuum": None,
                        },
                    }
                ],
                "views": [
                    {
                        "name": "active_users",
                        "schema": "my_dataset",
                        "owner": "",
                        "definition": "SELECT * FROM users WHERE active = TRUE",
                    }
                ],
            }
        ],
    }

    def test_assembles_empty_schema_list(self):
        svc = BigQueryProfilerService()
        now = datetime.now(tz=UTC)
        result = svc._assemble_response(self._RAW_EMPTY, now)

        assert isinstance(result, ProfilingResponse)
        assert result.database.name == "my-project"
        assert result.schemas == []
        assert result.profiled_at == now

    def test_assembles_schema_with_table_and_view(self):
        svc = BigQueryProfilerService()
        result = svc._assemble_response(self._RAW_FULL, datetime.now(tz=UTC))

        assert len(result.schemas) == 1
        schema = result.schemas[0]
        assert schema.name == "my_dataset"
        assert len(schema.tables) == 1
        assert len(schema.views) == 1

    def test_table_metadata_fields(self):
        svc = BigQueryProfilerService()
        result = svc._assemble_response(self._RAW_FULL, datetime.now(tz=UTC))

        table = result.schemas[0].tables[0]
        assert table.name == "users"
        assert table.row_count == 100
        assert table.size_bytes == 4096
        assert table.description == "User accounts"

    def test_column_metadata(self):
        svc = BigQueryProfilerService()
        result = svc._assemble_response(self._RAW_FULL, datetime.now(tz=UTC))

        columns = result.schemas[0].tables[0].columns
        assert len(columns) == 1
        assert columns[0].name == "id"
        assert columns[0].data_type == "INT64"
        assert columns[0].is_primary_key is False
        assert columns[0].description == "User ID"

    def test_data_freshness_mapping(self):
        svc = BigQueryProfilerService()
        result = svc._assemble_response(self._RAW_FULL, datetime.now(tz=UTC))

        freshness = result.schemas[0].tables[0].data_freshness
        assert freshness is not None
        assert freshness.last_analyze == datetime(2024, 1, 1, tzinfo=UTC)
        assert freshness.last_vacuum is None
        assert freshness.last_autoanalyze is None
        assert freshness.last_autovacuum is None


# ── BigQueryProfilerService metadata ──────────────────────────────────────────


class TestBigQueryProfilerServiceMetadata:
    def test_span_attributes(self):
        from ignite_data_connectors import BigQueryConfig

        svc = BigQueryProfilerService()
        conn = BigQueryConfig(
            project="my-gcp-project",
            location="US",
            dataset="my_dataset",
        )
        attrs = svc._span_attributes(conn)

        assert attrs["db.system"] == "bigquery"
        assert attrs["db.name"] == "my-gcp-project"
        assert attrs["bigquery.location"] == "US"
        assert attrs["bigquery.dataset"] == "my_dataset"

    def test_span_attributes_without_optional(self):
        from ignite_data_connectors import BigQueryConfig

        svc = BigQueryProfilerService()
        conn = BigQueryConfig(project="my-gcp-project")
        attrs = svc._span_attributes(conn)

        assert attrs["db.system"] == "bigquery"
        assert attrs["db.name"] == "my-gcp-project"
        assert "bigquery.location" not in attrs
        assert "bigquery.dataset" not in attrs

    def test_log_context(self):
        from ignite_data_connectors import BigQueryConfig

        svc = BigQueryProfilerService()
        conn = BigQueryConfig(
            project="my-gcp-project",
            location="EU",
            dataset="my_dataset",
        )
        ctx = svc._log_context(conn)

        assert ctx["host"] == "my-gcp-project"
        assert ctx["database"] == "my-gcp-project"
        assert ctx["location"] == "EU"
        assert ctx["dataset"] == "my_dataset"

    def test_service_name_and_span_name(self):
        svc = BigQueryProfilerService()
        assert svc.service_name == "BigQuery"
        assert svc.span_name == "profiler.bigquery"
