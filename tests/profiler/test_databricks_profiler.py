"""Unit tests for the Databricks profiler module.

Covers:
- ``_make_json_safe`` — JSON serialisation helper
- ``DatabricksProfiler._filter_tables`` — include/exclude filtering
- ``DatabricksProfiler`` async fetch methods (mocked connector)
- ``DatabricksProfilerService._assemble_response`` — raw-dict → ProfilingResponse
- ``DatabricksProfilerService._span_attributes`` / ``_log_context``
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.api.v1.schemas.profiler import ProfilingConfig, ProfilingResponse
from core.services.databricks.profiler import (
    DatabricksProfiler,
    DatabricksProfilerService,
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

    def test_float_passes_through(self):
        assert _make_json_safe(3.14) == 3.14

    def test_bool_passes_through(self):
        assert _make_json_safe(True) is True

    def test_decimal_to_float(self):
        assert _make_json_safe(Decimal("3.14")) == pytest.approx(3.14)

    def test_datetime_to_isoformat(self):
        dt = datetime(2024, 1, 15, 12, 0, 0)
        assert _make_json_safe(dt) == "2024-01-15T12:00:00"

    def test_date_to_isoformat(self):
        assert _make_json_safe(date(2024, 6, 1)) == "2024-06-01"

    def test_bytes_to_hex(self):
        assert _make_json_safe(b"\x00\xff") == "00ff"

    def test_bytearray_to_hex(self):
        assert _make_json_safe(bytearray(b"\xde\xad")) == "dead"

    def test_list_recursively_converted(self):
        result = _make_json_safe([Decimal("1.5"), None, "text"])
        assert result == [1.5, None, "text"]

    def test_dict_recursively_converted(self):
        result = _make_json_safe({"value": Decimal("2.5"), "name": "test"})
        assert result == {"value": 2.5, "name": "test"}

    def test_nested_structure(self):
        result = _make_json_safe({"rows": [Decimal("1"), None, b"\xab"]})
        assert result == {"rows": [1.0, None, "ab"]}

    def test_non_serialisable_falls_back_to_str(self):
        class Unserializable:
            def __str__(self):
                return "custom"

        result = _make_json_safe(Unserializable())
        assert result == "custom"


# ── DatabricksProfiler — filter logic ─────────────────────────────────────────


class TestDatabricksProfilerFilters:
    def _make_profiler(self, config: ProfilingConfig | None = None) -> DatabricksProfiler:
        return DatabricksProfiler(MagicMock(), config or ProfilingConfig())

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

    def test_include_filter_with_unknown_name_returns_empty(self):
        config = ProfilingConfig(include_tables=["nonexistent"])
        tables = [{"table_name": "a"}, {"table_name": "b"}]
        result = self._make_profiler(config)._filter_tables(tables)
        assert result == []


# ── DatabricksProfiler — async fetch methods ─────────────────────────────────


class TestDatabricksProfilerSchemaDiscovery:
    @pytest.mark.asyncio
    async def test_information_schema_always_excluded(self):
        """information_schema must be excluded even when exclude_schemas=[]."""
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [{"schema_name": "default"}]
        profiler = DatabricksProfiler(mock_db, ProfilingConfig(exclude_schemas=[]))

        await profiler._fetch_schema_names()

        params = mock_db.fetch_all.call_args[0][1]
        assert "information_schema" in params

    @pytest.mark.asyncio
    async def test_config_exclude_merged_with_system(self):
        """User-specified exclude_schemas should be combined with system exclusions."""
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [{"schema_name": "default"}]
        config = ProfilingConfig(exclude_schemas=["staging", "dev"])
        profiler = DatabricksProfiler(mock_db, config)

        await profiler._fetch_schema_names()

        params = mock_db.fetch_all.call_args[0][1]
        assert "staging" in params
        assert "dev" in params
        assert "information_schema" in params

    @pytest.mark.asyncio
    async def test_include_schemas_filter_applied(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [
            {"schema_name": "default"},
            {"schema_name": "analytics"},
            {"schema_name": "raw"},
        ]
        config = ProfilingConfig(include_schemas=["default", "raw"], exclude_schemas=[])
        profiler = DatabricksProfiler(mock_db, config)

        result = await profiler._fetch_schema_names()

        assert result == ["default", "raw"]

    @pytest.mark.asyncio
    async def test_placeholder_count_matches_exclude_list_length(self):
        """The SQL query must have the correct number of ? placeholders."""
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        config = ProfilingConfig(exclude_schemas=["extra1", "extra2"])
        profiler = DatabricksProfiler(mock_db, config)

        await profiler._fetch_schema_names()

        query: str = mock_db.fetch_all.call_args[0][0]
        params: tuple = mock_db.fetch_all.call_args[0][1]
        assert query.count("?") == len(params)


class TestDatabricksProfilerColumnFetching:
    @pytest.mark.asyncio
    async def test_primary_key_column_flagged(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.side_effect = [
            [
                {
                    "column_name": "id",
                    "ordinal_position": 1,
                    "data_type": "bigint",
                    "is_nullable": False,
                    "column_default": None,
                    "character_maximum_length": None,
                    "numeric_precision": 19,
                    "numeric_scale": 0,
                    "description": None,
                }
            ],
            [{"column_name": "id"}],  # PK
        ]
        profiler = DatabricksProfiler(mock_db, ProfilingConfig())

        columns = await profiler._fetch_columns("default", "users")

        assert columns[0]["is_primary_key"] is True
        assert columns[0]["enum_values"] is None

    @pytest.mark.asyncio
    async def test_enum_values_always_none(self):
        """Databricks does not have native enum types."""
        mock_db = AsyncMock()
        mock_db.fetch_all.side_effect = [
            [
                {
                    "column_name": "status",
                    "ordinal_position": 1,
                    "data_type": "string",
                    "is_nullable": True,
                    "column_default": None,
                    "character_maximum_length": None,
                    "numeric_precision": None,
                    "numeric_scale": None,
                    "description": "Account status",
                }
            ],
            [],  # no PK
        ]
        profiler = DatabricksProfiler(mock_db, ProfilingConfig())

        columns = await profiler._fetch_columns("default", "users")

        assert len(columns) == 1
        assert columns[0]["enum_values"] is None

    @pytest.mark.asyncio
    async def test_empty_description_normalised_to_none(self):
        """An empty COMMENT should become None."""
        mock_db = AsyncMock()
        mock_db.fetch_all.side_effect = [
            [
                {
                    "column_name": "col",
                    "ordinal_position": 1,
                    "data_type": "int",
                    "is_nullable": False,
                    "column_default": None,
                    "character_maximum_length": None,
                    "numeric_precision": 10,
                    "numeric_scale": 0,
                    "description": "",
                }
            ],
            [],
        ]
        profiler = DatabricksProfiler(mock_db, ProfilingConfig())

        columns = await profiler._fetch_columns("default", "t")

        assert columns[0]["description"] is None


class TestDatabricksProfilerIndexFetching:
    @pytest.mark.asyncio
    async def test_always_returns_empty_list(self):
        """Databricks does not have traditional indexes."""
        profiler = DatabricksProfiler(MagicMock(), ProfilingConfig())
        indexes = await profiler._fetch_indexes()
        assert indexes == []


class TestDatabricksProfilerRowCount:
    @pytest.mark.asyncio
    async def test_returns_exact_count(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {"row_count": 500}
        profiler = DatabricksProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_row_count("default", "orders")

        assert result == 500

    @pytest.mark.asyncio
    async def test_returns_none_when_table_not_found(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = None
        profiler = DatabricksProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_row_count("default", "ghost")

        assert result is None

    @pytest.mark.asyncio
    async def test_uses_backtick_escaped_identifiers(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {"row_count": 10}
        profiler = DatabricksProfiler(mock_db, ProfilingConfig())

        await profiler._fetch_row_count("my`schema", "my`table")

        query: str = mock_db.fetch_one.call_args[0][0]
        assert "`my``schema`" in query
        assert "`my``table`" in query


class TestDatabricksProfilerDataFreshness:
    @pytest.mark.asyncio
    async def test_maps_last_altered_to_last_analyze(self):
        altered_dt = datetime(2024, 3, 12, tzinfo=UTC)
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {
            "last_altered": altered_dt,
            "created": datetime(2024, 1, 1, tzinfo=UTC),
        }
        profiler = DatabricksProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_data_freshness("default", "orders")

        assert result["last_analyze"] == altered_dt

    @pytest.mark.asyncio
    async def test_maps_created_to_last_vacuum(self):
        created_dt = datetime(2024, 1, 1, tzinfo=UTC)
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {
            "last_altered": None,
            "created": created_dt,
        }
        profiler = DatabricksProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_data_freshness("default", "orders")

        assert result["last_vacuum"] == created_dt

    @pytest.mark.asyncio
    async def test_auto_fields_always_none(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {
            "last_altered": datetime(2024, 1, 2, tzinfo=UTC),
            "created": datetime(2024, 1, 1, tzinfo=UTC),
        }
        profiler = DatabricksProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_data_freshness("default", "t")

        assert result["last_autoanalyze"] is None
        assert result["last_autovacuum"] is None

    @pytest.mark.asyncio
    async def test_returns_none_when_row_missing(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = None
        profiler = DatabricksProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_data_freshness("default", "ghost")

        assert result is None


class TestDatabricksProfilerSampleData:
    @pytest.mark.asyncio
    async def test_uses_backtick_quoting(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = DatabricksProfiler(mock_db, ProfilingConfig(sample_size=5))

        await profiler._fetch_sample_data("default", "users")

        query: str = mock_db.fetch_all.call_args[0][0]
        assert "`default`" in query
        assert "`users`" in query
        assert "RAND()" in query
        assert "LIMIT" in query

    @pytest.mark.asyncio
    async def test_escapes_backticks_in_identifiers(self):
        """A backtick in the identifier name must be doubled to prevent injection."""
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = DatabricksProfiler(mock_db, ProfilingConfig(sample_size=3))

        await profiler._fetch_sample_data("my`schema", "t`able")

        query: str = mock_db.fetch_all.call_args[0][0]
        assert "`my``schema`" in query
        assert "`t``able`" in query

    @pytest.mark.asyncio
    async def test_sample_size_passed_as_param(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = DatabricksProfiler(mock_db, ProfilingConfig(sample_size=25))

        await profiler._fetch_sample_data("default", "t")

        params = mock_db.fetch_all.call_args[0][1]
        assert params == (25,)


class TestDatabricksProfilerViews:
    @pytest.mark.asyncio
    async def test_view_fields_mapped_correctly(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [
            {
                "table_name": "active_users",
                "table_schema": "default",
                "owner": "admin",
                "definition": "SELECT * FROM users WHERE status = 'active'",
            }
        ]
        profiler = DatabricksProfiler(mock_db, ProfilingConfig())

        views = await profiler._fetch_views("default")

        assert len(views) == 1
        v = views[0]
        assert v["name"] == "active_users"
        assert v["schema"] == "default"
        assert v["owner"] == "admin"
        assert "SELECT" in v["definition"]

    @pytest.mark.asyncio
    async def test_no_views_returns_empty_list(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = DatabricksProfiler(mock_db, ProfilingConfig())

        assert await profiler._fetch_views("default") == []


class TestDatabricksProfilerForeignKeys:
    @pytest.mark.asyncio
    async def test_foreign_key_fields_mapped_correctly(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [
            {
                "constraint_name": "fk_order_user",
                "column_name": "user_id",
                "foreign_table_schema": "default",
                "foreign_table_name": "users",
                "foreign_column_name": "id",
                "update_rule": "NO ACTION",
                "delete_rule": "NO ACTION",
            }
        ]
        profiler = DatabricksProfiler(mock_db, ProfilingConfig())

        fks = await profiler._fetch_foreign_keys("default", "orders")

        assert len(fks) == 1
        fk = fks[0]
        assert fk["constraint_name"] == "fk_order_user"
        assert fk["from_column"] == "user_id"
        assert fk["to_schema"] == "default"
        assert fk["to_table"] == "users"
        assert fk["to_column"] == "id"
        assert fk["on_update"] == "NO ACTION"
        assert fk["on_delete"] == "NO ACTION"

    @pytest.mark.asyncio
    async def test_no_foreign_keys_returns_empty_list(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = DatabricksProfiler(mock_db, ProfilingConfig())

        assert await profiler._fetch_foreign_keys("default", "standalone") == []


# ── DatabricksProfilerService._assemble_response ────────────────────────────


class TestDatabricksProfilerServiceAssemble:
    _RAW_EMPTY = {
        "database": {
            "name": "my_catalog",
            "version": "",
            "encoding": "UTF-8",
            "size_bytes": 0,
        },
        "schemas": [],
    }

    _RAW_FULL = {
        "database": {
            "name": "my_catalog",
            "version": "",
            "encoding": "UTF-8",
            "size_bytes": 0,
        },
        "schemas": [
            {
                "name": "default",
                "owner": "admin",
                "tables": [
                    {
                        "name": "users",
                        "schema": "default",
                        "owner": "admin",
                        "description": "User accounts",
                        "size_bytes": None,
                        "total_size_bytes": None,
                        "row_count": 250,
                        "columns": [
                            {
                                "name": "id",
                                "ordinal_position": 1,
                                "data_type": "bigint",
                                "is_nullable": False,
                                "column_default": None,
                                "character_maximum_length": None,
                                "numeric_precision": 19,
                                "numeric_scale": 0,
                                "is_primary_key": True,
                                "description": None,
                                "enum_values": None,
                                "sample_values": None,
                            },
                            {
                                "name": "status",
                                "ordinal_position": 2,
                                "data_type": "string",
                                "is_nullable": True,
                                "column_default": None,
                                "character_maximum_length": None,
                                "numeric_precision": None,
                                "numeric_scale": None,
                                "is_primary_key": False,
                                "description": "Account status",
                                "enum_values": None,
                                "sample_values": ["active", "active", "inactive"],
                            },
                        ],
                        "indexes": [],
                        "relationships": [
                            {
                                "constraint_name": "fk_users_group",
                                "from_column": "group_id",
                                "to_schema": "default",
                                "to_table": "groups",
                                "to_column": "id",
                                "on_update": "NO ACTION",
                                "on_delete": "NO ACTION",
                            }
                        ],
                        "data_freshness": {
                            "last_analyze": datetime(2024, 3, 12, tzinfo=UTC),
                            "last_autoanalyze": None,
                            "last_vacuum": datetime(2024, 1, 1, tzinfo=UTC),
                            "last_autovacuum": None,
                        },
                    }
                ],
                "views": [
                    {
                        "name": "active_users",
                        "schema": "default",
                        "owner": "admin",
                        "definition": "SELECT * FROM users WHERE status = 'active'",
                    }
                ],
            }
        ],
    }

    def test_assembles_empty_schema_list(self):
        svc = DatabricksProfilerService()
        now = datetime.now(tz=UTC)
        result = svc._assemble_response(self._RAW_EMPTY, now)

        assert isinstance(result, ProfilingResponse)
        assert result.database.name == "my_catalog"
        assert result.database.version == ""
        assert result.database.encoding == "UTF-8"
        assert result.database.size_bytes == 0
        assert result.schemas == []
        assert result.profiled_at == now

    def test_assembles_schema_with_table_and_view(self):
        svc = DatabricksProfilerService()
        result = svc._assemble_response(self._RAW_FULL, datetime.now(tz=UTC))

        assert len(result.schemas) == 1
        schema = result.schemas[0]
        assert schema.name == "default"
        assert schema.owner == "admin"
        assert len(schema.tables) == 1
        assert len(schema.views) == 1

    def test_table_metadata_fields(self):
        svc = DatabricksProfilerService()
        result = svc._assemble_response(self._RAW_FULL, datetime.now(tz=UTC))

        table = result.schemas[0].tables[0]
        assert table.name == "users"
        assert table.row_count == 250
        assert table.size_bytes is None
        assert table.description == "User accounts"

    def test_column_metadata(self):
        svc = DatabricksProfilerService()
        result = svc._assemble_response(self._RAW_FULL, datetime.now(tz=UTC))

        columns = result.schemas[0].tables[0].columns
        assert len(columns) == 2

        pk_col = columns[0]
        assert pk_col.name == "id"
        assert pk_col.is_primary_key is True
        assert pk_col.enum_values is None

        status_col = columns[1]
        assert status_col.name == "status"
        assert status_col.data_type == "string"
        assert status_col.sample_values == ["active", "active", "inactive"]

    def test_indexes_empty_list(self):
        svc = DatabricksProfilerService()
        result = svc._assemble_response(self._RAW_FULL, datetime.now(tz=UTC))

        indexes = result.schemas[0].tables[0].indexes
        assert indexes is not None
        assert len(indexes) == 0

    def test_relationship_metadata(self):
        svc = DatabricksProfilerService()
        result = svc._assemble_response(self._RAW_FULL, datetime.now(tz=UTC))

        rels = result.schemas[0].tables[0].relationships
        assert rels is not None
        assert len(rels) == 1
        rel = rels[0]
        assert rel.constraint_name == "fk_users_group"
        assert rel.from_column == "group_id"
        assert rel.to_table == "groups"

    def test_data_freshness_mapping(self):
        svc = DatabricksProfilerService()
        result = svc._assemble_response(self._RAW_FULL, datetime.now(tz=UTC))

        freshness = result.schemas[0].tables[0].data_freshness
        assert freshness is not None
        assert freshness.last_analyze == datetime(2024, 3, 12, tzinfo=UTC)
        assert freshness.last_vacuum == datetime(2024, 1, 1, tzinfo=UTC)
        assert freshness.last_autoanalyze is None
        assert freshness.last_autovacuum is None

    def test_view_metadata(self):
        svc = DatabricksProfilerService()
        result = svc._assemble_response(self._RAW_FULL, datetime.now(tz=UTC))

        view = result.schemas[0].views[0]
        assert view.name == "active_users"
        assert view.owner == "admin"
        assert "SELECT" in view.definition


# ── DatabricksProfilerService metadata ──────────────────────────────────────────


class TestDatabricksProfilerServiceMetadata:
    def test_span_attributes(self):
        from ignite_data_connectors import DatabricksConfig

        svc = DatabricksProfilerService()
        conn = DatabricksConfig(
            server_hostname="adb-123.azuredatabricks.net",
            http_path="/sql/1.0/warehouses/abc",
            access_token="dapi123",
            catalog="main",
        )
        attrs = svc._span_attributes(conn)

        assert attrs["db.system"] == "databricks"
        assert attrs["db.name"] == "main"
        assert attrs["databricks.server_hostname"] == "adb-123.azuredatabricks.net"
        assert attrs["databricks.http_path"] == "/sql/1.0/warehouses/abc"

    def test_span_attributes_without_catalog(self):
        from ignite_data_connectors import DatabricksConfig

        svc = DatabricksProfilerService()
        conn = DatabricksConfig(
            server_hostname="adb-123.azuredatabricks.net",
            http_path="/sql/1.0/warehouses/abc",
            access_token="dapi123",
        )
        attrs = svc._span_attributes(conn)

        assert attrs["db.name"] == ""

    def test_log_context(self):
        from ignite_data_connectors import DatabricksConfig

        svc = DatabricksProfilerService()
        conn = DatabricksConfig(
            server_hostname="adb-123.azuredatabricks.net",
            http_path="/sql/1.0/warehouses/abc",
            access_token="dapi123",
            catalog="main",
        )
        ctx = svc._log_context(conn)

        assert ctx["host"] == "adb-123.azuredatabricks.net"
        assert ctx["database"] == "main"
        assert ctx["http_path"] == "/sql/1.0/warehouses/abc"

    def test_service_name_and_span_name(self):
        svc = DatabricksProfilerService()
        assert svc.service_name == "Databricks"
        assert svc.span_name == "profiler.databricks"
