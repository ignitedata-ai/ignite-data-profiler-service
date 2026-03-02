"""Unit tests for the Snowflake profiler module.

Covers:
- ``_make_json_safe`` — JSON serialisation helper
- ``SnowflakeProfiler._filter_tables`` — include/exclude filtering
- ``SnowflakeProfiler`` async fetch methods (mocked connector)
- ``SnowflakeProfilerService._assemble_response`` — raw-dict → ProfilingResponse
- ``SnowflakeProfilerService._span_attributes`` / ``_log_context``
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.api.v1.schemas.profiler import ProfilingConfig, ProfilingResponse
from core.services.snowflake.profiler import (
    SnowflakeProfiler,
    SnowflakeProfilerService,
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


# ── SnowflakeProfiler — filter logic ─────────────────────────────────────────


class TestSnowflakeProfilerFilters:
    def _make_profiler(self, config: ProfilingConfig | None = None) -> SnowflakeProfiler:
        return SnowflakeProfiler(MagicMock(), config or ProfilingConfig())

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


# ── SnowflakeProfiler — async fetch methods ─────────────────────────────────


class TestSnowflakeProfilerSchemaDiscovery:
    @pytest.mark.asyncio
    async def test_information_schema_always_excluded(self):
        """INFORMATION_SCHEMA must be excluded even when exclude_schemas=[]."""
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [{"schema_name": "PUBLIC"}]
        profiler = SnowflakeProfiler(mock_db, ProfilingConfig(exclude_schemas=[]))

        await profiler._fetch_schema_names()

        params = mock_db.fetch_all.call_args[0][1]
        assert "INFORMATION_SCHEMA" in params

    @pytest.mark.asyncio
    async def test_config_exclude_merged_with_system(self):
        """User-specified exclude_schemas should be combined with system exclusions."""
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [{"schema_name": "PUBLIC"}]
        config = ProfilingConfig(exclude_schemas=["STAGING", "DEV"])
        profiler = SnowflakeProfiler(mock_db, config)

        await profiler._fetch_schema_names()

        params = mock_db.fetch_all.call_args[0][1]
        assert "STAGING" in params
        assert "DEV" in params
        assert "INFORMATION_SCHEMA" in params

    @pytest.mark.asyncio
    async def test_include_schemas_filter_applied(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [
            {"schema_name": "PUBLIC"},
            {"schema_name": "ANALYTICS"},
            {"schema_name": "RAW"},
        ]
        config = ProfilingConfig(include_schemas=["PUBLIC", "RAW"], exclude_schemas=[])
        profiler = SnowflakeProfiler(mock_db, config)

        result = await profiler._fetch_schema_names()

        assert result == ["PUBLIC", "RAW"]

    @pytest.mark.asyncio
    async def test_placeholder_count_matches_exclude_list_length(self):
        """The SQL query must have the correct number of %s placeholders."""
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        config = ProfilingConfig(exclude_schemas=["extra1", "extra2"])
        profiler = SnowflakeProfiler(mock_db, config)

        await profiler._fetch_schema_names()

        query: str = mock_db.fetch_all.call_args[0][0]
        params: tuple = mock_db.fetch_all.call_args[0][1]
        assert query.count("%s") == len(params)


class TestSnowflakeProfilerColumnFetching:
    @pytest.mark.asyncio
    async def test_primary_key_column_flagged(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.side_effect = [
            [
                {
                    "column_name": "id",
                    "ordinal_position": 1,
                    "data_type": "NUMBER",
                    "is_nullable": False,
                    "column_default": None,
                    "character_maximum_length": None,
                    "numeric_precision": 38,
                    "numeric_scale": 0,
                    "description": None,
                }
            ],
            [{"column_name": "id"}],  # PK
        ]
        profiler = SnowflakeProfiler(mock_db, ProfilingConfig())

        columns = await profiler._fetch_columns("PUBLIC", "USERS")

        assert columns[0]["is_primary_key"] is True
        assert columns[0]["enum_values"] is None

    @pytest.mark.asyncio
    async def test_enum_values_always_none(self):
        """Snowflake does not have native enum types."""
        mock_db = AsyncMock()
        mock_db.fetch_all.side_effect = [
            [
                {
                    "column_name": "status",
                    "ordinal_position": 1,
                    "data_type": "VARCHAR",
                    "is_nullable": True,
                    "column_default": None,
                    "character_maximum_length": 100,
                    "numeric_precision": None,
                    "numeric_scale": None,
                    "description": "Account status",
                }
            ],
            [],  # no PK
        ]
        profiler = SnowflakeProfiler(mock_db, ProfilingConfig())

        columns = await profiler._fetch_columns("PUBLIC", "USERS")

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
                    "data_type": "NUMBER",
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
        profiler = SnowflakeProfiler(mock_db, ProfilingConfig())

        columns = await profiler._fetch_columns("PUBLIC", "T")

        assert columns[0]["description"] is None


class TestSnowflakeProfilerIndexFetching:
    @pytest.mark.asyncio
    async def test_always_returns_empty_list(self):
        """Snowflake does not have traditional indexes."""
        profiler = SnowflakeProfiler(MagicMock(), ProfilingConfig())
        indexes = await profiler._fetch_indexes()
        assert indexes == []


class TestSnowflakeProfilerRowCount:
    @pytest.mark.asyncio
    async def test_returns_estimate_from_information_schema(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {"row_count": 500}
        profiler = SnowflakeProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_row_count("PUBLIC", "ORDERS")

        assert result == 500
        mock_db.fetch_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_exact_count_when_null(self):
        """ROW_COUNT = NULL triggers an exact COUNT(1)."""
        mock_db = AsyncMock()
        mock_db.fetch_one.side_effect = [
            {"row_count": None},
            {"row_count": 42},
        ]
        profiler = SnowflakeProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_row_count("PUBLIC", "ORDERS")

        assert result == 42
        assert mock_db.fetch_one.call_count == 2

    @pytest.mark.asyncio
    async def test_zero_rows_is_valid_and_returned_directly(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {"row_count": 0}
        profiler = SnowflakeProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_row_count("PUBLIC", "EMPTY_TABLE")

        assert result == 0
        mock_db.fetch_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_table_not_found(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = None
        profiler = SnowflakeProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_row_count("PUBLIC", "GHOST")

        assert result is None


class TestSnowflakeProfilerDataFreshness:
    @pytest.mark.asyncio
    async def test_maps_last_altered_to_last_analyze(self):
        altered_dt = datetime(2024, 3, 12, tzinfo=UTC)
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {
            "last_ddl": None,
            "last_altered": altered_dt,
            "created": datetime(2024, 1, 1, tzinfo=UTC),
        }
        profiler = SnowflakeProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_data_freshness("PUBLIC", "ORDERS")

        assert result["last_analyze"] == altered_dt

    @pytest.mark.asyncio
    async def test_maps_last_ddl_to_last_vacuum(self):
        ddl_dt = datetime(2024, 3, 10, tzinfo=UTC)
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {
            "last_ddl": ddl_dt,
            "last_altered": None,
            "created": datetime(2024, 1, 1, tzinfo=UTC),
        }
        profiler = SnowflakeProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_data_freshness("PUBLIC", "ORDERS")

        assert result["last_vacuum"] == ddl_dt

    @pytest.mark.asyncio
    async def test_auto_fields_always_none(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {
            "last_ddl": datetime(2024, 1, 1, tzinfo=UTC),
            "last_altered": datetime(2024, 1, 2, tzinfo=UTC),
            "created": datetime(2024, 1, 1, tzinfo=UTC),
        }
        profiler = SnowflakeProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_data_freshness("PUBLIC", "T")

        assert result["last_autoanalyze"] is None
        assert result["last_autovacuum"] is None

    @pytest.mark.asyncio
    async def test_returns_none_when_row_missing(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = None
        profiler = SnowflakeProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_data_freshness("PUBLIC", "GHOST")

        assert result is None


class TestSnowflakeProfilerSampleData:
    @pytest.mark.asyncio
    async def test_uses_double_quote_quoting(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = SnowflakeProfiler(mock_db, ProfilingConfig(sample_size=5))

        await profiler._fetch_sample_data("PUBLIC", "USERS")

        query: str = mock_db.fetch_all.call_args[0][0]
        assert '"PUBLIC"' in query
        assert '"USERS"' in query
        assert "RANDOM()" in query
        assert "LIMIT" in query

    @pytest.mark.asyncio
    async def test_escapes_double_quotes_in_identifiers(self):
        """A double-quote in the identifier name must be doubled to prevent injection."""
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = SnowflakeProfiler(mock_db, ProfilingConfig(sample_size=3))

        await profiler._fetch_sample_data('my"schema', 't"able')

        query: str = mock_db.fetch_all.call_args[0][0]
        assert '"my""schema"' in query
        assert '"t""able"' in query

    @pytest.mark.asyncio
    async def test_sample_size_passed_as_param(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = SnowflakeProfiler(mock_db, ProfilingConfig(sample_size=25))

        await profiler._fetch_sample_data("PUBLIC", "T")

        params = mock_db.fetch_all.call_args[0][1]
        assert params == (25,)


class TestSnowflakeProfilerViews:
    @pytest.mark.asyncio
    async def test_view_fields_mapped_correctly(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [
            {
                "table_name": "ACTIVE_USERS",
                "table_schema": "PUBLIC",
                "owner": "SYSADMIN",
                "definition": "SELECT * FROM USERS WHERE STATUS = 'active'",
            }
        ]
        profiler = SnowflakeProfiler(mock_db, ProfilingConfig())

        views = await profiler._fetch_views("PUBLIC")

        assert len(views) == 1
        v = views[0]
        assert v["name"] == "ACTIVE_USERS"
        assert v["schema"] == "PUBLIC"
        assert v["owner"] == "SYSADMIN"
        assert "SELECT" in v["definition"]

    @pytest.mark.asyncio
    async def test_no_views_returns_empty_list(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = SnowflakeProfiler(mock_db, ProfilingConfig())

        assert await profiler._fetch_views("PUBLIC") == []


class TestSnowflakeProfilerForeignKeys:
    @pytest.mark.asyncio
    async def test_foreign_key_fields_mapped_correctly(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [
            {
                "constraint_name": "FK_ORDER_USER",
                "column_name": "USER_ID",
                "foreign_table_schema": "PUBLIC",
                "foreign_table_name": "USERS",
                "foreign_column_name": "ID",
                "update_rule": "NO ACTION",
                "delete_rule": "NO ACTION",
            }
        ]
        profiler = SnowflakeProfiler(mock_db, ProfilingConfig())

        fks = await profiler._fetch_foreign_keys("PUBLIC", "ORDERS")

        assert len(fks) == 1
        fk = fks[0]
        assert fk["constraint_name"] == "FK_ORDER_USER"
        assert fk["from_column"] == "USER_ID"
        assert fk["to_schema"] == "PUBLIC"
        assert fk["to_table"] == "USERS"
        assert fk["to_column"] == "ID"
        assert fk["on_update"] == "NO ACTION"
        assert fk["on_delete"] == "NO ACTION"

    @pytest.mark.asyncio
    async def test_no_foreign_keys_returns_empty_list(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = SnowflakeProfiler(mock_db, ProfilingConfig())

        assert await profiler._fetch_foreign_keys("PUBLIC", "STANDALONE") == []


# ── SnowflakeProfilerService._assemble_response ────────────────────────────


class TestSnowflakeProfilerServiceAssemble:
    _RAW_EMPTY = {
        "database": {
            "name": "TESTDB",
            "version": "8.8.2",
            "encoding": "UTF-8",
            "size_bytes": 1024,
        },
        "schemas": [],
    }

    _RAW_FULL = {
        "database": {
            "name": "TESTDB",
            "version": "8.8.2",
            "encoding": "UTF-8",
            "size_bytes": 8192,
        },
        "schemas": [
            {
                "name": "PUBLIC",
                "owner": "SYSADMIN",
                "tables": [
                    {
                        "name": "USERS",
                        "schema": "PUBLIC",
                        "owner": "SYSADMIN",
                        "description": "User accounts",
                        "size_bytes": 1024,
                        "total_size_bytes": 2048,
                        "row_count": 250,
                        "columns": [
                            {
                                "name": "ID",
                                "ordinal_position": 1,
                                "data_type": "NUMBER",
                                "is_nullable": False,
                                "column_default": None,
                                "character_maximum_length": None,
                                "numeric_precision": 38,
                                "numeric_scale": 0,
                                "is_primary_key": True,
                                "description": None,
                                "enum_values": None,
                                "sample_values": None,
                            },
                            {
                                "name": "STATUS",
                                "ordinal_position": 2,
                                "data_type": "VARCHAR",
                                "is_nullable": True,
                                "column_default": "'active'",
                                "character_maximum_length": 100,
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
                                "constraint_name": "FK_USERS_GROUP",
                                "from_column": "GROUP_ID",
                                "to_schema": "PUBLIC",
                                "to_table": "GROUPS",
                                "to_column": "ID",
                                "on_update": "NO ACTION",
                                "on_delete": "NO ACTION",
                            }
                        ],
                        "data_freshness": {
                            "last_analyze": datetime(2024, 3, 12, tzinfo=UTC),
                            "last_autoanalyze": None,
                            "last_vacuum": datetime(2024, 3, 10, tzinfo=UTC),
                            "last_autovacuum": None,
                        },
                    }
                ],
                "views": [
                    {
                        "name": "ACTIVE_USERS",
                        "schema": "PUBLIC",
                        "owner": "SYSADMIN",
                        "definition": "SELECT * FROM USERS WHERE STATUS = 'active'",
                    }
                ],
            }
        ],
    }

    def test_assembles_empty_schema_list(self):
        svc = SnowflakeProfilerService()
        now = datetime.now(tz=UTC)
        result = svc._assemble_response(self._RAW_EMPTY, now)

        assert isinstance(result, ProfilingResponse)
        assert result.database.name == "TESTDB"
        assert result.database.version == "8.8.2"
        assert result.database.encoding == "UTF-8"
        assert result.database.size_bytes == 1024
        assert result.schemas == []
        assert result.profiled_at == now

    def test_assembles_schema_with_table_and_view(self):
        svc = SnowflakeProfilerService()
        result = svc._assemble_response(self._RAW_FULL, datetime.now(tz=UTC))

        assert len(result.schemas) == 1
        schema = result.schemas[0]
        assert schema.name == "PUBLIC"
        assert schema.owner == "SYSADMIN"
        assert len(schema.tables) == 1
        assert len(schema.views) == 1

    def test_table_metadata_fields(self):
        svc = SnowflakeProfilerService()
        result = svc._assemble_response(
            self._RAW_FULL,
            datetime.now(tz=UTC),
        )

        table = result.schemas[0].tables[0]
        assert table.name == "USERS"
        assert table.row_count == 250
        assert table.size_bytes == 1024
        assert table.total_size_bytes == 2048
        assert table.description == "User accounts"

    def test_column_metadata(self):
        svc = SnowflakeProfilerService()
        result = svc._assemble_response(
            self._RAW_FULL,
            datetime.now(tz=UTC),
        )

        columns = result.schemas[0].tables[0].columns
        assert len(columns) == 2

        pk_col = columns[0]
        assert pk_col.name == "ID"
        assert pk_col.is_primary_key is True
        assert pk_col.enum_values is None

        status_col = columns[1]
        assert status_col.name == "STATUS"
        assert status_col.data_type == "VARCHAR"
        assert status_col.enum_values is None
        assert status_col.sample_values == ["active", "active", "inactive"]

    def test_indexes_empty_list(self):
        svc = SnowflakeProfilerService()
        result = svc._assemble_response(
            self._RAW_FULL,
            datetime.now(tz=UTC),
        )

        indexes = result.schemas[0].tables[0].indexes
        assert indexes is not None
        assert len(indexes) == 0

    def test_relationship_metadata(self):
        svc = SnowflakeProfilerService()
        result = svc._assemble_response(
            self._RAW_FULL,
            datetime.now(tz=UTC),
        )

        rels = result.schemas[0].tables[0].relationships
        assert rels is not None
        assert len(rels) == 1
        rel = rels[0]
        assert rel.constraint_name == "FK_USERS_GROUP"
        assert rel.from_column == "GROUP_ID"
        assert rel.to_table == "GROUPS"
        assert rel.on_update == "NO ACTION"
        assert rel.on_delete == "NO ACTION"

    def test_data_freshness_mapping(self):
        svc = SnowflakeProfilerService()
        result = svc._assemble_response(
            self._RAW_FULL,
            datetime.now(tz=UTC),
        )

        freshness = result.schemas[0].tables[0].data_freshness
        assert freshness is not None
        assert freshness.last_analyze == datetime(2024, 3, 12, tzinfo=UTC)
        assert freshness.last_vacuum == datetime(2024, 3, 10, tzinfo=UTC)
        assert freshness.last_autoanalyze is None
        assert freshness.last_autovacuum is None

    def test_view_metadata(self):
        svc = SnowflakeProfilerService()
        result = svc._assemble_response(
            self._RAW_FULL,
            datetime.now(tz=UTC),
        )

        view = result.schemas[0].views[0]
        assert view.name == "ACTIVE_USERS"
        assert view.owner == "SYSADMIN"
        assert "SELECT" in view.definition

    def test_indexes_none_when_not_requested(self):
        raw = {
            "database": {"name": "DB", "version": "8.0", "encoding": "UTF-8", "size_bytes": 0},
            "schemas": [
                {
                    "name": "PUBLIC",
                    "owner": "",
                    "tables": [
                        {
                            "name": "T",
                            "schema": "PUBLIC",
                            "owner": "",
                            "description": None,
                            "size_bytes": None,
                            "total_size_bytes": None,
                            "row_count": None,
                            "columns": [],
                            "indexes": None,
                            "relationships": None,
                            "data_freshness": None,
                        }
                    ],
                    "views": [],
                }
            ],
        }
        svc = SnowflakeProfilerService()
        result = svc._assemble_response(
            raw,
            datetime.now(tz=UTC),
        )

        table = result.schemas[0].tables[0]
        assert table.indexes is None
        assert table.relationships is None


# ── SnowflakeProfilerService metadata ──────────────────────────────────────────


class TestSnowflakeProfilerServiceMetadata:
    def test_span_attributes(self):
        from ignite_data_connectors import SnowflakeConfig

        svc = SnowflakeProfilerService()
        conn = SnowflakeConfig(
            account="xy12345.us-east-1",
            database="PROD",
            username="u",
            password="p",
            warehouse="COMPUTE_WH",
        )
        attrs = svc._span_attributes(conn)

        assert attrs["db.system"] == "snowflake"
        assert attrs["db.name"] == "PROD"
        assert attrs["snowflake.account"] == "xy12345.us-east-1"
        assert attrs["snowflake.warehouse"] == "COMPUTE_WH"

    def test_span_attributes_without_warehouse(self):
        from ignite_data_connectors import SnowflakeConfig

        svc = SnowflakeProfilerService()
        conn = SnowflakeConfig(
            account="xy12345.us-east-1",
            database="PROD",
            username="u",
            password="p",
        )
        attrs = svc._span_attributes(conn)

        assert "snowflake.warehouse" not in attrs

    def test_log_context(self):
        from ignite_data_connectors import SnowflakeConfig

        svc = SnowflakeProfilerService()
        conn = SnowflakeConfig(
            account="xy12345.us-east-1",
            database="PROD",
            username="admin",
            password="p",
            warehouse="COMPUTE_WH",
            role="ANALYST",
        )
        ctx = svc._log_context(conn)

        assert ctx["host"] == "xy12345.us-east-1"
        assert ctx["database"] == "PROD"
        assert ctx["warehouse"] == "COMPUTE_WH"
        assert ctx["role"] == "ANALYST"
        assert ctx["username"] == "admin"

    def test_service_name_and_span_name(self):
        svc = SnowflakeProfilerService()
        assert svc.service_name == "Snowflake"
        assert svc.span_name == "profiler.snowflake"
