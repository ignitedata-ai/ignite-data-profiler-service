"""Unit tests for the MySQL profiler module.

Covers:
- ``_parse_enum_values`` — pure enum-string parser
- ``_make_json_safe`` — JSON serialisation helper
- ``MySQLProfiler._filter_tables`` — include/exclude filtering
- ``MySQLProfiler`` async fetch methods (mocked connector)
- ``MySQLProfilerService._assemble_response`` — raw-dict → ProfilingResponse
- ``MySQLProfilerService._span_attributes`` / ``_log_context``
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.api.v1.schemas.profiler import ProfilingConfig, ProfilingResponse
from core.services.mysql.profiler import (
    MySQLProfiler,
    MySQLProfilerService,
    _make_json_safe,
    _parse_enum_values,
)

# ── _parse_enum_values ─────────────────────────────────────────────────────────


class TestParseEnumValues:
    def test_standard_three_values(self):
        result = _parse_enum_values("enum('active','inactive','pending')")
        assert result == ["active", "inactive", "pending"]

    def test_single_value(self):
        assert _parse_enum_values("enum('only')") == ["only"]

    def test_two_values(self):
        assert _parse_enum_values("enum('yes','no')") == ["yes", "no"]

    def test_empty_string_value_included(self):
        result = _parse_enum_values("enum('','active')")
        assert result == ["", "active"]

    def test_case_insensitive_keyword(self):
        result = _parse_enum_values("ENUM('a','b')")
        assert result == ["a", "b"]

    def test_values_with_spaces(self):
        result = _parse_enum_values("enum('hello world','foo bar')")
        assert result == ["hello world", "foo bar"]

    def test_values_with_hyphens(self):
        result = _parse_enum_values("enum('in-progress','done')")
        assert result == ["in-progress", "done"]

    def test_non_enum_varchar_returns_empty(self):
        assert _parse_enum_values("varchar(255)") == []

    def test_non_enum_int_returns_empty(self):
        assert _parse_enum_values("int") == []

    def test_empty_string_returns_empty(self):
        assert _parse_enum_values("") == []


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


# ── MySQLProfiler — filter logic ───────────────────────────────────────────────


class TestMySQLProfilerFilters:
    def _make_profiler(self, config: ProfilingConfig | None = None) -> MySQLProfiler:
        return MySQLProfiler(MagicMock(), config or ProfilingConfig())

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


# ── MySQLProfiler — async fetch methods ───────────────────────────────────────


class TestMySQLProfilerSchemaDiscovery:
    @pytest.mark.asyncio
    async def test_system_databases_always_excluded(self):
        """MySQL system databases must be excluded even when exclude_schemas=[]."""
        mock_db = AsyncMock()
        profiler = MySQLProfiler(
            mock_db,
            ProfilingConfig(
                include_schemas=["mydb", "mysql", "information_schema", "performance_schema", "sys"],
                exclude_schemas=[],
            ),
        )

        result = await profiler._fetch_schema_names()

        assert result == ["mydb"]
        for system_db in ("mysql", "information_schema", "performance_schema", "sys"):
            assert system_db not in result, f"{system_db!r} should be excluded"

    @pytest.mark.asyncio
    async def test_config_exclude_merged_with_system(self):
        """User-specified exclude_schemas should be combined with system exclusions."""
        mock_db = AsyncMock()
        config = ProfilingConfig(
            include_schemas=["mydb", "staging", "dev", "mysql"],
            exclude_schemas=["staging", "dev"],
        )
        profiler = MySQLProfiler(mock_db, config)

        result = await profiler._fetch_schema_names()

        assert result == ["mydb"]
        assert "staging" not in result
        assert "dev" not in result
        assert "mysql" not in result

    @pytest.mark.asyncio
    async def test_include_schemas_filter_applied(self):
        mock_db = AsyncMock()
        config = ProfilingConfig(include_schemas=["db1", "db3"], exclude_schemas=[])
        profiler = MySQLProfiler(mock_db, config)

        result = await profiler._fetch_schema_names()

        assert result == ["db1", "db3"]

    @pytest.mark.asyncio
    async def test_placeholder_count_matches_exclude_list_length(self):
        """User exclude_schemas and system schemas are both applied."""
        mock_db = AsyncMock()
        config = ProfilingConfig(
            include_schemas=["a", "b", "mysql", "sys", "extra1"],
            exclude_schemas=["extra1", "extra2"],
        )
        profiler = MySQLProfiler(mock_db, config)

        result = await profiler._fetch_schema_names()

        # "a" and "b" pass; "mysql"/"sys" are system; "extra1" is user-excluded
        assert result == ["a", "b"]


class TestMySQLProfilerColumnFetching:
    @pytest.mark.asyncio
    async def test_enum_column_values_parsed_inline(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.side_effect = [
            # columns
            [
                {
                    "column_name": "status",
                    "ordinal_position": 1,
                    "data_type": "enum",
                    "is_nullable": True,
                    "column_default": "active",
                    "character_maximum_length": None,
                    "numeric_precision": None,
                    "numeric_scale": None,
                    "description": None,
                    "full_column_type": "enum('active','inactive','pending')",
                }
            ],
            # primary keys
            [],
        ]
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        columns = await profiler._fetch_columns("mydb", "orders")

        assert len(columns) == 1
        col = columns[0]
        assert col["data_type"] == "enum"
        assert col["enum_values"] == ["active", "inactive", "pending"]
        assert col["is_primary_key"] is False

    @pytest.mark.asyncio
    async def test_primary_key_column_flagged(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.side_effect = [
            [
                {
                    "column_name": "id",
                    "ordinal_position": 1,
                    "data_type": "int",
                    "is_nullable": False,
                    "column_default": None,
                    "character_maximum_length": None,
                    "numeric_precision": 10,
                    "numeric_scale": 0,
                    "description": None,
                    "full_column_type": "int",
                }
            ],
            [{"column_name": "id"}],  # PK
        ]
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        columns = await profiler._fetch_columns("mydb", "users")

        assert columns[0]["is_primary_key"] is True
        assert columns[0]["enum_values"] is None

    @pytest.mark.asyncio
    async def test_non_enum_column_has_no_enum_values(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.side_effect = [
            [
                {
                    "column_name": "name",
                    "ordinal_position": 1,
                    "data_type": "varchar",
                    "is_nullable": True,
                    "column_default": None,
                    "character_maximum_length": 255,
                    "numeric_precision": None,
                    "numeric_scale": None,
                    "description": "User name",
                    "full_column_type": "varchar(255)",
                }
            ],
            [],
        ]
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        columns = await profiler._fetch_columns("mydb", "users")

        assert columns[0]["enum_values"] is None

    @pytest.mark.asyncio
    async def test_empty_column_comment_normalised_to_none(self):
        """An empty string description (from MySQL column_comment) should become None."""
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
                    "description": "",  # empty string from MySQL
                    "full_column_type": "int",
                }
            ],
            [],
        ]
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        columns = await profiler._fetch_columns("mydb", "t")

        assert columns[0]["description"] is None


class TestMySQLProfilerIndexFetching:
    @pytest.mark.asyncio
    async def test_columns_str_split_correctly(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [
            {
                "index_name": "idx_name_email",
                "is_unique": 0,
                "is_primary": 0,
                "index_type": "BTREE",
                "columns_str": "last_name,first_name,email",
            },
            {
                "index_name": "PRIMARY",
                "is_unique": 1,
                "is_primary": 1,
                "index_type": "BTREE",
                "columns_str": "id",
            },
        ]
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        indexes = await profiler._fetch_indexes("mydb", "users")

        assert len(indexes) == 2
        assert indexes[0]["columns"] == ["last_name", "first_name", "email"]
        assert indexes[0]["is_unique"] is False
        assert indexes[0]["is_primary"] is False
        assert indexes[1]["columns"] == ["id"]
        assert indexes[1]["is_unique"] is True
        assert indexes[1]["is_primary"] is True
        assert indexes[1]["index_type"] == "BTREE"

    @pytest.mark.asyncio
    async def test_null_columns_str_returns_empty_list(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [
            {
                "index_name": "idx_x",
                "is_unique": 0,
                "is_primary": 0,
                "index_type": "BTREE",
                "columns_str": None,
            }
        ]
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        indexes = await profiler._fetch_indexes("mydb", "t")

        assert indexes[0]["columns"] == []

    @pytest.mark.asyncio
    async def test_empty_table_has_no_indexes(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        indexes = await profiler._fetch_indexes("mydb", "t")

        assert indexes == []


class TestMySQLProfilerRowCount:
    @pytest.mark.asyncio
    async def test_returns_estimate_from_information_schema(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {"row_count": 500}
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_row_count("mydb", "orders")

        assert result == 500
        mock_db.fetch_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_exact_count_when_null(self):
        """table_rows = NULL in information_schema triggers an exact COUNT(1)."""
        mock_db = AsyncMock()
        mock_db.fetch_one.side_effect = [
            {"row_count": None},  # fast estimate is NULL
            {"row_count": 42},  # exact COUNT
        ]
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_row_count("mydb", "orders")

        assert result == 42
        assert mock_db.fetch_one.call_count == 2

    @pytest.mark.asyncio
    async def test_zero_rows_is_valid_and_returned_directly(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {"row_count": 0}
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_row_count("mydb", "empty_table")

        assert result == 0
        mock_db.fetch_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_table_not_found(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = None
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_row_count("mydb", "ghost")

        assert result is None


class TestMySQLProfilerDataFreshness:
    @pytest.mark.asyncio
    async def test_maps_check_time_to_last_analyze(self):
        check_dt = datetime(2024, 3, 12, tzinfo=UTC)
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {
            "update_time": None,
            "check_time": check_dt,
            "create_time": datetime(2024, 1, 1, tzinfo=UTC),
        }
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_data_freshness("mydb", "orders")

        assert result["last_analyze"] == check_dt

    @pytest.mark.asyncio
    async def test_maps_update_time_to_last_vacuum(self):
        update_dt = datetime(2024, 3, 10, tzinfo=UTC)
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {
            "update_time": update_dt,
            "check_time": None,
            "create_time": datetime(2024, 1, 1, tzinfo=UTC),
        }
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_data_freshness("mydb", "orders")

        assert result["last_vacuum"] == update_dt

    @pytest.mark.asyncio
    async def test_auto_fields_always_none(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {
            "update_time": datetime(2024, 1, 1, tzinfo=UTC),
            "check_time": datetime(2024, 1, 2, tzinfo=UTC),
            "create_time": datetime(2024, 1, 1, tzinfo=UTC),
        }
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_data_freshness("mydb", "t")

        assert result["last_autoanalyze"] is None
        assert result["last_autovacuum"] is None

    @pytest.mark.asyncio
    async def test_returns_none_when_row_missing(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = None
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_data_freshness("mydb", "ghost")

        assert result is None

    @pytest.mark.asyncio
    async def test_nullable_timestamps_propagated(self):
        """Tables with no DML have NULL update_time and check_time."""
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {
            "update_time": None,
            "check_time": None,
            "create_time": datetime(2024, 1, 1, tzinfo=UTC),
        }
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_data_freshness("mydb", "fresh_table")

        assert result["last_analyze"] is None
        assert result["last_vacuum"] is None


class TestMySQLProfilerSampleData:
    @pytest.mark.asyncio
    async def test_uses_backtick_quoting(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = MySQLProfiler(mock_db, ProfilingConfig(sample_size=5))

        await profiler._fetch_sample_data("mydb", "users")

        query: str = mock_db.fetch_all.call_args[0][0]
        assert "`mydb`" in query
        assert "`users`" in query
        assert "RAND()" in query
        assert "LIMIT" in query

    @pytest.mark.asyncio
    async def test_escapes_backticks_in_identifiers(self):
        """A backtick in the identifier name must be doubled to prevent injection."""
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = MySQLProfiler(mock_db, ProfilingConfig(sample_size=3))

        await profiler._fetch_sample_data("my`db", "t`able")

        query: str = mock_db.fetch_all.call_args[0][0]
        assert "`my``db`" in query
        assert "`t``able`" in query

    @pytest.mark.asyncio
    async def test_sample_size_passed_as_param(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = MySQLProfiler(mock_db, ProfilingConfig(sample_size=25))

        await profiler._fetch_sample_data("mydb", "t")

        params = mock_db.fetch_all.call_args[0][1]
        assert params == (25,)


class TestMySQLProfilerViews:
    @pytest.mark.asyncio
    async def test_view_fields_mapped_correctly(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [
            {
                "table_name": "active_users",
                "table_schema": "mydb",
                "definition": "SELECT * FROM users WHERE status = 'active'",
            }
        ]
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        views = await profiler._fetch_views("mydb")

        assert len(views) == 1
        v = views[0]
        assert v["name"] == "active_users"
        assert v["schema"] == "mydb"
        assert v["owner"] == ""
        assert "SELECT" in v["definition"]

    @pytest.mark.asyncio
    async def test_no_views_returns_empty_list(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        assert await profiler._fetch_views("mydb") == []


class TestMySQLProfilerForeignKeys:
    @pytest.mark.asyncio
    async def test_foreign_key_fields_mapped_correctly(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [
            {
                "constraint_name": "fk_order_user",
                "column_name": "user_id",
                "foreign_table_schema": "mydb",
                "foreign_table_name": "users",
                "foreign_column_name": "id",
                "update_rule": "CASCADE",
                "delete_rule": "RESTRICT",
            }
        ]
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        fks = await profiler._fetch_foreign_keys("mydb", "orders")

        assert len(fks) == 1
        fk = fks[0]
        assert fk["constraint_name"] == "fk_order_user"
        assert fk["from_column"] == "user_id"
        assert fk["to_schema"] == "mydb"
        assert fk["to_table"] == "users"
        assert fk["to_column"] == "id"
        assert fk["on_update"] == "CASCADE"
        assert fk["on_delete"] == "RESTRICT"

    @pytest.mark.asyncio
    async def test_no_foreign_keys_returns_empty_list(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = MySQLProfiler(mock_db, ProfilingConfig())

        assert await profiler._fetch_foreign_keys("mydb", "standalone") == []


# ── MySQLProfilerService._assemble_response ───────────────────────────────────


class TestMySQLProfilerServiceAssemble:
    _RAW_EMPTY = {
        "database": {
            "name": "testdb",
            "version": "8.0.32",
            "encoding": "utf8mb4",
            "size_bytes": 1024,
        },
        "schemas": [],
    }

    _RAW_FULL = {
        "database": {
            "name": "testdb",
            "version": "8.0.32",
            "encoding": "utf8mb4",
            "size_bytes": 8192,
        },
        "schemas": [
            {
                "name": "mydb",
                "owner": "",
                "tables": [
                    {
                        "name": "users",
                        "schema": "mydb",
                        "owner": "",
                        "description": "User accounts",
                        "size_bytes": 1024,
                        "total_size_bytes": 2048,
                        "row_count": 250,
                        "columns": [
                            {
                                "name": "id",
                                "ordinal_position": 1,
                                "data_type": "int",
                                "is_nullable": False,
                                "column_default": None,
                                "character_maximum_length": None,
                                "numeric_precision": 10,
                                "numeric_scale": 0,
                                "is_primary_key": True,
                                "description": None,
                                "enum_values": None,
                                "sample_values": None,
                            },
                            {
                                "name": "status",
                                "ordinal_position": 2,
                                "data_type": "enum",
                                "is_nullable": True,
                                "column_default": "active",
                                "character_maximum_length": None,
                                "numeric_precision": None,
                                "numeric_scale": None,
                                "is_primary_key": False,
                                "description": "Account status",
                                "enum_values": ["active", "inactive"],
                                "sample_values": ["active", "active", "inactive"],
                            },
                        ],
                        "indexes": [
                            {
                                "name": "PRIMARY",
                                "columns": ["id"],
                                "is_unique": True,
                                "is_primary": True,
                                "index_type": "BTREE",
                            }
                        ],
                        "relationships": [
                            {
                                "constraint_name": "fk_users_group",
                                "from_column": "group_id",
                                "to_schema": "mydb",
                                "to_table": "groups",
                                "to_column": "id",
                                "on_update": "CASCADE",
                                "on_delete": "SET NULL",
                            }
                        ],
                        "data_freshness": {
                            "last_analyze": None,
                            "last_autoanalyze": None,
                            "last_vacuum": datetime(2024, 3, 10, tzinfo=UTC),
                            "last_autovacuum": None,
                        },
                    }
                ],
                "views": [
                    {
                        "name": "active_users",
                        "schema": "mydb",
                        "owner": "",
                        "definition": "SELECT * FROM users WHERE status = 'active'",
                    }
                ],
            }
        ],
    }

    def test_assembles_empty_schema_list(self):
        svc = MySQLProfilerService()
        now = datetime.now(tz=UTC)
        result = svc._assemble_response(self._RAW_EMPTY, now)

        assert isinstance(result, ProfilingResponse)
        assert result.database.name == "testdb"
        assert result.database.version == "8.0.32"
        assert result.database.encoding == "utf8mb4"
        assert result.database.size_bytes == 1024
        assert result.schemas == []
        assert result.profiled_at == now

    def test_assembles_schema_with_table_and_view(self):
        svc = MySQLProfilerService()
        result = svc._assemble_response(self._RAW_FULL, datetime.now(tz=UTC))

        assert len(result.schemas) == 1
        schema = result.schemas[0]
        assert schema.name == "mydb"
        assert schema.owner == ""
        assert len(schema.tables) == 1
        assert len(schema.views) == 1

    def test_table_metadata_fields(self):
        svc = MySQLProfilerService()
        result = svc._assemble_response(
            self._RAW_FULL,
            datetime.now(tz=UTC),
        )

        table = result.schemas[0].tables[0]
        assert table.name == "users"
        assert table.row_count == 250
        assert table.size_bytes == 1024
        assert table.total_size_bytes == 2048
        assert table.description == "User accounts"

    def test_column_metadata_including_enum(self):
        svc = MySQLProfilerService()
        result = svc._assemble_response(
            self._RAW_FULL,
            datetime.now(tz=UTC),
        )

        columns = result.schemas[0].tables[0].columns
        assert len(columns) == 2

        pk_col = columns[0]
        assert pk_col.name == "id"
        assert pk_col.is_primary_key is True
        assert pk_col.enum_values is None

        enum_col = columns[1]
        assert enum_col.name == "status"
        assert enum_col.data_type == "enum"
        assert enum_col.enum_values == ["active", "inactive"]
        assert enum_col.sample_values == ["active", "active", "inactive"]

    def test_index_metadata(self):
        svc = MySQLProfilerService()
        result = svc._assemble_response(
            self._RAW_FULL,
            datetime.now(tz=UTC),
        )

        indexes = result.schemas[0].tables[0].indexes
        assert indexes is not None
        assert len(indexes) == 1
        idx = indexes[0]
        assert idx.name == "PRIMARY"
        assert idx.columns == ["id"]
        assert idx.is_primary is True
        assert idx.is_unique is True
        assert idx.index_type == "BTREE"

    def test_relationship_metadata(self):
        svc = MySQLProfilerService()
        result = svc._assemble_response(
            self._RAW_FULL,
            datetime.now(tz=UTC),
        )

        rels = result.schemas[0].tables[0].relationships
        assert rels is not None
        assert len(rels) == 1
        rel = rels[0]
        assert rel.constraint_name == "fk_users_group"
        assert rel.from_column == "group_id"
        assert rel.to_table == "groups"
        assert rel.on_update == "CASCADE"
        assert rel.on_delete == "SET NULL"

    def test_data_freshness_mysql_mapping(self):
        svc = MySQLProfilerService()
        result = svc._assemble_response(
            self._RAW_FULL,
            datetime.now(tz=UTC),
        )

        freshness = result.schemas[0].tables[0].data_freshness
        assert freshness is not None
        assert freshness.last_analyze is None
        assert freshness.last_vacuum == datetime(2024, 3, 10, tzinfo=UTC)
        assert freshness.last_autoanalyze is None
        assert freshness.last_autovacuum is None

    def test_view_metadata(self):
        svc = MySQLProfilerService()
        result = svc._assemble_response(
            self._RAW_FULL,
            datetime.now(tz=UTC),
        )

        view = result.schemas[0].views[0]
        assert view.name == "active_users"
        assert view.owner == ""
        assert "SELECT" in view.definition

    def test_indexes_none_when_not_requested(self):
        raw = {
            "database": {"name": "db", "version": "8.0", "encoding": "utf8mb4", "size_bytes": 0},
            "schemas": [
                {
                    "name": "mydb",
                    "owner": "",
                    "tables": [
                        {
                            "name": "t",
                            "schema": "mydb",
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
        svc = MySQLProfilerService()
        result = svc._assemble_response(
            raw,
            datetime.now(tz=UTC),
        )

        table = result.schemas[0].tables[0]
        assert table.indexes is None
        assert table.relationships is None


# ── MySQLProfilerService metadata ──────────────────────────────────────────────


class TestMySQLProfilerServiceMetadata:
    def test_span_attributes(self):
        from ignite_data_connectors import MySQLConfig

        svc = MySQLProfilerService()
        conn = MySQLConfig(
            host="db.example.com",
            port=3307,
            database="prod",
            username="u",
            password="p",
        )
        attrs = svc._span_attributes(conn)

        assert attrs["db.system"] == "mysql"
        assert attrs["db.name"] == "prod"
        assert attrs["net.peer.name"] == "db.example.com"
        assert attrs["net.peer.port"] == 3307

    def test_log_context(self):
        from ignite_data_connectors import MySQLConfig

        svc = MySQLProfilerService()
        conn = MySQLConfig(
            host="db.example.com",
            port=3307,
            database="prod",
            username="admin",
            password="p",
        )
        ctx = svc._log_context(conn)

        assert ctx["host"] == "db.example.com"
        assert ctx["port"] == 3307
        assert ctx["database"] == "prod"
        assert ctx["username"] == "admin"

    def test_service_name_and_span_name(self):
        svc = MySQLProfilerService()
        assert svc.service_name == "MySQL"
        assert svc.span_name == "profiler.mysql"
