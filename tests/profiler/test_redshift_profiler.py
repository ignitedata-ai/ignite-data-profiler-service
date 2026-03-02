"""Unit tests for the Redshift profiler module.

Covers:
- ``_make_json_safe`` — JSON serialisation helper
- ``RedshiftProfiler._filter_tables`` — include/exclude filtering
- ``RedshiftProfiler`` async fetch methods (mocked connector)
- ``RedshiftProfilerService._assemble_response`` — raw-dict → ProfilingResponse
- ``RedshiftProfilerService._span_attributes`` / ``_log_context``
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.api.v1.schemas.profiler import ProfilingConfig, ProfilingResponse
from core.services.redshift.profiler import (
    RedshiftProfiler,
    RedshiftProfilerService,
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


# ── RedshiftProfiler — filter logic ─────────────────────────────────────────


class TestRedshiftProfilerFilters:
    def _make_profiler(self, config: ProfilingConfig | None = None) -> RedshiftProfiler:
        return RedshiftProfiler(MagicMock(), config or ProfilingConfig())

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


# ── RedshiftProfiler — async fetch methods ─────────────────────────────────


class TestRedshiftProfilerSchemaDiscovery:
    @pytest.mark.asyncio
    async def test_exclude_schemas_passed_as_params(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [{"schema_name": "public"}]
        config = ProfilingConfig(exclude_schemas=["pg_catalog", "information_schema"])
        profiler = RedshiftProfiler(mock_db, config)

        await profiler._fetch_schema_names()

        params = mock_db.fetch_all.call_args[0][1]
        assert "pg_catalog" in params
        assert "information_schema" in params

    @pytest.mark.asyncio
    async def test_include_schemas_filter_applied(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [
            {"schema_name": "public"},
            {"schema_name": "analytics"},
            {"schema_name": "raw_data"},
        ]
        config = ProfilingConfig(include_schemas=["public", "raw_data"], exclude_schemas=[])
        profiler = RedshiftProfiler(mock_db, config)

        result = await profiler._fetch_schema_names()

        assert result == ["public", "raw_data"]

    @pytest.mark.asyncio
    async def test_placeholder_count_matches_exclude_list_length(self):
        """The SQL query must have the correct number of %s placeholders."""
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        config = ProfilingConfig(exclude_schemas=["extra1", "extra2"])
        profiler = RedshiftProfiler(mock_db, config)

        await profiler._fetch_schema_names()

        query: str = mock_db.fetch_all.call_args[0][0]
        params: tuple = mock_db.fetch_all.call_args[0][1]
        assert query.count("%s") == len(params)


class TestRedshiftProfilerColumnFetching:
    @pytest.mark.asyncio
    async def test_primary_key_column_flagged(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.side_effect = [
            [
                {
                    "column_name": "id",
                    "ordinal_position": 1,
                    "data_type": "integer",
                    "is_nullable": False,
                    "column_default": None,
                    "character_maximum_length": None,
                    "numeric_precision": 32,
                    "numeric_scale": 0,
                    "description": None,
                }
            ],
            [{"column_name": "id"}],  # PK
        ]
        profiler = RedshiftProfiler(mock_db, ProfilingConfig())

        columns = await profiler._fetch_columns("public", "users")

        assert columns[0]["is_primary_key"] is True
        assert columns[0]["enum_values"] is None

    @pytest.mark.asyncio
    async def test_enum_values_always_none(self):
        """Redshift does not have native enum types."""
        mock_db = AsyncMock()
        mock_db.fetch_all.side_effect = [
            [
                {
                    "column_name": "status",
                    "ordinal_position": 1,
                    "data_type": "character varying",
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
        profiler = RedshiftProfiler(mock_db, ProfilingConfig())

        columns = await profiler._fetch_columns("public", "users")

        assert len(columns) == 1
        assert columns[0]["enum_values"] is None


class TestRedshiftProfilerIndexFetching:
    @pytest.mark.asyncio
    async def test_always_returns_empty_list(self):
        """Redshift does not have traditional indexes."""
        profiler = RedshiftProfiler(MagicMock(), ProfilingConfig())
        indexes = await profiler._fetch_indexes()
        assert indexes == []


class TestRedshiftProfilerRowCount:
    @pytest.mark.asyncio
    async def test_returns_estimate_from_pg_class(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {"row_count": 500}
        profiler = RedshiftProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_row_count("public", "orders")

        assert result == 500
        mock_db.fetch_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_exact_count_when_negative(self):
        """reltuples = -1 triggers an exact COUNT(1)."""
        mock_db = AsyncMock()
        mock_db.fetch_one.side_effect = [
            {"row_count": -1},
            {"row_count": 42},
        ]
        profiler = RedshiftProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_row_count("public", "orders")

        assert result == 42
        assert mock_db.fetch_one.call_count == 2

    @pytest.mark.asyncio
    async def test_zero_rows_is_valid_and_returned_directly(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {"row_count": 0}
        profiler = RedshiftProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_row_count("public", "empty_table")

        assert result == 0
        mock_db.fetch_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_table_not_found(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = None
        profiler = RedshiftProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_row_count("public", "ghost")

        assert result is None


class TestRedshiftProfilerDataFreshness:
    @pytest.mark.asyncio
    async def test_returns_last_analyze_from_stl_analyze(self):
        analyze_dt = datetime(2024, 3, 12, tzinfo=UTC)
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {
            "last_analyze": analyze_dt,
            "last_autoanalyze": None,
            "last_vacuum": None,
            "last_autovacuum": None,
        }
        profiler = RedshiftProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_data_freshness("public", "orders")

        assert result["last_analyze"] == analyze_dt

    @pytest.mark.asyncio
    async def test_auto_fields_always_none(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = {
            "last_analyze": datetime(2024, 1, 1, tzinfo=UTC),
            "last_autoanalyze": None,
            "last_vacuum": None,
            "last_autovacuum": None,
        }
        profiler = RedshiftProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_data_freshness("public", "t")

        assert result["last_autoanalyze"] is None
        assert result["last_autovacuum"] is None

    @pytest.mark.asyncio
    async def test_returns_none_when_row_missing(self):
        mock_db = AsyncMock()
        mock_db.fetch_one.return_value = None
        profiler = RedshiftProfiler(mock_db, ProfilingConfig())

        result = await profiler._fetch_data_freshness("public", "ghost")

        assert result is None


class TestRedshiftProfilerSampleData:
    @pytest.mark.asyncio
    async def test_uses_double_quote_quoting(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = RedshiftProfiler(mock_db, ProfilingConfig(sample_size=5))

        await profiler._fetch_sample_data("public", "users")

        query: str = mock_db.fetch_all.call_args[0][0]
        assert '"public"' in query
        assert '"users"' in query
        assert "RANDOM()" in query
        assert "LIMIT" in query

    @pytest.mark.asyncio
    async def test_escapes_double_quotes_in_identifiers(self):
        """A double-quote in the identifier name must be doubled to prevent injection."""
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = RedshiftProfiler(mock_db, ProfilingConfig(sample_size=3))

        await profiler._fetch_sample_data('my"schema', 't"able')

        query: str = mock_db.fetch_all.call_args[0][0]
        assert '"my""schema"' in query
        assert '"t""able"' in query

    @pytest.mark.asyncio
    async def test_sample_size_passed_as_param(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = RedshiftProfiler(mock_db, ProfilingConfig(sample_size=25))

        await profiler._fetch_sample_data("public", "t")

        params = mock_db.fetch_all.call_args[0][1]
        assert params == (25,)

    @pytest.mark.asyncio
    async def test_uses_percent_s_placeholder(self):
        """Redshift uses %s placeholders, not $1."""
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = RedshiftProfiler(mock_db, ProfilingConfig(sample_size=10))

        await profiler._fetch_sample_data("public", "t")

        query: str = mock_db.fetch_all.call_args[0][0]
        assert "%s" in query
        assert "$1" not in query


class TestRedshiftProfilerViews:
    @pytest.mark.asyncio
    async def test_view_fields_mapped_correctly(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [
            {
                "table_name": "active_users",
                "table_schema": "public",
                "owner": "admin",
                "definition": "SELECT * FROM users WHERE status = 'active'",
            }
        ]
        profiler = RedshiftProfiler(mock_db, ProfilingConfig())

        views = await profiler._fetch_views("public")

        assert len(views) == 1
        v = views[0]
        assert v["name"] == "active_users"
        assert v["schema"] == "public"
        assert v["owner"] == "admin"
        assert "SELECT" in v["definition"]

    @pytest.mark.asyncio
    async def test_no_views_returns_empty_list(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = RedshiftProfiler(mock_db, ProfilingConfig())

        assert await profiler._fetch_views("public") == []


class TestRedshiftProfilerForeignKeys:
    @pytest.mark.asyncio
    async def test_foreign_key_fields_mapped_correctly(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = [
            {
                "constraint_name": "fk_order_user",
                "column_name": "user_id",
                "foreign_table_schema": "public",
                "foreign_table_name": "users",
                "foreign_column_name": "id",
                "update_rule": "NO ACTION",
                "delete_rule": "NO ACTION",
            }
        ]
        profiler = RedshiftProfiler(mock_db, ProfilingConfig())

        fks = await profiler._fetch_foreign_keys("public", "orders")

        assert len(fks) == 1
        fk = fks[0]
        assert fk["constraint_name"] == "fk_order_user"
        assert fk["from_column"] == "user_id"
        assert fk["to_schema"] == "public"
        assert fk["to_table"] == "users"
        assert fk["to_column"] == "id"
        assert fk["on_update"] == "NO ACTION"
        assert fk["on_delete"] == "NO ACTION"

    @pytest.mark.asyncio
    async def test_no_foreign_keys_returns_empty_list(self):
        mock_db = AsyncMock()
        mock_db.fetch_all.return_value = []
        profiler = RedshiftProfiler(mock_db, ProfilingConfig())

        assert await profiler._fetch_foreign_keys("public", "standalone") == []


# ── RedshiftProfilerService._assemble_response ────────────────────────────


class TestRedshiftProfilerServiceAssemble:
    _RAW_EMPTY = {
        "database": {
            "name": "dev",
            "version": "PostgreSQL 8.0.2 on i686-pc-linux-gnu",
            "encoding": "UTF8",
            "size_bytes": 1024,
        },
        "schemas": [],
    }

    _RAW_FULL = {
        "database": {
            "name": "dev",
            "version": "PostgreSQL 8.0.2 on i686-pc-linux-gnu",
            "encoding": "UTF8",
            "size_bytes": 8192,
        },
        "schemas": [
            {
                "name": "public",
                "owner": "awsuser",
                "tables": [
                    {
                        "name": "users",
                        "schema": "public",
                        "owner": "awsuser",
                        "description": "User accounts",
                        "size_bytes": 1024,
                        "total_size_bytes": 2048,
                        "row_count": 250,
                        "columns": [
                            {
                                "name": "id",
                                "ordinal_position": 1,
                                "data_type": "integer",
                                "is_nullable": False,
                                "column_default": None,
                                "character_maximum_length": None,
                                "numeric_precision": 32,
                                "numeric_scale": 0,
                                "is_primary_key": True,
                                "description": None,
                                "enum_values": None,
                                "sample_values": None,
                            },
                            {
                                "name": "status",
                                "ordinal_position": 2,
                                "data_type": "character varying",
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
                                "constraint_name": "fk_users_group",
                                "from_column": "group_id",
                                "to_schema": "public",
                                "to_table": "groups",
                                "to_column": "id",
                                "on_update": "NO ACTION",
                                "on_delete": "NO ACTION",
                            }
                        ],
                        "data_freshness": {
                            "last_analyze": datetime(2024, 3, 12, tzinfo=UTC),
                            "last_autoanalyze": None,
                            "last_vacuum": None,
                            "last_autovacuum": None,
                        },
                    }
                ],
                "views": [
                    {
                        "name": "active_users",
                        "schema": "public",
                        "owner": "awsuser",
                        "definition": "SELECT * FROM users WHERE status = 'active'",
                    }
                ],
            }
        ],
    }

    def test_assembles_empty_schema_list(self):
        svc = RedshiftProfilerService()
        now = datetime.now(tz=UTC)
        result = svc._assemble_response(self._RAW_EMPTY, now)

        assert isinstance(result, ProfilingResponse)
        assert result.database.name == "dev"
        assert result.database.encoding == "UTF8"
        assert result.database.size_bytes == 1024
        assert result.schemas == []
        assert result.profiled_at == now

    def test_assembles_schema_with_table_and_view(self):
        svc = RedshiftProfilerService()
        result = svc._assemble_response(self._RAW_FULL, datetime.now(tz=UTC))

        assert len(result.schemas) == 1
        schema = result.schemas[0]
        assert schema.name == "public"
        assert schema.owner == "awsuser"
        assert len(schema.tables) == 1
        assert len(schema.views) == 1

    def test_table_metadata_fields(self):
        svc = RedshiftProfilerService()
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

    def test_column_metadata(self):
        svc = RedshiftProfilerService()
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

        status_col = columns[1]
        assert status_col.name == "status"
        assert status_col.data_type == "character varying"
        assert status_col.enum_values is None
        assert status_col.sample_values == ["active", "active", "inactive"]

    def test_indexes_empty_list(self):
        svc = RedshiftProfilerService()
        result = svc._assemble_response(
            self._RAW_FULL,
            datetime.now(tz=UTC),
        )

        indexes = result.schemas[0].tables[0].indexes
        assert indexes is not None
        assert len(indexes) == 0

    def test_relationship_metadata(self):
        svc = RedshiftProfilerService()
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
        assert rel.on_update == "NO ACTION"
        assert rel.on_delete == "NO ACTION"

    def test_data_freshness_mapping(self):
        svc = RedshiftProfilerService()
        result = svc._assemble_response(
            self._RAW_FULL,
            datetime.now(tz=UTC),
        )

        freshness = result.schemas[0].tables[0].data_freshness
        assert freshness is not None
        assert freshness.last_analyze == datetime(2024, 3, 12, tzinfo=UTC)
        assert freshness.last_vacuum is None
        assert freshness.last_autoanalyze is None
        assert freshness.last_autovacuum is None

    def test_view_metadata(self):
        svc = RedshiftProfilerService()
        result = svc._assemble_response(
            self._RAW_FULL,
            datetime.now(tz=UTC),
        )

        view = result.schemas[0].views[0]
        assert view.name == "active_users"
        assert view.owner == "awsuser"
        assert "SELECT" in view.definition

    def test_indexes_none_when_not_requested(self):
        raw = {
            "database": {"name": "db", "version": "8.0", "encoding": "UTF8", "size_bytes": 0},
            "schemas": [
                {
                    "name": "public",
                    "owner": "",
                    "tables": [
                        {
                            "name": "t",
                            "schema": "public",
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
        svc = RedshiftProfilerService()
        result = svc._assemble_response(
            raw,
            datetime.now(tz=UTC),
        )

        table = result.schemas[0].tables[0]
        assert table.indexes is None
        assert table.relationships is None


# ── RedshiftProfilerService metadata ──────────────────────────────────────────


class TestRedshiftProfilerServiceMetadata:
    def test_span_attributes(self):
        from ignite_data_connectors import RedshiftConfig

        svc = RedshiftProfilerService()
        conn = RedshiftConfig(
            host="examplecluster.abc123.us-west-1.redshift.amazonaws.com",
            database="dev",
            username="awsuser",
            password="password",
            port=5439,
        )
        attrs = svc._span_attributes(conn)

        assert attrs["db.system"] == "redshift"
        assert attrs["db.name"] == "dev"
        assert attrs["net.peer.name"] == "examplecluster.abc123.us-west-1.redshift.amazonaws.com"
        assert attrs["net.peer.port"] == 5439

    def test_log_context(self):
        from ignite_data_connectors import RedshiftConfig

        svc = RedshiftProfilerService()
        conn = RedshiftConfig(
            host="examplecluster.abc123.us-west-1.redshift.amazonaws.com",
            database="dev",
            username="awsuser",
            password="password",
        )
        ctx = svc._log_context(conn)

        assert ctx["host"] == "examplecluster.abc123.us-west-1.redshift.amazonaws.com"
        assert ctx["database"] == "dev"
        assert ctx["username"] == "awsuser"
        assert ctx["port"] == 5439

    def test_service_name_and_span_name(self):
        svc = RedshiftProfilerService()
        assert svc.service_name == "Redshift"
        assert svc.span_name == "profiler.redshift"
