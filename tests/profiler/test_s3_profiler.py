"""Unit tests for the S3 file profiler.

All tests are pure unit tests — no AWS credentials or network access required.
Integration tests that require real S3/MinIO are marked @pytest.mark.integration.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from core.api.v1.schemas.profiler import ProfilingConfig, ProfilingResponse
from core.api.v1.schemas.s3 import S3ConnectionConfig, S3FileProfilingRequest, S3PathConfig
from core.services.s3.profiler import (
    S3FileProfilerService,
    _build_read_expression,
    _build_s3_uri,
    _derive_table_name,
    _esc,
    _esc_sql_string,
    _make_json_safe,
)

# ── Helper factories ────────────────────────────────────────────────────────────


def _make_path(bucket: str = "my-bucket", key: str = "data/orders.csv", **kwargs) -> S3PathConfig:
    return S3PathConfig(bucket=bucket, key=key, **kwargs)


def _make_conn(**kwargs) -> S3ConnectionConfig:
    return S3ConnectionConfig(**kwargs)


def _make_request(paths=None, **kwargs) -> S3FileProfilingRequest:
    if paths is None:
        paths = [_make_path()]
    return S3FileProfilingRequest(datasource_type="s3_file", paths=paths, **kwargs)


# ── _esc / _esc_sql_string ──────────────────────────────────────────────────────


class TestEscHelpers:
    def test_esc_passthrough(self):
        assert _esc("column_name") == "column_name"

    def test_esc_double_quotes(self):
        assert _esc('col"name') == 'col""name'

    def test_esc_sql_string_passthrough(self):
        assert _esc_sql_string("value") == "value"

    def test_esc_sql_string_single_quote(self):
        assert _esc_sql_string("it's") == "it''s"


# ── _make_json_safe ─────────────────────────────────────────────────────────────


class TestMakeJsonSafe:
    def test_none(self):
        assert _make_json_safe(None) is None

    def test_primitives(self):
        assert _make_json_safe(1) == 1
        assert _make_json_safe(1.5) == 1.5
        assert _make_json_safe("hello") == "hello"
        assert _make_json_safe(True) is True

    def test_datetime(self):
        dt = datetime(2024, 1, 15, 12, 0, 0)
        assert _make_json_safe(dt) == "2024-01-15T12:00:00"

    def test_bytes(self):
        assert _make_json_safe(b"\x00\xff") == "00ff"

    def test_list(self):
        assert _make_json_safe([1, None, "a"]) == [1, None, "a"]

    def test_non_serialisable_fallback(self):
        class _Weird:
            def __repr__(self):
                return "weird"

        result = _make_json_safe(_Weird())
        assert isinstance(result, str)


# ── _build_s3_uri ────────────────────────────────────────────────────────────────


class TestBuildS3Uri:
    def test_basic(self):
        path = _make_path(bucket="bucket", key="folder/file.csv")
        assert _build_s3_uri(path) == "s3://bucket/folder/file.csv"

    def test_strips_leading_slash(self):
        path = _make_path(key="/data/file.csv")
        assert _build_s3_uri(path) == "s3://my-bucket/data/file.csv"

    def test_glob_pattern(self):
        path = _make_path(key="data/2024/*.csv")
        assert _build_s3_uri(path) == "s3://my-bucket/data/2024/*.csv"


# ── _build_read_expression ──────────────────────────────────────────────────────


class TestBuildReadExpression:
    def test_csv_explicit(self):
        path = _make_path(key="f.csv", file_format="csv")
        expr = _build_read_expression(path)
        assert expr.startswith("read_csv_auto(")
        assert "s3://my-bucket/f.csv" in expr
        assert "header=true" in expr

    def test_csv_no_header(self):
        path = _make_path(key="f.csv", file_format="csv", has_header=False)
        assert "header=false" in _build_read_expression(path)

    def test_csv_custom_delimiter(self):
        path = _make_path(key="f.csv", file_format="csv", delimiter="|")
        assert "delim='|'" in _build_read_expression(path)

    def test_csv_delimiter_escapes_single_quote(self):
        path = _make_path(key="f.csv", file_format="csv", delimiter="'")
        assert "delim=''" in _build_read_expression(path)

    def test_parquet_explicit(self):
        path = _make_path(key="events.parquet", file_format="parquet")
        assert _build_read_expression(path) == "read_parquet('s3://my-bucket/events.parquet')"

    def test_json_explicit(self):
        path = _make_path(key="records.json", file_format="json")
        assert _build_read_expression(path) == "read_json_auto('s3://my-bucket/records.json')"

    def test_auto_infers_parquet_by_extension(self):
        path = _make_path(key="data/events.parquet", file_format="auto")
        expr = _build_read_expression(path)
        assert "read_parquet" in expr

    def test_auto_infers_json_by_extension(self):
        path = _make_path(key="data/records.jsonl", file_format="auto")
        expr = _build_read_expression(path)
        assert "read_json_auto" in expr

    def test_auto_defaults_to_csv_for_unknown_extension(self):
        path = _make_path(key="data/dump.gz", file_format="auto")
        assert "read_csv_auto" in _build_read_expression(path)

    def test_auto_defaults_to_csv_for_glob(self):
        path = _make_path(key="data/2024/*.csv", file_format="auto")
        assert "read_csv_auto" in _build_read_expression(path)

    def test_uri_single_quote_escaped(self):
        path = _make_path(bucket="my-bucket", key="data/it's.csv", file_format="csv")
        expr = _build_read_expression(path)
        assert "it''s.csv" in expr


# ── _derive_table_name ──────────────────────────────────────────────────────────


class TestDeriveTableName:
    def test_explicit_name(self):
        path = _make_path(name="my_table")
        assert _derive_table_name(path) == "my_table"

    def test_strips_csv(self):
        path = _make_path(key="orders.csv")
        assert _derive_table_name(path) == "orders"

    def test_strips_parquet(self):
        path = _make_path(key="events.parquet")
        assert _derive_table_name(path) == "events"

    def test_strips_json(self):
        path = _make_path(key="logs.json")
        assert _derive_table_name(path) == "logs"

    def test_strips_gz(self):
        path = _make_path(key="data.csv.gz")
        # Strips .gz, leaving "data.csv"; re.sub then replaces "." with "_" → "data_csv"
        assert _derive_table_name(path) == "data_csv"

    def test_nested_path_takes_last_segment(self):
        path = _make_path(key="a/b/c/users.csv")
        assert _derive_table_name(path) == "users"

    def test_glob_last_segment(self):
        # "*.csv" → strip .csv → "*" → replaced by "_"
        path = _make_path(key="data/2024/*.csv")
        name = _derive_table_name(path)
        assert name  # non-empty
        assert re.match(r"^[a-zA-Z0-9_]+$", name)

    def test_normalises_special_chars(self):
        path = _make_path(key="my-orders.csv")
        assert _derive_table_name(path) == "my_orders"

    def test_fallback_for_empty_result(self):
        # Key with only special characters after stripping
        path = _make_path(key="---.csv")
        assert _derive_table_name(path) == "s3_file"


# ── S3ConnectionConfig schema ───────────────────────────────────────────────────


class TestS3ConnectionConfigSchema:
    def test_all_optional_for_iam_role(self):
        cfg = S3ConnectionConfig()
        assert cfg.aws_access_key_id is None
        assert cfg.aws_secret_access_key is None
        assert cfg.aws_session_token is None
        assert cfg.aws_region == "us-east-1"
        assert cfg.endpoint_url is None
        assert cfg.use_ssl is True

    def test_explicit_credentials(self):
        cfg = S3ConnectionConfig(
            aws_access_key_id="AKID123",
            aws_secret_access_key="SECRET456",
            aws_region="eu-west-1",
        )
        assert cfg.aws_access_key_id == "AKID123"
        assert cfg.aws_secret_access_key.get_secret_value() == "SECRET456"
        assert cfg.aws_region == "eu-west-1"

    def test_minio_endpoint(self):
        cfg = S3ConnectionConfig(endpoint_url="http://localhost:9000", use_ssl=False)
        assert cfg.endpoint_url == "http://localhost:9000"
        assert cfg.use_ssl is False

    def test_secret_not_in_repr(self):
        cfg = S3ConnectionConfig(aws_secret_access_key="topsecret")
        assert "topsecret" not in repr(cfg)

    def test_frozen_model(self):
        cfg = S3ConnectionConfig()
        with pytest.raises(Exception):
            cfg.aws_region = "us-west-2"  # type: ignore[misc]


# ── S3PathConfig schema ─────────────────────────────────────────────────────────


class TestS3PathConfigSchema:
    def test_requires_bucket_and_key(self):
        with pytest.raises(ValidationError):
            S3PathConfig()  # type: ignore[call-arg]

    def test_requires_key(self):
        with pytest.raises(ValidationError):
            S3PathConfig(bucket="b")  # type: ignore[call-arg]

    def test_basic(self):
        path = S3PathConfig(bucket="my-bucket", key="data/orders.csv")
        assert path.bucket == "my-bucket"
        assert path.key == "data/orders.csv"
        assert path.file_format == "auto"
        assert path.has_header is True
        assert path.delimiter is None
        assert path.name is None

    def test_glob_pattern_accepted(self):
        path = S3PathConfig(bucket="b", key="data/2024/**/*.csv")
        assert "**" in path.key

    def test_explicit_name(self):
        path = S3PathConfig(bucket="b", key="f.csv", name="orders")
        assert path.name == "orders"

    def test_csv_options(self):
        path = S3PathConfig(bucket="b", key="f.csv", delimiter="|", has_header=False)
        assert path.delimiter == "|"
        assert path.has_header is False


# ── S3FileProfilingRequest schema ───────────────────────────────────────────────


class TestS3FileProfilingRequestSchema:
    def test_requires_paths(self):
        with pytest.raises(ValidationError):
            S3FileProfilingRequest(datasource_type="s3_file")  # type: ignore[call-arg]

    def test_requires_at_least_one_path(self):
        with pytest.raises(ValidationError):
            S3FileProfilingRequest(datasource_type="s3_file", paths=[])

    def test_discriminator(self):
        req = _make_request()
        assert req.datasource_type == "s3_file"

    def test_default_connection(self):
        req = _make_request()
        assert isinstance(req.connection, S3ConnectionConfig)

    def test_default_config_is_profiling_config(self):
        req = _make_request()
        assert isinstance(req.config, ProfilingConfig)

    def test_max_paths(self):
        with pytest.raises(ValidationError):
            S3FileProfilingRequest(
                datasource_type="s3_file",
                paths=[_make_path(key=f"f{i}.csv") for i in range(51)],
            )

    def test_discriminated_union_resolves_s3(self):
        from pydantic import TypeAdapter

        from core.api.v1.schemas.profiler import ProfilingRequest

        adapter = TypeAdapter(ProfilingRequest)
        obj = adapter.validate_python(
            {
                "datasource_type": "s3_file",
                "paths": [{"bucket": "my-bucket", "key": "data/orders.csv"}],
            }
        )
        assert isinstance(obj, S3FileProfilingRequest)

    def test_discriminated_union_still_resolves_postgres(self):
        from pydantic import TypeAdapter

        from core.api.v1.schemas.profiler import ProfilingRequest

        adapter = TypeAdapter(ProfilingRequest)
        obj = adapter.validate_python(
            {
                "datasource_type": "postgres",
                "connection": {
                    "host": "localhost",
                    "port": 5432,
                    "username": "user",
                    "password": "pass",
                    "database": "mydb",
                },
            }
        )
        from core.api.v1.schemas.profiler import PostgresProfilingRequest

        assert isinstance(obj, PostgresProfilingRequest)


# ── S3FileProfilerService metadata ──────────────────────────────────────────────


class TestS3FileProfilerServiceMetadata:
    def test_span_attributes_basic(self):
        svc = S3FileProfilerService()
        conn = _make_conn(aws_region="eu-west-1")
        attrs = svc._span_attributes(conn)
        assert attrs["db.system"] == "s3"
        assert attrs["s3.region"] == "eu-west-1"
        assert "s3.endpoint" not in attrs

    def test_span_attributes_custom_endpoint(self):
        svc = S3FileProfilerService()
        conn = _make_conn(endpoint_url="http://minio:9000")
        attrs = svc._span_attributes(conn)
        assert attrs["s3.endpoint"] == "http://minio:9000"

    def test_log_context_default(self):
        svc = S3FileProfilerService()
        conn = _make_conn(aws_region="ap-southeast-1")
        ctx = svc._log_context(conn)
        assert "ap-southeast-1" in ctx["host"]
        assert ctx["database"] == "s3_file"
        assert ctx["has_explicit_credentials"] is False

    def test_log_context_with_credentials(self):
        svc = S3FileProfilerService()
        conn = _make_conn(aws_access_key_id="AKID")
        ctx = svc._log_context(conn)
        assert ctx["has_explicit_credentials"] is True

    def test_log_context_custom_endpoint(self):
        svc = S3FileProfilerService()
        conn = _make_conn(endpoint_url="http://minio:9000")
        ctx = svc._log_context(conn)
        assert ctx["host"] == "http://minio:9000"


# ── S3FileProfilerService._assemble_response ────────────────────────────────────

_RAW_RESULT = {
    "database": {"name": "my-bucket", "version": "S3", "encoding": "UTF-8", "size_bytes": 0},
    "schemas": [
        {
            "name": "my-bucket",
            "owner": "s3.us-east-1.amazonaws.com",
            "tables": [
                {
                    "name": "orders",
                    "schema": "my-bucket",
                    "owner": "aws_s3",
                    "description": "s3://my-bucket/data/orders.csv",
                    "size_bytes": None,
                    "total_size_bytes": None,
                    "row_count": 1000,
                    "columns": [
                        {
                            "name": "order_id",
                            "ordinal_position": 1,
                            "data_type": "integer",
                            "is_nullable": True,
                            "column_default": None,
                            "character_maximum_length": None,
                            "numeric_precision": None,
                            "numeric_scale": None,
                            "is_primary_key": False,
                            "description": None,
                            "enum_values": None,
                            "sample_values": None,
                            "statistics": None,
                        },
                        {
                            "name": "customer_name",
                            "ordinal_position": 2,
                            "data_type": "varchar",
                            "is_nullable": True,
                            "column_default": None,
                            "character_maximum_length": None,
                            "numeric_precision": None,
                            "numeric_scale": None,
                            "is_primary_key": False,
                            "description": None,
                            "enum_values": None,
                            "sample_values": None,
                            "statistics": None,
                        },
                    ],
                    "indexes": [],
                    "relationships": [],
                    "data_freshness": None,
                }
            ],
            "views": [],
        }
    ],
}


class TestS3FileProfilerServiceAssemble:
    def test_returns_profiling_response(self):
        svc = S3FileProfilerService()
        result = svc._assemble_response(_RAW_RESULT, datetime.now(UTC))
        assert isinstance(result, ProfilingResponse)

    def test_database_metadata(self):
        svc = S3FileProfilerService()
        result = svc._assemble_response(_RAW_RESULT, datetime.now(UTC))
        assert result.database.name == "my-bucket"
        assert result.database.version == "S3"

    def test_schema_count(self):
        svc = S3FileProfilerService()
        result = svc._assemble_response(_RAW_RESULT, datetime.now(UTC))
        assert len(result.schemas) == 1
        assert result.schemas[0].name == "my-bucket"

    def test_table_metadata(self):
        svc = S3FileProfilerService()
        result = svc._assemble_response(_RAW_RESULT, datetime.now(UTC))
        table = result.schemas[0].tables[0]
        assert table.name == "orders"
        assert table.row_count == 1000
        assert table.data_freshness is None
        assert table.indexes == []
        assert table.relationships == []

    def test_column_count(self):
        svc = S3FileProfilerService()
        result = svc._assemble_response(_RAW_RESULT, datetime.now(UTC))
        assert len(result.schemas[0].tables[0].columns) == 2

    def test_column_metadata(self):
        svc = S3FileProfilerService()
        result = svc._assemble_response(_RAW_RESULT, datetime.now(UTC))
        col = result.schemas[0].tables[0].columns[0]
        assert col.name == "order_id"
        assert col.data_type == "integer"
        assert col.is_primary_key is False

    def test_views_empty(self):
        svc = S3FileProfilerService()
        result = svc._assemble_response(_RAW_RESULT, datetime.now(UTC))
        assert result.schemas[0].views == []

    def test_kpis_none_by_default(self):
        svc = S3FileProfilerService()
        result = svc._assemble_response(_RAW_RESULT, datetime.now(UTC))
        assert result.kpis is None

    def test_profiled_at_preserved(self):
        svc = S3FileProfilerService()
        ts = datetime(2024, 6, 15, 10, 30, 0, tzinfo=UTC)
        result = svc._assemble_response(_RAW_RESULT, ts)
        assert result.profiled_at == ts


# ── S3FileProfilerService.profile() (mocked) ─────────────────────────────────


class TestS3FileProfilerServiceProfile:
    """Test the profile() method with mocked DuckDB calls."""

    def _make_mock_raw_table(self) -> dict:
        return {
            "name": "orders",
            "schema": "my-bucket",
            "owner": "aws_s3",
            "description": "s3://my-bucket/data/orders.csv",
            "size_bytes": None,
            "total_size_bytes": None,
            "row_count": 42,
            "columns": [
                {
                    "name": "id",
                    "ordinal_position": 1,
                    "data_type": "integer",
                    "is_nullable": True,
                    "column_default": None,
                    "character_maximum_length": None,
                    "numeric_precision": None,
                    "numeric_scale": None,
                    "is_primary_key": False,
                    "description": None,
                    "enum_values": None,
                    "sample_values": None,
                    "statistics": None,
                }
            ],
            "indexes": [],
            "relationships": [],
            "data_freshness": None,
        }

    @pytest.mark.asyncio
    async def test_profile_calls_test_connection_and_profiler(self):
        svc = S3FileProfilerService()
        req = _make_request()
        raw_table = self._make_mock_raw_table()

        with (
            patch("core.services.s3.profiler._test_connection_sync") as mock_test,
            patch("core.services.s3.profiler._profile_one_path_sync", return_value=raw_table) as mock_profile,
        ):
            result = await svc.profile(req)

        mock_test.assert_called_once()
        mock_profile.assert_called_once()
        assert isinstance(result, ProfilingResponse)

    @pytest.mark.asyncio
    async def test_profile_skips_failed_paths_and_succeeds(self):
        """If one path fails but another succeeds, the result includes only the successes."""
        svc = S3FileProfilerService()
        path1 = _make_path(key="ok.csv")
        path2 = _make_path(key="bad.csv")
        req = _make_request(paths=[path1, path2])
        raw_table = self._make_mock_raw_table()

        call_count = 0

        def _side_effect(conn_cfg, path_cfg, config):
            nonlocal call_count
            call_count += 1
            if path_cfg.key == "bad.csv":
                raise RuntimeError("Simulated read error")
            return raw_table

        with (
            patch("core.services.s3.profiler._test_connection_sync"),
            patch("core.services.s3.profiler._profile_one_path_sync", side_effect=_side_effect),
        ):
            result = await svc.profile(req)

        assert call_count == 2
        assert len(result.schemas[0].tables) == 1
        assert result.schemas[0].tables[0].name == "orders"

    @pytest.mark.asyncio
    async def test_profile_all_paths_fail_raises(self):
        from core.exceptions import ExternalServiceError

        svc = S3FileProfilerService()
        req = _make_request()

        with (
            patch("core.services.s3.profiler._test_connection_sync"),
            patch("core.services.s3.profiler._profile_one_path_sync", side_effect=RuntimeError("fail")),
        ):
            with pytest.raises(ExternalServiceError):
                await svc.profile(req)

    @pytest.mark.asyncio
    async def test_profile_connection_error_raises(self):
        import duckdb

        from core.exceptions import ExternalServiceError

        svc = S3FileProfilerService()
        req = _make_request()

        with patch("core.services.s3.profiler._test_connection_sync", side_effect=duckdb.Error("403 Access Denied")):
            with pytest.raises(ExternalServiceError):
                await svc.profile(req)

    @pytest.mark.asyncio
    async def test_profile_progress_reported(self):
        from unittest.mock import AsyncMock

        svc = S3FileProfilerService()
        req = _make_request()
        raw_table = self._make_mock_raw_table()

        progress = MagicMock()
        progress.update = MagicMock()
        progress.flush = AsyncMock()  # AsyncMock returns a new coroutine on each call

        with (
            patch("core.services.s3.profiler._test_connection_sync"),
            patch("core.services.s3.profiler._profile_one_path_sync", return_value=raw_table),
        ):
            await svc.profile(req, progress=progress)

        assert progress.update.call_count >= 2  # at least connecting + profiling + completed
        phase_values = [c.kwargs.get("phase") for c in progress.update.call_args_list]
        assert "connecting" in phase_values
        assert "completed" in phase_values


# ── column_stats integration (sanity check) ─────────────────────────────────────


class TestColumnStatsIntegration:
    """Verify that classify_column_type handles DuckDB type names correctly."""

    def test_duckdb_integer_types(self):
        from core.services.column_stats import ColumnTypeCategory, classify_column_type

        for t in ("integer", "bigint", "smallint", "tinyint", "hugeint", "ubigint", "uinteger", "usmallint", "utinyint"):
            assert classify_column_type(t) == ColumnTypeCategory.NUMERIC, f"Expected NUMERIC for {t!r}"

    def test_duckdb_float_types(self):
        from core.services.column_stats import ColumnTypeCategory, classify_column_type

        for t in ("double", "float", "double precision"):
            assert classify_column_type(t) == ColumnTypeCategory.NUMERIC, f"Expected NUMERIC for {t!r}"

    def test_duckdb_varchar(self):
        from core.services.column_stats import ColumnTypeCategory, classify_column_type

        assert classify_column_type("varchar") == ColumnTypeCategory.STRING

    def test_duckdb_boolean(self):
        from core.services.column_stats import ColumnTypeCategory, classify_column_type

        assert classify_column_type("boolean") == ColumnTypeCategory.BOOLEAN

    def test_duckdb_date_types(self):
        from core.services.column_stats import ColumnTypeCategory, classify_column_type

        for t in ("date", "timestamp", "timestamptz", "timetz"):
            assert classify_column_type(t) == ColumnTypeCategory.TEMPORAL, f"Expected TEMPORAL for {t!r}"

    def test_type_with_precision_stripped(self):
        from core.services.column_stats import ColumnTypeCategory, classify_column_type

        assert classify_column_type("decimal(18,4)") == ColumnTypeCategory.NUMERIC
        assert classify_column_type("varchar(255)") == ColumnTypeCategory.STRING

    def test_uppercase_input_normalised(self):
        from core.services.column_stats import ColumnTypeCategory, classify_column_type

        assert classify_column_type("INTEGER") == ColumnTypeCategory.NUMERIC
        assert classify_column_type("VARCHAR") == ColumnTypeCategory.STRING
