"""Unit tests for profiler Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from core.api.v1.schemas.profiler import (
    MySQLProfilingRequest,
    PostgresProfilingRequest,
    ProfilingConfig,
    ProfilingRequest,
)


class TestProfilingConfig:
    def test_defaults(self):
        cfg = ProfilingConfig()
        assert cfg.sample_size == 10
        assert cfg.timeout_seconds == 1000
        assert cfg.include_schemas is None
        assert "pg_catalog" in cfg.exclude_schemas

    def test_sample_size_min(self):
        with pytest.raises(PydanticValidationError):
            ProfilingConfig(sample_size=0)

    def test_sample_size_max(self):
        with pytest.raises(PydanticValidationError):
            ProfilingConfig(sample_size=1001)

    def test_timeout_min(self):
        with pytest.raises(PydanticValidationError):
            ProfilingConfig(timeout_seconds=0)

    def test_timeout_max(self):
        with pytest.raises(PydanticValidationError):
            ProfilingConfig(timeout_seconds=3601)

    def test_timeout_boundary_values_accepted(self):
        assert ProfilingConfig(timeout_seconds=1).timeout_seconds == 1
        assert ProfilingConfig(timeout_seconds=3600).timeout_seconds == 3600
        assert ProfilingConfig(sample_size=1).sample_size == 1
        assert ProfilingConfig(sample_size=1000).sample_size == 1000

    def test_include_column_stats_defaults_false(self):
        cfg = ProfilingConfig()
        assert cfg.include_column_stats is False

    def test_top_values_limit_range(self):
        assert ProfilingConfig(top_values_limit=1).top_values_limit == 1
        assert ProfilingConfig(top_values_limit=50).top_values_limit == 50
        with pytest.raises(PydanticValidationError):
            ProfilingConfig(top_values_limit=0)
        with pytest.raises(PydanticValidationError):
            ProfilingConfig(top_values_limit=51)

    def test_top_values_cardinality_threshold_range(self):
        assert ProfilingConfig(top_values_cardinality_threshold=1).top_values_cardinality_threshold == 1
        assert ProfilingConfig(top_values_cardinality_threshold=10000).top_values_cardinality_threshold == 10000
        with pytest.raises(PydanticValidationError):
            ProfilingConfig(top_values_cardinality_threshold=0)
        with pytest.raises(PydanticValidationError):
            ProfilingConfig(top_values_cardinality_threshold=10001)


class TestProfilingRequest:
    _base_conn = dict(host="localhost", database="db", username="u", password="p")

    def test_discriminated_union_resolves_postgres(self):
        """ProfilingRequest alias should resolve to PostgresProfilingRequest."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(ProfilingRequest)
        obj = adapter.validate_python(
            {
                "datasource_type": "postgres",
                "connection": self._base_conn,
                "config": {"sample_size": 5},
            }
        )
        assert isinstance(obj, PostgresProfilingRequest)
        assert obj.config.sample_size == 5

    def test_discriminated_union_resolves_mysql(self):
        """ProfilingRequest alias should resolve to MySQLProfilingRequest for mysql type."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(ProfilingRequest)
        obj = adapter.validate_python(
            {
                "datasource_type": "mysql",
                "connection": self._base_conn,
                "config": {"sample_size": 20},
            }
        )
        assert isinstance(obj, MySQLProfilingRequest)
        assert obj.config.sample_size == 20
        assert obj.connection.port == 3306  # MySQL default port

    def test_unknown_datasource_type_rejected(self):
        from pydantic import TypeAdapter

        adapter = TypeAdapter(ProfilingRequest)
        with pytest.raises(PydanticValidationError):
            adapter.validate_python(
                {
                    "datasource_type": "oracle",
                    "connection": self._base_conn,
                }
            )
