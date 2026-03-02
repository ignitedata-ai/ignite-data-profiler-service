"""Unit tests for PostgreSQL column statistics."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core.api.v1.schemas.profiler import (
    BooleanColumnStats,
    ColumnStatistics,
    NumericColumnStats,
    ProfilingConfig,
    StringColumnStats,
    TemporalColumnStats,
    TopValueEntry,
)
from core.services.postgres.profiler import PostgresProfiler


def _make_profiler(
    *,
    include_column_stats: bool = True,
    top_values_limit: int = 10,
    top_values_cardinality_threshold: int = 100,
) -> tuple[PostgresProfiler, AsyncMock]:
    """Create a PostgresProfiler with a mocked connector."""
    connector = AsyncMock()
    config = ProfilingConfig(
        include_column_stats=include_column_stats,
        top_values_limit=top_values_limit,
        top_values_cardinality_threshold=top_values_cardinality_threshold,
        include_sample_data=False,
        include_indexes=False,
        include_relationships=False,
        include_data_freshness=False,
        include_row_counts=False,
    )
    return PostgresProfiler(connector, config), connector


class _MockRow(dict):
    """A dict subclass that behaves like a database row."""

    pass


def _mock_row(data: dict) -> _MockRow:
    """Create a mock row that supports dict-style access."""
    return _MockRow(data)


class TestFetchColumnStatsCommon:
    @pytest.mark.asyncio
    async def test_returns_common_stats_for_all_columns(self):
        profiler, connector = _make_profiler()
        columns = [
            {"name": "id", "data_type": "integer", "enum_values": None},
            {"name": "name", "data_type": "varchar", "enum_values": None},
        ]
        # Common stats query response.
        connector.fetch_one = AsyncMock(
            return_value=_mock_row(
                {
                    "total_count": 100,
                    "id__non_null": 100,
                    "id__distinct": 100,
                    "name__non_null": 90,
                    "name__distinct": 50,
                    # Numeric stats for id.
                    "id__min": 1.0,
                    "id__max": 100.0,
                    "id__mean": 50.5,
                    "id__stddev": 29.0,
                    "id__variance": 841.0,
                    "id__sum": 5050.0,
                    "id__p50": 50.0,
                    "id__p5": 5.0,
                    "id__p25": 25.0,
                    "id__p75": 75.0,
                    "id__p95": 95.0,
                    "id__zero_count": 0,
                    "id__negative_count": 0,
                    # Outlier count (reused for outlier query).
                    "outlier_count": 2,
                    # String stats for name.
                    "name__min_length": 2,
                    "name__max_length": 50,
                    "name__avg_length": 12.5,
                    "name__empty_count": 3,
                }
            )
        )
        connector.fetch_all = AsyncMock(return_value=[])

        result = await profiler._fetch_column_stats("public", "users", columns)

        assert "id" in result
        assert "name" in result
        assert isinstance(result["id"], ColumnStatistics)
        assert result["id"].total_count == 100
        assert result["id"].null_count == 0
        assert result["id"].null_percentage == 0.0
        assert result["id"].distinct_count == 100
        assert result["name"].null_count == 10
        assert result["name"].null_percentage == 10.0
        assert result["name"].distinct_count == 50

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self):
        profiler, connector = _make_profiler()
        connector.fetch_one = AsyncMock(side_effect=RuntimeError("connection lost"))
        columns = [{"name": "id", "data_type": "integer", "enum_values": None}]
        result = await profiler._fetch_column_stats("public", "t", columns)
        assert result == {}


class TestFetchNumericStats:
    @pytest.mark.asyncio
    async def test_returns_numeric_stats_with_percentiles(self):
        profiler, connector = _make_profiler()
        row_data = {
            "age__min": 18.0,
            "age__max": 99.0,
            "age__mean": 45.2,
            "age__stddev": 15.3,
            "age__variance": 234.09,
            "age__sum": 45200.0,
            "age__p50": 44.0,
            "age__p5": 20.0,
            "age__p25": 32.0,
            "age__p75": 58.0,
            "age__p95": 80.0,
            "age__zero_count": 0,
            "age__negative_count": 0,
        }
        connector.fetch_one = AsyncMock(return_value=_mock_row(row_data))

        result = await profiler._fetch_numeric_stats('"public"."users"', [{"name": "age", "data_type": "integer"}])

        assert "age" in result
        ns = result["age"]
        assert isinstance(ns, NumericColumnStats)
        assert ns.min == 18.0
        assert ns.max == 99.0
        assert ns.mean == 45.2
        assert ns.median == 44.0
        assert ns.p25 == 32.0
        assert ns.p75 == 58.0
        assert ns.zero_count == 0

    @pytest.mark.asyncio
    async def test_returns_empty_on_no_row(self):
        profiler, connector = _make_profiler()
        connector.fetch_one = AsyncMock(return_value=None)
        result = await profiler._fetch_numeric_stats('"s"."t"', [{"name": "x", "data_type": "integer"}])
        assert result == {}


class TestFetchStringStats:
    @pytest.mark.asyncio
    async def test_returns_string_length_stats(self):
        profiler, connector = _make_profiler()
        connector.fetch_one = AsyncMock(
            return_value=_mock_row(
                {
                    "email__min_length": 5,
                    "email__max_length": 255,
                    "email__avg_length": 22.7,
                    "email__empty_count": 0,
                }
            )
        )
        result = await profiler._fetch_string_stats('"s"."t"', [{"name": "email", "data_type": "varchar"}])
        assert "email" in result
        ss = result["email"]
        assert isinstance(ss, StringColumnStats)
        assert ss.min_length == 5
        assert ss.max_length == 255
        assert ss.avg_length == 22.7
        assert ss.empty_count == 0


class TestFetchBooleanStats:
    @pytest.mark.asyncio
    async def test_returns_true_false_counts(self):
        profiler, connector = _make_profiler()
        connector.fetch_one = AsyncMock(return_value=_mock_row({"active__true_count": 80, "active__false_count": 20}))
        result = await profiler._fetch_boolean_stats('"s"."t"', [{"name": "active", "data_type": "boolean"}])
        assert "active" in result
        bs = result["active"]
        assert isinstance(bs, BooleanColumnStats)
        assert bs.true_count == 80
        assert bs.false_count == 20
        assert bs.true_percentage == 80.0


class TestFetchTemporalStats:
    @pytest.mark.asyncio
    async def test_returns_min_max_dates(self):
        profiler, connector = _make_profiler()
        connector.fetch_one = AsyncMock(return_value=_mock_row({"created__min": "2020-01-01", "created__max": "2025-12-31"}))
        result = await profiler._fetch_temporal_stats('"s"."t"', [{"name": "created", "data_type": "timestamp"}])
        assert "created" in result
        ts = result["created"]
        assert isinstance(ts, TemporalColumnStats)
        assert ts.min == "2020-01-01"
        assert ts.max == "2025-12-31"


class TestFetchTopValues:
    @pytest.mark.asyncio
    async def test_returns_top_values_with_percentages(self):
        profiler, connector = _make_profiler(top_values_limit=3)
        connector.fetch_all = AsyncMock(
            return_value=[
                _mock_row({"value": "active", "count": 60}),
                _mock_row({"value": "inactive", "count": 30}),
                _mock_row({"value": "pending", "count": 10}),
            ]
        )
        result = await profiler._fetch_top_values('"s"."t"', "status", 100)
        assert len(result) == 3
        assert isinstance(result[0], TopValueEntry)
        assert result[0].value == "active"
        assert result[0].count == 60
        assert result[0].percentage == 60.0


class TestFetchOutlierCount:
    @pytest.mark.asyncio
    async def test_counts_outliers_with_iqr(self):
        profiler, connector = _make_profiler()
        connector.fetch_one = AsyncMock(return_value=_mock_row({"outlier_count": 5}))
        # Q1=25, Q3=75, IQR=50, lower=-50, upper=150.
        result = await profiler._fetch_outlier_count('"s"."t"', "salary", 25.0, 75.0)
        assert result == 5


class TestEscapeIdentifier:
    def test_plain_name(self):
        assert PostgresProfiler._esc("users") == "users"

    def test_double_quote_escaped(self):
        assert PostgresProfiler._esc('my"table') == 'my""table'
