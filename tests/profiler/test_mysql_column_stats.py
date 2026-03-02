"""Unit tests for MySQL column statistics."""

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
from core.services.mysql.profiler import MySQLProfiler


def _make_profiler(
    *,
    include_column_stats: bool = True,
    top_values_limit: int = 10,
    top_values_cardinality_threshold: int = 100,
) -> tuple[MySQLProfiler, AsyncMock]:
    """Create a MySQLProfiler with a mocked connector."""
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
    return MySQLProfiler(connector, config), connector


class _MockRow(dict):
    """A dict subclass that behaves like a database row."""

    pass


def _mock_row(data: dict) -> _MockRow:
    """Create a mock row that supports dict-style access."""
    return _MockRow(data)


class TestFetchColumnStatsCommon:
    @pytest.mark.asyncio
    async def test_returns_common_stats(self):
        profiler, connector = _make_profiler()
        columns = [
            {"name": "id", "data_type": "int", "enum_values": None},
            {"name": "email", "data_type": "varchar", "enum_values": None},
        ]
        connector.fetch_one = AsyncMock(
            return_value=_mock_row(
                {
                    "total_count": 200,
                    "id__non_null": 200,
                    "id__distinct": 200,
                    "email__non_null": 180,
                    "email__distinct": 170,
                    # Numeric stats for id.
                    "id__min": 1,
                    "id__max": 200,
                    "id__mean": 100.5,
                    "id__stddev": 57.7,
                    "id__variance": 3329.0,
                    "id__sum": 20100,
                    "id__zero_count": 0,
                    "id__negative_count": 0,
                    # String stats for email.
                    "email__min_length": 10,
                    "email__max_length": 100,
                    "email__avg_length": 25.5,
                    "email__empty_count": 0,
                    # Median count + value for id.
                    "cnt": 200,
                    "median_value": 100.0,
                }
            )
        )
        connector.fetch_all = AsyncMock(return_value=[])

        result = await profiler._fetch_column_stats("mydb", "users", columns)

        assert "id" in result
        assert "email" in result
        assert isinstance(result["id"], ColumnStatistics)
        assert result["id"].total_count == 200
        assert result["id"].null_count == 0
        assert result["email"].null_count == 20
        assert result["email"].null_percentage == 10.0

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self):
        profiler, connector = _make_profiler()
        connector.fetch_one = AsyncMock(side_effect=RuntimeError("connection lost"))
        columns = [{"name": "id", "data_type": "int", "enum_values": None}]
        result = await profiler._fetch_column_stats("mydb", "t", columns)
        assert result == {}


class TestMySQLNumericStats:
    @pytest.mark.asyncio
    async def test_uses_sum_case_instead_of_filter(self):
        """MySQL uses SUM(CASE WHEN) for conditional counts."""
        profiler, connector = _make_profiler()
        connector.fetch_one = AsyncMock(
            return_value=_mock_row(
                {
                    "price__min": 0.99,
                    "price__max": 999.99,
                    "price__mean": 49.99,
                    "price__stddev": 80.0,
                    "price__variance": 6400.0,
                    "price__sum": 49990.0,
                    "price__zero_count": 2,
                    "price__negative_count": 0,
                }
            )
        )
        result = await profiler._fetch_numeric_stats("`db`.`t`", [{"name": "price", "data_type": "decimal"}])
        ns = result["price"]
        assert isinstance(ns, NumericColumnStats)
        assert ns.min == 0.99
        assert ns.max == 999.99
        assert ns.zero_count == 2
        # MySQL does not compute percentiles.
        assert ns.p5 is None
        assert ns.p25 is None
        assert ns.p75 is None
        assert ns.p95 is None
        assert ns.outlier_count is None

    @pytest.mark.asyncio
    async def test_median_populated_via_separate_query(self):
        profiler, connector = _make_profiler()
        connector.fetch_one = AsyncMock(
            side_effect=[
                _mock_row({"cnt": 100}),
                _mock_row({"median_value": 42.0}),
            ]
        )
        result = await profiler._fetch_median("`db`.`t`", "age")
        assert result == 42.0

    @pytest.mark.asyncio
    async def test_median_returns_none_on_empty(self):
        profiler, connector = _make_profiler()
        connector.fetch_one = AsyncMock(return_value=None)
        result = await profiler._fetch_median("`db`.`t`", "age")
        assert result is None


class TestMySQLStringStats:
    @pytest.mark.asyncio
    async def test_uses_char_length(self):
        profiler, connector = _make_profiler()
        connector.fetch_one = AsyncMock(
            return_value=_mock_row(
                {
                    "name__min_length": 1,
                    "name__max_length": 100,
                    "name__avg_length": 15.3,
                    "name__empty_count": 5,
                }
            )
        )
        result = await profiler._fetch_string_stats("`db`.`t`", [{"name": "name", "data_type": "varchar"}])
        assert "name" in result
        ss = result["name"]
        assert isinstance(ss, StringColumnStats)
        assert ss.min_length == 1
        assert ss.empty_count == 5


class TestMySQLBooleanStats:
    @pytest.mark.asyncio
    async def test_uses_sum_case_for_tinyint_booleans(self):
        profiler, connector = _make_profiler()
        connector.fetch_one = AsyncMock(return_value=_mock_row({"is_active__true_count": 150, "is_active__false_count": 50}))
        result = await profiler._fetch_boolean_stats("`db`.`t`", [{"name": "is_active", "data_type": "boolean"}])
        bs = result["is_active"]
        assert isinstance(bs, BooleanColumnStats)
        assert bs.true_count == 150
        assert bs.false_count == 50
        assert bs.true_percentage == 75.0


class TestMySQLTemporalStats:
    @pytest.mark.asyncio
    async def test_returns_min_max_as_strings(self):
        profiler, connector = _make_profiler()
        connector.fetch_one = AsyncMock(
            return_value=_mock_row({"created__min": "2020-01-01 00:00:00", "created__max": "2025-12-31 23:59:59"})
        )
        result = await profiler._fetch_temporal_stats("`db`.`t`", [{"name": "created", "data_type": "datetime"}])
        ts = result["created"]
        assert isinstance(ts, TemporalColumnStats)
        assert ts.min == "2020-01-01 00:00:00"


class TestMySQLTopValues:
    @pytest.mark.asyncio
    async def test_returns_top_values(self):
        profiler, connector = _make_profiler(top_values_limit=2)
        connector.fetch_all = AsyncMock(
            return_value=[
                _mock_row({"value": "admin", "count": 30}),
                _mock_row({"value": "user", "count": 70}),
            ]
        )
        result = await profiler._fetch_top_values("`db`.`t`", "role", 100)
        assert len(result) == 2
        assert isinstance(result[0], TopValueEntry)
        assert result[0].value == "admin"
        assert result[0].count == 30
        assert result[0].percentage == 30.0


class TestMySQLEscapeIdentifier:
    def test_plain_name(self):
        assert MySQLProfiler._esc("users") == "users"

    def test_backtick_escaped(self):
        assert MySQLProfiler._esc("my`table") == "my``table"
