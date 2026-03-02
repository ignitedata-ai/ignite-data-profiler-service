"""Unit tests for Databricks column statistics."""

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
from core.services.databricks.profiler import DatabricksProfiler


def _make_profiler(
    *,
    include_column_stats: bool = True,
    top_values_limit: int = 10,
    top_values_cardinality_threshold: int = 100,
) -> tuple[DatabricksProfiler, AsyncMock]:
    """Create a DatabricksProfiler with a mocked connector."""
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
    return DatabricksProfiler(connector, config), connector


class TestFetchColumnStatsCommon:
    @pytest.mark.asyncio
    async def test_returns_common_stats_for_all_columns(self):
        profiler, connector = _make_profiler()
        columns = [
            {"name": "id", "data_type": "bigint", "enum_values": None},
            {"name": "name", "data_type": "string", "enum_values": None},
        ]
        connector.fetch_one = AsyncMock(
            return_value={
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
                # Outlier count.
                "outlier_count": 2,
                # String stats for name.
                "name__min_length": 2,
                "name__max_length": 50,
                "name__avg_length": 12.5,
                "name__empty_count": 3,
            }
        )
        connector.fetch_all = AsyncMock(return_value=[])

        result = await profiler._fetch_column_stats("default", "users", columns)

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
        columns = [{"name": "id", "data_type": "bigint", "enum_values": None}]
        result = await profiler._fetch_column_stats("default", "t", columns)
        assert result == {}


class TestFetchNumericStats:
    @pytest.mark.asyncio
    async def test_returns_numeric_stats_with_percentiles(self):
        profiler, connector = _make_profiler()
        columns = [{"name": "amount", "data_type": "double"}]
        connector.fetch_one.return_value = {
            "amount__min": 1.0,
            "amount__max": 999.0,
            "amount__mean": 250.0,
            "amount__stddev": 100.0,
            "amount__variance": 10000.0,
            "amount__sum": 25000.0,
            "amount__p50": 200.0,
            "amount__p5": 10.0,
            "amount__p25": 100.0,
            "amount__p75": 400.0,
            "amount__p95": 900.0,
            "amount__zero_count": 5,
            "amount__negative_count": 2,
        }

        result = await profiler._fetch_numeric_stats("`default`.`orders`", columns)

        assert "amount" in result
        ns = result["amount"]
        assert isinstance(ns, NumericColumnStats)
        assert ns.min == 1.0
        assert ns.max == 999.0
        assert ns.mean == 250.0
        assert ns.median == 200.0
        assert ns.p25 == 100.0
        assert ns.p75 == 400.0
        assert ns.zero_count == 5
        assert ns.negative_count == 2

    @pytest.mark.asyncio
    async def test_query_uses_backtick_identifiers(self):
        profiler, connector = _make_profiler()
        columns = [{"name": "val", "data_type": "int"}]
        connector.fetch_one.return_value = {
            "val__min": 1.0,
            "val__max": 10.0,
            "val__mean": 5.0,
            "val__stddev": 2.0,
            "val__variance": 4.0,
            "val__sum": 50.0,
            "val__p50": 5.0,
            "val__p5": 1.0,
            "val__p25": 3.0,
            "val__p75": 8.0,
            "val__p95": 10.0,
            "val__zero_count": 0,
            "val__negative_count": 0,
        }

        await profiler._fetch_numeric_stats("`default`.`t`", columns)

        query: str = connector.fetch_one.call_args[0][0]
        assert "CAST(" in query
        assert "AS DOUBLE)" in query
        assert "PERCENTILE_CONT" in query
        assert "`val`" in query


class TestFetchStringStats:
    @pytest.mark.asyncio
    async def test_returns_string_length_stats(self):
        profiler, connector = _make_profiler()
        columns = [{"name": "email", "data_type": "string"}]
        connector.fetch_one.return_value = {
            "email__min_length": 5,
            "email__max_length": 100,
            "email__avg_length": 25.0,
            "email__empty_count": 2,
        }

        result = await profiler._fetch_string_stats("`default`.`users`", columns)

        assert "email" in result
        ss = result["email"]
        assert isinstance(ss, StringColumnStats)
        assert ss.min_length == 5
        assert ss.max_length == 100
        assert ss.avg_length == 25.0
        assert ss.empty_count == 2


class TestFetchBooleanStats:
    @pytest.mark.asyncio
    async def test_returns_boolean_stats(self):
        profiler, connector = _make_profiler()
        columns = [{"name": "active", "data_type": "boolean"}]
        connector.fetch_one.return_value = {
            "active__true_count": 80,
            "active__false_count": 20,
        }

        result = await profiler._fetch_boolean_stats("`default`.`users`", columns)

        assert "active" in result
        bs = result["active"]
        assert isinstance(bs, BooleanColumnStats)
        assert bs.true_count == 80
        assert bs.false_count == 20
        assert bs.true_percentage == 80.0


class TestFetchTemporalStats:
    @pytest.mark.asyncio
    async def test_returns_temporal_min_max(self):
        profiler, connector = _make_profiler()
        columns = [{"name": "created_at", "data_type": "timestamp"}]
        connector.fetch_one.return_value = {
            "created_at__min": "2023-01-01T00:00:00",
            "created_at__max": "2024-06-15T12:30:00",
        }

        result = await profiler._fetch_temporal_stats("`default`.`events`", columns)

        assert "created_at" in result
        ts = result["created_at"]
        assert isinstance(ts, TemporalColumnStats)
        assert ts.min == "2023-01-01T00:00:00"
        assert ts.max == "2024-06-15T12:30:00"


class TestFetchTopValues:
    @pytest.mark.asyncio
    async def test_returns_top_values_with_percentages(self):
        profiler, connector = _make_profiler()
        connector.fetch_all.return_value = [
            {"value": "active", "count": 80},
            {"value": "inactive", "count": 20},
        ]

        result = await profiler._fetch_top_values("`default`.`users`", "status", 100)

        assert len(result) == 2
        assert isinstance(result[0], TopValueEntry)
        assert result[0].value == "active"
        assert result[0].count == 80
        assert result[0].percentage == 80.0
        assert result[1].value == "inactive"
        assert result[1].percentage == 20.0


class TestFetchOutlierCount:
    @pytest.mark.asyncio
    async def test_returns_outlier_count(self):
        profiler, connector = _make_profiler()
        connector.fetch_one.return_value = {"outlier_count": 5}

        result = await profiler._fetch_outlier_count("`default`.`orders`", "amount", 25.0, 75.0)

        assert result == 5

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_outliers(self):
        profiler, connector = _make_profiler()
        connector.fetch_one.return_value = {"outlier_count": 0}

        result = await profiler._fetch_outlier_count("`default`.`orders`", "amount", 10.0, 90.0)

        assert result == 0


class TestEscapeHelper:
    def test_plain_name(self):
        assert DatabricksProfiler._esc("my_table") == "my_table"

    def test_backtick_in_name(self):
        assert DatabricksProfiler._esc("my`table") == "my``table"

    def test_multiple_backticks(self):
        assert DatabricksProfiler._esc("a`b`c") == "a``b``c"
