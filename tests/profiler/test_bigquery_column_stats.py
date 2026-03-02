"""Unit tests for BigQuery column statistics."""

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
from core.services.bigquery.profiler import BigQueryProfiler


def _make_profiler(
    *,
    include_column_stats: bool = True,
    top_values_limit: int = 10,
    top_values_cardinality_threshold: int = 100,
) -> tuple[BigQueryProfiler, AsyncMock]:
    """Create a BigQueryProfiler with a mocked connector."""
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
    return BigQueryProfiler(connector, config, project="test-project"), connector


class TestFetchColumnStatsCommon:
    @pytest.mark.asyncio
    async def test_returns_common_stats_for_all_columns(self):
        profiler, connector = _make_profiler()
        columns = [
            {"name": "id", "data_type": "INT64", "enum_values": None},
            {"name": "name", "data_type": "STRING", "enum_values": None},
        ]
        connector.fetch_one = AsyncMock(
            return_value={
                "total_count": 100,
                "id__non_null": 100,
                "id__distinct": 100,
                "name__non_null": 90,
                "name__distinct": 50,
                # Numeric agg stats for id.
                "id__min": 1.0,
                "id__max": 100.0,
                "id__mean": 50.5,
                "id__stddev": 29.0,
                "id__variance": 841.0,
                "id__sum": 5050.0,
                "id__zero_count": 0,
                "id__negative_count": 0,
                # Percentiles for id.
                "id__p5": 5.0,
                "id__p25": 25.0,
                "id__p50": 50.0,
                "id__p75": 75.0,
                "id__p95": 95.0,
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

        result = await profiler._fetch_column_stats("my_dataset", "users", columns)

        assert "id" in result
        assert "name" in result
        assert isinstance(result["id"], ColumnStatistics)
        assert result["id"].total_count == 100
        assert result["id"].null_count == 0
        assert result["name"].null_count == 10
        assert result["name"].distinct_count == 50

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self):
        profiler, connector = _make_profiler()
        connector.fetch_one = AsyncMock(side_effect=RuntimeError("connection lost"))
        columns = [{"name": "id", "data_type": "INT64", "enum_values": None}]
        result = await profiler._fetch_column_stats("my_dataset", "t", columns)
        assert result == {}


class TestFetchNumericStats:
    @pytest.mark.asyncio
    async def test_returns_numeric_stats_with_approx_quantiles(self):
        profiler, connector = _make_profiler()
        columns = [{"name": "amount", "data_type": "FLOAT64"}]
        # Two concurrent queries: aggregates + percentiles.
        connector.fetch_one.side_effect = [
            {
                "amount__min": 1.0,
                "amount__max": 999.0,
                "amount__mean": 250.0,
                "amount__stddev": 100.0,
                "amount__variance": 10000.0,
                "amount__sum": 25000.0,
                "amount__zero_count": 5,
                "amount__negative_count": 2,
            },
            {
                "amount__p5": 10.0,
                "amount__p25": 100.0,
                "amount__p50": 200.0,
                "amount__p75": 400.0,
                "amount__p95": 900.0,
            },
        ]

        result = await profiler._fetch_numeric_stats("`my_dataset`.`orders`", columns)

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
    async def test_query_uses_approx_quantiles(self):
        profiler, connector = _make_profiler()
        columns = [{"name": "val", "data_type": "INT64"}]
        connector.fetch_one.side_effect = [
            {
                "val__min": 1.0,
                "val__max": 10.0,
                "val__mean": 5.0,
                "val__stddev": 2.0,
                "val__variance": 4.0,
                "val__sum": 50.0,
                "val__zero_count": 0,
                "val__negative_count": 0,
            },
            {
                "val__p5": 1.0,
                "val__p25": 3.0,
                "val__p50": 5.0,
                "val__p75": 8.0,
                "val__p95": 10.0,
            },
        ]

        await profiler._fetch_numeric_stats("`my_dataset`.`t`", columns)

        # Second call is the percentiles query.
        pct_query: str = connector.fetch_one.call_args_list[1][0][0]
        assert "APPROX_QUANTILES" in pct_query
        assert "OFFSET" in pct_query

    @pytest.mark.asyncio
    async def test_query_uses_countif(self):
        profiler, connector = _make_profiler()
        columns = [{"name": "val", "data_type": "INT64"}]
        connector.fetch_one.side_effect = [
            {
                "val__min": 0,
                "val__max": 10,
                "val__mean": 5.0,
                "val__stddev": 2.0,
                "val__variance": 4.0,
                "val__sum": 50.0,
                "val__zero_count": 1,
                "val__negative_count": 0,
            },
            {
                "val__p5": 1.0,
                "val__p25": 3.0,
                "val__p50": 5.0,
                "val__p75": 8.0,
                "val__p95": 10.0,
            },
        ]

        await profiler._fetch_numeric_stats("`ds`.`t`", columns)

        agg_query: str = connector.fetch_one.call_args_list[0][0][0]
        assert "COUNTIF(" in agg_query


class TestFetchStringStats:
    @pytest.mark.asyncio
    async def test_returns_string_length_stats(self):
        profiler, connector = _make_profiler()
        columns = [{"name": "email", "data_type": "STRING"}]
        connector.fetch_one.return_value = {
            "email__min_length": 5,
            "email__max_length": 100,
            "email__avg_length": 25.0,
            "email__empty_count": 2,
        }

        result = await profiler._fetch_string_stats("`my_dataset`.`users`", columns)

        assert "email" in result
        ss = result["email"]
        assert isinstance(ss, StringColumnStats)
        assert ss.min_length == 5
        assert ss.max_length == 100
        assert ss.avg_length == 25.0
        assert ss.empty_count == 2

    @pytest.mark.asyncio
    async def test_query_uses_countif_for_empty(self):
        profiler, connector = _make_profiler()
        columns = [{"name": "val", "data_type": "STRING"}]
        connector.fetch_one.return_value = {
            "val__min_length": 1,
            "val__max_length": 10,
            "val__avg_length": 5.0,
            "val__empty_count": 0,
        }

        await profiler._fetch_string_stats("`ds`.`t`", columns)

        query: str = connector.fetch_one.call_args[0][0]
        assert "COUNTIF(" in query


class TestFetchBooleanStats:
    @pytest.mark.asyncio
    async def test_returns_boolean_stats(self):
        profiler, connector = _make_profiler()
        columns = [{"name": "active", "data_type": "BOOL"}]
        connector.fetch_one.return_value = {
            "active__true_count": 80,
            "active__false_count": 20,
        }

        result = await profiler._fetch_boolean_stats("`my_dataset`.`users`", columns)

        assert "active" in result
        bs = result["active"]
        assert isinstance(bs, BooleanColumnStats)
        assert bs.true_count == 80
        assert bs.false_count == 20
        assert bs.true_percentage == 80.0

    @pytest.mark.asyncio
    async def test_query_uses_countif(self):
        profiler, connector = _make_profiler()
        columns = [{"name": "flag", "data_type": "BOOL"}]
        connector.fetch_one.return_value = {
            "flag__true_count": 10,
            "flag__false_count": 90,
        }

        await profiler._fetch_boolean_stats("`ds`.`t`", columns)

        query: str = connector.fetch_one.call_args[0][0]
        assert "COUNTIF(" in query


class TestFetchTemporalStats:
    @pytest.mark.asyncio
    async def test_returns_temporal_min_max(self):
        profiler, connector = _make_profiler()
        columns = [{"name": "created_at", "data_type": "TIMESTAMP"}]
        connector.fetch_one.return_value = {
            "created_at__min": "2023-01-01 00:00:00 UTC",
            "created_at__max": "2024-06-15 12:30:00 UTC",
        }

        result = await profiler._fetch_temporal_stats("`my_dataset`.`events`", columns)

        assert "created_at" in result
        ts = result["created_at"]
        assert isinstance(ts, TemporalColumnStats)
        assert ts.min == "2023-01-01 00:00:00 UTC"
        assert ts.max == "2024-06-15 12:30:00 UTC"


class TestFetchTopValues:
    @pytest.mark.asyncio
    async def test_returns_top_values_with_percentages(self):
        profiler, connector = _make_profiler()
        connector.fetch_all.return_value = [
            {"value": "active", "count": 80},
            {"value": "inactive", "count": 20},
        ]

        result = await profiler._fetch_top_values("`my_dataset`.`users`", "status", 100)

        assert len(result) == 2
        assert isinstance(result[0], TopValueEntry)
        assert result[0].value == "active"
        assert result[0].count == 80
        assert result[0].percentage == 80.0


class TestFetchOutlierCount:
    @pytest.mark.asyncio
    async def test_returns_outlier_count(self):
        profiler, connector = _make_profiler()
        connector.fetch_one.return_value = {"outlier_count": 5}

        result = await profiler._fetch_outlier_count("`my_dataset`.`orders`", "amount", 25.0, 75.0)

        assert result == 5

    @pytest.mark.asyncio
    async def test_query_uses_percent_s_params(self):
        profiler, connector = _make_profiler()
        connector.fetch_one.return_value = {"outlier_count": 0}

        await profiler._fetch_outlier_count("`ds`.`t`", "col", 10.0, 90.0)

        query: str = connector.fetch_one.call_args[0][0]
        assert "%s" in query
        params = connector.fetch_one.call_args[0][1]
        assert len(params) == 2


class TestEscapeHelper:
    def test_plain_name(self):
        assert BigQueryProfiler._esc("my_table") == "my_table"

    def test_backtick_in_name(self):
        assert BigQueryProfiler._esc("my`table") == "my``table"

    def test_multiple_backticks(self):
        assert BigQueryProfiler._esc("a`b`c") == "a``b``c"
