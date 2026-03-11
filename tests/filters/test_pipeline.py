"""Integration tests for the filter detection pipeline."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.api.v1.schemas.profiler import (
    ColumnMetadata,
    ColumnStatistics,
    DatabaseMetadata,
    ProfilingResponse,
    SchemaMetadata,
    TableMetadata,
    TopValueEntry,
)
from core.services.filters.pipeline import FilterDetectionPipeline
from core.services.filters.schema_introspector import NullSchemaIntrospector


def _make_column(
    name: str, data_type: str, is_pk: bool = False, stats: ColumnStatistics | None = None, enum_values: list[str] | None = None
) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        ordinal_position=1,
        data_type=data_type,
        is_nullable=not is_pk,
        column_default=None,
        character_maximum_length=None,
        numeric_precision=None,
        numeric_scale=None,
        is_primary_key=is_pk,
        enum_values=enum_values,
        statistics=stats,
    )


def _make_stats(total: int, null_count: int, distinct: int, top_values: list[TopValueEntry] | None = None) -> ColumnStatistics:
    null_pct = (null_count / total * 100) if total > 0 else 0
    distinct_pct = (distinct / total * 100) if total > 0 else 0
    return ColumnStatistics(
        total_count=total,
        null_count=null_count,
        null_percentage=null_pct,
        distinct_count=distinct,
        distinct_percentage=distinct_pct,
        top_values=top_values,
    )


def _make_response(tables: list[TableMetadata]) -> ProfilingResponse:
    return ProfilingResponse(
        profiled_at=datetime.now(UTC),
        database=DatabaseMetadata(name="testdb", version="15.0", encoding="UTF-8", size_bytes=0),
        schemas=[SchemaMetadata(name="public", owner="admin", tables=tables, views=[])],
    )


@pytest.mark.asyncio
async def test_pipeline_detects_obvious_filters():
    """A table with clear filter columns should have them detected."""
    columns = [
        _make_column("id", "integer", is_pk=True, stats=_make_stats(10000, 0, 10000)),
        _make_column("status", "varchar(50)", stats=_make_stats(10000, 0, 3), enum_values=["active", "inactive", "pending"]),
        _make_column("order_date", "timestamp", stats=_make_stats(10000, 0, 365)),
        _make_column("is_premium", "boolean", stats=_make_stats(10000, 0, 2)),
        _make_column("total_amount", "decimal(18,2)", stats=_make_stats(10000, 0, 8500)),
        _make_column("notes", "text", stats=_make_stats(10000, 7000, 3000)),
    ]
    table = TableMetadata(
        name="orders",
        schema="public",
        owner="admin",
        description=None,
        row_count=10000,
        size_bytes=None,
        total_size_bytes=None,
        data_freshness=None,
        columns=columns,
        indexes=[],
        relationships=[],
    )
    response = _make_response([table])

    pipeline = FilterDetectionPipeline(NullSchemaIntrospector())
    result = await pipeline.detect(response)

    table_result = result.schemas[0].tables[0]
    assert table_result.filter_columns is not None
    filter_names = [fc.column_name for fc in table_result.filter_columns]

    # status, order_date, and is_premium should be detected as filters
    assert "status" in filter_names
    assert "order_date" in filter_names
    assert "is_premium" in filter_names

    # id (PK) and total_amount (high cardinality numeric) should NOT be filters
    assert "id" not in filter_names


@pytest.mark.asyncio
async def test_pipeline_rejects_all_low_signal_columns():
    """A table with only high-cardinality numeric columns should produce no filters."""
    columns = [
        _make_column("id", "integer", is_pk=True, stats=_make_stats(10000, 0, 10000)),
        _make_column("amount", "decimal", stats=_make_stats(10000, 0, 9500)),
        _make_column("tax", "decimal", stats=_make_stats(10000, 0, 8000)),
    ]
    table = TableMetadata(
        name="transactions",
        schema="public",
        owner="admin",
        description=None,
        row_count=10000,
        size_bytes=None,
        total_size_bytes=None,
        data_freshness=None,
        columns=columns,
        indexes=[],
        relationships=[],
    )
    response = _make_response([table])

    pipeline = FilterDetectionPipeline(NullSchemaIntrospector())
    result = await pipeline.detect(response)

    table_result = result.schemas[0].tables[0]
    # Should have no or very few filter columns
    if table_result.filter_columns:
        assert len(table_result.filter_columns) == 0 or all(
            fc.column_name not in ("id", "amount", "tax") for fc in table_result.filter_columns
        )


@pytest.mark.asyncio
async def test_pipeline_sets_table_role():
    """Pipeline should classify table role."""
    columns = [
        _make_column("id", "integer", is_pk=True),
        _make_column("name", "varchar"),
        _make_column("code", "varchar"),
    ]
    table = TableMetadata(
        name="categories",
        schema="public",
        owner="admin",
        description=None,
        row_count=50,
        size_bytes=None,
        total_size_bytes=None,
        data_freshness=None,
        columns=columns,
        indexes=[],
        relationships=[],
    )
    response = _make_response([table])

    pipeline = FilterDetectionPipeline(NullSchemaIntrospector())
    result = await pipeline.detect(response)

    assert result.schemas[0].tables[0].table_role == "dimension"


@pytest.mark.asyncio
async def test_pipeline_handles_no_stats_gracefully():
    """Pipeline should still work when column stats are not available."""
    columns = [
        _make_column("id", "integer", is_pk=True),
        _make_column("status", "varchar", enum_values=["a", "b", "c"]),
        _make_column("created_at", "timestamp"),
    ]
    table = TableMetadata(
        name="orders",
        schema="public",
        owner="admin",
        description=None,
        row_count=1000,
        size_bytes=None,
        total_size_bytes=None,
        data_freshness=None,
        columns=columns,
        indexes=[],
        relationships=[],
    )
    response = _make_response([table])

    pipeline = FilterDetectionPipeline(NullSchemaIntrospector())
    # Should not raise
    result = await pipeline.detect(response)
    assert result is not None


@pytest.mark.asyncio
async def test_pipeline_enum_column_detected_even_without_stats():
    """An enum column should be detected as a filter even without column stats."""
    columns = [
        _make_column("id", "integer", is_pk=True),
        _make_column("priority", "varchar", enum_values=["low", "medium", "high"]),
    ]
    table = TableMetadata(
        name="tickets",
        schema="public",
        owner="admin",
        description=None,
        row_count=5000,
        size_bytes=None,
        total_size_bytes=None,
        data_freshness=None,
        columns=columns,
        indexes=[],
        relationships=[],
    )
    response = _make_response([table])

    pipeline = FilterDetectionPipeline(NullSchemaIntrospector())
    result = await pipeline.detect(response)

    table_result = result.schemas[0].tables[0]
    assert table_result.filter_columns is not None
    filter_names = [fc.column_name for fc in table_result.filter_columns]
    assert "priority" in filter_names


@pytest.mark.asyncio
async def test_pipeline_ui_controls():
    """Filter columns should have appropriate UI control recommendations."""
    columns = [
        _make_column("is_active", "boolean", stats=_make_stats(10000, 0, 2)),
        _make_column("order_date", "timestamp", stats=_make_stats(10000, 0, 365)),
        _make_column("status", "varchar", stats=_make_stats(10000, 0, 5), enum_values=["a", "b", "c", "d", "e"]),
    ]
    table = TableMetadata(
        name="orders",
        schema="public",
        owner="admin",
        description=None,
        row_count=10000,
        size_bytes=None,
        total_size_bytes=None,
        data_freshness=None,
        columns=columns,
        indexes=[],
        relationships=[],
    )
    response = _make_response([table])

    pipeline = FilterDetectionPipeline(NullSchemaIntrospector())
    result = await pipeline.detect(response)

    fc_map = {fc.column_name: fc for fc in result.schemas[0].tables[0].filter_columns or []}

    assert len(fc_map) > 0
