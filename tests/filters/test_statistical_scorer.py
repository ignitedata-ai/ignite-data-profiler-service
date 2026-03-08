"""Tests for the statistical scorer (Stage 2)."""

from __future__ import annotations

import pytest

from core.api.v1.schemas.profiler import ColumnMetadata, ColumnStatistics, TopValueEntry
from core.services.filters.statistical_scorer import (
    BUCKET_BOOLEAN,
    BUCKET_CONSTANT,
    BUCKET_HIGH_CARDINALITY,
    BUCKET_HIGH_CATEGORICAL,
    BUCKET_LOW_CATEGORICAL,
    BUCKET_NEAR_UNIQUE,
    classify_value_pattern,
    compute_cardinality_bucket,
    compute_cardinality_score,
    compute_dtype_score,
    compute_naming_score,
    compute_null_penalty,
    compute_statistical_signals,
)

# ── Cardinality bucketing ─────────────────────────────────────────────────────


class TestCardinalityBucket:
    @pytest.mark.parametrize(
        "distinct, rows, expected",
        [
            (0, 1000, BUCKET_CONSTANT),
            (1, 1000, BUCKET_CONSTANT),
            (2, 10000, BUCKET_BOOLEAN),
            (5, 10000, BUCKET_LOW_CATEGORICAL),
            (50, 10000, BUCKET_LOW_CATEGORICAL),
            (51, 10000, BUCKET_HIGH_CATEGORICAL),
            (500, 100000, BUCKET_HIGH_CATEGORICAL),
            (501, 100000, BUCKET_HIGH_CARDINALITY),
            (10001, 20000, BUCKET_NEAR_UNIQUE),
            (50000, 100000, BUCKET_NEAR_UNIQUE),
        ],
    )
    def test_bucketing(self, distinct: int, rows: int, expected: str):
        assert compute_cardinality_bucket(distinct, rows) == expected

    def test_zero_rows(self):
        assert compute_cardinality_bucket(0, 0) == BUCKET_CONSTANT


class TestCardinalityScore:
    def test_constant_zero(self):
        assert compute_cardinality_score(BUCKET_CONSTANT) == 0.0

    def test_low_categorical_highest(self):
        assert compute_cardinality_score(BUCKET_LOW_CATEGORICAL) == 0.95

    def test_unknown_bucket(self):
        assert compute_cardinality_score("unknown") == 0.05


# ── Data type scoring ─────────────────────────────────────────────────────────


class TestDtypeScore:
    def test_temporal_high(self):
        assert compute_dtype_score("timestamp", BUCKET_LOW_CATEGORICAL, False) == 0.90

    def test_boolean_high(self):
        assert compute_dtype_score("boolean", BUCKET_BOOLEAN, False) == 0.90

    def test_varchar_low_cardinality(self):
        assert compute_dtype_score("varchar(255)", BUCKET_LOW_CATEGORICAL, False) == 0.80

    def test_varchar_high_cardinality(self):
        assert compute_dtype_score("text", BUCKET_NEAR_UNIQUE, False) == 0.15

    def test_integer_pk(self):
        assert compute_dtype_score("integer", BUCKET_NEAR_UNIQUE, True) == 0.05

    def test_integer_low_cardinality_not_pk(self):
        assert compute_dtype_score("integer", BUCKET_LOW_CATEGORICAL, False) == 0.75

    def test_float_low(self):
        assert compute_dtype_score("decimal(18,4)", BUCKET_NEAR_UNIQUE, False) == 0.05

    def test_precision_stripped(self):
        # "decimal(18,4)" should be treated the same as "decimal"
        assert compute_dtype_score("decimal(18,4)", BUCKET_NEAR_UNIQUE, False) == compute_dtype_score(
            "decimal", BUCKET_NEAR_UNIQUE, False
        )


# ── Null penalty ──────────────────────────────────────────────────────────────


class TestNullPenalty:
    @pytest.mark.parametrize(
        "ratio, expected",
        [
            (0.0, 1.0),
            (0.04, 1.0),
            (0.10, 0.9),
            (0.29, 0.9),
            (0.45, 0.7),
            (0.70, 0.4),
            (0.90, 0.1),
        ],
    )
    def test_penalty_ranges(self, ratio: float, expected: float):
        assert compute_null_penalty(ratio) == expected


# ── Name pattern matching ─────────────────────────────────────────────────────


class TestNamingScore:
    def test_status_suffix(self):
        assert compute_naming_score("order_status") == 0.80

    def test_is_prefix(self):
        assert compute_naming_score("is_active") == 0.80

    def test_has_prefix(self):
        assert compute_naming_score("has_discount") == 0.80

    def test_temporal_suffix(self):
        assert compute_naming_score("order_date") == 0.85

    def test_audit_created_at(self):
        assert compute_naming_score("created_at") == 0.10

    def test_audit_updated_at(self):
        assert compute_naming_score("updated_at") == 0.10

    def test_description_anti_filter(self):
        assert compute_naming_score("product_description") == 0.05

    def test_id_high_cardinality(self):
        assert compute_naming_score("customer_id", BUCKET_NEAR_UNIQUE) == 0.05

    def test_id_low_cardinality(self):
        # Low cardinality ID might be a coded value — no anti-filter penalty
        assert compute_naming_score("customer_id", BUCKET_LOW_CATEGORICAL) == 0.50

    def test_neutral_name(self):
        assert compute_naming_score("amount") == 0.50

    def test_region_suffix(self):
        assert compute_naming_score("sales_region") == 0.80


# ── Value pattern classification ──────────────────────────────────────────────


class TestValuePattern:
    def test_iso_codes(self):
        top = [TopValueEntry(value=v, count=10, percentage=10.0) for v in ["US", "CA", "GB", "DE", "FR"]]
        pattern, score = classify_value_pattern(top)
        assert pattern == "code"
        assert score >= 0.8

    def test_identifiers(self):
        top = [TopValueEntry(value=v, count=5, percentage=5.0) for v in ["ORD-001", "ORD-002", "ORD-003", "INV-001", "INV-002"]]
        pattern, score = classify_value_pattern(top)
        assert pattern == "identifier"
        assert score <= 0.2

    def test_free_text(self):
        long_text = "This is a long description of a product that contains many words and details"
        top = [TopValueEntry(value=long_text, count=1, percentage=1.0) for _ in range(5)]
        pattern, score = classify_value_pattern(top)
        assert pattern == "free_text"
        assert score <= 0.2

    def test_categorical(self):
        top = [TopValueEntry(value=v, count=100, percentage=20.0) for v in ["Active", "Inactive", "Pending", "Closed", "Draft"]]
        pattern, score = classify_value_pattern(top)
        assert pattern == "categorical"
        assert score >= 0.7

    def test_empty(self):
        pattern, score = classify_value_pattern(None)
        assert pattern == "unknown"

    def test_none_values(self):
        top = [TopValueEntry(value=None, count=100, percentage=100.0)]
        pattern, score = classify_value_pattern(top)
        assert pattern == "unknown"


# ── Full signal computation ───────────────────────────────────────────────────


class TestComputeStatisticalSignals:
    def _make_column(self, name: str, data_type: str, stats: ColumnStatistics | None = None) -> ColumnMetadata:
        return ColumnMetadata(
            name=name,
            ordinal_position=1,
            data_type=data_type,
            is_nullable=True,
            column_default=None,
            character_maximum_length=None,
            numeric_precision=None,
            numeric_scale=None,
            statistics=stats,
        )

    def test_with_stats(self):
        stats = ColumnStatistics(
            total_count=10000,
            null_count=100,
            null_percentage=1.0,
            distinct_count=5,
            distinct_percentage=0.05,
        )
        col = self._make_column("order_status", "varchar(50)", stats)
        signals = compute_statistical_signals(col, 10000)

        assert signals.cardinality_bucket == BUCKET_LOW_CATEGORICAL
        assert signals.cardinality_score == 0.95
        assert signals.dtype_score == 0.80
        assert signals.naming_score == 0.80
        assert signals.null_penalty == 1.0

    def test_without_stats(self):
        col = self._make_column("amount", "decimal(18,2)")
        signals = compute_statistical_signals(col, 10000)

        assert signals.cardinality_bucket == BUCKET_NEAR_UNIQUE
        assert signals.distinct_count == 0
