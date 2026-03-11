"""Tests for composite scoring, triage, and filter type classification."""

from __future__ import annotations

from core.services.filters.composite_scorer import (
    classify_filter_type,
    compute_composite_score,
    triage_candidates,
)
from core.services.filters.models import ColumnFilterCandidate, SchemaSignals, StatisticalSignals


def _make_candidate(
    name: str = "col",
    data_type: str = "varchar",
    schema_score: float = 0.0,
    cardinality_score: float = 0.5,
    dtype_score: float = 0.5,
    naming_score: float = 0.5,
    null_penalty: float = 1.0,
    cardinality_bucket: str = "low_categorical",
    distinct_count: int = 10,
    has_schema: bool = True,
) -> ColumnFilterCandidate:
    schema_signals = SchemaSignals(schema_score=schema_score) if has_schema else None
    stat_signals = StatisticalSignals(
        cardinality_score=cardinality_score,
        dtype_score=dtype_score,
        naming_score=naming_score,
        null_penalty=null_penalty,
        cardinality_bucket=cardinality_bucket,
        distinct_count=distinct_count,
    )
    composite = compute_composite_score(schema_signals, stat_signals)
    return ColumnFilterCandidate(
        column_name=name,
        data_type=data_type,
        schema_signals=schema_signals,
        statistical_signals=stat_signals,
        composite_score=composite,
    )


class TestCompositeScore:
    def test_all_high_signals(self):
        c = _make_candidate(schema_score=0.95, cardinality_score=0.95, dtype_score=0.90, naming_score=0.80)
        assert c.composite_score > 0.85

    def test_all_low_signals(self):
        c = _make_candidate(schema_score=0.0, cardinality_score=0.05, dtype_score=0.05, naming_score=0.05, has_schema=False)
        assert c.composite_score < 0.15

    def test_null_penalty_applied(self):
        c_no_penalty = _make_candidate(null_penalty=1.0)
        c_high_null = _make_candidate(null_penalty=0.1)
        assert c_high_null.composite_score < c_no_penalty.composite_score

    def test_no_schema_signals(self):
        c = _make_candidate(has_schema=False, cardinality_score=0.95, dtype_score=0.80, naming_score=0.80)
        # Should still produce a reasonable score from statistical signals alone
        assert c.composite_score > 0.4


class TestFilterTypeClassification:
    def test_temporal(self):
        c = _make_candidate(data_type="timestamp")
        c.preliminary_filter_type = classify_filter_type(c)
        assert c.preliminary_filter_type == "temporal"

    def test_boolean(self):
        c = _make_candidate(data_type="boolean", cardinality_bucket="boolean")
        c.preliminary_filter_type = classify_filter_type(c)
        assert c.preliminary_filter_type == "boolean"

    def test_categorical_varchar(self):
        c = _make_candidate(data_type="varchar", cardinality_bucket="low_categorical")
        c.preliminary_filter_type = classify_filter_type(c)
        assert c.preliminary_filter_type == "categorical"

    def test_range_integer_high_cardinality(self):
        c = _make_candidate(data_type="integer", cardinality_bucket="high_cardinality")
        c.preliminary_filter_type = classify_filter_type(c)
        assert c.preliminary_filter_type == "range"



class TestTriage:
    def test_auto_accept_high_score_with_schema(self):
        c = _make_candidate(schema_score=0.95, cardinality_score=0.95, dtype_score=0.90, naming_score=0.85)
        accept, reject, review = triage_candidates([c])
        assert len(accept) == 1
        assert len(review) == 0

    def test_auto_reject_low_score(self):
        c = _make_candidate(schema_score=0.0, cardinality_score=0.0, dtype_score=0.05, naming_score=0.05, has_schema=False)
        accept, reject, review = triage_candidates([c])
        assert len(reject) == 1
        assert len(accept) == 0

    def test_review_band_medium_score(self):
        c = _make_candidate(schema_score=0.5, cardinality_score=0.5, dtype_score=0.5, naming_score=0.5)
        accept, reject, review = triage_candidates([c])
        assert len(review) == 1

    def test_high_score_without_schema_goes_to_review(self):
        c = _make_candidate(has_schema=False, cardinality_score=0.95, dtype_score=0.90, naming_score=0.85)
        accept, reject, review = triage_candidates([c])
        # High score but no schema signal → review band
        assert len(review) == 1
        assert len(accept) == 0
