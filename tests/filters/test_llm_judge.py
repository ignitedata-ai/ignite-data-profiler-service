"""Tests for the LLM judge reconciliation logic."""

from __future__ import annotations

import pytest

from core.services.filters.llm_judge import build_llm_payload, reconcile
from core.services.filters.models import ColumnFilterCandidate, SchemaSignals, StatisticalSignals


class TestReconcile:
    def test_agreement(self):
        score, source = reconcile(0.7, 0.8, has_schema_signals=True)
        assert source == "llm_agreed"
        assert score == pytest.approx(0.75, abs=0.01)

    def test_mild_disagreement_with_schema(self):
        score, source = reconcile(0.7, 0.4, has_schema_signals=True)
        assert source == "llm_adjusted"
        # 0.7 * 0.7 + 0.3 * 0.4 = 0.49 + 0.12 = 0.61
        assert score == pytest.approx(0.61, abs=0.01)

    def test_mild_disagreement_without_schema(self):
        score, source = reconcile(0.7, 0.4, has_schema_signals=False)
        assert source == "llm_adjusted"
        # 0.4 * 0.7 + 0.6 * 0.4 = 0.28 + 0.24 = 0.52
        assert score == pytest.approx(0.52, abs=0.01)

    def test_strong_disagreement(self):
        score, source = reconcile(0.9, 0.2, has_schema_signals=True)
        assert source == "flagged_for_review"
        assert score == pytest.approx(0.55, abs=0.01)

    def test_exact_agreement(self):
        score, source = reconcile(0.5, 0.5, has_schema_signals=False)
        assert source == "llm_agreed"
        assert score == 0.5


class TestBuildPayload:
    def test_payload_structure(self):
        candidate = ColumnFilterCandidate(
            column_name="status",
            data_type="varchar",
            schema_signals=SchemaSignals(
                schema_score=0.95,
                has_enum_type=True,
                is_foreign_key=False,
                has_check_constraint=False,
            ),
            statistical_signals=StatisticalSignals(
                cardinality_bucket="low_categorical",
                cardinality_score=0.95,
                dtype_score=0.80,
                naming_score=0.80,
                null_ratio=0.01,
                distinct_count=5,
            ),
            composite_score=0.88,
            preliminary_filter_type="categorical",
        )
        payload = build_llm_payload("public.orders", "fact", 10000, [candidate])

        assert len(payload) == 1
        item = payload[0]
        assert item["column_name"] == "status"
        assert item["composite_score"] == 0.88
        assert item["schema_score"] == 0.95
        assert item["has_enum_type"] is True
        assert item["cardinality_bucket"] == "low_categorical"

    def test_payload_without_schema(self):
        candidate = ColumnFilterCandidate(
            column_name="amount",
            data_type="decimal",
            statistical_signals=StatisticalSignals(
                cardinality_score=0.05,
                dtype_score=0.05,
                naming_score=0.50,
            ),
            composite_score=0.10,
        )
        payload = build_llm_payload("public.orders", "fact", 10000, [candidate])
        item = payload[0]
        assert "schema_score" not in item
