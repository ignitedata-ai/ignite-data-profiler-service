"""Stage 4 (LLM part) — Filter column judge and reconciliation.

Builds the payload sent to the LLM and reconciles heuristic vs LLM
scores according to the agreement/disagreement rules.
"""

from __future__ import annotations

from core.services.filters.models import ColumnFilterCandidate


def build_llm_payload(
    table_name: str,
    table_role: str,
    row_count: int | None,
    candidates: list[ColumnFilterCandidate],
) -> list[dict]:
    """Format candidates as dicts suitable for the LLM prompt."""
    items: list[dict] = []
    for c in candidates:
        item: dict = {
            "column_name": c.column_name,
            "data_type": c.data_type,
            "composite_score": round(c.composite_score, 3),
            "preliminary_filter_type": c.preliminary_filter_type,
            "cardinality_bucket": c.statistical_signals.cardinality_bucket,
            "distinct_count": c.statistical_signals.distinct_count,
            "null_ratio": round(c.statistical_signals.null_ratio, 3),
            "naming_score": round(c.statistical_signals.naming_score, 2),
            "dtype_score": round(c.statistical_signals.dtype_score, 2),
            "cardinality_score": round(c.statistical_signals.cardinality_score, 2),
        }
        if c.schema_signals:
            item["schema_score"] = round(c.schema_signals.schema_score, 2)
            item["is_foreign_key"] = c.schema_signals.is_foreign_key
            item["has_enum_type"] = c.schema_signals.has_enum_type
            item["has_check_constraint"] = c.schema_signals.has_check_constraint
        if c.statistical_signals.value_pattern and c.statistical_signals.value_pattern != "unknown":
            item["value_pattern"] = c.statistical_signals.value_pattern
        items.append(item)
    return items


def reconcile(
    heuristic_score: float,
    llm_score: float,
    has_schema_signals: bool,
) -> tuple[float, str]:
    """Merge heuristic and LLM scores.

    Returns:
        (final_score, confidence_source)
    """
    diff = abs(heuristic_score - llm_score)

    if diff < 0.2:
        # Agreement
        final = (heuristic_score + llm_score) / 2
        return round(final, 4), "llm_agreed"

    if diff <= 0.4:
        # Mild disagreement — bias depends on schema signal presence
        if has_schema_signals:
            final = 0.7 * heuristic_score + 0.3 * llm_score
        else:
            final = 0.4 * heuristic_score + 0.6 * llm_score
        return round(final, 4), "llm_adjusted"

    # Strong disagreement — average, flag for review
    final = (heuristic_score + llm_score) / 2
    return round(final, 4), "flagged_for_review"
