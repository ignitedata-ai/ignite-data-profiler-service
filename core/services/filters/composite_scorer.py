"""Composite scoring, triage, and filter type classification.

Combines schema and statistical signals into a single composite score
per column, then triages columns into auto-accept / auto-reject /
LLM-review buckets.
"""

from __future__ import annotations

from core.services.filters.models import ColumnFilterCandidate, SchemaSignals, StatisticalSignals
from core.services.filters.statistical_scorer import (
    _TEMPORAL_TYPES,
    BUCKET_BOOLEAN,
    BUCKET_HIGH_CATEGORICAL,
    BUCKET_LOW_CATEGORICAL,
    _normalize_type,
)

# ── Weights (ANOVA deferred — redistributed across remaining signals) ────────

W_SCHEMA = 0.35
W_CARDINALITY = 0.30
W_DTYPE = 0.20
W_NAMING = 0.15

# ── Thresholds ────────────────────────────────────────────────────────────────

AUTO_ACCEPT_THRESHOLD = 0.85
AUTO_REJECT_THRESHOLD = 0.15
FINAL_INCLUDE_THRESHOLD = 0.50


def compute_composite_score(
    schema_signals: SchemaSignals | None,
    statistical_signals: StatisticalSignals,
) -> float:
    schema_score = schema_signals.schema_score if schema_signals else 0.0

    raw = (
        W_SCHEMA * schema_score
        + W_CARDINALITY * statistical_signals.cardinality_score
        + W_DTYPE * statistical_signals.dtype_score
        + W_NAMING * statistical_signals.naming_score
    )
    return round(raw * statistical_signals.null_penalty, 4)


# ── Filter Type Classification ────────────────────────────────────────────────


def classify_filter_type(candidate: ColumnFilterCandidate) -> str:
    dt = _normalize_type(candidate.data_type)

    if dt in _TEMPORAL_TYPES:
        return "temporal"

    bucket = candidate.statistical_signals.cardinality_bucket
    if dt in ("boolean", "bool") or bucket == BUCKET_BOOLEAN:
        return "boolean"

    if dt in ("integer", "int", "bigint", "smallint", "float", "double", "decimal", "numeric", "real"):
        if bucket in (BUCKET_HIGH_CATEGORICAL, BUCKET_LOW_CATEGORICAL):
            return "categorical"
        return "range"

    return "categorical"



# ── Triage ────────────────────────────────────────────────────────────────────


def triage_candidates(
    candidates: list[ColumnFilterCandidate],
) -> tuple[list[ColumnFilterCandidate], list[ColumnFilterCandidate], list[ColumnFilterCandidate]]:
    """Split candidates into (auto_accept, auto_reject, review_band)."""
    auto_accept: list[ColumnFilterCandidate] = []
    auto_reject: list[ColumnFilterCandidate] = []
    review_band: list[ColumnFilterCandidate] = []

    for c in candidates:
        has_schema = c.schema_signals is not None and c.schema_signals.schema_score > 0
        if c.composite_score > AUTO_ACCEPT_THRESHOLD and has_schema:
            c.in_review_band = False
            auto_accept.append(c)
        elif c.composite_score < AUTO_REJECT_THRESHOLD:
            c.in_review_band = False
            auto_reject.append(c)
        else:
            c.in_review_band = True
            review_band.append(c)

    return auto_accept, auto_reject, review_band
