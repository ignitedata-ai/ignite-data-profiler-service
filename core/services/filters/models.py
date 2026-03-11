"""Internal models for the filter column detection pipeline.

These models carry signals between pipeline stages and are not exposed
in the public API.  The final output is converted to
``FilterColumnInfo`` (defined in the response schemas).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Stage 1: Schema Signals ──────────────────────────────────────────────────


class SchemaSignals(BaseModel):
    """Schema-level signals extracted for a single column."""

    is_foreign_key: bool = False
    fk_referenced_table: str | None = None
    fk_referenced_table_row_count: int | None = None

    is_non_unique_index: bool = False
    composite_index_partners: list[str] = Field(default_factory=list)

    is_primary_key: bool = False
    is_composite_pk_member: bool = False
    pk_column_count: int = 0

    has_check_constraint: bool = False
    check_constraint_values: list[str] | None = None

    has_enum_type: bool = False

    schema_score: float = 0.0


# ── Stage 2: Statistical Signals ─────────────────────────────────────────────


class StatisticalSignals(BaseModel):
    """Data-level signals computed from column statistics."""

    distinct_count: int = 0
    row_count: int = 0
    repetition_factor: float = 0.0
    null_ratio: float = 0.0

    cardinality_bucket: str = "near_unique"
    cardinality_score: float = 0.0
    dtype_score: float = 0.0
    naming_score: float = 0.0
    null_penalty: float = 1.0

    value_pattern: str | None = None
    value_pattern_score: float = 0.0


# ── Pipeline Candidate ───────────────────────────────────────────────────────


class ColumnFilterCandidate(BaseModel):
    """Aggregated scoring result for a single column across all stages."""

    column_name: str
    data_type: str
    is_primary_key: bool = False

    schema_signals: SchemaSignals | None = None
    statistical_signals: StatisticalSignals

    composite_score: float = 0.0
    preliminary_filter_type: str = "categorical"
    in_review_band: bool = False
