"""Stage 2 — Statistical signal computation for filter column detection.

All functions are pure and operate on data already present in the
profiling response (ColumnMetadata, ColumnStatistics).  No database
queries are issued by this module.
"""

from __future__ import annotations

import re

from core.api.v1.schemas.profiler import ColumnMetadata, ColumnStatistics, TopValueEntry
from core.services.filters.models import StatisticalSignals

# ── Cardinality Bucketing ─────────────────────────────────────────────────────

BUCKET_CONSTANT = "constant"
BUCKET_BOOLEAN = "boolean"
BUCKET_LOW_CATEGORICAL = "low_categorical"
BUCKET_HIGH_CATEGORICAL = "high_categorical"
BUCKET_HIGH_CARDINALITY = "high_cardinality"
BUCKET_NEAR_UNIQUE = "near_unique"

_CARDINALITY_SCORES: dict[str, float] = {
    BUCKET_CONSTANT: 0.0,
    BUCKET_BOOLEAN: 0.85,
    BUCKET_LOW_CATEGORICAL: 0.95,
    BUCKET_HIGH_CATEGORICAL: 0.70,
    BUCKET_HIGH_CARDINALITY: 0.30,
    BUCKET_NEAR_UNIQUE: 0.05,
}


def compute_cardinality_bucket(distinct_count: int, row_count: int) -> str:
    if row_count == 0 or distinct_count == 0:
        return BUCKET_CONSTANT

    if distinct_count == 1:
        return BUCKET_CONSTANT
    if distinct_count == 2:
        return BUCKET_BOOLEAN
    if distinct_count <= 50:
        return BUCKET_LOW_CATEGORICAL
    if distinct_count <= 500:
        return BUCKET_HIGH_CATEGORICAL
    if distinct_count <= 10_000 or distinct_count <= row_count * 0.10:
        return BUCKET_HIGH_CARDINALITY
    return BUCKET_NEAR_UNIQUE


def compute_cardinality_score(bucket: str) -> float:
    return _CARDINALITY_SCORES.get(bucket, 0.05)


# ── Data Type Scoring ─────────────────────────────────────────────────────────

_TEMPORAL_TYPES = frozenset(
    {
        "date",
        "time",
        "timestamp",
        "datetime",
        "year",
        "timestamp with time zone",
        "timestamp without time zone",
        "timestamp_ntz",
        "timestamp_ltz",
        "timestamp_tz",
        "timestamptz",
        "timetz",
    }
)

_BOOLEAN_TYPES = frozenset({"boolean", "bool"})

_TEXT_TYPES = frozenset(
    {
        "character varying",
        "varchar",
        "character",
        "char",
        "text",
        "name",
        "citext",
        "bpchar",
        "tinytext",
        "mediumtext",
        "longtext",
        "string",
    }
)

_FLOAT_TYPES = frozenset(
    {
        "float",
        "double",
        "double precision",
        "real",
        "decimal",
        "numeric",
        "money",
        "bignumeric",
        "float64",
    }
)

_INTEGER_TYPES = frozenset(
    {
        "smallint",
        "integer",
        "bigint",
        "int",
        "serial",
        "tinyint",
        "mediumint",
        "int64",
        "long",
        "short",
        "byte",
        "hugeint",
        "uhugeint",
        "ubigint",
        "uinteger",
        "usmallint",
        "utinyint",
    }
)


def _normalize_type(data_type: str) -> str:
    """Lowercase, strip precision/length annotations."""
    dt = data_type.lower().strip()
    # Remove parenthesised precision: "decimal(18,4)" → "decimal"
    dt = re.sub(r"\(.*\)", "", dt).strip()
    return dt


def compute_dtype_score(
    data_type: str,
    cardinality_bucket: str,
    is_primary_key: bool,
) -> float:
    dt = _normalize_type(data_type)

    if dt in _TEMPORAL_TYPES:
        return 0.90
    if dt in _BOOLEAN_TYPES:
        return 0.90
    if dt in _TEXT_TYPES:
        if cardinality_bucket in (BUCKET_LOW_CATEGORICAL, BUCKET_HIGH_CATEGORICAL, BUCKET_BOOLEAN):
            return 0.80
        return 0.15
    if dt in _INTEGER_TYPES:
        if is_primary_key:
            return 0.05
        if cardinality_bucket in (BUCKET_LOW_CATEGORICAL, BUCKET_HIGH_CATEGORICAL, BUCKET_BOOLEAN):
            return 0.75
        return 0.10
    if dt in _FLOAT_TYPES:
        return 0.05
    # Unknown types — neutral
    return 0.30


# ── Null Ratio Penalty ────────────────────────────────────────────────────────


def compute_null_penalty(null_ratio: float) -> float:
    if null_ratio < 0.05:
        return 1.0
    if null_ratio < 0.30:
        return 0.9
    if null_ratio < 0.60:
        return 0.7
    if null_ratio < 0.80:
        return 0.4
    return 0.1


# ── Column Name Pattern Matching ──────────────────────────────────────────────

_STRONG_FILTER_RE = re.compile(
    r"(?:_(?:status|type|category|code|class|group|tier|level|flag|mode|state"
    r"|region|country|city|department|channel))$"
    r"|^(?:is|has)_",
    re.IGNORECASE,
)

_TEMPORAL_NAME_RE = re.compile(
    r"(?:_(?:date|time|at|on|year|month|quarter|week))$"
    r"|^(?:date|time|fiscal)_",
    re.IGNORECASE,
)

_ANTI_FILTER_RE = re.compile(
    r"(?:_(?:description|notes|comment|text|blob|json|xml|hash|token|uuid))$",
    re.IGNORECASE,
)

_AUDIT_RE = re.compile(
    r"^(?:created|updated|modified|inserted|deleted)_(?:at|on|date|time|timestamp)$"
    r"|^last_login$"
    r"|_timestamp$",
    re.IGNORECASE,
)

_ID_NAME_RE = re.compile(r"_id$|^id$", re.IGNORECASE)
_NAME_HIGH_CARD_RE = re.compile(r"_name$|^name$", re.IGNORECASE)


def compute_naming_score(column_name: str, cardinality_bucket: str = BUCKET_NEAR_UNIQUE) -> float:
    name = column_name.strip()

    # Audit patterns take priority — they look temporal but aren't analytical filters
    if _AUDIT_RE.search(name):
        return 0.10

    if _ANTI_FILTER_RE.search(name):
        return 0.05

    # ID columns with high cardinality are anti-filter
    if _ID_NAME_RE.search(name) and cardinality_bucket in (BUCKET_HIGH_CARDINALITY, BUCKET_NEAR_UNIQUE):
        return 0.05
    # Name columns with high cardinality are anti-filter
    if _NAME_HIGH_CARD_RE.search(name) and cardinality_bucket in (BUCKET_HIGH_CARDINALITY, BUCKET_NEAR_UNIQUE):
        return 0.05

    if _TEMPORAL_NAME_RE.search(name):
        return 0.85

    if _STRONG_FILTER_RE.search(name):
        return 0.80

    # Neutral — no name signal
    return 0.50


# ── Value Pattern Analysis ────────────────────────────────────────────────────

_CODE_RE = re.compile(r"^[A-Z0-9_\-]{2,5}$")
_ISO_CODE_RE = re.compile(r"^[A-Z]{2,3}$")
_IDENTIFIER_RE = re.compile(r"^[A-Z]{2,5}[-_]\d+", re.IGNORECASE)


def classify_value_pattern(top_values: list[TopValueEntry] | None) -> tuple[str, float]:
    """Classify string value patterns from top-value frequency data.

    Returns:
        A tuple of (pattern_name, score).
    """
    if not top_values:
        return ("unknown", 0.50)

    values = [tv.value for tv in top_values if tv.value is not None]
    if not values:
        return ("unknown", 0.50)

    total = len(values)

    # Check for ISO-like codes (2-3 uppercase chars)
    iso_matches = sum(1 for v in values if _ISO_CODE_RE.match(v))
    if iso_matches / total >= 0.8:
        return ("code", 0.90)

    # Check for short uppercase codes
    code_matches = sum(1 for v in values if _CODE_RE.match(v))
    if code_matches / total >= 0.8:
        return ("code", 0.80)

    # Check for identifier patterns (e.g., ORD-001, INV-2024)
    id_matches = sum(1 for v in values if _IDENTIFIER_RE.match(v))
    if id_matches / total >= 0.5:
        return ("identifier", 0.10)

    # Check for free text (long mixed-case strings)
    long_mixed = sum(1 for v in values if len(v) > 30 and not v.isupper())
    if long_mixed / total >= 0.5:
        return ("free_text", 0.10)

    # Consistent formatting with short values suggests categorical
    avg_len = sum(len(v) for v in values) / total
    if avg_len <= 25:
        return ("categorical", 0.80)

    return ("mixed", 0.50)


# ── Orchestrator ──────────────────────────────────────────────────────────────


def compute_statistical_signals(
    column: ColumnMetadata,
    row_count: int,
) -> StatisticalSignals:
    """Compute all Stage 2 signals for a single column."""
    stats: ColumnStatistics | None = column.statistics

    # When column stats are unavailable, infer what we can from metadata
    if stats is None:
        if column.enum_values:
            distinct_count = len(column.enum_values)
            cardinality_bucket = compute_cardinality_bucket(distinct_count, row_count)
        else:
            distinct_count = 0
            cardinality_bucket = BUCKET_NEAR_UNIQUE
        null_ratio = 0.0
    else:
        distinct_count = stats.distinct_count
        null_ratio = stats.null_percentage / 100.0 if stats.null_percentage else 0.0
        cardinality_bucket = compute_cardinality_bucket(distinct_count, row_count)

    repetition_factor = (row_count / distinct_count) if distinct_count > 0 else 0.0
    cardinality_score = compute_cardinality_score(cardinality_bucket)
    dtype_score = compute_dtype_score(column.data_type, cardinality_bucket, column.is_primary_key)
    null_penalty = compute_null_penalty(null_ratio)
    naming_score = compute_naming_score(column.name, cardinality_bucket)

    # Value pattern from top_values (only for string-like columns with stats)
    value_pattern = "unknown"
    value_pattern_score = 0.50
    if stats and stats.top_values:
        value_pattern, value_pattern_score = classify_value_pattern(stats.top_values)

    return StatisticalSignals(
        distinct_count=distinct_count,
        row_count=row_count,
        repetition_factor=repetition_factor,
        null_ratio=null_ratio,
        cardinality_bucket=cardinality_bucket,
        cardinality_score=cardinality_score,
        dtype_score=dtype_score,
        naming_score=naming_score,
        null_penalty=null_penalty,
        value_pattern=value_pattern,
        value_pattern_score=value_pattern_score,
    )
