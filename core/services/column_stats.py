"""Column data-type classification for statistical analysis.

Used by both PostgresProfiler and MySQLProfiler to decide which
statistical queries to run for each column.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ColumnTypeCategory(str, Enum):
    NUMERIC = "numeric"
    STRING = "string"
    BOOLEAN = "boolean"
    TEMPORAL = "temporal"
    OTHER = "other"


_NUMERIC_TYPES: frozenset[str] = frozenset(
    {
        "smallint",
        "integer",
        "bigint",
        "int",
        "int2",
        "int4",
        "int8",
        "decimal",
        "numeric",
        "real",
        "double precision",
        "float",
        "float4",
        "float8",
        "money",
        "smallserial",
        "serial",
        "bigserial",
        "tinyint",
        "mediumint",
        "double",
        "number",
        # BigQuery-specific types
        "int64",
        "float64",
        "bignumeric",
        # Databricks aliases
        "long",
        "short",
        "byte",
        # DuckDB-native types
        "hugeint",
        "uhugeint",
        "ubigint",
        "uinteger",
        "usmallint",
        "utinyint",
    }
)

_STRING_TYPES: frozenset[str] = frozenset(
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

_BOOLEAN_TYPES: frozenset[str] = frozenset({"boolean", "bool"})

_TEMPORAL_TYPES: frozenset[str] = frozenset(
    {
        "date",
        "time",
        "time with time zone",
        "time without time zone",
        "timestamp",
        "timestamp with time zone",
        "timestamp without time zone",
        "datetime",
        "year",
        "interval",
        "timestamp_ntz",
        "timestamp_ltz",
        "timestamp_tz",
        # DuckDB aliases
        "timestamptz",
        "timetz",
    }
)


def classify_column_type(data_type: str) -> ColumnTypeCategory:
    """Classify a SQL data type string into a statistical category."""
    dt = data_type.lower().strip()
    # Strip precision/scale annotation e.g. "decimal(18,4)" → "decimal", "varchar(255)" → "varchar"
    base_dt = dt.split("(")[0].strip()
    if base_dt in _NUMERIC_TYPES:
        return ColumnTypeCategory.NUMERIC
    if base_dt in _STRING_TYPES:
        return ColumnTypeCategory.STRING
    if base_dt in _BOOLEAN_TYPES:
        return ColumnTypeCategory.BOOLEAN
    if base_dt in _TEMPORAL_TYPES:
        return ColumnTypeCategory.TEMPORAL
    return ColumnTypeCategory.OTHER


def classify_columns(
    columns: list[dict[str, Any]],
) -> dict[ColumnTypeCategory, list[dict[str, Any]]]:
    """Group columns by their type category.

    Args:
        columns: List of column dicts (must have ``data_type`` key).

    Returns:
        Mapping from category to list of column dicts in that category.

    """
    result: dict[ColumnTypeCategory, list[dict[str, Any]]] = {cat: [] for cat in ColumnTypeCategory}
    for col in columns:
        cat = classify_column_type(col["data_type"])
        result[cat].append(col)
    return result
