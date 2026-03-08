"""Stage 1 — Schema signal extraction for filter column detection.

Most signals are derived from data already present in ``TableMetadata``
(indexes, relationships, primary keys, enum values).  Only CHECK
constraints and referenced-table row counts require additional queries.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from core.api.v1.schemas.profiler import TableMetadata
from core.logging import get_logger
from core.services.column_stats import ColumnTypeCategory, classify_column_type
from core.services.filters.models import SchemaSignals

logger = get_logger(__name__)

FK_SMALL_DIM_THRESHOLD = 500


class BaseSchemaIntrospector(ABC):
    """Extract schema-level signals for filter column detection."""

    # ── Shared logic (operates on already-fetched TableMetadata) ──────────

    def extract_signals(self, table: TableMetadata) -> dict[str, SchemaSignals]:
        """Build SchemaSignals for every column from existing metadata."""
        signals: dict[str, SchemaSignals] = {}

        # Index column sets
        pk_columns = self._pk_columns(table)
        fk_map = self._fk_map(table)
        index_info = self._index_info(table)

        for col in table.columns:
            sig = SchemaSignals()

            # Primary key
            if col.is_primary_key:
                sig.is_primary_key = True
                sig.pk_column_count = len(pk_columns)
                sig.is_composite_pk_member = len(pk_columns) > 1

            # Foreign key
            if col.name in fk_map:
                sig.is_foreign_key = True
                sig.fk_referenced_table = fk_map[col.name]

            # Indexes
            if col.name in index_info:
                info = index_info[col.name]
                sig.is_non_unique_index = info["non_unique"]
                sig.composite_index_partners = info["partners"]

            # Enum type
            if col.enum_values and len(col.enum_values) > 0:
                sig.has_enum_type = True

            signals[col.name] = sig

        return signals

    def score_signals(
        self,
        signals: dict[str, SchemaSignals],
        check_constraints: dict[str, list[str]] | None = None,
    ) -> None:
        """Compute schema_score for each column in-place."""
        check_constraints = check_constraints or {}

        for col_name, sig in signals.items():
            if col_name in check_constraints:
                sig.has_check_constraint = True
                sig.check_constraint_values = check_constraints[col_name]

            scores: list[float] = []

            if sig.has_check_constraint or sig.has_enum_type:
                scores.append(0.95)

            if sig.is_foreign_key:
                if sig.fk_referenced_table_row_count is not None and sig.fk_referenced_table_row_count < FK_SMALL_DIM_THRESHOLD:
                    scores.append(0.9)
                else:
                    scores.append(0.6)

            if sig.is_non_unique_index:
                scores.append(0.7)

            if sig.composite_index_partners:
                scores.append(0.5)

            if sig.is_primary_key and sig.pk_column_count == 1:
                scores.append(0.05)

            sig.schema_score = max(scores) if scores else 0.0

    def classify_table_role(self, table: TableMetadata) -> str:
        """Classify a table as fact, dimension, or unknown."""
        row_count = table.row_count or 0
        fk_count = len(table.relationships) if table.relationships else 0

        numeric_cols = 0
        string_cols = 0
        for col in table.columns:
            cat = classify_column_type(col.data_type)
            if cat == ColumnTypeCategory.NUMERIC:
                numeric_cols += 1
            elif cat == ColumnTypeCategory.STRING:
                string_cols += 1

        # Fact table heuristics
        if row_count > 10_000 and fk_count >= 2 and numeric_cols >= 2:
            return "fact"
        # Dimension table heuristics
        if row_count < 5_000 and fk_count == 0 and string_cols >= 2:
            return "dimension"

        return "unknown"

    # ── Engine-specific abstract methods ──────────────────────────────────

    @abstractmethod
    async def fetch_check_constraints(
        self,
        connector: Any,
        schema_name: str,
        table_name: str,
    ) -> dict[str, list[str]]:
        """Return {column_name: [allowed_values]} for CHECK constraints."""

    @abstractmethod
    async def fetch_referenced_table_row_count(
        self,
        connector: Any,
        schema_name: str,
        table_name: str,
    ) -> int | None:
        """Return approximate row count for a referenced table."""

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _pk_columns(table: TableMetadata) -> set[str]:
        return {c.name for c in table.columns if c.is_primary_key}

    @staticmethod
    def _fk_map(table: TableMetadata) -> dict[str, str]:
        """Map from_column → 'to_schema.to_table'."""
        if not table.relationships:
            return {}
        return {r.from_column: f"{r.to_schema}.{r.to_table}" for r in table.relationships}

    @staticmethod
    def _index_info(table: TableMetadata) -> dict[str, dict]:
        """Build per-column index metadata."""
        result: dict[str, dict] = {}
        if not table.indexes:
            return result

        for idx in table.indexes:
            if idx.is_primary:
                continue
            for col in idx.columns:
                entry = result.setdefault(col, {"non_unique": False, "partners": []})
                if not idx.is_unique:
                    entry["non_unique"] = True
                # Partners are the other columns in the same index
                partners = [c for c in idx.columns if c != col]
                for p in partners:
                    if p not in entry["partners"]:
                        entry["partners"].append(p)
        return result


# ── PostgreSQL ────────────────────────────────────────────────────────────────

_PG_CHECK_CONSTRAINTS = """
SELECT
    a.attname AS column_name,
    pg_get_constraintdef(c.oid) AS definition
FROM pg_constraint c
JOIN pg_attribute a ON a.attnum = ANY(c.conkey) AND a.attrelid = c.conrelid
WHERE c.conrelid = '{schema}.{table}'::regclass
  AND c.contype = 'c'
"""

_PG_ROW_COUNT = """
SELECT reltuples::bigint AS approx_rows
FROM pg_class
WHERE oid = '{schema}.{table}'::regclass
"""


def _parse_check_values(definition: str) -> list[str]:
    """Extract values from a CHECK constraint like: CHECK ((status = ANY (ARRAY[...])))."""
    # Match quoted strings inside the definition
    return re.findall(r"'([^']*)'", definition)


class PostgresSchemaIntrospector(BaseSchemaIntrospector):
    async def fetch_check_constraints(
        self,
        connector: Any,
        schema_name: str,
        table_name: str,
    ) -> dict[str, list[str]]:
        try:
            query = _PG_CHECK_CONSTRAINTS.format(schema=schema_name, table=table_name)
            rows = await connector.fetch_all(query)
            result: dict[str, list[str]] = {}
            for row in rows:
                values = _parse_check_values(row["definition"])
                if values:
                    result[row["column_name"]] = values
            return result
        except Exception as exc:
            logger.debug("Could not fetch CHECK constraints", error=str(exc))
            return {}

    async def fetch_referenced_table_row_count(
        self,
        connector: Any,
        schema_name: str,
        table_name: str,
    ) -> int | None:
        try:
            query = _PG_ROW_COUNT.format(schema=schema_name, table=table_name)
            row = await connector.fetch_one(query)
            return int(row["approx_rows"]) if row else None
        except Exception:
            return None


# ── MySQL ─────────────────────────────────────────────────────────────────────

_MYSQL_CHECK_CONSTRAINTS = """
SELECT
    cc.constraint_name,
    cc.check_clause
FROM information_schema.check_constraints cc
JOIN information_schema.table_constraints tc
  ON tc.constraint_name = cc.constraint_name
  AND tc.constraint_schema = cc.constraint_schema
WHERE tc.table_schema = %s AND tc.table_name = %s
  AND tc.constraint_type = 'CHECK'
"""

_MYSQL_ROW_COUNT = """
SELECT table_rows AS approx_rows
FROM information_schema.tables
WHERE table_schema = %s AND table_name = %s
"""

_MYSQL_CHECK_COLUMN_RE = re.compile(r"`(\w+)`")


class MySQLSchemaIntrospector(BaseSchemaIntrospector):
    async def fetch_check_constraints(
        self,
        connector: Any,
        schema_name: str,
        table_name: str,
    ) -> dict[str, list[str]]:
        try:
            rows = await connector.fetch_all(
                _MYSQL_CHECK_CONSTRAINTS,
                (schema_name, table_name),
            )
            result: dict[str, list[str]] = {}
            for row in rows:
                clause = row["check_clause"] or ""
                col_match = _MYSQL_CHECK_COLUMN_RE.search(clause)
                if col_match:
                    col_name = col_match.group(1)
                    values = re.findall(r"'([^']*)'", clause)
                    if values:
                        result[col_name] = values
            return result
        except Exception as exc:
            logger.debug("Could not fetch CHECK constraints", error=str(exc))
            return {}

    async def fetch_referenced_table_row_count(
        self,
        connector: Any,
        schema_name: str,
        table_name: str,
    ) -> int | None:
        try:
            row = await connector.fetch_one(
                _MYSQL_ROW_COUNT,
                (schema_name, table_name),
            )
            return int(row["approx_rows"]) if row and row["approx_rows"] is not None else None
        except Exception:
            return None


# ── Snowflake ─────────────────────────────────────────────────────────────────

_SNOWFLAKE_ROW_COUNT = """
SELECT ROW_COUNT AS "approx_rows"
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
"""


class SnowflakeSchemaIntrospector(BaseSchemaIntrospector):
    async def fetch_check_constraints(
        self,
        connector: Any,
        schema_name: str,
        table_name: str,
    ) -> dict[str, list[str]]:
        # Snowflake doesn't expose CHECK constraint details via INFORMATION_SCHEMA
        return {}

    async def fetch_referenced_table_row_count(
        self,
        connector: Any,
        schema_name: str,
        table_name: str,
    ) -> int | None:
        try:
            row = await connector.fetch_one(
                _SNOWFLAKE_ROW_COUNT,
                (schema_name, table_name),
            )
            return int(row["approx_rows"]) if row and row["approx_rows"] is not None else None
        except Exception:
            return None


# ── Null (S3/DuckDB, BigQuery, Databricks) ────────────────────────────────────


class NullSchemaIntrospector(BaseSchemaIntrospector):
    """No-op introspector for datasources without schema metadata."""

    async def fetch_check_constraints(
        self,
        connector: Any,
        schema_name: str,
        table_name: str,
    ) -> dict[str, list[str]]:
        return {}

    async def fetch_referenced_table_row_count(
        self,
        connector: Any,
        schema_name: str,
        table_name: str,
    ) -> int | None:
        return None


# ── Registry ──────────────────────────────────────────────────────────────────

INTROSPECTOR_REGISTRY: dict[str, type[BaseSchemaIntrospector]] = {
    "postgres": PostgresSchemaIntrospector,
    "mysql": MySQLSchemaIntrospector,
    "snowflake": SnowflakeSchemaIntrospector,
    "redshift": PostgresSchemaIntrospector,  # Redshift is PG-compatible
    "bigquery": NullSchemaIntrospector,
    "databricks": NullSchemaIntrospector,
    "s3_file": NullSchemaIntrospector,
}
