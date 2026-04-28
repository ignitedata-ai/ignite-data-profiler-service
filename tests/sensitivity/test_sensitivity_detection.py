"""Tests for PII/PHI sensitivity detection in LLM client."""

from __future__ import annotations

import pytest

from core.api.v1.schemas.profiler import (
    ColumnMetadata,
    GlossaryTerm,
    KPITerm,
    SensitivityType,
    TableMetadata,
)
from core.llm.base import BaseLLMClient


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_column(
    name: str,
    data_type: str = "varchar(255)",
    description: str | None = None,
    sample_values: list | None = None,
) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        ordinal_position=1,
        data_type=data_type,
        is_nullable=True,
        column_default=None,
        character_maximum_length=255,
        numeric_precision=None,
        numeric_scale=None,
        description=description,
        sample_values=sample_values,
    )


def _make_table(
    name: str = "users",
    schema: str = "public",
    description: str | None = None,
    columns: list[ColumnMetadata] | None = None,
) -> TableMetadata:
    return TableMetadata(
        name=name,
        schema=schema,
        owner="postgres",
        description=description,
        row_count=100,
        size_bytes=None,
        total_size_bytes=None,
        data_freshness=None,
        columns=columns or [_make_column("id", "integer")],
        indexes=None,
        relationships=None,
    )


# ── Test Client Implementations ───────────────────────────────────────────────


class _SensitivitySucceedingClient(BaseLLMClient):
    """Client that successfully detects sensitivity based on column name patterns."""

    provider_name = "test"

    async def _describe_table(self, table: TableMetadata) -> str | None:
        return None

    async def _describe_column(self, column: ColumnMetadata, table: TableMetadata) -> str | None:
        return None

    async def _infer_glossary(self, table: TableMetadata) -> list[GlossaryTerm]:
        return []

    async def _cluster_tables_into_domains(
        self, tables: list[TableMetadata], max_domains: int
    ) -> dict[str, list[str]]:
        return {}

    async def _generate_domain_kpis(
        self, domain_name: str, domain_tables: list[TableMetadata], max_kpis: int
    ) -> list[KPITerm]:
        return []

    async def _judge_filter_columns(
        self, table_name: str, table_role: str, candidates: list[dict]
    ) -> list[dict]:
        return []

    async def _detect_column_sensitivity(
        self, table: TableMetadata, columns: list[ColumnMetadata]
    ) -> dict[str, tuple[bool, SensitivityType | None]]:
        """Detect sensitivity based on simple column name patterns."""
        results: dict[str, tuple[bool, SensitivityType | None]] = {}
        for col in columns:
            name_lower = col.name.lower()
            if "email" in name_lower:
                results[col.name] = (True, SensitivityType.EMAIL)
            elif "ssn" in name_lower or "social_security" in name_lower:
                results[col.name] = (True, SensitivityType.SSN)
            elif "phone" in name_lower:
                results[col.name] = (True, SensitivityType.PHONE_NUMBER)
            elif "address" in name_lower:
                results[col.name] = (True, SensitivityType.ADDRESS)
            elif "name" in name_lower and "user" in name_lower:
                results[col.name] = (True, SensitivityType.NAME)
            else:
                results[col.name] = (False, None)
        return results


class _SensitivityFailingClient(BaseLLMClient):
    """Client that always fails on sensitivity detection."""

    provider_name = "test"

    async def _describe_table(self, table: TableMetadata) -> str | None:
        return None

    async def _describe_column(self, column: ColumnMetadata, table: TableMetadata) -> str | None:
        return None

    async def _infer_glossary(self, table: TableMetadata) -> list[GlossaryTerm]:
        return []

    async def _cluster_tables_into_domains(
        self, tables: list[TableMetadata], max_domains: int
    ) -> dict[str, list[str]]:
        return {}

    async def _generate_domain_kpis(
        self, domain_name: str, domain_tables: list[TableMetadata], max_kpis: int
    ) -> list[KPITerm]:
        return []

    async def _judge_filter_columns(
        self, table_name: str, table_role: str, candidates: list[dict]
    ) -> list[dict]:
        return []

    async def _detect_column_sensitivity(
        self, table: TableMetadata, columns: list[ColumnMetadata]
    ) -> dict[str, tuple[bool, SensitivityType | None]]:
        raise RuntimeError("LLM API error")


class _SensitivityPartialClient(BaseLLMClient):
    """Client that fails for specific tables."""

    provider_name = "test"

    async def _describe_table(self, table: TableMetadata) -> str | None:
        return None

    async def _describe_column(self, column: ColumnMetadata, table: TableMetadata) -> str | None:
        return None

    async def _infer_glossary(self, table: TableMetadata) -> list[GlossaryTerm]:
        return []

    async def _cluster_tables_into_domains(
        self, tables: list[TableMetadata], max_domains: int
    ) -> dict[str, list[str]]:
        return {}

    async def _generate_domain_kpis(
        self, domain_name: str, domain_tables: list[TableMetadata], max_kpis: int
    ) -> list[KPITerm]:
        return []

    async def _judge_filter_columns(
        self, table_name: str, table_role: str, candidates: list[dict]
    ) -> list[dict]:
        return []

    async def _detect_column_sensitivity(
        self, table: TableMetadata, columns: list[ColumnMetadata]
    ) -> dict[str, tuple[bool, SensitivityType | None]]:
        if table.name == "bad":
            raise RuntimeError("LLM API error for bad table")
        # Mark all columns as non-sensitive for successful tables
        return {col.name: (False, None) for col in columns}


# ── detect_sensitivity Tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_detect_sensitivity_updates_columns_in_place():
    """Test that detect_sensitivity updates column objects in place."""
    client = _SensitivitySucceedingClient()

    email_col = _make_column("user_email")
    id_col = _make_column("id", "integer")
    table = _make_table(columns=[email_col, id_col])

    await client.detect_sensitivity([table], batch_size=10)

    # Email column should be marked sensitive
    assert email_col.is_sensitive is True
    assert email_col.sensitivity_type == SensitivityType.EMAIL

    # ID column should remain non-sensitive
    assert id_col.is_sensitive is False
    assert id_col.sensitivity_type is None


@pytest.mark.asyncio
async def test_detect_sensitivity_handles_multiple_sensitive_columns():
    """Test detection of multiple sensitive columns in one table."""
    client = _SensitivitySucceedingClient()

    columns = [
        _make_column("email"),
        _make_column("ssn"),
        _make_column("phone_number"),
        _make_column("created_at"),  # Not sensitive
    ]
    table = _make_table(columns=columns)

    await client.detect_sensitivity([table], batch_size=10)

    assert columns[0].is_sensitive is True
    assert columns[0].sensitivity_type == SensitivityType.EMAIL

    assert columns[1].is_sensitive is True
    assert columns[1].sensitivity_type == SensitivityType.SSN

    assert columns[2].is_sensitive is True
    assert columns[2].sensitivity_type == SensitivityType.PHONE_NUMBER

    assert columns[3].is_sensitive is False
    assert columns[3].sensitivity_type is None


@pytest.mark.asyncio
async def test_detect_sensitivity_handles_multiple_tables():
    """Test detection across multiple tables."""
    client = _SensitivitySucceedingClient()

    users_columns = [_make_column("email"), _make_column("id")]
    users_table = _make_table(name="users", columns=users_columns)

    orders_columns = [_make_column("order_id"), _make_column("amount")]
    orders_table = _make_table(name="orders", columns=orders_columns)

    await client.detect_sensitivity([users_table, orders_table], batch_size=10)

    # Users table - email should be sensitive
    assert users_columns[0].is_sensitive is True
    assert users_columns[0].sensitivity_type == SensitivityType.EMAIL
    assert users_columns[1].is_sensitive is False

    # Orders table - nothing sensitive
    assert orders_columns[0].is_sensitive is False
    assert orders_columns[1].is_sensitive is False


@pytest.mark.asyncio
async def test_detect_sensitivity_handles_empty_tables():
    """Test that empty tables are skipped gracefully."""
    client = _SensitivitySucceedingClient()

    table = _make_table(columns=[])

    # Should not raise
    await client.detect_sensitivity([table], batch_size=10)


@pytest.mark.asyncio
async def test_detect_sensitivity_respects_batch_size():
    """Test that columns are processed in batches."""
    client = _SensitivitySucceedingClient()

    # Create more columns than batch size
    columns = [_make_column(f"col_{i}") for i in range(25)]
    columns.append(_make_column("email"))  # Add one sensitive column
    table = _make_table(columns=columns)

    await client.detect_sensitivity([table], batch_size=10)

    # The email column should still be detected even across batches
    assert columns[-1].is_sensitive is True
    assert columns[-1].sensitivity_type == SensitivityType.EMAIL


@pytest.mark.asyncio
async def test_detect_sensitivity_error_isolation():
    """Test that errors in one batch don't affect other batches."""
    client = _SensitivityFailingClient()

    email_col = _make_column("email")
    table = _make_table(columns=[email_col])

    # Should not raise, just log warning
    await client.detect_sensitivity([table], batch_size=10)

    # Column should retain default (non-sensitive) due to error
    assert email_col.is_sensitive is False
    assert email_col.sensitivity_type is None


@pytest.mark.asyncio
async def test_detect_sensitivity_partial_failure():
    """Test that failure for one table doesn't affect others."""
    client = _SensitivityPartialClient()

    good_col = _make_column("email")
    good_table = _make_table(name="good", columns=[good_col])

    bad_col = _make_column("email")
    bad_table = _make_table(name="bad", columns=[bad_col])

    # Should process both without raising
    await client.detect_sensitivity([good_table, bad_table], batch_size=10)

    # Good table processed normally (all non-sensitive per partial client logic)
    assert good_col.is_sensitive is False

    # Bad table failed, retains defaults
    assert bad_col.is_sensitive is False


# ── SensitivityType Enum Tests ────────────────────────────────────────────────


def test_sensitivity_type_enum_values():
    """Test that SensitivityType enum has expected values."""
    assert SensitivityType.EMAIL.value == "email"
    assert SensitivityType.SSN.value == "ssn"
    assert SensitivityType.PHONE_NUMBER.value == "phone_number"
    assert SensitivityType.MEDICAL_RECORD_NUMBER.value == "medical_record_number"


def test_sensitivity_type_can_be_constructed_from_string():
    """Test that SensitivityType can be constructed from string values."""
    assert SensitivityType("email") == SensitivityType.EMAIL
    assert SensitivityType("ssn") == SensitivityType.SSN


def test_column_metadata_sensitivity_fields_defaults():
    """Test that ColumnMetadata has correct default sensitivity values."""
    col = _make_column("test")

    assert col.is_sensitive is False
    assert col.sensitivity_type is None


def test_column_metadata_accepts_sensitivity_values():
    """Test that ColumnMetadata can be created with sensitivity values."""
    col = ColumnMetadata(
        name="email",
        ordinal_position=1,
        data_type="varchar(255)",
        is_nullable=True,
        column_default=None,
        character_maximum_length=255,
        numeric_precision=None,
        numeric_scale=None,
        is_sensitive=True,
        sensitivity_type=SensitivityType.EMAIL,
    )

    assert col.is_sensitive is True
    assert col.sensitivity_type == SensitivityType.EMAIL
