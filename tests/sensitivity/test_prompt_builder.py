"""Tests for PII/PHI sensitivity prompt builder and response parser."""

from __future__ import annotations

import pytest

from core.api.v1.schemas.profiler import ColumnMetadata, SensitivityType, TableMetadata
from core.services.sensitivity.prompt_builder import (
    build_sensitivity_prompt,
    get_sensitivity_system_prompt,
    parse_sensitivity_response,
)


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


# ── System Prompt Tests ───────────────────────────────────────────────────────


def test_get_sensitivity_system_prompt_returns_string():
    prompt = get_sensitivity_system_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_system_prompt_contains_compliance_keywords():
    prompt = get_sensitivity_system_prompt()
    assert "PII" in prompt
    assert "PHI" in prompt
    assert "HIPAA" in prompt
    assert "GDPR" in prompt
    assert "CCPA" in prompt


def test_system_prompt_instructs_json_response():
    prompt = get_sensitivity_system_prompt()
    assert "JSON" in prompt


# ── Build Prompt Tests ────────────────────────────────────────────────────────


def test_build_prompt_includes_table_context():
    table = _make_table(name="customers", schema="sales", description="Customer data")
    columns = [_make_column("email", sample_values=["test@example.com"])]

    prompt = build_sensitivity_prompt(table, columns)

    assert "sales.customers" in prompt
    assert "Customer data" in prompt


def test_build_prompt_includes_column_details():
    table = _make_table()
    columns = [
        _make_column("email", "varchar(255)", "User email address", ["john@test.com"]),
        _make_column("ssn", "varchar(11)", "Social security number"),
    ]

    prompt = build_sensitivity_prompt(table, columns)

    assert "email" in prompt
    assert "ssn" in prompt
    assert "varchar(255)" in prompt


def test_build_prompt_includes_sample_values():
    table = _make_table()
    columns = [_make_column("email", sample_values=["john@test.com", "jane@example.org"])]

    prompt = build_sensitivity_prompt(table, columns)

    assert "john@test.com" in prompt


def test_build_prompt_truncates_long_sample_values():
    long_value = "x" * 100
    table = _make_table()
    columns = [_make_column("data", sample_values=[long_value])]

    prompt = build_sensitivity_prompt(table, columns)

    # Should be truncated with "..."
    assert "..." in prompt
    # Full value should not appear
    assert long_value not in prompt


def test_build_prompt_handles_no_description():
    table = _make_table(description=None)
    columns = [_make_column("id")]

    prompt = build_sensitivity_prompt(table, columns)

    assert "(no description)" in prompt


def test_build_prompt_includes_sensitivity_type_descriptions():
    table = _make_table()
    columns = [_make_column("id")]

    prompt = build_sensitivity_prompt(table, columns)

    # Should include at least some sensitivity type descriptions
    assert "email" in prompt.lower()
    assert "ssn" in prompt.lower()
    assert "phone" in prompt.lower()


def test_build_prompt_includes_important_distinctions():
    table = _make_table()
    columns = [_make_column("created_at")]

    prompt = build_sensitivity_prompt(table, columns)

    # Should warn about non-PII timestamps
    assert "created_at" in prompt
    assert "NOT PII" in prompt


# ── Parse Response Tests ──────────────────────────────────────────────────────


def test_parse_valid_sensitive_response():
    columns = [_make_column("email"), _make_column("ssn")]
    response = """{
        "columns": [
            {"column_name": "email", "is_sensitive": true, "sensitivity_type": "email"},
            {"column_name": "ssn", "is_sensitive": true, "sensitivity_type": "ssn"}
        ]
    }"""

    results = parse_sensitivity_response(response, columns)

    assert results["email"] == (True, SensitivityType.EMAIL)
    assert results["ssn"] == (True, SensitivityType.SSN)


def test_parse_non_sensitive_response():
    columns = [_make_column("id"), _make_column("created_at")]
    response = """{
        "columns": [
            {"column_name": "id", "is_sensitive": false, "sensitivity_type": null},
            {"column_name": "created_at", "is_sensitive": false, "sensitivity_type": null}
        ]
    }"""

    results = parse_sensitivity_response(response, columns)

    assert results["id"] == (False, None)
    assert results["created_at"] == (False, None)


def test_parse_mixed_response():
    columns = [_make_column("id"), _make_column("email"), _make_column("status")]
    response = """{
        "columns": [
            {"column_name": "id", "is_sensitive": false, "sensitivity_type": null},
            {"column_name": "email", "is_sensitive": true, "sensitivity_type": "email"},
            {"column_name": "status", "is_sensitive": false, "sensitivity_type": null}
        ]
    }"""

    results = parse_sensitivity_response(response, columns)

    assert results["id"] == (False, None)
    assert results["email"] == (True, SensitivityType.EMAIL)
    assert results["status"] == (False, None)


def test_parse_invalid_json_returns_defaults():
    columns = [_make_column("email")]
    response = "not valid json"

    results = parse_sensitivity_response(response, columns)

    # Should default to non-sensitive
    assert results["email"] == (False, None)


def test_parse_empty_response_returns_defaults():
    columns = [_make_column("email")]
    response = ""

    results = parse_sensitivity_response(response, columns)

    assert results["email"] == (False, None)


def test_parse_markdown_wrapped_json():
    columns = [_make_column("email")]
    response = """```json
{
    "columns": [
        {"column_name": "email", "is_sensitive": true, "sensitivity_type": "email"}
    ]
}
```"""

    results = parse_sensitivity_response(response, columns)

    assert results["email"] == (True, SensitivityType.EMAIL)


def test_parse_unknown_sensitivity_type_maps_to_other():
    columns = [_make_column("custom_id")]
    response = """{
        "columns": [
            {"column_name": "custom_id", "is_sensitive": true, "sensitivity_type": "unknown_type"}
        ]
    }"""

    results = parse_sensitivity_response(response, columns)

    # Unknown types should map to OTHER_UNIQUE_ID
    assert results["custom_id"] == (True, SensitivityType.OTHER_UNIQUE_ID)


def test_parse_missing_column_in_response():
    columns = [_make_column("email"), _make_column("phone")]
    # Response only includes email, not phone
    response = """{
        "columns": [
            {"column_name": "email", "is_sensitive": true, "sensitivity_type": "email"}
        ]
    }"""

    results = parse_sensitivity_response(response, columns)

    assert results["email"] == (True, SensitivityType.EMAIL)
    # Missing column should default to non-sensitive
    assert results["phone"] == (False, None)


def test_parse_extra_column_in_response_ignored():
    columns = [_make_column("email")]
    # Response includes extra column not in input
    response = """{
        "columns": [
            {"column_name": "email", "is_sensitive": true, "sensitivity_type": "email"},
            {"column_name": "extra", "is_sensitive": true, "sensitivity_type": "ssn"}
        ]
    }"""

    results = parse_sensitivity_response(response, columns)

    assert results["email"] == (True, SensitivityType.EMAIL)
    assert "extra" not in results


def test_parse_is_sensitive_string_true():
    columns = [_make_column("email")]
    response = """{
        "columns": [
            {"column_name": "email", "is_sensitive": "true", "sensitivity_type": "email"}
        ]
    }"""

    results = parse_sensitivity_response(response, columns)

    assert results["email"] == (True, SensitivityType.EMAIL)


def test_parse_all_hipaa_safe_harbor_types():
    """Test that all HIPAA Safe Harbor 18 identifier types are properly parsed."""
    hipaa_types = [
        "name", "address", "date", "phone_number", "email", "ssn",
        "medical_record_number", "health_plan_id", "account_number",
        "license_number", "vehicle_id", "device_id", "web_url",
        "ip_address", "biometric", "photo", "age_over_89", "other_unique_id"
    ]

    for type_name in hipaa_types:
        columns = [_make_column("test_col")]
        response = f'{{"columns": [{{"column_name": "test_col", "is_sensitive": true, "sensitivity_type": "{type_name}"}}]}}'

        results = parse_sensitivity_response(response, columns)

        is_sensitive, sensitivity_type = results["test_col"]
        assert is_sensitive is True
        assert sensitivity_type is not None
        assert sensitivity_type.value == type_name


def test_parse_phi_types():
    """Test that PHI types are properly parsed."""
    phi_types = [
        "diagnosis_code", "procedure_code", "medication",
        "lab_result", "vital_sign", "genetic_data",
        "mental_health", "substance_abuse"
    ]

    for type_name in phi_types:
        columns = [_make_column("test_col")]
        response = f'{{"columns": [{{"column_name": "test_col", "is_sensitive": true, "sensitivity_type": "{type_name}"}}]}}'

        results = parse_sensitivity_response(response, columns)

        is_sensitive, sensitivity_type = results["test_col"]
        assert is_sensitive is True
        assert sensitivity_type is not None
        assert sensitivity_type.value == type_name
