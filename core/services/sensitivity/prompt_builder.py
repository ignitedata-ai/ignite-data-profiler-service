"""Prompt construction for PII/PHI sensitivity detection.

Builds optimized prompts for LLM-based detection of sensitive data columns.
"""

from __future__ import annotations

from typing import Any

from core.api.v1.schemas.profiler import ColumnMetadata, SensitivityType, TableMetadata

# Maximum characters per sample value to send to the LLM
MAX_SAMPLE_VALUE_LENGTH = 50
# Maximum number of sample values per column
MAX_SAMPLE_VALUES = 5


def get_sensitivity_system_prompt() -> str:
    """Return the system prompt for PII/PHI detection."""
    return """You are a data privacy and compliance expert specializing in PII/PHI detection for HIPAA, GDPR, and CCPA compliance.

Your task is to identify columns that contain or could contain personally identifiable information (PII) or protected health information (PHI).

Be CONSERVATIVE: Only flag columns where you have reasonable confidence the data is sensitive.
Be PRECISE: Use the exact sensitivity_type codes provided.
Be CONSISTENT: Similar column patterns should receive similar classifications.

Always respond with valid JSON only."""


def _truncate_sample_value(value: Any) -> str:
    """Truncate a sample value to a reasonable length for the prompt."""
    if value is None:
        return "null"
    str_value = str(value)
    if len(str_value) > MAX_SAMPLE_VALUE_LENGTH:
        return str_value[:MAX_SAMPLE_VALUE_LENGTH] + "..."
    return str_value


def _format_column_for_prompt(column: ColumnMetadata) -> dict:
    """Format a column's metadata for inclusion in the prompt."""
    col_data: dict[str, Any] = {
        "name": column.name,
        "data_type": column.data_type,
    }

    if column.description:
        col_data["description"] = column.description

    if column.sample_values:
        # Truncate sample values and limit count
        samples = [
            _truncate_sample_value(v) for v in column.sample_values[:MAX_SAMPLE_VALUES]
        ]
        col_data["sample_values"] = samples

    return col_data


def _get_sensitivity_type_descriptions() -> str:
    """Return descriptions for all sensitivity types."""
    descriptions = {
        # HIPAA Safe Harbor 18 Identifiers
        "name": "Personal names (first, last, full names)",
        "address": "Street address, city, state, ZIP code, postal code, county, or any geographic subdivision smaller than a state",
        "date": "Dates directly related to an individual: date of birth (DOB/dob/birth_date), admission, discharge, or death dates (NOT operational timestamps like created_at, updated_at, order_date)",
        "phone_number": "Telephone and fax numbers",
        "email": "Email addresses",
        "ssn": "Social Security Numbers (format: XXX-XX-XXXX)",
        "medical_record_number": "Medical record numbers (MRN)",
        "health_plan_id": "Health plan beneficiary numbers",
        "account_number": "Financial account numbers",
        "license_number": "Certificate or license numbers",
        "vehicle_id": "Vehicle identification numbers (VIN), license plates",
        "device_id": "Device identifiers and serial numbers",
        "web_url": "Personal web URLs",
        "ip_address": "IP addresses",
        "biometric": "Biometric identifiers: fingerprints, voice prints, retinal scans",
        "photo": "Full-face photographs or comparable images",
        "age_over_89": "Ages over 89 years",
        "other_unique_id": "Any other unique identifying number or code",
        # Extended PII
        "credit_card": "Credit or debit card numbers",
        "passport": "Passport numbers",
        "national_id": "National ID numbers (non-US equivalents of SSN)",
        "bank_routing": "Bank routing numbers",
        "tax_id": "Tax identification numbers (EIN, ITIN)",
        "drivers_license": "Driver's license numbers",
        "insurance_policy": "Insurance policy numbers",
        "employee_id": "Employee identification numbers (when linkable to individuals)",
        "customer_id": "Customer IDs (when linkable to individuals with other PII)",
        "login_credential": "Usernames, passwords, PINs, security questions",
        # PHI
        "diagnosis_code": "ICD codes, medical diagnoses",
        "procedure_code": "CPT codes, medical procedures",
        "medication": "Medication names, prescriptions, dosages",
        "lab_result": "Laboratory test results, test values",
        "vital_sign": "Vital signs: blood pressure, heart rate, temperature, etc.",
        "genetic_data": "Genetic or genomic information, DNA sequences",
        "mental_health": "Mental health information, psychiatric records",
        "substance_abuse": "Substance abuse treatment records",
        # Other Sensitive
        "financial_data": "Salary, income, financial records",
        "ethnicity": "Race or ethnicity",
        "religion": "Religious affiliation",
        "sexual_orientation": "Sexual orientation",
        "political_affiliation": "Political affiliation or beliefs",
        "criminal_record": "Criminal history, arrest records",
        "gender": "Gender identity, biological sex, or sex assigned at birth (e.g. male, female, non-binary)",
        "geolocation": "GPS coordinates, precise location data",
        "internal_metrics": "Internal business metrics, KPIs, or operational data not intended for external exposure",
    }

    lines = []
    for code, desc in descriptions.items():
        lines.append(f"- {code}: {desc}")
    return "\n".join(lines)


def build_sensitivity_prompt(
    table: TableMetadata,
    columns: list[ColumnMetadata],
) -> str:
    """Build the user prompt for PII/PHI detection.

    Args:
        table: The table containing the columns.
        columns: The columns to analyze (a batch).

    Returns:
        A formatted prompt string for the LLM.
    """
    # Format columns as JSON-like structure
    columns_data = [_format_column_for_prompt(col) for col in columns]

    # Build the column list as a formatted string
    import json

    columns_json = json.dumps(columns_data, indent=2)

    prompt = f"""Analyze the following database columns for PII/PHI:

## Table Context
- Table: {table.schema_name}.{table.name}
- Description: {table.description or "(no description)"}

## Columns to Analyze
{columns_json}

## Valid Sensitivity Types
{_get_sensitivity_type_descriptions()}

## Important Distinctions
- created_at, updated_at, order_date, modified_at are NOT PII dates (they are operational timestamps)
- product_id, order_id, transaction_id are NOT PII unless they encode personal information
- Generic status, type, category, flag columns are NOT sensitive
- Aggregated or anonymized metrics are NOT sensitive
- Column names like "user_count", "total_orders" contain aggregate data, not PII

## Response Format
Respond with ONLY valid JSON in this exact format:
{{
  "columns": [
    {{
      "column_name": "exact_column_name_from_input",
      "is_sensitive": true,
      "sensitivity_type": "email"
    }},
    {{
      "column_name": "non_sensitive_column",
      "is_sensitive": false,
      "sensitivity_type": null
    }}
  ]
}}

Include ALL columns from the input in your response. Use null for sensitivity_type when is_sensitive is false."""

    return prompt


def parse_sensitivity_response(
    response_text: str,
    columns: list[ColumnMetadata],
) -> dict[str, tuple[bool, SensitivityType | None]]:
    """Parse the LLM response and map results to columns.

    Args:
        response_text: The raw JSON response from the LLM.
        columns: The original columns that were analyzed.

    Returns:
        A dict mapping column_name -> (is_sensitive, sensitivity_type).
        Unknown sensitivity types are mapped to OTHER_UNIQUE_ID.
        Columns not found in the response are marked as non-sensitive.
    """
    import json

    # Build a set of valid sensitivity type values for validation
    valid_types = {st.value for st in SensitivityType}

    # Default all columns to non-sensitive
    results: dict[str, tuple[bool, SensitivityType | None]] = {
        col.name: (False, None) for col in columns
    }

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        import re

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                return results
        else:
            return results

    if not isinstance(data, dict) or "columns" not in data:
        return results

    for col_result in data.get("columns", []):
        if not isinstance(col_result, dict):
            continue

        col_name = col_result.get("column_name")
        if not col_name or col_name not in results:
            continue

        is_sensitive = col_result.get("is_sensitive", False)
        if not isinstance(is_sensitive, bool):
            is_sensitive = str(is_sensitive).lower() == "true"

        if not is_sensitive:
            results[col_name] = (False, None)
            continue

        sensitivity_type_str = col_result.get("sensitivity_type")
        if sensitivity_type_str and sensitivity_type_str in valid_types:
            sensitivity_type = SensitivityType(sensitivity_type_str)
        elif sensitivity_type_str:
            # Unknown type, map to OTHER_UNIQUE_ID
            sensitivity_type = SensitivityType.OTHER_UNIQUE_ID
        else:
            sensitivity_type = SensitivityType.OTHER_UNIQUE_ID

        results[col_name] = (True, sensitivity_type)

    return results
