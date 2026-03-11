"""Tests for OpenAILLMClient KPI prompt builders and API integration."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.api.v1.schemas.profiler import ColumnMetadata, KPITerm, TableMetadata
from core.llm.openai import (
    _KPI_CLUSTER_MAX_TOKENS,
    _KPI_GENERATE_MAX_TOKENS,
    OpenAILLMClient,
    _build_kpi_cluster_prompt,
    _build_kpi_generate_prompt,
    _build_kpi_synthesize_prompt,
    _build_valid_columns_reference,
    _strip_invalid_linked_columns,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_column(name: str = "id", data_type: str = "integer") -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        ordinal_position=1,
        data_type=data_type,
        is_nullable=True,
        column_default=None,
        character_maximum_length=None,
        numeric_precision=None,
        numeric_scale=None,
    )


def _make_table(name: str = "orders", schema: str = "public") -> TableMetadata:
    return TableMetadata(
        name=name,
        schema=schema,
        owner="postgres",
        description=f"Stores {name} records",
        row_count=1000,
        size_bytes=None,
        total_size_bytes=None,
        data_freshness=None,
        columns=[_make_column("id"), _make_column("amount", "numeric")],
        indexes=None,
        relationships=None,
    )


def _make_kpi(name: str) -> KPITerm:
    return KPITerm(
        name=name,
        description=f"Description of {name}",
        linked_columns=["orders.total_amount"],
    )


def _make_client() -> OpenAILLMClient:
    with patch("core.llm.openai.AsyncOpenAI"):
        return OpenAILLMClient(api_key="sk-test", model="gpt-4o-mini")


def _mock_response(content: str | None) -> MagicMock:
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


# ── Phase A prompt building ────────────────────────────────────────────────────


class TestBuildKpiClusterPrompt:
    def test_prompt_includes_all_qualified_table_names(self):
        tables = [_make_table("orders"), _make_table("invoices")]
        prompt = _build_kpi_cluster_prompt(tables, max_domains=5)
        assert "public.orders" in prompt
        assert "public.invoices" in prompt

    def test_prompt_mentions_max_domains(self):
        tables = [_make_table("orders")]
        prompt = _build_kpi_cluster_prompt(tables, max_domains=8)
        assert "8" in prompt

    def test_prompt_requests_json_format(self):
        tables = [_make_table("orders")]
        prompt = _build_kpi_cluster_prompt(tables, max_domains=5)
        assert "domains" in prompt
        assert "json" in prompt.lower() or "JSON" in prompt

    def test_prompt_does_not_include_column_metadata(self):
        """Cluster prompt should send table names only to keep tokens minimal."""
        tables = [_make_table("orders")]
        prompt = _build_kpi_cluster_prompt(tables, max_domains=5)
        # Column names like 'amount' should NOT appear in the cluster prompt
        assert "amount" not in prompt


# ── Helper function tests ─────────────────────────────────────────────────────


class TestBuildValidColumnsReference:
    def test_returns_all_column_refs_in_table_dot_column_format(self):
        tables = [_make_table("orders")]
        text, valid = _build_valid_columns_reference(tables, max_cols_per_table=10)
        assert "orders.id" in text
        assert "orders.amount" in text
        assert valid == {"orders.id", "orders.amount"}

    def test_respects_max_cols_per_table(self):
        table = _make_table("orders")
        table.columns = [_make_column(f"col_{i}") for i in range(20)]
        _, valid = _build_valid_columns_reference([table], max_cols_per_table=5)
        assert len(valid) == 5

    def test_multiple_tables(self):
        tables = [_make_table("orders"), _make_table("users")]
        _, valid = _build_valid_columns_reference(tables, max_cols_per_table=10)
        assert "orders.id" in valid
        assert "users.id" in valid

    def test_empty_tables_returns_empty(self):
        text, valid = _build_valid_columns_reference([], max_cols_per_table=10)
        assert text == ""
        assert valid == set()


class TestStripInvalidLinkedColumns:
    def test_keeps_valid_columns(self):
        kpi = KPITerm(name="Test", description="desc", linked_columns=["orders.amount"])
        _strip_invalid_linked_columns([kpi], {"orders.amount"}, "test", "test")
        assert kpi.linked_columns == ["orders.amount"]

    def test_removes_invalid_columns(self):
        kpi = KPITerm(
            name="Test",
            description="desc",
            linked_columns=["orders.amount", "orders.nonexistent"],
        )
        _strip_invalid_linked_columns([kpi], {"orders.amount"}, "test", "test")
        assert kpi.linked_columns == ["orders.amount"]

    def test_all_invalid_results_in_empty_list(self):
        kpi = KPITerm(name="Test", description="desc", linked_columns=["fake.col1", "fake.col2"])
        _strip_invalid_linked_columns([kpi], {"orders.amount"}, "test", "test")
        assert kpi.linked_columns == []

    def test_empty_linked_columns_unchanged(self):
        kpi = KPITerm(name="Test", description="desc", linked_columns=[])
        _strip_invalid_linked_columns([kpi], {"orders.amount"}, "test", "test")
        assert kpi.linked_columns == []


# ── Phase B prompt building ────────────────────────────────────────────────────


class TestBuildKpiGeneratePrompt:
    def test_prompt_includes_domain_name(self):
        tables = [_make_table("orders")]
        prompt = _build_kpi_generate_prompt("Finance", tables, max_kpis=5)
        assert "Finance" in prompt

    def test_prompt_includes_qualified_table_names(self):
        tables = [_make_table("orders", "sales")]
        prompt = _build_kpi_generate_prompt("Finance", tables, max_kpis=5)
        assert "sales.orders" in prompt

    def test_prompt_includes_table_description(self):
        tables = [_make_table("orders")]
        prompt = _build_kpi_generate_prompt("Finance", tables, max_kpis=5)
        assert "Stores orders records" in prompt

    def test_prompt_includes_column_names(self):
        tables = [_make_table("orders")]
        prompt = _build_kpi_generate_prompt("Finance", tables, max_kpis=5)
        assert "amount" in prompt

    def test_prompt_mentions_max_kpis(self):
        tables = [_make_table("orders")]
        prompt = _build_kpi_generate_prompt("Finance", tables, max_kpis=7)
        assert "7" in prompt

    def test_prompt_requests_json_format(self):
        tables = [_make_table("orders")]
        prompt = _build_kpi_generate_prompt("Finance", tables, max_kpis=5)
        assert "kpis" in prompt
        assert "json" in prompt.lower() or "JSON" in prompt

    def test_prompt_truncates_excess_tables(self):
        tables = [_make_table(f"t{i}") for i in range(20)]
        prompt = _build_kpi_generate_prompt("Finance", tables, max_kpis=5)
        assert "omitted" in prompt.lower() or "additional" in prompt.lower()

    def test_prompt_shows_unknown_row_count_when_none(self):
        table = _make_table("orders")
        table.row_count = None
        prompt = _build_kpi_generate_prompt("Finance", [table], max_kpis=5)
        assert "unknown" in prompt

    def test_prompt_shows_none_when_no_description(self):
        table = _make_table("orders")
        table.description = None
        prompt = _build_kpi_generate_prompt("Finance", [table], max_kpis=5)
        assert "(none)" in prompt

    def test_prompt_includes_valid_columns_reference_section(self):
        tables = [_make_table("orders")]
        prompt = _build_kpi_generate_prompt("Finance", tables, max_kpis=5)
        assert "Valid Columns Reference" in prompt
        assert "orders.id" in prompt
        assert "orders.amount" in prompt

    def test_prompt_includes_critical_constraint(self):
        tables = [_make_table("orders")]
        prompt = _build_kpi_generate_prompt("Finance", tables, max_kpis=5)
        assert "CRITICAL CONSTRAINT" in prompt
        assert "DO NOT invent" in prompt

    def test_prompt_includes_quality_guidelines(self):
        tables = [_make_table("orders")]
        prompt = _build_kpi_generate_prompt("Finance", tables, max_kpis=5)
        assert "Ratio or rate" in prompt
        assert "AVOID" in prompt

    def test_prompt_encourages_fewer_over_low_quality(self):
        tables = [_make_table("orders")]
        prompt = _build_kpi_generate_prompt("Finance", tables, max_kpis=5)
        assert "fewer" in prompt.lower()


# ── Phase C prompt building ────────────────────────────────────────────────────


class TestBuildKpiSynthesizePrompt:
    def test_prompt_includes_all_kpi_names(self):
        kpis = [_make_kpi("Monthly Revenue"), _make_kpi("Churn Rate")]
        prompt = _build_kpi_synthesize_prompt(kpis)
        assert "Monthly Revenue" in prompt
        assert "Churn Rate" in prompt

    def test_prompt_requests_json_format(self):
        kpis = [_make_kpi("Revenue")]
        prompt = _build_kpi_synthesize_prompt(kpis)
        assert "kpis" in prompt
        assert "json" in prompt.lower() or "JSON" in prompt

    def test_prompt_requests_new_cross_domain_kpis(self):
        kpis = [_make_kpi("Revenue")]
        prompt = _build_kpi_synthesize_prompt(kpis)
        assert "cross-domain" in prompt.lower() or "NEW" in prompt

    def test_prompt_requests_calculation_and_linked_columns(self):
        kpis = [_make_kpi("Revenue")]
        prompt = _build_kpi_synthesize_prompt(kpis)
        assert "calculation" in prompt
        assert "linked_columns" in prompt

    def test_prompt_discourages_trivial_metrics(self):
        kpis = [_make_kpi("Revenue")]
        prompt = _build_kpi_synthesize_prompt(kpis)
        assert "trivial" in prompt.lower() or "COUNT(*)" in prompt

    def test_prompt_requires_dashboard_ready(self):
        kpis = [_make_kpi("Revenue")]
        prompt = _build_kpi_synthesize_prompt(kpis)
        assert "dashboard" in prompt.lower()


# ── OpenAILLMClient._cluster_tables_into_domains ──────────────────────────────


def _valid_cluster_json(domains: list[dict] | None = None) -> str:
    if domains is None:
        domains = [
            {"domain": "Finance", "tables": ["public.orders", "public.invoices"]},
            {"domain": "Customer", "tables": ["public.users"]},
        ]
    return json.dumps({"domains": domains})


class TestOpenAIClusterTablesIntoDomains:
    async def test_returns_parsed_clusters(self):
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(return_value=_mock_response(_valid_cluster_json()))
        tables = [_make_table("orders"), _make_table("invoices"), _make_table("users")]
        result = await client._cluster_tables_into_domains(tables, max_domains=5)
        assert "Finance" in result
        assert "Customer" in result
        assert "public.orders" in result["Finance"]
        assert "public.users" in result["Customer"]

    async def test_returns_empty_dict_on_none_content(self):
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(return_value=_mock_response(None))
        result = await client._cluster_tables_into_domains([_make_table()], max_domains=5)
        assert result == {}

    async def test_returns_empty_dict_on_invalid_json(self):
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(return_value=_mock_response("not valid json {{"))
        result = await client._cluster_tables_into_domains([_make_table()], max_domains=5)
        assert result == {}

    async def test_returns_empty_dict_on_missing_domains_key(self):
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(return_value=_mock_response(json.dumps({"wrong_key": []})))
        result = await client._cluster_tables_into_domains([_make_table()], max_domains=5)
        assert result == {}

    async def test_propagates_api_exceptions(self):
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))
        with pytest.raises(Exception, match="API error"):
            await client._cluster_tables_into_domains([_make_table()], max_domains=5)

    async def test_uses_cluster_max_tokens(self):
        client = _make_client()
        create_mock = AsyncMock(return_value=_mock_response(_valid_cluster_json()))
        client._client.chat.completions.create = create_mock
        await client._cluster_tables_into_domains([_make_table()], max_domains=5)
        assert create_mock.call_args.kwargs["max_tokens"] == _KPI_CLUSTER_MAX_TOKENS

    async def test_passes_json_object_response_format(self):
        client = _make_client()
        create_mock = AsyncMock(return_value=_mock_response(_valid_cluster_json()))
        client._client.chat.completions.create = create_mock
        await client._cluster_tables_into_domains([_make_table()], max_domains=5)
        assert create_mock.call_args.kwargs["response_format"] == {"type": "json_object"}

    async def test_system_and_user_messages_included(self):
        client = _make_client()
        create_mock = AsyncMock(return_value=_mock_response(_valid_cluster_json()))
        client._client.chat.completions.create = create_mock
        await client._cluster_tables_into_domains([_make_table()], max_domains=5)
        messages = create_mock.call_args.kwargs["messages"]
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles


# ── OpenAILLMClient._generate_domain_kpis ────────────────────────────────────


def _valid_kpi_json(kpis: list[dict] | None = None) -> str:
    if kpis is None:
        kpis = [
            {
                "name": "Monthly Revenue",
                "description": "Total revenue per month.",
                "calculation": "SELECT SUM(amount) FROM orders WHERE ...",
                "linked_columns": ["orders.amount"],
            },
            {
                "name": "Average Order Value",
                "description": "Mean value of each order.",
                "calculation": "SELECT AVG(amount) FROM orders",
                "linked_columns": ["orders.amount"],
            },
        ]
    return json.dumps({"kpis": kpis})


class TestOpenAIGenerateDomainKpis:
    async def test_returns_parsed_kpi_terms(self):
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(return_value=_mock_response(_valid_kpi_json()))
        kpis = await client._generate_domain_kpis("Finance", [_make_table()], max_kpis=5)
        assert len(kpis) == 2
        assert kpis[0].name == "Monthly Revenue"
        assert kpis[1].name == "Average Order Value"

    async def test_enforces_max_kpis_limit(self):
        many_kpis = [
            {"name": f"KPI{i}", "description": "desc", "source_tables": [], "calculation_hint": None, "is_cross_domain": False}
            for i in range(10)
        ]
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(return_value=_mock_response(json.dumps({"kpis": many_kpis})))
        kpis = await client._generate_domain_kpis("Finance", [_make_table()], max_kpis=3)
        assert len(kpis) == 3

    async def test_returns_calculation_and_linked_columns(self):
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(return_value=_mock_response(_valid_kpi_json()))
        kpis = await client._generate_domain_kpis("Finance", [_make_table()], max_kpis=5)
        assert kpis[0].calculation is not None
        assert kpis[0].linked_columns == ["orders.amount"]

    async def test_returns_empty_list_on_none_content(self):
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(return_value=_mock_response(None))
        kpis = await client._generate_domain_kpis("Finance", [_make_table()], max_kpis=5)
        assert kpis == []

    async def test_returns_empty_list_on_invalid_json(self):
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(return_value=_mock_response("not valid json {{"))
        kpis = await client._generate_domain_kpis("Finance", [_make_table()], max_kpis=5)
        assert kpis == []

    async def test_propagates_api_exceptions(self):
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))
        with pytest.raises(Exception, match="API error"):
            await client._generate_domain_kpis("Finance", [_make_table()], max_kpis=5)

    async def test_uses_generate_max_tokens(self):
        client = _make_client()
        create_mock = AsyncMock(return_value=_mock_response(_valid_kpi_json()))
        client._client.chat.completions.create = create_mock
        await client._generate_domain_kpis("Finance", [_make_table()], max_kpis=5)
        assert create_mock.call_args.kwargs["max_tokens"] == _KPI_GENERATE_MAX_TOKENS

    async def test_passes_json_object_response_format(self):
        client = _make_client()
        create_mock = AsyncMock(return_value=_mock_response(_valid_kpi_json()))
        client._client.chat.completions.create = create_mock
        await client._generate_domain_kpis("Finance", [_make_table()], max_kpis=5)
        assert create_mock.call_args.kwargs["response_format"] == {"type": "json_object"}

    async def test_strips_hallucinated_linked_columns(self):
        """linked_columns not matching actual table columns should be stripped."""
        kpi_data = [
            {
                "name": "Revenue",
                "description": "Total revenue",
                "calculation": "SUM(orders.amount)",
                "linked_columns": ["orders.amount", "orders.hallucinated_col"],
            }
        ]
        client = _make_client()
        client._client.chat.completions.create = AsyncMock(return_value=_mock_response(json.dumps({"kpis": kpi_data})))
        kpis = await client._generate_domain_kpis("Finance", [_make_table()], max_kpis=5)
        assert len(kpis) == 1
        assert "orders.amount" in kpis[0].linked_columns
        assert "orders.hallucinated_col" not in kpis[0].linked_columns


# ── OpenAILLMClient._synthesize_kpis ─────────────────────────────────────────


def _valid_synthesis_json(kpis: list[dict] | None = None) -> str:
    if kpis is None:
        kpis = [
            {
                "name": "Monthly Revenue",
                "description": "Total revenue per month.",
                "calculation": None,
                "linked_columns": [],
            },
            {
                "name": "Customer Lifetime Value",
                "description": "Combines revenue and customer data.",
                "calculation": None,
                "linked_columns": [],
            },
        ]
    return json.dumps({"kpis": kpis})
