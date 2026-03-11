"""Tests for BaseLLMClient.augment_kpis Map-Reduce pipeline and error isolation."""

from __future__ import annotations

from core.api.v1.schemas.profiler import ColumnMetadata, GlossaryTerm, KPITerm, TableMetadata
from core.llm.base import BaseLLMClient

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_column(name: str = "id") -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        ordinal_position=1,
        data_type="integer",
        is_nullable=False,
        column_default=None,
        character_maximum_length=None,
        numeric_precision=32,
        numeric_scale=0,
    )


def _make_table(name: str, schema: str = "public") -> TableMetadata:
    return TableMetadata(
        name=name,
        schema=schema,
        owner="postgres",
        description=None,
        row_count=100,
        size_bytes=None,
        total_size_bytes=None,
        data_freshness=None,
        columns=[_make_column()],
        indexes=None,
        relationships=None,
    )


def _make_kpi(name: str) -> KPITerm:
    return KPITerm(name=name, description=f"Description of {name}")


# ── Concrete stub base class ───────────────────────────────────────────────────
# Provides no-op implementations for the non-KPI abstract methods so KPI-specific
# stubs only need to override what they care about.


class _BaseStub(BaseLLMClient):
    provider_name = "test"

    async def _describe_table(self, table: TableMetadata) -> str | None:
        return None

    async def _describe_column(self, column: ColumnMetadata, table: TableMetadata) -> str | None:
        return None

    async def _infer_glossary(self, table: TableMetadata) -> list[GlossaryTerm]:
        return []

    async def _cluster_tables_into_domains(self, tables: list[TableMetadata], max_domains: int) -> dict[str, list[str]]:
        return {}

    async def _generate_domain_kpis(self, domain_name: str, domain_tables: list[TableMetadata], max_kpis: int) -> list[KPITerm]:
        return []

    async def _synthesize_kpis(self, all_domain_kpis: list[KPITerm]) -> list[KPITerm]:
        return all_domain_kpis

    async def _judge_filter_columns(self, table_name: str, table_role: str, candidates: list[dict]) -> list[dict]:
        return []


# ── Domain-specific stubs ──────────────────────────────────────────────────────


class _SucceedingKPIClient(_BaseStub):
    """Full happy-path: clusters into one domain, generates 1 KPI, passes synthesis."""

    async def _cluster_tables_into_domains(self, tables: list[TableMetadata], max_domains: int) -> dict[str, list[str]]:
        return {"Finance": [f"{t.schema_name}.{t.name}" for t in tables]}

    async def _generate_domain_kpis(self, domain_name: str, domain_tables: list[TableMetadata], max_kpis: int) -> list[KPITerm]:
        return [_make_kpi(f"KPI for {domain_name}")]

    async def _synthesize_kpis(self, all_domain_kpis: list[KPITerm]) -> list[KPITerm]:
        return all_domain_kpis


class _ClusterFailingClient(_BaseStub):
    """Phase A always raises."""

    async def _cluster_tables_into_domains(self, tables: list[TableMetadata], max_domains: int) -> dict[str, list[str]]:
        raise RuntimeError("clustering failed")


class _EmptyClusterClient(_BaseStub):
    """Phase A returns an empty dict."""

    async def _cluster_tables_into_domains(self, tables: list[TableMetadata], max_domains: int) -> dict[str, list[str]]:
        return {}


class _MultiDomainClient(_BaseStub):
    """Two domains: Finance succeeds, HR fails in Phase B."""

    async def _cluster_tables_into_domains(self, tables: list[TableMetadata], max_domains: int) -> dict[str, list[str]]:
        return {
            "Finance": ["public.orders"],
            "HR": ["public.employees"],
        }

    async def _generate_domain_kpis(self, domain_name: str, domain_tables: list[TableMetadata], max_kpis: int) -> list[KPITerm]:
        if domain_name == "HR":
            raise RuntimeError("domain generation failed")
        return [_make_kpi(f"KPI for {domain_name}")]

    async def _synthesize_kpis(self, all_domain_kpis: list[KPITerm]) -> list[KPITerm]:
        return all_domain_kpis


class _AllDomainsFailClient(_BaseStub):
    """Phase B always fails for every domain."""

    async def _cluster_tables_into_domains(self, tables: list[TableMetadata], max_domains: int) -> dict[str, list[str]]:
        return {"Finance": ["public.orders"]}

    async def _generate_domain_kpis(self, domain_name: str, domain_tables: list[TableMetadata], max_kpis: int) -> list[KPITerm]:
        raise RuntimeError("domain generation failed")


class _SynthesisFailingClient(_BaseStub):
    """Phase C always raises — pipeline should fall back to raw domain KPIs."""

    async def _cluster_tables_into_domains(self, tables: list[TableMetadata], max_domains: int) -> dict[str, list[str]]:
        return {"Finance": ["public.orders"]}

    async def _generate_domain_kpis(self, domain_name: str, domain_tables: list[TableMetadata], max_kpis: int) -> list[KPITerm]:
        return [_make_kpi("Revenue")]

    async def _synthesize_kpis(self, all_domain_kpis: list[KPITerm]) -> list[KPITerm]:
        raise RuntimeError("synthesis failed")


class _UnknownTableClient(_BaseStub):
    """Phase A returns table names that don't match any real tables."""

    async def _cluster_tables_into_domains(self, tables: list[TableMetadata], max_domains: int) -> dict[str, list[str]]:
        return {"Finance": ["public.nonexistent_table"]}

    async def _generate_domain_kpis(self, domain_name: str, domain_tables: list[TableMetadata], max_kpis: int) -> list[KPITerm]:
        return [_make_kpi("Should not appear")]  # pragma: no cover


class _MaxDomainsTrackingClient(_BaseStub):
    """Records the max_domains value passed to Phase A."""

    captured_max_domains: int = 0

    async def _cluster_tables_into_domains(self, tables: list[TableMetadata], max_domains: int) -> dict[str, list[str]]:
        _MaxDomainsTrackingClient.captured_max_domains = max_domains
        return {}


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestAugmentKpisEmptyGuard:
    async def test_empty_tables_returns_empty_list(self):
        client = _SucceedingKPIClient()
        result = await client.augment_kpis([], max_domains=5, kpis_per_domain=3)
        assert result == []


class TestAugmentKpisPhaseAFailures:
    async def test_clustering_exception_returns_empty_list(self):
        client = _ClusterFailingClient()
        tables = [_make_table("orders")]
        result = await client.augment_kpis(tables, max_domains=5, kpis_per_domain=3)
        assert result == []

    async def test_empty_cluster_result_returns_empty_list(self):
        client = _EmptyClusterClient()
        tables = [_make_table("orders")]
        result = await client.augment_kpis(tables, max_domains=5, kpis_per_domain=3)
        assert result == []

    async def test_domain_with_only_unknown_table_names_produces_no_kpis(self):
        """Domain references a table not present in the profiling response."""
        client = _UnknownTableClient()
        tables = [_make_table("orders")]  # only 'public.orders' exists
        result = await client.augment_kpis(tables, max_domains=5, kpis_per_domain=3)
        # The domain is skipped because 'public.nonexistent_table' not in table_lookup
        assert result == []


class TestAugmentKpisPhaseBFailures:
    async def test_partial_domain_failure_does_not_affect_other_domains(self):
        """If Phase B fails for 'HR', KPIs from 'Finance' are still returned."""
        client = _MultiDomainClient()
        tables = [_make_table("orders"), _make_table("employees")]
        result = await client.augment_kpis(tables, max_domains=5, kpis_per_domain=3)
        assert len(result) == 1
        assert result[0].name == "KPI for Finance"

    async def test_all_domain_failures_return_empty_list(self):
        client = _AllDomainsFailClient()
        tables = [_make_table("orders")]
        result = await client.augment_kpis(tables, max_domains=5, kpis_per_domain=3)
        assert result == []


class TestAugmentKpisPhaseCFailures:
    async def test_synthesis_failure_returns_raw_domain_kpis(self):
        """Phase C failure must not discard the Phase B results."""
        client = _SynthesisFailingClient()
        tables = [_make_table("orders")]
        result = await client.augment_kpis(tables, max_domains=5, kpis_per_domain=3)
        assert len(result) == 1
        assert result[0].name == "Revenue"


class TestAugmentKpisHappyPath:
    async def test_successful_pipeline_returns_synthesized_kpis(self):
        client = _SucceedingKPIClient()
        tables = [_make_table("orders"), _make_table("invoices")]
        result = await client.augment_kpis(tables, max_domains=5, kpis_per_domain=3)
        assert len(result) >= 1
        assert all(isinstance(kpi, KPITerm) for kpi in result)

    async def test_kpis_are_kpi_term_instances(self):
        client = _SucceedingKPIClient()
        tables = [_make_table("orders")]
        result = await client.augment_kpis(tables, max_domains=5, kpis_per_domain=3)
        assert all(isinstance(kpi, KPITerm) for kpi in result)


class TestAugmentKpisConfigPropagation:
    async def test_max_domains_is_forwarded_to_phase_a(self):
        client = _MaxDomainsTrackingClient()
        tables = [_make_table("orders")]
        await client.augment_kpis(tables, max_domains=7, kpis_per_domain=3)
        assert _MaxDomainsTrackingClient.captured_max_domains == 7
