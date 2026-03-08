"""Tests for BaseLLMClient batching orchestration and error isolation."""

from __future__ import annotations

from core.api.v1.schemas.profiler import ColumnMetadata, GlossaryTerm, KPITerm, TableMetadata
from core.llm.base import BaseLLMClient

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_column(name: str = "id", description: str | None = None) -> ColumnMetadata:
    return ColumnMetadata(
        name=name,
        ordinal_position=1,
        data_type="integer",
        is_nullable=False,
        column_default=None,
        character_maximum_length=None,
        numeric_precision=32,
        numeric_scale=0,
        description=description,
    )


def _make_table(name: str, schema: str = "public", description: str | None = None) -> TableMetadata:
    return TableMetadata(
        name=name,
        schema=schema,
        owner="postgres",
        description=description,
        row_count=100,
        size_bytes=None,
        total_size_bytes=None,
        data_freshness=None,
        columns=[_make_column()],
        indexes=None,
        relationships=None,
    )


# ── Concrete test subclasses ───────────────────────────────────────────────────


def _make_glossary_term(term: str) -> GlossaryTerm:
    return GlossaryTerm(business_term=term, description=f"Description of {term}", synonyms=[])


class _SucceedingClient(BaseLLMClient):
    provider_name = "test"

    async def _describe_table(self, table: TableMetadata) -> str | None:
        return f"Description for {table.name}"

    async def _describe_column(self, column: ColumnMetadata, table: TableMetadata) -> str | None:
        return f"Description for {table.name}.{column.name}"

    async def _infer_glossary(self, table: TableMetadata) -> list[GlossaryTerm]:
        return [_make_glossary_term(f"Term for {table.name}")]

    async def _cluster_tables_into_domains(self, tables: list[TableMetadata], max_domains: int) -> dict[str, list[str]]:
        return {"General": [f"{t.schema_name}.{t.name}" for t in tables]}

    async def _generate_domain_kpis(self, domain_name: str, domain_tables: list[TableMetadata], max_kpis: int) -> list[KPITerm]:
        return [KPITerm(name=f"KPI for {domain_name}", description="A KPI", domain=domain_name)]

    async def _synthesize_kpis(self, all_domain_kpis: list[KPITerm]) -> list[KPITerm]:
        return all_domain_kpis

    async def _judge_filter_columns(self, table_name: str, table_role: str, candidates: list[dict]) -> list[dict]:
        return []


class _FailingClient(BaseLLMClient):
    provider_name = "test"

    async def _describe_table(self, table: TableMetadata) -> str | None:
        raise RuntimeError("LLM API error")

    async def _describe_column(self, column: ColumnMetadata, table: TableMetadata) -> str | None:
        raise RuntimeError("LLM API error")

    async def _infer_glossary(self, table: TableMetadata) -> list[GlossaryTerm]:
        raise RuntimeError("LLM API error")

    async def _cluster_tables_into_domains(self, tables: list[TableMetadata], max_domains: int) -> dict[str, list[str]]:
        raise RuntimeError("LLM API error")

    async def _generate_domain_kpis(self, domain_name: str, domain_tables: list[TableMetadata], max_kpis: int) -> list[KPITerm]:
        raise RuntimeError("LLM API error")

    async def _synthesize_kpis(self, all_domain_kpis: list[KPITerm]) -> list[KPITerm]:
        raise RuntimeError("LLM API error")

    async def _judge_filter_columns(self, table_name: str, table_role: str, candidates: list[dict]) -> list[dict]:
        raise RuntimeError("LLM API error")


class _PartialClient(BaseLLMClient):
    """Fails for tables named 'bad', succeeds for all others."""

    provider_name = "test"

    async def _describe_table(self, table: TableMetadata) -> str | None:
        if table.name == "bad":
            raise RuntimeError("deliberate failure")
        return f"Description for {table.name}"

    async def _describe_column(self, column: ColumnMetadata, table: TableMetadata) -> str | None:
        if column.name == "bad":
            raise RuntimeError("deliberate failure")
        return f"Description for {table.name}.{column.name}"

    async def _infer_glossary(self, table: TableMetadata) -> list[GlossaryTerm]:
        if table.name == "bad":
            raise RuntimeError("deliberate failure")
        return [_make_glossary_term(f"Term for {table.name}")]

    async def _cluster_tables_into_domains(self, tables: list[TableMetadata], max_domains: int) -> dict[str, list[str]]:
        return {"General": [f"{t.schema_name}.{t.name}" for t in tables]}

    async def _generate_domain_kpis(self, domain_name: str, domain_tables: list[TableMetadata], max_kpis: int) -> list[KPITerm]:
        return [KPITerm(name=f"KPI for {domain_name}", description="A KPI", domain=domain_name)]

    async def _synthesize_kpis(self, all_domain_kpis: list[KPITerm]) -> list[KPITerm]:
        return all_domain_kpis

    async def _judge_filter_columns(self, table_name: str, table_role: str, candidates: list[dict]) -> list[dict]:
        return []


class _NoneReturningClient(BaseLLMClient):
    """Returns None / empty list for all tables (empty response)."""

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


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestBatchAugmentation:
    async def test_successful_augmentation_sets_descriptions(self):
        client = _SucceedingClient()
        tables = [_make_table("users"), _make_table("orders")]
        await client.augment_tables(tables, batch_size=2)
        assert tables[0].description == "Description for users"
        assert tables[1].description == "Description for orders"

    async def test_failed_table_does_not_affect_siblings(self):
        """A table that raises must not clear descriptions of its batch siblings."""
        client = _PartialClient()
        tables = [_make_table("users"), _make_table("bad"), _make_table("orders")]
        await client.augment_tables(tables, batch_size=3)
        assert tables[0].description == "Description for users"
        assert tables[1].description is None  # failed — original preserved
        assert tables[2].description == "Description for orders"

    async def test_all_tables_fail_does_not_raise(self):
        """Total LLM failure must not raise an exception."""
        client = _FailingClient()
        tables = [_make_table("t1"), _make_table("t2")]
        await client.augment_tables(tables, batch_size=2)
        assert tables[0].description is None
        assert tables[1].description is None

    async def test_batching_processes_all_tables(self):
        """Every table must be processed exactly once regardless of batch size."""
        call_count: dict[str, int] = {}

        class _TrackingClient(BaseLLMClient):
            provider_name = "test"

            async def _describe_table(self, table: TableMetadata) -> str | None:
                call_count[table.name] = call_count.get(table.name, 0) + 1
                return f"desc:{table.name}"

            async def _describe_column(self, column: ColumnMetadata, table: TableMetadata) -> str | None:
                return None

            async def _infer_glossary(self, table: TableMetadata) -> list[GlossaryTerm]:
                return []

            async def _cluster_tables_into_domains(self, tables: list[TableMetadata], max_domains: int) -> dict[str, list[str]]:
                return {}

            async def _generate_domain_kpis(
                self, domain_name: str, domain_tables: list[TableMetadata], max_kpis: int
            ) -> list[KPITerm]:
                return []

            async def _synthesize_kpis(self, all_domain_kpis: list[KPITerm]) -> list[KPITerm]:
                return all_domain_kpis

            async def _judge_filter_columns(self, table_name: str, table_role: str, candidates: list[dict]) -> list[dict]:
                return []

        client = _TrackingClient()
        tables = [_make_table(f"t{i}") for i in range(7)]
        await client.augment_tables(tables, batch_size=3)

        assert len(call_count) == 7
        assert all(v == 1 for v in call_count.values())
        assert all(t.description is not None for t in tables)

    async def test_empty_tables_list_is_noop(self):
        client = _SucceedingClient()
        result = await client.augment_tables([], batch_size=5)
        assert result == []

    async def test_none_returning_client_leaves_description_unchanged(self):
        """A provider returning None must not overwrite the existing description."""
        client = _NoneReturningClient()
        tables = [_make_table("users", description="Existing description")]
        await client.augment_tables(tables, batch_size=1)
        assert tables[0].description == "Existing description"

    async def test_batch_size_one_processes_all(self):
        """Batch size of 1 (sequential) must still process all tables."""
        client = _SucceedingClient()
        tables = [_make_table(f"t{i}") for i in range(5)]
        await client.augment_tables(tables, batch_size=1)
        assert all(t.description == f"Description for t{i}" for i, t in enumerate(tables))

    async def test_augment_tables_returns_the_same_list(self):
        """Return value must be the same list object (mutation in-place)."""
        client = _SucceedingClient()
        tables = [_make_table("users")]
        result = await client.augment_tables(tables, batch_size=5)
        assert result is tables


class TestColumnBatchAugmentation:
    def _pairs(self, col_names: list[str], table_name: str = "users") -> list:
        table = _make_table(table_name)
        return [(_make_column(n), table) for n in col_names]

    async def test_successful_augmentation_sets_descriptions(self):
        client = _SucceedingClient()
        table = _make_table("orders")
        col_a = _make_column("id")
        col_b = _make_column("total")
        await client.augment_columns([(col_a, table), (col_b, table)], batch_size=2)
        assert col_a.description == "Description for orders.id"
        assert col_b.description == "Description for orders.total"

    async def test_failed_column_does_not_affect_siblings(self):
        client = _PartialClient()
        table = _make_table("users")
        col_good = _make_column("id")
        col_bad = _make_column("bad")
        col_other = _make_column("email")
        await client.augment_columns([(col_good, table), (col_bad, table), (col_other, table)], batch_size=3)
        assert col_good.description == "Description for users.id"
        assert col_bad.description is None  # failed — original preserved
        assert col_other.description == "Description for users.email"

    async def test_all_columns_fail_does_not_raise(self):
        client = _FailingClient()
        table = _make_table("t1")
        col_a = _make_column("a")
        col_b = _make_column("b")
        await client.augment_columns([(col_a, table), (col_b, table)], batch_size=2)
        assert col_a.description is None
        assert col_b.description is None

    async def test_batching_processes_all_columns(self):
        call_count: dict[str, int] = {}

        class _TrackingClient(BaseLLMClient):
            provider_name = "test"

            async def _describe_table(self, table: TableMetadata) -> str | None:
                return None

            async def _describe_column(self, column: ColumnMetadata, table: TableMetadata) -> str | None:
                key = f"{table.name}.{column.name}"
                call_count[key] = call_count.get(key, 0) + 1
                return f"desc:{key}"

            async def _infer_glossary(self, table: TableMetadata) -> list[GlossaryTerm]:
                return []

            async def _cluster_tables_into_domains(self, tables: list[TableMetadata], max_domains: int) -> dict[str, list[str]]:
                return {}

            async def _generate_domain_kpis(
                self, domain_name: str, domain_tables: list[TableMetadata], max_kpis: int
            ) -> list[KPITerm]:
                return []

            async def _synthesize_kpis(self, all_domain_kpis: list[KPITerm]) -> list[KPITerm]:
                return all_domain_kpis

            async def _judge_filter_columns(self, table_name: str, table_role: str, candidates: list[dict]) -> list[dict]:
                return []

        client = _TrackingClient()
        table = _make_table("users")
        pairs = [(_make_column(f"col_{i}"), table) for i in range(7)]
        await client.augment_columns(pairs, batch_size=3)

        assert len(call_count) == 7
        assert all(v == 1 for v in call_count.values())
        assert all(col.description is not None for col, _ in pairs)

    async def test_empty_pairs_list_is_noop(self):
        client = _SucceedingClient()
        await client.augment_columns([], batch_size=5)  # must not raise

    async def test_none_returning_client_leaves_description_unchanged(self):
        client = _NoneReturningClient()
        table = _make_table("users")
        col = _make_column("id", description="Existing description")
        await client.augment_columns([(col, table)], batch_size=1)
        assert col.description == "Existing description"

    async def test_batch_size_one_processes_all(self):
        client = _SucceedingClient()
        table = _make_table("users")
        cols = [_make_column(f"c{i}") for i in range(5)]
        await client.augment_columns([(c, table) for c in cols], batch_size=1)
        assert all(c.description == f"Description for users.c{i}" for i, c in enumerate(cols))


class TestGlossaryAugmentation:
    async def test_successful_inference_sets_glossary(self):
        client = _SucceedingClient()
        tables = [_make_table("users"), _make_table("orders")]
        await client.augment_glossary_terms(tables, batch_size=2)
        assert tables[0].glossary is not None
        assert tables[0].glossary[0].business_term == "Term for users"
        assert tables[1].glossary is not None
        assert tables[1].glossary[0].business_term == "Term for orders"

    async def test_failed_table_does_not_affect_siblings(self):
        """A table that raises must not clear glossary of its batch siblings."""
        client = _PartialClient()
        tables = [_make_table("users"), _make_table("bad"), _make_table("orders")]
        await client.augment_glossary_terms(tables, batch_size=3)
        assert tables[0].glossary is not None
        assert tables[1].glossary is None  # failed — original preserved
        assert tables[2].glossary is not None

    async def test_all_tables_fail_does_not_raise(self):
        """Total LLM failure must not raise an exception."""
        client = _FailingClient()
        tables = [_make_table("t1"), _make_table("t2")]
        await client.augment_glossary_terms(tables, batch_size=2)
        assert tables[0].glossary is None
        assert tables[1].glossary is None

    async def test_batching_processes_all_tables(self):
        """Every table must be processed exactly once regardless of batch size."""
        call_count: dict[str, int] = {}

        class _TrackingClient(BaseLLMClient):
            provider_name = "test"

            async def _describe_table(self, table: TableMetadata) -> str | None:
                return None

            async def _describe_column(self, column: ColumnMetadata, table: TableMetadata) -> str | None:
                return None

            async def _infer_glossary(self, table: TableMetadata) -> list[GlossaryTerm]:
                call_count[table.name] = call_count.get(table.name, 0) + 1
                return [_make_glossary_term(f"Term for {table.name}")]

            async def _cluster_tables_into_domains(self, tables: list[TableMetadata], max_domains: int) -> dict[str, list[str]]:
                return {}

            async def _generate_domain_kpis(
                self, domain_name: str, domain_tables: list[TableMetadata], max_kpis: int
            ) -> list[KPITerm]:
                return []

            async def _synthesize_kpis(self, all_domain_kpis: list[KPITerm]) -> list[KPITerm]:
                return all_domain_kpis

            async def _judge_filter_columns(self, table_name: str, table_role: str, candidates: list[dict]) -> list[dict]:
                return []

        client = _TrackingClient()
        tables = [_make_table(f"t{i}") for i in range(7)]
        await client.augment_glossary_terms(tables, batch_size=3)

        assert len(call_count) == 7
        assert all(v == 1 for v in call_count.values())
        assert all(t.glossary is not None for t in tables)

    async def test_empty_tables_list_is_noop(self):
        client = _SucceedingClient()
        result = await client.augment_glossary_terms([], batch_size=5)
        assert result == []

    async def test_empty_returning_client_leaves_glossary_as_none(self):
        """A provider returning [] must not set glossary on the table."""
        client = _NoneReturningClient()
        tables = [_make_table("users")]
        await client.augment_glossary_terms(tables, batch_size=1)
        assert tables[0].glossary is None

    async def test_augment_glossary_terms_returns_the_same_list(self):
        """Return value must be the same list object (mutation in-place)."""
        client = _SucceedingClient()
        tables = [_make_table("users")]
        result = await client.augment_glossary_terms(tables, batch_size=5)
        assert result is tables

    async def test_batch_size_one_processes_all(self):
        """Batch size of 1 (sequential) must still process all tables."""
        client = _SucceedingClient()
        tables = [_make_table(f"t{i}") for i in range(5)]
        await client.augment_glossary_terms(tables, batch_size=1)
        assert all(t.glossary is not None for t in tables)
