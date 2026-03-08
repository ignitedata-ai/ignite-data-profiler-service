"""Abstract base for LLM augmentation clients.

All LLM providers must subclass this.  The base class owns the batch
orchestration loop; concrete subclasses only implement the single-call
I/O via ``_describe_table`` and ``_describe_column``.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from core.api.v1.schemas.profiler import ColumnMetadata, GlossaryTerm, KPITerm, TableMetadata
from core.logging import get_logger

logger = get_logger(__name__)


class BaseLLMClient(ABC):
    """Abstract async LLM client for generating table and column descriptions.

    Subclasses must implement:
        - :meth:`_describe_table`: issue one LLM call for a single table.
        - :meth:`_describe_column`: issue one LLM call for a single column.
        - :attr:`provider_name`: human-readable name for logs.

    The base class owns:
        - Batch slicing logic (``augment_tables``, ``augment_columns``).
        - Concurrent execution within a batch via ``asyncio.gather``.
        - Per-item error isolation: if a description fails the original
          field is left untouched.
    """

    #: Human-readable provider name used in log messages.
    provider_name: str

    @abstractmethod
    async def _describe_table(self, table: TableMetadata) -> str | None:
        """Generate a business-oriented description for a single table.

        Args:
            table: Fully assembled ``TableMetadata`` including columns,
                   sample values, and any existing pg_description.

        Returns:
            A non-empty description string, or ``None`` if the provider
            returned no meaningful output.

        Raises:
            Any exception: callers treat raised exceptions as soft failures —
            the table description will remain as-is.

        """

    @abstractmethod
    async def _describe_column(self, column: ColumnMetadata, table: TableMetadata) -> str | None:
        """Generate a business-oriented description for a single column.

        Args:
            column: The column to describe.
            table: The parent table, used as context for the prompt.

        Returns:
            A non-empty description string, or ``None`` if the provider
            returned no meaningful output.

        Raises:
            Any exception: callers treat raised exceptions as soft failures —
            the column description will remain as-is.

        """

    @abstractmethod
    async def _infer_glossary(self, table: TableMetadata) -> list[GlossaryTerm]:
        """Infer up to 5 business glossary terms for a single table.

        Args:
            table: Fully assembled ``TableMetadata`` including columns,
                   sample values, and any existing description.

        Returns:
            A list of ``GlossaryTerm`` objects (may be empty).

        Raises:
            Any exception: callers treat raised exceptions as soft failures —
            the table glossary will remain as-is.

        """

    @abstractmethod
    async def _cluster_tables_into_domains(
        self,
        tables: list[TableMetadata],
        max_domains: int,
    ) -> dict[str, list[str]]:
        """Phase A — cluster table names into business domain groups.

        Implementations should send only qualified table names (no column
        metadata) to keep the token count minimal.

        Args:
            tables: All tables being profiled.
            max_domains: Upper bound on the number of clusters to produce.

        Returns:
            A dict mapping domain_name -> list of qualified table names
            (``schema.table`` strings). Every table should appear in exactly
            one domain. Return ``{}`` if clustering cannot be performed so
            the caller can skip Phases B and C gracefully.

        Raises:
            Any exception: callers treat raised exceptions as soft failures.

        """

    @abstractmethod
    async def _generate_domain_kpis(
        self,
        domain_name: str,
        domain_tables: list[TableMetadata],
        max_kpis: int,
    ) -> list[KPITerm]:
        """Phase B — generate KPIs for a single business domain.

        Args:
            domain_name: The domain label returned from Phase A.
            domain_tables: Full ``TableMetadata`` objects for tables in this domain.
            max_kpis: Maximum number of KPIs to return for this domain.

        Returns:
            A list of ``KPITerm`` objects (may be empty).

        Raises:
            Any exception: callers treat raised exceptions as soft failures —
            the domain is skipped and other domains are unaffected.

        """

    @abstractmethod
    async def _judge_filter_columns(
        self,
        table_name: str,
        table_role: str,
        candidates: list[dict],
    ) -> list[dict]:
        """Judge whether candidate columns are suitable analytical filters.

        The LLM receives pre-computed evidence (scores, cardinality, data
        type) and acts as a judge — it validates rather than generates.

        Args:
            table_name: Qualified table name (``schema.table``).
            table_role: ``"fact"``, ``"dimension"``, or ``"unknown"``.
            candidates: List of dicts with pre-computed signal data per column.

        Returns:
            A list of dicts, one per candidate, each containing at minimum:
            ``column_name``, ``llm_confidence`` (0.0–1.0),
            ``llm_filter_type``, ``agrees_with_heuristic`` (bool),
            ``reasoning`` (str).

        Raises:
            Any exception: callers treat raised exceptions as soft failures.

        """

    async def augment_tables(
        self,
        tables: list[TableMetadata],
        batch_size: int,
    ) -> list[TableMetadata]:
        """Augment a list of tables with LLM-generated descriptions.

        Tables are processed in parallel batches of ``batch_size``.
        Failures within a batch are caught per-table and logged as
        warnings; they never propagate to the caller.

        Args:
            tables: Tables to augment.  Modified in-place (description
                    field only); the same list is also returned.
            batch_size: Maximum number of concurrent LLM calls per batch.

        Returns:
            The same ``tables`` list, with ``description`` fields updated
            where the LLM call succeeded.

        """
        for batch_start in range(0, len(tables), batch_size):
            batch = tables[batch_start : batch_start + batch_size]
            await asyncio.gather(
                *[self._augment_single(table) for table in batch],
                return_exceptions=True,
            )
        return tables

    async def augment_columns(
        self,
        pairs: list[tuple[ColumnMetadata, TableMetadata]],
        batch_size: int,
    ) -> None:
        """Augment a list of columns with LLM-generated descriptions.

        Columns are processed in parallel batches of ``batch_size``.
        Each column is paired with its parent table for prompt context.
        Failures are caught per-column and logged as warnings.

        Args:
            pairs: ``(column, table)`` pairs to augment.  Columns are
                   modified in-place (description field only).
            batch_size: Maximum number of concurrent LLM calls per batch.

        """
        for batch_start in range(0, len(pairs), batch_size):
            batch = pairs[batch_start : batch_start + batch_size]
            await asyncio.gather(
                *[self._augment_single_column(col, tbl) for col, tbl in batch],
                return_exceptions=True,
            )

    async def _augment_single(self, table: TableMetadata) -> None:
        """Augment a single table in-place, isolating errors."""
        try:
            description = await self._describe_table(table)
            if description:
                table.description = description
        except Exception as exc:
            logger.warning(
                "LLM description failed for table",
                provider=self.provider_name,
                table=f"{table.schema_name}.{table.name}",
                error=str(exc),
            )

    async def _augment_single_column(self, column: ColumnMetadata, table: TableMetadata) -> None:
        """Augment a single column in-place, isolating errors."""
        try:
            description = await self._describe_column(column, table)
            if description:
                column.description = description
        except Exception as exc:
            logger.warning(
                "LLM description failed for column",
                provider=self.provider_name,
                table=f"{table.schema_name}.{table.name}",
                column=column.name,
                error=str(exc),
            )

    async def augment_glossary_terms(
        self,
        tables: list[TableMetadata],
        batch_size: int,
    ) -> list[TableMetadata]:
        """Augment a list of tables with LLM-inferred business glossary terms.

        Tables are processed in parallel batches of ``batch_size``.
        Failures within a batch are caught per-table and logged as
        warnings; they never propagate to the caller.

        Args:
            tables: Tables to augment.  Modified in-place (glossary field
                    only); the same list is also returned.
            batch_size: Maximum number of concurrent LLM calls per batch.

        Returns:
            The same ``tables`` list, with ``glossary`` fields updated
            where the LLM call succeeded.

        """
        for batch_start in range(0, len(tables), batch_size):
            batch = tables[batch_start : batch_start + batch_size]
            await asyncio.gather(
                *[self._augment_single_glossary(table) for table in batch],
                return_exceptions=True,
            )
        return tables

    async def _augment_single_glossary(self, table: TableMetadata) -> None:
        """Augment a single table's glossary in-place, isolating errors."""
        try:
            terms = await self._infer_glossary(table)
            if terms:
                table.glossary = terms
        except Exception as exc:
            logger.warning(
                "LLM glossary inference failed for table",
                provider=self.provider_name,
                table=f"{table.schema_name}.{table.name}",
                error=str(exc),
            )

    async def judge_filter_columns(
        self,
        table_name: str,
        table_role: str,
        candidates: list[dict],
    ) -> list[dict]:
        """Send filter candidates to the LLM for evidence-based judgment.

        Wraps ``_judge_filter_columns`` with error isolation.

        Returns:
            A list of judgment dicts, or an empty list on failure.
        """
        try:
            results = await self._judge_filter_columns(table_name, table_role, candidates)
            logger.debug(
                "LLM filter judgment received",
                provider=self.provider_name,
                table=table_name,
                candidate_count=len(candidates),
                result_count=len(results),
            )
            return results
        except Exception as exc:
            logger.warning(
                "LLM filter judgment failed",
                provider=self.provider_name,
                table=table_name,
                error=str(exc),
            )
            return []

    async def augment_kpis(
        self,
        tables: list[TableMetadata],
        max_domains: int,
        kpis_per_domain: int,
    ) -> list[KPITerm]:
        """Run the three-phase KPI inference Map-Reduce pipeline.

        Phase A: Cluster all table names into business domains via LLM.
        Phase B: For each domain, generate KPIs using full table metadata.
        Phase C: Deduplicate and add cross-domain KPIs.

        All three phases are best-effort. If a phase fails completely, the
        pipeline returns whatever was collected up to that point (possibly []).

        Args:
            tables: All tables from the profiling response.
            max_domains: Forwarded to Phase A (``_cluster_tables_into_domains``).
            kpis_per_domain: Forwarded to Phase B (``_generate_domain_kpis``).

        Returns:
            A flat list of ``KPITerm`` objects, or ``[]`` if the pipeline
            produced nothing useful.

        """
        if not tables:
            return []

        # ── Phase A: cluster table names into domains ─────────────────────────
        try:
            domain_clusters = await self._cluster_tables_into_domains(tables, max_domains)
        except Exception as exc:
            logger.warning(
                "KPI clustering (Phase A) failed — skipping KPI inference",
                provider=self.provider_name,
                error=str(exc),
            )
            return []

        if not domain_clusters:
            logger.info(
                "KPI clustering returned no domains — skipping KPI inference",
                provider=self.provider_name,
                table_count=len(tables),
            )
            return []

        # Build a qualified-name lookup for Phase B table resolution.
        table_lookup: dict[str, TableMetadata] = {f"{t.schema_name}.{t.name}": t for t in tables}

        # ── Phase B: generate KPIs per domain (sequential to avoid token bursts) ─
        all_domain_kpis: list[KPITerm] = []
        for domain_name, table_names in domain_clusters.items():
            domain_tables = [table_lookup[qn] for qn in table_names if qn in table_lookup]
            if not domain_tables:
                logger.warning(
                    "Domain references no known tables — skipping",
                    provider=self.provider_name,
                    domain=domain_name,
                    referenced_tables=table_names,
                )
                continue
            try:
                domain_kpis = await self._generate_domain_kpis(domain_name, domain_tables, kpis_per_domain)
                all_domain_kpis.extend(domain_kpis)
                logger.debug(
                    "KPI generation succeeded for domain",
                    provider=self.provider_name,
                    domain=domain_name,
                    table_count=len(domain_tables),
                    kpi_count=len(domain_kpis),
                )
            except Exception as exc:
                logger.warning(
                    "KPI generation (Phase B) failed for domain — skipping",
                    provider=self.provider_name,
                    domain=domain_name,
                    error=str(exc),
                )

        if not all_domain_kpis:
            logger.info(
                "No KPIs generated across all domains",
                provider=self.provider_name,
            )
            return []

        return all_domain_kpis
