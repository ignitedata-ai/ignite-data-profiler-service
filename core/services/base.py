"""BaseProfiler — abstract orchestrator for all datasource profilers.

Subclass this for every new datasource.  Only four abstract methods need
implementing; all cross-cutting concerns (tracing, connection lifecycle,
timeout, error mapping, structured logging) live here and are never
repeated.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.services.task_manager import ProgressReporter

from ignite_data_connectors import (
    BaseConnector,
    ConnectorConnectionError,
    QueryError,
    create_connector,
)
from ignite_data_connectors import (
    ConfigurationError as ConnectorConfigError,
)

from core.api.v1.schemas.profiler import ProfilingConfig, ProfilingResponse
from core.exceptions import ConfigurationError, DatabaseError, ExternalServiceError
from core.exceptions.base import InternalTimeoutError, ProfilingTimeoutError
from core.logging import get_logger
from core.observability import get_tracer

logger = get_logger(__name__)
tracer = get_tracer()


class BaseProfilerService(ABC):
    """Abstract base for all datasource profilers.

    Subclasses must declare :attr:`service_name` and :attr:`span_name` as
    class-level strings and implement the four abstract methods below.

    Usage::

        class PostgresProfiler(BaseProfiler):
            service_name = "PostgreSQL"
            span_name = "profiler.postgres"
            ...

        PROFILER_REGISTRY = {"postgres": PostgresProfiler()}
    """

    #: Human-readable service label used in error messages and OTel attributes.
    service_name: str
    #: OpenTelemetry span name for the top-level profiling span.
    span_name: str

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def _span_attributes(self, conn: Any) -> dict[str, str | int]:
        """Return OTel span attributes for this datasource connection."""

    @abstractmethod
    def _log_context(self, conn: Any) -> dict[str, Any]:
        """Return structured log fields for this connection.

        Must include at minimum ``host`` and ``database`` keys so that the
        base orchestrator can produce consistent log output.
        """

    @abstractmethod
    async def _run(
        self,
        connector: BaseConnector,
        config: ProfilingConfig,
        progress: ProgressReporter | None = None,
    ) -> dict[str, Any]:
        """Execute the datasource-specific profiling logic.

        Args:
            connector: An active, already-tested ``BaseConnector``.
            config: Validated ``ProfilingConfig`` from the request.
            progress: Optional progress reporter for background task tracking.

        Returns:
            Raw profiling result dict to be passed to :meth:`_assemble_response`.

        """

    @abstractmethod
    def _assemble_response(
        self,
        raw: dict[str, Any],
        profiled_at: datetime,
    ) -> ProfilingResponse:
        """Convert the raw profiling dict into a ``ProfilingResponse``."""

    # ── Orchestration ──────────────────────────────────────────────────────────

    async def profile(
        self,
        body: Any,
        progress: ProgressReporter | None = None,
    ) -> ProfilingResponse:
        """Run the full profiling flow for a single request.

        Handles tracing, connection lifecycle, timeout enforcement, error
        mapping, and structured logging.  Subclasses should not override this.

        Args:
            body: The validated profiling request body.
            progress: Optional progress reporter for background task tracking.
        """
        conn = body.connection
        cfg = body.config
        ctx = self._log_context(conn)
        profiled_at = datetime.now(UTC)

        with tracer.start_as_current_span(self.span_name) as span:
            for key, value in self._span_attributes(conn).items():
                span.set_attribute(key, value)

            try:
                if progress:
                    progress.update(phase="connecting", percent=5)
                    await progress.flush()

                async with create_connector(conn) as db:
                    await db.test_connection()
                    logger.info("Connection test passed", **ctx)

                    if progress:
                        progress.update(phase="profiling", percent=10)
                        await progress.flush()

                    try:
                        raw = await asyncio.wait_for(
                            self._run(db, cfg, progress=progress),
                            timeout=cfg.timeout_seconds,
                        )
                    except TimeoutError as exc:
                        # asyncio.wait_for raises TimeoutError when the
                        # deadline expires, but so does asyncpg when pool
                        # acquisition or a query times out *inside* _run.
                        # Detect whether the overall deadline actually
                        # elapsed to distinguish the two cases.
                        elapsed = (datetime.now(UTC) - profiled_at).total_seconds()
                        if elapsed >= cfg.timeout_seconds * 0.95:
                            raise  # genuine overall timeout — handled below
                        raise InternalTimeoutError(
                            message=(
                                f"An internal timeout occurred after {elapsed:.0f}s "
                                f"(overall limit is {cfg.timeout_seconds}s). "
                                "This usually means the connection pool is exhausted "
                                "— consider increasing pool_max_size or reducing "
                                "max_concurrent_tables."
                            ),
                            source="pool_or_query",
                        ) from exc

                    logger.info(
                        "Profiling complete",
                        **ctx,
                        schema_count=len(raw.get("schemas", [])),
                    )
                    response = self._assemble_response(raw, profiled_at)

                    if cfg.detect_filter_columns:
                        if progress:
                            progress.update(phase="detecting_filters", percent=93, detail={"augmentation_step": "filter_columns"})
                            await progress.flush()
                        response = await self._detect_filter_columns(response, cfg, db)

            except InternalTimeoutError:
                raise  # already a well-described service exception

            except TimeoutError:
                logger.warning(
                    "Profiling timed out",
                    **ctx,
                    timeout_seconds=cfg.timeout_seconds,
                )
                raise ProfilingTimeoutError(
                    message=f"Profiling exceeded the configured timeout of {cfg.timeout_seconds}s",
                    timeout_seconds=cfg.timeout_seconds,
                ) from None

            except (ConnectorConfigError, ConnectorConnectionError, QueryError) as exc:
                logger.error(
                    "Connector error during profiling",
                    **ctx,
                    error=str(exc),
                    exc_info=True,
                )
                raise self._map_error(exc, ctx) from exc

        # Reset per-task LLM cost accumulator before any augmentation calls.
        from core.utils.llm_config import get_accumulated_stats, reset_cost_accumulator

        reset_cost_accumulator()

        def _phase_log(event: str) -> None:
            s = get_accumulated_stats()
            logger.info(
                event,
                input_tokens=int(s["input_tokens"]),
                output_tokens=int(s["output_tokens"]),
                input_cost_usd=round(s["input_cost"], 8),
                output_cost_usd=round(s["output_cost"], 8),
                total_cost_usd=round(s["total_cost"], 8),
            )

        if cfg.augment_descriptions:
            if progress:
                progress.update(phase="augmenting", percent=70, detail={"augmentation_step": "table_descriptions"})
                await progress.flush()
            response = await self._augment_response(response, cfg)
            _phase_log("LLM cost after table descriptions")

        if cfg.augment_column_descriptions:
            if progress:
                progress.update(phase="augmenting", percent=80, detail={"augmentation_step": "column_descriptions"})
                await progress.flush()
            response = await self._augment_column_response(response, cfg)
            _phase_log("LLM cost after column descriptions")

        if cfg.augment_glossary:
            if progress:
                progress.update(phase="augmenting", percent=85, detail={"augmentation_step": "glossary"})
                await progress.flush()
            response = await self._augment_glossary_response(response, cfg)
            _phase_log("LLM cost after glossary inference")

        if cfg.infer_kpis:
            if progress:
                progress.update(phase="augmenting", percent=90, detail={"augmentation_step": "kpis"})
                await progress.flush()
            response = await self._augment_kpis_response(response, cfg)
            _phase_log("LLM cost after KPI inference")

        if progress:
            progress.update(phase="completed", percent=100)
            await progress.flush()

        # Attach aggregated LLM usage stats for the caller service.
        from core.api.v1.schemas.profiler import LLMUsageStats

        stats = get_accumulated_stats()
        if stats["total_cost"] > 0 or stats["estimated_total_tokens"] > 0 or stats["total_latency_ms"] > 0:
            response.llm_usage = LLMUsageStats(
                input_tokens=int(stats["input_tokens"]),
                output_tokens=int(stats["output_tokens"]),
                input_cost=round(stats["input_cost"], 8),
                output_cost=round(stats["output_cost"], 8),
                total_cost=round(stats["total_cost"], 8),
                estimated_text_tokens=int(stats["estimated_text_tokens"]),
                estimated_overhead_tokens=int(stats["estimated_overhead_tokens"]),
                estimated_total_tokens=int(stats["estimated_total_tokens"]),
                estimated_message_count=int(stats["estimated_message_count"]),
                total_latency_ms=round(stats["total_latency_ms"], 2),
            )
            logger.info(
                "Total LLM cost for profiling run",
                input_tokens=int(stats["input_tokens"]),
                output_tokens=int(stats["output_tokens"]),
                input_cost_usd=round(stats["input_cost"], 8),
                output_cost_usd=round(stats["output_cost"], 8),
                total_cost_usd=round(stats["total_cost"], 8),
                estimated_text_tokens=int(stats["estimated_text_tokens"]),
                estimated_overhead_tokens=int(stats["estimated_overhead_tokens"]),
                estimated_total_tokens=int(stats["estimated_total_tokens"]),
                estimated_message_count=int(stats["estimated_message_count"]),
                total_latency_ms=round(stats["total_latency_ms"], 2),
            )

        return response

    async def _augment_response(
        self,
        response: ProfilingResponse,
        cfg: ProfilingConfig,
    ) -> ProfilingResponse:
        """Augment table descriptions with LLM-generated text.

        Always a best-effort step: if the LLM client is not configured or
        all calls fail, the original ``response`` is returned unmodified.
        Partial success is valid — only successfully described tables are updated.
        """
        from core.llm import get_llm_client

        llm = get_llm_client(
            provider=cfg.llm_provider,
            model=cfg.llm_model,
            portkey_api_key=cfg.portkey_api_key,
            portkey_virtual_key=cfg.portkey_virtual_key,
        )
        if llm is None:
            logger.info("augment_descriptions=True but no LLM client is configured; skipping")
            return response

        all_tables = [table for schema in response.schemas for table in schema.tables]

        if not all_tables:
            return response

        logger.info(
            "Starting LLM description augmentation",
            table_count=len(all_tables),
            batch_size=cfg.llm_batch_size,
            provider=llm.provider_name,
        )

        t_llm = time.monotonic()
        try:
            await llm.augment_tables(all_tables, batch_size=cfg.llm_batch_size)
        except Exception as exc:
            logger.error(
                "LLM augmentation encountered an unexpected error",
                error=str(exc),
                exc_info=True,
            )
        else:
            logger.info(
                "LLM augmentation complete",
                table_count=len(all_tables),
                llm_duration_seconds=round(time.monotonic() - t_llm, 3),
            )

        return response

    async def _augment_column_response(
        self,
        response: ProfilingResponse,
        cfg: ProfilingConfig,
    ) -> ProfilingResponse:
        """Augment column descriptions with LLM-generated text.

        Always a best-effort step: if the LLM client is not configured or
        all calls fail, the original ``response`` is returned unmodified.
        Partial success is valid — only successfully described columns are updated.
        """
        from core.llm import get_llm_client

        llm = get_llm_client(
            provider=cfg.llm_provider,
            model=cfg.llm_model,
            portkey_api_key=cfg.portkey_api_key,
            portkey_virtual_key=cfg.portkey_virtual_key,
        )
        if llm is None:
            logger.info("augment_column_descriptions=True but no LLM client is configured; skipping")
            return response

        pairs = [(column, table) for schema in response.schemas for table in schema.tables for column in table.columns]

        if not pairs:
            return response

        logger.info(
            "Starting LLM column description augmentation",
            column_count=len(pairs),
            batch_size=cfg.llm_column_batch_size,
            provider=llm.provider_name,
        )

        t_llm = time.monotonic()
        try:
            await llm.augment_columns(pairs, batch_size=cfg.llm_column_batch_size)
        except Exception as exc:
            logger.error(
                "LLM column augmentation encountered an unexpected error",
                error=str(exc),
                exc_info=True,
            )
        else:
            logger.info(
                "LLM column augmentation complete",
                column_count=len(pairs),
                llm_duration_seconds=round(time.monotonic() - t_llm, 3),
            )

        return response

    async def _augment_glossary_response(
        self,
        response: ProfilingResponse,
        cfg: ProfilingConfig,
    ) -> ProfilingResponse:
        """Infer business glossary terms for each table via LLM.

        Always a best-effort step: if the LLM client is not configured or
        all calls fail, the original ``response`` is returned unmodified.
        Partial success is valid — only successfully processed tables are updated.
        """
        from core.llm import get_llm_client

        llm = get_llm_client(
            provider=cfg.llm_provider,
            model=cfg.llm_model,
            portkey_api_key=cfg.portkey_api_key,
            portkey_virtual_key=cfg.portkey_virtual_key,
        )
        if llm is None:
            logger.info("augment_glossary=True but no LLM client is configured; skipping")
            return response

        all_tables = [table for schema in response.schemas for table in schema.tables]

        if not all_tables:
            return response

        logger.info(
            "Starting LLM glossary inference",
            table_count=len(all_tables),
            batch_size=cfg.llm_glossary_batch_size,
            provider=llm.provider_name,
        )

        t_llm = time.monotonic()
        try:
            await llm.augment_glossary_terms(all_tables, batch_size=cfg.llm_glossary_batch_size)
        except Exception as exc:
            logger.error(
                "LLM glossary inference encountered an unexpected error",
                error=str(exc),
                exc_info=True,
            )
        else:
            logger.info(
                "LLM glossary inference complete",
                table_count=len(all_tables),
                llm_duration_seconds=round(time.monotonic() - t_llm, 3),
            )

        return response

    async def _augment_kpis_response(
        self,
        response: ProfilingResponse,
        cfg: ProfilingConfig,
    ) -> ProfilingResponse:
        """Infer business KPIs via the three-phase Map-Reduce LLM pipeline.

        Always a best-effort step: if the LLM client is not configured or
        all phases fail, the original ``response`` is returned with
        ``kpis=None`` (not requested) preserved as-is.
        """
        from core.llm import get_llm_client

        llm = get_llm_client(
            provider=cfg.llm_provider,
            model=cfg.llm_model,
            portkey_api_key=cfg.portkey_api_key,
            portkey_virtual_key=cfg.portkey_virtual_key,
        )
        if llm is None:
            logger.info("infer_kpis=True but no LLM client is configured; skipping")
            return response

        all_tables = [table for schema in response.schemas for table in schema.tables]

        if not all_tables:
            response.kpis = []
            return response

        logger.info(
            "Starting LLM KPI inference",
            table_count=len(all_tables),
            max_domains=cfg.llm_kpi_max_domains,
            kpis_per_domain=cfg.llm_kpis_per_domain,
            provider=llm.provider_name,
        )

        t_llm = time.monotonic()
        try:
            kpis = await llm.augment_kpis(
                all_tables,
                max_domains=cfg.llm_kpi_max_domains,
                kpis_per_domain=cfg.llm_kpis_per_domain,
            )
            response.kpis = kpis
        except Exception as exc:
            logger.error(
                "LLM KPI inference encountered an unexpected error",
                error=str(exc),
                exc_info=True,
            )
        else:
            logger.info(
                "LLM KPI inference complete",
                kpi_count=len(kpis),
                llm_duration_seconds=round(time.monotonic() - t_llm, 3),
            )

        return response

    async def _detect_filter_columns(
        self,
        response: ProfilingResponse,
        cfg: ProfilingConfig,
        connector: BaseConnector | None = None,
    ) -> ProfilingResponse:
        """Run the filter column detection pipeline.

        Always a best-effort step: if any stage fails, the original
        ``response`` is returned unmodified.
        """
        from core.llm import get_llm_client
        from core.services.filters.pipeline import FilterDetectionPipeline
        from core.services.filters.schema_introspector import INTROSPECTOR_REGISTRY, NullSchemaIntrospector

        introspector_cls = INTROSPECTOR_REGISTRY.get(
            getattr(self, "_datasource_type", ""),
            NullSchemaIntrospector,
        )
        introspector = introspector_cls()
        pipeline = FilterDetectionPipeline(introspector)

        llm = get_llm_client()

        logger.info("Starting filter column detection")
        t0 = time.monotonic()
        try:
            response = await pipeline.detect(response, connector=connector, llm=llm)
        except Exception as exc:
            logger.error(
                "Filter column detection encountered an unexpected error",
                error=str(exc),
                exc_info=True,
            )
        else:
            total_filters = sum(len(t.filter_columns or []) for s in response.schemas for t in s.tables)
            logger.info(
                "Filter column detection complete",
                filter_count=total_filters,
                duration_seconds=round(time.monotonic() - t0, 3),
            )

        return response

    # ── Error mapping ──────────────────────────────────────────────────────────

    def _map_error(self, exc: Exception, ctx: dict[str, Any]) -> Exception:
        """Convert a connector exception into the service exception hierarchy."""
        host = ctx.get("host", "unknown")
        database = ctx.get("database", "unknown")

        if isinstance(exc, ConnectorConfigError):
            return ConfigurationError(
                message=f"Invalid database connection configuration: {exc}",
            )
        if isinstance(exc, ConnectorConnectionError):
            return ExternalServiceError(
                message=f"Cannot connect to {self.service_name} at {host}/{database}: {exc}",
                service_name=self.service_name,
            )
        if isinstance(exc, QueryError):
            return DatabaseError(
                message=f"SQL execution error during profiling: {exc}",
                operation="profiling_query",
            )
        return exc
