"""Filter column detection pipeline orchestrator.

Runs Stages 1 → 2 → 4 per table and attaches results to the
``ProfilingResponse``.  Stage 3 (cross-column analysis) is deferred.
"""

from __future__ import annotations

from typing import Any

from core.api.v1.schemas.profiler import FilterColumnInfo, ProfilingResponse, TableMetadata
from core.llm.base import BaseLLMClient
from core.logging import get_logger
from core.services.filters.composite_scorer import (
    FINAL_INCLUDE_THRESHOLD,
    classify_filter_type,
    compute_composite_score,
    triage_candidates,
)
from core.services.filters.llm_judge import build_llm_payload, reconcile
from core.services.filters.models import ColumnFilterCandidate
from core.services.filters.schema_introspector import BaseSchemaIntrospector
from core.services.filters.statistical_scorer import compute_statistical_signals

logger = get_logger(__name__)


class FilterDetectionPipeline:
    """Orchestrate the multi-stage filter column detection pipeline."""

    def __init__(self, schema_introspector: BaseSchemaIntrospector) -> None:
        self._introspector = schema_introspector

    async def detect(
        self,
        response: ProfilingResponse,
        connector: Any | None = None,
        llm: BaseLLMClient | None = None,
    ) -> ProfilingResponse:
        """Run filter detection on every table in the response.

        Modifies ``TableMetadata`` objects in-place (sets ``table_role``
        and ``filter_columns``).
        """
        row_count_lookup: dict[str, int | None] = {
            f"{t.schema_name}.{t.name}": t.row_count for s in response.schemas for t in s.tables
        }

        for schema in response.schemas:
            for table in schema.tables:
                try:
                    await self._detect_for_table(table, connector, llm, row_count_lookup)
                except Exception as exc:
                    logger.warning(
                        "Filter detection failed for table",
                        table=f"{table.schema_name}.{table.name}",
                        error=str(exc),
                    )
        return response

    async def _detect_for_table(
        self,
        table: TableMetadata,
        connector: Any | None,
        llm: BaseLLMClient | None,
        row_count_lookup: dict[str, int | None] | None = None,
    ) -> None:
        row_count = table.row_count or 0

        # ── Stage 1: Schema signals ──────────────────────────────────────
        signals = self._introspector.extract_signals(table)

        # Fetch additional schema data if connector available
        if connector is not None:
            check_constraints = await self._introspector.fetch_check_constraints(
                connector,
                table.schema_name,
                table.name,
            )
            # Resolve FK referenced table row counts (prefer lookup, fallback to query)
            for _, sig in signals.items():
                if sig.is_foreign_key and sig.fk_referenced_table:
                    if row_count_lookup and sig.fk_referenced_table in row_count_lookup:
                        sig.fk_referenced_table_row_count = row_count_lookup[sig.fk_referenced_table]
                    else:
                        parts = sig.fk_referenced_table.split(".", 1)
                        ref_schema = parts[0] if len(parts) == 2 else table.schema_name
                        ref_table = parts[-1]
                        sig.fk_referenced_table_row_count = await self._introspector.fetch_referenced_table_row_count(
                            connector,
                            ref_schema,
                            ref_table,
                        )
        else:
            check_constraints = {}

        self._introspector.score_signals(signals, check_constraints)

        # Table role classification
        table.table_role = self._introspector.classify_table_role(table)

        # ── Stage 2: Statistical signals ─────────────────────────────────
        candidates: list[ColumnFilterCandidate] = []
        for col in table.columns:
            stat_signals = compute_statistical_signals(col, row_count)
            schema_sig = signals.get(col.name)

            composite = compute_composite_score(schema_sig, stat_signals)
            candidate = ColumnFilterCandidate(
                column_name=col.name,
                data_type=col.data_type,
                is_primary_key=col.is_primary_key,
                schema_signals=schema_sig,
                statistical_signals=stat_signals,
                composite_score=composite,
            )
            candidate.preliminary_filter_type = classify_filter_type(candidate)
            candidates.append(candidate)

        # ── Stage 4: Triage + LLM judge ──────────────────────────────────
        auto_accept, _auto_reject, review_band = triage_candidates(candidates)

        # Build results from auto-accepted columns
        results: list[FilterColumnInfo] = []
        for c in auto_accept:
            results.append(self._to_filter_info(c, "heuristic_only"))

        # LLM judge for review band
        if llm and review_band:
            judgments = await llm.judge_filter_columns(
                table_name=f"{table.schema_name}.{table.name}",
                table_role=table.table_role or "unknown",
                candidates=build_llm_payload(
                    f"{table.schema_name}.{table.name}",
                    table.table_role or "unknown",
                    row_count,
                    review_band,
                ),
            )
            # Index judgments by column name
            judgment_map = {j["column_name"]: j for j in judgments}

            for c in review_band:
                j = judgment_map.get(c.column_name)
                if j:
                    has_schema = c.schema_signals is not None and c.schema_signals.schema_score > 0
                    final_score, source = reconcile(
                        c.composite_score,
                        j["llm_confidence"],
                        has_schema,
                    )
                    if final_score >= FINAL_INCLUDE_THRESHOLD:
                        results.append(
                            FilterColumnInfo(
                                column_name=c.column_name,
                                confidence=final_score,
                                confidence_source=source,
                                filter_type=j.get("llm_filter_type", c.preliminary_filter_type),
                                reasoning=j.get("reasoning"),
                            )
                        )
                else:
                    # LLM didn't return judgment — use heuristic only
                    if c.composite_score >= FINAL_INCLUDE_THRESHOLD:
                        results.append(self._to_filter_info(c, "heuristic_only"))
        else:
            # No LLM available — include review band columns above threshold
            for c in review_band:
                if c.composite_score >= FINAL_INCLUDE_THRESHOLD:
                    results.append(self._to_filter_info(c, "heuristic_only"))

        # Sort by confidence descending
        results.sort(key=lambda r: r.confidence, reverse=True)
        table.filter_columns = results if results else None

        logger.debug(
            "Filter detection complete for table",
            table=f"{table.schema_name}.{table.name}",
            table_role=table.table_role,
            filter_count=len(results),
            total_columns=len(table.columns),
        )

    @staticmethod
    def _to_filter_info(c: ColumnFilterCandidate, source: str) -> FilterColumnInfo:
        return FilterColumnInfo(
            column_name=c.column_name,
            confidence=c.composite_score,
            confidence_source=source,
            filter_type=c.preliminary_filter_type,
            reasoning=None,
        )
