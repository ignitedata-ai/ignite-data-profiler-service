"""LLM cost tracker initialization for the profiler service.

Uses compute_cost() only — no DB writes.  Pricing data is fetched from
LiteLLM on first use and cached locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ignite_llmops_lib.cost_tracker import LLMCostTracker  # type: ignore[import-untyped]  # noqa: F401

# Attempt to import at runtime (graceful degradation if library unavailable)
try:
    from ignite_llmops_lib.cost_tracker import LLMTrackerConfig, init_llm_tracker  # type: ignore[import-untyped]

    _lib_available = True
except ImportError:
    LLMTrackerConfig = None  # type: ignore[assignment,misc]
    init_llm_tracker = None  # type: ignore[assignment]
    _lib_available = False
    logger.warning("ignite-llmops-lib-cost-tracker not installed — cost tracking disabled")

_cost_tracker: LLMCostTracker | None = None


async def init_cost_tracker() -> LLMCostTracker | None:
    """Initialize the global cost tracker.

    Uses the profiler's own SQLite database for the underlying engine
    (required by the library constructor) but never calls .record() —
    only .compute_cost() is used so no cost rows are ever written.

    Returns None (and logs a warning) when the cost-tracker library is
    not installed or pricing fetch fails.
    """
    global _cost_tracker

    if _cost_tracker is not None:
        return _cost_tracker

    if not _lib_available:
        logger.warning("Cost tracker library unavailable — skipping init")
        return None

    try:
        logger.info("Initializing LLM cost tracker (compute-only mode)")

        config = LLMTrackerConfig(
            database_url=settings.DATABASE_URL,
            default_application="ignite-data-profiler-service",
            cache_dir=Path.home() / ".cache" / "ignite_llmops_lib" / "profiler",
        )

        _cost_tracker = await init_llm_tracker(config)  # type: ignore[arg-type]

        logger.info(
            "LLM cost tracker initialized",
            pricing_version=_cost_tracker._registry.version,
        )
        return _cost_tracker

    except Exception as exc:
        logger.error(
            "Failed to initialize LLM cost tracker — costs will not be computed",
            error=str(exc),
            exc_info=True,
        )
        raise


def get_cost_tracker() -> LLMCostTracker | None:
    """Return the global cost tracker, or None if not yet initialised."""
    if _cost_tracker is None:
        logger.debug("Cost tracker accessed before initialization")
    return _cost_tracker
