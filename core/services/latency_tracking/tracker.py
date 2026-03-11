"""LLM Latency Tracker singleton for the profiler service.

In-memory only — no DB writes.  Initialised once at application startup via
:func:`init_latency_tracker`.  All other modules call :func:`get_latency_tracker`.
"""

from __future__ import annotations

from typing import Any

from core.logging import get_logger

logger = get_logger(__name__)

_latency_tracker: Any | None = None
_latency_tracking_available = False

try:
    from ignite_llmops_lib.latency_tracker import init_llm_latency_tracker  # type: ignore[import-untyped]

    _latency_tracking_available = True
except ImportError:
    init_llm_latency_tracker = None  # type: ignore[assignment]
    logger.warning("Latency tracker library not importable — latency tracking disabled")


def init_latency_tracker() -> Any | None:
    """Initialise the global LLM latency tracker.

    Safe to call more than once; subsequent calls are no-ops.

    Returns:
        The tracker instance, or ``None`` if the library is unavailable.
    """
    global _latency_tracker

    if not _latency_tracking_available or init_llm_latency_tracker is None:
        return None

    if _latency_tracker is not None:
        return _latency_tracker

    try:
        logger.info("Initializing LLM latency tracker")
        _latency_tracker = init_llm_latency_tracker()
        logger.info("LLM latency tracker initialized successfully")
        return _latency_tracker
    except Exception as exc:
        logger.warning("Failed to initialize LLM latency tracker", error=str(exc))
        return None


def get_latency_tracker() -> Any | None:
    """Return the global latency tracker instance (or ``None`` if unavailable)."""
    return _latency_tracker
