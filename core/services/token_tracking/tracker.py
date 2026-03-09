"""LLM Token Counter singleton for the profiler service.

Provides a module-level counter instance that can be used to estimate
token counts before LLM calls.  Initialised once at application startup via
:func:`init_token_counter`.  All other modules call :func:`get_token_counter`.
"""

from __future__ import annotations

from typing import Any

from core.logging import get_logger

logger = get_logger(__name__)

_token_counter: Any | None = None
_token_counting_available = False

try:
    from ignite_llmops_lib.token_counter import LLMMessage, LLMMessageRole, init_llm_counter  # type: ignore[import-untyped]  # noqa: F401

    _token_counting_available = True
except ImportError:
    init_llm_counter = None  # type: ignore[assignment]
    LLMMessage = None  # type: ignore[assignment,misc]
    LLMMessageRole = None  # type: ignore[assignment,misc]
    logger.warning("Token counter library not importable — pre-call token estimation disabled")


def init_token_counter() -> Any | None:
    """Initialise the global LLM token counter.

    Safe to call more than once; subsequent calls are no-ops.

    Returns:
        The counter instance, or ``None`` if the library is unavailable.
    """
    global _token_counter

    if not _token_counting_available or init_llm_counter is None:
        return None

    if _token_counter is not None:
        return _token_counter

    try:
        logger.info("Initializing LLM token counter")
        _token_counter = init_llm_counter()
        logger.info("LLM token counter initialized successfully")
        return _token_counter
    except Exception as exc:
        logger.warning("Failed to initialize LLM token counter", error=str(exc))
        return None


def get_token_counter() -> Any | None:
    """Return the global token counter instance (or ``None`` if unavailable)."""
    return _token_counter
