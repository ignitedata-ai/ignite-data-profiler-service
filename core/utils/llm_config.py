"""Centralised LLM chat-completion helper.

All LLM provider clients must call :func:`get_llm_response` instead of
invoking ``client.chat.completions.create`` directly.  This single entry-point
handles cost computation, request/response logging, and exposes a per-task
cost accumulator via :func:`reset_cost_accumulator` /
:func:`get_accumulated_cost`.

Usage::

    from core.utils.llm_config import get_llm_response, reset_cost_accumulator, get_accumulated_cost

    reset_cost_accumulator()
    raw = await get_llm_response(
        self._client,
        system_prompt="You are a helpful assistant.",
        user_prompt="Describe this table...",
        model="gpt-4o",
        provider="openai",
        temperature=0.2,
        max_tokens=512,
    )
    total_cost = get_accumulated_cost()
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from typing import Any

from portkey_ai import AsyncPortkey

from core.logging import get_logger

logger = get_logger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def extract_json(text: str) -> str:
    """Extract JSON from a markdown code block, or return the text unchanged.

    Handles both ````json`` and bare ```` ``` ```` fences.  When no fence is
    present the original string is returned so it can be passed directly to
    ``json.loads``.

    Args:
        text: Raw LLM response content.

    Returns:
        The extracted JSON string (stripped), or *text* itself if no code
        fence was found.

    """
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


# ── Cost tracking ─────────────────────────────────────────────────────────────
# Per-asyncio-task accumulator stored as a mutable dict so that mutations inside
# asyncio.gather() child tasks are visible to the parent task. ContextVar copies
# the *reference* to the dict into child tasks; in-place mutations to the dict's
# values are shared, whereas reassigning the ContextVar itself is not.
_ACCUMULATOR_ZERO: dict[str, float] = {
    "input_tokens": 0.0,
    "output_tokens": 0.0,
    "input_cost": 0.0,
    "output_cost": 0.0,
    "total_cost": 0.0,
    "estimated_text_tokens": 0.0,
    "estimated_overhead_tokens": 0.0,
    "estimated_total_tokens": 0.0,
    "estimated_message_count": 0.0,
    "total_latency_ms": 0.0,
}
_llm_cost_accumulator: ContextVar[dict[str, float] | None] = ContextVar(
    "_llm_cost_accumulator",
    default=None,
)


def _get_accumulator() -> dict[str, float]:
    """Return the current accumulator, initialising it if not yet set in this context."""
    acc = _llm_cost_accumulator.get()
    if acc is None:
        acc = dict(_ACCUMULATOR_ZERO)
        _llm_cost_accumulator.set(acc)
    return acc


# Attempt to import cost tracking types (graceful degradation if unavailable)
try:
    from ignite_llmops_lib.cost_tracker import (
        LLMCallInput,
        LLMCallStatus,
        LLMTokenUsage,
    )

    _cost_tracking_available = True
except ImportError:
    LLMCallInput = None
    LLMCallStatus = None
    LLMTokenUsage = None
    _cost_tracking_available = False
    logger.warning("Cost tracking library not importable — costs will not be computed")

# Attempt to import token counter types (graceful degradation if unavailable)
try:
    from ignite_llmops_lib.token_counter import LLMMessage, LLMMessageRole

    _token_counting_available = True
except ImportError:
    LLMMessage = None
    LLMMessageRole = None
    _token_counting_available = False
    logger.warning("Token counter library not importable — pre-call token estimation disabled")

# Attempt to import latency tracking context manager (graceful degradation if unavailable)
try:
    from ignite_llmops_lib.latency_tracker.context import llm_track_latency

    _latency_tracking_available = True
except ImportError:
    llm_track_latency = None
    _latency_tracking_available = False
    logger.warning("Latency tracker library not importable — per-call latency tracking disabled")


def reset_cost_accumulator() -> None:
    """Reset the per-task LLM cost accumulator to zero.

    Call this at the start of each profiling run so that accumulated costs
    reflect only the current request.
    """
    _llm_cost_accumulator.set(dict(_ACCUMULATOR_ZERO))


def get_accumulated_cost() -> float:
    """Return the total USD cost accumulated in the current async task."""
    return _get_accumulator()["total_cost"]


def get_accumulated_stats() -> dict[str, float]:
    """Return a snapshot of all accumulated LLM usage stats for the current task.

    Keys: ``input_tokens``, ``output_tokens``, ``input_cost``, ``output_cost``,
    ``total_cost``.
    """
    return dict(_get_accumulator())


async def get_llm_response(
    client: AsyncPortkey,
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    provider: str,
    temperature: float,
    max_tokens: int,
    response_format: dict[str, Any] | None = None,
) -> str | None:
    """Issue a single chat-completion request and return the raw text content.

    This is the **single call-site** for all LLM invocations in the profiler
    service.  In addition to returning the text response, it computes the cost
    of each call via :mod:`ignite_llmops_lib.cost_tracker` and accumulates the
    total in a per-task :class:`~contextvars.ContextVar`.

    Use :func:`reset_cost_accumulator` before a profiling run and
    :func:`get_accumulated_cost` after to retrieve the aggregated cost.

    Args:
        client: A configured ``AsyncPortkey`` (or compatible) async client.
        system_prompt: The ``system`` role message content.
        user_prompt: The ``user`` role message content.
        model: Model identifier (e.g. ``"gpt-4o"``).
        provider: LLM provider name (e.g. ``"openai"``, ``"anthropic"``).
            Defaults to ``"openai"``.
        temperature: Sampling temperature.
        max_tokens: Maximum number of tokens in the completion.
        response_format: Optional response format dict, e.g.
            ``{"type": "json_object"}``.  Omitted from the API call when
            ``None``.

    Returns:
        The raw string content from the first completion choice, or ``None``
        if the response contained no content.

    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    logger.debug(
        "LLM request dispatched",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        has_response_format=response_format is not None,
    )

    # ── Pre-call token estimation ──────────────────────────────────────────────
    if _token_counting_available:
        _estimate_and_accumulate_tokens(system_prompt, user_prompt, model)

    # ── LLM call (with optional latency measurement) ───────────────────────────
    if _latency_tracking_available and llm_track_latency is not None:
        from core.services.latency_tracking import get_latency_tracker as _get_lat

        _tracker = _get_lat()
        if _tracker is not None:
            try:
                async with llm_track_latency(_tracker, provider=provider, model=model) as _lat_ctx:
                    response = await client.chat.completions.create(**kwargs)
                _result = _lat_ctx.result
                if _result is not None:
                    _get_accumulator()["total_latency_ms"] += _result.total_ms
            except Exception as _lat_exc:
                logger.warning("Latency tracking failed — falling back to untracked call", error=str(_lat_exc))
                response = await client.chat.completions.create(**kwargs)
        else:
            response = await client.chat.completions.create(**kwargs)
    else:
        response = await client.chat.completions.create(**kwargs)

    content: str | None = response.choices[0].message.content  # type: ignore

    logger.debug(
        "LLM response received",
        model=model,
        chars=len(content) if content else 0,
        has_content=content is not None,
    )

    # ── Cost computation ───────────────────────────────────────────────────────
    if _cost_tracking_available:
        _compute_and_accumulate_cost(provider, model, response)

    return content


def _estimate_and_accumulate_tokens(system_prompt: str, user_prompt: str, model: str) -> None:
    """Estimate input tokens before an LLM call and add to the accumulator.

    Uses the token counter library (tiktoken-backed) to count text and overhead
    tokens for the two-message prompt.  Errors are swallowed so estimation
    never breaks the main flow.
    """
    if not _token_counting_available:
        return

    from core.services.token_tracking import get_token_counter

    counter = get_token_counter()
    if counter is None:
        return

    try:
        messages = [
            LLMMessage(role=LLMMessageRole.SYSTEM, content=system_prompt),  # type: ignore[call-arg]
            LLMMessage(role=LLMMessageRole.USER, content=user_prompt),  # type: ignore[call-arg]
        ]
        token_count = counter.count_messages(messages, model=model)
        acc = _get_accumulator()
        acc["estimated_text_tokens"] += token_count.text_tokens
        acc["estimated_overhead_tokens"] += token_count.overhead_tokens
        acc["estimated_total_tokens"] += token_count.total
        acc["estimated_message_count"] += len(messages)
    except Exception as exc:
        logger.warning("Failed to estimate tokens", model=model, error=str(exc))


def _compute_and_accumulate_cost(provider: str, model: str, response: Any) -> None:
    """Extract token usage from the response, compute cost, and add to accumulator.

    Errors are swallowed so cost tracking never breaks the main flow.
    """
    from core.services.cost_tracking import get_cost_tracker

    tracker = get_cost_tracker()
    if tracker is None:
        return

    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            logger.warning(
                "No usage object in response",
                provider=provider,
                model=model,
                response_type=type(response).__name__,
            )
            return

        # Log full usage object for Groq debugging
        if provider.lower() == "groq":
            usage_dict = {}
            if hasattr(usage, "__dict__"):
                usage_dict = usage.__dict__
            elif isinstance(usage, dict):
                usage_dict = dict(usage)

            logger.info(
                "🔍 GROQ Usage object structure",
                provider=provider,
                model=model,
                usage_type=type(usage).__name__,
                usage_dict=usage_dict,
                usage_repr=repr(usage),
            )

        # Standard extraction (works for OpenAI, Azure OpenAI, Groq)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        logger.info(
            "Token extraction attempt",
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        # For Groq and Anthropic: try alternate field names if standard ones are 0
        if prompt_tokens == 0 and completion_tokens == 0:
            # Try Anthropic-style field names
            prompt_tokens = getattr(usage, "input_tokens", 0) or 0
            completion_tokens = getattr(usage, "output_tokens", 0) or 0

            logger.info(
                "Tried alternate field names",
                provider=provider,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            # If still 0, log diagnostic info
            if prompt_tokens == 0 and completion_tokens == 0:
                usage_attrs = {}
                if hasattr(usage, "__dict__"):
                    usage_attrs = list(usage.__dict__.keys())
                elif isinstance(usage, dict):
                    usage_attrs = list(usage.keys())

                logger.error(
                    "⚠️ Token extraction FAILED - all attempts returned 0",
                    provider=provider,
                    model=model,
                    usage_type=type(usage).__name__,
                    available_fields=usage_attrs,
                    usage_value=str(usage)[:500],
                )

        call_input = LLMCallInput(  # type: ignore[call-arg]
            provider=provider,
            model=model,
            tokens=LLMTokenUsage(  # type: ignore[call-arg]
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
            ),
            status=LLMCallStatus.SUCCESS,  # type: ignore[arg-type]
        )

        breakdown = tracker.compute_cost(call_input)
        acc = _get_accumulator()
        acc["input_tokens"] += prompt_tokens
        acc["output_tokens"] += completion_tokens
        acc["input_cost"] += breakdown.input_cost
        acc["output_cost"] += breakdown.output_cost
        acc["total_cost"] += breakdown.total_cost

    except LookupError:
        logger.debug("No pricing entry found for model — skipping cost computation", model=model)
    except Exception as exc:
        logger.warning("Failed to compute LLM cost", model=model, error=str(exc))
