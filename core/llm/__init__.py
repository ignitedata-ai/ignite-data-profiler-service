"""LLM client factory.

Usage::

    from core.llm import get_llm_client

    client = get_llm_client()   # returns the configured provider or None
"""

from __future__ import annotations

from core.config import settings
from core.llm.base import BaseLLMClient
from core.logging import get_logger

logger = get_logger(__name__)


def get_llm_client(
    portkey_api_key: str | None = None,
    portkey_virtual_key: str | None = None,
) -> BaseLLMClient | None:
    """Return a configured LLM client based on ``settings.LLM_PROVIDER``.

    Args:
        portkey_api_key: Per-request Portkey API key override.  Falls back to
            ``settings.PORTKEY_API_KEY`` when ``None`` or empty.
        portkey_virtual_key: Per-request Portkey virtual key override.  Falls
            back to ``settings.PORTKEY_VIRTUAL_KEY`` when ``None`` or empty.

    Returns:
        A ready-to-use ``BaseLLMClient`` subclass instance, or ``None`` if
        LLM augmentation is disabled or the provider is unrecognised.

    """
    if not settings.LLM_ENABLED:
        return None

    provider = settings.LLM_PROVIDER.lower()

    if provider == "openai":
        from core.llm.openai import OpenAILLMClient

        # Resolve keys: request-level overrides take precedence over env vars.
        resolved_api_key = portkey_api_key or settings.PORTKEY_API_KEY
        resolved_virtual_key = portkey_virtual_key or settings.PORTKEY_VIRTUAL_KEY

        if not resolved_api_key or not resolved_virtual_key:
            logger.warning("LLM_ENABLED is True but PORTKEY_API_KEY or PORTKEY_VIRTUAL_KEY is not set; description augmentation is disabled")
            return None

        logger.info(
            "LLM client initialised",
            provider="openai (portkey)",
            model=settings.LLM_MODEL,
            key_source="request" if portkey_api_key else "env",
            virtual_key_source="request" if portkey_virtual_key else "env",
            passed_api_key_prefix=portkey_api_key[:8] + "..." if portkey_api_key else None,
            passed_virtual_key_prefix=portkey_virtual_key[:12] + "..." if portkey_virtual_key else None,
        )
        return OpenAILLMClient(
            portkey_api_key=resolved_api_key,
            portkey_virtual_key=resolved_virtual_key,
        )

    logger.warning("Unknown LLM_PROVIDER; description augmentation disabled", provider=provider)
    return None


__all__ = ["get_llm_client", "BaseLLMClient"]
