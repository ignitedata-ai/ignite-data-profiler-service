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
    provider: str | None = None,
    model: str | None = None,
    portkey_api_key: str | None = None,
    portkey_virtual_key: str | None = None,
) -> BaseLLMClient | None:
    """Return a configured LLM client based on provider.

    Args:
        provider: LLM provider name (e.g., 'openai', 'groq', 'anthropic').
            Falls back to ``settings.LLM_PROVIDER`` when ``None``.
        model: LLM model name (e.g., 'gpt-4o', 'llama-3.3-70b-versatile').
            Falls back to ``settings.LLM_MODEL`` when ``None``.
        portkey_api_key: Per-request Portkey API key override. Falls back to
            ``settings.PORTKEY_API_KEY`` when ``None``.
        portkey_virtual_key: Per-request Portkey virtual key override. Falls
            back to ``settings.PORTKEY_VIRTUAL_KEY`` when ``None``.

    Returns:
        A ready-to-use ``BaseLLMClient`` subclass instance, or ``None`` if
        LLM augmentation is disabled or the provider is unrecognised.

    """
    if not settings.LLM_ENABLED:
        return None

    # Use passed values or fall back to env vars
    provider = provider or settings.LLM_PROVIDER
    model = model or settings.LLM_MODEL
    portkey_api_key = portkey_api_key or settings.PORTKEY_API_KEY
    portkey_virtual_key = portkey_virtual_key or settings.PORTKEY_VIRTUAL_KEY

    # Supported providers via Portkey
    SUPPORTED_PROVIDERS = ["openai", "groq", "anthropic", "azure", "cohere", "google"]

    if provider.lower() not in SUPPORTED_PROVIDERS:
        logger.warning("Unknown LLM_PROVIDER; description augmentation disabled", provider=provider)
        return None

    from core.llm.openai import OpenAILLMClient

    if not portkey_api_key or not portkey_virtual_key:
        logger.warning(
            "LLM_ENABLED is True but PORTKEY_API_KEY or PORTKEY_VIRTUAL_KEY is not set; description augmentation is disabled"
        )
        return None

    logger.info(
        "LLM client initialised",
        provider=provider,
        model=model,
        key_source="request" if portkey_api_key else "env",
        virtual_key_source="request" if portkey_virtual_key else "env",
        passed_virtual_key_prefix=portkey_virtual_key[:12] + "..." if portkey_virtual_key else None,
    )
    return OpenAILLMClient(
        provider=provider,
        model=model,
        portkey_api_key=portkey_api_key,
        portkey_virtual_key=portkey_virtual_key,
    )


__all__ = ["get_llm_client", "BaseLLMClient"]
