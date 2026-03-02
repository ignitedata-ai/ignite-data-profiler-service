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


def get_llm_client() -> BaseLLMClient | None:
    """Return a configured LLM client based on ``settings.LLM_PROVIDER``.

    Returns:
        A ready-to-use ``BaseLLMClient`` subclass instance, or ``None`` if
        LLM augmentation is disabled or the provider is unrecognised.

    """
    if not settings.LLM_ENABLED:
        return None

    provider = settings.LLM_PROVIDER.lower()

    if provider == "openai":
        from core.llm.openai import OpenAILLMClient

        if not settings.LLM_OPENAI_API_KEY:
            logger.warning("LLM_ENABLED is True but LLM_OPENAI_API_KEY is not set; description augmentation is disabled")
            return None

        logger.info("LLM client initialised", provider="openai", model=settings.LLM_MODEL)
        return OpenAILLMClient()

    logger.warning("Unknown LLM_PROVIDER; description augmentation disabled", provider=provider)
    return None


__all__ = ["get_llm_client", "BaseLLMClient"]
