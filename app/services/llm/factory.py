# app/services/llm/factory.py
"""
LLM Provider Factory - Factory Method Pattern.

Responsible for instantiating the correct LLMProvider strategy based on the
AI_PROVIDER environment variable. The factory is the only place in the
codebase that knows which concrete strategy classes exist.

Design Pattern:
    Factory Method - the factory function acts as the "creator" that decides
    which "product" (LLMProvider subclass) to instantiate. Callers only
    depend on the abstract LLMProvider interface.

Usage:
    from app.services.llm import get_llm_provider

    provider = get_llm_provider() # uses AI_PROVIDER from settings
    provider = get_llm_provider(override="anthropic") # force a specific provider

Extending:
    To add a new provider (e.g. "cohere"):
        1. Create CohereProvider(LLMProvider) in strategies.py
        2. Add "cohere": CohereProvider to _PROVIDER_REGISTRY below
        3. Add COHERE = "cohere" to AIProvider enum in app/core/config.py
        4. Done - no other files need to change.
"""

from __future__ import annotations

from typing import Optional, Type

from app.core.config import AIProvider, settings
from app.core.logging import get_logger
from app.services.llm.base import LLMProvider

logger = get_logger(__name__)

# — Provider Registry ——————————————————————————————————————
# Maps AIProvider enum values -> concrete strategy classes.
# Import strategies lazily inside the function to avoid circular imports
# and to keep optional heavy dependencies (anthropic, etc.) from being
# imported when they are not needed.

def _build_registry() -> dict[str, Type[LLMProvider]]:
    """
    Build the provider name -> strategy class mapping.

    Imported lazily so that missing optional packages (e.g. anthropic)
    do not cause ImportError at module load time.

    Returns:
        Dict mapping provider name strings to LLMProvider subclasses.
    """
    from app.services.llm.strategies import (
        AnthropicProvider,
        AzureOpenAIProvider,
        OllamaProvider,
        OpenAIProvider,
    )

    return {
        AIProvider.OPENAI.value: OpenAIProvider,
        AIProvider.ANTHROPIC.value: AnthropicProvider,
        AIProvider.AZURE_OPENAI.value: AzureOpenAIProvider,
        AIProvider.OLLAMA.value: OllamaProvider,
    }

# — Factory Function ——————————————————————————————————————

def get_llm_provider(override: Optional[str] = None) -> LLMProvider:
    """
    A fully configured LLMProvider instance ready to call.complete().

    Raises:
        AIServiceUnavailableError: If the provider name is unknown, or if
            validate_config() fails (missing credentials).

    Example:
        # Use the provider from.env / environment
        provider = get_llm_provider()

        # Force OpenAI regardless of settings
        provider = get_llm_provider(override="openai")

        # Use in async context
        response = await provider.complete(system_prompt, messages)
    """
    from app.core.exceptions import AIServiceUnavailableError

    registry = _build_registry()

    # Determine which provider to use
    provider_name: str = override or settings.AI_PROVIDER.value

    logger.debug("llm_factory_resolving", provider=provider_name)

    # Look up the strategy class
    strategy_class: Optional[Type[LLMProvider]] = registry.get(provider_name)

    if strategy_class is None:
        available = ", ".join(sorted(registry.keys()))
        raise AIServiceUnavailableError(
            provider=provider_name,
            reason=(
                f"Unknown AI provider '{provider_name}'. "
                f"Available providers: {available}. "
                f"Check AI_PROVIDER in your.env file."
            ),
        )

    # Instantiate the strategy
    provider_instance: LLMProvider = strategy_class()

    # Validate credentials immediately (fail-fast)
    provider_instance.validate_config()

    logger.info(
        "llm_provider_created",
        provider=provider_name,
        strategy_class=strategy_class.__name__,
    )

    return provider_instance