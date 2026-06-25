# app/services/llm/__init__.py
"""
LLM Provider Package.

Exports the factory function and all strategy classes for external use.

Usage:
    from app.services.llm import get_llm_provider
    provider = get_llm_provider()
    text, tokens = await provider.complete(system_prompt, messages)
"""

from app.services.llm.base import LLMProvider
from app.services.llm.factory import get_llm_provider
from app.services.llm.strategies import (
    AnthropicProvider,
    AzureOpenAIProvider,
    OllamaProvider,
    OpenAIProvider,
)

__all__ = [
    "LLMProvider",
    "get_llm_provider",
    "OpenAIProvider",
    "AnthropicProvider",
    "AzureOpenAIProvider",
    "OllamaProvider",
]