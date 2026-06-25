# app/services/llm/base.py

"""
Abstract LLM Provider - Strategy Interface.

Defines the contract that every concrete LLM provider strategy must fulfill.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# =============================================================================
# Message Types
# =============================================================================

@dataclass(frozen=True)
class LLMMessage:
    """
    A single conversation message.
    """

    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        """
        Convert to provider-compatible dictionary.
        """
        return {
            "role": self.role,
            "content": self.content,
        }


@dataclass
class LLMResponse:
    """
    Normalized response returned by every provider.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    raw: Optional[Any] = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        """
        Total tokens consumed.
        """
        return self.input_tokens + self.output_tokens


# =============================================================================
# Abstract Provider Strategy
# =============================================================================

class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Every provider implementation must:

    - implement complete()
    - implement validate_config()

    Implementations should be safe for concurrent async use.
    """

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        messages: List[LLMMessage],
    ) -> LLMResponse:
        """
        Send messages to the model and return a completion.

        Args:
            system_prompt: System-level instructions.
            messages: Conversation messages.

        Returns:
            LLMResponse
        """
        raise NotImplementedError

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate provider configuration.

        Raises:
            Exception if required credentials/settings are missing.
        """
        raise NotImplementedError

    @property
    def provider_name(self) -> str:
        """
        Human-readable provider name.
        """
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.provider_name}()"