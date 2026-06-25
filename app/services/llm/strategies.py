# app/services/llm/strategies.py
"""
Concrete LLM Provider Strategies.

Each class implements the LLMProvider interface for a specific AI provider.
Provider-specific SDK calls, auth, and response parsing are fully encapsulated
here - the rest of the codebase only ever sees LLMProvider + LLMResponse.

Providers implemented:
    - OpenAIProvider - OpenAI Chat Completions API (gpt-4o, gpt-4o-mini, etc.)
    - AnthropicProvider - Anthropic Messages API (claude-3.5-sonnet, etc.)
    - AzureOpenAIProvider - Azure-hosted OpenAI (same SDK, different base URL)
    - OllamaProvider - Local Ollama server (OpenAI-compatible endpoint)

Adding a new provider:
    1. Add a class here that inherits from LLMProvider.
    2. Implement complete() and validate_config().
    3. Register it in factory.py - one dict entry.
"""

from __future__ import annotations

import json
from typing import List

from app.core.config import settings
from app.core.logging import get_logger
from app.services.llm.base import LLMMessage, LLMProvider, LLMResponse

logger = get_logger(__name__)

# — OpenAI Strategy ——————————————————————————————————————

class OpenAIProvider(LLMProvider):
    """
    Strategy for the OpenAI Chat Completions API.

    Supports all models accessible via the standard OpenAI API:
        gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, etc.

    Configuration (via.env):
        OPENAI_API_KEY - required
        OPENAI_ORG_ID - optional
        OPENAI_BASE_URL - default: https://api.openai.com/v1
        AI_MODEL - default: gpt-4o-mini
        AI_TEMPERATURE - default: 0.0
        AI_MAX_TOKENS - default: 1024
        AI_TIMEOUT_SECONDS - default: 30
    """

    def validate_config(self) -> None:
        """
        Ensure OPENAI_API_KEY is set before attempting any API call.

        Raises:
            AIServiceUnavailableError: If OPENAI_API_KEY is missing.
        """
        from app.core.exceptions import AIServiceUnavailableError

        if not settings.OPENAI_API_KEY:
            raise AIServiceUnavailableError(
                provider="openai",
                reason="OPENAI_API_KEY is not configured. "
                       "Set it in your.env file or environment.",
            )

    async def complete(
        self,
        system_prompt: str,
        messages: List[LLMMessage],
    ) -> LLMResponse:
        """
        Call the OpenAI Chat Completions API.

        Returns:
            LLMResponse with generated text and token usage.

        Raises:
            AIServiceUnavailableError: On auth failure or network error.
            AIServiceError: On unexpected API error.
        """
        from openai import AsyncOpenAI, APIConnectionError, AuthenticationError, RateLimitError
        from app.core.exceptions import AIServiceError, AIServiceUnavailableError

        self.validate_config()

        client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            organization=settings.OPENAI_ORG_ID or None,
            base_url=settings.OPENAI_BASE_URL,
            timeout=settings.AI_TIMEOUT_SECONDS,
            max_retries=0,
        )

        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(m.to_dict() for m in messages)

        logger.debug(
            "openai_request",
            model=settings.AI_MODEL,
            message_count=len(api_messages),
            temperature=settings.AI_TEMPERATURE,
        )

        try:
            response = await client.chat.completions.create(
                model=settings.AI_MODEL,
                messages=api_messages,  # type: ignore[arg-type]
                temperature=settings.AI_TEMPERATURE,
                max_tokens=settings.AI_MAX_TOKENS,
            )
        except AuthenticationError as exc:
            raise AIServiceUnavailableError(
                provider="openai",
                reason=f"Authentication failed: {exc}",
            ) from exc
        except RateLimitError as exc:
            raise AIServiceUnavailableError(
                provider="openai",
                reason=f"Rate limit exceeded: {exc}",
            ) from exc
        except APIConnectionError as exc:
            raise AIServiceUnavailableError(
                provider="openai",
                reason=f"Connection error: {exc}",
            ) from exc
        except Exception as exc:
            raise AIServiceError(
                provider="openai",
                reason=f"OpenAI API error: {exc}",
            ) from exc

        choice = response.choices[0]
        usage = response.usage

        logger.debug(
            "openai_response",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=response.model,
            raw=response,
        )

    @property
    def provider_name(self) -> str:
        return "openai"

# — Anthropic Strategy ——————————————————————————————————————

class AnthropicProvider(LLMProvider):
    """
    Supports Claude models:
        claude-3-5-sonnet-20241022, claude-3-haiku-20240307, etc.

    Note: Anthropic separates the system prompt from the messages array,
    which maps cleanly to our LLMProvider interface.

    Configuration (via.env):
        ANTHROPIC_API_KEY - required
        AI_MODEL - default: claude-3-5-sonnet-20241022
        AI_TEMPERATURE - default: 0.0
        AI_MAX_TOKENS - default: 1024
        AI_TIMEOUT_SECONDS - default: 30
    """

    def validate_config(self) -> None:
        """
        Ensure ANTHROPIC_API_KEY is set.

        Raises:
            AIServiceUnavailableError: If ANTHROPIC_API_KEY is missing.
        """
        from app.core.exceptions import AIServiceUnavailableError

        if not settings.ANTHROPIC_API_KEY:
            raise AIServiceUnavailableError(
                provider="anthropic",
                reason="ANTHROPIC_API_KEY is not configured. "
                       "Set it in your.env file or environment.",
            )

    async def complete(
        self,
        system_prompt: str,
        messages: List[LLMMessage],
    ) -> LLMResponse:
        """
        Call the Anthropic Messages API.

        Anthropic's API accepts system as a top-level parameter (not inside
        the messages array), so we pass system_prompt directly.

        Args:
            system_prompt: System-level instruction.
            messages: Ordered user/assistant conversation turns.

        Returns:
            LLMResponse with generated text and token usage.

        Raises:
            AIServiceUnavailableError: On auth failure or network error.
            AIServiceError: On unexpected API error.
        """
        import anthropic
        from app.core.exceptions import AIServiceError, AIServiceUnavailableError

        self.validate_config()

        client = anthropic.AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            timeout=settings.AI_TIMEOUT_SECONDS,
            max_retries=0,
        )

        # Anthropic requires alternating user/assistant roles; filter system msgs
        api_messages = [
            m.to_dict() for m in messages
            if m.role in ("user", "assistant")
        ]

        logger.debug(
            "anthropic_request",
            model=settings.AI_MODEL,
            message_count=len(api_messages),
        )

        try:
            response = await client.messages.create(
                model=settings.AI_MODEL,
                system=system_prompt,
                messages=api_messages, # type: ignore[arg-type]
                temperature=settings.AI_TEMPERATURE,
                max_tokens=settings.AI_MAX_TOKENS,
            )
        except anthropic.AuthenticationError as exc:
            raise AIServiceUnavailableError(
                provider="anthropic",
                reason=f"Authentication failed: {exc}",
            ) from exc
        except anthropic.RateLimitError as exc:
            raise AIServiceUnavailableError(
                provider="anthropic",
                reason=f"Rate limit exceeded: {exc}",
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise AIServiceUnavailableError(
                provider="anthropic",
                reason=f"Connection error: {exc}",
            ) from exc
        except Exception as exc:
            raise AIServiceError(
                provider="anthropic",
                reason=f"Anthropic API error: {exc}",
            ) from exc

        text = response.content[0].text if response.content else ""
        usage = response.usage

        logger.debug(
            "anthropic_response",
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

        return LLMResponse(
            text=text,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            model=response.model,
            raw=response,
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

# — Azure OpenAI Strategy ——————————————————————————————————————

class AzureOpenAIProvider(LLMProvider):
    """
    Strategy for Azure-hosted OpenAI deployments.

    Uses the same openai SDK as OpenAIProvider but with Azure-specific
    authentication (api_key + azure_endpoint + api_version).

    Configuration (via.env):
        AZURE_OPENAI_API_KEY - required
        AZURE_OPENAI_ENDPOINT - required (e.g. https://my.openai.azure.com)
        AZURE_OPENAI_API_VERSION - default: 2024-02-01
        AZURE_OPENAI_DEPLOYMENT_NAME - required (your deployment name)
        AI_TEMPERATURE - default: 0.0
        AI_MAX_TOKENS - default: 1024
        AI_TIMEOUT_SECONDS - default: 30
    """

    def validate_config(self) -> None:
        """
        Ensure all Azure OpenAI credentials are set.

        Raises:
            AIServiceUnavailableError: If any required Azure credential is missing.
        """
        from app.core.exceptions import AIServiceUnavailableError

        missing = []
        if not settings.AZURE_OPENAI_API_KEY:
            missing.append("AZURE_OPENAI_API_KEY")
        if not settings.AZURE_OPENAI_ENDPOINT:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not settings.AZURE_OPENAI_DEPLOYMENT_NAME:
            missing.append("AZURE_OPENAI_DEPLOYMENT_NAME")

        if missing:
            raise AIServiceUnavailableError(
                provider="azure_openai",
                reason=f"Missing required configuration: {', '.join(missing)}",
            )

    async def complete(
        self,
        system_prompt: str,
        messages: List[LLMMessage],
    ) -> LLMResponse:
        """
        Call the Azure OpenAI Chat Completions API.

        Uses the deployment name (not model name) as the model parameter,
        as required by Azure OpenAI.

        Args:
            system_prompt: System-level instruction.
            messages: Ordered user/assistant conversation turns.

        Returns:
            LLMResponse with generated text and token usage.

        Raises:
            AIServiceUnavailableError: On auth failure or network error.
            AIServiceError: On unexpected API error.
        """
        from openai import AsyncAzureOpenAI, APIConnectionError, AuthenticationError, RateLimitError
        from app.core.exceptions import AIServiceError, AIServiceUnavailableError

        self.validate_config()

        client = AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT, # type: ignore[arg-type]
            api_version=settings.AZURE_OPENAI_API_VERSION,
            timeout=settings.AI_TIMEOUT_SECONDS,
            max_retries=0,
        )

        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(m.to_dict() for m in messages)

        # Azure uses deployment name, not model name
        deployment = settings.AZURE_OPENAI_DEPLOYMENT_NAME or settings.AI_MODEL

        logger.debug(
            "azure_openai_request",
            deployment=deployment,
            message_count=len(api_messages),
        )

        try:
            response = await client.chat.completions.create(
                model=deployment,
                messages=api_messages, # type: ignore[arg-type]
                temperature=settings.AI_TEMPERATURE,
                max_tokens=settings.AI_MAX_TOKENS,
            )
        except AuthenticationError as exc:
            raise AIServiceUnavailableError(
                provider="azure_openai",
                reason=f"Authentication failed: {exc}",
            ) from exc
        except RateLimitError as exc:
            raise AIServiceUnavailableError(
                provider="azure_openai",
                reason=f"Rate limit exceeded: {exc}",
            ) from exc
        except APIConnectionError as exc:
            raise AIServiceUnavailableError(
                provider="azure_openai",
                reason=f"Connection error: {exc}",
            ) from exc
        except Exception as exc:
            raise AIServiceError(
                provider="azure_openai",
                reason=f"Azure OpenAI API error: {exc}",
            ) from exc

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
                text=choice.message.content or "",
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                model=response.model,
                raw=response,
            )

    @property
    def provider_name(self) -> str:
        return "azure_openai"

# — Ollama Strategy ——————————————————————————————————————

class OllamaProvider(LLMProvider):
    """
    Strategy for a local Ollama server (OpenAI-compatible endpoint).

    Ollama exposes an OpenAI-compatible /v1/chat/completions endpoint,
    so we reuse the openai SDK pointed at the local server.

    Configuration (via.env):
        OLLAMA_BASE_URL - default: http://localhost:11434
        OLLAMA_MODEL - default: llama3
        AI_TEMPERATURE - default: 0.0
        AI_MAX_TOKENS - default: 1024
        AI_TIMEOUT_SECONDS - default: 30

    Note: No API key is required for local Ollama. A placeholder key is
    passed to satisfy the openai SDK's validation.
    """

    def validate_config(self) -> None:
        """
        Validate Ollama configuration.

        Raises:
            AIServiceUnavailableError: If OLLAMA_BASE_URL is not set.
        """
        from app.core.exceptions import AIServiceUnavailableError

        if not settings.OLLAMA_BASE_URL:
            raise AIServiceUnavailableError(
                provider="ollama",
                reason="OLLAMA_BASE_URL is not configured.",
            )

    async def complete(
        self,
        system_prompt: str,
        messages: List[LLMMessage],
    ) -> LLMResponse:
        """
        Call the local Ollama server via its OpenAI-compatible endpoint.

        Args:
            system_prompt: System-level instruction.
            messages: Ordered user/assistant conversation turns.

        Returns:
            LLMResponse with generated text and token usage.

        Raises:
            AIServiceUnavailableError: If Ollama server is not running.
            AIServiceError: On unexpected error.
        """
        from openai import AsyncOpenAI, APIConnectionError
        from app.core.exceptions import AIServiceError, AIServiceUnavailableError

        self.validate_config()

        # Ollama's OpenAI-compatible endpoint lives at /v1
        base_url = settings.OLLAMA_BASE_URL.rstrip("/") + "/v1"

        client = AsyncOpenAI(
            api_key="ollama", # Placeholder — Ollama ignores the key
            base_url=base_url,
            timeout=settings.AI_TIMEOUT_SECONDS,
            max_retries=0,
        )

        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(m.to_dict() for m in messages)
        model = settings.OLLAMA_MODEL

        logger.debug(
            "ollama_request",
            model=model,
            base_url=base_url,
            message_count=len(api_messages),
        )

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=api_messages, # type: ignore[arg-type]
                temperature=settings.AI_TEMPERATURE,
                max_tokens=settings.AI_MAX_TOKENS,
            )
        except APIConnectionError as exc:
            raise AIServiceUnavailableError(
                provider="ollama",
                reason=(
                    f"Cannot connect to Ollama at {base_url}. "
                    "Ensure Ollama is running: 'ollama serve'"
                ),
            ) from exc
        except Exception as exc:
            raise AIServiceError(
                provider="ollama",
                reason=f"Ollama error: {exc}",
            ) from exc

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=model,
            raw=response,
        )

    @property
    def provider_name(self) -> str:
        return "ollama"