"""Model access through PydanticAI.

Every prompt's output schema is a Pydantic model; PydanticAI turns it into an output tool, so the model fills in
tool arguments instead of writing JSON as text, validates the result, and on a validation error sends the error back
to the model for another attempt. This module builds the PydanticAI models for each tier and maps PydanticAI's
errors onto ours. Caching, call records and the failure log stay in the gateway.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic_ai.models import Model  # noqa: E402

OPENROUTER_URL = "https://openrouter.ai/api/v1"
OUTPUT_RETRIES = 2          # validation failures sent back to the model before the call fails


class LLMError(RuntimeError): ...
class LLMOutputError(LLMError):
    """The model did not produce output matching the schema, even after the validation retries."""
class LLMAuthError(LLMError):
    """The provider rejected the credentials (HTTP 401/403). Never retried; nothing else will work either."""
class LLMTransportError(LLMError):
    """Network, rate-limit or server failure. Retried with backoff by the gateway."""


class Backend:
    """The real models for both tiers, built once from Settings."""

    def __init__(self, settings):
        self.settings = settings
        self.name = settings.provider
        self._models: dict[str, Model] = {}

    def model(self, tier: str) -> Model:
        name = self.settings.model if tier == "reasoning" else self.settings.fast_model
        if name not in self._models:
            self._models[name] = self._build(name)
        return self._models[name]

    def model_settings(self) -> dict:
        s = {"temperature": 0.0, "max_tokens": self.settings.max_tokens, "timeout": self.settings.timeout_s}
        r = self.settings.reasoning
        if r and self.settings.provider == "openrouter":
            s["extra_body"] = {"reasoning": {"enabled": False} if r == "none" else {"effort": r}}
        elif r and self.settings.provider == "openai":
            s["openai_reasoning_effort"] = "minimal" if r == "none" else r
        return s

    def _build(self, name: str) -> Model:
        st = self.settings
        if st.provider == "anthropic":
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider
            return AnthropicModel(name, provider=AnthropicProvider(api_key=st.api_key))
        from pydantic_ai.models.openai import OpenAIChatModel
        if st.provider == "openrouter":
            if st.base_url:
                from pydantic_ai.providers.openai import OpenAIProvider
                return OpenAIChatModel(name, provider=OpenAIProvider(base_url=st.base_url, api_key=st.api_key))
            from pydantic_ai.providers.openrouter import OpenRouterProvider
            return OpenAIChatModel(name, provider=OpenRouterProvider(api_key=st.api_key))
        if st.provider == "openai":
            from pydantic_ai.providers.openai import OpenAIProvider
            return OpenAIChatModel(name, provider=OpenAIProvider(base_url=st.base_url, api_key=st.api_key or "not-needed"))
        raise ValueError(f"unknown provider {st.provider!r} (expected fake | anthropic | openai | openrouter)")


def make_backend(settings):
    from .fake import FakeProvider
    if not settings.uses_real_llm:
        return FakeProvider()
    return Backend(settings)
