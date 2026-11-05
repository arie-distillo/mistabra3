"""Uniform LLM interface.

A provider implements one method: complete(model, system, user, max_tokens) -> (text, usage).
Everything else (prompts, JSON parsing, caching, logging, retries) is in the gateway,
so swapping providers never touches product code.
"""
from __future__ import annotations
from typing import Protocol
import httpx


class LLMError(RuntimeError): ...
class LLMOutputError(LLMError):
    """Model returned output that could not be parsed or validated. Not retried."""
class LLMAuthError(LLMError):
    """The provider rejected the credentials (HTTP 401/403). Not retried; nothing else will work either."""
class LLMTransportError(LLMError):
    """Network / provider failure. Retried with backoff."""


class LLM(Protocol):
    name: str
    def complete(self, model: str, system: str, user: str, max_tokens: int) -> tuple[str, tuple[int | None, int | None]]: ...


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, base_url: str | None = None, timeout: float = 120):
        self.client = httpx.Client(base_url=base_url or "https://api.anthropic.com", timeout=timeout,
                                   headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"})

    def complete(self, model, system, user, max_tokens):
        try:
            r = self.client.post("/v1/messages", json={"model": model, "max_tokens": max_tokens, "temperature": 0,
                                                       "system": system, "messages": [{"role": "user", "content": user}]})
        except httpx.HTTPError as e:
            raise LLMTransportError(str(e)) from None
        if r.status_code == 429 or r.status_code >= 500:
            raise LLMTransportError(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code in (401, 403):
            raise LLMAuthError(f"HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code >= 400:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        j = r.json()
        text = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
        u = j.get("usage", {})
        return text, (u.get("input_tokens"), u.get("output_tokens"))


class OpenAICompatibleProvider:
    """OpenAI, OpenRouter, Ollama, vLLM, LM Studio, Groq, ... — anything with /chat/completions."""
    name = "openai"

    def __init__(self, api_key: str | None, base_url: str | None = None, timeout: float = 120):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(base_url=(base_url or "https://api.openai.com/v1").rstrip("/"), timeout=timeout, headers=headers)

    def complete(self, model, system, user, max_tokens):
        try:
            r = self.client.post("/chat/completions", json={"model": model, "max_tokens": max_tokens, "temperature": 0,
                                                            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        except httpx.HTTPError as e:
            raise LLMTransportError(str(e)) from None
        if r.status_code == 429 or r.status_code >= 500:
            raise LLMTransportError(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code in (401, 403):
            raise LLMAuthError(f"HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code >= 400:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        j = r.json()
        u = j.get("usage", {})
        return j["choices"][0]["message"]["content"] or "", (u.get("prompt_tokens"), u.get("completion_tokens"))


OPENROUTER_URL = "https://openrouter.ai/api/v1"


def make_provider(settings) -> LLM:
    from .fake import FakeProvider
    if not settings.uses_real_llm:
        return FakeProvider()
    if settings.provider == "anthropic":
        return AnthropicProvider(settings.api_key, settings.base_url, settings.timeout_s)
    if settings.provider == "openai":
        return OpenAICompatibleProvider(settings.api_key, settings.base_url, settings.timeout_s)
    if settings.provider == "openrouter":
        return OpenAICompatibleProvider(settings.api_key, settings.base_url or OPENROUTER_URL, settings.timeout_s)
    raise ValueError(f"unknown provider {settings.provider!r} (expected fake | anthropic | openai | openrouter)")
