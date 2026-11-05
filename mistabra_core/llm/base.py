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
    def complete(self, model: str, system: str, user: str, max_tokens: int, schema: dict | None = None) -> tuple[str, tuple[int | None, int | None]]: ...


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, base_url: str | None = None, timeout: float = 120):
        self.client = httpx.Client(base_url=base_url or "https://api.anthropic.com", timeout=timeout,
                                   headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"})

    def complete(self, model, system, user, max_tokens, schema=None):
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

    def __init__(self, api_key: str | None, base_url: str | None = None, timeout: float = 120, reasoning: str | None = None,
                 structured: bool = True):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(base_url=(base_url or "https://api.openai.com/v1").rstrip("/"), timeout=timeout, headers=headers)
        self.openrouter = "openrouter.ai" in str(self.client.base_url)
        self.reasoning = reasoning           # None (model default) | none | low | medium | high
        self.structured = structured         # send each prompt's JSON schema as response_format

    def complete(self, model, system, user, max_tokens, schema=None):
        body = {"model": model, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if schema is not None and self.structured:
            # constrained decoding where the endpoint supports it; elsewhere a strong hint
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": schema.get("title", "output"), "schema": schema, "strict": False}}
        if self.reasoning:
            if self.openrouter:   # OpenRouter's unified parameter; hybrid models (Qwen 3.5, Gemini, ...) honour it
                body["reasoning"] = {"enabled": False} if self.reasoning == "none" else {"effort": self.reasoning}
            else:                 # OpenAI-style
                body["reasoning_effort"] = self.reasoning
        try:
            r = self.client.post("/chat/completions", json=body)
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
        choice = (j.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        if not text.strip():
            why = choice.get("finish_reason")
            spent = bool(msg.get("reasoning") or msg.get("reasoning_content"))
            hint = (" — the model spent its token budget thinking; set MISTABRA_REASONING=none or low, or raise MISTABRA_MAX_TOKENS"
                    if spent or why == "length" else "")
            raise LLMOutputError(f"empty output (finish_reason={why}, reasoning={'yes' if spent else 'no'}, "
                                 f"completion_tokens={u.get('completion_tokens')}){hint}")
        return text, (u.get("prompt_tokens"), u.get("completion_tokens"))


OPENROUTER_URL = "https://openrouter.ai/api/v1"


def make_provider(settings) -> LLM:
    from .fake import FakeProvider
    if not settings.uses_real_llm:
        return FakeProvider()
    if settings.provider == "anthropic":
        return AnthropicProvider(settings.api_key, settings.base_url, settings.timeout_s)
    if settings.provider == "openai":
        return OpenAICompatibleProvider(settings.api_key, settings.base_url, settings.timeout_s, settings.reasoning, settings.structured)
    if settings.provider == "openrouter":
        return OpenAICompatibleProvider(settings.api_key, settings.base_url or OPENROUTER_URL, settings.timeout_s, settings.reasoning, settings.structured)
    raise ValueError(f"unknown provider {settings.provider!r} (expected fake | anthropic | openai | openrouter)")
