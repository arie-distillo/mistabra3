"""LLM gateway: two model tiers, versioned prompts, structured outputs, cache, call log.

Malformed output raises LLMOutputError and is NOT retried (retrying a deterministic
failure only wastes time). Only TransportError is retried.
"""
from __future__ import annotations
import hashlib, json, re, time
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError
from ..config import Settings
from ..store import Store, new_id, now
from .prompts import PROMPTS, Prompt

T = TypeVar("T", bound=BaseModel)


class LLMOutputError(RuntimeError):
    """Model returned output that could not be parsed/validated. Non-transient."""


class TransportError(RuntimeError):
    """Network/provider failure. Transient — may be retried."""


class Gateway:
    def __init__(self, store: Store, settings: Settings, provider):
        self.store, self.settings, self.provider = store, settings, provider
        self.stats = {"calls": 0, "cached": 0, "errors": 0}

    def model_for(self, tier: str) -> str:
        return self.settings.reasoning_model if tier == "reasoning" else self.settings.operational_model

    def call(self, task: str, tier: str, variables: dict, schema: Type[T], max_tokens: int = 4000) -> tuple[T, str]:
        prompt: Prompt = PROMPTS[task]
        system, user = prompt.render(variables)
        model = self.model_for(tier)
        input_hash = hashlib.sha256(user.encode()).hexdigest()
        cache_key = hashlib.sha256(f"{prompt.id}|{prompt.version}|{model}|{self.provider.name}|{input_hash}".encode()).hexdigest()
        call_id, t0 = new_id(), time.perf_counter()
        cached = self.store.one("SELECT output_text FROM llm_cache WHERE cache_key=?", (cache_key,))
        output, error, tokens = None, None, (None, None)
        try:
            if cached:
                output = cached["output_text"]
                self.stats["cached"] += 1
            else:
                output, tokens = self._complete_with_retry(model, system, user, max_tokens)
                self.stats["calls"] += 1
            parsed = self._parse(output, schema)
            parsed_ok = 1
        except (LLMOutputError, TransportError) as e:
            error, parsed_ok, parsed = f"{type(e).__name__}: {e}", 0, None
            self.stats["errors"] += 1
        latency = int((time.perf_counter() - t0) * 1000)
        with self.store.tx("system") as tx:
            tx.execute(
                "INSERT INTO llm_call(id,ts,task,tier,prompt_id,prompt_version,model,provider,input_hash,input_json,output_text,"
                "parsed_ok,cached,latency_ms,tokens_in,tokens_out,error) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (call_id, now(), task, tier, prompt.id, prompt.version, model, self.provider.name, input_hash,
                 json.dumps(variables, ensure_ascii=False, default=str), output, parsed_ok, 1 if cached else 0, latency,
                 tokens[0], tokens[1], error))
            if parsed_ok and not cached:
                tx.execute("INSERT OR REPLACE INTO llm_cache(cache_key,output_text,ts) VALUES(?,?,?)", (cache_key, output, now()))
        if parsed is None:
            raise LLMOutputError(error) if error and error.startswith("LLMOutputError") else TransportError(error)
        return parsed, call_id

    def _complete_with_retry(self, model, system, user, max_tokens, attempts: int = 3):
        last = None
        for i in range(attempts):
            try:
                return self.provider.complete(model, system, user, max_tokens)
            except TransportError as e:
                last = e
                time.sleep(min(2 ** i * 0.5, 4))
        raise last

    @staticmethod
    def _parse(text: str | None, schema: Type[T]) -> T:
        if not text or not text.strip():
            raise LLMOutputError("empty output")
        s = text.strip()
        m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
        if m:
            s = m.group(1).strip()
        if not s.startswith("{"):
            i = s.find("{")
            if i < 0:
                raise LLMOutputError("no JSON object in output")
            s = s[i:]
        try:
            data = json.loads(s)
        except json.JSONDecodeError as e:
            raise LLMOutputError(f"invalid JSON: {e}") from None
        try:
            return schema.model_validate(data)
        except ValidationError as e:
            raise LLMOutputError(f"schema mismatch: {e.errors()[:3]}") from None
