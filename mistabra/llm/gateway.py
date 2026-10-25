"""Gateway: renders a prompt, calls the provider on the right tier, parses + validates JSON,
caches by (prompt id, version, model, input), logs every call. Malformed output is not retried."""
from __future__ import annotations
import hashlib, json, re, time
from pydantic import BaseModel, ValidationError
from ..store import Store, new_id, now, dumps
from .base import LLM, LLMOutputError, LLMTransportError
from .prompts import PROMPTS


class Gateway:
    def __init__(self, store: Store, settings, provider: LLM):
        self.store, self.settings, self.provider = store, settings, provider
        self.stats = {"calls": 0, "cached": 0, "errors": 0}

    @staticmethod
    def version_of(task: str) -> str:
        return PROMPTS[task].version

    def model_for(self, tier: str) -> str:
        return self.settings.model if tier == "reasoning" else self.settings.fast_model

    def call(self, task: str, variables: dict, *, subject: tuple[str, str] | None = None, max_tokens: int = 2000,
             model: str | None = None, prompt_version: str | None = None) -> tuple[BaseModel, str]:
        p = PROMPTS[task]
        if prompt_version and prompt_version != p.version:
            raise ValueError(f"prompt {task} version {prompt_version} is not available (have {p.version})")
        system, user = p.render(variables)
        model = model or self.model_for(p.tier)
        ih = hashlib.sha256(user.encode()).hexdigest()
        key = hashlib.sha256(f"{p.id}|{p.version}|{self.provider.name}|{model}|{ih}".encode()).hexdigest()
        cached = self.store.one("SELECT output_text FROM llm_cache WHERE key=?", (key,))
        t0, call_id, text, err, tokens, parsed = time.perf_counter(), new_id(), None, None, (None, None), None
        try:
            if cached:
                text = cached["output_text"]; self.stats["cached"] += 1
            else:
                text, tokens = self._complete(model, system, user, max_tokens); self.stats["calls"] += 1
            parsed = self._parse(text, p.schema)
        except (LLMOutputError, LLMTransportError) as e:
            err = f"{type(e).__name__}: {e}"; self.stats["errors"] += 1
        with self.store.tx("system") as tx:
            tx.execute("INSERT INTO llm_call(id,ts,task,prompt_version,provider,model,input_hash,input_json,output_text,ok,cached,latency_ms,"
                       "tokens_in,tokens_out,error,subject_type,subject_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (call_id, now(), task, p.version, self.provider.name, model, ih, dumps(variables), text, int(parsed is not None),
                        int(bool(cached)), int((time.perf_counter() - t0) * 1000), tokens[0], tokens[1], err,
                        subject[0] if subject else None, subject[1] if subject else None))
            if parsed is not None and not cached:
                tx.execute("INSERT OR REPLACE INTO llm_cache(key,output_text,ts) VALUES(?,?,?)", (key, text, now()))
        if parsed is None:
            raise (LLMOutputError if err.startswith("LLMOutputError") else LLMTransportError)(err)
        return parsed, call_id

    def _complete(self, model, system, user, max_tokens, attempts=3):
        last = None
        for i in range(attempts):
            try:
                return self.provider.complete(model, system, user, max_tokens)
            except LLMTransportError as e:
                last = e; time.sleep(min(0.5 * 2 ** i, 4))
        raise last

    @staticmethod
    def _parse(text: str | None, schema: type[BaseModel]) -> BaseModel:
        if not text or not text.strip():
            raise LLMOutputError("empty output")
        s = text.strip()
        m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
        if m: s = m.group(1).strip()
        i = s.find("{")
        if i < 0: raise LLMOutputError("no JSON object in output")
        s = s[i:s.rfind("}") + 1]
        try:
            data = json.loads(s)
        except json.JSONDecodeError as e:
            raise LLMOutputError(f"invalid JSON: {e}") from None
        try:
            return schema.model_validate(data)
        except ValidationError as e:
            raise LLMOutputError(f"schema mismatch: {e.errors()[:2]}") from None
