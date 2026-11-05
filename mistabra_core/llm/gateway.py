"""Gateway: renders a prompt, calls the provider on the right tier, parses and validates JSON, caches by
(prompt id, version, provider, model, input) and records every call for provenance.

Adapted from the PoC gateway: the SQLite store is gone (this layer keeps no state between calls);
the cache is in memory, or on disk when `Settings.cache_path` is set, so repeated testbed runs are cheap
and identical inputs return identical outputs. Malformed output is an error, not a retry.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path

from pydantic import BaseModel, ValidationError

from ..types import LLMCallRecord
from .base import LLM, LLMOutputError, LLMTransportError
from .prompts import PROMPTS


class _Cache:
    def __init__(self, path: Path | None):
        self._mem: dict[str, str] = {}
        self._lock = threading.Lock()
        self._db = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, text TEXT NOT NULL)")

    def get(self, key: str) -> str | None:
        with self._lock:
            if key in self._mem:
                return self._mem[key]
            if self._db is not None:
                row = self._db.execute("SELECT text FROM cache WHERE key=?", (key,)).fetchone()
                if row:
                    self._mem[key] = row[0]
                    return row[0]
        return None

    def put(self, key: str, text: str) -> None:
        with self._lock:
            self._mem[key] = text
            if self._db is not None:
                self._db.execute("INSERT OR REPLACE INTO cache(key, text) VALUES(?, ?)", (key, text))
                self._db.commit()


class Recorder:
    """Collects the calls made for one `assess` call, for its provenance."""
    def __init__(self):
        self.calls: list[LLMCallRecord] = []

    def add(self, rec: LLMCallRecord):
        self.calls.append(rec)


class Gateway:
    def __init__(self, settings, provider: LLM):
        self.settings, self.provider = settings, provider
        self.cache = _Cache(settings.cache_path)
        self.stats = {"calls": 0, "cached": 0, "errors": 0}

    def model_for(self, tier: str) -> str:
        return self.settings.model if tier == "reasoning" else self.settings.fast_model

    @staticmethod
    def version_of(task: str) -> str:
        return PROMPTS[task].version

    def call(self, task: str, variables: dict, *, recorder: Recorder | None = None, max_tokens: int | None = None) -> BaseModel:
        p = PROMPTS[task]
        system, user = p.render(variables)
        max_tokens = max_tokens or self.settings.max_tokens
        model = self.model_for(p.tier)
        key = hashlib.sha256(f"{p.id}|{p.version}|{self.provider.name}|{model}|{user}".encode()).hexdigest()
        t0, cached, tokens, parsed, err = time.perf_counter(), False, (None, None), None, None
        text = self.cache.get(key)
        try:
            if text is not None:
                cached = True
                self.stats["cached"] += 1
            else:
                text, tokens = self._complete(model, system, user, max_tokens)
                self.stats["calls"] += 1
            parsed = self._parse(text, p.schema)
            if not cached:
                self.cache.put(key, text)
        except (LLMOutputError, LLMTransportError) as e:
            err = e
            self.stats["errors"] += 1
        if recorder is not None:
            recorder.add(LLMCallRecord(task=task, prompt_version=p.version, model=model, cached=cached, ok=parsed is not None,
                                       latency_ms=int((time.perf_counter() - t0) * 1000), tokens_in=tokens[0], tokens_out=tokens[1]))
        if err is not None:
            raise err
        return parsed

    def _complete(self, model, system, user, max_tokens, attempts=3):
        last = None
        for i in range(attempts):
            try:
                return self.provider.complete(model, system, user, max_tokens)
            except LLMTransportError as e:
                last = e
                time.sleep(min(0.5 * 2 ** i, 4))
        raise last

    @staticmethod
    def _parse(text: str | None, schema: type[BaseModel]) -> BaseModel:
        if not text or not text.strip():
            raise LLMOutputError("empty output")
        s = text.strip()
        m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
        if m:
            s = m.group(1).strip()
        i = s.find("{")
        if i < 0:
            raise LLMOutputError("no JSON object in output")
        s = s[i:s.rfind("}") + 1]
        try:
            data = json.loads(s)
        except json.JSONDecodeError as e:
            raise LLMOutputError(f"invalid JSON: {e}") from None
        try:
            return schema.model_validate(data)
        except ValidationError as e:
            raise LLMOutputError(f"schema mismatch: {e.errors()[:2]}") from None
