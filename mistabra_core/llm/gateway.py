"""Gateway: renders a prompt, runs it through PydanticAI on the right tier with the prompt's schema as the output
type, caches the validated output by (prompt id, version, provider, model, input), records every call for
provenance (including how many attempts the model needed), and logs failed calls with their full message history.

The layer keeps no state between `assess` calls besides this cache: in memory, or on disk when
`Settings.cache_path` is set, so repeated testbed runs are cheap and identical inputs return identical outputs.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessagesTypeAdapter

from ..types import LLMCallRecord
from .base import OUTPUT_RETRIES, LLMAuthError, LLMError, LLMOutputError, LLMTransportError
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
    def __init__(self, settings, backend):
        self.settings, self.backend = settings, backend
        self.cache = _Cache(settings.cache_path)
        self.stats = {"calls": 0, "cached": 0, "errors": 0, "retries": 0}
        self._agents: dict[tuple, Agent] = {}

    def model_for(self, tier: str) -> str:
        return self.settings.model if tier == "reasoning" else self.settings.fast_model

    @staticmethod
    def version_of(task: str) -> str:
        return PROMPTS[task].version

    def _agent(self, task: str, tier: str) -> Agent:
        k = (task, tier)
        if k not in self._agents:
            p = PROMPTS[task]
            self._agents[k] = Agent(self.backend.model(tier), output_type=p.schema, system_prompt=p.system,
                                    retries=OUTPUT_RETRIES, name=task.replace(".", "_"))
        return self._agents[k]

    def call(self, task: str, variables: dict, *, recorder: Recorder | None = None) -> BaseModel:
        p = PROMPTS[task]
        _, user = p.render(variables)
        model = self.model_for(p.tier)
        key = hashlib.sha256(f"{p.id}|{p.version}|{self.backend.name}|{model}|{user}".encode()).hexdigest()
        t0 = time.perf_counter()
        hit = self.cache.get(key)
        if hit is not None:
            self.stats["cached"] += 1
            if recorder is not None:
                recorder.add(LLMCallRecord(task=task, prompt_version=p.version, model=model, cached=True, ok=True, latency_ms=0))
            return p.schema.model_validate_json(hit)
        out, usage, err = None, None, None
        for attempt in range(3):
            with capture_run_messages() as messages:
                try:
                    res = self._agent(task, p.tier).run_sync(user, model_settings=self.backend.model_settings())
                    out, usage = res.output, res.usage
                    break
                except ModelHTTPError as e:
                    if e.status_code in (401, 403):
                        err = LLMAuthError(f"{task}: HTTP {e.status_code}: {str(e.body)[:300]}")
                    elif e.status_code == 429 or e.status_code >= 500:
                        err = LLMTransportError(f"{task}: HTTP {e.status_code}: {str(e.body)[:200]}")
                    else:
                        err = LLMError(f"{task}: HTTP {e.status_code}: {str(e.body)[:300]}")
                except UnexpectedModelBehavior as e:
                    err = LLMOutputError(f"{task}: {e}")
                except ModelAPIError as e:
                    err = LLMTransportError(f"{task}: {type(e).__name__}: {e}")
                if not isinstance(err, LLMTransportError) or attempt == 2:
                    self._log_failure(task, model, user, messages, err)
                    break
                time.sleep(min(0.5 * 2 ** attempt, 4))
        latency = int((time.perf_counter() - t0) * 1000)
        if out is None:
            self.stats["errors"] += 1
            if recorder is not None:
                recorder.add(LLMCallRecord(task=task, prompt_version=p.version, model=model, cached=False, ok=False, latency_ms=latency))
            raise err
        self.stats["calls"] += 1
        retries = max(0, (usage.requests or 1) - 1)
        self.stats["retries"] += retries
        self.cache.put(key, out.model_dump_json())
        if recorder is not None:
            recorder.add(LLMCallRecord(task=task, prompt_version=p.version, model=model, cached=False, ok=True, latency_ms=latency,
                                       tokens_in=usage.input_tokens, tokens_out=usage.output_tokens, retries=retries))
        return out

    def _log_failure(self, task, model, user, messages, err):
        path = self.settings.failure_log
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            history = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
        except Exception:  # noqa: BLE001 — the log must never hide the original error
            history = [repr(m) for m in messages]
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "task": task, "model": model, "error": str(err),
                                "prompt": user, "messages": history}, ensure_ascii=False) + "\n")
