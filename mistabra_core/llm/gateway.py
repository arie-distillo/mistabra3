"""Gateway: renders a prompt, runs it through PydanticAI on the right tier with the prompt's schema as the output
type, caches the validated output by (prompt id, version, provider, model, input), records every call for
provenance (including how many attempts the model needed), and logs failed calls with their full message history.

The layer keeps no state between `assess` calls besides this cache: in memory, or on disk when
`Settings.cache_path` is set, so repeated testbed runs are cheap and identical inputs return identical outputs.

`call` is synchronous and thread-safe. Every model request runs on one background event loop owned by the gateway,
at most `Settings.concurrency` at a time, so callers can fan out independent calls from threads (the core does:
radar reads, risks, groups) without sharing HTTP clients across event loops.
"""
from __future__ import annotations

import asyncio
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
from .base import LLMBudgetError, OUTPUT_RETRIES, LLMAuthError, LLMError, LLMOutputError, LLMTransportError
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


def _retry_reasons(messages) -> list[str]:
    """Why the model's earlier answers in this run were sent back (validation errors), for the call log."""
    from pydantic_ai.messages import ModelRequest, RetryPromptPart, ToolReturnPart
    out = []
    for m in messages or []:
        if isinstance(m, ModelRequest):
            for part in m.parts:
                c = getattr(part, "content", None)
                # a rejected output-tool call comes back either as a retry prompt or as a tool return holding the errors
                if isinstance(part, RetryPromptPart) or (isinstance(part, ToolReturnPart) and isinstance(c, list)):
                    out.append((json.dumps(c, default=str) if not isinstance(c, str) else c)[:400])
    return out


class Recorder:
    """Collects the calls made for one `assess` call, for its provenance. `label` (the episode id) tags the call log."""
    def __init__(self, label: str | None = None):
        self.calls: list[LLMCallRecord] = []
        self.label = label

    def add(self, rec: LLMCallRecord):
        self.calls.append(rec)


class Gateway:
    def __init__(self, settings, backend):
        self.settings, self.backend = settings, backend
        self.cache = _Cache(settings.cache_path)
        self.stats = {"calls": 0, "cached": 0, "errors": 0, "retries": 0}
        self._agents: dict[tuple, Agent] = {}
        self._inflight: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sem: asyncio.Semaphore | None = None

    def _count(self, key: str, n: int = 1):
        with self._lock:
            self.stats[key] += n

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, name="mistabra-llm", daemon=True).start()
                self._sem = asyncio.run_coroutine_threadsafe(self._make_sem(), loop).result()
                self._loop = loop
            return self._loop

    async def _make_sem(self) -> asyncio.Semaphore:
        return asyncio.Semaphore(max(1, int(getattr(self.settings, "concurrency", 8) or 1)))

    async def _arun(self, agent: Agent, user: str):
        async with self._sem:
            with capture_run_messages() as messages:
                try:
                    return await agent.run(user, model_settings=self.backend.model_settings()), messages, None
                except Exception as e:  # noqa: BLE001 — mapped by the caller
                    return None, messages, e

    def model_for(self, tier: str) -> str:
        return self.settings.model if tier == "reasoning" else self.settings.fast_model

    @staticmethod
    def version_of(task: str) -> str:
        return PROMPTS[task].version

    def _agent(self, p, tier: str) -> Agent:
        k = (p.id, p.version, tier)
        with self._lock:
            if k not in self._agents:
                self._agents[k] = Agent(self.backend.model(tier), output_type=p.schema, system_prompt=p.system,
                                        retries=OUTPUT_RETRIES, name=p.id.replace(".", "_"))
            return self._agents[k]

    def call(self, task: str, variables: dict, *, recorder: Recorder | None = None, prompt=None) -> BaseModel:
        """Run one prompt. `prompt` lets a caller outside the layer (the testbed's grader) bring its own versioned prompt."""
        p = prompt or PROMPTS[task]
        _, user = p.render(variables)
        model = self.model_for(p.tier)
        salt = getattr(self.settings, "cache_salt", "") or ""
        temp = getattr(self.settings, "temperature", 0.0) or 0.0
        key = hashlib.sha256(f"{p.id}|{p.version}|{self.backend.name}|{model}|{user}"
                             f"{'|' + salt if salt else ''}{f'|t{temp}' if temp else ''}".encode()).hexdigest()
        t0 = time.perf_counter()
        while True:
            hit = self.cache.get(key)
            if hit is not None:
                self._count("cached")
                if recorder is not None:
                    recorder.add(LLMCallRecord(task=task, prompt_version=p.version, model=model, cached=True, ok=True, latency_ms=0))
                self._log_call(task, model, user, hit, True, 0, recorder, 0)
                return p.schema.model_validate_json(hit)
            # coalesce: the same request already in flight in another thread is awaited, not sent again
            with self._lock:
                waiting = self._inflight.get(key)
                if waiting is None:
                    mine = self._inflight[key] = threading.Event()
            if waiting is None:
                break
            waiting.wait()   # then re-read the cache; if the other call failed, this one tries itself
        try:
            return self._miss(task, p, user, model, key, t0, recorder)
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            mine.set()

    def _miss(self, task, p, user, model, key, t0, recorder):
        cap = getattr(self.settings, "max_calls", None)
        with self._lock:
            if cap is not None and self.stats["calls"] + self.stats["errors"] >= cap:
                raise LLMBudgetError(f"{task}: the limit of {cap} model calls is reached (Settings.max_calls / --max-calls)")
        out, usage, err, retry_reasons = None, None, None, []
        loop = self._ensure_loop()
        agent = self._agent(p, p.tier)
        for attempt in range(4):
            res, messages, e = asyncio.run_coroutine_threadsafe(self._arun(agent, user), loop).result()
            if e is None:
                out, usage = res.output, res.usage
                retry_reasons = _retry_reasons(messages)
                break
            if isinstance(e, ModelHTTPError):
                if e.status_code in (401, 403):
                    err = LLMAuthError(f"{task}: HTTP {e.status_code}: {str(e.body)[:300]}")
                elif e.status_code == 429 or e.status_code >= 500:
                    err = LLMTransportError(f"{task}: HTTP {e.status_code}: {str(e.body)[:200]}")
                else:
                    err = LLMError(f"{task}: HTTP {e.status_code}: {str(e.body)[:300]}")
            elif isinstance(e, UnexpectedModelBehavior):
                err = LLMOutputError(f"{task}: {e}")
            elif isinstance(e, ModelAPIError):
                err = LLMTransportError(f"{task}: {type(e).__name__}: {e}")
            else:
                raise e
            if not isinstance(err, LLMTransportError) or attempt == 3:
                self._log_failure(task, model, user, messages, err)
                break
            time.sleep(min(1.0 * 2 ** attempt, 8))
        latency = int((time.perf_counter() - t0) * 1000)
        if out is None:
            self._count("errors")
            if recorder is not None:
                recorder.add(LLMCallRecord(task=task, prompt_version=p.version, model=model, cached=False, ok=False, latency_ms=latency))
            raise err
        self._count("calls")
        retries = max(0, (usage.requests or 1) - 1)
        self._count("retries", retries)
        self.cache.put(key, out.model_dump_json())
        self._log_call(task, model, user, out.model_dump_json(), False, latency, recorder, retries, retry_reasons)
        if recorder is not None:
            recorder.add(LLMCallRecord(task=task, prompt_version=p.version, model=model, cached=False, ok=True, latency_ms=latency,
                                       tokens_in=usage.input_tokens, tokens_out=usage.output_tokens, retries=retries))
        return out

    def _log_call(self, task, model, user, output, cached, latency, recorder, retries=0, retry_reasons=()):
        path = getattr(self.settings, "call_log", None)
        if path is None:
            return
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "episode": getattr(recorder, "label", None), "task": task, "model": model,
               "cached": cached, "ms": latency, "retries": retries, "retry_reasons": list(retry_reasons), "prompt": user, "output": json.loads(output)}
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _log_failure(self, task, model, user, messages, err):
        path = self.settings.failure_log
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            history = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
        except Exception:  # noqa: BLE001 — the log must never hide the original error
            history = [repr(m) for m in messages]
        with self._lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "task": task, "model": model, "error": str(err),
                                "prompt": user, "messages": history}, ensure_ascii=False) + "\n")
