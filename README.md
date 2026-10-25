# Mistabra PoC

An analyst's co-pilot that turns intuition into reasoning — proof of concept, built from scratch
against the API spec v3 (PoC subset). In-process Python library; all data in one directory.

## What it does

Context (documents) → risks (a sentence each) → radars (queries derived from each risk) → episodes.
Each episode arriving triggers, inline: every enabled radar reads it (relevance + structured extraction);
a hit creates a **signal**; for open risks, an episode no radar matched gets the open-risk check and may
become a **Lambda signal**; every signal gets a **risk trajectory** (one hypothesis, a rationale, reach, fit)
and **baseline trajectories** (ordinary outcomes), written before any strength; strength is computed from
reach, fit, gap, specificity and source credibility. The analyst rates blind, then sees the system's view,
confirms or rejects trajectories (confirmed ones become precedent), and reads hypothesis coverage per risk.
Everything is audited with a hash chain; any signal can be traced to its model calls; replay re-reasons
signals and diffs against what was stored.

## Install and run

```bash
pip install -e .            # pydantic, httpx
pip install -e .[dev]       # + pytest

python -m pytest                          # unit tier: offline, deterministic, ~1 s
python -m tests.acceptance.run            # the PoC demonstration list, fake provider
python -m tests.acceptance.run --live     # same against the configured real model
mistabra bench                            # benchmark on testdata/, reported metrics
```

## LLM setup — one interface, any provider

```bash
export MISTABRA_PROVIDER=anthropic            # anthropic | openai | fake
export MISTABRA_MODEL=claude-sonnet-4-5       # reasoning tier: radar derivation, trajectories
export MISTABRA_FAST_MODEL=claude-haiku-4-5   # operational tier: radar reads, open-risk check (defaults to MODEL)
export MISTABRA_API_KEY=...                   # or ANTHROPIC_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY
export MISTABRA_BASE_URL=...                  # openai: any OpenAI-compatible endpoint (OpenRouter, Ollama, vLLM, Groq)
```

Examples: OpenRouter → `provider=openai base_url=https://openrouter.ai/api/v1 model=anthropic/claude-sonnet-4.5`;
Ollama → `provider=openai base_url=http://localhost:11434/v1 model=llama3.1` (no key needed).
`MISTABRA_FORCE_FAKE=1` overrides everything (unit tests set it). Or in code:

```python
from mistabra import Mistabra, Settings
m = Mistabra("./data", Settings(data_dir="./data", provider="openai", base_url="https://openrouter.ai/api/v1",
                                api_key="...", model="anthropic/claude-sonnet-4.5", fast_model="anthropic/claude-haiku-4.5"))
```

A provider is one class with one method (`complete(model, system, user, max_tokens) -> (text, usage)`);
prompts, JSON parsing, caching, logging and retries live in the gateway. Malformed output is an error, not a retry.

## The API (PoC subset)

```python
m.context.create(text, actor, title=…, kind=…) / create_file(path, actor) / search(query) / list(kind)
m.risks.create(text, actor, open=False) / get / list / update(id, actor, open=…, status=…)
m.radars.create(risk_id, query, actor) / derive(risk_id, actor) / list(risk_id) / update(id, actor, enabled=…)
m.sources.create(name, actor, credibility=…)
m.episodes.create(text, actor, source=…, occurred=…, metadata=…) / create_many / get / list      # drives everything
m.signals.list(risk_id=…, status=…, via=…, episode_id=…) / get / update(id, actor, strength=…, status=…) / create(episode_id, risk_id, actor, note=…) / remove
m.trajectories.list(signal_id=…) / get / update(id, actor, status="confirmed"|"rejected")
m.hypotheses.list(risk_id=…) / get / signals(id) / coverage(risk_id)
m.audit.events(…) / trace(signal_id) / verify()
m.learn.replay(signal_ids, actor, model=…, prompt_version=…)
m.stats()
```

Blind rule: `Signal.strength_system`, `metrics`, `trajectory` and `baselines` read as empty until the analyst sets
`strength` on the signal (`Settings(blind=False)` turns it off; the mode is recorded on every audit event).

Strength: `0.4·fit + 0.4·max(gap,0) + 0.2·reach`, scaled down by `(1+gap)` when an ordinary explanation fits better,
then by specificity and source credibility. Deterministic, so `learn.replay` can isolate model/prompt changes.

## Layout

```
mistabra/   settings.py  store.py (+schema.sql)  models.py  engine.py
            llm/ base.py (interface + Anthropic + OpenAI-compatible)  fake.py  prompts.py  gateway.py
            context.py  risks.py (risks, radars)  episodes.py (sources, episodes)  signals.py (signals, trajectories, hypotheses)
            api.py (Mistabra, audit, learn)  eval/benchmark.py  cli.py
tests/      unit/ (29 tests, offline)   acceptance/run.py (15 checks, --live)
testdata/   context/*.md  risks.json  episodes.jsonl  benchmark/labels.json
```

## What the fake provider does and does not prove

With `provider=fake` the suite proves the plumbing: consequences fire, the blind rule holds, trajectories precede
strength, Lambda only on open risks, precedent and replay work. Its benchmark numbers are heuristics, not method
quality. `mistabra bench` with a real provider produces the numbers that matter (signal recall, false signals,
band accuracy, ordering constraints, Lambda precision, missing-info flags); they are reported, not gated.

## Not in the PoC (MVP)

Background execution, risk derivation from context, radar edits replaying over history, hypothesis creation and
premises, entity resolution and corrections, per-radar/risk stats, precedent export, events, the MCP server.
