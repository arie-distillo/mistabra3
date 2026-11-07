# Mistabra Core

An analyst's co-pilot that turns intuition into reasoning — the reasoning layer, implementing the conceptual API
in [`docs/mistabra_core_api_v3.md`](docs/mistabra_core_api_v3.md).

One call. It takes one episode with any part of its trajectory (signal, hypothesis, risk), completes what is
missing, and assesses the result against baseline alternatives — on its own (`score`) and together with what is
already attached to the same hypothesis (`standing`):

```python
from mistabra_core import Core, Settings, TrajectoryInput, Episode

core = Core(Settings.from_env())
a = core.assess(mission, operational, TrajectoryInput(episode=Episode(id="e1", text="...", source="Field reports", credibility=0.8)))
for g in a.groups:
    t = g.target
    print(t.risk.risk_id, t.hypothesis.origin, t.hypothesis.claim, t.score, g.standing.value, t.lambda_signal, t.lambda_hypothesis)
```

The layer reads the mission context (documents, risks; versioned) and the operational context (hypotheses with
`baseline_of`, every trajectory kept, with status) and writes neither. Whatever it creates — signals, new
hypotheses, new baselines — comes back in the result; the other layer decides what to keep.

## Layout

```
mistabra_core/          the layer
  types.py              API types: MissionContext, OperationalContext, TrajectoryInput → Assessment
  core.py               the reference implementation of assess()
  scoring.py            score (documented formula) and standing strategies
  settings.py           provider, models, standing strategy, thresholds
  llm/                  model access via PydanticAI (output tools, validation retries), gateway (cache, call records,
                        failure log), prompts, fake model
testbed/                quality measurement
  scenario.py           one scenario format + adapters for the PoC labelled set, ACH case files, phase-1 cases
  harness.py            a stand-in for the other layer (keeps what assess returns) and the runner
  metrics.py            A1–A6, R1, step expectations
  run.py                CLI; writes testbed/reports/*.json and *.md
  scenarios/            authored scenarios (flood conjunction + control, analyst hunches, noise drift, stability)
  sources/              case files in their own formats (burnt bread ACH, phase-1 conj/miss cases)
  data/community_safety the PoC's 31 labelled episodes
  private/              (git-ignored) drop more case files here to include them locally
tests/                  functional tests, offline (55)
docs/                   the spec
```

## Setup

```bash
pip install -e ".[dev]"
```

Per episode and candidate risk the layer makes three model calls (plus `radar.derive`, from the risk and mission
documents only, once per mission version):

1. `episode.triage` (fast tier) — is the episode relevant, directly or through a held hypothesis, and which radar does
   it answer? An irrelevant episode stops here, after one call.
2. `episode.assess` (reasoning tier) — a verdict on every held hypothesis (supports / weakens / unrelated), new
   hypotheses only for mechanisms the held ones miss, and the ordinary explanations (baselines) of each.
   If a supported held hypothesis now has 3 supporting facts and no refinement yet, `hypothesis.sharpen` (reasoning
   tier) asks whether the facts together show a more specific mechanism; a sharper claim becomes a refinement.
3. Standing of every hypothesis the episode touched, before and after it: computed (noisy-OR with the safeguards of
   spec §5.2a–b; the default) or, with `MISTABRA_STANDING=judged`, one `standing.judge` call (reasoning tier).

All model calls go through PydanticAI: each prompt's Pydantic schema is the output type, the model answers through an
output tool, and a validation error is sent back to the model for up to two more attempts. Settings come from `.env`
(copy `.env.example`), the environment, or `Settings(...)` in code.

```bash
export MISTABRA_PROVIDER=anthropic            # anthropic | openrouter | openai (any OpenAI-compatible endpoint) | fake
export MISTABRA_MODEL=claude-sonnet-4-5       # reasoning tier: assessment, standing
export MISTABRA_FAST_MODEL=claude-haiku-4-5   # operational tier: episode triage
export MISTABRA_API_KEY=...
export MISTABRA_BASE_URL=...                  # openai provider: OpenRouter, Ollama, vLLM, ...
```

## Tests and testbed

```bash
pytest                                        # functional tests: fake provider, offline, < 1 s
python -m testbed.run                         # every scenario end to end with the fake provider: proves the plumbing
python -m testbed.run --live --compare --cache .cache/m.sqlite
                                              # the measurements: judged vs noisy-OR standing on identical inputs
python -m testbed.run --live --only miss_ --only flood
python -m testbed.run --live --jobs 8                 # scenarios in parallel (default 4)
```

The functional tests cover all eight input patterns, scope and detail, inconsistency notes, reuse of stored
hypotheses and baselines, rejected hypotheses, Lambda signals and Lambda hypotheses, standing weights and
`without`/`delta`, provenance, determinism and that inputs are never mutated.

The testbed measures (spec §8):

| Id | What | Scenarios |
| --- | --- | --- |
| A1 | score lands in labelled bands; positives above hard negatives; orderings hold | community_safety (31 labelled), authored |
| A2 | the missing hypothesis is articulated and reaches standing; no false alarms in controls; Lambda flags | miss_01–04, miss_ctrl_*, analyst_hunches |
| A3 | weak episodes combine into high standing; refuting sets lower a stated hypothesis | flood_conjunction/control, conj_01–06, conj_ctrl_* |
| A4 | articulated claims match the stored hypothesis they repeat | flood_conjunction, analyst_hunches |
| A5 | score spread across paraphrases of one episode | stability_elm |
| A6 | ordinary episodes do not build a hypothesis from the layer's own proposals | noise_drift |
| R1 | final standing orders the stated claims as the ACH ground truth does | burnt_bread_ach_classic (+ any private ACH file) |

With the fake provider the numbers prove that every path runs; they are not measurements of the method. The
fake cannot articulate a missing hypothesis or recognise a refutation, so A2 and A3 come out near zero offline —
which is also a check that the metrics discriminate.

`--compare` runs the same scenarios with `standing=noisy_or` (computed: supporting trajectories accumulate, each
weakening one removes up to half, neutral ones count nothing; the default) and `standing=judged` (the reasoning
model weighs the attached trajectories together). Both modes share every other model answer, so the comparison is
paired. Over three runs on 2026-11-06 noisy-OR was equal or better on every standing metric, at no model cost, so it
became the default; `judged` stays available.

## History

The previous PoC (`mistabra/` package, SQLite store, facade APIs `risks`, `radars`, `signals`, …) is kept on the
branch `legacy/poc-0.2`. Reused from it, adapted: provider abstraction, gateway, prompts, fake provider, the score formula and
the labelled community-safety data. Its storage and facade belong to the other layer and are not carried over.
