# Mistabra Core

An analyst's co-pilot that turns intuition into reasoning: the reasoning layer.

Analysts receive a stream of short reports. Most mean nothing; a few matter, often only in combination, and sometimes
they point to an explanation nobody has stated yet. Mistabra Core reads one report (an *episode*) at a time and asks:
does it bear on a risk, through which hypothesis, how strongly, and against which ordinary explanations? It keeps
weak signals instead of dropping them, lets several weak ones add up, weakens hypotheses that the evidence contradicts,
and articulates new hypotheses when the evidence points somewhere the analyst has not looked.

> **Status: research preview (0.x).** The API may still change. The method is measured on a testbed of synthetic
> cases (see [Results](#results)), not yet on real analyst work. Scores and standings are ordered strengths for an
> analyst to weigh, not probabilities. Read [Limitations](#limitations) before relying on it.

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env              # choose a provider and model, add your key
python examples/cold_store.py     # or --offline: the fake model, shows the flow only
```

[`examples/cold_store.py`](examples/cold_store.py): a warehouse keeps losing chilled stock. The analyst suspects a
failing compressor or doors left open. Eight reports arrive: the compressors are fine, the doors are closed, a new
night cleaning crew started in March, one cold room's temperature log has gaps between 1 and 3 a.m., the cleaners
charge their scrubbers from the socket next to that room's controller, every spoiled batch came from that room, plus
two irrelevant reports. Mistabra weakens both of the analyst's hypotheses, proposes the one nobody stated (the
controller is unplugged at night), and builds its standing as the clues combine.

```python
from mistabra.casebook import Casebook
from mistabra.core import Core, Episode, Settings

case = Casebook(Core(Settings.from_env()), mission, path="case.sqlite")   # keeps the case between calls
case.add_hypothesis("Stock spoils because a compressor is failing.", risk="spoilage")

got = case.ingest(Episode(id="e5", text="The temperature log in Cold Room 2 shows gaps between 1 and 3 a.m."))
for g, hid in zip(got.assessment.groups, got.hypothesis_ids):
    print(g.target.stance, hid, g.target.score, g.standing.value, g.target.hypothesis.claim)

case.confirm(got.trajectories[0], by="analyst", note="checked the logger")   # analyst decisions are recorded
case.export("feedback.jsonl")                                                # decisions joined with what Mistabra said
```

## How it works

Two parts:

- **`Core.assess(mission, operational, trajectory)`**: one stateless call per episode
  ([spec](docs/mistabra_core_api_v3.md)). The *mission context* is what the analyst sets up: documents and risks.
  The *operational context* is the case so far: hypotheses and the trajectories attached to them. The trajectory
  input is the episode, optionally with the signal, hypothesis or risk the analyst already sees in it. The result is
  one group per hypothesis the episode bears on: the trajectory (episode → signal → hypothesis → risk) with its
  stance (supports / weakens / neutral), `score` (this episode alone), the competing ordinary explanations, and
  `standing` (the hypothesis given everything attached, before and after this episode).
- **`Casebook`**: keeps the case between calls. It stores what the core returns, re-reads earlier episodes when a new
  hypothesis appears (look-back), records analyst decisions with who, when and why, persists to SQLite, and exports
  the decisions with the assessments they concern.

Per episode and risk, the core makes 1 model call if the episode is irrelevant and 2-4 if it is relevant:

1. **Triage** (fast model): is the episode relevant, directly or through a held hypothesis, and which *radar* does it
   answer? Radars are derived once per mission from the risk and the documents only, never from current hypotheses,
   so the layer does not look only where the analyst already looks.
2. **Assessment** (reasoning model): a verdict on each held hypothesis it touches, at most two new hypotheses for
   mechanisms none of them covers, and the ordinary explanations of each.
3. **Sharpening** (reasoning model, when a hypothesis has three supporting facts): do the facts together show a more
   specific mechanism? A sharper claim becomes a refinement that inherits the evidence.
4. **Standing**: computed by noisy-OR with safeguards (support never lowers it, counter-evidence never raises it,
   at most 0.8 until the analyst confirms something), or optionally judged by the reasoning model.

Every element of a trajectory says how it came to be (`origin`: given by the analyst, predefined, i.e. matched to a
radar or stored hypothesis, or articulated by the layer). An articulated signal or hypothesis is a *Lambda* one:
something nobody had stated. [The glossary](https://claude.ai/artifact/UomNUKdAnmSzKYnWQKMdei) explains every term.

## Models

Any of: Anthropic, OpenAI, OpenRouter, or any OpenAI-compatible endpoint, including local servers (Ollama, vLLM,
llama.cpp) for data that must not leave your machine. Episodes and mission documents are sent to the provider you
configure.

```bash
MISTABRA_PROVIDER=openrouter          # anthropic | openai | openrouter | fake
MISTABRA_MODEL=qwen/qwen3.5-27b       # reasoning tier
MISTABRA_FAST_MODEL=qwen/qwen3.5-flash-02-23
MISTABRA_API_KEY=...
MISTABRA_BASE_URL=http://localhost:11434/v1   # with MISTABRA_PROVIDER=openai: a local OpenAI-compatible server
MISTABRA_MAX_CALLS=1500               # optional: stop after this many model calls
```

All calls go through PydanticAI: each answer must fit a schema, and an invalid one is sent back with the error (up
to three times). Answers are cached by their exact input (`cache_path`), so identical inputs are not paid for twice.

## Results

Measured on the testbed (`python -m testbed.run --live --repeat 3`) on 2026-11-08, with Qwen 3.5 27B (reasoning)
and Qwen 3.5 Flash (fast) through OpenRouter, three independent runs. The cases are synthetic, written for the
testbed; ranges are min-max over the runs.

| Capability | Measure | Result |
| --- | --- | --- |
| Separate signals from noise | labelled positives ranked above hard negatives (AUROC); false-signal rate | 0.98-1.0; 0 |
| Find the cause the analyst missed | 16 cases: same mechanism as the answer key / at least partly / reaching standing 0.5 | 11-14 / 15-16 / 9-10 |
| No false alarms | 6 control cases where nothing is missing: an established new cause | 0 in every run |
| Combine facts that only matter together | refuted vs supported analyst hypotheses (AUROC) | 1.0 |
| Combine weak signals | two weak episodes reaching standing 0.4 together | 2 of 3 runs |
| Match instead of duplicating; stable under paraphrase; no build-up from noise | | pass in every run |

Cost and speed with that setup: a full pass over the 38 scenarios takes about 10 minutes and 400 model calls; an
irrelevant episode takes about 2 s, a relevant one about 15 s.

## Limitations

- **Synthetic evaluation.** The cases and answer keys were written by the developers. The numbers show whether a
  change helps or hurts; they do not show that the method is right on real work. A blinded evaluation with domain
  analysts on representative data has not been done yet.
- **Run-to-run variance.** Hosted models are not reproducible even at temperature 0: the same case can be solved in
  one run and missed in the next (11 to 14 of 16 above). Judge changes over several runs.
- **Not probabilities.** Score and standing order things by strength; the bands (weak from 0.2, strong from 0.4) are
  a convention calibrated on a small labelled set.
- **Known failure patterns:** a vague hypothesis formed early can absorb later, more specific evidence (sharpening
  helps, not always); a routine-looking first clue may be missed until a hypothesis gives it meaning (the look-back
  helps); weak single episodes sit close to the band boundaries.
- **Cost grows with risks.** Each risk triages each episode separately.

## Responsible use

Mistabra is a reasoning aid for analysts investigating systems, processes and incidents. A score, a standing or an
articulated hypothesis is a prompt for human judgment, never evidence on its own, and never a basis for action
against a person. Do not use it to profile or monitor individuals or groups, or to infer anything about a person's
beliefs, identity or intentions from their ordinary behaviour.

## Testbed

```bash
pytest                                              # functional tests, offline, about 10 s
python -m testbed.run                               # every scenario with the fake model: proves the plumbing
python -m testbed.run --live --repeat 3 --cache .cache/m.sqlite      # the measurements, with run-to-run spread
python -m testbed.run --live --compare              # noisy-OR vs model-judged standing on the same answers
python -m testbed.run --live --only miss_ --jobs 6  # a subset, scenarios in parallel
```

Each run writes `testbed/reports/<stamp>-*.json` and `.md`, plus `<stamp>-calls.jsonl` with every model prompt and
answer. `--max-calls` (default 1500 per run) stops a run that grows beyond budget.

| Id | What it measures | Cases |
| --- | --- | --- |
| A1 | score lands in labelled bands; positives above hard negatives | a labelled set of 31 episodes, authored scenarios |
| A2 | the missing hypothesis is articulated and established; no false alarms in controls | `miss_01`-`miss_17`, `miss_ctrl_01`-`06` |
| A3 | weak episodes combine; facts that refute only together lower a stated hypothesis | flood pair, `conj_01`-`06`, `conj_ctrl_*` |
| A4 | a claim that repeats a stored hypothesis is matched to it | flood pair, analyst input patterns |
| A5 | score spread across paraphrases of one episode | stability |
| A6 | ordinary episodes do not build a hypothesis from the layer's own proposals | noise stream |
| R1 | standing orders competing hypotheses as an ACH answer key does | burnt bread ACH case |

## Layout

```
mistabra/             the library
  core/               the reasoning layer (stateless)
    engine.py         assess(): triage, assessment, sharpening, score, standing
    types.py          the API types
    scoring.py        score formula, noisy-OR standing
    settings.py       provider, models, thresholds, limits
    llm/              model access (PydanticAI), cache, call log, prompts, offline fake model
  casebook.py         the layer above: keeps a case, look-back, analyst decisions, export, SQLite
examples/             cold_store.py, the quickstart
testbed/              scenarios, adapters for case files, runner, metrics, reports
tests/                functional tests (offline)
docs/                 the API spec
```

## License

To be decided before the first public release.
