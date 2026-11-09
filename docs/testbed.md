# Testbed and code layout

How Mistabra is measured, and where things are in the repository.

## Running the testbed

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

