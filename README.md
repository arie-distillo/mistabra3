# Mistabra (Phase 1 — M0/M1/M2)

Weak/strong signal detection and hypothesis management, delivered as an **in-process Python
library**. All data lives in a directory the host controls: SQLite (system of record, audit,
model-call log, jobs), a persistent vector index, and immutable ontology version files.

Implemented in this drop: **M0** foundations (store, transactional audit with hash chain, jobs,
LLM gateway with two model tiers, cache and call log, evaluation harness), **M1** ontology builder
(documents → extraction + inference → draft with provenance → review → versioned publish, import,
export), **M2** ingestion, extraction, entity resolution (unresolved state, merge/split/relink).
M3 (assessment) and M4 (analyst workflow) are next; their tables already exist in the schema and
their benchmark labels ship in `testdata/benchmark/`.

## Install

```bash
pip install -e .            # core: pydantic, httpx, rapidfuzz, numpy, pyyaml
pip install -e .[splink]    # optional probabilistic entity-resolution backend
pip install -e .[vectors]   # optional LanceDB backend (see note below)
pip install -e .[dev]       # pytest
```

## Quick start (offline, fake model)

```python
from mistabra import Mistabra

m = Mistabra("./data")                      # provider defaults to the offline fake
did = m.ontology.build_draft("default", [("risks.md", open("risks.md").read())], actor="expert")
m.ontology.accept_all(did, "expert")        # or review element by element
vid = m.ontology.publish(did, "expert")     # immutable v1 under ./data/ontology/default/v1.json

src = m.sources.register("Field reports", "manual", credibility=0.8, actor="admin")
m.episodes.ingest(src, "A passer-by photographed Jens Hofer's house on Elm Street.", actor="host",
                  occurrence_time="2026-08-03T10:15:00Z")
m.process()                                 # runs extraction and entity-resolution jobs

m.episodes.list(); m.entities.list(); m.entities.unresolved()
m.audit.events(); m.audit.verify_chain()
m.callbacks.on("entity.merged", print)
```

CLI equivalents: `mistabra ontology-build ./data --docs ./testdata/docs --accept-all --publish`,
`mistabra ingest ./data --source "Field reports" --jsonl ./testdata/episodes.jsonl`,
`mistabra process ./data`, `mistabra stats ./data`, `mistabra eval ./data --testdata ./testdata`.

## Real models

Configure by environment (or pass a `Settings`):

```bash
export MISTABRA_PROVIDER=anthropic            # or openai_compatible
export ANTHROPIC_API_KEY=...                  # or MISTABRA_API_KEY / OPENROUTER_API_KEY
export MISTABRA_BASE_URL=https://openrouter.ai/api/v1   # openai_compatible only
export MISTABRA_REASONING_MODEL=...           # strong model: signal inference, (M3) assessment
export MISTABRA_OPERATIONAL_MODEL=...         # cheap model: extraction, adjudication
```

`MISTABRA_FORCE_MOCK=1` overrides everything and routes to the fake — unit tests set it, so a key
in the environment can never turn a unit run into a billable one.

## Tests and evaluation

```bash
python -m pytest                              # unit tier: offline, deterministic, ~1 s
python -m tests.acceptance.run                # acceptance journeys via the public API, fake model
python -m tests.acceptance.run --live         # same journeys against the configured real model
mistabra eval ./eval-data --testdata ./testdata   # measurements (not gates): extraction P/R, resolution P/R
```

The fake provider proves plumbing, not quality: with it, evaluation metrics are trivially perfect
because the test documents are written in the shapes it parses. Run `eval` with a real provider to
measure method quality; those numbers are tracked over time, not used as pass/fail.

## Test data (`testdata/`)

Fictional organization "Community Safety Trust": four domain documents, a reviewed reference
ontology, 31 seed episodes (signals across the strength range, hard negatives that mirror
positives, easy negatives, missing-information cases), entity-resolution ground truth, and the
human-labelled (episode × hypothesis) severity benchmark for M3, including a Lambda case and
counterfactual pairs.

## Design notes and known limits

- **Ontology JSON schema** has two behavioural layers, observables and signal types, linked by
  `can_serve` with a base-rate prior; concepts, entity types, sources. Baseline explanations are
  deliberately not in the ontology (generated at runtime in M3).
- **Entity resolution** default: normalization + alias/surname rules + fuzzy similarity, with the
  operational model adjudicating the ambiguous band and `unresolved` as a first-class outcome.
  `SplinkResolver` is available as an opt-in backend with hand-set parameters; it becomes useful
  once confirmed links exist to train on.
- **Vector backend** defaults to a persistent numpy index (adequate to tens of thousands of
  vectors, deterministic). LanceDB is opt-in (`vector_backend="lancedb"`) because it crashed
  intermittently at interpreter shutdown in the development environment.
- **Embeddings** default to a hashing embedder (offline). Use `OpenAICompatibleEmbedder` for
  production similarity quality, including multilingual/Hebrew.
- The fake NER is English-only and naive by design; the real extraction prompt is language-agnostic.
- Single-process: the host should use one `Mistabra` instance per data directory.
