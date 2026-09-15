# Milestone status

| Milestone | Scope | Status |
|---|---|---|
| M0 Foundations | package, SQLite schema (incl. M3/M4 tables), audit-in-transaction + hash chain, job runner, gateway (2 tiers, prompts versioned, cache, call log, no retry on malformed output), eval harness, seed benchmark | done |
| M1 Ontology builder | chunking, extraction (operational), signal inference (reasoning), draft with provenance, review accept/edit/reject/merge, publish immutable version with reference validation, revision carry-over, import, export | done (review API; local UI deferred) |
| M2 Ingest/extract/entities | sources with credibility, idempotent episodes with exact/range/unknown occurrence time, observation + mention extraction with modifiers, vector indexing, resolution (rules+fuzzy+adjudication, Splink optional), merge/split/relink audited, entity links and graph view | done (connectors: jsonl/text/direct; HTTP pull deferred) |
| Tests | 62 unit tests (offline, ~1 s), 22 acceptance checks via public API with `--live` switch, eval harness | done |

Exit criteria met: mocked end-to-end path (M0); confirmed ontology version from documents with extraction P/R measured (M1);
pilot-style episodes flow to resolved entities with resolution P/R measured (M2). Quality numbers with a real model
require a key and are produced by `mistabra eval` / `tests.acceptance.run --live`.

Next: M3 assessment engine (candidate linking, Stage 1 relevance/Lambda, Stage 2 baseline explanations + severity,
hypothesis aggregation, replay), consuming `testdata/benchmark/`.
