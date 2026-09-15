# Test data

- `docs/` — four documents describing a fictional organization (Community Safety Trust). Lines of the form
  `Risk R1: ...`, `Goal G1: ...`, `Term: label (aliases): description, a kind of Parent`, `Entity type: name (attrs)`,
  `Source: name (credibility x)` are parsed by the offline fake provider; a real model reads the prose.
- `reference_ontology.json` — reviewed reference labels for measuring ontology-extraction precision/recall.
- `episodes.jsonl` — 31 seed episodes (signals, hard negatives, easy negatives, missing-info) with occurrence times
  (exact, range, or null) and source names. Entities recur with aliases to exercise resolution.
- `labelled_entities.json` — ground truth clusters for entity resolution and must-not-merge pairs.
- `benchmark/hypotheses.json`, `benchmark/assessments.json` — human-labelled severities per (episode, hypothesis),
  hard-negative mirrors, Lambda case, missing-info cases, and counterfactual pairs. Consumed by M3.

All names, places, and organizations are fictional.
