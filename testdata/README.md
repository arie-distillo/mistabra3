# Test data — fictional "Community Safety Trust"

- `context/` — three documents the customer would upload: who they are, past cases, sources and credibility.
- `risks.json` — two risks; `attack` is the labelled one, `burglary` is unrelated (Lambda signals must not land on it).
- `episodes.jsonl` — 31 episodes: 16 signals across the strength range (ep16 is the Lambda case), 8 hard negatives that mirror
  a positive's surface with an ordinary explanation, 4 easy negatives, 3 missing-information cases.
- `benchmark/labels.json` — human-labelled bands per episode for the `attack` risk, ordering constraints (counterfactual pairs),
  and the expected accumulation order on Elm Street.

All names, places and organisations are fictional.
