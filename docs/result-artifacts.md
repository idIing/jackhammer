# Result artifacts

`scripts/evaluate.py` writes raw run JSONL plus two stable JSON envelopes.

## `jackhammer.result/v1`

One agent over one seed split. Required top-level fields:

- `schema`, `created`, and `agent` — the agent block carries `name`, `description`,
  `deterministic`, the `slot1`/`slot2` provenance labels, and `tactical`, which names the
  in-blind play policy (the baselines record `GreedyTactical(score_budget=300)`);
- `provenance`: protocol, seed file path/digest/split/count, engine pin and dirty state, kit pin, and
  Python runtime;
- `attributable`: true only when an exact clean engine commit is known;
- `runs_path`: raw JSONL for drill-down; and
- `summary`: run depth, blind statistics, win interval, and advance curve.

## `jackhammer.comparison/v1`

Two compatible results joined by seed. The writer refuses different protocols, engine commits,
seed-file digests, or splits. `comparison` includes paired seed records, per-arm summaries, mean
highest-ante delta with bootstrap interval, advance-curve overlays, and McNemar reads for wins and
reaching a declared ante.

Machine-readable envelope schemas live in `schemas/`. The raw JSONL remains the audit source; the
summary can be re-derived with `src.playground.metrics`.
