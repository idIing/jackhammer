# Result artifacts

`scripts/evaluate.py` writes raw run JSONL plus two stable JSON envelopes.

## `jackhammer.result/v1`

One agent over one seed split. Required top-level fields:

- `schema`, `created`, and `agent` — the agent block carries `name`, `description`,
  `deterministic`, the `slot1`/`slot2` provenance labels, `tactical`, which names the
  in-blind play policy (the baselines record `GreedyTactical(score_budget=300)`), and
  `declared_actions`, the action types the agent's description claims it emits (`null` for an
  agent that publishes no such claim);
- `provenance`: protocol, seed file path/digest/split/count, engine pin and dirty state, kit pin, and
  Python runtime. The kit pin is `{version, commit, dirty}`: `version` is the installed release
  (`1.0.0` and up), `commit` resolves the exact tree and is `null` for a consumer who installed the
  kit rather than cloning it. Either may be absent information; neither substitutes for the other;
- `attributable`: true only when an exact clean engine commit is known;
- `runs_path`: raw JSONL for drill-down; and
- `summary`: run depth, blind statistics, win interval, advance curve, and `repertoire` —
  `n_decisions`, `n_fallback`, the `observed` action-type histogram, and the `undeclared` /
  `unexercised` differences against `agent.declared_actions`. That block is what makes a
  behavioural claim about an agent checkable from the artifact instead of from the source; see
  [known limits](known-limits.md#declared-repertoires).

## `jackhammer.comparison/v1`

Two compatible results joined by seed. The writer refuses different protocols, engine commits,
seed-file digests, or splits. `comparison` includes paired seed records, per-arm summaries, mean
highest-ante delta with bootstrap interval, advance-curve overlays, and McNemar reads for wins and
reaching a declared ante.

Machine-readable envelope schemas live in `schemas/`. The raw JSONL remains the audit source; the
summary can be re-derived with `jackhammer.playground.metrics`.
