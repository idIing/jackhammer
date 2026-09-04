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
- `summary`: run depth, blind statistics, win interval, advance curve, and `fallback`.

`summary.fallback` (added in kit v1.1; additive, so v1 consumers are unaffected) is the audit that
says the numbers beside it came from the agent and not from the harness. `build_decider` replaces
any illegal or raising slot decision with a legal fallback so one bad state cannot kill a battery,
which means a policy that never executes still produces a complete, significant result. The block
carries `n_decisions`, `n_fallback`, `rate`, `runs_affected`, `n_runs`, `by_reason` (keyed by
`fallback-error:<ExcType>` / `fallback-illegal` / `fallback-phase`), and `errors` — the
`fallback-error:*` subtotal, which is always a defect in the agent. **Read it before reading
`run_depth`:** a result with a nonzero `errors` is not a measurement of the named agent.

## `jackhammer.comparison/v1`

Two compatible results joined by seed. The writer refuses different protocols, engine commits,
seed-file digests, or splits. `comparison` includes paired seed records, per-arm summaries, mean
highest-ante delta with bootstrap interval, advance-curve overlays, and McNemar reads for wins and
reaching a declared ante.

Machine-readable envelope schemas live in `schemas/`. The raw JSONL remains the audit source; the
summary can be re-derived with `src.playground.metrics`.
