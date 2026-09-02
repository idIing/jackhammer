# Contributing

Jackhammer welcomes agent baselines, diagnostic datasets, benchmark tooling, documentation, and
small simulator fixes. The review question is not “is this idea fashionable?” It is “can another
person reproduce the claim and understand exactly what it covers?”

## Development setup

```bash
git clone https://github.com/idIing/jackhammer.git
cd jackhammer
uv sync --locked
uv run ruff check src scripts tests
uv run pytest
```

Run a four-seed evaluator smoke after changing the agent or runner path:

```bash
uv run python scripts/evaluate.py \
  --agent greedy-shop --vs random-shop --limit 4 --workers 2
```

## Agent contributions

- Use a new, stable agent ID. Do not change a published policy under an existing ID.
- Seed every policy RNG from the game seed and declare any remaining nondeterminism.
- Compare against the closest existing baseline on identical seeds with `--vs`.
- Report mean highest ante, the paired bootstrap interval, failures, engine commit, and dataset
  digest. Null and negative results are welcome.
- Do not use an LLM as the gameplay policy or teacher. Supervised targets must come from measured
  outcomes, not model opinions.

See [docs/adding-an-agent.md](docs/adding-an-agent.md) for the code interface.

## Dataset contributions

Do not edit `config/seed_battery_v1.json`. That file is frozen. Add a versioned
`jackhammer.seed-dataset/v1` manifest and explain the sampling method, intended estimand, overlap,
and known selection effects. Diagnostic results must not be presented as v1 performance.

## Simulator issues

Jackdaw is a proxy. A simulator discrepancy needs the exact Jackdaw commit, a minimal seed/state
reproduction, observed behavior, expected behavior, and a source or live-game witness for the
expectation. Generally useful fixes should go upstream to Jackdaw when practical.

## Pull requests

Keep changes focused. Include the commands you ran and their outcomes. Do not mix a protocol change
with an agent result or unrelated cleanup. Changes to the engine pin, v1 battery, primary metric,
holdout rule, baseline slate, or evaluation procedure require a new protocol version.
