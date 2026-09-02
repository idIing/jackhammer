# Adding an agent

An agent is registered with `src.bench.agents.AgentSpec`:

```python
from src.bench.agents import AgentSpec, register


def make_decider(env, seed):
    policy = MyPolicy(env=env, seed=seed)

    def decide(raw_state, mask, history):
        action = policy.choose(raw_state, mask)
        return action, "my-policy", "choose", {}, False

    return decide


register(
    AgentSpec(
        name="my-agent-v1",
        description="One honest line shown by evaluate.py --list.",
        make_decider=make_decider,
        deterministic=True,
    )
)
```

Place the implementation in `src/bench/agents.py` for a small baseline, or import and register it
there from a separate module. Registration happens when the evaluator imports the registry.

## Decider contract

`make_decider(env, seed)` runs once per game. The returned callable receives:

- `raw_state`: Jackdaw's current state dictionary;
- `mask`: the current legal-action mask; and
- `history`: prior non-fallback decision summaries.

It returns `(FactoredAction, reasoning, method, params, was_fallback)`. The harness records this
packet and the state transition. Always choose through the mask. A policy that previews actions may
use `env.get_state()` / `env.load_state()` but must restore the environment before returning.

Randomized policies must derive their random generator from `seed`. If exact repeated replay is not
expected, set `deterministic=False`; do not publish it as reproducible.

## The shared tactical layer

The two shop baselines are not written from scratch. Both compose
`build_decider(env, GreedyTactical(), <shop policy>, MarginValue())`, where `GreedyTactical`
(`src/playground/harness.py`) is a fixed in-blind policy: it exact-scores up to `score_budget=300`
legal play-card subsets, then clinches or digs. Holding it constant is what lets a paired
comparison attribute a difference to the shop policy alone.

You are free to replace it — an agent that plays cards better is a legitimate submission — but say
so, because a comparison against `greedy-shop` then measures both layers at once, not just the
shop. Set `AgentSpec.tactical` to a label for your in-blind policy: it is written into the result
artifact's agent identity, so a reader can see which layers differed.

## Compare it

First test registration, then run a smoke:

```bash
uv run python scripts/evaluate.py --list
uv run python scripts/evaluate.py \
  --agent my-agent-v1 --vs greedy-shop --limit 8 --workers 4
```

Audit raw runs with `scripts/inspect_run.py`. Only after that should you remove `--limit` for the
full 240-seed comparison. Report negative or null intervals as results, not invitations to tune on
the same public battery until the sign changes.

Never change an existing stable agent ID's behavior. A behavioral revision gets a new ID.
