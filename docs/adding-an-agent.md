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
packet and the state transition. `reasoning` is a short label shown per-decision by
`scripts/inspect_run.py`, and `was_fallback` marks a decision your policy did not really make —
set it truthfully, because it is what the substitution audit below counts. Always choose through the mask. A policy that previews actions may
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

A shop policy is a class with `decide_shop(env, raw_state, mask, value) -> FactoredAction`.
`src/bench/agents.py` imports only what the two built-in baselines happen to use, so **your policy
almost certainly needs imports the file does not have yet** — add them:

```python
from src.playground.harness import (
    _BUY_CARD,          # action-type constants: also _PLAY, _DISCARD, _SELECT_BLIND,
    _NEXT_ROUND,        #   _CASHOUT, _NEXT_ROUND, _SKIP_PACK, _PICK_PACK
    FactoredAction,     # the action you return
    build_decider,      # composes your policy into a decide_fn
    get_fallback_action,  # a legal action when you have nothing to do
)
```

Forgetting one is not a loud failure. See the next section.

## The harness will not let your agent crash — including when it should

`build_decider` gates every slot output through the legality mask and wraps every slot call in
`except Exception`. Anything illegal or anything that raises is replaced with a legal fallback
action and the game continues. That is deliberate: one bad state must not kill a 240-seed battery.

The consequence is that **a completely broken policy still produces a complete, plausible,
statistically significant result.** A missing import raises `NameError` on every shop decision; all
240 games still finish, `evaluate.py` still reports `0 fails` (a "fail" is a crashed *game*, not a
dead policy), and the paired interval can comfortably exclude zero — measuring the fallback action,
not your agent.

So check the substitution rate, every time, before you believe a number:

```
$ uv run python scripts/evaluate.py --agent my-agent-v1 --vs greedy-shop --limit 8 --workers 4

my-agent-v1: 8 runs -> ...
  mean highest ante: 1.375
  fallback substitutions: 40 / 180 decisions (22.2%) across 7/8 runs -- fallback-error:NameError=40
  ERROR: 40 decision(s) raised inside the my-agent-v1 policy (fallback-error:*). ...
```

A healthy agent prints `fallback substitutions: 0 / N decisions`. The reasons mean:

| reason | meaning |
| --- | --- |
| `fallback-error:<ExcType>` | your policy raised. Always a bug in your agent. |
| `fallback-illegal` | your policy returned an action the mask forbids. Always a bug in your agent. |
| `fallback-phase` | the game reached a phase no slot handles. A harness gap, not yours. |

The same counts are in the result artifact under `summary.fallback`, and
`scripts/inspect_run.py <runs.jsonl> --list` shows which runs were affected; inspecting one marks
each substituted decision `!! FALLBACK`. **A result from an agent with a nonzero
`fallback-error`/`fallback-illegal` rate is not a measurement of that agent and must not be
reported as one.**

## Compare it

First test registration, then run a smoke:

```bash
uv run python scripts/evaluate.py --list
uv run python scripts/evaluate.py \
  --agent my-agent-v1 --vs greedy-shop --limit 8 --workers 4
```

Read the `fallback substitutions:` line for **both** arms before reading the delta, then audit raw
runs with `scripts/inspect_run.py`. Only after that should you remove `--limit` for the full
240-seed comparison. Report negative or null intervals as results, not invitations to tune on
the same public battery until the sign changes.

Never change an existing stable agent ID's behavior. A behavioral revision gets a new ID.
