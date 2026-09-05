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

### Your decider is called for every phase

**Changed in protocol v2.** The episode loop holds no policy of its own, so `decide` is called for
every phase the engine offers an action in — including `BLIND_SELECT` and `ROUND_EVAL`, which the v1
loop played on the agent's behalf. An agent ported from v1 that only handles `SELECTING_HAND` and
`SHOP` will now be asked something it has no branch for.

The two phases carry real decisions. `SkipBlind` is legal on Small and Big blinds only; it takes a
tag and advances to the next blind. A consumable used at `ROUND_EVAL` frees its key before the next
shop is rolled, so it can change what the shop offers — one step later, in the shop, is too late.

If you do not want to make those decisions, decline them explicitly, the way the shop baselines do:

```python
_SELECT_BLIND = int(ActionType.SelectBlind)
_CASHOUT = int(ActionType.CashOut)

phase = str(raw_state.get("phase", "")).upper()
if "BLIND_SELECT" in phase and mask.type_mask[_SELECT_BLIND]:
    return FactoredAction(action_type=_SELECT_BLIND), "always-select-blind", "SelectBlind", {}, False
if "ROUND_EVAL" in phase and mask.type_mask[_CASHOUT]:
    return FactoredAction(action_type=_CASHOUT), "always-cash-out", "CashOut", {}, False
```

Declining is a legitimate policy and costs nothing measurable on the current slate. Say so in your
`description`: the string is stamped into every artifact you publish, and a description that claims
a broader action set than the agent uses is the exact defect v2 exists to fix.

Leave `was_fallback` false for decisions your policy actually made. It means "the harness
substituted for the agent", not "the agent had no preference" — a policy that internally reaches for
a default still returns an ordinary action and is recorded as having decided.

## The shared tactical layer

The two shop baselines are not written from scratch. Both compose
`build_decider(env, GreedyTactical(), <shop policy>, MarginValue())`, where `GreedyTactical`
(`src/playground/harness.py`) is a fixed in-blind policy: it exact-scores up to `score_budget=300`
legal play-card subsets, then clinches or digs. Holding it constant is what makes a paired
comparison a shop-policy contrast — imperfectly, though: the 300-subset cap binds more often on
`greedy-shop`, worth about 0.1 ante, so a margin measured against it carries that handicap on the
affected seeds. `--score-budget` re-runs either baseline at another cap if you want to size that
against your own agent. See [known limits](known-limits.md).

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
