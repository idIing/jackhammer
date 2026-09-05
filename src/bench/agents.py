"""Named agents the benchmark can evaluate.

The registry exists so that "which agent produced this number" is a *string in an
artifact* rather than an edit to the runner. Hardcoding the arms inside the runner
means every new agent forks it; the registry instead keeps third-party agents on
the same evaluation path as the built-in baselines.

An agent is a *name* plus ``make_decider(env, seed) -> decide_fn``. That is the
widest contract the harness honours (see ``harness.run_battery_with``) and
deliberately does not assume the two-slot decomposition: a search agent or a
learned policy registers here on equal footing with the baselines below.

Registering an agent::

    from src.bench.agents import AgentSpec, register

    register(AgentSpec(
        name="my-agent",
        description="one line, shown by --list",
        make_decider=lambda env, seed: my_decide_fn,
    ))
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from src.playground.harness import (
    GreedyShop,
    GreedyTactical,
    MarginValue,
    RandomShop,
    build_decider,
)
from src.selfplay.runner import random_decider


@dataclass(frozen=True)
class AgentSpec:
    """One evaluable agent.

    Attributes:
        name: registry key and the identity stamped into the result artifact.
            Stable across versions — renaming an agent breaks comparability with
            every previously published number under that name.
        description: one line, shown by ``evaluate.py --list``.
        make_decider: ``(env, seed) -> decide_fn(raw_state, mask, history)``.
            Called once per seed with a fresh env, so an agent may hold per-run
            state. A stochastic agent must derive its RNG from ``seed`` — see
            ``deterministic``.
        tactical: free-text label for the in-blind play policy, written into the
            result artifact. The baselines' shared ``GreedyTactical`` decides every
            hand played, so an artifact that names only the shop and value slots
            omits the component doing most of the work. Empty when an agent has no
            separable tactical layer.
        slot1, slot2: free-text provenance labels written into each run's
            ``RunMeta``. Historically the Slot-1 shop policy and Slot-2 value
            estimator; agents that do not decompose that way may leave them
            empty or use them for whatever two labels aid later triage.
        deterministic: whether repeated evaluation on one seed is expected to
            reproduce bit-identically. False for anything drawing from an
            unseeded RNG, which makes its numbers non-reproducible — recorded so
            a reader knows not to expect replay to match. Seeding a stochastic
            policy from the run seed is what buys this back.
    """

    name: str
    description: str
    make_decider: Callable[[Any, str], Any] = field(repr=False)
    slot1: str = ""
    slot2: str = ""
    tactical: str = ""
    deterministic: bool = True

    def identity(self) -> dict[str, Any]:
        """The agent block embedded in the result artifact."""
        return {
            "name": self.name,
            "description": self.description,
            "slot1": self.slot1,
            "slot2": self.slot2,
            "tactical": self.tactical,
            "deterministic": self.deterministic,
        }


_REGISTRY: dict[str, AgentSpec] = {}


def register(spec: AgentSpec, *, replace: bool = False) -> AgentSpec:
    """Add *spec* to the registry.

    Raises on a duplicate name unless ``replace=True``. Silent replacement is a
    correctness hazard here: two agents sharing a name would publish numbers
    under one identity.
    """
    if spec.name in _REGISTRY and not replace:
        raise ValueError(
            f"agent {spec.name!r} is already registered; pass replace=True to override"
        )
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> AgentSpec:
    """Look up a registered agent, or raise with the available names listed."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown agent {name!r}; registered: {', '.join(names())}") from None


def names() -> list[str]:
    """Registered agent names, sorted."""
    return sorted(_REGISTRY)


def all_specs() -> list[AgentSpec]:
    """Every registered agent, sorted by name."""
    return [_REGISTRY[n] for n in names()]


# ---------------------------------------------------------------------------
# Built-in baselines
# ---------------------------------------------------------------------------
# Three rungs, deliberately ordered so a submitted agent can be placed:
#
#   random-legal  no tactics at all              -- the floor
#   random-shop   greedy tactics, blind shop     -- isolates the shop decision
#   greedy-shop   greedy tactics, cheapest joker -- the hand-coded reference
#
# On greedy-shop's shop policy, measured 2026-09-02: it is *described* upstream as
# value-ranked, but the ranking degenerates. GreedyShop._rank_by_value previews each
# candidate buy and scores the result with MarginValue, which outside SELECTING_HAND
# returns `n_jokers + dollars/100` (see `MarginValue` in harness.py). Every previewed
# branch holds the
# same joker count, so the score reduces to dollars-after-purchase and the argmax is
# always the cheapest candidate. It reads no joker text, rarity, or synergy. Named
# here because the name "greedy-shop" otherwise implies a judgement it does not make.
#
# random-shop and greedy-shop share the FIXED tactical layer (GreedyTactical)
# and value estimator (MarginValue), differing only in the Slot-1 shop policy;
# that is what makes them a clean A/B. Neither is a floor, though, because both
# play cards with the same greedy tactics -- which is what random-legal is for.
# Without a floor, "N times better than random" has no referent.
#
# On blind selection and cash-out, changed in protocol v2: both shop baselines
# always select the blind and always cash out immediately. Under v1 the episode
# loop did that for them and no agent could do otherwise; it is now each agent's
# own declared policy, recorded per decision. The two abstentions it declares are
# real strategy, not absent mechanics -- SkipBlind is legal on every Small and Big
# blind and takes a tag, and a consumable used at ROUND_EVAL is freed from the pool
# before the next shop is rolled. Holding both fixed is deliberate: it is what keeps
# the paired difference between these two arms a measurement of the shop policy and
# nothing else. random-legal declares no such abstention and exercises both.


# Both shop baselines construct GreedyTactical() with its default budget; the label
# records that so a result artifact names the policy that played every hand.
DEFAULT_SCORE_BUDGET = GreedyTactical().score_budget


def _tactical_label(score_budget: int = DEFAULT_SCORE_BUDGET) -> str:
    return f"GreedyTactical(score_budget={score_budget})"


_TACTICAL_LABEL = _tactical_label()


register(
    AgentSpec(
        name="random-legal",
        description=(
            "Uniformly-random legal action in every phase, including skipping blinds "
            "and using consumables before cash-out. The floor."
        ),
        make_decider=lambda env, seed: random_decider(random.Random(seed)),
        slot1="random_decider",
        slot2="",
    )
)


def _random_shop_at(score_budget: int):
    def _random_shop_decider(env, seed: str):
        # RandomShop's RNG is seeded from the *game* seed, so the value-blind arm is
        # as reproducible as the deterministic one. This mirrors the original
        # baseline runner and is why `deterministic=True` below is honest.
        return build_decider(
            env,
            GreedyTactical(score_budget=score_budget),
            RandomShop(random.Random(seed)),
            MarginValue(),
        )

    return _random_shop_decider


def _greedy_shop_at(score_budget: int):
    def _greedy_shop_decider(env, _seed: str):
        return build_decider(
            env, GreedyTactical(score_budget=score_budget), GreedyShop(), MarginValue()
        )

    return _greedy_shop_decider


# Agents whose in-blind budget `evaluate.py --score-budget` can vary. Only the two
# shop baselines qualify: they share one GreedyTactical whose cap is the thing under
# test. A submitted agent constructs its own tactical layer, so the flag cannot reach
# it -- `with_score_budget` says so rather than silently doing nothing.
_BUDGET_VARIANTS = {"random-shop": _random_shop_at, "greedy-shop": _greedy_shop_at}


_random_shop_decider = _random_shop_at(DEFAULT_SCORE_BUDGET)


register(
    AgentSpec(
        name="random-shop",
        description=(
            "Uniformly-random legal shop action; fixed greedy tactics; "
            "never skips a blind, never uses a consumable before cash-out."
        ),
        make_decider=_random_shop_decider,
        slot1=RandomShop.name,
        slot2=MarginValue.name,
        tactical=_TACTICAL_LABEL,
    )
)

register(
    AgentSpec(
        name="greedy-shop",
        description=(
            "Buys the cheapest affordable joker; fixed greedy tactics; "
            "never skips a blind, never uses a consumable before cash-out."
        ),
        make_decider=_greedy_shop_at(DEFAULT_SCORE_BUDGET),
        slot1=GreedyShop.name,
        slot2=MarginValue.name,
        tactical=_TACTICAL_LABEL,
    )
)


def with_score_budget(spec: AgentSpec, score_budget: int) -> AgentSpec:
    """A copy of *spec* whose shared ``GreedyTactical`` uses *score_budget*.

    Exists so the scan-cap limitation in ``docs/known-limits.md`` is reproducible
    from the CLI instead of by editing this file. The returned spec keeps the
    agent's name -- it is the same shop policy -- but its ``tactical`` label
    records the budget actually used, so a result artifact can never claim the
    v1 cap while running another. ``evaluate.py`` additionally stamps any
    non-default budget as a diagnostic protocol.

    Raises:
        ValueError: if *spec* has no tunable tactical layer. ``random-legal`` has
            none, and a submitted agent builds its own, so silently returning it
            unchanged would let a sweep report a budget it never applied.
    """
    if score_budget < 1:
        raise ValueError(f"score_budget must be >= 1, got {score_budget}")
    factory = _BUDGET_VARIANTS.get(spec.name)
    if factory is None:
        raise ValueError(
            f"agent {spec.name!r} has no tunable tactical layer, so --score-budget "
            f"cannot apply to it; it is settable for: {', '.join(sorted(_BUDGET_VARIANTS))}"
        )
    return replace(
        spec,
        make_decider=factory(score_budget),
        tactical=_tactical_label(score_budget),
    )
