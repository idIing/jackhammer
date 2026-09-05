from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_machine_readable_schemas_are_valid_json():
    result = json.loads((ROOT / "schemas" / "result-v1.schema.json").read_text())
    comparison = json.loads((ROOT / "schemas" / "comparison-v1.schema.json").read_text())
    assert result["properties"]["schema"]["const"] == "jackhammer.result/v1"
    assert comparison["properties"]["schema"]["const"] == "jackhammer.comparison/v1"


def test_example_dataset_loads():
    from src.bench.datasets import load_dataset

    dataset = load_dataset(ROOT / "examples" / "datasets" / "coverage-example.json")
    assert dataset.name == "coverage-example"
    assert len(dataset.seeds("sample")) == 3


def test_shop_baselines_record_their_tactical_layer():
    """The shared in-blind policy must be named in the artifact.

    ``GreedyTactical`` decides every hand ``random-shop`` and ``greedy-shop``
    play, so an identity block listing only the shop and value slots hides the
    component doing most of the work. ``random-legal`` has no tactical layer and
    correctly records an empty label.
    """
    from src.bench import agents

    for name in ("random-shop", "greedy-shop"):
        identity = agents.get(name).identity()
        assert identity["tactical"] == "GreedyTactical(score_budget=300)", identity

    assert agents.get("random-legal").identity()["tactical"] == ""


def test_protocol_is_v2():
    """v1 numbers were produced by agents that could not skip a blind.

    The stamp is what tells a reader which apparatus produced a number. Pinning it
    here means demoting the protocol has to be deliberate.
    """
    from src.bench.provenance import PROTOCOL

    assert PROTOCOL == "jackhammer/v2"


def test_episode_loop_plays_no_phase_for_the_agent():
    """The runner must not decide anything. Protocol v2's whole content.

    Guarding the *absence* of a policy is awkward, so this asserts the observable
    consequence: a decider that refuses to act is asked about every phase the engine
    offers an action in, including the two the v1 loop auto-played. If someone
    reintroduces an auto-step, BLIND_SELECT or ROUND_EVAL stops appearing here.
    """
    from jackdaw.env import BalatroEnvironment, DirectAdapter

    from src.playground.seeds import load_battery
    from src.selfplay.runner import play_episode

    seen: list[str] = []

    def recording_decider(raw_state, mask, history):
        seen.append(str(raw_state.get("phase", "")).upper())
        raise RuntimeError("stop")

    env = BalatroEnvironment(adapter_factory=DirectAdapter)
    seed = load_battery("train")[0]
    # The engine opens on BLIND_SELECT; under v1 the loop stepped past it and the
    # first phase any decider ever saw was SELECTING_HAND.
    try:
        play_episode(env, seed, recording_decider)
    except RuntimeError:
        pass
    assert seen == ["BLIND_SELECT"], seen


def test_shop_baselines_declare_their_abstentions_as_policy():
    """Never-skip and never-use-before-cash-out are the agent's choice, not the harness's.

    They must be recorded as ordinary decisions -- ``was_fallback`` false, under
    their own reasoning label -- so an artifact shows the choice was made. Routing
    them through ``get_fallback_action`` instead would take the same actions while
    reporting the reference baseline as substituted-for on thousands of decisions.
    """
    from jackdaw.env import ActionType, BalatroEnvironment, DirectAdapter, get_action_mask

    from src.playground.harness import GreedyShop, GreedyTactical, MarginValue, build_decider
    from src.playground.seeds import load_battery

    env = BalatroEnvironment(adapter_factory=DirectAdapter)
    decide_fn = build_decider(env, GreedyTactical(score_budget=64), GreedyShop(), MarginValue())
    _obs, mask, info = env.reset(seed=load_battery("train")[0])
    raw = info["raw_state"]

    assert "BLIND_SELECT" in str(raw.get("phase", "")).upper()
    # The abstention is only meaningful where the alternative was actually offered.
    assert mask.type_mask[int(ActionType.SkipBlind)], "seed does not offer SkipBlind"

    fa, reasoning, _method, _params, was_fallback = decide_fn(raw, mask, [])
    assert int(fa.action_type) == int(ActionType.SelectBlind)
    assert reasoning == "always-select-blind"
    assert was_fallback is False

    # Same at cash-out: step to ROUND_EVAL by losing the blind outright.
    while "ROUND_EVAL" not in str(raw.get("phase", "")).upper():
        mask = get_action_mask(raw)
        if not mask.type_mask.any():
            break
        fa, _r, _m, _p, _w = decide_fn(raw, mask, [])
        _obs, term, trunc, mask, info = env.step(fa)
        raw = info["raw_state"]
        if term or trunc:
            break

    if "ROUND_EVAL" in str(raw.get("phase", "")).upper():
        mask = get_action_mask(raw)
        fa, reasoning, _method, _params, was_fallback = decide_fn(raw, mask, [])
        assert int(fa.action_type) == int(ActionType.CashOut)
        assert reasoning == "always-cash-out"
        assert was_fallback is False


def test_random_legal_can_skip_a_blind():
    """The floor baseline's published description is a claim about its action set.

    ``random_decider`` was always written to sample whatever the mask offers; under
    v1 it never saw a blind-select mask, so the description was false in 241 of 241
    artifacts. This asserts the capability, not a particular sampled outcome.
    """
    import random as _random

    from jackdaw.env import ActionType, BalatroEnvironment, DirectAdapter

    from src.playground.seeds import load_battery
    from src.selfplay.runner import random_decider

    env = BalatroEnvironment(adapter_factory=DirectAdapter)
    _obs, mask, info = env.reset(seed=load_battery("train")[0])
    raw = info["raw_state"]
    assert mask.type_mask[int(ActionType.SkipBlind)]

    # Uniform over legal types, so a skip must appear within a modest number of draws.
    chosen = {
        int(random_decider(_random.Random(i))(raw, mask, [])[0].action_type) for i in range(50)
    }
    assert int(ActionType.SkipBlind) in chosen, chosen
    assert int(ActionType.SelectBlind) in chosen, chosen


def test_score_budget_override_records_the_budget_that_ran():
    """A swept run must not claim the v1 cap.

    ``docs/known-limits.md`` quantifies what the 300-subset scan cap costs, and
    ``--score-budget`` is how a reader reproduces that. The override is only
    honest if the identity block names the budget actually used -- an artifact
    reading ``score_budget=300`` while 8000 produced it would misattribute the
    result to the frozen v1 configuration.
    """
    from src.bench import agents

    for name in ("random-shop", "greedy-shop"):
        spec = agents.get(name)
        swept = agents.with_score_budget(spec, 8000)
        assert swept.identity()["tactical"] == "GreedyTactical(score_budget=8000)"
        assert swept.name == spec.name, "the shop policy is unchanged; only the cap moves"
        # The registry copy stays at the frozen cap.
        assert agents.get(name).identity()["tactical"] == "GreedyTactical(score_budget=300)"


def test_score_budget_override_refuses_agents_it_cannot_reach():
    """Silently ignoring the flag would let a sweep report a budget it never applied.

    ``random-legal`` has no tactical layer, and a submitted agent constructs its
    own, so the flag genuinely cannot reach either. That must be an error, not a
    no-op.
    """
    import pytest

    from src.bench import agents

    with pytest.raises(ValueError, match="no tunable tactical layer"):
        agents.with_score_budget(agents.get("random-legal"), 8000)

    with pytest.raises(ValueError, match="must be >= 1"):
        agents.with_score_budget(agents.get("greedy-shop"), 0)


def test_a_swept_budget_is_not_stamped_as_a_v1_result():
    """The sweep protocol must be distinct from the frozen benchmark protocol."""
    from src.bench import provenance

    assert provenance.TACTICAL_PROTOCOL != provenance.PROTOCOL
    # The frozen protocol itself is pinned by `test_protocol_is_v2`; what matters
    # here is that a sweep can never be stamped with it, whatever its version.
    assert provenance.PROTOCOL == "jackhammer/v2"
