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
    from jackhammer.bench.datasets import load_dataset

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
    from jackhammer.bench import agents

    for name in ("random-shop", "greedy-shop"):
        identity = agents.get(name).identity()
        assert identity["tactical"] == "GreedyTactical(score_budget=300)", identity

    assert agents.get("random-legal").identity()["tactical"] == ""


def test_protocol_is_v2():
    """v1 numbers were produced by agents that could not skip a blind.

    The stamp is what tells a reader which apparatus produced a number. Pinning it
    here means demoting the protocol has to be deliberate.
    """
    from jackhammer.bench.provenance import PROTOCOL

    assert PROTOCOL == "jackhammer/v2"


def test_episode_loop_plays_no_phase_for_the_agent():
    """The runner must not decide anything. Protocol v2's whole content.

    Guarding the *absence* of a policy is awkward, so this asserts the observable
    consequence: a decider that refuses to act is asked about every phase the engine
    offers an action in, including the two the v1 loop auto-played. If someone
    reintroduces an auto-step, BLIND_SELECT or ROUND_EVAL stops appearing here.
    """
    from jackdaw.env import BalatroEnvironment, DirectAdapter

    from jackhammer.playground.seeds import load_battery
    from jackhammer.selfplay.runner import play_episode

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

    from jackhammer.playground.harness import GreedyShop, GreedyTactical, MarginValue, build_decider
    from jackhammer.playground.seeds import load_battery

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

    from jackhammer.playground.seeds import load_battery
    from jackhammer.selfplay.runner import random_decider

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
    from jackhammer.bench import agents

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

    from jackhammer.bench import agents

    with pytest.raises(ValueError, match="no tunable tactical layer"):
        agents.with_score_budget(agents.get("random-legal"), 8000)

    with pytest.raises(ValueError, match="must be >= 1"):
        agents.with_score_budget(agents.get("greedy-shop"), 0)


def test_a_swept_budget_is_not_stamped_as_a_v1_result():
    """The sweep protocol must be distinct from the frozen benchmark protocol."""
    from jackhammer.bench import provenance

    assert provenance.TACTICAL_PROTOCOL != provenance.PROTOCOL
    # The frozen protocol itself is pinned by `test_protocol_is_v2`; what matters
    # here is that a sweep can never be stamped with it, whatever its version.
    assert provenance.PROTOCOL == "jackhammer/v2"


def test_the_engine_dependency_pins_the_protocol_commit():
    """A bare ``jackdaw`` requirement installs somebody else's package.

    ``jackdaw`` on PyPI is an unrelated Active Directory tool. Before this was
    pinned by direct reference, the built wheel's ``Requires-Dist: jackdaw``
    resolved to it, and a ``pip install`` of the kit failed at
    ``import jackdaw.env`` -- while ``uv sync`` in a checkout stayed green,
    because the git pin lived in ``[tool.uv.sources]`` and never shipped.

    So this asserts the pin travels in installable metadata, and that it names
    the same commit protocol v2 section 1 does. The two drifting apart would
    silently evaluate agents on an engine the protocol does not describe.
    """
    import tomllib

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = pyproject["project"]["dependencies"]
    jackdaw = [d for d in deps if d.split()[0].split("@")[0].strip() == "jackdaw"]
    assert len(jackdaw) == 1, deps
    assert "git+https://github.com/idIing/jackdaw-balatro.git@" in jackdaw[0], jackdaw[0]

    commit = jackdaw[0].rsplit("@", 1)[1].strip()
    assert len(commit) == 40, f"pin the full commit, not {commit!r}"
    protocol = (ROOT / "docs" / "protocol-v2.md").read_text()
    assert commit in protocol, f"{commit} is not the engine protocol v2 names"


def test_installed_kit_stamps_its_release_version():
    """``kit.version`` must survive into a wheel; ``kit.commit`` need not.

    A consumer who installed the kit has no checkout, so a git-derived version
    would be None for exactly the users a pip-installable release is for. The
    version therefore comes from the installed distribution metadata.
    """
    from importlib import metadata

    from jackhammer.bench import provenance

    kit = provenance.kit_pin()
    assert set(kit) == {"version", "commit", "dirty"}, kit
    assert kit["version"] == metadata.version(provenance.DIST_NAME)
    assert isinstance(kit["version"], str) and kit["version"], kit


def test_the_frozen_battery_is_found_and_named_the_same_way_either_way():
    """The battery must load from a wheel, and stamp the path protocol v2 names.

    It lives at the repo root in a checkout and is copied into the package by the
    wheel build. Both must resolve, and both must stamp the same string -- an
    artifact should not record how the kit was installed.
    """
    from jackhammer.bench import provenance
    from jackhammer.playground.seeds import BATTERY_PATH, load_battery

    assert BATTERY_PATH.exists(), BATTERY_PATH
    assert len(load_battery("train")) == 240

    stamped = provenance.stamp(battery_path=BATTERY_PATH, split="train", n_seeds=240)
    assert stamped["battery"]["path"] == "config/seed_battery_v1.json"
    assert stamped["battery"]["path"] in (ROOT / "docs" / "protocol-v2.md").read_text()

    packaged = provenance.PACKAGE_ROOT / "config" / "seed_battery_v1.json"
    assert provenance._battery_ref(packaged) == "config/seed_battery_v1.json"


def test_known_limits_carries_the_generated_repertoire_block():
    """The published description of a baseline must be generated, not retyped.

    ``docs/known-limits.md`` shipped "takes the first pack card" for `greedy-shop`,
    describing a branch that cannot execute — the policy never buys a booster, so
    ``PACK_OPENING`` is never entered. It was written by reading the code, which is
    the one way that claim could not be checked. Generating the table from the
    registry means the file cannot describe an agent the registry does not.
    """
    from jackhammer.bench.repertoire import docs_block

    text = (ROOT / "docs" / "known-limits.md").read_text()
    start = text.index("<!-- BEGIN declared-repertoire -->") + len(
        "<!-- BEGIN declared-repertoire -->"
    )
    end = text.index("<!-- END declared-repertoire -->")
    assert text[start:end].strip() == docs_block(), (
        "docs/known-limits.md is stale; replace the block between the markers with:\n\n"
        + docs_block()
    )


def test_every_built_in_baseline_declares_a_repertoire():
    """A published baseline with no declaration is a claim nothing checks.

    Submitted agents may decline to declare — the gate is for the numbers this
    repository publishes.
    """
    from jackhammer.bench import agents

    for name in ("random-legal", "random-shop", "greedy-shop"):
        declared = agents.get(name).declared_actions
        assert declared, name
        assert agents.get(name).identity()["declared_actions"] == list(declared)


def test_greedy_shop_declares_no_pack_action():
    """The corrected limit, pinned as a test rather than as prose.

    ``GreedyShop._joker_buys`` filters shop candidates to ``ability.set == "Joker"``,
    so no booster is ever bought and the pack-choice branch is unreachable. If the
    shop policy ever gains packs, this fails and the description gets rewritten with
    it, rather than one drifting from the other.
    """
    from jackhammer.bench import agents

    declared = agents.get("greedy-shop").declared_actions
    assert "OpenBooster" not in declared
    assert "PickPackCard" not in declared
    assert set(declared) == {
        "PlayHand",
        "Discard",
        "SelectBlind",
        "CashOut",
        "NextRound",
        "BuyCard",
    }


def test_a_declaration_cannot_name_a_non_action():
    """A typo would read as a permanent mismatch against a correct agent."""
    import pytest

    from jackhammer.bench.agents import AgentSpec

    with pytest.raises(ValueError, match="not action type name"):
        AgentSpec(
            name="typo",
            description="",
            make_decider=lambda e, s: None,
            declared_actions=("PlayHand", "BuyJoker"),
        )


# The three baselines' policies, as source-level components. `greedy-shop` and
# `random-shop` are compositions, so their ceiling is the union of the parts.
_POLICY_SOURCES = {
    "greedy-shop": (
        "playground/harness.py",
        ["GreedyTactical", "GreedyShop", "MarginValue", "build_decider"],
    ),
    "random-shop": (
        "playground/harness.py",
        ["GreedyTactical", "RandomShop", "MarginValue", "build_decider"],
    ),
    "random-legal": ("selfplay/runner.py", ["random_decider"]),
}
_SRC = ROOT / "src" / "jackhammer"


def test_no_baseline_declares_an_action_its_code_cannot_construct():
    """The cheap half of the gate, and the only half that fits in CI.

    Checking a declaration against a *run* needs the whole battery, which is minutes.
    Checking it against the *source* is milliseconds and no games, and it catches the
    error at the moment someone writes it rather than the next time someone publishes a
    number. A policy that builds its action from a computed value can construct anything,
    so the check is vacuous there and says so instead of passing quietly.
    """
    from jackhammer.bench import agents
    from jackhammer.bench.repertoire import scan_union

    for name, (rel, parts) in _POLICY_SOURCES.items():
        scan = scan_union(_SRC / rel, parts)
        declared = set(agents.get(name).declared_actions)
        if scan.dynamic:
            continue  # covered by the dynamic-arm test below
        impossible = declared - scan.actions
        assert not impossible, (
            f"{name} declares {sorted(impossible)}, which its source never constructs"
        )


def test_greedy_shops_only_unreachable_branch_is_the_pack_choice():
    """The defect that started this, pinned where it can be caught without running.

    The source can construct seven actions; the battery records six. The odd one out is
    `PickPackCard`: `decide_shop` handles `PACK_OPENING` correctly, and `_joker_buys`
    filters shop candidates to Jokers so no booster is ever bought, so that branch cannot
    execute. Reachable-in-code minus declared is therefore exactly the dead branch, and
    this test fails if that set changes in either direction — a new dead branch, or the
    pack branch becoming live.
    """
    from jackhammer.bench import agents
    from jackhammer.bench.repertoire import scan_union

    rel, parts = _POLICY_SOURCES["greedy-shop"]
    scan = scan_union(_SRC / rel, parts)
    declared = set(agents.get("greedy-shop").declared_actions)

    assert not scan.dynamic, "greedy-shop names every action it builds; keep it that way"
    assert scan.actions - declared == {"PickPackCard"}, sorted(scan.actions - declared)
    assert declared - scan.actions == set()


def test_the_random_arms_are_vacuous_for_a_stated_reason():
    """Both sample from the legal-action mask, so their source ceiling is everything.

    Pinned rather than left implicit: if either is ever rewritten to name the actions it
    builds, this fails, and the static check above becomes real for it and should be
    tightened rather than silently staying vacuous.
    """
    from jackhammer.bench.repertoire import scan_union

    for name in ("random-shop", "random-legal"):
        rel, parts = _POLICY_SOURCES[name]
        assert scan_union(_SRC / rel, parts).dynamic, name


def test_a_fallback_decision_is_not_blamed_on_the_agent():
    """`was_fallback` means the harness substituted, so it is not the agent's claim.

    It stays in the histogram, because it happened. It stays out of both differences,
    because an agent must not be accused of an action it did not choose, nor credited
    with one it did not choose. All three baselines fall back zero times, so this only
    matters the day one does.
    """
    from jackhammer.bench import repertoire

    runs = [
        {
            "meta": {"seed": "S"},
            "summary": {"highest_ante": 1, "won": False},
            "events": [
                {"step": 0, "action": "PlayHand", "was_fallback": False},
                {"step": 1, "action": "Reroll", "was_fallback": True},
            ],
        }
    ]
    rep = repertoire.audit(runs, declared=("PlayHand",))
    assert rep["observed"] == {"PlayHand": 1, "Reroll": 1}
    assert rep["n_fallback"] == 1
    assert rep["undeclared"] == []
    assert rep["unexercised"] == []
