"""Contract tests for the benchmark substrate (``jackhammer.bench``).

These guard the properties that make a published number trustworthy: that it is
pinned to an engine, that two numbers are only ever paired when they are
comparable, and that an agent claiming determinism actually replays.
"""

from __future__ import annotations

import json

import pytest

from jackhammer.bench import agents as agent_registry
from jackhammer.bench import artifact, provenance
from jackhammer.bench.datasets import DATASET_PROTOCOL
from jackhammer.playground.seeds import BATTERY_PATH


# --------------------------------------------------------------------- fixtures
def _stamp(**kw):
    base = dict(battery_path=BATTERY_PATH, split="train", n_seeds=240)
    base.update(kw)
    return provenance.stamp(**base)


def _runs(seed: str, ante: int) -> list[dict]:
    """Minimal recorder-shaped run objects; only the fields metrics reads."""
    return [
        {
            "meta": {"seed": seed, "config_label": "t", "slot1": "s1", "slot2": "s2"},
            "summary": {"highest_ante": ante, "won": False},
            "events": [],
        }
    ]


# ------------------------------------------------------------------- provenance
def test_stamp_pins_engine_battery_and_kit():
    s = _stamp()
    assert s["protocol"] == provenance.PROTOCOL
    assert s["battery"]["digest"].startswith("sha256:")
    assert s["battery"]["split"] == "train"
    assert s["battery"]["n_seeds"] == 240
    # The engine commit is the only field with resolving power; dist_version has
    # read 0.1.0 across every engine change the fork has ever made.
    assert len(s["engine"]["commit"]) == 40
    assert s["engine"]["source"] in {"checkout", "installed-vcs"}
    for key in ("python", "platform", "implementation"):
        assert s["runtime"][key]


def test_stamp_is_json_serialisable():
    json.dumps(_stamp())


def test_digest_tracks_file_contents(tmp_path):
    a = tmp_path / "a.json"
    a.write_text('{"version": 1}')
    first = provenance.digest_file(a)
    assert first == provenance.digest_file(a)
    a.write_text('{"version": 2}')
    assert provenance.digest_file(a) != first


def test_attributable_requires_known_and_clean_engine():
    s = _stamp()
    s["engine"] = {"commit": "a" * 40, "dirty": False}
    assert provenance.is_attributable(s)
    s["engine"] = {"commit": "a" * 40, "dirty": True}
    assert not provenance.is_attributable(s)
    # Installed-from-wheel: no checkout, so no pin, so not publishable.
    s["engine"] = {"commit": None, "dirty": None}
    assert not provenance.is_attributable(s)


def test_missing_checkout_degrades_instead_of_raising(tmp_path):
    assert provenance._pin(tmp_path / "nope") == {"commit": None, "dirty": None}


# ---------------------------------------------------------------------- agents
def test_builtin_baselines_are_registered():
    assert set(agent_registry.names()) == {"random-legal", "random-shop", "greedy-shop"}


def test_baselines_are_reproducible_by_contract():
    """A registered agent must seed any RNG from the run seed.

    ``random-shop`` is stochastic but declares ``deterministic=True``; that is
    only honest because its RNG is seeded from the game seed.
    """
    for spec in agent_registry.all_specs():
        assert spec.deterministic, f"{spec.name} publishes non-replayable numbers"


def test_unknown_agent_names_the_alternatives():
    with pytest.raises(KeyError, match="greedy-shop"):
        agent_registry.get("no-such-agent")


def test_duplicate_registration_is_rejected():
    spec = agent_registry.AgentSpec(
        name="dup-probe", description="d", make_decider=lambda env, seed: None
    )
    agent_registry.register(spec)
    try:
        with pytest.raises(ValueError, match="already registered"):
            agent_registry.register(spec)
        agent_registry.register(spec, replace=True)  # explicit override is allowed
    finally:
        agent_registry._REGISTRY.pop("dup-probe", None)


def test_agent_identity_is_the_published_surface():
    ident = agent_registry.get("greedy-shop").identity()
    assert ident["name"] == "greedy-shop"
    assert set(ident) == {
        "name",
        "description",
        "slot1",
        "slot2",
        "tactical",
        "deterministic",
    }
    # The shared in-blind policy is part of the published identity: it decides
    # every hand this agent plays, so a reader must be able to see it.
    assert ident["tactical"] == "GreedyTactical(score_budget=300)"


# -------------------------------------------------------------------- artifact
def test_result_round_trips_through_disk(tmp_path):
    spec = agent_registry.get("greedy-shop")
    res = artifact.build_result(
        spec=spec, runs=_runs("PVRQ4K5A", 3), provenance=_stamp(), runs_path="x.jsonl"
    )
    assert res["schema"] == artifact.RESULT_SCHEMA
    assert res["agent"]["name"] == "greedy-shop"
    path = artifact.write(tmp_path / "r.json", res)
    assert artifact.read(path)["summary"]["n_runs"] == 1


def test_validate_rejects_unknown_schema_and_missing_keys():
    with pytest.raises(ValueError, match="unknown artifact schema"):
        artifact.validate({"schema": "nope/v9"})
    with pytest.raises(ValueError, match="missing required key"):
        artifact.validate({"schema": artifact.RESULT_SCHEMA, "created": "t"})


def test_validate_requires_a_battery_digest():
    spec = agent_registry.get("greedy-shop")
    res = artifact.build_result(spec=spec, runs=_runs("PVRQ4K5A", 3), provenance=_stamp())
    del res["provenance"]["battery"]["digest"]
    with pytest.raises(ValueError, match="battery missing key"):
        artifact.validate(res)


def _result(agent: str, seed: str, ante: int, **stamp_kw):
    return artifact.build_result(
        spec=agent_registry.get(agent),
        runs=_runs(seed, ante),
        provenance=_stamp(**stamp_kw),
    )


def test_comparison_pairs_on_seed():
    a = _result("random-shop", "PVRQ4K5A", 1)
    b = _result("greedy-shop", "PVRQ4K5A", 4)
    comp = artifact.build_comparison(
        result_a=a, result_b=b, runs_a=_runs("PVRQ4K5A", 1), runs_b=_runs("PVRQ4K5A", 4)
    )
    assert comp["comparison"]["n_paired"] == 1
    assert comp["comparison"]["depth_delta"]["mean_delta"] == 3.0
    artifact.validate(comp)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda p: p["engine"].__setitem__("commit", "b" * 40), "different engine"),
        (lambda p: p["battery"].__setitem__("digest", "sha256:beef"), "different seed"),
        (lambda p: p["battery"].__setitem__("split", "val"), "different splits"),
    ],
)
def test_comparison_refuses_incomparable_arms(mutate, match):
    """An uncontrolled comparison must fail loudly rather than serialize.

    Pairing across engine commits is the failure that makes results
    unattributable after the fact; the artifact layer refuses to express it.
    """
    a = _result("random-shop", "PVRQ4K5A", 1)
    b = _result("greedy-shop", "PVRQ4K5A", 4)
    mutate(b["provenance"])
    with pytest.raises(ValueError, match=match):
        artifact.build_comparison(
            result_a=a,
            result_b=b,
            runs_a=_runs("PVRQ4K5A", 1),
            runs_b=_runs("PVRQ4K5A", 4),
        )


def test_comparison_refuses_different_protocols():
    a = _result("random-shop", "PVRQ4K5A", 1)
    b = _result("greedy-shop", "PVRQ4K5A", 4, protocol=DATASET_PROTOCOL)
    with pytest.raises(ValueError, match="different protocols"):
        artifact.build_comparison(
            result_a=a,
            result_b=b,
            runs_a=_runs("PVRQ4K5A", 1),
            runs_b=_runs("PVRQ4K5A", 4),
        )
