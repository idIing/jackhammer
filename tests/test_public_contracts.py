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
    assert provenance.PROTOCOL == "jackhammer/v1"
