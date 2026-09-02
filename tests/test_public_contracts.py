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
