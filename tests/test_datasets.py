from __future__ import annotations

import json

import pytest

from src.bench.datasets import DATASET_SCHEMA, load_dataset


def _manifest(**overrides):
    obj = {
        "schema": DATASET_SCHEMA,
        "name": "tiny",
        "version": 1,
        "description": "test dataset",
        "splits": {"sample": ["PVRQ4K5A", "4NNGD2DN"]},
    }
    obj.update(overrides)
    return obj


def test_dataset_loads_named_split(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(_manifest()))
    dataset = load_dataset(path)
    assert dataset.name == "tiny"
    assert dataset.version == "1"
    assert dataset.seeds("sample") == ["PVRQ4K5A", "4NNGD2DN"]
    assert dataset.provenance()["dataset"]["scope"] == "diagnostic"


def test_dataset_rejects_bad_schema(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(_manifest(schema="something-else")))
    with pytest.raises(ValueError, match="dataset schema"):
        load_dataset(path)


def test_dataset_rejects_duplicate_seeds_within_a_split(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(_manifest(splits={"a": ["PVRQ4K5A", "PVRQ4K5A"]})))
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(path)


def test_dataset_allows_declared_nested_coverage_splits(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps(
            _manifest(
                splits={"n1": ["PVRQ4K5A"], "n2": ["PVRQ4K5A", "4NNGD2DN"]},
                metadata={"relationship": "nested prefixes"},
            )
        )
    )
    dataset = load_dataset(path)
    assert dataset.seeds("n2") == ["PVRQ4K5A", "4NNGD2DN"]
    assert dataset.metadata == {"relationship": "nested prefixes"}


def test_dataset_names_available_splits(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(_manifest()))
    with pytest.raises(ValueError, match="available: sample"):
        load_dataset(path).seeds("missing")
