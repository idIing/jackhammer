"""Versioned seed datasets for diagnostic and exploratory evaluations.

Protocol v1 always uses ``config/seed_battery_v1.json``.  A dataset loaded by
this module is deliberately stamped under a different protocol identifier so a
coverage study cannot be mistaken for a v1 benchmark result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.playground.seeds import _check_seeds

DATASET_SCHEMA = "jackhammer.seed-dataset/v1"
DATASET_PROTOCOL = "jackhammer/dataset-eval/v1"


@dataclass(frozen=True)
class SeedDataset:
    path: Path
    name: str
    version: str
    description: str
    splits: dict[str, list[str]]
    metadata: dict[str, Any]

    def seeds(self, split: str) -> list[str]:
        try:
            return list(self.splits[split])
        except KeyError:
            available = ", ".join(sorted(self.splits))
            raise ValueError(
                f"dataset {self.name!r} has no split {split!r}; available: {available}"
            ) from None

    def provenance(self) -> dict[str, Any]:
        return {
            "dataset": {
                "schema": DATASET_SCHEMA,
                "name": self.name,
                "version": self.version,
                "description": self.description,
                "scope": "diagnostic",
                "metadata": self.metadata,
            }
        }


def load_dataset(path: str | Path) -> SeedDataset:
    """Load and validate a ``jackhammer.seed-dataset/v1`` JSON manifest."""
    dataset_path = Path(path).resolve()
    try:
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"seed dataset not found at {dataset_path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"seed dataset at {dataset_path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"seed dataset at {dataset_path} must be a JSON object")
    if data.get("schema") != DATASET_SCHEMA:
        raise ValueError(f"dataset schema must be {DATASET_SCHEMA!r}, got {data.get('schema')!r}")
    for key in ("name", "version", "description", "splits"):
        if key not in data:
            raise ValueError(f"seed dataset is missing required key {key!r}")
    if not isinstance(data["name"], str) or not data["name"].strip():
        raise ValueError("seed dataset 'name' must be a non-empty string")
    if not isinstance(data["description"], str):
        raise ValueError("seed dataset 'description' must be a string")
    if not isinstance(data["version"], (str, int)):
        raise ValueError("seed dataset 'version' must be a string or integer")
    if not isinstance(data["splits"], dict) or not data["splits"]:
        raise ValueError("seed dataset 'splits' must be a non-empty object")

    splits: dict[str, list[str]] = {}
    for split, raw_seeds in data["splits"].items():
        if not isinstance(split, str) or not split:
            raise ValueError("dataset split names must be non-empty strings")
        seeds = list(_check_seeds(raw_seeds, split))
        if not seeds:
            raise ValueError(f"seed dataset split {split!r} must not be empty")
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"seed dataset split {split!r} contains duplicate seeds")
        splits[split] = seeds

    metadata = data.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("seed dataset 'metadata' must be an object when present")

    return SeedDataset(
        path=dataset_path,
        name=data["name"].strip(),
        version=str(data["version"]),
        description=data["description"],
        splits=splits,
        metadata=metadata,
    )
