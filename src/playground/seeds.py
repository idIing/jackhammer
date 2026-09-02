"""Seed battery loader — the fixed seed bank the playground benchmarks against.

The bank lives in ``config/seed_battery_v1.json`` (frozen, versioned, committed as
data). Seeds use Balatro's own seed format: 8 uppercase chars, each drawn from the
34-char alphabet ``123456789ABCDEFGHIJKLMNPQRSTUVWXYZ`` (digits 1-9 and letters
A-N + P-Z — **no ``0``, no ``O``**). They stay typeable into the live game, though
this repository ships no live-game cross-check.

The bank is split ``train`` (240) / ``val`` (60), disjoint and dup-free. The ``val``
slice is retired: it was consumed during development and is gated behind an
explicit CLI flag. v1 evaluations use ``train``.

Protocol: ``docs/protocol-v1.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

# Canonical seed format (single source of truth for the validation below).
ALPHABET = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"  # 34 chars: 1-9, A-N, P-Z (no 0, no O)
SEED_LENGTH = 8
_ALPHABET_SET = frozenset(ALPHABET)

# Repo-relative path to the committed bank. seeds.py is src/playground/seeds.py, so
# parents[2] is the repo root.
BATTERY_PATH = Path(__file__).resolve().parents[2] / "config" / "seed_battery_v1.json"

_VALID_SPLITS = ("train", "val", "all")


def _check_seeds(seeds: object, where: str) -> list[str]:
    """Validate a list of seed strings; return it unchanged or raise ValueError."""
    if not isinstance(seeds, list):
        raise ValueError(f"seed battery: '{where}' must be a list, got {type(seeds).__name__}")
    for s in seeds:
        if not isinstance(s, str):
            raise ValueError(f"seed battery: '{where}' contains a non-string seed {s!r}")
        if len(s) != SEED_LENGTH:
            raise ValueError(
                f"seed battery: '{where}' seed {s!r} has length {len(s)}, expected {SEED_LENGTH}"
            )
        bad = set(s) - _ALPHABET_SET
        if bad:
            raise ValueError(
                f"seed battery: '{where}' seed {s!r} has chars {sorted(bad)} "
                f"outside alphabet {ALPHABET!r}"
            )
    return seeds


def _read_battery(path: Path) -> dict:
    """Read + fully validate the battery JSON at *path*. Raise ValueError on any defect."""
    if not path.exists():
        raise FileNotFoundError(
            f"seed battery not found at {path}; restore config/seed_battery_v1.json "
            f"from the repository"
        )
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"seed battery at {path} is not valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"seed battery at {path} must be a JSON object")

    for key in (
        "version",
        "alphabet",
        "seed_length",
        "n_train",
        "n_val",
        "train",
        "val",
    ):
        if key not in data:
            raise ValueError(f"seed battery at {path} is missing required key {key!r}")

    if data["alphabet"] != ALPHABET:
        raise ValueError(f"seed battery alphabet {data['alphabet']!r} != expected {ALPHABET!r}")
    if data["seed_length"] != SEED_LENGTH:
        raise ValueError(
            f"seed battery seed_length {data['seed_length']!r} != expected {SEED_LENGTH}"
        )

    train = _check_seeds(data["train"], "train")
    val = _check_seeds(data["val"], "val")

    if len(train) != data["n_train"]:
        raise ValueError(f"seed battery: n_train={data['n_train']} but len(train)={len(train)}")
    if len(val) != data["n_val"]:
        raise ValueError(f"seed battery: n_val={data['n_val']} but len(val)={len(val)}")

    train_set, val_set = set(train), set(val)
    if len(train_set) != len(train):
        raise ValueError("seed battery: 'train' contains duplicate seeds")
    if len(val_set) != len(val):
        raise ValueError("seed battery: 'val' contains duplicate seeds")
    overlap = train_set & val_set
    if overlap:
        raise ValueError(
            f"seed battery: 'train' and 'val' overlap on {sorted(overlap)[:5]} "
            f"({len(overlap)} seed(s)); splits must be disjoint"
        )

    return data


def load_battery(split: str = "train") -> list[str]:
    """Load seeds from the committed bank.

    Args:
        split: one of ``"train"`` (240 seeds), ``"val"`` (60 seeds, final reports
            only), or ``"all"`` (train + val, 300 seeds).

    Returns:
        A fresh list of 8-char seed strings.

    Raises:
        ValueError: if *split* is unknown or the on-disk bank fails validation.
        FileNotFoundError: if the bank file is missing.
    """
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")

    data = _read_battery(BATTERY_PATH)
    if split == "train":
        return list(data["train"])
    if split == "val":
        return list(data["val"])
    return list(data["train"]) + list(data["val"])
