"""Seed-battery tests: format/disjointness/loader contract + a real-Jackdaw
determinism sanity check on bank seeds.

The format tests guard the frozen battery contract. The
determinism test is the grounding payoff: it proves our generated seeds are valid,
loadable, and reproducible on the actual simulator (same seed -> same scrubbed
state fingerprint across two fresh resets).
"""

from __future__ import annotations

import json

import pytest
from jackdaw.env import BalatroEnvironment, DirectAdapter

from src.playground.seeds import (
    ALPHABET,
    SEED_LENGTH,
    _read_battery,
    load_battery,
)
from src.selfplay.determinism import fingerprint


def _good_battery() -> dict:
    return {
        "version": 1,
        "generator_seed": 20260630,
        "alphabet": ALPHABET,
        "seed_length": SEED_LENGTH,
        "created": "2026-06-30",
        "n_train": 2,
        "n_val": 1,
        "train": ["1234ABCD", "WXYZ5678"],
        "val": ["MNPQ9JKL"],
    }


# --- loader / split contract ------------------------------------------------


def test_split_sizes_and_all_is_concatenation():
    train = load_battery("train")
    val = load_battery("val")
    alls = load_battery("all")
    assert len(train) == 240
    assert len(val) == 60
    assert len(alls) == 300
    assert alls == train + val  # 'all' == train ++ val, in order


def test_default_split_is_train():
    assert load_battery() == load_battery("train")


def test_unknown_split_raises():
    with pytest.raises(ValueError, match="split must be one of"):
        load_battery("test")


def test_loader_returns_fresh_list():
    a = load_battery("train")
    a.append("XXXXXXXX")
    assert len(load_battery("train")) == 240  # mutation didn't leak into the bank


# --- format / disjointness of the committed bank ----------------------------


def test_train_val_disjoint_and_dupfree():
    train = load_battery("train")
    val = load_battery("val")
    assert len(set(train)) == len(train), "train has duplicates"
    assert len(set(val)) == len(val), "val has duplicates"
    assert not (set(train) & set(val)), "train/val overlap"


def test_every_seed_matches_balatro_format():
    alphabet_set = set(ALPHABET)
    for s in load_battery("all"):
        assert len(s) == SEED_LENGTH
        assert set(s) <= alphabet_set
        assert "0" not in s and "O" not in s  # the two excluded glyphs
        assert s == s.upper()


def test_alphabet_is_34_chars_no_zero_no_o():
    assert len(ALPHABET) == 34
    assert "0" not in ALPHABET
    assert "O" not in ALPHABET


# --- validator catches corruption ------------------------------------------


def _write(tmp_path, data):
    p = tmp_path / "battery.json"
    p.write_text(json.dumps(data))
    return p


def test_validator_accepts_good_file(tmp_path):
    assert _read_battery(_write(tmp_path, _good_battery()))["n_train"] == 2


def test_validator_rejects_overlap(tmp_path):
    d = _good_battery()
    d["val"] = ["1234ABCD"]  # collides with train[0]
    with pytest.raises(ValueError, match="disjoint"):
        _read_battery(_write(tmp_path, d))


def test_validator_rejects_bad_char(tmp_path):
    d = _good_battery()
    d["train"] = ["1234ABC0", "WXYZ5678"]  # contains forbidden '0'
    with pytest.raises(ValueError, match="outside alphabet"):
        _read_battery(_write(tmp_path, d))


def test_validator_rejects_bad_length(tmp_path):
    d = _good_battery()
    d["train"] = ["1234ABC", "WXYZ5678"]  # 7 chars
    with pytest.raises(ValueError, match="length"):
        _read_battery(_write(tmp_path, d))


def test_validator_rejects_count_mismatch(tmp_path):
    d = _good_battery()
    d["n_train"] = 99
    with pytest.raises(ValueError, match="n_train"):
        _read_battery(_write(tmp_path, d))


def test_validator_rejects_missing_key(tmp_path):
    d = _good_battery()
    del d["train"]
    with pytest.raises(ValueError, match="missing required key"):
        _read_battery(_write(tmp_path, d))


# --- the grounding payoff: bank seeds are valid + deterministic on Jackdaw ---


def test_bank_seeds_are_deterministic_on_jackdaw():
    seeds = load_battery("all")[:3]
    for s in seeds:
        a = BalatroEnvironment(adapter_factory=DirectAdapter)
        _o, _m, ia = a.reset(seed=s)
        b = BalatroEnvironment(adapter_factory=DirectAdapter)
        _o, _m, ib = b.reset(seed=s)
        assert fingerprint(ia["raw_state"]) == fingerprint(ib["raw_state"]), (
            f"seed {s!r}: two fresh resets produced different scrubbed states"
        )
