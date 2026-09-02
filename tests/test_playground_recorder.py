"""CPU/GPU-free unit tests for the playground recorder extension.

Covers the `blinds[]` per-blind records and the `RunMeta` slot fields. Synthetic
prev/next `raw_state` dicts model only the fields the recorder reads; a final
test drives a couple of real short games to assert the `blinds` array is well-formed.

Key fact this exercises: the sim adapter is zero-copy, so the recorder only ever
sees the *post-step* state. Every blind field is therefore read from the
just-resolved `blind` object / `raw_state`; the only field the resolution
pollutes is `ante` (a boss clear advances it), which the recorder un-advances.

Run with: ``uv run python -m pytest tests/test_playground_recorder.py``
"""

import json
import os
import sys
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.selfplay.recorder import RunMeta, RunRecorder  # noqa: E402

_BLIND_KEYS = {
    "ante",
    "blind_type",
    "boss_key",
    "target",
    "realized_score",
    "margin",
    "cleared",
    "hands_used",
    "hands_left",
    "discards_left",
    "played_hand_histogram",
}


def _blind(chips, name, *, boss=False, key=None):
    return SimpleNamespace(chips=chips, name=name, boss=boss, key=key)


def _score(hand_type, total=0, chips=0, mult=0):
    return SimpleNamespace(hand_type=hand_type, total=total, chips=chips, mult=mult)


def _state(phase, blind, *, chips, ante, hands_left, discards_left, last_score):
    """Minimal post-step gs the recorder reads for a PlayHand resolution."""
    return {
        "phase": phase,
        "blind": blind,
        "chips": chips,
        "round_resets": {"ante": ante},
        "current_round": {"hands_left": hands_left, "discards_left": discards_left},
        "last_score_result": last_score,
        "jokers": [],
    }


def _play(card_target=(0, 1, 2, 3, 4)):
    return SimpleNamespace(action_type=0, card_target=card_target, entity_target=None)


def _new_recorder(tmp_path):
    rec = RunRecorder(output_path=str(tmp_path / "runs.jsonl"))
    rec.start_run(
        RunMeta(
            seed="S",
            seed_mode="set",
            temperature_regime="t",
            temperature_value=1.0,
        )
    )
    return rec


def test_cleared_blind_emits_once_with_margin_ge_1(tmp_path):
    rec = _new_recorder(tmp_path)
    # Zero-copy reality: the same post-step dict is both prev and next.
    st = _state(
        "round_eval",
        _blind(450, "Big Blind", boss=False, key="bl_big"),
        chips=500,
        ante=1,
        hands_left=2,
        discards_left=1,
        last_score=_score("Flush", total=500, chips=100, mult=5),
    )
    rec.record_step(st, st, _play())

    assert len(rec._blinds) == 1
    b = rec._blinds[0]
    assert b["cleared"] is True
    assert b["blind_type"] == "Big"
    assert b["boss_key"] is None  # not a boss
    assert b["target"] == 450
    assert b["realized_score"] == 500
    assert b["margin"] >= 1.0
    assert b["ante"] == 1  # non-boss clear: no ante advance
    assert b["hands_used"] == 1
    assert b["hands_left"] == 2
    assert b["discards_left"] == 1
    assert b["played_hand_histogram"] == {"Flush": 1}


def test_failed_blind_record_cleared_false_margin_lt_1(tmp_path):
    rec = _new_recorder(tmp_path)
    st = _state(
        "game_over",
        _blind(300, "Small Blind", boss=False, key="bl_small"),
        chips=116,
        ante=1,
        hands_left=0,
        discards_left=4,
        last_score=_score("Pair", total=40, chips=20, mult=2),
    )
    rec.record_step(st, st, _play(card_target=(0, 1)))

    assert len(rec._blinds) == 1
    b = rec._blinds[0]
    assert b["cleared"] is False
    assert b["margin"] < 1.0
    assert b["realized_score"] == 116
    assert b["target"] == 300
    assert b["hands_left"] == 0  # out of hands is the fail condition
    assert b["played_hand_histogram"] == {"Pair": 1}


def test_boss_clear_unadvances_ante_and_sets_boss_key(tmp_path):
    rec = _new_recorder(tmp_path)
    # A boss clear runs _advance_ante, so the post-step ante we observe is 3 for a
    # boss that belonged to ante 2. The recorder must record 2.
    st = _state(
        "round_eval",
        _blind(1200, "The Hook", boss=True, key="bl_hook"),
        chips=1530,
        ante=3,
        hands_left=1,
        discards_left=0,
        last_score=_score("Flush", total=1530, chips=200, mult=7),
    )
    rec.record_step(st, st, _play())

    b = rec._blinds[0]
    assert b["ante"] == 2  # 3 - 1 (un-advanced boss clear)
    assert b["blind_type"] == "Boss"
    assert b["boss_key"] == "bl_hook"
    assert b["cleared"] is True
    assert abs(b["margin"] - (1530 / 1200)) < 1e-9


def test_histogram_and_hands_used_reset_across_blinds(tmp_path):
    rec = _new_recorder(tmp_path)
    blind_a = _blind(300, "Small Blind")

    # Blind A, hand 1: a non-resolving Pair (stays SELECTING_HAND -> no record).
    st_a1 = _state(
        "selecting_hand",
        blind_a,
        chips=100,
        ante=1,
        hands_left=3,
        discards_left=4,
        last_score=_score("Pair", total=100),
    )
    rec.record_step(st_a1, st_a1, _play(card_target=(0, 1)))
    assert rec._blinds == []

    # Blind A, hand 2: a Flush that clears.
    st_a2 = _state(
        "round_eval",
        blind_a,
        chips=320,
        ante=1,
        hands_left=2,
        discards_left=4,
        last_score=_score("Flush", total=220),
    )
    rec.record_step(st_a2, st_a2, _play())
    assert len(rec._blinds) == 1
    assert rec._blinds[0]["played_hand_histogram"] == {"Pair": 1, "Flush": 1}
    assert rec._blinds[0]["hands_used"] == 2

    # Blind B: a single Two Pair that clears. Histogram must be ONLY this blind.
    st_b = _state(
        "round_eval",
        _blind(450, "Big Blind"),
        chips=500,
        ante=1,
        hands_left=3,
        discards_left=4,
        last_score=_score("Two Pair", total=500),
    )
    rec.record_step(st_b, st_b, _play())
    assert len(rec._blinds) == 2
    assert rec._blinds[1]["played_hand_histogram"] == {"Two Pair": 1}
    assert rec._blinds[1]["hands_used"] == 1


def test_slot_fields_round_trip_through_jsonl(tmp_path):
    out = tmp_path / "runs.jsonl"
    rec = RunRecorder(output_path=str(out))
    rec.start_run(
        RunMeta(
            seed="S",
            seed_mode="set",
            temperature_regime="t",
            temperature_value=1.0,
            slot1="GreedyShop",
            slot2="MarginValue",
            config_label="greedy+margin",
        )
    )
    rec.finish_run(
        highest_ante=1,
        won=False,
        episode_length=0,
        fallback_occurred=False,
        terminal_reason="terminated",
    )

    record = json.loads(out.read_text().strip().splitlines()[-1])
    assert record["meta"]["slot1"] == "GreedyShop"
    assert record["meta"]["slot2"] == "MarginValue"
    assert record["meta"]["config_label"] == "greedy+margin"
    assert record["blinds"] == []
    assert record["summary"]["n_blinds"] == 0


def test_defaults_keep_old_records_valid():
    # Pre-playground construction (no slot fields) must still default cleanly.
    m = RunMeta(seed="S", seed_mode="set", temperature_regime="t", temperature_value=1.0)
    assert m.slot1 == "" and m.slot2 == "" and m.config_label == ""


def test_real_short_games_produce_well_formed_blinds(tmp_path):
    # Drive a couple of real (LLM-free) games; assert the blinds array is sound.
    import random as _random

    from jackdaw.env import BalatroEnvironment, DirectAdapter

    from src.selfplay.runner import play_episode, random_decider

    out = tmp_path / "runs.jsonl"
    seen_any = False
    for i in range(3):
        seed = f"GREED{i:03d}"
        env = BalatroEnvironment(adapter_factory=DirectAdapter)
        rec = RunRecorder(output_path=str(out))
        meta = RunMeta(
            seed=seed,
            seed_mode="set",
            temperature_regime="t",
            temperature_value=1.0,
            backend="random",
            slot1="RandomShop",
            slot2="MarginValue",
            config_label="random",
        )
        play_episode(env, seed, random_decider(_random.Random(i)), rec, meta, max_steps=200)

        record = json.loads(out.read_text().strip().splitlines()[-1])
        blinds = record["blinds"]
        assert record["summary"]["n_blinds"] == len(blinds)
        assert record["meta"]["slot1"] == "RandomShop"

        for b in blinds:
            assert _BLIND_KEYS.issubset(b.keys())
            assert b["blind_type"] in ("Small", "Big", "Boss")
            assert b["target"] > 0
            assert abs(b["margin"] - b["realized_score"] / b["target"]) < 1e-9
            # The per-blind histogram must account for exactly the hands played.
            assert sum(b["played_hand_histogram"].values()) == b["hands_used"]
            if b["blind_type"] == "Boss":
                assert b["boss_key"]  # bosses carry an engine key
            else:
                assert b["boss_key"] is None

        # A random agent reliably dies; a lost game ends on a failed blind.
        if blinds and not record["summary"]["won"]:
            assert blinds[-1]["cleared"] is False
            seen_any = True

    assert seen_any, "expected at least one game to record a failed final blind"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("run via: uv run python -m pytest tests/test_playground_recorder.py")
