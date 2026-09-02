"""Tests for the two-slot playground harness (``src/playground/harness.py``).

Covers the load-bearing invariants:

* **Non-mutation / zero-copy**: every value estimator and the shop policy's
  value-ranking leave the live env byte-identical (checked via the determinism
  fingerprint) across a real game.
* **Legality**: every action the composed ``decide_fn`` emits is legal.
* **Slot1 -> Slot2 path**: ``GreedyShop`` actually ranks ≥2 candidate buys by the
  value estimator (proved with a fake env + counting value).
* **Determinism**: a battery on the same seed with deterministic slots reproduces
  the same ``highest_ante``.
* **Smoke battery**: ``RandomShop`` vs ``GreedyShop`` (both on
  ``GreedyTactical`` + ``MarginValue``) produce well-formed ``blinds`` records, and
  greedy goes at least as deep (the known-difference sanity check).

Run with: ``uv run --no-sync python -m pytest tests/test_playground_harness.py -q``
"""

import copy
import json
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from jackdaw.env import (  # noqa: E402
    ActionType,
    BalatroEnvironment,
    DirectAdapter,
    FactoredAction,
    get_action_mask,
)

from src.playground.harness import (  # noqa: E402
    GreedyShop,
    GreedyTactical,
    MarginValue,
    RandomShop,
    RolloutValue,
    ShopPolicy,
    Tactical,
    ValueEstimator,
    build_decider,
    run_battery,
)
from src.playground.seeds import load_battery  # noqa: E402
from src.selfplay.decision import is_action_legal  # noqa: E402
from src.selfplay.determinism import fingerprint  # noqa: E402

_SELECT_BLIND = int(ActionType.SelectBlind)
_CASHOUT = int(ActionType.CashOut)
_NEXT_ROUND = int(ActionType.NextRound)
_BUY_CARD = int(ActionType.BuyCard)

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


# ---------------------------------------------------------------------------
# Protocols / names
# ---------------------------------------------------------------------------


def test_baselines_satisfy_protocols_and_carry_names():
    assert isinstance(GreedyTactical(), Tactical)
    assert isinstance(MarginValue(), ValueEstimator)
    assert isinstance(RolloutValue(), ValueEstimator)
    assert isinstance(RandomShop(), ShopPolicy)
    assert isinstance(GreedyShop(), ShopPolicy)
    assert GreedyTactical().name == "GreedyTactical"
    assert MarginValue().name == "MarginValue"
    assert RolloutValue().name == "RolloutValue"
    assert RandomShop().name == "RandomShop"
    assert GreedyShop().name == "GreedyShop"


# ---------------------------------------------------------------------------
# Slot1 -> Slot2 path (fake env + counting value)
# ---------------------------------------------------------------------------


class _FakeEnv:
    """Minimal env honoring get_state/load_state/step for shop value-ranking."""

    def __init__(self, raw):
        self._raw = raw
        self.steps: list = []

    def get_state(self):
        return {"raw": copy.deepcopy(self._raw)}

    def load_state(self, s):
        self._raw = copy.deepcopy(s["raw"])

    def step(self, fa):
        self.steps.append(fa)
        return None, False, False, None, {"raw_state": self._raw}


class _CountingValue:
    name = "counting"

    def __init__(self):
        self.calls = 0

    def value(self, env, raw_state):
        self.calls += 1
        return float(env.steps[-1].entity_target)  # prefers the higher index


def _joker(cost):
    return SimpleNamespace(ability={"set": "Joker"}, cost=cost, edition=None)


def test_greedyshop_ranks_candidate_buys_by_value():
    # Two affordable joker buys -> GreedyShop must consult Slot 2 and pick argmax.
    raw = {"phase": "shop", "shop_cards": [_joker(6), _joker(4)], "jokers": []}
    type_mask = np.zeros(21, dtype=bool)
    type_mask[_BUY_CARD] = True
    type_mask[_NEXT_ROUND] = True
    mask = SimpleNamespace(
        type_mask=type_mask,
        card_mask=np.zeros(0, dtype=bool),
        entity_masks={_BUY_CARD: np.array([True, True])},
        min_card_select=1,
        max_card_select=5,
    )
    env = _FakeEnv(raw)
    val = _CountingValue()
    fa = GreedyShop().decide_shop(env, raw, mask, val)

    assert fa.action_type == _BUY_CARD
    assert fa.entity_target == 1  # the higher-valued candidate
    assert val.calls >= 2  # both candidates were scored — the path is real


def test_greedyshop_single_candidate_skips_ranking():
    raw = {"phase": "shop", "shop_cards": [_joker(4)], "jokers": []}
    type_mask = np.zeros(21, dtype=bool)
    type_mask[_BUY_CARD] = True
    type_mask[_NEXT_ROUND] = True
    mask = SimpleNamespace(
        type_mask=type_mask,
        card_mask=np.zeros(0, dtype=bool),
        entity_masks={_BUY_CARD: np.array([True])},
        min_card_select=1,
        max_card_select=5,
    )
    env = _FakeEnv(raw)
    val = _CountingValue()
    fa = GreedyShop().decide_shop(env, raw, mask, val)
    assert fa.action_type == _BUY_CARD and fa.entity_target == 0
    assert val.calls == 0  # single candidate: no ranking needed


# ---------------------------------------------------------------------------
# Real-game invariants: non-mutation + legality of every emitted action
# ---------------------------------------------------------------------------


def _auto_action(raw, mask):
    phase_u = str(raw.get("phase", "")).upper()
    if "BLIND_SELECT" in phase_u and mask.type_mask[_SELECT_BLIND]:
        return FactoredAction(action_type=_SELECT_BLIND)
    if "ROUND_EVAL" in phase_u:
        if mask.type_mask[_CASHOUT]:
            return FactoredAction(action_type=_CASHOUT)
        if mask.type_mask[_NEXT_ROUND]:
            return FactoredAction(action_type=_NEXT_ROUND)
    return None


def test_real_game_non_mutation_and_legality():
    seed = load_battery("train")[0]
    # Invariants (non-mutation / legality) hold at any search depth; use a light
    # budget to keep the test cheap.
    tactical = GreedyTactical(score_budget=64)
    shop = GreedyShop()
    margin = MarginValue()
    rollout = RolloutValue(max_rollout_steps=20)
    env = BalatroEnvironment(adapter_factory=DirectAdapter)
    decide_fn = build_decider(env, tactical, shop, margin)
    env.reset(seed=seed)

    history: list = []
    done = False
    steps = 0
    rolled_once = False
    checked_shop = False
    while not done and steps < 60:
        raw = env._adapter.raw_state
        mask = get_action_mask(raw)
        steps += 1

        auto = _auto_action(raw, mask)
        if auto is not None:
            _o, term, trunc, _m, _i = env.step(auto)
            done = term or trunc
            continue

        phase_u = str(raw.get("phase", "")).upper()

        # --- non-mutation probes on the LIVE state -------------------------
        fp = fingerprint(raw)
        margin.value(env, raw)
        assert fingerprint(env._adapter.raw_state) == fp, "MarginValue mutated env"

        if "SHOP" in phase_u:
            shop.decide_shop(env, raw, get_action_mask(env._adapter.raw_state), margin)
            assert fingerprint(env._adapter.raw_state) == fp, "GreedyShop mutated env"
            checked_shop = True
            if not rolled_once:
                rollout.value(env, env._adapter.raw_state)
                assert fingerprint(env._adapter.raw_state) == fp, "RolloutValue mutated env"
                rolled_once = True

        # --- real decision: must be legal, then advance --------------------
        raw = env._adapter.raw_state
        mask = get_action_mask(raw)
        fa, _reason, _method, _params, _wf = decide_fn(raw, mask, history)
        legal, reason = is_action_legal(fa, mask)
        assert legal, f"illegal emitted action: {reason}"
        _o, term, trunc, _m, _i = env.step(fa)
        done = term or trunc

    assert steps > 1
    assert checked_shop, "game never reached a SHOP — cannot exercise Slot 1"
    assert rolled_once, "RolloutValue non-mutation never probed"


# ---------------------------------------------------------------------------
# Battery determinism
# ---------------------------------------------------------------------------


def test_battery_is_deterministic_with_deterministic_slots(tmp_path):
    seeds = load_battery("train")[:2]
    out_a = str(tmp_path / "a.jsonl")
    out_b = str(tmp_path / "b.jsonl")
    # Determinism holds at any search depth; a light budget keeps the test cheap.
    tac = GreedyTactical(score_budget=80)
    a = run_battery(seeds, tac, GreedyShop(), MarginValue(), out_a, "det")
    b = run_battery(seeds, tac, GreedyShop(), MarginValue(), out_b, "det")
    assert [r.highest_ante for r in a] == [r.highest_ante for r in b]
    assert [r.won for r in a] == [r.won for r in b]


# ---------------------------------------------------------------------------
# Smoke battery: RandomShop vs GreedyShop (the known-difference sanity check)
# ---------------------------------------------------------------------------


def _read_records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _assert_well_formed(records, expected_slot1):
    for rec in records:
        assert rec["meta"]["slot1"] == expected_slot1
        assert rec["meta"]["slot2"] == "MarginValue"
        assert isinstance(rec["blinds"], list)
        assert rec["summary"]["n_blinds"] == len(rec["blinds"])
        for b in rec["blinds"]:
            assert _BLIND_KEYS.issubset(b.keys())
            assert b["blind_type"] in ("Small", "Big", "Boss")
            assert b["target"] > 0
            assert abs(b["margin"] - b["realized_score"] / b["target"]) < 1e-9


@pytest.mark.legacy_slow
def test_smoke_battery_random_vs_greedy(tmp_path):
    seeds = load_battery("train")[:6]
    rand_out = str(tmp_path / "random_train.jsonl")
    greedy_out = str(tmp_path / "greedy_train.jsonl")

    rand_res = run_battery(
        seeds,
        GreedyTactical(),
        RandomShop(random.Random(0)),
        MarginValue(),
        rand_out,
        "random",
    )
    greedy_res = run_battery(
        seeds, GreedyTactical(), GreedyShop(), MarginValue(), greedy_out, "greedy"
    )

    rand_records = _read_records(rand_out)
    greedy_records = _read_records(greedy_out)
    assert len(rand_records) == 6 and len(greedy_records) == 6
    _assert_well_formed(rand_records, "RandomShop")
    _assert_well_formed(greedy_records, "GreedyShop")

    rand_antes = [r.highest_ante for r in rand_res]
    greedy_antes = [r.highest_ante for r in greedy_res]
    print(f"\n[smoke] seeds={seeds}")
    print(f"[smoke] RandomShop antes={rand_antes}  max={max(rand_antes)}")
    print(f"[smoke] GreedyShop antes={greedy_antes}  max={max(greedy_antes)}")

    # Both arms share the strong GreedyTactical, so both clear early blinds; the
    # known difference is that systematic joker-buying (GreedyShop) goes deeper.
    assert max(greedy_antes) >= 2, "GreedyShop failed to clear ante 1"
    assert sum(greedy_antes) >= sum(rand_antes), "GreedyShop should be >= RandomShop"
    # At least one game records a per-blind resolution (the metrics layer needs them).
    assert any(rec["blinds"] for rec in greedy_records)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(
        "run via: uv run --no-sync python -m pytest tests/test_playground_harness.py -q"
    )
