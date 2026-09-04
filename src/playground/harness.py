"""Two-slot playground harness — composes a battery run from swappable slots.

The benchmark fixes the *tactical* (in-blind card play) layer and swaps two slots:

- **Slot 1 — shop policy** (:class:`ShopPolicy`): buy / sell / reroll / pack picks.
- **Slot 2 — value estimator** (:class:`ValueEstimator`): scores a state so Slot 1
  can rank candidate moves.

Every idea = fill a slot, run :func:`run_battery` over the fixed seed bank, read one
comparable number out of the recorder JSONL and result artifact.

This module is *plumbing*: the baseline fills are deliberately crude. The only hard
invariants are (a) every emitted action is **legal** (the composer legality-gates each
slot's output), and (b) any slot that previews/rolls out is **non-mutating** — it
``env.get_state()`` before stepping and ``env.load_state()`` after, exactly like
``src/selfplay/tools.py:calculate_score``. A value estimator that mutates the live env
would corrupt the real game.

The composer (:func:`build_decider`) closes over the *same* ``env`` object that
``play_episode`` drives, so value estimators get checkpoint access without
``decide_fn`` ever receiving ``env`` directly.
"""

from __future__ import annotations

import itertools
import random
from typing import Protocol, runtime_checkable

import numpy as np
from jackdaw.env import (
    ActionType,
    BalatroEnvironment,
    DirectAdapter,
    FactoredAction,
    get_action_mask,
)

from src.playground.exact_score import preview_play
from src.selfplay.decision import get_fallback_action, is_action_legal
from src.selfplay.recorder import ACTION_NAMES, RunMeta, RunRecorder
from src.selfplay.runner import RunResult, play_episode
from src.selfplay.tools import calculate_score

# Action-type ids (mirror jackdaw ActionType / recorder ACTION_NAMES).
_PLAY = int(ActionType.PlayHand)
_DISCARD = int(ActionType.Discard)
_SELECT_BLIND = int(ActionType.SelectBlind)
_CASHOUT = int(ActionType.CashOut)
_NEXT_ROUND = int(ActionType.NextRound)
_SKIP_PACK = int(ActionType.SkipPack)
_BUY_CARD = int(ActionType.BuyCard)
_PICK_PACK = int(ActionType.PickPackCard)

# Hands strong enough to spend a (scarce) hand on; anything weaker we dig past with a
# free discard.
STRONG_HANDS = frozenset(
    {
        "Three of a Kind",
        "Straight",
        "Flush",
        "Full House",
        "Four of a Kind",
        "Straight Flush",
        "Five of a Kind",
        "Flush House",
        "Flush Five",
    }
)


# ---------------------------------------------------------------------------
# Slot protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Tactical(Protocol):
    """Slot (FIXED across A/B) — chooses the card play inside a blind."""

    name: str

    def decide_play(self, env, raw_state: dict, mask) -> FactoredAction:
        """Return a legal SELECTING_HAND action (PlayHand or Discard)."""
        ...


@runtime_checkable
class ValueEstimator(Protocol):
    """Slot 2 — scores a state (higher = better). MUST be non-mutating."""

    name: str

    def value(self, env, raw_state: dict) -> float:
        """Return a scalar value for ``raw_state``; leave ``env`` unchanged."""
        ...


@runtime_checkable
class ShopPolicy(Protocol):
    """Slot 1 — chooses the SHOP / PACK_OPENING action, consulting Slot 2."""

    name: str

    def decide_shop(self, env, raw_state: dict, mask, value: ValueEstimator) -> FactoredAction:
        """Return a legal shop/pack action."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _legal_cards(mask) -> list[int]:
    return [int(i) for i in np.nonzero(mask.card_mask)[0]]


def _blind_target(raw_state: dict) -> int:
    return int(getattr(raw_state.get("blind"), "chips", 0) or 0)


def _enumerate_play_combos(legal_cards: list[int], mask, budget: int) -> list[tuple]:
    """Card subsets the tactical may play, small-k first, capped at ``budget``.

    Small-k first: with the default budget (300) every subset of an 8-card hand
    (218 combos) is covered, so the 5-card straights/flushes are always scored.
    Above 8 cards the cap truncates, and the dropped subsets are the largest —
    see ``docs/known-limits.md``.
    """
    lo = max(1, int(mask.min_card_select))
    hi = min(int(mask.max_card_select), len(legal_cards))
    combos: list[tuple] = []
    for k in range(lo, hi + 1):
        for c in itertools.combinations(legal_cards, k):
            combos.append(c)
            if len(combos) >= budget:
                return combos
    return combos


def _best_play_scan_slow(env, combos):
    """The original checkpoint-based scan (``calculate_score`` per combo) — exact engine
    semantics including boss play-restrictions (an illegal selection errors and is skipped)."""
    best = None
    for combo in combos:
        res = calculate_score(env, list(combo))
        if "error" in res:
            continue
        if best is None or res["score"] > best[0]:
            best = (int(res["score"]), res["hand_type"], combo)
    return best


def best_play_scan(env, raw_state, mask, score_budget: int):
    """Exact-score every legal play-card subset; return ``(score, hand_type, combo)`` or None.

    Module-level so callers other than ``GreedyTactical`` can run the *same* exact scan
    without a tactical instance. Enumerates play-card subsets small-k first (capped at
    ``score_budget``) and returns the best. Returns ``None`` when there is no legal play.
    Behaviour identical to ``GreedyTactical._best_play``.

    Combos are scored by the engine-exact ``preview_play`` (no per-combo ``get_state``
    deep copy), with two anchors to a plain all-``calculate_score`` scan:

    1. **The FIRST combo still goes through the real checkpoint dry-run** — not for its
       score, but for its SIDE EFFECT: the dry-run mutates the caller's ``raw_state``
       dict in place (chips += score, hands_left −= 1, hand refilled) before the restore
       swaps the adapter to a fresh copy. ``GreedyTactical.decide_play`` reads
       ``need``/``discards_left`` from that dict AFTER the scan, so the baselines'
       decisions depend on exactly this first-combo mutation; it is preserved
       deliberately, not incidentally.
    2. **The winner is validated once** against ``calculate_score``. On any disagreement —
       e.g. a boss play-restriction the preview does not model made the fast winner
       engine-illegal — the whole state falls back to the original slow scan.

    Previews read the FRESH post-restore adapter dict, which is what every
    ``calculate_score`` call after the first effectively scored too.
    """
    legal = _legal_cards(mask)
    if not legal:
        return None
    combos = _enumerate_play_combos(legal, mask, score_budget)

    # Anchor 1: the old code's first call, side effects and all.
    first_res = calculate_score(env, list(combos[0]))
    best = None
    if "error" not in first_res:
        best = (int(first_res["score"]), first_res["hand_type"], combos[0])

    fresh = env._adapter.raw_state  # pristine post-restore copy for the pure previews
    ok = True
    for combo in combos[1:]:
        try:
            sr = preview_play(fresh, tuple(combo))
        except Exception:  # noqa: BLE001 — unpreviewable combo; slow path decides
            ok = False
            break
        score = int(sr.total)
        if best is None or score > best[0]:
            best = (score, sr.hand_type, combo)
    if ok and best is not None:
        if best[2] == combos[0]:
            return best  # the winner IS the engine-validated first dry-run
        # Anchor 2: validate the preview winner (pollutes only the fresh adapter
        # copy, which calculate_score restores — caller-visible state unchanged).
        res = calculate_score(env, list(best[2]))
        if "error" not in res and int(res["score"]) == best[0] and res["hand_type"] == best[1]:
            return best
    return _best_play_scan_slow(env, combos)


# ---------------------------------------------------------------------------
# Baseline tactical (FIXED layer used by both A/B arms)
# ---------------------------------------------------------------------------


class GreedyTactical:
    """The fixed in-blind play policy shared by both shop baselines.

    Works directly on ``FactoredAction`` + the ``ActionMask`` + exact
    ``calculate_score``:

    * enumerate every legal play-card subset (small-k first, capped at
      ``score_budget``) and score each **exactly** with ``calculate_score``;
    * **clinch** (play the best hand) if it reaches the chips still needed to clear
      the blind;
    * else **dig** with a free discard (discards don't cost a hand) — *unless*
      already holding a STRONG hand or out of discards, in which case make progress.

    Nuances preserved: discards are free so we dig to find a better hand; the
    ``score_budget`` caps the exact-score scan. The dig keeps the current best
    subset and discards the rest (deterministic), a faithful-in-spirit "dig deepest".
    """

    name = "GreedyTactical"

    def __init__(self, score_budget: int = 300) -> None:
        self.score_budget = score_budget

    def _best_play(self, env, raw_state, mask):
        """Exact-score every legal subset; return (score, hand_type, combo) or None.

        Thin delegate to the module-level :func:`best_play_scan`.
        """
        return best_play_scan(env, raw_state, mask, self.score_budget)

    def decide_play(self, env, raw_state: dict, mask) -> FactoredAction:
        play_legal = bool(mask.type_mask[_PLAY])
        discard_legal = bool(mask.type_mask[_DISCARD])
        best = self._best_play(env, raw_state, mask)

        need = _blind_target(raw_state) - int(raw_state.get("chips", 0) or 0)
        cr = raw_state.get("current_round") or {}
        discards_left = int(cr.get("discards_left", 0) or 0)

        # Clinch: the best hand clears the blind right now.
        if best is not None and play_legal and best[0] >= need:
            return FactoredAction(action_type=_PLAY, card_target=best[2])

        # Dig: best hand is weak and discards are free — throw the cards NOT in the
        # best subset and draw replacements (keeps the strongest partial hand).
        if discard_legal and discards_left > 0 and (best is None or best[1] not in STRONG_HANDS):
            return FactoredAction(action_type=_DISCARD, card_target=self._dig_cards(best, mask))

        # Strong hand, or out of discards: make progress with the best play.
        if best is not None and play_legal:
            return FactoredAction(action_type=_PLAY, card_target=best[2])
        return get_fallback_action(mask)

    def _dig_cards(self, best, mask) -> tuple:
        """Cards to discard: those not in the best subset, capped at max_card_select."""
        legal = _legal_cards(mask)
        hi = max(1, min(int(mask.max_card_select), len(legal)))
        keep = set(best[2]) if best is not None else set()
        rest = [c for c in legal if c not in keep]
        if not rest:  # best subset already covers the whole hand — dig anyway
            rest = legal
        return tuple(rest[:hi])


# ---------------------------------------------------------------------------
# Baseline value estimators (Slot 2)
# ---------------------------------------------------------------------------


class MarginValue:
    """Cheap margin proxy. Non-mutating (snapshots around any preview).

    * In SELECTING_HAND with a hand: best exact hand score (over a small sample of
      plays, ``calculate_score``) / blind target — how close one hand gets to the bar.
    * Otherwise (SHOP / PACK / between blinds, no hand to play): a board/economy
      proxy read directly from ``raw_state`` — ``#jokers + dollars/100`` — which still
      differs between candidate buys (so Slot-1 ranking is meaningful), without any
      env mutation.
    """

    name = "MarginValue"

    def __init__(self, sample_cap: int = 24) -> None:
        self.sample_cap = sample_cap

    def value(self, env, raw_state: dict) -> float:
        snapshot = env.get_state()
        try:
            phase_u = str(raw_state.get("phase", "")).upper()
            if "SELECTING_HAND" in phase_u:
                return self._hand_margin(env, raw_state)
            return self._economy_proxy(raw_state)
        finally:
            env.load_state(snapshot)  # non-mutation guarantee

    def _hand_margin(self, env, raw_state) -> float:
        mask = get_action_mask(raw_state)
        legal = _legal_cards(mask)
        if not legal or not mask.type_mask[_PLAY]:
            return self._economy_proxy(raw_state)
        best = 0
        for combo in _enumerate_play_combos(legal, mask, self.sample_cap):
            res = calculate_score(env, list(combo))
            if "error" not in res:
                best = max(best, int(res["score"]))
        target = _blind_target(raw_state)
        return (best / target) if target > 0 else float(best)

    @staticmethod
    def _economy_proxy(raw_state) -> float:
        n_jokers = len(raw_state.get("jokers", []) or [])
        dollars = int(raw_state.get("dollars", 0) or 0)
        return float(n_jokers) + dollars / 100.0


class RolloutValue:
    """Short greedy-tactical rollout. Non-mutating (checkpoint -> roll -> restore).

    Checkpoints the env, plays forward with the tactical layer inside blinds and a
    trivial "leave the shop" policy (so it measures how far the *current board* coasts
    on tactics alone), then restores exactly. Returns ``reached_ante + best_margin`` so
    deeper rollouts (better boards) rank higher, with the in-blind margin as a smooth
    tie-break. Capped at ``max_rollout_steps`` to stay cheap.
    """

    name = "RolloutValue"

    def __init__(self, max_rollout_steps: int = 30, tactical: Tactical | None = None) -> None:
        self.max_rollout_steps = max_rollout_steps
        # A lighter tactical keeps the rollout cheap; the signal is coarse by design.
        self._tactical = tactical or GreedyTactical(score_budget=128)

    def value(self, env, raw_state: dict) -> float:
        snapshot = env.get_state()
        try:
            return self._rollout(env)
        finally:
            env.load_state(snapshot)  # non-mutation guarantee

    def _rollout(self, env) -> float:
        raw = env._adapter.raw_state
        reached_ante = int((raw.get("round_resets") or {}).get("ante", 1))
        best_margin = 0.0
        for _ in range(self.max_rollout_steps):
            mask = get_action_mask(raw)
            action = self._rollout_action(env, raw, mask)
            if action is None:
                break
            try:
                _o, term, trunc, _m, info = env.step(action)
            except Exception:  # noqa: BLE001 — a bad rollout step ends the rollout
                break
            raw = info["raw_state"]
            reached_ante = max(reached_ante, int((raw.get("round_resets") or {}).get("ante", 1)))
            target = _blind_target(raw)
            if target > 0:
                best_margin = max(best_margin, int(raw.get("chips", 0) or 0) / target)
            if term or trunc:
                break
        return float(reached_ante) + min(best_margin, 0.999)

    def _rollout_action(self, env, raw, mask):
        phase_u = str(raw.get("phase", "")).upper()
        if "BLIND_SELECT" in phase_u and mask.type_mask[_SELECT_BLIND]:
            return FactoredAction(action_type=_SELECT_BLIND)
        if "ROUND_EVAL" in phase_u:
            if mask.type_mask[_CASHOUT]:
                return FactoredAction(action_type=_CASHOUT)
            if mask.type_mask[_NEXT_ROUND]:
                return FactoredAction(action_type=_NEXT_ROUND)
        if "SELECTING_HAND" in phase_u:
            return self._tactical.decide_play(env, raw, mask)
        if "SHOP" in phase_u and mask.type_mask[_NEXT_ROUND]:
            return FactoredAction(action_type=_NEXT_ROUND)
        if "PACK_OPENING" in phase_u and mask.type_mask[_SKIP_PACK]:
            return FactoredAction(action_type=_SKIP_PACK)
        if not mask.type_mask.any():
            return None
        return get_fallback_action(mask)


# ---------------------------------------------------------------------------
# Baseline shop policies (Slot 1)
# ---------------------------------------------------------------------------


class RandomShop:
    """Pick a uniformly-random LEGAL shop/pack action (mirrors ``random_decider``).

    Ignores Slot 2 (a deliberately value-blind baseline). Constructs targets the way
    ``random_decider`` does and validates legality, retrying other legal types if a
    construction is illegal (e.g. a consumable that needs card targets), so it never
    emits an illegal action.
    """

    name = "RandomShop"

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def decide_shop(self, env, raw_state, mask, value) -> FactoredAction:
        legal_types = [int(t) for t in np.nonzero(mask.type_mask)[0]]
        self.rng.shuffle(legal_types)
        for at in legal_types:
            card_target = None
            entity_target = None
            if at in (_PLAY, _DISCARD):
                cards = _legal_cards(mask)
                if not cards:
                    continue
                lo = max(1, int(mask.min_card_select))
                hi = min(int(mask.max_card_select), len(cards))
                if hi < lo:
                    continue
                card_target = tuple(self.rng.sample(cards, self.rng.randint(lo, hi)))
            elif at in mask.entity_masks:
                ents = [int(e) for e in np.nonzero(mask.entity_masks[at])[0]]
                if not ents:
                    continue
                entity_target = self.rng.choice(ents)
            fa = FactoredAction(
                action_type=at, card_target=card_target, entity_target=entity_target
            )
            legal, _ = is_action_legal(fa, mask)
            if legal:
                return fa
        return get_fallback_action(mask)


class GreedyShop:
    """Crude buy heuristic that genuinely consults Slot 2.

    * PACK_OPENING: take the first pack card if offered, else skip.
    * SHOP: if an affordable joker with a free slot exists, buy it; with ≥2
      candidates, **rank them by the value estimator** (simulate each buy on a
      checkpoint, score the resulting state, restore) — the real Slot1→Slot2 path.
      Otherwise advance to the next round.

    Affordability / slot legality come from the mask (``_mask_shop_buy`` already
    blocks unaffordable buys and full-slot joker buys); we never re-derive them, and
    never sell the only joker.
    """

    name = "GreedyShop"

    def decide_shop(self, env, raw_state, mask, value) -> FactoredAction:
        phase_u = str(raw_state.get("phase", "")).upper()
        if "PACK_OPENING" in phase_u:
            if mask.type_mask[_PICK_PACK]:
                ents = np.nonzero(mask.entity_masks[_PICK_PACK])[0]
                if len(ents):
                    return FactoredAction(action_type=_PICK_PACK, entity_target=int(ents[0]))
            return get_fallback_action(mask)

        # SHOP: candidate joker buys (mask already guarantees affordable + slot-free).
        candidates = self._joker_buys(raw_state, mask)
        if candidates:
            if len(candidates) >= 2:
                chosen = self._rank_by_value(env, candidates, value)
            else:
                chosen = candidates[0]
            return FactoredAction(action_type=_BUY_CARD, entity_target=int(chosen))

        if mask.type_mask[_NEXT_ROUND]:
            return FactoredAction(action_type=_NEXT_ROUND)
        return get_fallback_action(mask)

    @staticmethod
    def _joker_buys(raw_state, mask) -> list[int]:
        """Legal BuyCard entity indices whose shop card is a Joker, cheapest first."""
        em = mask.entity_masks.get(_BUY_CARD)
        if em is None:
            return []
        shop_cards = raw_state.get("shop_cards", []) or []
        out = []
        for i, ok in enumerate(em):
            if not ok or i >= len(shop_cards):
                continue
            ability = getattr(shop_cards[i], "ability", None)
            cset = ability.get("set", "") if isinstance(ability, dict) else ""
            if cset == "Joker":
                out.append(i)
        out.sort(key=lambda i: getattr(shop_cards[i], "cost", 0))
        return out

    def _rank_by_value(self, env, candidates, value) -> int:
        """Simulate each buy on a checkpoint, score with Slot 2, pick argmax; restore."""
        snapshot = env.get_state()
        best_i, best_v = candidates[0], float("-inf")
        try:
            for i in candidates:
                env.load_state(snapshot)
                _o, _t, _tr, _m, info = env.step(
                    FactoredAction(action_type=_BUY_CARD, entity_target=int(i))
                )
                v = value.value(env, info["raw_state"])
                if v > best_v:
                    best_v, best_i = v, i
        finally:
            env.load_state(snapshot)  # undo all previewed buys
        return best_i


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------


def build_decider(env, tactical: Tactical, shop: ShopPolicy, value: ValueEstimator):
    """Compose a ``decide_fn(raw_state, mask, history)`` for ``play_episode``.

    Closes over ``env`` (the same object ``play_episode`` drives) so the slots get
    checkpoint access — ``decide_fn`` itself never receives ``env``. Dispatches by
    phase and **legality-gates** every slot output (illegal -> legal fallback).

    The blind-select and cash-out branches are this composed agent's **own policy**,
    not a harness convenience: it never skips a blind and never uses a consumable
    before cashing out. That is a real strategic abstention and it is stamped into
    every recorded decision as ``always-select-blind`` / ``always-cash-out``, so a
    reader can see the choice was made rather than assume the phase did not exist.
    Holding it fixed is also what keeps ``random-shop`` vs ``greedy-shop`` a clean
    A/B: the two arms differ in the shop and nowhere else.

    Exceptions are **not** caught here. ``scripts/evaluate.py``'s ``_play_one``
    already isolates every game and reports failures as data, so a second net at
    this level only converted a named crash into a silent legal move and a clean
    exit code.
    """

    def decide_fn(raw_state, mask, history):
        phase_u = str(raw_state.get("phase", "")).upper()
        if "SELECTING_HAND" in phase_u:
            fa = tactical.decide_play(env, raw_state, mask)
            reasoning = "greedy-tactical"
        elif "SHOP" in phase_u or "PACK_OPENING" in phase_u:
            fa = shop.decide_shop(env, raw_state, mask, value)
            reasoning = shop.name
        elif "BLIND_SELECT" in phase_u and mask.type_mask[_SELECT_BLIND]:
            fa = FactoredAction(action_type=_SELECT_BLIND)
            reasoning = "always-select-blind"
        elif "ROUND_EVAL" in phase_u and mask.type_mask[_CASHOUT]:
            fa = FactoredAction(action_type=_CASHOUT)
            reasoning = "always-cash-out"
        else:
            # No phase left that the engine offers an action in; a genuine gap.
            return _packet(get_fallback_action(mask), "fallback-phase", was_fallback=True)
        legal, _ = is_action_legal(fa, mask)
        if not legal:
            return _packet(get_fallback_action(mask), "fallback-illegal", was_fallback=True)
        return _packet(fa, reasoning, was_fallback=False)

    return decide_fn


def _packet(fa: FactoredAction, reasoning: str, was_fallback: bool):
    """Build the 5-tuple ``play_episode`` expects from a chosen action.

    ``was_fallback`` is passed explicitly by the caller and means exactly one thing:
    *this composer substituted for a slot*. It was previously derived as
    ``reasoning.startswith("fallback")`` over the slot's own name, which made the
    flag a property of a string rather than of what happened.

    It still cannot see a slot that calls :func:`get_fallback_action` itself — such a
    policy returns an ordinary action and is recorded as having decided. Reading this
    flag as "the agent had no opinion" is therefore wrong in a way no counter here can
    detect; it is a lower bound on abstention, not a measure of it.
    """
    method = ACTION_NAMES.get(int(fa.action_type), str(int(fa.action_type)))
    return fa, reasoning, method, {}, bool(was_fallback)


# ---------------------------------------------------------------------------
# Battery runner
# ---------------------------------------------------------------------------


def run_battery_with(
    seeds: list[str],
    make_decider,
    out_path: str,
    config_label: str,
    slot1: str = "",
    slot2: str = "",
    max_steps: int = 2000,
) -> list[RunResult]:
    """Run an arbitrary decider over ``seeds``; one JSONL at ``out_path``.

    The general form of :func:`run_battery`. ``make_decider(env, seed) -> decide_fn``
    is the widest agent contract the harness can honour: it hands the agent the
    same live ``env`` that ``play_episode`` drives, so an agent may checkpoint and
    preview via ``get_state``/``load_state``, and returns the
    ``decide_fn(raw_state, mask, history)`` the episode loop calls.

    ``seed`` is passed so a stochastic agent can derive its RNG from the run seed
    and stay reproducible; an agent that seeds itself from wall-clock entropy
    produces numbers nobody can replay.

    Agents that do not decompose into (tactical, shop, value) — search, learned
    policies, anything with cross-phase state — plug in here. ``slot1``/``slot2``
    are free-text provenance labels written into ``RunMeta``; they no longer have
    to name a policy object.

    For each seed: fresh ``BalatroEnvironment`` (Red Deck ``b_red`` / White Stake
    ``stake=1`` — the env defaults), fresh ``RunRecorder`` appending to
    ``out_path``, a tagged ``RunMeta``, then ``play_episode``. Returns one
    ``RunResult`` per seed, in order.
    """
    results: list[RunResult] = []
    for seed in seeds:
        env = BalatroEnvironment(adapter_factory=DirectAdapter)
        recorder = RunRecorder(output_path=out_path)
        run_meta = RunMeta(
            seed=seed,
            seed_mode="set",
            temperature_regime="",
            temperature_value=0.0,
            slot1=slot1,
            slot2=slot2,
            config_label=config_label,
        )
        decide_fn = make_decider(env, seed)
        results.append(play_episode(env, seed, decide_fn, recorder, run_meta, max_steps=max_steps))
    return results


def run_battery(
    seeds: list[str],
    tactical: Tactical,
    shop: ShopPolicy,
    value: ValueEstimator,
    out_path: str,
    config_label: str,
    max_steps: int = 2000,
) -> list[RunResult]:
    """Run the (tactical, shop, value) config over ``seeds``; one JSONL at ``out_path``.

    The two-slot special case of :func:`run_battery_with`, kept because it is the
    vocabulary the playground A/Bs are written in.
    """
    return run_battery_with(
        seeds,
        lambda env, _seed: build_decider(env, tactical, shop, value),
        out_path=out_path,
        config_label=config_label,
        slot1=shop.name,
        slot2=value.name,
        max_steps=max_steps,
    )
