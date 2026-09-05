"""Engine-exact, checkpoint-free score preview (``preview_play``).

A pure, non-mutating preview of Jackdaw's PlayHand scoring: it replicates the
engine's own ``_handle_play_hand`` steps 2-6 (``jackdaw/engine/game.py:473-589``)
on defensive copies and calls the engine's own ``score_hand`` — never a
re-implementation of scoring semantics.

It scores the CURRENT visible state exactly like ``calculate_score``, ~50× faster
because it skips the full ``env.get_state()``/``load_state()`` deep copy and
copies only the objects scoring can mutate. Concept attribution:
EFHIII/balatro-calculator (MIT) — the *concept* of a standalone best-play
scorer; the scoring semantics here are Jackdaw's own pipeline.

Determinism note: scoring consumes RNG only via
- jokers: ``RNG_SCORING_JOKERS`` below (jokers.py:417,731,1261,1277,2019,2369)
- played Lucky Cards (card.py:635,723) and scoring Glass Cards (scoring.py:792)
``_copy_rng`` clones the live stream state, so previews of RNG hands match the
engine's own dry-run from the same state.
"""

from __future__ import annotations

import copy
from typing import Any

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card
from jackdaw.engine.game import _press_play
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import ScoreResult, score_hand

__all__ = [
    "RNG_SCORING_JOKERS",
    "hand_has_rng_effects",
    "preview_play",
]

# Jokers whose SCORING consumes RNG (jackdaw/engine/jokers.py:417,731,1261,
# 1277,2019,2369). Previews of them are still exact: the rng clone shares the
# live stream state.
RNG_SCORING_JOKERS: frozenset[str] = frozenset(
    {
        "j_misprint",  # Misprint: mult uniform 0..23
        "j_bloodstone",  # Bloodstone: 1-in-2 x1.5 per Heart
        "j_business",  # Business Card: 1-in-2 $2 per face
        "j_reserved_parking",  # Reserved Parking: 1-in-2 $1 per held face
        "j_8_ball",  # 8 Ball: 1-in-4 tarot per played 8
        "j_space",  # Space Joker: 1-in-4 level-up (also mutates hand_levels)
    }
)


def hand_has_rng_effects(played: list[Card], jokers: list[Card]) -> bool:
    """True if scoring this hand would consume RNG (non-certifiable)."""
    if any(getattr(j, "center_key", None) in RNG_SCORING_JOKERS for j in jokers):
        return True
    for c in played:
        ability = getattr(c, "ability", None) or {}
        if ability.get("effect") == "Lucky Card" or ability.get("name") == "Lucky Card":
            return True
        if ability.get("name") == "Glass Card":
            return True
    return False


def _copy_rng(rng: Any) -> PseudoRandom:
    """Exact clone of a live PseudoRandom (stream counters aligned)."""
    clone = PseudoRandom(rng.seed_str)
    # Direct state copy — load_state() re-hashes 0-valued streams (the save
    # convention), which would diverge from a live rng that holds a true 0.
    clone.state.clear()
    clone.state.update(rng.get_state())
    return clone


def preview_play(gs: dict[str, Any], indices: tuple[int, ...]) -> ScoreResult:
    """Score the hand the engine WOULD score for PlayHand(indices) on *gs*.

    Pure: *gs* (a live Jackdaw game-state dict, e.g. ``info["raw_state"]``) is
    never mutated — everything scoring can touch is defensively copied. The
    returned ``ScoreResult`` is bit-identical to the ``last_score_result`` the
    engine produces when actually stepping PlayHand(indices) from this state.

    Mirrors ``_handle_play_hand`` (game.py:473-589) steps 2-6 exactly:
    played/held split in selection order, hands_left decrement, per-card stat
    flags, ``_press_play`` boss effects (The Hook consumes rng + moves held
    cards), the Group-A game_state keys, then the engine's ``score_hand``.
    """
    hand: list[Card] = gs.get("hand", [])
    if not indices or not hand:
        raise ValueError("Must select at least 1 card")
    if len(indices) > 5:
        raise ValueError("Cannot play more than 5 cards")
    if any(i < 0 or i >= len(hand) for i in indices):
        raise ValueError("Card index out of range")
    cr = gs["current_round"]
    if cr["hands_left"] <= 0:
        raise ValueError("No hands remaining")

    # --- defensive copies of everything scoring can mutate ---
    hand_copy: list[Card] = copy.deepcopy(hand)
    jokers: list[Card] = copy.deepcopy(gs.get("jokers", []))
    hand_levels: HandLevels = copy.deepcopy(gs.get("hand_levels"))
    blind: Blind = copy.deepcopy(gs["blind"])
    rng = _copy_rng(gs["rng"])

    # Step 2: split in SELECTION order (game.py:513-515).
    idx_set = set(indices)
    played = [hand_copy[i] for i in indices]
    held = [c for i, c in enumerate(hand_copy) if i not in idx_set]

    # Step 3: hands_left decrement / hands_played increment (game.py:520-522).
    hands_left = cr["hands_left"] - 1
    hands_played = cr["hands_played"] + 1

    # Step 4: per-card stats (game.py:528-535) — on copies.
    for card in played:
        base = getattr(card, "base", None)
        if base is not None:
            base.times_played = getattr(base, "times_played", 0) + 1
        ability = getattr(card, "ability", None)
        if isinstance(ability, dict):
            ability["played_this_ante"] = True

    # Step 5: boss press_play (game.py:540, game.py:2296) — the engine's own
    # code on a synthetic gs. The Hook mutates 'hand' (held) + consumes rng;
    # The Tooth mutates 'dollars' BEFORE money is snapshotted below.
    synth: dict[str, Any] = {
        "hand": held,
        "discard_pile": [],
        "dollars": gs.get("dollars", 0),
        "jokers": jokers,
    }
    _press_play(synth, blind, played, rng)
    held = synth["hand"]  # The Hook may have removed cards

    # Step 6 prelude: Group-A game_state keys (game.py:550-576).
    live_deck = gs.get("deck", [])
    live_discard = gs.get("discard_pile", [])
    # Read-only tally sweep — union identical to the engine's post-press_play
    # deck + hand + discard_pile + played (Hook moves held→discard; len same).
    all_cards = [*live_deck, *held, *live_discard, *synth["discard_pile"], *played]
    synth["hands_left"] = hands_left
    synth["current_round_hands_played"] = hands_played
    synth["discards_left"] = cr.get("discards_left", 0)
    synth["discards_used"] = cr.get("discards_used", 0)
    synth["money"] = synth["dollars"]
    synth["deck_cards_remaining"] = len(live_deck)
    synth["playing_cards_count"] = len(all_cards)
    synth["stone_tally"] = sum(1 for c in all_cards if getattr(c, "center_key", None) == "m_stone")
    synth["steel_tally"] = sum(1 for c in all_cards if getattr(c, "center_key", None) == "m_steel")
    synth["enhanced_card_count"] = sum(
        1 for c in all_cards if getattr(c, "center_key", "") not in ("", "c_base")
    )
    synth["mail_card_id"] = cr.get("mail_card", {}).get("id")
    synth["idol_card"] = cr.get("idol_card")
    synth["ancient_suit"] = cr.get("ancient_card", {}).get("suit")
    synth["consumable_usage_tarot"] = gs.get("consumable_usage_total", {}).get("tarot", 0)
    # Group-B keys score_hand reads from gs (run_init.py:46,93,292,327).
    synth["joker_slots"] = gs.get("joker_slots", 5)
    synth["starting_deck_size"] = gs.get("starting_deck_size", 52)

    return score_hand(
        played_cards=played,
        held_cards=held,
        jokers=jokers,
        hand_levels=hand_levels,
        blind=blind,
        rng=rng,
        probabilities_normal=gs.get("probabilities", {}).get("normal", 1),
        game_state=synth,
        back_key=gs.get("selected_back_key"),
        blind_chips=blind.chips,
    )
