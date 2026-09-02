"""Exact, non-mutating score preview used by the baseline tactical player.

``calculate_score`` previews the score of a
hypothetical hand by checkpointing the Jackdaw env, dry-running the play, reading
the real engine's result, and restoring — non-mutating and exact (no
approximation). Because
``env.load_state`` restores RNG too, the preview and the eventual real play
consume identical randomness, so the preview is exact even for probabilistic
cards.
"""

from __future__ import annotations

from typing import Any


def calculate_score(env: Any, cards: list[int]) -> dict:
    """Exact, non-mutating score preview for playing ``cards`` (hand indices).

    Checkpoints the env, dry-runs a PlayHand, reads the real engine's score, and
    restores. Returns ``{hand_type, score, chips, mult}`` or ``{error}`` (so the
    model can self-correct an illegal selection).
    """
    from jackdaw.env import FactoredAction

    if not cards:
        return {"error": "No cards provided — pass the hand indices you want to play."}
    try:
        idxs = tuple(int(c) for c in cards)
    except (TypeError, ValueError):
        return {"error": f"Card indices must be integers, got {cards!r}."}

    snapshot = env.get_state()
    try:
        _obs, _term, _trunc, _mask, info = env.step(FactoredAction(action_type=0, card_target=idxs))
        sr = info["raw_state"].get("last_score_result")
        if sr is None:
            return {"error": f"Playing {list(idxs)} produced no score (illegal selection?)."}
        return {
            "hand_type": getattr(sr, "hand_type", None),
            "score": int(getattr(sr, "total", 0)),
            "chips": float(getattr(sr, "chips", 0)),
            "mult": float(getattr(sr, "mult", 0)),
        }
    except Exception as e:  # illegal / invalid selection — let the caller retry
        return {"error": f"Could not score {list(idxs)}: {e}"}
    finally:
        env.load_state(snapshot)
