"""Backend-agnostic per-run event recorder for self-play.

Consumes the Jackdaw ``raw_state`` before/after each decision plus the chosen
``FactoredAction``, and derives semantic events (hands played, jokers
bought/sold, vouchers redeemed) by diffing the two states. One JSON record per
run is appended to a JSONL store.

It works for any player because it only reads state dictionaries and an action's
``action_type``; it imports nothing from ``jackdaw``.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

# Action-type names, kept local so this module stays jackdaw-free.
ACTION_NAMES = {
    0: "PlayHand",
    1: "Discard",
    2: "SelectBlind",
    3: "SkipBlind",
    4: "CashOut",
    5: "Reroll",
    6: "NextRound",
    7: "SkipPack",
    8: "BuyCard",
    9: "SellJoker",
    10: "SellConsumable",
    11: "UseConsumable",
    12: "RedeemVoucher",
    13: "OpenBooster",
    14: "PickPackCard",
    15: "SwapJokersLeft",
    16: "SwapJokersRight",
    17: "SwapHandLeft",
    18: "SwapHandRight",
    19: "SortHandRank",
    20: "SortHandSuit",
}

_PLAY_HAND, _DISCARD, _BUY_CARD, _SELL_JOKER, _REDEEM_VOUCHER = 0, 1, 8, 9, 12


def _serialize_action(fa: Any) -> dict[str, Any]:
    """Full, replayable form of a FactoredAction (these are its only fields)."""
    ct = getattr(fa, "card_target", None)
    return {
        "action_type": int(getattr(fa, "action_type", -1)),
        "card_target": list(ct) if ct else None,
        "entity_target": getattr(fa, "entity_target", None),
    }


@dataclass
class RunMeta:
    seed: str
    seed_mode: str  # "random" | "set"
    temperature_regime: str
    temperature_value: float
    deck: str = ""
    stake: str = ""
    model_id: str = ""
    backend: str = ""  # "random" | "local-hf" | "gemini-api" | ...
    # A/B grouping dimensions carried into the raw run record.
    # Defaults keep pre-playground records/tests valid.
    slot1: str = ""  # shop-policy label, e.g. "RandomShop" | "GreedyShop"
    slot2: str = ""  # value-estimator label, e.g. "MarginValue" | "RolloutValue"
    config_label: str = ""  # human label for the whole config, e.g. "greedy+margin"
    tools: list[str] = field(default_factory=list)  # enabled tool names (A/B grouping dimension)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.time)


def _ante(gs: dict) -> int:
    return int((gs.get("round_resets") or {}).get("ante", 1))


def _money(gs: dict) -> int:
    return int(gs.get("dollars", 0) or 0)


def _consumable_count(gs: dict) -> int:
    # Jackdaw uses "consumables"; tolerate the Lua spelling defensively.
    return len(gs.get("consumables") or gs.get("consumeables") or [])


def _jokers(gs: dict) -> list[dict]:
    out: list[dict] = []
    for j in gs.get("jokers", []) or []:
        ability = getattr(j, "ability", None) or {}
        label = ability.get("name", "") if isinstance(ability, dict) else ""
        out.append({"key": getattr(j, "center_key", None), "label": label})
    return out


def _phase_str(gs: dict) -> str:
    """Uppercased phase string — robust to GamePhase enum or plain str."""
    return str(gs.get("phase", "")).upper()


def _blind_type_from_name(name: str) -> str:
    """Map a Blind's display name to {"Small","Big","Boss"} (mirrors Blind.get_type).

    Derived from ``blind.name`` rather than ``gs["blind_on_deck"]`` because, by the
    time a clear is observed, ``blind_on_deck`` has already advanced to the *next*
    blind (``_round_won`` in game.py), whereas the ``blind`` object itself is left
    untouched until the next ``SelectBlind``.
    """
    if name == "Small Blind":
        return "Small"
    if name == "Big Blind":
        return "Big"
    if name:
        return "Boss"
    return ""


class RunRecorder:
    """Collects one run's events and appends a JSON record to a JSONL store."""

    def __init__(self, output_path: str | None = None) -> None:
        self.output_path = output_path or os.path.join(
            "data", "selfplay", "runs", f"{date.today().isoformat()}.jsonl"
        )
        self._meta: RunMeta | None = None
        self._events: list[dict] = []
        self._tool_calls: list[dict] = []
        # Per-blind playground state (one record per blind resolution).
        self._blinds: list[dict] = []
        self._blind_hist: dict[str, int] = {}  # hand_type -> count, current blind only
        self._blind_hands_used: int = 0  # PlayHands played in the current blind

    def start_run(self, meta: RunMeta) -> None:
        self._meta = meta
        self._events = []
        self._tool_calls = []
        self._blinds = []
        self._blind_hist = {}
        self._blind_hands_used = 0

    def record_step(
        self,
        prev_raw_state: dict,
        raw_state: dict,
        factored_action: Any,
        method: str | None = None,
        params: Any = None,
        reasoning: str | None = None,
        was_fallback: bool = False,
    ) -> None:
        """Derive and store one decision event from the state delta."""
        at = int(getattr(factored_action, "action_type", -1))
        # Reasoning: prefer the canonical top-level string, but fall back to one nested
        # in `parameters`, so a decider that puts it there does not silently lose
        # the "why".
        rsn = reasoning or ""
        if not rsn and isinstance(params, dict):
            rsn = params.get("reasoning") or ""
        ev: dict[str, Any] = {
            "step": len(self._events),
            "ante": _ante(prev_raw_state),
            "action_type": at,
            "action": ACTION_NAMES.get(at, str(at)),
            # Full replayable action plus a concise policy reason.
            "factored_action": _serialize_action(factored_action),
            "reasoning": rsn,
            "was_fallback": bool(was_fallback),
        }
        if method:
            ev["method"] = method

        if at == _PLAY_HAND:
            sr = raw_state.get("last_score_result")
            ev["hand_type"] = getattr(sr, "hand_type", None)
            ev["score"] = getattr(sr, "total", None)
            ev["chips"] = getattr(sr, "chips", None)
            ev["mult"] = getattr(sr, "mult", None)
            ct = getattr(factored_action, "card_target", None)
            ev["n_cards"] = len(ct) if ct else 0
        elif at == _DISCARD:
            ct = getattr(factored_action, "card_target", None)
            ev["n_cards"] = len(ct) if ct else 0
        elif at == _BUY_CARD:
            ev["cost"] = _money(prev_raw_state) - _money(raw_state)
            prev_keys = {j["key"] for j in _jokers(prev_raw_state)}
            cur = _jokers(raw_state)
            if len(cur) > len(prev_keys):
                new = [j for j in cur if j["key"] not in prev_keys]
                ev["item_kind"] = "joker"
                if new:
                    ev["label"], ev["key"] = new[-1]["label"], new[-1]["key"]
            elif _consumable_count(raw_state) > _consumable_count(prev_raw_state):
                ev["item_kind"] = "consumable"
            else:
                ev["item_kind"] = "other"
        elif at == _SELL_JOKER:
            ev["proceeds"] = _money(raw_state) - _money(prev_raw_state)
            cur_keys = {j["key"] for j in _jokers(raw_state)}
            sold = [j for j in _jokers(prev_raw_state) if j["key"] not in cur_keys]
            if sold:
                ev["label"], ev["key"] = sold[0]["label"], sold[0]["key"]
        elif at == _REDEEM_VOUCHER:
            prev_v = set((prev_raw_state.get("used_vouchers") or {}).keys())
            new_v = [v for v in (raw_state.get("used_vouchers") or {}) if v not in prev_v]
            if new_v:
                ev["voucher"] = new_v[0]

        self._events.append(ev)

        # --- Playground: per-blind resolution tracking ----------------------
        # Tally PlayHands into the current blind, then — if this PlayHand moved
        # the round out of SELECTING_HAND — emit one blind record and reset the
        # tally for the next blind (which starts on the next SelectBlind).
        if at == _PLAY_HAND:
            ht = ev.get("hand_type")
            if ht:
                self._blind_hist[ht] = self._blind_hist.get(ht, 0) + 1
            self._blind_hands_used += 1
            phase_u = _phase_str(raw_state)
            cleared = "ROUND_EVAL" in phase_u  # _round_won -> ROUND_EVAL
            failed = "GAME_OVER" in phase_u  # out of hands & not saved
            if cleared or failed:
                self._record_blind_resolution(raw_state, cleared)

    def _record_blind_resolution(self, raw_state: dict, cleared: bool) -> None:
        """Append one per-blind record and reset the per-blind tally.

        IMPORTANT — the sim adapter is zero-copy: the recorder only ever sees the
        *post-step* state (``prev_raw_state is raw_state`` in the real runner).  So
        every field is read from the just-resolved ``blind`` object / ``raw_state``:

        * ``target``/``boss_key``/``blind_type`` come from ``raw_state["blind"]``,
          which ``_round_won`` leaves untouched (it advances ``blind_on_deck`` and
          ``round_resets["ante"]``, not the ``blind`` object), so they are not
          polluted by the resolution.
        * ``realized_score`` is ``raw_state["chips"]`` — the accumulated round score,
          only zeroed at the *next* ``SelectBlind``, so it still holds the full
          pre-reset total here (includes the resolving hand).
        * ``ante`` is the only polluted field: a *boss clear* runs ``_advance_ante``
          (game.py), bumping ``round_resets["ante"]`` by 1 before we observe it, so
          we subtract 1 in that one case to record the ante the blind belonged to.
        """
        blind = raw_state.get("blind")
        target = int(getattr(blind, "chips", 0) or 0)
        boss = bool(getattr(blind, "boss", False))
        blind_type = _blind_type_from_name(getattr(blind, "name", "") or "")
        boss_key = getattr(blind, "key", None) if boss else None

        realized = int(raw_state.get("chips", 0) or 0)
        cr = raw_state.get("current_round") or {}
        post_ante = int((raw_state.get("round_resets") or {}).get("ante", 1))
        ante = post_ante - 1 if (cleared and boss) else post_ante
        margin = (realized / target) if target > 0 else 0.0

        self._blinds.append(
            {
                "ante": ante,
                "blind_type": blind_type,
                "boss_key": boss_key,
                "target": target,
                "realized_score": realized,
                "margin": margin,
                "cleared": bool(cleared),
                "hands_used": self._blind_hands_used,
                "hands_left": int(cr.get("hands_left", 0) or 0),
                "discards_left": int(cr.get("discards_left", 0) or 0),
                "played_hand_histogram": dict(self._blind_hist),
            }
        )
        # Reset the per-blind tally; the next blind starts fresh.
        self._blind_hist = {}
        self._blind_hands_used = 0

    def record_tool_call(self, name: str, args: Any, result: Any) -> None:
        """Log a neuro-symbolic tool call, tagged with the decision it precedes.

        Kept separate from decision events so ``n_decisions`` stays clean; feeds the
        explainable-AI angle (what the agent consulted before acting).
        """
        self._tool_calls.append(
            {
                "decision_step": len(self._events),
                "tool": name,
                "args": args,
                "result": result,
            }
        )

    def finish_run(
        self,
        highest_ante: int,
        won: bool,
        episode_length: int,
        fallback_occurred: bool,
        terminal_reason: str,
        final_raw_state: dict | None = None,
    ) -> dict:
        # The engine's won flag is non-latching: it's reset to False on any blind
        # failure (game.py:658), so a run that beats the ante-8 boss and then dies in
        # endless ante 9 arrives here with won=False. Recover the true outcome from
        # the per-blind ground truth (a cleared Boss at ante >= 8). `won_env_flag`
        # preserves the raw flag for debugging.
        won_real = bool(won) or any(
            b.get("cleared") and b.get("blind_type") == "Boss" and int(b.get("ante", 0) or 0) >= 8
            for b in self._blinds
        )
        summary: dict[str, Any] = {
            "highest_ante": int(highest_ante),
            "won": won_real,
            "won_env_flag": bool(won),
            "episode_length": int(episode_length),
            "fallback_occurred": bool(fallback_occurred),
            "terminal_reason": terminal_reason,
            "n_decisions": len(self._events),
            "n_blinds": len(self._blinds),
        }
        if final_raw_state is not None:
            jl = _jokers(final_raw_state)
            summary["final_jokers"] = [j["label"] for j in jl]
            summary["final_joker_keys"] = [j["key"] for j in jl]
            summary["final_vouchers"] = list((final_raw_state.get("used_vouchers") or {}).keys())
            summary["final_money"] = _money(final_raw_state)

        record = {
            "meta": asdict(self._meta) if self._meta else {},
            "events": self._events,
            "tool_calls": self._tool_calls,
            "blinds": self._blinds,
            "summary": summary,
        }
        self._append(record)
        return record

    def _append(self, record: dict) -> None:
        directory = os.path.dirname(self.output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
