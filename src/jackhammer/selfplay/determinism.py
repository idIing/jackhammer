"""Whole-run determinism check — sim-to-sim reproducibility of a complete episode.

The bet: same seed + same action stream must reproduce the same observable state
at *every* step. We generate a complete, legal action stream with the RandomPlayer,
then replay that exact stream through a fresh env and compare
a per-step fingerprint. Any mismatch is a desync; we report the first diverging step.

Fingerprint = sha256 of the canonical state serialization (a clean, address-free
dict). It is robust across fresh envs, and any
RNG call-ordering desync surfaces as a different shop / draw / boss downstream.

Whole-run RNG call ordering is the residual risk this instrument measures.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jackdaw.bridge.serializer import game_state_to_bot_response
from jackdaw.env import BalatroEnvironment, DirectAdapter, FactoredAction

# Transitions play_episode_async auto-applies (and does NOT record): select blind,
# cash out, advance round. Replay must re-derive them to rebuild the full stream.
_SELECT_BLIND, _CASHOUT, _NEXT_ROUND = 2, 4, 6

# decide_fn(raw_state, mask, history) -> (factored, ...) | factored
DecideFn = Callable[[Any, Any, list], Any]


# Keys that carry per-process OBJECT IDENTITY, not seeded game CONTENT. A card's
# `id` is a global creation counter (the Nth Card built in this process), so it is
# offset between two fresh envs even on the same seed — stripping it leaves the
# content identity (rank/suit/key/position), which is what determinism is about.
_VOLATILE_KEYS = frozenset({"id"})


def _scrub(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items() if k not in _VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_scrub(x) for x in obj]
    return obj


def fingerprint(raw_state: Any) -> str:
    """Stable hash of the observable game CONTENT (serializer output, identity scrubbed)."""
    payload = _scrub(game_state_to_bot_response(raw_state))
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class RunCapture:
    seed: str
    actions: list  # full stepped FactoredAction stream
    fingerprints: list[str]  # len == len(actions) + 1 (initial state + post-step states)


def record_run(seed: str, decide_fn: DecideFn, max_steps: int = 2000) -> RunCapture:
    """Play one episode, capturing every stepped action and per-step state fingerprint."""
    env = BalatroEnvironment(adapter_factory=DirectAdapter)
    _o, mask, info = env.reset(seed=seed)
    actions: list = []
    fps = [fingerprint(info["raw_state"])]
    history: list = []
    done = False
    steps = 0
    while not done and steps < max_steps:
        steps += 1
        out = decide_fn(info["raw_state"], mask, history)
        factored = out[0] if isinstance(out, tuple) else out
        actions.append(factored)
        _o, term, trunc, mask, info = env.step(factored)
        fps.append(fingerprint(info["raw_state"]))
        done = term or trunc
    return RunCapture(seed=seed, actions=actions, fingerprints=fps)


@dataclass
class ReplayResult:
    seed: str
    n_steps: int
    matched: bool
    first_divergence: int | None  # step index of first mismatch, or None
    detail: str = ""


def replay_and_compare(cap: RunCapture) -> ReplayResult:
    """Replay the captured action stream through a fresh env; compare fingerprints."""
    env = BalatroEnvironment(adapter_factory=DirectAdapter)
    _o, mask, info = env.reset(seed=cap.seed)
    if fingerprint(info["raw_state"]) != cap.fingerprints[0]:
        return ReplayResult(cap.seed, 0, False, 0, "initial state differs on reset")
    for i, factored in enumerate(cap.actions, start=1):
        try:
            _o, term, trunc, mask, info = env.step(factored)
        except Exception as e:  # noqa: BLE001 — an action going illegal on replay IS a desync signal
            return ReplayResult(cap.seed, i, False, i, f"step raised: {type(e).__name__}: {e}")
        if fingerprint(info["raw_state"]) != cap.fingerprints[i]:
            return ReplayResult(cap.seed, i, False, i, "state fingerprint mismatch")
        if (term or trunc) and i < len(cap.actions):
            return ReplayResult(cap.seed, i, False, i, "replay terminated early")
    return ReplayResult(cap.seed, len(cap.actions), True, None, "exact match")


# ---------------------------------------------------------------------------
# Replay a recorder JSONL run from its serialized factored actions.
# ---------------------------------------------------------------------------


def factored_from_dict(d: dict) -> FactoredAction:
    """Reconstruct a FactoredAction from its serialized (recorder) form."""
    ct = d.get("card_target")
    return FactoredAction(
        action_type=int(d["action_type"]),
        card_target=tuple(ct) if ct else None,
        entity_target=d.get("entity_target"),
    )


def _auto_action(raw_state: Any, mask: Any):
    """The deterministic transition the runner auto-applies (and never records)."""
    phase_u = str(raw_state.get("phase", "")).upper()
    if "BLIND_SELECT" in phase_u and mask.type_mask[_SELECT_BLIND]:
        return FactoredAction(action_type=_SELECT_BLIND)
    if "ROUND_EVAL" in phase_u:
        if mask.type_mask[_CASHOUT]:
            return FactoredAction(action_type=_CASHOUT)
        if mask.type_mask[_NEXT_ROUND]:
            return FactoredAction(action_type=_NEXT_ROUND)
    return None


def _play_recorded(seed: str, recorded_actions: list, max_steps: int = 5000):
    """Rebuild the full action stream from auto-plays and recorded policy actions."""
    env = BalatroEnvironment(adapter_factory=DirectAdapter)
    _o, mask, info = env.reset(seed=seed)
    fps = [fingerprint(info["raw_state"])]
    queue = list(recorded_actions)
    done = False
    steps = 0
    while not done and steps < max_steps:
        steps += 1
        auto = _auto_action(info["raw_state"], mask)
        if auto is not None:
            action = auto
        elif queue:
            action = factored_from_dict(queue.pop(0))
        else:
            break  # recorded policy decisions exhausted
        _o, term, trunc, mask, info = env.step(action)
        fps.append(fingerprint(info["raw_state"]))
        done = term or trunc
    return env, fps


@dataclass
class RecordedReplayResult:
    seed: str
    matched: bool
    first_divergence: int | None
    highest_ante: int
    detail: str = ""


def replay_recorded_run(record: dict) -> RecordedReplayResult:
    """Replay a recorder JSONL record twice; confirm determinism AND that it
    reproduces the recorded outcome (faithful action serialization)."""
    seed = record["meta"]["seed"]
    actions = [ev["factored_action"] for ev in record["events"] if "factored_action" in ev]
    env_a, fps_a = _play_recorded(seed, actions)
    _env_b, fps_b = _play_recorded(seed, actions)

    n = min(len(fps_a), len(fps_b))
    div = next((i for i in range(n) if fps_a[i] != fps_b[i]), None)
    if div is None and len(fps_a) != len(fps_b):
        div = n
    matched = div is None
    detail = "replays identical" if matched else "replay-vs-replay divergence"

    summ = record.get("summary", {})
    if matched and "highest_ante" in summ and int(env_a.episode_ante) != int(summ["highest_ante"]):
        matched, detail = (
            False,
            (f"outcome mismatch: replay ante={env_a.episode_ante} recorded={summ['highest_ante']}"),
        )
    return RecordedReplayResult(seed, matched, div, int(env_a.episode_ante), detail)
