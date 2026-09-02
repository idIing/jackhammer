"""Shared episode loop for self-play.

Auto-plays the deterministic transitions (blind select, cash-out / next-round)
and delegates decision phases to a pluggable ``decide_fn``. A ``RunRecorder`` hook
turns every decision into a recorded event.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from jackdaw.env import FactoredAction

from src.selfplay.recorder import ACTION_NAMES, RunMeta, RunRecorder

# decide_fn(raw_state, mask, history) -> (factored, reasoning, method, params, was_fallback)
DecideFn = Callable[[dict, Any, list], tuple]

_SELECT_BLIND, _CASHOUT, _NEXT_ROUND = 2, 4, 6


@dataclass
class RunResult:
    highest_ante: int
    won: bool
    episode_length: int
    fallback_occurred: bool
    bought_joker: bool
    terminal_reason: str
    final_raw_state: dict


def random_decider(rng: random.Random | None = None) -> DecideFn:
    """A simple baseline: samples a legal action type uniformly, then a legal target.

    Also provides a CPU-only smoke path for the recorder and metrics stack.
    """
    rng = rng or random.Random()

    def decide(raw_state, mask, history):
        legal = [int(t) for t in np.nonzero(mask.type_mask)[0]]
        rng.shuffle(legal)
        for at in legal:
            card_target = None
            entity_target = None
            if at in (0, 1):  # PlayHand / Discard need cards
                cards = [int(c) for c in np.nonzero(mask.card_mask)[0]]
                if not cards:
                    continue
                lo = max(1, int(mask.min_card_select))
                hi = min(int(mask.max_card_select), len(cards))
                if hi < lo:
                    continue
                card_target = tuple(rng.sample(cards, rng.randint(lo, hi)))
            elif at in mask.entity_masks:
                ents = [int(e) for e in np.nonzero(mask.entity_masks[at])[0]]
                if not ents:
                    continue
                entity_target = rng.choice(ents)
            factored = FactoredAction(
                action_type=at, card_target=card_target, entity_target=entity_target
            )
            return factored, "random", ACTION_NAMES.get(at, str(at)), {}, False
        # Nothing constructable (shouldn't happen with a valid mask).
        at = legal[0]
        return (
            FactoredAction(action_type=at),
            "random-degenerate",
            ACTION_NAMES.get(at, str(at)),
            {},
            True,
        )

    return decide


def play_episode(
    env,
    seed,
    decide_fn: DecideFn,
    recorder: RunRecorder | None = None,
    run_meta: RunMeta | None = None,
    max_steps: int = 200,
    verbose: bool = False,
) -> RunResult:
    """Run one episode end to end, recording each decision if a recorder is given."""
    _obs, mask, info = env.reset(seed=seed)
    if recorder is not None and run_meta is not None:
        recorder.start_run(run_meta)

    done = terminated = truncated = False
    step_count = 0
    history: list = []
    fallback_occurred = False
    bought_joker = False
    last_raw = info["raw_state"]

    while not done and step_count < max_steps:
        step_count += 1
        raw_state = info["raw_state"]
        last_raw = raw_state
        phase_u = str(raw_state.get("phase", "")).upper()

        auto = None
        if "BLIND_SELECT" in phase_u and mask.type_mask[_SELECT_BLIND]:
            auto = FactoredAction(action_type=_SELECT_BLIND)
        elif "ROUND_EVAL" in phase_u:
            if mask.type_mask[_CASHOUT]:
                auto = FactoredAction(action_type=_CASHOUT)
            elif mask.type_mask[_NEXT_ROUND]:
                auto = FactoredAction(action_type=_NEXT_ROUND)
        if auto is not None:
            _obs, terminated, truncated, mask, info = env.step(auto)
            done = terminated or truncated
            last_raw = info["raw_state"]
            continue

        if len(raw_state.get("jokers", []) or []) > 0:
            bought_joker = True

        factored, reasoning, method, params, was_fallback = decide_fn(raw_state, mask, history)
        if was_fallback:
            fallback_occurred = True

        prev_raw = raw_state
        _obs, terminated, truncated, mask, info = env.step(factored)
        done = terminated or truncated
        last_raw = info["raw_state"]

        if recorder is not None:
            recorder.record_step(
                prev_raw,
                info["raw_state"],
                factored,
                method,
                params,
                reasoning,
                was_fallback,
            )
        if not was_fallback and method:
            history.append(
                {
                    "method": method,
                    "params": json.dumps(params) if params else "",
                    "reasoning": reasoning,
                }
            )
        if verbose:
            ante = (raw_state.get("round_resets") or {}).get("ante", "?")
            print(
                f"  step {step_count}: {ACTION_NAMES.get(int(factored.action_type))} (ante {ante})"
            )

    terminal_reason = "terminated" if terminated else ("truncated" if truncated else "max_steps")
    result = RunResult(
        highest_ante=int(env.episode_ante),
        won=bool(env.episode_won),
        episode_length=int(env.episode_length),
        fallback_occurred=fallback_occurred,
        bought_joker=bought_joker,
        terminal_reason=terminal_reason,
        final_raw_state=last_raw,
    )
    if recorder is not None and run_meta is not None:
        recorder.finish_run(
            result.highest_ante,
            result.won,
            result.episode_length,
            result.fallback_occurred,
            result.terminal_reason,
            result.final_raw_state,
        )
    return result
