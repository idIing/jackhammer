#!/usr/bin/env python
"""Inspect one raw Jackhammer JSONL run without the Balatro client.

Shows what the agent actually decided, and — just as importantly — which
decisions it did *not* make. ``build_decider`` substitutes a legal fallback for
any slot output that is illegal or raises, so a broken policy yields a complete,
normal-looking run. Every decision here is labelled with the reasoning the
harness recorded, and substituted ones are marked, so that failure mode is
visible to a reader instead of buried in the JSONL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.playground.metrics import fallback_stats  # noqa: E402


def load_runs(path: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number}: run must be a JSON object")
            runs.append(obj)
    if not runs:
        raise ValueError(f"{path}: no runs found")
    return runs


def _ante(run: dict[str, Any]) -> int:
    return int((run.get("summary") or {}).get("highest_ante", 1) or 1)


def _seed(run: dict[str, Any]) -> str:
    return str((run.get("meta") or {}).get("seed", "?"))


def _fallback(run: dict[str, Any]) -> dict[str, Any]:
    """Fallback-substitution stats for a single run (shared with the artifact)."""
    return fallback_stats([run])


def select_run(
    runs: list[dict[str, Any]], *, seed: str | None, index: int | None, worst: bool
) -> dict[str, Any]:
    if seed is not None:
        for run in runs:
            if _seed(run) == seed:
                return run
        raise ValueError(f"seed {seed!r} not found")
    if index is not None:
        try:
            return runs[index]
        except IndexError:
            raise ValueError(f"index {index} out of range for {len(runs)} runs") from None
    if worst:
        return min(runs, key=lambda run: (_ante(run), _seed(run)))
    return runs[0]


def render(run: dict[str, Any]) -> str:
    meta = run.get("meta") or {}
    summary = run.get("summary") or {}
    lines = [
        f"seed={_seed(run)}  agent={meta.get('config_label', '?')}",
        (
            f"highest_ante={_ante(run)}  won={bool(summary.get('won'))}  "
            f"reason={summary.get('terminal_reason', '?')}"
        ),
        (
            f"money=${summary.get('final_money', '?')}  "
            f"decisions={summary.get('n_decisions', len(run.get('events') or []))}"
        ),
    ]
    jokers = summary.get("final_jokers") or []
    if jokers:
        lines.append("jokers=" + ", ".join(map(str, jokers)))
    lines.extend(_fallback_lines(_fallback(run)))
    lines.append("\ndecisions:")
    for event in run.get("events") or []:
        detail = []
        for key in ("hand_type", "score", "item_kind", "label", "cost", "voucher"):
            if event.get(key) not in (None, ""):
                detail.append(f"{key}={event[key]}")
        suffix = "  " + " ".join(detail) if detail else ""
        reasoning = str(event.get("reasoning") or "?")
        mark = "!! FALLBACK " if event.get("was_fallback") else ""
        lines.append(
            f"  {int(event.get('step', 0)):>3}  ante={event.get('ante', '?')}  "
            f"{event.get('action', '?')}{suffix}  [{mark}{reasoning}]"
        )
    return "\n".join(lines)


def _fallback_lines(fb: dict[str, Any]) -> list[str]:
    """The substitution banner: silent when clean, explicit when not."""
    n, total = fb["n_fallback"], fb["n_decisions"]
    if not n:
        return [f"fallback=0/{total} decisions -- every decision came from the agent"]
    reasons = ", ".join(f"{k}={v}" for k, v in fb["by_reason"].items())
    lines = [
        f"fallback={n}/{total} decisions ({n / total:.1%})  {reasons}",
        f"  !! {n} decision(s) were substituted by the harness, not chosen by the agent.",
    ]
    if fb["errors"]:
        lines.append(
            f"  !! {fb['errors']} of those raised inside the policy (fallback-error:*). "
            "That is a bug in the agent;"
        )
        lines.append(
            "     this run does not measure the agent, and no number derived from it is about it."
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="raw run JSONL from scripts/evaluate.py")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--seed", help="select an exact game seed")
    selection.add_argument("--index", type=int, help="select a zero-based run index")
    selection.add_argument("--worst", action="store_true", help="select the shallowest run")
    selection.add_argument("--list", action="store_true", help="list run outcomes and exit")
    args = parser.parse_args()

    try:
        runs = load_runs(args.path)
        if args.list:
            for index, run in enumerate(runs):
                fb = _fallback(run)
                flag = ""
                if fb["n_fallback"]:
                    flag = f"  fallback={fb['n_fallback']}/{fb['n_decisions']}"
                print(f"{index:>4}  {_seed(run)}  ante={_ante(run)}{flag}")
            return 0
        print(render(select_run(runs, seed=args.seed, index=args.index, worst=args.worst)))
        return 0
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
