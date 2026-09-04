"""Inferential metrics over playground battery JSONL records.

Consumes the raw-run record shape described in ``docs/result-artifacts.md``
(``meta`` with ``seed``/``slot1``/``slot2``/``config_label``, ``summary`` with
``highest_ante``/``won``, and a top-level ``blinds[]`` array) and computes:

- run depth (mean/median ``highest_ante`` + distribution),
- blind-clear rate and score-margin distribution (overall and per ``blind_type``),
- win rate with a **Wilson 95% CI**,
- the **conditional-advance curve** with per-ante Wilson CIs (the primary
  "is it improving" signal when the win-rate CI still includes zero),
- **fallback substitution rate**: how often the harness replaced a policy's
  decision with a legal fallback (the audit that says a reported number came
  from the policy under test and not from the harness standing in for it),
- **paired A/B** on identical seeds: per-seed depth deltas (mean + bootstrap CI),
  an advance-curve overlay, and a McNemar paired-significance read.

This is the *inferential* layer, kept separate from the descriptive per-run
records it consumes. **scipy is not a dependency**: Wilson, McNemar, and the bootstrap are
implemented in pure Python / numpy.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

import numpy as np

__all__ = [
    "load_runs",
    "wilson_ci",
    "mcnemar",
    "bootstrap_ci",
    "run_depth",
    "blind_stats",
    "win_rate",
    "advance_curve",
    "fallback_stats",
    "paired_join",
    "paired_depth_delta",
    "summarize",
    "compare",
]


# --------------------------------------------------------------------------- io
def load_runs(path: str) -> list[dict]:
    """Read one JSONL battery file into a list of run-record dicts.

    Blank lines are skipped. Each non-blank line must be a single JSON object in
    the record shape described in ``docs/result-artifacts.md``.
    """
    runs: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


# ------------------------------------------------------------------ statistics
def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (pure Python).

    For ``successes`` out of ``n`` with critical value ``z`` (default 1.96 ⇒ 95%):

        center = (p + z²/2n) / (1 + z²/n)
        margin = (z / (1 + z²/n)) * sqrt(p(1-p)/n + z²/4n²)

    Returns ``(lo, hi)`` clamped to ``[0, 1]``. With ``n == 0`` there is no
    information, so the maximally-uncertain ``(0.0, 1.0)`` is returned.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)


def _binom_coeff(n: int, k: int) -> int:
    return math.comb(n, k)


def mcnemar(b: int, c: int) -> float:
    """McNemar's paired test, returning a two-sided p-value (pure Python).

    ``b`` and ``c`` are the **discordant** counts of the paired 2x2 table
    (pairs where exactly one arm succeeded). The test is symmetric in ``b``/``c``.

    - ``b + c == 0`` (no discordance) ⇒ ``p = 1.0``.
    - small discordant total (``b + c < 25``) ⇒ **exact binomial**: under H0
      ``b ~ Binomial(b+c, 0.5)``, two-sided
      ``p = min(1, 2 * sum_{k=0}^{min(b,c)} C(n,k) 0.5^n)``.
    - otherwise ⇒ chi-square with **continuity correction**,
      ``stat = (|b-c|-1)² / (b+c)``, ``df=1``; the survival function for
      ``df=1`` is ``erfc(sqrt(stat/2))`` (no scipy needed).
    """
    b = int(b)
    c = int(c)
    n = b + c
    if n == 0:
        return 1.0
    if n < 25:
        lo = min(b, c)
        tail = sum(_binom_coeff(n, k) for k in range(lo + 1))
        p = 2.0 * tail * (0.5**n)
        return min(1.0, p)
    stat = (abs(b - c) - 1) ** 2 / n
    return math.erfc(math.sqrt(stat / 2.0))


def bootstrap_ci(
    values: list[float],
    ci: float = 0.95,
    n_boot: int = 10000,
    seed: int = 12345,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the **mean** of ``values`` (numpy).

    Deterministic given ``seed`` so tests and reports are reproducible. Returns
    ``(nan, nan)`` for an empty sample.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return (lo, hi)


# ----------------------------------------------------------------- accessors
def _highest_ante(run: dict) -> int:
    return int((run.get("summary") or {}).get("highest_ante", 1) or 1)


def _won(run: dict) -> bool:
    """Benchmark win = cleared the ante-8 (White Stake) boss, from ground truth.

    NOT ``summary.won``: the engine's ``gs["won"]`` flag is non-latching — it is
    reset to False on any later blind failure (game.py:658), so a run that beats the
    ante-8 boss and then dies in endless ante 9 reports ``won=False``. We instead
    derive the win from the per-blind records (a cleared Boss at ante >= win_ante),
    OR-ed with the raw flag for robustness.
    """
    summary = run.get("summary") or {}
    if summary.get("won"):
        return True
    for b in run.get("blinds") or []:
        if b.get("cleared") and b.get("blind_type") == "Boss" and int(b.get("ante", 0) or 0) >= 8:
            return True
    return False


def _seed(run: dict) -> str | None:
    return (run.get("meta") or {}).get("seed")


# ------------------------------------------------------------------- metrics
def run_depth(runs: list[dict]) -> dict[str, Any]:
    """Mean/median/max ``highest_ante`` and the count-per-ante distribution."""
    antes = [_highest_ante(r) for r in runs]
    if not antes:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "max": None,
            "distribution": {},
        }
    dist = dict(sorted(Counter(antes).items()))
    return {
        "n": len(antes),
        "mean": float(np.mean(antes)),
        "median": float(np.median(antes)),
        "max": int(max(antes)),
        "distribution": dist,
    }


def _margin_summary(margins: list[float], cleared: list[bool]) -> dict[str, Any]:
    n = len(margins)
    n_cleared = int(sum(1 for x in cleared if x))
    out: dict[str, Any] = {
        "n": n,
        "n_cleared": n_cleared,
        "clear_rate": (n_cleared / n) if n else float("nan"),
    }
    if n:
        arr = np.asarray(margins, dtype=float)
        out.update(
            margin_mean=float(arr.mean()),
            margin_median=float(np.median(arr)),
            margin_min=float(arr.min()),
            margin_max=float(arr.max()),
        )
        lo, hi = wilson_ci(n_cleared, n)
        out["clear_ci_lo"] = lo
        out["clear_ci_hi"] = hi
    return out


def blind_stats(runs: list[dict]) -> dict[str, Any]:
    """Blind-clear rate and score-margin distribution from ``blinds[]``.

    Returns an ``overall`` summary plus a ``by_blind_type`` map keyed by
    ``"Small"``/``"Big"``/``"Boss"``.
    """
    all_margins: list[float] = []
    all_cleared: list[bool] = []
    by_type: dict[str, dict[str, list]] = {}
    for r in runs:
        for blind in r.get("blinds") or []:
            margin = blind.get("margin")
            cleared = bool(blind.get("cleared", False))
            if margin is None:
                continue
            margin = float(margin)
            all_margins.append(margin)
            all_cleared.append(cleared)
            bt = blind.get("blind_type") or "Unknown"
            bucket = by_type.setdefault(bt, {"margins": [], "cleared": []})
            bucket["margins"].append(margin)
            bucket["cleared"].append(cleared)
    return {
        "overall": _margin_summary(all_margins, all_cleared),
        "by_blind_type": {
            bt: _margin_summary(v["margins"], v["cleared"]) for bt, v in sorted(by_type.items())
        },
    }


def win_rate(runs: list[dict]) -> dict[str, Any]:
    """Win rate with a Wilson 95% CI."""
    n = len(runs)
    wins = int(sum(1 for r in runs if _won(r)))
    lo, hi = wilson_ci(wins, n)
    return {
        "wins": wins,
        "n": n,
        "rate": (wins / n) if n else float("nan"),
        "ci_lo": lo,
        "ci_hi": hi,
    }


def advance_curve(runs: list[dict]) -> list[dict[str, Any]]:
    """Conditional-advance curve with Wilson CIs.

    For each ante ``A`` in the observed range, of runs with ``highest_ante >= A``,
    the fraction with ``highest_ante >= A+1``. Each row:
    ``{ante, n_reached, n_advanced, rate, ci_lo, ci_hi}``.
    """
    antes = [_highest_ante(r) for r in runs]
    if not antes:
        return []
    lo_a, hi_a = min(antes), max(antes)
    rows: list[dict[str, Any]] = []
    for a in range(lo_a, hi_a + 1):
        n_reached = sum(1 for x in antes if x >= a)
        n_advanced = sum(1 for x in antes if x >= a + 1)
        ci_lo, ci_hi = wilson_ci(n_advanced, n_reached)
        rows.append(
            {
                "ante": a,
                "n_reached": n_reached,
                "n_advanced": n_advanced,
                "rate": (n_advanced / n_reached) if n_reached else None,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            }
        )
    return rows


def fallback_stats(runs: list[dict]) -> dict[str, Any]:
    """How often the harness substituted a legal fallback for the policy's choice.

    ``build_decider`` legality-gates every slot output and catches every slot
    exception, replacing either with a legal fallback so one bad state cannot kill
    a 240-seed battery. That robustness is deliberate, but it is also silent: a
    policy that raises on every decision still produces a complete, plausible,
    fully significant battery. This is the statistic that makes the substitution
    visible, so a reader can tell whether a reported number came from the agent
    under test or from the harness standing in for it.

    Reads ``events[].was_fallback`` / ``events[].reasoning`` from the raw records::

        {n_decisions, n_fallback, rate, runs_affected, n_runs, by_reason, errors}

    ``by_reason`` counts only substituted decisions, keyed by the harness reason
    (``fallback-error:<ExcType>``, ``fallback-illegal``, ``fallback-phase``).
    ``errors`` is the subtotal for ``fallback-error:*`` alone — a raised exception
    inside a policy is always a defect in that policy, never a property of the run.
    """
    n_decisions = n_fallback = runs_affected = errors = 0
    by_reason: Counter[str] = Counter()
    for run in runs:
        hit = False
        for event in run.get("events") or []:
            n_decisions += 1
            if not event.get("was_fallback"):
                continue
            n_fallback += 1
            hit = True
            reason = str(event.get("reasoning") or "fallback-unlabelled")
            by_reason[reason] += 1
            if reason.startswith("fallback-error:"):
                errors += 1
        runs_affected += hit
    return {
        "n_decisions": n_decisions,
        "n_fallback": n_fallback,
        "rate": (n_fallback / n_decisions) if n_decisions else float("nan"),
        "runs_affected": runs_affected,
        "n_runs": len(runs),
        "errors": errors,
        "by_reason": dict(by_reason.most_common()),
    }


def summarize(runs: list[dict]) -> dict[str, Any]:
    """Single-config structured summary (all metrics for one battery)."""
    meta = (runs[0].get("meta") or {}) if runs else {}
    return {
        "n_runs": len(runs),
        "config_label": meta.get("config_label", ""),
        "slot1": meta.get("slot1", ""),
        "slot2": meta.get("slot2", ""),
        "run_depth": run_depth(runs),
        "blind_stats": blind_stats(runs),
        "win_rate": win_rate(runs),
        "advance_curve": advance_curve(runs),
        "fallback": fallback_stats(runs),
    }


# ----------------------------------------------------------------- paired A/B
def paired_join(runs_a: list[dict], runs_b: list[dict]) -> list[tuple[str, dict, dict]]:
    """Join two run lists on ``meta.seed``; return only seeds present in both.

    First occurrence of each seed wins within an arm (records are 1-per-seed by
    contract). Result is sorted by seed for determinism.
    """
    a_by_seed: dict[str, dict] = {}
    for r in runs_a:
        s = _seed(r)
        if s is not None and s not in a_by_seed:
            a_by_seed[s] = r
    b_by_seed: dict[str, dict] = {}
    for r in runs_b:
        s = _seed(r)
        if s is not None and s not in b_by_seed:
            b_by_seed[s] = r
    shared = sorted(set(a_by_seed) & set(b_by_seed))
    return [(s, a_by_seed[s], b_by_seed[s]) for s in shared]


def paired_depth_delta(
    paired: list[tuple[str, dict, dict]],
) -> dict[str, Any]:
    """Per-seed ``highest_ante`` delta (arm B minus arm A) + bootstrap 95% CI."""
    per_seed = [
        {
            "seed": s,
            "a": _highest_ante(ra),
            "b": _highest_ante(rb),
            "delta": _highest_ante(rb) - _highest_ante(ra),
        }
        for s, ra, rb in paired
    ]
    deltas = [d["delta"] for d in per_seed]
    mean_delta = float(np.mean(deltas)) if deltas else float("nan")
    lo, hi = bootstrap_ci(deltas) if deltas else (float("nan"), float("nan"))
    return {
        "per_seed": per_seed,
        "mean_delta": mean_delta,
        "bootstrap_ci": [lo, hi],
    }


def _mcnemar_on(
    paired: list[tuple[str, dict, dict]],
    outcome,
) -> dict[str, Any]:
    """McNemar discordant counts + p-value for a per-seed binary ``outcome(run)``.

    ``b`` = seeds where only arm A is positive; ``c`` = only arm B is positive.
    """
    b = c = a_pos = d_neg = 0
    for _s, ra, rb in paired:
        oa = bool(outcome(ra))
        ob = bool(outcome(rb))
        if oa and ob:
            a_pos += 1
        elif oa and not ob:
            b += 1
        elif ob and not oa:
            c += 1
        else:
            d_neg += 1
    return {
        "both": a_pos,
        "a_only": b,
        "b_only": c,
        "neither": d_neg,
        "b": b,
        "c": c,
        "p_value": mcnemar(b, c),
    }


def compare(
    runs_a: list[dict],
    runs_b: list[dict],
    advance_k: int = 2,
) -> dict[str, Any]:
    """Paired A/B comparison on identical seeds.

    Joins by ``meta.seed`` (only shared seeds), then returns per-arm summaries on
    the paired subset, the per-seed depth delta with bootstrap CI, an
    advance-curve overlay, and McNemar reads on two paired binary outcomes:
    ``won`` and ``reached ante >= advance_k``.
    """
    paired = paired_join(runs_a, runs_b)
    paired_a = [ra for _s, ra, _rb in paired]
    paired_b = [rb for _s, _ra, rb in paired]
    return {
        "n_paired": len(paired),
        "seeds": [s for s, _ra, _rb in paired],
        "arm_a": summarize(paired_a),
        "arm_b": summarize(paired_b),
        "depth_delta": paired_depth_delta(paired),
        "advance_overlay": {
            "a": advance_curve(paired_a),
            "b": advance_curve(paired_b),
        },
        "advance_k": advance_k,
        "mcnemar": {
            "won": _mcnemar_on(paired, _won),
            "reached_k": _mcnemar_on(paired, lambda r: _highest_ante(r) >= advance_k),
        },
    }
