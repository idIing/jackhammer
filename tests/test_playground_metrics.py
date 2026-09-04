"""Tests for the inferential playground metrics layer.

Covers the pure-math primitives (Wilson CI, McNemar) against hand-checked
reference values, and the record-consuming metrics (advance curve, paired
compare) against small synthetic JSONL fixtures.
"""

from __future__ import annotations

import json

from src.playground import metrics


# --------------------------------------------------------------------- wilson
def test_wilson_ci_hand_checked():
    """Wilson 95% (z=1.96) reference values, derived by hand from

        center = (p + z²/2n) / (1 + z²/n)
        margin = (z/(1+z²/n)) * sqrt(p(1-p)/n + z²/4n²)

    0/100:   p=0    -> (0.0000, 0.0370)
    50/100:  p=0.5  -> (0.4038, 0.5962)
    100/100: p=1    -> (0.9630, 1.0000)
    """
    lo, hi = metrics.wilson_ci(0, 100)
    assert abs(lo - 0.0) < 1e-3
    assert abs(hi - 0.0370) < 1e-3

    lo, hi = metrics.wilson_ci(50, 100)
    assert abs(lo - 0.4038) < 1e-3
    assert abs(hi - 0.5962) < 1e-3

    lo, hi = metrics.wilson_ci(100, 100)
    assert abs(lo - 0.9630) < 1e-3
    assert abs(hi - 1.0) < 1e-3


def test_wilson_ci_n_zero():
    # No data ⇒ maximally uncertain interval, no crash.
    assert metrics.wilson_ci(0, 0) == (0.0, 1.0)


def test_wilson_ci_bounds_clamped():
    lo, hi = metrics.wilson_ci(3, 3)
    assert 0.0 <= lo <= hi <= 1.0


# -------------------------------------------------------------------- mcnemar
def test_mcnemar_toy_discordant():
    """b=10, c=0 ⇒ exact two-sided binomial p = 2 * C(10,0) * 0.5^10
    = 2 / 1024 = 0.001953125."""
    p = metrics.mcnemar(10, 0)
    assert abs(p - 0.001953125) < 1e-9


def test_mcnemar_balanced_is_one():
    # b == c ⇒ no asymmetry; two-sided exact p saturates at 1.0.
    assert metrics.mcnemar(5, 5) == 1.0


def test_mcnemar_degenerate_no_discordance():
    assert metrics.mcnemar(0, 0) == 1.0


def test_mcnemar_symmetric():
    assert metrics.mcnemar(8, 1) == metrics.mcnemar(1, 8)


def test_mcnemar_known_small_value():
    """b=8, c=1, n=9 ⇒ p = 2*(C(9,0)+C(9,1))*0.5^9 = 2*10/512 = 0.0390625."""
    assert abs(metrics.mcnemar(8, 1) - 0.0390625) < 1e-9


def test_mcnemar_large_uses_chisquare():
    # b+c >= 25 path: continuity-corrected chi-square, df=1.
    # b=30,c=0: stat = (|30|-1)^2/30 = 841/30 = 28.0333 -> tiny p.
    p = metrics.mcnemar(30, 0)
    assert 0.0 < p < 1e-6


# ------------------------------------------------------------------ fixtures
def _run(
    seed,
    highest_ante,
    won,
    blinds=None,
    config_label="cfgA",
    slot1="RandomShop",
    slot2="MarginValue",
):
    return {
        "meta": {
            "seed": seed,
            "slot1": slot1,
            "slot2": slot2,
            "config_label": config_label,
        },
        "events": [],
        "tool_calls": [],
        "blinds": blinds or [],
        "summary": {"highest_ante": highest_ante, "won": won},
    }


def _blind(blind_type, target, realized, cleared, ante=1):
    return {
        "ante": ante,
        "blind_type": blind_type,
        "boss_key": "bl_hook" if blind_type == "Boss" else None,
        "target": target,
        "realized_score": realized,
        "margin": realized / target,
        "cleared": cleared,
        "hands_used": 2,
        "hands_left": 2,
        "discards_left": 1,
        "played_hand_histogram": {"Pair": 1},
    }


def _write_jsonl(path, runs):
    with open(path, "w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")


# ------------------------------------------------------------------ load/depth
def test_load_runs_roundtrip(tmp_path):
    runs = [_run("AAAA1111", 2, False), _run("BBBB2222", 3, True)]
    p = tmp_path / "battery.jsonl"
    _write_jsonl(p, runs)
    loaded = metrics.load_runs(str(p))
    assert len(loaded) == 2
    assert loaded[0]["meta"]["seed"] == "AAAA1111"


def test_run_depth_distribution():
    runs = [
        _run("s1", 1, False),
        _run("s2", 2, False),
        _run("s3", 2, False),
        _run("s4", 4, True),
    ]
    rd = metrics.run_depth(runs)
    assert rd["max"] == 4
    assert rd["median"] == 2.0
    assert rd["distribution"] == {1: 1, 2: 2, 4: 1}


# --------------------------------------------------------------- advance curve
def test_advance_curve_counts_and_rates():
    # highest antes: 1,2,2,3  -> from A=1 all 4 reach; 3 reach A>=2; 1 reaches A>=3.
    runs = [
        _run("s1", 1, False),
        _run("s2", 2, False),
        _run("s3", 2, False),
        _run("s4", 3, False),
    ]
    curve = metrics.advance_curve(runs)
    by_ante = {row["ante"]: row for row in curve}
    assert by_ante[1]["n_reached"] == 4
    assert by_ante[1]["n_advanced"] == 3  # antes >= 2
    assert abs(by_ante[1]["rate"] - 0.75) < 1e-9
    assert by_ante[2]["n_reached"] == 3
    assert by_ante[2]["n_advanced"] == 1  # antes >= 3
    assert abs(by_ante[2]["rate"] - (1 / 3)) < 1e-9
    # deepest ante row: reached by 1, advanced by 0.
    assert by_ante[3]["n_reached"] == 1
    assert by_ante[3]["n_advanced"] == 0
    assert by_ante[3]["rate"] == 0.0
    # CI present and ordered for every row.
    for row in curve:
        assert 0.0 <= row["ci_lo"] <= row["ci_hi"] <= 1.0


def test_advance_curve_empty():
    assert metrics.advance_curve([]) == []


# ------------------------------------------------------------------- blinds
def test_blind_stats_overall_and_by_type():
    runs = [
        _run(
            "s1",
            2,
            False,
            blinds=[
                _blind("Small", 300, 450, True),
                _blind("Big", 600, 300, False),
            ],
        ),
        _run(
            "s2",
            2,
            False,
            blinds=[
                _blind("Small", 300, 600, True),
                _blind("Boss", 1200, 1200, True),
            ],
        ),
    ]
    bs = metrics.blind_stats(runs)
    ov = bs["overall"]
    assert ov["n"] == 4
    assert ov["n_cleared"] == 3
    assert abs(ov["clear_rate"] - 0.75) < 1e-9
    small = bs["by_blind_type"]["Small"]
    assert small["n"] == 2 and small["n_cleared"] == 2
    big = bs["by_blind_type"]["Big"]
    assert big["clear_rate"] == 0.0


# ------------------------------------------------------------------ paired A/B
def test_paired_join_uses_only_shared_seeds():
    runs_a = [_run("s1", 2, False), _run("s2", 3, False), _run("only_a", 1, False)]
    runs_b = [_run("s1", 3, False), _run("s2", 2, False), _run("only_b", 5, True)]
    paired = metrics.paired_join(runs_a, runs_b)
    seeds = [s for s, _a, _b in paired]
    assert seeds == ["s1", "s2"]  # sorted, intersection only


def test_paired_depth_delta_per_seed_correct():
    runs_a = [_run("s1", 2, False), _run("s2", 3, False)]
    runs_b = [_run("s1", 3, False), _run("s2", 2, False)]
    paired = metrics.paired_join(runs_a, runs_b)
    dd = metrics.paired_depth_delta(paired)
    by_seed = {d["seed"]: d for d in dd["per_seed"]}
    assert by_seed["s1"]["delta"] == 1  # B(3) - A(2)
    assert by_seed["s2"]["delta"] == -1  # B(2) - A(3)
    assert abs(dd["mean_delta"] - 0.0) < 1e-9
    lo, hi = dd["bootstrap_ci"]
    assert lo <= 0.0 <= hi


def test_compare_structure_and_mcnemar(tmp_path):
    # Arm B strictly deeper on every shared seed (advances past ante 2),
    # arm A never reaches ante 2 ⇒ McNemar reached>=2 is maximally discordant.
    runs_a = [_run(f"s{i}", 1, False, config_label="A") for i in range(6)]
    runs_b = [_run(f"s{i}", 3, False, config_label="B") for i in range(6)]
    pa = tmp_path / "a.jsonl"
    pb = tmp_path / "b.jsonl"
    _write_jsonl(pa, runs_a)
    _write_jsonl(pb, runs_b)

    cmp = metrics.compare(metrics.load_runs(str(pa)), metrics.load_runs(str(pb)))
    assert cmp["n_paired"] == 6
    assert cmp["arm_a"]["run_depth"]["mean"] == 1.0
    assert cmp["arm_b"]["run_depth"]["mean"] == 3.0
    assert abs(cmp["depth_delta"]["mean_delta"] - 2.0) < 1e-9

    rk = cmp["mcnemar"]["reached_k"]
    assert rk["b"] == 0  # A-only positives
    assert rk["c"] == 6  # B-only positives (all shared seeds)
    # exact binomial, b=0,c=6 ⇒ 2 * 0.5^6 = 0.03125
    assert abs(rk["p_value"] - 0.03125) < 1e-9

    # won is False everywhere ⇒ no discordance ⇒ p=1.0
    assert cmp["mcnemar"]["won"]["p_value"] == 1.0


# ------------------------------------------------------------------- fallback
def _ev(reasoning: str, was_fallback: bool):
    return {"step": 0, "ante": 1, "action": "NextRound", "reasoning": reasoning,
            "was_fallback": was_fallback}


def test_fallback_stats_counts_substitutions_and_isolates_errors():
    """A policy that raises must be countable, and separable from benign fallbacks."""
    runs = [
        {"events": [_ev("GreedyShop", False), _ev("fallback-error:NameError", True)]},
        {"events": [_ev("greedy-tactical", False), _ev("fallback-illegal", True),
                    _ev("fallback-error:IndexError", True)]},
        {"events": [_ev("greedy-tactical", False)]},
    ]
    fb = metrics.fallback_stats(runs)
    assert fb["n_decisions"] == 6
    assert fb["n_fallback"] == 3
    assert abs(fb["rate"] - 0.5) < 1e-9
    assert fb["runs_affected"] == 2
    assert fb["n_runs"] == 3
    # Only raised-inside-the-policy substitutions count as errors; illegal does not.
    assert fb["errors"] == 2
    assert fb["by_reason"] == {
        "fallback-error:NameError": 1,
        "fallback-illegal": 1,
        "fallback-error:IndexError": 1,
    }


def test_fallback_stats_clean_battery_is_zero():
    runs = [{"events": [_ev("GreedyShop", False), _ev("greedy-tactical", False)]}]
    fb = metrics.fallback_stats(runs)
    assert fb["n_fallback"] == 0
    assert fb["rate"] == 0.0
    assert fb["by_reason"] == {}


def test_summarize_carries_the_fallback_block():
    """The published artifact must not be able to hide a dead policy."""
    runs = [
        {
            "meta": {"seed": "PVRQ4K5A", "config_label": "probe"},
            "summary": {"highest_ante": 1, "won": False},
            "events": [_ev("fallback-error:NameError", True)],
        }
    ]
    summary = metrics.summarize(runs)
    assert summary["fallback"]["errors"] == 1
    assert summary["fallback"]["n_fallback"] == 1
