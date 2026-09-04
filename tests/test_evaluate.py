"""Tests for the evaluator's fallback reporting.

The evaluator's progress line counts *crashed games*. A policy that raises on
every decision crashes no game -- ``build_decider`` substitutes a legal action --
so it reports zero failures and a well-formed, significant result. These tests
cover the line that makes that case visible instead.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("evaluate", SCRIPT)
assert SPEC and SPEC.loader
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


def test_report_fallbacks_flags_a_policy_that_raised(capsys):
    fb = {
        "n_decisions": 180,
        "n_fallback": 40,
        "rate": 40 / 180,
        "runs_affected": 7,
        "n_runs": 8,
        "errors": 40,
        "by_reason": {"fallback-error:NameError": 40},
    }
    assert evaluate._report_fallbacks("priciest-shop", fb) is True
    out = capsys.readouterr().out
    assert "40 / 180 decisions (22.2%)" in out
    assert "7/8 runs" in out
    assert "fallback-error:NameError=40" in out
    assert "made by the harness, not by priciest-shop" in out


def test_report_fallbacks_is_quiet_but_explicit_when_clean(capsys):
    fb = {
        "n_decisions": 345,
        "n_fallback": 0,
        "rate": 0.0,
        "runs_affected": 0,
        "n_runs": 8,
        "errors": 0,
        "by_reason": {},
    }
    assert evaluate._report_fallbacks("greedy-shop", fb) is False
    assert "fallback substitutions: 0 / 345 decisions" in capsys.readouterr().out


def test_illegal_actions_are_reported_without_being_called_a_policy_error(capsys):
    """An illegal action is still a substitution, but it is not a raised exception."""
    fb = {
        "n_decisions": 100,
        "n_fallback": 3,
        "rate": 0.03,
        "runs_affected": 2,
        "n_runs": 8,
        "errors": 0,
        "by_reason": {"fallback-illegal": 3},
    }
    assert evaluate._report_fallbacks("some-agent", fb) is False
    out = capsys.readouterr().out
    assert "fallback-illegal=3" in out
    assert "ERROR" not in out
