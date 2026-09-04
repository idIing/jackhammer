from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect_run.py"
SPEC = importlib.util.spec_from_file_location("inspect_run", SCRIPT)
assert SPEC and SPEC.loader
inspect_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspect_run)


def _run(seed: str, ante: int):
    return {
        "meta": {"seed": seed, "config_label": "probe"},
        "summary": {"highest_ante": ante, "won": False, "final_money": 3},
        "events": [{"step": 0, "ante": 1, "action": "PlayHand", "score": 42}],
    }


def test_select_worst_then_render():
    run = inspect_run.select_run(
        [_run("PVRQ4K5A", 3), _run("4NNGD2DN", 1)],
        seed=None,
        index=None,
        worst=True,
    )
    rendered = inspect_run.render(run)
    assert "seed=4NNGD2DN" in rendered
    assert "highest_ante=1" in rendered
    assert "score=42" in rendered


def test_missing_seed_is_explicit():
    with pytest.raises(ValueError, match="not found"):
        inspect_run.select_run([_run("PVRQ4K5A", 3)], seed="4NNGD2DN", index=None, worst=False)


def _broken_run(seed: str = "PVRQ4K5A"):
    """A run whose shop policy raised on every shop decision."""
    return {
        "meta": {"seed": seed, "config_label": "priciest-shop"},
        "summary": {"highest_ante": 1, "won": False, "final_money": 1},
        "events": [
            {"step": 0, "ante": 1, "action": "PlayHand", "score": 316,
             "reasoning": "greedy-tactical", "was_fallback": False},
            {"step": 1, "ante": 1, "action": "Reroll",
             "reasoning": "fallback-error:NameError", "was_fallback": True},
        ],
    }


def test_render_marks_substituted_decisions():
    """The whole point: a dead policy must not render as a normal timeline."""
    rendered = inspect_run.render(_broken_run())
    assert "fallback=1/2 decisions" in rendered
    assert "fallback-error:NameError=1" in rendered
    assert "!! FALLBACK fallback-error:NameError" in rendered
    assert "bug in the agent" in rendered
    # Non-fallback decisions still carry their reasoning, unmarked.
    assert "[greedy-tactical]" in rendered


def test_render_says_so_when_nothing_was_substituted():
    rendered = inspect_run.render(_run("PVRQ4K5A", 3))
    assert "every decision came from the agent" in rendered
    assert "FALLBACK" not in rendered
