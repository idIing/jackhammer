"""What an agent *did*, checked against what its description *claims* it does.

Every gate in this kit terminates at a number or at code correctness — determinism,
the engine tests, ruff, paired intervals. All of them pass on an agent that only ever
does four things, and none of them read the decision stream the recorder writes on
every run. That is how ``docs/known-limits.md`` shipped a clause describing a branch of
``greedy-shop`` that cannot execute: the description was written by reading the code,
and reading code cannot find a run-time fact.

So the claim is made machine-readable — ``AgentSpec.declared_actions``, the action types
an agent says it may emit — and this module compares it against the recorded events:

* **undeclared**: an action the agent emitted that its description does not cover. Wrong
  at any sample size, so ``evaluate.py`` fails on it unconditionally.
* **unexercised**: an action the description claims that the run never contains. Only
  measurable over a complete battery — on eight seeds it means nothing — so it is an
  error there and a warning otherwise.

Both read decisions the agent *chose*. A decision the recorder marks ``was_fallback`` is the
harness substituting for an agent that produced no legal action, so it is counted in the
histogram (it happened) and excluded from the two differences (the agent did not choose it).
All three baselines fall back zero times over the battery, so today the two sets are identical.

``scan_source`` is the same question asked of the source instead of the runs: which actions can
this policy *construct at all*. It runs in milliseconds with no engine and no games, so it belongs
in CI, where the battery check cannot go. The two bound the answer from opposite sides — the
source gives a ceiling, the runs give a floor — and an action in the ceiling but not the floor is
code that cannot be reached. That is exactly the pack-choice branch this module was written for.

An agent that declares nothing (``declared_actions=None``, the default for submitted
agents) is only reported on, never failed. The gate is for published claims.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jackhammer.selfplay.recorder import ACTION_NAMES

# Every action type the engine can offer, in action-type order. The single source is
# the recorder's own table, so a name can never be declared that a record cannot hold.
ALL_ACTIONS: tuple[str, ...] = tuple(ACTION_NAMES[i] for i in sorted(ACTION_NAMES))


def check_names(names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return *names* in action-type order, raising on anything not an action type.

    A typo in a declaration would otherwise show up as a permanent ``unexercised``
    finding against an agent that is behaving correctly.
    """
    unknown = [n for n in names if n not in ALL_ACTIONS]
    if unknown:
        raise ValueError(f"not action type name(s): {unknown}; expected from {list(ALL_ACTIONS)}")
    seen = set(names)
    return tuple(a for a in ALL_ACTIONS if a in seen)


def observe(runs: list[dict[str, Any]]) -> Counter[str]:
    """Histogram of action names over every decision in *runs*.

    Reads ``events[].action``, the name the recorder already stores beside the
    numeric type, so this works on any JSONL the benchmark has ever written.
    """
    hist: Counter[str] = Counter()
    for run in runs:
        for ev in run.get("events") or []:
            hist[str(ev.get("action", "?"))] += 1
    return hist


def _chosen(runs: list[dict[str, Any]]) -> Counter[str]:
    """Histogram over decisions the agent itself made (``was_fallback`` false)."""
    hist: Counter[str] = Counter()
    for run in runs:
        for ev in run.get("events") or []:
            if not ev.get("was_fallback"):
                hist[str(ev.get("action", "?"))] += 1
    return hist


def count_fallbacks(runs: list[dict[str, Any]]) -> int:
    """Decisions the harness substituted for because the agent produced no legal action."""
    return sum(1 for run in runs for ev in (run.get("events") or []) if ev.get("was_fallback"))


def audit(runs: list[dict[str, Any]], declared: tuple[str, ...] | None) -> dict[str, Any]:
    """Compare the recorded action types against *declared*.

    ``declared=None`` (an agent that publishes no claim) yields the observation with
    ``declared``/``undeclared``/``unexercised`` all ``None``: nothing to contradict.
    """
    hist = observe(runs)
    chosen = _chosen(runs)
    out: dict[str, Any] = {
        "n_decisions": int(sum(hist.values())),
        "n_fallback": count_fallbacks(runs),
        # Action-type order, not count order: two runs' blocks then diff cleanly.
        "observed": {a: int(hist[a]) for a in ALL_ACTIONS if hist[a]},
        "declared": list(declared) if declared is not None else None,
        "undeclared": None,
        "unexercised": None,
    }
    if declared is not None:
        # Against `chosen`, not `hist`: blaming an agent for an action the harness
        # substituted would be a false accusation, and crediting it with one would be a
        # false alibi.
        out["undeclared"] = [a for a in ALL_ACTIONS if chosen[a] and a not in declared]
        out["unexercised"] = [a for a in declared if not chosen[a]]
    return out


def format_report(
    name: str, report: dict[str, Any], *, complete: bool = True, indent: str = "  "
) -> str:
    """The human-readable block ``evaluate.py`` prints under each agent's mean.

    ``complete=False`` for a run that did not cover the whole split: a declared action
    going unseen there is a sample size, not a defect, and calling it a mismatch would
    train submitters out of declaring anything.
    """
    obs = report["observed"]
    lines = [
        f"{indent}repertoire: {len(obs)} action type(s) over {report['n_decisions']:,} "
        f"decisions ({report['n_fallback']:,} fallback)",
        f"{indent}{indent}" + (" · ".join(f"{a} {c:,}" for a, c in obs.items()) or "(none)"),
    ]
    if report["undeclared"]:
        lines.append(
            f"{indent}REPERTOIRE MISMATCH ({name}): emitted but not declared: "
            f"{', '.join(report['undeclared'])}"
        )
    if report["unexercised"] and complete:
        lines.append(
            f"{indent}REPERTOIRE MISMATCH ({name}): declared but never emitted over the "
            f"whole split: {', '.join(report['unexercised'])}"
        )
    elif report["unexercised"]:
        lines.append(
            f"{indent}not seen in this partial run (a sample size, not a defect): "
            f"{', '.join(report['unexercised'])}"
        )
    if report["undeclared"] or (report["unexercised"] and complete):
        lines.append(
            f"{indent}{indent}the published description and the run disagree. Fix "
            f"AgentSpec.declared_actions and docs/known-limits.md together."
        )
    return "\n".join(lines)


def docs_block() -> str:
    """The declared repertoires as the markdown table ``docs/known-limits.md`` carries.

    Generated from the registry rather than written by hand: the failure this whole
    module exists for was a hand-written description drifting from the agent. The
    docs test asserts the file contains exactly this.
    """
    from jackhammer.bench import agents

    rows = [
        f"| `{spec.name}` | {len(spec.declared_actions)} | "
        f"{' · '.join(f'`{a}`' for a in spec.declared_actions)} |"
        for spec in agents.all_specs()
        if spec.declared_actions is not None
    ]
    return "\n".join(["| agent | types | declared action types |", "|---|---|---|", *rows])


# --------------------------------------------------------------------------- static
@dataclass(frozen=True)
class StaticScan:
    """What a policy's *source* can construct, as opposed to what a run contains.

    Attributes:
        actions: action types the code names outright in a ``FactoredAction(...)``.
        dynamic: it builds an action from a computed value — sampling the legal-action
            mask, say — so the real ceiling is every action and ``actions`` is a floor.
        fallback: it can delegate to ``get_fallback_action``, which may return anything.
            Recorded separately because such a decision is marked ``was_fallback`` and is
            the harness's choice, not the agent's, so it does not widen what the agent
            *claims*.
    """

    actions: frozenset[str]
    dynamic: bool
    fallback: bool

    def __or__(self, other: StaticScan) -> StaticScan:
        """Union, for a policy composed of several classes."""
        return StaticScan(
            actions=self.actions | other.actions,
            dynamic=self.dynamic or other.dynamic,
            fallback=self.fallback or other.fallback,
        )


_EMPTY_SCAN = StaticScan(frozenset(), False, False)


def _action_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``_NAME = int(ActionType.X)`` / ``= ActionType.X`` / ``= <int>``."""
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        name = _action_of(node.value, {})
        if name:
            out[node.targets[0].id] = name
    return out


def _action_of(node: ast.expr, consts: dict[str, str]) -> str | None:
    """Resolve one expression to an action-type name, or None if it is not one."""
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "int" and node.args:
        return _action_of(node.args[0], consts)
    if isinstance(node, ast.Attribute) and node.attr in ALL_ACTIONS:
        return node.attr
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return ACTION_NAMES.get(node.value)
    return None


def scan_source(path: str | Path) -> dict[str, StaticScan]:
    """Scan one module for what each top-level class or function can construct.

    Deliberately syntactic: it resolves ``FactoredAction(action_type=...)`` against the
    module's own action constants and gives up honestly rather than guessing. Giving up
    sets ``dynamic``, which widens the ceiling to everything — the safe direction, since a
    ceiling that is too high can only make a check vacuous, never make it accuse wrongly.
    """
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    consts = _action_constants(tree)
    out: dict[str, StaticScan] = {}
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        actions: set[str] = set()
        dynamic = fallback = False
        for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
            func = getattr(call.func, "id", "") or getattr(call.func, "attr", "")
            if func == "get_fallback_action":
                fallback = True
            if func != "FactoredAction":
                continue
            kw = next((k for k in call.keywords if k.arg == "action_type"), None)
            name = _action_of(kw.value, consts) if kw is not None else None
            if name is None:
                dynamic = True
            else:
                actions.add(name)
        out[node.name] = StaticScan(frozenset(actions), dynamic, fallback)
    return out


def scan_union(path: str | Path, names: list[str]) -> StaticScan:
    """Union the scans of *names* in one module. Raises on a name that is not there."""
    scans = scan_source(path)
    missing = [n for n in names if n not in scans]
    if missing:
        raise KeyError(f"{path}: no top-level definition(s) named {missing}")
    out = _EMPTY_SCAN
    for n in names:
        out = out | scans[n]
    return out
