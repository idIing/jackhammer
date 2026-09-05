"""What produced a benchmark number: engine pin, battery digest, kit commit.

Every result artifact carries one of these. The motivating failure mode is real
and cheap to hit: run many agent versions over months, later discover that a
substantial share of the simulator's joker implementations were wrong — wrong in
ways that change what optimal play is — and every number already reported becomes
unattributable, because nothing recorded *which* engine produced it.

So a number without an engine pin is not a result. The stamp is cheap; regretting
its absence costs the whole campaign.

The public install resolves Jackdaw from a pinned Git dependency. PEP 610's
``direct_url.json`` records the resolved commit in the installed distribution;
a source checkout instead uses Git directly so local modifications are visible.
If neither route can identify a commit, the result degrades to unattributable
rather than crashing.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

# Bump when the *meaning* of a stamped field changes, not when a value changes.
#
# v2 (2026-09-04): the episode loop stopped auto-playing blind select and cash-out,
# so an agent's action set is now the engine's. Every v1 number was produced by an
# agent that could not skip a blind or use a consumable before the shop was rolled.
# v1 and v2 artifacts are not comparable; see docs/protocol-v2.md § What changed.
PROTOCOL = "jackhammer/v2"

# A run whose shared tactical budget is not the frozen cap is not a benchmark result.
# Stamped separately so a sweep can never be mistaken for the benchmark number.
TACTICAL_PROTOCOL = "jackhammer/tactical-sweep/v1"

# provenance.py is src/bench/provenance.py -> parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = REPO_ROOT / "vendor" / "jackdaw-balatro"


def _git(cwd: Path, *args: str) -> str | None:
    """Run ``git *args`` in *cwd*; return stripped stdout, or None if unavailable.

    Returns None for every failure mode a consumer can legitimately hit: no git
    binary, not a checkout (installed from a wheel), or a submodule that was
    never initialised.
    """
    if not cwd.exists():
        return None
    try:
        out = subprocess.run(
            ("git", *args),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _pin(path: Path) -> dict[str, Any]:
    """Commit SHA + dirty flag for the checkout at *path*.

    ``dirty`` is None (not False) when the commit itself is unknown — absence of
    evidence, so callers can tell "clean" apart from "unknowable".
    """
    commit = _git(path, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    status = _git(path, "status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def digest_file(path: Path) -> str:
    """SHA-256 of *path*'s raw bytes, as ``sha256:<hex>``.

    Raw bytes rather than parsed-and-recanonicalised JSON: the point is to detect
    that the frozen battery changed at all, including reorderings that a
    semantic comparison would forgive.
    """
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def engine_pin() -> dict[str, Any]:
    """Identify the Jackdaw build that will execute the runs.

    The installed distribution version is recorded but is *not* the pin: the
    engine has reported ``0.1.0`` across many changes. A source checkout commit
    or the installed VCS commit is the field with resolving power.
    """
    try:
        dist = metadata.distribution("jackdaw")
        dist_version = dist.version
    except metadata.PackageNotFoundError:
        return {
            "name": "jackdaw",
            "dist_version": None,
            "commit": None,
            "dirty": None,
            "source": "unknown",
        }

    checkout = _pin(ENGINE_PATH)
    if checkout["commit"] is not None:
        return {
            "name": "jackdaw",
            "dist_version": dist_version,
            **checkout,
            "source": "checkout",
        }

    try:
        direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
        vcs = direct_url.get("vcs_info") or {}
        commit = vcs.get("commit_id")
    except (AttributeError, json.JSONDecodeError, OSError):
        commit = None
    return {
        "name": "jackdaw",
        "dist_version": dist_version,
        "commit": commit,
        # A non-editable installed VCS artifact has no mutable source tree.
        "dirty": False if commit else None,
        "source": "installed-vcs" if commit else "unknown",
    }


def stamp(
    *,
    battery_path: Path,
    split: str,
    n_seeds: int,
    protocol: str = PROTOCOL,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance block embedded in every result artifact.

    Args:
        battery_path: the frozen seed bank actually read for this evaluation.
        split: which slice was drawn (``train`` / ``val`` / ``all``).
        n_seeds: how many seeds were actually evaluated — may be below the split
            size when ``--limit`` was used for a smoke run, which is exactly the
            case a reader must be able to detect.
        protocol: evaluation contract identifier. The default is the frozen
            benchmark protocol (``PROTOCOL``); diagnostic datasets and tactical
            sweeps use their own identifiers.
        extra: additional caller-supplied fields, merged at the top level.

    Returns:
        A JSON-serialisable dict. No field is permitted to raise.
    """
    battery: dict[str, Any] = {
        "path": str(battery_path.relative_to(REPO_ROOT))
        if battery_path.is_relative_to(REPO_ROOT)
        else str(battery_path),
        "split": split,
        "n_seeds": n_seeds,
        "digest": digest_file(battery_path),
    }
    try:
        battery["version"] = json.loads(battery_path.read_text()).get("version")
    except (OSError, json.JSONDecodeError):
        battery["version"] = None

    out: dict[str, Any] = {
        "protocol": protocol,
        "battery": battery,
        "engine": engine_pin(),
        "kit": _pin(REPO_ROOT),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "implementation": sys.implementation.name,
        },
    }
    if extra:
        out.update(extra)
    return out


def is_attributable(prov: dict[str, Any]) -> bool:
    """True when this stamp pins an exact, unmodified engine.

    The bar for a *reportable* number: we know the engine commit and nothing was
    edited underneath it. A dirty or unknown engine still produces a valid
    artifact — it just cannot be compared against anyone else's.
    """
    engine = prov.get("engine") or {}
    return bool(engine.get("commit")) and engine.get("dirty") is False
