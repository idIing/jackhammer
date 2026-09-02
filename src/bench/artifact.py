"""The standardized result artifact: one JSON file per evaluation.

Two shapes are self-describing via a ``schema`` field:

* ``jackhammer.result/v1`` — one agent over one split.
* ``jackhammer.comparison/v1`` — two results joined per-seed, with the paired
  statistics that are the only honest way to report a difference here.

The comparison shape is the one that matters. A single unpaired number — "2.35%
win rate", "~30% white stake" — fixes no seeds, carries no interval, and has no
baseline on the same seeds, so it cannot support a claim about a difference.
A paired design makes each seed its own control, which is why the artifact
carries ``n_paired`` and the joined seed list rather than two independent totals.

The statistics are *not* computed here; ``src.playground.metrics`` already owns
and tests them. This module only stamps, shapes, and serializes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.bench.agents import AgentSpec
from src.bench.provenance import is_attributable
from src.playground import metrics

RESULT_SCHEMA = "jackhammer.result/v1"
COMPARISON_SCHEMA = "jackhammer.comparison/v1"


def _jsonable(obj: Any) -> Any:
    """Coerce numpy scalars/arrays to plain Python so ``json.dump`` succeeds.

    ``metrics`` casts most values already, but not exhaustively, and a numpy
    float reaching ``json.dump`` fails at write time — after the expensive run.
    Cheap insurance at the only place it matters.
    """
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item) and getattr(obj, "ndim", None) == 0:
        return obj.item()  # numpy scalar
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return obj.tolist()  # numpy array
    return obj


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def build_result(
    *,
    spec: AgentSpec,
    runs: list[dict],
    provenance: dict[str, Any],
    runs_path: str | None = None,
) -> dict[str, Any]:
    """Assemble one agent's result over one split.

    Args:
        spec: the evaluated agent; its ``name`` is the published identity.
        runs: recorder JSONL objects, one per seed.
        provenance: from ``provenance.stamp``.
        runs_path: where the raw per-run JSONL lives, for drill-down. Kept as a
            reference rather than inlined — trajectories are large and the
            artifact is meant to be readable and diffable.
    """
    return _jsonable(
        {
            "schema": RESULT_SCHEMA,
            "created": _now(),
            "agent": spec.identity(),
            "provenance": provenance,
            "attributable": is_attributable(provenance),
            "runs_path": runs_path,
            "summary": metrics.summarize(runs),
        }
    )


def build_comparison(
    *,
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    runs_a: list[dict],
    runs_b: list[dict],
    advance_k: int = 2,
) -> dict[str, Any]:
    """Join two results per-seed and attach the paired statistics.

    Both arms must come from the same engine and the same battery digest;
    otherwise the pairing is not a controlled comparison and we refuse to
    produce an artifact that would read as though it were.

    Raises:
        ValueError: if the arms' provenance is incompatible.
    """
    pa, pb = result_a["provenance"], result_b["provenance"]
    if pa["protocol"] != pb["protocol"]:
        raise ValueError(
            f"cannot pair results from different protocols: {pa['protocol']} vs {pb['protocol']}"
        )
    if pa["engine"]["commit"] != pb["engine"]["commit"]:
        raise ValueError(
            "cannot pair results from different engine commits: "
            f"{pa['engine']['commit']} vs {pb['engine']['commit']}"
        )
    if pa["battery"]["digest"] != pb["battery"]["digest"]:
        raise ValueError(
            "cannot pair results from different seed batteries: "
            f"{pa['battery']['digest']} vs {pb['battery']['digest']}"
        )
    if pa["battery"]["split"] != pb["battery"]["split"]:
        raise ValueError(
            "cannot pair results from different splits: "
            f"{pa['battery']['split']} vs {pb['battery']['split']}"
        )

    return _jsonable(
        {
            "schema": COMPARISON_SCHEMA,
            "created": _now(),
            "arm_a": result_a["agent"],
            "arm_b": result_b["agent"],
            # Shared by construction (validated above), so recorded once.
            "provenance": pa,
            "attributable": bool(result_a.get("attributable") and result_b.get("attributable")),
            "comparison": metrics.compare(runs_a, runs_b, advance_k=advance_k),
        }
    )


_REQUIRED = {
    RESULT_SCHEMA: ("created", "agent", "provenance", "summary"),
    COMPARISON_SCHEMA: ("created", "arm_a", "arm_b", "provenance", "comparison"),
}


def validate(obj: dict[str, Any]) -> None:
    """Check *obj* against its declared schema. Raises ValueError on any defect.

    Structural only — that the artifact is readable by a consumer who knows the
    schema name. It does not re-derive the statistics.
    """
    schema = obj.get("schema")
    if schema not in _REQUIRED:
        raise ValueError(f"unknown artifact schema {schema!r}; expected one of {sorted(_REQUIRED)}")
    missing = [k for k in _REQUIRED[schema] if k not in obj]
    if missing:
        raise ValueError(f"{schema}: missing required key(s) {missing}")

    prov = obj["provenance"]
    for key in ("protocol", "battery", "engine", "kit", "runtime"):
        if key not in prov:
            raise ValueError(f"{schema}: provenance missing required key {key!r}")
    for key in ("split", "n_seeds", "digest"):
        if key not in prov["battery"]:
            raise ValueError(f"{schema}: provenance.battery missing key {key!r}")


def write(path: str | Path, obj: dict[str, Any]) -> Path:
    """Validate *obj*, then write it as indented JSON to *path*."""
    validate(obj)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
    return p


def read(path: str | Path) -> dict[str, Any]:
    """Read and validate an artifact from *path*."""
    obj = json.loads(Path(path).read_text())
    validate(obj)
    return obj
