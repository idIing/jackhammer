#!/usr/bin/env python
"""Run a registered agent on benchmark v1 or a versioned diagnostic dataset.

Agents come from ``src.bench.agents``, so adding one is a registration, not an
edit to this runner.

Usage::

    python scripts/evaluate.py --list
    python scripts/evaluate.py --agent greedy-shop --limit 8 --workers 8
    python scripts/evaluate.py --agent greedy-shop --vs random-shop --split train
    python scripts/evaluate.py --agent greedy-shop --dataset my-seeds.json --split sample

Every run writes a standardized artifact (``src.bench.artifact``) stamped with
the engine commit, battery digest, and agent identity. ``--vs`` additionally
writes the paired comparison, which is the only form in which a *difference*
between two agents should be reported.

The ``val`` split is retired. It is gated behind an explicit flag on purpose: its
value is destroyed by repeated looks and it was consumed during development. v1
evaluations use ``train`` (``docs/protocol-v1.md`` section 2).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.bench import agents as agent_registry  # noqa: E402
from src.bench import artifact, provenance  # noqa: E402
from src.bench.datasets import DATASET_PROTOCOL, load_dataset  # noqa: E402
from src.playground import metrics  # noqa: E402
from src.playground.harness import run_battery_with  # noqa: E402
from src.playground.seeds import BATTERY_PATH, load_battery  # noqa: E402

HOLDOUT_FLAG = "--i-am-consuming-the-holdout"


def _repo_rel(path: Path) -> str:
    """Repo-relative path when possible, else absolute (``--out-dir /tmp/...``)."""
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def _shard_path(out_dir: Path, agent: str, seed: str) -> Path:
    return out_dir / "shards" / agent / f"{seed}.jsonl"


def _play_one(task: tuple[str, str, str]) -> dict:
    """Run one (agent, seed) game into its own shard. Never raises.

    One bad seed must not kill a 240-seed battery, so failures are returned as
    data: they are printed as they occur, omitted from the JSONL, and make the
    process exit nonzero.
    """
    agent_name, seed, out_dir = task
    shard = _shard_path(Path(out_dir), agent_name, seed)
    shard.parent.mkdir(parents=True, exist_ok=True)
    if shard.exists():
        shard.unlink()  # idempotent re-run
    try:
        spec = agent_registry.get(agent_name)
        res = run_battery_with(
            [seed],
            spec.make_decider,
            out_path=str(shard),
            config_label=spec.name,
            slot1=spec.slot1,
            slot2=spec.slot2,
        )[0]
        return {
            "agent": agent_name,
            "seed": seed,
            "ok": True,
            "highest_ante": res.highest_ante,
            "won": res.won,
        }
    except Exception as e:  # noqa: BLE001 — see docstring
        return {"agent": agent_name, "seed": seed, "ok": False, "error": repr(e)}


def _merge(out_dir: Path, agent: str, seeds: list[str]) -> Path:
    """Concatenate per-seed shards in seed order into one JSONL."""
    final = out_dir / f"{agent}.jsonl"
    with final.open("w", encoding="utf-8") as out:
        for seed in seeds:
            shard = _shard_path(out_dir, agent, seed)
            if shard.exists():
                out.write(shard.read_text(encoding="utf-8"))
    return final


def _run_agents(
    agent_names: list[str], seeds: list[str], out_dir: Path, workers: int
) -> tuple[dict[str, Path], int]:
    """Fan (agent x seed) games across a process pool. Returns paths and fail count."""
    tasks = [(a, s, str(out_dir)) for a in agent_names for s in seeds]
    print(
        f"evaluating: {len(seeds)} seeds x {len(agent_names)} agent(s) "
        f"= {len(tasks)} games on {workers} workers",
        flush=True,
    )
    t0 = time.time()
    done = fails = 0
    # Recycle workers: deepcopy-heavy games grow RSS over a long battery.
    with Pool(processes=workers, maxtasksperchild=25) as pool:
        for st in pool.imap_unordered(_play_one, tasks):
            done += 1
            if not st["ok"]:
                fails += 1
                print(f"  FAIL {st['agent']} {st['seed']}: {st.get('error')}", flush=True)
            # Throttled plain-text progress only -- never a live-redrawing display.
            if done % 25 == 0 or done == len(tasks):
                rate = done / max(1e-9, time.time() - t0)
                eta = (len(tasks) - done) / max(1e-9, rate)
                print(
                    f"  {done}/{len(tasks)} games ({rate:.2f}/s, "
                    f"eta {eta / 60:.1f} min, {fails} fails)",
                    flush=True,
                )
    return {a: _merge(out_dir, a, seeds) for a in agent_names}, fails


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a registered Jackhammer agent.")
    ap.add_argument("--list", action="store_true", help="list registered agents and exit")
    ap.add_argument("--agent", help="agent to evaluate (see --list)")
    ap.add_argument("--vs", help="second agent; also writes a paired comparison")
    ap.add_argument(
        "--dataset",
        type=Path,
        help="versioned diagnostic seed manifest (default: frozen v1 battery)",
    )
    ap.add_argument("--split", default="train", help="dataset split to evaluate")
    ap.add_argument("--limit", type=int, default=0, help="cap #seeds (smoke runs)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out-dir", default=str(REPO / "data" / "bench"))
    ap.add_argument(
        HOLDOUT_FLAG,
        dest="consume_holdout",
        action="store_true",
        help="required to evaluate on the one-shot val holdout",
    )
    args = ap.parse_args()

    if args.list:
        for spec in agent_registry.all_specs():
            flag = "" if spec.deterministic else "  [non-deterministic]"
            print(f"  {spec.name:<16} {spec.description}{flag}")
        return 0

    if not args.agent:
        ap.error("--agent is required (or use --list)")
    if args.dataset is None and args.split in ("val", "all") and not args.consume_holdout:
        ap.error(
            f"split={args.split!r} reads the one-shot holdout. It has already been "
            f"consumed once; a second look does not mean what the first did. "
            f"Pass {HOLDOUT_FLAG} if that is genuinely intended."
        )

    agent_names = [args.agent] + ([args.vs] if args.vs else [])
    for name in agent_names:
        agent_registry.get(name)  # fail fast on a typo, before running anything

    try:
        if args.dataset is None:
            seeds = load_battery(args.split)
            dataset_path = BATTERY_PATH
            protocol = provenance.PROTOCOL
            extra = None
        else:
            dataset = load_dataset(args.dataset)
            seeds = dataset.seeds(args.split)
            dataset_path = dataset.path
            protocol = DATASET_PROTOCOL
            extra = dataset.provenance()
    except (FileNotFoundError, ValueError) as exc:
        ap.error(str(exc))
    if args.limit:
        seeds = seeds[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prov = provenance.stamp(
        battery_path=dataset_path,
        split=args.split,
        n_seeds=len(seeds),
        protocol=protocol,
        extra=extra,
    )
    if not provenance.is_attributable(prov):
        engine = prov["engine"]
        print(
            f"  WARNING: engine not attributable (commit={engine['commit']}, "
            f"dirty={engine['dirty']}). Results are local-only and must not be "
            f"published or compared against other runs.",
            flush=True,
        )

    paths, fails = _run_agents(agent_names, seeds, out_dir, args.workers)

    results: dict[str, dict] = {}
    runs: dict[str, list[dict]] = {}
    for name in agent_names:
        runs[name] = metrics.load_runs(str(paths[name]))
        results[name] = artifact.build_result(
            spec=agent_registry.get(name),
            runs=runs[name],
            provenance=prov,
            runs_path=_repo_rel(paths[name]),
        )
        written = artifact.write(out_dir / f"{name}.result.json", results[name])
        depth = results[name]["summary"]["run_depth"]
        print(f"\n{name}: {len(runs[name])} runs -> {written}", flush=True)
        print(f"  mean highest ante: {depth.get('mean'):.3f}", flush=True)

    if args.vs:
        comparison = artifact.build_comparison(
            result_a=results[args.vs],  # A = the reference arm
            result_b=results[args.agent],  # B = the agent under test
            runs_a=runs[args.vs],
            runs_b=runs[args.agent],
        )
        path = artifact.write(out_dir / f"{args.agent}_vs_{args.vs}.comparison.json", comparison)
        delta = comparison["comparison"]["depth_delta"]
        lo, hi = delta["bootstrap_ci"]
        print(
            f"\npaired ({comparison['comparison']['n_paired']} seeds): "
            f"{args.agent} - {args.vs} = {delta['mean_delta']:+.3f} ante "
            f"[{lo:+.3f}, {hi:+.3f}] boot-95",
            flush=True,
        )
        excludes_zero = (lo > 0) or (hi < 0)
        print(f"  interval {'excludes' if excludes_zero else 'includes'} zero", flush=True)
        print(f"  -> {path}", flush=True)

    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
