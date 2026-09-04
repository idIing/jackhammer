# Benchmark protocol v2

Status: frozen on 2026-09-04. Changing a numbered item creates v3 rather than silently editing v2.
Supersedes [protocol v1](protocol-v1.md), which stays published so existing `jackhammer/v1`
artifacts remain interpretable.

1. **Engine:** public `idIing/jackdaw-balatro` commit
   `4d6f19d9fe73f96603412a59ad5eab16d08937e7`. Every artifact records the resolved commit and dirty
   state. An unknown or dirty engine produces a local artifact marked non-attributable.
2. **Battery:** `config/seed_battery_v1.json`, `train`, 240 seeds, in committed order. The raw file
   SHA-256 is stamped. The 60-seed `val` split was consumed during development and is retired; the
   CLI requires an explicit warning flag to read it.
3. **Environment:** Jackdaw defaults: Red Deck, White Stake, seeded play, maximum 2,000 steps.
4. **Action set:** **an agent is offered every action the engine makes legal, in every phase.** The
   episode loop holds no policy. New in v2 — see § What changed.
5. **Primary metric:** mean highest ante reached. Win rate, blind-clear summaries, conditional
   advance curves, and per-seed records remain in the artifact.
6. **Comparison:** both arms run on the identical seed list. Results are joined by seed; the primary
   difference is the mean per-seed highest-ante delta with a deterministic 10,000-resample
   percentile bootstrap 95% interval.
7. **Published slate:** `random-legal`, `random-shop`, and `greedy-shop`. Their policy behavior and
   stable IDs are frozen. `greedy-shop` is displayed as Cheapest-Joker Shop because that is what its
   selection logic actually does. Both shop baselines declare a fixed blind-select and cash-out
   policy of their own (§ What changed).

## What changed in v2

Under v1 the episode loop played two phases *for* the agent, before any policy was consulted: it
always selected the blind at `BLIND_SELECT`, and always cashed out at `ROUND_EVAL`. No agent could
do otherwise, including `random-legal`, whose published description nevertheless claimed it sampled
"uniformly-random legal action everywhere."

Both are real decisions, not formalities:

- `SkipBlind` is legal on every Small and Big blind and **nowhere else**
  (`jackdaw/env/action_space.py`). Skipping takes a tag, advances to the next blind, and fires every
  joker's `skip_blind` trigger.
- A consumable used at `ROUND_EVAL` releases its key back to the pool **before** the next shop is
  rolled (`jackdaw/engine/game.py` populates the shop inside the cash-out handler, then flips the
  phase). Using it one step later, in the shop, is too late to affect what the shop offers.

In v2 the loop delegates every phase, and an agent that wants a blind selected must select it.

**This changed no v1 number.** All three baselines returned byte-identical per-seed outcomes on the
240-seed `train` split, and the headline paired delta is unchanged at **+1.567 ante
[+1.400, +1.729]**. That is expected rather than lucky: the two shop baselines now *declare* the
same never-skip, always-cash-out policy the loop used to impose, and `random-legal` is too weak for
its new options to reach the primary metric — it cleared 1 blind in 240 games under both protocols.

The version bump is therefore about the **contract, not the numbers**. A v1 result and a v2 result
are comparable for these three agents and are *not* comparable in general, because any agent that
would use the restored actions was silently prevented from doing so under v1.

What did move is the record and the coverage:

| | v1 | v2 |
|---|---:|---:|
| blind-select / cash-out decisions recorded, all three agents | 0 | 5,240 |
| `random-legal` blinds faced: Small / Big / Boss | 240 / 1 / 0 | 124 / 62 / 55 |

The floor baseline had never faced a boss blind in 240 games. It reaches 55 of them under v2 by
skipping into them — still clearing none, which is why the primary metric does not notice.

## Reporting checklist

A v2 claim reports both arm names, seed count, failure count, mean highest ante for each arm, paired
delta and interval, win counts, engine commit/dirty state, battery digest, and artifact paths. A
limited smoke run is not a v2 result even though it uses the same machinery.

## Scope

The protocol measures policies inside the pinned Jackdaw simulator. Deterministic replay establishes
simulator reproducibility, not equivalence to live Balatro. Results from custom datasets are stamped
under `jackhammer/dataset-eval/v1` and are not v2 benchmark numbers.
