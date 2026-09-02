# Benchmark protocol v1

Status: frozen on 2026-09-02. Changing a numbered item creates v2 rather than silently editing v1.

1. **Engine:** public `idIing/jackdaw-balatro` commit
   `4d6f19d9fe73f96603412a59ad5eab16d08937e7`. Every artifact records the resolved commit and dirty
   state. An unknown or dirty engine produces a local artifact marked non-attributable.
2. **Battery:** `config/seed_battery_v1.json`, `train`, 240 seeds, in committed order. The raw file
   SHA-256 is stamped. The 60-seed `val` split was consumed during development and is retired; the
   CLI requires an explicit warning flag to read it.
3. **Environment:** Jackdaw defaults: Red Deck, White Stake, seeded play, maximum 2,000 steps.
4. **Primary metric:** mean highest ante reached. Win rate, blind-clear summaries, conditional
   advance curves, and per-seed records remain in the artifact.
5. **Comparison:** both arms run on the identical seed list. Results are joined by seed; the primary
   difference is the mean per-seed highest-ante delta with a deterministic 10,000-resample
   percentile bootstrap 95% interval.
6. **Published slate:** `random-legal`, `random-shop`, and `greedy-shop`. Their policy behavior and
   stable IDs are frozen. `greedy-shop` is displayed as Cheapest-Joker Shop because that is what its
   selection logic actually does.

## Reporting checklist

A v1 claim reports both arm names, seed count, failure count, mean highest ante for each arm, paired
delta and interval, win counts, engine commit/dirty state, battery digest, and artifact paths. A
limited smoke run is not a v1 result even though it uses the same machinery.

## Scope

The protocol measures policies inside the pinned Jackdaw simulator. Deterministic replay establishes
simulator reproducibility, not equivalence to live Balatro. Results from custom datasets are stamped
under `jackhammer/dataset-eval/v1` and are not v1 benchmark numbers.
