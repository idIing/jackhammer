# Benchmark protocol v2

Current revision: **v2.1**, frozen on 2026-09-05. Artifacts stamp `jackhammer/v2.1`.
Supersedes [protocol v1](protocol-v1.md), which stays published so existing `jackhammer/v1`
artifacts remain interpretable.

**How this document is versioned — amended at v2.1.** As frozen at v2.0 this document said that
changing *any* numbered item creates v3. Read literally that makes every engine re-pin a major
version, because item 1 names a commit; the number would then inflate on routine maintenance while
telling a reader nothing about comparability. The rule is now split by *what* moved rather than by
whether a clause was edited:

- **Major — v3**, and this document is frozen and superseded the way v1 was: a clause changes what
  it *means*. The action set, the primary metric, the comparison procedure, the battery, the
  holdout rule, the slate, or a deliberate change to how a baseline plays.
- **Point — v2.*x***, edited into this document: the contract is unchanged and the conditions
  under it move. The engine pin advances, or a defect in the evaluation machinery is corrected.
  Every published figure is re-baselined and § What changed records what moved.

Both are a new protocol version and both are stamped, so no artifact is ever ambiguous about which
conditions produced it. The difference is what a reader may do with two numbers: v2.0 and v2.1 ask
the same question of two different apparatus, v2 and v3 ask different questions.

**Being explicit about what that reclassifies.** v2.1 edits the commit in item 1, and it changes how
`greedy-shop` plays by repairing the scorer it ranks candidate plays with. Under the rule as frozen
at v2.0 that was v3. It is a point release under the rule above, and amending the rule is this
release's only change to the contract's own text — recorded here rather than made quietly.

Point releases so far: **v2.1** (2026-09-05) — engine re-pin and a scorer fidelity fix, § What
changed in v2.1.

1. **Engine:** public `idIing/jackdaw-balatro` commit
   `de733ebd494a5da71fb7049b3b6b18ecd039786c`. Every artifact records the resolved commit and dirty
   state. An unknown or dirty engine produces a local artifact marked non-attributable.
2. **Battery:** `config/seed_battery_v1.json`, `train`, 240 seeds, in committed order. The raw file
   SHA-256 is stamped. The 60-seed `val` split was consumed during development and is retired; the
   CLI requires an explicit warning flag to read it.
3. **Environment:** Jackdaw defaults: Red Deck, White Stake, seeded play, maximum 2,000 steps.
4. **Action set:** **an agent is offered every action the engine makes legal, in every phase.** The
   episode loop holds no policy. New in v2 — see § What changed in v2.
5. **Primary metric:** mean highest ante reached. Win rate, blind-clear summaries, conditional
   advance curves, and per-seed records remain in the artifact.
6. **Comparison:** both arms run on the identical seed list. Results are joined by seed; the primary
   difference is the mean per-seed highest-ante delta with a deterministic 10,000-resample
   percentile bootstrap 95% interval.
7. **Published slate:** `random-legal`, `random-shop`, and `greedy-shop`. Their policy behavior and
   stable IDs are frozen. `greedy-shop` is displayed as Cheapest-Joker Shop because that is what its
   selection logic actually does. Both shop baselines declare a fixed blind-select and cash-out
   policy of their own (§ What changed in v2).

## What changed in v2.1

Two changes, both of which move published numbers. Between them they edit item 1's commit and
change how the reference agent plays; § How this document is versioned says why that is a point
release and not v3.

**The engine pin moved** from `4d6f19d` to `de733eb`. The fork this benchmark runs on was rebased
onto canonical Jackdaw `8712c1e`, taking eleven upstream commits — among them an O(n²) hot-loop fix
in `get_x_same` and stake-sticker flag handling — and dropping two of its own that upstream had
since absorbed. Neither upstream commit touching scoring-relevant code changes an output *at this
battery's configuration*: the `get_x_same` change is exact-output by construction, and the
stake-sticker change only moves where the enable flags are read, which at item 3's White Stake
enables nothing under either reading. At Gold Stake the sticker change is a real behaviour change,
so the claim is scoped to the published configuration rather than to the commits. Run on its own —
new pin, scorer untouched — the battery returns `greedy-shop` 3.204,
`random-shop` 1.637, paired `+1.567 [+1.400, +1.729]`, 15,349 decisions: **the v2.0 headline to the
digit.** Everything that moves below is therefore the second change, not the pin.

**`preview_play` now builds the scorer's `game_state` the way the engine does.**
`playground/exact_score.py` reconstructs by hand the synthetic `game_state` that `_handle_play_hand`
passes to `score_hand`, and `GreedyTactical` ranks every candidate play through it — so it sits
inside the reference agent's play selection, not in a diagnostic path. Four of the nineteen keys
`score_hand` reads were wrong: `hands_played`, `skips` and `chips` were never set at all
(jackhammer#9), and `current_round_hands_played` was passed one ahead of the engine, which increments
both hand counters *after* `score_hand` returns. A wrong key does not raise — it defaults — so the
preview simply disagreed with the engine on the boards that read it: Loyalty Card, Throwback,
Mr. Bones, DNA, Sixth Sense.

Measured before the fix, over six battery seeds: 218 of 36,941 previewed plays disagreed with the
engine's own dry-run of the same play, every one a Loyalty Card x4 the engine applied and the
preview did not. `tests/test_exact_score.py` is the standing gate: it diffs the two `game_state`
dicts key for key at real positions, and compares preview totals against the engine over every play
the tactical enumerates.

### The numbers that moved

| | v2.0 (`4d6f19d`) | v2.1 (`de733eb` + scorer fix) |
|---|---:|---:|
| `greedy-shop` mean highest ante | 3.204 | **3.196** |
| `random-shop` mean highest ante | 1.637 | 1.637 |
| `random-legal` mean highest ante | 1.000 | 1.000 |
| paired `greedy-shop` − `random-shop` | +1.567 `[+1.400, +1.729]` | **+1.558 `[+1.396, +1.721]`** |
| `greedy-shop` decisions recorded | 15,349 | **15,307** |
| `random-shop` / `random-legal` decisions | 9,157 / 5,713 | 9,157 / 5,713 |
| wins, all three agents | 0/240 | 0/240 |

Only `greedy-shop` moves, and that is the expected shape. `random-legal` never runs the tactical
layer at all. `random-shop` does, and a corrected key only changes a preview on a board that reads
it — so what matters is whether an arm ever holds one of the five jokers *while playing a hand*.
Replaying the decision streams: `greedy-shop` does in **25 of 240** runs, `random-shop` in **none**.
It is not that it never buys them; it acquires one in 5 runs (`K1U9J9UF`, `Q358C3MG`, `K9ADQ6YV`
Mr. Bones, `EZG9JGQS`, `JBVSIWHI` Loyalty Card) and sells each before it reaches a play decision.
Its per-seed outcomes and its whole decision histogram are unchanged.

`build_comparison` refuses to pair arms whose engine commits differ, so a v2.0 artifact and a v2.1
artifact cannot be silently joined. Read `provenance.engine.commit` before comparing anything.

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
240-seed `train` split, and the headline paired delta was unchanged at **+1.567 ante
[+1.400, +1.729]** — the v2.0 figure, at engine `4d6f19d`; § What changed in v2.1 carries the
current one. That is expected rather than lucky: the two shop baselines now *declare* the
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
under `jackhammer/dataset-eval/v2.1` and are not v2 benchmark numbers.
