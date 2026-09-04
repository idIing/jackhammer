# Known limits

- **Simulator proxy:** Jackdaw is not live Balatro. The v1 pin has deterministic and engine tests,
  but no claim of complete simulator equivalence follows from those checks.
- **One configuration:** v1 covers Red Deck, White Stake, one engine pin, and one IID seed battery.
- **Weak baselines:** all three built-ins won 0/240. Mean highest ante separates them, but the slate
  does not represent the field's strongest agents.
- **Cheapest-Joker policy:** `greedy-shop` buys only Jokers, never evaluates their text or synergy,
  never rerolls or sells, takes the first pack card, and ignores vouchers and consumables.
- **Tactical scan cap:** `GreedyTactical` exact-scores at most `score_budget=300` card subsets per
  play decision. That is exhaustive for a standard 8-card hand (218 subsets of size <=5); above 8 it
  truncates, and because it enumerates small-k first, the subsets it drops are the largest. Hand
  size grows in ordinary play, so this is reached routinely — see
  [the tactical scan cap](#the-tactical-scan-cap) below.
- **Scan-cap asymmetry:** the shared tactical layer is not automatically a symmetric control.
  `greedy-shop` truncates 2.4x as often as `random-shop`, and re-running the battery with the cap
  raised moves the paired delta from `+1.567` to `+1.667` — a paired difference-of-differences of
  **`+0.100 [+0.046, +0.167]`** boot-95, which excludes zero, so the published number slightly
  *understates* the shop contrast. `+1.667` still falls inside the published interval
  `[+1.400, +1.729]`, so no v1 conclusion changes.
- **Published battery:** the 240 training seeds are public and therefore overfittable. The old
  validation split has already been consumed and is retired, not a reusable secret leaderboard.
- **No live client:** the benchmark and text run inspector work headlessly. This repository does not
  distribute Balatro, its source, or its assets.
- **Dataset selection:** a diagnostic manifest can answer a scoped coverage question but cannot be
  relabeled as ordinary-distribution policy strength. Sampling and overlap must be reported.
- **Artifact schema:** v1 validates the stable envelope and preserves the raw decision records by
  reference; it does not cryptographically sign results or fully validate every nested summary field.

## The tactical scan cap

**How it degrades.** Enumeration is small-k first and stops at the budget, so severity is a ladder,
not a cliff. At hand size 9, 45 of the 126 five-card subsets are scored; at 10, none are, and the
four-card subsets begin truncating too; at 13 the budget runs out inside k=3, so at most three cards
can be selected and only high card, pair and three of a kind stay reachable at all. Hand size grows
in ordinary play — Juggler +1, Troubadour +2, the Paint Brush and Palette vouchers +1 each, and
transiently Turtle Bean +5 and the Juggle Tag +3.

**How often.** Instrumenting the true hand size at every scan, the cap binds on 576 of 8341
`greedy-shop` play scans (6.91%, in 30/240 games) against 151 of 5176 for `random-shop` (2.92%,
14/240); a scan runs on every in-blind decision, discards included. The instrumentation is pure
observation — the seeds re-run under it reproduce their published `highest_ante`, 16/16
spot-checked — but it is not shipped, because the published decision records store the subset
played, not the hand it was drawn from. These counts supersede the 19/240 and 8/240 published at
launch, which came from a terminal-state estimate that omitted Troubadour and could not see
transient hand size at all.

**Why it is asymmetric.** Both arms run the layer at the same budget, but a large hand is downstream
of buying Jokers and surviving longer, so `greedy-shop` meets the cap more often. The cap is not
what splits the arms: they are bit-identical through the ante-1 Small blind in 240/240 seeds, and
the first differing action is a shop decision in 232/240, never a tactical one.

**What it costs.** Re-run the battery yourself at a raised cap:

```
uv run python scripts/evaluate.py --agent greedy-shop --vs random-shop --score-budget 8000
```

Any budget other than the v1 `300` is stamped `jackhammer/tactical-sweep/v1` with
`scope: diagnostic`, and the result's `agent.tactical` records the budget that actually ran, so a
sweep can never be read as a v1 number. Doing so shifts `greedy-shop` by +0.104 ante
`[+0.046, +0.175]` and `random-shop` by +0.004 `[+0.000, +0.013]`, and changes the outcome of 13/240
seeds against 1/240. The cost is concentrated rather than diffuse: on the 201 seeds that never
truncate the difference-of-differences is exactly zero with zero variance — as it must be, since an
untruncated scan enumerates the same subsets at either budget — while the 39 exposed seeds shift
`+0.615 [+0.308, +0.974]`.

**The worst case.** The Psychic scores any play of fewer than five cards as zero (jackdaw's
`h_size_ge=5` boss debuff), and at a true hand size of 10 or more the cap enumerates no five-card
subset at all — so every play the layer can reach scores exactly zero, and the blind is unwinnable
for as long as the hand stays that large. It bit on 2 of 45 `greedy-shop` Psychic blinds and 0 of 26
for `random-shop`: seeds `657P5QGW` (Troubadour, +2) and `PM4RVISW` (Turtle Bean, +5) both scored
0/600 and lost at ante 1, and both clear 720/600 and reach ante 3 and ante 4 at the raised cap. The
lockups are in the published records —
`uv run python scripts/inspect_run.py data/bench/greedy-shop.jsonl --seed 657P5QGW` prints three
`High Card score=0` plays.

**Why v1 keeps it.** Raising the cap moves published numbers, so it is a v2 question, not a v1
patch. The outcome plateaus at `score_budget=2000` (mean highest ante 3.308, unchanged at 4000, 8000
and 16000), and 2379 — every subset of size <=5 of the largest hand this battery dealt, 13
cards — is the budget above which no scan in these runs can truncate at all. Going from 300 to 8000
scans 7.4% more combos for 14.5% more wall clock (82.9s -> 94.9s, 14 workers).

**If you are submitting an agent.** An agent with its own tactical layer is not subject to the cap,
but its measured margin over `greedy-shop` still carries this handicap on the exposed seeds. An
agent that reuses `GreedyTactical` inherits the cap outright.

Please open a narrowly reproducible issue when you find a simulator divergence or a gap in the
getting-started path.
