# Known limits

- **Simulator proxy:** Jackdaw is not live Balatro. The pinned engine has deterministic and engine
  tests, but no claim of complete simulator equivalence follows from those checks.
- **One configuration:** v2 covers Red Deck, White Stake, one engine pin, and one IID seed battery.
- **The shop baselines decline two decisions:** `random-shop` and `greedy-shop` always select the
  blind and always cash out immediately, so neither ever takes a skip tag or uses a consumable while
  it can still change the next shop. That is their declared policy, not a harness limit — under
  protocol v1 it *was* a harness limit imposed on every agent, `random-legal` included. Holding it
  fixed keeps the paired difference between the two arms attributable to the shop policy, but it
  means the slate contains no baseline that skips deliberately. `random-legal` skips at random.
  Measured over the published battery rather than read off the code: neither shop arm ever emits
  `SkipBlind`, and neither ever uses or sells anything at `ROUND_EVAL` (0 of `random-shop`'s 272
  consumable and sell actions fall there; all are shop-phase).
- **Weak baselines:** all three built-ins won 0/240. Mean highest ante separates them, but the slate
  does not represent the field's strongest agents.
- **Cheapest-Joker policy:** `greedy-shop` buys only Jokers and never evaluates their text, rarity
  or synergy. It never rerolls, never sells, never redeems a voucher, never uses a consumable — and
  it **never buys a booster pack**: `GreedyShop._joker_buys` filters shop candidates to
  `ability.set == "Joker"`, so no `OpenBooster` is ever issued. The pinned engine enters
  `PACK_OPENING` from exactly two places — `_handle_open_booster`, which requires that action, and
  `_open_tag_pack` for a tag's pack — and over the whole battery this agent emits neither
  `OpenBooster` nor `SkipBlind` nor any pack action, so the pack-choice branch of `decide_shop` is
  unreachable in practice. Over the published battery it emits **6 of the engine's 21 action types
  across 15,349 decisions**, with zero fallback substitutions, against 16 for `random-shop`.
  The paired `+1.567` is a clean A/B, but what it prices is *buying the cheapest Joker at all*
  against an arm that touches most of the shop surface at random — the headline is not "greedy beats
  random" but "the narrowest agent on the slate beats the widest one". See
  [declared repertoires](#declared-repertoires) below.
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

## Declared repertoires

Every claim above about what a baseline *does* is a run-time fact, and reading the source cannot
establish one. The 2026-09-02 audit of `greedy-shop` was a careful code read; it still shipped
"takes the first pack card", describing a branch that cannot execute, because the policy never buys
a pack in the first place. Determinism, the engine tests, the paired intervals and the test suite
all pass on an agent that only ever does six things.

So each built-in baseline now *declares* the action types its description covers
(`AgentSpec.declared_actions`), and the declaration is checked from two sides.

Against the **runs**: `scripts/evaluate.py` reads the recorded decision stream every run writes.
Emitting an action the declaration omits fails the evaluation at any sample size; declaring one the
run never contains fails a complete-battery run, and is reported as a sample-size caveat on a
`--limit` smoke run. Only a complete battery can distinguish *declined* from *never reached*, so
that half cannot run in CI.

Against the **source**: `repertoire.scan_source` lists the action types a policy's code can
construct at all. It runs in milliseconds with no engine and no games, so it runs in CI on every
push, and it catches the error where it is made rather than the next time somebody publishes a
number. The two bound the answer from opposite sides — the source is a ceiling, the runs are a
floor — and an action in the ceiling but not the floor is code that cannot be reached. For
`greedy-shop` that difference is exactly one action, `PickPackCard`, and
`tests/test_public_contracts.py` pins it: had the false clause been declared as well as written,
CI would have rejected it. A policy that builds its action from a computed value (both random arms
sample the legal-action mask) can construct anything, so the source check is vacuous for it and the
tests record *that*, rather than passing quietly.

The block below is generated from the registry and checked by the same file, so the table and the
agent cannot drift apart silently. The prose around it is still prose.

<!-- BEGIN declared-repertoire -->
| agent | types | declared action types |
|---|---|---|
| `greedy-shop` | 6 | `PlayHand` · `Discard` · `SelectBlind` · `CashOut` · `NextRound` · `BuyCard` |
| `random-legal` | 13 | `PlayHand` · `Discard` · `SelectBlind` · `SkipBlind` · `CashOut` · `NextRound` · `SkipPack` · `SellConsumable` · `PickPackCard` · `SwapHandLeft` · `SwapHandRight` · `SortHandRank` · `SortHandSuit` |
| `random-shop` | 16 | `PlayHand` · `Discard` · `SelectBlind` · `CashOut` · `Reroll` · `NextRound` · `SkipPack` · `BuyCard` · `SellJoker` · `SellConsumable` · `UseConsumable` · `RedeemVoucher` · `OpenBooster` · `PickPackCard` · `SwapJokersLeft` · `SwapJokersRight` |
<!-- END declared-repertoire -->

Measured on the 240-seed `train` split at the pinned engine, protocol v2: `greedy-shop` 15,349
decisions, `random-shop` 9,157, `random-legal` 5,713, all three with **zero** fallback
substitutions. The reference agent's six, in full:

```
PlayHand 4,527 · Discard 3,814 · SelectBlind 2,132 · CashOut 1,892 · NextRound 1,892 · BuyCard 1,092
```

`scripts/evaluate.py` prints the same line for whatever agent it runs, and `summary.repertoire` in
each result artifact carries the counts, so this table is a summary of the artifacts rather than a
separate claim about them. Two things a reader should not over-read:

- **A repertoire is not a capability.** `random-legal` samples uniformly from whatever the mask
  offers, so it *may* reroll or redeem a voucher; it never does, because it survives to a shop in 1
  of 240 games (one `CashOut`, one `NextRound`). Its 13 types are what this battery lets it show, not the limit of its policy.
- **Protocol version changes the count.** Under v1 the episode loop auto-played blind selection and
  cash-out, so those two never appeared in an agent's decision stream; the same `greedy-shop` runs
  recorded 4 action types and 11,325 decisions. The 4,024-decision difference is exactly the
  `SelectBlind` and `CashOut` calls v2 hands back to the agent. Repertoire counts are only
  comparable within one protocol version.

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

**What it costs.** Re-run the battery yourself at a raised cap. Send it somewhere other than
`data/bench/`: `--out-dir` defaults there, which is also where the README's baseline run lands, and
a sweep left to that default silently overwrites it — including the records the next paragraph asks
you to inspect.

```
uv run python scripts/evaluate.py --agent greedy-shop --vs random-shop \
    --score-budget 8000 --out-dir data/sweeps/budget-8000
```

Any budget other than the frozen `300` is stamped `jackhammer/tactical-sweep/v1` with
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
lockups are visible in the decision records — run the README's baseline evaluation first (this
repository ships no `data/`; every artifact is reproduced locally), then:

```
uv run python scripts/inspect_run.py data/bench/greedy-shop.jsonl --seed 657P5QGW
```

which prints three `High Card score=0` plays.

**Why the frozen protocol keeps it.** Raising the cap moves published numbers, so it is a question
for the next protocol version — v3 — and not a patch to the current one. The outcome plateaus at `score_budget=2000` (mean highest ante 3.308, unchanged at 4000, 8000
and 16000), and 2379 — every subset of size <=5 of the largest hand this battery dealt, 13
cards — is the budget above which no scan in these runs can truncate at all. Going from 300 to 8000
scans 7.4% more combos for 14.5% more wall clock (82.9s -> 94.9s, 14 workers).

**If you are submitting an agent.** An agent with its own tactical layer is not subject to the cap,
but its measured margin over `greedy-shop` still carries this handicap on the exposed seeds. An
agent that reuses `GreedyTactical` inherits the cap outright.

Please open a narrowly reproducible issue when you find a simulator divergence or a gap in the
getting-started path.
