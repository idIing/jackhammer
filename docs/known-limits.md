# Known limits

- **Simulator proxy:** Jackdaw is not live Balatro. The v1 pin has deterministic and engine tests,
  but no claim of complete simulator equivalence follows from those checks.
- **One configuration:** v1 covers Red Deck, White Stake, one engine pin, and one IID seed battery.
- **Weak baselines:** all three built-ins won 0/240. Mean highest ante separates them, but the slate
  does not represent the field's strongest agents.
- **Cheapest-Joker policy:** `greedy-shop` buys only Jokers, never evaluates their text or synergy,
  never rerolls or sells, takes the first pack card, and ignores vouchers and consumables.
- **Tactical scan cap:** `GreedyTactical` exact-scores at most `score_budget=300` card subsets per
  play decision. That is exhaustive for a standard 8-card hand (218 subsets of size <=5), but above
  8 cards it truncates, and because it enumerates small-k first, the subsets it drops are the
  largest — the 5-card straights, flushes and full houses. Hand size rises with Juggler
  (`h_size = 1`) and the Paint Brush / Palette vouchers, so this is reachable in ordinary play: in
  the v1 runs, 19/240 `greedy-shop` and 8/240 `random-shop` games ended at hand size >= 9 from
  verified non-decaying sources. At hand size 9 only 45 of the 126 five-card subsets are scored; at
  10, none are. Both shop arms share the layer, so the paired comparison stays valid, but "exact
  scoring" overstates what happens in those runs. Raising the cap would change published numbers and
  is therefore a v2 question, not a v1 patch.
- **Published battery:** the 240 training seeds are public and therefore overfittable. The old
  validation split has already been consumed and is retired, not a reusable secret leaderboard.
- **No live client:** the benchmark and text run inspector work headlessly. This repository does not
  distribute Balatro, its source, or its assets.
- **Dataset selection:** a diagnostic manifest can answer a scoped coverage question but cannot be
  relabeled as ordinary-distribution policy strength. Sampling and overlap must be reported.
- **Artifact schema:** v1 validates the stable envelope and preserves the raw decision records by
  reference; it does not cryptographically sign results or fully validate every nested summary field.

Please open a narrowly reproducible issue when you find a simulator divergence or a front-door gap.
