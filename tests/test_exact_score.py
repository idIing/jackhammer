"""``preview_play`` is only useful if it is the engine.

``preview_play`` (``playground/exact_score.py``) rebuilds, by hand, the synthetic
``game_state`` that ``_handle_play_hand`` hands to ``score_hand``. Every key it
forgets, or sets to a value the engine would not hold yet, **defaults silently**:
no exception, no warning, just a preview that disagrees with the hand the engine
will actually run. ``GreedyTactical`` ranks every candidate play through it, so a
divergence reaches the reference agent's play selection.

That failure mode is invisible to a scan-cap or determinism test, so it needs a
gate of its own, and the gate has to be *differential* — the preview compared
against the engine, at states real play reaches. Two forms, because they fail
independently:

* :func:`test_preview_game_state_matches_the_engines_key_for_key` is the **class**
  gate. It intercepts ``score_hand`` on both paths for the same (state, play) and
  diffs the two ``game_state`` dicts over the exact key set the scorer reads. It
  fails on a key that is missing *and* on a key that is present with the wrong
  value, which a score comparison only catches when some card on the board happens
  to read it.
* :func:`test_preview_total_matches_the_engine_for_every_legal_play` is the
  **outcome** gate: over every play subset the tactical would enumerate, the
  preview's total and hand type equal the engine's own dry-run.

History: at engine ``4d6f19d`` this preview omitted ``hands_played``, ``skips`` and
``chips`` (jackhammer#9) and passed ``current_round_hands_played`` one ahead of the
engine, which increments both counters *after* ``score_hand`` returns. Four of the
nineteen keys the scorer reads. Measured effect of the four before the fix: 218 of
36,941 previewed plays across six battery seeds disagreed with the engine, every
one of them a Loyalty Card x4 the engine applied and the preview did not.

Run with: ``uv run --no-sync python -m pytest tests/test_exact_score.py -q``
"""

import pytest
from jackdaw.env import BalatroEnvironment, DirectAdapter

from jackhammer.bench import agents as agent_registry
from jackhammer.playground import exact_score
from jackhammer.playground.exact_score import preview_play
from jackhammer.playground.harness import _enumerate_play_combos, _legal_cards
from jackhammer.selfplay.runner import play_episode
from jackhammer.selfplay.tools import calculate_score

# Two frozen-battery seeds, walked by the reference agent. Real positions rather
# than hand-built boards: the bug this file exists for was found in play and is
# about states the harness actually visits.
SEEDS = ("PVRQ4K5A", "4NNGD2DN")

# Every key ``score_hand`` reads out of ``game_state``: the ScoringContext
# construction (``jackdaw/engine/scoring.py:437-458``) plus the two Mr. Bones
# death checks (``scoring.py:535,908``). A key added there and not here narrows
# this gate silently, which is why the count is asserted below.
READ_KEYS = (
    "ancient_suit",
    "chips",
    "consumable_usage_tarot",
    "current_round_hands_played",
    "deck_cards_remaining",
    "discards_left",
    "discards_used",
    "enhanced_card_count",
    "hands_left",
    "hands_played",
    "idol_card",
    "joker_slots",
    "mail_card_id",
    "money",
    "playing_cards_count",
    "skips",
    "starting_deck_size",
    "steel_tally",
    "stone_tally",
)

MAX_POSITIONS = 6

# The jokers that read one of the four keys this file's history is about: Loyalty
# Card (``hands_played`` run-wide), Throwback (``skips``), Mr. Bones (``chips``),
# DNA and Sixth Sense (``current_round_hands_played`` == 0).
_KEY_READING_JOKERS = frozenset(
    {"j_loyalty_card", "j_throwback", "j_mr_bones", "j_dna", "j_sixth_sense"}
)


class _Enough(Exception):
    """Stop the walk once enough positions have been inspected."""


def _walk(seed: str, visit, limit: int = MAX_POSITIONS, start: int = 0) -> int:
    """Drive ``greedy-shop`` over *seed*, calling ``visit(env, mask)`` at each
    SELECTING_HAND position from the *start*-th onward, for *limit* of them.

    ``visit`` runs *after* the agent has chosen, so nothing it does can change the
    trajectory, and it reads ``env._adapter.raw_state`` — the pristine
    post-restore dict, which is the same one ``harness.best_play_scan`` previews
    against.
    """
    env = BalatroEnvironment(adapter_factory=DirectAdapter)
    decide_inner = agent_registry.get("greedy-shop").make_decider(env, seed)
    reached = visited = 0

    def decide(raw_state, mask, history):
        nonlocal reached, visited
        playable = str(raw_state.get("phase", "")).upper().endswith("SELECTING_HAND")
        out = decide_inner(raw_state, mask, history)
        if playable and _legal_cards(mask):
            if reached >= start:
                visit(env, mask)
                visited += 1
                if visited >= limit:
                    raise _Enough
            reached += 1
        return out

    try:
        play_episode(env, seed, decide, None, None, max_steps=2000)
    except _Enough:
        pass
    return visited


def test_read_keys_covers_everything_the_scorer_reads():
    """The audited key set is the scorer's, not a remembered subset of it."""
    import inspect

    from jackdaw.engine import scoring

    src = inspect.getsource(scoring)
    reads = {
        line.split('gs.get("', 1)[1].split('"', 1)[0]
        for line in src.splitlines()
        if 'gs.get("' in line
    }
    assert reads == set(READ_KEYS), f"scorer reads {sorted(reads - set(READ_KEYS))} unaudited"


@pytest.mark.parametrize("seed", SEEDS)
def test_preview_game_state_matches_the_engines_key_for_key(seed, monkeypatch):
    real = exact_score.score_hand
    captured: dict[str, dict] = {}

    def spy(tag):
        def wrapper(*args, **kwargs):
            captured[tag] = {k: kwargs["game_state"].get(k, "<ABSENT>") for k in READ_KEYS}
            return real(*args, **kwargs)

        return wrapper

    from jackdaw.engine import game as game_mod
    from jackdaw.engine import scoring as scoring_mod
    from jackdaw.env import FactoredAction

    compared = 0

    def visit(env, mask):
        nonlocal compared
        combo = tuple(_enumerate_play_combos(_legal_cards(mask), mask, 300)[0])

        # The engine's own game_state, from a dry-run that is rolled straight back.
        snapshot = env.get_state()
        monkeypatch.setattr(scoring_mod, "score_hand", spy("engine"))
        try:
            env.step(FactoredAction(action_type=0, card_target=combo))
        except Exception:  # noqa: BLE001 — boss play restriction; nothing to compare
            return
        finally:
            monkeypatch.setattr(scoring_mod, "score_hand", real)
            env.load_state(snapshot)

        monkeypatch.setattr(exact_score, "score_hand", spy("preview"))
        try:
            preview_play(env._adapter.raw_state, combo)
        finally:
            monkeypatch.setattr(exact_score, "score_hand", real)

        assert captured["preview"] == captured["engine"], (
            f"seed {seed}, play {combo}: preview's game_state differs from the engine's"
        )
        compared += 1

    assert game_mod is not None  # the dry-run path under test
    _walk(seed, visit)
    assert compared > 0, f"no comparable position reached on {seed}"


# The window of ``PVRQ4K5A`` that held the four-key divergence. Position 28 of that
# walk is a Loyalty Card board on which the engine applied x4 and the preview did
# not, on all 218 enumerated plays; positions 26-30 bracket it so the case survives
# a small trajectory shift. Deliberately pinned rather than sampled: a generic early
# position holds no joker that reads any of the four keys, which is exactly why the
# outcome gate alone would have passed through the original bug.
_DIVERGENCE_WINDOW = ("PVRQ4K5A", 26, 5)


@pytest.mark.parametrize("seed,start,limit", [_DIVERGENCE_WINDOW, ("4NNGD2DN", 0, 3)])
def test_preview_total_matches_the_engine_for_every_legal_play(seed, start, limit):
    checked = 0
    saw_key_reader = False

    def visit(env, mask):
        nonlocal checked, saw_key_reader
        raw = env._adapter.raw_state
        if any(
            getattr(j, "center_key", None) in _KEY_READING_JOKERS for j in (raw.get("jokers") or [])
        ):
            saw_key_reader = True
        for combo in _enumerate_play_combos(_legal_cards(mask), mask, 300):
            truth = calculate_score(env, list(combo))
            if "error" in truth:
                continue  # engine-illegal selection; the preview is not consulted
            sr = preview_play(env._adapter.raw_state, tuple(combo))
            assert (int(sr.total), sr.hand_type) == (int(truth["score"]), truth["hand_type"]), (
                f"seed {seed}, play {list(combo)}: preview "
                f"{int(sr.total)}/{sr.hand_type} != engine "
                f"{int(truth['score'])}/{truth['hand_type']}"
            )
            checked += 1

    _walk(seed, visit, limit=limit, start=start)
    assert checked > 0, f"no play scored on {seed}"
    if (seed, start, limit) == _DIVERGENCE_WINDOW:
        assert saw_key_reader, (
            f"{seed} positions {start}..{start + limit - 1} no longer hold a joker that "
            f"reads one of the four keys, so this window has stopped covering the "
            f"regression it was pinned to. Re-pin it: walk the seed, find a position "
            f"whose board holds one of {sorted(_KEY_READING_JOKERS)}, and move the window."
        )


def test_the_repaired_keys_are_read_from_the_state_not_hard_coded():
    """Sentinel values must travel from the live state through to the scorer.

    The differential tests above cannot catch a constant. No baseline ever skips a
    blind, so ``skips`` is 0 at every position this battery reaches; ``chips`` is 0
    on the first hand of every round and the counters are small. A ``preview_play``
    that wrote ``synth["skips"] = 0`` instead of reading ``gs`` would pass every
    other test in this file. This one puts a value in the live state that the
    battery never produces and asserts the scorer is handed it.
    """
    import copy

    real = exact_score.score_hand
    captured: dict[str, object] = {}

    def spy(*args, **kwargs):
        captured.update(kwargs["game_state"])
        return real(*args, **kwargs)

    sentinels = {"skips": 7, "chips": 4321, "hands_played": 41}
    checked = 0

    def visit(env, mask):
        nonlocal checked
        if checked:
            return
        gs = copy.deepcopy(env._adapter.raw_state)
        gs.update(sentinels)
        gs["current_round"]["hands_played"] = 3
        combo = tuple(_enumerate_play_combos(_legal_cards(mask), mask, 300)[0])

        exact_score.score_hand = spy
        try:
            preview_play(gs, combo)
        finally:
            exact_score.score_hand = real

        for key, value in sentinels.items():
            assert captured[key] == value, f"{key} did not reach the scorer"
        assert captured["current_round_hands_played"] == 3
        checked = 1

    _walk(SEEDS[0], visit, limit=1)
    assert checked, f"no position reached on {SEEDS[0]}"
