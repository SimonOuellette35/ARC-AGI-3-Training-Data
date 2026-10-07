"""Generate Phase-1 training data for the SK48 game.

Mirrors generate_ar25_training.py / generate_enqueue_training.py, but for the
local SK48 game (games/sk48/sk48.py). Each written episode is one multi-level
JSON in the shared encoder/dynamics schema:

    {
      "game_id": "sk48",
      "levels": [
        {
          "level_id": 0,
          "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
          "actions":      [a_0, a_1, ..., a_{T-1}], # length T
        },
        {"level_id": 1, "observations": ..., "actions": ...},
        ...
      ]
    }

actions[0] is the RESET that produced obs[0] (each level is recorded from a
fresh start via set_level, so obs[0] is that level's initial frame); actions[i]
for i>=1 is the *screen* action (post rotation-remap) that took the agent from
obs[i-1] to obs[i]. All actions are simple, so each entry is
{"type": "simple", "index": int}. Replaying the recorded screen actions
reproduces the recorded frames exactly.

--------------------------------------------------------------------------------
The game (decoded from the obfuscated source)
--------------------------------------------------------------------------------
SK48 is a "match the legend" bar-puzzle. Each level holds one or more *bars*: a
controllable head plus a contiguous line of segments extending in the head's
fixed orientation. A bar "threads" a coloured square when a segment lands on the
square's cell. Every playable (upper) bar is paired with a *legend* bar (drawn
along the bottom row) that is already threaded through a fixed sequence of
coloured squares. The level is WON when, for every pair, the sequence of colours
threaded by the player's bar equals the legend bar's sequence, position by
position (the engine lights a check-mark per matched slot; win == all lit).

The player moves the currently-selected bar with the four direction actions:

  * pressing along the bar's forward axis EXTENDS it (adds a base segment and
    pushes the tip forward, punching / pushing squares ahead of it),
  * pressing backward RETRACTS it (drops the base segment),
  * pressing perpendicular SLIDES the whole bar sideways along a rail,

pushing any coloured squares (and blocks) it runs into. ACTION6 clicks to switch
which bar is active; ACTION7 is undo. A per-level rotation augmentation remaps
the direction actions, and the palette is randomised per play.

SK48 action set (logical effects, before the per-level rotation remap):
    ACTION1 up (y-1)   ACTION2 down (y+1)   ACTION3 left (x-1)   ACTION4 right (x+1)
    ACTION6 click-select a bar    ACTION7 undo

--------------------------------------------------------------------------------
Which levels are generated
--------------------------------------------------------------------------------
We emit every level that can be won with the four DIRECTIONAL actions alone
(no ACTION6), because only those are expressible in the simple-index schema.
Crucially this is NOT the same as "single-pair": a bar threads a coloured square
whenever a *segment lands on the square's cell*, so a stationary player bar is
threaded simply by pushing squares onto it. A multi-pair level is therefore
directional-only solvable whenever a single moving bar (the one auto-selected at
level start) can push the squares so that EVERY pair's player bar ends up
threaded -- the other player bars never need to move, hence never need a click.
Whether that is possible is not something we can read off the layout, so we let
the engine-as-oracle search decide: we run the directional-only solve on every
level and emit it if a winning plan is found.

  * A level is EMITTED if the search finds a directional-only winning plan. This
    includes single-pair levels (the one auto-selected bar moves + pushes) and
    multi-pair levels whose extra player bars are threaded purely by pushed
    squares (e.g. level 3: two stationary vertical player bars, threaded by a
    separate auto-selected horizontal pusher bar).

  * A level is SKIPPED, with a WARN, if no directional-only plan exists within
    the search budget. That is the case when two or more player bars must each be
    *moved* to be threaded: moving a non-selected bar needs an ACTION6 click, and
    a click carries (x, y) data that the training schema (which stores only an
    integer `index`) cannot represent, so replaying a bare "index 6" would not
    reproduce the frames. The skip decision is cached alongside the plans so it
    stays a one-time cost. Each level is an independent trajectory in the schema,
    so an episode holding only the emitted levels is valid; level_id preserves
    the true (game, level) provenance.

--------------------------------------------------------------------------------
The solver (engine-as-oracle, no hand-coded geometry)
--------------------------------------------------------------------------------
Because the win predicate and every dynamics rule live in the engine, each level
is solved by weighted-A* over the REAL engine dynamics:

  * State successor = the settled board after one directional action, produced by
    stepping the engine with rendering skipped (fast_step). Wins are detected the
    instant the engine starts its win flash (lgdrixfno >= 0).
  * State is snapshotted compactly. Under directional-only play only the
    auto-selected bar ever moves (switching bars would need an ACTION6 click), so
    the canonical search key is that moving bar's (x, y, count) plus every
    square's (colour, x, y); the other bars are fixed. Each bar is ALWAYS a
    contiguous line of `count` segments from its head (extend/retract/slide all
    preserve this). Restoring mirrors the engine's own undo (uqclctlhyh), so it
    reproduces the dynamics exactly (verified against a deep-copy oracle) while
    keeping the frontier tiny.
  * Primary heuristic = number of unmatched legend slots summed over ALL pairs
    (0 at the goal), so weighted-A* stays goal-directed on multi-pair levels too.
    On single-pair levels a secondary "sortedness" potential -- how far the
    player's squares are from being laid out in the legend's colour order along
    one row -- breaks the large plateaus of the push/sort levels; on multi-pair
    levels that per-row layout notion does not apply, so the potential is 0 and
    the search leans on the unmatched-slot count alone.

The solve runs once per level on a rotation_k=0 reference game, yielding a plan
of LOGICAL directions. The puzzle geometry is identical across plays (only the
display rotation and palette are randomised, and both dynamics and matching are
palette-/rotation-invariant), so one cached logical plan wins every play: for
each episode the level is re-started (fresh random rotation + colours) and the
plan is replayed with each logical direction remapped to the screen action that
realises it under that play's rotation. Every emitted trajectory is replayed
through the real engine and verified to reach the win, so broken plays are
dropped rather than written.

Usage (run from the repo root):
    python solvers/generate_sk48_training.py --episodes 1000 \
        --out data/training_multi_level/sk48
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

# Repo root (parent of solvers/) -- that's where the games/ and utils/ packages
# live, so the engine + game imports below resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameAction, GameState  # noqa: E402
from games.sk48.sk48 import Sk48, sprites, hhvuoijeua, udenqlsrfq  # noqa: E402
from solvers.base_solver import BaseSolver  # noqa: E402
from utils.explore import (  # noqa: E402
    Action, EpsilonSchedule, ExplorationPolicy, RESET_ACTION)
from utils.rotation import ACTION_REMAP_INVERSE  # noqa: E402

# GameAction is not value-indexable (``GameAction(4)`` raises), so map ids -> enum members;
# used to turn an exploratory ``Action`` (a raw screen id 1..4) into the enum ``_drive`` wants.
_ID_TO_GAMEACTION = {int(a.value): a for a in GameAction}

GAME_ID = "sk48"

# Logical direction actions (the only ones a single-pair solve needs).
_DIR_ACTIONS = [
    GameAction.ACTION1,  # up    (y - 1)
    GameAction.ACTION2,  # down  (y + 1)
    GameAction.ACTION3,  # left  (x - 1)
    GameAction.ACTION4,  # right (x + 1)
]
_INT_TO_ACTION = {a.value: a for a in _DIR_ACTIONS}
_RESET_ACTION = {"type": "simple", "index": int(GameAction.RESET.value)}

# Guard on the internal step loop (movement settle + win flash never run long).
_STEP_GUARD = 2000


# ---------------------------------------------------------------------------
# Engine-as-oracle primitives
# ---------------------------------------------------------------------------

def _pairs(game):
    """The level's (player_bar, legend_bar) pairs; win == every pair threaded."""
    return list(game.xpmcmtbcv.items())


def _fast_step(game, action_id) -> bool:
    """Advance one directional action WITHOUT rendering (search inner loop).

    Returns True iff this action triggers the win (the engine begins its win
    flash, lgdrixfno >= 0, or has already queued the level transition)."""
    game._full_reset = False
    game._set_action(ActionInput(id=action_id))
    guard = 0
    while not game.is_action_complete():
        guard += 1
        if guard > _STEP_GUARD:
            break
        if game._next_level:
            return True
        game.step()
        if game.lgdrixfno >= 0:
            return True
    return False


def _snapshot(game):
    """Compact rest-state: every head's (x, y, segment count) plus every coloured
    square's (x, y). The player's bar is always a contiguous line of `count`
    segments from its head, so this is a complete, exact state."""
    heads = tuple((h, h.x, h.y, len(game.mwfajkguqx[h])) for h in game.mwfajkguqx)
    squares = tuple((s, s.x, s.y) for s in game.vbelzuaian)
    return heads, squares


def _restore(game, snap) -> None:
    """Rebuild `snap` into the live game (mirrors the engine's own undo,
    uqclctlhyh): drop all bar segments, replace every head's contiguous run of
    segments, and reposition every square. Because it is the engine's own
    reconstruction, the restored state steps identically to the original."""
    heads, squares = snap
    for segs in game.mwfajkguqx.values():
        for seg in segs:
            game.current_level.remove_sprite(seg)
        segs.clear()
    for head, x, y, count in heads:
        head.set_position(x, y)
        lx, ly = hhvuoijeua[head.rotation]
        dx, dy = lx * udenqlsrfq, ly * udenqlsrfq
        for i in range(count):
            seg = sprites["qtjqovumxf"].clone().set_rotation(head.rotation)
            game._remap_sprite(seg)
            seg.set_position(head.x + i * dx, head.y + i * dy)
            seg.set_layer(1 if head.rotation in (0, 180) else 0)
            game.current_level.add_sprite(seg)
            game.mwfajkguqx[head].append(seg)
    for sprite, x, y in squares:
        sprite.set_position(x, y)
    game.crbbymputr(game.vzvypfsnt, znewmxtdei=True)
    game.gvtmoopqgy()


def _state_key(game, snap, mover):
    """Canonical hashable state: the auto-selected (moving) bar's (x, y, count)
    plus every square's (colour, x, y). Under directional-only play no other bar
    moves (switching would need an ACTION6 click), so this is a full canonical
    key -- for both single-pair levels (mover == the player) and multi-pair
    levels (mover pushes squares onto the other, stationary player bars)."""
    heads, _ = snap
    ph = next((x, y, c) for h, x, y, c in heads if h is mover)
    sqs = tuple(sorted((int(s.pixels[1, 1]), s.x, s.y) for s in game.vbelzuaian))
    return ph, sqs


# ---------------------------------------------------------------------------
# Heuristic: unmatched-slot count (primary) + sortedness potential (tie-break)
# ---------------------------------------------------------------------------

def _make_evaluator(game):
    """Build an (uncovered, potential) evaluator closed over this level's target.

    uncovered  -- total number of legend slots not yet matched, summed over EVERY
                  player/legend pair; 0 at the goal, so weighted-A* on it stays
                  goal-directed on single- and multi-pair levels alike.
    potential  -- a plateau-breaking secondary that only makes sense for a single
                  pair: the player's squares must end up threaded in the legend's
                  colour order, and since a single-pair player bar extends
                  horizontally the not-yet-parked squares should lie in ONE row,
                  contiguous, in the legend colour order read left-to-right.
                  potential sums the colour-order mismatch, the number of extra
                  rows they span, and their horizontal overspread -- 0 once they
                  are correctly laid out. On multi-pair levels this per-row layout
                  notion does not apply, so potential is 0 and the search leans on
                  the unmatched-slot count alone.
    """
    pairs = _pairs(game)
    game.gvtmoopqgy()
    legend_seqs = [[int(s.pixels[1, 1]) for s in game.vjfbwggsd[legend]]
                   for _, legend in pairs]
    n_slots = [len(game.jdojcthkf[legend]) for _, legend in pairs]
    total_slots = sum(n_slots)
    single = len(pairs) == 1
    legend_row = max(s.y for s in game.vbelzuaian) if single else None

    def evaluate(g):
        g.gvtmoopqgy()
        uncovered = 0
        for (player, _legend), lseq, ns in zip(pairs, legend_seqs, n_slots):
            pseq = [int(s.pixels[1, 1]) for s in g.vjfbwggsd[player]]
            uncovered += sum(
                1 for i in range(ns)
                if i >= len(pseq) or pseq[i] != lseq[i]
            )
        if not single:
            return uncovered, 0
        legend_seq = legend_seqs[0]
        # Squares still up in the play area (not on the legend row).
        loose = [s for s in g.vbelzuaian if s.y < legend_row - udenqlsrfq]
        loose_sorted = sorted(loose, key=lambda s: (s.x, s.y))
        seq = [int(s.pixels[1, 1]) for s in loose_sorted]
        mism = sum(1 for i in range(min(len(seq), len(legend_seq)))
                   if seq[i] != legend_seq[i])
        mism += abs(len(seq) - len(legend_seq))
        rows = len(set(s.y for s in loose))
        if loose:
            xs = sorted(s.x for s in loose)
            overspread = (xs[-1] - xs[0]) // udenqlsrfq - (len(loose) - 1)
        else:
            overspread = 0
        potential = mism * 4 + max(0, rows - 1) * 3 + max(0, overspread)
        return uncovered, potential

    return evaluate, total_slots


# ---------------------------------------------------------------------------
# Weighted-A* solver over the real dynamics (one-time, per level)
# ---------------------------------------------------------------------------

def solve_level(level: int, weight: int = 2, node_cap: int = 6_000_000,
                time_cap: float = 500.0):
    """Solve `level` with DIRECTIONAL actions only, once on a rotation_k=0
    reference game. Only the auto-selected bar moves (switching would need an
    ACTION6 click); a multi-pair level is winnable this way when that one bar can
    push squares to thread every other, stationary player bar. Returns a list of
    LOGICAL direction indices (1..4) that reach the win, or None if the search
    exhausts its budget (i.e. no directional-only solution -- an ACTION6 click is
    needed to move a second player bar)."""
    game = Sk48()
    game.set_level(level)
    game._rotation_k = 0                 # dynamics are rotation-agnostic; solve

    mover = game.vzvypfsnt               # the only bar directional actions move
    evaluate, _ = _make_evaluator(game)

    start = _snapshot(game)
    u0, p0 = evaluate(game)
    counter = 0
    # Frontier entries: ((f, potential, uncovered), g, tiebreak, snap, path).
    frontier = [((weight * u0, p0, u0), 0, counter, start, [])]
    best_g = {_state_key(game, start, mover): 0}
    start_t = time.time()
    nodes = 0

    while frontier:
        _f, g_cost, _tb, snap, path = heapq.heappop(frontier)
        for action in _DIR_ACTIONS:
            _restore(game, snap)
            won = _fast_step(game, action)
            nodes += 1
            if won:
                return path + [action.value]
            child = _snapshot(game)
            key = _state_key(game, child, mover)
            ng = g_cost + 1
            if best_g.get(key, 1 << 30) <= ng:
                continue
            best_g[key] = ng
            counter += 1
            u, p = evaluate(game)
            heapq.heappush(
                frontier,
                ((ng + weight * u, p, u), ng, counter, child, path + [action.value]),
            )
            if nodes >= node_cap or time.time() - start_t > time_cap:
                return None
    return None


# ---------------------------------------------------------------------------
# Trajectory recording (real engine == ground truth)
# ---------------------------------------------------------------------------

def _screen_action(logical_index: int, rotation_k: int) -> GameAction:
    """The screen action to press so that, after the level's rotation remap, the
    engine applies `logical_index` (1..4)."""
    action = _INT_TO_ACTION[logical_index]
    k = rotation_k % 4
    return action


def _drive(game, action_id):
    """Perform one action (a GameAction) against the real (rendering) engine,
    returning (frames, won) -- EVERY frame it rendered, i.e. the action's whole
    animation, not just its settled board.

    Why the whole span: `solver.py` hands the policy the live
    ``FrameData.frame`` LIST, and sk48 slides over 2 frames on more than half its
    actions. Recording only the last one trained a token layout inference never
    uses (71% of live sk48 steps had a span length absent from the whole corpus),
    which is why sk48 scored 0/12 despite 98.3% teacher-forced accuracy.

    Unlike perform_action this still stops at the instant the win is detected
    (lgdrixfno >= 0), so the LAST captured frame is the clean solved board --
    before the win-flash blinks the squares and before the level transition."""
    game._full_reset = False
    game._set_action(ActionInput(id=action_id))
    frames: list = []
    won = False
    guard = 0
    while not game.is_action_complete():
        guard += 1
        if guard > _STEP_GUARD:
            break
        if game._next_level:
            break
        game.step()
        frames.append(game.camera.render(game.current_level.get_sprites()))
        if game.lgdrixfno >= 0:
            won = True
            break
    if not frames:  # a no-op / fully-blocked action rendered nothing new
        frames.append(game.camera.render(game.current_level.get_sprites()))
    return frames, won


# ---------------------------------------------------------------------------
# Solve / discovery / episode assembly
# ---------------------------------------------------------------------------

def _load_plan_cache(path: Path):
    """Load cached logical plans and cached skip reasons (both keyed by level
    index) written by a prior run. Returns (plans, unsolvable)."""
    if not path.exists():
        return {}, {}
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}, {}
    if raw.get("game_id") != GAME_ID:
        return {}, {}
    plans = {int(k): list(v) for k, v in raw.get("plans", {}).items()}
    unsolvable = {int(k): str(v) for k, v in raw.get("unsolvable", {}).items()}
    return plans, unsolvable


def _save_plan_cache(path: Path, plans: dict[int, list[int]],
                     unsolvable: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "game_id": GAME_ID,
        "plans": {str(k): v for k, v in plans.items()},
        "unsolvable": {str(k): v for k, v in unsolvable.items()},
    }
    path.write_text(json.dumps(payload))


def classify_and_solve(levels, weights, node_cap, time_cap, cache_path):
    """Return (plans, skipped): plans maps level -> directional-only logical plan
    for every level the search wins within budget, skipped maps the rest to a
    reason. A level is skipped only when NO directional-only plan is found, which
    means two or more player bars must each be moved -- moving a non-selected bar
    needs an ACTION6 click the simple-index schema cannot encode.

    Both plans and skip decisions are cached to `cache_path`: the solve (and the
    proof-of-no-solution) is a deterministic, one-time cost per level, so
    subsequent runs load instantly. For each un-cached level the solver tries the
    weight ladder `weights` in order -- a greedier weight cracks the deeper
    push/sort levels that a low weight cannot close within budget."""
    plans, unsolvable = _load_plan_cache(cache_path)
    skipped: dict[int, str] = {}
    for level in levels:
        if level in plans:
            print(f"    level {level}: loaded cached plan (len {len(plans[level])})")
            continue
        if level in unsolvable:
            print(f"    level {level}: cached skip ({unsolvable[level]})")
            skipped[level] = unsolvable[level]
            continue
        plan = None
        for weight in weights:
            plan = solve_level(level, weight, node_cap, time_cap)
            if plan is not None:
                print(f"    level {level}: solved at weight {weight} "
                      f"(plan len {len(plan)})")
                break
        if plan is None:
            reason = ("no directional-only solution within search budget: a "
                      "second player bar must be moved, which needs an ACTION6 "
                      "click the simple-index action schema cannot encode")
            print(f"    level {level}: no directional-only plan -- skipping")
            skipped[level] = reason
            unsolvable[level] = reason
        else:
            plans[level] = plan
    if cache_path is not None:
        _save_plan_cache(cache_path, plans, unsolvable)
    # Only report the levels actually requested this run.
    plans = {lvl: plans[lvl] for lvl in levels if lvl in plans}
    return plans, skipped


# ---------------------------------------------------------------------------
# BaseSolver subclass
# ---------------------------------------------------------------------------
class Sk48Solver(BaseSolver):
    """Thin ``BaseSolver`` binding for SK48 (harness = save + CLI; the shared schema).

    SK48 deviates from the native record loop in two ways, so ``solve_episode`` is
    overridden rather than driven by the base's plan-cache loop:

      * **Level skipping.** Only levels the DIRECTIONAL-only search can win are emitted
        (the others need an ACTION6 click the simple-index schema cannot encode). Those
        happen to be a tail, so the emitted set is the prefix that solved; ``level_id``
        preserves the true (game, level) provenance.
      * **Win-flash driving.** ``_drive`` stops the instant the win flash begins
        (``lgdrixfno >= 0``), before the squares blink and the level transitions, so the
        captured frame is the clean solved board.

    ``make_game`` passes the seed, so BOTH the display rotation AND the colour palette are
    a deterministic function of (seed, level_index); each episode is reproducible and a
    RESET (a fresh ``set_level``) restores a level's initial frame byte-for-byte.

    ``supports_recovery`` is True with ``recovery_mode = "reset"``. The cached LOGICAL plan
    is only valid from a level's INITIAL board, so recovery follows the reset paradigm:
    explore the epsilon prefix (raw directional actions perturb the board), then ONE RESET
    -- a fresh ``set_level`` restoring the initial board (same deterministic rotation/palette)
    -- then replay the plan remapped to that rotation. Because ``_record_level`` keeps its
    own custom loop (win-flash driving), the base's RESET-recovery is REPLICATED inline
    rather than inherited."""

    game_id = GAME_ID
    supports_recovery = True
    recovery_mode = "reset"

    def __init__(self, *args, plan_cache_path: Path | None = None,
                 levels_filter=None,
                 weights=(4, 2), node_cap: int = 6_000_000, time_cap: float = 600.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._plan_cache_path = plan_cache_path
        self._levels_filter = levels_filter
        self._weights = tuple(weights)
        self._node_cap = node_cap
        self._time_cap = time_cap
        self._plans = None            # solved (or loaded from cache) once, lazily

    def make_game(self, seed: int):
        # Pass the seed so BOTH the display rotation and the colour palette are a
        # pure function of (seed, level): each episode is reproducible and a RESET
        # restores a level's initial frame exactly (see Sk48._randomize_colors).
        return Sk48(seed=seed)

    def available_actions(self, game) -> list:
        return [int(a.value) for a in _DIR_ACTIONS]

    def _ensure_plans(self, game) -> dict:
        """Solve (or load) the directional-only plans once; cache them on the instance.
        Levels with no directional-only plan are dropped (they need an ACTION6 click)."""
        if self._plans is None:
            levels = (self._levels_filter
                      or list(range(len(game._levels))))
            cache_path = self._plan_cache_path or (
                Path("data/training_multi_level") / GAME_ID / "plans.sk48.json")
            plans, _skipped = classify_and_solve(
                levels, self._weights, self._node_cap, self._time_cap, cache_path)
            self._plans = plans
        return self._plans

    def solve_from(self, game, level_idx: int, seed: int) -> list:
        """The level's cached LOGICAL plan, remapped to the SCREEN actions that realise
        it under this play's rotation. Empty if the level has no directional-only plan."""
        plans = self._plans or {}
        logical = plans.get(level_idx)
        if not logical:
            return []
        rot = game._rotation_k
        return [ActionInput(id=_screen_action(li, rot)) for li in logical]

    def _record_level(self, game, level: int, logical_plan: list, *,
                      schedule: EpsilonSchedule | None = None,
                      exploration: ExplorationPolicy | None = None):
        """Re-start ``level`` and replay the logical plan, recording shared-schema
        step records. Returns (observations, actions) on a verified win, else
        (None, None).

        The rotation + palette are a deterministic function of (seed, level) (see
        ``Sk48._randomize_colors``), so re-starting a level via ``set_level``
        reproduces its initial frame exactly -- which is what makes a recovery RESET
        restore ``observations[0]`` byte-for-byte.

        RESET-recovery (recovery_mode="reset") is replicated here from the base
        ``record_level`` -- SK48 keeps a custom loop because it drives to the win flash.
        With ``schedule``/``exploration`` omitted this is pure optimal replay; with them
        present the epsilon prefix takes raw directional detours, then ONE RESET
        (a fresh ``set_level``) restores the initial board and the logical plan replays
        from there -- remapped to the board's rotation, so it still wins."""
        game.set_level(level)
        rotation_k = game._rotation_k
        self._note_rotation(game)      # the base records screen actions off this
        prev = np.asarray(game.camera.render(game.current_level.get_sprites()))
        observations = [prev.tolist()]
        actions = [self._reset_step()]

        recovering = exploration is not None
        reset_mode = self.recovery_mode == "reset"

        def do_reset() -> None:
            """Fresh ``set_level`` -> initial board (same deterministic rotation/palette),
            recorded as a RESET action, mirroring the base ``record_level``'s do_reset."""
            nonlocal prev, rotation_k
            game.set_level(level)
            rotation_k = game._rotation_k
            self._note_rotation(game)
            frame = np.asarray(game.camera.render(game.current_level.get_sprites()))
            observations.append(frame.tolist())
            actions.append(self._encode_step(
                Action(RESET_ACTION), optimal=[Action(RESET_ACTION)],
                phase="reset", changed=not np.array_equal(frame, prev)))
            prev = frame

        plan: list | None = None       # remaining screen GameActions for the current board
        resets = 0
        perturbed = False              # has an exploratory action moved us off-plan?
        committed = False              # reset-mode: optimal phase started, stop exploring
        for _ in range(self.step_guard):
            allow_explore = recovering and not (reset_mode and committed)
            exploring = False
            if allow_explore and schedule is not None and schedule.explore():
                exploring = True
            if schedule is not None:
                schedule.advance()

            if exploring:
                ex = exploration.action(prev)          # Action (raw screen id 1..4)
                screen = _ID_TO_GAMEACTION[ex.action_id]
                taken = ex
                optimal = None                          # reset-mode: relabel is stale
                phase = "explore"
                perturbed = True
                plan = None
            else:
                if reset_mode and not committed:
                    committed = True
                    if perturbed:
                        if resets >= self.max_resets:
                            return None, None
                        do_reset(); resets += 1; plan = None
                        continue
                if plan is None:
                    # UPRIGHT (logical) actions -- not screen ones. This solver's own
                    # ``_drive`` uses ``_set_action``, which bypasses the rotation
                    # wrapper, so the core game wants the logical action; the base
                    # converts to screen when it RECORDS. Converting here as well
                    # rotated twice and walked the block the wrong way at k != 0.
                    plan = [_INT_TO_ACTION[li] for li in logical_plan]
                if not plan:                            # plan exhausted without a win
                    return None, None
                screen = plan.pop(0)
                taken = Action(int(screen.value))
                optimal = [taken]
                phase = "expert"

            span, won = _drive(game, screen)
            span = np.asarray(span)                 # (k,H,W): the whole animation
            frame = span[-1]                        # settled board == the state
            changed = not np.array_equal(frame, prev)
            observations.extend(f.tolist() for f in span)
            actions.append(self._encode_step(taken, optimal=optimal, phase=phase,
                                             changed=changed, n_obs=len(span)))
            prev = frame
            if won:
                return observations, actions
            if game._state == GameState.GAME_OVER:      # ran out of energy
                if recovering and resets < self.max_resets:
                    do_reset(); resets += 1; plan = None
                    perturbed, committed = False, reset_mode
                    continue
                return None, None
        return None, None                                # step budget exhausted

    def solve_episode(self, seed: int, explore: bool = True):
        """Record every solvable level for one play (deterministic per-seed rotation +
        palette). Emits the solvable levels only; ``level_id`` keeps the true provenance.
        All-or-nothing over those levels: if any plan fails to win this play the episode
        is retried.

        The ``EpsilonSchedule`` + ``ExplorationPolicy`` are built ONCE per episode (only when
        ``explore and self.supports_recovery``), so the exploration arc spans the whole
        playthrough, and handed to every ``_record_level`` call -- mirroring the base
        ``solve_episode``."""
        game = self.make_game(seed)
        plans = self._ensure_plans(game)
        if not plans:
            return False, []
        do_explore = explore and self.supports_recovery
        schedule = (EpsilonSchedule(self.rng, center=self._explore_center,
                                    jitter=self._explore_jitter)
                    if do_explore else None)
        exploration = (ExplorationPolicy(self.available_actions(game), self.rng)
                       if do_explore else None)
        levels_out: list[dict] = []
        for level in sorted(plans):
            try:
                obs, acts = self._record_level(
                    game, level, plans[level],
                    schedule=schedule, exploration=exploration)
            except Exception:                            # noqa: BLE001 -- a bad play just fails
                return False, levels_out
            if obs is None:
                return False, levels_out
            levels_out.append({"level_id": level, "observations": obs, "actions": acts})
        return True, levels_out

    # ── CLI: keep SK48's solver knobs ────────────────────────────────────────────
    @classmethod
    def build_argparser(cls) -> argparse.ArgumentParser:
        p = super().build_argparser()
        p.add_argument("--levels", type=str, default="",
                       help="Comma-separated level indices to attempt (default: all).")
        p.add_argument("--weights", type=str, default="4,2",
                       help="Comma-separated weighted-A* heuristic weights, tried in "
                            "order per level until one solves (higher = greedier).")
        p.add_argument("--node-cap", type=int, default=6_000_000,
                       help="A* node budget per level per weight.")
        p.add_argument("--time-cap", type=float, default=600.0,
                       help="Wall-clock budget (s) per level per weight.")
        p.add_argument("--plan-cache", type=Path, default=None,
                       help="JSON of solved logical plans (default: "
                            "<out>/plans.sk48.json).")
        return p

    @classmethod
    def main(cls, argv=None) -> int:
        args = cls.build_argparser().parse_args(argv)
        out = args.out or (Path("data/training_multi_level") / cls.game_id)
        levels_filter = ([int(x) for x in args.levels.split(",") if x.strip()]
                         if args.levels.strip() else None)
        weights = [int(x) for x in args.weights.split(",") if x.strip()]
        cache_path = args.plan_cache or (out / "plans.sk48.json")
        solver = cls(rng=random.Random(args.seed), burst_prob=args.noise,
                     burst_mean=args.burst, plan_cache_path=cache_path,
                     levels_filter=levels_filter,
                     weights=weights, node_cap=args.node_cap, time_cap=args.time_cap)
        return solver.run(args.episodes, out, start_seed=args.start_seed,
                          max_attempts=args.max_attempts,
                          progress_every=args.progress_every,
                          explore=not args.no_exploration)


if __name__ == "__main__":
    sys.exit(Sk48Solver.main())
