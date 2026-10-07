"""Generate Phase-1 training data for the AR25 game.

Mirrors generate_maze_training.py, but for the local AR25 game
(games/ar25/ar25.py). An expert solver beats every level of a seed and the
trajectory is written as one multi-level episode JSON in the format expected by
the encoder/dynamics stack:

    {
      "game_id": "ar25",
      "levels": [
        {
          "level_id": 0,
          "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
          "actions":      [a_0, a_1, ..., a_{T-1}]  # length T
        },
        {"level_id": 1, "observations": ..., "actions": ...},
        ...
      ]
    }

actions[0] is the RESET that produced obs[0] (game start for level 0, the level
transition for the rest); actions[i] for i>=1 is the action that took the agent
from obs[i-1] to obs[i]. All actions are simple, so each entry is
{"type": "simple", "index": int}.

One AR25 seed is already a multi-level game (8 levels), so each WIN seed yields
one complete multi-level episode -- no cross-seed bundling needed.

--------------------------------------------------------------------------------
The game (decoded from the obfuscated source)
--------------------------------------------------------------------------------
AR25 is a "complete the mirror symmetry" puzzle. Each level holds:

  * one or more movable shadow PIECES (sprites tagged "gljpmsnsnx"),
  * one or two movable MIRROR AXES (sprites tagged "edyhkfhkcf"):
      - a vertical axis ("zxikvwjsyl") reflects x -> 2*ax - x, moves only in x,
      - a horizontal axis ("ezdsyuixsn") reflects y -> 2*ay - y, moves only in y,
  * a set of yellow GOAL cells (sprites tagged "vrfjzqaker").

The engine renders each piece together with all of its reflections across the
axes (a small reflection group). The level is WON when this reflected composite
covers every goal cell. The player moves the currently-selected sprite one cell
at a time with the four direction actions, and cycles the selection with ACTION5.
A per-level rotation augmentation remaps the direction actions, and the piece
shapes / start positions / goal layout are randomized per seed.

--------------------------------------------------------------------------------
The solver (engine-as-oracle, no hand-coded geometry)
--------------------------------------------------------------------------------
Because the win predicate is purely a function of sprite POSITIONS, and the
engine itself computes the reflected composite (`jhajdrieqn`) and the win check
(`etzeptsuxx`) cheaply, we solve each level in two phases:

  1. SOLVE -- block coordinate descent over the movable sprites. Repeatedly scan
     one sprite over all its legal positions (others fixed), greedily minimizing
     the number of uncovered goal cells, until zero remain. Random restarts
     escape the occasional local minimum. This yields a winning *configuration*
     (a target (x, y) for each movable sprite).

  2. PLAN -- turn that configuration into key presses: for each sprite that must
     move, select it with ACTION5 and walk it to its target with the four
     direction actions. The direction actions are rotation-corrected so the raw
     action sent produces the intended logical move under the level's random
     rotation.

The plan is then REPLAYED through the real `perform_action`, which is the ground
truth: it captures the exact rendered frames and guarantees the recorded
trajectory actually reaches GameState.WIN. Seeds whose plan fails to win (or
would exceed the energy budget) are skipped, so every written episode is a
verified full clear.

AR25 action set:
    ACTION1 up (y-1)   ACTION2 down (y+1)   ACTION3 left (x-1)   ACTION4 right (x+1)
    ACTION5 cycle selected sprite
(direction actions above are the *logical* effects, before rotation remap)

Usage (run from the repo root):
    python solvers/generate_ar25_training.py --episodes 1000 --out data/ar25_training_multi_level
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

# Repo root (parent of solvers/) -- that's where the adapters/ and games/
# packages live, so the engine + game imports below resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from arcengine import ActionInput, GameAction, GameState
from games.ar25.ar25 import Ar25
from utils.rotation import ACTION_REMAP_INVERSE
from solvers.base_solver import BaseSolver
from utils.explore import EpsilonSchedule, ExplorationPolicy

GAME_ID = "ar25"

# GameAction is not value-indexable (``GameAction(4)`` raises), so map ids -> the
# enum member explicitly. Used to turn an exploration Action's id back into the
# engine action for the RESET-recovery prefix.
_ID_TO_GAMEACTION = {int(a.value): a for a in GameAction}


def frame_np(fr):
    """A frame (ndarray or nested list) as an ndarray, for cheap equality tests."""
    return np.asarray(fr)


def _as_list(fr):
    """A frame as a nested list (the on-disk observation form)."""
    return fr.tolist() if hasattr(fr, "tolist") else fr

A1 = GameAction.ACTION1  # logical up    (y - 1)
A2 = GameAction.ACTION2  # logical down  (y + 1)
A3 = GameAction.ACTION3  # logical left  (x - 1)
A4 = GameAction.ACTION4  # logical right (x + 1)
A5 = GameAction.ACTION5  # cycle selection

# Sprite tags (decoded).
TAG_AXIS = "edyhkfhkcf"   # a mirror axis
TAG_VAXIS = "zxikvwjsyl"  # vertical axis (reflects horizontally, moves in x)
TAG_HAXIS = "ezdsyuixsn"  # horizontal axis (reflects vertically, moves in y)


# ── Win evaluation ───────────────────────────────────────────────────────────

def _uncovered(game: Ar25) -> int:
    """Number of goal cells NOT covered by the reflected composite -- the engine's
    own win predicate (`etzeptsuxx`) is True exactly when this is 0."""
    composite = game.jhajdrieqn()
    count = 0
    for goal in game.mqedygxur:
        if composite[goal.y, goal.x] < 0:
            count += 1
    return count


def _is_axis(s) -> bool:
    return TAG_AXIS in s.tags


def _is_vaxis(s) -> bool:
    return TAG_VAXIS in s.tags


def _is_haxis(s) -> bool:
    return TAG_HAXIS in s.tags


# ── Phase 1: solve for a winning configuration ───────────────────────────────

def _scan_sprite(game: Ar25, s) -> int:
    """Move sprite `s` to the position (over its full legal range) that minimizes
    uncovered goal cells, with every other sprite held fixed. Vertical axes only
    slide in x, horizontal axes only in y, pieces translate freely. Returns the
    best uncovered count and leaves `s` parked there."""
    h, w = s.pixels.shape
    ox, oy = s._x, s._y
    best = (_uncovered(game), ox, oy)

    if _is_axis(s) and _is_vaxis(s):
        for x in range(0, game.huzumkfia):
            s._x, s._y = x, oy
            u = _uncovered(game)
            if u < best[0]:
                best = (u, x, oy)
    elif _is_axis(s) and _is_haxis(s):
        for y in range(0, game.aqnahsxpq):
            s._x, s._y = ox, y
            u = _uncovered(game)
            if u < best[0]:
                best = (u, ox, y)
    else:
        for y in range(0, game.aqnahsxpq - h + 1):
            for x in range(0, game.huzumkfia - w + 1):
                s._x, s._y = x, y
                u = _uncovered(game)
                if u < best[0]:
                    best = (u, x, y)

    s._x, s._y = best[1], best[2]
    return best[0]


def _solve_config(game: Ar25, rng: random.Random, restarts: int = 12):
    """Block coordinate descent to find a winning configuration. Returns the list
    of target (x, y) for each sprite in `game.xefwpvwoh`, or None if no restart
    converges. Mutates sprite positions; the caller is expected to restore them."""
    movables = game.xefwpvwoh
    original = [(s._x, s._y) for s in movables]

    for r in range(restarts):
        # Restart 0 keeps the natural start positions (usually closest to a
        # solution -> shortest plan); later restarts jitter to escape minima.
        for i, s in enumerate(movables):
            if r == 0:
                s._x, s._y = original[i]
            else:
                h, w = s.pixels.shape
                s._x = rng.randint(0, max(0, game.huzumkfia - w))
                s._y = rng.randint(0, max(0, game.aqnahsxpq - h))

        prev = None
        for _ in range(40):
            for s in movables:
                _scan_sprite(game, s)
            u = _uncovered(game)
            if u == 0:
                return [(s._x, s._y) for s in movables]
            if u == prev:
                break  # stuck on this restart
            prev = u

    return None


def _legal_positions(game: Ar25, s):
    """All in-bounds positions a movable sprite may occupy: a vertical axis slides
    only in x, a horizontal axis only in y, a piece translates freely."""
    W, H = game.huzumkfia, game.aqnahsxpq
    if _is_axis(s) and _is_vaxis(s):
        return [(x, s._y) for x in range(W)]
    if _is_axis(s) and _is_haxis(s):
        return [(s._x, y) for y in range(H)]
    h, w = s.pixels.shape
    return [(x, y) for x in range(W - w + 1) for y in range(H - h + 1)]


def _min_travel_config(game: Ar25, max_configs: int = 20000):
    """Exact MINIMUM-TRAVEL winning configuration, by enumerating the joint sprite
    position space when it is small enough (e.g. the one-axis / one-piece
    reflection levels such as level 1). Plan length is the total Manhattan travel
    (plus a couple of ACTION5 cycles), so minimizing travel yields the shortest
    plan -- and, crucially, avoids the coordinate-descent detour that drives the
    axis all the way to the goal (overshooting the reflected 'shadow') and then
    drags the piece back. Returns the target (x, y) per movable, or None if the
    space is too large (caller falls back to coordinate descent). Restores the
    sprites' original positions before returning."""
    movs = game.xefwpvwoh
    init = [(s._x, s._y) for s in movs]
    pos_lists = [_legal_positions(game, s) for s in movs]

    total = 1
    for pl in pos_lists:
        total *= len(pl)
        if total > max_configs:
            return None                       # too big -> use coordinate descent

    best = None
    for combo in itertools.product(*pos_lists):
        for s, (x, y) in zip(movs, combo):
            s._x, s._y = x, y
        if _uncovered(game) == 0:
            travel = sum(abs(combo[i][0] - init[i][0]) + abs(combo[i][1] - init[i][1])
                         for i in range(len(movs)))
            if best is None or travel < best[0]:
                best = (travel, list(combo))

    for s, (x, y) in zip(movs, init):         # restore
        s._x, s._y = x, y
    return best[1] if best else None


# ── Phase 2: turn a configuration into key presses ───────────────────────────

def _raw(logical: GameAction, k: int) -> GameAction:
    """The raw action to send so that, after the level's rotation remap (k), the
    engine applies the given *logical* direction."""
    k %= 4
    return logical if k == 0 else ACTION_REMAP_INVERSE[k][logical]


def _plan_actions(game: Ar25, config, k: int) -> list[GameAction]:
    """Single forward sweep through the selection ring: for each sprite that must
    move, select it (ACTION5) then walk it to its target. Moving along one axis
    at a time between two in-bounds endpoints never leaves the grid, and pieces
    may overlap, so each step is guaranteed to land."""
    movables = game.xefwpvwoh
    n = len(movables)
    init = [(s._x, s._y) for s in movables]
    need = {i: config[i] for i in range(n) if config[i] != init[i]}

    cur = movables.index(game.llludejph) if game.llludejph in movables else 0
    cx = {i: init[i][0] for i in range(n)}
    cy = {i: init[i][1] for i in range(n)}

    actions: list[GameAction] = []
    idx = cur
    remaining = set(need)
    guard = 0
    while remaining:
        guard += 1
        if guard > 4 * n + 5:  # defensive: never loop forever
            break
        if idx in remaining:
            tx, ty = need[idx]
            dx, dy = tx - cx[idx], ty - cy[idx]
            for _ in range(abs(dx)):
                actions.append(_raw(A4 if dx > 0 else A3, k))
            for _ in range(abs(dy)):
                actions.append(_raw(A2 if dy > 0 else A1, k))
            remaining.discard(idx)
        if remaining:
            actions.append(A5)          # cycle to the next selectable sprite
            idx = (idx + 1) % n
    return actions


def _restore(game: Ar25, positions, selected_idx: int, steps: int) -> None:
    """Reset the live game to a level's initial state after coordinate descent."""
    for s, (x, y) in zip(game.xefwpvwoh, positions):
        s._x, s._y = x, y
    game.llludejph = game.xefwpvwoh[selected_idx] if selected_idx >= 0 else None
    game.zdrbnrjbr.current_steps = steps
    game.hqiorgefxt()


# ── Episode assembly ─────────────────────────────────────────────────────────

def _solve_episode_impl(seed: int, config_seeds: int = 8, *,
                        schedule=None, exploration=None, solver=None):
    """Solve every level for one seed. Returns (final_state, levels). On failure
    (a level can't be solved, the plan would exceed the energy budget, or the
    replay doesn't win) returns the partial trajectory so the caller can decide
    to skip it.

    When ``exploration`` is supplied (recovery ON), each level opens with an
    epsilon-decayed exploration prefix, then -- unless it wins outright -- ONE
    RESET restores the level's initial state (the engine RESET action does a
    level_reset: re-clones the clean level, re-seeds energy and clears any
    GAME_OVER), after which the two-phase expert re-solves and replays a winning
    plan from that restored state. See ``BaseSolver._reset_prefix``."""
    game = Ar25(seed=seed)
    game.perform_action(ActionInput(id=GameAction.RESET))  # full reset -> level 0
    num_levels = len(game._levels)
    levels: list[dict] = []
    reset_action = {"type": "simple", "index": int(GameAction.RESET.value)}

    for level_idx in range(num_levels):
        # Clean initial frame for this level (rotation + HUD applied by camera).
        initial_frame = game.camera.render(game.current_level.get_sprites()).tolist()
        observations: list[list[list[int]]] = [initial_frame]
        actions: list[dict] = [reset_action]

        # RESET-recovery exploration prefix (no-op unless recovery is enabled).
        # The engine RESET action does a level_reset (re-clones the clean level,
        # re-seeds energy, clears GAME_OVER), so the two-phase expert below can
        # re-solve from the restored initial state. Positions / selection / energy
        # are (re)captured AFTER the prefix so the plan matches the live state.
        if exploration is not None and solver is not None:
            def _drive_explore(act, _lvl=level_idx):
                if act.is_click:
                    ai = ActionInput(id=GameAction.ACTION6,
                                     data={"x": act.click_xy[0], "y": act.click_xy[1]})
                else:
                    ai = ActionInput(id=_ID_TO_GAMEACTION[act.action_id])
                fd = game.perform_action(ai)
                won = game._state == GameState.WIN or game.level_index > _lvl
                if fd.frame:
                    # Keep the WHOLE animation (`_reset_prefix` records n_obs from
                    # its length) -- that is what the live roll-out feeds the
                    # policy. On the winning action the engine has already stepped
                    # into the next level, so only frame[0] (the win composite,
                    # still showing THIS level) belongs to this level's stream.
                    fr = fd.frame[0] if won else fd.frame
                else:                              # action rendered nothing new
                    fr = game.camera.render(game.current_level.get_sprites())
                term = ("win" if won
                        else "over" if game._state == GameState.GAME_OVER else None)
                return frame_np(fr), term

            def _reset_to_level():
                game.perform_action(ActionInput(id=GameAction.RESET))
                return game.camera.render(game.current_level.get_sprites())

            solver._reset_prefix(
                schedule=schedule, exploration=exploration,
                prev=frame_np(initial_frame),
                drive_explore=_drive_explore, reset_to_level=_reset_to_level,
                record_obs=lambda fr: observations.append(_as_list(fr)),
                record_act=lambda d: actions.append(d))

            # If an exploratory action already won the level, finalize it here --
            # the engine has advanced to the next level (or the game is WON).
            if game._state == GameState.WIN or game.level_index > level_idx:
                levels.append({"level_id": level_idx, "observations": observations,
                               "actions": actions})
                if game._state == GameState.WIN:
                    break
                continue

        k = game._rotation_k
        pos0 = [(s._x, s._y) for s in game.xefwpvwoh]
        sel0 = game.xefwpvwoh.index(game.llludejph) if game.llludejph in game.xefwpvwoh else -1
        steps0 = game.zdrbnrjbr.current_steps

        # Phase 1: find a winning configuration. Prefer the EXACT minimum-travel
        # config where the position space is small enough to enumerate (the
        # one-axis/one-piece reflection levels like level 1) -- this gives the
        # shortest plan and avoids coordinate descent's axis-overshoot detour.
        # Fall back to block coordinate descent (retry restarts) on big levels.
        config = _min_travel_config(game)
        _restore(game, pos0, sel0, steps0)
        if config is None:
            for cs in range(config_seeds):
                config = _solve_config(game, random.Random(cs))
                _restore(game, pos0, sel0, steps0)
                if config is not None:
                    break
        if config is None:
            levels.append({"level_id": level_idx, "observations": observations, "actions": actions})
            return GameState.GAME_OVER, levels

        # Phase 2: plan key presses; reject plans that would drain the energy bar.
        plan = _plan_actions(game, config, k)
        if not plan or len(plan) >= steps0:
            levels.append({"level_id": level_idx, "observations": observations, "actions": actions})
            return GameState.GAME_OVER, levels

        # Replay through the real engine (ground truth) and record frames.
        advanced = False
        for action in plan:
            frame_data = game.perform_action(ActionInput(id=action))
            advanced = game._state == GameState.WIN or game.level_index > level_idx
            # Record the action's WHOLE animation, not just its settled frame:
            # `solver.py` feeds the live `FrameData.frame` list to the policy, so
            # keeping one frame per action trains a token layout inference never
            # sees. `n_obs` carries the span length; `_normalise_levels` derives
            # phase/optimal/changed from it (see record_level's alignment rule).
            #
            # EXCEPT on the winning action: the engine has already stepped into
            # the next level by then, so only frame[0] -- the win composite still
            # showing THIS level -- belongs to this level's stream.
            span = ([frame_data.frame[0]] if advanced
                    else list(frame_data.frame) or
                    [game.camera.render(game.current_level.get_sprites())])
            observations.extend(span)
            actions.append({"type": "simple", "index": int(action.value),
                            "n_obs": len(span)})
            if game._state == GameState.GAME_OVER:
                levels.append({"level_id": level_idx, "observations": observations, "actions": actions})
                return GameState.GAME_OVER, levels
            if advanced:
                break

        levels.append({"level_id": level_idx, "observations": observations, "actions": actions})
        if not advanced:
            return GameState.GAME_OVER, levels  # plan didn't actually win
        if game._state == GameState.WIN:
            break

    return game._state, levels


# ---------------------------------------------------------------------------
# BaseSolver adapter
# ---------------------------------------------------------------------------
#
# AR25 runs on the native Ar25 game, but its episode assembly is bespoke: a fresh
# game per seed (piece shapes / start positions / goal layout randomize per seed,
# so plans are seed-dependent and NOT cached across seeds), a two-phase
# solve-then-plan expert, an energy-budget rejection, rotation-corrected raw
# actions, and an early break once the final level wins. That doesn't fit the
# base's per-level record/replay loop, so we override ``solve_episode`` to call
# the original expert and keep the exact trajectory, borrowing only the base's
# CLI (``main``/``run``) and ``{"game_id","levels":[...]}`` save schema.
#
# ``supports_recovery = True`` with ``recovery_mode = "reset"``: the plan is built
# as a single forward sweep for a specific winning configuration, so it CANNOT
# re-plan from an arbitrary perturbed state (that would be ``"replan"``). Instead
# each level opens with the episode-wide epsilon prefix exploring freely, then ONE
# RESET (the engine's level_reset re-clones the clean level, re-seeds the energy
# bar and clears any GAME_OVER) restores the initial state, from which the
# two-phase solve-then-sweep expert re-solves and replays a verified WIN -- the
# human "flail, hit reset, then solve" arc.
class Ar25Solver(BaseSolver):
    #: Drives the engine through ``game.perform_action`` -- the AGENT-facing entry
    #: point, which de-rotates whatever it is handed. So this generator's plan is
    #: already in SCREEN space (it converts it itself), and the base must NOT
    #: convert again when recording. Generators that use the base's ``drive``
    #: (``_set_action``, which bypasses the wrapper) leave this False.
    plans_in_screen_space = True
    game_id = GAME_ID
    supports_recovery = True
    recovery_mode = "reset"

    def make_game(self, seed: int):
        game = Ar25(seed=seed)
        game.perform_action(ActionInput(id=GameAction.RESET))
        return game

    def solve_episode(self, seed: int, explore: bool = True):
        """Reproduce the original per-seed episode via the two-phase expert;
        succeed only on a verified full clear (GameState.WIN). Returns
        ``(ok, levels)``.

        Exploration (the episode-wide epsilon prefix + RESET-recovery) engages only
        when requested AND ``supports_recovery`` is set; the schedule + policy are
        built ONCE here so the exploration arc spans the whole episode."""
        do_explore = explore and self.supports_recovery
        schedule = exploration = None
        if do_explore:
            probe = self.make_game(seed)
            schedule = EpsilonSchedule(self.rng, center=self._explore_center,
                                       jitter=self._explore_jitter)
            exploration = ExplorationPolicy(self.available_actions(probe), self.rng)
        state, levels = _solve_episode_impl(seed, schedule=schedule,
                                            exploration=exploration, solver=self)
        return state == GameState.WIN, levels


if __name__ == "__main__":
    sys.exit(Ar25Solver.main())
