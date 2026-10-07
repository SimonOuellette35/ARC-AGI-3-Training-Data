"""Generate Phase-1 training data for the Balloon Festival game.

A thin ``BaseSolver`` subclass (record/replay, schema, WIN-filter CLI and the
optional exploration prefix all live in ``solvers/base_solver.py``). A fresh
expert solver (embedded below) solves every level for a seed; each seed is a full multi-level game, so each WIN seed yields
one complete episode. Only simple actions (index 0..5) emit; ACTION7 (undo) never.

Balloon Festival mechanics
--------------------------
Balloons float up one cell/tick. An active deflector at (x, y) makes a balloon
at (x, y+1) move sideways (left if type 0, right if type 1) that tick instead of
up. Cursor moves (ACTION1..4) are free (no tick). ACTION5 activates the inactive
deflector marker under the cursor (if any) AND always advances the simulation by
one tick. Only the level's ``sol_deflectors`` are correct markers -- noise/trap/
decoy markers must be left inactive.

Expert solver
-------------
Because each activation costs a tick and balloons keep rising, timing matters. We
simulate the intended fully-solved deflector map (pre_placed + sol) to learn, for
each solution deflector, the DEADLINE tick by which it must be active (the tick
its balloon is deflected). Activations are ordered to MINIMISE cursor travel
while meeting every deadline (closest schedulable deflector each tick, else
earliest-deadline). Once all are active we COAST in place -- the cursor sits on
the last activated (now-active) deflector, a safe ACTION5 no-op-plus-tick -- until
every balloon is caught. Timing is irreversible (a missed balloon can't be
re-caught), so an off-plan detour can't be re-planned from; recovery is
RESET-mode (``supports_recovery = True``, ``recovery_mode = "reset"``): explore
the prefix, then a single RESET to the level's initial state, then replay the
timing-optimal plan from there.

Usage (run from the repo root):
    python solvers/generate_balloon_festival_training.py --episodes 1000 \
        --out data/training_multi_level/balloon_festival
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction                             # noqa: E402
from games.balloon_festival.balloon_festival import BalloonFestival  # noqa: E402
from solvers.base_solver import BaseSolver                   # noqa: E402

_MOVE_ACTIONS = {
    (0, -1): GameAction.ACTION1,   # up
    (0,  1): GameAction.ACTION2,   # down
    (-1, 0): GameAction.ACTION3,   # left
    (1,  0): GameAction.ACTION4,   # right
}


def _bfs_path(start, goal, grid_w, grid_h):
    """Return list of (dx, dy) unit deltas to move the cursor start -> goal."""
    if start == goal:
        return []
    q = deque([(start, [])])
    seen = {start}
    while q:
        (cx, cy), path = q.popleft()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in seen:
                continue
            if not (0 <= nx < grid_w and 0 <= ny < grid_h):
                continue
            np_ = path + [(dx, dy)]
            if (nx, ny) == goal:
                return np_
            seen.add((nx, ny))
            q.append(((nx, ny), np_))
    return []


def _order_activations(deadlines, cursor_start, grid_w, grid_h):
    """Order the solution-deflector activations to minimise cursor travel while
    meeting every deadline. Each activation consumes one tick (1..K). A set is
    schedulable from tick ``t0`` iff, sorted by deadline, the i-th deadline is
    >= t0 + i. We greedily pick the CLOSEST remaining deflector whose removal
    still leaves the rest schedulable, falling back to earliest-deadline where
    deadlines are tight. Returns the ordered cell list, or None if infeasible."""
    def schedulable(subset, t0):
        dls = sorted(deadlines[c] for c in subset)
        return all(dl >= t0 + i for i, dl in enumerate(dls))

    remaining = set(deadlines)
    if not schedulable(remaining, 1):
        return None
    order = []
    cursor = cursor_start
    t = 1
    while remaining:
        best = None
        for cell in remaining:
            if deadlines[cell] < t:
                continue
            if schedulable(remaining - {cell}, t + 1):
                dist = abs(cell[0] - cursor[0]) + abs(cell[1] - cursor[1])
                if best is None or dist < best[0]:
                    best = (dist, cell)
        if best is None:
            return None
        cell = best[1]
        order.append(cell)
        remaining.discard(cell)
        cursor = cell
        t += 1
    return order


def _simulate(d: dict):
    """Simulate the intended solve (pre_placed + all sol deflectors active).

    Returns (deadlines, t_win, won): deadlines {(x,y): tick} for each sol
    deflector that actually deflects a balloon, t_win the tick every balloon is
    caught (or the guard cap), won True iff all balloons were caught."""
    grid_w, grid_h = d["grid_w"], d["grid_h"]
    sol = d["sol_deflectors"]
    defl_map = dict(d["pre_placed"])
    defl_map.update(sol)

    pos = list(d["balloon_starts"])
    caught = [False] * d["num_balloons"]
    deadlines: dict[tuple[int, int], int] = {}

    max_ticks = grid_h * 10
    for tick in range(1, max_ticks + 1):
        for i, (bx, by) in enumerate(pos):
            if caught[i] or by == 0:
                continue
            above_y = by - 1
            defl = defl_map.get((bx, above_y))
            if defl is not None:
                if (bx, above_y) in sol and (bx, above_y) not in deadlines:
                    deadlines[(bx, above_y)] = tick
                nx, ny = bx + (-1 if defl == 0 else 1), by
                if not (0 <= nx < grid_w):
                    continue  # clamp at side boundary (matches engine)
            else:
                nx, ny = bx, by - 1
            pos[i] = (nx, ny)
            if nx == d["basket_cols"][i] and ny == d["basket_rows"][i]:
                caught[i] = True
        if all(caught):
            return deadlines, tick, True
    return deadlines, max_ticks, all(caught)


class BalloonFestivalSolver(BaseSolver):
    game_id = "balloon_festival"
    # Timing is irreversible: a balloon missed on a tick can't be re-caught, so
    # an exploratory detour cannot be re-planned from -- recovery is RESET-mode:
    # explore the prefix, then RESET to the level's initial state and replay the
    # timing-optimal plan from there.
    supports_recovery = True
    recovery_mode = "reset"

    def make_game(self, seed: int):
        return BalloonFestival(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4, 5]

    def solve_from(self, game, level_idx: int, seed: int):
        """Return a simple-action plan solving the level, or [] if infeasible."""
        d = game._level_data[level_idx]
        grid_w, grid_h = d["grid_w"], d["grid_h"]

        deadlines, t_win, won = _simulate(d)
        if not won:
            return []

        cursor = d["cursor_start"]
        activate_cells = _order_activations(deadlines, cursor, grid_w, grid_h)
        if activate_cells is None:
            return []                          # deadlines infeasible
        k = len(activate_cells)

        actions: list[GameAction] = []

        # Phase 1: activate each solution deflector on consecutive ticks 1..K.
        for cell in activate_cells:
            for dx, dy in _bfs_path(cursor, cell, grid_w, grid_h):
                actions.append(_MOVE_ACTIONS[(dx, dy)])
            cursor = cell
            actions.append(GameAction.ACTION5)  # activates + advances one tick

        # Phase 2: coast in place. The cursor sits on the last activated (now
        # ACTIVE) deflector, so ACTION5 there activates nothing and only advances
        # a tick. Only when nothing was activated (k == 0) is the cursor not
        # guaranteed safe, so park it on column 0 (never an inactive marker).
        coast = t_win - k
        if coast > 0:
            if k == 0:
                safe = (0, 0)
                for dx, dy in _bfs_path(cursor, safe, grid_w, grid_h):
                    actions.append(_MOVE_ACTIONS[(dx, dy)])
                cursor = safe
            # +10 margin against off-by-one; record_level stops the instant we win.
            for _ in range(coast + 10):
                actions.append(GameAction.ACTION5)

        return actions


if __name__ == "__main__":
    sys.exit(BalloonFestivalSolver.main())
