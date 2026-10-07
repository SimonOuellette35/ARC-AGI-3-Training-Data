"""Generate Phase-1 training data for the Badge Placement game.

A thin ``BaseSolver`` subclass (record/replay, schema, WIN-filter CLI and the
optional exploration prefix all live in ``solvers/base_solver.py``). A BFS expert
solves every level for a seed and the trajectory is written as one multi-level
episode JSON; each seed is a full 7-level game, so each WIN seed yields one
complete episode. Different seeds give different (procedurally generated) boards.

Badge Placement action set (only indices 0..5 emit; ACTION7 undo is never used):
    ACTION1 (up)  ACTION2 (down)  ACTION3 (left)  ACTION4 (right)
    ACTION5 pick up badge at cursor / place held badge on matching target

``solve_from`` BFS's over the abstract state (cursor, held badge, placed-mask).
It reads that start state LIVE from the game -- the cursor sprite's cell, the
currently-held badge (``game._holding``), and the placed-mask inferred from which
target slots have already been filled (a slot hides once its badge lands) -- so
the plan is valid from ANY perturbed board, not just the level's initial layout.
The grid connectivity guarantees every badge can still reach its target, so a
detour (even a stranded pick-up) is always re-planned; recovery is REPLAN-mode
(``supports_recovery = True``, ``recovery_mode = "replan"``): after an
exploratory action the base re-invokes ``solve_from`` on the live game, and the
rare unrecoverable burst is rolled back by the burst-undo, not RESET.

Usage (run from the repo root):
    python solvers/generate_badge_placement_training.py --episodes 1000 \
        --out data/training_multi_level/badge_placement
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction                             # noqa: E402
from games.badge_placement.badge_placement import BadgePlacement, CELL  # noqa: E402
from solvers.base_solver import BaseSolver                   # noqa: E402

_MOVE_ACTIONS = [
    GameAction.ACTION1,  # up    (0, -1)
    GameAction.ACTION2,  # down  (0,  1)
    GameAction.ACTION3,  # left  (-1, 0)
    GameAction.ACTION4,  # right ( 1, 0)
]
_DELTAS = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}


class BadgePlacementSolver(BaseSolver):
    game_id = "badge_placement"
    # solve_from reads the LIVE state (cursor cell, held badge, placed-mask) and
    # the connected grid guarantees every badge can still reach its target, so it
    # re-plans from any perturbed board -- recovery is REPLAN-mode: after an
    # exploratory detour the base re-invokes solve_from on the live game, and a
    # rare unrecoverable burst is rolled back by the burst-undo (not RESET).
    supports_recovery = True
    recovery_mode = "replan"

    def make_game(self, seed: int):
        return BadgePlacement(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4, 5]

    def solve_from(self, game, level_idx: int, seed: int):
        """BFS over the abstract state (cursor_x, cursor_y, holding, placed_mask)
        for a simple-action plan (moves + ACTION5) that clears the board, starting
        from the game's LIVE state. Badge k uniquely matches target k (distinct
        colours), so the abstract state fully captures the game. Returns [] if no
        plan exists (fails the episode).

        The start state is read LIVE off ``game`` -- the cursor sprite's cell, the
        held badge (``game._holding``), and the placed-mask inferred from which
        target slots are already filled (a slot's sprite is hidden once its badge
        lands) -- so the plan is valid from a perturbed board, not just the level's
        initial layout. Only the static geometry (walls / original badge+target
        cells) comes from ``_level_data``; positions of un-picked badges and of
        every target never move, and the placed-mask + holding cover the rest."""
        grid, _init_cursor, badge_positions, target_positions, w, h = \
            game._level_data[level_idx]
        n = len(badge_positions)
        full_mask = (1 << n) - 1
        if full_mask == 0:
            return []

        def wall_at(x, y):
            return not (0 <= x < w and 0 <= y < h and grid[y][x])

        badge_at = {pos: k for k, pos in enumerate(badge_positions)}
        target_at = {pos: k for k, pos in enumerate(target_positions)}

        # ── LIVE start state: cursor cell, held badge, and placed-mask read from
        # the game so a perturbed board re-plans correctly. A target slot hides
        # (``is_visible`` False) once a badge lands on it AND STAYS hidden even if
        # that badge is later picked back off -- so a hidden slot means badge k is
        # DONE only when we are not currently holding k. Holding a badge whose slot
        # is already hidden is a permanently STRANDED badge (its only matching slot
        # is gone): the state is unsolvable, so return [] and let the recorder's
        # burst-undo roll the stranding burst back.
        cur = game._cursor()
        cx0, cy0 = cur.x // CELL, cur.y // CELL
        holding = game._holding
        placed = 0
        for s in game.current_level.get_sprites_by_tag("target"):
            if not s.is_visible:
                k = next(int(t[6:]) for t in s.tags if t.startswith("color_"))
                if k == holding:
                    return []                  # stranded badge -> unsolvable here
                placed |= (1 << k)

        init = (cx0, cy0, holding, placed)
        if placed == full_mask and holding is None:
            return []                          # already solved -- nothing to do

        visited = {init}
        q = deque([(init, [])])
        while q:
            (cx, cy, holding, placed), path = q.popleft()

            for action in _MOVE_ACTIONS:
                dx, dy = _DELTAS[action]
                nx, ny = cx + dx, cy + dy
                if wall_at(nx, ny):
                    continue
                state = (nx, ny, holding, placed)
                if state not in visited:
                    visited.add(state)
                    q.append((state, path + [action]))

            if holding is None:
                k = badge_at.get((cx, cy))
                if k is not None and not (placed & (1 << k)):
                    state = (cx, cy, k, placed)
                    if state not in visited:
                        visited.add(state)
                        q.append((state, path + [GameAction.ACTION5]))
            else:
                k = target_at.get((cx, cy))
                if k is not None and k == holding:
                    new_placed = placed | (1 << k)
                    if new_placed == full_mask:
                        return path + [GameAction.ACTION5]
                    state = (cx, cy, None, new_placed)
                    if state not in visited:
                        visited.add(state)
                        q.append((state, path + [GameAction.ACTION5]))

        return []                              # unsolvable -> fail episode


if __name__ == "__main__":
    sys.exit(BadgePlacementSolver.main())
