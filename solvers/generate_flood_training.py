"""Generate Phase-1 training data for the Flood (Color Flood) game.

A greedy expert solves every level for a seed and the trajectory is written as
one multi-level episode JSON. One Flood *seed* is already a 7-level game, so each
WIN seed yields one complete episode.

All of the harness -- record/replay, the episode schema, the WIN-filter CLI, and
the optional exploration prefix -- lives in ``BaseSolver``; this file only wires
in the game and its expert.

NOTE: the expert lives here. An older duplicate under ``games/flood/solver.py``
had drifted (a stale ``_flood()`` call signature) and has been deleted along with
the rest of the obsolete per-game solvers.

Flood action set (colour cycling + apply):
    ACTION1/ACTION3 prev colour   ACTION2/ACTION4 next colour
    ACTION5 apply flood fill      ACTION7 undo

Usage (run from the repo root):
    python solvers/generate_flood_training.py --episodes 1000 \
        --out data/training_multi_level/flood
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                         # noqa: E402
from arcengine import GameAction                           # noqa: E402
from games.flood.flood import Flood, _flood                # noqa: E402
from solvers.base_solver import BaseSolver                 # noqa: E402


def _owned_size(grid: np.ndarray, origin: tuple[int, int]) -> int:
    """Count cells reachable from ``origin`` sharing the origin's colour."""
    h, w = grid.shape
    ox, oy = origin
    color = int(grid[oy, ox])
    seen: set = set()
    q = deque([(ox, oy)])
    while q:
        x, y = q.popleft()
        if (x, y) in seen:
            continue
        seen.add((x, y))
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in seen \
                    and int(grid[ny, nx]) == color:
                q.append((nx, ny))
    return len(seen)


class FloodSolver(BaseSolver):
    game_id = "flood"
    # Flood is solvable from ANY reachable grid (you can always flood to a single
    # colour), and solve_from below reads the LIVE grid, so exploratory detours
    # are always recoverable -> the epsilon prefix + bursts are safe.
    supports_recovery = True

    def make_game(self, seed: int):
        return Flood(seed=seed)

    def available_actions(self, game) -> list[int]:
        # colour cycle (1-4) + apply flood (5); no click, undo (7) excluded.
        return [1, 2, 3, 4, 5]

    def solve_from(self, game, level_idx: int, seed: int):
        """Greedy expert planning from ``game``'s CURRENT grid: repeatedly flood
        the origin with the colour maximising the owned region, cursor-cycling
        there first. Reads the live game so it re-plans correctly after an
        exploratory detour has changed the board."""
        grid = game._current_grid.copy()
        origin = game._origin
        n = game._num_colors
        ox, oy = origin

        actions: list = []
        selected = game._selected_color
        for _ in range(400):
            if np.all(grid == grid[oy, ox]):
                return actions
            best_size, best_ci, best_grid = -1, None, None
            for ci in range(n):
                if ci == int(grid[oy, ox]):
                    continue
                cand = grid.copy()
                _flood(cand, ci, origin)
                size = _owned_size(cand, origin)
                if size > best_size:
                    best_size, best_ci, best_grid = size, ci, cand
            if best_ci is None:
                break
            right = (best_ci - selected) % n
            left = (selected - best_ci) % n
            actions += ([GameAction.ACTION4] * right if right <= left
                        else [GameAction.ACTION3] * left)
            selected = best_ci
            actions.append(GameAction.ACTION5)
            grid = best_grid
        return actions

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """The equally-optimal next actions at the LIVE state.

        The greedy expert (``solve_from``) is not provably globally optimal, so we
        do NOT treat two colours with equal region growth as co-optimal (a
        different colour can lengthen the greedy continuation -- unprovable, so
        excluded). The ONE safe tie is the cursor DIRECTION: to reach the chosen
        target colour the cursor can cycle ``right`` (ACTION4/next) or ``left``
        (ACTION3/prev); when those distances are EQUAL both directions take the
        same number of presses, land on the same colour, and leave the grid
        unchanged, so continuing either way gives an identical-length solution.
        In every other case the head is unique (the shorter cycle, or the apply).
        O(colours) region probe on copies of the grid -- no game copy."""
        grid = game._current_grid
        origin = game._origin
        n = game._num_colors
        ox, oy = origin
        cur_color = int(grid[oy, ox])
        if np.all(grid == cur_color):
            return []                              # solved
        best_size, best_ci = -1, None
        for ci in range(n):
            if ci == cur_color:
                continue
            cand = grid.copy()
            _flood(cand, ci, origin)
            size = _owned_size(cand, origin)
            if size > best_size:                   # strict > : mirrors solve_from
                best_size, best_ci = size, ci
        if best_ci is None:
            return []
        selected = game._selected_color
        right = (best_ci - selected) % n           # ACTION4 (next) steps
        left = (selected - best_ci) % n            # ACTION3 (prev) steps
        if right == 0:                             # already selected -> apply
            return [GameAction.ACTION5]
        if right < left:
            return [GameAction.ACTION4]
        if left < right:
            return [GameAction.ACTION3]
        return [GameAction.ACTION4, GameAction.ACTION3]   # equidistant: either way


if __name__ == "__main__":
    sys.exit(FloodSolver.main())
