"""Generate Phase-1 training data for the Ms01 (minesweeper-navigation) game.

A thin ``BaseSolver`` subclass over the local Ms01 game (games/ms01/ms01.py).
Ms01 is a minesweeper-style navigation puzzle: a player sprite moves one cell at
a time (up/down/left/right) across a grid, must avoid hidden mines and walls, and
wins a level by reaching the goal cell. There are 5 levels of increasing size
(8x8 .. 16x16).

    ACTION1 up (dy=-1)   ACTION2 down (dy=+1)
    ACTION3 left (dx=-1) ACTION4 right (dx=+1)

The mine positions are RANDOMISED per seed from the game's instance RNG. This
generator makes that augmentation DETERMINISTIC per (seed, level): the overridden
``set_level`` seeds ``game._rng`` with ``ms01:<seed>:<level>`` before
``game.set_level``, which on_set_level -> _randomize_mines consumes.

The solver is NON-PRIVILEGED: it plays like a real player under partial
observability, knowing only the visible walls/goal and the clue of every cell it
has ALREADY stepped on. It bootstraps from the game's guarantee that the start is
count-0 (its neighbours are therefore mine-free), then explores the provably-safe
frontier while drifting toward the goal, reacting to each revealed clue. Because a
revealed count-0 cell proves its 8 neighbours are mine-free and a count-0 path to
the goal is guaranteed, it always wins without ever hitting a mine.

Because the honest solver reveals clues incrementally, ``solve_from`` returns just
the NEXT one-cell move (a single-step plan): the base re-invokes it after every
step, and the accumulated observations live in ``self._revealed`` (reset per level
by ``set_level``). Clues are read from ``game._mine_count`` ONLY for cells already
visited -- exactly what a player sees on screen -- never for unvisited cells.

supports_recovery is False: mines are hidden and stepping on one is instant death
(irreversible), so an exploratory detour onto an unproven cell could brick it.

Usage (run from the repo root):
    python solvers/generate_ms01_training.py --episodes 1000 \
        --out data/training_multi_level/ms01
"""

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

# Repo root (parent of solvers/) -- that's where the games/ package lives.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.ms01.ms01 import Ms01  # noqa: E402
from solvers.base_solver import BaseSolver  # noqa: E402

# (dx, dy) -> action id, matching Ms01.step():
#   1 -> dy=-1 (up), 2 -> dy=+1 (down), 3 -> dx=-1 (left), 4 -> dx=+1 (right)
_MOVE_TO_ACTION = {(0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4}


def _safe_cells(revealed, walls, grid_w, grid_h):
    """Cells PROVABLY mine-free from what has been observed so far: every visited
    cell (we survived stepping on it) plus every 8-neighbour of a visited COUNT-0
    cell (a count-0 clue means no adjacent mine). Walls excluded. This is the only
    knowledge the non-privileged solver acts on -- it never consults mine positions
    or the clue of an unvisited cell."""
    safe = set(revealed)
    for (cx, cy), cnt in revealed.items():
        if cnt == 0:
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    nb = (cx + ddx, cy + ddy)
                    if 0 <= nb[0] < grid_w and 0 <= nb[1] < grid_h and nb not in walls:
                        safe.add(nb)
    return safe


def _next_move(pos, goal, revealed, walls, grid_w, grid_h):
    """Pick the next one-cell move for the non-privileged explorer.

    BFS over the 4-connected graph of currently-known-safe cells. If the goal is
    already reachable through safe cells, head straight for it; otherwise pick the
    nearest-to-goal *unvisited* safe cell (the exploration frontier) and step one
    cell toward it. Returns a (dx, dy) unit move, or None if no safe frontier
    remains (should not happen: a count-0 path is guaranteed)."""
    safe = _safe_cells(revealed, walls, grid_w, grid_h)
    prev = {pos: None}
    dist = {pos: 0}
    q = deque([pos])
    while q:
        c = q.popleft()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nb = (c[0] + dx, c[1] + dy)
            if nb in prev or nb not in safe:
                continue
            prev[nb] = (c, (dx, dy))
            dist[nb] = dist[c] + 1
            q.append(nb)

    if goal in prev:
        target = goal
    else:
        frontier = [c for c in prev if c not in revealed]
        if not frontier:
            return None
        target = min(frontier,
                     key=lambda c: (abs(c[0] - goal[0]) + abs(c[1] - goal[1]),
                                    dist[c], c))

    node, move = target, None
    while prev[node] is not None:            # walk back to `pos`; keep first move
        node, move = prev[node][0], prev[node][1]
    return move


class Ms01Solver(BaseSolver):
    game_id = "ms01"
    # Hidden mines make a step onto an unproven cell instant, irreversible death,
    # so exploration cannot re-plan from the perturbed state -- but RESET-recovery
    # models a human who flails, dies, resets, and only THEN walks the safe path:
    # run the epsilon prefix, take one RESET back to the initial board, then replay
    # the honest safe-frontier plan. ``reset_level`` re-seeds the mine layout and
    # clears ``self._revealed``, so the restored board matches ``observations[0]``.
    supports_recovery = True
    recovery_mode = "reset"
    # Lethal: a death mid-prefix costs one RESET; give headroom above the default.
    max_resets = 5

    def make_game(self, seed: int):
        self._seed = seed
        return Ms01()

    def set_level(self, game, level_idx: int) -> None:
        """Make level `level_idx` DETERMINISTIC in (seed, level): seed the
        instance RNG (mine positions) before set_level, then cache the visible
        board and reset the accumulated observations for the honest explorer."""
        game._rng = random.Random(f"ms01:{self._seed}:{level_idx}")
        game.set_level(level_idx)
        # One reused instance spans all levels; the native drive breaks on the
        # engine's ``_next_level`` flag without clearing it, so clear it here or
        # the next level is spuriously reported solved in a single action.
        game._next_level = False
        lvl = game.current_level
        self._walls = {(w.x, w.y) for w in lvl.get_sprites_by_tag("wall")}
        gs = lvl.get_sprites_by_tag("goal")[0]
        self._goal = (gs.x, gs.y)
        self._gw, self._gh = lvl.grid_size
        self._revealed = {}                  # observed lazily, current cell first

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    def solve_from(self, game, level_idx: int, seed: int):
        """Return the NEXT single move for the non-privileged explorer, reading
        only the clue of the CURRENT (just-survived) cell plus prior observations.
        Empty -> no safe frontier -> fail the episode (should not happen)."""
        pos = (game._player.x, game._player.y)
        # We survived stepping onto `pos`, so its clue is now visible.
        if pos not in self._revealed:
            self._revealed[pos] = game._mine_count.get(pos, 0)
        move = _next_move(pos, self._goal, self._revealed,
                          self._walls, self._gw, self._gh)
        if move is None:
            return []
        return [_MOVE_TO_ACTION[move]]


if __name__ == "__main__":
    sys.exit(Ms01Solver.main())
