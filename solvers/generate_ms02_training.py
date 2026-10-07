"""Generate Phase-1 training data for the ms02 game.

ms02 is a minesweeper-themed grid navigation game (games/ms02/ms02.py). Despite
the minesweeper dressing (ACTION6 places/removes flags on mines), the WIN
condition is purely navigational: the player reaches the goal cell, which calls
``next_level()``. Moving onto a mine loses; walls and out-of-bounds moves are
no-ops. Flags are never required to win, so every level is solvable with the
directional actions ACTION1..ACTION4 alone -- no clicks (ACTION6) needed.

Directional action map (from ms02.step):
    ACTION1 up (dy=-1)   ACTION2 down (dy=+1)
    ACTION3 left (dx=-1) ACTION4 right (dx=+1)

The solver is NON-PRIVILEGED: like a real player under partial observability it
knows only the visible walls/goal and the clue of cells it has ALREADY stepped
on -- never mine positions or unvisited-cell clues. It explores the provably-safe
frontier (a visited count-0 cell proves its 8 neighbours are mine-free) while
drifting toward the goal, reacting to each revealed clue, and never places a flag
(ACTION6), so it can never mis-flag a safe cell. It always wins without hitting a
mine (a count-0 path is guaranteed), but the recorded trajectory is realistic
exploration rather than a straight walk down a path it could only know by
cheating.

Because the explorer's knowledge is *accumulated* (which cells it has stepped on
and their clues), ``solve_from`` returns ONE move at a time from the LIVE game:
the base harness re-invokes it after every step, and the solver records the clue
at the player's current cell before choosing the next move -- reproducing the
original online explorer exactly. The per-(seed, level) mine layout is made
DETERMINISTIC by seeding ``game._rng`` with ``ms02:<seed>:<level>`` before each
``set_level`` (in ``set_level`` below), so one episode == one seed.

Recovery is OFF: stepping onto an unproven cell can hit a mine (GAME_OVER), so
random exploratory moves are unsafe -- exploration stays disabled.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_ms02_training.py --episodes 1000 \
        --out data/training_multi_level/ms02
"""

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction                            # noqa: E402
from games.ms02.ms02 import Ms02                            # noqa: E402
from solvers.base_solver import BaseSolver                  # noqa: E402

# Directional simple actions -> (dx, dy), matching ms02.step().
_MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}
_MOVE_TO_ACTION = {v: k for k, v in _MOVES.items()}   # (dx, dy) -> GameAction


def _safe_cells(revealed, walls, grid_w, grid_h):
    """Cells PROVABLY mine-free from what has been observed so far: every visited
    cell (we survived stepping on it) plus every 8-neighbour of a visited COUNT-0
    cell (a count-0 clue means no adjacent mine). Walls excluded. This is the only
    knowledge the non-privileged solver acts on -- it never consults mine
    positions or the clue of an unvisited cell."""
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
    cell toward it -- discover new cells while drifting toward the goal, then react
    to each revealed clue on the next call. Returns a (dx, dy) unit move, or None
    if no safe frontier remains (should not happen: a count-0 path is guaranteed)."""
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


class Ms02Solver(BaseSolver):
    game_id = "ms02"
    # Mines kill: a random exploratory step can end the episode and can't be
    # re-planned around, but RESET-recovery still applies -- run the epsilon
    # prefix, take one RESET back to the initial board, then replay the honest
    # safe-frontier plan (a human who flails, dies, resets, then walks it clean).
    # ``set_level`` re-seeds the mine layout and clears ``self._revealed`` on RESET
    # so the restored board matches ``observations[0]``.
    supports_recovery = True
    recovery_mode = "reset"
    # Lethal: a death mid-prefix costs one RESET; give headroom above the default.
    max_resets = 5

    def make_game(self, seed: int):
        self._seed = seed
        return Ms02()

    def set_level(self, game, level_idx: int) -> None:
        """Seed the mine layout DETERMINISTICALLY per (seed, level_idx) -- exactly
        the original ``_make_level`` -- then reset the explorer's accumulated
        clue memory for the fresh level.

        Must go through ``super().set_level`` (not ``game.set_level``): the base
        also clears the engine's pending ``_next_level`` flag, which `drive`
        deliberately leaves raised after a win. Left set, the FIRST action of
        every subsequent level is mis-read as an instant solve and each level
        after the first collapses to a bogus 2-frame "win"."""
        game._rng = random.Random(f"ms02:{self._seed}:{level_idx}")
        super().set_level(game, level_idx)
        self._revealed = {}

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    def solve_from(self, game, level_idx: int, seed: int):
        """ONE move from the live board. Records the clue at the player's CURRENT
        cell (which we survived stepping on -- exactly the on-screen clue, never an
        unvisited cell or a mine position) into the accumulated memory, then picks
        the next safe-frontier move. The base re-invokes this each step, so the
        accumulate-observe-move loop reproduces the original online explorer."""
        lvl = game.current_level
        grid_w, grid_h = lvl.grid_size
        walls = {(s.x, s.y) for s in lvl.get_sprites_by_tag("wall")}
        goal_sprite = lvl.get_sprites_by_tag("goal")[0]
        goal = (goal_sprite.x, goal_sprite.y)
        pos = (game._player.x, game._player.y)
        if pos not in self._revealed:
            self._revealed[pos] = game._mine_count.get(pos, 0)
        move = _next_move(pos, goal, self._revealed, walls, grid_w, grid_h)
        if move is None:
            return []
        return [_MOVE_TO_ACTION[move]]


if __name__ == "__main__":
    sys.exit(Ms02Solver.main())
