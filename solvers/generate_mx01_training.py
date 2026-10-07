"""Generate Phase-1 training data for the mx01 (maze-melt) game.

A thin ``BaseSolver`` subclass over the local mx01 game (games/mx01/mx01.py).
mx01 is a 10x10 grid maze. The player reaches a goal cell using four move
actions and a wall-melt action:

    ACTION1 up (dy-1)   ACTION2 down (dy+1)
    ACTION3 left (dx-1) ACTION4 right (dx+1)
    ACTION5 melt: removes the FIRST wall adjacent to the player, scanning
            neighbours in the fixed order up, down, left, right, limited by a
            per-level melt budget.

Walls (collidable) block movement; the goal (non-collidable) is reached by
stepping onto it. The board is RANDOMISED per seed from the game's instance RNG.
This generator makes that augmentation DETERMINISTIC per (seed, level): the
overridden ``set_level`` seeds ``game._rng`` with ``mx01:<seed>:<level>`` before
``game.set_level``, which on_set_level -> _randomize_level consumes, so one
episode == one seed, reproducible byte-for-byte.

``solve_from`` is a direct BFS over the true state -- (player x, player y,
frozenset of remaining wall cells) -- read from the LIVE game, giving a shortest
simple-action plan. The melt budget left is derivable from how many walls have
been removed, so it need not be part of the key.

supports_recovery is False: melting a wall is IRREVERSIBLE (the wall is gone and
the budget is spent), so an exploratory detour could brick the level.

Usage (run from the repo root):
    python solvers/generate_mx01_training.py --episodes 1000 \
        --out data/training_multi_level/mx01
"""

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

# Repo root (parent of solvers/) so games/ + arcengine resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction  # noqa: E402
from games.mx01.mx01 import Mx01  # noqa: E402
from solvers.base_solver import BaseSolver  # noqa: E402

# Move deltas keyed by GameAction, matching mx01.step().
_MOVES = {
    GameAction.ACTION1: (0, -1),   # up
    GameAction.ACTION2: (0, 1),    # down
    GameAction.ACTION3: (-1, 0),   # left
    GameAction.ACTION4: (1, 0),    # right
}
# Order in which ACTION5 scans neighbours for a wall to melt.
_MELT_ORDER = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _level_state(game):
    """Read the current state of the current level from the engine."""
    player = game.current_level.get_sprites_by_tag("player")[0]
    goal = game.current_level.get_sprites_by_tag("goal")[0]
    walls = frozenset(
        (s.x, s.y) for s in game.current_level.get_sprites_by_tag("wall")
    )
    gw, gh = game.current_level.grid_size
    budget = int(game.current_level.get_data("melt_budget") or 3)
    return (player.x, player.y), (goal.x, goal.y), walls, (gw, gh), budget


class Mx01Solver(BaseSolver):
    game_id = "mx01"
    # Melting a wall is irreversible (one-way), so a detour can spend the budget
    # and strand a live re-plan. RESET-recovery fits: run the epsilon prefix, take
    # solve_from does a live BFS over (px, py, frozenset(remaining walls)) via
    # _level_state(game), so it re-plans from any perturbed-but-solvable state ->
    # replan mode (burst-undo rollback handles the irreversible wall-melt).
    supports_recovery = True
    recovery_mode = "replan"

    def make_game(self, seed: int):
        self._seed = seed
        return Mx01()

    def set_level(self, game, level_idx: int) -> None:
        """Make level `level_idx` DETERMINISTIC in (seed, level): seed the
        instance RNG (spawns / hole / translation / colours) before set_level,
        which on_set_level -> _randomize_level consumes."""
        game._rng = random.Random(f"mx01:{self._seed}:{level_idx}")
        game.set_level(level_idx)
        # One reused instance spans all levels; the native drive breaks on the
        # engine's ``_next_level`` flag without clearing it, so clear it here or
        # the next level is spuriously reported solved in a single action.
        game._next_level = False

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4, 5]

    def solve_from(self, game, level_idx: int, seed: int):
        """BFS over (px, py, remaining walls) from the LIVE state for a shortest
        simple-action plan that reaches the goal. Empty -> fail the episode."""
        start, goal, walls0, (gw, gh), budget = _level_state(game)
        n0 = len(walls0)

        start_state = (start[0], start[1], walls0)
        if (start[0], start[1]) == goal:
            return []

        visited = {start_state}
        q = deque([(start_state, [])])   # (state, path-of-GameActions)
        while q:
            (px, py, walls), path = q.popleft()
            melts_left = budget - (n0 - len(walls))

            # Move actions.
            for act, (dx, dy) in _MOVES.items():
                nx, ny = px + dx, py + dy
                if not (0 <= nx < gw and 0 <= ny < gh):
                    continue
                if (nx, ny) in walls:
                    continue  # blocked -> no-op, skip
                nstate = (nx, ny, walls)
                if nstate in visited:
                    continue
                npath = path + [act]
                if (nx, ny) == goal:
                    return npath
                visited.add(nstate)
                q.append((nstate, npath))

            # Melt action: remove first adjacent wall in scan order.
            if melts_left > 0:
                for dx, dy in _MELT_ORDER:
                    cell = (px + dx, py + dy)
                    if cell in walls:
                        nwalls = walls - {cell}
                        nstate = (px, py, nwalls)
                        if nstate not in visited:
                            visited.add(nstate)
                            q.append((nstate, path + [GameAction.ACTION5]))
                        break  # only the first adjacent wall is meltable
        return []

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-optimal next action at the LIVE state: each move/melt whose
        successor lies on a shortest solution (``dist == d0 - 1``).

        Distance is measured over the SAME state ``(px, py, remaining-walls)`` the
        BFS uses, so an action is returned only if it keeps a solution of the same
        minimal length -- melting a wall (irreversible, spends the budget) is
        included only when it is provably co-optimal for reaching the goal, never
        just because it removes a nearby wall. A BFS per candidate first move (no
        game copy); the transition mirrors ``solve_from`` exactly, so the canonical
        head is always in the set."""
        start, goal, walls0, (gw, gh), budget = _level_state(game)
        n0 = len(walls0)
        if (start[0], start[1]) == goal:
            return []

        def neighbors(state):
            px, py, walls = state
            melts_left = budget - (n0 - len(walls))
            for act, (dx, dy) in _MOVES.items():
                nx, ny = px + dx, py + dy
                if not (0 <= nx < gw and 0 <= ny < gh):
                    continue
                if (nx, ny) in walls:
                    continue                       # blocked -> no-op, skip
                yield act, (nx, ny, walls)
            if melts_left > 0:
                for dx, dy in _MELT_ORDER:
                    cell = (px + dx, py + dy)
                    if cell in walls:
                        yield GameAction.ACTION5, (px, py, walls - {cell})
                        break                      # only the first adjacent wall melts

        def dist_to_goal(s0):
            q = deque([s0])
            dist = {s0: 0}
            while q:
                st = q.popleft()
                if (st[0], st[1]) == goal:
                    return dist[st]
                for _act, ns in neighbors(st):
                    if ns in dist:
                        continue
                    dist[ns] = dist[st] + 1
                    q.append(ns)
            return None

        s0 = (start[0], start[1], walls0)
        d0 = dist_to_goal(s0)
        if not d0:                                 # already solved / unreachable
            return []
        opt: list = []
        for act, ns in neighbors(s0):
            d = dist_to_goal(ns)
            if d is not None and d == d0 - 1:
                opt.append(act)
        return opt


if __name__ == "__main__":
    sys.exit(Mx01Solver.main())
