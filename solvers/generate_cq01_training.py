"""Generate Phase-1 training data for the Conquest-ring game `cq01`.

A thin ``BaseSolver`` subclass (the record/replay loop, episode schema, WIN-filter
CLI and exploration prefix all live in ``solvers/base_solver.py``). `cq01` is a
tiny grid where the player (colour 9) must step onto every orange ring marker
(colour 12) and then reach the green goal (colour 14).

Action set (simple, index in 0..5):
    ACTION1 (1) up      dy=-1
    ACTION2 (2) down    dy=+1
    ACTION3 (3) left    dx=-1
    ACTION4 (4) right   dx=+1

Per-(seed, level) augmentation: the board LAYOUT is fixed but its COLOURS
(background, agent, ring markers + the matching legend) are randomised from the
game's instance RNG in ``on_set_level``. This generator makes that deterministic
per (seed, level) like the other seeded games: ``set_level`` reseeds
``game._rng`` with ``cq01:<seed>:<level>`` right before ``game.set_level``, so one
episode == one seed, reproducible byte-for-byte, one episode per seed.

``solve_from`` reads the LIVE board (player position + the rings still to visit,
``_rings - _visited``) and BFS's an optimal visit-all-then-goal plan, so it
recovers from any perturbed state -- the grid is fully reversible with no death,
so ``supports_recovery = True`` (the exploration prefix + bursts are safe).

Usage (run from the repo root):
    python solvers/generate_cq01_training.py --episodes 1000 \
        --out data/training_multi_level/cq01
"""

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction                             # noqa: E402
from games.cq01.cq01 import Cq01                             # noqa: E402
from solvers.base_solver import BaseSolver                   # noqa: E402

# Simple-action move deltas, matching Cq01.step():
#   1 up (dy=-1), 2 down (dy=+1), 3 left (dx=-1), 4 right (dx=+1)
_MOVES = {
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
}
_INDEX_TO_ACTION = {
    1: GameAction.ACTION1,
    2: GameAction.ACTION2,
    3: GameAction.ACTION3,
    4: GameAction.ACTION4,
}


class Cq01Solver(BaseSolver):
    game_id = "cq01"
    # Reversible grid nav (no death); solve_from re-plans from the live player
    # position and the remaining rings, so exploratory detours always recover.
    supports_recovery = True

    def make_game(self, seed: int):
        game = Cq01()
        game._gen_seed = seed          # stashed for the per-level RNG reseed
        return game

    def set_level(self, game, level_idx: int) -> None:
        """Deterministic per (seed, level): reseed the colour RNG that
        ``on_set_level -> _randomize_level`` consumes, then set the level.

        Must go through ``super().set_level`` (not ``game.set_level``): the base
        also clears the engine's pending ``_next_level`` flag, which `drive`
        deliberately leaves raised after a win. Left set, the FIRST action of
        every subsequent level is mis-read as an instant solve and each of levels
        2..5 collapses to a bogus 2-frame "win"."""
        game._rng = random.Random(f"cq01:{game._gen_seed}:{level_idx}")
        super().set_level(game, level_idx)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]            # moves only; ACTION5 is unused

    def solve_from(self, game, level_idx: int, seed: int):
        """Exact BFS from the LIVE state: shortest plan that visits every ring
        still outstanding (``_rings - _visited``) and then lands on the goal.

        Engine adds a ring to ``_visited`` only when the player MOVES onto it, so
        the current cell counts as unvisited even if it coincides with a ring
        (matches Cq01.step())."""
        start = (game._player.x, game._player.y)
        goal = (game._goal.x, game._goal.y)
        rings = frozenset(game._rings)
        visited0 = frozenset(game._visited)
        walls = {(s.x, s.y)
                 for s in game.current_level.get_sprites_by_tag("wall")}
        gw, gh = game.current_level.grid_size

        start_state = (start[0], start[1], visited0)
        q = deque([start_state])
        prev = {start_state: (None, None)}    # state -> (parent, action_index)

        goal_state = None
        while q:
            x, y, visited = q.popleft()
            if visited == rings and (x, y) == goal:
                goal_state = (x, y, visited)
                break
            for idx, (dx, dy) in _MOVES.items():
                nx, ny = x + dx, y + dy
                if not (0 <= nx < gw and 0 <= ny < gh):
                    continue
                if (nx, ny) in walls:
                    continue
                nvis = visited
                if (nx, ny) in rings:
                    nvis = visited | {(nx, ny)}
                nstate = (nx, ny, nvis)
                if nstate not in prev:
                    prev[nstate] = ((x, y, visited), idx)
                    q.append(nstate)

        if goal_state is None:
            return []                          # unreachable -> fail episode

        indices: list[int] = []
        cur = goal_state
        while prev[cur][0] is not None:
            parent, idx = prev[cur]
            indices.append(idx)
            cur = parent
        indices.reverse()
        return [_INDEX_TO_ACTION[i] for i in indices]

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-optimal next move at the LIVE state: each direction whose
        successor is exactly one step closer to solving the whole visit-all-then-
        goal task. Feeds the base's STOCHASTIC OPTIMAL sampling so each episode
        takes a different shortest tour.

        The task has a monotone ``visited`` set, so the goal is not a single cell
        and a plain grid distance field is wrong -- distance is measured over the
        FULL abstract state ``(x, y, visited)``. Cheap: a BFS over that abstract
        state (no game copy) per candidate first move. The transition mirrors
        ``solve_from`` exactly (a ring is marked only when the player MOVES onto
        it), so the canonical head is always in the returned set."""
        start = (game._player.x, game._player.y)
        goal = (game._goal.x, game._goal.y)
        rings = frozenset(game._rings)
        visited0 = frozenset(game._visited)
        walls = {(s.x, s.y)
                 for s in game.current_level.get_sprites_by_tag("wall")}
        gw, gh = game.current_level.grid_size

        def succ(state, dx, dy):
            x, y, vis = state
            nx, ny = x + dx, y + dy
            if not (0 <= nx < gw and 0 <= ny < gh) or (nx, ny) in walls:
                return None
            nvis = vis | {(nx, ny)} if (nx, ny) in rings else vis
            return (nx, ny, nvis)

        def dist_to_goal(s0):
            q = deque([s0])
            dist = {s0: 0}
            while q:
                st = q.popleft()
                x, y, vis = st
                if vis == rings and (x, y) == goal:
                    return dist[st]
                for _idx, (dx, dy) in _MOVES.items():
                    ns = succ(st, dx, dy)
                    if ns is None or ns in dist:
                        continue
                    dist[ns] = dist[st] + 1
                    q.append(ns)
            return None

        s0 = (start[0], start[1], visited0)
        d0 = dist_to_goal(s0)
        if not d0:                                 # already solved / unreachable
            return []
        opt: list = []
        for idx, (dx, dy) in _MOVES.items():
            ns = succ(s0, dx, dy)
            if ns is None:
                continue
            d = dist_to_goal(ns)
            if d is not None and d == d0 - 1:
                opt.append(_INDEX_TO_ACTION[idx])
        return opt


if __name__ == "__main__":
    sys.exit(Cq01Solver.main())
