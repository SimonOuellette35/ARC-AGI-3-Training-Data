"""Generate Phase-1 training data for the Number-Fuse (nu01) game.

A thin ``BaseSolver`` subclass over the local nu01 game (games/nu01/nu01.py).
nu01 is a small grid game: the player moves one cell per simple action and must
pick up the numbered tokens in *strictly descending* order (3, then 2, then 1);
stepping on the wrong token loses. Once all three tokens are collected the player
must reach the goal cell to clear the level.

    ACTION1 up (dy=-1)   ACTION2 down (dy=+1)
    ACTION3 left (dx=-1) ACTION4 right (dx=+1)

Positions are RANDOMISED per seed from the game's instance RNG. This generator
makes that augmentation DETERMINISTIC per (seed, level): the overridden
``set_level`` seeds ``game._rng`` with ``nu01:<seed>:<level>`` (positions, per
level) and ``game._color_rng`` with ``nu01-colours:<seed>`` (colour scheme, per
EPISODE, so it is identical across every level of one seed), then clears the
cached scheme so set_level recomputes it.

``solve_from`` runs a BFS over (player_x, player_y, next_needed_value) read from
the LIVE game. Because tokens must be collected 3->2->1, next_needed_value
uniquely determines which tokens remain, so the state is sufficient.

supports_recovery is False: collecting a token (and stepping on a wrong-order
token = loss) is IRREVERSIBLE, so an exploratory detour could brick the level.

Usage (run from the repo root):
    python solvers/generate_nu01_training.py --episodes 1000 \
        --out data/training_multi_level/nu01
"""

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

# Repo root (parent of solvers/) -- that's where the games/ package lives.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.nu01.nu01 import Nu01, _token_value  # noqa: E402
from solvers.base_solver import BaseSolver  # noqa: E402

# (action id, dx, dy) matching nu01.step().
_MOVES = [
    (1, 0, -1),  # up
    (2, 0, 1),   # down
    (3, -1, 0),  # left
    (4, 1, 0),   # right
]


class Nu01Solver(BaseSolver):
    game_id = "nu01"
    # Collecting a token (or mis-stepping onto a wrong-order one = loss) is
    # irreversible (one-way), so a detour can't be re-planned around. RESET-recovery
    # fits: run the epsilon prefix, take one RESET back to the initial board
    # (``set_level`` re-seeds positions/colours, matching ``observations[0]``), then
    # replay the 3->2->1->goal BFS plan from the clean state.
    supports_recovery = True
    # solve_from does a live BFS over (player, next_val) read straight off the game,
    # so it re-plans from any perturbed-but-solvable state -> replan mode (bursts +
    # burst-undo rollback handle the irreversible wrong-order-token death).
    recovery_mode = "replan"
    # A wrong-order-token death mid-prefix costs one RESET; give a little headroom.
    max_resets = 5

    def make_game(self, seed: int):
        self._seed = seed
        return Nu01()

    def set_level(self, game, level_idx: int) -> None:
        """Make level `level_idx` DETERMINISTIC in (seed, level): positions from
        ``nu01:<seed>:<level>`` (per level); the token/agent COLOUR scheme from
        ``nu01-colours:<seed>`` (per EPISODE, level-independent). The cached
        scheme is cleared so set_level recomputes it from the seeded colour RNG."""
        game._rng = random.Random(f"nu01:{self._seed}:{level_idx}")
        game._color_rng = random.Random(f"nu01-colours:{self._seed}")
        game._color_scheme = None
        game.set_level(level_idx)
        # One reused instance spans all levels; the native drive breaks on the
        # engine's ``_next_level`` flag without clearing it, so clear it here or
        # the next level is spuriously reported solved in a single action.
        game._next_level = False

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    def solve_from(self, game, level_idx: int, seed: int):
        """BFS over (player_x, player_y, next_val) using the game's LIVE state.
        Returns a shortest simple-action plan collecting 3->2->1 then landing on
        the goal; empty -> fail the episode."""
        gw, gh = game.current_level.grid_size
        walls = {(s.x, s.y) for s in game.current_level.get_sprites_by_tag("wall")}
        tokens = {(s.x, s.y): _token_value(s) for s in game._tokens}
        goal = (game._goal.x, game._goal.y)

        start = (game._player.x, game._player.y, game._next_val)
        prev: dict[tuple, tuple | None] = {start: None}
        q = deque([start])
        goal_state = None

        while q:
            px, py, nv = state = q.popleft()
            if nv == 0 and (px, py) == goal:
                goal_state = state
                break
            for act, dx, dy in _MOVES:
                nx, ny = px + dx, py + dy
                if not (0 <= nx < gw and 0 <= ny < gh):
                    continue                      # out of bounds: engine no-ops
                if (nx, ny) in walls:
                    continue                      # wall: engine no-ops
                nnv = nv
                if (nx, ny) in tokens:
                    v = tokens[(nx, ny)]
                    if v != nv:
                        continue                  # wrong-order token: would lose
                    nnv = nv - 1
                ns = (nx, ny, nnv)
                if ns not in prev:
                    prev[ns] = (state, act)
                    q.append(ns)

        if goal_state is None:
            return []

        plan: list[int] = []
        s = goal_state
        while prev[s] is not None:
            ps, act = prev[s]
            plan.append(act)
            s = ps
        plan.reverse()
        return plan

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-optimal next move at the LIVE state: each direction whose
        successor lies on a shortest 3->2->1->goal solution (``dist == d0 - 1``).

        Distance is measured over the SAME ordered state ``(x, y, next_val)`` the
        BFS uses, so the one-way constraint (stepping onto a wrong-order token is a
        loss and is pruned) is respected for the FULL remaining objective -- an
        action is returned only if it keeps a solution of the same minimal length.
        A BFS per candidate first move (no game copy); the transition mirrors
        ``solve_from`` exactly, so the canonical head is always in the set."""
        gw, gh = game.current_level.grid_size
        walls = {(s.x, s.y) for s in game.current_level.get_sprites_by_tag("wall")}
        tokens = {(s.x, s.y): _token_value(s) for s in game._tokens}
        goal = (game._goal.x, game._goal.y)

        def succ(state, dx, dy):
            px, py, nv = state
            nx, ny = px + dx, py + dy
            if not (0 <= nx < gw and 0 <= ny < gh):
                return None                       # out of bounds: engine no-ops
            if (nx, ny) in walls:
                return None                       # wall: engine no-ops
            nnv = nv
            if (nx, ny) in tokens:
                v = tokens[(nx, ny)]
                if v != nv:
                    return None                   # wrong-order token: would lose
                nnv = nv - 1
            return (nx, ny, nnv)

        def dist_to_goal(s0):
            q = deque([s0])
            dist = {s0: 0}
            while q:
                st = q.popleft()
                px, py, nv = st
                if nv == 0 and (px, py) == goal:
                    return dist[st]
                for _act, dx, dy in _MOVES:
                    ns = succ(st, dx, dy)
                    if ns is None or ns in dist:
                        continue
                    dist[ns] = dist[st] + 1
                    q.append(ns)
            return None

        s0 = (game._player.x, game._player.y, game._next_val)
        d0 = dist_to_goal(s0)
        if not d0:                                # already solved / unreachable
            return []
        opt: list[int] = []
        for act, dx, dy in _MOVES:
            ns = succ(s0, dx, dy)
            if ns is None:
                continue
            d = dist_to_goal(ns)
            if d is not None and d == d0 - 1:
                opt.append(act)
        return opt


if __name__ == "__main__":
    sys.exit(Nu01Solver.main())
