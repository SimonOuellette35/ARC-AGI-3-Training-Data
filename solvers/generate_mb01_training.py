"""Generate Phase-1 training data for the mb01 (Magnet Crates) game.

A thin ``BaseSolver`` subclass over the local mb01 game (games/mb01/mb01.py).
mb01 is a small sokoban-like puzzle: the player moves one cell per action
(ACTION1..4 = up/down/left/right) and, after every move, each metal crate slides
one cell TOWARD the player if the destination is clear. The level is won when the
player steps onto the goal.

The board layout is fixed but its COLOURS (background, agent, metal crates + the
matching crate-count legend) are randomised per seed from the game's instance
RNG. This generator makes that augmentation DETERMINISTIC per (seed, level): the
overridden ``set_level`` seeds ``game._rng`` with ``mb01:<seed>:<level>`` before
``game.set_level``, which on_set_level -> _randomize_level consumes, so one
episode == one seed, reproducible byte-for-byte.

There is no reliable stand-alone solver for mb01, so ``solve_from`` embeds an
engine-as-oracle breadth-first search over a deepcopy of the LIVE game. The state
space (player + crate positions on a 10x10 grid) is tiny, so BFS finds a shortest
win plan quickly. Each candidate simple action is applied via a fast
(non-rendering) step; visited states are keyed by the player position plus the
sorted crate positions.

supports_recovery is False: crates can slide into positions from which they can
never be recovered (a sokoban-style permanent stuck state), so an exploratory
detour could brick the level.

Usage (run from the repo root):
    python solvers/generate_mb01_training.py --episodes 1000 \
        --out data/training_multi_level/mb01
"""

from __future__ import annotations

import copy
import sys
import random
from collections import deque
from pathlib import Path

# Repo root (parent of solvers/) -- that's where the games/ package lives.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameAction, GameState  # noqa: E402
from games.mb01.mb01 import Mb01  # noqa: E402
from solvers.base_solver import BaseSolver  # noqa: E402

_STEP_GUARD = 4000

# Movement actions available in mb01.
_MOVES = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]


def _fast_step(game, action: GameAction):
    """Apply one simple action WITHOUT rendering. Returns (solved, dead)."""
    game._full_reset = False
    game._set_action(ActionInput(id=action))
    guard = 0
    while not game.is_action_complete():
        guard += 1
        if guard > _STEP_GUARD:
            break
        if game._next_level:
            break
        game.step()
    solved = game._next_level or game._state == GameState.WIN
    dead = game._state == GameState.GAME_OVER
    return solved, dead


def _key(game):
    """Hashable state key: player position + sorted crate positions."""
    player = (game._player.x, game._player.y)
    crates = tuple(sorted((c.x, c.y) for c in game._crates))
    return (player, crates)


class Mb01Solver(BaseSolver):
    game_id = "mb01"
    # ``solve_from`` runs a fresh engine-as-oracle BFS over a deepcopy of the LIVE
    # game, so it genuinely re-plans from any perturbed state -> "replan" recovery.
    # If exploration slides a crate into a permanently stuck position the BFS
    # returns [] (unsolvable) and the base falls back to a RESET to the initial
    # board, so the episode still wins.
    supports_recovery = True
    recovery_mode = "replan"

    def make_game(self, seed: int):
        self._seed = seed
        return Mb01()

    def set_level(self, game, level_idx: int) -> None:
        """Make level `level_idx` DETERMINISTIC in (seed, level): seed the
        instance RNG (colours) before set_level, which on_set_level ->
        _randomize_level consumes."""
        game._rng = random.Random(f"mb01:{self._seed}:{level_idx}")
        game.set_level(level_idx)
        # One reused instance spans all levels; the native drive breaks on the
        # engine's ``_next_level`` flag without clearing it, so clear it here or
        # the next level is spuriously reported solved in a single action.
        game._next_level = False

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    def solve_from(self, game, level_idx: int, seed: int):
        """Engine-as-oracle BFS from a deepcopy of the LIVE game for a shortest
        win plan (moves only). Empty -> fail the episode."""
        root = copy.deepcopy(game)
        seen = {_key(root)}
        queue = deque([(root, [])])

        while queue:
            node, path = queue.popleft()
            for action in _MOVES:
                child = copy.deepcopy(node)
                solved, dead = _fast_step(child, action)
                if dead:
                    continue
                if solved:
                    return path + [action]
                k = _key(child)
                if k in seen:
                    continue
                seen.add(k)
                queue.append((child, path + [action]))
        return []


if __name__ == "__main__":
    sys.exit(Mb01Solver.main())
