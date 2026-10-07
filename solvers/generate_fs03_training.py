"""Generate Phase-1 training data for the fs03 game.

fs03 (games/fs03/fs03.py): a k-of-n pressure-plate game. The player (ACTION1..4 =
up/down/left/right) must step on ``required`` distinct yellow switch plates (any
order; extra plates optional). Once enough distinct plates are activated the door
sprite is removed, and walking onto the target wins the level. There are 5
levels. The board is RANDOMISED per seed (player / switch / target placement --
wall+door structure fixed -- plus colours). This generator makes that
augmentation DETERMINISTIC per (seed, level): before recording each level it
seeds ``game._rng`` with ``fs03:<seed>:<level>`` (consumed by on_set_level ->
_randomize_level). A seed whose board is unsolvable is skipped.

The expert is an exact BFS over (px, py, frozenset(activated_plates)).
``solve_from`` reads the LIVE game (player cell, already-activated plates
``game._activated``, ``game._required``, and wall/door/switch/target cells from
the live sprites), so it re-plans correctly from any state.

Reversible / always-solvable: plate activation is monotone and movement is
reversible, so ``supports_recovery = True`` -- the exploration prefix + bursts
are safe.

Usage (run from repo root):
    python solvers/generate_fs03_training.py --episodes 1000 \
        --out data/training_multi_level/fs03
"""

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction                            # noqa: E402
from games.fs03.fs03 import Fs03                            # noqa: E402
from solvers.base_solver import BaseSolver                  # noqa: E402

# ACTION -> (dx, dy) exactly as fs03.step() interprets them (1:1 grid coords).
_ACTION_DELTA = [
    (GameAction.ACTION1, (0, -1)),
    (GameAction.ACTION2, (0, 1)),
    (GameAction.ACTION3, (-1, 0)),
    (GameAction.ACTION4, (1, 0)),
]


class Fs03Solver(BaseSolver):
    game_id = "fs03"
    # Plate activation monotone, movement reversible -> always solvable and
    # solve_from reads live state, so recovery is safe.
    supports_recovery = True

    def make_game(self, seed: int):
        self._seed = seed
        return Fs03()

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    def set_level(self, game, level_idx: int) -> None:
        # Deterministic per-(seed, level) augmentation: seed the instance RNG that
        # on_set_level -> _randomize_level consumes, BEFORE set_level. Also clear
        # the engine's ``_next_level`` flag the base breaks on. on_set_level
        # rebuilds the level from the clean descriptor, so re-setting is clean.
        game._next_level = False
        game._rng = random.Random(f"fs03:{self._seed}:{level_idx}")
        game.set_level(level_idx)

    def solve_from(self, game, level_idx: int, seed: int):
        """BFS to the target from the LIVE player state on the current level.

        State = (px, py, frozenset(activated)). Reads live: player cell, the
        already-activated plate set (``game._activated``), the required count
        (``game._required``), and wall/door/switch/target cells from the live
        sprites. The door blocks until ``len(activated) >= required``; the
        door-cell set reflects the CURRENT layout (empty once open) and the count
        gate makes the plan correct from a closed or already-open state. Returns a
        simple-action (ACTION1..4) plan."""
        level = game.current_level
        gw, gh = level.grid_size
        walls = frozenset((s.x, s.y) for s in level.get_sprites_by_tag("wall"))
        door_cells = frozenset((s.x, s.y) for s in level.get_sprites_by_tag("door"))
        switches = frozenset(game._switch_positions)
        targets = frozenset((t.x, t.y) for t in game._targets)
        required = int(game._required)

        def _apply(state, dx, dy):
            px, py, activated = state
            nx, ny = px + dx, py + dy
            if not (0 <= nx < gw and 0 <= ny < gh):
                return state
            if (nx, ny) in walls:
                return state
            if (nx, ny) in door_cells and len(activated) < required:
                return state
            px, py = nx, ny
            if (px, py) in switches and (px, py) not in activated:
                activated = activated | {(px, py)}
            return (px, py, activated)

        def _is_win(state):
            px, py, activated = state
            return (px, py) in targets and len(activated) >= required

        start = (game._player.x, game._player.y, frozenset(game._activated))
        if _is_win(start):
            return []
        seen = {start}
        q = deque([(start, [])])
        while q:
            state, path = q.popleft()
            for action, (dx, dy) in _ACTION_DELTA:
                nxt = _apply(state, dx, dy)
                if nxt in seen:
                    continue
                npath = path + [action]
                if _is_win(nxt):
                    return npath
                seen.add(nxt)
                q.append((nxt, npath))
        return []                                  # unsolvable from here

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-optimal next move at the LIVE state: each direction whose
        successor is one step closer to the target with enough distinct plates
        activated. Feeds the base's STOCHASTIC OPTIMAL sampling so each episode
        takes a different shortest path.

        The activated-plate set is part of the (monotone) state, so distance is
        measured over ``(x, y, frozenset(activated))`` -- a BFS per candidate
        first move (no game copy). The transition (``_apply``/``_is_win``) mirrors
        ``solve_from`` exactly, so the canonical head is always in the set."""
        level = game.current_level
        gw, gh = level.grid_size
        walls = frozenset((s.x, s.y) for s in level.get_sprites_by_tag("wall"))
        door_cells = frozenset((s.x, s.y) for s in level.get_sprites_by_tag("door"))
        switches = frozenset(game._switch_positions)
        targets = frozenset((t.x, t.y) for t in game._targets)
        required = int(game._required)

        def _apply(state, dx, dy):
            px, py, activated = state
            nx, ny = px + dx, py + dy
            if not (0 <= nx < gw and 0 <= ny < gh):
                return state
            if (nx, ny) in walls:
                return state
            if (nx, ny) in door_cells and len(activated) < required:
                return state
            px, py = nx, ny
            if (px, py) in switches and (px, py) not in activated:
                activated = activated | {(px, py)}
            return (px, py, activated)

        def _is_win(state):
            px, py, activated = state
            return (px, py) in targets and len(activated) >= required

        def dist_to_goal(s0):
            q = deque([s0])
            dist = {s0: 0}
            while q:
                st = q.popleft()
                if _is_win(st):
                    return dist[st]
                for _action, (dx, dy) in _ACTION_DELTA:
                    ns = _apply(st, dx, dy)
                    if ns == st or ns in dist:     # blocked no-op / already seen
                        continue
                    dist[ns] = dist[st] + 1
                    q.append(ns)
            return None

        s0 = (game._player.x, game._player.y, frozenset(game._activated))
        d0 = dist_to_goal(s0)
        if not d0:                                 # already solved / unreachable
            return []
        opt: list = []
        for action, (dx, dy) in _ACTION_DELTA:
            ns = _apply(s0, dx, dy)
            if ns == s0:                           # blocked no-op
                continue
            d = dist_to_goal(ns)
            if d is not None and d == d0 - 1:
                opt.append(action)
        return opt


if __name__ == "__main__":
    sys.exit(Fs03Solver.main())
