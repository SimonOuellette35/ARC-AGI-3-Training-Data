"""Generate Phase-1 training data for the fs02 game.

fs02 (games/fs02/fs02.py): the player moves on a grid with ACTION1..4 (up/down/
left/right). Stepping on ANY orange pressure plate opens the door (OR rule,
redundant plates); reaching the target with the door open advances to the next
level. There are 5 levels. The board is RANDOMISED per seed (player / switch /
target placement -- wall+door structure fixed -- plus all colours). This
generator makes that augmentation DETERMINISTIC per (seed, level): before
recording each level it seeds ``game._rng`` with ``fs02:<seed>:<level>``
(consumed by on_set_level -> _randomize_level), so one episode == one seed,
reproducible byte-for-byte. A seed whose board can't be solved is skipped.

The expert is a BFS over (player_x, player_y, door_open). ``solve_from`` reads
the LIVE game (player cell, whether the door is already open, wall/door/switch/
target cells from the live sprites), so it re-plans correctly from any state.

Reversible / always-solvable: the door only ever opens (plate activation is
monotone) and movement is reversible, so ``supports_recovery = True`` -- the
exploration prefix + bursts are safe.

Usage (run from the repo root):
    python solvers/generate_fs02_training.py --episodes 1000 \
        --out data/training_multi_level/fs02
"""

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction                            # noqa: E402
from games.fs02.fs02 import Fs02                            # noqa: E402
from solvers.base_solver import BaseSolver                  # noqa: E402

# ACTION -> (dx, dy) exactly as fs02.step() interprets them (1:1 grid coords).
_MOVE_DELTAS = [
    (GameAction.ACTION1, (0, -1)),
    (GameAction.ACTION2, (0, 1)),
    (GameAction.ACTION3, (-1, 0)),
    (GameAction.ACTION4, (1, 0)),
]


class Fs02Solver(BaseSolver):
    game_id = "fs02"
    # Door only opens (monotone), movement reversible -> always solvable and
    # solve_from reads live state, so recovery is safe.
    supports_recovery = True

    def make_game(self, seed: int):
        self._seed = seed
        return Fs02()

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    def set_level(self, game, level_idx: int) -> None:
        # Deterministic per-(seed, level) augmentation: seed the instance RNG that
        # on_set_level -> _randomize_level consumes, BEFORE set_level. Also clear
        # the engine's ``_next_level`` flag the base breaks on. on_set_level
        # rebuilds the level from the clean descriptor, so re-setting is clean.
        game._next_level = False
        game._rng = random.Random(f"fs02:{self._seed}:{level_idx}")
        game.set_level(level_idx)

    def solve_from(self, game, level_idx: int, seed: int):
        """BFS to the target from the LIVE player state on the current level.

        State = (px, py, door_open). Reads live: player cell, whether the door is
        already open (``len(game._door) == 0``), and wall/door/switch/target cells
        from the live sprites. The door-cell set reflects the CURRENT layout
        (empty once open); the ``door_open`` flag gates it, so the plan is correct
        from either a closed or an already-open state. Returns a simple-action
        (ACTION1..4) plan."""
        level = game.current_level
        gw, gh = level.grid_size
        walls = {(s.x, s.y) for s in level.get_sprites_by_tag("wall")}
        door_cells = {(s.x, s.y) for s in level.get_sprites_by_tag("door")}
        switches = set(game._switch_positions)
        targets = {(t.x, t.y) for t in game._targets}
        door_open = len(game._door) == 0

        start = (game._player.x, game._player.y, door_open)
        if start[:2] in targets and door_open:
            return []
        seen = {start}
        q = deque([(start, [])])
        while q:
            (px, py, opened), path = q.popleft()
            for action, (dx, dy) in _MOVE_DELTAS:
                nx, ny = px + dx, py + dy
                if not (0 <= nx < gw and 0 <= ny < gh):
                    continue
                if (nx, ny) in walls:
                    continue
                if (nx, ny) in door_cells and not opened:
                    continue
                nopened = opened or (nx, ny) in switches
                nstate = (nx, ny, nopened)
                if nstate in seen:
                    continue
                npath = path + [action]
                if (nx, ny) in targets and nopened:
                    return npath
                seen.add(nstate)
                q.append((nstate, npath))
        return []                                  # unsolvable from here

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-optimal next move at the LIVE state: each direction whose
        successor is one step closer to the target with the door open. Feeds the
        base's STOCHASTIC OPTIMAL sampling so each episode takes a different
        shortest path.

        The door-open flag is part of the state (monotone), so distance is
        measured over ``(x, y, door_open)`` -- a BFS per candidate first move (no
        game copy). The transition mirrors ``solve_from`` exactly, so the
        canonical head is always in the returned set."""
        level = game.current_level
        gw, gh = level.grid_size
        walls = {(s.x, s.y) for s in level.get_sprites_by_tag("wall")}
        door_cells = {(s.x, s.y) for s in level.get_sprites_by_tag("door")}
        switches = set(game._switch_positions)
        targets = {(t.x, t.y) for t in game._targets}
        door_open = len(game._door) == 0

        def succ(state, dx, dy):
            px, py, opened = state
            nx, ny = px + dx, py + dy
            if not (0 <= nx < gw and 0 <= ny < gh) or (nx, ny) in walls:
                return None
            if (nx, ny) in door_cells and not opened:
                return None
            return (nx, ny, opened or (nx, ny) in switches)

        def dist_to_goal(s0):
            q = deque([s0])
            dist = {s0: 0}
            while q:
                st = q.popleft()
                px, py, opened = st
                if (px, py) in targets and opened:
                    return dist[st]
                for _action, (dx, dy) in _MOVE_DELTAS:
                    ns = succ(st, dx, dy)
                    if ns is None or ns in dist:
                        continue
                    dist[ns] = dist[st] + 1
                    q.append(ns)
            return None

        s0 = (game._player.x, game._player.y, door_open)
        d0 = dist_to_goal(s0)
        if not d0:                                 # already solved / unreachable
            return []
        opt: list = []
        for action, (dx, dy) in _MOVE_DELTAS:
            ns = succ(s0, dx, dy)
            if ns is None:
                continue
            d = dist_to_goal(ns)
            if d is not None and d == d0 - 1:
                opt.append(action)
        return opt


if __name__ == "__main__":
    sys.exit(Fs02Solver.main())
