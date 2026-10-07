"""Generate Phase-1 training data for the Maze game (games/maze/maze.py).

A BFS expert solves every level of a seed; each WIN seed is one multi-level
episode. One maze seed is already a multi-level game (9x9..33x33). The harness
(record/replay, schema, CLI, exploration prefix + bursts) lives in ``BaseSolver``.

Maze action set (absolute 4-direction movement; no turning):
    ACTION1 left (0,-1)   ACTION2 right (0,1)   ACTION3 up (-1,0)   ACTION4 down (1,0)

Usage (run from the repo root):
    python solvers/generate_maze_training.py --episodes 1000 --out data/maze_training_multi_level
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                         # noqa: E402
from arcengine import ActionInput, GameAction, GameState   # noqa: E402
from games.maze.maze import Maze                           # noqa: E402
from solvers.base_solver import BaseSolver, DriveResult, _ID_TO_GAMEACTION  # noqa: E402

# (dx, dy) -> the simple action that produces it. Matches Maze._DELTAS.
_DELTA_TO_ACTION = {
    (0, -1): GameAction.ACTION1, (0, 1): GameAction.ACTION2,
    (-1, 0): GameAction.ACTION3, (1, 0): GameAction.ACTION4,
}


class MazeSolver(BaseSolver):
    game_id = "maze"
    # BFS re-plans from the player's LIVE position and the maze is fully
    # reversible, so exploratory detours are always recoverable.
    supports_recovery = True

    def make_game(self, seed: int):
        return Maze(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    def solve_from(self, game, level_idx: int, seed: int):
        """BFS from the player's CURRENT position to the goal on the live level."""
        level = game.current_level
        grid_w, grid_h = level.grid_size
        player = level.get_sprites_by_tag("player")[0]
        start = (player.x, player.y)
        goal_sprite = level.get_sprites_by_tag("goal")[0]
        goal = (goal_sprite.x, goal_sprite.y)

        walls = {(s.x, s.y) for s in level.get_sprites()
                 if s.is_collidable and "player" not in s._tags}

        queue = deque([(start, [])])
        visited = {start}
        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == goal:
                return [_DELTA_TO_ACTION[step] for step in path]
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = x + dx, y + dy
                if (nx, ny) in visited or (nx, ny) in walls:
                    continue
                if not (0 <= nx < grid_w and 0 <= ny < grid_h):
                    continue
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(dx, dy)]))
        return []                                          # unreachable -> fail episode

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-optimal next move: the directions whose target cell is one
        step closer to the goal (``dist == dist(start) - 1``). Feeds the base's
        STOCHASTIC OPTIMAL sampling so each episode takes a different shortest path.
        O(cells) via one BFS from the goal -- no game copy needed."""
        level = game.current_level
        grid_w, grid_h = level.grid_size
        player = level.get_sprites_by_tag("player")[0]
        start = (player.x, player.y)
        goal_sprite = level.get_sprites_by_tag("goal")[0]
        goal = (goal_sprite.x, goal_sprite.y)
        if start == goal:
            return []
        walls = {(s.x, s.y) for s in level.get_sprites()
                 if s.is_collidable and "player" not in s._tags}
        dist = {goal: 0}
        q = deque([goal])
        while q:
            x, y = q.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = x + dx, y + dy
                if ((nx, ny) in dist or (nx, ny) in walls
                        or not (0 <= nx < grid_w and 0 <= ny < grid_h)):
                    continue
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))
        d0 = dist.get(start)
        if d0 is None:
            return []
        return [_DELTA_TO_ACTION[(dx, dy)] for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))
                if dist.get((start[0] + dx, start[1] + dy), 1 << 30) == d0 - 1]

    def drive(self, game, action) -> DriveResult:
        """Maze drives via ``perform_action`` and advances levels in place, so a
        solve is a level change (or WIN), not the native ``_next_level`` flag.

        ``perform_action`` is the AGENT-facing entry point, so it takes a SCREEN action
        and de-rotates it for the core game. ``solve_from`` plans off the sprites, which
        are upright, so the plan is converted on the way in. (The base's own ``drive``
        uses ``_set_action``, which bypasses the wrapper and needs no conversion.)"""
        start_level = game.current_level
        self._note_rotation(game)
        screen = self._to_screen(action)
        fd = game.perform_action(ActionInput(id=_ID_TO_GAMEACTION[screen.action_id]))
        frame = np.asarray(fd.frame[0])
        dead = game._state == GameState.GAME_OVER
        solved = game._state == GameState.WIN or game.current_level is not start_level
        return DriveResult(frame, solved, dead)


if __name__ == "__main__":
    sys.exit(MazeSolver.main())
