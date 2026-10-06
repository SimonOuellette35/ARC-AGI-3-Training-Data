"""
Maze Navigation Game
====================
Navigate a randomly-generated perfect maze from the start (top-left)
to the goal (bottom-right, green). A step budget is shown as a progress
bar along the bottom of the frame; exhausting it ends the game.

Actions:
  ACTION1 (↑) — Move up
  ACTION2 (↓) — Move down
  ACTION3 (←) — Move left
  ACTION4 (→) — Move right
  ACTION7     — Undo last move

Color key:
  0  (#FFFFFF) — open floor      1  (#1E93FF) — wall
  3  (#4FCC30) — goal            14 (#FFFFFF) — player
  5  (#000000) — letterbox border

Win condition : Reach the green goal cell.
Lose condition: Step budget exhausted.
Levels        : 4 levels with increasing maze size:
                  Level 1 —  9× 9  (budget 100)
                  Level 2 — 13×13  (budget 200)
                  Level 3 — 17×17  (budget 400)
                  Level 4 — 21×21  (budget 600)
"""

from __future__ import annotations

import random

import numpy as np
from utils.arc_game import AugmentedGame
from arcengine import (
    ARCBaseGame,
    BlockingMode,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)

from utils.arc_game import clear_dynamic_sprites

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
C_FLOOR  = 0   # white  — open path
C_WALL   = 1   # blue   — wall
C_GOAL   = 14  # green  — goal
C_BORDER = 5   # black  — camera letterbox
C_PLAYER = 3   # green  — player

# Step budgets per level
_STEP_BUDGETS = [100, 200, 400, 600, 900, 1300, 1800]


# ---------------------------------------------------------------------------
# Step counter HUD
# ---------------------------------------------------------------------------

class StepCounter(RenderableUserDisplay):
    """Progress bar rendered on the bottom row of the 64×64 frame."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.steps_remaining = max_steps

    def decrement(self) -> None:
        if self.steps_remaining > 0:
            self.steps_remaining -= 1

    def reset(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.steps_remaining = max_steps

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.max_steps == 0:
            return frame
        filled = round(64 * self.steps_remaining / self.max_steps)
        for x in range(64):
            frame[63, x] = 7 if x < filled else 4  # light-pink / near-black
        return frame


# ---------------------------------------------------------------------------
# Sprite factories
# ---------------------------------------------------------------------------

def _wall_sprite(x: int, y: int) -> Sprite:
    s = Sprite(pixels=[[C_WALL]], name="wall", collidable=True, layer=-1).set_position(x, y)
    s.set_blocking(BlockingMode.BOUNDING_BOX)
    return s


def _goal_sprite(x: int, y: int) -> Sprite:
    return (
        Sprite(pixels=[[C_GOAL]], name="goal", collidable=False, tags=["goal"], layer=0)
        .set_position(x, y)
    )


def _player_sprite(x: int, y: int) -> Sprite:
    s = Sprite(pixels=[[C_PLAYER]], name="player", collidable=True, tags=["player"], layer=1).set_position(x, y)
    s.set_blocking(BlockingMode.BOUNDING_BOX)
    return s


# ---------------------------------------------------------------------------
# Maze generator — recursive DFS backtracking
# ---------------------------------------------------------------------------

def _generate_maze(grid_w: int, grid_h: int, rng: random.Random) -> list[list[int]]:
    assert grid_w % 2 == 1 and grid_h % 2 == 1
    cols = (grid_w - 1) // 2
    rows = (grid_h - 1) // 2
    grid = [[1] * grid_w for _ in range(grid_h)]

    def carve(cx: int, cy: int) -> None:
        grid[2 * cy + 1][2 * cx + 1] = 0
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        rng.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cols and 0 <= ny < rows and grid[2 * ny + 1][2 * nx + 1] == 1:
                grid[2 * cy + 1 + dy][2 * cx + 1 + dx] = 0
                carve(nx, ny)

    carve(0, 0)
    return grid


def _build_level(grid_w: int, grid_h: int, rng: random.Random) -> Level:
    maze = _generate_maze(grid_w, grid_h, rng)
    sprites: list[Sprite] = []
    for y in range(grid_h):
        for x in range(grid_w):
            if maze[y][x] == 1:
                sprites.append(_wall_sprite(x, y))
    sprites.append(_goal_sprite(grid_w - 2, grid_h - 2))
    return Level(sprites=sprites, grid_size=(grid_w, grid_h))


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class Maze(AugmentedGame):

    _LEVEL_SIZES: list[tuple[int, int]] = [
        ( 9,  9),
        (13, 13),
        (17, 17),
        (21, 21),
        (25, 25),
        (29, 29),
        (33, 33),
    ]

    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: the base draws this level's rotation
        self._rng = random.Random(seed)
        self._history: list[tuple[int, int]] = []
        self._step_counter = StepCounter(_STEP_BUDGETS[0])

        levels = [_build_level(w, h, self._rng) for w, h in self._LEVEL_SIZES]

        super().__init__(
            game_id="maze",
            levels=levels,
            camera=Camera(
                background=C_FLOOR,
                letter_box=C_BORDER,
                interfaces=[self._step_counter],
            ),
            available_actions=[1, 2, 3, 4, 7],
        )

    def on_set_level(self, level: Level) -> None:
        # Levels are persistent objects: seating one twice re-runs this hook, so
        # drop the previous pass's player before adding a new one.
        clear_dynamic_sprites(level, tags=["player"])
        self._history.clear()
        self._step_counter.reset(_STEP_BUDGETS[self._current_level_index])
        level.add_sprite(_player_sprite(1, 1))

    _DELTAS: dict[GameAction, tuple[int, int]] = {
        GameAction.ACTION1: ( 0, -1),
        GameAction.ACTION2: ( 0,  1),
        GameAction.ACTION3: (-1,  0),
        GameAction.ACTION4: ( 1,  0),
    }

    def step(self) -> None:
        player = self.current_level.get_sprites_by_tag("player")[0]

        self._step_counter.decrement()

        if self.action.id == GameAction.ACTION7:
            if self._history:
                prev_x, prev_y = self._history.pop()
                player.set_position(prev_x, prev_y)
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        dx, dy = self._DELTAS.get(self.action.id, (0, 0))
        if dx or dy:
            prev = (player.x, player.y)
            collisions = self.try_move(player.name, dx, dy)
            if not collisions:
                self._history.append(prev)
                goal = self.current_level.get_sprites_by_tag("goal")[0]
                if player.x == goal.x and player.y == goal.y:
                    self.next_level()
                    self.complete_action()
                    return

        if not self._step_counter.steps_remaining:
            self.lose()

        self.complete_action()
