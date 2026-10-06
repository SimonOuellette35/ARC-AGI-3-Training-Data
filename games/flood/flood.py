"""
Color Flood
===========
One corner owns an expanding region (randomized per level from the 4 corners).
Use ACTION1–ACTION4 to cycle
through the 5 available colors — the owned region immediately recolors to show
a live preview of what color will be applied.  Press ACTION5 to confirm the
flood: the owned region locks in that color and absorbs adjacent cells that
already share it.  Flood the entire grid to win.

  ACTION1 (↑) / ACTION3 (←) — cycle to previous color (preview updates live)
  ACTION2 (↓) / ACTION4 (→) — cycle to next color (preview updates live)
  ACTION5                    — apply flood fill with previewed color
  ACTION7                    — undo last flood fill

Colors (in cycle order):
  red (#F93C31) · blue (#1E93FF) · yellow (#FFDC00) · green (#4FCC30) · orange (#FF851B)

Win condition : every cell is the same color.
Lose condition: step budget exhausted (shown as progress bar at bottom).
Levels        : 4, with grids from 8×8 to 14×14, all with 5 colors.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
from utils.arc_game import AugmentedGame
from arcengine import (
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------

# Five ARC palette indices used for the flood grid
_PALETTE = [8, 9, 11, 14, 12, 6, 1]   # red, blue, yellow, green, orange, pink, teal
C_BORDER  = 5                    # black letterbox / background

# (grid_w, grid_h, num_colors) — all capped at 5 so ACTION1–5 suffice
_CONFIGS = [
    ( 8,  8, 5),
    (10, 10, 5),
    (12, 12, 5),
    (14, 14, 5),
    (16, 16, 6),
    (18, 18, 6),
    (20, 20, 7),
]

# Step budgets per level (number of flood-fill moves allowed)
_STEP_BUDGETS = [30, 45, 65, 90, 130, 175, 230]

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
            frame[63, x] = 7 if x < filled else 4
        return frame


# ---------------------------------------------------------------------------
# Flood fill helper
# ---------------------------------------------------------------------------

def _to_pixels(grid: np.ndarray, num_colors: int) -> np.ndarray:
    lookup = np.array(_PALETTE[:num_colors], dtype=np.int8)
    return lookup[grid]


def _flood(grid: np.ndarray, new_idx: int, origin: tuple[int, int]) -> bool:
    """BFS flood-fill from origin. Returns True if the grid changed."""
    h, w = grid.shape
    ox, oy = origin
    old = int(grid[oy, ox])
    if new_idx == old:
        return False
    q = deque([(ox, oy)])
    seen = {(ox, oy)}
    while q:
        x, y = q.popleft()
        grid[y, x] = new_idx
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            p = (nx, ny)
            if p not in seen and 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == old:
                seen.add(p)
                q.append(p)
    return True


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class Flood(AugmentedGame):

    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: the base draws this level's rotation
        rng = random.Random(seed)
        self._step_counter = StepCounter(_STEP_BUDGETS[0])

        self._initial_grids: list[np.ndarray] = []
        self._num_colors_list: list[int] = []
        self._origins: list[tuple[int, int]] = []
        self._current_grid: np.ndarray = np.zeros((1, 1), dtype=np.int8)
        self._num_colors: int = 5
        self._origin: tuple[int, int] = (0, 0)
        self._selected_color: int = 0
        self._history: list[np.ndarray] = []

        levels: list[Level] = []
        for grid_w, grid_h, num_colors in _CONFIGS:
            grid = np.array(
                [[rng.randint(0, num_colors - 1) for _ in range(grid_w)]
                 for _ in range(grid_h)],
                dtype=np.int8,
            )
            self._initial_grids.append(grid)
            self._num_colors_list.append(num_colors)
            self._origins.append(
                rng.choice(
                    [(0, 0), (grid_w - 1, 0), (0, grid_h - 1), (grid_w - 1, grid_h - 1)]
                )
            )
            pixels = _to_pixels(grid, num_colors)
            sprite = Sprite(
                pixels=pixels, name="board", collidable=False, layer=0
            ).set_position(0, 0)
            levels.append(Level(sprites=[sprite], grid_size=(grid_w, grid_h)))

        super().__init__(
            game_id="flood",
            levels=levels,
            camera=Camera(
                background=C_BORDER,
                letter_box=C_BORDER,
                interfaces=[self._step_counter],
            ),
            available_actions=[1, 2, 3, 4, 5, 7],
        )

    def _board_sprite(self) -> Sprite:
        return next(s for s in self.current_level.get_sprites() if s.name == "board")

    def _owned_cells(self) -> list[tuple[int, int]]:
        """BFS from level origin — return all cells in the owned region."""
        h, w = self._current_grid.shape
        ox, oy = self._origin
        own = int(self._current_grid[oy, ox])
        visited: set[tuple[int, int]] = {(ox, oy)}
        q: deque[tuple[int, int]] = deque([(ox, oy)])
        while q:
            x, y = q.popleft()
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and 0 <= nx < w and 0 <= ny < h \
                        and int(self._current_grid[ny, nx]) == own:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return list(visited)

    def _refresh(self) -> None:
        """Redraw board with owned region shown in the currently selected color."""
        pixels = _to_pixels(self._current_grid, self._num_colors)
        preview = _PALETTE[self._selected_color]
        for (x, y) in self._owned_cells():
            pixels[y, x] = preview
        self._board_sprite().pixels = pixels

    def on_set_level(self, level: Level) -> None:
        # Clear board sprites to ensure idempotency
        for sprite in list(level.get_sprites()):
            if sprite.name == "board":
                level.remove_sprite(sprite)
        idx = self._current_level_index
        self._current_grid = self._initial_grids[idx].copy()
        self._num_colors = self._num_colors_list[idx]
        self._origin = self._origins[idx]
        self._selected_color = 0
        self._history.clear()
        self._step_counter.reset(_STEP_BUDGETS[idx])
        pixels = _to_pixels(self._current_grid, self._num_colors)
        sprite = Sprite(
            pixels=pixels, name="board", collidable=False, layer=0
        ).set_position(0, 0)
        level.add_sprite(sprite)
        self._refresh()  # apply initial preview

    def step(self) -> None:
        action = self.action.id

        # ACTION7 — undo last flood fill (costs a step)
        if action == GameAction.ACTION7:
            self._step_counter.decrement()
            if self._history:
                self._current_grid = self._history.pop()
            self._refresh()  # always redraw (restores preview for current selection)
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        # ACTION1 / ACTION3 — cycle to previous color, update preview (free)
        if action in (GameAction.ACTION1, GameAction.ACTION3):
            self._selected_color = (self._selected_color - 1) % self._num_colors
            self._refresh()
            self.complete_action()
            return

        # ACTION2 / ACTION4 — cycle to next color, update preview (free)
        if action in (GameAction.ACTION2, GameAction.ACTION4):
            self._selected_color = (self._selected_color + 1) % self._num_colors
            self._refresh()
            self.complete_action()
            return

        # ACTION5 — apply flood fill with selected color (costs a step)
        if action == GameAction.ACTION5:
            self._step_counter.decrement()
            color_idx = self._selected_color
            if color_idx < self._num_colors:
                prev = self._current_grid.copy()
                if _flood(self._current_grid, color_idx, self._origin):
                    self._history.append(prev)
                    self._refresh()
                    ox, oy = self._origin
                    if np.all(self._current_grid == self._current_grid[oy, ox]):
                        self.next_level()
                        self.complete_action()
                        return
            if not self._step_counter.steps_remaining:
                self.lose()
            self.complete_action()
            return

        self.complete_action()
