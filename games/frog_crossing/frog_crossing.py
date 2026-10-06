"""
Frog Crossing
=============
Hop colored frogs from their starting lily pads to matching colored target pads.
A frog can only hop one cell at a time (ACTION1-4) to an adjacent lily pad or
stone step; open water is impassable. If the target cell is occupied by another
frog the hop is blocked. ACTION5 cycles the active frog selection (highlighted
in white). ACTION7 undoes the last hop.

Color key:
  9  (#1E93FF) — water (background)
  14 (#4FCC30) — lily pad
  7  (#E6E6E6) — stone step (lighter grey)
  4  (#333333) — dark stone step
  0  (#FFFFFF) — active frog (white highlight)
  8  (#F93C31) — red frog / red target pad
  11 (#FFDC00) — yellow frog / yellow target pad
  12 (#FF851B) — orange frog / orange target pad
  6  (#E36EF6) — pink frog / pink target pad
  3  (#4FCC30) — green frog / green target pad
  5  (#000000) — letterbox border

Actions:
  ACTION1 (↑) — hop active frog up
  ACTION2 (↓) — hop active frog down
  ACTION3 (←) — hop active frog left
  ACTION4 (→) — hop active frog right
  ACTION5     — cycle active frog selection
  ACTION7     — undo last hop

Win condition : Every frog occupies the lily pad whose color matches it.
Lose condition: Step budget exhausted (progress bar at bottom).
Levels        : 4 puzzles — 3 frogs (easy) → 4 → 5 → 6 frogs (expert).
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

from utils.arc_game import AugmentedGame

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
C_WATER  = 9   # blue water background
C_PAD    = 14  # green lily pad
C_STONE  = 7   # stone step
C_BORDER = 5   # black border

# Frog / target colors (index i → frog i)
_FROG_COLORS = [8, 11, 12, 6, 3, 1, 2, 4, 10]  # red, yellow, orange, pink, green, blue, magenta, dark, teal
C_ACTIVE = 0   # white — active frog highlight

_STEP_BUDGETS = [120, 240, 380, 550, 750, 980, 1250]

# ---------------------------------------------------------------------------
# Cell size: CELL=4 gives max viewport 64×64 for 16×16 grid.
# ---------------------------------------------------------------------------
CELL = 4

# ---------------------------------------------------------------------------
# Sprite pixel art  (all CELL×CELL = 4×4 pixels)
# ---------------------------------------------------------------------------

def _frog_pixels(c: int, active: bool = False) -> list[list[int]]:
    """4×4 frog: eyes (white dots), body fill; active shows white highlight."""
    if active:
        return [
            [C_ACTIVE, C_ACTIVE, C_ACTIVE, C_ACTIVE],
            [c,        c,        c,        C_ACTIVE],
            [c,        c,        c,        c       ],
            [-1,       c,        c,        -1      ],
        ]
    return [
        [c, 0,  0,  c ],  # eyes (white dots)
        [c, c,  c,  c ],  # body top
        [c, c,  c,  c ],  # body bottom
        [-1, c, c, -1 ],  # legs
    ]

# Lily pad: 4×4 rounded diamond
_PAD_PIXELS = [
    [-1,    C_PAD,  C_PAD, -1   ],
    [C_PAD, C_PAD,  C_PAD, C_PAD],
    [C_PAD, C_PAD,  C_PAD, C_PAD],
    [-1,    C_PAD,  C_PAD, -1   ],
]

# Target pad: same shape but colored
def _target_pixels(c: int) -> list[list[int]]:
    return [
        [-1, c,  c,  -1],
        [c,  c,  c,   c],
        [c,  c,  c,   c],
        [-1, c,  c,  -1],
    ]

# Stone step: 4×4 textured grey block
_STONE_PIXELS = [
    [C_STONE, C_STONE, 4,        C_STONE],
    [C_STONE, 4,       C_STONE,  C_STONE],
    [4,       C_STONE, C_STONE,  4      ],
    [C_STONE, C_STONE, 4,        C_STONE],
]


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

class StepCounter(RenderableUserDisplay):
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
# Level configs: (grid_size, num_frogs, scramble_moves, removal_fraction)
# Grids are kept small; difficulty ramps via narrower paths (higher removal)
# and more scramble moves — not by growing the grid.
# ---------------------------------------------------------------------------
_CONFIGS = [
    (10, 3, 35, 0.35),   # easy:   moderate connectivity, 3 frogs
    (12, 4, 55, 0.48),   # medium: narrower paths, 4 frogs, same-ish grid
    (12, 5, 75, 0.58),   # hard:   quite narrow, 5 frogs, same grid size
    (14, 6, 95, 0.62),   # expert: very narrow corridors, 6 frogs
    (14, 7, 115, 0.66),  # master: 7 frogs, tighter corridors
    (16, 8, 140, 0.70),  # grandmaster: 8 frogs, 16x16 grid
    (16, 9, 165, 0.73),  # legendary: 9 frogs, densest corridors
]


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _make_grid(size: int, rng: random.Random,
               removal_fraction: float = 0.35) -> list[list[int]]:
    """
    Create a pond grid where 0=water, 1=lily_pad, 2=stone_step.
    All passable cells form a connected region.  Higher removal_fraction
    produces narrower, more corridor-like layouts that force frogs to
    navigate around each other.
    """
    grid = [[1] * size for _ in range(size)]
    # Border = water
    for i in range(size):
        grid[0][i] = grid[size - 1][i] = 0
        grid[i][0] = grid[i][size - 1] = 0

    interior = [(x, y) for y in range(2, size - 2) for x in range(2, size - 2)]
    rng.shuffle(interior)

    def passable(g: list[list[int]]) -> set[tuple[int, int]]:
        return {(x, y) for y in range(size) for x in range(size) if g[y][x] > 0}

    def connected(g: list[list[int]]) -> bool:
        cells = passable(g)
        if not cells:
            return False
        start = next(iter(cells))
        seen: set[tuple[int, int]] = {start}
        q: deque[tuple[int, int]] = deque([start])
        while q:
            cx, cy = q.popleft()
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nb = (cx + dx, cy + dy)
                if nb not in seen and nb in cells:
                    seen.add(nb)
                    q.append(nb)
        return len(seen) == len(cells)

    # Remove up to removal_fraction of interior cells as water, preserving connectivity.
    # Iterate the interior list multiple times to get close to the target fraction.
    target_remove = int(len(interior) * removal_fraction)
    removed = 0
    for pass_n in range(3):  # up to 3 passes to approach target
        if removed >= target_remove:
            break
        rng.shuffle(interior)
        for (x, y) in interior:
            if removed >= target_remove:
                break
            if grid[y][x] == 0:
                continue  # already removed
            grid[y][x] = 0
            if connected(grid):
                removed += 1
            else:
                grid[y][x] = 1  # restore

    # Convert some pads to stone steps
    pads = [(x, y) for y in range(size) for x in range(size) if grid[y][x] == 1]
    rng.shuffle(pads)
    for (x, y) in pads[: len(pads) // 5]:
        grid[y][x] = 2  # stone step

    return grid


def _passable(grid: list[list[int]], x: int, y: int) -> bool:
    size = len(grid)
    return 0 <= x < size and 0 <= y < size and grid[y][x] > 0


def _generate(size: int, num_frogs: int, scramble: int,
              removal_fraction: float, rng: random.Random):
    """Return (grid, frog_start_positions, target_positions, colors)."""
    grid = _make_grid(size, rng, removal_fraction)
    cells = [(x, y) for y in range(size) for x in range(size) if grid[y][x] > 0]
    rng.shuffle(cells)

    # targets are fixed positions; frogs start on targets (solved), then scramble
    targets = cells[:num_frogs]
    colors = _FROG_COLORS[:num_frogs]

    frog_pos: list[tuple[int, int]] = list(targets)

    for _ in range(scramble):
        frog_set = set(frog_pos)
        movable = list(range(num_frogs))
        rng.shuffle(movable)
        for fi in movable:
            fx, fy = frog_pos[fi]
            dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            rng.shuffle(dirs)
            for dx, dy in dirs:
                nx, ny = fx + dx, fy + dy
                if _passable(grid, nx, ny) and (nx, ny) not in frog_set:
                    frog_pos[fi] = (nx, ny)
                    break
            else:
                continue
            break  # moved one frog, done for this scramble step

    return grid, frog_pos, targets, colors


# ---------------------------------------------------------------------------
# Level builder
# ---------------------------------------------------------------------------

def _build_level(grid: list[list[int]], targets: list[tuple[int, int]],
                 colors: list[int]) -> Level:
    size = len(grid)
    sprites: list[Sprite] = []

    # Lily pads and stone steps as background sprites
    for y in range(size):
        for x in range(size):
            if grid[y][x] == 1:  # lily pad
                sprites.append(
                    Sprite(pixels=_PAD_PIXELS, name=f"pad_{x}_{y}",
                           collidable=False, tags=["pad"], layer=0)
                    .set_position(x * CELL, y * CELL)
                )
            elif grid[y][x] == 2:  # stone step
                sprites.append(
                    Sprite(pixels=_STONE_PIXELS, name=f"stone_{x}_{y}",
                           collidable=False, tags=["stone"], layer=0)
                    .set_position(x * CELL, y * CELL)
                )

    # Target markers on lily pads (shown as colored diamond pads)
    for i, (tx, ty) in enumerate(targets):
        sprites.append(
            Sprite(pixels=_target_pixels(colors[i]), name=f"target_{i}",
                   collidable=False, tags=["target", f"target_color_{colors[i]}"], layer=1)
            .set_position(tx * CELL, ty * CELL)
        )

    return Level(sprites=sprites, grid_size=(size * CELL, size * CELL))


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class FrogCrossing(AugmentedGame):

    def __init__(self, seed: int | None = None) -> None:
        rng = random.Random(seed)
        self._step_counter = StepCounter(_STEP_BUDGETS[0])
        self._level_data: list[tuple[list[list[int]], list[tuple[int, int]],
                                     list[tuple[int, int]], list[int]]] = []
        levels: list[Level] = []

        for size, num_frogs, scramble, removal_fraction in _CONFIGS:
            grid, frog_pos, targets, colors = _generate(size, num_frogs, scramble, removal_fraction, rng)
            self._level_data.append((grid, frog_pos, targets, colors))
            levels.append(_build_level(grid, targets, colors))

        self._history: list[tuple[int, list[tuple[int, int]]]] = []
        self._active_frog: int = 0
        self._frog_colors: list[int] = []
        self._init_augmentation(seed)

        super().__init__(
            game_id="frog_crossing",
            levels=levels,
            camera=Camera(
                background=C_WATER,
                letter_box=C_BORDER,
                interfaces=[self._step_counter],
            ),
            available_actions=[1, 2, 3, 4, 5, 7],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_frogs(self) -> list[Sprite]:
        return self.current_level.get_sprites_by_tag("frog")

    def _frog_at(self, x: int, y: int) -> Sprite | None:
        return next((s for s in self._get_frogs() if s.x == x and s.y == y), None)

    def _pad_or_stone_at(self, x: int, y: int) -> bool:
        return any(
            s.x == x and s.y == y
            for s in self.current_level.get_sprites()
            if s.name.startswith(("pad_", "stone_"))
        )

    def _target_color_at(self, x: int, y: int) -> int | None:
        for s in self.current_level.get_sprites_by_tag("target"):
            if s.x == x and s.y == y:
                # extract color from tag
                for tag in s.tags:
                    if tag.startswith("target_color_"):
                        return int(tag.split("_")[2])
        return None

    def _snapshot(self) -> tuple[int, list[tuple[int, int]]]:
        return self._active_frog, [(s.x, s.y) for s in self._get_frogs()]

    def _highlight_active(self) -> None:
        frogs = self._get_frogs()
        for i, frog in enumerate(frogs):
            if i >= len(self._frog_colors):
                break
            active = i == self._active_frog
            frog.pixels = np.array(_frog_pixels(self._frog_colors[i], active), dtype=np.int8)

    def _check_win(self) -> bool:
        frogs = self._get_frogs()
        for i, frog in enumerate(frogs):
            tc = self._target_color_at(frog.x, frog.y)
            if tc != self._frog_colors[i]:
                return False
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_set_level(self, level: Level) -> None:
        # Clear any frog sprites left over from a previous set_level call
        for frog in level.get_sprites_by_tag("frog"):
            level.remove_sprite(frog)
        self._history.clear()
        self._active_frog = 0
        self._step_counter.reset(_STEP_BUDGETS[self._current_level_index])
        _, frog_pos, _, colors = self._level_data[self._current_level_index]
        self._frog_colors = list(colors)

        for i, (fx, fy) in enumerate(frog_pos):
            level.add_sprite(
                Sprite(pixels=_frog_pixels(colors[i]), name=f"frog_{i}",
                       collidable=False, tags=["frog", f"frog_{i}"], layer=2)
                .set_position(fx * CELL, fy * CELL)
            )
        self._highlight_active()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    _DELTAS: dict[int, tuple[int, int]] = {
        GameAction.ACTION1: (0,    -CELL),
        GameAction.ACTION2: (0,     CELL),
        GameAction.ACTION3: (-CELL, 0),
        GameAction.ACTION4: (CELL,  0),
    }

    def step(self) -> None:
        self._step_counter.decrement()
        _action_id = self.screen_action_to_game(self.action.id)

        # --- Undo -------------------------------------------------------
        if _action_id == GameAction.ACTION7:
            if self._history:
                prev_active, prev_pos = self._history.pop()
                self._active_frog = prev_active
                for frog, (px, py) in zip(self._get_frogs(), prev_pos):
                    frog.set_position(px, py)
                self._highlight_active()
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        # --- Cycle active frog -----------------------------------------
        if _action_id == GameAction.ACTION5:
            self._history.append(self._snapshot())
            self._active_frog = (self._active_frog + 1) % len(self._get_frogs())
            self._highlight_active()
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        # --- Hop -------------------------------------------------------
        delta = self._DELTAS.get(_action_id)
        if delta is None:
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        dx, dy = delta
        frogs = self._get_frogs()
        if not frogs:
            self.complete_action()
            return

        frog = frogs[self._active_frog]
        nx, ny = frog.x + dx, frog.y + dy

        # Must land on a pad or stone
        if not self._pad_or_stone_at(nx, ny):
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        # Must not be occupied
        if self._frog_at(nx, ny) is not None:
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        self._history.append(self._snapshot())
        frog.set_position(nx, ny)
        self._highlight_active()

        if self._check_win():
            self.next_level()
            self.complete_action()
            return

        if not self._step_counter.steps_remaining:
            self.lose()
        self.complete_action()
