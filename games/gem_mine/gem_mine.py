"""
Gem Mine
========
Guide the miner through a cave to collect gems one at a time and deposit
them in the mine cart at the depot. The miner can carry only one gem.
Pick up: ACTION5 adjacent to a gem. Deposit: ACTION5 at the depot while
carrying a gem. ACTION7 undoes the last move or pick/deposit.

Color key:
  4  (#333333) — tunnel floor (dark grey)
  0  (#FFFFFF) — wall (impassable, shown as white/light)
  9  (#1E93FF) — miner sprite (blue)
  8  (#F93C31) — ruby (red gem)
  1  (#1E93FF) — sapphire (blue gem)
  14 (#4FCC30) — emerald (green gem)
  11 (#FFDC00) — topaz (yellow gem)
  12 (#FF851B) — depot / mine cart (orange)
  7  (#E6E6E6) — miner carrying gem (lighter, gem shown as dot on border)
  5  (#000000) — letterbox border

Actions:
  ACTION1 (↑) — move miner up
  ACTION2 (↓) — move miner down
  ACTION3 (←) — move miner left
  ACTION4 (→) — move miner right
  ACTION5     — pick up gem (adjacent) / deposit gem (at depot)
  ACTION7     — undo last action

Win condition : All gems deposited in the cart (gem counter reaches 0).
Lose condition: Step budget exhausted.
Levels        : 4 puzzles with 3→4→5→6 gems in a cave maze.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
from utils.arc_game import AugmentedGame
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

from utils.arc_game import clear_dynamic_sprites

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
C_TUNNEL = 4   # dark grey tunnel
C_WALL   = 0   # white (shown as walls around dark tunnel)
C_MINER  = 9   # blue miner
C_DEPOT  = 12  # orange depot/cart
C_BORDER = 5   # black

_GEM_COLORS = [8, 1, 14, 11, 6, 3, 2, 7, 10]  # red ruby, blue sapphire, green emerald, yellow topaz, pink, teal, magenta, grey, cyan

_STEP_BUDGETS = [120, 200, 300, 600, 800, 1000, 1300]

# ---------------------------------------------------------------------------
# Cell size: CELL=3 gives max viewport 51×45 for 17×15 grid.
# ---------------------------------------------------------------------------
CELL = 3

# ---------------------------------------------------------------------------
# Sprite pixel art  (all CELL×CELL = 3×3 pixels)
# ---------------------------------------------------------------------------

# Miner: 3×3 person with helmet
_MINER_PIXELS = [
    [-1, C_MINER,  0     ],  # helmet (white top)
    [C_MINER, C_MINER, C_MINER],  # body
    [C_MINER, -1,  C_MINER],  # legs
]

def _miner_carrying_pixels(gem_color: int) -> list[list[int]]:
    """Miner carrying a gem — gem shown as dot in body center."""
    return [
        [-1,      C_MINER,   0        ],  # helmet
        [C_MINER, gem_color, C_MINER  ],  # body with gem dot
        [C_MINER, -1,        C_MINER  ],  # legs
    ]

def _gem_pixels(c: int) -> list[list[int]]:
    """3×3 crystal gem: diamond shape with bright center."""
    return [
        [-1, c,  c ],  # top facets
        [c,  c,  0 ],  # body + highlight (white spark)
        [-1, c, -1 ],  # bottom point
    ]

# Depot (mine cart): 3×3 cart shape
_DEPOT_PIXELS = [
    [C_DEPOT, C_DEPOT, C_DEPOT],  # cart top rim
    [C_DEPOT, 0,       C_DEPOT],  # cart interior (white)
    [5,       C_DEPOT, 5      ],  # wheels (black axle ends)
]

# Level configs: (width, height, num_gems, extra_branches)
_CONFIGS = [
    (11, 9,  3, 2),
    (13, 11, 4, 3),
    (15, 13, 5, 4),
    (17, 15, 6, 5),
    (19, 17, 7, 6),
    (21, 19, 8, 7),
    (21, 19, 9, 8),
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
# Cave generator (maze-like tunnels)
# ---------------------------------------------------------------------------

def _generate_cave(width: int, height: int, rng: random.Random) -> list[list[int]]:
    """Generate a cave with tunnels (0=wall, 1=tunnel) using DFS maze."""
    assert width % 2 == 1 and height % 2 == 1
    grid = [[0] * width for _ in range(height)]
    cols, rows = (width - 1) // 2, (height - 1) // 2

    def carve(cx: int, cy: int) -> None:
        grid[2 * cy + 1][2 * cx + 1] = 1
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        rng.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cols and 0 <= ny < rows and grid[2 * ny + 1][2 * nx + 1] == 0:
                grid[2 * cy + 1 + dy][2 * cx + 1 + dx] = 1
                carve(nx, ny)

    carve(0, 0)

    # Add extra connections (branches) to make it less linear
    for _ in range(20):
        y = rng.randrange(1, height - 1)
        x = rng.randrange(1, width - 1)
        grid[y][x] = 1

    return grid


def _tunnel_cells(cave: list[list[int]]) -> list[tuple[int, int]]:
    return [(x, y) for y in range(len(cave)) for x in range(len(cave[0])) if cave[y][x] == 1]


def _generate(width: int, height: int, num_gems: int, rng: random.Random):
    """Generate cave with miner start, depot, and gem positions."""
    cave = _generate_cave(width, height, rng)
    tunnels = _tunnel_cells(cave)
    rng.shuffle(tunnels)

    # Depot at tunnel cell far from start
    depot = tunnels[0]
    miner_start = tunnels[1]
    gem_positions = tunnels[2: 2 + num_gems]
    gem_colors = _GEM_COLORS[:num_gems]

    return cave, miner_start, depot, gem_positions, gem_colors


def _build_level(cave: list[list[int]], depot: tuple[int, int]) -> Level:
    height, width = len(cave), len(cave[0])
    sprites: list[Sprite] = []

    for y in range(height):
        for x in range(width):
            if cave[y][x] == 1:
                sprites.append(
                    Sprite(pixels=[[C_TUNNEL]*CELL]*CELL, name=f"tunnel_{x}_{y}",
                           collidable=False, tags=["floor"], layer=0)
                    .set_position(x * CELL, y * CELL)
                )
            else:
                sprites.append(
                    Sprite(pixels=[[C_WALL]*CELL]*CELL, name=f"wall_{x}_{y}",
                           collidable=True, tags=["wall"], layer=-1)
                    .set_position(x * CELL, y * CELL)
                )

    # Depot sprite (mine cart)
    dx, dy = depot
    sprites.append(
        Sprite(pixels=_DEPOT_PIXELS, name="depot",
               collidable=False, tags=["depot"], layer=1)
        .set_position(dx * CELL, dy * CELL)
    )

    return Level(sprites=sprites, grid_size=(width * CELL, height * CELL))


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class GemMine(AugmentedGame):

    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: the base draws this level's rotation
        rng = random.Random(seed)
        self._step_counter = StepCounter(_STEP_BUDGETS[0])
        self._level_data: list[tuple[
            list[list[int]],
            tuple[int, int],
            tuple[int, int],
            list[tuple[int, int]],
            list[int],
        ]] = []
        levels: list[Level] = []

        for width, height, num_gems, _ in _CONFIGS:
            cave, miner_start, depot, gem_pos, gem_colors = _generate(
                width, height, num_gems, rng
            )
            self._level_data.append((cave, miner_start, depot, gem_pos, gem_colors))
            levels.append(_build_level(cave, depot))

        self._history: list[tuple[
            tuple[int, int],
            int | None,  # carried gem color or None
            list[tuple[int, int]],  # gem positions (None if deposited)
        ]] = []
        self._carried_gem: int | None = None  # color of carried gem, or None

        super().__init__(
            game_id="gem_mine",
            levels=levels,
            camera=Camera(
                background=C_WALL,
                letter_box=C_BORDER,
                interfaces=[self._step_counter],
            ),
            available_actions=[1, 2, 3, 4, 5, 7],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _miner(self) -> Sprite:
        return self.current_level.get_sprites_by_tag("miner")[0]

    def _gems(self) -> list[Sprite]:
        return self.current_level.get_sprites_by_tag("gem")

    def _gem_at(self, x: int, y: int) -> Sprite | None:
        return next((s for s in self._gems() if s.x == x and s.y == y), None)

    def _depot(self) -> Sprite:
        return self.current_level.get_sprites_by_tag("depot")[0]

    def _wall_at(self, x: int, y: int) -> bool:
        return any(s.x == x and s.y == y and "wall" in s.tags
                   for s in self.current_level.get_sprites())

    def _snapshot(self) -> tuple[tuple[int, int], int | None, list[tuple[int, int] | None]]:
        m = self._miner()
        gem_positions: list[tuple[int, int] | None] = []
        for g in self._gems():
            gem_positions.append((g.x, g.y) if g.is_visible else None)
        return (m.x, m.y), self._carried_gem, gem_positions

    def _refresh_miner(self) -> None:
        """Update miner sprite to reflect carrying state."""
        miner = self._miner()
        if self._carried_gem is not None:
            miner.pixels = np.array(_miner_carrying_pixels(self._carried_gem), dtype=np.int8)
        else:
            miner.pixels = np.array(_MINER_PIXELS, dtype=np.int8)

    def on_set_level(self, level: Level) -> None:
        # Seating a level twice re-runs this hook on the same persistent Level.
        clear_dynamic_sprites(level, tags=["miner", "gem"])
        self._history.clear()
        self._carried_gem = None
        self._step_counter.reset(_STEP_BUDGETS[self._current_level_index])
        _, miner_start, _, gem_pos, gem_colors = self._level_data[self._current_level_index]

        mx, my = miner_start
        level.add_sprite(
            Sprite(pixels=_MINER_PIXELS, name="miner",
                   collidable=False, tags=["miner"], layer=3)
            .set_position(mx * CELL, my * CELL)
        )

        for i, ((gx, gy), gc) in enumerate(zip(gem_pos, gem_colors)):
            level.add_sprite(
                Sprite(pixels=_gem_pixels(gc), name=f"gem_{i}",
                       collidable=False, tags=["gem", f"gem_{i}"], layer=2)
                .set_position(gx * CELL, gy * CELL)
            )

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

        # --- Undo -------------------------------------------------------
        if self.action.id == GameAction.ACTION7:
            if self._history:
                (mx, my), carried, gem_states = self._history.pop()
                self._miner().set_position(mx, my)
                self._carried_gem = carried
                gems = self._gems()
                for gem, state in zip(gems, gem_states):
                    if state is None:
                        gem.set_visible(False)
                    else:
                        gx, gy = state
                        gem.set_position(gx, gy)
                        gem.set_visible(True)
                self._refresh_miner()
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        # --- Interact (pick up / deposit) --------------------------------
        if self.action.id == GameAction.ACTION5:
            miner = self._miner()
            depot = self._depot()

            # Deposit: miner on depot with gem
            if miner.x == depot.x and miner.y == depot.y and self._carried_gem is not None:
                self._history.append(self._snapshot())
                self._carried_gem = None
                self._refresh_miner()
                # Check win
                remaining = [g for g in self._gems() if g.is_visible]
                if not remaining and self._carried_gem is None:
                    self.next_level()
                    self.complete_action()
                    return
                self.complete_action()
                if not self._step_counter.steps_remaining:
                    self.lose()
                return

            # Pick up: adjacent gem, not already carrying
            if self._carried_gem is None:
                for dx, dy in ((0, -CELL), (0, CELL), (-CELL, 0), (CELL, 0)):
                    gem = self._gem_at(miner.x + dx, miner.y + dy)
                    if gem is not None and gem.is_visible:
                        self._history.append(self._snapshot())
                        gem_color = int(gem.pixels[1, 0])  # row1,col0 = body color
                        gem.set_visible(False)
                        self._carried_gem = gem_color
                        self._refresh_miner()
                        self.complete_action()
                        if not self._step_counter.steps_remaining:
                            self.lose()
                        return

            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        # --- Move -------------------------------------------------------
        delta = self._DELTAS.get(self.action.id)
        if delta is None:
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        dx, dy = delta
        miner = self._miner()
        nx, ny = miner.x + dx, miner.y + dy

        if self._wall_at(nx, ny):
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        self._history.append(self._snapshot())
        miner.set_position(nx, ny)

        if not self._step_counter.steps_remaining:
            self.lose()
        self.complete_action()
