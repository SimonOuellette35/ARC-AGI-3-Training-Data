"""
Potion Brewer
=============
Guide the brewer around the laboratory to pick up ingredient jars and add
them to the cauldron in exactly the order shown in the recipe. The brewer
can carry up to 2 jars. ACTION5 when adjacent to a jar picks it up; when
adjacent to the cauldron with a jar it adds it; otherwise it swaps the two
held jars (so the second jar becomes ready to add next). ACTION7 undoes the
last action.

Color key:
  4  (#333333) — lab floor (dark)
  0  (#FFFFFF) — bench / worktop (impassable, where jars sit)
  9  (#1E93FF) — brewer character (blue)
  8  (#F93C31) — red potion jar
  11 (#FFDC00) — yellow potion jar
  14 (#4FCC30) — green potion jar
  12 (#FF851B) — orange potion jar
  6  (#E36EF6) — purple potion jar
  7  (#E6E6E6) — cauldron (grey)
  3  (#4FCC30) — cauldron active (green, when brewer adjacent)
  5  (#000000) — letterbox border

Recipe shown as a sequence of colored cells on the border strip.

Actions:
  ACTION1 (↑) — move brewer up
  ACTION2 (↓) — move brewer down
  ACTION3 (←) — move brewer left
  ACTION4 (→) — move brewer right
  ACTION5     — pick up jar / add to cauldron / swap held jars
  ACTION7     — undo last action

Win condition : All recipe ingredients added to cauldron in correct order.
Lose condition: Step budget exhausted.
Levels        : 4 labs — 3-step recipe (easy) → 4 → 5 → 6-step recipe (expert).
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
C_FLOOR    = 4   # dark floor
C_BENCH    = 0   # white bench/worktop
C_BREWER   = 9   # blue brewer
C_CAULDRON = 7   # grey cauldron
C_BORDER   = 5   # black

_JAR_COLORS = [8, 11, 14, 12, 6, 2]  # red, yellow, green, orange, purple, pink

_STEP_BUDGETS = [100, 160, 240, 340, 480, 660, 880]

# ---------------------------------------------------------------------------
# Cell size: CELL=3 gives max viewport 51×36 for 17×12 grid.
# ---------------------------------------------------------------------------
CELL = 3

# ---------------------------------------------------------------------------
# Sprite pixel art  (all CELL×CELL = 3×3 pixels)
# ---------------------------------------------------------------------------

# Brewer: 3×3 person in apron
_BREWER_PIXELS = [
    [-1,       0,        C_BREWER],  # head (white) + arm
    [C_BREWER, C_BREWER, C_BREWER],  # apron body
    [C_BREWER, -1,       C_BREWER],  # legs
]

def _brewer_carrying_pixels(jar_color: int) -> list[list[int]]:
    """Brewer holding a jar — jar shown in arm."""
    return [
        [jar_color, 0,        C_BREWER],  # jar in hand + head
        [C_BREWER,  C_BREWER, C_BREWER],  # apron body
        [C_BREWER,  -1,       C_BREWER],  # legs
    ]

def _jar_pixels(c: int) -> list[list[int]]:
    """3×3 glass jar: cap + colored liquid + base."""
    return [
        [7,  7,  7],   # grey cap
        [c,  c,  c],   # colored liquid
        [c,  7,  c],   # liquid + glass base
    ]

# Cauldron: 3×3 big pot (already 3×3!)
_CAULDRON_PIXELS = [
    [5,          C_CAULDRON, 5         ],   # rim handles
    [C_CAULDRON, 0,          C_CAULDRON],   # opening (white inside)
    [C_CAULDRON, C_CAULDRON, C_CAULDRON],   # pot body
]

# Level configs: (grid_w, grid_h, recipe_len, num_colors)
_CONFIGS = [
    (11, 9,  3, 3),
    (13, 10, 4, 3),
    (15, 11, 5, 4),
    (17, 12, 6, 4),
    (19, 13, 7, 5),
    (21, 14, 8, 5),
    (21, 14, 9, 6),
]

_CARRY_LIMIT = 2


# ---------------------------------------------------------------------------
# HUD: step counter + recipe display
# ---------------------------------------------------------------------------

class BrewHUD(RenderableUserDisplay):
    def __init__(self, max_steps: int, recipe: list[int]) -> None:
        self.max_steps = max_steps
        self.steps_remaining = max_steps
        self.recipe = recipe
        self.added_count = 0

    def reset(self, max_steps: int, recipe: list[int]) -> None:
        self.max_steps = max_steps
        self.steps_remaining = max_steps
        self.recipe = recipe
        self.added_count = 0

    def decrement(self) -> None:
        if self.steps_remaining > 0:
            self.steps_remaining -= 1

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.max_steps == 0:
            return frame
        filled = round(64 * self.steps_remaining / self.max_steps)
        for x in range(64):
            frame[63, x] = 7 if x < filled else 4

        # Recipe shown on top border row
        for i, color in enumerate(self.recipe):
            if i < 32:
                c = color if i >= self.added_count else 4  # dim completed ones
                frame[0, i * 2] = c
                frame[0, i * 2 + 1] = c
        return frame


# ---------------------------------------------------------------------------
# Level generator
# ---------------------------------------------------------------------------

def _generate(grid_w: int, grid_h: int, recipe_len: int, num_colors: int,
              rng: random.Random) -> dict:
    """Generate lab with benches, jar positions, cauldron, and recipe."""
    # All floor initially, add border benches and interior bench rows
    grid = [[1] * grid_w for _ in range(grid_h)]  # 1=floor, 0=bench

    # Border = benches
    for i in range(grid_w):
        grid[0][i] = grid[grid_h - 1][i] = 0
    for i in range(grid_h):
        grid[i][0] = grid[i][grid_w - 1] = 0

    # Add some interior bench rows for variety
    bench_rows = [grid_h // 3, 2 * grid_h // 3]
    for br in bench_rows:
        for x in range(1, grid_w - 1):
            if rng.random() < 0.6:
                grid[br][x] = 0

    # Cauldron in center
    cx, cy = grid_w // 2, grid_h // 2
    grid[cy][cx] = 1  # make sure cauldron position is floor

    # Floor cells
    floor_cells = [(x, y) for y in range(grid_h) for x in range(grid_w) if grid[y][x] == 1]

    # Brewer starts at top-left open area
    brewer_start = (2, 2) if grid[2][2] == 1 else floor_cells[0]

    # Recipe: sequence of color indices
    colors = _JAR_COLORS[:num_colors]
    recipe = [rng.choice(range(num_colors)) for _ in range(recipe_len)]

    # Jar placement: place one jar per required color (with duplicates if needed)
    bench_cells = [(x, y) for y in range(grid_h) for x in range(grid_w) if grid[y][x] == 0
                   and x > 0 and x < grid_w - 1 and y > 0 and y < grid_h - 1]
    rng.shuffle(bench_cells)

    # Count needed jars per color
    needed: dict[int, int] = {}
    for ci in recipe:
        needed[ci] = needed.get(ci, 0) + 1

    jar_positions: list[tuple[int, int]] = []
    jar_colors: list[int] = []
    used_bench: set[tuple[int, int]] = set()

    for ci, count in needed.items():
        for _ in range(count):
            for bc in bench_cells:
                if bc not in used_bench:
                    jar_positions.append(bc)
                    jar_colors.append(ci)
                    used_bench.add(bc)
                    break

    return {
        "grid": grid,
        "grid_w": grid_w,
        "grid_h": grid_h,
        "brewer_start": brewer_start,
        "cauldron": (cx, cy),
        "jar_positions": jar_positions,
        "jar_colors": jar_colors,
        "colors": colors,
        "recipe": recipe,
    }


def _build_level(data: dict) -> Level:
    grid = data["grid"]
    grid_w, grid_h = data["grid_w"], data["grid_h"]
    cx, cy = data["cauldron"]
    sprites: list[Sprite] = []

    for y in range(grid_h):
        for x in range(grid_w):
            if grid[y][x] == 0:
                sprites.append(
                    Sprite(pixels=[[C_BENCH]*CELL]*CELL, name=f"bench_{x}_{y}",
                           collidable=True, tags=["bench"], layer=-1)
                    .set_position(x * CELL, y * CELL)
                )
            else:
                sprites.append(
                    Sprite(pixels=[[C_FLOOR]*CELL]*CELL, name=f"floor_{x}_{y}",
                           collidable=False, tags=["floor"], layer=0)
                    .set_position(x * CELL, y * CELL)
                )

    # Cauldron
    sprites.append(
        Sprite(pixels=_CAULDRON_PIXELS, name="cauldron",
               collidable=False, tags=["cauldron"], layer=1)
        .set_position(cx * CELL, cy * CELL)
    )

    return Level(sprites=sprites, grid_size=(grid_w * CELL, grid_h * CELL))


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class PotionBrewer(AugmentedGame):

    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: the base rotates the board on set_level
        rng = random.Random(seed)
        self._hud = BrewHUD(_STEP_BUDGETS[0], [])
        self._level_data: list[dict] = []
        levels: list[Level] = []

        for grid_w, grid_h, recipe_len, num_colors in _CONFIGS:
            data = _generate(grid_w, grid_h, recipe_len, num_colors, rng)
            self._level_data.append(data)
            levels.append(_build_level(data))

        self._held: list[int] = []      # color indices of held jars (max 2)
        self._added: int = 0            # how many recipe steps completed
        self._history: list[tuple[
            tuple[int, int],
            list[int],
            int,
            list[bool],  # jar visibility
        ]] = []

        super().__init__(
            game_id="potion_brewer",
            levels=levels,
            camera=Camera(
                background=C_FLOOR,
                letter_box=C_BORDER,
                interfaces=[self._hud],
            ),
            available_actions=[1, 2, 3, 4, 5, 7],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _brewer(self) -> Sprite:
        return self.current_level.get_sprites_by_tag("brewer")[0]

    def _jars(self) -> list[Sprite]:
        return self.current_level.get_sprites_by_tag("jar")

    def _cauldron(self) -> Sprite:
        return self.current_level.get_sprites_by_tag("cauldron")[0]

    def _jar_color(self, jar: Sprite) -> int:
        for tag in jar.tags:
            if tag.startswith("jar_color_"):
                return int(tag.split("_")[2])
        return -1

    def _bench_at(self, x: int, y: int) -> bool:
        return any("bench" in s.tags and s.x == x and s.y == y
                   for s in self.current_level.get_sprites())

    def _get_adjacent_jar(self) -> Sprite | None:
        brewer = self._brewer()
        for dx, dy in ((0, -CELL), (0, CELL), (-CELL, 0), (CELL, 0)):
            nx, ny = brewer.x + dx, brewer.y + dy
            for jar in self._jars():
                if jar.x == nx and jar.y == ny and jar.is_visible:
                    return jar
        return None

    def _cauldron_adjacent(self) -> bool:
        brewer = self._brewer()
        caul = self._cauldron()
        return abs(brewer.x - caul.x) + abs(brewer.y - caul.y) == CELL

    def _snapshot(self):
        brewer = self._brewer()
        jar_vis = [j.is_visible for j in self._jars()]
        return (brewer.x, brewer.y), list(self._held), self._added, jar_vis

    def _refresh_brewer(self) -> None:
        brewer = self._brewer()
        if self._held:
            brewer.pixels = np.array(_brewer_carrying_pixels(self._held[0]), dtype=np.int8)
        else:
            brewer.pixels = np.array(_BREWER_PIXELS, dtype=np.int8)

    def on_set_level(self, level: Level) -> None:
        # Seating a level twice re-runs this hook on the same persistent Level.
        clear_dynamic_sprites(level, tags=["brewer", "jar"])
        self._history.clear()
        self._held.clear()
        self._added = 0
        idx = self._current_level_index
        data = self._level_data[idx]
        self._hud.reset(_STEP_BUDGETS[idx], [data["colors"][i] for i in data["recipe"]])

        bx, by = data["brewer_start"]
        level.add_sprite(
            Sprite(pixels=_BREWER_PIXELS, name="brewer",
                   collidable=False, tags=["brewer"], layer=3)
            .set_position(bx * CELL, by * CELL)
        )

        for i, ((jx, jy), jc_idx) in enumerate(
            zip(data["jar_positions"], data["jar_colors"])
        ):
            jc = data["colors"][jc_idx]
            level.add_sprite(
                Sprite(pixels=_jar_pixels(jc), name=f"jar_{i}",
                       collidable=False,
                       tags=["jar", f"jar_{i}", f"jar_color_{jc}"],
                       layer=2)
                .set_position(jx * CELL, jy * CELL)
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
        self._hud.decrement()

        # --- Undo -------------------------------------------------------
        if self.action.id == GameAction.ACTION7:
            if self._history:
                (bx, by), held, added, jar_vis = self._history.pop()
                self._brewer().set_position(bx, by)
                self._held = list(held)
                self._added = added
                self._hud.added_count = added
                for jar, vis in zip(self._jars(), jar_vis):
                    jar.set_visible(vis)
                self._refresh_brewer()
            self.complete_action()
            if not self._hud.steps_remaining:
                self.lose()
            return

        # --- Interact (ACTION5) ----------------------------------------
        if self.action.id == GameAction.ACTION5:
            # Try to add to cauldron
            if self._held and self._cauldron_adjacent():
                data = self._level_data[self._current_level_index]
                recipe = data["recipe"]
                if self._added < len(recipe):
                    next_color_idx = recipe[self._added]
                    next_color = data["colors"][next_color_idx]
                    if self._held[0] == next_color:
                        self._history.append(self._snapshot())
                        self._held.pop(0)
                        self._added += 1
                        self._hud.added_count = self._added
                        self._refresh_brewer()
                        if self._added == len(recipe):
                            self.next_level()
                            self.complete_action()
                            return
                self.complete_action()
                if not self._hud.steps_remaining:
                    self.lose()
                return

            # Try to pick up jar
            jar = self._get_adjacent_jar()
            if jar is not None and len(self._held) < _CARRY_LIMIT:
                self._history.append(self._snapshot())
                color = self._jar_color(jar)
                self._held.append(color)
                jar.set_visible(False)
                self._refresh_brewer()
                self.complete_action()
                if not self._hud.steps_remaining:
                    self.lose()
                return

            # No cauldron or jar nearby: swap held jars
            if len(self._held) == 2:
                self._history.append(self._snapshot())
                self._held[0], self._held[1] = self._held[1], self._held[0]
                self._refresh_brewer()

            self.complete_action()
            if not self._hud.steps_remaining:
                self.lose()
            return

        # --- Move -------------------------------------------------------
        delta = self._DELTAS.get(self.action.id)
        if delta is None:
            self.complete_action()
            if not self._hud.steps_remaining:
                self.lose()
            return

        dx, dy = delta
        brewer = self._brewer()
        nx, ny = brewer.x + dx, brewer.y + dy

        if self._bench_at(nx, ny):
            self.complete_action()
            if not self._hud.steps_remaining:
                self.lose()
            return

        self._history.append(self._snapshot())
        brewer.set_position(nx, ny)

        if not self._hud.steps_remaining:
            self.lose()
        self.complete_action()
