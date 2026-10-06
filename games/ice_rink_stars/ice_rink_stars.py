"""
Ice Rink Stars
==============
Launch a skater across a slippery ice rink to collect all gold stars.
The skater slides in a given direction (ACTION1-4) until hitting a wall
board or a placed ice block. Stars are collected automatically as the
skater slides through them. ACTION5 places a temporary ice block at the
skater's current resting position (limited budget). The red cursor marker
shows where the next block will land. ACTION7 undoes the last launch or
removes the last-placed block.

Color key:
  1  (#1E93FF) — ice floor (blue)
  4  (#333333) — wall board (dark, impassable)
  9  (#1E93FF) — skater (bright blue)
  11 (#FFDC00) — star collectible (yellow)
  8  (#F93C31) — cursor / block-placement marker (red)
  0  (#FFFFFF) — placed ice block (white)
  12 (#FF851B) — block budget indicator (orange)
  5  (#000000) — letterbox border

Actions:
  ACTION1 (↑) — launch skater upward (slides until wall)
  ACTION2 (↓) — launch skater downward
  ACTION3 (←) — launch skater left
  ACTION4 (→) — launch skater right
  ACTION5     — place ice block at skater's current position
  ACTION7     — undo last launch or remove last block

Win condition : All stars collected.
Lose condition: Step budget exhausted OR block budget exhausted before win.
Levels        : 4 rinks with increasing star count and shrinking block budget.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
from utils.arc_game import AugmentedGame
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
C_ICE    = 1   # blue ice floor
C_WALL   = 4   # dark wall/board
C_SKATER = 9   # bright blue skater
C_STAR   = 11  # yellow star
C_CURSOR = 8   # red cursor
C_BLOCK  = 0   # white ice block
C_BORDER = 5   # black

_STEP_BUDGETS  = [100, 180, 280, 400, 560, 750, 1000]
_BLOCK_BUDGETS = [6,   5,   4,   3,   2,   2,   1  ]

# ---------------------------------------------------------------------------
# Cell size: CELL=3 gives max viewport 54×48 for 18×16 grid.
# ---------------------------------------------------------------------------
CELL = 3

# ---------------------------------------------------------------------------
# Sprite pixel art  (all CELL×CELL = 3×3 pixels)
# ---------------------------------------------------------------------------

# Skater: 3×3 figure with white skates
_SKATER_PIXELS = [
    [-1,      0,        C_SKATER],  # arm raised + head (white)
    [C_SKATER, C_SKATER, C_SKATER],  # body
    [0,       C_SKATER, 0       ],  # skate blades (white)
]

# Star: 3×3 X-pattern (sparkle shape)
_STAR_PIXELS = [
    [C_STAR, -1,     C_STAR],  # top points
    [-1,     C_STAR, -1    ],  # center
    [C_STAR, -1,     C_STAR],  # bottom points
]

# Cursor: 3×3 cross
_CURSOR_PIXELS = [
    [-1,      C_CURSOR, -1      ],
    [C_CURSOR, -1,      C_CURSOR],
    [-1,      C_CURSOR, -1      ],
]

# Level configs: (width, height, num_stars, wall_density)
_CONFIGS = [
    (12, 10, 5, 0.10),
    (14, 12, 7, 0.12),
    (16, 14, 9, 0.14),
    (18, 16, 11, 0.15),
    (20, 18, 13, 0.17),
    (20, 18, 15, 0.18),
    (21, 19, 17, 0.20),
]


# ---------------------------------------------------------------------------
# HUD: step counter + block budget
# ---------------------------------------------------------------------------

class StepCounter(RenderableUserDisplay):
    def __init__(self, max_steps: int, block_budget: int) -> None:
        self.max_steps = max_steps
        self.steps_remaining = max_steps
        self.blocks_remaining = block_budget
        self.max_blocks = block_budget

    def reset(self, max_steps: int, block_budget: int) -> None:
        self.max_steps = max_steps
        self.steps_remaining = max_steps
        self.blocks_remaining = block_budget
        self.max_blocks = block_budget

    def decrement_steps(self) -> None:
        if self.steps_remaining > 0:
            self.steps_remaining -= 1

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.max_steps == 0:
            return frame
        filled = round(62 * self.steps_remaining / self.max_steps)
        for x in range(62):
            frame[63, x] = 7 if x < filled else 4
        # Block budget indicator on right side
        for b in range(self.max_blocks):
            frame[63, 62 + b % 2] = C_SKATER if b < self.blocks_remaining else 4
        return frame


# ---------------------------------------------------------------------------
# Level generator
# ---------------------------------------------------------------------------

def _is_solvable(walls: set[tuple[int, int]], stars: list[tuple[int, int]],
                  skater: tuple[int, int], width: int, height: int) -> bool:
    """Check if all stars can be collected by sliding without block placements."""
    n = len(stars)
    all_mask = (1 << n) - 1

    def slide(sx: int, sy: int, dx: int, dy: int, mask: int):
        x, y = sx, sy
        while True:
            nx, ny = x + dx, y + dy
            if (nx, ny) in walls or not (0 <= nx < width and 0 <= ny < height):
                break
            x, y = nx, ny
            for i, (stx, sty) in enumerate(stars):
                if x == stx and y == sty:
                    mask |= (1 << i)
        return x, y, mask

    start = (*skater, 0)
    seen: set[tuple[int, int, int]] = {start}
    q = deque([start])
    while q:
        sx, sy, mask = q.popleft()
        if mask == all_mask:
            return True
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny, new_mask = slide(sx, sy, dx, dy, mask)
            ns = (nx, ny, new_mask)
            if ns not in seen:
                seen.add(ns)
                q.append(ns)
    return False


def _generate(width: int, height: int, num_stars: int, wall_density: float,
              rng: random.Random):
    """Return (wall_set, star_positions, skater_start) — guaranteed solvable.

    Stars are placed anywhere on non-wall ice (including edge cells) so that
    the skater can reach them by sliding against walls.  A diversity check
    ensures stars span at least 2 different rows AND 2 different columns, so
    no level reduces to a trivial single-row sweep.
    """

    def _is_diverse(positions: list[tuple[int, int]]) -> bool:
        """True if stars are spread across ≥2 rows AND ≥2 columns."""
        rows = {y for _, y in positions}
        cols = {x for x, _ in positions}
        return len(rows) >= 2 and len(cols) >= 2

    for _ in range(2000):
        walls: set[tuple[int, int]] = set()

        # Border walls
        for x in range(width):
            walls.add((x, 0))
            walls.add((x, height - 1))
        for y in range(height):
            walls.add((0, y))
            walls.add((width - 1, y))

        # Interior walls (sparse)
        interior = [(x, y) for y in range(2, height - 2) for x in range(2, width - 2)]
        rng.shuffle(interior)
        n_walls = int(len(interior) * wall_density)
        for (x, y) in interior[:n_walls]:
            walls.add((x, y))

        # All ice cells (stars can be anywhere — edge cells are valid stops)
        ice = [(x, y) for y in range(height) for x in range(width) if (x, y) not in walls]
        rng.shuffle(ice)

        if len(ice) < num_stars + 1:
            continue

        # Try a few random star placements until we find a diverse one
        found = False
        for attempt in range(20):
            rng.shuffle(ice)
            star_positions = ice[:num_stars]
            if not _is_diverse(star_positions):
                continue
            star_set = set(star_positions)
            skater_candidates = [c for c in ice if c not in star_set]
            if not skater_candidates:
                continue
            skater = skater_candidates[0]
            if _is_solvable(walls, star_positions, skater, width, height):
                found = True
                break
        if found:
            return walls, star_positions, skater

    # Fallback: open rink with stars distributed across multiple rows/cols.
    # Grid-pattern placement ensures spread and trivial solvability.
    fb_walls: set[tuple[int, int]] = set()
    for x in range(width):
        fb_walls.add((x, 0))
        fb_walls.add((x, height - 1))
    for y in range(height):
        fb_walls.add((0, y))
        fb_walls.add((width - 1, y))

    # Place stars in a rough grid pattern spread across the rink interior
    cols_avail = list(range(2, width - 2))
    rows_avail = list(range(2, height - 2))
    rng.shuffle(cols_avail)
    rng.shuffle(rows_avail)
    fb_stars: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for i in range(num_stars):
        cx = cols_avail[i % len(cols_avail)]
        cy = rows_avail[(i * 3) % len(rows_avail)]
        # Avoid duplicates
        while (cx, cy) in seen or (cx, cy) in fb_walls:
            cy = rows_avail[(rows_avail.index(cy) + 1) % len(rows_avail)]
        fb_stars.append((cx, cy))
        seen.add((cx, cy))

    # Skater starts at a corner away from stars
    fb_skater = (1, 1)
    if (1, 1) in seen:
        fb_skater = (width - 2, height - 2)
    return fb_walls, fb_stars, fb_skater


def _build_level(walls: set[tuple[int, int]], width: int, height: int) -> Level:
    sprites: list[Sprite] = []
    for y in range(height):
        for x in range(width):
            if (x, y) in walls:
                sprites.append(
                    Sprite(pixels=[[C_WALL]*CELL]*CELL, name=f"wall_{x}_{y}",
                           collidable=True, tags=["wall"], layer=-1)
                    .set_position(x * CELL, y * CELL)
                )
            else:
                sprites.append(
                    Sprite(pixels=[[C_ICE]*CELL]*CELL, name=f"ice_{x}_{y}",
                           collidable=False, tags=["floor"], layer=0)
                    .set_position(x * CELL, y * CELL)
                )
    return Level(sprites=sprites, grid_size=(width * CELL, height * CELL))


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class IceRinkStars(AugmentedGame):

    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: the base draws this level's rotation
        rng = random.Random(seed)
        self._step_counter = StepCounter(_STEP_BUDGETS[0], _BLOCK_BUDGETS[0])
        self._level_data: list[tuple[set[tuple[int, int]], list[tuple[int, int]],
                                     tuple[int, int], int, int]] = []
        levels: list[Level] = []

        for width, height, num_stars, wall_density in _CONFIGS:
            walls, stars, skater = _generate(width, height, num_stars, wall_density, rng)
            self._level_data.append((walls, stars, skater, width, height))
            levels.append(_build_level(walls, width, height))

        # History: each entry = (skater_pos, stars_collected_set, placed_blocks, cursor_pos)
        self._history: list[tuple[
            tuple[int, int],
            set[int],
            list[tuple[int, int]],
            tuple[int, int],
        ]] = []
        self._placed_blocks: list[tuple[int, int]] = []
        self._collected_stars: set[int] = set()
        self._cursor: tuple[int, int] = (0, 0)

        super().__init__(
            game_id="ice_rink_stars",
            levels=levels,
            camera=Camera(
                background=C_ICE,
                letter_box=C_BORDER,
                interfaces=[self._step_counter],
            ),
            available_actions=[1, 2, 3, 4, 5, 7],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _skater(self) -> Sprite:
        return self.current_level.get_sprites_by_tag("skater")[0]

    def _stars(self) -> list[Sprite]:
        return self.current_level.get_sprites_by_tag("star")

    def _cursor_sprite(self) -> Sprite:
        return self.current_level.get_sprites_by_tag("cursor")[0]

    def _is_wall(self, px: int, py: int) -> bool:
        """Check if pixel position (px, py) is a wall. Walls stored in grid coords."""
        walls, _, _, width, height = self._level_data[self._current_level_index]
        gx, gy = px // CELL, py // CELL
        if not (0 <= gx < width and 0 <= gy < height):
            return True
        if (gx, gy) in walls:
            return True
        if (px, py) in self._placed_blocks:  # placed_blocks stored in pixel coords
            return True
        return False

    def _snapshot(self):
        skater = self._skater()
        return (
            (skater.x, skater.y),
            set(self._collected_stars),
            list(self._placed_blocks),
            self._cursor,
        )

    def _slide(self, dx: int, dy: int) -> tuple[tuple[int, int], list[int]]:
        """Slide skater, return (final_pos, list_of_collected_star_indices)."""
        skater = self._skater()
        x, y = skater.x, skater.y
        collected_indices: list[int] = []
        stars = self._stars()

        while True:
            nx, ny = x + dx, y + dy
            if self._is_wall(nx, ny):
                break
            x, y = nx, ny
            # Collect any star here
            for i, star in enumerate(stars):
                if star.is_visible and star.x == x and star.y == y and i not in self._collected_stars:
                    collected_indices.append(i)

        return (x, y), collected_indices

    def on_set_level(self, level: Level) -> None:
        self._history.clear()
        self._placed_blocks.clear()
        self._collected_stars.clear()
        # Clear dynamic sprites from previous call
        for s in list(level.get_sprites()):
            if any(t in s.tags for t in ("skater", "star", "cursor", "block")):
                level.remove_sprite(s)
        idx = self._current_level_index
        _, stars, skater, width, height = self._level_data[idx]
        self._step_counter.reset(_STEP_BUDGETS[idx], _BLOCK_BUDGETS[idx])
        # Cursor marker starts at skater's initial position
        self._cursor = (skater[0] * CELL, skater[1] * CELL)

        sx, sy = skater
        level.add_sprite(
            Sprite(pixels=_SKATER_PIXELS, name="skater",
                   collidable=False, tags=["skater"], layer=3)
            .set_position(sx * CELL, sy * CELL)
        )

        for i, (stx, sty) in enumerate(stars):
            level.add_sprite(
                Sprite(pixels=_STAR_PIXELS, name=f"star_{i}",
                       collidable=False, tags=["star", f"star_{i}"], layer=2)
                .set_position(stx * CELL, sty * CELL)
            )

        cx, cy = self._cursor
        level.add_sprite(
            Sprite(pixels=_CURSOR_PIXELS, name="cursor",
                   collidable=False, tags=["cursor"], layer=4)
            .set_position(cx, cy)
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
        self._step_counter.decrement_steps()

        # --- Undo -------------------------------------------------------
        if self.action.id == GameAction.ACTION7:
            if self._history:
                (sx, sy), prev_collected, prev_blocks, prev_cursor = self._history.pop()
                self._skater().set_position(sx, sy)
                # Restore stars
                removed = self._collected_stars - prev_collected
                for i in removed:
                    self._stars()[i].set_visible(True)
                self._collected_stars = prev_collected
                # Restore blocks
                if len(prev_blocks) < len(self._placed_blocks):
                    removed_block = self._placed_blocks[-1]
                    # Remove block sprite
                    for s in list(self.current_level.get_sprites()):
                        if s.name == f"block_{removed_block[0]}_{removed_block[1]}":
                            self.current_level.remove_sprite(s)
                            break
                    self._step_counter.blocks_remaining += 1
                self._placed_blocks = list(prev_blocks)
                self._cursor = prev_cursor
                self._cursor_sprite().set_position(*self._cursor)
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        # --- Place block at skater's position --------------------------
        if self.action.id == GameAction.ACTION5:
            skater = self._skater()
            px, py = skater.x, skater.y
            if (self._step_counter.blocks_remaining > 0
                    and not self._is_wall(px, py)):
                self._history.append(self._snapshot())
                self._placed_blocks.append((px, py))
                self._cursor = (px, py)
                self._cursor_sprite().set_position(px, py)
                self._step_counter.blocks_remaining -= 1
                self.current_level.add_sprite(
                    Sprite(pixels=[[C_BLOCK]*CELL]*CELL, name=f"block_{px}_{py}",
                           collidable=False, tags=["block"], layer=1)
                    .set_position(px, py)
                )
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        # --- Launch skater ---------------------------------------------
        delta = self._DELTAS.get(self.action.id)
        if delta is None:
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        dx, dy = delta
        snap = self._snapshot()
        (fx, fy), collected_indices = self._slide(dx, dy)
        skater = self._skater()

        if fx == skater.x and fy == skater.y and not collected_indices:
            # Didn't move
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        self._history.append(snap)
        skater.set_position(fx, fy)
        # Sync cursor marker to skater's new resting position
        self._cursor = (fx, fy)
        self._cursor_sprite().set_position(fx, fy)

        for i in collected_indices:
            self._stars()[i].set_visible(False)
            self._collected_stars.add(i)

        # Check win
        if all(not s.is_visible for s in self._stars()):
            self.next_level()
            self.complete_action()
            return

        if not self._step_counter.steps_remaining:
            self.lose()
        self.complete_action()
