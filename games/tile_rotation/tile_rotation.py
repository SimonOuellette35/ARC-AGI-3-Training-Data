"""
Tile Rotation Matching
======================
Two columns of colored tiles are shown side by side. The LEFT column
shows the reference (target) orientations — these tiles are fixed and
cannot be moved. The RIGHT column starts with each tile randomly rotated.

Your task: rotate every right-column tile until it matches the
corresponding left-column tile exactly.

Actions:
  ACTION1 (↑) — move cursor up to previous tile
  ACTION2 (↓) — move cursor down to next tile
  ACTION3 (←) — rotate selected right tile counter-clockwise (90°)
  ACTION4 (→) — rotate selected right tile clockwise (90°)
  ACTION7     — undo last rotation

Color key:
  0  (#FFFFFF) — background
  5  (#000000) — separator line / letterbox
  4  (#333333) — cursor indicator
  Each tile pair uses a unique color.

Win condition : All right-column tiles match their left-column references.
Lose condition: Step budget exhausted (progress bar at bottom).
Levels        : 7 levels with increasing tile counts (3 → 10).
"""

from __future__ import annotations

import random

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
# Color constants  (all unique)
# ---------------------------------------------------------------------------
C_BG       = 0   # white background
C_SEP      = 4   # separator / cursor border
C_BORDER   = 5   # black letterbox

# Tile colors — one per tile type (up to 10 tiles)
TILE_COLORS = [8, 9, 11, 12, 14, 6, 3, 2, 10, 1]

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
CELL = 3          # each tile is CELL×CELL pixels
CAM_W = 32        # camera width in pixels (constant across levels)
LEFT_X  = 2 * CELL   # pixel x of left column (=6)
RIGHT_X = 7 * CELL   # pixel x of right column (=21)
SEP_X   = 5 * CELL   # pixel x of separator (=15)
CURSOR_X = 5 * CELL + 1  # pixel x of cursor dot (=16)
ROW_START = CELL     # first tile y position
ROW_STEP  = CELL + 1 # pixels between tile rows (small gap)

# ---------------------------------------------------------------------------
# Asymmetric tile pixel art (CELL×CELL = 3×3)
# Each pattern is rotationally asymmetric so 4 rotations are distinguishable.
# Color placeholder = 99 (replaced with actual tile color when building sprite)
# ---------------------------------------------------------------------------
_C = 99  # placeholder color
_D = -1  # transparent
_TILE_PATTERNS = [
    [
        [_C, _C, _D],
        [_C, _D, _D],
        [_C, _C, _C],
    ],
    [
        [_C, _D, _C],
        [_C, _D, _D],
        [_C, _C, _D],
    ],
    [
        [_D, _C, _C],
        [_C, _D, _D],
        [_C, _C, _D],
    ],
    [
        [_C, _C, _D],
        [_D, _C, _D],
        [_D, _C, _C],
    ],
    [
        [_C, _D, _C],
        [_D, _D, _C],
        [_C, _C, _D],
    ],
    [
        [_D, _C, _D],
        [_C, _D, _C],
        [_C, _C, _D],
    ],
    [
        [_C, _C, _D],
        [_D, _D, _C],
        [_C, _D, _C],
    ],
    [
        [_C, _D, _D],
        [_D, _C, _C],
        [_C, _C, _D],
    ],
    [
        [_D, _C, _C],
        [_C, _D, _C],
        [_C, _D, _D],
    ],
    [
        [_C, _D, _C],
        [_C, _C, _D],
        [_D, _C, _D],
    ],
]

# Step budgets (generous to allow full exploration)
_STEP_BUDGETS = [40, 60, 80, 100, 130, 160, 200]

# Tile counts per level
_TILE_COUNTS = [3, 4, 5, 6, 7, 8, 10]


# ---------------------------------------------------------------------------
# Step counter HUD
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
# Tile sprite factory
# ---------------------------------------------------------------------------

def _rotate_pattern(pattern: list[list[int]]) -> list[list[int]]:
    n = len(pattern)
    return [[pattern[n - 1 - r][c] for r in range(n)] for c in range(n)]


def _has_4_distinct_rotations(pattern: list[list[int]]) -> bool:
    seen: set[tuple[tuple[int, ...], ...]] = set()
    cur = [row[:] for row in pattern]
    for _ in range(4):
        key = tuple(tuple(row) for row in cur)
        seen.add(key)
        cur = _rotate_pattern(cur)
    return len(seen) == 4


_ORIENTATION_DISTINCT_PATTERNS = [
    pattern for pattern in _TILE_PATTERNS if _has_4_distinct_rotations(pattern)
]


def _tile_pixels(color: int, pattern: list[list[int]]) -> list[list[int]]:
    """Return a CELL×CELL pixel array using `color` as the tile color."""
    return [[color if p == _C else p for p in row] for row in pattern]


def _make_tile_sprite(name: str, color: int, pattern: list[list[int]], rotation: int,
                      px: int, py: int, tags: list[str]) -> Sprite:
    s = Sprite(
        pixels=_tile_pixels(color, pattern),
        name=name,
        collidable=False,
        tags=tags,
        layer=1,
    )
    s.set_position(px, py)
    s.set_rotation(rotation)
    return s


def _make_cursor(py: int) -> Sprite:
    """Thin 1×CELL cursor indicator beside the right column."""
    return Sprite(
        pixels=[[C_SEP]] * CELL,
        name="cursor",
        collidable=False,
        tags=["cursor"],
        layer=2,
    ).set_position(CURSOR_X - 2, py)


def _make_sep(cam_h: int) -> Sprite:
    """Vertical separator line between columns."""
    return Sprite(
        pixels=[[C_SEP]] * cam_h,
        name="separator",
        collidable=False,
        layer=-1,
    ).set_position(SEP_X, 0)


def _build_level(n_tiles: int) -> Level:
    """Build an empty level with just the separator. Tiles added in on_set_level."""
    cam_h = ROW_START + n_tiles * ROW_STEP + 2
    return Level(sprites=[_make_sep(cam_h)], grid_size=(CAM_W, cam_h))


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class TileRotation(AugmentedGame):

    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: the base rotates the board on set_level
        self._rng = random.Random(seed)
        self._step_counter = StepCounter(_STEP_BUDGETS[0])

        # Pre-generate all level data: (pattern, ref_rotations, start_rotations)
        self._level_data: list[tuple[list[list[int]], list[int], list[int]]] = []
        levels: list[Level] = []

        for n in _TILE_COUNTS:
            pattern = self._rng.choice(_ORIENTATION_DISTINCT_PATTERNS)
            ref_rots = [self._rng.choice([0, 90, 180, 270]) for _ in range(n)]
            # Start rotations: each tile has a random rotation ≠ reference
            start_rots = []
            for ref in ref_rots:
                choices = [r for r in [0, 90, 180, 270] if r != ref]
                start_rots.append(self._rng.choice(choices))
            self._level_data.append((pattern, ref_rots, start_rots))
            levels.append(_build_level(n))

        self._history: list[tuple[int, list[int]]] = []  # (cursor, rotations)
        self._cursor: int = 0
        self._current_rots: list[int] = []

        super().__init__(
            game_id="tr87",
            levels=levels,
            camera=Camera(
                background=C_BG,
                letter_box=C_BORDER,
                interfaces=[self._step_counter],
            ),
            available_actions=[1, 2, 3, 4, 7],
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_set_level(self, level: Level) -> None:
        self._history.clear()
        idx = self._current_level_index
        self._step_counter.reset(_STEP_BUDGETS[idx])

        # Clear tiles from previous call
        for s in level.get_sprites():
            if "tile" in s.tags or s.name == "cursor":
                level.remove_sprite(s)

        pattern, ref_rots, start_rots = self._level_data[idx]
        self._current_rots = list(start_rots)
        self._cursor = 0
        n = len(ref_rots)

        for i in range(n):
            py = ROW_START + i * ROW_STEP
            color = TILE_COLORS[i % len(TILE_COLORS)]
            # Left reference tile
            level.add_sprite(_make_tile_sprite(
                f"ref_{i}", color, pattern, ref_rots[i], LEFT_X, py, ["tile", "ref"]
            ))
            # Right adjustable tile
            level.add_sprite(_make_tile_sprite(
                f"adj_{i}", color, pattern, start_rots[i], RIGHT_X, py, ["tile", "adj"]
            ))

        # Cursor indicator
        level.add_sprite(_make_cursor(ROW_START))

    def _get_adj_tile(self, idx: int) -> Sprite:
        return self.current_level.get_sprites_by_name(f"adj_{idx}")[0]

    def _get_cursor_sprite(self) -> Sprite:
        return self.current_level.get_sprites_by_tag("cursor")[0]

    def _update_cursor(self) -> None:
        cur = self._get_cursor_sprite()
        cur.set_position(CURSOR_X - 2, ROW_START + self._cursor * ROW_STEP)

    def _snapshot(self) -> tuple[int, list[int]]:
        return (self._cursor, list(self._current_rots))

    def _restore(self, snap: tuple[int, list[int]]) -> None:
        self._cursor, rots = snap
        self._current_rots = list(rots)
        # Update all adj tile rotations
        idx = self._current_level_index
        _, ref_rots, _ = self._level_data[idx]
        n = len(ref_rots)
        for i in range(n):
            tile = self._get_adj_tile(i)
            tile.set_rotation(self._current_rots[i])
        self._update_cursor()

    def _check_win(self) -> bool:
        idx = self._current_level_index
        _, ref_rots, _ = self._level_data[idx]
        return all(self._current_rots[i] == ref_rots[i] for i in range(len(ref_rots)))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> None:
        self._step_counter.decrement()
        action = self.action.id
        n = len(self._current_rots)

        if action == GameAction.ACTION7:
            if self._history:
                self._restore(self._history.pop())
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        snap = self._snapshot()

        if action == GameAction.ACTION1:  # cursor up
            self._cursor = (self._cursor - 1) % n
            self._update_cursor()

        elif action == GameAction.ACTION2:  # cursor down
            self._cursor = (self._cursor + 1) % n
            self._update_cursor()

        elif action == GameAction.ACTION3:  # rotate CCW
            self._history.append(snap)
            old = self._current_rots[self._cursor]
            new = (old - 90) % 360
            self._current_rots[self._cursor] = new
            self._get_adj_tile(self._cursor).set_rotation(new)

        elif action == GameAction.ACTION4:  # rotate CW
            self._history.append(snap)
            old = self._current_rots[self._cursor]
            new = (old + 90) % 360
            self._current_rots[self._cursor] = new
            self._get_adj_tile(self._cursor).set_rotation(new)

        if self._check_win():
            self.next_level()
            self.complete_action()
            return

        if not self._step_counter.steps_remaining:
            self.lose()

        self.complete_action()
