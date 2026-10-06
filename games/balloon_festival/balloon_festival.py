"""
Balloon Festival
================
Guide colored balloons to their matching wicker baskets at the top of the
sky grid. Balloons automatically float upward one cell per tick. When a
balloon is about to enter a cell that has an active deflector tile, it
moves sideways (left or right per the arrow) instead of up that tick.

Pale-gray inactive deflector markers are visible on the grid — these show
exactly where deflector tiles need to be placed. Move the cursor to an
inactive marker and press ACTION5 to activate it (the tile locks in and
shows its arrow direction). Pressing ACTION5 anywhere always advances the
simulation by one tick.

Cursor movement (ACTION1-4) is free (no tick advance).
If any balloon floats to the top row at the wrong column, the level is lost.

Some deflectors are already switched on when the level opens; how many, and
which, varies per seed. Not every already-active tile is part of the solution -
some sit in cells no balloon passes beneath and are simply inert. Activation is
one-way: an activated tile can never be switched back off.

Color key:
  1  (#1E93FF) — sky (background)
  8  (#F93C31) — red balloon / basket
  11 (#FFDC00) — yellow balloon / basket
  3  (#4FCC30) — green balloon / basket
  9  (#0074D9) — blue balloon / basket
  4  (#FF851B) — left-deflector tile (active)
  7  (#E6E6E6) — right-deflector tile (active)
  2             — inactive deflector marker (pale indicator)
  0  (#FFFFFF) — cursor
  5  (#000000) — border

Actions:
  ACTION1 (↑) — move cursor up   (no tick)
  ACTION2 (↓) — move cursor down (no tick)
  ACTION3 (←) — move cursor left (no tick)
  ACTION4 (→) — move cursor right (no tick)
  ACTION5     — activate deflector at cursor if on an inactive marker;
                always advances simulation by 1 tick
  ACTION7     — undo last action

Win condition : All balloons caught by their matching baskets.
Lose condition: Step budget exhausted, OR a balloon reaches the top row at
                the wrong column (permanently stuck).
Levels        : 2 balloons 10×10 (easy) → 4 balloons 16×14 (expert).
"""

from __future__ import annotations

import random

import numpy as np
from utils.arc_game import AugmentedGame
from arcengine import (
    ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite,
)

# ── Colors ───────────────────────────────────────────────────────────────────
C_SKY           = 1
C_CURSOR        = 0
C_LEFT_DEFL     = 4    # orange – active left-arrow deflector
C_RIGHT_DEFL    = 7    # light-grey – active right-arrow deflector
C_INACTIVE_DEFL = 2    # pale indicator for not-yet-activated deflector marker
C_BORDER        = 5

_BALLOON_COLORS = [8, 11, 3, 9, 6, 12, 14]   # red, yellow, green, blue, pink, orange, lime

CELL = 4   # each grid cell = CELL×CELL pixels

_STEP_BUDGETS = [80, 120, 180, 240, 320, 420, 540]

# ── Level configs: (grid_w, grid_h, num_balloons, max_dx) ────────────────────
_CONFIGS = [
    (10, 10, 2, 1),
    (12, 12, 3, 1),
    (14, 12, 4, 2),
    (16, 14, 4, 2),
    (16, 14, 4, 3),
    (16, 14, 4, 4),
    (16, 14, 4, 5),
]

# ── Sprite pixel-art ─────────────────────────────────────────────────────────

def _balloon_pixels(c: int) -> list[list[int]]:
    """4×4 round balloon with highlight and string."""
    W = 0   # white highlight
    S = 5   # black string
    return [
        [-1,  c,  c, -1],
        [ c,  c,  c,  c],
        [ c,  W,  c, -1],
        [-1,  S,  S, -1],
    ]


def _basket_pixels(c: int) -> list[list[int]]:
    """4×4 wicker basket with handles."""
    return [
        [ c, -1, -1,  c],
        [ c,  c,  c,  c],
        [-1,  c,  c, -1],
        [-1,  c,  c, -1],
    ]


def _left_defl_pixels() -> list[list[int]]:
    """4×4 left-arrow deflector tile (active)."""
    c, t = C_LEFT_DEFL, -1
    return [
        [ t,  c,  t,  t],
        [ c,  c,  c,  c],
        [ c,  c,  c,  c],
        [ t,  c,  t,  t],
    ]


def _right_defl_pixels() -> list[list[int]]:
    """4×4 right-arrow deflector tile (active)."""
    c, t = C_RIGHT_DEFL, -1
    return [
        [ t,  t,  c,  t],
        [ c,  c,  c,  c],
        [ c,  c,  c,  c],
        [ t,  t,  c,  t],
    ]


def _inactive_defl_pixels() -> list[list[int]]:
    """4×4 ghost frame for a not-yet-activated deflector marker."""
    c, t = C_INACTIVE_DEFL, -1
    return [
        [ c,  c,  c,  c],
        [ c,  t,  t,  c],
        [ c,  t,  t,  c],
        [ c,  c,  c,  c],
    ]


_CURSOR_PIXELS = [
    [C_CURSOR, C_CURSOR, C_CURSOR, C_CURSOR],
    [C_CURSOR,       -1,       -1, C_CURSOR],
    [C_CURSOR,       -1,       -1, C_CURSOR],
    [C_CURSOR, C_CURSOR, C_CURSOR, C_CURSOR],
]


# ── HUD ──────────────────────────────────────────────────────────────────────

class SupplyDisplay(RenderableUserDisplay):
    """Progress bar showing remaining step budget."""

    def __init__(self) -> None:
        self.max_steps       = 0
        self.steps_remaining = 0

    def reset(self, max_steps: int) -> None:
        self.max_steps       = max_steps
        self.steps_remaining = max_steps

    def decrement(self) -> None:
        if self.steps_remaining > 0:
            self.steps_remaining -= 1

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.max_steps == 0:
            return frame
        # Progress bar – bottom row (cols 1..60)
        filled = round(59 * self.steps_remaining / self.max_steps)
        for x in range(59):
            frame[63, x + 1] = 7 if x < filled else 4
        return frame


# ── Level generator ──────────────────────────────────────────────────────────

def _simulate_balloon_path(
    bx: int,
    by: int,
    grid_h: int,
    deflectors: dict[tuple[int, int], int],
    grid_w: int,
) -> list[tuple[int, int]]:
    """
    Return the ordered (bottom-to-top) list of grid positions a balloon visits,
    starting from (bx, by), given the active deflector map.
    """
    cx, cy = bx, by
    path: list[tuple[int, int]] = [(cx, cy)]
    visited: set[tuple[int, int]] = {(cx, cy)}
    for _ in range(grid_h * 6):
        if cy <= 0:
            break
        above_y = cy - 1
        defl = deflectors.get((cx, above_y))
        if defl is not None:
            nx = cx + (-1 if defl == 0 else 1)
            ny = cy
            if not (0 <= nx < grid_w):
                break  # hit boundary
        else:
            nx, ny = cx, cy - 1
        key = (nx, ny)
        if key in visited:
            break  # cycle guard
        visited.add(key)
        path.append(key)
        cx, cy = nx, ny
    return path


def _reaches_target_with_deflectors(
    bx: int,
    by: int,
    tx: int,
    ty: int,
    grid_h: int,
    deflectors: dict[tuple[int, int], int],
    grid_w: int,
) -> bool:
    """True if a balloon can reach (tx, ty) under the given active deflectors."""
    cx, cy = bx, by
    visited: set[tuple[int, int]] = {(cx, cy)}
    for _ in range(grid_h * 8):
        if (cx, cy) == (tx, ty):
            return True
        if cy <= 0:
            return False
        above_y = cy - 1
        defl = deflectors.get((cx, above_y))
        if defl is not None:
            nx = cx + (-1 if defl == 0 else 1)
            ny = cy
            if not (0 <= nx < grid_w):
                return False
        else:
            nx, ny = cx, cy - 1
        key = (nx, ny)
        if key in visited:
            return False
        visited.add(key)
        cx, cy = nx, ny
    return False


def _compute_path(
    bx: int,
    by: int,
    tx: int,
    basket_row: int,
    grid_h: int,
    rng: random.Random,
) -> list[tuple[int, int, int]] | None:
    """
    Return deflector positions to route a balloon from column bx to column tx.
    Each entry is (x, y, type) where type 0=left, 1=right.
    Deflector y-positions are randomized in the top half of the grid.
    """
    dx = tx - bx
    if dx == 0:
        return []
    direction = 1 if dx > 0 else -1
    defl_type = 1 if dx > 0 else 0
    n_steps = abs(dx)

    y_min = 1
    # Keep true solution deflectors in upper half, route to the basket row,
    # and ensure the first one sits at least 4 rows above the start y.
    y_max = min(grid_h // 2, by - 4)
    y_min = max(y_min, basket_row)
    if y_max < y_min:
        return None
    if (y_max - y_min + 1) < n_steps:
        return None
    ys = sorted(rng.sample(range(y_min, y_max + 1), n_steps), reverse=True)

    deflectors: list[tuple[int, int, int]] = []
    curr_x = bx
    for defl_y in ys:
        deflectors.append((curr_x, defl_y, defl_type))
        curr_x += direction
    return deflectors


def _free_cells(
    bx: int,
    by: int,
    grid_h: int,
    path: list[tuple[int, int, int]],
) -> set[tuple[int, int]]:
    """
    Return the set of cells the balloon passes through WITHOUT deflection
    when following the provided deflector path.
    """
    defl_map = {(p[0], p[1]): p[2] for p in path}

    cx, cy = bx, by
    free: set[tuple[int, int]] = set()
    for _ in range(grid_h * 4):   # safety upper bound
        if cy <= 0:
            break
        above = (cx, cy - 1)
        if above in defl_map:
            cx += (-1 if defl_map[above] == 0 else 1)
        else:
            free.add(above)
            cy -= 1
    return free


def _generate(grid_w: int, grid_h: int, num_balloons: int,
              max_dx: int, rng: random.Random) -> dict:
    colors = rng.sample(_BALLOON_COLORS, num_balloons)
    interior_cols = list(range(1, grid_w - 1))
    rng.shuffle(interior_cols)
    balloon_cols = interior_cols[:num_balloons]
    balloon_starts: list[tuple[int, int]] = []
    basket_cols:  list[int] = []
    basket_rows:  list[int] = []
    basket_cells: set[tuple[int, int]] = set()
    all_defl:      dict[tuple[int, int], int] = {}
    all_free:      set[tuple[int, int]]       = set()

    # Balloons start in the lower half of the grid.
    start_y_min = max(grid_h // 2 + 1, 2)
    start_y_max = max(start_y_min, grid_h - 1)

    for bx in balloon_cols:
        by = rng.randint(start_y_min, start_y_max)
        basket_y_max = max(0, min(grid_h // 2, 3))
        basket_row = rng.randint(0, basket_y_max)

        candidates = [d for d in range(-max_dx, max_dx + 1) if d != 0]
        rng.shuffle(candidates)
        candidates.append(0)

        used_baskets = set(basket_cols)
        chosen_tx: int | None = None
        chosen_path: list[tuple[int, int, int]] = []
        for dx in candidates:
            tx = bx + dx
            if not (1 <= tx <= grid_w - 2):
                continue
            if tx in used_baskets:
                continue
            path = _compute_path(bx, by, tx, basket_row, grid_h, rng)
            if path is None:
                continue
            defl_xy = {(p[0], p[1]) for p in path}
            free_xy = _free_cells(bx, by, grid_h, path)
            proposed_basket = (tx, basket_row)
            if (not defl_xy.intersection(all_defl.keys()) and
                    not defl_xy.intersection(all_free) and
                    not free_xy.intersection(all_defl.keys()) and
                    proposed_basket not in defl_xy and
                    proposed_basket not in all_defl):
                chosen_tx   = tx
                chosen_path = path
                break

        # Safe fallback: allow straight-up route only when basket column is free.
        if chosen_tx is None and bx not in used_baskets:
            chosen_tx = bx
            chosen_path = []

        if chosen_tx is None:
            raise RuntimeError("Unable to assign a non-blocking unique basket column.")

        balloon_starts.append((bx, by))
        basket_cols.append(chosen_tx)
        basket_rows.append(basket_row)
        basket_cells.add((chosen_tx, basket_row))
        for x, y, t in chosen_path:
            all_defl[(x, y)] = t
        all_free.update(_free_cells(bx, by, grid_h, chosen_path))

    # Split deflectors: some pre-placed (already active), the rest shown as
    # inactive markers that the player must activate.  The split point is drawn
    # per-seed rather than fixed at half, so how much of the solution is handed
    # to the player varies from level to level.  At least one is always left
    # for the player to activate.
    defl_list = list(all_defl.items())
    rng.shuffle(defl_list)
    n_defl = len(defl_list)
    if n_defl:
        n_pre = rng.randint(n_defl // 4, max(n_defl // 4, (2 * n_defl) // 3))
        n_pre = min(n_pre, n_defl - 1)
    else:
        n_pre = 0
    pre_placed     = dict(defl_list[:n_pre])
    sol_deflectors = dict(defl_list[n_pre:])

    # Build combined deflector map (pre-placed + solution) for path simulation.
    combined_defls: dict[tuple[int, int], int] = dict(pre_placed)
    combined_defls.update(sol_deflectors)

    # Safety: ensure every balloon still reaches its own basket under the full
    # active true-deflector map (pre-placed + solution).
    for (bx, by), tx, ty in zip(balloon_starts, basket_cols, basket_rows):
        if not _reaches_target_with_deflectors(
            bx, by, tx, ty, grid_h, combined_defls, grid_w
        ):
            raise RuntimeError("Generated true deflectors block another basket path.")

    # Add TRAP deflectors: inactive markers placed INSIDE a balloon's actual
    # flight path.  Activating a trap misdirects that balloon so it arrives at
    # the wrong column and the level is lost.  This prevents the trivial
    # "activate every inactive marker" strategy.
    trap_deflectors: dict[tuple[int, int], int] = {}
    for i, ((bx, by), basket_col) in enumerate(zip(balloon_starts, basket_cols)):
        path = _simulate_balloon_path(bx, by, grid_h, combined_defls, grid_w)
        # Cells the balloon visits that are free (not a deflector, not the
        # bottom spawn row, not the top basket row).
        free_path = [
            (x, y) for (x, y) in path
            if 0 < y < grid_h - 1
            and (x, y) not in combined_defls
            and (x, y) not in basket_cells
            and (x, y) not in trap_deflectors
        ]
        if len(free_path) >= 3:
            # Pick a cell near the TOP of the path (late in the journey).
            # A deflection here is guaranteed to cause the balloon to reach
            # the wrong basket column.
            upper_start = max(0, len(free_path) * 2 // 3)
            trap_x, trap_y = free_path[upper_start]
            trap_deflectors[(trap_x, trap_y)] = rng.randint(0, 1)

    # ── Column Decoys ────────────────────────────────────────────────────────
    # Problem: solution deflectors sit at small y (near top) while trap/free
    # cells are at large y — so the top-most inactive marker in any column is
    # always the correct one, making the puzzle trivially easy.
    # Fix: for ~50% of solution deflectors, place a "wrong-direction" inactive
    # marker ABOVE it (smaller y) in the same column so the top-most marker in
    # that column is the WRONG choice.
    all_taken: set[tuple[int, int]] = (
        set(all_defl.keys()) | all_free | set(trap_deflectors.keys()) | basket_cells
    )
    column_decoys: dict[tuple[int, int], int] = {}

    for (sx, sy), stype in sol_deflectors.items():
        if sy <= 1:
            continue  # no room to place a decoy above
        if rng.random() >= 0.5:
            continue  # only add decoys ~50% of the time
        # Scan upward (decreasing y) from sy-1 down to y=1
        for decoy_y in range(sy - 1, 0, -1):
            pos = (sx, decoy_y)
            if pos in all_taken or pos in column_decoys:
                continue
            # Wrong direction: opposite to the real deflector
            column_decoys[pos] = 1 - stype
            all_taken.add(pos)
            break

    # Also add some harmless noise in cells outside all balloon paths (same
    # total inactive-marker count as before).
    occupied_cells = set(all_defl.keys()) | all_free | basket_cells
    excluded_rows = {0, grid_h - 1}
    noise_candidates = [
        (x, y)
        for y in range(1, grid_h - 1)
        for x in range(1, grid_w - 1)
        if (x, y) not in occupied_cells
        and (x, y) not in trap_deflectors
        and (x, y) not in column_decoys
        and y not in excluded_rows
    ]
    rng.shuffle(noise_candidates)
    num_noise = max(1, len(sol_deflectors))
    # Start with traps + column decoys, then fill remaining slots with harmless noise.
    noise_deflectors: dict[tuple[int, int], int] = dict(trap_deflectors)
    noise_deflectors.update(column_decoys)
    remaining_slots = max(0, num_noise - len(noise_deflectors))
    for (x, y) in noise_candidates[:remaining_slots]:
        noise_deflectors[(x, y)] = rng.randint(0, 1)

    # ── Per-seed starting-state augmentation ─────────────────────────────────
    # Vary which deflectors are already switched on when the level opens.  This
    # closes a solution leak: without it EVERY active deflector on screen is a
    # true-solution deflector, so "active tile => part of the answer" reads the
    # answer straight off the board.
    #
    # Two hard constraints shape what may start active:
    #   * Activation is ONE-WAY (`_cycle_deflector` never deactivates, and
    #     pre-placed tiles are immutable).  Anything switched on at t=0 is on
    #     forever, so it must be harmless under the intended solve.
    #   * TRAP markers sit ON a balloon's flight path by construction, so a trap
    #     that starts active misdirects its balloon and makes the level
    #     unwinnable before the first keypress.  Traps are never promoted.
    # Column decoys are also left alone: an inactive wrong-direction marker
    # above a solution deflector is exactly what stops "top-most inactive marker
    # in the column" from being a winning heuristic, and activating it would
    # hand that heuristic back.

    # Cells a deflector could ever act on: (x, y) such that some balloon stands
    # at (x, y+1) at some tick under the fully-solved deflector map.
    swept: set[tuple[int, int]] = set()
    for (bx, by) in balloon_starts:
        for (px, py) in _simulate_balloon_path(
            bx, by, grid_h, combined_defls, grid_w
        ):
            swept.add((px, py - 1))

    start_active: dict[tuple[int, int], int] = {}

    # (a) Promote a random subset of the harmless inactive markers to active.
    for pos, t in list(noise_deflectors.items()):
        if pos in trap_deflectors or pos in column_decoys:
            continue
        if rng.random() < 0.5:
            start_active[pos] = t
            del noise_deflectors[pos]

    # (b) Scatter a few brand-new active deflectors in never-swept cells.  They
    #     are inert under the intended solve but punish an off-plan balloon.
    taken_now = (set(all_defl) | set(noise_deflectors) | set(start_active)
                 | all_free | basket_cells)
    extra_candidates = [
        (x, y)
        for y in range(1, grid_h - 1)
        for x in range(1, grid_w - 1)
        if (x, y) not in swept and (x, y) not in taken_now
    ]
    rng.shuffle(extra_candidates)
    for pos in extra_candidates[:rng.randint(1, 4)]:
        start_active[pos] = rng.randint(0, 1)

    pre_placed.update(start_active)

    # Re-verify with the augmentation applied: every balloon must still reach
    # its own basket once all real deflectors are active.  (The earlier check
    # ran before traps, decoys and this augmentation existed.)  Raising here
    # just makes the caller re-roll the level.
    verify_defls = dict(pre_placed)
    verify_defls.update(sol_deflectors)
    for (bx, by), tx, ty in zip(balloon_starts, basket_cols, basket_rows):
        if not _reaches_target_with_deflectors(
            bx, by, tx, ty, grid_h, verify_defls, grid_w
        ):
            raise RuntimeError("Start-state augmentation blocks a basket path.")

    # Cursor starts at first sol_deflector (deterministic), or grid interior
    if sol_deflectors:
        cursor_start = min(sol_deflectors.keys())
    else:
        cursor_start = (grid_w // 2, grid_h // 2)

    return {
        "grid_w":            grid_w,
        "grid_h":            grid_h,
        "num_balloons":      num_balloons,
        "balloon_cols":      balloon_cols,
        "balloon_starts":    balloon_starts,
        "basket_cols":       basket_cols,
        "basket_rows":       basket_rows,
        "colors":            colors,
        "pre_placed":        pre_placed,
        "sol_deflectors":    sol_deflectors,
        "noise_deflectors":  noise_deflectors,
        "trap_deflectors":   trap_deflectors,
        "cursor_start":      cursor_start,
    }


def _build_level(data: dict) -> Level:
    sprites: list[Sprite] = []

    # Pre-placed active deflectors
    for (x, y), t in data["pre_placed"].items():
        pixels = _left_defl_pixels() if t == 0 else _right_defl_pixels()
        sprites.append(
            Sprite(pixels=pixels, name=f"predefl_{x}_{y}",
                   collidable=False,
                   tags=["pre_defl", f"pre_defl_{x}_{y}"],
                   layer=1)
            .set_position(x * CELL, y * CELL)
        )

    # Baskets near top (random y in upper half)
    for i, (tx, ty, c) in enumerate(
        zip(data["basket_cols"], data["basket_rows"], data["colors"])
    ):
        sprites.append(
            Sprite(pixels=_basket_pixels(c), name=f"basket_{i}",
                   collidable=False,
                   tags=["basket", f"basket_{i}"],
                   layer=2)
            .set_position(tx * CELL, ty * CELL)
        )

    # Cursor
    cx, cy = data["cursor_start"]
    sprites.append(
        Sprite(pixels=_CURSOR_PIXELS, name="cursor",
               collidable=False, tags=["cursor"], layer=5)
        .set_position(cx * CELL, cy * CELL)
    )

    grid_w, grid_h = data["grid_w"], data["grid_h"]
    return Level(sprites=sprites, grid_size=(grid_w * CELL, grid_h * CELL))


# ── Game class ────────────────────────────────────────────────────────────────

class BalloonFestival(AugmentedGame):

    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: the base draws this level's rotation
        rng = random.Random(seed)
        self._supply_display = SupplyDisplay()
        self._level_data: list[dict] = []
        levels: list[Level] = []

        for grid_w, grid_h, num_balloons, max_dx in _CONFIGS:
            data: dict | None = None
            for _ in range(64):
                try:
                    data = _generate(grid_w, grid_h, num_balloons, max_dx, rng)
                    break
                except RuntimeError:
                    continue
            if data is None:
                raise RuntimeError(
                    f"Failed to generate valid non-blocking level {grid_w}x{grid_h}."
                )
            self._level_data.append(data)
            levels.append(_build_level(data))

        # Runtime state (set in on_set_level)
        self._balloon_positions: list[tuple[int, int]] = []
        self._balloon_caught:    list[bool] = []
        self._deflectors:        dict[tuple[int, int], int] = {}  # activated sol deflectors
        self._inactive_defls:    dict[tuple[int, int], int] = {}  # pending inactive markers
        self._cursor_pos:        tuple[int, int] = (0, 0)
        self._history:           list[tuple] = []

        super().__init__(
            game_id="balloon_festival",
            levels=levels,
            camera=Camera(
                background=C_SKY,
                letter_box=C_BORDER,
                interfaces=[self._supply_display],
            ),
            available_actions=[1, 2, 3, 4, 5, 7],
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _cursor_sprite(self) -> Sprite:
        return self.current_level.get_sprites_by_tag("cursor")[0]

    def _balloon_sprite(self, i: int) -> Sprite | None:
        sprites = self.current_level.get_sprites_by_tag(f"balloon_{i}")
        return sprites[0] if sprites else None

    def _in_bounds(self, x: int, y: int) -> bool:
        d = self._level_data[self._current_level_index]
        return 0 <= x < d["grid_w"] and 0 <= y < d["grid_h"]

    def _get_deflector(self, x: int, y: int) -> int | None:
        """Return 0 (left), 1 (right), or None for an active deflector."""
        if (x, y) in self._deflectors:
            return self._deflectors[(x, y)]
        d = self._level_data[self._current_level_index]
        return d["pre_placed"].get((x, y))

    # ── Sprite helpers ────────────────────────────────────────────────────────

    def _add_defl_sprite(self, x: int, y: int, t: int) -> None:
        pixels = _left_defl_pixels() if t == 0 else _right_defl_pixels()
        self.current_level.add_sprite(
            Sprite(pixels=pixels, name=f"pdefl_{x}_{y}",
                   collidable=False,
                   tags=["player_defl", f"pdefl_{x}_{y}"],
                   layer=2)
            .set_position(x * CELL, y * CELL)
        )

    def _remove_defl_sprite(self, x: int, y: int) -> None:
        for s in list(self.current_level.get_sprites_by_tag(f"pdefl_{x}_{y}")):
            self.current_level.remove_sprite(s)

    def _add_inactive_sprite(self, x: int, y: int) -> None:
        self.current_level.add_sprite(
            Sprite(pixels=_inactive_defl_pixels(), name=f"idefl_{x}_{y}",
                   collidable=False,
                   tags=["inactive_defl", f"idefl_{x}_{y}"],
                   layer=1)
            .set_position(x * CELL, y * CELL)
        )

    def _remove_inactive_sprite(self, x: int, y: int) -> None:
        for s in list(self.current_level.get_sprites_by_tag(f"idefl_{x}_{y}")):
            self.current_level.remove_sprite(s)

    # ── Simulation ───────────────────────────────────────────────────────────

    def _advance_tick(self) -> None:
        """Move all live balloons one step upward (or sideways at deflectors)."""
        d      = self._level_data[self._current_level_index]
        grid_w = d["grid_w"]

        for i, (bx, by) in enumerate(self._balloon_positions):
            if self._balloon_caught[i]:
                continue

            if by == 0:
                # Already at top row (stuck or just caught) — stay put
                continue

            above_y = by - 1
            defl = self._get_deflector(bx, above_y)
            if defl is not None:
                nx = bx + (-1 if defl == 0 else 1)
                ny = by
                if not (0 <= nx < grid_w):
                    continue  # clamp at side boundary
            else:
                nx, ny = bx, by - 1

            self._balloon_positions[i] = (nx, ny)

            # Check catch: basket is at (basket_col, basket_row)
            basket_col = d["basket_cols"][i]
            basket_row = d["basket_rows"][i]
            if nx == basket_col and ny == basket_row:
                self._balloon_caught[i] = True
                s = self._balloon_sprite(i)
                if s:
                    s.set_visible(False)
                continue

            # Update sprite position
            s = self._balloon_sprite(i)
            if s:
                s.set_position(nx * CELL, ny * CELL)

    def _check_win(self) -> bool:
        return bool(self._balloon_positions) and all(self._balloon_caught)

    def _any_stuck_at_top(self) -> bool:
        """True if any live balloon is at row 0 but not caught (wrong column)."""
        return any(
            not caught and by == 0
            for caught, (_, by) in zip(self._balloon_caught, self._balloon_positions)
        )

    # ── Deflector activation ──────────────────────────────────────────────────

    def _cycle_deflector(self) -> None:
        """Activate the inactive deflector marker at the cursor, if present."""
        cx, cy = self._cursor_pos
        d = self._level_data[self._current_level_index]

        # Pre-placed deflectors are immutable
        if (cx, cy) in d["pre_placed"]:
            return

        # Activate an inactive deflector marker with one press
        if (cx, cy) in self._inactive_defls:
            defl_type = self._inactive_defls.pop((cx, cy))
            self._deflectors[(cx, cy)] = defl_type
            self._remove_inactive_sprite(cx, cy)
            self._add_defl_sprite(cx, cy, defl_type)

    # ── Snapshot / restore ───────────────────────────────────────────────────

    def _snapshot(self) -> tuple:
        return (
            list(self._balloon_positions),
            list(self._balloon_caught),
            dict(self._deflectors),
            dict(self._inactive_defls),
            self._cursor_pos,
        )

    def _restore(self, snap: tuple) -> None:
        bal_pos, bal_caught, defls, inactive_defls, cursor = snap

        self._balloon_positions = list(bal_pos)
        self._balloon_caught    = list(bal_caught)

        # Remove all player-placed active deflector sprites
        for s in list(self.current_level.get_sprites_by_tag("player_defl")):
            self.current_level.remove_sprite(s)
        # Remove all inactive deflector sprites
        for s in list(self.current_level.get_sprites_by_tag("inactive_defl")):
            self.current_level.remove_sprite(s)

        # Restore active deflectors
        self._deflectors = {}
        for (x, y), t in defls.items():
            self._deflectors[(x, y)] = t
            self._add_defl_sprite(x, y, t)

        # Restore inactive deflectors
        self._inactive_defls = {}
        for (x, y), t in inactive_defls.items():
            self._inactive_defls[(x, y)] = t
            self._add_inactive_sprite(x, y)

        # Update balloon sprites
        d = self._level_data[self._current_level_index]
        for i in range(d["num_balloons"]):
            s = self._balloon_sprite(i)
            if s is not None:
                if self._balloon_caught[i]:
                    s.set_visible(False)
                else:
                    s.set_visible(True)
                    bx, by = self._balloon_positions[i]
                    s.set_position(bx * CELL, by * CELL)

        # Restore cursor
        self._cursor_pos = cursor
        self._cursor_sprite().set_position(cursor[0] * CELL, cursor[1] * CELL)

    # ── on_set_level ─────────────────────────────────────────────────────────

    def on_set_level(self, level: Level) -> None:
        # Remove dynamic sprites from previous calls (idempotent)
        for s in list(level.get_sprites()):
            if any(t in s.tags for t in ("balloon", "player_defl", "inactive_defl")):
                level.remove_sprite(s)

        self._history.clear()
        idx  = self._current_level_index
        d    = self._level_data[idx]

        self._balloon_positions = list(d["balloon_starts"])
        self._balloon_caught    = [False] * d["num_balloons"]
        self._deflectors        = {}
        # Both solution and noise deflectors shown as inactive markers
        self._inactive_defls    = dict(d["sol_deflectors"])
        self._inactive_defls.update(d["noise_deflectors"])

        self._supply_display.reset(_STEP_BUDGETS[idx])

        # Add balloon sprites
        for i, ((bx, by), c) in enumerate(
            zip(self._balloon_positions, d["colors"])
        ):
            level.add_sprite(
                Sprite(pixels=_balloon_pixels(c), name=f"balloon_{i}",
                       collidable=False,
                       tags=["balloon", f"balloon_{i}"],
                       layer=4)
                .set_position(bx * CELL, by * CELL)
            )

        # Add inactive deflector marker sprites
        for (x, y) in self._inactive_defls:
            self._add_inactive_sprite(x, y)

        # Position cursor at its start cell
        self._cursor_pos = d["cursor_start"]
        self._cursor_sprite().set_position(
            self._cursor_pos[0] * CELL,
            self._cursor_pos[1] * CELL,
        )

    # ── step ─────────────────────────────────────────────────────────────────

    _MOVE_DELTAS: dict[int, tuple[int, int]] = {
        GameAction.ACTION1: (0, -1),
        GameAction.ACTION2: (0,  1),
        GameAction.ACTION3: (-1, 0),
        GameAction.ACTION4: (1,  0),
    }

    def step(self) -> None:
        self._supply_display.decrement()
        action = self.action.id

        # ── Undo ──────────────────────────────────────────────────────────
        if action == GameAction.ACTION7:
            if self._history:
                self._restore(self._history.pop())
            self.complete_action()
            if not self._supply_display.steps_remaining:
                self.lose()
            return

        # Save state before any other action
        self._history.append(self._snapshot())

        # ── Move cursor (ACTION1–4) — no tick advance ──────────────────────
        delta = self._MOVE_DELTAS.get(action)
        if delta:
            dx, dy = delta
            cx, cy = self._cursor_pos
            nx, ny = cx + dx, cy + dy
            if self._in_bounds(nx, ny):
                self._cursor_pos = (nx, ny)
                self._cursor_sprite().set_position(nx * CELL, ny * CELL)
            self.complete_action()
            if not self._supply_display.steps_remaining:
                self.lose()
            return

        # ── ACTION5: activate deflector (if on inactive marker) + advance 1 tick
        if action == GameAction.ACTION5:
            self._cycle_deflector()
            self._advance_tick()

            if self._check_win():
                self.complete_action()
                self.next_level()
                return

            # Loss if any balloon floated to wrong top column
            if self._any_stuck_at_top():
                self.complete_action()
                self.lose()
                return

        self.complete_action()
        if not self._supply_display.steps_remaining:
            self.lose()
