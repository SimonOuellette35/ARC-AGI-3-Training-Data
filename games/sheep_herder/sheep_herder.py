"""
Sheep Herder
============
Push sheep into their matching colored pens using Sokoban-style push mechanics.
The shepherd moves into a cell adjacent to a sheep to push it one cell in the
same direction. Sheep cannot be pushed into walls, fences, or trees.

On levels 2–4, pen gates start closed (locked). Each locked gate has a
matching colored key placed somewhere in the pasture. Walk the shepherd over a
key to collect it — the matching gate opens immediately. Once unlocked, the
gate can be toggled open/closed with ACTION5 (when shepherd stands on gate).

Levels 3–4 also add immovable tree obstacles (dark blocks with green centers)
that force the shepherd to navigate around them when pushing sheep.

Color key:
  14 (#4FCC30) — grass (floor, passable)
  4  (#333333) — fence / tree outline (impassable)
  9  (#1E93FF) — shepherd (blue figure)
  8  (#F93C31) — red pen floor / red sheep / red key
  11 (#FFDC00) — yellow pen floor / yellow sheep / yellow key
  12 (#FF851B) — orange pen floor / orange sheep / orange key
  6  (#E36EF6) — pink pen floor / pink sheep / pink key
  1  (#1E93FF) — blue pen floor / blue sheep / blue key
  3  (#4FCC30) — gate (open, same shade as grass)
  7  (#E6E6E6) — gate (closed, light grey)
  5  (#000000) — letterbox border

Actions:
  ACTION1 (↑) — move shepherd up (pushes adjacent sheep)
  ACTION2 (↓) — move shepherd down
  ACTION3 (←) — move shepherd left
  ACTION4 (→) — move shepherd right
  ACTION5     — toggle gate open/closed (only works on unlocked gates)
  ACTION7     — undo last shepherd move

Win condition : All sheep inside their matching colored pens.
Lose condition: Step budget exhausted.
Levels        : 4 puzzles — 2 sheep / open gates (easy) →
                            3 sheep / locked gates + keys (medium) →
                            4 sheep / keys + trees (hard) →
                            5 sheep / keys + more trees (expert).
"""

from __future__ import annotations

import random
from collections import deque
from itertools import permutations as _iterperms

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite
from utils.arc_game import AugmentedGame

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
C_GRASS       = 14  # green grass background
C_FENCE       = 4   # dark fence / tree outline
C_SHEPHERD    = 9   # blue shepherd
C_GATE_OPEN   = 3   # open gate (same shade as grass)
C_GATE_CLOSED = 7   # closed gate (light grey)
C_BORDER      = 5   # black letterbox

# Pen / sheep / key colors — red, yellow, orange, pink, blue
_PEN_COLORS = [8, 11, 12, 6, 1, 2, 10, 13]

# Step budgets — increased for levels 1-3 to allow key collection
_STEP_BUDGETS = [80, 200, 300, 420, 560, 700, 880]

# Level configs: (grid_w, grid_h, num_sheep)
_CONFIGS = [
    (11, 9,  2),
    (13, 11, 3),
    (15, 13, 4),
    (17, 15, 5),
    (19, 17, 5),
    (21, 19, 5),
    (21, 19, 5),
]

# Per-level features
_GATES_START_CLOSED = [False, True,  True,  True,  True,  True,  True ]
_NUM_TREES          = [0,     0,     3,     5,     7,     9,     11   ]

# ---------------------------------------------------------------------------
# Cell size: each grid cell = CELL × CELL pixels in camera space.
# Max grid 17×15 → 51×45 viewport (fits 64×64).
# ---------------------------------------------------------------------------
CELL = 3

# ---------------------------------------------------------------------------
# Sprite pixel art  (all CELL×CELL = 3×3 pixels)
# ---------------------------------------------------------------------------

_SHEPHERD_PIXELS = [
    [-1,         C_SHEPHERD, -1        ],  # head
    [C_SHEPHERD, C_SHEPHERD, C_SHEPHERD],  # body
    [C_SHEPHERD, -1,         0         ],  # legs + crook tip (white)
]


def _sheep_pixels(c: int) -> list[list[int]]:
    """3×3 sheep: fluffy body with colored coat and black hooves."""
    return [
        [ c,  c,  c],
        [ c,  c,  c],
        [ 5,  c,  5],   # black hooves + colored body
    ]


def _fence_pixels() -> list[list[int]]:
    """3×3 solid dark fence block."""
    return [[C_FENCE] * CELL] * CELL


def _tree_pixels() -> list[list[int]]:
    """3×3 tree: dark border with green center — distinct from solid fences."""
    return [
        [C_FENCE, C_FENCE,  C_FENCE],
        [C_FENCE, C_GRASS,  C_FENCE],
        [C_FENCE, C_FENCE,  C_FENCE],
    ]


def _gate_pixels(open_state: bool) -> list[list[int]]:
    """3×3 gate in open (grass-coloured) or closed (grey) state."""
    c = C_GATE_OPEN if open_state else C_GATE_CLOSED
    return [[c] * CELL] * CELL


def _pen_pixels(c: int) -> list[list[int]]:
    """3×3 pen interior tile in the pen's color."""
    return [[c] * CELL] * CELL


def _key_pixels(c: int) -> list[list[int]]:
    """3×3 key: bow and shaft in the matching pen color."""
    return [
        [-1,  c,  c],   # bow (circular part)
        [ c,  c, -1],   # bow + shaft start
        [ c,  c,  c],   # shaft + teeth
    ]


# ---------------------------------------------------------------------------
# HUD — step progress bar
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
# Solvability checker (used during level generation)
# ---------------------------------------------------------------------------

def _check_order(order, shepherd, sheep_positions, gate_positions, pen_cells,
                 grid, w, h, num_sheep) -> bool:
    """Return True if herding sheep in the given order is BFS-feasible."""
    current_shepherd = shepherd
    current_sheep = list(sheep_positions)
    closed: set = set()

    for i in order:
        pen_set = set(pen_cells[i])

        def _blk(x, y):
            if not (0 <= x < w and 0 <= y < h):
                return True
            v = grid[y][x]
            if v == 20:          # tree — always blocked
                return True
            if v == 0:           # fence cell — might be an open/closed gate
                for j in closed:
                    if (x, y) == gate_positions[j]:
                        return True  # closed gate
                for k in range(num_sheep):
                    if k not in closed and (x, y) == gate_positions[k]:
                        return False  # open gate (not yet closed)
                return True          # plain fence
            # Unherded sheep (other than i) block movement
            for j in range(num_sheep):
                if j != i and j not in closed and (x, y) == current_sheep[j]:
                    return True
            return False

        start = (*current_shepherd, *current_sheep[i])
        parent: dict = {start: None}
        q: deque = deque([start])
        goal = None

        while q and goal is None:
            sx, sy, ex, ey = q.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = sx + dx, sy + dy
                new_ex, new_ey = ex, ey
                if nx == ex and ny == ey:
                    # Shepherd is pushing sheep
                    pnx, pny = ex + dx, ey + dy
                    if _blk(pnx, pny):
                        continue
                    new_ex, new_ey = pnx, pny
                else:
                    if _blk(nx, ny):
                        continue
                ns = (nx, ny, new_ex, new_ey)
                if ns not in parent:
                    parent[ns] = (sx, sy, ex, ey)
                    q.append(ns)
                    if (new_ex, new_ey) in pen_set:
                        goal = ns
                        break

        if goal is None:
            return False

        current_shepherd = (goal[0], goal[1])
        current_sheep[i] = (goal[2], goal[3])
        closed.add(i)

    return True


def _find_solvable_order(shepherd, sheep_positions, gate_positions, pen_cells,
                         grid, w, h, num_sheep, rng=None, max_perms=None):
    """Try herding permutations; return first feasible one or None.

    When max_perms is set (or num_sheep >= 5), tries natural order + reversed +
    a small number of random orders instead of exhaustive search.  This keeps
    generation fast on large grids where 5! × 30 BFS runs would be too slow.
    """
    n = num_sheep
    # Fast path: try natural order and reversed first (O(1) permutations)
    natural = list(range(n))
    if _check_order(natural, shepherd, sheep_positions, gate_positions,
                    pen_cells, grid, w, h, num_sheep):
        return natural
    reversed_order = list(reversed(range(n)))
    if _check_order(reversed_order, shepherd, sheep_positions, gate_positions,
                    pen_cells, grid, w, h, num_sheep):
        return reversed_order

    # Decide how many additional random orders to try
    if max_perms is None:
        import math
        full = math.factorial(n)
        max_perms = min(full, 8) if n >= 5 else full

    tried = {tuple(natural), tuple(reversed_order)}
    if rng is not None:
        for _ in range(max_perms * 3):  # extra attempts to find untried orders
            if len(tried) - 2 >= max_perms:
                break
            order = list(range(n))
            rng.shuffle(order)
            key = tuple(order)
            if key in tried:
                continue
            tried.add(key)
            if _check_order(order, shepherd, sheep_positions, gate_positions,
                            pen_cells, grid, w, h, num_sheep):
                return order
    else:
        for order in _iterperms(range(n)):
            key = tuple(order)
            if key in tried:
                continue
            if _check_order(list(order), shepherd, sheep_positions,
                            gate_positions, pen_cells, grid, w, h, num_sheep):
                return list(order)
    return None


# ---------------------------------------------------------------------------
# Level generation
# ---------------------------------------------------------------------------

def _make_pasture(width: int, height: int, num_sheep: int,
                  level_idx: int, rng: random.Random) -> dict:
    """Generate a solvable pasture layout for the given level.

    Grid value encoding:
      0   — fence / wall
      1   — grass (passable)
      2+i — pen interior for sheep i
      10+i — gate for pen i
      20  — tree (impassable obstacle)
    """
    gates_start_closed = _GATES_START_CLOSED[level_idx]
    target_trees       = _NUM_TREES[level_idx]

    # Start with all grass; add perimeter fence
    grid = [[1] * width for _ in range(height)]
    for i in range(width):
        grid[0][i] = grid[height - 1][i] = 0
    for i in range(height):
        grid[i][0] = grid[i][width - 1] = 0

    colors = rng.sample(_PEN_COLORS, k=num_sheep)
    pen_cells:     list[list[tuple[int, int]]] = []
    gate_positions: list[tuple[int, int]]       = []
    gate_pen_idx:   list[int]                   = []

    # Pen geometry: fixed depth, width scales with sheep count.
    pen_height = 3
    pen_width  = max(3, (width - 4) // num_sheep)

    def _base_anchor(i: int) -> tuple[int, int]:
        px = width - 2 - pen_width * (i + 1) + i
        py = height - 2 - pen_height
        return (
            max(2, min(px, width - pen_width - 2)),
            max(2, min(py, height - pen_height - 2)),
        )

    # Randomize pen section position by translating the whole cluster.
    base_anchors = [_base_anchor(i) for i in range(num_sheep)]
    min_px = min(px for px, _ in base_anchors)
    max_px = max(px for px, _ in base_anchors)
    min_py = min(py for _, py in base_anchors)
    max_py = max(py for _, py in base_anchors)

    dx_min = 2 - min_px
    dx_max = (width - pen_width - 2) - max_px
    dy_min = 2 - min_py
    dy_max = (height - pen_height - 2) - max_py

    dx = rng.randint(dx_min, dx_max) if dx_min <= dx_max else 0
    dy = rng.randint(dy_min, dy_max) if dy_min <= dy_max else 0
    pen_anchors = [(px + dx, py + dy) for px, py in base_anchors]

    for i in range(num_sheep):
        px, py = pen_anchors[i]
        cells = []
        for fy in range(py, py + pen_height):
            for fx in range(px, px + pen_width):
                on_border = (fy == py or fy == py + pen_height - 1 or
                             fx == px or fx == px + pen_width - 1)
                if on_border:
                    grid[fy][fx] = 0  # fence
                else:
                    grid[fy][fx] = 2 + i  # pen interior
                    cells.append((fx, fy))

        gate_x = px + pen_width // 2
        gate_y = py
        grid[gate_y][gate_x] = 10 + i
        gate_positions.append((gate_x, gate_y))
        gate_pen_idx.append(i)
        pen_cells.append(cells)

    shepherd = (2, 2)

    # Sheep must start above the highest pen row.
    py_top = min(py for _, py in pen_anchors)

    def _pushable(x: int, y: int) -> bool:
        """True if sheep at (x,y) can be pushed in ≥1 horiz + ≥1 vert direction."""
        h_push = v_push = 0
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            dest     = (x + dx, y + dy)
            approach = (x - dx, y - dy)
            dest_ok  = (0 <= dest[0] < width  and 0 <= dest[1] < height
                        and grid[dest[1]][dest[0]] not in (0, 20))
            appr_ok  = (0 <= approach[0] < width  and 0 <= approach[1] < height
                        and grid[approach[1]][approach[0]] not in (0, 20))
            if dest_ok and appr_ok:
                if dy == 0:
                    h_push += 1
                else:
                    v_push += 1
        return h_push >= 1 and v_push >= 1

    open_cells = [
        (x, y) for y in range(1, py_top)
        for x in range(1, width - 1)
        if grid[y][x] == 1 and (x, y) != shepherd and _pushable(x, y)
    ]

    sheep_positions = None
    solve_order     = None
    pool = list(open_cells)
    # Limit placement attempts for large grids to keep generation fast.
    max_placements = 10 if num_sheep >= 5 else 30
    for _ in range(max_placements):
        rng.shuffle(pool)
        candidates = pool[:num_sheep]
        if len(candidates) < num_sheep:
            break
        order = _find_solvable_order(
            shepherd, candidates, gate_positions, pen_cells,
            grid, width, height, num_sheep, rng=rng,
        )
        if order is not None:
            sheep_positions = candidates
            solve_order     = order
            break

    if sheep_positions is None:
        # Deterministic fallback: one sheep per gate column, row 2
        sheep_positions = [(gate_positions[i][0], 2) for i in range(num_sheep)]
        solve_order     = list(range(num_sheep))

    # ------------------------------------------------------------------
    # Place trees (levels 2+) — one at a time, verifying solvability.
    # Optimization: check only the existing solve_order rather than
    # searching all permutations, which keeps generation fast on large grids.
    # ------------------------------------------------------------------
    tree_positions: list[tuple[int, int]] = []
    if target_trees > 0:
        tree_candidates = [
            (x, y) for y in range(1, py_top)
            for x in range(1, width - 1)
            if grid[y][x] == 1
            and (x, y) != shepherd
            and (x, y) not in sheep_positions
        ]
        rng.shuffle(tree_candidates)
        # Cap candidates checked to keep generation fast on large grids.
        max_candidates = min(len(tree_candidates), target_trees * 6)
        for tx, ty in tree_candidates[:max_candidates]:
            if len(tree_positions) >= target_trees:
                break
            grid[ty][tx] = 20  # tentative tree
            # Fast check: verify the already-known solve_order still works.
            if _check_order(
                solve_order, shepherd, sheep_positions, gate_positions,
                pen_cells, grid, width, height, num_sheep,
            ):
                tree_positions.append((tx, ty))
            else:
                grid[ty][tx] = 1  # undo — tree breaks solvability

    # ------------------------------------------------------------------
    # Place keys (levels 1+)
    # Keys must be reachable from shepherd without going through sheep
    # (since solver treats sheep as immovable walls during navigation).
    # ------------------------------------------------------------------
    key_positions: list[tuple[int, int]] = []
    if gates_start_closed:
        # BFS from shepherd treating fences, trees, AND all sheep as walls
        sheep_set = set(sheep_positions)
        accessible: set[tuple[int, int]] = set()
        bq: deque = deque([shepherd])
        while bq:
            cx, cy = bq.popleft()
            if (cx, cy) in accessible:
                continue
            accessible.add((cx, cy))
            for ddx, ddy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = cx + ddx, cy + ddy
                if (0 <= nx < width and 0 <= ny < height
                        and (nx, ny) not in accessible
                        and (nx, ny) not in sheep_set
                        and grid[ny][nx] not in (0, 20)):
                    bq.append((nx, ny))

        available_for_keys = [
            (x, y) for x, y in accessible
            if grid[y][x] == 1   # grass only (not pen interior or gate)
            and (x, y) != shepherd
        ]
        rng.shuffle(available_for_keys)

        for i in range(num_sheep):
            placed = False
            for pos in available_for_keys:
                if pos not in key_positions:
                    key_positions.append(pos)
                    placed = True
                    break
            if not placed:
                # Last-resort fallback: place key near shepherd (immediate pickup)
                key_positions.append(shepherd)

    return {
        "grid":              grid,
        "shepherd":          shepherd,
        "sheep_positions":   sheep_positions,
        "sheep_colors":      colors,
        "gate_positions":    gate_positions,
        "gate_pen_idx":      gate_pen_idx,
        "pen_cells":         pen_cells,
        "num_sheep":         num_sheep,
        "solve_order":       solve_order,
        "key_positions":     key_positions,      # [] if level 0
        "tree_positions":    tree_positions,
        "gates_start_closed": gates_start_closed,
    }


def _build_level(data: dict) -> Level:
    """Build static level structure (fences, pens, gates, trees).
    Dynamic entities (shepherd, sheep, keys) are added by on_set_level.
    """
    grid         = data["grid"]
    height, width = len(grid), len(grid[0])
    colors        = data["sheep_colors"]
    closed_init   = data["gates_start_closed"]
    sprites: list[Sprite] = []

    for y in range(height):
        for x in range(width):
            val = grid[y][x]
            if val == 0:
                sprites.append(
                    Sprite(pixels=_fence_pixels(), name=f"fence_{x}_{y}",
                           collidable=True, tags=["fence"], layer=-1)
                    .set_position(x * CELL, y * CELL)
                )
            elif val == 20:
                sprites.append(
                    Sprite(pixels=_tree_pixels(), name=f"tree_{x}_{y}",
                           collidable=True, tags=["tree"], layer=-1)
                    .set_position(x * CELL, y * CELL)
                )
            elif val == 1:
                pass  # grass — camera background handles it
            elif 2 <= val < 10:
                pen_i = val - 2
                sprites.append(
                    Sprite(pixels=_pen_pixels(colors[pen_i]),
                           name=f"pen_{pen_i}_{x}_{y}",
                           collidable=False, tags=["pen", f"pen_{pen_i}"], layer=0)
                    .set_position(x * CELL, y * CELL)
                )
            elif val >= 10:
                gate_i     = val - 10
                open_state = not closed_init
                tags = ["gate", f"gate_{gate_i}",
                        "gate_open" if open_state else "gate_closed"]
                sprites.append(
                    Sprite(pixels=_gate_pixels(open_state),
                           name=f"gate_{gate_i}",
                           collidable=closed_init,   # closed gates block movement
                           tags=tags, layer=1)
                    .set_position(x * CELL, y * CELL)
                )

    return Level(sprites=sprites, grid_size=(width * CELL, height * CELL))


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class SheepHerder(AugmentedGame):

    def __init__(self, seed: int | None = None) -> None:
        rng = random.Random(seed)
        self._step_counter = StepCounter(_STEP_BUDGETS[0])
        self._level_data: list[dict] = []
        levels: list[Level] = []

        for level_idx, (width, height, num_sheep) in enumerate(_CONFIGS):
            data = _make_pasture(width, height, num_sheep, level_idx, rng)
            self._level_data.append(data)
            levels.append(_build_level(data))

        # State reset in on_set_level; initialised here so type hints are happy
        self._history: list[tuple[
            tuple[int, int],    # shepherd pos (camera coords)
            list[tuple[int, int]],  # sheep positions
            list[bool],         # gate open states
            list[bool],         # key collected states
        ]] = []
        self._gate_open:     list[bool] = []
        self._key_collected: list[bool] = []

        self._init_augmentation(seed)

        super().__init__(
            game_id="sheep_herder",
            levels=levels,
            camera=Camera(
                background=C_GRASS,
                letter_box=C_BORDER,
                interfaces=[self._step_counter],
            ),
            available_actions=[1, 2, 3, 4, 5, 7],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _shepherd(self) -> Sprite:
        return self.current_level.get_sprites_by_tag("shepherd")[0]

    def _sheep_list(self) -> list[Sprite]:
        return self.current_level.get_sprites_by_tag("sheep")

    def _sheep_at(self, x: int, y: int) -> Sprite | None:
        return next((s for s in self._sheep_list() if s.x == x and s.y == y), None)

    def _gate_at(self, x: int, y: int) -> Sprite | None:
        return next(
            (s for s in self.current_level.get_sprites_by_tag("gate")
             if s.x == x and s.y == y),
            None,
        )

    def _is_blocked(self, x: int, y: int) -> bool:
        """True if (x, y) is impassable (fence, tree, or closed gate)."""
        for s in self.current_level.get_sprites():
            if s.x == x and s.y == y:
                if "fence" in s.tags or "tree" in s.tags:
                    return True
                if "gate" in s.tags and "gate_closed" in s.tags:
                    return True
        return False

    def _pen_for_cell(self, x: int, y: int) -> int | None:
        """Return pen index if (x,y) is a pen interior cell, else None."""
        for s in self.current_level.get_sprites():
            if s.x == x and s.y == y:
                for tag in s.tags:
                    if tag.startswith("pen_") and tag != "pen":
                        try:
                            return int(tag.split("_")[1])
                        except (IndexError, ValueError):
                            pass
        return None

    def _snapshot(self):
        sh       = self._shepherd()
        sheep_p  = [(s.x, s.y) for s in self._sheep_list()]
        return (sh.x, sh.y), sheep_p, list(self._gate_open), list(self._key_collected)

    def _set_gate(self, gate_sprite: Sprite, open_state: bool) -> None:
        gate_i = int(gate_sprite.name.split("_")[1])
        self._gate_open[gate_i] = open_state
        if open_state:
            gate_sprite.set_collidable(False)
            gate_sprite.pixels = np.array(_gate_pixels(True), dtype=np.int8)
            gate_sprite._tags[:] = [t for t in gate_sprite.tags if t != "gate_closed"]
            if "gate_open" not in gate_sprite.tags:
                gate_sprite._tags.append("gate_open")
        else:
            gate_sprite.set_collidable(True)
            gate_sprite.pixels = np.array(_gate_pixels(False), dtype=np.int8)
            gate_sprite._tags[:] = [t for t in gate_sprite.tags if t != "gate_open"]
            if "gate_closed" not in gate_sprite.tags:
                gate_sprite._tags.append("gate_closed")

    def _check_win(self) -> bool:
        for i, sheep in enumerate(self._sheep_list()):
            if self._pen_for_cell(sheep.x, sheep.y) != i:
                return False
        return True

    def _collect_key_at(self, px: int, py: int) -> None:
        """If a key is at (px, py), collect it and open the matching gate."""
        for sprite in list(self.current_level.get_sprites_by_tag("key")):
            if sprite.x == px and sprite.y == py:
                key_i = int(sprite.name.split("_")[1])
                self.current_level.remove_sprite(sprite)
                self._key_collected[key_i] = True
                gate_sprites = self.current_level.get_sprites_by_tag(f"gate_{key_i}")
                if gate_sprites:
                    self._set_gate(gate_sprites[0], open_state=True)
                break

    # ------------------------------------------------------------------
    # on_set_level
    # ------------------------------------------------------------------

    def on_set_level(self, level: Level) -> None:
        self._history.clear()
        self._step_counter.reset(_STEP_BUDGETS[self._current_level_index])
        data             = self._level_data[self._current_level_index]
        num_gates        = len(data["gate_positions"])
        gates_closed     = data["gates_start_closed"]

        self._gate_open     = [not gates_closed] * num_gates
        self._key_collected = [not gates_closed] * num_gates

        # Idempotency: clear dynamic sprites from previous call
        for sprite in list(level.get_sprites()):
            if any(t in sprite.tags for t in ("shepherd", "sheep", "key")):
                level.remove_sprite(sprite)

        # Reset gate sprites to initial state
        for gate_sprite in level.get_sprites_by_tag("gate"):
            self._set_gate(gate_sprite, not gates_closed)

        # Add shepherd
        sx, sy = data["shepherd"]
        level.add_sprite(
            Sprite(pixels=_SHEPHERD_PIXELS, name="shepherd",
                   collidable=False, tags=["shepherd"], layer=3)
            .set_position(sx * CELL, sy * CELL)
        )

        # Add sheep
        for i, ((ox, oy), color) in enumerate(
            zip(data["sheep_positions"], data["sheep_colors"])
        ):
            level.add_sprite(
                Sprite(pixels=_sheep_pixels(color), name=f"sheep_{i}",
                       collidable=False, tags=["sheep", f"sheep_{i}"], layer=2)
                .set_position(ox * CELL, oy * CELL)
            )

        # Add keys (levels with locked gates)
        if gates_closed:
            for i, (kx, ky) in enumerate(data["key_positions"]):
                level.add_sprite(
                    Sprite(pixels=_key_pixels(data["sheep_colors"][i]),
                           name=f"key_{i}",
                           collidable=False, tags=["key", f"key_{i}"], layer=1)
                    .set_position(kx * CELL, ky * CELL)
                )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    _DELTAS: dict[int, tuple[int, int]] = {
        GameAction.ACTION1: (0,   -CELL),
        GameAction.ACTION2: (0,    CELL),
        GameAction.ACTION3: (-CELL, 0  ),
        GameAction.ACTION4: ( CELL, 0  ),
    }

    def step(self) -> None:
        self._step_counter.decrement()
        _action_id = self.screen_action_to_game(self.action.id)

        # --- Undo -------------------------------------------------------
        if _action_id == GameAction.ACTION7:
            if self._history:
                (sx, sy), sheep_pos, gate_states, key_states = self._history.pop()
                self._shepherd().set_position(sx, sy)
                for sheep, (ox, oy) in zip(self._sheep_list(), sheep_pos):
                    sheep.set_position(ox, oy)
                # Restore gates
                for gate_sprite in self.current_level.get_sprites_by_tag("gate"):
                    gate_i = int(gate_sprite.name.split("_")[1])
                    self._gate_open = gate_states
                    self._set_gate(gate_sprite, gate_states[gate_i])
                # Restore keys: remove current, re-add any that were uncollected
                for s in list(self.current_level.get_sprites_by_tag("key")):
                    self.current_level.remove_sprite(s)
                self._key_collected = key_states
                data = self._level_data[self._current_level_index]
                if data["gates_start_closed"]:
                    for i, collected in enumerate(self._key_collected):
                        if not collected:
                            kx, ky = data["key_positions"][i]
                            self.current_level.add_sprite(
                                Sprite(pixels=_key_pixels(data["sheep_colors"][i]),
                                       name=f"key_{i}",
                                       collidable=False,
                                       tags=["key", f"key_{i}"], layer=1)
                                .set_position(kx * CELL, ky * CELL)
                            )
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        # --- Toggle gate (ACTION5) --------------------------------------
        if _action_id == GameAction.ACTION5:
            shepherd = self._shepherd()
            gate = self._gate_at(shepherd.x, shepherd.y)
            if gate is not None:
                gate_i = int(gate.name.split("_")[1])
                # Only toggle if gate has been unlocked by its key
                if self._key_collected[gate_i]:
                    self._history.append(self._snapshot())
                    self._set_gate(gate, not self._gate_open[gate_i])
                    if self._check_win():
                        self.next_level()
                        self.complete_action()
                        return
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        # --- Move shepherd (with push) ----------------------------------
        delta = self._DELTAS.get(_action_id)
        if delta is None:
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        dx, dy    = delta
        shepherd  = self._shepherd()
        nx, ny    = shepherd.x + dx, shepherd.y + dy

        if self._is_blocked(nx, ny):
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        sheep = self._sheep_at(nx, ny)
        if sheep is not None:
            snx, sny = sheep.x + dx, sheep.y + dy
            if self._is_blocked(snx, sny) or self._sheep_at(snx, sny) is not None:
                # Cannot push sheep
                self.complete_action()
                if not self._step_counter.steps_remaining:
                    self.lose()
                return
            self._history.append(self._snapshot())
            sheep.set_position(snx, sny)
            shepherd.set_position(nx, ny)
        else:
            self._history.append(self._snapshot())
            shepherd.set_position(nx, ny)

        # Collect key if shepherd stepped onto one
        self._collect_key_at(shepherd.x, shepherd.y)

        if self._check_win():
            self.next_level()
            self.complete_action()
            return

        if not self._step_counter.steps_remaining:
            self.lose()
        self.complete_action()
