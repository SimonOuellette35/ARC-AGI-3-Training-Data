"""
Paired Piece Mover
==================
Move animal pieces to their matching goal positions. Some pieces are PAIRED:
moving a paired piece also simultaneously moves its partner by the same offset.
Wall tiles block movement; a move is blocked if either the active piece or its
partner (if paired) would enter a wall or go off-board.

Actions:
  ACTION1 (↑) — move active piece up
  ACTION2 (↓) — move active piece down
  ACTION3 (←) — move active piece left
  ACTION4 (→) — move active piece right
  ACTION5     — cycle to next piece
  ACTION7     — undo last move

Color key:
  0  (#FFFFFF) — floor
  4  (#333333) — wall
  9  (#1E93FF) — active piece (blue highlight)
  8  (#F93C31) — inactive piece (red)
  14 (#4FCC30) — goal for each piece (green)
  3  (#4FCC30) — piece on its goal (teal)
  5  (#000000) — letterbox border

Win condition : All pieces are on their goal positions simultaneously.
Lose condition: Step budget exhausted.
Levels        : 7 levels with more pieces and larger grids.
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
# Colors
# ---------------------------------------------------------------------------
C_FLOOR    = 0   # white floor
C_WALL     = 4   # dark wall
C_ACTIVE   = 9   # blue — active piece
C_INACTIVE = 8   # red — inactive piece
C_ON_GOAL  = 3   # teal — piece on goal
C_GOAL     = 14  # green goal marker
C_BORDER   = 5   # black

CELL = 3

_RANDOM_COLOR_POOL = [c for c in range(1, 15) if c not in {C_WALL, C_BORDER}]

_WALL_PIX = [[C_WALL]*CELL]*CELL
_FLOOR_PIX = [[C_FLOOR]*CELL]*CELL

_PIECE_SHAPES = [
    [
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ],
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
]

_GOAL_SHAPES = [
    [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ],
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ],
    [
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ],
    [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ],
]


def _shape_pix(shape: list[list[int]], color: int) -> list[list[int]]:
    return [[color if cell else -1 for cell in row] for row in shape]


def _piece_pix(color: int, shape: list[list[int]]) -> list[list[int]]:
    return _shape_pix(shape, color)


def _goal_pix(color: int, shape: list[list[int]]) -> list[list[int]]:
    return _shape_pix(shape, color)


def _level_palette(rng: random.Random) -> dict[str, object]:
    colors = list(_RANDOM_COLOR_POOL)
    rng.shuffle(colors)
    active = colors.pop()
    inactive = colors.pop()
    goal = colors.pop()
    on_goal = colors.pop()
    piece_shape = rng.choice(_PIECE_SHAPES)
    goal_shape = rng.choice(_GOAL_SHAPES)
    return {
        "active": active,
        "inactive": inactive,
        "goal": goal,
        "on_goal": on_goal,
        "piece_shape": piece_shape,
        "goal_shape": goal_shape,
    }

# (grid_w, grid_h, n_singles, n_pairs, n_walls, scramble)
# scramble ≤ budget // n_pieces − 1 to ensure the reverse-scramble solution fits in budget
_CONFIGS = [
    ( 7,  7, 2, 0,  3, 15),   # n_pieces=2,  budget=50  → max=24
    ( 9,  9, 2, 1,  6, 17),   # n_pieces=4,  budget=80  → max=19
    ( 9,  9, 1, 2,  8, 21),   # n_pieces=5,  budget=120 → max=23
    (11, 11, 2, 2, 10, 23),   # n_pieces=6,  budget=160 → max=25
    (11, 11, 1, 3, 12, 28),   # n_pieces=7,  budget=220 → max=30
    (13, 13, 2, 3, 14, 32),   # n_pieces=8,  budget=280 → max=34
    (13, 13, 2, 4, 16, 33),   # n_pieces=10, budget=360 → max=35
]

_STEP_BUDGETS = [50, 80, 120, 160, 220, 280, 360]

_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
_DELTA_TO_ACTION = {
    (0, -1): GameAction.ACTION1,
    (0, 1):  GameAction.ACTION2,
    (-1, 0): GameAction.ACTION3,
    (1, 0):  GameAction.ACTION4,
}


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
# Level generator
# ---------------------------------------------------------------------------

def _reachable(grid, start, w, h, blocked=None):
    if blocked is None:
        blocked = set()
    if not (0 <= start[0] < w and 0 <= start[1] < h) or not grid[start[1]][start[0]]:
        return set()
    visited = {start}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in _DIRS:
            nb = (x+dx, y+dy)
            if nb not in visited and 0 <= nb[0] < w and 0 <= nb[1] < h:
                if grid[nb[1]][nb[0]] and nb not in blocked:
                    visited.add(nb)
                    q.append(nb)
    return visited


def _add_walls(grid, w, h, n_walls, rng):
    candidates = [(x, y) for y in range(2, h-2) for x in range(2, w-2)]
    rng.shuffle(candidates)
    for x, y in candidates:
        if n_walls <= 0:
            break
        grid[y][x] = 0
        floor = [(fx, fy) for fy in range(1, h-1) for fx in range(1, w-1) if grid[fy][fx]]
        if not floor:
            grid[y][x] = 1
            continue
        vis = _reachable(grid, floor[0], w, h)
        if len(vis) == len(floor):
            n_walls -= 1
        else:
            grid[y][x] = 1


def _generate(w, h, n_singles, n_pairs, n_walls, scramble, rng):
    grid = [[0]*w for _ in range(h)]
    for y in range(1, h-1):
        for x in range(1, w-1):
            grid[y][x] = 1

    _add_walls(grid, w, h, n_walls, rng)

    floor = [(x, y) for y in range(1, h-1) for x in range(1, w-1) if grid[y][x]]
    rng.shuffle(floor)

    n_pieces = n_singles + 2 * n_pairs
    positions = []
    goals = []
    pairs = []  # list of (i, j) indices indicating paired pieces

    # Assign positions and goals
    used = set()
    for i in range(n_pieces):
        while floor:
            p = floor.pop()
            if p not in used:
                positions.append(p)
                used.add(p)
                break
        while floor:
            g = floor.pop()
            if g not in used:
                goals.append(g)
                used.add(g)
                break

    if len(positions) < n_pieces:
        raise ValueError("Not enough floor cells")

    # Set up pairs: pair (i, i+1) for i in range(n_singles, n_pieces, 2)
    for i in range(n_singles, n_pieces, 2):
        if i+1 < n_pieces:
            pairs.append((i, i+1))

    # Scramble by doing random reverse moves from goal state.
    # Record each actual move so the solver can reverse it directly.
    cur_pos = list(goals[:n_pieces])  # start from solved state
    scramble_log = []  # list of (pi, ddx, ddy) moves actually applied
    for _ in range(scramble):
        # Pick random piece
        pi = rng.randrange(n_pieces)
        # Find paired partner
        partner = None
        for pa, pb in pairs:
            if pa == pi:
                partner = pb
                break
            elif pb == pi:
                partner = pa
                break

        shuffled_dirs = list(_DIRS)
        rng.shuffle(shuffled_dirs)
        for ddx, ddy in shuffled_dirs:
            nx, ny = cur_pos[pi][0]+ddx, cur_pos[pi][1]+ddy
            blocked = not (0 <= nx < w and 0 <= ny < h and grid[ny][nx])
            if partner is not None:
                pnx, pny = cur_pos[partner][0]+ddx, cur_pos[partner][1]+ddy
                if not (0 <= pnx < w and 0 <= pny < h and grid[pny][pnx]):
                    blocked = True

            if not blocked:
                cur_pos[pi] = (nx, ny)
                if partner is not None:
                    cur_pos[partner] = (pnx, pny)
                scramble_log.append((pi, ddx, ddy))
                break

    # Solution = reverse of scramble: (pi, -dx, -dy) in reversed order
    solution_steps = [(pi, -dx, -dy) for pi, dx, dy in reversed(scramble_log)]

    return grid, cur_pos, goals, pairs, solution_steps


def _build_level(grid, goals, w, h, goal_color: int, goal_shape: list[list[int]]) -> Level:
    sprites = []
    for y in range(h):
        for x in range(w):
            if grid[y][x]:
                sprites.append(
                    Sprite(pixels=_FLOOR_PIX, name=f"floor_{x}_{y}",
                           collidable=False, layer=-2)
                    .set_position(x*CELL, y*CELL)
                )
            else:
                sprites.append(
                    Sprite(pixels=_WALL_PIX, name=f"wall_{x}_{y}",
                           collidable=True, layer=-2)
                    .set_position(x*CELL, y*CELL)
                )
    for i, (gx, gy) in enumerate(goals):
        sprites.append(
            Sprite(pixels=_goal_pix(goal_color, goal_shape), name=f"goal_{i}",
                   collidable=False, tags=["goal"], layer=-1)
            .set_position(gx*CELL, gy*CELL)
        )
    return Level(sprites=sprites, grid_size=(w*CELL, h*CELL))


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class PairedPieces(AugmentedGame):

    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: the base rotates the board on set_level
        self._rng = random.Random(seed)
        self._step_counter = StepCounter(_STEP_BUDGETS[0])

        self._level_data = []
        levels = []

        for cfg_idx, (w, h, n_singles, n_pairs, n_walls, scramble) in enumerate(_CONFIGS):
            grid, positions, goals, pairs, solution_steps = _generate(
                w, h, n_singles, n_pairs, n_walls, scramble, self._rng
            )
            palette = _level_palette(self._rng)
            levels.append(_build_level(grid, goals, w, h, palette["goal"], palette["goal_shape"]))
            self._level_data.append((grid, positions, goals, pairs, w, h, solution_steps, palette))

        self._history = []
        self._active = 0

        super().__init__(
            game_id="paired_pieces",
            levels=levels,
            camera=Camera(background=C_FLOOR, letter_box=C_BORDER,
                          interfaces=[self._step_counter]),
            available_actions=[1, 2, 3, 4, 5, 7],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _n_pieces(self) -> int:
        return len(self.current_level.get_sprites_by_tag("piece"))

    def _piece(self, i: int) -> Sprite:
        return self.current_level.get_sprites_by_name(f"piece_{i}")[0]

    def _idx_data(self):
        idx = self._current_level_index
        return self._level_data[idx]

    def _wall_at(self, x, y) -> bool:
        grid, _, _, _, w, h, _, _ = self._idx_data()
        if not (0 <= x < w and 0 <= y < h):
            return True
        return grid[y][x] == 0

    def _is_on_goal(self, i: int) -> bool:
        _, _, goals, _, _, _, _, _ = self._idx_data()
        p = self._piece(i)
        gx, gy = goals[i]
        return p.x == gx*CELL and p.y == gy*CELL

    def _update_piece_visuals(self) -> None:
        _, positions, _, _, _, _, _, palette = self._idx_data()
        n = len(positions)
        piece_shape = palette["piece_shape"]
        for i in range(n):
            p = self._piece(i)
            if i == self._active:
                p.pixels = np.array(_piece_pix(palette["active"], piece_shape), dtype=np.int8)
            elif self._is_on_goal(i):
                p.pixels = np.array(_piece_pix(palette["on_goal"], piece_shape), dtype=np.int8)
            else:
                p.pixels = np.array(_piece_pix(palette["inactive"], piece_shape), dtype=np.int8)

    def _snapshot(self):
        n = len(self._level_data[self._current_level_index][1])
        return (self._active, [(self._piece(i).x, self._piece(i).y) for i in range(n)])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_set_level(self, level: Level) -> None:
        self._history.clear()
        idx = self._current_level_index
        self._step_counter.reset(_STEP_BUDGETS[idx])
        self._active = 0

        for s in level.get_sprites():
            if "piece" in s.tags:
                level.remove_sprite(s)

        _, positions, goals, pairs, w, h, _, palette = self._level_data[idx]
        n = len(positions)
        piece_shape = palette["piece_shape"]

        for i, (px, py) in enumerate(positions):
            pix = (
                _piece_pix(palette["active"], piece_shape)
                if i == 0
                else _piece_pix(palette["inactive"], piece_shape)
            )
            level.add_sprite(
                Sprite(pixels=pix, name=f"piece_{i}",
                       collidable=False, tags=["piece"], layer=2)
                .set_position(px*CELL, py*CELL)
            )

        # Apply the on-goal colouring to the OPENING frame too. Without this a
        # piece that starts on its own goal is drawn in the inactive colour and
        # only turns C_ON_GOAL after the first move, so the first frame of every
        # level (and of every RESET) contradicts the rule the rest of the episode
        # follows -- and any agent reading the on-goal indicator is misled.
        self._update_piece_visuals()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    _DELTAS = {
        GameAction.ACTION1: (0, -1),
        GameAction.ACTION2: (0,  1),
        GameAction.ACTION3: (-1, 0),
        GameAction.ACTION4: (1,  0),
    }

    def step(self) -> None:
        self._step_counter.decrement()
        action = self.action.id
        _, positions, goals, pairs, w, h, _, _ = self._idx_data()
        n = len(positions)

        if action == GameAction.ACTION7:
            if self._history:
                saved_active, saved_pos = self._history.pop()
                self._active = saved_active
                for i, (px, py) in enumerate(saved_pos):
                    self._piece(i).set_position(px, py)
                self._update_piece_visuals()
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        if action == GameAction.ACTION5:
            self._active = (self._active + 1) % n
            self._update_piece_visuals()
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        dx, dy = self._DELTAS.get(action, (0, 0))
        if not (dx or dy):
            self.complete_action()
            return

        # Find partner
        partner = None
        for pa, pb in pairs:
            if pa == self._active:
                partner = pb
                break
            elif pb == self._active:
                partner = pa
                break

        piece = self._piece(self._active)
        cx, cy = piece.x // CELL, piece.y // CELL
        nx, ny = cx + dx, cy + dy

        if self._wall_at(nx, ny):
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        # Check partner
        if partner is not None:
            pp = self._piece(partner)
            pcx, pcy = pp.x // CELL, pp.y // CELL
            pnx, pny = pcx + dx, pcy + dy
            if self._wall_at(pnx, pny):
                self.complete_action()
                if not self._step_counter.steps_remaining:
                    self.lose()
                return

        snap = self._snapshot()
        self._history.append(snap)

        piece.set_position(nx*CELL, ny*CELL)
        if partner is not None:
            pp = self._piece(partner)
            pcx, pcy = pp.x // CELL, pp.y // CELL
            pp.set_position((pcx+dx)*CELL, (pcy+dy)*CELL)

        self._update_piece_visuals()

        # Win check
        if all(self._is_on_goal(i) for i in range(n)):
            self.next_level()
            self.complete_action()
            return

        if not self._step_counter.steps_remaining:
            self.lose()

        self.complete_action()
