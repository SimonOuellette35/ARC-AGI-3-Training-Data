"""
Symmetric Cursor Maze
=====================
Four cursors are positioned symmetrically around the maze center.
When you press a direction, all four cursors move simultaneously:
  - Cursor 0: moves as directed
  - Cursor 1: mirrors the X axis (left/right is flipped)
  - Cursor 2: mirrors the Y axis (up/down is flipped)
  - Cursor 3: mirrors both axes

The maze is built to be perfectly symmetric so each cursor always has
a mirrored valid path. Use ACTION5 to toggle single-cursor mode for
fine adjustments.

Goal: maneuver all four cursors to the exact same cell simultaneously.

Actions:
  ACTION1 (↑) — move up (mirrored for each cursor)
  ACTION2 (↓) — move down
  ACTION3 (←) — move left
  ACTION4 (→) — move right
  ACTION5     — toggle single-cursor mode
  ACTION7     — undo last move

Color key:
  2  (#3399FF) — floor
  0  (#FFFFFF) — wall
  5  (#000000) — letterbox border
  14 (#4FCC30) — goal marker (center)

HUD (bottom row): the step-budget bar on the left, and on the right four pips —
one per cursor, in that cursor's colour — showing the CONTROL MODE:
  * all four pips coloured, one of them colour 3 -> all-cursor mode; the colour-3
    pip is the cursor ACTION5 would hand you next;
  * one pip coloured, the other three dark -> single-cursor mode, and the
    coloured pip is the cursor you are currently steering.
Without this the mode and the selected cursor are invisible, so a player reading
only the frame cannot tell whether an arrow key moves one cursor or all four.

Win condition : All four cursors occupy the same cell.
Lose condition: Step budget exhausted.
Levels        : 7 levels with larger mazes.
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

C_FLOOR   = 2
C_WALL    = 0
C_BORDER  = 5
C_GOAL    = 14
C_CURSOR  = [8, 9, 11, 12]
C_DIM     = 4          # HUD: spent budget / a cursor that ignores the arrow keys
C_FILL    = 7          # HUD: remaining budget
C_NEXT    = 3          # HUD: the cursor ACTION5 would select next

CELL = 3

# HUD row 63 layout. The grid never reaches row 63 (the widest level, 15*3 = 45px,
# letterboxes; the narrowest, 7*3 = 21px, scales x3 to 63px and so ends at row 62),
# so the whole row is free for the interface.
_BAR_W  = 44           # step-budget bar: x in [0, 44)
_PIP_X0 = 46           # four mode pips: x in [46, 62)
_PIP_W  = 4

_FLOOR_PIX = [[C_FLOOR]*CELL]*CELL
_WALL_PIX  = [[C_WALL]*CELL]*CELL
_GOAL_PIX  = [
    [-1,    C_GOAL, -1   ],
    [C_GOAL, -1,   C_GOAL],
    [-1,    C_GOAL, -1   ],
]

def _cursor_pix(color: int) -> list:
    return [
        [-1,    color, -1   ],
        [color, color, color],
        [-1,    color, -1   ],
    ]

# (grid_w, grid_h) — must be ODD for maze carving
_CONFIGS = [
    ( 7,  7),
    ( 9,  9),
    (11, 11),
    (13, 13),
    (13, 13),
    (15, 15),
    (15, 15),
]

_STEP_BUDGETS = [40, 60, 90, 120, 160, 200, 260]

_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

_MIRROR_DIRS = {
    GameAction.ACTION1: [(0, -1), (0, -1), (0, 1),  (0, 1) ],
    GameAction.ACTION2: [(0,  1), (0,  1), (0, -1), (0, -1)],
    GameAction.ACTION3: [(-1, 0), (1, 0),  (-1, 0), (1, 0) ],
    GameAction.ACTION4: [(1,  0), (-1, 0), (1,  0), (-1, 0)],
}

_OPPOSITE_ACTION = {
    GameAction.ACTION1: GameAction.ACTION2,
    GameAction.ACTION2: GameAction.ACTION1,
    GameAction.ACTION3: GameAction.ACTION4,
    GameAction.ACTION4: GameAction.ACTION3,
}

# Nominal (cursor-0) direction -> the action that produces it.
_ACTION_BY_DIR = {_MIRROR_DIRS[a][0]: a for a in _MIRROR_DIRS}


class StepCounter(RenderableUserDisplay):
    """Bottom-row HUD: the step-budget bar plus the control-mode pips.

    The mode pips exist because ``_single_mode`` / ``_active_cursor`` are otherwise
    invisible: nothing in the maze itself says whether an arrow key moves one
    cursor or all four, nor which cursor ACTION5 hands over next, so a player who
    reads only the frame cannot plan. The game pushes its state in through
    `set_mode` (no back-reference to the game, so the interface stays cheap to
    deep-copy)."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.steps_remaining = max_steps
        self.single_mode = False
        self.active_cursor = 0

    def decrement(self) -> None:
        if self.steps_remaining > 0:
            self.steps_remaining -= 1

    def reset(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.steps_remaining = max_steps

    def set_mode(self, single_mode: bool, active_cursor: int) -> None:
        self.single_mode = bool(single_mode)
        self.active_cursor = int(active_cursor)

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.max_steps:
            filled = round(_BAR_W * self.steps_remaining / self.max_steps)
            for x in range(_BAR_W):
                frame[63, x] = C_FILL if x < filled else C_DIM
        nxt = (self.active_cursor + 1) % 4
        for i in range(4):
            if self.single_mode:
                col = C_CURSOR[i] if i == self.active_cursor else C_DIM
            else:
                col = C_NEXT if i == nxt else C_CURSOR[i]
            x0 = _PIP_X0 + i * _PIP_W
            frame[63, x0:x0 + _PIP_W] = col
        return frame


def _generate_symmetric_maze(w: int, h: int, rng: random.Random) -> list[list[int]]:
    """
    Generate a maze that is symmetric around both axes.
    Start with a standard recursive-DFS maze, then force symmetry
    (union of a cell and its 3 mirrors: if any is floor, all become floor).
    """
    assert w % 2 == 1 and h % 2 == 1
    cols, rows = (w - 1) // 2, (h - 1) // 2

    # Full recursive DFS maze
    grid = [[1] * w for _ in range(h)]

    def carve(cx: int, cy: int) -> None:
        grid[2 * cy + 1][2 * cx + 1] = 0
        dirs = list(_DIRS)
        rng.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cols and 0 <= ny < rows and grid[2 * ny + 1][2 * nx + 1] == 1:
                grid[2 * cy + 1 + dy][2 * cx + 1 + dx] = 0
                carve(nx, ny)

    carve(0, 0)

    # Force symmetry: if any mirror is floor, make all floor
    for y in range(h // 2 + 1):
        for x in range(w // 2 + 1):
            mirror_positions = [
                (y, x), (y, w - 1 - x),
                (h - 1 - y, x), (h - 1 - y, w - 1 - x),
            ]
            v = min(grid[ry][rx] for ry, rx in mirror_positions)
            for ry, rx in mirror_positions:
                grid[ry][rx] = v

    # Ensure border is walls
    for y in range(h):
        for x in range(w):
            if x == 0 or x == w - 1 or y == 0 or y == h - 1:
                grid[y][x] = 1

    # Ensure center is floor (the goal meeting point)
    cx, cy = w // 2, h // 2
    grid[cy][cx] = 0

    # ...and that it is REACHABLE. When w (== h) is 9 or 13 the center lands on an
    # even coordinate, i.e. on the wall lattice between maze cells rather than on a
    # carved cell, so force-flooring it can leave an isolated one-cell island: the
    # goal is then unreachable and the level starts with all four cursors already
    # stacked on it. Opening the center's left+right neighbours fixes that while
    # preserving symmetry (they are each other's X mirror, and each is its own Y
    # mirror), and each is orthogonally adjacent to two carved cell centers, which
    # are always floor -- so one floor neighbour is enough to join the whole maze.
    if all(grid[cy + dy][cx + dx] for dx, dy in _DIRS):
        grid[cy][cx - 1] = 0
        grid[cy][cx + 1] = 0

    return grid


def _dist_from_center(grid: list[list[int]], w: int, h: int) -> dict:
    """BFS distance from the center cell to every reachable floor cell.

    Forcing symmetry only ever turns walls into floor, so the carved maze's
    connectivity survives and this reaches the whole floor.
    """
    cx, cy = w // 2, h // 2
    dist = {(cx, cy): 0}
    queue = deque([(cx, cy)])
    while queue:
        x, y = queue.popleft()
        for dx, dy in _DIRS:
            nx, ny = x + dx, y + dy
            if (0 <= nx < w and 0 <= ny < h and grid[ny][nx] == 0
                    and (nx, ny) not in dist):
                dist[(nx, ny)] = dist[(x, y)] + 1
                queue.append((nx, ny))
    return dist


def _generate(w: int, h: int, rng: random.Random):
    """
    Generate a symmetric maze level.

    Goal state: all 4 cursors at center (cx, cy). The cursors START far from it,
    at a cell drawn from the outer band of the distance-to-center field, and the
    other three follow by mirror symmetry.

    START FAR, NOT SCRAMBLED. This used to random-walk cursor 0 away from the
    center for ``budget // 2 - 1`` moves and take the reverse as the solution --
    but a random walk in a maze doubles back, so the start landed a mean of 2.7
    (7x7) to 7.0 (15x15) cells from the goal against eccentricities of 4.8 to
    12.3, i.e. every level collapsed to a handful of moves and the seven maze
    sizes barely ramped. One seed even started ON the center, making level 5
    vacuous. Sampling the outer distance band instead puts the optimal solution
    near the maze's eccentricity, keeps per-seed variety, and guarantees the
    level does not start solved.
    """
    grid = _generate_symmetric_maze(w, h, rng)
    cx, cy = w // 2, h // 2

    dist = _dist_from_center(grid, w, h)
    far = max(dist.values())
    threshold = max(2, (3 * far + 3) // 4)
    # Sorted so the candidate order (and therefore the level) is a pure function
    # of the seed, independent of dict/hash iteration order.
    cands = sorted(p for p, d in dist.items() if d >= threshold)
    if not cands:                                  # degenerate tiny maze
        cands = sorted(p for p, d in dist.items() if d == far)
    px, py = cands[rng.randrange(len(cands))]

    starts = [
        (px,         py        ),  # cursor 0
        (w - 1 - px, py        ),  # cursor 1: mirror X
        (px,         h - 1 - py),  # cursor 2: mirror Y
        (w - 1 - px, h - 1 - py),  # cursor 3: mirror both
    ]

    # One optimal all-cursor route, kept for reference/inspection only (the
    # cursors stay mirror images under all-cursor moves, so cursor 0 reaching the
    # center means all four meet there). Nothing in the game reads it, and a
    # solver must NOT: it is the stashed answer.
    solution_steps: list[GameAction] = []
    x, y = px, py
    while (x, y) != (cx, cy):
        for dx, dy in _DIRS:
            step = (x + dx, y + dy)
            if dist.get(step, 1 << 20) == dist[(x, y)] - 1:
                solution_steps.append(_ACTION_BY_DIR[(dx, dy)])
                x, y = step
                break

    return grid, starts, (cx, cy), solution_steps


def _build_level(grid: list[list[int]], cx: int, cy: int, w: int, h: int) -> Level:
    sprites = []
    for y in range(h):
        for x in range(w):
            pix = _WALL_PIX if grid[y][x] else _FLOOR_PIX
            name = f"wall_{x}_{y}" if grid[y][x] else f"floor_{x}_{y}"
            sprites.append(
                Sprite(pixels=pix, name=name,
                       collidable=(grid[y][x] != 0), layer=-1)
                .set_position(x * CELL, y * CELL)
            )
    # Goal marker at center
    sprites.append(
        Sprite(pixels=_GOAL_PIX, name="goal",
               collidable=False, tags=["goal"], layer=0)
        .set_position(cx * CELL, cy * CELL)
    )
    return Level(sprites=sprites, grid_size=(w * CELL, h * CELL))


class SymmetricMaze(AugmentedGame):

    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: the base draws this level's rotation
        self._rng = random.Random(seed)
        self._step_counter = StepCounter(_STEP_BUDGETS[0])

        self._level_data = []
        levels = []

        for w, h in _CONFIGS:
            grid, starts, goal, solution_steps = _generate(w, h, self._rng)
            levels.append(_build_level(grid, goal[0], goal[1], w, h))
            self._level_data.append((grid, starts, goal, w, h, solution_steps))

        self._history = []
        self._single_mode = False
        self._active_cursor = 0

        super().__init__(
            game_id="symmetric_maze",
            levels=levels,
            camera=Camera(background=C_FLOOR, letter_box=C_BORDER,
                          interfaces=[self._step_counter]),
            available_actions=[1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self._history.clear()
        self._single_mode = False
        self._active_cursor = 0
        idx = self._current_level_index
        self._step_counter.reset(_STEP_BUDGETS[idx])

        for s in list(level.get_sprites()):
            if "cursor" in s.tags:
                level.remove_sprite(s)

        _, starts, _, _, _, _ = self._level_data[idx]
        for i, (sx, sy) in enumerate(starts):
            level.add_sprite(
                Sprite(pixels=_cursor_pix(C_CURSOR[i]), name=f"cursor_{i}",
                       collidable=False, tags=["cursor", f"ci_{i}"], layer=2)
                .set_position(sx * CELL, sy * CELL)
            )
        self._sync_hud()

    def _sync_hud(self) -> None:
        """Publish the control mode to the HUD, so which cursors respond to the
        arrow keys is readable from the frame."""
        self._step_counter.set_mode(self._single_mode, self._active_cursor)

    def _cursor(self, i: int) -> Sprite:
        return self.current_level.get_sprites_by_name(f"cursor_{i}")[0]

    def _wall_at(self, x: int, y: int) -> bool:
        grid, _, _, w, h, _ = self._level_data[self._current_level_index]
        if not (0 <= x < w and 0 <= y < h):
            return True
        return grid[y][x] != 0

    def _snapshot(self) -> tuple:
        return (self._single_mode, self._active_cursor,
                [(self._cursor(i).x, self._cursor(i).y) for i in range(4)])

    def _check_win(self) -> bool:
        positions = [(self._cursor(i).x, self._cursor(i).y) for i in range(4)]
        return len(set(positions)) == 1

    def step(self) -> None:
        self._step_counter.decrement()
        action = self.action.id

        if action == GameAction.ACTION7:
            if self._history:
                sm, ac, positions = self._history.pop()
                self._single_mode = sm
                self._active_cursor = ac
                for i, (px, py) in enumerate(positions):
                    self._cursor(i).set_position(px, py)
                self._sync_hud()
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        if action == GameAction.ACTION5:
            self._history.append(self._snapshot())
            if self._single_mode:
                self._single_mode = False
            else:
                self._single_mode = True
                self._active_cursor = (self._active_cursor + 1) % 4
            self._sync_hud()
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        if action not in _MIRROR_DIRS:
            self.complete_action()
            return

        snap = self._snapshot()
        self._history.append(snap)

        if self._single_mode:
            dx, dy = _MIRROR_DIRS[action][self._active_cursor]
            c = self._cursor(self._active_cursor)
            nx, ny = c.x // CELL + dx, c.y // CELL + dy
            if not self._wall_at(nx, ny):
                c.set_position(nx * CELL, ny * CELL)
        else:
            for i in range(4):
                dx, dy = _MIRROR_DIRS[action][i]
                c = self._cursor(i)
                nx, ny = c.x // CELL + dx, c.y // CELL + dy
                if not self._wall_at(nx, ny):
                    c.set_position(nx * CELL, ny * CELL)

        if self._check_win():
            self.next_level()
            self.complete_action()
            return

        if not self._step_counter.steps_remaining:
            self.lose()

        self.complete_action()
