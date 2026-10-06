"""
Projectile Dodge
================
A grid-based game where you move player pieces to their exit positions
while avoiding moving projectiles.

The board has passable cells and wall cells. One or more player pieces
start at initial positions; each must reach its matching exit cell.
Projectiles travel in fixed directions (up/down/left/right), advancing
one cell each turn after every player action. A projectile that moves
into a player piece destroys that piece — game over.

Actions:
  ACTION1 (↑) — move all player pieces up (if passable)
  ACTION2 (↓) — move all player pieces down
  ACTION3 (←) — move all player pieces left
  ACTION4 (→) — move all player pieces right
  ACTION7     — undo last move

Color key:
  2  (#0074D9) — passable floor
  0  (#FFFFFF) — wall
  14 (#4FCC30) — exit position
  5  (#000000) — letterbox border
  Player/projectile color is randomized each level.

Win condition : All player pieces are on their exit positions simultaneously.
Lose condition: A projectile hits a player piece, OR step budget exhausted.
Levels        : 7 levels with increasing grid size and projectile count.
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
C_FLOOR      = 2   # blue passable
C_WALL       = 0   # white wall
C_PLAYER     = 4   # dark player
C_EXIT       = 14  # green exit
C_PROJECTILE = 8   # red projectile
C_BORDER     = 5   # black

CELL = 3

_PLAYER_PIX = [
    [-1, C_PLAYER, -1],
    [C_PLAYER, C_PLAYER, C_PLAYER],
    [-1, C_PLAYER, -1],
]

_PROJ_PIX = [
    [-1, C_PROJECTILE, -1],
    [C_PROJECTILE, C_PROJECTILE, C_PROJECTILE],
    [C_PROJECTILE, -1, -1],
]

_PLAYER_SHAPES = [
    [
        [-1, 1, -1],
        [1, 1, 1],
        [-1, 1, -1],
    ],
    [
        [1, 1, 1],
        [1, -1, 1],
        [1, 1, 1],
    ],
    [
        [1, -1, 1],
        [-1, 1, -1],
        [1, -1, 1],
    ],
]

_PROJ_SHAPES = [
    [
        [-1, 1, -1],
        [1, 1, 1],
        [1, -1, -1],
    ],
    [
        [-1, 1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
    ],
    [
        [-1, -1, 1],
        [1, 1, 1],
        [-1, -1, 1],
    ],
]

_PLAYER_COLORS = [1, 3, 4, 6, 7, 9, 10, 11, 12, 13, 15]
_PROJECTILE_COLORS = [1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15]

_EXIT_PIX = [
    [C_EXIT, -1, C_EXIT],
    [-1, C_EXIT, -1],
    [C_EXIT, -1, C_EXIT],
]

_FLOOR_PIX = [[C_FLOOR] * CELL] * CELL
_WALL_PIX  = [[C_WALL]  * CELL] * CELL

# (grid_w, grid_h, n_players, n_projectiles, n_interior_walls)
_CONFIGS = [
    ( 7,  7, 1, 1,  3),
    ( 9,  9, 1, 2,  6),
    (11, 11, 2, 2,  8),
    (11, 11, 2, 3, 10),
    (13, 13, 2, 3, 12),
    (13, 13, 2, 3, 14),
    (15, 15, 2, 4, 16),
]

_STEP_BUDGETS = [50, 80, 120, 160, 220, 300, 400]

# Direction vectors
_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up down left right


def _colorize_shape(shape: list[list[int]], color: int) -> list[list[int]]:
    return [[color if cell != -1 else -1 for cell in row] for row in shape]


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

def _bfs_dist(grid: list[list[int]], start: tuple[int,int],
              w: int, h: int) -> dict[tuple[int,int], int]:
    """BFS distances from start on passable cells (0=wall in grid)."""
    dist = {start: 0}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in _DIRS:
            nx, ny = x+dx, y+dy
            if (nx, ny) not in dist and 0 <= nx < w and 0 <= ny < h:
                if grid[ny][nx] == 1:
                    dist[(nx, ny)] = dist[(x, y)] + 1
                    q.append((nx, ny))
    return dist


def _generate(w: int, h: int, n_players: int, n_proj: int,
              n_walls: int, rng: random.Random):
    """Generate a level with guaranteed solvability."""
    # Build grid: perimeter wall + random interior walls
    grid = [[0]*w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if 1 <= x < w-1 and 1 <= y < h-1:
                grid[y][x] = 1  # passable interior

    # Add interior walls
    interior = [(x, y) for y in range(2, h-2) for x in range(2, w-2)]
    rng.shuffle(interior)
    added = 0
    for x, y in interior:
        if added >= n_walls:
            break
        grid[y][x] = 0
        # Check connectivity
        floor = [(fx, fy) for fy in range(1, h-1) for fx in range(1, w-1) if grid[fy][fx]]
        if not floor:
            grid[y][x] = 1
            continue
        # BFS from first floor cell
        visited = {floor[0]}
        q = deque([floor[0]])
        while q:
            cx, cy = q.popleft()
            for ddx, ddy in _DIRS:
                nb = (cx+ddx, cy+ddy)
                if nb not in visited and 0 <= nb[0] < w and 0 <= nb[1] < h:
                    if grid[nb[1]][nb[0]]:
                        visited.add(nb)
                        q.append(nb)
        if len(visited) == len(floor):
            added += 1
        else:
            grid[y][x] = 1

    floor = [(x, y) for y in range(1, h-1) for x in range(1, w-1) if grid[y][x]]
    rng.shuffle(floor)

    # Place players and exits (guaranteed reachable from each other)
    players = []
    exits = []
    used = set()
    for i in range(n_players):
        # Pick player start
        while floor:
            pos = floor.pop()
            if pos not in used:
                players.append(pos)
                used.add(pos)
                break
        # Pick exit reachable from player
        dist = _bfs_dist(grid, players[-1], w, h)
        reachable = [p for p in floor if p in dist and p not in used and dist[p] >= 3]
        if not reachable:
            reachable = [p for p in floor if p in dist and p not in used]
        if reachable:
            # Pick a distant exit
            reachable.sort(key=lambda p: -dist[p])
            ex = reachable[0]
            exits.append(ex)
            used.add(ex)
            floor = [p for p in floor if p not in used]

    if len(players) != n_players or len(exits) != n_players:
        # Fallback: just pick any positions
        remaining = [p for p in [(x, y) for y in range(1,h-1) for x in range(1,w-1)
                                 if grid[y][x]] if p not in used]
        while len(players) < n_players and remaining:
            players.append(remaining.pop(0))
            used.add(players[-1])
        while len(exits) < n_players and remaining:
            exits.append(remaining.pop(0))
            used.add(exits[-1])

    # Place projectiles on passable cells, not occupied, with valid trajectory
    proj_list = []  # (x, y, dx, dy)
    for _ in range(n_proj):
        candidates = [(x, y) for y in range(1, h-1) for x in range(1, w-1)
                      if grid[y][x] and (x,y) not in used and (x,y) not in players and (x,y) not in exits]
        rng.shuffle(candidates)
        for px, py in candidates:
            dir_candidates = list(_DIRS)
            rng.shuffle(dir_candidates)
            for dx, dy in dir_candidates:
                # Check that projectile can move at least 2 steps
                nx, ny = px+dx, py+dy
                if 0 <= nx < w and 0 <= ny < h and grid[ny][nx]:
                    proj_list.append((px, py, dx, dy))
                    used.add((px, py))
                    break
            else:
                continue
            break

    return grid, players, exits, proj_list


def _build_level(grid: list[list[int]], exits: list[tuple[int,int]],
                 w: int, h: int) -> Level:
    sprites = []
    for y in range(h):
        for x in range(w):
            if grid[y][x]:
                sprites.append(
                    Sprite(pixels=_FLOOR_PIX, name=f"floor_{x}_{y}",
                           collidable=False, layer=-1)
                    .set_position(x*CELL, y*CELL)
                )
            else:
                sprites.append(
                    Sprite(pixels=_WALL_PIX, name=f"wall_{x}_{y}",
                           collidable=True, layer=-1)
                    .set_position(x*CELL, y*CELL)
                )
    for i, (ex, ey) in enumerate(exits):
        sprites.append(
            Sprite(pixels=_EXIT_PIX, name=f"exit_{i}",
                   collidable=False, tags=["exit"], layer=0)
            .set_position(ex*CELL, ey*CELL)
        )
    return Level(sprites=sprites, grid_size=(w*CELL, h*CELL))


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class ProjectileDodge(AugmentedGame):

    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: the base rotates the board on set_level
        self._rng = random.Random(seed)
        self._step_counter = StepCounter(_STEP_BUDGETS[0])

        self._level_data = []  # (grid, players, exits, proj_list, w, h)
        levels = []

        for w, h, n_pl, n_pr, n_walls in _CONFIGS:
            grid, players, exits, proj_list = _generate(
                w, h, n_pl, n_pr, n_walls, self._rng
            )
            levels.append(_build_level(grid, exits, w, h))
            self._level_data.append((grid, players, exits, proj_list, w, h))

        self._history = []  # list of (players, proj_list) snapshots

        super().__init__(
            game_id="tu93",
            levels=levels,
            camera=Camera(background=C_FLOOR, letter_box=C_BORDER,
                          interfaces=[self._step_counter]),
            available_actions=[1, 2, 3, 4, 7],
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _wall_at(self, x: int, y: int) -> bool:
        _, _, _, _, w, h = self._level_data[self._current_level_index]
        grid = self._level_data[self._current_level_index][0]
        if not (0 <= x < w and 0 <= y < h):
            return True
        return grid[y][x] == 0

    def _players(self):
        return self.current_level.get_sprites_by_tag("player")

    def _projectiles(self):
        return self.current_level.get_sprites_by_tag("proj")

    def _snapshot(self):
        pl = [(s.x // CELL, s.y // CELL) for s in self._players()]
        proj = [(s.x // CELL, s.y // CELL,
                 int(s._tags[1].split("_")[0]),  # dx tag
                 int(s._tags[1].split("_")[1]))   # dy tag
                for s in self._projectiles()]
        return (pl, proj)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_set_level(self, level: Level) -> None:
        self._history.clear()
        idx = self._current_level_index
        self._step_counter.reset(_STEP_BUDGETS[idx])

        for s in level.get_sprites():
            if "player" in s.tags or "proj" in s.tags:
                level.remove_sprite(s)

        _, players, exits, proj_list, w, h = self._level_data[idx]

        for i, (px, py) in enumerate(players):
            player_color = self._rng.choice(_PLAYER_COLORS)
            player_shape = self._rng.choice(_PLAYER_SHAPES)
            level.add_sprite(
                Sprite(pixels=_colorize_shape(player_shape, player_color), name=f"player_{i}",
                       collidable=False, tags=["player"], layer=2)
                .set_position(px*CELL, py*CELL)
            )

        for i, (px, py, dx, dy) in enumerate(proj_list):
            projectile_color = self._rng.choice(_PROJECTILE_COLORS)
            projectile_shape = self._rng.choice(_PROJ_SHAPES)
            level.add_sprite(
                Sprite(pixels=_colorize_shape(projectile_shape, projectile_color), name=f"proj_{i}",
                       collidable=False,
                       tags=["proj", f"{dx}_{dy}"],
                       layer=2)
                .set_position(px*CELL, py*CELL)
            )

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

        players = self._players()
        projectiles = self._projectiles()

        if action == GameAction.ACTION7:
            if self._history:
                pl_snap, proj_snap = self._history.pop()
                for s, (x, y) in zip(players, pl_snap):
                    s.set_position(x*CELL, y*CELL)
                for s, (x, y, dx, dy) in zip(projectiles, proj_snap):
                    s.set_position(x*CELL, y*CELL)
                    s._tags[1] = f"{dx}_{dy}"
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        # Snapshot before action
        snap = (
            [(s.x//CELL, s.y//CELL) for s in players],
            [(s.x//CELL, s.y//CELL,
              int(s._tags[1].split("_")[0]),
              int(s._tags[1].split("_")[1])) for s in projectiles],
        )
        self._history.append(snap)

        # Move players
        dx, dy = self._DELTAS.get(action, (0, 0))
        if dx or dy:
            for s in players:
                nx = s.x//CELL + dx
                ny = s.y//CELL + dy
                if not self._wall_at(nx, ny):
                    s.set_position(nx*CELL, ny*CELL)

        # Advance projectiles (one step after player moves)
        _, _, _, _, w, h = self._level_data[self._current_level_index]
        for s in projectiles:
            px_g = s.x // CELL
            py_g = s.y // CELL
            pdx = int(s._tags[1].split("_")[0])
            pdy = int(s._tags[1].split("_")[1])
            npx, npy = px_g + pdx, py_g + pdy
            if not self._wall_at(npx, npy):
                s.set_position(npx*CELL, npy*CELL)
            else:
                # Bounce: reverse direction
                pdx, pdy = -pdx, -pdy
                s._tags[1] = f"{pdx}_{pdy}"
                bnpx, bnpy = px_g + pdx, py_g + pdy
                if not self._wall_at(bnpx, bnpy):
                    s.set_position(bnpx*CELL, bnpy*CELL)

        # Check collision: projectile on player
        player_positions = {(s.x, s.y) for s in players}
        for p in projectiles:
            if (p.x, p.y) in player_positions:
                self.lose()
                self.complete_action()
                return

        # Check win: all players on exits
        exit_sprites = self.current_level.get_sprites_by_tag("exit")
        exit_positions = {(s.x, s.y) for s in exit_sprites}
        players_now = self._players()
        if all((s.x, s.y) in exit_positions for s in players_now):
            self.next_level()
            self.complete_action()
            return

        if not self._step_counter.steps_remaining:
            self.lose()

        self.complete_action()
