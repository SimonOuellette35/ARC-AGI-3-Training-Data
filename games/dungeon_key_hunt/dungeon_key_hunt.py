"""
Dungeon Key Hunt
================
Navigate a dungeon, collect colored keys and use them to unlock matching
doors. Reach the treasure chest to win. The hero can carry up to 3 keys.
ACTION5 picks up a key on or adjacent to the hero, or unlocks an adjacent
door (using the matching key). ACTION7 undoes the last move or last key
pickup (but not door unlocks).

Color key:
  4  (#333333) — dungeon floor (dark stone)
  0  (#FFFFFF) — wall (impassable)
  9  (#1E93FF) — hero (blue)
  8  (#F93C31) — red key / red door (locked)
  11 (#FFDC00) — yellow key / yellow door
  12 (#FF851B) — orange key / orange door
  6  (#E36EF6) — pink key / pink door
  14 (#4FCC30) — unlocked door (open archway, passable)
  3  (#4FCC30) — treasure chest (goal)
  5  (#000000) — letterbox border

Actions:
  ACTION1 (↑) — move hero up
  ACTION2 (↓) — move hero down
  ACTION3 (←) — move hero left
  ACTION4 (→) — move hero right
  ACTION5     — pick up key (on hero or adjacent) / unlock adjacent door
  ACTION7     — undo last move or last key pickup

Win condition : Hero reaches the treasure chest.
Lose condition: Step budget exhausted.
Levels        : 4 dungeons — 2 keys/doors (easy) → 3 → 4 → 5 (expert).
"""

from __future__ import annotations

import random
from collections import deque
from typing import Set, Tuple

import numpy as np
from utils.arc_game import AugmentedGame
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
C_FLOOR   = 4   # dark stone floor
C_WALL    = 0   # white wall
C_HERO    = 9   # blue hero
C_CHEST   = 3   # green treasure chest
C_OPEN_DOOR = 14  # green open archway
C_BORDER  = 5   # black

_KEY_COLORS  = [8, 11, 12, 6, 1, 2, 7, 10]   # red, yellow, orange, pink, blue, magenta, grey, teal
_DOOR_COLORS = [8, 11, 12, 6, 1]   # same colors for doors

_STEP_BUDGETS = [100, 160, 250, 360, 500, 680, 900]
_CARRY_LIMIT = 3
# Minimum doors that must lie on the direct hero→chest path per level.
# Higher level = more guaranteed blocking doors the player must unlock.
_MIN_PATH_DOORS = [1, 2, 3, 4, 4, 4, 5]

# ---------------------------------------------------------------------------
# Cell size: CELL=3 gives max viewport 51×45 for 17×15 grid.
# ---------------------------------------------------------------------------
CELL = 3

# ---------------------------------------------------------------------------
# Sprite pixel art  (all CELL×CELL = 3×3 pixels)
# ---------------------------------------------------------------------------

# Hero: 3×3 armored figure
_HERO_PIXELS = [
    [ 7,      C_HERO,  7     ],  # helmet + head
    [C_HERO, C_HERO,  C_HERO],  # armor body
    [C_HERO, -1,      C_HERO],  # legs
]

def _key_pixels(c: int) -> list[list[int]]:
    """3×3 key: circular bow and shaft."""
    return [
        [-1, c, c],   # bow (round part)
        [ c, c, -1],  # bow + shaft
        [ c, c,  c],  # bit/notch
    ]

def _door_pixels(c: int) -> list[list[int]]:
    """3×3 locked door: colored with knob."""
    return [
        [c, c, c],   # top of door
        [c, 0, c],   # door + handle (white knob)
        [c, c, c],   # bottom of door
    ]

# Open door: 3×3 passable archway
_OPEN_DOOR_PIXELS = [
    [C_OPEN_DOOR, C_OPEN_DOOR, C_OPEN_DOOR],  # arch top
    [-1,          -1,           -1          ],  # open passage
    [C_FLOOR,     C_FLOOR,      C_FLOOR     ],  # floor
]

# Treasure chest: 3×3
_CHEST_PIXELS = [
    [C_CHEST, 11,      C_CHEST],  # lid + gold latch (yellow)
    [C_CHEST, C_CHEST, C_CHEST],  # chest body
    [4,       4,       4      ],  # dark base/shadow
]

# Level configs: (width, height, num_keys)
_CONFIGS = [
    (11, 9,  2),
    (13, 11, 3),
    (15, 13, 4),
    (17, 15, 5),
    (19, 17, 5),
    (21, 19, 5),
    (21, 19, 5),
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
# BFS helper
# ---------------------------------------------------------------------------

def _reachable(grid: list, width: int, height: int,
               start: Tuple[int, int],
               blocked: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """Return set of (x,y) floor cells reachable from start via BFS,
    treating cells in *blocked* as impassable walls."""
    visited: Set[Tuple[int, int]] = set()
    sx, sy = start
    if (sx, sy) in blocked or grid[sy][sx] != 1:
        return visited
    queue: deque = deque([start])
    visited.add(start)
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if (0 <= nx < width and 0 <= ny < height
                    and grid[ny][nx] == 1
                    and (nx, ny) not in visited
                    and (nx, ny) not in blocked):
                visited.add((nx, ny))
                queue.append((nx, ny))
    return visited


# ---------------------------------------------------------------------------
# Dungeon generator
# ---------------------------------------------------------------------------

def _find_path_cells(grid: list, width: int, height: int,
                     start: Tuple[int, int],
                     goal: Tuple[int, int]) -> list[Tuple[int, int]]:
    """BFS – return the cells (inclusive) on the unique path from start to goal."""
    visited: dict[Tuple[int, int], Tuple[int, int] | None] = {start: None}
    queue: deque = deque([start])
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            path: list[Tuple[int, int]] = []
            cur: Tuple[int, int] | None = goal
            while cur is not None:
                path.append(cur)
                cur = visited[cur]
            path.reverse()
            return path
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if (0 <= nx < width and 0 <= ny < height
                    and grid[ny][nx] == 1 and (nx, ny) not in visited):
                visited[(nx, ny)] = (x, y)
                queue.append((nx, ny))
    return []


def _generate_dungeon(width: int, height: int, num_keys: int,
                       min_path_doors: int,
                       rng: random.Random) -> dict:
    """Generate a dungeon level with rooms connected by corridors."""
    assert width % 2 == 1 and height % 2 == 1

    # Retry maze generation if door placement fails (can happen when hero starts
    # in a small pocket that doesn't leave room for key placement).
    for _maze_attempt in range(50):
        # Use DFS maze as base
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

        floor_cells = [(x, y) for y in range(height) for x in range(width) if grid[y][x] == 1]
        rng.shuffle(floor_cells)

        # Hero starts at first floor cell
        hero_start = floor_cells[0]
        # Chest at last floor cell (far from start)
        chest_pos = floor_cells[-1]

        def passable_neighbor_count(x: int, y: int) -> int:
            count = 0
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == 1:
                    count += 1
            return count

        # All corridor cells that could hold a door
        all_corridors = [
            p for p in floor_cells
            if p != hero_start and p != chest_pos
            and passable_neighbor_count(*p) == 2
        ]

        colors = rng.sample(_KEY_COLORS, num_keys)

        # --- Classify corridors: on the hero→chest path vs off-path branches ---
        hero_chest_path = _find_path_cells(grid, width, height, hero_start, chest_pos)
        path_set = set(hero_chest_path)
        # Path corridors (cells that block direct hero→chest passage when used as doors)
        path_corridors = [
            p for p in hero_chest_path
            if p != hero_start and p != chest_pos
            and passable_neighbor_count(*p) == 2
        ]
        non_path_corridors = [c for c in all_corridors if c not in path_set]

        # --- Guarantee solvability with enforced path difficulty ---
        # Place at least min_path_doors doors on the hero→chest path so the player
        # MUST unlock them.  Fill remaining door slots from off-path branches.
        # All keys must be accessible from hero_start with every door closed.
        used_base = {hero_start, chest_pos}
        door_positions: list[tuple[int, int]] = []
        key_positions: list[tuple[int, int]] = []

        n_on_path = min(min_path_doors, len(path_corridors), num_keys)

        for _attempt in range(400):
            rng.shuffle(path_corridors)
            rng.shuffle(non_path_corridors)

            chosen_path = path_corridors[:n_on_path]
            remaining = num_keys - len(chosen_path)
            chosen_off = non_path_corridors[:remaining]
            trial_doors = chosen_path + chosen_off

            if len(trial_doors) < num_keys:
                # Not enough corridors at all; pad from whichever pool has extras
                extra = [c for c in (path_corridors + non_path_corridors)
                         if c not in trial_doors]
                trial_doors = trial_doors + extra[:num_keys - len(trial_doors)]

            hero_region = _reachable(grid, width, height, hero_start, set(trial_doors))
            free = [c for c in hero_region if c not in used_base | set(trial_doors)]
            if len(free) >= num_keys:
                door_positions = trial_doors
                rng.shuffle(free)
                key_positions = free[:num_keys]
                break

        if door_positions:
            break  # Successfully placed doors and keys; exit maze retry loop

    if not door_positions:
        # Ultimate fallback: use last generated maze, best-effort placement
        door_positions = (path_corridors + non_path_corridors)[:num_keys]
        leftover = [c for c in floor_cells if c not in used_base | set(door_positions)]
        rng.shuffle(leftover)
        key_positions = leftover[:num_keys]

    return {
        "grid": grid,
        "width": width,
        "height": height,
        "hero_start": hero_start,
        "chest_pos": chest_pos,
        "door_positions": door_positions,
        "key_positions": key_positions,
        "colors": colors,
    }


def _build_level(data: dict) -> Level:
    grid = data["grid"]
    width, height = data["width"], data["height"]
    sprites: list[Sprite] = []

    for y in range(height):
        for x in range(width):
            if grid[y][x] == 1:
                sprites.append(
                    Sprite(pixels=[[C_FLOOR]*CELL]*CELL, name=f"floor_{x}_{y}",
                           collidable=False, tags=["floor"], layer=0)
                    .set_position(x * CELL, y * CELL)
                )
            else:
                sprites.append(
                    Sprite(pixels=[[C_WALL]*CELL]*CELL, name=f"wall_{x}_{y}",
                           collidable=True, tags=["wall"], layer=-1)
                    .set_position(x * CELL, y * CELL)
                )

    # Doors (initially closed/blocking)
    colors = data["colors"]
    for i, (dx, dy) in enumerate(data["door_positions"]):
        sprites.append(
            Sprite(pixels=_door_pixels(colors[i]), name=f"door_{i}",
                   collidable=True,
                   tags=["door", f"door_{i}", "door_closed", f"door_color_{colors[i]}"],
                   layer=2)
            .set_position(dx * CELL, dy * CELL)
        )

    # Chest
    cx, cy = data["chest_pos"]
    sprites.append(
        Sprite(pixels=_CHEST_PIXELS, name="chest",
               collidable=False, tags=["chest"], layer=1)
        .set_position(cx * CELL, cy * CELL)
    )

    return Level(sprites=sprites, grid_size=(width * CELL, height * CELL))


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class DungeonKeyHunt(AugmentedGame):

    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: the base draws this level's rotation
        rng = random.Random(seed)
        self._step_counter = StepCounter(_STEP_BUDGETS[0])
        self._level_data: list[dict] = []
        levels: list[Level] = []

        for level_idx, (width, height, num_keys) in enumerate(_CONFIGS):
            data = _generate_dungeon(width, height, num_keys,
                                     _MIN_PATH_DOORS[level_idx], rng)
            self._level_data.append(data)
            levels.append(_build_level(data))

        self._inventory: list[int] = []  # list of color indices held
        self._history: list[tuple[
            tuple[int, int],
            list[int],  # inventory
            list[bool],  # key visibility (True=on floor)
            list[bool],  # door state (True=locked)
        ]] = []

        super().__init__(
            game_id="dungeon_key_hunt",
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

    def _hero(self) -> Sprite:
        return self.current_level.get_sprites_by_tag("hero")[0]

    def _keys(self) -> list[Sprite]:
        return self.current_level.get_sprites_by_tag("key")

    def _doors(self) -> list[Sprite]:
        return self.current_level.get_sprites_by_tag("door")

    def _key_at(self, x: int, y: int) -> Sprite | None:
        return next((s for s in self._keys() if s.x == x and s.y == y and s.is_visible), None)

    def _door_at(self, x: int, y: int) -> Sprite | None:
        return next((s for s in self._doors()
                     if s.x == x and s.y == y and "door_closed" in s.tags), None)

    def _is_blocked(self, x: int, y: int) -> bool:
        return any(s.x == x and s.y == y and s.is_collidable
                   for s in self.current_level.get_sprites())

    def _get_adjacent_key(self) -> Sprite | None:
        hero = self._hero()
        for dx, dy in ((0, 0), (0, -CELL), (0, CELL), (-CELL, 0), (CELL, 0)):
            k = self._key_at(hero.x + dx, hero.y + dy)
            if k is not None:
                return k
        return None

    def _get_adjacent_door(self) -> Sprite | None:
        hero = self._hero()
        for dx, dy in ((0, 0), (0, -CELL), (0, CELL), (-CELL, 0), (CELL, 0)):
            d = self._door_at(hero.x + dx, hero.y + dy)
            if d is not None:
                return d
        return None

    def _door_color(self, door: Sprite) -> int:
        for tag in door.tags:
            if tag.startswith("door_color_"):
                return int(tag.split("_")[2])
        return -1

    def _key_color(self, key: Sprite) -> int:
        for tag in key.tags:
            if tag.startswith("key_color_"):
                return int(tag.split("_")[2])
        return -1

    def _snapshot(self):
        hero = self._hero()
        key_vis = [k.is_visible for k in self._keys()]
        door_locked = ["door_closed" in d.tags for d in self._doors()]
        return (hero.x, hero.y), list(self._inventory), key_vis, door_locked

    def on_set_level(self, level: Level) -> None:
        # Remove any hero/key sprites from a previous call (idempotency)
        for sprite in list(level.get_sprites()):
            if any(t in sprite.tags for t in ("hero", "key")):
                level.remove_sprite(sprite)

        self._history.clear()
        self._inventory.clear()
        self._step_counter.reset(_STEP_BUDGETS[self._current_level_index])
        data = self._level_data[self._current_level_index]

        # Add hero
        hx, hy = data["hero_start"]
        level.add_sprite(
            Sprite(pixels=_HERO_PIXELS, name="hero",
                   collidable=False, tags=["hero"], layer=3)
            .set_position(hx * CELL, hy * CELL)
        )

        # Add keys
        colors = data["colors"]
        for i, (kx, ky) in enumerate(data["key_positions"]):
            level.add_sprite(
                Sprite(pixels=_key_pixels(colors[i]), name=f"key_{i}",
                       collidable=False,
                       tags=["key", f"key_{i}", f"key_color_{colors[i]}"],
                       layer=2)
                .set_position(kx * CELL, ky * CELL)
            )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    _DELTAS: dict[int, tuple[int, int]] = {
        GameAction.ACTION1: (0, -CELL),
        GameAction.ACTION2: (0,  CELL),
        GameAction.ACTION3: (-CELL, 0),
        GameAction.ACTION4: (CELL,  0),
    }

    def step(self) -> None:
        self._step_counter.decrement()

        # --- Undo -------------------------------------------------------
        if self.action.id == GameAction.ACTION7:
            if self._history:
                (hx, hy), inv, key_vis, door_locked = self._history.pop()
                self._hero().set_position(hx, hy)
                self._inventory = list(inv)
                for key, vis in zip(self._keys(), key_vis):
                    key.set_visible(vis)
                for door, locked in zip(self._doors(), door_locked):
                    if locked and "door_closed" not in door.tags:
                        door.tags.append("door_closed")
                        door.set_collidable(True)
                        door.pixels = np.array(
                            _door_pixels(self._door_color(door)), dtype=np.int8
                        )
                    elif not locked and "door_closed" in door.tags:
                        door._tags[:] = [t for t in door.tags if t != "door_closed"]
                        door.set_collidable(False)
                        door.pixels = np.array(_OPEN_DOOR_PIXELS, dtype=np.int8)
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        # --- Interact (ACTION5) ----------------------------------------
        if self.action.id == GameAction.ACTION5:
            # Try pick up key
            key = self._get_adjacent_key()
            if key is not None and len(self._inventory) < _CARRY_LIMIT:
                self._history.append(self._snapshot())
                color = self._key_color(key)
                self._inventory.append(color)
                key.set_visible(False)
                self.complete_action()
                if not self._step_counter.steps_remaining:
                    self.lose()
                return

            # Try unlock door
            door = self._get_adjacent_door()
            if door is not None:
                door_c = self._door_color(door)
                if door_c in self._inventory:
                    self._history.append(self._snapshot())
                    self._inventory.remove(door_c)
                    # Open door
                    door._tags[:] = [t for t in door.tags if t != "door_closed"]
                    door.set_collidable(False)
                    door.pixels = np.array(_OPEN_DOOR_PIXELS, dtype=np.int8)

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
        hero = self._hero()
        nx, ny = hero.x + dx, hero.y + dy

        if self._is_blocked(nx, ny):
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        self._history.append(self._snapshot())
        hero.set_position(nx, ny)

        # Check win: hero on chest
        chest = self.current_level.get_sprites_by_tag("chest")
        if chest and hero.x == chest[0].x and hero.y == chest[0].y:
            self.next_level()
            self.complete_action()
            return

        if not self._step_counter.steps_remaining:
            self.lose()
        self.complete_action()
