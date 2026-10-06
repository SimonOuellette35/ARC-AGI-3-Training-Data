"""
Badge Placement
=======================
Colored badge sprites are scattered on a grid. Target slots (outline frames)
mark where each badge should go. Navigate a cursor, pick up a badge, and
place it on its matching target slot.

A badge and target match when they share the same color.
Carry only one badge at a time. Placing on a wrong target is rejected.

Actions:
  ACTION1 (↑) — move cursor up
  ACTION2 (↓) — move cursor down
  ACTION3 (←) — move cursor left
  ACTION4 (→) — move cursor right
  ACTION5     — pick up badge at cursor / place held badge
  ACTION7     — undo last action

Color key:
  0  (#FFFFFF) — floor
  4  (#333333) — wall
  9  (#1E93FF) — cursor (empty)
  5  (#000000) — letterbox

Win condition : All badges in their matching target slots.
Lose condition: Step budget exhausted.
Levels        : 7 levels with more badges.
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

C_FLOOR   = 0
C_WALL    = 4
C_CURSOR  = 9
C_BORDER  = 5

BADGE_COLORS = [8, 11, 12, 6, 3, 2, 10, 1]

CELL = 3

_CURSOR_PIX = [
    [C_CURSOR, -1, C_CURSOR],
    [-1, C_CURSOR, -1],
    [C_CURSOR, -1, C_CURSOR],
]

_WALL_PIX = [[C_WALL]*CELL]*CELL

def _badge_pix(color: int) -> list[list[int]]:
    return [
        [color, color, color],
        [color, C_FLOOR, color],
        [color, color, color],
    ]

def _target_pix(color: int) -> list[list[int]]:
    return [
        [color, -1, color],
        [-1, C_FLOOR, -1],
        [color, -1, color],
    ]

def _cursor_holding_pix(color: int) -> list[list[int]]:
    # Inverse of the empty-cursor-over-badge render (9-corners/colour-edges)
    # and fully opaque: "holding" must be readable from the frame alone,
    # otherwise carrying the last badge onto its slot is pixel-identical to
    # the won board (badge placed + empty cursor on top).
    return [
        [color, C_CURSOR, color],
        [C_CURSOR, color, C_CURSOR],
        [color, C_CURSOR, color],
    ]

# (grid_w, grid_h, n_badges, n_walls)
_CONFIGS = [
    ( 7,  7, 2,  3),
    ( 9,  9, 3,  6),
    ( 9,  9, 4,  7),
    (11, 11, 4,  9),
    (11, 11, 5, 10),
    (13, 13, 6, 12),
    (13, 13, 7, 14),
]

_STEP_BUDGETS = [50, 80, 110, 150, 200, 260, 340]

_DIRS = [(0,-1),(0,1),(-1,0),(1,0)]


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


def _generate(w, h, n_badges, n_walls, rng):
    grid = [[0]*w for _ in range(h)]
    for y in range(1, h-1):
        for x in range(1, w-1):
            grid[y][x] = 1

    candidates = [(x, y) for y in range(2, h-2) for x in range(2, w-2)]
    rng.shuffle(candidates)
    added = 0
    for x, y in candidates:
        if added >= n_walls:
            break
        grid[y][x] = 0
        floor = [(fx,fy) for fy in range(1,h-1) for fx in range(1,w-1) if grid[fy][fx]]
        if floor:
            vis = {floor[0]}
            q = deque([floor[0]])
            while q:
                cx,cy = q.popleft()
                for dx,dy in _DIRS:
                    nb = (cx+dx,cy+dy)
                    if nb not in vis and 0<=nb[0]<w and 0<=nb[1]<h and grid[nb[1]][nb[0]]:
                        vis.add(nb); q.append(nb)
            if len(vis) == len(floor):
                added += 1; continue
        grid[y][x] = 1

    floor = [(x,y) for y in range(1,h-1) for x in range(1,w-1) if grid[y][x]]
    rng.shuffle(floor)

    cursor_pos = floor.pop()
    badge_positions = []
    target_positions = []
    used = {cursor_pos}

    for k in range(n_badges):
        while floor:
            p = floor.pop()
            if p not in used:
                badge_positions.append(p)
                used.add(p)
                break
    for k in range(n_badges):
        while floor:
            p = floor.pop()
            if p not in used:
                target_positions.append(p)
                used.add(p)
                break

    return grid, cursor_pos, badge_positions, target_positions


def _build_level(grid, target_positions, w, h) -> Level:
    sprites = []
    for y in range(h):
        for x in range(w):
            if not grid[y][x]:
                sprites.append(
                    Sprite(pixels=_WALL_PIX, name=f"wall_{x}_{y}",
                           collidable=True, layer=-1)
                    .set_position(x*CELL, y*CELL)
                )
    for k, (tx, ty) in enumerate(target_positions):
        color = BADGE_COLORS[k % len(BADGE_COLORS)]
        sprites.append(
            Sprite(pixels=_target_pix(color), name=f"target_{k}",
                   collidable=False, tags=["target", f"color_{k}"], layer=0)
            .set_position(tx*CELL, ty*CELL)
        )
    return Level(sprites=sprites, grid_size=(w*CELL, h*CELL))


class BadgePlacement(AugmentedGame):

    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: the base draws this level's rotation
        self._rng = random.Random(seed)
        self._step_counter = StepCounter(_STEP_BUDGETS[0])

        self._level_data = []
        levels = []

        for w, h, n_badges, n_walls in _CONFIGS:
            grid, cursor_pos, badge_positions, target_positions = _generate(
                w, h, n_badges, n_walls, self._rng
            )
            levels.append(_build_level(grid, target_positions, w, h))
            self._level_data.append((grid, cursor_pos, badge_positions, target_positions, w, h))

        self._history = []
        self._holding = None  # color index of held badge

        super().__init__(
            game_id="r11l",
            levels=levels,
            camera=Camera(background=C_FLOOR, letter_box=C_BORDER,
                          interfaces=[self._step_counter]),
            available_actions=[1, 2, 3, 4, 5, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self._history.clear()
        self._holding = None
        idx = self._current_level_index
        self._step_counter.reset(_STEP_BUDGETS[idx])

        for s in level.get_sprites():
            if "badge" in s.tags or "cursor" in s.tags:
                level.remove_sprite(s)

        grid, cursor_pos, badge_positions, target_positions, w, h = self._level_data[idx]

        for k, (bx, by) in enumerate(badge_positions):
            color = BADGE_COLORS[k % len(BADGE_COLORS)]
            level.add_sprite(
                Sprite(pixels=_badge_pix(color), name=f"badge_{k}",
                       collidable=False, tags=["badge", f"color_{k}"], layer=1)
                .set_position(bx*CELL, by*CELL)
            )

        level.add_sprite(
            Sprite(pixels=_CURSOR_PIX, name="cursor",
                   collidable=False, tags=["cursor"], layer=3)
            .set_position(cursor_pos[0]*CELL, cursor_pos[1]*CELL)
        )

    def _cursor(self):
        return self.current_level.get_sprites_by_tag("cursor")[0]

    def _wall_at(self, x, y):
        grid, _, _, _, w, h = self._level_data[self._current_level_index]
        if not (0 <= x < w and 0 <= y < h):
            return True
        return grid[y][x] == 0

    def _badge_at(self, x, y):
        for s in self.current_level.get_sprites_by_tag("badge"):
            if s.x//CELL == x and s.y//CELL == y and s.is_visible:
                return s
        return None

    def _target_at(self, x, y):
        for s in self.current_level.get_sprites_by_tag("target"):
            if s.x//CELL == x and s.y//CELL == y and s.is_visible:
                return s
        return None

    def _snapshot(self):
        cur = self._cursor()
        return (
            cur.x//CELL, cur.y//CELL,
            self._holding,
            [(s.x//CELL, s.y//CELL, s.is_visible)
             for s in self.current_level.get_sprites_by_tag("badge")]
        )

    def _restore(self, snap):
        cx, cy, held, badge_data = snap
        self._cursor().set_position(cx*CELL, cy*CELL)
        self._holding = held
        badges = self.current_level.get_sprites_by_tag("badge")
        for s, (bx, by, vis) in zip(badges, badge_data):
            s.set_position(bx*CELL, by*CELL)
            s.set_visible(vis)
        # Update cursor visuals
        if held is not None:
            color = BADGE_COLORS[held % len(BADGE_COLORS)]
            self._cursor().pixels = np.array(_cursor_holding_pix(color), dtype=np.int8)
        else:
            self._cursor().pixels = np.array(_CURSOR_PIX, dtype=np.int8)

    _DELTAS = {
        GameAction.ACTION1: (0, -1),
        GameAction.ACTION2: (0, 1),
        GameAction.ACTION3: (-1, 0),
        GameAction.ACTION4: (1, 0),
    }

    def step(self) -> None:
        self._step_counter.decrement()
        action = self.action.id

        if action == GameAction.ACTION7:
            if self._history:
                self._restore(self._history.pop())
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        cur = self._cursor()
        cx, cy = cur.x//CELL, cur.y//CELL

        if action == GameAction.ACTION5:
            snap = self._snapshot()
            if self._holding is None:
                # Try to pick up badge at current position
                badge = self._badge_at(cx, cy)
                if badge:
                    # Get color index from tag
                    ci = next(int(t[6:]) for t in badge.tags if t.startswith("color_"))
                    self._holding = ci
                    badge.set_visible(False)
                    color = BADGE_COLORS[ci % len(BADGE_COLORS)]
                    cur.pixels = np.array(_cursor_holding_pix(color), dtype=np.int8)
                    self._history.append(snap)
            else:
                # Try to place on target
                target = self._target_at(cx, cy)
                if target:
                    ti = next(int(t[6:]) for t in target.tags if t.startswith("color_"))
                    if ti == self._holding:
                        # Place badge here
                        badges = self.current_level.get_sprites_by_tag("badge")
                        badge = next((b for b in badges
                                      if any(t.startswith(f"color_{self._holding}")
                                             for t in b.tags) and not b.is_visible), None)
                        if badge:
                            badge.set_position(cx*CELL, cy*CELL)
                            badge.set_visible(True)
                            target.set_visible(False)
                        self._holding = None
                        cur.pixels = np.array(_CURSOR_PIX, dtype=np.int8)
                        self._history.append(snap)

            # Win check
            remaining_targets = [s for s in self.current_level.get_sprites_by_tag("target")
                                  if s.is_visible]
            if not remaining_targets and self._holding is None:
                self.next_level()
                self.complete_action()
                return

            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        dx, dy = self._DELTAS.get(action, (0, 0))
        if not (dx or dy):
            self.complete_action()
            return

        nx, ny = cx+dx, cy+dy
        if not self._wall_at(nx, ny):
            cur.set_position(nx*CELL, ny*CELL)

        if not self._step_counter.steps_remaining:
            self.lose()

        self.complete_action()

