"""
Rail Selector
=============
Colored tiles sit on horizontal rails. Use the selector to choose which
rail is active, then slide the tile along its rail to reach the target position.

Each rail has exactly one tile. The target position for each rail is shown
with a matching-color marker on the far end of the rail.

Actions:
  ACTION1 (↑) — select previous rail
  ACTION2 (↓) — select next rail
  ACTION3 (←) — slide active tile left
  ACTION4 (→) — slide active tile right
  ACTION7     — undo last slide

Color key:
  0  (#000000) — background
  5  (#888888) — rail track / letterbox
  7  (#E05010) — selector indicator

Win condition : All tiles at their target positions.
Lose condition: Step budget exhausted.
Levels        : 7 levels with more rails.
"""

from __future__ import annotations

import random

import numpy as np
from arcengine import (
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)
from utils.arc_game import AugmentedGame

C_BG     = 0
C_RAIL   = 5
C_SEL    = 7
C_BORDER = 5

CELL = 4

RAIL_COLORS = [8, 2, 3, 12, 6, 9, 10, 11, 14, 1]  # 10 distinct rail colors

# (n_rails, rail_length)
_CONFIGS = [
    (3, 6),
    (4, 7),
    (5, 7),
    (5, 8),
    (6, 8),
    (7, 9),
    (8, 9),
]

_STEP_BUDGETS = [40, 60, 80, 100, 130, 170, 210]

RAIL_SPACING = CELL + 2   # vertical pixels between rails

# Left-hand gutter reserved for the selector indicator. The canvas is only
# ``rail_length * CELL + 8`` wide, so a selector drawn at x = -CELL (as it was)
# falls outside the grid and the camera clips it away entirely -- the indicator
# was never visible. Shift the playfield one cell right instead, which leaves the
# gutter on-canvas and still keeps 4px of slack on the right.
PLAY_X0 = CELL

# Colours the rail TRACK may not take. C_BG would make the rails invisible,
# C_RAIL is also the letterbox/border grey, and C_SEL is the highlight colour of
# the selected tile's border -- an orange track makes the selection unreadable.
# Unrestricted, 27.6% of (seed, level) boards drew one of these.
_TRACK_RESERVED = frozenset({C_BG, C_RAIL, C_SEL})


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


def _tile_pix(color, selected=False):
    border = C_SEL if selected else color
    return [
        [border, border, border, border],
        [border, color,  color,  border],
        [border, color,  color,  border],
        [border, border, border, border],
    ]

def _target_pix(color):
    return [
        [-1,    color, color, -1   ],
        [color, -1,    -1,    color],
        [color, -1,    -1,    color],
        [-1,    color, color, -1   ],
    ]

def _rail_pix(rail_color):
    return [
        [rail_color, rail_color, rail_color, rail_color],
        [C_BG,   C_BG,   C_BG,   C_BG  ],
        [C_BG,   C_BG,   C_BG,   C_BG  ],
        [rail_color, rail_color, rail_color, rail_color],
    ]

def _sel_pix():
    return [
        [-1,    C_SEL, C_SEL, -1   ],
        [C_SEL, C_SEL, C_SEL, C_SEL],
        [C_SEL, C_SEL, C_SEL, C_SEL],
        [-1,    C_SEL, C_SEL, -1   ],
    ]


def _generate(n_rails, rail_length, rng):
    """Generate rail positions and targets."""
    tile_positions = [rng.randint(0, rail_length-1) for _ in range(n_rails)]
    target_positions = [rng.randint(0, rail_length-1) for _ in range(n_rails)]
    return tile_positions, target_positions


class RailSelector(AugmentedGame):

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._step_counter = StepCounter(_STEP_BUDGETS[0])
        self._level_data = []
        levels = []

        for n_rails, rail_length in _CONFIGS:
            tile_pos, target_pos = _generate(n_rails, rail_length, self._rng)
            canvas_w = rail_length * CELL + 8
            canvas_h = n_rails * RAIL_SPACING + 4
            rail_colors = self._rng.sample(RAIL_COLORS, n_rails)
            rail_color_choices = [c for c in range(16)
                                 if c not in rail_colors and c not in _TRACK_RESERVED]
            rail_color = self._rng.choice(rail_color_choices)
            self._level_data.append(
                (
                    tile_pos,
                    target_pos,
                    rail_colors,
                    rail_color,
                    n_rails,
                    rail_length,
                    canvas_w,
                    canvas_h,
                )
            )
            levels.append(Level(sprites=[], grid_size=(canvas_w, canvas_h)))

        self._selector = 0
        self._tile_positions = []
        self._history = []
        self._init_augmentation(seed)

        super().__init__(
            game_id="rail_selector",
            levels=levels,
            camera=Camera(background=C_BG, letter_box=C_BORDER,
                          interfaces=[self._step_counter]),
            available_actions=[1, 2, 3, 4, 7],
        )

    def on_set_level(self, level: Level) -> None:
        self._history.clear()
        self._selector = 0
        idx = self._current_level_index
        self._step_counter.reset(_STEP_BUDGETS[idx])

        for s in list(level.get_sprites()):
            if any(t in s.tags for t in ['rail', 'tile', 'target', 'selector']):
                level.remove_sprite(s)

        tile_pos, target_pos, rail_colors, rail_color, n_rails, rail_length, canvas_w, canvas_h = self._level_data[idx]
        self._tile_positions = list(tile_pos)

        for r in range(n_rails):
            ry = r * RAIL_SPACING + 2
            color = rail_colors[r]

            # Draw rail background
            for col in range(rail_length):
                level.add_sprite(
                    Sprite(pixels=_rail_pix(rail_color), name=f"rail_{r}_{col}",
                           collidable=False, tags=['rail'], layer=-1)
                    .set_position(PLAY_X0 + col * CELL, ry)
                )

            # Target marker at this rail's goal position
            tp = target_pos[r]
            level.add_sprite(
                Sprite(pixels=_target_pix(color), name=f"target_{r}",
                       collidable=False, tags=['target', f'tr_{r}'], layer=0)
                .set_position(PLAY_X0 + tp * CELL, ry)
            )

            # Tile
            sel = (r == 0)
            level.add_sprite(
                Sprite(pixels=_tile_pix(color, sel), name=f"tile_{r}",
                       collidable=False, tags=['tile', f'tl_{r}'], layer=1)
                .set_position(PLAY_X0 + tile_pos[r] * CELL, ry)
            )

        # Selector indicator, in the left-hand gutter
        sel_ry = 0 * RAIL_SPACING + 2
        level.add_sprite(
            Sprite(pixels=_sel_pix(), name="selector",
                   collidable=False, tags=['selector'], layer=2)
            .set_position(0, sel_ry)
        )

    def _rail_y(self, r):
        return r * RAIL_SPACING + 2

    def _check_win(self):
        idx = self._current_level_index
        _, target_pos, _, _, n_rails, _, _, _ = self._level_data[idx]
        return all(self._tile_positions[r] == target_pos[r] for r in range(n_rails))

    def _snapshot(self):
        return (self._selector, list(self._tile_positions))

    def _restore(self, snap):
        sel, positions = snap
        idx = self._current_level_index
        _, _, rail_colors, _, n_rails, _, _, _ = self._level_data[idx]
        # Update old selector
        old_sel = self._selector
        t = self.current_level.get_sprites_by_name(f"tile_{old_sel}")
        if t:
            t[0].pixels = np.array(_tile_pix(rail_colors[old_sel], False), dtype=np.int8)
        self._selector = sel
        self._tile_positions = positions
        for r in range(n_rails):
            color = rail_colors[r]
            sel2 = (r == sel)
            t = self.current_level.get_sprites_by_name(f"tile_{r}")
            if t:
                t[0].pixels = np.array(_tile_pix(color, sel2), dtype=np.int8)
                ry = self._rail_y(r)
                t[0].set_position(PLAY_X0 + positions[r] * CELL, ry)
        # Update selector sprite position
        s = self.current_level.get_sprites_by_name("selector")
        if s:
            s[0].set_position(0, self._rail_y(sel))

    def step(self) -> None:
        _action_id = self.screen_action_to_game(self.action.id)
        # The engine routes RESET through step() too (perform_action calls
        # handle_reset and THEN runs the action loop), so guard the budget: a reset
        # restores the level, it must not also cost the player a move. Left
        # unguarded, the first observation of live play is one step further along
        # than the same board's recorded demo, which shows up as a HUD-bar
        # mismatch on frame 0. (flood.py guards the same way.)
        if _action_id != GameAction.RESET:
            self._step_counter.decrement()
        idx = self._current_level_index
        _, _, rail_colors, _, n_rails, rail_length, _, _ = self._level_data[idx]

        if _action_id == GameAction.ACTION7:
            if self._history:
                self._restore(self._history.pop())
            if not self._step_counter.steps_remaining:
                self.lose()
            self.complete_action()
            return

        if _action_id == GameAction.ACTION1:
            old_r = self._selector
            new_r = max(0, self._selector - 1)
            if new_r != old_r:
                # Deselect old
                color = rail_colors[old_r]
                t = self.current_level.get_sprites_by_name(f"tile_{old_r}")
                if t:
                    t[0].pixels = np.array(_tile_pix(color, False), dtype=np.int8)
                self._selector = new_r
                # Select new
                color = rail_colors[new_r]
                t = self.current_level.get_sprites_by_name(f"tile_{new_r}")
                if t:
                    t[0].pixels = np.array(_tile_pix(color, True), dtype=np.int8)
                s = self.current_level.get_sprites_by_name("selector")
                if s:
                    s[0].set_position(0, self._rail_y(new_r))

        elif _action_id == GameAction.ACTION2:
            old_r = self._selector
            new_r = min(n_rails - 1, self._selector + 1)
            if new_r != old_r:
                color = rail_colors[old_r]
                t = self.current_level.get_sprites_by_name(f"tile_{old_r}")
                if t:
                    t[0].pixels = np.array(_tile_pix(color, False), dtype=np.int8)
                self._selector = new_r
                color = rail_colors[new_r]
                t = self.current_level.get_sprites_by_name(f"tile_{new_r}")
                if t:
                    t[0].pixels = np.array(_tile_pix(color, True), dtype=np.int8)
                s = self.current_level.get_sprites_by_name("selector")
                if s:
                    s[0].set_position(0, self._rail_y(new_r))

        elif _action_id in (GameAction.ACTION3, GameAction.ACTION4):
            r = self._selector
            pos = self._tile_positions[r]
            new_pos = pos - 1 if _action_id == GameAction.ACTION3 else pos + 1
            if 0 <= new_pos < rail_length:
                self._history.append(self._snapshot())
                self._tile_positions[r] = new_pos
                t = self.current_level.get_sprites_by_name(f"tile_{r}")
                if t:
                    t[0].set_position(PLAY_X0 + new_pos * CELL, self._rail_y(r))
                if self._check_win():
                    self.next_level()
                    self.complete_action()
                    return

        if not self._step_counter.steps_remaining:
            self.lose()

        self.complete_action()
