# MIT License
#
# Copyright (c) 2026 ARC Prize Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import random as _random_module

from utils.arc_game import AugmentedGame

from arcengine import (
    ARCBaseGame,
    Camera,
    Level,
    RenderableUserDisplay,
    Sprite,
)

BACKGROUND_COLOR = 5
PADDING_COLOR = 5
HIDDEN_COLOR = 3

CAMERA_SIZE = 64
MAX_STEPS = 90

# Triplets: each color appears exactly three times; flip three matching tiles to clear
# one triplet. These are the clean template: per seed the colors are redrawn and the
# slots reshuffled (see _randomize_level), so only the shape of each layout is fixed.
LEVEL_LAYOUTS: list[list[int]] = [
    [8, 8, 8, 9, 9, 9],
    [8, 8, 8, 11, 11, 11, 10, 10, 10],
    [12, 12, 12, 14, 14, 14, 15, 15, 15],
    [8, 8, 8, 9, 9, 9, 11, 11, 11, 10, 10, 10],
    [14, 14, 14, 12, 12, 12, 15, 15, 15, 11, 11, 11],
]

LEVEL_DIMS: list[tuple[int, int]] = [
    (2, 3),
    (3, 3),
    (3, 3),
    (3, 4),
    (3, 4),
]

# Colors the per-seed augmentation may draw from. Excludes 0 and the padding color
# 4 (repo convention), and HIDDEN_COLOR, so no tile or background is ever the same
# color as a face-down tile. The background and the tile palette are drawn mutually
# distinct, so no tile hides against another tile or against the background.
_COLOR_POOL = [c for c in range(1, 16) if c not in (HIDDEN_COLOR, 4)]

# One ordered palette covers every level: a level with k triplets takes its first k
# colors, so an episode's levels share a consistent, nested color scheme.
_MAX_COLORS = max(len(layout) // 3 for layout in LEVEL_LAYOUTS)


def _compute_tile_size_and_offsets(rows: int, cols: int) -> tuple[int, int, int]:
    # Use most of the 64x64 canvas.
    # Leave a small margin so the top time bar doesn't feel cramped.
    target = 56
    tile_size = max(2, target // max(rows, cols))
    grid_w = cols * tile_size
    grid_h = rows * tile_size
    offset_x = (CAMERA_SIZE - grid_w) // 2
    offset_y = (CAMERA_SIZE - grid_h) // 2
    return tile_size, offset_x, offset_y


def make_hidden_sprite(slot_index: int, tile_size: int) -> Sprite:
    pixels = [[HIDDEN_COLOR] * tile_size for _ in range(tile_size)]
    return Sprite(
        pixels=pixels,
        name=f"hidden_{slot_index}",
        visible=True,
        collidable=False,
        tags=["hidden", f"slot_{slot_index}"],
    )


class Mm03UI(RenderableUserDisplay):
    def __init__(self, triplets_remaining: int) -> None:
        self._pairs_remaining = triplets_remaining
        self._steps_remaining = MAX_STEPS
        self._level = 1
        self._click_pos: tuple[int, int] | None = None
        self._click_frames = 0

    def update(self, triplets_remaining: int, steps_remaining: int, level: int = 1) -> None:
        self._pairs_remaining = triplets_remaining
        self._steps_remaining = steps_remaining
        self._level = level

    def set_click(self, x: int, y: int) -> None:
        """Tap marker in final 64×64 frame space (same coords as ACTION6)."""
        self._click_pos = (x, y)
        self._click_frames = 8

    def render_interface(self, frame):
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return frame
        h, w = frame.shape

        bar_width = max(0, min(20, self._steps_remaining * 20 // MAX_STEPS))
        for i in range(bar_width):
            frame[3, 2 + i] = 3

        # Level marker (blue + accent) — does not overlap the 2×2/L2+ tile grids
        frame[1, 2] = 9
        level_colors = [10, 11, 12, 14, 15, 6, 7]
        frame[1, 3] = level_colors[(self._level - 1) % len(level_colors)]

        if self._click_pos and self._click_frames > 0:
            cx, cy = self._click_pos
            if 0 <= cx < w and 0 <= cy < h:
                hit = 11
                for px, py in (
                    (cx, cy),
                    (cx - 1, cy),
                    (cx + 1, cy),
                    (cx, cy - 1),
                    (cx, cy + 1),
                ):
                    if 0 <= px < w and 0 <= py < h:
                        frame[py, px] = hit
            self._click_frames -= 1
        else:
            self._click_pos = None

        return frame


def create_level(level_index: int) -> Level:
    slot_colors = LEVEL_LAYOUTS[level_index]
    rows, cols = LEVEL_DIMS[level_index]

    tile_size, offset_x, offset_y = _compute_tile_size_and_offsets(rows, cols)

    sprites = []
    for slot_idx in range(len(slot_colors)):
        row = slot_idx // cols
        col = slot_idx % cols
        x = offset_x + col * tile_size
        y = offset_y + row * tile_size

        hidden = make_hidden_sprite(slot_idx, tile_size)
        hidden.set_position(x, y)
        sprites.append(hidden)

    return Level(
        sprites=sprites,
        grid_size=(CAMERA_SIZE, CAMERA_SIZE),
        data={
            "slot_colors": slot_colors,
            "num_triplets": len(slot_colors) // 3,
            "rows": rows,
            "cols": cols,
            "tile_size": tile_size,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "level_index": level_index,
        },
        name=f"Level {level_index + 1}",
    )


levels = [create_level(i) for i in range(5)]


class Mm03(AugmentedGame):
    def __init__(self, seed: int | None = None) -> None:
        # FIRST: super().__init__ seats level 0, which calls
        # on_set_level -> level_rng, and that needs the seed installed.
        self._init_augmentation(seed)
        # Set BEFORE super().__init__: it calls set_level(0) -> on_set_level ->
        # _randomize_level, which consumes these RNGs.
        self._rng = self.level_rng("layout")  # per-level: which slot hides which color
        # Per-EPISODE colour scheme, from a separate RNG so a generator can seed it
        # level-independently: the palette and background then stay identical across
        # every level of one episode. Live play draws a fresh scheme per playthrough.
        self._color_rng = self.level_rng("colors")
        self._color_scheme: dict | None = None
        self._ui = Mm03UI(2)
        self._display_level = 1
        super().__init__(
            "mm03",
            levels,
            Camera(
                0,
                0,
                CAMERA_SIZE,
                CAMERA_SIZE,
                BACKGROUND_COLOR,
                PADDING_COLOR,
                [self._ui],
            ),
            False,
            1,
            [6],
        )

    def _ensure_color_scheme(self) -> None:
        """Draw, once per episode, the background and the ordered tile palette.

        A level with k triplets takes the palette's first k colors, so every level of
        one episode shares a single, nested color scheme.
        """
        if self._color_scheme is not None:
            return
        drawn = self._color_rng.sample(_COLOR_POOL, _MAX_COLORS + 1)
        self._color_scheme = {"background": drawn[0], "palette": drawn[1:]}

    def _randomize_level(self, level: Level) -> None:
        """Recolor the level from the episode scheme and reshuffle which slot hides
        which color. Rebuilt from LEVEL_LAYOUTS rather than the level's own data, so
        a repeated set_level on the same level re-randomizes rather than compounds.
        """
        self._ensure_color_scheme()
        self.camera.background = self._color_scheme["background"]
        self.camera.letter_box = self._color_scheme["background"]

        num_triplets = len(LEVEL_LAYOUTS[level.get_data("level_index")]) // 3
        slot_colors = [
            color
            for color in self._color_scheme["palette"][:num_triplets]
            for _ in range(3)
        ]
        self._rng.shuffle(slot_colors)
        level._data["slot_colors"] = slot_colors

    def full_reset(self) -> None:
        # A full reset restarts the game -> a new episode -> a new colour scheme
        # (drawn on the next set_level).
        self._color_scheme = None
        super().full_reset()

    def on_set_level(self, level: Level) -> None:
        # Re-key per level so the board is a pure function of
        # (seed, level), identical whether this level was jumped to
        # or played into. See AugmentedGame.level_rng.
        self._rng = self.level_rng("layout")
        self._color_rng = self.level_rng("colors")
        self._randomize_level(level)
        self._slot_colors = level.get_data("slot_colors")
        self._num_triplets = level.get_data("num_triplets")
        self._rows = level.get_data("rows")
        self._cols = level.get_data("cols")
        self._tile_size = level.get_data("tile_size")
        self._offset_x = level.get_data("offset_x")
        self._offset_y = level.get_data("offset_y")

        num_tiles = len(self._slot_colors)
        self._slots: list[Sprite | None] = [None] * num_tiles
        self._matched = [False] * num_tiles
        self._flipped = []
        self._pairs_remaining = self._num_triplets
        self._steps_remaining = MAX_STEPS
        self._waiting_for_flip_back = False
        self._flip_back_timer = 0

        for sprite in self.current_level.get_sprites_by_tag("hidden"):
            for tag in sprite.tags:
                if tag.startswith("slot_"):
                    slot_idx = int(tag.split("_")[1])
                    self._slots[slot_idx] = sprite

        self._display_level = level.get_data("level_index") + 1
        self._ui.update(self._pairs_remaining, self._steps_remaining, self._display_level)

    def _get_slot_from_click(self, click_x: int, click_y: int):
        col = (click_x - self._offset_x) // self._tile_size
        row = (click_y - self._offset_y) // self._tile_size

        if 0 <= row < self._rows and 0 <= col < self._cols:
            slot_idx = row * self._cols + col
            if 0 <= slot_idx < len(self._slot_colors):
                return row, col, slot_idx
        return None, None, None

    def _flip_slot(self, row: int, col: int) -> bool:
        slot_idx = row * self._cols + col
        if self._matched[slot_idx]:
            return False

        for tile in self._flipped:
            if tile[0] == row and tile[1] == col:
                return False

        if len(self._flipped) >= 3:
            return False

        return True

    def _get_tile_color(self, slot_idx: int) -> int:
        return self._slot_colors[slot_idx]

    def _create_revealed_sprite(self, color: int, slot_idx: int) -> Sprite:
        pixels = [[color] * self._tile_size for _ in range(self._tile_size)]
        return Sprite(
            pixels=pixels,
            name=f"revealed_{slot_idx}",
            visible=True,
            collidable=False,
            tags=["revealed", f"slot_{slot_idx}"],
        )

    def step(self) -> None:
        self._steps_remaining -= 1
        self._ui.update(self._pairs_remaining, self._steps_remaining, self._display_level)

        if self._steps_remaining <= 0:
            self.lose()
            self.complete_action()
            return

        if self._waiting_for_flip_back:
            self._flip_back_timer -= 1
            if self._flip_back_timer <= 0:
                self._do_flip_back()
                self._waiting_for_flip_back = False
            self.complete_action()
            return

        if self.action.id.value == 6:
            x = self.action.data.get("x", 0)
            y = self.action.data.get("y", 0)
            if 0 <= int(x) < CAMERA_SIZE and 0 <= int(y) < CAMERA_SIZE:
                self._ui.set_click(int(x), int(y))

            row, col, slot_idx = self._get_slot_from_click(x, y)
            if row is None:
                self.complete_action()
                return

            if self._matched[slot_idx]:
                self.complete_action()
                return

            if not self._flip_slot(row, col):
                self.complete_action()
                return

            color = self._get_tile_color(slot_idx)
            sprite = self._slots[slot_idx]
            if sprite is None:
                self.complete_action()
                return

            revealed = self._create_revealed_sprite(color, slot_idx)
            revealed.set_position(sprite.x, sprite.y)
            self.current_level.add_sprite(revealed)
            sprite.set_visible(False)

            self._flipped.append((row, col, slot_idx, color))

            if len(self._flipped) == 3:
                c0 = self._flipped[0][3]
                if all(t[3] == c0 for t in self._flipped):
                    for t in self._flipped:
                        self._matched[t[2]] = True
                    self._pairs_remaining -= 1
                    self._flipped = []
                    self._ui.update(
                        self._pairs_remaining, self._steps_remaining, self._display_level
                    )

                    if self._pairs_remaining == 0:
                        self.next_level()
                else:
                    self._waiting_for_flip_back = True
                    self._flip_back_timer = 2

        self.complete_action()

    def _do_flip_back(self) -> None:
        for row, col, slot_idx, color in self._flipped:
            sprite = self._slots[slot_idx]
            if sprite:
                sprite.set_visible(True)

            for s in list(self.current_level._sprites):
                if f"slot_{slot_idx}" in s.tags and "revealed" in s.tags:
                    self.current_level.remove_sprite(s)

        self._flipped = []
