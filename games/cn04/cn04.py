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

import random
from collections import deque

import numpy as np
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)
from numpy import ndarray

from utils.arc_game import AugmentedGame

sprites = {
    "aznnuvumhs": Sprite(
        pixels=[
            [8, -1, 8],
            [13, 13, 13],
            [13, 13, 13],
            [13, -1, 13],
            [8, -1, 8],
        ],
        name="aznnuvumhs",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "dnkiufrohe": Sprite(
        pixels=[
            [8, 14, -1, 14, 8],
            [-1, 14, -1, 14, -1],
            [-1, 14, -1, 14, -1],
            [-1, 14, 14, 14, -1],
        ],
        name="dnkiufrohe",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "dzfylahsaw": Sprite(
        pixels=[
            [-1, 14, 14, 14],
            [-1, 14, -1, 14],
            [8, 14, -1, 14],
            [-1, 14, -1, 14],
            [8, 14, -1, 14],
            [-1, 14, -1, 14],
            [-1, 14, 14, 14],
        ],
        name="dzfylahsaw",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "eetyyobbee": Sprite(
        pixels=[
            [12, 12, 12, 12, 12],
            [12, 8, -1, 8, 12],
        ],
        name="eetyyobbee",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "ezhdijgxgn": Sprite(
        pixels=[
            [8, -1, -1, -1, -1, -1, -1, -1],
            [6, -1, -1, -1, -1, -1, -1, -1],
            [6, -1, -1, -1, -1, -1, -1, -1],
            [6, -1, -1, -1, -1, -1, -1, -1],
            [6, 6, 6, 6, 6, 6, 6, 8],
        ],
        name="ezhdijgxgn",
        visible=True,
        collidable=True,
    ),
    "fcdadbjhyk": Sprite(
        pixels=[
            [-1, -1, -1, -1, 8, -1],
            [-1, 10, 10, 10, 10, 10],
            [-1, 10, -1, -1, -1, 10],
            [-1, 10, -1, -1, 10, 10],
            [8, 10, -1, 10, 10, 8],
            [-1, 10, 10, 10, 8, -1],
        ],
        name="fcdadbjhyk",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "fjxymoivnf": Sprite(
        pixels=[
            [-1, -1, -1, -1, -1, 8, -1, 8, -1, -1, -1],
            [-1, -1, -1, -1, 11, 11, 11, 11, 11, -1, -1],
            [-1, -1, -1, -1, 11, 11, 11, 11, 11, -1, -1],
            [-1, -1, -1, 11, 11, 11, 11, -1, -1, -1, -1],
            [-1, -1, -1, 11, 11, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, 11, 11, -1, -1, -1, 11, -1, -1],
            [-1, 8, -1, 11, 11, -1, -1, 11, 11, 11, -1],
            [8, 11, 11, 11, 11, -1, -1, 11, 11, 11, 8],
            [-1, 11, 11, 11, 11, 11, 11, 11, 11, 11, -1],
            [-1, 11, 11, 11, 11, 11, 11, 11, 11, 11, -1],
            [-1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1],
        ],
        name="fjxymoivnf",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "glojtydbea": Sprite(
        pixels=[
            [-1, -1, 8, 14],
            [-1, -1, -1, 14],
            [-1, -1, -1, 14],
            [-1, -1, -1, 14],
            [-1, -1, -1, 14],
            [8, -1, -1, 14],
            [14, 14, 14, 14],
        ],
        name="glojtydbea",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "gqxqpkuwab": Sprite(
        pixels=[
            [11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11],
            [-1, 8, -1, 8, -1],
        ],
        name="gqxqpkuwab",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "grpgefeksy": Sprite(
        pixels=[
            [-1, -1, -1, 11, 11, 11, -1],
            [-1, 8, -1, 11, -1, 11, 8],
            [-1, 11, 11, 11, -1, 11, -1],
            [8, 11, -1, -1, -1, 11, -1],
            [-1, 11, 11, 11, -1, 11, -1],
            [-1, -1, -1, 11, 11, 11, -1],
            [-1, -1, -1, -1, 8, -1, -1],
        ],
        name="grpgefeksy",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "hjqjqsmhmf": Sprite(
        pixels=[
            [12, 12, -1],
            [12, 12, 8],
            [12, 12, -1],
            [12, 12, 8],
            [12, 12, -1],
        ],
        name="hjqjqsmhmf",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "kddtjradhp": Sprite(
        pixels=[
            [7, 7, 7, 7, 8],
            [7, -1, -1, -1, -1],
            [7, -1, -1, -1, -1],
            [7, -1, -1, -1, -1],
            [8, -1, -1, -1, -1],
        ],
        name="kddtjradhp",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "kodeocvgpm": Sprite(
        pixels=[
            [-1, -1, -1, -1, -1, 8],
            [-1, -1, -1, -1, -1, 7],
            [-1, -1, -1, -1, -1, 7],
            [-1, -1, -1, -1, -1, 7],
            [8, 7, 7, 7, 7, 7],
        ],
        name="kodeocvgpm",
        visible=True,
        collidable=True,
    ),
    "lejuhuvrjg": Sprite(
        pixels=[
            [15, 15, 15],
            [15, -1, 15],
            [15, -1, 15],
            [15, -1, 8],
            [15, -1, -1],
            [15, 8, -1],
        ],
        name="lejuhuvrjg",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "lxgvcnpsir": Sprite(
        pixels=[
            [8, 12, 12, 12],
            [-1, 12, -1, 12],
            [-1, 12, -1, 12],
            [-1, 12, -1, 12],
            [-1, 12, -1, 12],
            [8, 12, 12, 12],
        ],
        name="lxgvcnpsir",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "nhppemmlcd": Sprite(
        pixels=[
            [-1, 8, -1, 8, -1],
            [10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10],
            [-1, 8, -1, 8, -1],
        ],
        name="nhppemmlcd",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "oncjjftokv": Sprite(
        pixels=[
            [8],
            [11],
            [11],
            [11],
            [8],
        ],
        name="oncjjftokv",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "rlaifclqmn": Sprite(
        pixels=[
            [-1, -1, -1, 8, 14],
            [-1, -1, -1, -1, 14],
            [-1, -1, -1, -1, 14],
            [8, -1, -1, -1, 14],
            [14, -1, -1, -1, 14],
            [14, 14, 14, 14, 14],
        ],
        name="rlaifclqmn",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "supigciwwi": Sprite(
        pixels=[
            [10, 10, 10, 10, 10, 8],
            [10, -1, -1, -1, 10, -1],
            [10, -1, -1, -1, 10, -1],
            [10, -1, -1, -1, 10, -1],
            [10, -1, -1, -1, -1, -1],
            [10, 10, 10, 10, 10, 8],
        ],
        name="supigciwwi",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "udepuflbbb": Sprite(
        pixels=[
            [8, -1, -1, -1, -1, -1, 8],
            [14, -1, -1, -1, -1, -1, 14],
            [14, 14, 14, -1, 14, 14, 14],
            [-1, -1, 14, -1, 14, -1, -1],
            [-1, -1, 14, 14, 14, -1, -1],
        ],
        name="udepuflbbb",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "vmgdxvpgkh": Sprite(
        pixels=[
            [12, 12, 12, -1, -1, -1],
            [12, -1, 12, 12, 12, 8],
            [12, -1, -1, -1, -1, -1],
            [12, -1, 12, 12, 12, 8],
            [12, 12, 12, -1, -1, -1],
        ],
        name="vmgdxvpgkh",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "ygyvhkssnz": Sprite(
        pixels=[
            [15, 15, 15, 15, 15, 15, -1],
            [15, -1, -1, -1, -1, 15, 8],
            [15, -1, 15, 15, 15, 15, -1],
            [15, -1, 15, -1, -1, -1, -1],
            [15, -1, 15, -1, -1, -1, -1],
            [15, 15, 15, -1, -1, -1, -1],
            [-1, 8, -1, -1, -1, -1, -1],
        ],
        name="ygyvhkssnz",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
    "zhhaofpbyw": Sprite(
        pixels=[
            [8, 11],
            [-1, 11],
            [-1, 11],
            [-1, 11],
            [-1, 11],
            [-1, 8],
        ],
        name="zhhaofpbyw",
        visible=True,
        collidable=True,
        tags=["sys_click"],
    ),
}


# ---------------------------------------------------------------------------
# Randomization helpers
# ---------------------------------------------------------------------------

def _cn04_is_connected(cells: set) -> bool:
    """Check if a set of (x, y) cells forms a single connected component."""
    if not cells:
        return True
    start = next(iter(cells))
    visited = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nb = (x + dx, y + dy)
            if nb in cells and nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited) == len(cells)


def _cn04_randomize_shape(pixels_2d: list, rng: random.Random) -> list:
    """Randomize the body cells of a sprite while keeping red (color-8) pixels fixed.

    Keeps all red connection-point pixels at their exact positions.  The
    colored body cells are lightly perturbed: some are randomly removed and
    some new adjacent cells are randomly added, as long as the overall region
    (body + red points) stays connected.
    """
    H = len(pixels_2d)
    if H == 0:
        return pixels_2d
    W = len(pixels_2d[0])

    red_positions: set[tuple[int, int]] = set()
    body_positions: set[tuple[int, int]] = set()
    body_color: int | None = None

    for y in range(H):
        for x in range(W):
            c = pixels_2d[y][x]
            if c == 8:
                red_positions.add((x, y))
            elif c != -1:
                body_positions.add((x, y))
                if body_color is None:
                    body_color = c

    if not red_positions or not body_positions or body_color is None:
        return pixels_2d

    current_body = set(body_positions)

    # Try removing ~30% of body cells (only if connectivity is preserved)
    removal_candidates = list(current_body)
    rng.shuffle(removal_candidates)
    for cell in removal_candidates:
        if rng.random() > 0.3:
            continue
        test_body = current_body - {cell}
        if _cn04_is_connected(test_body | red_positions):
            current_body = test_body

    # Try adding some adjacent empty cells to the body
    all_cells = current_body | red_positions
    adjacents: list[tuple[int, int]] = []
    seen_adj: set[tuple[int, int]] = set()
    for x, y in all_cells:
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nb = (x + dx, y + dy)
            nx, ny = nb
            if 0 <= nx < W and 0 <= ny < H and nb not in all_cells and nb not in seen_adj:
                adjacents.append(nb)
                seen_adj.add(nb)
    rng.shuffle(adjacents)
    n_add = rng.randint(0, max(0, len(adjacents) // 4))
    for cell in adjacents[:n_add]:
        current_body.add(cell)

    # Build new pixel array
    new_pixels = [[-1] * W for _ in range(H)]
    for x, y in current_body:
        new_pixels[y][x] = body_color
    for x, y in red_positions:
        new_pixels[y][x] = 8

    return new_pixels


def _cn04_clone_level(level: Level) -> Level:
    """Return a deep copy of a level with all sprites cloned."""
    cloned_sprites = [s.clone().set_position(s.x, s.y) for s in level._sprites]
    data = dict(level._data) if level._data else {}
    return Level(sprites=cloned_sprites, grid_size=level.grid_size, data=data)


def _build_cn04_randomized_levels(base_levels: list, rng: random.Random) -> list:
    """Return randomized copies of base_levels.

    For each level:
    - Shape body pixels are lightly perturbed while red connection points stay fixed.
    - Initial sprite positions are randomized within the grid bounds.
    """
    randomized = [_cn04_clone_level(lvl) for lvl in base_levels]
    for lvl in randomized:
        if lvl.grid_size is None:
            continue
        gw, gh = lvl.grid_size
        for sprite in lvl._sprites:
            # Randomize the shape (body only; red connection points are preserved)
            raw = sprite.pixels.tolist()
            new_raw = _cn04_randomize_shape(raw, rng)
            sprite.pixels = np.array(new_raw, dtype=sprite.pixels.dtype)

            # Randomize starting position within grid bounds
            sw, sh = sprite.width, sprite.height
            max_x = max(0, gw - sw)
            max_y = max(0, gh - sh)
            sprite.set_position(rng.randint(0, max_x), rng.randint(0, max_y))
    return randomized


levels = [
    # Level 1
    Level(
        sprites=[
            sprites["dzfylahsaw"].clone().set_position(12, 9),
            sprites["vmgdxvpgkh"].clone().set_position(4, 4),
        ],
        grid_size=(20, 20),
        data={
            "BackgroundColour": 10,
        },
    ),
    # Level 2
    Level(
        sprites=[
            sprites["glojtydbea"].clone().set_position(12, 4),
            sprites["grpgefeksy"].clone().set_position(4, 11),
            sprites["lejuhuvrjg"].clone().set_position(3, 3),
        ],
        grid_size=(20, 20),
        data={
            "BackgroundColour": 12,
        },
    ),
    # Level 3
    Level(
        sprites=[
            sprites["gqxqpkuwab"].clone().set_position(11, 4),
            sprites["hjqjqsmhmf"].clone().set_position(5, 5),
            sprites["nhppemmlcd"].clone().set_position(9, 11),
        ],
        grid_size=(20, 20),
        data={
            "BackgroundColour": 15,
        },
    ),
    # Level 4
    Level(
        sprites=[
            sprites["udepuflbbb"].clone().set_position(2, 4).set_rotation(90),
            sprites["ygyvhkssnz"].clone().set_position(9, 9).set_rotation(270),
            sprites["zhhaofpbyw"].clone().set_position(10, 3).set_rotation(90),
        ],
        grid_size=(20, 20),
        data={
            "BackgroundColour": 12,
        },
    ),
    # Level 5
    Level(
        sprites=[
            sprites["dnkiufrohe"].clone().set_position(11, 6),
            sprites["lxgvcnpsir"].clone().set_position(14, 13),
            sprites["oncjjftokv"].clone().set_position(5, 15).set_rotation(90),
            sprites["supigciwwi"].clone().set_position(2, 3).set_rotation(90),
        ],
        grid_size=(20, 20),
        data={
            "BackgroundColour": 9,
        },
    ),
]
BACKGROUND_COLOR = 4
PADDING_COLOR = 4


class qdcvayjdkm(RenderableUserDisplay):
    """."""

    def __init__(self, mimisncrjk: int):
        self.mimisncrjk = mimisncrjk
        self.current_steps = mimisncrjk

    def ivinkcxarj(self, ricxiwpfxg: int) -> None:
        self.current_steps = max(0, min(ricxiwpfxg, self.mimisncrjk))

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.mimisncrjk == 0:
            return frame
        hsoumyqjja = 32
        x_offset = (64 - hsoumyqjja) // 2
        xvawzpalbx = self.current_steps / self.mimisncrjk
        ukcleokzfb = round(hsoumyqjja * xvawzpalbx)
        ukcleokzfb = min(ukcleokzfb, hsoumyqjja)
        ckkdtrybdj = hsoumyqjja - ukcleokzfb
        for x in range(hsoumyqjja):
            if x < ckkdtrybdj:
                frame[0, x_offset + x] = 0
            else:
                frame[0, x_offset + x] = 4
        return frame


class Cn04(AugmentedGame):
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        randomized_levels = _build_cn04_randomized_levels(levels, self._rng)
        pzqnb = Camera(background=BACKGROUND_COLOR, letter_box=PADDING_COLOR)
        self.weqid: Sprite | None = None
        self.agupi: dict[str, ndarray] = {}
        self.npwwu: dict[str, ndarray] = {}
        self.dpmge: dict[str, set[tuple[int, int]]] = {}
        self.mctam = False
        self.guweh = 150
        self._step_counter_ui = qdcvayjdkm(mimisncrjk=self.guweh)
        self.fubdf = False
        self._init_augmentation(seed)
        pzqnb.replace_interface([self._step_counter_ui])
        super().__init__(game_id="cn04", levels=randomized_levels, camera=pzqnb, available_actions=[1, 2, 3, 4, 5, 6])

    def on_set_level(self, level: Level) -> None:
        """."""
        self.fubdf = False
        self._step_counter_ui.ivinkcxarj(self.guweh)
        gnkzqsghnz = level.get_data("BackgroundColour")
        if gnkzqsghnz is not None:
            self.camera.letter_box = gnkzqsghnz
            self.camera.background = gnkzqsghnz
        else:
            self.camera.letter_box = BACKGROUND_COLOR
            self.camera.background = BACKGROUND_COLOR
        self.mctam = self._current_level_index >= 4
        self.weqid = None
        self.agupi = {}
        self.npwwu = {}
        for fpyzd in level.get_sprites():
            self.npwwu[fpyzd.name] = fpyzd.pixels.copy()
            self.agupi[fpyzd.name] = fpyzd.pixels.copy()
            if self.mctam:
                fpyzd.pixels = np.where(fpyzd.pixels >= 0, 4, fpyzd.pixels)
        if self._current_level_index >= 4:
            for fpyzd in level.get_sprites():
                fpyzd.pixels = np.where(fpyzd.pixels >= 0, 4, fpyzd.pixels)
        zrriflazox = [s for s in level.get_sprites()]
        if zrriflazox:
            kxhwgaavgp = min(zrriflazox, key=lambda s: s.x**2 + s.y**2)
            self.lceflskdhw(kxhwgaavgp)
        self.gjhtwbvrel()

    def lceflskdhw(self, sprite: Sprite) -> None:
        """."""
        if self.weqid:
            self.pvrlqzlpjy()
        self.weqid = sprite
        self.weqid.set_layer(4)
        self.agupi[sprite.name] = sprite.pixels.copy()
        if self._current_level_index >= 4:
            sprite.pixels = self.npwwu[sprite.name].copy()
        else:
            sprite.pixels = np.where(
                (sprite.pixels >= 0) & (sprite.pixels != 8) & (sprite.pixels != 3),
                0,
                sprite.pixels,
            )
        self.gjhtwbvrel()

    def pvrlqzlpjy(self) -> None:
        """."""
        if self.weqid and self.weqid.name in self.agupi:
            self.weqid.pixels = self.agupi[self.weqid.name]
            del self.agupi[self.weqid.name]
            if self._current_level_index >= 4:
                self.weqid.pixels = np.where(
                    (self.weqid.pixels >= 0) & (self.weqid.pixels != 8),
                    4,
                    self.weqid.pixels,
                )
            self.weqid.set_layer(0)
        self.weqid = None

    def srvkjkhjla(self, sprite: Sprite, dx: int, dy: int) -> bool:
        """."""
        original_x = sprite.x
        original_y = sprite.y
        sprite.move(dx, dy)
        pixels = sprite.render()
        xlmrkximft = []
        for y in range(pixels.shape[0]):
            for x in range(pixels.shape[1]):
                if pixels[y, x] == 8:
                    xlmrkximft.append((sprite.x + x, sprite.y + y))
        ptssbzxhow = False
        for other in self.current_level.get_sprites():
            if other == sprite:
                continue
            other_pixels = other.render()
            for y in range(other_pixels.shape[0]):
                for x in range(other_pixels.shape[1]):
                    if other_pixels[y, x] == 8:
                        dbvmxmynon = (other.x + x, other.y + y)
                        if dbvmxmynon in xlmrkximft:
                            ptssbzxhow = True
                            break
                if ptssbzxhow:
                    break
        sprite.set_position(original_x, original_y)
        return ptssbzxhow

    def gjhtwbvrel(self) -> None:
        """."""
        sprites = self.current_level.get_sprites()
        self.dpmge = {}
        for sprite in sprites:
            if sprite == self.weqid:
                if self._current_level_index >= 4:
                    sprite.pixels = self.npwwu[sprite.name].copy()
                else:
                    xraldtaupp = self.npwwu[sprite.name]
                    sprite.pixels = xraldtaupp.copy()
                    sprite.pixels = np.where(
                        (sprite.pixels >= 0) & (sprite.pixels != 8) & (sprite.pixels != 3),
                        0,
                        sprite.pixels,
                    )
            else:
                sprite.pixels = self.npwwu[sprite.name].copy()
                if self._current_level_index >= 4:
                    sprite.pixels = np.where(sprite.pixels >= 0, 4, sprite.pixels)
        kwcxemeyuq: dict[tuple[int, int], list[tuple[Sprite, int, int]]] = {}
        for sprite in sprites:
            vwwkmczipq = self.npwwu[sprite.name]
            hrynucwkzb = sprite.pixels.copy()
            sprite.pixels = vwwkmczipq
            fbclbzwimr = sprite.render()
            sprite.pixels = hrynucwkzb
            for y in range(fbclbzwimr.shape[0]):
                for x in range(fbclbzwimr.shape[1]):
                    if fbclbzwimr[y, x] == 8:
                        hpiaagjpaz = sprite.x + x
                        dphrmwlkdz = sprite.y + y
                        if (hpiaagjpaz, dphrmwlkdz) not in kwcxemeyuq:
                            kwcxemeyuq[hpiaagjpaz, dphrmwlkdz] = []
                        kwcxemeyuq[hpiaagjpaz, dphrmwlkdz].append((sprite, x, y))
        for dbvmxmynon, amkqyhsyhd in kwcxemeyuq.items():
            if len(amkqyhsyhd) == 2:
                for sprite, xzezjuwdcj, nazicaigmb in amkqyhsyhd:
                    if sprite.name not in self.dpmge:
                        self.dpmge[sprite.name] = set()
                    self.dpmge[sprite.name].add((xzezjuwdcj, nazicaigmb))
        for sprite in sprites:
            if sprite.name in self.dpmge:
                ebpkesqfpj = set()
                for x, y in self.dpmge[sprite.name]:
                    if sprite.rotation == 0:
                        ebpkesqfpj.add((x, y))
                    elif sprite.rotation == 90:
                        rncmmybetj = y
                        psxrbynyaz = sprite.pixels.shape[0] - 1 - x
                        ebpkesqfpj.add((rncmmybetj, psxrbynyaz))
                    elif sprite.rotation == 180:
                        rncmmybetj = sprite.pixels.shape[1] - 1 - x
                        psxrbynyaz = sprite.pixels.shape[0] - 1 - y
                        ebpkesqfpj.add((rncmmybetj, psxrbynyaz))
                    elif sprite.rotation == 270:
                        rncmmybetj = sprite.pixels.shape[1] - 1 - y
                        psxrbynyaz = x
                        ebpkesqfpj.add((rncmmybetj, psxrbynyaz))
                for x, y in ebpkesqfpj:
                    if 0 <= y < sprite.pixels.shape[0] and 0 <= x < sprite.pixels.shape[1]:
                        if sprite.pixels[y, x] == 8:
                            sprite.pixels[y, x] = 3
            if self._current_level_index >= 4 and sprite != self.weqid:
                zwhopgpbgr = self.npwwu[sprite.name] == 8
                sprite.pixels = np.where(zwhopgpbgr & (sprite.pixels != 3), 4, sprite.pixels)

    def exlcvhdjsf(self) -> bool:
        """."""
        for sprite in self.current_level.get_sprites():
            rurikqkuoc = self.npwwu[sprite.name].copy()
            if sprite.name in self.dpmge:
                for x, y in self.dpmge[sprite.name]:
                    if sprite.rotation == 0:
                        rncmmybetj, psxrbynyaz = (x, y)
                    elif sprite.rotation == 90:
                        rncmmybetj = y
                        psxrbynyaz = rurikqkuoc.shape[0] - 1 - x
                    elif sprite.rotation == 180:
                        rncmmybetj = rurikqkuoc.shape[1] - 1 - x
                        psxrbynyaz = rurikqkuoc.shape[0] - 1 - y
                    elif sprite.rotation == 270:
                        rncmmybetj = rurikqkuoc.shape[1] - 1 - y
                        psxrbynyaz = x
                    else:
                        continue
                    if 0 <= psxrbynyaz < rurikqkuoc.shape[0] and 0 <= rncmmybetj < rurikqkuoc.shape[1]:
                        if rurikqkuoc[psxrbynyaz, rncmmybetj] == 8:
                            rurikqkuoc[psxrbynyaz, rncmmybetj] = 3
            if np.any(rurikqkuoc == 8):
                return False
        return True

    def step(self) -> None:
        """."""
        self._step_counter_ui.ivinkcxarj(self.guweh - self._action_count)
        if self._action_count >= self.guweh:
            self.lose()
            self.complete_action()
            return
        _action_id = self.screen_action_to_game(self.action.id)
        if _action_id == GameAction.ACTION6:
            ngkiv = self.action.data["x"]
            vnepx = self.action.data["y"]
            ngkiv, vnepx = self.screen_click_to_game(ngkiv, vnepx)
            gvrxddijis = self.camera.display_to_grid(int(ngkiv), int(vnepx))
            if gvrxddijis:
                dqudz, uermy = gvrxddijis
                quyphwtvrr = self.current_level.get_sprite_at(dqudz, uermy, ignore_collidable=True)
                if quyphwtvrr:
                    if quyphwtvrr == self.weqid:
                        self.pvrlqzlpjy()
                    else:
                        self.lceflskdhw(quyphwtvrr)
            self.fubdf = True
        elif _action_id == GameAction.ACTION5:
            if self.weqid:
                self.weqid.rotate(90)
                self.gjhtwbvrel()
                if self.exlcvhdjsf():
                    self.next_level()
            self.fubdf = False
        elif _action_id in [
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        ]:
            if self.weqid:
                wivja = 0
                qebim = 0
                if _action_id == GameAction.ACTION1:
                    qebim = -1
                elif _action_id == GameAction.ACTION2:
                    qebim = 1
                elif _action_id == GameAction.ACTION3:
                    wivja = -1
                elif _action_id == GameAction.ACTION4:
                    wivja = 1
                if self.current_level.grid_size is None:
                    return
                ckdfh, uehyx = self.current_level.grid_size
                lvjzdgbgnk = self.weqid.x + wivja
                fhksnihjqx = self.weqid.y + qebim
                if lvjzdgbgnk >= 0 and fhksnihjqx >= 0 and (lvjzdgbgnk + self.weqid.width <= ckdfh) and (fhksnihjqx + self.weqid.height <= uehyx):
                    self.weqid.move(wivja, qebim)
                self.gjhtwbvrel()
                if self.exlcvhdjsf():
                    self.next_level()
            self.fubdf = False
        self.complete_action()
