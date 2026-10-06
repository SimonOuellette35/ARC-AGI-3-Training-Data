"""3x3 Quad Twist -- ACTION6 rotates the 3x3 block anchored at the clicked cell 90 degrees
clockwise; match the 8x8 board to the target key drawn in the HUD.

Mechanic (unchanged from the original): a click selects a grid cell, that cell is the TOP-LEFT
anchor of a 3x3 block, and the nine tiles in it are cycled one quarter-turn clockwise. Tile
values never change -- a move only *permutes positions* -- so the board's colour multiset is
invariant. A click whose block would run off the board (anchor x or y > 5) or would cover a wall
is rejected (red flash, no step charged). Only a real rotation costs a step; running out of
steps loses.

Why the shipped levels had to be replaced
-----------------------------------------
The original level table paired a diagonal-stripe ``init`` with a *shifted* diagonal-stripe
``target`` (``_board(z) -> _board(z')``), which made **all seven levels mathematically
unwinnable**:

* A rotation moves the tile at ``(gx+j, gy+2-i)`` to ``(gx+i, gy+j)``, so the quantity
  ``q = value - x - y`` carried by a tile changes by ``2 - 2i (mod 4)`` -- always 0 or 2. Hence
  ``q mod 2`` is an **invariant of every tile**. On ``_board(z)`` every tile has ``q = z``, and
  every cell of ``_board(z')`` demands ``q = z'``; so when ``z`` and ``z'`` differ in parity NO
  cell can ever match, let alone all of them. That killed the old levels 1, 2, 4, 5 and 6
  (a beam search confirms the mismatch count never drops below its initial 64/56).
* Old level 3 asked for all-0 -> all-1, which no position permutation can do.
* Old levels 4 and 7 walled the board so that one non-wall cell ((4,4) and (0,0)) sits in no
  legal 3x3 block at all, yet its ``init`` value differed from its ``target`` value -- frozen
  forever on the wrong colour.

So the level table now *builds* each board the only way that guarantees reachability: draw the
TARGET, then produce the starting board by applying ``scramble`` counter-clockwise rotations to
it. The optimal solution is then at most ``scramble`` clockwise clicks (the exact scramble run
backwards), and the difficulty of a level is exactly that number.

Per-(seed, level) augmentation (see utils/arc_game.py)
-----------------------------------------------------
``Qr04(seed=S)`` is a pure function of ``(S, level_index)``:

* **display rotation** k in {0,1,2,3} -- owned by :class:`AugmentedGame`; clicks are mapped back
  with ``screen_click_to_game``.
* **target key** -- a diagonal-stripe pattern with a random slope (either diagonal) and offset,
  drawn per (seed, level).
* **start board** -- ``scramble`` counter-clockwise rotations at distinct random legal anchors,
  redrawn until the board differs from the target in enough cells to be plainly visible.
* **tile palette** -- 4 distinct colours drawn ONCE per episode (level-independent, so a whole
  playthrough shares one scheme), avoiding the wall / background / padding / HUD colours.

Walls stay fixed per level: they are the level's identity and they shape which anchors are legal.
An unseeded ``Qr04()`` keeps drawing fresh boards each run, for interactive play.
"""

from __future__ import annotations

import random

from arcengine import (
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)

from utils.arc_game import AugmentedGame

BACKGROUND_COLOR = 5
PADDING_COLOR = 4
GW = GH = 8
#: Camera viewport, in grid cells. MUST also be the level's ``grid_size``: ``set_level`` resizes
#: the camera to ``grid_size``, so declaring the level as 8x8 (as the original did) blew the board
#: up to fill all 64x64 px -- and the HUD, drawn at fixed frame coordinates, then landed ON the
#: board and overwrote the centre pixel of cell (0,7), a cell the win check reads. At 16 the board
#: renders 4 px/cell in the top-left 32x32 and the HUD has the rest of the frame to itself.
CAM = 16
WALL_C = 3
BAR_C = 10          # steps-remaining bar
REJECT_C = 8        # illegal-click flash
CLICK_C = 11        # click marker
N_VALUES = 4        # tile values 0..3

#: Target-key mini-board: 2 px per cell at this frame offset (bottom-right, clear of the board).
#: The original drew it 1 px per cell, which is a 64-pixel legend an agent has to read cell by
#: cell against a 4 px/cell board; 2 px costs nothing here and there is empty frame to spare.
KEY_PX = 2
KEY_X = 34
KEY_Y = 34

#: Tile-colour pool: everything except the wall (3), padding (4), background (5) and the three
#: HUD colours (8, 10, 11), so a tile can never be confused with the furniture.
TILE_POOL = (1, 2, 6, 7, 9, 12, 13, 14, 15)

#: Fallback palette used when no palette has been drawn yet (mirrors the original BASE list).
BASE = (2, 6, 9, 11)

#: How many redraws to spend looking for a scramble that is visibly different from the target.
_SCRAMBLE_TRIES = 40


class Qr04UI(RenderableUserDisplay):
    """HUD: the target key (bottom-left 8x8 mini-board), a steps-remaining bar, the
    illegal-click flash and a one-frame marker on the cell just clicked."""

    def __init__(self, steps: int, target: list[list[int]], palette=BASE,
                 walls: frozenset[tuple[int, int]] = frozenset()) -> None:
        self._steps = steps
        self._target = target
        self._palette = tuple(palette)
        self._walls = walls
        self._reject_frames = 0
        self._click: tuple[int, int] | None = None

    def update(self, steps: int, target: list[list[int]], palette=None,
               walls: frozenset[tuple[int, int]] | None = None) -> None:
        self._steps = steps
        self._target = target
        if palette is not None:
            self._palette = tuple(palette)
        if walls is not None:
            self._walls = walls

    def set_click(self, fx: int, fy: int) -> None:
        self._click = (fx, fy)

    def flash_reject(self, frames: int = 1) -> None:
        # ONE frame, not the original eight: every action here renders exactly one frame (a
        # rotation is instantaneous), so an 8-frame flash used to bleed the "illegal click"
        # marker across the next eight *actions* and label perfectly legal moves as rejected.
        self._reject_frames = frames

    def render_interface(self, frame):
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return frame
        h, w = frame.shape
        for yy in range(GH):
            for xx in range(GW):
                # Wall cells are exempt from the win check and can never move, so the key
                # shows them as walls rather than demanding a colour the player can't place.
                colour = (WALL_C if (xx, yy) in self._walls
                          else int(self._palette[self._target[yy][xx] % N_VALUES]))
                for dy in range(KEY_PX):
                    for dx in range(KEY_PX):
                        px, py = KEY_X + xx * KEY_PX + dx, KEY_Y + yy * KEY_PX + dy
                        if 0 <= px < w and 0 <= py < h:
                            frame[py, px] = colour
        for i in range(min(self._steps, 15)):
            frame[h - 2, 1 + i] = BAR_C
        if self._reject_frames > 0:
            for dx in range(3):
                for dy in range(2):
                    px, py = w - 3 + dx, dy
                    if 0 <= px < w and 0 <= py < h:
                        frame[py, px] = REJECT_C
            self._reject_frames -= 1
        if self._click:
            cx, cy = self._click
            for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                px, py = cx + dx, cy + dy
                if 0 <= px < w and 0 <= py < h:
                    frame[py, px] = CLICK_C
            self._click = None
        return frame


def cell_sprite(st: int, palette=BASE) -> Sprite:
    return Sprite(
        pixels=[[int(palette[st % N_VALUES])]],
        name="qt",
        visible=True,
        collidable=False,
        tags=["qtile"],
    )


def mk(walls: list[tuple[int, int]], scramble: int, diff: int) -> Level:
    """One level: a wall layout plus how many rotations the start board is scrambled by.

    The step budget is derived from the scramble depth (the optimal plan is at most that many
    clicks), leaving room to fumble but not to brute-force.
    """
    sl = []
    for wx, wy in walls:
        sl.append(
            Sprite(
                pixels=[[WALL_C]],
                name="wall",
                visible=True,
                collidable=True,
                tags=["wall"],
            ).set_position(wx, wy),
        )
    return Level(
        sprites=sl,
        grid_size=(CAM, CAM),
        data={
            "difficulty": diff,
            "scramble": scramble,
            "max_steps": 4 * scramble + 20,
        },
    )


# Walls unchanged from the original table; only the boards are now generated. `scramble` is the
# level's difficulty: the optimal solution is at most that many clicks.
levels = [
    mk([], 1, 1),
    mk([(4, y) for y in range(8)], 2, 2),
    mk([], 3, 3),
    mk([(x, 4) for x in range(8) if x != 4], 3, 4),
    mk([], 4, 5),
    mk([(0, y) for y in range(8)], 4, 6),
    mk([(7, y) for y in range(8)] + [(x, 0) for x in range(1, 7)], 5, 7),
]


def legal_anchors(walls) -> list[tuple[int, int]]:
    """Anchors whose 3x3 block fits on the board and covers no wall -- i.e. every click the
    engine will actually act on. Shared with the training-data generator's search."""
    return [
        (x, y)
        for y in range(GH - 2)
        for x in range(GW - 2)
        if not any((x + dx, y + dy) in walls for dx in range(3) for dy in range(3))
    ]


def rotate_cw(grid: list[list[int]], gx: int, gy: int) -> None:
    """Rotate the 3x3 block anchored at ``(gx, gy)`` a quarter-turn clockwise, in place.

    THE definition of the game's mechanic: ``step`` calls it for a legal click and the level
    builder calls it (three times = once counter-clockwise) to scramble a target into a start
    board, so the puzzle can never be generated unsolvable.
    """
    sub = [[grid[gy + j][gx + i] for i in range(3)] for j in range(3)]
    for j in range(3):
        for i in range(3):
            grid[gy + j][gx + i] = sub[2 - i][j]


class Qr04(AugmentedGame):
    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)
        # Episode-scoped (level-INDEPENDENT) palette, so one playthrough keeps one colour scheme.
        self._palette = self._draw_palette()
        self._walls: frozenset[tuple[int, int]] = frozenset()
        self._anchors: list[tuple[int, int]] = []
        z = [[0] * GW for _ in range(GH)]
        self._ui = Qr04UI(0, z, self._palette)
        super().__init__(
            "qr04",
            levels,
            Camera(0, 0, CAM, CAM, BACKGROUND_COLOR, PADDING_COLOR,
                   [self._ui]),
            False,
            1,
            [6],
        )

    # ── per-seed augmentation ───────────────────────────────────────────────
    def _draw_palette(self) -> tuple[int, ...]:
        rng = (random.Random() if self._seed_value is None
               else random.Random(f"qr04-colours:{int(self._seed_value)}"))
        return tuple(rng.sample(TILE_POOL, N_VALUES))

    def _board_rng(self, level_index: int) -> random.Random:
        if self._seed_value is None:
            return random.Random()
        return random.Random(f"qr04-board:{int(self._seed_value)}:{int(level_index)}")

    def _draw_target(self, rng: random.Random) -> list[list[int]]:
        """A diagonal-stripe key: slope +-1 in each axis and a random offset (16 variants).

        Stripes keep every 3x3 block multi-coloured, so a single rotation is always plainly
        visible -- which is what makes the puzzle readable from the frame.
        """
        sx = rng.choice((1, N_VALUES - 1))
        sy = rng.choice((1, N_VALUES - 1))
        z = rng.randrange(N_VALUES)
        return [[(sx * x + sy * y + z) % N_VALUES for x in range(GW)] for y in range(GH)]

    def _scramble(self, target: list[list[int]], rng: random.Random,
                  n: int) -> list[list[int]]:
        """Start board = ``target`` with ``n`` counter-clockwise rotations applied at DISTINCT
        random legal anchors.

        Counter-clockwise (three clockwise turns) is what makes the level's difficulty exact: the
        n moves run backwards are n *clockwise* clicks, so the optimal plan is at most n long.
        Distinct anchors stop a move from being undone by a repeat of itself; the redraw loop then
        rejects draws whose OVERLAPPING blocks happened to cancel most of each other, which would
        leave a board that is barely distinguishable from the key. Falls back to the most visible
        attempt if the loop runs out."""
        n = max(1, min(n, len(self._anchors)))
        need = 2 * n + 3
        best, best_mismatch = None, -1
        for _ in range(_SCRAMBLE_TRIES):
            grid = [row[:] for row in target]
            for ax, ay in rng.sample(self._anchors, n):
                for _turn in range(3):                    # 3 clockwise turns == 1 counter-cw
                    rotate_cw(grid, ax, ay)
            mismatch = sum(1 for x, y in self._free_cells()
                           if grid[y][x] != target[y][x])
            if mismatch >= need:
                return grid
            if mismatch > best_mismatch:
                best, best_mismatch = grid, mismatch
        return best

    def _free_cells(self):
        return ((x, y) for y in range(GH) for x in range(GW) if (x, y) not in self._walls)

    # ── level setup ─────────────────────────────────────────────────────────
    def on_set_level(self, level: Level) -> None:
        self._walls = frozenset(
            (s.x, s.y) for s in level.get_sprites() if "wall" in s.tags
        )
        self._anchors = legal_anchors(self._walls)
        rng = self._board_rng(self.level_index)
        self._target = self._draw_target(rng)
        self._g = self._scramble(self._target, rng, int(level.get_data("scramble") or 1))
        self._steps_left = int(level.get_data("max_steps") or 40)
        self._ui.update(self._steps_left, self._target, self._palette, self._walls)
        self._paint()

    def _paint(self) -> None:
        for s in list(self.current_level.get_sprites_by_tag("qtile")):
            self.current_level.remove_sprite(s)
        for x, y in self._free_cells():
            self.current_level.add_sprite(
                cell_sprite(self._g[y][x], self._palette).set_position(x, y),
            )

    def _win(self) -> bool:
        return all(self._g[y][x] == self._target[y][x] for x, y in self._free_cells())

    def _grid_to_frame_pixel(self, gx: int, gy: int) -> tuple[int, int]:
        cam = self.camera
        cw, ch = cam.width, cam.height
        scale = min(int(64 / cw), int(64 / ch))
        x_pad = int((64 - (cw * scale)) / 2)
        y_pad = int((64 - (ch * scale)) / 2)
        px = gx * scale + scale // 2 + x_pad
        py = gy * scale + scale // 2 + y_pad
        return px, py

    def step(self) -> None:
        if self.action.id != GameAction.ACTION6:
            self.complete_action()
            return

        if self._steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        # The frame is presented rotated by k*90 degrees, so a screen click has to be mapped
        # back into game space before it can be read as a grid cell.
        px, py = self.screen_click_to_game(
            self.action.data.get("x", 0), self.action.data.get("y", 0)
        )
        coords = self.camera.display_to_grid(px, py)
        if not coords:
            self._ui.flash_reject()
            self.complete_action()
            return
        gx, gy = coords
        # The marker is placed in GAME space; the rotation display turns the composited frame,
        # so it lands on the cell the player actually clicked.
        self._ui.set_click(*self._grid_to_frame_pixel(gx, gy))
        if gx + 2 >= GW or gy + 2 >= GH:
            self._ui.flash_reject()
            self.complete_action()
            return
        if any((gx + dx, gy + dy) in self._walls for dx in range(3) for dy in range(3)):
            self._ui.flash_reject()
            self.complete_action()
            return

        rotate_cw(self._g, gx, gy)

        self._paint()
        self._steps_left -= 1
        self._ui.update(self._steps_left, self._target)
        if self._win():
            self.next_level()

        self.complete_action()
