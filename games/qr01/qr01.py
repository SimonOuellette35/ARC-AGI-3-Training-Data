"""Quad Twist: ACTION6 rotates the 2x2 block of tiles anchored at the clicked cell
clockwise. Match the target pattern shown in the top-right reference panel.

Mechanic (unchanged from the original): a click at grid cell ``(gx, gy)`` cycles the
four tiles ``(gx,gy) -> (gx+1,gy) -> (gx+1,gy+1) -> (gx,gy+1) -> (gx,gy)``. Tile
*colours* are only ever permuted, never created -- so a level is solvable iff the
target is a rearrangement of the board. A block that runs off the grid, or touches a
wall, is a no-op and costs nothing.

Per-seed augmentation (`Qr01(seed=S)` is a pure function of ``(S, level)``)
------------------------------------------------------------------------
* **Rotation** -- inherited from `utils.arc_game.AugmentedGame`: every level is
  displayed at ``random_rotation_k(seed, level_index)`` and clicks come back through
  ``screen_click_to_game``.
* **Colours** -- four tile colours drawn per EPISODE (level-independent, so all
  levels of one seed share a palette); a level uses the first ``colors`` of them.
* **Layout** -- each level is a set of rectangular *panels* of free tiles carved out
  of the 8x8 board; everything else is wall. Panel sizes and count come from
  `LEVELS_SPEC`, their positions are drawn per (seed, level) with a >=1 cell gap
  between panels, so no 2x2 block ever spans two panels and each panel is an
  independent sub-puzzle.
* **Board + target** -- a balanced random colouring per panel, and a target produced
  by applying ``scramble`` *effective* rotations to it. The target is therefore
  reachable by construction, and no level can be unsolvable.

Why this replaced the original five levels: the old level 3 asked for an all-``1``
board from an all-``0`` one and the old level 4 left cell (4,4) isolated between two
wall rows with the wrong colour -- both impossible, since rotations only permute
what is already on the board. The old levels were also a single 64-cell 4-colour
shuffle, whose state space no exact solver can touch; the panel layout keeps the
same mechanic but bounds each sub-puzzle's orbit to a few tens of thousands of
states, which is what makes optimal play (and therefore optimal-action-set training
targets, and recovery from ANY perturbed state) computable -- see
``solvers/generate_qr01_training.py``.
"""

from __future__ import annotations

import random as _random_module

from arcengine import (
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)

from utils.arc_game import AugmentedGame, clear_dynamic_sprites

GW = GH = 8
BACKGROUND_COLOR = 5
PADDING_COLOR = 4
WALL_C = 3
STEP_C = 10

# Tile colours are drawn from this pool: no wall (3), background (5), padding (4),
# budget-bar (10) or 0, so nothing on the board is confusable with the furniture.
COLOR_POOL = [1, 2, 6, 7, 8, 9, 11, 12, 13, 14, 15]

# ── HUD geometry (frame pixels) ─────────────────────────────────────────────
# The board is 8x8 and the camera resizes to it, so one cell is exactly 8x8 px.
# The target reference is drawn at 2 px per cell in the top-right corner with the
# step-budget bar directly under it; `HUD_CELLS` is the block of BOARD cells those
# overlays cover, which panel placement then avoids -- so the HUD never hides a tile.
HUD_PX = 2
HUD_X0 = 64 - HUD_PX * GW          # 48
HUD_Y0 = 0
BAR_Y = HUD_PX * GH + 1            # 17 -- two rows, still inside board row 2
HUD_CELLS = frozenset((x, y) for x in range(6, GW) for y in range(3))


class Qr01UI(RenderableUserDisplay):
    """Target reference (same colours as the playfield, walls in grey) plus the
    remaining-step bar."""

    def __init__(self) -> None:
        self._target: list[list[int | None]] = [[None] * GW for _ in range(GH)]
        self._steps = 1
        self._max = 1

    def update(self, steps: int, max_steps: int | None = None,
               target: list[list[int | None]] | None = None) -> None:
        self._steps = steps
        if max_steps:
            self._max = max_steps
        if target is not None:
            self._target = [row[:] for row in target]

    def render_interface(self, frame):
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return frame
        for y in range(GH):
            for x in range(GW):
                c = self._target[y][x]
                # Non-panel cells use the padding colour, which appears nowhere
                # else in the frame, so the reference reads as its own inset panel
                # rather than blending into the grey wall field behind it.
                frame[HUD_Y0 + HUD_PX * y: HUD_Y0 + HUD_PX * (y + 1),
                      HUD_X0 + HUD_PX * x: HUD_X0 + HUD_PX * (x + 1)] = (
                    PADDING_COLOR if c is None else c)
        width = HUD_PX * GW
        filled = max(0, min(width, round(width * self._steps / max(1, self._max))))
        frame[BAR_Y:BAR_Y + 2, HUD_X0:HUD_X0 + width] = BACKGROUND_COLOR
        frame[BAR_Y:BAR_Y + 2, HUD_X0:HUD_X0 + filled] = STEP_C
        return frame


def _tile_sprite(color: int) -> Sprite:
    return Sprite(pixels=[[color]], name="qt", visible=True, collidable=False,
                  tags=["qtile"])


def _wall_sprite() -> Sprite:
    return Sprite(pixels=[[WALL_C]], name="wall", visible=True, collidable=True,
                  tags=["wall"])


# ── Level specs ─────────────────────────────────────────────────────────────
# ``panels``   panel sizes (w, h); each is an independent sub-puzzle
# ``colors``   how many of the episode's four tile colours this level uses
# ``scramble`` effective rotations applied to each panel to build its target
# ``home``     fallback panel positions if the random placement search fails
#
# ``scramble`` is an upper bound on a panel's true distance, not the distance
# itself: rotations partly undo one another and repeated colours make many of them
# equivalent. The values below were calibrated against the solver's exact distance
# oracle so that the mean optimal click count rises monotonically across the five
# levels -- roughly 3.0, 3.9, 5.8, 7.7, 8.9. (The two-colour 4x4 saturates near 6
# whatever you scramble it by; that is its orbit's diameter, not a mis-tuning.)
LEVELS_SPEC = [
    {"panels": [(3, 3)], "colors": 3, "scramble": 3, "difficulty": 1,
     "home": [(2, 3)]},
    {"panels": [(4, 3)], "colors": 3, "scramble": 4, "difficulty": 2,
     "home": [(1, 3)]},
    {"panels": [(4, 4)], "colors": 2, "scramble": 10, "difficulty": 3,
     "home": [(1, 3)]},
    {"panels": [(3, 3), (3, 3)], "colors": 4, "scramble": 4, "difficulty": 4,
     "home": [(0, 0), (0, 4)]},
    {"panels": [(3, 3), (3, 3), (3, 3)], "colors": 4, "scramble": 3,
     "difficulty": 5, "home": [(0, 0), (0, 4), (4, 4)]},
]


def _steps_budget(spec: dict) -> int:
    """Room for ~8 clicks per scrambled move plus slack -- generous, but finite."""
    return 12 + 8 * spec["scramble"] * len(spec["panels"])


levels = [
    Level(sprites=[], grid_size=(GW, GH),
          data={"difficulty": s["difficulty"], "max_steps": _steps_budget(s)})
    for s in LEVELS_SPEC
]


# ── per-seed board construction ─────────────────────────────────────────────
def _cells(box) -> list[tuple[int, int]]:
    x0, y0, w, h = box
    return [(x0 + dx, y0 + dy) for dy in range(h) for dx in range(w)]


def _touches(a, b) -> bool:
    """Do boxes ``a`` and ``b`` come within one cell of each other? Panels must not,
    or a 2x2 block could straddle both and couple two sub-puzzles."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw < bx or bx + bw < ax or ay + ah < by or by + bh < ay)


def _place_panels(sizes, rng) -> list[tuple[int, int, int, int]] | None:
    """Random positions for ``sizes`` inside the board: pairwise >=1 cell apart and
    clear of the HUD overlay. ``None`` if the search fails (caller uses ``home``)."""
    for _ in range(200):
        placed: list[tuple[int, int, int, int]] = []
        for (w, h) in sizes:
            for _attempt in range(120):
                box = (rng.randrange(GW - w + 1), rng.randrange(GH - h + 1), w, h)
                if any(c in HUD_CELLS for c in _cells(box)):
                    continue
                if any(_touches(box, q) for q in placed):
                    continue
                placed.append(box)
                break
            else:
                break
        if len(placed) == len(sizes):
            return placed
    return None


def _balanced_fill(n: int, ncolors: int, rng) -> list[int]:
    """``n`` colour indices with counts as equal as possible, shuffled -- so every
    colour of the level's palette is actually on the board."""
    vals = [i % ncolors for i in range(n)]
    rng.shuffle(vals)
    return vals


def _rotate_cw(grid, x: int, y: int) -> None:
    a, b = grid[y][x], grid[y][x + 1]
    c, d = grid[y + 1][x], grid[y + 1][x + 1]
    grid[y][x], grid[y][x + 1] = c, a
    grid[y + 1][x], grid[y + 1][x + 1] = d, b


def _scramble(grid, box, moves: int, rng) -> None:
    """Apply ``moves`` rotations that each actually change the colouring (a rotation
    of four equal tiles is invisible and would make the target easier than declared)."""
    x0, y0, w, h = box
    anchors = [(x0 + ax, y0 + ay) for ay in range(h - 1) for ax in range(w - 1)]
    done = 0
    for _ in range(200 * max(1, moves)):
        if done >= moves:
            break
        ax, ay = rng.choice(anchors)
        if len({grid[ay][ax], grid[ay][ax + 1],
                grid[ay + 1][ax], grid[ay + 1][ax + 1]}) == 1:
            continue                      # invisible rotation -- costs nothing
        _rotate_cw(grid, ax, ay)
        done += 1


class Qr01(AugmentedGame):
    def __init__(self, seed: int | None = None) -> None:
        # FIRST: the camera's interface list holds the rotation display, and
        # super().__init__ -> set_level(0) -> on_set_level already needs the board seed.
        self._init_augmentation(seed)
        self._board_seed = (seed if seed is not None
                            else _random_module.randrange(1 << 30))
        # Level-INDEPENDENT palette: every level of one episode shares its colours.
        self._palette = _random_module.Random(
            f"qr01-colors:{self._board_seed}").sample(COLOR_POOL, 4)
        self._ui = Qr01UI()
        self._panels: list[tuple[int, int, int, int]] = []
        self._free: frozenset[tuple[int, int]] = frozenset()
        self._g = [[0] * GW for _ in range(GH)]
        self._target = [[0] * GW for _ in range(GH)]
        self._steps_left = 0
        self._max_steps = 1
        super().__init__(
            "qr01",
            levels,
            Camera(0, 0, GW, GH, BACKGROUND_COLOR, PADDING_COLOR,
                   [self._ui]),
            False,
            1,
            [1, 2, 3, 4, 6],
        )

    # ── level construction ──────────────────────────────────────────────────
    def on_set_level(self, level: Level) -> None:
        # Levels ship empty and every sprite is built here, so drop whatever a
        # previous seating of this same Level object left behind.
        clear_dynamic_sprites(level, tags=["wall", "qtile"])
        idx = self.level_index
        spec = LEVELS_SPEC[idx]
        rng = _random_module.Random(f"qr01:{self._board_seed}:{idx}")

        sizes = [(w, h) if (w == h or rng.random() < 0.5) else (h, w)
                 for (w, h) in spec["panels"]]
        boxes = _place_panels(sizes, rng)
        if boxes is None:
            boxes = [(x, y, w, h)
                     for (x, y), (w, h) in zip(spec["home"], spec["panels"])]
        self._panels = boxes

        ncolors = spec["colors"]
        self._g = [[0] * GW for _ in range(GH)]
        self._target = [[0] * GW for _ in range(GH)]
        free = []
        for box in boxes:
            cells = _cells(box)
            free.extend(cells)
            for (x, y), v in zip(cells, _balanced_fill(len(cells), ncolors, rng)):
                self._g[y][x] = v
            for _ in range(60):            # a target equal to the board is no puzzle
                cand = [row[:] for row in self._g]
                _scramble(cand, box, spec["scramble"], rng)
                if any(cand[y][x] != self._g[y][x] for (x, y) in cells):
                    break
            for (x, y) in cells:
                self._target[y][x] = cand[y][x]
        self._free = frozenset(free)

        for x in range(GW):
            for y in range(GH):
                if (x, y) not in self._free:
                    level.add_sprite(_wall_sprite().set_position(x, y))

        self._max_steps = int(level.get_data("max_steps") or 60)
        self._steps_left = self._max_steps
        self._paint()
        self._refresh_ui()

    def _paint(self) -> None:
        for s in list(self.current_level.get_sprites_by_tag("qtile")):
            self.current_level.remove_sprite(s)
        for (x, y) in self._free:
            self.current_level.add_sprite(
                _tile_sprite(self._palette[self._g[y][x]]).set_position(x, y))

    def _refresh_ui(self) -> None:
        shown: list[list[int | None]] = [[None] * GW for _ in range(GH)]
        for (x, y) in self._free:
            shown[y][x] = self._palette[self._target[y][x]]
        self._ui.update(self._steps_left, self._max_steps, shown)

    # ── play ────────────────────────────────────────────────────────────────
    def _win(self) -> bool:
        return all(self._g[y][x] == self._target[y][x] for (x, y) in self._free)

    def step(self) -> None:
        if self.action.id != GameAction.ACTION6:
            self.complete_action()
            return
        if self._steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        sx, sy = self.screen_click_to_game(self.action.data.get("x", 0),
                                           self.action.data.get("y", 0))
        coords = self.camera.display_to_grid(sx, sy)
        if not coords:
            self.complete_action()
            return
        gx, gy = coords
        block = [(gx, gy), (gx + 1, gy), (gx, gy + 1), (gx + 1, gy + 1)]
        if any(c not in self._free for c in block):
            self.complete_action()          # off-grid or wall -- free no-op
            return

        _rotate_cw(self._g, gx, gy)
        self._paint()
        self._steps_left -= 1
        self._refresh_ui()
        if self._win():
            self.next_level()
        self.complete_action()
