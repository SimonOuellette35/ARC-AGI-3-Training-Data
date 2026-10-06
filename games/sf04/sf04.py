"""sf04: ACTION5 rotates 3-cell L stencil; ACTION6 paints stencil anchor on grid."""

from __future__ import annotations

import random

from utils.arc_game import AugmentedGame
from arcengine import (
    ARCBaseGame,
    Camera,
    GameAction,
    GameState,
    Level,
    RenderableUserDisplay,
    Sprite,
)

from utils.color_remap import ColorRemapDisplay, random_color_lut

BG, PAD = 5, 4
GW, GH = 16, 16
CAM = 16
WALL_C, HINT_C, PNT_C = 3, 2, 11


def rot(dx: int, dy: int, q: int) -> tuple[int, int]:
    for _ in range(q % 4):
        dx, dy = -dy, dx
    return dx, dy


BASE_STENCIL = [(0, 0), (1, 0), (0, 1)]


def _rp(frame, h, w, x, y, c):
    if 0 <= x < w and 0 <= y < h:
        frame[y, x] = c


def _r_dots(frame, h, w, li, n, y0=0):
    for i in range(min(n, 14)):
        cx = 1 + i * 2
        if cx >= w:
            break
        c = 14 if i < li else (11 if i == li else 3)
        _rp(frame, h, w, cx, y0, c)


def _r_bar(frame, h, w, game_over, win):
    if not (game_over or win):
        return
    r = h - 3
    if r < 0:
        return
    c = 14 if win else 8
    for x in range(min(w, 16)):
        _rp(frame, h, w, x, r, c)


class Sf04UI(RenderableUserDisplay):
    def __init__(self, rotq: int, ok: int, tot: int, li: int = 0, nl: int = 7) -> None:
        self._q, self._ok, self._tot = rotq, ok, tot
        self._li, self._nl = li, nl
        self._state = None
        self._click_pos: tuple[int, int] | None = None
        self._click_frames = 0
        self._click_ok = False

    def set_click_feedback(self, px: int, py: int, ok: bool) -> None:
        self._click_pos = (px, py)
        self._click_frames = 8
        self._click_ok = ok

    def update(
        self,
        rotq: int,
        ok: int,
        tot: int,
        *,
        level_index: int | None = None,
        num_levels: int | None = None,
        state=None,
    ) -> None:
        self._q, self._ok, self._tot = rotq, ok, tot
        if level_index is not None:
            self._li = level_index
        if num_levels is not None:
            self._nl = num_levels
        if state is not None:
            self._state = state

    def render_interface(self, frame):
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return frame
        h, w = frame.shape
        _r_dots(frame, h, w, self._li, self._nl, 0)
        frame[1, 1] = 10 + (self._q % 4)
        for i in range(min(self._tot, 14)):
            frame[2, 1 + i] = 14 if i < self._ok else 8
        go = self._state == GameState.GAME_OVER
        win = self._state == GameState.WIN
        _r_bar(frame, h, w, go, win)
        if self._click_pos and self._click_frames > 0:
            cx, cy = self._click_pos
            hit = 11 if self._click_ok else 8
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


W = Sprite(
    pixels=[[WALL_C]],
    name="w",
    visible=True,
    collidable=True,
    tags=["wall"],
)
H = Sprite(
    pixels=[[HINT_C]],
    name="h",
    visible=True,
    collidable=False,
    tags=["hint"],
)
P = Sprite(
    pixels=[[PNT_C]],
    name="p",
    visible=True,
    collidable=False,
    tags=["paint"],
)


def mk(
    hints: list[tuple[int, int]],
    walls: list[tuple[int, int]],
    max_steps: int,
    d: int,
) -> Level:
    sl = [W.clone().set_position(x, y) for x, y in walls]
    sl += [H.clone().set_position(x, y) for x, y in hints]
    return Level(
        sprites=sl,
        grid_size=(GW, GH),
        data={
            "goal": [list(p) for p in hints],
            "max_steps": max_steps,
            "difficulty": d,
        },
    )


levels = [
    mk([(5, 5), (6, 5), (5, 6)], [], 40, 1),
    mk([(4, 4), (5, 4), (4, 5), (7, 7)], [(6, 6)], 50, 2),
    mk([(3, 3), (4, 3), (3, 4), (8, 8), (9, 8), (8, 9)], [(5, 5), (5, 6)], 60, 3),
    mk([(2, 2), (3, 2), (2, 3), (10, 10), (11, 10), (10, 11)], [(6, 6), (7, 6)], 70, 4),
    mk([(4, 7), (5, 7), (4, 8), (9, 4), (10, 4), (9, 5)], [], 80, 5),
    mk([(6, 6), (7, 6), (6, 7), (6, 9), (7, 9), (6, 10)], [(8, 8)], 90, 6),
    mk([(2, 8), (3, 8), (2, 9), (12, 3), (13, 3), (12, 4), (7, 7)], [(5, 5)], 100, 7),
]


# ── Per-seed augmentation ────────────────────────────────────────────────────
#
# Shape of each shipped level, preserved by the randomiser: the hints are a
# union of `n_l` L-stencil footprints plus `n_extra` stray singletons, with
# `n_walls` walls. Only positions/orientations/colours vary, never the counts,
# the step budget or the difficulty ramp.
#
#                 (n_l, n_extra, n_walls, max_steps, difficulty)
_LEVEL_SPECS = [
    (1, 0, 0, 40, 1),
    (1, 1, 1, 50, 2),
    (2, 0, 2, 60, 3),
    (2, 0, 2, 70, 4),
    (2, 0, 0, 80, 5),
    (2, 0, 1, 90, 6),
    (2, 1, 1, 100, 7),
]

_ALL_CELLS = [(x, y) for x in range(GW) for y in range(GH)]


def stencil_cells(ax: int, ay: int, q: int) -> list[tuple[int, int]]:
    """The 3 cells an ACTION6 click at anchor (ax, ay) paints at rotation q."""
    return [
        (ax + rot(dx, dy, q)[0], ay + rot(dx, dy, q)[1]) for dx, dy in BASE_STENCIL
    ]


def placements_covering(
    cell: tuple[int, int], walls: set[tuple[int, int]]
) -> list[tuple[tuple[int, int], int]]:
    """Every (anchor, q) whose stamp paints `cell` and that `step` would accept.

    Mirrors the legality test in :meth:`Sf04.step`: all 3 stencil cells must be
    in bounds and none may be a wall.
    """
    cx, cy = cell
    out: list[tuple[tuple[int, int], int]] = []
    for q in range(4):
        for dx, dy in BASE_STENCIL:
            rx, ry = rot(dx, dy, q)
            anchor = (cx - rx, cy - ry)
            cells = stencil_cells(anchor[0], anchor[1], q)
            if any(not (0 <= x < GW and 0 <= y < GH) for x, y in cells):
                continue
            if any(c in walls for c in cells):
                continue
            out.append((anchor, q))
    return out


def is_solvable(goal: set[tuple[int, int]], walls: set[tuple[int, int]]) -> bool:
    """True iff every hint cell can be painted.

    Paint is additive and overpainting is free, so the win test ``goal <=
    painted`` is reachable exactly when each hint cell has at least one legal
    stamp covering it. The step budget is not a constraint in practice: a plan
    that visits each needed rotation in ascending order costs at most 3 turns
    plus one click per hint cell (<= 10), far under the 40-100 budgets.
    """
    return all(placements_covering(g, walls) for g in goal)


def _halo(cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """`cells` plus their 8-neighbours, so distinct hint shapes never touch."""
    return {
        (x + dx, y + dy)
        for x, y in cells
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    }


def _build_level(
    rng: random.Random,
    n_l: int,
    n_extra: int,
    n_walls: int,
    max_steps: int,
    difficulty: int,
    attempts: int = 200,
) -> Level | None:
    """Sample one randomised level, or None if `attempts` all failed.

    L footprints take a random anchor *and a random rotation* (the shipped
    levels are all q=0), kept mutually non-adjacent so each reads as a separate
    L. Walls are drawn off the hint cells -- which alone keeps every L coverable
    by its own generating stamp -- and the whole board is then put through
    :func:`is_solvable`, which is the ground truth and also covers the
    singletons.
    """
    for _ in range(attempts):
        goal: list[tuple[int, int]] = []
        taken: set[tuple[int, int]] = set()

        def place(cells: list[tuple[int, int]]) -> None:
            goal.extend(cells)
            taken.update(_halo(cells))

        shapes_ok = True
        for _ in range(n_l):
            for _ in range(200):
                q = rng.randrange(4)
                cells = stencil_cells(rng.randrange(GW), rng.randrange(GH), q)
                if any(not (0 <= x < GW and 0 <= y < GH) for x, y in cells):
                    continue
                if any(c in taken for c in cells):
                    continue
                place(cells)
                break
            else:
                shapes_ok = False
                break
        if not shapes_ok:
            continue

        for _ in range(n_extra):
            free = [c for c in _ALL_CELLS if c not in taken]
            if not free:
                shapes_ok = False
                break
            place([rng.choice(free)])
        if not shapes_ok:
            continue

        goal_set = set(goal)
        walls = set(rng.sample([c for c in _ALL_CELLS if c not in goal_set], n_walls))
        if is_solvable(goal_set, walls):
            return mk(sorted(goal_set), sorted(walls), max_steps, difficulty)
    return None


def build_randomized_levels(seed: int) -> list[Level]:
    """The 7 levels for `seed`, each a pure function of (seed, level index).

    A level whose rejection sampling is exhausted falls back to its shipped
    layout, which is solvable by construction.
    """
    out: list[Level] = []
    for i, (n_l, n_extra, n_walls, max_steps, difficulty) in enumerate(_LEVEL_SPECS):
        lvl = _build_level(
            random.Random(f"sf04:{seed}:{i}"),
            n_l,
            n_extra,
            n_walls,
            max_steps,
            difficulty,
        )
        out.append(lvl if lvl is not None else levels[i].clone())
    return out


class Sf04(AugmentedGame):
    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: the base draws this level's rotation
        """With `seed` given the board is a deterministic per-seed augmentation
        of the shipped one; with `seed` None it *is* the shipped one, unchanged.
        """
        game_levels = levels if seed is None else build_randomized_levels(seed)
        self._ui = Sf04UI(0, 0, 0, 0, len(game_levels))
        # Colour permutation, applied to the final composited frame after the
        # UI has drawn -- so sprites, background and HUD recolour together. The
        # seed is level-INDEPENDENT so one episode shares one colour scheme.
        self._color_display = ColorRemapDisplay()
        self._color_display.set_lut(
            None if seed is None else random_color_lut(f"sf04-colors:{seed}")
        )
        super().__init__(
            "sf04",
            game_levels,
            Camera(0, 0, CAM, CAM, BG, PAD, [self._ui, self._color_display]),
            False,
            1,
            [5, 6],
        )

    def on_set_level(self, level: Level) -> None:
        self._goal = {tuple(p) for p in level.get_data("goal")}
        self._max_steps = int(level.get_data("max_steps") or 50)
        self._steps = 0
        self._rotq = 0
        self._sync()

    def _cells(self, ax: int, ay: int) -> list[tuple[int, int]]:
        return [
            (ax + rot(dx, dy, self._rotq)[0], ay + rot(dx, dy, self._rotq)[1])
            for dx, dy in BASE_STENCIL
        ]

    def _painted(self) -> set[tuple[int, int]]:
        return {(s.x, s.y) for s in self.current_level.get_sprites_by_tag("paint")}

    def _sync(self) -> None:
        painted = self._painted()
        ok = len(painted & self._goal)
        self._ui.update(
            self._rotq,
            ok,
            len(self._goal),
            level_index=self.level_index,
            num_levels=len(self._levels),
            state=self._state,
        )

    def step(self) -> None:
        if self.action.id == GameAction.ACTION5:
            self._rotq = (self._rotq + 1) % 4
            self._steps += 1
            self._sync()
            if self._steps >= self._max_steps:
                self.lose()
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION6:
            px, py = int(self.action.data.get("x", 0)), int(
                self.action.data.get("y", 0)
            )
            hit = self.camera.display_to_grid(px, py)
            click_ok = False
            if hit:
                ax, ay = int(hit[0]), int(hit[1])
                bad = False
                for cx, cy in self._cells(ax, ay):
                    if not (0 <= cx < GW and 0 <= cy < GH):
                        bad = True
                        break
                    sp = self.current_level.get_sprite_at(cx, cy, ignore_collidable=True)
                    if sp and "wall" in sp.tags:
                        bad = True
                        break
                if not bad:
                    click_ok = True
                    for cx, cy in self._cells(ax, ay):
                        ex = self.current_level.get_sprite_at(
                            cx, cy, ignore_collidable=True
                        )
                        if ex and "paint" in ex.tags:
                            continue
                        self.current_level.add_sprite(P.clone().set_position(cx, cy))
            self._ui.set_click_feedback(px, py, click_ok)
            self._steps += 1
            self._sync()
            if self._goal <= self._painted():
                self.next_level()
            elif self._steps >= self._max_steps:
                self.lose()
            self.complete_action()
            return

        self.complete_action()
