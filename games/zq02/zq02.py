"""Dual phase hazards: two independent blinking hazard sets with different period/offset.

Reach the goal cell. Two hazard SETS blink on and off on separate periods and offsets;
a set that is ON blocks the move (the action is consumed, the player stays, and a
corner block flashes), a set that is OFF is free floor. Nothing is lethal and nothing
is irreversible -- this is purely a TIMING maze, and the only resource is parity: every
action costs exactly one tick, a blocked or off-board move included, so pushing into
the board edge is a free WAIT.

The two 2x2 patches in the bottom-right corner are the phase indicators, one per set:
each shows its own set's colour while that set blocks and a shared idle colour while it
does not. Neither shows how much of the current phase is LEFT -- that has to be inferred
from watching the blink, which is the game.

PER-SEED AUGMENTATION (the repo convention -- see utils/arc_game.py and the per-seed
augmentation notes). ``Zq02(seed=s)`` makes every level a pure function of
``(s, level_index)``:

  * DISPLAY ROTATION, owned by ``AugmentedGame``: the frame is rendered rotated by a
    per-(seed, level) k and ``step`` maps the pressed SCREEN direction back to game
    space via ``screen_action_to_game``. No rotation code lives here.
  * LAYOUT: board size, BOTH periods and offsets, both hazard patterns and the
    player/goal cells are re-drawn per (seed, level), keeping each level's stock
    character -- see ``_LEVEL_PARAMS`` (scattered posts, two crossing bars, two
    crossing combs, a double barrier, fast-flickering posts).
  * COLOURS: one EPISODE-scoped scheme (level-independent seed, cached), so every level
    of one episode looks the same. Padding (4) stays fixed.

SOLVABILITY CERTIFICATE. A drawn board is accepted only if
``utils.zone_timer.PhasedGridField`` -- the exact ``(cell, phase)`` distance-to-win
field, shared with zq01 and with this game's training generator -- says it is winnable
from the start state, that the optimum sits inside the level's step band, AND that the
optimum is strictly longer than the plain Manhattan walk, i.e. the two timers actually
cost the player something. Boards where the hazards are decoration are rejected rather
than recorded. Since the generator's expert descends that same field, every board this
game can draw is one the generator can solve optimally.
"""

import math as _math
import random as _random_module

from arcengine import (
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)

from utils.arc_game import AugmentedGame, clear_dynamic_sprites
from utils.zone_timer import PhasedGridField

BACKGROUND_COLOR = 5
PADDING_COLOR = 4
PLAYER_COLOR = 9
TARGET_COLOR = 11
HAZARD_A_COLOR = 8
HAZARD_B_COLOR = 12
IDLE_COLOR = 10
FLASH_COLOR = 8

#: Colours the per-episode scheme draws from, mutually distinct so nothing hides
#: against anything else. Excludes 0 and the fixed padding colour (4).
_COLOR_POOL = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# Render / collision layers. ``Camera.render`` draws LOW layers first and
# ``get_sprite_at(..., ignore_collidable=True)`` returns the HIGHEST layer at a cell,
# so the player must sit ABOVE the hazards: a player standing on a hazard cell when it
# blinks on has to stay visible (with every sprite on layer 0 the engine falls back to
# insertion order and the hazards, added last, would paint the player out).
_LAYER_FLOOR = 0
_LAYER_HAZARD = 1
_LAYER_PLAYER = 2


class Zq02UI(RenderableUserDisplay):
    """Two phase indicators (bottom-right, one per hazard set) plus a blocked-move
    flash (bottom-left).

    ``render_interface`` is PURE -- it draws and mutates nothing. The flash countdown
    is owned by ``Zq02.step`` (via `tick_flash`) instead, because a display that
    decremented it while rendering would make the frame depend on how many times it
    happened to be rendered: any harness that renders outside the action loop (the
    training recorder does, to capture a level's opening frame) would silently eat
    frames of an animation that belongs to the recording.
    """

    def __init__(self, a: bool, b: bool, colors: dict | None = None) -> None:
        self._a = a
        self._b = b
        self._hazard_flash = 0
        self._colors = colors or {"hazard_a": HAZARD_A_COLOR,
                                  "hazard_b": HAZARD_B_COLOR,
                                  "idle": IDLE_COLOR, "flash": FLASH_COLOR}

    def update(self, a: bool, b: bool) -> None:
        self._a = a
        self._b = b

    def set_colors(self, colors: dict) -> None:
        self._colors = colors

    def flash_hazard_block(self, frames: int = 6) -> None:
        self._hazard_flash = frames

    def tick_flash(self) -> None:
        """Age the blocked-move flash by one ACTION (called from ``Zq02.step``)."""
        if self._hazard_flash > 0:
            self._hazard_flash -= 1

    def clear_flash(self) -> None:
        self._hazard_flash = 0

    def render_interface(self, frame):
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return frame
        h, w = frame.shape
        col = self._colors
        c0 = col["hazard_a"] if self._a else col["idle"]
        c1 = col["hazard_b"] if self._b else col["idle"]
        for dy in range(2):
            for dx in range(2):
                frame[h - 3 + dy, w - 5 + dx] = c0
                frame[h - 3 + dy, w - 3 + dx] = c1
        if self._hazard_flash > 0:
            for dy in range(2):
                for dx in range(2):
                    frame[h - 5 + dy, 1 + dx] = col["flash"]
        return frame


sprites = {
    "player": Sprite(
        pixels=[[PLAYER_COLOR]],
        name="player",
        visible=True,
        collidable=True,
        tags=["player"],
        layer=_LAYER_PLAYER,
    ),
    "target": Sprite(
        pixels=[[TARGET_COLOR]],
        name="target",
        visible=True,
        collidable=False,
        tags=["target"],
        layer=_LAYER_FLOOR,
    ),
    "hazard_a": Sprite(
        pixels=[[HAZARD_A_COLOR]],
        name="hazard_a",
        visible=True,
        collidable=True,
        tags=["zq2_hazard_a"],
        layer=_LAYER_HAZARD,
    ),
    "hazard_b": Sprite(
        pixels=[[HAZARD_B_COLOR]],
        name="hazard_b",
        visible=True,
        collidable=True,
        tags=["zq2_hazard_b"],
        layer=_LAYER_HAZARD,
    ),
}


def mk(grid_size, difficulty: int) -> Level:
    """A level TEMPLATE. Every sprite is built per-seating in ``on_set_level``, so a
    template carries nothing but its board size and difficulty (``set_level`` pins the
    drawn size before the engine reads it)."""
    return Level(sprites=[], grid_size=grid_size, data={"difficulty": difficulty})


levels = [mk((8, 8), 1), mk((8, 8), 2), mk((8, 8), 3), mk((10, 8), 4), mk((8, 8), 5)]

# Per-level augmentation budget, index-aligned with ``levels`` and ``_FALLBACKS``:
#   style    -- the hazard pattern pair (see ``_sample_hazards``)
#   size     -- inclusive range each board dimension is drawn from
#   periods  -- inclusive range both blink periods are drawn from (they must differ,
#               or the two sets would be one set with two colours)
#   steps    -- inclusive band the optimal solution length must land in
_LEVEL_PARAMS = [
    # L1 -- scattered posts on open floor
    {"style": "posts", "size": (8, 9), "periods": (3, 6), "posts": (2, 3),
     "steps": (8, 28)},
    # L2 -- two crossing bars, one on a column and one on a row
    {"style": "bars", "size": (8, 9), "periods": (3, 6), "run": (2, 4),
     "steps": (8, 28)},
    # L3 -- two crossing combs (every other cell of a row and of a column)
    {"style": "combs", "size": (8, 10), "periods": (4, 7), "steps": (9, 34)},
    # L4 -- a double barrier: two parallel lines, the second with one doorway
    {"style": "walls", "size": (9, 10), "periods": (3, 6), "steps": (9, 40)},
    # L5 -- fast flicker: more posts, short periods, so parity changes constantly
    {"style": "posts", "size": (8, 10), "periods": (2, 4), "posts": (2, 4),
     "steps": (9, 40)},
]

#: The stock (pre-augmentation) layouts, used verbatim when sampling fails so a level is
#: always well-formed. ``(grid, pa, pb, oa, ob, cells_a, cells_b, player, target)``.
_FALLBACKS = [
    ((8, 8), 4, 6, 0, 2, [(3, 3), (4, 3)], [(5, 2), (5, 4)], (0, 3), (7, 3)),
    ((8, 8), 3, 5, 0, 1, [(2, y) for y in range(6, 8)],
     [(x, 2) for x in range(2, 6)], (1, 1), (6, 6)),
    ((8, 8), 5, 7, 1, 3, [(x, 4) for x in range(8) if x % 2 == 0],
     [(4, y) for y in range(8) if y % 2 == 1], (0, 0), (7, 7)),
    ((10, 8), 4, 5, 0, 2, [(3, y) for y in range(8)],
     [(6, y) for y in range(8) if y != 4], (0, 4), (9, 4)),
    ((8, 8), 2, 3, 0, 1, [(2, 2), (5, 5)], [(5, 2), (2, 5)], (0, 7), (7, 0)),
]

_LAYOUT_ATTEMPTS = 60


def blink_schedule(pa: int, pb: int, oa: int, ob: int) -> tuple:
    """``sched[t]`` = the ``(on_a, on_b)`` configuration at tick ``t``, for one full
    cycle. Mirrors ``Zq02.step`` exactly: both sets start OFF at tick 0 and set A
    toggles on every tick with ``(t + oa) % pa == 0`` (B likewise).

    The cycle length is ``lcm(2*pa, 2*pb)``: a set's own pattern repeats every ``2*p``
    ticks (one toggle per ``p`` ticks, so two per ``2*p`` -- an even number, which
    returns it to OFF)."""
    period = _math.lcm(2 * pa, 2 * pb)
    on_a = on_b = False
    sched = [(on_a, on_b)]
    for t in range(1, period):
        if (t + oa) % pa == 0:
            on_a = not on_a
        if (t + ob) % pb == 0:
            on_b = not on_b
        sched.append((on_a, on_b))
    return tuple(sched)


def _blocked_masks(gw: int, gh: int, cells_a, cells_b, sched):
    """The per-phase blocked-cell masks `PhasedGridField` is defined over: a cell
    blocks at phase ``t`` iff its set is ON at ``t``."""
    import numpy as np

    ncell = gw * gh
    a = np.zeros(ncell, dtype=bool)
    b = np.zeros(ncell, dtype=bool)
    for x, y in cells_a:
        a[y * gw + x] = True
    for x, y in cells_b:
        b[y * gw + x] = True
    blocked = np.empty((len(sched), ncell), dtype=bool)
    for t, (on_a, on_b) in enumerate(sched):
        blocked[t] = (a if on_a else False) | (b if on_b else False)
    return blocked


def _sample_periods(rng, lo: int, hi: int):
    """``(pa, pb, oa, ob)`` with ``pa != pb`` -- two genuinely independent timers."""
    pa = rng.randint(lo, hi)
    choices = [p for p in range(lo, hi + 1) if p != pa]
    if not choices:
        return None
    pb = rng.choice(choices)
    return pa, pb, rng.randrange(pa), rng.randrange(pb)


def _sample_hazards(rng, params: dict, gw: int, gh: int):
    """``(cells_a, cells_b)`` for the level's style, or ``(None, None)`` if the draw
    did not land. The two sets are kept DISJOINT: a shared cell would be decided by
    whichever sprite happens to be on top, so its partner would be invisible and the
    board would lie about which timer governs it."""
    style = params["style"]
    if style == "posts":
        n = rng.randint(*params["posts"])
        cells = [(x, y) for x in range(gw) for y in range(gh)]
        rng.shuffle(cells)
        pick: list = []
        for c in cells:
            if len(pick) == 2 * n:
                break
            # Never orthogonally adjacent, so posts stay obstacles to time rather
            # than merging into an accidental wall.
            if any(abs(c[0] - o[0]) + abs(c[1] - o[1]) <= 1 for o in pick):
                continue
            pick.append(c)
        if len(pick) < 2 * n:
            return None, None
        return pick[:n], pick[n:]

    if style == "bars":
        col, row = rng.randint(1, gw - 2), rng.randint(1, gh - 2)
        ra, rb = rng.randint(*params["run"]), rng.randint(*params["run"])
        if ra > gh or rb > gw:
            return None, None
        ya, xb = rng.randint(0, gh - ra), rng.randint(0, gw - rb)
        a = [(col, ya + i) for i in range(ra)]
        b = [(xb + i, row) for i in range(rb)]
        return (None, None) if set(a) & set(b) else (a, b)

    if style == "combs":
        row, col = rng.randint(2, gh - 3), rng.randint(2, gw - 3)
        a = [(x, row) for x in range(gw) if x % 2 == rng.randint(0, 1)]
        b = [(col, y) for y in range(gh) if y % 2 == rng.randint(0, 1)]
        return (None, None) if set(a) & set(b) else (a, b)

    if style == "walls":
        # Two parallel lines with at least one clear column/row between them; the
        # second carries a single doorway, so one timer is absolute and one is not.
        if rng.random() < 0.5:
            lo = rng.randint(1, gw - 4)
            hi = rng.randint(lo + 2, gw - 2)
            a = [(lo, y) for y in range(gh)]
            door = rng.randrange(gh)
            b = [(hi, y) for y in range(gh) if y != door]
        else:
            lo = rng.randint(1, gh - 4)
            hi = rng.randint(lo + 2, gh - 2)
            a = [(x, lo) for x in range(gw)]
            door = rng.randrange(gw)
            b = [(x, hi) for x in range(gw) if x != door]
        return a, b

    raise ValueError(f"unknown hazard style {style!r}")


def _sample_layout(rng, params: dict, fallback):
    """A per-seed layout, certified winnable and certified to make the timers MATTER.

    Returns ``{"gw", "gh", "pa", "pb", "oa", "ob", "cells_a", "cells_b", "player",
    "target", "steps"}``.

    The goal is drawn first and the field is built ONCE for it, which scores EVERY
    start cell at the same time -- so an attempt costs one BFS instead of one per
    candidate pair, and the player is then drawn uniformly among the cells that pass.
    A start passes when its optimum sits inside the level's step band and is strictly
    longer than the plain Manhattan walk (with no walls in this game, a shorter-than-
    Manhattan route is impossible and an equal one means the hazards never bit).
    Falls back to the stock layout if no attempt lands one."""
    lo, hi = params["size"]
    smin, smax = params["steps"]
    for _ in range(_LAYOUT_ATTEMPTS):
        gw, gh = rng.randint(lo, hi), rng.randint(lo, hi)
        timing = _sample_periods(rng, *params["periods"])
        if timing is None:
            continue
        pa, pb, oa, ob = timing
        cells_a, cells_b = _sample_hazards(rng, params, gw, gh)
        if not cells_a or not cells_b:
            continue
        haz = set(cells_a) | set(cells_b)
        # The HUD overlays both bottom corner cells (phase indicators bottom-right,
        # blocked-move flash bottom-left), so keep the player and the goal off them --
        # both have to read cleanly at every frame.
        free = [(x, y) for x in range(gw) for y in range(gh)
                if (x, y) not in haz and (x, y) not in ((gw - 1, gh - 1), (0, gh - 1))]
        if len(free) < 2:
            continue
        sched = blink_schedule(pa, pb, oa, ob)
        target = rng.choice(free)
        field = PhasedGridField(gw, gh,
                                _blocked_masks(gw, gh, cells_a, cells_b, sched),
                                [target[1] * gw + target[0]])
        starts = []
        for player in free:
            if player == target:
                continue
            steps = field.steps_to_win(field.cell(*player), 0)
            if steps is None or not (smin <= steps <= smax):
                continue
            if steps <= abs(player[0] - target[0]) + abs(player[1] - target[1]):
                continue
            starts.append((player, steps))
        if not starts:
            continue
        player, steps = rng.choice(starts)
        return {"gw": gw, "gh": gh, "pa": pa, "pb": pb, "oa": oa, "ob": ob,
                "cells_a": sorted(cells_a), "cells_b": sorted(cells_b),
                "player": player, "target": target, "steps": steps}

    grid, pa, pb, oa, ob, cells_a, cells_b, player, target = fallback
    return {"gw": grid[0], "gh": grid[1], "pa": pa, "pb": pb, "oa": oa, "ob": ob,
            "cells_a": sorted(cells_a), "cells_b": sorted(cells_b),
            "player": player, "target": target, "steps": None}


class Zq02(AugmentedGame):
    """Reach the goal past two hazard sets blinking on independent timers."""

    def __init__(self, seed: int | None = None) -> None:
        # FIRST: the base draws this level's rotation and turns the BOARD by it in
        # set_level, before the engine seats the level.
        self._init_augmentation(seed)
        self._color_scheme: dict | None = None
        self._layout: dict | None = None
        # Seeded draws are a pure function of (seed, level), so a level re-seated by a
        # RESET / a level advance / a jump would redraw the identical layout -- memoise
        # it instead of paying the rejection sampling again. Unseeded (interactive)
        # play deliberately redraws, so it is never cached.
        self._layout_cache: dict = {}
        self._ui = Zq02UI(False, False)
        super().__init__(
            "zq02",
            levels,
            Camera(0, 0, 16, 16, BACKGROUND_COLOR, PADDING_COLOR,
                   [self._ui]),
            False,
            1,
            [1, 2, 3, 4],
        )

    # ── per-seed augmentation ───────────────────────────────────────────────
    def _level_rng(self, index: int):
        if self._seed_value is None:               # interactive play: a fresh board
            return _random_module.Random()
        return _random_module.Random(f"zq02:{self._seed_value}:{int(index)}")

    def _ensure_color_scheme(self) -> None:
        if self._color_scheme is not None:
            return
        rng = (_random_module.Random(f"zq02-colors:{self._seed_value}")
               if self._seed_value is not None else _random_module.Random())
        bg, player, target, haz_a, haz_b, idle, flash = rng.sample(_COLOR_POOL, 7)
        self._color_scheme = {"background": bg, "player": player, "target": target,
                              "hazard_a": haz_a, "hazard_b": haz_b,
                              "idle": idle, "flash": flash}

    def full_reset(self) -> None:
        # A full reset starts a new episode -> re-draw the colour scheme.
        self._color_scheme = None
        super().full_reset()

    # ── engine hooks ────────────────────────────────────────────────────────
    def set_level(self, index: int) -> None:
        """Draw this level's layout and pin its board size before the engine reads it.

        ``ARCBaseGame.set_level`` resizes the camera from ``level.grid_size`` and only
        THEN calls ``on_set_level``, so the drawn size has to be installed on the level
        object here, ahead of the base call (which also draws the rotation -- see
        ``AugmentedGame.set_level``)."""
        if 0 <= index < len(self._levels):
            self._layout = self._layout_cache.get(index)
            if self._layout is None:
                self._layout = _sample_layout(self._level_rng(index),
                                              _LEVEL_PARAMS[index], _FALLBACKS[index])
                if self._seed_value is not None:
                    self._layout_cache[index] = self._layout
            self._levels[index]._grid_size = (self._layout["gw"], self._layout["gh"])
        super().set_level(index)

    def on_set_level(self, level: Level) -> None:
        # Every sprite is built here, so drop whatever a previous seating of this same
        # persistent Level object left behind -- otherwise a RESET or a jump-to-level
        # stacks a ghost player and second copies of both hazard sets on the board.
        clear_dynamic_sprites(
            level, tags=["player", "target", "zq2_hazard_a", "zq2_hazard_b"])
        lay = self._layout
        self._ensure_color_scheme()
        scheme = self._color_scheme
        self.camera.background = scheme["background"]
        self._ui.set_colors(scheme)
        # The blocked-move flash belongs to the level that produced it; without this a
        # flash from the last move of one level bleeds into the opening frames of the
        # next.
        self._ui.clear_flash()

        self._period_a = int(lay["pa"])
        self._period_b = int(lay["pb"])
        self._offset_a = int(lay["oa"])
        self._offset_b = int(lay["ob"])

        player = sprites["player"].clone()
        player.color_remap(PLAYER_COLOR, scheme["player"])
        level.add_sprite(player.set_position(*lay["player"]))
        target = sprites["target"].clone()
        target.color_remap(TARGET_COLOR, scheme["target"])
        level.add_sprite(target.set_position(*lay["target"]))
        self._player = player
        self._targets = [target]

        self._sprites_a: list[Sprite] = []
        self._sprites_b: list[Sprite] = []
        for key, cells, base, bucket in (
                ("hazard_a", lay["cells_a"], HAZARD_A_COLOR, self._sprites_a),
                ("hazard_b", lay["cells_b"], HAZARD_B_COLOR, self._sprites_b)):
            for hx, hy in cells:
                s = sprites[key].clone()
                s.color_remap(base, scheme[key])
                s.set_position(hx, hy)
                bucket.append(s)
                level.add_sprite(s)

        self._ticks = 0
        self._on_a = False
        self._on_b = False
        self._sync()

    def _sync(self) -> None:
        for s in self._sprites_a:
            s.set_visible(self._on_a)
            s.set_collidable(self._on_a)
        for s in self._sprites_b:
            s.set_visible(self._on_b)
            s.set_collidable(self._on_b)
        self._ui.update(self._on_a, self._on_b)

    def step(self) -> None:
        # The timers tick on EVERY action, before the move is resolved -- so a move is
        # resolved against the configuration it lands in, and a blocked move still
        # burns a tick (that is what makes stalling a legal way to fix parity).
        self._ui.tick_flash()
        self._ticks += 1
        if (self._ticks + self._offset_a) % self._period_a == 0:
            self._on_a = not self._on_a
        if (self._ticks + self._offset_b) % self._period_b == 0:
            self._on_b = not self._on_b
        self._sync()

        # The board is displayed rotated, so the pressed SCREEN direction has to be
        # mapped back to game space before it means anything here.
        action_id = self.screen_action_to_game(self.action.id)
        dx = dy = 0
        if action_id == GameAction.ACTION1:
            dy = -1
        elif action_id == GameAction.ACTION2:
            dy = 1
        elif action_id == GameAction.ACTION3:
            dx = -1
        elif action_id == GameAction.ACTION4:
            dx = 1

        if dx == 0 and dy == 0:
            self.complete_action()
            return

        nx = self._player.x + dx
        ny = self._player.y + dy
        grid_w, grid_h = self.current_level.grid_size
        if not (0 <= nx < grid_w and 0 <= ny < grid_h):
            self.complete_action()
            return

        sprite = self.current_level.get_sprite_at(nx, ny, ignore_collidable=True)
        if sprite and sprite.is_collidable and (
            "zq2_hazard_a" in sprite.tags or "zq2_hazard_b" in sprite.tags
        ):
            self._ui.flash_hazard_block()
            self.complete_action()
            return

        if not sprite or not sprite.is_collidable:
            self._player.set_position(nx, ny)

        for t in self._targets:
            if self._player.x == t.x and self._player.y == t.y:
                self.next_level()
                break

        self.complete_action()
