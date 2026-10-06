"""Zone Timer: hazard tiles blink on and off on a fixed timer; cross on the safe phase.

An open grid with a player, a goal cell and a fixed set of HAZARD cells. Every
``period`` actions the whole hazard set toggles: while it is ON the cells are visible
and block movement, while it is OFF they are invisible and harmless. Nothing is
lethal and there are no walls, so the only question the board asks is a TIMING one --
walk the long way round, or stand still until the barrier blinks off and go straight
through? A blocked move (into an on-hazard, or off the board) still consumes the
action, so stalling is a real move and phase parity is a resource the player controls.

The 3x3 patch in the bottom-right corner is the phase indicator: one colour while the
hazards are on, another while they are off. It is the only thing on the board that
shows the timer, and it does not show how much of the current phase is LEFT -- that
has to be inferred from watching the blink, which is the game.

PER-SEED AUGMENTATION (the repo convention -- see utils/arc_game.py and the per-seed
augmentation notes). ``Zq01(seed=s)`` makes every level a pure function of
``(s, level_index)``:

  * DISPLAY ROTATION, owned by ``AugmentedGame``: the frame is rendered rotated by a
    per-(seed, level) k and ``step`` maps the pressed SCREEN direction back to game
    space via ``screen_action_to_game``. No rotation code lives here.
  * LAYOUT: board size, the timer PERIOD, the hazard pattern and the player/goal
    cells are all re-drawn per (seed, level), keeping each level's stock character --
    see ``_LEVEL_PARAMS`` (a short segment, a solid blob, a comb, a full barrier, a
    barrier with open ends).
  * COLOURS: one EPISODE-scoped scheme (level-independent seed, cached), so every
    level of one episode looks the same. Padding (4) stays fixed.

SOLVABILITY CERTIFICATE. A drawn board is accepted only if ``utils.zone_timer``'s
exact ``(cell, phase)`` distance-to-win field says it is winnable from the start
state, that the optimum sits inside the level's step band, AND that the optimum is
strictly longer than the plain Manhattan walk -- i.e. the timer actually costs the
player something. Boards where the hazards are decoration are rejected rather than
recorded. That is the SAME field the training generator's expert descends
(solvers/generate_zq01_training.py), so every board this game can draw is one the
generator can solve optimally.
"""

import random as _random_module

from arcengine import (
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)

from utils.arc_game import AugmentedGame, clear_dynamic_sprites
from utils.zone_timer import ZoneTimerField

BACKGROUND_COLOR = 5
PADDING_COLOR = 4
PLAYER_COLOR = 9
TARGET_COLOR = 11
HAZARD_COLOR = 8
PHASE_ON_COLOR = 8
PHASE_OFF_COLOR = 14

#: Colours the per-episode scheme draws from, mutually distinct so nothing hides
#: against anything else. Excludes 0 and the fixed padding colour (4).
_COLOR_POOL = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# Render / collision layers. ``Camera.render`` draws LOW layers first and
# ``get_sprite_at(..., ignore_collidable=True)`` returns the HIGHEST layer at a cell,
# so the player must sit ABOVE the hazards: a player standing on a hazard cell when it
# blinks on has to stay visible (with every sprite on layer 0 the engine falls back to
# insertion order and the hazard, added last, would paint the player out). Ordering the
# hazards above the goal likewise makes "blocked" mean the same thing to the renderer,
# to ``step`` and to the solver's field, whatever cells a drawn layout happens to share.
_LAYER_FLOOR = 0
_LAYER_HAZARD = 1
_LAYER_PLAYER = 2


class Zq01UI(RenderableUserDisplay):
    """The phase indicator: a 3x3 patch in the bottom-right corner, one colour while
    the hazards block and another while they do not."""

    def __init__(self, phase: bool = False, on_color: int = PHASE_ON_COLOR,
                 off_color: int = PHASE_OFF_COLOR) -> None:
        self._phase = phase
        self._on_color = on_color
        self._off_color = off_color

    def update(self, phase: bool) -> None:
        self._phase = phase

    def set_colors(self, on_color: int, off_color: int) -> None:
        self._on_color = on_color
        self._off_color = off_color

    def render_interface(self, frame):
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return frame
        h, w = frame.shape
        c = self._on_color if self._phase else self._off_color
        for dy in range(3):
            for dx in range(3):
                frame[h - 3 + dy, w - 4 + dx] = c
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
    "hazard": Sprite(
        pixels=[[HAZARD_COLOR]],
        name="hazard",
        visible=True,
        collidable=True,
        tags=["zq_hazard"],
        layer=_LAYER_HAZARD,
    ),
}


def mk(grid_size, difficulty: int) -> Level:
    """A level TEMPLATE. Every sprite is built per-seating in ``on_set_level``, so a
    template carries nothing but its board size and difficulty; ``grid_size`` here is
    only the fallback size (``set_level`` pins the drawn one before the engine reads
    it)."""
    return Level(sprites=[], grid_size=grid_size, data={"difficulty": difficulty})


levels = [mk((8, 8), 1), mk((8, 8), 2), mk((8, 8), 3), mk((10, 8), 4), mk((8, 8), 5)]

# Per-level augmentation budget, index-aligned with ``levels`` and with
# ``_FALLBACKS``. Each entry keeps its stock level's character:
#   style   -- the hazard pattern (see ``_sample_hazards``)
#   size    -- inclusive range each board dimension is drawn from
#   period  -- inclusive range for the blink period, in actions
#   steps   -- inclusive band the optimal solution length must land in
_LEVEL_PARAMS = [
    # L1 -- a short bar you can either wait out or walk around
    {"style": "segment", "size": (8, 9), "period": (4, 6), "run": (3, 5),
     "steps": (7, 26)},
    # L2 -- a solid blob in the middle of an open floor
    {"style": "blob", "size": (8, 9), "period": (3, 5), "box": (2, 3),
     "steps": (7, 26)},
    # L3 -- a leaky barrier: a full line with a couple of doorways punched in it
    {"style": "comb", "size": (8, 10), "period": (5, 7), "holes": (2, 3),
     "steps": (9, 34)},
    # L4 -- a solid barrier: it MUST be crossed on an off phase
    {"style": "wall", "size": (9, 10), "period": (4, 6), "steps": (9, 40)},
    # L5 -- a barrier with both ends open: round the end, or straight through
    {"style": "gate", "size": (8, 10), "period": (3, 4), "steps": (9, 40)},
]

#: The stock (pre-augmentation) layouts, used verbatim when sampling fails so a level
#: is always well-formed. ``(grid, period, hazards, player, target)``.
_FALLBACKS = [
    ((8, 8), 5, [(4, 2), (4, 3), (4, 4), (4, 5)], (0, 3), (7, 3)),
    ((8, 8), 4, [(3, 3), (4, 3), (3, 4), (4, 4)], (1, 1), (6, 6)),
    ((8, 8), 6, [(2, y) for y in range(8) if y % 2 == 0], (0, 0), (7, 7)),
    ((10, 8), 5, [(5, y) for y in range(8)], (2, 4), (7, 4)),
    ((8, 8), 3, [(x, 4) for x in range(8) if x not in (0, 7)], (0, 7), (7, 0)),
]

_LAYOUT_ATTEMPTS = 80


def _sample_line(rng, gw: int, gh: int):
    """A random full-board line as ``(axis, index)`` -- ``("col", c)`` or
    ``("row", r)`` -- leaving at least two cells of open floor on each side."""
    if rng.random() < 0.5:
        return "col", rng.randint(2, gw - 3)
    return "row", rng.randint(2, gh - 3)


def _line_cells(axis: str, idx: int, gw: int, gh: int) -> list:
    if axis == "col":
        return [(idx, y) for y in range(gh)]
    return [(x, idx) for x in range(gw)]


def _sample_hazards(rng, params: dict, gw: int, gh: int):
    """A hazard pattern as ``(cells, axis, index)``.

    ``axis``/``index`` name the line the pattern lies on, or ``(None, None)`` for the
    blob -- the caller uses it to place the player and the goal on OPPOSITE sides, so
    the barrier is on the route rather than off to one side.
    """
    style = params["style"]
    if style == "blob":
        bw = rng.randint(*params["box"])
        bh = rng.randint(*params["box"])
        if bw >= gw - 2 or bh >= gh - 2:
            return [], None, None
        x0 = rng.randint(1, gw - bw - 1)
        y0 = rng.randint(1, gh - bh - 1)
        return [(x0 + i, y0 + j) for i in range(bw) for j in range(bh)], None, None

    axis, idx = _sample_line(rng, gw, gh)
    line = _line_cells(axis, idx, gw, gh)
    if style == "wall":
        return line, axis, idx
    if style == "gate":                        # both ends open
        return line[1:-1], axis, idx
    if style == "comb":                        # a few doorways punched in the line
        holes = rng.randint(*params["holes"])
        if holes >= len(line) - 1:
            return [], None, None
        open_cells = set(rng.sample(range(len(line)), holes))
        return [c for i, c in enumerate(line) if i not in open_cells], axis, idx
    if style == "segment":                     # a contiguous run of the line
        run = rng.randint(*params["run"])
        if run >= len(line):
            return [], None, None
        start = rng.randint(0, len(line) - run)
        return line[start:start + run], axis, idx
    raise ValueError(f"unknown hazard style {style!r}")


def _opposite_sides(axis, idx, player, target) -> bool:
    """Do ``player`` and ``target`` sit on opposite sides of the hazard line? (Always
    true for a blob, which has no line.)"""
    if axis is None:
        return True
    i = 0 if axis == "col" else 1
    a, b = player[i] - idx, target[i] - idx
    return a * b < 0                            # strictly opposite, neither ON the line


def _sample_layout(rng, params: dict, fallback):
    """A per-seed layout, certified winnable and certified to make the timer MATTER.

    Returns ``{"gw", "gh", "period", "hazard", "player", "target", "steps"}``.

    The goal is drawn first and the field is built ONCE for it, which scores EVERY
    start cell at the same time -- so an attempt costs one BFS instead of one per
    candidate pair, and the player is then drawn uniformly among the cells that pass.
    A start passes when it is on the far side of the barrier, its optimum sits inside
    the level's step band, and that optimum is strictly longer than the plain
    Manhattan walk (i.e. the timer costs the player something -- boards where the
    hazards are decoration are rejected). Falls back to the stock layout if no attempt
    lands one, so a level is always well-formed."""
    lo, hi = params["size"]
    smin, smax = params["steps"]
    for _ in range(_LAYOUT_ATTEMPTS):
        gw, gh = rng.randint(lo, hi), rng.randint(lo, hi)
        period = rng.randint(*params["period"])
        cells, axis, idx = _sample_hazards(rng, params, gw, gh)
        if not cells:
            continue
        haz = set(cells)
        # The phase indicator overlays the bottom-right corner cell, so keep the
        # player and the goal off it -- both have to read cleanly at every frame.
        free = [(x, y) for x in range(gw) for y in range(gh)
                if (x, y) not in haz and (x, y) != (gw - 1, gh - 1)]
        if len(free) < 2:
            continue
        target = rng.choice(free)
        field = ZoneTimerField(gw, gh, period, haz, target)
        starts = []
        for player in free:
            if player == target or not _opposite_sides(axis, idx, player, target):
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
        return {"gw": gw, "gh": gh, "period": period, "hazard": sorted(haz),
                "player": player, "target": target, "steps": steps}

    grid, period, cells, player, target = fallback
    return {"gw": grid[0], "gh": grid[1], "period": period,
            "hazard": sorted(cells), "player": player, "target": target,
            "steps": None}


class Zq01(AugmentedGame):
    """Cross to the goal past hazard tiles that blink on and off on a fixed timer."""

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
        self._ui = Zq01UI(False)
        super().__init__(
            "zq01",
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
        return _random_module.Random(f"zq01:{self._seed_value}:{int(index)}")

    def _ensure_color_scheme(self) -> None:
        if self._color_scheme is not None:
            return
        rng = (_random_module.Random(f"zq01-colors:{self._seed_value}")
               if self._seed_value is not None else _random_module.Random())
        bg, player, target, hazard, on, off = rng.sample(_COLOR_POOL, 6)
        self._color_scheme = {"background": bg, "player": player, "target": target,
                              "hazard": hazard, "phase_on": on, "phase_off": off}

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
        ``AugmentedGame.set_level``). The whole layout is drawn in ONE place so the
        size the camera gets and the sprites ``on_set_level`` builds always come from
        the same draw."""
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
        # stacks a ghost player and a second copy of the hazard set on the board.
        clear_dynamic_sprites(level, tags=["player", "target", "zq_hazard"])
        lay = self._layout
        self._ensure_color_scheme()
        scheme = self._color_scheme
        self.camera.background = scheme["background"]
        self._ui.set_colors(scheme["phase_on"], scheme["phase_off"])

        self._period = int(lay["period"])
        player = sprites["player"].clone()
        player.color_remap(PLAYER_COLOR, scheme["player"])
        level.add_sprite(player.set_position(*lay["player"]))
        target = sprites["target"].clone()
        target.color_remap(TARGET_COLOR, scheme["target"])
        level.add_sprite(target.set_position(*lay["target"]))
        self._player = player
        self._targets = [target]

        self._hazard_sprites: list[Sprite] = []
        for hx, hy in lay["hazard"]:
            s = sprites["hazard"].clone()
            s.color_remap(HAZARD_COLOR, scheme["hazard"])
            s.set_position(hx, hy)
            self._hazard_sprites.append(s)
            level.add_sprite(s)

        self._ticks = 0
        self._hazard_on = False
        self._sync_hazards()

    def _sync_hazards(self) -> None:
        for s in self._hazard_sprites:
            s.set_visible(self._hazard_on)
            s.set_collidable(self._hazard_on)
        self._ui.update(self._hazard_on)

    def step(self) -> None:
        # The timer ticks on EVERY action, before the move is resolved -- so a move is
        # resolved against the phase it lands in, and a blocked move still burns a
        # tick (that is what makes stalling a legal way to fix parity).
        self._ticks += 1
        if self._ticks % self._period == 0:
            self._hazard_on = not self._hazard_on
            self._sync_hazards()

        # The board is displayed rotated, so the pressed SCREEN direction has to be
        # mapped back to game space before it means anything here.
        action_id = self.screen_action_to_game(self.action.id)
        dx = 0
        dy = 0
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
        if sprite and "zq_hazard" in sprite.tags and sprite.is_collidable:
            self.complete_action()
            return

        if not sprite or not sprite.is_collidable:
            self._player.set_position(nx, ny)

        for t in self._targets:
            if self._player.x == t.x and self._player.y == t.y:
                self.next_level()
                break

        self.complete_action()
