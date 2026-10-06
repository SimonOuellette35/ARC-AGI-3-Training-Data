"""Telekinetic Tug (tk01): drag the block onto the goal.

A 10x10 board holding one player, one block, one goal marker and some walls.
Walking into the block PUSHES it one cell further (if the cell behind it is
free); ACTION5 TUGS it one cell towards the player, along whichever axis it is
further away on. Push and tug are inverses, so the block can be manoeuvred
anywhere the walls allow -- which is also what makes the game recoverable from an
arbitrary detour (see solvers/generate_tk01_training.py).

The level advances when a WALK leaves the block on the goal. Tugging the block
onto the goal therefore does not finish the level by itself: the HUD lights its
"block on goal" pixel and one more directional press -- any of the four; at most
one of them would shove the block back off -- closes it out. That is the stock
rule (``step`` only tests the goal after ACTION1..4) and it is kept.

PER-SEED AUGMENTATION (the repo convention -- see utils/arc_game.py and the
per-seed-augmentation notes). ``Tk01(seed=s)`` makes every level a pure function
of ``(s, level_index)``:

  * DISPLAY ROTATION, owned by ``AugmentedGame``: the frame is rendered rotated by
    a per-(seed, level) k and ``step`` maps the pressed SCREEN direction back to
    game space via ``screen_action_to_game``. No rotation code lives here.
  * LAYOUT: walls, goal, player and block are re-drawn per (seed, level).
  * COLOURS: the player and block colours come from one EPISODE-scoped scheme
    (level-independent seed, cached), so every level of an episode looks alike.
    Goal (14), wall (3), background (5) and padding (4) stay fixed, as in the rest
    of this game family (nu01 / mb01 / fs02 / pw01).

SOLVABILITY AND DIFFICULTY COME FROM THE SAME SEARCH THE SOLVER USES.
``utils/tug_puzzle.TugField`` is an exact distance-to-win field over all 10k
``(player, block)`` states, and it costs ~10 ms, so a sampled level does not need
a hand-written solvability certificate: draw walls and a goal, build the field,
and then pick the player/block start UNIFORMLY AMONG THE STATES WHOSE OPTIMAL
SOLUTION IS THE LENGTH THIS LEVEL WANTS (``_LEVEL_PARAMS``). Every level is
therefore solvable by construction *and* on the intended rung of the difficulty
ramp, and the generator's expert re-derives the same field.
"""

from __future__ import annotations

import random as _random_module

import numpy as np

from arcengine import Camera, GameAction, GameState, Level, RenderableUserDisplay, Sprite

from utils.arc_game import AugmentedGame
from utils.tug_puzzle import TugField

BACKGROUND_COLOR = 5
PADDING_COLOR = 4
PLAYER_COLOR = 9
GOAL_COLOR = 14
BLOCK_COLOR = 15
WALL_COLOR = 3

# Colours drawn for the player / block. Excludes the fixed goal (14), wall (3),
# background (5), padding (4) and 0; the two are sampled mutually distinct so
# neither hides against the other or against the board.
_COLOR_POOL = [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 15]

# Render/collision layers. ``get_sprite_at(..., ignore_collidable=True)`` returns
# the HIGHEST layer at a cell and the camera draws the highest layer last, so
# these keep collision and rendering in agreement: the goal is a floor marker, and
# a block parked on it still reads -- and still behaves -- as a block.
_LAYER_FLOOR = 0
_LAYER_SOLID = 1
_LAYER_PLAYER = 2

GRID_W, GRID_H = 10, 10


class Tk01UI(RenderableUserDisplay):
    """Bottom strip: one dot per level (done / current / to come), a pixel that
    lights while the block sits on the goal, and the win/lose bar."""

    def __init__(self, num_levels: int) -> None:
        self._num_levels = num_levels
        self._level_index = 0
        self._state = None
        self._block_on_goal = False
        self._block_color = BLOCK_COLOR

    def update(self, *, level_index: int | None = None, state=None,
               block_on_goal: bool | None = None,
               block_color: int | None = None) -> None:
        if level_index is not None:
            self._level_index = level_index
        if state is not None:
            self._state = state
        if block_on_goal is not None:
            self._block_on_goal = block_on_goal
        if block_color is not None:
            self._block_color = block_color

    def render_interface(self, f):
        if not isinstance(f, np.ndarray):
            return f
        h, w = f.shape
        # The board occupies the top-left of the frame, so the HUD lives in the
        # empty strip below it rather than overwriting the first row of cells.
        for i in range(min(self._num_levels, 14)):
            x = 1 + i * 2
            if x < w:
                f[h - 2, x] = 14 if i < self._level_index else (
                    11 if i == self._level_index else 3)
        if self._block_on_goal and w > 12:
            f[h - 2, 12] = self._block_color
        if self._state in (GameState.GAME_OVER, GameState.WIN):
            f[h - 1, :min(w, 16)] = 14 if self._state == GameState.WIN else 8
        return f


sprites = {
    "player": Sprite(pixels=[[PLAYER_COLOR]], name="player", visible=True,
                     collidable=True, tags=["player"], layer=_LAYER_PLAYER),
    "goal": Sprite(pixels=[[GOAL_COLOR]], name="goal", visible=True,
                   collidable=False, tags=["goal"], layer=_LAYER_FLOOR),
    "block": Sprite(pixels=[[BLOCK_COLOR]], name="block", visible=True,
                    collidable=True, tags=["block"], layer=_LAYER_SOLID),
    "wall": Sprite(pixels=[[WALL_COLOR]], name="wall", visible=True,
                   collidable=True, tags=["wall"], layer=_LAYER_SOLID),
}


def _mk(player, goal, block, walls, difficulty: int) -> Level:
    parts = [sprites["goal"].clone().set_position(*goal),
             sprites["block"].clone().set_position(*block),
             sprites["player"].clone().set_position(*player)]
    parts += [sprites["wall"].clone().set_position(*c) for c in walls]
    return Level(sprites=parts, grid_size=(GRID_W, GRID_H),
                 data={"difficulty": difficulty})


# The stock boards. They are the fallback an unseeded/failed sample falls back to;
# with a seed every one of them is redrawn by ``_sample_layout``.
levels = [
    _mk((1, 5), (8, 5), (5, 5), (), 1),
    _mk((2, 2), (8, 8), (6, 4), (), 2),
    _mk((1, 1), (8, 8), (4, 4), ((5, 4),), 3),
    _mk((0, 5), (9, 5), (7, 5), (), 4),
    _mk((3, 3), (7, 7), (5, 6), (), 5),
]

# Per-level augmentation budget, index-aligned with ``levels``:
#   line    -- draw a full wall line (row or column) with a two-cell doorway
#   scatter -- how many extra loose wall cells
#   length  -- inclusive range for the level's OPTIMAL solution, in actions
# The ramp is the point of the whole sampler: a level is re-drawn until it admits
# a start state whose exact optimal solution falls in ``length``.
_LEVEL_PARAMS = [
    {"line": False, "scatter": 0, "length": (4, 9)},
    {"line": False, "scatter": 3, "length": (9, 15)},
    {"line": True, "scatter": 0, "length": (14, 22)},
    {"line": True, "scatter": 3, "length": (18, 28)},
    {"line": True, "scatter": 5, "length": (22, 40)},
]

_LAYOUT_ATTEMPTS = 40


def _sample_walls(rng, params: dict, gw: int, gh: int) -> set:
    """This level's wall cells: an optional line with a two-cell doorway, plus
    ``scatter`` loose cells."""
    cells: set = set()
    if params["line"]:
        if rng.random() < 0.5:                     # a vertical wall at column `line`
            col = rng.randint(2, gw - 3)           # keep >=2 columns each side
            door = rng.randint(0, gh - 2)          # doorway at `door`, `door + 1`
            cells = {(col, y) for y in range(gh) if y not in (door, door + 1)}
        else:                                      # a horizontal wall at row `line`
            row = rng.randint(2, gh - 3)
            door = rng.randint(0, gw - 2)
            cells = {(x, row) for x in range(gw) if x not in (door, door + 1)}
    free = [(x, y) for x in range(gw) for y in range(gh) if (x, y) not in cells]
    cells.update(rng.sample(free, min(params["scatter"], len(free))))
    return cells


def _sample_layout(rng, params: dict, gw: int, gh: int):
    """A solvable layout on this level's difficulty rung, or None.

    Draw walls + a goal, build the exact field, and take the start state from the
    states whose optimal solution is in range. If no draw offers one (over-walled
    boards can cap out short), fall back to the deepest state seen -- still exact,
    just easier than asked."""
    lo, hi = params["length"]
    fallback = None
    for _ in range(_LAYOUT_ATTEMPTS):
        walls = _sample_walls(rng, params, gw, gh)
        free = [(x, y) for x in range(gw) for y in range(gh) if (x, y) not in walls]
        if len(free) < 3:
            continue
        goal = rng.choice(free)
        field = TugField(gw, gh, walls, goal)
        states = field.start_states(lo, hi)
        if states.size:
            code = int(states[rng.randrange(states.size)])
            player, block = divmod(code, field.n)
            return {"walls": sorted(walls), "goal": goal,
                    "player": field.xy(player), "block": field.xy(block)}
        deep = field.deepest()
        if deep >= 2 and (fallback is None or deep > fallback[0]):
            deepest = field.start_states(deep, deep)
            if deepest.size:
                code = int(deepest[rng.randrange(deepest.size)])
                player, block = divmod(code, field.n)
                fallback = (deep, {"walls": sorted(walls), "goal": goal,
                                   "player": field.xy(player),
                                   "block": field.xy(block)})
    return fallback[1] if fallback else None


class Tk01(AugmentedGame):
    def __init__(self, seed: int | None = None) -> None:
        # FIRST: the base draws this level's rotation and turns the BOARD by it in
        # set_level, before the engine seats the level.
        self._init_augmentation(seed)
        # Episode-scoped colour scheme (level-INDEPENDENT), drawn once and cached
        # so every level of one episode is coloured identically.
        self._color_scheme: dict | None = None
        self._ui = Tk01UI(len(levels))
        super().__init__(
            "tk01", levels,
            Camera(0, 0, 16, 16, BACKGROUND_COLOR, PADDING_COLOR,
                   [self._ui]),
            False, 1, [1, 2, 3, 4, 5])

    # ── per-seed augmentation ───────────────────────────────────────────────
    def _level_rng(self) -> "_random_module.Random":
        if self._seed_value is None:                # interactive play: fresh board
            return _random_module.Random()
        return _random_module.Random(
            f"tk01:{self._seed_value}:{self._current_level_index}")

    def _ensure_color_scheme(self) -> None:
        if self._color_scheme is not None:
            return
        rng = (_random_module.Random(f"tk01-colors:{self._seed_value}")
               if self._seed_value is not None else _random_module.Random())
        player, block = rng.sample(_COLOR_POOL, 2)
        self._color_scheme = {"player": player, "block": block}

    def _apply_layout(self, level: Level, rng) -> None:
        gw, gh = level.grid_size
        layout = _sample_layout(rng, _LEVEL_PARAMS[self._current_level_index], gw, gh)
        if layout is None:                          # keep the stock board
            return
        level.get_sprites_by_tag("player")[0].set_position(*layout["player"])
        level.get_sprites_by_tag("block")[0].set_position(*layout["block"])
        level.get_sprites_by_tag("goal")[0].set_position(*layout["goal"])
        # The wall COUNT varies per draw, so walls are rebuilt rather than moved.
        # ``on_set_level`` re-clones the clean template first, which is what makes
        # that idempotent under a repeated seating (RESET, jump-to-level).
        for w in level.get_sprites_by_tag("wall"):
            level.remove_sprite(w)
        for cell in layout["walls"]:
            level.add_sprite(sprites["wall"].clone().set_position(*cell))

    def _apply_colors(self, level: Level) -> None:
        self._ensure_color_scheme()
        scheme = self._color_scheme
        for s in level.get_sprites_by_tag("player"):
            s.color_remap(PLAYER_COLOR, scheme["player"])
        for s in level.get_sprites_by_tag("block"):
            s.color_remap(BLOCK_COLOR, scheme["block"])

    def full_reset(self) -> None:
        # A full reset starts a new episode -> a new colour scheme.
        self._color_scheme = None
        super().full_reset()

    # ── engine hooks ────────────────────────────────────────────────────────
    def on_set_level(self, level: Level) -> None:
        # Rebuild from the clean template first: the layout/colour passes mutate
        # sprites in place, so re-seating a level (a RESET, a jump-to-level) has to
        # start from pristine positions/pixels -- otherwise the colour remap would
        # be applied twice and the walls would accumulate.
        idx = self._current_level_index
        self._levels[idx] = self._clean_levels[idx].clone()
        lvl = self.current_level
        self._apply_layout(lvl, self._level_rng())
        self._apply_colors(lvl)
        self._p = lvl.get_sprites_by_tag("player")[0]
        self._g = lvl.get_sprites_by_tag("goal")[0]
        self._blocks = lvl.get_sprites_by_tag("block")
        self._sync()

    def _block_on_goal(self) -> bool:
        return any(b.x == self._g.x and b.y == self._g.y for b in self._blocks)

    def _sync(self) -> None:
        self._ui.update(level_index=self.level_index, state=self._state,
                        block_on_goal=self._block_on_goal(),
                        block_color=(self._color_scheme or {}).get("block",
                                                                   BLOCK_COLOR))

    # ── the mechanic ────────────────────────────────────────────────────────
    def _tug(self) -> None:
        """Drag the block nearest the player one cell towards them, along the axis
        it is further away on (ties -> vertical), if that cell is free."""
        best, best_d = None, None
        for b in self._blocks:
            d = abs(b.x - self._p.x) + abs(b.y - self._p.y)
            if best_d is None or d < best_d:
                best, best_d = b, d
        if best is None or best_d == 0:
            return
        dx = 1 if best.x < self._p.x else (-1 if best.x > self._p.x else 0)
        dy = 1 if best.y < self._p.y else (-1 if best.y > self._p.y else 0)
        if dx != 0 and dy != 0:
            if abs(best.x - self._p.x) > abs(best.y - self._p.y):
                dy = 0
            else:
                dx = 0
        nx, ny = best.x + dx, best.y + dy
        gw, gh = self.current_level.grid_size
        if not (0 <= nx < gw and 0 <= ny < gh):
            return
        hit = self.current_level.get_sprite_at(nx, ny, ignore_collidable=True)
        if hit is None or not hit.is_collidable or hit is best:
            best.set_position(nx, ny)

    def step(self) -> None:
        # The board is displayed rotated, so the pressed SCREEN direction has to be
        # mapped back to game space before it means anything here. ACTION5 is not
        # directional (the tug direction is read off the geometry) and passes
        # through unchanged.
        action_id = self.screen_action_to_game(self.action.id)

        if action_id == GameAction.ACTION5:
            self._tug()
            self._sync()
            self.complete_action()
            return

        dx = dy = 0
        if action_id == GameAction.ACTION1:
            dy = -1
        elif action_id == GameAction.ACTION2:
            dy = 1
        elif action_id == GameAction.ACTION3:
            dx = -1
        elif action_id == GameAction.ACTION4:
            dx = 1
        else:
            self.complete_action()
            return

        gw, gh = self.current_level.grid_size
        nx, ny = self._p.x + dx, self._p.y + dy
        if 0 <= nx < gw and 0 <= ny < gh:
            hit = self.current_level.get_sprite_at(nx, ny, ignore_collidable=True)
            if hit is not None and "block" in hit.tags:
                bx, by = nx + dx, ny + dy
                if 0 <= bx < gw and 0 <= by < gh:
                    behind = self.current_level.get_sprite_at(
                        bx, by, ignore_collidable=True)
                    if behind is None or not behind.is_collidable:
                        hit.set_position(bx, by)
                        self._p.set_position(nx, ny)
            elif hit is None or not hit.is_collidable:
                self._p.set_position(nx, ny)

        # Only a WALK closes the level out -- see the module docstring.
        won = self._block_on_goal()
        if won:
            self.next_level()
        self._sync()
        self.complete_action()
