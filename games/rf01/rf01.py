"""Mirror Half: on the RIGHT half of the grid, left and right inputs are swapped.

Mechanic (unchanged from the original): four movement keys walk a 1x1 player
around the board; walls block, and stepping onto the target finishes the level.
The twist is one line in ``step``: while the player stands at ``x >= gw // 2``,
the horizontal component of the move is negated -- so "left" walks right and
"right" walks left on the right half of the board, and the mirror line has to be
inferred from watching the player move.

PER-SEED AUGMENTATION (the repo convention -- see ``utils/arc_game.py`` and the
per-seed-augmentation notes). ``Rf01(seed=s)`` makes every level a pure function
of ``(s, level_index)``:

  * DISPLAY ROTATION, owned by `AugmentedGame`: the frame is rendered rotated by
    a per-(seed, level) ``k``, and ``step`` maps the pressed SCREEN direction back
    to game space via ``screen_action_to_game``. No rotation code lives here.
    Note the interaction with the mechanic: the mirror is defined on the GAME
    x-axis, so at ``k in (1, 3)`` it is the *vertical* screen keys that get
    swapped on the (screen-)half the game calls right. That is the point -- the
    agent has to read the rule off the frames, not off a memorised key mapping.
  * LAYOUT: the grid size, the player and the target are re-drawn per
    (seed, level), and level 3's wall line is re-placed with its gap moved --
    keeping each level's stock character (see `_LEVEL_SPEC`).
  * COLOURS: player / target / background come from one EPISODE-scoped scheme
    (level-independent seed, cached), so every level of one episode looks the
    same. Wall (3) and padding (4) stay fixed, and the pool excludes the HUD's
    own colours (11, 14) so nothing on the board is confusable with the level
    dots or the tick row -- as in the rest of this family (nu01/mb01/fs02/pw01).

Every level is solvable BY CONSTRUCTION and the constructor's layouts are
additionally certified: `_reachable` is the exact same mirrored-move BFS the
training generator uses, and a sampled layout is rejected and re-drawn unless it
says the target is reachable. (An open board always is -- the mirror swaps the
two horizontal keys but never removes one, so both directions stay available on
both halves -- and the wall line always keeps exactly one gap. The check is the
ground truth that keeps a future layout edit honest.)
"""

from __future__ import annotations

import random as _random_module
from collections import deque

from arcengine import (
    Camera,
    GameAction,
    GameState,
    Level,
    RenderableUserDisplay,
    Sprite,
)

from utils.arc_game import AugmentedGame, clear_dynamic_sprites

BACKGROUND_COLOR = 5
PADDING_COLOR = 4
PLAYER_COLOR = 9
TARGET_COLOR = 11
WALL_COLOR = 3

# Board colours are drawn from this pool: no 0, no wall (3), no padding (4) and
# none of the HUD's colours (11 level-dot/tick, 14 done-dot/win-bar), so a board
# cell is never confusable with a piece of interface.
_COLOR_POOL = [1, 2, 6, 7, 8, 9, 10, 12, 13, 15]


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


def _r_ticks(frame, h, w, n, y=None):
    row = (h - 1) if y is None else y
    for i in range(max(0, min(n, 8))):
        _rp(frame, h, w, 1 + i, row, 11)


def _r_bar(frame, h, w, game_over, win):
    if not (game_over or win):
        return
    r = h - 3
    if r < 0:
        return
    c = 14 if win else 8
    for x in range(min(w, 16)):
        _rp(frame, h, w, x, r, c)


class Rf01UI(RenderableUserDisplay):
    def __init__(
        self,
        targets_remaining: int,
        level_index: int = 0,
        num_levels: int = 5,
    ) -> None:
        self._targets = targets_remaining
        self._level_index = level_index
        self._num_levels = num_levels
        self._state: GameState | None = None

    def update(
        self,
        targets_remaining: int,
        *,
        level_index: int | None = None,
        num_levels: int | None = None,
        state: GameState | None = None,
    ) -> None:
        self._targets = targets_remaining
        if level_index is not None:
            self._level_index = level_index
        if num_levels is not None:
            self._num_levels = num_levels
        if state is not None:
            self._state = state

    def render_interface(self, frame):
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return frame
        h, w = frame.shape
        _r_dots(frame, h, w, self._level_index, self._num_levels, 0)
        _r_ticks(frame, h, w, self._targets)
        go = self._state == GameState.GAME_OVER
        win = self._state == GameState.WIN
        _r_bar(frame, h, w, go, win)
        return frame


sprites = {
    "player": Sprite(
        pixels=[[PLAYER_COLOR]],
        name="player",
        layer=2,
        visible=True,
        collidable=True,
        tags=["player"],
    ),
    "target": Sprite(
        pixels=[[TARGET_COLOR]],
        name="target",
        layer=0,
        visible=True,
        collidable=False,
        tags=["target"],
    ),
    "wall": Sprite(
        pixels=[[WALL_COLOR]],
        name="wall",
        layer=1,
        visible=True,
        collidable=True,
        tags=["wall"],
    ),
}


def mk(sl, grid_size, difficulty):
    return Level(sprites=sl, grid_size=grid_size, data={"difficulty": difficulty})


# The five levels ship as their STOCK layouts (what the game was before the
# augmentation, and the fallback if a draw somehow fails); `on_set_level` rebuilds
# every sprite from the per-(seed, level) RNG.
levels = [
    mk(
        [
            sprites["player"].clone().set_position(2, 3),
            sprites["target"].clone().set_position(6, 3),
        ],
        (8, 8),
        1,
    ),
    mk(
        [
            sprites["player"].clone().set_position(0, 0),
            sprites["target"].clone().set_position(7, 7),
        ],
        (8, 8),
        2,
    ),
    mk(
        [
            sprites["player"].clone().set_position(4, 2),
            sprites["target"].clone().set_position(1, 6),
        ]
        + [sprites["wall"].clone().set_position(3, y) for y in range(8) if y != 2],
        (8, 8),
        3,
    ),
    mk(
        [
            sprites["player"].clone().set_position(5, 4),
            sprites["target"].clone().set_position(2, 4),
        ],
        (10, 8),
        4,
    ),
    mk(
        [
            sprites["player"].clone().set_position(1, 1),
            sprites["target"].clone().set_position(6, 6),
        ],
        (8, 8),
        5,
    ),
]

# Per-level augmentation spec. ``widths``/``heights`` are the grid sizes the level
# may be drawn at (all inside metadata.json's declared grid_range [8, 10]); the
# rest keeps each stock level's character:
#   row   -- player and target share a row, on opposite halves (the plain
#            "walk across the mirror line" lesson). ``start`` fixes which half
#            the player begins on: level 1 walks left->right, level 4 right->left.
#   diag  -- opposite halves AND at least ``min_dy`` rows apart.
#   wall  -- one full-height wall column with a single gap; player and target on
#            opposite sides of it, so the gap must be found and crossed.
#   free  -- opposite halves, at least ``min_dist`` Manhattan apart, otherwise
#            anywhere (the starting side is drawn too).
_LEVEL_SPEC = [
    {"kind": "row", "widths": (8, 9, 10), "heights": (8, 9, 10),
     "start": "left", "min_dx": 3},
    {"kind": "diag", "widths": (8, 9, 10), "heights": (8, 9, 10),
     "start": "left", "min_dx": 3, "min_dy": 3},
    {"kind": "wall", "widths": (8, 9, 10), "heights": (8, 9, 10)},
    {"kind": "row", "widths": (9, 10), "heights": (8, 9, 10),
     "start": "right", "min_dx": 3},
    {"kind": "free", "widths": (8, 9, 10), "heights": (8, 9, 10),
     "min_dist": 6},
]

_MAX_LAYOUT_TRIES = 200


# ── the mechanic, as a pure function (shared by the game and its layout check) ──
_DELTAS = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}


def step_cell(x: int, y: int, dx: int, dy: int, walls, gw: int, gh: int):
    """Where a move of ``(dx, dy)`` from ``(x, y)`` lands, mirror rule included.

    This IS the game's movement rule (``step`` calls it): the horizontal
    component is negated while the player stands on the right half, the move is a
    no-op off-grid or into a wall. Returns the (possibly unchanged) cell.
    """
    if x >= gw // 2:
        dx = -dx
    nx, ny = x + dx, y + dy
    if not (0 <= nx < gw and 0 <= ny < gh):
        return x, y
    if (nx, ny) in walls:
        return x, y
    return nx, ny


def _reachable(start, target, walls, gw: int, gh: int) -> bool:
    """Is ``target`` reachable from ``start`` under the mirrored move rule?"""
    seen = {start}
    q = deque([start])
    while q:
        x, y = q.popleft()
        if (x, y) == target:
            return True
        for dx, dy in _DELTAS.values():
            nxt = step_cell(x, y, dx, dy, walls, gw, gh)
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return False


# ── layout sampling ─────────────────────────────────────────────────────────
def _side_cells(gw: int, gh: int, side: str, walls=frozenset()):
    mid = gw // 2
    xs = range(0, mid) if side == "left" else range(mid, gw)
    return [(x, y) for x in xs for y in range(gh) if (x, y) not in walls]


def _sample_layout(rng, idx: int, gw: int, gh: int):
    """One candidate ``(player, target, walls)`` for level ``idx``.

    Returns ``None`` when the draw is degenerate (a grid too small for the level's
    spacing constraint), which the caller retries / falls back on.
    """
    spec = _LEVEL_SPEC[idx]
    kind = spec["kind"]

    if kind == "wall":
        wx = rng.randrange(2, gw - 2)          # keep a >=2-wide region either side
        gap_y = rng.randrange(gh)
        walls = frozenset((wx, y) for y in range(gh) if y != gap_y)
        west = [(x, y) for x in range(wx) for y in range(gh)]
        east = [(x, y) for x in range(wx + 1, gw) for y in range(gh)]
        if not west or not east:
            return None
        near, far = (west, east) if rng.random() < 0.5 else (east, west)
        return rng.choice(near), rng.choice(far), walls   # player side, target side

    start = spec.get("start") or rng.choice(("left", "right"))
    other = "right" if start == "left" else "left"
    here = _side_cells(gw, gh, start)
    there = _side_cells(gw, gh, other)
    if not here or not there:
        return None

    if kind == "row":
        y = rng.randrange(gh)
        pxs = [x for (x, yy) in here if yy == y]
        txs = [x for (x, yy) in there if yy == y]
        cand = [(px, tx) for px in pxs for tx in txs
                if abs(px - tx) >= spec["min_dx"]]
        if not cand:
            return None
        px, tx = rng.choice(cand)
        return (px, y), (tx, y), frozenset()

    if kind == "diag":
        cand = [(p, t) for p in here for t in there
                if abs(p[0] - t[0]) >= spec["min_dx"]
                and abs(p[1] - t[1]) >= spec["min_dy"]]
        if not cand:
            return None
        p, t = rng.choice(cand)
        return p, t, frozenset()

    # kind == "free"
    cand = [(p, t) for p in here for t in there
            if abs(p[0] - t[0]) + abs(p[1] - t[1]) >= spec["min_dist"]]
    if not cand:
        return None
    p, t = rng.choice(cand)
    return p, t, frozenset()


class Rf01(AugmentedGame):
    def __init__(self, seed: int | None = None) -> None:
        # FIRST: the camera's interface list holds the rotation display, and
        # super().__init__ -> set_level(0) -> on_set_level already needs the seed.
        self._init_augmentation(seed)
        self._board_seed = (seed if seed is not None
                            else _random_module.randrange(1 << 30))
        self._color_scheme: dict[str, int] | None = None
        self._ui = Rf01UI(0)
        self._mid = 0                    # the mirror line; set per level in on_set_level
        super().__init__(
            "rf01",
            levels,
            Camera(0, 0, 16, 16, BACKGROUND_COLOR, PADDING_COLOR,
                   [self._ui]),
            False,
            1,
            [1, 2, 3, 4],
        )

    # ── per-(seed, level) draws ─────────────────────────────────────────────
    def _level_rng(self, idx: int):
        return _random_module.Random(f"rf01:{self._board_seed}:{idx}")

    def _grid_for(self, idx: int) -> tuple[int, int]:
        """This level's grid size. Drawn from its own RNG stream (NOT the layout
        one) because it is consumed BEFORE ``on_set_level`` -- the engine resizes
        the camera from ``level.grid_size`` on the way in -- and must therefore not
        depend on, or shift, the layout draw."""
        spec = _LEVEL_SPEC[idx]
        rng = _random_module.Random(f"rf01-grid:{self._board_seed}:{idx}")
        return rng.choice(spec["widths"]), rng.choice(spec["heights"])

    def _ensure_color_scheme(self) -> None:
        """Draw the EPISODE-scoped colour scheme once, from a LEVEL-INDEPENDENT
        seed, so every level of one episode shares it (the repo's standing rule).
        Unseeded play still varies run to run, because ``_board_seed`` is itself a
        fresh draw there."""
        if self._color_scheme is not None:
            return
        rng = _random_module.Random(f"rf01-colors:{self._board_seed}")
        player, target, background = rng.sample(_COLOR_POOL, 3)
        self._color_scheme = {"player": player, "target": target,
                              "background": background}

    def full_reset(self) -> None:
        # A full reset starts a new episode -> re-draw the colour scheme.
        self._color_scheme = None
        super().full_reset()

    # ── engine hooks ────────────────────────────────────────────────────────
    def set_level(self, index: int) -> None:
        """Pin this level's grid size before the engine reads it.

        ``ARCBaseGame.set_level`` resizes the camera from ``level.grid_size`` and
        only THEN calls ``on_set_level``, so the per-seed size has to be installed
        on the level object here, ahead of the base call (which also draws the
        rotation -- see `AugmentedGame.set_level`)."""
        if 0 <= index < len(self._levels):
            self._levels[index]._grid_size = self._grid_for(index)
        super().set_level(index)

    def on_set_level(self, level: Level) -> None:
        # Every sprite on the board is (re)built here, so drop whatever a previous
        # seating of this same persistent Level object left behind -- otherwise a
        # RESET or a jump-to-level stacks a ghost player on the start cell.
        clear_dynamic_sprites(level, tags=["player", "target", "wall"])
        idx = self.level_index
        gw, gh = level.grid_size
        self._mid = gw // 2
        self._ensure_color_scheme()
        scheme = self._color_scheme
        self.camera.background = scheme["background"]

        player, target, walls = self._draw_layout(idx, gw, gh)
        for cell in sorted(walls):
            level.add_sprite(
                sprites["wall"].clone().set_position(*cell))
        tgt = sprites["target"].clone()
        tgt.color_remap(TARGET_COLOR, scheme["target"])
        level.add_sprite(tgt.set_position(*target))
        ply = sprites["player"].clone()
        ply.color_remap(PLAYER_COLOR, scheme["player"])
        level.add_sprite(ply.set_position(*player))

        self._player = level.get_sprites_by_tag("player")[0]
        self._targets = level.get_sprites_by_tag("target")
        self._ui.update(
            len(self._targets),
            level_index=self.level_index,
            num_levels=len(levels),
            state=self._state,
        )

    def _draw_layout(self, idx: int, gw: int, gh: int):
        """The per-(seed, level) layout, certified reachable. Falls back to the
        level's stock layout (clamped to the grid) if every draw is rejected."""
        rng = self._level_rng(idx)
        for _ in range(_MAX_LAYOUT_TRIES):
            cand = _sample_layout(rng, idx, gw, gh)
            if cand is None:
                continue
            player, target, walls = cand
            if player == target or player in walls or target in walls:
                continue
            if _reachable(player, target, walls, gw, gh):
                return player, target, walls
        return self._stock_layout(idx, gw, gh)

    @staticmethod
    def _stock_layout(idx: int, gw: int, gh: int):
        clean = levels[idx]

        def clamp(s):
            return (min(s.x, gw - 1), min(s.y, gh - 1))

        player = clamp(clean.get_sprites_by_tag("player")[0])
        target = clamp(clean.get_sprites_by_tag("target")[0])
        walls = frozenset(clamp(s) for s in clean.get_sprites_by_tag("wall"))
        walls -= {player, target}
        return player, target, walls

    # ── play ────────────────────────────────────────────────────────────────
    def step(self) -> None:
        # The board is displayed rotated, so the pressed SCREEN direction has to be
        # mapped back to game space before it means anything here.
        action_id = self.screen_action_to_game(self.action.id)
        dx, dy = _DELTAS.get(action_id, (0, 0))

        if dx == 0 and dy == 0:
            self.complete_action()
            return

        grid_w, grid_h = self.current_level.grid_size
        walls = frozenset((s.x, s.y)
                          for s in self.current_level.get_sprites_by_tag("wall"))
        # THE MECHANIC: on the right half (x >= gw // 2) the horizontal component
        # is negated, so left walks right and right walks left.
        new_x, new_y = step_cell(self._player.x, self._player.y, dx, dy,
                                 walls, grid_w, grid_h)
        if (new_x, new_y) != (self._player.x, self._player.y):
            self._player.set_position(new_x, new_y)

            for t in self._targets:
                if self._player.x == t.x and self._player.y == t.y:
                    self.next_level()
                    break

        self._ui.update(
            len(self._targets),
            level_index=self.level_index,
            num_levels=len(levels),
            state=self._state,
        )
        self.complete_action()
