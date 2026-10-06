"""Weighted plates: two plates must both have a crate on them; then reach the goal.

A 10x10 push-only sokoban with two crate/plate pairs and an exit: walking into a
crate pushes it one cell (if the cell behind it is free), both plates have to end
up covered, and only THEN does standing on the goal finish the level.

PER-SEED AUGMENTATION (the repo convention -- see utils/arc_game.py and the
per-seed-augmentation notes). ``Pw01(seed=s)`` makes every level a pure function
of ``(s, level_index)``:

  * DISPLAY ROTATION, owned by ``AugmentedGame``: the frame is rendered rotated by
    a per-(seed, level) k, and ``step`` maps the pressed SCREEN direction back to
    game space via ``screen_action_to_game``. No rotation code lives here.
  * LAYOUT: the two crate/plate pairs, the wall line (level 3), the player and the
    goal are re-placed at random per (seed, level), keeping each level's stock
    character (wall budget, push length ramp) -- see ``_LEVEL_PARAMS``.
  * COLOURS: player / crate / plate colours are drawn from one EPISODE-scoped
    scheme (level-independent seed, cached), so every level of one episode looks
    the same. Goal (14), wall (3), background (5) and padding (4) stay fixed, as
    in the rest of this game family (nu01/mb01/fs02).

SOLVABILITY CERTIFICATE. Random sokoban layouts are easily unsolvable, and a full
state-space search per level would cost ~a second. So instead of checking a random
layout, every sampled layout carries an explicit plan skeleton: each pair is a
STRAIGHT corridor -- crate at ``c``, plate at ``c + dist*d``, with the push-start
cell ``c - d`` and every cell from ``c - d`` through the plate free of walls and of
the other pair -- so that pair is solved by walking to ``c - d`` and pushing
``dist`` times. A layout is accepted iff, for at least one of the two pair orders,
the player can walk to the first push-start, then (with crate 1 parked) to the
second, and finally reach the goal (with both crates parked). Those three checks
are 100-cell flood fills, so sampling stays cheap and an accepted layout is
solvable BY CONSTRUCTION. The training generator's search then finds the true
optimum, which is usually shorter than this skeleton.

Everything visible at t=0 is distinct: the player, both crates, both plates and
the goal occupy six different cells, so nothing starts hidden under anything else.
Sprite layers make that hold during play too (player above crate above plate/goal),
so a crate parked on a plate reads as a crate and the player never disappears
under a floor marker.
"""

import random as _random_module

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
PLAYER_COLOR = 9
BLOCK_COLOR = 15
PLATE_COLOR = 11
GOAL_COLOR = 14
WALL_COLOR = 3

# Colours drawn for the player / crate / plate. Excludes the fixed goal (14),
# wall (3), background (5), padding (4) and 0; the three are sampled mutually
# distinct so nothing hides against anything else.
_COLOR_POOL = [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 15]

# Render/collision layers: floor markers at the bottom, then crates, then the
# player. ``get_sprite_at(..., ignore_collidable=True)`` returns the HIGHEST layer
# at a cell, so a crate parked on a plate is still seen as a crate by ``step``
# (which is also why the crate, not the plate, is what you see there).
_LAYER_FLOOR = 0
_LAYER_BLOCK = 1
_LAYER_PLAYER = 2


class Pw01UI(RenderableUserDisplay):
    """One pixel per COVERED plate, bottom-left, in the plate colour."""

    def __init__(self, ok: int = 0, color: int = PLATE_COLOR) -> None:
        self._ok = ok
        self._color = color

    def update(self, ok: int, color: int | None = None) -> None:
        self._ok = ok
        if color is not None:
            self._color = color

    def render_interface(self, frame):
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return frame
        h, _w = frame.shape
        for i in range(min(self._ok, 5)):
            frame[h - 2, 1 + i] = self._color
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
    "block": Sprite(
        pixels=[[BLOCK_COLOR]],
        name="block",
        visible=True,
        collidable=True,
        tags=["block"],
        layer=_LAYER_BLOCK,
    ),
    "plate": Sprite(
        pixels=[[PLATE_COLOR]],
        name="plate",
        visible=True,
        collidable=False,
        tags=["plate"],
        layer=_LAYER_FLOOR,
    ),
    "goal": Sprite(
        pixels=[[GOAL_COLOR]],
        name="goal",
        visible=True,
        collidable=False,
        tags=["goal"],
        layer=_LAYER_FLOOR,
    ),
    "wall": Sprite(
        pixels=[[WALL_COLOR]],
        name="wall",
        visible=True,
        collidable=True,
        tags=["wall"],
        layer=_LAYER_BLOCK,
    ),
}


def mk(sl: list, d: int) -> Level:
    return Level(sprites=sl, grid_size=(10, 10), data={"difficulty": d})


levels = [
    mk(
        [
            sprites["player"].clone().set_position(0, 5),
            sprites["block"].clone().set_position(2, 4),
            sprites["block"].clone().set_position(2, 6),
            sprites["plate"].clone().set_position(4, 4),
            sprites["plate"].clone().set_position(4, 6),
            sprites["goal"].clone().set_position(8, 5),
        ],
        1,
    ),
    mk(
        [
            sprites["player"].clone().set_position(1, 1),
            sprites["block"].clone().set_position(3, 3),
            sprites["block"].clone().set_position(3, 5),
            sprites["plate"].clone().set_position(5, 3),
            sprites["plate"].clone().set_position(5, 5),
            sprites["goal"].clone().set_position(8, 8),
        ],
        2,
    ),
    mk(
        [
            sprites["player"].clone().set_position(0, 0),
            sprites["block"].clone().set_position(1, 2),
            sprites["block"].clone().set_position(2, 1),
            sprites["plate"].clone().set_position(4, 2),
            sprites["plate"].clone().set_position(2, 4),
            sprites["goal"].clone().set_position(9, 9),
        ]
        + [sprites["wall"].clone().set_position(6, y) for y in range(10) if y not in (4, 5)],
        3,
    ),
    mk(
        [
            sprites["player"].clone().set_position(2, 5),
            sprites["block"].clone().set_position(3, 4),
            sprites["block"].clone().set_position(3, 6),
            sprites["plate"].clone().set_position(5, 4),
            sprites["plate"].clone().set_position(5, 6),
            sprites["goal"].clone().set_position(8, 5),
        ],
        4,
    ),
    mk(
        [
            sprites["player"].clone().set_position(1, 5),
            sprites["block"].clone().set_position(2, 3),
            sprites["block"].clone().set_position(2, 7),
            sprites["plate"].clone().set_position(6, 3),
            sprites["plate"].clone().set_position(6, 7),
            sprites["goal"].clone().set_position(8, 5),
        ],
        5,
    ),
]

# Per-level augmentation budget, index-aligned with ``levels``:
#   wall_mode  -- "none", or "line" (a full row/column with a 2-cell doorway,
#                 which is exactly the 8 wall sprites level 3's template holds)
#   push_range -- inclusive range of cells each crate has to be pushed
#   goal_gap   -- minimum Manhattan distance from the player start to the goal
# The stock levels only give level 3 walls, so only level 3 gets a wall line; the
# ramp on the others is push length, matching their stock layouts (2, 2, 3, 3, 4).
_LEVEL_PARAMS = [
    {"wall_mode": "none", "push_range": (2, 2), "goal_gap": 4},
    {"wall_mode": "none", "push_range": (2, 3), "goal_gap": 5},
    {"wall_mode": "line", "push_range": (2, 3), "goal_gap": 6},
    {"wall_mode": "none", "push_range": (3, 4), "goal_gap": 6},
    {"wall_mode": "none", "push_range": (4, 6), "goal_gap": 6},
]

_DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
_LAYOUT_ATTEMPTS = 400


def _reachable(start, target, blockers, gw: int, gh: int) -> bool:
    """Can the player walk ``start`` -> ``target`` treating ``blockers`` (walls and
    crates) as impassable? Plain 4-neighbour flood fill over <=100 cells: pushing
    is never needed for these walks, so a crate is simply a wall here."""
    if start == target:
        return True
    if start in blockers or target in blockers:
        return False
    seen = {start}
    stack = [start]
    while stack:
        x, y = stack.pop()
        for dx, dy in _DIRS:
            nxt = (x + dx, y + dy)
            if not (0 <= nxt[0] < gw and 0 <= nxt[1] < gh):
                continue
            if nxt in blockers or nxt in seen:
                continue
            if nxt == target:
                return True
            seen.add(nxt)
            stack.append(nxt)
    return False


def _certified(player, pairs, goal, walls, gw: int, gh: int) -> bool:
    """Is the plan skeleton walkable for at least one pair order? (See the module
    docstring.) ``pairs[i]`` is ``(crate, plate, push_start, dir)``."""
    for order in ((0, 1), (1, 0)):
        first, second = pairs[order[0]], pairs[order[1]]
        crates = {first[0], second[0]}
        # 1. walk to the first push-start, both crates still at their start cells
        if not _reachable(player, first[2], walls | crates, gw, gh):
            continue
        # 2. after pushing crate 1 the player trails it, standing one cell behind
        #    its plate; walk from there to the second push-start
        after_first = (first[1][0] - first[3][0], first[1][1] - first[3][1])
        if not _reachable(after_first, second[2],
                          walls | {first[1], second[0]}, gw, gh):
            continue
        # 3. with both crates parked on their plates, walk to the goal
        after_second = (second[1][0] - second[3][0], second[1][1] - second[3][1])
        if not _reachable(after_second, goal,
                          walls | {first[1], second[1]}, gw, gh):
            continue
        return True
    return False


def _sample_walls(rng, mode: str, gw: int, gh: int) -> list:
    """The wall cells for this level, matching the template's wall budget."""
    if mode != "line":
        return []
    if rng.random() < 0.5:                  # a vertical wall at column `line`
        line = rng.randint(2, gw - 3)       # keep >=2 columns of play on each side
        door = rng.randint(0, gh - 2)       # a 2-cell doorway at `door`, `door+1`
        return [(line, y) for y in range(gh) if y not in (door, door + 1)]
    line = rng.randint(2, gh - 3)           # a horizontal wall at row `line`
    door = rng.randint(0, gw - 2)
    return [(x, line) for x in range(gw) if x not in (door, door + 1)]


def _sample_pair(rng, walls, taken, lo: int, hi: int, gw: int, gh: int):
    """One crate/plate pair as a straight push corridor, or None.

    Returns ``((crate, plate, push_start, dir), footprint)`` where ``footprint``
    is every cell the pair needs kept clear of walls and of the other pair: the
    push-start, the crate, and every cell up to and including the plate."""
    for _ in range(40):
        dx, dy = rng.choice(_DIRS)
        dist = rng.randint(lo, hi)
        cx, cy = rng.randrange(gw), rng.randrange(gh)
        start = (cx - dx, cy - dy)                    # where the player pushes from
        end = (cx + dist * dx, cy + dist * dy)       # the plate
        if not (0 <= start[0] < gw and 0 <= start[1] < gh):
            continue
        if not (0 <= end[0] < gw and 0 <= end[1] < gh):
            continue
        footprint = {start}
        footprint.update((cx + i * dx, cy + i * dy) for i in range(dist + 1))
        if footprint & walls or footprint & taken:
            continue
        return ((cx, cy), end, start, (dx, dy)), footprint
    return None


def _sample_layout(rng, params: dict, gw: int, gh: int):
    """A solvable layout, or None if sampling failed (caller keeps the stock one).

    Returns ``{"walls", "crates", "plates", "player", "goal"}`` in grid cells."""
    lo, hi = params["push_range"]
    for _ in range(_LAYOUT_ATTEMPTS):
        walls = set(_sample_walls(rng, params["wall_mode"], gw, gh))
        pairs, taken = [], set()
        for _ in range(2):
            got = _sample_pair(rng, walls, taken, lo, hi, gw, gh)
            if got is None:
                break
            pair, footprint = got
            pairs.append(pair)
            taken |= footprint
        if len(pairs) != 2:
            continue

        # The six meaningful cells stay distinct so all of them are visible at
        # t=0: nothing starts under a crate or under the player.
        occupied = {p[0] for p in pairs} | {p[1] for p in pairs}
        free = [(x, y) for x in range(gw) for y in range(gh)
                if (x, y) not in walls and (x, y) not in occupied]
        if len(free) < 2:
            continue
        player = rng.choice(free)
        gap = params["goal_gap"]
        far = [c for c in free
               if c != player and abs(c[0] - player[0]) + abs(c[1] - player[1]) >= gap]
        if not far:
            continue
        goal = rng.choice(far)
        if not _certified(player, pairs, goal, walls, gw, gh):
            continue
        return {"walls": sorted(walls),
                "crates": [p[0] for p in pairs],
                "plates": [p[1] for p in pairs],
                "player": player,
                "goal": goal}
    return None


class Pw01(AugmentedGame):
    def __init__(self, seed: int | None = None) -> None:
        # FIRST: the base draws this level's rotation and turns the BOARD by it in
        # set_level, before the engine seats the level.
        self._init_augmentation(seed)
        # Episode-scoped colour scheme (level-INDEPENDENT), drawn once and cached so
        # every level of one episode is coloured identically.
        self._color_scheme: dict | None = None
        self._ui = Pw01UI(0, PLATE_COLOR)
        super().__init__(
            "pw01",
            levels,
            Camera(0, 0, 16, 16, BACKGROUND_COLOR, PADDING_COLOR,
                   [self._ui]),
            False,
            1,
            [1, 2, 3, 4],
        )

    # ── per-seed augmentation ───────────────────────────────────────────────
    def _level_rng(self) -> "_random_module.Random":
        if self._seed_value is None:                  # interactive play: fresh board
            return _random_module.Random()
        return _random_module.Random(
            f"pw01:{self._seed_value}:{self._current_level_index}")

    def _ensure_color_scheme(self) -> None:
        if self._color_scheme is not None:
            return
        rng = (_random_module.Random(f"pw01-colors:{self._seed_value}")
               if self._seed_value is not None else _random_module.Random())
        player, block, plate = rng.sample(_COLOR_POOL, 3)
        self._color_scheme = {"player": player, "block": block, "plate": plate}

    def _apply_layout(self, level: Level, rng) -> None:
        gw, gh = level.grid_size
        params = _LEVEL_PARAMS[self._current_level_index]
        layout = _sample_layout(rng, params, gw, gh)
        if layout is None:                            # keep the stock layout
            return
        walls = level.get_sprites_by_tag("wall")
        for sprite, cell in zip(walls, layout["walls"]):
            sprite.set_position(*cell)
        for sprite, cell in zip(level.get_sprites_by_tag("block"), layout["crates"]):
            sprite.set_position(*cell)
        for sprite, cell in zip(level.get_sprites_by_tag("plate"), layout["plates"]):
            sprite.set_position(*cell)
        level.get_sprites_by_tag("player")[0].set_position(*layout["player"])
        level.get_sprites_by_tag("goal")[0].set_position(*layout["goal"])

    def _apply_colors(self, level: Level) -> None:
        self._ensure_color_scheme()
        scheme = self._color_scheme
        for sprite in level.get_sprites_by_tag("player"):
            sprite.color_remap(PLAYER_COLOR, scheme["player"])
        for sprite in level.get_sprites_by_tag("block"):
            sprite.color_remap(BLOCK_COLOR, scheme["block"])
        for sprite in level.get_sprites_by_tag("plate"):
            sprite.color_remap(PLATE_COLOR, scheme["plate"])

    def full_reset(self) -> None:
        # A full reset starts a new episode -> a new colour scheme.
        self._color_scheme = None
        super().full_reset()

    # ── engine hooks ────────────────────────────────────────────────────────
    def on_set_level(self, level: Level) -> None:
        # Rebuild from the clean template first: ``_apply_layout`` /
        # ``_apply_colors`` mutate sprites in place, so re-seating a level (a RESET,
        # a jump-to-level) must start from pristine positions/pixels -- otherwise
        # the colour remap would be applied twice and never match again.
        idx = self._current_level_index
        self._levels[idx] = self._clean_levels[idx].clone()
        lvl = self.current_level
        self._apply_layout(lvl, self._level_rng())
        self._apply_colors(lvl)
        self._player = lvl.get_sprites_by_tag("player")[0]
        self._goal = lvl.get_sprites_by_tag("goal")[0]
        self._plates = list(lvl.get_sprites_by_tag("plate"))
        self._blocks = list(lvl.get_sprites_by_tag("block"))
        self._sync()

    def _covered(self) -> int:
        """How many plates currently hold a crate."""
        cells = {(b.x, b.y) for b in self._blocks}
        return sum(1 for p in self._plates if (p.x, p.y) in cells)

    def _both_plated(self) -> bool:
        return self._covered() == len(self._plates)

    def _sync(self) -> None:
        self._ui.update(self._covered(),
                        (self._color_scheme or {}).get("plate", PLATE_COLOR))

    def step(self) -> None:
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
        gw, gh = self.current_level.grid_size
        if not (0 <= nx < gw and 0 <= ny < gh):
            self.complete_action()
            return

        sp = self.current_level.get_sprite_at(nx, ny, ignore_collidable=True)
        if sp and "wall" in sp.tags:
            self.complete_action()
            return

        if sp and "block" in sp.tags:
            bx, by = nx + dx, ny + dy
            if not (0 <= bx < gw and 0 <= by < gh):
                self.complete_action()
                return
            behind = self.current_level.get_sprite_at(bx, by, ignore_collidable=True)
            if behind and ("block" in behind.tags or "wall" in behind.tags):
                self.complete_action()
                return
            sp.set_position(bx, by)
            self._player.set_position(nx, ny)
        elif not sp or not sp.is_collidable:
            self._player.set_position(nx, ny)

        self._sync()

        if self._both_plated() and self._player.x == self._goal.x and self._player.y == self._goal.y:
            self.next_level()

        self.complete_action()
