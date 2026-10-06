"""Sokoban Push: walk into a block to shove it one cell; cover every target pad.

A push-only sokoban on an 8x8 (or 12x12) board. Walking into a block moves it one
cell in the same direction if the cell behind it is free of walls and of other
blocks; walls stop the player outright. The level is won the moment EVERY block
sits on a target pad, and a covered block is recoloured so the progress is visible
on the board itself.

PER-SEED AUGMENTATION (the repo convention -- see utils/arc_game.py and the
per-seed-augmentation notes). ``Sk01(seed=s)`` makes every level a pure function
of ``(s, level_index)``:

  * DISPLAY ROTATION, owned by ``AugmentedGame``: the frame is rendered rotated by
    a per-(seed, level) k, and ``step`` maps the pressed SCREEN direction back to
    game space via ``screen_action_to_game``. No rotation code lives here.
  * LAYOUT: the block/target pairs, the walls and the player start are re-placed at
    random per (seed, level), keeping each level's stock character -- board size,
    block count, wall budget and wall style -- see ``_LEVEL_PARAMS``.
  * COLOURS: player / block / target colours come from one EPISODE-scoped scheme
    (level-independent seed, cached), so every level of one episode looks the same.
    The COVERED colour (14), walls (3), background (5) and padding (4) stay fixed,
    as in the rest of this game family (pw01/nu01/mb01/fs02).

SOLVABILITY CERTIFICATE. A random sokoban board is usually unsolvable and a full
state-space check would cost ~a second per level, so -- exactly as in pw01 -- every
sampled layout instead ships a plan skeleton: each pair is a STRAIGHT corridor
(block at ``c``, target at ``c + dist*d``, push-start ``c - d``, every cell from the
push-start through the target free of walls and of the other pairs). A layout is
accepted iff for at least ONE order of the pairs the player can walk to the first
push-start, then -- with that block parked on its target -- to the second, and so
on. Those walks are 64/144-cell flood fills, so sampling stays cheap and an accepted
layout is solvable BY CONSTRUCTION. The training generator's search then finds the
true optimum, which is usually shorter than this skeleton.

Everything meaningful is visible at t=0: no block starts on a target and the player
starts off every corridor. Sprite layers keep that true during play (player above
block above target), so a block parked on a pad reads as a covered block rather
than vanishing under the pad.
"""

import itertools as _itertools
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
TARGET_COLOR = 11
DONE_COLOR = 14
WALL_COLOR = 3

# Colours drawn for the player / block / target. Excludes the fixed covered-block
# colour (14), wall (3), background (5), padding (4) and 0; the three are sampled
# mutually distinct so nothing hides against anything else.
_COLOR_POOL = [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 15]

# Render/collision layers: pads on the floor, then blocks, then the player.
# ``Camera.render`` draws LOW layers first and ``get_sprite_at(...,
# ignore_collidable=True)`` returns the HIGHEST layer at a cell, so with explicit
# layers a block parked on a pad both renders as a block and is still seen as a
# block by the push logic. (With every sprite on layer 0 the engine falls back to
# insertion order, which is the opposite way round for rendering: the pad would
# paint over the block and the player would disappear under any pad it stepped on.)
_LAYER_FLOOR = 0
_LAYER_BLOCK = 1
_LAYER_PLAYER = 2


class Sk01UI(RenderableUserDisplay):
    """One pixel per COVERED target, bottom-left, in the covered-block colour."""

    def __init__(self, covered: int = 0) -> None:
        self._covered = covered

    def update(self, covered: int) -> None:
        self._covered = covered

    def render_interface(self, frame):
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return frame
        h, _w = frame.shape
        for i in range(min(self._covered, 5)):
            frame[h - 2, 1 + i] = DONE_COLOR
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
    "target": Sprite(
        pixels=[[TARGET_COLOR]],
        name="target",
        visible=True,
        collidable=False,
        tags=["target"],
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


def make_wall_level(
    grid_size, player_pos, block_positions, target_positions, wall_coords, difficulty
):
    sprite_list = [
        sprites["player"].clone().set_position(*player_pos),
    ]
    for bp in block_positions:
        sprite_list.append(sprites["block"].clone().set_position(*bp))
    for tp in target_positions:
        sprite_list.append(sprites["target"].clone().set_position(*tp))
    for wp in wall_coords:
        sprite_list.append(sprites["wall"].clone().set_position(*wp))
    return Level(
        sprites=sprite_list,
        grid_size=grid_size,
        data={"difficulty": difficulty, "step_limit": 80 + difficulty * 30},
    )


# Five levels: size / block count / wall blockers ramp up. These are the TEMPLATES
# -- they fix each level's board size, block count and wall budget, and are also the
# fallback layout if sampling ever fails. BFS-verified solvable (optimal 9 / 13 / 23
# / 45 / 63 moves, all well inside their step limits).
levels = [
    # L1 -- tutorial: one block, open floor
    make_wall_level((8, 8), (1, 1), [(3, 3)], [(5, 5)], [], 1),
    # L2 -- one block, corner wall posts (forces routing, block stays movable)
    make_wall_level(
        (8, 8),
        (1, 1),
        [(3, 3)],
        [(5, 5)],
        [(2, 2), (4, 2), (4, 4), (2, 4)],
        2,
    ),
    # L3 -- two blocks + scattered blockers
    make_wall_level(
        (8, 8),
        (1, 1),
        [(2, 4), (5, 2)],
        [(6, 6), (6, 4)],
        [(4, 3), (4, 4), (4, 5), (3, 5), (5, 5)],
        3,
    ),
    # L4 -- larger grid, two blocks, vertical barrier with a single gap
    make_wall_level(
        (12, 12),
        (1, 1),
        [(3, 4), (4, 3)],
        [(9, 9), (10, 8)],
        [(6, y) for y in range(12) if y != 5],
        4,
    ),
    # L5 -- three blocks on 8x8 + split corridor (hardest routing)
    make_wall_level(
        (8, 8),
        (0, 0),
        [(1, 1), (2, 2), (3, 1)],
        [(5, 5), (6, 6), (5, 6)],
        [(4, y) for y in range(8) if y != 3],
        5,
    ),
]

# Per-level augmentation budget, index-aligned with ``levels``:
#   wall_mode  -- "none"  : no walls (the template has none either)
#                 "posts" : isolated single-cell blockers, the template's count
#                 "line"  : a full row/column with a ONE-cell doorway, which is
#                           exactly the wall budget levels 4 and 5 carry
#   push_range -- inclusive range of cells each block has to be pushed
# Board size, block count and wall COUNT are read off the template, so this table
# only picks the wall STYLE and the push-length ramp.
_LEVEL_PARAMS = [
    {"wall_mode": "none", "push_range": (2, 4)},
    {"wall_mode": "posts", "push_range": (2, 4)},
    {"wall_mode": "posts", "push_range": (2, 4)},
    {"wall_mode": "line", "push_range": (2, 5)},
    {"wall_mode": "line", "push_range": (2, 3)},
]

_DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
_LAYOUT_ATTEMPTS = 400


def _reachable(start, target, blockers, gw: int, gh: int) -> bool:
    """Can the player walk ``start`` -> ``target`` treating ``blockers`` (walls and
    parked blocks) as impassable? A plain 4-neighbour flood fill over <=144 cells:
    the certificate's walks never need a push, so a block is simply a wall here."""
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


def _certified(player, pairs, walls, gw: int, gh: int) -> bool:
    """Is the plan skeleton walkable for at least one order of the pairs? (See the
    module docstring.) ``pairs[i]`` is ``(block, target, push_start, dir)``.

    Pairs are solved one at a time: walk to the next push-start with every
    still-unparked block and every already-parked one treated as an obstacle, then
    push that block the length of its corridor -- which leaves the player one cell
    short of the target, where the next walk begins."""
    starts = {p[0] for p in pairs}
    for order in _itertools.permutations(range(len(pairs))):
        at = player
        parked: set = set()
        pending = set(starts)
        ok = True
        for i in order:
            block, target, push_start, d = pairs[i]
            # The block being fetched is STILL on its start cell during this walk,
            # so it stays in the obstacle set until after the push.
            if not _reachable(at, push_start, walls | parked | pending, gw, gh):
                ok = False
                break
            pending.discard(block)
            parked.add(target)
            at = (target[0] - d[0], target[1] - d[1])   # trailing the pushed block
        if ok:
            return True
    return False


def _sample_walls(rng, mode: str, count: int, gw: int, gh: int) -> list:
    """``count`` wall cells in the style ``mode``, or [] when the level has none.

    The count comes from the template (the level owns exactly that many wall
    sprites), so each mode has to hit it exactly."""
    if count == 0 or mode == "none":
        return []
    if mode == "line":
        # A full row or column minus a ONE-cell doorway -- so the line spans
        # ``count + 1`` cells and the level's wall budget fixes its orientation.
        if count + 1 == gh and rng.random() < 0.5:
            line = rng.randint(2, gw - 3)          # vertical: >=2 columns each side
            door = rng.randrange(gh)
            return [(line, y) for y in range(gh) if y != door]
        if count + 1 == gw:
            line = rng.randint(2, gh - 3)          # horizontal
            door = rng.randrange(gw)
            return [(x, line) for x in range(gw) if x != door]
        return []
    # "posts": isolated single cells, never orthogonally adjacent to each other, so
    # they stay blockers to route around rather than forming pockets.
    cells = [(x, y) for x in range(gw) for y in range(gh)]
    rng.shuffle(cells)
    out: list = []
    for c in cells:
        if len(out) == count:
            break
        if any(abs(c[0] - o[0]) + abs(c[1] - o[1]) <= 1 for o in out):
            continue
        out.append(c)
    return out if len(out) == count else []


def _sample_pair(rng, walls, taken, lo: int, hi: int, gw: int, gh: int):
    """One block/target pair as a straight push corridor, or None.

    Returns ``((block, target, push_start, dir), footprint)`` where ``footprint`` is
    every cell the pair needs kept clear of walls and of the other pairs: the
    push-start, the block, and every cell up to and including the target."""
    for _ in range(40):
        dx, dy = rng.choice(_DIRS)
        dist = rng.randint(lo, hi)
        cx, cy = rng.randrange(gw), rng.randrange(gh)
        start = (cx - dx, cy - dy)                    # where the player pushes from
        end = (cx + dist * dx, cy + dist * dy)        # the target pad
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


def _sample_layout(rng, params: dict, n_blocks: int, n_walls: int, gw: int, gh: int):
    """A solvable layout, or None if sampling failed (caller keeps the template).

    Returns ``{"walls", "blocks", "targets", "player"}`` in grid cells."""
    lo, hi = params["push_range"]
    for _ in range(_LAYOUT_ATTEMPTS):
        wall_list = _sample_walls(rng, params["wall_mode"], n_walls, gw, gh)
        if len(wall_list) != n_walls:
            continue
        walls = set(wall_list)
        pairs, taken = [], set()
        for _ in range(n_blocks):
            got = _sample_pair(rng, walls, taken, lo, hi, gw, gh)
            if got is None:
                break
            pair, footprint = got
            pairs.append(pair)
            taken |= footprint
        if len(pairs) != n_blocks:
            continue

        # The player starts off every corridor, so no block starts on a pad and
        # nothing is hidden under anything else at t=0.
        free = [(x, y) for x in range(gw) for y in range(gh)
                if (x, y) not in walls and (x, y) not in taken]
        if not free:
            continue
        player = rng.choice(free)
        if not _certified(player, pairs, walls, gw, gh):
            continue
        return {"walls": sorted(walls),
                "blocks": [p[0] for p in pairs],
                "targets": [p[1] for p in pairs],
                "player": player}
    return None


class Sk01(AugmentedGame):
    """Push blocks onto target pads. A covered block turns green; cover them all."""

    def __init__(self, seed: int | None = None) -> None:
        # FIRST: the base draws this level's rotation and turns the BOARD by it in
        # set_level, before the engine seats the level.
        self._init_augmentation(seed)
        # Episode-scoped colour scheme (level-INDEPENDENT), drawn once and cached so
        # every level of one episode is coloured identically.
        self._color_scheme: dict | None = None
        self._ui = Sk01UI(0)
        super().__init__(
            "sk01",
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
            f"sk01:{self._seed_value}:{self._current_level_index}")

    def _ensure_color_scheme(self) -> None:
        if self._color_scheme is not None:
            return
        rng = (_random_module.Random(f"sk01-colors:{self._seed_value}")
               if self._seed_value is not None else _random_module.Random())
        player, block, target = rng.sample(_COLOR_POOL, 3)
        self._color_scheme = {"player": player, "block": block, "target": target}

    def _apply_layout(self, level: Level, rng) -> None:
        gw, gh = level.grid_size
        params = _LEVEL_PARAMS[self._current_level_index]
        walls = level.get_sprites_by_tag("wall")
        blocks = level.get_sprites_by_tag("block")
        targets = level.get_sprites_by_tag("target")
        layout = _sample_layout(rng, params, len(blocks), len(walls), gw, gh)
        if layout is None:                            # keep the template layout
            return
        for sprite, cell in zip(walls, layout["walls"]):
            sprite.set_position(*cell)
        for sprite, cell in zip(blocks, layout["blocks"]):
            sprite.set_position(*cell)
        for sprite, cell in zip(targets, layout["targets"]):
            sprite.set_position(*cell)
        level.get_sprites_by_tag("player")[0].set_position(*layout["player"])

    def _apply_colors(self, level: Level) -> None:
        self._ensure_color_scheme()
        scheme = self._color_scheme
        for sprite in level.get_sprites_by_tag("player"):
            sprite.color_remap(PLAYER_COLOR, scheme["player"])
        for sprite in level.get_sprites_by_tag("target"):
            sprite.color_remap(TARGET_COLOR, scheme["target"])
        # Blocks are recoloured live by ``_sync`` (covered vs not), so their base
        # colour is only recorded here.

    def full_reset(self) -> None:
        # A full reset starts a new episode -> a new colour scheme.
        self._color_scheme = None
        super().full_reset()

    # ── engine hooks ────────────────────────────────────────────────────────
    def on_set_level(self, level: Level) -> None:
        # Rebuild from the clean template first: ``_apply_layout`` / ``_apply_colors``
        # and the live block recolouring all mutate sprites in place, so re-seating a
        # level (a RESET, a jump-to-level) must start from pristine positions and
        # pixels -- otherwise a colour remap would be applied twice and never match.
        idx = self._current_level_index
        self._levels[idx] = self._clean_levels[idx].clone()
        lvl = self.current_level
        self._apply_layout(lvl, self._level_rng())
        self._apply_colors(lvl)
        self._player = lvl.get_sprites_by_tag("player")[0]
        self._blocks = list(lvl.get_sprites_by_tag("block"))
        self._targets = list(lvl.get_sprites_by_tag("target"))
        self._step_limit = level.get_data("step_limit")
        self._steps = 0
        self._sync()

    @property
    def steps_remaining(self) -> int:
        """Moves left before the level is lost. Named for the generator harness:
        ``BaseSolver._step_counter_probe`` finds it and lifts step exhaustion during
        data generation, so an exploration prefix can never cost a trajectory."""
        return max(0, self._step_limit - self._steps)

    def _covered(self) -> int:
        """How many blocks currently sit on a target pad."""
        pads = {(t.x, t.y) for t in self._targets}
        return sum(1 for b in self._blocks if (b.x, b.y) in pads)

    def _sync(self) -> None:
        """Recolour every block for its CURRENT cell and refresh the HUD.

        The colour is a pure function of the live board -- a block pushed OFF a pad
        goes back to the plain block colour -- so what the frame shows is exactly
        what the win check reads. (The colour is the only place a block's progress is
        visible, so letting it go stale would make the board lie.)"""
        self._ensure_color_scheme()
        pads = {(t.x, t.y) for t in self._targets}
        base = self._color_scheme["block"]
        for b in self._blocks:
            b.color_remap(None, DONE_COLOR if (b.x, b.y) in pads else base)
        self._ui.update(self._covered())

    def step(self) -> None:
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

        new_x = self._player.x + dx
        new_y = self._player.y + dy

        grid_w, grid_h = self.current_level.grid_size
        if not (0 <= new_x < grid_w and 0 <= new_y < grid_h):
            self.complete_action()
            return

        sprite = self.current_level.get_sprite_at(new_x, new_y, ignore_collidable=True)

        if sprite and "wall" in sprite.tags:
            self.complete_action()
            return

        if sprite and "block" in sprite.tags:
            block_new_x = new_x + dx
            block_new_y = new_y + dy
            if not (0 <= block_new_x < grid_w and 0 <= block_new_y < grid_h):
                self.complete_action()
                return
            block_behind = self.current_level.get_sprite_at(
                block_new_x, block_new_y, ignore_collidable=True
            )
            if block_behind and ("block" in block_behind.tags
                                 or "wall" in block_behind.tags):
                self.complete_action()
                return
            sprite.set_position(block_new_x, block_new_y)
            self._player.set_position(new_x, new_y)
        elif not sprite or not sprite.is_collidable:
            self._player.set_position(new_x, new_y)

        self._steps += 1
        self._sync()

        if self._covered() == len(self._blocks):
            self.next_level()
        elif self._steps >= self._step_limit:
            self.lose()

        self.complete_action()
