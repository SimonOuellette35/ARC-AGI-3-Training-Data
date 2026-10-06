"""Key & Door: pick the key up to unlock the door, and step through it.

An 8x8 open board holds three things -- the player, a key and a door. ACTION1..4
walk one cell. Walking onto the key collects it (the key vanishes and the
bottom-right indicator lights up in the key's colour); walking onto the door is
blocked while the key is still out there, and finishes the level once it is
carried. Five levels, each a longer walk than the last.

PER-SEED AUGMENTATION (the repo convention -- see utils/arc_game.py and the
per-seed-augmentation notes). ``Ul01(seed=s)`` makes every level a pure function
of ``(s, level_index)``:

  * DISPLAY ROTATION, owned by ``AugmentedGame``: the frame is rendered rotated by
    a per-(seed, level) k, and ``step`` maps the pressed SCREEN direction back to
    game space via ``screen_action_to_game``. No rotation code lives here.
  * LAYOUT: the player / key / door cells are re-drawn per (seed, level), keeping
    each level's character -- the DIFFICULTY RAMP is stated as the range the
    level's optimal solution must fall in (``_LEVEL_PARAMS``) and enforced by the
    exact distance field in ``utils/keydoor_puzzle.py``, the same model the
    training generator solves with. So a sampled board is solvable by
    construction, and it is as long a walk as the level is supposed to be.
  * COLOURS: player / key / door colours come from one EPISODE-scoped scheme
    (level-independent seed, cached), so every level of one episode looks the
    same. Background (5) and padding (4) stay fixed, as in the rest of this game
    family (sk01/pw01/nu01).

Nothing is hidden: the three cells are distinct, so at t=0 the whole board is
visible, and the one bit of state the board does not carry directly -- whether the
key has been picked up -- is shown twice over, by the indicator and by the key's
absence.
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
from utils.keydoor_puzzle import KeyDoorField

BACKGROUND_COLOR = 5
PADDING_COLOR = 4
PLAYER_COLOR = 9
KEY_COLOR = 11
DOOR_COLOR = 3

# Colours drawn for the player / key / door, sampled mutually distinct so nothing
# hides against anything else. Excludes the background (5), the padding (4) and 0.
_COLOR_POOL = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


class Ul01UI(RenderableUserDisplay):
    """Has-key indicator: a 4x4 patch in the bottom-right corner, painted in the
    key's colour once it is carried and in the background colour before that (so
    it simply is not there until the key is picked up)."""

    def __init__(self, has_key: bool, key_color: int = KEY_COLOR) -> None:
        self._has_key = has_key
        self._key_color = key_color

    def update(self, has_key: bool, key_color: int | None = None) -> None:
        self._has_key = has_key
        if key_color is not None:
            self._key_color = key_color

    def render_interface(self, frame):
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return frame
        h, w = frame.shape
        color = self._key_color if self._has_key else BACKGROUND_COLOR
        for dy in range(4):
            for dx in range(4):
                frame[h - 4 + dy, w - 4 + dx] = color
        return frame


sprites = {
    "player": Sprite(
        pixels=[[PLAYER_COLOR]],
        name="player",
        visible=True,
        collidable=True,
        tags=["player"],
    ),
    "key": Sprite(
        pixels=[[KEY_COLOR]],
        name="key",
        visible=True,
        collidable=False,
        tags=["key"],
    ),
    "door": Sprite(
        pixels=[[DOOR_COLOR]],
        name="door",
        visible=True,
        collidable=True,
        tags=["door"],
    ),
}


def make_level(player_pos, key_pos, door_pos, difficulty):
    return Level(
        sprites=[
            sprites["player"].clone().set_position(*player_pos),
            sprites["key"].clone().set_position(*key_pos),
            sprites["door"].clone().set_position(*door_pos),
        ],
        grid_size=(8, 8),
        data={"difficulty": difficulty},
    )


# The stock five levels. These are the TEMPLATES: they fix the board size and are
# the fallback layout if per-seed sampling ever fails. Their optimal solutions are
# 9 / 12 / 12 / 13 / 21 walks, which is the ramp ``_LEVEL_PARAMS`` reproduces.
levels = [
    make_level((1, 1), (3, 3), (6, 1), 1),
    make_level((1, 1), (5, 5), (7, 3), 2),
    make_level((1, 1), (4, 4), (7, 7), 3),
    make_level((0, 0), (3, 3), (7, 0), 4),
    make_level((0, 7), (7, 0), (7, 7), 5),
]

#: Per-level augmentation budget, index-aligned with ``levels``: the inclusive
#: range the sampled board's OPTIMAL solution length must fall in. 26 is close to
#: the 8x8 ceiling (two corner-to-corner legs = 28), so level 5 stays a long walk.
_LEVEL_PARAMS = [(6, 9), (9, 12), (12, 15), (15, 19), (19, 26)]

_LAYOUT_ATTEMPTS = 400


def _sample_layout(rng, gw: int, gh: int, lo: int, hi: int):
    """A layout whose optimal solution is ``lo..hi`` walks, or None if sampling
    failed (the caller then keeps the template layout).

    Draw the key and the door, build the exact field for that geometry, and read
    off the player start cells that give the wanted solution length -- so the
    difficulty is measured, not estimated. (Estimating it as ``|p-k| + |k-d|``
    would be wrong wherever the locked door sits across the walk to the key, and
    would say nothing about solvability.)
    """
    cells = [(x, y) for x in range(gw) for y in range(gh)]
    for _ in range(_LAYOUT_ATTEMPTS):
        key, door = rng.sample(cells, 2)
        field = KeyDoorField(gw, gh, key, door)
        starts = field.start_cells(lo, hi)
        if not starts:
            continue
        return {"player": field.xy(rng.choice(starts)), "key": key, "door": door}
    return None


class Ul01(AugmentedGame):
    """Pick up key to unlock door and advance to next level."""

    def __init__(self, seed: int | None = None) -> None:
        # FIRST: the base draws this level's rotation and turns the BOARD by it in
        # set_level, before the engine seats the level.
        self._init_augmentation(seed)
        # Episode-scoped colour scheme (level-INDEPENDENT), drawn once and cached so
        # every level of one episode is coloured identically.
        self._color_scheme: dict | None = None
        self._ui = Ul01UI(False)
        super().__init__(
            "ul01",
            levels,
            Camera(0, 0, 8, 8, BACKGROUND_COLOR, PADDING_COLOR,
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
            f"ul01:{self._seed_value}:{self._current_level_index}")

    def _ensure_color_scheme(self) -> None:
        if self._color_scheme is not None:
            return
        rng = (_random_module.Random(f"ul01-colors:{self._seed_value}")
               if self._seed_value is not None else _random_module.Random())
        player, key, door = rng.sample(_COLOR_POOL, 3)
        self._color_scheme = {"player": player, "key": key, "door": door}

    def _apply_layout(self, level: Level, rng) -> None:
        gw, gh = level.grid_size
        lo, hi = _LEVEL_PARAMS[self._current_level_index]
        layout = _sample_layout(rng, gw, gh, lo, hi)
        if layout is None:                            # keep the template layout
            return
        for tag in ("player", "key", "door"):
            level.get_sprites_by_tag(tag)[0].set_position(*layout[tag])

    def _apply_colors(self, level: Level) -> None:
        self._ensure_color_scheme()
        scheme = self._color_scheme
        for sprite in level.get_sprites_by_tag("player"):
            sprite.color_remap(PLAYER_COLOR, scheme["player"])
        for sprite in level.get_sprites_by_tag("key"):
            sprite.color_remap(KEY_COLOR, scheme["key"])
        for sprite in level.get_sprites_by_tag("door"):
            sprite.color_remap(DOOR_COLOR, scheme["door"])

    def full_reset(self) -> None:
        # A full reset starts a new episode -> a new colour scheme.
        self._color_scheme = None
        super().full_reset()

    # ── engine hooks ────────────────────────────────────────────────────────
    def on_set_level(self, level: Level) -> None:
        # Rebuild from the clean template first. ``set_level`` re-seats the SAME
        # persistent Level object, and play mutates it destructively (the key and
        # the door sprites are removed as they are used) while ``_apply_layout`` /
        # ``_apply_colors`` mutate positions and pixels in place -- so re-seating a
        # level (a RESET, a jump-to-level, a generator's replay) has to start from
        # the pristine template or it would find a board with pieces missing and a
        # colour remap applied twice.
        idx = self._current_level_index
        self._levels[idx] = self._clean_levels[idx].clone()
        lvl = self.current_level
        self._apply_layout(lvl, self._level_rng())
        self._apply_colors(lvl)
        self._player = lvl.get_sprites_by_tag("player")[0]
        self._key = lvl.get_sprites_by_tag("key")
        self._door = lvl.get_sprites_by_tag("door")
        self._has_key = False
        self._ui.update(self._has_key, self._color_scheme["key"])

    def step(self) -> None:
        # The board is displayed rotated, so the pressed SCREEN direction has to be
        # mapped back to game space before it means anything here.
        action_id = self.screen_action_to_game(self.action.id)
        dx = 0
        dy = 0
        moved = False

        if action_id == GameAction.ACTION1:
            dy = -1
            moved = True
        elif action_id == GameAction.ACTION2:
            dy = 1
            moved = True
        elif action_id == GameAction.ACTION3:
            dx = -1
            moved = True
        elif action_id == GameAction.ACTION4:
            dx = 1
            moved = True

        if not moved:
            self.complete_action()
            return

        new_x = self._player.x + dx
        new_y = self._player.y + dy

        grid_w, grid_h = self.current_level.grid_size
        if 0 <= new_x < grid_w and 0 <= new_y < grid_h:
            sprite = self.current_level.get_sprite_at(
                new_x, new_y, ignore_collidable=True
            )

            if sprite and "key" in sprite.tags:
                self.current_level.remove_sprite(sprite)
                self._key.remove(sprite)
                self._has_key = True
                self._player.set_position(new_x, new_y)
                self._ui.update(self._has_key)
            elif sprite and "door" in sprite.tags:
                if self._has_key:
                    self.current_level.remove_sprite(sprite)
                    self._door.remove(sprite)
                    self._player.set_position(new_x, new_y)
                # else: door blocks player
            elif not sprite or not sprite.is_collidable:
                self._player.set_position(new_x, new_y)

        if len(self._door) == 0 and len(self._key) == 0:
            self.next_level()

        self.complete_action()
