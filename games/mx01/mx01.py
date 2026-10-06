"""Maze melt: ACTION5 removes one orthogonally adjacent wall (budgeted). Reach the goal.

The board is randomised per (instance) RNG in ``on_set_level``: agent/goal
spawns, divider hole positions, divider translation and the background / divider
/ agent colours are all drawn from ``self._rng``. Live play uses an unseeded RNG
(fresh board every episode); the training generator seeds ``self._rng`` with
``mx01:<seed>:<level>`` before each level so a given seed reproduces the same
board byte-for-byte.

The five levels keep their distinct characters:

    0 vdivider   single vertical divider (one hole)
    1 hdivider   single horizontal divider (one hole)
    2 vdivider2  two vertical dividers, 1px apart, one hole each
    3 hcorridor  horizontal corridor (translated + widened, no hole)
    4 diagonal   diagonal wall (main or anti orientation)
"""

import random as _random_module

from utils.arc_game import AugmentedGame

from arcengine import (
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)


class Mx01UI(RenderableUserDisplay):
    def __init__(self, melts: int) -> None:
        self._melts = melts

    def update(self, melts: int) -> None:
        self._melts = melts

    def render_interface(self, frame):
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return frame
        h, w = frame.shape
        for i in range(min(self._melts, 12)):
            frame[h - 2, 1 + i] = 11
        return frame


sprites = {
    "player": Sprite(
        pixels=[[9]],
        name="player",
        visible=True,
        collidable=True,
        tags=["player"],
    ),
    "goal": Sprite(
        pixels=[[14]],
        name="goal",
        visible=True,
        collidable=False,
        tags=["goal"],
    ),
    "wall": Sprite(
        pixels=[[3]],
        name="wall",
        visible=True,
        collidable=True,
        tags=["wall"],
    ),
}


def mk(kind: str, melts: int, d: int):
    """A level descriptor. Sprites are built procedurally in ``_randomize_level``;
    only the geometry ``kind``, melt budget and difficulty live in the data."""
    return Level(
        sprites=[],
        grid_size=(10, 10),
        data={"difficulty": d, "melt_budget": melts, "kind": kind},
    )


levels = [
    mk("vdivider", 2, 1),
    mk("hdivider", 1, 2),
    mk("vdivider2", 3, 3),
    mk("hcorridor", 2, 4),
    mk("diagonal", 4, 5),
]

BACKGROUND_COLOR = 5
PADDING_COLOR = 4
GOAL_COLOR = 14
# Base pixel colours of the sprite templates (remapped per-level).
_WALL_BASE = 3
_PLAYER_BASE = 9
# Colours available for the randomised background / divider / agent. Excludes
# the fixed goal colour (14), the padding border (4), the melt-UI colour (11)
# and 0; the three are sampled distinct so every element stays visible.
_COLOR_POOL = [1, 2, 3, 6, 7, 8, 9, 10, 12, 13, 15]


def _wall(x: int, y: int) -> Sprite:
    return sprites["wall"].clone().set_position(x, y)


def _player(x: int, y: int) -> Sprite:
    return sprites["player"].clone().set_position(x, y)


def _goal(x: int, y: int) -> Sprite:
    return sprites["goal"].clone().set_position(x, y)


class Mx01(AugmentedGame):
    def __init__(self, seed: int | None = None) -> None:
        # FIRST: super().__init__ seats level 0, which calls
        # on_set_level -> level_rng, and that needs the seed installed.
        self._init_augmentation(seed)
        # Set BEFORE super().__init__: it calls set_level(0) -> on_set_level ->
        # _randomize_level, which consumes this RNG.
        self._rng = self.level_rng("layout")
        self._ui = Mx01UI(0)
        super().__init__(
            "mx01",
            levels,
            Camera(0, 0, 16, 16, BACKGROUND_COLOR, PADDING_COLOR, [self._ui]),
            False,
            1,
            [1, 2, 3, 4, 5],
        )

    # ----- board construction -------------------------------------------------

    def _build_vdivider(self, level: Level) -> None:
        """Single vertical divider with one hole; agent left, goal right."""
        r = self._rng
        cx = 5 + r.choice([-2, -1, 0, 1, 2])          # divider column, in [3, 7]
        hy = r.randint(0, 9)                            # hole row
        for y in range(10):
            if y != hy:
                level.add_sprite(_wall(cx, y))
        level.add_sprite(_player(r.randint(0, cx - 1), r.randint(0, 9)))
        level.add_sprite(_goal(r.randint(cx + 1, 9), r.randint(0, 9)))

    def _build_hdivider(self, level: Level) -> None:
        """Single horizontal divider with one hole; agent above, goal below."""
        r = self._rng
        cy = 5 + r.choice([-2, -1, 0, 1, 2])          # divider row, in [3, 7]
        hx = r.randint(0, 9)                            # hole column
        for x in range(10):
            if x != hx:
                level.add_sprite(_wall(x, cy))
        level.add_sprite(_player(r.randint(0, 9), r.randint(0, cy - 1)))
        level.add_sprite(_goal(r.randint(0, 9), r.randint(cy + 1, 9)))

    def _build_vdivider2(self, level: Level) -> None:
        """Two vertical dividers 1px apart (columns cx, cx+2), one hole each;
        both translate together so the 1px gap is preserved."""
        r = self._rng
        cx = 4 + r.choice([-2, -1, 0, 1, 2])          # left divider col, in [2, 6]
        hy1 = r.randint(0, 9)
        hy2 = r.randint(0, 9)
        for y in range(10):
            if y != hy1:
                level.add_sprite(_wall(cx, y))
            if y != hy2:
                level.add_sprite(_wall(cx + 2, y))
        level.add_sprite(_player(r.randint(0, cx - 1), r.randint(0, 9)))
        level.add_sprite(_goal(r.randint(cx + 3, 9), r.randint(0, 9)))

    def _build_hcorridor(self, level: Level) -> None:
        """Horizontal corridor (no hole): the whole block (walls + agent + goal)
        is translated up/down and the corridor is randomly widened."""
        r = self._rng
        w = r.choice([1, 2, 3])                        # corridor width in rows
        yt = r.randint(0, 8 - w)                        # top wall row
        yb = yt + w + 1                                 # bottom wall row
        for x in range(10):
            level.add_sprite(_wall(x, yt))
            level.add_sprite(_wall(x, yb))
        py = (yt + yb) // 2                             # a corridor row
        level.add_sprite(_player(0, py))
        level.add_sprite(_goal(9, py))

    def _build_diagonal(self, level: Level) -> None:
        """Diagonal wall, either main (x==y) or anti (x==9-y) orientation.
        Start positions are fixed (agent bottom-right, goal top-left)."""
        r = self._rng
        if r.choice(["main", "anti"]) == "main":
            for i in range(1, 9):
                level.add_sprite(_wall(i, i))
        else:
            for i in range(1, 9):
                level.add_sprite(_wall(i, 9 - i))
        level.add_sprite(_player(9, 9))
        level.add_sprite(_goal(0, 0))

    _BUILDERS = {
        "vdivider": _build_vdivider,
        "hdivider": _build_hdivider,
        "vdivider2": _build_vdivider2,
        "hcorridor": _build_hcorridor,
        "diagonal": _build_diagonal,
    }

    def _randomize_level(self) -> None:
        level = self.current_level
        level.remove_all_sprites()

        # Colours: background / divider / agent distinct, none equal to the goal
        # (14), padding (4) or melt-UI (11) colours.
        bg, divider, agent = self._rng.sample(_COLOR_POOL, 3)
        self.camera.background = bg

        self._BUILDERS[level.get_data("kind")](self, level)

        for s in level.get_sprites_by_tag("wall"):
            s.color_remap(_WALL_BASE, divider)
        for s in level.get_sprites_by_tag("player"):
            s.color_remap(_PLAYER_BASE, agent)

    # ----- engine hooks -------------------------------------------------------

    def on_set_level(self, level: Level) -> None:
        # Re-key per level so the board is a pure function of
        # (seed, level), identical whether this level was jumped to
        # or played into. See AugmentedGame.level_rng.
        self._rng = self.level_rng("layout")
        # Rebuild from the clean descriptor, then randomise deterministically
        # from self._rng.
        self._levels[self._current_level_index] = (
            self._clean_levels[self._current_level_index].clone()
        )
        self._randomize_level()
        self._player = self.current_level.get_sprites_by_tag("player")[0]
        self._goal = self.current_level.get_sprites_by_tag("goal")[0]
        self._left = int(self.current_level.get_data("melt_budget") or 3)
        self._ui.update(self._left)

    def step(self) -> None:
        aid = self.action.id

        if aid == GameAction.ACTION5:
            if self._left <= 0:
                self.complete_action()
                return
            px, py = self._player.x, self._player.y
            best = None
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = px + dx, py + dy
                sp = self.current_level.get_sprite_at(nx, ny, ignore_collidable=True)
                if sp and "wall" in sp.tags:
                    best = sp
                    break
            if best:
                self.current_level.remove_sprite(best)
                self._left -= 1
                self._ui.update(self._left)
            self.complete_action()
            return

        dx = dy = 0
        if aid == GameAction.ACTION1:
            dy = -1
        elif aid == GameAction.ACTION2:
            dy = 1
        elif aid == GameAction.ACTION3:
            dx = -1
        elif aid == GameAction.ACTION4:
            dx = 1
        else:
            self.complete_action()
            return

        nx, ny = self._player.x + dx, self._player.y + dy
        gw, gh = self.current_level.grid_size
        if 0 <= nx < gw and 0 <= ny < gh:
            sp = self.current_level.get_sprite_at(nx, ny, ignore_collidable=True)
            if not sp or not sp.is_collidable:
                self._player.set_position(nx, ny)

        if self._player.x == self._goal.x and self._player.y == self._goal.y:
            self.next_level()

        self.complete_action()
