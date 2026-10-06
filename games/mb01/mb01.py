"""Magnet crates: after each move, each metal crate slides one cell toward the player if the path is clear.

Colours are randomised per-seed in ``on_set_level`` from the instance RNG: the
background, the agent and the metal crates (with the matching crate-count legend).
Live play uses an unseeded RNG (fresh colours each episode); the training
generator seeds ``self._rng`` with ``mb01:<seed>:<level>`` so a given seed
reproduces the same board byte-for-byte."""

import random as _random_module

from utils.arc_game import AugmentedGame

from arcengine import (
    ARCBaseGame,
    Camera,
    Level,
    RenderableUserDisplay,
    Sprite,
)


class Mb01UI(RenderableUserDisplay):
    def __init__(self, n_crates: int, crate_color: int = 12) -> None:
        self._n_crates = n_crates
        self._crate_color = crate_color

    def update(self, n_crates: int, crate_color: int | None = None) -> None:
        self._n_crates = n_crates
        if crate_color is not None:
            self._crate_color = crate_color

    def render_interface(self, frame):
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return frame
        h, _w = frame.shape
        for i in range(min(self._n_crates, 12)):
            frame[h - 2, 1 + i] = self._crate_color
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
    "metal": Sprite(
        pixels=[[2]],
        name="metal",
        visible=True,
        collidable=True,
        tags=["metal", "crate"],
    ),
}


def mk(sl: list, d: int) -> Level:
    return Level(sprites=sl, grid_size=(10, 10), data={"difficulty": d})


levels = [
    mk(
        [
            sprites["player"].clone().set_position(1, 5),
            sprites["metal"].clone().set_position(5, 5),
            sprites["goal"].clone().set_position(8, 5),
        ],
        1,
    ),
    mk(
        [
            sprites["player"].clone().set_position(0, 5),
            sprites["metal"].clone().set_position(4, 5),
            sprites["goal"].clone().set_position(9, 5),
        ],
        2,
    ),
    mk(
        [
            sprites["player"].clone().set_position(2, 2),
            sprites["metal"].clone().set_position(5, 2),
            sprites["metal"].clone().set_position(5, 5),
            sprites["goal"].clone().set_position(8, 8),
        ],
        3,
    ),
    mk(
        [
            sprites["player"].clone().set_position(1, 1),
            sprites["metal"].clone().set_position(4, 4),
            sprites["goal"].clone().set_position(8, 1),
            sprites["wall"].clone().set_position(6, 3),
            sprites["wall"].clone().set_position(6, 4),
            sprites["wall"].clone().set_position(6, 5),
        ],
        4,
    ),
    mk(
        [
            sprites["player"].clone().set_position(0, 0),
            sprites["metal"].clone().set_position(3, 3),
            sprites["goal"].clone().set_position(9, 9),
        ],
        5,
    ),
]

BACKGROUND_COLOR = 5
PADDING_COLOR = 4
GOAL_COLOR = 14
WALL_COLOR = 3
# Base pixel colours of the sprite templates (remapped per level).
_PLAYER_BASE = 9
_CRATE_BASE = 2
# Colours for the randomised background / agent / metal crate. Excludes the
# fixed goal (14), wall (3) and padding (4) colours and 0; the three are sampled
# distinct so no cell type is confusable with another or the background.
_COLOR_POOL = [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 15]


class Mb01(AugmentedGame):
    def __init__(self, seed: int | None = None) -> None:
        # FIRST: super().__init__ seats level 0, which calls
        # on_set_level -> level_rng, and that needs the seed installed.
        self._init_augmentation(seed)
        # Set BEFORE super().__init__: it calls set_level(0) -> on_set_level ->
        # _randomize_level, which consumes this RNG.
        self._rng = self.level_rng("layout")
        self._crate_color = _CRATE_BASE
        self._ui = Mb01UI(1)
        super().__init__(
            "mb01",
            levels,
            Camera(0, 0, 16, 16, BACKGROUND_COLOR, PADDING_COLOR, [self._ui]),
            False,
            1,
            [1, 2, 3, 4],
        )

    def _randomize_line_y(self, level: Level) -> None:
        """Levels 1 & 2: agent | crate | goal share a row. Translate all three to
        a random row together (x kept), so they stay aligned. This is a pure
        vertical shift of a solvable open-grid layout, so it stays solvable."""
        ry = self._rng.randint(0, level.grid_size[1] - 1)
        for tag in ("player", "metal", "goal"):
            for s in level.get_sprites_by_tag(tag):
                s.set_position(s.x, ry)

    def _translate_level3(self, level: Level) -> None:
        """Level 3: shift all blocks together by a small (<=3) x/y offset,
        clamped so the whole group stays on the grid. A rigid translation keeps
        the x/y axes (and the horizontal tie-break) intact, so it stays solvable."""
        gw, gh = level.grid_size
        cells = level.get_sprites()
        xs = [s.x for s in cells]
        ys = [s.y for s in cells]
        dx = self._rng.randint(max(-3, -min(xs)), min(3, (gw - 1) - max(xs)))
        dy = self._rng.randint(max(-3, -min(ys)), min(3, (gh - 1) - max(ys)))
        if dx or dy:
            for s in cells:
                s.set_position(s.x + dx, s.y + dy)

    def _flip_board(self, level: Level) -> None:
        """Randomly mirror the ENTIRE board horizontally and/or vertically. A
        flip preserves both |px-cx| and |py-cy| and the axis identity, so the
        magnet pull is exactly equivariant -- solvability is preserved on every
        level (unlike 90-degree rotations)."""
        gw, gh = level.grid_size
        fx = self._rng.random() < 0.5
        fy = self._rng.random() < 0.5
        if not (fx or fy):
            return
        for s in level.get_sprites():
            x = gw - 1 - s.x if fx else s.x
            y = gh - 1 - s.y if fy else s.y
            s.set_position(x, y)

    def _rotate_board(self, level: Level, choices) -> None:
        """Rotate the ENTIRE board (agent, crates, goal, walls) by a random
        number of 90-degree turns drawn from `choices`. A rigid rotation is an
        isometry of the layout, but the magnet pull is NOT rotation-invariant:
        `_pull_crates` breaks ties toward the horizontal axis, so 0/180-degree
        turns keep that bias (safe) while 90/270-degree turns swap the axes and
        can make a level unsolvable (they do for level 3). Callers pass the
        solvability-preserving subset. Square grid only."""
        gw, gh = level.grid_size
        if gw != gh:
            return
        n = gw
        k = self._rng.choice(choices)
        if k == 0:
            return
        for s in level.get_sprites():
            x, y = s.x, s.y
            for _ in range(k):  # each turn: 90 deg clockwise
                x, y = n - 1 - y, x
            s.set_position(x, y)

    def _randomize_level(self) -> None:
        """Randomise positions (per-level scheme + a whole-board rotation) then
        recolour the background / agent / metal crates (+ the crate-count
        legend). The goal (14) and walls (3) keep their fixed colours; all cell
        types are distinct."""
        level = self.current_level
        idx = self._current_level_index
        if idx in (0, 1):  # levels 1 & 2
            self._randomize_line_y(level)
        elif idx == 2:  # level 3
            self._translate_level3(level)
        # Level 3's two-crate solution relies on the horizontal tie-break, so
        # only 0/180 rotations keep it solvable; all other levels allow all four.
        rot_choices = (0, 2) if idx == 2 else (0, 1, 2, 3)
        self._rotate_board(level, rot_choices)
        self._flip_board(level)  # flips are solvability-safe on every level

        bg, agent, crate = self._rng.sample(_COLOR_POOL, 3)
        self.camera.background = bg
        for s in level.get_sprites_by_tag("player"):
            s.color_remap(_PLAYER_BASE, agent)
        for s in level.get_sprites_by_tag("metal"):
            s.color_remap(_CRATE_BASE, crate)
        # goal keeps GOAL_COLOR (14), walls keep WALL_COLOR (3)
        self._crate_color = crate

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
        self._crates = list(self.current_level.get_sprites_by_tag("metal"))
        self._ui.update(len(self._crates), crate_color=self._crate_color)

    def _blocked_move(self, x: int, y: int, ignore: Sprite | None = None) -> bool:
        gw, gh = self.current_level.grid_size
        if not (0 <= x < gw and 0 <= y < gh):
            return True
        sp = self.current_level.get_sprite_at(x, y, ignore_collidable=True)
        if sp is ignore:
            return False
        if not sp:
            return False
        if "goal" in sp.tags:
            return False
        return sp.is_collidable

    def _pull_crates(self) -> None:
        px, py = self._player.x, self._player.y
        for c in self._crates:
            cx, cy = c.x, c.y
            dx = (1 if px > cx else -1) if px != cx else 0
            dy = (1 if py > cy else -1) if py != cy else 0
            if dx != 0 and dy != 0:
                if abs(px - cx) >= abs(py - cy):
                    dy = 0
                else:
                    dx = 0
            nx, ny = cx + dx, cy + dy
            if not self._blocked_move(nx, ny, ignore=c):
                c.set_position(nx, ny)

    def step(self) -> None:
        dx = dy = 0
        if self.action.id.value == 1:
            dy = -1
        elif self.action.id.value == 2:
            dy = 1
        elif self.action.id.value == 3:
            dx = -1
        elif self.action.id.value == 4:
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
        if sp and "metal" in sp.tags:
            self.complete_action()
            return

        self._player.set_position(nx, ny)
        self._pull_crates()

        if self._player.x == self._goal.x and self._player.y == self._goal.y:
            self.next_level()

        self.complete_action()
