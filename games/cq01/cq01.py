"""Conquest ring: visit every orange ring marker cell, then reach the green goal.

Colours are randomised per-seed in ``on_set_level`` from the instance RNG: the
background, the agent and the ring markers (with the matching bottom-left legend).
Live play uses an unseeded RNG (fresh colours each episode); the training
generator seeds ``self._rng`` with ``cq01:<seed>:<level>`` so a given seed
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


class Cq01UI(RenderableUserDisplay):
    def __init__(self, n_levels: int, ring_color: int = 12) -> None:
        self._left = 0
        self._n_levels = n_levels
        self._li = 0
        self._ring_color = ring_color

    def update(
        self, left: int, level_index: int | None = None, ring_color: int | None = None
    ) -> None:
        self._left = left
        if level_index is not None:
            self._li = level_index
        if ring_color is not None:
            self._ring_color = ring_color

    def render_interface(self, frame):
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return frame
        h, w = frame.shape
        for i in range(min(self._n_levels, 14)):
            cx = 1 + i * 2
            if cx >= w:
                break
            c = 14 if i < self._li else (11 if i == self._li else 3)
            frame[0, cx] = c
        for i in range(min(self._left, 10)):
            frame[h - 2, 1 + i] = self._ring_color
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
    "ring": Sprite(
        pixels=[[12]],
        name="ring",
        visible=True,
        collidable=False,
        tags=["ring"],
    ),
}


def mk(sl: list, d: int) -> Level:
    return Level(sprites=sl, grid_size=(10, 10), data={"difficulty": d})


levels = [
    mk(
        [
            sprites["player"].clone().set_position(0, 5),
            sprites["goal"].clone().set_position(9, 5),
            sprites["ring"].clone().set_position(4, 5),
            sprites["ring"].clone().set_position(5, 5),
            sprites["ring"].clone().set_position(6, 5),
        ],
        1,
    ),
    mk(
        [
            sprites["player"].clone().set_position(1, 1),
            sprites["goal"].clone().set_position(8, 8),
        ]
        + [sprites["ring"].clone().set_position(x, 5) for x in range(2, 8)],
        2,
    ),
    mk(
        [
            sprites["player"].clone().set_position(0, 0),
            sprites["goal"].clone().set_position(9, 9),
            sprites["ring"].clone().set_position(3, 3),
            sprites["ring"].clone().set_position(6, 3),
            sprites["ring"].clone().set_position(6, 6),
            sprites["ring"].clone().set_position(3, 6),
        ],
        3,
    ),
    mk(
        [
            sprites["player"].clone().set_position(2, 5),
            sprites["goal"].clone().set_position(8, 5),
            sprites["ring"].clone().set_position(4, 4),
            sprites["ring"].clone().set_position(5, 4),
            sprites["ring"].clone().set_position(6, 4),
        ],
        4,
    ),
    mk(
        [
            sprites["player"].clone().set_position(0, 9),
            sprites["goal"].clone().set_position(9, 0),
            sprites["ring"].clone().set_position(5, 3),
            sprites["ring"].clone().set_position(5, 6),
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
_RING_BASE = 12
# Colours for the randomised background / agent / ring. Excludes the fixed goal
# (14), wall (3) and padding (4) colours and 0; the three are sampled distinct so
# no cell type is confusable with another or the background.
_COLOR_POOL = [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 15]


class Cq01(AugmentedGame):
    def __init__(self, seed: int | None = None) -> None:
        # FIRST: super().__init__ seats level 0, which calls
        # on_set_level -> level_rng, and that needs the seed installed.
        self._init_augmentation(seed)
        # Set BEFORE super().__init__: it calls set_level(0) -> on_set_level ->
        # _randomize_level, which consumes this RNG.
        self._rng = self.level_rng("layout")
        self._ring_color = _RING_BASE
        self._ui = Cq01UI(len(levels))
        super().__init__(
            "cq01",
            levels,
            Camera(0, 0, 16, 16, BACKGROUND_COLOR, PADDING_COLOR, [self._ui]),
            False,
            1,
            [1, 2, 3, 4],
        )

    def _randomize_line(self, level: Level) -> None:
        """Levels 1/2/4: agent | ring cluster | goal along x. Vary the y of the
        agent, goal and ring row; slightly vary the ring cluster's x (kept
        strictly between the agent and goal); then rotate the board 0/90/180."""
        r = self._rng
        gw, gh = level.grid_size
        players = level.get_sprites_by_tag("player")
        goals = level.get_sprites_by_tag("goal")
        rings = level.get_sprites_by_tag("ring")
        n = len(rings)
        ax, gx = players[0].x, goals[0].x  # agent left of goal (ax < gx)

        orig_rx = min(s.x for s in rings)
        lo, hi = ax + 1, gx - n  # cluster must sit strictly between agent & goal
        rx = max(lo, min(orig_rx + r.choice([-2, -1, 0, 1, 2]), hi))
        ry = r.randint(0, gh - 1)
        ay = r.randint(0, gh - 1)
        gy = r.randint(0, gh - 1)

        k = r.choice([0, 1, 2]) if gw == gh else 0  # rotate the whole board

        def place(sprite, x, y):
            if k == 1:  # 90 deg
                x, y = gh - 1 - y, x
            elif k == 2:  # 180 deg
                x, y = gw - 1 - x, gh - 1 - y
            sprite.set_position(x, y)

        place(players[0], ax, ay)
        place(goals[0], gx, gy)
        for i, s in enumerate(rings):
            place(s, rx + i, ry)

    def _randomize_rect(self, level: Level) -> None:
        """Level 3: 4 rings at the corners of an axis-aligned rectangle whose
        size (spacing) and grid position vary; agent and goal placed on random
        free cells."""
        r = self._rng
        gw, gh = level.grid_size
        players = level.get_sprites_by_tag("player")
        goals = level.get_sprites_by_tag("goal")
        rings = level.get_sprites_by_tag("ring")

        w = r.randint(2, gw - 2)
        h = r.randint(2, gh - 2)
        x0 = r.randint(0, gw - 1 - w)
        y0 = r.randint(0, gh - 1 - h)
        corners = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
        for s, c in zip(rings, corners):
            s.set_position(*c)

        ring_set = set(corners)
        free = [
            (x, y) for x in range(gw) for y in range(gh) if (x, y) not in ring_set
        ]
        a, g = r.sample(free, 2)
        players[0].set_position(*a)
        goals[0].set_position(*g)

    def _randomize_free(self, level: Level) -> None:
        """Level 5: agent, goal and both rings on fully random distinct cells."""
        r = self._rng
        gw, gh = level.grid_size
        players = level.get_sprites_by_tag("player")
        goals = level.get_sprites_by_tag("goal")
        rings = level.get_sprites_by_tag("ring")
        picks = r.sample([(x, y) for x in range(gw) for y in range(gh)], 2 + len(rings))
        players[0].set_position(*picks[0])
        goals[0].set_position(*picks[1])
        for s, c in zip(rings, picks[2:]):
            s.set_position(*c)

    def _randomize_level(self) -> None:
        """Randomise positions (per-level scheme) then recolour the background /
        agent / rings (+ the ring legend). The goal (14) and walls (3) keep their
        fixed colours; all cell types are distinct."""
        level = self.current_level
        idx = self._current_level_index
        if idx in (0, 1, 3):  # levels 1, 2, 4
            self._randomize_line(level)
        elif idx == 2:  # level 3
            self._randomize_rect(level)
        elif idx == 4:  # level 5
            self._randomize_free(level)

        bg, agent, ring = self._rng.sample(_COLOR_POOL, 3)
        self.camera.background = bg
        for s in level.get_sprites_by_tag("player"):
            s.color_remap(_PLAYER_BASE, agent)
        for s in level.get_sprites_by_tag("ring"):
            s.color_remap(_RING_BASE, ring)
        # goal keeps GOAL_COLOR (14), walls keep WALL_COLOR (3)
        self._ring_color = ring

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
        self._rings = {(r.x, r.y) for r in self.current_level.get_sprites_by_tag("ring")}
        self._visited: set[tuple[int, int]] = set()
        self._ui.update(len(self._rings), self.level_index, ring_color=self._ring_color)

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

        self._player.set_position(nx, ny)
        pos = (nx, ny)
        if pos in self._rings:
            self._visited.add(pos)
        self._ui.update(len(self._rings - self._visited), self.level_index)

        if self._rings <= self._visited and self._player.x == self._goal.x and self._player.y == self._goal.y:
            self.next_level()

        self.complete_action()
