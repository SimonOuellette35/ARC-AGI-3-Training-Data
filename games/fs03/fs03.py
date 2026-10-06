from collections import deque

import random as _random_module

from utils.arc_game import AugmentedGame

from arcengine import (
    ARCBaseGame,
    Camera,
    Level,
    RenderableUserDisplay,
    Sprite,
)


class Fs03UI(RenderableUserDisplay):
    """Exactly ``required`` (k) HUD ticks — fill with the switch colour as distinct
    plates activate, then all-green when k are reached; empty ticks blend into the
    background."""

    def __init__(
        self,
        activated: int,
        required: int,
        active_color: int = 11,
        done_color: int = 14,
        empty_color: int = 5,
    ) -> None:
        self._activated = activated
        self._required = required
        self._active_color = active_color
        self._done_color = done_color
        self._empty_color = empty_color

    def update(
        self,
        activated: int,
        required: int,
        active_color: int | None = None,
        done_color: int | None = None,
        empty_color: int | None = None,
    ) -> None:
        self._activated = activated
        self._required = required
        if active_color is not None:
            self._active_color = active_color
        if done_color is not None:
            self._done_color = done_color
        if empty_color is not None:
            self._empty_color = empty_color

    def render_interface(self, frame):
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return frame
        h, w = frame.shape
        k = min(max(self._required, 1), 4)
        done = self._activated >= self._required and self._required > 0
        color = self._done_color if done else self._active_color
        n_fill = min(self._activated, k)
        base_x = w - 2 * k
        for i in range(k):
            c = color if i < n_fill else self._empty_color
            for dy in range(2):
                for dx in range(2):
                    frame[h - 4 + dy, base_x + i * 2 + dx] = c
        return frame


sprites = {
    "player": Sprite(
        pixels=[[9]],
        name="player",
        visible=True,
        collidable=True,
        tags=["player"],
    ),
    "switch": Sprite(
        pixels=[[11]],
        name="switch",
        visible=True,
        collidable=False,
        tags=["switch"],
    ),
    "door": Sprite(
        pixels=[[3]],
        name="door",
        visible=True,
        collidable=True,
        tags=["door"],
    ),
    "target": Sprite(
        pixels=[[14]],
        name="target",
        visible=True,
        collidable=False,
        tags=["target"],
    ),
    "wall": Sprite(
        pixels=[[3]],
        name="wall",
        visible=True,
        collidable=True,
        tags=["wall"],
    ),
}


def mk(sl, grid_size, difficulty, required_plates: int):
    return Level(
        sprites=sl,
        grid_size=grid_size,
        data={
            "difficulty": difficulty,
            "required_plates": required_plates,
        },
    )


# k-of-n: step on **required_plates** distinct yellow plates (any order); extras are optional.
# The wall/door obstacle structure is fixed per level; the player / switch / target
# positions (and all colours) are randomised per-seed in _randomize_level.
levels = [
    mk(
        [
            sprites["player"].clone().set_position(0, 3),
            sprites["switch"].clone().set_position(2, 3),
            sprites["switch"].clone().set_position(4, 1),
            sprites["switch"].clone().set_position(4, 5),
            sprites["door"].clone().set_position(6, 3),
            sprites["target"].clone().set_position(7, 3),
        ],
        (8, 8),
        1,
        2,
    ),
    mk(
        [
            sprites["player"].clone().set_position(1, 1),
            sprites["switch"].clone().set_position(2, 2),
            sprites["switch"].clone().set_position(2, 6),
            sprites["switch"].clone().set_position(6, 2),
            sprites["switch"].clone().set_position(6, 6),
            sprites["door"].clone().set_position(4, 4),
            sprites["target"].clone().set_position(4, 7),
        ]
        + [
            sprites["wall"].clone().set_position(x, 4)
            for x in range(8)
            if x not in (1, 4, 6)
        ],
        (8, 8),
        2,
        2,
    ),
    mk(
        [
            sprites["player"].clone().set_position(0, 0),
            sprites["switch"].clone().set_position(2, 0),
            sprites["switch"].clone().set_position(0, 3),
            sprites["switch"].clone().set_position(2, 3),
            sprites["switch"].clone().set_position(1, 5),
            sprites["door"].clone().set_position(5, 2),
            sprites["target"].clone().set_position(8, 2),
        ]
        + [sprites["wall"].clone().set_position(4, y) for y in range(6) if y != 2],
        (9, 6),
        3,
        3,
    ),
    mk(
        [
            sprites["player"].clone().set_position(1, 4),
            sprites["switch"].clone().set_position(2, 2),
            sprites["switch"].clone().set_position(2, 6),
            sprites["switch"].clone().set_position(7, 2),
            sprites["switch"].clone().set_position(7, 6),
            sprites["door"].clone().set_position(5, 4),
            sprites["target"].clone().set_position(9, 4),
        ]
        + [sprites["wall"].clone().set_position(4, y) for y in range(8) if y != 4],
        (10, 8),
        4,
        2,
    ),
    mk(
        [
            sprites["player"].clone().set_position(0, 7),
            sprites["switch"].clone().set_position(1, 1),
            sprites["switch"].clone().set_position(6, 1),
            sprites["switch"].clone().set_position(1, 5),
            sprites["switch"].clone().set_position(6, 5),
            sprites["door"].clone().set_position(3, 4),
            sprites["target"].clone().set_position(7, 4),
        ]
        + [
            sprites["wall"].clone().set_position(x, 4)
            for x in range(8)
            if x not in (3, 4, 7)
        ],
        (8, 8),
        5,
        3,
    ),
]

BACKGROUND_COLOR = 5
PADDING_COLOR = 4
TARGET_COLOR = 14
# Base pixel colours of the sprite templates (remapped per level).
_PLAYER_BASE = 9
_SWITCH_BASE = 11
_OBSTACLE_BASE = 3
# Colours for the randomised background / agent / switch (inactive) / switch
# (activated) / obstacle. Excludes the fixed target/HUD-green colour (14), the
# padding border (4) and 0; the five are sampled distinct so no element hides
# against another or the background.
_COLOR_POOL = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15]


class Fs03(AugmentedGame):
    """k-of-n: activate **required_plates** distinct yellow plates (order-free); then the door opens."""

    def __init__(self, seed: int | None = None) -> None:
        # FIRST: super().__init__ seats level 0, which calls
        # on_set_level -> level_rng, and that needs the seed installed.
        self._init_augmentation(seed)
        # Set BEFORE super().__init__: it calls set_level(0) -> on_set_level ->
        # _randomize_level, which consumes this RNG.
        self._rng = self.level_rng("layout")
        self._switch_color = _SWITCH_BASE
        self._activated_color = 10
        self._ui = Fs03UI(0, 1)
        super().__init__(
            "fs03",
            levels,
            Camera(0, 0, 16, 16, BACKGROUND_COLOR, PADDING_COLOR, [self._ui]),
            False,
            1,
            [1, 2, 3, 4],
        )

    def _randomize_level(self) -> None:
        """Relocate the player / switches / target onto free cells (keeping the
        wall+door structure) so the board stays solvable, then recolour."""
        level = self.current_level
        gw, gh = level.grid_size
        walls = {(s.x, s.y) for s in level.get_sprites_by_tag("wall")}
        doors = {(s.x, s.y) for s in level.get_sprites_by_tag("door")}
        players = level.get_sprites_by_tag("player")
        switches = level.get_sprites_by_tag("switch")
        targets = level.get_sprites_by_tag("target")
        n_sw = len(switches)

        free = [
            (x, y)
            for x in range(gw)
            for y in range(gh)
            if (x, y) not in walls and (x, y) not in doors
        ]

        def reachable(start, blocked):
            seen = {start}
            dq = deque([start])
            while dq:
                x, y = dq.popleft()
                for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < gw
                        and 0 <= ny < gh
                        and (nx, ny) not in blocked
                        and (nx, ny) not in seen
                    ):
                        seen.add((nx, ny))
                        dq.append((nx, ny))
            return seen

        # Door closed blocks the door cell too; when open only walls block. Every
        # plate must be pressable with the door still closed (so any k of the n
        # can be activated); the target must be reachable once the door is open.
        blocked_closed = walls | doors
        blocked_open = walls
        free_set = set(free)

        r = self._rng
        chosen = None
        for _ in range(400):
            p = r.choice(free)
            r_closed = reachable(p, blocked_closed)
            r_open = reachable(p, blocked_open)
            sw_cand = [c for c in r_closed if c != p and c in free_set]
            if len(sw_cand) < n_sw:
                continue
            sws = r.sample(sw_cand, n_sw)
            sws_set = set(sws)
            tg_pool = [
                c for c in r_open if c != p and c in free_set and c not in sws_set
            ]
            if not tg_pool:
                continue
            chosen = (p, sws, r.choice(tg_pool))
            break

        if chosen is None:  # extremely unlikely; keep the clean layout
            p = (players[0].x, players[0].y)
            sws = [(s.x, s.y) for s in switches]
            tg = (targets[0].x, targets[0].y)
        else:
            p, sws, tg = chosen

        players[0].set_position(*p)
        for sp, pos in zip(switches, sws):
            sp.set_position(*pos)
        targets[0].set_position(*tg)

        # Colours: background / agent / switch (inactive) / switch (activated) /
        # obstacle mutually distinct and none equal to the target (14) or padding.
        bg, agent, sw_color, act_color, obs_color = r.sample(_COLOR_POOL, 5)
        self.camera.background = bg
        for s in players:
            s.color_remap(_PLAYER_BASE, agent)
        for s in switches:
            s.color_remap(_SWITCH_BASE, sw_color)
        for s in level.get_sprites_by_tag("wall"):
            s.color_remap(_OBSTACLE_BASE, obs_color)
        for s in level.get_sprites_by_tag("door"):
            s.color_remap(_OBSTACLE_BASE, obs_color)
        # target keeps TARGET_COLOR (14)
        self._switch_color = sw_color
        self._activated_color = act_color

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
        lvl = self.current_level
        self._player = lvl.get_sprites_by_tag("player")[0]
        self._switches = lvl.get_sprites_by_tag("switch")
        self._switch_positions = frozenset((s.x, s.y) for s in self._switches)
        n = len(self._switches)
        req = int(lvl.get_data("required_plates") or n)
        self._required = max(1, min(req, n))
        self._activated: set[tuple[int, int]] = set()
        self._door = lvl.get_sprites_by_tag("door")
        self._targets = lvl.get_sprites_by_tag("target")
        # Pin the HUD colours: fills with the switch colour, empties with the bg.
        self._ui.update(
            len(self._activated),
            self._required,
            active_color=self._switch_color,
            empty_color=self.camera.background,
        )
        self._open_door_if_ready()

    def _open_door_if_ready(self) -> None:
        if len(self._activated) >= self._required and self._door:
            for d in list(self._door):
                self.current_level.remove_sprite(d)
            self._door = []
        self._ui.update(len(self._activated), self._required)

    def step(self) -> None:
        dx = 0
        dy = 0
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

        if sprite and "door" in sprite.tags:
            self.complete_action()
            return

        if not sprite or not sprite.is_collidable:
            self._player.set_position(new_x, new_y)

        pos = (self._player.x, self._player.y)
        if pos in self._switch_positions and pos not in self._activated:
            self._activated.add(pos)
            for sw in self._switches:
                if sw.x == pos[0] and sw.y == pos[1]:
                    sw.color_remap(self._switch_color, self._activated_color)
                    break
            self._open_door_if_ready()

        for t in self._targets:
            if self._player.x == t.x and self._player.y == t.y:
                if len(self._door) == 0:
                    self.next_level()
                break

        self.complete_action()
