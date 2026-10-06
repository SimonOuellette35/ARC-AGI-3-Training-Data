"""Minesweeper with flags: ACTION6 places a flag on an unrevealed cell; wrong flag on empty safe cell = lose.

Mine positions are randomised per-seed in ``on_set_level`` from the instance RNG
(same mine count per level, player-start & goal cells kept clear, and always a
guaranteed fully-safe COUNT-0 path to the goal). Live play uses an unseeded RNG;
the training generator seeds ``self._rng`` with ``ms02:<seed>:<level>`` so a given
seed reproduces the same board byte-for-byte."""

import random as _random_module

from utils.arc_game import AugmentedGame
from collections import deque

from arcengine import (
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)

BACKGROUND_COLOR = 5
PADDING_COLOR = 4

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
    "mine": Sprite(
        pixels=[[8]],
        name="mine",
        visible=False,
        collidable=False,
        tags=["mine"],
    ),
    "tile": Sprite(
        pixels=[[4]],
        name="tile",
        visible=True,
        collidable=False,
        tags=["tile"],
    ),
    "wall": Sprite(
        pixels=[[3]],
        name="wall",
        visible=True,
        collidable=True,
        tags=["wall"],
    ),
    "flag": Sprite(
        pixels=[[12]],
        name="flag",
        visible=True,
        collidable=False,
        tags=["flag"],
    ),
}


class Ms02UI(RenderableUserDisplay):
    def __init__(self, n: int) -> None:
        self._n = n

    def update(self, n: int) -> None:
        self._n = n

    def render_interface(self, frame):
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return frame
        h, w = frame.shape
        for i in range(min(self._n, 15)):
            frame[h - 2, 1 + i] = 12
        return frame


def make_level(grid_size, player_pos, goal_pos, mine_coords, wall_coords, difficulty):
    sprite_list = [
        sprites["player"].clone().set_position(*player_pos),
        sprites["goal"].clone().set_position(*goal_pos),
    ]
    for mc in mine_coords:
        sprite_list.append(sprites["mine"].clone().set_position(*mc))
    for wc in wall_coords:
        sprite_list.append(sprites["wall"].clone().set_position(*wc))
    for y in range(grid_size[1]):
        for x in range(grid_size[0]):
            if (x, y) in wall_coords or (x, y) == player_pos or (x, y) == goal_pos:
                continue
            if (x, y) in mine_coords:
                continue
            sprite_list.append(sprites["tile"].clone().set_position(x, y))
    return Level(
        sprites=sprite_list,
        grid_size=grid_size,
        data={"difficulty": difficulty},
    )


levels = [
    make_level((8, 8), (0, 0), (7, 7), [(2, 2), (3, 3), (4, 4)], [], 1),
    make_level((8, 8), (0, 3), (7, 3), [(3, 2), (3, 3), (3, 4), (4, 3)], [], 2),
    make_level((10, 10), (0, 0), (9, 9), [(2, 2), (2, 3), (3, 2), (5, 5), (5, 6)], [], 3),
    make_level((10, 10), (1, 1), (8, 8), [(4, y) for y in range(2, 8)], [], 4),
    make_level((12, 12), (0, 0), (11, 11), [(x, x) for x in range(2, 10)], [], 5),
]


class Ms02(AugmentedGame):
    CAMERA_W = 16
    CAMERA_H = 16

    def __init__(self, seed: int | None = None) -> None:
        # FIRST: super().__init__ seats level 0, which calls
        # on_set_level -> level_rng, and that needs the seed installed.
        self._init_augmentation(seed)
        # Set BEFORE super().__init__: it calls set_level(0) -> on_set_level ->
        # _randomize_mines, which consumes this RNG.
        self._rng = self.level_rng("layout")
        self._ui = Ms02UI(0)
        super().__init__(
            "ms02",
            levels,
            Camera(0, 0, 16, 16, BACKGROUND_COLOR, PADDING_COLOR, [self._ui]),
            False,
            1,
            [1, 2, 3, 4, 6],
        )

    def _randomize_mines(self, level: Level) -> None:
        """Relocate the mines to random cells (same count), keeping the player
        start and goal cells clear and guaranteeing a fully-safe COUNT-0 path
        from start to goal (every path cell has no adjacent mine, so the solution
        never needs a clue tile). Then rebuild the tile layer, since ms02 leaves
        mine cells tile-less (they render as background)."""
        gw, gh = level.grid_size
        player = level.get_sprites_by_tag("player")[0]
        goal = level.get_sprites_by_tag("goal")[0]
        walls = {(w.x, w.y) for w in level.get_sprites_by_tag("wall")}
        mines = level.get_sprites_by_tag("mine")
        n = len(mines)
        start = (player.x, player.y)
        goal_pos = (goal.x, goal.y)

        candidates = [
            (x, y)
            for x in range(gw)
            for y in range(gh)
            if (x, y) not in walls and (x, y) != start and (x, y) != goal_pos
        ]

        def count0_path_exists(mine_set):
            def is_safe(cell):
                if cell in mine_set or cell in walls:
                    return False
                cx, cy = cell
                for ddx in (-1, 0, 1):
                    for ddy in (-1, 0, 1):
                        if (ddx or ddy) and (cx + ddx, cy + ddy) in mine_set:
                            return False
                return True

            if not is_safe(start) or not is_safe(goal_pos):
                return False
            seen = {start}
            dq = deque([start])
            while dq:
                x, y = dq.popleft()
                if (x, y) == goal_pos:
                    return True
                for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    nxt = (x + dx, y + dy)
                    if (
                        0 <= nxt[0] < gw
                        and 0 <= nxt[1] < gh
                        and nxt not in seen
                        and is_safe(nxt)
                    ):
                        seen.add(nxt)
                        dq.append(nxt)
            return False

        chosen = None
        for _ in range(1000):
            cells = self._rng.sample(candidates, n)
            if count0_path_exists(set(cells)):
                chosen = cells
                break
        if chosen is None:  # extremely unlikely; keep the clean layout
            chosen = [(m.x, m.y) for m in mines]
        for m, pos in zip(mines, chosen):
            m.set_position(*pos)

        # Rebuild the tile layer to match the new mine positions (ms02 places no
        # tile on wall / player / goal / mine cells).
        new_mines = set(chosen)
        for t in list(level.get_sprites_by_tag("tile")):
            level.remove_sprite(t)
        for x in range(gw):
            for y in range(gh):
                cell = (x, y)
                if (
                    cell in walls
                    or cell == start
                    or cell == goal_pos
                    or cell in new_mines
                ):
                    continue
                level.add_sprite(sprites["tile"].clone().set_position(x, y))

    def on_set_level(self, level: Level) -> None:
        # Re-key per level so the board is a pure function of
        # (seed, level), identical whether this level was jumped to
        # or played into. See AugmentedGame.level_rng.
        self._rng = self.level_rng("layout")
        # Rebuild from the clean descriptor, then randomise the mine positions
        # deterministically from self._rng.
        self._levels[self._current_level_index] = (
            self._clean_levels[self._current_level_index].clone()
        )
        lvl = self.current_level
        self._randomize_mines(lvl)
        self._player = lvl.get_sprites_by_tag("player")[0]
        self._minefield: set[tuple[int, int]] = set()
        for m in lvl.get_sprites_by_tag("mine"):
            self._minefield.add((m.x, m.y))
        self._revealed: set[tuple[int, int]] = set()
        self._flags: set[tuple[int, int]] = set()
        gw, gh = lvl.grid_size
        self._mine_count = {}
        for y in range(gh):
            for x in range(gw):
                if (x, y) not in self._minefield:
                    c = 0
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dx == dy == 0:
                                continue
                            if (x + dx, y + dy) in self._minefield:
                                c += 1
                    self._mine_count[(x, y)] = c
        self._ui.update(0)

    def _grid_to_frame_pixel(self, gx: int, gy: int) -> tuple[int, int]:
        cw, ch = self.CAMERA_W, self.CAMERA_H
        scale = min(64 // cw, 64 // ch)
        x_pad = (64 - cw * scale) // 2
        y_pad = (64 - ch * scale) // 2
        return gx * scale + scale // 2 + x_pad, gy * scale + scale // 2 + y_pad

    def _get_clue_color(self, count: int) -> int:
        if count == 0:
            return 1
        if count == 1:
            return 8
        if count == 2:
            return 11
        if count == 3:
            return 14
        return 15

    def step(self) -> None:
        if self.action.id == GameAction.ACTION6:
            px = int(self.action.data.get("x", 0))
            py = int(self.action.data.get("y", 0))
            hit = self.camera.display_to_grid(px, py)
            if hit is None:
                self.complete_action()
                return
            gx, gy = int(hit[0]), int(hit[1])
            if (gx, gy) in self._revealed:
                self.complete_action()
                return
            if (gx, gy) in self._flags:
                for s in list(self.current_level.get_sprites_by_tag("flag")):
                    if s.x == gx and s.y == gy:
                        self.current_level.remove_sprite(s)
                        break
                self._flags.discard((gx, gy))
                self._ui.update(len(self._flags))
                self.complete_action()
                return
            if (gx, gy) not in self._minefield:
                self.lose()
                self.complete_action()
                return
            self.current_level.add_sprite(sprites["flag"].clone().set_position(gx, gy))
            self._flags.add((gx, gy))
            self._ui.update(len(self._flags))
            self.complete_action()
            return

        dx = dy = 0
        if self.action.id.value == 1:
            dy = -1
        elif self.action.id.value == 2:
            dy = 1
        elif self.action.id.value == 3:
            dx = -1
        elif self.action.id.value == 4:
            dx = 1
        else:
            self.complete_action()
            return

        nx, ny = self._player.x + dx, self._player.y + dy
        gw, gh = self.current_level.grid_size
        if not (0 <= nx < gw and 0 <= ny < gh):
            self.complete_action()
            return
        sp = self.current_level.get_sprite_at(nx, ny, ignore_collidable=True)
        if sp and "wall" in sp.tags:
            self.complete_action()
            return
        if (nx, ny) in self._minefield:
            self.lose()
            self.complete_action()
            return
        if (nx, ny) in self._flags:
            self.complete_action()
            return

        self._player.set_position(nx, ny)
        if (nx, ny) not in self._revealed:
            self._revealed.add((nx, ny))
            cnt = self._mine_count.get((nx, ny), 0)
            for s in self.current_level._sprites:
                if s.x == nx and s.y == ny and "tile" in s.tags:
                    s.color_remap(s.pixels[0][0], self._get_clue_color(cnt))
                    break

        if sp and "goal" in sp.tags:
            self.next_level()

        self.complete_action()
