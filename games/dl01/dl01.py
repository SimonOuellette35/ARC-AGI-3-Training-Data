"""Delay line: ACTION1–4 enqueue a cardinal move (FIFO, max 3 pending). Each step runs the oldest queued move first, then enqueues the current action. ACTION5 clears the queue."""

from __future__ import annotations

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
CAM = 12
GOAL_COLOR = 14
HAZARD_COLOR = 8
# Base pixel colours of the sprite templates (remapped per level).
_PLAYER_BASE = 9
_WALL_BASE = 3
# Colours for the randomised background / agent / wall. Excludes the fixed goal
# (14), hazard (8) and padding (4) colours and 0; the three are sampled distinct
# so no cell type is confusable with another or the background.
_COLOR_POOL = [1, 2, 3, 6, 7, 9, 10, 11, 12, 13, 15]


class Dl01UI(RenderableUserDisplay):
    _DIR_COLOR = {(0, -1): 10, (0, 1): 6, (-1, 0): 11, (1, 0): 9}

    def __init__(self) -> None:
        self._q: list[tuple[int, int]] = []
        self._bump = 0

    def update(
        self,
        q: list[tuple[int, int]],
        *,
        bump: bool = False,
    ) -> None:
        self._q = list(q)
        if bump:
            self._bump = 6

    def render_interface(self, frame):
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return frame
        h, w = frame.shape
        base_x = 2
        for slot in range(3):
            cx = base_x + slot * 5
            if slot < len(self._q):
                dx, dy = self._q[slot]
                c = self._DIR_COLOR.get((dx, dy), 11)
                frame[h - 2, cx] = c
                if dy == -1:
                    frame[h - 3, cx] = c
                elif dy == 1:
                    frame[h - 1, cx] = c
                elif dx == -1:
                    frame[h - 2, cx - 1] = c
                elif dx == 1:
                    frame[h - 2, cx + 1] = c
            else:
                frame[h - 2, cx] = 3
        if self._bump > 0:
            frame[h - 2, w - 3] = 8
            self._bump -= 1
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
    "hazard": Sprite(
        pixels=[[8]],
        name="hazard",
        visible=True,
        collidable=True,
        tags=["hazard"],
    ),
}


def mk(
    player: tuple[int, int],
    goal: tuple[int, int],
    walls: list[tuple[int, int]],
    hazards: list[tuple[int, int]],
    diff: int,
) -> Level:
    sl: list[Sprite] = [
        sprites["player"].clone().set_position(player[0], player[1]),
        sprites["goal"].clone().set_position(goal[0], goal[1]),
    ]
    for wx, wy in walls:
        sl.append(sprites["wall"].clone().set_position(wx, wy))
    for hx, hy in hazards:
        sl.append(sprites["hazard"].clone().set_position(hx, hy))
    return Level(
        sprites=sl,
        grid_size=(CAM, CAM),
        data={"difficulty": diff},
    )


levels = [
    mk((1, 1), (10, 10), [], [], 1),
    mk((1, 1), (10, 10), [(6, y) for y in range(12) if y != 6], [], 2),
    # Two-cell gap in the x=5 wall so (5,6) is reachable from (5,7); hazards off the gap.
    mk((2, 2), (9, 9), [(5, y) for y in range(12) if y not in (6, 7)], [(3, 3), (9, 3)], 3),
    mk((0, 6), (11, 6), [(x, 5) for x in range(12) if x != 6], [], 4),
    mk((1, 1), (10, 10), [(x, x) for x in range(12) if x in (3, 8)], [(3, 8), (8, 3)], 5),
]


class Dl01(AugmentedGame):
    def __init__(self, seed: int | None = None) -> None:
        # FIRST: super().__init__ seats level 0, which calls
        # on_set_level -> level_rng, and that needs the seed installed.
        self._init_augmentation(seed)
        # Set BEFORE super().__init__: it calls set_level(0) -> on_set_level ->
        # _randomize_level, which consumes this RNG.
        self._rng = self.level_rng("layout")
        self._ui = Dl01UI()
        self._q: deque[tuple[int, int]] = deque()
        super().__init__(
            "dl01",
            levels,
            Camera(0, 0, CAM, CAM, BACKGROUND_COLOR, PADDING_COLOR, [self._ui]),
            False,
            1,
            [1, 2, 3, 4, 5],
        )

    def _rotate_board(self, level: Level) -> None:
        """Rotate the ENTIRE board (player, goal, walls, hazards) by a random
        number of 90-degree turns. The delay-line mechanic treats all four
        directions symmetrically, so any rotation preserves solvability (square
        grid only)."""
        gw, gh = level.grid_size
        if gw != gh:
            return
        n = gw
        k = self._rng.randint(0, 3)
        if k == 0:
            return
        for s in level.get_sprites():
            x, y = s.x, s.y
            for _ in range(k):  # each turn: 90 deg clockwise
                x, y = n - 1 - y, x
            s.set_position(x, y)

    def _flip_board(self, level: Level) -> None:
        """Randomly mirror the ENTIRE board horizontally and/or vertically. The
        mechanic is direction-symmetric, so flips preserve solvability."""
        gw, gh = level.grid_size
        fx = self._rng.random() < 0.5
        fy = self._rng.random() < 0.5
        if not (fx or fy):
            return
        for s in level.get_sprites():
            x = gw - 1 - s.x if fx else s.x
            y = gh - 1 - s.y if fy else s.y
            s.set_position(x, y)

    def _randomize_walls(self, level: Level) -> None:
        """Levels 2/3/4: the wall is a single straight line (a full column or row)
        with a hole. Move the hole to a random spot along the line and translate
        the whole line laterally by a random amount. The hole size is preserved,
        so the two sides stay connected through it."""
        walls = level.get_sprites_by_tag("wall")
        if not walls:
            return
        gw, gh = level.grid_size
        r = self._rng
        xs = {w.x for w in walls}
        ys = {w.y for w in walls}

        if len(xs) == 1:  # vertical wall spanning rows, hole = missing rows
            hole_size = gh - len(walls)
            new_x = r.randint(1, gw - 2)  # lateral shift, kept interior
            hole_start = r.randint(0, gh - hole_size)
            hole = set(range(hole_start, hole_start + hole_size))
            new_cells = [(new_x, y) for y in range(gh) if y not in hole]
        elif len(ys) == 1:  # horizontal wall spanning columns
            hole_size = gw - len(walls)
            new_y = r.randint(1, gh - 2)
            hole_start = r.randint(0, gw - hole_size)
            hole = set(range(hole_start, hole_start + hole_size))
            new_cells = [(x, new_y) for x in range(gw) if x not in hole]
        else:
            return  # not a simple line wall -- leave untouched

        for w in list(walls):
            level.remove_sprite(w)
        for x, y in new_cells:
            level.add_sprite(sprites["wall"].clone().set_position(x, y))

    def _randomize_positions(self, level: Level) -> None:
        """Place the agent start and goal on random free cells (off walls and
        hazards) with a guaranteed cardinal path between them (BFS-verified;
        resampled until one exists). Done before rotate/flip -- an isometry
        preserves the path, and the delay-line solver works for any path."""
        gw, gh = level.grid_size
        blocked = {(s.x, s.y) for s in level.get_sprites_by_tag("wall")}
        blocked |= {(s.x, s.y) for s in level.get_sprites_by_tag("hazard")}
        free = [
            (x, y)
            for x in range(gw)
            for y in range(gh)
            if (x, y) not in blocked
        ]
        player = level.get_sprites_by_tag("player")[0]
        goal = level.get_sprites_by_tag("goal")[0]

        def path_exists(start, target):
            seen = {start}
            dq = deque([start])
            while dq:
                x, y = dq.popleft()
                if (x, y) == target:
                    return True
                for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    nxt = (x + dx, y + dy)
                    if (
                        0 <= nxt[0] < gw
                        and 0 <= nxt[1] < gh
                        and nxt not in blocked
                        and nxt not in seen
                    ):
                        seen.add(nxt)
                        dq.append(nxt)
            return False

        for _ in range(400):
            p, g = self._rng.sample(free, 2)
            if path_exists(p, g):
                player.set_position(*p)
                goal.set_position(*g)
                return
        # extremely unlikely; keep the clean positions

    def _randomize_level(self) -> None:
        """Randomise the wall hole + lateral position (levels 2/3/4), then the
        agent/goal positions, then apply a whole-board rotation + flips, then
        recolour the background / agent / walls. The goal (14) and hazards (8)
        keep their fixed colours; all cell types are distinct."""
        level = self.current_level
        if self._current_level_index in (1, 2, 3):  # levels 2, 3, 4
            self._randomize_walls(level)
        self._randomize_positions(level)
        self._rotate_board(level)
        self._flip_board(level)

        bg, agent, wall = self._rng.sample(_COLOR_POOL, 3)
        self.camera.background = bg
        for s in level.get_sprites_by_tag("player"):
            s.color_remap(_PLAYER_BASE, agent)
        for s in level.get_sprites_by_tag("wall"):
            s.color_remap(_WALL_BASE, wall)
        # goal keeps GOAL_COLOR (14), hazards keep HAZARD_COLOR (8)

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
        self._q.clear()
        self._ui.update(list(self._q))

    def _apply_move(self, dx: int, dy: int) -> bool:
        nx, ny = self._player.x + dx, self._player.y + dy
        gw, gh = self.current_level.grid_size
        if not (0 <= nx < gw and 0 <= ny < gh):
            return False
        sp = self.current_level.get_sprite_at(nx, ny, ignore_collidable=True)
        if sp and ("wall" in sp.tags or "hazard" in sp.tags):
            return False
        self._player.set_position(nx, ny)
        g = self.current_level.get_sprites_by_tag("goal")[0]
        if self._player.x == g.x and self._player.y == g.y:
            self.next_level()
        return True

    def _sync_queue_ui(self, *, bump: bool = False) -> None:
        self._ui.update(list(self._q), bump=bump)

    def step(self) -> None:
        aid = self.action.id

        if aid == GameAction.ACTION5:
            self._q.clear()
            self._sync_queue_ui()
            self.complete_action()
            return

        blocked = False
        if self._q:
            dx, dy = self._q.popleft()
            if not self._apply_move(dx, dy):
                blocked = True

        if aid == GameAction.ACTION1:
            self._q.append((0, -1))
        elif aid == GameAction.ACTION2:
            self._q.append((0, 1))
        elif aid == GameAction.ACTION3:
            self._q.append((-1, 0))
        elif aid == GameAction.ACTION4:
            self._q.append((1, 0))

        while len(self._q) > 3:
            self._q.popleft()

        self._sync_queue_ui(bump=blocked)
        self.complete_action()
