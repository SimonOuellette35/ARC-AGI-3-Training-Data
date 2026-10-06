"""Number fuse: pick up numbered tokens in strictly descending order (highest first); wrong order loses.

Positions are randomised per-seed in ``on_set_level`` from the instance RNG (so
far: level 1's aligned row is translated vertically as one). Live play uses an
unseeded RNG; the training generator seeds ``self._rng`` with ``nu01:<seed>:<level>``
so a given seed reproduces the same board byte-for-byte."""

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


# Distinct token colors (match token_sprite) so HUD dots map to board tokens.
TOKEN_COLOR = {3: 11, 2: 12, 1: 10}


class Nu01UI(RenderableUserDisplay):
    def __init__(self, need: int, token_colors: dict[int, int] | None = None) -> None:
        self._need = need
        self._token_colors = dict(token_colors) if token_colors else dict(TOKEN_COLOR)

    def update(
        self, need: int, token_colors: dict[int, int] | None = None
    ) -> None:
        self._need = need
        if token_colors is not None:
            self._token_colors = {v: token_colors[v] for v in (3, 2, 1) if v in token_colors}

    def render_interface(self, frame):
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return frame
        h, w = frame.shape
        for i in range(3):
            val = 3 - i
            if self._need == 0:
                c = 14
            elif val == self._need:
                c = self._token_colors.get(val, TOKEN_COLOR[val])
            elif val > self._need:
                c = 3
            else:
                c = 2
            frame[h - 2, 2 + i] = c
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


def token_sprite(n: int) -> Sprite:
    c = TOKEN_COLOR.get(n, 10)
    return Sprite(
        pixels=[[c]],
        name=f"tok{n}",
        visible=True,
        collidable=False,
        tags=["token", f"n{n}"],
    )


def mk(sl, d: int):
    return Level(sprites=sl, grid_size=(10, 10), data={"difficulty": d})


levels = [
    mk(
        [
            sprites["player"].clone().set_position(1, 5),
            token_sprite(3).clone().set_position(3, 5),
            token_sprite(2).clone().set_position(5, 5),
            token_sprite(1).clone().set_position(7, 5),
            sprites["goal"].clone().set_position(9, 5),
        ],
        1,
    ),
    mk(
        [
            sprites["player"].clone().set_position(2, 2),
            token_sprite(1).clone().set_position(4, 2),
            token_sprite(3).clone().set_position(6, 4),
            token_sprite(2).clone().set_position(3, 6),
            sprites["goal"].clone().set_position(8, 8),
        ],
        2,
    ),
    mk(
        [
            sprites["player"].clone().set_position(0, 0),
            token_sprite(3).clone().set_position(2, 1),
            token_sprite(2).clone().set_position(4, 3),
            token_sprite(1).clone().set_position(6, 5),
            sprites["goal"].clone().set_position(9, 9),
        ],
        3,
    ),
    mk(
        [
            sprites["player"].clone().set_position(5, 5),
            token_sprite(3).clone().set_position(5, 2),
            token_sprite(2).clone().set_position(8, 5),
            token_sprite(1).clone().set_position(5, 8),
            sprites["goal"].clone().set_position(1, 5),
        ],
        4,
    ),
    mk(
        [
            sprites["player"].clone().set_position(1, 1),
            token_sprite(3).clone().set_position(7, 1),
            token_sprite(2).clone().set_position(1, 7),
            token_sprite(1).clone().set_position(7, 7),
            sprites["goal"].clone().set_position(4, 4),
        ]
        + [sprites["wall"].clone().set_position(4, y) for y in (0, 1, 8, 9)],
        5,
    ),
]

BACKGROUND_COLOR = 5
PADDING_COLOR = 4
GOAL_COLOR = 14
WALL_COLOR = 3
_PLAYER_BASE = 9
# Token collection order (highest value first): the puzzle is solved by stepping
# on the value-3 token, then value-2, then value-1.
_TOKEN_ORDER = (3, 2, 1)
# Colours for the randomised agent + three tokens. Excludes the fixed goal (14),
# wall (3), background (5) and padding (4) colours and 0; the four are sampled
# distinct so every token/agent is visible and unambiguous.
_COLOR_POOL = [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 15]


def _token_value(sp: Sprite) -> int | None:
    for t in sp.tags:
        if t.startswith("n") and t[1:].isdigit():
            return int(t[1:])
    return None


class Nu01(AugmentedGame):
    def __init__(self, seed: int | None = None) -> None:
        # FIRST: super().__init__ seats level 0, which calls
        # on_set_level -> level_rng, and that needs the seed installed.
        self._init_augmentation(seed)
        # Set BEFORE super().__init__: it calls set_level(0) -> on_set_level ->
        # _randomize_level, which consumes these RNGs.
        self._rng = self.level_rng("layout")  # per-level (positions)
        # Per-EPISODE colour scheme: drawn from a separate RNG and cached, so the
        # token/agent colours stay identical across every level of one episode.
        # The generator seeds this level-independently (only from the episode
        # seed); live play draws a fresh scheme per playthrough.
        self._color_rng = self.level_rng("colors")
        self._color_scheme: dict | None = None
        self._ui = Nu01UI(3)
        super().__init__(
            "nu01",
            levels,
            Camera(0, 0, 16, 16, BACKGROUND_COLOR, PADDING_COLOR, [self._ui]),
            False,
            1,
            [1, 2, 3, 4],
        )

    def _ensure_color_scheme(self) -> None:
        """Draw (once) the episode-wide colour scheme: distinct random colours for
        the agent and for the three tokens (in collection order 3->2->1), all
        different from the goal (14), walls (3) and background (5)."""
        if self._color_scheme is not None:
            return
        agent, c3, c2, c1 = self._color_rng.sample(_COLOR_POOL, 4)
        self._color_scheme = {"agent": agent, 3: c3, 2: c2, 1: c1}

    def _randomize_line_y(self, level: Level) -> None:
        """Level 1: player | tokens | goal all share one row (walking right
        collects 3->2->1 in order). Translate all of them to a random row
        together (x kept), so they stay aligned -- a pure vertical shift keeps the
        level solvable."""
        ry = self._rng.randint(0, level.grid_size[1] - 1)
        for s in level.get_sprites():
            s.set_position(s.x, ry)

    @staticmethod
    def _order_solvable(start, goal, tok_pos, walls, gw, gh) -> bool:
        """Is there a route that collects the tokens in order 3->2->1 and then
        reaches the goal, without stepping on a not-yet-needed token (lethal) or a
        wall? BFS over (x, y, next_value) -- mirrors the training solver exactly."""
        by_cell = {pos: v for v, pos in tok_pos.items()}
        start_state = (start[0], start[1], 3)
        seen = {start_state}
        dq = deque([start_state])
        while dq:
            x, y, nv = dq.popleft()
            if nv == 0 and (x, y) == goal:
                return True
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < gw and 0 <= ny < gh) or (nx, ny) in walls:
                    continue
                nnv = nv
                if (nx, ny) in by_cell:
                    v = by_cell[(nx, ny)]
                    if v != nv:
                        continue  # wrong-order token: lethal, cannot enter
                    nnv = nv - 1
                ns = (nx, ny, nnv)
                if ns not in seen:
                    seen.add(ns)
                    dq.append(ns)
        return False

    def _randomize_positions(self, level: Level) -> None:
        """Levels 2-5: place the player, the three tokens and the goal on random
        distinct free cells (walls stay fixed), keeping the puzzle solvable in the
        strict 3->2->1 collection order (BFS-verified; resampled until solvable)."""
        gw, gh = level.grid_size
        walls = {(w.x, w.y) for w in level.get_sprites_by_tag("wall")}
        free = [
            (x, y)
            for x in range(gw)
            for y in range(gh)
            if (x, y) not in walls
        ]
        player = level.get_sprites_by_tag("player")[0]
        goal = level.get_sprites_by_tag("goal")[0]
        tok_by_val = {
            _token_value(t): t for t in level.get_sprites_by_tag("token")
        }
        vals = sorted(tok_by_val)  # e.g. [1, 2, 3]

        for _ in range(400):
            cells = self._rng.sample(free, 2 + len(vals))
            p, g = cells[0], cells[1]
            tok_pos = {v: cells[2 + i] for i, v in enumerate(vals)}
            if self._order_solvable(p, g, tok_pos, walls, gw, gh):
                player.set_position(*p)
                goal.set_position(*g)
                for v, pos in tok_pos.items():
                    tok_by_val[v].set_position(*pos)
                return
        # extremely unlikely; keep the clean positions

    def _randomize_level(self) -> None:
        self._ensure_color_scheme()
        level = self.current_level
        if self._current_level_index == 0:  # level 1
            self._randomize_line_y(level)
        else:  # levels 2-5
            self._randomize_positions(level)

        scheme = self._color_scheme
        for tok in level.get_sprites_by_tag("token"):
            v = _token_value(tok)
            if v in scheme:
                tok.color_remap(TOKEN_COLOR[v], scheme[v])
        for p in level.get_sprites_by_tag("player"):
            p.color_remap(_PLAYER_BASE, scheme["agent"])

    def full_reset(self) -> None:
        # A full reset restarts the whole game -> a new episode -> a new colour
        # scheme (drawn on the next set_level).
        self._color_scheme = None
        super().full_reset()

    def on_set_level(self, level: Level) -> None:
        # Re-key per level so the board is a pure function of
        # (seed, level), identical whether this level was jumped to
        # or played into. See AugmentedGame.level_rng.
        self._rng = self.level_rng("layout")
        self._color_rng = self.level_rng("colors")
        # Rebuild from the clean descriptor, then randomise deterministically
        # from the RNGs.
        self._levels[self._current_level_index] = (
            self._clean_levels[self._current_level_index].clone()
        )
        self._randomize_level()
        self._player = self.current_level.get_sprites_by_tag("player")[0]
        self._goal = self.current_level.get_sprites_by_tag("goal")[0]
        self._tokens = list(self.current_level.get_sprites_by_tag("token"))
        self._next_val = 3
        self._ui.update(self._next_val, self._color_scheme)

    def step(self) -> None:
        dx = dy = 0
        if self.action.id == GameAction.ACTION1:
            dy = -1
        elif self.action.id == GameAction.ACTION2:
            dy = 1
        elif self.action.id == GameAction.ACTION3:
            dx = -1
        elif self.action.id == GameAction.ACTION4:
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

        if sp and "token" in sp.tags:
            v = _token_value(sp)
            if v is None or v != self._next_val:
                self.lose()
                self.complete_action()
                return
            self.current_level.remove_sprite(sp)
            if sp in self._tokens:
                self._tokens.remove(sp)
            self._next_val -= 1
            self._ui.update(self._next_val)

        self._player.set_position(nx, ny)

        if self._next_val == 0 and self._player.x == self._goal.x and self._player.y == self._goal.y:
            self.next_level()

        self.complete_action()
