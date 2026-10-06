"""
Sokoban
=======
Procedurally generated Sokoban puzzles. Push all crates onto the target squares.
Each run produces a fresh set of levels derived from the provided seed.

Generation method: start from the solved state (boxes on targets) and perform
n random forward Sokoban pushes to reach the starting configuration. Because
every step is a legal push, the puzzle is guaranteed solvable by replaying
those steps in reverse.

Color palette (index → hex):
  0  #FFFFFF — white     — open floor
  4  #333333 — near-black — wall
  9  #1E93FF — blue      — player
  11 #FFDC00 — yellow    — crate
  12 #FF851B — orange    — crate sitting on a target
  14 #4FCC30 — green     — target square
  5  #000000 — black     — letterbox border

Actions:
  ACTION1 (↑) — Move up
  ACTION2 (↓) — Move down
  ACTION3 (←) — Move left
  ACTION4 (→) — Move right
  ACTION7     — Undo last move

Win condition : All crates are on target squares.
Lose condition: Step budget exhausted (progress bar at bottom).
Levels        : 7 puzzles of increasing size and complexity (11x9 / 2 crates up to
               19x17 / 5 crates; see ``_LEVEL_CONFIGS``), with budgets
               120 / 220 / 380 / 580 / 820 / 1100 / 1450. Higher levels add interior
               wall obstacles that force narrow paths and create deadlock traps —
               pushing a crate into a corner it cannot escape will block your
               progress, and ACTION7 is then the only way out.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
from utils.arc_game import AugmentedGame
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

from utils.arc_game import clear_dynamic_sprites

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
C_FLOOR  = 0   # white
C_WALL   = 4   # near-black
C_TARGET = 14  # green
C_BOX    = 11  # yellow
C_BOX_ON = 12  # orange — crate on target
C_PLAYER = 9   # blue
C_BORDER = 5   # black

_STEP_BUDGETS = [120, 220, 380, 580, 820, 1100, 1450]


# ---------------------------------------------------------------------------
# Step counter HUD
# ---------------------------------------------------------------------------

class StepCounter(RenderableUserDisplay):
    """Progress bar rendered on the bottom row of the 64×64 frame."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.steps_remaining = max_steps

    def decrement(self) -> None:
        if self.steps_remaining > 0:
            self.steps_remaining -= 1

    def reset(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.steps_remaining = max_steps

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.max_steps == 0:
            return frame
        filled = round(64 * self.steps_remaining / self.max_steps)
        for x in range(64):
            frame[63, x] = 7 if x < filled else 4
        return frame


# (grid_width, grid_height, num_boxes, n_scramble_moves, n_interior_walls)
_LEVEL_CONFIGS = [
    (11,  9, 2,  30,  0),   # Level 1 — small,         2 crates, open room
    (13, 11, 2,  55,  5),   # Level 2 — medium,         2 crates, 5 interior walls
    (15, 13, 3,  90, 10),   # Level 3 — medium-large,   3 crates, 10 interior walls
    (17, 15, 3, 130, 16),   # Level 4 — large,           3 crates, 16 interior walls
    (17, 15, 4, 175, 20),   # Level 5 — large,           4 crates, 20 interior walls
    (19, 17, 4, 230, 24),   # Level 6 — extra-large,     4 crates, 24 interior walls
    (19, 17, 5, 300, 30),   # Level 7 — extra-large,     5 crates, 30 interior walls
]

# ---------------------------------------------------------------------------
# BFS helper
# ---------------------------------------------------------------------------

def _reachable(width: int, height: int, walls: list[list[bool]],
               start: tuple[int, int], blocked: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Positions the player can reach from *start* without moving any box."""
    sx, sy = start
    if walls[sy][sx] or start in blocked:
        return set()
    visited: set[tuple[int, int]] = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            p = (x + dx, y + dy)
            if p not in visited and 0 <= p[0] < width and 0 <= p[1] < height:
                if not walls[p[1]][p[0]] and p not in blocked:
                    visited.add(p)
                    q.append(p)
    return visited


# ---------------------------------------------------------------------------
# Level generator
# ---------------------------------------------------------------------------

def _add_interior_walls(walls: list[list[bool]], width: int, height: int,
                        rng: random.Random, n: int) -> None:
    """Add up to n interior walls while keeping floor connectivity.

    Walls are placed on interior cells (not on the perimeter rows/cols or their
    immediate neighbours) so there is always enough room to push crates along
    the edges.  Each candidate is rejected if it would disconnect the floor.
    """
    def _floor_connected() -> bool:
        floor = [(x, y) for y in range(1, height - 1) for x in range(1, width - 1)
                 if not walls[y][x]]
        if not floor:
            return False
        start = floor[0]
        visited = {start}
        q: deque[tuple[int, int]] = deque([start])
        while q:
            cx, cy = q.popleft()
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nb = (cx + dx, cy + dy)
                if nb not in visited and 0 <= nb[0] < width and 0 <= nb[1] < height:
                    if not walls[nb[1]][nb[0]]:
                        visited.add(nb)
                        q.append(nb)
        return len(visited) == len(floor)

    # Candidate cells: interior cells at least 2 away from perimeter so that
    # the corridors remain at least 1 cell wide.
    candidates = [(x, y) for y in range(2, height - 2) for x in range(2, width - 2)]
    rng.shuffle(candidates)
    added = 0
    for x, y in candidates:
        if added >= n:
            break
        if walls[y][x]:
            continue
        walls[y][x] = True
        if _floor_connected():
            added += 1
        else:
            walls[y][x] = False  # revert — would disconnect floor


def _generate(width: int, height: int, num_boxes: int,
              rng: random.Random, n_moves: int,
              n_interior_walls: int = 0) -> tuple[
                  list[list[bool]],      # walls[y][x]
                  list[tuple[int, int]], # target positions
                  list[tuple[int, int]], # box starting positions
                  tuple[int, int],       # player starting position
              ]:
    """
    Return a guaranteed-solvable level.

    Strategy:
      1. Build a rectangular room.
      2. Randomly scatter targets on interior floor cells.
      3. Place boxes on targets (= solved state).
      4. Pick a player start position adjacent to one box.
      5. Perform *n_moves* random legal forward Sokoban pushes to scramble
         the board. The result is the puzzle's starting state.
    """
    walls: list[list[bool]] = [
        [x == 0 or x == width - 1 or y == 0 or y == height - 1
         for x in range(width)]
        for y in range(height)
    ]

    # Add interior walls BEFORE placing targets/boxes so they land on open floor.
    if n_interior_walls > 0:
        _add_interior_walls(walls, width, height, rng, n_interior_walls)

    floor: list[tuple[int, int]] = [
        (x, y) for y in range(1, height - 1) for x in range(1, width - 1)
        if not walls[y][x]
    ]

    # --- place targets --------------------------------------------------
    rng.shuffle(floor)
    targets: list[tuple[int, int]] = floor[:num_boxes]
    boxes:   list[tuple[int, int]] = list(targets)      # start solved

    # --- place player adjacent to first box ----------------------------
    bx, by = boxes[0]
    candidates = [(bx + dx, by + dy) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))]
    rng.shuffle(candidates)
    player: tuple[int, int] = next(
        (p for p in candidates
         if 0 <= p[0] < width and 0 <= p[1] < height
         and not walls[p[1]][p[0]] and p not in boxes),
        next(c for c in floor if c not in boxes)
    )

    target_set = set(targets)

    def _frozen(test_boxes: list[tuple[int, int]]) -> bool:
        """Return True if any non-target box is completely unmovable."""
        tset = set(test_boxes)
        for tbx, tby in test_boxes:
            if (tbx, tby) in target_set:
                continue
            can_push = False
            for ddx, ddy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nbx2, nby2 = tbx + ddx, tby + ddy
                pf = (tbx - ddx, tby - ddy)
                if not (0 <= nbx2 < width and 0 <= nby2 < height):
                    continue
                if walls[nby2][nbx2] or (nbx2, nby2) in tset:
                    continue
                if not (0 <= pf[0] < width and 0 <= pf[1] < height):
                    continue
                if walls[pf[1]][pf[0]] or pf in tset:
                    continue
                can_push = True
                break
            if not can_push:
                return True
        return False

    # --- scramble by doing random forward pushes -----------------------
    for _ in range(n_moves):
        box_set = set(boxes)
        reach = _reachable(width, height, walls, player, box_set)

        valid: list[tuple[int, int, int, int, int]] = []   # (box_idx, bx,by, nbx,nby)
        for i, (bx, by) in enumerate(boxes):
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                push_from = (bx - dx, by - dy)  # player pushes from here
                nbx, nby  = bx + dx, by + dy    # box lands here

                if push_from not in reach:
                    continue
                if not (0 <= nbx < width and 0 <= nby < height):
                    continue
                if walls[nby][nbx]:
                    continue
                if (nbx, nby) in box_set:
                    continue
                # Reversibility check: the reverse push requires player at
                # (nbx+dx, nby+dy); that position must be passable and reachable
                # after the forward push (player moves to (bx,by)).
                rpx, rpy = nbx + dx, nby + dy
                if not (0 <= rpx < width and 0 <= rpy < height):
                    continue
                if walls[rpy][rpx]:
                    continue
                # Deadlock check: skip pushes that freeze any box off-target.
                test_boxes = list(boxes)
                test_boxes[i] = (nbx, nby)
                if _frozen(test_boxes):
                    continue
                # Full reachability check: after the push the player is at (bx,by);
                # verify they can reach (rpx,rpy) given the new box positions.
                new_box_set = set(test_boxes)
                post_push_player = (bx, by)
                rev_reach = _reachable(width, height, walls, post_push_player, new_box_set)
                if (rpx, rpy) not in rev_reach:
                    continue
                valid.append((i, bx, by, nbx, nby))

        if not valid:
            break

        i, bx, by, nbx, nby = rng.choice(valid)
        boxes[i] = (nbx, nby)
        player = (bx, by)   # player ends up where box was

    return walls, targets, boxes, player


# ---------------------------------------------------------------------------
# Level builder (walls + targets only; player/boxes added on_set_level)
# ---------------------------------------------------------------------------

def _build_level(walls: list[list[bool]],
                 targets: list[tuple[int, int]]) -> Level:
    height = len(walls)
    width  = len(walls[0])
    sprites: list[Sprite] = []
    for y in range(height):
        for x in range(width):
            if walls[y][x]:
                sprites.append(
                    Sprite(pixels=[[C_WALL]], name="wall", collidable=True, layer=-1)
                    .set_position(x, y)
                )
    for idx, (tx, ty) in enumerate(targets):
        sprites.append(
            Sprite(pixels=[[C_TARGET]], name=f"target_{idx}",
                   collidable=False, tags=["target"], layer=0)
            .set_position(tx, ty)
        )
    return Level(sprites=sprites, grid_size=(width, height))


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class Sokoban(AugmentedGame):

    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: the base draws this level's rotation
        rng = random.Random(seed)
        self._step_counter = StepCounter(_STEP_BUDGETS[0])

        # Generate all levels before calling super().__init__ so that
        # on_set_level (called from super) can already access _level_data.
        self._level_data: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
        levels: list[Level] = []

        for width, height, num_boxes, n_moves, n_walls in _LEVEL_CONFIGS:
            walls, targets, boxes, player = _generate(
                width, height, num_boxes, rng, n_moves, n_walls
            )
            levels.append(_build_level(walls, targets))
            self._level_data.append((player, boxes))

        self._history: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []

        super().__init__(
            game_id="sokoban",
            levels=levels,
            camera=Camera(
                background=C_FLOOR,
                letter_box=C_BORDER,
                interfaces=[self._step_counter],
            ),
            available_actions=[1, 2, 3, 4, 7],
        )

    # ------------------------------------------------------------------
    # Sprite query helpers
    # ------------------------------------------------------------------

    def _wall_at(self, x: int, y: int) -> bool:
        return any(s.x == x and s.y == y and s.name == "wall"
                   for s in self.current_level.get_sprites())

    def _box_at(self, x: int, y: int) -> Sprite | None:
        return next(
            (s for s in self.current_level.get_sprites_by_tag("box")
             if s.x == x and s.y == y),
            None
        )

    def _is_on_target(self, box: Sprite) -> bool:
        return any(t.x == box.x and t.y == box.y
                   for t in self.current_level.get_sprites_by_tag("target"))

    def _refresh_box_colors(self) -> None:
        for box in self.current_level.get_sprites_by_tag("box"):
            color = C_BOX_ON if self._is_on_target(box) else C_BOX
            box.pixels = np.array([[color]], dtype=np.int8)

    def _snapshot(self) -> tuple[tuple[int, int], list[tuple[int, int]]]:
        p = self.current_level.get_sprites_by_tag("player")[0]
        return (p.x, p.y), [(b.x, b.y) for b in
                             self.current_level.get_sprites_by_tag("box")]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_set_level(self, level: Level) -> None:
        # Seating a level twice re-runs this hook on the same persistent Level.
        clear_dynamic_sprites(level, tags=["player", "box"])
        self._history.clear()
        self._step_counter.reset(_STEP_BUDGETS[self._current_level_index])
        player_pos, box_positions = self._level_data[self._current_level_index]
        level.add_sprite(
            Sprite(pixels=[[C_PLAYER]], name="player",
                   collidable=False, tags=["player"], layer=2)
            .set_position(*player_pos)
        )
        for i, (bx, by) in enumerate(box_positions):
            level.add_sprite(
                Sprite(pixels=[[C_BOX]], name=f"box_{i}",
                       collidable=False, tags=["box"], layer=1)
                .set_position(bx, by)
            )
        self._refresh_box_colors()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    _DELTAS: dict[int, tuple[int, int]] = {
        GameAction.ACTION1: (0, -1),
        GameAction.ACTION2: (0,  1),
        GameAction.ACTION3: (-1, 0),
        GameAction.ACTION4: (1,  0),
    }

    def step(self) -> None:
        player = self.current_level.get_sprites_by_tag("player")[0]

        self._step_counter.decrement()

        # --- Undo -------------------------------------------------------
        if self.action.id == GameAction.ACTION7:
            if self._history:
                (px, py), bpos = self._history.pop()
                player.set_position(px, py)
                for box, (bx, by) in zip(
                    self.current_level.get_sprites_by_tag("box"), bpos
                ):
                    box.set_position(bx, by)
                self._refresh_box_colors()
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        dx, dy = self._DELTAS.get(self.action.id, (0, 0))
        if not (dx or dy):
            if not self._step_counter.steps_remaining:
                self.lose()
            self.complete_action()
            return

        nx, ny = player.x + dx, player.y + dy

        if self._wall_at(nx, ny):
            if not self._step_counter.steps_remaining:
                self.lose()
            self.complete_action()
            return

        box = self._box_at(nx, ny)
        if box is not None:
            bnx, bny = box.x + dx, box.y + dy
            if self._wall_at(bnx, bny) or self._box_at(bnx, bny) is not None:
                if not self._step_counter.steps_remaining:
                    self.lose()
                self.complete_action()
                return
            self._history.append(self._snapshot())
            box.set_position(bnx, bny)
            player.set_position(nx, ny)
            self._refresh_box_colors()
        else:
            self._history.append(self._snapshot())
            player.set_position(nx, ny)

        # --- Win check --------------------------------------------------
        boxes = self.current_level.get_sprites_by_tag("box")
        if all(self._is_on_target(b) for b in boxes):
            self.next_level()
            self.complete_action()
            return

        if not self._step_counter.steps_remaining:
            self.lose()

        self.complete_action()
