"""
Klotski (Block Slide)
=====================
Slide the red 2×2 target block down to the green exit. Other blocks can be
moved to clear the path. Cycle through selectable blocks with ACTION5.

  ACTION1 (↑) — slide selected block up
  ACTION2 (↓) — slide selected block down
  ACTION3 (←) — slide selected block left
  ACTION4 (→) — slide selected block right
  ACTION5     — cycle to next block
  ACTION7     — undo last slide

Color key:
  8  (#F93C31) — target block        7  (#FF7BCC) — selected block
  0  (#FFFFFF) — floor               4  (#333333) — wall
  14 (#4FCC30) — exit                other colors — other blocks

Win condition: target block's bottom edge reaches the exit row.
Lose condition: step budget exhausted (progress bar at bottom).
Levels: 4 randomly generated puzzles; budgets 80 / 120 / 200 / 300.
"""

from __future__ import annotations

import random
from copy import deepcopy

import numpy as np
from arcengine import (
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)
from utils.arc_game import AugmentedGame

C_FLOOR    = 0    # white
C_WALL     = 4    # near-black
C_EXIT     = 14   # green
C_TARGET   = 8    # red — target block
C_SELECTED = 7    # light pink — selected non-target block
C_BORDER   = 5    # black
_OTHER_COLORS = [9, 11, 12, 15, 6, 10]

# (board_w, board_h, num_other_blocks, scramble_moves, min_bfs_depth)
_CONFIGS = [
    (4, 5, 3,  40,  4),
    (4, 5, 4,  70,  8),
    (5, 6, 5, 110, 12),
    (5, 6, 6, 160, 16),
    (5, 6, 7, 230, 20),
    (6, 7, 7, 310, 25),
    (6, 7, 8, 420, 30),
]

_STEP_BUDGETS = [80, 120, 200, 300, 450, 620, 850]


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


# ---------------------------------------------------------------------------
# Block dataclass
# ---------------------------------------------------------------------------

class Block:
    __slots__ = ("x", "y", "w", "h", "is_target")

    def __init__(self, x: int, y: int, w: int, h: int, is_target: bool = False) -> None:
        self.x, self.y, self.w, self.h, self.is_target = x, y, w, h, is_target

    def cells(self) -> list[tuple[int, int]]:
        return [(self.x + dx, self.y + dy)
                for dx in range(self.w) for dy in range(self.h)]

    def copy(self) -> "Block":
        return Block(self.x, self.y, self.w, self.h, self.is_target)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _exit_x(board_w: int) -> int:
    return (board_w - 2) // 2


def _can_move(blocks: list[Block], idx: int, dx: int, dy: int,
              board_w: int, board_h: int) -> bool:
    b = blocks[idx]
    nx, ny = b.x + dx, b.y + dy
    if nx < 0 or nx + b.w > board_w or ny < 0 or ny + b.h > board_h:
        return False
    new_cells = {(nx + ddx, ny + ddy) for ddx in range(b.w) for ddy in range(b.h)}
    for j, ob in enumerate(blocks):
        if j == idx:
            continue
        for cx in range(ob.x, ob.x + ob.w):
            for cy in range(ob.y, ob.y + ob.h):
                if (cx, cy) in new_cells:
                    return False
    return True


def _target_blocked(blocks: list[Block], board_w: int, board_h: int) -> bool:
    """Return True if the path from target block straight down to exit is blocked.

    Checks whether at least one other block occupies a cell directly below the
    target block (in the columns it spans), between it and the exit row.
    This filters out trivially easy 1-move solutions.
    """
    target = blocks[0]
    exit_x = _exit_x(board_w)
    # target's eventual x must equal exit_x for it to be solvable straight down
    # Check for any block occupying cells below target in same columns
    target_cols = set(range(target.x, target.x + target.w))
    for b in blocks[1:]:
        for bx in range(b.x, b.x + b.w):
            for by in range(b.y, b.y + b.h):
                if bx in target_cols and by > target.y + target.h - 1:
                    return True  # some block is below target in its column
    return False


def _generate(board_w: int, board_h: int, num_other: int,
              rng: random.Random, n_scramble: int, min_depth: int = 0) -> list[Block]:
    ex = _exit_x(board_w)
    target = Block(ex, board_h - 2, 2, 2, is_target=True)

    occupied: set[tuple[int, int]] = set(target.cells())
    other: list[Block] = []
    block_types = [(1, 1), (1, 2), (2, 1)] * 4
    attempts = 0
    while len(other) < num_other and attempts < 2000:
        attempts += 1
        w, h = rng.choice(block_types)
        x = rng.randint(0, board_w - w)
        y = rng.randint(0, board_h - h)
        cells = [(x + dx, y + dy) for dx in range(w) for dy in range(h)]
        if all(c not in occupied for c in cells):
            occupied.update(cells)
            other.append(Block(x, y, w, h))

    blocks: list[Block] = [target] + other

    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for _ in range(n_scramble):
        valid = [(i, dx, dy)
                 for i in range(len(blocks))
                 for dx, dy in dirs
                 if _can_move(blocks, i, dx, dy, board_w, board_h)]
        if not valid:
            break
        i, dx, dy = rng.choice(valid)
        blocks[i].x += dx
        blocks[i].y += dy

    # Ensure puzzle is not trivially easy: target must not be directly at exit,
    # and must have at least one block in its downward path.
    # If too easy, keep scrambling in bursts.
    if min_depth > 0:
        exit_x = _exit_x(board_w)
        for _ in range(50):
            target = blocks[0]
            already_solved = (target.x == exit_x and target.y == board_h - target.h)
            has_blocker = _target_blocked(blocks, board_w, board_h)
            if not already_solved and has_blocker:
                break
            # More random moves to get a harder position
            for _ in range(10):
                valid = [(i, dx, dy)
                         for i in range(len(blocks))
                         for dx, dy in dirs
                         if _can_move(blocks, i, dx, dy, board_w, board_h)]
                if not valid:
                    break
                i, dx, dy = rng.choice(valid)
                blocks[i].x += dx
                blocks[i].y += dy

    return blocks


# ---------------------------------------------------------------------------
# Board pixel renderer
# ---------------------------------------------------------------------------

def _render(board_w: int, board_h: int, blocks: list[Block], sel: int) -> np.ndarray:
    sw, sh = board_w + 2, board_h + 2
    px = np.full((sh, sw), C_WALL, dtype=np.int8)
    px[1:sh - 1, 1:sw - 1] = C_FLOOR

    ex = _exit_x(board_w)
    px[sh - 1, ex + 1] = C_EXIT
    px[sh - 1, ex + 2] = C_EXIT

    for i, b in enumerate(blocks):
        if b.is_target:
            color = C_TARGET
        elif i == sel:
            color = C_SELECTED
        else:
            color = _OTHER_COLORS[(i - 1) % len(_OTHER_COLORS)]
        for dx in range(b.w):
            for dy in range(b.h):
                px[b.y + 1 + dy, b.x + 1 + dx] = color

    return px


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class Klotski(AugmentedGame):

    def __init__(self, seed: int | None = None) -> None:
        rng = random.Random(seed)
        self._step_counter = StepCounter(_STEP_BUDGETS[0])

        self._initial_blocks: list[list[Block]] = []
        self._board_dims: list[tuple[int, int]] = []
        self._blocks: list[Block] = []
        self._sel: int = 0
        self._history: list[tuple[list[Block], int]] = []

        levels: list[Level] = []
        for board_w, board_h, num_other, n_scramble, min_depth in _CONFIGS:
            blocks = _generate(board_w, board_h, num_other, rng, n_scramble, min_depth)
            self._initial_blocks.append([b.copy() for b in blocks])
            self._board_dims.append((board_w, board_h))

            px = _render(board_w, board_h, blocks, 0)
            sw, sh = board_w + 2, board_h + 2
            sprite = Sprite(
                pixels=px, name="board", collidable=False, layer=0
            ).set_position(0, 0)
            levels.append(Level(sprites=[sprite], grid_size=(sw, sh)))

        self._init_augmentation(seed)

        super().__init__(
            game_id="klotski",
            levels=levels,
            camera=Camera(
                background=C_BORDER,
                letter_box=C_BORDER,
                interfaces=[self._step_counter],
            ),
            available_actions=[1, 2, 3, 4, 5, 7],
        )

    def _board_sprite(self) -> Sprite:
        return next(s for s in self.current_level.get_sprites() if s.name == "board")

    def _refresh(self) -> None:
        bw, bh = self._board_dims[self._current_level_index]
        self._board_sprite().pixels = _render(bw, bh, self._blocks, self._sel)

    def on_set_level(self, level: Level) -> None:
        self._history.clear()
        self._sel = 0
        idx = self._current_level_index
        self._blocks = [b.copy() for b in self._initial_blocks[idx]]
        self._step_counter.reset(_STEP_BUDGETS[idx])
        self._refresh()

    def step(self) -> None:
        bw, bh = self._board_dims[self._current_level_index]

        self._step_counter.decrement()
        _action_id = self.screen_action_to_game(self.action.id)

        if _action_id == GameAction.ACTION7:
            if self._history:
                self._blocks, self._sel = self._history.pop()
                self._refresh()
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        if _action_id == GameAction.ACTION5:
            self._sel = (self._sel + 1) % len(self._blocks)
            self._refresh()
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        delta = {
            GameAction.ACTION1: (0, -1),
            GameAction.ACTION2: (0,  1),
            GameAction.ACTION3: (-1, 0),
            GameAction.ACTION4: (1,  0),
        }.get(_action_id)

        if delta and _can_move(self._blocks, self._sel, *delta, bw, bh):
            self._history.append(([b.copy() for b in self._blocks], self._sel))
            dx, dy = delta
            self._blocks[self._sel].x += dx
            self._blocks[self._sel].y += dy
            self._refresh()

            target = self._blocks[0]
            ex = _exit_x(bw)
            if target.is_target and target.x == ex and target.y == bh - target.h:
                self.next_level()
                self.complete_action()
                return

        if not self._step_counter.steps_remaining:
            self.lose()

        self.complete_action()
