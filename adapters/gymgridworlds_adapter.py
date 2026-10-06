"""gymgridworlds_adapter.py — Wraps gym_gridworlds (sparisi/gym_gridworlds) environments
as ARCBaseGame-compatible objects.

Why gym_gridworlds fits ARC-AGI-3 training:
  - Pure top-down absolute movement: UP/DOWN/LEFT/RIGHT directly move the agent
    one cell. No egocentric turning needed — maps exactly to the W/A/S/D controls
    the solver client already provides.
  - Rich tile vocabulary: goal (green), penalty (red), pits (black), quicksand
    (yellow), walls (dark gray), directional arrow tiles (purple), random-movement
    tiles (pink) — all visually distinct in the ARC palette.
  - 30 environments covering simple rooms, multi-room mazes, cliff-walk,
    quicksand puzzles, and taxi delivery — zero language required to understand
    the objective (reach the green tile).
  - Grid state is always fully observable via env.unwrapped.grid.

Frame pipeline:
  env.unwrapped.grid  (H×W int array of tile IDs)
  + agent_pos overlay (blue cell)
  → map each tile to ARC palette index via _TILE_TO_ARC table
  → draw directional arrows on arrow tiles (when cell_px >= 4)
  → center-pad to 64×64 with black border, integer cell size (no aliasing)
  → 64×64 uint8 ndarray of palette indices

ARC palette mapping:
  EMPTY  (-1) →  0  white       — open floor
  WALL   (-3) →  4  dark gray   — impassable wall
  PIT    (-4) →  5  black       — instant death (-100 reward)
  QCKSND (-2) → 11  yellow      — quicksand (90% stuck)
  GOOD   (10) → 14  green       — main goal  (+1 reward, terminates)
  GOOD_SMALL (9) → 3 light green — small reward (+0.1)
  BAD    (11) →  8  red         — penalty tile (-10 reward)
  BAD_SMALL (12) → 12 orange    — small penalty (-0.1)
  Arrow tiles (0-8) → 15 purple  — forced-direction tile (arrow drawn on top)
  RND_MOVE (13) →  6  pink      — random forced action
  Agent       →  9  blue        — player position

Action mapping (absolute movement, no turning):
  GameAction.ACTION1 (W / ↑)  → UP    (3)
  GameAction.ACTION2 (S / ↓)  → DOWN  (2)
  GameAction.ACTION3 (A / ←)  → LEFT  (0)
  GameAction.ACTION4 (D / →)  → RIGHT (1)
  GameAction.ACTION5 (E / sp) → STAY  (4)

WIN / GAME_OVER semantics:
  terminated=True AND reward > 0  →  GameState.WIN    (reached goal)
  terminated=True AND reward ≤ 0  →  GameState.GAME_OVER  (fell in pit)
  truncated=True                  →  GameState.GAME_OVER  (step limit)

Level concept:
  set_level(idx) resets with seed = base_seed + idx, giving varied starting
  positions for environments registered with random_start support.

Taxi environment note:
  Passengers (dark-green GOOD_SMALL tiles) disappear as the agent picks them up
  by walking over them. The goal tile (bright green) terminates the episode; if
  passengers were delivered the reward is > 0 → WIN, otherwise 0 → GAME_OVER.

Usage:
    from adapters import GymGridworldsAdapter
    from arcengine import GameAction, ActionInput

    game = GymGridworldsAdapter("Gym-Gridworlds/DangerMaze-5x6-v0", seed=0)
    game.set_level(0)
    result = game.perform_action(ActionInput(id=GameAction.ACTION1))
    print(result.state)           # GameState.NOT_FINISHED
    print(result.frame[0].shape)  # (64, 64)
"""

from __future__ import annotations

import copy as _copy
import random as _random_module

import numpy as np
from PIL import Image

import gymnasium as gym
import gym_gridworlds  # noqa: F401 — side-effect: registers Gym-Gridworlds/* envs

from gym_gridworlds.gridworld import (
    EMPTY, WALL, PIT, QCKSND,
    GOOD, GOOD_SMALL, BAD, BAD_SMALL, RND_MOVE,
    LEFT, RIGHT, DOWN, UP, STAY,
)

from arcengine import ActionInput, GameAction, GameState, FrameDataRaw

from utils.rotation import remap_action as _remap_action
from adapters.base import BaseAdapter

# ---------------------------------------------------------------------------
# Tile → ARC palette index
#
# ARC 16-color palette (index → visual meaning):
#   0  white        7  light gray
#   1  blue         8  red-orange (danger/cursor)
#   2  red          9  blue (player)
#   3  green        10 mid gray
#   4  dark gray    11 yellow
#   5  black        12 orange
#   6  pink/magenta 13 brown
#                   14 green (target/goal)
#                   15 purple
# ---------------------------------------------------------------------------

# Directional tile IDs: LEFT=0, RIGHT=1, DOWN=2, UP=3, STAY=4,
#                       UP_LEFT=5, DOWN_LEFT=6, DOWN_RIGHT=7, UP_RIGHT=8
_DIRECTIONAL_TILES: frozenset[int] = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8})

_TILE_TO_ARC: dict[int, int] = {
    EMPTY:       0,   # white — open floor
    WALL:        4,   # dark gray — wall
    PIT:         5,   # black — instant death
    QCKSND:     11,   # yellow — quicksand
    GOOD:       14,   # green (goal) — +1 reward
    GOOD_SMALL:  3,   # light green — +0.1 reward / passenger in Taxi
    BAD:         8,   # red — penalty (-10)
    BAD_SMALL:  12,   # orange — small penalty (-0.1)
    RND_MOVE:    6,   # pink — random forced action
    # Cardinal directional tiles → purple (arrow drawn on top)
    LEFT:       15,
    RIGHT:      15,
    DOWN:       15,
    UP:         15,
    STAY:       15,
    # Diagonal directional tiles
    5:          15,   # UP_LEFT
    6:          15,   # DOWN_LEFT
    7:          15,   # DOWN_RIGHT
    8:          15,   # UP_RIGHT
    # TravelField tiles (rare; included for completeness)
    99:          3,   # GRASS → light green
    98:         11,   # ROAD → yellow
    97:         13,   # SWAMP → brown
}

_AGENT_ARC: int = 9  # blue (player)

# Lookup table: tile_id + _LUT_OFFSET → arc_idx
# tile IDs span -4 (PIT) to 99 (GRASS in TravelField)
_LUT_OFFSET = 4
_LUT_SIZE = 104
_TILE_LUT: np.ndarray = np.zeros(_LUT_SIZE, dtype=np.uint8)
for _tile, _arc in _TILE_TO_ARC.items():
    idx = _tile + _LUT_OFFSET
    if 0 <= idx < _LUT_SIZE:
        _TILE_LUT[idx] = _arc


# ---------------------------------------------------------------------------
# Directional arrow drawing
# ---------------------------------------------------------------------------

# Minigrid direction convention reused here:
#   0 = east (▶)   1 = south (▽)   2 = west (◀)   3 = north (▲)
# gym_gridworlds: LEFT=0 RIGHT=1 DOWN=2 UP=3; diagonals 5-8
_GYM_TO_MGDIR: dict[int, int] = {
    LEFT:  2,   # west  ◀
    RIGHT: 0,   # east  ▶
    DOWN:  1,   # south ▽
    UP:    3,   # north ▲
}


def _draw_cardinal_arrow(
    frame: np.ndarray,
    r1: int, c1: int,
    cell_px: int,
    direction: int,
) -> None:
    """Draw a white filled triangle arrow inside a purple cell.

    direction: 0=east(▶) 1=south(▽) 2=west(◀) 3=north(▲)
    Same geometry as _draw_agent_direction in minigrid_adapter.
    """
    r2 = r1 + cell_px
    c2 = c1 + cell_px

    for dr in range(cell_px):
        for dc in range(cell_px):
            nr = (dr + 0.5) / cell_px   # 0 = top,  1 = bottom
            nc = (dc + 0.5) / cell_px   # 0 = left, 1 = right
            if direction == 3:           # north ▲: apex top
                in_tri = abs(nc - 0.5) < nr * 0.45
            elif direction == 1:         # south ▽: apex bottom
                in_tri = abs(nc - 0.5) < (1.0 - nr) * 0.45
            elif direction == 0:         # east  ▶: apex right
                in_tri = abs(nr - 0.5) < (1.0 - nc) * 0.45
            else:                        # west  ◀: apex left
                in_tri = abs(nr - 0.5) < nc * 0.45
            if in_tri:
                frame[r1 + dr, c1 + dc] = 0   # white


def _draw_diagonal_arrow(
    frame: np.ndarray,
    r1: int, c1: int,
    cell_px: int,
    tile_id: int,
) -> None:
    """Draw a white diagonal slash for diagonal directional tiles."""
    # Diagonal: 5=UL 6=DL 7=DR 8=UR
    # Draw a single pixel diagonal line from the "base" corner to center
    cy = r1 + cell_px // 2
    cx = c1 + cell_px // 2
    # Leading pixel in the forced direction
    dr_sign = -1 if tile_id in (5, 8) else 1   # UP family vs DOWN family
    dc_sign = -1 if tile_id in (5, 6) else 1   # LEFT family vs RIGHT family
    for step in range(max(1, cell_px // 3)):
        rr = cy + dr_sign * step
        cc = cx + dc_sign * step
        if r1 <= rr < r1 + cell_px and c1 <= cc < c1 + cell_px:
            frame[rr, cc] = 0


# ---------------------------------------------------------------------------
# Frame builder
# ---------------------------------------------------------------------------

def _build_arc_frame(
    grid: np.ndarray,
    agent_pos: tuple[int, int],
) -> np.ndarray:
    """Build a 64×64 ARC palette frame from a gym_gridworlds grid.

    Args:
        grid:      (H, W) int array of tile IDs (from env.unwrapped.grid)
        agent_pos: (row, col) of the current agent position
    """
    H, W = grid.shape

    # Vectorized tile → ARC color via LUT
    clipped = np.clip(grid + _LUT_OFFSET, 0, _LUT_SIZE - 1).astype(np.uint8)
    arc_grid = _TILE_LUT[clipped]

    # Overlay agent as blue
    ar, ac = int(agent_pos[0]), int(agent_pos[1])
    if 0 <= ar < H and 0 <= ac < W:
        arc_grid[ar, ac] = _AGENT_ARC

    # Integer cell size → uniform cells, no aliasing
    cell_px = max(1, 64 // max(H, W))
    rh, rw = H * cell_px, W * cell_px

    img = Image.fromarray(arc_grid, mode="L")
    img = img.resize((rw, rh), Image.NEAREST)   # PIL: (width, height)

    pad_r = (64 - rh) // 2
    pad_c = (64 - rw) // 2
    frame = np.full((64, 64), 5, dtype=np.uint8)   # black border
    frame[pad_r:pad_r + rh, pad_c:pad_c + rw] = np.asarray(img, dtype=np.uint8)

    # Draw directional arrows when cells are large enough
    if cell_px >= 4:
        for gr in range(H):
            for gc in range(W):
                tile = int(grid[gr, gc])
                if tile not in _DIRECTIONAL_TILES:
                    continue
                pr = pad_r + gr * cell_px
                pc = pad_c + gc * cell_px
                if tile in _GYM_TO_MGDIR:
                    _draw_cardinal_arrow(frame, pr, pc, cell_px, _GYM_TO_MGDIR[tile])
                elif tile in (5, 6, 7, 8):
                    _draw_diagonal_arrow(frame, pr, pc, cell_px, tile)
                elif tile == STAY:
                    # Center dot for STAY tile
                    mr = pr + cell_px // 2
                    mc = pc + cell_px // 2
                    if 0 <= mr < 64 and 0 <= mc < 64:
                        frame[mr, mc] = 0

    return frame


# ---------------------------------------------------------------------------
# Grid randomization helpers
# ---------------------------------------------------------------------------

def _randomize_barrier_5x5_grid(u) -> None:
    """Randomize the Barrier-5x5 grid at level construction time.

    Two things are randomized each reset:
      1. **Cluster position** — the 3×3 block (GOOD cell + 8 surrounding
         directional-arrow cells) is placed at a randomly chosen center
         position within the 5×5 grid.  Valid center positions are (r,c)
         with 1 ≤ r ≤ 3 and 1 ≤ c ≤ 3, excluding any that would overlap
         the agent's fixed start cell at (0, 0).
      2. **Access direction** — one of UP / DOWN / LEFT / RIGHT is chosen
         at random.  The surrounding cell in that direction is left EMPTY
         (the only opening in the barrier ring), while all other surrounding
         cells receive outward-pointing arrows:
           corner/side tiles: LEFT (west column), RIGHT (east column),
                              UP (north edge), DOWN (south edge).

    The grid is fully cleared to EMPTY before the cluster is placed, so
    no stale tiles remain from the previous layout.
    """
    grid = u.grid
    H, W = grid.shape
    agent_r, agent_c = int(u.agent_pos[0]), int(u.agent_pos[1])

    # ------------------------------------------------------------------
    # Build candidate goal positions: 3×3 cluster must fit inside grid
    # and must not overlap the agent start cell.
    # ------------------------------------------------------------------
    valid_positions: list[tuple[int, int]] = []
    for gr in range(1, H - 1):
        for gc in range(1, W - 1):
            # The 9 cluster cells span rows [gr-1, gr+1] × cols [gc-1, gc+1]
            cluster_overlaps_agent = (
                gr - 1 <= agent_r <= gr + 1 and
                gc - 1 <= agent_c <= gc + 1
            )
            if not cluster_overlaps_agent:
                valid_positions.append((gr, gc))

    pos_idx = int(u.np_random.integers(0, len(valid_positions)))
    gr, gc = valid_positions[pos_idx]

    # Pick one of 4 access directions: 0=UP 1=DOWN 2=LEFT 3=RIGHT
    # Only directions where the entry cell (one step beyond the access cell,
    # away from the goal) is within the grid are valid — otherwise the access
    # cell lies on the grid edge and there is no path into it, producing an
    # unsolvable level.
    valid_dirs: list[int] = []
    if gr - 2 >= 0:       # UP:    entry cell (gr-2, gc) must be in bounds
        valid_dirs.append(0)
    if gr + 2 <= H - 1:   # DOWN:  entry cell (gr+2, gc) must be in bounds
        valid_dirs.append(1)
    if gc - 2 >= 0:       # LEFT:  entry cell (gr, gc-2) must be in bounds
        valid_dirs.append(2)
    if gc + 2 <= W - 1:   # RIGHT: entry cell (gr, gc+2) must be in bounds
        valid_dirs.append(3)
    dir_idx = int(u.np_random.integers(0, len(valid_dirs)))
    access_dir = valid_dirs[dir_idx]
    access_cell = [
        (gr - 1, gc),      # 0 = UP:    top edge is EMPTY
        (gr + 1, gc),      # 1 = DOWN:  bottom edge is EMPTY
        (gr,     gc - 1),  # 2 = LEFT:  left edge is EMPTY
        (gr,     gc + 1),  # 3 = RIGHT: right edge is EMPTY
    ][access_dir]

    # ------------------------------------------------------------------
    # Write cluster into grid (clear first so no leftover tiles)
    # ------------------------------------------------------------------
    grid[:] = EMPTY
    grid[gr, gc] = GOOD

    # Default outward-pointing arrows for the 8 surrounding positions:
    #   west column  → LEFT,  east column → RIGHT
    #   north center → UP,    south center → DOWN
    surrounding: dict[tuple[int, int], int] = {
        (gr - 1, gc - 1): LEFT,
        (gr - 1, gc    ): UP,
        (gr - 1, gc + 1): RIGHT,
        (gr,     gc - 1): LEFT,
        (gr,     gc + 1): RIGHT,
        (gr + 1, gc - 1): LEFT,
        (gr + 1, gc    ): DOWN,
        (gr + 1, gc + 1): RIGHT,
    }

    for pos, tile in surrounding.items():
        if pos != access_cell:          # access cell stays EMPTY
            grid[pos[0], pos[1]] = tile


def _is_reachable(grid: np.ndarray, start: tuple[int, int]) -> bool:
    """BFS from *start*; return True if the GOOD tile is reachable.

    WALLs and PITs are treated as impassable — all other tile types
    (EMPTY, BAD, BAD_SMALL, QCKSND, RND_MOVE, …) may be traversed.
    """
    H, W = grid.shape
    _IMPASSABLE = {WALL, PIT}
    visited = np.zeros((H, W), dtype=bool)
    queue: list[tuple[int, int]] = [start]
    visited[start[0], start[1]] = True
    while queue:
        r, c = queue.pop()
        if int(grid[r, c]) == GOOD:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                if int(grid[nr, nc]) not in _IMPASSABLE:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
    return False


def _randomize_danger_maze_5x6(u) -> None:
    """Randomize the DangerMaze-5x6 grid each reset.

    All non-EMPTY tiles (PITs, BAD tiles, WALLs, and the GOOD goal) are
    collected, the grid is cleared to EMPTY, and the tiles are placed at
    random positions while keeping the agent's start cell EMPTY so the
    episode always begins on safe floor.

    A solvability check (BFS) is run after each shuffle; if no path from
    the agent's start to the GOOD tile exists the shuffle is retried until
    a valid layout is found (up to _MAX_RETRIES attempts).

    This varies both the obstacle layout and the trajectory needed to reach
    the goal each episode.
    """
    _MAX_RETRIES = 200

    grid = u.grid
    H, W = grid.shape
    agent_r, agent_c = int(u.agent_pos[0]), int(u.agent_pos[1])

    # Collect every non-EMPTY tile value
    non_empty_vals: list[int] = []
    for r in range(H):
        for c in range(W):
            val = int(grid[r, c])
            if val != EMPTY:
                non_empty_vals.append(val)

    # Candidate positions: every cell except the agent start
    available_pos = [
        (r, c) for r in range(H) for c in range(W)
        if (r, c) != (agent_r, agent_c)
    ]

    for _ in range(_MAX_RETRIES):
        perm = u.np_random.permutation(len(available_pos))
        grid[:] = EMPTY
        for i, val in enumerate(non_empty_vals):
            r, c = available_pos[perm[i]]
            grid[r, c] = val
        if _is_reachable(grid, (agent_r, agent_c)):
            return

    # Fallback: keep the last generated layout (very unlikely to be needed)


def _randomize_cliffwalk_4x12(u) -> None:
    """Randomize the CliffWalk-4x12 grid each reset.

    Three aspects are randomized each episode:

    1. **Start and goal column positions** — both are placed at independently
       chosen columns within the 12-column grid (guaranteed to differ).

    2. **Cliff row placement** — the row containing the start cell, the PITs,
       and the goal cell is placed either at the top (row 0) or at the bottom
       (last row), chosen uniformly at random.  In the bottom variant the
       navigable safe rows sit *above* the cliff row, mirroring the usual
       layout upside-down.

    3. **Navigable space width** — the number of fully-open safe rows is drawn
       uniformly from 1 to 5.  The cliff row is always one additional row,
       giving a total grid height of 2–6 rows.

    The grid array, its dimensions, and the agent start position are all
    updated on the unwrapped environment object so that _build_arc_frame
    sees the new layout immediately.
    """
    rng = u.np_random
    n_cols = 12

    # ------------------------------------------------------------------ #
    # 1. Narrowness: 1–5 rows of safe (all-EMPTY) navigable space
    # ------------------------------------------------------------------ #
    n_safe_rows = int(rng.integers(1, 6))   # 1 .. 5 inclusive
    n_rows = 1 + n_safe_rows                 # cliff row + safe rows

    # ------------------------------------------------------------------ #
    # 2. Cliff row position: top (row 0) or bottom (last row)
    # ------------------------------------------------------------------ #
    cliff_at_top = bool(rng.integers(0, 2))
    cliff_row = 0 if cliff_at_top else n_rows - 1

    # ------------------------------------------------------------------ #
    # 3. Start and goal columns (must differ)
    # ------------------------------------------------------------------ #
    start_col = int(rng.integers(0, n_cols))
    available_cols = [c for c in range(n_cols) if c != start_col]
    goal_col = available_cols[int(rng.integers(0, len(available_cols)))]

    # ------------------------------------------------------------------ #
    # Build the new grid
    # ------------------------------------------------------------------ #
    dtype = u.original_grid.dtype
    grid = np.full((n_rows, n_cols), EMPTY, dtype=dtype)

    # Cliff row: PITs fill every cell except the start and goal columns
    for c in range(n_cols):
        if c != start_col and c != goal_col:
            grid[cliff_row, c] = PIT
    # start_col remains EMPTY (agent spawn cell)
    grid[cliff_row, goal_col] = GOOD

    # ------------------------------------------------------------------ #
    # Patch the unwrapped environment
    # ------------------------------------------------------------------ #
    u.grid = grid
    u.original_grid = grid.copy()
    u.n_rows, u.n_cols = n_rows, n_cols
    u.agent_pos = (cliff_row, start_col)


def _randomize_quicksand_4x4(u) -> None:
    """Randomize the Quicksand-4x4 grid each reset.

    Collects all non-EMPTY tiles (2 BAD penalty tiles, 1 GOOD goal, 1 QCKSND
    quicksand tile), clears the grid, and places them at random positions
    excluding the agent's fixed start cell at (0, 0).

    No solvability check is needed: this grid contains no WALLs or PITs, so
    every cell is always reachable from every other cell.
    """
    grid = u.grid
    H, W = grid.shape
    agent_r, agent_c = int(u.agent_pos[0]), int(u.agent_pos[1])

    # Collect every non-EMPTY tile value
    non_empty_vals: list[int] = []
    for r in range(H):
        for c in range(W):
            val = int(grid[r, c])
            if val != EMPTY:
                non_empty_vals.append(val)

    # Candidate positions: every cell except the agent start
    available_pos = [
        (r, c) for r in range(H) for c in range(W)
        if (r, c) != (agent_r, agent_c)
    ]

    # Pick len(non_empty_vals) distinct positions at random
    perm = u.np_random.permutation(len(available_pos))

    # Reset grid to all EMPTY, then place tiles at shuffled positions
    grid[:] = EMPTY
    for i, val in enumerate(non_empty_vals):
        r, c = available_pos[perm[i]]
        grid[r, c] = val


def _randomize_quicksand_distract_4x4(u) -> None:
    """Randomize the Quicksand-Distract-4x4 grid each reset.

    Collects all non-EMPTY tiles (2 BAD penalty tiles, 1 GOOD goal, 1 QCKSND
    quicksand tile, 2 GOOD_SMALL distractor tiles), clears the grid, and places
    them at random positions excluding the agent's fixed start cell at (0, 0).

    No solvability check is needed: this grid contains no WALLs or PITs, so
    every cell is always reachable from every other cell.
    """
    grid = u.grid
    H, W = grid.shape
    agent_r, agent_c = int(u.agent_pos[0]), int(u.agent_pos[1])

    # Collect every non-EMPTY tile value
    non_empty_vals: list[int] = []
    for r in range(H):
        for c in range(W):
            val = int(grid[r, c])
            if val != EMPTY:
                non_empty_vals.append(val)

    # Candidate positions: every cell except the agent start
    available_pos = [
        (r, c) for r in range(H) for c in range(W)
        if (r, c) != (agent_r, agent_c)
    ]

    # Pick len(non_empty_vals) distinct positions at random
    perm = u.np_random.permutation(len(available_pos))

    # Reset grid to all EMPTY, then place tiles at shuffled positions
    grid[:] = EMPTY
    for i, val in enumerate(non_empty_vals):
        r, c = available_pos[perm[i]]
        grid[r, c] = val


def _randomize_tworoom_quicksand_3x5(u) -> None:
    """Randomize the TwoRoom-Quicksand-3x5 grid each reset.

    Collects all non-EMPTY tiles (1 LEFT arrow, 1 QCKSND quicksand, 1 GOOD
    goal), clears the grid, and places them at random positions excluding the
    agent's fixed start cell at (0, 0).

    A BFS solvability check ensures the GOOD tile is reachable from the agent
    start; if not, the layout is regenerated (up to _MAX_RETRIES attempts).
    Arrow tiles are not treated as impassable by BFS — the agent can traverse
    them (just gets pushed), so reachability is always well-defined.
    """
    _MAX_RETRIES = 200

    grid = u.grid
    H, W = grid.shape
    agent_r, agent_c = int(u.agent_pos[0]), int(u.agent_pos[1])

    # Collect every non-EMPTY tile value from the original layout
    non_empty_vals: list[int] = []
    for r in range(H):
        for c in range(W):
            val = int(grid[r, c])
            if val != EMPTY:
                non_empty_vals.append(val)

    # Candidate positions: every cell except the agent start
    available_pos = [
        (r, c) for r in range(H) for c in range(W)
        if (r, c) != (agent_r, agent_c)
    ]

    for _ in range(_MAX_RETRIES):
        perm = u.np_random.permutation(len(available_pos))
        grid[:] = EMPTY
        for i, val in enumerate(non_empty_vals):
            r, c = available_pos[perm[i]]
            grid[r, c] = val
        if _is_reachable(grid, (agent_r, agent_c)):
            return

    # Fallback: keep last generated layout


def _randomize_tworoom_distract_middle_2x11(u) -> None:
    """Randomize the TwoRoom-Distract-Middle-2x11 grid each reset.

    Collects all non-EMPTY tiles (1 GOOD_SMALL distractor, 1 RIGHT arrow,
    1 DOWN arrow, 1 LEFT arrow, 1 GOOD goal), clears the grid, and places
    them at random positions excluding the agent's fixed start cell at (1, 5).

    A BFS solvability check ensures the GOOD tile is reachable from the agent
    start; if not, the layout is regenerated (up to _MAX_RETRIES attempts).
    """
    _MAX_RETRIES = 200

    grid = u.grid
    H, W = grid.shape
    agent_r, agent_c = int(u.agent_pos[0]), int(u.agent_pos[1])

    # Collect every non-EMPTY tile value from the original layout
    non_empty_vals: list[int] = []
    for r in range(H):
        for c in range(W):
            val = int(grid[r, c])
            if val != EMPTY:
                non_empty_vals.append(val)

    # Candidate positions: every cell except the agent start
    available_pos = [
        (r, c) for r in range(H) for c in range(W)
        if (r, c) != (agent_r, agent_c)
    ]

    for _ in range(_MAX_RETRIES):
        perm = u.np_random.permutation(len(available_pos))
        grid[:] = EMPTY
        for i, val in enumerate(non_empty_vals):
            r, c = available_pos[perm[i]]
            grid[r, c] = val
        if _is_reachable(grid, (agent_r, agent_c)):
            return

    # Fallback: keep last generated layout


def _randomize_maze_12x12(u) -> None:
    """Generate a random solvable maze in the Maze-12x12-v0 grid each reset.

    Uses recursive backtracking (DFS) to carve passages through a grid
    initialized entirely to WALLs.  Logical cell positions sit at odd indices
    (1, 3, 5, 7, 9) giving a 5×5 cell lattice within the 12×12 physical grid;
    the extra row/column at index 10 and the outer border at index 0/11 remain
    as walls.

    The DFS spanning-tree property guarantees every cell is connected to every
    other cell, so the maze is always solvable — no BFS check is needed.

    Agent start and goal are placed at two distinct randomly chosen cells.
    """
    rng = u.np_random
    grid = u.grid
    H, W = grid.shape  # 12 × 12

    # --- Initialize grid to all walls ---
    grid[:] = WALL

    # --- Cell lattice: odd indices within the interior [1 .. H-2] ---
    cell_rows = list(range(1, H - 1, 2))  # [1, 3, 5, 7, 9]
    cell_cols = list(range(1, W - 1, 2))  # [1, 3, 5, 7, 9]
    n_cr = len(cell_rows)  # 5
    n_cc = len(cell_cols)  # 5

    # Mark every cell position as EMPTY
    for cr in cell_rows:
        for cc in cell_cols:
            grid[cr, cc] = EMPTY

    # --- Recursive backtracking (DFS) maze carving ---
    visited = [[False] * n_cc for _ in range(n_cr)]

    sr = int(rng.integers(0, n_cr))
    sc = int(rng.integers(0, n_cc))
    visited[sr][sc] = True
    stack = [(sr, sc)]

    _DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

    while stack:
        r, c = stack[-1]
        nbrs = [
            (dr, dc, r + dr, c + dc)
            for dr, dc in _DIRS
            if 0 <= r + dr < n_cr and 0 <= c + dc < n_cc
            and not visited[r + dr][c + dc]
        ]
        if nbrs:
            dr, dc, nr, nc = nbrs[int(rng.integers(0, len(nbrs)))]
            # Carve through the wall cell between current and chosen neighbour.
            # When moving vertically (dr=±1, dc=0): passage at (cell_rows[r]+dr, cell_cols[c])
            # When moving horizontally (dr=0, dc=±1): passage at (cell_rows[r], cell_cols[c]+dc)
            grid[cell_rows[r] + dr, cell_cols[c] + dc] = EMPTY
            visited[nr][nc] = True
            stack.append((nr, nc))
        else:
            stack.pop()

    # --- Place agent and goal at two distinct random cell positions ---
    all_cells = [
        (cell_rows[ri], cell_cols[ci])
        for ri in range(n_cr)
        for ci in range(n_cc)
    ]
    perm = rng.permutation(len(all_cells))
    agent_pos = all_cells[int(perm[0])]
    goal_pos  = all_cells[int(perm[1])]

    grid[goal_pos[0], goal_pos[1]] = GOOD
    u.agent_pos = agent_pos


def _shuffle_full_4x5_grid(u) -> None:
    """Randomly redistribute all non-EMPTY tiles across the Full-4x5 grid.

    Keeps the agent's starting cell EMPTY so the episode begins on a safe
    floor tile.  The set (and count) of each tile type is preserved; only
    their positions change.  Uses u.np_random so the shuffle is reproducible
    from the seed passed to env.reset().
    """
    grid = u.grid
    H, W = grid.shape
    agent_r, agent_c = int(u.agent_pos[0]), int(u.agent_pos[1])

    # Collect every non-EMPTY, non-WALL tile value in the current grid
    non_empty_vals: list[int] = []
    for r in range(H):
        for c in range(W):
            val = int(grid[r, c])
            if val not in (EMPTY, WALL):
                non_empty_vals.append(val)

    # Build list of candidate positions (all cells except the agent start)
    available_pos = [
        (r, c) for r in range(H) for c in range(W)
        if (r, c) != (agent_r, agent_c)
    ]

    # Pick len(non_empty_vals) distinct positions at random
    perm = u.np_random.permutation(len(available_pos))

    # Reset grid to all EMPTY, then place tiles at shuffled positions
    grid[:] = EMPTY
    for i, val in enumerate(non_empty_vals):
        r, c = available_pos[perm[i]]
        grid[r, c] = val


# ---------------------------------------------------------------------------
# Available environments
# ---------------------------------------------------------------------------

_AVAILABLE_GYMGRIDWORLDS_ENVS: tuple[str, ...] = (
    # --- Simple navigation ---
    "Gym-Gridworlds/Empty-10x10-v0",
    "Gym-Gridworlds/Empty-Distract-6x6-v0",
    "Gym-Gridworlds/Corridor-3x4-v0",
    # --- Mixed tile types ---
    "Gym-Gridworlds/Penalty-3x3-v0",
    "Gym-Gridworlds/Penalty-Randomized-4x4-v0",
    "Gym-Gridworlds/Full-4x5-v0",
    "Gym-Gridworlds/Full-RandomGoalAndStart-4x5-v0",
    "Gym-Gridworlds/Barrier-5x5-v0",
    "Gym-Gridworlds/DangerMaze-5x6-v0",
    "Gym-Gridworlds/CliffWalk-4x12-v0",
    # --- Quicksand ---
    "Gym-Gridworlds/Quicksand-4x4-v0",
    "Gym-Gridworlds/Quicksand-Distract-4x4-v0",
    "Gym-Gridworlds/TwoRoom-Quicksand-3x5-v0",
    "Gym-Gridworlds/TwoRoom-Distract-Middle-2x11-v0",
    # --- Four-rooms ---
    "Gym-Gridworlds/FourRooms-Symmetrical-11x11-v0",
    "Gym-Gridworlds/FourRooms-8x7-v0",
    "Gym-Gridworlds/FourRooms-Loop-8x7-v0",
    "Gym-Gridworlds/FourRooms-Stuck-8x7-v0",
    "Gym-Gridworlds/FourRooms-13x13-v0",
    "Gym-Gridworlds/FourRooms-Loop-13x13-v0",
    "Gym-Gridworlds/FourRooms-Stuck-13x13-v0",
    "Gym-Gridworlds/FourRooms-Wall-7x7-v0",
    "Gym-Gridworlds/FourRooms-Cross-14x16-v0",
    # --- Three-rooms ---
    "Gym-Gridworlds/ThreeRooms-Wall-11x8-v0",
    "Gym-Gridworlds/ThreeRooms-Quicksand-11x8-v0",
    "Gym-Gridworlds/ThreeRooms-Wall-14x16-v0",
    "Gym-Gridworlds/ThreeRooms-Quicksand-14x16-v0",
    # --- Maze ---
    "Gym-Gridworlds/Maze-12x12-v0",
    # --- Taxi delivery ---
    "Gym-Gridworlds/Taxi-6x7-v0",
)


# GameAction → gym_gridworlds action integer
# ACTION1 = W / ↑   → UP    (3)
# ACTION2 = S / ↓   → DOWN  (2)
# ACTION3 = A / ←   → LEFT  (0)
# ACTION4 = D / →   → RIGHT (1)
# ACTION5 = E / sp  → STAY  (4)
_ACTION_MAP: dict[GameAction, int] = {
    GameAction.ACTION1: 3,  # UP
    GameAction.ACTION2: 2,  # DOWN
    GameAction.ACTION3: 0,  # LEFT
    GameAction.ACTION4: 1,  # RIGHT
    GameAction.ACTION5: 4,  # STAY
}


# ---------------------------------------------------------------------------
# GymGridworldsAdapter
# ---------------------------------------------------------------------------

class GymGridworldsAdapter(BaseAdapter):
    """Wraps a gym_gridworlds environment as an ARCBaseGame-compatible object.

    Args:
        env_id: Gymnasium environment ID, e.g. "Gym-Gridworlds/DangerMaze-5x6-v0".
        seed:   Base random seed.  set_level(idx) resets with seed + idx.
    """

    def __init__(self, env_id: str, seed: int = 0) -> None:
        self._env_id = env_id
        self._seed = seed

        # Derive a clean game_id (strip namespace prefix and version suffix)
        clean = (
            env_id
            .replace("Gym-Gridworlds/", "")
            .replace("-v0", "")
            .replace("-v1", "")
        )
        self._game_id = "gymgridworlds_" + clean.lower().replace("-", "_")

        self._env = gym.make(env_id)

        # Rotation augmentation (randomized per level reset)
        self._rotation_k: int = 0

        # Shared adapter scaffolding (state, step counter, undo stack) + 1st frame.
        # The step counter is cosmetic here — episode truncation is handled by
        # the gym TimeLimit wrapper, not the HUD counter.
        self._init_base(self._game_id, max_steps=self._env.spec.max_episode_steps or 100,
                        available_actions=[1, 2, 3, 4, 5])
        self._reset_env(self._seed)

    # ------------------------------------------------------------------
    # Internal reset (BaseAdapter hook)
    # ------------------------------------------------------------------

    def _reset_env(self, seed: int) -> None:
        self._env.reset(seed=seed)
        u = self._env.unwrapped
        if self._env_id == "Gym-Gridworlds/CliffWalk-4x12-v0":
            _randomize_cliffwalk_4x12(u)
        elif self._env_id == "Gym-Gridworlds/Full-4x5-v0":
            _shuffle_full_4x5_grid(u)
        elif self._env_id == "Gym-Gridworlds/Barrier-5x5-v0":
            _randomize_barrier_5x5_grid(u)
        elif self._env_id == "Gym-Gridworlds/DangerMaze-5x6-v0":
            _randomize_danger_maze_5x6(u)
        elif self._env_id == "Gym-Gridworlds/Quicksand-4x4-v0":
            _randomize_quicksand_4x4(u)
        elif self._env_id == "Gym-Gridworlds/Quicksand-Distract-4x4-v0":
            _randomize_quicksand_distract_4x4(u)
        elif self._env_id == "Gym-Gridworlds/TwoRoom-Quicksand-3x5-v0":
            _randomize_tworoom_quicksand_3x5(u)
        elif self._env_id == "Gym-Gridworlds/TwoRoom-Distract-Middle-2x11-v0":
            _randomize_tworoom_distract_middle_2x11(u)
        elif self._env_id == "Gym-Gridworlds/Maze-12x12-v0":
            _randomize_maze_12x12(u)
        self._rotation_k = _random_module.randint(0, 3)
        self._current_frame = self._rotate_frame(_build_arc_frame(u.grid, u.agent_pos))
        self._state = GameState.NOT_FINISHED
        self._action_count = 0

    def _rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply the current rotation augmentation to a frame."""
        if self._rotation_k == 0:
            return frame
        return np.rot90(frame, k=self._rotation_k).copy()

    # ------------------------------------------------------------------
    # BaseAdapter hooks (game_id/set_level/perform_action/_make_frame_data +
    # RESET + ACTION7-undo + terminal short-circuit come from BaseAdapter).
    # ------------------------------------------------------------------

    def _apply(self, action_input: ActionInput) -> None:
        """Step the environment with one action.

        WIN: terminated AND reward > 0 (reached goal with positive reward).
        GAME_OVER: terminated AND reward ≤ 0 (fell in pit), OR truncated."""
        effective_id = _remap_action(action_input.id, self._rotation_k)
        gym_action = _ACTION_MAP.get(effective_id, 3)  # default: UP

        _obs, reward, terminated, truncated, _info = self._env.step(gym_action)
        self._action_count += 1

        u = self._env.unwrapped

        # gym_gridworlds computes reward/termination from the cell the agent
        # was on *before* this step's move, so reaching a goal cell only ends
        # the episode on the next action. Detect arrival on the goal here so
        # the WIN fires on the step that actually reaches it.
        if not terminated and not truncated:
            ar, ac = int(u.agent_pos[0]), int(u.agent_pos[1])
            if int(u.grid[ar, ac]) == GOOD:
                terminated = True
                if hasattr(u, "passengers_picked"):
                    # Taxi: reward scales with the number of passengers delivered
                    rewards_per_passengers = [0.0, 1.0, 3.0, 15.0]
                    reward = rewards_per_passengers[int(np.sum(u.passengers_picked))]
                else:
                    reward = float(u.rewards[GOOD])

        self._current_frame = self._rotate_frame(_build_arc_frame(u.grid, u.agent_pos))

        if terminated:
            self._state = GameState.WIN if reward > 0 else GameState.GAME_OVER
        elif truncated:
            self._state = GameState.GAME_OVER
        else:
            self._state = GameState.NOT_FINISHED

    # ── snapshot / restore for generic UNDO ──────────────────────────────────
    def _snapshot(self):
        """Full mutable state: the gridworld env's grid/agent + gym TimeLimit
        step count, plus adapter bookkeeping. grid is deep-copied so quicksand /
        distractor mutations are reversible."""
        u = self._env.unwrapped
        return (
            _copy.deepcopy(u.grid), tuple(u.agent_pos),
            getattr(u, "last_action", None), getattr(u, "last_pos", None),
            _copy.deepcopy(getattr(u, "passengers_picked", None)),
            getattr(self._env, "_elapsed_steps", None),
            self._action_count, self._state, self._current_frame,
        )

    def _restore(self, snap) -> None:
        u = self._env.unwrapped
        (grid, agent_pos, last_action, last_pos, passengers,
         elapsed, self._action_count, self._state, self._current_frame) = snap
        u.grid = grid
        u.agent_pos = agent_pos
        if hasattr(u, "last_action"):
            u.last_action = last_action
        if hasattr(u, "last_pos"):
            u.last_pos = last_pos
        if passengers is not None and hasattr(u, "passengers_picked"):
            u.passengers_picked = passengers
        if elapsed is not None and hasattr(self._env, "_elapsed_steps"):
            self._env._elapsed_steps = elapsed

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def list_available_envs() -> list[str]:
        """Return all supported Gym-Gridworlds environment IDs."""
        return list(_AVAILABLE_GYMGRIDWORLDS_ENVS)

    def close(self) -> None:
        """Release the underlying gymnasium environment."""
        self._env.close()

    def __repr__(self) -> str:
        return (
            f"GymGridworldsAdapter(env={self._env_id!r}, "
            f"level={self._current_level_index}, state={self._state})"
        )
