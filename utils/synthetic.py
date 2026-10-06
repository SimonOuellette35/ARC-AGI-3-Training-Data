"""
utils/synthetic.py
==================

THE single implementation of the ``synthetic_geodesic`` environment: maze
generation, BFS geodesic distance, optimal-path reconstruction, frame rendering,
AND the env itself (`SyntheticMazeGame`, `step_agent`). Factored out of the former
``topology_learning/`` POC, which also carried a large torch-based value-learning
benchmark that this repo never used at runtime.

Why the env lives here rather than in either consumer: it used to be written
TWICE -- once as ``_SyntheticMazeGame`` in ``solvers/generate_synthetic_geodesic.py``
(what the corpus was recorded from) and once as ``SyntheticGeodesicEnv`` in
``solver.py`` (what the policy is evaluated on). The two had silently diverged on
all three things that matter:

  * **seeding** -- the generator drew every level from ONE ``Random(seed)``, the
    live env from ``Random(seed * 1_000_003 + start_level)``, so a ``(seed, level)``
    at eval was not the ``(seed, level)`` the corpus ever contained;
  * **level structure** -- the generator built ``n_levels`` mazes and advanced
    through them (the auto-advance the in-context policy is trained to exploit),
    while the live env built exactly ONE maze, never incremented
    ``levels_completed``, and treated ``--level N`` as nothing but an RNG tweak;
  * **the movement rule** itself, duplicated in the generator's ``drive`` and the
    live env's ``step``.

Both consumers now import from here, so a change to the environment cannot reach
training without also reaching evaluation.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

# Env defaults -- shared by the generator and the live roll-out so the corpus and
# the evaluation are drawn from the same distribution by construction.
GRID_N = 7             # maze is GRID_N x GRID_N cells
FRAME_SIZE = 64        # rendered board edge, in pixels
GAMMA = 0.95           # make_demo's discount (unused by the env, kept for parity)
N_LEVELS = 10          # mazes per episode

# Action id (the GameAction enum value) <-> grid step (row, col).
ACTION_DELTA: dict[int, tuple[int, int]] = {
    1: (-1, 0),   # up    (row decreases)
    2: (1, 0),    # down
    3: (0, -1),   # left  (col decreases)
    4: (0, 1),    # right
}
DELTA_ACTION = {d: a for a, d in ACTION_DELTA.items()}
DELTAS = tuple(DELTA_ACTION)

# Maze render palette (matches the synthetic_geodesic corpus / encoder targets):
# 0 floor, 1 wall, 3 player, 14 goal.
P_FLOOR, P_WALL, P_PLAYER, P_GOAL = 0, 1, 3, 14


def gen_maze(n_cells, rng):
    """Recursive-backtracker perfect maze on a (2*n_cells+1)^2 grid. 1=wall, 0=open."""
    H = W = 2 * n_cells + 1
    grid = np.ones((H, W), np.uint8)
    stack = [(0, 0)]
    visited = {(0, 0)}
    grid[1, 1] = 0
    while stack:
        ci, cj = stack[-1]
        nbrs = [(ci + di, cj + dj, di, dj)
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= ci + di < n_cells and 0 <= cj + dj < n_cells
                and (ci + di, cj + dj) not in visited]
        if not nbrs:
            stack.pop(); continue
        ni, nj, di, dj = nbrs[rng.randrange(len(nbrs))]
        grid[1 + 2 * ci + di, 1 + 2 * cj + dj] = 0     # knock the wall between cells
        grid[1 + 2 * ni, 1 + 2 * nj] = 0
        visited.add((ni, nj)); stack.append((ni, nj))
    return grid


def bfs_dist(grid, goal):
    """Geodesic distance from every open cell to `goal` (INF for walls/unreachable)."""
    H, W = grid.shape
    dist = np.full((H, W), np.inf, np.float32)
    dist[goal] = 0.0
    dq = deque([goal])
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0 and dist[nr, nc] > dist[r, c] + 1:
                dist[nr, nc] = dist[r, c] + 1
                dq.append((nr, nc))
    return dist


def shortest_path(grid, start, goal, dist):
    path = [start]; cur = start
    while cur != goal:
        r, c = cur
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] \
                    and grid[nr, nc] == 0 and dist[nr, nc] == dist[cur] - 1:
                cur = (nr, nc); path.append(cur); break
        else:
            break
    return path


def make_demo(n_cells, gamma, rng):
    """One maze + goal + the FARTHEST start's optimal path (long, detour-rich)."""
    grid = gen_maze(n_cells, rng)
    open_cells = list(zip(*np.where(grid == 0)))
    goal = open_cells[rng.randrange(len(open_cells))]
    dist = bfs_dist(grid, goal)
    reach = np.isfinite(dist)
    rd = np.where(reach, dist, -1.0)
    start = tuple(int(v) for v in np.unravel_index(int(rd.argmax()), rd.shape))
    path = shortest_path(grid, start, goal, dist)
    return grid, goal, dist, reach, path


def render_frame(grid, goal, agent, frame_size):
    """Toy maze -> a maze-palette frame [frame_size,frame_size] (nearest-neighbour upscale)."""
    H, W = grid.shape
    cp = frame_size // H; off = (frame_size - cp * H) // 2
    f = np.full((frame_size, frame_size), P_WALL, np.int64)
    up = np.where(np.repeat(np.repeat(grid, cp, 0), cp, 1) > 0, P_WALL, P_FLOOR)
    f[off:off + cp * H, off:off + cp * W] = up
    def block(cell, col):
        r, c = cell
        f[off + r * cp:off + (r + 1) * cp, off + c * cp:off + (c + 1) * cp] = col
    block(goal, P_GOAL); block(agent, P_PLAYER)
    am = np.zeros_like(f, np.float32)
    r, c = agent
    am[off + r * cp:off + (r + 1) * cp, off + c * cp:off + (c + 1) * cp] = 1.0
    return f, am


# ---------------------------------------------------------------------------
# The environment (shared by the generator and the live roll-out)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MazeLevel:
    """One maze: the grid, the goal, the farthest start, and the geodesic field.

    ``dist`` is the BFS distance-to-goal over the whole grid (inf for walls). It is
    computed once because the maze never changes -- only the agent moves -- so an
    expert's `solve_from` / `optimal_set_from` are O(path) / O(1) lookups rather
    than a fresh search per step.
    """

    grid: np.ndarray            # [H, W] uint8, 1 = wall, 0 = open
    goal: tuple[int, int]
    start: tuple[int, int]      # the farthest reachable cell (long, detour-rich)
    dist: np.ndarray            # [H, W] float32 geodesic distance to `goal`


def make_maze(grid_n: int, rng) -> MazeLevel:
    """One fresh maze whose farthest-start optimal path has at least one move."""
    while True:
        grid, goal, dist, _reach, path = make_demo(grid_n, GAMMA, rng)
        if len(path) >= 2:                                # need at least one move
            return MazeLevel(grid=grid,
                             goal=(int(goal[0]), int(goal[1])),
                             start=(int(path[0][0]), int(path[0][1])),
                             dist=dist)


class SyntheticMazeGame:
    """Minimal stand-in for an arcengine game: N independent mazes as N levels.

    Mutable state is exactly ``(cur, agent)``, which is why a snapshot can be a
    two-tuple instead of a whole-game deepcopy.

    Every level comes from ONE ``Random(seed)`` drawn in order, so level ``i`` of
    seed ``s`` is the same maze for the generator and for the live roll-out.
    """

    def __init__(self, seed: int, n_levels: int = N_LEVELS, grid_n: int = GRID_N,
                 frame_size: int = FRAME_SIZE) -> None:
        import random as _random
        rng = _random.Random(seed)
        self.frame_size = frame_size
        self.levels = [make_maze(grid_n, rng) for _ in range(n_levels)]
        self.cur = 0
        self.agent = self.levels[0].start

    @property
    def level(self) -> MazeLevel:
        return self.levels[self.cur]

    def set_level(self, level_idx: int) -> None:
        """Seat level ``level_idx`` at its start cell."""
        self.cur = level_idx
        self.agent = self.levels[level_idx].start

    def advance(self) -> bool:
        """Move to the next level, seated at its start. False if none is left."""
        if self.cur + 1 >= len(self.levels):
            return False
        self.set_level(self.cur + 1)
        return True

    @property
    def solved(self) -> bool:
        return self.agent == self.level.goal


def step_agent(level: MazeLevel, agent: tuple[int, int], action_id: int):
    """The movement rule: step unless a wall (or the border) blocks it.

    Nothing here can kill the agent, and a blocked move is a legal no-op -- which
    is why the maze is fully reversible and needs no death handling."""
    dr, dc = ACTION_DELTA.get(int(action_id), (0, 0))
    r, c = agent
    nr, nc = r + dr, c + dc
    if (0 <= nr < level.grid.shape[0] and 0 <= nc < level.grid.shape[1]
            and level.grid[nr, nc] == 0):
        return (nr, nc)
    return agent


def render_game(game: SyntheticMazeGame) -> np.ndarray:
    """Current level + agent as one rendered board (drops render_frame's mask)."""
    lvl = game.level
    return np.asarray(render_frame(lvl.grid, lvl.goal, game.agent,
                                   game.frame_size)[0])
