"""Generate Phase-1 training data for the Klotski (Block Slide) game.

A thin ``BaseSolver`` subclass over the local Klotski game
(games/klotski/klotski.py). An embedded A* expert solves every level for a seed;
``BaseSolver`` owns the record/replay loop, the episode schema, the WIN-filter
CLI, and the (here disabled) exploration prefix.

Klotski action set (index in 0..5 -- ACTION7/undo is NEVER used):
    ACTION1 (up)  ACTION2 (down)  ACTION3 (left)  ACTION4 (right)
    ACTION5 cycle the selected block

Rotation note: each level is presented at a random 90-degree rotation. Klotski is
an AugmentedGame, so ``set_level`` draws the deterministic per-(seed, level)
rotation and the game inverse-remaps directional input inside ``step()``. The A*
plans in engine (block) coordinates; ``solve_from`` converts each engine-space
direction to the SCREEN action the agent must press (via
``inverse_remap_action_full``) so the recorded index matches what drives the
rotated board. ACTION5 is non-directional and passes through.

One Klotski *seed* is a multi-level game (7 levels), so each WIN seed yields one
complete multi-level episode -- the base's default episode model fits directly.

supports_recovery is False: A* from an arbitrary perturbed board is expensive
(a weak target-only heuristic + the memory guard), so the game degrades to pure
optimal replay with exploration switched off.

Usage (run from the repo root):
    python solvers/generate_klotski_training.py --episodes 1000 \
        --out data/training_multi_level/klotski
"""

from __future__ import annotations

import heapq
import sys
from pathlib import Path

# Repo root (parent of solvers/) -- that's where the games/ package lives.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction  # noqa: E402
from games.klotski.klotski import _exit_x  # noqa: E402
from games.klotski.klotski import Klotski  # noqa: E402
from solvers.base_solver import BaseSolver  # noqa: E402
from utils.rotation import inverse_remap_action_full  # noqa: E402

# Hard ceiling on A* node expansions per level solve. The target-only Manhattan
# heuristic is weak, so on a pathological seed the open/closed structures would
# otherwise grow without bound and OOM the machine. Exceeding it raises
# RuntimeError -> the seed is skipped upstream (solve_episode catches it).
_MAX_EXPANSIONS = 2_000_000

# Engine-space directional moves (dx, dy) -> GameAction.
_DIRS = [
    ((0, -1), GameAction.ACTION1),   # up
    ((0,  1), GameAction.ACTION2),   # down
    ((-1, 0), GameAction.ACTION3),   # left
    ((1,  0), GameAction.ACTION4),   # right
]


def _can_move(positions, dims, idx, dx, dy, bw, bh):
    """True if block `idx` can slide by (dx, dy) without leaving the board or
    overlapping another block. Operates on plain position/dim tuples."""
    x, y = positions[idx]
    w, h = dims[idx]
    nx, ny = x + dx, y + dy
    if nx < 0 or nx + w > bw or ny < 0 or ny + h > bh:
        return False
    new_cells = {(nx + i, ny + j) for i in range(w) for j in range(h)}
    for j, (ox, oy) in enumerate(positions):
        if j == idx:
            continue
        ow, oh = dims[j]
        for cx in range(ox, ox + ow):
            for cy in range(oy, oy + oh):
                if (cx, cy) in new_cells:
                    return False
    return True


class KlotskiSolver(BaseSolver):
    game_id = "klotski"
    # ``solve_from`` runs A* over the LIVE block positions/selection, so it genuinely
    # re-plans from an arbitrary perturbed board (Klotski is fully reversible). Enable
    # exploration with REPLAN recovery; the target-only heuristic is weak but the
    # exploration prefix is short, and _MAX_EXPANSIONS guards a pathological detour.
    supports_recovery = True
    recovery_mode = "replan"

    def __init__(self, *args, max_expansions: int = _MAX_EXPANSIONS, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_expansions = max_expansions

    def make_game(self, seed: int):
        # AugmentedGame: set_level draws the deterministic per-(seed, level)
        # rotation, so the episode renders/remaps byte-for-byte on replay.
        return Klotski(seed=seed)

    def set_level(self, game, level_idx: int) -> None:
        # One reused instance spans all 7 levels; the native drive breaks on the
        # engine's ``_next_level`` flag without clearing it, so clear it here or
        # the next level is spuriously reported solved in a single action.
        game.set_level(level_idx)
        game._next_level = False

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4, 5]

    def solve_from(self, game, level_idx: int, seed: int):
        """A* over (block positions, selected index) from the game's LIVE state,
        returning the SCREEN actions to press (engine-space plan inverse-remapped
        through the level's live rotation ``game._rotation_k``)."""
        bw, bh = game._board_dims[level_idx]
        blocks = game._blocks
        dims = tuple((b.w, b.h) for b in blocks)
        n = len(blocks)
        start = tuple((b.x, b.y) for b in blocks)
        sel0 = int(game._sel)
        k = game._rotation_k

        exit_x = _exit_x(bw)
        goal_y = bh - dims[0][1]   # target height

        def is_goal(positions):
            tx, ty = positions[0]
            return tx == exit_x and ty == goal_y

        def heuristic(positions):
            tx, ty = positions[0]
            return abs(tx - exit_x) + abs(ty - goal_y)

        start_node = (start, sel0)
        g_cost = {start_node: 0}
        parent = {start_node: None}
        action_taken = {start_node: None}
        pq = [(heuristic(start), 0, start_node)]
        counter = 1
        goal_node = None
        expansions = 0

        while pq:
            f, _, node = heapq.heappop(pq)
            positions, sel = node
            if is_goal(positions):
                goal_node = node
                break
            base_g = g_cost[node]
            if f - heuristic(positions) > base_g:
                continue  # stale queue entry

            expansions += 1
            if expansions > self._max_expansions:
                raise RuntimeError(
                    f"A* exceeded {self._max_expansions} expansions for "
                    f"level={level_idx} seed={seed} (skipping seed)")

            # ACTION5: cycle selection (does not move any block).
            nxt = (positions, (sel + 1) % n)
            ng = base_g + 1
            if ng < g_cost.get(nxt, 1 << 30):
                g_cost[nxt] = ng
                parent[nxt] = node
                action_taken[nxt] = GameAction.ACTION5
                heapq.heappush(pq, (ng + heuristic(positions), counter, nxt))
                counter += 1

            # Directional moves on the selected block.
            for (dx, dy), action in _DIRS:
                if not _can_move(positions, dims, sel, dx, dy, bw, bh):
                    continue
                new_pos = list(positions)
                x, y = new_pos[sel]
                new_pos[sel] = (x + dx, y + dy)
                new_pos = tuple(new_pos)
                nxt = (new_pos, sel)
                ng = base_g + 1
                if ng < g_cost.get(nxt, 1 << 30):
                    g_cost[nxt] = ng
                    parent[nxt] = node
                    action_taken[nxt] = action
                    heapq.heappush(pq, (ng + heuristic(new_pos), counter, nxt))
                    counter += 1

        if goal_node is None:
            raise RuntimeError(f"No solution for level={level_idx} seed={seed}")

        engine_plan: list[GameAction] = []
        node = goal_node
        while action_taken[node] is not None:
            engine_plan.append(action_taken[node])
            node = parent[node]
        engine_plan.reverse()

        # Convert each engine-space direction to the screen action to press so
        # the engine's own remap_action(..., k) turns it back into engine_action.
        return list(engine_plan)


if __name__ == "__main__":
    sys.exit(KlotskiSolver.main())
