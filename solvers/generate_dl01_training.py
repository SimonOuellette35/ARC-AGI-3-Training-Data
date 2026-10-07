"""Generate Phase-1 training data for the Delay-Line (dl01) game.

A thin ``BaseSolver`` subclass (record/replay, schema, WIN-filter CLI and the
optional exploration prefix all live in ``solvers/base_solver.py``). dl01 is a
navigation game with a one-step "delay line" between input and movement. The
board is randomised per seed from the instance RNG (a whole-board 90-degree
rotation + h/v flips, plus background / agent / wall colours). This generator
makes that augmentation DETERMINISTIC per (seed, level): ``set_level`` reseeds
``game._rng`` with ``dl01:<seed>:<level>`` right before ``game.set_level``, so one
episode == one seed, reproducible byte-for-byte, one episode per seed.

Game mechanics (see games/dl01/dl01.py):
    ACTION1..4 enqueue a cardinal move into a FIFO queue (max 3 pending):
        ACTION1 -> up (0,-1)  ACTION2 -> down (0,1)
        ACTION3 -> left (-1,0)  ACTION4 -> right (1,0)
    ACTION5 clears the queue.
    Each step first pops+applies the OLDEST queued move (if any), THEN enqueues
    the current action's direction. A move is a blocked no-op (still dequeued)
    against walls, hazards or grid bounds -- there is no death. Reaching the goal
    calls next_level().

Because the queue starts EMPTY and every directional press pops one + appends
one, the queue length stays at 1 while pressing directions: the move enqueued at
step t executes at step t+1 (a fixed one-step delay). To execute a desired move
list [m1..mN] we press the directions [m1, m2, ..., mN, FLUSH] where FLUSH is any
extra directional press that pops+executes mN (FLUSH's own enqueued direction is
never executed -- the level is already solved).

``solve_from`` runs BFS over the live level grid (avoiding walls/hazards/bounds)
for a shortest player->goal path, then wraps it with the delay-line encoding. It
also reads the LIVE pending-move FIFO (``game._q``, <=3 entries): those queued
moves fire (oldest first) before the next pressed direction lands, so the BFS
starts from where the queue drains the player TO and the plan presses one FLUSH
per queued move (>=1). That makes the plan valid from an ARBITRARY perturbed
state, so ``supports_recovery = True`` with ``recovery_mode = "replan"``
(exploration + perturbation bursts ON). Only simple actions (1..5) emit.

Usage (run from the repo root):
    python solvers/generate_dl01_training.py --episodes 1000 \
        --out data/training_multi_level/dl01
"""

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction                             # noqa: E402
from games.dl01.dl01 import Dl01                             # noqa: E402
from solvers.base_solver import BaseSolver                   # noqa: E402

# Cardinal move (dx, dy) -> directional simple action.
_DIR_ACTION = {
    (0, -1): GameAction.ACTION1,  # up
    (0, 1): GameAction.ACTION2,   # down
    (-1, 0): GameAction.ACTION3,  # left
    (1, 0): GameAction.ACTION4,   # right
}


def _blocked_cells(game) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for tag in ("wall", "hazard"):
        for sp in game.current_level.get_sprites_by_tag(tag):
            cells.add((sp.x, sp.y))
    return cells


def _bfs_path(start, target, blocked, gw, gh):
    """Shortest cardinal path start->target as a list of (dx, dy) moves."""
    if start == target:
        return []
    prev: dict[tuple[int, int], tuple[int, int]] = {start: None}
    q = deque([start])
    while q:
        x, y = q.popleft()
        if (x, y) == target:
            break
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < gw and 0 <= ny < gh):
                continue
            if (nx, ny) in blocked or (nx, ny) in prev:
                continue
            prev[(nx, ny)] = (x, y)
            q.append((nx, ny))
    if target not in prev:
        return None
    moves: list[tuple[int, int]] = []
    cur = target
    while prev[cur] is not None:
        px, py = prev[cur]
        moves.append((cur[0] - px, cur[1] - py))
        cur = (px, py)
    moves.reverse()
    return moves


class Dl01Solver(BaseSolver):
    game_id = "dl01"
    # solve_from reads BOTH the live grid AND the live one-step delay FIFO
    # (``game._q``), folding the pending queued moves into the BFS start state, so
    # it re-plans correctly from an ARBITRARY perturbed state (not just the
    # queue-empty initial one) -> REPLAN recovery: after an exploratory detour /
    # burst the base just re-calls solve_from on the live game; RESET/undo is only
    # the last resort for a genuinely unsolvable state.
    supports_recovery = True
    recovery_mode = "replan"

    def make_game(self, seed: int):
        game = Dl01()
        game._gen_seed = seed          # stashed for the per-level RNG reseed
        return game

    def set_level(self, game, level_idx: int) -> None:
        """Deterministic per (seed, level): reseed the RNG that
        ``on_set_level -> _randomize_level`` (rotation/flips/colours) consumes,
        then set the level.

        Must go through ``super().set_level`` (not ``game.set_level``): the base
        also clears the engine's pending ``_next_level`` flag, which `drive`
        deliberately leaves raised after a win. Left set, the FIRST action of
        every subsequent level is mis-read as an instant solve and each level
        after the first collapses to a bogus 2-frame "win"."""
        game._rng = random.Random(f"dl01:{game._gen_seed}:{level_idx}")
        super().set_level(game, level_idx)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    def solve_from(self, game, level_idx: int, seed: int):
        """BFS the live grid for a shortest player->goal path, wrapped with the
        one-step delay-line encoding, VALID FROM ANY LIVE STATE.

        The live pending-move FIFO (``game._q``, <=3 entries) is folded into the
        start: those queued moves execute (oldest first) BEFORE the next pressed
        direction lands, so the BFS starts from where the queue drains the player
        TO, and the plan presses one FLUSH per queued move (>=1) to pop the whole
        queue out (the final FLUSH's own enqueued direction is never executed --
        the level is already solved). At the queue-empty initial state this is
        exactly the old ``[moves..., FLUSH]`` encoding."""
        player = game.current_level.get_sprites_by_tag("player")[0]
        goal = game.current_level.get_sprites_by_tag("goal")[0]
        gw, gh = game.current_level.grid_size
        blocked = _blocked_cells(game)
        gx, gy = goal.x, goal.y

        # Drain the LIVE delay queue: advance the start past each queued move (a
        # move into a wall/hazard/out-of-bounds is a blocked no-op == stay put).
        forced = list(getattr(game, "_q", ()))
        sx, sy = player.x, player.y
        reached_goal = False
        for dx, dy in forced:
            nx, ny = sx + dx, sy + dy
            if 0 <= nx < gw and 0 <= ny < gh and (nx, ny) not in blocked:
                sx, sy = nx, ny
            if (sx, sy) == (gx, gy):           # a queued move already wins
                reached_goal = True
                break

        # One FLUSH per queued move (>=1) pops the whole queue.
        flush = [GameAction.ACTION1] * max(len(forced), 1)
        if reached_goal:
            return flush                       # draining the queue alone solves

        moves = _bfs_path((sx, sy), (gx, gy), blocked, gw, gh)
        if moves is None:
            return []                          # unreachable -> fail / trigger undo
        return [_DIR_ACTION[m] for m in moves] + flush

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Return ``None`` -> the base uses the single canonical head of
        ``solve_from``'s plan as the training target.

        ``solve_from`` now folds the live delay queue into its start state, so it
        DOES plan correctly from a perturbed (non-empty-queue) state -- that is
        what makes ``recovery_mode == 'replan'`` and the perturbation bursts safe.
        We still emit only the single head rather than the full geometric tie set:
        deriving that set would need a per-candidate ``_copy_game`` probe (off
        here) and the delay-line game is effectively single-path for training, so
        the single-head target with no extra solves is the right trade."""
        return None


if __name__ == "__main__":
    sys.exit(Dl01Solver.main())
