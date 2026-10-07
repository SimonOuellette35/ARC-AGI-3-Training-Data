"""Solve one_step_tetris using live geometry and the shared BaseSolver harness.

Match suspended shapes to the gaps in the bottom row band by simulating their
vertical drops. No seed recipes or stored ``solution_x`` values are used.
An A* search then arranges the pieces with clicks and arrows before ACTION5.
Its state includes every column and the active piece: overlapping pieces are
selectable only at cells not occupied by an earlier piece in the game's list.
Searching temporary placements lets the solver recover even from those overlaps.

Plans use upright coordinates; BaseSolver handles screen rotation and recording.
The inherited CLI can generate a corpus when explicitly run, for example:
    python solvers/generate_one_step_tetris_training.py --episodes 1000
Importing the module or calling solve_from does not write any data.
"""

from __future__ import annotations

import heapq
import itertools
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.one_step_tetris.one_step_tetris import (  # noqa: E402
    H, SCALE, W, X_OFF, Y_OFF, OneStepTetris,
)
from solvers.base_solver import BaseSolver  # noqa: E402
from utils.explore import Action  # noqa: E402


def _click(x: int, y: int) -> Action:
    """Click the centre of an upright board cell."""
    return Action(6, (Y_OFF + y * SCALE + SCALE // 2,
                      X_OFF + x * SCALE + SCALE // 2))


def _drop(offsets, x, y, occupied):
    cells = {(x + dx, y + dy) for dx, dy in offsets}
    while all(cy + 1 < H and (cx, cy + 1) not in occupied
              for cx, cy in cells):
        cells = {(cx, cy + 1) for cx, cy in cells}
    return cells


@lru_cache(maxsize=128)
def _targets(shapes, heights, stack, required):
    """Find non-overlapping drops covering every gap in the bottom band.

    A notch can fill an entire column of that band, so contiguous runs of gaps
    do not necessarily delimit wells. Enumerate columns and cover missing cells
    directly instead. Final candidates are checked in the engine's drop order.
    """
    missing = {(x, y) for x in range(W) for y in range(H - required, H)
               if (x, y) not in stack}
    candidates = []
    for shape, y in zip(shapes, heights):
        width = max(dx for dx, _ in shape) + 1
        placements = []
        for x in range(W - width + 1):
            cells = _drop(shape, x, y, stack)
            if cells & missing:
                placements.append((x, cells))
        candidates.append(placements)

    targets = []
    before = sum(all((x, y) in stack for x in range(W)) for y in range(H))
    for assignment in itertools.product(*candidates):
        cells = set().union(*(placement[1] for placement in assignment))
        if not missing <= cells or len(cells) != sum(len(p[1]) for p in assignment):
            continue
        xs = tuple(p[0] for p in assignment)
        occupied = set(stack)
        for i in sorted(range(len(shapes)), key=lambda i: (xs[i], heights[i])):
            occupied.update(_drop(shapes[i], xs[i], heights[i], occupied))
        after = sum(all((x, y) in occupied for x in range(W)) for y in range(H))
        if after - before >= required:
            targets.append(xs)
    return tuple(targets)


def _arrange(shapes, heights, start, targets):
    """Shortest arrangement plan, including moves needed to expose a piece.

    Every legal column is reachable in one click on the bottom row, which is below
    all suspended pieces. Selection clicks follow the engine's first-hit priority.
    One representative per distinct successor suffices; adjacent slides use arrows.
    The lower bound charges one move per misplaced piece and one selection per
    misplaced inactive piece, ignoring any additional work needed to expose it.
    """
    widths = [max(dx for dx, _ in shape) + 1 for shape in shapes]
    goals = set(targets)

    def estimate(state):
        xs, active = state
        return min(2 * sum(x != goal[i] for i, x in enumerate(xs))
                   - (xs[active] != goal[active]) for goal in targets)

    serial = itertools.count()
    queue = [(estimate(start), 0, next(serial), start)]
    distance = {start: 0}
    parent = {}
    while queue:
        _, neg_cost, _, state = heapq.heappop(queue)
        cost = -neg_cost
        if cost != distance[state]:
            continue
        xs, active = state
        if xs in goals:
            plan = [Action(5)]
            while state != start:
                state, action = parent[state]
                plan.append(action)
            return list(reversed(plan))

        successors = []
        # A stack cell is fine: _click tests suspended pieces only.
        for x in range(W - widths[active] + 1):
            if x == xs[active]:
                continue
            moved = xs[:active] + (x,) + xs[active + 1:]
            delta = x - xs[active]
            action = Action(3 if delta < 0 else 4) if abs(delta) == 1 else _click(x, H - 1)
            successors.append(((moved, active), action))

        covered = set()
        for i, shape in enumerate(shapes):
            cells = {(xs[i] + dx, heights[i] + dy) for dx, dy in shape}
            exposed = cells - covered
            if i != active and exposed:
                x, y = min(exposed)
                successors.append(((xs, i), _click(x, y)))
            covered.update(cells)

        for successor, action in successors:
            new_cost = cost + 1
            if new_cost >= distance.get(successor, float("inf")):
                continue
            distance[successor] = new_cost
            parent[successor] = (state, action)
            heapq.heappush(queue, (new_cost + estimate(successor), -new_cost,
                                   next(serial), successor))
    return []


class OneStepTetrisSolver(BaseSolver):
    game_id = "one_step_tetris"
    supports_recovery = True
    recovery_mode = "replan"

    def make_game(self, seed: int):
        return OneStepTetris(seed=seed)

    def solve_from(self, game, level_idx: int, seed: int) -> list[Action]:
        if game._committed:
            return []
        pieces = game._pieces
        shapes = tuple(tuple(map(tuple, p["offsets"])) for p in pieces)
        heights = tuple(p["y"] for p in pieces)
        targets = _targets(shapes, heights, frozenset(game._stack), game._req)
        if not targets:
            return []
        start = (tuple(p["x"] for p in pieces), game._active)
        return _arrange(shapes, heights, start, targets)


if __name__ == "__main__":
    sys.exit(OneStepTetrisSolver.main())
