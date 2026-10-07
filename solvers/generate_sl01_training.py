"""Generate Phase-1 training data for sl01 (Slide Puzzle -- the 8-puzzle).

sl01 (games/sl01/sl01.py) is the classic sliding-tile puzzle: a 3x3 board holds
tiles 1..8 plus one hole, and each action swaps the hole with the orthogonally
adjacent tile above / below / left / right of it. A level is won the instant the
board matches ``level.data["goal"]``. One sl01 *game* is 5 levels of increasing
scramble depth, so every WIN seed yields one complete multi-level episode -- the
base's default episode model fits directly.

sl01 action set (simple actions only; the hole is what moves):
    ACTION1 (hole up)  ACTION2 (hole down)  ACTION3 (hole left)  ACTION4 (right)

THE EXPERT IS AN EXACT DISTANCE FIELD, NOT A SEARCH PER QUERY. A 3x3 board is a
permutation of the 9 values ``0..8`` (0 = hole), so the entire state space is
``9! = 362 880`` states -- small enough to hold a byte of distance-to-goal for
*every* one of them. A move (swap the hole with a neighbour) is its own inverse,
so a single BFS *from the goal board* labels the whole reachable half of that
space (181 440 states, deepest 31 moves) with its exact optimal distance. Then:

  * ``solve_from`` is a greedy descent down the field, so it is optimal from ANY
    board at ~zero cost. That is what makes ``supports_recovery`` free here: an
    exploration prefix or a perturbation burst just lands the agent on a
    different cell of the same precomputed field, and the very next expert step
    plans the true optimal recovery from there.
  * ``optimal_set_from`` is a 4-way lookup, so the recorded target is the FULL
    set of equally-optimal moves at no extra cost. The 8-puzzle is rich in
    co-optimal moves, so the base's stochastic-optimal sampling makes each
    episode take a genuinely different (still perfect) route to the goal.
  * the puzzle is fully REVERSIBLE and legal play never leaves the reachable
    parity class, so no reachable board is ever a dead end: the field is set
    everywhere the agent can actually get to, and ``solve_from`` never has to
    give up. RESET recovery therefore stays unused in practice.

The BFS is vectorised over whole frontiers with numpy (a frontier is an
``(M, 9)`` array of boards; ranking a frontier is a couple of array ops), which
fills the whole field in well under a second -- paid ONCE per process and then
shared by every seed, level, replan and burst.

Step limit: sl01 counts steps upward against ``level.data["step_limit"]`` rather
than through an engine ``StepCounter``, so the base's generic probe cannot see
it. ``_step_counter_probe`` is overridden to expose ``step_limit - steps``, which
lets the base lift step-exhaustion death during generation -- exploration
overhead must never discard a trajectory.

Rotation: sl01 is a plain ``ARCBaseGame`` (not an ``AugmentedGame``) and reads
``self.action.id.value`` directly, so there is no rotation to invert -- game
space and screen space coincide and plans are emitted as-is.

Usage (run from the repo root):
    python solvers/generate_sl01_training.py --episodes 1000 \
        --out data/training_multi_level/sl01
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402

from arcengine import GameAction                                # noqa: E402
from games.sl01.sl01 import Sl01                                # noqa: E402
from solvers.base_solver import BaseSolver                       # noqa: E402

# Direction deltas for the HOLE, index-aligned with ``_ACTIONS`` and with the
# game's own ACTION1..4 -> (dx, dy) mapping in ``Sl01.step``.
_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))
_ACTIONS = (GameAction.ACTION1, GameAction.ACTION2,
            GameAction.ACTION3, GameAction.ACTION4)

# Distances are stored as ``d + 1`` in a uint8 field (0 means "not reachable
# from the goal"), so the deepest representable optimal solution is 254 moves.
# The 8-puzzle's true diameter is 31, so there is no risk of overflow; assert it
# anyway rather than silently wrapping if a bigger board ever appears.
_MAX_DEPTH = 254

# The field is a flat ``n!`` byte array, so it only stays a lookup table for
# small boards. 3x3 (9! = 363 KB) is what sl01 ships and what its metadata pins
# (``grid_range: [3, 3]``); 10 cells would already be 3.6 MB and 11 would be
# 40 MB, so refuse past that rather than quietly exhausting memory.
_MAX_CELLS = 10


class _Field:
    """Exact distance-to-goal field for one (board geometry, goal board) pair.

    A board is the flat tuple of its ``n`` cell values -- a permutation of
    ``0..n-1`` with 0 the hole -- and is indexed by its Lehmer (lexicographic)
    rank, so the field is a flat ``n!`` byte array and a whole BFS frontier is
    just an ``(M, n)`` int8 array of boards.
    """

    def __init__(self, gw: int, gh: int, goal: tuple[int, ...]) -> None:
        n = gw * gh
        if n > _MAX_CELLS:
            raise ValueError(
                f"sl01 field assumes at most {_MAX_CELLS} cells, got {n} "
                f"({gw}x{gh}); a table over n! states no longer fits")
        if tuple(sorted(goal)) != tuple(range(n)):
            raise ValueError(
                f"sl01 board must be a permutation of 0..{n - 1}, got {goal}")
        self.gw, self.gh, self.n = gw, gh, n

        # ``_weights[i] = (n-1-i)!`` -- the place value of position i in the
        # Lehmer code, which is what turns a board into its rank.
        self._weights = np.array([math.factorial(n - 1 - i) for i in range(n)],
                                 dtype=np.int64)
        # ``_later`` masks the (i, j) pairs with j > i, so counting
        # ``board[i] > board[j]`` under it gives the Lehmer digits.
        self._later = np.triu(np.ones((n, n), dtype=bool), 1)

        # ``nbr[d][c]`` = the cell one step in direction d from c, or -1 when
        # that is off the board (the game's "edge bump": a no-op move).
        nbr = np.full((4, n), -1, dtype=np.int64)
        for d, (dx, dy) in enumerate(_DELTAS):
            for y in range(gh):
                for x in range(gw):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < gw and 0 <= ny < gh:
                        nbr[d][y * gw + x] = ny * gw + nx
        self.nbr = nbr

        self.dist = np.zeros(math.factorial(n), dtype=np.uint8)
        self._bfs(goal)

    # ── ranking ─────────────────────────────────────────────────────────────
    def rank(self, board) -> int:
        """Lehmer rank of one board (a tuple/list of ``n`` distinct values)."""
        r = 0
        for i, v in enumerate(board):
            r += sum(1 for u in board[i + 1:] if u < v) * int(self._weights[i])
        return r

    def _ranks(self, boards: np.ndarray) -> np.ndarray:
        """Lehmer ranks of an ``(M, n)`` array of boards, all at once."""
        digits = ((boards[:, :, None] > boards[:, None, :]) & self._later).sum(2)
        return digits @ self._weights

    # ── the one BFS ─────────────────────────────────────────────────────────
    def _bfs(self, goal: tuple[int, ...]) -> None:
        """Fill ``dist`` with (optimal moves to the goal + 1), layer by layer.

        Swapping the hole with a neighbour is an INVOLUTION, so sl01's
        transition relation is symmetric and one forward BFS out of the goal
        board yields distance-*to*-the-goal for every state it reaches. Every
        layer is a handful of numpy ops over the whole frontier: find each
        board's hole, swap it with the neighbour in each of the 4 directions,
        rank the results, and keep the ones not yet labelled."""
        n, nbr, dist = self.n, self.nbr, self.dist
        frontier = np.array([goal], dtype=np.int8)
        dist[self.rank(goal)] = 1

        for depth in range(2, _MAX_DEPTH + 2):
            hole = np.argmax(frontier == 0, axis=1)
            nxt = []
            for d in range(4):
                target = nbr[d][hole]
                ok = target >= 0
                if not ok.any():
                    continue
                boards = frontier[ok].copy()
                rows = np.arange(boards.shape[0])
                h, t = hole[ok], target[ok]
                # Fancy indexing on the RHS copies, so this is a true swap.
                boards[rows, h], boards[rows, t] = boards[rows, t], boards[rows, h]
                nxt.append(boards)
            if not nxt:
                return
            boards = np.concatenate(nxt)
            # One representative row per distinct successor, then drop the ones
            # a shallower layer already labelled.
            codes, first = np.unique(self._ranks(boards), return_index=True)
            fresh = dist[codes] == 0
            if not fresh.any():
                return
            codes, first = codes[fresh], first[fresh]
            if depth > _MAX_DEPTH:
                raise RuntimeError(
                    f"sl01 BFS exceeded depth {_MAX_DEPTH}; the field's uint8 "
                    f"distances would overflow")
            dist[codes] = depth
            frontier = boards[first]

    # ── queries ─────────────────────────────────────────────────────────────
    def steps_to_win(self, board) -> int | None:
        """Optimal moves from ``board`` to the goal, or None if it cannot reach
        the goal at all (the other permutation-parity class -- unreachable by
        legal play, so this never fires on a live board)."""
        d = int(self.dist[self.rank(board)])
        return None if d == 0 else d - 1

    def successor(self, board, d: int):
        """sl01's forward transition for direction ``d``, or None when the move
        changes nothing (the hole is against that edge). Mirrors ``Sl01.step``:
        the hole moves to the target cell and the tile there moves to the hole."""
        h = board.index(0)
        t = int(self.nbr[d][h])
        if t < 0:                                    # edge bump: nothing moves
            return None
        nxt = list(board)
        nxt[h], nxt[t] = nxt[t], nxt[h]
        return tuple(nxt)

    def optimal_dirs(self, board) -> list[int]:
        """Every direction that strictly descends the field (the equally-optimal
        moves), in ACTION1..4 order."""
        here = self.steps_to_win(board)
        if not here:                                 # dead, or already solved
            return []
        out = []
        for d in range(4):
            nxt = self.successor(board, d)
            if nxt is None:
                continue
            there = self.steps_to_win(nxt)
            if there is not None and there == here - 1:
                out.append(d)
        return out


class Sl01Solver(BaseSolver):
    game_id = "sl01"
    # ``solve_from`` is a lookup in an exact distance-to-goal field over the LIVE
    # board, so it re-plans optimally from any reachable position -- including one
    # an exploratory detour or a perturbation burst left behind. The puzzle is
    # fully reversible, so replanning is always possible and RESET is never the
    # only way out.
    supports_recovery = True
    recovery_mode = "replan"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The field depends ONLY on the board geometry and the goal, never on the
        # current scramble, so one entry serves every level of every seed (all
        # sl01 levels share the canonical goal) -- one BFS per process.
        self._fields: dict[tuple, _Field] = {}

    def make_game(self, seed: int):
        # Sl01 takes no seed: its 5 levels are hand-authored fixed scrambles, so
        # every seed sees the same boards and the per-episode variety comes from
        # the exploration prefix, the perturbation bursts and the stochastic
        # sampling among co-optimal moves (all driven by ``self.rng``).
        return Sl01()

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    def _step_counter_probe(self, game):
        """sl01 counts steps UP against ``level.data["step_limit"]`` instead of
        using an engine ``StepCounter``, so the base's generic probe finds
        nothing and step-exhaustion death would stay armed during generation.
        Expose the remaining budget so ``_lift_step_limit`` can neutralise it:
        the HUD block still flips to the out-of-steps colour, but a long
        exploration prefix can no longer throw the episode away. sl01 has no
        other loss condition, so nothing real is masked."""
        def remaining():
            limit = getattr(game, "_step_limit", None)
            if limit is None:
                return None
            return int(limit) - int(getattr(game, "_steps", 0))
        return remaining

    # ── live state / field ──────────────────────────────────────────────────
    @staticmethod
    def _live_board(game):
        """The LIVE board read off the game's sprites, as
        ``(gw, gh, flat_board, flat_goal)`` -- flat row-major tuples of cell
        values, 0 for the hole and the tile id (1..8) for a tile."""
        level = game.current_level
        gw, gh = level.grid_size
        board = [0] * (gw * gh)
        for tile in level.get_sprites_by_tag("tile"):
            tile_id = next(int(t) for t in tile.tags if t.isdigit())
            board[tile.y * gw + tile.x] = tile_id
        goal = tuple(int(v) for row in game._goal for v in row)
        return gw, gh, tuple(board), goal

    def _field(self, game) -> _Field:
        gw, gh, _board, goal = self._live_board(game)
        key = (gw, gh, goal)
        if key not in self._fields:
            self._fields[key] = _Field(gw, gh, goal)
        return self._fields[key]

    # ── the solver API ──────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int):
        """An optimal plan from the game's LIVE board: greedy descent down the
        distance field, breaking ties at random so repeated episodes take
        different (equally optimal) routes. ``[]`` only if the live board cannot
        reach the goal, which legal play cannot produce."""
        field = self._field(game)
        _gw, _gh, board, _goal = self._live_board(game)
        remaining = field.steps_to_win(board)
        if not remaining:
            return []
        plan = []
        while remaining:
            dirs = field.optimal_dirs(board)
            if not dirs:                             # unreachable: field is exact
                return []
            d = self.rng.choice(dirs)
            plan.append(_ACTIONS[d])
            board = field.successor(board, d)
            remaining -= 1
        return plan

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-optimal next action at the LIVE board -- the moves that
        strictly descend the distance field. A 4-way lookup, so the full tie set
        is recorded as the training target at no extra cost."""
        field = self._field(game)
        _gw, _gh, board, _goal = self._live_board(game)
        return [_ACTIONS[d] for d in field.optimal_dirs(board)]


if __name__ == "__main__":
    sys.exit(Sl01Solver.main())
