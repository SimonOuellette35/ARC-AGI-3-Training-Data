"""Generate Phase-1 training data for the TB01 game (games/tb01/tb01.py).

TB01 ("Bridge Builder") is a 24x24 pathfinding puzzle over cyan water: maroon
3x3 islands are safe ground, gray reefs are impassable, and the level is solved
the instant the player stands on a cell of the green goal island. ACTION1-4 move
the player one cell; an ACTION6 click TOGGLES a bridge (color 12) on an open
water cell, turning it into walkable ground. Stepping into unbridged water costs
a life and teleports the player back to the level's start cell (three deaths =
GAME_OVER); stepping into a reef or off-grid is a no-op that still burns a step.

The expert is therefore a plain weighted shortest path. Every action costs one
step, so walking onto ground costs 1 and walking onto water costs 2 (the click
that bridges it, then the move); reefs are infinite. The minimum number of
actions from the current state is exactly the weighted distance from the player
to the goal island, which one Dijkstra over the 576 cells answers -- fast enough
to re-run from scratch at every single step.

Which is what makes recovery free (`supports_recovery`, "replan"): the whole
state that matters -- where the player is, which cells are bridged -- is re-read
from the LIVE frame on every call, so an exploration prefix or a perturbation
burst that walks the player somewhere else, drowns it back to the start cell, or
toggles a bridge out from under it is simply a different start state for the same
Dijkstra. Nothing is cached from the level's initial state, and nothing the game
does is irreversible: a removed bridge can be re-clicked and a death restores the
start cell with the bridges intact.

Nothing is read from hidden engine state ([[no-privileged-solvers]]): the board
is decoded from the rendered 64x64 frame by sampling each grid cell's centre.
The one cell the frame cannot show is the terrain UNDER the player (the player
sprite is drawn on top of it), and it does not need to: it is treated as ground,
and a shortest path never re-enters the cell it starts from.

Per-episode variety comes from the co-optimal tie set rather than the board (the
game's five levels are hand-authored and take no seed). It is a big set: any
water cell that lies on ANY shortest path may be bridged next -- bridges are
order-independent, so "click ahead, walk later" and "bridge just-in-time" cost
exactly the same -- plus whichever first steps begin a shortest path. See
`optimal_set_from`.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_tb01_training.py --episodes 1000 \
        --out data/training_multi_level/tb01
"""
from __future__ import annotations

import heapq
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                         # noqa: E402

from games.tb01.tb01 import Tb01                           # noqa: E402
from solvers.base_solver import BaseSolver                 # noqa: E402
from utils.explore import Action, CLICK_ACTION             # noqa: E402

# Palette (games/tb01/tb01.py). Read off the frame, never off the game.
WATER = 10          # COLOR_BG -- open sea; walking into it drowns
REEF = 3            # COLOR_ROCK -- can be neither bridged nor stood on
BRIDGE = 12         # COLOR_BRIDGE -- a placed bridge; walkable ground
ISLAND = 13         # COLOR_ISLAND -- maroon island / waypoint
GOAL = 14           # COLOR_GOAL_ISLAND -- the green goal island
PLAYER = 9          # COLOR_PLAYER -- drawn on top of whatever it stands on

#: Cells the player can stand on. PLAYER is here because the frame hides the
#: terrain beneath it; see the module docstring.
GROUND = (BRIDGE, ISLAND, GOAL, PLAYER)

#: Entry costs, in ACTIONS, of stepping onto a cell: ground is one move, water is
#: one click plus one move. 0 marks impassable (reef, or any unexpected colour).
COST_GROUND = 1
COST_WATER = 2
IMPASSABLE = 0

INF = 1 << 30

#: (dy, dx) -> the movement action that produces it (tb01's ``step``).
_DIR_ACTION = {(-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}
#: Fixed order, so the canonical plan is a pure function of the state (no
#: dependence on dict/set iteration order -- PYTHONHASHSEED must not matter).
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _dijkstra(cost: np.ndarray, sources, *, backward: bool = False) -> np.ndarray:
    """Weighted distances over the grid, where the weight of an edge is the ENTRY
    cost of the cell being stepped onto (``cost``, with 0 meaning impassable).

    ``backward=False``: ``dist[c]`` = cost of walking from a source to ``c``,
    i.e. ``dist[c] = min(dist[n] + cost[c])`` over neighbours ``n``.

    ``backward=True``: ``dist[c]`` = cost of walking from ``c`` to a source,
    i.e. ``dist[c] = min(cost[n] + dist[n])`` -- the cost of the first step OUT
    of ``c`` plus the rest. Run from the goal cells this gives steps-to-win from
    every cell, which is the solver's distance oracle.
    """
    gh, gw = cost.shape
    dist = np.full((gh, gw), INF, dtype=np.int64)
    pq: list[tuple[int, int, int]] = []
    for sy, sx in sources:
        if dist[sy, sx] == 0:                   # a repeated source cell
            continue
        dist[sy, sx] = 0
        heapq.heappush(pq, (0, int(sy), int(sx)))
    while pq:
        d, y, x = heapq.heappop(pq)
        if d > dist[y, x]:
            continue
        for dy, dx in _DIRS:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < gh and 0 <= nx < gw):
                continue
            if cost[ny, nx] == IMPASSABLE:      # can't stand there either way
                continue
            step = int(cost[y, x] if backward else cost[ny, nx])
            nd = d + step
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                heapq.heappush(pq, (nd, ny, nx))
    return dist


class _Board:
    """The whole planning state, decoded from one rendered frame."""

    __slots__ = ("cells", "cost", "player", "goals", "to_win", "from_player")

    def __init__(self, cells: np.ndarray) -> None:
        self.cells = cells
        cost = np.full(cells.shape, IMPASSABLE, dtype=np.int64)
        cost[np.isin(cells, GROUND)] = COST_GROUND
        cost[cells == WATER] = COST_WATER
        self.cost = cost
        players = np.argwhere(cells == PLAYER)
        self.player = (int(players[0][0]), int(players[0][1])) if len(players) else None
        self.goals = [(int(y), int(x)) for y, x in np.argwhere(cells == GOAL)]
        # steps-to-win from every cell, and steps-from-the-player to every cell
        self.to_win = (_dijkstra(cost, self.goals, backward=True)
                       if self.goals else np.full(cells.shape, INF, dtype=np.int64))
        self.from_player = (_dijkstra(cost, [self.player])
                            if self.player else np.full(cells.shape, INF, dtype=np.int64))

    @property
    def distance(self) -> int:
        """Minimum actions from the player's cell to a win, or ``INF``."""
        if self.player is None:
            return INF
        return int(self.to_win[self.player])


class Tb01Solver(BaseSolver):
    game_id = "tb01"

    #: `solve_from` re-derives everything from the live frame and tb01 has no
    #: irreversible move (a bridge can be re-clicked; a drowning restores the
    #: start cell but keeps the bridges), so bursts and the exploration prefix
    #: are recovered from by re-planning, never by RESET.
    supports_recovery = True
    recovery_mode = "replan"
    #: The exploration prefix flails around a 3x3 island surrounded by water, so
    #: it drowns often; three deaths is a GAME_OVER, and each one costs a RESET.
    max_resets = 8

    #: This solver plans off the RENDERED FRAME (`_board`), which is the board as
    #: shown -- i.e. at the level's rotation. So it solves the rotated board directly
    #: and the plan it returns is already in screen space: it is recorded as-is, and
    #: `drive` de-rotates on the way into the core game. Nothing un-rotates the frame.
    plans_in_screen_space = True

    #: One-entry ``(frame key) -> _Board`` memo; see `_board`. Content-addressed,
    #: so it is not solver STATE: a burst rollback cannot leave it stale, because
    #: a restored frame simply hashes back to its own entry.
    _board_cache: tuple | None = None

    def make_game(self, seed: int):
        # Tb01 takes no seed: its five levels are hand-authored, so the boards are
        # the same every episode and the variety comes from the exploration
        # prefix, the bursts, and sampling among co-optimal actions (`self.rng`).
        return Tb01()

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4, CLICK_ACTION]

    # ── board reading: FROM THE FRAME, never from engine state ───────────────
    def _cell_geometry(self, game) -> tuple[int, int, int, int, int]:
        scale, offx, offy = game.camera._calculate_scale_and_offset()
        return scale, offx, offy, int(game.camera.width), int(game.camera.height)

    def _board(self, game) -> _Board:
        """Decode the live 64x64 frame into a `_Board` (memoised on the frame's
        bytes: the frame IS the state this solver plans from, so a content key is
        exact -- and it makes the pair of `solve_from` / `optimal_set_from` calls
        the base does per step share one decode + one pair of Dijkstras)."""
        frame = np.asarray(self.render(game))
        key = (frame.shape, frame.tobytes())
        if self._board_cache is not None and self._board_cache[0] == key:
            return self._board_cache[1]
        scale, offx, offy, gw, gh = self._cell_geometry(game)
        rows = offy + np.arange(gh) * scale + scale // 2
        cols = offx + np.arange(gw) * scale + scale // 2
        board = _Board(frame[np.ix_(rows, cols)])           # cells[gy, gx]
        self._board_cache = (key, board)
        return board

    def _click(self, game, gx: int, gy: int) -> Action:
        """A click Action on the centre of grid cell ``(gx, gy)``, in the engine's
        display-pixel space -- the inverse of ``camera.display_to_grid``."""
        scale, offx, offy, _, _ = self._cell_geometry(game)
        px = int(gx * scale + scale // 2 + offx)
        py = int(gy * scale + scale // 2 + offy)
        return Action(CLICK_ACTION, (py, px))               # click_rc = (row=y, col=x)

    # ── the expert ───────────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int):
        """An optimal plan from the game's CURRENT state: walk the steps-to-win
        field downhill from the player, bridging each water cell just before
        stepping onto it.

        The plan's length is exactly the field's value at the player, because
        that value counts the clicks too. Bridging just-in-time (rather than
        laying the whole span first) costs the same and leaves less for a later
        burst to un-toggle behind us."""
        board = self._board(game)
        if board.player is None or board.distance >= INF:
            return []                                       # nothing reachable
        cells, cost, to_win = board.cells, board.cost, board.to_win
        gh, gw = cells.shape
        y, x = board.player
        plan: list[Action] = []
        while to_win[y, x] > 0:
            for dy, dx in _DIRS:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < gh and 0 <= nx < gw):
                    continue
                if cost[ny, nx] == IMPASSABLE:
                    continue
                if int(cost[ny, nx]) + int(to_win[ny, nx]) == int(to_win[y, x]):
                    break
            else:                                           # field is broken
                return []
            if cells[ny, nx] == WATER:
                plan.append(self._click(game, nx, ny))       # bridge it first
            plan.append(Action(_DIR_ACTION[(dy, dx)]))
            y, x = ny, nx
        return plan

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """STOCHASTIC OPTIMAL: every action that drops steps-to-win by exactly one.

        Two families, and the second is what makes tb01's tie set large:

        * a MOVE onto a ground neighbour that starts a shortest path
          (``1 + to_win[n] == D``);
        * a CLICK on any water cell ``c`` lying on ANY shortest path
          (``from_player[c] + to_win[c] == D``). Bridging is order-independent
          and pays for itself exactly once, so laying a plank ten cells ahead is
          worth precisely as much as laying the one under your feet -- and a cell
          off every shortest path is not, since the click would be wasted.

        Both leave a state whose plan is one action shorter, which is the
        invariant the base relies on when it samples one and re-plans. Clicking a
        cell that already holds a bridge would REMOVE it (tb01's click toggles),
        so only water is ever offered."""
        board = self._board(game)
        if board.player is None:
            return None                                     # base falls back
        D = board.distance
        if D >= INF:
            return None
        cells, cost, to_win = board.cells, board.cost, board.to_win
        gh, gw = cells.shape
        py, px = board.player
        out: list[Action] = []
        for dy, dx in _DIRS:
            ny, nx = py + dy, px + dx
            if not (0 <= ny < gh and 0 <= nx < gw):
                continue
            if cost[ny, nx] == COST_GROUND and 1 + int(to_win[ny, nx]) == D:
                out.append(Action(_DIR_ACTION[(dy, dx)]))
        on_path = (cells == WATER) & (board.from_player + to_win == D)
        for wy, wx in np.argwhere(on_path):
            out.append(self._click(game, int(wx), int(wy)))
        return out or None


if __name__ == "__main__":
    sys.exit(Tb01Solver.main())
