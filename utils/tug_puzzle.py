"""Exact transition model + distance-to-win field for the tk01 "telekinetic tug".

ONE statement of the mechanic, used by both sides of the game:

  * ``games/tk01/tk01.py`` builds a field while SAMPLING a level, so a layout is
    accepted only if it is genuinely solvable and its optimal solution has the
    length the difficulty ramp asks for (solvability by search, exactly -- not by
    a hand-written certificate);
  * ``solvers/generate_tk01_training.py`` builds the same field as its EXPERT:
    ``solve_from`` is a greedy descent down it and ``optimal_set_from`` a 5-way
    lookup.

Keeping the two in one place is what stops them drifting; the module is verified
against the real engine by an exhaustive differential test (every ``(player,
block)`` state x every action, per level) -- see ``scratchpad/verify_tk01.py``.

THE MECHANIC (game space, before the display rotation)::

    ACTION1/2/3/4  walk up/down/left/right; walking into the block PUSHES it one
                   cell further when the cell behind it is free (edge and wall
                   block the push, and then nothing moves at all);
    ACTION5        TUG: the block is dragged one cell TOWARDS the player, along
                   whichever axis it is further away on (ties go vertical), if
                   that cell is free. The player never moves.

So the block is dragged with ACTION5 and shoved with a walk -- the two are
inverses, which is why almost nothing in this game is irreversible and why the
solver can recover from an arbitrary exploratory detour.

THE WIN IS A TRANSITION, NOT A STATE. The engine only advances the level at the
end of a *walk* (``step`` checks "block on goal" after ACTION1..4 and not after
ACTION5), so tugging the block onto the goal does not by itself finish the level:
one more directional press is needed. That press can be any of the four -- at most
one of them pushes the block back off the goal, and even a press into a wall
counts -- so a state with the block already on the goal is always exactly one move
from winning. The field encodes that faithfully: ``dist`` counts ACTIONS TO THE
WINNING PRESS, and the win test is applied to a *move*, so a push that lands the
block on the goal wins in one action rather than two.

STATE + ENCODING. A state is ``(player_cell, block_cell)`` -- at most 100*100 =
10k states on a 10x10 board -- encoded as ``player * n + block``. Every successor
is precomputed as a flat ``(5, n*n)`` int array with numpy (each action is a
handful of whole-array ops), and the exact distance-to-win field is one reverse
BFS over those arrays, so a whole field costs ~10 ms. The scalar queries below
read those same arrays, so there is exactly one implementation of the transition
relation, not a fast one and a readable one that can disagree.
"""

from __future__ import annotations

import numpy as np

# Game-space deltas for ACTION1..4, index-aligned with the action indices used
# throughout this module (and with tk01's own ACTION1..4 -> (dx, dy) mapping).
DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))
TUG = 4                      # ACTION5
NUM_ACTIONS = 5


class TugField:
    """Every reachable ``(player, block)`` state's exact distance to the win.

    ``walls`` and ``goal`` are ``(x, y)`` grid cells; the field depends ONLY on
    that geometry (never on where the player/block currently are), so one field
    answers every query for a level -- each replan, burst and RESET inside it.
    """

    def __init__(self, gw: int, gh: int, walls, goal) -> None:
        self.gw, self.gh = gw, gh
        self.n = n = gw * gh
        self.goal = goal[1] * gw + goal[0]

        wall = np.zeros(n, dtype=bool)
        for (x, y) in walls:
            wall[y * gw + x] = True
        self.wall = wall

        # ``nbr[d][c]`` = the cell one step in direction d from c, or -1 when that
        # is off the board. Walls are NOT folded in here: a wall blocks the player
        # and the block, but the *push target* check needs the raw neighbour.
        nbr = np.full((4, n), -1, dtype=np.int64)
        for d, (dx, dy) in enumerate(DELTAS):
            for y in range(gh):
                for x in range(gw):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < gw and 0 <= ny < gh:
                        nbr[d][y * gw + x] = ny * gw + nx
        self.nbr = nbr

        self.succ, self.win_move = self._transitions()
        self.dist = self._reverse_bfs()

    # ── the transition relation (one vectorised pass over ALL states) ────────
    def _transitions(self):
        """``succ[a][s]`` = the state action ``a`` leads to, and ``win_move[s]`` =
        whether SOME walk from ``s`` leaves the block on the goal (i.e. wins)."""
        n, gw, gh, nbr, wall = self.n, self.gw, self.gh, self.nbr, self.wall
        size = n * n
        s = np.arange(size, dtype=np.int64)
        p, b = s // n, s % n

        succ = np.empty((NUM_ACTIONS, size), dtype=np.int32)
        win_move = np.zeros(size, dtype=bool)

        for d in range(4):
            tgt = nbr[d][p]                       # cell the player steps into
            onboard = tgt >= 0
            t = np.where(onboard, tgt, 0)         # safe index for the masked ops
            # The block sits ABOVE the goal marker (layers), so a cell holding the
            # block is a block whether or not the goal is under it -> a push.
            push = onboard & (t == b)
            behind = np.where(push, nbr[d][t], -1)
            bh = np.where(behind >= 0, behind, 0)
            can_push = push & (behind >= 0) & ~wall[bh]
            walk = onboard & ~push & ~wall[t]
            new_p = np.where(can_push | walk, t, p)
            new_b = np.where(can_push, bh, b)
            succ[d] = new_p * n + new_b
            # The engine tests "block on goal" after EVERY walk, including one that
            # changed nothing (a bump into a wall), so a blocked press still wins
            # when the block already sits on the goal.
            win_move |= new_b == self.goal

        # TUG: the block is pulled one cell towards the player along the dominant
        # axis (ties -> vertical), blocked by the board edge, a wall, or the player.
        px, py = p % gw, p // gw
        bx, by = b % gw, b // gw
        adx, ady = px - bx, py - by
        dx, dy = np.sign(adx), np.sign(ady)
        diagonal = (dx != 0) & (dy != 0)
        horizontal = diagonal & (np.abs(adx) > np.abs(ady))
        dx = np.where(diagonal & ~horizontal, 0, dx)
        dy = np.where(horizontal, 0, dy)
        nx, ny = bx + dx, by + dy
        moving = (dx != 0) | (dy != 0)            # false only for player == block
        inside = moving & (nx >= 0) & (nx < gw) & (ny >= 0) & (ny < gh)
        cell = np.where(inside, ny * gw + nx, 0)
        ok = inside & (cell != p) & ~wall[cell]
        succ[TUG] = p * n + np.where(ok, cell, b)
        return succ, win_move

    # ── the one reverse BFS ─────────────────────────────────────────────────
    def _reverse_bfs(self) -> np.ndarray:
        """``dist[s]`` = actions to the winning press from ``s`` (-1 = can't win).

        Layer 1 is every state a single walk wins from; layer k+1 is every state
        with SOME action (walk or tug) into layer k. Whole layers are stepped at
        once: ``frontier[succ[a]]`` asks, for all states in parallel, whether
        action ``a`` lands in the current frontier.
        """
        size = self.n * self.n
        dist = np.full(size, -1, dtype=np.int16)
        dist[self.win_move] = 1
        frontier = self.win_move
        depth = 1
        while True:
            depth += 1
            reach = np.zeros(size, dtype=bool)
            for a in range(NUM_ACTIONS):
                reach |= frontier[self.succ[a]]
            fresh = reach & (dist < 0)
            if not fresh.any():
                return dist
            dist[fresh] = depth
            frontier = fresh

    # ── scalar queries (all read the arrays above -- no second implementation) ─
    def cell(self, x: int, y: int) -> int:
        return y * self.gw + x

    def xy(self, cell: int) -> tuple[int, int]:
        return cell % self.gw, cell // self.gw

    def steps_to_win(self, p: int, b: int) -> int | None:
        d = int(self.dist[p * self.n + b])
        return None if d < 0 else d

    def step(self, p: int, b: int, a: int) -> tuple[int, int]:
        """The successor state of action ``a`` (0..3 walk, 4 tug)."""
        return divmod(int(self.succ[a][p * self.n + b]), self.n)

    def wins(self, p: int, b: int, a: int) -> bool:
        """Does ``a`` end the level? Only a WALK does, and only if it leaves the
        block on the goal."""
        return a < 4 and self.step(p, b, a)[1] == self.goal

    def optimal_actions(self, p: int, b: int) -> list[int]:
        """Every equally-optimal action at ``(p, b)``, in ACTION1..5 order.

        At distance 1 that is exactly the winning presses (a walk that would shove
        the block back off the goal is NOT among them); deeper, it is every action
        that strictly descends the field."""
        here = self.steps_to_win(p, b)
        if not here:
            return []
        if here == 1:
            return [a for a in range(4) if self.wins(p, b, a)]
        out = []
        for a in range(NUM_ACTIONS):
            there = self.steps_to_win(*self.step(p, b, a))
            if there is not None and there == here - 1:
                out.append(a)
        return out

    # ── level sampling (used by the game's per-seed layout generator) ────────
    def start_states(self, lo: int, hi: int) -> np.ndarray:
        """Every legal STARTING state whose optimal solution is ``lo..hi`` actions.

        "Legal" means the player, the block and the goal are three distinct cells
        and neither actor stands in a wall -- so nothing is hidden under anything
        else at t=0 and the level does not start already solved.
        """
        n, g = self.n, self.goal
        idx = np.nonzero((self.dist >= lo) & (self.dist <= hi))[0]
        if idx.size == 0:
            return idx
        p, b = idx // n, idx % n
        keep = ((p != b) & (p != g) & (b != g)
                & ~self.wall[p] & ~self.wall[b])
        return idx[keep]

    def deepest(self) -> int:
        """The longest optimal solution over all states (0 if nothing wins)."""
        return int(self.dist.max())
