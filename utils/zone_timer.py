"""Exact distance-to-win fields for grid games whose obstacles are a function of TIME.

The family: an open grid, a player that steps one cell per action, a goal, and a set
of cells whose *blocking* is decided by a clock rather than by the player. Every
action advances that clock by exactly one, including an action that changes nothing
(a move into a blocking cell, or off the board), so **stalling is a first-class move**
and clock parity is a resource the player controls. Two games in this repo are
exactly that:

  * **zq01** ("Zone Timer") -- ONE hazard set toggling every ``period`` actions;
  * **zq02** ("Dual Phase Hazards") -- TWO sets on independent periods and offsets,
    plus static walls.

They differ only in *which cells block at clock phase t*. Everything else -- the
successor relation, the reverse BFS, the co-optimal tie set, the greedy descent -- is
one algorithm, and it lives here once (`PhasedGridField`) rather than being written
out per game. `ZoneTimerField` is the single-blinking-set convenience wrapper zq01
uses; zq02's solver builds its per-phase masks from its own cell-kind table and blink
schedule and hands them straight to `PhasedGridField`.

WHY A FIELD AND NOT A SEARCH. The state is exactly ``(player cell, clock phase)``,
which is a few thousand states for any board these games draw -- so the successor
table is built once and ONE reverse BFS from the goal answers every later question
(how far from here? which moves are co-optimal?) by array lookup. That is what makes
replanning from an arbitrary state free, which is what a training generator's
``solve_from`` needs to recover from an exploratory detour or a perturbation burst.

ONE implementation, TWO kinds of caller (the repo's shared-expert convention, cf.
``utils/tug_puzzle.py``): the GAMES' per-seed level samplers use it as their
solvability certificate, and the GENERATORS' experts descend it. Keeping both on the
same object is what guarantees every board a game can draw is one its generator can
solve optimally.

Conventions, matching both games' ``step``:

  * the clock ticks BEFORE the move is resolved, so a move is resolved against the
    phase it LANDS in -- ``succ`` at phase ``t`` reads ``blocked[t + 1]``;
  * phase 0 is a level's opening state (the engine seats a level at tick 0);
  * the win is a TRANSITION, not a state: the level ends the instant the player steps
    onto the goal, so the BFS is seeded at distance 1 on the winning MOVES and goal
    cells are never states to plan from.
"""
from __future__ import annotations

import numpy as np

#: Game-space deltas, index-aligned with the engine's ACTION1..ACTION4
#: (up / down / left / right) and with everything this module calls a "direction".
DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))


class PhasedGridField:
    """Exact distance-to-win field over ``(cell, phase)`` for a time-varying grid.

    ``blocked`` is the whole obstacle history: ``blocked[t][c]`` says whether cell
    ``c`` blocks a move that lands in it at phase ``t``, for ``t`` in ``[0, P)``.
    Static obstacles (walls) are simply cells that block at every phase. ``targets``
    is the set of goal cells -- entering any of them wins.

    Attributes a caller may read:
      ``dist[s]``    -- optimal actions-to-win from state ``s``, ``-1`` if never;
      ``succ[d, s]`` -- the state direction ``d`` leads to, ``-1`` where ``s`` is not
                        a state the player can be in (i.e. a goal cell);
      ``wins[d, s]`` -- whether that move ends the level.
    """

    def __init__(self, gw: int, gh: int, blocked, targets) -> None:
        self.gw, self.gh = int(gw), int(gh)
        self.ncell = self.gw * self.gh
        blocked = np.asarray(blocked, dtype=bool)
        if blocked.ndim != 2 or blocked.shape[1] != self.ncell:
            raise ValueError(
                f"blocked must be (P, {self.ncell}), got {blocked.shape}")
        self.nphase = int(blocked.shape[0])
        if self.nphase < 1:
            raise ValueError("blocked needs at least one phase")
        self.targets = frozenset(int(c) for c in targets)
        self.N = self.nphase * self.ncell
        self._build(blocked)

    # ── packing ─────────────────────────────────────────────────────────────
    def cell(self, x: int, y: int) -> int:
        return int(y) * self.gw + int(x)

    def xy(self, cell: int) -> tuple[int, int]:
        return int(cell) % self.gw, int(cell) // self.gw

    def state(self, cell: int, phase: int) -> int:
        return (int(phase) % self.nphase) * self.ncell + int(cell)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= int(x) < self.gw and 0 <= int(y) < self.gh

    # ── the state graph + the one reverse BFS ───────────────────────────────
    def _build(self, blocked: np.ndarray) -> None:
        gw, gh, ncell, P = self.gw, self.gh, self.ncell, self.nphase
        cells = np.arange(ncell, dtype=np.int64)
        xs, ys = cells % gw, cells // gw

        # nbr[d][c] = the cell one step in direction d from c, or -1 off-board.
        nbr = np.full((4, ncell), -1, dtype=np.int64)
        for d, (dx, dy) in enumerate(DELTAS):
            nx, ny = xs + dx, ys + dy
            inb = (nx >= 0) & (nx < gw) & (ny >= 0) & (ny < gh)
            nbr[d] = np.where(inb, ny * gw + nx, -1)

        tgt = np.zeros(ncell, dtype=bool)
        for c in self.targets:
            if 0 <= c < ncell:
                tgt[c] = True

        succ = np.empty((4, self.N), dtype=np.int64)
        wins = np.zeros((4, self.N), dtype=bool)
        for t in range(P):
            t2 = (t + 1) % P
            blk = blocked[t2]                  # the move meets the NEXT phase
            lo = t * ncell
            for d in range(4):
                dest = nbr[d]
                safe = np.maximum(dest, 0)
                stuck = (dest < 0) | blk[safe]
                nd = np.where(stuck, cells, dest)
                succ[d, lo:lo + ncell] = t2 * ncell + nd
                wins[d, lo:lo + ncell] = tgt[nd]
        # A goal cell is where the level ENDED; it is never a state to plan from.
        playable = ~np.tile(tgt, P)
        succ[:, ~playable] = -1
        wins[:, ~playable] = False
        self.succ, self.wins = succ, wins

        # Reverse BFS from the winning TRANSITIONS, one vectorised layer at a time.
        dist = np.full(self.N, -1, dtype=np.int32)
        won = np.zeros(self.N, dtype=bool)
        for d in range(4):
            won |= wins[d] & (succ[d] >= 0)
        dist[won] = 1
        step = 1
        while True:
            upd = np.zeros(self.N, dtype=bool)
            for d in range(4):
                sa = succ[d]
                ok = sa >= 0
                upd |= ok & (dist[np.where(ok, sa, 0)] == step)
            newly = upd & (dist < 0)
            if not newly.any():
                break
            dist[newly] = step + 1
            step += 1
        self.dist = dist

    # ── queries ─────────────────────────────────────────────────────────────
    def successor(self, cell: int, phase: int, d: int) -> tuple[int, int]:
        """The state direction ``d`` leads to, or ``None`` from a goal cell."""
        s = int(self.succ[d, self.state(cell, phase)])
        if s < 0:
            return None
        return s % self.ncell, s // self.ncell

    def steps_to_win(self, cell: int, phase: int) -> int | None:
        """Optimal actions-to-win from ``(cell, phase)``, or ``None`` when the goal
        cannot be reached from here -- which includes standing ON the goal, since
        that means the level is already over."""
        v = int(self.dist[self.state(cell, phase)])
        return None if v < 0 else v

    def optimal_dirs(self, cell: int, phase: int) -> list[int]:
        """Every direction that strictly descends the field -- the exact co-optimal
        tie set, in ACTION1..4 order. Often more than one move: with the phase in the
        state, stalling against the board edge is regularly as good as advancing."""
        s = self.state(cell, phase)
        here = int(self.dist[s])
        if here <= 0:                          # unreachable, or already won
            return []
        out = []
        for d in range(4):
            sa = int(self.succ[d, s])
            if sa < 0:
                continue
            if here == 1:
                if self.wins[d, s]:
                    out.append(d)
            elif not self.wins[d, s] and int(self.dist[sa]) == here - 1:
                out.append(d)
        return out

    def plan(self, cell: int, phase: int, rng=None) -> list[int]:
        """One optimal direction plan from ``(cell, phase)`` -- a greedy descent of
        the field, tie-broken with ``rng`` when given so repeated recordings of the
        same board take different (equally optimal) routes. ``[]`` if unreachable."""
        out: list[int] = []
        while True:
            s = self.state(cell, phase)
            dirs = self.optimal_dirs(cell, phase)
            if not dirs:
                return [] if not out else out
            d = rng.choice(dirs) if rng is not None else dirs[0]
            out.append(d)
            if self.wins[d, s]:
                return out
            cell, phase = self.successor(cell, phase, d)


class ZoneTimerField(PhasedGridField):
    """`PhasedGridField` for the ONE-blinking-set case (zq01).

    A single hazard set toggles every ``period`` actions starting from OFF, so the
    clock is ``phase in [0, 2 * period)``: one full off-run then one full on-run.
    """

    def __init__(self, gw: int, gh: int, period: int, hazard_cells, target) -> None:
        period = int(period)
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}")
        gw, gh = int(gw), int(gh)
        ncell = gw * gh
        self.period = period
        self.hazard = frozenset((int(x), int(y)) for x, y in hazard_cells)
        self.target = (int(target[0]), int(target[1]))

        on_mask = np.zeros(ncell, dtype=bool)
        for x, y in self.hazard:
            if 0 <= x < gw and 0 <= y < gh:
                on_mask[y * gw + x] = True
        blocked = np.zeros((2 * period, ncell), dtype=bool)
        blocked[period:] = on_mask             # phases [period, 2p) are the ON run
        super().__init__(gw, gh, blocked,
                         [self.target[1] * gw + self.target[0]])

    def hazards_on(self, phase: int) -> bool:
        """Are the hazards blocking at ``phase``?"""
        return ((int(phase) % self.nphase) // self.period) % 2 == 1
