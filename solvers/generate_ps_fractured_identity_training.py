"""Generate Phase-1 training data for the PuzzleScript game ps:fractured_identity
("Fractured Identity", Adam Conway -- the one where the other you moves when you
move, and burns the floor it stands on).

The harness -- the rotation contract, the trajectory recorder, the plan cache and
the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: a native model of the
ruleset, the searches over it, and the rendering fix.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_ps_fractured_identity",
      "levels": [
        {"level_id": 0,
         "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
         "actions":      [a_0, ..., a_{T-1}]},     # length T
        ...
      ]
    }

actions[0] is the RESET that produced obs[0]; actions[i] for i>=1 is the action
that took the agent from obs[i-1] to obs[i]. The recorded index is the *screen*
action (post rotation remap), so replaying the recorded actions reproduces the
recorded frames exactly. Every step carries the set of equally-optimal presses.

THE GAME
--------
Win = ``No Teleport``: every teleporter has to be walked over and absorbed. The
blue ones and the black one answer to Player1, the red ones to Player2.

SIX FACTS, and only the first two are visible in the art:

  * EVERY PLAYER OBJECT TAKES THE SAME PRESS. ``Player = Player1 or Player2`` is
    the input class, so one arrow key moves the hero, the doppelganger, and all
    thirty-one of the doppelganger's copies on level 9. (The game's first rule,
    ``[ > Player1 ][ Player2 ] -> [ > Player1 ][ > Player2 ]``, is redundant --
    the interpreter has already given every Player the force.) There is no way
    to move one and not the other, so a level is a puzzle about the RELATIVE
    geometry of two bodies in one maze.
  * PLAYER2 BURNS THE FLOOR IT IS STANDING ON, on any press it is given a force
    -- including a press that a wall then cancels. ``[ > Player2 Forevers] ->
    [ > Player2 death ]`` fires before the blocking rules, so a bump into a wall
    still costs the tile underfoot. Player2's route is therefore a SELF-AVOIDING
    WALK, and it is what makes the game finite: it can never retrace a step.
  * ...WHICH IS ALSO WHY PLAYERS CANNOT CONVOY. That burn happens before
    ``[ > Player | death] -> [Player | death]``, so anybody directly behind a
    moving Player2 is walking into a tile that just became lethal and is
    cancelled. Two Player2s in a column never both move down; a Player1 tucked
    in behind one never follows it. The exception is a Player2 standing on a
    tile with no floor to burn (a button, a key, a lock, a teleporter): the
    follower goes through fine. Level 9's ring of thirty-two is one long
    demonstration -- press UP and only the bottom row moves, because every
    other body is standing behind somebody who just scorched the ground.
  * THE GREEN BUTTON SWAPS THE TWO PLANES. It is three buttons stacked in one
    cell, and their three rules run in order: floor -> filler, death -> floor,
    filler -> death. Net, every walkable tile and every lethal tile trade
    places -- including the tile you are standing on, which is legal: death
    only blocks you from ENTERING it. That makes the intended shape of a level
    a there-and-back: walk out burning a trail, press the button at the far
    end, and walk home along the trail, which is now the only floor left.
    Walking it burns it again, so a second button press hands the whole board
    back untouched.
  * PICKUPS ARE ONE PRESS LATE. ``[Player1 Teleporter] -> [Player1]`` is a main
    rule, and main rules run BEFORE movement, so the teleporter you step onto
    this press is absorbed on the next one -- any press, a wall bump included.
    Every plan here ends with a throwaway. Keys are the opposite: their rule is
    ``late``, so a key is spent the instant you arrive on it.
  * A KEY OPENS THE FIRST LOCK ON THE BOARD, not the nearest one -- and "first"
    is set iteration order, not row-major. See `_Board._lock_to_clear`. Only
    Player1 can spend a key, which makes level 3's key and its two locks pure
    scenery: that level has no Player1 at all.

THE MODEL
---------
`_Board` re-implements those rules as bit-parallel operations on cell masks (see
`_Board.step`): one press is a handful of shifts, so the force chains resolve as
a mask fixpoint rather than a per-body loop -- which is what makes level 9's
thirty-two bodies cost the same as level 0's one. It exists because the
interpreter is far too slow to search: 300 steps/s on level 9 and 1.2k on level
8, against the ~25k/s a plain Sokoban in this tree runs at. ``--fuzz`` is the
proof that it is faithful (random boards played press-for-press against the real
interpreter, comparing every mask after every press, plus a replay of each
shipped level's own solution) and ``--verify`` replays each plan on the
interpreter and requires a genuine WIN, so nothing here rests on the model being
right by inspection.

THE SEARCH
----------
The state is seven integers: Player1's cell, the Player2 occupancy mask, the
death and filler masks, and the surviving key / lock / teleporter masks. The
Player2s are interchangeable, so a mask (not a tuple of positions) is the
canonical state -- and it makes level 9's ring collapse from 32 coordinates to
one number.

A* at weight 1 comes first, on a route bound -- the shortest walls-only tour that
still collects every key and every teleporter -- because a PROOF of shortest is
what buys the labels: every press costs 1, so a BFS bounded by that length hands
back the exact optimal-action SETS with one backward sweep and no per-step
re-solve (`_Board.bfs`). Nine of the ten winnable levels come out that way, 15 to
47 presses. Level 9 does not, and takes a weighted rung instead.

What actually makes the searches terminate is not the bound but `_Board.reach`:
burning the floor is irreversible until a button flips the world, so most of what
a wander produces is a body sealed into a scorched pocket, and a state with a
perfectly ordinary heuristic that can never win will otherwise sit in the queue
generating more of the same. Getting that relaxation right is most of the work
here -- it has to allow for the fact that burning MORE floor is how the far side
of the board opens up, and for the fact that the body that needs to travel is
often not the body that presses the button.

LEVEL 8 IS NOT WINNABLE, and it is this interpreter's arbitrary choice that makes
it so, not the level design. See `FracturedIdentitySolver.skip_levels`.

Related: solvers/common/ps_astar.py, solvers/generate_ps_flood_training.py.
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame            # noqa: E402
from solvers.common.ps_astar import (                              # noqa: E402
    Plan, PSAStarSolver, PSExpert,
)

GAME_NAME = "Fractured_Identity"

#: The five presses. Every one of them costs a turn, and every one of them
#: triggers the teleporter-absorption rules, so even ACTION on a level with no
#: button is a legal (and sometimes necessary) move.
PRESSES = ("up", "down", "left", "right", "action")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}

#: Objects `_Board` needs to find. Everything else in the game file (Target,
#: Crate, the Background) is either unused by every shipped level or invariant.
_OBJECTS = ("wall", "blackened", "forevers", "death", "filler", "player1",
            "player2", "teleporter", "teleporter2", "teleporter3", "key",
            "lock", "offbutton", "obutton", "onbutton")

_INF = 1 << 20


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class _Board:
    """Static geometry of one board, plus `step`: the whole ruleset on masks.

    Every dynamic quantity is a bitmask over CELLS (``bit r * w + c``), which is
    what lets a press be a dozen integer operations instead of a loop over
    bodies. A *state* is ``(p1, p2, dead, filler, keys, locks, teles)``:

      ``p1``      Player1's cell index, or -1 on the two levels that have no
                  Player1 (3 and 4). Not a mask: no level ships two of him, and
                  keeping him a scalar makes the key rule a single test.
      ``p2``      occupancy mask of the Player2s. They are interchangeable --
                  same object, same rules, no identity anywhere in the game --
                  so the mask is a canonical state and not a lossy summary.
      ``dead`` / ``filler``
                  the tiles currently holding a ``death`` / ``filler`` object.
                  ``filler`` is 0 on every shipped level (all eleven stack the
                  three buttons in one cell, so the three swap rules always fire
                  together and filler is created and consumed inside one press)
                  -- carried anyway so ``--fuzz`` can drive the buttons apart
                  and still be an honest comparison.
      ``keys`` / ``locks`` / ``teles``
                  the surviving keys / locks / teleporters. These three classes
                  are only ever removed, never created or moved, so their cell
                  sets can be read once off the start board -- and a teleporter
                  needs only one bit per cell because all three teleporter
                  objects share a collision layer and cannot stack.

    ``swap`` -- every tile that holds floor, death or filler -- is closed under
    the rules for the same reason: the burn rule needs a ``Forevers`` to consume
    and the button rules only ever cycle a tile inside the set, so no press can
    enrol a new cell.
    """

    __slots__ = ("h", "w", "n", "ids", "wall", "black", "swap", "off", "ob",
                 "on", "t1", "t2", "keymask", "lockmask", "telemask", "edge",
                 "colkeep", "_lockpick", "_free_dist", "_route")

    def __init__(self, h, w, ids, walls, blacks, swap, off, ob, on,
                 t1, t2, keys, locks):
        self.h, self.w = h, w
        self.n = h * w
        self.ids = ids
        self.wall, self.black, self.swap = walls, blacks, swap
        self.off, self.ob, self.on = off, ob, on
        self.t1, self.t2 = t1, t2
        self.keymask, self.lockmask = keys, locks
        self.telemask = t1 | t2

        # Shift helpers. `edge[d]` is the row/column that has no neighbour in
        # direction d (so a body there simply cannot move), and `colkeep[d]`
        # is what a horizontal shift has to mask off first or it wraps a row
        # into the next one.
        full = (1 << self.n) - 1
        first_col = sum(1 << (r * w) for r in range(h))
        last_col = sum(1 << (r * w + w - 1) for r in range(h))
        first_row = (1 << w) - 1
        last_row = first_row << ((h - 1) * w)
        self.edge = {"up": first_row, "down": last_row,
                     "left": first_col, "right": last_col}
        self.colkeep = {"left": full & ~first_col, "right": full & ~last_col,
                        "up": full, "down": full}

        self._lockpick = {}
        self._free_dist = {}
        self._route = {}

    # -- masks ----------------------------------------------------------------

    def shift(self, mask, d):
        """``mask`` moved one cell in direction ``d``; cells that would leave
        the grid are dropped."""
        if d == "right":
            return (mask & self.colkeep["right"]) << 1
        if d == "left":
            return (mask & self.colkeep["left"]) >> 1
        if d == "down":
            return (mask << self.w) & ((1 << self.n) - 1)
        return mask >> self.w

    # -- reading the interpreter ----------------------------------------------

    def read_state(self, eng):
        """The dynamic half of the state, off a live engine grid.

        Takes the cell sets as already fixed (see the class docstring), so it
        stays comparable with a model state after objects have been consumed --
        which is the whole point during `_fuzz`."""
        i = self.ids
        P1, P2, DE, FI = i["player1"], i["player2"], i["death"], i["filler"]
        KE, LO = i["key"], i["lock"]
        TE = (i["teleporter"], i["teleporter2"], i["teleporter3"])
        p1, p2, dead, filler, keys, locks, teles = -1, 0, 0, 0, 0, 0, 0
        w = self.w
        for r in range(self.h):
            row = eng.grid[r]
            for c in range(w):
                cell = row[c]
                if not cell:
                    continue
                bit = 1 << (r * w + c)
                if P1 in cell:
                    p1 = r * w + c
                if P2 in cell:
                    p2 |= bit
                if DE in cell:
                    dead |= bit
                if FI in cell:
                    filler |= bit
                if KE in cell:
                    keys |= bit
                if LO in cell:
                    locks |= bit
                if not cell.isdisjoint(TE):
                    teles |= bit
        return (p1, p2, dead, filler, keys, locks, teles)

    # -- the ruleset ----------------------------------------------------------

    def step(self, st, press):
        """One press. Returns ``(state, won)``.

        The order is the interpreter's, and every line of it is load-bearing:
        the burn happens before the blocking rules (so a cancelled press still
        costs the tile underfoot, and so a body directly behind a Player2 is
        cancelled by the tile it just scorched), the pickups happen before
        movement (so they see where you were, not where you are going), and the
        key rule is ``late`` so it happens after."""
        p1, p2, dead, filler, keys, locks, teles = st
        p1bit = (1 << p1) if p1 >= 0 else 0

        if press == "action":
            # R7-R9, the three button rules in file order. Together they swap
            # the two planes; apart (only `_fuzz` gets there) they are a
            # three-stage cycle through `filler`.
            on_cell = p2 | p1bit
            if on_cell & self.off:
                filler |= self.swap & ~dead & ~filler
            if on_cell & self.ob:
                dead = 0
            if on_cell & self.on:
                dead |= filler
                filler = 0
            teles &= ~((p1bit & self.t1) | (p2 & self.t2))
            return (p1, p2, dead, filler, keys, locks, teles), teles == 0

        back = _OPPOSITE[press]
        bodies = p2 | p1bit

        # R3: every Player2 with a force burns the floor it stands on.
        dead |= p2 & self.swap & ~filler

        # R4-R6: the cancels, reading the board R3 just changed. `shift(S,
        # back)` is "the cells whose target-in-direction is in S".
        stuck = bodies & self.shift(dead | locks, back)
        stuck |= p2 & self.shift(self.black, back)
        forced = bodies & ~stuck

        # R10-R12: absorption, at the PRE-movement positions.
        teles &= ~((p1bit & self.t1) | (p2 & self.t2))

        # Force resolution. Everything still moving is heading the same way, so
        # there are no counter-chains and no two-chains-one-cell conflicts: a
        # run of adjacent bodies moves exactly when its leader can. Blocking
        # propagates backwards from the walls, the board edge and the bodies
        # whose force was just cancelled.
        blocked = forced & (self.edge[press]
                            | self.shift(self.wall | stuck, back))
        while True:
            grown = blocked | (forced & self.shift(blocked, back))
            if grown == blocked:
                break
            blocked = grown
        movers = forced & ~blocked

        p2 = self.shift(p2 & movers, press) | (p2 & ~movers)
        if p1bit & movers:
            p1bit = self.shift(p1bit, press)
            p1 = p1bit.bit_length() - 1

        # R13 (late): Player1 standing on a key spends it on one lock.
        if locks and (p1bit & keys):
            keys &= ~p1bit
            locks &= ~self._lock_to_clear(locks)

        return (p1, p2, dead, filler, keys, locks, teles), teles == 0

    def _lock_to_clear(self, locks):
        """Which lock a key destroys, as a cell mask, given the locks standing.

        NOT the nearest one, and not row-major either. The interpreter builds
        the rule's candidate list with `PSEngine._candidate_positions`, which
        returns a freshly built ``set`` of ``(row, col)`` tuples, and then takes
        the first match it iterates -- so the answer is whichever surviving lock
        lands first in a CPython set of that size. Tuples of small ints hash
        deterministically (no PYTHONHASHSEED involvement), so rebuilding the
        same set here reproduces the same choice. ``--fuzz`` is what says so,
        and it disagreed on nearly every board that had two locks until this
        replaced "row-major first"."""
        pick = self._lockpick.get(locks)
        if pick is None:
            w, rest = self.w, locks
            live = set()
            while rest:
                bit = rest & -rest
                cell = bit.bit_length() - 1
                live.add((cell // w, cell % w))
                rest ^= bit
            r, c = next(iter(live))
            pick = 1 << (r * w + c)
            self._lockpick[locks] = pick
        return pick

    # -- bounds ---------------------------------------------------------------

    def free_dist(self, target):
        """BFS distance from every cell to ``target`` through walls only.

        A lower bound on presses: nothing but a wall is permanent (death flips
        back at the button, locks open, bodies move), and one press moves a body
        at most one cell."""
        d = self._free_dist.get(target)
        if d is not None:
            return d
        d = [_INF] * self.n
        d[target] = 0
        q = deque([target])
        w, h = self.w, self.h
        while q:
            cur = q.popleft()
            r0, c0 = divmod(cur, w)
            for dr, dc in _DELTA.values():
                r, c = r0 + dr, c0 + dc
                if 0 <= r < h and 0 <= c < w:
                    nx = r * w + c
                    if not (self.wall >> nx) & 1 and d[nx] == _INF:
                        d[nx] = d[cur] + 1
                        q.append(nx)
        self._free_dist[target] = d
        return d

    def _cells(self, mask):
        out = []
        while mask:
            bit = mask & -mask
            out.append(bit.bit_length() - 1)
            mask ^= bit
        return out

    def _route_cost(self, start, targets):
        """Shortest walls-only tour from ``start`` visiting every target.

        Held-Karp over the (at most six) surviving keys and Player1 teleporters
        -- exact for the relaxed metric, and the relaxation only ever removes
        obstacles, so it is a lower bound on presses."""
        key = (start, targets)
        got = self._route.get(key)
        if got is not None:
            return got
        tg = self._cells(targets)
        if not tg:
            return 0
        k = len(tg)
        d0 = [self.free_dist(t)[start] for t in tg]
        dd = [[self.free_dist(a)[b] for b in tg] for a in tg]
        best = {(1 << i, i): d0[i] for i in range(k)}
        for _size in range(1, k):
            nxt = {}
            for (mask, last), cost in best.items():
                for j in range(k):
                    if mask & (1 << j):
                        continue
                    nm, nc = mask | (1 << j), cost + dd[last][j]
                    if nxt.get((nm, j), _INF) > nc:
                        nxt[(nm, j)] = nc
            if not nxt:
                break
            best.update(nxt)
        full = (1 << k) - 1
        out = min((c for (m, _l), c in best.items() if m == full), default=_INF)
        self._route[key] = out
        return out

    def heuristic(self, st):
        """Presses that must still happen, as a walls-only route bound.

        Player1 is a single body that has to collect every surviving key (the
        teleporter is always walled in behind the locks they open) and then
        every surviving blue/black teleporter, so his term is a TOUR over the
        lot. The red ones are the same tour when there is only one Player2 --
        levels 3 and 4, where the doppelganger is the whole puzzle -- and only
        a farthest-from-any-of-them distance when several could share them out.
        One further press is added for the absorption, which is a rule pass and
        therefore always happens on the press AFTER arriving."""
        p1, p2, _dead, _fil, keys, _locks, teles = st
        if not teles:
            return 0
        worst = 0
        t1 = teles & self.t1
        if t1:
            if p1 < 0:
                return _INF
            worst = self._route_cost(p1, t1 | keys)
        t2 = teles & self.t2
        if t2:
            bodies = self._cells(p2)
            if len(bodies) == 1:
                worst = max(worst, self._route_cost(bodies[0], t2))
            else:
                for cell in self._cells(t2):
                    d = self.free_dist(cell)
                    best = min((d[p] for p in bodies), default=_INF)
                    worst = max(worst, best)
        return worst + 1

    def unreachable(self, st):
        """True when a surviving teleporter can provably never be reached.

        This is the test that makes the searches terminate at all. Burning the
        floor is irreversible until a button flips the world, so most of what a
        wander produces is a body sealed into a scorched pocket with no button
        in it -- a state with a perfectly ordinary-looking heuristic that can
        never win, and, left in the queue, one that keeps generating more of
        the same."""
        _p1, _p2, _dead, _fil, _keys, _locks, teles = st
        if not teles:
            return False
        r1, r2 = self.reach(st)
        return bool((teles & self.t1 & ~r1) or (teles & self.t2 & ~r2))

    def reach(self, st):
        """``(everywhere Player1 could ever stand, everywhere a Player2 could)``
        in a relaxation that only ever ADDS moves -- so a teleporter outside the
        matching set is provably lost.

        Think in terms of each tile's death BIT and a global parity: a tile is
        lethal when ``bit xor parity`` is 1, and Player2 walking on a tile sets
        its bit to 1. The two parities are therefore NOT symmetric, and that
        asymmetry is the whole argument:

          * At parity 0 the walkable tiles are the ``bit == 0`` ones, and
            Player2 walking CONSUMES them. Parity 0 can only ever shrink.
          * At parity 1 the walkable tiles are the ``bit == 1`` ones, and
            walking leaves them alone. Parity 1 is stable -- and it GROWS every
            time Player2 burns something at parity 0.

        Which is why a single flood is wrong twice over, and both mistakes cost
        real levels here:

          * It treats the death set as fixed, and so declares a body that has
            burnt its own bridge to be trapped -- when burning MORE is exactly
            how the far side opens up. Hence the fixpoint below.
          * It floods one body at a time, and so asks whether the body that
            needs to TRAVEL can reach a button. Nobody has to: the flip is
            global and any player can press it. Levels 1 and 5 are built on
            exactly that -- Player1 starts walled in by death and never moves
            until the doppelganger, half a board away, hits the button.

        So the state is four sets (each body class at each parity), a flip is
        available whenever ANY of them holds a button, and what parity 1 has to
        walk on is ``dead`` plus whatever PLAYER2 (not Player1 -- he leaves no
        trail) can reach at parity 0."""
        p1, p2, dead, _fil, _keys, _locks, _teles = st
        full = (1 << self.n) - 1
        ok1 = full & ~self.wall
        ok2 = ok1 & ~self.black
        buttons = self.off | self.ob | self.on
        alive = full & ~(self.swap & dead)
        a1 = self._flood((1 << p1) if p1 >= 0 else 0, alive & ok1)
        a2 = self._flood(p2, alive & ok2)
        b1 = b2 = 0
        while True:
            flipped = full & ~(self.swap & ~dead & ~a2)
            n1, n2 = b1, b2
            if (a1 | a2) & buttons:
                n1 = self._flood(a1 | b1, flipped & ok1)
                n2 = self._flood(a2 | b2, flipped & ok2)
            m1, m2 = a1, a2
            if (n1 | n2) & buttons:
                m1 = self._flood(a1 | n1, alive & ok1)
                m2 = self._flood(a2 | n2, alive & ok2)
            if (m1, m2, n1, n2) == (a1, a2, b1, b2):
                return a1 | b1, a2 | b2
            a1, a2, b1, b2 = m1, m2, n1, n2

    def _flood(self, seed, passable):
        """Cells reachable from ``seed`` through ``passable``. The seed itself
        is kept whether or not it is passable -- death stops a body ENTERING a
        tile, never standing on one, and standing on scorched ground is the
        normal case here."""
        reach = seed
        frontier = seed
        while frontier:
            grown = 0
            for d in _DELTA:
                grown |= self.shift(frontier, d)
            grown &= passable & ~reach
            reach |= grown
            frontier = grown
        return reach

    # -- searches -------------------------------------------------------------

    def bfs(self, start, d_star, cap):
        """Layered BFS bounded by a known optimal length, returning
        ``(plan, optsets)`` or ``(None, None)``.

        Called after `astar` at weight 1 has proved ``d_star`` shortest, and
        only to LABEL: every press costs 1, so the depth layers hand back the
        exact optimal-action SETS with one backward sweep and no per-step
        re-solve. A successor ``j`` of a state at depth ``k`` can only be on a
        shortest path when ``depth[j] == k + 1`` -- a cross edge to a shallower
        ``j`` is already too long to finish in the presses remaining -- so "can
        still win in exactly the presses left" propagates backwards through the
        forward edges alone, with no reverse adjacency. Flood does the same
        thing; see `solvers/generate_ps_flood_training.py`.

        The bound is what makes it affordable where a plain BFS to depth 47 is
        not: ``h`` is admissible, so any state with ``depth + h > d_star`` is
        provably off every shortest path and is dropped un-expanded."""
        depth = {start: 0}
        layers = [[start]]
        succ = {}
        for k in range(1, d_star + 1):
            nxt = []
            for st in layers[-1]:
                row = []
                for press in PRESSES:
                    ns, _won = self.step(st, press)
                    row.append(ns)
                    if ns in depth:
                        continue
                    if k + self.heuristic(ns) > d_star:
                        continue
                    depth[ns] = k
                    nxt.append(ns)
                succ[st] = row
            if len(depth) > cap:
                return None, None
            layers.append(nxt)
        if not any(s[6] == 0 for s in layers[d_star]):
            return None, None

        # Backward sweep: `ontrack` = "wins in exactly d* - depth presses".
        ontrack = {s for s in layers[d_star] if s[6] == 0}
        for k in range(d_star - 1, -1, -1):
            live = set()
            for st in layers[k]:
                for ns in succ[st]:
                    if ns in ontrack and depth[ns] == k + 1:
                        live.add(st)
                        break
            ontrack |= live
            layers[k] = live

        plan, optsets, st = [], [], start
        for k in range(d_star):
            row = succ[st]
            best = [i for i in range(len(PRESSES))
                    if row[i] in ontrack and depth[row[i]] == k + 1]
            optsets.append([PRESSES[i] for i in best])
            plan.append(PRESSES[best[0]])
            st = row[best[0]]
        return plan, optsets

    def astar(self, start, cap, weight=1):
        """(Weighted) A* to a win.

        At ``weight == 1`` the plan is provably shortest: `heuristic` is
        admissible and `unreachable` only ever drops states that cannot win at
        all. Returning the win the moment it is GENERATED is sound here rather
        than the usual mistake, because every press costs 1 and `heuristic` is
        at least 1 at any non-winning state -- so the node just popped had
        ``f <= g + 1``, and nothing left in the queue can beat that."""
        h_of = self.heuristic
        pq = [(weight * h_of(start), 0, 0, start)]
        best = {start: 0}
        # Parent pointers, not a path per queue entry: plans here run to 47
        # presses and the weight-1 rung on level 9 queues a million nodes
        # before giving up, so carrying the path in the node roughly doubles
        # the peak.
        parent = {start: None}
        counter = generated = 0
        while pq:
            _f, g, _c, st = heapq.heappop(pq)
            if best.get(st, 1 << 30) < g:
                continue
            for press in PRESSES:
                ns, won = self.step(st, press)
                if won:
                    out = [press]
                    while parent[st] is not None:
                        st, back = parent[st]
                        out.append(back)
                    out.reverse()
                    return out
                if best.get(ns, 1 << 30) <= g + 1:
                    continue
                h = h_of(ns)
                if h >= _INF or self.unreachable(ns):
                    continue
                best[ns] = g + 1
                parent[ns] = (st, press)
                counter += 1
                generated += 1
                heapq.heappush(pq, (g + 1 + weight * h, g + 1, counter, ns))
            if generated > cap:
                return None
        return None


def read_board(eng, g):
    """``(board, state)`` for the engine's current grid."""
    ids = {name: g.obj_name_to_idx[name] for name in _OBJECTS}
    h, w = eng.height, eng.width
    got = dict.fromkeys(("wall", "black", "swap", "off", "ob", "on",
                         "t1", "t2", "key", "lock"), 0)
    for r in range(h):
        for c in range(w):
            cell = eng.grid[r][c]
            if not cell:
                continue
            bit = 1 << (r * w + c)
            if ids["wall"] in cell:
                got["wall"] |= bit
            if ids["blackened"] in cell:
                got["black"] |= bit
            if (ids["forevers"] in cell or ids["death"] in cell
                    or ids["filler"] in cell):
                got["swap"] |= bit
            if ids["offbutton"] in cell:
                got["off"] |= bit
            if ids["obutton"] in cell:
                got["ob"] |= bit
            if ids["onbutton"] in cell:
                got["on"] |= bit
            if ids["teleporter"] in cell or ids["teleporter3"] in cell:
                got["t1"] |= bit
            if ids["teleporter2"] in cell:
                got["t2"] |= bit
            if ids["key"] in cell:
                got["key"] |= bit
            if ids["lock"] in cell:
                got["lock"] |= bit
    board = _Board(h, w, ids, got["wall"], got["black"], got["swap"],
                   got["off"], got["ob"], got["on"], got["t1"], got["t2"],
                   got["key"], got["lock"])
    return board, board.read_state(eng)


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class FracturedExpert(PSExpert):
    """Plans on `_Board`; the interpreter only ever certifies the result.

    `plan_cache_path` keeps every level's start plan on disk, so a shard of
    `parallelize_generator` does not re-derive the searches (see
    `PSExpert.plan`)."""

    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "fractured_identity_plans.json")

    #: Distinct states the LABELLING pass may hold. Sized as MEMORY, not
    #: patience: every state carries five open-ended Python ints, and the
    #: layers keep them all alive at once. A level that overruns it keeps its
    #: (still optimal) plan and loses only its tie sets.
    bfs_cap = 2_000_000
    #: Generated nodes per A*. This is the memory knob: level 9's weight-1 rung
    #: cannot win and simply runs until it hits this, at roughly 0.5 GB per
    #: million nodes, so it buys nothing to make it large.
    astar_cap = 1_000_000
    #: Weights tried after the optimal pass fails, in order, so a level that CAN
    #: be proved shortest still is. Level 9 is the only one that ever gets here
    #: -- and then weight 2 answers in a second, which is the usual shape: the
    #: expensive rung is the one that fails, not the one that succeeds.
    weights = (2, 3, 4, 6)

    def heuristic(self, eng) -> int:                      # pragma: no cover
        raise NotImplementedError("the model owns the heuristic; see _Board")

    def _search(self, eng) -> list | None:
        """A* at weight 1 first, because a proof of shortest is what buys the
        exact tie sets: `_Board.bfs` can then be BOUNDED by that length and
        label every step. Everything below it trades optimality away and
        labels each step with itself."""
        board, state = read_board(eng, self.g)
        plan = board.astar(state, self.astar_cap, 1)
        if plan is not None:
            labelled, optsets = board.bfs(state, len(plan), self.bfs_cap)
            if labelled is not None:
                return Plan(labelled, optsets)
            return Plan(plan, [[p] for p in plan])
        for weight in self.weights:
            plan = board.astar(state, self.astar_cap, weight)
            if plan is not None:
                return Plan(plan, [[p] for p in plan])
        return None


class FracturedIdentitySolver(PSAStarSolver):
    game_id = "puzzlescript_ps_fractured_identity"
    game_name = GAME_NAME
    expert_cls = FracturedExpert

    #: Record against the game folder's adapter -- the one `game_envs` hands a
    #: live agent -- so anything that folder ever does to the game (a step
    #: limit, a sprite fix) cannot silently diverge from what is taped here.
    game_module_id = "ps:fractured_identity"

    #: LEVEL 8 CANNOT BE WON, and it is the interpreter's arbitrary choice that
    #: makes it so. Its teleporter sits in a pocket sealed by two locks -- (7,6)
    #: from the board, (8,8) from the teleporter -- and the pocket contains one
    #: of the two keys. So Player1 can only ever spend the OUTSIDE key, and
    #: `_Board._lock_to_clear` (which mirrors the interpreter exactly, and is
    #: confirmed against it on this very board) spends it on (8,8): the far
    #: lock, behind the pocket, leaving (7,6) shut and the second key
    #: unreachable forever. Row-major order -- which is what the reference
    #: PuzzleScript engine would most likely pick -- would open (7,6) first and
    #: the level would solve. Skipped rather than "fixed" here, because the
    #: order comes out of `PSEngine._candidate_positions` and changing it would
    #: reach every other ps: game in this tree.
    skip_levels = frozenset({8})

    #: The longest plan is level 4's 47 presses, so the adapter's stock 200-step
    #: per-level budget has room for 3x the expert's route and needs no table;
    #: `--plans` prints the budget column that says so, and
    #: `games/ps:fractured_identity` records the check.
    max_steps = 150


# ---------------------------------------------------------------------------
# Reports and checks
# ---------------------------------------------------------------------------

def _levels(solver=None):
    solver = solver or FracturedIdentitySolver()
    game = solver.make_game(0)
    return game, FracturedExpert(game)


def _report() -> int:
    game, expert = _levels()
    eng = game._engine
    total = solved = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, st = read_board(eng, game._game)
        if level in FracturedIdentitySolver.skip_levels:
            print(f"level {level:2d}: SKIPPED", flush=True)
            continue
        found = expert.plan(eng, level)
        if found is None:
            print(f"level {level:2d}: NO PLAN", flush=True)
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(found)
        solved += 1
        game.set_level(level)
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"level {level:2d}: {eng.height:2d}x{eng.width:2d} "
              f"{bin(st[1]).count('1'):2d} player2 "
              f"{bin(st[6]).count('1')} teleporters  "
              f"{len(found):3d} presses ({room}), {ties:3d} tie steps "
              f"({ties / max(1, len(found)):.0%})", flush=True)
    print(f"total {total} presses over {solved} solved levels "
          f"({game.n_levels} shipped)")
    return 0


def _verify() -> int:
    """Replay every plan on the REAL interpreter and require a WIN."""
    game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        if level in FracturedIdentitySolver.skip_levels:
            continue
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level:2d}: NO PLAN")
            continue
        game.set_level(level)
        for d in plan:
            eng.step(d)
        ok = eng.check_win()
        bad += not ok
        print(f"level {level:2d}: {len(plan):3d} presses -> "
              f"{'WIN' if ok else 'NOT WON'}")
    print("verify clean" if not bad else f"VERIFY FAILED on {bad} levels")
    return 0 if not bad else 1


def _ties(levels=None) -> int:
    """Re-derive every labelled step with an independent BFS and compare.

    At each step of the plan, press everything, re-solve from each successor for
    its true distance to a win, and require that the presses reaching one in the
    moves remaining are exactly the labelled set -- so both an unlabelled press
    that ties and a labelled press that does not are failures. The re-solve is
    an independent A* at weight 1, not the layered sweep the labels came from.

    Skips the levels in `_UNLABELLED_LEVELS`, whose plans make no tie claim."""
    game, expert = _levels()
    eng = game._engine
    bad = tested = 0
    for level in range(game.n_levels):
        if ((levels and level not in levels)
                or level in _UNLABELLED_LEVELS
                or level in FracturedIdentitySolver.skip_levels):
            continue
        game.set_level(level)
        board, state = read_board(eng, game._game)
        plan = expert.plan(eng, level)
        if plan is None:
            continue
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        for i, press in enumerate(plan):
            remaining = len(plan) - i
            measured = []
            for alt in PRESSES:
                ns, won = board.step(state, alt)
                tested += 1
                if won:
                    if remaining == 1:
                        measured.append(alt)
                    continue
                sub = board.astar(ns, expert.astar_cap, 1)
                if sub is not None and 1 + len(sub) == remaining:
                    measured.append(alt)
            if sorted(measured) != sorted(sets[i]):
                bad += 1
                print(f"level {level:2d} step {i:3d}: labelled {sets[i]} "
                      f"but measured {measured}")
            state, _won = board.step(state, press)
        print(f"level {level:2d}: {len(plan):3d} steps checked", flush=True)
    print(f"ties: {tested} alternatives re-solved, {bad} disagreements")
    return 0 if not bad else 1


#: Levels whose plan is NOT proved shortest, so its per-step labels are
#: singletons by construction and claim nothing. Only level 9 is here: its
#: weight-1 A* cannot finish inside the node cap, so the plan comes off the
#: weight-2 rung and the layered labelling pass -- which needs a proved length
#: to bound itself -- is never run. Only `--ties` uses this.
_UNLABELLED_LEVELS = frozenset({9})


def _audit() -> int:
    """Assert every cell COMPOSITION renders distinctly, at every cell size.

    Composition, not object: what a body is standing ON is most of the state
    here -- the green button under Player2 is the difference between a press
    that flips the world and a press that does nothing -- and the shipped
    sprites are opaque 5x5 blocks that hide all of it. See the module docstring
    of `games/ps:fractured_identity` for the sprite fix this checks."""
    game = FracturedIdentitySolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    BUTTON = ("offbutton", "obutton", "onbutton")
    floors = {"bg": (), "floor": ("forevers",), "death": ("death",),
              "black": ("blackened",), "button": BUTTON,
              "tel1": ("teleporter",), "tel2": ("teleporter2",),
              "tel3": ("teleporter3",), "key": ("key",), "lock": ("lock",)}
    comps = {}
    for fname, fobjs in floors.items():
        comps[fname] = fobjs
        for who in ("player1", "player2"):
            comps[f"{who}_on_{fname}"] = fobjs + (who,)
    # A teleporter standing on lethal ground is a teleporter you cannot enter,
    # so those four have to differ from the same teleporter on live floor.
    for tel in ("teleporter", "teleporter2", "teleporter3"):
        comps[f"{tel}_on_floor"] = ("forevers", tel)
        comps[f"{tel}_on_death"] = ("death", tel)
    comps["wall"] = ("wall",)

    sizes = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    def shoot(h, w, objs):
        # Fill the WHOLE board with the composition and compare whole frames:
        # `_render_frame` upscales any sub-64 render to fill the frame, so
        # cropping one cell out by arithmetic lands in the wrong place and every
        # composition reads as identical. Rendering is per-cell independent, so
        # a whole-frame comparison is the same test done where the geometry
        # cannot drift.
        eng.height, eng.width = h, w
        eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                     for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        return np.asarray(_render_frame(eng, g))

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {name: shoot(h, w, objs) for name, objs in comps.items()}
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad else f"AUDIT FAILED: {bad} problems")
    return 0 if not bad else 1


def _fuzz(n_boards: int = 1500, n_steps: int = 25, seed: int = 0) -> int:
    """Differential fuzz: random boards played press-for-press against the real
    interpreter, comparing the whole state after every press.

    Random boards alone are not enough -- they never build the long convoys and
    plane flips a solution does -- so every shipped level's own plan is replayed
    first, and then the random phase is built to hit what those plans do not:
    the three buttons driven apart so `filler` survives a press, keys spent with
    several locks standing, and bodies packed tight enough that the force chains
    have to resolve. The per-rule COVERAGE counters below are there so a clean
    run cannot quietly mean "never exercised"."""
    solver = FracturedIdentitySolver()
    game = solver.make_game(0)
    eng, g = game._engine, game._game
    ids = {name: g.obj_name_to_idx[name] for name in _OBJECTS}
    rng = random.Random(seed)
    cover = dict.fromkeys(("burn", "cancel_death", "cancel_lock",
                           "cancel_black", "convoy", "swap", "filler_alive",
                           "pickup", "keyspend", "blocked"), 0)
    mismatches = steps = 0

    def compare(board, state, press):
        nonlocal mismatches, steps
        eng.step(press)
        truth_won = eng.check_win()
        truth = board.read_state(eng)
        nxt, won = board.step(state, press)
        steps += 1
        # Coverage, read off the transition the model just produced.
        if press != "action":
            if nxt[2] & ~state[2]:
                cover["burn"] += 1
            back = _OPPOSITE[press]
            bodies = state[1] | ((1 << state[0]) if state[0] >= 0 else 0)
            if bodies & board.shift(nxt[2], back):
                cover["cancel_death"] += 1
            if bodies & board.shift(state[5], back):
                cover["cancel_lock"] += 1
            if state[1] & board.shift(board.black, back):
                cover["cancel_black"] += 1
            moved = bin(state[1] & ~nxt[1]).count("1")
            if moved > 1:
                cover["convoy"] += 1
            if moved == 0 and state[1]:
                cover["blocked"] += 1
            if nxt[4] != state[4]:
                cover["keyspend"] += 1
        elif nxt[2] != state[2] or nxt[3] != state[3]:
            cover["swap"] += 1
        if nxt[3]:
            cover["filler_alive"] += 1
        if nxt[6] != state[6]:
            cover["pickup"] += 1
        if nxt != truth or won != truth_won:
            mismatches += 1
            print(f"MISMATCH press={press} {eng.height}x{eng.width}\n"
                  f"  model  {nxt} won={won}\n  engine {truth} won={truth_won}")
            return truth
        return nxt

    # Phase 1: the shipped levels, replayed along their own solutions.
    expert = FracturedExpert(game)
    for level in range(game.n_levels):
        if level in FracturedIdentitySolver.skip_levels:
            continue
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            continue
        game.set_level(level)
        board, state = read_board(eng, g)
        for press in plan:
            state = compare(board, state, press)

    # Phase 2: random boards.
    for _ in range(n_boards):
        h, w = rng.randint(3, 6), rng.randint(3, 6)
        cells = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(cells)
        n = len(cells)
        take = 0

        def grab(k):
            nonlocal take
            out = set(cells[take:take + k])
            take += k
            return out

        walls = grab(rng.randint(0, n // 4))
        p1s = grab(rng.randint(0, 1))
        p2s = grab(rng.randint(1, max(1, n // 2)))
        # Layer 6 holds one of these at most -- they share a collision layer.
        l6 = {cell: rng.choice(["teleporter", "teleporter2", "teleporter3",
                                "key", "lock"])
              for cell in grab(rng.randint(0, n // 2))}
        floor = grab(rng.randint(0, n))
        blacks = grab(rng.randint(0, 2))
        rest = cells[take:]
        deaths = {c for c in rest if rng.random() < 0.5}
        fillers = {c for c in rest if c not in deaths and rng.random() < 0.3}
        buttons = {}
        for cell in grab(rng.randint(0, 3)):
            buttons[cell] = (["offbutton", "obutton", "onbutton"]
                             if rng.random() < 0.6 else
                             rng.sample(["offbutton", "obutton", "onbutton"],
                                        rng.randint(1, 2)))

        grid = []
        for r in range(h):
            row = []
            for c in range(w):
                cell = {g.obj_name_to_idx["background"]}
                key = (r, c)
                for name in buttons.get(key, ()):
                    cell.add(ids[name])
                if key in blacks:
                    cell.add(ids["blackened"])
                elif key in floor:
                    cell.add(ids["forevers"])
                if key in deaths:
                    cell.add(ids["death"])
                elif key in fillers:
                    cell.add(ids["filler"])
                if key in l6:
                    cell.add(ids[l6[key]])
                if key in walls:
                    cell.add(ids["wall"])
                elif key in p1s:
                    cell.add(ids["player1"])
                elif key in p2s:
                    cell.add(ids["player2"])
                row.append(cell)
            grid.append(row)
        eng.grid, eng.height, eng.width = grid, h, w
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()
        eng._rule_win = False

        board, state = read_board(eng, g)
        for _s in range(n_steps):
            state = compare(board, state, rng.choice(PRESSES))
        if mismatches > 4:
            return 1
    print(f"fuzz: {steps} transitions, {mismatches} mismatches")
    print("coverage: " + "  ".join(f"{k}={v}" for k, v in cover.items()))
    missing = [k for k, v in cover.items() if not v]
    if missing:
        print(f"NOT EXERCISED: {missing}")
    return 0 if not mismatches and not missing else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--ties" in sys.argv:
        sys.exit(_ties([int(x) for x in sys.argv[sys.argv.index("--ties") + 1:]
                        if x.isdigit()]))
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    sys.exit(FracturedIdentitySolver.main())
