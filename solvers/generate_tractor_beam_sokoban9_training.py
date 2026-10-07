"""Generate Phase-1 training data for the PuzzleScript game
ps:tractor_beam_sokoban9 ("Tractor Beam Sokoban9", Franklin P. Dyer).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanic, a native model of it, the search that model is planned with, the
proof that all ten plans are shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_tractor_beam_sokoban9",
      "levels": [
        {
          "level_id": 0,
          "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
          "actions":      [a_0, a_1, ..., a_{T-1}], # length T
        },
        ...
      ]
    }

actions[0] is the RESET that produced obs[0]; actions[i] for i>=1 is the action
that took the agent from obs[i-1] to obs[i]. The recorded index is the *screen*
action (post rotation remap), i.e. the button an agent presses in the augmented
view, so replaying the recorded actions reproduces the recorded frames exactly.
Every recorded expert step carries an optimal-action set, and on this game every
one of those sets is EXACT.

The game
--------
A sokoban in which you may not touch a crate. Walking into one does not push it,
it CANCELS the turn outright, and the same is true of a wall and of the
grey PlayerWall lattice. The only thing you can do to a crate is shoot a
TRACTOR BEAM at it (the X button), and then it is welded to you until you shoot
again.

**The beam picks exactly one crate and you do not choose which.** Pressing X
with empty hands runs a fixed scan -- UP, then DOWN, then LEFT, then RIGHT --
and grabs the NEAREST crate along the first of those four rays that has one.
The ray is unconstrained: it passes straight through walls, through PlayerWalls
and across the whole board, so "in line with" is the only thing that matters and
"behind a wall" is not a defence. That scan order is the whole positioning
puzzle: to take the crate on your left you must first make sure there is nothing
above you, nothing below you, and nothing nearer on the left.

**While a crate is held it copies your displacement, at any range.** It is
drawn dark blue instead of orange (`PickedUp`, a different object from `Crate`),
and every direction press moves it one square the way you moved -- across the
room, through a PlayerWall, over a target -- with no line of sight required. The
board's one global rule is that you may hold at most ONE crate at a time; the
game implements it with a `Sensor`/`On` pair sitting invisibly on every wall
square, flipped at the top of each turn by

    [On] -> [Sensor]
    [Sensor][PickedUp] -> [On][PickedUp]

which reads as "the beam is available iff nothing is held", and is why pressing
X while holding DROPS instead of grabbing. Nothing about that is hidden state:
`On` is recomputed from scratch every turn, so it says exactly what the
dark-blue crate on the screen already says.

**A held crate that cannot move simply stays put while you walk on**, and that
is the second half of the mechanic. The player and the crate are two independent
movers: if the crate's next square is a wall, another crate or the board edge it
is left behind and the offset between you changes, which is the only way to
re-aim the beam without dropping. But it is not symmetric -- if it is YOUR next
square that holds a wall, a PlayerWall or a crate, the cancel rules fire and the
whole turn is a no-op, crate included.

The win is ``All Target on Crate``: every target square must carry an orange
crate. A held crate is a `PickedUp` and does not count, so a level ends on a
DROP, never on a move.

Three details that decide plans and are not visible from the rule list:

  * **The rule list runs ONCE per turn in this interpreter**, not to a fixed
    point. That is what makes the `Sensor`/`On` pair mean "holding at the START
    of this turn" -- the grab rules run after the flip and before anything they
    do can flip it back -- and it is why X grabs on one press and drops on the
    next rather than doing both in one.
  * **The grab clears the player's Action flag** (the rules' right-hand sides
    say `Player`, not `Action Player`), so the drop rule that follows them
    cannot fire on the same press. With no crate on any of the four rays the
    flag survives, the press does nothing at all, and the state is unchanged.
  * **A cancelled turn is a true no-op**, including the `Sensor`/`On` flip: the
    interpreter reverts the whole turn, so a press into a wall costs a move and
    changes nothing else.

Expert solver
-------------
A NATIVE model of the turn above (`_Board.step`), fuzz-verified against the real
interpreter (`--selfcheck`), planned in two stages:

  * `_astar` -- A* over the model with the admissible `_Board.heuristic`, at
    unit cost and weight 1, so its answer is the SHORTEST press count. All ten
    levels close: 12, 6, 27, 11, 21, 39, 35, 8, 19 and 40 presses, the whole set
    in 2.6 seconds, of which level 6 is 2.4.
  * `_Field` -- one more pass over the states A* proved relevant, which turns
    the plan into an exact distance-to-win field: a forward sweep keeping every
    state with ``g + h <= bound``, then a backward BFS from the winning edges.
    At the exact bound that cone runs from 13 states to 42132 and costs about a
    tenth of the A* that sized it. Every step of every recorded plan is
    labelled with the EXACT set of presses that are equally shortest there,
    read off that field rather than inferred.

`_Board.heuristic` is the lower bound both stages lean on. A press moves at most
one crate at most one square, and a crate that is not already standing on its
target has to be grabbed and dropped -- two presses that move nothing -- so for
any assignment of distinct crates to targets

    presses-to-win  >=  SUM over targets of ( 2 + squares from its crate )

with the 2 dropped to 1 for the crate already in the beam (it still owes its
drop) and to 0 for a crate already parked on its target. Taking the cheapest
assignment (a subset-DP over the at most six targets) is a relaxation of a
relaxation, so the estimate is admissible, which is what makes A*'s answers
proofs and the field's tie sets exact. "Squares from its crate" is a BFS over
NON-WALL squares, because a held crate crosses PlayerWalls and targets freely
and is stopped only by walls, other crates and the board edge.

Shortest, and how that is known
-------------------------------
All ten plans are provably shortest, and `--selfcheck` derives that three
independent ways:

  * A* with the admissible estimate above;
  * a plain breadth-first search over the same model with no estimate at all,
    on the eight levels whose whole reachable space is small enough to sweep
    (levels 0, 1, 2, 3, 5, 7, 8 and 9 -- 1333 to 196107 states that can still
    be won, out of 1362 to 199084 reachable, so a quarter of level 5's space is
    already lost);
  * the field's own ``dist(start)``, which is built by a backward BFS and never
    consults the heuristic at all.

The tie sets get the same treatment: on those eight levels every set the field
reports is compared against the one a full breadth-first distance table gives,
set for set.

Recovery
--------
``recovery_mode = "reset"``. The mechanic is irreversible in the strong sense --
a crate shoved into a corner cannot be pulled out, and a crate dropped where it
blocks the ray you needed cannot be picked up again from where you stand -- so
the episode-wide exploration prefix explores freely and ONE RESET restores the
level's initial state, from which the cached plan replays a guaranteed WIN.

`_search` is nonetheless a genuine re-plan from the LIVE board: it reads the
engine grid every time it is called, so a probe from an arbitrary state is
answered from that state. It answers from the level's field when the field can
CERTIFY the answer (see `_Field.certified`) and falls back to a fresh A* plus a
fresh field otherwise, so the answer is exact either way and never a replayed
prefix.

Epsilon detours are ON everywhere except level 6, through
`PSAStarSolver.epsilon_for`. `record_level` vets every detour by re-planning
from it, and on the nine cheap levels the level's field is built once with eight
presses of slack so that vetting is a dictionary lookup -- all nine of those
fields together take 1.3 seconds to build. Level 6's would be 982747 states and
44 seconds against 42132 and a fifth of a second at the exact bound, and
without it every detour probe is a fresh 2.4-second A*, which is not worth one
extra mistake per twenty presses. Recovery data for level 6 comes from the
exploration prefix and its RESET, which is unaffected.

The eight-press slack is also a policy, not just a budget: a detour the field
cannot price is REFUSED, so the mistakes that get recorded are the ones a player
can come back from within eight presses, and the ones that strand a crate
forever are never taken.

Rendering
---------
One sprite in ``data/puzzlescript_games/Tractor_Beam_Sokoban9.txt`` is redrawn;
the header comment there is the full account and ``--audit`` is the check. In
one line: this game is a copy of the stock PuzzleScript sokoban art and it
shipped the template's bug -- Target was a ring drawn exactly under the opaque
middle of the Player, so a player standing on a bare target was pixel-identical
to a player standing on grass, on all eight board shapes. No rule, level, layer
or win condition is touched.
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

from arcengine import ActionInput, GameState                        # noqa: E402
from solvers.common.ps_astar import (                               # noqa: E402
    DIRECTIONS, PSAStarSolver, PSExpert, Plan, screen_action,
)

GAME_NAME = "Tractor_Beam_Sokoban9"
GAME_ID = "puzzlescript_tractor_beam_sokoban9"

#: Where the ten level-start plans live between processes. Level 6's A* is a
#: several-second search and the rest are together about as much again; without
#: this every shard `parallelize_generator` starts would re-derive all ten. See
#: `PSExpert.plan` for when an entry is trusted.
PLAN_CACHE = (Path(__file__).resolve().parent.parent / "data"
              / "tractor_beam_sokoban9_plans.json")

#: Engine actions, in the order every search branches on them. This is also the
#: tie-break order for a plan read off the field, which is what makes a
#: re-derived plan byte-identical across processes.
DIRS = tuple(DIRECTIONS)
DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: The beam's scan order. Spelled out as its own tuple rather than taken off
#: `DELTA`: it is the order the four grab rules appear in the .txt (UP, DOWN,
#: LEFT, RIGHT), it is the single most surprising fact about this game, and it
#: happening to match the order some other list is written in is not a reason
#: for one to silently follow the other.
SCAN = ("up", "down", "left", "right")

_INF = 1 << 30

#: State: ``(player, held, mask)`` -- the player's cell index, the cell index of
#: the crate currently in the beam (-1 for none) and a bitmask of the cells
#: holding a loose Crate. `held` is NOT in `mask`: a PickedUp is a different
#: object from a Crate and the win condition only counts Crates.


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """A level's static scenery, its turn function and its lower bound.

    Built once per LAYOUT (keyed by `sig` on the expert) and shared by every
    state read from it: walls, PlayerWalls and targets are static here -- no
    rule in the game creates or destroys any of them -- so the per-target
    distance tables and the heuristic memo below can live on the board rather
    than being rebuilt per node.
    """

    __slots__ = ("h", "w", "wall", "pwall", "targets", "tmask", "sig",
                 "_tab", "_memo")

    def __init__(self, h, w, walls, pwalls, targets):
        self.h, self.w = h, w
        self.wall = frozenset(walls)
        self.pwall = frozenset(pwalls)
        self.targets = tuple(sorted(targets))
        self.tmask = 0
        for t in self.targets:
            self.tmask |= 1 << t
        self.sig = (h, w, tuple(sorted(self.wall)), tuple(sorted(self.pwall)),
                    self.targets)
        self._tab = {t: self._reach(t) for t in self.targets}
        self._memo: dict = {}

    # -- geometry ---------------------------------------------------------
    def _reach(self, src) -> dict:
        """``{cell: squares a crate must travel from cell to src}``, over the
        NON-WALL squares.

        A held crate is stopped by walls, by other crates and by the board
        edge, and by nothing else -- it crosses PlayerWalls, targets and the
        square the player is vacating. Dropping the other crates is the
        relaxation that makes this a lower bound rather than a route."""
        h, w = self.h, self.w
        dist = {src: 0}
        queue = deque([src])
        while queue:
            i = queue.popleft()
            r, c = divmod(i, w)
            for dr, dc in DELTA.values():
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    j = rr * w + cc
                    if j not in self.wall and j not in dist:
                        dist[j] = dist[i] + 1
                        queue.append(j)
        return dist

    def cells(self, mask) -> list:
        """The set bits of ``mask``, as cell indices."""
        out = []
        while mask:
            low = mask & -mask
            out.append(low.bit_length() - 1)
            mask ^= low
        return out

    # -- the turn ---------------------------------------------------------
    def won(self, state) -> bool:
        """``All Target on Crate``. The held crate is a PickedUp and is
        deliberately not counted -- that is why a level ends on a drop."""
        return (state[2] & self.tmask) == self.tmask

    def step(self, state, d):
        """One press. Returns the settled state; an unchanged state means the
        press did nothing (a cancelled turn, a beam that found no crate, a walk
        into the board edge with nothing held).

        Verified press-for-press against the interpreter by `--selfcheck`."""
        p, held, mask = state
        h, w = self.h, self.w
        if d == "action":
            if held >= 0:
                # Holding: the Sensors are all On, so the four grab rules
                # cannot match and the drop rule downstream of them fires.
                return (p, -1, mask | (1 << held))
            # Empty-handed: the first ray with a crate on it wins, and the
            # NEAREST crate on that ray is the one taken. The ray ignores
            # everything it passes over -- `...` constrains nothing.
            r, c = divmod(p, w)
            for ray in SCAN:
                dr, dc = DELTA[ray]
                rr, cc = r + dr, c + dc
                while 0 <= rr < h and 0 <= cc < w:
                    i = rr * w + cc
                    if mask >> i & 1:
                        return (p, i, mask & ~(1 << i))
                    rr += dr
                    cc += dc
            return state                      # the Action flag dies unused

        dr, dc = DELTA[d]
        r, c = divmod(p, w)
        nr, nc = r + dr, c + dc
        pin = 0 <= nr < h and 0 <= nc < w
        npi = nr * w + nc if pin else -1
        if pin and (npi in self.wall or npi in self.pwall or mask >> npi & 1):
            return state                      # cancel: the WHOLE turn is undone
        if held < 0:
            return (npi, -1, mask) if pin else state

        hr, hc = divmod(held, w)
        khr, khc = hr + dr, hc + dc
        hin = 0 <= khr < h and 0 <= khc < w
        nhi = khr * w + khc if hin else -1
        # The player and the crate are independent movers that happen to share
        # a direction, so each is blocked on its own account -- and then at most
        # one of them can be waiting on the other (the two dependencies would
        # need `2 * delta == 0`).
        cb = (not hin) or (nhi in self.wall) or bool(mask >> nhi & 1)
        pb = not pin
        if pb and not cb and nhi == p:
            cb = True                          # the crate is following a wall
        if cb and not pb and npi == held:
            pb = True                          # the player is behind its crate
        return (p if pb else npi, held if cb else nhi, mask)

    # -- the lower bound --------------------------------------------------
    def heuristic(self, state) -> int:
        """Presses that must still be spent, never more than the truth.

        A press moves at most one crate at most one square, and every crate not
        already parked on the target it is assigned to owes a grab and a drop
        (two presses that move nothing at all) -- one press for the crate
        already in the beam, which still owes its drop. Summing that over the
        cheapest assignment of distinct crates to targets is a lower bound on
        any winning press sequence, so the estimate is admissible.

        Memoised on ``(mask, held)``: the player's cell does not enter it, and
        a search sees the same crate layout from many squares.

        `_INF` means some target has no crate that can ever reach it -- a crate
        wedged in a pocket whose only exits are walls. `_astar` drops such a
        node instead of queueing it: nothing downstream can win."""
        key = (state[2], state[1])
        got = self._memo.get(key)
        if got is not None:
            return got
        items = [(c, 2) for c in self.cells(state[2])]
        if state[1] >= 0:
            items.append((state[1], 1))
        tgts = self.targets
        n = len(tgts)
        full = (1 << n) - 1
        best = [_INF] * (full + 1)
        best[0] = 0
        for cell, base in items:
            row = []
            for t in tgts:
                if cell == t and base == 2:
                    row.append(0)              # already parked, costs nothing
                    continue
                reach = self._tab[t].get(cell)
                row.append(_INF if reach is None else base + reach)
            nxt = best[:]
            for msk in range(full + 1):
                have = best[msk]
                if have >= _INF:
                    continue
                for j in range(n):
                    if msk >> j & 1 or row[j] >= _INF:
                        continue
                    cost = have + row[j]
                    if cost < nxt[msk | (1 << j)]:
                        nxt[msk | (1 << j)] = cost
            best = nxt
        self._memo[key] = best[full]
        return best[full]


# ---------------------------------------------------------------------------
# The searches
# ---------------------------------------------------------------------------

def _astar(board: _Board, start, cap: int = 2_000_000):
    """Shortest press sequence from ``start``, or None (unwinnable, or past
    ``cap`` expanded states).

    Unit costs, weight 1 and `_Board.heuristic` admissible, so the length of
    what comes back is the exact distance to a win. The win is tested on
    GENERATION rather than on pop, which is sound here precisely because every
    edge costs the same one press.
    """
    if board.won(start):
        return []
    counter = 0
    heur = board.heuristic
    step = board.step
    pq = [(heur(start), 0, counter, start, ())]
    best_g = {start: 0}
    while pq:
        _f, g, _c, state, path = heapq.heappop(pq)
        if g > best_g.get(state, _INF):
            continue                            # a stale heap entry
        for d in DIRS:
            nxt = step(state, d)
            if nxt == state:
                continue                        # a press that did nothing
            if board.won(nxt):
                return list(path) + [d]
            ng = g + 1
            if best_g.get(nxt, _INF) <= ng:
                continue
            est = heur(nxt)
            if est >= _INF:
                continue                        # a target no crate can reach
            best_g[nxt] = ng
            counter += 1
            heapq.heappush(pq, (ng + est, ng, counter, nxt, path + (d,)))
        if len(best_g) > cap:
            return None
    return None


def _bfs_dist(board: _Board, start, cap: int = 400_000):
    """``{state: presses to win}`` for the WHOLE component reachable from
    ``start``, by plain breadth-first search with no estimate at all. None past
    ``cap`` states.

    This is `--selfcheck`'s independent derivation: it shares the turn function
    with `_astar` and the field and shares nothing else, so an agreement
    between the three is a real check on the heuristic being admissible and on
    the field's pruning being sound. Affordable on six of the ten levels.
    """
    seen = {start}
    queue = deque([start])
    edges: dict = {}
    while queue:
        state = queue.popleft()
        out = {}
        for d in DIRS:
            nxt = board.step(state, d)
            if nxt == state:
                continue
            if board.won(nxt):
                out[d] = None                   # a winning edge
                continue
            if nxt not in seen:
                if len(seen) >= cap:
                    return None
                seen.add(nxt)
                queue.append(nxt)
            out[d] = nxt
        edges[state] = out
    return _backward(edges)


def _backward(edges: dict) -> dict:
    """Presses-to-win for every state in ``edges``, by backward BFS from the
    winning edges over the reversed graph. States absent from the result cannot
    win at all."""
    rev: dict = {}
    dist: dict = {}
    frontier = []
    for state, out in edges.items():
        for _d, nxt in out.items():
            if nxt is None:
                if state not in dist:
                    dist[state] = 1
                    frontier.append(state)
            else:
                rev.setdefault(nxt, []).append(state)
    queue = deque(frontier)
    while queue:
        state = queue.popleft()
        for prev in rev.get(state, ()):
            if prev not in dist:
                dist[prev] = dist[state] + 1
                queue.append(prev)
    return dist


class _Field:
    """The exact distance-to-win field over one CONE of a board's state space.

    Built from a root and a bound: the forward sweep keeps every state whose
    ``g + h`` is within the bound, and the backward BFS then reads the exact
    distance off the winning edges. With ``bound = d*(root)`` the cone is the
    set of states on some shortest path from the root (levels 0-9 here: 13 to
    42k states); with a slack on top it also holds everything reachable by
    that many wasted presses, which is what makes an epsilon detour a lookup
    instead of a fresh search.

    WHY A CONE AND NOT THE WHOLE SPACE. The full reachable space is 1.35M
    states on level 4 and past 4M on level 6, so a plain enumeration is not on
    the table for either; the cone for the same two is 738 and 42132. What it
    buys is the same three things enumeration buys -- provably shortest plans,
    EXACT optimal-action sets and an oracle that answers from any state it
    covers rather than replaying a prefix -- for the cost of about one extra
    A*.

    WHY IT IS EXACT, in one paragraph. `dist` can only ever over-report (an
    optimal route that leaves the cone is not seen), so it is enough to show
    the routes that matter stay inside. Take a state ``s`` with
    ``g(s) + dist(s) <= bound``, where ``g`` is its true distance from the root.
    Every state ``t`` on a shortest path from ``s`` to a win has
    ``g(t) <= g(s) + dist(s) - dist(t)`` and, since the heuristic is
    admissible, ``h(t) <= dist(t)``; adding them gives
    ``g(t) + h(t) <= g(s) + dist(s) <= bound``, so ``t`` is kept -- and the
    same sum bounds the edge test, so the edges between them are kept too.
    Hence the field's `dist` is the true distance for exactly the states whose
    ``g + dist`` fits the bound, and `certified` is that test, evaluated on the
    field's own numbers (if the true distance were smaller the lemma would
    apply to it and contradict `dist` being the minimum found).

    That is also why the optimal SETS are exact: a successor that ties is on a
    shortest path by definition, so it is measured exactly, and one that does
    not tie can only be over-reported, so it cannot sneak into the set.
    """

    __slots__ = ("board", "root", "bound", "g", "succ", "dist")

    def __init__(self, board: _Board, root, bound: int, cap: int = 2_000_000):
        self.board = board
        self.root = root
        self.bound = bound
        self.g = {root: 0}
        self.succ: dict = {}
        queue = deque([root])
        heur = board.heuristic
        step = board.step
        g = self.g
        while queue:
            state = queue.popleft()
            gs = g[state]
            out = {}
            for d in DIRS:
                nxt = step(state, d)
                if nxt == state:
                    continue
                if board.won(nxt):
                    out[d] = None
                    continue
                if gs + 1 + heur(nxt) > bound:
                    continue
                if nxt not in g:
                    if len(g) > cap:
                        raise MemoryError("field cap exceeded")
                    g[nxt] = gs + 1
                    queue.append(nxt)
                out[d] = nxt
            self.succ[state] = out
        self.dist = _backward(self.succ)

    def certified(self, state) -> bool:
        """True when this field's `dist` for ``state`` is the real distance to
        a win -- see the class docstring."""
        d = self.dist.get(state)
        return d is not None and self.g.get(state, _INF) + d <= self.bound

    def optimal(self, state) -> list:
        """Every press that is equally shortest at ``state``, in `DIRS` order."""
        best = self.dist[state]
        out = self.succ[state]
        keep = []
        for d in DIRS:
            if d not in out:
                continue
            nxt = out[d]
            cost = 1 if nxt is None else (
                _INF if nxt not in self.dist else self.dist[nxt] + 1)
            if cost == best:
                keep.append(d)
        return keep

    def plan(self, state) -> Plan:
        """The shortest press sequence from ``state``, each step carrying its
        exact optimal set. Ties break by `DIRS` order, which is what makes a
        re-derived plan byte-identical across processes."""
        presses, optsets = [], []
        while True:
            best = self.optimal(state)
            presses.append(best[0])
            optsets.append(best)
            nxt = self.succ[state][best[0]]
            if nxt is None:
                return Plan(presses, optsets)
            state = nxt


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class TractorBeamSokoban9Expert(PSExpert):
    """`PSExpert`'s plan memo, disk cache and snapshot discipline around the
    native model.

    The base class keeps the memo, the level scoping and the disk cache; only
    the strategy underneath changes. `_search` reads the ENGINE's grid every
    time it is called, so a re-plan from an arbitrary board -- a recovery
    probe, an epsilon detour, a `--selfcheck` perturbation -- is answered from
    that board and not from a replayed prefix.
    """

    directions = list(DIRS)
    #: `_key` is the dynamic objects only, canonical WITHIN a level but not
    #: across them: walls, PlayerWalls and targets are static per level and
    #: differ between levels.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    #: Presses of slack the level-START field is built with, by level. The
    #: field then prices any state within that many wasted presses of a
    #: shortest path, so `record_level`'s detour vetting is a dictionary lookup
    #: rather than a fresh search -- and a detour it cannot price is refused
    #: rather than taken blind.
    #:
    #: MEASURED, not guessed. At slack 8 the nine cheap levels' cones are 389
    #: to 55333 states and 1.3 seconds to build between them, while level 6's is
    #: 982747 states and 44 seconds, against 42132 and 0.22s at slack 0. Level
    #: 6 therefore gets the exact cone and no detours; see
    #: `TractorBeamSokoban9Solver.epsilon_for`.
    slack: dict = {6: 0}
    default_slack: int = 8

    def setup(self) -> None:
        g = self.g
        self.player_ids = set(self.game._engine._player_indices)
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.pwall_ids = set(g.resolve_object_name("playerwall"))
        self.target_ids = set(g.resolve_object_name("target"))
        self.crate_ids = set(g.resolve_object_name("crate"))
        self.held_ids = set(g.resolve_object_name("pickedup"))
        #: `_Board`s by static signature, so a level's scenery, its per-target
        #: distance tables and its heuristic memo are built once and shared by
        #: every state read from it.
        self._boards: dict = {}
        #: Level starts, by board signature -- `_search` needs to know which
        #: state is the root the level's field should be built from.
        self._starts: dict = {}
        #: Each board's field, and the level it was noted for.
        self._fields: dict = {}
        self._levels: dict = {}

    def heuristic(self, eng) -> int:
        """Unused: `_search` plans on the model, not on engine snapshots. The
        estimate that matters is `_Board.heuristic`, and it is applied to model
        states."""
        raise AssertionError(
            "TractorBeamSokoban9Expert plans on the native model; "
            "heuristic is unused")

    # -- engine <-> model -------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, state)`` for the engine's current grid."""
        w = eng.width
        walls, pwalls, targets = [], [], []
        player = held = -1
        mask = 0
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.pwall_ids:
                    pwalls.append(i)
                if cell & self.target_ids:
                    targets.append(i)
                if cell & self.player_ids:
                    player = i
                if cell & self.held_ids:
                    held = i
                if cell & self.crate_ids:
                    mask |= 1 << i
        # `_Board.sig` spelled out without building a `_Board`: `read` runs on
        # every plan and every detour probe, and a board costs a BFS per
        # target. The grid scan is row-major, so these tuples come out in the
        # order `_Board.sig` sorts them into; `note_start` asserts it.
        sig = (eng.height, w, tuple(walls), tuple(pwalls), tuple(targets))
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(eng.height, w, walls, pwalls,
                                               targets)
        return board, (player, held, mask)

    def note_start(self, eng, level: int) -> None:
        """Record the board the engine is sitting on as ``level``'s START.

        `_search` treats a start differently -- it is the root the level's
        reusable field is built from, and the only state whose field is worth
        the slack -- so it has to be able to recognise one.
        `TractorBeamSokoban9Solver.prepare_expert` calls this once per level
        before anything plans."""
        board, state = self.read(eng)
        assert board.sig in self._boards, "read() and _Board disagree on sig"
        self._starts[board.sig] = state
        self._levels[board.sig] = level

    def _key(self, eng) -> frozenset:
        """``(row, col, tag)`` triples for everything that moves: the player
        (0), the crate in the beam (1) and each loose crate (2).

        Overrides the base's "every non-background cell", which would put the
        `Sensor` / `On` markers in the key. Those are not state: rules 1 and 2
        recompute them from scratch at the top of every turn, so two grids
        differing only in which of the pair is on the walls have identical
        futures -- keying on them would split every state in the game in two
        and halve the plan memo's hit rate for nothing. (The board's static
        scenery is left out for the usual reason; hence ``scope_by_level``.)"""
        _board, (p, held, mask) = self.read(eng)
        w = eng.width
        out = {(i // w, i % w, 2) for i in _board.cells(mask)}
        if p >= 0:
            out.add((p // w, p % w, 0))
        if held >= 0:
            out.add((held // w, held % w, 1))
        return frozenset(out)

    # -- planning ---------------------------------------------------------
    def field_for(self, board: _Board):
        """The level-START field for ``board``, built on first use.

        Lazy rather than eager because a process that hits the disk plan cache
        never needs one: the cached `Plan` already carries its exact optimal
        sets. The first epsilon detour is what pays for it, once per process."""
        got = self._fields.get(board.sig)
        if got is not None:
            return got
        root = self._starts.get(board.sig)
        if root is None:
            return None
        shortest = _astar(board, root, self.node_cap)
        if shortest is None:
            self._fields[board.sig] = False
            return False
        lvl = self._levels.get(board.sig)
        bound = len(shortest) + self.slack.get(lvl, self.default_slack)
        try:
            field = _Field(board, root, bound, self.node_cap)
        except MemoryError:
            field = _Field(board, root, len(shortest), self.node_cap)
        self._fields[board.sig] = field
        return field

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if state[0] < 0:
            return None                    # no player: nothing to drive
        if board.won(state):
            return Plan([], [])
        field = self.field_for(board)
        if field and field.certified(state):
            return field.plan(state)
        # Off the level's field (a detour past its slack, or a board this
        # expert has never seen a start for). Solve this state on its own:
        # A* for the exact distance, then its own exact cone for the sets.
        shortest = _astar(board, state, self.node_cap)
        if shortest is None:
            return None
        try:
            local = _Field(board, state, len(shortest), self.node_cap)
        except MemoryError:
            return Plan(shortest, [[d] for d in shortest])
        if not local.certified(state):      # cannot happen; do not guess
            return Plan(shortest, [[d] for d in shortest])
        return local.plan(state)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class TractorBeamSokoban9Solver(PSAStarSolver):
    game_id = GAME_ID
    game_name = GAME_NAME
    expert_cls = TractorBeamSokoban9Expert

    #: `games/ps:tractor_beam_sokoban9/ps:tractor_beam_sokoban9.py` is a plain
    #: passthrough -- it builds the adapter and nothing else, and the rendering
    #: fix is in the .txt that both paths read. Set this if that wrapper ever
    #: grows a patch.
    game_module_id = ""

    #: Both are the expert's, not the base's: `TractorBeamSokoban9Expert`
    #: never runs `PSExpert._astar` on engine snapshots. ``node_cap`` is passed
    #: through to the model searches; ``weight`` is left at 1 because that is
    #: what makes the plans proofs, and there is nothing here to trade it for.
    weight = 1
    node_cap = 2_000_000

    #: The longest plan is level 9's 40 presses, and the field's eight presses
    #: of detour slack are the most an episode can add, so this is over twice
    #: the worst case -- and well under the adapter's own 200-press per-level
    #: cap, which `record_level` restarts with the `set_level` that ends the
    #: exploration prefix.
    max_steps = 120

    #: Every level is solved; nothing is skipped.
    skip_levels = frozenset()

    #: On top of the RESET prefix: about one press in twenty is a random legal
    #: alternative, after which the expert re-plans from wherever it landed --
    #: the taken action is the mistake and ``optimal`` is the recovery.
    #: `record_level` vets each alternative by re-planning from it first, and
    #: here that re-plan is a field lookup, so a detour never bricks a level
    #: and never costs a search.
    epsilon = 0.05

    #: Levels whose START field is built with slack, so a detour off the
    #: shortest path is still priced exactly. Level 6 is the exception and the
    #: reason is measured: its slack-8 cone is 982747 states and 44 seconds to
    #: build against 42132 and 0.22s at slack 0, and without the field every
    #: detour probe would be a fresh 2.4-second A*. See
    #: `TractorBeamSokoban9Expert.slack`.
    detour_levels: frozenset = frozenset({0, 1, 2, 3, 4, 5, 7, 8, 9})

    def epsilon_for(self, level: int) -> float:
        """Detours everywhere the level's field can vet them for free.

        Recovery data for level 6 comes from the exploration prefix and its
        RESET, which is the `recovery_mode = "reset"` contract and is
        unaffected by this."""
        return self.epsilon if level in self.detour_levels else 0.0

    def prepare_expert(self, game, expert) -> None:
        """Seat every level once, to tell the expert which board is which
        level's start, then plan them all.

        Planning here rather than letting `discover_solvable` trigger it is the
        same work either way, but it fills the disk plan cache in one pass and
        makes the startup cost visible as startup."""
        for level in range(game.n_levels):
            game.set_level(level)
            expert.note_start(game._engine, level)
        for level in range(game.n_levels):
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    """The solver, its adapter and an expert that knows where every level
    starts -- without planning anything.

    The reports below each want a different slice (``--audit`` needs no plan at
    all, ``--selfcheck`` needs only the model), so planning is left to whoever
    asks for it rather than paid on every entry point."""
    solver = TractorBeamSokoban9Solver()
    game = solver.make_game(seed)
    expert = TractorBeamSokoban9Expert(game, node_cap=solver.node_cap)
    for level in range(game.n_levels):
        game.set_level(level)
        expert.note_start(game._engine, level)
    return solver, game, expert


def _ascii(board: _Board, state) -> str:
    """One board as text -- ``#`` wall, ``/`` PlayerWall, ``o`` target,
    ``*`` crate, ``@`` crate on target, ``U`` the crate in the beam,
    ``P`` player, ``p`` player on target."""
    p, held, mask = state
    rows = []
    for r in range(board.h):
        line = ""
        for c in range(board.w):
            i = r * board.w + c
            tgt = i in board.targets
            if i in board.wall:
                ch = "#"
            elif i == p:
                ch = "p" if tgt else "P"
            elif i == held:
                ch = "U"
            elif mask >> i & 1:
                ch = "@" if tgt else "*"
            elif i in board.pwall:
                ch = "/"
            elif tgt:
                ch = "o"
            else:
                ch = "."
            line += ch
        rows.append(line)
    return "\n".join(rows)


def _ids(game) -> tuple:
    """``(background, target, wall, player, playerwall, crate, pickedup,
    sensor, on)`` object indices, for `_seat`."""
    g = game._game
    name = g.obj_name_to_idx
    return tuple(name[k] for k in ("background", "target", "wall", "player",
                                   "playerwall", "crate", "pickedup",
                                   "sensor", "on"))


def _seat(eng, board: _Board, state, ids) -> None:
    """Write a model state onto the engine's grid.

    The Sensor / On marker is seated to match the state -- ``On`` on every wall
    square iff a crate is held -- which is what the interpreter would be
    holding after the press that produced this state. It makes no difference to
    what happens next (rules 1 and 2 recompute it before anything reads it),
    and seating it correctly is what lets `--selfcheck` compare whole grids
    rather than a filtered view of them."""
    bg, tgt, wall, player, pwall, crate, held, sensor, on = ids
    flag = on if state[1] >= 0 else sensor
    grid = []
    for r in range(board.h):
        row = []
        for c in range(board.w):
            i = r * board.w + c
            cell = {bg}
            if i in board.wall:
                cell |= {wall, flag}
            if i in board.pwall:
                cell.add(pwall)
            if i in board.targets:
                cell.add(tgt)
            if i == state[0]:
                cell.add(player)
            if i == state[1]:
                cell.add(held)
            if state[2] >> i & 1:
                cell.add(crate)
            row.append(cell)
        grid.append(row)
    eng.grid = grid
    eng.height, eng.width = board.h, board.w
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


def _plans(verbose: bool = True) -> int:
    """Re-derive every level's plan from scratch and check the three things a
    plan has to be: a WIN in the real interpreter, the length A* says is
    shortest, and the length the field's backward BFS independently says.

    Also reports the exact tie sets' size, which is the part of the recording
    that no other report measures."""
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    total = ties = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, start = expert.read(eng)
        shortest = _astar(board, start, expert.node_cap)
        field = expert.field_for(board)
        plan = expert.plan(eng, level)
        if plan is None or shortest is None or not field:
            print(f"  L{level}: NO PLAN (astar={shortest is not None}, "
                  f"field={bool(field)}, plan={plan is not None})")
            bad += 1
            continue
        d0 = field.dist.get(start)
        agree = len(plan) == len(shortest) == d0
        # replay in the interpreter
        game.set_level(level)
        for d in plan:
            eng.step(d)
        won = eng.check_win()
        sets = getattr(plan, "optsets", None)
        n_alt = 0 if sets is None else sum(len(s) for s in sets)
        total += len(plan)
        ties += n_alt
        if not agree or not won or sets is None or len(sets) != len(plan):
            print(f"  L{level}: plan {len(plan)}, astar {len(shortest)}, "
                  f"field {d0}, won={won}, sets={0 if sets is None else len(sets)}")
            bad += 1
        elif verbose:
            print(f"  L{level}: {len(plan)} presses, PROVED shortest 2 ways "
                  f"(A* {len(shortest)}, field {d0}), wins in the interpreter; "
                  f"{n_alt / len(plan):.2f} optimal presses per step; "
                  f"cone {len(field.g)} states at bound {field.bound}")
    if verbose and total:
        print(f"  {total} recorded presses, {ties} optimal labels "
              f"({ties / total:.2f} per step)")
    return bad


def _reachable(board: _Board, start, cap: int):
    """Every state reachable from ``start``, or None past ``cap``."""
    seen = {start}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if board.won(state):
            continue                       # a win ends the episode
        for d in DIRS:
            nxt = board.step(state, d)
            if nxt not in seen:
                if len(seen) >= cap:
                    return None
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _walk_states(board: _Board, start, rng, count: int) -> list:
    """``count`` states sampled by random walks from ``start`` -- reachable by
    construction, and spread over the whole space rather than over the first
    ``count`` states a BFS happens to reach."""
    out = []
    state = start
    for _ in range(count):
        if board.won(state) or rng.random() < 0.02:
            state = start
        state = board.step(state, rng.choice(DIRS))
        out.append(state)
    return out


def _random_boards(board: _Board, start, rng, count: int) -> list:
    """Synthetic states on the same scenery: the same number of crates, thrown
    anywhere, held or not.

    Most of these are unreachable, which is the point -- the turn function has
    to be right everywhere, not only on the boards a shipped level happens to
    produce, because a recovery probe is answered from wherever it is asked."""
    cells = [i for i in range(board.h * board.w) if i not in board.wall]
    walk = [i for i in cells if i not in board.pwall]
    n = bin(start[2]).count("1") + (1 if start[1] >= 0 else 0)
    out = []
    for _ in range(count):
        rng.shuffle(cells)
        p = rng.choice(walk)
        rest = [i for i in cells if i != p][:n]
        held = -1
        if rest and rng.random() < 0.5:
            held, rest = rest[0], rest[1:]
        mask = 0
        for i in rest:
            mask |= 1 << i
        out.append((p, held, mask))
    return out


def _compare(eng, board: _Board, states, ids) -> tuple:
    """Step every state in ``states`` through all five presses on BOTH the
    interpreter and the model, and count the disagreements.

    Returns ``(checked, mismatches, first)``. The comparison is the whole board
    plus the win flag, read back off the engine's grid, so a rule the model
    does not know about shows up as a mismatch rather than as a plausible
    board."""
    checked = wrong = 0
    first = None
    for state in states:
        for d in DIRS:
            _seat(eng, board, state, ids)
            eng.step(d)
            got = _read_engine(eng, board, ids)
            want = board.step(state, d)
            checked += 1
            if got != want or eng.check_win() != board.won(want):
                wrong += 1
                if first is None:
                    first = (state, d, got, want)
    return checked, wrong, first


def _read_engine(eng, board: _Board, ids) -> tuple:
    """The engine's grid as a model state (scenery asserted, not re-read)."""
    _bg, tgt, wall, player, pwall, crate, held, _s, _o = ids
    w = board.w
    p = h = -1
    mask = 0
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            i = r * w + c
            if player in cell:
                p = i
            if held in cell:
                h = i
            if crate in cell:
                mask |= 1 << i
            # scenery must be exactly what the board says it is
            assert (wall in cell) == (i in board.wall), (r, c, "wall")
            assert (pwall in cell) == (i in board.pwall), (r, c, "playerwall")
            assert (tgt in cell) == (i in board.targets), (r, c, "target")
    return (p, h, mask)


def _selfcheck(exhaustive_cap: int = 2_500, walks: int = 400,
               randoms: int = 150, bfs_cap: int = 250_000,
               probes: int = 12, verbose: bool = True) -> int:
    """Five passes. Nothing here trusts the file's own reasoning: every claim
    is measured against the interpreter or against a second derivation.

    1. **The turn function, exhaustively**, on every level whose entire
       reachable space fits ``exhaustive_cap``: each state stepped through all
       five presses on both the model and the real interpreter, whole board and
       win flag compared. This is the strongest form of the check -- there is
       no state of those levels the model has not been asked about.
    2. **The turn function, sampled**, on every level: states from random walks
       (reachable) and synthetic boards with the crates thrown anywhere
       (mostly not), the same comparison.
    3. **Shortest, twice over**: a plain breadth-first distance table with no
       heuristic at all, on every level whose space fits ``bfs_cap``, against
       A*'s answer and against the field's ``dist(start)``.
    4. **The tie sets, twice over**: on those same levels, the set the field
       reports at every step of the recorded plan against the set the
       breadth-first table gives.
    5. **Recovery**: perturb a level start with random presses, re-plan from
       where that lands, and replay the answer in the interpreter to a WIN --
       the `recovery_mode = "reset"` contract's other half, which is that
       `_search` answers from a LIVE board and not from a prefix.
    """
    _solver, game, expert = _new()
    eng = game._engine
    ids = _ids(game)
    rng = random.Random(20260821)
    bad = 0

    starts = {}
    for level in range(game.n_levels):
        game.set_level(level)
        starts[level] = expert.read(eng)

    # 1 + 2 -- the turn function
    checked = wrong = 0
    done = []
    for level in range(game.n_levels):
        board, start = starts[level]
        space = _reachable(board, start, exhaustive_cap)
        states = list(space) if space is not None else []
        if space is not None:
            done.append((level, len(space)))
        states += _walk_states(board, start, rng, walks)
        states += _random_boards(board, start, rng, randoms)
        n, w, first = _compare(eng, board, states, ids)
        checked += n
        wrong += w
        if w:
            bad += 1
            state, d, got, want = first
            print(f"  L{level}: {w} of {n} transitions disagree; first is "
                  f"'{d}' from\n{_ascii(board, state)}\n"
                  f"    the interpreter settled on\n{_ascii(board, got)}\n"
                  f"    the model said\n{_ascii(board, want)}")
    if verbose:
        print(f"  turn function: {checked} transitions compared against the "
              f"interpreter, {wrong} disagree")
        print("    exhaustive on " + (", ".join(
            f"L{lv} ({n} states, its whole space)" for lv, n in done)
            or "no level"))

    # 3 + 4 -- shortest and the tie sets, independently
    swept = []
    for level in range(game.n_levels):
        board, start = starts[level]
        table = _bfs_dist(board, start, bfs_cap)
        if table is None:
            continue
        game.set_level(level)
        plan = expert.plan(eng, level)
        field = expert.field_for(board)
        d0 = table.get(start)
        if plan is None or d0 != len(plan) or d0 != field.dist.get(start):
            print(f"  L{level}: BFS says {d0}, plan is "
                  f"{None if plan is None else len(plan)}, field says "
                  f"{field.dist.get(start)}")
            bad += 1
            continue
        # every step's set, against the BFS table
        state = start
        clashes = 0
        for i, taken in enumerate(plan):
            want = []
            for d in DIRS:
                nxt = board.step(state, d)
                if nxt == state:
                    continue
                cost = 1 if board.won(nxt) else (
                    _INF if nxt not in table else table[nxt] + 1)
                if cost == table[state]:
                    want.append(d)
            if want != plan.optsets[i]:
                clashes += 1
                if clashes == 1:
                    print(f"  L{level} step {i}: field says "
                          f"{plan.optsets[i]}, BFS says {want}")
            state = board.step(state, taken)
        bad += clashes
        swept.append((level, len(table), d0))
    if verbose:
        print(f"  shortest + tie sets re-derived by plain BFS on "
              f"{len(swept)} levels: " + ", ".join(
                  f"L{lv} ({n} winnable states, d*={d})" for lv, n, d in swept))

    # 5 -- recovery from a perturbed board
    solved = won = refused = 0
    for level in range(game.n_levels):
        board, start = starts[level]
        for _ in range(probes):
            game.set_level(level)
            state = start
            for _ in range(rng.randint(1, 8)):
                state = board.step(state, rng.choice(DIRS))
                if board.won(state):
                    break
            if board.won(state):
                continue
            _seat(eng, board, state, ids)
            plan = expert.plan(eng, level)
            if plan is None:
                refused += 1
                continue
            solved += 1
            _seat(eng, board, state, ids)
            for d in plan:
                eng.step(d)
                if eng.check_win():
                    break
            if eng.check_win():
                won += 1
            else:
                bad += 1
                print(f"  L{level}: a recovery plan did not win")
    if verbose:
        print(f"  recovery: {solved} perturbed boards re-planned "
              f"({won} of them won in the interpreter), {refused} refused")
    return bad


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

#: Every cell stack a board of this game can hold.
#:
#: Background is on its own layer and is everywhere. A Wall square holds
#: nothing else -- no legend character places anything under one, and nothing
#: can enter one -- but it always carries the Sensor / On marker, which is
#: Transparent and so cannot be seen; the audit renders both to say so.
#: PlayerWall shares no square with Wall or Target (again, no legend character
#: does) and never with the Player (the cancel rule forbids it), but a crate --
#: held or dropped -- crosses one freely and can be left standing on it.
_COMPOSITIONS = [
    (),                                        # bare floor
    ("target",),
    ("playerwall",),
    ("wall", "sensor"),
    ("wall", "on"),
    ("player",),
    ("player", "target"),
    ("crate",),
    ("crate", "target"),                       # the win composition
    ("crate", "playerwall"),
    ("pickedup",),
    ("pickedup", "target"),
    ("pickedup", "playerwall"),
]

#: `_render_frame`'s letterbox colour. A composition that renders as a uniform
#: frame of it would have no seam against the pad (the ps:stand_iii lesson).
_PAD = 5


def _comp_name(comp) -> str:
    return "+".join(comp) or "floor"


def _audit(verbose: bool = True) -> int:
    """Two passes over every reachable cell composition, at every board shape
    the ten levels use.

    **Pass 1 -- distinctness.** Whole 64x64 frames of uniform boards are
    compared rather than one cell cut out of a mixed board: `_render_frame`
    upscales and centre-pads, so slicing a cell by ``cell_px`` arithmetic reads
    the wrong pixels (the ps:explod lesson). Two uniform boards render
    identically iff their cells do.

    This is the pass that fails on the shipped .txt, in the same place at all
    eight board shapes: ``player == player+target``. See the header comment in
    ``data/puzzlescript_games/Tractor_Beam_Sokoban9.txt``.

    **Pass 2 -- nothing is the letterbox.**

    The quarter-turn orbit that ps:tornado_tamer and ps:stand_off have to check
    has no counterpart here: this game owns no directional art at all. Every
    sprite it draws is a fixed picture of a fixed object, so a turned frame can
    only ever show turned copies of pictures the game already owns.
    """
    from adapters.puzzlescript_adapter import _render_frame            # noqa: PLC0415

    _solver, game, _expert = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    def shoot(h, w, comp):
        eng.height, eng.width = h, w
        cell = {idx["background"]} | {idx[o] for o in comp}
        eng.grid = [[set(cell) for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        return np.asarray(_render_frame(eng, g)).copy()

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {c: shoot(h, w, c) for c in _COMPOSITIONS}
        clashes = [(a, b) for a, b in itertools.combinations(_COMPOSITIONS, 2)
                   if np.array_equal(shots[a], shots[b])
                   and {a, b} != {("wall", "sensor"), ("wall", "on")}]
        pad = [c for c in _COMPOSITIONS if np.all(shots[c] == _PAD)]
        bad += len(clashes) + len(pad)
        if verbose:
            print(f"  {h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): "
                  f"{len(_COMPOSITIONS)} compositions, "
                  f"{'OK' if not clashes and not pad else 'CLASH'}")
        for a, b in clashes:
            print(f"      IDENTICAL: {_comp_name(a)}  ==  {_comp_name(b)}")
        for c in pad:
            print(f"      INVISIBLE AGAINST THE LETTERBOX: {_comp_name(c)}")
    if verbose:
        print("  (wall+sensor == wall+on is the one deliberate identity: both "
              "markers are Transparent, and they say nothing the dark blue "
              "crate on the screen does not already say)")
    return bad


def _symmetry(verbose: bool = True) -> int:
    """Replay every level's plan through the ADAPTER at every presentation it
    can draw, and assert the frames are the transform of the unaugmented ones.

    ps: games are augmented by the ADAPTER, not by `utils/arc_game.py`: it
    turns the presented frame by a quarter turn per (seed, level) and
    forward-remaps directional input to match, so a generator that plans in
    ENGINE space has to emit the SCREEN press (`screen_action`). This drives
    the plans through `perform_action` exactly as `record_level` does, at every
    rotation the adapter produces, and compares the taped frames against the
    k=0 run turned by hand. It is the check that the rotation contract is the
    right way round; get it backwards and nothing raises, the recording is just
    wrong on three presentations out of four.

    It also checks the thing this particular game needs the rotation to be:
    IDENTIFIABLE. The beam's scan order is UP, DOWN, LEFT, RIGHT in engine
    space, so a quarter turn presents a board whose scan order on SCREEN is a
    different one -- which is a consistent relabelling of the game and nothing
    worse, provided a frame says which turn it is looking at. It does: the
    forty (level, turn) start frames this walks are pairwise distinct, so the
    presentation is readable off the very first observation of an episode and
    the whole trajectory after it is one game.

    This game is NOT in `PuzzleScriptAdapter._FLIP_GAMES` and should not be:
    the scan order is CHIRAL. Under a mirror, UP-DOWN-LEFT-RIGHT becomes
    UP-DOWN-RIGHT-LEFT, which the interpreter will not play -- and unlike a
    quarter turn, which permutes the four rays cyclically, a mirror leaves the
    board's own shape untouched on one axis, so the two presentations of a
    mirror-symmetric board would be indistinguishable frames with different
    answers.
    """
    _solver, game, expert = _new()
    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        plans[level] = expert.plan(game._engine, level) or []

    # The adapter re-draws the presentation on every `set_level`, so a seed
    # does NOT pin one rotation for the whole episode -- each (seed, level)
    # gets its own. Read it after seating the level, never before.
    ref: dict = {}
    seen = set()
    wins = 0
    bad = 0
    for seed in range(24):
        g = TractorBeamSokoban9Solver().make_game(seed)
        for level, plan in plans.items():
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            frames = [np.asarray(g._current_frame)]
            for direction in plan:
                act = screen_action(direction, *k)
                fd = g.perform_action(ActionInput(id=act))
                frames.append(np.asarray(fd.frame[-1] if fd.frame
                                         else g._current_frame))
            if plan and g._state != GameState.WIN:
                print(f"  seed {seed} L{level} {k}: plan did not win")
                bad += 1
            else:
                wins += 1
            if (level, k) in ref:
                continue
            if k == (0, False, False):
                ref[(level, k)] = frames
                continue
            base = ref.get((level, (0, False, False)))
            if base is None:
                continue                  # nothing to compare against yet

            def turned(frame, kk=k):
                out = np.rot90(frame, k=kk[0])
                if kk[1]:
                    out = np.fliplr(out)
                if kk[2]:
                    out = np.flipud(out)
                return np.ascontiguousarray(out)

            ref[(level, k)] = frames
            if any(not np.array_equal(turned(a), b)
                   for a, b in zip(base, frames)):
                print(f"  seed {seed} L{level} {k}: frames are not the "
                      f"transform of the unaugmented ones")
                bad += 1

    # the presentation has to be readable off the first frame
    firsts = {key: frames[0].tobytes() for key, frames in ref.items()}
    if len(set(firsts.values())) != len(firsts):
        collisions = len(firsts) - len(set(firsts.values()))
        print(f"  {collisions} (level, presentation) pairs share a start "
              f"frame -- the scan order would be unreadable there")
        bad += collisions
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}; "
              f"{wins} plan replays won; {len(ref)} (level, presentation) "
              f"pairs compared, all with distinct start frames")
    return bad


if __name__ == "__main__":
    if "--plans" in sys.argv:
        failures = _plans()
        print(f"plans: {failures} failures")
        sys.exit(1 if failures else 0)
    if "--selfcheck" in sys.argv:
        violations = _selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = _audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = _symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    sys.exit(TractorBeamSokoban9Solver.main())
