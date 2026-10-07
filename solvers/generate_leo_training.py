"""Generate Phase-1 training data for the PuzzleScript game ps:leo
("Leo", Phillip Abram).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
its disk cache, and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: a native model of the mechanic (differentially fuzzed
against the interpreter), the planner that searches it, the optimal-action sets
it hands out, and the sprite work the shipped file needed.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_leo",
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
Every expert step also carries the full set of equally-optimal presses.

The game
--------
Leo is a lion cub with two chores on every one of the 13 levels, and the win
condition is both of them at once (``All Water on Elephant`` and ``No Poacher``):

  * **Every water tile must end up under an elephant.** Elephants are pushed,
    sokoban-style, and the counts are always exactly equal, so this half is a
    perfect matching of |E| crates onto |E| targets. ``[ > Elephant | Water ]``
    is a sound-only rule -- water does not stop anything, it is scenery that the
    win condition reads.
  * **Every poacher must be annihilated against an OPPOSITE one.** Poachers are
    pushed exactly like elephants, and ``[ > PoacherLeft | PoacherRight] -> [|]``
    (plus the three mirrored copies) deletes BOTH when one is shoved into its
    opposite. The rule is expanded over all four directions, so which way you
    push is irrelevant -- only the two facings matter, and they pair {L,R} and
    {U,D}.
  * **Arrows re-face a poacher, and are consumed doing it.** ``[ > Poacher |
    ArrowDown] -> [ | PoacherDown]`` moves the poacher onto the arrow's cell,
    changes its facing to the arrow's, and destroys the arrow. Nothing else can
    ever enter an arrow's cell -- arrows share the collision layer with the
    player, the walls and every piece -- so an arrow is also a DOOR that opens
    exactly once, and on levels 6 and 8 opening it is the only route out of a
    region (see ``_Guide`` and ``arrow_plan`` below).

Three mechanical facts that decide how this file is written, all measured with
``--fuzz`` rather than read off the rule listing:

  * **A blocked press is a total no-op.** Nothing ticks, no piece moves, the
    player does not even step. So the only state-changing press is a PUSH and
    every plan is a sequence of ``walk to a stance, push``: the planner searches
    MACROS and the walks are shortest paths found by one BFS.
  * **Push chains do not exist.** No rule gives a pushed piece the power to push
    the next one, so an elephant behind an elephant is a wall. Same for a poacher
    behind anything that is not its opposite.
  * **Water, black and cages do not block.** They are on the layer *below* every
    piece, so the player, elephants and poachers all walk over water freely and
    an elephant can be pushed straight across a water tile -- which is exactly
    what level 1 needs (its three waters are a row, and the far one has to be
    pushed THROUGH the other two).

Coverage
--------
**10 of the 13 levels** (0-7, 10, 12; 1095 presses in total), every plan
certified by replaying it on the interpreter and every press carrying an
optimal-action set. Levels 8, 9 and 11 are not solved -- see
`LeoSolver.skip_levels` for what defeats the search there. They are dropped by
`discover_solvable`, so an episode is the ten that do win, with ``level_id``
preserving the true provenance.

The planner
-----------
The state is (player, elephants, poachers+facings, arrows+facings) and the big
levels carry 8 elephants and 10 poachers, so there is no exact field here and no
useful admissible bound either: the elephant term moves by 1 per push while the
walk that set the push up costs 4, so a flat A* over presses degenerates into
Dijkstra and drowns (measured: level 5 at 400k macro nodes was still at h=29 of
46). What works is a two-level search:

  * the **inner** searches stop at the first states to score a SUBGOAL -- this
    water covered, two fewer poachers, any progress at all -- and return the
    cheapest few of each. Each runs under the SUBGOAL's own estimate rather than
    the global one, which is what puts the long routes (the poacher endgame is
    fifteen pushes and an arrow) in reach;
  * the **outer** search is a best-first over the order of those subgoals, ranked
    by presses spent plus the guide's estimate of what is left.

`_Guide` supplies both the estimate and the death tests that make the outer
search converge. Two of them are the game's own structure:

  * **the arrow economy** (`arrow_plan`). Every annihilation pairs a facing with
    its opposite, so the surviving multiset must be (p, p, q, q) over
    (L, R, U, D), and a facing a poacher does not already have costs one arrow of
    that facing. Spending the last ArrowRight on the wrong poacher is therefore a
    dead end that is detectable from four counts, and on the arrow-tight levels
    (6 has exactly two ArrowDown for four PoacherUp) it is most of the puzzle.
  * **the parked-elephant test** (`Guide.eleph_strict`). An elephant that stays
    on its water is a wall, and that is what makes an ORDER wrong rather than a
    move: on level 4, water (4,1) can only be entered by a push whose stance is
    water (6,1), so filling (6,1) first loses the level. The same test is unsound
    as a hard prune, because an elephant crossing a water tile in transit looks
    identical to a parked one -- so the ladder runs it BOTH ways (`strict`), and
    the levels split cleanly: 1 and 2 need the transit, 4 needs the prune.

A third piece is not a test but an aim: a poacher that no press can ever move
(`Guide.immobile` -- walls on both sides of every axis) can only be a TARGET, so
`_successors` gives each one a search of its own. A heuristic that ranks pairs by
cheapness never looks at the far one, and level 8's whole endgame is a poacher
walked the length of the board into a frozen corner pocket.

Plans are certified by replaying them on the real interpreter (``--plans``); the
model itself is checked against it by a differential fuzz from COLD level loads
(``--fuzz``: 520k transitions, 0 divergences, with a coverage counter per
mechanic so a clean run cannot mean "never exercised it").

Optimal-action sets
-------------------
A push press is labelled with itself. Each press of a WALK is labelled with every
direction that (a) moves nothing but the player and (b) stays on a shortest route
to the same stance -- no rule fires on a bare move, so every interleaving of the
two axes leaves an identical board and labelling one of them as "the" answer
would train a coin flip. `--verify` re-derives every labelled press by replaying
the variant it names through the interpreter.

The shipped art had to be fixed (`data/puzzlescript_games/Leo.txt`)
-------------------------------------------------------------------
The four poacher sprites were mirror images of each other, not ROTATIONS: at the
board rotation the adapter applies per (seed, level), ``np.rot90(PoacherUp)``
matched none of the game's own four pictures, so "the arrow pointing this way
makes a poacher facing that way" -- the central mechanic of six levels -- was
only readable at k=0. They are now a hollow ring with a red bar on the leading
edge, an exact orbit of the rotation group -- the same orbit the ARROWS already
formed, which is how the handedness was fixed, and ``--orbit`` asserts it on
RENDERED frames rather than on the sprite source. That is what makes
``--symmetry``'s facing relabel a statement about the picture and not only about
the rules. The
arrows were also thickened, because at ``cell_px = 4`` -- levels 7, 9 and 11, i.e.
the arrow-heavy ones -- the sampler drops the middle row and column and the old
arrow rendered as two pixels. See [[ps-engine-render-gotchas]] and
[[ps-palette-collisions]]; ``--audit`` renders every cell composition the game
can build at every cell size its levels use and asserts pairwise distinctness.

Run:
    python solvers/generate_leo_training.py --episodes 10 --out data/leo
    python solvers/generate_leo_training.py --plans     # certify every plan
    python solvers/generate_leo_training.py --verify    # re-derive every label
    python solvers/generate_leo_training.py --fuzz      # model vs interpreter
    python solvers/generate_leo_training.py --symmetry  # rotation equivariance
    python solvers/generate_leo_training.py --orbit     # the sprite orbit
    python solvers/generate_leo_training.py --audit     # rendering
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame                # noqa: E402
from solvers.common.ps_astar import (                                  # noqa: E402
    Plan, PSAStarSolver, PSExpert,
)

_LEVEL_SOLVER = None            # set by `_audit` so `_compositions` can skip

GAME_NAME = "Leo"
PLAN_CACHE = Path(__file__).resolve().parent.parent / "data" / "leo_plans.json"

#: Engine direction -> (dr, dc). The game declares ``noaction``, so these four
#: presses are the entire vocabulary.
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
_ORDER = ("up", "down", "left", "right")

#: Facing pairs that annihilate. The rules are expanded over all four directions,
#: so the push direction never matters -- only these two couples do.
_OPP = {"l": "r", "r": "l", "u": "d", "d": "u"}
_FACINGS = ("l", "r", "u", "d")

_POACHER = {"poacherleft": "l", "poacherright": "r",
            "poacherup": "u", "poacherdown": "d"}
_ARROW = {"arrowleft": "l", "arrowright": "r",
          "arrowup": "u", "arrowdown": "d"}

#: A number no real distance can reach, used instead of ``inf`` so the matching
#: routines stay in integer arithmetic.
BIG = 10 ** 6


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class _Board:
    """The whole mechanic of ps:leo, verified against the interpreter by
    ``--fuzz`` (520k transitions from cold level loads, 0 divergences, every
    branch below covered -- annihilations, arrow conversions, elephant and
    poacher pushes, refusals).

    Walls and arrows block everything; water, black and cages block nothing
    (they are on the layer under the pieces). The player, elephants and poachers
    share one layer, so anything of theirs blocks anything else of theirs.
    """

    __slots__ = ("H", "W", "walls", "water", "player", "eleph", "poach", "arrow")

    def __init__(self, H, W, walls, water, player, eleph, poach, arrow):
        self.H, self.W = H, W
        self.walls, self.water = walls, water          # immutable, shared
        self.player, self.eleph = player, eleph
        self.poach, self.arrow = poach, arrow

    def copy(self):
        return _Board(self.H, self.W, self.walls, self.water, self.player,
                      set(self.eleph), dict(self.poach), dict(self.arrow))

    def step(self, d) -> bool:
        """Apply one press. Returns whether anything moved; a refused press is a
        total no-op (nothing on the board ticks), which is what makes the macro
        abstraction exact."""
        dr, dc = _DELTA[d]
        r, c = self.player
        t = (r + dr, c + dc)
        if not (0 <= t[0] < self.H and 0 <= t[1] < self.W):
            return False
        if t in self.walls or t in self.arrow:
            return False
        t2 = (t[0] + dr, t[1] + dc)
        inb = 0 <= t2[0] < self.H and 0 <= t2[1] < self.W
        if t in self.eleph:
            # No chain rule exists, so anything at all behind the elephant stops
            # the whole press -- the player is on the elephant's layer too.
            if (not inb or t2 in self.walls or t2 in self.arrow
                    or t2 in self.eleph or t2 in self.poach):
                return False
            self.eleph.discard(t)
            self.eleph.add(t2)
            self.player = t
            return True
        if t in self.poach:
            facing = self.poach[t]
            if not inb or t2 in self.walls or t2 in self.eleph:
                return False
            if t2 in self.poach:
                if self.poach[t2] == _OPP[facing]:     # both are deleted
                    del self.poach[t]
                    del self.poach[t2]
                    self.player = t
                    return True
                return False
            if t2 in self.arrow:                       # re-face, arrow consumed
                self.poach[t2] = self.arrow.pop(t2)
                del self.poach[t]
                self.player = t
                return True
            self.poach[t2] = self.poach.pop(t)
            self.player = t
            return True
        self.player = t
        return True

    def won(self) -> bool:
        return not self.poach and all(w in self.eleph for w in self.water)

    def key(self):
        return (self.player, frozenset(self.eleph),
                frozenset(self.poach.items()), frozenset(self.arrow.items()))

    def covered(self) -> int:
        return sum(1 for w in self.water if w in self.eleph)


def board_from_engine(eng, idx_to_name) -> _Board:
    """Read the interpreter's grid into a `_Board`."""
    H, W = len(eng.grid), len(eng.grid[0])
    walls, water, eleph = set(), set(), set()
    poach, arrow = {}, {}
    player = None
    for r in range(H):
        for c in range(W):
            for o in eng.grid[r][c]:
                n = idx_to_name[o]
                if n == "wall":
                    walls.add((r, c))
                elif n == "water":
                    water.add((r, c))
                elif n == "elephant":
                    eleph.add((r, c))
                elif n == "player":
                    player = (r, c)
                elif n in _POACHER:
                    poach[(r, c)] = _POACHER[n]
                elif n in _ARROW:
                    arrow[(r, c)] = _ARROW[n]
    return _Board(H, W, frozenset(walls), frozenset(water), player,
                  eleph, poach, arrow)


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------

def _walk_dists(b: _Board) -> dict:
    """Presses from the player to every cell it can reach without pushing."""
    blocked = b.walls | set(b.arrow) | b.eleph | set(b.poach)
    dist = {b.player: 0}
    queue = deque([b.player])
    while queue:
        cur = queue.popleft()
        nd = dist[cur] + 1
        for dr, dc in _DELTA.values():
            n = (cur[0] + dr, cur[1] + dc)
            if (0 <= n[0] < b.H and 0 <= n[1] < b.W
                    and n not in blocked and n not in dist):
                dist[n] = nd
                queue.append(n)
    return dist


def _macros(b: _Board):
    """``(presses, stance, direction, board)`` for every walk-to-and-push.

    A press that pushes nothing changes only the player's cell and no rule sees
    it, so every shortest primitive plan IS a sequence of these and a macro edge
    costs ``walk + 1``.
    """
    dist = _walk_dists(b)
    out = []
    for q in itertools.chain(b.eleph, b.poach):
        for d in _ORDER:
            dr, dc = _DELTA[d]
            stance = (q[0] - dr, q[1] - dc)
            walk = dist.get(stance)
            if walk is None:
                continue
            nb = b.copy()
            nb.player = stance
            if not nb.step(d):
                continue
            out.append((walk + 1, stance, d, nb))
    return out


def _push_table(H, W, blk, target) -> dict:
    """Min pushes to bring a piece from each cell to ``target``, ignoring every
    other piece. A push from ``prev`` needs both ``prev`` and the stance behind
    it clear, which is what makes corners and wall-lines unreachable and gives
    the deadlock test for free."""
    def ok(p):
        return 0 <= p[0] < H and 0 <= p[1] < W and p not in blk

    dist = {target: 0}
    queue = deque([target])
    while queue:
        cur = queue.popleft()
        nd = dist[cur] + 1
        for dr, dc in _DELTA.values():
            prev = (cur[0] - dr, cur[1] - dc)
            stance = (prev[0] - dr, prev[1] - dc)
            if prev not in dist and ok(prev) and ok(stance):
                dist[prev] = nd
                queue.append(prev)
    return dist


def _free_dists(H, W, blk) -> dict:
    """All-pairs walk distances over the cells walls leave open."""
    cells = [(r, c) for r in range(H) for c in range(W) if (r, c) not in blk]
    out = {}
    for s in cells:
        dist = {s: 0}
        queue = deque([s])
        while queue:
            cur = queue.popleft()
            nd = dist[cur] + 1
            for dr, dc in _DELTA.values():
                n = (cur[0] + dr, cur[1] + dc)
                if n not in dist and 0 <= n[0] < H and 0 <= n[1] < W and n not in blk:
                    dist[n] = nd
                    queue.append(n)
        out[s] = dist
    return out


def _hungarian(cost) -> int:
    """Min-cost perfect assignment (Jonker-Volgenant), n <= 8 here."""
    n = len(cost)
    inf = float("inf")
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return sum(cost[p[j] - 1][j - 1] for j in range(1, n + 1))


def _match_pairs(n, cost) -> int:
    """Min-weight perfect matching on <= 10 nodes, by bitmask DP."""
    if n == 0:
        return 0
    if n % 2:
        return BIG
    full = (1 << n) - 1
    memo = {}

    def go(mask):
        if mask == full:
            return 0
        r = memo.get(mask)
        if r is not None:
            return r
        i = 0
        while mask >> i & 1:
            i += 1
        best = BIG
        for j in range(i + 1, n):
            if mask >> j & 1:
                continue
            c = cost[i][j]
            if c >= BIG:
                continue
            v = c + go(mask | (1 << i) | (1 << j))
            if v < best:
                best = v
        memo[mask] = best
        return best

    return go(0)


def arrow_plan(poach, arrow):
    """``(feasible, min conversions)`` for the poacher/arrow COUNTS alone.

    Every annihilation pairs a facing with its opposite, so the surviving
    multiset must be ``(p, p, q, q)`` over (L, R, U, D) for some ``p + q = N/2``.
    A poacher can only acquire a facing it does not already have by being pushed
    onto an ARROW of that facing, and each arrow is consumed, so a target split
    costs ``max(0, F[t] - n[t])`` arrows of type t.

    That is the whole arrow economy, from four counts, and it is what turns "you
    spent the last ArrowRight on the wrong poacher" into a dead end the search
    can see. On level 6 -- four PoacherUp, four PoacherLeft, two ArrowDown, two
    ArrowRight -- it pins the split to exactly p = q = 2.
    """
    n = dict.fromkeys(_FACINGS, 0)
    for t in poach.values():
        n[t] += 1
    have = dict.fromkeys(_FACINGS, 0)
    for t in arrow.values():
        have[t] += 1
    total = len(poach)
    if total == 0:
        return True, 0
    if total % 2:
        return False, BIG
    best = BIG
    for p in range(total // 2 + 1):
        q = total // 2 - p
        want = {"l": p, "r": p, "u": q, "d": q}
        need = 0
        for t in _FACINGS:
            short = want[t] - n[t]
            if short > 0:
                if short > have[t]:
                    need = BIG
                    break
                need += short
        if need < best:
            best = need
    return best < BIG, best


class _Guide:
    """Estimate of the presses left, plus the two death tests.

    The two halves of the win condition are estimated independently and added:
    a min-cost assignment of elephants to waters priced by per-water reverse
    push tables, and a min-weight perfect matching of poachers priced by how far
    apart an annihilating pair is (routed through an ARROW when their facings do
    not already oppose). Neither is admissible and neither needs to be -- the
    outer beam ranks with it, it does not prove anything with it.
    """

    #: What ranking charges a state whose elephant ORDER has already made a water
    #: unfillable. Large enough that the beam takes any untrapped alternative
    #: first, finite so that a level with no alternative can still proceed.
    TRAP = 200

    def __init__(self, b: _Board, strict: bool):
        self.H, self.W = b.H, b.W
        self.walls = b.walls
        self.waters = sorted(b.water)
        self.strict = strict
        self.free = _free_dists(b.H, b.W, b.walls)
        self.base = {w: _push_table(b.H, b.W, b.walls, w) for w in self.waters}
        self._tables = {frozenset(): self.base}
        self._ptab = {}
        self._pairs = {}
        self._immobile = {}
        self._eleph = {}
        self._eleph_strict = {}
        self._poach = {}

    # -- elephants ------------------------------------------------------------
    def tables(self, fixed):
        r = self._tables.get(fixed)
        if r is None:
            blk = self.walls | set(fixed)
            r = {w: _push_table(self.H, self.W, blk, w)
                 for w in self.waters if w not in fixed}
            self._tables[fixed] = r
        return r

    @staticmethod
    def _assign(free, tabs, wat) -> int:
        if not wat:
            return 0
        return _hungarian([[tabs[w].get(e, BIG) for w in wat] for e in free])

    def eleph(self, b) -> int:
        """Relaxed: walls are the only obstacle, so this never calls a state dead
        for standing somewhere it is merely passing through."""
        k = frozenset(b.eleph)
        r = self._eleph.get(k)
        if r is None:
            r = self._assign(sorted(b.eleph), self.base, self.waters)
            self._eleph[k] = r
        return r

    def eleph_strict(self, b) -> int:
        """With the already-covered waters treated as PARKED, i.e. as walls."""
        k = frozenset(b.eleph)
        r = self._eleph_strict.get(k)
        if r is None:
            fixed = frozenset(e for e in b.eleph if e in b.water)
            tabs = self.tables(fixed)
            r = self._assign(sorted(e for e in b.eleph if e not in fixed),
                             tabs, sorted(tabs))
            self._eleph_strict[k] = r
        return r

    # -- poachers -------------------------------------------------------------
    def ptab(self, cell) -> dict:
        """Min pushes to bring a piece from each cell TO ``cell``, walls only.

        Poacher distances have to be PUSH distances, not walk distances: a
        poacher with a wall behind it cannot be pushed at all (level 4's
        PoacherRight at (1,8) is walled on three sides and can only ever be a
        TARGET), and a walk-distance guide happily reports such a pair as one
        move from annihilating for ever.
        """
        r = self._ptab.get(cell)
        if r is None:
            r = _push_table(self.H, self.W, self.walls, cell)
            self._ptab[cell] = r
        return r

    def immobile(self, cell) -> bool:
        """True when NO press can ever move a poacher out of ``cell``.

        Walls alone decide it, so it is a property of the level. Such a poacher
        can only ever be a TARGET -- something has to be shoved into it -- and
        `_successors` gives each one its own search, because a heuristic that
        ranks pairs by cheapness will never aim at the far one. Level 8 ships a
        four-poacher pocket in the top-right corner, two of whose members are
        immobile, and it is the whole endgame of that level.
        """
        r = self._immobile.get(cell)
        if r is None:
            r = True
            for dr, dc in _DELTA.values():
                stance = (cell[0] - dr, cell[1] - dc)
                dest = (cell[0] + dr, cell[1] + dc)
                if (0 <= stance[0] < self.H and 0 <= stance[1] < self.W
                        and 0 <= dest[0] < self.H and 0 <= dest[1] < self.W
                        and stance not in self.walls and dest not in self.walls):
                    r = False
                    break
            self._immobile[cell] = r
        return r

    def target_cost(self, b, target) -> int:
        """Cheapest way to annihilate the poacher standing on ``target``."""
        if target not in b.poach:
            return BIG
        ps, cost = self.pair_matrix(b)
        j = ps.index(target)
        return min((cost[i][j] for i in range(len(ps)) if i != j), default=BIG)

    def _meet(self, mover, target) -> int:
        """Pushes to shove ``mover`` into ``target``'s cell: reach one of its
        neighbours, then one more push."""
        best = BIG
        for dr, dc in _DELTA.values():
            n = (target[0] + dr, target[1] + dc)
            if n in self.walls:
                continue
            d = self.ptab(n).get(mover, BIG)
            if d < best:
                best = d
        return best + 1 if best < BIG else BIG

    def pair_matrix(self, b):
        """``(cells, cost)`` -- what it costs to annihilate each pair of
        poachers, alone on the board. ``BIG`` means never: not from either side,
        and not through any arrow, so a matching that cannot avoid it is a state
        no continuation wins from."""
        k = (frozenset(b.poach.items()), frozenset(b.arrow.items()))
        r = self._pairs.get(k)
        if r is not None:
            return r
        ps = sorted(b.poach)
        n = len(ps)
        arrows = sorted(b.arrow.items())
        cost = [[BIG] * n for _ in range(n)]
        for i in range(n):
            ti = b.poach[ps[i]]
            for j in range(i + 1, n):
                tj = b.poach[ps[j]]
                c = BIG
                if tj == _OPP[ti]:                     # already opposable
                    c = min(self._meet(ps[i], ps[j]), self._meet(ps[j], ps[i]))
                for pos, facing in arrows:             # ONE of them re-faced
                    # pushing a poacher onto an arrow IS the re-facing, so the
                    # arrow's cell is where it ends up
                    if facing == _OPP[tj]:
                        x = self.ptab(pos).get(ps[i], BIG)
                        if x < BIG:
                            c = min(c, x + min(self._meet(pos, ps[j]),
                                               self._meet(ps[j], pos)))
                    if facing == _OPP[ti]:
                        x = self.ptab(pos).get(ps[j], BIG)
                        if x < BIG:
                            c = min(c, x + min(self._meet(pos, ps[i]),
                                               self._meet(ps[i], pos)))
                if c >= BIG:
                    # BOTH re-faced. Level 7 ships five PoacherLeft and no
                    # ArrowRight, so an L/L pair has no single-conversion route
                    # and this is the only way it ever dies.
                    for p1, f1 in arrows:
                        for p2, f2 in arrows:
                            if p1 == p2 or f2 != _OPP[f1]:
                                continue
                            x = self.ptab(p1).get(ps[i], BIG)
                            y = self.ptab(p2).get(ps[j], BIG)
                            if x < BIG and y < BIG:
                                c = min(c, x + y + min(self._meet(p1, p2),
                                                       self._meet(p2, p1)))
                cost[i][j] = cost[j][i] = min(c, BIG)
        r = (ps, cost)
        self._pairs[k] = r
        return r

    def poach(self, b) -> int:
        k = (frozenset(b.poach.items()), frozenset(b.arrow.items()))
        r = self._poach.get(k)
        if r is not None:
            return r
        if not b.poach:
            self._poach[k] = 0
            return 0
        if not arrow_plan(b.poach, b.arrow)[0]:
            self._poach[k] = BIG
            return BIG
        ps, cost = self.pair_matrix(b)
        r = _match_pairs(len(ps), cost)
        self._poach[k] = r
        return r

    # -- the two things the search asks ---------------------------------------
    def h(self, b):
        """None when the state is provably lost."""
        e = self.eleph(b)
        if e >= BIG:
            return None
        if self.strict and self.eleph_strict(b) >= BIG:
            return None
        p = self.poach(b)
        if p >= BIG:
            return None
        return e + p

    def rank(self, b):
        hh = self.h(b)
        if hh is None:
            return None
        if not self.strict and self.eleph_strict(b) >= BIG:
            return hh + self.TRAP
        return hh


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------

def _sub_search(b, guide, goal, h, sig, want, cap, deadline, weight=3):
    """Cheapest routes to ``want`` distinct boards satisfying ``goal``.

    A goal state is a LEAF -- the outer search decides what to do next. ``h`` is
    the SUBGOAL's own estimate, not the global one: a global heuristic is flat
    between events (an elephant push moves it by 1 while the walk that set the
    push up costs 4), so ordering by it degenerates to Dijkstra and the long
    routes -- the poacher endgame, where the last pair is fifteen pushes and an
    arrow apart -- are simply out of reach. A per-subgoal estimate falls with
    every correct push.

    ``sig`` is what the returned states are deduplicated on, and it is NOT the
    board: dedup by board key fills the whole quota with the same annihilation
    seen from eight different player cells, which is one branch dressed as eight.
    Keyed on what the subgoal is ABOUT -- which poachers are left, where the
    elephants are -- the quota buys genuinely different next moves.
    """
    out, seen_out = [], set()
    queue = [(weight * h(b), 0, 0, b, [])]
    best = {b.key(): 0}
    counter = gen = 0
    while queue and len(out) < want:
        _f, g, _c, cur, path = heapq.heappop(queue)
        for cost, stance, d, nb in _macros(cur):
            gen += 1
            ng = g + cost
            k = nb.key()
            if best.get(k, 1 << 30) <= ng:
                continue
            best[k] = ng
            if guide.h(nb) is None:
                continue
            if goal(nb):
                sg = sig(nb)
                if sg not in seen_out:
                    seen_out.add(sg)
                    out.append((ng, nb, path + [(stance, d)]))
                continue
            hh = h(nb)
            if hh >= BIG:
                continue
            counter += 1
            heapq.heappush(queue, (ng + weight * hh, ng, counter, nb,
                                   path + [(stance, d)]))
        if gen >= cap or (deadline and time.time() > deadline):
            break
    return out


def _pair_cost(guide, b) -> int:
    """The cheapest annihilating PAIR on the board -- the poacher subgoal's own
    estimate. Falls as any two opposable poachers are pushed together, or as one
    is pushed toward an arrow that would make them opposable."""
    if not b.poach:
        return 0
    _ps, cost = guide.pair_matrix(b)
    return min((c for row in cost for c in row), default=BIG)


def _kill_sig(b):
    """What makes one annihilation DIFFERENT from another, for dedup purposes:
    which two facings died and which arrows it spent -- not where the survivors
    ended up.

    Level 8 is why. Its cheapest kill converts a PoacherLeft on the ArrowRight
    and shoves it into the other PoacherLeft, and that is fatal: the surviving
    PoacherRight is walled into a pocket where only a PoacherLeft can reach it,
    and both Lefts are now gone. Dedup on the board (or even on the poacher
    positions) fills the whole quota with that one kill seen from thirty
    different survivor positions, so the search never looks for the other kind.
    """
    facings = {}
    for f in b.poach.values():
        facings[f] = facings.get(f, 0) + 1
    return (tuple(sorted(facings.items())), frozenset(b.arrow.items()))


def _score(b, poach_first: bool):
    """The monotone progress measure the unfocused search stops on."""
    if poach_first:
        return (-len(b.poach), b.covered())
    return (b.covered(), -len(b.poach))


def _successors(b, guide, poach_first, want, cap, deadline):
    """Every way of scoring the next subgoal, from two complementary searches.

    The FOCUSED ones (one per uncovered water, one for the poachers) chase a
    named subgoal down its own gradient and are what reach the long routes -- the
    poacher endgame, where the last pair is fifteen pushes and an arrow apart.
    They are blind to a subgoal that first needs an unrelated piece shoved out of
    the way, because shoving it does not move their estimate at all; the
    UNFOCUSED Dijkstra, which stops at any progress whatsoever, covers exactly
    that case (level 2 is unsolvable without it, level 6 without the focused
    ones). Running both and taking the union costs one extra search per node and
    is the difference between 6 levels and 13.
    """
    out = []
    if b.poach:
        n0 = len(b.poach)
        out += _sub_search(b, guide,
                           lambda bd, n0=n0: len(bd.poach) < n0,
                           lambda bd: _pair_cost(guide, bd),
                           _kill_sig,
                           want * 2, cap * 2, deadline)
        for t in sorted(b.poach):
            if not guide.immobile(t):
                continue
            out += _sub_search(b, guide,
                               lambda bd, n0=n0: len(bd.poach) < n0,
                               lambda bd, t=t: guide.target_cost(bd, t),
                               _kill_sig, want, cap, deadline)
    if not (poach_first and b.poach):
        for w in guide.waters:
            if w in b.eleph:
                continue
            tab = guide.base[w]
            out += _sub_search(
                b, guide,
                lambda bd, w=w: w in bd.eleph,
                lambda bd, tab=tab: min((tab.get(e, BIG) for e in bd.eleph),
                                        default=BIG),
                lambda bd: frozenset(bd.eleph),
                want, cap, deadline)
    base = _score(b, poach_first)
    out += _sub_search(b, guide,
                       lambda bd, base=base: _score(bd, poach_first) > base,
                       lambda _bd: 0,
                       lambda bd: (frozenset(bd.eleph),
                                   frozenset(bd.poach.items()),
                                   frozenset(bd.arrow.items())),
                       want, cap, deadline)
    return out


def _plan_macros(b, *, strict, poach_first, expansions, want, cap, tcap):
    """Best-first over the ORDER of the subgoals. Returns ``(macros, presses)``.

    Best-first rather than a layered beam, because a layer of successors throws
    the states themselves away: a good state whose own successors are all worse
    drops out of the search, and the beam then oscillates between it and its
    parent for ever (measured on level 2, where the trap is an elephant parked
    where the PLAYER can no longer get behind it -- something no push-distance
    table can see). A priority queue simply expands the trap, gets nothing, and
    carries on with the next best state.
    """
    guide = _Guide(b, strict)
    if guide.h(b) is None:
        return None
    deadline = time.time() + tcap if tcap else None
    queue = [(0, 0, 0, b, [])]
    seen = {b.key()}
    counter = 0
    for _ in range(expansions):
        if not queue or (deadline and time.time() > deadline):
            return None
        _f, g, _c, cur, path = heapq.heappop(queue)
        for ng, nb, npath in _successors(cur, guide, poach_first,
                                         want, cap, deadline):
            if nb.won():
                return path + npath, g + ng
            k = nb.key()
            if k in seen:
                continue
            rk = guide.rank(nb)
            if rk is None:
                continue
            seen.add(k)
            counter += 1
            heapq.heappush(queue, (g + ng + 6 * rk, g + ng, counter, nb,
                                   path + npath))
    return None


#: The ladder, in TIERS: every variant of a tier is run and the shortest plan
#: wins, then the next tier is only reached if the whole of this one failed.
#: ``strict`` decides whether an elephant standing on a water tile is treated as
#: parked (a prune) or as possibly in transit (a ranking penalty), and the levels
#: genuinely split on it -- 1 and 2 need the transit, 4 needs the prune -- so
#: both are tried rather than one being chosen. ``poach_first`` suppresses the
#: per-water searches while poachers remain, which is what the arrow-tight
#: levels need and what the pure-sokoban ones must not have.
_LADDER = tuple(
    tuple((strict, poach_first, expansions, want, cap, tcap)
          for strict in (False, True)
          for poach_first in (False, True))
    for expansions, want, cap, tcap in ((150, 8, 60_000, 60),
                                        (800, 8, 60_000, 240),
                                        (3000, 12, 150_000, 480))
)


def _replay(b, macros):
    """Run a macro list on a fresh copy; returns the board or None if it breaks."""
    cur = b.copy()
    for stance, d in macros:
        dist = _walk_dists(cur)
        if stance not in dist:
            return None
        cur.player = stance
        if not cur.step(d):
            return None
    return cur


def _cost(b, macros) -> int:
    """Presses a macro list costs, or BIG if it does not replay to a win."""
    cur = b.copy()
    total = 0
    for stance, d in macros:
        dist = _walk_dists(cur)
        walk = dist.get(stance)
        if walk is None:
            return BIG
        cur.player = stance
        if not cur.step(d):
            return BIG
        total += walk + 1
    return total if cur.won() else BIG


def _shorten(b, macros):
    """Cut engine-verified blocks out of a macro list, then re-order what is
    left, keeping anything that still replays to a win and costs fewer presses.

    The beam commits to a subgoal order and never revisits it, so its plans carry
    whole detours the rest of the plan turns out not to need, and it interleaves
    subgoals in whatever order it happened to score them -- which is what costs
    the walks. Both are cheap to repair afterwards: the model replays a plan in
    microseconds, and a shorter list that still wins IS a better plan.
    """
    def cut(ms):
        ms = list(ms)
        changed = True
        while changed:                               # deletions, longest first
            changed = False
            for span in range(len(ms), 0, -1):
                for i in range(len(ms) - span + 1):
                    trial = ms[:i] + ms[i + span:]
                    end = _replay(b, trial)
                    if end is not None and end.won():
                        ms = trial
                        changed = True
                        break
                if changed:
                    break
        return ms

    def reorder(ms):
        ms = list(ms)
        best = _cost(b, ms)
        changed = True
        while changed:
            changed = False
            for i in range(len(ms)):
                for j in range(len(ms)):
                    if i == j:
                        continue
                    rest = ms[:i] + ms[i + 1:]
                    trial = rest[:j] + [ms[i]] + rest[j:]
                    c = _cost(b, trial)
                    if c < best:
                        ms, best, changed = trial, c, True
                        break
                if changed:
                    break
        return ms, best

    # Deleting a macro can LENGTHEN the plan -- the walks that were passing
    # through it now have to go the long way round -- so the un-cut list stays
    # in the running and the cheaper of the two is what ships.
    cand = [reorder(list(macros)), reorder(cut(macros))]
    return min(cand, key=lambda t: t[1])[0]


def _optsets(b, macros):
    """Per-press optimal-action SETS for a flat macro plan.

    A push is labelled with itself. A walk press is labelled with every
    direction that moves nothing but the player AND stays on a shortest route to
    the same stance: no rule fires on a bare move, so those alternatives leave a
    pixel-identical board and choosing between them is arbitrary.
    """
    cur = b.copy()
    presses, sets = [], []
    for stance, d in macros:
        dist_to = _stance_dists(cur, stance)
        while cur.player != stance:
            here = dist_to[cur.player]
            alts = []
            for cand in _ORDER:
                dr, dc = _DELTA[cand]
                nxt = (cur.player[0] + dr, cur.player[1] + dc)
                if dist_to.get(nxt, 1 << 30) != here - 1:
                    continue
                probe = cur.copy()
                if not probe.step(cand):
                    continue
                # (b) it must be a bare move: nothing else on the board changed.
                if (probe.eleph != cur.eleph or probe.poach != cur.poach
                        or probe.arrow != cur.arrow or probe.player != nxt):
                    continue
                alts.append(cand)
            assert alts, "a reachable stance always has a shortest first step"
            presses.append(alts[0])
            sets.append(alts)
            cur.player = (cur.player[0] + _DELTA[alts[0]][0],
                          cur.player[1] + _DELTA[alts[0]][1])
        presses.append(d)
        sets.append([d])
        cur.step(d)
    return presses, sets


def _stance_dists(b, stance) -> dict:
    """Walk distance from every cell TO ``stance`` (the walk graph is undirected,
    so this is one BFS out of the stance)."""
    blocked = b.walls | set(b.arrow) | b.eleph | set(b.poach)
    dist = {stance: 0}
    queue = deque([stance])
    while queue:
        cur = queue.popleft()
        nd = dist[cur] + 1
        for dr, dc in _DELTA.values():
            n = (cur[0] + dr, cur[1] + dc)
            if (0 <= n[0] < b.H and 0 <= n[1] < b.W
                    and n not in blocked and n not in dist):
                dist[n] = nd
                queue.append(n)
    return dist


def solve_board(b, ladder=_LADDER, report=None):
    """Run the ladder tier by tier until one wins; shorten; keep the best."""
    for tier in ladder:
        best = None
        for strict, poach_first, expansions, want, cap, tcap in tier:
            t0 = time.time()
            found = _plan_macros(b, strict=strict, poach_first=poach_first,
                                 expansions=expansions, want=want, cap=cap,
                                 tcap=tcap)
            if found is not None:
                macros = _shorten(b, found[0])
                presses, sets = _optsets(b, macros)
                if best is None or len(presses) < len(best):
                    best = Plan(presses, sets)
            if report:
                tag = (f"{'strict' if strict else 'transit'}/"
                       f"{'poach' if poach_first else 'water'}/{expansions}")
                got = ("FAILED" if found is None
                       else f"{found[1]} -> {len(presses)} presses")
                report(f"      {tag:24s} {got:24s} {time.time() - t0:6.1f}s")
        if best is not None:
            return best
    return None


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

class LeoExpert(PSExpert):
    """Plans on the native `_Board` model and lets the interpreter only certify.

    `PSExpert` owns everything around `_search`: the in-memory memo, the on-disk
    plan cache with its staleness check, and the snapshot/restore discipline. The
    interpreter is never stepped by the planner -- it runs at ~1.5k steps/s and
    the ladder generates tens of millions of successors.
    """

    directions = list(_ORDER)
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        self.idx_to_name = {v: k for k, v in self.g.obj_name_to_idx.items()}
        self.report = None

    def heuristic(self, eng) -> int:                          # pragma: no cover
        raise AssertionError("planning happens on the model; no engine A* runs")

    def board(self, eng) -> _Board:
        return board_from_engine(eng, self.idx_to_name)

    def _search(self, eng):
        return solve_board(self.board(eng), report=self.report)


class LeoSolver(PSAStarSolver):
    game_id = "puzzlescript_leo"
    game_name = GAME_NAME
    game_module_id = "ps:leo"
    expert_cls = LeoExpert

    #: Levels 8, 9 and 11 are NOT SOLVED by this planner and are skipped, so a
    #: cold start does not burn the whole ladder (~50 min each) on them at every
    #: startup. They are not known to be unwinnable -- the arrow-economy test
    #: says all three are count-feasible, and level 8's endgame is reachable by
    #: hand (a PoacherUp walked the length of the board and shoved up into the
    #: frozen top-right pocket, saving the ArrowRight that the cheap opening
    #: kill spends). What defeats the search is that their first annihilation
    #: has to be an EXPENSIVE one: the cheap opening kill spends an arrow a
    #: later pair needs, and nothing here looks far enough ahead to see that.
    #: `discover_solvable` would drop them anyway; this only saves the time.
    skip_levels = frozenset({8, 9, 11})

    #: The longest plan is well over the adapter's default 200-step budget, which
    #: `games/ps:leo` raises; this is that plus room for the exploration prefix.
    max_steps = 700


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _kept(solver, game):
    """The levels the reports run over: everything but `skip_levels`."""
    return [lv for lv in range(game.n_levels) if lv not in solver.skip_levels]


def _levels(discover: bool = False):
    """The adapter + expert the reports run against.

    ``discover`` runs `PSAStarSolver._ensure`, which plans every level up front.
    The rendering reports do not need that (and a cold ladder run is minutes), so
    they build the pair directly and pull plans from the disk cache on demand."""
    solver = LeoSolver()
    if discover:
        game, expert, _solvable = solver._ensure(0)
        return solver, game, expert
    game = solver.make_game(0)
    expert = LeoExpert(game, node_cap=solver.node_cap, weight=solver.weight)
    return solver, game, expert


def _report() -> int:
    """Solve every level, then CERTIFY each plan by replaying it on the real
    interpreter -- the model is what the planner searched, so nothing counts
    until the interpreter agrees it wins."""
    solver, game, expert = _levels()
    eng = game._engine
    total = bad = 0
    levels = _kept(solver, game)
    print(f"{len(levels)} of {game.n_levels} levels "
          f"(skipping {sorted(solver.skip_levels)})\n")
    for level in levels:
        game.set_level(level)
        t0 = time.time()
        plan = expert.plan(eng, level)
        dt = time.time() - t0
        if plan is None:
            print(f"level {level:2d}: NO PLAN ({dt:.1f}s)", flush=True)
            bad += 1
            continue
        # Replay on the interpreter from a fresh load of the level.
        eng.load_level(game._game.levels[level])
        steps = 0
        for d in plan:
            eng.step(d)
            steps += 1
            if eng.check_win():
                break
        ok = eng.check_win() and steps == len(plan)
        labelled = sum(len(s) for s in plan.optsets)
        ties = sum(1 for s in plan.optsets if len(s) > 1)
        total += len(plan)
        bad += 0 if ok else 1
        print(f"level {level:2d}: {len(plan):3d} presses "
              f"({ties:3d} with ties, {labelled:4d} labelled) "
              f"{dt:5.1f}s  {'WINS' if ok else 'DOES NOT REPLAY'}", flush=True)
    print(f"\n{total} presses over {len(levels)} levels")
    print("every plan wins on the interpreter" if not bad
          else f"PLANS FAILED: {bad} levels")
    return 0 if not bad else 1


def _route(b, target):
    """A shortest walk from the player to ``target`` over the free cells."""
    dist = _stance_dists(b, target)
    steps = []
    cur = b.player
    while cur != target:
        for d in _ORDER:
            dr, dc = _DELTA[d]
            nxt = (cur[0] + dr, cur[1] + dc)
            if dist.get(nxt, 1 << 30) == dist[cur] - 1:
                steps.append(d)
                cur = nxt
                break
        else:                                              # pragma: no cover
            raise AssertionError("no route to a reachable cell")
    return steps


def _verify() -> int:
    """Re-derive every labelled press end to end.

    A walk tie cannot be checked by comparing boards -- the alternative puts the
    player somewhere else, which is the whole point of it being a tie. So each
    labelled alternative is expanded into a COMPLETE variant plan (take the
    alternative, walk the rest of the way to the same stance, then the original
    plan from that push onward) and the variant is replayed on the interpreter:
    it has to win, and to take exactly as many presses as the plan it varies.
    That is the claim the label makes, tested rather than restated.
    """
    solver, game, expert = _levels()
    eng, gm = game._engine, game._game
    total = bad = 0
    for level in _kept(solver, game):
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level:2d}: no plan")
            continue
        start = board_from_engine(eng, expert.idx_to_name)

        # Walk the model along the plan once, recording where each press leaves
        # the player and which presses are PUSHES (the ends of the walks).
        cur = start.copy()
        players, pushes = [], []
        for d in plan:
            before = (set(cur.eleph), dict(cur.poach), dict(cur.arrow))
            players.append(cur.player)
            cur.step(d)
            pushes.append((set(cur.eleph), dict(cur.poach), dict(cur.arrow))
                          != before)
        assert cur.won(), "the model must agree the plan wins"

        notes, checked = [], 0
        for i, alts in enumerate(plan.optsets):
            if len(alts) < 2:
                continue
            if pushes[i]:
                notes.append(f"step {i}: a PUSH was labelled with a tie")
                continue
            j = next(k for k in range(i, len(plan)) if pushes[k])
            stance = players[j]
            cur = start.copy()
            for d in plan[:i]:
                cur.step(d)
            for alt in alts:
                if alt == plan[i]:
                    continue
                probe = cur.copy()
                probe.step(alt)
                variant = list(plan[:i]) + [alt] + _route(probe, stance) \
                    + list(plan[j:])
                checked += 1
                if len(variant) != len(plan):
                    notes.append(f"step {i}: {alt} costs {len(variant)}")
                    continue
                eng.load_level(gm.levels[level])
                for d in variant:
                    eng.step(d)
                if not eng.check_win():
                    notes.append(f"step {i}: {alt} does not win")
        total += checked
        bad += len(notes)
        print(f"level {level:2d}: {checked:5d} variant plans replayed: "
              f"{'AGREE' if not notes else '; '.join(notes[:3])}")
    print(f"{total} variant plans replayed on the interpreter")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} disagreements")
    return 0 if not bad else 1


def _fuzz(trials=200, steps=200, seed=7) -> int:
    """Differential fuzz: the model against the interpreter, from a COLD level
    load (no warm-up press -- turn-one bookkeeping is exactly the class of bug a
    warm-up hides), with a coverage counter per mechanic so a clean run cannot
    mean "never exercised it"."""
    _solver, game, expert = _levels()
    eng, gm = game._engine, game._game
    rng = random.Random(seed)
    cov = {}
    n = bad = 0

    def state():
        b = board_from_engine(eng, expert.idx_to_name)
        return b.key()

    for level in range(game.n_levels):
        for _t in range(trials):
            eng.load_level(gm.levels[level])
            b = board_from_engine(eng, expert.idx_to_name)
            for _s in range(steps):
                d = rng.choice(_ORDER)
                before = (b.player, set(b.eleph), dict(b.poach), dict(b.arrow))
                eng.step(d)
                b.step(d)
                n += 1
                tag = ("refused" if before[0] == b.player else
                       "annihilate" if len(before[2]) > len(b.poach) else
                       "convert" if len(before[3]) > len(b.arrow) else
                       "push-elephant" if before[1] != b.eleph else
                       "push-poacher" if before[2] != b.poach else "walk")
                cov[tag] = cov.get(tag, 0) + 1
                if state() != b.key():
                    print(f"  DIVERGE level {level} step {_s} dir={d}")
                    bad += 1
                    break
                if eng.check_win() != b.won():
                    print(f"  WIN MISMATCH level {level}")
                    bad += 1
                    break
                if b.won():
                    cov["win"] = cov.get("win", 0) + 1
                    break
    print(f"{n} transitions from cold level loads, {bad} divergences")
    for k in sorted(cov):
        print(f"  {k:16s} {cov[k]}")
    missing = [k for k in ("annihilate", "convert", "push-elephant",
                           "push-poacher", "refused", "walk") if k not in cov]
    if missing:
        print(f"COVERAGE GAP: {missing}")
    print("model matches the interpreter" if not bad and not missing
          else "FUZZ FAILED")
    return 0 if (not bad and not missing) else 1


_MIRROR = {"left": "right", "right": "left", "up": "up", "down": "down"}
_FLIP = {"up": "down", "down": "up", "left": "left", "right": "right"}


def _symmetry() -> int:
    """Replay every level's plan on the eight turned/mirrored copies of its own
    LAYOUT and require the pieces to land where the transform says.

    Nothing in this game's rules names an absolute direction -- the four
    annihilation rules and the four arrow rules are all written with the
    relative ``>``, so the interpreter is equivariant under the whole square
    group PROVIDED the direction-carrying objects are RELABELLED as the layout
    is turned. That relabel is the point: it is only sound because the sprites
    are drawn as an exact orbit, so a quarter-turned PoacherUp is
    pixel-identical to a PoacherLeft and the presentation shows a board the game
    could have shipped. `_orbit` asserts that separately, on rendered frames.
    """
    solver, game, expert = _levels()
    eng, gm = game._engine, game._game
    idx = gm.obj_name_to_idx

    # rot90 (the augmentation's own direction) sends up -> left -> down -> right.
    turn = {}
    for fam in (_POACHER, _ARROW):
        names = {v: k for k, v in fam.items()}
        for a, b in (("u", "l"), ("l", "d"), ("d", "r"), ("r", "u")):
            turn.setdefault("rot", {})[idx[names[a]]] = idx[names[b]]
        for a, b in (("l", "r"), ("r", "l")):
            turn.setdefault("mir", {})[idx[names[a]]] = idx[names[b]]

    def relabel(cell, kind):
        table = turn[kind]
        return {table.get(o, o) for o in cell}

    def turn_layout(layout, k, mirror):
        rows = [[set(c) for c in row] for row in layout]
        for _ in range(k):
            rows = [[relabel(rows[r][c], "rot") for r in range(len(rows))]
                    for c in range(len(rows[0]) - 1, -1, -1)]
        if mirror:
            rows = [[relabel(c, "mir") for c in row[::-1]] for row in rows]
        return rows

    def pieces():
        want = {idx[n] for n in ("player", "elephant", "wall", "water",
                                 "cage", "black")}
        want |= {idx[n] for n in _POACHER} | {idx[n] for n in _ARROW}
        return frozenset((r, c, o) for r, row in enumerate(eng.grid)
                         for c, cell in enumerate(row) for o in cell
                         if o in want)

    def turn_key(cells, k, mirror, h0, w0):
        """Where the transform says each object ends up. ``h``/``w`` are rebound
        per CELL -- a quarter turn swaps them, so carrying the swap across cells
        would place every object after the first with the wrong width."""
        out = set()
        for (r, c, o) in cells:
            h, w = h0, w0
            for _ in range(k):                         # np.rot90, i.e. CCW
                r, c, o = (w - 1 - c), r, turn["rot"].get(o, o)
                h, w = w, h
            if mirror:
                c, o = w - 1 - c, turn["mir"].get(o, o)
            out.add((r, c, o))
        return frozenset(out)

    bad = 0
    for level in _kept(solver, game):
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            continue
        layout = gm.levels[level]
        h, w = len(layout), len(layout[0])
        notes = []
        for k, mirror in itertools.product(range(4), (False, True)):
            eng.load_level(turn_layout(layout, k, mirror))
            for d in plan:
                dd = d
                for _ in range(k):
                    dd = {"up": "left", "left": "down",
                          "down": "right", "right": "up"}[dd]
                if mirror:
                    dd = _MIRROR[dd]
                eng.step(dd)
            got = (pieces(), eng.check_win())
            eng.load_level(layout)
            for d in plan:
                eng.step(d)
            want = (turn_key(pieces(), k, mirror, h, w), eng.check_win())
            if got != want:
                notes.append(f"k={k} mirror={mirror}")
        bad += len(notes)
        print(f"level {level:2d}: {len(plan):3d} presses on 8 presentations: "
              f"{'EQUIVARIANT' if not notes else 'SPLIT BY ' + ', '.join(notes)}")
    print("the square group is a symmetry of the mechanic once the facings are "
          "relabelled -- which is exactly what the sprites render" if not bad
          else f"PRESENTATIONS DISAGREE: {bad}")
    return 0 if not bad else 1


def _board_shot(eng, gm, h, w, objs, idx):
    eng.height, eng.width = h, w
    eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                 for _ in range(w)] for _ in range(h)]
    eng._position_index_dirty = True
    return np.asarray(_render_frame(eng, gm)).copy()


def _orbit() -> int:
    """Assert the direction-carrying sprites are an exact ORBIT of the frame
    rotation, which is what makes `_symmetry`'s relabel a statement about the
    picture and not only about the rules."""
    _solver, game, _expert = _levels()
    eng, gm = game._engine, game._game
    idx = gm.obj_name_to_idx
    bad = 0
    for fam, label in ((_ARROW, "arrow"), (_POACHER, "poacher")):
        names = {v: k for k, v in fam.items()}
        shots = {f: _board_shot(eng, gm, 8, 8, (names[f],), idx)
                 for f in _FACINGS}
        cycle = ("u", "l", "d", "r")
        for k in range(4):
            for i, f in enumerate(cycle):
                want = cycle[(i + k) % 4]
                if not np.array_equal(np.rot90(shots[f], k=k), shots[want]):
                    print(f"  {label}: rot90({f}, {k}) != {want}")
                    bad += 1
    print("arrows and poachers are exact rotation orbits" if not bad
          else f"ORBIT FAILED: {bad} mismatches")
    return 0 if not bad else 1


def _compositions(game, expert):
    """Every cell COMPOSITION the game can build, per board size.

    Layer 2 (background / black / water / cage) and layer 3 (the walls, the
    player and every piece) are independent, so the reachable stacks are exactly
    the pairs the level supplies -- enumerated from the level grids and from
    every state along the level's own plan, rather than listed by hand.
    """
    eng, gm = game._engine, game._game
    name = gm.obj_idx_to_name
    ground_names = {"black", "water", "cage"}
    per_size = {}
    keep = _kept(_LEVEL_SOLVER, game)

    def scan(seen):
        for cell in itertools.chain.from_iterable(eng.grid):
            seen.add(tuple(sorted(name[o] for o in cell
                                  if name[o] != "background")))

    for level in keep:
        game.set_level(level)
        h, w = eng.height, eng.width
        seen = per_size.setdefault((h, w), set())
        seen.add(())
        present = {name[o] for cell in itertools.chain.from_iterable(eng.grid)
                   for o in cell}
        # every ground x body pair the level's own pieces can build: the two
        # collision layers are independent, so this is exactly the stack space.
        grounds = {()} | {(g,) for g in present & ground_names}
        bodies = {()} | {(g,) for g in present - ground_names - {"background"}}
        for g in grounds:
            for body in bodies:
                if g and body == ("wall",):
                    continue                       # walls are never laid on one
                seen.add(tuple(sorted(g + body)))
        # ...plus whatever the level actually shows along its own plan
        plan = expert.plan(eng, level)
        eng.load_level(gm.levels[level])
        scan(seen)
        for d in (plan or []):
            eng.step(d)
            scan(seen)
    return per_size


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct in the rendered frame, at every
    cell size the levels actually use.

    Whole frames are compared, never an arithmetic cell crop: `_render_frame`
    upscales the board to fill 64x64 and letterboxes it, so the output's cell
    grid is not ``cell_px``-aligned (see ps:explod / ps:esl_puzzle_game, where
    exactly that made an audit report every composition identical to floor).
    """
    solver, game, expert = _levels()
    eng, gm = game._engine, game._game
    idx = gm.obj_name_to_idx
    global _LEVEL_SOLVER
    _LEVEL_SOLVER = solver
    per_size = _compositions(game, expert)
    clashes = []
    for (h, w), comps in sorted(per_size.items()):
        comps = sorted(comps)
        shots = {objs: _board_shot(eng, gm, h, w, objs, idx) for objs in comps}
        cell_px = min(64 // h, 64 // w)
        here = [(a, b) for a, b in itertools.combinations(comps, 2)
                if np.array_equal(shots[a], shots[b])]
        print(f"{h}x{w} boards, cell {cell_px}px, {len(comps)} compositions"
              + ("" if not here else "   <-- CLASHES"))
        for objs in comps:
            painted = int((shots[objs] != shots[()]).sum())
            label = "+".join(objs) if objs else "floor"
            print(f"    {label:34s} {painted:5d} px differ from bare floor")
        for a, b in here:
            print(f"    IDENTICAL: {'+'.join(a) or 'floor'} == "
                  f"{'+'.join(b) or 'floor'}")
        clashes += here
    print("audit clean" if not clashes
          else f"AUDIT FAILED: {len(clashes)} indistinguishable pairs")
    return 0 if not clashes else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--orbit" in sys.argv:
        sys.exit(_orbit())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(LeoSolver.main())
