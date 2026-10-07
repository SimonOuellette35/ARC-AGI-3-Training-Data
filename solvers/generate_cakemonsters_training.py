"""Generate Phase-1 training data for the PuzzleScript game ps:cakemonsters
("Cake Monsters" by Matt Rix).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, the native model of the turn, the search ladder over it, and the
optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_cakemonsters",
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
action (post rotation remap), i.e. the button an agent presses in the presented
view, so replaying the recorded actions reproduces the recorded frames exactly.

The game
--------
Every monster on the board is a ``Player``, so ONE arrow key moves ALL of them
one cell in that direction. Monsters come in seven colours -- the three
primaries red / yellow / blue, the three secondaries purple / orange / green,
and brown ("trash") -- and so do the cakes. **The win is ``No Cake``**: every
cake must be eaten, and a cake is only edible by a monster of its own colour.
The game is declared ``noaction``, so the whole action space is the four arrows.

Four mechanics do all the work, and all four are consequences of the rule order
inside a single turn (rules run to a fixpoint BEFORE anything actually moves):

  * **Eating is a teleport, not a move.** ``[> RedMon | RedCake] -> [ | RedMon]``
    deletes the cake and re-seats the monster on the cake's cell WITHOUT a
    movement flag. So the eater lands one cell forward and is *stationary* for
    the rest of the turn, and the cell it came from is empty -- which is what
    lets a follower step into it on the same press.
  * **Anything in front that is not your cake stops you**: a wall, a
    wrong-colour cake, or the board edge. That is the only way the formation
    ever breaks, since every monster is pushed by the same key.
  * **A moving monster that runs into a STATIONARY one eats it and takes its
    cell**, and the survivor's colour is the MIX of the two: red+blue=purple,
    red+yellow=orange, blue+yellow=green, and every other pair (primary with
    secondary, two different secondaries, anything with trash) = trash. Same
    colour into same colour just survives. Because the "stopped by a wall" rule
    fires first and the whole rule list then loops, a *line* of monsters shoved
    into a wall collapses pairwise from the front -- which is how a level with
    only red, yellow and blue monsters clears purple, orange and green cakes.
    Mixing is IRREVERSIBLE and it destroys a monster every time, so the number
    of monsters only ever goes down: spending two primaries to make a secondary
    is the central resource decision of the whole game.
  * **A Destroyer tile deletes any monster that walks into it** -- the six
    ``destroyerz`` levels are built on steering the *other* monsters while the
    one you want gone marches into the X.

There is no lose condition. A level whose last red monster has been merged into
trash while a red cake is still on the board is simply unwinnable and will sit
there forever, which is exactly why the search needs the dead-state prunes below.

Model + search
--------------
The interpreter costs ~2.2 ms per press, so searching it directly is hopeless at
these depths (the deepest level is 61 presses). `_step` re-implements one turn
natively at ~10 us.

It gets the rule fixpoint in ONE PASS by walking the monsters **front-first**
(most advanced along the pressed direction first). A monster's fate depends only
on what is in front of it, and everything in front has already been resolved by
then, so the loop never needs a second sweep:

    in front is a wall / the edge   -> stationary
    in front is a Destroyer        -> deleted
    in front is my cake            -> eat it, jump onto that cell, stationary
    in front is a wrong cake       -> stationary
    in front is a stationary mon   -> merge into it (mix colours), I am gone
    in front is a moving mon /
      an emptied cell              -> keep moving

and afterwards every still-moving monster advances one cell. No collision check
is needed at that point: all monsters share one direction, so two of them can
never target the same cell, and the fixpoint has already resolved every monster
whose target was occupied.

``--selfcheck`` drives 200 random rollouts x 60 presses through BOTH the
interpreter and `_step` on all 36 levels, comparing every monster's cell AND
colour and every surviving cake after every press. That is the guard that lets
the searches trust the model; it currently reports 0 mismatches in ~430k
transitions.

The state is ``(sorted (cell, colour) monsters, live-cake bitmask)`` -- cakes
never move, so a bitmask over the level's initial cake list is an exact
canonical record of which ones are left.

The heuristic is the max of two admissible bounds plus two dead-state tests:

  * **Travel.** ``max`` over live cakes of the shortest walk from the nearest
    monster (walls and destroyers block, cakes ignored). Whoever eats that cake
    has to get there, so no plan is shorter. A cake NO monster can walk to is
    dead outright.
  * **Throughput.** A monster eats at most one cake per press, and only cakes of
    its own colour, so for each colour ``ceil(cakes of that colour / monsters
    that could ever be that colour at once)``. The divisor is
    ``have + (rest // 2)``: making one more monster of a colour costs at least
    two others, and merging never increases the population. This is the bound
    that carries the big levels -- level 11 has 50 cakes but only one blue
    monster for its 16 blue ones.
  * **Colour closure.** A cake whose colour is not in the mixing closure of the
    monsters still alive can never be eaten -- the state is dead. This is what
    prunes the enormous "everybody merged into trash" subtree that random play
    falls into immediately.

`_astar_exact` runs A* to *proven* optimality (goal test on pop, then the queue
is drained of everything with ``f <= D``), which makes ``g`` exact for every
state on a shortest path; one backward sweep then marks, for each such state,
every press that stays on one. **32 of the 36 levels are solved exactly this
way** (worst: 13 s, 146k states), and they get TRUE optimal-action sets rather
than one arbitrary interleaving.

The four that exact A* cannot close inside its node cap (11, 23, 25 and 35 --
the 50-cake serpentine, the 13x13, and the two "final challenge" boards) fall
through a ladder: weighted A*, then a width-capped breadth-first beam ranked on
cakes-remaining. Their plans are engine-verified wins but not provably shortest,
so those levels label each step with the press the expert took.

Plan cache
----------
The whole ladder costs ~5 minutes ONCE. Engine state after reset is
seed-independent for these games (only the presentation is augmented), so every
plan and every optimal set is written to ``data/cakemonsters_plans.json`` and
re-read by every later seed, every later run and -- what actually matters --
every shard of `parallelize_generator.py`, which would otherwise redo the search
on each core. Each entry records the start layout it was solved from and is
ignored if that no longer matches, so editing the level in the .txt cannot
silently serve a stale plan. Delete the file to force a fresh search.

Recovery
--------
``recovery_mode = "reset"`` and ``epsilon = 0``, the family default, because
this game is genuinely IRREVERSIBLE: an eaten cake does not come back and two
merged monsters cannot be un-merged. A random detour can strand a level for
good, so recovery data comes from the explore-then-RESET prefix in front of
every level rather than from mid-plan detours.

Augmentation
------------
Per (seed, level) the adapter draws a frame rotation (0..3) and, since
``cakemonsters`` is in `PuzzleScriptAdapter._FLIP_GAMES`, an independent
horizontal and vertical flip -- 16 presentations of each of the 36 levels. The
flips are an exact symmetry of this game rather than a plausible one: not one
rule names a direction (they are all stated with the relative ``>`` force), the
win condition is ``No Cake``, there is no gravity and input is screen-relative,
so reflecting the board maps the game onto itself.

Rendering was audited at every cell_px the 36 levels use (4 through 9) and needs
no sprite fix: all seven monster colours, all seven cake colours, wall,
background, blackness and destroyer render to 18 mutually distinct cell blocks,
and monsters stay distinguishable from cakes of the same colour by their eye
pixels even at cell_px 4.

Usage (run from the repo root):
    python solvers/generate_cakemonsters_training.py --episodes 200 \
        --out data/training_multi_level/cakemonsters
    python solvers/generate_cakemonsters_training.py --selfcheck
    python solvers/generate_cakemonsters_training.py --plans
"""

from __future__ import annotations

import heapq
import json
import os
import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import PuzzleScriptAdapter   # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert     # noqa: E402

GAME_NAME = "cakemonsters"

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Where the ladder's results are kept between runs, in the repo's usual
#: ``data/<game>_plans.json`` shape. The plans are seed-independent, so this
#: turns the one-off search cost into a one-off-EVER cost.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "cakemonsters_plans.json"

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

#: Monster object indices are 0..6 and the matching cake is ``colour + 7``, so
#: this tuple order IS the engine's object order and must not be permuted.
COLOURS = ("red", "yellow", "blue", "purple", "orange", "green", "trash")
RED, YELLOW, BLUE, PURPLE, ORANGE, GREEN, TRASH = range(7)

#: The three primary pairs; everything else mixes to trash.
_PAIR = {(RED, BLUE): PURPLE, (BLUE, RED): PURPLE,
         (RED, YELLOW): ORANGE, (YELLOW, RED): ORANGE,
         (BLUE, YELLOW): GREEN, (YELLOW, BLUE): GREEN}

#: ``MIX[mover][stationary]`` -- the colour of the monster left standing. Same
#: into same keeps the colour; a primary pair makes its secondary; every other
#: combination (primary+secondary, two secondaries, anything with trash) is
#: trash. Symmetric, which is why the two orderings of each PuzzleScript rule
#: collapse to one table.
MIX: list[list[int]] = [
    [a if a == b else _PAIR.get((a, b), TRASH) for b in range(7)]
    for a in range(7)
]

#: Mixing closure of a colour SET, as bitmask -> bitmask: every colour reachable
#: by merging some of these monsters together. Precomputed for all 128 sets
#: because the heuristic asks for it at every node.
def _closure_table() -> list[int]:
    out = []
    for start in range(128):
        cur = start
        while True:
            nxt = cur
            for a in range(7):
                if not (cur >> a) & 1:
                    continue
                for b in range(7):
                    if (cur >> b) & 1:
                        nxt |= 1 << MIX[a][b]
            if nxt == cur:
                break
            cur = nxt
        out.append(cur)
    return out


CLOSURE: list[int] = _closure_table()

#: The whole action space -- the game is declared ``noaction``.
DIRS = ("up", "down", "left", "right")
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Engine object indices. Monsters are 0..6 in `COLOURS` order, their cakes are
#: 7..13 in the same order, then destroyer / background / wall / blackness. Only
#: the two that constrain movement need naming; background and blackness are
#: scenery (a monster may in principle stand on either, and no shipped level
#: leaves the walled play area anyway).
_DESTROYER, _WALL = 14, 16

_INF = 1 << 30


# ---------------------------------------------------------------------------
# Static level geometry
# ---------------------------------------------------------------------------

class _Level:
    """Everything about a level that never changes, compiled to flat int arrays.

    Cells are ``r * w + c``. ``nbr[d][cell]`` is the cell one step along ``d``,
    or -1 when that is a wall or off the board -- which folds the two "you are
    blocked" tests of the turn into one lookup. ``rank[d]`` orders the monsters
    front-first for `_step`, and ``dist[a][b]`` is the walking distance used by
    the heuristic (walls and destroyers block; cakes do not, since another
    monster may clear them first, and ignoring them keeps the bound admissible).
    """

    __slots__ = ("h", "w", "n", "walls", "isdest", "cake_at", "cake_col",
                 "cake_cell", "nbr", "rank", "dist", "start_mons", "start_mask")

    def __init__(self, eng):
        grid = eng.grid
        self.h = h = len(grid)
        self.w = w = len(grid[0])
        self.n = h * w
        walls: set[int] = set()
        self.isdest = [False] * self.n
        self.cake_at = [-1] * self.n
        self.cake_col: list[int] = []
        self.cake_cell: list[int] = []
        mons: dict[int, int] = {}
        for r in range(h):
            for c in range(w):
                cell = grid[r][c]
                i = r * w + c
                if _WALL in cell:
                    walls.add(i)
                if _DESTROYER in cell:
                    self.isdest[i] = True
                for o in cell:
                    if o < 7:
                        mons[i] = o
                    elif o < 14:
                        self.cake_at[i] = len(self.cake_col)
                        self.cake_col.append(o - 7)
                        self.cake_cell.append(i)
        self.walls = walls
        self.nbr = {}
        self.rank = {}
        for d, (dr, dc) in _DELTA.items():
            nb = [-1] * self.n
            for r in range(h):
                for c in range(w):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        j = nr * w + nc
                        if j not in walls:
                            nb[r * w + c] = j
            self.nbr[d] = nb
            # How far along ``d`` a cell lies. Sorting monsters by -rank puts the
            # frontmost first, which is the whole ordering `_step` needs.
            self.rank[d] = [r * dr + c * dc for r in range(h) for c in range(w)]
        self.dist = self._distances()
        self.start_mons = tuple(sorted(mons.items()))
        self.start_mask = (1 << len(self.cake_col)) - 1

    def _distances(self) -> list:
        """``dist[src][dst]`` = shortest walk, or -1. One BFS per floor cell."""
        blocked = [(i in self.walls) or self.isdest[i] for i in range(self.n)]
        nbrs = [[self.nbr[d][i] for d in DIRS] for i in range(self.n)]
        out: list = [None] * self.n
        for s in range(self.n):
            if blocked[s]:
                continue
            d = [-1] * self.n
            d[s] = 0
            queue = deque([s])
            while queue:
                p = queue.popleft()
                for j in nbrs[p]:
                    if j >= 0 and not blocked[j] and d[j] < 0:
                        d[j] = d[p] + 1
                        queue.append(j)
            out[s] = d
        return out


# ---------------------------------------------------------------------------
# One turn, natively
# ---------------------------------------------------------------------------

def _step(lvl: _Level, mons: tuple, mask: int, d: str) -> tuple:
    """One press of ``d``. ``mons`` is a sorted tuple of ``(cell, colour)`` and
    ``mask`` the live-cake bitmask; returns the settled ``(mons, mask)``.

    Front-first resolution of the rule fixpoint -- see the module docstring for
    why one pass suffices.
    """
    nbr = lvl.nbr[d]
    isdest = lvl.isdest
    cake_at = lvl.cake_at
    cake_col = lvl.cake_col
    rank = lvl.rank[d]
    occ = dict(mons)
    moving = set(occ)
    for p in sorted(occ, key=lambda cell: -rank[cell]):
        col = occ.get(p)
        if col is None:
            continue                       # already merged away by a follower
        q = nbr[p]
        if q < 0:                          # wall or the board edge
            moving.discard(p)
            continue
        if isdest[q]:                      # walked into an X
            del occ[p]
            moving.discard(p)
            continue
        ci = cake_at[q]
        if ci >= 0 and (mask >> ci) & 1:
            if cake_col[ci] == col:        # eat: the monster JUMPS onto the cake
                mask &= ~(1 << ci)
                del occ[p]
                occ[q] = col
            moving.discard(p)              # eater and blocked-by-cake both stop
            continue
        if q in occ and q not in moving:   # merge into a stationary neighbour
            occ[q] = MIX[col][occ[q]]
            del occ[p]
            moving.discard(p)
    if moving:
        occ = {(nbr[p] if p in moving else p): c for p, c in occ.items()}
    return tuple(sorted(occ.items())), mask


def _read(lvl: _Level, eng) -> tuple:
    """``(mons, mask)`` read off the engine grid."""
    mons: dict[int, int] = {}
    mask = 0
    w = lvl.w
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            for o in cell:
                if o < 7:
                    mons[r * w + c] = o
                elif o < 14:
                    mask |= 1 << lvl.cake_at[r * w + c]
    return tuple(sorted(mons.items())), mask


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

def _heuristic(lvl: _Level, mons: tuple, mask: int) -> tuple[int, int]:
    """``(lower bound on presses, sum of nearest-monster distances)``.

    The first is admissible and is what A* steers on; ``_INF`` means the state
    is dead (a cake nobody can reach, or nobody can ever be the colour of). The
    second is a pure ranking signal for the beam -- it is NOT a bound.

    IT IS ALSO CONSISTENT, which is what `_astar_exact` and therefore every
    optimal-action set in this file rest on -- a merely admissible estimate
    would let A* pop a state at more than its true ``g*`` and the backward sweep
    would then mark the wrong presses. Both terms survive one press losing at
    most 1, and a max of consistent terms is consistent:

      * Travel. Each monster moves at most one cell and monsters only ever
        vanish, so a surviving cake's nearest-monster distance drops by at most
        one. A cake that vanished was being stood next to, i.e. its distance was
        1 -- so it cannot have been the max unless every other cake was at 1 too,
        in which case the estimate was 1 and can only fall to 0.
      * Throughput. A press eats at most ``mcnt[col]`` cakes of a colour (one per
        monster of it), and ``mcnt[col] <= ub``, so ``k`` falls by at most ``ub``
        and ``ceil(k/ub)`` by at most one. ``ub`` shrinking only ever raises the
        estimate, which consistency permits.
    """
    if not mask:
        return 0, 0
    if not mons:
        return _INF, _INF
    have = 0
    mcnt = [0] * 7
    for _cell, col in mons:
        have |= 1 << col
        mcnt[col] += 1
    reachable = CLOSURE[have]
    total = len(mons)

    hmax = 0
    ccnt = [0] * 7
    m = mask
    while m:
        bit = m & -m
        i = bit.bit_length() - 1
        m ^= bit
        ccnt[lvl.cake_col[i]] += 1
    for col in range(7):
        k = ccnt[col]
        if not k:
            continue
        if not (reachable >> col) & 1:
            return _INF, _INF              # nobody can ever eat this colour
        # At most this many monsters can be ``col`` at one time: the ones that
        # already are, plus one per PAIR of the others (a merge consumes two).
        ub = mcnt[col] + (total - mcnt[col]) // 2
        if ub == 0:
            return _INF, _INF
        v = -(-k // ub)
        if v > hmax:
            hmax = v

    dist = lvl.dist
    tot = 0
    m = mask
    while m:
        bit = m & -m
        i = bit.bit_length() - 1
        m ^= bit
        cell = lvl.cake_cell[i]
        best = _INF
        for p, _col in mons:
            row = dist[p]
            if row is None:
                continue
            v = row[cell]
            if 0 <= v < best:
                best = v
        if best >= _INF:
            return _INF, _INF              # this cake is walled off from everyone
        if best > hmax:
            hmax = best
        tot += best
    return hmax, tot


# ---------------------------------------------------------------------------
# Searches
# ---------------------------------------------------------------------------

def _astar_exact(lvl: _Level, mons: tuple, mask: int, node_cap: int):
    """A* run to PROVEN optimality. Returns ``(D, gmap)`` or None.

    The goal test is on pop, and the queue keeps draining until its ``f``
    exceeds the winning cost ``D``. Because the heuristic is admissible, every
    state on a shortest path has ``g* + h <= D`` and is therefore popped, so
    ``gmap`` holds an exact ``g*`` for all of them -- which is exactly what
    `_sweep` needs to recover the optimal-action sets.
    """
    if not mask:
        return 0, {(mons, mask): 0}
    h0, _tot = _heuristic(lvl, mons, mask)
    if h0 >= _INF:
        return None
    pq = [(h0, 0, 0, mons, mask)]
    best = {(mons, mask): 0}
    gmap: dict[tuple, int] = {}
    counter = 0
    nodes = 0
    won = None
    while pq:
        f, g, _c, m, k = heapq.heappop(pq)
        if won is not None and f > won:
            break
        state = (m, k)
        prev = gmap.get(state)
        if prev is not None and prev <= g:
            continue                       # stale duplicate
        # Consistency (see `_heuristic`) makes the first pop of a state its g*,
        # so this reopen never fires today. It is the belt to that braces: get
        # the estimate wrong later and the sweep silently mislabels instead of
        # failing, which is the one bug class this file cannot afford.
        gmap[state] = g
        if not k:
            if won is None or g < won:
                won = g
            continue
        for d in DIRS:
            nxt = _step(lvl, m, k, d)
            nodes += 1
            if nxt == state:
                continue                   # a press that changed nothing
            ng = g + 1
            if best.get(nxt, _INF) <= ng:
                continue
            hv, _t = _heuristic(lvl, nxt[0], nxt[1])
            if hv >= _INF or (won is not None and ng + hv > won):
                continue
            best[nxt] = ng
            counter += 1
            heapq.heappush(pq, (ng + hv, ng, counter, nxt[0], nxt[1]))
        if nodes >= node_cap:
            return None
    if won is None:
        return None
    return won, gmap


def _sweep(lvl: _Level, gmap: dict, won: int) -> dict:
    """``{state: [optimal presses]}`` for every state on a shortest path.

    Walks the exact ``g*`` map backwards: a state at depth ``g`` is on a
    shortest path when some press reaches a state at depth ``g + 1`` that is
    itself on one. Wins seed the recursion.
    """
    by_depth: dict[int, list] = {}
    for state, g in gmap.items():
        by_depth.setdefault(g, []).append(state)
    on_opt = {s for s in by_depth.get(won, ()) if not s[1]}
    opt: dict[tuple, list] = {}
    for g in range(won - 1, -1, -1):
        for state in by_depth.get(g, ()):
            dirs = [d for d in DIRS
                    if (nxt := _step(lvl, state[0], state[1], d)) in on_opt
                    and gmap.get(nxt) == g + 1]
            if dirs:
                on_opt.add(state)
                opt[state] = dirs
    return opt


def _astar(lvl: _Level, mons: tuple, mask: int, node_cap: int, weight: int):
    """Weighted A* -- a genuine win path, not necessarily the shortest."""
    if not mask:
        return []
    h0, _tot = _heuristic(lvl, mons, mask)
    if h0 >= _INF:
        return None
    pq = [(weight * h0, 0, 0, mons, mask, ())]
    best = {(mons, mask): 0}
    counter = 0
    nodes = 0
    while pq:
        _f, g, _c, m, k, path = heapq.heappop(pq)
        if not k:
            return list(path)
        if best.get((m, k), _INF) < g:
            continue
        for d in DIRS:
            nm, nk = _step(lvl, m, k, d)
            nodes += 1
            if (nm, nk) == (m, k):
                continue
            ng = g + 1
            if best.get((nm, nk), _INF) <= ng:
                continue
            hv, _t = _heuristic(lvl, nm, nk)
            if hv >= _INF:
                continue
            best[(nm, nk)] = ng
            counter += 1
            heapq.heappush(pq, (ng + weight * hv, ng, counter, nm, nk,
                                path + (d,)))
        if nodes >= node_cap:
            return None
    return None


def _beam(lvl: _Level, mons: tuple, mask: int, width: int,
          depth: int = 400, node_cap: int = 20_000_000):
    """Width-capped breadth-first beam, ranked on cakes remaining.

    WHY (what A* runs out of). The travel bound goes FLAT across the middle of
    the big boards: for twenty presses the monsters are shuffling into position
    to merge, and the distance to the farthest cake does not move at all, so A*
    has nothing to steer with and degenerates to uniform-cost search at a depth
    where that is hopeless. Cakes-eaten is not a bound and can never drive A*,
    but it is a perfectly good survivor ranking, which is all a flat heuristic
    is good for.
    """
    if not mask:
        return []
    frontier = [(mons, mask, ())]
    seen = {(mons, mask)}
    nodes = 0
    for _depth in range(depth):
        kids = []
        for m, k, path in frontier:
            for d in DIRS:
                nxt = _step(lvl, m, k, d)
                nodes += 1
                if not nxt[1]:
                    return list(path) + [d]
                if nxt == (m, k) or nxt in seen:
                    continue
                hv, tot = _heuristic(lvl, nxt[0], nxt[1])
                if hv >= _INF:
                    continue
                seen.add(nxt)
                kids.append((bin(nxt[1]).count("1"), tot, hv,
                             nxt[0], nxt[1], path + (d,)))
            if nodes >= node_cap:
                return None
        if not kids:
            return None                    # the reachable space closed
        kids.sort(key=lambda kid: kid[:3])
        frontier = [(m, k, p) for _c, _t, _h, m, k, p in kids[:width]]
    return None


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

#: Exact-A* budget, in generated nodes. 32 of the 36 levels close inside it
#: (worst measured need ~500k, level 33); the four that do not are not close
#: enough for a larger cap to help -- a 3M-node run bounded by the beam's own
#: answer still fails on all four -- so the ladder moves on instead.
EXACT_CAP = 800_000

#: ``(weight, node budget)`` rungs tried after exact A* gives up, then beam
#: widths. Measured: level 23 wins at w=2, levels 11 / 25 / 35 need the beam
#: (widths 16000 / 4000 / 4000).
LADDER: tuple[tuple[int, int], ...] = ((2, 600_000), (3, 600_000), (5, 600_000))
BEAM_WIDTHS: tuple[int, ...] = (4_000, 16_000, 48_000)


class CakeMonstersExpert(PSExpert):
    """Planner over the native model, with a disk-backed plan + optimal-set cache.

    `PSExpert` supplies the in-memory plan memo, the restore discipline and the
    level scoping; `_search` swaps in the ladder, so the interpreter is only ever
    stepped by the recorder -- which is also what verifies every plan, since a
    level is kept only when the engine reports WIN.
    """

    directions = list(DIRS)

    #: `_key` is monsters + live cakes, canonical only WITHIN a level (the walls,
    #: destroyers and cake layout that complete the state are static per level
    #: but differ between levels).
    scope_by_level = True

    def setup(self) -> None:
        self._lvl: dict[int, _Level] = {}
        self._opt: dict[int, dict] = {}       # level -> {state: [dirs]}
        self._disk: dict[int, dict] = self._load_disk()
        self._cur: tuple | None = None
        #: Set by `CakeMonstersSolver.prepare_expert` to the generator's own RNG,
        #: so a level with several equally short routes is walked down a
        #: DIFFERENT one on each seed (see `_materialise`).
        self.rng: random.Random | None = None

    # -- level geometry ------------------------------------------------------
    def level(self, eng, level: int | None) -> _Level:
        lvl = self._lvl.get(level)
        if lvl is None:
            lvl = self._lvl[level] = _Level(eng)
        return lvl

    def read(self, eng, level: int | None = None) -> tuple:
        return _read(self.level(eng, level), eng)

    # -- disk plan cache -----------------------------------------------------
    @staticmethod
    def _layout_json(lvl: _Level) -> dict:
        """The start layout a cached entry was solved from. Compared verbatim on
        load, so editing the level in the .txt invalidates its plan instead of
        silently serving a stale one."""
        return {"size": [lvl.h, lvl.w],
                "mons": [[c, col] for c, col in lvl.start_mons],
                "cakes": [[c, col] for c, col
                          in zip(lvl.cake_cell, lvl.cake_col)],
                "walls": sorted(lvl.walls),
                "dest": [i for i, x in enumerate(lvl.isdest) if x]}

    def _load_disk(self) -> dict:
        """``{level: entry}``, or empty if unreadable. A cache that cannot be
        parsed is a cache miss, never a crash."""
        try:
            raw = json.loads(PLAN_CACHE.read_text())
            return {int(k): v for k, v in raw.items()}
        except Exception:                                  # noqa: BLE001
            return {}

    def _save_disk(self, level: int, entry: dict) -> None:
        """Merge ``entry`` into the file and rewrite it atomically.

        Re-reading first is what makes this safe under `parallelize_generator`:
        shards start together, each solves the levels it reaches first, and a
        blind dump of one shard's dict would drop what the others had already
        proved. ``os.replace`` keeps a reader from ever seeing a half-written
        file."""
        self._disk[level] = entry
        try:
            merged = self._load_disk()
            merged.update(self._disk)
            self._disk = merged
            PLAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = PLAN_CACHE.with_suffix(f".json.tmp{os.getpid()}")
            tmp.write_text(json.dumps(
                {str(k): merged[k] for k in sorted(merged)}, indent=1))
            os.replace(tmp, PLAN_CACHE)
        except OSError:
            pass                                           # the cache is optional

    @staticmethod
    def _opt_json(opt: dict) -> list:
        return [[[c for cell in state[0] for c in cell], state[1], dirs]
                for state, dirs in sorted(opt.items())]

    @staticmethod
    def _opt_from_json(raw: list) -> dict:
        out = {}
        for flat, mask, dirs in raw:
            mons = tuple((flat[i], flat[i + 1]) for i in range(0, len(flat), 2))
            out[(mons, mask)] = list(dirs)
        return out

    # -- planning ------------------------------------------------------------
    def plan(self, eng, level: int | None = None) -> list | None:
        """A WIN sequence of presses from the engine's current state, or None.
        Leaves the engine untouched -- the search never runs it.

        `PSExpert.plan`'s memo is kept (keyed by ``(level, state)``, and the
        engine state after reset is seed-independent, so seed 0 pays for every
        search), but the plan handed back is MATERIALISED fresh from the
        optimal-set map each time, so successive seeds record genuinely
        different routes rather than the same key sequence in four rotations."""
        lvl = self.level(eng, level)
        self._cur = (level, lvl)
        state = _read(lvl, eng)
        key = (level, state)
        if key not in self.cache:
            self.cache[key] = self._search(eng)
        plan = self.cache[key]
        if plan is None:
            return None
        table = self._opt.get(level)
        if table and state in table:
            return self._materialise(lvl, table, state)
        return list(plan)

    def _materialise(self, lvl: _Level, table: dict, state: tuple) -> list:
        """Walk the optimal-set map from ``state`` to a win, drawing each step
        from `rng`. Every route it can produce is exactly as short as any
        other, so this is free variety -- and the labels stay the same sets."""
        out: list = []
        while state[1]:
            dirs = table[state]
            d = dirs[0] if self.rng is None else self.rng.choice(dirs)
            out.append(d)
            state = _step(lvl, state[0], state[1], d)
        return out

    def _key(self, eng) -> tuple:
        # Monsters + live cakes. Cheaper than the base's whole-grid key and
        # exact: nothing else on these boards ever changes.
        return _read(self._cur[1], eng)

    def heuristic(self, eng) -> int:
        mons, mask = self._key(eng)
        return _heuristic(self._cur[1], mons, mask)[0]

    def _search(self, eng) -> list | None:
        level, lvl = self._cur
        mons, mask = _read(lvl, eng)
        at_start = (mons, mask) == (lvl.start_mons, lvl.start_mask)

        if at_start:
            entry = self._disk.get(level)
            if entry is not None and entry["start"] == self._layout_json(lvl):
                if entry.get("opt"):
                    self._opt[level] = self._opt_from_json(entry["opt"])
                return list(entry["plan"]) if entry["plan"] is not None else None

        plan, opt = self._ladder(lvl, mons, mask)
        if opt:
            self._opt.setdefault(level, {}).update(opt)
        if at_start:
            self._save_disk(level, {"start": self._layout_json(lvl),
                                    "exact": bool(opt),
                                    "plan": plan,
                                    "opt": self._opt_json(opt) if opt else None})
        return plan

    def _ladder(self, lvl: _Level, mons: tuple, mask: int):
        """``(plan, optimal-set map)``. The map is empty when only an inexact
        rung won, in which case `CakeMonstersSolver.optimal_for` falls back to
        the press the expert took."""
        res = _astar_exact(lvl, mons, mask, self.node_cap)
        if res is not None:
            won, gmap = res
            opt = _sweep(lvl, gmap, won)
            plan: list = []
            state = (mons, mask)
            while state[1]:
                d = opt[state][0]
                plan.append(d)
                state = _step(lvl, state[0], state[1], d)
            return plan, opt
        for weight, cap in LADDER:
            plan = _astar(lvl, mons, mask, cap, weight)
            if plan is not None:
                return plan, {}
        for width in BEAM_WIDTHS:
            plan = _beam(lvl, mons, mask, width)
            if plan is not None:
                return plan, {}
        return None, {}

    def optimal_dirs(self, eng, level: int | None) -> list | None:
        """Every press that keeps the game on a SHORTEST route to the win, or
        None when this level was only solved by an inexact rung."""
        table = self._opt.get(level)
        if not table:
            return None
        return table.get(_read(self.level(eng, level), eng))


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class CakeMonstersSolver(PSAStarSolver):
    game_id = "puzzlescript_cakemonsters"
    game_name = GAME_NAME
    expert_cls = CakeMonstersExpert

    node_cap = EXACT_CAP
    #: The adapter GAME_OVERs at 200 actions per level; the longest plan is 69
    #: presses (level 25), leaving the rest to the exploration prefix.
    max_steps = 200
    #: Zero, the family default: eating a cake and merging two monsters are both
    #: irreversible, so a mid-plan detour can strand a level for good. Recovery
    #: data comes from the explore-then-RESET prefix instead.
    epsilon = 0.0

    def prepare_expert(self, game, expert) -> None:
        """Hand the expert the generator's RNG before any level is planned, so
        the route it walks down a level with ties varies from seed to seed. One
        shared stream keeps a run reproducible from ``--seed`` alone."""
        expert.rng = self.rng

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """Every press that keeps the game on a SHORTEST route to the win, read
        off the LIVE engine state.

        `record_level` asks for this before executing the press, so the engine is
        still at the state being labelled. The set is exact on the 32 levels that
        exact A* closed -- a step whose plan is one of several equally short
        routes is labelled with all of them instead of with one arbitrary
        interleaving.

        Falls back to the press about to be taken (the four levels only a
        weighted / beam rung wins, where no optimal set is known): no expert step
        may ship unlabelled."""
        best = expert.optimal_dirs(expert.game._engine, level)
        if best:
            return best
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Model self-check
# ---------------------------------------------------------------------------

def selfcheck(trials: int = 200, steps: int = 60, verbose: bool = True) -> int:
    """Drive random presses through BOTH the interpreter and `_step`, comparing
    every monster's cell AND colour and every surviving cake after every press.
    This is the guard that lets the searches trust the native model.

    Returns the number of mismatches."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        lvl = _Level(game._engine)
        rng = random.Random(f"cakemonsters:selfcheck:{level}")
        bad = 0
        for _t in range(trials):
            game.set_level(level)
            eng = game._engine
            mons, mask = _read(lvl, eng)
            for _s in range(steps):
                d = rng.choice(DIRS)
                model = _step(lvl, mons, mask, d)
                eng.step(d)
                actual = _read(lvl, eng)
                if model != actual:
                    bad += 1
                    if bad == 1:
                        print(f"  L{level} MISMATCH dir={d}\n"
                              f"    before {mons} {mask:b}\n"
                              f"    model  {model[0]} {model[1]:b}\n"
                              f"    engine {actual[0]} {actual[1]:b}")
                    break
                mons, mask = actual
                if eng.check_win():
                    break
        total += bad
        if verbose:
            print(f"  L{level}: {'OK' if not bad else f'{bad} MISMATCHES'} "
                  f"({trials} rollouts x {steps} presses)")
    return total


def _plan_report() -> None:
    """Print (and engine-verify) the plan for every level -- the quick "is this
    game still fully solved" check. Populates the disk plan cache as it goes."""
    solver = CakeMonstersSolver()
    game, expert, solvable = solver._ensure(0)
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level:2d}: UNSOLVED")
            continue
        for d in plan:
            eng.step(d)
        exact = bool(expert._opt.get(level))
        print(f"  L{level:2d}: {len(plan):3d} presses  win={eng.check_win()}  "
              f"{'exact' if exact else 'ladder'}")
    print(f"  solvable levels: {len(solvable)}/{game.n_levels}")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        mismatches = selfcheck()
        print(f"selfcheck: {mismatches} mismatches")
        sys.exit(1 if mismatches else 0)
    if "--plans" in sys.argv:
        _plan_report()
        sys.exit(0)
    sys.exit(CakeMonstersSolver.main())
