"""Generate Phase-1 training data for the PuzzleScript game ps:autumn.

The harness -- the trajectory recorder, the rotation contract and the BaseSolver
plumbing -- lives in `solvers/common/ps_astar.py`, shared with the other ps:
generators. This file is the game-specific part: a re-implemented MODEL of
Autumn's rules, a search over that model, and an engine verification pass that
throws away any plan the real interpreter does not actually win with.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema::

    {"game_id": "puzzlescript_autumn",
     "levels": [{"level_id": 0,
                 "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette
                 "actions":      [a_0, ..., a_{T-1}]},      # length T
                ...]}

actions[0] is the RESET that produced obs[0]; actions[i] for i>=1 is the *screen*
action (post rotation-remap) that took the agent from obs[i-1] to obs[i].

The game
========
Autumn, a leaf-spirit, colours the forest floor. The grey ground starts pale
(`lit`) and turns orange (`it`) permanently -- and the exit door opens only once
NOTHING pale is left and no bird is still on the board. So the puzzle is a
COVERAGE puzzle, and the mechanic that makes it one is how ground gets coloured:

  * every turn, the ground cells CONNECTED to Autumn (4-neighbour, through empty
    ground only) all turn orange at once, and
  * a fallen leaf-pile permanently colours the cell it sits on.

Everything else on the board blocks that connectivity -- trees, leaf-piles, the
river, the nests, the closed door, birds and bunnies -- so the pale cells are
exactly the ones Autumn's region has never reached.

Mechanics, as reverse-engineered from the rules and then fuzz-checked against
the interpreter (see "Why a model", below):

  * TREES (green blobs) are one-shot crates. Walking into a tree converts it to
    a leaf-pile *in place* and shoves that pile one cell on; Autumn ends up on
    the tree's old cell. So one push colours TWO cells -- the tree's (Autumn
    stands there) and the destination's (the pile covers it) -- and often opens
    a whole region behind it. If the far cell is blocked the whole TURN is
    cancelled: nothing moves, nothing ticks.
  * LEAF-PILES are permanent scenery. The rule that shoves a new pile along is
    keyed on a one-tick `start` marker which the same tick consumes, so a pile
    can never be pushed again -- every push is irreversible, and a pile dropped
    in the wrong place can seal a region off for good.
  * The RIVER cancels Autumn's move, but a tree pushed into it is consumed and
    leaves a BRIDGE. Autumn can walk the bridge; connectivity cannot cross it,
    so the far bank has to be coloured by standing on it.
  * BIRDS fall one cell per tick until something stops them and vanish off the
    bottom row ("south for winter"). While ANY bird is on the board the door
    stays shut, so every bird must be flushed out. Autumn also SHOVES birds one
    cell in any direction, which is the only way to move a bird out of a column
    that is walled off further down -- levels 8 and 10 are unwinnable without it.
  * BUNNIES step one cell horizontally toward Autumn, but only when they share
    her row and she is at least two cells away, and they ignore walls in
    between. They block connectivity, so a bunny has to be lured off any cell it
    is squatting on. A bunny shoved into the river drowns and the level RESTARTS
    -- the search treats that as a dead state and never plans through it.
  * Level 3 is a cut-scene: it ships a `summer` object whose countdown wins the
    level on the first key press, whatever it is.

Why a model, not engine A*
==========================
The family's default is to search the real interpreter (`PSExpert`). Here that
is hopeless: one `step` costs 2.4 ms on level 0 and 25 ms on level 17, while
solutions run 30-90 moves and the search needs six figures of nodes. The model
in this file steps in ~15 us, i.e. three orders of magnitude cheaper, which is
the difference between searching these levels and not.

The risk a model carries is silent drift from the engine, and it is paid for
twice:

  * a differential fuzz test (random play, all levels, full state compared after
    every single step) drove the implementation -- it is what caught the two
    rules a reading of the source missed, namely that Autumn pushes BIRDS
    (without which the bird levels look unwinnable) and that the bunny rule runs
    to a FIXED POINT, so a queue of bunnies shuffles up as a block. It ships as
    ``--verify-model`` (currently 0 mismatches over 40 x 60 random steps on each
    of the 19 playable levels) and should be re-run after any change to the
    adapter or to ``Autumn.txt``; and
  * every plan is REPLAYED ON THE REAL INTERPRETER inside `AutumnExpert.plan`
    before it is ever returned. A plan the engine does not win with is discarded
    as if the search had failed, so model drift can only ever cost coverage --
    it can never emit a broken demonstration.

The expert
==========
A* over the model, from the level's start state, with

  * successors = the five primitives PLUS ``walk to a push cell, then shove``
    macros for every tree and every bird. The macros give the search the reach
    to plan pushes dozens of moves apart; the primitives are still needed
    because the exact walk matters (dodging round a bunny through a side
    corridor, or standing still to let one come to you), which a macro-only
    search cannot express.
  * ``g`` = primitive moves, so plans are short in the units the agent pays.
  * ``h`` = ``lit_w * (pale cells + flock cost) + relaxed distance to the door``,
    where the flock cost is a 0-1 BFS per bird: falling is free, a shove or a
    tree in the way costs one. A bird with no wall-free route to the bottom row
    makes the state DEAD, which is the prune that makes the nest levels
    tractable -- without it the search cheerfully colours the whole board and
    only then notices it has bricked a bird in.
  * a two-rung ``lit_w`` ladder, see `LADDER`. Plans stay engine-verified at
    every rung.

17 of the 20 levels solve, including the cut-scene. The three that do not --
14, 17 and 18 -- are the big tree mazes, and they are NOT unwinnable: a greedy
probe drives 17 down to 8 pale cells and 18 to 3 pale cells and 2 birds, so
their solutions are simply deeper than this search reaches (see `LADDER` for what
was tried). They are skipped by `ps_astar.discover_solvable` rather than emitted
as broken data. ``--report`` prints the per-level plan lengths, and the plans are
cached in ``data/autumn_plans.json`` so the searches are paid once ever rather
than once per process -- which is what matters under
`parallelize_generator.py`, where every shard would otherwise redo them.

Augmentation
============
The engine state after reset is identical for every seed (levels are fixed ASCII
maps); only the presentation varies per (seed, level) -- the frame rotation
(``rotation_k`` in {0,1,2,3}) and the recolour of the surfaces listed in
``PuzzleScriptAdapter._recolor_surfaces``. The plan is therefore seed-independent:
solved once per level, cached, and replayed per seed with that seed's
rotation-remapped screen actions and recoloured frames.

Usage (run from the repo root)::

    python solvers/generate_autumn_training.py --report
    python solvers/generate_autumn_training.py --episodes 200 \
        --out data/training_multi_level/autumn
"""

from __future__ import annotations

import heapq
import json
import os
import sys
from collections import deque
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from solvers.common.ps_astar import (                      # noqa: E402
    PSAStarSolver, PSExpert, restore, snapshot,
)

GAME_NAME = "Autumn"

#: Engine direction -> (dr, dc). ``action`` is a pure WAIT in this game: no rule
#: mentions it, yet the turn still ticks, so it is how the expert stands still
#: while a bunny walks over to it.
_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
    "action": (0, 0),
}
_D4 = {k: v for k, v in _DELTA.items() if k != "action"}
MOVES = ["up", "down", "left", "right", "action"]

#: Seed-independent plans, in the repo's usual ``data/<game>_plans.json`` shape.
#: Each entry records the start layout it was solved from and is ignored if that
#: no longer matches, so editing the level in the .txt cannot serve a stale plan.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "autumn_plans.json"

#: ``(lit_w, weight, node budget)`` rungs, tried in order until one wins.
#:
#: Rung 1 is the near-optimal one and solves 16 of the 17 levels this expert
#: gets; rung 2 leans on the pale count and is there for level 19, where the
#: heuristic goes flat across the middle of the solution (a run of pushes that
#: rearrange piles without colouring much) and plain A* degenerates. Deliberately
#: only two rungs: levels 14, 17 and 18 resist EVERY setting tried -- 1.2M nodes
#: at rung 1, greedy (weight 20/50), bigger pale weights, and a width-1200 macro
#: beam -- so further rungs would only burn a minute apiece to fail again.
LADDER: tuple[tuple[int, int, int], ...] = ((1, 1, 200_000), (4, 1, 200_000))


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class Static:
    """Level geometry no rule ever changes."""

    __slots__ = ("H", "W", "walls", "water0", "door", "nests", "south")

    def __init__(self, H, W, walls, water0, door, nests, south):
        self.H, self.W = H, W
        self.walls = walls        # block cells
        self.water0 = water0      # cells that STARTED as river (bridged or not)
        self.door = door          # cell index, or None on the cut-scene level
        self.nests = nests
        self.south = south        # rows a bird migrates out of (see `south_rows`)


def south_rows(H: int, W: int, walls: frozenset) -> frozenset:
    """The rows a bird vanishes from ("south for winter"), i.e. the game's
    `vert` marker, worked out from the three rules that build it.

    `vert` is stamped everywhere, then stripped from any cell with a non-block
    below it, then stripped from any ROW that is not wholly `vert`. What survives
    is the bottom row -- nothing is below it -- plus every row whose entire
    suffix is solid wall. That is usually just the last row, but level 8's board
    ends in two wall rows, and getting this wrong makes its bird look permanently
    trapped and the level unwinnable.
    """
    out = {H - 1}
    for r in range(H - 2, -1, -1):
        if all((rr * W + c) in walls
               for rr in range(r + 1, H) for c in range(W)):
            out.add(r)
        else:
            break
    return frozenset(out)


class State:
    """Everything a rule can change, hashable through `key`."""

    __slots__ = ("pos", "trees", "fells", "bridges", "lit", "bunnies", "birds",
                 "starts", "dead")

    def __init__(self, pos, trees, fells, bridges, lit, bunnies, birds, starts,
                 dead=False):
        self.pos = pos
        self.trees = trees        # bloomed
        self.fells = fells        # leaf-piles (permanent)
        self.bridges = bridges    # former river cells: walkable, non-conducting
        self.lit = lit            # ground still PALE
        self.bunnies = bunnies
        self.birds = birds
        self.starts = starts      # live `start` markers (see `Model.step`)
        self.dead = dead          # a bunny drowned -> the level restarts

    def key(self):
        return (self.pos, self.trees, self.fells, self.bridges, self.lit,
                self.bunnies, self.birds, self.dead)


class Model:
    """One turn of Autumn, in the interpreter's rule order.

    Fuzz-checked step-for-step against `PuzzleScriptAdapter` on every level; see
    the module docstring.
    """

    def __init__(self, static: Static):
        self.s = static

    def door_open(self, s: State) -> bool:
        """The door is open exactly while nothing is pale and no bird is left.
        It is re-evaluated at the END of every turn, so this reads the state the
        turn STARTED from -- which is what decides whether Autumn may step in."""
        return not s.lit and not s.birds

    def won(self, s: State) -> bool:
        return (not s.dead and self.s.door is not None
                and s.pos == self.s.door and not s.lit and not s.birds)

    def step(self, s: State, d: str) -> State:
        st = self.s
        W, H = st.W, st.H
        if s.dead:
            return s
        pos = s.pos
        trees, fells, bridges = s.trees, s.fells, s.bridges
        lit, bunnies, birds, starts = s.lit, s.bunnies, s.birds, s.starts
        water = st.water0 - bridges
        door_blocks = st.door is not None and not self.door_open(s)

        # ---- main phase: Autumn's move, and whatever she shoves -------------
        if d != "action":
            dr, dc = _DELTA[d]
            r, c = divmod(pos, W)
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                tgt = nr * W + nc
                fr, fc = nr + dr, nc + dc
                far = fr * W + fc if (0 <= fr < H and 0 <= fc < W) else None
                if tgt in birds:
                    # Autumn shoves birds one cell in any direction.
                    if far is not None and not (
                            far in trees or far in st.walls or far in water
                            or far in st.nests or far in fells or far in birds
                            or far in bunnies or far == st.door):
                        birds = tuple(sorted(set(birds) - {tgt} | {far}))
                        pos = tgt
                elif tgt in trees:
                    if far is None:
                        return s                       # cancel
                    occupied = (far in trees or far in fells or far in birds
                                or far in bunnies
                                or (door_blocks and far == st.door))
                    if far in water and not occupied:
                        # Into the river: the tree is consumed, leaving a bridge.
                        trees = trees - {tgt}
                        bridges = bridges | {far}
                        pos = tgt
                    elif (not occupied and far not in st.walls
                          and far not in water and far not in st.nests
                          and far != st.door):
                        # The `start` the conversion rule stamps on the tree's
                        # cell is consumed in the same tick by the rule that
                        # shoves the new pile along, so it leaves no trace --
                        # which is what makes leaf-piles permanent.
                        trees = trees - {tgt}
                        fells = fells | {far}
                        pos = tgt
                    else:
                        return s                       # cancel: nothing ticks
                elif tgt in fells and tgt in starts:
                    # The one exception: a pile shoved onto a cell that still
                    # carries a `start` (only ever Autumn's spawn) matches the
                    # shove rule again. The rule fires -- consuming the start --
                    # even when the move is then blocked.
                    starts = starts - {tgt}
                    if far is not None and not (
                            far in st.walls or far in fells or far in trees
                            or far in birds or far in bunnies
                            or far in st.nests
                            or (door_blocks and far == st.door)):
                        fells = (fells - {tgt}) | {far}
                        pos = tgt
                elif (tgt not in st.walls and tgt not in water
                      and tgt not in fells and tgt not in birds
                      and tgt not in bunnies and tgt not in st.nests
                      and not (door_blocks and tgt == st.door)):
                    pos = tgt
                # else: bumped -- Autumn stays put, but the turn still ticks

        # ---- late phase ----------------------------------------------------
        lit = lit - fells                     # a pile colours the cell under it
        lit = lit - {pos}

        # Bunnies: one step toward Autumn, at most once per turn.
        pr, pc = divmod(pos, W)
        if bunnies:
            fixed = st.walls | trees | fells | set(birds)
            occupied = set(bunnies)
            movers = [b for b in bunnies
                      if b // W == pr and abs(pc - b % W) >= 2]
            for b in movers:
                a = b + (1 if pc > b % W else -1)
                if a not in fixed and a not in occupied and a in water:
                    return State(pos, trees, fells, bridges, lit, bunnies,
                                 birds, starts, dead=True)      # drowned
            # The move rule runs to a FIXED POINT, so a bunny queued behind
            # another steps into the cell that one just vacated.
            pending = list(movers)
            moved = True
            while moved:
                moved = False
                rest = []
                for b in sorted(pending, key=lambda x: abs(pc - x % W)):
                    a = b + (1 if pc > b % W else -1)
                    if a in fixed or a in occupied:
                        rest.append(b)
                        continue
                    occupied.discard(b)
                    occupied.add(a)
                    moved = True
                pending = rest
            bunnies = tuple(sorted(occupied))

        # Birds: the turn's `again` loop drops them until nothing moves, and any
        # that reach the bottom row have migrated south and are gone.
        if birds:
            blk = st.walls | trees | fells | set(bunnies) | {pos}
            if door_blocks:
                blk = blk | {st.door}
            cur = list(birds)
            while True:
                moved = False
                occ = set(cur)
                for b in sorted(cur, reverse=True):
                    below = b + W
                    if below // W < H and below not in blk and below not in occ:
                        occ.discard(b)
                        occ.add(below)
                        moved = True
                cur = [b for b in occ if b // W not in st.south]
                if not moved:
                    break
            birds = tuple(sorted(cur))

        # Colouring. The flood is wiped on any tick where a bird flew, but the
        # again-loop's last iteration always has a settled flock, so it lands
        # exactly once per turn -- after the birds have finished falling.
        if lit:
            stop = (st.walls | water | bridges | st.nests | trees | fells
                    | set(birds) | set(bunnies))
            if door_blocks:
                stop = stop | {st.door}
            region = {pos}
            stack = [pos]
            while stack:
                cur = stack.pop()
                r, c = divmod(cur, W)
                for ddr, ddc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + ddr, c + ddc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    n = nr * W + nc
                    if n in region or n in stop:
                        continue
                    region.add(n)
                    stack.append(n)
            lit = lit - region

        return State(pos, trees, fells, bridges, lit, bunnies, birds, starts)


def read_state(eng, game) -> tuple[Static, State]:
    """Build ``(Static, State)`` from the interpreter's live grid.

    Read rather than derived from the .txt so the model always starts from the
    engine's own post-``run_rules_on_level_start`` state -- the tick that seeds
    the ground, spawns each nest's single bird and lets it fall onto whatever is
    below it. That tick is fiddly (its `again` loop is what the adapter's
    `load_level` comment is about) and there is no reason to re-derive it.
    """
    N = game.obj_name_to_idx
    W, H = eng.width, eng.height
    fell_ids = {N['fell' + ch] for ch in 'abcdef'}
    walls, water0, nests = set(), set(), set()
    trees, fells, bridges, lit, starts = set(), set(), set(), set(), set()
    bunnies, birds = [], []
    door = pos = None
    for r in range(H):
        for c in range(W):
            i = r * W + c
            cell = eng.grid[r][c]
            if N['block'] in cell:
                walls.add(i)
            if N['water'] in cell:
                water0.add(i)
            if N['hor'] in cell or N['ver'] in cell:
                bridges.add(i)
                water0.add(i)          # a bridge is a river cell, minus the water
            if N['nest'] in cell:
                nests.add(i)
            if N['door'] in cell or N['nodoor'] in cell:
                door = i
            if N['autumn'] in cell:
                pos = i
            if N['bloomed'] in cell:
                trees.add(i)
            if cell & fell_ids:
                fells.add(i)
            if N['bunnyl'] in cell or N['bunnyr'] in cell:
                bunnies.append(i)
            if N['bird'] in cell:
                birds.append(i)
            if N['start'] in cell:
                starts.add(i)
            if N['lit'] in cell:
                lit.add(i)
    static = Static(H, W, frozenset(walls), frozenset(water0), door,
                    frozenset(nests), south_rows(H, W, frozenset(walls)))
    state = State(pos, frozenset(trees), frozenset(fells), frozenset(bridges),
                  frozenset(lit), tuple(sorted(bunnies)), tuple(sorted(birds)),
                  frozenset(starts))
    return static, state


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    """A* over `Model`, branching on primitives plus walk-and-shove macros."""

    def __init__(self, model: Model, node_cap: int, weight: int, lit_w: int):
        self.m = model
        self.st = model.s
        self.node_cap = node_cap
        self.weight = weight
        self.lit_w = lit_w
        self._relax = self._relaxed_dist()

    # -- geometry ----------------------------------------------------------
    def _relaxed_dist(self) -> dict:
        """Distance to the door over everything that is neither wall nor river.
        Trees and piles are ignored: a lower bound on the walk still to come."""
        st = self.st
        if st.door is None:
            return {}
        dist = {st.door: 0}
        q = deque([st.door])
        while q:
            cur = q.popleft()
            r, c = divmod(cur, st.W)
            for dr, dc in _D4.values():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < st.H and 0 <= nc < st.W):
                    continue
                n = nr * st.W + nc
                if n in dist or n in st.walls or n in st.water0:
                    continue
                dist[n] = dist[cur] + 1
                q.append(n)
        return dist

    def walk_tree(self, s: State) -> dict:
        """``{cell: (prev, direction) | None}`` over the cells Autumn can stand
        on right now -- the BFS both the macros and their walks come from."""
        st = self.st
        water = st.water0 - s.bridges
        blocked = (st.walls | water | st.nests | s.trees | s.fells
                   | set(s.birds) | set(s.bunnies))
        if st.door is not None and not self.m.door_open(s):
            blocked = blocked | {st.door}
        parent = {s.pos: None}
        q = deque([s.pos])
        while q:
            cur = q.popleft()
            r, c = divmod(cur, st.W)
            for d, (dr, dc) in _D4.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < st.H and 0 <= nc < st.W):
                    continue
                n = nr * st.W + nc
                if n in parent or n in blocked:
                    continue
                parent[n] = (cur, d)
                q.append(n)
        return parent

    @staticmethod
    def path_to(parent: dict, cell: int) -> list[str]:
        out = []
        while parent[cell] is not None:
            cell, d = parent[cell]
            out.append(d)
        out.reverse()
        return out

    def macros(self, s: State) -> list[list[str]]:
        """``walk to the shove cell, then shove`` for every tree and bird, plus
        the walk that ends on the door once it is open."""
        st = self.st
        parent = self.walk_tree(s)
        out = []
        for piece in list(s.trees) + list(s.birds):
            pr, pc = divmod(piece, st.W)
            for d, (dr, dc) in _D4.items():
                sr, sc = pr - dr, pc - dc
                if not (0 <= sr < st.H and 0 <= sc < st.W):
                    continue
                stand = sr * st.W + sc
                if stand in parent:
                    out.append(self.path_to(parent, stand) + [d])
        if st.door is not None and self.m.door_open(s) and st.door in parent:
            out.append(self.path_to(parent, st.door))
        return [mac for mac in out if mac]

    # -- heuristic ---------------------------------------------------------
    def flock_cost(self, s: State):
        """Cost of getting every bird off the board, or ``None`` if one never
        can -- which makes the state dead, since the door stays shut while any
        bird remains.

        A bird leaves only by dropping into the south rows (`south_rows`).
        Falling is free; a shove sideways or upward costs Autumn a move, and a
        tree in the way costs the push that clears it. Walls, piles and the door
        are permanent,
        so a bird with no such route is stuck for good. Deliberately optimistic
        (it ignores whether Autumn can get behind the bird), so it prunes only
        genuinely dead states.
        """
        st = self.st
        H, W = st.H, st.W
        hard = st.walls | s.fells
        if st.door is not None:
            hard = hard | {st.door}
        soft = hard | (st.water0 - s.bridges) | st.nests | s.trees
        total = 0
        for b in s.birds:
            dist = {b: 0}
            dq = deque([b])
            best = None
            while dq:
                cur = dq.popleft()
                if cur // W in st.south:
                    best = dist[cur]
                    break
                d0 = dist[cur]
                r, c = divmod(cur, W)
                for dr, dc in _D4.values():
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    n = nr * W + nc
                    if n in hard:
                        continue
                    if dr == 1:
                        cost = 1 if n in s.trees else 0
                    elif n in soft:
                        continue
                    else:
                        cost = 1
                    if dist.get(n, 1 << 30) <= d0 + cost:
                        continue
                    dist[n] = d0 + cost
                    (dq.append if cost else dq.appendleft)(n)
            if best is None:
                return None
            total += best
        return total

    def hopeless(self, s: State) -> bool:
        """True when some pale cell can never be coloured again.

        A pale cell only ever colours by Autumn's region reaching it or by a
        pile landing on it, and a pile only lands one push away from somewhere
        she stands -- so both need the cell to be in the region she can still
        open up. That region is the flood over everything except the PERMANENT
        obstacles: walls, leaf-piles, nests, and river cells no tree can be
        shoved into any more (trees only ever vanish, and a pushed tree becomes a
        pile, so no cell ever gains a tree neighbour). Trees, bunnies, birds and
        the door are all transient, hence passable here.

        Irreversible pushes make "sealed a region off for good" the dominant way
        to lose this game, and a sealed region is invisible to the heuristic --
        the pale count keeps looking finishable. This is the prune that catches
        it, one flood per node.
        """
        st = self.st
        if not s.lit:
            return False
        H, W = st.H, st.W
        water = st.water0 - s.bridges
        bridgeable = set()
        for w in water:
            wr, wc = divmod(w, W)
            for dr, dc in _D4.values():
                nr, nc = wr + dr, wc + dc
                if 0 <= nr < H and 0 <= nc < W and nr * W + nc in s.trees:
                    bridgeable.add(w)
                    break
        stop = st.walls | s.fells | st.nests | (water - bridgeable)
        seen = {s.pos}
        stack = [s.pos]
        while stack:
            cur = stack.pop()
            r, c = divmod(cur, W)
            for dr, dc in _D4.values():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                n = nr * W + nc
                if n in seen or n in stop:
                    continue
                seen.add(n)
                stack.append(n)
        return not s.lit <= seen

    def heuristic(self, s: State):
        if s.dead or self.hopeless(s):
            return None
        flock = self.flock_cost(s) if s.birds else 0
        if flock is None:
            return None
        far = 4 * self.st.H * self.st.W
        return (self.lit_w * (len(s.lit) + flock + len(s.birds))
                + self._relax.get(s.pos, far))

    # -- search ------------------------------------------------------------
    def search(self, s0: State) -> list[str] | None:
        m = self.m
        if m.won(s0):
            return []
        h0 = self.heuristic(s0)
        if h0 is None:
            return None
        counter = 0
        heap = [(self.weight * h0, 0, counter, s0, [])]
        best = {s0.key(): 0}
        nodes = 0
        while heap:
            _f, g, _c, s, path = heapq.heappop(heap)
            if g > best.get(s.key(), -1):
                continue                      # stale duplicate, already improved
            for edge in [[mv] for mv in MOVES] + self.macros(s):
                cur = s
                won = -1
                for i, mv in enumerate(edge):
                    cur = m.step(cur, mv)
                    if cur.dead:
                        break
                    if m.won(cur):
                        # Check after every primitive, not just at the macro's
                        # end: a macro that wins partway and keeps walking would
                        # step straight off the door again.
                        won = i
                        break
                nodes += 1
                if won >= 0:
                    return path + edge[:won + 1]
                if cur.dead:
                    continue
                k = cur.key()
                ng = g + len(edge)
                if best.get(k, 1 << 30) <= ng:
                    continue
                h = self.heuristic(cur)
                if h is None:
                    continue                  # dead end (a bird is walled in)
                best[k] = ng
                counter += 1
                heapq.heappush(heap, (ng + self.weight * h, ng, counter, cur,
                                      path + edge))
            if nodes >= self.node_cap:
                return None
        return None


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class AutumnExpert(PSExpert):
    """Plan on the model, then prove the plan on the real interpreter.

    `PSExpert`'s A* is replaced wholesale (see the module docstring on why the
    interpreter is too slow to search); what is inherited is the plan memo, the
    snapshot/restore discipline and the `record_level` contract that a plan is a
    flat list of engine directions. The base's ``node_cap`` / ``weight`` are
    unused -- `LADDER` carries both, per rung.
    """

    scope_by_level = True

    def setup(self) -> None:
        self._disk = self._load_disk()

    # -- disk cache --------------------------------------------------------
    @staticmethod
    def _signature(static: Static, state: State) -> list:
        """Cheap fingerprint of a level's start layout. A cached plan is only
        served when this still matches, so editing Autumn.txt cannot silently
        replay a plan for a level that no longer exists."""
        return [static.H, static.W, state.pos, sorted(static.walls),
                sorted(static.water0), static.door, sorted(state.trees),
                sorted(state.bunnies), sorted(state.birds)]

    def _load_disk(self) -> dict:
        """``{level: {"start": signature, "plan": "<digits>"}}`` -- the plan is
        one digit per move, indexing `MOVES`. Empty if unreadable: a cache that
        cannot be parsed is a miss, never a crash."""
        try:
            raw = json.loads(PLAN_CACHE.read_text())
            return {int(k): v for k, v in raw.items()}
        except Exception:                                      # noqa: BLE001
            return {}

    def _save_disk(self) -> None:
        """Write atomically: shards started together (`parallelize_generator`)
        would otherwise interleave into a truncated file."""
        try:
            PLAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = PLAN_CACHE.with_suffix(f".json.tmp{os.getpid()}")
            tmp.write_text(json.dumps(
                {str(k): self._disk[k] for k in sorted(self._disk)}, indent=1))
            os.replace(tmp, PLAN_CACHE)
        except OSError:
            pass                                              # cache is optional

    # -- planning ----------------------------------------------------------
    def plan(self, eng, level: int | None = None) -> list | None:
        key = (level, self._key(eng))
        if key in self.cache:
            return self.cache[key]
        sol = self._solve(eng, level)
        self.cache[key] = sol
        return sol

    def _solve(self, eng, level: int | None) -> list | None:
        static, state = read_state(eng, self.g)
        if static.door is None:
            # Level 3 is a cut-scene: a `summer` object counts down to a win, so
            # the first press of anything ends it.
            return self._verify(eng, ["action"])

        cached = self._disk.get(level)
        sig = self._signature(static, state)
        if cached is not None and cached.get("start") == sig:
            raw = cached.get("plan")
            if raw is None:
                # A level the search could not win. Only trusted while the
                # LADDER is the one that failed: a cached "no" must never be
                # what stops a strengthened search from being tried.
                if cached.get("ladder") == list(map(list, LADDER)):
                    return None
            else:
                checked = self._verify(eng, [MOVES[int(ch)] for ch in raw])
                if checked is not None:
                    return checked
                # The cache disagrees with the interpreter -- search afresh.

        model = Model(static)
        for lit_w, weight, cap in LADDER:
            raw = Planner(model, cap, weight, lit_w).search(state)
            if raw is None:
                continue
            checked = self._verify(eng, raw)
            if checked is not None:
                self._remember(level, sig, checked)
                return checked
        self._remember(level, sig, None)
        return None

    def _remember(self, level, sig, plan) -> None:
        if level is None:
            return
        entry = {"start": sig,
                 "plan": (None if plan is None
                          else "".join(str(MOVES.index(d)) for d in plan))}
        if plan is None:
            entry["ladder"] = list(map(list, LADDER))
        self._disk[level] = entry
        self._save_disk()

    def _verify(self, eng, plan: list[str]) -> list[str] | None:
        """Replay ``plan`` on the real interpreter; return it truncated at the
        winning step, or ``None`` if the engine does not win with it. This is
        what keeps model drift from ever reaching the corpus.

        Leaves the engine exactly as it was -- including the win/restart FLAGS,
        which `ps_astar.restore` does not touch. `step` clears them on entry, so
        they only matter to a `check_win` asked before the next step, which is
        precisely what `record_level` does on its first iteration: leaving a
        winning flag behind there would end the recording before it took a single
        action."""
        snap = snapshot(eng)
        flags = (eng._rule_win, eng._rule_restart)
        won = -1
        for i, d in enumerate(plan):
            eng.step(d)
            if eng.check_win():
                won = i
                break
        restore(eng, snap)
        eng._rule_win, eng._rule_restart = flags
        return plan[:won + 1] if won >= 0 else None

    def heuristic(self, eng) -> int:            # pragma: no cover - unused
        raise NotImplementedError("AutumnExpert plans on the model, not the engine")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

#: Process-wide adapter + plan cache + solvable-level set. `test_datagen.py`
#: builds a fresh solver per run, which would otherwise re-pay every search.
_SHARED: dict = {}


class AutumnSolver(PSAStarSolver):
    game_id = "puzzlescript_autumn"
    game_name = GAME_NAME
    expert_cls = AutumnExpert

    #: Plans run to ~90 moves on the big tree mazes, plus the exploration prefix.
    max_steps = 400

    def _ensure(self, seed: int):
        if self._game is None and "game" in _SHARED:
            self._game = _SHARED["game"]
            self._expert = _SHARED["expert"]
            self._solvable = _SHARED["solvable"]
        game, expert, solvable = super()._ensure(seed)
        _SHARED.update(game=game, expert=expert, solvable=solvable)
        return game, expert, solvable


def _engine_state(eng, game) -> tuple:
    """The slice of the interpreter's grid the model claims to reproduce."""
    N = game.obj_name_to_idx
    W = eng.width
    fell_ids = {N['fell' + ch] for ch in 'abcdef'}
    out = {k: set() for k in ("trees", "fells", "bridges", "lit", "bunnies",
                              "birds", "starts")}
    pos = None
    dead = False
    for r in range(eng.height):
        for c in range(W):
            i = r * W + c
            cell = eng.grid[r][c]
            if N['autumn'] in cell:
                pos = i
            if N['bloomed'] in cell:
                out["trees"].add(i)
            if cell & fell_ids:
                out["fells"].add(i)
            if N['hor'] in cell or N['ver'] in cell:
                out["bridges"].add(i)
            if N['lit'] in cell:
                out["lit"].add(i)
            if N['bunnyl'] in cell or N['bunnyr'] in cell:
                out["bunnies"].add(i)
            if N['bird'] in cell:
                out["birds"].add(i)
            if N['start'] in cell:
                out["starts"].add(i)
            if N['lose'] in cell:
                dead = True
    return tuple(sorted(out[k]) for k in sorted(out)) + (pos, dead)


def _model_state(s: State) -> tuple:
    out = {"trees": s.trees, "fells": s.fells, "bridges": s.bridges,
           "lit": s.lit, "bunnies": s.bunnies, "birds": s.birds,
           "starts": s.starts}
    return tuple(sorted(out[k]) for k in sorted(out)) + (s.pos, s.dead)


def _verify_model(episodes: int = 40, steps: int = 60) -> int:
    """Differential test: play randomly and compare the model to the
    interpreter after EVERY step, on every level.

    This is the test that built `Model` -- it is what caught the bird-shoving
    rule and the bunny fixed point -- and it is in the repo rather than in a
    scratch file because the model's fidelity is the whole safety argument for
    planning off the engine. Re-run it after any change to the adapter or to
    ``Autumn.txt``. Expected result: zero mismatches.
    """
    import random

    from adapters.puzzlescript_adapter import PuzzleScriptAdapter

    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    bad = 0
    for level in range(game.n_levels):
        for ep in range(episodes):
            rng = random.Random(level * 1000 + ep)
            game.set_level(level)
            eng = game._engine
            static, s = read_state(eng, game._game)
            if static.door is None:
                continue                       # the cut-scene level ends at once
            model = Model(static)
            for t in range(steps):
                d = rng.choice(MOVES)
                eng.step(d)
                s = model.step(s, d)
                if s.dead:
                    break                      # the engine restarts; stop here
                if (_engine_state(eng, game._game) != _model_state(s)
                        or eng.check_win() != model.won(s)):
                    print(f"  MISMATCH level {level} episode {ep} step {t} "
                          f"({d})")
                    bad += 1
                    break
                if eng.check_win():
                    break
        print(f"  level {level:2d}: checked {episodes} episodes")
    print("model matches the interpreter" if not bad
          else f"{bad} MISMATCHING episodes")
    return 1 if bad else 0


def _report() -> int:
    """Print per-level plan lengths -- the coverage check for this game."""
    solver = AutumnSolver()
    game, expert, solvable = solver._ensure(0)
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(game._engine, level)
        if plan is None:
            print(f"  level {level:2d}: unsolved")
        else:
            total += len(plan)
            print(f"  level {level:2d}: {len(plan):3d} moves")
    print(f"{len(solvable)}/{game.n_levels} levels solved, {total} moves total")
    return 0


if __name__ == "__main__":
    if "--report" in sys.argv:
        sys.exit(_report())
    if "--verify-model" in sys.argv:
        sys.exit(_verify_model())
    sys.exit(AutumnSolver.main())
