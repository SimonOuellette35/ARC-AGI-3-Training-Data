"""Generate Phase-1 training data for the PuzzleScript game ps:strata_gems
("STRATA-GEMS", Silvano Sorrentino).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, a native model of them, the two searches that model is solved with,
the deadlock test that makes the second one finish, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_strata_gems",
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
Every expert step carries a set of equally-optimal presses.

The game
--------
A gem is FOUR QUADRANTS and each is either filled or empty, so a gem is a 4-bit
value and its name spells the bits: ``Gem1010`` is yellow top-left + red
bottom-left, ``Gem0001`` is blue bottom-right alone. ``Gem1111`` is a COMPLETE
gem. The win condition is ``No Gem``, and the ``Gem`` or-group is every value
from 0001 to 1110 -- so you win when no INCOMPLETE gem is left on the board, and
a complete one is not a piece any more, it is scenery.

Four rules are the whole game, and every clause of them was measured against the
interpreter rather than read off the .txt:

* ``[ > Player Gem ] -> [ > Player > Gem ]`` -- **the gem under you comes with
  you.** There is no pick-up button and no drop button: the player's cell IS its
  hand, so stepping onto a gem arms it and the next press carries it. Gem1111 is
  not in the or-group, which is why a player standing on a finished gem walks
  away from it freely -- the one way this game ever lets go of anything
  voluntarily.
* ``[ > Player | Gem1111 ] -> cancel`` and ``[ > Player | Wall ] -> cancel`` --
  a complete gem is a WALL you built. It blocks the player permanently and it is
  never removed, so every gem it fences off is lost for good; that is the
  deadlock this game actually has (see `_Board.dead`).
* ``[ > GemA | GemB ] -> [ Background | GemA|B ]`` when ``A & B == 0`` -- two
  gems with no quadrant in common MERGE, the carried one vanishing into the
  target cell. Bits are conserved: a merge is a disjoint union, so the number of
  gems carrying any given quadrant NEVER changes for the whole level, and a level
  is winnable only if those four counts are equal. All thirty are.
* ``[ > GemA | GemB ] -> [ GemA | GemB ]`` when ``A & B != 0`` -- and this one is
  the game's secret. It cancels the GEM's move and says nothing about the
  player's, so the player walks on ALONE: the carried gem is left standing on the
  cell you pushed from and you pick up the one you bumped into. It is the only
  way to put a gem down, it is a SWAP rather than a drop, and the message
  before the game's third level ("Try walking right to see what happens") is the
  game teaching it to you.

Thirty levels, all 7x9, 2 to 48 gems. ACTION5 is dead (the prelude says
``noaction`` and no rule reads it), so the four directions are the whole
vocabulary. Every level number below is an INDEX (0..29); the .txt's own
comments number the same levels from 1.

Everything above is what ``--selfcheck`` measures: a seeded random walk on every
level with `_Board`'s board compared against the engine's grid after EVERY
press, including the presses that do nothing, which is where a collision-layer
mistake hides.

Expert solver
-------------
A NATIVE model of the mechanic above (`_Board`), solved by whichever of two
searches reaches the level -- the thirty levels span three orders of magnitude of
difficulty, from a ten-press corridor to a 48-gem board with two free cells on
it, and no single search covers that range.

  1. **The exact field** (`_Board.field`): a forward BFS over PRIMITIVE presses
     stopped at the depth ``d*`` of the first win, then a backward layer sweep
     for the distance-to-win of every state on a shortest path. Plans from it are
     provably shortest and their optimal-action sets are EXACT -- every press
     whose successor is one step nearer a win -- with no heuristic and no
     inference anywhere. Eleven of the thirty levels fit inside its state cap.

  2. **The macro searches** (`_Board.macro_plan`, `_Board.macro_beam`):
     successors are ``walk over empty cells, then step into a gem`` MACROS, so
     the search depth is the number of INTERACTIONS rather than the number of
     presses. That reformulation is exact rather than a relaxation -- a walk over
     empty cells is a pure relocation of the player and whatever it is carrying,
     so any two shortest walks to the same cell leave identical boards, and a
     shortest press sequence can always be re-cut into shortest walks; the win
     itself always lands on a merge. The heuristic they run under is a GUIDE and
     not a bound, though, so their plans win and are engine-verified but are not
     claimed shortest. The other nineteen levels come from here.

The macro side is a LADDER of A* weights and then beam widths, climbed until a
rung returns a plan that fits the adapter's per-level press budget, and the
ordering is measured rather than assumed: on level 27 weight 4 returns a
130-press plan in a fifth of a second where weight 6 returns a 203-press one, so
a ladder that jumped to a big weight for speed would hand back a plan too long to
record. Whatever a rung returns is then re-cut (`_Board.recut` re-lays every walk
run along a shortest route) and shortened (`_Board.shorten` deletes any block of
presses the plan turns out not to need, verified on the model), which is what
brings the two dense boards to 130 and 195 presses against a 200-press budget.

The deadlock test
-----------------
Both searches prune with `_Board.dead`, and it is what makes the macro search
finish at all on the dense boards. It is two questions, both about things that
can never be undone:

* **Reachability.** Walls and complete gems are the only permanent obstacles, and
  complete gems are only ever CREATED. So the set of cells the player can ever
  stand on is the connected component of its own cell in the graph of
  not-wall-not-Gem1111 cells, and that component only ever shrinks. An incomplete
  gem outside it can never be touched again.
* **Arithmetic.** Bit counts are conserved, but a *partition* into complete gems
  is not: merge the 1000 of one group with the 0100 of another and you are left
  with ``{1100, 0111, 1011}``, whose bit counts are still perfectly balanced and
  which cannot be split into two complete gems at all -- three gems, two groups.
  So the test is a real exact cover: can this multiset of values be partitioned
  into blocks that each union to 1111? It is solved as an integer program over
  the FOURTEEN set partitions of the four quadrants (memoised on the residual
  count vector), which is small because the values, not the gems, are what it
  counts.

Optimal-action sets
-------------------
Exact from the field for the eleven levels that have one: at a state ``d`` presses
from a win, every press whose successor is ``d - 1`` from one. ``--selfcheck``
re-derives every one of those sets by brute force (a fresh depth-bounded BFS from
each successor) and requires an exact match.

For the macro-search levels the plan is not claimed shortest, so neither are the
sets: they say which presses are equally good AS THE PLAN, which is the honest
and still useful answer. A press is classified by what it does -- stepping into a
gem is an INTERACTION and is labelled with itself alone (which gem you take, and
from which side, IS the puzzle), while a step onto an empty cell is a WALK, and a
maximal run of walks is labelled with every direction that keeps it on a shortest
route to the cell the following interaction is launched from. That is sound
rather than inferred: over empty cells a press moves the player and the gem in
its hand and nothing else, so two shortest routes to the same cell reach the same
board in the same number of presses. ``--selfcheck`` holds that claim to the
interpreter by BUILDING each alternative -- take it, re-walk to the same cell,
splice the rest of the plan back on -- and requiring a win at the same length.

40 of the 1580 expert presses across the thirty levels have a second equally
right answer. That is a small number, and it is the game rather than the
labelling: these boards are packed, so most walks are one-cell corridors with
nothing to choose. No step ever ships unlabelled (the
always-emit-optimal-targets rule).

Coverage, and what is NOT claimed
---------------------------------
All 30 levels are solved and every plan is replayed through the real interpreter
to a WIN (``--plans``). Eleven of them are provably shortest (0, 1, 2, 4, 7,
10, 12, 13, 16, 19, 20); the other nineteen are winning plans of unproven
length, and ``--plans`` says which is which rather than letting the distinction
blur. The two densest boards -- 27 and 29, with 42 and 48 gems on seven and two
free cells -- are reached only because
`_Board.dead` prunes them: without it the macro search does not close them at any
weight or beam width this file tries, and with it they come out at 130 and 195
presses.

The 200-press budget is a hard edge rather than a target. `_search` refuses a
plan longer than the adapter's ``_max_steps`` and reports the level as
unsolvable, because `PSAStarSolver.solve_episode` counts a seed only when EVERY
level it considers solvable wins -- an over-long plan would not lose its own
level, it would lose every episode of every seed. Level 29 lands at 195 of
200, so if that board is ever edited this is the first thing to re-check.

Recovery
--------
`supports_recovery` with RESET-mode recovery. The expert re-plans from the LIVE
board rather than replaying a stored path, so an exploration prefix that leaves
the board anywhere is answered from there -- including "this board is now dead",
which a prefix really can produce here (bump two gems together wrong and the
arithmetic test above says so). `record_level` reads that None as "reset", which
is the human flail-then-restart arc this game deserves: its merges are
irreversible.

``--selfcheck`` walks random boards on every level and requires the expert to be
right about both halves: 295 of them re-planned to a WIN when the plan was
replayed through the interpreter, 37 were refused with a PROOF (the field closed
the whole reachable space, or `dead` fired), and none was refused for lack of
budget or answered wrongly.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variable is the frame rotation
(``rotation_k`` in 0..3) with its matching directional action remap. 28 levels x
4 rotations = 112 presentations.

No flips. The mechanic names no axis -- every rule is written with the relative
force ``>`` -- and ``--symmetry`` measures it: every level's plan AND a seeded
random walk replayed at all four rotations, requiring every frame to be the exact
transform of the unrotated one. A mirror is a different matter and is deliberately
not used: a gem's four quadrants are its VALUE, drawn as a 2x2 of colours, so
reflecting the art turns Gem1010 into Gem0101 -- a legal frame of a board that is
not this one. Rotation has exactly the same problem, and it is safe only because
the adapter rotates the WHOLE frame: a turned board is drawn with every gem
turned, so the picture stays self-consistent and the value encoded by the
quadrants is the same permutation everywhere. ``--audit`` is what pins that down.

Rendering
---------
``--audit`` renders every cell composition the game can show -- wall, floor, the
player, each of the fifteen gem values, and the player standing on each of them,
33 in all -- as whole 64x64 frames of a uniform board at the one cell size the
thirty levels use, and requires them pairwise distinct. Whole frames rather than
one cell sliced out of a mixed board: `_render_frame` upscales and centre-pads,
so slicing by ``cell_px`` arithmetic reads the wrong pixels (the ps:explod
lesson). All 33 come out distinct at 7px with no change to the game file: the
gem art is a 2x2 of quadrant colours over a white body and the player is a small
orange square that covers one pixel of each quadrant, never a whole one.

Usage (run from the repo root):
    python solvers/generate_strata_gems_training.py --episodes 200 \
        --out data/training_multi_level/strata_gems

    python solvers/generate_strata_gems_training.py --plans      # level report
    python solvers/generate_strata_gems_training.py --plans --fresh  # ...re-derived
    python solvers/generate_strata_gems_training.py --selfcheck  # model + optimality
    python solvers/generate_strata_gems_training.py --audit      # rendering
    python solvers/generate_strata_gems_training.py --symmetry   # augmentation
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

from arcengine import ActionInput, GameState                          # noqa: E402
from solvers.base_solver import _ID_TO_GAMEACTION                     # noqa: E402
from solvers.common.ps_astar import (PSAStarSolver, PSExpert, Plan,   # noqa: E402
                                     screen_action)
from utils.rotation import inverse_remap_action_full                  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "STRATA-GEMS"

#: Engine directions, in the order ties are broken. ACTION5 is bound to nothing
#: (``noaction`` in the prelude, and no rule has ``action`` on a left-hand side),
#: so it is not a move and branching on it would double every search for nothing.
#: The exploration prefix still presses it -- a live agent has that button --
#: which is why ``--symmetry``'s random walk presses it too.
DIRS: tuple[str, ...] = ("up", "down", "left", "right")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: ``_OPP[di]`` is the index of the direction that undoes ``DIRS[di]``. Used to
#: turn "step into this gem going ``di``" into "and therefore stand here".
_OPP = (1, 0, 3, 2)

#: Disk cache of every level's plan AND its optimal-action sets. Load-bearing
#: here: the macro searches on the dense boards are the whole cost of generation
#: (tens of seconds each, once), and without the file every `parallelize_generator`
#: shard would re-derive all of them. Delete it to re-derive.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "strata_gems_plans.json"

#: States the exact primitive field may hold before it gives up on a level.
#: 400k is ~1s and ~150 MB; it is the line between the ten levels that get a
#: proof and the twenty that get a plan, and raising it does not move that line
#: (the next-easiest level is many orders of magnitude away, not a factor of two).
EXACT_CAP = 400_000

#: The macro search's ladder: ``(weight, node cap)`` rungs, climbed until one
#: returns a plan that fits the press budget. Weight 1 first, so a level A* can
#: close honestly is not handed a padded plan by a greedier rung -- but the
#: heuristic is a guide rather than a bound, so not even that rung is a proof of
#: shortness. The ladder rises in small steps for a measured reason: on the
#: densest board weight 4 returns a 130-press plan in a fifth of a second while
#: weight 6 returns a 203-press one, so skipping straight to a big weight buys
#: speed by handing back a plan that does not fit. (It is level index 27.)
MACRO_LADDER: tuple[tuple[int, int], ...] = ((1, 150_000), (3, 300_000),
                                             (4, 400_000), (6, 400_000),
                                             (20, 400_000))

#: Beam rungs ``(width, depth)`` for the levels no weight reaches.
BEAM_LADDER: tuple[tuple[int, int], ...] = ((300, 120), (1_500, 120),
                                            (6_000, 120))
BEAM_NODE_CAP = 1_500_000

_INF = 1 << 30


# ---------------------------------------------------------------------------
# The quadrant arithmetic
# ---------------------------------------------------------------------------

def _set_partitions(items):
    """Every partition of ``items`` into non-empty blocks, as lists of lists."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for sub in _set_partitions(rest):
        for i in range(len(sub)):
            yield sub[:i] + [[first] + sub[i]] + sub[i + 1:]
        yield [[first]] + sub


#: The FOURTEEN ways a complete gem can be assembled out of two or more
#: incomplete ones, each as a sorted tuple of the values it consumes. (There are
#: fifteen partitions of a four-element set; the one-block one is "a gem that is
#: already complete", which is not made of anything.)
GEM_PARTITIONS: tuple[tuple[int, ...], ...] = tuple(sorted(
    tuple(sorted(sum(block) for block in part))
    for part in _set_partitions([1, 2, 4, 8]) if len(part) > 1))


def _feasible(counts: tuple, _memo={}) -> bool:            # noqa: B006
    """Can this multiset of incomplete gem values be partitioned into complete
    gems? ``counts[v - 1]`` is how many gems have value ``v``.

    An exact cover, solved by always placing the LOWEST value still present:
    that gem has to be in some block, so branching over the partitions that
    contain it is exhaustive and never explores the same cover twice. Memoised
    on the residual vector, which is what makes it cheap -- these boards have a
    handful of distinct values and dozens of gems, so the residuals collide
    constantly.
    """
    if not any(counts):
        return True
    hit = _memo.get(counts)
    if hit is not None:
        return hit
    low = next(i for i, n in enumerate(counts) if n) + 1
    ok = False
    for part in GEM_PARTITIONS:
        if low not in part:
            continue
        rest = list(counts)
        for v in part:
            rest[v - 1] -= 1
        if min(rest) < 0:
            continue
        if _feasible(tuple(rest)):
            ok = True
            break
    _memo[counts] = ok
    return ok


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

#: What a press did, as `_Board.advance` reports it. The searches only care
#: whether it was a MERGE (the one event that changes the arithmetic and can
#: build a permanent wall); the annotator only cares whether it was a WALK (the
#: one press whose route is free).
NOOP, WALK, TAKE, SWAP, MERGE = 0, 1, 2, 3, 4


class _Board:
    """One level's walls, plus the mechanic, plus the two searches.

    Cells are flat ``r * w + c`` indices. A STATE is ``(player, gems)`` where
    ``gems`` is a ``bytes`` of one value per cell: 0 for none, 1..14 for an
    incomplete gem and 15 for a complete one. There is no separate "carried"
    field and there must not be -- the player's cell IS its hand, so
    ``gems[player]`` is what it is holding and the two can never disagree.

    Walls are the only static thing; complete gems are as permanent as walls but
    are created during play, so they live in the state.
    """

    __slots__ = ("h", "w", "wall", "edge", "sd")

    def __init__(self, h: int, w: int, walls):
        self.h, self.w = h, w
        self.wall = bytearray(h * w)
        for i in walls:
            self.wall[i] = 1
        #: ``edge[cell][di]`` is the neighbour cell, or -1 off the board; built
        #: once so the inner loops never do bounds arithmetic.
        self.edge = tuple(
            tuple(
                (r + _DELTA[d][0]) * w + (c + _DELTA[d][1])
                if 0 <= r + _DELTA[d][0] < h and 0 <= c + _DELTA[d][1] < w
                else -1
                for d in DIRS)
            for r, c in (divmod(i, w) for i in range(h * w)))
        #: All-pairs walk distance with the gems taken off the board, for
        #: `heuristic`. Built here because it depends on the walls alone.
        self.sd = self.static_dist()

    # -- the mechanic --------------------------------------------------------
    def advance(self, state, di: int):
        """``(next state, what happened)`` for pressing ``DIRS[di]``.

        The four rules, in the order the interpreter resolves them:

        * the destination is off the board, a wall, or a COMPLETE gem: nothing
          moves at all (a complete gem blocks exactly like a wall, and unlike
          one it was built by the player);
        * the player is empty-handed (bare floor, or standing on a complete gem,
          which is not in the ``Gem`` or-group and so is never carried): it just
          walks, and if it walked onto a gem it is now holding it;
        * it is carrying ``v`` and the destination is empty: the gem comes along;
        * ...and the destination holds ``u``: they MERGE into ``u | v`` if they
          share no quadrant, and otherwise the gem's move alone is cancelled --
          the player walks on, ``v`` is left behind on the cell it was pushed
          from, and ``u`` is what the player is now holding.
        """
        p, gems = state
        q = self.edge[p][di]
        if q < 0 or self.wall[q]:
            return state, NOOP
        u = gems[q]
        if u == 15:
            return state, NOOP
        v = gems[p]
        if v == 0 or v == 15:
            return (q, gems), (TAKE if u else WALK)
        if u == 0:
            g = bytearray(gems)
            g[p] = 0
            g[q] = v
            return (q, bytes(g)), WALK
        if v & u:
            return (q, gems), SWAP           # v stays put; the player walks on
        g = bytearray(gems)
        g[p] = 0
        g[q] = v | u
        return (q, bytes(g)), MERGE

    def step(self, state, di: int):
        return self.advance(state, di)[0]

    def won(self, state) -> bool:
        return not any(1 <= x <= 14 for x in state[1])

    # -- the two permanent facts (see `dead`) --------------------------------
    def component(self, state) -> set:
        """Every cell the player can ever stand on: the connected component of
        its own cell over the cells that are neither wall nor complete gem.

        Complete gems are only ever created, so this set only ever shrinks --
        which is what makes it a sound deadlock test rather than a guess."""
        p, gems = state
        seen = {p}
        queue = deque([p])
        wall, edge = self.wall, self.edge
        while queue:
            cur = queue.popleft()
            for nxt in edge[cur]:
                if nxt >= 0 and not wall[nxt] and gems[nxt] != 15 and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    def dead(self, state) -> bool:
        """True when this board can never be won. Sound, not complete: it says
        no to boards that are provably lost and nothing about the rest.

        Two independent reasons, both permanent (see the module docstring): a
        gem fenced off behind complete gems and walls, and a value multiset that
        no longer partitions into complete gems."""
        gems = state[1]
        loose = [i for i, v in enumerate(gems) if 1 <= v <= 14]
        if not loose:
            return False
        reach = self.component(state)
        counts = [0] * 14
        for i in loose:
            if i not in reach:
                return True
            counts[gems[i] - 1] += 1
        return not _feasible(tuple(counts))

    # -- search 1: the exact primitive field ---------------------------------
    def field(self, state, cap: int = EXACT_CAP):
        """``(distance-to-win, d_star, exhausted)``: presses-to-win for every
        state on a shortest path from ``state``, the length of that path, and
        whether the sweep CLOSED.

        ``dist is None`` means no plan came out, and ``exhausted`` is what
        separates the two ways that happens: True when the sweep enumerated the
        whole reachable space and found no win (a PROOF that this board is lost,
        which the caller can act on -- there is no point running a macro search
        after it), False when it hit ``cap`` and simply does not know.

        Sweep 1 is a forward BFS stopped at the depth of the first win, which
        both fixes ``d_star`` and collects every state within ``d_star`` presses
        of the start, kept in layers. Sweep 2 walks those layers back down,
        taking each state's distance from its successors.

        Reading the successors again in sweep 2 rather than storing a reverse
        edge list in sweep 1 is a memory trade, and it is the right one here: the
        forward sweep is already the memory ceiling that decides which levels get
        a proof at all, and one extra `advance` per edge is far cheaper than a
        second dictionary the size of the first.

        The layer sweep is exact where it is read, and the reason is worth
        stating because it is what the optimal SETS rest on: if ``s`` lies on a
        shortest start-to-win path then its own optimal successor has depth
        exactly ``depth(s) + 1`` (any smaller depth would give a start-to-win
        path shorter than ``d_star``), so the value the sweep computes for ``s``
        is its true distance. States on no shortest path can come out too high or
        be missing entirely; nothing ever asks about those, because the plan and
        `optimal` only compare against ``d_star - i``.
        """
        if self.won(state):
            return {state: 0}, 0, False
        if self.dead(state):
            return None, None, True
        depth = {state: 0}
        layers = [[state]]
        d_star = None
        d = 0
        while True:
            nxt_layer = []
            for s in layers[d]:
                for di in range(4):
                    t, kind = self.advance(s, di)
                    if kind == NOOP or t in depth:
                        continue
                    if self.won(t):
                        depth[t] = d + 1
                        if d_star is None:
                            d_star = d + 1
                        continue
                    # `dead` is only asked after a MERGE: it is the one press
                    # that changes the arithmetic or builds a complete gem, so
                    # nothing else can turn a live board into a lost one, and
                    # asking on every press would cost more than the search.
                    # Dead states are still recorded in `depth` (so they are not
                    # re-derived on every re-reach) but never expanded, and they
                    # never enter `dist`, so sweep 2 reads them as unreachable.
                    if kind == MERGE and self.dead(t):
                        depth[t] = d + 1
                        continue
                    if len(depth) >= cap:
                        return None, None, False
                    depth[t] = d + 1
                    nxt_layer.append(t)
            if d_star is not None:
                break
            if not nxt_layer:
                return None, None, True      # the reachable space closed
            layers.append(nxt_layer)
            d += 1

        dist: dict = {}
        for dd in range(d_star - 1, -1, -1):
            for s in layers[dd]:
                best = None
                for di in range(4):
                    t, kind = self.advance(s, di)
                    if kind == NOOP:
                        continue
                    if self.won(t):
                        cost = 1
                    else:
                        sub = dist.get(t)
                        if sub is None:
                            continue
                        cost = sub + 1
                    if best is None or cost < best:
                        best = cost
                if best is not None:
                    dist[s] = best
        return dist, d_star, False

    def optimal(self, dist, state):
        """``[(direction index, successor), ...]`` for every press on a shortest
        path from ``state``. Ties come out in ``DIRS`` order, which is what makes
        a re-derived plan byte-identical across processes."""
        rest = dist.get(state)
        if not rest:
            return []
        out = []
        for di in range(4):
            t, kind = self.advance(state, di)
            if kind == NOOP:
                continue
            cost = 1 if self.won(t) else (
                dist[t] + 1 if t in dist else None)
            if cost == rest:
                out.append((di, t))
        return out

    def exact(self, state, cap: int = EXACT_CAP) -> tuple:
        """``(shortest plan with EXACT optimal sets, exhausted)``.

        The plan is None when the field could not be built; ``exhausted`` then
        says whether that was a PROOF of unwinnability or merely the state cap.
        See `field`."""
        dist, d_star, exhausted = self.field(state, cap)
        if dist is None:
            return None, exhausted
        presses, optsets = [], []
        cur = state
        for _ in range(d_star):
            best = self.optimal(dist, cur)
            if not best:                     # unreachable: dist[cur] > 0 has one
                return None, False
            presses.append(DIRS[best[0][0]])
            optsets.append([DIRS[di] for di, _ in best])
            cur = best[0][1]
        return Plan(presses, optsets), False

    def solve_exact(self, state, cap: int = EXACT_CAP) -> "Plan | None":
        """`exact` without the proof flag, for the callers that only want the
        plan."""
        return self.exact(state, cap)[0]

    # -- search 2: A* over walk-then-interact MACROS --------------------------
    def walk_tree(self, state) -> dict:
        """``{cell: (previous cell, direction) | None}`` -- shortest walks from
        the player over the cells no gem is standing on.

        The player's own cell is the root whether or not it holds a gem: it is
        where the walk starts, never somewhere the walk has to enter."""
        p, gems = state
        parent: dict = {p: None}
        queue = deque([p])
        wall, edge = self.wall, self.edge
        while queue:
            cur = queue.popleft()
            for di, nxt in enumerate(edge[cur]):
                if nxt < 0 or wall[nxt] or gems[nxt] or nxt in parent:
                    continue
                parent[nxt] = (cur, di)
                queue.append(nxt)
        return parent

    def walk_path(self, state, target) -> list:
        """A shortest walk from the player to ``target`` over empty cells, as
        direction NAMES, or None when ``target`` is not reachable without
        touching a gem. Used only by the reports, to rebuild a run the expert
        took some other way."""
        parent = self.walk_tree(state)
        if target not in parent:
            return None
        out = []
        cell = target
        while parent[cell] is not None:
            cell, di = parent[cell]
            out.append(DIRS[di])
        out.reverse()
        return out

    def macros(self, state) -> list:
        """Every ``walk over empty cells, then step into a gem`` the player can
        start right now, each as a list of primitive direction indices.

        This is the game's whole move set at the level that matters, and the
        reformulation is exact rather than a relaxation: over empty cells a press
        relocates the player and whatever is in its hand and changes nothing
        else, so two shortest walks to the same cell reach the same board in the
        same number of presses, and the only presses that DO anything are the
        ones that enter a gem. A shortest press sequence therefore always
        re-cuts into these macros -- including its last press, since the win can
        only land on a merge."""
        parent = self.walk_tree(state)
        edge = self.edge

        def walk_to(cell):
            out = []
            while parent[cell] is not None:
                cell, di = parent[cell]
                out.append(di)
            out.reverse()
            return out

        out = []
        for cell, v in enumerate(state[1]):
            if not 1 <= v <= 14:
                continue
            for di in range(4):
                stand = edge[cell][_OPP[di]]
                if stand >= 0 and stand in parent:
                    out.append(walk_to(stand) + [di])
        return out

    def merges_left(self, gems) -> int:
        """How many merges any solution still owes: every merge fuses two gems
        into one, and the number of complete gems the loose ones must become is
        fixed (their quadrants are conserved), so this is exact."""
        loose = [v for v in gems if 1 <= v <= 14]
        bits = sum(bin(v).count("1") for v in loose)
        return len(loose) - bits // 4

    def heuristic(self, state, sd) -> int:
        """A GUIDE, not a bound: merges still owed, plus half the distance from
        each gem to the nearest gem it could merge with.

        The halving is because that distance is counted from both ends of every
        pair. ``sd`` is the walk distance with the gems taken off the board, so
        the travel term never over-charges for a detour -- but the merge term and
        the pairing are both optimistic in ways that do not compose into a
        bound, which is why nothing here claims a plan from this search is
        shortest. On the open boards it is a good guide; on the dense ones every
        gem is already beside a partner, the term collapses to a constant, and
        the search is really being carried by `dead` and the beam."""
        gems = state[1]
        cells = [(i, v) for i, v in enumerate(gems) if 1 <= v <= 14]
        if not cells:
            return 0
        total = 0
        for i, v in cells:
            best = _INF
            for j, u in cells:
                if j != i and not (v & u) and sd[i][j] < best:
                    best = sd[i][j]
            if best < _INF:
                total += best
        return self.merges_left(gems) + total // 2

    def static_dist(self) -> list:
        """All-pairs walk distance with every gem taken off the board -- an
        under-estimate of the real walk, built once per level for `heuristic`."""
        n = self.h * self.w
        out = []
        for s in range(n):
            d = [_INF] * n
            if not self.wall[s]:
                d[s] = 0
                queue = deque([s])
                while queue:
                    cur = queue.popleft()
                    for nxt in self.edge[cur]:
                        if nxt >= 0 and not self.wall[nxt] and d[nxt] == _INF:
                            d[nxt] = d[cur] + 1
                            queue.append(nxt)
            out.append(d)
        return out

    def _run_macro(self, state, macro):
        """``(state after the macro, index of the press that won or -1)``.

        The win is tested after EVERY press, not only at the macro's end: a macro
        whose walk happens to pass the winning merge would otherwise keep going
        and the plan would be both longer and, if the extra presses moved
        something, wrong. (It cannot happen as macros are built -- only the last
        press touches a gem -- but the plan is truncated here rather than by
        trusting that.)"""
        cur = state
        for i, di in enumerate(macro):
            cur = self.step(cur, di)
            if self.won(cur):
                return cur, i
        return cur, -1

    def macro_plan(self, state, sd, weight: int, cap: int) -> "list | None":
        """Weighted A* over `macros`, returning a flat list of primitive engine
        directions, or None if the level did not fall inside ``cap`` macro
        expansions.

        ``g`` counts primitive PRESSES, so the cost being minimised is the one
        the agent actually pays. Dedup is on the exact ``(player, gems)`` pair
        rather than on the player's reachable REGION: the region key is the usual
        sokoban canonicalisation and it is unsound for a move-costed search
        (`PSPushExpert`'s docstring calls the error "at most the region's
        diameter", which is a real loss of optimality). It costs nothing here --
        after a macro the player is always standing on the gem it just touched,
        so the exact key barely has more states than the region one."""
        if self.won(state):
            return []
        counter = 0
        pq = [(weight * self.heuristic(state, sd), 0, counter, state, [],
               self.macros(state))]
        best_g = {state: 0}
        nodes = 0
        while pq:
            _f, g, _c, cur, path, macros = heapq.heappop(pq)
            for macro in macros:
                nxt, won_at = self._run_macro(cur, macro)
                if won_at >= 0:
                    return path + macro[:won_at + 1]
                if nxt == cur:
                    continue                 # a walk the board refused
                nodes += 1
                ng = g + len(macro)
                if best_g.get(nxt, _INF) <= ng:
                    continue
                if self.dead(nxt):
                    continue
                best_g[nxt] = ng
                counter += 1
                heapq.heappush(pq, (ng + weight * self.heuristic(nxt, sd), ng,
                                    counter, nxt, path + macro,
                                    self.macros(nxt)))
                if nodes >= cap:
                    return None
        return None

    def macro_beam(self, state, sd, width: int, depth: int,
                   cap: int = BEAM_NODE_CAP) -> "list | None":
        """`macros` explored by a width-capped breadth-first beam, for the boards
        A* cannot reach.

        Same trade as `PSBeamExpert`: A* spends its budget where the heuristic
        points, which is the right call only while the heuristic can tell good
        boards from bad. On the dense levels it cannot -- every gem is already
        beside a partner, so the estimate is flat over the whole middle of a
        solution and A* degenerates into uniform-cost search at a depth where
        that is hopeless. A beam buys breadth at every depth instead and uses the
        heuristic only to decide who survives, which is all a flat one is good
        for. Plans win but wander."""
        if self.won(state):
            return []
        frontier = [(state, [], self.macros(state))]
        seen = {state}
        nodes = 0
        for _ in range(depth):
            kids = []
            for cur, path, macros in frontier:
                for macro in macros:
                    nxt, won_at = self._run_macro(cur, macro)
                    if won_at >= 0:
                        return path + macro[:won_at + 1]
                    if nxt == cur or nxt in seen:
                        continue
                    nodes += 1
                    seen.add(nxt)
                    if self.dead(nxt):
                        continue
                    kids.append((self.heuristic(nxt, sd), len(path) + len(macro),
                                 nxt, path + macro, self.macros(nxt)))
                    if nodes >= cap:
                        return None
            if not kids:
                return None                  # the reachable space closed
            # Rank on (estimate, presses so far) only -- the tuple also carries
            # a bytes-keyed state and two lists, which are not an ordering.
            kids.sort(key=lambda kid: (kid[0], kid[1]))
            frontier = [(s, p, m) for _h, _g, s, p, m in kids[:width]]
        return None

    def recut(self, state, presses) -> list:
        """Re-lay every WALK run of a plan along a shortest route to the cell the
        interaction after it fires from. Same interactions in the same order, so
        the plan still wins; never longer, and usually the same length.

        Two things need it. A macro plan comes out of `macros` with shortest
        walks already, but `shorten` deletes blocks across run boundaries and can
        leave a run wandering; and a BEAM plan wanders by construction. Both then
        lose their tie labels, because `annotate` measures a run against a
        shortest route and falls back to "what the expert did" when the run is
        not on one -- so a plan that was not re-cut trains one arbitrary route as
        the only answer AND is longer than it needs to be."""
        out = []
        cur = state
        i = 0
        while i < len(presses):
            di = DIRS.index(presses[i])
            if self.advance(cur, di)[1] != WALK:
                out.append(presses[i])
                cur = self.step(cur, di)
                i += 1
                continue
            run_start, run_state = i, cur
            while i < len(presses) and self.advance(cur, DIRS.index(presses[i]))[1] == WALK:
                cur = self.step(cur, DIRS.index(presses[i]))
                i += 1
            if i >= len(presses):
                out.extend(presses[run_start:i])      # a trailing walk: keep it
                continue
            bridge = self.walk_path(run_state, cur[0])
            out.extend(bridge if bridge is not None else presses[run_start:i])
        return out

    def shorten(self, state, presses, max_block: int = 40) -> list:
        """Delete any contiguous block of presses the plan does not need, longest
        block first, until nothing more comes out. Returns direction NAMES.

        A macro plan is already tight between interactions -- the walks are
        shortest by construction -- so what this removes is whole INTERACTIONS
        the search took and then undid: a swap that put a gem down and picked it
        straight back up, a detour into a gem that the rest of the plan turned
        out not to need. Each candidate is verified by replaying it on the model
        from ``state``, so a shortened plan is a plan, not a guess.

        Cheap enough to always run (a few seconds on the longest board here) and
        the only thing standing between a beam plan and the adapter's per-level
        press budget."""
        presses = self.recut(state, list(presses))

        def wins(seq) -> bool:
            cur = state
            for name in seq:
                cur = self.step(cur, DIRS.index(name))
                if self.won(cur):
                    return True
            return False

        improved = True
        while improved:
            improved = False
            for length in range(min(len(presses) - 1, max_block), 0, -1):
                for i in range(len(presses) - length + 1):
                    cand = presses[:i] + presses[i + length:]
                    if cand and wins(cand):
                        presses = cand
                        improved = True
                        break
                if improved:
                    break
        # The final re-cut is by construction a plan (same interactions, same
        # states at each of them) but it is cheap to be sure, and a silent
        # non-winning plan here would surface as a mysterious GAME_OVER much
        # later, inside the recorder.
        recut = self.recut(state, presses)
        if wins(recut):
            presses = recut
        # Truncate anything after the press that won.
        cur = state
        for k, name in enumerate(presses):
            cur = self.step(cur, DIRS.index(name))
            if self.won(cur):
                return presses[:k + 1]
        return presses

    # -- optimal-action sets for a plan that is not claimed shortest ----------
    def annotate(self, state, presses) -> list:
        """Per-press optimal sets for a flat ``presses`` list (engine direction
        names in, lists of names out).

        A press is classified by what it DID rather than by a macro boundary the
        flattened plan no longer carries. Stepping into a gem is an INTERACTION
        and is labelled with itself alone: which gem you take, and from which
        side, is the whole puzzle, and the side decides where a swapped gem is
        left standing. A step onto an empty cell is a WALK, and a maximal run of
        walks always ends on the cell the next interaction is launched from, so
        every step of it is labelled with every direction that keeps it on a
        shortest route to that cell -- and that is also asked of the model
        rather than of the distance field alone, so a direction is offered only
        when pressing it really is a bare walk.

        That is sound rather than inferred -- over empty cells a press moves the
        player and the gem in its hand and nothing else, so an alternative
        shortest route reaches the identical board in the identical number of
        presses, and the rest of the plan still applies verbatim. What it does
        NOT claim is that the plan itself is shortest; for the levels where that
        is claimed the sets come from `field` instead and are exact."""
        # One replay, keeping each press's kind and the cell it was taken from.
        steps = []
        cur = state
        for name in presses:
            di = DIRS.index(name)
            nxt, kind = self.advance(cur, di)
            steps.append((kind, cur, di))
            cur = nxt

        out: list = [None] * len(presses)
        i = 0
        while i < len(steps):
            if steps[i][0] != WALK:
                out[i] = [presses[i]]
                i += 1
                continue
            run = i
            while i < len(steps) and steps[i][0] == WALK:
                i += 1
            if i >= len(steps):
                # A trailing walk: nothing after it to aim at, so keep what the
                # expert chose. Plans end on a merge, so this is unreachable.
                for j in range(run, i):
                    out[j] = [presses[j]]
                continue
            stand = steps[i][1][0]           # the cell the interaction fires from
            dist = self._walk_distances(steps[run][1], stand)
            for j in range(run, i):
                here = steps[j][1][0]
                d0 = dist.get(here)
                alts = [] if d0 is None else [
                    DIRS[di] for di in range(4)
                    if dist.get(self.edge[here][di]) == d0 - 1
                    and self.advance(steps[j][1], di)[1] == WALK]
                out[j] = alts if presses[j] in alts else [presses[j]]
        return out

    def _walk_distances(self, state, target) -> dict:
        """BFS distance to ``target`` over the cells the player may walk on for
        the whole of one walk run: no wall, no gem -- plus the run's own starting
        cell, which is free the instant the player steps off it (the gem it may
        be holding travels with it)."""
        p, gems = state
        wall, edge = self.wall, self.edge

        def free(c):
            return c >= 0 and not wall[c] and (not gems[c] or c == p)

        dist = {target: 0}
        queue = deque([target])
        while queue:
            cur = queue.popleft()
            for nxt in edge[cur]:
                if free(nxt) and nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
        return dist


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class StrataGemsExpert(PSExpert):
    """`PSExpert`'s plan memo, level scoping and disk cache around `_Board`.

    The base class keeps all of that; only the strategy underneath changes, the
    way `PSEnumExpert` replaces it. Here the strategy is the two-search ladder --
    the exact field first, then macro A* under a weight ladder, then a beam -- so
    `heuristic` (the base's per-engine-state one) is never called and asserts
    rather than returning a number nothing would use. `_Board.heuristic` is a
    different function and is the one the macro search runs under.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary board -- a recovery prefix, an epsilon detour -- is answered from
    there, including "this board is now dead", which a prefix really can produce
    in a game whose merges are irreversible.
    """

    directions = list(DIRS)
    #: `_key` is the player and the gems, which is canonical WITHIN a level but
    #: not across them: the walls are static per level and are not in the key.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        g = self.g
        self.player_ids = set(self.game._engine._player_indices)
        self.wall_ids = set(g.resolve_object_name("wall"))
        #: ``{object id: 4-bit value}`` for all fifteen gems that exist. Gem0000
        #: is declared but commented out of the legend and appears on no level,
        #: so it is simply absent -- the model would treat it as empty anyway.
        self.gem_value = {}
        for v in range(1, 16):
            for i in g.resolve_object_name(f"gem{v:04b}"):
                self.gem_value[i] = v
        #: `_Board`s by wall layout, not by level index: the sweeps never
        #: rebuild the edge table, and one board serves every state of its level.
        self._boards: dict = {}
        #: ``{level: which rung of the ladder answered}``, for `--plans`. Not
        #: stored in the plan cache: it is documentation, not part of the plan.
        self.tier: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "StrataGemsExpert plans with _Board's own searches; "
            "PSExpert.heuristic is unused")

    def _key(self, eng) -> frozenset:
        """``(row, col, value)`` for the player (value -1) and every gem.

        Built from the MODEL's reading so the key is exactly what the plan
        depends on -- and so the ``plan_cache_path`` signature, which is this key
        written out as triples, cannot drift from it."""
        _board, (p, gems) = self.read(eng)
        w = eng.width
        out = {(i // w, i % w, v) for i, v in enumerate(gems) if v}
        if p is not None:
            out.add((p // w, p % w, -1))
        return frozenset(out)

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, state)`` for the engine's current grid."""
        h, w = eng.height, eng.width
        walls = []
        gems = bytearray(h * w)
        player = None
        gem_value = self.gem_value
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.player_ids:
                    player = i
                for o in cell:
                    v = gem_value.get(o)
                    if v is not None:
                        gems[i] = v
        sig = (h, w, tuple(walls))
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(h, w, walls)
        return board, (player, bytes(gems))

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if state[0] is None:                 # no player on the board
            return None
        if board.won(state):
            return Plan([], [])
        sd = board.sd

        exact, exhausted = board.exact(state)
        if exact is not None:
            self._tier = "field"
            return exact
        if exhausted:
            # The field enumerated the whole reachable space and found no win,
            # or the board is provably lost outright. Either way that is a
            # PROOF, not a budget, so the macro ladder below cannot help and is
            # skipped -- which matters at recovery time, where an exploration
            # prefix that bricks a small level would otherwise pay for the whole
            # ladder before agreeing.
            self._tier = "proved unwinnable"
            return None

        # Climb the ladder until a rung returns a plan that FITS. A rung that
        # returns an over-budget plan is not a stopping point -- the next rung
        # regularly returns a much shorter one (see `MACRO_LADDER`) -- but it is
        # kept, so that if nothing fits the report can say how close it came.
        best = None
        rungs = ([(f"A* w={w}", lambda w=w, c=c: board.macro_plan(state, sd, w, c))
                  for w, c in MACRO_LADDER]
                 + [(f"beam {wd}", lambda wd=wd, dp=dp:
                     board.macro_beam(state, sd, wd, dp))
                    for wd, dp in BEAM_LADDER])
        for tier, run in rungs:
            presses = run()
            if presses is None:
                continue
            names = board.shorten(state, [DIRS[di] for di in presses])
            if best is None or len(names) < len(best[1]):
                best = (tier, names)
            if len(names) <= self.game._max_steps:
                break
        if best is None:
            self._tier = "no rung returned a plan"
            return None
        tier, names = best
        if len(names) > self.game._max_steps:
            # Over the adapter's per-level press budget, so this level cannot be
            # recorded at all. Refusing it here is not a nicety: the adapter ends
            # a level in GAME_OVER at `_max_steps` presses and
            # `PSAStarSolver.solve_episode` counts a seed only when EVERY level
            # it considers solvable wins -- so one over-long plan would not lose
            # that level, it would lose every episode of every seed. Returning
            # None makes `discover_solvable` drop it once, seed-independently.
            self._tier = f"{tier}, {len(names)} presses OVER BUDGET"
            return None
        self._tier = tier
        return Plan(names, board.annotate(state, names))

    def plan(self, eng, level=None):
        """`PSExpert.plan`, plus a note of which rung answered.

        The tier is recorded here rather than inside `_search` because the disk
        cache can serve a plan without running a search at all, and a report that
        silently attributed a cached plan to whichever rung ran last would be
        worse than one that says it does not know."""
        self._tier = "cache"
        found = super().plan(eng, level)
        if level is not None and (level not in self.tier
                                  or self._tier != "cache"):
            self.tier[level] = self._tier
        return found


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class StrataGemsSolver(PSAStarSolver):
    game_id = "puzzlescript_strata_gems"
    game_name = GAME_NAME
    expert_cls = StrataGemsExpert

    #: `games/ps:strata_gems/ps:strata_gems.py` is a plain passthrough (it
    #: constructs the adapter and nothing else), so there is nothing to gain by
    #: routing through it -- but it IS what a live agent is handed, so if that
    #: wrapper ever grows a patch this must be set to ``"ps:strata_gems"``.
    game_module_id = ""

    #: Unused: the expert runs `_Board`'s own searches, whose budgets are
    #: `EXACT_CAP` / `MACRO_LADDER` / `BEAM_LADDER`. Left at the base values so
    #: nothing reads a lie off them.
    weight = 1
    node_cap = 400_000

    #: The longest plan is well under the adapter's own 200-press per-level
    #: budget, which `set_level` resets before the plan starts anyway; the rest
    #: is room for a re-plan after the exploration prefix.
    max_steps = 200

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it fills the disk cache in one pass and makes the
        startup cost visible as startup."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(fresh: bool = False):
    """Solver, adapter, expert and the solvable-level set.

    ``fresh`` detaches the disk plan cache, so a report re-derives every plan
    (and therefore every tier label) instead of reading one back. Slower, and
    what the optimality reports want: a cached plan proves nothing about the
    search that would produce it today."""
    solver = StrataGemsSolver()
    if fresh:
        StrataGemsExpert.plan_cache_path = None
        try:
            game, expert, solvable = solver._ensure(0)
        finally:
            StrataGemsExpert.plan_cache_path = PLAN_CACHE
    else:
        game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _brute(board, state, limit):
    """Presses to a win from ``state``, searched fresh, or None past ``limit``.

    Shares nothing with `_Board.field` but `_Board.advance` itself, and does not
    prune with `dead` -- which is the point: it is the independent answer every
    optimality and every deadlock claim below is checked against."""
    if board.won(state):
        return 0
    seen = {state}
    queue = deque([(state, 0)])
    while queue:
        cur, d = queue.popleft()
        if d >= limit:
            continue
        for di in range(4):
            nxt, kind = board.advance(cur, di)
            if kind == NOOP or nxt in seen:
                continue
            if board.won(nxt):
                return d + 1
            seen.add(nxt)
            queue.append((nxt, d + 1))
    return None


def _plans(verbose: bool = True) -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, both searches and the tie labelling at
    once.

    Shortness is re-established here rather than taken from whichever rung
    happened to answer: `field` is rebuilt for every level, and where it returns
    the plan is shortest AND must have the length the recorded one has. That
    keeps the report honest when the plan comes from the disk cache, which has
    no idea how the plan it stores was found. Pass ``--fresh`` to re-derive the
    plans themselves as well."""
    fresh = "--fresh" in sys.argv
    _solver, game, expert, solvable = _new(fresh)
    eng = game._engine
    total = ties = bad = 0
    proved = []
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        n_gems = sum(1 for v in state[1] if 1 <= v <= 14)
        plan = expert.plan(eng, level)
        if plan is None:
            if verbose:
                print(f"  L{level:2d}: {n_gems:2d} gems -- NO PLAN (dropped, "
                      f"{expert.tier.get(level) or 'no rung returned'})")
            continue
        for direction in plan:
            eng.step(direction)
        won = eng.check_win()
        bad += not won
        exact = board.solve_exact(state)
        note = f"{expert.tier.get(level) or '?'}"
        if exact is not None:
            proved.append(level)
            note = f"{note}, SHORTEST"
            if len(exact) != len(plan):
                note = f"{note} -- but the field says {len(exact)}!"
                bad += 1
        tie_steps = sum(1 for s in plan.optsets if len(s) > 1)
        total += len(plan)
        ties += tie_steps
        room = "ok" if len(plan) <= game._max_steps else "OVER BUDGET"
        bad += room != "ok"
        if verbose:
            print(f"  L{level:2d}: {n_gems:2d} gems  {len(plan):3d} presses  "
                  f"win={won}  ({note})  budget {game._max_steps} {room}  "
                  f"{tie_steps:3d} steps with a tie set")
    print(f"  solvable levels: {len(solvable)}/{game.n_levels} -> {solvable}")
    print(f"  provably shortest: {len(proved)} -> {proved}")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _check_model(game, expert, walk_presses: int, verbose: bool) -> int:
    """`_Board` is the interpreter's, measured rather than argued.

    A seeded random walk on every level, comparing the model's board against the
    engine's grid after EVERY press -- including the presses that do nothing,
    which is where a collision-layer mistake hides. The walk is what measures the
    mechanic listed in the module docstring: it carries gems into walls, into
    complete gems and into gems that overlap (the swap), and it walks off
    finished gems, which no plan ever bothers to do twice."""
    eng = game._engine
    bad = seen_kinds = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        start_component = board.component(state)
        rng = random.Random(f"strata_gems:model:{level}")
        drift = escaped = 0
        kinds = [0] * 5
        for _ in range(walk_presses):
            di = rng.randrange(4)
            eng.step(DIRS[di])
            state, kind = board.advance(state, di)
            kinds[kind] += 1
            _b, live = expert.read(eng)
            if live != state or eng.check_win() != board.won(state):
                drift = 1
                break
            # The reachability half of `dead`: the player can never leave the
            # component it started in, whatever it does to the board.
            if state[0] not in start_component:
                escaped = 1
                break
        bad += drift + escaped
        seen_kinds += 1 if all(kinds) else 0
        if verbose:
            names = ("blocked", "walk", "take", "swap", "merge")
            tally = " ".join(f"{n}={k}" for n, k in zip(names, kinds))
            print(f"  L{level:2d}: model "
                  f"{'MATCHES' if not drift else 'DIVERGES from'} the "
                  f"interpreter over {walk_presses} random presses "
                  f"[{tally}]"
                  f"{'; PLAYER LEFT ITS COMPONENT' if escaped else ''}")
    if verbose:
        print(f"  {seen_kinds}/{game.n_levels} levels exercised all five "
              f"outcomes (blocked / walk / take / swap / merge)")
    return bad


def _check_arithmetic(trials: int, verbose: bool) -> int:
    """`_feasible` against an exhaustive partition search, on random multisets.

    The memoised integer program over `GEM_PARTITIONS` is the load-bearing half
    of `dead`, and it is the half with no runtime evidence behind it -- an
    over-eager answer there would silently declare live boards lost and the
    searches would return "unsolvable" instead of crashing. So it is checked
    against the naive thing: partition the actual multiset of values by brute
    force."""
    def naive(vals) -> bool:
        vals = tuple(sorted(vals))
        if not vals:
            return True
        first, rest = vals[0], list(vals[1:])
        # Grow a block containing the first gem, in every possible way.
        def grow(block, pool):
            if block == 15:
                return naive(pool)
            for i, v in enumerate(pool):
                if not (block & v) and grow(block | v, pool[:i] + pool[i + 1:]):
                    return True
            return False
        return grow(first, rest)

    rng = random.Random("strata_gems:arithmetic")
    bad = 0
    for _ in range(trials):
        vals = [rng.randrange(1, 15) for _ in range(rng.randrange(0, 9))]
        counts = [0] * 14
        for v in vals:
            counts[v - 1] += 1
        if _feasible(tuple(counts)) != naive(vals):
            bad += 1
            print(f"    _feasible disagrees on {sorted(vals)}")
    if verbose:
        print(f"  {trials} random gem multisets: "
              f"{'exact cover AGREES with brute force' if not bad else f'{bad} DISAGREEMENTS'}")
    return bad


def _check_prune(game, expert, verbose: bool) -> int:
    """`dead` changes no answer, on every level whose field can be built twice.

    The forward sweep drops a merge that leaves the board unwinnable, and that is
    only sound if `dead` never fires on a live board. Rather than argue it, the
    field is rebuilt with the prune switched off and both the plan LENGTH and
    every optimal SET are required to match. It is the strongest available check
    because it covers every state the sweep touched, not a sample of them."""
    eng = game._engine
    real_dead = _Board.dead
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        with_prune = board.solve_exact(state)
        if with_prune is None:
            continue                        # no field on this level; nothing to do
        try:
            _Board.dead = lambda self, st: False
            # A larger cap for the unpruned sweep: the prune is what brings some
            # levels under `EXACT_CAP` in the first place, so reusing it would
            # report "no field" as a disagreement.
            without = board.solve_exact(state, cap=3 * EXACT_CAP)
        finally:
            _Board.dead = real_dead
        if without is None:
            if verbose:
                print(f"  L{level:2d}: no field without the prune inside "
                      f"{3 * EXACT_CAP} states -- not comparable")
            continue
        ok = (len(without) == len(with_prune)
              and [sorted(s) for s in without.optsets]
              == [sorted(s) for s in with_prune.optsets])
        bad += not ok
        if verbose:
            print(f"  L{level:2d}: field with and without the deadlock prune "
                  f"{'AGREE' if ok else 'DISAGREE'} "
                  f"({len(with_prune)} presses)")
    return bad


def _check_shortest(game, expert, verbose: bool) -> int:
    """The field's plans are shortest, and its tie sets are exact.

    An independent forward BFS (`_brute`, no field, no prune) must return the
    plan's length; then every step's optimal set is re-derived by a fresh
    depth-bounded BFS from each of the four successors and must match exactly.
    Only the levels the field reached are checked -- the macro-search levels
    make no shortness claim, and `_check_ties` is what holds them to the claim
    they do make."""
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        plan = board.solve_exact(state)
        if plan is None:
            continue
        shortest = _brute(board, state, len(plan) + 1)
        length_ok = shortest == len(plan)
        mismatched = 0
        cur = state
        for i, direction in enumerate(plan):
            rest = len(plan) - i
            truth = []
            for di in range(4):
                nxt, kind = board.advance(cur, di)
                if kind == NOOP:
                    continue
                if board.won(nxt):
                    if rest == 1:
                        truth.append(DIRS[di])
                elif _brute(board, nxt, rest - 1) == rest - 1:
                    truth.append(DIRS[di])
            if sorted(truth) != sorted(plan.optsets[i]):
                mismatched += 1
            cur = board.step(cur, DIRS.index(direction))
        bad += (not length_ok) + mismatched
        if verbose:
            print(f"  L{level:2d}: field {len(plan)} presses vs BFS {shortest}"
                  f" -- {'SHORTEST' if length_ok else 'NOT SHORTEST'}; "
                  f"{len(plan)} tie sets "
                  f"{'match' if not mismatched else f'{mismatched} MISMATCH'}"
                  f" brute force")
    return bad


def _check_ties(game, expert, verbose: bool) -> int:
    """Every press `_Board.annotate` labelled is genuinely as good as the one the
    expert took -- proved by BUILDING the alternative plan and winning with it.

    For each step and each alternative in its optimal set, the plan is spliced:
    take the alternative, walk to the cell the run was heading for by a fresh
    BFS, then continue with the rest of the recorded plan unchanged. The result
    must be the SAME length and must reach a win when replayed through the real
    interpreter. That is exactly the claim the macro-search levels make about
    their sets, and it is checked end to end rather than re-read off the
    annotator that produced them.

    Only those levels. A FIELD level's sets are a different and stronger claim --
    every press that is one step nearer a win, whether or not it is a reroute of
    the same walk -- so the splice does not apply to them and would report a
    failure where there is none. `_check_shortest` is their check, and it is
    exact."""
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            continue
        if board.solve_exact(state) is not None:
            if verbose:
                print(f"  L{level:2d}: field level -- sets checked exactly by "
                      f"_check_shortest, not by splicing")
            continue
        # Re-derive the run structure independently of `annotate`.
        kinds, states = [], []
        cur = state
        for name in plan:
            states.append(cur)
            nxt, kind = board.advance(cur, DIRS.index(name))
            kinds.append(kind)
            cur = nxt
        checked = failed = 0
        for j, alts in enumerate(plan.optsets):
            for alt in alts:
                if alt == plan[j]:
                    continue
                checked += 1
                end = j
                while end < len(kinds) and kinds[end] == WALK:
                    end += 1
                if end >= len(kinds):
                    failed += 1
                    continue
                stand = states[end][0]
                mid = board.step(states[j], DIRS.index(alt))
                bridge = board.walk_path(mid, stand)
                if bridge is None:
                    failed += 1
                    continue
                spliced = (list(plan[:j]) + [alt] + bridge + list(plan[end:]))
                if len(spliced) != len(plan):
                    failed += 1
                    continue
                game.set_level(level)
                for name in spliced:
                    eng.step(name)
                if not eng.check_win():
                    failed += 1
        bad += failed
        if verbose:
            print(f"  L{level:2d}: {checked:3d} alternative press(es) spliced -- "
                  f"{'all win at the same length' if not failed else f'{failed} FAILED'}")
    return bad


def _check_recovery(game, expert, verbose: bool) -> int:
    """The expert answers from ANY board, not just the start.

    This is the one that matters at generation time: the exploration prefix
    really does strand this game (a wrong merge is permanent), so `_search` has
    to be right about arbitrary states. From a handful of boards reached by
    seeded random presses, every plan it returns is replayed through the
    interpreter and must WIN, and every None is separated into "provably dead"
    (`dead` says so, which the checks above have already held to brute force) and
    "gave up" -- the latter is not a bug, it is the search's budget, and
    `record_level` reads either as "reset", but the counts should not drift.

    Sampled thinly on the macro-search levels and thickly on the field ones: a
    random board is a memo MISS, so each sample runs a whole ladder from scratch,
    which is a fifth of a second on a level with a field and can be a minute on
    the dense boards. The code path is identical either way, so the thin sample
    is a smoke test of the same thing rather than of something weaker."""
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        samples = 25 if board.solve_exact(state) is not None else 3
        rng = random.Random(f"strata_gems:recovery:{level}")
        won_after = proved_dead = gave_up = wrong = 0
        for _ in range(samples):
            game.set_level(level)
            for _ in range(rng.randrange(1, 20)):
                eng.step(DIRS[rng.randrange(4)])
                if eng.check_win():
                    break
            if eng.check_win():
                continue
            board, state = expert.read(eng)
            plan = expert._search(eng)
            if plan is None:
                if expert._tier == "proved unwinnable":
                    proved_dead += 1
                else:
                    gave_up += 1
                continue
            for direction in plan:
                eng.step(direction)
            if eng.check_win():
                won_after += 1
            else:
                wrong += 1
        bad += wrong
        if verbose:
            print(f"  L{level:2d}: {samples} random boards -- {won_after} "
                  f"re-planned to a WIN in the interpreter, {proved_dead} "
                  f"proved unwinnable, {gave_up} over budget, {wrong} WRONG")
    return bad


def _selfcheck(verbose: bool = True) -> int:
    """Six things the expert would otherwise be trusted on. See each ``_check_``."""
    _solver, game, expert, _solvable = _new(fresh=True)
    total = 0
    print("-- the model is the interpreter's")
    total += _check_model(game, expert, 600, verbose)
    print("-- the exact-cover arithmetic")
    total += _check_arithmetic(4000, verbose)
    print("-- the deadlock prune changes no answer")
    total += _check_prune(game, expert, verbose)
    print("-- the field's plans are shortest and its tie sets exact")
    total += _check_shortest(game, expert, verbose)
    print("-- every labelled alternative wins at the same length")
    total += _check_ties(game, expert, verbose)
    print("-- the expert re-plans from arbitrary boards")
    total += _check_recovery(game, expert, verbose)
    return total


#: Every cell stack the engine can put on the board. The gem layer holds at most
#: one gem and the player has a layer to itself, so "player standing on a gem" is
#: a real composition and there are 2 x 16 - 1 of them plus the wall.
def _compositions():
    comps = [("wall",), ("background",), ("background", "player")]
    for v in range(1, 16):
        comps.append(("background", f"gem{v:04b}"))
        comps.append(("background", f"gem{v:04b}", "player"))
    return comps


def _comp_name(comp) -> str:
    return "+".join(o for o in comp if o != "background") or "floor"


def _audit(verbose: bool = True) -> int:
    """Two passes over every reachable cell composition.

    **Pass 1 -- distinctness**, at every cell size the levels use (all thirty are
    7x9, so that is one size, 7px, but the list is derived rather than assumed).
    Whole 64x64 frames of uniform boards are compared rather than one cell sliced
    out of a mixed board: `_render_frame` upscales and centre-pads, so slicing by
    ``cell_px`` arithmetic reads the wrong pixels (the ps:explod lesson). Two
    uniform boards render identically iff their cells do.

    This is the pass that matters most for this particular game, because the
    thing the agent has to read off the screen is a FOUR-BIT NUMBER drawn as four
    quadrants of a 5x5 sprite, upscaled to 7px, with a 2x2 player square
    sometimes sitting on top of it. If any two of those sixteen values collided
    the game would be unlearnable and nothing else would report it.

    **Pass 2 -- the turn.** The game is augmented with a frame ROTATION, so every
    board is drawn at one of four turns. A gem's quadrants are its VALUE, so a
    turn permutes values -- Gem1010 turned is Gem0101's art. That is safe only
    because the adapter turns the WHOLE frame: every gem on the board is
    permuted the same way, so a turned board is a consistent picture of a turned
    board, and the four turns of one composition landing on OTHER GEMS' art is
    expected rather than a collision. What would not be safe is a turn landing on
    a composition of a different KIND (a gem reading as the wall, or as the
    player), so that is what this checks. It runs on square boards, because a
    turn of a non-square board also moves the letterbox and every comparison
    would pass for the wrong reason.
    """
    from adapters.puzzlescript_adapter import _render_frame            # noqa: PLC0415

    _solver, game, _expert, _solvable = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = _compositions()

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    def shoot(h, w, comp):
        eng.height, eng.width = h, w
        eng.grid = [[{idx[o] for o in comp} for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        return np.asarray(_render_frame(eng, g)).copy()

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {c: shoot(h, w, c) for c in comps}
        clashes = [(a, b) for a, b in itertools.combinations(comps, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        if verbose:
            print(f"  {h}x{w} (cell {min(64 // h, 64 // w)}px, {len(levels)} "
                  f"levels): {len(shots)} compositions, "
                  f"{'all distinct' if not clashes else 'CLASH'}")
        for a, b in clashes:
            print(f"      IDENTICAL: {_comp_name(a)}  ==  {_comp_name(b)}")

    def kind(comp) -> str:
        if "wall" in comp:
            return "wall"
        gem = [o for o in comp if o.startswith("gem")]
        return ("gem+player" if gem and "player" in comp else
                "gem" if gem else "player" if "player" in comp else "floor")

    for cell in sorted({min(64 // h, 64 // w) for h, w in sizes}):
        n = 64 // cell
        shots = {c: shoot(n, n, c) for c in comps}
        clashes = [(a, b, k) for a, b in itertools.permutations(comps, 2)
                   for k in range(1, 4)
                   if kind(a) != kind(b)
                   and np.array_equal(np.ascontiguousarray(
                       np.rot90(shots[a], k=k)), shots[b])]
        bad += len(clashes)
        if verbose:
            print(f"  cell {cell}px ({n}x{n}): {len(clashes)} composition(s) "
                  f"whose turn is another KIND's art")
        for a, b, k in clashes:
            print(f"      {_comp_name(a)} turned {k} == {_comp_name(b)}")
    return bad


def _symmetry(walk_presses: int = 80, verbose: bool = True) -> int:
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unaugmented ones.

    This is the evidence for the rotation augmentation. The structural argument
    is in the module docstring -- every rule is written with the relative force
    ``>`` and names no axis -- but Gobble Rush's chirality hid inside exactly
    that kind of argument, so it is measured. Both the PLANS and a seeded random
    walk are replayed: the walk reaches boards a plan never visits (a gem shoved
    flat against a wall, a swap chain, a merge the plan would never make) and it
    presses the unbound ACTION key as well."""
    solver, game, expert, _solvable = _new()
    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        plans[level] = expert.plan(game._engine, level)

    def drive(g, level, presses):
        g.set_level(level)
        out = [np.asarray(g._current_frame)]
        for act in presses:
            fd = g.perform_action(ActionInput(id=act))
            out.append(np.asarray(fd.frame[-1] if fd.frame else g._current_frame))
        return out

    def transform(frame, k, hflip, vflip):
        if k:
            frame = np.rot90(frame, k=k)
        if hflip:
            frame = np.fliplr(frame)
        if vflip:
            frame = np.flipud(frame)
        return np.ascontiguousarray(frame)

    ref, seen, bad = {}, set(), 0
    # Every (seed, level) draws its own rotation (`set_level` seeds on
    # ``seed + level``), so thirty levels cover the group within a couple of
    # seeds; the extra seeds are there to re-draw each turn from a different
    # (seed, level) pair rather than to find new ones. Kept small on purpose --
    # this replays whole plans through the ADAPTER, which renders a 64x64 frame
    # per press, and the plans here run to 195.
    for seed in range(400):
        if len(seen) == 4 and seed > 10:
            break
        g = solver.make_game(seed)
        for level in range(g.n_levels):
            if plans[level] is None:
                continue
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            rng = random.Random(f"strata_gems:symmetry:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(walk_presses)]
            for tag, presses in (
                    ("plan", [screen_action(d, *k) for d in plans[level]]),
                    ("walk", [inverse_remap_action_full(a, *k) for a in walk])):
                frames = drive(g, level, presses)
                if tag == "plan" and g._state != GameState.WIN:
                    print(f"    seed {seed} L{level} {k}: plan did not win")
                    bad += 1
                key = (level, tag)
                if key not in ref:
                    if k != (0, False, False):
                        continue          # nothing to compare against yet
                    ref[key] = frames
                    continue
                if any(not np.array_equal(transform(a, *k), b)
                       for a, b in zip(ref[key], frames)):
                    print(f"    seed {seed} L{level} {k}: {tag} frames are not "
                          f"the transform of the unaugmented ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
    return bad


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
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
    sys.exit(StrataGemsSolver.main())
