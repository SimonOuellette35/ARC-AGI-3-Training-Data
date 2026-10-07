"""Generate Phase-1 training data for the PuzzleScript game ps:dang_im_huge
("Dang I'm Huge" by Guilherme Tows -- a sokoban in which the player, and the
crates, switch between a 1x1 and a 2x2 body by stepping on a pair of teleport
pads).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the `BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: a
native re-implementation of the mechanic (`_Board`), a macro A* / beam over it
(`_astar` / `_beam` with `_Heur`), the interpreter certification that makes
planning on a model safe, the per-step optimal-action sets, and the disk plan
cache.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_dang_im_huge",
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
Four arrow keys, no ACTION (the file declares ``noaction``). Every level is won
by covering all of its purple targets with blue crates -- ``all Target on
Pushable``, so a target is satisfied by any crate cell, whether that cell
belongs to a small crate or to one quarter of a big one.

Both the player and the crates come in two sizes. A SMALL body is one cell; a
BIG body is a 2x2 square held together by the four quadrant objects
``Player1..4`` / ``Block1..4`` (1 = top-left, 2 = top-right, 3 = bottom-left,
4 = bottom-right), which the three growth rules at the top of the file keep
complete. Size is not chosen, it is a place: each level (all but the first)
carries ONE small pad and ONE big 2x2 pad, and the late rules swap a body
between them --

  * a BIG player standing exactly on the big pad is replaced by a SMALL player
    on the small pad, and vice versa (growing also requires the big pad to be
    empty of crates);
  * a BIG crate sitting exactly on the big pad becomes a SMALL crate on the
    small pad, and a small crate pushed onto the small pad becomes a big crate
    on the big pad.

A swap switches every pad OFF, and the pads only come back ON once all five pad
cells are clear of players and crates -- so a body that lands on a pad and stays
there jams the mechanism until it moves off. That is the one piece of state in
this game that is NOT a position, and `_Board` carries it as a flag.

The push rules, in the order the interpreter applies them:

  * ``rigid [ > PlayerBig | BlockBig ]`` -- a big player shoves a big crate, all
    four quadrants together. The crate needs only ONE quadrant against the
    player's leading edge, so a big crate offset by a row from the player is
    still pushable.
  * ``[ > PlayerBig | Block | Block | no Solid ]`` -- a big player pushes a LINE
    OF TWO small crates. This is the move that unlocks level 1, and the only way
    to plug a one-cell gap while standing clear of it.
  * ``[ > Player | Block | no Solid ]`` -- anybody pushes one small crate.
  * a small player can never move a big crate, and no crate ever pushes another.

Anything else -- a wall ahead, a crate with no room behind it, a big crate whose
destination is not clear -- stops the player, and the remaining two rules
(``[ MOVING Player ] [ STATIONARY Player ]`` and ``[ STATIONARY Player ]
[ MOVING Pushable ]``) make that all-or-nothing: if any part of the player is
blocked, NOTHING moves and the turn is a no-op.

The interpreter fix this game needed
------------------------------------
That last paragraph is the author's intent, and on puzzlescript.net it is what
happens, because PuzzleScript re-runs the whole rule loop after a RIGID group is
refused: on the second pass the crate is ``Stationary`` and the two cancel rules
fire. `PuzzleScriptAdapter`'s interpreter cancels a failed rigid group during
movement resolution and never re-runs the rules, so those rules saw a still
-``Moving`` crate, declined to fire, and the player quadrants whose own way was
clear walked out from under the ones that were blocked. The 2x2 body TORE -- and
the growth rules then regrew the missing quadrants from the pieces, so the board
ended up with FIVE player objects. Reproducible in 18 random presses on level 0.

Fixing the interpreter would mean giving it PuzzleScript's rigid backoff, which
changes the dynamics of the twenty other games in ``data/puzzlescript_games``
that carry ``rigid`` rules -- a shared-adapter change with no way to check what
it does to games nobody has looked at. The fix is in the game file instead: a
torn body is a state no rule of this game can produce, and the outcome the
author's rules encode for it is "nothing moves at all", so
``data/puzzlescript_games/Dang_I'm_Huge.txt`` now opens its late section with
twelve guards of the form ``late right [ Player1 | no Player2 ] -> cancel``
(both diagonals of both bodies), which revert the whole turn. Nothing else in
the file's logic was touched.

Expert solver
-------------
`_Board` is a native re-implementation of the mechanic -- state is
``(player size, player cell, small crates, big crates, pads-on)`` -- and
``--fuzz`` asserts it against the interpreter by comparing that FULL state, plus
the win flag, after every single press. Each rollout replays a prefix of the
level's expert plan and then walks randomly: ``--fuzz 120 160`` is 120 rollouts
x 160 presses x 8 levels with 0 mismatches, having exercised 2141 player
resizes, 141 crate resizes, 4431 pad flips, 10137 pushes and 61 wins. The plan
prefix is what makes those numbers mean anything -- see `_fuzz`. Every plan is
then replayed through the real interpreter before it is accepted (`_certify`),
so a modelling slip costs a level, never a recorded trajectory that does not
win.

The interpreter is fast enough to search here (0.3-1.5 ms/press, so unlike
CrateBlob a blackbox search is not hopeless) but not fast enough: a primitive
`PSExpert` A* over the engine did not close level 0 inside a 300k-step budget,
which is already minutes. The model is two orders of magnitude faster per press,
which is what lets the ladder below finish the whole game in ~3.5 minutes.

`_Board.successors` is the shape of the search. From a state it runs a BFS over
the presses that change NOTHING but where the player is standing (and the pad
flag), and every press that does anything else -- a push, a resize, a teleport --
is returned as a MACRO: the walk that got there plus that press. So the search
branches on decisions rather than on steps and ``g`` counts primitive presses.
Note the BFS must be over the real dynamics and not over a free-space map:
stepping onto a pad is a TELEPORT, and walking off one can re-arm the pads and
resize a crate that was parked on the other.

States are keyed EXACTLY, not by the sokoban "same pieces, same player region"
merge, because that merge is unsound in this game: the walk graph is directed
(nothing but a swap turns the pads back off, so a state with them down reaches
one with them up and never the reverse), and the usual ``min(reachable)``
canonical id then merges states with different futures. It is not a small
effect -- level 4 is 145 macro expansions with an exact key and was still
unsolved after 400_000 with a region key. Little is lost by keying exactly: with
macros, every node is a state immediately after a push or a teleport, so the
player's cell is already pinned by the move that got there.

`_Heur` is the sokoban push-distance table -- a reverse BFS from each target
over push edges (a crate reaches ``X`` from ``X-d`` only when ``X-d`` and
``X-2d`` are both on the board and not walls, which every push in this game
satisfies, including the two-crate line and the big-crate shove) -- summed over
a greedy one-crate-per-target matching. Two details are this game's:

  * the tables carry the pad TELEPORT as an edge of cost 1 in both directions,
    so a crate that has to grow to reach its target is not scored as unreachable
    and the search is not steered away from the pads;
  * a matching that runs out of crates does NOT declare the state dead, it
    reuses one (up to four times), because a crate that grows covers four
    targets at once. Level 2 is exactly that -- one crate, a 2x2 of targets --
    and the consume-only version of this heuristic pruned its start state.

Straight distance is useless here: the boards are corridors and one-cell shafts,
and the shortest route for a crate is regularly through the pads and out the
other side of the map.

Levels
------
All 8 levels solve, 493 presses in total, six of them on the first (weight-1)
rung. The two that need a weighted search are the two big ones:

  * Level 6 is an EXACT PACKING -- ten crates, ten targets, and nothing on that
    board can ever grow, because its small pad sits at the end of a one-cell
    shaft whose only approach is the cell a crate would have to be pushed from,
    so the player can never be behind a crate that is going onto it. Every crate
    has to land on a target and any crate parked wrong loses the level.
  * Level 7 is the biggest board and the only one where the win is set up by
    SHRINKING a crate: two of its three targets start covered, and the third
    sits beside the small pad, so the move is to shove the big crate onto the
    big pad and collect the small crate it becomes.

Optimal-action sets
-------------------
Most presses in a plan are the player WALKING to the cell it will push from, and
a walk's order is free: any interleaving of the two axes that stays on a shortest
route to the same state costs the same and leaves an identical board.
`_optimal_sets` replays the plan on the model, cuts it into maximal walk runs,
and labels every press of a run with each direction that keeps it on a shortest
route to the state the following push is taken from. Pushes are labelled with
themselves -- which crate to shove where is the puzzle, and a sibling push is a
different plan. No step ever ships unlabelled (the always-emit-optimal-targets
rule).

The distance behind that labelling is measured over the walk graph itself
(``(size, cell, pads-on)`` under the run's fixed crates), not over free space,
for the same reason `successors` is: a route that crosses a pad is not a walk.
And because that graph is directed, the field is a reverse BFS over edges
collected forwards -- BFSing outwards from the goal would answer a different
question.

Worth knowing what this actually yields: 328 of the 493 presses are walks and
only 25 of them have a real tie. That is the boards, not a gap in the analysis --
these levels are one- and two-cell corridors joined by shafts, and a big player
needs a 2x2 of clear floor to stand in at all, so most of the time there is
exactly one way round. The ties that do exist sit in the open rooms every
trajectory crosses.

The render adaptation
---------------------
One sprite change in ``data/puzzlescript_games/Dang_I'm_Huge.txt``, and it is
not cosmetic. The pads were drawn with their OUTER corners cut, and every body
that can stand on a pad -- a crate, the player, and each quadrant of a 2x2 one --
is transparent at exactly those corners and opaque everywhere else. So an
occupied pad rendered as bare floor: the player standing on the small pad was
pixel-identical to the player standing anywhere else, and whether the pads were
ARMED -- the one piece of this game's state that is not a position -- was
unreadable the moment anything stood on one. The pads are now solid squares, so
they show through a body at its corners the way a target does.

``--audit`` is the regression test: it renders every cell COMPOSITION the game
can produce (bare, and each body on each pad state and on a target) at every
cell size the levels use, and asserts they are pairwise distinct. It allows one
deliberate exception, ``_PAD_CLASSES``: a quadrant of the big pad renders the
same square as the small pad, because painting the outer corner is the entire
point of the fix. The two pads are told apart by EXTENT -- one green cell against
a 2x2 green square -- which is the same thing that distinguishes a small crate
from a big one.

Everything else in the palette was already safe: crate blue (9), player yellow
(11), wall red (8), target purple (15), pad armed dark green (14) and disarmed
dark red (13) are six distinct ARC indices, and a crate covering a target still
shows the target's purple through its own transparent corners.

Augmentation
------------
Rotation AND both flips: the game is in `PuzzleScriptAdapter._FLIP_GAMES`, so a
(seed, level) draws a ``rotation_k`` in {0,1,2,3} and an independent horizontal
and vertical mirror, each with the matching directional action remap. 8 levels x
16 presentations = 128.

It qualifies because it is gravity-free with screen-relative input and a win
(``all Target on Pushable``) that is positional in no way. It DOES carry
absolute-direction rules, but every one of them -- the three growth rules and
the eight rigid propagation rules -- only asserts the internal geometry of a 2x2
body, which a mirror or a rotation carries along with the body it is drawn on.
Its chiral sprites are the quadrants, and under a mirror each maps inside its
own class or onto nothing (``Player1 <-> Player2`` and ``Player3 <-> Player4``
horizontally; the pads are symmetric squares now that they are solid). No mirror
lands one object's art on another's, which is the test that matters.

No colour augmentation -- crate blue, target purple and the armed/disarmed pads
are exactly what the frames have to teach apart.

Usage (run from the repo root):
    python solvers/generate_dang_im_huge_training.py --episodes 200 \
        --out data/training_multi_level/dang_im_huge

    python solvers/generate_dang_im_huge_training.py --plans   # per-level report
    python solvers/generate_dang_im_huge_training.py --fuzz    # model vs engine
    python solvers/generate_dang_im_huge_training.py --audit   # render check
"""

from __future__ import annotations

import heapq
import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import PuzzleScriptAdapter        # noqa: E402
from solvers.common.ps_astar import (PSAStarSolver, PSExpert, Plan,  # noqa: E402
                                     restore, snapshot)

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Dang_I'm_Huge"

#: Disk cache of each level's start plan AND its optimal-action sets. The
#: searches are seed-independent, so without a file on disk every shard
#: `parallelize_generator` starts would re-derive all of them. Delete to
#: re-derive; cold that is ~1 minute, warm a run starts in about a second.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "dang_im_huge_plans.json"

_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}
DIRS = ("up", "down", "left", "right")

#: Heuristic value standing for "this state cannot reach a win". Finite so the
#: search stays complete, large enough that such a node is never expanded.
BIG = 9999


def _quad(tl: tuple[int, int]) -> tuple:
    """The four cells of a 2x2 body whose top-left corner is ``tl``."""
    r, c = tl
    return ((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1))


# ---------------------------------------------------------------------------
# Native model of the mechanic
# ---------------------------------------------------------------------------

class _Board:
    """One level's static scenery plus the game's dynamics.

    A state is ``(size, cell, smalls, bigs, pads_on)`` -- ``size`` 0 for a 1x1
    player and 1 for a 2x2 one, ``cell`` its top-left corner, ``smalls`` the
    frozenset of one-cell crates, ``bigs`` the frozenset of TOP-LEFT corners of
    2x2 crates, and ``pads_on`` whether the teleport pads are armed. Walls,
    targets and the two pad positions never change, so they live here.
    """

    def __init__(self, eng, game):
        idx = game.obj_name_to_idx
        grid = eng.grid
        self.h, self.w = len(grid), len(grid[0])
        self.idx = idx
        walls, targets = set(), set()
        self.sp = self.bp = None          # small pad cell, big pad top-left
        for r in range(self.h):
            for c, cell in enumerate(grid[r]):
                if idx["wall"] in cell:
                    walls.add((r, c))
                if idx["target"] in cell:
                    targets.add((r, c))
                if idx["zap1"] in cell or idx["zap1off"] in cell:
                    self.bp = (r, c)
                if idx["zapsmall"] in cell or idx["zapsmalloff"] in cell:
                    self.sp = (r, c)
        self.walls = frozenset(walls)
        self.targets = frozenset(targets)
        self.bpq = _quad(self.bp) if self.bp is not None else ()
        #: static signature, so one expert can hold a board per level
        self.sig = (self.h, self.w, self.walls, self.targets, self.sp, self.bp)

    # -- reading the interpreter ------------------------------------------
    def read(self, eng) -> tuple:
        """The model state of the interpreter's current grid."""
        idx = self.idx
        smalls, bigs = set(), set()
        size = cell = None
        on = True
        for r in range(self.h):
            for c, objs in enumerate(eng.grid[r]):
                if idx["block"] in objs:
                    smalls.add((r, c))
                if idx["block1"] in objs:
                    bigs.add((r, c))
                if idx["playersmall"] in objs:
                    size, cell = 0, (r, c)
                if idx["player1"] in objs:
                    size, cell = 1, (r, c)
                if idx["zapsmalloff"] in objs or idx["zap1off"] in objs:
                    on = False
        return (size, cell, frozenset(smalls), frozenset(bigs), on)

    def body(self, size, cell) -> tuple:
        return (cell,) if size == 0 else _quad(cell)

    def won(self, st) -> bool:
        occ = set(st[2])
        for tl in st[3]:
            occ.update(_quad(tl))
        return self.targets <= occ

    # -- dynamics ----------------------------------------------------------
    def step(self, st, d: str) -> tuple:
        """One press. Returns the settled state, or ``st`` itself when the turn
        is a no-op (which is what every blocked move is in this game)."""
        size, cell, smalls, bigs, on = st
        dr, dc = _DELTA[d]
        body = self.body(size, cell)
        here = set(body)
        big_at = {}
        for tl in bigs:
            for cc in _quad(tl):
                big_at[cc] = tl
        walls, h, w = self.walls, self.h, self.w

        def solid(c2) -> bool:
            r, c = c2
            if not (0 <= r < h and 0 <= c < w):
                return True
            return c2 in walls or c2 in smalls or c2 in big_at

        mv_small, mv_big = set(), set()
        for lead in body:
            t = (lead[0] + dr, lead[1] + dc)
            if t in here:
                continue                      # not on the leading edge
            r, c = t
            if not (0 <= r < h and 0 <= c < w) or t in walls:
                return st
            if t in smalls:
                t2 = (r + dr, c + dc)
                if not solid(t2):
                    mv_small.add(t)           # [ > Player | Block | no Solid ]
                elif size == 1 and t2 in smalls:
                    t3 = (t2[0] + dr, t2[1] + dc)
                    if solid(t3):
                        return st
                    mv_small.add(t)           # [ > PlayerBig | Block | Block | ]
                    mv_small.add(t2)
                else:
                    return st
            elif t in big_at:
                if size == 0:                 # a small player cannot shove one
                    return st
                tl = big_at[t]
                own = set(_quad(tl))
                for cc in own:
                    dest = (cc[0] + dr, cc[1] + dc)
                    if dest in own:
                        continue
                    r2, c2 = dest
                    if (not (0 <= r2 < h and 0 <= c2 < w) or dest in walls
                            or dest in smalls
                            or big_at.get(dest, tl) != tl):
                        return st
                mv_big.add(tl)

        smalls = (smalls - mv_small) | {(r + dr, c + dc) for r, c in mv_small}
        bigs = (bigs - mv_big) | {(r + dr, c + dc) for r, c in mv_big}
        return self._late(size, (cell[0] + dr, cell[1] + dc), smalls, bigs, on)

    def _late(self, size, cell, smalls, bigs, on) -> tuple:
        """The pad rules, in the interpreter's order. At most one swap fires per
        turn (each one switches the pads off), and the re-arm rule is tested
        last, so it can never re-trigger a swap in the same turn."""
        sp, bp = self.sp, self.bp
        if sp is not None:
            body = set(self.body(size, cell))

            def crates_in(cells) -> bool:
                for cc in cells:
                    if cc in smalls:
                        return True
                for tl in bigs:
                    if set(_quad(tl)) & set(cells):
                        return True
                return False

            if on:
                if size == 1 and cell == bp:
                    # The shrink rule carries no "empty" guard on the small pad,
                    # exactly as the author wrote it.
                    size, cell, on = 0, sp, False
                elif size == 0 and cell == sp and not crates_in(self.bpq):
                    size, cell, on = 1, bp, False
                elif bp in bigs and sp not in body and not crates_in((sp,)):
                    bigs = bigs - {bp}
                    smalls = smalls | {sp}
                    on = False
                elif (sp in smalls and not crates_in(self.bpq)
                        and not (body & set(self.bpq))):
                    smalls = smalls - {sp}
                    bigs = bigs | {bp}
                    on = False
            if not on:
                allc = set(self.bpq) | {sp}
                if not crates_in(allc) and not (body & allc):
                    on = True
        return (size, cell, frozenset(smalls), frozenset(bigs), on)

    # -- macro successors --------------------------------------------------
    def _walk_graph(self, st) -> tuple:
        """``(parent tree, edges, outcomes)`` -- the walk states reachable from
        ``st`` as a BFS tree and as an edge list, plus ``{resulting state:
        (walk state it was taken from, press)}`` for every non-walk outcome met
        on the way.

        A WALK is a press that moves nothing but the player -- the pad flag may
        follow it. The BFS has to run the real `step`: walking onto a pad
        TELEPORTS, and walking off one can re-arm the pads and resize a crate
        parked on the other, so neither is a walk and neither can be read off a
        free-space map.

        NOTE the walk graph is DIRECTED. Every press has an inverse except for
        the pad re-arm, which only ever fires one way (nothing but a swap turns
        the pads back off), so a state with the pads down can reach one with
        them up and never the reverse. That is why callers get the edge list
        rather than assuming they can BFS backwards from a goal, and why the
        searches key states exactly instead of by the reachable region -- the
        usual sokoban "same pieces, same player region" merge needs the relation
        to be an equivalence, and here it is not. Merging on it anyway loses
        solutions: level 4, which is 145 macro expansions with an exact key,
        was still unsolved after 400_000 with a region key."""
        smalls, bigs = st[2], st[3]
        start = (st[0], st[1], st[4])
        parent = {start: None}
        edges: list = []
        queue = deque([start])
        outs = {}
        while queue:
            cur = queue.popleft()
            cs = (cur[0], cur[1], smalls, bigs, cur[2])
            for d in DIRS:
                ns = self.step(cs, d)
                if ns is cs:
                    continue                      # the turn was a no-op
                if ns[0] == cur[0] and ns[2] == smalls and ns[3] == bigs:
                    k = (ns[0], ns[1], ns[4])
                    edges.append((cur, k))
                    if k not in parent:
                        parent[k] = (cur, d)
                        queue.append(k)
                else:
                    outs.setdefault(ns, (cur, d))
        return parent, edges, outs

    def successors(self, st) -> list:
        """``[(macro presses, resulting state), ...]`` -- for every push, resize
        or teleport the player can reach, the walk that gets there plus that
        press."""
        parent, _edges, outs = self._walk_graph(st)

        def walk_to(node) -> list:
            out = []
            while parent[node] is not None:
                node, d = parent[node]
                out.append(d)
            out.reverse()
            return out

        return [(walk_to(node) + [d], ns) for ns, (node, d) in outs.items()]

    def walk_distances(self, st, goal) -> dict:
        """Distance from every walk state reachable from ``st`` TO ``goal``.

        A reverse BFS over the walk edges collected forwards, because the graph
        is directed (see `_walk_graph`). Used only by `_optimal_sets`."""
        _parent, edges, _outs = self._walk_graph(st)
        back: dict = {}
        for src, dst in edges:
            back.setdefault(dst, []).append(src)
        dist = {goal: 0}
        queue = deque([goal])
        while queue:
            cur = queue.popleft()
            for src in back.get(cur, ()):
                if src not in dist:
                    dist[src] = dist[cur] + 1
                    queue.append(src)
        return dist


# ---------------------------------------------------------------------------
# Heuristic: per-target push-distance tables + greedy matching
# ---------------------------------------------------------------------------

class _Heur:
    """The sokoban estimate, in units of primitive presses.

    ``tables[t][cell]`` is the number of pushes needed to bring a crate from
    ``cell`` onto target ``t``, ignoring the other crates -- a reverse BFS over
    push edges, which is also the deadlock test for free (a cell no push can
    ever leave lands in no table). The two game-specific parts are documented in
    the module docstring: pad teleport edges, and a matching that reuses a crate
    rather than declaring the state dead."""

    def __init__(self, board: _Board):
        self.board = board
        self.targets = sorted(board.targets)
        self.tables = {t: self._table(t) for t in self.targets}

    def _table(self, target) -> dict:
        b = self.board
        h, w = b.h, b.w
        steps = {target: 0}
        queue = deque([target])
        sp, bpq = b.sp, set(b.bpq)
        while queue:
            cur = queue.popleft()
            r, c = cur
            cost = steps[cur]
            for dr, dc in _DELTA.values():
                pr, pc = r - dr, c - dc         # where the crate comes from
                sr, sc = pr - dr, pc - dc       # where the pusher stands
                if not (0 <= pr < h and 0 <= pc < w) or (pr, pc) in b.walls:
                    continue
                if not (0 <= sr < h and 0 <= sc < w) or (sr, sc) in b.walls:
                    continue
                if (pr, pc) not in steps:
                    steps[(pr, pc)] = cost + 1
                    queue.append((pr, pc))
            if sp is not None:
                if cur in bpq and sp not in steps:
                    steps[sp] = cost + 1
                    queue.append(sp)
                if cur == sp:
                    for cc in bpq:
                        if cc not in steps:
                            steps[cc] = cost + 1
                            queue.append(cc)
        return steps

    def __call__(self, st, mode: str = "sum") -> int:
        _size, cell, smalls, bigs, _on = st
        occ = set(smalls)
        for tl in bigs:
            occ.update(_quad(tl))
        bare = [t for t in self.targets if t not in occ]
        if not bare:
            return 0
        loose = [c for c in occ if c not in self.board.targets]
        parts = []
        used: dict[int, int] = {}
        for t in bare:
            steps = self.tables[t]
            best = pick = None
            for i, c in enumerate(loose):
                if i in used:
                    continue
                d = steps.get(c)
                if d is not None and (best is None or d < best):
                    best, pick = d, i
            if best is None and self.board.sp is not None:
                # Every crate is spoken for -- but on a board WITH pads one that
                # grows covers four targets at once, so let a crate be reused
                # instead of calling the state dead (level 2 is one crate and
                # four targets). Gated on the pads existing because the reuse
                # costs a real prune: without it, "this target has no crate left
                # that can reach it" is a proof the state is lost, and on level 0
                # -- the one board with no pads, where the crate count can never
                # change -- that proof is most of what makes the search close.
                for i, c in enumerate(loose):
                    if used.get(i, 0) >= 4:
                        continue
                    d = steps.get(c)
                    if d is not None and (best is None or d < best):
                        best, pick = d, i
            if best is None:
                return BIG
            used[pick] = used.get(pick, 0) + 1
            parts.append(best)
        walk = min((abs(cell[0] - r) + abs(cell[1] - c) for r, c in loose),
                   default=1)
        base = max(parts) if mode == "max" else sum(parts)
        return base + max(0, walk - 1)


# ---------------------------------------------------------------------------
# Searches over the model
# ---------------------------------------------------------------------------

def _path(nodes: list, i: int) -> list:
    """Materialise a plan from the parent-pointer tree. The searches keep
    ``(parent index, macro)`` rather than a path per node: level 6's beam holds
    ~50k nodes and copying an 85-press list into each of them is most of the
    run's memory."""
    out = []
    while i:
        prev, macro = nodes[i]
        out.append(macro)
        i = prev
    out.reverse()
    return [d for m in out for d in m]


def _astar(board: _Board, st, h: _Heur, mode: str, weight: int,
           cap: int) -> list | None:
    """Weighted A* over macros; ``g`` counts primitive presses.

    Successors are derived when a node is POPPED, not when it is pushed. It is
    the same work per expanded node and strictly less for the rest (a node that
    never reaches the front never pays), and it keeps the queue to one state per
    entry -- holding each node's macro list instead cost ~4.5 KB a node and put
    level 6 at 1.8 GB before it had finished a single rung."""
    if board.won(st):
        return []
    nodes = [(None, None)]
    pq = [(weight * h(st, mode), 0, 0, st, 0)]
    best = {st: 0}
    counter = expanded = 0
    while pq:
        _f, g, _c, cur, ni = heapq.heappop(pq)
        for macro, ns in board.successors(cur):
            expanded += 1
            if board.won(ns):
                nodes.append((ni, macro))
                return _path(nodes, len(nodes) - 1)
            hv = h(ns, mode)
            if hv >= BIG:
                continue
            ng = g + len(macro)
            if best.get(ns, 1 << 30) <= ng:
                continue
            best[ns] = ng
            counter += 1
            nodes.append((ni, macro))
            heapq.heappush(pq, (ng + weight * hv, ng, counter, ns,
                                len(nodes) - 1))
        if expanded >= cap:
            return None
    return None


def _beam(board: _Board, st, h: _Heur, mode: str, width: int, depth: int,
          cap: int) -> list | None:
    """Width-capped breadth-first beam over the same macros.

    Reach for it where the heuristic is good enough to rank states but A*
    cannot afford the breadth -- level 6 is an EXACT packing (ten crates, ten
    targets, nothing on the board can grow), so its estimate barely moves for
    the first dozen pushes while A* pays for every ordering of them."""
    if board.won(st):
        return []
    nodes = [(None, None)]
    frontier = [(st, 0)]
    seen = {st}
    expanded = 0
    for _depth in range(depth):
        kids = []
        for cur, ni in frontier:
            for macro, ns in board.successors(cur):
                expanded += 1
                if board.won(ns):
                    nodes.append((ni, macro))
                    return _path(nodes, len(nodes) - 1)
                if ns in seen:
                    continue
                hv = h(ns, mode)
                if hv >= BIG:
                    continue
                seen.add(ns)
                nodes.append((ni, macro))
                kids.append((hv, ns, len(nodes) - 1))
                if expanded >= cap:
                    return None
        if not kids:
            return None                      # the reachable space closed
        # Sort on the estimate alone: the tuple also carries a state, which has
        # no ordering. Successors are derived at the next depth, so the states
        # cut by the width never pay for theirs (see `_astar`).
        kids.sort(key=lambda kid: kid[0])
        frontier = [(ns, ni) for _h, ns, ni in kids[:width]]
    return None


#: Search rungs, tried in order, FIRST success wins: ``("astar", mode, weight,
#: expansion cap)`` or ``("beam", mode, width, depth, cap)``.
#:
#: The order is by expected plan quality, and on these eight levels it is also
#: the observed order -- nothing further down ever beat what an earlier rung
#: returned, so there is nothing to gain by running the rest and keeping the
#: shortest. The first three rungs solve everything:
#:
#:     level    0   1   2   3   4   5   6    7
#:     rung   max1 max1 max1 max1 max1 max1 sum3 sum3
#:     presses 30  28   32   55   78   74   76  120
#:
#: for ~3.5 minutes of one-time search, cached to `PLAN_CACHE`.
#:
#: The last two rungs are a safety net for a re-plan from an off-plan state, and
#: for a level added later. Neither is idle insurance: on level 6 the greedy
#: rung answers 131 and the beam 106 where ``sum3`` answers 76, so both DO find
#: plans the earlier rungs would have needed a much larger budget for.
#:
#: Weight is not monotone in either direction here -- level 6 is unsolved at
#: weight 8 after 72 seconds and solved at weight 3 in 11, and at weight 20 in 2
#: -- so the ladder is an empirical ordering, not a schedule. Beam width behaves
#: the same way (300 and 1000 close out on level 6, 3000 wins), which is what a
#: beam is: a wider one keeps DIFFERENT states, not more of the same ones.
_LADDER: tuple = (
    ("astar", "max", 1, 400_000),
    ("astar", "sum", 1, 400_000),
    ("astar", "sum", 3, 800_000),
    ("astar", "sum", 20, 1_500_000),
    ("beam", "sum", 3000, 220, 6_000_000),
)


# ---------------------------------------------------------------------------
# Optimal-action sets
# ---------------------------------------------------------------------------

def _optimal_sets(board: _Board, st, plan: list) -> list:
    """Per-press optimal-direction SETS for ``plan`` starting at ``st``.

    A press that changed anything but the player's cell is a push and is
    labelled with itself; a maximal run of pure walks is labelled with every
    direction that keeps each of its presses on a shortest walk to the state the
    next push is taken from. Classification watches the board rather than
    trusting a macro boundary the flattened plan no longer carries."""
    steps = []
    cur = st
    for d in plan:
        nxt = board.step(cur, d)
        kind = ("push" if (nxt[2] != cur[2] or nxt[3] != cur[3]
                           or nxt[0] != cur[0])
                else "walk" if nxt[1] != cur[1] or nxt[4] != cur[4]
                else "stuck")
        steps.append((kind, cur, d))
        cur = nxt

    out: list = [None] * len(plan)
    i = 0
    while i < len(steps):
        if steps[i][0] != "walk":
            out[i] = [steps[i][2]]
            i += 1
            continue
        run = i
        while i < len(steps) and steps[i][0] == "walk":
            i += 1
        if i >= len(steps):
            # A trailing walk pushes nothing: keep the recorded choice.
            for j in range(run, i):
                out[j] = [steps[j][2]]
            continue
        goal_state = steps[i][1]
        goal = (goal_state[0], goal_state[1], goal_state[4])
        dist = board.walk_distances(steps[run][1], goal)
        for j in range(run, i):
            here = steps[j][1]
            d0 = dist.get((here[0], here[1], here[4]))
            alts = []
            if d0 is not None:
                for d in DIRS:
                    ns = board.step(here, d)
                    if ns is here or ns[2] != here[2] or ns[3] != here[3] \
                            or ns[0] != here[0]:
                        continue
                    if dist.get((ns[0], ns[1], ns[4])) == d0 - 1:
                        alts.append(d)
            # The recorded press is on a shortest route by construction, so an
            # empty (or disagreeing) `alts` means the reconstruction drifted --
            # fall back to labelling what the expert did.
            out[j] = alts if steps[j][2] in alts else [steps[j][2]]
    return out


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class DangImHugeExpert(PSExpert):
    """Plans on `_Board`, certifies on the interpreter.

    `PSExpert.plan` owns the memo, the disk cache and the snapshot discipline;
    only the strategy hook `_search` is replaced, so nothing downstream knows
    the search did not run on the engine."""

    directions = list(DIRS)               # `noaction`: ACTION does nothing here
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        self._boards: dict = {}           # static signature -> (board, heur)

    def heuristic(self, eng) -> int:
        """Unused: `_search` never calls `PSExpert._astar`. Defined because the
        base declares it abstract."""
        return 0

    def _board_for(self, eng) -> tuple:
        board = _Board(eng, self.g)
        got = self._boards.get(board.sig)
        if got is None:
            got = (board, _Heur(board))
            self._boards[board.sig] = got
        return got

    def _certify(self, eng, plan: list) -> list | None:
        """Replay ``plan`` on the real interpreter and return it truncated at
        the press that wins, or None if it does not. Leaves ``eng`` unchanged."""
        start = snapshot(eng)
        try:
            for i, d in enumerate(plan):
                eng.step(d)
                if eng.check_win():
                    return plan[:i + 1]
            return None
        finally:
            restore(eng, start)

    def _search(self, eng) -> list | None:
        board, h = self._board_for(eng)
        st = board.read(eng)
        if st[1] is None:                 # no player: nothing to plan
            return None
        for rung in _LADDER:
            if rung[0] == "astar":
                _k, mode, weight, cap = rung
                found = _astar(board, st, h, mode, weight, cap)
            else:
                _k, mode, width, depth, cap = rung
                found = _beam(board, st, h, mode, width, depth, cap)
            if found is None:
                continue
            certified = self._certify(eng, found)
            if certified is None:
                continue                  # a model slip: try the next rung
            return Plan(certified, _optimal_sets(board, st, certified))
        return None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class DangImHugeSolver(PSAStarSolver):
    game_id = "puzzlescript_dang_im_huge"
    game_name = GAME_NAME
    expert_cls = DangImHugeExpert
    game_module_id = "ps:dang_im_huge"

    #: Unused by `_search` (the ladder carries its own budgets) but passed to
    #: the expert's constructor by the harness.
    node_cap = 3_000_000
    weight = 1

    #: Room for the longest plan (level 7's 120) plus a re-plan; `epsilon` is 0
    #: for this family, so nothing else lengthens a replay.
    max_steps = 400

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time cost visible as startup
        rather than as a mysteriously slow first seed, and it fills the disk
        cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """The full set of equally-shortest presses at this step. Falls back to
        the press about to be taken, so no expert step ever ships unlabelled."""
        sets = getattr(plan, "optsets", None)
        if sets is not None and pi < len(sets):
            return sets[pi]
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# CLI extras
# ---------------------------------------------------------------------------

def _fuzz(rolls: int = 80, steps: int = 120, seed: int = 77) -> bool:
    """Compare `_Board` against the interpreter press by press, over rollouts on
    every level: the FULL state plus the win flag.

    Each rollout opens by replaying a PREFIX of the level's expert plan -- the
    cut point slides from nothing to the whole plan across the rollouts -- and
    then walks randomly from there. Purely random play almost never reaches the
    states this game is ABOUT: 80 x 120 random presses across the eight levels
    resized a crate three times in total, because doing it needs a crate walked
    onto a one-cell pad with the other pad clear. Seeding from the plan puts
    every resize, teleport and pad re-arm the expert uses under the comparison,
    and the random tail explores off each of them."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    expert = DangImHugeExpert(game, node_cap=DangImHugeSolver.node_cap)
    counts = {"resize": 0, "crate-resize": 0, "pad-flip": 0, "push": 0,
              "win": 0}
    ok = True
    for level in range(game.n_levels):
        game.set_level(level)
        board = _Board(game._engine, game._game)
        plan = expert.plan(game._engine, level) or []
        rng = random.Random(seed + level)
        for roll in range(rolls):
            game.set_level(level)
            eng = game._engine
            st = board.read(eng)
            prefix = (plan[:(roll * len(plan)) // max(1, rolls)]
                      if plan else [])
            for i in range(steps):
                d = prefix[i] if i < len(prefix) else rng.choice(DIRS)
                eng.step(d)
                prev, st = st, board.step(st, d)
                real = board.read(eng)
                if real != st:
                    print(f"MISMATCH level {level} roll {roll} step {i} {d}\n"
                          f"  from   {prev}\n  model  {st}\n  engine {real}")
                    ok = False
                    break
                if board.won(st) != eng.check_win():
                    print(f"WIN MISMATCH level {level} roll {roll} step {i}")
                    ok = False
                    break
                if prev[0] != st[0]:
                    counts["resize"] += 1
                if len(prev[3]) != len(st[3]):
                    counts["crate-resize"] += 1
                if prev[4] != st[4]:
                    counts["pad-flip"] += 1
                if prev[2] != st[2] or prev[3] != st[3]:
                    counts["push"] += 1
                if eng.check_win():
                    counts["win"] += 1
                    break
            if not ok:
                return False
        print(f"level {level}: ok", flush=True)
    print("model matches the interpreter;", counts)
    return ok


#: Cell compositions that are deliberately identical: every quadrant of the big
#: pad renders the same square as the small pad, on purpose. Painting a pad's
#: outer corner is the whole point of the sprite fix (that corner is the only
#: pixel a body standing on it leaves transparent), so the pads cannot also
#: carry a per-cell mark saying which one they are -- and they do not need to.
#: The two are told apart by EXTENT, one green cell against a 2x2 green square,
#: which is the same thing the crates and the player are told apart by.
_PAD_CLASSES: tuple = (
    ("zapsmall", "zap1", "zap2", "zap3", "zap4"),
    ("zapsmalloff", "zap1off", "zap2off", "zap3off", "zap4off"),
)


def _audit() -> int:
    """Render every cell COMPOSITION the game can show (bare, and each body on
    each pad state / on a target) at every cell size the levels use, and assert
    they are pairwise distinct apart from `_PAD_CLASSES`.

    Composition, not object: the bugs this catches live in the STACK. Both the
    ones it did catch here were of that kind -- an occupied pad rendered as bare
    floor, so the player standing on the small pad was pixel-identical to the
    player standing anywhere else."""
    import numpy as np
    from adapters.puzzlescript_adapter import _render_cell_sprite

    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    g, eng = game._game, game._engine
    layers = eng._obj_layers
    sizes = set()
    for level in range(game.n_levels):
        game.set_level(level)
        grid = game._engine.grid
        sizes.add(max(1, min(64 // len(grid), 64 // len(grid[0]))))

    bodies = ["block", "block1", "block2", "block3", "block4",
              "playersmall", "player1", "player2", "player3", "player4"]
    grounds = ["", "target"] + [n for cls in _PAD_CLASSES for n in cls]
    canon = {n: cls[0] for cls in _PAD_CLASSES for n in cls}

    def render(names, px):
        objs = sorted(((layers[g.obj_name_to_idx[n]], g.objects[n])
                       for n in names), key=lambda x: x[0])
        return tuple(_render_cell_sprite(objs, px, None, 5).flatten().tolist())

    bad = 0
    for px in sorted(sizes):
        seen: dict = {}
        for ground in grounds:
            for body in [""] + bodies:
                names = [n for n in (ground, body) if n]
                label = "/".join(canon.get(n, n) for n in names) or "floor"
                tile = (tuple(np.full((px, px), 5, dtype=np.int16)
                              .flatten().tolist()) if not names
                        else render(names, px))
                if tile in seen and seen[tile] != label:
                    print(f"  cell_px {px}: COLLIDE {seen[tile]} == {label}")
                    bad += 1
                seen.setdefault(tile, label)
        # walls and bare floor are compositions too
        for extra in (["wall"],):
            tile = render(extra, px)
            if tile in seen:
                print(f"  cell_px {px}: COLLIDE {seen[tile]} == {extra[0]}")
                bad += 1
            seen[tile] = extra[0]
        print(f"cell_px {px}: {len(seen)} distinct compositions")
    print("audit clean" if not bad else f"AUDIT FAILED: {bad} collisions")
    return 1 if bad else 0


def _report() -> int:
    """Per-level plan report: length, push count, and how many presses carry a
    real tie set."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    expert = DangImHugeExpert(game, node_cap=DangImHugeSolver.node_cap)
    total = ties = pushes = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(game._engine, level)
        if plan is None:
            print(f"level {level}: UNSOLVED")
            continue
        sets = getattr(plan, "optsets", []) or []
        tied = sum(1 for s in sets if len(s) > 1)
        board, _h = expert._board_for(game._engine)
        st = board.read(game._engine)
        npush = 0
        for d in plan:
            ns = board.step(st, d)
            if ns[2] != st[2] or ns[3] != st[3] or ns[0] != st[0]:
                npush += 1
            st = ns
        total += len(plan)
        ties += tied
        pushes += npush
        print(f"level {level}: {len(plan):3d} presses, {npush:3d} pushes, "
              f"{tied:3d} with a tie set")
    print(f"total {total} presses, {pushes} pushes, {ties} tied")
    return 0


if __name__ == "__main__":
    if "--fuzz" in sys.argv:
        rest = [int(a) for a in sys.argv[sys.argv.index("--fuzz") + 1:]
                if a.isdigit()]
        sys.exit(0 if _fuzz(*rest) else 1)
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--plans" in sys.argv:
        sys.exit(_report())
    sys.exit(DangImHugeSolver.main())
