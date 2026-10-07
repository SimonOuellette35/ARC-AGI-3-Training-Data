"""Generate Phase-1 training data for the PuzzleScript game ps:tornado_tamer
("Tornado Tamer", Mark Foster).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanic, a native model of it, the two searches that model is planned with,
the proof that three of the six plans are shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_tornado_tamer",
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
Every recorded expert step carries an optimal-action set.

The game
--------
You are a wizard sweeping leaves. You cannot touch a leaf; the only thing you
can do to the board is CAST A TORNADO, and a tornado, once cast, is out of your
hands forever.

**Casting takes two presses and the second one is a direction.** ACTION toggles
a "casting" mode on (and pressing it again toggles it off, which is how you
wait). While casting, a direction press does not move you -- it drops a tornado
of that direction into the square you pressed towards and ends the mode. If that
square holds a Wall or another tornado the cast simply fizzles, and you have
still spent the press and the mode. Cones and leaves do NOT block a cast; you can
drop a tornado onto either.

**Tornadoes move one square per TURN, every turn, on their own**, and the turn
clock is your keyboard: every press you make is one tick of the whole board, so
"walk three squares" and "let that gust travel three squares" are the same three
presses. That is the whole shape of the game -- it is a scheduling puzzle wearing
a sokoban's clothes.

A tornado standing on leaves drags them along with it, provided the square ahead
is not a wall. When its own next square IS a wall it stops dead and becomes a
TornadoStill, and a Still stays exactly where it is for the rest of the level
unless one of two things happens to it:

  * **you walk into it** -- the player steps onto a Still and deletes it, the
    only way anything is ever taken off the board outright, and (because that
    rule is checked before the one that makes cones solid) also the only way a
    player ever stands on a cone; or
  * **a moving tornado runs into it** -- and then it SPLITS. The moving tornado
    is destroyed and the Still becomes a split tornado *across* the incoming
    axis: a tornado arriving horizontally leaves a TornadoSplitVert, one
    arriving vertically leaves a TornadoSplitHoriz. On the NEXT turn that split
    bursts into two tornadoes in the neighbouring squares -- up and down, or
    left and right -- and vanishes from its own. Whichever of the two sides is a
    wall simply gets no tornado.

So a Still is not debris, it is a MIRROR you have built: two casts down a
corridor buy you a tornado travelling ACROSS it, and that is the only way to
turn a corner. Level 0 never needs one; levels 2 and 5 are built on one; level 3
is nothing but chains of them.

The win is ``All Leaves on GoalArea``. Leaves are never destroyed and never
created, so the whole game is "get every pile onto the green".

Three details that decide plans and are not visible from the rule list:

  * **The split writes straight over whatever is in the two side squares** --
    including the player, who is simply gone. `_Board.step` models that (and
    `TornadoTamerExpert._refuse` will not plan from a board with no player),
    because a split that lands on you is a real way to lose a level that still
    looks playable.
  * **The rule list runs ONCE per turn in this interpreter**, not to a fixed
    point, so a split created by a collision does not burst until the following
    turn, and a tornado created by a burst does not move until the turn after
    that. Those one-turn delays are load-bearing: every relay in level 3 is
    counted in them.
  * **A tornado stops pushing the leaf it carries the moment it is deleted.**
    The collision rule comes before the leaf-dragging rule, so a gust that
    reaches a Still with a pile on board leaves the pile exactly where it died.
    That is how a leaf is parked on a goal square that is not at the end of its
    corridor.

Expert solver
-------------
A NATIVE model of the turn above (`_Board`), planned two ways:

  * `_astar` -- A* over the model with the admissible `_Heur` below, at unit
    cost and weight 1, so anything it returns is PROVABLY SHORTEST. It answers
    levels 1, 2 and 4 (10, 10 and 13 presses) in 11k, 3.7k and 24k model steps.
  * `_beam` -- a depth-synchronised beam ranked by the same estimate summed over
    the leaves, for the three levels A* cannot close. It answers levels 0 and 5
    at 23 and 43 presses, and level 3 at 128.

(Level indices are 0-based throughout, as ``level_id`` is.)

Which one a level gets is measured -- `_search` offers a board to A* under a
node cap and falls through to the beam only when that cap runs out -- and then
REMEMBERED per board, because re-planning a beam board would otherwise burn the
whole cap before every fallback. `TornadoTamerSolver.exact_levels` is that
memory in its startup form, so a process that hits the disk plan cache and
searches nothing still knows which is which; ``--plans`` re-runs A* on all six
starts and fails if the two disagree.

`_Heur` is the lower bound the whole file leans on. A leaf cannot move until a
tornado is standing on it, and once it moves it covers one square per turn, so
for every leaf still off the green

    turns-to-finish  >=  (earliest a tornado can be on it) + (squares to a goal)

and the level cannot end before the largest of those. The second term is a BFS
over non-wall squares from the goals. The first is the smaller of "some tornado
that already exists is this many squares away" (a tornado covers one square per
turn whatever it does, so straight-line distance is a bound however many splits
it takes) and "you walk to a square, spend two presses casting, and the gust
covers the rest" -- a multi-source BFS over the open grid seeded with the walk
distances. Every step of that is a relaxation, so the estimate is admissible,
which is what makes A*'s answers proofs and what makes the tie sets exact.

Shortest, and how that is known
-------------------------------
  * **Levels 1, 2 and 4 are provably shortest** (10, 10 and 13). Two independent
    derivations agree: this file's A* with an admissible estimate, and a plain
    breadth-first search over the same model with no estimate at all
    (``--selfcheck``). Every step of those three plans carries an EXACT optimal
    set, measured by re-solving each alternative under a bound rather than
    inferred.
  * **Levels 0, 3 and 5 are NOT claimed shortest** (23, 128, 43). They are beam
    results, they win in the real interpreter, and their steps are labelled with
    the press the expert took and nothing else -- labelling a beam's siblings as
    ties would be asserting something not measured. `--plans` reports the
    admissible lower bound beside each length so the gap is on the record: 11
    against 23, 9 against 128, 8 against 43. Those bounds are honest but weak,
    and structurally so -- the bound counts TURNS, and this game's cost is
    dominated by the player's serial schedule of casts, which no per-leaf bound
    can see.

Level 3 is the outlier and it is worth saying why. Its player starts in a
15-square room whose only door is a cone, and the only way to cross a cone is to
delete a Still standing on it. No tornado can ever stand still on that one. A
Still needs a wall in its direction of travel, and the two walls beside that
square are the ones above and below it -- so the tornado would have to be
travelling vertically there, and the only two ways to put a vertically
travelling tornado on a square are to cast it from the square behind (a wall) or
to burst a split centred on the square behind (also a wall). **The player is
sealed in for the whole level** -- `_closure` derives that rather than taking it
on trust, and ``--selfcheck`` asserts it. Everything on the far side of the map
therefore has to be done by relay: two casts make a Still at the end of the
corridor, two more turn it into a tornado climbing the shaft, and the far leaf's
corridor is several such relays deep. That is what 128 presses buys, against a
200-press level budget.

Recovery
--------
``recovery_mode = "reset"``. The mechanic is irreversible in the strong sense --
a tornado you cast cannot be recalled, and a leaf shoved past its goal has to be
brought back by a fresh gust from the other side -- so the episode-wide
exploration prefix explores freely and ONE RESET restores the level's initial
state, from which the cached plan replays a guaranteed WIN. `solve_from` is
nonetheless a genuine re-plan from the LIVE board: `_search` reads the engine
grid every time it is called, so a probe from an arbitrary state is answered
(A* on the exact levels, beam elsewhere) or refused, and `--selfcheck`'s
recovery pass exercises exactly that.

Epsilon detours are ON for the three levels A* closes and OFF for the three the
beam closes, through `PSAStarSolver.epsilon_for`. That is not squeamishness
about the mechanic: `record_level` vets every detour by re-planning from it, and
a beam re-plan costs seconds where an A* re-plan costs a fraction of one. Level
3 has a second reason -- its plan is 128 presses against the adapter's 200-press
per-level cap, and a detour there is worth ten.

Rendering
---------
Fourteen sprites in ``data/puzzlescript_games/Tornado_Tamer.txt`` are redrawn;
the header comment there is the full account and ``--audit`` is the check. In
one line: three of the seven tornado states shipped as pixel-identical art, a
tornado painted over the leaves it was carrying, and a decorative five-frame
animation sprayed noise over the top of everything -- so neither "which way is
that going", nor "is it carrying the pile", nor "is that about to burst" was on
the screen. No rule, level, layer or win condition is touched.
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

GAME_NAME = "Tornado_Tamer"
GAME_ID = "puzzlescript_tornado_tamer"

#: Where the six level-start plans live between processes. Level 4's beam is a
#: several-minute search; without this every shard `parallelize_generator`
#: starts would re-derive it. See `PSExpert.plan` for when an entry is trusted.
PLAN_CACHE = Path(__file__).resolve().parent.parent / "data" / "tornado_tamer_plans.json"

#: Engine actions, in the order every search branches on them.
DIRS = tuple(DIRECTIONS)
MOVES = ("up", "down", "left", "right")
DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Tornado states. The four directional ones travel; TS sits until something
#: happens to it; TV/TH are the one-turn states a collision leaves behind.
TU, TD, TL, TR, TS, TV, TH = range(7)
TDIR = {TU: "up", TD: "down", TL: "left", TR: "right"}
DIRT = {"up": TU, "down": TD, "left": TL, "right": TR}

#: Collision layers that matter. Player, Wall and every tornado share LP, so at
#: most one of them is in a square; DenseLeaves and ThinLeaves share LL.
LP, LL = 0, 1

_INF = 1 << 30


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static scenery, and the turn its dynamics are.

    A state is ``(player, cast_at, tornadoes, leaves)``:

      * ``player`` -- square index, or None once a split has written over it;
      * ``cast_at`` -- the square the Casting marker is on, or None. Under any
        reachable play it is either None or the player's own square (while
        casting, EVERY direction press is consumed by a cast rule, so the player
        cannot walk out from under it). It is kept as a square rather than a
        flag because when a split deletes the player the marker is orphaned
        where it stood, and the interpreter leaves it there forever;
      * ``tornadoes`` -- a sorted ``(square, state)`` tuple;
      * ``leaves`` -- a frozenset of squares. DenseLeaves and ThinLeaves are one
        thing to every rule and to the win condition, so the model does not
        distinguish them.

    `step` is a transcription of the interpreter's rule list, in ITS order and
    with ITS one-pass-per-turn semantics, and `--selfcheck` is what says so:
    the model and `PSEngine.step` are driven side by side over random walks from
    every level start and over thousands of randomly seated boards.
    """

    def __init__(self, h, w, walls, cones, goals):
        self.h, self.w = h, w
        self.walls = frozenset(walls)
        self.cones = frozenset(cones)
        self.goals = frozenset(goals)
        self.sig = (h, w, tuple(sorted(self.walls)), tuple(sorted(self.cones)),
                    tuple(sorted(self.goals)))
        nb = {}
        for i in range(h * w):
            r, c = divmod(i, w)
            for d, (dr, dc) in DELTA.items():
                rr, cc = r + dr, c + dc
                nb[(i, d)] = rr * w + cc if 0 <= rr < h and 0 <= cc < w else None
        self.nbmap = nb
        self._heur = None

    # -- goal ------------------------------------------------------------
    def won(self, state) -> bool:
        return all(i in self.goals for i in state[3])

    @property
    def heur(self) -> "_Heur":
        if self._heur is None:
            self._heur = _Heur(self)
        return self._heur

    # -- one turn --------------------------------------------------------
    def step(self, state, act):
        """The board after one press. ``act`` is an engine direction or
        ``"action"``.

        The rule list, in the interpreter's order (see `--selfcheck`):

          0-3   every directional tornado is given its force;
          4-5   ACTION toggles the Casting marker (and consumes the action flag,
                which is why the second toggle rule does not immediately undo
                the first);
          6-11  a direction press WHILE casting is a cast, not a move: it drops
                a tornado ahead of the player unless a Wall or another tornado
                is there, and either way the press and the mode are spent;
          12    a tornado facing a wall becomes a Still and loses its force;
          13-20 the eight split rules, INTERLEAVED horizontal-then-vertical per
                wall case -- the interpreter runs them in that order and it is
                observable, because two splits can write into the same square;
          21-22 a moving tornado that reaches a Still is destroyed and turns it
                into a split across the incoming axis;
          23    the player steps onto a Still and deletes it (before the cone
                rule, so this is how a player crosses a cone);
          24    a tornado with leaves under it drags them, unless a wall is
                ahead;
          25    a cone cancels the player's move;
          then  forces are resolved.
        """
        player, cast_at, tor, leaves = state
        tor = dict(tor)
        leaves = set(leaves)
        nb = self.nbmap
        walls = self.walls

        tf = {}                                    # tornado square -> direction
        pf = None                                  # the player's own force
        if act == "action":
            if player is not None:
                cast_at = None if cast_at == player else player
        else:
            pf = act

        for p, t in tor.items():                   # rules 0-3
            d = TDIR.get(t)
            if d is not None:
                tf[p] = d

        if pf is not None and player is not None and cast_at == player:
            q = nb[(player, pf)]                   # rules 6-11
            if q is not None:
                cast_at = None
                if q not in walls and q not in tor:
                    tor[q] = DIRT[pf]
                pf = None

        for p in list(tf):                         # rule 12
            q = nb[(p, tf[p])]
            if q is not None and q in walls:
                tor[p] = TS
                del tf[p]

        for wb, wf in ((False, False), (True, False),
                       (False, True), (True, True)):
            for kind, back, fwd, new_back, new_fwd in (
                    (TH, "left", "right", TL, TR),
                    (TV, "up", "down", TU, TD)):
                for _ in range(200):               # rules 13-20
                    fired = False
                    for p in sorted(tor):
                        if tor.get(p) != kind:
                            continue
                        a, b = nb[(p, back)], nb[(p, fwd)]
                        if a is None or b is None:
                            continue
                        if ((a in walls) is not wb) or ((b in walls) is not wf):
                            continue
                        del tor[p]
                        tf.pop(p, None)
                        for cell, born, walled in ((a, new_back, wb),
                                                   (b, new_fwd, wf)):
                            if walled:
                                continue
                            # The interpreter keys forces by OBJECT, so a square
                            # that already held this very tornado state keeps
                            # the force rules 0-3 gave it and moves off in the
                            # same turn; anything else is overwritten and its
                            # force key goes stale. Getting this backwards costs
                            # a two-chain collision that never happens.
                            if tor.get(cell) != born:
                                tf.pop(cell, None)
                            tor[cell] = born
                            if player == cell:
                                player = None
                                pf = None
                        fired = True
                    if not fired:
                        break

        for dirs, kind in ((("left", "right"), TV), (("up", "down"), TH)):
            for d in dirs:                         # rules 21-22
                for _ in range(200):
                    fired = False
                    for p in sorted(tor):
                        if tf.get(p) != d:
                            continue
                        q = nb[(p, d)]
                        if q is None or tor.get(q) != TS:
                            continue
                        del tor[p]
                        del tf[p]
                        tor[q] = kind
                        fired = True
                    if not fired:
                        break

        if pf is not None and player is not None:  # rule 23
            q = nb[(player, pf)]
            if q is not None and tor.get(q) == TS:
                del tor[q]
                player = q
                pf = None

        lf = {}
        for p, d in tf.items():                    # rule 24
            if p in leaves:
                q = nb[(p, d)]
                if q is not None and q not in walls:
                    lf[p] = d

        if pf is not None and player is not None:  # rule 25
            q = nb[(player, pf)]
            if q is not None and q in self.cones:
                pf = None

        forces = {}
        for p, d in tf.items():
            if p in tor and tor[p] in TDIR:
                forces[(LP, p)] = d
        if pf is not None and player is not None:
            forces[(LP, player)] = pf
        for p, d in lf.items():
            forces[(LL, p)] = d
        if forces:
            player, tor, leaves = self._resolve(forces, player, tor, leaves)

        return (player, cast_at, tuple(sorted(tor.items())), frozenset(leaves))

    def _resolve(self, forces, player, tor, leaves):
        """`PSEngine._resolve_forces`, for the two layers this game moves.

        Chain-based: an object blocked by one carrying the SAME force joins its
        chain and they move together (a tornado nudging the player along in
        front of it); one blocked by anything else is blocked, unless that
        blocker is itself moving somewhere, in which case the chain is deferred
        to a later pass and dropped if nothing ever moves. Two chains claiming
        the same square on the same layer block each other -- which is the whole
        reason this is transcribed rather than approximated, because a split can
        easily aim two fresh tornadoes at one square.
        """
        nb = self.nbmap
        walls = self.walls

        def occupied(layer, pos):
            if layer == LP:
                return pos in walls or pos in tor or pos == player
            return pos in leaves

        for _ in range(20):
            moved = deferred = False
            at_start = dict(forces)
            resolved = set()
            movable = []
            for key, d in list(forces.items()):
                if key in resolved:
                    continue
                layer, pos = key
                if not occupied(layer, pos):
                    forces.pop(key, None)
                    resolved.add(key)
                    continue
                chain = [key]
                cur = pos
                chain_free = False
                blocker_moves = False
                while True:
                    nxt = nb[(cur, d)]
                    if nxt is None:
                        break
                    if not occupied(layer, nxt):
                        chain_free = True
                        break
                    bf = forces.get((layer, nxt))
                    if bf == d:
                        chain.append((layer, nxt))
                        cur = nxt
                    else:
                        blocker_moves = bf is not None
                        break
                if chain_free:
                    movable.append((chain, d))
                elif blocker_moves:
                    deferred = True
                else:
                    for k in chain:
                        forces.pop(k, None)
                        resolved.add(k)

            non_head = {}
            for i, (chain, _d) in enumerate(movable):
                for ent in chain[1:]:
                    non_head[ent] = i
            subsumed = {i for i, (chain, _d) in enumerate(movable)
                        if chain[0] in non_head}

            claims = {}
            conflicting = set()
            for i, (chain, d) in enumerate(movable):
                if i in subsumed:
                    continue
                for layer, pos in chain:
                    tgt = (layer, nb[(pos, d)])
                    other = claims.get(tgt)
                    if other is not None and other != i:
                        conflicting.add(i)
                        conflicting.add(other)
                    else:
                        claims[tgt] = i

            for i, (chain, d) in enumerate(movable):
                if i in subsumed:
                    continue
                if i in conflicting:
                    for k in chain:
                        forces.pop(k, None)
                    continue
                for layer, pos in reversed(chain):
                    dst = nb[(pos, d)]
                    if layer == LP:
                        if pos == player:
                            player = dst
                        else:
                            tor[dst] = tor.pop(pos)
                    else:
                        leaves.discard(pos)
                        leaves.add(dst)
                    forces.pop((layer, pos), None)
                    moved = True

            if not moved:
                if deferred and forces == at_start:
                    forces.clear()
                break
        return player, tor, leaves


# ---------------------------------------------------------------------------
# The admissible estimate
# ---------------------------------------------------------------------------

class _Heur:
    """A lower bound on the presses left, in turns.

    A leaf cannot move until a tornado is standing on it, and thereafter it
    covers at most one square per turn. So for a leaf still off the green,

        remaining >= arrive(leaf) + push(leaf)

    and since the level ends only when EVERY leaf is home, the largest of those
    is a bound on the whole thing. Both terms are relaxations:

    ``push`` is a breadth-first distance from the goals over non-wall squares --
    leaves are only ever pushed in straight lines, so a free walk can only be
    shorter.

    ``arrive`` is the smaller of

      * ``|p - leaf|`` over the tornadoes that already exist, in Manhattan
        steps. A tornado covers exactly one square per turn no matter what it
        does, and a split hands its motion to a NEIGHBOUR, so no sequence of
        relays beats straight-line distance; and
      * the cast field: walk to a square (breadth-first over non-wall squares,
        ignoring cones and tornadoes, both of which can only slow you down),
        spend the two presses, and let a fresh gust cover the rest one square
        per turn. Seeded per square and relaxed over the OPEN grid, walls
        included, so it stays a bound.

    The fields are cached per (board, player square), which is what makes this
    affordable inside a search: everything except the tornado term is a table
    lookup.
    """

    def __init__(self, board: _Board):
        self.b = board
        n = board.h * board.w
        self.open_nb = [[] for _ in range(n)]
        self.free_nb = [[] for _ in range(n)]
        for i in range(n):
            for d in DELTA:
                j = board.nbmap[(i, d)]
                if j is not None:
                    self.open_nb[i].append(j)
                    if j not in board.walls:
                        self.free_nb[i].append(j)
        self.push = self._bfs([g for g in board.goals if g not in board.walls],
                              self.free_nb)
        self._walk: dict = {}
        self._cast: dict = {}

    def _bfs(self, sources, adj, init=None):
        d = [_INF] * (self.b.h * self.b.w)
        q = deque()
        for i in sources:
            v = 0 if init is None else init[i]
            if v < d[i]:
                d[i] = v
                q.append(i)
        while q:
            i = q.popleft()
            for j in adj[i]:
                if d[j] > d[i] + 1:
                    d[j] = d[i] + 1
                    q.append(j)
        return d

    def walk(self, src):
        got = self._walk.get(src)
        if got is None:
            got = self._walk[src] = self._bfs([src], self.free_nb)
        return got

    def cast_field(self, src, cost):
        """Earliest turn a tornado CAST FROM SCRATCH could stand on each square,
        starting from a player at ``src`` who needs ``cost`` presses to cast
        (two normally, one when the mode is already on)."""
        key = (src, cost)
        got = self._cast.get(key)
        if got is not None:
            return got
        wd = self.walk(src)
        seed = [_INF] * (self.b.h * self.b.w)
        srcs = []
        for c in range(len(seed)):
            if c in self.b.walls:
                continue
            best = min((wd[y] for y in self.free_nb[c]), default=_INF)
            if best < _INF:
                seed[c] = best + cost
                srcs.append(c)
        srcs.sort(key=lambda i: seed[i])
        got = self._cast[key] = self._bfs(srcs, self.open_nb, seed)
        return got

    def spread(self, state) -> tuple:
        """``(sum, max)`` of the per-leaf estimate over the leaves still off the
        green. Both readings come from one pass, because both callers want the
        same per-leaf number and only differ in how they fold it.

        The MAX is the admissible bound `__call__` hands to A*: every leaf needs
        its own estimate worth of turns and they overlap in time, so the largest
        is what the level cannot finish inside.

        The SUM is what the beam ranks on, and the max would be useless there --
        it only falls when the single hardest leaf is served, and says nothing
        about the other ten. The sum falls whenever any leaf is brought nearer,
        whenever a gust is cast towards one, and whenever the player walks
        towards a square it could cast from, which is enough signal to steer by.
        It is not a bound and is never used as one.
        """
        player, cast_at, tor, leaves = state
        goals = self.b.goals
        off = [x for x in leaves if x not in goals]
        if not off:
            return (0, 0)
        w = self.b.w
        cast = (self.cast_field(player, 1 if cast_at == player else 2)
                if player is not None else None)
        total = top = 0
        for x in off:
            xr, xc = divmod(x, w)
            a = cast[x] if cast is not None else _INF
            for pos, _t in tor:
                pr, pc = divmod(pos, w)
                m = abs(pr - xr) + abs(pc - xc)
                if m < a:
                    a = m
            v = a + self.push[x]
            total += v
            if v > top:
                top = v
        return (total, top)

    def __call__(self, state) -> int:
        return self.spread(state)[1]


# ---------------------------------------------------------------------------
# The two searches
# ---------------------------------------------------------------------------

def _astar(board: _Board, start, cap: int, bound: int = _INF):
    """Shortest press sequence from ``start``, or None.

    Unit costs, weight 1 and an ADMISSIBLE estimate, with re-opening (a state
    reached again more cheaply is pushed again), so a returned path is a proof
    of its own length. The goal is tested when a winning successor is GENERATED
    rather than when it is popped, which is sound here: a cheaper win would have
    a parent whose f is no larger than that win's cost -- the estimate one step
    from a win is at most one -- so that parent pops first and generates it.

    ``bound`` prunes every node whose f exceeds it. `_optsets` uses that to ask
    "is there a win in at most k presses" for a fraction of the cost of asking
    how long the shortest one is. ``cap`` bounds the node count; running out
    returns None, which callers must read as "not found", never as "no path".
    """
    h = board.heur
    if board.won(start):
        return []
    h0 = h(start)
    if h0 > bound:
        return None
    pq = [(h0, 0, 0, start, ())]
    best = {start: 0}
    nodes = counter = 0
    while pq:
        f, g, _c, state, path = heapq.heappop(pq)
        if best.get(state, _INF) < g:
            continue
        for a in DIRS:
            nxt = board.step(state, a)
            nodes += 1
            if nodes > cap:
                return None
            ng = g + 1
            if ng > bound:
                continue
            if board.won(nxt):
                return list(path) + [a]
            if best.get(nxt, _INF) <= ng:
                continue
            nh = h(nxt)
            if ng + nh > bound:
                continue
            best[nxt] = ng
            counter += 1
            heapq.heappush(pq, (ng + nh, ng, counter, nxt, path + (a,)))
    return None


def _bfs_plan(board: _Board, start, cap: int, limit: int = _INF):
    """The same answer as `_astar` with no estimate at all -- breadth-first over
    the model. Only `--selfcheck` calls it: it is the independent derivation the
    "provably shortest" claim is checked against, and it is slower by an order
    of magnitude, which is the point of the estimate.

    ``limit`` stops the sweep after that many presses and returns None. The tie
    check needs it, and needs it to be a DEPTH bound rather than a node one: it
    asks "does this press still finish in k" of every press including the ones
    that do not, and without a depth bound each of those answers costs the whole
    node cap rather than the ball it was asking about."""
    if board.won(start):
        return []
    if limit <= 0:
        return None
    seen = {start: (None, None, 0)}
    q = deque([start])
    nodes = 0
    while q:
        state = q.popleft()
        depth = seen[state][2]
        if depth >= limit:
            continue
        for a in DIRS:
            nxt = board.step(state, a)
            nodes += 1
            if nodes > cap:
                return None
            if nxt in seen:
                continue
            seen[nxt] = (state, a, depth + 1)
            if board.won(nxt):
                path = [a]
                cur = state
                while seen[cur][0] is not None:
                    prev, act, _d = seen[cur]
                    path.append(act)
                    cur = prev
                return list(reversed(path))
            q.append(nxt)
    return None


def _beam(board: _Board, start, width: int, max_depth: int):
    """A depth-synchronised beam ranked by `_Heur.spread`.

    Deterministic: the frontier is a dict (insertion-ordered), the ranking is a
    stable sort of it, and the model contains nothing but ints and tuples, so
    two runs of this file give byte-identical plans.

    It is here because A* cannot close three of the six levels and the reason is
    structural rather than a tuning problem. The estimate is a bound on TURNS,
    and this game's cost is dominated by the player's SERIAL schedule -- the five
    leaves of level 0 need five separate casts, which the bound cannot see at
    all, so it comes out at 11 against a true 23. A beam does not need a bound,
    only a direction to walk in, and the summed estimate is one.
    """
    h = board.heur
    if board.won(start):
        return []
    layer = {start: ()}
    seen = {start}
    for _ in range(max_depth):
        nxt = {}
        for state, path in layer.items():
            for a in DIRS:
                sub = board.step(state, a)
                if board.won(sub):
                    return list(path) + [a]
                if sub in seen or sub in nxt:
                    continue
                nxt[sub] = path + (a,)
        if not nxt:
            return None
        if len(nxt) > width:
            nxt = dict(sorted(nxt.items(), key=lambda kv: h.spread(kv[0]))[:width])
        seen.update(nxt)
        layer = nxt
    return None


def _optsets(board: _Board, start, plan, memo: dict):
    """The EXACT optimal set for every step of a shortest ``plan``.

    At a state with ``r`` presses left, a press is optimal exactly when a win is
    still ``r - 1`` presses away after it. Three sound prunes keep that cheap
    and none of them is a heuristic: a press that leaves the board untouched
    cannot be on a shortest path (it spends one press to reach the state it
    started from), a press whose admissible estimate already exceeds ``r - 1``
    cannot either, and the remaining candidates are asked with `_astar` under a
    bound of ``r - 1`` -- so each answer is a yes/no, not a distance.

    Only ever called on a plan A* returned, i.e. one already known shortest.
    """
    out = []
    state = start
    for i, taken in enumerate(plan):
        remaining = len(plan) - i
        best = []
        for a in DIRS:
            nxt = board.step(state, a)
            if board.won(nxt):
                if remaining == 1:
                    best.append(a)
                continue
            if nxt == state or remaining == 1:
                continue
            key = (nxt, remaining - 1)
            got = memo.get(key)
            if got is None:
                got = memo[key] = (
                    _astar(board, nxt, _OPTSET_CAP, remaining - 1) is not None)
            if got:
                best.append(a)
        out.append(best if taken in best else [taken])
        state = board.step(state, taken)
    return out


#: Node cap on one `_optsets` probe. Generous: the probes are bounded searches
#: that mostly fail fast, and a probe that ran out would silently drop a tie.
_OPTSET_CAP = 400_000


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class TornadoTamerExpert(PSExpert):
    """`PSExpert`'s plan memo, disk cache and snapshot discipline around the
    native model.

    The base class keeps the memo, the level scoping and the disk cache; only
    the strategy underneath changes. `_search` reads the ENGINE's grid every
    time it is called, so a re-plan from an arbitrary board -- a recovery probe,
    a detour, a `--selfcheck` perturbation -- is answered from that board and
    not from a replayed prefix.
    """

    directions = list(DIRS)
    #: `_key` is the dynamic objects only, canonical WITHIN a level but not
    #: across them: walls, cones and goals are static per level and differ.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    #: Model steps one A* may spend before a board is written off as one the
    #: beam has to handle. Levels 1, 2 and 4 close in 11k, 3.7k and 24k, so this
    #: is ~5x the largest success -- big enough that raising it would not add a
    #: level, small enough that the three beam levels are diagnosed in seconds.
    exact_cap: int = 120_000

    #: Beam widths tried at a LEVEL START, in order, keeping the shortest plan
    #: ANY of them finds -- all three are run, the sweep does not stop at the
    #: first success. Wider is not monotonically better here and that is why:
    #: swept on level 5 alone, widths 1000/1500/2500/3000/5000/8000 come out at
    #: 54/45/43/46/47/53 presses. Three cheap widths sample that curve for a few
    #: seconds; a single ladder would have taken whichever rung answered first.
    start_widths: tuple = (1500, 2500, 4000)

    #: Only reached when every width above failed, which on the shipped levels
    #: means level 3 and nothing else. Both are tried and the shorter plan wins,
    #: because on that level width buys real length: 8000 finds nothing, 15000
    #: and 30000 both find 169 presses and 50000 finds 128 -- and 128 against
    #: the adapter's 200-press cap is the difference between a comfortable
    #: recording and a tight one. It costs about fifteen minutes and ~7 GB of
    #: resident memory at the widest (the beam remembers every state it has ever
    #: kept, so it never re-enters one), once, into the disk plan cache. Kept off
    #: the normal path so the other five levels never pay for it.
    escalate_widths: tuple = (15_000, 50_000)

    #: One width for a RE-PLAN (recovery probe, detour vetting). A recovery plan
    #: has to win, not to be shortest, and these are on the interactive path.
    replan_widths: tuple = (1500,)

    #: Depth ceiling for a beam. Above level 3's 128 with room to spare; a beam
    #: that reaches it has failed, and failing costs the full sweep -- which is
    #: most of what level 3's first search spends, since its three
    #: `start_widths` all run to this depth before the escalation starts.
    beam_depth: int = 260

    def setup(self) -> None:
        g = self.g
        self.player_ids = set(self.game._engine._player_indices)
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.cone_ids = set(g.resolve_object_name("cone"))
        self.goal_ids = set(g.resolve_object_name("goalarea"))
        self.leaf_ids = set(g.resolve_object_name("leaves"))
        self.cast_ids = set(g.resolve_object_name("casting"))
        self.tor_ids = {}
        for name, t in (("tornadoup", TU), ("tornadodown", TD),
                        ("tornadoleft", TL), ("tornadoright", TR),
                        ("tornadostill", TS), ("tornadosplitvert", TV),
                        ("tornadosplithoriz", TH)):
            for i in g.resolve_object_name(name):
                self.tor_ids[i] = t
        #: `_Board`s by STATIC signature, so a level's scenery, its neighbour
        #: table and its `_Heur` fields are built once and shared by every state
        #: read from it.
        self._boards: dict = {}
        #: Level starts, by board signature -- see `_search`.
        self._starts: dict = {}
        #: Boards whose start A* could not close inside `exact_cap`. Measured
        #: once, at the start, and then believed: re-planning a beam level would
        #: otherwise burn the whole cap before every fallback.
        self._exact_ok: dict = {}
        #: `_optsets`' bounded-reachability answers, shared across levels (the
        #: key carries the state, which carries its board).
        self._probe: dict = {}

    def heuristic(self, eng) -> int:
        """Unused: `_search` plans on the model, not on engine snapshots. The
        estimate that matters is `_Heur`, and it is applied to model states."""
        raise AssertionError(
            "TornadoTamerExpert plans on the native model; heuristic is unused")

    # -- engine <-> model ------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, state)`` for the engine's current grid."""
        w = eng.width
        walls, cones, goals, leaves = [], [], [], []
        tor = {}
        player = cast_at = None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.cone_ids:
                    cones.append(i)
                if cell & self.goal_ids:
                    goals.append(i)
                if cell & self.leaf_ids:
                    leaves.append(i)
                if cell & self.player_ids:
                    player = i
                if cell & self.cast_ids:
                    cast_at = i
                for o in cell:
                    t = self.tor_ids.get(o)
                    if t is not None:
                        tor[i] = t
        sig = (eng.height, w, tuple(walls), tuple(cones), tuple(goals))
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(eng.height, w, walls, cones, goals)
        return board, (player, cast_at, tuple(sorted(tor.items())),
                       frozenset(leaves))

    def note_start(self, eng) -> None:
        """Record the board the engine is sitting on as its level's START.

        `_search` treats a start differently in two ways -- it is the state the
        anytime beam sweep is worth paying for, and it is the only state whose
        A* failure is allowed to write the board off -- so it has to be able to
        recognise one. `TornadoTamerSolver.prepare_expert` calls this once per
        level before anything plans."""
        board, state = self.read(eng)
        self._starts[board.sig] = state

    def _key(self, eng) -> frozenset:
        """``(row, col, tag)`` triples for everything that moves: the player
        (0), the Casting marker (1), each tornado (2 + its state) and the leaves
        (9).

        Built from the MODEL's reading rather than from raw object ids, so the
        key is exactly the state the search plans over -- which is also what the
        disk cache stores as a level's signature, so an edited level is a cache
        MISS rather than a wrong plan."""
        _board, (p, cast_at, tor, leaves) = self.read(eng)
        w = eng.width
        out = {(i // w, i % w, 2 + t) for i, t in tor}
        out |= {(i // w, i % w, 9) for i in leaves}
        if p is not None:
            out.add((p // w, p % w, 0))
        if cast_at is not None:
            out.add((cast_at // w, cast_at % w, 1))
        return frozenset(out)

    def _refuse(self, state) -> bool:
        """A board with no player is one this expert will not plan from.

        A split writes over whatever is in the two squares it bursts into, the
        player included. Existing tornadoes might in principle still finish the
        level on their own, so this is a REFUSAL rather than a claim of
        unwinnability -- and refusing is the safe direction: `record_level` vets
        every detour by asking for a plan, so a state this rejects is simply
        never entered.

        Deliberately NOT an override of `PSExpert.dead`, which is a hook for the
        base class's own engine-snapshot A* and takes an engine, not a state.
        Nothing here runs that search."""
        return state[0] is None

    # -- planning ---------------------------------------------------------
    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if self._refuse(state):
            return None
        if board.won(state):
            return Plan([], [])
        is_start = state == self._starts.get(board.sig)

        if self._exact_ok.get(board.sig, True):
            path = _astar(board, state, self.exact_cap)
            if path is not None:
                return Plan(path, _optsets(board, state, path, self._probe))
            if is_start:
                self._exact_ok[board.sig] = False

        best = None
        for width in (self.start_widths if is_start else self.replan_widths):
            got = _beam(board, state, width, self.beam_depth)
            if got is not None and (best is None or len(got) < len(best)):
                best = got
        if best is None and is_start:
            for width in self.escalate_widths:
                got = _beam(board, state, width, self.beam_depth)
                if got is not None and (best is None or len(got) < len(best)):
                    best = got
        if best is None:
            return None
        # A beam plan is not known shortest, so its siblings are not known to
        # tie. Every step is labelled with the press the expert took -- which is
        # a target, not a gap (see the always-emit-optimal-targets rule).
        return Plan(best, [[a] for a in best])


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class TornadoTamerSolver(PSAStarSolver):
    game_id = GAME_ID
    game_name = GAME_NAME
    expert_cls = TornadoTamerExpert

    #: `games/ps:tornado_tamer/ps:tornado_tamer.py` is a plain passthrough -- it
    #: builds the adapter and nothing else, and the rendering fix is in the .txt
    #: that both paths read. Set this if that wrapper ever grows a patch.
    game_module_id = ""

    #: Unused: `TornadoTamerExpert._search` never calls `_astar` on engine
    #: snapshots. `TornadoTamerExpert.exact_cap` is the knob that bounds the
    #: model search; these are left at the base values so nothing reads a lie.
    weight = 1
    node_cap = 400_000

    #: Room for level 3's 128 presses plus the handful a detour on another level
    #: can add, under the adapter's own 200-press per-level cap (which
    #: `record_level` restarts with the `set_level` that ends the exploration
    #: prefix, so the plan gets the whole budget). Set below 200 so the expert
    #: stops itself rather than being cut off by a GAME_OVER.
    max_steps = 195

    #: Every level is solved; nothing is skipped.
    skip_levels = frozenset()

    #: Levels whose start A* closes inside `TornadoTamerExpert.exact_cap`, so
    #: their plans are provably shortest, their steps carry exact tie sets and a
    #: re-plan costs a fraction of a second. Only these take epsilon detours --
    #: see `epsilon_for`.
    #:
    #: MEASURED, and it is a table only so that a startup which hits the disk
    #: plan cache still knows which boards are beam boards without re-deriving
    #: it (a beam board would otherwise burn the whole node cap before every
    #: fallback). ``--plans`` re-runs A* on all six starts and fails if this
    #: does not match what it finds.
    exact_levels: frozenset = frozenset({1, 2, 4})

    #: On top of the RESET prefix, on the exact levels only: about one press in
    #: twenty is a random legal alternative, after which the expert re-plans
    #: from wherever it landed -- the taken action is the mistake and
    #: ``optimal`` is the recovery. `record_level` vets each alternative by
    #: re-planning from it first, so a detour never bricks a level.
    epsilon = 0.05

    def epsilon_for(self, level: int) -> float:
        """Detours on the three levels A* closes; none on the three the beam
        closes.

        Two reasons, both about cost rather than safety. `record_level` vets a
        detour by asking the expert to plan from it, and on a beam level that is
        a multi-second search per candidate press -- paid on every episode, for
        every seed, since each seed detours somewhere new. And level 3's plan is
        128 presses against a 200-press cap, where a detour that drags a leaf
        the wrong way costs a relay to undo.

        Recovery data for those three levels comes from the exploration prefix
        and its RESET, which is the `recovery_mode = "reset"` contract and is
        unaffected by this.
        """
        return self.epsilon if level in self.exact_levels else 0.0

    def prepare_expert(self, game, expert) -> None:
        """Seat every level once -- to tell the expert which board is which
        level's start, and which of them the beam owns -- then plan them all.

        Planning here rather than letting `discover_solvable` trigger it is the
        same work either way, but it fills the disk cache in one pass and makes
        the startup cost visible as startup.
        """
        for level in range(game.n_levels):
            game.set_level(level)
            board, _state = expert.read(game._engine)
            expert.note_start(game._engine)
            expert._exact_ok[board.sig] = level in self.exact_levels
        for level in range(game.n_levels):
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    """The solver, its adapter and an expert -- without planning anything.

    The reports below each want a different slice (``--audit`` needs no plan at
    all, ``--selfcheck`` needs only the model), so planning is left to whoever
    asks for it rather than paid on every entry point."""
    solver = TornadoTamerSolver()
    game = solver.make_game(seed)
    expert = TornadoTamerExpert(game, node_cap=solver.node_cap)
    for level in range(game.n_levels):
        game.set_level(level)
        board, _state = expert.read(game._engine)
        expert.note_start(game._engine)
        expert._exact_ok[board.sig] = level in solver.exact_levels
    return solver, game, expert


def _ascii(board: _Board, state) -> str:
    """A model state as ASCII, so a failing check can print the board it failed
    on rather than four tuples."""
    player, cast_at, tor, leaves = state
    glyph = {TU: "u", TD: "d", TL: "l", TR: "r",
             TS: "s", TV: "V", TH: "H"}
    tord = dict(tor)
    out = []
    for r in range(board.h):
        line = ""
        for c in range(board.w):
            i = r * board.w + c
            if i in board.walls:
                ch = "#"
            elif i == player:
                ch = "p" if cast_at == i else "P"
            elif i in tord:
                ch = glyph[tord[i]]
            elif i in leaves:
                ch = "~"
            elif i in board.cones:
                ch = "@"
            elif i in board.goals:
                ch = "X"
            else:
                ch = "."
            line += ch
        out.append(line)
    return "\n".join(out)


def _model_walk(board: _Board, state, plan):
    """Every state the model passes through, ``state`` first."""
    out = [state]
    for a in plan:
        state = board.step(state, a)
        out.append(state)
    return out


def _plans(verbose: bool = True) -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, both searches and the tie labelling at
    once.

    Also re-derives `TornadoTamerSolver.exact_levels` by running A* on all six
    starts under the same cap the expert uses, and fails if the table disagrees:
    the table decides which levels get exact tie sets and which get epsilon
    detours, so a stale entry would quietly downgrade a level's labels.
    """
    solver, game, expert = _new()
    eng = game._engine
    bad = 0
    measured = set()
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(game._engine)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level}: NO PLAN")
            bad += 1
            continue
        # the model's own account of the plan, then the interpreter's
        model_end = _model_walk(board, state, plan)[-1]
        game.set_level(level)
        for direction in plan:
            eng.step(direction)
        _b, engine_end = expert.read(eng)
        won = eng.check_win()
        if not won:
            print(f"  L{level}: plan did not win in the interpreter")
            bad += 1
        if engine_end != model_end:
            print(f"  L{level}: model and interpreter disagree on the end state")
            bad += 1

        game.set_level(level)
        hit = _astar(board, state, expert.exact_cap)
        if hit is not None:
            measured.add(level)
            if len(hit) != len(plan):
                print(f"  L{level}: A* says {len(hit)}, plan is {len(plan)}")
                bad += 1
        sets = getattr(plan, "optsets", None)
        ties = sum(1 for s in (sets or []) if len(s) > 1)
        unlabelled = sum(1 for s in (sets or []) if not s)
        bad += unlabelled
        if verbose:
            kind = "A*, shortest" if level in measured else "beam"
            print(f"  L{level}: {len(plan):3d} presses ({kind}), lower bound "
                  f"{board.heur(state):3d}, {ties} step(s) with a tie set, "
                  f"{unlabelled} unlabelled, "
                  f"{'WIN' if won else 'NO WIN'}")
    if measured != set(solver.exact_levels):
        print(f"  exact_levels says {sorted(solver.exact_levels)}, A* closes "
              f"{sorted(measured)}")
        bad += 1
    elif verbose:
        print(f"  exact_levels {sorted(measured)} confirmed by re-running A* "
              f"on all {game.n_levels} starts")
    return bad


def _seat(eng, board: _Board, state, ids) -> None:
    """Write a model state into the interpreter's grid.

    The exact inverse of `TornadoTamerExpert.read` (which `--selfcheck` checks
    by round-tripping before it trusts a single comparison). Leaves are seated
    as DenseLeaves: the model does not distinguish the two leaf objects because
    no rule and no win condition does, so which one is written back is a
    rendering choice, not a state.
    """
    player, cast_at, tor, leaves = state
    bg, wall, cone, goal, dense, ply, cast, tors = ids
    w = board.w
    eng.grid = [[{bg} for _ in range(board.w)] for _ in range(board.h)]

    def put(i, o):
        eng.grid[i // w][i % w].add(o)

    for i in board.walls:
        put(i, wall)
    for i in board.cones:
        put(i, cone)
    for i in board.goals:
        put(i, goal)
    for i in leaves:
        put(i, dense)
    for i, t in tor:
        put(i, tors[t])
    if player is not None:
        put(player, ply)
    if cast_at is not None:
        put(cast_at, cast)
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


def _ids(game) -> tuple:
    g = game._game
    idx = g.obj_name_to_idx
    tors = {TU: idx["tornadoup"], TD: idx["tornadodown"],
            TL: idx["tornadoleft"], TR: idx["tornadoright"],
            TS: idx["tornadostill"], TV: idx["tornadosplitvert"],
            TH: idx["tornadosplithoriz"]}
    return (idx["background"], idx["wall"], idx["cone"], idx["goalarea"],
            idx["denseleaves"], idx["player"], idx["casting"], tors)


OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}


def _closure(board: _Board, state) -> tuple:
    """``(room, mov, still)`` -- an OVER-APPROXIMATION of everything that can
    ever happen on this level, as a fixed point:

      * ``room``   -- squares the player can ever stand on;
      * ``mov[d]`` -- squares a tornado travelling ``d`` can ever be on;
      * ``still``  -- squares a Still can ever stand on.

    The five closure steps are the five ways those sets feed each other:

      cast    a player in ``room`` puts a ``d`` tornado in the square ``d`` of it
      travel  a ``d`` tornado on a square is on the next square ``d`` of it too
      stall   a ``d`` tornado with a wall ``d`` of it becomes a Still there
      split   a Still that some tornado can reach along an axis becomes a split
              across that axis, which puts tornadoes in its two side squares
      step    the player walks into any non-wall neighbour of the room, and into
              a CONE only where a Still can stand (rule 23 fires before the cone
              rule, so deleting a Still is how a player crosses a cone)

    Every step drops the things that make the real game hard -- timing, one
    tornado per square, the fact that making a Still may consume the very gust
    you needed elsewhere -- so each set is a superset of the truth. That is the
    useful direction: a square this says the player CANNOT reach, it provably
    cannot. `--selfcheck` uses it for exactly one claim, that level 3 seals its
    player into a 15-square room, which is why that level costs 128 presses.
    """
    walls, cones, nb = board.walls, board.cones, board.nbmap
    _player, _cast_at, tor, _leaves = state
    room = set()
    mov = {d: set() for d in MOVES}
    still = set()
    if state[0] is not None:
        room.add(state[0])
    for i, t in tor:
        if t in TDIR:
            mov[TDIR[t]].add(i)
        elif t == TS:
            still.add(i)
        elif t == TV:
            for d in ("up", "down"):
                j = nb[(i, d)]
                if j is not None and j not in walls:
                    mov[d].add(j)
        else:
            for d in ("left", "right"):
                j = nb[(i, d)]
                if j is not None and j not in walls:
                    mov[d].add(j)

    changed = True
    while changed:
        changed = False
        frontier = deque(room)
        while frontier:
            i = frontier.popleft()
            for d in MOVES:
                j = nb[(i, d)]
                if j is None or j in walls or j in room:
                    continue
                if j in cones and j not in still:
                    continue
                room.add(j)
                frontier.append(j)
                changed = True
        for i in room:
            for d in MOVES:
                j = nb[(i, d)]
                if j is not None and j not in walls and j not in mov[d]:
                    mov[d].add(j)
                    changed = True
        for d in MOVES:
            frontier = deque(mov[d])
            while frontier:
                i = frontier.popleft()
                j = nb[(i, d)]
                if j is not None and j not in walls and j not in mov[d]:
                    mov[d].add(j)
                    frontier.append(j)
                    changed = True
        for d in MOVES:
            for i in mov[d]:
                j = nb[(i, d)]
                if j is not None and j in walls and i not in still:
                    still.add(i)
                    changed = True
        for s in list(still):
            for axis, out in ((("left", "right"), ("up", "down")),
                              (("up", "down"), ("left", "right"))):
                if not any(nb[(s, OPP[a])] is not None
                           and nb[(s, OPP[a])] in mov[a] for a in axis):
                    continue
                for d in out:
                    j = nb[(s, d)]
                    if j is not None and j not in walls and j not in mov[d]:
                        mov[d].add(j)
                        changed = True
    return room, mov, still


def _selfcheck(walk_presses: int = 500, boards: int = 1200,
               probes: int = 60, verbose: bool = True) -> int:
    """Six independent passes over the model, the searches and the labels.

    1. **round trip** -- `_seat` is proved the exact inverse of
       `TornadoTamerExpert.read` before anything else compares the two, because
       every later pass reads a state back out of the interpreter.
    2. **model vs interpreter, on play** -- a random press walk from every level
       start, compared square by square after every single press.
    3. **model vs interpreter, off play** -- randomly seated boards (random
       player, random Casting, up to eight tornadoes of random states, random
       leaves) driven four presses each. This is the pass that finds the
       mechanics a plan never visits: two splits writing into one square, a
       split landing on the player, a gust and the player contesting a square.
    4. **shortest** -- on the levels A* closes, a plain breadth-first search
       over the same model, with no estimate at all, must agree on the length.
    5. **tie sets** -- every step of those plans is re-derived by brute force:
       for each of the five presses, breadth-first search from the successor and
       keep it iff it still finishes in the moves remaining. The set must equal
       the one `_optsets` produced with its bounded searches and its prunes.
    6. **recovery** -- boards reached by random play from each level start are
       offered to `expert.plan`; a plan that comes back must win in the
       interpreter. This is the `supports_recovery` contract, measured.

    Plus the one structural claim the docstring makes: `_closure` must seal
    level 3's player into a room that does not contain its far leaf.
    """
    solver, game, expert = _new()
    eng = game._engine
    ids = _ids(game)
    rng = random.Random(20260821)
    bad = 0

    # 1. round trip
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        _seat(eng, board, state, ids)
        _b2, back = expert.read(eng)
        if back != state:
            print(f"  L{level}: seat/read round trip is not the identity")
            bad += 1
    if verbose:
        print(f"  round trip: {game.n_levels} level starts, {bad} violations")

    # 2. model vs interpreter on play
    mism = presses = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        for _ in range(walk_presses):
            act = rng.choice(DIRS)
            want = board.step(state, act)
            _seat(eng, board, state, ids)
            eng.step(act)
            _b, got = expert.read(eng)
            presses += 1
            if got != want:
                mism += 1
                if mism <= 3:
                    print(f"  L{level} press {act} diverges\n"
                          f"{_ascii(board, state)}")
                state = got
            else:
                state = want
            if board.won(state):
                game.set_level(level)
                _b, state = expert.read(eng)
    bad += mism
    if verbose:
        print(f"  model vs interpreter, on play: {presses} presses, "
              f"{mism} divergences")

    # 3. model vs interpreter off play
    mism = presses = 0
    for _ in range(boards):
        level = rng.randrange(game.n_levels)
        game.set_level(level)
        board, _start = expert.read(eng)
        free = [i for i in range(board.h * board.w) if i not in board.walls]
        cells = rng.sample(free, min(len(free), rng.randrange(0, 9) + 1))
        player = cells[0] if rng.random() < 0.9 else None
        tor = tuple(sorted((i, rng.randrange(7)) for i in cells[1:]))
        leaves = frozenset(rng.sample(free, rng.randrange(0, 7)))
        cast_at = player if (player is not None and rng.random() < 0.5) else None
        state = (player, cast_at, tor, leaves)
        for _ in range(4):
            act = rng.choice(DIRS)
            want = board.step(state, act)
            _seat(eng, board, state, ids)
            eng.step(act)
            _b, got = expert.read(eng)
            presses += 1
            if got != want:
                mism += 1
                if mism <= 3:
                    print(f"  seeded board, press {act} diverges\n"
                          f"{_ascii(board, state)}")
                state = got
            else:
                state = want
    bad += mism
    if verbose:
        print(f"  model vs interpreter, off play: {boards} seeded boards, "
              f"{presses} presses, {mism} divergences")

    # 4 + 5. shortest, and the tie sets, on the levels A* closes
    for level in sorted(solver.exact_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        plan = expert.plan(eng, level)
        ref = _bfs_plan(board, state, 4_000_000)
        if ref is None or len(ref) != len(plan):
            print(f"  L{level}: BFS says {ref and len(ref)}, plan is {len(plan)}")
            bad += 1
        sets = getattr(plan, "optsets", None) or []
        if len(sets) != len(plan):
            print(f"  L{level}: {len(sets)} optimal sets for {len(plan)} steps")
            bad += 1
            continue
        walk = _model_walk(board, state, plan)
        ties = 0
        for i, taken in enumerate(plan):
            remaining = len(plan) - i
            brute = []
            for act in DIRS:
                nxt = board.step(walk[i], act)
                if board.won(nxt):
                    if remaining == 1:
                        brute.append(act)
                    continue
                sub = _bfs_plan(board, nxt, 8_000_000, remaining - 1)
                if sub is not None and len(sub) == remaining - 1:
                    brute.append(act)
            if sorted(brute) != sorted(sets[i]):
                print(f"  L{level} step {i}: brute force {sorted(brute)} vs "
                      f"labelled {sorted(sets[i])}")
                bad += 1
            if len(brute) > 1:
                ties += 1
            if taken not in brute:
                print(f"  L{level} step {i}: taken press {taken} is not optimal")
                bad += 1
        if verbose:
            print(f"  L{level}: {len(plan)} presses confirmed shortest by an "
                  f"independent BFS, {ties} step(s) with a genuine tie, all "
                  f"{len(plan)} sets exact")

    # 6. recovery. Far fewer probes on the three beam levels, and level 3's are
    # expected to come back REFUSED: a re-plan there runs one beam at
    # `replan_widths`, which is nowhere near the 50000 its start needed, and
    # that is exactly why that level takes no detours and recovers by RESET.
    solved = refused = won = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, start = expert.read(eng)
        n = probes if level in solver.exact_levels else 3
        for _ in range(n):
            state = start
            for _ in range(rng.randrange(1, 12)):
                state = board.step(state, rng.choice(DIRS))
            if board.won(state):
                continue
            _seat(eng, board, state, ids)
            plan = expert.plan(eng, level)
            if plan is None:
                refused += 1
                continue
            solved += 1
            _seat(eng, board, state, ids)
            for direction in plan:
                eng.step(direction)
            if eng.check_win():
                won += 1
            else:
                print(f"  L{level}: a recovery plan did not win")
                bad += 1
    if verbose:
        print(f"  recovery: {solved} boards re-planned ({won} of them won in "
              f"the interpreter), {refused} refused")

    # the structural claim
    game.set_level(3)
    board, state = expert.read(eng)
    room, _mov, _still = _closure(board, state)
    stranded = [x for x in state[3] if x not in room]
    if len(room) != 15 or not stranded:
        print(f"  L3: closure gives a {len(room)}-square room with "
              f"{len(stranded)} leaf/leaves outside it")
        bad += 1
    elif verbose:
        print(f"  L3: the player is sealed into {len(room)} squares, with "
              f"{len(stranded)} of its {len(state[3])} leaves outside them -- "
              f"everything there has to be done by relay")
    return bad


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

#: Every cell stack a board of this game can hold, as (terrain, leaves, top).
#: Background is on its own layer and is everywhere. GoalArea and Cone never
#: share a square (no legend character places both and nothing moves either).
#: A Wall square holds nothing else: no level puts scenery under one, and
#: nothing can enter one.
_TERRAIN = [(), ("goalarea",), ("cone",)]
_LEAVES = [(), ("thinleaves",), ("denseleaves",)]
_TORNADOES = ("tornadoup", "tornadodown", "tornadoleft", "tornadoright",
              "tornadostill", "tornadosplitvert", "tornadosplithoriz")
#: The player carries the Casting marker; a tornado carries an AnimTornado
#: (a `late` rule puts one on every tornado and takes it off again), so no
#: composition without one is reachable and none is tested.
_TOPS = ([(), ("player",), ("player", "casting")]
         + [(t, "animtornado1") for t in _TORNADOES])
_COMPOSITIONS = [terr + leaf + top
                 for terr in _TERRAIN for leaf in _LEAVES for top in _TOPS]
_COMPOSITIONS.append(("wall",))

#: `_render_frame`'s letterbox colour. A composition that renders as a uniform
#: frame of it would have no seam against the pad (the ps:stand_iii lesson).
_PAD = 5


def _comp_name(comp) -> str:
    return "+".join(comp) or "floor"


def _audit(verbose: bool = True) -> int:
    """Three passes over every reachable cell composition.

    **Pass 1 -- distinctness**, at every cell size the six levels use (4, 5, 7,
    8 and 9 px; the size list is derived from the levels rather than assumed).
    Whole 64x64 frames of uniform boards are compared rather than one cell out
    of a mixed board: `_render_frame` upscales and centre-pads, so slicing a
    cell by ``cell_px`` arithmetic reads the wrong pixels (the ps:explod
    lesson). Two uniform boards render identically iff their cells do.

    This is the pass that fails on the shipped .txt, in three separate places at
    once -- see the header comment in
    ``data/puzzlescript_games/Tornado_Tamer.txt``.

    **Pass 2 -- nothing is the letterbox.**

    **Pass 3 -- the five animation frames are invisible.** AnimTornado1-5 are
    now blank, and this asserts it the only way that matters: a tornado must
    render identically under all five, at every size. Without that a tornado
    would look like five different things depending on when you looked at it.

    **Pass 4 -- the quarter-turn orbit.** The adapter presents the board at a
    random quarter turn. The four directional tornadoes are drawn so that rot90
    carries each one's art to the NEXT direction's exactly, and TornadoStill and
    the Player are drawn turn-invariant; this measures both, on square boards
    (a turn of a non-square board also moves the letterbox, and every comparison
    would pass for the wrong reason).
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
                   if np.array_equal(shots[a], shots[b])]
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

        moving = 0
        for t in _TORNADOES:
            frames = [shoot(h, w, (t, "animtornado%d" % k)) for k in range(1, 6)]
            if any(not np.array_equal(frames[0], f) for f in frames[1:]):
                print(f"      {t} draws differently under the anim cycle")
                moving += 1
        bad += moving
        if verbose:
            print(f"      animation frames: {len(_TORNADOES)} tornado states x "
                  f"5 phases, {moving} that move")

    order = ("tornadoup", "tornadoright", "tornadodown", "tornadoleft")
    for cell in sorted({min(64 // h, 64 // w) for h, w in sizes}):
        n = 64 // cell
        art = {o: shoot(n, n, (o, "animtornado1")) for o in order}
        art["tornadostill"] = shoot(n, n, ("tornadostill", "animtornado1"))
        art["player"] = shoot(n, n, ("player",))
        orbit = all(np.array_equal(np.rot90(art[order[i]], k=-1),
                                   art[order[(i + 1) % 4]]) for i in range(4))
        fixed = [o for o in ("tornadostill", "player")
                 if all(np.array_equal(np.rot90(art[o], k=k), art[o])
                        for k in range(4))]
        if not orbit:
            print(f"      cell {cell}px: the directional tornadoes are NOT a "
                  f"quarter-turn orbit")
            bad += 1
        if len(fixed) != 2:
            print(f"      cell {cell}px: turn-invariant art is {fixed}, "
                  f"expected still + player")
            bad += 1
        if verbose:
            print(f"  cell {cell}px ({n}x{n}): directional orbit "
                  f"{'OK' if orbit else 'BROKEN'}, turn-invariant: "
                  f"{', '.join(fixed) or 'none'}")
    return bad


def _symmetry(verbose: bool = True) -> int:
    """Replay every level's plan through the ADAPTER at every presentation it
    can draw, and assert the frames are the transform of the unaugmented ones.

    ps: games are augmented by the ADAPTER, not by `utils/arc_game.py`: it turns
    the presented frame by a quarter turn per seed and forward-remaps directional
    input to match, so a generator that plans in ENGINE space has to emit the
    SCREEN press (`screen_action`). This drives the plans through
    `perform_action` exactly as `record_level` does, at every rotation the
    adapter produces, and compares the taped frames against the k=0 run turned
    by hand. It is the check that the rotation contract is the right way round;
    get it backwards and nothing raises, the recording is just wrong on three
    presentations out of four.

    This game is NOT in `PuzzleScriptAdapter._FLIP_GAMES`, so the group here is
    the four quarter turns and nothing else. That is deliberate. The split rules
    are written as single-direction blocks scanning the grid in a fixed order,
    and two split tornadoes standing next to each other are settled by that
    order -- a fact about the screen, not about the board -- so a mirrored copy
    of such a board would settle the other way. The rotations do not have that
    problem (a turn permutes the scan consistently with the rule directions) and
    are measured here.
    """
    _solver, game, expert = _new()
    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        plans[level] = expert.plan(game._engine, level) or []

    # The adapter re-draws the presentation on every `set_level`, so a seed does
    # NOT pin one rotation for the whole episode -- each (seed, level) gets its
    # own. Read it after seating the level, never before.
    ref: dict = {}
    seen = set()
    wins = 0
    bad = 0
    for seed in range(24):
        g = TornadoTamerSolver().make_game(seed)
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
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}; "
              f"{wins} plan replays won; {len(ref)} (level, presentation) "
              f"pairs compared")
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
    sys.exit(TornadoTamerSolver.main())
