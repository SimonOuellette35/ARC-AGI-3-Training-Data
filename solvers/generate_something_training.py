"""Generate Phase-1 training data for the PuzzleScript game ps:something
("Something", arrogant.gamer).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, a native model of them, the exact distance FIELD that model is
searched to, the proof that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_something",
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
action (post rotation/flip remap), i.e. the button an agent presses in the
augmented view, so replaying the recorded actions reproduces the recorded frames
exactly. Every expert step carries the full set of equally-optimal presses.

The mechanic: one board, two complementary worlds
-------------------------------------------------
Every cell of the board carries ONE BIT, and the whole game is what that bit
means to whoever is holding the controller.

There are two characters, ``A`` (red) and ``B`` (blue), and two parallel sets of
scenery objects, ``AFloor``/``AWall`` and ``BFloor``/``BWall``. Exactly one
character is ON at a time, and the board is always drawn entirely in that
character's paradigm: while B is active every cell is either ``BFloor`` (blue,
walkable by B) or ``BWall`` (invisible, so it renders as the grey Background,
and blocks B).

Pressing ACTION -- the paradigm swap -- is three rules:

    [ action BPlayerOn ] [ BWall  ] -> [ action BPlayerOn ] [ AFloor ]
    [ action BPlayerOn ] [ BFloor ] -> [ action BPlayerOn ] [ AWall  ]
    [ action BPlayerOn ] [ APlayerOff ] -> [ BPlayerOff ] [ APlayerOn ]

Read the first two together: **every wall becomes the other character's floor
and every floor becomes the other character's wall.** The two characters walk
on exactly complementary halves of the board, and ACTION hands the controller
over along with the inversion. Two ACTIONs in a row are the identity.

The other rule is a push, and it is a plain, single-cell one:

    rigid [ > Player | Wall     ] -> [ > Player | > Wall     ]
  + rigid [ > Player | PlayerOff] -> [ > Player | > PlayerOff]

with no ``[ > Wall | Wall ]`` companion, so a CHAIN never moves: shoving a wall
whose far side is another wall (or the board edge) is refused outright, and
``rigid`` cancels the pusher's own step with it. The vacated cell is refloored
by the two ``late`` rules ("every cell has exactly one of floor / wall"), so a
push is a bit SWAP between two adjacent cells and the number of walls is
conserved. The `+` joins the two rules into one rigid body: walking into the
sleeping character shoves it, and if it is standing on a wall the wall goes with
it -- that is the "ferrying" the author's design notes are about.

The measurements behind all of that, taken against the interpreter (they are
`--selfcheck`'s first pass in executable form):

  * B into a wall whose far side is floor -> the wall slides one, the vacated
    cell turns to floor, B takes it;
  * B into a wall whose far side is a wall, or is off the board -> NOTHING moves,
    B included (rigid);
  * B into a wall with the sleeping A standing in it -> both slide together;
  * B into a cell holding the sleeping A on FLOOR -> only A slides, the terrain
    is untouched, and A may be shoved into a wall cell (different collision
    layers);
  * ACTION -> every ``BWall`` becomes ``AFloor``, every ``BFloor`` becomes
    ``AWall``, and control passes; ACTION twice is a no-op.

THE ONE INVARIANT, and the trap it disarms
------------------------------------------
The sleeping character always stands on a WALL cell -- which is exactly what
makes the swap legal, since that cell is its own floor on the other side. The
invariant holds because the only thing that moves the sleeper is a push, and a
push takes the wall it stands in along with it; and ACTION preserves it, because
the character that goes to sleep was standing on floor, which the swap turns to
wall.

It matters because the interpreter's swap rule is a same-layer OVERWRITE:
``[ action BPlayerOn ] [ BFloor ] -> [ AWall ]`` fired on a cell holding
``APlayerOff`` (layer ``AWall, APlayer``) DESTROYS the sleeper, control never
transfers, and -- since ``All APlayer on ATarget`` is VACUOUSLY true with zero
APlayers -- the survivor can then walk to its own target and the engine reports
a WIN. That is a cheat win, and the reason this generator does not have to
defend against it is that no state it can ever be pressed from is reachable:
`--selfcheck` proves the invariant over every state in every level's search
ball, and demonstrates the destruction itself on a hand-built board so the claim
is about a real branch and not a misreading.

The board never MIXES paradigms either (ACTION rewrites all of it, pushes stay
inside one, and the ``late`` refloor uses the active character's floor), so
`_Board` can carry a single wall bitmask meaning "blocks whoever is active".

Expert solver
-------------
A NATIVE model (`_Board`): a state is ``(wall bitmask, A's cell, B's cell, who
is active)``, and the interpreter is never stepped during planning. The reason
is measured, not stylistic -- this game's interpreter runs at ~385 presses/s
(three whole-board rewrite rules per ACTION plus four ``late`` sweeps per press),
so the engine-blackbox `PSExpert` would need days on the boards the model
searches in a second.

Every level is solved to a provably SHORTEST plan with EXACT optimal-action sets,
in three sweeps:

  1. **A-star** to ``d*``, under the admissible heuristic below, with the goal
     tested on POP (not on generation), so ``d*`` is proved, not guessed.
  2. A **bounded layered BFS** keeping only states with ``g + h <= d*``. Every
     state on a shortest start-to-win path satisfies that by construction, so
     the ball contains the whole shortest-path subgraph.
  3. A **backward BFS** from the won boards over `_Board.preds` (the inverses
     enumerated directly -- no successor table), pruned to the ball, which gives
     exact presses-to-win for every state a shortest plan can touch.

THE HEURISTIC. One press moves the active character one cell in the pressed
direction, and moves the sleeper one cell in the SAME direction when it is a
shove; ACTION moves neither. So the number of presses spent going ``up`` is at
least the net upward travel A still owes AND at least B's, hence at least the
max of the two -- and summing that over the four directions is admissible:

    h = sum over d in {up, down, left, right} of
        max(A's net travel in d, B's net travel in d)

It is a strict sharpening of "max of the two Manhattan distances" (which is
itself the sum's own lower bound) and it is what makes level 7 -- 50 presses
deep, with a forward space past 4 million states -- close in 210k nodes and 0.9s.

Optimal-action sets
-------------------
The field gives them EXACTLY, with no inference: at a state ``k`` presses from a
win, the optimal presses are every press whose successor is ``k - 1`` from one.
The prune cannot hide one, either: if ``s`` is on a shortest path and a press
takes it to a true ``k - 1`` state ``t``, then ``g(t) + h(t) <= (d* - k + 1) +
(k - 1) = d*``, so ``t`` is in the ball and was measured. ``--selfcheck``
re-derives every set on the small levels by brute force (a fresh bounded BFS from
each successor) and requires an exact match. No step ever ships unlabelled (the
always-emit-optimal-targets rule).

The levels
----------
Fourteen parse. Ten are winnable and are what this generator records --
**0-8 (the author's own "GOOD LEVELS") and 11** -- at

    5, 14, 8, 37, 28, 18, 30, 50, 18, 5 presses (213 total),

all provably shortest. The other four are unwinnable, and each is proved so
rather than given up on:

  * **9, 10 and 13 have no ``+`` in their ASCII at all**, i.e. no ``BTarget``, so
    ``All BPlayer on BTarget`` can never hold with B alive. They are the author's
    scratch levels ("elbow: easily broken", "a lock ... too easy to solve here")
    and the last unfinished one; the model reports them from the board, in no
    time, without a search.
  * **12 is the author's own "examples with no solution"** and is proved by
    EXHAUSTION: its whole reachable space is 1440 states and not one of them has
    a winning edge. (Its two halves are joined only through a corridor whose
    parity the two characters cannot both satisfy.)

Level 11 is the little bipartite-graph demo and wins in five presses: B steps to
its target, ACTION, A steps to its own two cells away. Level 7 is the deep one
-- the author's "unlocking" board, where B has to break a block out of a wall to
free A, then be ferried back across the screen.

Three .txt fixes were needed
----------------------------
All three are in `data/puzzlescript_games/Something.txt`, and none touches a
rule, a level layout or a win condition:

  * **``BPlayerOn`` / ``BPlayerOff`` / ``BTarget`` were ``#00f``, and ``BFloor``
    is ``blue``.** Both quantize to ARC palette index 9, so for the whole half of
    the game in which B is awake, B and its goal were drawn in the floor's own
    colour -- an invisible player on an invisible target
    ([[ps-palette-collisions]] again). They are now ``lightblue`` (10), which
    mirrors the A side's already-correct darkred floor (13) under a red figure
    (8).
  * **The characters and the targets were photographic negatives of each
    other**, so a character standing on its own target composited to a SOLID
    5x5 -- pretty, and unreadable in two separate ways that ``--audit`` caught
    at every cell size the game uses. A solid cell hides the FLOOR beneath it,
    so "A asleep on its target (in a wall)" was pixel-identical to "A awake on
    its target (on A floor)" -- and the floor's colour is the only thing on
    screen that says whose turn it is. Worse, the adapter draws the awake
    character ABOVE everything regardless of collision layer
    (`_render_frame` appends ``player_objs`` last), and the two 5x5s are opaque
    exactly where the OTHER character's target draws, so walking over the other
    goal ERASED it from the frame -- real information loss, since a policy
    re-planning from that frame can no longer see where the goal is.
    The four sprites are redrawn so figures and targets never share a pixel:
    each character is a centred 3x3 (A a ring, B a plus) and each target is a
    4-pixel mark on the cell's rim (A the corners, B the edge midpoints). The
    rim and the gaps let the terrain colour through, all six sprites stay
    invariant under the full 8-element symmetry group, and all 21 reachable
    compositions are now pairwise distinct at every cell size (``--audit``).
  * **Level 6 ended with an inline comment**, ``.##..##..+ ( I like this one
    better )``. The level parser strips the comment but keeps the line's
    length, so the board came out 7x11 with a phantom eleventh column of bare
    Background -- which the ``late`` refloor rule turns into real floor on the
    first press, handing the level a free corridor that is not in the ASCII. The
    comment is now on its own line and the board is the 7x10 the author drew.

Recovery
--------
`supports_recovery` with RESET-mode recovery. The expert re-plans from the LIVE
board: `_search` reads whatever the engine currently holds and runs all three
sweeps from there, so nothing is tied to a level start -- the start is just the
state the cache happens to be warmed at. ``epsilon`` stays 0: unlike a game with
a fully enumerated field, a detour here can land in a state whose A-star would have
to exhaust an unbounded space to declare it lost, so the recovery data comes from
the episode-wide exploration prefix and its one RESET instead.

Augmentation
------------
Engine state after reset is identical for every seed (the levels are fixed ASCII
maps), so the only per-(seed, level) variables are the presentation
augmentations: the mandatory frame rotation plus -- this game is added to
`PuzzleScriptAdapter._FLIP_GAMES` here -- an independent horizontal and vertical
flip with the matching directional action remap, which together re-sample the
board's full 8-element symmetry group. 10 levels x 16 presentations = 160.

The argument for the flips is the ESCAPE! one at its strongest. The game is
gravity-free and its input is screen-relative; both movement rules are written
with the relative ``>`` force and name no axis; the ``late`` refloor rules and
the ACTION swap are whole-board rewrites with no geometry at all; and the win
condition is positional. Nothing can contest a cell either: there is one active
character, and the at-most-two bodies a press moves land on ``p`` and ``p + d``,
which are never the same square -- so the rule-order chirality Gobble Rush has to
argue around cannot arise. No sprite encodes a facing, and all six drawn sprites
(two characters, two targets, in two colours) are invariant under the whole
group: they are concentric ring patterns about the cell centre. ``--symmetry``
measures the lot anyway, replaying both the plans and a seeded random walk at all
16 presentations.

There is deliberately no colour augmentation. Red-versus-blue is not decoration
here: the floor's colour is the ONLY thing on screen that says whose turn it is,
and a flattening recolor would erase the distinction between the two halves of
the mechanic.

Rendering
---------
`--audit` renders every cell COMPOSITION the game can reach -- three terrains
(A floor, B floor, wall) x the characters and targets each can carry -- as whole
64x64 frames of uniform boards, at every cell size the ten recorded levels use
(5, 6, 7, 8, 9, 12 and 16 px), and requires them pairwise distinct. Whole frames
rather than one cell sliced out of a mixed board, because `_render_frame`
upscales and centre-pads and cell arithmetic reads the wrong pixels (the
ps:explod lesson). A second pass turns and mirrors every composition eight ways
and checks that no transform of one is another's art, which is what
`_FLIP_GAMES` membership needs.

The composition list (21 of them) is derived from the mechanic, not guessed: an
awake character is always on its own floor and a sleeping one always in a wall
(the invariant above), so "red figure on darkred" and "red figure on grey" are
different game states and the audit checks they are different pictures. The
smallest cell the recorded levels use is 5px (level 7, 4x12), so every sprite
renders at full 5x5 resolution and none of this family's downsampling traps
(the cell_px 2 / 3 / 4 decimations) applies. The two unwinnable big boards,
which would render at 3-4px, are not recorded.

What makes the current art work is that the figure and the target occupy
DISJOINT pixels -- a centred 3x3 and four marks on the rim -- so the cell shows
three independent facts at once: its terrain colour (whose turn it is, or that
it is a wall), which character is standing there, and whose goal it is.

Usage (run from the repo root):
    python solvers/generate_something_training.py --episodes 200 \
        --out data/training_multi_level/something

    python solvers/generate_something_training.py --plans      # level report
    python solvers/generate_something_training.py --selfcheck  # model + optimality
    python solvers/generate_something_training.py --audit      # rendering
    python solvers/generate_something_training.py --symmetry   # augmentation
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

GAME_NAME = "Something"

#: Engine presses, in the order ties are broken. ACTION is a real move here (it
#: is the paradigm swap), not the usual unbound key -- see the module docstring.
PRESSES: tuple[str, ...] = ("up", "down", "left", "right", "action")

#: The four that carry a direction, and their (dr, dc).
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
_DIRS: tuple[str, ...] = ("up", "down", "left", "right")

#: Disk cache of every level's start plan AND its optimal-action sets. The three
#: sweeps are seed-independent but cost ~2.3s together (most of it on level 7),
#: which every shard `parallelize_generator` starts would otherwise repeat on
#: every core. Delete the file to re-derive it. (Measured: 3.2s cold startup
#: against 0.8s warm.)
PLAN_CACHE: Path = _REPO_ROOT / "data" / "something_plans.json"

#: Runaway guard for `_Board.astar`. The deepest shipped level closes in 210k
#: states, so this is ~20x headroom; blowing it is a loud failure rather than a
#: level silently reported unwinnable, which is what returning None would mean.
NODE_CAP = 4_000_000


class SearchTooBig(RuntimeError):
    """`_Board.astar` passed `NODE_CAP` without settling the level."""


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry (its shape and its two targets), plus the
    mechanic, plus the search.

    Cells are flat ``r * w + c`` indices. A STATE is

        ``(walls, pa, pb, active)``

    where ``walls`` is a bitmask over the whole board, ``pa`` / ``pb`` are the
    two characters' cells and ``active`` is 0 when A holds the controller and 1
    when B does. The mask means "blocks whoever is active" -- it is the ``AWall``
    set while A is awake and the ``BWall`` set while B is, and ACTION simply
    complements it. Nothing else in the game moves: the two targets are on their
    own collision layer, no rule mentions them outside the win condition, and the
    board's floor/wall bit is conserved by every push.

    Unlike the sokobans in this family the walls are DYNAMIC, so they live in the
    state and only the shape and the targets live here -- which is why one
    `_Board` serves a level rather than one wall layout.
    """

    __slots__ = ("h", "w", "n", "ta", "tb", "full", "_edge")

    def __init__(self, h: int, w: int, ta: int, tb: int):
        self.h, self.w, self.n = h, w, h * w
        self.ta, self.tb = ta, tb
        self.full = (1 << self.n) - 1
        #: ``_edge[cell][di]`` is the neighbour cell, or -1 off the board. Built
        #: once so the inner loops never do bounds arithmetic.
        self._edge = []
        for i in range(self.n):
            r, c = divmod(i, w)
            row = []
            for d in _DIRS:
                dr, dc = _DELTA[d]
                nr, nc = r + dr, c + dc
                row.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1)
            self._edge.append(tuple(row))

    # -- the mechanic --------------------------------------------------------
    def step(self, state, pi: int):
        """The state after pressing ``PRESSES[pi]``, or None when that press
        cannot be taken.

        None covers three things, all of them non-moves no shortest path can
        contain: a step off the board, a push the ``rigid`` group cancelled (the
        far side is a wall or is off the board -- and the cancel takes the
        pusher's own step with it, so the board is untouched), and the ACTION
        that would DESTROY the sleeping character. The last one is unreachable
        rather than merely refused; see the invariant in the module docstring,
        which `_selfcheck` proves over every state in every search ball.
        """
        walls, pa, pb, active = state
        if pi == 4:                                   # ACTION: swap paradigms
            off = pa if active else pb
            if not (walls >> off) & 1:
                return None                           # would destroy the sleeper
            return (self.full ^ walls, pa, pb, 1 - active)
        p = pb if active else pa
        x = self._edge[p][pi]
        if x < 0:
            return None
        wall_x = (walls >> x) & 1
        off = pa if active else pb
        if not wall_x and off != x:                   # a plain walk
            return (walls, pa, x, active) if active else (walls, x, pb, active)
        y = self._edge[x][pi]
        if y < 0:
            return None                               # the body has nowhere to go
        if wall_x and (walls >> y) & 1:
            return None                               # wall into wall: rigid cancel
        nwalls = walls ^ (1 << x) ^ (1 << y) if wall_x else walls
        noff = y if off == x else off
        if active:
            return (nwalls, noff, x, active)
        return (nwalls, x, noff, active)

    def preds(self, state):
        """Every state one press BEFORE ``state``.

        Enumerating the inverses directly is what lets the backward sweep run
        without a reverse adjacency table. Writing ``p`` for the cell the mover
        stands on now, ``q = p - d`` for the cell it came from and ``y = p + d``
        for the cell beyond it, they are:

          * **undo an ACTION** -- complement the mask and hand the controller
            back. Legal exactly when the character that is awake HERE was
            standing on a wall THERE, which the complement makes the same test
            as "it is standing on floor here", i.e. always.
          * **undo a plain walk** -- put the mover back on ``q`` and change
            nothing else. This one is available on EVERY direction, including
            the ones where something is sitting on ``y``: a press that walked
            into an empty ``p`` is not distinguishable, from ``state`` alone,
            from one that shoved something into ``y`` -- and both really are
            predecessors. Leaving it out whenever the sleeper happens to be on
            ``y`` is what a first draft did, and it cost three levels a
            genuinely tied press each, because the tie's successor got no
            backward distance at all.
          * **undo a wall push** -- the wall now on ``y`` was on ``p``, so the
            earlier mask is this one with those two bits swapped back.
          * **undo a ferry** -- the sleeper now on ``y`` was on ``p``, riding
            the wall it stood in if there was one.

        Predecessors that violate the model's invariants (a mover starting on a
        wall, a sleeper on floor) are emitted rather than filtered: `field` only
        ever keeps the ones its forward sweep already reached, so an unreachable
        pred costs one dict lookup and no correctness.
        """
        walls, pa, pb, active = state
        out = []

        # undo an ACTION
        flipped = self.full ^ walls
        awake = pa if active == 0 else pb
        if (flipped >> awake) & 1:
            out.append((flipped, pa, pb, 1 - active))

        p = pb if active else pa
        off = pa if active else pb
        for di in range(4):
            q = self._edge[p][di ^ 1]          # _DIRS pairs up/down, left/right
            if q < 0 or (walls >> q) & 1:
                continue                        # the mover cannot start in a wall
            # undo a plain walk
            out.append((walls, pa, q, active) if active
                       else (walls, q, pb, active))
            y = self._edge[p][di]
            if y < 0:
                continue
            wall_y = (walls >> y) & 1
            if wall_y:                          # undo a wall push
                w2 = walls ^ (1 << p) ^ (1 << y)
                out.append((w2, pa, q, active) if active
                           else (w2, q, pb, active))
            if off == y:                        # undo a ferry
                w2 = walls ^ (1 << p) ^ (1 << y) if wall_y else walls
                out.append((w2, p, q, active) if active
                           else (w2, q, p, active))
        return out

    def won(self, state) -> bool:
        return state[1] == self.ta and state[2] == self.tb

    def sleeper_in_wall(self, state) -> bool:
        """The model's one invariant: whoever is asleep is standing in a wall,
        which is what makes the next ACTION legal. `_selfcheck` asserts it over
        every state of every search ball."""
        walls, pa, pb, active = state
        return bool((walls >> (pa if active else pb)) & 1)

    # -- the heuristic -------------------------------------------------------
    def est(self, state) -> int:
        """Admissible presses-to-win estimate; see the module docstring.

        A press moves the awake character one cell in the pressed direction and
        the sleeping one at most one cell in the SAME direction, so the count of
        presses spent on a direction is at least the larger of the two net
        travels still owed in it."""
        w = self.w
        _walls, pa, pb, _active = state
        ra, ca = divmod(pa, w)
        rb, cb = divmod(pb, w)
        rta, cta = divmod(self.ta, w)
        rtb, ctb = divmod(self.tb, w)
        return (max(ra - rta, rb - rtb, 0) + max(rta - ra, rtb - rb, 0)
                + max(ca - cta, cb - ctb, 0) + max(cta - ca, ctb - cb, 0))

    # -- the search ----------------------------------------------------------
    def astar(self, state) -> "int | None":
        """``d*``, the length of a shortest win from ``state``, or None when this
        board can never be won from here (the search closed with nothing).

        The goal is tested on POP, not on generation: a win generated at ``g + 1``
        is only known shortest once the frontier's ``f`` has reached it. With
        `h` admissible and unit press costs that makes the answer a proof.
        Closing the frontier without a win is equally a proof the other way --
        which is how level 12 is reported unwinnable rather than given up on.
        """
        if self.won(state):
            return 0
        pq = [(self.est(state), 0, state)]
        best = {state: 0}
        best_win = None
        while pq:
            f, g, cur = heapq.heappop(pq)
            if best_win is not None and f >= best_win:
                return best_win
            if best[cur] < g:
                continue
            for pi in range(5):
                nxt = self.step(cur, pi)
                if nxt is None:
                    continue
                ng = g + 1
                if self.won(nxt):
                    if best_win is None or ng < best_win:
                        best_win = ng
                    continue
                if best.get(nxt, 1 << 30) <= ng:
                    continue
                best[nxt] = ng
                heapq.heappush(pq, (ng + self.est(nxt), ng, nxt))
            if len(best) > NODE_CAP:
                raise SearchTooBig(
                    f"{self.h}x{self.w} board passed {NODE_CAP} states")
        return best_win

    def field(self, state, d_star: int):
        """``dist``: presses-to-win for every state a shortest ``state``-to-win
        path can touch.

        Sweep 1 is a layered forward BFS pruned by ``g + h <= d_star``. Every
        state on a shortest path satisfies that bound by construction, so the
        ball holds the whole shortest-path subgraph; states outside it are
        irrelevant and are never asked about. Sweep 2 is a backward BFS from the
        won boards over `preds`, pruned to the ball.
        """
        ball = {state: 0}
        wins = []
        frontier = [state]
        for g in range(d_star):
            nxt_frontier = []
            for cur in frontier:
                for pi in range(5):
                    nxt = self.step(cur, pi)
                    if nxt is None or nxt in ball:
                        continue
                    if self.won(nxt):
                        ball[nxt] = g + 1
                        wins.append(nxt)         # terminal: never expanded
                        continue
                    if g + 1 + self.est(nxt) > d_star:
                        continue
                    ball[nxt] = g + 1
                    nxt_frontier.append(nxt)
            frontier = nxt_frontier
        dist = {win: 0 for win in wins}
        queue = deque(wins)
        while queue:
            cur = queue.popleft()
            d = dist[cur]
            for prev in self.preds(cur):
                if prev in dist or prev not in ball:
                    continue
                dist[prev] = d + 1
                queue.append(prev)
        return dist, ball

    def optimal(self, dist, state):
        """``[(press index, successor), ...]`` for every press on a shortest path
        from ``state``. Ties come out in `PRESSES` order, which is what makes a
        re-derived plan byte-identical across processes."""
        rest = dist.get(state)
        if not rest:
            return []
        out = []
        for pi in range(5):
            nxt = self.step(state, pi)
            if nxt is not None and dist.get(nxt, -1) == rest - 1:
                out.append((pi, nxt))
        return out

    def solve(self, state) -> "Plan | None":
        """The shortest press sequence from ``state``, with the EXACT optimal set
        at every step, or None if this board cannot be won from here."""
        if self.won(state):
            return Plan([], [])
        d_star = self.astar(state)
        if d_star is None:
            return None
        dist, _ball = self.field(state, d_star)
        presses, optsets = [], []
        cur = state
        for _ in range(d_star):
            best = self.optimal(dist, cur)
            if not best:                              # cannot happen; see `field`
                return None
            presses.append(PRESSES[best[0][0]])
            optsets.append([PRESSES[pi] for pi, _ in best])
            cur = best[0][1]
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class SomethingExpert(PSExpert):
    """`PSExpert`'s plan memo, level scoping and disk cache around a `_Board`
    field.

    Only the strategy underneath changes, the way `PSEnumExpert` replaces it:
    here it is "read the live board, prove ``d*``, sweep the ball forwards then
    backwards, hand back the shortest plan and its exact tie sets", so
    `heuristic` is never called and asserts rather than returning a number
    nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state is answered exactly and not by patching up a stored path.
    """

    directions = list(PRESSES)
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        g = self.g
        idx = g.obj_name_to_idx
        self.wall_ids = {idx["awall"], idx["bwall"]}
        self.a_on = idx["aplayeron"]
        self.a_off = idx["aplayeroff"]
        self.b_on = idx["bplayeron"]
        self.b_off = idx["bplayeroff"]
        self.a_target = idx["atarget"]
        self.b_target = idx["btarget"]
        #: `_Board`s by (shape, targets) -- the only STATIC facts of a level.
        self._boards: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "SomethingExpert reads an exact distance field; heuristic is unused")

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, state)`` for the engine's current grid, or ``(None, None)``
        when the board is not one this model describes (a character or a target
        missing -- levels 9, 10 and 13 ship with no ``BTarget`` at all)."""
        h, w = eng.height, eng.width
        walls = 0
        pa = pb = ta = tb = None
        active = None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls |= 1 << i
                if self.a_on in cell:
                    pa, active = i, 0
                elif self.a_off in cell:
                    pa = i
                if self.b_on in cell:
                    pb, active = i, 1
                elif self.b_off in cell:
                    pb = i
                if self.a_target in cell:
                    ta = i
                if self.b_target in cell:
                    tb = i
        if None in (pa, pb, ta, tb, active):
            return None, None
        sig = (h, w, ta, tb)
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(h, w, ta, tb)
        return board, (walls, pa, pb, active)

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if board is None:
            return None
        return board.solve(state)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class SomethingSolver(PSAStarSolver):
    game_id = "puzzlescript_something"
    game_name = GAME_NAME
    expert_cls = SomethingExpert

    #: `games/ps:something/ps:something.py` is a plain passthrough (it constructs
    #: the adapter and nothing else), so there is nothing to gain by routing
    #: through it -- but it IS what a live agent is handed, so if that wrapper
    #: ever grows a patch this must be set to ``"ps:something"``.
    game_module_id = ""

    #: Unused: the expert proves ``d*`` and reads an exact field rather than
    #: searching under a weight, so there is no optimality to trade. Left at the
    #: base values so nothing reads a lie off them. (`_Board.astar` has its own
    #: `NODE_CAP`, which raises rather than reporting a level unwinnable.)
    weight = 1
    node_cap = 400_000

    #: The longest plan is 50 presses (level 7); the rest is room for a re-plan
    #: after the exploration prefix. Stays under the adapter's own 200-step
    #: per-level budget, which `set_level` resets before the plan starts anyway.
    max_steps = 120

    #: A detour here can strand the episode in a state whose unwinnability the
    #: expert could only establish by exhausting an unbounded space, so the
    #: recovery data comes from the RESET prefix alone.
    epsilon = 0.0

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time ~2.3s visible as startup
        rather than as a mysteriously slow first seed, and it fills the disk
        cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new():
    solver = SomethingSolver()
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _plans() -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the field and the tie labelling at
    once."""
    _solver, game, expert, solvable = _new()
    eng = game._engine
    total = ties = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            why = ("no BTarget on the board" if board is None
                   else "reachable space closed with no win")
            print(f"  L{level:2d}: {eng.height:2d}x{eng.width:2d}  "
                  f"UNWINNABLE ({why})")
            continue
        for direction in plan:
            eng.step(direction)
        won = eng.check_win()
        bad += not won
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        tie_steps = sum(1 for s in sets if len(s) > 1)
        swaps = sum(1 for p in plan if p == "action")
        total += len(plan)
        ties += tie_steps
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"  L{level:2d}: {eng.height:2d}x{eng.width:2d}  h0={board.est(state):2d}"
              f"  {len(plan):3d} presses ({swaps} swaps)  win={won}"
              f"  (budget {game._max_steps}, {room})"
              f"  {tie_steps:3d} steps with a tie set")
    print(f"  recorded levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _selfcheck(walk_presses: int = 400, verbose: bool = True) -> int:
    """Five things the expert would otherwise be trusted on.

    1. **The native model is the interpreter's.** Seeded random walks on every
       level -- from the level start AND from every prefix of the level's own
       solution, which is where the interesting boards are (the
       [[escaping-limbo-solver]] lesson) -- comparing `_Board.step`'s state
       against the engine's grid after EVERY press, including the presses that
       do nothing, which is where a collision-layer mistake hides.
    2. **The destruction branch is real, and unreachable.** A hand-built board
       with the sleeper on FLOOR is stepped through the interpreter to show that
       ACTION deletes it and that the survivor can then win vacuously; then the
       invariant that forbids it is asserted over EVERY state of EVERY level's
       search ball.
    3. **The plans are shortest.** A plain forward BFS to the first win, which is
       shortest by construction and shares no code with the heuristic, the ball
       prune or `preds`, must return the field's plan length -- on the eight
       boards whose forward space fits under its cap. For the two it cannot
       reach, 3b compares `est` against the field's exact distances at every
       state of every shortest-path subgraph, which is precisely the
       admissibility that makes A-star's ``d*`` a proof.
    4. **The tie sets are exact.** Every step's optimal set is re-derived from
       each of the five successors twice: by a fresh A-star (on every level --
       a path that never touches `preds`, the ball prune or the field), and on
       the small boards by a heuristic-free depth-bounded BFS as well.
    5. **The unwinnable levels really are.** 9, 10 and 13 by having no BTarget on
       the board at all; 12 by exhaustion of its whole reachable space.
    """
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    bad = 0

    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        plans[level] = expert.plan(eng, level)

    # 1 -- the model is the interpreter's
    for level in range(game.n_levels):
        game.set_level(level)
        board, _state = expert.read(eng)
        if board is None:
            continue
        plan = plans[level] or []
        # Anchors: the level start, plus every prefix of its own solution.
        anchors = list(range(len(plan) + 1))
        drift = 0
        per_anchor = max(8, walk_presses // max(1, len(anchors)))
        for anchor in anchors:
            game.set_level(level)
            for direction in plan[:anchor]:
                eng.step(direction)
            _b, state = expert.read(eng)
            rng = random.Random(f"something:selfcheck:{level}:{anchor}")
            for _ in range(per_anchor):
                pi = rng.randrange(5)
                eng.step(PRESSES[pi])
                nxt = board.step(state, pi)
                state = state if nxt is None else nxt
                _b, live = expert.read(eng)
                if live != state:
                    drift += 1
                    break
                if eng.check_win() != board.won(state):
                    drift += 1
                    break
                if eng.check_win():
                    break
        bad += drift
        if verbose:
            print(f"  L{level:2d}: model {'MATCHES' if not drift else 'DIVERGES from'}"
                  f" the interpreter over {len(anchors)} anchored walks")

    # 2 -- the destruction branch, and the invariant that forbids it
    idx = game._game.obj_name_to_idx
    bg = idx["background"]
    game.set_level(0)
    eng.load_level([
        [{bg, idx["bfloor"]}, {bg, idx["bfloor"]}, {bg, idx["bfloor"]}],
        [{bg, idx["bfloor"], idx["aplayeroff"]}, {bg, idx["bfloor"]},
         {bg, idx["bfloor"]}],
        [{bg, idx["bfloor"], idx["bplayeron"]}, {bg, idx["bfloor"]},
         {bg, idx["bfloor"], idx["btarget"]}],
    ])
    eng.step("action")
    survivors = sum(1 for row in eng.grid for cell in row
                    if idx["aplayeroff"] in cell or idx["aplayeron"] in cell)
    eng.load_level([
        [{bg, idx["bfloor"]}, {bg, idx["bfloor"]}],
        [{bg, idx["bfloor"], idx["bplayeron"], idx["btarget"]},
         {bg, idx["bfloor"]}],
    ])
    vacuous = eng.check_win()
    demo_ok = survivors == 0 and vacuous
    bad += not demo_ok
    if verbose:
        print(f"  ACTION with the sleeper on floor deletes it ({survivors} left) "
              f"and 'All APlayer on ATarget' is then vacuously true "
              f"({vacuous}) -- {'as modelled' if demo_ok else 'UNEXPECTED'}")

    violations = 0
    balls = {}
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        if board is None or plans[level] is None:
            continue
        _dist, ball = board.field(state, len(plans[level]))
        balls[level] = (board, state, ball)
        violations += sum(1 for s in ball if not board.sleeper_in_wall(s))
    bad += violations
    if verbose:
        reach = sum(len(b[2]) for b in balls.values())
        print(f"  the sleeper is in a wall in all {reach} states of the "
              f"{len(balls)} search balls -- {violations} violation(s), so no "
              f"reachable state can press the deleting ACTION")

    # 3 -- the plans are shortest, by an independent heuristic-FREE BFS
    #
    # Plain forward BFS to the first win is shortest by construction and shares
    # no code with the heuristic, the ball prune or `preds`. It is affordable on
    # eight of the ten boards; level 7's forward space passes 4 million states
    # before its 50-press win and level 8's is 3.2 million, so `BFS_CAP` decides
    # which ones it settles and check 3b covers what it cannot reach.
    BFS_CAP = 3_500_000
    unsettled = []
    for level in sorted(balls):
        board, state, _ball = balls[level]
        seen = {state}
        queue = deque([(state, 0)])
        shortest = None
        while queue and shortest is None:
            cur, d = queue.popleft()
            for pi in range(5):
                nxt = board.step(cur, pi)
                if nxt is None or nxt in seen:
                    continue
                if board.won(nxt):
                    shortest = d + 1
                    break
                seen.add(nxt)
                queue.append((nxt, d + 1))
            if len(seen) > BFS_CAP:
                break
        if shortest is None:
            unsettled.append(level)
            if verbose:
                print(f"  L{level:2d}: BFS passed {BFS_CAP} states without a win "
                      f"-- left to the admissibility check")
            continue
        ok = shortest == len(plans[level])
        bad += not ok
        if verbose:
            print(f"  L{level:2d}: field {len(plans[level])} presses vs BFS "
                  f"{shortest} ({len(seen)} states) -- "
                  f"{'SHORTEST' if ok else 'NOT SHORTEST'}")

    # 3b -- the heuristic is admissible wherever the proof reads it
    #
    # A-star's answer is a proof exactly insofar as `est` never over-estimates,
    # and the field is exact presses-to-win on the whole shortest-path subgraph,
    # so this compares the two on every state that proof touched. It is the
    # check that covers the level(s) the BFS above could not reach.
    over = 0
    checked = 0
    for level in sorted(balls):
        board, state, _ball = balls[level]
        dist, _b = board.field(state, len(plans[level]))
        for st, d in dist.items():
            checked += 1
            if board.est(st) > d:
                over += 1
    bad += over
    if verbose:
        print(f"  est() <= true distance at all {checked} states of the "
              f"{len(balls)} shortest-path subgraphs -- {over} over-estimate(s)"
              f"{' (this is what settles ' + str(unsettled) + ')' if unsettled else ''}")

    # 4 -- the tie sets, re-derived twice
    #
    # (a) On EVERY level, by a fresh A-star from each of the five successors of
    #     each plan step: a press belongs in the set iff its successor is
    #     exactly ``rest - 1`` presses from a win. That path never touches
    #     `preds`, the ball prune or the field, so it is an independent
    #     derivation of the labels the corpus ships.
    # (b) On the boards small enough to afford it, the same question answered by
    #     a heuristic-free depth-bounded BFS -- the one derivation an
    #     inadmissible `est` could not fool.
    def brute(board, state, limit):
        """Presses to a win from ``state``, searched fresh with no heuristic at
        all, or None past ``limit``."""
        if board.won(state):
            return 0
        seen = {state}
        queue = deque([(state, 0)])
        while queue:
            cur, d = queue.popleft()
            if d >= limit:
                continue
            for pi in range(5):
                nxt = board.step(cur, pi)
                if nxt is None or nxt in seen:
                    continue
                if board.won(nxt):
                    return d + 1
                seen.add(nxt)
                queue.append((nxt, d + 1))
        return None

    #: Levels whose plan is short enough for (b): the bounded BFS re-explores
    #: the whole ball once per successor per step, so its cost is roughly the
    #: plan length times the space, and the deep boards (3, 6, 7) would take
    #: hours to say what (a) says in seconds. The six it does cover contain
    #: every move shape the game has -- walks, wall pushes, ferries and swaps.
    BRUTE_LEVELS = frozenset({0, 1, 2, 4, 5, 11})
    for level in sorted(balls):
        board, state, _ball = balls[level]
        plan = plans[level]
        sets = getattr(plan, "optsets", None) or []
        mismatched = {"astar": 0, "bfs": 0}
        cur = state
        for i, press in enumerate(plan):
            rest = len(plan) - i
            truth = {"astar": [], "bfs": []}
            for pi in range(5):
                nxt = board.step(cur, pi)
                if nxt is None:
                    continue
                if board.astar(nxt) == rest - 1:
                    truth["astar"].append(PRESSES[pi])
                if level in BRUTE_LEVELS and brute(board, nxt, rest - 1) == rest - 1:
                    truth["bfs"].append(PRESSES[pi])
            for how in ("astar", "bfs"):
                if how == "bfs" and level not in BRUTE_LEVELS:
                    continue
                if sorted(truth[how]) != sorted(sets[i]):
                    mismatched[how] += 1
            cur = board.step(cur, PRESSES.index(press))
        bad += mismatched["astar"] + mismatched["bfs"]
        if verbose:
            tail = ("" if level not in BRUTE_LEVELS else
                    f", {mismatched['bfs']} against a heuristic-free BFS")
            print(f"  L{level:2d}: {len(plan)} tie sets, "
                  f"{mismatched['astar']} mismatch against a fresh A-star"
                  f"{tail}")

    # 5 -- the four unwinnable levels, each proved its own way
    for level in range(game.n_levels):
        if plans[level] is not None:
            continue
        game.set_level(level)
        board, state = expert.read(eng)
        if board is None:
            has_b_target = any(expert.b_target in cell
                               for row in eng.grid for cell in row)
            ok = not has_b_target
            bad += not ok
            if verbose:
                print(f"  L{level:2d}: UNWINNABLE -- no BTarget on the board, so "
                      f"'All BPlayer on BTarget' can never hold "
                      f"({'confirmed' if ok else 'WRONG'})")
            continue
        seen = {state}
        queue = deque([state])
        winning = 0
        while queue:
            cur = queue.popleft()
            for pi in range(5):
                nxt = board.step(cur, pi)
                if nxt is None:
                    continue
                if board.won(nxt):
                    winning += 1
                    continue
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        bad += bool(winning)
        if verbose:
            print(f"  L{level:2d}: UNWINNABLE -- {len(seen)} states enumerated, "
                  f"{winning} winning edges "
                  f"({'proved' if not winning else 'WRONG'})")
    return bad


#: Every cell composition the mechanic can put on screen, as
#: ``(label, object names)``. Derived from the rules, not guessed: while a
#: character is awake the board is drawn in ITS paradigm, so an awake character
#: stands on its own floor and a sleeping one stands in a wall (which draws as
#: bare Background) -- and either target can be sitting on any of the three
#: terrains, because a target never moves while the floor under it inverts.
_COMPOSITIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("wall", ()),
    ("wall+Atarget", ("atarget",)),
    ("wall+Btarget", ("btarget",)),
    ("wall+Asleep", ("aplayeroff",)),
    ("wall+Asleep+Atarget", ("aplayeroff", "atarget")),
    ("wall+Asleep+Btarget", ("aplayeroff", "btarget")),
    ("wall+Bsleep", ("bplayeroff",)),
    ("wall+Bsleep+Atarget", ("bplayeroff", "atarget")),
    ("wall+Bsleep+Btarget", ("bplayeroff", "btarget")),
    ("Afloor", ("afloor",)),
    ("Afloor+Atarget", ("afloor", "atarget")),
    ("Afloor+Btarget", ("afloor", "btarget")),
    ("Afloor+Aawake", ("afloor", "aplayeron")),
    ("Afloor+Aawake+Atarget", ("afloor", "aplayeron", "atarget")),
    ("Afloor+Aawake+Btarget", ("afloor", "aplayeron", "btarget")),
    ("Bfloor", ("bfloor",)),
    ("Bfloor+Atarget", ("bfloor", "atarget")),
    ("Bfloor+Btarget", ("bfloor", "btarget")),
    ("Bfloor+Bawake", ("bfloor", "bplayeron")),
    ("Bfloor+Bawake+Atarget", ("bfloor", "bplayeron", "atarget")),
    ("Bfloor+Bawake+Btarget", ("bfloor", "bplayeron", "btarget")),
)


def _audit(verbose: bool = True) -> int:
    """Two passes over every reachable cell COMPOSITION.

    **Pass 1 -- distinctness**, at every cell size the recorded boards use.
    Whole 64x64 frames of uniform boards are compared rather than one cell out of
    a mixed board: `_render_frame` upscales and centre-pads, so slicing a cell by
    ``cell_px`` arithmetic reads the wrong pixels (the ps:explod lesson). Two
    uniform boards render identically iff their cells do.

    This pass is what convicted the game's ORIGINAL art (see the module
    docstring): a character and its target were drawn as photographic negatives,
    so "on target" filled the cell and hid the floor colour that says whose turn
    it is, and each character was opaque exactly where the other's target draws,
    so standing on the other goal erased it. Both show up here as an IDENTICAL
    pair, at every cell size.

    **Pass 2 -- the group.** The game is added to `_FLIP_GAMES`, so a board is
    drawn at any of the eight turns and mirrors. All six sprites here are
    centred, rim- or ring-symmetric patterns and should be invariant under the
    whole group; this measures that, and separately checks that no transform of
    one composition is another composition's art. It runs on SQUARE boards,
    because a rotation of a non-square board also moves the letterbox and every
    comparison would pass for the wrong reason."""
    from adapters.puzzlescript_adapter import _render_frame            # noqa: PLC0415

    _solver, game, expert, solvable = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    sizes: dict = {}
    for level in solvable:
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    def shoot(h, w, objs):
        eng.height, eng.width = h, w
        eng.grid = [[{idx[o] for o in objs} | {idx["background"]}
                     for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        return np.asarray(_render_frame(eng, g)).copy()

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {name: shoot(h, w, objs) for name, objs in _COMPOSITIONS}
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        if verbose:
            print(f"  {h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): {len(shots)} "
                  f"compositions, {'OK' if not clashes else 'CLASH'}")
        for a, b in clashes:
            print(f"      IDENTICAL: {a}  ==  {b}")

    def transform(frame, k, hflip, vflip):
        if k:
            frame = np.rot90(frame, k=k)
        if hflip:
            frame = np.fliplr(frame)
        if vflip:
            frame = np.flipud(frame)
        return np.ascontiguousarray(frame)

    group = [(k, hf, vf) for k in range(4)
             for hf in (False, True) for vf in (False, True)]
    for cell in sorted({min(64 // h, 64 // w) for h, w in sizes}):
        n = 64 // cell
        shots = {name: shoot(n, n, objs) for name, objs in _COMPOSITIONS}
        clashes = [(a, b, t) for a, b in itertools.permutations(shots, 2)
                   for t in group
                   if np.array_equal(transform(shots[a], *t), shots[b])]
        variant = [name for name in shots
                   if any(not np.array_equal(transform(shots[name], *t),
                                             shots[name]) for t in group)]
        bad += len(clashes) + len(variant)
        if verbose:
            print(f"  cell {cell}px ({n}x{n}): {len(clashes)} composition(s) "
                  f"whose transform is another's art, {len(variant)} not "
                  f"group-invariant")
        for a, b, t in clashes:
            print(f"      {a} under {t} == {b}")
        for name in variant:
            print(f"      NOT INVARIANT: {name}")
    return bad


def _symmetry(walk_presses: int = 200, verbose: bool = True) -> int:
    """Replay every recorded level through the ADAPTER at every presentation it
    can draw and require the frames to be exactly the transform of the
    unaugmented ones.

    This is the evidence for putting the game in `_FLIP_GAMES` (the rotation is
    mandatory and is checked here too). The structural argument is in the module
    docstring -- two rules, both written with the relative ``>`` force, moving at
    most two bodies onto squares that can never coincide, plus two whole-board
    rewrites with no geometry -- but Gobble Rush's chirality hid inside exactly
    that kind of argument, so it is measured.

    Both the PLANS and a seeded random walk are replayed: the walk reaches the
    boards a plan never visits (a wall jammed against the edge, the sleeper
    ferried into a dead pocket) and it presses ACTION far more often than a plan
    does."""
    solver, game, expert, solvable = _new()
    plans = {}
    for level in solvable:
        game.set_level(level)
        plans[level] = expert.plan(game._engine, level)

    def drive(gm, level, presses):
        gm.set_level(level)
        out = [np.asarray(gm._current_frame)]
        for act in presses:
            fd = gm.perform_action(ActionInput(id=act))
            out.append(np.asarray(fd.frame[-1] if fd.frame else gm._current_frame))
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
    for seed in range(400):
        if len(seen) == 16 and seed > 40:
            break
        gm = solver.make_game(seed)
        for level in solvable:
            gm.set_level(level)
            # The presentation is per (seed, LEVEL) -- `set_level` re-seeds from
            # ``seed + idx`` -- so it must be read AFTER the set_level, not once
            # per seed (the ps:pegs lesson).
            k = (gm._rotation_k, gm._hflip, gm._vflip)
            seen.add(k)
            rng = random.Random(f"something:symmetry:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(walk_presses)]
            for tag, presses in (
                    ("plan", [screen_action(d, *k) for d in plans[level]]),
                    ("walk", [inverse_remap_action_full(a, *k) for a in walk])):
                frames = drive(gm, level, presses)
                if tag == "plan" and gm._state != GameState.WIN:
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
    sys.exit(SomethingSolver.main())
