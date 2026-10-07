"""Generate Phase-1 training data for the PuzzleScript game ps:twinpush
("TwinPush", Alex & Michael).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, a native model of them, the exact distance FIELDS that model is
searched with, the proof that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_twinpush",
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

Result: all 4 levels, 236 presses, every plan PROVABLY SHORTEST and every step
labelled with an exact optimal set.

The game: one sokoban played twice, on one keyboard
---------------------------------------------------
The whole rule listing is two lines and the second one only plays a sound:

    [ >  Player | Crate ] -> [ >  Player | > Crate ]  sfx1
    [ >  Player ]         -> [ >  Player ]            sfx0

so the mechanic is the orthodox sokoban push, resolved under the collision
layers ``Background`` / ``Target`` / ``Player, Wall, Crate``:

  * step onto a wall or off the board -> nothing happens;
  * step onto a crate -> it slides one further, UNLESS that cell is a wall or
    holds another crate, in which case nothing happens AT ALL -- the rule's
    right-hand side moves both bodies or neither, so the player's own move is
    cancelled with the crate's, which is also why two crates in a row are a
    wall;
  * anything else -> the player moves.

``Target`` is on its own collision layer, so it is scenery a crate or a player
stands ON rather than something that can be pushed, and no rule creates or
destroys one. The win condition is the orthodox one too:

    All Target on Crate

What makes it a puzzle is the level geometry. **Every one of the four levels is
15x26 and split down the middle by a solid double column of Wall, with a Player,
a Crate and a Target on each side.** Nothing crosses that wall -- a player
cannot walk through it, a crate is only ever pushed to an adjacent floor cell --
so a level is TWO INDEPENDENT SOKOBANS that happen to share a keyboard, and the
win is both of them at once. That is the title: a press pushes on both sides.

  * L0 "warm-up": two bare 3x13 shafts. On the left the crate is above the
    player and its target is above the crate, so the answer is UP; on the right
    everything is upside down and the answer is DOWN. Each side alone takes 5
    presses. Together they take 15, because the two sides want opposite keys and
    the only way to give one side a press the other can ignore is to walk into a
    wall.
  * L1: two open rooms with an interior block each. 46 and 43 presses alone, 73
    together.
  * L2: two mazes. 44 and 39 alone, 61 together.
  * L3: two dense mazes, the longest at 87. 49 and 56 alone.

Expert solver
-------------
A NATIVE model of the mechanic above, searched to exact distance FIELDS rather
than heuristically. The per-room answers are not estimates: every room's field
is enumerated, so a state's distance, its tie set and "this board is dead" are
all read off a table.

**The decomposition.** A level's floor splits into connected components of
everything that is not a Wall, and nothing crosses a boundary, so a level is a
set of independent boards -- `_Room` -- driven by one keyboard. All four levels
of this game split into exactly two, each with one player, one crate and one
target. (The model does not assume those counts: a room carries a SET of crates
and a SET of targets, and a room with no player at all is handled as frozen
scenery whose half of the win condition is decided at reset.)

**Per room, an exact field.** `_Room.field` enumerates the room's whole
reachable state space ``(player, sorted crates)`` forward, then runs a backward
BFS from every WON state over inverted moves, pruned to that reachable set. The
prune loses nothing -- a shortest path out of a reachable state is itself
reachable -- so the value at every reachable state is its true distance, and
**a reachable state absent from the field is PROVED unwinnable**, not "the
search gave up". The rooms are 1482 to 14276 states, and enumerating all eight
of them costs about a second.

**Where the two players make it hard.** Both rooms take the same press, so the
answer is the shortest sequence that wins BOTH -- a path in the PRODUCT, and the
product is not enumerable: a plain joint BFS passes 6 million states on three of
the four levels without closing. It does not have to be. Each room's own field
is an exact lower bound on the joint answer (a joint plan of length n hands room
i a sequence of n presses ending in a won state, so n >= d_i), which makes

    h(l, r) = max(d_left(l), d_right(r))

admissible, and A* under it finds d* in 116 to 210100 pops. The joint optimal
SETS then come from a second pass: a forward BFS keeping only states with
``g + h <= d*``, and a backward BFS from the won ones over the edges the forward
pass recorded. Every press on a shortest joint path is in that corridor by
construction (if ``dist(s') = d* - i - 1`` then ``g(s') + h(s') <= d*``), and a
corridor value can only be too high, so comparing them against ``d* - i - 1`` is
exact in both directions.

That is the reusable shape, and it is the one
``solvers/generate_straighten_up_training.py`` needed for its "Stunt Double"
level: **when one key drives several independent boards, do not search the
product blind -- solve each board exactly and let the MAX of those distances be
the heuristic.** TwinPush is the game that is nothing BUT that situation, four
levels of it, so the piece that was one level's problem there is the whole
solver here.

**How good is that heuristic, exactly?** ``--levelsets`` answers it and the
answer is "as good as per-room information can possibly be". For each room it
computes the BACKWARD LEVEL SETS ``B(0) = won``, ``B(n+1) = {s : some press
lands in B(n)}`` -- so ``s`` is in ``B(n)`` iff room i can be won from ``s`` in
EXACTLY n presses -- and measures two things:

  * ``B(n)`` is UP-CLOSED (``B(n) <= B(n+1)``) on every room of every level, and
    the sequence reaches its fixpoint after 26 to 91 steps. So the set of
    lengths at which a room can be won from ``s`` is exactly ``{n : n >= d(s)}``
    with no holes: a room can always waste a press and still win. That is what
    makes ``max`` not merely admissible but the tightest bound available from
    the rooms taken separately -- the sharper-looking heuristic "the smallest n
    that BOTH rooms can win at" is equal to it, and was measured to expand the
    identical number of nodes.
  * ``d(s) = min {n : s in B(n)}`` re-derives the whole field, forward, sharing
    no code with `_Room.field`'s backward sweep. The two agree on every state of
    every room.

Optimal-action sets
-------------------
Exact everywhere, with no inference: at a joint state ``d`` presses from a win
the optimal presses are exactly those whose successor is ``d - 1`` from one, and
the corridor sweep has both numbers. ``--selfcheck`` re-derives a sample of them
from scratch with a fresh joint A* per candidate press, and ``--engine``
re-derives every one of L0's 15 from the real interpreter.

The measured answer is that the shared key REMOVES almost all the freedom an
open-floored sokoban looks like it should have: only 20 of the 236 steps have a
second equally-right press, and not one has a third. Either room alone is a wide
puzzle -- L1's left room has 5451 winnable states and 46 presses of slack to
arrange them in -- but a press has to be right for one room and harmless for the
other at the same time, and being harmless is expensive: the only way to give a
room a press it can ignore is to walk it into a wall or to spend a pair of moves
undoing one, and there is usually exactly one wall in reach. What looks from one
side like a free choice of how to idle is, on the other side, the move that has
to happen next.

Shortest, and how that is known
-------------------------------
All four plans are provably shortest, by independent derivations agreeing:

  * this file's per-room fields plus the product A*, which is complete under an
    admissible heuristic at weight 1;
  * a plain forward BFS to the first win in each room, which is shortest by
    construction and shares no code with the fields' backward half, confirming
    the per-room lower bounds the joint search rests on (``--selfcheck`` pass 3);
  * for L0, whose joint product IS small enough (118206 states), a plain joint
    forward BFS over the model that never consults a heuristic at all
    (``--selfcheck`` pass 3);
  * for L0 again, a full enumeration of the real INTERPRETER -- `StateGraph.build`
    walking all 118133 reachable engine states over 472532 actual button
    presses, with no native model involved (``--engine``). It agrees on the
    length and on all 15 tie sets. L1-L3 are out of reach that way (their products pass 6M states),
    so for those the native model is fuzz-verified against the interpreter
    instead, which is what makes the other derivations statements about this
    game rather than about a model of it.

Recovery
--------
`supports_recovery` with RESET-mode recovery. The expert re-plans from the LIVE
board: `_search` reads whatever the engine currently holds, and because a room's
field covers its whole reachable space, an exploration prefix that leaves the
board anywhere is answered exactly -- including "this board is now dead", which
is what `record_level` needs to hear to fall back to a RESET. That is not a rare
corner in a one-crate sokoban: shove the crate into a corner, or flat against a
wall its target is not on, and the level is over. ``--selfcheck``'s fifth pass
perturbs 640 boards -- 80 per level at each of two walk lengths -- and checks
both halves against the interpreter: every plan returned from a random board is
replayed and must WIN, and every refusal must name a room an independent forward
BFS also proves dead. 516 replanned to a win, 124 were proved dead, none was
wrong.

The two walk lengths are there because they measure different halves and the
short one alone would not have covered the second. A 1-29 press walk -- which is
what the recorded exploration prefix actually does -- almost always leaves the
board still winnable: 306 of its 320 boards were, and every one of the 14 dead
ones was on L0, whose rooms are bare 3x13 shafts where a crate shoved to a side
wall is stuck immediately. It takes 30-199 presses to reliably kill a crate in
the maze levels, and under that regime 110 of 320 boards were dead. Both
regimes matter: the first is the distribution generation actually meets, the
second is the one that exercises the claim that a refusal is a PROOF.

A refusal has one other possible cause, and it is reported separately rather
than swept in with the dead rooms: both rooms alive but no common press sequence
winning them together, which the joint A* can only establish by closing the
whole product. `_Level.joint_node_cap` bounds that, so such a refusal is
"unproved" and the recording falls back to a RESET, which is safe either way.
``--selfcheck`` counts how often it happens; over all 640 perturbed boards it
never did, and it cannot happen on the four level STARTS, which are all solved.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`), which together re-sample the board's
8-element symmetry group. 4 levels x 16 presentations = 64.

The structural argument for the flips: the game is gravity-free, its push is
written with the relative ``>`` force and names no axis, input is
screen-relative, the win condition names no direction, and no sprite encodes a
facing (there is one Player object and no direction states). The one thing worth
checking rather than asserting -- because this is a two-player game and Gobble
Rush's chirality hid in exactly that gap -- is whether two bodies can ever
contest a cell, since the interpreter would then settle it by the order it
expands a rule's four directions and "the leftward one wins" would be a fact
about the screen. They cannot: the two players are in wall-separated components,
so no press can bring their bodies or their crates near each other, and within
one room a press moves the player to ``p + d`` and its crate to ``p + 2d``,
which are never the same square. ``--symmetry`` measures the lot anyway: every
level's plan AND a seeded 200-press random walk (which does what a plan never
does -- shoves crates into walls, into each other and into dead corners, and
presses the unbound ACTION key) replayed at all 16 presentations, requiring
every frame to be the exact transform of the unaugmented one.

There is deliberately no colour augmentation: the crate's orange against the
target's dark blue IS the win condition being rendered, and a flattening recolor
would erase it.

Rendering
---------
TWO sprites had to change, and between them they were the pieces AND the win
condition. Every level here is 15x26, which `_render_frame` draws at
``cell_px = 2`` -- the smallest cell size in the ps: set -- and a 2x2 cell
samples a 5x5 sprite at exactly four pixels: rows 1 and 3 by columns 1 and 3.
The Crate shipped as a hollow box whose outline is on rows/cols 0 and 4, so ALL
FOUR sampled pixels were transparent and a crate rendered as precisely the cell
underneath it: a crate on floor was floor, a crate on a Target was a Target. The
Player's four were all opaque, so a player standing on a Target painted it out.
See the header comment in ``data/puzzlescript_games/TwinPush.txt`` for the full
statement. The fix in both sprites is the same and is forced by the geometry:
paint three of the four sampled pixels and leave the fourth transparent.

``--audit`` renders every cell COMPOSITION the game can show -- floor, wall,
target, crate, crate on target, player, player on target -- as a whole 64x64
frame of a uniform board, at every cell size the levels render at, and requires
them pairwise distinct; then it repeats the comparison across all eight
transforms of the square's symmetry group, which is what the flips need. Whole
frames rather than one cell sliced out of a mixed board: `_render_frame`
upscales and centre-pads, so slicing by ``cell_px`` arithmetic reads the wrong
pixels (the ps:explod lesson). ``player + crate`` is absent on purpose and is
not an omission -- they share a collision layer, so the engine can never put
them in one cell.

Usage (run from the repo root):
    python solvers/generate_twinpush_training.py --episodes 200 \
        --out data/training_multi_level/twinpush

    python solvers/generate_twinpush_training.py --plans      # level report
    python solvers/generate_twinpush_training.py --levelsets  # the h bound
    python solvers/generate_twinpush_training.py --selfcheck  # model + optimality
    python solvers/generate_twinpush_training.py --engine     # interpreter proof
    python solvers/generate_twinpush_training.py --audit      # rendering
    python solvers/generate_twinpush_training.py --symmetry   # augmentation
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

GAME_NAME = "TwinPush"

#: Engine directions, in the order ties are broken. ACTION5 is bound to nothing
#: in this game (no rule has ``action`` on its left-hand side, and pressing it
#: leaves every level's grid untouched -- ``--selfcheck`` measures that), so it
#: is not a move and branching on it would inflate every search for nothing. The
#: exploration prefix still presses it -- a live agent has that button -- which
#: is why ``--symmetry``'s random walk presses it too.
DIRS: tuple[str, ...] = ("up", "down", "left", "right")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: ``DIRS`` index of the opposite direction, for the backward sweep.
_OPPOSITE = (1, 0, 3, 2)

#: Disk cache of every level's start plan AND its optimal-action sets. The
#: searches are the whole cost of generation here (a cold derivation enumerates
#: the eight rooms and runs four product A*s, about ten seconds), and every
#: `parallelize_generator` shard would otherwise repeat it. Delete the file to
#: re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "twinpush_plans.json"


# ---------------------------------------------------------------------------
# The native model -- one connected room
# ---------------------------------------------------------------------------

class _Room:
    """One connected component of a level's floor: its geometry, its crates and
    targets, the mechanic, and its exact distance-to-win field.

    Cells are flat ``r * w + c`` indices into the WHOLE board (rooms share a
    coordinate system, which is what lets `_Level` decode a joint state back
    into a grid without bookkeeping). A room STATE is the tuple
    ``(player, *sorted crate cells)``.

    A room with no player is a special case handled by `_Level`: its crates can
    never move, so it has no state at all and the level is winnable only if its
    targets are already covered.
    """

    __slots__ = ("h", "w", "cells", "solid", "edge", "player", "crates",
                 "targets", "_reachable", "_dist")

    def __init__(self, h, w, cells, solid, player, crates, targets):
        self.h, self.w = h, w
        #: The component's walkable cells.
        self.cells = frozenset(cells)
        self.solid = solid                     # bytearray over the whole board
        self.player = player                   # start cell, or None
        self.crates = tuple(sorted(crates))
        self.targets = frozenset(targets)
        #: ``edge[cell][d]`` is the neighbour cell, or -1 off the board. Built
        #: once so the inner loops never do bounds arithmetic.
        self.edge = []
        for i in range(h * w):
            r, c = divmod(i, w)
            row = []
            for d in DIRS:
                dr, dc = _DELTA[d]
                nr, nc = r + dr, c + dc
                row.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1)
            self.edge.append(tuple(row))
        self._reachable: set = set()
        self._dist: dict = {}

    # -- the mechanic ---------------------------------------------------------
    def step(self, state: tuple, di: int) -> tuple:
        """The state after pressing ``DIRS[di]``, or ``state`` if the press does
        nothing (which is a non-move: no shortest path contains one).

        This is the single push rule resolved under the collision layers. Note
        the third branch: a blocked push cancels the PLAYER's move as well, so
        two crates in a row -- and a crate against a wall -- are walls."""
        p = state[0]
        n1 = self.edge[p][di]
        if n1 < 0 or self.solid[n1]:
            return state
        k = -1
        for j in range(1, len(state)):
            if state[j] == n1:
                k = j
                break
        if k < 0:                              # a bare walk
            return (n1,) + state[1:]
        n2 = self.edge[n1][di]
        if n2 < 0 or self.solid[n2]:
            return state
        for j in range(1, len(state)):
            if state[j] == n2:                 # a second crate behind the first
                return state
        body = list(state[1:])
        body[k - 1] = n2
        body.sort()
        return (n1,) + tuple(body)

    def preds(self, state: tuple):
        """Every state one press BEFORE ``state``.

        Two inverses. A walk is undone by stepping the player back onto
        ``p - d``, which has to be on the board, not solid and not a crate (it
        was free when the player left it). A push is undone the same way with
        the crate dragged one cell towards the player: the crate now at ``p + d``
        was at ``p``, so the earlier board is ``crates - {p + d} + {p}``, and
        ``p - d`` has to be free of THAT board, not of this one.

        Presses the room REFUSED are deliberately not inverted: they leave the
        state alone, so no shortest path within one room contains one. (A press
        that does nothing HERE can still be the press that advances the other
        room on a two-room level -- which is exactly why `_Level._joint` does not
        use this method at all and sweeps backward over the forward edges its
        corridor pass recorded.)"""
        p = state[0]
        out = []
        body = state[1:]
        for di in range(4):
            q = self.edge[p][_OPPOSITE[di]]
            if q < 0 or self.solid[q]:
                continue
            if q not in body:
                out.append((q,) + body)                      # undo a walk
            n2 = self.edge[p][di]
            if n2 >= 0 and n2 in body:                       # undo a push
                crates = sorted(x for x in body if x != n2)
                crates.append(p)
                crates.sort()
                if q not in crates:
                    out.append((q,) + tuple(crates))
        return out

    # -- the win condition ----------------------------------------------------
    def won(self, state: tuple) -> bool:
        """``All Target on Crate``, restricted to this room.

        A target in this room can only ever be covered by a crate in this room
        (crates do not cross a wall), so the global win condition is the AND of
        the rooms' -- which is what `_Level.won` computes. Extra crates are
        allowed to sit anywhere: the condition names targets, not crates."""
        return self.targets <= set(state[1:])

    # -- the field ------------------------------------------------------------
    def field(self, state: tuple) -> dict:
        """``{state: presses to a win}`` for this room, exact on every state
        reachable from ``state``.

        Two sweeps. Forward, the whole reachable space (a won state is NOT
        terminal: on a two-room level this room may have to hold its answer
        while the other one finishes). Backward from every won state over
        `preds`, pruned to that reachable set.

        The prune is exact, not an approximation: a shortest path out of a
        reachable state is itself reachable, so the backward sweep sees all of
        it. Hence **a reachable state absent from the result is proved
        unwinnable** -- which is what the recovery contract needs, and what a
        heuristic search can never say.

        Cached, and reused for any later query inside the reachable set already
        enumerated -- which during generation is every query, since a level is
        always planned from its start and RESET-recovery returns there."""
        if state in self._reachable:
            return self._dist
        seen = {state}
        queue = deque([state])
        won = []
        step = self.step
        while queue:
            cur = queue.popleft()
            if self.won(cur):
                won.append(cur)
            for di in range(4):
                nxt = step(cur, di)
                if nxt == cur or nxt in seen:
                    continue
                seen.add(nxt)
                queue.append(nxt)
        dist = {}
        queue = deque()
        for s in won:
            dist[s] = 0
            queue.append(s)
        while queue:
            cur = queue.popleft()
            d = dist[cur] + 1
            for prev in self.preds(cur):
                if prev in dist or prev not in seen:
                    continue
                dist[prev] = d
                queue.append(prev)
        # Merge rather than replace: a re-plan from outside the cached set (only
        # `--selfcheck` does this) enumerates its own reachable set, and both
        # halves are exact, so agreement on the overlap is guaranteed.
        self._reachable |= seen
        self._dist.update(dist)
        return self._dist

    def level_sets(self) -> list:
        """The BACKWARD LEVEL SETS ``B(0), B(1), ...`` over this room's whole
        reachable space, up to the fixpoint.

        ``B(0)`` is the won states and ``B(n+1) = {s : some press lands in
        B(n)}``, so ``s in B(n)`` iff this room can be won from ``s`` in EXACTLY
        ``n`` presses. Two things come out of it, both used by ``--levelsets``:

        * whether the sequence is UP-CLOSED. If it is, the lengths at which a
          room can be won from ``s`` are exactly ``{n : n >= d(s)}`` -- no holes,
          no parity -- and then `_Level.heuristic`'s ``max`` is the strongest
          bound per-room information can give, because the obvious sharpening
          ("the smallest length BOTH rooms admit") is the same number.
        * ``d(s) = min {n : s in B(n)}``, which re-derives `field`'s answer
          FORWARD, sharing none of its code.

        Requires `field` to have enumerated the reachable set first."""
        seen = self._reachable
        rev: dict = {}
        for s in seen:
            for di in range(4):
                # No-op presses are deliberately KEPT here, unlike in `preds`.
                # A press the room refuses is a self-loop, and a self-loop on a
                # won state is precisely how a room wastes a press -- i.e. it is
                # the mechanism that makes the sequence up-closed, so leaving it
                # out would measure a different question.
                rev.setdefault(self.step(s, di), set()).add(s)
        cur = frozenset(s for s in seen if self.won(s))
        out = [cur]
        while True:
            nxt = set()
            for s in cur:
                nxt |= rev.get(s, ())
            nxt = frozenset(nxt)
            if nxt == cur:
                return out
            out.append(nxt)
            cur = nxt


# ---------------------------------------------------------------------------
# The native model -- a whole level
# ---------------------------------------------------------------------------

class _Level:
    """A level as the set of independent `_Room`s one keyboard drives.

    A level splits into connected components of everything that is not a Wall,
    and nothing crosses a boundary: the player cannot walk out of its room and a
    crate is only ever pushed to an adjacent floor cell. All four levels of
    TwinPush split into exactly two rooms, each with its own player -- that IS
    the game -- and both take every press.

    A joint STATE is the tuple of the ACTIVE rooms' states, in room order.
    Playerless rooms carry no state at all -- their crates can never move -- so
    they are checked once, at construction, and make the level unwinnable if
    their targets are not already covered.
    """

    #: Generated-node budget for `_astar` when it is asked about a board that
    #: cannot be won. A start state is always solved long before this (the
    #: worst level generates 302814 nodes to find its answer), but a perturbed
    #: board with both rooms still alive and no joint answer can only be refused
    #: by CLOSING the product, which is not affordable. Hitting the cap returns
    #: None, i.e. "reset", which is the safe answer either way -- and it is
    #: reported separately from a proved-dead room everywhere it can happen, so
    #: nothing here claims a proof it does not have. Measured: never hit over
    #: `--selfcheck`'s 640 perturbed boards.
    #:
    #: Sized as a MEMORY budget, not a patience one (the ps: family's lesson): a
    #: joint state is a tuple of two ``(player, crate)`` tuples and every
    #: generated one is held in ``best`` and on the heap, so three million of
    #: them is under two gigabytes and six would not be.
    joint_node_cap: int = 3_000_000

    __slots__ = ("h", "w", "rooms", "active", "static_ok")

    def __init__(self, h, w, walls, crates, targets, players):
        self.h, self.w = h, w
        solid = bytearray(h * w)
        for i in walls:
            solid[i] = 1

        comp: dict = {}
        groups: list = []
        for i in range(h * w):
            if solid[i] or i in comp:
                continue
            gid = len(groups)
            groups.append([i])
            comp[i] = gid
            queue = deque([i])
            while queue:
                x = queue.popleft()
                r, c = divmod(x, w)
                for dr, dc in _DELTA.values():
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    y = nr * w + nc
                    if solid[y] or y in comp:
                        continue
                    comp[y] = gid
                    groups[gid].append(y)
                    queue.append(y)

        self.rooms = []
        for members in groups:
            cells = set(members)
            here = [p for p in players if p in cells]
            if len(here) > 1:
                raise ValueError(
                    f"TwinPush: {len(here)} players in one component -- the "
                    f"model keys a room on a single player cell")
            self.rooms.append(_Room(
                h, w, members, solid,
                here[0] if here else None,
                [i for i in crates if i in cells],
                [i for i in targets if i in cells]))
        #: The rooms a press can change. The rest are frozen scenery.
        self.active = [rm for rm in self.rooms if rm.player is not None]
        #: A playerless room's crates never move, so its half of the win
        #: condition is decided at reset and can never change. False makes the
        #: whole level dead. (The leading 0 is a dummy player cell: `_Room.won`
        #: reads only the crate half of a state, and such a room has no player
        #: to put there.)
        self.static_ok = all(
            rm.won((0,) + rm.crates)
            for rm in self.rooms if rm.player is None)

    # -- states ---------------------------------------------------------------
    def step(self, jstate: tuple, di: int) -> tuple:
        return tuple(rm.step(s, di) for rm, s in zip(self.active, jstate))

    def won(self, jstate: tuple) -> bool:
        return self.static_ok and all(
            rm.won(s) for rm, s in zip(self.active, jstate))

    def decode(self, jstate: tuple):
        """``(players, crates)`` as sorted cell tuples, for comparing the
        model's board against the interpreter's grid."""
        players, crates = [], []
        for s in jstate:
            players.append(s[0])
            crates.extend(s[1:])
        for rm in self.rooms:
            if rm.player is None:
                crates.extend(rm.crates)
        return tuple(sorted(players)), tuple(sorted(crates))

    # -- the search -----------------------------------------------------------
    def solve(self, jstate: tuple) -> "Plan | None":
        """The shortest press sequence from ``jstate``, with the EXACT optimal
        set at every step, or None if this board cannot be won from here.

        One room: descend its field, which already holds every answer. Several:
        `_joint` -- see the module docstring for why the max of the per-room
        fields is the heuristic that makes it finish."""
        if not self.static_ok:
            return None
        if not self.active:
            return Plan([], [])
        fields = [rm.field(s) for rm, s in zip(self.active, jstate)]
        if any(f.get(s) is None for f, s in zip(fields, jstate)):
            return None                        # PROVED dead, not "gave up"
        if len(self.active) == 1:
            room, dist = self.active[0], fields[0]
            return _descend(room.step, dist, jstate[0])
        return self._joint(jstate, fields)

    def heuristic(self, jstate: tuple, fields) -> "int | None":
        """``max`` over rooms of the room's own distance to a win.

        Admissible: a joint plan of length ``n`` hands room ``i`` a sequence of
        ``n`` presses whose last state is won, so ``n >= d_i`` for every ``i``.
        None means some room can no longer win at all, which sinks the state.

        ``--levelsets`` is the measurement that this is the best a per-room
        argument can do: each room's achievable win-lengths from a state form
        the up-set of its distance, so no combination of the rooms' own fields
        rules out any ``n >= max``."""
        best = 0
        for f, s in zip(fields, jstate):
            d = f.get(s)
            if d is None:
                return None
            if d > best:
                best = d
        return best

    def _joint(self, jstate: tuple, fields) -> "Plan | None":
        """A* to ``d*`` under `heuristic`, then a bounded corridor sweep for the
        exact optimal sets. See the module docstring for the correctness
        argument; the short form is that every press on a shortest joint path
        satisfies ``g + h <= d*``, so the corridor contains all of them, and a
        corridor value can only be too high, never too low."""
        dstar = self._astar(jstate, fields)
        if dstar is None:
            return None
        if dstar == 0:
            return Plan([], [])

        # The corridor: every state a shortest joint path can pass through,
        # with the forward edges recorded so the backward sweep needs no
        # product-of-preds enumeration.
        gmap = {jstate: 0}
        succ: dict = {}
        queue = deque([jstate])
        while queue:
            cur = queue.popleft()
            g = gmap[cur]
            edges = {}
            if not self.won(cur):
                for di in range(4):
                    nxt = self.step(cur, di)
                    if nxt == cur:
                        continue
                    h = self.heuristic(nxt, fields)
                    if h is None or g + 1 + h > dstar:
                        continue
                    edges[di] = nxt
                    if nxt not in gmap:
                        gmap[nxt] = g + 1
                        queue.append(nxt)
            succ[cur] = edges

        rev: dict = {}
        for cur, edges in succ.items():
            for nxt in edges.values():
                rev.setdefault(nxt, []).append(cur)
        dist = {}
        queue = deque()
        for cur in succ:
            if self.won(cur):
                dist[cur] = 0
                queue.append(cur)
        while queue:
            cur = queue.popleft()
            d = dist[cur] + 1
            for prev in rev.get(cur, ()):
                if prev not in dist:
                    dist[prev] = d
                    queue.append(prev)
        return _descend(self.step, dist, jstate)

    def _astar(self, jstate: tuple, fields) -> "int | None":
        """``d*``, the length of a shortest press sequence winning every room at
        once. Exact when it returns: `heuristic` is admissible and the weight is
        1, so the first winning state popped is at its true distance. None means
        either the product closed with no win in it (a proof) or
        `joint_node_cap` was reached (not one) -- `unproved` says which, and
        every caller that can meet the second case reports it separately."""
        h0 = self.heuristic(jstate, fields)
        if h0 is None:
            return None
        if self.won(jstate):
            return 0
        counter = 0
        pq = [(h0, 0, counter, jstate)]
        best = {jstate: 0}
        while pq:
            _f, g, _c, cur = heapq.heappop(pq)
            if best.get(cur, 1 << 30) < g:
                continue
            if self.won(cur):
                return g
            for di in range(4):
                nxt = self.step(cur, di)
                if nxt == cur:
                    continue
                h = self.heuristic(nxt, fields)
                if h is None:
                    continue
                ng = g + 1
                if best.get(nxt, 1 << 30) <= ng:
                    continue
                best[nxt] = ng
                counter += 1
                if counter >= self.joint_node_cap:
                    return None
                heapq.heappush(pq, (ng + h, ng, counter, nxt))
        return None

    def unproved(self, jstate: tuple) -> bool:
        """True when `solve` refused a board with every room still individually
        winnable -- i.e. the refusal rests on the joint search rather than on a
        dead room, and so is a proof only if `_astar` closed instead of hitting
        `joint_node_cap`. Used by the reports to keep the two apart."""
        if not self.static_ok or not self.active:
            return False
        fields = [rm.field(s) for rm, s in zip(self.active, jstate)]
        return all(f.get(s) is not None for f, s in zip(fields, jstate))


def _descend(step, dist: dict, state) -> "Plan | None":
    """Walk a distance field down to zero, recording every press that ties.

    Ties come out in ``DIRS`` order, which is what makes a re-derived plan
    byte-identical across processes."""
    d = dist.get(state)
    if d is None:
        return None
    presses, optsets = [], []
    cur = state
    while d:
        best = []
        for di in range(4):
            nxt = step(cur, di)
            if nxt != cur and dist.get(nxt, -1) == d - 1:
                best.append((di, nxt))
        if not best:                 # unreachable: dist[cur] > 0 has a step
            return None
        presses.append(DIRS[best[0][0]])
        optsets.append([DIRS[di] for di, _ in best])
        cur = best[0][1]
        d -= 1
    return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class TwinPushExpert(PSExpert):
    """`PSExpert`'s plan memo, disk cache and snapshot discipline around a
    `_Level`'s fields.

    The base class keeps the memo, the level scoping and the disk cache; only
    the strategy underneath changes, the way `PSEnumExpert` replaces it. Here
    the strategy is "split the board into rooms, enumerate each one's exact
    distance field, and descend the product of them", so `heuristic` is never
    called and asserts rather than returning a number nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state (a recovery prefix, an epsilon detour) is answered exactly
    -- including "this board is now dead", which an exploration prefix really
    does produce here: one crate per room and one target to put it on, so a
    crate shoved into a corner ends the level."""

    directions = list(DIRS)
    #: `_key` is the dynamic objects only (players + crates), which is canonical
    #: WITHIN a level but not across them -- walls and targets are static per
    #: level and differ between levels.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        g = self.g
        self.player_ids = set(self.game._engine._player_indices)
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.crate_ids = set(g.resolve_object_name("crate"))
        self.target_ids = set(g.resolve_object_name("target"))
        self.dyn_ids = self.player_ids | self.crate_ids
        #: `_Level`s by STATIC signature (see `read`), not by level index, so a
        #: level's rooms -- and the fields they cache -- are built once.
        self._levels: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "TwinPushExpert descends exact distance fields; heuristic is unused")

    def _key(self, eng) -> frozenset:
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn))

    # -- engine <-> model -----------------------------------------------------
    def read(self, eng) -> tuple:
        """``(level model, joint state)`` for the engine's current grid.

        Models are cached by their STATIC signature (dimensions, walls,
        targets), not by level index, so the room decomposition, the edge tables
        and -- crucially -- the enumerated fields are shared by every state of
        the level."""
        h, w = eng.height, eng.width
        walls, targets, crates, players = [], [], [], []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.target_ids:
                    targets.append(i)
                if cell & self.crate_ids:
                    crates.append(i)
                if cell & self.player_ids:
                    players.append(i)
        sig = (h, w, tuple(walls), tuple(targets))
        model = self._levels.get(sig)
        if model is None:
            model = self._levels[sig] = _Level(h, w, walls, crates,
                                               targets, players)
        jstate = []
        for rm in model.active:
            here = [p for p in players if p in rm.cells]
            if len(here) != 1:
                return model, ()       # a player left the board; see `_search`
            jstate.append((here[0],)
                          + tuple(sorted(i for i in crates if i in rm.cells)))
        return model, tuple(jstate)

    def _search(self, eng) -> "Plan | None":
        model, jstate = self.read(eng)
        if len(jstate) != len(model.active):
            return None                        # a player left the board
        if model.won(jstate):
            return Plan([], [])
        return model.solve(jstate)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class TwinPushSolver(PSAStarSolver):
    game_id = "puzzlescript_twinpush"
    game_name = GAME_NAME
    expert_cls = TwinPushExpert

    #: `games/ps:twinpush/ps:twinpush.py` is a plain passthrough (it constructs
    #: the adapter and nothing else), so there is nothing to gain by routing
    #: through it -- but it IS what a live agent is handed, so if that wrapper
    #: ever grows a patch this must be set to ``"ps:twinpush"``.
    game_module_id = ""

    #: Unused: the expert enumerates exact fields rather than searching under a
    #: heuristic, so there is no weight to trade, and the one search that could
    #: run away -- the product A* -- carries its own `_Level.joint_node_cap`.
    weight = 1
    node_cap = 400_000

    #: The longest plan is 87 presses (level 3); the rest is room for a re-plan
    #: after the exploration prefix. Stays well under the adapter's own 200-step
    #: per-level budget, which `set_level` resets before the plan starts anyway.
    max_steps = 160

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it fills the disk cache in one pass and makes the
        cold-start cost visible as startup rather than as a mysteriously slow
        first seed."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new():
    solver = TwinPushSolver()
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _plans() -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the fields, the product search and the
    tie labelling at once."""
    _solver, game, expert, solvable = _new()
    total = ties = triples = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        model, jstate = expert.read(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level:2d}: UNSOLVED")
            bad += 1
            continue
        alone = [rm.field(s).get(s)
                 for rm, s in zip(model.active, jstate)]
        for direction in plan:
            eng.step(direction)
        won = eng.check_win()
        bad += not won
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        tie_steps = sum(1 for s in sets if len(s) > 1)
        total += len(plan)
        ties += tie_steps
        triples += sum(1 for s in sets if len(s) > 2)
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"  L{level:2d}: {eng.height:2d}x{eng.width:2d}  "
              f"{len(model.active)} room(s)  per-room optima {alone}  "
              f"joint {len(plan):3d} presses  win={won}  "
              f"(budget {game._max_steps}, {room})  "
              f"{tie_steps:3d} steps with a tie set")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second equally-right "
          f"answer ({triples} with a third)")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _levelsets(verbose: bool = True) -> int:
    """The measurement behind `_Level.heuristic`, and a second derivation of
    every room's field.

    For each room, `_Room.level_sets` builds ``B(0) = won``,
    ``B(n+1) = {s : some press lands in B(n)}``, so ``s in B(n)`` iff the room
    can be won from ``s`` in EXACTLY ``n`` presses. Two things are then checked:

    * **UP-CLOSEDNESS** (``B(n) <= B(n+1)`` for every n). If it holds, a room
      that can win in n presses can also win in n+1 -- it can always waste one --
      so the achievable lengths from a state are exactly ``{n : n >= d(s)}``.
      That is what makes ``max`` the tightest per-room bound there is: the
      sharper-looking heuristic "the smallest length BOTH rooms admit" is
      ``max(d_left, d_right)`` exactly, and there is no parity or timing
      obstruction for the joint search to exploit. If it ever failed, this
      report is where the stronger heuristic would have to be built.
    * **the field, forward.** ``d(s) = min {n : s in B(n)}`` is compared against
      `_Room.field`, which derived the same numbers by a backward BFS over
      `_Room.preds`. The two share no code -- one inverts the mechanic, the
      other only ever steps it forwards -- so agreement on every state of every
      room is an independent derivation of every distance this file uses,
      including the ones the product A* is steered by.

    Returns the number of violations."""
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        model, jstate = expert.read(eng)
        for ri, (rm, s) in enumerate(zip(model.active, jstate)):
            dist = rm.field(s)
            sets = rm.level_sets()
            closed = all(sets[i] <= sets[i + 1] for i in range(len(sets) - 1))
            forward: dict = {}
            for n, B in enumerate(sets):
                for x in B:
                    forward.setdefault(x, n)
            agree = forward == {k: v for k, v in dist.items()
                                if k in rm._reachable}
            bad += (not closed) + (not agree)
            if verbose:
                print(f"  L{level:2d} room {ri}: {len(rm._reachable):6d} states, "
                      f"{len(dist):6d} winnable, {len(sets):3d} level sets, "
                      f"start d={dist.get(s)} -- "
                      f"{'UP-CLOSED' if closed else 'NOT up-closed'}, field "
                      f"{'AGREES' if agree else 'DISAGREES'} with the backward "
                      f"sweep")
    return bad


def _selfcheck(walk_presses: int = 3000, samples: int = 80,
               tie_steps: int = 6, verbose: bool = True) -> int:
    """Five things the expert would otherwise be trusted on.

    1. **The native model is the interpreter's, board AND win flag.** A seeded
       random walk on every level, comparing `_Level`'s decoded board against
       the engine's grid AND `_Level.won` against ``check_win`` after EVERY
       press -- including the presses that do nothing, which is where a
       collision-layer mistake hides, and including ACTION5, which must be a
       no-op (the model does not model it at all). The walk restarts the level
       whenever it stumbles into a win, so it keeps sampling fresh boards.
    2. **The per-room bounds are real.** A plain forward BFS to the first win in
       each room, which is shortest by construction and shares no code with the
       fields' backward half. The joint plan must be at least the max of them
       (and the A* under that admissible max at weight 1 is the other half of
       the proof that it is exactly d*).
    3. **The joint answer, without a heuristic, where that is affordable.** L0's
       product is 118206 states, so a plain joint forward BFS closes it and
       returns d* directly. That number owes nothing to `_Level.heuristic` --
       no admissibility argument, no field, no A*. The bigger levels' products
       pass 6M states without closing and are skipped here; ``--engine`` proves
       L0 a third time, through the real interpreter.
    4. **The tie sets are exact.** ``tie_steps`` steps of every level's plan
       have their optimal SETS re-derived from scratch: for each of the four
       presses, a fresh joint A* from the successor, kept iff it finishes in
       exactly the moves remaining. That A* shares nothing with the corridor
       sweep the sets were read off.
    5. **Recovery is answered from ANY board, not just the start.** This is the
       one that matters at generation time: the exploration prefix really does
       strand this game (one crate, one target, and a crate in a corner is
       over), so `_search` has to be right about arbitrary states in both
       directions. From ``samples`` states reached by seeded random presses, a
       returned plan must WIN when replayed through the interpreter, and a
       refusal must name a room an independent forward BFS also proves dead. A
       refusal with every room still alive is the joint search closing (or
       hitting `_Level.joint_node_cap`); those are counted and reported rather
       than re-checked, because re-deriving one means enumerating the product.

       Two walk lengths, because they measure different halves. **short**
       (1-29 presses) is what the recorded exploration prefix actually does, and
       it mostly lands on boards that are still winnable -- which is the half
       that has to return a plan that WINS. **long** (30-199) is what it takes
       to reliably kill a crate in a game with one crate per room, and it is
       what gives the "a refusal is a PROOF" claim its coverage: measured, the
       short regime finds 14 dead boards in 320 (all of them on L0) and the long
       one finds 110.
    """
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    idx = game._game.obj_name_to_idx
    bad = 0

    def live(eng):
        players, crates = [], []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                i = r * eng.width + c
                if cell & expert.player_ids:
                    players.append(i)
                if idx["crate"] in cell:
                    crates.append(i)
        return tuple(sorted(players)), tuple(sorted(crates))

    # 1 -- the model is the interpreter's.
    for level in range(game.n_levels):
        game.set_level(level)
        model, jstate = expert.read(eng)
        rng = random.Random(f"twinpush:selfcheck:{level}")
        drift = wrong_win = noop = 0
        for _ in range(walk_presses):
            if rng.randrange(10) == 0:            # the unbound ACTION5 key
                before = [[set(cell) for cell in row] for row in eng.grid]
                eng.step("action")
                noop += eng.grid != before
                continue
            di = rng.randrange(4)
            eng.step(DIRS[di])
            jstate = model.step(jstate, di)
            if model.decode(jstate) != live(eng):
                drift += 1
                break
            if eng.check_win() != model.won(jstate):
                wrong_win += 1
                break
            if eng.check_win():
                game.set_level(level)
                model, jstate = expert.read(eng)
        bad += drift + wrong_win + noop
        if verbose:
            print(f"  L{level:2d}: model {'MATCHES' if not drift else 'DIVERGES from'}"
                  f" the interpreter over {walk_presses} random presses; win flag "
                  f"{'agrees' if not wrong_win else 'DISAGREES'}; ACTION5 "
                  f"{'inert' if not noop else 'MOVED SOMETHING'}")

    # 2 -- the per-room lower bounds, independently.
    for level in range(game.n_levels):
        game.set_level(level)
        model, jstate = expert.read(eng)
        plan = expert.plan(eng, level)
        bounds = [_first_win(rm, s) for rm, s in zip(model.active, jstate)]
        ok = plan is not None and len(plan) >= max(bounds)
        bad += not ok
        if verbose:
            print(f"  L{level:2d}: joint {len(plan) if plan else None} presses "
                  f"vs per-room lower bounds {bounds} (independent forward BFS) "
                  f"-- {'CONSISTENT' if ok else 'IMPOSSIBLE'}")

    # 3 -- the joint answer with no heuristic at all, where it fits.
    for level in range(game.n_levels):
        game.set_level(level)
        model, jstate = expert.read(eng)
        plan = expert.plan(eng, level)
        got = _joint_bfs(model, jstate)
        if got is None:
            if verbose:
                print(f"  L{level:2d}: heuristic-free joint BFS SKIPPED "
                      f"(product past {_JOINT_BFS_CAP} states)")
            continue
        ok = plan is not None and got == len(plan)
        bad += not ok
        if verbose:
            print(f"  L{level:2d}: A* {len(plan)} presses vs a heuristic-free "
                  f"joint BFS {got} -- {'SHORTEST' if ok else 'NOT SHORTEST'}")

    # 4 -- the tie sets, re-derived by a fresh joint A*.
    for level in range(game.n_levels):
        game.set_level(level)
        model, jstate = expert.read(eng)
        plan = expert.plan(eng, level)
        fields = [rm.field(s) for rm, s in zip(model.active, jstate)]
        sets = getattr(plan, "optsets", None) or []
        stride = max(1, len(plan) // tie_steps)
        checked = mismatched = 0
        cur = jstate
        for i, direction in enumerate(plan):
            if i % stride == 0 and checked < tie_steps:
                rest = len(plan) - i
                truth = []
                for di in range(4):
                    nxt = model.step(cur, di)
                    if nxt == cur:
                        continue
                    got = model._astar(nxt, fields)
                    if got is not None and got == rest - 1:
                        truth.append(DIRS[di])
                mismatched += sorted(truth) != sorted(sets[i])
                checked += 1
            cur = model.step(cur, DIRS.index(direction))
        bad += mismatched
        if verbose:
            print(f"  L{level:2d}: {checked} of {len(plan)} tie set(s) "
                  f"re-derived by a fresh joint A* -- "
                  f"{'match' if not mismatched else f'{mismatched} MISMATCH'}")

    # 5 -- recovery from arbitrary boards, checked against the interpreter.
    for level in range(game.n_levels):
        for tag, lo, hi in (("short", 1, 30), ("long", 30, 200)):
            game.set_level(level)
            rng = random.Random(f"twinpush:recovery:{tag}:{level}")
            wrong = dead = exhausted = won_after = 0
            for _ in range(samples):
                game.set_level(level)
                for _ in range(rng.randrange(lo, hi)):
                    eng.step(DIRS[rng.randrange(4)])
                    if eng.check_win():
                        break
                if eng.check_win():
                    continue             # the walk already won; nothing to plan
                model, js = expert.read(eng)
                plan = expert._search(eng)
                if plan is None:
                    if model.unproved(js):
                        exhausted += 1
                        continue
                    fields = [rm.field(s) for rm, s in zip(model.active, js)]
                    culprits = [(rm, s)
                                for rm, f, s in zip(model.active, fields, js)
                                if f.get(s) is None]
                    dead += 1
                    wrong += not any(_first_win(rm, s) is None
                                     for rm, s in culprits)
                    continue
                for direction in plan:
                    eng.step(direction)
                if eng.check_win():
                    won_after += 1
                else:
                    wrong += 1
            bad += wrong
            if verbose:
                print(f"  L{level:2d} {tag:5s}: {samples} random boards -- "
                      f"{won_after} re-planned to a WIN in the interpreter, "
                      f"{dead} refused with a dead room (confirmed by an "
                      f"independent BFS), {exhausted} refused by the joint "
                      f"search, {wrong} WRONG")
    return bad


def _first_win(room: _Room, state: tuple) -> "int | None":
    """Presses to a win in ONE room, by a plain forward BFS that stops at the
    first one. Shortest by construction, and it shares no code with
    `_Room.field`'s backward half -- which is the point of running it."""
    if room.won(state):
        return 0
    seen = {state}
    queue = deque([(state, 0)])
    while queue:
        cur, d = queue.popleft()
        for di in range(4):
            nxt = room.step(cur, di)
            if nxt == cur or nxt in seen:
                continue
            if room.won(nxt):
                return d + 1
            seen.add(nxt)
            queue.append((nxt, d + 1))
    return None


#: How much of a level's PRODUCT ``--selfcheck``'s heuristic-free joint BFS will
#: enumerate before giving up. L0 closes at 118206; L1-L3 pass six million.
_JOINT_BFS_CAP = 400_000


def _joint_bfs(model: _Level, jstate: tuple) -> "int | None":
    """``d*`` by a plain forward BFS over the PRODUCT, with no heuristic, no
    per-room field and no A* -- or None if the product is bigger than
    `_JOINT_BFS_CAP`.

    This is the derivation that owes nothing to `_Level.heuristic`'s
    admissibility argument. It only fits on L0."""
    if model.won(jstate):
        return 0
    seen = {jstate}
    queue = deque([(jstate, 0)])
    while queue:
        cur, d = queue.popleft()
        for di in range(4):
            nxt = model.step(cur, di)
            if nxt == cur or nxt in seen:
                continue
            if model.won(nxt):
                return d + 1
            if len(seen) >= _JOINT_BFS_CAP:
                return None
            seen.add(nxt)
            queue.append((nxt, d + 1))
    return None


def _engine(levels=(0,), verbose: bool = True) -> int:
    """The proof that owes nothing to the native model: enumerate the REAL
    interpreter.

    `StateGraph.build` walks every state reachable from the level start by
    pressing actual buttons on the engine and keying the resulting grid, then
    solves the graph exactly. It shares no line of code with `_Room` -- no
    modelled push, no modelled collision layer, no modelled win test, and no
    modelled room decomposition -- so agreement on both the plan LENGTH and the
    per-step optimal SET is an independent derivation of everything this file
    claims, product search included.

    L0 is the default and the only level in reach: its product is 118206 model
    states -- the enumeration keys 118133 of them, stopping at a winning edge
    rather than keying the won board -- and 472532 interpreter presses, where
    L1-L3 pass six million states. States are re-seated by DECODING their key
    rather than from a snapshot -- 118k
    snapshots of a 15x26 grid is gigabytes -- which is sound because the key
    holds every object that can move (Player and Crate) and the decoder rebuilds
    the board from a static template of everything that cannot (Background,
    Target, Wall). The report checks that claim rather than asserting it: before
    enumerating, it walks a seeded 300-press sequence and requires the decoded
    board to equal the stepped one after every press."""
    from solvers.common.ps_astar import StateGraph, snapshot   # noqa: PLC0415

    _solver, game, expert, _solvable = _new()
    eng = game._engine
    bad = 0
    for level in levels:
        game.set_level(level)
        plan = expert.plan(eng, level)
        game.set_level(level)
        w = eng.width
        pid = next(iter(expert.player_ids))
        cid = next(iter(expert.crate_ids))
        dyn = expert.dyn_ids
        static = [[frozenset(cell - dyn) for cell in row] for row in eng.grid]

        def key_fn(e, w=w):
            players, crates = [], []
            for r, row in enumerate(e.grid):
                for c, cell in enumerate(row):
                    if cell & expert.player_ids:
                        players.append(r * w + c)
                    if cell & expert.crate_ids:
                        crates.append(r * w + c)
            return (tuple(sorted(players)), tuple(sorted(crates)))

        def decode(e, k, static=static, w=w, pid=pid, cid=cid):
            e.grid = [[set(cell) for cell in row] for row in static]
            for i in k[0]:
                e.grid[i // w][i % w].add(pid)
            for i in k[1]:
                e.grid[i // w][i % w].add(cid)
            e._position_index_dirty = True
            e._rule_noop_cache.clear()

        # The decoder is the one thing here that could silently enumerate a
        # DIFFERENT game, so measure it before trusting it.
        rng = random.Random(f"twinpush:decode:{level}")
        drift = 0
        for _ in range(300):
            eng.step(DIRS[rng.randrange(4)])
            want = snapshot(eng)
            decode(eng, key_fn(eng))
            drift += snapshot(eng) != want
            if eng.check_win():
                game.set_level(level)
        bad += drift
        if verbose:
            print(f"  L{level:2d}: decoder {'is exact' if not drift else 'DRIFTS'}"
                  f" over 300 stepped boards")

        game.set_level(level)
        graph = StateGraph.build(eng, key_fn, list(DIRS),
                                 node_cap=1_000_000, decode=decode)
        eplan = graph.plan(list(DIRS)) if graph is not None else None
        if eplan is None:
            print(f"  L{level:2d}: the interpreter enumeration found no win")
            bad += 1
            continue
        same_len = len(eplan) == len(plan)
        esets = [sorted(s) for s in eplan.optsets]
        fsets = [sorted(s) for s in plan.optsets]
        same_sets = esets == fsets
        bad += (not same_len) + (not same_sets)
        if verbose:
            print(f"  L{level:2d}: {len(graph.succ)} engine states, "
                  f"{graph.steps} interpreter presses, {len(eplan)} presses vs "
                  f"the field's {len(plan)} -- "
                  f"{'AGREE' if same_len else 'DISAGREE'}; tie sets "
                  f"{'AGREE' if same_sets else 'DISAGREE'}")
    return bad


def _audit(verbose: bool = True) -> int:
    """Two passes over every reachable cell COMPOSITION.

    **Pass 1 -- distinctness**, at every cell size the levels use (2 px for all
    four 15x26 boards; the size list is derived rather than assumed so an edited
    level is covered). Whole 64x64 frames of uniform boards are compared rather
    than one cell out of a mixed board: `_render_frame` upscales and centre-pads,
    so slicing a cell by ``cell_px`` arithmetic reads the wrong pixels (the
    ps:explod lesson). Two uniform boards render identically iff their cells do.

    This is the pass the sprite fix in ``data/puzzlescript_games/TwinPush.txt``
    exists for: at cell_px 2 a 5x5 sprite is sampled at four pixels, and as
    shipped the Crate had none of them (it was a hollow box) and the Player had
    all four (it hid the Target it stood on), so ``crate == floor``,
    ``crate+target == target`` and ``player+target == player``. Three of the
    seven compositions were invisible, and one of them was the win condition.

    ``player + crate`` is absent on purpose and is not an omission -- they share
    a collision layer, so the engine can never put them in one cell. Nor can a
    Wall share a cell with a Target, a Crate or a Player.

    **Pass 2 -- the group.** The game is in `_FLIP_GAMES`, so a board is drawn at
    any of the eight turns and mirrors. That is fine as long as no transform of
    one composition is another composition's art, which is what this checks: all
    eight transforms of each, against all others. It runs on SQUARE boards,
    because a rotation of a non-square board also moves the letterbox and every
    comparison would pass for the wrong reason."""
    from adapters.puzzlescript_adapter import _render_frame            # noqa: PLC0415

    _solver, game, _expert, _solvable = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = [(), ("wall",), ("target",),
             ("crate",), ("target", "crate"),
             ("player",), ("target", "player")]

    def name(comp):
        return "+".join(comp) or "floor"

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    def shoot(h, w, comp):
        eng.height, eng.width = h, w
        eng.grid = [[{idx[o] for o in comp} | {idx["background"]}
                     for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        return np.asarray(_render_frame(eng, g)).copy()

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {c: shoot(h, w, c) for c in comps}
        clashes = [(a, b) for a, b in itertools.combinations(comps, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        if verbose:
            print(f"  {h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): {len(shots)} "
                  f"compositions, {'OK' if not clashes else 'CLASH'}")
        for a, b in clashes:
            print(f"      IDENTICAL: {name(a)}  ==  {name(b)}")

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
        shots = {c: shoot(n, n, c) for c in comps}
        clashes = [(a, b, t) for a, b in itertools.permutations(comps, 2)
                   for t in group
                   if np.array_equal(transform(shots[a], *t), shots[b])]
        invariant = [name(c) for c in comps
                     if all(np.array_equal(transform(shots[c], *t), shots[c])
                            for t in group)]
        bad += len(clashes)
        if verbose:
            print(f"  cell {cell}px ({n}x{n}): {len(clashes)} composition(s) "
                  f"whose transform is another's art; group-invariant: "
                  f"{', '.join(invariant)}")
        for a, b, t in clashes:
            print(f"      {name(a)} under {t} == {name(b)}")
    return bad


def _symmetry(walk_presses: int = 200, verbose: bool = True) -> int:
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unaugmented ones.

    This is the evidence for putting the game in `_FLIP_GAMES` (the rotation is
    mandatory and is checked here too). The structural argument is in the module
    docstring -- one push rule written with the relative ``>``, two players who
    cannot reach each other, and a win condition that names no direction -- but
    Gobble Rush's chirality hid inside exactly this kind of argument, so it is
    measured.

    Both the PLANS and a seeded random walk are replayed: the walk reaches the
    boards a plan never visits (a crate flat against a wall, a crate dead in a
    corner) and it presses the unbound ACTION key as well."""
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
    for seed in range(400):
        if len(seen) == 16 and seed > 40:
            break
        g = solver.make_game(seed)
        for level in range(g.n_levels):
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            rng = random.Random(f"twinpush:symmetry:{level}")
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
    if "--levelsets" in sys.argv:
        violations = _levelsets()
        print(f"level sets: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--selfcheck" in sys.argv:
        violations = _selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--engine" in sys.argv:
        want = [int(a) for a in sys.argv[sys.argv.index("--engine") + 1:]
                if a.isdigit()]
        violations = _engine(tuple(want) if want else (0,))
        print(f"engine enumeration: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = _audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = _symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    sys.exit(TwinPushSolver.main())
