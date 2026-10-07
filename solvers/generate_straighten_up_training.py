"""Generate Phase-1 training data for the PuzzleScript game ps:straighten_up
("Straighten Up", Ricky).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, a native model of them, the exact distance FIELDS that model is
searched with, the proof that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_straighten_up",
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

Result: all 6 levels, 246 presses, every plan PROVABLY SHORTEST and every step
labelled with an exact optimal set.

The game: a sokoban with no targets
-----------------------------------
The push is the orthodox one and the only rule with a player on its left:

    [ >  GreenPlayer | MoveCrate ] -> [  >  GreenPlayer | > MoveCrate ]

(there is a mirror-image `[< RedPlayer | MoveCrate]` PULL rule, but no level
places a RedPlayer, so it never fires -- ``--selfcheck`` is what says so rather
than the .txt). Crates, walls and players share one collision layer, so walking
into a crate is the push, a crate whose far side is blocked does not move and
neither does the player, and two crates in a row are a wall. **FixedCrate has no
push rule at all**: it is a crate for the win condition and a wall for
everything else.

What makes it not a sokoban is the win condition, which names no squares:

    all Crate on Path

``Path`` is not level scenery -- it is recomputed from scratch after every press
by six ``late`` rules that mark a crate with PathHoriz/PathVert when it sits in
a run of three, and strip the mark off runs that have fallen to two or one.
Reduced, and this reduction is the whole game:

    WIN  <=>  every crate -- MoveCrate and FixedCrate alike -- is part of a
              horizontal or a vertical run of THREE OR MORE crates.

So there is no goal square to push a crate onto; the crates have to be
straightened into lines, and which lines is part of the puzzle. That is why the
usual sokoban machinery does not fit: `PSSokobanExpert`'s per-target push
tables need targets, and the standard deadlock test ("a crate in a corner can
never reach a target") is meaningless when a corner may well BE where a line
wants to be.

Three rules exist purely to draw the answer and are worth naming because they
are why a state key over the raw grid is still deterministic:

  * ``late [Path] -> [Path BlueCrate]`` repaints every crate on a line blue --
    the win condition rendered, live, on every frame.
  * ``late random[no Crate] -> [Flag] ...`` plants an INVISIBLE flag on a random
    empty cell, walks it onto any crate that is not on a line, and deletes it
    again before the tick ends. It is the game's "are we done" test; the
    randomness is real but every flag is removed by the last of the six rules,
    so no flag ever survives into a state and the engine is deterministic
    (measured: the same press sequence twice gives the same grid).
  * ``late right [no Arm | Player | no Arm] [Flag no Crate] -> [RightArm |
    Player | LeftArm] [Flag]`` raises the player's arms when the flag found no
    loose crate -- the victory pose, on the winning frame only.

Six levels, five of them 9x9 and the sixth 9x17:

  * L0 "Tic Tac Toe": six crates on a plus-shaped floor; the answer is two
    horizontal triples, one in each stub of the plus. 23 presses.
  * L1 "No Child Left Behind": five crates in an open room; the answer is a
    horizontal triple sharing its end crate with a vertical one. 31.
  * L2 "Connect Four": four crates in a lattice of pillars, and four is one
    more than a line needs, so the answer is a run of four. 43.
  * L3 "You Can't Push Me Around": the first FixedCrate. Three crates start
    already lined up -- blue at reset, because of
    ``run_rules_on_level_start`` -- and the puzzle is that the answer breaks
    that line up again: the final board is a horizontal run of four THROUGH the
    immovable crate, crossed by a vertical triple. 43.
  * L4 "Jack in the Box": four FixedCrates at the corners of a 3x3 square, four
    MoveCrates, and the player starting in the middle of the square. Each fixed
    crate has to have one of its two lines completed, which sounds like it
    forces the ring of eight around the player -- and it does not: 75 layouts
    win, and the cheapest is two VERTICAL runs down the square's sides, one of
    them three long and the other five, with the player left standing outside
    them. 40.
  * L5 "Stunt Double": 9x17, TWO GreenPlayers in two rooms separated by a solid
    wall column, both driven by the same key. 66 presses -- see below.

Expert solver
-------------
A NATIVE model of the mechanic above, searched to exact distance FIELDS rather
than heuristically. Nothing here is an estimate: every plan length, every
optimal set and every "this board is dead" is read off a field that was
enumerated.

**The decomposition.** A level's floor splits into connected components (walls
and fixed crates are the only static obstacles, and neither is ever created or
destroyed), and nothing crosses a component boundary: a player cannot walk out
of its room, a crate can only be pushed to an adjacent floor cell, and a run of
crates is a chain of adjacent crates, so it lies inside one room too. A level is
therefore a set of independent boards -- `_Room` -- that happen to share a
keyboard. Five levels have one room; L5 has two.

**Per room, an exact field.** `_Room.field` enumerates the room's whole
reachable state space ``(player, sorted crates)`` forward, then runs a backward
BFS from every WON state over inverted moves, pruned to that reachable set. The
prune loses nothing: a shortest path out of a reachable state is itself
reachable, so the value at every reachable state is its true distance. Which
gives the property the rest of this file leans on -- **a reachable state absent
from the field is PROVED unwinnable**, not "the search gave up". The spaces are
11k to 650k states for the one-room levels and 1865333 per room on L5 -- 5.1M
in all, and the whole cold derivation of all six levels takes 22 seconds in
800 MB. States are packed as ``bytes`` rather than tuples of tuples for exactly
that reason: at 1.9M states per room the difference is hundreds of megabytes.

**Where the two players make it hard.** On L5 both rooms take the same press,
so the answer is the shortest sequence that wins BOTH -- a path in the product,
and the product of two 1.87M-state rooms is not enumerable. It does not have to
be. Each room's own field is an exact lower bound on the joint answer (any joint
plan of length n hands room i a sequence of n presses ending in a won state, so
n >= d_i), which makes

    h(l, r) = max(d_left(l), d_right(r))

admissible, and A* under it finds d* = 66 in 212639 pops and under three
seconds, against per-room optima of 25 and 33 (the left room's field covers
622407 of its states, the right room's 172201; the rest can no longer be won at
all, and being able to say that is the same property the recovery contract
rests on). The joint optimal SETS then come from a second pass: a forward BFS
keeping only states with ``g + h <= d*`` -- 212642 of them, because that bound
is tight -- and a backward BFS from the won ones over the edges the forward
pass recorded. Every press on a shortest joint path is in that corridor by
construction (if ``dist(s') = d* - i - 1`` then ``g(s') + h(s') <= d*``), and
the corridor's values can only be too high, so comparing them against
``d* - i - 1`` is exact in both directions.

That is the reusable shape: **when one key drives several independent boards,
do not search the product blind -- solve each board exactly and let the MAX of
those distances be the heuristic.** It is admissible for free and it is the
only thing that made L5 finish.

Optimal-action sets
-------------------
Exact everywhere, with no inference: at a state ``d`` presses from a win the
optimal presses are exactly those whose successor is ``d - 1`` from one, and the
fields have both numbers already. ``--selfcheck`` re-derives every set on every
one-room level by brute force (a fresh depth-bounded BFS from each successor)
and requires an exact match; ``--engine`` re-derives L0's and L3's yet again
from the real interpreter, and ``--selfcheck``'s sixth pass re-derives a sample
of L5's with a fresh joint A*. No step ever ships unlabelled (the
always-emit-optimal-targets rule).

The measured answer is that this game has FEWER ties than an open-floored
sokoban looks like it should: 21 of its 246 steps have a second equally-right
press, and L4 has exactly one such step in 40. The reason is the same one that
makes the game hard -- with no goal squares, a shortest plan has to reach one
specific layout out of the dozens that win, and on the way to it almost every
walk is threaded past crates that must not be nudged. An open room does not buy
freedom when the room is full of things you are not allowed to touch.

Shortest, and how that is known
-------------------------------
All six plans are provably shortest, and the proof is independent derivations
agreeing:

  * this file's fields (forward BFS + backward BFS, plus the product A* on L5);
  * a plain forward BFS to the first win, which is shortest by construction and
    shares no code with the fields' backward half (``--selfcheck``);
  * for L5, the product A* is complete under an admissible heuristic at weight
    1, and ``--selfcheck`` re-derives the same 66 with a second, independent
    bound (a joint forward BFS run to depth 66 in the model, confirming no
    shorter joint win exists is not affordable -- what IS affordable and is
    checked is that neither room can be won in fewer than its own field says);
  * a full-space enumeration of the real INTERPRETER -- `StateGraph.build`
    walking every reachable engine state by pressing real buttons, with no
    native model involved at all. ``--engine`` runs L0 (10872 engine states) by
    default and ``--engine 0 3`` adds L3 (87325), which is the first level with
    a FixedCrate and so the one that exercises the immovable-crate half of the
    win condition. Both agree with the field on the plan length AND on every one
    of the 66 tie sets. The bigger levels are out of reach (L2 alone is 650k
    states, which is 2.6M interpreter steps); for those, the native model is
    fuzz-verified against the interpreter instead, which is what makes the other
    derivations statements about this game rather than about a model of it.

An independent derivation of the WIN CONDITION itself is in ``--goals``: the
winning crate layouts are enumerated combinatorially (every union of >=3-length
runs of crate-able cells whose size is the room's crate count and which covers
the fixed crates), with no reference to `_Room.won`, and ``--selfcheck`` checks
the two agree on every reachable state of every one-room level.

Recovery
--------
`supports_recovery` with RESET-mode recovery. The expert re-plans from the LIVE
board: `_search` reads whatever the engine currently holds, and because a room's
field covers its whole reachable space, an exploration prefix that leaves the
board anywhere is answered exactly -- including "this board is now dead", which
is what `record_level` needs to hear to fall back to a RESET. That is not a rare
corner. Shoving a crate flat against a wall where no line can contain it kills a
level for good, and ``--selfcheck``'s fourth pass walks 120 random boards per
level and finds many of them genuinely unwinnable; it checks both halves against
the interpreter -- every plan returned from a random board is replayed and must
WIN, and every None must be a board an independent forward BFS also proves dead.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`), which together re-sample the board's
8-element symmetry group. 6 levels x 16 presentations = 96.

The structural argument for the flips: the game is gravity-free, its push is
written with the relative ``>`` force and names no axis, input is
screen-relative, and one press moves at most two bodies which land on ``p + d``
and ``p + 2d`` -- never the same square -- so no two moves can contest a cell
and the order the interpreter expands a rule's directions cannot decide
anything. The win condition names no direction: the path rules come as a
``right``/``down`` PAIR marking PathHoriz and PathVert, the pair is a complete
orbit under a quarter turn (a rotation exchanges the two exactly as it exchanges
the axes), and ``all Crate on Path`` reads the two identically. The one piece of
art that is not axis-blind is the victory pose -- ``RightArm | Player |
LeftArm`` is drawn horizontally -- and it is a proper mirror orbit (each arm is
one pixel column hugging the player, and an hflip maps the pair onto itself), it
carries no facing, and it is on screen only on the frame the level ends.
``--symmetry`` measures the lot anyway: every level's plan AND a seeded 200-press
random walk (which does what a plan never does -- shoves crates into walls, into
each other and into dead corners, and presses the unbound ACTION key) replayed at
all 16 presentations, requiring every frame to be the exact transform of the
unaugmented one.

There is deliberately no colour augmentation: a crate's colour IS the win
condition here (orange = loose, blue = on a line), and a flattening recolor
would erase it.

Rendering
---------
ONE sprite had to change, and it is the one that says which crates the player is
allowed to push. See the header comment in
``data/puzzlescript_games/Straighten_Up.txt`` for the full statement; in short,
MoveCrate and FixedCrate ship as the same ORANGE/BROWN square (and brown and
orange are both ARC index 12), so the only thing separating them is FixedDot --
which was a single DARKBLUE pixel, and DARKBLUE is index 9, the same as the BLUE
interior of the BlueCrate a crate wears once it is on a line. A fixed crate on a
finished line was therefore PIXEL-IDENTICAL to a movable one, at both cell sizes
the game uses, on the three levels built around fixed crates. FixedDot is now a
PURPLE (15) DIAMOND: purple to survive both the orange and the blue underneath,
a diamond because it is exactly the five pixels that survive L5's 3 px cells and
because it leaves the sprite's top corners free for the victory arms drawn on
the layer beneath it.

``--audit`` renders every cell COMPOSITION the game can show -- floor, wall, a
movable crate loose and on a line, a fixed crate loose and on a line, the
player, and each of those under an arm -- as a whole 64x64 frame of a uniform
board, at every cell size the six levels render at (7 px and 3 px), and requires
them pairwise distinct. Whole frames rather than one cell sliced out of a mixed
board: `_render_frame` upscales and centre-pads, so slicing by ``cell_px``
arithmetic reads the wrong pixels (the ps:explod lesson). ``player + crate`` is
absent on purpose and is not an omission -- they share a collision layer, so the
engine can never put them in one cell.

Usage (run from the repo root):
    python solvers/generate_straighten_up_training.py --episodes 200 \
        --out data/training_multi_level/straighten_up

    python solvers/generate_straighten_up_training.py --plans      # level report
    python solvers/generate_straighten_up_training.py --goals      # the win set
    python solvers/generate_straighten_up_training.py --selfcheck  # model + optimality
    python solvers/generate_straighten_up_training.py --engine     # interpreter proof
    python solvers/generate_straighten_up_training.py --audit      # rendering
    python solvers/generate_straighten_up_training.py --symmetry   # augmentation
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

GAME_NAME = "Straighten_Up"

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

#: A run of this many adjacent crates is a Path. The three ``late right/down
#: [Crate | Crate | Crate]`` rules and the pair that strips runs of one and two
#: back off reduce to exactly this threshold; see the module docstring.
RUN = 3

#: Disk cache of every level's start plan AND its optimal-action sets. Unlike
#: the small ps: sokobans this one IS load-bearing: a cold derivation enumerates
#: ~3.9M model states across the six levels and runs L5's product search, about
#: a minute and a couple of GB, and every `parallelize_generator` shard would
#: otherwise repeat it. Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "straighten_up_plans.json"


# ---------------------------------------------------------------------------
# The native model -- one connected room
# ---------------------------------------------------------------------------

class _Room:
    """One connected component of a level's floor: its geometry, its crates,
    the mechanic, and its exact distance-to-win field.

    Cells are flat ``r * w + c`` indices into the WHOLE board (rooms share a
    coordinate system, which is what lets `_Level` decode a joint state back
    into a grid without bookkeeping). A room STATE is
    ``bytes([player]) + bytes(sorted movable crate cells)`` -- packed rather
    than a tuple of tuples because L5's two rooms are 1.87M states each and the
    difference is hundreds of megabytes.

    ``solid`` is "never moves and blocks": walls AND FixedCrates, which have no
    push rule and so are walls with one extra job (they are crates for the win
    condition). ``fixed`` is this room's FixedCrate cells, needed for exactly
    that job.

    The mechanic is the single push rule resolved under the collision layers:

      * step onto a wall, a fixed crate, or off the board -> nothing happens;
      * step onto a movable crate -> it slides one further, UNLESS that cell is
        blocked or holds another crate, in which case nothing happens at all
        (the rule's right-hand side moves both bodies or neither, so the
        player's own move is cancelled with the crate's -- which is also why
        two crates in a row are a wall);
      * anything else -> the player moves.

    A room with no player is a special case handled by `_Level`: its crates can
    never move, so it has no state at all and the level is winnable only if it
    is already won.
    """

    __slots__ = ("h", "w", "cells", "solid", "fixed", "edge", "player",
                 "crates", "_won_cache", "_reachable", "_dist")

    def __init__(self, h, w, cells, solid, fixed, player, crates):
        self.h, self.w = h, w
        #: The component's walkable cells (fixed crates are NOT among them).
        self.cells = frozenset(cells)
        self.solid = solid                     # bytearray over the whole board
        self.fixed = frozenset(fixed)
        self.player = player                   # start cell, or None
        self.crates = tuple(sorted(crates))
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
        #: `won` depends only on the CRATE half of a state, and a room has ~40
        #: player cells per crate layout, so memoising on that half alone turns
        #: the win test from the enumeration's dominant cost into a lookup.
        self._won_cache: dict = {}
        self._reachable: set = set()
        self._dist: dict = {}

    # -- the mechanic ---------------------------------------------------------
    def step(self, state: bytes, di: int) -> bytes:
        """The state after pressing ``DIRS[di]``, or ``state`` if the press does
        nothing (which is a non-move: no shortest path contains one)."""
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
            return bytes([n1]) + state[1:]
        n2 = self.edge[n1][di]
        if n2 < 0 or self.solid[n2]:
            return state
        for j in range(1, len(state)):
            if state[j] == n2:                 # a second crate behind the first
                return state
        out = bytearray(state)
        out[0] = n1
        out[k] = n2
        crates = sorted(out[1:])
        out[1:] = bytes(crates)
        return bytes(out)

    def preds(self, state: bytes):
        """Every state one press BEFORE ``state``.

        Two inverses. A walk is undone by stepping the player back onto
        ``p - d``, which has to be on the board, not solid and not a crate (it
        was free when the player left it). A push is undone the same way with
        the crate dragged one cell towards the player: the crate now at ``p + d``
        was at ``p``, so the earlier board is ``crates - {p + d} + {p}``, and
        ``p - d`` has to be free of THAT board, not of this one.

        Presses the room REFUSED are deliberately not inverted: they leave the
        state alone, so no shortest path contains one. (A press that does
        nothing HERE can still be the press that advances the other room on a
        two-room level -- which is exactly why `_Level._joint` does not use this
        method at all and sweeps backward over the forward edges its corridor
        pass recorded.)

        Enumerating the inverses directly is what lets the single-room backward
        sweep run without a reverse adjacency table over 1.9M states."""
        p = state[0]
        out = []
        body = state[1:]
        for di in range(4):
            q = self.edge[p][_OPPOSITE[di]]
            if q < 0 or self.solid[q]:
                continue
            if q not in body:
                out.append(bytes([q]) + body)                # undo a walk
            n2 = self.edge[p][di]
            if n2 >= 0 and n2 in body:                       # undo a push
                crates = sorted(x for x in body if x != n2)
                crates.append(p)
                crates.sort()
                if q not in crates:
                    out.append(bytes([q]) + bytes(crates))
        return out

    # -- the win condition ----------------------------------------------------
    def won(self, state: bytes) -> bool:
        """Every crate in this room -- movable and fixed -- in a run of `RUN`.

        This is the reduction of the six ``late`` Path rules stated in the
        module docstring, and it is checked against the interpreter's own
        ``check_win`` after every press of ``--selfcheck``'s random walk, and
        against an independent combinatorial enumeration of the winning layouts
        in ``--goals``."""
        body = state[1:]
        got = self._won_cache.get(body)
        if got is None:
            got = self._won_cache[body] = self._is_won(frozenset(body)
                                                       | self.fixed)
        return got

    def _is_won(self, crates: frozenset) -> bool:
        w, h = self.w, self.h
        for i in crates:
            r, c = divmod(i, w)
            n = 1
            j = c - 1
            while j >= 0 and r * w + j in crates:
                n += 1
                j -= 1
            j = c + 1
            while j < w and r * w + j in crates:
                n += 1
                j += 1
            if n >= RUN:
                continue
            n = 1
            j = r - 1
            while j >= 0 and j * w + c in crates:
                n += 1
                j -= 1
            j = r + 1
            while j < h and j * w + c in crates:
                n += 1
                j += 1
            if n < RUN:
                return False
        return True

    # -- the field ------------------------------------------------------------
    def field(self, state: bytes) -> dict:
        """``{state: presses to a win}`` for this room, exact on every state
        reachable from ``state``.

        Two sweeps. Forward, the whole reachable space (a won state is NOT
        terminal here: on a two-room level this room may have to hold its
        answer while the other one finishes, and on a one-room level the
        overshoot is a few percent of a space that costs seconds). Backward
        from every won state over `preds`, pruned to that reachable set.

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


# ---------------------------------------------------------------------------
# The native model -- a whole level
# ---------------------------------------------------------------------------

class _Level:
    """A level as the set of independent `_Room`s one keyboard drives.

    A level splits into connected components of everything that is not a Wall,
    and nothing crosses a boundary: the player cannot walk out of its room, a
    crate is only ever pushed to an adjacent floor cell, and a run of crates is
    a chain of adjacent crates so it lies in one room too. Five of the six
    levels have a single room; L5 ("Stunt Double") has two, each with its own
    player, and both take every press.

    A joint STATE is the tuple of the ACTIVE rooms' states, in room order.
    Playerless rooms carry no state at all -- their crates can never move -- so
    they are checked once, at construction, and make the level unwinnable if
    they are not already won.

    THE ONE THING THAT HAS TO BE CHECKED. A room is cut out over the
    CRATE-ABLE cells (everything that is not a Wall, so floor AND fixed
    crates), because that is the granularity the WIN condition needs: a run of
    three can pass through a fixed crate, and a room whose boundary ignored
    that could split a legal line in half and declare a won board lost. Movement
    is coarser -- a player and a crate move over FLOOR only, and a fixed crate
    is a wall to both -- so a crate-able component could in principle contain
    two floor components sealed off from each other by a fixed crate, and then
    the two would share a win condition while taking the same presses, which
    this decomposition cannot express. It does not happen on any shipped level
    (each crate-able component's floor is a single component on all six), and
    the constructor asserts it rather than trusting it, so an edited level fails
    loudly instead of being solved against the wrong win condition.
    """

    __slots__ = ("h", "w", "rooms", "active", "static_ok")

    def __init__(self, h, w, walls, fixed, crates, players):
        self.h, self.w = h, w
        solid = bytearray(h * w)
        for i in walls:
            solid[i] = 1
        for i in fixed:
            solid[i] = 1
        wall_only = bytearray(h * w)
        for i in walls:
            wall_only[i] = 1

        def components(blocked):
            comp: dict = {}
            groups: list = []
            for i in range(h * w):
                if blocked[i] or i in comp:
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
                        if blocked[y] or y in comp:
                            continue
                        comp[y] = gid
                        groups[gid].append(y)
                        queue.append(y)
            return comp, groups

        _crateable, groups = components(wall_only)     # rooms, for the win
        floor_of, _floor_groups = components(solid)    # reach, for the moves

        self.rooms = []
        for gid, members in enumerate(groups):
            cells = [i for i in members if not solid[i]]
            inside = {floor_of[i] for i in cells}
            if len(inside) > 1:
                raise ValueError(
                    f"Straighten Up: crate-able component {gid} holds "
                    f"{len(inside)} separate floor components -- a line of "
                    f"three could span them while their players cannot reach "
                    f"each other, which _Level does not model")
            self.rooms.append(_Room(
                h, w, cells, solid,
                [i for i in members if solid[i]],
                next((p for p in players if p in cells), None),
                [i for i in crates if i in cells]))
        #: The rooms a press can change. The rest are frozen scenery.
        self.active = [rm for rm in self.rooms if rm.player is not None]
        #: A playerless room's crates never move, so its half of the win
        #: condition is decided at reset and can never change. False makes the
        #: whole level dead. (The leading 0 is a dummy player byte: `_Room.won`
        #: reads only the crate half of a state, and such a room has no player
        #: to put there.)
        self.static_ok = all(
            rm.won(bytes([0]) + bytes(rm.crates))
            for rm in self.rooms if rm.player is None)

    # -- states ---------------------------------------------------------------
    def step(self, jstate: tuple, di: int) -> tuple:
        return tuple(rm.step(s, di) for rm, s in zip(self.active, jstate))

    def won(self, jstate: tuple) -> bool:
        return self.static_ok and all(
            rm.won(s) for rm, s in zip(self.active, jstate))

    def decode(self, jstate: tuple):
        """``(players, movable crates)`` as sorted cell tuples, for comparing
        the model's board against the interpreter's grid."""
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
        None means some room can no longer win at all, which sinks the state."""
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
        once. Complete and exact: `heuristic` is admissible and the weight is 1,
        so the first winning state popped is at its true distance."""
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
                heapq.heappush(pq, (ng + h, ng, counter, nxt))
        return None


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
# The winning layouts, derived combinatorially (an independent win condition)
# ---------------------------------------------------------------------------

def win_layouts(room: _Room, cap: int = 2_000_000) -> "list[frozenset] | None":
    """Every crate layout that satisfies this room's win condition, derived
    WITHOUT `_Room.won`.

    A layout wins iff every crate is in a run of >= `RUN`, which is the same as
    saying the layout is a union of such runs. So: enumerate the runs (every
    window of >= `RUN` consecutive crate-able cells along a row or a column --
    crate-able being this room's floor plus its fixed cells), then take every
    union of runs whose size is exactly the room's crate count and which
    contains every fixed crate. Returns None past ``cap`` unions, a runaway
    guard that the shipped rooms are nowhere near.

    Used by ``--goals`` to show what each level is actually asking for, and by
    ``--selfcheck`` as a second opinion on `_Room.won` over every reachable
    state of the one-room levels."""
    w, h = room.w, room.h
    crateable = set(room.cells) | room.fixed
    total = len(room.crates) + len(room.fixed)

    runs = []
    for r in range(h):
        line = [r * w + c for c in range(w)]
        runs.extend(_runs_in(line, crateable))
    for c in range(w):
        line = [r * w + c for r in range(h)]
        runs.extend(_runs_in(line, crateable))
    runs = [frozenset(x) for x in runs]

    out: list = []
    seen: set = set()

    def grow(i, cur):
        if len(out) > cap or len(seen) > cap:
            return
        if len(cur) == total:
            if room.fixed <= cur and cur not in seen:
                seen.add(cur)
                out.append(cur)
            return
        for j in range(i, len(runs)):
            nxt = cur | runs[j]
            if len(nxt) > total:
                continue
            grow(j + 1, nxt)

    grow(0, frozenset())
    return None if len(seen) > cap else out


def _runs_in(line, crateable) -> list:
    """Every window of `RUN` or more consecutive crate-able cells in ``line``."""
    out = []
    n = len(line)
    i = 0
    while i < n:
        if line[i] not in crateable:
            i += 1
            continue
        j = i
        while j < n and line[j] in crateable:
            j += 1
        for a in range(i, j):
            for b in range(a + RUN, j + 1):
                out.append(line[a:b])
        i = j
    return out


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class StraightenUpExpert(PSExpert):
    """`PSExpert`'s plan memo, disk cache and snapshot discipline around a
    `_Level`'s fields.

    The base class keeps the memo, the level scoping and the disk cache; only
    the strategy underneath changes, the way `PSEnumExpert` replaces it. Here
    the strategy is "split the board into rooms, enumerate each one's exact
    distance field, and descend it (jointly, if there is more than one room)",
    so `heuristic` is never called and asserts rather than returning a number
    nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state (a recovery prefix, an epsilon detour) is answered exactly
    -- including "this board is now dead", which an exploration prefix really
    does produce here: a crate shoved flat against a wall where no line of three
    can contain it ends the level."""

    directions = list(DIRS)
    #: `_key` is the dynamic objects only (players + movable crates), which is
    #: canonical WITHIN a level but not across them -- walls and fixed crates
    #: are static per level and differ between levels.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        g = self.g
        self.player_ids = set(self.game._engine._player_indices)
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.move_ids = set(g.resolve_object_name("movecrate"))
        self.fixed_ids = set(g.resolve_object_name("fixedcrate"))
        self.dyn_ids = self.player_ids | self.move_ids
        #: `_Level`s by STATIC signature (see `read`), not by level index, so a
        #: level's rooms -- and the fields they cache -- are built once.
        self._levels: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "StraightenUpExpert descends exact distance fields; "
            "heuristic is unused")

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

        Models are cached by their STATIC signature (dimensions, walls, fixed
        crates), not by level index, so the room decomposition, the edge tables
        and -- crucially -- the enumerated fields are shared by every state of
        the level."""
        h, w = eng.height, eng.width
        walls, fixed, crates, players = [], [], [], []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.fixed_ids:
                    fixed.append(i)
                if cell & self.move_ids:
                    crates.append(i)
                if cell & self.player_ids:
                    players.append(i)
        sig = (h, w, tuple(walls), tuple(fixed))
        model = self._levels.get(sig)
        if model is None:
            model = self._levels[sig] = _Level(h, w, walls, fixed,
                                               crates, players)
        jstate = []
        for rm in model.active:
            here = [p for p in players if p in rm.cells]
            if len(here) != 1:
                return model, ()       # a player left the board; see `_search`
            jstate.append(bytes(here)
                          + bytes(sorted(i for i in crates if i in rm.cells)))
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

class StraightenUpSolver(PSAStarSolver):
    game_id = "puzzlescript_straighten_up"
    game_name = GAME_NAME
    expert_cls = StraightenUpExpert

    #: `games/ps:straighten_up/ps:straighten_up.py` is a plain passthrough (it
    #: constructs the adapter and nothing else), so there is nothing to gain by
    #: routing through it -- but it IS what a live agent is handed, so if that
    #: wrapper ever grows a patch this must be set to ``"ps:straighten_up"``.
    game_module_id = ""

    #: Unused: the expert enumerates exact fields rather than searching under a
    #: heuristic, so there is no weight to trade. ``node_cap`` is likewise never
    #: consulted -- the one search in this file that could run away, L5's joint
    #: A*, is bounded by an admissible heuristic and finishes in 213k pops.
    weight = 1
    node_cap = 400_000

    #: The longest plan is 66 presses (level 5); the rest is room for a re-plan
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
    solver = StraightenUpSolver()
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _plans() -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the fields and the tie labelling at
    once."""
    _solver, game, expert, solvable = _new()
    idx = game._game.obj_name_to_idx
    total = ties = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        model, jstate = expert.read(eng)
        moves = sum(1 for row in eng.grid for cell in row
                    if idx["movecrate"] in cell)
        fixed = sum(1 for row in eng.grid for cell in row
                    if idx["fixedcrate"] in cell)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level:2d}: UNSOLVED")
            bad += 1
            continue
        for direction in plan:
            eng.step(direction)
        won = eng.check_win()
        bad += not won
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        tie_steps = sum(1 for s in sets if len(s) > 1)
        total += len(plan)
        ties += tie_steps
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"  L{level:2d}: {eng.height:2d}x{eng.width:2d}  "
              f"{len(model.active)} room(s), {len(model.rooms)} component(s)  "
              f"{moves} movable + {fixed} fixed crates  {len(plan):3d} presses  "
              f"win={won}  (budget {game._max_steps}, {room})  "
              f"{tie_steps:3d} steps with a tie set")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _goals(verbose: bool = True) -> int:
    """The win condition, derived combinatorially and drawn.

    `win_layouts` enumerates the winning crate layouts of every room from the
    geometry alone -- unions of runs of three -- with no reference to
    `_Room.won`. Printing them is what makes a level's ASK concrete: L4's four
    fixed crates admit exactly ONE layout (the eight cells of the ring around
    the player's starting square), which is why its plan is forced.

    Returns the number of rooms with no winning layout at all, which would mean
    a level that cannot be won for a reason no search should have to discover."""
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    impossible = 0
    for level in range(game.n_levels):
        game.set_level(level)
        model, jstate = expert.read(eng)
        for ri, rm in enumerate(model.rooms):
            layouts = win_layouts(rm)
            n = "capped" if layouts is None else len(layouts)
            impossible += layouts is not None and not layouts
            if not verbose:
                continue
            print(f"  L{level:2d} room {ri}: {len(rm.cells):3d} floor cells, "
                  f"{len(rm.crates)} movable + {len(rm.fixed)} fixed crates, "
                  f"{n} winning layout(s)")
            if layouts:
                best = min(layouts, key=lambda s: sorted(s))
                for r in range(rm.h):
                    row = ""
                    for c in range(rm.w):
                        i = r * rm.w + c
                        row += ("O" if i in best and i in rm.fixed else
                                "o" if i in best else
                                "." if i in rm.cells else
                                "F" if i in rm.fixed else "#")
                    print(f"      {row}")
    return impossible


def _selfcheck(walk_presses: int = 3000, samples: int = 120,
               joint_samples: int = 40, joint_tie_steps: int = 4,
               verbose: bool = True) -> int:
    """Five things the expert would otherwise be trusted on.

    1. **The native model is the interpreter's, board AND win flag.** A seeded
       random walk on every level, comparing `_Level`'s decoded board against
       the engine's grid AND `_Level.won` against ``check_win`` after EVERY
       press -- including the presses that do nothing, which is where a
       collision-layer mistake hides, and including ACTION5, which must be a
       no-op (the model does not model it at all). The walk restarts the level
       whenever it stumbles into a win, so it keeps sampling fresh boards.
    2. **The win condition is what the docstring says it is.** For every
       one-room level, `_Room.won` is compared against `win_layouts`'
       combinatorial enumeration on every reachable state. Two derivations of
       "all Crate on Path" that share no code.
    3. **The plans are shortest.** A plain forward BFS to the first win, which
       is shortest by construction and shares no code with the fields' backward
       half. On L5 that BFS is not affordable in the product, so what is checked
       instead is the bound the joint search rests on: neither room can be won
       in fewer presses than its own field says, so ``d* >= max`` -- and the
       joint A* is complete under an admissible heuristic at weight 1, which is
       the other half of the proof.
    4. **The tie sets are exact.** Every step's optimal set on the one-room
       levels is re-derived by brute force -- a fresh depth-bounded BFS from
       each of the four successors -- and must match the field's answer exactly.
    5. **Recovery is answered from ANY board, not just the start.** This is the
       one that matters at generation time: the exploration prefix really does
       strand this game (shove a crate somewhere no line of three can reach it
       and no press wins again), so `_search` has to be right about arbitrary
       states in both directions. From ``samples`` states reached by seeded
       random presses on the one-room levels, the field must either return a
       plan that WINS when replayed through the interpreter, or return None on
       a board an independent forward BFS also finds no win from. A None that
       is merely a give-up would be a silent bug: `record_level` reads it as
       "reset", and a wrong plan length here would not show up in any of the
       checks above, all of which only ever ask about starts.
    6. **L5, which every pass above has to skip, sampled.** The product is not
       enumerable, so the brute-force answers 2, 4 and 5 lean on do not exist
       there. What is affordable is a re-derivation of the pieces:
       ``joint_samples`` perturbed boards, each either re-planned to a plan that
       WINS when replayed through the interpreter, or refused -- and a refusal
       is checked by running an independent plain forward BFS (`_first_win`) on
       the room whose field disowned the state, which must agree that the room
       can no longer be won. (A refusal with every room still alive would be
       the joint A* having closed the product instead, which is a proof as well
       but not one a second search can afford to repeat, so those are counted
       and reported rather than re-checked.) Then ``joint_tie_steps`` steps
       have their optimal SETS re-derived from scratch: for each of the four
       presses, a fresh joint A* from the successor, kept iff it finishes in
       exactly the moves remaining. That A* shares nothing with the corridor
       sweep the sets were read off, so agreement is a second derivation of the
       one thing on L5 that is otherwise argued rather than measured.
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
                if idx["movecrate"] in cell:
                    crates.append(i)
        return tuple(sorted(players)), tuple(sorted(crates))

    # 1 -- the model is the interpreter's.
    for level in range(game.n_levels):
        game.set_level(level)
        model, jstate = expert.read(eng)
        rng = random.Random(f"straighten_up:selfcheck:{level}")
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

    # 2 -- the win condition, twice.
    for level in range(game.n_levels):
        game.set_level(level)
        model, jstate = expert.read(eng)
        if len(model.active) != 1:
            if verbose:
                print(f"  L{level:2d}: win-layout cross-check SKIPPED "
                      f"({len(model.active)} rooms -- see --goals)")
            continue
        room = model.active[0]
        layouts = set(win_layouts(room) or ())
        room.field(jstate[0])
        mismatch = 0
        for state in room._reachable:
            truth = (frozenset(state[1:]) | room.fixed) in layouts
            if truth != room.won(state):
                mismatch += 1
                break
        bad += mismatch
        if verbose:
            print(f"  L{level:2d}: {len(room._reachable)} reachable states, "
                  f"{len(layouts)} winning layouts -- "
                  f"{'AGREE' if not mismatch else 'DISAGREE'}")

    # 3 -- the plans are shortest.
    for level in range(game.n_levels):
        game.set_level(level)
        model, jstate = expert.read(eng)
        plan = expert.plan(eng, level)
        if len(model.active) == 1:
            room = model.active[0]
            shortest = _first_win(room, jstate[0])
            ok = plan is not None and shortest == len(plan)
            if verbose:
                print(f"  L{level:2d}: field {len(plan) if plan else None} presses"
                      f" vs an independent forward BFS {shortest} -- "
                      f"{'SHORTEST' if ok else 'NOT SHORTEST'}")
        else:
            bounds = [_first_win(rm, s)
                      for rm, s in zip(model.active, jstate)]
            ok = plan is not None and len(plan) >= max(bounds)
            if verbose:
                print(f"  L{level:2d}: joint {len(plan) if plan else None} presses"
                      f" vs per-room lower bounds {bounds} -- "
                      f"{'CONSISTENT' if ok else 'IMPOSSIBLE'} (A* under an "
                      f"admissible max is what makes it exact)")
        bad += not ok

    # 4 -- the tie sets are exact.
    for level in range(game.n_levels):
        game.set_level(level)
        model, jstate = expert.read(eng)
        plan = expert.plan(eng, level)
        if len(model.active) != 1:
            if verbose:
                print(f"  L{level:2d}: tie-set brute force SKIPPED "
                      f"({len(model.active)} rooms -- the product is not "
                      f"enumerable; pass 6 samples it with a fresh joint A*)")
            continue
        room = model.active[0]
        sets = getattr(plan, "optsets", None) or []
        cur = jstate[0]
        mismatched = 0
        for i, direction in enumerate(plan):
            rest = len(plan) - i
            truth = []
            for di in range(4):
                nxt = room.step(cur, di)
                if nxt == cur:
                    continue
                if _bounded(room, nxt, rest - 1) == rest - 1:
                    truth.append(DIRS[di])
            if sorted(truth) != sorted(sets[i]):
                mismatched += 1
            cur = room.step(cur, DIRS.index(direction))
        bad += mismatched
        if verbose:
            print(f"  L{level:2d}: {len(plan)} tie sets "
                  f"{'match' if not mismatched else f'{mismatched} MISMATCH'}"
                  f" brute force")

    # 5 -- recovery from arbitrary boards, checked against the interpreter.
    for level in range(game.n_levels):
        game.set_level(level)
        model, _jstate = expert.read(eng)
        if len(model.active) != 1:
            if verbose:
                print(f"  L{level:2d}: recovery sampling SKIPPED "
                      f"({len(model.active)} rooms -- each independent forward "
                      f"BFS would be a product enumeration)")
            continue
        room = model.active[0]
        rng = random.Random(f"straighten_up:recovery:{level}")
        wrong = dead = won_after = 0
        for _ in range(samples):
            game.set_level(level)
            for _ in range(rng.randrange(1, 30)):
                eng.step(DIRS[rng.randrange(4)])
                if eng.check_win():
                    break
            if eng.check_win():
                continue                 # the walk already won; nothing to plan
            _m, js = expert.read(eng)
            plan = expert._search(eng)
            truth = _bounded(room, js[0], len(room.solid) * 8)
            if plan is None:
                dead += 1
                wrong += truth is not None
                continue
            if truth != len(plan):
                wrong += 1
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
                  f"re-planned to a WIN in the interpreter, {dead} proved dead, "
                  f"{wrong} WRONG")

    # 6 -- the multi-room levels, which passes 2, 4 and 5 cannot reach.
    for level in range(game.n_levels):
        game.set_level(level)
        model, _jstate = expert.read(eng)
        if len(model.active) < 2:
            continue
        rng = random.Random(f"straighten_up:joint:{level}")
        wrong = dead = exhausted = won_after = 0
        for _ in range(joint_samples):
            game.set_level(level)
            for _ in range(rng.randrange(1, 30)):
                eng.step(DIRS[rng.randrange(4)])
                if eng.check_win():
                    break
            if eng.check_win():
                continue
            _m, js = expert.read(eng)
            plan = expert._search(eng)
            if plan is None:
                # Two ways to refuse, and only one of them has an independent
                # second opinion. Usually a ROOM is dead, and that is checked
                # by a plain forward BFS over the room -- code the field's
                # backward half shares nothing with. The other way is the joint
                # A* closing the whole product with no win in it; that is a
                # proof too (the search is complete under an admissible
                # heuristic) but re-deriving it means enumerating the product,
                # so it is counted and reported rather than re-checked.
                fields = [rm.field(s) for rm, s in zip(model.active, js)]
                culprits = [(rm, s) for rm, f, s in zip(model.active, fields, js)
                            if f.get(s) is None]
                if not culprits:
                    exhausted += 1
                    continue
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
            print(f"  L{level:2d}: {joint_samples} random boards -- {won_after} "
                  f"re-planned to a WIN in the interpreter, {dead} refused with "
                  f"a dead room (confirmed by an independent BFS), {exhausted} "
                  f"refused by the joint search closing, {wrong} WRONG")

        game.set_level(level)
        model, jstate = expert.read(eng)
        plan = expert.plan(eng, level)
        fields = [rm.field(s) for rm, s in zip(model.active, jstate)]
        sets = getattr(plan, "optsets", None) or []
        stride = max(1, len(plan) // joint_tie_steps)
        checked = mismatched = 0
        cur = jstate
        for i, direction in enumerate(plan):
            if i % stride == 0 and checked < joint_tie_steps:
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
            print(f"  L{level:2d}: {checked} tie set(s) re-derived by a fresh "
                  f"joint A* -- "
                  f"{'match' if not mismatched else f'{mismatched} MISMATCH'}")
    return bad


def _first_win(room: _Room, state: bytes) -> "int | None":
    """Presses to a win, by a plain forward BFS that stops at the first one.
    Shortest by construction, and it shares no code with `_Room.field`'s
    backward half -- which is the point of running it."""
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


def _bounded(room: _Room, state: bytes, limit: int) -> "int | None":
    """`_first_win` with a depth cap: presses to a win, or None past ``limit``."""
    if room.won(state):
        return 0
    if limit <= 0:
        return None
    seen = {state}
    queue = deque([(state, 0)])
    while queue:
        cur, d = queue.popleft()
        if d >= limit:
            continue
        for di in range(4):
            nxt = room.step(cur, di)
            if nxt == cur or nxt in seen:
                continue
            if room.won(nxt):
                return d + 1
            seen.add(nxt)
            queue.append((nxt, d + 1))
    return None


def _engine(levels=(0,), verbose: bool = True) -> int:
    """The proof that owes nothing to the native model: enumerate the REAL
    interpreter.

    `StateGraph.build` walks every state reachable from the level start by
    pressing actual buttons on the engine and keying the resulting grid, then
    solves the graph exactly. It shares no line of code with `_Room` -- no
    modelled push, no modelled collision layer, no modelled win test, and in
    particular no modelled reduction of the six Path rules -- so agreement on
    both the plan LENGTH and the per-step optimal SET is an independent
    derivation of everything this file claims.

    L0 (about 11k model states) is the default because it is instant; L3 is
    affordable too at 87k and is worth the couple of minutes, being the first
    level with a FixedCrate. Pass level numbers on the command line
    (``--engine 0 3``) to choose. L2 alone is 650k states, i.e. 2.6M interpreter
    steps, and is not."""
    from solvers.common.ps_astar import StateGraph                     # noqa: PLC0415

    _solver, game, expert, _solvable = _new()
    eng = game._engine
    bad = 0
    for level in levels:
        game.set_level(level)
        plan = expert.plan(eng, level)
        game.set_level(level)
        graph = StateGraph.build(eng, expert._key, list(DIRS),
                                 node_cap=1_000_000)
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
                  f"{len(eplan)} presses vs the field's {len(plan)} -- "
                  f"{'AGREE' if same_len else 'DISAGREE'}; tie sets "
                  f"{'AGREE' if same_sets else 'DISAGREE'}")
    return bad


def _audit(verbose: bool = True) -> int:
    """Two passes over every reachable cell COMPOSITION.

    **Pass 1 -- distinctness**, at every cell size the six boards use (7 px for
    the five 9x9 levels, 3 px for L5's 9x17; the size list is derived rather
    than assumed so an edited level is covered). Whole 64x64 frames of uniform
    boards are compared rather than one cell out of a mixed board:
    `_render_frame` upscales and centre-pads, so slicing a cell by ``cell_px``
    arithmetic reads the wrong pixels (the ps:explod lesson). Two uniform boards
    render identically iff their cells do.

    The compositions are the ones the engine can actually produce. A crate wears
    BlueCrate exactly when it is on a Path, so ``crate`` and ``crate+bluecrate``
    are the loose and the finished crate; PathHoriz/PathVert are TRANSPARENT and
    draw nothing, so they are not compositions at all. ``player + crate`` is
    absent on purpose and is not an omission -- they share a collision layer.
    The Arm compositions are the victory pose: an arm is parked in the cell
    either side of the player, so it lands on whatever is there.

    **Pass 2 -- the group.** The game is in `_FLIP_GAMES`, so a board is drawn at
    any of the eight turns and mirrors. That is fine as long as no transform of
    one composition is another composition's art, which is what this checks: all
    eight transforms of each, against all others. It runs on SQUARE boards,
    because a rotation of a non-square board also moves the letterbox and every
    comparison would pass for the wrong reason.

    ONE pair is exempt, and it is exempt because it is the thing that makes the
    mirrors sound rather than a hole in them: LeftArm and RightArm ARE each
    other under a mirror, by construction. The victory pose parks RightArm in
    the cell left of the player and LeftArm in the cell right of it, each a
    single pixel column hugging the player, so the three-cell strip is
    symmetric as a whole and an hflip maps the pose onto itself -- exactly the
    orbit argument `_FLIP_GAMES` needs. A composition is therefore compared
    against every other EXCEPT its own arm-swapped partner; anything else
    matching is still a clash."""
    from adapters.puzzlescript_adapter import _render_frame            # noqa: PLC0415

    _solver, game, _expert, _solvable = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = [(), ("wall",),
             ("movecrate",), ("movecrate", "bluecrate"),
             ("fixedcrate", "fixeddot"),
             ("fixedcrate", "bluecrate", "fixeddot"),
             ("greenplayer",),
             ("leftarm",), ("rightarm",),
             ("movecrate", "leftarm"), ("movecrate", "bluecrate", "rightarm"),
             ("fixedcrate", "fixeddot", "leftarm"),
             ("fixedcrate", "bluecrate", "fixeddot", "rightarm"),
             ("wall", "leftarm")]

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

    def swap_arms(comp):
        """``comp`` with LeftArm and RightArm exchanged -- the one partner a
        composition is allowed to map onto under a mirror. See the docstring."""
        return tuple("rightarm" if o == "leftarm" else
                     "leftarm" if o == "rightarm" else o for o in comp)

    group = [(k, hf, vf) for k in range(4)
             for hf in (False, True) for vf in (False, True)]
    for cell in sorted({min(64 // h, 64 // w) for h, w in sizes}):
        n = 64 // cell
        shots = {c: shoot(n, n, c) for c in comps}
        clashes = [(a, b, t) for a, b in itertools.permutations(comps, 2)
                   for t in group
                   if b != swap_arms(a)
                   and np.array_equal(transform(shots[a], *t), shots[b])]
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
    docstring -- one push rule written with the relative ``>``, moving at most
    two bodies onto squares that can never coincide, and a Path pair that is a
    complete orbit under a quarter turn -- but Gobble Rush's chirality hid
    inside exactly this kind of argument, so it is measured.

    Both the PLANS and a seeded random walk are replayed: the walk reaches the
    boards a plan never visits (a crate flat against a wall, two crates jammed
    together, a crate dead where no line can reach it) and it presses the
    unbound ACTION key as well."""
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
            rng = random.Random(f"straighten_up:symmetry:{level}")
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
    if "--goals" in sys.argv:
        impossible = _goals()
        print(f"goals: {impossible} room(s) with no winning layout")
        sys.exit(1 if impossible else 0)
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
    sys.exit(StraightenUpSolver.main())
