"""Generate Phase-1 training data for the PuzzleScript game ps:the_blob
("The Blob", Guillem G T).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, a native model of them, the exact distance FIELD that model is
searched with, the proof that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_the_blob",
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

The game
--------
YOU ARE THE WHOLE BLOB AT ONCE, and the puzzle is its SHAPE.

Every cell of the blob is its own `Player` object, they all take the same force
from one arrow key, and the win condition is positional:

    All Target on Player
    No AntiTarget on Player
    Some Target

So a press TRANSLATES a set of cells by one. If that were all, the game would be
unwinnable by construction on four of its five levels -- level 0 starts as a
down-right diagonal and its targets are a down-LEFT one, and no translation
turns one into the other. The mechanic is what happens when the translation is
only PARTLY legal:

* a blob cell whose destination is a Wall or is OFF THE BOARD does not move;
* every other cell does -- including the ones directly behind a cell that just
  moved, so the blob slides through itself freely;
* but blocking PROPAGATES BACKWARDS along the pressed axis: a cell whose
  destination holds a blob cell that is itself blocked is blocked too.

Shoving the blob into a wall (or into the edge of the board, which on these
small boards is the geometry that matters most -- level 0's only wall is one
square) therefore SHEARS it: the free part advances, the jammed part stays, and
the blob comes out a different shape. Reshaping it into the target pattern by
choosing what to jam against what is the entire game. That is also why plans
here contain stretches that look wasted -- level 2 presses ``left`` four times
and then ``right`` three times, which is not a detour but the compression stroke
followed by the re-expansion, and it is provably part of a shortest answer.

Two consequences worth stating because they shape the solver:

* **The blob's cell COUNT never changes**, and it never splits or merges in any
  way that a set of cells does not already express. Every shipped level has
  exactly as many blob cells as Targets (3, 4, 8, 8, 8), so a win is the blob
  standing on precisely the target set.
* **Presses are not reversible.** ``right`` then ``left`` returns the blob to
  where it started only if nothing was blocked; the moment a shear happens, the
  press that undoes the motion does not undo the shape. So the space is a
  genuine digraph rather than an undirected graph, and the search cannot assume
  a board is winnable (see Recovery for what was measured).

THE RULES DO NOT FIRE. The .txt carries a second, larger game -- the blob GROWS
from `Life` tiles (a `GoodPlayer` on one becomes a `BabyPlayer`, which seeds
every free neighbour, which matures the following turn) and DIES from `Poison`
(a wave of `DyingPlayer` that eats one ring of the blob per turn). It is a nice
mechanic and no shipped level uses it: not one places a `Life`, a `Poison` or an
`AntiTarget`, so `PlayerSeed`, `BabyPlayer`, `DyingPlayer` and
`RecentDyingPlayer` can never come into existence either. That is not assumed --
`_Board.read` raises if any of them is ever on the board, and ``--selfcheck``'s
fuzz walk re-checks it after every one of its presses on every level. What
remains, and what this file models, is the movement resolution above.

``noaction`` is declared in the header and no rule has ``action`` on its
left-hand side, so ACTION5 is not a move; the search does not branch on it. The
exploration prefix still presses it (a live agent has that button), which is why
``--symmetry``'s random walk presses it too.

The five levels, all winnable, none already won at reset:

  * **L0** 5x6, 3 cells, 3 targets, ONE wall. A down-right diagonal that has to
    become a down-left one, and the board edge does most of the work.
  * **L1** 4x4, 4 cells, 4 targets, two walls. Starts with one blob cell already
    parked on a target (the ``Q`` legend char), and the four targets are a
    spread 2x2 that the solid 2x2 start block has to be pulled apart into.
  * **L2** 3x7, 8 cells, 8 targets, one wall. The targets are a RING around the
    wall; the blob is a 3x2 block with a two-cell arm on its middle row. 18
    presses, the long one, and the level whose answer contains the clearest
    compress-then-re-expand stroke (four ``left`` then three ``right``).
  * **L3** 5x6, 8 cells, 8 targets, one wall. A ring again, against a 3x2 block
    with a two-cell tail hanging off its corner.
  * **L4** 7x7, 8 cells, 8 targets, two walls. A 2x4 slab that has to become a
    ring around a wall, using a second wall floating above it as the only tool.
    Its reachable space is 18,531,158 states, which is why it is the level the
    search is designed around.

Expert solver
-------------
A NATIVE model of the mechanic above (`_Board`), searched to an exact distance
field rather than heuristically. A state is the set of blob cells and nothing
else -- walls and targets never move, and no other object can exist -- so it is
packed into ONE PYTHON INT, a bitmask over ``r * w + c``. That is not a
micro-optimisation: level 4's field visits 893,835 states, and a `frozenset` of
eight ``(r, c)`` tuples costs about a kilobyte each against ~32 bytes for the
int. The whole mechanic is then five bit operations:

    blocked  = P & wall_block[d]                 # destination is wall or edge
    repeat:    blocked |= P & shift(blocked, -d) # ...or a blocked blob cell
    result   = blocked | shift(P & ~blocked, d)

`_Board.field` is a forward BFS kept in LAYERS, plus a backward induction over
those layers:

  1. FORWARD, layer by layer, stopping at the first depth ``d*`` that produces a
     winning board. That fixes the answer's length and collects every state
     within ``d* - 1`` presses of the start.
  2. BACKWARD, ``on_path[d*]`` being the winning boards found while expanding
     layer ``d* - 1``, and ``on_path[j]`` every state of layer ``j`` with a
     successor in ``on_path[j + 1]``.

`on_path[j]` is exactly the set of states whose true distance to a win is
``d* - j``: having a successor in ``on_path[j+1]`` exhibits a path of that
length, and a shorter one would make ``g + h < d*``, contradicting the
minimality of ``d*``.

The reason for the layered shape rather than the backward BFS over inverse moves
that the ps: sokobans use is that THIS MOVE HAS NO ENUMERABLE INVERSE. Undoing a
push is local (step back, drag the crate towards you); undoing a shear is not --
the predecessor's blocked set is a function of the predecessor, so recovering it
from the result would mean guessing which subset of the blob stayed put. The
layered induction needs no inverse at all, at the cost of one extra sweep over
the layers: 2 x 4 x |F| model steps, which is 2.5 s on level 4 and under 0.1 s
on the other four.

Optimal-action sets
-------------------
The field gives them EXACTLY, with no inference: at plan step ``i`` the optimal
presses are every direction whose successor lies in ``on_path[i + 1]``.

The measured answer is that 9 of the 69 expert steps have a second equally-right
press (L0 0, L1 1, L2 3, L3 5, L4 0). Ties are rare here for a structural
reason: a press moves EVERY cell of the blob, so two different presses reaching
the same distance means two different shapes that are both exactly as close --
which the shear geometry mostly does not allow. Level 4 is the extreme case,
15 presses and not one of them with an alternative.

``--selfcheck`` re-derives every set on every level by brute force (a fresh
depth-bounded BFS from each of the four successors) and requires an exact match;
``--engine`` re-derives them again from the interpreter on the four levels small
enough to enumerate. No step ever ships unlabelled (the always-emit-optimal-
targets rule).

Shortest, and how that is known
-------------------------------
All five plans are provably shortest (10, 12, 18, 14 and 15 presses, 69 in
total), and the proof is three independent derivations agreeing:

  * this file's field (layered forward BFS + backward induction);
  * a plain forward BFS to the first win, which is shortest by construction and
    shares no code with the field's backward half (``--selfcheck``);
  * a full-space enumeration of the real INTERPRETER -- `StateGraph.build`
    walking every reachable engine state by pressing real buttons, with no
    native model involved at all (``--engine``: 1992, 595, 5181 and 86,659
    states on levels 0-3). That is the derivation that closes the loop back to
    the engine, and it agrees on all four lengths AND on every one of those
    levels' 54 tie sets.

Level 4 is excluded from the third pass and only from it: 18.5 million engine
states at ~20k interpreter steps per second is a fifteen-minute run and several
gigabytes of snapshots. Its plan is still checked end to end -- ``--plans``
replays it through the real interpreter to a WIN, and ``--selfcheck``'s
independent forward BFS confirms 15 is shortest.

Recovery
--------
`supports_recovery` with RESET-mode recovery. The expert re-plans from the LIVE
board, not from a stored path: `_search` reads whatever the engine currently
holds and runs both sweeps from there, so an exploration prefix that leaves the
blob in a shape a shortest plan would never produce is answered exactly --
including "this board is now dead", which `record_level` needs to hear in order
to fall back to a RESET. ``--selfcheck``'s fourth pass walks 120 random boards
per level and checks both halves against the interpreter: every plan returned
from a random board is replayed and must WIN, and every None must be a board an
independent forward BFS also proves dead, never a give-up.

WHAT THAT MEASURED, and it is worth writing down because it is not what the
irreversibility above predicts: NO DEAD BOARD HAS EVER BEEN FOUND. All 600 of
``--selfcheck``'s boards re-planned to a win, and so did a separate 1,320-board
probe over all five levels with random walks of up to 400 presses -- long enough
to shear the blob well away from anything a plan produces. Reachability here
appears to be
symmetric in practice even though single presses are not -- a shear can be
worked back out by shearing against something else. That is a measurement, not a
theorem, so the None path stays wired up and stays checked; it simply has not
fired.

The one thing `_search` will not do is answer a board whose field exceeds
`TheBlobExpert.state_cap`. That cap is a runaway guard for `--selfcheck`'s
random boards, not something generation can hit: the largest field any level
START needs is level 4's 893,835 states, and generation only ever plans from a
level start (the prefix ends in a RESET back to it).

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip with the matching directional action remap. This
game was ADDED to `PuzzleScriptAdapter._FLIP_GAMES` for this generator; the
argument is in the comment beside it and is the strongest form this family has:
a press gives every Player the same force, so the move is a TRANSLATION of a set
of cells, which is injective -- no two bodies can ever contest a destination, so
there is nothing for the interpreter's rule-expansion order to decide and the
chirality Gobble Rush has to argue around cannot arise. The blocking rule is
"wall or edge, propagating back along the pressed axis", which is exactly as
mirror-symmetric as it is rotation-symmetric, and the win condition names no
direction. 5 levels x 16 presentations = 80, against 20 with rotation alone.

``--symmetry`` measures it rather than trusting it: every level's plan AND a
seeded 200-press random walk (which does what a plan never does -- jams the blob
flat against every wall and presses the unbound ACTION key) replayed at all 16
presentations, requiring the ENGINE GRID to be identical at every one of them
and every FRAME to be the exact transform of the unaugmented frame.

There is deliberately no colour augmentation: the win is read off four corner
pixels of a Target showing through the blob standing on it, and a flattening
recolor would erase exactly that.

Rendering
---------
TWO bugs, both fixed in ``data/puzzlescript_games/The_Blob.txt`` (see the header
comment there for the full statement), and one of them was the whole win
condition:

* **The walls were the letterbox.** `Wall` shipped as bare ``Black``, ARC index
  5, which is the colour `_render_frame` pads a non-square board with. Every
  board here is 7x7 or smaller so every frame is mostly border, and level 1's
  wall sits ON the left edge. Wall is now ``DarkGray`` (3).
* **The goal vanished under the blob.** `Target` was a dark-blue ring on
  rows/cols 1-3 and every Player variant shipped with NO sprite lines, which the
  parser renders as a solid opaque 5x5 on the layer above -- so `player on
  target`, the composition ``All Target on Player`` is entirely made of, was
  pixel-identical to a player on bare floor. Target and AntiTarget gained the
  four CORNERS and the Players gave up exactly those four corners, which is the
  ps:stickyban / ps:swap_sokoban fix.

``--audit`` renders every cell COMPOSITION the game can show -- floor, wall,
target, antitarget, player, player-on-target, player-on-antitarget -- as a whole
64x64 frame of a uniform board, at every cell size the five levels render at
(9, 10 and 16 px), and requires them pairwise distinct. Whole frames rather than
one cell sliced out of a mixed board: `_render_frame` upscales and centre-pads,
so slicing by ``cell_px`` arithmetic reads the wrong pixels (the ps:explod
lesson). It also asserts no composition renders as the uniform pad colour (the
ps:stand_iii check, which is the one that catches the wall bug), and runs the
whole 8-element group over every composition on square boards, because the game
is in `_FLIP_GAMES`.

``player + wall`` is absent on purpose and is not an omission -- they share a
collision layer, so the engine can never put them in one cell. So are every
stack involving Poison, Life, PlayerSeed and the three non-Good Player variants,
which no level can produce.

Usage (run from the repo root):
    python solvers/generate_the_blob_training.py --episodes 200 \
        --out data/training_multi_level/the_blob

    python solvers/generate_the_blob_training.py --plans      # level report
    python solvers/generate_the_blob_training.py --selfcheck  # model + optimality
    python solvers/generate_the_blob_training.py --engine     # interpreter proof
    python solvers/generate_the_blob_training.py --audit      # rendering
    python solvers/generate_the_blob_training.py --symmetry   # augmentation
"""

from __future__ import annotations

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

GAME_NAME = "The_Blob"

#: Engine directions, in the order ties are broken. ACTION5 is bound to nothing
#: (the game declares ``noaction`` and no rule reads it), so it is not a move and
#: branching on it would inflate every sweep for nothing.
DIRS: tuple[str, ...] = ("up", "down", "left", "right")

#: ``_OPP[di]`` is the index of the opposite direction. The shear resolution
#: propagates blocking BACKWARDS along the pressed axis, which is the only place
#: this is used.
_OPP: tuple[int, ...] = (1, 0, 3, 2)

#: Objects that no shipped level places and no rule can create from a board that
#: has none of them. `_Board.read` raises if one ever appears, because the model
#: below does not implement the growth/death half of the game -- see the module
#: docstring. ``playerseed``/``babyplayer`` need a `Life` (or a seed) to exist;
#: ``dyingplayer``/``recentdyingplayer`` need a `Poison`.
_UNMODELLED: tuple[str, ...] = ("poison", "life", "playerseed", "babyplayer",
                                "dyingplayer", "recentdyingplayer")

#: Disk cache of every level's start plan AND its optimal-action sets. Level 4's
#: field is 894k states and ~2.5 s; the other four together are under 0.1 s, so
#: this exists mainly so a `parallelize_generator` fan-out derives level 4 once
#: rather than once per core. Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "the_blob_plans.json"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic, plus the search.

    A STATE is the set of blob cells, packed as a bitmask over ``r * w + c``.
    Nothing else in this game moves: walls and targets are fixed, and the
    objects that could appear and change that (Life, Poison and the babies,
    seeds and corpses they make) exist on no level -- `read` enforces it.

    THE MECHANIC. One press gives every blob cell the same force, so it is a
    translation of the whole set, minus whatever is jammed:

      * a cell whose destination is a Wall or is off the board is BLOCKED;
      * a cell whose destination holds a blocked cell is blocked too, and that
        propagates back along the pressed axis until it stops;
      * every other cell moves one square.

    Nothing can ever contest a destination -- translation by a fixed vector is
    injective, and a moving cell's destination is either free or is being
    vacated by the cell in front of it, which by the propagation rule is not
    blocked. So the result is exactly ``blocked | shift(free, d)``, with no
    ordering anywhere in it. (That is also the argument for the mirror
    augmentation; see `PuzzleScriptAdapter._FLIP_GAMES`.)

    Note the asymmetry that makes the game a puzzle: a press with NOTHING
    blocked is a pure translation and is undone by the opposite press, but a
    press that shears is NOT undoable -- the blob comes out a different shape and
    the opposite press does not put it back. So `field` has to be able to say
    "no win from here", and does; what it has never once had to say it about is
    a board any probe has reached (see Recovery in the module docstring).
    """

    __slots__ = ("h", "w", "n", "targets", "anti", "_sh", "_wb")

    def __init__(self, h: int, w: int, walls, targets, anti=()):
        self.h, self.w = h, w
        self.n = n = h * w
        all_mask = (1 << n) - 1

        col_first = sum(1 << (r * w) for r in range(h))
        col_last = sum(1 << (r * w + w - 1) for r in range(h))
        row_first = (1 << w) - 1
        row_last = row_first << ((h - 1) * w)

        #: ``_sh[di] == (keep, amount, towards_high)``: shifting ``P & keep`` by
        #: ``amount`` moves every cell one square in direction ``DIRS[di]``.
        #: ``keep`` drops the rank that would wrap around the board edge (the row
        #: or column the move leaves the board from), which is what makes the
        #: shift total -- so nothing after this ever masks or bounds-checks.
        self._sh = (
            (all_mask & ~row_first, w, False),   # up
            (all_mask & ~row_last, w, True),     # down
            (all_mask & ~col_first, 1, False),   # left
            (all_mask & ~col_last, 1, True),     # right
        )

        wall_mask = 0
        for i in walls:
            wall_mask |= 1 << i
        free = all_mask & ~wall_mask
        #: ``_wb[di]``: the cells whose neighbour in direction ``di`` is a wall or
        #: is off the board, i.e. the cells a press in that direction blocks
        #: outright. Everything else blocks only by propagation. Derived as "not
        #: the cells that CAN move there", so the board edge and the walls are one
        #: rule rather than two -- which matters because on level 0 the edge is
        #: almost the whole of the geometry.
        self._wb = tuple(all_mask & ~self._shift(free, _OPP[di])
                         for di in range(4))

        #: ``All Target on Player`` -- every target square must hold a blob cell.
        self.targets = 0
        for i in targets:
            self.targets |= 1 << i
        #: ``No AntiTarget on Player`` -- no anti-target square may. Empty on
        #: every shipped level; kept because the win condition is not.
        self.anti = 0
        for i in anti:
            self.anti |= 1 << i

    # -- the mechanic --------------------------------------------------------
    def _shift(self, mask: int, di: int) -> int:
        keep, amount, high = self._sh[di]
        return ((mask & keep) << amount) if high else ((mask & keep) >> amount)

    def step(self, state: int, di: int) -> int:
        """The board after pressing ``DIRS[di]``. Returns ``state`` unchanged when
        the press does nothing (every cell blocked), which is a non-move: no
        shortest path contains one."""
        blocked = state & self._wb[di]
        if not blocked:
            return self._shift(state, di)
        back = _OPP[di]
        while True:
            grown = blocked | (state & self._shift(blocked, back))
            if grown == blocked:
                break
            blocked = grown
        return blocked | self._shift(state & ~blocked, di)

    def won(self, state: int) -> bool:
        """``All Target on Player`` and ``No AntiTarget on Player``.

        ``Some Target`` is the third condition and is a constant: no rule in this
        game creates or destroys a Target, and every level places at least one,
        which `read` asserts."""
        return not (self.targets & ~state) and not (self.anti & state)

    # -- the search ----------------------------------------------------------
    def field(self, state: int, cap: int):
        """``(on_path, d_star)``: for every ``j``, the states exactly ``j``
        presses from ``state`` whose distance to a win is exactly ``d_star - j``.
        ``(None, None)`` if this board cannot be won, or if the sweep passed
        ``cap`` states without finding a win.

        Sweep 1 is a forward BFS kept in LAYERS, stopped at the first depth that
        produces a win. Sweep 2 walks those layers backwards: ``on_path[d_star]``
        is the winning boards found while expanding the last layer, and
        ``on_path[j]`` is every state of layer ``j`` with a successor in
        ``on_path[j + 1]``.

        Why layers instead of the backward BFS over inverse moves that this
        family's sokobans use: a shear has no enumerable inverse. Its blocked set
        is a function of the PREDECESSOR, so reconstructing the predecessor from
        the result would mean guessing which subset of the blob stayed put. The
        induction above needs no inverse and costs one extra forward sweep.

        Exactness. ``on_path[j]`` exhibits a win path of length ``d_star - j``,
        so the true distance is at most that; and it cannot be less, because a
        state at forward distance ``j`` with a win ``h`` presses away gives a
        start-to-win path of ``j + h``, which cannot beat ``d_star``. States on no
        shortest path are simply absent, and nothing ever asks about them.
        """
        if self.won(state):
            return [{state}], 0
        seen = {state}
        layers = [[state]]
        wins: set = set()
        while not wins:
            frontier = []
            for cur in layers[-1]:
                for di in range(4):
                    nxt = self.step(cur, di)
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    if self.won(nxt):
                        wins.add(nxt)          # a won board is terminal
                    else:
                        frontier.append(nxt)
            if wins:
                break
            if not frontier or len(seen) > cap:
                return None, None
            layers.append(frontier)

        d_star = len(layers)
        on_path: list = [None] * (d_star + 1)
        on_path[d_star] = wins
        for j in range(d_star - 1, -1, -1):
            ahead = on_path[j + 1]
            on_path[j] = {s for s in layers[j]
                          if any(self.step(s, di) in ahead for di in range(4))}
        return on_path, d_star

    def optimal(self, on_path, state: int, j: int):
        """``[(direction index, successor), ...]`` for every press on a shortest
        path from ``state``, which sits at forward distance ``j``. Ties come out
        in ``DIRS`` order, which is what makes a re-derived plan byte-identical
        across processes."""
        ahead = on_path[j + 1]
        out = []
        for di in range(4):
            nxt = self.step(state, di)
            if nxt in ahead:
                out.append((di, nxt))
        return out

    def solve(self, state: int, cap: int) -> "Plan | None":
        """The shortest press sequence from ``state``, with the EXACT optimal set
        at every step, or None if this board cannot be won from here (or is past
        ``cap``)."""
        on_path, d_star = self.field(state, cap)
        if on_path is None:
            return None
        presses, optsets = [], []
        cur = state
        for j in range(d_star):
            best = self.optimal(on_path, cur, j)
            if not best:                    # unreachable: on_path[j] has a step
                return None
            presses.append(DIRS[best[0][0]])
            optsets.append([DIRS[di] for di, _ in best])
            cur = best[0][1]
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class TheBlobExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Board` field.

    The base class keeps the memo, the level scoping and the disk cache; only the
    strategy underneath changes, the way `PSEnumExpert` replaces it. Here the
    strategy is "read the live board, sweep it forwards then backwards, hand back
    the shortest plan and its exact tie sets", so `heuristic` is never called and
    asserts rather than returning a number nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state (a recovery prefix, an epsilon detour) is answered exactly --
    including "this board is now dead", which an exploration prefix really can
    produce here, presses being irreversible the moment a shear happens.

    `_key` is INHERITED (every non-background cell), not narrowed to the blob:
    that key is canonical across levels because the walls and targets are in it
    too, so no ``scope_by_level`` is needed and the ``plan_cache_path``
    signature covers the whole board -- an edited level is then a cache MISS
    rather than a wrong plan.
    """

    directions = list(DIRS)
    plan_cache_path = PLAN_CACHE

    #: Runaway guard on `_Board.field`: give up past this many states rather than
    #: exhaust RAM. Sized off the largest field any level START needs (level 4's
    #: 893,835) with room to spare, because generation only ever plans from a
    #: level start -- the RESET-mode recovery prefix ends by restoring it. It is
    #: `--selfcheck`'s random boards that can wander somewhere expensive.
    state_cap: int = 2_000_000

    def setup(self) -> None:
        self.player_ids = set(self.game._engine._player_indices)
        self.wall_ids = set(self.g.resolve_object_name("wall"))
        self.target_ids = set(self.g.resolve_object_name("target"))
        self.anti_ids = set(self.g.resolve_object_name("antitarget"))
        self.unmodelled_ids = {i for name in _UNMODELLED
                               for i in self.g.resolve_object_name(name)}
        #: `_Board`s by STATIC signature (see `read`), not by level index.
        self._boards: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "TheBlobExpert reads an exact distance field; heuristic is unused")

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, state)`` for the engine's current grid.

        Boards are cached by their STATIC signature (dimensions, walls, targets,
        anti-targets), not by level index, so the sweeps never rebuild the shift
        tables and one board is shared by every state of its level.

        Raises if the board holds any object this model does not implement. That
        can only happen if a level is edited to place a `Life` or a `Poison`
        (nothing in the game creates one otherwise), and it is a raise rather
        than a ``return None`` on purpose: None reads downstream as "this level
        is unsolvable, skip it", which would silently ship a smaller corpus for
        what is really an out-of-date model."""
        h, w = eng.height, eng.width
        walls, targets, anti = [], [], []
        state = 0
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                if cell & self.unmodelled_ids:
                    names = sorted(self.g.obj_idx_to_name[o]
                                   for o in (cell & self.unmodelled_ids))
                    raise AssertionError(
                        f"The_Blob: {names} at ({r},{c}) -- the growth/death "
                        "half of this game is not modelled and no shipped level "
                        "uses it (see the module docstring)")
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.target_ids:
                    targets.append(i)
                if cell & self.anti_ids:
                    anti.append(i)
                if cell & self.player_ids:
                    state |= 1 << i
        assert targets, "The_Blob: 'Some Target' is unsatisfiable on this board"
        sig = (h, w, tuple(walls), tuple(targets), tuple(anti))
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(h, w, walls, targets, anti)
        return board, state

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if not state:                        # no blob left on the board
            return None
        if board.won(state):
            return Plan([], [])
        return board.solve(state, self.state_cap)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class TheBlobSolver(PSAStarSolver):
    game_id = "puzzlescript_the_blob"
    game_name = GAME_NAME
    expert_cls = TheBlobExpert

    #: `games/ps:the_blob/ps:the_blob.py` is a plain passthrough (it constructs
    #: the adapter and nothing else), so there is nothing to gain by routing
    #: through it -- but it IS what a live agent is handed, so if that wrapper
    #: ever grows a patch this must be set to ``"ps:the_blob"``.
    game_module_id = ""

    #: Unused: the expert enumerates an exact field rather than searching under a
    #: heuristic, so there is no weight to trade and no node budget that could
    #: quietly bite. Left at the base values so nothing reads a lie off them --
    #: the cap that IS real is `TheBlobExpert.state_cap`.
    weight = 1
    node_cap = 400_000

    #: The longest plan is 18 presses (level 2); the rest is room for a re-plan
    #: after the exploration prefix. Stays well under the adapter's own 200-step
    #: per-level budget, which `set_level` resets before the plan starts anyway.
    max_steps = 80

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it fills the disk cache in one pass and makes level
        4's 2.5 s field visible as startup rather than as a mysterious pause."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new():
    solver = TheBlobSolver()
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _brute(board, state, limit):
    """Presses to a win from ``state``, searched fresh by forward BFS, or None
    past ``limit``. Shares no code with `_Board.field`'s backward half, which is
    the point of it."""
    if board.won(state):
        return 0
    seen = {state}
    queue = deque([(state, 0)])
    while queue:
        cur, d = queue.popleft()
        if d >= limit:
            continue
        for di in range(4):
            nxt = board.step(cur, di)
            if nxt == cur or nxt in seen:
                continue
            if board.won(nxt):
                return d + 1
            seen.add(nxt)
            queue.append((nxt, d + 1))
    return None


def _plans() -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the field and the tie labelling at
    once."""
    _solver, game, expert, solvable = _new()
    total = ties = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board, state = expert.read(eng)
        blob = bin(state).count("1")
        targets = bin(board.targets).count("1")
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
        print(f"  L{level:2d}: {eng.height:2d}x{eng.width:2d}  {blob} blob cells "
              f"/ {targets} targets  {len(plan):3d} presses  win={won}  "
              f"(budget {game._max_steps}, {room})  "
              f"{tie_steps:3d} steps with a tie set")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _selfcheck(walk_presses: int = 400, samples: int = 120,
               verbose: bool = True) -> int:
    """Four things the expert would otherwise be trusted on.

    1. **The native model is the interpreter's.** A seeded random walk on every
       level, comparing `_Board.step`'s bitmask against the engine's grid after
       EVERY press -- including the presses that do nothing, which is where a
       collision-layer mistake hides, and including ACTION5, which must be a
       no-op. The same walk re-checks after every press that none of the
       unmodelled objects (Life, Poison, seeds, babies, corpses) has appeared,
       which is the claim the whole model rests on.
    2. **The plans are shortest.** A plain forward BFS to the first win, which is
       shortest by construction and shares no code with the field's backward
       half, must return the field's plan length on every level -- level 4
       included, which is the level ``--engine`` cannot reach.
    3. **The tie sets are exact.** Every step's optimal set is re-derived by
       brute force (a fresh depth-bounded BFS from each of the four successors)
       and must match the field's answer exactly.
    4. **Recovery is answered from ANY board, not just the start.** This is the
       one that matters at generation time: presses here are irreversible the
       moment a shear happens, so `_search` has to be right about arbitrary
       states in both directions. From ``samples`` boards reached by seeded
       random presses, the field must either return a plan that WINS when
       replayed through the interpreter, or return None on a board an
       independent forward BFS also finds no win from. The measured answer is
       that the "dead" column stays at zero -- see Recovery in the module
       docstring -- but a None that is merely a give-up would be a silent bug:
       `record_level` reads it as "reset", and a wrong plan length here would not
       show up in any of the three checks above, all of which only ever ask about
       starts.
    """
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    bad = 0

    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        rng = random.Random(f"the_blob:selfcheck:{level}")
        drift = 0
        for _ in range(walk_presses):
            direction = rng.choice(DIRS + ("action",))
            eng.step(direction)
            if direction != "action":
                state = board.step(state, DIRS.index(direction))
            _b, live = expert.read(eng)      # also re-asserts nothing unmodelled
            if live != state:
                drift += 1
                break
        bad += drift
        if verbose:
            print(f"  L{level:2d}: model {'MATCHES' if not drift else 'DIVERGES from'}"
                  f" the interpreter over {walk_presses} random presses")

    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        plan = expert.plan(eng, level)
        shortest = _brute(board, state, board.n)
        ok = plan is not None and shortest == len(plan)
        bad += not ok
        if verbose:
            print(f"  L{level:2d}: field {len(plan) if plan else None} presses vs "
                  f"BFS {shortest} -- {'SHORTEST' if ok else 'NOT SHORTEST'}")

    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        plan = expert.plan(eng, level)
        sets = getattr(plan, "optsets", None) or []
        mismatched = 0
        cur = state
        for i, direction in enumerate(plan):
            rest = len(plan) - i
            truth = []
            for di in range(4):
                nxt = board.step(cur, di)
                if nxt == cur:
                    continue
                if _brute(board, nxt, rest - 1) == rest - 1:
                    truth.append(DIRS[di])
            if sorted(truth) != sorted(sets[i]):
                mismatched += 1
            cur = board.step(cur, DIRS.index(direction))
        bad += mismatched
        if verbose:
            print(f"  L{level:2d}: {len(plan)} tie sets "
                  f"{'match' if not mismatched else f'{mismatched} MISMATCH'}"
                  f" brute force")

    # 4 -- recovery from arbitrary boards, checked against the interpreter.
    for level in range(game.n_levels):
        rng = random.Random(f"the_blob:recovery:{level}")
        wrong = dead = won_after = 0
        for _ in range(samples):
            game.set_level(level)
            for _ in range(rng.randrange(1, 25)):
                eng.step(DIRS[rng.randrange(4)])
                if eng.check_win():
                    break
            if eng.check_win():
                continue                 # the walk already won; nothing to plan
            board, state = expert.read(eng)
            plan = expert._search(eng)
            # Independent: an unbounded forward BFS over the whole reachable
            # space, which either finds a win or PROVES there is none. Bounded by
            # the cell count only so it terminates; every plan here is far under.
            truth = _brute(board, state, board.n)
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
    return bad


#: Levels ``--engine`` enumerates. Level 4 is excluded and only from this pass:
#: 18,531,158 reachable engine states at ~20k interpreter steps per second is a
#: fifteen-minute run. Its plan is covered end to end by ``--plans`` (replayed
#: through the real interpreter to a WIN) and its shortestness by
#: ``--selfcheck``'s independent forward BFS.
_ENGINE_LEVELS: tuple[int, ...] = (0, 1, 2, 3)


def _engine(verbose: bool = True) -> int:
    """The proof that owes nothing to the native model: enumerate the REAL
    interpreter.

    `StateGraph.build` walks every state reachable from the level start by
    pressing actual buttons on the engine and keying the resulting grid, then
    solves the graph exactly. It shares no line of code with `_Board` -- no
    modelled shear, no modelled collision layer, no modelled win test -- so
    agreement on both the plan LENGTH and the per-step optimal SET is an
    independent derivation of everything this file claims about these levels.

    It is affordable because the key DECODES. `StateGraph.build`'s default is to
    snapshot every state (a fresh ``set`` per cell), which on level 3's 86,660
    states is gigabytes; here a state is the blob's cells and the rest of the
    board is fixed, so a state is re-seated by writing the level's static grid
    and dropping GoodPlayers onto the bits. That is exact rather than lossy
    precisely because the growth/death objects cannot exist -- the same fact the
    whole model rests on, and `expert.read` raises if it is ever false.
    """
    from solvers.common.ps_astar import StateGraph, restore, snapshot  # noqa: PLC0415

    _solver, game, expert, _solvable = _new()
    eng = game._engine
    good = game._game.obj_name_to_idx["goodplayer"]
    bad = 0
    for level in _ENGINE_LEVELS:
        game.set_level(level)
        plan = expert.plan(eng, level)

        game.set_level(level)
        w = eng.width
        # The level with the blob lifted off it: every state is this plus bits.
        static = [[cell - expert.player_ids for cell in row] for row in eng.grid]

        def key_fn(e, _w=w):
            k = 0
            for r, row in enumerate(e.grid):
                for c, cell in enumerate(row):
                    if cell & expert.player_ids:
                        k |= 1 << (r * _w + c)
            return k

        def decode(e, key, _static=static, _w=w):
            e.grid = [[set(cell) for cell in row] for row in _static]
            i = 0
            while key:
                if key & 1:
                    e.grid[i // _w][i % _w].add(good)
                key >>= 1
                i += 1
            e._position_index_dirty = True
            e._rule_noop_cache.clear()

        # The decoder is only sound if it is key_fn's inverse; check it against a
        # snapshotted round trip on the level start before trusting 86k of them.
        snap = snapshot(eng)
        decode(eng, key_fn(eng))
        round_trip = eng.grid == snap
        restore(eng, snap)
        if not round_trip:
            print(f"  L{level:2d}: the decoder is NOT key_fn's inverse")
            bad += 1
            continue

        graph = StateGraph.build(eng, key_fn, list(DIRS),
                                 node_cap=200_000, decode=decode)
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
    skipped = sorted(set(range(game.n_levels)) - set(_ENGINE_LEVELS))
    if verbose and skipped:
        print(f"  levels {skipped} not enumerated (too large -- see "
              f"_ENGINE_LEVELS)")
    return bad


def _audit(verbose: bool = True) -> int:
    """Three passes over every reachable cell COMPOSITION.

    **Pass 1 -- distinctness**, at every cell size the five boards use (9, 10 and
    16 px; the size list is derived rather than assumed so an edited level is
    covered). Whole 64x64 frames of uniform boards are compared rather than one
    cell out of a mixed board: `_render_frame` upscales and centre-pads, so
    slicing a cell by ``cell_px`` arithmetic reads the wrong pixels (the
    ps:explod lesson). Two uniform boards render identically iff their cells do.

    **Pass 2 -- the letterbox.** No composition may render as the uniform pad
    colour, because a board's border is painted with it and there would then be
    no seam between the terrain and the edge of the picture. This is the
    ps:stand_iii check, and it is the one that catches this game's ``Wall``
    bug -- a pairwise matrix passes it every time, the pad not being a
    composition.

    **Pass 3 -- the group.** The game is in `_FLIP_GAMES`, so a board is drawn at
    any of the eight turns and mirrors. Every composition here happens to be
    invariant under all eight (solid blocks, a ring plus corners, a block minus
    corners), which this reports rather than assumes; what it requires is the
    weaker and load-bearing thing, that no transform of one composition is
    another's art. It runs on SQUARE boards, because a rotation of a non-square
    board also moves the letterbox and every comparison would pass for the wrong
    reason.

    ``player + wall`` is absent on purpose and is not an omission: Player and
    Wall share a collision layer, so the engine can never put them in one cell.
    So is every stack involving Poison, Life, PlayerSeed and the three non-Good
    Player variants -- no level places the first two and nothing creates the
    rest, which is what `TheBlobExpert.read` enforces."""
    from adapters.puzzlescript_adapter import _render_frame            # noqa: PLC0415

    _solver, game, _expert, _solvable = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = [(), ("wall",), ("target",), ("antitarget",), ("goodplayer",),
             ("goodplayer", "target"), ("goodplayer", "antitarget")]
    pad = 5

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
        blank = [c for c in comps if not (shots[c] != pad).any()]
        bad += len(clashes) + len(blank)
        if verbose:
            print(f"  {h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): {len(shots)} "
                  f"compositions, {'OK' if not clashes and not blank else 'BAD'}")
        for a, b in clashes:
            print(f"      IDENTICAL: {name(a)}  ==  {name(b)}")
        for c in blank:
            print(f"      {name(c)} renders as the uniform letterbox colour")

    def transform(frame, k, hflip, vflip):
        if k:
            frame = np.rot90(frame, k=k)
        if hflip:
            frame = np.fliplr(frame)
        if vflip:
            frame = np.flipud(frame)
        return np.ascontiguousarray(frame)

    for cell in sorted({min(64 // h, 64 // w) for h, w in sizes}):
        n = 64 // cell
        shots = {c: shoot(n, n, c) for c in comps}
        group = [(k, hf, vf) for k in range(4)
                 for hf in (False, True) for vf in (False, True)]
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
    and require both halves of the augmentation contract.

    * The ENGINE GRID must be IDENTICAL at all 16 presentations. This game has no
      in-engine rotation mechanic, so driving the same engine directions (via
      `screen_action`, which pre-inverts the adapter's own remap) must produce
      the same board whatever the frame is doing. A failure here would mean the
      mechanic is not symmetric -- which is the claim `_FLIP_GAMES` membership
      rests on.
    * The FRAMES must be exactly the transform of the unaugmented ones, which is
      the claim the ART has to satisfy.

    Both the PLANS and a seeded random walk are replayed: the walk reaches the
    boards a plan never visits (the blob jammed flat against each wall and each
    edge, sheared into shapes no shortest path produces) and it presses the
    unbound ACTION key as well."""
    solver, game, expert, _solvable = _new()
    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        plans[level] = expert.plan(game._engine, level)

    def grid_key(g):
        return tuple(tuple(frozenset(cell) for cell in row)
                     for row in g._engine.grid)

    def drive(g, level, presses):
        g.set_level(level)
        frames = [np.asarray(g._current_frame)]
        grids = [grid_key(g)]
        for act in presses:
            fd = g.perform_action(ActionInput(id=act))
            frames.append(np.asarray(fd.frame[-1] if fd.frame
                                     else g._current_frame))
            grids.append(grid_key(g))
        return frames, grids

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
            rng = random.Random(f"the_blob:symmetry:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(walk_presses)]
            for tag, presses in (
                    ("plan", [screen_action(d, *k) for d in plans[level]]),
                    ("walk", [inverse_remap_action_full(a, *k) for a in walk])):
                frames, grids = drive(g, level, presses)
                if tag == "plan" and g._state != GameState.WIN:
                    print(f"    seed {seed} L{level} {k}: plan did not win")
                    bad += 1
                key = (level, tag)
                if key not in ref:
                    if k != (0, False, False):
                        continue          # nothing to compare against yet
                    ref[key] = (frames, grids)
                    continue
                rframes, rgrids = ref[key]
                if rgrids != grids:
                    print(f"    seed {seed} L{level} {k}: {tag} engine grids "
                          f"differ from the unaugmented run")
                    bad += 1
                if any(not np.array_equal(transform(a, *k), b)
                       for a, b in zip(rframes, frames)):
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
    if "--engine" in sys.argv:
        violations = _engine()
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
    sys.exit(TheBlobSolver.main())
