"""Generate Phase-1 training data for the PuzzleScript game
ps:sokubunny_and_the_colored_boxes ("Sokubunny and the colored Boxes",
Lucas Boedeker).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, a native model of them, the exact distance FIELD that model is
searched with, the proof that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_sokubunny_and_the_colored_boxes",
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
The canonical PuzzleScript sokoban with the crates TYPED. The whole mechanic is
still one rule,

    [ >  Player | Crate ] -> [  >  Player | > Crate  ]

where ``Crate = RedCrate or BlueCrate``, and the collision layers are the
orthodox

    Background
    Target                 (RedTarget or BlueTarget)
    Player, Wall, Crate

so crates and walls block, targets block nothing, and there is no chain push
(the rule names ``Player`` on its left, so a moving crate never re-triggers it).
Everything `ps:sokoban_sanity`'s docstring says about the mechanic holds here
word for word -- this is that game plus a colour on each piece.

What the colour changes is the WIN, and it changes it into a conjunction:

    All RedTarget on RedCrate
    All BlueTarget on BlueCrate

A red crate parked on a blue target is not a partial win, it is a mistake, and
frequently a fatal one: these boards are built as "the box that has to end up
here has to be got past the mark it must NOT end up on". The author's own level
names say so -- six of the twenty-three are called some form of *stashaway*
(park one colour out of the way, walk the other one home, come back for it), and
another is "there is only one way". That is what makes them long for their size: 23 boards between 4x5 and 8x8, all 23
solvable, none already won at reset, and the shortest answers run 12 to 52
presses (652 in total, the longest being level 2's 52 on a 7x8 room).

The typing has one more consequence that a plain sokoban does not have. On a
plain board an unmatched crate is slack -- any crate can cover any target -- so
a solver may permute them freely. Here the pieces are partitioned, and the
model's state has to carry the partition: ``(player, red crates, blue crates)``
with the two crate sets sorted INDEPENDENTLY. Sorting them together (or keying
on "crates" and a colour map) is the bug this game invites, and it does not
crash -- it silently identifies a won board with a lost one.

Every shipped level has exactly as many crates as targets of each colour (one
red pair on twenty of them, two red pairs on levels 4, 16, 17 and 18; one blue
pair on all twenty-three), so a win is the crates standing on precisely their
own colour's target set -- but `_Board.won` is written as the two SUBSET tests
the win condition literally states, so an edited level with a crate to spare
would still be answered correctly.

Level 2 starts with a red crate already sitting on a blue target (the ``@``
legend character), which is the clearest statement of the game there is: that
square is occupied, it is not satisfied, and getting it right means pushing the
red crate OFF a mark before anything else can happen.

Everything above was MEASURED against the interpreter rather than read off the
.txt; ``--selfcheck`` is the executable form of it (a seeded random walk on every
level, the model's board compared against the engine's grid after every single
press, including the presses that do nothing -- which is where a collision-layer
mistake hides). The measurements the walk covers on the shipped boards: a step
onto a target of either colour (the player passes over both), a push of either
crate onto either target (it slides on, and the level may end there or may
not -- which is the typed part), a push into a wall or off the board (nothing
moves, player included), and a push into a second crate of either colour
(likewise nothing).

Six of the twenty-three boards (levels 14-18 and 22) have NO wall ring: their
boundary is partly the board EDGE, which is geometry the .txt does not draw. The
model treats an off-board neighbour exactly as a wall -- a push that would send a
crate off the board moves nothing, player included -- which is what the
interpreter does, and the random walk hammers it.

Expert solver
-------------
A NATIVE model of the mechanic above (`_Board`), searched to an exact distance
field rather than heuristically. This is `ps:sokoban_sanity`'s `_Board` with the
crate set split in two, and it is affordable for the same reason: the whole game
is small. The 23 forward sets together are 80614 states and deriving all 23
fields from cold takes 0.28s.

`_Board.solve` is two breadth-first sweeps over ``(player, red, blue)`` states:

  1. FORWARD from the state being planned, stopping at the depth ``d*`` of the
     first win. That fixes the answer's length and collects ``F``, every state
     within ``d*`` presses of the start.
  2. BACKWARD from the winning configurations over INVERTED moves (a walk is
     undone by stepping back onto a free cell; a push is undone by stepping back
     and dragging the crate one cell towards you -- and the drag has to preserve
     the crate's COLOUR, which is the only line of `preds` that is not
     sokoban_sanity's), pruned to ``F``.

The inverse moves are enumerated directly, so the backward sweep needs no
reverse adjacency table.

Pruning the backward sweep to ``F`` costs nothing that matters: if a state ``s``
lies on any shortest start-to-win path then its whole continuation has forward
distance at most ``d*``, so that continuation never leaves ``F`` and the field's
value at ``s`` is its true distance. States off every shortest path may be
over-estimated, and none of them is ever asked about.

Optimal-action sets
-------------------
The field gives them EXACTLY, with no inference: at a state ``d`` presses from a
win, the optimal presses are every direction whose successor is ``d - 1`` from
one.

The measured answer is that this game is nearly all forced: 17 of its 652 expert
steps have a second equally-right press, and 14 of the 23 levels have none at
all. That is what a board of corridors and pockets with two crates that must not
be interchanged looks like -- almost every press is either the one push that
works or the one step towards it.

``--selfcheck`` re-derives every set on every level by brute force (a fresh
depth-bounded BFS from each of the four successors) and requires an exact match;
``--engine`` re-derives them again from the interpreter. No step ever ships
unlabelled (the always-emit-optimal-targets rule).

Shortest, and how that is known
-------------------------------
All 23 plans are provably shortest, and the proof is three independent
derivations agreeing:

  * this file's field (forward BFS + backward BFS);
  * a plain forward BFS to the first win, which is shortest by construction and
    shares no code with the field's backward half (``--selfcheck``);
  * a full-space enumeration of the real INTERPRETER -- `StateGraph.build`
    walking every reachable engine state by pressing real buttons, with no
    native model involved at all (``--engine``: 133078 states over the 23
    levels, the biggest being level 18's 35793). That is the derivation that
    closes the loop back to the engine, and it agrees on all 23 lengths AND on
    all 652 tie sets.

This game is small enough to afford that third pass on EVERY level, which the
bigger ps: sokobans are not -- `Sokoban_Flipped`'s level 6 alone passes 4 million
states -- so here the engine-side proof is exhaustive rather than a spot check.
The native model is separately fuzz-verified against the interpreter, which is
what makes the first two derivations statements about this game rather than about
a model of it.

Why this game gets its OWN expert rather than two attribute lines on the shared
`PSSokobanExpert` -- and the answer is not "that would not work". ``--macro`` runs
it and it WINS all 23 levels, because its win test is the engine's. It is that its
heuristic is colour-BLIND in exactly the place this game is not: one push-distance
table per target, greedily matched to "a free piece", with no notion that a red
crate cannot satisfy a blue mark. So it happily scores a board as nearly done when
the two crates are sitting on each other's targets, and it pays for that on 7 of
the 23 levels, 17 presses of slack in total (4 each on levels 0 and 9). It also
ships no exact tie sets: with ``annotate = True`` its inferred sets come from a
rule of thumb, and the field has the real answer for free.

Recovery
--------
`supports_recovery` with RESET-mode recovery. The expert re-plans from the LIVE
board, not from a stored path: `_search` reads whatever the engine currently
holds and runs both sweeps from there, so an exploration prefix that leaves the
board anywhere -- including somewhere a shortest plan would never go -- is
answered exactly. A prefix CAN strand this game for good (shove a crate into a
corner and the level is dead), and the field says so by returning None, which is
exactly what `record_level` needs to hear to fall back to a RESET.

That is not a rare corner here, and it is a good deal commoner than on the
untyped sokobans, because a board can also be stranded WITHOUT any crate being
stuck: shove the red crate onto the only blue target's square-and-a-half of
corridor and the blue crate can never get home. ``--selfcheck``'s fourth pass
walks random boards on every level and checks both halves against the
interpreter -- every plan it returns from a random board is replayed and must
WIN, and every None must be a board an independent forward BFS also proves dead,
never a give-up.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`), which together re-sample the board's
8-element symmetry group. 23 levels x 16 presentations = 368.

The structural argument for the flips is `ps:sokoban_sanity`'s, unchanged --
there is exactly ONE rule, it is written with the relative force ``>``, it names
no axis, one press moves at most two bodies and they land on ``p + d`` and
``p + 2d``, which can never be the same square, so no two moves can contest a
cell and the order the interpreter expands the rule's four directions cannot
decide anything. Both win conditions are positional and neither names a
direction. The crate typing is a colour, and a colour is invariant under every
turn and mirror. ``--symmetry`` measures it anyway: every level's plan AND a
seeded random walk (which does what a plan never does -- shoves crates into
walls, into each other and into corners, parks the wrong colour on a target, and
presses the unbound ACTION key) replayed at all 16 presentations, requiring every
frame to be the exact transform of the unaugmented one.

The art survives the group too. Floor, both targets, both crates and every
crate-on-target stack are invariant under all eight turns and mirrors -- a flat
field, a lattice-and-block, a border, and a border round a lattice-and-block.
Wall's brown/darkbrown weave and the little bunny (left-right symmetric, but a
figure, so not top-bottom) are NOT, so a mirrored board does show art the .txt
does not literally contain. That is only a problem if the mirrored art of one
composition is another composition's art, and it is not: ``--audit``'s second
pass turns and mirrors every composition eight ways and compares it against every
other.

There is deliberately no colour augmentation. In this game the colour IS the
win condition -- red belongs on red, blue on blue -- so a recolor that flattened
or permuted the two would either erase the rule or teach it wrong.

Rendering
---------
TWO sprites had to change, and they are the ones the win condition is made of.
See the header comment in
``data/puzzlescript_games/Sokubunny_and_the_colored_Boxes.txt`` for the full
statement; in short, both targets shipped as a `black` ring on rows/cols 1-3
carrying their colour on the SINGLE centre pixel, and the Player sprite is opaque
across all of that ring except the one cell (1,2) -- which is `black` in both.
So "player on a red target" was PIXEL-IDENTICAL to "player on a blue target" at
every one of the six cell sizes these boards render at, i.e. the single
distinction the entire game is about was unreadable whenever the bunny stood on a
mark. Both targets are now a SOLID 3x3 in their own colour surrounded by a 3x3
LATTICE (the corners and the edge midpoints): solid so the crate's open middle
reports the target's colour, a lattice outside it because rows/cols {0,2,4} are
exactly where the player sprite is transparent.

Deliberately NOT the solid 5x5 that `ps:sokoban_sanity` uses. There the crate is
a different colour from the target; here RedCrate is the same `red` as RedTarget,
so a solid tile under a same-colour crate ring would render as one uniform block
and ``redcrate + redtarget`` -- the winning composition -- would be
pixel-identical to ``redtarget`` alone. The eight holes the lattice leaves are
what the crate's border fills in, and they are the whole difference between "goal"
and "goal satisfied".

``--audit`` renders every cell COMPOSITION the game can show -- floor, wall, each
target, each crate, each of the FOUR crate-on-target stacks (two of which are
wins in the making and two of which are mistakes), the player, and the player on
each target -- as a whole 64x64 frame of a uniform board, at every cell size the
23 levels render at (7, 8, 9, 10 and 12 px), and requires them pairwise distinct.
Whole frames rather than one cell sliced out of a mixed board: `_render_frame`
upscales and centre-pads, so slicing by ``cell_px`` arithmetic reads the wrong
pixels (the ps:explod lesson). ``player + crate`` and ``redtarget + bluetarget``
are absent on purpose and are not omissions -- each pair shares a collision
layer, so the engine can never put either in one cell.

Usage (run from the repo root):
    python solvers/generate_sokubunny_training.py --episodes 200 \
        --out data/training_multi_level/sokubunny_and_the_colored_boxes

    python solvers/generate_sokubunny_training.py --plans      # level report
    python solvers/generate_sokubunny_training.py --selfcheck  # model + optimality
    python solvers/generate_sokubunny_training.py --engine     # interpreter proof
    python solvers/generate_sokubunny_training.py --macro      # the colour-blind A*
    python solvers/generate_sokubunny_training.py --audit      # rendering
    python solvers/generate_sokubunny_training.py --symmetry   # augmentation
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

GAME_NAME = "Sokubunny_and_the_colored_Boxes"

#: Engine directions, in the order ties are broken. ACTION5 is bound to nothing
#: in this game (no rule has ``action`` on its left-hand side), so it is not a
#: move and branching on it would double the search for nothing. The exploration
#: prefix still presses it -- a live agent has that button -- which is why
#: ``--symmetry``'s random walk presses it too.
DIRS: tuple[str, ...] = ("up", "down", "left", "right")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Disk cache of every level's start plan AND its optimal-action sets. The 23
#: boards together cost about a quarter of a second, so this is not the
#: load-bearing cache it is on the bigger ps: sokobans -- it is here so a
#: `parallelize_generator` fan-out shares one derivation rather than repeating it
#: on every core. Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "sokubunny_plans.json"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic, plus the search.

    Cells are flat ``r * w + c`` indices and a STATE is

        (player, sorted tuple of RED crates, sorted tuple of BLUE crates)

    -- the only things any rule in this game moves, with the two crate sets kept
    APART. That separation is the whole difference from a plain sokoban: the
    win is a conjunction of two per-colour subset tests, so a state that merged
    the crates (or sorted them together) would identify boards the win condition
    distinguishes. Walls and both target sets never change, so they live here
    rather than in the state.

    The mechanic is the single rule ``[> Player | Crate] -> [> Player | > Crate]``
    with ``Crate = RedCrate or BlueCrate``, resolved under the collision layers:

      * step onto a wall, or off the board -> nothing happens;
      * step onto a crate of EITHER colour -> it slides one further, keeping its
                                              colour, UNLESS that cell is a wall,
                                              is off the board or holds another
                                              crate of either colour, in which
                                              case nothing happens at all (the
                                              rule's RHS moves both bodies or
                                              neither, so the player's own move
                                              is cancelled with the crate's);
      * anything else                      -> the player moves.

    Targets appear in exactly one place -- the win test -- because they are alone
    on their own collision layer and block nothing. A crate standing on the WRONG
    colour's target is in every mechanical respect an ordinary crate on ordinary
    floor; it is only the win test that cares, which is why the mistake is so
    easy to make and so expensive to undo.
    """

    __slots__ = ("h", "w", "wall", "red_targets", "blue_targets", "_edge")

    def __init__(self, h: int, w: int, walls, red_targets, blue_targets):
        self.h, self.w = h, w
        self.wall = bytearray(h * w)
        for i in walls:
            self.wall[i] = 1
        #: The two win sets. ``All RedTarget on RedCrate`` reads as "every red
        #: target square has a red crate on it", so each is a SUBSET test, not an
        #: equality one -- the same thing on every shipped board (crates ==
        #: targets per colour) and still correct on an edited one with a crate to
        #: spare.
        self.red_targets = frozenset(red_targets)
        self.blue_targets = frozenset(blue_targets)
        #: ``_edge[cell][d]`` is the neighbour cell, or -1 off the board. Built
        #: once so the inner loops never do bounds arithmetic. On the six boards
        #: with no wall ring (levels 14-18 and 22) the board edge is the only
        #: geometry there is, so this table is most of their map.
        self._edge = []
        for i in range(h * w):
            r, c = divmod(i, w)
            row = []
            for d in DIRS:
                dr, dc = _DELTA[d]
                nr, nc = r + dr, c + dc
                row.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1)
            self._edge.append(tuple(row))

    # -- the mechanic --------------------------------------------------------
    def step(self, state, di: int):
        """The state after pressing ``DIRS[di]``, or ``state`` if the press does
        nothing (which is a non-move: no shortest path contains one)."""
        p, red, blue = state
        n1 = self._edge[p][di]
        if n1 < 0 or self.wall[n1]:
            return state
        red_hit = n1 in red
        if red_hit or n1 in blue:
            n2 = self._edge[n1][di]
            if n2 < 0 or self.wall[n2] or n2 in red or n2 in blue:
                return state
            if red_hit:
                s = set(red)
                s.discard(n1)
                s.add(n2)
                return (n1, tuple(sorted(s)), blue)
            s = set(blue)
            s.discard(n1)
            s.add(n2)
            return (n1, red, tuple(sorted(s)))
        return (n1, red, blue)

    def preds(self, state):
        """Every state one press BEFORE ``state``.

        A walk is undone by stepping the player back onto ``p - d``, which has to
        be on the board, not a wall and not a crate of either colour (it was free
        when the player left it). A push is undone the same way with the crate
        dragged one cell towards the player: the crate now at ``p + d`` was at
        ``p``, so the earlier board is that colour's set minus ``p + d`` plus
        ``p`` -- the OTHER colour's set is carried through untouched, which is the
        one line here that a plain sokoban does not have. ``p - d`` has to be free
        of THAT board, not of this one.

        Enumerating the inverses directly is what lets the backward sweep run
        without a reverse adjacency table."""
        p, red, blue = state
        out = []
        for di in range(4):
            dr, dc = _DELTA[DIRS[di]]
            r, c = divmod(p, self.w)
            qr, qc = r - dr, c - dc
            if not (0 <= qr < self.h and 0 <= qc < self.w):
                continue
            q = qr * self.w + qc
            if self.wall[q]:
                continue
            occupied = q in red or q in blue
            if not occupied:
                out.append((q, red, blue))                # undo a walk
            n2 = self._edge[p][di]
            if n2 < 0:
                continue
            if n2 in red:                                 # undo a RED push
                s = set(red)
                s.discard(n2)
                s.add(p)
                if q not in s and q not in blue:
                    out.append((q, tuple(sorted(s)), blue))
            elif n2 in blue:                              # undo a BLUE push
                s = set(blue)
                s.discard(n2)
                s.add(p)
                if q not in s and q not in red:
                    out.append((q, red, tuple(sorted(s))))
        return out

    def won(self, state) -> bool:
        """Both win conditions, which is a conjunction and not a count: it is
        entirely possible for every target to be covered and the level to be
        lost, and levels 4, 16, 17 and 18 all have shortest answers that walk
        through exactly that board."""
        return (self.red_targets <= set(state[1])
                and self.blue_targets <= set(state[2]))

    # -- the search ----------------------------------------------------------
    def field(self, state):
        """``(dist, d_star)``: presses-to-win for every state on a shortest path
        from ``state``, and the length of that path. ``(None, None)`` if this
        board can never be won from here.

        Sweep 1 is a forward BFS stopped at the depth of the first win, which
        both fixes ``d_star`` and collects ``F`` (every state within ``d_star``
        presses of the start). Sweep 2 is a backward BFS from the won boards over
        `preds`, pruned to ``F``.

        The prune is exact where it is read. If ``s`` lies on a shortest
        start-to-win path then everything after it on that path has forward
        distance at most ``d_star`` and so is in ``F``, hence the backward sweep
        finds ``s``'s true distance. States on no shortest path can come out too
        high; nothing asks about those, because `solve` and `optimal` only ever
        compare against ``d_star - i``.
        """
        if self.won(state):
            return {state: 0}, 0
        seen = {state: 0}
        queue = deque([state])
        d_star = None
        while queue:
            cur = queue.popleft()
            d = seen[cur]
            if d_star is not None and d >= d_star:
                break
            for di in range(4):
                nxt = self.step(cur, di)
                if nxt == cur or nxt in seen:
                    continue
                seen[nxt] = d + 1
                if self.won(nxt):
                    if d_star is None:
                        d_star = d + 1
                    continue                # a won board is terminal
                queue.append(nxt)
        if d_star is None:
            return None, None

        dist = {}
        queue = deque()
        for s in seen:
            if self.won(s):
                dist[s] = 0
                queue.append(s)
        while queue:
            cur = queue.popleft()
            d = dist[cur]
            for prev in self.preds(cur):
                if prev in dist or prev not in seen:
                    continue
                dist[prev] = d + 1
                queue.append(prev)
        return dist, d_star

    def optimal(self, dist, state):
        """``[(direction index, successor), ...]`` for every press on a shortest
        path from ``state``. Ties come out in ``DIRS`` order, which is what makes
        a re-derived plan byte-identical across processes."""
        rest = dist.get(state)
        if not rest:
            return []
        out = []
        for di in range(4):
            nxt = self.step(state, di)
            if nxt != state and dist.get(nxt, -1) == rest - 1:
                out.append((di, nxt))
        return out

    def solve(self, state) -> "Plan | None":
        """The shortest press sequence from ``state``, with the EXACT optimal set
        at every step, or None if this board cannot be won from here."""
        dist, d_star = self.field(state)
        if dist is None:
            return None
        presses, optsets = [], []
        cur = state
        for _ in range(d_star):
            best = self.optimal(dist, cur)
            if not best:                    # unreachable: dist[cur] > 0 has a step
                return None
            presses.append(DIRS[best[0][0]])
            optsets.append([DIRS[di] for di, _ in best])
            cur = best[0][1]
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class SokubunnyExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Board` field.

    The base class keeps the memo, the level scoping and the disk cache; only the
    strategy underneath changes. Here the strategy is "read the live board, sweep
    it forwards then backwards, hand back the shortest plan and its exact tie
    sets", so `heuristic` is never called and asserts rather than returning a
    number nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state (a recovery prefix, an epsilon detour) is answered exactly --
    including "this board is now dead", which an exploration prefix really can
    produce here, and by two different routes: a crate wedged where no push can
    ever move it, or a crate of the wrong colour parked on the only route to a
    target."""

    directions = list(DIRS)
    #: `_key` is the dynamic objects only (player + both crate colours), which is
    #: canonical WITHIN a level but not across them -- walls and targets are
    #: static per level and differ between the 23.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        self.player_ids = set(self.game._engine._player_indices)
        self.red_target_ids = set(self.g.resolve_object_name("redtarget"))
        self.blue_target_ids = set(self.g.resolve_object_name("bluetarget"))
        self.wall_ids = set(self.g.resolve_object_name("wall"))
        self.red_crate_ids = set(self.g.resolve_object_name("redcrate"))
        self.blue_crate_ids = set(self.g.resolve_object_name("bluecrate"))
        #: The `_key` alphabet. Both crate colours are in it as DISTINCT object
        #: indices, so two boards that differ only by swapping a red crate for a
        #: blue one key differently -- which they must, since the win condition
        #: tells them apart.
        self.dyn_ids = (self.player_ids | self.red_crate_ids
                        | self.blue_crate_ids)
        #: `_Board`s by STATIC signature (see `read`), not by level index.
        self._boards: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "SokubunnyExpert reads an exact distance field; heuristic is unused")

    def _key(self, eng) -> frozenset:
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn))

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, state)`` for the engine's current grid.

        Boards are cached by their STATIC signature (dimensions, walls, both
        target sets), not by level index, so the two sweeps never rebuild the edge
        table and a board is shared by every state of its level."""
        h, w = eng.height, eng.width
        walls, red_t, blue_t, red_c, blue_c = [], [], [], [], []
        player = None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.red_target_ids:
                    red_t.append(i)
                if cell & self.blue_target_ids:
                    blue_t.append(i)
                if cell & self.red_crate_ids:
                    red_c.append(i)
                if cell & self.blue_crate_ids:
                    blue_c.append(i)
                if cell & self.player_ids:
                    player = i
        sig = (h, w, tuple(walls), tuple(red_t), tuple(blue_t))
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(h, w, walls, red_t, blue_t)
        return board, (player, tuple(sorted(red_c)), tuple(sorted(blue_c)))

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if state[0] is None:                 # no player on the board
            return None
        if board.won(state):
            return Plan([], [])
        if (len(state[1]) < len(board.red_targets)
                or len(state[2]) < len(board.blue_targets)):
            # Not reachable on the shipped levels (every board has as many crates
            # as targets, per colour) and not a crash if a level is ever edited:
            # with fewer crates than targets of a colour that half of the win
            # condition is unsatisfiable.
            return None
        return board.solve(state)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class SokubunnySolver(PSAStarSolver):
    game_id = "puzzlescript_sokubunny_and_the_colored_boxes"
    game_name = GAME_NAME
    expert_cls = SokubunnyExpert

    #: `games/ps:sokubunny_and_the_colored_boxes/...py` is a plain passthrough (it
    #: constructs the adapter and nothing else), so there is nothing to gain by
    #: routing through it -- but it IS what a live agent is handed, so if that
    #: wrapper ever grows a patch this must be set to
    #: ``"ps:sokubunny_and_the_colored_boxes"``.
    game_module_id = ""

    #: Unused: the expert enumerates an exact field rather than searching under a
    #: heuristic, so there is no weight to trade and no node budget that could
    #: quietly bite. Left at the base values so nothing reads a lie off them.
    weight = 1
    node_cap = 400_000

    #: The longest plan is 52 presses (level 2); the rest is room for a re-plan
    #: after the exploration prefix. Stays under the adapter's own 200-step
    #: per-level budget, which `set_level` resets before the plan starts anyway.
    max_steps = 150

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

def _new():
    solver = SokubunnySolver()
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _plans() -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the field and the tie labelling at
    once.

    The ``mis-set`` column counts the crates that START on a target of the WRONG
    colour, which is this game's signature opening and is what makes level 2
    (and the four two-red-crate boards) longer than their size suggests."""
    _solver, game, expert, solvable = _new()
    idx = game._game.obj_name_to_idx
    total = ties = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board, state = expert.read(eng)
        red, blue = set(state[1]), set(state[2])
        wrong = (len(red & board.blue_targets) + len(blue & board.red_targets))
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
              f"{len(red)}R/{len(board.red_targets)} "
              f"{len(blue)}B/{len(board.blue_targets)}  "
              f"mis-set {wrong}  {len(plan):3d} presses  win={won}  "
              f"(budget {game._max_steps}, {room})  "
              f"{tie_steps:3d} steps with a tie set")
    print(f"  solvable levels: {len(solvable)}/{game.n_levels}")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _brute(board, state, limit):
    """Presses to a win from ``state``, searched fresh, or None past ``limit``.

    Shares no code with `_Board.field` beyond the mechanic itself: a plain
    forward BFS whose first win is shortest by construction."""
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


def _selfcheck(walk_presses: int = 400, samples: int = 60,
               verbose: bool = True) -> int:
    """Four things the expert would otherwise be trusted on.

    1. **The native model is the interpreter's.** A seeded random walk on every
       level, comparing `_Board.step`'s board against the engine's grid after
       EVERY press -- including the presses that do nothing, which is where a
       collision-layer mistake hides, and including the pushes that put a crate
       on the wrong colour's target, which is where a typing mistake hides.
    2. **The plans are shortest.** A plain forward BFS to the first win, which is
       shortest by construction and shares no code with the field's backward
       half, must return the field's plan length on every level.
    3. **The tie sets are exact.** Every step's optimal set is re-derived by
       brute force -- a fresh depth-bounded BFS from each of the four successors
       -- and must match the field's answer exactly. This game is small enough to
       afford that on EVERY step of EVERY level, so nothing here is sampled.
    4. **Recovery is answered from ANY board, not just the start.** This is the
       one that matters at generation time: the exploration prefix really does
       strand this game, and by two routes (a crate wedged in a corner, or a
       crate of the wrong colour parked in the only corridor to a target), so
       `_search` has to be right about arbitrary states in both directions. From
       ``samples`` states reached by seeded random presses, the field must either
       return a plan that WINS when replayed through the interpreter, or return
       None on a board an independent forward BFS also finds no win from. A None
       that is merely a give-up would be a silent bug: `record_level` reads it as
       "reset", and a wrong plan length here would not show up in any of the
       three checks above, all of which only ever ask about starts.
    """
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    bad = 0

    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        rng = random.Random(f"sokubunny:selfcheck:{level}")
        drift = 0
        for _ in range(walk_presses):
            di = rng.randrange(4)
            eng.step(DIRS[di])
            state = board.step(state, di)
            _b, live = expert.read(eng)
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
        shortest = _brute(board, state, board.h * board.w * 8)
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
        rng = random.Random(f"sokubunny:recovery:{level}")
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
            # space, which either finds a win or PROVES there is none.
            truth = _brute(board, state, board.h * board.w * 8)
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


def _engine(node_cap: int = 400_000, verbose: bool = True) -> int:
    """The proof that owes nothing to the native model: enumerate the REAL
    interpreter.

    `StateGraph.build` walks every state reachable from the level start by
    pressing actual buttons on the engine and keying the resulting grid, then
    solves the graph exactly. It shares no line of code with `_Board` -- no
    modelled push, no modelled collision layer, no modelled win test, and in
    particular no modelled crate TYPING (the key is built from the engine's own
    object indices, which are what tell RedCrate from BlueCrate) -- so agreement
    on both the plan LENGTH and the per-step optimal SET is an independent
    derivation of everything this file claims.

    Affordable because the whole game is a few hundred thousand engine states
    across 23 boards; on a board like `Sokoban_Flipped`'s level 6 this pass is
    the one that would not run."""
    from solvers.common.ps_astar import StateGraph                     # noqa: PLC0415

    _solver, game, expert, _solvable = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(eng, level)
        game.set_level(level)
        graph = StateGraph.build(eng, expert._key, list(DIRS), node_cap=node_cap)
        eplan = graph.plan(list(DIRS)) if graph is not None else None
        if eplan is None:
            print(f"  L{level:2d}: the interpreter enumeration found no win "
                  f"(node_cap {node_cap})")
            bad += 1
            continue
        same_len = len(eplan) == len(plan)
        esets = [sorted(s) for s in eplan.optsets]
        fsets = [sorted(s) for s in plan.optsets]
        same_sets = esets == fsets
        bad += (not same_len) + (not same_sets)
        if verbose:
            print(f"  L{level:2d}: {len(graph.succ)}"
                  f" engine states, {len(eplan)} presses vs the field's "
                  f"{len(plan)} -- {'AGREE' if same_len else 'DISAGREE'}; tie sets "
                  f"{'AGREE' if same_sets else 'DISAGREE'}")
    return bad


def _macro(verbose: bool = True) -> int:
    """Run the shared `PSSokobanExpert` on this game and print its plans beside
    the field's.

    Not a check that has to pass -- it is the measurement behind the module
    docstring's account of why this game gets its own expert, and here the
    interesting part is that the shared expert is COLOUR-BLIND in exactly the
    place this game is not. `PSSokobanExpert` builds one push-distance table per
    target and greedily matches "a free piece" to it, with no notion that a red
    crate cannot satisfy a blue mark; its win test is the engine's, so any plan
    it returns is a real win, but its estimate cheerfully believes a board is
    nearly done when the two crates are on each other's targets. So the expected
    result is "wins, but not shortest", and this reports by how much.

    Measured: it wins all 23, is 17 presses long in total, and the slack lands on
    7 levels (4 each on 0 and 9; 2 each on 6, 11, 12 and 13; 1 on 21).

    Read the "labelled differently" column with care where the two plans differ in
    LENGTH: it compares the sets step by step, and once the routes have parted
    that is a comparison of two different walks, not of two labellings of one. It
    means something only on the 16 levels where both plans are the same length."""
    from solvers.common.ps_astar import PSSokobanExpert                # noqa: PLC0415

    _solver, game, expert, _solvable = _new()
    eng = game._engine

    class _Macro(PSSokobanExpert):
        pushable_names = ("crate",)
        target_names = ("target",)
        #: Off by default, so a plain `PSSokobanExpert` game ships with NO sets
        #: at all; turned on here to compare the inferred answer against the
        #: exact one rather than against nothing.
        annotate = True

    macro = _Macro(game, node_cap=400_000, weight=1)
    worse = extra = 0
    for level in range(game.n_levels):
        game.set_level(level)
        exact = expert.plan(eng, level)
        game.set_level(level)
        mplan = macro.plan(eng, level)
        n = None if mplan is None else len(mplan)
        worse += n is None or n > len(exact)
        extra += 0 if n is None else max(0, n - len(exact))
        if verbose:
            tag = ("SHORTEST" if n == len(exact) else
                   "no plan" if n is None else f"{n - len(exact)} presses LONGER")
            fsets = [sorted(s) for s in exact.optsets]
            msets = [sorted(s) for s in (getattr(mplan, "optsets", None) or [])]
            differ = (sum(a != b for a, b in zip(fsets, msets))
                      if len(msets) == len(fsets) else len(fsets))
            print(f"  L{level:2d}: field {len(exact):3d}  macro "
                  f"{'None' if n is None else f'{n:3d}'}  -- {tag}; "
                  f"{sum(1 for s in fsets if len(s) > 1)} exact ties vs "
                  f"{sum(1 for s in msets if len(s) > 1)} inferred, "
                  f"{differ} step(s) labelled differently")
    if verbose:
        print(f"  {extra} presses of slack in total")
    return worse


#: Every cell stack the engine can produce. The two absences are structural, not
#: omissions: Player/Wall/Crate share one collision layer and RedTarget/BlueTarget
#: share another, so ``player + crate``, ``redcrate + bluecrate`` and
#: ``redtarget + bluetarget`` can never occur. The four crate-on-target stacks
#: are all reachable and all matter -- two of them are the win being assembled and
#: two of them are the mistake this game is built around.
_COMPOSITIONS = [
    (),
    ("wall",),
    ("redtarget",), ("bluetarget",),
    ("redcrate",), ("bluecrate",),
    ("redcrate", "redtarget"), ("bluecrate", "bluetarget"),
    ("redcrate", "bluetarget"), ("bluecrate", "redtarget"),
    ("player",), ("player", "redtarget"), ("player", "bluetarget"),
]


def _audit(verbose: bool = True) -> int:
    """Two passes over every reachable cell COMPOSITION.

    **Pass 1 -- distinctness**, at every cell size the 23 boards use (7, 8, 9, 10
    and 12 px; the size list is derived rather than assumed so an edited level is
    covered). Whole 64x64 frames of uniform boards are compared rather than one
    cell out of a mixed board: `_render_frame` upscales and centre-pads, so
    slicing a cell by ``cell_px`` arithmetic reads the wrong pixels (the ps:explod
    lesson). Two uniform boards render identically iff their cells do.

    This is the pass that caught the bug the .txt header documents -- with the
    shipped ring sprites, ``player + redtarget`` and ``player + bluetarget`` were
    identical at all five sizes.

    **Pass 2 -- the group.** The game is in `_FLIP_GAMES`, so a board is drawn at
    any of the eight turns and mirrors, and two of its sprites (the wall weave,
    the bunny) are not invariant under them. That is fine as long as no transform
    of one composition is another composition's art, which is what this checks:
    all eight transforms of each, against all others. It runs on SQUARE boards,
    because a rotation of a non-square board also moves the letterbox and every
    comparison would pass for the wrong reason."""
    from adapters.puzzlescript_adapter import _render_frame            # noqa: PLC0415

    _solver, game, _expert, _solvable = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = _COMPOSITIONS

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
    and require the frames to be exactly the transform of the unaugmented ones.

    This is the evidence for putting the game in `_FLIP_GAMES` (the rotation is
    mandatory and is checked here too). The structural argument is in the module
    docstring and is `ps:sokoban_sanity`'s -- one rule, written with the relative
    ``>``, moving at most two bodies onto squares that can never coincide, and a
    colour that no turn or mirror touches -- but Gobble Rush's chirality hid
    inside exactly this kind of argument, so it is measured.

    Both the PLANS and a seeded random walk are replayed: the walk reaches the
    boards a plan never visits (a crate flat against a wall, two crates jammed
    together, a crate dead in a corner, a red crate sitting on a blue mark) and
    it presses the unbound ACTION key as well."""
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
            rng = random.Random(f"sokubunny:symmetry:{level}")
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
    if "--engine" in sys.argv:
        violations = _engine()
        print(f"engine enumeration: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--macro" in sys.argv:
        worse = _macro()
        print(f"macro A*: {worse} level(s) where the colour-blind heuristic "
              f"costs presses")
        sys.exit(0)
    if "--audit" in sys.argv:
        collisions = _audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = _symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    sys.exit(SokubunnySolver.main())
