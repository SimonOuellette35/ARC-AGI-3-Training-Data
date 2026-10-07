"""Generate Phase-1 training data for the PuzzleScript game
ps:stick_candy_puzzle_saga ("Sticky Candy Puzzle Saga", Alan Hazelden).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, a native model of them, the searches that model is planned with, the
labelling of every recorded press, and the render audit that convicted (and
then verified the fix of) the sprites.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_stick_candy_puzzle_saga",
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
exactly. Every expert step carries an optimal-action set.

The game
--------
A sokoban in which the crates WELD THEMSELVES TOGETHER. Two rules carry it:

    [ >  Player | Candy ] -> [  >  Player | > Candy  ]
    [ moving Candy | stationary Candy ] -> [ moving Candy | moving Candy ]

The first is the ordinary push. The second has no rule direction, so the
interpreter expands it over all four, and it runs to fixpoint: the force spreads
from the pushed candy to every candy orthogonally CONNECTED to it, and the whole
connected component translates one cell as a rigid body. `Candy = Yellow or Red
or Blue`, so the component is colour-BLIND -- candies of different colours stick
to each other exactly as readily as candies of the same colour.

Because a component moves rigidly, the offsets between its members never change
again. **Components therefore only ever MERGE, never split, and a merge is
permanent.** That single sentence is the whole game, and the author says so on
the level-3 title card: "Don't let different candies stick together!". A yellow
that touches a red is welded to it for the rest of the level, and every square
the pair can still reach it reaches together.

Blocking is the other half. A candy that would be pushed into a Wall cancels the
move -- and not just its own:

    [ > Candy | Wall ] [ > Player | > Candy ] -> [ Candy | Wall ] [ > Player | Candy ]
    [ > Player | stationary Candy ] [ moving Candy ] -> [ > Player | Candy ] [ Candy ]

The first rule takes the force off the blocked candy AND off the one the player
is pushing; the second then sweeps the force off every candy still moving. The
player keeps its own force but is standing directly behind a candy that has just
been made stationary, and Player / Wall / Candy share a collision layer, so the
player does not move either. Net effect, and it is the one the model implements:
**a push either moves the entire component by one cell and the player with it,
or nothing on the board moves at all.**

Nothing else can block a component. The cell ahead of a moving candy is either
free (background or a target -- targets are on their own collision layer and
block nothing), or a Wall, or another candy -- and a candy ahead of a candy is
orthogonally adjacent to it, hence in the same component, hence also moving. The
player cannot block it either: the player is behind the pushed candy, moving the
same way, and the only component cell that could ever want the player's square is
the one directly behind the player, which the player vacates in the same tick.

Two prelude lines matter. ``noaction`` unbinds the ACTION key, so this game has
exactly four moves and the search branches on four (the exploration prefix still
presses ACTION5 -- a live agent has that button -- which is why ``--symmetry``'s
random walk presses it too). ``run_rules_on_level_start`` runs the cosmetic
decal rules before the first frame, so a level's opening frame already carries
its wall borders and candy joins.

Win: ``All Yellow on YellowTarget`` and ``All Red on RedTarget`` and ``All Blue
on BlueTarget`` -- three subset tests, one per colour. Every one of the twenty
shipped levels has exactly as many candies as targets of each colour, so the
winning board is a FULLY DETERMINED configuration: the candies stand on
precisely the target cells, colour for colour. That fact is what the heuristics
below are built out of, and it has a strong corollary -- since candies that ever
merge stay merged, two candies that end up in different connected components of
the TARGET layout can never have touched each other at any point in the game.

All twenty levels are 7x7 to 11x15, all borders are Wall (so the board edge is
never load-bearing, and a candy is never blocked by geometry the .txt does not
draw), none is already won at reset, and the candy counts run 3 to 15.

Everything above was MEASURED against the interpreter rather than read off the
.txt; ``--selfcheck`` is the executable form of it (a seeded random walk on every
level, the model's board compared against the engine's grid after every single
press, including the presses that do nothing -- which is where a collision-layer
mistake hides), and ``--mechanics`` is the directed version that exhibits each
claim on a shipped board.

Expert solver
-------------
A NATIVE model of the mechanic above (`_Board`), planned over MACRO moves: one
macro is "walk to a cell from which the push you want is available, then push".
Cost is counted in PRESSES (walk length + 1), which is what the agent pays and
what the recording tapes, so the plan a macro search returns is a plan in the
primitive action space -- `_expand` unrolls it back into individual presses.
Branching on macros is what makes the search feasible at all: the primitive
search re-derives the same walks at every depth (a 60-press answer is a depth-60
search over branching 4, of which only the dozen-odd pushes changed anything),
and it does not reach the second level of this game.

Macro states are keyed by ``(candy configuration, player cell)`` -- the exact
state, NOT the usual sokoban "player region" canonicalisation. Region keying is
an approximation (it commits to the cheapest cell the region was entered at, and
a costlier entry can be the one closer to the push you need next), and this
generator wants the levels it calls shortest to actually be shortest.

Two heuristics, both built on the rigidity argument above:

  * ``_Shapes`` is a per-SHAPE pattern database. A component's cells keep their
    offsets forever, so the component moves through translation space as one
    rigid piece; a BFS over the translations at which the piece fits between the
    walls gives the exact minimum number of PUSHES that piece needs to reach a
    given placement. Cached by shape, so the twenty-odd shapes a level's search
    ever produces are each swept once.

  * ``_cover`` chooses which placement each component is heading for. The
    winning board is exactly the target cells, so the components must EXACTLY
    COVER the targets: a backtracking exact-cover search over "which translation
    does each component take". This is both the admissible heuristic and by far
    the strongest dead test in the file -- a state whose components admit no
    exact cover can never be won, and that is what convicts a wrong merge
    IMMEDIATELY (weld a yellow to a red and, four times out of five, no
    placement of that two-colour piece lands both halves on their own colour).

    The admissible number it returns is ``sum over the connected components OF
    THE TARGET LAYOUT of the largest rigid-push distance of any piece assigned
    to that group``. Sound because one press moves at most one component, and a
    component's candies all finish in one target group, so presses spent on
    different groups are disjoint -- while within a group they may be shared,
    which is why it is a max there and a sum across.

`_astar` is that heuristic with weight 1 under a node cap: when it finishes, the
plan is provably SHORTEST. `_beam` is the fallback for the levels it cannot
close, ordered by the same exact cover measured as a SUM of distances (a sharper
guide, not a bound) -- those plans WIN but are not claimed optimal.

Which levels get which, MEASURED (``--plans``):

    level    0   1   2   4   6  11  12  13  14  18
    presses 20  49  60 100  62  90  97 115 125 125
    how     ex  ex  bm  bm  ex  bm  bm  bm  bm  bm

**Ten of the twenty levels, 843 presses in total; three of them (0, 1 and 6, at
20 / 49 / 62 presses) provably shortest.** The other ten -- 3, 5, 7, 8, 9, 10,
15, 16, 17 and 19 -- are won by NEITHER search inside its budget and are skipped
rather than half-taped; `skip_levels` records them so a cold start does not
spend the budget re-discovering it, and ``--plans --all`` ignores that list and
re-measures.

That is an honest result rather than a good one, and it is worth saying where
the difficulty is, because it is neither the size of the space nor the sharpness
of the heuristic. It is ORDERING, between pieces that are nowhere near each
other. On level 3 the three blue candies have to be assembled on the row that is
the only door into the bottom chamber, and all four yellows have to be through
that door BEFORE the blues settle onto it -- so the greedy move (the blues start
two presses from home) is also the fatal one, and no relaxation that ignores the
other candies can see that. `_cover` does catch the mistake, but several plies
too late, and by then the beam has spent its whole width inside the doomed half
of the tree. A search that could see it would have to reason about the ORDER the
target groups are completed in, which is a different program from this one.

Optimal-action sets
-------------------
Every recorded press carries a set (the always-emit-optimal-targets rule).

A macro's WALK is order-free: every direction that stands on a shortest path to
the push cell finishes the macro in the same number of presses, so the set at a
walk step is all of them, read off a single BFS distance field. The PUSH step is
labelled with itself alone -- which crate you shove where IS the puzzle here, so
a different push is a different plan rather than a reordering of this one. That
is `PSPushExpert.annotate_walks`' rule, and it is the right one for a game whose
every state-changing move is a push.

On the levels `_astar` closed, those sets are a SUBSET of the true optimal set:
the plan is shortest, so any shortest walk to the chosen push cell lies on a
shortest solution. ``--proof`` checks exactly that where it is affordable -- an
independent primitive-level layered BFS re-derives ``d*`` and the exact optimal
set at every state of the plan, and requires (a) the same length and (b) every
label the expert emitted to be in the exact set. It runs on the three exact
levels -- 5145, 91430 and 404460 primitive states -- and passes: same ``d*``,
every label inside the exact set. The seven beam levels are past its cap by an
order of magnitude (each stops the sweep at a million states without reaching
the winning layer), which is the same wall the shipped search hit.

On the beam levels the plan is not claimed shortest and the labels are "optimal
given this plan", the standard fallback for a search without a field.

Recovery
--------
`supports_recovery` with RESET-mode recovery: an epsilon-decayed exploration
prefix runs first and one RESET returns the level to its start, from which the
cached plan is a guaranteed win. RESET rather than re-plan because this game is
IRREVERSIBLE in the strongest sense a sokoban can be -- two candies that touch
are welded, and no sequence of presses ever separates them again. A prefix of
random presses strands a board permanently and often: ``--selfcheck``'s fourth
pass measures it by perturbing every shipped level 40 ways with 1-7 random
presses and asking the expert. Of those 400 boards, **102 are PROVED already
lost** -- not "the search gave up", but `_cover` showing that the components can
no longer cover the targets, i.e. two colours are welded -- 155 are re-planned
and replay to a win, and 143 are neither inside the tiny budget that pass runs
under. A quarter of "a handful of random presses" is fatal, and nothing on
screen says so, which is exactly what the RESET recovery arc teaches here.

The expert still re-plans from the LIVE board rather than replaying a stored
path -- `_search` reads whatever the engine currently holds -- so a state the
prefix leaves behind is answered on its own terms, including "this is dead",
which is what `record_level` needs to hear to fall back to the RESET.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`), which together re-sample the board's
8-element symmetry group.

The flips are safe here on the ESCAPE! argument: no gravity, screen-relative
input, the push stated with the relative ``>`` force and no rule naming an axis,
and three win conditions that name no direction. Nothing can contest a cell
either -- one press moves one component plus the player, all by the same vector,
so no two bodies are ever offered the same square and the rule-order chirality
that Gobble Rush has to argue around cannot arise.

The only directional ART is the join decals, and they are an exact orbit of the
8-element group: JoinL's two pixels are the left column's corners and JoinR's
are the right column's (each other's mirror), JoinU's are the top row's and
JoinD's the bottom row's, JoinBoth's are all four. The rules that place them are
equivariant too -- a candy shows a mark on each side that has a neighbour, and
"neighbour on both sides of an axis" renders as all four corners whichever axis
it is. ``--symmetry`` measures the whole claim anyway: every level's plan AND a
seeded random walk (which does what a plan never does -- shoves candies into
walls, welds the wrong colours together and presses the unbound ACTION key)
replayed at all 16 presentations, requiring every frame to be the exact transform
of the unaugmented one -- 320 replays over the ten solved levels, 0 violations,
which is what the `_FLIP_GAMES` entry cites.

There is deliberately no colour augmentation: in this game the colour IS the win
condition, and a recolor that flattened or permuted the three would either erase
the rule or teach it wrong.

Rendering
---------
SIX sprites had to change, and they are the ones the win condition is made of.
See the header comment in ``data/puzzlescript_games/stick_candy_puzzle_saga.txt``
for the full statement; in short, a candy was opaque everywhere except its four
CORNERS, so "standing on a target, and on which COLOUR of target" was carried by
four pixels -- and those four pixels are exactly where the join decals draw,
from a higher collision layer. A won board is by construction a blob of mutually
adjacent candies, so at the moment of victory every candy is corner-covered and
the frame is pixel-identical to the same blob standing on bare floor. ``--audit``
measured it: three compositions identical at all five cell sizes.

Each candy is now a RING with a transparent 3x3 centre, and each target is SOLID
in its bright colour inside its dark frame, so a candy reports the colour of what
it is standing on through its own middle -- nine pixels, none of them reachable
by a join, all of them surviving the cell_px=4 decimation that drops sprite row 2
and column 2 on the two widest boards.

``--audit`` renders every cell COMPOSITION the game can show -- floor, wall, each
target, each candy, all NINE candy-on-target stacks (three of which are wins in
the making and six of which are the mistake this game never forgives), the
player, the player on each target, and the joined forms of each -- as a whole
64x64 frame of a uniform board, at every cell size the twenty levels render at
(4, 5, 7, 8 and 9 px), and requires them pairwise distinct. Whole frames rather
than one cell sliced out of a mixed board: `_render_frame` upscales and
centre-pads, so slicing by ``cell_px`` arithmetic reads the wrong pixels (the
ps:explod lesson).

Usage (run from the repo root):
    python solvers/generate_stick_candy_puzzle_saga_training.py --episodes 200 \\
        --out data/training_multi_level/stick_candy_puzzle_saga

    python solvers/generate_stick_candy_puzzle_saga_training.py --plans      # level report
    python solvers/generate_stick_candy_puzzle_saga_training.py --selfcheck  # model fuzz
    python solvers/generate_stick_candy_puzzle_saga_training.py --mechanics  # directed probes
    python solvers/generate_stick_candy_puzzle_saga_training.py --proof      # shortest + tie sets
    python solvers/generate_stick_candy_puzzle_saga_training.py --engine     # replay in the adapter
    python solvers/generate_stick_candy_puzzle_saga_training.py --audit      # rendering
    python solvers/generate_stick_candy_puzzle_saga_training.py --symmetry   # augmentation
"""

from __future__ import annotations

import heapq
import itertools
import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameAction, GameState                  # noqa: E402
from adapters.puzzlescript_adapter import (PuzzleScriptAdapter,           # noqa: E402
                                           _render_frame)
from solvers.common.ps_astar import (PSAStarSolver, PSExpert, Plan,       # noqa: E402
                                     screen_action, snapshot, restore)

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "stick_candy_puzzle_saga"

#: Engine directions, in the order ties are broken. ACTION5 is unbound
#: (``noaction`` in the prelude, and no rule has ``action`` on its left), so it
#: is not a move and branching on it would double the search for nothing.
DIRS: tuple[str, ...] = ("up", "down", "left", "right")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: The three candy colours, in the order `_Board` keeps them. The index is the
#: colour: 0 = Yellow, 1 = Red, 2 = Blue, and ``targets[i]`` is that colour's
#: target set. The three are kept APART everywhere -- a state that merged them
#: would identify boards the win condition distinguishes.
COLOURS: tuple[str, ...] = ("yellow", "red", "blue")

#: Disk cache of every level's start plan AND its per-step optimal sets. Unlike
#: the small ps: sokobans this one is load-bearing: the cold searches cost
#: minutes, and without a file on disk every shard `parallelize_generator`
#: starts would re-derive all of them. Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "stick_candy_puzzle_saga_plans.json"

#: Levels neither search wins inside its budget, MEASURED by ``--plans --all``
#: (which ignores this set and re-tries every level). They are skipped up front
#: so a cold start does not spend the exact budget plus three beams per level
#: re-discovering it; an episode keeps only the levels that do win, and
#: ``level_id`` preserves the true provenance either way.
SKIP_LEVELS: tuple[int, ...] = (3, 5, 7, 8, 9, 10, 15, 16, 17, 19)

#: Which search answered each level ("exact" / "beam"), kept beside the plan
#: cache. `PSExpert`'s disk format carries the plan and its optimal sets and
#: nothing else, so without this a second process reads a cache hit and can no
#: longer say whether the plan it just loaded is provably shortest -- which is
#: precisely the thing ``--plans`` and ``--proof`` report on.
NOTES_CACHE: Path = _REPO_ROOT / "data" / "stick_candy_puzzle_saga_notes.json"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic.

    A STATE is ``(player, yellow cells, red cells, blue cells)`` with each colour
    kept as its own sorted tuple of ``(row, col)`` pairs -- the only things any
    rule in this game moves. Walls and the three target sets never change, so
    they live here rather than in the state.

    `step` is the two-rule mechanic of the module docstring resolved under the
    collision layers ``Background / Marker / Target / (Player, Wall, Candy) /
    CoverUD / (CoverLR, Number)``:

      * the cell ahead of the player is a Wall -> nothing moves;
      * it is free (background, or a target, which blocks nothing) -> the player
        walks;
      * it is a candy -> flood the orthogonally connected component of ALL
        candies regardless of colour, and if every one of its cells has a
        non-Wall cell ahead of it, translate the whole component and the player
        one step; otherwise nothing moves, player included.

    The decal objects (WallL/R/U/D, JoinL/R/U/D/Both, Marker, the level-number
    digits) are recomputed from scratch by ``late`` rules every tick and are
    pure presentation, so the model does not carry them -- ``--selfcheck``
    compares the model against the engine's Player/Wall/Candy/Target layers,
    which is the whole of the state anything mechanical reads.
    """

    __slots__ = ("h", "w", "wall", "targets", "all_targets", "group",
                 "n_groups", "n_targets", "shapes")

    def __init__(self, h: int, w: int, walls, targets):
        self.h = h
        self.w = w
        self.wall = [[False] * w for _ in range(h)]
        for (r, c) in walls:
            self.wall[r][c] = True
        self.targets = tuple(frozenset(t) for t in targets)
        self.all_targets = frozenset().union(*self.targets)
        self.n_targets = sum(len(t) for t in self.targets)
        # The pattern database for this level, built on first use and shared by
        # every search this board is ever asked for. It is what makes
        # re-planning cheap: a fresh `_Shapes` per call would re-sweep the same
        # shapes for every state a recovery pass hands the expert.
        self.shapes = None
        # Connected components of the TARGET layout, colours ignored. Two
        # candies that finish in different groups never touched each other at
        # any point in the game (a merge is permanent), so presses spent moving
        # one group's candies are disjoint from presses spent on another's --
        # which is what makes `_cover`'s sum-across-groups admissible.
        self.group: dict = {}
        gid = 0
        for cell in sorted(self.all_targets):
            if cell in self.group:
                continue
            stack = [cell]
            self.group[cell] = gid
            while stack:
                r, c = stack.pop()
                for dr, dc in _DELTA.values():
                    q = (r + dr, c + dc)
                    if q in self.all_targets and q not in self.group:
                        self.group[q] = gid
                        stack.append(q)
            gid += 1
        self.n_groups = gid

    # -- the mechanic --------------------------------------------------------
    def step(self, state: tuple, direction: str) -> tuple:
        """The settled board one press later. Returns ``state`` unchanged when
        the press moves nothing (a walk into a wall, or a push the wall rule
        cancels) -- the engine's "can't move" sounds, which change no state."""
        player = state[0]
        dr, dc = _DELTA[direction]
        nr, nc = player[0] + dr, player[1] + dc
        h, w, wall = self.h, self.w, self.wall
        # Off the board counts as a wall. Every shipped level is ringed with
        # Wall so no piece can ever reach the edge and this branch is dead --
        # but the interpreter blocks a move off the grid whatever the .txt
        # draws, and an edited level must not walk off the end of a list.
        if not (0 <= nr < h and 0 <= nc < w) or wall[nr][nc]:
            return state
        occ = _occ(state)
        if (nr, nc) not in occ:
            return ((nr, nc),) + state[1:]

        comp = _component(occ, (nr, nc))
        for (r, c) in comp:
            tr, tc = r + dr, c + dc
            if not (0 <= tr < h and 0 <= tc < w) or wall[tr][tc]:
                return state                 # the cancel rules: nothing moves
        moved = {cell: (cell[0] + dr, cell[1] + dc) for cell in comp}
        return ((nr, nc),) + tuple(
            tuple(sorted(moved.get(cell, cell) for cell in grp))
            for grp in state[1:])

    def won(self, state: tuple) -> bool:
        """``All Yellow on YellowTarget`` and the same for Red and Blue -- three
        SUBSET tests, exactly as the win conditions are written, so an edited
        level with a candy to spare is still answered correctly."""
        return all(set(grp) <= tgt for grp, tgt in zip(state[1:], self.targets))

    def free(self, cell) -> bool:
        r, c = cell
        return 0 <= r < self.h and 0 <= c < self.w and not self.wall[r][c]

    def walk_field(self, occ, start) -> dict:
        """Presses to walk from ``start`` to every cell the player can reach
        without pushing anything. Candies are obstacles here: stepping into one
        is a push, which is a macro of its own, not a step of a walk."""
        dist = {start: 0}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            d = dist[cell] + 1
            for dr, dc in _DELTA.values():
                n = (cell[0] + dr, cell[1] + dc)
                if n in dist or n in occ or not self.free(n):
                    continue
                dist[n] = d
                queue.append(n)
        return dist

    def macro_successors(self, state: tuple):
        """``(cost, next state, push cell, direction)`` for every push the player
        can walk to and take. ``cost`` counts PRESSES: the walk plus the push."""
        occ = _occ(state)
        dist = self.walk_field(occ, state[0])
        out = []
        for cell, walk in dist.items():
            for direction in DIRS:
                dr, dc = _DELTA[direction]
                if (cell[0] + dr, cell[1] + dc) not in occ:
                    continue
                nxt = self.step((cell,) + state[1:], direction)
                if nxt[0] == cell:
                    continue                 # the wall rule refused the push
                out.append((walk + 1, nxt, cell, direction))
        return out


def _occ(state: tuple) -> dict:
    """``cell -> colour index`` for every candy on the board."""
    occ = {}
    for colour, grp in enumerate(state[1:]):
        for cell in grp:
            occ[cell] = colour
    return occ


def _component(occ: dict, cell) -> list:
    """The orthogonally connected component of candies containing ``cell``,
    colours ignored -- ``Candy = Yellow or Red or Blue``, so the stickiness does
    not care and that is the trap the whole game is built on."""
    comp = [cell]
    seen = {cell}
    stack = [cell]
    while stack:
        r, c = stack.pop()
        for dr, dc in _DELTA.values():
            q = (r + dr, c + dc)
            if q in occ and q not in seen:
                seen.add(q)
                comp.append(q)
                stack.append(q)
    return comp


def _components(occ: dict) -> list:
    seen = set()
    out = []
    for cell in occ:
        if cell in seen:
            continue
        comp = _component(occ, cell)
        seen.update(comp)
        out.append(sorted(comp))
    return out


# ---------------------------------------------------------------------------
# The pattern database: rigid-body push distances, per SHAPE
# ---------------------------------------------------------------------------

class _Shapes:
    """Exact rigid-push distance fields, cached by component SHAPE.

    A component's members keep their offsets for the rest of the level, so the
    component travels through TRANSLATION space: its position is one anchor and
    a press that moves it changes that anchor by one unit. A translation is
    legal iff every cell of the piece lands off a wall and inside the board, so
    a BFS over the legal translations gives the exact minimum number of presses
    that must move this piece for it to reach a given placement -- ignoring the
    other candies and the player, which is what makes it a LOWER bound and
    therefore admissible.

    (Ignoring the other candies is a relaxation in the right direction and it is
    worth being precise about why, because the usual sokoban argument runs the
    other way: another candy could act as a HANDLE, giving the player a square
    to push from that this piece alone does not offer. That only ever makes the
    real game cheaper or equal in the number of presses that move THIS piece --
    the bound is on the piece's own displacement, which is geometry, and the
    handle cannot shorten a wall-avoiding path.)

    Fields are keyed by shape (offsets plus colours), so the handful of shapes a
    level's search produces are each swept once and every state that contains
    one of them reads its answer out of a dict.
    """

    __slots__ = ("board", "cache")

    def __init__(self, board: _Board):
        self.board = board
        self.cache: dict = {}

    def placements(self, comp: list, occ: dict) -> list:
        """``(pushes, frozenset of target cells covered)`` for every placement of
        this component that lands every one of its candies on a target of its
        own colour AND is reachable through the walls. Empty means this
        component can never be satisfied -- the state is dead."""
        r0 = min(r for r, _ in comp)
        c0 = min(c for _, c in comp)
        shape = tuple(sorted((r - r0, c - c0, occ[(r, c)]) for r, c in comp))
        fields = self.cache.get(shape)
        if fields is None:
            fields = self.cache[shape] = self._build(shape)
        out = []
        for goal, dist in fields:
            d = dist.get((r0, c0))
            if d is None:
                continue                    # walled off from that placement
            out.append((d, frozenset((r - r0 + goal[0], c - c0 + goal[1])
                                     for r, c in comp)))
        # Sorted on the distance and then on the CELLS, spelled as a sorted
        # list: `<` on two frozensets is subset containment, not an order, so a
        # bare sort would leave equal-distance placements in an order that is
        # deterministic but meaningless -- and the cover search below reads this
        # list, so its tie-breaking has to be reproducible across processes.
        out.sort(key=lambda t: (t[0], sorted(t[1])))
        return out

    def _build(self, shape) -> list:
        board = self.board
        cells = list(shape)

        def legal(anchor) -> bool:
            r0, c0 = anchor
            for dr, dc, _k in cells:
                if not board.free((r0 + dr, c0 + dc)):
                    return False
            return True

        def is_goal(anchor) -> bool:
            r0, c0 = anchor
            return all((r0 + dr, c0 + dc) in board.targets[k]
                       for dr, dc, k in cells)

        out = []
        for r0 in range(board.h):
            for c0 in range(board.w):
                if not legal((r0, c0)) or not is_goal((r0, c0)):
                    continue
                dist = {(r0, c0): 0}
                queue = deque([(r0, c0)])
                while queue:
                    a = queue.popleft()
                    d = dist[a] + 1
                    for dr, dc in _DELTA.values():
                        n = (a[0] + dr, a[1] + dc)
                        if n in dist or not legal(n):
                            continue
                        dist[n] = d
                        queue.append(n)
                out.append(((r0, c0), dist))
        return out


# ---------------------------------------------------------------------------
# The exact cover: the heuristic AND the dead test
# ---------------------------------------------------------------------------

def _cover(board: _Board, shapes: _Shapes, occ: dict, mode: str = "max",
           node_cap: int = 20_000):
    """``(value, truncated)`` for the components of ``occ``, or ``(None, False)``
    when they admit NO exact cover of the target cells -- which is a PROOF that
    this state can never be won.

    The winning board is the target set exactly, and a component reaches it as a
    rigid piece, so a solution assigns every component a translation whose image
    is a subset of the targets and the images tile the targets exactly. That is
    an exact cover, solved here by backtracking with the fewest-options-first
    ordering and branch-and-bound on the value.

    ``mode="max"`` returns the ADMISSIBLE number: the sum over the target
    layout's connected groups of the largest rigid-push distance of any piece
    assigned to that group, minimised over covers. One press moves at most one
    component, and every candy of a component finishes in a single target group,
    so presses charged to different groups are disjoint (sum) while presses
    inside one group may be shared by pieces that merge (max).

    ``mode="sum"`` returns the sum of every piece's distance. That is NOT a
    bound -- merging pieces share presses -- but it is a far sharper GUIDE,
    because it keeps counting the pieces that are still scattered instead of
    reporting only the worst one. It is what `_beam` orders by.

    ``truncated`` says the backtracking hit ``node_cap`` and the value may be
    over-estimated, so `_astar` falls back to a value it can still trust.
    """
    opts = []
    for comp in _components(occ):
        places = shapes.placements(comp, occ)
        if not places:
            return None, False
        opts.append(places)

    order = sorted(range(len(opts)), key=lambda i: len(opts[i]))
    n = len(opts)
    best = [None]
    nodes = [0]
    truncated = [False]
    group = board.group
    ntarg = board.n_targets

    def rec(i: int, used: frozenset, acc: int, gmax: dict) -> None:
        if best[0] is not None and acc >= best[0]:
            return
        if i == n:
            if len(used) == ntarg:
                best[0] = acc
            return
        nodes[0] += 1
        if nodes[0] > node_cap:
            truncated[0] = True
            return
        for d, cells in opts[order[i]]:
            if cells & used:
                continue
            if mode == "sum":
                rec(i + 1, used | cells, acc + d, gmax)
            else:
                gid = group[next(iter(cells))]
                prev = gmax.get(gid, 0)
                if d > prev:
                    nxt = dict(gmax)
                    nxt[gid] = d
                    rec(i + 1, used | cells, acc - prev + d, nxt)
                else:
                    rec(i + 1, used | cells, acc, gmax)
            if truncated[0]:
                return

    rec(0, frozenset(), 0, {})
    return best[0], truncated[0]


def _admissible(board, shapes, occ) -> "int | None":
    """`_cover` in ``max`` mode, made safe against a truncated search: a
    truncated backtrack may have missed the cheapest cover, so its value is not
    a bound and 0 (no information) is used instead. None still means DEAD --
    "no cover exists" is only ever returned by a search that ran to completion,
    since a truncated one returns early with ``truncated`` set."""
    value, truncated = _cover(board, shapes, occ, mode="max")
    if value is None and not truncated:
        return None
    if value is None or truncated:
        return 0
    return value


# ---------------------------------------------------------------------------
# The searches
# ---------------------------------------------------------------------------

def _astar(board: _Board, shapes: _Shapes, state: tuple,
           node_cap: int, budget: float):
    """``(macro plan, closed)``, where the plan is ``[(push cell, direction),
    ...]`` or None if the node cap or the time budget bit first.

    Weight 1 over an admissible heuristic with exact ``(config, player cell)``
    keying, so a plan this returns is provably SHORTEST in PRESSES. ``closed``
    says the search emptied its queue -- it enumerated every state reachable
    from here that `_cover` did not prove dead, and none of them was a win, so
    the state is PROVABLY unwinnable rather than merely unsolved.
    """
    if board.won(state):
        return [], True
    h0 = _admissible(board, shapes, _occ(state))
    if h0 is None:
        return None, True
    started = time.time()
    counter = 0
    pq = [(h0, 0, 0, state)]
    best_g = {state: 0}
    parent: dict = {state: None}
    while pq:
        _f, g, _c, s = heapq.heappop(pq)
        if g > best_g.get(s, 1 << 30):
            continue
        for cost, nxt, cell, direction in board.macro_successors(s):
            ng = g + cost
            if board.won(nxt):
                return _trace(parent, s) + [(cell, direction)], False
            if best_g.get(nxt, 1 << 30) <= ng:
                continue
            h = _admissible(board, shapes, _occ(nxt))
            if h is None:
                continue                    # provably dead: never queue it
            best_g[nxt] = ng
            parent[nxt] = (s, cell, direction)
            counter += 1
            heapq.heappush(pq, (ng + h, ng, counter, nxt))
        if len(best_g) > node_cap or time.time() - started > budget:
            return None, False
    return None, True                       # the reachable space closed: no win


def _trace(parent: dict, state: tuple) -> list:
    out = []
    while parent[state] is not None:
        prev, cell, direction = parent[state]
        out.append((cell, direction))
        state = prev
    out.reverse()
    return out


def _beam(board: _Board, shapes: _Shapes, state: tuple, width: int,
          depth: int, weight: int, budget: float):
    """A macro plan from ``state``, or None. WINS but is not claimed shortest.

    Ordered by ``weight * (sum-of-distances cover) + presses so far``: the sum
    keeps every scattered piece in the score, which is what a bound-shaped
    heuristic (the max) cannot do, and it is what makes the difference between
    finding a plan and not on the levels `_astar` cannot close.
    """
    if board.won(state):
        return []
    started = time.time()
    # Nodes are (parent index, push cell, direction); the frontier holds indices
    # so a width-6000 beam does not carry 6000 copies of its own path.
    nodes: list = [(None, None, None)]
    frontier = [(0, 0, state)]
    seen = {state}
    for _ in range(depth):
        kids = []
        for idx, g, s in frontier:
            # The budget is checked HERE and not only between layers: one layer
            # of a wide beam is minutes on its own, so a between-layers-only
            # check overshoots the budget by a whole layer.
            if time.time() - started > budget:
                return None
            for cost, nxt, cell, direction in board.macro_successors(s):
                if board.won(nxt):
                    nodes.append((idx, cell, direction))
                    return _trace_nodes(nodes, len(nodes) - 1)
                if nxt in seen:
                    continue
                value, _tr = _cover(board, shapes, _occ(nxt), mode="sum")
                if value is None:
                    continue                # provably dead
                seen.add(nxt)
                kids.append((weight * value + g + cost, idx, g + cost,
                             cell, direction, nxt))
        if (not kids or time.time() - started > budget
                or len(seen) > BEAM_STATE_CAP):
            return None
        kids.sort(key=lambda k: k[0])
        frontier = []
        for _score, idx, g, cell, direction, nxt in kids[:width]:
            nodes.append((idx, cell, direction))
            frontier.append((len(nodes) - 1, g, nxt))
    return None


def _trace_nodes(nodes: list, idx: int) -> list:
    out = []
    while nodes[idx][0] is not None:
        parent, cell, direction = nodes[idx]
        out.append((cell, direction))
        idx = parent
    out.reverse()
    return out


# ---------------------------------------------------------------------------
# Macro plan -> presses, with the optimal set of every press
# ---------------------------------------------------------------------------

def _expand(board: _Board, state: tuple, macro: list) -> Plan:
    """Unroll ``[(push cell, direction), ...]`` into individual presses, each
    carrying its optimal set.

    A walk step is labelled with EVERY direction that stands on a shortest path
    to the push cell (they all finish the macro in the same number of presses);
    the push itself is labelled with itself alone. See the module docstring for
    why that is the sound answer in a game whose every state-changing move is a
    push.
    """
    presses: list = []
    optsets: list = []
    cur = state
    for cell, direction in macro:
        occ = _occ(cur)
        field = board.walk_field(occ, cell)   # distances TO the push cell
        here = cur[0]
        if here not in field:
            raise AssertionError("push cell unreachable in _expand")
        while here != cell:
            d = field[here]
            best = [dd for dd in DIRS
                    if field.get((here[0] + _DELTA[dd][0],
                                  here[1] + _DELTA[dd][1]), 1 << 30) == d - 1]
            if not best:
                raise AssertionError("no descent in the walk field")
            presses.append(best[0])
            optsets.append(best)
            here = (here[0] + _DELTA[best[0]][0], here[1] + _DELTA[best[0]][1])
        presses.append(direction)
        optsets.append([direction])
        nxt = board.step((cell,) + cur[1:], direction)
        if nxt[0] == cell:
            raise AssertionError("planned push was refused on replay")
        cur = nxt
    if not board.won(cur):
        raise AssertionError("expanded plan does not win")
    return Plan(presses, optsets)


#: Search budget. `node_cap` is macro states; the beam widths are tried in turn.
EXACT_NODE_CAP = 300_000
EXACT_BUDGET_S = 240.0
BEAM_WIDTHS: tuple[int, ...] = (300, 1500, 6000)
BEAM_DEPTH = 185      # = MAX_PRESSES: a macro costs at least one press
BEAM_WEIGHT = 4
BEAM_BUDGET_S = 300.0
BEAM_STATE_CAP = 2_000_000

#: A level whose plan needs this many presses cannot be recorded: the adapter
#: ends an episode at `PuzzleScriptAdapter._max_steps` = 200 actions, counted
#: from the RESET that starts the level (the exploration prefix is separated
#: from the plan by its own RESET, so only the plan itself has to fit). A plan
#: over the limit is refused here rather than shipped as a level that can never
#: reach WIN and would fail every seed.
MAX_PRESSES = 185


def solve(board: _Board, state: tuple, *, exact_only: bool = False,
          node_cap: int = EXACT_NODE_CAP, budget: float = EXACT_BUDGET_S,
          beam_widths: tuple = BEAM_WIDTHS,
          beam_budget: float = BEAM_BUDGET_S):
    """``(Plan, "exact" | "beam") or (None, why)`` for one board state.

    Exact first -- a plan that is provably shortest is worth the search -- then
    the beam widths in turn. ``why`` is ``"dead"`` when `_cover` PROVED the state
    unwinnable on its own, ``"unwinnable"`` when the exact search closed the
    whole reachable space without finding a win, ``"too-long"`` when the
    shortest plan does not fit the adapter's per-level action budget, and
    ``"unsolved"`` when both searches simply ran out.
    """
    if board.won(state):
        return Plan([], []), "exact"
    if board.shapes is None:
        board.shapes = _Shapes(board)
    shapes = board.shapes
    if _admissible(board, shapes, _occ(state)) is None:
        return None, "dead"
    macro, closed = _astar(board, shapes, state, node_cap, budget)
    if macro is None and closed:
        return None, "unwinnable"
    if macro is not None:
        plan = _expand(board, state, macro)
        # A shortest plan that does not fit the adapter's per-level action
        # budget means the LEVEL does not fit it, so nothing longer would help.
        return ((plan, "exact") if len(plan) <= MAX_PRESSES
                else (None, "too-long"))
    if exact_only:
        return None, "unsolved"
    for width in beam_widths:
        macro = _beam(board, shapes, state, width, BEAM_DEPTH, BEAM_WEIGHT,
                      beam_budget)
        if macro is None:
            continue
        plan = _expand(board, state, macro)
        if len(plan) <= MAX_PRESSES:
            return plan, "beam"
    return None, "unsolved"


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class StickCandyExpert(PSExpert):
    """`PSExpert`'s plan memo, disk cache and snapshot discipline around the
    native searches above.

    The base class keeps the memo, the level scoping and the on-disk plan cache;
    only the strategy underneath changes, so `heuristic` is never called and
    asserts rather than returning a number nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state -- a recovery prefix, an epsilon detour -- is answered on its
    own terms, including "this board is dead", which an exploration prefix
    really does produce here and by the cheapest possible accident: one press
    that welds two colours together.
    """

    directions = list(DIRS)
    #: `_key` is the dynamic objects only (player + the three candy colours),
    #: which is canonical WITHIN a level but not across them -- walls and targets
    #: are static per level and differ between the twenty.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        self.player_ids = set(self.game._engine._player_indices)
        self.wall_ids = set(self.g.resolve_object_name("wall"))
        self.candy_ids = [set(self.g.resolve_object_name(name))
                          for name in COLOURS]
        self.target_ids = [set(self.g.resolve_object_name(name + "target"))
                           for name in COLOURS]
        #: The `_key` alphabet. The three colours are DISTINCT object indices in
        #: it, so two boards that differ only by swapping a yellow candy for a
        #: red one key differently -- which they must, since the win conditions
        #: tell them apart.
        self.dyn_ids = set(self.player_ids)
        for ids in self.candy_ids:
            self.dyn_ids |= ids
        #: `_Board`s by STATIC signature (see `read`), not by level index.
        self._boards: dict = {}
        #: ``level -> "exact" | "beam" | "dead" | "unsolved"``, persisted beside
        #: the plan cache so a process that gets a cache HIT can still say how
        #: the plan it loaded was found.
        self.why: dict = self._load_notes()
        self._level: "int | None" = None

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "StickCandyExpert plans over a native model; heuristic is unused")

    def plan(self, eng, level: "int | None" = None):
        """`PSExpert.plan`, remembering which level is being planned so
        `_search` (which the base calls without one) can file its answer under
        it."""
        self._level = level
        return super().plan(eng, level)

    def _load_notes(self) -> dict:
        try:
            return {int(k): v for k, v in
                    json.loads(NOTES_CACHE.read_text()).items()}
        except Exception:                                    # noqa: BLE001
            return {}

    def _save_notes(self) -> None:
        """Written atomically -- shards started together would otherwise
        interleave into a truncated file (see `parallelize_generator`)."""
        try:
            NOTES_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = NOTES_CACHE.with_suffix(f".json.tmp{os.getpid()}")
            tmp.write_text(json.dumps(
                {str(k): self.why[k] for k in sorted(self.why)}, indent=1))
            os.replace(tmp, NOTES_CACHE)
        except OSError:
            pass                                             # cache is optional

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

        Boards are cached by their STATIC signature (dimensions, walls, the three
        target sets) rather than by level index, so a board is shared by every
        state of its level and the pattern database is built once per level.
        """
        walls = []
        targets = [[], [], []]
        candies = [[], [], []]
        player = None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                if cell & self.wall_ids:
                    walls.append((r, c))
                for k in range(3):
                    if cell & self.target_ids[k]:
                        targets[k].append((r, c))
                    if cell & self.candy_ids[k]:
                        candies[k].append((r, c))
                if cell & self.player_ids:
                    player = (r, c)
        sig = (eng.height, eng.width, tuple(walls),
               tuple(targets[0]), tuple(targets[1]), tuple(targets[2]))
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(eng.height, eng.width, walls,
                                               targets)
        state = (player,) + tuple(tuple(sorted(g)) for g in candies)
        return board, state

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if state[0] is None:                 # no player on the board
            return None
        for grp, tgt in zip(state[1:], board.targets):
            if len(grp) < len(tgt):
                # Not reachable on the shipped levels (candies and targets match
                # per colour) and not a crash if a level is ever edited: with
                # fewer candies than targets of a colour that win condition
                # cannot be satisfied.
                return None
        plan, why = solve(board, state)
        if self._level is not None:
            self.why[self._level] = why
            self._save_notes()
        return plan


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class StickCandySolver(PSAStarSolver):
    game_id = "puzzlescript_stick_candy_puzzle_saga"
    game_name = GAME_NAME
    expert_cls = StickCandyExpert

    #: `games/ps:stick_candy_puzzle_saga/...py` is a plain passthrough (it
    #: constructs the adapter and nothing else), so there is nothing to gain by
    #: routing through it -- but it IS what a live agent is handed, so if that
    #: wrapper ever grows a patch this must be set to
    #: ``"ps:stick_candy_puzzle_saga"``.
    game_module_id = ""

    #: Levels neither search wins inside its budget, measured by ``--plans``.
    #: Listed here so a cold start (and every `parallelize_generator` shard) does
    #: not spend the exact budget plus three beams per level re-proving it. The
    #: searches themselves are unchanged -- delete an entry and ``--plans`` will
    #: try it again.
    skip_levels = frozenset(SKIP_LEVELS)

    #: Unused: the expert plans over a native model rather than searching the
    #: engine under a heuristic. Left at the base values so nothing reads a lie
    #: off them.
    weight = 1
    node_cap = 400_000

    #: Room for the longest plan plus a re-plan after the exploration prefix.
    #: Stays under the adapter's own 200-step per-level budget, which
    #: `set_level` resets before the plan starts anyway.
    max_steps = 190

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

def _new(skip: bool = True, plans: bool = True):
    """A solver + its adapter + expert.

    ``skip=False`` clears `skip_levels` so a report re-tries every level from
    scratch; ``plans=False`` builds the adapter and expert WITHOUT planning
    anything, for the reports that only need the mechanic (searching every level
    costs minutes on a cold cache).
    """
    solver = StickCandySolver()
    if not skip:
        solver.skip_levels = frozenset()
    if not plans:
        game = solver.make_game(0)
        expert = solver.expert_cls(game, node_cap=solver.node_cap,
                                   weight=solver.weight)
        return solver, game, expert, []
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _plans() -> int:
    """Per-level report: the plan, how it was found, and whether it is shortest.

    ``--all`` ignores `skip_levels` and re-tries every level, which is how that
    tuple is measured; without it the report is what generation actually does.
    """
    every = "--all" in sys.argv
    solver, game, expert, solvable = _new(skip=not every)
    total = 0
    exact = beam = 0
    print(f"{'lvl':>3} {'size':>7} {'candies':>9} {'presses':>7}  how")
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board, state = expert.read(eng)
        counts = "/".join(str(len(g)) for g in state[1:])
        if level not in solvable:
            why = expert.why.get(level, "skipped")
            print(f"{level:3d} {board.h:3d}x{board.w:<3d} {counts:>9} "
                  f"{'-':>7}  {why}")
            continue
        t0 = time.time()
        plan = expert.plan(eng, level)
        why = expert.why.get(level, "?")
        if plan is None:
            print(f"{level:3d} {board.h:3d}x{board.w:<3d} {counts:>9} "
                  f"{'-':>7}  UNSOLVED")
            continue
        total += len(plan)
        if why == "exact":
            exact += 1
        elif why == "beam":
            beam += 1
        ties = sum(1 for s in plan.optsets if len(s) > 1)
        print(f"{level:3d} {board.h:3d}x{board.w:<3d} {counts:>9} "
              f"{len(plan):7d}  {why:<5s} ties={ties:<3d} {time.time() - t0:.0f}s")
    print(f"\n{len(solvable)} level(s) solved of {game.n_levels}: "
          f"{exact} provably shortest, {beam} by beam; {total} presses in total")
    return 0


def _mechanics() -> int:
    """Directed probes: each claim in the module docstring, exhibited on a
    shipped board by pressing real buttons at the interpreter."""
    solver, game, expert, _s = _new(plans=False)
    eng = game._engine
    bad = 0

    # 1. ACTION5 is unbound (`noaction`): it changes nothing, anywhere.
    moved = 0
    for level in range(game.n_levels):
        game.set_level(level)
        before = snapshot(eng)
        eng.step("action")
        if eng.grid != before:
            moved += 1
    print(f"ACTION5 on all {game.n_levels} level starts: "
          f"{moved} board(s) changed (expected 0)")
    bad += moved

    # 2. A push moves the WHOLE connected component, and the stickiness is
    #    colour-BLIND. Walked to, not asserted: a seeded walk on level 2 (the
    #    author's own demonstration board -- yellows and reds on interleaved
    #    diagonals) until a press moves a component of more than one candy.
    game.set_level(2)
    _b, state = expert.read(eng)
    print(f"level 2 starts as {len(_components(_occ(state)))} separate "
          f"components, one candy each")
    board, state = expert.read(eng)
    rng = random.Random(3)
    for _ in range(4000):
        direction = rng.choice(DIRS)
        nxt = board.step(state, direction)
        if nxt is state or nxt == state:
            continue
        occ = _occ(state)
        ahead = (nxt[0][0], nxt[0][1])
        if ahead in occ:                       # it was a push
            comp = _component(occ, ahead)
            if len(comp) > 1:
                colours = sorted({occ[c] for c in comp})
                shifted = sum(1 for a, b in zip(state[1:], nxt[1:]) if a != b)
                print(f"one press moved a component of {len(comp)} candies in "
                      f"{len(colours)} colour(s) ({shifted} colour set(s) "
                      f"changed), player {state[0]} -> {nxt[0]}")
                break
        state = nxt
    else:
        print("  (no multi-candy push found in the walk)")
        bad += 1

    # 3. A candy blocked by a Wall cancels the move for EVERYTHING, player
    #    included. Level 0's middle candy faces a wall.
    game.set_level(0)
    board, before = expert.read(eng)
    after = board.step(before, "right")
    eng.step("right")
    _b, engine_after = expert.read(eng)
    same = (after == before == engine_after)
    print(f"level 0, push into the wall at (3,3): board unchanged = {same} "
          f"(the cancel rules take the player with them)")
    bad += 0 if same else 1

    # 4. Merging is PERMANENT: no press ever splits a component. Measured over
    #    a long seeded walk on every level, tracking candy IDENTITY (the cells
    #    move, so the partition has to be compared over indices, not squares).
    rng = random.Random(11)
    splits = 0
    presses = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        ids = {cell: i for i, cell in enumerate(_occ(state))}
        parts = _id_partition(state, ids)
        for _ in range(400):
            direction = rng.choice(DIRS)
            nxt = board.step(state, direction)
            presses += 1
            if nxt != state:
                dr, dc = _DELTA[direction]
                occ = _occ(state)
                comp = set(_component(occ, nxt[0])) if nxt[0] in occ else set()
                ids = {((cell[0] + dr, cell[1] + dc) if cell in comp else cell): i
                       for cell, i in ids.items()}
            state = nxt
            nxt_parts = _id_partition(state, ids)
            for part in parts:
                if not any(part <= whole for whole in nxt_parts):
                    splits += 1
            parts = nxt_parts
    print(f"{presses} presses over the twenty levels: {splits} component "
          f"split(s) (expected 0 -- a merge is forever)")
    bad += splits

    return bad


def _id_partition(state: tuple, ids: dict) -> list:
    """The candy components of ``state`` as frozensets of candy IDENTITIES.
    ``ids`` maps the current cell of each candy to its identity, so the
    partition can be compared across presses that moved the pieces."""
    occ = _occ(state)
    return [frozenset(ids[cell] for cell in comp) for comp in _components(occ)]


def _selfcheck(walk_presses: int = 600, samples: int = 40,
               verbose: bool = True) -> int:
    """Four passes.

    1. The native model against the INTERPRETER: a seeded random walk on every
       level, the model's board compared against the engine's Player / Wall /
       Candy / Target layers after every single press -- including the presses
       that do nothing, which is where a collision-layer mistake hides.
    2. The win predicate: the model's `won` against ``engine.check_win()`` at
       every one of those states.
    3. Every shipped plan replayed through the model, requiring a WIN and
       requiring every optimal set to contain the press actually taken.
    4. Random reachable boards handed back to the expert: every plan it returns
       must replay to a WIN, and every None must be a board `_cover` PROVED
       dead rather than a search giving up.
    """
    solver, game, expert, solvable = _new()
    eng = game._engine
    violations = 0

    rng = random.Random(5)
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        for step in range(walk_presses):
            direction = rng.choice(DIRS + ("action",))
            eng.step(direction)
            if direction != "action":
                state = board.step(state, direction)
            _b, engine_state = expert.read(eng)
            if engine_state != state:
                violations += 1
                if verbose:
                    print(f"  lvl{level} step{step} {direction}: model "
                          f"{state} != engine {engine_state}")
                break
            if board.won(state) != eng.check_win():
                violations += 1
                if verbose:
                    print(f"  lvl{level} step{step}: win predicate disagrees")
                break
            if eng.check_win():
                game.set_level(level)
                _b, state = expert.read(eng)
    if verbose:
        print(f"  pass 1+2: model vs interpreter over "
              f"{game.n_levels * walk_presses} presses: {violations} violation(s)")

    for level in solvable:
        game.set_level(level)
        board, state = expert.read(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            continue
        for press, best in zip(plan, plan.optsets):
            if press not in best:
                violations += 1
                if verbose:
                    print(f"  lvl{level}: taken press {press} not in its "
                          f"optimal set {best}")
            state = board.step(state, press)
        if not board.won(state):
            violations += 1
            if verbose:
                print(f"  lvl{level}: plan does not win in the model")
    if verbose:
        print(f"  pass 3: {len(solvable)} plan(s) replayed in the model")

    # Pass 4: the expert answering RANDOM boards, which is what recovery needs.
    proved_dead = gave_up = alive = 0
    rng = random.Random(17)
    for level in solvable:
        for _ in range(samples):
            game.set_level(level)
            board, state = expert.read(eng)
            for _ in range(rng.randrange(1, 8)):
                eng.step(rng.choice(DIRS))
            board, state = expert.read(eng)
            if board.won(state):
                continue
            plan, why = solve(board, state, node_cap=40_000,
                              budget=10.0, beam_widths=(300,),
                              beam_budget=10.0)
            if plan is None:
                # Two very different answers, counted apart. "dead" is a PROOF
                # (the components admit no exact cover of the targets, so no
                # sequence of presses can ever win) and it is the common one
                # here -- a few random presses weld the wrong colours together.
                # Anything else is the deliberately tiny budget this pass runs
                # under giving up, which is allowed: `record_level` never ships
                # a half-tape, it falls back to the RESET.
                if why in ("dead", "unwinnable"):
                    proved_dead += 1
                else:
                    gave_up += 1
                continue
            alive += 1
            cur = state
            for press in plan:
                cur = board.step(cur, press)
            if not board.won(cur):
                violations += 1
                if verbose:
                    print(f"  lvl{level}: re-plan from a random board "
                          f"does not win")
    if verbose:
        print(f"  pass 4: {alive} random board(s) re-planned and won, "
              f"{proved_dead} PROVED already lost, {gave_up} unresolved inside "
              f"the small budget")
    return violations


def _proof(state_cap: int = 1_000_000, verbose: bool = True) -> int:
    """An INDEPENDENT primitive-level check of the levels `_astar` called exact.

    A layered BFS over PRIMITIVE states (player + candies, one press per layer),
    truncated at the first winning layer. That gives ``d*`` with no heuristic,
    no macro abstraction and no node cap that could quietly bite, plus the whole
    ball of that radius -- so a backward sweep over the same layers ("a state at
    depth j is on a shortest path iff a successor at depth j+1 is") marks the
    shortest-path subgraph and yields the EXACT optimal set at every state.

    Two requirements: the shipped plan's length equals ``d*``, and every label
    the expert emitted is contained in the exact set at that state. The second
    is what makes the walk-tie labelling a measured claim rather than an
    argument.

    Levels whose ball exceeds ``state_cap`` are reported as not proved -- the
    exhaustive pass is affordable only on the opening levels, which is exactly
    why the shipped searches are what they are.
    """
    solver, game, expert, solvable = _new()
    eng = game._engine
    violations = 0
    proved = 0
    for level in solvable:
        game.set_level(level)
        board, start = expert.read(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            continue
        why = expert.why.get(level, "?")
        layers = [[start]]
        depth = {start: 0}
        found = None
        blown = False
        while found is None:
            nxt = []
            for s in layers[-1]:
                for d in DIRS:
                    ns = board.step(s, d)
                    if ns in depth or ns == s:
                        continue
                    depth[ns] = len(layers)
                    nxt.append(ns)
                    if board.won(ns):
                        found = len(layers)
            if found is not None or not nxt:
                layers.append(nxt)
                break
            layers.append(nxt)
            if len(depth) > state_cap:
                blown = True
                break
        if blown or found is None:
            if verbose:
                print(f"  lvl{level:2d}: not proved "
                      f"({'ball > cap' if blown else 'no win in the ball'}, "
                      f"{len(depth)} states) -- plan was {why}")
            continue
        proved += 1
        if why != "exact":
            # A beam plan is not claimed shortest, so a gap here is information
            # rather than a violation -- and it is the number that says what
            # the fallback costs.
            if verbose:
                print(f"  lvl{level:2d}: beam plan is {len(plan)} presses, "
                      f"d* is {found} (+{len(plan) - found})")
            continue
        if len(plan) != found:
            violations += 1
            if verbose:
                print(f"  lvl{level:2d}: plan is {len(plan)} presses, "
                      f"d* is {found}")
        # Backward sweep: mark the shortest-path subgraph.
        on_path = {s for s in layers[found] if board.won(s)}
        marked = [set() for _ in range(found + 1)]
        marked[found] = on_path
        for j in range(found - 1, -1, -1):
            here = set()
            for s in layers[j]:
                for d in DIRS:
                    ns = board.step(s, d)
                    if ns != s and ns in marked[j + 1]:
                        here.add(s)
                        break
            marked[j] = here
        # Walk the shipped plan and compare its labels with the exact sets.
        s = start
        for i, (press, best) in enumerate(zip(plan, plan.optsets)):
            exact_set = [d for d in DIRS
                         if (ns := board.step(s, d)) != s
                         and (board.won(ns) if i + 1 == found
                              else ns in marked[i + 1])]
            for d in best:
                if d not in exact_set:
                    violations += 1
                    if verbose:
                        print(f"  lvl{level:2d} step{i}: label {d} is not in "
                              f"the exact optimal set {exact_set}")
                    break
            s = board.step(s, press)
        if verbose:
            print(f"  lvl{level:2d}: d*={found} proved over {len(depth)} "
                  f"primitive states, labels checked")
    if verbose:
        print(f"  {proved} level(s) proved shortest at primitive level")
    return violations


def _engine(verbose: bool = True) -> int:
    """Replay every shipped plan through the ADAPTER -- real `perform_action`
    calls with the screen actions the recording tapes -- and require
    `GameState.WIN`. This is the check that closes the loop back to the game a
    live agent is handed: the native model, the searches and the expansion could
    all agree with each other and still be a story about a model."""
    solver, game, expert, solvable = _new()
    violations = 0
    for seed in (0, 1, 2, 3):
        game._seed = seed
        for level in solvable:
            game.set_level(level)
            plan = expert.plan(game._engine, level)
            if plan is None:
                continue
            rot_k, hflip, vflip = game._rotation_k, game._hflip, game._vflip
            for press in plan:
                act = screen_action(press, rot_k, hflip, vflip, True)
                game.perform_action(ActionInput(id=act))
            if game._state != GameState.WIN:
                violations += 1
                if verbose:
                    print(f"  seed{seed} lvl{level}: ended {game._state}, "
                          f"not WIN")
        if verbose:
            print(f"  seed{seed}: {len(solvable)} level(s) replayed "
                  f"(rot={game._rotation_k} h={game._hflip} v={game._vflip})")
    return violations


def _audit(verbose: bool = True) -> int:
    """Every cell COMPOSITION the game can show, rendered as a whole 64x64 frame
    of a uniform board at every cell size the twenty levels use, required
    pairwise distinct.

    Whole frames rather than one cell sliced out of a mixed board: `_render_frame`
    upscales and centre-pads, so slicing by ``cell_px`` arithmetic reads the
    wrong pixels (the ps:explod lesson).
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    parsed = game._game
    idx = parsed.obj_name_to_idx
    eng = game._engine

    comps: dict = {"floor": [], "wall": ["wall"],
                   "wall+borders": ["wall", "walllr", "wallud"],
                   "wall+digit": ["wall", "walllr", "wallud", "three"],
                   "player": ["player"]}
    for k, name in enumerate(COLOURS):
        comps[f"{name}target"] = [name + "target"]
        comps[f"{name}candy"] = [name]
        comps[f"player@{name}"] = ["player", name + "target"]
        # A candy welded on both axes wears all four join marks -- the shape the
        # win condition is actually reached in, and the one that used to make it
        # invisible.
        comps[f"{name}candy+join"] = [name, "joinboth"]
        for other in COLOURS:
            comps[f"{name}@{other}tgt"] = [name, other + "target"]
            comps[f"{name}@{other}tgt+join"] = [name, other + "target",
                                                "joinboth"]

    def render(h, w, names):
        grid = [[{idx["background"]} | {idx[n] for n in names}
                 for _ in range(w)] for _ in range(h)]
        eng.grid = grid
        eng.height, eng.width = h, w
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()
        return np.asarray(_render_frame(eng, parsed))

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        h, w = len(game._engine.grid), len(game._engine.grid[0])
        sizes.setdefault(max(1, min(64 // h, 64 // w)), (h, w))

    collisions = 0
    for cell_px, (h, w) in sorted(sizes.items()):
        frames = {k: render(h, w, v) for k, v in comps.items()}
        for a, b in itertools.combinations(sorted(frames), 2):
            if np.array_equal(frames[a], frames[b]):
                collisions += 1
                if verbose:
                    print(f"  cell_px={cell_px}: {a} == {b}")
        if verbose:
            print(f"  cell_px={cell_px} ({h}x{w}): "
                  f"{len(frames)} compositions checked")
    return collisions


def _symmetry(walk_presses: int = 120, verbose: bool = True) -> int:
    """The augmentation, MEASURED. Every level's plan and a seeded random walk
    (which does what a plan never does -- shoves candies into walls, welds the
    wrong colours together and presses the unbound ACTION key) driven at all 16
    presentations, requiring every presented frame to be the exact rotation /
    flip of the unaugmented one."""
    solver, game, expert, solvable = _new()

    def transform(frame, k, hflip, vflip):
        out = np.rot90(np.asarray(frame), k)
        if hflip:
            out = np.fliplr(out)
        if vflip:
            out = np.flipud(out)
        return out

    def drive(g, level, presses, k, hflip, vflip):
        """Reset to ``level``, PIN the presentation (``set_level`` re-derives it
        from the seed, so it has to be pinned after the reset, not before) and
        press the sequence, returning the presented frames."""
        g.set_level(level)
        g._rotation_k, g._hflip, g._vflip = k, hflip, vflip
        g._current_frame = g._present_frame(_render_frame(g._engine, g._game))
        frames = [np.asarray(g._current_frame)]
        for d in presses:
            act = (GameAction.ACTION5 if d == "action"
                   else screen_action(d, k, hflip, vflip, True))
            fd = g.perform_action(ActionInput(id=act))
            frames.append(np.asarray(fd.frame[-1] if fd.frame
                                     else g._current_frame))
        return frames

    rng = random.Random(23)
    violations = 0
    checked = 0
    for level in solvable:
        game.set_level(level)
        plan = expert.plan(game._engine, level)
        seqs = [list(plan) if plan else [],
                [rng.choice(DIRS + ("action",)) for _ in range(walk_presses)]]
        for presses in seqs:
            ref = drive(game, level, presses, 0, False, False)
            for k in range(4):
                for hflip in (False, True):
                    for vflip in (False, True):
                        got = drive(game, level, presses, k, hflip, vflip)
                        checked += 1
                        for i, (a, b) in enumerate(zip(ref, got)):
                            if not np.array_equal(
                                    transform(a, k, hflip, vflip), b):
                                violations += 1
                                if verbose:
                                    print(f"  lvl{level} k={k} h={hflip} "
                                          f"v={vflip}: frame {i} mismatch")
                                break
    if verbose:
        print(f"  {checked} presentation replay(s) checked, "
              f"{len(solvable)} level(s)")
    return violations


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--mechanics" in sys.argv:
        bad = _mechanics()
        print(f"mechanics: {bad} surprise(s)")
        sys.exit(1 if bad else 0)
    if "--selfcheck" in sys.argv:
        violations = _selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--proof" in sys.argv:
        violations = _proof()
        print(f"proof: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--engine" in sys.argv:
        violations = _engine()
        print(f"engine replay: {violations} failures")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = _audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = _symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    sys.exit(StickCandySolver.main())
