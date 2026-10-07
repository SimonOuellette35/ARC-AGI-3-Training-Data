"""Generate Phase-1 training data for the PuzzleScript game ps:rainbow_apples
("Rainbow Apples", Alexander In Uganda -- a sokoban whose crates are REPAINTED
by the floor they are shoved across, plus a mirror twin that walks backwards).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
its disk cache, and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: a native model of the thirty-seven rules, the searches
that plan over it, and the checks that pin both to the interpreter.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema::

    {
      "game_id": "puzzlescript_rainbow_apples",
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
exactly. Every step also carries a set of equally-optimal presses (see
"Optimal-action sets").

The game
--------
Push five colours of apple onto five colours of target::

    WINCONDITIONS: All RedTarget on RedApple ... (one per colour)

Five mechanic facts, all MEASURED against the interpreter (`--selfcheck` states
each one as a board, `--fuzz` proves the model equals the interpreter over every
reachable state of eight of the ten levels):

  * **A push carries the whole SAME-COLOURED run.** ``[ > RegPlayer | RedApple ]``
    hands the force to the one apple the player walks into, whatever its colour;
    the five chain rules ``[ > RedApple | RedApple ]`` then propagate it, and each
    is written for ONE colour. So ``Prrr`` moving right shifts three apples, and
    ``Prb`` is a total no-op -- the red cannot enter the blue's cell, so the
    player behind it cannot move either. That is the difference between this game
    and a plain sokoban: a run of one colour is a single rigid piece, and a wall
    of a DIFFERENT colour is a wall.
  * **Paint repaints whatever stands on it, at the end of the turn.** Twenty
    ``late`` rules, one per (apple colour, paint colour) ordered pair. Their net
    effect is exactly "an apple standing on a paint takes the paint's colour",
    including the game's one bug: ``RedApple`` on ``PurplePaint`` is written to
    become BLUE, but the ``BlueApple + PurplePaint -> PurpleApple`` rule is later
    in the same list and fires in the same turn, so a red shoved onto purple
    paint comes out purple after all. The paint is not consumed and nothing stops
    an apple crossing three of them; only the LAST one counts.
  * **Both players are handed the pressed force, and the mirror rule takes it
    back.** ``Player = RegPlayer or NegaPlayer``, so `PSEngine.step` gives the
    press to BOTH; the last rule, ``[ > RegPlayer ] [ NegaPlayer ] -> [ >
    RegPlayer ] [ < NegaPlayer ]``, is what turns the twin around.
  * **...so a RegPlayer facing a WALL makes the twin an ordinary player.** The
    first rule, ``[ > Player | Wall ] -> [ Player | Wall ]``, cancels the force of
    a player walking into a wall; the mirror rule then has no ``> RegPlayer`` to
    match, never fires, and NegaPlayer keeps the force the engine gave it -- so it
    moves WITH the press instead of against it. Parking Reg against a wall is
    therefore the twin's steering wheel, and level 9 cannot be solved without it.
    (Being blocked by an APPLE does not do this: no rule cancels that force, the
    mirror fires as usual, and only the movement fails.)
  * **The twin pushes the apple BEHIND it.** The five ``[ > NegaPlayer | Apple ]``
    rules run before the mirror rule, i.e. while NegaPlayer still carries the
    PRESSED direction, so the apple that receives a force is the one on the
    pressed side -- the side the twin is about to walk away from. An apple in the
    twin's actual path is never given a force at all and simply blocks it.

`--selfcheck` also pins the two facts that keep the state space finite: ACTION is
a total no-op (no rule in the game mentions it, and the ``late`` rules are all
conditioned on an apple standing on a paint, so there is nothing for a wait to
tick), and the game contains no ``restart``/``cancel``/``again`` (see the
`ps:impasse` note in [[ps-astar-generator-family]] -- ``-> restart`` is executed
by the ADAPTER, not by ``PSEngine.step``, so a model built on ``eng.step`` would
plan straight through it).

The levels
----------
Ten shipped boards, none authored, none skipped::

    lvl  size   apples targets paints  d*   ties  how it was solved
    0    7x 9    1      1       0        7   14%  exact field (1190 states)
    1    7x 9    2      2       0       22    9%  exact field (39262)
    2    3x 9    1      1       1        5    0%  exact field (21)
    3    7x 9    1      1       1       13    0%  exact field (2346)
    4    7x 9    2      2       2       31   13%  exact field (252625)
    5    7x11    3      3       3       60   15%  macro A*, proved shortest
    6    8x 9    7      2       0        7    0%  exact field (83)
    7    7x 8    3      3       1       25    0%  exact field (422)
    8   12x13    6      6       0      189    6%  beam, NOT proved shortest
    9    7x 9    2      2       0       10   20%  exact field (204), the twin

The three that make the game a game:

  * **level 5** is the paint puzzle. The blue target sits beside the red paint,
    so an apple delivered to it along that row arrives RED; the blue has to be
    made at the blue paint six cells away and brought in from ABOVE, through the
    pink target's cell. The red for the red target has to be made on that same
    red paint and walked back out. 60 presses, and the whole reachable space is
    far too large to close (over 2 000 000 states), so this one is planned by the
    macro A* below and PROVED shortest by it.
  * **level 6** looks like the big one (seven apples) and has the smallest space
    in the game (83 states): every apple sits in a one-cell corridor, so there is
    exactly one cell to push each run from and almost nothing is reachable. Seven
    presses, two of which are pushes -- a column of three reds and a row of three
    blues, each shifted one cell onto its target.
  * **level 9** is the twin, and the mirror rule above is the entire puzzle. Its
    ten presses are five LEFTs and five RIGHTs: Reg starts against the left wall,
    so each LEFT is cancelled for it, the mirror rule never fires, and the twin
    walks left and shoves the blue apple onto the blue target. Then the five
    RIGHTs walk Reg's pink apple onto the pink target while the mirror IS firing
    -- the twin is pushed back toward the apple it just parked, and is held off it
    by the one rule that gives an apple in the twin's path no force at all.

Level 8 is a six-crate sokoban in a maze and is the one level here whose plan is
not proved shortest -- see "Expert solver".

Expert solver
-------------
`_Board` is the native model: a state is ``(packed, reg, nega)``, where ``packed``
holds each free cell's 3-bit apple colour (0 = empty, 1..5 = the five colours) at
bit ``3 * i`` and the two players are indices into the same list of free cells.
Walls are the cells absent from ``free``, so "off the board" and "into a wall"
are the same lookup miss (the ONE place they differ is the rule that cancels a
player's force, which names a Wall OBJECT -- ``wall_at`` keeps them apart).
Targets and paints never move and are not part of the state.

Three tiers, tried in order, so every level is solved by the strongest method
that fits it:

  1. **The exact field** (`_Board.field`), the ps:one_way_street / ps:palette
     shape: a forward BFS from the start closes the reachable set recording
     predecessors, then a backward BFS from every reachable WINNING state gives
     the exact distance-to-win of every state. Plans are proved shortest and the
     optimal-action SETS are MEASURED (at distance ``d``, a press is optimal iff
     it reaches ``d - 1``). Eight of the ten levels close under `FIELD_CAP`; the
     largest is level 4 at 252 625 states. A reverse pull-BFS from the win is not
     available: repainting is irreversible and ``All RedTarget on RedApple`` with
     spare apples (level 6 is seven apples for two targets) has no unique goal
     configuration to seed from.
  2. **Macro A-star** (`_Board.astar`) for the levels whose space will not close --
     successors are ``walk to the push cell, then push`` macros, ``g`` counts
     PRIMITIVE presses, and a win is kept as an incumbent until the frontier's
     ``f`` reaches its cost, which at ``weight == 1`` with an admissible
     heuristic is a proof of shortest (the `exact_goal_test` lesson: a macro is
     ``walk + act``, so the first win GENERATED is not the shortest one). The
     heuristic is a colour-aware, TURN-COSTED push distance -- see
     `_Board._table`. Level 5 closes at 444k macros and proves d* = 60 (~25s for
     the A* itself, ~47s counting the field attempt that has to fail first and
     the labelling sweep that follows).
  3. **A width-capped beam** (`_Board.beam`) over the same macros, for level 8,
     where the admissible heuristic is ~2.7x short (70 against 189) and A* has
     nothing to steer with. Its plan wins and is certified on the interpreter but
     is NOT proved shortest; it is then shortened by deleting engine-verified
     blocks longest-first.

Optimal-action sets
-------------------
Every recorded step ships a set of equally-shortest presses, and no step ships
unlabelled (the always-emit-optimal-targets rule). Where they come from depends
on the tier, and the report says so per level:

  * **tier 1 (the field): MEASURED and complete.** Every press that reaches a
    state one closer to the win is in the set -- ties between two different
    pushes, and between "walk now, push later" and its reverse, included.
  * **tier 2 (macro A*): MEASURED and complete as well**, by a bounded layered
    sweep once ``d*`` is proved (`_Board.label_exact`, the ps:pegs labelling):
    keep every state with ``g + h <= d*``, record the edges, and run the same
    backward pass over that subgraph. Nothing on a shortest path can be missing
    from it, because an admissible ``h`` never overestimates what a shortest
    continuation costs. Level 5 labels this way, so the only level here with
    inferred sets is level 8.
  * **tier 3 (beam): INFERRED for the walks, forced for the pushes**
    (`_Board.annotate`, the `PSPushExpert.annotate_walks` argument on the model
    instead of the interpreter). No rule in this game fires on a bare move onto
    an empty cell, so any interleaving of the two axes that stays on a shortest
    route to the same push cell reaches an IDENTICAL state in the same number of
    presses, and is therefore equally good. Sound but not complete: a tie
    between two different pushes is not found. It is the honest label for level
    8, whose plan is not shortest to begin with -- "optimal" there means "on a
    shortest route to the push this plan makes next", not "on a shortest route
    to the win".

Checks
------
``--plans`` reports the table above and CERTIFIES every plan by replaying it on
the real interpreter, requiring the win on the LAST press and no earlier.

``--fuzz`` asserts `_Board` reproduces the interpreter. For the eight levels
whose space closes it is EXHAUSTIVE -- every reachable state is seated back into
the engine grid (`_seat`) and all four presses are compared, board and win flag
alike; for levels 5 and 8 it random-walks from a COLD start (no warm-up press --
see the ps:idols_to_the_burnt_god note about turn-one bookkeeping). Measured:
**1 216 612 transitions, 0 mismatches**. Coverage counters are printed per
outcome, because "the model agrees" is worth nothing if the run never repainted
an apple or never turned the twin around -- and here they come out as 947 362
walks, 224 384 refused presses, 42 373 single pushes, 157 chain pushes, 2002
repaints, and on level 9 the three twin branches (232 walks, 47 pushes, and 55
of the one that matters, Reg against a wall leaving the twin to follow the
press).

``--verify`` is the independent check of the SEARCH, which the fuzz cannot reach
(it only ever exercises `step`). For a field level: the backward field recomputed
by layer scan with no predecessor map, the double-entry formula against the
forward field, and every labelled press spliced into a variant plan and replayed
on the INTERPRETER. For a searched level: every labelled press spliced in the
same way, requiring a win at the plan's own length.

``--selfcheck`` states the five mechanic facts above (and the two negatives) as
hand-built boards.

``--symmetry`` proves on the interpreter that the 8-element presentation group is
an exact symmetry of the mechanic, which is what entitles the game to
`PuzzleScriptAdapter._FLIP_GAMES`: every level's plan replayed on all eight
turned and mirrored boards, and then every reachable state and press of the six
levels small enough to sweep -- level 9, the only board where two movers can
contest a cell, included.

``--audit`` renders every cell COMPOSITION at every cell size the levels use and
asserts pairwise distinctness. It FOUND one bug, now fixed in
``data/puzzlescript_games/Rainbow_Apples.txt``: the paint sprite carried its
colour in the middle three rows, exactly where both the apple and the player are
opaque, so a piece standing on a paint hid WHICH paint it was -- all five
rendered identically at every cell size. The sprite is now a 3x3 lattice of
coloured pixels on the grey field, whose corners and edge midpoints fall in the
holes of both bodies (and survive the renderer's centred sampling at cell_px 3,
4 and 5); the paint under a standing player now reads 5.9-28 px/cell apart.

All six are clean as shipped: 10/10 plans CERTIFIED (369 presses), 0 selfcheck
violations, 0 fuzz mismatches, `--verify`'s 399 labelled presses all replayed to
a win at exactly ``d*``, 0 chiral presentations, 0 indistinguishable
compositions. Three seeds recorded end to end win all 10 levels each, label every
one of their 1107 expert steps, and re-generate byte-identically.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the presentation
augmentations: rotation_k in {0,1,2,3} plus an independent horizontal and
vertical flip, each with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`, which this game was added to). 10 levels x 16
presentations = 160. There is deliberately no colour augmentation (the game is
not in ``_RECOLOR_GAMES`` and must not be): which apple matches which target, and
which paint makes which colour, IS the puzzle.

The expert plan is therefore seed-independent: solved once per level, cached to
``data/rainbow_apples_plans.json``, and replayed per seed with that seed's
remapped screen actions.

Usage (run from the repo root):
    python solvers/generate_rainbow_apples_training.py \
        --episodes 200 --out data/training_multi_level/rainbow_apples

    python solvers/generate_rainbow_apples_training.py --plans
    python solvers/generate_rainbow_apples_training.py --selfcheck
    python solvers/generate_rainbow_apples_training.py --fuzz
    python solvers/generate_rainbow_apples_training.py --verify
    python solvers/generate_rainbow_apples_training.py --symmetry
    python solvers/generate_rainbow_apples_training.py --audit
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame                # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert, Plan      # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Rainbow_Apples"
GAME_MODULE = "ps:rainbow_apples"

#: Disk cache of each level's start plan AND its optimal-action sets. Level 5's
#: A* is ~25s and level 8's beam ~70s, so without a file on disk every shard
#: `parallelize_generator` starts would re-derive them. Delete it to re-derive.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "rainbow_apples_plans.json"

#: Engine direction -> (dr, dc).
_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}
#: Fixed iteration order, so a plan (and its tie sets) is reproducible.
_ORDER: tuple[str, ...] = ("up", "down", "left", "right")
_OPP: dict[str, str] = {"up": "down", "down": "up", "left": "right",
                        "right": "left"}

#: The five colours, in the order their 1..5 codes are assigned. 0 means "no
#: apple"; three bits per cell, so a whole board packs into one int.
COLOURS: tuple[str, ...] = ("red", "blue", "green", "purple", "pink")
_CODE: dict[str, int] = {name: i + 1 for i, name in enumerate(COLOURS)}

#: Refuse to close a reachable set larger than this, and fall through to the
#: macro A*. It is a MEMORY budget, not a time one (each state is a dict entry
#: plus its predecessor list). The largest level that DOES close is level 4 at
#: 252 625 states; levels 5 and 8 hit the cap in ~15s and ~30s respectively,
#: which is the price of finding out, paid once and cached to disk.
FIELD_CAP = 600_000

#: Macro budget for `_Board.astar` (GENERATED macros, not expansions -- a dense
#: board generates ~40 successors per expansion, so an expansion cap would let
#: peak memory vary by that factor between levels). Level 5 proves d* = 60 at
#: 444k; level 8 is hopeless at any budget (its admissible estimate is short by
#: a factor of ~2.7), so this is sized at ~2x what the level that CAN be proved
#: needs, and the ~90s it spends failing on level 8 is the price of finding out
#: -- paid once, then cached to disk.
ASTAR_CAP = 1_000_000

#: `_Board.beam`: states kept per macro layer, and the layer budget.
BEAM_WIDTH = 1500
BEAM_DEPTH = 90

#: `--symmetry`'s second stage sweeps every reachable state of a level on all
#: eight presentations, at 32 interpreter steps per state. This is the level
#: size at which that stops being a few seconds; it keeps levels 0, 2, 3, 6, 7
#: and -- the one that matters -- 9, the only board where two movers can contest
#: a cell.
SYMMETRY_STATE_CAP = 5_000

#: Cap on the bounded layered sweep that MEASURES optimal sets for a searched
#: level (`_Board.label_exact`). Above it the labels fall back to `annotate`.
LABEL_CAP = 400_000

_INF = 1 << 20


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """Rainbow Apples over the level's free cells.

    A state is ``(packed, reg, nega)``: ``packed`` holds each free cell's 3-bit
    apple colour at bit ``3 * i``, and the players are indices into ``free``
    (``nega`` is -1 on the nine levels that have no twin).

    `step` is the game; `field` inverts it exactly, `astar` searches it, and
    `beam` is the fallback for the one level neither can reach.
    """

    def __init__(self, walls, paint, targets, apples, reg, nega, h, w):
        self.h, self.w = h, w
        self.free = [(r, c) for r in range(h) for c in range(w)
                     if (r, c) not in walls]
        self.idx = {cell: i for i, cell in enumerate(self.free)}
        self.n = len(self.free)
        self.paint_of = [paint.get(cell, 0) for cell in self.free]
        self.targets = tuple(sorted((self.idx[cell], col)
                                    for cell, col in targets.items()))
        # ``win`` is ``(packed & tsel) == tval``: tsel is 0b111 in every target
        # cell's slot, tval the colour each of them demands.
        self.tsel = self.tval = 0
        for i, col in self.targets:
            self.tsel |= 7 << (3 * i)
            self.tval |= col << (3 * i)
        packed = 0
        for cell, col in apples.items():
            packed |= col << (3 * self.idx[cell])
        self.has_nega = nega is not None
        self.start = (packed, self.idx[reg],
                      self.idx[nega] if nega is not None else -1)
        self.nxt = {d: [self.idx.get((r + dr, c + dc), -1)
                        for (r, c) in self.free]
                    for d, (dr, dc) in _DELTA.items()}
        # A neighbour that is off the BOARD is not the same as a neighbour that
        # holds a Wall: ``[ > Player | Wall ]`` names the object, so only the
        # latter cancels a player's force -- and whether Reg's force survives is
        # exactly what decides which way the twin walks (see the module
        # docstring). Both are ``nxt < 0``, so they are told apart here.
        self.wall_at = {
            d: [(0 <= r + dr < h and 0 <= c + dc < w
                 and (r + dr, c + dc) in walls)
                for (r, c) in self.free]
            for d, (dr, dc) in _DELTA.items()}
        self._tables: dict = {}

    # -- reading the interpreter ---------------------------------------------
    @classmethod
    def read(cls, eng, ids) -> "_Board":
        """Build the model from the interpreter's current grid."""
        wall_ids, apple_col, paint_col, tgt_col, reg_id, nega_id = ids
        walls, paint, targets, apples = set(), {}, {}, {}
        reg = nega = None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & wall_ids:
                    walls.add((r, c))
                    continue            # a wall cell can hold nothing else
                for obj in cell:
                    if obj in apple_col:
                        apples[(r, c)] = apple_col[obj]
                    elif obj in paint_col:
                        paint[(r, c)] = paint_col[obj]
                    elif obj in tgt_col:
                        targets[(r, c)] = tgt_col[obj]
                    elif obj == reg_id:
                        reg = (r, c)
                    elif obj == nega_id:
                        nega = (r, c)
        for cell, col in apples.items():
            # An apple standing on a paint of another colour is a state the
            # `late` rules cannot leave behind, so a level that SEATED one would
            # be a board the model has never had to represent. None does; say so
            # loudly rather than silently modelling a fiction.
            assert paint.get(cell, col) == col, f"apple on foreign paint at {cell}"
        return cls(walls, paint, targets, apples, reg, nega,
                   eng.height, eng.width)

    # -- dynamics -------------------------------------------------------------
    def win(self, state) -> bool:
        """``All RedTarget on RedApple`` and its four siblings: every TARGET cell
        carries an apple of its own colour. Note the direction -- spare apples
        anywhere else are free, and two red targets need two red apples."""
        return state[0] & self.tsel == self.tval

    def _run(self, packed, x, nxt) -> list[int]:
        """The maximal SAME-COLOURED run of apples starting at cell ``x``.

        The five chain rules are each written for one colour, so the force stops
        at the first apple of a different one -- which then blocks the run, and
        with it the player."""
        col = (packed >> (3 * x)) & 7
        run = [x]
        while True:
            y = nxt[run[-1]]
            if y < 0 or (packed >> (3 * y)) & 7 != col:
                return run
            run.append(y)

    def step(self, state, d):
        """One press. Returns the new state, or ``state`` itself when the turn
        resolves to nothing.

        The whole rule set in the order PuzzleScript applies it: the engine hands
        the press to BOTH player objects (``Player`` is an or-group); the wall
        rule takes it back from whichever of them faces a Wall; each player that
        still has one gives it to the apple ahead of it and the chain rules
        extend that to the apple's same-coloured run; the mirror rule reverses
        NegaPlayer's force, but only if RegPlayer still has one to match; the
        forces resolve with players, apples and walls all on one collision layer;
        and finally the `late` paint rules repaint every apple standing on a
        paint."""
        packed, p, q = state
        nxt, wall = self.nxt[d], self.wall_at[d]
        regf = p >= 0 and not wall[p]
        negf = q >= 0 and not wall[q]

        move: dict[int, str] = {}
        for pos, has in ((p, regf), (q, negf)):
            if not has:
                continue
            x = nxt[pos]
            if x >= 0 and (packed >> (3 * x)) & 7:
                for cell in self._run(packed, x, nxt):
                    move[cell] = d
        # The mirror rule is the LAST main rule, so the twin's pushes above were
        # already handed out in the PRESSED direction -- which is why it shoves
        # the apple behind it and is stopped by the one in front.
        if regf:
            move[p] = d
            if q >= 0:
                # The RHS assigns a fresh force, so the twin turns round even
                # when the wall rule had just taken its own away.
                move[q] = _OPP[d]
        elif negf:
            move[q] = d

        if not move:
            return state
        # Resolve. Everything lives on one collision layer, so a mover is refused
        # when its destination is off the board, holds something that is not
        # moving out of the way, or is wanted by a second mover (measured: the
        # interpreter refuses BOTH). Head-on swaps are refused for the same
        # reason -- neither vacates first.
        occupied = {i for i in range(self.n) if (packed >> (3 * i)) & 7}
        if p >= 0:
            occupied.add(p)
        if q >= 0:
            occupied.add(q)
        dest = {c: self.nxt[dd][c] for c, dd in move.items()}
        blocked: set[int] = set()
        changed = True
        while changed:
            changed = False
            live = [c for c in move if c not in blocked]
            wanted: dict[int, int] = {}
            for c in live:
                wanted[dest[c]] = wanted.get(dest[c], 0) + 1
            for c in live:
                t = dest[c]
                if (t < 0 or wanted[t] > 1
                        or (t in occupied and (t not in move or t in blocked))
                        or (t in move and t not in blocked and dest[t] == c)):
                    blocked.add(c)
                    changed = True
        final = {c: dest[c] for c in move if c not in blocked}
        if not final:
            return state

        npacked = 0
        for i in range(self.n):
            col = (packed >> (3 * i)) & 7
            if col:
                npacked |= col << (3 * final.get(i, i))
        # `late`: a paint repaints whatever is standing on it, every turn.
        for i in range(self.n):
            pc = self.paint_of[i]
            if pc and (npacked >> (3 * i)) & 7:
                npacked = (npacked & ~(7 << (3 * i))) | (pc << (3 * i))
        return (npacked, final.get(p, p), final.get(q, q) if q >= 0 else -1)

    # -- tier 1: the exact distance-to-win field ------------------------------
    def field(self, cap: int = FIELD_CAP) -> tuple[dict, dict]:
        """``(back, fwd)`` -- distance to the win and distance from the start,
        for every state reachable from the level start. Raises `MemoryError`
        past ``cap``.

        A forward closure recording predecessors, then a backward BFS from every
        reachable WINNING state over them. Exact in PRIMITIVE presses (the unit
        the agent pays) with no canonicalisation: every shortest path out of the
        start stays inside the reachable set, so restricting the backward pass to
        that set cannot shorten one.

        Dead states need no test -- an apple repainted into a colour no target
        wants, or shoved into a corner, still enters the closure, it just never
        receives a distance."""
        start = self.start
        pred: dict = {}
        fwd = {start: 0}
        queue: deque = deque([start])
        while queue:
            state = queue.popleft()
            d = fwd[state] + 1
            for direction in _ORDER:
                nxt = self.step(state, direction)
                if nxt == state:
                    continue                       # refused: a self-loop
                pred.setdefault(nxt, []).append(state)
                if nxt not in fwd:
                    fwd[nxt] = d
                    queue.append(nxt)
            if len(fwd) > cap:
                raise MemoryError(f"reachable set exceeded {cap} states")
        back: dict = {}
        queue = deque()
        for state in fwd:
            if self.win(state):
                back[state] = 0
                queue.append(state)
        while queue:
            state = queue.popleft()
            d = back[state] + 1
            for prev in pred.get(state, ()):
                if prev not in back:
                    back[prev] = d
                    queue.append(prev)
        return back, fwd

    def field_plan(self, back: dict):
        """``(presses, optsets)`` walked downhill from the start, or
        ``(None, None)``.

        At every state the optimal SET is exactly the presses reaching distance
        ``d - 1``: a refused press leaves the distance unchanged and is never in
        it, and a press onto a state the field does not hold is a press that
        threw the win away."""
        state = self.start
        if state not in back:
            return None, None
        presses, optsets = [], []
        while back[state]:
            want = back[state] - 1
            best = [a for a in _ORDER
                    if back.get(self.step(state, a), -1) == want]
            presses.append(best[0])
            optsets.append(best)
            state = self.step(state, best[0])
        return presses, optsets

    # -- the heuristic: colour-aware, turn-costed push distance ---------------
    @staticmethod
    def _turn(a: str, b: str) -> int:
        """A LOWER bound on the presses the player spends moving from "behind the
        apple for a push in direction ``a``" to "behind it for ``b``". The apple
        itself is in the way, so a reversal is at least three and a perpendicular
        swap at least two.

        This is the ps:pegs sharpening, and on this game it is the difference
        between a heuristic that closes level 5 in 25s and one that does not
        close it at all: an apple's route here bends around walls, and the
        repositioning between bends is most of what the plan costs."""
        if a == b:
            return 0
        (ar, ac), (br, bc) = _DELTA[a], _DELTA[b]
        return 3 if (ar == -br and ac == -bc) else 2

    def _table(self, ti: int, tcol: int) -> dict:
        """``{(cell, colour): presses}`` -- a lower bound on the presses needed to
        bring an apple that is at ``cell`` with colour ``colour`` onto target
        cell ``ti`` as colour ``tcol``, ignoring every other apple.

        A reverse Dijkstra over push edges in the ``(cell, colour, last push
        direction)`` graph. Two things make it more than a distance:

          * **colour travels with the apple.** Entering a painted cell forces the
            paint's colour, so the reverse edge into a painted cell accepts any
            source colour and the edge into a bare cell demands the same one --
            which is what makes "this apple can only satisfy that target by going
            the long way round, over the blue paint" a number rather than a
            story.
          * **the turns are charged** (`_turn`), so the estimate counts the
            player's repositioning as well as the pushes.

        A cell absent from the table is a cell no apple can ever reach that
        target from -- the sokoban deadlock test, for free."""
        dist: dict = {}
        pq: list = []
        for d in _ORDER:
            dist[(ti, tcol, d)] = 0
            heapq.heappush(pq, (0, ti, tcol, d))
        while pq:
            g, v, cv, dnext = heapq.heappop(pq)
            if dist.get((v, cv, dnext), _INF) < g:
                continue
            r, c = self.free[v]
            dr, dc = _DELTA[dnext]
            u = self.idx.get((r - dr, c - dc))          # where the apple was
            stand = self.idx.get((r - 2 * dr, c - 2 * dc))   # where the player was
            if u is None or stand is None:
                continue
            pv = self.paint_of[v]
            if pv:
                if cv != pv:
                    continue          # arriving on paint forces the paint's colour
                sources = range(1, 6)
            else:
                sources = (cv,)
            for cu in sources:
                if self.paint_of[u] and self.paint_of[u] != cu:
                    continue          # ... and so does standing on one
                for dprev in _ORDER:
                    ng = g + 1 + self._turn(dprev, dnext)
                    if ng < dist.get((u, cu, dprev), _INF):
                        dist[(u, cu, dprev)] = ng
                        heapq.heappush(pq, (ng, u, cu, dprev))
        flat: dict = {}
        for (cell, col, _d), g in dist.items():
            # The FIRST push of an apple has no incoming direction to turn from.
            if g < flat.get((cell, col), _INF):
                flat[(cell, col)] = g
        return flat

    def tables(self) -> dict:
        if not self._tables:
            self._tables = {(i, col): self._table(i, col)
                            for i, col in self.targets}
        return self._tables

    def heuristic(self, state) -> int:
        """A lower bound on the presses remaining, in the same unit as ``g``.

        The minimum-cost ASSIGNMENT of apples to the targets that are not yet
        satisfied (Hungarian over `_table`), plus the player's walk to the
        nearest apple. Admissible: in any winning continuation each bare target
        ends up carrying a DISTINCT apple, the presses spent on distinct apples
        are distinct presses, and each apple's relaxed cost is a lower bound on
        its own; a min-cost assignment is therefore no larger than the true
        total. `_INF` means some target has become unreachable for every apple,
        i.e. the state is dead."""
        packed, p = state[0], state[1]
        apples = [(i, (packed >> (3 * i)) & 7) for i in range(self.n)
                  if (packed >> (3 * i)) & 7]
        bare = [(i, col) for i, col in self.targets
                if ((packed >> (3 * i)) & 7) != col]
        if not bare:
            return 0
        if not apples:
            return _INF
        tables = self.tables()
        cost = np.full((len(bare), len(apples)), _INF, dtype=np.int64)
        for ti, key in enumerate(bare):
            table = tables[key]
            for ai, apple in enumerate(apples):
                d = table.get(apple)
                if d is not None:
                    cost[ti, ai] = d
        rows, cols = linear_sum_assignment(cost)
        total = int(cost[rows, cols].sum())
        if total >= _INF:
            return _INF
        pr, pc = self.free[p]
        walk = min(abs(pr - self.free[i][0]) + abs(pc - self.free[i][1])
                   for i, _col in apples)
        return total + max(0, walk - 1)

    # -- tiers 2 and 3: macro search ------------------------------------------
    def macros(self, state) -> list[list[str]]:
        """Every ``walk to the push cell, then push`` the player can reach right
        now, each as a flat list of primitive directions.

        A push is the only move with an effect in this game (no rule fires on a
        bare move onto an empty cell -- `--selfcheck` proves it), so branching on
        macros makes the search depth the number of PUSHES while ``g`` still
        counts the presses the agent pays. Only defined for the levels with no
        twin: a bare move is not inert when the twin is on the board."""
        assert not self.has_nega, "the twin makes a bare walk a move"
        packed, p = state[0], state[1]
        occupied = [bool((packed >> (3 * i)) & 7) for i in range(self.n)]
        parent: dict = {p: None}
        queue = deque([p])
        while queue:
            cur = queue.popleft()
            for d in _ORDER:
                nxt = self.nxt[d][cur]
                if nxt >= 0 and not occupied[nxt] and nxt not in parent:
                    parent[nxt] = (cur, d)
                    queue.append(nxt)

        def walk_to(cell) -> list[str]:
            out = []
            while parent[cell] is not None:
                cell, d = parent[cell]
                out.append(d)
            out.reverse()
            return out

        out = []
        for i in range(self.n):
            if not occupied[i]:
                continue
            r, c = self.free[i]
            for d, (dr, dc) in _DELTA.items():
                stand = self.idx.get((r - dr, c - dc))
                if stand is not None and stand in parent:
                    out.append(walk_to(stand) + [d])
        return out

    def _apply(self, state, macro):
        """``(end state, index of the primitive that WON or -1)``. The win is
        tested after EVERY primitive: a macro that wins partway and keeps
        stepping would silently un-win, and truncating there keeps the plan
        shortest."""
        for i, d in enumerate(macro):
            state = self.step(state, d)
            if self.win(state):
                return state, i
        return state, -1

    def astar(self, cap: int = ASTAR_CAP):
        """``(plan, closed)`` -- weight-1 A* over `macros`, ``g`` in presses.

        A win is kept as the INCUMBENT and the search runs until the frontier's
        ``f`` reaches its cost, so what comes back is proved shortest rather than
        merely first (`PSPushExpert.exact_goal_test`: a macro is ``walk + act``,
        so macros cost different amounts and the first win GENERATED can be
        beaten by a later node's shorter macro). ``closed`` is False when the cap
        stopped it, in which case the plan (if any) is a winning path with no
        optimality claim."""
        start = self.start
        if self.win(start):
            return [], True
        pq = [(self.heuristic(start), 0, 0, start, [])]
        best_g = {start: 0}
        counter = nodes = 0
        best_plan, best_cost = None, _INF
        while pq:
            f, g, _c, state, path = heapq.heappop(pq)
            if f >= best_cost:
                return best_plan, True        # nothing shorter can exist
            if best_g.get(state, _INF) < g:
                continue
            for macro in self.macros(state):
                nxt, won = self._apply(state, macro)
                nodes += 1
                if won >= 0:
                    if g + won + 1 < best_cost:
                        best_cost, best_plan = g + won + 1, path + macro[:won + 1]
                    continue
                if nxt == state:
                    continue                  # a push the engine refused
                ng = g + len(macro)
                if ng >= best_cost or best_g.get(nxt, _INF) <= ng:
                    continue
                h = self.heuristic(nxt)
                if h >= _INF:
                    continue                  # provably lost: drop un-expanded
                best_g[nxt] = ng
                counter += 1
                heapq.heappush(pq, (ng + h, ng, counter, nxt, path + macro))
                if nodes >= cap:
                    return best_plan, False
        return best_plan, True

    def beam(self, width: int = BEAM_WIDTH, depth: int = BEAM_DEPTH):
        """A width-capped breadth-first beam over the same macros, ranked by
        `heuristic`. For the level where the admissible estimate is short by a
        factor of ~2.7 and A* therefore has nothing to steer with; the plan wins
        but is not shortest."""
        frontier = [(self.start, [])]
        seen = {self.start}
        for _depth in range(depth):
            kids = []
            for state, path in frontier:
                for macro in self.macros(state):
                    nxt, won = self._apply(state, macro)
                    if won >= 0:
                        return path + macro[:won + 1]
                    if nxt == state or nxt in seen:
                        continue
                    h = self.heuristic(nxt)
                    if h >= _INF:
                        continue
                    seen.add(nxt)
                    kids.append((h, nxt, path + macro))
            if not kids:
                return None                   # the reachable space closed
            kids.sort(key=lambda kid: kid[0])
            frontier = [(s, p) for _h, s, p in kids[:width]]
        return None

    def shorten(self, plan: list[str]) -> list[str]:
        """Delete blocks of a beam plan, longest first, keeping every deletion
        that still wins. A beam plan wanders by construction; this is the cheap
        half of the fix, and every candidate is checked by re-simulating the
        whole plan rather than by reasoning about it."""
        plan = list(plan)
        changed = True
        while changed:
            changed = False
            n = len(plan)
            for length in range(n - 1, 0, -1):
                for i in range(0, n - length + 1):
                    cand = plan[:i] + plan[i + length:]
                    state = self.start
                    won = -1
                    for j, d in enumerate(cand):
                        state = self.step(state, d)
                        if self.win(state):
                            won = j
                            break
                    if won >= 0:
                        plan = cand[:won + 1]
                        changed = True
                        break
                if changed:
                    break
        return plan

    # -- optimal-action sets for a searched (non-field) level ------------------
    def label_exact(self, plan: list[str], cap: int = LABEL_CAP):
        """``(optsets, back)`` -- MEASURED optimal sets for a plan already PROVED
        shortest, and the restricted distance-to-win field they were read off.
        ``(None, None)`` if the sweep would exceed ``cap``.

        The ps:pegs labelling: sweep forward over PRIMITIVE presses keeping only
        states with ``g + h <= d*`` (admissible ``h``, so nothing on a shortest
        path is dropped), record the edges kept, then a backward pass over them
        gives exact distance-to-win for every state a shortest path can contain.
        The optimal set at each step is then the presses that go one closer, and
        it is complete -- unlike `annotate`, which can only find ties between
        walk steps."""
        star = len(plan)
        start = self.start
        fwd = {start: 0}
        pred: dict = {}
        frontier = [start]
        for depth in range(star):
            nxt_frontier = []
            for state in frontier:
                for a in _ORDER:
                    t = self.step(state, a)
                    if t == state:
                        continue
                    if depth + 1 + self.heuristic(t) > star:
                        continue
                    pred.setdefault(t, []).append(state)
                    if t not in fwd:
                        fwd[t] = depth + 1
                        nxt_frontier.append(t)
            frontier = nxt_frontier
            if len(fwd) > cap:
                return None, None
        back: dict = {}
        queue: deque = deque()
        for state in fwd:
            if self.win(state):
                back[state] = 0
                queue.append(state)
        while queue:
            state = queue.popleft()
            d = back[state] + 1
            for prev in pred.get(state, ()):
                if prev not in back:
                    back[prev] = d
                    queue.append(prev)
        if back.get(start) != star:
            return None, None                 # the bound cut a shortest path
        state = start
        optsets = []
        for i, taken in enumerate(plan):
            want = star - i - 1
            best = [a for a in _ORDER
                    if back.get(self.step(state, a), -1) == want]
            optsets.append(best if taken in best else [taken])
            state = self.step(state, taken)
        return optsets, back

    # -- rebuilding a plan around a labelled press (for `--verify`) -----------
    def downhill(self, back: dict, state) -> list[str] | None:
        """An optimal continuation from ``state``, read off a distance-to-win
        field (`field`'s, or `label_exact`'s restricted one)."""
        if state not in back:
            return None
        out = []
        while back[state]:
            want = back[state] - 1
            nxt = next((a for a in _ORDER
                        if back.get(self.step(state, a), -1) == want), None)
            if nxt is None:
                return None
            out.append(nxt)
            state = self.step(state, nxt)
        return out

    def splice_walk(self, plan: list[str], i: int, alt: str) -> list[str] | None:
        """``plan`` with step ``i`` replaced by an inferred walk alternative and
        the rest of that walk RUN re-routed to the cell it was heading for.

        This is how a label from `annotate` is certified when there is no field
        to plan a continuation from: the claim being made is exactly "this press
        also lies on a shortest route to the same push cell", so the check is to
        take it and finish the walk, and require the interpreter to win at the
        same press count."""
        state = self.start
        states = [state]
        for d in plan:
            state = self.step(state, d)
            states.append(state)
        run_end = i
        while run_end < len(plan) and states[run_end + 1][0] == states[run_end][0]:
            run_end += 1                       # the first step that pushes
        if run_end >= len(plan):
            return None
        stand = states[run_end][1]
        after = self.step(states[i], alt)
        dist = self._walk_distances(states[i], stand)
        detour: list[str] = []
        cell = after[1]
        while cell != stand:
            d0 = dist.get(cell)
            nxt = next((d for d in _ORDER
                        if dist.get(self.nxt[d][cell], -1) == d0 - 1), None)
            if nxt is None:
                return None
            detour.append(nxt)
            cell = self.nxt[nxt][cell]
        return plan[:i] + [alt] + detour + plan[run_end:]

    def annotate(self, plan: list[str]) -> list[list[str]]:
        """INFERRED optimal sets: every walk step is labelled with each direction
        that keeps it on a shortest route to the same push cell, every push with
        itself alone.

        The `PSPushExpert.annotate_walks` argument, applied to the model. A walk
        is inert here -- no rule fires on a bare move onto an empty cell -- so
        any interleaving of the two axes that reaches the same cell in the same
        number of presses leaves an IDENTICAL board, and labelling one arbitrary
        interleaving as the only right answer would train the policy to a coin
        flip it cannot win. Sound but not complete: a tie between two different
        PUSHES is not found, which is why the field levels do not use this."""
        assert not self.has_nega, "the twin makes a bare walk a move"
        steps = []                            # (kind, cell, direction, state)
        state = self.start
        for d in plan:
            nxt = self.step(state, d)
            kind = ("push" if nxt[0] != state[0]
                    else "walk" if nxt[1] != state[1] else "stuck")
            steps.append((kind, state[1], d, state))
            state = nxt

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
                # A trailing walk: this game always wins ON a push, so there is
                # no destination to measure against. Label what the expert did.
                for j in range(run, i):
                    out[j] = [steps[j][2]]
                continue
            stand = steps[i][1]
            dist = self._walk_distances(steps[run][3], stand)
            for j in range(run, i):
                here = steps[j][1]
                d0 = dist.get(here)
                alts = [] if d0 is None else [
                    d for d in _ORDER
                    if dist.get(self.nxt[d][here], -1) == d0 - 1]
                out[j] = alts if steps[j][2] in alts else [steps[j][2]]
        return out

    def _walk_distances(self, state, target: int) -> dict:
        """BFS distance to ``target`` over the cells the player may walk on: no
        wall, no apple (walking into one is a push, not a walk)."""
        packed = state[0]
        dist = {target: 0}
        queue = deque([target])
        while queue:
            cur = queue.popleft()
            for d in _ORDER:
                nxt = self.nxt[d][cur]
                if nxt >= 0 and nxt not in dist and not (packed >> (3 * nxt)) & 7:
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
        return dist

    # -- the three tiers, in order --------------------------------------------
    def solve(self):
        """``(presses, optsets, how)`` for the current start state, or
        ``(None, None, how)``.

        ``how`` names the tier that answered, and it is what `--plans` reports:
        the corpus is not the place to find out that a level shipped a beam plan
        with inferred labels."""
        try:
            back, _fwd = self.field()
            presses, optsets = self.field_plan(back)
            return presses, optsets, "field"
        except MemoryError:
            pass
        plan, closed = self.astar()
        if plan is not None and closed:
            optsets, _back = self.label_exact(plan)
            if optsets is not None:
                return plan, optsets, "astar+exact"
            return plan, self.annotate(plan), "astar"
        if plan is None:
            plan = self.beam()
            if plan is None:
                return None, None, "none"
        plan = self.shorten(plan)
        return plan, self.annotate(plan), "beam"


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class RainbowApplesExpert(PSExpert):
    """Plans on `_Board`, never on the interpreter.

    `PSExpert` still owns everything around that -- the in-memory memo, the
    on-disk plan cache with its staleness check, the level scoping and the
    snapshot/restore discipline -- so the only override is `_search`.
    `heuristic` here is `_Board`'s, not this class's: nothing runs the family's
    engine-blackbox A*.
    """

    #: ACTION is a total no-op (no rule in the game mentions it, and every
    #: `late` rule is conditioned on an apple standing on a paint, which is
    #: already settled -- `--selfcheck` measures it rather than trusting the
    #: rule listing, per the ps:ouroboros lesson). It is left out of the search
    #: because a press that cannot change the board can never be optimal; the
    #: exploration prefix still presses it, since that comes from the adapter's
    #: own action list.
    directions = list(_ORDER)
    #: `_key` below is dynamic-objects-only, canonical only WITHIN a level.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        g = self.g
        n2i = g.obj_name_to_idx
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.apple_col = {n2i[c + "apple"]: _CODE[c] for c in COLOURS}
        self.paint_col = {n2i[c + "paint"]: _CODE[c] for c in COLOURS}
        self.tgt_col = {n2i[c + "target"]: _CODE[c] for c in COLOURS}
        self.reg_id = n2i["regplayer"]
        self.nega_id = n2i["negaplayer"]
        self.apple_ids = set(self.apple_col)
        self.player_ids = {self.reg_id, self.nega_id}
        self.dyn_ids = self.apple_ids | self.player_ids
        self.ids = (self.wall_ids, self.apple_col, self.paint_col,
                    self.tgt_col, self.reg_id, self.nega_id)
        self.how: dict[int, str] = {}

    def _key(self, eng) -> frozenset:
        # Apples + players. Walls, paints and targets are static per level -- no
        # rule in this game creates, destroys or moves any of them -- hence
        # ``scope_by_level``.
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn)
        )

    def heuristic(self, eng) -> int:
        raise AssertionError("planning happens on _Board, not on the engine")

    def board(self, eng) -> _Board:
        return _Board.read(eng, self.ids)

    def _search(self, eng) -> list | None:
        board = self.board(eng)
        presses, optsets, how = board.solve()
        self.how[len(self.how)] = how
        return None if presses is None else Plan(presses, optsets)


class RainbowApplesSolver(PSAStarSolver):
    game_id = "puzzlescript_rainbow_apples"
    game_name = GAME_NAME
    #: The game folder's wrapper raises the per-level step limit for level 8,
    #: whose plan is 189 presses against the adapter's 200-step default; a
    #: generator that built its own bare adapter would tape frames the live
    #: agent never sees.
    game_module_id = GAME_MODULE
    expert_cls = RainbowApplesExpert

    #: Room for level 8's 189-press plan plus the replay after the exploration
    #: prefix; the per-level budget the wrapper sets is what actually bounds the
    #: episode.
    max_steps = 400

    def prepare_expert(self, game, expert) -> None:
        """Build every level's plan before `discover_solvable` asks for it -- the
        same work either way, but it makes the one-time cost (level 5's A* and
        level 8's beam) visible as startup rather than as a mysteriously slow
        first seed, and it fills the disk cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Seating a model state back into the interpreter
# ---------------------------------------------------------------------------

def _seat(eng, g, board: _Board, state) -> None:
    """Write ``state`` into the engine grid, so the interpreter can be asked what
    it does from a state the model reached.

    This is what makes `--fuzz` a proof rather than a sample on the levels whose
    space closes: every reachable state is seated and stepped, instead of
    random-walking and hoping to shove an apple across a paint."""
    idx = g.obj_name_to_idx
    grid = [[{idx["wall"]} for _ in range(board.w)] for _ in range(board.h)]
    for (r, c) in board.free:
        grid[r][c] = {idx["background"]}
    for i, (r, c) in enumerate(board.free):
        pc = board.paint_of[i]
        if pc:
            grid[r][c].add(idx[COLOURS[pc - 1] + "paint"])
    for i, col in board.targets:
        r, c = board.free[i]
        grid[r][c].add(idx[COLOURS[col - 1] + "target"])
    packed, p, q = state
    for i, (r, c) in enumerate(board.free):
        col = (packed >> (3 * i)) & 7
        if col:
            grid[r][c].add(idx[COLOURS[col - 1] + "apple"])
    grid[board.free[p][0]][board.free[p][1]].add(idx["regplayer"])
    if q >= 0:
        grid[board.free[q][0]][board.free[q][1]].add(idx["negaplayer"])
    eng.grid = grid
    eng.height, eng.width = board.h, board.w
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


def _read_state(eng, expert, board: _Board):
    """``(packed, reg, nega)`` read back off the interpreter grid, or a string
    naming what went wrong (an apple inside a wall, a vanished player) -- no rule
    in this game can do either, so the fuzz should say so rather than crash."""
    packed, p, q = 0, -1, -1
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            i = board.idx.get((r, c))
            for obj in cell:
                if obj in expert.apple_col:
                    if i is None:
                        return "apple inside a wall"
                    packed |= expert.apple_col[obj] << (3 * i)
                elif obj == expert.reg_id:
                    p = i
                elif obj == expert.nega_id:
                    q = i
    if p is None or p < 0:
        return "RegPlayer vanished"
    return (packed, p, q)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _levels():
    """``(solver, game, expert)`` with the adapter parked at level 0."""
    solver = RainbowApplesSolver()
    game = solver.make_game(0)
    return solver, game, RainbowApplesExpert(game)


def _report() -> int:
    """Per-level size, piece counts, plan length, tie coverage and which tier
    solved it -- and CERTIFY each plan by replaying it on the interpreter.

    This one re-derives every plan from scratch rather than reading the disk
    cache, because the tier and its cost are what it is reporting; it takes a
    few minutes, almost all of it level 8's beam. Every other check reads the
    SHIPPED plan out of the cache. The searches are deterministic, so the two
    are the same plan."""
    _solver, game, expert = _levels()
    total = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = expert.board(eng)
        if eng.check_win():
            print(f"level {level:2d}: STARTS ALREADY WON")
            bad += 1
            continue
        t = time.time()
        presses, optsets, how = board.solve()
        took = time.time() - t
        if presses is None:
            print(f"level {level:2d}: UNSOLVABLE ({how})")
            bad += 1
            continue
        # Certification: the interpreter must win on the LAST press and no
        # earlier (an earlier win would mean the plan is not shortest).
        won_at = None
        for i, direction in enumerate(presses):
            eng.step(direction)
            if eng.check_win():
                won_at = i
                break
        ok = won_at == len(presses) - 1
        bad += not ok
        total += len(presses)
        ties = sum(1 for s in optsets if len(s) > 1)
        apples = sum(1 for i in range(board.n)
                     if (board.start[0] >> (3 * i)) & 7)
        game.set_level(level)
        room = "ok" if len(presses) < game._max_steps else "OVER BUDGET"
        bad += room != "ok"
        print(f"level {level:2d}: {board.h:2d}x{board.w:2d} free={board.n:3d} "
              f"{apples} apples {len(board.targets)} targets "
              f"{sum(1 for x in board.paint_of if x)} paints"
              f"{' TWIN' if board.has_nega else '     '}, "
              f"{len(presses):3d} presses (budget {game._max_steps}, {room}), "
              f"{ties:3d} tie steps ({ties / len(presses):3.0%}), "
              f"{how:11s} in {took:6.2f}s, "
              f"{'CERTIFIED' if ok else 'PLAN REJECTED BY THE INTERPRETER'}")
    print(f"total {total} presses over {game.n_levels} levels")
    return 0 if not bad else 1


def _verify() -> int:
    """Independent check of the SEARCH -- the part `--fuzz` cannot reach.

    The fuzz only ever exercises `step`, the forward rule. Everything the plans
    rest on is downstream of it: the predecessor bookkeeping in `field`, the
    backward pass, and the claim that a labelled press is genuinely one of the
    shortest. For a field level, three checks in increasing independence:

      * **the backward field, recomputed without the predecessor map** -- one
        scan of the whole reachable set per BFS layer. Slower and completely
        separate bookkeeping; it must agree on every state.
      * **the double-entry formula** -- a forward BFS from the start must put the
        nearest winning state at exactly the plan length, and step ``i``'s
        optimal set must be ``{a : fwd(next) == i + 1 and back(next) ==
        d* - i - 1}``, which uses both passes and so fails if either is wrong.
      * **every labelled press replayed on the INTERPRETER** -- spliced into a
        variant plan (the prefix, that press, then a fresh optimal continuation)
        and driven through the real engine, which must win on press ``d*`` and
        not before. This one leaves `_Board` behind entirely.

    A searched level has no full field, so it gets the third check alone, with
    the continuation rebuilt from `label_exact`'s bounded field where there is
    one (level 5) and by `splice_walk` where there is not (level 8) -- an
    inferred walk alternative claims only "this press is also on a shortest route
    to the same push cell", so it is certified by taking it, finishing the walk,
    and requiring the interpreter to win at the plan's own length.

    The plan checked is the SHIPPED one -- fetched through `expert.plan`, i.e.
    out of the same disk cache the recorder reads -- not a fresh re-derivation.
    """
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        layout = game._game.levels[level]
        board = expert.board(eng)
        notes = []
        t = time.time()
        shipped = expert.plan(eng, level)
        optsets = list(getattr(shipped, "optsets", []) or [])
        presses = list(shipped)
        star = len(presses)
        try:
            back, fwd = board.field()
        except MemoryError:
            back = fwd = None
        if back is None:
            # No full field: try the bounded one the labels were measured on.
            _opt, back = board.label_exact(presses)
        if fwd is not None:
            field_presses, field_optsets = board.field_plan(back)
            if field_presses != presses or field_optsets != optsets:
                notes.append("the shipped plan is not the field's own")

            # 1. the backward field again, by layer scan, with no predecessors.
            scan = {s: 0 for s in fwd if board.win(s)}
            frontier = set(scan)
            d = 0
            while frontier:
                d += 1
                frontier = {s for s in fwd if s not in scan
                            and any(board.step(s, a) in frontier for a in _ORDER)}
                for s in frontier:
                    scan[s] = d
            if scan != back:
                notes.append(f"LAYER SCAN differs on "
                             f"{len(set(scan) ^ set(back))} states")

            # 2. the double-entry formula, over both passes.
            wins = [fwd[s] for s in fwd if board.win(s)]
            if min(wins) != star:
                notes.append(f"LENGTH {min(wins)} != {star}")
            state = board.start
            for i, direction in enumerate(presses):
                want = [a for a in _ORDER
                        if fwd.get(board.step(state, a), -1) == i + 1
                        and back.get(board.step(state, a), -1) == star - i - 1]
                if want != optsets[i]:
                    notes.append(f"step {i}: {optsets[i]} != {want}")
                state = board.step(state, direction)

        # 3. every labelled press, spliced into a variant plan and replayed on
        #    the real interpreter.
        replays = 0
        state = board.start
        for i, alts in enumerate(optsets):
            for alt in alts:
                if back is not None:
                    tail = board.downhill(back, board.step(state, alt))
                    variant = None if tail is None else presses[:i] + [alt] + tail
                elif alt == presses[i]:
                    variant = presses
                else:
                    variant = board.splice_walk(presses, i, alt)
                if variant is None:
                    notes.append(f"step {i} press {alt}: no continuation")
                    continue
                replays += 1
                eng.load_level(layout)
                won_at = None
                for j, direction in enumerate(variant):
                    eng.step(direction)
                    if eng.check_win():
                        won_at = j
                        break
                if won_at != star - 1 or len(variant) != star:
                    notes.append(f"step {i} press {alt}: interpreter won at "
                                 f"{won_at} of {len(variant)}")
            state = board.step(state, presses[i])

        bad += len(notes)
        how = ("%6d reachable" % len(fwd)) if fwd else (
            "bounded field" if back is not None else "walk labels ")
        print(f"level {level:2d}: {how}, "
              f"d*={star:3d}, {replays:4d} labelled presses replayed in "
              f"{time.time() - t:6.1f}s: "
              f"{'AGREES' if not notes else '; '.join(notes[:3])}")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} disagreements")
    return 0 if not bad else 1


#: The outcome labels `_kind` can return, in report order.
_KINDS: tuple[str, ...] = ("refused", "walk", "push", "chain", "repaint",
                           "twin-walk", "twin-push", "twin-follows")


def _colours(board: _Board, packed: int) -> list[int]:
    return sorted((packed >> (3 * i)) & 7 for i in range(board.n)
                  if (packed >> (3 * i)) & 7)


def _kind(board: _Board, state, nxt, direction) -> str:
    """Classify a transition for the fuzz's coverage counters. ``twin-follows``
    is the mirror rule NOT firing -- Reg facing a wall, so the twin walks WITH
    the press instead of against it, which is the one branch a level with no twin
    can never reach and the one the whole of level 9 turns on."""
    if nxt == state:
        return "refused"
    packed, p, q = state
    if _colours(board, nxt[0]) != _colours(board, packed):
        return "repaint"
    if q >= 0:
        if board.wall_at[direction][p]:
            return "twin-follows"
        x = board.nxt[direction][q]
        if x >= 0 and (packed >> (3 * x)) & 7:
            return "twin-push"
        if nxt[2] != q:
            return "twin-walk"
    x = board.nxt[direction][p]
    if x < 0 or not (packed >> (3 * x)) & 7:
        return "walk"
    return ("chain" if len(board._run(packed, x, board.nxt[direction])) > 1
            else "push")


def _fuzz(trials: int = 400, steps: int = 40) -> int:
    """Assert `_Board` reproduces the interpreter.

    EXHAUSTIVELY where the space closes (`_seat` every reachable state, compare
    all four presses, board and win flag alike) -- that is not a sample of the
    space, it is the space. Levels 5 and 8 do not close, so they are random
    walked from a COLD start: no warm-up press, because turn-one bookkeeping is
    exactly the class of mechanic a warm-up hides.

    Coverage counters are printed per outcome. A run that reported agreement
    having never repainted an apple, never shoved a run of two, and never turned
    the twin around would have checked a different game."""
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    total = bad = 0
    kinds: dict[str, int] = {}
    rng = random.Random(20260815)
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        seen: dict[str, int] = {}
        try:
            _back, fwd = board.field()
            states, how = fwd, "exhaustive"
        except MemoryError:
            states, how = None, "random walk"

        def check(state, direction) -> int:
            nonlocal bad
            mine = board.step(state, direction)
            _seat(eng, g, board, state)
            eng.step(direction)
            theirs = _read_state(eng, expert, board)
            hurt = 0
            if theirs != mine:
                hurt += 1
                if bad < 5:
                    print(f"  level {level} MISMATCH from {state} {direction}: "
                          f"model {mine} engine {theirs}")
            if eng.check_win() != board.win(mine):
                hurt += 1
                if bad < 5:
                    print(f"  level {level} WIN MISMATCH from {state} {direction}")
            bad += hurt
            kind = _kind(board, state, mine, direction)
            seen[kind] = seen.get(kind, 0) + 1
            return hurt

        if states is not None:
            for state in states:
                for direction in _ORDER:
                    total += 1
                    check(state, direction)
        else:
            for _trial in range(trials):
                state = board.start
                for _step in range(steps):
                    direction = rng.choice(_ORDER)
                    total += 1
                    if check(state, direction):
                        state = _read_state(eng, expert, board)
                        if isinstance(state, str):
                            break
                    else:
                        state = board.step(state, direction)
        for k, v in seen.items():
            kinds[k] = kinds.get(k, 0) + v
        print(f"level {level:2d}: {how:11s} "
              + "  ".join(f"{k}={seen.get(k, 0)}" for k in _KINDS if seen.get(k)))
    print("coverage: " + "  ".join(f"{k}={kinds.get(k, 0)}" for k in _KINDS))
    missing = [k for k in _KINDS if not kinds.get(k)]
    if missing:
        print(f"  NOT REACHED by any level: {', '.join(missing)} "
              f"-- covered by --selfcheck only")
    print(f"{total} transitions: "
          f"{'model matches the interpreter' if not bad else f'{bad} MISMATCHES'}")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# Mechanic self-check
# ---------------------------------------------------------------------------

#: Level-file legend letters used by `_selfcheck`'s hand-built boards.
_SHORT = {"#": "wall", "X": "regplayer", "@": "negaplayer",
          "r": "redapple", "b": "blueapple", "g": "greenapple",
          "p": "purpleapple", "i": "pinkapple",
          "A": "redpaint", "M": "bluepaint", "E": "greenpaint",
          "C": "purplepaint", "D": "pinkpaint",
          "F": "redtarget", "H": "bluetarget", "K": "greentarget",
          "L": "purpletarget", "J": "pinktarget"}


def _selfcheck() -> int:
    """State the module docstring's mechanic facts as boards, and measure them.

    `--fuzz` already proves the MODEL equals the interpreter over every
    reachable state of eight levels, which subsumes much of this. What it cannot
    do is state the claims in a form a reader can check -- and it cannot reach
    the configurations no level contains at all (the twin beside a paint, a red
    apple on purple paint, a five-apple run), where a future edit to the .txt
    would otherwise break something silently."""
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    n2i = g.obj_name_to_idx
    bad = 0

    def fail(msg: str) -> None:
        nonlocal bad
        bad += 1
        print(f"  VIOLATION: {msg}")

    def build(rows) -> None:
        h, w = len(rows), len(rows[0])
        assert len({len(r) for r in rows}) == 1
        grid = [[{n2i["background"]} for _ in range(w)] for _ in range(h)]
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch != ".":
                    grid[r][c].add(n2i[_SHORT[ch]])
        eng.grid = grid
        eng.height, eng.width = h, w
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()

    def cells(name: str) -> set:
        return {(r, c) for r, row in enumerate(eng.grid)
                for c, cell in enumerate(row) if n2i[name] in cell}

    def snap() -> list:
        return [[set(cell) for cell in row] for row in eng.grid]

    # 1. a push carries the whole SAME-COLOURED run, and stops dead at a
    #    different colour -- player included.
    build(["########", "#Xrrrr.#", "########"])
    eng.step("right")
    if cells("redapple") != {(1, 3), (1, 4), (1, 5), (1, 6)} \
            or cells("regplayer") != {(1, 2)}:
        fail("a four-apple run of one colour did not move as one piece")
    for row in ("#Xrb..#", "#Xrrb.#", "#Xbr..#"):
        build(["#######", row, "#######"])
        before = snap()
        eng.step("right")
        if eng.grid != before:
            fail(f"{row}: a push crossed a colour boundary")
    build(["#####", "#Xr##", "#####"])
    before = snap()
    eng.step("right")
    if eng.grid != before:
        fail("an apple against a wall did not cancel the whole turn")

    # 2. paint repaints whatever stands on it, in ONE turn -- including the
    #    game's red-on-purple bug, which its own rule order undoes.
    for paint, letter in (("red", "A"), ("blue", "M"), ("green", "E"),
                          ("purple", "C"), ("pink", "D")):
        for apple in ("r", "b", "g", "p", "i"):
            build(["#####", f"#X{apple}{letter}#", "#####"])
            eng.step("right")
            if cells(paint + "apple") != {(1, 3)}:
                fail(f"{apple} pushed onto {paint} paint did not become {paint}")
    build(["#######", "#Xr.A.#", "#######"])
    for _ in range(3):
        eng.step("right")
    if cells("redapple") != {(1, 5)} or cells("redpaint") != {(1, 4)}:
        fail("a paint was consumed by the apple that crossed it")
    build(["########", "#XrrA..#", "########"])
    eng.step("right")
    if cells("redapple") != {(1, 3), (1, 4)}:
        fail("a run pushed onto a paint of its own colour changed")

    # 3. targets are inert scenery on their own layer, and the win is per TARGET.
    build(["#####", "#XrF#", "#####"])
    eng.step("right")
    if cells("redtarget") != {(1, 3)} or not eng.check_win():
        fail("an apple parked on its own target did not win")
    build(["#####", "#XrH#", "#####"])
    eng.step("right")
    if eng.check_win():
        fail("a red apple on a blue target counted as a win")
    build(["######", "#XrFF#", "######"])
    eng.step("right")
    if eng.check_win():
        fail("one red apple satisfied two red targets")

    # 4. ACTION is a total no-op, on a board with every kind of piece on it.
    build(["#########", "#Xr.A.F@#", "#########"])
    before = snap()
    eng.step("action")
    if eng.grid != before:
        fail("ACTION changed the board")

    # 5. the twin walks BACKWARDS ...
    build(["#######", "#X...@#", "#######"])
    eng.step("right")
    if cells("regplayer") != {(1, 2)} or cells("negaplayer") != {(1, 4)}:
        fail("the twin did not mirror an ordinary press")
    #    ... unless Reg is facing a WALL, when it walks WITH the press ...
    build(["#######", "#X...@#", "#######"])
    eng.step("left")
    if cells("regplayer") != {(1, 1)} or cells("negaplayer") != {(1, 4)}:
        fail("Reg against a wall did not leave the twin following the press")
    #    ... including pushing, which is how level 9 is solved ...
    build(["########", "#X..r@.#", "########"])
    eng.step("left")
    if cells("redapple") != {(1, 3)} or cells("negaplayer") != {(1, 4)}:
        fail("Reg against a wall did not make the twin an ordinary pusher")
    #    ... and being blocked by an APPLE is NOT the same thing (no rule
    #    cancels that force, so the mirror still fires).
    build(["########", "#Xrb.@.#", "########"])
    eng.step("right")
    if cells("regplayer") != {(1, 1)} or cells("negaplayer") != {(1, 4)}:
        fail("an apple-blocked Reg suppressed the mirror rule")

    # 6. the twin pushes the apple BEHIND it and is stopped by the one ahead.
    build(["########", "#.X..r@#", "########"])
    eng.step("left")
    if cells("redapple") != {(1, 4)} or cells("negaplayer") != {(1, 6)}:
        fail("the twin did not shove the apple on the pressed side")
    build(["########", "#.X.@r.#", "########"])
    eng.step("left")
    if cells("redapple") != {(1, 5)} or cells("negaplayer") != {(1, 4)}:
        fail("the twin pushed the apple in its own path")

    # 7. two movers may not have the same destination, and may not swap.
    for rows, d in ((["#######", "#.X.@.#", "#######"], "right"),
                    (["#######", "#..X@.#", "#######"], "right"),
                    (["#######", "#Xr@..#", "#######"], "right")):
        build(rows)
        before = snap()
        eng.step(d)
        if eng.grid != before:
            fail(f"{rows[1]} {d}: a contested cell was not refused")

    # 8. no rule carries a command flag the model does not implement. `restart`
    #    is the one that would invalidate everything here without a symptom: it
    #    is executed by the ADAPTER, not by `PSEngine.step`, so a model built on
    #    `eng.step` would plan straight through a lethal object and no plan would
    #    replay (the ps:impasse lesson). `random` would make a press genuinely
    #    nondeterministic (ps:flying_kick), `again` would run extra ticks inside
    #    one press, and `rigid` has no backoff in this adapter (ps:dang_im_huge).
    flags = ("cancel", "restart", "win", "again", "random", "rigid")
    for rule in g.rules:
        for flag in flags:
            if getattr(rule, flag, False):
                fail(f"a rule carries `{flag}` -- the model does not implement it")
    return bad


# ---------------------------------------------------------------------------
# The symmetry group
# ---------------------------------------------------------------------------

def _transform(k: int, mirror: bool):
    """``(cell_fn, dims_fn, direction_map)`` for one element of the 8-group:
    mirror left-right first, then turn clockwise ``k`` times. The direction map
    is DERIVED from the same linear part that moves the cells, so the two cannot
    drift apart."""
    def cell(rc, hw):
        r, c = rc
        h, w = hw
        if mirror:
            c = w - 1 - c
        for _ in range(k):
            r, c, h, w = c, h - 1 - r, w, h
        return (r, c)

    def dims(hw):
        h, w = hw
        return (w, h) if k % 2 else (h, w)

    dmap = {}
    for name, (dr, dc) in _DELTA.items():
        if mirror:
            dc = -dc
        for _ in range(k):
            dr, dc = dc, -dr
        dmap[name] = next(n for n, d in _DELTA.items() if d == (dr, dc))
    return cell, dims, dmap


def _dynamic(eng, expert):
    """``(reg, nega, apples)`` off the interpreter grid -- the only things a
    presentation transform may be asserted on. Apples carry their COLOUR: a
    transform must not merely move them, it must leave the same colours in the
    transformed cells."""
    reg = nega = None
    apples = set()
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            for obj in cell:
                if obj == expert.reg_id:
                    reg = (r, c)
                elif obj == expert.nega_id:
                    nega = (r, c)
                elif obj in expert.apple_col:
                    apples.add((r, c, expert.apple_col[obj]))
    return reg, nega, frozenset(apples)


def _symmetry() -> int:
    """Prove ON THE INTERPRETER that the 8-element presentation group is an exact
    symmetry of the mechanic, which is what entitles this game to
    `PuzzleScriptAdapter._FLIP_GAMES`.

    The argument from the rules is strong -- all thirty-seven are stated with the
    relative ``>`` / ``<`` force, there is no gravity, the win conditions name no
    direction, and input is screen-relative -- but it has been wrong before (see
    ps:gobble_rush, where four per-direction rule copies contesting one cell made
    a mechanic chiral), and this game does have two objects that can contest a
    cell. So it is measured: each level's plan is replayed on all eight turned
    and mirrored copies of its own board, built by transforming the LEVEL LAYOUT
    so the interpreter re-runs its own level-start work, and every press must
    leave the pieces -- with their colours -- exactly where the transform of the
    reference run put them.

    There is nothing to RELABEL (the ps:l_a_s_e_r trap): no object here encodes a
    direction. The sprites are sound under the group too -- the apple, the target
    diamond and the repainted paint lattice are each invariant, and the players'
    faces are asymmetric art that encodes no facing, since neither has a
    direction state.

    A plan replay can be BLIND, though (the ps:lovendpieces correction: no
    SHORTEST plan need ever traverse the order-sensitive transition). So a second
    stage replays EVERY reachable state and press of every level whose space is
    small enough -- including level 9, the only level where two movers can
    contest a cell at all, which is precisely the shape that hid ps:gobble_rush's
    chirality."""
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        presses = list(expert.plan(eng, level))
        layout = g.levels[level]
        hw = (len(layout), len(layout[0]))

        eng.load_level(layout)
        ref = [_dynamic(eng, expert)]
        for direction in presses:
            eng.step(direction)
            ref.append(_dynamic(eng, expert))

        notes = []
        for k, mirror in itertools.product(range(4), (False, True)):
            if (k, mirror) == (0, False):
                continue
            cell, dims, dmap = _transform(k, mirror)
            th, tw = dims(hw)
            turned = [[set() for _ in range(tw)] for _ in range(th)]
            for r, row in enumerate(layout):
                for c, objs in enumerate(row):
                    tr, tc = cell((r, c), hw)
                    turned[tr][tc] = set(objs)
            eng.load_level(turned)
            for i, direction in enumerate(presses):
                eng.step(dmap[direction])
                reg, nega, apples = ref[i + 1]
                want = (cell(reg, hw),
                        None if nega is None else cell(nega, hw),
                        frozenset(cell(a[:2], hw) + (a[2],) for a in apples))
                if _dynamic(eng, expert) != want:
                    notes.append(f"rot{k}{'m' if mirror else ''}@press{i}")
                    break
        bad += len(notes)
        print(f"level {level:2d}: {len(presses):3d} presses x 7 presentations: "
              f"{'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:3])}")

    # Stage 2: every reachable state and press, on every presentation, for the
    # levels whose space is small enough to sweep.
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        try:
            _back, fwd = board.field(cap=SYMMETRY_STATE_CAP)
        except MemoryError:
            print(f"level {level:2d}: over {SYMMETRY_STATE_CAP} states, "
                  f"the plan replay above stands alone")
            continue
        layout = g.levels[level]
        hw = (len(layout), len(layout[0]))
        notes = []
        checked = 0
        for k, mirror in itertools.product(range(4), (False, True)):
            if (k, mirror) == (0, False):
                continue
            cell, dims, dmap = _transform(k, mirror)
            th, tw = dims(hw)
            turned = [[set() for _ in range(tw)] for _ in range(th)]
            for r, row in enumerate(layout):
                for c, objs in enumerate(row):
                    tr, tc = cell((r, c), hw)
                    turned[tr][tc] = set(objs)
            eng.load_level(turned)
            tboard = expert.board(eng)
            # cell index in ``board`` -> cell index in the turned one
            imap = [tboard.idx[cell(rc, hw)] for rc in board.free]
            for state in fwd:
                for direction in _ORDER:
                    tstate = _map_state(board, tboard, imap, state)
                    _seat(eng, g, tboard, tstate)
                    eng.step(dmap[direction])
                    got = _read_state(eng, expert, tboard)
                    want = _map_state(board, tboard, imap,
                                      board.step(state, direction))
                    checked += 1
                    if got != want:
                        notes.append(f"rot{k}{'m' if mirror else ''} "
                                     f"from {state} {direction}")
                        break
                if notes:
                    break
        bad += len(notes)
        print(f"level {level:2d}: {len(fwd):6d} states x 4 presses x 7 "
              f"presentations = {checked:7d} transitions: "
              f"{'SYMMETRIC' if not notes else 'CHIRAL ' + notes[0]}")
    print("symmetry clean -- the flip augmentation is exact" if not bad
          else f"SYMMETRY FAILED: {bad} chiral presentations")
    return 0 if not bad else 1


def _map_state(board: _Board, tboard: _Board, imap: list[int], state):
    """``state`` expressed on a turned/mirrored copy of the same board."""
    packed, p, q = state
    tpacked = 0
    for i in range(board.n):
        col = (packed >> (3 * i)) & 7
        if col:
            tpacked |= col << (3 * imap[i])
    return (tpacked, imap[p], imap[q] if q >= 0 else -1)


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    This is the check that found the one bug in the shipped art: the paint sprite
    carried its colour on the middle three rows, which is exactly where an apple
    and a player are both opaque, so ALL FIVE paints rendered identically under a
    standing piece -- the game's central mechanic unreadable whenever anything
    was on top of it. The paints are now a 3x3 lattice of coloured pixels whose
    corners and edge midpoints land in the holes of both bodies. (An apple is a
    different case and needed no fix: it takes the paint's colour the moment it
    lands, so the apple's own colour reports the paint underneath it.)

    Whole frames are compared, not cell crops: `_render_frame` upscales the board
    to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor)."""
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {"floor": (), "wall": ("wall",),
             "reg": ("regplayer",), "twin": ("negaplayer",)}
    for c in COLOURS:
        comps[f"paint_{c}"] = (c + "paint",)
        comps[f"target_{c}"] = (c + "target",)
        comps[f"apple_{c}"] = (c + "apple",)
        # An apple on a paint is always the paint's own colour (the `late` rules
        # see to that), so the mismatched stacks are unreachable and excluded.
        comps[f"apple_{c}_on_paint_{c}"] = (c + "paint", c + "apple")
        comps[f"apple_{c}_on_target_{c}"] = (c + "target", c + "apple")
        comps[f"reg_on_paint_{c}"] = (c + "paint", "regplayer")
        comps[f"reg_on_target_{c}"] = (c + "target", "regplayer")
        comps[f"twin_on_paint_{c}"] = (c + "paint", "negaplayer")
    # The stacks that carry the game's "wrong answer" signal: an apple parked on
    # a target of another colour must read as neither a win nor a bare apple.
    comps["apple_red_on_target_blue"] = ("bluetarget", "redapple")
    comps["apple_blue_on_target_red"] = ("redtarget", "blueapple")
    comps["apple_green_on_target_pink"] = ("pinktarget", "greenapple")

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for name, objs in comps.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)

        def margin(a, b) -> float:
            return int((shots[a] != shots[b]).sum()) / (h * w)

        win_px = min(margin(f"apple_{c}_on_target_{c}", f"apple_{c}")
                     for c in COLOURS)
        paint_px = min(margin(f"reg_on_paint_{a}", f"reg_on_paint_{b}")
                       for a, b in itertools.combinations(COLOURS, 2))
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {len(comps)} compositions, "
              f"an apple on its own target reads >={win_px:4.1f}px/cell against "
              f"the bare apple, two paints under a player >={paint_px:4.1f}, "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--selfcheck" in sys.argv:
        violations = _selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(RainbowApplesSolver.main())
