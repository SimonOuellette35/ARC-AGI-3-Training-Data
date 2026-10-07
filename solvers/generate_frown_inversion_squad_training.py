"""Generate Phase-1 training data for the PuzzleScript game
ps:frown_inversion_squad ("Frown Inversion Squad", Jim Palmeri -- the one where
you cannot touch a sad face, you can only make happiness reach it).

The harness -- the rotation contract, the trajectory recorder, the plan cache
and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared
with the other ps: generators. This file is the game-specific part: a native
model of the ruleset, the searches over it, and the checks.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_frown_inversion_squad",
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
Every step also carries a set of equally-optimal presses.

THE GAME
--------
Everyone on the board is sad and you have to cheer them all up. The win is
``No Sadness`` (no sad face, no sad wall) plus ``No BigFaceParts`` -- every
quadrant of the Frown Inverter has to be built into a whole one -- plus "no
animation is still running", which is free because every animation drains
inside the keypress that starts it.

FIVE RULES SHAPE EVERY PLAN.

  * YOU CANNOT PUSH A SAD FACE. ``[ > Player | FaceHappy ]`` is the only face
    push in the game, so a frown is a WALL until something has cheered it up,
    and the board's geometry changes as you solve it. The game's own message
    says it: "If someone is sad, don't push them! Cheer them up first!"
  * HAPPINESS IS CONTAGIOUS AND IRREVERSIBLE. ``late [ FaceSad | FaceHappy ]``
    runs to a fixpoint in each of the four directions IN TURN, so one happy
    face touching the end of a straight chain of frowns flips the whole chain
    in a single press -- but a chain that bends the wrong way for the pass
    order takes an extra press, which is why a press that moves nobody (a wall
    bump, or ACTION5) is a real move and is in the search. Nothing ever turns
    sad again, so the state space is monotone.
  * TWO HAPPY FACES SIDE BY SIDE ARE BOTH STUCK. Curing a frown leaves the
    new happy face pressed against the one that cured it, and neither can be
    pushed along that axis any more (the push has nowhere to stand). Curing a
    row of frowns from the side therefore strands the cure; curing them from
    the row behind does not.
  * A HEART IS A ONE-SHOT PAIR CHEER. ``late [ Face | Heart | Face ]`` needs
    faces on BOTH sides of it, and the late rules run AFTER movement, so the
    heart has to LAND between them -- you cannot walk it into position and
    then step away. The two halves ``{``/``}`` merge into a whole heart only
    when one is pushed HORIZONTALLY into the other (the merge rules are
    direction-locked to ``right``), and a half heart cheers nobody up.
  * THE FROWN INVERTER FIRES WHEN IT MOVES, AND WHEN IT IS BUILT. Four
    quadrants shoved into a 2x2 assemble (a 5-tick animation) and the finished
    face immediately fires eight happiness beams -- left and right along both
    of its rows, up and down along both of its columns -- and it fires them
    again on every press that MOVES it. A beam floods the whole run of empty
    cells it lands in and cheers up whatever stops it. A finished face also
    cheers up anything orthogonally adjacent to it, every press, for free.
  * A BLOCKED FROWN INVERTER CANCELS THE TURN.
    ``[ > BigFaceFull | stationary SolidObject ] -> cancel`` means shoving the
    2x2 into a wall or a face does not just fail, it undoes the whole press --
    so a plan can never "waste" a press against one, and the search sees those
    presses as pure no-ops.

THE MODEL
---------
`_Board` re-implements the ruleset natively over bitmasks (state = the player
cell plus fifteen masks), because the interpreter runs this game at 650-1600
presses/s -- 40+ rules, half of them whole-board late scans -- where the model
runs at ~30k. It reproduces the interpreter's turn structure, including the
``again`` loop that drains the heart and assembly animations inside one press,
because the extra late passes those ticks buy are visible: a contagion chain
that turns a corner needs the pass ordering, and the beams live for four ticks.

``--fuzz`` plays random boards press-for-press against the real interpreter and
compares every mask and the win flag after every press. It was accepted on
48,000 transitions over 1,200 boards with zero mismatches, and the boards are
SEEDED with the configurations random play never builds -- a quadrant one push
from completing a face, a heart one push from landing between two frowns, a
chain of frowns waiting for a cure -- plus the twenty shipped levels themselves.
Re-run it after ANY change to the .txt rules or to the adapter's rule code.

The one thing it caught that no amount of reading the rules would have: a MOVE
MARKER CAN OUTLIVE ITS PRESS. It rides its own collision layer, so when a
finished face is shoved into something that does NOT cancel the turn (a loose
quadrant, or the board edge) the face stays put and the marker slides off it,
where nothing will ever clear it -- except, of all things, the FaceMarker the
quadrant-pairing rule stamps on that cell and wipes three rules later. It is
inert where it lands, but the day a finished face arrives there the beams fire
with nothing having moved, so it is part of the state rather than a local.

THE SEARCHES
------------
Two, in order (`_solve`):

  * an exhaustive layered BFS over the five presses, which returns a genuinely
    SHORTEST plan and, in the same sweep, the EXACT set of equally-shortest
    presses at every step (`_Board.bfs`). It holds ten of the twenty levels.
  * a weighted best-first over ``walk somewhere and push`` MACROS for the rest
    (`_Board.best_first`), run at three points on a weight ladder with the
    shortest answer kept, then shortened by deleting the longest block of
    presses the plan can win without.

THE ONE HEURISTIC THAT MATTERS is `_Board.gather`: how far the loose quadrants
are from being a 2x2, measured over the WALL graph. A count of loose quadrants
is FLAT over the whole approach -- it only drops on the push that completes a
face -- so the search has nothing to steer with, and every level with a Frown
Inverter in it is unreachable without this. Levels 13 and 14 went from "no plan
inside 300k macro nodes" to 151 and 110k nodes when it was added, and using the
wall-aware distance rather than Manhattan is what unlocked 14 (its quadrants
have to be threaded through a one-cell gap that Manhattan cannot see). Same
lesson as [[fireproof-bomber-solver]]'s "a cover count needs a push-distance
term beside it".

THE LEVELS
----------
All twenty are Palmeri's own boards in shipped order (the ``message`` screens
between them are not levels and the adapter does not count them).

    level  size    what is wrong with it        plan  found by
    0      7x7      1 frown                        2  BFS
    1      7x9      3 frowns                      30  BFS
    2      8x10     6 frowns                      45  BFS
    3      7x7      2 frowns, 1 heart              3  BFS
    4      7x11     2 frowns, 1 heart             29  BFS
    5      11x11    8 frowns, 1 heart            130  macro
    6      7x9      2 frowns, 2 half hearts        5  BFS
    7      7x9      4 frowns, 4 half hearts       34  BFS
    8      9x11     5 frowns, 2 half hearts      130  macro
    9      7x7      2 frowns, 6 sad walls         23  BFS
    10     11x9    12 sad walls                   44  BFS
    11     11x13    7 frowns, 10 sad walls       183  macro
    12     7x7      one Frown Inverter            14  BFS
    13     7x9     16 sad walls + an Inverter     34  BFS
    14     9x11    10 sad walls + an Inverter     98  macro
    15-19  11x13   see below                       -  SKIPPED

804 presses over the fifteen, every one replayed through the real interpreter
to a WIN by ``--verify``, and all inside the adapter's 200-action per-level
budget. Eleven of the fifteen are certified shortest by the BFS.

THE FIVE THAT ARE SKIPPED are the ones whose Frown Inverter has to be BUILT out
of quadrants that start scattered across a maze:

  * 15, 17 and 19 ship TWO and THREE separate sets of quadrants, so the gather
    heuristic has to guess a matching as well as a route, and its greedy
    matching flips between assignments as the pieces move -- the estimate stops
    being a gradient and the search wanders;
  * 16 has forty-six sad walls, which is a long enough plan that the count term
    swamps everything else while staying flat until the first beam fires;
  * 18 puts the four quadrants in the four CORNERS of the board, in the wrong
    corners, with twelve frowns and nine hearts in the way -- assembling it at
    all is a harder sokoban than any level that ships solved here.

A staged search (assemble first, cheer up second) was tried on all five and
does not reach them either: on 15 and 18 the ASSEMBLY phase alone exhausts 200k
macro nodes. What would most likely crack them is a stable quadrant-to-set
matching plus a per-set assembly sub-search, not a bigger budget.

OPTIMAL-ACTION SETS
-------------------
For the eleven BFS levels they are EXACT and come free with the search: BFS by
layers gives every explored state its distance, and a state ``j`` can only be an
optimal successor of ``i`` when ``depth[j] == depth[i] + 1``, so the labelling
is one backward sweep over the forward edges. For the four macro levels the plan
is not proved shortest, so the claim is narrower and honest: the WALK stretches
are order-free, and every step of one is labelled with each direction that is
also a no-op and also one step closer to where that walk ends. ``--ties``
checks both claims independently -- a fresh BFS from every successor for the
first, and SUBSTITUTION (take the alternative, walk on to where the plan's walk
ended, replay the rest unchanged, require the same length and the same win) for
the second. 1,748 alternatives, no disagreements.

RUNNING IT
----------
    python solvers/generate_frown_inversion_squad_training.py --episodes N --out DIR
    python solvers/generate_frown_inversion_squad_training.py --plans   # per-level report
    python solvers/generate_frown_inversion_squad_training.py --verify  # replay on the interpreter
    python solvers/generate_frown_inversion_squad_training.py --ties    # re-check the labels
    python solvers/generate_frown_inversion_squad_training.py --audit   # rendering
    python solvers/generate_frown_inversion_squad_training.py --fuzz    # model vs interpreter

The searches are the whole cost of generation and they are seed-independent, so
they are cached in ``data/frown_inversion_squad_plans.json``: ~20 minutes cold
(the weight ladder pays for the levels it cannot plan tightly), ~4 seconds per
episode warm. Delete that file after changing a search or the model.
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
from array import array
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (                            # noqa: E402
    PSAStarSolver, PSExpert, Plan,
)
from adapters.puzzlescript_adapter import _render_frame           # noqa: E402

GAME_NAME = "FROWN_INVERSION_SQUAD"
PLAN_CACHE = (Path(__file__).resolve().parent.parent
              / "data" / "frown_inversion_squad_plans.json")

#: Every press is a turn: a direction the player cannot take still runs the
#: whole rule list (and so a late contagion pass), and ACTION5 is exactly that
#: with no direction at all. Both are in the search for that reason.
PRESSES = ["up", "down", "left", "right", "action"]

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
_OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}
_DIRS = ("up", "down", "left", "right")

# Indices into the state's mask tuple (state[0] is the player cell).
(SAD, HAPPY, HEART, HL, HR, B1, B2, B3, B4,
 TL, TR, BL, BR, WSAD, WHAPPY, MK) = range(16)
_NMASK = 16

#: The masks whose objects share the pushable collision layer: at most one of
#: them per cell, and they are the only things a force can ever move. WSAD /
#: WHAPPY decorate a wall on their own layer, and MK is a marker on a third.
_MOVABLE = _SOLO = (SAD, HAPPY, HEART, HL, HR, B1, B2, B3, B4,
                    TL, TR, BL, BR)

_MASK_NAMES = {
    SAD: "facesad", HAPPY: "facehappy", HEART: "heart", HL: "heartleft",
    HR: "heartright", B1: "bigface1", B2: "bigface2", B3: "bigface3",
    B4: "bigface4", TL: "bigfacetl", TR: "bigfacetr", BL: "bigfacebl",
    BR: "bigfacebr", WSAD: "wallsad", WHAPPY: "wallhappy",
    MK: "movemarker",
}


_INF = 1 << 20


def _cells(mask):
    """The cell indices set in ``mask``."""
    out = []
    while mask:
        b = mask & -mask
        mask ^= b
        out.append(b.bit_length() - 1)
    return out


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """The static half of a level (size, walls) plus the whole ruleset.

    A state is ``(p, *masks)``: the player's cell index and the fifteen
    bitmasks of `_MASK_NAMES`, over ``r * w + c``. Walls never change so they
    live on the board; everything else does.

    `step` reproduces one whole keypress in the interpreter's own order --
    animation ticks, beam decay, push forces, the cancel test, force
    resolution, then the late rules -- and then drains the ``again`` loop the
    same way the interpreter does, stopping when a tick changes nothing.
    """

    def __init__(self, h, w, walls):
        self.h, self.w, self.n = h, w, h * w
        self.walls = walls
        self.full = (1 << self.n) - 1
        col0 = colw = 0
        for r in range(h):
            col0 |= 1 << (r * w)
            colw |= 1 << (r * w + w - 1)
        # Shifting by 1 would wrap a row into its neighbour, so the two edge
        # columns are masked out of every horizontal shift.
        self.not_col0 = self.full & ~col0
        self.not_colw = self.full & ~colw
        self._dist = None

    def signature(self):
        return (self.h, self.w, self.walls)

    # -- geometry ------------------------------------------------------------
    def sh(self, m, d):
        """``m`` shifted one cell in direction ``d``: the mask of cells ``i+d``
        for every ``i`` in ``m``. Off-board bits fall out."""
        if d == "up":
            return m >> self.w
        if d == "down":
            return (m << self.w) & self.full
        if d == "left":
            return (m & self.not_col0) >> 1
        return (m & self.not_colw) << 1

    def nb(self, i, d):
        """Neighbour cell index in direction ``d``, or -1 off the board."""
        r, c = divmod(i, self.w)
        dr, dc = _DELTA[d]
        r += dr
        c += dc
        if 0 <= r < self.h and 0 <= c < self.w:
            return r * self.w + c
        return -1

    def nbrs(self, m):
        """The four-neighbourhood of every cell in ``m``."""
        return (self.sh(m, "up") | self.sh(m, "down")
                | self.sh(m, "left") | self.sh(m, "right"))

    # -- dynamics ------------------------------------------------------------
    def won(self, state):
        """The win predicate on a QUIESCENT state: ``No Sadness`` and ``No
        BigFaceParts``. The other four conditions (no animation, no anim
        marker, no beams) are about objects that only exist inside a press."""
        m = state
        return not (m[1 + SAD] | m[1 + WSAD]
                    | m[1 + B1] | m[1 + B2] | m[1 + B3] | m[1 + B4])

    def step(self, state, press):
        """One whole keypress: ``(new_state, won)``."""
        p = state[0]
        m = list(state[1:])
        # Transient layers: they exist only between the press and quiescence.
        ha = [0, 0, 0, 0]                 # HeartAnim1..4
        ban = [0] * 16                    # BigFace{1..4}{a..d}, k * 4 + stage
        am = 0                            # the AnimMarker layer
        # A MOVE marker can outlive its press: it rides its own collision
        # layer, so when a finished face is blocked by something that does not
        # cancel the turn (another quadrant, or the board edge) the marker
        # moves off it and is never cleared again. It is inert where it lands,
        # but the day a finished face reaches that cell the beams fire without
        # anything having moved -- which is why it is part of the state and not
        # a local. `--fuzz` found this in one transition out of 36,000.
        mk = m[MK]
        m[MK] = 0
        bh = [0, 0, 0]                    # BeamHoriz1..3
        bv = [0, 0, 0]                    # BeamVert1..3
        d = press if press in _DELTA else None
        fulls_of = (TL, TR, BL, BR)
        first = True

        for _tick in range(50):           # the interpreter's max_again
            before = (p, tuple(m), tuple(ha), tuple(ban), am, mk,
                      tuple(bh), tuple(bv))
            again = False

            # --- animation ticks (rules 1..30) --------------------------
            if ha[3]:
                ha[3] = 0
                again = True
            if ha[2]:
                ha[3], ha[2] = ha[2], 0
                again = True
            if ha[1]:
                ha[2], ha[1] = ha[1], 0
                again = True
            if ha[0]:
                ha[1], ha[0] = ha[0], 0
                again = True
            hit = m[HEART] & am
            if hit:
                ha[0] |= hit
                m[HEART] &= ~hit
                am &= ~hit
                again = True
            for k in range(4):
                b = k * 4
                if ban[b + 3]:
                    # A finished quadrant lands on the board WITH a move
                    # marker, which is what makes a freshly built face fire
                    # its beams without anyone pushing it.
                    m[fulls_of[k]] |= ban[b + 3]
                    mk = (mk & ~ban[b + 3]) | ban[b + 3]
                    am &= ~ban[b + 3]
                    ban[b + 3] = 0
                    again = True
                if ban[b + 2]:
                    ban[b + 3], ban[b + 2] = ban[b + 2], 0
                    again = True
                if ban[b + 1]:
                    ban[b + 2], ban[b + 1] = ban[b + 1], 0
                    again = True
                if ban[b + 0]:
                    ban[b + 1], ban[b + 0] = ban[b + 0], 0
                    again = True
                hit = m[B1 + k] & am
                if hit:
                    # BigFaceNa lives on the ANIMATION layer, so the quadrant
                    # leaves the board for five ticks and comes back whole.
                    ban[b] |= hit
                    m[B1 + k] &= ~hit
                    am &= ~hit
                    again = True

            # --- beam decay (rules 31..36) ------------------------------
            for beam in (bv, bh):
                if beam[2]:
                    beam[2] = 0
                    again = True
                if beam[1]:
                    beam[2], beam[1] = beam[1], 0
                    again = True
                if beam[0]:
                    beam[1], beam[0] = beam[0], 0
                    again = True

            # --- forces + movement (rules 37..44) -----------------------
            if first and d is not None and p >= 0:
                res = self._forces(p, m, mk, d)
                if res is None:                      # cancel: undo the turn
                    return state, self.won(state)
                p, mk = res

            # --- late rules ---------------------------------------------
            am, mk, again2 = self._late(p, m, mk, am, bh, bv)
            again |= again2

            after = (p, tuple(m), tuple(ha), tuple(ban), am, mk,
                     tuple(bh), tuple(bv))
            first = False
            # `again` only re-triggers when the grid actually changed over the
            # whole iteration -- the interpreter's own guard against rules that
            # keep matching without doing anything.
            if not again or after == before:
                break
            m[MK] = mk
            new = (p, *m)
            m[MK] = 0
            if self.won(new) and not (am | ha[0] | ha[1] | ha[2] | ha[3]
                                      or any(ban) or bh[0] | bh[1] | bh[2]
                                      or bv[0] | bv[1] | bv[2]):
                break
        m[MK] = mk
        new = (p, *m)
        return new, self.won(new)

    # -- one press's forces --------------------------------------------------
    def _forces(self, p, m, mk, d):
        """Assign the push forces, run the cancel test and resolve movement.

        Returns ``(player, markers)`` or None when the turn is CANCELLED (a
        finished face shoved into something solid), which the interpreter
        implements by restoring the whole grid."""
        occ = self.walls | (1 << p)
        for k in _MOVABLE:
            occ |= m[k]
        t = self.nb(p, d)
        if t < 0:
            return p, mk
        bit = 1 << t
        pushed = 0
        mk_forced = 0
        fulls = m[TL] | m[TR] | m[BL] | m[BR]
        parts = m[B1] | m[B2] | m[B3] | m[B4]
        if not (occ & bit):
            pass
        elif bit & m[HAPPY]:
            pushed = bit
        elif bit & (m[HEART] | m[HL] | m[HR]):
            pushed = bit
            # The two merge rules are locked to the `right` direction: a half
            # heart only ever merges with the half it is shoved into head-on.
            if d == "right" and (bit & m[HL]):
                nxt = self.nb(t, "right")
                if nxt >= 0 and ((m[HR] >> nxt) & 1):
                    m[HL] &= ~bit
                    m[HR] &= ~(1 << nxt)
                    m[HEART] |= 1 << nxt
                    pushed = 0
                    occ &= ~bit
            elif d == "left" and (bit & m[HR]):
                nxt = self.nb(t, "left")
                if nxt >= 0 and ((m[HL] >> nxt) & 1):
                    m[HR] &= ~bit
                    m[HL] &= ~(1 << nxt)
                    m[HEART] |= 1 << nxt
                    pushed = 0
                    occ &= ~bit
        elif bit & parts:
            pushed = bit
        elif bit & fulls:
            # `[ moving BigFaceFull | BigFaceFull ]` is a NON-late rule, and
            # `_execute_rule` re-runs the whole direction list until nothing
            # changes -- so the force reaches the entire connected component of
            # finished quadrants, not just the 2x2 that was pushed. Two faces
            # parked side by side move as one object. (The late rules are the
            # ones with the per-direction-only fixpoint; see `_late`.)
            moving = bit
            while True:
                grown = fulls & self.nbrs(moving) & ~moving
                if not grown:
                    break
                moving |= grown
            solid = (self.walls | m[SAD] | m[HAPPY]
                     | m[HEART] | m[HL] | m[HR])
            if self.sh(moving, d) & solid:
                return None                          # -> cancel
            pushed = moving
            mk |= moving
            mk_forced = moving
        forced = (1 << p) | pushed
        moved = self._resolve(forced, occ, d)
        if moved:
            for k in _MOVABLE:
                mv = m[k] & moved
                if mv:
                    m[k] = (m[k] & ~mv) | self.sh(mv, d)
            if (moved >> p) & 1:
                p = self.nb(p, d)
        if mk_forced:
            # `[ moving BigFaceFull ] -> [ ... moving MoveMarker ]` marks only
            # the finished quadrants, and markers ride their own collision
            # layer -- so a marker resolves independently of whether the face
            # under it got through, and a marker sitting on anything ELSE the
            # player pushed does not move at all.
            mmoved = self._resolve(mk_forced, mk, d)
            if mmoved:
                mk = (mk & ~mmoved) | self.sh(mmoved, d)
        return p, mk

    def _resolve(self, forced, occ, d):
        """Return the subset of ``forced`` that actually moves.

        The interpreter traces each force forward through objects carrying the
        SAME force and moves the whole run when the cell past its end is free;
        one direction per turn here, so runs never conflict."""
        if not forced:
            return 0
        blockers = occ & ~forced
        moved = 0
        rest = forced & ~self.sh(forced, d)          # run starts
        while rest:
            b = rest & -rest
            rest ^= b
            e = b.bit_length() - 1
            run = b
            while True:
                t = self.nb(e, d)
                if t >= 0 and ((forced >> t) & 1):
                    run |= 1 << t
                    e = t
                else:
                    break
            t = self.nb(e, d)
            if t >= 0 and not ((blockers >> t) & 1):
                moved |= run
        return moved

    # -- the late rule block -------------------------------------------------
    def _late(self, p, m, mk, am, bh, bv):
        """The whole ``late`` rule list, in the .txt's order: fire the beams of
        a face that just moved, flood them, clear the move markers, cheer up
        whatever the beams reached, run the contagion, then the finished-face
        adjacency, then the heart, then stamp a completed 2x2 for assembly.

        Mutates ``m`` / ``bh`` / ``bv`` in place; returns the new markers and
        whether any rule asked for another tick."""
        again = False
        blockers = (self.walls | (1 << p if p >= 0 else 0)
                    | m[SAD] | m[HAPPY] | m[HEART] | m[HL] | m[HR]
                    | m[B1] | m[B2] | m[B3] | m[B4]
                    | m[TL] | m[TR] | m[BL] | m[BR])
        free = self.full & ~blockers
        if mk:
            for corner, bdir in ((TL, "left"), (BL, "left"),
                                 (TR, "right"), (BR, "right"),
                                 (TL, "up"), (TR, "up"),
                                 (BL, "down"), (BR, "down")):
                src = m[corner] & mk
                if not src:
                    continue
                tgt = self.sh(src, bdir) & free
                if tgt:
                    again = True
                    if bdir in ("left", "right"):
                        bh[0] |= tgt
                    else:
                        bv[0] |= tgt
            if bh[0] or bv[0]:
                # A beam is a FLOOD along its axis, not a ray: it fills the
                # whole run of free cells it can reach in that row / column.
                while True:
                    gh = ((self.sh(bh[0], "left") | self.sh(bh[0], "right"))
                          & free & ~bh[0])
                    gv = ((self.sh(bv[0], "up") | self.sh(bv[0], "down"))
                          & free & ~bv[0])
                    if not (gh | gv):
                        break
                    bh[0] |= gh
                    bv[0] |= gv
            mk &= ~(m[TL] | m[TR] | m[BL] | m[BR])

        BH = bh[0] | bh[1] | bh[2]
        BV = bv[0] | bv[1] | bv[2]
        if BH:
            near = self.sh(BH, "left") | self.sh(BH, "right")
            hit = near & m[SAD]
            if hit:
                m[SAD] &= ~hit
                m[HAPPY] |= hit
            hit = near & m[WSAD]
            if hit:
                m[WSAD] &= ~hit
                m[WHAPPY] |= hit
        if BV:
            near = self.sh(BV, "up") | self.sh(BV, "down")
            hit = near & m[SAD]
            if hit:
                m[SAD] &= ~hit
                m[HAPPY] |= hit
            hit = near & m[WSAD]
            if hit:
                m[WSAD] &= ~hit
                m[WHAPPY] |= hit

        # Contagion: one fixpoint per direction, in the interpreter's order.
        if m[SAD]:
            for dd in _DIRS:
                back = _OPP[dd]
                while True:
                    grown = m[SAD] & self.sh(m[HAPPY], back)
                    if not grown:
                        break
                    m[SAD] &= ~grown
                    m[HAPPY] |= grown
                if not m[SAD]:
                    break
        if m[WSAD] and m[HAPPY]:
            hit = m[WSAD] & self.nbrs(m[HAPPY])
            if hit:
                m[WSAD] &= ~hit
                m[WHAPPY] |= hit
        fulls = m[TL] | m[TR] | m[BL] | m[BR]
        if fulls:
            near = self.nbrs(fulls)
            hit = near & m[SAD]
            if hit:
                m[SAD] &= ~hit
                m[HAPPY] |= hit
            hit = near & m[WSAD]
            if hit:
                m[WSAD] &= ~hit
                m[WHAPPY] |= hit

        # A heart between two faces cheers both and is consumed.
        if m[HEART]:
            faces = m[SAD] | m[HAPPY]
            for dd in ("up", "left"):
                back = _OPP[dd]
                match = m[HEART] & self.sh(faces, dd) & self.sh(faces, back)
                if match:
                    again = True
                    am |= match
                    mk &= ~match
                    conv = (self.sh(match, dd) | self.sh(match, back)) & m[SAD]
                    if conv:
                        m[SAD] &= ~conv
                        m[HAPPY] |= conv

        # A 2x2 of quadrants stamps itself for the assembly animation. The
        # stamp is a FaceMarker, and every marker shares one collision layer,
        # so laying it down DESTROYS a move marker that was sitting there --
        # which is the only way a stray one is ever cleaned up. (The four
        # markers are laid even when the pairs never make a square, and wiped
        # again three rules later; the side effect on the move marker is not.)
        pair12 = m[B1] & self.sh(m[B2], "left")
        pair34 = m[B3] & self.sh(m[B4], "left")
        fm = pair12 | self.sh(pair12, "right")
        fm |= pair34 | self.sh(pair34, "right")
        if fm:
            mk &= ~fm
            am &= ~fm
            top = (m[B1] & fm) & self.sh(m[B3] & fm, "up")
            if top:
                cells = top | self.sh(top, "down")
                am |= cells
                fm &= ~cells
                mk &= ~cells
            top = (m[B2] & fm) & self.sh(m[B4] & fm, "up")
            if top:
                cells = top | self.sh(top, "down")
                am |= cells
                fm &= ~cells
                mk &= ~cells
                again = True
        return am, mk, again

    # -- the macro search ----------------------------------------------------
    def occupancy(self, state):
        """Every cell a walker is blocked by: walls and the whole push layer."""
        occ = self.walls
        for k in _SOLO:
            occ |= state[1 + k]
        return occ

    def walks(self, state):
        """``(dist, parent)`` -- shortest walks from the player over free cells.

        Walking is a genuine no-op on a SETTLED board (no rule reads the player
        except the beams, and there are none at rest), so a macro's walk can be
        costed here and its board state taken from the push alone."""
        occ = self.occupancy(state)
        p = state[0]
        dist = {p: 0}
        parent: dict[int, tuple[int, str]] = {}
        queue = deque([p])
        while queue:
            cur = queue.popleft()
            for d in _DIRS:
                nxt = self.nb(cur, d)
                if nxt < 0 or nxt in dist or (occ >> nxt) & 1:
                    continue
                dist[nxt] = dist[cur] + 1
                parent[nxt] = (cur, d)
                queue.append(nxt)
        return dist, parent

    def path_to(self, parent, cell):
        """The walk `walks` found to ``cell``, as a list of directions."""
        out = []
        while cell in parent:
            cell, d = parent[cell]
            out.append(d)
        out.reverse()
        return out

    def macros(self, state):
        """Every ``walk to a cell and push`` from ``state``, plus the WAIT.

        A press that cannot move the player is still a turn, and the contagion
        rule only reaches a fixpoint in one direction per pass -- so "stand
        still" is a real move that can flip a frown a bend away. It is offered
        only when it actually changes something."""
        dist, parent = self.walks(state)
        pushable = 0
        for k in _SOLO:
            pushable |= state[1 + k]
        out = []
        for cell, dd in dist.items():
            for d in _DIRS:
                tgt = self.nb(cell, d)
                if tgt < 0 or not ((pushable >> tgt) & 1):
                    continue
                walked = (cell, *state[1:])
                nxt, won = self.step(walked, d)
                if nxt[1:] == state[1:]:
                    continue                     # nothing moved: not a macro
                out.append((self.path_to(parent, cell) + [d], nxt, won,
                            dd + 1))
        waited, won = self.step(state, "action")
        if waited != state:
            out.append((["action"], waited, won, 1))
        return out

    def region(self, state, dist=None):
        """Canonical id of the player's walkable region -- its lowest cell.

        Two states with the same board and the same region admit exactly the
        same futures, so merging them cannot lose a plan; it can only mis-cost
        one by the region's diameter (the standard sokoban canonicalisation).
        """
        if dist is None:
            dist, _p = self.walks(state)
        return min(dist)

    def dists(self):
        """All-pairs shortest walk over the WALLS only (pieces ignored).

        The gather heuristic asks "how far is this quadrant from that slot" for
        a lot of pairs, and the boards are at most 143 cells, so the whole
        matrix is one cheap sweep per level and a lookup afterwards. Ignoring
        the pieces is what makes it a lower bound worth steering with: they
        move, the walls do not."""
        if self._dist is None:
            inf = _INF
            self._dist = {}
            for s in range(self.n):
                if (self.walls >> s) & 1:
                    continue
                row = [inf] * self.n
                row[s] = 0
                queue = deque([s])
                while queue:
                    cur = queue.popleft()
                    for d in _DIRS:
                        nxt = self.nb(cur, d)
                        if nxt >= 0 and row[nxt] == inf and not (self.walls >> nxt) & 1:
                            row[nxt] = row[cur] + 1
                            queue.append(nxt)
                self._dist[s] = row
        return self._dist

    def gather(self, state):
        """How far the loose quadrants are from being 2x2s.

        A count of loose quadrants is FLAT over the whole approach -- it only
        drops on the push that completes a face -- so the search has nothing to
        steer with and levels 13-19 are hopeless without this. For each set of
        four it is the cheapest "walk every quadrant to its slot" over the four
        anchor positions the pieces themselves imply, matched greedily set by
        set. Same shape as the push-distance tables `PSSokobanExpert` uses, in
        a game that is not a sokoban."""
        groups = [_cells(state[1 + k]) for k in (B1, B2, B3, B4)]
        if not all(groups):
            return 0
        dist = self.dists()
        offs = (0, 1, self.w, self.w + 1)
        total = 0
        while all(groups):
            best = _INF
            pick = None
            for quad in itertools.product(*groups):
                cost = _INF
                for q0, off0 in zip(quad, offs):
                    anchor = q0 - off0
                    if anchor < 0 or anchor >= self.n or anchor not in dist:
                        continue
                    acc = 0
                    for q, off in zip(quad, offs):
                        slot = anchor + off
                        row = dist.get(q)
                        if slot >= self.n or row is None or row[slot] >= _INF:
                            acc = _INF
                            break
                        acc += row[slot]
                    if acc < cost:
                        cost = acc
                if cost < best:
                    best, pick = cost, quad
            if best >= _INF:
                return _INF
            total += best
            for gi, q in enumerate(pick):
                groups[gi] = [x for x in groups[gi] if x != q]
        return total

    def heuristic(self, state, sad_weight, gather_weight):
        """What is left to do, in rough presses."""
        left = (bin(state[1 + SAD]).count("1")
                + bin(state[1 + WSAD]).count("1"))
        return sad_weight * left + gather_weight * self.gather(state)

    def best_first(self, state, cap, sad_weight, gather_weight):
        """Weighted best-first over the macros, for the levels the exhaustive
        BFS cannot hold. Returns ``(plan, nodes)``; the plan is a genuine win
        path but is NOT proved shortest."""
        h0 = self.heuristic(state, sad_weight, gather_weight)
        pq = [(h0, 0, 0, state, [])]
        seen = {(state[1:], self.region(state)): 0}
        counter = nodes = 0
        while pq and nodes < cap:
            _f, cost, _c, cur, path = heapq.heappop(pq)
            for presses, nxt, won, price in self.macros(cur):
                nodes += 1
                if won:
                    return path + presses, nodes
                key = (nxt[1:], self.region(nxt))
                ncost = cost + price
                if seen.get(key, 1 << 30) <= ncost:
                    continue
                seen[key] = ncost
                counter += 1
                heapq.heappush(
                    pq, (ncost + self.heuristic(nxt, sad_weight, gather_weight),
                         ncost, counter, nxt, path + presses))
        return None, nodes

    # -- the exhaustive search -----------------------------------------------
    def bfs(self, state, cap):
        """Layered breadth-first search over the five presses.

        Returns ``(plan, optsets)`` -- a SHORTEST press sequence to a win and,
        for every step, the exact set of presses that are equally shortest --
        or ``(None, None)`` when the level cannot be won within ``cap`` states.

        Every press costs 1, so BFS is already the optimal search; what it also
        buys is the labelling. A state ``j`` can only be an optimal successor
        of ``i`` when ``depth[j] == depth[i] + 1``: a cross edge to a shallower
        ``j`` would need ``depth[j] + f(j) >= d*`` with ``depth[j] <
        depth[i] + 1``, i.e. ``f(j) > d* - depth[i] - 1``, which is already too
        long to be on a shortest path. So "can still win in exactly the presses
        remaining" propagates backwards through the depth layers of the forward
        edges alone -- one sweep, no reverse adjacency, no re-solve.
        """
        if self.won(state):
            return [], []
        states = [state]
        index = {state: 0}
        depth = [0]
        # succ[5 * i + k]: the successor of press k, -1 for a WIN, -2 for a
        # state never expanded. A flat array because these searches run to
        # hundreds of thousands of states and a list per state would be most
        # of the memory.
        succ = array('i', [-2] * 5)
        frontier = [0]
        won_depth = None
        while frontier and won_depth is None:
            nxt_frontier = []
            for i in frontier:
                s = states[i]
                base = 5 * i
                for k, press in enumerate(PRESSES):
                    ns, won = self.step(s, press)
                    if won:
                        succ[base + k] = -1
                        won_depth = depth[i] + 1
                        continue
                    j = index.get(ns)
                    if j is None:
                        if len(states) >= cap:
                            return None, None
                        j = len(states)
                        index[ns] = j
                        states.append(ns)
                        depth.append(depth[i] + 1)
                        succ.extend((-2, -2, -2, -2, -2))
                        nxt_frontier.append(j)
                    succ[base + k] = j
            frontier = nxt_frontier
        if won_depth is None:
            return None, None

        # `tight[i]`: a win is still reachable from i in exactly
        # ``won_depth - depth[i]`` presses.
        tight = bytearray(len(states))
        layers: list[list[int]] = [[] for _ in range(won_depth + 1)]
        for i, dep in enumerate(depth):
            layers[dep].append(i)
        for i in layers[won_depth - 1]:
            base = 5 * i
            if any(succ[base + k] == -1 for k in range(5)):
                tight[i] = 1
        for dep in range(won_depth - 2, -1, -1):
            for i in layers[dep]:
                base = 5 * i
                for k in range(5):
                    j = succ[base + k]
                    if j >= 0 and depth[j] == dep + 1 and tight[j]:
                        tight[i] = 1
                        break

        plan, optsets = [], []
        i = 0
        for dep in range(won_depth):
            base = 5 * i
            best = []
            for k, press in enumerate(PRESSES):
                j = succ[base + k]
                if dep == won_depth - 1:
                    if j == -1:
                        best.append((press, j))
                elif j >= 0 and depth[j] == dep + 1 and tight[j]:
                    best.append((press, j))
            plan.append(best[0][0])
            optsets.append([pr for pr, _j in best])
            i = best[0][1]
        return plan, optsets


# ---------------------------------------------------------------------------
# Reading the interpreter's grid
# ---------------------------------------------------------------------------

def read_board(eng, g):
    """``(_Board, state)`` for the interpreter's current grid."""
    inv = {v: k for k, v in g.obj_name_to_idx.items()}
    by_name = {name: k for k, name in _MASK_NAMES.items()}
    w = eng.width
    masks = [0] * _NMASK
    walls = 0
    player = -1
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            i = r * w + c
            for o in cell:
                name = inv[o]
                if name == "player":
                    player = i
                elif name == "wall":
                    walls |= 1 << i
                elif name in by_name:
                    masks[by_name[name]] |= 1 << i
    return _Board(eng.height, w, walls), (player, *masks)


def write_board(eng, g, board, state):
    """Load ``(board, state)`` into the interpreter -- the inverse of
    `read_board`, used by `_fuzz` to seed random positions."""
    idx = g.obj_name_to_idx
    h, w = board.h, board.w
    grid = [[{idx["background"]} for _ in range(w)] for _ in range(h)]
    for i in range(board.n):
        if (board.walls >> i) & 1:
            grid[i // w][i % w].add(idx["wall"])
    for k, name in _MASK_NAMES.items():
        mask = state[1 + k]
        while mask:
            b = mask & -mask
            mask ^= b
            i = b.bit_length() - 1
            grid[i // w][i % w].add(idx[name])
    if state[0] >= 0:
        grid[state[0] // w][state[0] % w].add(idx["player"])
    eng.grid, eng.height, eng.width = grid, h, w
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()
    eng._rule_win = False


# ---------------------------------------------------------------------------
# Differential fuzz against the real interpreter
# ---------------------------------------------------------------------------

def _random_state(board, rng, scenario="random"):
    """A random legal placement on ``board``'s free cells.

    Uniform random placement almost never builds a 2x2 of quadrants or drops a
    heart between two faces, so the interesting half of the ruleset would never
    be reached: the fuzz would report "the model matches" having exercised
    walking and nothing else. ``scenario`` SEEDS those configurations one push
    away from firing -- which is the same lesson [[escaping-limbo-solver]]
    records about fuzzing from a level's own solution prefixes."""
    free = [i for i in range(board.n) if not (board.walls >> i) & 1]
    rng.shuffle(free)
    masks = [0] * _NMASK
    used = set()
    w = board.w

    def put(k, i):
        if i is None or i in used or (board.walls >> i) & 1 or not 0 <= i < board.n:
            return False
        masks[k] |= 1 << i
        used.add(i)
        return True

    def square(top_left):
        r, c = divmod(top_left, w)
        if r + 1 >= board.h or c + 1 >= board.w:
            return None
        cells = [top_left, top_left + 1, top_left + w, top_left + w + 1]
        if any((board.walls >> i) & 1 or i in used for i in cells):
            return None
        return cells

    if scenario in ("assembly", "fullface"):
        for _try in range(30):
            cells = square(rng.choice(free))
            if cells is None:
                continue
            kinds = ((B1, B2, B3, B4) if scenario == "assembly"
                     else (TL, TR, BL, BR))
            loose = rng.randrange(4) if scenario == "assembly" else -1
            for j, (k, i) in enumerate(zip(kinds, cells)):
                if j == loose:
                    # Park the missing quadrant one push away, so a single
                    # press completes the face.
                    nxt = board.nb(i, rng.choice(_DIRS))
                    put(k, nxt if nxt >= 0 else i)
                else:
                    put(k, i)
            break
    forced_player = None
    if scenario == "hearts":
        # A heart one push away from landing between two faces, with the
        # player already lined up behind it: the late rules run AFTER the
        # move, so this is the only way the rule ever fires.
        for _try in range(30):
            i = rng.choice(free)
            axis = rng.choice(("up", "left"))
            a, b = board.nb(i, axis), board.nb(i, _OPP[axis])
            cross = "left" if axis == "up" else "up"
            hcell = board.nb(i, cross)
            pcell = board.nb(hcell, cross) if hcell >= 0 else -1
            if min(a, b, hcell, pcell) < 0:
                continue
            if not (put(rng.choice([SAD, HAPPY]), a)
                    and put(rng.choice([SAD, HAPPY]), b)
                    and put(HEART, hcell)):
                continue
            if pcell not in used and not (board.walls >> pcell) & 1:
                forced_player = pcell
                used.add(pcell)
            break
        # A pair of halves ready to merge, player behind the left one.
        for _try in range(20):
            i = rng.choice(free)
            j = board.nb(i, "right")
            k = board.nb(i, "left")
            if j < 0 or k < 0:
                continue
            if put(HL, i) and put(HR, j) and forced_player is None:
                if k not in used and not (board.walls >> k) & 1:
                    forced_player = k
                    used.add(k)
            break
    if scenario == "chain":
        # A run of frowns with a happy face at one end: the contagion rule
        # flips the whole run in a single press, but only along a straight
        # run in the direction its fixpoint reaches.
        for _try in range(20):
            i = rng.choice(free)
            d = rng.choice(_DIRS)
            cells = [i]
            for _k in range(rng.randint(2, 4)):
                nxt = board.nb(cells[-1], d)
                if nxt < 0:
                    break
                cells.append(nxt)
            if len(cells) < 3:
                continue
            for c in cells[:-1]:
                put(SAD, c)
            # The happy face is one push short of touching the run.
            tail = board.nb(cells[-1], d)
            put(HAPPY, tail if tail >= 0 else cells[-1])
            head = board.nb(tail if tail >= 0 else cells[-1], d)
            if head >= 0 and head not in used and not (board.walls >> head) & 1:
                forced_player = head
                used.add(head)
            break

    weights = ([SAD] * 5 + [HAPPY] * 3 + [HEART] * 2 + [HL, HR]
               + [B1, B2, B3, B4] * 2 + [TL, TR, BL, BR])
    for i in free:
        if i in used or rng.random() > 0.30:
            continue
        put(rng.choice(weights), i)
    # Sad / happy walls decorate wall cells only, which is how the levels ship
    # them (`D = WallSad and Wall`).
    walls = [i for i in range(board.n) if (board.walls >> i) & 1]
    rng.shuffle(walls)
    for i in walls[:rng.randint(0, len(walls) // 2)]:
        masks[WSAD if rng.random() < 0.7 else WHAPPY] |= 1 << i
    # A stray move marker, on its own collision layer and so on top of
    # whatever else is there. They are rare on a real board and permanent when
    # they happen, so the fuzz seeds them rather than waiting for one.
    if rng.random() < 0.25:
        masks[MK] |= 1 << rng.choice(free)
    if forced_player is not None:
        return (forced_player, *masks)
    rest = [i for i in free if i not in used]
    player = rest[0] if rest else -1
    return (player, *masks)


def _greedy_press(board, state, rng):
    """A press biased towards whatever the player could push.

    Uniform presses on a crowded board mostly bump walls; steering at the
    nearest pushable is what makes the fuzz actually exercise the push, merge
    and cancel paths."""
    if rng.random() < 0.3 or state[0] < 0:
        return rng.choice(PRESSES)
    targets = 0
    for k in _SOLO:
        targets |= state[1 + k]
    if not targets:
        return rng.choice(PRESSES)
    pr, pc = divmod(state[0], board.w)
    best, bestd = None, 1 << 30
    rest = targets
    while rest:
        b = rest & -rest
        rest ^= b
        i = b.bit_length() - 1
        r, c = divmod(i, board.w)
        d = abs(r - pr) + abs(c - pc)
        if d < bestd:
            best, bestd = (r, c), d
    dr, dc = best[0] - pr, best[1] - pc
    opts = []
    if dr:
        opts.append("down" if dr > 0 else "up")
    if dc:
        opts.append("right" if dc > 0 else "left")
    return rng.choice(opts) if opts else rng.choice(PRESSES)


def _fuzz(n_boards: int = 400, n_steps: int = 25, seed: int = 0,
          verbose: bool = False) -> int:
    """Random boards played press-for-press against the interpreter.

    This is the only thing standing between `_Board` and a silently wrong
    corpus. The boards are deliberately crowded: quadrants that assemble by
    accident, hearts landing between faces, contagion chains that turn
    corners, finished faces shoved into each other."""
    game = FrownSolver().make_game(0)
    eng, g = game._engine, game._game
    rng = random.Random(seed)
    mismatches = steps = 0
    counters = {"assembled": 0, "beams": 0, "hearts": 0, "merges": 0,
                "cancels": 0, "contagion": 0, "pushes": 0}
    scenarios = ("assembly", "fullface", "hearts", "chain", "random", "level")
    for bi in range(n_boards):
        scenario = scenarios[bi % len(scenarios)]
        if scenario == "level":
            # The shipped boards themselves, walked at random. Random boards
            # reach the rare rules; only the real levels reach the rules in the
            # combinations the corpus will actually be recorded from.
            game.set_level(rng.randrange(game.n_levels))
            board, state = read_board(eng, g)
        else:
            h, w = rng.choice([5, 6, 7]), rng.choice([5, 6, 7])
            walls = 0
            for r in range(h):
                for c in range(w):
                    edge = r in (0, h - 1) or c in (0, w - 1)
                    if edge or rng.random() < 0.12:
                        walls |= 1 << (r * w + c)
            board = _Board(h, w, walls)
            state = _random_state(board, rng, scenario)
            write_board(eng, g, board, state)
        for _s in range(n_steps):
            press = _greedy_press(board, state, rng)
            before = state
            eng.step(press)
            truth_won = eng.check_win()
            _b, truth = read_board(eng, g)
            state, won = board.step(state, press)
            steps += 1
            if state != truth or won != truth_won:
                mismatches += 1
                print(f"MISMATCH board {bi} step {_s} press={press}")
                _dump(board, before, "before")
                _dump(board, state, "model ")
                _dump(board, truth, "engine")
                if mismatches > 3:
                    return 1
                state = truth
                continue
            # coverage
            if before[1 + B1] != state[1 + B1] and state[1 + TL]:
                counters["assembled"] += 1
            if state[1 + TL] | state[1 + TR] | state[1 + BL] | state[1 + BR]:
                counters["beams"] += 1
            if bin(before[1 + HEART]).count("1") > bin(state[1 + HEART]).count("1"):
                counters["hearts"] += 1
            if bin(state[1 + HEART]).count("1") > bin(before[1 + HEART]).count("1"):
                counters["merges"] += 1
            if before == state and press in _DELTA:
                counters["cancels"] += 1
            if bin(before[1 + SAD]).count("1") > bin(state[1 + SAD]).count("1"):
                counters["contagion"] += 1
            for k in _SOLO:
                if before[1 + k] != state[1 + k]:
                    counters["pushes"] += 1
                    break
    print(f"fuzz: {steps} transitions over {n_boards} boards, "
          f"{mismatches} mismatches")
    print("      coverage " + "  ".join(f"{k}={v}" for k, v in counters.items()))
    return 0 if not mismatches else 1


def _dump(board, state, tag):
    """Print a state as ASCII, for a fuzz mismatch."""
    ch = {SAD: "s", HAPPY: "h", HEART: "L", HL: "{", HR: "}", B1: "F",
          B2: "A", B3: "C", B4: "E", TL: "T", TR: "R", BL: "B", BR: "X"}
    print(f"  {tag}: player={state[0]} markers={_cells(state[1 + MK])}")
    for r in range(board.h):
        row = ""
        for c in range(board.w):
            i = r * board.w + c
            if state[0] == i:
                row += "p"
                continue
            got = [s for k, s in ch.items() if (state[1 + k] >> i) & 1]
            if got:
                row += got[0]
            elif (board.walls >> i) & 1:
                if (state[1 + WSAD] >> i) & 1:
                    row += "D"
                elif (state[1 + WHAPPY] >> i) & 1:
                    row += "Y"
                else:
                    row += "#"
            else:
                row += "."
        print("    " + row)


# ---------------------------------------------------------------------------
# The expert and the BaseSolver harness
# ---------------------------------------------------------------------------

class FrownExpert(PSExpert):
    """Plans read off the native model; `PSExpert` supplies the in-memory memo,
    the on-disk start-plan cache and the snapshot discipline, so only `_search`
    is overridden."""

    directions = PRESSES
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        self._board = None
        self._board_sig = None

    def heuristic(self, eng) -> int:                 # pragma: no cover
        raise NotImplementedError("the model owns the search; see _Board")

    def _board_for(self, board):
        """Reuse one `_Board` per level so its caches survive across searches."""
        sig = board.signature()
        if sig != self._board_sig:
            self._board, self._board_sig = board, sig
        return self._board

    def _search(self, eng) -> list | None:
        board, state = read_board(eng, self.g)
        board = self._board_for(board)
        return _solve(board, state)


class FrownSolver(PSAStarSolver):
    game_id = "puzzlescript_frown_inversion_squad"
    game_name = GAME_NAME
    expert_cls = FrownExpert

    #: Record against the game folder's adapter -- the one `game_envs` hands a
    #: live agent -- so a later step limit or sprite fix there cannot silently
    #: diverge from what is taped here.
    game_module_id = "ps:frown_inversion_squad"

    #: The levels no search here wins inside the caps above; skipped up front
    #: so discovery does not burn the whole ladder on them at every startup.
    #: See the module docstring for what makes each of them hard.
    skip_levels = frozenset({15, 16, 17, 18, 19})

    #: The longest plan that ships is level 14's; the rest is margin inside the
    #: adapter's 200-action per-level budget.
    max_steps = 190


#: States the exhaustive BFS may hold before the macro search takes over. Sized
#: as a MEMORY budget, not a patience one: a state is a 16-tuple of bignums, so
#: ~800k of them plus their index is already >1 GB, and `parallelize_generator`
#: runs several shards at once.
BFS_CAP = 800_000

#: The weight ladder for the macro search, least greedy first: ``(sad weight,
#: gather weight, node cap)``. A low weight keeps the plan near-shortest but
#: cannot reach the crowded levels; a high one always arrives but wanders. Take
#: the first rung that returns, so the levels that CAN be planned tightly are.
#: (Same shape as the ladder in [[entrepotphage-demake-solver]].)
LADDER = ((2, 1, 60_000), (8, 4, 120_000), (30, 10, 700_000))

#: Presses a plan may use. The adapter's 200-action per-level budget is reset by
#: the RESET that ends the exploration prefix, so the whole 200 belongs to the
#: plan -- but `record_level` also stops after ``FrownSolver.max_steps``
#: presses, so this has to stay under THAT, or a plan that fits the game would
#: be cut off half-way and its level silently dropped.
PRESS_BUDGET = 185


def _solve(board, state, bfs_cap=BFS_CAP, ladder=LADDER, budget=PRESS_BUDGET):
    """A plan for ``state``: shortest if the exhaustive BFS can hold the level,
    else the shortest the weight ladder returns.

    Every rung is run, not just the first that answers: a greedier rung is not
    only faster, it sometimes finds a shorter plan than a rung that had to
    wander to reach the same win."""
    plan, optsets = board.bfs(state, bfs_cap)
    if plan is not None:
        return Plan(plan, optsets)
    best = None
    for sad_weight, gather_weight, cap in ladder:
        presses, _nodes = board.best_first(state, cap, sad_weight,
                                           gather_weight)
        if presses is None:
            continue
        presses = shorten(board, state, presses)
        if best is None or len(presses) < len(best):
            best = presses
    if best is None or len(best) > budget:
        return None
    return Plan(best, annotate(board, state, best))


def shorten(board, state, presses, window=24):
    """Delete the longest block of presses the plan can win without.

    A best-first plan wanders: it walks somewhere, changes its mind and walks
    back, and the whole detour is a block whose deletion leaves the rest of the
    plan replaying identically. Longest block first, re-simulated on the model
    each time, which is cheap enough to be worth it -- and the result is still
    a genuine win path because the win is what the test asks for."""
    cur = list(presses)
    improved = True
    while improved:
        improved = False
        for length in range(min(window, len(cur)), 0, -1):
            for start in range(len(cur) - length + 1):
                trial = cur[:start] + cur[start + length:]
                s = state
                won = False
                for press in trial:
                    s, won = board.step(s, press)
                    if won:
                        break
                if won:
                    cur = trial
                    improved = True
                    break
            if improved:
                break
    # Drop any tail past the winning press.
    s = state
    for i, press in enumerate(cur):
        s, won = board.step(s, press)
        if won:
            return cur[:i + 1]
    return cur


def annotate(board, state, presses):
    """Per-step optimal SETS for a leg plan.

    A leg plan is not proved shortest, so it cannot claim the full tie set the
    BFS measures -- but the WALK stretches inside it are genuinely order-free
    and labelling one arbitrary interleaving of the two axes as "the" answer
    would be teaching a coin flip. Every press that only moved the player is
    part of a walk, and its tie set is every direction that is also a no-op and
    also one step closer to where the walk ends. Everything else (a push, a
    merge, a wait) is labelled with itself alone.
    """
    sets: list[list[str]] = []
    # Split the plan into maximal runs of presses that moved nothing but the
    # player; the run's destination is where the next real move happens from.
    states = [state]
    walk = []
    cur = state
    for press in presses:
        nxt, _won = board.step(cur, press)
        walk.append(nxt[1:] == cur[1:] and nxt[0] != cur[0])
        states.append(nxt)
        cur = nxt
    i = 0
    while i < len(presses):
        if not walk[i]:
            sets.append([presses[i]])
            i += 1
            continue
        j = i
        while j < len(presses) and walk[j]:
            j += 1
        dest = states[j][0]
        # Distances to the walk's end over the cells free at the time.
        occ = board.occupancy(states[i])
        dist = {dest: 0}
        queue = deque([dest])
        while queue:
            c0 = queue.popleft()
            for d in _DIRS:
                nxt = board.nb(c0, d)
                if nxt < 0 or nxt in dist or (occ >> nxt) & 1:
                    continue
                dist[nxt] = dist[c0] + 1
                queue.append(nxt)
        for k in range(i, j):
            here = states[k]
            best = []
            for d in _DIRS:
                ns, _w = board.step(here, d)
                if (ns[1:] == here[1:] and ns[0] in dist
                        and dist[ns[0]] == dist[here[0]] - 1):
                    best.append(d)
            sets.append(best or [presses[k]])
        i = j
    return sets


# ---------------------------------------------------------------------------
# Reports and checks
# ---------------------------------------------------------------------------

def _levels():
    solver = FrownSolver()
    game = solver.make_game(0)
    return game, FrownExpert(game)


def _report(levels=None) -> int:
    """Per-level size, what is wrong with it, plan length and tie coverage."""
    game, expert = _levels()
    eng = game._engine
    total = solved = 0
    for level in range(game.n_levels):
        if levels and level not in levels:
            continue
        game.set_level(level)
        board, state = read_board(eng, game._game)
        shape = (f"{bin(state[1 + SAD]).count('1'):2d} frowns "
                 f"{bin(state[1 + WSAD]).count('1'):2d} sad walls "
                 f"{bin(state[1 + B1] | state[1 + B2] | state[1 + B3] | state[1 + B4]).count('1'):2d} quadrants")
        if level in FrownSolver.skip_levels:
            print(f"level {level:2d}: {board.h:2d}x{board.w:2d}  {shape}  SKIPPED")
            continue
        found = expert.plan(eng, level)
        if found is None:
            print(f"level {level:2d}: {board.h:2d}x{board.w:2d}  {shape}  NO PLAN")
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(found)
        solved += 1
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"level {level:2d}: {board.h:2d}x{board.w:2d}  {shape}  "
              f"{len(found):3d} presses ({room}), {ties:3d} steps with a tie "
              f"set ({ties / max(1, len(found)):.0%})")
    print(f"total {total} presses over {solved} solved levels "
          f"({game.n_levels} shipped)")
    return 0


def _verify() -> int:
    """Replay every level's plan on the REAL interpreter and require a WIN.

    The plans come off a model; this is the line that says the model and the
    interpreter agree about the boards that actually ship."""
    game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        if level in FrownSolver.skip_levels:
            continue
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level:2d}: NO PLAN")
            bad += 1
            continue
        game.set_level(level)
        for d in plan:
            eng.step(d)
        ok = eng.check_win()
        bad += not ok
        print(f"level {level:2d}: {len(plan):3d} presses -> "
              f"{'WIN' if ok else 'NOT WON'}")
    print("verify clean" if not bad else f"VERIFY FAILED on {bad} levels")
    return 0 if not bad else 1


def _ties(levels=None) -> int:
    """Check every step's tie set independently of the code that produced it.

    Two different claims are made and each gets its own check.

      * A BFS level's labels are the EXACT set of equally-shortest presses, so
        the check is a fresh BFS from every successor: a press that reaches a
        win in the moves remaining must be labelled, and every labelled press
        must reach one.
      * A macro-search plan is not proved shortest, so its labels claim only
        that the alternative is the SAME walk taken in a different order. That
        is checked by SUBSTITUTION: take the alternative, walk on to where the
        plan's walk ended, replay the rest of the plan unchanged, and require
        the same length and the same win. A label that names a press which
        changes the board, or that costs a press, fails here.
    """
    game, expert = _levels()
    eng = game._engine
    bad = tested = 0
    for level in range(game.n_levels):
        if level in FrownSolver.skip_levels or (levels and level not in levels):
            continue
        game.set_level(level)
        board, state = read_board(eng, game._game)
        board = expert._board_for(board)
        plan = expert.plan(eng, level)
        if plan is None:
            continue
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        exact = board.bfs(state, BFS_CAP)[0] is not None
        states = [state]
        for press in plan:
            states.append(board.step(states[-1], press)[0])
        for i, press in enumerate(plan):
            here = states[i]
            if exact:
                remaining = len(plan) - i
                measured = []
                for alt in PRESSES:
                    ns, won = board.step(here, alt)
                    tested += 1
                    if won:
                        if remaining == 1:
                            measured.append(alt)
                        continue
                    sub, _o = board.bfs(ns, BFS_CAP)
                    if sub is not None and 1 + len(sub) == remaining:
                        measured.append(alt)
                if sorted(measured) != sorted(sets[i]):
                    bad += 1
                    print(f"level {level:2d} step {i:3d}: labelled {sets[i]} "
                          f"but measured {measured}")
                continue
            # Substitution check for the macro plans.
            j = i
            while j < len(plan) and states[j + 1][1:] == states[j][1:]:
                j += 1                       # the end of this walk stretch
            dest = states[j][0]
            for alt in sets[i]:
                tested += 1
                if alt == press:
                    continue
                ns, _won = board.step(here, alt)
                bridge = _walk_between(board, ns, dest)
                trial = ([alt] + bridge + list(plan[j:])) if bridge is not None else None
                ok = False
                if trial is not None and len(trial) == len(plan) - i:
                    cur = here
                    for step_press in trial:
                        cur, ok = board.step(cur, step_press)
                        if ok:
                            break
                if not ok:
                    bad += 1
                    print(f"level {level:2d} step {i:3d}: labelled {alt} "
                          f"but substituting it does not replay the plan")
        print(f"level {level:2d}: {len(plan):3d} steps checked "
              f"({'exact tie set' if exact else 'walk substitution'})")
    print(f"ties: {tested} alternatives checked, {bad} disagreements")
    return 0 if not bad else 1


def _walk_between(board, state, dest):
    """A shortest walk from ``state``'s player to ``dest``, or None."""
    if state[0] == dest:
        return []
    occ = board.occupancy(state)
    prev = {state[0]: None}
    queue = deque([state[0]])
    while queue:
        cur = queue.popleft()
        if cur == dest:
            break
        for d in _DIRS:
            nxt = board.nb(cur, d)
            if nxt < 0 or nxt in prev or (occ >> nxt) & 1:
                continue
            prev[nxt] = (cur, d)
            queue.append(nxt)
    if dest not in prev:
        return None
    out = []
    cur = dest
    while prev[cur] is not None:
        cur, d = prev[cur]
        out.append(d)
    out.reverse()
    return out


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    Composition, not object: the whole game is "is this thing sad or happy",
    and the two states of a face differ only in the colour of four pixels while
    the two states of a WALL differ only in the colour of four more. Both are
    checked here at every cell size the twenty boards render at, by filling a
    whole board with the composition and comparing whole frames -- the frame is
    upscaled to fill 64x64, so slicing one cell out by arithmetic lands in the
    wrong place (see [[explod-rendering-fixes]])."""
    game = FrownSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {
        "floor": (), "wall": ("wall",),
        "sad wall": ("wall", "wallsad"), "happy wall": ("wall", "wallhappy"),
        "player": ("player",),
        "sad face": ("facesad",), "happy face": ("facehappy",),
        "heart": ("heart",), "half heart L": ("heartleft",),
        "half heart R": ("heartright",),
    }
    for name in ("bigface1", "bigface2", "bigface3", "bigface4",
                 "bigfacetl", "bigfacetr", "bigfacebl", "bigfacebr"):
        comps[name] = (name,)

    sizes = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    def shoot(h, w, objs):
        eng.height, eng.width = h, w
        eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                     for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        return np.asarray(_render_frame(eng, g))

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {name: shoot(h, w, objs) for name, objs in comps.items()}
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        note = "OK" if not clashes else "IDENTICAL " + str(clashes)
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {note}")
    print("audit clean" if not bad else f"AUDIT FAILED: {bad} problems")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report([int(x) for x in sys.argv[sys.argv.index("--plans") + 1:]
                          if x.isdigit()]))
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--ties" in sys.argv:
        sys.exit(_ties([int(x) for x in sys.argv[sys.argv.index("--ties") + 1:]
                        if x.isdigit()]))
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--fuzz" in sys.argv:
        args = [int(x) for x in sys.argv[sys.argv.index("--fuzz") + 1:]
                if x.isdigit()]
        sys.exit(_fuzz(*args) if args else _fuzz())
    sys.exit(FrownSolver.main())
