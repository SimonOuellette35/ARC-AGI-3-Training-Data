"""Generate Phase-1 training data for the PuzzleScript game ps:upstairs_downstairs
("Upstairs Downstairs", Jacsn).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, a native model of them, the searches that model is driven with, the
tie labelling, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_upstairs_downstairs",
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
Every expert step carries an optimal-action set.

11 of the 20 levels are solved, in 910 presses, 9 of them PROVABLY SHORTEST:

    level  0   46 presses   exact field (229472 macro states)   shortest
    level  1   45 presses   exact field (695 macro states)      shortest
    level  2   73 presses   exact field (703 macro states)      shortest
    level  3   98 presses   admissible A*                       shortest
    level  4   43 presses   exact field (156307 macro states)   shortest
    level  5   68 presses   exact field (103252 macro states)   shortest
    level  6  125 presses   weighted A* (w=3)
    level  7  142 presses   admissible A*                       shortest
    level  9  130 presses   exact field (423131 macro states)   shortest
    level 16   76 presses   weighted A* (w=2)
    level 17   64 presses   exact field (86954 macro states)    shortest

The other nine are listed in `UpstairsDownstairsSolver.skip_levels`. Level 13 is
there for a reason that is PROVED rather than assumed: `--bound` exhausts every
macro state whose admissible estimate still fits in the adapter's 200-press
cut-off -- 588010 of them -- and finds no win, so that level has no plan short
enough to record at all.

The game
--------
A ONE-KEY platformer drawn flat: the board is a floor plan of a TWO-STOREY
building seen from above, and the same eleven-by-eleven square is both a room
downstairs and a room upstairs. ``Floor1`` (dark) is the ground floor,
``Floor2`` (white) is a piece of the storey above it, and the two are static --
no rule in the game creates or destroys a floor tile. Everything else moves.

There is exactly ONE player and it exists as one of two objects:

* ``Player1`` walks the GROUND. Floor2 tiles are its walls (they are the
  ceiling it cannot climb), it shoves ``Crate1`` boxes and it shoves
  STAIRCASES.
* ``Player2`` walks the STOREY ABOVE. It can stand on a Floor2 tile, on a
  staircase, or on top of a Crate1 -- a crate on the ground floor is a PILLAR
  holding up a square of the floor above -- and it shoves ``Crate2`` boxes,
  which are the boxes that live upstairs.

The win is ``All Player1 on Target1`` and ``All Player2 on Target2`` at once.
Only one player object is ever on the board, so one of the two is vacuously
true and the condition reads "stand on the goal of the storey you are on".
Eighteen of the twenty levels put the goal upstairs (``Target2``, pink), two
put it on the ground (``Target1``, purple).

The four staircases are the joins between the storeys, and each one is
one-way-ish: ``UDStair`` is climbed by walking DOWN into it, ``DUStair`` by
walking UP, ``LRStair`` by walking RIGHT and ``RLStair`` by walking LEFT.
Standing on the staircase you are Player2; walking further in the climb
direction steps off onto the upper storey, and walking back OUT of it turns you
into Player1 again on the ground. The staircases are pushable by Player1, which
is a mechanic in its own right on the levels where the only way up is in the
wrong place.

The mechanics, all MEASURED against the interpreter rather than read off the
.txt (``--selfcheck`` is the executable form):

* **A Crate2 shoved off its support FALLS, and lands as a Crate1.** Shove a
  Crate2 into a square with no Floor2 and no Crate1 under it and it drops to the
  ground floor as a Crate1 -- which is then a new PILLAR, i.e. a new square of
  upper floor to walk on. This is the whole game: an upstairs box is the raw
  material for the bridge you need, and the conversion is IRREVERSIBLE (nothing
  ever turns a Crate1 back into a Crate2).
* **A Crate2 can be stacked on a Crate1**, and a stack shoved off the pillar
  falls the same way, so one pillar can seed the next one along.
* **Player1 cannot shove a stacked crate at all** (``[> Player1 | Crate2 Crate1
  ] -> [stationary Player1 | ...]``), so to move a pillar on the ground you
  first have to get the box off the top of it from upstairs.
* **Shoving a crate off the EDGE of the map destroys it** -- but only when the
  crate is stacked on a Crate1 and the player is standing on Floor2 or on a
  staircase. The rule that seats the player on the Crate1 is a direct placement
  onto the crate's own collision layer, so it CRUSHES the crate the force never
  managed to move. From another Crate1 there is no such rule and the whole push
  is simply blocked.
* **Walking off a staircase back onto the ground CANCELS unless the square you
  step onto is free ground**, because the rule turns you into a Player1 standing
  on the staircase and then ``late [Player1 Stair] -> Cancel`` undoes the turn
  if you did not manage to leave it. The one exception has its own rule: a
  staircase of the crossing orientation in the way is SHOVED aside and you land
  where it was.
* **The four staircase rule blocks run in file order (UD, DU, LR, RL), and the
  "board this staircase from the Crate1 you are standing on" rules sit in the
  middle of each block.** So a player standing on a staircase that shares its
  square with a Crate1 can still board a staircase of an EARLIER type, in any
  direction -- including sideways, which is otherwise forbidden. It is the only
  cross-block interaction in the game and `_Board._p2` reproduces it by
  comparing the two staircase types' block indices.
* **A rule that seats a player evicts whatever occupied that collision layer.**
  Player1 shares a layer with Crate1 and Player2 with Crate2, so climbing onto
  a staircase that has a Crate2 on it destroys the Crate2, and walking off one
  that shares its square with a Crate1 destroys the Crate1.

Nothing in the game is random, ``noaction`` is set so ACTION5 is a measured
no-op, and nothing depends on the player's facing -- so the state is exactly
``(player square, which storey, the Crate1 set, the Crate2 set, the four
staircase sets)``.

The native model
----------------
`_Board` is that state plus the rules above. It runs at about 2 MILLION presses
a second against the interpreter's ~700 on the same boards -- both measured by
``--speed``, a factor of nearly 3000 -- which is the whole reason a native model
exists here: level 0's reachable space alone is 1275428 states, and level 9's
exact field enumerates 423131 macro states and 1516910 macro edges.

``--selfcheck`` is what makes it a statement about this game rather than about a
model of it: seeded random walks on all 20 shipped levels AND on thousands of
randomly SEATED boards (floors, crates, stacks, staircases and the player
sprinkled at random, with crates deliberately seated UNDER staircases so the
rare compositions are exercised), with the model's state compared against the
interpreter's grid after EVERY press, and a coverage counter per branch of
`_Board.step` so a branch that was never reached cannot be reported as verified.

The searches
------------
A press that moves nothing but the player is a WALK, and a plan decomposes
uniquely into maximal walks each ending in one press that changes the board. So
the searches run over MACROS -- "walk somewhere, then make one change" -- whose
cost is the walk length plus one. That is exact (every press sequence maps to
exactly one macro sequence of the same length) and it collapses the space by one
about a factor of six -- level 0's 1275428 press-states are 229472 macro-states
-- and every walk press the primitive search would re-derive for each of them is
derived once.

Two searches, in that order:

1. **The exact field.** Enumerate the whole macro space forward, recording every
   macro edge, then run one backward Dijkstra from the win over the reversed
   edges. That gives `d*` for every reachable state -- the shortest plan, a proof
   it is shortest, and the EXACT optimal-action set at every step of it, for
   free. Used wherever the closure fits under `FIELD_CAP`.
2. **Weighted A-star** on a relaxed two-storey distance for everything else. The
   heuristic is a Dijkstra from the goal over a graph where Player1 walks the
   ground, Player2 walks anything that supports it, a staircase joins the two,
   and stepping onto a bare ground square that would first have to be given a
   crate costs ``pen + base + haul * (distance from the nearest crate)``. That
   last term is what gives the search a gradient at all: the plain support
   distance is FLAT over the whole approach, because it only drops on the single
   press that drops a crate into the gap. The FIRST rung of the ladder is
   ``pen = 1, haul = 0``, the one setting that is ADMISSIBLE -- every edge is
   then one press and every real press maps onto one -- so it is both the rung
   that proves an answer shortest and the one that answers levels 3 and 7, whose
   macro closures are just too big for the field.

The nine skipped levels are skipped for two different reasons and this docstring
does not blur them. Level 13 is PROVED to have no plan inside the press budget.
For the other eight the bounded proof runs out of budget before it closes, so
all that is honest is that neither search found a plan: the heuristic cannot see
that a crate has to be ferried the long way round before it is any use, which is
exactly what those levels are about. Level 8 is the clearest case -- five Crate2s
have to be walked along a corridor and stacked to build a five-pillar column up
through the empty middle of the board -- and it is also why the two searches are
not interchangeable: level 9 is 130 presses deep and neither weighted rung finds
it, but its closure happens to fit in `FIELD_CAP`, and a field needs no
heuristic at all.

Tie labelling
-------------
Off a field level there is no distance oracle, so a step is labelled with the
WALK TIE SET: every direction that leaves the board untouched and keeps the
player on a shortest route to the same macro exit. That is sound with no
reference to optimality (any interleaving of the two axes reaches the same
square in the same number of presses and leaves an identical board), and it is
the right answer for a sokoban-shaped game, where WHICH crate you shove where is
the puzzle rather than a reordering of it. On a field level the exact sets are
used instead, and ``--selfcheck`` checks the walk sets against them.

Rendering
---------
Every object was recoloured and eight sprites redrawn -- Player1 and Player2
were the same colour AND the same sprite, Crate2 was an opaque square that hid
the Crate1 under it, Floor1 was the same black the renderer letterboxes with,
and both targets were a filled middle 3x3 exactly where the player sprite is
opaque, so WINNING looked like standing next to the goal. The header comment of
``data/puzzlescript_games/Upstairs_Downstairs.txt`` states the new pixel
budget; ``--audit`` renders every composition the game can reach as a WHOLE
frame at every board size the levels use and reports the clashes.

Verified
--------
``--plans``      11/11 cached plans replayed through the interpreter, all WIN
``--selfcheck``  0 violations: 12000 presses of model-vs-interpreter on the 20
                 shipped levels and ~22000 more on randomly seated boards, with
                 all 44 branches of `_Board.step` covered -- the rare ones by the
                 hand-written boards in `_HAND_CASES`, which exist because random
                 seating reached one of them ONCE in 34000 presses; ACTION5
                 measured a no-op on every level; every field small enough
                 re-derived and re-proved by an estimate-free macro Dijkstra;
                 its optimal SETS brute-forced by re-solving from every successor
                 of every step; and every shipped plan replayed on the model with
                 its labels
``--audit``      40 clashes over the 6 board sizes the levels use, every one of
                 them a cell where a Crate1 shares a square with a STAIRCASE --
                 a composition only the "shove a stacked crate off its pillar
                 onto a staircase" rule can create, and one ``--selfcheck``
                 measures does not occur along any shipped plan nor anywhere in
                 the closures it enumerates
``--symmetry``   220 (level, seed) pairs: the same ENGINE presses driven through
                 `screen_action` at every rotation leave the same engine grid and
                 the same un-rotated frames, 0 violations
``--recovery``   70 perturbed mid-episode boards on the 7 levels the field
                 covers: 69 re-planned to a WIN, 1 correctly answered None (the
                 falls are irreversible, so some boards really are lost), 0 wrong
end-to-end       3 seeds x 11 levels replayed FRAME-EXACT through
                 `perform_action`, 33/33 WIN, 2772 frames, 0 of the expert steps
                 unlabelled, max action index 5, and two processes at the same
                 rng seed produce a byte-identical episode

Modes
-----
``--plans``      solve every level and replay each plan through the interpreter
                 (``--all`` also re-tries the skipped ones, ``--level N`` one)
``--selfcheck``  model-vs-interpreter fuzz, field proofs, tie-set checks
``--audit``      render every reachable cell composition and diff whole frames
``--symmetry``   drive the same engine presses at all four rotations and compare
``--recovery``   re-plan from perturbed mid-episode states
``--bound``      prove (or fail to) that a skipped level has no plan inside the
                 adapter's press budget
``--speed``      model presses/s against interpreter presses/s

Plan cache: ``data/upstairs_downstairs_plans.json`` (ships derived; the field
enumerations and the A* ladder are minutes to tens of minutes per level and are
seed-independent, so generation itself is ~5 s per episode). Corpus dir
``data/training_multi_level/upstairs_downstairs`` -- NOT generated here.
"""

from __future__ import annotations

import argparse
import heapq
import random
import sys
import time
from array import array
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import (                     # noqa: E402
    PuzzleScriptAdapter, _render_frame)
from solvers.common.ps_astar import (                           # noqa: E402
    Plan, PSAStarSolver, PSExpert)

GAME_NAME = "Upstairs_Downstairs"
GAME_ID = "ps:upstairs_downstairs"

#: Largest macro closure the exact field is allowed to enumerate. Every queued
#: edge is three machine ints in an `array`, so this is a MEMORY budget: at the
#: measured ~9 edges per state it is about 60 MB of edge arrays plus the state
#: dict, which is the part that actually costs (a state is a 5-tuple holding two
#: big ints, ~250 bytes).
FIELD_CAP = 800_000

#: `(weight, pen, haul)` rungs for the weighted-A* fallback, in the order they
#: are tried. The FIRST rung is the only one that proves anything: at
#: ``weight == 1`` with ``pen == 1`` and ``haul == 0`` the relaxed distance is
#: ADMISSIBLE (see `_Heur`), so a plan it returns is shortest. Every later rung
#: trades that away for a heuristic that can actually steer -- a crate-gap
#: penalty above 1 is not a lower bound, because filling a gap and stepping into
#: it can cost as little as two presses.
ASTAR_LADDER = (
    (1, 1, 0), (2, 8, 3), (3, 10, 4), (2, 12, 5),
    (5, 10, 4), (3, 16, 7), (8, 12, 5), (12, 20, 8),
)

#: Node budget for one A* rung. Counts macro STATES kept, not expansions, which
#: is what actually costs: every kept state carries its own board plus the walk
#: that reached it.
ASTAR_CAP = 600_000

#: Seconds one A* rung may run before it gives up and the next rung starts. The
#: admissible first rung gets its own, much longer budget: it is the only rung
#: that can prove an answer shortest, and on a level whose macro closure just
#: misses `FIELD_CAP` it is also the only rung that finds one at all.
ASTAR_DEADLINE = 300.0
EXACT_DEADLINE = 1200.0

#: The adapter cuts a level off at 200 presses, so a plan longer than this can
#: never be recorded even if the search finds one.
PRESS_BUDGET = 199


# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DIRS = ("up", "down", "left", "right")
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))
_OPP = (DOWN, UP, RIGHT, LEFT)

#: Staircase object names, IN THE ORDER THEIR RULE BLOCKS APPEAR IN THE .txt.
#: The order is load-bearing: see `_Board._p2`'s cross-block boarding.
STAIRS = ("udstair", "dustair", "lrstair", "rlstair")

#: The direction you walk to CLIMB each staircase (and, standing on it, the
#: direction that steps off onto the upper storey). Its opposite is the one that
#: walks back down onto the ground.
_CLIMB = (DOWN, UP, RIGHT, LEFT)

#: True for the two staircases whose climb axis is vertical. Only a staircase of
#: the CROSSING axis is shoved aside by someone walking off a staircase.
_VERTICAL = (True, True, False, False)


def _bits(mask: int) -> list[int]:
    """The set cell indices of a bitmask, ascending."""
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """The static part of a level, plus the transition function.

    A STATE is ``(player, storey, crate1, crate2, stairs)`` where ``player`` is a
    cell index, ``storey`` is 1 (on the ground, a Player1) or 2 (upstairs, a
    Player2), ``crate1`` / ``crate2`` are cell bitmasks and ``stairs`` is a
    4-tuple of cell bitmasks indexed by `STAIRS`.

    Floors and targets never change, so they live here rather than in the state.
    """

    __slots__ = ("h", "w", "n", "floor2", "target1", "target2", "nei", "tag")

    def __init__(self, h: int, w: int, floor2: int, target1: int, target2: int):
        self.h, self.w = h, w
        self.n = h * w
        self.floor2 = floor2
        self.target1 = target1
        self.target2 = target2
        self.tag = ""                      # branch of `step` last taken
        self.nei = []
        for d in range(4):
            dr, dc = _DELTA[d]
            row = []
            for i in range(self.n):
                r, c = divmod(i, w)
                r2, c2 = r + dr, c + dc
                row.append(r2 * w + c2 if 0 <= r2 < h and 0 <= c2 < w else -1)
            self.nei.append(row)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def stair_at(st, cell: int) -> int:
        """The `STAIRS` index of the staircase on ``cell``, or -1."""
        b = 1 << cell
        for i in range(4):
            if st[i] & b:
                return i
        return -1

    def won(self, s) -> bool:
        p, storey, _c1, _c2, _st = s
        b = 1 << p
        return bool((self.target1 & b) if storey == 1 else (self.target2 & b))

    def _t(self, tag: str, v):
        self.tag = tag
        return v

    # -- the transition ---------------------------------------------------
    def step(self, s, d: int):
        """The board after pressing direction ``d``. Returns ``s`` itself for a
        press with no effect, so callers can test identity."""
        t = self.nei[d][s[0]]
        if t < 0:
            return self._t("edge", s)
        return self._p1(s, d, t) if s[1] == 1 else self._p2(s, d, t)

    # -- Player1: the ground floor ---------------------------------------
    def _p1(self, s, d, t):
        p, _storey, c1, c2, st = s
        bt = 1 << t
        f2 = self.floor2
        sty = self.stair_at(st, t)
        if sty >= 0:
            if _CLIMB[sty] == d:
                # A direct placement of Player2, so a Crate2 on the staircase is
                # crushed by the player landing on its collision layer.
                return self._t("p1.climb", (t, 2, c1, c2 & ~bt, st))
            u = self.nei[d][t]
            if u < 0:
                return self._t("p1.pushstair.edge", s)
            bu = 1 << u
            if self.stair_at(st, u) >= 0 or (c1 & bu) or (f2 & bu):
                return self._t("p1.pushstair.blocked", s)
            st2 = list(st)
            st2[sty] ^= bt | bu
            if c1 & bt:
                if c2 & bt:
                    # "you may not shove a stacked crate" strips the force off
                    # the PLAYER and off the crate, but not off the staircase,
                    # so the stairs slide out from under the stack alone
                    return self._t("p1.pushstair.stackleft",
                                   (p, 1, c1, c2, tuple(st2)))
                return self._t("p1.pushstair.withcrate",
                               (t, 1, (c1 ^ bt) | bu, c2, tuple(st2)))
            return self._t("p1.pushstair", (t, 1, c1, c2, tuple(st2)))
        if c1 & bt:
            if c2 & bt:
                return self._t("p1.stack.blocked", s)
            u = self.nei[d][t]
            if u < 0:
                return self._t("p1.pushcrate.edge", s)
            bu = 1 << u
            if self.stair_at(st, u) >= 0 or (f2 & bu) or (c1 & bu):
                return self._t("p1.pushcrate.blocked", s)
            return self._t("p1.pushcrate", (t, 1, (c1 ^ bt) | bu, c2, st))
        if f2 & bt:
            return self._t("p1.ceiling", s)
        return self._t("p1.walk", (t, 1, c1, c2, st))

    # -- Player2: the storey above ---------------------------------------
    def _p2(self, s, d, t):
        p, _storey, c1, c2, st = s
        psty = self.stair_at(st, p)
        if psty < 0:
            return self._off_stair(s, d, t)
        # The four staircase rule blocks run in file order and each one's "board
        # this staircase from the Crate1 you are standing on" rule comes before
        # the LATER blocks' "you may not move that way off this staircase" ones,
        # so a player on a staircase that shares its square with a Crate1 can
        # still board a staircase of an earlier (or equal) type.
        if c1 & (1 << p):
            tsty = self.stair_at(st, t)
            if 0 <= tsty <= psty and d == _OPP[_CLIMB[tsty]]:
                return self._t("p2.stair.crossboard",
                               (t, 2, c1, c2 & ~(1 << t), st))
        if d == _CLIMB[psty]:
            return self._upper(s, d, t, True)
        if d != _OPP[_CLIMB[psty]]:
            return self._t("p2.stair.sideways", s)
        return self._descend(s, d, t)

    def _off_stair(self, s, d, t):
        """Player2 standing on Floor2 or on top of a Crate1."""
        _p, _storey, c1, c2, st = s
        bt = 1 << t
        tsty = self.stair_at(st, t)
        if tsty >= 0:
            if d == _OPP[_CLIMB[tsty]]:
                # boarding is a direct placement, so it crushes a Crate2 sitting
                # on the staircase
                return self._t("p2.up.board", (t, 2, c1, c2 & ~bt, st))
            return self._t("p2.up.stairblocked", s)
        if c2 & bt:
            return self._shove(s, d, t, False)
        if (c1 & bt) or (self.floor2 & bt):
            return self._t("p2.up.walk", (t, 2, c1, c2, st))
        return self._t("p2.up.nosupport", s)

    def _upper(self, s, d, t, from_stair):
        """Player2 stepping off the top of its staircase onto the upper storey."""
        _p, _storey, c1, c2, st = s
        bt = 1 << t
        if c2 & bt:
            return self._shove(s, d, t, from_stair)
        if c1 & bt:
            return self._t("p2.on.ontocrate", (t, 2, c1, c2, st))
        if self.floor2 & bt:
            return self._t("p2.on.ontofloor2", (t, 2, c1, c2, st))
        return self._t("p2.on.nosupport", s)

    def _shove(self, s, d, t, from_stair):
        """Player2 shoving the Crate2 that sits on ``t``."""
        p, _storey, c1, c2, st = s
        pre = "p2.on" if from_stair else "p2.up"
        bt = 1 << t
        f2 = self.floor2
        stacked = bool(c1 & bt)
        u = self.nei[d][t]
        if u < 0:
            # The crate cannot leave the grid so its force never resolves -- but
            # the rule that seats the player ON a Crate1 is a direct placement
            # onto the crate's own collision layer, so it CRUSHES it. That rule
            # only fires from Floor2 or from a staircase; from another Crate1
            # there is none and the whole chain is blocked.
            if stacked and (from_stair or (f2 & (1 << p))):
                return self._t(pre + ".shove.crush", (t, 2, c1, c2 ^ bt, st))
            return self._t(pre + ".shove.offmap", s)
        bu = 1 << u
        if c2 & bu:
            return self._t(pre + ".shove.cancel", s)
        if stacked and from_stair:
            # the on-staircase rules place the crate DIRECTLY and test the FALL
            # first, so from there it lands even on a staircase square
            if not ((c1 & bu) or (f2 & bu)):
                return self._t(pre + ".shove.stackfall",
                               (t, 2, c1 | bu, c2 ^ bt, st))
            return self._t(pre + ".shove.stackmove",
                           (t, 2, c1, (c2 ^ bt) | bu, st))
        if (c1 & bu) or (f2 & bu):
            return self._t(pre + ".shove.move", (t, 2, c1, (c2 ^ bt) | bu, st))
        if self.stair_at(st, u) >= 0:
            return self._t(pre + ".shove.stair", s)
        return self._t(pre + ".shove.fall", (t, 2, c1 | bu, c2 ^ bt, st))

    def _descend(self, s, d, t):
        """Player2 walking off the low end of its staircase, back to the ground."""
        p, _storey, c1, c2, st = s
        bp, bt = 1 << p, 1 << t
        f2 = self.floor2
        psty = self.stair_at(st, p)
        tsty = self.stair_at(st, t)
        if tsty >= 0:
            if _VERTICAL[psty] != _VERTICAL[tsty]:
                u = self.nei[d][t]
                if u >= 0:
                    bu = 1 << u
                    if not (self.stair_at(st, u) >= 0 or (c1 & bu) or (f2 & bu)):
                        st2 = list(st)
                        st2[tsty] ^= bt | bu
                        # Player1 is placed AT t, crushing a Crate1 there; the
                        # one under the player's own staircase is left alone
                        return self._t("p2.descend.shovestair",
                                       (t, 1, c1 & ~bt, c2, tuple(st2)))
            return self._t("p2.descend.ontostair", s)
        # Player1 is placed on the staircase square first, crushing a Crate1
        # there, and the turn is cancelled outright if it cannot then leave.
        c1n = c1 & ~bp
        if f2 & bt:
            return self._t("p2.descend.ceiling", s)
        if c1 & bt:
            if c2 & bt:
                return self._t("p2.descend.stack", s)
            u = self.nei[d][t]
            if u < 0:
                return self._t("p2.descend.pushedge", s)
            bu = 1 << u
            if self.stair_at(st, u) >= 0 or (f2 & bu) or (c1 & bu):
                return self._t("p2.descend.pushblocked", s)
            return self._t("p2.descend.push", (t, 1, (c1n ^ bt) | bu, c2, st))
        return self._t("p2.descend", (t, 1, c1n, c2, st))

    # -- macros -----------------------------------------------------------
    def quiet(self, s):
        """BFS over the presses that leave the crates and staircases untouched.

        Returns ``(dist, par, exits)``: ``dist`` maps ``(cell, storey)`` to the
        walk length from ``s``, ``par`` is its predecessor map, and ``exits`` is
        every ``(cost, from_key, direction, new_state)`` a press that DOES change
        the board leads to. A press sequence decomposes uniquely into walks each
        ending in one such press, so a shortest path over these macros is a
        shortest press sequence."""
        c1, c2, st = s[2], s[3], s[4]
        k0 = (s[0], s[1])
        dist = {k0: 0}
        par = {k0: None}
        q = deque([s])
        exits = []
        while q:
            cur = q.popleft()
            g = dist[(cur[0], cur[1])]
            for d in range(4):
                nxt = self.step(cur, d)
                if nxt[2] == c1 and nxt[3] == c2 and nxt[4] == st:
                    k = (nxt[0], nxt[1])
                    if k in dist:
                        continue
                    dist[k] = g + 1
                    par[k] = ((cur[0], cur[1]), d)
                    q.append(nxt)
                else:
                    exits.append((g + 1, (cur[0], cur[1]), d, nxt))
        return dist, par, exits

    def goal_walk(self, s, dist) -> int | None:
        """The shortest walk from ``s`` onto the goal square, or None."""
        best = None
        for (cell, storey), dd in dist.items():
            if self.won((cell, storey, s[2], s[3], s[4])):
                if best is None or dd < best:
                    best = dd
        return best


def _walk_path(par, k) -> list[int]:
    out = []
    while par[k] is not None:
        pk, d = par[k]
        out.append(d)
        k = pk
    out.reverse()
    return out


# ---------------------------------------------------------------------------
# The relaxed two-storey distance (the A* heuristic)
# ---------------------------------------------------------------------------

INF = 1 << 30


def _goal_of(board: _Board) -> tuple[int, int]:
    """``(cell, storey)`` of the level's single target."""
    if board.target2:
        return (board.target2 & -board.target2).bit_length() - 1, 2
    return (board.target1 & -board.target1).bit_length() - 1, 1


class _Heur:
    """Distance to the goal in a relaxed copy of the game, cached by the parts of
    the state it depends on.

    Node ``i`` is Player1 on cell ``i``, node ``n + i`` is Player2 on it. Player1
    walks the ground (a Floor2 tile is a wall; a crate is free, because shoving
    one moves the player too); Player2 walks anything that supports it; a
    staircase square joins the two storeys in both directions. Stepping onto a
    bare ground square as Player2 -- a square that would first have to be given a
    crate -- costs ``pen + base + haul * (grid distance from the nearest crate)``.

    ``pen == 1`` with ``haul == 0`` and ``strict == False`` is the ADMISSIBLE
    setting, and the only one: every edge is then one press, and every real press
    that moves the player maps onto one of them (Player1 only ever walks the
    ground, Player2 only ever walks a square that supports it AT THE TIME, and a
    storey change is a move onto a ground square), so the relaxed path is never
    longer than the real one. A gap penalty above 1 breaks that -- the presses
    that deliver a crate into the gap MOVE the player too, so they are already
    counted as edges elsewhere in the path and cannot be charged twice.
    """

    def __init__(self, board: _Board, pen: int = 6, haul: int = 0,
                 base: int = 4, strict: bool = True, source: str = "any"):
        self.b = board
        self.pen, self.haul, self.base = pen, haul, base
        self.strict = strict
        #: which crates the haul term measures from. "any" counts a Crate1 as
        #: raw material too; "c2" counts only the boxes UPSTAIRS, which is the
        #: honest reading whenever the gaps to be filled are out of Player1's
        #: reach -- a Crate1 in a room the ground player cannot enter is not
        #: material, it is scenery, and a heuristic that counts it is flat over
        #: exactly the stretch that does the work.
        self.source = source
        self.g, self.gs = _goal_of(board)
        self.cache: dict = {}
        self.nbrs = [[board.nei[d][i] for d in range(4) if board.nei[d][i] >= 0]
                     for i in range(board.n)]

    def _crate_dist(self, c1: int, c2: int) -> list[int]:
        cd = [INF] * self.b.n
        q = deque()
        for i in _bits(c1 | c2):
            cd[i] = 0
            q.append(i)
        while q:
            x = q.popleft()
            for y in self.nbrs[x]:
                if cd[y] == INF:
                    cd[y] = cd[x] + 1
                    q.append(y)
        return cd

    def table(self, c1: int, c2: int, st) -> list[int]:
        key = (c1, c2, st) if self.haul else (c1, st)  # c2 only matters via haul
        got = self.cache.get(key)
        if got is not None:
            return got
        b = self.b
        n = b.n
        pen, base, haul, strict = self.pen, self.base, self.haul, self.strict
        sm = st[0] | st[1] | st[2] | st[3]
        sup2 = b.floor2 | c1 | sm
        cd = (self._crate_dist(0 if self.source == "c2" else c1, c2)
              if haul else None)
        rev = [[] for _ in range(2 * n)]
        for x in range(n):
            bx = 1 << x
            ground = not (b.floor2 & bx)
            for y in self.nbrs[x]:
                by = 1 << y
                if ground and not (strict and (sm & bx)):
                    if not (b.floor2 & by):
                        if (sm & by) or not strict:
                            rev[n + y].append((x, 1))        # climb
                        if not (sm & by):
                            rev[y].append((x, 1))            # walk / push
                if (not strict) or (sm & bx):
                    if not (b.floor2 & by) and not (strict and (sm & by)):
                        rev[y].append((n + x, 1))            # walk back down
                if sup2 & by:
                    w = 1
                elif cd is None:
                    w = pen
                else:
                    w = pen + base + haul * min(cd[y], 40)
                rev[n + y].append((n + x, w))
        dist = [INF] * (2 * n)
        src = (self.gs - 1) * n + self.g
        dist[src] = 0
        pq = [(0, src)]
        while pq:
            dcur, u = heapq.heappop(pq)
            if dcur > dist[u]:
                continue
            for v, w in rev[u]:
                nd = dcur + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        self.cache[key] = dist
        return dist

    def __call__(self, s) -> int:
        return self.table(s[2], s[3], s[4])[(s[1] - 1) * self.b.n + s[0]]


# ---------------------------------------------------------------------------
# The exact field: forward closure + one backward Dijkstra
# ---------------------------------------------------------------------------

class _Field:
    """Exact presses-to-win for every state of a level whose macro closure fits.

    Built in two sweeps. The forward one enumerates every macro state reachable
    from the level start, recording each macro edge into flat `array`s (a macro
    is irreversible often enough -- a fallen crate, a crushed one, a shoved
    staircase -- that there is no analytic predecessor relation to exploit, so
    the edges have to be kept). The backward one is a Dijkstra from a virtual
    GOAL node whose incoming edges are each state's shortest walk onto the
    target square.

    `dist` then answers `d*` for any state, macro or mid-walk, which is a
    shortest plan, the proof that it is shortest, and the exact optimal-action
    set at every step.
    """

    __slots__ = ("board", "ids", "dist", "n_states", "n_edges")

    def __init__(self, board: _Board, ids: dict, dist: list,
                 n_edges: int):
        self.board = board
        self.ids = ids
        self.dist = dist
        self.n_states = len(ids)
        self.n_edges = n_edges

    @classmethod
    def build(cls, board: _Board, s0, cap: int = FIELD_CAP,
              deadline: float | None = None) -> "_Field | None":
        ids = {s0: 0}
        order = [s0]
        src = array("i")
        dst = array("i")
        wgt = array("i")
        goal = array("i")            # per state: walk length onto the goal, or -1
        i = 0
        while i < len(order):
            s = order[i]
            i += 1
            dist, _par, exits = board.quiet(s)
            gw = board.goal_walk(s, dist)
            goal.append(-1 if gw is None else gw)
            for cost, _kf, _d, ns in exits:
                j = ids.get(ns)
                if j is None:
                    if len(ids) >= cap:
                        return None
                    j = len(ids)
                    ids[ns] = j
                    order.append(ns)
                src.append(i - 1)
                dst.append(j)
                wgt.append(cost)
            if deadline is not None and time.time() > deadline:
                return None
        # backward Dijkstra from the virtual goal over the REVERSED edges
        n = len(order)
        head = [-1] * n
        nxt = array("i", [0]) * len(src) if src else array("i")
        for e in range(len(src)):
            v = dst[e]
            nxt[e] = head[v]
            head[v] = e
        dist_to = [INF] * n
        pq = []
        for u in range(n):
            if goal[u] >= 0:
                dist_to[u] = goal[u]
                pq.append((goal[u], u))
        heapq.heapify(pq)
        while pq:
            dcur, v = heapq.heappop(pq)
            if dcur > dist_to[v]:
                continue
            e = head[v]
            while e != -1:
                u = src[e]
                nd = dcur + wgt[e]
                if nd < dist_to[u]:
                    dist_to[u] = nd
                    heapq.heappush(pq, (nd, u))
                e = nxt[e]
        return cls(board, ids, dist_to, len(src))

    # -- queries ----------------------------------------------------------
    def dstar(self, s) -> int | None:
        """Shortest presses from ``s`` to a win, exact. ``s`` may be mid-walk."""
        board = self.board
        idx = self.ids.get(s)
        if idx is not None and self.dist[idx] < INF:
            return self.dist[idx]
        dist, _par, exits = board.quiet(s)
        best = board.goal_walk(s, dist)
        if best is None:
            best = INF
        for cost, _kf, _d, ns in exits:
            j = self.ids.get(ns)
            if j is None:
                continue
            v = self.dist[j]
            if v < INF and cost + v < best:
                best = cost + v
        return None if best >= INF else best

    def plan(self, s0) -> "Plan | None":
        """The shortest plan from ``s0``, with the EXACT optimal set per step."""
        d0 = self.dstar(s0)
        if d0 is None:
            return None
        presses, optsets = [], []
        s = s0
        remaining = d0
        board = self.board
        while remaining > 0:
            best = []
            take = None
            for d in range(4):
                ns = board.step(s, d)
                if ns == s:
                    continue
                sub = self.dstar(ns)
                if sub is not None and sub + 1 == remaining:
                    best.append(DIRS[d])
                    if take is None:
                        take = d
            if take is None:
                return None
            optsets.append(best)
            presses.append(DIRS[take])
            s = board.step(s, take)
            remaining -= 1
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# Weighted A* over macros, with walk tie sets
# ---------------------------------------------------------------------------

def _astar(board: _Board, s0, heur: _Heur, weight: int = 1,
           cap: int = ASTAR_CAP, deadline: float | None = None):
    """A* whose successors are macros. Returns a flat press list, or None.

    The win is pushed onto the SAME queue as a virtual node whose cost is the
    walk onto the target square, and the search runs until that entry is popped.
    Returning the first win the search GENERATES would not be shortest even at
    weight 1: macros cost different amounts, so a node popped at ``g = 40`` can
    reach the goal by an 11-press walk (51) while one popped at ``g = 45``
    reaches it in 3 (48). With ``weight == 1`` and an admissible heuristic --
    which is what the ladder's first rung is -- popping the virtual node IS the
    proof that the answer is shortest.
    """
    best = {s0: 0}
    par: dict = {}
    pq = [(weight * heur(s0), 0, 0, s0, None)]
    cnt = 0

    def rebuild(s, tail):
        presses: list[int] = []
        chain = []
        cur = s
        while cur in par:
            pk, pd, pw = par[cur]
            chain.append((pw, pd))
            cur = pk
        chain.reverse()
        for pw, pd in chain:
            presses.extend(pw)
            presses.append(pd)
        presses.extend(tail)
        return [DIRS[d] for d in presses]

    while pq:
        _f, g, _c, s, tail = heapq.heappop(pq)
        if tail is not None:                       # the virtual goal entry
            return rebuild(s, tail)
        if g > best.get(s, INF):
            continue
        dist, pr, exits = board.quiet(s)
        gw = board.goal_walk(s, dist)
        if gw is not None:
            key = next(k for k, dd in dist.items()
                       if dd == gw and board.won((k[0], k[1], s[2], s[3], s[4])))
            cnt += 1
            heapq.heappush(pq, (g + gw, g + gw, cnt, s, _walk_path(pr, key)))
        for cost, kf, d, ns in exits:
            ng = g + cost
            if ng < best.get(ns, INF):
                best[ns] = ng
                par[ns] = (s, d, _walk_path(pr, kf))
                cnt += 1
                heapq.heappush(pq, (ng + weight * heur(ns), ng, cnt, ns, None))
        if len(best) > cap or (deadline is not None and time.time() > deadline):
            return None
    return None


def _quiet_graph(board: _Board, s):
    """The WALK graph around ``s``: the presses that leave the crates and the
    staircases untouched, as ``(adj, states)`` over ``(cell, storey)`` nodes.

    It is not provably undirected -- the cross-block staircase boarding has no
    guaranteed inverse -- so the tie labelling measures distances TO the walk's
    destination over the reversed graph rather than assuming symmetry.
    """
    c1, c2, st = s[2], s[3], s[4]
    k0 = (s[0], s[1])
    adj: dict = {k0: []}
    states = {k0: s}
    q = deque([s])
    while q:
        cur = q.popleft()
        kc = (cur[0], cur[1])
        for d in range(4):
            nxt = board.step(cur, d)
            if nxt[2] != c1 or nxt[3] != c2 or nxt[4] != st:
                continue
            k = (nxt[0], nxt[1])
            if k == kc:
                continue
            adj[kc].append((k, d))
            if k not in states:
                states[k] = nxt
                adj.setdefault(k, [])
                q.append(nxt)
    return adj, states


def _dist_to(adj: dict, goal) -> dict:
    """BFS distances TO ``goal`` over the reversed walk graph."""
    rev: dict = {k: [] for k in adj}
    for u, outs in adj.items():
        for v, _d in outs:
            rev[v].append(u)
    dist = {goal: 0}
    q = deque([goal])
    while q:
        v = q.popleft()
        for u in rev[v]:
            if u not in dist:
                dist[u] = dist[v] + 1
                q.append(u)
    return dist


def _macro_split(board: _Board, s0, presses: list[str]):
    """A flat plan as ``[(state at macro start, walk dirs, changing dir|None)]``."""
    idx = {d: i for i, d in enumerate(DIRS)}
    out = []
    s = start = s0
    walk: list[int] = []
    for name in presses:
        d = idx[name]
        ns = board.step(s, d)
        if ns[2] == s[2] and ns[3] == s[3] and ns[4] == s[4]:
            walk.append(d)
        else:
            out.append((start, walk, d))
            walk = []
            start = ns
        s = ns
    if walk:
        out.append((start, walk, None))
    return out


def _bounded(board: _Board, s0, limit: int = PRESS_BUDGET,
             cap: int = 3_000_000, deadline: float | None = None):
    """Is there ANY plan of at most ``limit`` presses?

    A* over macros on the ADMISSIBLE relaxed distance, run until the frontier's
    ``f = g + h`` passes ``limit``. ``f`` is a lower bound on the total length of
    every completion below that node, so once the smallest one exceeds the limit
    no plan that short can exist -- which turns "the search did not find one"
    into "there is none", the only honest thing to say about a level this
    generator skips.

    Returns ``(verdict, states, seconds)`` with verdict ``d*`` (an int),
    ``"none"`` (proved) or ``"inconclusive"`` (the budget ran out first).
    """
    heur = _Heur(board, pen=1, haul=0, strict=False)
    t0 = time.time()
    best = {s0: 0}
    pq = [(heur(s0), 0, 0, s0)]
    cnt = 0
    while pq:
        f, g, _c, s = heapq.heappop(pq)
        if f > limit:
            return "none", len(best), time.time() - t0
        if g > best.get(s, INF):
            continue
        dist, _pr, exits = board.quiet(s)
        gw = board.goal_walk(s, dist)
        if gw is not None and g + gw <= limit:
            return g + gw, len(best), time.time() - t0
        for cost, _kf, _d, ns in exits:
            ng = g + cost
            if ng < best.get(ns, INF):
                nf = ng + heur(ns)
                if nf > limit:
                    continue
                best[ns] = ng
                cnt += 1
                heapq.heappush(pq, (nf, ng, cnt, ns))
        if len(best) > cap or (deadline is not None and time.time() > deadline):
            return "inconclusive", len(best), time.time() - t0
    return "none", len(best), time.time() - t0


def walk_optsets(board: _Board, s0, presses: list[str]) -> list[list[str]]:
    """The WALK tie set for every step of a flat plan.

    A press that leaves the crates and the staircases untouched is a walk, and
    every interleaving of a shortest walk to the same macro exit reaches the
    same square in the same number of presses and leaves an identical board --
    so each such direction is equally optimal. A press that CHANGES the board is
    labelled with itself: which crate you shove where is the puzzle, not a
    reordering of it. Sound with no reference to optimality, and never empty --
    the step actually taken is always in its own set.
    """
    out: list[list[str]] = []
    for start, walk, change in _macro_split(board, s0, presses):
        if walk:
            dest = start
            for d in walk:
                dest = board.step(dest, d)
            adj, _states = _quiet_graph(board, start)
            to_goal = _dist_to(adj, (dest[0], dest[1]))
            cur = start
            for d in walk:
                kc = (cur[0], cur[1])
                need = to_goal.get(kc)
                tie = []
                for e in range(4):
                    ns = board.step(cur, e)
                    if (ns[2] != cur[2] or ns[3] != cur[3] or ns[4] != cur[4]):
                        continue
                    k = (ns[0], ns[1])
                    if k == kc:
                        continue
                    rem = to_goal.get(k)
                    if need is not None and rem is not None and rem + 1 == need:
                        tie.append(DIRS[e])
                out.append(tie if DIRS[d] in tie else [DIRS[d]])
                cur = board.step(cur, d)
        if change is not None:
            out.append([DIRS[change]])
    return out


# ---------------------------------------------------------------------------
# Reading the engine
# ---------------------------------------------------------------------------

def read_board(eng, g) -> _Board:
    idx = g.obj_name_to_idx
    h, w = eng.height, eng.width
    f2 = t1 = t2 = 0
    for r in range(h):
        for c in range(w):
            cell = eng.grid[r][c]
            i = r * w + c
            if idx["floor2"] in cell:
                f2 |= 1 << i
            if idx["target1"] in cell:
                t1 |= 1 << i
            if idx["target2"] in cell:
                t2 |= 1 << i
    return _Board(h, w, f2, t1, t2)


def read_state(eng, g):
    idx = g.obj_name_to_idx
    w = eng.width
    c1 = c2 = 0
    st = [0, 0, 0, 0]
    p = storey = None
    for r in range(eng.height):
        for c in range(w):
            cell = eng.grid[r][c]
            i = r * w + c
            if idx["crate1"] in cell:
                c1 |= 1 << i
            if idx["crate2"] in cell:
                c2 |= 1 << i
            for k, name in enumerate(STAIRS):
                if idx[name] in cell:
                    st[k] |= 1 << i
            if idx["player1"] in cell:
                p, storey = i, 1
            elif idx["player2"] in cell:
                p, storey = i, 2
    return None if p is None else (p, storey, c1, c2, tuple(st))


def seat_state(eng, g, board: _Board, s) -> None:
    """Write a model state back into the engine grid (test harness only)."""
    idx = g.obj_name_to_idx
    p, storey, c1, c2, st = s
    eng.load_level([[set() for _ in range(board.w)] for _ in range(board.h)])
    for r in range(board.h):
        for c in range(board.w):
            i = r * board.w + c
            eng.grid[r][c] = set()
            eng._cell_add(r, c, idx["background"])
            eng._cell_add(r, c, idx["floor2"] if (board.floor2 >> i & 1)
                          else idx["floor1"])
            if board.target1 >> i & 1:
                eng._cell_add(r, c, idx["target1"])
            if board.target2 >> i & 1:
                eng._cell_add(r, c, idx["target2"])
            if c1 >> i & 1:
                eng._cell_add(r, c, idx["crate1"])
            if c2 >> i & 1:
                eng._cell_add(r, c, idx["crate2"])
            for k, name in enumerate(STAIRS):
                if st[k] >> i & 1:
                    eng._cell_add(r, c, idx[name])
            if i == p:
                eng._cell_add(r, c, idx["player1" if storey == 1 else "player2"])
    eng._build_position_index()


def show(board: _Board, s) -> str:
    """One-line-per-row ASCII dump of a state (diagnostics only)."""
    p, storey, c1, c2, st = s
    rows = []
    for r in range(board.h):
        cells = []
        for c in range(board.w):
            i = r * board.w + c
            t = "#" if board.floor2 >> i & 1 else "."
            if board.target1 >> i & 1:
                t += "I"
            if board.target2 >> i & 1:
                t += "J"
            if c1 >> i & 1:
                t += "1"
            if c2 >> i & 1:
                t += "2"
            for k, ch in enumerate("DURL"):
                if st[k] >> i & 1:
                    t += ch
            if i == p:
                t += "P" if storey == 1 else "Q"
            cells.append(f"{t:<4}")
        rows.append(" ".join(cells))
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class UpstairsDownstairsExpert(PSExpert):
    """Plans on `_Board`; the interpreter only ever certifies the answer.

    `_search` builds the model off the engine's current grid, tries the exact
    field first and weighted A* after it, and returns a `Plan` carrying the
    per-step optimal sets (exact off the field, walk ties off A*).
    """

    scope_by_level = True
    directions = list(DIRS)
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "upstairs_downstairs_plans.json")

    def setup(self) -> None:
        self._boards: dict[int, _Board] = {}
        self._fields: dict[int, "_Field | None"] = {}
        self.notes: dict[int, str] = {}
        self.field_cap = FIELD_CAP
        self.ladder = ASTAR_LADDER
        self.astar_cap = ASTAR_CAP
        self.astar_deadline = ASTAR_DEADLINE
        self.exact_deadline = EXACT_DEADLINE

    # `_key` narrows to the DYNAMIC objects, which is why `scope_by_level` is on.
    def _key(self, eng) -> frozenset:
        g = self.g
        idx = g.obj_name_to_idx
        dyn = {idx[n] for n in
               ("player1", "player2", "crate1", "crate2") + STAIRS}
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in cell
            if o in dyn
        )

    def heuristic(self, eng) -> int:                    # pragma: no cover
        raise AssertionError("planning runs on _Board, never on the engine")

    def board_for(self, eng, level: int | None) -> _Board:
        if level is None:
            return read_board(eng, self.g)
        got = self._boards.get(level)
        if got is None:
            got = self._boards[level] = read_board(eng, self.g)
        return got

    def plan(self, eng, level: int | None = None):
        self._level = level
        return super().plan(eng, level)

    def _search(self, eng):
        level = getattr(self, "_level", None)
        board = self.board_for(eng, level)
        s0 = read_state(eng, self.g)
        if s0 is None:
            return None
        if board.won(s0):
            return Plan([], [])
        return self.solve(board, s0, level)

    def solve(self, board: _Board, s0, level: int | None = None):
        """The search ladder: the exact field, then weighted A*."""
        # The field is cached per LEVEL and built from whatever state was first
        # planned at, which is always the level start (recovery is a RESET). It
        # answers any state reachable from that root, so a re-plan from a
        # perturbed board reuses it; if it cannot answer -- a root it was not
        # built from -- the ladder below still runs rather than the level being
        # declared lost.
        if level is not None and level in self._fields:
            field = self._fields[level]
        else:
            field = _Field.build(board, s0, cap=self.field_cap)
            if level is not None:
                self._fields[level] = field
        if field is not None:
            plan = field.plan(s0)
            if plan is None:
                if field.ids.get(s0) == 0:      # the root: the field IS the answer
                    if level is not None:
                        self.notes[level] = "exact field: the win is unreachable"
                    return None
                field = None                    # fall through to the ladder
        if field is not None:
            if len(plan) > PRESS_BUDGET:
                # the field is exact, so no search can do better: the level is
                # winnable but not inside the adapter's 200-press cut-off
                if level is not None:
                    self.notes[level] = (
                        f"exact field: shortest plan is {len(plan)} presses, "
                        f"over the {PRESS_BUDGET}-press budget")
                return None
            if level is not None:
                self.notes[level] = (
                    f"exact field ({field.n_states} macro states, "
                    f"{field.n_edges} edges) -- provably shortest")
            return plan
        for weight, pen, haul in self.ladder:
            admissible = (weight == 1 and pen == 1 and haul == 0)
            heur = _Heur(board, pen=pen, haul=haul,
                         strict=bool(haul), base=4)
            budget = self.exact_deadline if admissible else self.astar_deadline
            found = _astar(board, s0, heur, weight=weight, cap=self.astar_cap,
                           deadline=time.time() + budget)
            if found is None or len(found) > PRESS_BUDGET:
                continue
            if level is not None:
                proved = " -- provably shortest" if admissible else ""
                self.notes[level] = (
                    f"A* w={weight} pen={pen} haul={haul}{proved}")
            return Plan(found, walk_optsets(board, s0, found))
        if level is not None:
            self.notes[level] = "no plan"
        return None


# ---------------------------------------------------------------------------
# The BaseSolver harness
# ---------------------------------------------------------------------------

class UpstairsDownstairsSolver(PSAStarSolver):
    game_id = GAME_ID
    game_name = GAME_NAME
    expert_cls = UpstairsDownstairsExpert
    game_module_id = GAME_ID
    max_steps = 220

    #: Levels the searches cannot win inside `PRESS_BUDGET` presses and the node
    #: budgets above. Skipped up front so discovery does not burn a full ladder
    #: on them at every startup. ``--plans --all`` re-tries them and ``--bound``
    #: is what says whether a plan that short exists at all.
    skip_levels = frozenset({8, 10, 11, 12, 13, 14, 15, 18, 19})


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    game = PuzzleScriptAdapter(GAME_NAME, seed=seed)
    return game, UpstairsDownstairsExpert(game)


def _plans(levels=None, verbose: bool = True) -> int:
    """Solve each level and replay the plan through the interpreter."""
    game, expert = _new()
    eng = game._engine
    todo = levels if levels is not None else [
        lvl for lvl in range(game.n_levels)
        if lvl not in UpstairsDownstairsSolver.skip_levels]
    total = won = 0
    for level in todo:
        t0 = time.time()
        game.set_level(level)
        plan = expert.plan(eng, level)
        note = expert.notes.get(level, "cached")
        if plan is None:
            if verbose:
                print(f"level {level:2d}: NO PLAN                ({note}, "
                      f"{time.time() - t0:.1f}s)")
            continue
        # replay through the interpreter
        game.set_level(level)
        for d in plan:
            eng.step(d)
        ok = eng.check_win()
        total += 1
        won += bool(ok)
        if verbose:
            print(f"level {level:2d}: {len(plan):4d} presses  "
                  f"{'WIN ' if ok else 'FAIL'}  ({note}, "
                  f"{time.time() - t0:.1f}s)")
    if verbose:
        print(f"\n{won}/{len(todo)} levels solved and replayed "
              f"({total - won} replay failures)")
    return 0 if won == total and total == len(todo) else 1


def _random_board(rng: random.Random) -> _Board:
    h, w = rng.randint(3, 6), rng.randint(3, 6)
    n = h * w
    f2 = 0
    for i in range(n):
        if rng.random() < 0.35:
            f2 |= 1 << i
    return _Board(h, w, f2, 1 << rng.randrange(n), 1 << rng.randrange(n))


def _random_state(board: _Board, rng: random.Random):
    n = board.n
    cells = list(range(n))
    rng.shuffle(cells)
    st = [0, 0, 0, 0]
    c1 = c2 = 0
    used: set[int] = set()
    ground = [i for i in cells if not (board.floor2 >> i & 1)]
    for _ in range(rng.randint(0, 3)):
        cand = [i for i in ground if i not in used]
        if not cand:
            break
        i = rng.choice(cand)
        used.add(i)
        st[rng.randrange(4)] |= 1 << i
    for _ in range(rng.randint(0, 4)):
        # deliberately allow a crate to share a square with a staircase: it is
        # the composition the fall-onto-a-staircase rule creates
        cand = [i for i in ground if i not in used or rng.random() < 0.5]
        if not cand:
            break
        i = rng.choice(cand)
        used.add(i)
        c1 |= 1 << i
    for _ in range(rng.randint(0, 4)):
        cand = [i for i in range(n) if (board.floor2 >> i & 1) or (c1 >> i & 1)]
        if not cand:
            break
        c2 |= 1 << rng.choice(cand)
    stt = tuple(st)
    if rng.random() < 0.5:
        cand = [i for i in range(n)
                if not (board.floor2 >> i & 1) and not (c1 >> i & 1)
                and _Board.stair_at(stt, i) < 0]
        return None if not cand else (rng.choice(cand), 1, c1, c2, stt)
    cand = [i for i in range(n)
            if ((board.floor2 >> i & 1) or (c1 >> i & 1)
                or _Board.stair_at(stt, i) >= 0) and not (c2 >> i & 1)]
    return None if not cand else (rng.choice(cand), 2, c1, c2, stt)


#: Macro closure `--selfcheck` is allowed to enumerate per level when it
#: re-derives a field from scratch to double-check one. Deliberately smaller
#: than `FIELD_CAP`: the point is a second, independent measurement of the
#: levels where both answers exist, not a re-run of generation.
SELFCHECK_FIELD_CAP = 90_000

#: Macro closure above which `--selfcheck` stops BRUTE-FORCING the optimal sets.
#: The brute force re-solves the whole level from every successor of every step,
#: so it is quadratic in the closure; on the levels small enough to run it, it is
#: an independent measurement of the field's tie sets.
SELFCHECK_BRUTE_CAP = 25_000


#: Boards written out by hand, one per branch of `_Board.step` that random
#: seating reaches rarely or never. Each is a list of rows of per-cell tokens in
#: `show`'s notation: the first character is the floor (``.`` Floor1 / ``#``
#: Floor2) and the rest are ``I`` Target1, ``J`` Target2, ``1`` Crate1,
#: ``2`` Crate2, ``DURL`` the four staircases, ``P`` Player1, ``Q`` Player2.
#: `--selfcheck` drives all four presses from each of them against the
#: interpreter, so a branch that the fuzz never happens to hit is still measured
#: rather than reported as covered.
_HAND_CASES = (
    # -- Player2 stepping off the top of a staircase, over every kind of square
    (".  .   .", ".  .DQ .", ".  .   ."),                        # nosupport
    (".  .   .", ".  .DQ .", ".  #   ."),                        # onto Floor2
    (".  .   .", ".  .DQ .", ".  .1  ."),                        # onto a Crate1
    (".  .   .", ".  .DQ .", ".  .U  ."),                        # onto a staircase
    # -- and shoving the Crate2 it finds there
    (".  .   .", ".  .DQ .", ".  #2  .", ".  #   ."),            # shove.move
    (".  .   .", ".  .DQ .", ".  #2  .", ".  .   ."),            # shove.fall
    (".  .   .", ".  .DQ .", ".  #2  .", ".  .U  ."),            # shove.stair
    (".  .   .", ".  .DQ .", ".  #2  .", ".  #2  ."),            # shove.cancel
    (".  .   .", ".  .DQ .", ".  #2  ."),                        # shove.offmap
    (".  .   .", ".  .DQ .", ".  .12 .", ".  .   ."),            # shove.stackfall
    (".  .   .", ".  .DQ .", ".  .12 .", ".  .1  ."),            # shove.stackmove
    (".  .   .", ".  .DQ .", ".  .12 ."),                        # shove.crush
    # -- Player2 walking off the bottom of a staircase, back to the ground
    (".  .   .", ".  .   .", ".  .DQ .", ".  .   ."),            # descend
    (".  .   .", ".  #   .", ".  .DQ .", ".  .   ."),            # descend.ceiling
    (".  .   .", ".  .U  .", ".  .DQ .", ".  .   ."),            # descend.ontostair
    (".  .   .", ".  .R  .", ".  .DQ .", ".  .   ."),            # descend.shovestair
    (".  .   .", ".  .1  .", ".  .DQ .", ".  .   ."),            # descend.push
    (".  #   .", ".  .1  .", ".  .DQ .", ".  .   ."),            # descend.pushblocked
    (".  .12 .", ".  .DQ .", ".  .   ."),                        # descend.stack
    (".  .1  .", ".  .DQ .", ".  .   ."),                        # descend.pushedge
    # -- Player2 on Floor2: boarding a staircase, and shoving Crate2s about
    (".  .D  .", ".  #Q  .", ".  #   ."),                        # up.board
    (".  #Q  .", ".  .D  .", ".  .   ."),                        # up.stairblocked
    (".  #Q  .", ".  #2  .", ".  #   ."),                        # up.shove.move
    (".  #Q  .", ".  #2  .", ".  .   ."),                        # up.shove.fall
    (".  #Q  .", ".  #2  .", ".  .U  ."),                        # up.shove.stair
    (".  #Q  .", ".  #2  .", ".  #2  ."),                        # up.shove.cancel
    (".  #Q  .", ".  #2  ."),                                    # up.shove.offmap
    (".  #Q  .", ".  .12 ."),                                    # up.shove.crush
    (".  .1Q .", ".  .12 ."),                                    # up.shove.offmap (from a Crate1)
    (".  #Q  .", ".  .12 .", ".  .   ."),                        # up.shove.fall (stacked)
    (".  #Q  .", ".  .1  .", ".  .   ."),                        # up.walk onto a Crate1
    # -- the cross-block boarding: on a staircase, standing on a Crate1, an
    #    EARLIER staircase type is boarded even sideways
    (".  .D  .", ".  .1LQ .", ".  .   ."),
    (".  .   .", ".D .1LQ .", ".  .   ."),
    # -- Player1 on the ground: crates, staircases and stacks
    (".  .P  .1  .   .", ".  .   .   .   ."),                    # pushcrate
    (".  .P  .1  #   .", ".  .   .   .   ."),                    # pushcrate.blocked
    (".  .P  .1", ".  .   .  "),                                 # pushcrate.edge
    (".  .P  .12 .   .", ".  .   .   .   ."),                    # stack.blocked
    (".  .P  .U  .   .", ".  .   .   .   ."),                    # pushstair
    (".  .P  .U  #   .", ".  .   .   .   ."),                    # pushstair.blocked
    (".  .P  .U", ".  .   .  "),                                 # pushstair.edge
    (".  .P  .1U .   .", ".  .   .   .   ."),                    # pushstair.withcrate
    (".  .P  .12U .  .", ".  .   .    .  ."),                    # pushstair.stackleft
    (".  .P  .R  .   .", ".  .   .   .   ."),                    # climb
    (".  .P  #   .   .", ".  .   .   .   ."),                    # ceiling
    # -- climbing onto a staircase that has a Crate2 on it crushes the Crate2
    (".  .P  .1D2 .  .", ".  .   .    .  ."),
    # -- a target under everything, and a win on each storey
    (".I .P  .   .", ".  .   .   ."),
    (".  #J  #Q  .", ".  #   #   ."),
)


def _parse_case(rows):
    """A ``(board, state)`` pair from one `_HAND_CASES` entry."""
    grid = [r.split() for r in rows]
    h, w = len(grid), max(len(r) for r in grid)
    f2 = t1 = t2 = c1 = c2 = 0
    st = [0, 0, 0, 0]
    p = storey = None
    for r, row in enumerate(grid):
        for c, tok in enumerate(row):
            i = r * w + c
            b = 1 << i
            if tok[0] == "#":
                f2 |= b
            for ch in tok[1:]:
                if ch == "I":
                    t1 |= b
                elif ch == "J":
                    t2 |= b
                elif ch == "1":
                    c1 |= b
                elif ch == "2":
                    c2 |= b
                elif ch in "DURL":
                    st["DURL".index(ch)] |= b
                elif ch == "P":
                    p, storey = i, 1
                elif ch == "Q":
                    p, storey = i, 2
    return _Board(h, w, f2, t1, t2), (p, storey, c1, c2, tuple(st))


def _selfcheck(walk_presses: int = 600, boards: int = 4000,
               verbose: bool = True) -> int:
    """Model vs interpreter, plus the searches checked against each other.

    Six passes, each of which can only fail:

    1. seeded random walks on all 20 shipped levels, model state compared with
       the interpreter's grid after EVERY press;
    2. the same on thousands of randomly SEATED boards, with a coverage counter
       per branch of `_Board.step` (a branch nothing reached is reported, not
       silently counted as verified);
    3. ACTION5 is a no-op on every level (``noaction`` is set, so it must be);
    4. every level whose macro closure fits `SELFCHECK_FIELD_CAP` has its `d*`
       re-derived by an estimate-free macro Dijkstra and compared with the
       field's, and with the cached plan's length;
    5. the exact optimal SETS of those levels are brute-forced -- every press
       re-solved from the state it is taken at -- and compared with the ones the
       field emitted;
    6. every cached plan is replayed on the model, and each step's optimal set
       is required to contain the press taken and to name only presses that
       change something.
    """
    from collections import Counter
    game, expert = _new()
    g, eng = game._game, game._engine
    cov, bad = Counter(), Counter()
    shown = 0
    rng = random.Random(20260822)

    def compare(board, s, d) -> tuple:
        nonlocal shown
        seat_state(eng, g, board, s)
        eng.step(DIRS[d])
        model = board.step(s, d)
        actual = read_state(eng, g)
        cov[board.tag] += 1
        if model != actual:
            bad[board.tag] += 1
            if shown < 6 and verbose:
                shown += 1
                print(f"--- MISMATCH tag={board.tag} press={DIRS[d]}")
                print("FROM:\n" + show(board, s))
                print("MODEL:\n" + show(board, model))
                print("ENGINE:\n" + show(board, actual))
            return False, actual
        if board.won(model) != eng.check_win():
            bad[board.tag] += 1
            return False, actual
        return True, model

    # 1. random walks on the shipped levels
    presses = fails = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = read_board(eng, g)
        s = read_state(eng, g)
        for _ in range(walk_presses):
            ok, s = compare(board, s, rng.randrange(4))
            presses += 1
            fails += not ok
            if board.won(s):
                game.set_level(level)
                s = read_state(eng, g)
    if verbose:
        print(f"1. shipped levels: {presses} presses, {fails} mismatches")

    # 2. hand-written boards for the branches random seating reaches rarely,
    #    then thousands of randomly seated ones
    presses2 = fails2 = 0
    for rows in _HAND_CASES:
        board, s = _parse_case(rows)
        for d in range(4):
            ok, _ = compare(board, s, d)
            presses2 += 1
            fails2 += not ok
    for _ in range(boards):
        board = _random_board(rng)
        s = _random_state(board, rng)
        if s is None:
            continue
        for _ in range(6):
            ok, s = compare(board, s, rng.randrange(4))
            presses2 += 1
            fails2 += not ok
            if board.won(s):
                break
    missing = [t for t in _ALL_TAGS if t not in cov]
    if verbose:
        print(f"2. random boards:  {presses2} presses, {fails2} mismatches")
        print("   branch coverage:")
        for k in sorted(cov):
            print(f"     {k:28s} {cov[k]:8d}  mismatches={bad[k]}")
        if missing:
            print(f"   UNCOVERED BRANCHES: {missing}")

    # 3. ACTION5 is a no-op
    act_bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        before = read_state(eng, g)
        eng.step("action")
        act_bad += read_state(eng, g) != before
    if verbose:
        print(f"3. ACTION5 no-op: {act_bad} of {game.n_levels} levels changed")

    # 4/5. re-derive the small levels' fields and re-prove them
    proofs = proof_bad = ties = tie_bad = 0
    stair_crate = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = read_board(eng, g)
        s0 = read_state(eng, g)
        field = _Field.build(board, s0, cap=SELFCHECK_FIELD_CAP)
        if field is None:
            continue
        proofs += 1
        got = field.dstar(s0)
        ref = _macro_dijkstra(board, s0)
        if got != ref:
            proof_bad += 1
            if verbose:
                print(f"   level {level}: field {got} vs Dijkstra {ref}")
        plan = field.plan(s0)
        entry = expert._disk.get(level)
        cached = None if entry is None else entry["plan"]
        if cached is not None and plan is not None and len(cached) != len(plan):
            proof_bad += 1
            if verbose:
                print(f"   level {level}: cached plan {len(cached)} presses, "
                      f"field says {len(plan)}")
        # brute-force the exact sets (small levels only: quadratic)
        if plan is not None and field.n_states <= SELFCHECK_BRUTE_CAP:
            s = s0
            for i, taken in enumerate(plan):
                remaining = len(plan) - i
                brute = []
                for d in range(4):
                    ns = board.step(s, d)
                    if ns == s:
                        continue
                    sub = _macro_dijkstra(board, ns)
                    if sub is not None and sub + 1 == remaining:
                        brute.append(DIRS[d])
                ties += 1
                if sorted(brute) != sorted(plan.optsets[i]):
                    tie_bad += 1
                    if verbose and tie_bad <= 5:
                        print(f"   level {level} step {i}: field "
                              f"{plan.optsets[i]} vs brute {brute}")
                s = board.step(s, DIRS.index(taken))
        # 6a. does a Crate1 ever share a square with a staircase here?
        for st in field.ids:
            if st[2] & (st[4][0] | st[4][1] | st[4][2] | st[4][3]):
                stair_crate += 1
                break
    if verbose:
        print(f"4. field proofs:   {proofs} levels enumerated, "
              f"{proof_bad} disagreements")
        print(f"5. exact tie sets: {ties} steps brute-forced, "
              f"{tie_bad} disagreements")
        print(f"   levels whose closure reaches a Crate1 on a staircase (the "
              f"one aliased render): {stair_crate}")

    # 6. every cached plan replayed on the model, with its labels. Read
    #    straight out of the disk cache: `--selfcheck` checks what SHIPS, and
    #    calling the expert would silently re-derive a missing level instead.
    plan_bad = labelled = on_stair = 0
    for level in sorted(expert._disk):
        entry = expert._disk[level]
        if entry["plan"] is None:
            continue
        game.set_level(level)
        board = read_board(eng, g)
        s0 = read_state(eng, g)
        plan = entry["plan"]
        sets = entry.get("optsets")
        s = s0
        for i, name in enumerate(plan):
            if sets is None or i >= len(sets) or not sets[i]:
                plan_bad += 1
                break
            if name not in sets[i]:
                plan_bad += 1
                break
            for alt in sets[i]:
                if board.step(s, DIRS.index(alt)) == s:
                    plan_bad += 1
                    break
            labelled += 1
            s = board.step(s, DIRS.index(name))
            if s[2] & (s[4][0] | s[4][1] | s[4][2] | s[4][3]):
                on_stair += 1
        if not board.won(s):
            plan_bad += 1
            if verbose:
                print(f"   level {level}: cached plan does not win on the model")
    if verbose:
        print(f"6. cached plans:   {len(expert._disk)} levels, {labelled} "
              f"labelled steps, {plan_bad} violations")
        print(f"   states along a shipped plan with a Crate1 on a staircase "
              f"(the composition the render aliases, and the only place the "
              f"rule order is not rotation-equivariant): {on_stair}")

    total_bad = (fails + fails2 + act_bad + proof_bad + tie_bad + plan_bad
                 + sum(bad.values()) + len(missing))
    if verbose:
        print(f"\nTOTAL violations: {total_bad}")
    return 0 if total_bad == 0 else 1


def _macro_dijkstra(board: _Board, s0) -> int | None:
    """Shortest presses to a win, by an estimate-free Dijkstra over macros.

    The win rides the queue as a virtual node (see `_astar`), so this is a real
    proof rather than the first win the sweep happens to generate."""
    best = {s0: 0}
    pq = [(0, 0, s0, False)]
    cnt = 0
    while pq:
        g, _c, s, is_goal = heapq.heappop(pq)
        if is_goal:
            return g
        if g > best.get(s, INF):
            continue
        dist, _par, exits = board.quiet(s)
        gw = board.goal_walk(s, dist)
        if gw is not None:
            cnt += 1
            heapq.heappush(pq, (g + gw, cnt, s, True))
        for cost, _kf, _d, ns in exits:
            ng = g + cost
            if ng < best.get(ns, INF):
                best[ns] = ng
                cnt += 1
                heapq.heappush(pq, (ng, cnt, ns, False))
    return None


_ALL_TAGS = (
    "edge",
    "p1.climb", "p1.pushstair", "p1.pushstair.withcrate",
    "p1.pushstair.stackleft", "p1.pushstair.blocked", "p1.pushstair.edge",
    "p1.pushcrate", "p1.pushcrate.blocked", "p1.pushcrate.edge",
    "p1.stack.blocked", "p1.walk", "p1.ceiling",
    "p2.stair.crossboard", "p2.stair.sideways",
    "p2.up.board", "p2.up.stairblocked", "p2.up.walk", "p2.up.nosupport",
    "p2.on.ontocrate", "p2.on.ontofloor2", "p2.on.nosupport",
    "p2.up.shove.crush", "p2.up.shove.offmap", "p2.up.shove.cancel",
    "p2.up.shove.move", "p2.up.shove.stair", "p2.up.shove.fall",
    "p2.on.shove.crush", "p2.on.shove.offmap", "p2.on.shove.cancel",
    "p2.on.shove.move", "p2.on.shove.stair", "p2.on.shove.fall",
    "p2.on.shove.stackfall", "p2.on.shove.stackmove",
    "p2.descend", "p2.descend.ceiling", "p2.descend.stack",
    "p2.descend.push", "p2.descend.pushblocked", "p2.descend.pushedge",
    "p2.descend.ontostair", "p2.descend.shovestair",
)


def _bound(limit: int = PRESS_BUDGET, only: int | None = None) -> int:
    """Run `_bounded` over the levels this generator skips."""
    game, _expert = _new()
    eng = game._engine
    todo = ([only] if only is not None
            else sorted(UpstairsDownstairsSolver.skip_levels))
    for level in todo:
        game.set_level(level)
        board = read_board(eng, game._game)
        s0 = read_state(eng, game._game)
        verdict, states, secs = _bounded(board, s0, limit)
        if verdict == "none":
            what = f"PROVED: no plan of {limit} presses or fewer"
        elif verdict == "inconclusive":
            what = "inconclusive (budget ran out)"
        else:
            what = f"shortest plan is {verdict} presses"
        print(f"level {level:2d}: {what}  ({states} macro states, {secs:.0f}s)")
    return 0


def _recovery(trials: int = 10, verbose: bool = True) -> int:
    """Re-plan from PERTURBED mid-episode states, not just from level starts.

    ``recovery_mode = "reset"`` means the recorder only ever asks for a plan at
    the level start, so this is not exercised by generation -- but the expert is
    required to answer from any reachable board (a plan cache that only knows
    the start state is dead weight for the exploration data this corpus is for),
    and the exact field answers one instantly. Each trial walks a prefix of the
    shipped plan, takes one to three RANDOM presses off it, and re-plans; the
    answer is replayed on the model and must win.

    Only the levels whose macro closure fits `FIELD_CAP` are run: on the other
    four a re-plan is a fresh A* ladder, minutes per trial, and what that would
    measure is the ladder's patience rather than the expert's contract.
    """
    game, expert = _new()
    g, eng = game._game, game._engine
    rng = random.Random(3)
    ok = none = bad = 0
    for level in sorted(expert._disk):
        entry = expert._disk[level]
        if entry["plan"] is None:
            continue
        game.set_level(level)
        board = read_board(eng, g)
        s0 = read_state(eng, g)
        field = _Field.build(board, s0, cap=FIELD_CAP)
        if field is None:
            if verbose:
                print(f"level {level:2d}: no field (closure over FIELD_CAP) "
                      f"-- skipped")
            continue
        plan = entry["plan"]
        won = lost = 0
        for _ in range(trials):
            s = s0
            for d in plan[:rng.randrange(0, len(plan))]:
                s = board.step(s, DIRS.index(d))
            for _ in range(rng.randint(1, 3)):
                s = board.step(s, rng.randrange(4))
            got = field.plan(s)
            if got is None:
                none += 1
                lost += 1
                continue
            cur = s
            for d in got:
                cur = board.step(cur, DIRS.index(d))
            if board.won(cur):
                ok += 1
                won += 1
            else:
                bad += 1
        if verbose:
            print(f"level {level:2d}: {won}/{trials} re-planned to a WIN, "
                  f"{lost} correctly answered None")
    if verbose:
        print(f"\nrecovery: {ok} re-planned to a WIN, {none} correctly "
              f"returned None, {bad} wrong")
    return 0 if bad == 0 else 1


def _compositions() -> list:
    """Every set of objects that can share one cell in this game."""
    out = []
    for t1 in (False, True):
        for stair in (None,) + STAIRS:
            base = ["floor1"] + (["target1"] if t1 else []) \
                + ([stair] if stair else [])
            out.append(base)
            if stair is None:
                out.append(base + ["player1"])
            else:
                out.append(base + ["player2"])       # standing on the staircase
            crate = base + ["crate1"]
            out.append(crate)
            out.append(crate + ["crate2"])
            out.append(crate + ["player2"])
    for t2 in (False, True):
        base = ["floor2"] + (["target2"] if t2 else [])
        out.append(base)
        out.append(base + ["crate2"])
        out.append(base + ["player2"])
    return out


def _audit(verbose: bool = True) -> int:
    """Render every reachable composition as a WHOLE frame and diff them.

    Whole frames, not a cell crop: the renderer upscales the board to fill the
    64x64 view, so cropping a cell by arithmetic measures the wrong pixels.
    """
    game, _expert = _new()
    g, eng = game._game, game._engine
    idx = g.obj_name_to_idx
    comps = _compositions()

    def shoot(comp, h, w):
        eng.load_level([[set() for _ in range(w)] for _ in range(h)])
        for r in range(h):
            for c in range(w):
                eng.grid[r][c] = set()
                eng._cell_add(r, c, idx["background"])
                for name in comp:
                    eng._cell_add(r, c, idx[name])
        eng._build_position_index()
        return _render_frame(eng, g)

    sizes = sorted({(len(lvl), len(lvl[0])) for lvl in g.levels})
    total = 0
    for (h, w) in sizes:
        frames: dict = {}
        clashes = []
        for comp in comps:
            key = tuple(sorted(comp))
            data = shoot(comp, h, w).tobytes()
            if data in frames:
                clashes.append((frames[data], key))
            else:
                frames[data] = key
        total += len(clashes)
        if verbose:
            cell_px = max(1, min(64 // h, 64 // w))
            print(f"{h:2d}x{w:<2d} (cell_px {cell_px}): "
                  f"{len(comps)} compositions, {len(clashes)} clashes")
            for a, b in clashes:
                print(f"     {a}  ==  {b}")
    if verbose:
        print(f"\n{total} clashes over {len(sizes)} board sizes")
    return 0 if total == 0 else 1


def _symmetry(presses: int = 200, verbose: bool = True) -> int:
    """The augmentation is presentation-only, so the same ENGINE presses must
    leave the same engine grid at every rotation, and the frames must differ by
    exactly that rotation."""
    from solvers.common.ps_astar import screen_action
    from arcengine import ActionInput

    rng = random.Random(7)
    bad = 0
    checked = 0
    for level in range(20):
        script = [rng.randrange(4) for _ in range(presses)]
        ref_frames = None
        ref_grid = None
        for seed in range(12):
            game = PuzzleScriptAdapter(GAME_NAME, seed=seed)
            game.set_level(level)
            k = game._rotation_k
            frames = [np.rot90(np.asarray(game._current_frame), k=-k)]
            for d in script:
                act = screen_action(DIRS[d], k, game._hflip, game._vflip)
                fd = game.perform_action(ActionInput(id=act))
                fr = np.asarray(fd.frame[-1] if fd.frame
                                else game._current_frame)
                frames.append(np.rot90(fr, k=-k))
            grid = [[frozenset(cell) for cell in row]
                    for row in game._engine.grid]
            if ref_frames is None:
                ref_frames, ref_grid = frames, grid
                continue
            checked += 1
            if grid != ref_grid:
                bad += 1
                if verbose:
                    print(f"  level {level} seed {seed}: engine grid differs")
            elif any(not np.array_equal(a, b)
                     for a, b in zip(frames, ref_frames)):
                bad += 1
                if verbose:
                    print(f"  level {level} seed {seed}: un-rotated frames "
                          f"differ")
    if verbose:
        print(f"symmetry: {checked} (level, seed) pairs, {bad} violations")
    return 0 if bad == 0 else 1


def _speed(verbose: bool = True) -> int:
    """Model presses/s against interpreter presses/s, on the shipped levels."""
    game, _expert = _new()
    g, eng = game._game, game._engine
    rng = random.Random(1)
    game.set_level(0)
    board = read_board(eng, g)
    s0 = read_state(eng, g)
    script = [rng.randrange(4) for _ in range(4000)]
    t = time.time()
    s = s0
    for d in script:
        s = board.step(s, d)
    model = len(script) / (time.time() - t)
    game.set_level(0)
    t = time.time()
    for d in script[:1500]:
        eng.step(DIRS[d])
    interp = 1500 / (time.time() - t)
    if verbose:
        print(f"model {model:,.0f} presses/s   interpreter {interp:,.0f} "
              f"presses/s   ({model / interp:.0f}x)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plans", action="store_true",
                    help="solve every level and replay it through the engine")
    ap.add_argument("--all", action="store_true",
                    help="with --plans, also re-try the skipped levels")
    ap.add_argument("--level", type=int, default=None,
                    help="with --plans, only this level")
    ap.add_argument("--selfcheck", action="store_true",
                    help="model vs interpreter, field proofs, tie-set checks")
    ap.add_argument("--audit", action="store_true",
                    help="render every cell composition and diff whole frames")
    ap.add_argument("--symmetry", action="store_true",
                    help="drive the same presses at every rotation")
    ap.add_argument("--recovery", action="store_true",
                    help="re-plan from perturbed mid-episode states")
    ap.add_argument("--bound", action="store_true",
                    help="prove (or fail to) that a skipped level has no plan "
                         "inside the adapter's press budget")
    ap.add_argument("--limit", type=int, default=PRESS_BUDGET,
                    help="with --bound, the press budget to test")
    ap.add_argument("--speed", action="store_true",
                    help="model vs interpreter presses per second")
    args, rest = ap.parse_known_args(argv)

    if args.plans:
        levels = ([args.level] if args.level is not None
                  else list(range(20)) if args.all else None)
        return _plans(levels)
    if args.selfcheck:
        return _selfcheck()
    if args.recovery:
        return _recovery()
    if args.bound:
        return _bound(args.limit, args.level)
    if args.audit:
        return _audit()
    if args.symmetry:
        return _symmetry()
    if args.speed:
        return _speed()
    return UpstairsDownstairsSolver.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
