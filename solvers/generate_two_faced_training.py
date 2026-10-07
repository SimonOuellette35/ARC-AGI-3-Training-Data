"""Generate Phase-1 training data for the PuzzleScript game ps:two_faced
("Two-faced", Le Slo).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanic, a native model of it, the exact distance FIELD that model is solved
with, the proof that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_two_faced",
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

THE GAME: you are a DOMINO, and which face you lead with picks the crate
--------------------------------------------------------------------------
The player is two cells, not one: either a VERTICAL domino (PlayerT on top of
PlayerB) or a HORIZONTAL one (PlayerL left of PlayerR). Nine levels, 6x9 to
10x15, and two kinds of crate -- CrateH and CrateV -- each of which has to end
on its own kind of target (``All TargetH on CrateH`` and ``All TargetV on
CrateV``, both at once).

The 16 push rules in the .txt are one sentence once they are tabulated. Call a
press PERPENDICULAR when it is across the domino's long axis (a vertical body
pressed left/right, a horizontal body pressed up/down) and PARALLEL when it is
along it. Then:

    PERPENDICULAR presses push CrateH and are STOPPED by CrateV
    PARALLEL      presses push CrateV and are STOPPED by CrateH

So CrateH is the crate you shove with your broad side -- and a perpendicular
press shoves the cell in front of BOTH halves, so it can move two CrateH at
once, which is what levels 6 and 7 are made of. CrateV is the crate you shove
with your end, one at a time, because a parallel press has only one leading
half. Nothing else in the game moves a crate, so which crate you can touch at
all is decided by how you are lying, and how you are lying is decided by:

**THE PIVOT, and the fact that a WALL is not a pivot.** A press where exactly
one half is blocked does not fail -- the free half steps and the blocked half
is abandoned, and the two auxM cells the rule pass leaves behind are the free
half's OLD cell and its NEW one, which are adjacent ACROSS the old axis. The
domino has tipped over 90 degrees. But this only happens when the blocking
thing is a crate: a player half moving into a Wall matches ``[ > Player | Wall ]
-> cancel``, which reverts the entire turn, so a wall gives you nothing at all.
The only ways to turn are therefore to lean on a crate of the type this press
cannot push, or on one that is jammed against a wall -- and on the levels with
no CrateV at all (6 and 7), the jam is the ONLY way, which is why they are the
long ones.

**A jammed half jams the whole body.** ``[walls auxFP][> crate] -> [walls auxFP]
[crate]`` kills every crate force on the board the moment either front cell
reads as blocked, and ``[collisionT auxFP| walls auxFP]`` then marks the OTHER
front cell blocked too if it holds a crate. So a press with a crate in front of
one half and a jam in front of the other does nothing whatsoever -- no push, no
step, no pivot. The pivot needs the free half's square to be genuinely EMPTY.

Everything above is measured, not read: `_Board` is the model and
``--selfcheck`` fuzzes it against the real interpreter press for press.

Expert solver
-------------
A NATIVE model (`_Board`, ~1.1M presses/s against the interpreter's 269 --
both measured) plus an exact distance field (`_Field`). The shared
`PSPushExpert` does not apply: its macro is "walk to a push square, then push",
which presumes a one-cell player that can reach any square, and here the body
is two cells whose ORIENTATION is part of the position and can only be changed
by leaning on a crate.

`_Field` is a forward breadth-first sweep from the state it is asked about,
stopped at the depth ``d*`` where the first win appears, recording as it goes

  * ``depth[id]``  -- presses from the start, and
  * ``succ[4*id + di]`` -- the id each press leads to (-1 for a press that
    changes nothing),

both as flat `array('i')`s, because level 7's ball is 9.09 million states and a
dict of Python lists is not a shape that fits in memory at that size.

The tie sets then come out of ONE backward pass over those arrays and NO
predecessor map, which is the trick worth keeping. This game is IRREVERSIBLE
(you cannot pull a crate), so swap_sokoban's single backward sweep from the win
is illegal here and towers_of_saigon's recorded predecessor map is what one
normally falls back to -- at 9.09M states that map is gigabytes. It is not needed:

    ONPATH(s)  ==  s lies on some shortest start-to-win path
               ==  s is a won state at depth d*, or some successor t of s has
                   depth(s) + 1 and is itself ONPATH

and ids are handed out in BFS order, so depth is non-decreasing in id and
scanning ids DOWNWARDS is exactly scanning layers downwards. The recurrence
only ever looks FORWARD along `succ`, which is already stored.

That set is all the labelling needs, because for a state ``s_i`` at step ``i``
of a shortest plan, a press is equally-shortest exactly when its successor
``t`` has ``depth(t) == i+1`` and is ONPATH: forwards, ``t`` at depth ``i+1``
and ONPATH gives ``dist-to-win(t) = d* - i - 1``; backwards, a ``t`` with that
distance cannot have ``depth(t) < i+1`` or ``d*`` would not be minimal. So the
sets are EXACT, with nothing inferred and no re-solving.

Shortest, and how that is known
-------------------------------
Every plan on every level is provably shortest, and the proof is three
independent derivations agreeing:

  * this file's field, which is a breadth-first sweep and therefore shortest by
    construction;
  * a plain forward BFS to the first win (`_forward`) that shares no line with
    `_Field` but `_Board.step` itself (``--selfcheck`` pass 4), whose per-step
    tie sets are then re-derived a second time by brute force (pass 5);
  * the same two sweeps driven by the REAL INTERPRETER on the levels small
    enough to afford 269 presses a second (``--engine``): forward BFS by
    pressing actual buttons until a win appears, edges recorded, then backward
    over those edges. No `_Board` call participates.

Recovery
--------
`supports_recovery` with the family's RESET-mode prefix. The mechanic is
irreversible, so a perturbed board can be genuinely lost and the episode-wide
exploration prefix is undone by ONE RESET back to the level start, from which
the cached plan is a guaranteed win.

On top of that the five levels whose field costs under half a second
(``_EPSILON_LEVELS``) also take epsilon detours INSIDE the expert replay: the
expert re-plans from the LIVE board by building a fresh field there, so the
taken action is the mistake and ``optimal`` is the exact recovery. The four big
levels are excluded by `epsilon_for` rather than by a node cap that would fire
silently -- level 7's field is 38 seconds, which is a startup cost once and an
unacceptable per-detour cost. ``--selfcheck``'s recovery pass measures both
halves of that: from random off-plan boards the field must return a plan an
independent BFS agrees with and which WINS when replayed through the
interpreter, and it reports how many of those boards are genuinely dead (they
exist -- shove a CrateH into a corner and the level is over).

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed
ASCII maps), so the only per-(seed, level) variables are the frame rotation
(``rotation_k`` in 0..3) and the two flips, with their matching directional
action remap. 9 levels x 16 presentations = 144.

``Two-faced`` is in `PuzzleScriptAdapter._FLIP_GAMES`; the argument is recorded
there and ``--symmetry`` measures it. The short version is that this game's ART
is an exact orbit of the symmetry group in the strongest possible way: the
anti-diagonal reflection (``rot90`` then ``fliplr``) carries PlayerT's sprite
pixel-for-pixel onto PlayerR's, PlayerB's onto PlayerL's, and back -- which is
exactly the relabelling the mechanic makes, since that reflection turns a
vertical body into a horizontal one and sends its top half to the right and its
bottom half to the left. ``--audit`` prints that orbit; it is not a clash,
because a board holds EITHER a T/B pair or an L/R pair and never one of each,
so no frame is ever ambiguous.

Rendering
---------
Two sprite classes had to change and between them they were 114 identical
frames.

  * **The goal vanished under the player.** TargetH and TargetV were 3x3 rings
    on rows/cols 1-3, which every one of the four player sprites paints over,
    so ``player on target`` rendered as a player on bare floor -- on every level,
    at every size. In a game where the body is TWO cells and both halves are
    walking over the target squares constantly, that erases the win condition
    from the picture more or less permanently.
  * **On level 1 the win condition itself was invisible.** That level is 9x15,
    which renders at 4 px per cell, and at 4 px the centred nearest-neighbour
    sampler drops sprite row and column 2 -- which is the only pixel that
    separates ``crate on the RIGHT target`` (CrateWH/CrateWV, the green marker)
    from ``crate on the WRONG one`` (CrateNWH/CrateNWV, the red one). Both also
    matched a bare crate.

The fix is one shape for both problems: the four corners. Every player sprite
now leaves all four corners transparent (each already left two), and the two
Target sprites paint them. The crate sprites are hollow 5x5 boxes whose corners
were already transparent, so ``crate on target`` now shows the target's colour
in its corners as well -- which is what separates the green marker from the red
one at 4 px. The full statement is the header comment in
``data/puzzlescript_games/Two-faced.txt``.

``--audit`` renders every cell COMPOSITION a board of this game can hold -- 22
of them -- as whole 64x64 frames of uniform boards, at every board shape the
nine levels use, and requires them pairwise distinct. Whole frames rather than
one cell sliced out of a mixed board: `_render_frame` upscales and centre-pads,
so slicing by ``cell_px`` arithmetic reads the wrong pixels (the ps:explod
lesson).

What it measures out at
-----------------------
All 9 levels solved, 435 presses, every one of them provably shortest, 29 steps
carrying a second equally-right answer. The fields: 8.6k / 10.9k / 4.6k / 58k /
109k / 4.34M / 1.92M / 9.09M / 832k states for plans of 22 / 21 / 41 / 58 / 35 /
35 / 68 / 79 / 76 presses, ~70 seconds for the whole game and then cached.

``--selfcheck``  0 violations: 5400 random-walk presses of model-vs-interpreter
with 0 divergences and 0 refusals, 14400 more from 3600 randomly seated boards,
every one of the five small levels' WHOLE reachable space enumerated (11.9k to
196k states) with 0 presses the model had to refuse, all nine plan lengths
re-proved by an independent forward BFS, all 177 tie sets on those levels
re-derived by brute force with 0 mismatches, and 300 off-plan recovery boards of
which 226 re-planned to a WIN in the interpreter and 74 were correctly answered
"no plan" (each confirmed genuinely dead by the same independent BFS -- this
game really does strand, which is why the family's RESET recovery is the right
default here).
``--engine``    0 disagreements on levels 0-2: 24116 boards walked with real
button presses, the same plan lengths and the same tie sets.
``--audit``     0 colliding compositions over all 23 of them at all nine board
shapes (114 before the sprite fix), and the group pass shows the 24 orbit pairs
are all player-half-to-player-half with 0 strays.
``--symmetry``  0 violations with all 16 presentations drawn.
End-to-end: 12 seeds x 9 levels replayed FRAME-EXACT through `perform_action`,
108/108 WIN, max action index 5, 0 of 5686 expert steps unlabelled, ~4.8
s/episode against a warm plan cache, and two processes at the same rng seed
produce byte-identical episodes.

Usage (run from the repo root):
    python solvers/generate_two_faced_training.py --episodes 200 \
        --out data/training_multi_level/two_faced

    python solvers/generate_two_faced_training.py --plans      # level report
    python solvers/generate_two_faced_training.py --selfcheck  # model + optimality
    python solvers/generate_two_faced_training.py --engine     # interpreter proof
    python solvers/generate_two_faced_training.py --audit      # rendering
    python solvers/generate_two_faced_training.py --symmetry   # augmentation
"""

from __future__ import annotations

import itertools
import random
import sys
import time
from array import array
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameState                          # noqa: E402
from solvers.base_solver import _ID_TO_GAMEACTION                     # noqa: E402
from solvers.common.ps_astar import (PSAStarSolver, PSExpert, Plan,   # noqa: E402
                                     screen_action, snapshot, restore)
from utils.rotation import inverse_remap_action_full                  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Two-faced"

#: Engine directions, in the order ties are broken. No rule in this game has
#: ``action`` on a left-hand side, so the X button is not a move and branching on
#: it would double every sweep for nothing (measured: a press of it leaves the
#: grid byte-identical). The exploration prefix still presses it -- a live agent
#: has that button -- which is why ``--symmetry``'s random walk presses it too.
DIRS: tuple[str, ...] = ("up", "down", "left", "right")

#: DIRS index -> (dr, dc). ``di < 2`` is therefore "this press moves vertically",
#: which `_Board.step` uses to decide parallel vs perpendicular.
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Disk cache of every level's start plan AND its optimal-action sets. This IS
#: the load-bearing cache here: the nine fields together are ~70 seconds (level
#: 7's alone is 38) and they are seed-independent, so without a file on disk
#: every shard `parallelize_generator` starts would re-derive all of them.
#: Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "two_faced_plans.json"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic.

    Cells are flat ``r * w + c`` indices. A STATE is ``(p1, p2, mh, mv)``: the
    two cells the player body occupies with ``p1 < p2``, and bitmasks of the
    CrateH and CrateV squares. Walls and the two target sets never change, so
    they live here rather than in the state.

    The body's ORIENTATION is not stored because it is implied: ``p2 - p1 == w``
    is the vertical domino (PlayerT above PlayerB) and ``p2 - p1 == 1`` the
    horizontal one (PlayerL left of PlayerR). That is exact rather than a
    convenience -- the .txt DELETES both halves at the end of every turn and
    re-creates them from the two auxM cells with ``up [auxM|auxM] -> [playerB |
    playerT]`` and ``right [auxM|auxM] -> [playerL | playerR]``, so which half is
    which is a function of the pair's geometry and of nothing else.

    ONE PRESS, in the order the rules fire:

    1. **Fronts.** A perpendicular press has two front cells (one per half); a
       parallel press has one (the trailing half's "front" is the leading half's
       own square).
    2. **Wall -> cancel.** ``[ > Player | Wall ] -> cancel`` reverts the entire
       turn, so a wall in front of either half means the press did not happen.
       This is why a wall never pivots the body.
    3. **Push or jam.** A perpendicular press pushes CrateH and jams on CrateV;
       a parallel press pushes CrateV and jams on CrateH (see the module
       docstring). A pushed crate drags the whole RUN of crates in front of it
       (``[> Crate | crate no auxMultiban]``, any mix of types), and if that run
       ends at a wall the whole run jams instead (``[> Crate | wall]`` marks the
       head, then ``[> Crate | Crate auxWall]`` walks the mark backwards).
    4. **A jam anywhere in front stops everything.** ``[walls auxFP][> crate]``
       drops every crate force on the board, and ``[collisionT auxFP| walls
       auxFP]`` then jams the other front cell too if it holds a crate.
    5. **Move.** Both halves free: the body translates. Exactly one jammed: the
       free half steps and the body PIVOTS onto (free half's old cell, its new
       one). Both jammed: nothing.

    `step` writes step 5 as a table rather than simulating the auxM/auxNM
    marker rules, and the two are equivalent case by case:

        both free (perp)  auxM = {f1, f2}                    -> translate
        one jammed        auxNM = {jammed half}, auxM = {f}, and the GIRO rule
                          ``[no auxNM no auxM player] -> [auxM player]`` adds the
                          free half's OWN cell                -> pivot
        both jammed       ``[auxNM player | auxNM player]`` puts auxM back on
                          both current cells                  -> no move
        parallel free     ``[>P|>P|no walls]`` puts auxM on the leading half and
                          the new cell                        -> translate
        parallel jammed   the ``| walls]`` twin puts auxM on both current cells
                                                              -> no move

    ``--selfcheck`` is the measurement of all of it: a seeded random walk on
    every level comparing this model's state against the interpreter's grid
    after EVERY press, including the presses that do nothing.
    """

    __slots__ = ("h", "w", "n", "wall", "is_wall", "thmask", "tvmask",
                 "free", "nb", "sig")

    def __init__(self, h, w, walls, targets_h, targets_v):
        self.h, self.w = h, w
        self.n = h * w
        self.wall = frozenset(walls)
        #: Bytewise wall test. `step` runs tens of millions of times per level;
        #: a bytearray index beats a frozenset lookup and it is the same answer.
        self.is_wall = bytearray(self.n)
        for i in walls:
            self.is_wall[i] = 1
        self.thmask = 0
        for t in targets_h:
            self.thmask |= 1 << t
        self.tvmask = 0
        for t in targets_v:
            self.tvmask |= 1 << t
        self.free = tuple(i for i in range(self.n) if not self.is_wall[i])
        #: ``nb[cell][di]`` is the neighbouring cell, or -1 off the board. The
        #: wall test is deliberately NOT folded in here (unlike the sokobans in
        #: this family): a wall in front cancels the turn while a jammed crate
        #: pivots the body, so the two have to stay tellable apart.
        self.nb = []
        for i in range(self.n):
            r, c = divmod(i, w)
            row = []
            for d in DIRS:
                dr, dc = _DELTA[d]
                nr, nc = r + dr, c + dc
                row.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1)
            self.nb.append(tuple(row))
        self.sig = (h, w, tuple(sorted(self.wall)),
                    tuple(sorted(targets_h)), tuple(sorted(targets_v)))

    # -- the mechanic --------------------------------------------------------
    def step(self, state, di: int):
        """The state after pressing ``DIRS[di]``.

        Returns ``state`` itself when the press changes nothing (a non-move: no
        shortest path contains one), and None for the two board shapes this
        model refuses to guess at -- a crate shoved at the grid EDGE with no wall
        to stop it, and a body with exactly one half hanging off the edge, where
        the marker rules leave one or three auxM cells and the interpreter would
        delete or duplicate the player. Neither is reachable on any shipped
        level: ``--selfcheck`` enumerates every level's whole reachable space and
        counts the refusals, and the count is 0.
        """
        p1, p2, mh, mv = state
        w = self.w
        nb = self.nb
        is_wall = self.is_wall

        parallel = ((p2 - p1) == w) == (di < 2)

        if parallel:
            # One leading half; the trailing half's front is the leader's cell.
            if (p2 - p1) == w:
                lead, trail = (p1, p2) if di == 0 else (p2, p1)
            else:
                lead, trail = (p1, p2) if di == 2 else (p2, p1)
            f1 = nb[lead][di]
            if f1 < 0 or is_wall[f1]:
                return state                       # off the board / -> cancel
            f2 = -1
            front_bits = 1 << f1
        else:
            f1 = nb[p1][di]
            f2 = nb[p2][di]
            if f1 < 0 or f2 < 0:
                # Both off the board is a stationary turn; exactly one off it is
                # the shape this model refuses (see the docstring).
                return state if (f1 < 0 and f2 < 0) else None
            if is_wall[f1] or is_wall[f2]:
                return state                       # -> cancel
            front_bits = (1 << f1) | (1 << f2)

        # 3 -- push the type this press can move, jam on the type it cannot.
        shoved = mh if not parallel else mv
        stops = mv if not parallel else mh
        occupied = mh | mv
        jam = 0
        chain = None
        for f in (f1, f2):
            if f < 0:
                continue
            bit = 1 << f
            if shoved & bit:
                run = [f]
                cur = f
                while True:
                    nxt = nb[cur][di]
                    if nxt < 0:
                        return None                # a crate shoved off the grid
                    if is_wall[nxt]:
                        for c in run:              # the whole run jams
                            jam |= 1 << c
                        run = None
                        break
                    if (occupied >> nxt) & 1:
                        run.append(nxt)
                        cur = nxt
                        continue
                    break
                if run:
                    chain = run if chain is None else chain + run
            elif stops & bit:
                jam |= bit

        # 4 -- a jam in either front cell stops every crate on the board, and
        # jams the other front cell too when that one also holds a crate.
        if jam & front_bits:
            chain = None
            if f2 >= 0:
                for f in (f1, f2):
                    if (occupied >> f) & 1:
                        jam |= 1 << f

        # 5 -- where the body ends up.
        if parallel:
            if (jam >> f1) & 1:
                q1, q2 = p1, p2
            else:
                q1, q2 = (lead, f1) if lead < f1 else (f1, lead)
        else:
            b1 = (jam >> f1) & 1
            b2 = (jam >> f2) & 1
            if b1:
                if b2:
                    q1, q2 = p1, p2                        # both jammed
                else:
                    q1, q2 = (p2, f2) if p2 < f2 else (f2, p2)   # pivot
            elif b2:
                q1, q2 = (p1, f1) if p1 < f1 else (f1, p1)       # pivot
            else:
                q1, q2 = f1, f2                            # translate

        if chain:
            moving_h = 0
            for c in chain:
                bit = 1 << c
                if mh & bit:
                    moving_h |= bit
            moving_v = 0
            for c in chain:
                bit = 1 << c
                if not (moving_h & bit):
                    moving_v |= bit
            mh ^= moving_h
            mv ^= moving_v
            for c in chain:
                dest = 1 << nb[c][di]
                if (moving_h >> c) & 1:
                    mh |= dest
                else:
                    mv |= dest

        return (q1, q2, mh, mv)

    def won(self, state) -> bool:
        """``All TargetH on CrateH`` and ``All TargetV on CrateV``, verbatim --
        every H target carries an H crate and every V target a V crate. Spare
        crates of either kind are free to sit anywhere; on the shipped levels
        the counts match exactly except on level 3, which has two crates of each
        type and only one target."""
        _p1, _p2, mh, mv = state
        return ((mh & self.thmask) == self.thmask
                and (mv & self.tvmask) == self.tvmask)


# ---------------------------------------------------------------------------
# The exact distance field
# ---------------------------------------------------------------------------

class _Field:
    """Exact shortest plan AND exact per-step tie sets, from one forward sweep.

    A breadth-first sweep from ``start``, stopped at the depth ``d*`` at which
    the first win appears, storing two flat arrays indexed by a state id handed
    out at discovery:

      * ``depth[id]``          -- presses from ``start``;
      * ``succ[4*id + di]``    -- the id pressing ``DIRS[di]`` leads to, or -1
        when that press changes nothing (or the model refuses it).

    Flat `array('i')`s rather than dicts of tuples because level 7's ball is
    9.09 million states; at that size the arrays are 145 MB and the discovery
    dict (which is transient -- it is dropped when the sweep ends) is the only
    part that is measured in gigabytes.

    Then ONE backward pass computes ONPATH -- "this state lies on some shortest
    start-to-win path" -- from the recurrence in the module docstring, using
    only `succ`, which is why this class needs no predecessor map even though
    the game is irreversible. Ids are assigned in BFS order, so depth is
    non-decreasing in id and scanning ids downwards IS scanning layers
    downwards.

    ``cap`` is a runaway guard, not a budget: past it the field answers "no
    plan" rather than quietly returning a long one. It is never reached on the
    shipped levels -- the largest is level 7's 9.09M against a 12M cap -- and
    the re-plans an epsilon detour asks for run under a much smaller cap on the
    five cheap levels only (`TwoFacedExpert.replan_cap`).
    """

    __slots__ = ("board", "dstar", "depth", "succ", "onpath", "n_states",
                 "capped")

    def __init__(self, board: _Board, start, cap: int):
        self.board = board
        step = board.step
        won = board.won

        ids = {start: 0}
        depth = array("i", [0])
        succ = array("i", [-1, -1, -1, -1])
        queue = deque()
        wins: list[int] = []
        dstar = None
        capped = False

        if won(start):
            dstar = 0
            wins.append(0)
        else:
            queue.append((start, 0))

        while queue:
            state, sid = queue.popleft()
            d = depth[sid]
            if dstar is not None and d >= dstar:
                continue                       # nothing past the winning layer
            base = 4 * sid
            for di in range(4):
                nxt = step(state, di)
                if nxt is None or nxt == state:
                    continue                   # refused, or a press that did nothing
                nid = ids.get(nxt)
                if nid is None:
                    nid = len(depth)
                    ids[nxt] = nid
                    depth.append(d + 1)
                    succ.extend((-1, -1, -1, -1))
                    if won(nxt):
                        wins.append(nid)
                        if dstar is None:
                            dstar = d + 1      # a win ENDS the episode: terminal
                    else:
                        queue.append((nxt, nid))
                succ[base + di] = nid
            if len(depth) > cap:
                capped = True
                break

        self.n_states = len(depth)
        self.capped = capped
        self.depth = depth
        self.succ = succ
        self.dstar = None if capped else dstar
        self.onpath = bytearray(self.n_states)
        if self.dstar is None:
            return

        for wid in wins:
            self.onpath[wid] = 1
        onpath = self.onpath
        for sid in range(self.n_states - 1, -1, -1):
            if onpath[sid]:
                continue
            nd = depth[sid] + 1
            if nd > dstar:
                continue
            base = 4 * sid
            for di in range(4):
                t = succ[base + di]
                if t >= 0 and depth[t] == nd and onpath[t]:
                    onpath[sid] = 1
                    break

    def optimal(self, sid: int) -> list:
        """``[(direction index, successor id), ...]`` for every press on a
        shortest path from state ``sid``. Ties come out in ``DIRS`` order, which
        is what makes a re-derived plan byte-identical across processes.

        Exact, with nothing inferred: see the module docstring for why "depth is
        one more and ONPATH" is the same predicate as "one press nearer the
        win"."""
        depth, succ, onpath = self.depth, self.succ, self.onpath
        nd = depth[sid] + 1
        base = 4 * sid
        out = []
        for di in range(4):
            t = succ[base + di]
            if t >= 0 and depth[t] == nd and onpath[t]:
                out.append((di, t))
        return out

    def plan(self) -> "Plan | None":
        """The shortest press sequence from the state this field was built at,
        with the EXACT optimal set at every step, or None when no win is
        reachable (or the cap stopped the sweep)."""
        if self.dstar is None:
            return None
        presses, optsets = [], []
        sid = 0
        for _ in range(self.dstar):
            best = self.optimal(sid)
            if not best:                       # an ONPATH state always has one
                return None
            presses.append(DIRS[best[0][0]])
            optsets.append([DIRS[di] for di, _t in best])
            sid = best[0][1]
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class TwoFacedExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Field`.

    The base class keeps the memo, the level scoping and the disk cache; only the
    strategy underneath changes. Here the strategy is "read the live board and
    sweep an exact field from it", so `heuristic` is never called and asserts
    rather than returning a number nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state (a recovery prefix, an epsilon detour) is answered exactly
    -- including answering NO, which this game really can: shove a CrateH into a
    corner and the level is over. `record_level` turns a None into a dropped
    perturbation or a RESET, which is the whole point of the family's
    reset-mode recovery.
    """

    directions = list(DIRS)
    #: `_key` is the dynamic objects only, which is canonical WITHIN a level but
    #: not across them -- walls and targets are static per level and differ
    #: between levels.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    #: Runaway guard on a LEVEL START's sweep. The largest this game reaches is
    #: level 7's 9.09M states (38 s, ~1.1 GB); the cap is set above it rather
    #: than at it so an edited level fails loudly rather than silently losing a
    #: layer.
    field_cap: int = 12_000_000
    #: The same guard for a sweep from anywhere else -- an epsilon detour or a
    #: recovery board. Only `TwoFacedSolver._EPSILON_LEVELS` ever asks (their
    #: whole balls are 4.6k to 109k), so this is the guard and not the budget.
    replan_cap: int = 400_000

    def setup(self) -> None:
        g = self.g
        self.player_ids = set(self.game._engine._player_indices)
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.crateh_ids = set(g.resolve_object_name("crateh"))
        self.cratev_ids = set(g.resolve_object_name("cratev"))
        self.targeth_ids = set(g.resolve_object_name("targeth"))
        self.targetv_ids = set(g.resolve_object_name("targetv"))
        #: `_Board`s by STATIC signature (see `read`), not by level index, so the
        #: neighbour table is built once per level however many states are read.
        self._boards: dict = {}
        #: The level START states, so `_search` can tell a startup sweep (which
        #: may cost 105 seconds and is cached to disk) from a re-plan (which may
        #: not). Filled by `prepare_expert`.
        self._starts: set = set()

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "TwoFacedExpert reads an exact distance field; heuristic is unused")

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, state)`` for the engine's current grid.

        The crate-on-target MARKERS (CrateWH / CrateWV / CrateNWH / CrateNWV) are
        deliberately not read: they are late-rule decoration derived from the
        crate and target already in the cell, they live on their own collision
        layer and block nothing, so including them would only make the state key
        wider without making it finer."""
        h, w = eng.height, eng.width
        walls, th, tv = [], [], []
        mh = mv = 0
        halves = []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.targeth_ids:
                    th.append(i)
                if cell & self.targetv_ids:
                    tv.append(i)
                if cell & self.crateh_ids:
                    mh |= 1 << i
                if cell & self.cratev_ids:
                    mv |= 1 << i
                if cell & self.player_ids:
                    halves.append(i)
        sig = (h, w, tuple(walls), tuple(th), tuple(tv))
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(h, w, walls, th, tv)
        if len(halves) != 2:
            return board, None
        halves.sort()
        return board, (halves[0], halves[1], mh, mv)

    def _key(self, eng) -> frozenset:
        """``(row, col, tag)`` triples for the player halves (0), the CrateH (1)
        and the CrateV (2).

        Built from the MODEL's reading rather than from raw object ids so the key
        is exactly what a state consists of -- which is also what the
        ``plan_cache_path`` signature is stored as, so a cached plan is matched
        against the board it was solved from and an edited level is a miss rather
        than a wrong plan. The two halves share tag 0 because which one is
        PlayerT and which PlayerB is a function of the pair's geometry (see
        `_Board`), not extra state."""
        _board, state = self.read(eng)
        if state is None:
            return frozenset()
        p1, p2, mh, mv = state
        w = eng.width
        out = {(p1 // w, p1 % w, 0), (p2 // w, p2 % w, 0)}
        for i in range(eng.height * w):
            if (mh >> i) & 1:
                out.add((i // w, i % w, 1))
            if (mv >> i) & 1:
                out.add((i // w, i % w, 2))
        return frozenset(out)

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if state is None:                        # no player on the board
            return None
        if board.won(state):
            return Plan([], [])
        cap = self.field_cap if state in self._starts else self.replan_cap
        return _Field(board, state, cap).plan()


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class TwoFacedSolver(PSAStarSolver):
    game_id = "puzzlescript_two_faced"
    game_name = GAME_NAME
    expert_cls = TwoFacedExpert

    #: `games/ps:two_faced/ps:two_faced.py` is a plain passthrough -- it builds
    #: the adapter and nothing else, and the rendering fix is in the .txt, which
    #: both paths read. Set this if that wrapper ever grows a patch.
    game_module_id = ""

    #: Unused: `TwoFacedExpert._search` never calls `_astar`. Left at the base
    #: values so nothing reads a lie off them; `TwoFacedExpert.field_cap` is the
    #: knob that actually bounds the sweep.
    weight = 1
    node_cap = 400_000

    #: Room for the longest plan (79 presses on level 7) plus the re-plans an
    #: epsilon detour costs on the levels that take them. The adapter's own
    #: 200-press per-level budget is separate and is reset by the `set_level`
    #: that ends the exploration prefix, so the plan starts it from zero.
    max_steps = 140

    #: Every level is solved; nothing is skipped.
    skip_levels = frozenset()

    #: Base rate; `epsilon_for` is what actually decides, per level.
    epsilon = 0.0

    #: The levels whose whole ball to ``d*`` is 4.6k to 109k states -- a field
    #: in 0.4 seconds or less -- so an epsilon detour can be answered by
    #: re-planning from the live board. The other four are 0.83M to 9.09M
    #: (3 to 38 seconds) and are excluded HERE rather than left to a node cap
    #: that would fire silently mid-recording.
    _EPSILON_LEVELS = frozenset({0, 1, 2, 3, 4})

    #: One press in twelve on those levels is a random legal alternative, after
    #: which the expert re-plans from wherever it landed -- the taken action is
    #: the mistake and ``optimal`` is the recovery. Lower than the reversible
    #: games in this family run at, because this game is IRREVERSIBLE: most
    #: detours that shove a crate are rejected by `record_level`'s winnability
    #: probe, and each rejection costs a field sweep.
    _EPSILON = 0.08

    def epsilon_for(self, level: int) -> float:
        return self._EPSILON if level in self._EPSILON_LEVELS else 0.0

    def prepare_expert(self, game, expert) -> None:
        """Record every level's START state and sweep its field before
        `discover_solvable` asks for it -- the same work either way, but it fills
        the disk cache in one pass and makes the ~200 seconds visible as
        startup rather than as a mysteriously slow first episode."""
        for level in range(game.n_levels):
            game.set_level(level)
            _board, state = expert.read(game._engine)
            if state is not None:
                expert._starts.add(state)
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
    solver = TwoFacedSolver()
    game = solver.make_game(seed)
    expert = TwoFacedExpert(game, node_cap=solver.node_cap)
    for level in range(game.n_levels):
        game.set_level(level)
        _b, state = expert.read(game._engine)
        if state is not None:
            expert._starts.add(state)
    return solver, game, expert


def _ascii(board: _Board, state) -> str:
    """A model state as ASCII, so a failing check can print the board it failed
    on rather than a tuple of integers."""
    p1, p2, mh, mv = state
    out = []
    for r in range(board.h):
        line = ""
        for c in range(board.w):
            i = r * board.w + c
            th = (board.thmask >> i) & 1
            tv = (board.tvmask >> i) & 1
            if board.is_wall[i]:
                ch = "#"
            elif i in (p1, p2):
                vertical = (p2 - p1) == board.w
                ch = ("T" if i == p1 else "B") if vertical \
                    else ("L" if i == p1 else "R")
            elif (mh >> i) & 1:
                ch = "H" if th else ("x" if tv else "1")
            elif (mv >> i) & 1:
                ch = "V" if tv else ("x" if th else "2")
            elif th:
                ch = "O"
            elif tv:
                ch = "h"
            else:
                ch = "."
            line += ch
        out.append(line)
    return "\n".join(out)


def _start(game, expert, level: int):
    """``(board, state)`` at a level's start, engine left seated there."""
    game.set_level(level)
    return expert.read(game._engine)


def _plans(verbose: bool = True) -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the field and the tie labelling at
    once."""
    _solver, game, expert = _new()
    eng = game._engine
    total = ties = bad = 0
    for level in range(game.n_levels):
        board, (p1, p2, mh, mv) = _start(game, expert, level)
        t0 = time.time()
        plan = expert.plan(eng, level)
        dt = time.time() - t0
        game.set_level(level)
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
        if verbose:
            print(f"  L{level:2d}: {board.h:2d}x{board.w:2d}  "
                  f"{len(board.free):3d} floor  "
                  f"{bin(mh).count('1')}H+{bin(mv).count('1')}V crates  "
                  f"{len(plan):3d} presses  win={won}  "
                  f"(budget {game._max_steps}, {room})  "
                  f"{tie_steps:3d} steps with a tie set   [{dt:6.1f}s]")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _forward(board: _Board, state, limit: int) -> "int | None":
    """Presses to a win from ``state`` by a plain FORWARD BFS, or None past
    ``limit``.

    Shares nothing with `_Field` but `_Board.step` itself, which is the point: it
    is the independent answer every optimality claim below is checked against.
    Deliberately a plain dict/deque with no id arrays and no ONPATH pass, so a
    mistake in those cannot be reproduced here."""
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
            if nxt is None or nxt == cur or nxt in seen:
                continue
            if board.won(nxt):
                return d + 1
            seen.add(nxt)
            queue.append((nxt, d + 1))
    return None


def _reachable(board: _Board, state, cap: int):
    """The WHOLE reachable space from ``state`` and the number of presses the
    model REFUSED (`_Board.step` returning None) anywhere in it."""
    seen = {state}
    queue = deque([state])
    refused = 0
    while queue:
        cur = queue.popleft()
        for di in range(4):
            nxt = board.step(cur, di)
            if nxt is None:
                refused += 1
                continue
            if nxt != cur and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
        if len(seen) > cap:
            return seen, refused, True
    return seen, refused, False


#: Levels whose whole reachable space the model can enumerate in seconds (8.6k
#: to 109k states in the ball, plans of 21 to 58 presses). The other four are
#: 0.83M to 9.09M; ``--selfcheck --deep`` does them too.
_SMALL = (0, 1, 2, 3, 4)


def _selfcheck(walk_presses: int = 600, boards: int = 400, samples: int = 60,
               deep: bool = False, verbose: bool = True) -> int:
    """Six things the expert would otherwise be trusted on.

    1. **The native model is the interpreter's, along a walk.** A seeded random
       walk on every level, comparing `_Board.step`'s state against the engine's
       grid after EVERY press -- including the presses that do nothing, which is
       where a collision-layer mistake hides. The walk is what measures the
       mechanic listed in the module docstring: it pivots the body off crates,
       shoves runs of crates into walls, presses into walls (which cancels) and
       leans on the crate type the current press cannot move.
    2. **The native model is the interpreter's, from ARBITRARY boards.** A walk
       only reaches what a walk reaches. This pass SEATS randomly generated
       boards -- body and crates dropped on random free squares, in both
       orientations -- straight onto the engine and compares all four presses
       from each. It is what covers the jam cases a plan never produces: two
       crates in front of the two halves at once, a run of three, a crate flat
       against a wall with the other half free (the pivot that levels 6 and 7
       are built on).
    3. **The model never has to REFUSE a press.** `_Board.step` returns None for
       a crate shoved off the grid edge and for a body with one half off it. The
       claim in its docstring is that neither is reachable; this enumerates the
       whole reachable space of each level and counts. Anything but 0 is a hole
       in the model, not a curiosity.
    4. **The plans are shortest.** A plain forward BFS to the first win, which is
       shortest by construction and shares no line with `_Field`, must return the
       field's plan length on every level.
    5. **The tie sets are exact.** Every step's optimal set is re-derived by
       brute force -- a fresh depth-bounded forward BFS from each of the four
       successors -- and must match the field's answer exactly. Affordable on
       every step of the five small levels; on the four big ones the brute force
       is a full ball each and is only run with ``--deep``.
    6. **Recovery is answered from ANY board, not just the start.** This is the
       one that matters at generation time: the exploration prefix and the
       epsilon detours both leave the board off every shortest path. From
       ``samples`` states reached by seeded random presses the field must return
       a plan whose length an independent forward BFS agrees with and which WINS
       when replayed through the interpreter -- or say None, which in THIS game
       is frequently the true answer (a crate in a corner is a lost level) and is
       cross-checked against the same forward BFS rather than taken on trust.
    """
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    levels = list(range(game.n_levels))

    if verbose:
        print("1 -- model vs interpreter along a random walk")
    for level in levels:
        board, state = _start(game, expert, level)
        rng = random.Random(f"two_faced:walk:{level}")
        drift = refused = 0
        for _ in range(walk_presses):
            di = rng.randrange(4)
            eng.step(DIRS[di])
            pred = board.step(state, di)
            _b, live = expert.read(eng)
            if pred is None:
                refused += 1
                state = live
                continue
            if pred != live or board.won(pred) != eng.check_win():
                drift += 1
                print(f"    L{level} DIVERGED on {DIRS[di]} from\n"
                      f"{_ascii(board, state)}")
                break
            state = pred
            if eng.check_win():
                game.set_level(level)
                _b, state = expert.read(eng)
        bad += drift + refused
        if verbose:
            print(f"  L{level:2d}: model {'MATCHES' if not drift else 'DIVERGES from'}"
                  f" the interpreter over {walk_presses} random presses "
                  f"({refused} refused)")

    if verbose:
        print("2 -- model vs interpreter from randomly seated boards")
    for level in levels:
        board, state = _start(game, expert, level)
        game.set_level(level)
        static = _statics(eng, expert)
        rng = random.Random(f"two_faced:seat:{level}")
        cells = _component(board, state[0])
        n_h = bin(state[2]).count("1")
        n_v = bin(state[3]).count("1")
        drift = refused = tried = 0
        for _ in range(boards):
            cand = _random_state(board, cells, n_h, n_v, rng)
            if cand is None:
                continue
            for di in range(4):
                _seat(eng, static, expert, cand)
                eng.step(DIRS[di])
                _b, live = expert.read(eng)
                pred = board.step(cand, di)
                tried += 1
                if pred is None:
                    refused += 1
                    continue
                if pred != live or board.won(pred) != eng.check_win():
                    drift += 1
                    print(f"    L{level} seated board DIVERGED on {DIRS[di]}:\n"
                          f"{_ascii(board, cand)}")
                    break
            if drift:
                break
        bad += drift + refused
        if verbose:
            print(f"  L{level:2d}: {tried} presses from {boards} random boards -- "
                  f"{'MATCHES' if not drift else 'DIVERGES'} "
                  f"({refused} refused)")
        game.set_level(level)

    if verbose:
        print("3 -- the whole reachable space, and the refusal count")
    for level in (levels if deep else _SMALL):
        board, state = _start(game, expert, level)
        space, refused, capped = _reachable(board, state, 12_000_000)
        bad += refused + capped
        if verbose:
            print(f"  L{level:2d}: {len(space):8d} reachable states, "
                  f"{refused} refused press(es)"
                  + ("  [CAPPED]" if capped else ""))

    if verbose:
        print("4 -- plans are shortest (independent forward BFS)")
    for level in levels:
        board, state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        shortest = _forward(board, state, len(plan) + 1 if plan else 200)
        ok = plan is not None and shortest == len(plan)
        bad += not ok
        if verbose:
            print(f"  L{level:2d}: field {len(plan) if plan else None} presses vs "
                  f"forward BFS {shortest} -- "
                  f"{'SHORTEST' if ok else 'NOT SHORTEST'}")

    if verbose:
        print("5 -- tie sets re-derived by brute force")
    for level in (levels if deep else _SMALL):
        board, state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        sets = getattr(plan, "optsets", None) or []
        mismatched = 0
        cur = state
        for i, direction in enumerate(plan):
            rest = len(plan) - i
            truth = []
            for di in range(4):
                nxt = board.step(cur, di)
                if nxt is None or nxt == cur:
                    continue
                if board.won(nxt):
                    if rest == 1:
                        truth.append(DIRS[di])
                    continue
                if _forward(board, nxt, rest - 1) == rest - 1:
                    truth.append(DIRS[di])
            if sorted(truth) != sorted(sets[i]):
                mismatched += 1
                print(f"    L{level} step {i}: brute {sorted(truth)} vs "
                      f"field {sorted(sets[i])}")
            cur = board.step(cur, DIRS.index(direction))
        bad += mismatched
        if verbose:
            print(f"  L{level:2d}: {len(plan)} tie sets "
                  f"{'match' if not mismatched else f'{mismatched} MISMATCH'}"
                  f" brute force")

    if verbose:
        print("6 -- recovery from off-plan boards")
    for level in _SMALL:
        rng = random.Random(f"two_faced:recovery:{level}")
        wrong = dead = won_after = 0
        for _ in range(samples):
            game.set_level(level)
            for _ in range(rng.randrange(1, 20)):
                eng.step(DIRS[rng.randrange(4)])
                if eng.check_win():
                    break
            if eng.check_win():
                continue                 # the walk already won; nothing to plan
            board, state = expert.read(eng)
            plan = expert._search(eng)
            truth = _forward(board, state, board.n * 4)
            if plan is None:
                dead += 1
                if truth is not None:
                    wrong += 1
                    print(f"    L{level} NO PLAN but forward BFS says {truth}:\n"
                          f"{_ascii(board, state)}")
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
                  f"re-planned to a WIN in the interpreter, {dead} genuinely "
                  f"dead, {wrong} WRONG")
    return bad


def _component(board: _Board, cell: int) -> list:
    """The non-wall cells 4-connected to ``cell``.

    Several levels have decorative Background pockets OUTSIDE their wall ring
    (level 1 has four of them) which are free squares the play area can never
    reach -- and which touch the grid EDGE, where `_Board.step` legitimately
    refuses to guess. `_random_state` seats boards inside this component only,
    so the seated-board fuzz stays inside the game rather than manufacturing
    positions no level can hold."""
    seen = {cell}
    queue = deque([cell])
    while queue:
        cur = queue.popleft()
        for di in range(4):
            nxt = board.nb[cur][di]
            if nxt >= 0 and not board.is_wall[nxt] and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return sorted(seen)


def _random_state(board: _Board, cells, n_h: int, n_v: int,
                  rng: random.Random):
    """A random legal board: a domino on two adjacent free squares plus ``n_h``
    CrateH and ``n_v`` CrateV on other free squares.

    ``cells`` is the play area (see `_component`). Legal in the sense the ENGINE
    means it -- the player and the crates share no square (they are on different
    collision layers, but no rule in this game can ever put one on the other,
    which pass 2 is partly there to confirm) and nothing sits in a wall. Not
    necessarily REACHABLE, which is the point: the seated-board pass is there to
    reach the shapes a real trajectory does not."""
    pool = set(cells)
    for _attempt in range(40):
        p1 = rng.choice(cells)
        opts = [q for q in (p1 + 1, p1 + board.w)
                if q in pool
                and (q != p1 + 1 or q // board.w == p1 // board.w)]
        if not opts:
            continue
        p2 = rng.choice(opts)
        rest = [i for i in cells if i != p1 and i != p2]
        if len(rest) < n_h + n_v:
            continue
        picked = rng.sample(rest, n_h + n_v)
        mh = mv = 0
        for i in picked[:n_h]:
            mh |= 1 << i
        for i in picked[n_h:]:
            mv |= 1 << i
        return (p1, p2, mh, mv)
    return None


# ---------------------------------------------------------------------------
# Seating: model state <-> engine grid
# ---------------------------------------------------------------------------

def _statics(eng, expert):
    """The level's static layer -- every cell with the player, the crates and the
    crate-on-target markers stripped out -- so a state can be SEATED back onto
    the engine instead of keeping a snapshot of every board it reaches.

    Taken from the live grid rather than assembled from object ids, so it carries
    whatever the level actually holds (the BG01/BG02 floor pattern the level-start
    rules flood in, and the auxBG that stops them flooding again) and a seated
    board is byte-identical to the one the level loaded."""
    g = expert.g
    markers = set()
    for name in ("cratewh", "cratewv", "cratenwh", "cratenwv"):
        markers |= set(g.resolve_object_name(name))
    dynamic = expert.player_ids | expert.crateh_ids | expert.cratev_ids | markers
    return [[cell - dynamic for cell in row] for row in eng.grid]


def _seat(eng, static, expert, state) -> None:
    """Put ``state`` back on the engine.

    The crate-on-target markers are NOT placed: they are late-rule decoration and
    the first press re-derives them (``late [crateH TargetH no crateWH] -> ...``),
    so seating them would be seating an output. That it does not matter -- that a
    seated board steps exactly like a snapshot-restored one -- is checked in
    `_engine` before any of it is trusted."""
    p1, p2, mh, mv = state
    w = len(static[0])
    crateh = next(iter(expert.crateh_ids))
    cratev = next(iter(expert.cratev_ids))
    vertical = (p2 - p1) == w
    ids = expert.g.obj_name_to_idx
    first = ids["playert"] if vertical else ids["playerl"]
    second = ids["playerb"] if vertical else ids["playerr"]
    grid = []
    for r, row in enumerate(static):
        out = []
        for c, cell in enumerate(row):
            i = r * w + c
            new = set(cell)
            if (mh >> i) & 1:
                new.add(crateh)
            if (mv >> i) & 1:
                new.add(cratev)
            if i == p1:
                new.add(first)
            elif i == p2:
                new.add(second)
            out.append(new)
        grid.append(out)
    eng.grid = grid
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


def _read_state(eng, expert):
    """``(p1, p2, crateH mask, crateV mask)`` straight off the engine grid. A
    grid scan and nothing else -- no `_Board`, no mechanic -- so the sweeps in
    `_engine` owe the native model nothing."""
    w = eng.width
    mh = mv = 0
    halves = []
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            i = r * w + c
            if cell & expert.crateh_ids:
                mh |= 1 << i
            if cell & expert.cratev_ids:
                mv |= 1 << i
            if cell & expert.player_ids:
                halves.append(i)
    halves.sort()
    if len(halves) != 2:
        return None
    return (halves[0], halves[1], mh, mv)


#: Levels whose ball to ``d*`` the INTERPRETER can be walked over in minutes at
#: its measured 269 presses a second. The others are 58k to 9.09M states, i.e.
#: hours to never; pass level numbers on the command line to try them anyway.
_ENGINE_LEVELS = (0, 1, 2)


def _engine(levels=_ENGINE_LEVELS, verbose: bool = True) -> int:
    """The proof that owes nothing to the native model: the same two sweeps,
    driven by the REAL INTERPRETER.

    Forward BFS from the level start by pressing actual buttons, keying each
    board by what is on the grid, stopping at the depth the first `check_win`
    appears -- which fixes the shortest length by construction -- with every edge
    recorded; then a backward sweep from the won boards over those edges, which
    gives the exact optimal SET at every state on a shortest path. No `_Board`
    call of any kind participates: successors come from `PSEngine.step` and the
    goal test is `check_win`.

    States are re-seated from their keys rather than snapshotted, which is what
    keeps tens of thousands of boards in megabytes; the seating is checked to be
    the exact inverse of the reading, and a seated board is checked to step
    identically to a snapshot-restored one, before any of it is trusted."""
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in levels:
        _board, _state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        game.set_level(level)
        static = _statics(eng, expert)
        start = _read_state(eng, expert)

        # The decoder is the inverse of the encoder, and a seated board behaves
        # like a restored one -- both checked here rather than assumed, because a
        # decoder that is not exactly the inverse enumerates a DIFFERENT game.
        original = snapshot(eng)
        for di in range(4):
            restore(eng, original)
            eng.step(DIRS[di])
            expected = snapshot(eng)
            _seat(eng, static, expert, start)
            eng.step(DIRS[di])
            if eng.grid != expected:
                print(f"  L{level:2d}: a seated board steps differently "
                      f"({DIRS[di]})")
                bad += 1
        _seat(eng, static, expert, start)
        if _read_state(eng, expert) != start:
            print(f"  L{level:2d}: SEATING IS NOT THE INVERSE OF READING")
            bad += 1
            continue

        t0 = time.time()
        depth = {start: 0}
        queue = deque([start])
        rev: dict = {}
        won_states: set = set()
        d_star = None
        while queue:
            cur = queue.popleft()
            d = depth[cur]
            if d_star is not None and d >= d_star:
                continue
            for di in range(4):
                _seat(eng, static, expert, cur)
                eng.step(DIRS[di])
                nxt = _read_state(eng, expert)
                if nxt is None or nxt == cur:
                    continue                    # a press that did nothing
                rev.setdefault(nxt, []).append(cur)
                if nxt in depth:
                    continue
                depth[nxt] = d + 1
                if eng.check_win():
                    won_states.add(nxt)
                    if d_star is None:
                        d_star = d + 1
                    continue                    # a won board is terminal
                queue.append(nxt)
        if d_star is None:
            print(f"  L{level:2d}: the interpreter found no win")
            bad += 1
            continue

        dist = {s: 0 for s in won_states}
        back = deque(won_states)
        while back:
            cur = back.popleft()
            for prev in rev.get(cur, ()):
                if prev not in dist:
                    dist[prev] = dist[cur] + 1
                    back.append(prev)

        epresses, esets = [], []
        cur = start
        while dist.get(cur):
            rest = dist[cur]
            best = []
            for di in range(4):
                _seat(eng, static, expert, cur)
                eng.step(DIRS[di])
                nxt = _read_state(eng, expert)
                if nxt is not None and nxt != cur and dist.get(nxt) == rest - 1:
                    best.append((di, nxt))
            epresses.append(DIRS[best[0][0]])
            esets.append(sorted(DIRS[di] for di, _n in best))
            cur = best[0][1]

        same_len = len(epresses) == len(plan) == d_star
        same_sets = esets == [sorted(s) for s in plan.optsets]
        bad += (not same_len) + (not same_sets)
        if verbose:
            print(f"  L{level:2d}: {len(depth):7d} engine boards within d*, "
                  f"{d_star:3d} presses vs the field's {len(plan):3d} -- "
                  f"{'AGREE' if same_len else 'DISAGREE'}; tie sets "
                  f"{'AGREE' if same_sets else 'DISAGREE'}  "
                  f"[{time.time() - t0:.0f}s]")
        game.set_level(level)
    return bad


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

#: Every cell stack a board of this game can hold.
#:
#: The floor is Background PLUS BG01 (or BG02): the level-start rules flood an
#: alternating BG01/BG02 checkerboard over every non-wall square, and since both
#: objects' colours collapse to the same ARC palette index the two are one flat
#: floor tile. The bare-Background composition is the OUTSIDE -- the decorative
#: pockets several levels have beyond their wall ring, which the flood cannot
#: reach because it will not cross a Wall.
#:
#: A crate on a target is always drawn with its marker, because the late rules
#: put one there unconditionally: CrateWH/CrateWV for a crate on its OWN kind of
#: target (the win condition) and CrateNWH/CrateNWV for one on the wrong kind.
#:
#: ``wall + target`` and ``player + crate`` are absent on purpose. No legend
#: character places a target under a wall and no level does; and no rule in the
#: game can put a player half on a crate -- a press that would need to only
#: happens when the crate MOVES, and when it cannot the half is jammed instead.
#: ``--selfcheck`` pass 2 is partly there to confirm that second claim.
_COMPOSITIONS = [
    ("wall",), ("bg01",), (),
    ("bg01", "targeth"), ("bg01", "targetv"),
    ("bg01", "playert"), ("bg01", "playert", "targeth"),
    ("bg01", "playert", "targetv"),
    ("bg01", "playerb"), ("bg01", "playerb", "targeth"),
    ("bg01", "playerb", "targetv"),
    ("bg01", "playerl"), ("bg01", "playerl", "targeth"),
    ("bg01", "playerl", "targetv"),
    ("bg01", "playerr"), ("bg01", "playerr", "targeth"),
    ("bg01", "playerr", "targetv"),
    ("bg01", "crateh"), ("bg01", "cratev"),
    ("bg01", "crateh", "targeth", "cratewh"),
    ("bg01", "cratev", "targetv", "cratewv"),
    ("bg01", "cratev", "targeth", "cratenwh"),
    ("bg01", "crateh", "targetv", "cratenwv"),
]


def _comp_name(comp) -> str:
    if not comp:
        return "outside"
    return "+".join(c for c in comp if c != "bg01") or "floor"


def _audit(verbose: bool = True) -> int:
    """Two passes over every reachable cell COMPOSITION.

    **Pass 1 -- distinctness**, at every board shape the nine levels use. Whole
    64x64 frames of uniform boards are compared rather than one cell out of a
    mixed board: `_render_frame` upscales and centre-pads, so slicing a cell by
    ``cell_px`` arithmetic reads the wrong pixels (the ps:explod lesson). Two
    uniform boards render identically iff their cells do.

    This is the pass that fails 114 times on the shipped .txt -- the player over
    every target at every size, and on level 1 (4 px per cell) the crate-on-
    target markers over each other and over a bare crate, i.e. the win condition.
    See the header comment in ``data/puzzlescript_games/Two-faced.txt``.

    **Pass 2 -- the group.** The game is augmented with a frame rotation AND both
    flips, so a board is drawn at any of 16 presentations. This lists every
    composition whose transform is another composition's art. For this game that
    list is not empty and is not a fault: the anti-diagonal reflection carries
    PlayerT onto PlayerR and PlayerB onto PlayerL (and back), which is exactly
    the relabelling the mechanic makes -- that reflection turns a vertical body
    into a horizontal one and sends its top half to the right. A board holds
    EITHER a T/B pair or an L/R pair and never one of each, so no frame is
    ambiguous; what the pass has to show is that the orbit is CLOSED (nothing
    lands on a target, a crate or the floor) and that the static furniture --
    wall, floor, outside and both bare targets -- is group-INVARIANT, so a turned
    board cannot rename a goal. It runs on SQUARE boards, because a transform of
    a non-square board also moves the letterbox and every comparison would pass
    for the wrong reason.
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
        bad += len(clashes)
        if verbose:
            print(f"  {h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): {len(shots)} "
                  f"compositions, {'OK' if not clashes else 'CLASH'}")
        for a, b in clashes:
            print(f"      IDENTICAL: {_comp_name(a)}  ==  {_comp_name(b)}")

    def transforms(frame):
        for k in range(4):
            turned = np.rot90(frame, k=k)
            for hf in (False, True):
                for vf in (False, True):
                    out = turned
                    if hf:
                        out = np.fliplr(out)
                    if vf:
                        out = np.flipud(out)
                    yield (k, hf, vf), np.ascontiguousarray(out)

    #: The compositions that may map onto each other under the group, and the
    #: only ones: the four player halves (with or without a target under them),
    #: whose orbit IS the T->R / B->L relabelling the anti-diagonal reflection
    #: makes. Anything outside this set landing on anything else is a fault.
    def _players_only(comp):
        return any(c.startswith("player") for c in comp)

    for cell in sorted({min(64 // h, 64 // w) for h, w in sizes}):
        n = 64 // cell
        shots = {c: shoot(n, n, c) for c in _COMPOSITIONS}
        pairs = [(a, b, t)
                 for a, b in itertools.permutations(_COMPOSITIONS, 2)
                 for t, turned in transforms(shots[a])
                 if np.array_equal(turned, shots[b])]
        stray = [(a, b, t) for a, b, t in pairs
                 if not (_players_only(a) and _players_only(b)
                         and [c for c in a if not c.startswith("player")]
                         == [c for c in b if not c.startswith("player")])]
        invariant = [_comp_name(c) for c in _COMPOSITIONS
                     if all(np.array_equal(t, shots[c])
                            for _k, t in transforms(shots[c]))]
        bad += len(stray)
        if verbose:
            print(f"  cell {cell}px ({n}x{n}): {len(pairs)} composition pair(s) "
                  f"in one group orbit ({len(stray)} of them STRAY); "
                  f"group-invariant: {', '.join(invariant)}")
        for a, b, t in stray:
            print(f"      STRAY: {_comp_name(a)} under {t} == {_comp_name(b)}")
    return bad


def _symmetry(walk_presses: int = 200, verbose: bool = True) -> int:
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unaugmented ones.

    This is the evidence for the rotation + flip augmentation. The structural
    argument is in `PuzzleScriptAdapter._FLIP_GAMES`; Gobble Rush's chirality
    hid inside exactly that kind of argument, so it is measured.

    Both the PLANS and a seeded random walk are replayed: the walk reaches the
    boards a plan never visits (crates flat against walls, the body pivoting off
    a jam, both halves jammed at once) and it presses the unbound ACTION key as
    well."""
    solver, game, expert = _new()
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
        if len(seen) == 16 and seed > 80:
            break
        g = solver.make_game(seed)
        for level in range(g.n_levels):
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            rng = random.Random(f"two_faced:symmetry:{level}")
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
        violations = _selfcheck(deep="--deep" in sys.argv)
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--engine" in sys.argv:
        args = [int(a) for a in sys.argv[sys.argv.index("--engine") + 1:]
                if a.isdigit()]
        violations = _engine(args or _ENGINE_LEVELS)
        print(f"engine sweeps: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = _audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = _symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    sys.exit(TwoFacedSolver.main())
