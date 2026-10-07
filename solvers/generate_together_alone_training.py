"""Generate Phase-1 training data for the PuzzleScript game ps:together_alone
("Together Alone", Qwok Games).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanic, a native model of it, the exact distance FIELD that model is solved
with, the proof that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_together_alone",
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
Two characters on one board, ONE of which is active at a time. **The X button
(ACTION5) is the character switch and it is a real move** -- unlike most of this
family, where ACTION is dead. Ben is brown, Isabelle is blue, and the win is
three conditions at once:

    No BreakableTile          (every brown and every blue tile destroyed)
    All Ben on BrownExit
    All Isabelle on BlueExit

**A tile is destroyed by WALKING OFF IT, not onto it**, and only by its own
colour's character: the rules that move a player name ``TileWithBrown`` /
``TileWithBlue`` in the cell being VACATED and leave the right-hand side of that
cell empty. So the puzzle is a covering problem with a twist -- Ben must be the
last thing to leave every brown tile, and he must still have somewhere to step.

One press, completely (``d`` the pressed direction, ``p`` the active character's
cell, ``q = p + d``, ``s = p + 2d``):

  * ACTION: the two characters swap which is active. Nothing else happens, and
    it always fires (both characters exist for the whole game).
  * ``q`` off the board, or ``q`` has no Ground and the jump below does not
    apply: nothing happens at all. Ground is any Tile (brown, blue, two-colour,
    gray) or either Exit -- everything else is the void, which is a hole.
  * ``q`` has Ground: the active character steps onto it.
  * ``q`` is the VOID, the active character is BEN, and ``s`` has Ground: Ben
    JUMPS the one-square gap and lands on ``s``. Isabelle cannot jump: the three
    jump rules name Ben and nothing else. (The gap rule reads ``No Tile`` rather
    than "no Ground", but the plain step rule is listed first and an Exit is
    Ground, so a cell holding an Exit is stepped onto, never jumped over.)

and, whichever of the three ways the character left ``p``, the cell it VACATED
resolves as:

  * ``p`` holds a tile of the mover's own colour and the OTHER character is not
    standing there: the tile is DESTROYED (this is the whole game);
  * ``p`` holds a tile of the mover's own colour and the other character IS
    standing there: the tile FLIPS to the other colour instead -- Ben leaving a
    shared brown tile turns it blue, Isabelle leaving a shared blue tile turns
    it brown. The author's comment says this stands in for the original game's
    "the tile cracks and breaks later"; in ARC terms it means **a shared tile is
    a tile whose owner you get to choose**, and the level-4 message ("A tile
    won't break with a character standing on it") is pointing at it.
  * otherwise the cell is left exactly as it was -- Ben walking off a blue tile,
    a gray tile or an Exit changes nothing.

The inactive character never moves: two rules cancel its force before any
movement rule is reached. Ben and Isabelle are on different collision layers, so
they can share a cell freely, and the trailing ``[ > Player ] -> [ Player ]``
means the movement phase itself moves nobody -- every step in this game is a
rule firing, which is why a press into a hole is a true no-op.

TwoColTile (the ``+`` legend, "broken by both") is modelled -- Ben leaving one
destroys it outright, Isabelle leaving one destroys it, and a shared one flips
to whichever single colour the mover is not -- but **no shipped level places
one**, which `--fuzz` asserts rather than assumes.

Expert solver
-------------
A native model of the mechanic above (`_Board`) plus an exact distance-to-win
field (`_Field`) built by ENUMERATING the whole reachable space. Measured, not
assumed: the four levels close at 767,732 / 1,654 / 764 / 7,856 states, about
3.3 s and 340 MB for the whole game, so there is nothing to trade away -- the
plans come out provably shortest, the optimal-action SETS exact, and "can this
board still be won" is a dict lookup.

That last one is not a luxury here. **This game is directed and most of its
boards are already lost**: breaking a tile is irreversible, and 711,076 of level
0's 767,732 reachable boards (92.6%) can never reach a win. Two ways to strand
yourself, both common under random play -- destroy the tile you were going to
come back across and cut a region off, or step off the last tile of your own
colour while standing somewhere you cannot leave without wasting the exit.
Everything downstream (the RESET prefix, the checked epsilon detours, the choice
of a forward closure over a backward sweep) follows from that one fact.

The field is:

  * one forward closure from the state it is asked about, over `_Board.step`,
    with a won board treated as TERMINAL (a win ends the episode, so no shortest
    path continues through one);
  * then a backward BFS from the won boards over that closure's own reversed
    edges, which labels every state in it with the true shortest presses to a
    win -- and leaves the rest unlabelled, which is a PROOF of deadness rather
    than a search giving up.

The first query is the level's start, whose closure holds every board the
recording can subsequently reach, so everything after it is a lookup: the plan
is "walk down the gradient", the optimal set at a state is every press whose
successor is one nearer, recovery is the same lookup from wherever the board
happens to be, and the deadness test that makes epsilon detours safe is that
lookup returning nothing.

A backward sweep from the enumerated win set -- the shape ps:swap_sokoban uses
-- is not available: the predecessor of "a tile was destroyed" is "a tile was
created", which no rule does, so the goal set's backward ball is full of boards
no play can produce. Forward closure is the only cheap direction in a game that
only ever removes things.

Boards are keyed by their STATIC layout -- dimensions, gray tiles and the two
exits, none of which any rule touches -- and the four levels' signatures are
mutually distinct (`_Board` asserts it), so the model identifies which level the
engine is showing from the grid alone. That matters because a mid-game grid has
FEWER tiles on it than the level start: a board built from "whichever cells hold
a tile right now" would be a different board for every state, and every detour
would re-enumerate. The tile SLOTS come from `PSGame.levels`, the parsed start
maps, which no play mutates.

Shortest, and how that is known
-------------------------------
Four independent derivations, agreeing on every level:

  * this file's closure field;
  * a plain forward BFS to the first win over the same model, shortest by
    construction and sharing no line with the field, plus a brute-force
    re-derivation of every step's tie set from depth-bounded searches out of
    each of the five successors (``--selfcheck`` passes 3 and 4);
  * the same two sweeps driven by the REAL INTERPRETER (``--engine``): forward
    BFS from the level start by pressing actual buttons until a win appears,
    edges recorded, then backward over those edges. No `_Board` call takes part
    -- a successor is whatever `PSEngine.step` makes of the board and the goal
    test is `check_win`;
  * the shared `StateGraph.build` walking the interpreter's WHOLE reachable
    space (``--enumerate``), which additionally re-derives the count of boards
    that cannot win -- the claim the detours rest on.

The four plans come to 76 presses (19 + 16 + 16 + 25), of which 13 stand at a
step with a second equally right answer.

The last two derivations DEFAULT to levels 1-3, because level 0's ball is
315,265 interpreter boards and its whole space 767,732. All of it has been run:
``--engine 0`` (~5 min) agrees on 19 presses and on every tie set over the
315,265-board ball, and ``--enumerate 0`` (~25 min, 3,838,650 real presses)
agrees on the length, on the tie sets, on the space size and on all 711,076
boards that cannot win. ``--fuzz 0`` (same cost) presses every button at every
one of the 767,732 states and finds no disagreement with the model at all. They
are not in the default set because a report nobody waits for is a report nobody
runs; both name what they skipped rather than quietly covering three levels and
printing a clean line.

Recovery
--------
`supports_recovery` with the family's RESET-mode prefix, PLUS ``epsilon = 0.12``
detours inside the expert replay. The expert re-plans from the LIVE board (a
field lookup at whatever state the engine is in), so a detour is answered
exactly: the taken action is the mistake and ``optimal`` is the recovery.

The division of labour between the two is forced by the irreversibility:

  * a DETOUR is offered a random alternative press and taken only if the field
    still gives the resulting board a distance. That test is exact -- the field
    enumerates, so "no distance" means "provably cannot win" -- and it is O(1),
    so it is affordable at every step. A detour therefore never bricks a board
    and the replay always finishes;
  * the PREFIX is not offered that choice. It explores freely, and on this game
    that strands it often (``--selfcheck``'s recovery pass walks 1-20 random
    presses from a level start and finds the board already lost on 12, 38, 2 and
    38 of 80 tries on the four levels), so it ends in ONE RESET back to the level
    start, from which the cached plan is a guaranteed win. Exactly the human arc:
    flail, get stuck, hit reset, then solve.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the frame rotation
(``rotation_k`` in 0..3) and the two flips, with their matching directional
action remap. 4 levels x 16 presentations = 64.

``Together_Alone`` is in `PuzzleScriptAdapter._FLIP_GAMES`; the argument is
recorded there and ``--symmetry`` measures it -- every level's plan AND a seeded
200-press random walk (which breaks tiles, strands both characters, parks them
on the same square and presses the switch) replayed at all 16 presentations,
requiring every frame to be the exact transform of the unaugmented one.

Rendering
---------
The shipped art was unusable and every sprite had to be repainted; the full
statement is the header comment in
``data/puzzlescript_games/Together_Alone.txt``. In one line: the characters and
the active-player marker between them covered all 25 sprite pixels, so the
ground under a character -- which is two thirds of the win condition -- was
invisible, ``ben+isabelle`` drew exactly as ``isabelle``, and BenActive and
IsaActive were the same white bar. 43 cell compositions produced 174 identical
pairs. Each collision layer now owns a disjoint region of the sprite (ground =
the top and bottom bands, Ben = the left block, the marker = the centre bar,
Isabelle = the right block) and Background was moved off palette 5, which is
also the letterbox pad.

``--audit`` renders every cell composition a board of this game can hold as a
whole 64x64 frame of a uniform board and requires them pairwise distinct, none
equal to the letterbox, and no transform of one equal to another. Whole frames
rather than one cell sliced out of a mixed board: `_render_frame` upscales and
centre-pads, so slicing by ``cell_px`` arithmetic reads the wrong pixels (the
ps:explod lesson).

Usage (run from the repo root):
    python solvers/generate_together_alone_training.py --episodes 200 \
        --out data/training_multi_level/together_alone

    python solvers/generate_together_alone_training.py --plans
    python solvers/generate_together_alone_training.py --selfcheck
    python solvers/generate_together_alone_training.py --fuzz
    python solvers/generate_together_alone_training.py --engine
    python solvers/generate_together_alone_training.py --enumerate
    python solvers/generate_together_alone_training.py --audit
    python solvers/generate_together_alone_training.py --symmetry
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
                                     screen_action, snapshot, restore)
from utils.rotation import inverse_remap_action_full                  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Together_Alone"

#: Engine presses, in the order ties are broken. ACTION is IN the list and is
#: not padding: it is the character switch, the only way to move the other
#: body, and several levels' shortest plans spend a quarter of their presses on
#: it. Dropping it would make every level unwinnable.
DIRS: tuple[str, ...] = ("up", "down", "left", "right", "action")

#: Index of ACTION inside `DIRS`, used everywhere a loop wants "the four
#: directions" or "the switch" specifically.
SWITCH = 4

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

# -- cell kinds (STATIC: no rule in this game changes any of them) ------------
#: The void. A hole: nobody can stand on it, and Ben's jump crosses exactly one.
VOID = 0
#: GrayTile. Ground, and unbreakable -- it is not a `BreakableTile`, so the win
#: condition ignores it and leaving one destroys nothing.
GRAY = 1
#: BrownExit -- where Ben has to finish. Ground, never breakable.
BEXIT = 2
#: BlueExit -- where Isabelle has to finish.
IEXIT = 3
#: A cell that holds a breakable tile at the level start. Its tile is the only
#: per-cell state this game has.
SLOT = 4

# -- tile values, the 2 bits a SLOT carries ----------------------------------
GONE, BROWN, BLUE, TWOCOL = 0, 1, 2, 3

#: A press's outcome, as the model's transition function classifies it. Kept as
#: a returned label rather than reconstructed from a before/after diff, because
#: a diff cannot tell a walk into a wall from a switch that changed nothing --
#: both are "the board did not move" -- and a fuzz whose counters are guessed
#: that way reports clean coverage of branches it never reached
#: (the ps:silver_lungs lesson).
SWITCHED, BLOCKED, WALK, BREAK, FLIP, JUMP, JUMP_BREAK, JUMP_FLIP = range(8)
BRANCHES = ("switch", "blocked", "step+walk", "step+break", "step+flip",
            "jump+walk", "jump+break", "jump+flip")

#: Disk cache of every level's start plan AND its optimal-action sets. The whole
#: game's fields take about three seconds, so this is not the load-bearing cache
#: it is on the big ps: sokobans -- it is here so a `parallelize_generator`
#: fan-out shares one derivation rather than repeating it on every core. Delete
#: the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "together_alone_plans.json"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic.

    Cells are flat ``r * w + c`` indices. A STATE is packed into a single Python
    int, which is what lets the field key three quarters of a million of them
    without the memory going anywhere interesting:

        bits 0 .. 2*nslots-1   two bits per breakable SLOT, in slot order:
                               GONE / BROWN / BLUE / TWOCOL
        the next ``pb`` bits   Ben's cell
        the next ``pb`` bits   Isabelle's cell
        the next bit           0 = Ben is active, 1 = Isabelle is active

    ``kind`` never changes -- the void, the gray tiles and the two exits are
    untouched by every rule in the game -- so it lives here rather than in the
    state, and so does the SLOT set, which comes from the level's parsed start
    map. That last point is the one that matters: a mid-game grid holds fewer
    tiles than the start, so a board inferred from "the cells that hold a tile
    right now" would be a different board at every state and the field would
    re-enumerate on every detour. Instead the board is a pure function of the
    static layout, and `signature` identifies which level a live grid is
    showing.
    """

    __slots__ = ("h", "w", "n", "kind", "slots", "slot_of", "nslots", "pb",
                 "SB", "SI", "SA", "MP", "TILEMASK", "ground_static",
                 "edge", "edge2", "sig")

    def __init__(self, h: int, w: int, kind, slots):
        self.h, self.w = h, w
        self.n = h * w
        self.kind = tuple(kind)
        #: Cells that hold a breakable tile at the level start, in scan order.
        self.slots = tuple(slots)
        self.nslots = len(self.slots)
        #: cell -> bit offset of its tile, or -1 for a cell that is not a slot.
        off = {c: 2 * i for i, c in enumerate(self.slots)}
        self.slot_of = tuple(off.get(i, -1) for i in range(self.n))
        self.pb = max(1, (self.n - 1).bit_length())
        self.SB = 2 * self.nslots
        self.SI = self.SB + self.pb
        self.SA = self.SI + self.pb
        self.MP = (1 << self.pb) - 1
        self.TILEMASK = (1 << self.SB) - 1
        #: True for the cells that are Ground no matter what the state is --
        #: gray tiles and the two exits. A slot is Ground iff its tile is not
        #: GONE, which is the only thing `_ground` has to look at.
        self.ground_static = tuple(k in (GRAY, BEXIT, IEXIT) for k in kind)
        #: ``edge[cell][di]`` / ``edge2[cell][di]``: one and two steps along
        #: ``DIRS[di]``, or -1 off the board. Walls are NOT folded in here (a
        #: tile's presence is state, not geometry), which is the difference from
        #: the sokoban-shaped boards in this family.
        self.edge, self.edge2 = [], []
        for i in range(self.n):
            r, c = divmod(i, w)
            one, two = [], []
            for d in DIRS[:SWITCH]:
                dr, dc = _DELTA[d]
                nr, nc = r + dr, c + dc
                one.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1)
                nr, nc = r + 2 * dr, c + 2 * dc
                two.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1)
            self.edge.append(tuple(one))
            self.edge2.append(tuple(two))
        self.sig = self.signature(
            h, w,
            [i for i, k in enumerate(kind) if k == GRAY],
            [i for i, k in enumerate(kind) if k == BEXIT],
            [i for i, k in enumerate(kind) if k == IEXIT])

    @staticmethod
    def signature(h: int, w: int, gray, bexit, iexit):
        """The STATIC identity of a level: its shape and the cells no rule can
        ever change. Two levels with the same signature would be
        indistinguishable to `TogetherAloneExpert.read`, which is why `setup`
        asserts the four are distinct.

        Taken as three ascending cell lists rather than as a ``kind`` array,
        because `read` is on the hot path of the interpreter sweeps (millions of
        calls) and collects them during its single grid scan; rebuilding a
        56-entry array and scanning it three more times per press was the
        report's dominant cost."""
        return (h, w, tuple(gray), tuple(bexit), tuple(iexit))

    # -- packing -------------------------------------------------------------
    def pack(self, ben: int, isa: int, active: int, tiles) -> int:
        """``tiles`` maps a slot CELL to its value; absent means GONE."""
        s = 0
        for i, cell in enumerate(self.slots):
            s |= tiles.get(cell, GONE) << (2 * i)
        return s | (ben << self.SB) | (isa << self.SI) | (active << self.SA)

    def unpack(self, s: int):
        """``(ben, isa, active, {cell: value} for the tiles still standing)``."""
        tiles = {cell: (s >> (2 * i)) & 3
                 for i, cell in enumerate(self.slots)
                 if (s >> (2 * i)) & 3}
        return ((s >> self.SB) & self.MP, (s >> self.SI) & self.MP,
                s >> self.SA, tiles)

    def _ground(self, s: int, cell: int) -> bool:
        if self.ground_static[cell]:
            return True
        off = self.slot_of[cell]
        return off >= 0 and ((s >> off) & 3) != GONE

    # -- the mechanic --------------------------------------------------------
    def move(self, s: int, di: int):
        """``(branch, next state)`` for pressing ``DIRS[di]`` at ``s``.

        The single transition function. `step` is this with the label dropped,
        and the label is RETURNED rather than reconstructed from a before/after
        diff, so the fuzz's coverage counters and the field's edges can never
        come from two different pieces of code (see `BRANCHES`)."""
        if di == SWITCH:
            return SWITCHED, s ^ (1 << self.SA)
        active = s >> self.SA
        ben = (s >> self.SB) & self.MP
        isa = (s >> self.SI) & self.MP
        p = isa if active else ben
        q = self.edge[p][di]
        if q < 0:
            return BLOCKED, s                    # off the board
        jumped = False
        if self._ground(s, q):
            dest = q
        else:
            # The void in front. Only Ben jumps it, and only onto Ground.
            if active:
                return BLOCKED, s
            t = self.edge2[p][di]
            if t < 0 or not self._ground(s, t):
                return BLOCKED, s
            dest, jumped = t, True

        # The cell being VACATED. This is where the game happens: a tile of the
        # mover's own colour is destroyed, unless the other character is
        # standing on it, in which case it flips to the other colour instead.
        ns = s
        branch = JUMP if jumped else WALK
        off = self.slot_of[p]
        if off >= 0:
            val = (s >> off) & 3
            mine = (val in (BROWN, TWOCOL)) if active == 0 else (val in (BLUE, TWOCOL))
            if mine:
                shared = (ben == isa)
                new = (BLUE if active == 0 else BROWN) if shared else GONE
                ns = (s & ~(3 << off)) | (new << off)
                branch = ((JUMP_FLIP if jumped else FLIP) if shared
                          else (JUMP_BREAK if jumped else BREAK))
        if active:
            return branch, (ns & ~(self.MP << self.SI)) | (dest << self.SI)
        return branch, (ns & ~(self.MP << self.SB)) | (dest << self.SB)

    def step(self, s: int, di: int) -> int:
        return self.move(s, di)[1]

    def won(self, s: int) -> bool:
        """The win condition, verbatim: ``No BreakableTile`` (every slot GONE --
        gray tiles are not breakable and are not counted), ``All Ben on
        BrownExit``, ``All Isabelle on BlueExit``."""
        if s & self.TILEMASK:
            return False
        return (self.kind[(s >> self.SB) & self.MP] == BEXIT
                and self.kind[(s >> self.SI) & self.MP] == IEXIT)


class _Field:
    """Exact presses-to-win for every reachable state, by ENUMERATION.

    One forward closure from the state asked about, over `_Board.move`, then a
    backward BFS from the won boards over that closure's own reversed edges. A
    won board is TERMINAL in the closure: a win ends the episode, so no shortest
    path continues through one, and the boards that exist only beyond a win are
    not boards the game can be in.

    The result is exact in BOTH directions -- a state the backward BFS labels is
    that many presses from a win, and a state it does not label cannot win at
    all, which is a proof rather than a search giving up, because the closure
    holds every board reachable from the query. That second half is what the
    epsilon detours in `record_level` rest on: this game destroys tiles
    irreversibly and 92.6% of level 0's reachable boards are already lost, so
    "is this alternative press still winnable" has to be answered exactly and in
    O(1), and it is.

    The closure from a state contains the closure of everything in it, so one
    solve labels every state it will ever be asked about downstream. In practice
    the first query is the level's start and every later one -- a detour, a
    re-plan, a recovery probe -- is a dict lookup. ``states`` counts what has
    actually been enumerated so the reports can show it.

    ``cap`` is a runaway guard, not a budget: past it `get` answers None, which
    `_search` turns into "no plan" and `record_level` turns into a RESET. The
    largest closure in the game is level 0's 767,732 states.
    """

    __slots__ = ("board", "dist", "dead", "cap", "capped", "states", "solves")

    def __init__(self, board: _Board, cap: int):
        self.board = board
        self.cap = cap
        #: state -> exact presses to the nearest win.
        self.dist: dict = {}
        #: states proved unable to win at all.
        self.dead: set = set()
        self.capped = False
        self.states = 0
        self.solves = 0

    def get(self, state: int) -> "int | None":
        """Presses to a win from ``state``, or None when this board can never be
        won from there (or the cap stopped the closure short of knowing)."""
        hit = self.dist.get(state)
        if hit is not None:
            return hit
        if state in self.dead:
            return None
        if not self._solve(state):
            return None
        return self.dist.get(state)

    def _solve(self, state: int) -> bool:
        """Enumerate the closure of ``state`` and label all of it. False if the
        cap stopped it (nothing is merged in that case -- a partial closure would
        label states that have unexplored escapes as dead)."""
        board = self.board
        seen = {state}
        queue = deque([state])
        rev: dict = {}
        wins = []
        while queue:
            cur = queue.popleft()
            if board.won(cur):
                wins.append(cur)
                continue                         # a win ends the episode
            for di in range(5):
                nxt = board.step(cur, di)
                if nxt == cur:
                    continue                     # a press that does nothing
                rev.setdefault(nxt, []).append(cur)
                if nxt not in seen:
                    if len(seen) >= self.cap:
                        self.capped = True
                        return False
                    seen.add(nxt)
                    queue.append(nxt)

        dist = {s: 0 for s in wins}
        back = deque(wins)
        while back:
            cur = back.popleft()
            nd = dist[cur] + 1
            for prev in rev.get(cur, ()):
                if prev not in dist:
                    dist[prev] = nd
                    back.append(prev)

        self.dist.update(dist)
        self.dead |= seen - dist.keys()
        self.states += len(seen)
        self.solves += 1
        return True

    def optimal(self, state: int) -> list:
        """``[(direction index, successor), ...]`` for every press on a shortest
        path from ``state``. Ties come out in ``DIRS`` order, which is what makes
        a re-derived plan byte-identical across processes.

        Exact, with nothing inferred: a press is optimal iff its successor is one
        step nearer the win, and both distances are the field's. Every successor
        is already labelled -- it is in the closure that labelled ``state`` -- so
        this never enumerates anything."""
        rest = self.get(state)
        if not rest:                     # None (dead / capped) or 0 (already won)
            return []
        out = []
        for di in range(5):
            nxt = self.board.step(state, di)
            if nxt != state and self.dist.get(nxt) == rest - 1:
                out.append((di, nxt))
        return out

    def plan(self, state: int) -> "Plan | None":
        """The shortest press sequence from ``state``, with the EXACT optimal set
        at every step, or None if this board can no longer be won."""
        rest = self.get(state)
        if rest is None:
            return None
        presses, optsets = [], []
        cur = state
        for _ in range(rest):
            best = self.optimal(cur)
            if not best:                 # a labelled state always has a step
                return None
            presses.append(DIRS[best[0][0]])
            optsets.append([DIRS[di] for di, _ in best])
            cur = best[0][1]
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class TogetherAloneExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Field`.

    The base class keeps the memo, the level scoping and the disk cache; only the
    strategy underneath changes, the way `PSEnumExpert` replaces it. Here the
    strategy is "read the live board and walk down the exact distance field", so
    `heuristic` is never called and asserts rather than returning a number
    nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state (a recovery probe, an epsilon detour) is answered exactly --
    including the answer "no", which on this game is a common one and is a proof
    rather than a budget running out (see `_Field`).
    """

    directions = list(DIRS)
    #: `_key` is the dynamic objects only, which is canonical WITHIN a level but
    #: not across them -- the void, the gray tiles and the exits are static per
    #: level and differ between levels.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    #: Runaway guard on one closure; see `_Field`. The largest this game reaches
    #: is level 0's 767,732 states.
    field_cap: int = 4_000_000

    def setup(self) -> None:
        g = self.g
        self.ben_ids = set(g.resolve_object_name("ben"))
        self.isa_ids = set(g.resolve_object_name("isabelle"))
        self.benactive_ids = set(g.resolve_object_name("benactive"))
        self.isaactive_ids = set(g.resolve_object_name("isaactive"))
        #: The two objects that live for one rule application inside a single
        #: turn. They must never be on a rendered grid; `--fuzz` asserts it.
        self.pending_ids = (set(g.resolve_object_name("benwillbeactive"))
                            | set(g.resolve_object_name("isawillbeactive")))
        self.gray_ids = set(g.resolve_object_name("graytile"))
        self.bexit_ids = set(g.resolve_object_name("brownexit"))
        self.iexit_ids = set(g.resolve_object_name("blueexit"))
        #: Breakable tile object -> its packed value.
        self.tile_value = {}
        for name, val in (("browntile", BROWN), ("bluetile", BLUE),
                          ("twocoltile", TWOCOL)):
            for idx in g.resolve_object_name(name):
                self.tile_value[idx] = val
        self.tile_ids = set(self.tile_value)
        #: The inverse of `tile_value`, and the two character ids -- resolved
        #: once here rather than per call, because `_seat` runs once per press
        #: in the interpreter sweeps.
        self.value_tile = {BROWN: min(g.resolve_object_name("browntile")),
                           BLUE: min(g.resolve_object_name("bluetile")),
                           TWOCOL: min(g.resolve_object_name("twocoltile"))}
        self.ben_id = min(self.ben_ids)
        self.isa_id = min(self.isa_ids)
        self.marker_id = (min(self.benactive_ids), min(self.isaactive_ids))
        #: Everything a press can move or destroy. `_statics` strips exactly
        #: this set to get the layer a state is seated back onto.
        self.dynamic_ids = (self.ben_ids | self.isa_ids | self.benactive_ids
                            | self.isaactive_ids | self.pending_ids
                            | self.tile_ids)

        # One `_Board` per level, built from the PARSED START MAPS rather than
        # from a live grid: the slot set is "the cells that hold a breakable
        # tile at the start", which a mid-game grid no longer shows. Boards are
        # then looked up by static signature, so `read` never has to be told
        # which level it is looking at.
        self._boards: dict = {}
        for level, rows in enumerate(self.g.levels):
            board = self._board_from(rows)
            if board.sig in self._boards:
                raise AssertionError(
                    f"levels {self._boards[board.sig][0]} and {level} have the "
                    f"same static signature; `read` could not tell them apart")
            self._boards[board.sig] = (level, board)
        self._fields: dict = {}

    def _board_from(self, rows) -> _Board:
        """A `_Board` from one parsed level map (a grid of object-id sets)."""
        h, w = len(rows), len(rows[0])
        kind: list = []
        slots: list = []
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                i = r * w + c
                if cell & self.gray_ids:
                    kind.append(GRAY)
                elif cell & self.bexit_ids:
                    kind.append(BEXIT)
                elif cell & self.iexit_ids:
                    kind.append(IEXIT)
                elif cell & self.tile_ids:
                    kind.append(SLOT)
                    slots.append(i)
                else:
                    kind.append(VOID)
        return _Board(h, w, kind, slots)

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "TogetherAloneExpert reads an exact distance field; "
            "heuristic is unused")

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, packed state)`` for the engine's current grid.

        The level is identified by its STATIC signature -- shape, gray tiles and
        the two exits -- none of which any rule changes, so this works at any
        state and not only at a level start.

        ONE pass over the grid: this is called once per press by the interpreter
        sweeps, which run to millions of presses, so everything it needs comes
        out of the same scan."""
        h, w = eng.height, eng.width
        gray: list = []
        bexit: list = []
        iexit: list = []
        tiles = {}
        ben = isa = None
        active = 0
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.gray_ids:
                    gray.append(i)
                elif cell & self.bexit_ids:
                    bexit.append(i)
                elif cell & self.iexit_ids:
                    iexit.append(i)
                got = cell & self.tile_ids
                if got:
                    tiles[i] = self.tile_value[next(iter(got))]
                if cell & self.ben_ids:
                    ben = i
                if cell & self.isa_ids:
                    isa = i
                if cell & self.isaactive_ids:
                    active = 1
        found = self._boards.get(_Board.signature(h, w, gray, bexit, iexit))
        if found is None:
            raise AssertionError("this grid matches no level of Together Alone")
        _level, board = found
        if ben is None or isa is None:
            raise AssertionError("a character is missing from the grid")
        stray = set(tiles) - set(board.slots)
        if stray:
            raise AssertionError(f"a tile appeared outside the start slots: {stray}")
        return board, board.pack(ben, isa, active, tiles)

    def field(self, board: _Board) -> _Field:
        got = self._fields.get(board.sig)
        if got is None:
            got = self._fields[board.sig] = _Field(board, self.field_cap)
        return got

    def _key(self, eng) -> frozenset:
        """``(row, col, tag)`` triples: the tiles by colour, the two characters,
        and which of them is active.

        Built from the MODEL's reading rather than from raw object ids, so the
        key is exactly what a state consists of -- which is also what the
        ``plan_cache_path`` signature is stored as, so a cached plan is matched
        against the board it was solved from and an edited level is a miss
        rather than a wrong plan. The active flag is carried as a separate tag
        on the active character's cell, because when the two share a square the
        two states differ in nothing else."""
        board, state = self.read(eng)
        ben, isa, active, tiles = board.unpack(state)
        w = board.w
        out = {(cell // w, cell % w, val) for cell, val in tiles.items()}
        out.add((ben // w, ben % w, 4))
        out.add((isa // w, isa % w, 5))
        who = isa if active else ben
        out.add((who // w, who % w, 6 + active))
        return frozenset(out)

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if board.won(state):
            return Plan([], [])
        return self.field(board).plan(state)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class TogetherAloneSolver(PSAStarSolver):
    game_id = "puzzlescript_together_alone"
    game_name = GAME_NAME
    expert_cls = TogetherAloneExpert

    #: `games/ps:together_alone/ps:together_alone.py` is a plain passthrough --
    #: it builds the adapter and nothing else, and the rendering fix is in the
    #: .txt, which both paths read. Set this if that wrapper ever grows a patch.
    game_module_id = ""

    #: Unused: `TogetherAloneExpert._search` never calls `_astar`. Left at the
    #: base values so nothing reads a lie off them;
    #: `TogetherAloneExpert.field_cap` is the knob that bounds the closures.
    weight = 1
    node_cap = 400_000

    #: Room for the longest plan (25 presses on level 3) plus the detours. The
    #: adapter's own 200-press per-level budget is the real ceiling and it is
    #: reset by the `set_level` that ends the exploration prefix, so the plan
    #: starts it from zero; this is set under it so the expert stops itself
    #: rather than being cut off by a GAME_OVER.
    max_steps = 120

    #: Every level is solved; nothing is skipped.
    skip_levels = frozenset()

    #: On top of the RESET prefix: about one press in eight is a random legal
    #: alternative, after which the expert re-plans from wherever it landed --
    #: the taken action is the mistake and ``optimal`` is the recovery, which is
    #: the signal a policy needs after its own error.
    #:
    #: Detours are safe here ONLY because the field's deadness test is exact and
    #: free: `record_level` offers each alternative to `expert.plan` first and
    #: keeps it only if the board can still be won. On a game where 92.6% of
    #: level 0's reachable boards are already lost, an unchecked detour would
    #: brick most attempts.
    #:
    #: The rate is what a MEASURED sweep supports rather than a guess, and it is
    #: high for this family for a reason specific to the game: a detour that
    #: breaks the wrong tile is nearly always REJECTED outright (it makes the
    #: board dead and the field says so in O(1)), so what actually survives the
    #: probe is a step or a switch, which costs 1-2 presses to undo. Swept over
    #: 200 seeds x 4 levels at 0.0 / 0.06 / 0.08 / 0.10 / 0.12 / 0.18: not one
    #: seed is rejected at any rate, and at 0.12 the worst level recording is 53
    #: presses (against ``max_steps`` 120 and the adapter's 200) with 9.8% of
    #: the 19,120 expert steps carrying a taken action outside their own optimal
    #: set. 0.18 was left on the table because it starts to dilute the shortest
    #: trajectory rather than because it fails. A rejected seed would cost an
    #: extra attempt and nothing else (the WIN-only contract drops it), so this
    #: is a throughput knob, not a correctness one.
    epsilon = 0.12

    def prepare_expert(self, game, expert) -> None:
        """Enumerate every level's closure before `discover_solvable` asks for
        it -- the same work either way, but it fills the disk cache in one pass
        and makes the startup cost visible as startup."""
        for level in range(game.n_levels):
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    """The solver, its adapter and an expert -- without planning anything.

    The reports below each want a different slice (``--audit`` needs no plan at
    all, ``--fuzz`` needs only the model), so planning is left to whoever asks
    for it rather than paid on every entry point."""
    solver = TogetherAloneSolver()
    game = solver.make_game(seed)
    return solver, game, TogetherAloneExpert(game, node_cap=solver.node_cap)


def _ascii(board: _Board, state: int) -> str:
    """A model state as ASCII, so a failing check can print the board it failed
    on rather than an integer.

    Two grids side by side, because a character standing on a square hides what
    is under it -- which is the very confusion the render fix was about, and it
    would be perverse to reproduce it in the debug dump. Left is the ground
    (``.`` void, ``_`` a slot whose tile is gone, ``1`` brown, ``2`` blue, ``+``
    two-colour, ``#`` gray, ``E`` Ben's exit, ``X`` Isabelle's exit); right is
    who is standing where, UPPERCASE for whichever of them is active (``b``/``B``
    Ben, ``i``/``I`` Isabelle, ``&`` both on one square)."""
    ben, isa, active, tiles = board.unpack(state)
    out = []
    for r in range(board.h):
        ground = occupants = ""
        for c in range(board.w):
            i = r * board.w + c
            k = board.kind[i]
            ground += ({GRAY: "#", BEXIT: "E", IEXIT: "X", VOID: "."}.get(k)
                       or {GONE: "_", BROWN: "1", BLUE: "2",
                           TWOCOL: "+"}[tiles.get(i, GONE)])
            if i == ben and i == isa:
                occupants += "&"
            elif i == ben:
                occupants += "b" if active else "B"
            elif i == isa:
                occupants += "I" if active else "i"
            else:
                occupants += "."
        out.append(f"|{ground}|  |{occupants}|")
    out.append(f"active: {'Isabelle' if active else 'Ben'}")
    return "\n".join(out)


def _start(game, expert, level: int):
    """``(board, state)`` at a level's start, engine left seated there."""
    game.set_level(level)
    return expert.read(game._engine)


def _plans() -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the field and the tie labelling at
    once."""
    _solver, game, expert = _new()
    eng = game._engine
    total = ties = bad = 0
    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        field = expert.field(board)
        # Force the closure: `plan` below may answer straight from the disk
        # cache, and then the space counts printed here would all read zero.
        field.get(state)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level}: UNSOLVED")
            bad += 1
            continue
        for direction in plan:
            eng.step(direction)
        won = eng.check_win()
        bad += not won
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        tie_steps = sum(1 for s in sets if len(s) > 1)
        switches = sum(1 for p in plan if p == "action")
        total += len(plan)
        ties += tie_steps
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"  L{level}: {board.h}x{board.w}  {board.nslots:2d} tiles  "
              f"{len(plan):3d} presses ({switches:2d} of them the switch)  "
              f"win={won}  (budget {game._max_steps}, {room})  "
              f"{tie_steps:2d} steps with a tie set  "
              f"[space {field.states}, {len(field.dead)} of it already lost]")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _forward(board: _Board, state: int, limit: int) -> "int | None":
    """Presses to a win from ``state``, by a plain FORWARD BFS, or None past
    ``limit``.

    Shares nothing with `_Field` but `_Board.move` itself, which is the point: it
    is the independent answer every optimality claim below is checked against.
    Depth-bounded, so it is also the cheap way to ask "is there a win within k"
    for one successor without labelling anything."""
    if board.won(state):
        return 0
    seen = {state}
    queue = deque([(state, 0)])
    while queue:
        cur, d = queue.popleft()
        if d >= limit:
            continue
        for di in range(5):
            nxt = board.step(cur, di)
            if nxt == cur or nxt in seen:
                continue
            if board.won(nxt):
                return d + 1
            seen.add(nxt)
            queue.append((nxt, d + 1))
    return None


def _selfcheck(samples: int = 80, verbose: bool = True) -> int:
    """Four things the expert would otherwise be trusted on. (The model itself is
    checked against the interpreter by ``--fuzz``, which is a bigger job and has
    its own entry point.)

    1. **The plans are shortest.** A plain forward BFS to the first win, which is
       shortest by construction and shares no line with the field, must return
       the field's plan length on every level.
    2. **The tie sets are exact.** Every step's optimal set is re-derived by
       brute force -- a fresh depth-bounded forward BFS from each of the five
       successors -- and must match the field's answer exactly.
    3. **The field's won set is the win condition.** The states the closure
       labels 0 must be exactly the reachable states `_Board.won` accepts. A
       field that missed a goal would price every distance too high and nothing
       else would notice.
    4. **Recovery is answered from ANY board, and its "no" is right too.** This
       is the one that matters at generation time: the exploration prefix and the
       epsilon detours both leave the board off every shortest path, and on this
       game they usually leave it unwinnable. From ``samples`` states reached by
       seeded random presses, the field must either return a plan whose length an
       independent forward BFS agrees with AND which wins when replayed through
       the interpreter, or declare the board dead -- in which case the
       independent BFS, run past the diameter of the whole space, must agree that
       there is no win. Both halves are checked; a field that called live boards
       dead would silently turn detours into resets.
    """
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0

    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        shortest = _forward(board, state, len(plan) + 1 if plan else 200)
        ok = plan is not None and shortest == len(plan)
        bad += not ok
        if verbose:
            print(f"  L{level}: field {len(plan) if plan else None} presses vs "
                  f"forward BFS {shortest} -- "
                  f"{'SHORTEST' if ok else 'NOT SHORTEST'}")

    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        sets = getattr(plan, "optsets", None) or []
        mismatched = 0
        cur = state
        for i, direction in enumerate(plan):
            rest = len(plan) - i
            truth = []
            for di in range(5):
                nxt = board.step(cur, di)
                if nxt == cur:
                    continue
                if _forward(board, nxt, rest - 1) == rest - 1:
                    truth.append(DIRS[di])
            if sorted(truth) != sorted(sets[i]):
                mismatched += 1
                print(f"    L{level} step {i}: brute force {sorted(truth)} vs "
                      f"field {sorted(sets[i])}")
            cur = board.step(cur, DIRS.index(direction))
        bad += mismatched
        if verbose:
            print(f"  L{level}: {len(plan)} tie sets "
                  f"{'match' if not mismatched else f'{mismatched} MISMATCH'}"
                  f" brute force")

    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        field = expert.field(board)
        field.get(state)     # force the closure; `plan` would hit the disk cache
        labelled = {s for s, d in field.dist.items() if d == 0}
        by_predicate = {s for s in itertools.chain(field.dist, field.dead)
                        if board.won(s)}
        wrong = labelled ^ by_predicate
        bad += len(wrong)
        if verbose:
            print(f"  L{level}: {len(field.dist) + len(field.dead):6d} reachable "
                  f"states, {len(labelled)} of them won ({len(field.dead)} "
                  f"already lost); {len(wrong)} disagreement(s) with the win "
                  f"condition")

    for level in range(game.n_levels):
        rng = random.Random(f"together_alone:recovery:{level}")
        board, state = _start(game, expert, level)
        field = expert.field(board)
        field.get(state)     # force the closure; `plan` would hit the disk cache
        diameter = max(field.dist.values()) + 1
        wrong = dead = won_after = 0
        for _ in range(samples):
            game.set_level(level)
            for _ in range(rng.randrange(1, 20)):
                eng.step(DIRS[rng.randrange(5)])
                if eng.check_win():
                    break
            if eng.check_win():
                continue                 # the walk already won; nothing to plan
            board, state = expert.read(eng)
            plan = expert._search(eng)
            truth = _forward(board, state, diameter)
            if plan is None:
                dead += 1
                if truth is not None:
                    wrong += 1
                    print(f"    L{level} called DEAD but a win is {truth} "
                          f"presses away:\n{_ascii(board, state)}")
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
            print(f"  L{level}: {samples} random boards -- {won_after} re-planned "
                  f"to a WIN in the interpreter, {dead} proved already lost, "
                  f"{wrong} WRONG")
    return bad


# ---------------------------------------------------------------------------
# Seating a model state back onto the interpreter
# ---------------------------------------------------------------------------

def _statics(eng, expert):
    """The level's static layer -- every cell with the tiles, the characters and
    the active markers stripped out -- so a state can be SEATED back onto the
    engine instead of keeping a snapshot of every board it reaches.

    Taken from the live grid rather than assembled from object ids, so it carries
    whatever the level actually holds (the background under everything included)
    and a seated board is byte-identical to the one the level loaded."""
    return [[cell - expert.dynamic_ids for cell in row] for row in eng.grid]


def _seat(eng, static, expert, board: _Board, state: int) -> None:
    """Put ``state`` back on the engine. The inverse of `read`; that it IS the
    inverse is checked before either is used (see `_engine`)."""
    ben, isa, active, tiles = board.unpack(state)
    obj_of = expert.value_tile
    ben_id, isa_id = expert.ben_id, expert.isa_id
    marker = expert.marker_id[active]
    w = board.w
    grid = []
    for r, row in enumerate(static):
        out = []
        for c, cell in enumerate(row):
            i = r * w + c
            new = set(cell)
            if i in tiles:
                new.add(obj_of[tiles[i]])
            if i == ben:
                new.add(ben_id)
            if i == isa:
                new.add(isa_id)
            if i == (isa if active else ben):
                new.add(marker)
            out.append(new)
        grid.append(out)
    eng.grid = grid
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


def _check_seating(eng, static, expert, board, state, level, verbose=True) -> int:
    """A decoder that is not exactly the inverse of the encoder enumerates a
    DIFFERENT game, so this is checked rather than assumed: seating the read
    state must reproduce the grid byte-for-byte, and a seated board must step
    identically to a snapshot-restored one on all five presses."""
    bad = 0
    original = snapshot(eng)
    _seat(eng, static, expert, board, state)
    if eng.grid != original or expert.read(eng)[1] != state:
        print(f"  L{level}: SEATING IS NOT THE INVERSE OF READING")
        return 1
    for di in range(5):
        restore(eng, original)
        eng.step(DIRS[di])
        expected = snapshot(eng)
        _seat(eng, static, expert, board, state)
        eng.step(DIRS[di])
        if eng.grid != expected:
            print(f"  L{level}: a seated board steps differently ({DIRS[di]})")
            bad += 1
    _seat(eng, static, expert, board, state)
    if verbose and not bad:
        print(f"  L{level}: seating is the exact inverse of reading")
    return bad


# ---------------------------------------------------------------------------
# The differential fuzz: is the native model the interpreter's game?
# ---------------------------------------------------------------------------

def _fuzz(walk_presses: int = 1500, witnesses: int = 60,
          exhaustive=(1, 2, 3), verbose: bool = True) -> int:
    """Compare `_Board` against `PSEngine.step` press by press, three ways.

    1. **A seeded random walk** on every level, comparing the model's board
       against the engine's grid after EVERY press -- including the presses that
       do nothing, which is where a collision-layer mistake hides. It also
       asserts the two invariants the model assumes about the switch: exactly one
       active marker is on the board after every press, and BenWillBeActive /
       IsaWillBeActive (which exist for one rule application inside a turn) are
       never on a grid anything renders.
    2. **A witness per BRANCH.** A random walk almost never produces the rare
       configurations -- both characters on one square, Ben with a hole in front
       of him and ground beyond -- and a shortest plan produces them on purpose,
       which is the ps:sokobaiogenesis lesson. So the whole reachable space of
       each level is classified by the branch `_Board.move` reports, and up to
       ``witnesses`` (state, press) pairs are drawn from EACH of the eight
       branches -- by reservoir, so level 0's 3.8M pairs are sampled uniformly
       without being materialised -- and checked against the interpreter. The
       report prints the class sizes, so a branch no shipped level can reach is
       visible as such rather than as silence.
    3. **Exhaustively**, for the levels named in ``exhaustive``: every reachable
       state, every press, seated and stepped on the real interpreter. Levels
       1-3 are 10,274 states (~25 s) and are the default; level 0 is 767,732
       states / 3,838,650 presses (~25 min) and is opt-in via ``--fuzz 0``,
       which the report says out loud rather than printing a clean line for a
       space it did not walk. It HAS been run, and the model and the interpreter
       agree on all 3,838,650 of them -- so on this game "the model is the
       interpreter's game" is not a sampling claim.
    """
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0

    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        rng = random.Random(f"together_alone:fuzz:{level}")
        drift = 0
        for t in range(walk_presses):
            di = rng.randrange(5)
            branch, predicted = board.move(state, di)
            eng.step(DIRS[di])
            _b, live = expert.read(eng)
            markers = sum(bool(cell & (expert.benactive_ids | expert.isaactive_ids))
                          for row in eng.grid for cell in row)
            pending = any(cell & expert.pending_ids
                          for row in eng.grid for cell in row)
            if (live != predicted or eng.check_win() != board.won(predicted)
                    or markers != 1 or pending):
                drift += 1
                print(f"    L{level} press {t} ({DIRS[di]}, {BRANCHES[branch]}): "
                      f"model\n{_ascii(board, predicted)}\nengine\n"
                      f"{_ascii(board, live)}\n"
                      f"    markers={markers} pending={pending}")
                break
            state = live
            if board.won(state):
                board, state = _start(game, expert, level)
        bad += drift
        if verbose:
            print(f"  L{level}: model {'MATCHES' if not drift else 'DIVERGES from'}"
                  f" the interpreter over {walk_presses} random presses")

    for level in range(game.n_levels):
        board, start = _start(game, expert, level)
        static = _statics(eng, expert)
        bad += _check_seating(eng, static, expert, board, start, level, verbose=False)

        # Classify the WHOLE space by branch and keep a RESERVOIR of witnesses
        # per branch rather than the pairs themselves: level 0 has 3.8M
        # (state, press) pairs and materialising them would cost more memory
        # than the field it is checking.
        rng = random.Random(f"together_alone:witness:{level}")
        counts = [0] * 8
        pool: list = [[] for _ in range(8)]
        seen = {start}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            if board.won(cur):
                continue
            for di in range(5):
                branch, nxt = board.move(cur, di)
                counts[branch] += 1
                bag = pool[branch]
                if len(bag) < witnesses:
                    bag.append((cur, di))
                else:
                    j = rng.randrange(counts[branch])
                    if j < witnesses:
                        bag[j] = (cur, di)
                if nxt != cur and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)

        wrong = 0
        checked = 0
        for b in range(8):
            for cur, di in pool[b]:
                _seat(eng, static, expert, board, cur)
                predicted = board.step(cur, di)
                eng.step(DIRS[di])
                checked += 1
                if expert.read(eng)[1] != predicted:
                    wrong += 1
                    print(f"    L{level} {BRANCHES[b]} witness disagrees:\n"
                          f"{_ascii(board, cur)}\npress {DIRS[di]}")
        bad += wrong
        if verbose:
            sizes = ", ".join(f"{BRANCHES[b]} {counts[b]}" for b in range(8))
            print(f"  L{level}: {len(seen):6d} states; {sizes}")
            print(f"          {checked} branch witnesses checked on the "
                  f"interpreter, {wrong} disagreement(s)")
        game.set_level(level)

    for level in range(game.n_levels):
        if level not in exhaustive:
            print(f"  L{level}: exhaustive sweep SKIPPED -- run "
                  f"`--fuzz {level}` to press every button at every one of its "
                  f"reachable states")
            continue
        board, start = _start(game, expert, level)
        static = _statics(eng, expert)
        seen = {start}
        queue = deque([start])
        wrong = presses = 0
        while queue:
            cur = queue.popleft()
            if board.won(cur):
                continue
            for di in range(5):
                _seat(eng, static, expert, board, cur)
                eng.step(DIRS[di])
                presses += 1
                got = expert.read(eng)[1]
                predicted = board.step(cur, di)
                if got != predicted or eng.check_win() != board.won(predicted):
                    wrong += 1
                    if wrong <= 3:
                        print(f"    L{level} exhaustive disagreement, press "
                              f"{DIRS[di]}:\n{_ascii(board, cur)}")
                if predicted != cur and predicted not in seen:
                    seen.add(predicted)
                    queue.append(predicted)
        bad += wrong
        if verbose:
            print(f"  L{level}: EXHAUSTIVE -- {len(seen)} states, {presses} "
                  f"interpreter presses, {wrong} disagreement(s)")
        game.set_level(level)

    # The docstring claims no shipped level places a TwoColTile, which is why
    # `jump+flip` on a two-colour tile is unreachable rather than untested.
    # Asserted rather than assumed.
    twocol_ids = set(expert.g.resolve_object_name("twocoltile"))
    twocol = sum(1 for rows in expert.g.levels for row in rows for cell in row
                 if cell & twocol_ids)
    print(f"  TwoColTile: modelled, placed on {twocol} cell(s) across the four "
          f"levels -- {'unused, as documented' if not twocol else 'IN USE'}")
    return bad


# ---------------------------------------------------------------------------
# The interpreter proofs
# ---------------------------------------------------------------------------

def _engine(levels=(1, 2, 3), verbose: bool = True) -> int:
    """The proof that owes nothing to the native model: the same two sweeps,
    driven by the REAL INTERPRETER.

    Forward BFS from the level start by pressing actual buttons, keying each
    board by what is on the grid, stopping at the depth the first `check_win`
    appears -- which fixes the shortest length by construction -- with every edge
    recorded; then a backward sweep from the won boards over those edges, which
    gives the exact optimal SET at every state on a shortest path. No `_Board`
    call takes part: successors come from `PSEngine.step` and the goal test is
    `check_win`. `_Board` is used only to name the level and to seat a key.

    States are re-seated from their keys rather than snapshotted, which is what
    keeps the balls in megabytes; the seating is checked to be the exact inverse
    of the reading first.

    Level 0's ball is 315,265 boards (~13 min) so it is not in the default set;
    ``--engine 0`` runs it."""
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        if level not in levels:
            print(f"  L{level}: SKIPPED -- run `--engine {level}` to sweep it "
                  f"(level 0's ball is 315,265 interpreter boards, ~5 min; the "
                  f"others are seconds)")
            continue
        board, start = _start(game, expert, level)
        plan = expert.plan(eng, level)
        game.set_level(level)
        static = _statics(eng, expert)
        bad += _check_seating(eng, static, expert, board, start, level,
                              verbose=False)

        depth = {start: 0}
        queue = deque([start])
        rev: dict = {}
        won_states: set = set()
        d_star = None
        while queue:
            cur = queue.popleft()
            d = depth[cur]
            if d_star is not None and d >= d_star:
                break
            for di in range(5):
                _seat(eng, static, expert, board, cur)
                eng.step(DIRS[di])
                nxt = expert.read(eng)[1]
                if nxt == cur:
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
            print(f"  L{level}: the interpreter found no win")
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

        # Read the plan and its tie sets back out of the interpreter's own field.
        epresses, esets = [], []
        cur = start
        while dist.get(cur):
            rest = dist[cur]
            best = []
            for di in range(5):
                _seat(eng, static, expert, board, cur)
                eng.step(DIRS[di])
                nxt = expert.read(eng)[1]
                if nxt != cur and dist.get(nxt) == rest - 1:
                    best.append((di, nxt))
            epresses.append(DIRS[best[0][0]])
            esets.append(sorted(DIRS[di] for di, _ in best))
            cur = best[0][1]

        same_len = len(epresses) == len(plan) == d_star
        same_sets = esets == [sorted(s) for s in plan.optsets]
        bad += (not same_len) + (not same_sets)
        if verbose:
            print(f"  L{level}: {len(depth):6d} engine boards within d*, "
                  f"{d_star:3d} presses vs the field's {len(plan):3d} -- "
                  f"{'AGREE' if same_len else 'DISAGREE'}; tie sets "
                  f"{'AGREE' if same_sets else 'DISAGREE'}")
        game.set_level(level)
    return bad


def _enumerate(levels=(1, 2, 3), verbose: bool = True) -> int:
    """Walk the interpreter's WHOLE reachable space with the shared
    `StateGraph.build` and solve the graph exactly.

    A fourth derivation, and the one that shares no algorithm with anything here
    either -- `StateGraph` is `solvers/common/ps_astar.py`'s, written for other
    games. It buys two things `--engine` cannot: plans that are shortest against
    the entire space rather than against a ball, and an independent count of the
    boards that CANNOT win, which is the claim `_Field`'s deadness test makes and
    the epsilon detours rest on.

    Level 0's whole space is 767,732 states (~30 min); ``--enumerate 0`` runs
    it."""
    from solvers.common.ps_astar import StateGraph                     # noqa: PLC0415

    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        if level not in levels:
            print(f"  L{level}: SKIPPED -- run `--enumerate {level}` to walk "
                  f"it (level 0's space is 767,732 interpreter boards; the "
                  f"others are seconds)")
            continue
        board, state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        field = expert.field(board)
        field.get(state)     # force the closure; `plan` above may hit the cache
        won = sum(1 for d in field.dist.values() if d == 0)
        game.set_level(level)
        static = _statics(eng, expert)
        graph = StateGraph.build(
            eng, lambda e: expert.read(e)[1], list(DIRS),
            node_cap=2_000_000,
            decode=lambda e, k: _seat(e, static, expert, board, k))
        if graph is None:
            print(f"  L{level}: past the enumeration cap")
            bad += 1
            continue
        eplan = graph.plan(list(DIRS))
        stranded = [k for k in graph.succ if k not in graph.dist]
        same_len = eplan is not None and len(eplan) == len(plan)
        same_sets = eplan is not None and (
            [sorted(s) for s in eplan.optsets]
            == [sorted(s) for s in plan.optsets])
        # The interpreter's own count of hopeless boards, against the field's --
        # the claim `_Field`'s deadness test makes, checked against a space that
        # was walked by pressing real buttons. `StateGraph` keys a WIN as an EDGE
        # rather than as a state, so its space is the field's minus the won
        # boards, and that identity is asserted too rather than left to eyeball.
        same_dead = len(stranded) == len(field.dead)
        same_size = (len(graph.succ) + won
                     == len(field.dist) + len(field.dead))
        bad += ((not same_len) + (not same_sets) + (not same_dead)
                + (not same_size))
        if verbose:
            print(f"  L{level}: {len(graph.succ):6d} engine states "
                  f"({graph.steps} presses), "
                  f"{len(eplan) if eplan else None} presses vs the field's "
                  f"{len(plan)} -- {'AGREE' if same_len else 'DISAGREE'}; "
                  f"tie sets {'AGREE' if same_sets else 'DISAGREE'}; "
                  f"{len(stranded)} state(s) that cannot win vs the field's "
                  f"{len(field.dead)} -- "
                  f"{'AGREE' if same_dead else 'DISAGREE'}; "
                  f"space +{won} won "
                  f"{'AGREE' if same_size else 'DISAGREE'}")
        game.set_level(level)
    return bad


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

#: Every cell stack a board of this game can hold. Five collision layers, but
#: only four can ever be occupied at once: Background is everywhere, Ground is
#: one of six things (or nothing, which is the void), Ben and Isabelle are each
#: present or not, and the HelperObjects layer holds exactly one active marker,
#: on whichever character is active -- so a cell with a character in it either
#: carries the marker or does not, and a cell with BOTH carries the marker of
#: one of them. Nobody can stand on the void, so those stacks are excluded.
_GROUNDS = [(), ("browntile",), ("bluetile",), ("graytile",),
            ("brownexit",), ("blueexit",), ("twocoltile",)]
_OCCUPANTS = [(), ("ben", "benactive"), ("ben",),
              ("isabelle", "isaactive"), ("isabelle",),
              ("ben", "isabelle", "benactive"),
              ("ben", "isabelle", "isaactive")]
_COMPOSITIONS = [g + o for g in _GROUNDS for o in _OCCUPANTS
                 if not (g == () and o != ())]

#: `_render_frame`'s letterbox colour. A composition that renders as a uniform
#: frame of it is invisible against the pad (the ps:stand_iii lesson) -- which
#: is exactly what the shipped Background (``Black``) did, since the levels are
#: 5-7 rows tall in a 64x64 frame and the pad rows ran straight into the board.
_PAD = 5


def _comp_name(comp) -> str:
    return "+".join(comp) or "void"


def _audit(verbose: bool = True) -> int:
    """Three passes over every cell COMPOSITION this game can build.

    **Pass 1 -- distinctness**, at every cell size the four boards use (8px for
    all of them; the size list is derived from the levels rather than assumed, so
    an edited level is covered). Whole 64x64 frames of uniform boards are
    compared rather than one cell out of a mixed board: `_render_frame` upscales
    and centre-pads, so slicing a cell by ``cell_px`` arithmetic reads the wrong
    pixels (the ps:explod lesson). Two uniform boards render identically iff
    their cells do.

    This is the pass that fails on the shipped .txt -- 174 identical pairs out of
    43 compositions, including every ``<any ground>+ben+benactive``. See the
    header comment in ``data/puzzlescript_games/Together_Alone.txt``.

    **Pass 2 -- nothing is the letterbox.** No composition may render as a
    uniform frame of the pad colour, or it would have no seam against the border
    of the picture. The void is the case here and it is why Background is no
    longer black.

    **Pass 3 -- the group.** The game is augmented with a frame rotation AND both
    flips, so a board is drawn at any of 16 presentations, and the sprites are
    not all invariant under them (Ground is two horizontal bands; Ben is a block
    on the left). That is fine -- the whole frame is transformed together, so a
    turned board is a real board -- as long as no transform of one composition
    lands on a DIFFERENT one, which is what this checks: all 16 transforms of
    each, against all others. It runs on SQUARE boards, because a transform of a
    non-square board also moves the letterbox and every comparison would pass for
    the wrong reason.

    It also reports which compositions are actually REACHABLE, measured over the
    four levels' whole enumerated state spaces plus their winning boards --
    because a reachability count taken from an expansion misses the win frame,
    which is the one composition the corpus exists to teach (the
    ps:sokoban_dungeon lesson)."""
    from adapters.puzzlescript_adapter import _render_frame            # noqa: PLC0415

    _solver, game, expert = _new()
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
        pad = [c for c in _COMPOSITIONS if np.all(shots[c] == _PAD)]
        bad += len(clashes) + len(pad)
        if verbose:
            print(f"  {h}x{w} (cell {min(64 // h, 64 // w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): {len(shots)} "
                  f"compositions, {'OK' if not clashes and not pad else 'CLASH'}")
        for a, b in clashes:
            print(f"      IDENTICAL: {_comp_name(a)}  ==  {_comp_name(b)}")
        for c in pad:
            print(f"      INVISIBLE AGAINST THE LETTERBOX: {_comp_name(c)}")

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

    for cell in sorted({min(64 // h, 64 // w) for h, w in sizes}):
        n = 64 // cell
        shots = {c: shoot(n, n, c) for c in _COMPOSITIONS}
        clashes = [(a, b, t) for a, b in itertools.permutations(_COMPOSITIONS, 2)
                   for t, turned in transforms(shots[a])
                   if np.array_equal(turned, shots[b])]
        invariant = [_comp_name(c) for c in _COMPOSITIONS
                     if all(np.array_equal(t, shots[c])
                            for _k, t in transforms(shots[c]))]
        bad += len(clashes)
        if verbose:
            print(f"  cell {cell}px ({n}x{n}): {len(clashes)} composition(s) "
                  f"whose transform is another's art; group-invariant: "
                  f"{', '.join(invariant)}")
        for a, b, t in clashes:
            print(f"      {_comp_name(a)} under {t} == {_comp_name(b)}")

    reachable = set()
    for level in range(game.n_levels):
        board, start = _start(game, expert, level)
        field = expert.field(board)
        field.get(start)
        for state in itertools.chain(field.dist, field.dead):
            ben, isa, active, tiles = board.unpack(state)
            for i in range(board.n):
                ground = {SLOT: None, GRAY: ("graytile",), BEXIT: ("brownexit",),
                          IEXIT: ("blueexit",), VOID: ()}[board.kind[i]]
                if ground is None:
                    ground = {GONE: (), BROWN: ("browntile",), BLUE: ("bluetile",),
                              TWOCOL: ("twocoltile",)}[tiles.get(i, GONE)]
                occ = ()
                if i == ben and i == isa:
                    occ = ("ben", "isabelle",
                           "isaactive" if active else "benactive")
                elif i == ben:
                    occ = ("ben",) + (() if active else ("benactive",))
                elif i == isa:
                    occ = ("isabelle",) + (("isaactive",) if active else ())
                reachable.add(ground + occ)
    unreachable = [c for c in _COMPOSITIONS if c not in reachable]
    if verbose:
        print(f"  {len(reachable)} of {len(_COMPOSITIONS)} compositions are "
              f"reachable in the four levels (the win frames included, since the "
              f"field stores won boards as states)")
        print(f"  never reachable: "
              f"{', '.join(_comp_name(c) for c in unreachable) or 'none'}")
    stray = reachable - set(_COMPOSITIONS)
    if stray:
        print(f"      COMPOSITION NOT IN THE AUDIT SET: "
              f"{', '.join(_comp_name(c) for c in stray)}")
        bad += len(stray)
    return bad


def _symmetry(walk_presses: int = 200, verbose: bool = True) -> int:
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unaugmented ones.

    This is the evidence for the rotation + flip augmentation. The structural
    argument is in `PuzzleScriptAdapter._FLIP_GAMES` -- every rule is written
    with the relative force ``>``, only one character carries a force at a time
    so nothing can contest a cell, and the win condition names no direction --
    but Gobble Rush's chirality hid inside exactly that kind of argument, so it
    is measured.

    Both the PLANS and a seeded random walk are replayed: the walk reaches the
    boards a plan never visits (tiles broken in the wrong order, a character
    stranded on a lone tile, both parked on the same square) and it presses the
    switch as well."""
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
            rng = random.Random(f"together_alone:symmetry:{level}")
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


def _levels_arg(flag: str, default) -> tuple:
    """Levels named after ``flag`` on the command line, else ``default``."""
    args = [int(a) for a in sys.argv[sys.argv.index(flag) + 1:] if a.isdigit()]
    return tuple(args) if args else tuple(default)


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--selfcheck" in sys.argv:
        violations = _selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--fuzz" in sys.argv:
        violations = _fuzz(exhaustive=_levels_arg("--fuzz", (1, 2, 3)))
        print(f"fuzz: {violations} disagreements with the interpreter")
        sys.exit(1 if violations else 0)
    if "--engine" in sys.argv:
        violations = _engine(_levels_arg("--engine", (1, 2, 3)))
        print(f"engine sweeps: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--enumerate" in sys.argv:
        violations = _enumerate(_levels_arg("--enumerate", (1, 2, 3)))
        print(f"whole-space enumeration: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = _audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = _symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    sys.exit(TogetherAloneSolver.main())
