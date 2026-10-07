"""Generate Phase-1 training data for the PuzzleScript game ps:the_nuevo_asylum
("The Nuevo Asylum", Kyle Davis / Zack Lewis / Zack Long).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanic, a native model of it, the two searches that model is solved with (an
exact distance-to-win field for four rooms of five, a beam for the fifth), the
proof of shortestness where there is one, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_the_nuevo_asylum",
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
exactly. Every expert step carries a set of equally-optimal presses -- the
complete one on levels 0-3, and on level 4 the complete set of equally-short
WALKS (see `_Beam`).

The game
--------
Five rooms of a sokoban whose crates are PAINT. Six crate colours exist and
three of them are mixtures:

    blue + yellow -> GREEN        yellow + red -> ORANGE        red + blue -> PURPLE

You shove crates around the way you shove them in any sokoban -- one at a time,
one square, nothing behind it -- with one extra rule: shoving crate X into crate
Y, when {X, Y} is one of those three pairs, ANNIHILATES both and leaves the
mixture standing on Y's square. Neither crate travels; the far square changes
colour and the near one empties, so a mixture can be made against a wall.

Each room is locked by one to three coloured DOORS, and the win condition is
``no target`` -- every door sits on a Target, and a door is the only thing a
Target is ever under, so "open every door" is the goal. A door of colour C is
opened by standing on a KEY of colour C and pressing ACTION, which consumes, in
one press, the key, one crate of colour C, and one door of colour C together
with the Target beneath it.

Keys come from exactly two places:

  * a **blue** key is placed in the level, always underneath the blue crate, so
    the first move of the game is to shove that crate off its own key;
  * a **mixture** key appears WHEN THE MIXTURE IS MADE -- but only if a door of
    that colour is still on the board, and it appears UNDER the crate it came
    with. So making the green crate does not give you the green key so much as
    bury it, and the next press has to shove the new crate off it.

That last detail is the game: every mixture costs three presses beyond the
sokoban (make it, uncover the key, press ACTION), the mixture must be made on a
square the player can then walk to, and the crate that carries the key is also
the crate the ACTION will eat.

Two more rules that only the interpreter will tell you, both measured against
it in ``--fuzz``:

  * ``[ > player | gcrate bkey] -> [ > player | gcrate]``. Shoving a GREEN crate
    that is standing on a BLUE key DESTROYS that key -- and it destroys it even
    when the shove is refused, so a press that does nothing visible can make a
    level unwinnable. It exists because the third room has a blue key under
    the blue crate and no green door: mixing yellow into that blue crate would
    otherwise leave the blue key exposed under a green crate for free.
  * a mixture made on a square that already holds a key REPLACES that key when
    its own door still exists, and leaves it alone when it does not.

Levels are 7x9 to 13x13, with 1 to 6 crates and 1 to 3 doors. The last two have
exactly enough crates for the doors they carry -- two of each primary for three
mixtures on the last -- so nothing is spare and every crate is spoken for.

Level numbers below are the 0-based indices every report here prints, so the
five rooms are 0 to 4.

Expert solver
-------------
A NATIVE model of the mechanic above (`_Board`), and TWO tiers on top of it,
chosen per level because one of the five rooms does not fit the first. The
interpreter is not searched: it runs at ~2 900 presses/s, and level 3's
primitive state space alone passes three million.

`_Field` is built on the MACRO graph, and that choice is what makes it
affordable. A press is either a WALK -- which changes nothing but the player's
square -- or one of the handful of presses that changes the board: a shove, a
mixture, a key-destroying refused shove, an ACTION. So a state that matters is
"the board, plus where the player stands after the last press that changed it",
and an edge is "walk to the square you press from, then press", costing the walk
plus one. Level 3 has 300k boards and 828k such macro states, against more than
three million primitive ones -- and every macro edge is a press that does
something, where four of the five primitive presses are usually a walk.

TIER 1, levels 0-3: an EXACT distance-to-win field (`_Field`), in two layers:

  * **the macro layer.** Enumerate every macro state reachable from the level
    start, keeping the edges; then a backward DIJKSTRA (edges have costs, so
    not a BFS) from the states with no doors left labels every one of them with
    its true shortest presses-to-win, ``D``.
  * **the primitive layer.** For a board ``X``, run one multi-source Dijkstra
    over ``X``'s free squares, seeding each square you can press FROM with
    ``1 + D(state that press lands in)`` and paying 1 per walk step. That gives
    the true presses-to-win for EVERY square of that board in one sweep -- which
    is the plan (walk downhill), the optimal SETS (a walk is optimal iff it
    steps to ``d - 1``; a board-changing press is optimal iff ``1 + D`` equals
    ``d``) and the recovery answer, all measured and none inferred.

The argument that this is exact is one line: any press sequence splits into
walk runs separated by board-changing presses, a walk run inside a fixed board
can be no shorter than the distance between its ends, and the macro edge charges
exactly that distance. So no shortest primitive path is missed and none is
invented.

That costs 62 / 147 / 137 / 827 578 macro states on levels 0 to 3 -- and level 4
does not fit it. A states-only sweep (no edges, no snapshots) was still going at
18.1 million macro states after seventeen minutes and 2.9 GB, with its frontier
still two million wide and growing -- ``--space`` is that sweep, so the claim is
a measurement and re-measurable rather than an assertion. A field there, which
also has to hold ~6 edges per state and reverse them, would be tens of gigabytes
and most of an hour at every cold start, and every `parallelize_generator` shard
would pay it again.

TIER 2, level 4: a width-capped layered BEAM over the same macros (`_Beam`),
ordered by an admissible count of the presses the board still owes. Its plan is
a genuine WIN, replayed through the interpreter press by press like every other,
and its walk tie sets are exact; what it is NOT is proved shortest, and every
report here says so where that level appears. The evidence that the beam is not
merely lucky is level 3, where the exact field is affordable and says 63: the
beam returns 63 too, at every width and weight tried.

Shortest, and how that is known
-------------------------------
Every plan on levels 0-3 is provably shortest, and the proof is four
independent derivations agreeing:

  * this file's macro field;
  * a plain FORWARD BFS over PRIMITIVE presses on the same model, shortest by
    construction and sharing nothing with the macro layer, compared at EVERY
    reachable state rather than only along the plan -- distances and tie sets
    both (``--selfcheck`` pass 1). Its companion pass 2 is a Bellman certificate
    over the macro field, which is the half that reaches level 3 as well;
  * the same forward/backward sweeps driven by the REAL INTERPRETER -- no
    `_Board` call of any kind, successors from `PSEngine.step` and the goal test
    from `check_win` (``--engine``);
  * the whole reachable space enumerated on the interpreter and solved exactly
    (``--enumerate``), which additionally reports how many reachable boards are
    DEAD -- this game can be lost, see the green-crate rule above.

The last two are affordable on levels 0-2 only (931 / 4 205 / 8 557 primitive
states); level 3 is covered by the first two and by ``--fuzz``, and level 4 by
`_beamcheck` -- which states what it checks (the replay, the model/interpreter
agreement press by press, and every labelled tie re-derived) and what it cannot.

Recovery
--------
`supports_recovery` with the family's RESET-mode prefix, and ``epsilon = 0``.
The expert re-plans from the LIVE board -- a field lookup, or a fresh beam, at
whatever state the engine is in -- so a perturbed board is answered exactly and
with the true optimal set, which is the signal a policy needs after its own
error. That is what `supports_recovery` asks for and it is why the expert reads
the board rather than replaying a cached path.

Detours are nevertheless off, and the reason is cost rather than capability:
levels 3 and 4 are a 45-second field and a 3-minute beam, and the disk cache can
only serve the level START (that is the one state every seed and every process
plans from). An epsilon detour would ask for a board it cannot serve, so every
detour on those two levels would re-derive the whole thing. Recovery data comes
from the RESET prefix instead, whose end state IS the cached one.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variable is the frame rotation
(``rotation_k`` in 0..3) and its matching directional action remap. 5 levels x 4
rotations = 20 presentations. ``--symmetry`` measures it: every level's plan and
a seeded random walk replayed at all four, requiring every frame to be the exact
rotation of the unaugmented one.

Rendering
---------
Two sprites had to change, and ``--audit`` is what convicted them. Both are in
``data/puzzlescript_games/The_Nuevo_Asylum.txt``, with the reasoning repeated
there:

  * **the keys were invisible.** A key is a 2x2 blob on rows 1-2 and a stem on
    rows 3-4, which is exactly where the Player and a crate paint. So "player
    standing on the orange key" and "player standing on the purple key" and
    "player standing on nothing" were the SAME PICTURE at level 4's 4px cells (the
    13x13 board) --
    on the level that has three key colours and where standing on the right one
    is the whole move. The keys now also paint the four CORNER pixels, which the
    Player leaves transparent at every cell size the game renders at.
  * **the crates were opaque**, so the blue key under the blue crate (levels 0
    and 2) and every mixture key under its own fresh crate were invisible too.
    The crates now leave their four corners transparent, which is where the keys
    paint.

Target itself is still the background colour and is left that way: it is under a
door in every level, an ACTION removes a door and ITS OWN target in the same
press (the adapter's multi-bracket overlap preference, measured in
``--selfcheck``), so a bare Target never appears on any reachable board and
``--audit`` asserts the composition list it checks is the reachable one.

Usage (run from the repo root):
    python solvers/generate_the_nuevo_asylum_training.py --episodes 200 \
        --out data/training_multi_level/the_nuevo_asylum

    python solvers/generate_the_nuevo_asylum_training.py --plans      # level report
    python solvers/generate_the_nuevo_asylum_training.py --selfcheck  # model + optimality
    python solvers/generate_the_nuevo_asylum_training.py --fuzz       # model vs interpreter
    python solvers/generate_the_nuevo_asylum_training.py --engine     # interpreter proof
    python solvers/generate_the_nuevo_asylum_training.py --enumerate  # whole-space proof
    python solvers/generate_the_nuevo_asylum_training.py --space      # why level 4 is beamed
    python solvers/generate_the_nuevo_asylum_training.py --audit      # rendering
    python solvers/generate_the_nuevo_asylum_training.py --symmetry   # augmentation
"""

from __future__ import annotations

import heapq
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
from solvers.common.ps_astar import (PSAStarSolver, PSExpert, Plan,   # noqa: E402
                                     StateGraph, screen_action,
                                     snapshot, restore)

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "The_Nuevo_Asylum"

#: Engine directions, in the order ties are broken. ACTION is the key press and
#: is a real move; the four arrows are the walk and the shove.
DIRS: tuple[str, ...] = ("up", "down", "left", "right", "action")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: The four arrows, paired so ``di ^ 1`` is the opposite direction -- which is
#: how `_Field` turns "a crate at k shoved in direction di" into "the square the
#: player must stand on".
ACTION_DI = 4

#: Crate colours. B/Y/R are the primaries the levels ship; G/O/Q are the
#: mixtures, and only a mixture (or blue) has a door and a key.
B, G, Y, R, O, Q = range(6)
CRATE_NAMES = {B: "bcrate", G: "gcrate", Y: "ycrate", R: "rcrate",
               O: "ocrate", Q: "qcrate"}
KEY_NAMES = {B: "bkey", G: "gkey", O: "okey", Q: "qkey"}
DOOR_NAMES = {B: "bdoor", G: "gdoor", O: "odoor", Q: "qdoor"}
COLOUR_CH = {B: "b", G: "g", Y: "y", R: "r", O: "o", Q: "q"}

#: The three mixtures, both orders. A pair NOT in here is an ordinary sokoban
#: shove that the crate in front blocks.
FUSE = {(B, Y): G, (Y, B): G, (Y, R): O, (R, Y): O, (R, B): Q, (B, R): Q}

#: Disk cache of every level's start plan AND its optimal-action sets. Levels 3
#: and 4 are minutes of derivation, so this is load-bearing: without it every
#: `parallelize_generator` shard re-derives them. Delete the file to re-derive.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "the_nuevo_asylum_plans.json"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic.

    Cells are flat ``r * w + c`` indices. Walls never change and live here; a
    STATE is the four things a rule can move or destroy::

        (player cell, crates, keys, dmask)

    ``crates`` is a sorted tuple of ``(cell, colour)`` and ``keys`` a sorted
    tuple of ``(cell, colour)`` -- both tiny (at most six and three entries), and
    both INTERNED by `_Field` so the million states of a big level share a few
    thousand distinct tuples. ``dmask`` is a bitmask over `doorcells`, the
    level's fixed door squares: a door is only ever removed, never moved, and it
    takes the Target under it with it, so the mask is the whole of the win
    condition (``no target`` == ``dmask == 0``).

    THE MECHANIC, as one press, and every branch of it is measured against the
    interpreter in ``--fuzz``, whose coverage table is per branch:

      * an ARROW into the board edge, a wall or a live door: nothing happens at
        all -- no rule has any of those on a left-hand side, so the movement
        phase simply refuses the move;
      * an arrow onto an empty square (a key does not block, it is on its own
        collision layer): the player walks there;
      * an arrow into a crate X with a crate Y behind it and ``{X, Y}`` one of
        the three mixtures: the two ANNIHILATE, the mixture appears on Y's
        square, and the player walks onto X's. Nothing behind Y matters, so this
        works against a wall. If a door of the mixture's colour is still on the
        board, its KEY appears on that same square, replacing whatever key was
        there;
      * an arrow into a GREEN crate standing on a BLUE key: the key is destroyed
        -- whether or not the shove itself goes through;
      * an arrow into a crate with a free square behind it: an ordinary sokoban
        shove;
      * an arrow into a crate with anything else behind it: refused;
      * ACTION on a key of colour C, with a door of colour C on the board and
        (for green/orange/purple) a crate of colour C somewhere on it: the key,
        that door, its target and one crate of colour C are consumed. Blue has a
        second rule with no crate in it, so a blue key opens a blue door with or
        without a blue crate -- the crate is eaten when there is one.

    `resolve` names the branch and `step` applies it; splitting them is what
    lets ``--fuzz`` count coverage per branch instead of guessing it from a
    before/after diff, which cannot tell a refused shove from a walk into a wall.
    """

    __slots__ = ("h", "w", "n", "wall", "doorcells", "doorcolour", "edge",
                 "free", "frozen", "sig")

    def __init__(self, h: int, w: int, walls, doorcells):
        self.h, self.w = h, w
        self.n = h * w
        self.wall = frozenset(walls)
        #: ``((cell, colour), ...)`` in cell order; ``dmask`` bit i is door i.
        self.doorcells = tuple(doorcells)
        self.doorcolour = tuple(col for _cell, col in doorcells)
        self.free = tuple(i for i in range(self.n) if i not in self.wall)
        #: ``edge[cell][di]`` is the square reached by pressing ``DIRS[di]``, or
        #: -1 when that press cannot move the player at all (off the board or
        #: into a wall). Doors are NOT folded in: they come and go.
        self.edge = []
        for i in range(self.n):
            r, c = divmod(i, w)
            row = []
            for d in DIRS[:4]:
                dr, dc = _DELTA[d]
                nr, nc = r + dr, c + dc
                j = nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1
                row.append(-1 if j < 0 or j in self.wall else j)
            self.edge.append(tuple(row))
        #: Squares a crate can never leave, whatever else happens. A crate moves
        #: along an axis only when BOTH its neighbours on that axis are free --
        #: one for the player to stand on, one for the crate to land on -- and a
        #: WALL is the one obstacle that never goes away, so a square with a wall
        #: on one side of each axis freezes anything standing on it for good.
        #: `dead` is what this is for; see the argument there.
        self.frozen = frozenset(
            i for i in self.free
            if not ((self.edge[i][0] >= 0 and self.edge[i][1] >= 0)
                    or (self.edge[i][2] >= 0 and self.edge[i][3] >= 0)))
        self.sig = (h, w, tuple(sorted(self.wall)), self.doorcells)

    # -- the mechanic --------------------------------------------------------
    def doorset(self, dmask: int) -> set:
        return {cell for i, (cell, _c) in enumerate(self.doorcells)
                if (dmask >> i) & 1}

    def resolve(self, state, di: int) -> tuple:
        """``(branch, newstate)``. ``newstate`` is None when the press changes
        nothing at all, which every sweep here treats as a non-move."""
        p, crates, keys, dmask = state
        if di == ACTION_DI:
            return self._resolve_action(state)
        q = self.edge[p][di]
        if q < 0:
            return ("refused_wall", None)
        doors = self.doorset(dmask)
        if q in doors:
            return ("refused_door", None)
        cd = dict(crates)
        x = cd.get(q)
        if x is None:
            return ("walk", (q, crates, keys, dmask))
        q2 = self.edge[q][di]
        y = cd.get(q2) if q2 >= 0 else None
        if y is not None and (x, y) in FUSE:
            z = FUSE[(x, y)]
            kd = dict(keys)
            del cd[q]
            cd[q2] = z
            if z in self.doorcolour and any(
                    self.doorcolour[i] == z for i in range(len(self.doorcells))
                    if (dmask >> i) & 1):
                kd[q2] = z
                branch = "fuse_key"
            else:
                branch = "fuse_nokey"
            return (branch, (q, _pack(cd), _pack(kd), dmask))
        kd = dict(keys)
        killed = x == G and kd.get(q) == B
        if killed:
            del kd[q]
        blocked = q2 < 0 or q2 in doors or q2 in cd
        if blocked:
            if not killed:
                return ("refused_crate", None)
            return ("shove_blocked_killkey", (p, crates, _pack(kd), dmask))
        del cd[q]
        cd[q2] = x
        return ("shove_killkey" if killed else "shove",
                (q, _pack(cd), _pack(kd), dmask))

    def _resolve_action(self, state) -> tuple:
        p, crates, keys, dmask = state
        kd = dict(keys)
        kc = kd.get(p)
        if kc is None:
            return ("action_nokey", None)
        doors = [i for i in range(len(self.doorcells))
                 if (dmask >> i) & 1 and self.doorcolour[i] == kc]
        if not doors:
            return ("action_nodoor", None)
        if len(doors) > 1:                      # no shipped level has two
            raise _Ambiguous("two doors of one colour")
        mine = [cell for cell, col in crates if col == kc]
        if kc != B and not mine:
            return ("action_nocrate", None)
        if len(mine) > 1:                       # no shipped level reaches this
            raise _Ambiguous("two crates of one colour")
        del kd[p]
        cd = dict(crates)
        if mine:
            del cd[mine[0]]
        return (("action" if mine else "action_nocrate_blue"),
                (p, _pack(cd), _pack(kd), dmask & ~(1 << doors[0])))

    def step(self, state, di: int):
        return self.resolve(state, di)[1]

    def won(self, state) -> bool:
        return state[3] == 0

    # -- a SOUND death test --------------------------------------------------
    def dead(self, state) -> bool:
        """True only when this board provably cannot be won -- a counting
        argument with no geometry in it, so it never prunes a live state.

        Opening a door of colour C consumes, in one press, a key of colour C and
        (for the three mixtures) a crate of colour C. Neither can be conjured:

          * a BLUE key is only ever placed in the level. Nothing creates one, and
            the green-crate rule DESTROYS one, so a level with a blue door and no
            blue key on the board is over -- which is exactly the trap level 2
            is built around;
          * a mixture's crate and key are born together, out of one crate of each
            of its two primaries, and those two crates are consumed. So a mixture
            colour that has neither a crate nor a key still on the board needs a
            fresh fusion, and every such fusion competes for the same pile of
            blue / yellow / red crates.

        Counting that competition is exact for one fusion per colour -- green
        eats a blue and a yellow, orange a yellow and a red, purple a red and a
        blue -- and needing a SECOND fusion of one colour (possible: a fusion can
        overwrite another colour's key) only makes the demand larger, so
        requiring one apiece is a lower bound and the test stays sound.

        A FROZEN crate (see `_Board.frozen`) is worth nothing to either side of
        that count, and the reason is worth stating because it is not obvious. A
        mixture appears on the FAR square of the pair, so a frozen crate can only
        ever be the far one -- being the near one needs free squares on both
        sides of it, which is exactly what frozen means it does not have -- and
        the mixture that lands on it is frozen too, with the new key buried
        underneath it forever. So a frozen primary is not a primary, and a key
        under a frozen crate is not a key. Both are subtractions from the supply,
        which is the direction that keeps the test sound.

        It is also what makes `_Field`'s ambiguity refusal free. Level 4 is the
        only board that can put two crates of one colour on the map at once (mix
        green twice and both blues and both yellows are gone), and every such
        board fails this test with both reds stranded -- which ``--selfcheck``
        asserts rather than assumes.
        """
        _p, crates, keys, dmask = state
        live = {self.doorcolour[i] for i in range(len(self.doorcells))
                if (dmask >> i) & 1}
        if not live:
            return False
        cd = dict(crates)
        have_c: dict = {}
        prim: dict = {}
        for cell, col in crates:
            have_c[col] = have_c.get(col, 0) + 1
            if cell not in self.frozen:
                prim[col] = prim.get(col, 0) + 1
        #: A key under a crate that can never move again is buried for good.
        have_k = {col for cell, col in keys
                  if not (cell in self.frozen and cell in cd)}
        if B in live and B not in have_k:
            return True                      # a blue key is never created
        fuse_needed = {c for c in live if c != B
                       and (not have_c.get(c) or c not in have_k)}
        need_b = (G in fuse_needed) + (Q in fuse_needed)
        need_y = (G in fuse_needed) + (O in fuse_needed)
        need_r = (O in fuse_needed) + (Q in fuse_needed)
        return (prim.get(B, 0) < need_b or prim.get(Y, 0) < need_y
                or prim.get(R, 0) < need_r)


class _Ambiguous(Exception):
    """The interpreter would pick one of several equal matches by set-iteration
    order (which crate of a colour the ACTION eats). No shipped level puts two
    crates or two doors of one colour on the board at once, so rather than
    mirroring an order nothing can justify, the transition is refused and
    ``--fuzz`` / ``--enumerate`` report the count, which is 0."""


def _pack(d: dict) -> tuple:
    return tuple(sorted(d.items()))


# ---------------------------------------------------------------------------
# The exact distance-to-win field
# ---------------------------------------------------------------------------

_INF = 1 << 30


def _reach(board: _Board, p: int, crates, dmask) -> dict:
    """Walk distances from ``p`` over the board's free squares. Crates and live
    doors block; keys and targets do not (they are on their own collision layer
    and nothing stands on them)."""
    blocked = set(board.wall) | {c for c, _col in crates} | board.doorset(dmask)
    dist = {p: 0}
    queue = deque([p])
    edge = board.edge
    while queue:
        u = queue.popleft()
        du = dist[u] + 1
        for di in range(4):
            v = edge[u][di]
            if v < 0 or v in blocked or v in dist:
                continue
            dist[v] = du
            queue.append(v)
    return dist


def _press_cells(board: _Board, crates, keys, dmask):
    """``([(press-from cell, direction index, resulting state), ...], ambiguous)``
    -- every press on this board that changes it, wherever the player has to
    stand to make it.

    This is the whole macro alphabet: a board-changing arrow needs a crate in
    front of the player, so the square is fixed by the crate and the direction;
    an ACTION needs a key underfoot. `_Ambiguous` transitions are dropped and
    counted -- see `_Board.dead` for why that is free."""
    out = []
    amb = 0
    for k, _col in crates:
        for di in range(4):
            back = board.edge[k][di ^ 1]        # the square to press from
            if back < 0:
                continue
            try:
                _br, nxt = board.resolve((back, crates, keys, dmask), di)
            except _Ambiguous:
                amb += 1
                continue
            if nxt is not None:
                out.append((back, di, nxt))
    for kc, _col in keys:
        try:
            _br, nxt = board.resolve((kc, crates, keys, dmask), ACTION_DI)
        except _Ambiguous:
            amb += 1
            continue
        if nxt is not None:
            out.append((kc, ACTION_DI, nxt))
    return out, amb


def _macro_edges(board: _Board, state):
    """``([(cost, press-from cell, direction, successor), ...], ambiguous)`` --
    one BFS over the board's free squares, then every press it puts in reach."""
    p, crates, keys, dmask = state
    reach = _reach(board, p, crates, dmask)
    cells, amb = _press_cells(board, crates, keys, dmask)
    out = []
    for cell, di, nxt in cells:
        step = reach.get(cell)
        if step is None:                        # not in the player's region
            continue
        out.append((step + 1, cell, di, nxt))
    return out, amb


class _Field:
    """Exact presses-to-win for every state of one level, in two layers.

    LAYER 1, the macro graph. A press is either a WALK (only the player's square
    changes) or one of the few presses that change the board. So the states that
    have to be enumerated are the boards paired with where the player stands
    after the last board-changing press -- ``(cell, crates, keys, dmask)`` -- and
    an edge is "walk to the square you press from, then press", costing
    ``walk + 1``. `_macros` builds the edges out of one breadth-first sweep over
    the board's free squares; `_enumerate` walks the whole reachable component
    keeping them, and `_solve` runs a backward DIJKSTRA (the edges have costs, so
    a BFS is not enough) from every state with no doors left. The result, ``D``,
    is the true shortest presses-to-win for every macro state.

    LAYER 2, the primitive field. `cellfield` takes a BOARD and returns the true
    presses-to-win for every square of it: seed each square that can be pressed
    FROM with ``1 + D(what that press lands in)``, then a multi-source Dijkstra
    with unit walk steps. One sweep per board answers every player position on
    it, which is what makes the plan, the tie sets and the recovery lookups all
    the same operation.

    WHY IT IS EXACT. Any press sequence splits into walk runs separated by
    board-changing presses. A walk run inside a fixed board is at least the grid
    distance between its ends, and the macro edge charges exactly that, so the
    macro graph neither misses a shortest path nor invents one. Both layers are
    checked against a primitive forward BFS in ``--selfcheck`` and against the
    interpreter in ``--engine``.

    ``cap`` is a runaway guard on layer 1, not a budget: past it the field
    reports itself CAPPED and the expert returns no plan, which `record_level`
    turns into a skipped level rather than a wrong one.
    """

    __slots__ = ("board", "start", "cap", "capped", "ambiguous", "pruned",
                 "index", "states", "eu", "ev", "ec", "dist", "_cells",
                 "_intern")

    def __init__(self, board: _Board, start, cap: int = 4_000_000):
        self.board = board
        self.start = start
        self.cap = cap
        self.capped = False
        self.ambiguous = 0
        self.pruned = 0
        self.index: dict = {}
        self.states: list = []
        self.eu = array("i")
        self.ev = array("i")
        self.ec = array("i")
        self.dist: array = array("i")
        self._cells: dict = {}
        self._intern: dict = {}
        self._enumerate()
        if not self.capped:
            self._solve()

    # -- layer 1 -------------------------------------------------------------
    def _keep(self, state):
        """Intern the two tuple components. A big level holds a few thousand
        distinct crate/key tuples across a million states; sharing them is the
        difference between ~700 bytes a state and ~150."""
        p, crates, keys, dmask = state
        it = self._intern
        crates = it.setdefault(crates, crates)
        keys = it.setdefault(keys, keys)
        return (p, crates, keys, dmask)

    def _macros(self, state):
        """``[(cost, successor), ...]`` from one macro state."""
        edges, amb = _macro_edges(self.board, state)
        self.ambiguous += amb
        return [(cost, nxt) for cost, _cell, _di, nxt in edges]

    def _enumerate(self) -> None:
        board = self.board
        start = self._keep(self.start)
        self.index[start] = 0
        self.states.append(start)
        queue = deque([0])
        eu, ev, ec = self.eu, self.ev, self.ec
        while queue:
            ui = queue.popleft()
            state = self.states[ui]
            if board.won(state):
                continue
            for cost, nxt in self._macros(state):
                if board.dead(nxt):
                    self.pruned += 1
                    continue
                nxt = self._keep(nxt)
                vi = self.index.get(nxt)
                if vi is None:
                    if len(self.states) >= self.cap:
                        self.capped = True
                        return
                    vi = len(self.states)
                    self.index[nxt] = vi
                    self.states.append(nxt)
                    queue.append(vi)
                eu.append(ui)
                ev.append(vi)
                ec.append(cost)

    def _solve(self) -> None:
        """Backward Dijkstra from every won state over the reversed macro edges.

        The reverse adjacency is a CSR pair of `array`s rather than a list of
        lists: a million empty lists costs more than the edges themselves (the
        ps:ouroboros lesson)."""
        n = len(self.states)
        m = len(self.eu)
        offs = array("i", [0]) * (n + 1)
        for v in self.ev:
            offs[v + 1] += 1
        for i in range(1, n + 1):
            offs[i] += offs[i - 1]
        radj = array("i", [0]) * m if m else array("i")
        rcost = array("i", [0]) * m if m else array("i")
        cursor = array("i", offs)
        for e in range(m):
            v = self.ev[e]
            pos = cursor[v]
            radj[pos] = self.eu[e]
            rcost[pos] = self.ec[e]
            cursor[v] = pos + 1
        dist = array("i", [_INF]) * n if n else array("i")
        heap = []
        board = self.board
        for i, st in enumerate(self.states):
            if board.won(st):
                dist[i] = 0
                heap.append((0, i))
        heapq.heapify(heap)
        while heap:
            d, v = heapq.heappop(heap)
            if d > dist[v]:
                continue
            for e in range(offs[v], offs[v + 1]):
                u = radj[e]
                nd = d + rcost[e]
                if nd < dist[u]:
                    dist[u] = nd
                    heapq.heappush(heap, (nd, u))
        self.dist = dist

    # -- layer 2 -------------------------------------------------------------
    def _d(self, state) -> int:
        """Presses-to-win for a macro state, or `_INF` when it cannot win."""
        if self.board.won(state):
            return 0
        i = self.index.get(self._keep(state))
        return _INF if i is None else self.dist[i]

    def cellfield(self, crates, keys, dmask) -> dict:
        """``{cell: presses-to-win}`` for every square of this board, exact.

        Cached by board, because the plan walks many primitive steps across the
        same one and every step asks for it."""
        key = (crates, keys, dmask)
        got = self._cells.get(key)
        if got is not None:
            return got
        board = self.board
        blocked = set(board.wall) | {c for c, _col in crates} | board.doorset(dmask)
        src: dict = {}
        cells, amb = _press_cells(board, crates, keys, dmask)
        self.ambiguous += amb
        for cell, _di, nxt in cells:
            if cell in blocked:                    # cannot stand there
                continue
            val = self._d(nxt)
            if val >= _INF:
                continue
            if src.get(cell, _INF) > val + 1:
                src[cell] = val + 1
        dist: dict = {}
        heap = [(v, c) for c, v in src.items()]
        heapq.heapify(heap)
        edge = board.edge
        while heap:
            d, u = heapq.heappop(heap)
            if u in dist:
                continue
            dist[u] = d
            for di in range(4):
                v = edge[u][di]
                if v < 0 or v in blocked or v in dist:
                    continue
                heapq.heappush(heap, (d + 1, v))
        self._cells[key] = dist
        return dist

    # -- the answers ---------------------------------------------------------
    def get(self, state) -> "int | None":
        """Presses to a win from any primitive state, or None when this board can
        never be won from there."""
        if self.capped:
            return None
        if self.board.won(state):
            return 0
        p, crates, keys, dmask = state
        return self.cellfield(crates, keys, dmask).get(p)

    def optimal(self, state) -> list:
        """``[(direction index, successor), ...]``: every press on a shortest
        path from ``state``, in ``DIRS`` order.

        Exact, with nothing inferred. A WALK is optimal iff the square it steps
        to is one press nearer; any other press is optimal iff ``1 + D`` of what
        it lands in equals the distance here. Both numbers are the field's."""
        rest = self.get(state)
        if not rest:                    # None (dead / capped) or 0 (already won)
            return []
        board = self.board
        p, crates, keys, dmask = state
        here = self.cellfield(crates, keys, dmask)
        out = []
        for di in range(5):
            try:
                branch, nxt = board.resolve(state, di)
            except _Ambiguous:
                continue
            if nxt is None:
                continue
            if branch == "walk":
                if here.get(nxt[0], _INF) == rest - 1:
                    out.append((di, nxt))
            elif self._d(nxt) + 1 == rest:
                out.append((di, nxt))
        return out

    def plan(self, state) -> "Plan | None":
        """The shortest press sequence from ``state``, with the EXACT optimal set
        at every step, or None if this board cannot be won from here."""
        rest = self.get(state)
        if rest is None:
            return None
        presses, optsets = [], []
        cur = state
        for _ in range(rest):
            best = self.optimal(cur)
            if not best:                # a labelled state always has a step
                return None
            presses.append(DIRS[best[0][0]])
            optsets.append([DIRS[di] for di, _ in best])
            cur = best[0][1]
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# The second tier: a beam, for the one level whose field does not fit
# ---------------------------------------------------------------------------

class _Beam:
    """A width-capped layered beam over the same macros, ordered by PROGRESS.

    Level 4 is the one room `_Field` cannot have: a states-only sweep was still
    going at 18.1 million macro states after seventeen minutes and 2.9 GB, with
    its frontier still two million wide and growing (``--space`` is that sweep,
    so re-measure rather than trust the number), and a field there -- which also
    has to hold and reverse ~6 edges per state -- would be tens of gigabytes and
    most of an hour every time a shard cold-starts. So that level gets a search
    instead, and the honest consequence is stated everywhere it matters: its plan
    is a genuine WIN, replayed through the interpreter like every other, but it
    is NOT proved shortest.

    The ordering is `_work`, an admissible count of the presses the board still
    owes -- one ACTION per door, plus the mixture press and the shove that
    uncovers its key, plus the shoves that bring each pair together -- with the
    cost so far added. Ordering a beam by PROGRESS rather than by a heuristic's
    absolute value is the ps:idols_to_the_burnt_god lesson; `_work` is far short
    of the truth over the middle of a level (it has no walking in it at all), and
    a weighted A* on it burns nodes without arriving.

    The evidence that the beam is not simply wrong is level 3, where the exact
    field is affordable and says 63: the beam returns 63 at every width and
    weight tried, which is what ``--selfcheck`` re-measures.

    The shipped configuration is the cheapest one that reaches the best answer a
    sweep found. Measured on level 4, cost by (weight, width)::

        width      1 000    5 000   25 000
        weight 1   no win     138      126
        weight 2   no win     140      126
        weight 4   no win     138      132

    so widths 5 000 and 25 000 at weight 1 -- about three minutes, and the plan
    then lives in ``plan_cache_path`` for every later process. Both widths are
    run rather than only the larger, because a beam is a sample rather than a
    refinement: a narrower one is not guaranteed to be worse, and the incumbent
    the cheap run finds also prunes the expensive one.

    The per-step optimal SETS are exact WITHIN the macro sequence and inferred
    across it, and the difference matters. A macro is "walk to a square, then
    press": a walk changes nothing but where the player stands, so every
    interleaving of a shortest route to that square leaves an identical board and
    every one of them is equally right -- that part is measured, off a BFS from
    the press square. What is not known is whether a DIFFERENT macro sequence is
    equally short, so the press steps are labelled with themselves alone.
    """

    __slots__ = ("board", "widths", "weights", "layers")

    def __init__(self, board: _Board, widths=(5_000, 25_000),
                 weights=(1,), layers: int = 100):
        self.board = board
        self.widths = widths
        self.weights = weights
        self.layers = layers

    def _run(self, state, width: int, weight: int, bound: int = _INF):
        """The cheapest macro path this (width, weight) finds, as
        ``[(press-from cell, direction, successor), ...]``, or None.

        ``bound`` is the best cost any earlier run reached: a node already past
        it cannot lead anywhere useful, so passing it forward is what makes the
        wide run cheaper than it would be on its own."""
        board = self.board
        back: dict = {state: None}
        layer = {state: 0}
        seen = {state}
        best = None
        best_cost = bound
        for _ in range(self.layers):
            nxt: dict = {}
            for st, g in layer.items():
                edges, _amb = _macro_edges(board, st)
                for cost, cell, di, ns in edges:
                    if board.dead(ns):
                        continue
                    ng = g + cost
                    if board.won(ns):
                        if ng < best_cost:
                            best_cost = ng
                            best = (st, cell, di, ns)
                        continue
                    if ns in seen or ng >= best_cost:
                        continue
                    if ns not in nxt or nxt[ns][0] > ng:
                        nxt[ns] = (ng, st, cell, di)
            if not nxt:
                break
            items = sorted(nxt.items(),
                           key=lambda kv: _work(board, kv[0]) * weight + kv[1][0])
            items = items[:width]
            layer = {}
            for ns, (ng, st, cell, di) in items:
                back[ns] = (st, cell, di)
                layer[ns] = ng
                seen.add(ns)
        if best is None:
            return None
        st, cell, di, ns = best
        path = [(cell, di, ns)]
        while st is not None and back.get(st) is not None:
            prev, pcell, pdi = back[st]
            path.append((pcell, pdi, st))
            st = prev
        path.reverse()
        return path

    def plan(self, state) -> "Plan | None":
        """The best plan any of the configured (width, weight) pairs finds,
        expanded into primitive presses with their tie sets.

        The pairs are tried cheapest-first and all of them are run, because a
        wider beam is not monotonically better -- it is a different sample of the
        same space -- and the whole point of the tier is to take the shortest
        answer available rather than the first. Each run is handed the best cost
        so far as a bound, so the cheap runs pay for the expensive ones."""
        best = None
        bound = _INF
        for width in self.widths:
            for weight in self.weights:
                path = self._run(state, width, weight, bound)
                if path is None:
                    continue
                presses, optsets = self._expand(state, path)
                if best is None or len(presses) < len(best[0]):
                    best = (presses, optsets)
                    bound = len(presses)
        return None if best is None else Plan(best[0], best[1])

    def _expand(self, state, path):
        """A macro path as primitive presses, with the exact walk tie sets."""
        board = self.board
        presses, optsets = [], []
        cur = state
        for cell, di, nxt in path:
            p, crates, keys, dmask = cur
            to_cell = _reach(board, cell, crates, dmask)   # distances TO the press square
            here = p
            while here != cell:
                d_here = to_cell[here]
                tie = []
                for wd in range(4):
                    v = board.edge[here][wd]
                    if v >= 0 and to_cell.get(v, _INF) == d_here - 1:
                        tie.append(DIRS[wd])
                presses.append(tie[0])
                optsets.append(tie)
                here = board.edge[here][DIRS.index(tie[0])]
            presses.append(DIRS[di])
            optsets.append([DIRS[di]])
            cur = nxt
        return presses, optsets


def _work(board: _Board, state) -> int:
    """An ADMISSIBLE count of the presses this board still owes.

    Every term is a press no other term can be: one ACTION per live door; the
    mixture press and the shove that uncovers its key, for each door whose key
    is not already lying uncovered; and the shoves that bring each mixture's two
    primaries into contact, which cannot be fewer than their Manhattan distance
    less one (a shove moves one crate one square). The distance terms take the
    MAXIMUM rather than the sum, because two mixtures can share a primary colour
    and the same shove could serve both.

    It has no walking in it, so it is far below the truth -- it is a beam
    ordering, not a bound worth proving anything with.
    """
    _p, crates, keys, dmask = state
    live = [board.doorcolour[i] for i in range(len(board.doorcells))
            if (dmask >> i) & 1]
    if not live:
        return 0
    cd = dict(crates)
    by_colour: dict = {}
    for cell, col in crates:
        by_colour.setdefault(col, []).append(cell)
    kcells = {col: cell for cell, col in keys}
    total = 0
    far = 0
    for c in live:
        total += 1                                   # the ACTION press
        key_cell = kcells.get(c)
        if c == B:
            if key_cell is None:
                return _INF
            total += key_cell in cd
            continue
        if key_cell is not None and by_colour.get(c):
            total += key_cell in cd
            continue
        a, b = _PRIMARIES[c]
        best = _INF
        for ca in by_colour.get(a, ()):
            for cb in by_colour.get(b, ()):
                best = min(best, abs(ca // board.w - cb // board.w)
                           + abs(ca % board.w - cb % board.w) - 1)
        if best >= _INF:
            return _INF
        total += 2                                   # the mixture and the shove
        far = max(far, best)
    return total + far


#: Which two primaries each mixture is made of.
_PRIMARIES = {G: (B, Y), O: (Y, R), Q: (R, B)}



# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class NuevoAsylumExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Field`.

    The base class keeps the memo, the level scoping and the disk cache; only the
    strategy underneath changes, the way `PSEnumExpert` replaces it. Here the
    strategy is "read the live board and walk down the exact distance field", so
    `heuristic` is never called and asserts rather than returning a number
    nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state is answered exactly -- and can honestly answer NO: this game
    is losable (shove the green crate off the blue key and level 2 is over), so a
    None from here is a real verdict and not a search giving up. ``--enumerate``
    counts how many reachable boards are in that condition.
    """

    directions = list(DIRS)
    #: `_key` is the dynamic objects only, which is canonical WITHIN a level but
    #: not across them -- walls and door squares are static per level and differ
    #: between levels.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    #: Runaway guard on one level's macro enumeration. Levels 0-2 need tens to
    #: hundreds of states and level 3 needs 828k; nothing here should ever reach
    #: this. See ``--plans`` for the measured counts.
    field_cap: int = 4_000_000

    #: Levels solved by `_Beam` instead of `_Field`. Only level 4, whose macro
    #: space does not enumerate -- see `_Beam` for the measurement and for what
    #: is given up. A set rather than a size rule because it is a MEASURED fact
    #: about one board, not a guess from its dimensions.
    beam_levels = frozenset({4})

    def setup(self) -> None:
        g = self.g
        self.player_ids = set(self.game._engine._player_indices)
        self.wall_ids = (set(g.resolve_object_name("wall"))
                         | set(g.resolve_object_name("wall2")))
        self.crate_ids = {i: col for col, name in CRATE_NAMES.items()
                          for i in g.resolve_object_name(name)}
        self.key_ids = {i: col for col, name in KEY_NAMES.items()
                        for i in g.resolve_object_name(name)}
        self.door_ids = {i: col for col, name in DOOR_NAMES.items()
                         for i in g.resolve_object_name(name)}
        self.target_ids = set(g.resolve_object_name("target"))
        #: `_Board`s and `_Field`s by STATIC signature (see `read`), not by level
        #: index, so a board is shared by every state of its level and the field
        #: is kept across queries.
        self._boards: dict = {}
        self._fields: dict = {}
        self._beams: dict = {}
        self._level: "int | None" = None

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "NuevoAsylumExpert reads an exact distance field; heuristic is unused")

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, state)`` for the engine's current grid.

        A `_Board` is keyed on its WALLS ALONE, not on the doors: opening a door
        deletes it from the grid, so a signature that carried the live doors
        would make every ACTION press look like a different level and grow a
        second field for it. The door SQUARES are fixed (a door is only ever
        removed, never moved), so the board records the ones present the first
        time this level is read -- which is the level start, because
        `NuevoAsylumSolver.prepare_expert` seats and reads every level before
        anything else asks -- and the live grid then only decides ``dmask``."""
        h, w = eng.height, eng.width
        walls, crates, keys = [], {}, {}
        doors: dict = {}
        player = None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                for o in cell:
                    if o in self.crate_ids:
                        crates[i] = self.crate_ids[o]
                    elif o in self.key_ids:
                        keys[i] = self.key_ids[o]
                    elif o in self.door_ids:
                        doors[i] = self.door_ids[o]
                    elif o in self.player_ids:
                        player = i
        sig = (h, w, tuple(walls))
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(h, w, walls,
                                               sorted(doors.items()))
        dmask = 0
        for i, (cell, _col) in enumerate(board.doorcells):
            if cell in doors:
                dmask |= 1 << i
        return board, (player, _pack(crates), _pack(keys), dmask)

    def field(self, board: _Board, state) -> _Field:
        """The level's field, built from ``state`` the first time and kept.

        Keyed by the board SIGNATURE (walls plus the level's door squares), so
        every query about that level -- the start, a recovery board, an
        `--selfcheck` perturbation -- reuses the one field. ``state`` therefore
        only matters on the first call, and `NuevoAsylumSolver.prepare_expert`
        makes that call from the level start."""
        key = board.sig
        got = self._fields.get(key)
        if got is None:
            got = self._fields[key] = _Field(board, state, self.field_cap)
        return got

    def _key(self, eng) -> frozenset:
        """``(row, col, tag)`` triples for everything a rule can move or destroy.

        Built from the MODEL's reading rather than from raw object ids so the key
        is exactly what a state consists of -- which is also what the
        ``plan_cache_path`` signature is stored as, so a cached plan is matched
        against the board it was solved from and an edited level is a miss rather
        than a wrong plan."""
        board, (p, crates, keys, dmask) = self.read(eng)
        w = board.w
        out = set()
        if p is not None:
            out.add((p // w, p % w, 0))
        for cell, col in crates:
            out.add((cell // w, cell % w, 1 + col))
        for cell, col in keys:
            out.add((cell // w, cell % w, 10 + col))
        for i, (cell, col) in enumerate(board.doorcells):
            if (dmask >> i) & 1:
                out.add((cell // w, cell % w, 20 + col))
        return frozenset(out)

    def plan(self, eng, level: int | None = None):
        """Remember which level is being asked about, then defer to the base.

        `_search` gets no level argument and the tier is a per-level fact, so it
        is recorded here -- the same two-line override ps:crocodiles_love_cookies
        uses to keep its report honest."""
        self._level = level
        return super().plan(eng, level)

    def beam(self, board: _Board) -> _Beam:
        got = self._beams.get(board.sig)
        if got is None:
            got = self._beams[board.sig] = _Beam(board)
        return got

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if state[0] is None:                 # no player on the board
            return None
        if board.won(state):
            return Plan([], [])
        if self._level in self.beam_levels:
            return self.beam(board).plan(state)
        return self.field(board, state).plan(state)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class NuevoAsylumSolver(PSAStarSolver):
    game_id = "puzzlescript_the_nuevo_asylum"
    game_name = GAME_NAME
    expert_cls = NuevoAsylumExpert

    #: `games/ps:the_nuevo_asylum/ps:the_nuevo_asylum.py` is a plain passthrough
    #: -- it builds the adapter and nothing else, and the two sprite fixes are in
    #: the .txt, which both paths read. Set this if that wrapper grows a patch.
    game_module_id = ""

    #: Unused: `NuevoAsylumExpert._search` never calls `_astar`. Left at the base
    #: values so nothing reads a lie off them; `NuevoAsylumExpert.field_cap` is
    #: the knob that actually bounds the derivation.
    weight = 1
    node_cap = 400_000

    #: Room for the longest plan plus the presses an exploration prefix costs.
    #: The adapter's own 200-press per-level budget is separate and is reset by
    #: the `set_level` that ends the prefix, so the plan starts it from zero --
    #: see ``--plans``, which prints each level's headroom against it.
    max_steps = 220

    #: Every level is recorded; nothing is skipped. Level 4 is solved by the
    #: second tier -- see `_Beam` and the module docstring for what that costs.
    skip_levels = frozenset()

    #: Detours are off: the field for the two big levels is a multi-minute
    #: derivation and the disk cache only keeps the level START, so a detour
    #: would ask for a board the cache cannot serve and every shard would rebuild
    #: the field. Recovery data comes from the RESET prefix, whose end state IS
    #: the cached one. See the module docstring.
    epsilon = 0.0

    def prepare_expert(self, game, expert) -> None:
        """Build every level's field before `discover_solvable` asks for it --
        the same work either way, but it fills the disk cache in one pass and
        makes the startup cost visible as startup."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
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
    solver = NuevoAsylumSolver()
    game = solver.make_game(seed)
    return solver, game, NuevoAsylumExpert(game, node_cap=solver.node_cap)


def _ascii(board: _Board, state) -> str:
    """A model state as ASCII, so a failing check can print the board it failed
    on rather than a tuple of integers."""
    p, crates, keys, dmask = state
    cd, kd = dict(crates), dict(keys)
    doors = {cell: col for i, (cell, col) in enumerate(board.doorcells)
             if (dmask >> i) & 1}
    out = []
    for r in range(board.h):
        line = ""
        for c in range(board.w):
            i = r * board.w + c
            if i in board.wall:
                ch = "#"
            elif i in doors:
                ch = COLOUR_CH[doors[i]].upper()
            elif i in cd:
                ch = COLOUR_CH[cd[i]]
            elif i == p:
                ch = "P"
            elif i in kd:
                ch = COLOUR_CH[kd[i]] + "!"
            else:
                ch = "."
            if i == p and i in cd:
                ch = "?"                 # impossible; shows up if it happens
            if i == p and i in kd:
                ch = COLOUR_CH[kd[i]].upper() + "@"
            line += ch.ljust(2)
        out.append(line)
    return "\n".join(out)


def _start(game, expert, level: int):
    """``(board, state)`` at a level's start, engine left seated there."""
    game.set_level(level)
    return expert.read(game._engine)


def _plans(levels=None, verbose: bool = True) -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the field and the tie labelling at
    once. Also the report that shows what each level costs to derive -- with one
    caveat about the timing: it is the time `PSExpert.plan` took, so it is ~0
    whenever ``plan_cache_path`` is warm. Delete ``data/the_nuevo_asylum_
    plans.json`` to see the cold numbers (about 45 s for the level 3 field and
    two minutes for the level 4 beam); the macro-state counts beside them are
    re-derived either way."""
    _solver, game, expert = _new()
    eng = game._engine
    levels = list(range(game.n_levels)) if levels is None else list(levels)
    total = ties = bad = 0
    for level in levels:
        board, state = _start(game, expert, level)
        t0 = time.time()
        beamed = level in expert.beam_levels
        plan = expert.plan(eng, level)
        secs = time.time() - t0
        if beamed:
            tier = "beam  (NOT proved shortest)"
        else:
            field = expert.field(board, state)
            tier = (f"field {len(field.states):7d} macro states / "
                    f"{len(field.eu):8d} edges, shortest")
        ncr = len(state[1])
        ndoor = len(board.doorcells)
        if plan is None:
            print(f"  L{level:2d}: UNSOLVED  ({tier})")
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
                  f"{len(board.free):3d} floor  {ncr} crates  {ndoor} doors  "
                  f"{tier} in {secs:6.1f}s  ->  {len(plan):3d} presses  "
                  f"win={won}  (budget {game._max_steps}, {room})  "
                  f"{tie_steps:3d} steps with a tie set")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print(f"  (times are `PSExpert.plan`, so ~0 with {PLAN_CACHE.name} warm)")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# Model <-> interpreter
# ---------------------------------------------------------------------------

#: Every branch `_Board.resolve` can take. ``--fuzz`` prints the table and fails
#: if a branch a shipped level can reach was never exercised.
BRANCHES = ("refused_wall", "refused_door", "walk", "fuse_key", "fuse_nokey",
            "shove", "shove_killkey", "shove_blocked_killkey", "refused_crate",
            "action", "action_nocrate_blue", "action_nokey", "action_nodoor",
            "action_nocrate")


def _seat(eng, expert, board: _Board, state) -> None:
    """Write a model state onto the engine's grid.

    This is what lets ``--fuzz`` compare the model against the interpreter on
    states the level's own play never reaches -- every reachable board of levels
    0-2, and a random sample of the two big ones -- rather than only on the
    boards a random walk happens to find. Doors carry their Target, which is the
    invariant the fuzz then re-checks after every press."""
    idx = expert.g.obj_name_to_idx
    p, crates, keys, dmask = state
    bg = idx["background"]
    w = board.w
    eng.height, eng.width = board.h, board.w
    grid = [[{bg} for _ in range(board.w)] for _ in range(board.h)]
    for cell in board.wall:
        grid[cell // w][cell % w] |= {idx["wall"], idx["wall2"]}
    for i, (cell, col) in enumerate(board.doorcells):
        if (dmask >> i) & 1:
            grid[cell // w][cell % w] |= {idx[DOOR_NAMES[col]], idx["target"]}
    for cell, col in crates:
        grid[cell // w][cell % w].add(idx[CRATE_NAMES[col]])
    for cell, col in keys:
        grid[cell // w][cell % w].add(idx[KEY_NAMES[col]])
    if p is not None:
        grid[p // w][p % w].add(idx["player"])
    eng.grid = grid
    eng._position_index_dirty = True


def _primitive_space(board: _Board, start, cap: int = 400_000):
    """Every primitive state reachable from ``start``, with its shortest
    distance-to-win, by forward BFS then backward BFS over the collected edges.

    Shares nothing with `_Field` but `_Board.resolve` itself, which is the point:
    it is the independent answer every optimality claim below is checked
    against, and it uses no macro decomposition at all. Returns
    ``(dist, succ, capped)``; states absent from ``dist`` cannot win."""
    succ: dict = {}
    seen = {start}
    queue = deque([start])
    capped = False
    while queue:
        cur = queue.popleft()
        if board.won(cur):
            continue
        edges = {}
        for di in range(5):
            try:
                _br, nxt = board.resolve(cur, di)
            except _Ambiguous:
                continue
            if nxt is None:
                continue
            edges[di] = nxt
            if nxt not in seen:
                if len(seen) >= cap:
                    capped = True
                    queue.clear()
                    break
                seen.add(nxt)
                queue.append(nxt)
        succ[cur] = edges
    rev: dict = {}
    dist: dict = {}
    frontier = []
    for k, edges in succ.items():
        for _di, nxt in edges.items():
            if board.won(nxt):
                if k not in dist:
                    dist[k] = 1
                    frontier.append(k)
            else:
                rev.setdefault(nxt, []).append(k)
    for st in seen:
        if board.won(st):
            dist[st] = 0
    queue = deque(frontier)
    while queue:
        k = queue.popleft()
        for prev in rev.get(k, ()):
            if prev not in dist:
                dist[prev] = dist[k] + 1
                queue.append(prev)
    return dist, succ, capped


#: Levels whose whole primitive space fits comfortably, so the checks that need
#: it (the exhaustive fuzz, the primitive-BFS optimality proof, the interpreter
#: sweeps) run on them. Levels 3 and 4 are covered by the sampled fuzz, the
#: Bellman certificate and the plan replay instead. Derived, not assumed:
#: ``--selfcheck`` prints the state counts it measured.
_SMALL = (0, 1, 2)


def _fuzz(samples: int = 900, verbose: bool = True) -> int:
    """Differential test of `_Board` against the real interpreter.

    Levels 0-2 are EXHAUSTIVE: every reachable primitive state is seated on the
    engine and every one of the five presses is compared, board and win flag.
    Levels 3-4 sample reachable states two ways, because neither alone is
    enough:

      * seeded random walks from a COLD level start, never from a warmed-up one,
        because turn-one bookkeeping is exactly the class of mechanic a warm-up
        hides (the ps:idols_to_the_burnt_god lesson);
      * random walks branching off every PREFIX of the level's own plan, because
        a mixture on those two boards is twenty-odd presses deep and a walk from
        the start reaches one about never (the ps:escaping_limbo lesson -- fuzz
        from the solution's own prefixes or the report is about the shallow half
        of the game).

    Three things are checked on every pair:

      * the resulting board, read back through `NuevoAsylumExpert.read`, equals
        the model's successor -- and a press the model refuses must leave the
        interpreter's board untouched;
      * ``eng.check_win()`` equals `_Board.won`;
      * every Target square carries a door and every door square carries a
        Target. That invariant is why the audit does not have to make Target
        visible, and it is the one an ACTION could break -- the rule's
        ``[Gdoor][Gcrate][target]`` brackets match independently, and only the
        adapter's overlap preference pairs the target with its own door.

    The per-branch coverage table is printed PER LEVEL: a differential fuzz that
    reports "model matches" having never exercised a mixture is worth nothing,
    and a total that hides which level did the exercising is nearly as bad.
    """
    _solver, game, expert = _new()
    eng = game._engine
    counts = {b: 0 for b in BRANCHES}
    per_level: dict = {}
    bad = 0
    for level in range(game.n_levels):
        board, start = _start(game, expert, level)
        mine = {b: 0 for b in BRANCHES}
        if level in _SMALL:
            _dist, succ, _capped = _primitive_space(board, start)
            states = list(succ.keys())
            mode = f"exhaustive ({len(states)} states)"
        else:
            rng = random.Random(9000 + level)
            plan = expert.plan(eng, level)
            game.set_level(level)
            spine = [start]
            cur = start
            for direction in (plan or ()):
                cur = board.step(cur, DIRS.index(direction))
                spine.append(cur)
            states = _walk_sample(board, spine, samples, rng)
            mode = (f"sampled ({len(states)} states off the start and off all "
                    f"{len(spine)} states of the plan)")
        checked = 0
        for state in states:
            for di in range(5):
                try:
                    branch, nxt = board.resolve(state, di)
                except _Ambiguous:
                    bad += 1
                    print(f"    L{level} AMBIGUOUS at\n{_ascii(board, state)}")
                    continue
                counts[branch] += 1
                mine[branch] += 1
                _seat(eng, expert, board, state)
                before = [[set(c) for c in row] for row in eng.grid]
                eng.step(DIRS[di])
                _b2, got = expert.read(eng)
                want = state if nxt is None else nxt
                checked += 1
                if got != want:
                    bad += 1
                    if bad < 6:
                        print(f"    L{level} MISMATCH press={DIRS[di]} "
                              f"branch={branch}\n  model wanted:\n"
                              f"{_ascii(board, want)}\n  interpreter gave:\n"
                              f"{_ascii(board, got)}\n  from:\n"
                              f"{_ascii(board, state)}")
                if nxt is None and any(
                        eng.grid[r][c] != before[r][c]
                        for r in range(board.h) for c in range(board.w)):
                    bad += 1
                    print(f"    L{level} a REFUSED press changed the board "
                          f"({DIRS[di]}, {branch})")
                if eng.check_win() != board.won(want):
                    bad += 1
                    print(f"    L{level} win flag disagrees ({DIRS[di]})")
                bad += _door_target_invariant(eng, expert, level)
        per_level[level] = mine
        if verbose:
            print(f"  L{level}: {mode}, {checked} (state, press) pairs")
    if verbose:
        print("  branch coverage (per level, then the total):")
        head = "  ".join(f"L{lvl}" for lvl in sorted(per_level))
        print(f"    {'':26s} {head:>34s}      total")
        for b in BRANCHES:
            cells = "  ".join(f"{per_level[lvl][b]:6d}" for lvl in sorted(per_level))
            print(f"    {b:26s} {cells}  {counts[b]:9d}"
                  + ("   NEVER EXERCISED" if not counts[b] else ""))
    missed = [b for b in BRANCHES if not counts[b]]
    if missed:
        print(f"  {len(missed)} branch(es) no shipped level reaches: "
              f"{', '.join(missed)} -- see _witnesses()")
        bad += _witnesses(missed, verbose)
    return bad


def _door_target_invariant(eng, expert, level: int) -> int:
    """0 when every Target square is a door square and vice versa."""
    doors, targets = set(), set()
    w = eng.width
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            i = r * w + c
            for o in cell:
                if o in expert.door_ids:
                    doors.add(i)
                elif o in expert.target_ids:
                    targets.add(i)
    if doors == targets:
        return 0
    print(f"    L{level} door/target split: doors={sorted(doors)} "
          f"targets={sorted(targets)}")
    return 1


def _walk_sample(board: _Board, seeds, n: int, rng: random.Random) -> list:
    """``n`` distinct reachable states out of seeded random walks from each of
    ``seeds``.

    Walks restart from a seed rather than continuing, so the sample is a spread
    over the space instead of one long trajectory, and the seed list is the level
    start plus every state its plan passes through -- which is how the deep half
    of a big level (anything after the first mixture) gets sampled at all."""
    out, seen = [], set()
    seeds = list(seeds)
    while len(out) < n:
        cur = seeds[rng.randrange(len(seeds))]
        for _ in range(rng.randint(1, 25)):
            if cur not in seen:
                seen.add(cur)
                out.append(cur)
                if len(out) >= n:
                    break
            opts = []
            for di in range(5):
                try:
                    _br, nxt = board.resolve(cur, di)
                except _Ambiguous:
                    continue
                if nxt is not None:
                    opts.append(nxt)
            if not opts:
                break
            cur = rng.choice(opts)
    return out


def _witnesses(missed, verbose: bool = True) -> int:
    """Purpose-built boards for the branches no shipped level can reach.

    "The fuzz did not cover it" and "no level can reach it" are different facts,
    and only these boards separate them. Both missing branches are structural
    rather than accidental: a key of colour C is only ever CREATED while a door
    of colour C is on the board, it is created together with its crate, and the
    ACTION that spends it spends the door and the crate in the same press -- so
    a key that outlives its door (``action_nodoor``) or its crate
    (``action_nocrate``) needs two keys of one colour, which needs two mixtures
    of one colour, which `_Board.dead` proves is a lost board. The witnesses
    below build the stack directly and are compared against the interpreter
    exactly the way the fuzz compares a reachable one.
    """
    _solver, game, expert = _new()
    eng = game._engine
    #: 5x5 rooms: a ring of wall, the player on a key in the middle, one crate,
    #: and one door in the top-right corner. ``(name, door colour, crate colour,
    #: key colour under the player)``.
    plans = {
        # a green key with no green door: the blue door cannot hear it
        "action_nodoor": (B, G, G),
        # a green key and a green door, but the green crate is gone
        "action_nocrate": (G, B, G),
    }
    bad = 0
    for name in missed:
        spec = plans.get(name)
        if spec is None:
            print(f"    no witness board for {name}")
            bad += 1
            continue
        door_col, crate_col, key_col = spec
        h = w = 5
        walls = [i for i in range(h * w)
                 if i // w in (0, h - 1) or i % w in (0, w - 1)]
        door_cell = w - 1                       # the top-right corner
        walls.remove(door_cell)
        board = _Board(h, w, walls, [(door_cell, door_col)])
        player = 2 * w + 2
        state = (player, ((2 * w + 1, crate_col),), ((player, key_col),), 1)
        branch, nxt = board.resolve(state, ACTION_DI)
        _seat(eng, expert, board, state)
        eng.step("action")
        _b, got = expert.read(eng)
        want = state if nxt is None else nxt
        ok = got == want and branch == name
        if verbose:
            print(f"    witness {name}: branch={branch} "
                  f"interpreter agrees={got == want}")
        if not ok:
            print(f"      wanted\n{_ascii(board, want)}\n      got\n"
                  f"{_ascii(board, got)}")
        bad += not ok
    return bad


# ---------------------------------------------------------------------------
# Optimality
# ---------------------------------------------------------------------------

def _selfcheck(levels=None, sample: int = 4000,
               verbose: bool = True) -> int:
    """Four passes, none of which trusts `_Field`'s own arithmetic.

    **Pass 1 -- the macro field against a PRIMITIVE BFS**, on the levels whose
    whole primitive space fits. `_primitive_space` walks every one of the five
    presses from every reachable state and solves the resulting graph backwards;
    it shares nothing with the macro decomposition. Every state's distance AND
    its optimal-press set are compared, not only the ones on the plan -- so this
    is also the check on the argument the macro layer rests on (that a walk run
    inside a fixed board costs the grid distance and nothing else).

    **Pass 2 -- a BELLMAN certificate** over the macro field, on every level it
    was built for, including the one pass 1 cannot afford. Two halves, because a
    fixpoint over the wrong graph is still a fixpoint: for EVERY macro state
    ``D(u)`` must equal ``min(cost + D(v))`` over its stored edges and be 0
    exactly at the states with no doors left; and for a sample of states (all of
    them on a small level) the stored edge list must equal what `_Board`
    re-derives now. The transitions underneath are what ``--fuzz`` measures
    against the interpreter, so the three together cover model, edges and
    solve.

    **Pass 3 -- the plan walks downhill.** Stepping the plan through `_Board`,
    the field's distance must fall by exactly one at every single press and the
    last one must land on a won board. That is the join between the two layers:
    every step reads `cellfield` at a state the plan reached by pressing, so a
    primitive field that disagreed with the macro distances it was seeded from
    would show up as a step that did not fall.

    **Pass 4 -- recovery.** From boards knocked off the plan by random presses,
    `_Field` must either answer None (and the board really is lost -- pass 1's
    primitive field says so on the small levels) or answer a plan that the REAL
    INTERPRETER replays to a win.
    """
    _solver, game, expert = _new()
    eng = game._engine
    levels = list(range(game.n_levels)) if levels is None else list(levels)
    bad = 0

    for level in levels:
        board, start = _start(game, expert, level)
        if level in expert.beam_levels:
            bad += _beamcheck(game, expert, board, start, level, verbose)
            continue
        field = expert.field(board, start)
        if field.capped:
            print(f"  L{level}: field CAPPED at {len(field.states)} macro "
                  f"states -- skipped")
            bad += 1
            continue

        # -- pass 1 ----------------------------------------------------------
        if level in _SMALL:
            dist, succ, capped = _primitive_space(board, start)
            wrong_d = wrong_sets = 0
            for st in succ:
                mine = field.get(st)
                theirs = dist.get(st)
                if (mine if mine is not None else -1) != (
                        theirs if theirs is not None else -1):
                    wrong_d += 1
                if theirs:
                    want = {di for di, nxt in succ[st].items()
                            if (0 if board.won(nxt) else dist.get(nxt, -1))
                            == theirs - 1}
                    got = {di for di, _n in field.optimal(st)}
                    if want != got:
                        wrong_sets += 1
            bad += wrong_d + wrong_sets
            if verbose:
                print(f"  L{level} pass 1: {len(succ)} primitive states, "
                      f"capped={capped}, {wrong_d} distance and {wrong_sets} "
                      f"tie-set disagreements, "
                      f"{sum(1 for st in succ if st not in dist)} of them DEAD")
        elif verbose:
            print(f"  L{level} pass 1: skipped (primitive space too large)")

        # -- pass 2 ----------------------------------------------------------
        n, m = len(field.states), len(field.eu)
        best = array("i", [_INF]) * n
        for e in range(m):
            dv = field.dist[field.ev[e]]
            if dv >= _INF:
                continue
            u = field.eu[e]
            if dv + field.ec[e] < best[u]:
                best[u] = dv + field.ec[e]
        wrong = 0
        for i, st in enumerate(field.states):
            want = 0 if board.won(st) else best[i]
            if field.dist[i] != want:
                wrong += 1
        # ... and the edges themselves, re-derived from `_Board` on a sample
        # (all of it on a small level), because a fixpoint over the wrong graph
        # is still a fixpoint.
        rng = random.Random(31 + level)
        picks = (range(n) if n <= sample
                 else rng.sample(range(n), sample))
        stored: dict = {}
        for e in range(m):
            stored.setdefault(field.eu[e], []).append(
                (field.ec[e], field.ev[e]))
        redrawn = 0
        for i in picks:
            st = field.states[i]
            if board.won(st):
                continue
            mine = sorted((cost, field.index[field._keep(nxt)])
                          for cost, nxt in field._macros(st)
                          if not board.dead(nxt))
            if mine != sorted(stored.get(i, [])):
                redrawn += 1
        bad += wrong + redrawn
        if verbose:
            print(f"  L{level} pass 2: {n} macro states, {m} edges, {wrong} "
                  f"Bellman violations; {len(picks)} of them re-derived off "
                  f"`_Board` with {redrawn} edge-set disagreements")

        # -- pass 3 ----------------------------------------------------------
        plan = field.plan(start)
        if plan is None:
            print(f"  L{level} pass 3: NO PLAN")
            bad += 1
            continue
        cur = start
        drop = 0
        for direction in plan:
            here = field.get(cur)
            nxt = board.step(cur, DIRS.index(direction))
            there = 0 if board.won(nxt) else field.get(nxt)
            if there is None or here != there + 1:
                drop += 1
            cur = nxt
        bad += drop + (not board.won(cur))
        if verbose:
            print(f"  L{level} pass 3: {len(plan)} presses, {drop} steps that "
                  f"did not fall by exactly one, ends won={board.won(cur)}")

        # -- pass 4 ----------------------------------------------------------
        rng = random.Random(4242 + level)
        live = dead = replayed = 0
        for _ in range(60):
            game.set_level(level)
            cur = start
            for _ in range(rng.randint(1, 25)):
                opts = []
                for di in range(5):
                    try:
                        _br, nxt = board.resolve(cur, di)
                    except _Ambiguous:
                        continue
                    if nxt is not None:
                        opts.append((di, nxt))
                if not opts:
                    break
                di, cur = rng.choice(opts)
                eng.step(DIRS[di])
            got = field.plan(cur)
            if got is None:
                dead += 1
                continue
            live += 1
            for direction in got:
                eng.step(direction)
            if eng.check_win():
                replayed += 1
            else:
                bad += 1
        if verbose:
            print(f"  L{level} pass 4: 60 perturbed boards, {live} still "
                  f"winnable ({replayed} replayed to a WIN on the "
                  f"interpreter), {dead} genuinely lost")
    return bad


def _beamcheck(game, expert, board: _Board, start, level: int,
               verbose: bool = True) -> int:
    """What can be checked about a beam level, stated as what it is.

    There is no field to certify here, so this checks the three things that do
    not need one:

      * the plan REPLAYS to a win on the real interpreter, press by press, and
        fits inside the adapter's per-level budget;
      * every press the model says the plan takes is the press the interpreter
        takes -- the plan is expanded from macros, so a walk step that turned out
        to be a shove would silently change the board;
      * every labelled ALTERNATIVE really is a tie: it must be a walk (the board
        unchanged but for the player) that lands one square nearer the SAME press
        square the plan is walking to. That is exactly the claim the labels make,
        and it is re-derived here from a fresh breadth-first sweep and from the
        plan's own press squares, neither of which the expansion handed over.

    What it cannot check is the one thing that matters most, and the report says
    so rather than implying otherwise: whether a different macro sequence is
    shorter.
    """
    eng = game._engine
    plan = expert.plan(eng, level)
    if plan is None:
        print(f"  L{level}: the beam found NO plan")
        return 1
    game.set_level(level)
    bad = 0

    # Walk the plan once on the model to get the square each press is taken
    # from, and which of those presses change the board.
    where, changing = [], []
    cur = start
    for direction in plan:
        di = DIRS.index(direction)
        branch, nxt = board.resolve(cur, di)
        if nxt is None:
            print(f"    L{level} step {len(where)} ({direction}) does nothing "
                  f"({branch})")
            return bad + 1
        where.append((cur, branch))
        changing.append(branch != "walk")
        cur = nxt
    #: the press square each step is walking towards
    target = [None] * len(plan)
    nxt_press = None
    for i in range(len(plan) - 1, -1, -1):
        if changing[i]:
            nxt_press = where[i][0][0]
        target[i] = nxt_press

    cur = start
    ties = alts = 0
    for pi, direction in enumerate(plan):
        opts = plan.optsets[pi]
        if len(opts) > 1:
            ties += 1
            alts += len(opts) - 1
            goal = target[pi]
            to_goal = _reach(board, goal, cur[1], cur[3])
            for alt in opts:
                abr, anx = board.resolve(cur, DIRS.index(alt))
                if (abr != "walk" or anx[1:] != cur[1:]
                        or to_goal.get(anx[0], _INF) != to_goal[cur[0]] - 1):
                    bad += 1
                    print(f"    L{level} step {pi}: {alt} is labelled optimal "
                          f"but is a {abr} that does not step nearer the press "
                          f"square")
        eng.step(direction)
        _b, got = expert.read(eng)
        want = board.step(cur, DIRS.index(direction))
        if got != want:
            bad += 1
            print(f"    L{level} step {pi}: the interpreter and the model part "
                  f"ways\n{_ascii(board, want)}\n{_ascii(board, got)}")
            break
        cur = want
    won = eng.check_win()
    bad += not won
    if verbose:
        print(f"  L{level} beam tier: {len(plan)} presses "
              f"({sum(changing)} of them change the board), replayed to "
              f"win={won} on the interpreter (budget {game._max_steps}), "
              f"{ties} steps with a tie set ({alts} alternatives, every one "
              f"re-derived to be a walk one square nearer the same press "
              f"square). NOT proved shortest -- see `_Beam`.")
    return bad


def _space(levels=(3, 4), report_every: int = 250_000, cap: int = 40_000_000,
           verbose: bool = True) -> int:
    """Measure how big a level's macro space is, which is the evidence behind
    `NuevoAsylumExpert.beam_levels`.

    It counts states only -- no edges, no snapshots -- so it answers "would the
    field fit" for a fraction of what the field itself would cost, and it prints
    the frontier as it goes so a space that is still growing says so instead of
    looking like a slow one that is about to finish. Always 0 violations; it is
    a measurement, not a check.
    """
    import resource                                                # noqa: PLC0415
    _solver, game, expert = _new()
    for level in levels:
        board, start = _start(game, expert, level)
        seen = {start}
        queue = deque([start])
        intern: dict = {}
        expanded = 0
        t0 = time.time()
        while queue:
            st = queue.popleft()
            expanded += 1
            if board.won(st):
                continue
            edges, _amb = _macro_edges(board, st)
            for _cost, _cell, _di, nxt in edges:
                if board.dead(nxt):
                    continue
                p, crates, keys, dmask = nxt
                nxt = (p, intern.setdefault(crates, crates),
                       intern.setdefault(keys, keys), dmask)
                if nxt not in seen:
                    if len(seen) >= cap:
                        queue.clear()
                        break
                    seen.add(nxt)
                    queue.append(nxt)
            if verbose and expanded % report_every == 0:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
                print(f"    L{level}: {expanded} expanded, {len(seen)} seen, "
                      f"{len(queue)} queued, {time.time() - t0:.0f}s, {rss} MB",
                      flush=True)
        print(f"  L{level}: {len(seen)} macro states, "
              f"{'EXHAUSTED' if not queue else 'CAPPED'}, "
              f"{time.time() - t0:.0f}s")
    return 0

# ---------------------------------------------------------------------------
# The interpreter, on its own
# ---------------------------------------------------------------------------

def _engine_key(eng, expert):
    """A canonical key for an interpreter board, built without `_Board`."""
    bg = expert.bg_id
    return frozenset((r, c, o)
                     for r, row in enumerate(eng.grid)
                     for c, cell in enumerate(row)
                     for o in cell if o != bg)


def _engine(levels=_SMALL, verbose: bool = True) -> int:
    """Forward BFS then backward BFS driven by the REAL INTERPRETER.

    No `_Board` call of any kind participates: the successor of a board is
    whatever `PSEngine.step` makes of it and the goal test is `check_win`. It
    re-derives the shortest length AND the per-step optimal set for each level
    and compares them against `_Field`, so the whole native model is under test
    rather than the plan it happened to produce.
    """
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in levels:
        board, start = _start(game, expert, level)
        field = expert.field(board, start)
        snaps = {}
        k0 = _engine_key(eng, expert)
        snaps[k0] = snapshot(eng)
        seen = {k0}
        succ: dict = {}
        queue = deque([k0])
        steps = 0
        while queue:
            k = queue.popleft()
            edges = {}
            for direction in DIRS:
                restore(eng, snaps[k])
                eng.step(direction)
                steps += 1
                if eng.check_win():
                    edges[direction] = "WIN"
                    continue
                nk = _engine_key(eng, expert)
                if nk == k:
                    continue
                if nk not in seen:
                    seen.add(nk)
                    snaps[nk] = snapshot(eng)
                    queue.append(nk)
                edges[direction] = nk
            succ[k] = edges
        rev: dict = {}
        dist: dict = {}
        frontier = []
        for k, edges in succ.items():
            for _d, nk in edges.items():
                if nk == "WIN":
                    if k not in dist:
                        dist[k] = 1
                        frontier.append(k)
                else:
                    rev.setdefault(nk, []).append(k)
        queue = deque(frontier)
        while queue:
            k = queue.popleft()
            for prev in rev.get(k, ()):
                if prev not in dist:
                    dist[prev] = dist[k] + 1
                    queue.append(prev)
        # walk the interpreter's own gradient and compare, step by step
        plan = field.plan(start)
        mine = len(plan) if plan is not None else None
        theirs = dist.get(k0)
        bad += mine != theirs
        tie_bad = 0
        k = k0
        model_state = start
        for pi in range(theirs or 0):
            want = set()
            for direction in DIRS:
                nk = succ[k].get(direction)
                if nk is None:
                    continue
                cost = 1 if nk == "WIN" else (
                    dist[nk] + 1 if nk in dist else None)
                if cost == dist[k]:
                    want.add(direction)
            got = {DIRS[di] for di, _n in field.optimal(model_state)}
            if want != got:
                tie_bad += 1
            chosen = plan[pi]
            model_state = board.step(model_state, DIRS.index(chosen))
            k = succ[k][chosen]
            if k == "WIN":
                break
        bad += tie_bad
        restore(eng, snaps[k0])
        if verbose:
            print(f"  L{level}: {len(seen)} interpreter states, {steps} "
                  f"`eng.step`s, d*={theirs} (model says {mine}), "
                  f"{tie_bad} tie-set disagreements, "
                  f"{len(seen) - len(dist)} states that can never win")
    return bad


def _enumerate(levels=_SMALL, verbose: bool = True) -> int:
    """The whole reachable space through the shared `StateGraph`, a fourth
    independent derivation -- and the report that establishes what ``--audit``
    is allowed to assume.

    The state key here is the RAW grid (every non-background object id, walls
    included), not `NuevoAsylumExpert._key`, on purpose: the model's key folds a
    door and the Target under it into one tag, and the two checks below are
    precisely about those being the same square. Besides the shortest length it
    checks, over EVERY reachable board rather than over a sample:

      * every Target square carries a door and every door square carries a
        Target;
      * the set of cell COMPOSITIONS a board of this game can hold is inside
        `_COMPOSITIONS`, which is what makes the audit's list a claim rather than
        a hope.
    """
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    seen_comps: set = set()
    for level in levels:
        board, start = _start(game, expert, level)
        field = expert.field(board, start)
        graph = StateGraph.build(eng, lambda e: _engine_key(e, expert),
                                 list(DIRS), node_cap=200_000)
        if graph is None:
            print(f"  L{level}: over the enumeration cap")
            bad += 1
            continue
        plan = graph.plan(list(DIRS))
        mine = field.plan(start)
        theirs = len(plan) if plan is not None else None
        bad += (len(mine) if mine is not None else None) != theirs
        split = 0
        for k in graph.succ:
            comps = _cell_stacks(expert, k)
            doors = {rc for rc, names in comps.items()
                     if any(n.endswith("door") for n in names)}
            targets = {rc for rc, names in comps.items() if "target" in names}
            if doors != targets:
                split += 1
            for names in comps.values():
                seen_comps.add(tuple(sorted(names)))
        bad += split
        seen_comps.add(())
        if verbose:
            print(f"  L{level}: {len(graph.succ)} reachable states, d*={theirs} "
                  f"(model says {len(mine) if mine else None}), "
                  f"{len(graph.succ) - len(graph.dist)} of them can never win, "
                  f"{split} with a door and its target apart")
    extra = seen_comps - set(_COMPOSITIONS)
    missing = set(_COMPOSITIONS) - seen_comps
    if verbose:
        print(f"  reachable cell compositions: {len(seen_comps)}; "
              f"{len(extra)} not in the audit list, {len(missing)} listed but "
              f"never seen")
    for c in sorted(extra):
        print(f"    NOT AUDITED: {_comp_name(c)}")
    bad += len(extra)
    return bad


def _cell_stacks(expert, key) -> dict:
    """``{(row, col): {object name, ...}}`` for a raw interpreter state key."""
    names = expert.g.obj_idx_to_name
    out: dict = {}
    for (r, c, o) in key:
        out.setdefault((r, c), set()).add(names.get(o, str(o)))
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

#: Every cell stack a board of this game can hold. A SUPERSET of what is
#: reachable, deliberately: it is every combination the collision layers permit
#: on the two layers that can share a square -- keys and Target underneath,
#: player / crates / doors on top -- so an audit run cannot pass by having
#: forgotten a stack. ``--enumerate`` walks the whole reachable space of levels
#: 0-2 on the interpreter and fails on anything it sees that is NOT listed here;
#: it also prints how many of these no level reaches, which is the honest half
#: of the same report.
#:
#: Two families are absent and both are consequences of that walk rather than of
#: an argument: a bare ``target`` and a bare ``door`` never occur, because a door
#: and the target under it are removed by the same press (the adapter's
#: multi-bracket overlap preference); and nothing ever stands on a door, because
#: a door blocks the whole of its collision layer.
_COMPOSITIONS = (
    [(), ("wall", "wall2"), ("player",)]
    + [(CRATE_NAMES[c],) for c in (B, G, Y, R, O, Q)]
    + [(KEY_NAMES[c],) for c in (B, G, O, Q)]
    + [tuple(sorted((DOOR_NAMES[c], "target"))) for c in (B, G, O, Q)]
    + [tuple(sorted((KEY_NAMES[k], "player"))) for k in (B, G, O, Q)]
    + [tuple(sorted((CRATE_NAMES[c], KEY_NAMES[k])))
       for c in (B, G, Y, R, O, Q) for k in (B, G, O, Q)]
)


def _comp_name(comp) -> str:
    return "+".join(comp) or "floor"


def _audit(verbose: bool = True) -> int:
    """Two passes over every reachable cell COMPOSITION.

    **Pass 1 -- distinctness**, at every cell size the five boards use (7, 5 and
    4 px; the size list is derived from the levels rather than assumed, so an
    edited level is covered). Whole 64x64 frames of uniform boards are compared
    rather than one cell out of a mixed board: `_render_frame` upscales and
    centre-pads, so slicing by ``cell_px`` arithmetic reads the wrong pixels (the
    ps:explod lesson).

    This is the pass that fails on the shipped .txt, and it fails hardest at 4px
    -- the size level 4 renders at -- where ``player`` and all four
    ``key + player`` stacks were one picture. See the header comment in
    ``data/puzzlescript_games/The_Nuevo_Asylum.txt``.

    **Pass 2 -- the group.** The game is augmented with a frame rotation, so a
    board is drawn at any of 4 presentations and the sprites are not all
    invariant under them (the Player is a little person, the key a key). That is
    fine -- the whole frame is turned together, so a turned board is a real board
    -- as long as no rotation of one composition lands on a DIFFERENT one, which
    is what this checks. It runs on SQUARE boards, because a rotation of a
    non-square board also moves the letterbox and every comparison would pass for
    the wrong reason.
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

    for cell in sorted({min(64 // h, 64 // w) for h, w in sizes}):
        n = 64 // cell
        shots = {c: shoot(n, n, c) for c in _COMPOSITIONS}
        clashes = [(a, b, k) for a, b in itertools.permutations(_COMPOSITIONS, 2)
                   for k in range(4)
                   if np.array_equal(np.ascontiguousarray(
                       np.rot90(shots[a], k=k)), shots[b])]
        invariant = [_comp_name(c) for c in _COMPOSITIONS
                     if all(np.array_equal(np.ascontiguousarray(
                         np.rot90(shots[c], k=k)), shots[c]) for k in range(4))]
        bad += len(clashes)
        if verbose:
            print(f"  cell {cell}px ({n}x{n}): {len(clashes)} composition(s) "
                  f"whose rotation is another's art; rotation-invariant: "
                  f"{', '.join(invariant)}")
        for a, b, k in clashes:
            print(f"      {_comp_name(a)} rotated {k * 90} == {_comp_name(b)}")
    return bad


def _symmetry(levels=None, walk_presses: int = 300,
              verbose: bool = True) -> int:
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the rotation of the unaugmented ones.

    This is the evidence for the rotation augmentation, measured rather than
    argued. Two stages, because a plan-only replay can be blind (the ps:
    lovendpieces lesson -- no SHORTEST plan need traverse an order-sensitive
    transition):

      * each level's own plan, replayed at all four rotations;
      * a seeded random walk that shoves crates into walls, mixes colours, opens
        doors and presses ACTION on nothing, replayed the same way.

    ``screen_action`` is what turns an engine direction into the button an agent
    presses in the augmented view, so this also tests the remap the recorder
    uses.
    """
    solver, game, expert = _new()
    levels = list(range(game.n_levels)) if levels is None else list(levels)
    bad = 0
    for level in levels:
        board, start = _start(game, expert, level)
        plan = expert.plan(game._engine, level)
        if plan is None:
            continue
        rng = random.Random(77 + level)
        walk = []
        cur = start
        for _ in range(walk_presses):
            opts = []
            for di in range(5):
                try:
                    _br, nxt = board.resolve(cur, di)
                except _Ambiguous:
                    continue
                opts.append((di, nxt if nxt is not None else cur))
            di, cur = rng.choice(opts)
            walk.append(DIRS[di])
            if board.won(cur):
                break
        seen_k: set = set()
        for name, presses in (("plan", list(plan)), ("walk", walk)):
            canon_by_k: dict = {}
            for seed in range(64):
                game._seed = seed
                game.set_level(level)
                k, hf, vf = game._rotation_k, game._hflip, game._vflip
                seen_k.add((k, hf, vf))
                if (k, hf, vf) in canon_by_k:
                    if len(canon_by_k) == 4:
                        break
                    continue
                frames = [np.asarray(game._current_frame)]
                for direction in presses:
                    act = screen_action(direction, k, hf, vf,
                                        solver.remap_actions)
                    fd = game.perform_action(ActionInput(id=act))
                    frames.append(np.asarray(fd.frame[-1] if fd.frame
                                             else game._current_frame))
                    if game._state == GameState.WIN:
                        break
                canon_by_k[(k, hf, vf)] = [np.rot90(f, k=-k) for f in frames]
            ref_key = min(canon_by_k)
            ref = canon_by_k[ref_key]
            for pres, canon in sorted(canon_by_k.items()):
                if pres == ref_key:
                    continue
                if len(canon) != len(ref) or any(
                        not np.array_equal(ref[i], canon[i])
                        for i in range(len(ref))):
                    bad += 1
                    print(f"    L{level} {name}: presentation {pres} is not the "
                          f"rotation of {ref_key}'s frames")
        if verbose:
            print(f"  L{level}: plan ({len(plan)}) and a {len(walk)}-press "
                  f"random walk replayed at {len(seen_k)} presentation(s) "
                  f"{sorted(seen_k)}")
    return bad


if __name__ == "__main__":
    if "--plans" in sys.argv:
        args = [int(a) for a in sys.argv[sys.argv.index("--plans") + 1:]
                if a.isdigit()]
        sys.exit(_plans(args or None))
    if "--fuzz" in sys.argv:
        violations = _fuzz()
        print(f"fuzz: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--selfcheck" in sys.argv:
        args = [int(a) for a in sys.argv[sys.argv.index("--selfcheck") + 1:]
                if a.isdigit()]
        violations = _selfcheck(args or None)
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--engine" in sys.argv:
        violations = _engine()
        print(f"engine sweeps: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--enumerate" in sys.argv:
        violations = _enumerate()
        print(f"whole-space enumeration: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--space" in sys.argv:
        args = [int(a) for a in sys.argv[sys.argv.index("--space") + 1:]
                if a.isdigit()]
        sys.exit(_space(args or (3, 4)))
    if "--audit" in sys.argv:
        collisions = _audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        args = [int(a) for a in sys.argv[sys.argv.index("--symmetry") + 1:]
                if a.isdigit()]
        violations = _symmetry(args or None)
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    sys.exit(NuevoAsylumSolver.main())
