"""Generate Phase-1 training data for the PuzzleScript game ps:vrps
("VRPS" -- Virtual Reality Puzzle Script -- by Jack Lance).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the `BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
measured mechanic, a native model of it (pieces AND art), the two exact searches
that model is driven with, the optimal-action labelling, and the reports that
back all of it.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_vrps",
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
action (post rotation/mirror remap), i.e. the button an agent presses in the
presented view, so replaying the recorded actions reproduces the recorded frames
exactly -- which ``--replay`` checks, frame for frame, on a freshly built
adapter. Every expert step carries the full set of equally-optimal presses.

All EIGHT levels are solved, in 224 presses, and every one of them is PROVABLY
SHORTEST -- not "the best the search found", but the distance in a space that
was enumerated (or swept by layers) to exhaustion:

    level  0   8 presses        10 states           exact field
    level  1  18 presses     16314 states           exact field
    level  2  14 presses     12893 states           exact field
    level  3  28 presses    170161 states           exact field
    level  4  31 presses      6568 states           exact field
    level  5  32 presses   6191437 states           exact field
    level  6  30 presses      2792 states           exact field
    level  7  63 presses  89878656 states           layered sweep

THE GAME: you are wearing a VR headset, and the room you are standing in is
also a puzzle
------------------------------------------------------------------------------
Every level is TWO boards side by side on one grid. On the left is the game the
headset is showing you -- grass, brick walls, crates, goal squares, scooters and
your AVATAR. On the right is your living room, four squares by four, with the
PLAYER (you, a dark figure seen from above) standing in it. The win condition is
``all goal on crate``: an ordinary sokoban, on the left-hand board.

The joke, and the whole mechanic, is that only ONE of those two boards is the
one you are looking at:

* **The avatar copies you.** ``[moving player][avatar] -> [moving player][moving
  avatar]``. One key press moves both, the same way.
* **A real wall cancels the turn.** ``[> player | wall] -> cancel``. Walk toward
  the edge of your living room and NOTHING happens -- not to you, not to the
  avatar, not to a crate you were leaning on. The VR world is huge; the room is
  four squares across; so the avatar cannot simply walk to the far side of the
  level, and every level is built on that.
* **A brick freezes you.** ``[stationary avatar][player] -> [stationary
  avatar][stationary player]``. The coupling runs both ways: an avatar stopped
  by a wall it can see stops the body it cannot.
* **A SCOOTER is how you cross the room without crossing the room.**
  ``up[up avatar up scooterup][up player] -> [...][stationary player]``. Stand on
  a scooter that points the way you are pressing and the avatar (and the
  scooter, which travels under it) moves while the player stays exactly where it
  is. That is the only way to spend a press on the VR board without spending a
  square of living room, and a scooter only works in the ONE direction it points.
* **The warnings are the headset being polite.** For each side of the room the
  player is standing against, a tick-mark is drawn on the matching side of the
  avatar -- the guardian boundary, drawn into the game world. It is pure art:
  the rules that paint it read nothing and change nothing, and the player is
  visible in the room anyway.

THE INVARIANT THAT IS THE PUZZLE
--------------------------------
Write ``f = avatar_col - player_col``. A rightward WALK moves both bodies, so f
is unchanged; a rightward RIDE moves only the avatar, so f goes up by one; a
leftward ride would take it down. So **f changes only by rides, and only in the
direction of a scooter that exists in the level.** A level with no scooterleft
has a monotonically non-decreasing f, and since the player is trapped in four
columns, that is a hard, one-way budget on how far right the avatar can ever
have travelled. The same holds per axis for rows.

Level 7 is that observation as a level: it ships one scooterdown and one
scooterright, so BOTH offsets are one-way, and the two crates have to be walked
to two goals on opposite corners in an order that never spends a budget it will
need again. Its shortest solution is 63 presses, and its reachable space is over
89 million states.

That budget very nearly proves level 7 UNWINNABLE, and the way it fails is worth
writing down. Completing the goal at (3,9) leaves the avatar in row 3, so the
row offset is low; completing the goal at (9,3) leaves it in row 8, so the row
offset is high; the offset only rises, so the first must come first -- and the
column budget then reads the other way round, which is a contradiction. The
argument assumes the avatar is ADJACENT to the crate it pushes. It is not always:
``[ > item | item ]`` propagates through a chain, so a ridden scooter shoves a
crate two or three cells in front of the avatar, and that is enough slack for the
level to be solvable after all. The search found the win at depth 63; the
argument would have thrown the level away.

WHAT IS SEARCHED, AND WHY TWICE
-------------------------------
Both searches run on the native `_Board` model, which ``--model`` fuzzes against
the interpreter cell for cell (320 seated boards, 8000 presses, art included).
The interpreter only ever CERTIFIES the answer -- ``--plans`` replays every plan
on it, ``--verify`` presses every optimal label on it -- because it runs at ~360
steps a second and these searches take tens of millions of them.

* **The exact field** (`_Board.enumerate`), for seven of the eight levels. One
  forward BFS from the level start packing each state into eight bytes and four
  ``int32`` successor columns, then a backward BFS from the winning states as
  vectorised gathers straight over those columns -- no predecessor map is ever
  materialised, which is what makes level 5's 6.2M states cost 51 seconds and
  1.1 GB. The field answers "how far from a win" for EVERY reachable board, so
  the recorded epsilon detours are exact: the detour probe asks whether the
  board is still winnable and gets a lookup rather than a guess.
* **The layered sweep** (`_Board.layered_plan`), for level 7, whose space does
  not close inside 120M states -- 89878656 of them are swept to reach the first
  winning layer, in around ten minutes and 9.6 GB. Forward BFS keeping only a sorted
  key array per DEPTH (8 bytes a state), stopping at the first layer that wins
  -- that layer
  index is d* by construction -- then a backward walk from those winning states
  over `_Board.predecessors`, filtered against the layer arrays, which yields
  the ON-PATH set of every depth and therefore the exact tie sets. It is checked
  against the other algorithm rather than trusted: ``--verify`` runs it on the
  six small levels and requires the same d*, the same plan and the same sets.

A crate shoved somewhere it can never be pushed out of is dropped un-expanded --
`_Board.live_cells`, the standard sokoban relaxation, which only ever ADDS
pushes and so cannot kill a live state. It is worth the six lines: level 3 goes
from 431563 states to 170161 without a single distance changing. Note the DEAD
counts in ``--field``: 99.7% of level 3's space and 97.3% of level 5's is
boards that can no longer be won. That is the shape of this game, and it is
exactly why the detours have to be probed against a complete field.

RENDERING: FOUR ART FIXES, IN `data/puzzlescript_games/VRPS.txt`
---------------------------------------------------------------
``--audit`` renders every one of the 231 cell compositions this game can show,
at all five board sizes the eight levels use, and requires them to be pairwise
distinct. As shipped they were not, and the win condition was the casualty:

1. **The goal was invisible on three levels.** Its sprite was a plus sign drawn
   on the middle row and column, and `_render_cell_sprite` samples a 5x5 sprite
   at cell_px=4 through sprite indices 0,1,3,4 -- the middle row and column are
   DROPPED. On levels 1, 3 and 5 (10x14 and 10x16 boards, four pixels a cell) a
   goal square was pixel-identical to bare grass, and a crate standing on one
   was pixel-identical to a crate standing anywhere. Redrawn as four corner
   pixels plus a centre dot: the corners survive every cell size this game uses,
   and every body that can stand on a goal -- crate, avatar, either -- is
   transparent at all four of them.
2. **A scooter parked on a goal hid it completely**, at every cell size. The
   scooters are the one thing that paints a full-width edge bar, so they cover
   one PAIR of corners; the redrawn goal is still visible at the other two.
3. **The warning ticks were drawn ON those corners**, which put a goal under a
   scooter and a warning back to invisible. They now sit on the middle three
   cells of their edge, where nothing else draws.
4. **The faint tick-marks vanished at three pixels a cell** (levels 2, 4, 6, 7),
   where a 5x5 sprite is sampled at indices 0,2,4 and their two pixels sat at 1
   and 3. Same fix.
   Plus: the combined avatar-on-scooter sprite (``scavatar``) was fully opaque
   and gave up its four corner pixels for the same reason.

No object, collision layer, legend entry, rule, level or win condition is
touched, so the mechanic is bit-for-bit the game it was -- and ``--model``
re-fuzzes the whole board against the interpreter after the change.

AUGMENTATION
------------
The adapter's mandatory quarter-turn is joined here by the `_FLIP_GAMES`
mirrors, and both are earned rather than assumed. This game has more directional
art than anything else in that set -- four scooters, four avatar facings, four
player facings, four scavatars, four boundary markers and sixteen warning ticks
-- and all 36 sprites are an exact ORBIT of the 8-element group: turning the
scooterright sprite clockwise IS the scooterdown sprite, pixel for pixel, so in
a turned view every piece points the way the remapped key drives it.
``--audit`` measures all 288 of those comparisons. The MECHANIC is not stated
relatively either -- the scooter carry, the player freeze and the warning marks
are four per-direction blocks each, which is the shape that hid ps:gobble_rush's
chirality -- so ``--symmetry`` measures that too, replaying each level's plan
plus a 300-press random walk on all seven other turned and mirrored copies of
the LAYOUT, with every crate and every scooter required to land where the
transform says.

RECOVERY
--------
`PSAStarSolver`'s RESET prefix runs in front of every level, and on the six
levels whose field stays resident the expert replay also takes epsilon detours
(one press in eight): the probe steps a random alternative, asks the field
whether the board can still be won, and keeps the detour only if it can. The
taken action is then the mistake and ``optimal`` is the exact recovery. Levels 5
and 7 are recorded as clean shortest replays -- their fields are not resident
after the first plan, and a detour there would make a cold run and a
disk-cached run record different trajectories.

USAGE
-----
    python solvers/generate_vrps_training.py --episodes 1000 \
        --out data/training_multi_level/vrps

    python solvers/generate_vrps_training.py --plans      # level report
    python solvers/generate_vrps_training.py --field      # re-measure the spaces
    python solvers/generate_vrps_training.py --model      # model vs interpreter
    python solvers/generate_vrps_training.py --verify     # fields + labels
    python solvers/generate_vrps_training.py --audit      # rendering
    python solvers/generate_vrps_training.py --symmetry   # augmentation
    python solvers/generate_vrps_training.py --replay DIR # recorded episodes

Every plan is cached in ``data/vrps_plans.json`` after the first run, and the
two that cost minutes to derive -- levels 5 and 7 -- are kept in this file as
`_CACHED_PLANS` as well, so a cold machine, and every `parallelize_generator.py`
shard started on a cold cache, is ready in under three seconds. ``--field`` is
the report that re-derives them, and it is the expensive one: budget ~12 minutes
and ~10 GB.
"""

from __future__ import annotations

import itertools
import random
import sys
from array import array
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import (                     # noqa: E402
    PuzzleScriptAdapter, _render_frame)
from solvers.common.ps_astar import (                           # noqa: E402
    PSAStarSolver, PSExpert, Plan, restore, snapshot)

GAME_NAME = "VRPS"

#: Engine direction -> (dr, dc).
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: The presses the search branches on, in the order ties are broken. ``action``
#: is bound to nothing -- no rule in the game has it on a left-hand side and the
#: board is already a fixpoint of the `late` art rules when the turn starts, so
#: pressing it re-derives the same grid. `_model` puts it in the fuzz alphabet
#: and MEASURES that against the interpreter rather than reading it off the
#: rules section (ps:ouroboros is the game where exactly that assumption was
#: false), but the search does not branch on it.
_DIRS: tuple[str, ...] = ("up", "down", "left", "right")

_OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}
#: The two directions a warning tick-mark spreads along: perpendicular to the
#: edge it marks.
_PERP = {"up": ("left", "right"), "down": ("left", "right"),
         "left": ("up", "down"), "right": ("up", "down")}

#: Objects the model paints; everything else in a level's grid is static
#: scenery (grass, brick, goal, world, wall and the four `boundary` markers,
#: which `run_rules_on_level_start` puts on the room's walls once and no rule
#: ever moves).
_DYNAMIC_PREFIXES = ("crate", "scooter", "avatar", "scavatar", "player",
                     "1warning", "2warning")


#: The two plans that cost MINUTES to derive, kept here rather than only in the
#: disk cache: level 5's is read off a 6.2M-state field (51 seconds, 1.1 GB) and
#: level 7's off an 89.9M-state layered sweep (ten minutes, ten gigabytes).
#: Without them every machine -- and every `parallelize_generator.py` shard
#: started on a cold cache -- pays that again, and with them a cold start is two
#: seconds. `VrpsSolver.seed_plan` installs them, and ONLY when the level's
#: start board still matches the one they were derived from, so an edited level
#: is a miss and the real search runs.
#:
#: Nothing here is taken on trust: ``--field`` re-derives both from scratch and
#: prints their d*, ``--plans`` replays them press by press on the interpreter,
#: and ``--verify`` presses every alternative they label optimal. One group per
#: press, each group every direction that is equally shortest there, in
#: ``u``/``d``/``l``/``r``; the plan itself is each group's first letter.
_L5_START = (67, 92, (134,), ((38, 'down'), (55, 'left'), (66, 'right'),
                              (83, 'up')))
_L5_OPTSETS = (
    "l r d u r u l r ur r r d r l l l d l u l r u d d d d d d r r r r"
)

_L7_START = (18, 99, (36, 54), ((20, 'down'), (52, 'right')))
_L7_OPTSETS = (
    "d d r r u u d u l l d r r r l d r r r u u l d l d r ud r dr r d "
    "l d l u l u r u r d dr udr r d dl u l d d udr r d d ul l d d r d "
    "l l l"
)

#: level -> (start board, packed tie sets). `VrpsSolver.seed_plan` reads it.
_CACHED_PLANS = {5: (_L5_START, _L5_OPTSETS), 7: (_L7_START, _L7_OPTSETS)}

_LETTER = {"u": "up", "d": "down", "l": "left", "r": "right"}

# ---------------------------------------------------------------------------
# The level, natively: one turn, the art, the whole state space, the field
# ---------------------------------------------------------------------------

class _Board:
    """A level's static geometry plus one turn of its mechanic, the art that
    turn paints, and the two exact searches over its state space -- the full
    distance-to-win field (`enumerate`) where that fits in memory, and the
    layered sweep (`layered_plan`) where it does not.

    Cells are flat ``r * w + c`` ids and the MECHANICAL state is

        ``(avatar cell, player cell, sorted crate cells, sorted (scooter cell,
        type) pairs)``

    -- everything the rules read. The bricks of the VR world, the walls of the
    room, the goals and the `boundary` markers are static (no rule creates,
    destroys or moves any of them), so they live here rather than in the state.

    The facing sprites and the warning tick-marks are NOT in the state: no rule
    reads them, so they cannot change what a press does. They are a function of
    the state plus the direction of the last press that moved the avatar, which
    is what `art` computes -- and ``--model`` checks the whole grid, art
    included, against the interpreter.
    """

    __slots__ = ("h", "w", "bricks", "walls", "goals", "start", "start_facing",
                 "nxt", "is_brick", "is_wall", "types", "n_crates",
                 "_live", "keys", "index", "succ", "dist", "n_dead",
                 "n_states")

    def __init__(self, h, w, bricks, walls, goals, start, facing):
        self.h, self.w = h, w
        assert h * w <= 256, "a cell id must fit in one byte of the state key"
        self.bricks = frozenset(bricks)
        self.walls = frozenset(walls)
        self.goals = frozenset(goals)
        self.start = start
        self.start_facing = facing
        self.n_crates = len(start[2])
        #: scooter types in SLOT order: sorted, so two scooters of one type are
        #: interchangeable and the key stays canonical.
        self.types = sorted(t for _, t in start[3])

        self.is_brick = bytearray(h * w)
        self.is_wall = bytearray(h * w)
        for k in self.bricks:
            self.is_brick[k] = 1
        for k in self.walls:
            self.is_wall[k] = 1
        self.nxt = {}
        for d, (dr, dc) in _DELTA.items():
            t = [-1] * (h * w)
            for r in range(h):
                for c in range(w):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        t[r * w + c] = nr * w + nc
            self.nxt[d] = t

        self._live: frozenset | None = None
        self.keys: array | None = None
        self.index: dict = {}
        self.succ: list = []
        self.dist = None
        self.n_dead = 0
        #: kept across `release` so the reports can still say how big the space
        #: was after the field itself has been dropped.
        self.n_states = 0

    # -- one turn -------------------------------------------------------------
    def won(self, state) -> bool:
        """``all goal on crate`` -- every goal square covered."""
        return self.goals.issubset(state[2])

    def step(self, state, d: str):
        """One press, natively -- the rules of the .txt in the order it states
        them.

        The avatar copies the player's force; a scooter under the avatar joins
        in when its type matches the press; crates are shoved and item chains
        propagate; bricks and stationary items stop a chain from the front; and
        THEN the two coupling rules fire, which are what the game is about:

          * an avatar riding a moving scooter freezes the player, so the ride
            costs nothing in the room, and
          * an avatar stopped by a brick freezes the player with it.

        Last of all, ``[> player | wall] -> cancel``: if the player is still
        moving and the room's wall is in the way the WHOLE turn is undone, VR
        side included. Reaching for a wall you cannot see is how this game says
        "you have run out of living room"."""
        if d == "action":
            return state
        A, P, crates, scoots = state
        nx = self.nxt[d]
        nA = nx[A]
        if nA < 0 or self.is_brick[nA]:
            return state                     # avatar stopped -> player stopped
        ride = False
        for cell, t in scoots:
            if cell == A:
                ride = t == d
                break
        if not ride and nA not in crates:
            nP = nx[P]                       # a plain walk: nothing else moves
            if nP < 0 or self.is_wall[nP]:
                return state                 # cancel
            return (nA, nP, crates, scoots)

        cr = set(crates)
        sc = dict(scoots)
        mv_cr, mv_sc = set(), ({A} if ride else set())
        if nA in cr:
            mv_cr.add(nA)                    # [ > avatar | crate ]
        changed = True
        while changed:                       # [ > item | item ]
            changed = False
            for cell in list(mv_cr) + list(mv_sc):
                n = nx[cell]
                if n < 0:
                    continue
                if n in cr and n not in mv_cr:
                    mv_cr.add(n); changed = True
                if n in sc and n not in mv_sc:
                    mv_sc.add(n); changed = True
        mv_av = True
        changed = True
        while changed:                       # the two blocking rules
            changed = False
            if mv_av:
                n = nx[A]
                if n < 0 or self.is_brick[n] or (n in cr and n not in mv_cr):
                    mv_av = False; changed = True
            for cell in list(mv_cr):
                n = nx[cell]
                if (n < 0 or self.is_brick[n]
                        or (n in cr and n not in mv_cr)
                        or (n in sc and n not in mv_sc)):
                    mv_cr.discard(cell); changed = True
            for cell in list(mv_sc):
                n = nx[cell]
                if (n < 0 or self.is_brick[n]
                        or (n in cr and n not in mv_cr)
                        or (n in sc and n not in mv_sc)):
                    mv_sc.discard(cell); changed = True
        if not mv_av:
            return state                     # the player freezes with it
        if A in mv_sc:
            nP = P                           # the ride: the player stays put
        else:
            nP = nx[P]
            if nP < 0 or self.is_wall[nP]:
                return state                 # cancel
        return (nx[A],
                nP,
                tuple(sorted((nx[x] if x in mv_cr else x) for x in cr)),
                tuple(sorted((nx[x] if x in mv_sc else x, t)
                             for x, t in sc.items())))

    # -- the art the turn paints ---------------------------------------------
    def art(self, state, facing: str) -> dict:
        """``cell -> {object name}`` for everything the model paints.

        The facing sprites first: one press turns the avatar AND the player to
        face it, but only when the avatar actually moved (every rule that
        repaints them names a moving body), so a press into a brick or a
        cancelled press leaves both facing where they were. ``scavatar`` is the
        combined sprite drawn when the avatar stands on a scooter pointing the
        way it faces.

        Then the warnings, which are the game's whole conceit: for each side of
        the ROOM the player is standing against, a tick-mark is painted on the
        matching side of the AVATAR -- the VR headset drawing the living-room
        wall you are about to walk into. Each mark spreads a fainter pair of
        ticks along its own edge; a cell that would carry both a vertical and a
        horizontal mark is cleared outright, and one that lands on a brick is
        deleted."""
        A, P, crates, scoots = state
        out: dict = {}
        def add(cell, name):
            out.setdefault(cell, set()).add(name)

        add(A, "avatar" + facing)
        add(P, "player" + facing)
        for c in crates:
            add(c, "crate")
        for c, t in scoots:
            add(c, "scooter" + t)
            if c == A and t == facing:
                add(A, "scavatar" + facing)

        ver, hor = {}, {}
        for dd in _DIRS:
            n = self.nxt[dd][P]
            if n < 0 or not self.is_wall[n]:
                continue
            kind = _OPP[dd]
            cell = self.nxt[dd][A]
            if cell < 0:
                continue
            marks = ver if kind in ("up", "down") else hor
            marks[cell] = "2warning" + kind
            for p in _PERP[kind]:
                s = self.nxt[p][cell]
                if s >= 0:
                    marks.setdefault(s, "2warning" + kind + "2")
        for cell in set(ver) | set(hor):
            if self.is_brick[cell]:
                continue                     # late [ warning brick ] -> [ brick ]
            if cell in ver and cell in hor:
                continue                     # late [ warningver warninghor ] -> []
            add(cell, ver[cell] if cell in ver else hor[cell])
        return out

    # -- the state key the enumeration is built on ---------------------------
    def enc(self, state) -> int:
        """Pack a state into one int: avatar, player, the sorted crate cells,
        then the scooter cells in ``types`` (slot) order. One byte each, which
        `__init__` asserts is enough."""
        A, P, crates, scoots = state
        v = A | (P << 8)
        sh = 16
        for c in crates:
            v |= c << sh
            sh += 8
        for _t, c in sorted((t, c) for c, t in scoots):
            v |= c << sh
            sh += 8
        return v

    def dec(self, key: int):
        """`enc`'s exact inverse -- which is what lets the enumeration keep 8
        bytes per state instead of a nested tuple, and is re-checked over every
        state of every field by ``--verify``."""
        A = key & 255
        P = (key >> 8) & 255
        sh = 16
        crates = []
        for _ in range(self.n_crates):
            crates.append((key >> sh) & 255)
            sh += 8
        scoots = []
        for t in self.types:
            scoots.append(((key >> sh) & 255, t))
            sh += 8
        return (A, P, tuple(crates), tuple(sorted(scoots)))

    # -- pruning: crates that can never come back ----------------------------
    def live_cells(self) -> frozenset:
        """Cells a crate can still be pushed to a goal from, relaxed to ONE
        crate on an otherwise empty board with a free avatar.

        Backward BFS from the goals: a crate can be pushed from ``x`` to ``y``
        only if some body stands at ``x - d`` and ``y`` is not a brick, so the
        relaxation keeps just those two conditions. It only ever ADDS moves --
        the pushing body may be the avatar, a shoved crate or a ridden scooter,
        and all three need that cell -- so a cell it rules out is genuinely
        dead, and dropping the successor is sound rather than a heuristic.

        It is worth the line: on level 3 it takes the reachable space from
        431563 states to 170161 without changing a single distance."""
        if self._live is None:
            live = set(self.goals)
            queue = deque(self.goals)
            while queue:
                y = queue.popleft()
                for d in _DIRS:
                    x = self.nxt[_OPP[d]][y]
                    if x < 0 or self.is_brick[x] or x in live:
                        continue
                    if self.nxt[_OPP[d]][x] < 0 or self.is_brick[
                            self.nxt[_OPP[d]][x]]:
                        continue
                    live.add(x)
                    queue.append(x)
            self._live = frozenset(live)
        return self._live

    # -- the whole space ------------------------------------------------------
    def enumerate(self, cap: int) -> "tuple[int, int] | None":
        """Build the reachable state set, its successor table and the exact
        distance-to-win field. Returns ``(states, dead states)``, or None when
        the space passes ``cap`` (the board is then left unenumerated).

        One forward BFS from the level start, keeping 8-byte packed keys and
        four ``int32`` successor columns rather than snapshots, then a backward
        BFS from the winning states straight over those columns -- one
        vectorised gather per direction per layer, so no predecessor map is ever
        materialised. That is what makes the two big levels affordable: at ~6M
        and ~30M states an edge list would not fit, and the columns are 16 bytes
        a state.

        The set is CLOSED under every press, so any state the agent can reach --
        plan, exploration prefix, epsilon detour, RESET back to the start -- is
        in it, except the ones `live_cells` rules out, which are dead and are
        exactly what the detour probe must refuse. Winning states are terminal:
        the adapter ends the level there, so they are never expanded."""
        if self.keys is not None:
            return len(self.keys), self.n_dead
        live = self.live_cells()
        step, enc, dec, dirs = self.step, self.enc, self.dec, _DIRS
        keys = array("q", [enc(self.start)])
        index = {keys[0]: 0}
        succ = [array("i") for _ in dirs]
        wins = array("i")
        i = 0
        while i < len(keys):
            state = dec(keys[i])
            if self.won(state):
                wins.append(i)
                for column in succ:
                    column.append(-1)        # terminal: the level ends here
            else:
                for k, d in enumerate(dirs):
                    nxt = step(state, d)
                    if not live.issuperset(nxt[2]):
                        succ[k].append(-1)   # a crate shoved somewhere dead
                        continue
                    nk = enc(nxt)
                    j = index.get(nk)
                    if j is None:
                        if len(keys) >= cap:
                            return None
                        j = len(keys)
                        index[nk] = j
                        keys.append(nk)
                    succ[k].append(j)
            i += 1

        n = len(keys)
        cols = [np.frombuffer(memoryview(c), dtype=np.int32) for c in succ]
        dist = np.full(n, -1, dtype=np.int32)
        frontier = np.zeros(n, dtype=bool)
        win_idx = np.frombuffer(memoryview(wins), dtype=np.int32)
        dist[win_idx] = 0
        frontier[win_idx] = True
        depth = 0
        while frontier.any():
            reach = np.zeros(n, dtype=bool)
            for column in cols:
                ok = column >= 0
                hit = np.zeros(n, dtype=bool)
                hit[ok] = frontier[column[ok]]
                reach |= hit
            reach &= dist < 0
            depth += 1
            dist[reach] = depth
            frontier = reach

        self.keys, self.index, self.succ, self.dist = keys, index, cols, dist
        self.n_dead = int((dist < 0).sum())
        self.n_states = n
        return n, self.n_dead

    # -- the space that does NOT fit: one layered sweep, then the shortest
    #    paths only ------------------------------------------------------------
    def predecessors(self, state, d: str) -> list:
        """Every state ``s`` with ``step(s, d) == state``.

        `step` moves a body by exactly ``+d`` or not at all, so every
        predecessor is this state with some SUBSET of its bodies shifted one
        cell back -- avatar, player, each crate, each scooter. Enumerating the
        subsets and re-running `step` forward on each is therefore complete, and
        it is checked rather than argued: every candidate is verified against
        the model itself, so a predecessor this returns really is one."""
        back = self.nxt[_OPP[d]]
        A, P, crates, scoots = state
        bodies = [A, P] + list(crates) + [c for c, _t in scoots]
        n = len(bodies)
        out = []
        for mask in range(1 << n):
            moved = []
            ok = True
            for i, cell in enumerate(bodies):
                if mask >> i & 1:
                    cell = back[cell]
                    if cell < 0:
                        ok = False
                        break
                moved.append(cell)
            if not ok:
                continue
            k = 2 + len(crates)
            cand = (moved[0], moved[1],
                    tuple(sorted(moved[2:k])),
                    tuple(sorted(zip(moved[k:], (t for _c, t in scoots)))))
            if self.step(cand, d) == state:
                out.append(cand)
        return out

    def layered_plan(self, start, cap: int) -> "Plan | None":
        """A provably SHORTEST plan with exact tie sets for a level whose state
        space is too large to hold a successor table for.

        Two sweeps and no predecessor map:

          * **Forward, by layers.** BFS from ``start`` keeping only the packed
            keys of each depth (a sorted ``int64`` array per layer, 8 bytes a
            state) and stopping at the first layer that produces a win. That
            layer index IS d*, because BFS: no shorter win exists anywhere in
            the space.
          * **Backward, over the SHORTEST paths only.** From the winning states
            of that last layer, walk back one layer at a time with
            `predecessors`, keeping a candidate only when the layer arrays say
            it sits at exactly the right depth. What that produces is the
            ON-PATH set of each depth -- every state on a shortest route from
            the start to a win -- which is all the tie sets need, and it is tiny
            next to the layers it is filtered against.

        The plan is then a descent through those sets, taking the first tied
        press in ``_DIRS`` order so a re-derived plan is byte-identical across
        processes. ``--verify`` runs this against the full field on the levels
        that have one and requires the same d*, the same plan and the same sets.
        """
        live = self.live_cells()
        step, enc, dec = self.step, self.enc, self.dec
        seen = {enc(start)}
        layers = [array("q", [enc(start)])]
        wins: list = []
        while not wins:
            nxt = array("q")
            for key in layers[-1]:
                s = dec(key)
                for d in _DIRS:
                    x = step(s, d)
                    if not live.issuperset(x[2]):
                        continue
                    k = enc(x)
                    if k in seen:
                        continue
                    seen.add(k)
                    if self.won(x):
                        wins.append(x)          # terminal: the level ends here
                        continue
                    if len(seen) > cap:
                        return None
                    nxt.append(k)
            if not wins:
                if not nxt:
                    return None                 # closed with no win: unwinnable
                layers.append(nxt)
        self.n_states = len(seen)
        del seen
        sorted_layers = [np.sort(np.frombuffer(memoryview(a), dtype=np.int64))
                         for a in layers]
        del layers

        def at_depth(key, i) -> bool:
            arr = sorted_layers[i]
            j = int(np.searchsorted(arr, key))
            return j < arr.size and int(arr[j]) == key

        onpath = [None] * len(sorted_layers) + [{enc(w) for w in wins}]
        frontier = list(wins)
        for i in range(len(sorted_layers) - 1, -1, -1):
            here = {}
            for state in frontier:
                for d in _DIRS:
                    for pred in self.predecessors(state, d):
                        key = enc(pred)
                        if key not in here and at_depth(key, i):
                            here[key] = pred
            onpath[i] = set(here)
            frontier = list(here.values())
        presses, optsets = [], []
        cur = start
        for i in range(len(sorted_layers)):
            best = [d for d in _DIRS if enc(step(cur, d)) in onpath[i + 1]]
            presses.append(best[0])
            optsets.append(best)
            cur = step(cur, best[0])
        assert self.won(cur), "the layered descent did not reach a win"
        return Plan(presses, optsets)

    def release(self) -> None:
        """Drop the field, keeping the geometry. The two big levels are planned
        once and then answered from the disk plan cache, and their tables are
        gigabytes."""
        self.keys, self.index, self.succ, self.dist = None, {}, [], None

    # -- reading the field ----------------------------------------------------
    def distance(self, state) -> "int | None":
        """Presses to a win from ``state``, or None when no win is reachable
        from it (or a crate has been shoved to a cell `live_cells` rules out,
        which is the same answer)."""
        if self.keys is None:
            return None
        j = self.index.get(self.enc(state))
        if j is None or self.dist[j] < 0:
            return None
        return int(self.dist[j])

    def optimal(self, state) -> list:
        """Every press on a SHORTEST route to the win, straight off the field.

        A refused press (a brick, a cancelled turn, a jammed crate) maps the
        state to itself, whose distance is unchanged rather than one less, so
        no-ops exclude themselves; so does every shove that kills the level,
        because a dead state has no distance at all."""
        if self.keys is None:
            return []
        j = self.index.get(self.enc(state))
        if j is None or self.dist[j] <= 0:
            return []
        here = int(self.dist[j])
        out = []
        for k, d in enumerate(_DIRS):
            n = int(self.succ[k][j])
            if n >= 0 and self.dist[n] == here - 1:
                out.append(d)
        return out

    def plan(self, state) -> "Plan | None":
        """A SHORTEST press sequence from ``state``, carrying the EXACT optimal
        set at every step, or None if the board can no longer be won.

        A descent of the field taking the first tied press in ``_DIRS`` order,
        so a re-derived plan is byte-identical across processes."""
        if self.distance(state) is None:
            return None
        presses, optsets = [], []
        cur = state
        while not self.won(cur):
            best = self.optimal(cur)
            presses.append(best[0])
            optsets.append(best)
            cur = self.step(cur, best[0])
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class VrpsExpert(PSExpert):
    """`PSExpert`'s plan memo, disk cache and snapshot discipline around the two
    `_Board` searches.

    The base class keeps the memo, the restore discipline and the level scoping;
    only the strategy underneath changes, the way `PSEnumExpert` replaces it.
    Here the strategy is "read the answer off an exact distance field" for seven
    levels and "sweep the space by layers and walk the shortest paths back" for
    the eighth, so `heuristic` is never called and asserts rather than returning
    a number nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary board -- what the recovery prefix and the epsilon detours leave
    behind -- is answered exactly, including "this board is now dead", which is
    what `record_level` needs to hear to fall back to a RESET.
    """

    directions = list(_DIRS)

    #: `_key` is the DYNAMIC pieces only, which is canonical within a level but
    #: not across them: the bricks, the room and the goals that complete the
    #: state are static per level yet differ between levels.
    scope_by_level = True

    #: Levels whose whole field is small enough to keep RESIDENT, so a re-plan
    #: from any board -- an epsilon detour, a recovery prefix -- is a lookup.
    #: The other two are 6.2M states (level 5, whose field is read once and then
    #: dropped by `_Board.release`) and past 89M (level 7, which never has a
    #: field at all); both normally answer from `_CACHED_PLANS` without
    #: searching, and they are recorded as clean replays with no detours.
    field_levels = frozenset({0, 1, 2, 3, 4, 6})

    #: Levels planned by `_Board.layered_plan` instead of by a full field. Only
    #: level 7: its space does not close inside 120M states, but its shortest
    #: win is 63 presses and one layered forward sweep (89878656 states, ~10
    #: minutes, 9.6 GB) reaches it, after which the backward on-path sweep gives
    #: the same exact tie sets a field would. `--field` measures both numbers
    #: and `--verify` checks the two algorithms against each other on the six
    #: levels that have a field.
    layered_levels = frozenset({7})

    #: State cap for `layered_plan`'s forward sweep. Level 7 needs 89878656
    #: of it; the rest is headroom before the runaway guard fires.
    layer_cap = 120_000_000

    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "vrps_plans.json")

    def setup(self) -> None:
        g = self.g
        self.id_brick = set(g.resolve_object_name("brick"))
        self.id_wall = set(g.resolve_object_name("wall"))
        self.id_goal = set(g.resolve_object_name("goal"))
        self.id_crate = set(g.resolve_object_name("crate"))
        self.id_scooter = {d: set(g.resolve_object_name("scooter" + d))
                           for d in _DIRS}
        self.id_avatar = {d: set(g.resolve_object_name("avatar" + d))
                          for d in _DIRS}
        self.id_player = {d: set(g.resolve_object_name("player" + d))
                          for d in _DIRS}
        self.crate_id = min(self.id_crate)
        self.scooter_id = {d: min(self.id_scooter[d]) for d in _DIRS}
        self._boards: dict[int | None, _Board] = {}
        self._cur: _Board | None = None
        self._cur_level: int | None = None

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "VrpsExpert reads an exact distance field; heuristic is unused")

    # -- reading the engine ---------------------------------------------------
    def read(self, eng):
        """The MECHANICAL state of the engine grid (facings excluded)."""
        w = eng.width
        A = P = None
        crates, scoots = [], []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                k = r * w + c
                if cell & self.id_crate:
                    crates.append(k)
                for d in _DIRS:
                    if cell & self.id_scooter[d]:
                        scoots.append((k, d))
                    if cell & self.id_avatar[d]:
                        A = k
                    if cell & self.id_player[d]:
                        P = k
        if A is None or P is None:                       # pragma: no cover
            raise AssertionError("the board has no avatar or no player")
        return (A, P, tuple(sorted(crates)), tuple(sorted(scoots)))

    def facing(self, eng) -> str:
        """Which way the avatar is drawn. Art, never mechanics -- but the
        recorded FRAMES show it, so `art` and ``--model`` track it."""
        for row in eng.grid:
            for cell in row:
                for d in _DIRS:
                    if cell & self.id_avatar[d]:
                        return d
        raise AssertionError("the board has no avatar")    # pragma: no cover

    def board(self, eng, level: int | None) -> _Board:
        """The level's `_Board`, built once and cached. NOT enumerated here --
        two of the eight levels cost minutes and gigabytes to enumerate, so that
        is left to whoever actually needs the field.

        The bricks, the room's walls and the goals are static -- no rule touches
        any of them -- so whichever state happens to build it describes the same
        level; the START recorded in it is the state the engine holds at that
        moment, which `prepare_expert` makes the level's true start."""
        cached = self._boards.get(level)
        if cached is not None:
            return cached
        w = eng.width
        bricks, walls, goals = set(), set(), set()
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                k = r * w + c
                if cell & self.id_brick:
                    bricks.add(k)
                if cell & self.id_wall:
                    walls.add(k)
                if cell & self.id_goal:
                    goals.add(k)
        board = _Board(eng.height, w, bricks, walls, goals,
                       self.read(eng), self.facing(eng))
        self._boards[level] = board
        return board

    # -- planning -------------------------------------------------------------
    def _key(self, eng) -> frozenset:
        """The dynamic pieces as ``(row, col, id)`` triples, with the four
        avatar and the four player facings each collapsed onto ONE id.

        Facing is art: no rule reads it, so two boards that differ only in which
        way the two bodies are drawn have the same future and must share a plan
        -- and must share a `plan_cache_path` signature, which is built from
        this."""
        w = eng.width
        out = set()
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                if cell & self.id_crate:
                    out.add((r, c, self.crate_id))
                for d in _DIRS:
                    if cell & self.id_scooter[d]:
                        out.add((r, c, self.scooter_id[d]))
                    if cell & self.id_avatar[d]:
                        out.add((r, c, -1))
                    if cell & self.id_player[d]:
                        out.add((r, c, -2))
        return frozenset(out)

    def plan(self, eng, level: int | None = None):
        self._cur = self.board(eng, level)
        self._cur_level = level
        return super().plan(eng, level)

    def _search(self, eng) -> "Plan | None":
        board, level = self._cur, self._cur_level
        if level in self.layered_levels:
            return board.layered_plan(self.read(eng), self.layer_cap)
        if board.enumerate(self.node_cap) is None:          # pragma: no cover
            return None
        found = board.plan(self.read(eng))
        if level not in self.field_levels:
            board.release()
        return found

    def optimal_dirs(self, eng, level: int | None) -> list:
        """Every press on a shortest route, from the engine's CURRENT state.
        Empty when this level's field is not resident -- see `field_levels`."""
        return self.board(eng, level).optimal(self.read(eng))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class VrpsSolver(PSAStarSolver):
    game_id = "puzzlescript_vrps"
    game_name = GAME_NAME
    expert_cls = VrpsExpert

    #: `games/ps:vrps/ps:vrps.py` is a plain passthrough (it constructs the
    #: adapter and nothing else), so routing through it would gain nothing --
    #: but it IS what a live agent is handed, so if that wrapper ever grows a
    #: patch this must be set to ``"ps:vrps"``.
    game_module_id = ""

    #: Read as a STATE cap by `_Board.enumerate`, not as a node budget for a
    #: heuristic search: there is no heuristic here. It is a runaway guard sized
    #: above the biggest level measured (`--field` prints them all), so a level
    #: that somehow blew past it would be reported unsolvable rather than
    #: swallow the machine.
    node_cap = 40_000_000
    weight = 1

    #: The longest plan is 63 presses (level 7) and the epsilon detours only run
    #: on the six short levels; well under the adapter's own 200-press per-level
    #: budget, which the ``set_level`` that ends the exploration prefix resets
    #: anyway.
    max_steps = 150

    #: Non-zero even though a shoved crate is IRREVERSIBLE, which the exact
    #: field earns: `record_level`'s detour probe steps a random alternative,
    #: asks the expert whether it can still plan a win, and keeps the detour
    #: only if it can. For a bounded search that test is a guess; here it is a
    #: lookup in a field covering the whole reachable space, so a detour
    #: provably cannot strand the level. The taken action is then the mistake
    #: and ``optimal`` is the exact recovery -- the signal a policy needs after
    #: its own error.
    epsilon = 0.12

    def prepare_expert(self, game, expert) -> None:
        """Build every level's geometry, enumerate the six whose field is cheap,
        and hand the expert level 7's answer, before `discover_solvable` asks
        for plans.

        The enumerations are pure front-loading -- the reachable component does
        not depend on which of its states builds it -- but they pin each
        `_Board.start` to the level's reset state, so `--plans` and `--verify`
        report the shipped answer. Level 5 is deliberately NOT enumerated here:
        it costs 51 seconds and a gigabyte, and on any run after the first its
        plan comes from `plan_cache_path` without a field ever being built."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            board = expert.board(game._engine, level)
            if level in expert.field_levels:
                board.enumerate(self.node_cap)
            else:
                self.seed_plan(game, expert, board, level)

    def seed_plan(self, game, expert, board, level) -> bool:
        """Put `_CACHED_PLANS`' answer for ``level`` into the expert's memo, if
        the level's start board is still the one it was derived from.

        This is the hook `PSExpert.prepare_expert` documents -- "for pre-seeding
        ``expert.cache`` with plans that are too expensive to re-derive at every
        startup" -- and levels 5 and 7 are the case it describes:
        `discover_solvable`, the very next thing to run, would otherwise spend
        eleven minutes and ten gigabytes on two searches whose answers are
        already known.

        It also DROPS a poisoned disk entry. `PSExpert.plan` consults
        `plan_cache_path` before the memo and believes a stored ``"plan": null``
        as "this level cannot be won" -- which is exactly what a run that hit
        the state cap once would have written, and it would then shadow this
        seed forever."""
        entry = _CACHED_PLANS.get(level)
        if entry is None or board.start != entry[0]:
            return False
        optsets = [[_LETTER[c] for c in group] for group in entry[1].split()]
        expert.cache[(level, expert._key(game._engine))] = Plan(
            [s[0] for s in optsets], optsets)
        if expert._disk.get(level, {}).get("plan", True) is None:
            expert._disk.pop(level, None)
        return True

    def epsilon_for(self, level: int) -> float:
        """Detour only on the levels whose field stays resident.

        A detour forces a re-plan from wherever it landed, which on levels 5 and
        7 would mean rebuilding a multi-gigabyte field -- and would make a cold
        run (field resident) and a warm one (plan served from disk) record
        DIFFERENT trajectories, which is exactly the reproducibility this family
        checks. So those two are recorded as clean shortest replays."""
        return self.epsilon if level in VrpsExpert.field_levels else 0.0

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """Every press that keeps the game on a SHORTEST route, read off the
        LIVE engine state where the field is resident and off the plan's own
        stored sets where it is not.

        `record_level` asks for this BEFORE executing the press, so the engine
        is still at the state being labelled -- which is what makes the label
        right after an epsilon detour too, where the state is off the plan
        entirely and replaying the plan from the level start would label the
        wrong states. Both sources are the same exact field; the second one was
        read from it when the plan was first derived and then cached.

        Falls back to the press about to be taken if neither has anything to
        say -- no expert step may ship unlabelled."""
        best = expert.optimal_dirs(expert.game._engine, level)
        if best:
            return best
        sets = getattr(plan, "optsets", None)
        if sets is not None and pi < len(sets):
            return sets[pi]
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _new():
    """``(solver, game, expert, solvable)``, with the six cheap levels
    enumerated and every level's plan resolved (from `plan_cache_path` on any
    run after the first)."""
    solver = VrpsSolver()
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _replay_plan(game, level: int, plan) -> str:
    """Drive ``plan`` press by press on the REAL interpreter from the level's
    start. Returns "win", or what went wrong -- the one check that the model's
    answer is a genuine win path and not a story about a game of its own."""
    game.set_level(level)
    eng = game._engine
    for i, d in enumerate(plan):
        before = [[frozenset(c) for c in row] for row in eng.grid]
        eng.step(d)
        if [[frozenset(c) for c in row] for row in eng.grid] == before:
            return f"press {i} ({d}) was a no-op on the engine"
        if eng.check_win():
            return "win" if i == len(plan) - 1 else f"won early at press {i}"
    return "did not win"


def _plans(verbose: bool = True) -> int:
    """Every level's shipped plan: its length, how much of it is a tie, and a
    replay of it against the interpreter.

    Each plan is the shortest one that exists -- the space it was found in was
    enumerated (or swept by layers) to exhaustion, so "shortest" here is a proof
    and not a search result. The state counts are printed for the levels this
    process actually searched; levels 5 and 7 normally answer from
    `_CACHED_PLANS` or from the disk cache, and ``--field`` is what re-measures
    them. The replay is the check that matters here: every press driven on the
    real interpreter, ending in its win."""
    solver, game, expert, solvable = _new()
    bad = total = tied = 0
    for level in range(game.n_levels):
        if level in solver.skip_levels:
            print(f"level {level}: SKIPPED -- see VrpsSolver.skip_levels")
            continue
        game.set_level(level)
        board = expert.board(game._engine, level)
        plan = expert.plan(game._engine, level)
        if plan is None:
            print(f"level {level}: NO PLAN")
            bad += 1
            continue
        sets = getattr(plan, "optsets", []) or []
        ties = sum(1 for x in sets if len(x) > 1)
        total += len(plan)
        tied += ties
        verdict = _replay_plan(game, level, plan)
        bad += verdict != "win"
        size = ("  plan from the constant/cache" if not board.n_states else
                f"{board.n_states:8d} states swept by layer"
                if level in expert.layered_levels else
                f"{board.n_states:8d} states, {board.n_dead:8d} dead")
        if verbose:
            print(f"level {level} {board.h:2d}x{board.w:2d}: shortest "
                  f"{len(plan):3d} presses, {ties:3d} tie steps "
                  f"({ties / max(1, len(plan)):3.0%}), {size} -- replay "
                  f"{verdict}")
    print(f"{len(solvable)}/{game.n_levels} levels recorded, {total} presses, "
          f"{tied} tie steps, "
          + ("every plan replays to a WIN and is provably shortest"
             if not bad else f"{bad} LEVELS BAD"))
    return 0 if not bad else 1


def _field(verbose: bool = True) -> int:
    """Re-measure every level's state space from scratch, including the two the
    generator answers from disk.

    This is the expensive report -- around twelve minutes and ten gigabytes,
    nearly all of it level 7 -- and it is the one that backs the word
    "shortest": the enumeration terminates having
    generated every state the mechanic admits from the level start, so a level's
    d* is the distance in a complete graph rather than the best a search found.
    It also prints the DEAD count, which is the shape of this game -- a crate
    shoved one square too far is usually unrecoverable, and the exact field is
    what lets the recorded epsilon detours never fall into one."""
    import time
    solver = VrpsSolver()
    game = solver.make_game(0)
    expert = VrpsExpert(game, node_cap=solver.node_cap)
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(game._engine, level)
        t0 = time.time()
        if level in expert.layered_levels:
            plan = board.layered_plan(board.start, expert.layer_cap)
            dt = time.time() - t0
            verdict = _replay_plan(game, level, plan) if plan else "no plan"
            bad += verdict != "win"
            if verbose:
                print(f"level {level} {board.h:2d}x{board.w:2d}: "
                      f"{board.n_states:9d} states swept by LAYER (the space "
                      f"does not close), d*={len(plan) if plan else None:3}, "
                      f"{dt:6.1f}s -- replay {verdict}", flush=True)
            continue
        got = board.enumerate(solver.node_cap)
        dt = time.time() - t0
        if got is None:                                   # pragma: no cover
            print(f"level {level} {board.h:2d}x{board.w:2d}: OVER the "
                  f"{solver.node_cap} state cap after {dt:.0f}s -- it is not in "
                  f"layered_levels either, so it ships no plan")
            bad += 1
            continue
        n, dead = got
        plan = board.plan(board.start)
        verdict = _replay_plan(game, level, plan) if plan else "no plan"
        bad += verdict != "win"
        if verbose:
            print(f"level {level} {board.h:2d}x{board.w:2d}: {n:9d} states "
                  f"({dead:9d} dead, {100 * (n - dead) / n:5.2f}% live), "
                  f"d*={len(plan) if plan else None:3}, {dt:6.1f}s -- replay "
                  f"{verdict}", flush=True)
        board.release()
    print("every level enumerated and its shortest plan replays to a WIN"
          if not bad else f"FIELD REPORT FAILED: {bad} levels")
    return 0 if not bad else 1


def _dyn_ids(g) -> dict:
    """``{object id: name}`` for everything `_Board.art` paints."""
    return {i: n for n, i in g.obj_name_to_idx.items()
            if n.startswith(_DYNAMIC_PREFIXES)}


def _base_grid(game, level: int) -> list:
    """The level's STATIC cells: grass, brick, goal, world, wall and the
    `boundary` markers `run_rules_on_level_start` painted on the room."""
    game.set_level(level)
    dyn = _dyn_ids(game._game)
    return [[{o for o in cell if o not in dyn} for cell in row]
            for row in game._engine.grid]


def _seat(game, base, board, state, facing) -> None:
    """Write ``(state, facing)`` onto the engine grid, over the static base."""
    idx = game._game.obj_name_to_idx
    grid = [[set(cell) for cell in row] for row in base]
    for cell, names in board.art(state, facing).items():
        r, c = divmod(cell, board.w)
        for name in names:
            grid[r][c].add(idx[name])
    eng = game._engine
    eng.grid = grid
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


def _painted(game, board) -> dict:
    """``cell -> {name}`` for everything dynamic on the engine grid."""
    dyn = _dyn_ids(game._game)
    out = {}
    for r, row in enumerate(game._engine.grid):
        for c, cell in enumerate(row):
            names = {dyn[o] for o in cell if o in dyn}
            if names:
                out[r * board.w + c] = names
    return out


def _reachable(board, start: int, blocked) -> list:
    """The cells a body can stand on, flood-filled from ``start``."""
    seen, queue = {start}, [start]
    while queue:
        k = queue.pop()
        for d in _DIRS:
            n = board.nxt[d][k]
            if n >= 0 and not blocked[n] and n not in seen:
                seen.add(n)
                queue.append(n)
    return sorted(seen)


def _model(runs: int = 40, presses: int = 25, verbose: bool = True) -> int:
    """Fuzz the native model against the interpreter, WHOLE GRID for whole grid.

    Each run seats a state on the engine and then drives both it and the model
    with the same random presses, comparing every dynamic object in every cell
    after each one -- pieces, the two facing sprites, the combined ``scavatar``
    and all eight warning tick-marks. Nothing here is projected away, so the
    comparison also proves the ART derivation, which is what the recorded
    FRAMES show and what a `_Board` state alone does not carry.

    Run 0 of each level is the level's own start; the rest are randomly seated
    boards -- crates, scooters and the avatar scattered over the VR floor and
    the player anywhere in the room -- because a random walk from the start
    almost never reaches the configurations the exotic rules are about: a ridden
    scooter shoving a second scooter, a crate pushed from two cells away through
    one, an avatar walking off a scooter that a brick has just stopped.

    ``action`` is in the alphabet on purpose. No rule in the game reads it, but
    that is an assumption about the .txt until it is pressed against the
    interpreter -- ps:ouroboros is the game where exactly that assumption was
    false."""
    solver = VrpsSolver()
    game = solver.make_game(0)
    expert = VrpsExpert(game, node_cap=solver.node_cap)
    eng = game._engine
    alphabet = _DIRS + ("action",)
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(game._engine, level)
        base = _base_grid(game, level)
        floor = _reachable(board, board.start[0], board.is_brick)
        room = _reachable(board, board.start[1], board.is_wall)
        assert not set(floor) & board.walls, "the avatar can reach the room"
        assert not set(room) & board.bricks, "the player can reach the VR world"
        rng = random.Random(f"vrps:model:{level}")
        n_cr, n_sc = board.n_crates, len(board.types)
        mism = actions = 0
        for run in range(runs):
            if run == 0:
                state, facing = board.start, board.start_facing
            else:
                cells = rng.sample(floor, n_cr + n_sc + 1)
                state = (cells[0], rng.choice(room),
                         tuple(sorted(cells[1:1 + n_cr])),
                         tuple(sorted(zip(sorted(cells[1 + n_cr:]),
                                          board.types))))
                facing = rng.choice(_DIRS)
            _seat(game, base, board, state, facing)
            for _ in range(presses):
                d = rng.choice(alphabet)
                before = _painted(game, board)
                eng.step(d)
                nxt = board.step(state, d)
                turned = facing if nxt[0] == state[0] else d
                want = board.art(nxt, turned)
                got = _painted(game, board)
                if d == "action":
                    actions += 1
                    if got != before:
                        mism += 1
                        if verbose:
                            print(f"  L{level}: ACTION changed the board")
                if want != got:
                    mism += 1
                    if verbose and mism <= 3:
                        print(f"  L{level} run {run}: press {d} from {state} "
                              f"facing {facing}")
                        for cell in sorted(set(want) | set(got)):
                            if want.get(cell) != got.get(cell):
                                print(f"    {divmod(cell, board.w)}: "
                                      f"model={sorted(want.get(cell, ()))} "
                                      f"engine={sorted(got.get(cell, ()))}")
                    state, facing = expert.read(eng), expert.facing(eng)
                else:
                    state, facing = nxt, turned
                if eng.check_win():
                    break
        bad += mism
        if verbose:
            print(f"  L{level}: {runs} seated boards x {presses} presses "
                  f"({actions} of them ACTION): {mism} mismatches")
    print("the native model matches the interpreter cell for cell" if not bad
          else f"MODEL FAILED: {bad} mismatches")
    return 0 if not bad else 1


def _verify(sample: int = 20000, verbose: bool = True) -> int:
    """Double-entry check of every distance field and of every tie set it ships.

    `_Board.enumerate` answers everything from ONE pass -- a forward sweep that
    packs the successor table and a backward sweep of gathers over it. A bug in
    either would produce a self-consistent field, a plan that still wins, and
    tie sets that are quietly wrong; and the training labels ARE those tie sets,
    so "it wins" is not enough of a check. Four passes, which fail differently:

      * **The key round-trips.** ``dec(enc(s)) == s`` on a sample of each space,
        because the whole enumeration is stored as packed ints and a lossy
        packing would silently MERGE two states into one.
      * **The successor table is the mechanic.** Re-derive a sample of rows by
        calling `_Board.step` again and require the same landing state (or the
        same "dead crate" refusal) -- the table is built once and read
        thereafter, so nothing else would ever notice it drifting.
      * **The Bellman certificate.** Every state must satisfy
        ``d(s) = 1 + min d(succ)``, ``d == 0`` exactly at the winning states,
        and every state with no distance must have no successor that has one.
        Given a transition function that matches the interpreter (which
        ``--model`` fuzzes) that is a PROOF of minimality over a space the
        enumeration closed, so the plans are shortest and the ties are complete.
      * **Executable labels.** Walk each level's plan on the INTERPRETER and, at
        every step, actually press every direction the label calls optimal: the
        board the engine lands on must be the native successor, and its field
        distance must be exactly one less. That takes the labels out of the
        model and puts them through the real thing.
    """
    solver, game, expert, _solvable = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        if level in expert.layered_levels:
            # No field to certify: this level's exactness rests on the layered
            # algorithm, which is certified against the field on the six levels
            # that have one (pass 5 below). What is checkable here is that the
            # shipped plan and every alternative it labels are real moves on the
            # real interpreter.
            plan = expert.plan(eng, level)
            game.set_level(level)
            moves = 0
            for i, press in enumerate(plan):
                before = snapshot(eng)
                for alt in plan.optsets[i]:
                    eng.step(alt)
                    if eng.grid == before:
                        bad += 1
                        print(f"  L{level}: step {i} labels {alt} optimal, but "
                              f"it does not move anything")
                    moves += 1
                    restore(eng, before)
                if press not in plan.optsets[i]:
                    bad += 1
                    print(f"  L{level}: step {i} is not in its own optimal set")
                eng.step(press)
            if not eng.check_win():
                bad += 1
                print(f"  L{level}: the interpreter does not report a win")
            print(f"  L{level}: no field (layered sweep), {len(plan):3d} plan "
                  f"steps, {moves:3d} labels pressed on the interpreter")
            continue
        if board.enumerate(solver.node_cap) is None:      # pragma: no cover
            print(f"  L{level}: over the state cap")
            return 1
        n = board.n_states
        dist = board.dist
        rng = random.Random(f"vrps:verify:{level}")
        live = board.live_cells()

        # -- pass 1 + 2: the packing and the successor table ------------------
        idxs = (range(n) if n <= sample
                else [rng.randrange(n) for _ in range(sample)])
        pack = rows = 0
        for j in idxs:
            state = board.dec(board.keys[j])
            if board.enc(state) != board.keys[j]:
                pack += 1
            if board.won(state):
                continue
            for k, d in enumerate(_DIRS):
                nxt = board.step(state, d)
                want = (-1 if not live.issuperset(nxt[2])
                        else board.index[board.enc(nxt)])
                if int(board.succ[k][j]) != want:
                    rows += 1
        bad += bool(pack) + bool(rows)

        # -- pass 3: the Bellman certificate ----------------------------------
        best = np.full(n, 1 << 20, dtype=np.int64)
        for column in board.succ:
            ok = column >= 0
            step_d = np.full(n, 1 << 20, dtype=np.int64)
            got = dist[column[ok]]
            step_d[ok] = np.where(got < 0, 1 << 20, got)
            best = np.minimum(best, step_d)
        wins = dist == 0
        live_states = dist >= 1
        bell = int((best[live_states] != dist[live_states] - 1).sum())
        dead_bad = int((best[dist < 0] < (1 << 20)).sum())
        win_bad = int(sum(1 for j in np.nonzero(wins)[0]
                          if not board.won(board.dec(board.keys[j]))))
        win_missing = int(sum(1 for j in idxs
                              if board.won(board.dec(board.keys[j]))
                              and dist[j] != 0))
        bad += bool(bell) + bool(dead_bad) + bool(win_bad) + bool(win_missing)

        # -- pass 4: every shipped label, pressed on the interpreter ---------
        game.set_level(level)
        plan = board.plan(board.start)
        state = board.start
        labels = 0
        for i, press in enumerate(plan):
            here = board.distance(state)
            best_set = board.optimal(state)
            if press not in best_set:
                bad += 1
                print(f"  L{level}: step {i} is not in its own optimal set")
            before = snapshot(eng)
            for alt in best_set:
                eng.step(alt)
                landed = expert.read(eng)
                labels += 1
                if (landed != board.step(state, alt)
                        or board.distance(landed) != here - 1):
                    bad += 1
                    print(f"  L{level}: step {i} labels {alt} optimal, but the "
                          f"interpreter does not land one press closer")
                restore(eng, before)
            eng.step(press)
            state = board.step(state, press)
        if not eng.check_win():
            bad += 1
            print(f"  L{level}: the interpreter does not report a win")
        if verbose:
            print(f"  L{level}: {n:9d} states certified ({len(idxs)} keys and "
                  f"successor rows re-derived), {len(plan):3d} plan steps, "
                  f"{labels:3d} labels pressed on the interpreter -- "
                  f"{pack} packing, {rows} successor, {bell} Bellman, "
                  f"{dead_bad} dead-state, {win_bad + win_missing} win-state "
                  f"problems")
        # -- pass 5: the two searches against each other ---------------------
        layered = board.layered_plan(board.start, expert.layer_cap)
        same = (layered is not None and list(layered) == list(plan)
                and layered.optsets == plan.optsets)
        bad += not same
        if verbose:
            verdict = ("the same plan and the same tie sets" if same
                       else "A DIFFERENT ANSWER")
            print(f"        and the layered sweep -- the algorithm level 7 is "
                  f"planned by -- returns {verdict}")
        board.release()
    print(f"verify: {bad} problems")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# Presentation: the render audit and the symmetry proof
# ---------------------------------------------------------------------------

def _compositions() -> list:
    """Every cell composition this game can put on screen.

    The layers do most of the work: ``brick``, ``crate`` and the avatar share
    one, so no cell ever holds two of them; the room's ``world`` and ``wall``
    never meet the VR world's ``grass``; and a warning is only ever painted
    beside the avatar, on a cell of the VR floor, so it can land on grass, a
    goal, a crate or a scooter but never on the avatar itself and never in the
    room. What is left is the product below."""
    comps = [("brick",), ("wall",)]
    comps += [("wall", "boundary" + d) for d in _DIRS]
    comps += [("world",)] + [("world", "player" + d) for d in _DIRS]
    bodies = [(), ("crate",)] + [("avatar" + d,) for d in _DIRS]
    scooters = [()] + [("scooter" + d,) for d in _DIRS]
    warnings = [()] + [("2warning" + d + s,) for d in _DIRS for s in ("", "2")]
    for goal in [(), ("goal",)]:
        for body in bodies:
            for scoot in scooters:
                pair = body + scoot
                if body and scoot and body[0][6:] == scoot[0][7:]:
                    pair += ("scavatar" + body[0][6:],)
                for warn in ([()] if body and body[0].startswith("avatar")
                             else warnings):
                    comps.append(("grass",) + goal + pair + warn)
    return comps


def _shoot(game, comps, h: int, w: int) -> dict:
    """A whole 64x64 frame per composition, on a uniform ``h`` x ``w`` board.

    Whole frames rather than cell crops: `_render_frame` upscales the board to
    fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor)."""
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    shots = {}
    for objs in comps:
        eng.height, eng.width = h, w
        eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                     for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        shots[objs] = np.asarray(_render_frame(eng, g)).copy()
    return shots


def _orbit(game, verbose: bool = True) -> int:
    """The check the ADAPTER'S AUGMENTATION rests on: every directional sprite
    is an exact orbit of the 8-element presentation group.

    The adapter turns (and, for `_FLIP_GAMES`, mirrors) the rendered frame and
    remaps the keys to match, so a scooter that points right on the engine grid
    is driven by the screen-DOWN key in a quarter-turned view. That is only
    honest if the turned scooterright sprite IS the scooterdown sprite -- and it
    is, pixel for pixel, for every scooter, avatar, player, combined
    ``scavatar``, boundary marker and warning tick in the game. This measures
    it, on a one-cell board at the largest cell size, rather than reading the
    .txt.

    (Only the DIRECTIONAL art has to be an orbit. The crate is chiral and the
    grass texture is chiral, so a mirrored board shows art the .txt does not
    literally contain -- which is harmless, and is also why no turned frame can
    ever be mistaken for an unturned one: `_audit` measures that too.)"""
    g = game._game
    eng = game._engine
    idx = g.obj_name_to_idx
    names = [stem + d + tail
             for stem in ("scooter", "avatar", "player", "scavatar",
                          "boundary", "1warning", "2warning")
             for d in _DIRS
             for tail in (("", "2") if stem.endswith("warning") else ("",))]
    shots = {}
    for nm in names:
        eng.height, eng.width = 1, 1
        eng.grid = [[{idx["background"], idx[nm]}]]
        eng._position_index_dirty = True
        shots[nm] = np.asarray(_render_frame(eng, g)).copy()
    bad = checked = 0
    for k, mirror in itertools.product(range(4), (False, True)):
        _c, _d, _dm, omap = _transform(k, mirror)
        for nm, img in shots.items():
            turned = np.rot90(np.fliplr(img) if mirror else img, -k)
            checked += 1
            if not np.array_equal(turned, shots[omap[nm]]):
                bad += 1
                if verbose:
                    print(f"  NOT AN ORBIT: {nm} turned {k}"
                          f"{' and mirrored' if mirror else ''} is not "
                          f"{omap[nm]}")
    print(f"{len(shots)} directional sprites x 8 presentations: "
          f"{checked} checks, "
          + ("every turn maps a sprite exactly onto the sprite its direction "
             "map names" if not bad else f"{bad} FAILURES"))
    return bad


def _audit(verbose: bool = True) -> int:
    """Assert every cell COMPOSITION the game can show is distinct in the
    rendered frame, at every board size the eight levels use -- and stays
    distinct under the eight turns and mirrors the augmentation applies.

    The pairs this exists for:

      * ``crate`` against ``crate+goal``. The win condition is ``all goal on
        crate``, so a crate that hid the goal under it would make a solved board
        indistinguishable from an unsolved one.
      * ``avatar`` against ``avatar+scooter``. Standing on a scooter is what
        makes the next press free in the living room; an agent that cannot see
        it cannot see the mechanic.
      * the four ``scooter`` types against each other, since a scooter only
        works in the direction it points.
      * the warning tick-marks against bare floor, which are the only thing on
        the VR side that says where the real walls are.

    Three of the five board sizes render at THREE pixels a cell, where a 5x5
    sprite is decimated to 3x3 and margins are the first thing to go -- which is
    why this is a measurement and not a reading of the .txt."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    comps = _compositions()
    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    name = lambda t: "+".join(t) if t else "floor"               # noqa: E731
    clashes_total = _orbit(game)

    for (h, w), levels in sorted(sizes.items()):
        shots = _shoot(game, comps, h, w)
        floor = shots[("grass",)]
        print(f"{h}x{w} board (level{'s' if len(levels) > 1 else ''} "
              f"{', '.join(str(x) for x in levels)}), "
              f"cell {max(1, min(64 // h, 64 // w))}px")
        for objs, under in [
                (("grass", "goal"), ("grass",)),
                (("grass", "crate"), ("grass",)),
                (("grass", "goal", "crate"), ("grass", "crate")),
                (("grass", "goal", "scooterright"), ("grass", "scooterright")),
                (("grass", "scooterright"), ("grass",)),
                (("grass", "avatarup"), ("grass",)),
                (("grass", "avatarup", "scooterup", "scavatarup"),
                 ("grass", "avatarup")),
                (("grass", "avatarup", "scooterright"), ("grass", "avatarup")),
                (("grass", "2warningup"), ("grass",)),
                (("grass", "2warningup2"), ("grass",)),
                (("grass", "crate", "2warningleft"), ("grass", "crate")),
                (("world", "playerup"), ("world",)),
                (("wall", "boundaryup"), ("wall",))]:
            print(f"  {name(objs):44s} {int((shots[objs] != floor).sum()):4d} "
                  f"px differ from bare grass, "
                  f"{int((shots[objs] != shots[under]).sum()):4d} from "
                  f"{name(under)}")

        # -- pass 1: distinct as rendered ------------------------------------
        seen: dict = {}
        for objs, img in shots.items():
            seen.setdefault(img.tobytes(), []).append(objs)
        clashes = 0
        for bucket in seen.values():
            for a, b in itertools.combinations(bucket, 2):
                clashes += 1
                if clashes <= 8:
                    print(f"  IDENTICAL: {name(a)} == {name(b)}")

        # -- pass 2: the 8 presentations map compositions ONTO compositions ---
        # A turn of the screen turns the art with it, and this game's
        # directional sprites are an exact ORBIT of the group -- scooterright
        # turned clockwise IS scooterdown, pixel for pixel -- which is what
        # makes the adapter's rotation honest: in the turned view the scooter
        # points the way the remapped key drives it. So the check is not "no
        # frame equals another frame under a turn", which would flag the orbit
        # itself; it is that a turned frame equals the frame of the composition
        # the DIRECTION MAP says it should be, and never any other one.
        exact = mapped = 0
        for k, mirror in itertools.product(range(4), (False, True)):
            _cell, _dims, _dmap, omap = _transform(k, mirror)
            index = {img.tobytes(): objs for objs, img in shots.items()}
            for objs, img in shots.items():
                turned = np.rot90(np.fliplr(img) if mirror else img, -k)
                want = tuple(omap[o] for o in objs)
                mapped += 1
                if np.array_equal(turned, shots[want]):
                    exact += 1
                    continue
                other = index.get(turned.tobytes())
                if other is not None and other != objs:
                    clashes += 1
                    if clashes <= 8:
                        print(f"  CLASH: {name(objs)} turned {k}"
                              f"{' and mirrored' if mirror else ''} == "
                              f"{name(other)}, not {name(want)}")
        clashes_total += clashes
        print(f"  {len(comps)} compositions: "
              f"{'all distinct' if not clashes else 'SEE ABOVE'}; "
              f"{exact}/{mapped} presentations render the composition the "
              f"direction map names")
    print("audit clean" if not clashes_total
          else f"AUDIT FAILED: {clashes_total} indistinguishable pairs")
    return 0 if not clashes_total else 1


def _transform(k: int, mirror: bool):
    """``(cell_fn, dims_fn, direction_map, object_map)`` for one element of the
    8-group: mirror left-right first, then turn clockwise ``k`` times.

    The direction map is DERIVED from the same linear part that moves the cells,
    and the object map from the direction map, so the three cannot drift apart.
    The object map is what a game with directional pieces needs and a plain
    sokoban does not: turning this board clockwise turns every scooter, avatar,
    player, boundary marker and warning tick with it."""
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
    for nm, (dr, dc) in _DELTA.items():
        if mirror:
            dc = -dc
        for _ in range(k):
            dr, dc = dc, -dr
        dmap[nm] = next(n for n, d in _DELTA.items() if d == (dr, dc))
    #: ACTION carries no direction, so every element of the group fixes it. It
    #: is in the map because `_symmetry`'s random walk presses it -- the button
    #: a live agent has -- and a walk that skipped it would not be measuring the
    #: alphabet the exploration prefix uses.
    dmap["action"] = "action"

    stems = ("scooter", "ascooter", "scavatar", "avatar", "player", "boundary",
             "1warning", "2warning")
    def obj(nm):
        for stem in stems:
            if not nm.startswith(stem):
                continue
            rest = nm[len(stem):]
            tail = "2" if rest.endswith("2") else ""
            base = rest[:-1] if tail else rest
            if base in dmap:
                return stem + dmap[base] + tail
        return nm

    return cell, dims, dmap, _NameMap(obj)


class _NameMap(dict):
    """``omap[name]`` for a rule that is a function rather than a table."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def __missing__(self, key):
        value = self.fn(key)
        self[key] = value
        return value


def _replay_transformed(game, expert, layout, presses) -> list:
    """Replay ``presses`` on all seven non-identity turned/mirrored copies of
    ``layout`` and return a note per presentation that diverged.

    The transformed board is built from the LEVEL LAYOUT -- cells moved and
    every directional object renamed -- and reloaded, so the interpreter parses
    and resolves it itself rather than being handed a grid this file transformed
    after the fact. Every piece is compared, not just the avatar: a turn that
    moved a crate somewhere else while leaving the avatar where the transform
    says is exactly the chirality this is looking for."""
    eng = game._engine
    idx = game._game.obj_name_to_idx
    names = {i: n for n, i in idx.items()}
    hw = (len(layout), len(layout[0]))
    eng.load_level(layout)
    ref = [expert.read(eng)]
    for d in presses:
        eng.step(d)
        ref.append(expert.read(eng))

    notes = []
    for k, mirror in itertools.product(range(4), (False, True)):
        if (k, mirror) == (0, False):
            continue
        cellf, dims, dmap, omap = _transform(k, mirror)
        th, tw = dims(hw)
        turned = [[set() for _ in range(tw)] for _ in range(th)]
        for r, row in enumerate(layout):
            for c, objs in enumerate(row):
                tr, tc = cellf((r, c), hw)
                turned[tr][tc] = {idx[omap[names[o]]] for o in objs}
        eng.load_level(turned)
        flat = lambda x: (lambda rc: rc[0] * tw + rc[1])(        # noqa: E731
            cellf(divmod(x, hw[1]), hw))
        for i, d in enumerate(presses):
            eng.step(dmap[d])
            A, P, crates, scoots = ref[i + 1]
            want = (flat(A), flat(P),
                    tuple(sorted(flat(x) for x in crates)),
                    tuple(sorted((flat(x), dmap[t]) for x, t in scoots)))
            if expert.read(eng) != want:
                notes.append(f"rot{k}{'m' if mirror else ''}@press{i}")
                break
    return notes


def _symmetry(walk: int = 300, verbose: bool = True) -> int:
    """Prove ON THE INTERPRETER that the 8-element presentation group is an
    exact symmetry of the MECHANIC, which is what entitles this game to the
    adapter's mandatory rotation and to the `_FLIP_GAMES` mirrors on top of it.

    The structural argument is the usual one -- no gravity, screen-relative
    input, a win condition that names no direction (``all goal on crate``), and
    every rule stated either with the relative ``>`` force or as a complete
    four-direction block -- plus the one thing this game adds: its art is
    directional, and the sprites were drawn as an exact ORBIT of the group
    (``--audit`` measures that). But the rules that decide a turn here are not
    all relative: the scooter carry, the player freeze and the warning marks are
    written out per direction, which is the shape that hid ps:gobble_rush's
    chirality. So it is MEASURED instead:

      * each level's own PLAN, which is the sequence the corpus records and the
        only one that reaches a win, and
      * a seeded RANDOM WALK per level, which presses into walls, rides
        scooters the wrong way and presses the unbound ACTION key.

    Both are replayed on all seven other presentations of the level -- cells
    moved and every directional object renamed -- and every piece must land
    where the transform says."""
    solver, game, expert, _solvable = _new()
    g = game._game
    alphabet = _DIRS + ("action",)
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = ([] if level in solver.skip_levels
                else expert.plan(game._engine, level) or [])
        layout = g.levels[level]
        rng = random.Random(f"vrps:symmetry:{level}")
        runs = {"plan": list(plan),
                "walk": [rng.choice(alphabet) for _ in range(walk)]}
        notes = []
        for kind, presses in runs.items():
            notes += [f"{kind}:{n}" for n in
                      _replay_transformed(game, expert, layout, presses)]
        bad += len(notes)
        if verbose:
            verdict = ("SYMMETRIC" if not notes
                       else "CHIRAL " + ", ".join(notes[:3]))
            print(f"  L{level}: ({len(runs['plan'])} plan + {walk} random) "
                  f"presses x 7 presentations: {verdict}")
    print("symmetry clean -- every presentation is the same game" if not bad
          else f"SYMMETRY FAILED: {bad} chiral presentations")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# End to end: replay a recorded episode against a fresh adapter
# ---------------------------------------------------------------------------

def _replay_episode(path, expert, verbose: bool = True) -> int:
    """Replay a recorded episode against a fresh adapter, frame for frame, and
    re-derive every label it ships.

    The end-to-end check that the tape is a real game: every recorded action is
    pressed on a newly built adapter at the episode's own seed, and every frame
    it renders must equal the frame the file stores. It is what says the SCREEN
    actions (rotation- and mirror-remapped) were recorded the way an agent would
    have to press them, and that the level really ends in a WIN.

    On top of that, for every EXPERT step of a level whose field is resident,
    the recorded ``optimal`` set is compared against the field's answer for the
    board the engine is standing on at that moment -- re-derived here, from the
    file, in screen coordinates. That closes the last gap `--verify` leaves: it
    certifies the field and the sets the SEARCH produced, while this certifies
    the sets that actually reached the disk, remap included."""
    import json

    from arcengine import ActionInput, GameState

    from solvers.base_solver import _ID_TO_GAMEACTION
    from solvers.common.ps_astar import screen_action

    data = json.loads(Path(path).read_text())
    seed = int(Path(path).stem.split("seed")[-1])
    solver = VrpsSolver()
    game = solver.make_game(seed)
    bad = 0
    checked = 0
    for entry in data["levels"]:
        level = entry["level_id"]
        obs, acts = entry["observations"], entry["actions"]
        game._seed = seed
        game.set_level(level)
        rot = (game._rotation_k, game._hflip, game._vflip)
        frames = [np.asarray(game._current_frame)]
        labels = 0
        for act in acts[1:]:
            if act.get("phase") == "expert":
                best = expert.optimal_dirs(game._engine, level)
                if best:
                    want = {int(screen_action(d, *rot,
                                              solver.remap_actions).value)
                            for d in best}
                    got = {int(o["index"]) for o in (act.get("optimal") or ())}
                    labels += 1
                    if want != got:
                        bad += 1
                        print(f"  level {level}: a recorded optimal set is "
                              f"{sorted(got)}, the field says {sorted(want)}")
            fd = game.perform_action(
                ActionInput(id=_ID_TO_GAMEACTION[act["index"]]))
            frames.append(np.asarray(
                fd.frame[-1] if fd.frame else game._current_frame))
        checked += labels
        mism = sum(1 for a, b in zip(frames, obs)
                   if not np.array_equal(a, np.asarray(b)))
        # EXPERT steps are the ones that must carry a target: a step without
        # one stays in the context and contributes nothing to the loss, which is
        # silent. The opening exploration prefix and the RESET that ends it are
        # the standing exceptions -- `train_policy` masks both out of the policy
        # loss, and the prefix's whole point is that its actions are mistakes.
        blank = sum(1 for a in acts
                    if a.get("phase") == "expert" and not a.get("optimal"))
        ok = (len(frames) == len(obs) and not mism and not blank
              and game._state == GameState.WIN)
        bad += 0 if ok else 1
        if verbose and not ok:
            print(f"  level {level}: {len(frames)} frames vs {len(obs)} "
                  f"recorded, {mism} differ, {blank} steps unlabelled, "
                  f"final state {game._state}")
    if verbose:
        steps = sum(len(e["actions"]) for e in data["levels"])
        labelled = sum(1 for e in data["levels"] for a in e["actions"]
                       if a.get("optimal"))
        tied = sum(1 for e in data["levels"] for a in e["actions"]
                   if len(a.get("optimal") or ()) > 1)
        print(f"{Path(path).name}: {len(data['levels'])} levels, {steps} "
              f"actions ({labelled} labelled, {tied} with a tie, {checked} "
              f"re-derived from the field), "
              + ("replays frame for frame to a WIN" if not bad
                 else f"{bad} LEVELS DO NOT REPLAY"))
    return bad


def _replays(out_dir, verbose: bool = True) -> int:
    """`_replay_episode` over every episode JSON in a directory."""
    files = sorted(Path(out_dir).glob("episode_*.json"))
    if not files:
        print(f"no episodes in {out_dir}")
        return 1
    _solver, _game, expert, _solvable = _new()
    bad = sum(_replay_episode(f, expert, verbose) for f in files)
    print("all episodes replay" if not bad else f"{bad} LEVELS BAD")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--field" in sys.argv:
        sys.exit(_field())
    if "--model" in sys.argv:
        sys.exit(_model())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--replay" in sys.argv:
        where = sys.argv[sys.argv.index("--replay") + 1]
        sys.exit(_replays(where))
    sys.exit(VrpsSolver.main())
