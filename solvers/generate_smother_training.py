"""Generate Phase-1 training data for the PuzzleScript game ps:smother
("Smother" by Team Borse).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the `BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: a
native model of one turn, the exact distance field over every level's WHOLE
reachable space, the optimal-action labeller, and the reports that prove each of
them.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_smother",
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
action (post rotation remap), i.e. the button an agent presses in the presented
view, so replaying the recorded actions reproduces the recorded frames exactly.
Every expert step carries the full set of equally-optimal presses.

The game
--------
Three games share one file and one key set, and the .txt announces the switch
only with a `message`. All sixteen levels are recorded; nothing here is skipped.

**Chapter 1, "I PROTECT YOU" (levels 0-4, and 15).** Two bodies -- a blue
PlayerFull and a purple Protector -- move together on every arrow press, and BOTH
have to end on an Exit. Guns in the walls fire lasers, permanently, down
(LaserGV) and right (LaserGH). The beam runs until it meets a `LaserBlock` --
Wall, Exit, or any player body -- and is drawn ON that cell as well as before it.
So a laser is not a corridor you can duck through: standing anywhere in it, even
at the cell that stops it, is standing in it. **PlayerFull dies there. Protector
does not, and blocks the beam.** That asymmetry is the whole chapter: the
Protector walks into the beam so the victim can cross the column behind it, and
since one key moves both, the puzzle is finding a walk whose walls and corners
break the lockstep into the offset the shielding needs. A corpse (PlayerDead)
never moves again and still counts for `All PlayerWin on Exit`, so a death is
final unless it happens to land on an Exit -- the field below says so state by
state rather than assuming it.

**Chapter 2, "I AM ALWAYS SEARCHING" (levels 5-9).** One green ProtectorSolo, no
Exit, no lasers, and the only win condition left is `No Heart`. The move is not a
step: `[ > ProtectorSolo | ... | Heart ] -> [ | ... | ProtectorSolo ]` TELEPORTS
it to the nearest Heart in the pressed direction and eats it, and the rule right
after cancels the press when there is none. Two consequences the .txt does not
say and the levels lean on: the jump passes straight **through walls** (an
ellipsis constrains nothing it spans -- measured, not read off the rule), and
**every effective press eats exactly one Heart**, so every winning line is the
same length -- the number of Hearts -- and the puzzle is purely the ORDER. Get it
wrong and a Heart is left with nothing sharing its row or column; the level is
lost with the board still looking almost solved. Of level 9's 5483 reachable
states, 5169 are already lost.

**Chapter 3, "I AM INDEPENDENT" (levels 10-14).** One lightblue victimSolo, which
dies in a laser like the PlayerFull did, plus orange pushBlocks it can shove one
at a time. `[pushBlock laser] -> [wall]` is the tool: a block is not a LaserBlock,
so a beam passes straight over it -- and then, at the START of the next turn
(rules run before movement), that block becomes a **Wall**, which does block. So
the way through a laser is to shove a block into it, and WHERE matters more than
anything: the new wall shadows only the part of the beam BEYOND it, so a block
fed into the far end of a beam opens nothing at all. There is no undo -- a block
spent in the wrong cell is a wall forever -- which is why 144041 of level 13's
183271 states cannot win. This is also the one chapter where **ACTION is not a
no-op**: no rule reads the action force, but the rules phase runs on every press
whatever the input, so ACTION is a PASS that converts a block standing in a beam
while leaving the victim exactly where it is -- something no directional press
does on open floor. `--selfcheck` found that; the rules section does not say it.

Model + search
--------------
`_Board.step` re-implements one turn natively: the laser raycast, the push, the
block-to-wall conversion, the heart teleport, PuzzleScript's simultaneous
movement resolution over the shared collision layer, and the two death rules.
``--selfcheck`` drives 38k random presses (arrow keys AND ACTION) through BOTH
the interpreter and the model across all sixteen levels and compares the FULL
board -- every object class including the derived laser cells -- plus the win
flag after every one. That is the guard that lets everything below trust it.

There is then no search. A state is (the four bodies, the corpses, a Heart
bitmask, the blocks, the blocks that have become walls), and the whole reachable
space is 322693 states across the sixteen levels, enumerated in about 11 seconds
and 260 MB. `_Board.enumerate` walks it once from the level start, keeps the
successor table, and runs one backward BFS from the winning states to get the
exact distance-to-win field. A plan is a descent of the field, a tie set is a
lookup, an unwinnable level would be a PROOF rather than a search giving up, and
a state reached by an exploration detour is answered from the same table as the
start state. Shortest plans are 4 to 184 presses; the whole game is built once
per process and shared by every seed.

Optimal-action targets and recovery
-----------------------------------
`optimal_for` reads the LIVE engine state and returns every press whose successor
is exactly one step closer -- true tie sets straight off the field, not a
reordering heuristic. They matter most in chapter 1, where the two bodies often
have several equally good ways to reach the same offset, and least in chapter 2,
where a press is a whole teleport and rarely ties.

``epsilon`` stays at the `PSAStarSolver` family default of 0, so recovery comes
from the explore-then-RESET prefix that opens every episode. The field itself
would happily support in-episode detours -- it answers from any reachable state,
and `record_level` only commits a detour that leaves the level winnable -- but
the adapter GAME_OVERs a level at 200 actions and level 13's optimum is 184
presses, which leaves no room for the extra presses a detour costs.

Rendering
---------
Three fixes, all in the game FILE (see the header of
data/puzzlescript_games/smother.txt) and all of them things that were silently
undrawn: the `Laser` collision layer was below Wall / pushBlock / PlayerDead so a
beam on any of them was invisible; the one-pixel-wide beam sprites disappeared
entirely at the cell_px 4 and cell_px 2 that levels 4 and 13 render at; and the
Exit was a ring on the sprite's outer columns, which cell_px 2 never samples, so
on level 13 the goal -- and a victim standing on it -- was not drawn at all.
``--audit`` is the check, and it does not guess at the compositions: it collects
every cell composition that occurs anywhere in the SOLVED state space of each
level and asserts they render distinctly at that level's own cell size. Whole
frames are compared, never cell crops (`_render_frame` letterboxes, so an
arithmetic crop reads the wrong window).

Augmentation
------------
Per (seed, level) the adapter draws a frame rotation (0..3); ``Smother`` is not in
`PuzzleScriptAdapter._FLIP_GAMES`, so there are 4 presentations of each of the 16
levels. ``--symmetry`` checks the contract the recording depends on: for each
level and each rotation, drive the level's own plan through the ADAPTER using
`screen_action` and require both that it still wins and that every presented
frame is exactly the rotation of the unrotated one.

Usage (run from the repo root):
    python solvers/generate_smother_training.py --episodes 200 \
        --out data/training_multi_level/puzzlescript_smother
    python solvers/generate_smother_training.py --selfcheck
    python solvers/generate_smother_training.py --plans
    python solvers/generate_smother_training.py --verify
    python solvers/generate_smother_training.py --audit
    python solvers/generate_smother_training.py --symmetry
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import (                     # noqa: E402
    PuzzleScriptAdapter, _render_frame)
from arcengine import ActionInput                               # noqa: E402
from solvers.common.ps_astar import (                           # noqa: E402
    PSAStarSolver, PSExpert, screen_action)

GAME_NAME = "smother"

#: Engine direction -> (dr, dc).
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: The press alphabet. ACTION is in it, and it is NOT a no-op: no rule reads the
#: action force (`[ > ProtectorSolo ]` matches a MOVEMENT force), but the rules
#: phase runs on every press regardless of input, so ACTION is a PASS that still
#: fires `[pushBlock laser] -> [wall]` -- it turns a block standing in a beam
#: into a wall without moving the victim, which no directional press does on an
#: open floor. That was found by `selfcheck`, not read off the rules section,
#: and the state space is enumerated over all five presses so it stays closed
#: under anything the exploration prefix can do.
_PRESSES = ("up", "down", "left", "right", "action")

#: Object classes the model reads off a level, and the state fields they fill.
#: Static (never created, destroyed or moved by any rule): Wall as laid out,
#: Exit, and the two gun types. Dynamic: everything else.
_STATIC_NAMES = ("wall", "exit", "lasergv", "lasergh")
_DYNAMIC_NAMES = ("playerfull", "protector", "protectorsolo", "victimsolo",
                  "playerdead", "heart", "pushblock")


# ---------------------------------------------------------------------------
# The level, natively: one turn, the whole state space, the distance field
# ---------------------------------------------------------------------------

class _Board:
    """A level's static geometry plus its COMPLETE state space and the exact
    distance-to-win field over it.

    Cells are flat ``r * w + c`` ids. A state is the tuple

        (playerfull, protector, protectorsolo, victimsolo,
         corpses, heart_mask, blocks, new_walls)

    where the four bodies are a cell id or -1, ``corpses`` / ``blocks`` /
    ``new_walls`` are sorted tuples of cell ids and ``heart_mask`` is a bitmask
    over `heart_cells`. Every field is small and hashable, which is what makes
    the 183k-state enumeration of level 13 fit in a dict: the same state written
    as a set of occupied cells costs several times as much and hashes slower.

    Only the DYNAMIC objects are in the state -- the walls a level ships with,
    the Exits and the guns live on the board -- so a state is canonical only
    WITHIN one level, which is why `SmotherExpert` sets ``scope_by_level``.
    ``new_walls`` is separate from the shipped walls for the same reason: it is
    the part of the terrain chapter 3 rewrites.

    `enumerate` is what everything else reads: one forward BFS from the level
    start collecting every reachable state and its four successors, then one
    backward BFS from the winning states. `plan` is a descent down the field,
    `optimal` a lookup, and both work from ANY reachable state -- which is what
    lets the recording label a state an exploration detour landed on.
    """

    __slots__ = ("h", "w", "base_walls", "exits", "gv", "gh", "guns",
                 "heart_cells", "start", "states", "index", "succ", "dist")

    def __init__(self, eng, g):
        idx = g.obj_name_to_idx
        h, w = eng.height, eng.width
        self.h, self.w = h, w
        walls, exits, gv, gh, hearts, blocks, corpses = [], [], [], [], [], [], []
        p = n = q = k = -1
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                i = r * w + c
                if idx["wall"] in cell:
                    walls.append(i)
                if idx["exit"] in cell:
                    exits.append(i)
                if idx["lasergv"] in cell:
                    gv.append(i)
                if idx["lasergh"] in cell:
                    gh.append(i)
                if idx["heart"] in cell:
                    hearts.append(i)
                if idx["pushblock"] in cell:
                    blocks.append(i)
                if idx["playerdead"] in cell:
                    corpses.append(i)
                if idx["playerfull"] in cell:
                    p = i
                if idx["protector"] in cell:
                    n = i
                if idx["protectorsolo"] in cell:
                    q = i
                if idx["victimsolo"] in cell:
                    k = i
        self.base_walls = frozenset(walls)
        self.exits = frozenset(exits)
        self.gv, self.gh = tuple(gv), tuple(gh)
        self.guns = frozenset(gv) | frozenset(gh)
        self.heart_cells = tuple(sorted(hearts))

        # `run_rules_on_level_start` runs a full turn with no input before the
        # first frame: nothing moves, no block is on a laser yet (there are no
        # lasers until the late phase), so the level's start state is the ASCII
        # map with the LATE phase applied once -- the beams drawn, and anything
        # already standing in one killed.
        self.start = self.late((p, n, q, k, tuple(sorted(corpses)),
                                (1 << len(hearts)) - 1,
                                tuple(sorted(blocks)), ()))

        self.states: list | None = None
        self.index: dict = {}
        self.succ: list = []
        self.dist: list = []

    # -- the laser field ------------------------------------------------------
    def field(self, state) -> set:
        """The cells a beam occupies, from the four laser rules read as one
        raycast.

        ``late down [LaserGV | no Laser] -> [LaserGV | LaserV]`` starts a beam in
        the cell below each down-gun and ``late down [LaserV no LaserBlock | ] ->
        [LaserV | LaserV]`` extends it -- note the ``no LaserBlock`` is on the
        SOURCE cell and the target is unconstrained, so the beam is painted onto
        the blocker and stops there. That "inclusive of the blocker" is the whole
        reason a laser cannot be sheltered in: the cell that stops the beam is a
        cell the beam is on, and PlayerFull / victimSolo standing there die. The
        remaining laser rules only decide which of LaserV / LaserH / LaserB is
        drawn where two beams meet, which changes no behaviour -- every kind is
        equally lethal and equally passed over -- so one cell set answers for
        all three.
        """
        p, n, q, k, corpses, _hearts, _blocks, new_walls = state
        h, w = self.h, self.w
        blockers = set(self.base_walls)
        blockers.update(new_walls)
        blockers.update(self.exits)
        blockers.update(corpses)
        for body in (p, n, q, k):
            if body >= 0:
                blockers.add(body)
        laser = set()
        for i in self.gv:
            r, c = divmod(i, w)
            for rr in range(r + 1, h):
                j = rr * w + c
                laser.add(j)
                if j in blockers:
                    break
        for i in self.gh:
            r, c = divmod(i, w)
            for cc in range(c + 1, w):
                j = r * w + cc
                laser.add(j)
                if j in blockers:
                    break
        return laser

    def late(self, state):
        """The two death rules: PlayerFull and victimSolo standing in a beam
        become a PlayerDead. The Protector is deliberately absent from both, and
        a corpse is a LaserBlock exactly as the body was, so the field it is
        checked against does not change when one dies."""
        p, n, q, k, corpses, hearts, blocks, new_walls = state
        if (p >= 0 or k >= 0):
            laser = self.field(state)
            hit = [x for x in (p, k) if x >= 0 and x in laser]
            if hit:
                if p in hit:
                    p = -1
                if k in hit:
                    k = -1
                corpses = tuple(sorted(corpses + tuple(hit)))
        return (p, n, q, k, corpses, hearts, blocks, new_walls)

    # -- one turn -------------------------------------------------------------
    def step(self, state, d: str):
        """One press, natively: the rules phase in file order, then PuzzleScript's
        movement resolution, then the late phase.

        ``d`` may be ``"action"``, which grants no force: the push, the teleport
        and the movement pass are all skipped and only the rules that read the
        board -- i.e. the block-to-wall conversion -- and the late phase run."""
        p, n, q, k, corpses, hearts, blocks, new_walls = state
        h, w = self.h, self.w
        move = _DELTA.get(d)
        dr, dc = move if move is not None else (0, 0)
        laser = self.field(state)
        blocks = list(blocks)
        new_walls = list(new_walls)

        # (a) `[ > victimSolo | pushBlock ] -> [ > victimSolo | > pushBlock ]`.
        # One block, never a chain: the rule names victimSolo, so a second block
        # behind the first is never given a force and simply blocks the push.
        pushed = -1
        if k >= 0 and move is not None:
            kr, kc = divmod(k, w)
            tr, tc = kr + dr, kc + dc
            if 0 <= tr < h and 0 <= tc < w and tr * w + tc in blocks:
                pushed = tr * w + tc

        # (b) `[pushBlock laser] -> [wall]`, and it runs AFTER the push rule, so
        # a block that was already standing in a beam turns to wall even in the
        # turn you try to shove it -- the force is granted and then the object it
        # was granted to stops existing. That also makes the conversion a whole
        # turn LATE: the block spends the turn it was pushed sitting in the beam
        # and only becomes a wall at the start of the next one.
        if blocks:
            keep = []
            for b in blocks:
                if b in laser:
                    new_walls.append(b)
                    if pushed == b:
                        pushed = -1
                else:
                    keep.append(b)
            blocks = keep

        # (c) the ProtectorSolo teleport, and (d) the rule that cancels its move
        # when there was no Heart to teleport to -- so it never takes a step.
        if q >= 0 and hearts and move is not None:
            qr, qc = divmod(q, w)
            rr, cc = qr + dr, qc + dc
            while 0 <= rr < h and 0 <= cc < w:
                bit = self._heart_bit(rr * w + cc)
                if bit & hearts:
                    hearts &= ~bit
                    q = rr * w + cc
                    break
                rr += dr
                cc += dc

        # Movement resolution over the one shared collision layer (`get_collision_layer`
        # takes the LAST layer an object is named on, so ProtectorSolo, Wall and
        # every body land on it; Heart and Exit have their own and block nothing).
        # Everything moves the same way, so the chain settles front-first: retry
        # until no mover can advance, which is what cancels a push into a wall
        # AND the victim behind it in one pass.
        occ = set(self.base_walls)
        occ.update(new_walls, self.guns, corpses, blocks)
        if q >= 0:
            occ.add(q)
        movers = [x for x in (p, n, k) if x >= 0] if move is not None else []
        n_bodies = len(movers)
        if pushed >= 0:
            movers.append(pushed)
        occ.update(movers)
        done = [False] * len(movers)
        progress = True
        while progress:
            progress = False
            for i, cell in enumerate(movers):
                if done[i]:
                    continue
                cr, cc = divmod(cell, w)
                tr, tc = cr + dr, cc + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    done[i] = True                   # walked into the board edge
                    continue
                t = tr * w + tc
                if t in occ:
                    continue                         # blocked -- maybe not later
                occ.discard(cell)
                occ.add(t)
                movers[i] = t
                done[i] = True
                progress = True
        if move is not None:
            mi = 0
            if p >= 0:
                p = movers[mi]
                mi += 1
            if n >= 0:
                n = movers[mi]
                mi += 1
            if k >= 0:
                k = movers[mi]
            if pushed >= 0:
                blocks.remove(pushed)
                blocks.append(movers[n_bodies])

        return self.late((p, n, q, k, corpses, hearts,
                          tuple(sorted(blocks)), tuple(sorted(new_walls))))

    def _heart_bit(self, cell: int) -> int:
        """The bitmask bit for a Heart at ``cell``, or 0 if no Heart ever sat
        there. Bisection over the sorted start positions -- this is the inner
        loop of the teleport scan."""
        cells = self.heart_cells
        lo, hi = 0, len(cells)
        while lo < hi:
            mid = (lo + hi) // 2
            if cells[mid] < cell:
                lo = mid + 1
            else:
                hi = mid
        return (1 << lo) if lo < len(cells) and cells[lo] == cell else 0

    def won(self, state) -> bool:
        """``All PlayerWin on Exit`` and ``No Heart``.

        PlayerWin is PlayerFull, PlayerDead, Protector and victimSolo -- note
        ProtectorSolo is NOT one, which is what makes the first condition
        vacuously true through the whole of chapter 2 and leaves `No Heart` as
        the only thing to satisfy there."""
        p, n, q, k, corpses, hearts, _blocks, _new_walls = state
        if hearts:
            return False
        for body in (p, n, k):
            if body >= 0 and body not in self.exits:
                return False
        return all(x in self.exits for x in corpses)

    # -- the whole space ------------------------------------------------------
    def enumerate(self) -> tuple[int, int]:
        """Build the reachable state set, its successor table and the exact
        distance-to-win field. Returns ``(states, states that cannot win)``.

        The set is CLOSED under every press (each state contributes all four of
        its successors, ACTION included -- on most boards it maps a state to
        itself, but see `_PRESSES` for the one case where it does not), so
        any state the agent can reach by any sequence of presses -- plan,
        exploration prefix, epsilon detour, or a RESET back to the start -- is
        already in it. Winning states are terminal: the adapter ends the level
        there, so they are not expanded.
        """
        if self.states is not None:
            return len(self.states), sum(1 for x in self.dist if x < 0)

        states = [self.start]
        index = {self.start: 0}
        succ: list = []
        i = 0
        while i < len(states):
            state = states[i]
            if self.won(state):
                succ.append(None)                    # terminal: the level ends
            else:
                row = []
                for d in _PRESSES:
                    nxt = self.step(state, d)
                    j = index.get(nxt)
                    if j is None:
                        j = len(states)
                        index[nxt] = j
                        states.append(nxt)
                    row.append(j)
                succ.append(tuple(row))
            i += 1

        pred: list[list[int]] = [[] for _ in states]
        for src, row in enumerate(succ):
            if row is not None:
                for dst in row:
                    pred[dst].append(src)

        dist = [-1] * len(states)
        queue = deque()
        for j, state in enumerate(states):
            if self.won(state):
                dist[j] = 0
                queue.append(j)
        while queue:
            j = queue.popleft()
            for src in pred[j]:
                if dist[src] < 0:
                    dist[src] = dist[j] + 1
                    queue.append(src)

        self.states, self.index, self.succ, self.dist = states, index, succ, dist
        return len(states), sum(1 for x in dist if x < 0)

    # -- reading the field ----------------------------------------------------
    def distance(self, state) -> int | None:
        """Presses to a win from ``state``, or None when it cannot win at all
        (a victim dead off an Exit, a Heart stranded, a block spent in the wrong
        cell) or is not reachable from the level start."""
        self.enumerate()
        j = self.index.get(state)
        if j is None or self.dist[j] < 0:
            return None
        return self.dist[j]

    def plan(self, state) -> list | None:
        """A SHORTEST press sequence from ``state`` to a win, or None.

        A descent of the field taking the first tied press in ``_PRESSES``
        order, so the plan is a pure function of the board and is byte-identical
        across processes and seeds."""
        self.enumerate()
        j = self.index.get(state)
        if j is None or self.dist[j] < 0:
            return None
        out = []
        while self.dist[j] > 0:
            for k, nxt in enumerate(self.succ[j]):
                if self.dist[nxt] == self.dist[j] - 1:
                    out.append(_PRESSES[k])
                    j = nxt
                    break
            else:                                    # pragma: no cover
                raise AssertionError("distance field has no descent")
        return out

    def optimal(self, state) -> list:
        """Every press that keeps the game on a SHORTEST route to the win.

        Exact, straight off the field. A press that changes nothing maps the
        state to itself, whose distance is unchanged rather than one less, so
        no-ops exclude themselves for free -- and so does every press that walks
        into a beam, because the state it lands on cannot win."""
        self.enumerate()
        j = self.index.get(state)
        if j is None or self.dist[j] <= 0:
            return []
        here = self.dist[j]
        return [_PRESSES[k] for k, nxt in enumerate(self.succ[j])
                if self.dist[nxt] == here - 1]


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class SmotherExpert(PSExpert):
    """Exact planner over the enumerated state space (see `_Board`).

    `PSExpert` supplies the plan memo, the restore discipline and the level
    scoping; `_search` swaps in the field descent, so the interpreter is only
    ever stepped by the recorder -- which is also what verifies every plan, since
    a level is kept only when the engine itself reports WIN.
    """

    directions = list(_PRESSES)

    #: `_key` carries the dynamic objects only: canonical within a level, but
    #: the walls, Exits and guns that complete the board are static per level
    #: and differ between them.
    scope_by_level = True

    def setup(self) -> None:
        g = self.g
        self.ids = {name: set(g.resolve_object_name(name))
                    for name in _STATIC_NAMES + _DYNAMIC_NAMES}
        self._boards: dict[int | None, _Board] = {}
        self._cur: _Board | None = None

    # -- reading the engine ---------------------------------------------------
    def board(self, eng, level: int | None) -> _Board:
        """The level's `_Board`, built (and enumerated) once and cached.

        Built from whatever state the engine happens to be in, which is safe
        only because the geometry it reads -- the shipped walls, the Exits, the
        guns -- is static: no rule creates or destroys any of them. The one
        terrain object that IS dynamic, a block turned to wall, is state rather
        than geometry, so `prepare_expert` builds every board from its level's
        start to keep the two apart."""
        board = self._boards.get(level)
        if board is None:
            board = _Board(eng, self.g)
            board.enumerate()
            self._boards[level] = board
        return board

    def read(self, eng, board: _Board):
        """The `_Board` state tuple for the engine's current grid.

        Scan order is row-major, i.e. increasing cell id, so the corpse / block /
        new-wall tuples come out sorted for free and compare equal to the ones
        `_Board.step` builds."""
        ids = self.ids
        w = board.w
        p = n = q = k = -1
        corpses, blocks, new_walls = [], [], []
        hearts = 0
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & ids["playerfull"]:
                    p = i
                if cell & ids["protector"]:
                    n = i
                if cell & ids["protectorsolo"]:
                    q = i
                if cell & ids["victimsolo"]:
                    k = i
                if cell & ids["playerdead"]:
                    corpses.append(i)
                if cell & ids["pushblock"]:
                    blocks.append(i)
                if cell & ids["heart"]:
                    hearts |= board._heart_bit(i)
                if cell & ids["wall"] and i not in board.base_walls:
                    new_walls.append(i)
        return (p, n, q, k, tuple(corpses), hearts,
                tuple(blocks), tuple(new_walls))

    # -- planning -------------------------------------------------------------
    def _key(self, eng):
        return self.read(eng, self._cur)

    def plan(self, eng, level: int | None = None) -> list | None:
        self._cur = self.board(eng, level)
        return super().plan(eng, level)

    def heuristic(self, eng) -> int:
        """The EXACT remaining press count. The base `_astar` is never reached
        here -- `_search` answers first -- but the contract is that this is 0 at
        a win and admissible, and the field is both."""
        d = self._cur.distance(self.read(eng, self._cur))
        return 0 if d is None else d

    def _search(self, eng) -> list | None:
        return self._cur.plan(self.read(eng, self._cur))

    def optimal_dirs(self, eng, level: int | None) -> list:
        """Every press on a shortest route, from the engine's CURRENT state."""
        board = self.board(eng, level)
        return board.optimal(self.read(eng, board))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class SmotherSolver(PSAStarSolver):
    game_id = "puzzlescript_smother"
    game_name = GAME_NAME
    expert_cls = SmotherExpert

    #: Unused -- `SmotherExpert._search` never calls the base A* -- but left at
    #: the family default so a future subclass that does is not silently starved.
    node_cap = 2_000_000
    weight = 1

    #: The adapter GAME_OVERs a level at 200 actions, and the RESET that ends the
    #: exploration prefix zeroes that counter, so the whole budget is the plan's.
    #: The longest here is level 13 at 184 presses.
    max_steps = 200

    #: Zero, the family default. The field would answer a detour from anywhere
    #: it landed -- it covers the whole reachable space, and `record_level` only
    #: keeps a detour that leaves the level winnable -- but level 13's 184-press
    #: optimum leaves 16 presses under the adapter's cap and a detour costs at
    #: least two, so a uniform epsilon would silently start losing that level.
    #: Recovery comes from the explore-then-RESET prefix instead.
    epsilon = 0.0

    def prepare_expert(self, game, expert) -> None:
        """Enumerate every level's state space up front, from its START state.

        Not just front-loading: `_Board` reads the shipped walls off whatever
        board it is handed, and in chapter 3 a spent block IS a wall, so a board
        built mid-level would bake a block's grave into the static terrain and
        every state after it would be keyed wrong. Building at ``set_level`` is
        what guarantees the split between geometry and state."""
        for level in range(game.n_levels):
            game.set_level(level)
            expert.board(game._engine, level)

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """Every press that keeps the game on a SHORTEST route to the win, read
        off the LIVE engine state.

        `record_level` asks for this before executing the press, so the engine is
        still at the state being labelled -- which is what would keep the label
        right after an epsilon detour, where replaying the plan from the level
        start would label the wrong states.

        Falls back to the press about to be taken if the field has nothing to say
        -- no expert step may ship unlabelled."""
        best = expert.optimal_dirs(expert.game._engine, level)
        if best:
            return best
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _levels():
    """``(solver, game, expert)`` with every level's space enumerated."""
    solver = SmotherSolver()
    game, expert, _solvable = solver._ensure(0)
    return solver, game, expert


def _engine_state(eng, g, board: _Board):
    """Every object class's cell set as the INTERPRETER has it, including the
    derived laser cells. This is the whole board, which is what `selfcheck`
    compares -- a model checked only on the player's cell would miss a beam
    drawn one square short or a block that converted a turn early."""
    idx = g.obj_name_to_idx
    lasers = {idx["laserv"], idx["laserh"], idx["laserb"]}
    out = {name: set() for name in _STATIC_NAMES + _DYNAMIC_NAMES}
    out["laser"] = set()
    w = board.w
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if not cell:
                continue
            i = r * w + c
            if cell & lasers:
                out["laser"].add(i)
            for name in _STATIC_NAMES + _DYNAMIC_NAMES:
                if idx[name] in cell:
                    out[name].add(i)
    return {k: frozenset(v) for k, v in out.items()}


def _model_state(board: _Board, state):
    """The same dict, from the model, so the two can be compared field by field."""
    p, n, q, k, corpses, hearts, blocks, new_walls = state
    live = {"playerfull": p, "protector": n, "protectorsolo": q, "victimsolo": k}
    out = {name: frozenset() if cell < 0 else frozenset({cell})
           for name, cell in live.items()}
    out["playerdead"] = frozenset(corpses)
    out["pushblock"] = frozenset(blocks)
    out["heart"] = frozenset(c for i, c in enumerate(board.heart_cells)
                             if hearts & (1 << i))
    out["wall"] = board.base_walls | frozenset(new_walls)
    out["exit"] = board.exits
    out["lasergv"] = frozenset(board.gv)
    out["lasergh"] = frozenset(board.gh)
    out["laser"] = frozenset(board.field(state))
    return out


def selfcheck(trials: int = 40, steps: int = 60, verbose: bool = True) -> int:
    """Drive random presses through BOTH the interpreter and `_Board.step`,
    comparing the WHOLE board and the win flag after every one. This is the
    guard that lets the enumeration trust the native model.

    ``action`` is in the press alphabet, and it is what put it there: it was
    modelled as a pure no-op first, on the strength of the rules section, and
    this check caught it converting a block in a beam on four of the five
    chapter-3 levels. That claim is exactly the kind of thing to measure against
    the interpreter rather than read.

    A random walk is a weak prober of chapter 3 on its own -- a block only
    converts if it is shoved into a beam -- so the run also counts the
    conversions and the deaths it saw and prints them, and a level whose
    mechanic never fired is reported rather than passing quietly.

    Returns the number of mismatches."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    g = game._game
    expert = SmotherExpert(game)
    alphabet = _PRESSES
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = expert.board(eng, level)
        rng = random.Random(f"smother:selfcheck:{level}")
        bad = presses = converted = deaths = eaten = 0
        note = ""
        # The level start is a comparison too: it is the one board the model
        # derives rather than steps to (see `_Board.__init__`).
        if _model_state(board, board.start) != _engine_state(eng, g, board):
            bad += 1
            note = "level-start board disagrees"
        for _ in range(trials):
            game.set_level(level)
            eng = game._engine
            state = board.start
            for s in range(steps):
                d = rng.choice(alphabet)
                want = board.step(state, d)
                eng.step(d)
                presses += 1
                converted += len(want[7]) - len(state[7])
                deaths += len(want[4]) - len(state[4])
                eaten += bin(state[5] ^ want[5]).count("1")
                state = want
                if (_model_state(board, state) != _engine_state(eng, g, board)
                        or board.won(state) != eng.check_win()):
                    bad += 1
                    if not note:
                        note = f"diverged at press {s} ({d})"
                    break
                if eng.check_win():
                    break
        total += bad
        if verbose:
            fired = (f"{converted} blocks walled, {deaths} deaths, "
                     f"{eaten} hearts eaten")
            print(f"  L{level:2d}: {'OK' if not bad else f'{bad} MISMATCHES'} "
                  f"({presses} presses vs the interpreter; {fired}) {note}")
    return total


def _walk(board: _Board, plan) -> list:
    """The states ``plan`` passes through, starting at the level start."""
    out, state = [], board.start
    for d in plan:
        out.append(state)
        state = board.step(state, d)
    return out


def _report() -> int:
    """Print each level's space, its shortest plan and the engine's verdict --
    the quick "is this game still fully solved" check."""
    _solver, game, expert = _levels()
    eng = game._engine
    states_total = dead_total = unsolved = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        n_states, n_dead = board.enumerate()
        states_total += n_states
        dead_total += n_dead
        plan = board.plan(board.start)
        if plan is None:
            print(f"  L{level:2d}: UNSOLVED")
            unsolved += 1
            continue
        for d in plan:
            eng.step(d)
        won = eng.check_win()                   # read BEFORE anything reloads
        unsolved += 0 if won else 1
        ties = sum(len(board.optimal(s)) for s in _walk(board, plan))
        print(f"  L{level:2d}: {board.h:2d}x{board.w:<2d} cell_px "
              f"{min(64 // board.h, 64 // board.w)}  {len(plan):3d} presses  "
              f"win={won}  {n_states:6d} states ({n_dead:6d} cannot win)  "
              f"{ties / len(plan):.2f} optimal presses/step")
    print(f"  total: {states_total} states, {dead_total} of them lost; "
          f"{game.n_levels - unsolved}/{game.n_levels} levels solved")
    return 1 if unsolved else 0


def _verify() -> int:
    """Double-entry check of the distance field and of every tie set it ships.

    `_Board.enumerate` answers everything from ONE pass -- a forward sweep that
    builds the successor table and a reverse sweep over the predecessor map it
    inverts to. A bug in that inversion would produce a self-consistent field, a
    plan that still wins, and tie sets that are quietly wrong; and the training
    labels ARE those tie sets, so "it wins" is not enough of a check. Two passes,
    which fail differently:

      * **Bellman fixpoint.** Re-derive every distance by relaxing
        ``d(s) = 1 + min d(succ)`` to a fixpoint over the successor table alone
        -- no predecessor map, no BFS ordering, nothing shared with the reverse
        sweep -- and require it to agree everywhere, plus ``d == 0`` exactly at
        the winning states. Given a transition function that matches the
        interpreter (which `selfcheck` fuzzes) this is a proof of minimality,
        because the enumerated set is closed under every press.
      * **Executable labels.** Walk each level's plan on the INTERPRETER and, at
        every step, actually press each direction the label calls optimal: the
        board the engine lands on must be the native successor, and its field
        distance must be exactly one less. That takes the labels out of the model
        and puts them through the real thing.
    """
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)

        # -- pass 1: Bellman fixpoint over the successor table ----------------
        check = [0 if board.won(s) else -1 for s in board.states]
        changed = True
        while changed:
            changed = False
            for j, row in enumerate(board.succ):
                if row is None:
                    continue
                best = min((check[n] for n in row if check[n] >= 0), default=-1)
                if best >= 0 and (check[j] < 0 or best + 1 < check[j]):
                    check[j] = best + 1
                    changed = True
        if check != board.dist:
            bad += 1
            print(f"  L{level}: the field disagrees with its Bellman fixpoint "
                  f"at {sum(1 for a, b in zip(check, board.dist) if a != b)} "
                  f"states")
        if any((d == 0) != board.won(s)
               for d, s in zip(board.dist, board.states)):
            bad += 1
            print(f"  L{level}: distance 0 is not exactly the winning states")

        # -- pass 2: every shipped label, pressed on the interpreter ----------
        plan = board.plan(board.start)
        state = board.start
        labels = 0
        for i, press in enumerate(plan):
            here = board.distance(state)
            best = board.optimal(state)
            if press not in best:
                bad += 1
                print(f"  L{level}: step {i} is not in its own optimal set")
            before = [[set(cell) for cell in row] for row in eng.grid]
            for alt in best:
                eng.step(alt)
                landed = expert.read(eng, board)
                labels += 1
                if (landed != board.step(state, alt)
                        or board.distance(landed) != here - 1):
                    bad += 1
                    print(f"  L{level}: step {i} labels {alt} optimal, but the "
                          f"interpreter does not land one press closer")
                eng.grid = [[set(cell) for cell in row] for row in before]
                eng._position_index_dirty = True
            eng.step(press)
            state = board.step(state, press)
        if not eng.check_win():
            bad += 1
            print(f"  L{level}: the interpreter does not report a win")
        print(f"  L{level:2d}: {len(board.states):6d} states relaxed, "
              f"{len(plan):3d} plan steps, {labels:3d} labels pressed on the "
              f"interpreter -- {'OK' if not bad else 'see above'}")
    print(f"verify: {bad} problems")
    return 1 if bad else 0


def _compositions(board: _Board) -> list:
    """Every cell COMPOSITION that occurs anywhere in a level's solved state
    space, as a sorted tuple of object names, plus bare floor.

    Read off the enumeration rather than guessed: the interesting compositions
    here are the ones a rule produces mid-play (a beam falling on the Protector
    that is absorbing it, a block standing in a beam for the one turn before it
    becomes a wall, a corpse under a beam), and a hand-written list is exactly
    where such a case goes missing. Laser kind is derived the way the rules
    derive it -- LaserB where a down-beam and a right-beam meet -- so the audit
    checks the three sprites the game can actually draw."""
    board.enumerate()
    out = {()}
    for state in board.states:
        p, n, q, k, corpses, hearts, blocks, new_walls = state
        cells: dict = {}

        def add(cell, name):
            cells.setdefault(cell, set()).add(name)

        for cell in board.base_walls:
            add(cell, "wall")
        for cell in new_walls:
            add(cell, "wall")
        for cell in board.exits:
            add(cell, "exit")
        for cell in board.gv:
            add(cell, "lasergv")
        for cell in board.gh:
            add(cell, "lasergh")
        for i, cell in enumerate(board.heart_cells):
            if hearts & (1 << i):
                add(cell, "heart")
        for cell in blocks:
            add(cell, "pushblock")
        for cell in corpses:
            add(cell, "playerdead")
        for name, cell in (("playerfull", p), ("protector", n),
                           ("protectorsolo", q), ("victimsolo", k)):
            if cell >= 0:
                add(cell, name)
        for cell in board.field(state):
            r, c = divmod(cell, board.w)
            down = any(divmod(i, board.w)[1] == c and divmod(i, board.w)[0] < r
                       for i in board.gv)
            right = any(divmod(i, board.w)[0] == r and divmod(i, board.w)[1] < c
                        for i in board.gh)
            add(cell, "laserb" if down and right else
                ("laserv" if down else "laserh"))
        for objs in cells.values():
            out.add(tuple(sorted(objs)))
    return sorted(out)


def _audit() -> int:
    """Assert every cell composition a level can SHOW renders distinctly, at
    that level's own cell size.

    Three things shipped undrawn here and all of them are silent (see the header
    of data/puzzlescript_games/smother.txt): the beam was composited under Wall,
    pushBlock and PlayerDead; the one-pixel-wide beam sprites were point-sampled
    away at the cell_px 4 and cell_px 2 that levels 4 and 13 render at; and the
    Exit's outer-column ring was sampled away at cell_px 2, which on level 13
    made the goal -- and the win -- invisible.

    Whole frames are compared, not cell crops: `_render_frame` upscales the board
    to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor). The rotation augmentation needs no
    separate pass: a presented frame is ``np.rot90`` of this one, which cannot
    merge two frames that differ."""
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    clashes_total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        comps = _compositions(board)
        h, w = board.h, board.w
        shots = {}
        for objs in comps:
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} for _ in range(w)]
                        for _ in range(h)]
            eng.grid[h // 2][w // 2] = ({idx["background"]}
                                        | {idx[o] for o in objs})
            eng._position_index_dirty = True
            shots[objs] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(comps, 2)
                   if np.array_equal(shots[a], shots[b])]
        clashes_total += len(clashes)
        name = lambda t: "+".join(t) if t else "floor"           # noqa: E731
        print(f"  L{level:2d}: {h:2d}x{w:<2d} cell_px "
              f"{min(64 // h, 64 // w)}  {len(comps):2d} reachable "
              f"compositions  {'OK' if not clashes else 'CLASHES'}")
        for a, b in clashes:
            print(f"       IDENTICAL {name(a)} == {name(b)}")
    print("audit clean" if not clashes_total
          else f"AUDIT FAILED: {clashes_total} indistinguishable pairs")
    return 0 if not clashes_total else 1


def _symmetry() -> int:
    """Prove ON THE ADAPTER that the rotation augmentation is exact, which is the
    one presentation transform this recording relies on.

    The generator plans in ENGINE space and emits the SCREEN press through
    `screen_action`; the adapter forward-remaps that press and rotates the frame
    it renders. Get either half backwards and nothing raises -- the recording
    just drives a different direction than it labels. So for every level and
    every rotation this replays the level's own plan through
    ``perform_action`` and requires two things at once: the level still WINS,
    and every presented frame is exactly ``np.rot90`` of the unrotated one.

    ``Smother`` is not in `PuzzleScriptAdapter._FLIP_GAMES`, so the group is the
    four rotations and there is nothing else to check."""
    _solver, game, expert = _levels()
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(game._engine, level)
        plan = board.plan(board.start)

        runs = {}
        for k in range(4):
            game.set_level(level)
            game._rotation_k = k
            game._current_frame = game._present_frame(
                _render_frame(game._engine, game._game))
            frames = [np.asarray(game._current_frame).copy()]
            for d in plan:
                act = screen_action(d, k, game._hflip, game._vflip)
                fd = game.perform_action(ActionInput(id=act))
                frames.append(np.asarray(
                    fd.frame[-1] if fd.frame else game._current_frame).copy())
            runs[k] = (frames, game._engine.check_win())

        notes = []
        for k in range(4):
            if not runs[k][1]:
                notes.append(f"rot{k}:no win")
            for i, (a, b) in enumerate(zip(runs[0][0], runs[k][0])):
                if not np.array_equal(np.rot90(a, k=k), b):
                    notes.append(f"rot{k}@frame{i}")
                    break
        bad += len(notes)
        print(f"  L{level:2d}: {len(plan):3d}-press plan x 4 rotations: "
              f"{'SYMMETRIC' if not notes else 'BROKEN ' + ', '.join(notes[:3])}")
    print("symmetry clean -- the rotation contract holds" if not bad
          else f"SYMMETRY FAILED: {bad} presentations")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        bad = selfcheck()
        print(f"selfcheck: {bad} mismatches")
        sys.exit(1 if bad else _report())
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    sys.exit(SmotherSolver.main())
