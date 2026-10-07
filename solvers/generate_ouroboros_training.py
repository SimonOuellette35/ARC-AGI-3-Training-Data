"""Generate Phase-1 training data for the PuzzleScript game ps:ouroboros
("Ouroboros" by Loneship Games).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, the native model of the turn, the exact distance field over the
whole state space, and the optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_ouroboros",
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
presented view, so replaying the recorded actions reproduces the recorded frames
exactly.

The game
--------
(Levels are numbered 1..6 in this docstring, as the .txt's LEVELS block writes
them; ``--plans`` prints them 0-based, L0..L5.)

A snake that must EAT ITS OWN TAIL. The title is the win condition:
``[> player|tail] -> win``, and every other rule exists to make the snake behave
like a snake with objects rather than with a stored list.

  * The body is written on the BOARD. When the head moves in direction D the
    rule ``up [> player| ] -> [> player|u]`` stamps a marker object naming D in
    the destination cell, and the head then moves onto it -- ``player`` has its
    own collision layer, so head and marker share the cell. So a marker in a
    cell says "the snake travelled D to get here", which makes the body a
    self-describing linked list.
  * The tail walks that list. ``late up [tail|u] -> [ |tail2]`` and its three
    siblings move the tail one cell per turn into whichever neighbour holds the
    marker pointing away from it -- and that neighbour is unambiguous, because a
    ``u`` above the tail can only have been entered from the tail's own cell.
    The marker is CONSUMED by the move, and ``[tail2] -> [tail]`` at the top of
    the next turn re-arms it. (So the resting board always shows ``tail2``, never
    ``tail``: the two are one object as far as any frame is concerned.)
  * Head and tail therefore both advance one cell per press, so as long as the
    snake only walks, the body length is CONSTANT. That is the puzzle the levels
    were built for. The head must step onto the cell the tail is standing on, so
    its trajectory has to revisit a cell exactly L presses later -- i.e. the
    snake has to lay itself out as a CLOSED LOOP of exactly its own length, in a
    maze, without ever crossing itself on the way. Grid cycles are even, and
    every level's body length is even, level 6's after its apple.
  * Two items break the length. ``[> player|apple] [tail] -> [> player| ]
    [tail2]`` eats a green apple: the head advances and the tail is re-armed
    instead of moving, so the snake GROWS by one. ``[> player|redapple] ->
    [player| ]`` is the opposite -- the ``>`` is dropped, so the head does not
    move at all while the tail still advances, and the snake SHRINKS. Level 6
    ships the only item in the game, one green apple; it is not needed once
    ACTION is on the table (see below), but without ACTION it is forced, since
    the body starts at 15 cells and no odd cycle exists.
  * A press into a wall or into the snake's own body is ``-> cancel``, i.e. a
    total no-op -- the tail does not advance either. A press into the board EDGE
    is NOT: nothing cancels it, the head simply fails to move, no marker is
    stamped, and the late rules still run, so walking into open space off the
    map shrinks the snake. Every shipped level is walled, so this never happens
    here, but the model implements it and ``--selfcheck`` fuzzes it on a
    synthetic unwalled board rather than assuming it away.
  * **ACTION is not a no-op, and it is the most important thing about this
    game's action space.** The adapter offers every ps: game five keys
    (``available_actions=[1, 2, 3, 4, 5]``) and the rules section never mentions
    ACTION -- which reads as "nothing happens", and for most of this family it
    is true. Here it lands in exactly the same branch as the map edge: the press
    carries no movement force, so no ``<dir> [> player| ] -> [> player|<mark>]``
    rule matches and the head stays put, but the ``late [tail|<mark>]`` family
    still runs and the tail advances. ACTION is therefore a SHRINK button, and
    a snake that can shorten itself only has to find a smaller loop -- with the
    two-cell snake (head adjacent to tail, reverse into it) as the floor.
    Measured over the six levels, ``d*`` with the four arrows alone is
    4/16/23/26/24/6; with ACTION it is 4/9/12/6/14/5, so four of the six levels
    are more than halved and level 4 goes from 26 presses to 6. The search
    branches on all five keys, because an expert that did not would label a
    26-press route optimal with a 6-press one in front of the agent.

    That is a corpus decision worth stating plainly, because the shrink is
    degenerate: level 2's shortest solution is eight ACTIONs followed by one
    step (shrink to a two-cell snake, then reverse into the tail), which is
    nothing like the loop-building puzzle the levels were authored for. It is
    shipped anyway because it is what the environment DOES, and a target set
    that omitted it would be wrong. If the intended puzzle is wanted instead,
    the fix is one rule at the top of the RULES section of
    data/puzzlescript_games/Ouroboros.txt -- ``[ action player ] -> cancel``,
    measured to make ACTION a total no-op -- after which ``_DIRS`` drops
    ``"action"`` and every number in this file goes back to the four-arrow
    column above. That is a change to the GAME, not to the model, so it is not
    made here.

The head's own marker is the one piece of board state a frame never shows -- the
head sprite is opaque -- and it never needs to. At L >= 3 no rule reads it: the
only reader is the tail advance and the tail's successor is the L-1'th marker,
not the head's. At L == 2 it IS read, but its value is forced (the head entered
its cell from the tail's cell, so the marker can only point tail -> head). At
L == 1 head and tail share a cell, which the frame states plainly by showing a
one-cell snake. See `_EXPECTED_IDENTICAL`, which is where ``--audit`` asserts
the four head compositions render alike rather than forbidding it.

Model + search
--------------
`_Board.step` re-implements one turn natively; ``--selfcheck`` drives ~28k random
presses through BOTH the interpreter and the model across the six levels plus
three synthetic boards (a redapple field, a two-cell snake shrinking to one cell,
and an unwalled map, none of which any shipped level can reach) and compares the
reconstructed snake and the win flag after every one. That is the guard that lets
the search trust it. The interpreter runs at ~290 presses/s here, so it is also
what makes the search affordable at all.

The state is the body as a PATH of cells plus which items are left, and the snake
fills so much of every board that the reachable space stays enumerable: 2.48M
states for the whole game, 1.44M of them level 1's (an open 7x7 room, where the
snake really can go anywhere) and 755k level 6's (a 5x5 room plus the apple bit),
against 3k-160k on the four maze levels. So there is no search. `_Board.enumerate`
walks the space once from the level start, keeps the successor table and runs one
backward BFS from the states that win in one press to get the exact
distance-to-win field. A plan is a descent of the field, a tie set is a lookup,
and a state reached by an exploration detour is answered from the same table as
the start state. The whole game is built in ~9 s, once per process and shared by
every seed, for ~450 MB of field on top of the interpreter's own ~90 MB.

A state is a ``bytes``: one byte per body cell from tail to head, plus a
trailing byte of item bitmask. That is the memory budget -- a tuple of ints per
state was 1.8x the resident set for the identical field -- and it is exact, so
the grid is never snapshotted.

Unlike most of this family the win here is a TRANSITION, not a board: there is no
"won state" to give distance 0, because the interpreter latches ``_rule_win``
during the press and the level ends. So the field starts at 1 on the states with
a winning press.

Optimal-action targets and recovery
-----------------------------------
`optimal_for` reads the LIVE engine state and returns every press that lowers the
exact distance -- true tie sets, not a reordering heuristic. What produces the
ties here is mostly ACTION: a shrink and a step toward the loop are frequently
interchangeable for the first few presses, so ``--plans`` measures 1.00, 1.00,
1.25, 1.50, 1.64 and 1.60 optimal presses per step over the six shortest plans.
Labelling one of those as the single right answer would train a coin flip the
policy cannot win.

``epsilon = 0.10`` on top of the explore-then-RESET prefix. This game HAS dead
states, and they are exactly one shape: a snake that has shrunk to a single cell
can never win (the head can never step onto its own cell) and can never grow
again unless an apple is still on the board. That is 281 states out of 2.48M --
one per free cell, plus level 6's 24 one-cell snakes that CAN still eat their way
back to two. A detour is nonetheless probed before it is committed:
`record_level` only accepts an alternative that leaves ``expert.plan`` non-None,
and with the exact field that test is a dict lookup rather than a search. The
step then records the mistake as the action taken and the recovery as
``optimal``.

Rendering
---------
Colours and four sprites CHANGED, in the game FILE (see the comment at the top of
data/puzzlescript_games/Ouroboros.txt). The four body markers are the game's
entire memory and all four shipped as ``green darkgreen``, which the ARC palette
collapses onto a single index -- so the body, the tail and the apple were one
flat green and the frame did not say where the tail would move next. Each marker
now carries a black bar on the edge it flows toward, drawn as a full edge because
level 5 renders at ``cell_px`` 4 where the centred nearest sample drops sprite
row/column 2 and the original arrows lived entirely on rows 1..3. The head, the
tail and the two apples were recoloured for the same reason. ``--audit`` is
the check, and it compares WHOLE FRAMES rather than cell crops (`_render_frame`
upscales the board to fill 64x64 and letterboxes it, so an arithmetic cell crop
reads the wrong window).

Augmentation
------------
Per (seed, level) the adapter draws a frame rotation (0..3) and, since
``Ouroboros`` is in `PuzzleScriptAdapter._FLIP_GAMES`, an independent horizontal
and vertical flip -- 16 presentations of each of the 6 levels. The markers are
directional ART, and the marker rule and the tail advance are both written as
four per-direction blocks, which is exactly the shape that hid ps:gobble_rush's
chirality -- so the flips are MEASURED, not argued: ``--symmetry`` rebuilds every
level's LAYOUT under all 8 transforms (relabelling u/d/l/r through the same
transform), reloads it into the interpreter, and replays the level's own plan
plus a 400-press random walk with the presses transformed; every object must land
where the transform says after every one.

Usage (run from the repo root):
    python solvers/generate_ouroboros_training.py --episodes 200 \
        --out data/training_multi_level/ouroboros
    python solvers/generate_ouroboros_training.py --selfcheck
    python solvers/generate_ouroboros_training.py --plans
    python solvers/generate_ouroboros_training.py --verify
    python solvers/generate_ouroboros_training.py --audit
    python solvers/generate_ouroboros_training.py --symmetry
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
    PSAStarSolver, PSExpert, restore, snapshot)

GAME_NAME = "Ouroboros"

#: Engine direction -> (dr, dc).
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: The action space the search branches on -- all five keys the adapter offers
#: (``available_actions=[1, 2, 3, 4, 5]``), because in this game ACTION is not the
#: no-op the rules section suggests. See `_Board.step`.
_DIRS = ("up", "down", "left", "right", "action")

#: Branching factor of the successor table: one row of `_NA` entries per state.
_NA = len(_DIRS)

#: Successor-table entry meaning "this press ends the level in a win". There is
#: no won STATE to point at (see the module docstring), so the win is an edge.
_WIN = -1

#: The body marker naming each travel direction, and the objects that make up a
#: snake. ``tail`` never rests on the board -- ``[tail2] -> [tail]`` runs at the
#: top of a turn and the late rules put ``tail2`` back -- but both are read.
_MARKER = {"u": "up", "d": "down", "l": "left", "r": "right"}
_TAILS = ("tail", "tail2")


# ---------------------------------------------------------------------------
# The level, natively: one turn, the whole state space, the distance field
# ---------------------------------------------------------------------------

class _Board:
    """A level's static geometry plus its COMPLETE state space and the exact
    distance-to-win field over it.

    Cells are flat ``r * w + c`` ids and a STATE is a ``bytes``: ``state[:-1]``
    is the snake from TAIL to HEAD, one byte per cell, and ``state[-1]`` is a
    bitmask over `items` saying which are still on the board. Nothing else in a
    level ever changes -- walls are static, the head's own marker is dead state
    (see the module docstring), and ``tail`` vs ``tail2`` is a within-turn
    distinction that never survives a press -- so that bytes object is an exact
    canonical key, and the grid is never snapshotted.

    `enumerate` is what everything else reads: one forward sweep from the level
    start collecting every reachable state and its four successors, then one
    backward BFS from the states that win in one press. `plan` is a descent down
    the field, `optimal` a lookup, and both work from ANY reachable state --
    which is what lets the recording take real detours and label the recovery.
    """

    __slots__ = ("h", "w", "walls", "items", "reds", "_bit", "_ahead", "start",
                 "states", "index", "succ", "dist")

    def __init__(self, h: int, w: int, walls, items, reds, start_body):
        self.h, self.w = h, w
        self.walls = frozenset(walls)
        #: Item cells in a fixed order; bit i of a state's mask is `items[i]`.
        self.items = tuple(sorted(items))
        self.reds = frozenset(reds)
        assert len(self.items) <= 8, "the item mask is one byte"
        assert h * w <= 256, "a body cell is one byte"
        self._bit = {c: 1 << i for i, c in enumerate(self.items)}
        self.start = bytes(start_body) + bytes([(1 << len(self.items)) - 1])

        # ``_ahead[d][cell]`` is the cell one step along d, or None when that is
        # off the board -- which is NOT the same as a wall here (a wall cancels
        # the turn, the map edge shrinks the snake), so the two are kept apart.
        # Precomputed because it is the inner loop of the enumeration.
        self._ahead = {}
        for d, (dr, dc) in _DELTA.items():
            table: list = [None] * (h * w)
            for r in range(h):
                for c in range(w):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        table[r * w + c] = nr * w + nc
            self._ahead[d] = table

        self.states: list | None = None
        self.index: dict = {}
        self.succ = array("i")
        self.dist = array("i")

    # -- one turn -------------------------------------------------------------
    def step(self, state: bytes, d: str) -> bytes | None:
        """One press, natively. Returns the next state, or None for a WIN.

        The outcomes, in the interpreter's own rule order: eat the tail (win),
        refuse into a wall or into the body (a `cancel`, so nothing moves at
        all), eat a green apple (head advances, tail re-armed -> grow), eat a
        redapple / press ACTION / walk off the map edge (head stays, tail
        advances -> shrink), or the ordinary move (both ends advance).

        ACTION lands in the shrink branch for the same reason the map edge does,
        and it is the single most important fact about this game's action space:
        nothing cancels the press, no ``<dir> [> player| ] -> [> player|<mark>]``
        rule matches without a movement force, so no marker is stamped and the
        head does not move -- but the ``late [tail|<mark>]`` family still runs.
        The rules section never mentions ACTION, so this is emergent, and it is
        not small: it takes level 4 from 26 presses to 6, because a snake that can
        shorten itself only has to find a SMALLER loop. A search over the four
        arrows alone would label a 26-press route optimal with a 6-press one in
        front of the agent."""
        body, mask = state[:-1], state[-1]
        if d == "action":                        # no force -> no marker stamped
            return (body[1:] or body) + bytes([mask])
        nxt = self._ahead[d][body[-1]]
        if nxt is None:                          # off the map: no marker stamped
            return (body[1:] or body) + bytes([mask])
        if nxt in self.walls:
            return state                         # [> player|wall] -> cancel
        if nxt == body[0]:
            return None                          # [> player|tail] -> win
        if nxt in body[1:]:
            return state                         # [> player|snake] -> cancel
        bit = self._bit.get(nxt, 0)
        if bit & mask:
            if nxt in self.reds:                 # shrink: the `>` is dropped
                return (body[1:] or body) + bytes([mask & ~bit])
            return body + bytes([nxt, mask & ~bit])          # grow
        return body[1:] + bytes([nxt, mask])                 # ordinary move

    # -- the whole space ------------------------------------------------------
    def enumerate(self) -> tuple[int, int]:
        """Build the reachable state set, its successor table and the exact
        distance-to-win field. Returns ``(states, dead states)``.

        The set is CLOSED under every press (each state contributes all four of
        its successors), so any state the agent can reach by any sequence of
        presses -- plan, exploration prefix, epsilon detour, or a RESET back to
        the start -- is already in it. A winning press is an EDGE to `_WIN`, not
        a state: the level ends there.

        Predecessors are held as a CSR pair rather than a list of lists, which is
        what keeps the biggest level (766k states) inside ~120 MB.
        """
        if self.states is not None:
            return len(self.states), sum(1 for x in self.dist if x < 0)

        states = [self.start]
        index = {self.start: 0}
        succ = array("i")
        ahead, walls, bits, reds = self._ahead, self.walls, self._bit, self.reds
        i = 0
        while i < len(states):
            state = states[i]
            body, mask = state[:-1], state[-1]
            head = body[-1]
            for d in _DIRS:
                nxt = None if d == "action" else ahead[d][head]
                if nxt is None:
                    nb = (body[1:] or body) + bytes([mask])
                elif nxt in walls or nxt in body[1:]:
                    succ.append(i)               # cancel: a self-loop
                    continue
                elif nxt == body[0]:
                    succ.append(_WIN)
                    continue
                else:
                    bit = bits.get(nxt, 0)
                    if bit & mask:
                        nb = ((body[1:] or body) + bytes([mask & ~bit])
                              if nxt in reds else
                              body + bytes([nxt, mask & ~bit]))
                    else:
                        nb = body[1:] + bytes([nxt, mask])
                j = index.get(nb)
                if j is None:
                    j = len(states)
                    index[nb] = j
                    states.append(nb)
                succ.append(j)
            i += 1

        n = len(states)
        starts = array("i", bytes(4 * (n + 1)))
        for k in range(_NA * n):
            dst = succ[k]
            if dst >= 0:
                starts[dst + 1] += 1
        for k in range(n):
            starts[k + 1] += starts[k]
        fill = array("i", starts[:n])
        pred = array("i", bytes(4 * starts[n]))
        for k in range(_NA * n):
            dst = succ[k]
            if dst >= 0:
                pred[fill[dst]] = k // _NA
                fill[dst] += 1

        dist = array("i", bytes(4 * n))
        queue = deque()
        for j in range(n):
            if _WIN in succ[_NA * j:_NA * j + _NA]:
                dist[j] = 1
                queue.append(j)
            else:
                dist[j] = -1
        while queue:
            j = queue.popleft()
            for k in range(starts[j], starts[j + 1]):
                src = pred[k]
                if dist[src] < 0:
                    dist[src] = dist[j] + 1
                    queue.append(src)

        self.states, self.index, self.succ, self.dist = states, index, succ, dist
        return n, sum(1 for x in dist if x < 0)

    # -- reading the field ----------------------------------------------------
    def _at(self, j: int, k: int) -> int:
        """The distance of the state press ``k`` leads to, with a win at 0."""
        nxt = self.succ[_NA * j + k]
        return 0 if nxt == _WIN else self.dist[nxt]

    def distance(self, state: bytes) -> int | None:
        """Presses to a win from ``state``, or None if it cannot win (or is not
        reachable from the level start, which cannot happen -- see `enumerate`)."""
        self.enumerate()
        j = self.index.get(state)
        if j is None or self.dist[j] < 0:
            return None
        return self.dist[j]

    def plan(self, state: bytes) -> list | None:
        """A SHORTEST press sequence from ``state`` to a win, or None.

        A descent of the field, taking the first tied direction in ``_DIRS``
        order, so the plan is a pure function of the board and does not vary with
        the interpreter's hash seed."""
        self.enumerate()
        j = self.index.get(state)
        if j is None or self.dist[j] < 0:
            return None
        out = []
        while self.dist[j] > 0:
            for k in range(_NA):
                if self._at(j, k) == self.dist[j] - 1:
                    out.append(_DIRS[k])
                    j = self.succ[_NA * j + k]
                    break
            else:                                    # pragma: no cover
                raise AssertionError("distance field has no descent")
            if j == _WIN:
                break
        return out

    def optimal(self, state: bytes) -> list:
        """Every press that keeps the game on a SHORTEST route to the win.

        Exact, straight off the field. A press into a wall or into the body maps
        the state to itself, whose distance is unchanged rather than one less, so
        no-ops are excluded for free."""
        self.enumerate()
        j = self.index.get(state)
        if j is None or self.dist[j] <= 0:
            return []
        here = self.dist[j]
        return [_DIRS[k] for k in range(_NA) if self._at(j, k) == here - 1]


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class OuroborosExpert(PSExpert):
    """Exact planner over the enumerated state space (see `_Board`).

    `PSExpert` supplies the plan memo, the restore discipline and the level
    scoping; `_search` swaps in the field descent, so the interpreter is only
    ever stepped by the recorder -- which is also what verifies every plan, since
    a level is kept only when the engine reports WIN.
    """

    directions = list(_DIRS)

    #: `_key` is the snake plus the item mask, canonical only WITHIN a level (the
    #: walls that complete the state are static per level but differ between
    #: them, and two levels can hold the same snake).
    scope_by_level = True

    def setup(self) -> None:
        idx = self.g.obj_name_to_idx
        self.wall_id = idx["wall"]
        self.head_id = idx["head"]
        self.apple_id = idx["apple"]
        self.red_id = idx["redapple"]
        self.marker_dir = {idx[n]: d for n, d in _MARKER.items()}
        self.tail_ids = {idx[n] for n in _TAILS}
        self._boards: dict[int | None, _Board] = {}
        self._cur: _Board | None = None

    # -- reading the engine ---------------------------------------------------
    def _scan(self, eng):
        """``(walls, items, reds, tail, head, markers)`` off the engine grid."""
        w = len(eng.grid[0])
        walls, items, reds, markers = set(), set(), set(), {}
        tail = head = None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                p = r * w + c
                if self.wall_id in cell:
                    walls.add(p)
                if self.apple_id in cell:
                    items.add(p)
                if self.red_id in cell:
                    items.add(p)
                    reds.add(p)
                if cell & self.tail_ids:
                    tail = p
                if self.head_id in cell:
                    head = p
                for o in cell:
                    d = self.marker_dir.get(o)
                    if d is not None:
                        markers[p] = d
        assert tail is not None and head is not None, "no snake on the board"
        return walls, items, reds, tail, head, markers

    def _chain(self, eng, tail: int, head: int, markers: dict) -> list:
        """The snake from TAIL to HEAD, walked off the markers.

        Unambiguous by construction: a ``u`` in the cell above X can only have
        been stamped by a head that entered it from X, so at most one neighbour
        of the tail is its own successor. The walk stops at the head, which is
        why the marker UNDER the head is never read (and need not be drawn)."""
        w = len(eng.grid[0])
        h = len(eng.grid)
        body, seen, cur = [tail], {tail}, tail
        while cur != head:
            r, c = divmod(cur, w)
            for d, (dr, dc) in _DELTA.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                p = nr * w + nc
                if markers.get(p) == d and p not in seen:
                    body.append(p)
                    seen.add(p)
                    cur = p
                    break
            else:                                    # pragma: no cover
                raise AssertionError("the snake's markers do not reach the head")
        return body

    def read(self, eng, board: _Board) -> bytes:
        """The engine's current state, in `_Board`'s encoding."""
        _walls, items, _reds, tail, head, markers = self._scan(eng)
        body = self._chain(eng, tail, head, markers)
        mask = 0
        for i, cell in enumerate(board.items):
            if cell in items:
                mask |= 1 << i
        return bytes(body) + bytes([mask])

    def board(self, eng, level: int | None) -> _Board:
        """The level's `_Board`, built (and enumerated) once and cached.

        It must be built from a level START, because the items it indexes are
        eaten as the level is played -- `OuroborosSolver.prepare_expert` builds
        every level's board up front so this is a hit at every later call."""
        board = self._boards.get(level)
        if board is not None:
            return board
        walls, items, reds, tail, head, markers = self._scan(eng)
        body = self._chain(eng, tail, head, markers)
        board = _Board(len(eng.grid), len(eng.grid[0]),
                       walls, items, reds, body)
        board.enumerate()
        self._boards[level] = board
        return board

    # -- planning -------------------------------------------------------------
    def _key(self, eng):
        return self.read(eng, self._cur)

    def plan(self, eng, level: int | None = None) -> list | None:
        self._cur = self.board(eng, level)
        return super().plan(eng, level)

    def heuristic(self, eng) -> int:
        """The EXACT remaining press count (the base `_astar` is never used here,
        but the contract is that this is admissible, and an exact field is)."""
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

class OuroborosSolver(PSAStarSolver):
    game_id = "puzzlescript_ouroboros"
    game_name = GAME_NAME
    expert_cls = OuroborosExpert

    #: Unused -- `OuroborosExpert._search` never calls the base A* -- but left at
    #: the family default so a future subclass that does is not silently starved.
    node_cap = 2_000_000
    weight = 1
    #: The adapter GAME_OVERs at 200 actions per level; the longest plan here is
    #: 14 presses, leaving the rest of the budget to the exploration prefix and
    #: the detours.
    max_steps = 200

    #: Non-zero even though this game HAS dead states. They are exactly one
    #: shape, which is why the risk is bounded: a snake shrunk to a SINGLE cell
    #: can never win (the head cannot step onto its own cell) and can only grow
    #: again by eating an apple, so 281 of the game's 2.48M reachable states are
    #: lost -- one per free cell, minus level 6's 24 that still have their apple
    #: (``--plans`` prints the column). `record_level`'s detour probe keeps only
    #: an alternative that leaves ``expert.plan`` non-None, and with the exact
    #: field in hand that test is a dict lookup, not a search, so the guard is
    #: affordable at every step. What it buys is the signal a policy needs after
    #: its OWN error: the taken action is the mistake and ``optimal`` is the
    #: recovery, on top of the RESET prefix that still runs in front of every
    #: episode.
    epsilon = 0.10

    def prepare_expert(self, game, expert) -> None:
        """Enumerate every level's state space up front.

        Not just front-loading here, unlike the rest of this family: a `_Board`
        indexes the items by their START cells, so it has to be built before
        anything eats one."""
        for level in range(game.n_levels):
            game.set_level(level)
            expert.board(game._engine, level)

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """Every press that keeps the game on a SHORTEST route to the win, read
        off the LIVE engine state.

        `record_level` asks for this before executing the press, so the engine is
        still at the state being labelled -- which is what makes the label right
        after an epsilon detour too, where the state is off the plan entirely and
        replaying the plan from the level start would label the wrong states.

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
    solver = OuroborosSolver()
    game, expert, _solvable = solver._ensure(0)
    return solver, game, expert


#: Synthetic boards `selfcheck` adds to the six shipped levels, each one covering
#: a branch of `_Board.step` that no level can reach. Without them the redapple
#: shrink, the length-2 snake and the open map edge would be modelled from the
#: rules section and never measured -- and the edge case in particular is a
#: branch the rules do NOT state (see the module docstring).
_SYNTHETIC = {
    "redapples": [
        "#######",
        "#.3llu#",
        "#t.@.u#",
        "#d...u#",
        "#d.@.u#",
        "#drrrr#",
        "#######",
    ],
    "two-cell snake in a redapple field": [
        "#######",
        "#..3t.#",
        "#.@@@.#",
        "#.....#",
        "#.@@@.#",
        "#.....#",
        "#######",
    ],
    "no walls at all": [
        ".3llu.",
        "t...u.",
        "d...u.",
        "drrrr.",
        "......",
    ],
}

#: Level-layout characters, as the game's own LEGEND declares them.
_CHARS = {"#": ("wall",), ".": (), "t": ("tail2",), "u": ("u",), "d": ("d",),
          "l": ("l",), "r": ("r",), "1": ("head", "u"), "2": ("head", "d"),
          "3": ("head", "l"), "4": ("head", "r"), "a": ("apple",),
          "@": ("redapple",)}


def _layout(g, rows) -> list:
    """A loadable level grid from a block of legend characters."""
    idx = g.obj_name_to_idx
    return [[{idx["background"]} | {idx[o] for o in _CHARS[ch]} for ch in row]
            for row in rows]


def _fuzz(game, expert, board, layout, rng, trials, steps, alphabet):
    """Drive random presses through BOTH the interpreter and `_Board.step`,
    comparing the reconstructed snake AND the win flag after every one.

    Returns ``(mismatches, presses)``."""
    eng = game._engine
    bad = presses = 0
    for _ in range(trials):
        eng.load_level(layout)
        state = expert.read(eng, board)
        for _ in range(steps):
            d = rng.choice(alphabet)
            want = board.step(state, d)
            eng.step(d)
            presses += 1
            if eng.check_win() != (want is None):
                bad += 1
                break
            if want is None:
                break
            if expert.read(eng, board) != want:
                bad += 1
                break
            state = want
            # The closure `_Board.enumerate` relies on, checked rather than
            # argued: a state a random walk from the start can reach is already
            # in the enumerated space, so the field can answer for anything the
            # exploration prefix or an epsilon detour lands on.
            if board.states is not None and state not in board.index:
                bad += 1
                break
    return bad, presses


def selfcheck(trials: int = 40, steps: int = 120, verbose: bool = True) -> int:
    """Fuzz the native model against the interpreter on every shipped level and
    on the three synthetic boards in `_SYNTHETIC`.

    ``action`` is in the alphabet because it is in the SEARCH -- the rules section
    never mentions it and it is not a no-op (see `_Board.step`), which is exactly
    why the claim is measured against the interpreter instead of read off the
    rules.

    Returns the number of mismatches."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    expert = OuroborosExpert(game)
    eng, g = game._engine, game._game
    alphabet = _DIRS
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        rng = random.Random(f"ouroboros:selfcheck:{level}")
        bad, presses = _fuzz(game, expert, board, g.levels[level], rng,
                             trials, steps, alphabet)
        total += bad
        if verbose:
            print(f"  L{level}: {'OK' if not bad else f'{bad} MISMATCHES'} "
                  f"({presses} presses vs the interpreter)")
    for name, rows in _SYNTHETIC.items():
        layout = _layout(g, rows)
        eng.load_level(layout)
        walls, items, reds, tail, head, markers = expert._scan(eng)
        board = _Board(len(layout), len(layout[0]), walls, items, reds,
                       expert._chain(eng, tail, head, markers))
        rng = random.Random(f"ouroboros:selfcheck:{name}")
        bad, presses = _fuzz(game, expert, board, layout, rng,
                             trials, steps, alphabet)
        total += bad
        if verbose:
            print(f"  synthetic ({name}): "
                  f"{'OK' if not bad else f'{bad} MISMATCHES'} "
                  f"({presses} presses vs the interpreter)")
    return total


def _walk(board: _Board, plan) -> list:
    """The states ``plan`` passes through, starting at the level start."""
    out, state = [], board.start
    for d in plan:
        out.append(state)
        state = board.step(state, d)
        if state is None:
            break
    return out


def _report() -> int:
    """Print each level's space, its shortest plan and the engine's verdict --
    the quick "is this game still fully solved" check.

    It also CLASSIFIES the dead states rather than only counting them, because
    ``epsilon`` is set on the strength of what they are: every state from which
    the win is unreachable must be a snake shrunk to a single cell with no apple
    left, which is a shape the epsilon detour's probe rejects on the press that
    would create it. A dead state of any other shape would mean a snake can trap
    itself in the maze, and the detour rate should be reconsidered."""
    _solver, game, expert = _levels()
    eng = game._engine
    states_total = dead_total = odd_total = 0
    unsolved = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        n_states, n_dead = board.enumerate()
        states_total += n_states
        dead_total += n_dead
        odd = sum(1 for j, st in enumerate(board.states)
                  if board.dist[j] < 0 and (len(st) - 1 > 1 or st[-1]))
        odd_total += odd
        plan = board.plan(board.start)
        if plan is None:
            print(f"  L{level}: UNSOLVED")
            unsolved += 1
            continue
        for d in plan:
            eng.step(d)
        won = eng.check_win()                   # read BEFORE anything reloads
        steps = _walk(board, plan)
        ties = sum(len(board.optimal(s)) for s in steps)
        print(f"  L{level}: {eng.height}x{eng.width}  snake {len(board.start)-1:2d}"
              f"  {len(plan):3d} presses  win={won}  "
              f"{n_states:7d} states ({n_dead:3d} dead, {odd:3d} of them not a "
              f"one-cell snake)  {ties / len(plan):.2f} optimal presses/step")
    print(f"  total: {states_total} states, {dead_total} dead "
          f"({100 * dead_total / states_total:.2f}%)"
          + (" -- every one of them a snake shrunk to a SINGLE cell with no "
             "apple left to grow it back" if not odd_total else
             f" -- {odd_total} of them are NOT one-cell snakes, so a snake CAN "
             f"trap itself: re-check epsilon"))
    return 1 if unsolved else 0


def _verify() -> int:
    """Double-entry check of the distance field and of every tie set it ships.

    `_Board.enumerate` answers everything from ONE pass -- a forward sweep that
    builds the successor table and a reverse sweep over the CSR predecessor map
    it inverts to. A bug in that inversion would produce a self-consistent field,
    a plan that still wins, and tie sets that are quietly wrong; and the training
    labels ARE those tie sets, so "it wins" is not enough of a check. Two passes,
    which fail differently:

      * **Bellman fixpoint.** Re-derive every distance by relaxing
        ``d(s) = 1 + min d(succ)`` to a fixpoint over the successor table alone
        -- no predecessor map, no BFS ordering, nothing shared with the reverse
        sweep -- and require it to agree everywhere. Given a transition function
        that matches the interpreter (which `selfcheck` fuzzes) this is a proof
        of minimality, because the enumerated set is closed under every press.
      * **Executable labels.** Walk each level's plan on the INTERPRETER and, at
        every step, actually press each direction the label calls optimal: the
        board the engine lands on must be the native successor, and its field
        distance must be exactly one less (or the press must WIN). That takes the
        labels out of the model and puts them through the real thing.
    """
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        n = len(board.states)

        # -- pass 1: Bellman fixpoint over the successor table ----------------
        # Vectorised because the biggest level is 1.4M states x 5 presses and the
        # relaxation needs one sweep per layer of the field; the loop is still a
        # plain "d(s) = 1 + min d(succ)" to a fixpoint, sharing nothing with
        # `enumerate`'s predecessor inversion.
        big = n + 2
        succ = np.frombuffer(board.succ, dtype=np.int32).reshape(n, _NA)
        wins = (succ == _WIN)
        edge = np.where(wins, 0, np.maximum(succ, 0))
        check = np.where(wins.any(axis=1), 1, big).astype(np.int64)
        while True:
            nxt = np.where(wins, 0, check[edge])
            relaxed = np.minimum(check, nxt.min(axis=1) + 1)
            if np.array_equal(relaxed, check):
                break
            check = relaxed
        check = np.where(check >= big, -1, check)
        wrong = int((check != np.frombuffer(board.dist, dtype=np.int32)).sum())
        if wrong:
            bad += 1
            print(f"  L{level}: the field disagrees with its Bellman fixpoint "
                  f"at {wrong} states")

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
            before = snapshot(eng)
            for alt in best:
                eng.step(alt)
                landed = board.step(state, alt)
                if landed is None:
                    ok = eng.check_win() and here == 1
                else:
                    ok = (not eng.check_win()
                          and expert.read(eng, board) == landed
                          and board.distance(landed) == here - 1)
                if not ok:
                    bad += 1
                    print(f"  L{level}: step {i} labels {alt} optimal, but the "
                          f"interpreter does not land one press closer")
                labels += 1
                restore(eng, before)
            eng.step(press)
            state = board.step(state, press)
        if not eng.check_win():
            bad += 1
            print(f"  L{level}: the interpreter does not report a win")
        print(f"  L{level}: {n:7d} states relaxed, {len(plan):3d} plan steps, "
              f"{labels:3d} labels pressed on the interpreter -- "
              f"{'OK' if not bad else 'see above'}")
    print(f"verify: {bad} problems")
    return 1 if bad else 0


def _transform(k: int, mirror: bool):
    """``(cell_fn, dims_fn, direction_map)`` for one element of the 8-group:
    mirror left-right first, then turn clockwise ``k`` times.

    The direction map is DERIVED from the same linear part that moves the cells,
    so the two cannot drift apart -- and it is what RELABELS the body markers,
    which are directional art (a ``u`` turned clockwise is an ``r``)."""
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
    # ACTION is non-directional -- `screen_action` passes it through untouched at
    # every presentation -- but its EFFECT is the tail advance, which is stated as
    # four per-direction rules, so the walk below presses it and this is what lets
    # it through the transform unchanged.
    dmap["action"] = "action"
    return cell, dims, dmap


def _replay_transformed(eng, g, layout, presses) -> list[str]:
    """Replay ``presses`` on all seven non-identity turned/mirrored copies of
    ``layout`` and return a note per presentation that diverged.

    The transformed board is built from the LEVEL LAYOUT and reloaded, so the
    interpreter parses and resolves everything itself rather than being handed a
    grid this file transformed after the fact. Every object is compared, not just
    the head: a chirality in the tail advance would move the wrong end."""
    idx = g.obj_name_to_idx
    name_of = {v: k for k, v in idx.items()}
    bg = idx["background"]
    hw = (len(layout), len(layout[0]))

    def board(e):
        return [[frozenset(name_of[o] for o in cell if o != bg) for cell in row]
                for row in e.grid]

    eng.load_level(layout)
    ref = [board(eng)]
    for d in presses:
        eng.step(d)
        ref.append(board(eng))

    notes = []
    for k, mirror in itertools.product(range(4), (False, True)):
        if (k, mirror) == (0, False):
            continue
        cell, dims, dmap = _transform(k, mirror)
        th, tw = dims(hw)
        #: The direction-carrying objects have to be relabelled through the same
        #: transform: ``u`` names a motion, not a colour.
        relabel = {n: next(m for m, d in _MARKER.items() if d == dmap[dd])
                   for n, dd in _MARKER.items()}
        turned = [[set() for _ in range(tw)] for _ in range(th)]
        for r, row in enumerate(layout):
            for c, objs in enumerate(row):
                tr, tc = cell((r, c), hw)
                turned[tr][tc] = {idx[relabel[name_of[o]]]
                                  if name_of[o] in relabel else o for o in objs}
        eng.load_level(turned)
        for i, d in enumerate(presses):
            eng.step(dmap[d])
            got = board(eng)
            want = [[frozenset() for _ in range(tw)] for _ in range(th)]
            for r, row in enumerate(ref[i + 1]):
                for c, objs in enumerate(row):
                    tr, tc = cell((r, c), hw)
                    want[tr][tc] = frozenset(relabel.get(o, o) for o in objs)
            if got != want:
                notes.append(f"rot{k}{'m' if mirror else ''}@press{i}")
                break
    return notes


def _symmetry(walk: int = 400) -> int:
    """Prove ON THE INTERPRETER that the 8-element presentation group is an exact
    symmetry of the mechanic, which is what entitles this game to the
    `PuzzleScriptAdapter._FLIP_GAMES` augmentation.

    This game needs the measurement more than most of the family does. Its two
    load-bearing rules -- the marker stamp and the tail advance -- are each
    written as FOUR per-direction blocks rather than with the relative ``>``
    force, which is exactly the shape that hid ps:gobble_rush's chirality (the
    interpreter drives a rule's four expansions in the fixed order up, down,
    left, right, so a contested cell is settled by the screen). And the markers
    are directional ART, so a transform that did not relabel them would be
    comparing a snake against a different snake.

    Two runs per level, which fail differently:

      * each level's own PLAN, the sequence the corpus actually records;
      * a seeded RANDOM WALK over all five keys, which is what reaches the
        refused presses and the self-collisions a shortest plan never makes.
    """
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        layout = g.levels[level]
        rng = random.Random(f"ouroboros:symmetry:{level}")
        runs = {"plan": board.plan(board.start),
                "walk": [rng.choice(_DIRS) for _ in range(walk)]}   # ACTION too
        notes = []
        for kind, presses in runs.items():
            notes += [f"{kind}:{n}" for n in
                      _replay_transformed(eng, g, layout, presses)]
        bad += len(notes)
        print(f"  L{level}: ({len(runs['plan'])} plan + {walk} random) presses "
              f"x 7 presentations: "
              f"{'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:3])}")
    print("symmetry clean -- the flip augmentation is exact" if not bad
          else f"SYMMETRY FAILED: {bad} chiral presentations")
    return 0 if not bad else 1


#: Every cell composition this game can show. ``head`` is always accompanied by
#: the marker stamped when it moved in (the level layouts ship it that way and no
#: rule removes it) EXCEPT at length 1, where the tail has caught up and replaced
#: that marker with itself; ``apple``/``redapple`` share the body's collision
#: layer, so they never coexist with a marker.
_COMPS = [(), ("wall",), ("u",), ("d",), ("l",), ("r",), ("tail",), ("tail2",),
          ("apple",), ("redapple",),
          ("head", "u"), ("head", "d"), ("head", "l"), ("head", "r"),
          ("head", "tail2")]

#: Pairs the audit REQUIRES to be identical rather than forbidding.
_EXPECTED_IDENTICAL = (
    # tail and tail2 are one object as far as any frame is concerned: [tail2] ->
    # [tail] runs at the top of a turn and the late rules put tail2 back, so a
    # resting board only ever shows tail2. They must render alike or a policy
    # would be shown a distinction that carries nothing.
    {("tail",), ("tail2",)},
    # The head is opaque, so it hides whatever it stands on -- the marker it
    # stamped when it moved in, or (at length 1) the tail that has caught up with
    # it. Neither is a variable a policy is being asked to guess:
    #   * at length >= 3 the marker under the head is never READ, because the only
    #     rule that reads a marker is the tail advance and the tail's successor is
    #     the L-1'th marker, not the head's;
    #   * at length 2 it is read, but its value is FORCED -- the head entered its
    #     cell from the tail's cell, so the marker can only point tail -> head;
    #   * at length 1 head and tail share a cell, which the frame says plainly:
    #     the snake is the one head cell and nothing else.
    # Hiding a determined variable is strictly better than showing it.
    {("head", "u"), ("head", "d"), ("head", "l"), ("head", "r"),
     ("head", "tail2")},
)


def _audit() -> int:
    """Assert every cell COMPOSITION the game can show is distinct in the frame,
    at every cell size the six levels render at, except the pairs
    `_EXPECTED_IDENTICAL` documents -- which are asserted to BE identical.

    The pair this exists for is the four body markers against each other. They
    are the game's entire memory (a marker says which way the snake travelled
    through that cell, and the tail follows them), and all four shipped as
    ``green darkgreen`` -- one ARC index -- so the whole snake was a flat green
    smear and its own future was not in the picture. See the header of
    data/puzzlescript_games/Ouroboros.txt.

    Whole frames are compared, not cell crops: `_render_frame` upscales the board
    to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor)."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    name = lambda t: "+".join(t) if t else "floor"               # noqa: E731
    problems = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for objs in _COMPS:
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[objs] = np.asarray(_render_frame(eng, g)).copy()
        print(f"{h}x{w} board (level{'s' if len(levels) > 1 else ''} "
              f"{', '.join(str(x) for x in levels)}), "
              f"cell {max(1, min(64 // h, 64 // w))}px")
        for objs in _COMPS:
            print(f"  {name(objs):12s} "
                  f"{int((shots[objs] != shots[()]).sum()):4d} px differ from "
                  f"bare floor")
        for a, b in itertools.combinations(_COMPS, 2):
            same = np.array_equal(shots[a], shots[b])
            want = any({a, b} <= grp for grp in _EXPECTED_IDENTICAL)
            if same and not want:
                problems += 1
                print(f"  INDISTINGUISHABLE: {name(a)} == {name(b)}")
            if want and not same:
                problems += 1
                print(f"  SPLIT (expected identical): {name(a)} != {name(b)}")
    print("audit clean" if not problems
          else f"AUDIT FAILED: {problems} problems")
    return 0 if not problems else 1


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
    sys.exit(OuroborosSolver.main())
