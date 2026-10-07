"""Generate Phase-1 training data for the PuzzleScript game ps:polyomino_puzzles
("Polyomino Puzzles" by Zithral).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, the native model of the turn, the CLOSED-FORM distance-to-win
field over the whole (unbounded) state space, and the optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_polyomino_puzzles",
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
A packing puzzle driven by a free-flying CURSOR. Press ACTION on a piece to lift
it, walk to drag it, press ACTION again to set it down; the win is
``all Target on LoSquare`` -- every cell of the grey target region covered. On
every shipped level the pieces' total area is EXACTLY the target's, so a win is
an exact tiling that uses all of them.

Four facts do all the work, and three of them are rules that are easy to read
past:

  * **The cursor is welded to the piece it holds.** Pickup happens where the
    cursor stands (`[action PlayerSeek LoSquare] -> [PlayerDrag RaiseSquare]`),
    and the drag rule forces the cursor and floods that same force through the
    connected HiSquare blob, so cursor and piece translate together. The cursor's
    offset INSIDE the piece therefore never changes while it is held -- which is
    what makes the routing cost below closed-form rather than a search.
  * **Pieces are delimited by EDGES, not by colour.** Every dropped cell is given
    ``EdgeN/S/W/E`` and only the edges strictly inside the dropped piece are
    removed, so a piece is a maximal edge-free connected run of ``LoSquare``.
    Colour is decoration: level 6 ships two separate pink pieces, and a piece
    laid flush against another does NOT merge with it (both keep the boundary
    edges, and the pickup flood tests the SOURCE cell's edge, so it is blocked
    from either side). ``selfcheck`` drives that case rather than trusting it.
  * **Pieces pass THROUGH each other, walls stop them.** ``LoSquare`` and
    ``HiSquare`` are on different collision layers, so a held piece sails over
    the dropped ones freely; what cancels a move is ``late [Square Wall]``. Every
    level is an open rectangle inside a solid wall border (asserted), so a drag
    costs exactly its Manhattan translation and a walk exactly its Manhattan
    distance.
  * **Only the DROP is constrained**, by two rules: ``[HiSquare LoSquare] ->
    cancel`` (no overlapping an already-placed piece) and ``[LowerSquare Target]
    [LowerSquare no Target] -> cancel`` (a piece may not straddle the target's
    border -- it goes fully in or fully out). The second is the one that turns
    this from "shove blocks about" into an exact-cover problem.

Nothing else moves: there is no gravity, no ``again``, no ``random``, no
``restart``, and ACTION on bare floor is cancelled outright.

**Level 1 is UNSOLVABLE as shipped, and it is dropped.** Its 3x3 target must be
tiled by a J-tetromino, a VERTICAL I-tromino and a HORIZONTAL domino, and pieces
cannot rotate. The I-tromino has to take a whole column and the J then takes the
next column plus one cell of a third, which always leaves a vertical pair for a
horizontal domino. `_enumerate_covers` tries every placement of every piece and finds
none, so `plan` returns None at the level start and `discover_solvable` drops the
level from every episode. That is a design bug in the game, not something a
generator may paper over: the levels are the author's and this file does not edit
them (contrast the RENDERING fixes other games in this family carry in their
.txt). The remaining six levels are all solved.

Model + search
--------------
`_Board.step` re-implements one turn natively; ``--selfcheck`` drives ~34k random
presses through BOTH the interpreter and the model across all seven levels and
compares the cursor, the held piece, the full LoSquare PARTITION (i.e. the edge
bookkeeping, not just which cells are filled) and the win flag after every one.

The state is ``(cursor, held piece cells, frozenset of dropped pieces' cells)``.
It is far too big to enumerate -- a single 6-piece level has ~10^11 piece
configurations -- so unlike the enumerated games in this family the field here is
CLOSED FORM (`_Board.distance`):

    d(s) = min over exact covers C, over shape-preserving assignments sigma of
           the pieces to C's placements, over drop ORDERS and over GRAB CELLS of
               sum_i [ walk to the grab cell + 1 pickup + |translation| + 1 drop ]

and, while a piece is being CARRIED, over every square it could be set down on
first (`_Board._carrying`). Covers are enumerated once per level (there are 1 to
4 of them), the assignment is a product of small permutations over equal-shaped
pieces, and the order/grab minimisation is a subset DP -- memoised, so the whole
field costs milliseconds after the first query.

WHAT IS PROVEN, WHAT IS MEASURED, AND WHERE THE LIMIT IS. The value is always
ACHIEVABLE: it is the cost of a route this file can write down, so it is an upper
bound on the true distance by construction, never an under-estimate -- which is
what makes a labelled press SOUND wherever the field is tight (if ``d(s)`` is the
true distance and ``d(s') = d(s) - 1``, then ``s'`` really is one press closer,
because ``d(s') >= true(s')`` forces equality). It is also 1-Lipschitz under
every press within its own strategy class -- a walk press changes one ``|u - q|``
term by one; a drag press moves cursor AND piece together, so the landing cell
``cursor + remaining translation`` is invariant and only ``|remaining
translation|`` changes; a pickup converts ``+2 +|delta|`` into ``+1 +|delta|``;
a drop is the mirror image -- so it always admits a descent, and a descent of an
achievable field is a plan that really wins in exactly that many presses.

What it does NOT model is SHUTTLING: carrying a piece part of the way along a
walk the cursor was making anyway, dropping it, and collecting it later. Because
a drag press moves the cursor too, that leg is free, and paying two extra ACTION
presses for it can buy back more than two presses of drag. That is a
pickup-and-delivery problem, not a closed form, so the field prices each piece as
one continuous carry (`_dp`), plus one relay for the piece in hand.

The consequence is bounded and, more to the point, measured. ``--verify``:

  * compares the closed form against an EXHAUSTIVE BFS over the whole state
    space, on two miniature instances of the same mechanic -- 22k states between
    them, every carry, every relay, every parked piece -- where it agrees
    everywhere it answers (`_ground_truth`);
  * and, on the shipped levels, checks the Lipschitz bound and the existence of a
    descending press separately for the states the recorder LABELS (every state
    of every plan and all five successors of each: clean on all six levels, which
    is the property the labels need) and for a wider seeded random walk (where
    exactly four states, all on level 5, are over-estimated -- by one press each,
    and by a shuttle the expert never takes).

The other modelling assumption is that no drop is ever BLOCKED, which holds
because every piece starts outside the target and every placement lies inside it,
so the ordering constraint in `_dp` is vacuous from the level start (asserted in
`_report`). Off-plan a piece CAN be parked in the target and block one, and
`_route` answers None there rather than guessing -- see its docstring.

Optimal-action targets and recovery
-----------------------------------
`optimal_for` reads the LIVE engine state and returns every press that lowers the
distance -- true tie sets, not a reordering heuristic, and sound at every state
the recorder reaches (see above). They are not a formality in this game: a walk
to the next piece is order-free across the two axes, WHICH piece to fetch next is
frequently tied, and so is which cell of it to grab, so a single-answer label
would train a coin flip the policy cannot win. ``--plans`` prints the measured
average (about 1.4 optimal presses per step).

``epsilon = 0.0``: recovery data comes from the RESET prefix, as everywhere in
this family. The reason to keep the detour dial at zero HERE, despite the field
being exact and cheap, is specific: a detour that presses ACTION at the wrong
moment can PARK a piece inside the target on a square no cover uses, and from
such a state the ordering constraint above stops being vacuous and `distance`
degrades to an upper bound (it still refuses to strand the level -- it returns
None, which `record_level`'s probe rejects -- but a label that is merely an upper
bound is exactly what this file exists to avoid).

Rendering
---------
No change to the game file. ``--audit`` renders every representable cell
composition at both board sizes and asserts they are distinct, with three
documented exemptions, all of them harmless and all of them checked rather than
asserted by hand:

  * ``Single`` is ``transparent`` and the level places it under the corner Wall,
    where it also parks the held piece's colour for the duration of a drag. It
    renders as bare wall. That is bookkeeping the frame is entitled not to show
    (the consequence -- the frame does not reveal what COLOUR the held piece will
    be when it lands -- costs nothing, since a piece is identified by its shape).
  * A piece's colour is a spriteless fill, so ``piece on target`` renders exactly
    like ``piece on floor``. This does NOT hide the goal: the uncovered target
    cells stay grey, so what the frame always shows is precisely the hole that is
    left, and "no grey left" is the win.
  * At ``cell_px = 4`` (the five 14x14 levels) ``PlayerSeek`` and ``PlayerDrag``
    sample to the same 2x2 block, and the cursor cell covers the target under it.
    Neither is reachable ambiguity: ``PlayerDrag`` cannot exist without a
    ``HiSquare`` under it (a pickup needs a LoSquare and the cursor stays inside
    the piece), and ``HiSquare + cursor`` is distinct from ``HiSquare`` alone. The
    fuzz asserts that co-occurrence on every state it visits instead of leaving it
    as an argument.

Augmentation
------------
Per (seed, level) the adapter draws a frame rotation (0..3) and, since this game
is in `PuzzleScriptAdapter._FLIP_GAMES`, an independent horizontal and vertical
flip -- 8 presentations of each of the 6 usable levels, which matters here
because the level count is small. The flips are measured rather than argued:
``--symmetry`` rebuilds every level's LAYOUT under all 8 transforms, reloads it
into the interpreter, and replays the level's own plan plus a 400-press random
walk with the presses transformed; the cursor, the held piece and every dropped
piece must land where the transform says after every one.

Usage (run from the repo root):
    python solvers/generate_polyomino_puzzles_training.py --episodes 200 \
        --out data/training_multi_level/polyomino_puzzles
    python solvers/generate_polyomino_puzzles_training.py --selfcheck
    python solvers/generate_polyomino_puzzles_training.py --plans
    python solvers/generate_polyomino_puzzles_training.py --verify
    python solvers/generate_polyomino_puzzles_training.py --audit
    python solvers/generate_polyomino_puzzles_training.py --symmetry
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import (                     # noqa: E402
    PuzzleScriptAdapter, _render_frame)
from solvers.common.ps_astar import (                           # noqa: E402
    PSAStarSolver, PSExpert, restore, snapshot)

GAME_NAME = "Polyomino_Puzzles"

#: Engine direction -> (dr, dc).
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: The press alphabet. All five matter: the four moves walk the cursor (and drag
#: whatever it holds) and ACTION is both the pickup and the drop.
_ACTIONS = ("up", "down", "left", "right", "action")

#: Sentinel for "this branch cannot finish the level".
_INF = 1 << 30

#: `dict.get` sentinel -- `_Board.distance` caches None (no completable cover)
#: as a real value, so a plain `.get(state)` could not tell it from a miss.
_MISS = object()


def _norm(cells) -> frozenset:
    """``cells`` translated so its bounding box starts at (0, 0) -- a piece's
    SHAPE, which is the only thing that decides where it may be placed (nothing
    in this game rotates or reflects a piece)."""
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _anchor(cells) -> tuple[int, int]:
    """The bounding box's top-left corner. Two equal-shaped cell sets differ by
    exactly the difference of their anchors, which is the drag translation."""
    return (min(r for r, _ in cells), min(c for _, c in cells))


# ---------------------------------------------------------------------------
# The level, natively: one turn, and the closed-form distance field
# ---------------------------------------------------------------------------

class _Board:
    """A level's static geometry, its exact covers, one turn of its mechanic, and
    the closed-form distance-to-win field over the whole state space.

    A state is ``(cursor, held, rest)``:

      * ``cursor`` -- ``(r, c)`` of PlayerSeek/PlayerDrag;
      * ``held``   -- the frozenset of cells of the piece being carried, or None;
      * ``rest``   -- a frozenset of frozensets, one per DROPPED piece.

    Pieces are anonymous. Two pieces of the same shape are interchangeable in
    every respect (they are not distinguishable in the frame either -- only their
    colours differ, and colour is decoration), so the partition alone is the
    state, and `distance` recovers the piece-to-placement matching itself.
    """

    __slots__ = ("h", "w", "walls", "targets", "start", "covers",
                 "_dist_memo", "_dp_memo", "_place_memo")

    def __init__(self, h, w, walls, targets, pieces, cursor):
        self.h, self.w = h, w
        self.walls = frozenset(walls)
        self.targets = frozenset(targets)
        self.start = (cursor, None, frozenset(pieces))

        # The cost model below charges a walk its Manhattan distance and a drag
        # its Manhattan translation. Both are only right on an OPEN rectangle:
        # with an interior wall a walk could need a detour, and a drag could need
        # a non-monotone path (or be impossible). Every shipped level is a bare
        # room inside a solid border, so this is asserted rather than handled.
        for r in range(h):
            for c in range(w):
                edge = r in (0, h - 1) or c in (0, w - 1)
                assert ((r, c) in self.walls) == edge, \
                    "level geometry is not an open room walled at its border"

        assert sum(len(p) for p in pieces) == len(self.targets), \
            "the pieces' area does not equal the target's -- a win would not be " \
            "an exact tiling and an exact tiling would be the wrong question"
        # No piece starts on the target, so a piece's PLACEMENT (always inside the
        # target) can never collide with an unmoved piece: the drop order is free
        # from the level start. `_dp` still checks, for the off-plan states an
        # exploration prefix can reach.
        assert not any(p & self.targets for p in pieces), \
            "a piece starts on the target region"

        self._dist_memo: dict = {}
        self._dp_memo: dict = {}
        self._place_memo: dict = {}
        self.covers = self._enumerate_covers(pieces)

    # -- exact covers ---------------------------------------------------------
    def _placements(self, shape) -> list[frozenset]:
        """Every translation of ``shape`` that lies wholly inside the target."""
        got = self._place_memo.get(shape)
        if got is not None:
            return got
        out = []
        for (tr, tc) in sorted(self.targets):
            cand = frozenset((r + tr, c + tc) for r, c in shape)
            if cand <= self.targets:
                out.append(cand)
        # ``shape`` is normalised, so anchoring it at each target cell in turn
        # generates every placement exactly once (a placement's own anchor is one
        # of those cells).
        got = sorted(set(out), key=sorted)
        self._place_memo[shape] = got
        return got

    def _enumerate_covers(self, pieces) -> list[frozenset]:
        """Every exact tiling of the target by the level's multiset of SHAPES,
        each as a frozenset of placements.

        Anonymous: a cover records WHERE the shapes go, not which piece went
        where -- `_assignments` recovers the matching against the pieces' live
        positions, which is what lets the field answer for a board whose pieces
        have already moved.

        Complete by construction: the search always fills the lowest uncovered
        target cell, so no tiling can be missed and none is generated twice.
        """
        cells = sorted(self.targets)
        counts = Counter(_norm(p) for p in pieces)
        out: set = set()
        acc: list = []

        def rec(covered: frozenset) -> None:
            if len(covered) == len(cells):
                out.add(frozenset(acc))
                return
            cell = next(t for t in cells if t not in covered)
            for shape in counts:
                if counts[shape] == 0:
                    continue
                counts[shape] -= 1
                for pl in self._placements(shape):
                    if cell in pl and not (pl & covered):
                        acc.append(pl)
                        rec(covered | pl)
                        acc.pop()
                counts[shape] += 1

        rec(frozenset())
        return sorted(out, key=lambda c: sorted(sorted(p) for p in c))

    # -- one turn -------------------------------------------------------------
    def won(self, state) -> bool:
        """``all Target on LoSquare``. A HELD piece is a HiSquare and does not
        count, which is why a level cannot be won mid-drag."""
        _cursor, _held, rest = state
        covered = set()
        for p in rest:
            covered |= p
        return self.targets <= covered

    def step(self, state, press: str):
        """One press, natively.

        A move walks the cursor and, if something is held, translates it with the
        cursor; ``late [Square Wall] -> cancel`` makes ANY collision with a wall
        cancel the whole turn, so the cursor does not budge either. ACTION lifts
        the piece under the cursor, or -- while holding one -- drops it, unless
        it would overlap a dropped piece or straddle the target's border.
        """
        cursor, held, rest = state
        if press == "action":
            if held is None:
                for p in rest:
                    if cursor in p:
                        return (cursor, p, rest - {p})
                return state                     # ACTION on bare floor: cancelled
            for p in rest:                       # [HiSquare LoSquare] -> cancel
                if held & p:
                    return state
            inside = len(held & self.targets)    # the straddle rule
            if 0 < inside < len(held):
                return state
            return (cursor, None, rest | {held})

        dr, dc = _DELTA[press]
        nxt = (cursor[0] + dr, cursor[1] + dc)
        if nxt in self.walls or not (0 <= nxt[0] < self.h and 0 <= nxt[1] < self.w):
            return state
        if held is None:
            return (nxt, None, rest)
        moved = frozenset((r + dr, c + dc) for r, c in held)
        for cell in moved:
            if cell in self.walls or not (0 <= cell[0] < self.h
                                          and 0 <= cell[1] < self.w):
                return state
        return (nxt, moved, rest)

    # -- the distance field, in closed form -----------------------------------
    def _assignments(self, pieces, cover):
        """Every shape-preserving bijection from ``pieces`` (a LIST) to
        ``cover``'s placements, as a tuple of placements indexed like ``pieces``.

        Usually there is exactly one; a board with two equal-shaped pieces has 2
        (level 0's pair of vertical dominoes), and they are genuinely different
        plans -- which domino goes where changes every walk in the route.

        Indexed by POSITION rather than keyed by cell set, which is not a style
        choice: a carried piece is a HiSquare and the pieces it flies over are
        LoSquares, different collision layers, so the cursor can park the piece
        it holds exactly on top of a dropped one. Those two pieces are then the
        same frozenset, and a dict keyed by it silently routes both of them to
        one placement -- which made the field under-report by a press on exactly
        those states (caught by `_ground_truth`).
        """
        by_shape: dict = {}
        for pl in cover:
            by_shape.setdefault(_norm(pl), []).append(pl)
        order = sorted(range(len(pieces)),
                       key=lambda i: (sorted(_norm(pieces[i])),
                                      sorted(pieces[i])))
        groups = []
        for shape, group in itertools.groupby(order,
                                              key=lambda i: _norm(pieces[i])):
            group = list(group)
            slots = by_shape.get(shape, ())
            if len(slots) != len(group):
                return                            # this cover is not this board's
            groups.append((group, slots))
        if sum(len(g) for g, _ in groups) != len(cover):
            return
        for choice in itertools.product(*(itertools.permutations(s)
                                          for _, s in groups)):
            out: list = [None] * len(pieces)
            for (group, _), perm in zip(groups, choice):
                for i, pl in zip(group, perm):
                    out[i] = pl
            yield tuple(out)

    def _dp(self, items, cursor) -> int:
        """Fewest presses to place every ``(cells, placement)`` pair in ``items``,
        starting with the cursor at ``cursor`` and holding nothing.

        Each piece costs ``walk to a grab cell + 1 pickup + |translation| +
        1 drop`` and leaves the cursor at the grab cell's image, so the choice of
        grab cell is part of the route, not a detail: grabbing a 5-cell piece by
        its far end can save four presses on the walk that follows.

        The ``pl & c`` test is the drop's overlap rule (`[HiSquare LoSquare] ->
        cancel`): a piece whose destination is still occupied by an unmoved piece
        has to wait for it. Vacuous from any level start (see `__init__`); it
        earns its keep off-plan, where it makes the answer None instead of an
        under-estimate.
        """
        if not items:
            return 0
        key = (items, cursor)
        got = self._dp_memo.get(key)
        if got is not None:
            return got
        best = _INF
        for i, (cells, pl) in enumerate(items):
            others = items[:i] + items[i + 1:]
            if any(pl & c for c, _ in others):
                continue
            ar, ac = _anchor(pl)
            br, bc = _anchor(cells)
            dr, dc = ar - br, ac - bc
            drag = abs(dr) + abs(dc)
            for (qr, qc) in cells:
                sub = self._dp(others, (qr + dr, qc + dc))
                if sub >= _INF:
                    continue
                cost = (abs(cursor[0] - qr) + abs(cursor[1] - qc)
                        + 2 + drag + sub)
                if cost < best:
                    best = cost
        self._dp_memo[key] = best
        return best

    def distance(self, state) -> int | None:
        """Presses to a win from ``state`` -- or None when no cover can be
        completed from here without parking a piece somewhere first.

        With nothing carried, the minimisation is over covers, over the
        piece-to-placement assignment, over the drop order and over the grab
        cells; a piece already sitting on its assigned placement is free. With a
        piece in hand it is `_carrying`, which enumerates where to put it down.

        The value is always ACHIEVABLE -- it is the cost of a route this file can
        write down -- and it is the true minimum everywhere the checks in
        ``--verify`` reach, which includes every state the recorder labels. See
        the module docstring for the one strategy it does not model and for what
        that costs.
        """
        got = self._dist_memo.get(state, _MISS)
        if got is not _MISS:
            return got
        cursor, held, rest = state
        if held is not None:
            best = self._carrying(cursor, held, rest)
        else:
            pieces = sorted(rest, key=sorted)
            best = None
            for cover in self.covers:
                for assign in self._assignments(pieces, cover):
                    cost = self._route(cursor, pieces, assign)
                    if cost is not None and (best is None or cost < best):
                        best = cost
        self._dist_memo[state] = best
        return best

    def _carrying(self, cursor, held, rest) -> int | None:
        """`distance` while a piece is being CARRIED: enumerate everywhere it
        could be set down.

        The obvious closed form -- "fly it straight to its placement" -- is not
        the optimum, and the difference is not a rounding error: picking a piece
        up COMMITS two things, that it is the next piece dealt with and that the
        cursor is stuck at the offset it grabbed. Setting it down again costs one
        press and buys both back, so on the shipped levels there really are
        states where dropping the piece where it stands is two presses better
        than carrying it home (`--verify` catches the discrepancy immediately if
        this is reduced to the direct route).

        Rather than guess which compromise wins, this enumerates every legal drop
        -- each translation of the piece that stays inside the room, misses the
        dropped pieces and does not straddle the target's border -- and lets the
        non-carrying field, which the drop lands in, price each one. The
        recursion is one level deep by construction: the sub-state carries
        nothing.

        Cost is bounded by the room's area, and it amortises: the sub-states
        recur across the queries of a plan, and `_dp`'s memo is keyed on content,
        so the sub-routes that do not involve the dropped piece are shared by all
        of them.
        """
        h, w = self.h, self.w
        occupied: set = set()
        for p in rest:
            occupied |= p
        rows = [r for r, _ in held]
        cols = [c for _, c in held]
        best = None
        # Bound the sweep by the piece's WHOLE bounding box, not just its
        # top-left corner: a translation that only keeps the anchor on the board
        # can still hang the far end off it, and an off-board cell is in neither
        # `self.walls` nor `occupied`, so the filter below would wave it through
        # and the field would price a drop the interpreter refuses.
        for dr in range(-min(rows), h - max(rows)):
            for dc in range(-min(cols), w - max(cols)):
                moved = frozenset((r + dr, c + dc) for r, c in held)
                if any(cell in self.walls or cell in occupied for cell in moved):
                    continue
                inside = len(moved & self.targets)
                if 0 < inside < len(moved):       # the straddle rule
                    continue
                sub = self.distance(((cursor[0] + dr, cursor[1] + dc), None,
                                     rest | {moved}))
                if sub is None:
                    continue
                cost = abs(dr) + abs(dc) + 1 + sub
                if best is None or cost < best:
                    best = cost
        return best

    def _route(self, cursor, pieces, assign) -> int | None:
        """`distance` for ONE assignment, with nothing carried. None when this
        assignment is outside the field's DOMAIN.

        The domain is "no piece is PARKED": every dropped piece either sits
        wholly outside the target -- where it obstructs nothing, since every
        placement is inside it -- or is already exactly on its assigned
        placement. That is the whole trajectory this generator records (pieces
        start outside and are carried straight to a placement), and it is what
        makes the closed form exact rather than merely achievable:

          * it makes `_dp`'s ordering constraint vacuous, since a placement can
            only collide with a piece that touches the target, i.e. one already
            standing on a placement of this same (disjoint) cover;
          * and it rules out the states where a piece would have to be moved
            TWICE, which is the only strategy the formula does not enumerate.

        Off the domain the honest answer is "I do not know", not an upper bound:
        ``--verify`` measures on small instances that a parked piece really can
        make the formula optimistic, and returning None here is what keeps such
        a state from ever being labelled (and what makes `record_level`'s detour
        probe refuse to enter one).
        """
        for p, goal in zip(pieces, assign):
            if (p & self.targets) and goal != p:
                return None
        todo = tuple(sorted(((p, goal) for p, goal in zip(pieces, assign)
                             if p != goal), key=lambda it: sorted(it[0])))
        sub = self._dp(todo, cursor)
        return None if sub >= _INF else sub

    # -- reading the field ----------------------------------------------------
    def optimal(self, state) -> list:
        """Every press that leaves the game as close to the win as any press can.

        The ARGMIN over the successors' distances rather than "the presses that
        subtract exactly one". The two coincide wherever the field is tight,
        which is every state the recorder labels (``--verify`` is what says so),
        and they are the same tie sets -- and where the field is not tight, an
        argmin is still exactly the right answer while "subtract one" would
        return nothing at all and leave a step unlabelled.

        The ties are real, not a reordering artefact: which piece to fetch next,
        which of its cells to grab it by, and how to interleave the two axes of
        the walk to it are all genuinely free at most steps.
        """
        here = self.distance(state)
        if not here:                              # won (0) or unreachable (None)
            return []
        best, out = None, []
        for press in _ACTIONS:
            there = self.distance(self.step(state, press))
            if there is None:
                continue
            if best is None or there < best:
                best, out = there, [press]
            elif there == best:
                out.append(press)
        return out if best is not None and best < here else []

    def plan(self, state) -> list | None:
        """A press sequence from ``state`` to a win, or None.

        A descent of the field, taking the first tied press in `_ACTIONS` order,
        so the plan is a pure function of the board. It terminates because
        `optimal` only returns presses that strictly lower the distance, and it
        is shortest wherever the field is tight -- which every state it walks
        through is (see the module docstring, and `_report`, which presses the
        whole thing into the interpreter and requires a WIN).
        """
        if self.distance(state) is None:
            return None
        out = []
        while self.distance(state):
            best = self.optimal(state)
            if not best:                                     # pragma: no cover
                raise AssertionError("the distance field has no descent")
            state = self.step(state, best[0])
            out.append(best[0])
        return out


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class PolyominoExpert(PSExpert):
    """Exact planner over the closed-form field (see `_Board`).

    `PSExpert` supplies the plan memo, the restore discipline and the level
    scoping; `_search` swaps in the field descent, so the interpreter is only
    ever stepped by the recorder -- which is also what verifies every plan, since
    a level is kept only when the engine reports WIN.
    """

    directions = list(_ACTIONS)

    #: `_key` is the native state, canonical only WITHIN a level (the walls and
    #: the target that complete it are static per level but differ between them).
    scope_by_level = True

    def setup(self) -> None:
        g = self.g
        idx = g.obj_name_to_idx
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.target_ids = set(g.resolve_object_name("target"))
        self.lo_id = idx["losquare"]
        self.hi_id = idx["hisquare"]
        # Edge N/S/W/E as (direction delta, object id): the pickup flood crosses a
        # cell boundary only where the SOURCE cell carries no edge, so these are
        # what partition the LoSquares into pieces.
        self.edges = ((-1, 0, idx["edgen"]), (1, 0, idx["edges"]),
                      (0, -1, idx["edgew"]), (0, 1, idx["edgee"]))
        self.player_ids = set(self.game._engine._player_indices)
        self._boards: dict[int | None, _Board] = {}
        self._cur: _Board | None = None

    # -- reading the engine ---------------------------------------------------
    def read(self, eng):
        """The native ``(cursor, held, rest)`` triple from the engine grid.

        The partition is read from the EDGES rather than from colour or from
        4-connectivity: two pieces laid flush are separate (both keep their
        boundary edges) and one piece can span two colours, so anything else
        would be a different game's state.
        """
        grid = eng.grid
        cursor = None
        held = set()
        lo = set()
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell & self.player_ids:
                    cursor = (r, c)
                if self.hi_id in cell:
                    held.add((r, c))
                if self.lo_id in cell:
                    lo.add((r, c))
        if cursor is None:                                   # pragma: no cover
            raise AssertionError("the board has no cursor")

        rest, seen = [], set()
        for cell in sorted(lo):
            if cell in seen:
                continue
            comp, stack = set(), [cell]
            seen.add(cell)
            while stack:
                (r, c) = stack.pop()
                comp.add((r, c))
                for dr, dc, eid in self.edges:
                    if eid in grid[r][c]:
                        continue
                    nxt = (r + dr, c + dc)
                    if nxt in lo and nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            rest.append(frozenset(comp))
        return (cursor, frozenset(held) if held else None, frozenset(rest))

    def build(self, eng) -> _Board:
        """A `_Board` for whatever level the engine is holding, from its START
        state -- the only state whose piece partition is the level's own."""
        h, w = len(eng.grid), len(eng.grid[0])
        walls, targets = set(), set()
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & self.wall_ids:
                    walls.add((r, c))
                if cell & self.target_ids:
                    targets.add((r, c))
        cursor, held, rest = self.read(eng)
        assert held is None, "build() wants a level's START state"
        return _Board(h, w, walls, targets, rest, cursor)

    def board(self, eng, level: int | None) -> _Board:
        """The level's `_Board`, built (and its covers enumerated) once, cached.

        Callers must not ask for a board while the engine is mid-solve on a level
        it has not seen before -- `build` reads the piece partition, which is only
        the level's own at its start. `PSAStarSolver` calls `prepare_expert`
        first, which walks every level cold.
        """
        got = self._boards.get(level)
        if got is None:
            got = self._boards[level] = self.build(eng)
        return got

    # -- planning -------------------------------------------------------------
    def _key(self, eng):
        return self.read(eng)

    def plan(self, eng, level: int | None = None) -> list | None:
        self._cur = self.board(eng, level)
        return super().plan(eng, level)

    def heuristic(self, eng) -> int:
        """The EXACT remaining press count (the base `_astar` is never used here,
        but the contract is that this is 0 at a win and admissible, and the field
        is both)."""
        d = self._cur.distance(self.read(eng))
        return 0 if d is None else d

    def _search(self, eng) -> list | None:
        return self._cur.plan(self.read(eng))

    def optimal_dirs(self, eng, level: int | None) -> list:
        """Every press on a shortest route, from the engine's CURRENT state."""
        return self.board(eng, level).optimal(self.read(eng))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class PolyominoSolver(PSAStarSolver):
    game_id = "puzzlescript_polyomino_puzzles"
    game_name = GAME_NAME
    expert_cls = PolyominoExpert

    #: Unused -- `PolyominoExpert._search` never calls the base A* -- but left at
    #: the family default so a future subclass that does is not silently starved.
    node_cap = 2_000_000
    weight = 1
    #: The adapter GAME_OVERs at 200 actions per level and the longest plan here
    #: is well inside that; the exploration prefix runs BEFORE the RESET, which
    #: zeroes the counter, so it does not eat into this.
    max_steps = 200

    #: No epsilon detours -- see the module docstring. A wrong ACTION can park a
    #: piece inside the target on a square no cover uses, and from there the
    #: closed-form field is only an upper bound.
    epsilon = 0.0

    def prepare_expert(self, game, expert) -> None:
        """Build every level's `_Board` up front, from its cold start state.

        Not just front-loading: `board` can only be built from a level START (it
        reads the piece partition), so this is what guarantees every later query
        -- including one made mid-solve -- finds a cached board.
        """
        for level in range(game.n_levels):
            game.set_level(level)
            expert.board(game._engine, level)

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """Every press that keeps the game on a SHORTEST route to the win, read
        off the LIVE engine state.

        `record_level` asks for this before executing the press, so the engine is
        still at the state being labelled. Falls back to the press about to be
        taken if the field has nothing to say -- no expert step may ship
        unlabelled.
        """
        best = expert.optimal_dirs(expert.game._engine, level)
        if best:
            return best
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _levels():
    """``(solver, game, expert)`` with every level's board built."""
    solver = PolyominoSolver()
    game, expert, _solvable = solver._ensure(0)
    return solver, game, expert


def selfcheck(trials: int = 40, steps: int = 120, verbose: bool = True) -> int:
    """Drive random presses through BOTH the interpreter and `_Board.step`,
    comparing the cursor, the held piece, the whole LoSquare PARTITION and the
    win flag after every one. This is the guard that lets the planner trust the
    native model.

    Comparing the partition and not merely the filled cells is the point: the
    model's central claim is that a piece dropped flush against another stays a
    separate piece (the edge bookkeeping), and a random walk drops pieces next to
    each other constantly. If the interpreter merged them, only the partition
    would say so.

    It also asserts the invariant the rendering audit leans on -- ``PlayerDrag``
    never exists without a ``HiSquare`` under the cursor -- on every state it
    visits.

    The walks run from a COLD level start, with no warm-up press, because
    turn-one bookkeeping is precisely the class of mechanic a warm-up hides.

    Returns the number of mismatches.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    expert = PolyominoExpert(game)
    idx = game._game.obj_name_to_idx
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(game._engine, level)
        rng = random.Random(f"polyomino:selfcheck:{level}")
        bad = presses = drags = drops = 0
        for _ in range(trials):
            game.set_level(level)
            eng = game._engine
            state = expert.read(eng)
            for _ in range(steps):
                press = rng.choice(_ACTIONS)
                want = board.step(state, press)
                eng.step(press)
                presses += 1
                got = expert.read(eng)
                if want != got or board.won(want) != eng.check_win():
                    bad += 1
                    break
                state = want
                if state[1] is not None:
                    drags += 1
                    # The audit's exemption, measured: a dragging cursor always
                    # stands on a cell of the piece it is dragging.
                    holding = any(idx["playerdrag"] in cell for row in eng.grid
                                  for cell in row)
                    if not holding or state[0] not in state[1]:
                        bad += 1
                        break
                elif any(idx["playerdrag"] in cell for row in eng.grid
                         for cell in row):
                    bad += 1
                    break
                else:
                    drops += 1
                if eng.check_win():
                    break
        total += bad
        if verbose:
            print(f"  L{level}: {'OK' if not bad else f'{bad} MISMATCHES'} "
                  f"({presses} presses vs the interpreter, "
                  f"{drags} carried / {drops} free)")
    return total


def _walk(board: _Board, plan) -> list:
    """The states ``plan`` passes through, starting at the level start."""
    out, state = [], board.start
    for press in plan:
        out.append(state)
        state = board.step(state, press)
    return out


def _report() -> int:
    """Print each level's covers, its shortest plan and the engine's verdict --
    the quick "is this game still fully solved" check."""
    _solver, game, expert = _levels()
    eng = game._engine
    bad = unsolvable = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        pieces = sorted(len(p) for p in board.start[2])
        head = (f"  L{level}: {eng.height}x{eng.width}  "
                f"{len(pieces)} pieces {pieces}  target {len(board.targets)}  "
                f"{len(board.covers)} cover(s)")
        # The claim `_dp`'s ordering constraint rests on, stated per level: no
        # placement of any cover ever lands on a piece where it starts, so the
        # drop order is completely free and no piece is ever worth moving twice.
        free = all(not (pl & p) for cover in board.covers for pl in cover
                   for p in board.start[2])
        plan = board.plan(board.start)
        if plan is None:
            unsolvable += 1
            print(f"{head}  UNSOLVABLE (no exact tiling exists)")
            continue
        for press in plan:
            eng.step(press)
        won = eng.check_win()                   # read BEFORE anything reloads
        bad += not won
        ties = sum(len(board.optimal(s)) for s in _walk(board, plan))
        print(f"{head}  {len(plan):3d} presses  win={won}  "
              f"order-free={free}  {ties / len(plan):.2f} optimal presses/step")
    print(f"  {game.n_levels - unsolvable}/{game.n_levels} levels solved "
          f"({unsolvable} unsolvable as shipped -- see the module docstring)")
    return 1 if bad else 0


#: Miniature boards, in the shipped levels' own ASCII: ``#`` wall, ``x`` the
#: wall the Single hides under, ``t`` target, ``p`` cursor, ``.`` floor, a letter
#: a piece cell of that colour. They exist for `_ground_truth`, which needs
#: instances whose ENTIRE state space fits in memory -- the shipped ones do not
#: come close (level 5 has ~10^11 piece configurations).
#:
#: They are chosen for the two things the shipped levels cannot exhibit:
#:
#:   * ``park`` -- a 1x3 target taken by a domino and a monomino. The two covers
#:     put the monomino at the left end or the right end, so the MIDDLE cell is a
#:     legal drop inside the target that no cover uses. That is the parked state
#:     the field abstains on, and the BFS says what it really costs.
#:   * ``swap`` -- a 2x2 target taken by a vertical domino and two monominoes,
#:     which has four covers, several assignments each, and enough room for a
#:     piece to be dropped in the target in front of another one's placement --
#:     i.e. for the drop-order constraint to actually bite.
_MINI = {
    "park": ("x######",
             "#.ttt.#",
             "#aa.b.#",
             "#..p..#",
             "#######"),
    "swap": ("x#####",
             "#.tt.#",
             "#.tt.#",
             "#a.b.#",
             "#a.pc#",
             "######"),
}


def _mini_layout(idx, rows) -> list:
    """An interpreter level grid (``list[list[set[int]]]``) from a `_MINI` spec."""
    named = {"#": ("wall",), "x": ("wall", "single"), "t": ("target",),
             "p": ("playerseek",), ".": ()}
    return [[{idx["background"]} | {idx[o] for o in named[ch]}
             if ch in named else {idx["background"], idx[ch]}
             for ch in line]
            for line in rows]


def _exhaustive(board: _Board):
    """The COMPLETE reachable state space of a (small) board and its exact
    distance-to-win field: one forward BFS collecting every state and its five
    successors, then one backward BFS from the wins.

    This is the ground truth `distance`'s closed form is checked against. It
    knows nothing about covers, orders or grab cells -- it just presses buttons
    -- so it prices the strategies the formula does not enumerate (parking a
    piece, carrying one twice, grabbing it by a different cell the second time)
    at whatever they actually cost.
    """
    states = [board.start]
    index = {board.start: 0}
    succ: list = []
    i = 0
    while i < len(states):
        state = states[i]
        if board.won(state):
            succ.append(None)                    # terminal: the level ends
        else:
            row = []
            for press in _ACTIONS:
                nxt = board.step(state, press)
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
        if board.won(state):
            dist[j] = 0
            queue.append(j)
    while queue:
        j = queue.popleft()
        for src in pred[j]:
            if dist[src] < 0:
                dist[src] = dist[j] + 1
                queue.append(src)
    return states, index, dist


def _ground_truth(fuzz: int = 3000) -> int:
    """Check the closed-form field against an EXHAUSTIVE search, on instances
    small enough to have one.

    This is the pass that earns the module docstring's "exact". Every other check
    here is internal to the formula (its plans win, its labels descend); this one
    puts it next to a distance field derived from nothing but the button presses,
    over the whole reachable space, and requires:

      * **agreement wherever the field answers.** ``distance(s) == BFS(s)`` for
        every state the field does not abstain on. This is what would fail if the
        one-move-per-piece enumeration were missing a cheaper strategy.
      * **the level start is answered**, and so is every state on the descent
        from it -- an "exact" field that abstained on the states that matter
        would pass the first requirement vacuously.
      * **the abstentions are real.** At least one state must exist where the
        field abstains AND the BFS says the level is still winnable; that is the
        parked piece the domain rule exists for. If that set were empty the rule
        would be dead weight and should go.

    Each instance is also fuzzed against the interpreter first, exactly as
    `selfcheck` does the shipped levels: the ground truth is only as good as the
    model it enumerates, and these boards exercise shapes the shipped levels do
    not (a 1x1 piece, a target one cell tall).
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    expert = PolyominoExpert(game)
    eng = game._engine
    idx = game._game.obj_name_to_idx
    bad = 0
    for name, rows in _MINI.items():
        layout = _mini_layout(idx, rows)
        eng.load_level(layout)
        board = expert.build(eng)

        # -- the model, against the interpreter, on THIS board -----------------
        rng = random.Random(f"polyomino:mini:{name}")
        mism = 0
        for _ in range(fuzz // 60):
            eng.load_level(layout)
            state = expert.read(eng)
            for _ in range(60):
                press = rng.choice(_ACTIONS)
                want = board.step(state, press)
                eng.step(press)
                if want != expert.read(eng) or board.won(want) != eng.check_win():
                    mism += 1
                    break
                state = want
                if eng.check_win():
                    break
        bad += mism

        # -- the closed form, against the whole space --------------------------
        states, _index, truth = _exhaustive(board)
        answered = wrong = abstain_live = 0
        for state, real in zip(states, truth):
            got = board.distance(state)
            if got is None:
                abstain_live += real >= 0
                continue
            answered += 1
            if got != real:
                wrong += 1
        bad += wrong
        if board.distance(board.start) is None:
            bad += 1
            print(f"  {name}: the field abstains on the level START")
        elif board.plan(board.start) is None:
            bad += 1
            print(f"  {name}: no plan from the level start")
        if not abstain_live:
            bad += 1
            print(f"  {name}: nothing is parked anywhere -- the domain rule in "
                  f"_route is dead weight on this instance")
        live = sum(1 for d in truth if d >= 0)
        note = "" if not mism else f" -- {mism} MISMATCHES"
        print(f"  {name}: {len(states):6d} states ({live} winnable), "
              f"{fuzz} presses vs the interpreter{note}, "
              f"field answers {answered} of them ({wrong} wrong), "
              f"abstains on {abstain_live} winnable (parked) states, "
              f"d*={truth[0]}")
    return bad


def _verify(walks: int = 12, steps: int = 140) -> int:
    """Double-entry check of the closed-form field and of every tie set it ships.

    `_Board.distance` is a formula, not a swept table, so "the plan wins" is a
    weak check of it: a formula that over-estimates by a constant would still
    produce a winning descent while every tie set it labels would be wrong. Four
    passes, which fail differently:

      * **Ground truth** (`_ground_truth`): the closed form against an exhaustive
        BFS over the WHOLE state space, on instances small enough to have one.
        The other three passes are all internal to the formula; this is the one
        that compares it with the game.
      * **Lipschitz + descent.** At every sampled state: no press may lower the
        distance by more than one, and at least one press must lower it by
        exactly one. Together with ``d == 0`` exactly at the wins and the field
        being achievable by construction, that is the statement that ``d`` is the
        true distance THERE -- so it is what makes a label sound.

        The sample is split, because the answer differs and the difference is the
        point. LABELLED states (every state of every plan, and all five
        successors of each -- the recorder asks for a tie set at the first and
        compares the second) must be clean, and are. The wider seeded RANDOM WALK
        may over-estimate by one press, and does, on four states of level 5: the
        walks wander into positions where SHUTTLING a piece pays, which the
        closed form does not model (see the module docstring). Those states are
        counted and printed rather than ignored, and a gap of more than one press
        -- which would mean something other than the known limit -- fails.
      * **Executable labels.** Walk each level's plan on the INTERPRETER and, at
        every step, actually press each direction the label calls optimal: the
        board the engine lands on must be the native successor and its field
        distance must be exactly one less.
      * **Cover soundness.** Every cover is disjoint, covers the target exactly,
        and its shape multiset is the level's -- so a bug in the enumeration
        cannot quietly ship a "win" the game would not accept.
    """
    bad = _ground_truth()
    _solver, game, expert = _levels()
    eng = game._engine
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        shapes = Counter(_norm(p) for p in board.start[2])

        # -- pass 3: the covers themselves ------------------------------------
        for cover in board.covers:
            cells: set = set()
            for pl in cover:
                if pl & cells:
                    bad += 1
                    print(f"  L{level}: a cover overlaps itself")
                cells |= pl
            if cells != set(board.targets):
                bad += 1
                print(f"  L{level}: a cover does not tile the target")
            if Counter(_norm(pl) for pl in cover) != shapes:
                bad += 1
                print(f"  L{level}: a cover uses shapes the level does not have")

        plan = board.plan(board.start)
        if plan is None:
            print(f"  L{level}: unsolvable as shipped -- nothing to verify")
            continue

        # -- pass 1: Lipschitz + descent over a large sample -------------------
        labelled: set = set(_walk(board, plan)) | {board.start}
        sample: set = set(labelled)
        for state in labelled:
            for press in _ACTIONS:
                sample.add(board.step(state, press))
        graded = set(sample)                      # everything reached from a plan
        rng = random.Random(f"polyomino:verify:{level}")
        for _ in range(walks):
            state = board.start
            for _ in range(steps):
                state = board.step(state, rng.choice(_ACTIONS))
                sample.add(state)
                if board.won(state):
                    break
        live = shuttle = worst = 0
        for state in sample:
            here = board.distance(state)
            if (here == 0) != board.won(state):
                bad += 1
                print(f"  L{level}: distance 0 is not exactly the winning states")
                break
            if here is None:
                continue
            live += 1
            nxt = [board.distance(board.step(state, p)) for p in _ACTIONS]
            gap = max((here - 1 - d for d in nxt if d is not None), default=0)
            if gap > 0:
                shuttle += 1
                worst = max(worst, gap)
                if state in graded:
                    bad += 1
                    print(f"  L{level}: a press lowers the distance by more than "
                          f"one at a state the recorder LABELS")
                    break
            if here and not any(d < here for d in nxt if d is not None):
                bad += 1
                print(f"  L{level}: no press descends from a live state")
                break
        if worst > 1:
            bad += 1
            print(f"  L{level}: an off-plan state is over-estimated by {worst} "
                  f"presses -- more than the known shuttling limit of one")

        # -- pass 2: every shipped label, pressed on the interpreter -----------
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
                labels += 1
                landed = expert.read(eng)
                if landed != board.step(state, alt) \
                        or board.distance(landed) != here - 1:
                    bad += 1
                    print(f"  L{level}: step {i} labels {alt} optimal, but the "
                          f"interpreter does not land one press closer")
                restore(eng, before)
            eng.step(press)
            state = board.step(state, press)
        if not eng.check_win():
            bad += 1
            print(f"  L{level}: the interpreter does not report a win")
        print(f"  L{level}: {len(sample):6d} states sampled "
              f"({live} in the field's domain, {len(sample) - live} parked by "
              f"the random walks and abstained on), {len(graded)} of them "
              f"labelled or one press off a label, {shuttle} off-plan "
              f"over-estimate(s) by <={worst}, "
              f"{len(plan):3d} plan steps, {labels} labels pressed on the "
              f"interpreter -- {'OK' if not bad else 'see above'}")
    print(f"verify: {bad} problems")
    return 1 if bad else 0


def _transform(k: int, mirror: bool):
    """``(cell_fn, dims_fn, direction_map)`` for one element of the 8-group:
    mirror left-right first, then turn clockwise ``k`` times.

    The direction map is DERIVED from the same linear part that moves the cells,
    so the two cannot drift apart."""
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

    dmap = {"action": "action"}
    for name, (dr, dc) in _DELTA.items():
        if mirror:
            dc = -dc
        for _ in range(k):
            dr, dc = dc, -dr
        dmap[name] = next(n for n, d in _DELTA.items() if d == (dr, dc))
    return cell, dims, dmap


def _replay_transformed(eng, expert, layout, presses) -> list[str]:
    """Replay ``presses`` on all seven non-identity turned/mirrored copies of
    ``layout`` and return a note per presentation that diverged.

    The WHOLE state is compared -- cursor, held piece and every dropped piece --
    not just the cursor: a mirror that split a piece differently, or that let a
    drop through the interpreter would have refused, would leave the cursor
    exactly where the transform says.

    The transformed board is built from the LEVEL LAYOUT and reloaded, so the
    interpreter parses and resolves everything itself rather than being handed a
    grid this file transformed after the fact.
    """
    hw = (len(layout), len(layout[0]))

    def move(state, cellf, tw):
        cursor, held, rest = state
        return (cellf(cursor, hw),
                None if held is None
                else frozenset(cellf(x, hw) for x in held),
                frozenset(frozenset(cellf(x, hw) for x in p) for p in rest))

    eng.load_level(layout)
    ref = []
    for press in presses:
        eng.step(press)
        ref.append(expert.read(eng))

    notes = []
    for k, mirror in itertools.product(range(4), (False, True)):
        if (k, mirror) == (0, False):
            continue
        cellf, dims, dmap = _transform(k, mirror)
        th, tw = dims(hw)
        turned = [[set() for _ in range(tw)] for _ in range(th)]
        for r, row in enumerate(layout):
            for c, objs in enumerate(row):
                tr, tc = cellf((r, c), hw)
                turned[tr][tc] = set(objs)
        eng.load_level(turned)
        for i, press in enumerate(presses):
            eng.step(dmap[press])
            if expert.read(eng) != move(ref[i], cellf, tw):
                notes.append(f"rot{k}{'m' if mirror else ''}@press{i}")
                break
    return notes


def _symmetry(walk: int = 400) -> int:
    """Prove ON THE INTERPRETER that the 8-element presentation group is an exact
    symmetry of the mechanic, which is what entitles this game to the
    `PuzzleScriptAdapter._FLIP_GAMES` augmentation.

    The argument is strong -- no gravity, no chiral sprite, a cursor whose input
    is screen-relative, a win condition that names no direction, and every rule
    either stated with the relative ``>`` force or shipped as a symmetric pair
    (the pickup flood and the edge merge are each given for ``right`` and
    ``down``, which covers both orientations of each axis) -- but the check is
    nearly free, so it is measured anyway, twice over:

      * each level's own PLAN, which is the sequence the corpus actually records;
      * a seeded RANDOM WALK per level, which is what actually exercises the
        chirality candidates -- refused drops, drags into a wall, and pieces laid
        flush against one another.
    """
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        layout = g.levels[level]
        rng = random.Random(f"polyomino:symmetry:{level}")
        runs = {"walk": [rng.choice(_ACTIONS) for _ in range(walk)]}
        plan = board.plan(board.start)
        if plan is not None:
            runs["plan"] = plan
        notes = []
        for kind, presses in runs.items():
            notes += [f"{kind}:{n}" for n in
                      _replay_transformed(eng, expert, layout, presses)]
        bad += len(notes)
        print(f"  L{level}: ({len(runs.get('plan', []))} plan + {walk} random) "
              f"presses x 7 presentations: "
              f"{'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:3])}")
    print("symmetry clean -- the flip augmentation is exact" if not bad
          else f"SYMMETRY FAILED: {bad} chiral presentations")
    return 0 if not bad else 1


#: Cell compositions that render identically ON PURPOSE. Each entry is a witness
#: pair plus the reason it costs the agent nothing; `_audit` requires every OTHER
#: pair to be distinguishable, and requires each witness here to actually still
#: be identical (a stale exemption is as bad as a missing one). `_excused` below
#: turns each into the structural family it stands for.
_AUDIT_EXEMPT = [
    (("wall",), ("wall", "single", "a"),
     "the corner Wall is opaque and hides the Single under it -- and the colour "
     "a drag parks THERE, so the frame does not say what colour the held piece "
     "will be (it does not need to: a piece is identified by its shape)"),
    (("a", "losquare"), ("a", "losquare", "target"),
     "a piece's colour is a spriteless fill, so it hides the Target it covers "
     "-- what the frame must show is the target still UNCOVERED, and that is "
     "exactly what stays grey"),
    (("hisquare", "playerdrag"), ("target", "hisquare", "playerdrag"),
     "the cursor block fills what is left of the cell it drags from"),
]


def _excused(a, b) -> bool:
    """Is this pair of compositions one of the documented families?"""
    sa, sb = set(a), set(b)
    diff = sa ^ sb
    if "wall" in sa & sb and diff <= {"single", *"abcdefg"}:
        return True                     # everything below an opaque Wall
    if diff == {"target"} and (sa & sb) & {"losquare", "playerseek",
                                           "playerdrag"}:
        return True                     # a fill covers the Target beneath it
    return False


def _audit() -> int:
    """Assert every cell COMPOSITION the game can show is distinct in the frame,
    at both board sizes, modulo the documented exemptions.

    Whole frames are compared, not cell crops: `_render_frame` upscales the board
    to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window.

    The pairs this exists for are the EDGE subsets. Edges are the only thing in
    the frame that says where one piece ends and the next begins -- two pieces
    laid flush are one solid colour otherwise -- and they are single-pixel lines
    on the outermost row/column of a 5x5 sprite, exactly what a 4px cell is prone
    to drop. (It does not: `_render_cell_sprite` samples centred, so rows 0 and 4
    both survive at ``cell_px = 4``. Measured here rather than believed.)
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    # The reachable compositions. A piece cell is `colour + LoSquare + an edge
    # subset`, optionally on the target and optionally under the cursor; a held
    # cell is `HiSquare`, optionally on the target and optionally under the
    # dragging cursor. `PlayerSeek` cannot share a cell with a `HiSquare` (the
    # HiSquare only exists while the cursor is a PlayerDrag), so that pair is not
    # listed -- and `selfcheck` asserts the co-occurrence rather than this file
    # assuming it.
    edge_sets = [tuple(e for e, keep in zip(("edgen", "edges", "edgew", "edgee"),
                                            bits) if keep)
                 for bits in itertools.product((0, 1), repeat=4)]
    comps = [(), ("wall",), ("wall", "single"), ("wall", "single", "a"),
             ("target",), ("playerseek",), ("target", "playerseek"),
             ("hisquare",), ("target", "hisquare"),
             ("hisquare", "playerdrag"), ("target", "hisquare", "playerdrag")]
    for colour in ("a", "b", "c", "d", "e", "f", "g"):
        comps.append((colour, "losquare"))
        comps.append((colour, "losquare", "target"))
        comps.append((colour, "losquare", "playerseek"))
    for edges in edge_sets:                       # one colour is enough for these
        comps.append(("a", "losquare") + edges)
    comps = list(dict.fromkeys(comps))             # the empty edge set repeats one

    name = lambda t: "+".join(t) if t else "floor"                # noqa: E731
    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    clashes_total = 0
    #: A witness only has to collide at SOME board size -- the target shows
    #: through the cursor at cell_px 5 and not at 4 -- but an exemption that is
    #: never needed anywhere is dead weight and is reported.
    needed = {i: False for i in range(len(_AUDIT_EXEMPT))}
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for objs in comps:
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[objs] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(comps, 2)
                   if np.array_equal(shots[a], shots[b]) and not _excused(a, b)]
        for i, (a, b, _why) in enumerate(_AUDIT_EXEMPT):
            needed[i] |= bool(np.array_equal(shots[a], shots[b]))
        clashes_total += len(clashes)
        print(f"{h}x{w} board (level{'s' if len(levels) > 1 else ''} "
              f"{', '.join(str(x) for x in levels)}), "
              f"cell {max(1, min(64 // h, 64 // w))}px")
        distinct = len({shots[o].tobytes() for o in comps})
        edges_seen = {shots[("a", "losquare") + e].tobytes() for e in edge_sets}
        edge_ok = "distinct" if len(edges_seen) == len(edge_sets) else "NOT distinct"
        print(f"  {len(comps)} representable compositions -> "
              f"{distinct} distinct frames "
              f"({len(edge_sets)} edge subsets, all {edge_ok})")
        for a, b in clashes:
            print(f"  IDENTICAL: {name(a)} == {name(b)}")
    for i, (a, b, why) in enumerate(_AUDIT_EXEMPT):
        if not needed[i]:
            clashes_total += 1
            print(f"  STALE EXEMPTION: {name(a)} != {name(b)} at any board size")
            continue
        print(f"  exempt: {name(a)} == {name(b)} -- {why}")
    print("audit clean" if not clashes_total
          else f"AUDIT FAILED: {clashes_total} indistinguishable pairs")
    return 0 if not clashes_total else 1


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
    sys.exit(PolyominoSolver.main())
