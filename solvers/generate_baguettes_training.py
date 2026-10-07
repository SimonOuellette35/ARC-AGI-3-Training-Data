"""Generate Phase-1 training data for the PuzzleScript game ps:baguettes
("Baguettes", by bregehr).

The BaseSolver plumbing, the rotation contract and the trajectory recorder live
in `solvers/common/ps_astar.py`, shared with the other ps: generators. This file
is the game-specific part: a native model of the mechanics, the A* that plans on
it, and the differential test that keeps the model honest.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_baguettes",
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
exactly.

The game
--------
Sokoban with RIGID MULTI-CELL pieces, plus a baking sub-mechanic.

*Pieces.* A baguette is a straight run of cells with typed ends: a HORIZONTAL one
is ``bagleft (bagmid)* bagright`` and a VERTICAL one is ``bagup (bagmidup)*
bagdown``; ``bagball`` is the degenerate single cell. The ``startloop`` rules
propagate ``moving`` along the piece's own axis (``horizontal [moving bagmid |
bagside] -> [moving bagmid | moving bagside]`` and the vertical mirror), and
``moving`` is direction-agnostic, so a piece is dragged as ONE RIGID BODY whether
it is pushed along its axis or broadside. ``[> bagget | wall] -> cancel`` then
undoes the *whole turn* the moment any cell of a piece would enter a wall, so a
piece is never sheared apart or partially moved: a push either translates the
entire piece by one cell or does nothing at all. Baguettes also shove each other
(``[> Bagget | Bagget]``), and a chain cancels together.

*The goal is shape-matched placement.* The win conditions pair each target class
with its own piece class (``All targetleft on bagleft``, ... , ``All targetball
on bagball``), so a run of targets is solved only by a piece of the SAME
orientation and the SAME length, translated onto it.

*Dough and ovens.* Raw ``doughball`` is rigid too, and in a stronger sense:
``[moving doughball | doughball] -> [moving doughball | moving doughball]`` has
no direction prefix, so an orthogonally-connected BLOB of dough moves as one body
in all four directions -- level 7's eight dough cells form a ring with the player
inside it. Dough sitting on an ``oven`` bakes at the end of the turn it lands,
into the shape of the dough run that is on ovens at that instant: a horizontal
run of two or more becomes ``bagleft ... bagright``, a vertical run becomes
``bagup ... bagdown``, and whatever the shape rules did not claim becomes a
``bagball`` (``late [doughball oven] -> [bagball oven]``). Three consequences drive
the puzzles. A blob can only be given a shape it can be *lined up* in. Baking is
the only way to BREAK a blob apart -- shove a two-dough domino half onto the oven
and its leading cell bakes to a ball while the trailing cell is left behind as
free, now separate dough. And blobs are how a dough cell escapes a wall: level
6's third dough sits against the left wall, so no player can ever stand to its
left and NOTHING can push it right on its own, but walk another dough up beside
it and the pair is one rigid body that a push from the other cell's free side
carries right, wall-pinned cell and all. (That is not a subtlety a hand proof
survives -- the first version of this file skipped level 6 as unwinnable on
exactly the reasoning that misses it.)

Why this plans on a model
-------------------------
One ``eng.step`` costs ~5 ms here: the ``startloop`` propagation plus thirteen
``late`` bake rules are re-scanned every turn, and the search is nothing but
interpreter steps. That is ~200 states/second, which is not enough for a
four-piece sokoban -- an engine-blackbox A* did solve level 0 (~17 s) but had not
finished level 1 after half an hour. `Model` below is the same mechanics in
~150 lines of Python and runs ~100x faster.

The safety argument is the same one `solvers/generate_autumn_training.py` makes,
in two layers:

  * ``--verify-model`` is a differential fuzz test: random play on every level,
    with the model's full piece/player state and its win flag compared against the
    interpreter after EVERY step. Expected result: zero mismatches. Re-run it
    after any change to the adapter or to ``Baguettes.txt``.
  * every plan is replayed on the real interpreter by `BaguettesExpert._verify`
    before it is returned, and a plan the engine does not win with is discarded.
    Model drift can therefore cost a level, never corrupt the corpus.

Expert solver
-------------
A* over the model whose successors are ``walk to the push cell, then push``
MACROS. Only a push changes anything, so a primitive search would burn its budget
re-deriving walks; branching on macros makes the depth the number of pushes,
while ``g`` stays in primitive MOVES (the unit the agent actually pays) so the
emitted plan is a flat list of directions. States are deduped on
``(pieces, player's reachable region)`` -- the standard sokoban canonicalisation:
two states with the same pieces and the same player region admit exactly the same
futures.

The heuristic matches PIECES, not cells. It segments the board into maximal
baguettes and the target map into maximal target runs, keys both by
``(orientation, length)``, and takes the minimum-cost assignment of target runs to
distinct pieces of the same key. A pair costs the target's own PUSH-DISTANCE
FIELD read at the piece's anchor: a BFS over the positions where that shape's
whole footprint fits, i.e. the relaxation "this piece alone on an otherwise empty
board". A per-cell "nearest matching bag cell" estimate -- the obvious first try
-- is much weaker, because the five cells of one baguette all claim the same
nearest target and the estimate collapses to a constant while the piece is
anywhere near the goal. Target runs with no piece of their shape yet must be
BAKED and are charged ``dough -> oven -> target``; a state with such a run and no
dough left is unwinnable and is charged ``_DEAD``, which is what prunes the
branches that bake a blob into the wrong shape and strand the level.

Solvable levels
---------------
All 9, in 510 moves total (``--report`` prints the per-level lengths). The whole
search costs ~100 s once at seed 0 -- level 8 alone is ~70 s of that and level 1
~20 s, the other seven are a couple of seconds together -- and every later seed
replays from the plan cache, since the engine state after reset is
seed-independent and only the presentation is augmented.

Palette fix (this game was UNPLAYABLE as shipped)
------------------------------------------------
``doughball``'s three shades (``#ccc266 #b7ae5d #c1b860``) all quantize to ARC
index 12 -- and so does the ``#a89855`` Background. Raw dough rendered as a solid
background-coloured block: literally invisible on all four oven levels. Fixed in
``data/puzzlescript_games/Baguettes.txt`` by recolouring dough to a pale grey
(ARC 1, unused elsewhere in this game), which keeps it distinct from both the
field (12) and the baked yellow baguettes (11). See the note in that file.

The remaining ambiguity is deliberate and harmless: ``bagmid`` and ``bagmidup``
both render as a solid 11 block (as do ``targetmid`` / ``targetmidup`` in 13),
because only a piece's END cells carry the transparent corner pixels that encode
orientation. A mid cell never occurs except inside a run whose ends are visible,
so a piece's orientation is always readable off the piece as a whole.

Augmentation
------------
Baguettes' engine state after reset is identical for every seed (levels are fixed
ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`, where the argument for this game is written
out), which together re-sample the board's 8-element symmetry group. There is
deliberately no RECOLOR augmentation: the recolor surfaces flatten an object to a
single flat colour, and that would erase the transparent-corner code that is the
ONLY thing distinguishing a bagleft from a bagright from a bagup.

The plans are therefore seed-independent: solved once per level, cached, and
replayed per seed with that seed's remapped screen actions.

Usage (run from the repo root):
    python solvers/generate_baguettes_training.py --episodes 200 \
        --out data/training_multi_level/baguettes
    python solvers/generate_baguettes_training.py --report
    python solvers/generate_baguettes_training.py --verify-model
"""

from __future__ import annotations

import heapq
import sys
from collections import deque
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (PSAStarSolver, PSExpert,  # noqa: E402
                                     restore, snapshot)

GAME_NAME = "Baguettes"

#: The four engine directions, in the order the search branches on them.
MOVES = ("up", "down", "left", "right")

#: Object codes used inside the model. Only these seven things ever move.
DOUGH, BL, BM, BR, BU, BMU, BD, BB = range(8)

#: The piece class each target class demands, straight off the WINCONDITIONS.
_TARGET_OF = {
    "targetleft": BL, "targetmid": BM, "targetright": BR,
    "targetup": BU, "targetmidup": BMU, "targetdown": BD,
    "targetball": BB,
}
_PIECE_OF = {
    "bagleft": BL, "bagmid": BM, "bagright": BR,
    "bagup": BU, "bagmidup": BMU, "bagdown": BD,
    "bagball": BB, "doughball": DOUGH,
}

#: Heuristic charge for a state that can no longer be won -- a target run whose
#: shape is gone from the board and cannot be baked (the dough is spent). Large
#: enough to sink such a node to the back of the queue, finite so the search
#: stays complete.
_DEAD = 10_000

#: Charge for pairing a target with a piece of its shape that no sequence of
#: pushes can bring to it (they are walled apart). Far above any real distance so
#: the assignment reaches for any reachable piece first, but well below `_DEAD`:
#: another piece of the same shape may still reach it, so this is not a verdict
#: on the state.
_UNREACHABLE = 500

#: Above this many candidates in one shape class, skip the exact assignment and
#: fall back to greedy. No shipped level comes close (the largest class holds
#: four runs), so this only bounds the pathological case.
_EXACT_ASSIGN_CAP = 6


# ---------------------------------------------------------------------------
# Static board
# ---------------------------------------------------------------------------

class Static:
    """Everything about a level that no rule can ever change.

    Cells are flat ``r * W + c`` indices. ``nb[d][i]`` is the neighbour of ``i``
    in direction ``d`` or ``-1`` off the board, which is where every bounds and
    row-wrap check lives -- the ASCII maps are all fully wall-bordered, so ``-1``
    is only ever reached by a pattern reaching past an edge, never by a move.
    """

    __slots__ = ("h", "w", "walls", "ovens", "targets", "nb", "runs", "fields",
                 "spots")

    def __init__(self, h, w, walls, ovens, targets):
        self.h, self.w = h, w
        self.walls = walls              # frozenset[int]
        self.ovens = ovens              # frozenset[int]
        self.targets = targets          # tuple[(cell, code)]
        self.nb = {
            d: tuple(self._step(i, d) for i in range(h * w)) for d in MOVES
        }
        self.runs = self._target_runs()
        self.fields = {(kind, n, a): self._field(kind, n, a)
                       for (kind, n), anchors in self.runs.items()
                       for a in anchors}
        self.spots = {key: self._bake_spots(*key) for key in self.runs}

    def _step(self, i, d) -> int:
        r, c = divmod(i, self.w)
        dr, dc = _DELTA[d]
        r += dr
        c += dc
        return r * self.w + c if 0 <= r < self.h and 0 <= c < self.w else -1

    def _target_runs(self) -> dict:
        """Segment the target map into maximal runs keyed ``(orientation,
        length)`` and anchored at the run's head cell. Two runs with the same key
        are the same rigid shape, which is what makes a piece interchangeable
        between them."""
        want = dict(self.targets)
        runs: dict = {}
        for head, mid, tail, kind, d in ((BL, BM, BR, "h", "right"),
                                         (BU, BMU, BD, "v", "down")):
            for i, code in want.items():
                if code != head:
                    continue
                n, j = 1, self.nb[d][i]
                while j >= 0 and want.get(j) == mid:
                    n += 1
                    j = self.nb[d][j]
                if j >= 0 and want.get(j) == tail:
                    n += 1
                runs.setdefault((kind, n), []).append(i)
        for i, code in want.items():
            if code == BB:
                runs.setdefault(("o", 1), []).append(i)
        return {k: tuple(sorted(v)) for k, v in runs.items()}

    def _bake_spots(self, kind, n) -> tuple:
        """Every place a run of this shape could be BAKED: the anchors of the
        contiguous stretches of ``n`` ovens in this orientation.

        A run of ``n`` only ever comes out of ``n`` oven cells carrying dough at
        the same instant -- every bake rule that produces a mid or an end cell
        requires ``oven`` on that cell -- so a target shape with no stretch this
        long is a shape the board simply cannot manufacture. That makes the list
        both the heuristic's aiming point while dough is still raw and, when it
        is empty, a proof that the state is dead."""
        along = "right" if kind == "h" else "down"
        out = []
        for i in self.ovens:
            j, ok = i, True
            for _ in range(n):
                if j < 0 or j not in self.ovens:
                    ok = False
                    break
                j = self.nb[along][j]
            if ok:
                out.append(i)
        return tuple(sorted(out))

    def footprint(self, kind, n, anchor) -> tuple:
        """The cells a run of this shape occupies when anchored at ``anchor``."""
        along = "right" if kind == "h" else "down"
        cells, i = [], anchor
        for _ in range(n):
            cells.append(i)
            i = self.nb[along][i]
        return tuple(cells)

    def _field(self, kind, n, anchor) -> dict:
        """Push-distance field for ONE target run: ``{anchor position: minimum
        pushes to translate a piece of this shape onto the target}``.

        Manhattan distance is the obvious estimate and it is badly wrong on these
        boards -- a five-cell baguette two cells from its target may have to be
        walked the length of the room and back, because every intermediate
        position puts one of its cells inside a wall. This is a BFS over the
        positions where the shape's whole footprint FITS, which is exactly the
        relaxation "this piece alone on an otherwise empty board", so it stays a
        lower bound on the real pushes while knowing about the geometry.

        A position missing from the field is one no sequence of pushes can bring
        to this target; `Planner._class_cost` charges those `_UNREACHABLE`."""
        along = "right" if kind == "h" else "down"

        def fits(i) -> bool:
            for _ in range(n):
                if i < 0 or i in self.walls:
                    return False
                i = self.nb[along][i]
            return True

        if not fits(anchor):
            return {}
        dist = {anchor: 0}
        queue = deque([anchor])
        while queue:
            i = queue.popleft()
            step = dist[i] + 1
            for d in MOVES:
                j = self.nb[d][i]
                if j >= 0 and j not in dist and fits(j):
                    dist[j] = step
                    queue.append(j)
        return dist


_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


class State:
    """The mutable half of a position: where the player is and what is where."""

    __slots__ = ("player", "occ")

    def __init__(self, player: int, occ: dict):
        self.player = player
        self.occ = occ

    def pieces(self) -> frozenset:
        return frozenset(self.occ.items())


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Model:
    """Baguettes' mechanics, natively.

    Verified against the interpreter by `_verify_model`; see the module
    docstring for why the interpreter itself is not used for search.
    """

    def __init__(self, static: Static):
        self.st = static

    # -- rigid bodies ---------------------------------------------------------
    def bodies(self, occ: dict) -> dict:
        """Map every occupied cell to the tuple of cells that translate WITH it.

        A body is one baguette (the ``moving`` propagation runs along the piece's
        own axis whichever way it is pushed) or one orthogonally-connected blob of
        dough (``[moving doughball | doughball]`` carries no direction prefix, so
        a blob is rigid in all four directions). Balls are singletons.

        Runs are claimed head-first -- all ``bagleft``s, then all ``bagup``s, then
        dough blobs -- so a mid cell is never mistaken for a loose piece. Anything
        still unclaimed after that (which only a partial bake can produce) falls
        through as a singleton rather than being dropped."""
        nb = self.st.nb
        out: dict = {}
        for head, mid, tail, d in ((BL, BM, BR, "right"), (BU, BMU, BD, "down")):
            for i, code in occ.items():
                if code != head or i in out:
                    continue
                cells = [i]
                j = nb[d][i]
                while j >= 0 and occ.get(j) == mid:
                    cells.append(j)
                    j = nb[d][j]
                if j >= 0 and occ.get(j) == tail:
                    cells.append(j)
                body = tuple(cells)
                for cell in cells:
                    out[cell] = body
        for i, code in occ.items():
            if code != DOUGH or i in out:
                continue
            blob = [i]
            out[i] = None
            queue = deque([i])
            while queue:
                cur = queue.popleft()
                for d in MOVES:
                    j = nb[d][cur]
                    if j >= 0 and j not in out and occ.get(j) == DOUGH:
                        out[j] = None
                        blob.append(j)
                        queue.append(j)
            body = tuple(blob)
            for cell in blob:
                out[cell] = body
        for i in occ:
            if i not in out:
                out[i] = (i,)
        return out

    # -- one turn -------------------------------------------------------------
    def step(self, s: State, d: str) -> State:
        """Apply one key press. Returns the new state, or ``s`` itself if the
        turn was a no-op (walked into a wall, or a push the rules cancelled)."""
        st = self.st
        nb = st.nb[d]
        front = nb[s.player]
        if front < 0 or front in st.walls:
            return s                              # the collision layer stops us
        occ = s.occ
        if front not in occ:
            # A bare player move. Nothing else can fire: every startloop rule
            # needs a PIECE moving, and no dough is ever at rest on an oven for
            # the bake rules to find (see `_bake`).
            return State(front, occ)

        # Everything the push sets in motion. `[> Bagget | Bagget]` and the dough
        # equivalents shove whatever is in the way, so this is the transitive
        # closure of "body, and any body its destination cells run into"; the
        # whole turn is cancelled the moment any of it would enter a wall
        # (`[> bagget | wall] -> cancel`).
        bodies = self.bodies(occ)
        moving = set(bodies[front])
        queue = list(moving)
        while queue:
            cell = queue.pop()
            dest = nb[cell]
            if dest < 0 or dest in st.walls:
                return s                          # cancel: the turn never happened
            if dest in occ and dest not in moving:
                for other in bodies[dest]:
                    if other not in moving:
                        moving.add(other)
                        queue.append(other)
        new = {i: code for i, code in occ.items() if i not in moving}
        for cell in moving:
            new[nb[cell]] = occ[cell]
        self._bake(new)
        return State(front, new)

    def won(self, s: State) -> bool:
        occ = s.occ
        return all(occ.get(cell) == code for cell, code in self.st.targets)

    # -- the oven ------------------------------------------------------------
    def _bake(self, occ: dict) -> None:
        """Run the thirteen ``late`` rules, in source order, each to fixpoint.

        Together they turn whatever dough is standing on ovens into a baguette of
        that dough's own shape: the interior of a run first (rules 1-4), then the
        ends (5-8), then bare pairs with nothing to extend (9-12), and finally
        `late [doughball oven] -> [bagball oven]` sweeps up every dough cell the
        shape rules did not claim. That last rule is why no dough is EVER at rest
        on an oven, which is both the reason a lone dough shoved onto an oven
        becomes a ball on the spot and the reason `step` can treat a bare player
        move as inert.

        Order is load-bearing where a dough shape is both horizontally and
        vertically extended: the horizontal rules are written first and so claim
        such a cell first. Getting that wrong is exactly what `_verify_model`
        catches."""
        st = self.st
        nb, ovens = st.nb, st.ovens

        def is_dough(i) -> bool:
            return i >= 0 and occ.get(i) == DOUGH and i in ovens

        def fix(collect) -> None:
            while True:
                changes = collect()
                if not changes:
                    return
                occ.update(changes)

        def triple(d, code):                 # rules 1 and 3: the interior of a run
            def collect():
                out = {}
                for i in occ:
                    if not is_dough(i):
                        continue
                    j = nb[d][i]
                    if is_dough(j) and is_dough(nb[d][j]):
                        out[j] = code
                return out
            return collect

        def extend(dirs, mid, code):         # rules 2 and 4: grow that interior
            def collect():
                out = {}
                for i, c in occ.items():
                    if c != mid or i not in ovens:
                        continue
                    for d in dirs:
                        j = nb[d][i]
                        if j >= 0 and is_dough(j) and is_dough(nb[d][j]):
                            out[j] = code
                return out
            return collect

        def cap(d, mid, code):               # rules 5-8: the end cells of a run
            def collect():
                out = {}
                for i, c in occ.items():
                    if c == mid and i in ovens and is_dough(nb[d][i]):
                        out[nb[d][i]] = code
                return out
            return collect

        def pair(d, back, head, tail, bare_oven):    # rules 9-12: a lone pair
            def collect():
                out = {}
                for i in occ:
                    if not is_dough(i):
                        continue
                    j = nb[d][i]
                    if not is_dough(j):
                        continue
                    a, b = nb[back][i], nb[d][j]
                    if a < 0 or b < 0:
                        continue             # the pattern runs off the board
                    if bare_oven:
                        ok = a not in ovens and b not in ovens
                    else:
                        ok = occ.get(a) != DOUGH and occ.get(b) != DOUGH
                    if ok:
                        out[i], out[j] = head, tail
                return out
            return collect

        fix(triple("right", BM))                                        # 1
        fix(extend(("right", "left"), BM, BM))                          # 2
        fix(triple("down", BMU))                                        # 3
        fix(extend(("down", "up"), BMU, BMU))                           # 4
        fix(cap("right", BM, BR))                                       # 5
        fix(cap("left", BM, BL))                                        # 6
        fix(cap("up", BMU, BU))                                         # 7
        fix(cap("down", BMU, BD))                                       # 8
        fix(pair("right", "left", BL, BR, False))                       # 9
        fix(pair("right", "left", BL, BR, True))                        # 10
        fix(pair("down", "up", BU, BD, False))                          # 11
        fix(pair("down", "up", BU, BD, True))                           # 12
        fix(lambda: {i: BB for i in occ if is_dough(i)})                 # 13


def read_state(eng, game) -> tuple[Static, State]:
    """Build ``(Static, State)`` from the interpreter's live grid.

    Read from the engine rather than re-parsed from the .txt so the model always
    starts from the post-``run_rules_on_level_start`` state -- which for this game
    is the tick that draws the wall shading and bakes any dough the map happened
    to place on an oven."""
    idx = game.obj_name_to_idx
    piece_ids = {idx[name]: code for name, code in _PIECE_OF.items()}
    target_ids = {idx[name]: code for name, code in _TARGET_OF.items()}
    wall_ids = set(game.resolve_object_name("wall"))
    player_ids = set(eng._player_indices)
    oven = idx["oven"]

    h, w = eng.height, eng.width
    walls, ovens, targets, occ = set(), set(), [], {}
    player = -1
    for r in range(h):
        for c in range(w):
            i = r * w + c
            cell = eng.grid[r][c]
            if cell & wall_ids:
                walls.add(i)
            if oven in cell:
                ovens.add(i)
            if cell & player_ids:
                player = i
            for obj in cell:
                if obj in target_ids:
                    targets.append((i, target_ids[obj]))
                elif obj in piece_ids:
                    occ[i] = piece_ids[obj]
    static = Static(h, w, frozenset(walls), frozenset(ovens),
                    tuple(sorted(targets)))
    return static, State(player, occ)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    """A* over `Model` branching on ``walk to the push cell, then push`` macros."""

    def __init__(self, model: Model, node_cap: int, weight: int):
        self.m = model
        self.st = model.st
        self.node_cap = node_cap
        self.weight = weight

    # -- macros ---------------------------------------------------------------
    def analyze(self, s: State) -> tuple:
        """Return ``(region, macros)``: the canonical id of the player's walkable
        region and every ``walk + push`` available right now, each a flat list of
        primitive directions.

        Exactly ONE macro per (body, direction). The generic per-CELL enumeration
        in `PSPushExpert` would offer five for a five-cell baguette shoved
        broadside -- five identical successors that differ only in which cell of
        the same piece the player stood under. Pushes whose body would land on a
        wall are dropped here too: ``[> bagget | wall] -> cancel`` makes them
        guaranteed no-ops. The test looks only at the body's own footprint, never
        at a piece it might shove into a wall further down a chain, so it can
        never drop a push that would have worked."""
        st, occ = self.st, s.occ
        nb, walls = st.nb, st.walls
        parent: dict = {s.player: None}
        queue = deque([s.player])
        while queue:
            cur = queue.popleft()
            for d in MOVES:
                j = nb[d][cur]
                if j >= 0 and j not in parent and j not in walls and j not in occ:
                    parent[j] = (cur, d)
                    queue.append(j)

        def walk_to(cell) -> list:
            out = []
            while parent[cell] is not None:
                cell, d = parent[cell]
                out.append(d)
            out.reverse()
            return out

        bodies = self.m.bodies(occ)
        macros = []
        # SORTED, not just deduped: a set of tuples iterates in hash order, and
        # that order decides which of two equal-cost macros A* expands first.
        # Leaving it unsorted makes the plan for a level depend on
        # PYTHONHASHSEED, so the shards of one parallel generation run would
        # disagree about a level's trajectory.
        for body in sorted(set(bodies.values())):
            cells = set(body)
            for d in MOVES:
                fwd, back = nb[d], nb[_OPPOSITE[d]]
                if any(fwd[cell] < 0 or fwd[cell] in walls
                       for cell in body if fwd[cell] not in cells):
                    continue                      # the engine would cancel
                stands = [back[cell] for cell in body
                          if back[cell] in parent]
                if not stands:
                    continue
                macros.append(min((walk_to(cell) for cell in stands), key=len)
                              + [d])
        return min(parent), macros

    # -- heuristic ------------------------------------------------------------
    def heuristic(self, s: State) -> int:
        st, occ = self.st, s.occ
        bag_runs = self._bag_runs(occ)
        dough = [i for i, code in occ.items() if code == DOUGH]

        # Dough budget. Every baked cell consumes exactly one dough cell and
        # nothing ever un-bakes, so if the shapes still missing need more cells
        # than there is dough left, the level is already lost. This is what
        # prunes the enormous subtree where a blob was shoved onto the ovens in
        # the wrong formation -- the state looks locally fine (the pieces are
        # right there) and only this counting argument says otherwise.
        if sum(max(0, len(a) - len(bag_runs.get(k, ()))) * k[1]
               for k, a in st.runs.items()) > len(dough):
            return _DEAD

        total = 0
        for (kind, n), anchors in st.runs.items():
            total += self._class_cost(kind, n, anchors,
                                      bag_runs.get((kind, n), ()), dough)
        if total == 0 or total >= _DEAD:
            return total
        # The player still has to walk to whatever it is going to push next.
        if occ:
            total += max(0, min(_manhattan(s.player, i, st.w) for i in occ) - 1)
        return total

    def _bag_runs(self, occ: dict) -> dict:
        """The pieces on the board, keyed the same way `Static._target_runs`
        keys the targets, so a key match means "this piece has exactly this
        target's shape"."""
        nb = self.st.nb
        runs: dict = {}
        for head, mid, tail, kind, d in ((BL, BM, BR, "h", "right"),
                                         (BU, BMU, BD, "v", "down")):
            for i, code in occ.items():
                if code != head:
                    continue
                n, j = 1, nb[d][i]
                while j >= 0 and occ.get(j) == mid:
                    n += 1
                    j = nb[d][j]
                if j >= 0 and occ.get(j) == tail:
                    n += 1
                runs.setdefault((kind, n), []).append(i)
        for i, code in occ.items():
            if code == BB:
                runs.setdefault(("o", 1), []).append(i)
        return runs

    def _class_cost(self, kind, n, anchors, pieces, dough) -> int:
        """Minimum assignment cost for ONE shape class.

        ``anchors`` are the target runs of this shape and ``pieces`` the existing
        baguettes of the same shape; each target takes a distinct piece, at that
        target's own push-distance field read at the piece's anchor, and any
        target left over has to be baked. Both lists are tiny -- four runs is the
        most any shipped level puts in one class -- so the injection is
        enumerated exactly rather than approximated."""
        t, b = len(anchors), len(pieces)
        if t == 0:
            return 0
        bake = self._bake_cost(kind, n, anchors, dough)
        if b == 0:
            return sum(bake)
        fields = self.st.fields
        cost = [[fields[(kind, n, a)].get(p, _UNREACHABLE) for p in pieces]
                for a in anchors]
        if t > _EXACT_ASSIGN_CAP or b > _EXACT_ASSIGN_CAP:
            return _greedy_cost(cost, bake)
        if b >= t:
            return min(sum(cost[i][j] for i, j in enumerate(combo))
                       for combo in permutations(range(b), t))
        # Fewer pieces than targets: pick which targets they cover, bake the rest.
        best = 1 << 30
        for combo in permutations(range(t), b):
            covered = set(combo)
            best = min(best,
                       sum(cost[i][j] for j, i in enumerate(combo))
                       + sum(bake[i] for i in range(t) if i not in covered))
        return best

    def _bake_cost(self, kind, n, anchors, dough) -> list:
        """Per-target charge for a run that has to be created from dough.

        Two legs. ASSEMBLY: pick the oven stretch the run will be baked on and
        charge, for each of its ``n`` cells, the distance to the nearest dough
        cell -- so the estimate falls as the blob is walked towards the ovens
        instead of sitting flat at a constant, which is what a "distance to the
        nearest oven" charge does and why it gives the search nothing to steer
        with through the whole assembly phase. DELIVERY: the target's own
        push-distance field read at that oven stretch, i.e. what carrying the
        baked piece to the target will cost. The cheapest stretch wins.

        The estimate double-counts a dough cell that two runs both want and it
        ignores that the pieces have to be lined up rather than merely nearby, so
        it is a search guide and not a bound. Where it IS exact is the negative
        case: no dough left, or no stretch of ``n`` ovens on the board at all,
        means this run can never exist and the state is dead."""
        st = self.st
        spots = st.spots.get((kind, n), ())
        if not dough or not spots:
            return [_DEAD] * len(anchors)
        assemble = {
            spot: sum(min(_manhattan(cell, d, st.w) for d in dough)
                      for cell in st.footprint(kind, n, spot))
            for spot in spots
        }
        return [min(assemble[spot]
                    + st.fields[(kind, n, a)].get(spot, _UNREACHABLE)
                    for spot in spots)
                for a in anchors]

    # -- search ---------------------------------------------------------------
    def search(self, s0: State) -> list | None:
        """Weighted A* in primitive MOVES. Any plan returned is a genuine WIN
        path on the model; at ``weight > 1`` it is not guaranteed shortest."""
        if self.m.won(s0):
            return []
        w = self.weight
        region, macros = self.analyze(s0)
        counter = nodes = 0
        pq = [(w * self.heuristic(s0), 0, counter, s0, [], macros)]
        best = {(s0.pieces(), region): 0}
        while pq:
            _f, g, _c, s, path, macros = heapq.heappop(pq)
            for macro in macros:
                nxt = s
                for d in macro:
                    nxt = self.m.step(nxt, d)
                nodes += 1
                if self.m.won(nxt):
                    return path + macro
                if nxt.occ == s.occ and nxt.player == s.player:
                    continue                      # a push the rules refused
                nregion, nmacros = self.analyze(nxt)
                key = (nxt.pieces(), nregion)
                ng = g + len(macro)
                if best.get(key, 1 << 30) <= ng:
                    continue
                best[key] = ng
                counter += 1
                heapq.heappush(pq, (ng + w * self.heuristic(nxt), ng, counter,
                                    nxt, path + macro, nmacros))
                if nodes >= self.node_cap:
                    return None
        return None


_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def _manhattan(a: int, b: int, w: int) -> int:
    return abs(a // w - b // w) + abs(a % w - b % w)


def _greedy_cost(cost, bake) -> int:
    """Nearest-first fallback for shape classes too large to enumerate."""
    free = set(range(len(cost[0])))
    total = 0
    for i, row in enumerate(cost):
        if not free:
            total += bake[i]
            continue
        j = min(free, key=lambda k: row[k])
        free.discard(j)
        total += row[j]
    return total


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class BaguettesExpert(PSExpert):
    """`PSExpert` that plans on `Model` and verifies on the interpreter.

    `PSExpert.plan` still owns the memo, the snapshot/restore discipline and the
    level scoping; only the search strategy is replaced."""

    directions = list(MOVES)

    def setup(self) -> None:
        self._static: dict = {}

    def _search(self, eng) -> list | None:
        static, state = read_state(eng, self.g)
        # `Static` builds one push-distance BFS per target run; walls, ovens and
        # targets are what it is derived from and no rule touches any of them, so
        # every level pays for its fields exactly once however often we re-plan.
        sig = (static.walls, static.ovens, static.targets)
        static = self._static.setdefault(sig, static)
        plan = Planner(Model(static), self.node_cap, self.weight).search(state)
        return None if plan is None else self._verify(eng, plan)

    def _verify(self, eng, plan: list) -> list | None:
        """Replay ``plan`` on the real interpreter; return it truncated at the
        winning step, or ``None`` if the engine does not win with it. This is what
        keeps model drift from ever reaching the corpus.

        Leaves the engine exactly as it was -- including the win/restart FLAGS,
        which `ps_astar.restore` does not touch. ``step`` clears them on entry, so
        they only matter to a ``check_win`` asked before the next step, which is
        precisely what `record_level` does on its first iteration: a winning flag
        left behind there would end the recording before it took a single
        action."""
        snap = snapshot(eng)
        flags = (eng._rule_win, eng._rule_restart)
        won = -1
        for i, d in enumerate(plan):
            eng.step(d)
            if eng.check_win():
                won = i
                break
        restore(eng, snap)
        eng._rule_win, eng._rule_restart = flags
        return plan[:won + 1] if won >= 0 else None

    def heuristic(self, eng) -> int:        # pragma: no cover - unused
        raise NotImplementedError("BaguettesExpert plans on the model")


#: Process-wide adapter + plan cache + solvable-level set; see `_ensure`.
_SHARED: dict = {}


class BaguettesSolver(PSAStarSolver):
    game_id = "puzzlescript_baguettes"
    game_name = GAME_NAME
    expert_cls = BaguettesExpert

    #: Level 8 is the only level that needs more than ~20 s / ~500k macros; the
    #: other eight are all under that put together.
    node_cap = 2_000_000
    #: The estimate counts PUSHES while ``g`` counts primitive MOVES, and these
    #: solutions run about three moves per push (the player walking around a piece
    #: to get behind it). Weighting by that ratio is what keeps the crowded boards
    #: tractable without the plans wandering the way a much larger weight makes
    #: them.
    weight = 3
    #: Level 8's plan is 90 moves, plus the exploration prefix.
    max_steps = 400

    def _ensure(self, seed: int):
        """Share the adapter, the plan cache and the solvable set process-wide.

        `test_datagen.py` builds a fresh solver per run, and the searches here
        cost ~100 s in total (nearly all of it level 8); without this every run
        would re-pay them for plans that cannot differ, since the engine state
        after reset is seed-independent."""
        if self._game is None and "game" in _SHARED:
            self._game = _SHARED["game"]
            self._expert = _SHARED["expert"]
            self._solvable = _SHARED["solvable"]
        game, expert, solvable = super()._ensure(seed)
        _SHARED.update(game=game, expert=expert, solvable=solvable)
        return game, expert, solvable


# ---------------------------------------------------------------------------
# Model verification and coverage report
# ---------------------------------------------------------------------------

def _engine_state(eng, game) -> tuple:
    """The slice of the interpreter's grid the model claims to reproduce."""
    idx = game.obj_name_to_idx
    piece_ids = {idx[name]: code for name, code in _PIECE_OF.items()}
    player_ids = set(eng._player_indices)
    w = eng.width
    occ = {}
    player = -1
    for r in range(eng.height):
        for c in range(w):
            cell = eng.grid[r][c]
            if cell & player_ids:
                player = r * w + c
            for obj in cell:
                if obj in piece_ids:
                    occ[r * w + c] = piece_ids[obj]
    return tuple(sorted(occ.items())), player


def _model_state(s: State) -> tuple:
    return tuple(sorted(s.occ.items())), s.player


def _verify_model(episodes: int = 40, steps: int = 150) -> int:
    """Differential test: play randomly and compare the model to the interpreter
    after EVERY step, on every level.

    This is the test that built `Model` -- it is what pinned down the rigidity of
    a dough blob and the order the bake rules claim a cell in -- and it lives in
    the repo rather than in a scratch file because the model's fidelity is the
    whole safety argument for planning off the engine. Re-run it after any change
    to the adapter or to ``Baguettes.txt``. Expected result: zero mismatches.
    """
    import random

    from adapters.puzzlescript_adapter import PuzzleScriptAdapter

    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    bad = 0
    for level in range(game.n_levels):
        for ep in range(episodes):
            rng = random.Random(level * 1000 + ep)
            game.set_level(level)
            eng = game._engine
            static, s = read_state(eng, game._game)
            model = Model(static)
            for t in range(steps):
                d = rng.choice(MOVES)
                eng.step(d)
                s = model.step(s, d)
                if (_engine_state(eng, game._game) != _model_state(s)
                        or eng.check_win() != model.won(s)):
                    print(f"  MISMATCH level {level} episode {ep} step {t} ({d})")
                    bad += 1
                    break
                if eng.check_win():
                    break
        print(f"  level {level:2d}: checked {episodes} episodes")
    print("model matches the interpreter" if not bad
          else f"{bad} MISMATCHING episodes")
    return 1 if bad else 0


def _report() -> int:
    """Print per-level plan lengths -- the coverage check for this game."""
    solver = BaguettesSolver()
    game, expert, solvable = solver._ensure(0)
    total = 0
    for level in range(game.n_levels):
        if level in solver.skip_levels:
            print(f"  level {level:2d}: skipped (not winnable)")
            continue
        game.set_level(level)
        plan = expert.plan(game._engine, level)
        if plan is None:
            print(f"  level {level:2d}: unsolved")
        else:
            total += len(plan)
            print(f"  level {level:2d}: {len(plan):3d} moves")
    print(f"{len(solvable)}/{game.n_levels} levels solved, {total} moves total")
    return 0


if __name__ == "__main__":
    if "--report" in sys.argv:
        sys.exit(_report())
    if "--verify-model" in sys.argv:
        sys.exit(_verify_model())
    sys.exit(BaguettesSolver.main())
