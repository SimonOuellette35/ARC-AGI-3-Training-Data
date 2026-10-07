"""Generate Phase-1 training data for the PuzzleScript game ps:full_circle
("FULL CIRCLE", Julien Grimard).

The harness -- the rotation contract, the trajectory recorder, the plan cache
and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared
with the other ps: generators. This file is the game-specific part: a native
model of the mechanic, a macro A* over it, the exact optimal-action oracle, the
render audit and the differential fuzz test that licenses planning natively at
all.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_full_circle",
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
augmented view, so replaying the recorded actions reproduces the recorded
frames exactly. Every expert step also carries the full set of equally-optimal
presses (see "Optimal-action sets" below).

The game
--------
Sokoban whose crates are QUARTER CIRCLES. Each piece carries two independent
attributes and the puzzle is that they are attributes of different things:

* its **shape** -- which quarter of a circle it draws (TL, TR, BL, BR) -- is
  what a target accepts. ``Target_TL`` takes *either* TL-shaped piece, and no
  other. Targets come in 2x2 blocks, so filling a block draws one full circle;
  the win condition (``No Round``, i.e. every piece has turned into its
  ``_Win`` form) is "every piece is standing on a target of its own shape".
* its **colour** -- blue, green, purple, orange -- is what it is BOLTED TO.
  The four movement rules are written against the four colour or-groups
  (``Round_T`` is blue = {TL, TR}, ``Round_B`` orange = {BL, BR}, ``Round_L``
  green = {TL, BL}, ``Round_R`` purple = {TR, BR}), so pushing a piece pushes
  every orthogonally-connected piece of the SAME COLOUR with it, in any
  direction -- broadside as readily as end-on. The rules are ``rigid``: if any
  member of the blob is blocked, nothing moves at all, the player included.

So a colour is a rigid body and a shape is a key, and the two partitions of the
eight piece types cross: the blue pair is {TL, TR} (a circle's top half) while
the green pair is {TL, BL} (its left half), which is why the two 2x2 blocks of
level 6 cannot simply be filled one at a time.

Three things worth stating because they drive the solver:

* **A blob is at most two pieces, always.** A colour has exactly two piece
  types (``Round_T = Round_TL or Round_TR2``) and no shipped level places a
  type twice, so no colour ever has three pieces on the board and a blob is a
  singleton or a domino. That matters more than it looks: the interpreter's
  ``rigid`` implementation tags each rule MATCH with its own group id rather
  than unioning a whole rigid body, so a three-piece blob can SHEAR: a
  measured example is a vertical run of three blues with the middle one
  against a wall, where the far end moves one cell and the near end and the
  player do not. A two-piece blob has only one propagation match and so cannot
  reach that, which is why `_Board` may implement the clean all-or-nothing
  rule -- ``--fuzz`` puts every two-piece case (end-on, broadside, blocked
  each way) in `_SCENARIOS` and proves the two agree.
* **The ``_Win`` flag is a function of position, not history.** The rules that
  set it are ``late [ Round_TL Target_TL ] -> [ Round_TL_Win Target_TL ]`` and
  -- for taking it away again -- ``late [ Round_TL_Win | Target_TL ] ->
  [ Round_TL | Target_TL ]``, which is not "not on a target" but "with a
  matching target in an ADJACENT cell". It still comes out right, because a
  push moves a piece exactly one cell, so a piece stepping off its target is
  always adjacent to it and always reverts. `_Board.__init__` asserts the one
  layout that would break the equivalence (two same-shape targets side by
  side, which would un-win a correctly parked piece); no level has one.
* **There is no ACTION button** (``noaction``), so the whole game is four
  directions.

Why a native model (and not the shared engine-blackbox A*)
----------------------------------------------------------
Measured first, per the ps:entrepotphage_demake lesson: this game's
interpreter runs at **265 ``eng.step``/s**, against ~20-25k for a plain push
game -- eight movement rules, each a two-rule ``+`` group looped to a
fixpoint over four directions, plus sixteen ``late`` rules that rescan the
board every turn. `PSPushExpert` over it (with the walk-teleport fast path, so
a macro costs ONE interpreter step) solved the tutorial and then failed to
finish level 1 in eight minutes. `_Board` is the same mechanic in about forty
lines of turn rule, and the whole eight-level ladder plans in twenty-five
seconds; the interpreter is left to do what it is good for -- being the
authority. ``--fuzz`` checks the model
against it cell-for-cell over random play, and `record_level` replays the
finished plan through the real adapter to a real WIN.

The search
----------
`_Board.astar` is A* whose successors are ``walk to the push cell, then push``
macros -- the same shape as `PSPushExpert`, re-implemented over the model.

* ``g`` counts primitive MOVES (what the agent pays). Every walk inside a macro
  is a shortest path, dedup is on the EXACT ``(player, pieces)`` pair rather
  than on the player's reachable region (the region key is a real loss of
  optimality for a move-costed search -- it cost ps:entrepotphage_demake 11
  moves), and the heuristic is admissible, so at ``weight = 1`` the plans are
  SHORTEST over primitives: any primitive solution splits into maximal walk
  runs each ending in a push, and those runs are exactly what `macros`
  enumerates.
* A win is kept as an INCUMBENT until the frontier's ``f`` reaches its cost,
  not returned the moment it is generated. Macros have different lengths, so
  generating a win proves nothing about it being the shortest one -- the
  documented `PSPushExpert.exact_goal_test` flaw. It is not academic here:
  returning on generation gave four of the eight levels plans that were one to
  four moves too long, and ``--ties`` is what convicted them.
* The heuristic is a push-distance field per RIGID BODY, not per piece, and
  that is the one design decision the whole thing turns on. A colour whose two
  pieces start adjacent can never come apart, so it is a two-cell body with a
  fixed shape; its landing spots are the placements putting each member on a
  target of that member's shape, and the estimate is the cheapest way of
  handing every body a landing spot that no other body wants (at most
  ``2^4 = 16`` combinations here, enumerated). Admissible because one press
  moves one body one cell, so no press can take more than 1 off the sum --
  where a per-PIECE table would double-count every domino push.
* The same fields are the deadlock test for free, and per-body is what makes
  it CORRECT. A body can be pushed from a cell the player cannot stand behind,
  as long as it can stand behind the body's other member; the per-piece
  sokoban table cannot express that, calls the push impossible and prunes
  states that win. Written that way first, it closed level 6's entire search
  space in 213 states and reported the level unsolvable.

Optimal-action sets
-------------------
Measured, not inferred, but by ONE bounded layered BFS rather than by the
re-solve-per-candidate oracle ps:escape and ps:escaping_limbo use -- level 1's
A* is seventeen seconds, so forty of them per level is not a labelling
strategy. Once weight-1 A* has proved ``d*``, a forward sweep that drops every
state with ``depth + h > d*`` holds exactly the states that can still be on a
shortest path, and a backward sweep marks the ones that are; the answer at
each step is then "every press landing on a marked state in the next layer".
``--ties`` re-derives the same sets the slow way and compares, which is how
the goal-test bug below was caught. No step ever ships unlabelled.

Inferring them instead (`PSPushExpert.annotate_walks`) would be wrong in both
directions here: it calls any step that moved a piece forced, but this game's
pieces sit in the player's way so two different first pushes frequently tie,
and it calls every equally-short walk step free, which a piece blocking one of
the two routes makes false.

Levels
------
Eight, shipped, ordered as the author wrote them (``--plans`` re-derives this
table). ``pieces`` counts the rigid DOMINOES in brackets, ``ties`` the share of
steps with more than one equally-optimal press::

    level  size  pieces  plan  ties  search   what it is
     0     6x 9  4 (0)      2   0%   0.00s  the tutorial: one push, one target
     1     8x 8  4 (0)     42   5%   17.2s  four loose quarters ringing one block
     2     8x 6  4 (0)     39   8%    0.3s  two pieces start on each OTHER's target
     3     7x 8  4 (1)     51   6%    0.1s  the first domino, and a target well
                                            that can only be entered from above
     4     6x 6  4 (0)     29   3%    0.0s  the block jammed against a wall
     5     7x 8  4 (0)     74   3%    1.0s  the block walled into a side room
                                            behind a one-cell doorway
     6     8x 8  8 (4)     73   1%    4.9s  four dominoes, two blocks: blue takes
                                            one block's top row and orange its
                                            bottom, so green and purple must
                                            take the other block's two columns
     7     6x 9  4 (0)     77   1%    0.8s  the four targets are not a block --
                                            a corridor runs between them, and
                                            the halves are swapped top for
                                            bottom

**387 moves over 8 levels, every one of them shortest**, ~25 s of one-time
search cached to ``data/full_circle_plans.json`` (level 1 is 17 s of it).
Delete the file to re-derive.

Shortest by construction (admissible heuristic, weight 1, incumbent goal
test), and shortest by measurement on the four levels small enough to settle
it without one: an exhaustive primitive BFS with no heuristic at all agrees
exactly on levels 0, 2, 3 and 4 (2 / 39 / 51 / 29 moves, up to 2.5M states).
That is the check worth re-running on any change to `_Board.heuristic` -- it
is the only one that does not share an assumption with the search.

The rendering fixes
-------------------
Audited per-composition at every cell size the boards use (``--audit``), which
found the recurring bug of this adapter in its purest form: **a piece standing
on a target it does not fit was pixel-identical to the same piece standing on
bare floor**, on all eight piece types and at every board size.

A target's 5x5 outline was drawn half in ``darkgray`` and half in ``black``,
and the Background is black -- so half of every target was invisible even with
nothing on it, and the visible half sat exactly under the opaque body of any
piece parked there. That is not cosmetic: the whole game is remembering which
quarter goes where, and a board mid-solve is full of pieces resting on targets
that are not theirs. The outlines are now entirely darkgray (colour 0, the
target's nominal blue/purple/green/orange, was already unused and is left that
way -- a coloured target would imply a colour constraint the game does not
have, since ``Target_TR`` accepts the purple TR *and* the blue one). Each of
the twelve piece-on-wrong-target pairs now shows three to five target pixels
through the piece's transparent corner.

The tutorial arrows had the same problem and got the same class of fix: every
body in this game is opaque over the middle of its cell, and each leaves a
DIFFERENT corner transparent, so a hint sprite drawn only in the middle
vanishes under anything. ``Arrow_R`` and ``Arrow_D`` gained their four corner
pixels, which is the smallest set visible under all nine bodies.

Sprites only -- no rule, object, legend or collision layer was touched, so
stored plans are unaffected.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed
ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip, each with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`). The game is gravity-free, its push rules
are stated with the relative ``>`` force, its win condition names no direction
and its input is screen-relative, so a flip is an exact symmetry of the
mechanic. The art IS directional -- a mirrored TL arc is a TR arc -- but it
mirrors CONSISTENTLY: pieces and targets transform together, so the
shape-matches-shape relation the puzzle is about survives untouched, and the
mandatory rotation augmentation already puts the frame in that regime. There
is deliberately no colour augmentation: colour IS the rigid-body grouping, and
a recolor that broke the piece-to-piece link would make the boards unreadable.
8 levels x 16 presentations = 128.

The expert plan is therefore seed-independent: solved once per level, cached
to ``data/full_circle_plans.json``, and replayed per seed with that seed's
remapped screen actions.

Usage (run from the repo root):
    python solvers/generate_full_circle_training.py --episodes 200 \
        --out data/training_multi_level/full_circle

    python solvers/generate_full_circle_training.py --plans  # level report
    python solvers/generate_full_circle_training.py --audit  # render audit
    python solvers/generate_full_circle_training.py --fuzz   # model vs engine
    python solvers/generate_full_circle_training.py --ties   # re-measure labels
    python solvers/generate_full_circle_training.py --bfs    # heuristic-free
                                                             # optimality check
"""

from __future__ import annotations

import heapq
import sys
from collections import deque
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (PSAStarSolver, PSExpert,      # noqa: E402
                                     Plan)

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "FULL_CIRCLE"

#: Disk cache of each level's start plan and its optimal-action sets. The
#: searches are seed-independent, so without a file on disk every shard
#: `parallelize_generator` starts would re-derive all of them. Delete to
#: re-derive.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "full_circle_plans.json"

#: Engine direction -> (dr, dc). The game declares ``noaction``.
_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}
_DIRS = ("up", "down", "left", "right")
_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}

#: The eight piece types, as ``object name -> (shape, colour)``.
#:
#: SHAPE is what a target accepts; COLOUR is the or-group the movement rules
#: are written against, i.e. what the piece is rigidly bolted to. The two
#: partitions cross -- blue is the top half of a circle, green the left half --
#: and that crossing is the game.
_PIECES: tuple[tuple[str, str, str], ...] = (
    ("round_tl",  "TL", "blue"),
    ("round_tl2", "TL", "green"),
    ("round_tr",  "TR", "purple"),
    ("round_tr2", "TR", "blue"),
    ("round_bl",  "BL", "green"),
    ("round_bl2", "BL", "orange"),
    ("round_br",  "BR", "orange"),
    ("round_br2", "BR", "purple"),
)

#: ``_Win`` form of each piece, same (shape, colour). A piece is in its ``_Win``
#: form exactly while it stands on a target of its shape (see the module
#: docstring), so these are the same eight bodies, not eight more.
_WIN_SUFFIX = "_win"

_TARGETS: dict[str, str] = {
    "target_tl": "TL", "target_tr": "TR",
    "target_bl": "BL", "target_br": "BR",
}

#: Heuristic charge for a state that can never win -- a piece pushed somewhere
#: no target of its shape can be reached from. Large enough to sink the node,
#: finite so the search stays complete.
_UNREACHABLE = 10_000


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry plus the turn rule, over ``(player, pieces)``.

    Cells are flat indices ``r * w + c``. ``pieces`` is a tuple of cells, one
    per piece present on this board, in ``kinds`` order; a state is that tuple
    plus the player's cell, which hashes for free and is EXACT (nothing else
    about the board can change -- walls, targets and the decorative arrows are
    static, and the ``_Win`` flag is a function of position).

    The turn, in the order the interpreter resolves it (verified by ``--fuzz``,
    which replays random play and hand-built scenario boards against the real
    engine and compares the grid cell-for-cell):

    1. The player's target cell is read. Off the board, a wall or the
       ``Outside`` filler blocks, and the press does nothing at all.
    2. If the target cell holds a piece, the whole orthogonally-connected blob
       of pieces of that piece's COLOUR moves with it -- in the push direction,
       whether they are lined up with it or beside it.
    3. The move is all-or-nothing: if any blob member's destination is off the
       board, a wall, or a piece outside the blob, then nothing moves, the
       player included. That is what ``rigid`` means.
    4. A piece standing on a target of its own shape is in its ``_Win`` form;
       one that is not, is not. `won` is "every piece is".
    """

    __slots__ = ("h", "w", "wall", "targets", "kinds", "shape", "colour",
                 "start", "nbr", "bodies", "_fields", "_combos", "_hcache")

    def __init__(self, h, w, wall, targets, kinds, start):
        self.h, self.w = h, w
        #: set of blocked cells (Wall_V / Wall_H / Outside -- one collision layer)
        self.wall: frozenset = wall
        #: cell -> target shape
        self.targets: dict = targets
        #: index -> ``_PIECES`` entry, for the pieces this level actually has
        self.kinds: tuple = kinds
        self.shape: tuple = tuple(sh for _n, sh, _c in kinds)
        self.colour: tuple = tuple(co for _n, _s, co in kinds)
        #: ``(player cell, pieces)`` on the grid this was read from
        self.start: tuple = start
        #: cell -> direction -> neighbour cell, or -1 off the board. Precomputed
        #: because `step` and the two BFSs are the whole inner loop, and doing
        #: the bounds arithmetic per call (with the column-wrap check a flat
        #: index needs) was measurably the largest single cost in the search.
        self.nbr = [{d: -1 for d in _DIRS} for _ in range(h * w)]
        for r in range(h):
            for c in range(w):
                for d, (dr, dc) in _DELTA.items():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        self.nbr[r * w + c][d] = nr * w + nc
        self._hcache: dict = {}

        # The ``_Win`` flag is treated as a function of position, which is only
        # sound while no piece can be un-won while standing where it belongs.
        # The reversion rule is `late [ Round_TL_Win | Target_TL ]` -- a
        # matching target in an ADJACENT cell, not "no target underneath" -- so
        # two same-shape targets side by side would strip the flag off a
        # correctly parked piece and the model would report a win the engine
        # does not. No shipped level has one; assert rather than model it.
        for cell, sh in self.targets.items():
            for d in _DIRS:
                nxt = self.nbr[cell][d]
                if nxt >= 0 and self.targets.get(nxt) == sh:
                    raise AssertionError(
                        f"two {sh} targets are adjacent ({cell}, {nxt}): the "
                        "win flag is no longer a function of position")
        # Everything the SEARCH needs is built lazily by `_build_bodies`, so
        # that a board which is not a level -- the hand-built `_SCENARIOS`,
        # which have a piece and no target on purpose -- can still be stepped
        # and compared against the interpreter without inventing targets for it.
        self.bodies = None

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_engine(cls, eng, game) -> "_Board":
        """Read a board (geometry + the grid's current state) off a live
        interpreter.

        A piece is looked up under BOTH of its forms: reading a mid-level grid
        would otherwise silently forget every piece that happens to be parked
        on its target at that moment."""
        blockers = {i for n in ("wall", "outside")
                    for i in game.resolve_object_name(n)}
        player_ids = set(game.resolve_object_name("player"))
        target_ids = {i: sh for n, sh in _TARGETS.items()
                      for i in game.resolve_object_name(n)}
        piece_ids = {}
        for k, (name, _sh, _co) in enumerate(_PIECES):
            for form in (name, name + _WIN_SUFFIX):
                for i in game.resolve_object_name(form):
                    piece_ids[i] = k

        h, w = eng.height, eng.width
        wall, targets, player = set(), {}, -1
        at: dict[int, int] = {}
        for r in range(h):
            for c in range(w):
                cell = eng.grid[r][c]
                if not cell:
                    continue
                idx = r * w + c
                if cell & blockers:
                    wall.add(idx)
                if cell & player_ids:
                    player = idx
                for oid in cell:
                    if oid in target_ids:
                        targets[idx] = target_ids[oid]
                    elif oid in piece_ids:
                        # A board is indexed BY piece type, which assumes a
                        # type appears at most once -- that is what caps a
                        # rigid blob at two pieces and lets `step`'s
                        # all-or-nothing rule stand in for the interpreter's
                        # shearing one. A third piece of a colour would be
                        # silently swallowed here, so say so instead.
                        if piece_ids[oid] in at:
                            raise AssertionError(
                                f"two {_PIECES[piece_ids[oid]][0]} on one "
                                "board: blobs are no longer capped at two")
                        at[piece_ids[oid]] = idx
        kinds = tuple(_PIECES[k] for k in sorted(at))
        pieces = tuple(at[k] for k in sorted(at))
        return cls(h, w, frozenset(wall), targets, kinds, (player, pieces))

    def read_state(self, eng, piece_ids, player_ids) -> tuple:
        """``(player cell, pieces)`` for a live grid of this board's level."""
        w = self.w
        order = {name: i for i, (name, _s, _c) in enumerate(self.kinds)}
        at = [-1] * len(self.kinds)
        player = -1
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & player_ids:
                    player = r * w + c
                for oid in cell & piece_ids.keys():
                    at[order[piece_ids[oid]]] = r * w + c
        if player < 0 or -1 in at:
            # Nothing in this game creates or destroys a piece or the player,
            # so a gap here means the grid is not this board's level at all --
            # which would otherwise plan quietly against a phantom layout.
            raise AssertionError(
                f"grid is missing the player or one of {len(self.kinds)} "
                "pieces this board was built for")
        return (player, tuple(at))

    # -- the turn -------------------------------------------------------------
    def step(self, state, direction):
        """The state after one press. Returns ``state`` itself when the press
        does nothing (a wall bump, or a blob the rules refuse to move)."""
        player, pieces = state
        target = self.nbr[player][direction]
        if target < 0 or target in self.wall:
            return state
        at = {cell: i for i, cell in enumerate(pieces)}
        first = at.get(target)
        if first is None:
            return (target, pieces)

        # The rigid body: every piece orthogonally connected to the pushed one
        # through pieces of the SAME colour. Direction-agnostic -- the
        # propagation rule `[ Moving Round_T | Round_T ]` carries no direction
        # prefix, so it expands over all four and a domino is as rigid
        # broadside as end-on.
        colour = self.colour[first]
        blob = {first}
        stack = [first]
        while stack:
            j = stack.pop()
            for d in _DIRS:
                nxt = self.nbr[pieces[j]][d]
                if nxt < 0:
                    continue
                k = at.get(nxt)
                if k is not None and k not in blob and self.colour[k] == colour:
                    blob.add(k)
                    stack.append(k)

        moved = {}
        for j in blob:
            dest = self.nbr[pieces[j]][direction]
            if dest < 0 or dest in self.wall:
                return state                     # rigid: nothing moves at all
            k = at.get(dest)
            if k is not None and k not in blob:
                return state
            moved[j] = dest
        out = list(pieces)
        for j, dest in moved.items():
            out[j] = dest
        return (target, tuple(out))

    def won(self, state) -> bool:
        """``No Round``: every piece has turned into its ``_Win`` form, i.e.
        stands on a target of its own shape."""
        _player, pieces = state
        targets, shape = self.targets, self.shape
        return all(targets.get(cell) == shape[i]
                   for i, cell in enumerate(pieces))

    # -- macros ---------------------------------------------------------------
    def _walk_bfs(self, state):
        """Shortest-walk tree from the player (``{cell: (prev, direction)}``,
        the root mapped to None) over the cells it may stand on: no wall, no
        piece. Walking into a piece is a push, which is a macro, not a walk."""
        player, pieces = state
        blocked = self.wall | set(pieces)
        parent = {player: None}
        queue = deque([player])
        while queue:
            cur = queue.popleft()
            for d in _DIRS:
                nxt = self.nbr[cur][d]
                if nxt < 0 or nxt in blocked or nxt in parent:
                    continue
                parent[nxt] = (cur, d)
                queue.append(nxt)
        return parent

    @staticmethod
    def _walk_to(parent, cell) -> list:
        out = []
        while parent[cell] is not None:
            cell, d = parent[cell]
            out.append(d)
        out.reverse()
        return out

    def macros(self, state) -> list[list[str]]:
        """``walk to a push cell then push``, for every piece the player can
        get behind and every direction.

        That is the whole move set: the win is placing pieces, so a plan that
        ends anywhere but on a push has a wasted tail, and a walk changes
        nothing (no rule in this game fires on a bare move). Any optimal
        primitive plan therefore splits into maximal walk runs each ending in
        a push, which is exactly what this enumerates."""
        parent = self._walk_bfs(state)
        _player, pieces = state
        out = []
        for cell in pieces:
            for d in _DIRS:
                back = self.nbr[cell][_OPPOSITE[d]]
                if back >= 0 and back in parent:
                    out.append(self._walk_to(parent, back) + [d])
        return out

    # -- rigid bodies ---------------------------------------------------------
    def _build_bodies(self) -> None:
        """Partition the pieces into the RIGID BODIES they are permanently
        welded into, and build a push-distance field per body per landing spot.

        A colour is a body. Two pieces of one colour that start orthogonally
        adjacent can never come apart -- every move translates the whole blob,
        which preserves the offset between them -- so the body's SHAPE is a
        constant of the level and can be reasoned about like a single piece
        with a footprint. (Two same-colour pieces that started apart would be
        two bodies that might later fuse; no level has any, and the assertion
        below says so rather than letting the heuristic quietly assume it.)

        This is what makes the heuristic both admissible and sharp:

        * the per-body push distance counts PRESSES, and one press moves one
          body one cell, so summing over bodies is a lower bound -- where
          summing over PIECES would double-count every domino push;
        * the landing spots of a body are the anchors that put each of its
          members on a target of that member's shape, which for a domino is a
          far stronger statement than "each piece is somewhere it fits". Level
          6's four dominoes have exactly two landing spots each, and only two
          of the sixteen combinations are disjoint;
        * the field is the DEADLOCK test for free, and a correct one: a body
          may be pushed from a cell the player cannot stand behind, as long as
          it can stand behind the body's OTHER member. The per-piece sokoban
          table has no way to express that and prunes states that are perfectly
          winnable -- it closed level 6's whole search space in 213 states.
        """
        # `heuristic` reaches 0 exactly at a win only when the shapes balance.
        for sh in set(self.shape):
            n_t = sum(1 for s in self.targets.values() if s == sh)
            n_p = sum(1 for s in self.shape if s == sh)
            if n_t != n_p:
                raise AssertionError(
                    f"{n_p} {sh} pieces for {n_t} {sh} targets")
        w = self.w
        _player, pieces = self.start
        groups: dict = {}
        for i, cell in enumerate(pieces):
            groups.setdefault(self.colour[i], []).append(i)

        self.bodies = []
        for _co, idxs in sorted(groups.items()):
            base = pieces[idxs[0]]
            br, bc = divmod(base, w)
            offs = tuple(tuple(x - y for x, y in
                               zip(divmod(pieces[i], w), (br, bc)))
                         for i in idxs)
            if len(idxs) > 1:
                touching = any(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                               for a in offs for b in offs if a != b)
                if not touching:
                    raise AssertionError(
                        f"{_co} starts as {len(idxs)} separate pieces: they "
                        "could fuse mid-level and the body model would be wrong")
            self.bodies.append((tuple(idxs), offs))

        # Per body: every landing spot, and the reverse push-distance field to
        # it over body ANCHORS (the position of the body's first member).
        self._fields = []
        for idxs, offs in self.bodies:
            spots = {}
            for ar in range(self.h):
                for ac in range(self.w):
                    for i, off in zip(idxs, offs):
                        cell = self._at(ar + off[0], ac + off[1])
                        if cell < 0 or self.targets.get(cell) != self.shape[i]:
                            break
                    else:
                        spots[ar * w + ac] = self._field(offs, ar * w + ac)
            self._fields.append(spots)

        # The combinations of landing spots that could all be taken at once.
        # Overlapping ones cannot, and dropping them is a sound tightening --
        # the true solution's spots are disjoint, so the minimum over disjoint
        # combinations is still a lower bound. Precomputed because it depends
        # on nothing but the level.
        self._combos = []
        for combo in product(*[sorted(f) for f in self._fields]):
            seen: set = set()
            for (idxs, offs), anchor in zip(self.bodies, combo):
                foot = self._foot(offs, anchor)
                if foot & seen:
                    break
                seen |= foot
            else:
                self._combos.append(combo)
        if not self._combos:
            raise AssertionError("no disjoint set of landing spots exists")

    def _at(self, r, c) -> int:
        return r * self.w + c if 0 <= r < self.h and 0 <= c < self.w else -1

    def _foot(self, offs, anchor) -> set:
        ar, ac = divmod(anchor, self.w)
        return {self._at(ar + dr, ac + dc) for dr, dc in offs}

    def _legal(self, offs, anchor) -> bool:
        """Can the body sit at ``anchor`` at all -- every member on the board
        and off a wall? Other pieces are ignored (the sokoban relaxation)."""
        ar, ac = divmod(anchor, self.w)
        for dr, dc in offs:
            cell = self._at(ar + dr, ac + dc)
            if cell < 0 or cell in self.wall:
                return False
        return True

    def _pushable(self, offs, anchor, direction) -> bool:
        """Could the player push this body from ``anchor`` along ``direction``,
        ignoring the other pieces?

        It needs somewhere to stand: a cell directly behind SOME member, on the
        board, off a wall, and not inside the body itself. For a domino lined
        up with the push that is the trailing member's back cell only; pushed
        broadside, either member's will do -- which is exactly the case the
        per-piece table cannot express."""
        dest = self.nbr[anchor][direction]
        if dest < 0 or not self._legal(offs, dest):
            return False
        foot = self._foot(offs, anchor)
        back = _OPPOSITE[direction]
        for cell in foot:
            stand = self.nbr[cell][back]
            if stand >= 0 and stand not in self.wall and stand not in foot:
                return True
        return False

    def _field(self, offs, goal) -> dict:
        """``{anchor: presses to bring the body from anchor onto goal}``.

        Reverse BFS over body pushes. A cell absent from every one of a body's
        fields is a placement it can never leave for a landing spot, so the
        state is lost -- see `_build_bodies`."""
        dist = {goal: 0}
        queue = deque([goal])
        while queue:
            cur = queue.popleft()
            for d in _DIRS:
                prev = self.nbr[cur][_OPPOSITE[d]]
                if prev < 0 or prev in dist:
                    continue
                if self._legal(offs, prev) and self._pushable(offs, prev, d):
                    dist[prev] = dist[cur] + 1
                    queue.append(prev)
        return dist

    # -- heuristic ------------------------------------------------------------
    def _body_cost(self, anchors) -> int:
        """``min over disjoint landing-spot combinations of sum over bodies of
        that body's push distance``. Admissible: one press moves one body one
        cell, so no press can take more than 1 off this sum."""
        got = self._hcache.get(anchors)
        if got is not None:
            return got
        best = _UNREACHABLE
        for combo in self._combos:
            total = 0
            for bi, goal in enumerate(combo):
                d = self._fields[bi][goal].get(anchors[bi])
                if d is None:
                    break
                total += d
                if total >= best:
                    break
            else:
                best = total
        self._hcache[anchors] = best
        return best

    def heuristic(self, state) -> int:
        """Estimated remaining PRIMITIVE moves. Admissible; 0 exactly at a win.

        `_body_cost` bounds the presses that move something; the walk term
        bounds the presses before the first of them -- the player has to stand
        beside a body to push it, so it owes at least ``manhattan - 1`` moves
        that shift nothing. Manhattan rather than the walk BFS, so it stays a
        lower bound on a board whose pieces the player may have to go round."""
        if self.bodies is None:
            self._build_bodies()
        player, pieces = state
        base = self._body_cost(tuple(pieces[idxs[0]]
                                     for idxs, _offs in self.bodies))
        if base >= _UNREACHABLE or base == 0:
            return base
        w = self.w
        pr, pc = divmod(player, w)
        near = min(abs(pr - r) + abs(pc - c)
                   for r, c in (divmod(cell, w) for cell in pieces))
        return base + max(0, near - 1)

    # -- search ---------------------------------------------------------------
    def astar(self, state, node_cap: int = 2_000_000) -> list | None:
        """Shortest primitive press sequence from ``state`` to a win, or None.

        A* over the macros, ``g`` in primitive moves, dedup on the exact
        ``(player, pieces)`` pair. The heuristic is admissible and every walk
        inside a macro is a shortest path, so at ``weight = 1`` the result is
        shortest over primitives (see `macros` for why the decomposition is
        lossless).

        A win is kept as an INCUMBENT and the search runs on until the
        frontier's ``f`` reaches its cost, rather than returning the first win
        it generates. That distinction is not pedantry: a macro is ``walk +
        push`` and macros have different lengths, so a node popped at ``g = 40``
        can generate a win with an 11-step macro (51) while a node popped at
        ``g = 45`` generates one with a 3-step macro (48). Returning on
        generation gave level 2 a plan that `--ties` then measured as not
        shortest -- the same flaw `PSPushExpert.exact_goal_test` exists to fix.
        """
        if self.won(state):
            return []
        counter = 0
        nodes = 0
        pq = [(self.heuristic(state), 0, counter, state, [])]
        best_g = {state: 0}
        best_plan: list | None = None
        best_cost = 1 << 30
        while pq:
            f, g, _c, cur, path = heapq.heappop(pq)
            if f >= best_cost:
                return best_plan               # proved: nothing shorter is left
            if best_g.get(cur, -1) != g:
                continue                       # superseded by a cheaper route
            for macro in self.macros(cur):
                nxt = cur
                for d in macro:
                    nxt = self.step(nxt, d)
                nodes += 1
                if self.won(nxt):
                    if g + len(macro) < best_cost:
                        best_cost, best_plan = g + len(macro), path + macro
                    continue
                if nxt == cur:
                    continue                   # a push the rules refused
                ng = g + len(macro)
                if ng >= best_cost or best_g.get(nxt, 1 << 30) <= ng:
                    continue
                h = self.heuristic(nxt)
                if h >= _UNREACHABLE:
                    continue                   # a piece is parked for good
                best_g[nxt] = ng
                counter += 1
                heapq.heappush(pq, (ng + h, ng, counter, nxt, path + macro))
            if nodes >= node_cap:
                return best_plan
        return best_plan

    def solve(self, state, node_cap: int = 2_000_000,
              label_cap: int = 4_000_000) -> "Plan | None":
        """The level's plan plus a MEASURED optimal set for each of its steps."""
        plan = self.astar(state, node_cap)
        if plan is None:
            return None
        return Plan(plan, self._label(state, plan, label_cap))

    def _label(self, start, plan, label_cap: int) -> list:
        """The exact optimal-action SET at every step of a proven-shortest
        ``plan``, from one BOUNDED layered BFS.

        The obvious oracle -- re-solve from each successor and keep every press
        that still finishes in the moves remaining -- is what ps:escape and
        ps:escaping_limbo use, and it costs one A* per candidate press. Level 1
        alone takes nine seconds per A*, so that is forty-odd searches for one
        level's labels. Instead, once weight-1 A* has PROVED ``d*``:

        1. sweep forward from the start over PRIMITIVE presses, dropping any
           state whose ``depth + h`` already exceeds ``d*`` (sound because
           ``h`` is admissible: such a state cannot be on a shortest path);
        2. sweep backward, marking a state GOOD when some press takes it to a
           good state one layer deeper, seeded with the wins in layer ``d*``.

        What survives is exactly the states on shortest paths, so the answer at
        each step is "every press that lands on a good state in the next
        layer" -- the same sets the re-solving oracle measures, for one search
        instead of dozens. (The ps:fractured_identity pattern.)

        Inferring the sets instead -- `PSPushExpert.annotate_walks`, which
        calls any step that moved a piece forced and only ties the WALKS --
        would be wrong here in both directions: two different first pushes
        frequently tie (this game's pieces are usually in the player's way, so
        which body you start with is free), and a walk that looks free can be
        forced by a piece that would block the alternative route.

        Falls back to labelling each step with the press the expert took if
        the bounded sweep would exceed ``label_cap`` states -- a singleton set
        is still a correct optimal target, just a poorer one, and no step ever
        ships unlabelled."""
        d_star = len(plan)
        dist = {start: 0}
        layers = [[start]]
        wins: set = set()
        for depth in range(d_star):
            nxt_layer = []
            for s in layers[depth]:
                for d in _DIRS:
                    t = self.step(s, d)
                    if t == s or t in dist:
                        continue              # a no-op, or already seen deeper
                    if self.won(t):
                        if depth + 1 < d_star:
                            # ``plan`` is not shortest after all -- only
                            # possible if `astar` stopped on its node cap
                            # without proving its incumbent. Every step below
                            # is then measured against the wrong bound, so
                            # label conservatively instead.
                            return [[d] for d in plan]
                        dist[t] = depth + 1
                        wins.add(t)
                        continue              # the game stops here
                    if depth + 1 + self.heuristic(t) > d_star:
                        continue
                    dist[t] = depth + 1
                    nxt_layer.append(t)
            if len(dist) > label_cap:
                return [[d] for d in plan]
            layers.append(nxt_layer)

        good = set(wins)
        for depth in range(d_star - 1, -1, -1):
            for s in layers[depth]:
                for d in _DIRS:
                    t = self.step(s, d)
                    if t != s and dist.get(t) == depth + 1 and t in good:
                        good.add(s)
                        break

        out = []
        cur = start
        for i, taken in enumerate(plan):
            best = []
            for d in _DIRS:
                t = self.step(cur, d)
                if t != cur and dist.get(t) == i + 1 and t in good:
                    best.append(d)
            # The recorded step is on a shortest path by construction, so a
            # set without it would mean the two sweeps disagree; fall back to
            # labelling what the expert actually did rather than shipping one.
            out.append(best if taken in best else [taken])
            cur = self.step(cur, taken)
        return out


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class FullCircleExpert(PSExpert):
    """`PSExpert`'s plan cache and snapshot discipline around a NATIVE search.

    `_search` is overridden to build a `_Board` off the interpreter grid and
    solve that; the interpreter is never stepped by the planner. The memo, the
    on-disk plan cache with its staleness check and the restore discipline all
    come from the base class unchanged, which is the pattern
    ps:crocodiles_love_cookies / ps:dotsnake / ps:escaping_limbo established
    for "the interpreter is too slow to search but is still the authority".

    `heuristic` is therefore never called and asserts rather than returning a
    number nothing would use.
    """

    #: `_key` is dynamic-objects-only, canonical only WITHIN a level.
    scope_by_level = True
    #: Keep each level's start plan (and its optimal sets) on disk.
    plan_cache_path = PLAN_CACHE
    #: ``noaction``: four directions and no ACTION button.
    directions = list(_DIRS)

    def setup(self) -> None:
        g = self.g
        self.player_ids = set(g.resolve_object_name("player"))
        self.wall_ids = {i for n in ("wall", "outside")
                         for i in g.resolve_object_name(n)}
        #: object id -> piece NAME (the non-win form), for both forms
        self.piece_ids: dict = {}
        for name, _sh, _co in _PIECES:
            for form in (name, name + _WIN_SUFFIX):
                for i in g.resolve_object_name(form):
                    self.piece_ids[i] = name
        self.dyn_ids = set(self.piece_ids) | self.player_ids
        self._boards: dict = {}

    def _key(self, eng) -> frozenset:
        # Pieces + player. Walls, targets and the decorative arrows are static
        # per level, so this is an exact state -- hence ``scope_by_level``.
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn)
        )

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "FullCircleExpert plans natively; heuristic is unused")

    def board(self, eng) -> _Board:
        """The `_Board` for the engine's current level, memoized on its STATIC
        objects -- walls, targets and the decorative arrows, i.e. everything no
        rule in this game creates, destroys or moves.

        Only the geometry is reused (the state to plan from is read
        separately), and the fields a board precomputes are worth keeping
        across the many `plan` calls one seed makes.

        Walls alone are NOT a level's identity here, which is worth stating
        because the obvious version of this method used them and was wrong:
        levels 1 and 6 are the same 8x8 room with the same eight wall cells and
        differ only in that level 6 has a second block of targets. Level 6 then
        got level 1's board, and the first thing that touched it -- reading
        the piece positions, whose ordering comes from the board's own piece
        list -- raised a KeyError. A quieter version of the same collision
        would have planned one level against another's targets."""
        dyn = self.dyn_ids
        key = frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in cell
            if o not in dyn
        )
        got = self._boards.get(key)
        if got is None:
            got = _Board.from_engine(eng, self.g)
            self._boards[key] = got
        return got

    def _search(self, eng) -> "Plan | None":
        board = self.board(eng)
        state = board.read_state(eng, self.piece_ids, self.player_ids)
        return board.solve(state, node_cap=self.node_cap)


class FullCircleSolver(PSAStarSolver):
    game_id = "puzzlescript_full_circle"
    game_name = GAME_NAME
    expert_cls = FullCircleExpert

    #: Record against the GAME FOLDER's adapter. ``games/ps:full_circle`` is a
    #: plain passthrough today; it is named anyway so a step cap or sprite
    #: patch added there later cannot silently make this generator tape a game
    #: nobody plays (the ps:count_mover trap).
    game_module_id = "ps:full_circle"

    #: Runaway guard on the native search, not a tuning dial -- the hardest
    #: level expands orders of magnitude fewer nodes than this.
    node_cap = 2_000_000
    #: Room for the longest plan plus the RESET exploration prefix and the
    #: re-plan after it. Stays under the adapter's own 200-step per-level
    #: budget, which would otherwise flip a level to GAME_OVER mid-plan.
    max_steps = 150

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time cost visible as startup
        rather than as a mysteriously slow first seed, and it fills the disk
        cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _report() -> int:
    """Per-level board size, piece inventory, plan length and tie coverage."""
    import time
    game = FullCircleSolver().make_game(0)
    expert = FullCircleExpert(game, node_cap=FullCircleSolver.node_cap)
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = _Board.from_engine(eng, game._game)
        colours = {}
        for co in board.colour:
            colours[co] = colours.get(co, 0) + 1
        pairs = sum(1 for n in colours.values() if n > 1)
        t0 = time.time()
        found = expert.plan(eng, level)
        dt = time.time() - t0
        if found is None:
            print(f"level {level}: NO PLAN")
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(found)
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"level {level}: {eng.height:2d}x{eng.width:2d} "
              f"{len(board.kinds)} pieces ({pairs} rigid pair"
              f"{'' if pairs == 1 else 's'}), {len(board.targets)} targets, "
              f"{len(found):3d} moves (budget {game._max_steps}, {room}), "
              f"{ties:3d} steps with a tie set "
              f"({ties / max(1, len(found)):3.0%}), {dt:6.2f}s")
    print(f"total {total} moves over {game.n_levels} levels")
    return 0


def _ties(levels: "list[int] | None" = None) -> int:
    """Re-derive every optimal-action set the SLOW way and compare.

    `_Board._label` gets the sets out of one bounded layered BFS, which is a
    different algorithm from the A* that produced the plan; this re-measures
    them the way ps:escape and ps:escaping_limbo do -- re-solve from each
    successor, keep every press that still finishes in the moves remaining --
    so a bug in either the sweep or its bound shows up as a disagreement
    rather than as quietly wrong training labels. It also re-proves each plan
    shortest, since a re-solve from the start has to return the plan's own
    length.

    Minutes per level on the big boards (one A* per candidate press, and level
    1's A* is nine seconds), which is why it is a command and not part of
    ``--plans``. Pass level numbers to check a subset::

        python solvers/generate_full_circle_training.py --ties 0 2 3 4
    """
    import time
    game = FullCircleSolver().make_game(0)
    expert = FullCircleExpert(game, node_cap=FullCircleSolver.node_cap)
    bad = 0
    for level in (levels if levels else range(game.n_levels)):
        game.set_level(level)
        eng = game._engine
        board = _Board.from_engine(eng, game._game)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level}: NO PLAN")
            continue
        sets = getattr(plan, "optsets", None)
        if sets is None:
            print(f"level {level}: UNLABELLED")
            bad += 1
            continue
        t0 = time.time()
        exact: dict = {}

        def remaining(st) -> "int | None":
            if st not in exact:
                sol = board.astar(st, FullCircleSolver.node_cap)
                exact[st] = None if sol is None else len(sol)
            return exact[st]

        wrong = 0
        cur = board.start
        if remaining(cur) != len(plan):
            print(f"level {level}: plan is {len(plan)} but the shortest is "
                  f"{remaining(cur)}")
            bad += 1
        for i, taken in enumerate(plan):
            left = len(plan) - i
            want = []
            for d in _DIRS:
                nxt = board.step(cur, d)
                if board.won(nxt):
                    cost = 1
                elif nxt == cur or board.heuristic(nxt) > left - 1:
                    cost = None            # sound prunes, as in ps:escape
                else:
                    sub = remaining(nxt)
                    cost = None if sub is None else 1 + sub
                if cost == left:
                    want.append(d)
            if sorted(want) != sorted(sets[i]):
                wrong += 1
                print(f"  level {level} step {i}: labelled {sorted(sets[i])}, "
                      f"measured {sorted(want)}")
            cur = board.step(cur, taken)
        bad += wrong
        print(f"level {level}: {len(plan):3d} steps -- "
              f"{'OK' if not wrong else str(wrong) + ' WRONG'} "
              f"({time.time() - t0:6.1f}s)")
    print("labels verified" if not bad else f"TIES FAILED: {bad} disagreements")
    return 0 if not bad else 1


def _bfs(levels: "list[int] | None" = None, cap: int = 4_000_000) -> int:
    """Re-derive each plan's LENGTH by exhaustive primitive BFS and compare.

    The one check that shares no assumption with the search: no heuristic, no
    macros, no incumbent bound -- just every press from every state until a win
    appears. If `_Board.heuristic` ever stops being admissible, or the macro
    decomposition ever stops being lossless, this is what says so; ``--ties``
    would not, because it re-solves with the same `astar`.

    Only the small boards fit (level 1 passes 9M states without finishing), so
    it covers levels 0, 2, 3 and 4 and skips the rest at ``cap`` states::

        python solvers/generate_full_circle_training.py --bfs
    """
    import time
    game = FullCircleSolver().make_game(0)
    expert = FullCircleExpert(game, node_cap=FullCircleSolver.node_cap)
    bad = 0
    for level in (levels if levels else range(game.n_levels)):
        game.set_level(level)
        board = _Board.from_engine(game._engine, game._game)
        plan = expert.plan(game._engine, level)
        t0 = time.time()
        seen = {board.start}
        frontier = [board.start]
        depth, found = 0, None
        while frontier and found is None and len(seen) <= cap:
            depth += 1
            nxt = []
            for s in frontier:
                for d in _DIRS:
                    t = board.step(s, d)
                    if t == s or t in seen:
                        continue
                    if board.won(t):
                        found = depth
                        break
                    seen.add(t)
                    nxt.append(t)
                if found is not None:
                    break
            frontier = nxt
        dt = time.time() - t0
        if found is None:
            print(f"level {level}: too big to settle "
                  f"({len(seen)} states, {dt:.1f}s) -- SKIPPED")
            continue
        got = None if plan is None else len(plan)
        ok = got == found
        bad += not ok
        print(f"level {level}: BFS says {found}, plan is {got} -- "
              f"{'OK' if ok else 'NOT SHORTEST'} "
              f"({len(seen)} states, {dt:.1f}s)")
    print("plans are shortest" if not bad
          else f"BFS FAILED: {bad} plans are not shortest")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# Differential fuzz: the model against the interpreter
# ---------------------------------------------------------------------------

#: Cell legend for `_SCENARIOS`. Upper case is a STACK -- a piece standing on a
#: target -- which the shipped levels author only for pieces that start won.
_SCEN_LEGEND = {
    ".": (), "#": ("wall_v",), "%": ("wall_h",), "/": ("outside",),
    "p": ("player",),
    "q": ("round_tl",), "Q": ("round_tl2",),
    "w": ("round_tr",), "W": ("round_tr2",),
    "e": ("round_bl",), "E": ("round_bl2",),
    "r": ("round_br",), "R": ("round_br2",),
    "a": ("target_tl",), "b": ("target_tr",),
    "c": ("target_bl",), "d": ("target_br",),
    "A": ("target_tl", "round_tl_win"), "B": ("target_tr", "round_tr_win"),
    "C": ("target_bl", "round_bl_win"), "D": ("target_br", "round_br_win"),
    "x": ("target_br", "round_tl"),          # a piece on the WRONG target
}

#: Hand-built boards for the turns random play does not reach, and for the ones
#: it reaches so rarely that a passing run would prove nothing. Each is
#: ``(name, rows, presses)`` and is checked against the interpreter press by
#: press exactly like the random runs. They are the model's unit tests, written
#: as boards because that is what they are about.
_SCENARIOS = (
    # A push is a push, and a blocked one leaves the board untouched.
    ("single piece", ["..pq.."], "right right"),
    ("single into a wall", ["..pq#"], "right"),
    ("single against the edge", ["...pq"], "right"),
    ("single into another colour", ["..pqw."], "right"),
    # A domino of ONE colour is rigid in every direction, not just its own
    # axis: `[ Moving Round_T | Round_T ]` carries no direction prefix.
    ("domino end-on", [".pqW.."], "right right"),
    ("domino broadside", [".pq..", "..W.."], "right right"),
    ("domino broadside, other side", ["..W..", ".pq.."], "right"),
    # ...and rigid means the PLAYER does not move either when it is refused.
    ("domino end-on blocked", [".pqW#"], "right"),
    ("domino broadside blocked", [".pq..", "..W#."], "right"),
    ("domino broadside blocked above", ["..W#.", ".pq.."], "right"),
    # Two pieces of the same colour that are only DIAGONAL are not a blob.
    ("diagonal is not a blob", [".pq..", "...W#"], "right"),
    # ...nor are two of the same colour with a gap between them. The second
    # press is the point: the first one closes the gap, and from there they
    # ARE one body -- which is also the only way a level could ever start with
    # a colour's two pieces apart and end up with them welded (none does; see
    # `_build_bodies`).
    ("gap is not a blob, until it is", [".pq.W."], "right right"),
    # Different colours never couple, however they are stacked.
    ("blue beside green is two bodies", [".pq..", "..Q#."], "right"),
    # The win flag follows POSITION: onto a matching target and back off it.
    ("onto its target", [".pq.a."], "right right"),
    ("off its target again", [".pqa.."], "right right right"),
    ("onto the wrong target", [".pq.d."], "right right"),
    ("a won piece is still pushable", [".pA..."], "right"),
    # A full 2x2 block: the last push is the win, and it must be seen as one.
    ("closing the circle", ["pq.ab", "..CD."], "right right"),
)


def _load_scenario(game, rows):
    """Load an ASCII scenario board onto the interpreter and return
    ``(board, state, engine)`` ready to step. The engine's grid is replaced
    wholesale, the way `_audit` does it."""
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    h, w = len(rows), max(len(r) for r in rows)
    eng.height, eng.width = h, w
    eng.grid = [[{idx["background"]}
                 | {idx[o] for o in _SCEN_LEGEND[(row + "." * w)[c]]}
                 for c in range(w)] for row in rows]
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()
    board = _Board.from_engine(eng, g)
    return board, board.start, eng


class _Referee:
    """Steps `_Board` and the interpreter side by side and compares the grid.

    Everything visible is compared: where every piece is AND which of its two
    forms it is wearing, so the model's "the ``_Win`` flag is a function of
    position" claim is under test on every single press, not just the win
    predicate at the end. It also counts what each run actually exercised,
    because a fuzz that never pushes a domino proves nothing about the rule
    this game is built on."""

    def __init__(self, game):
        g = game._game
        self.player_ids = set(g.resolve_object_name("player"))
        #: object id -> (piece name, is the ``_Win`` form)
        self.forms: dict = {}
        for name, _sh, _co in _PIECES:
            for i in g.resolve_object_name(name):
                self.forms[i] = (name, False)
            for i in g.resolve_object_name(name + _WIN_SUFFIX):
                self.forms[i] = (name, True)
        self.cover = dict.fromkeys(
            ("walk", "push", "domino", "refused", "won", "unwon"), 0)

    def press(self, eng, board, state, d):
        """Take ``d`` on both. Returns ``(new state, complaint or None)``."""
        before = state
        front = board.nbr[before[0]][d]
        aimed_at_piece = front >= 0 and front in set(before[1])
        won_before = sum(1 for i, cell in enumerate(before[1])
                         if board.targets.get(cell) == board.shape[i])

        eng.step(d)
        state = board.step(state, d)

        # How many pieces moved answers "was this a domino?" directly, without
        # re-deriving the blob the model just computed.
        moved = sum(1 for a, b in zip(before[1], state[1]) if a != b)
        if state == before:
            self.cover["refused"] += aimed_at_piece
        elif not moved:
            self.cover["walk"] += 1
        else:
            self.cover["push"] += 1
            self.cover["domino"] += moved > 1
        won_after = sum(1 for i, cell in enumerate(state[1])
                        if board.targets.get(cell) == board.shape[i])
        self.cover["won"] += won_after > won_before
        self.cover["unwon"] += won_after < won_before

        order = {name: i for i, (name, _s, _c) in enumerate(board.kinds)}
        got_p, got = -1, [-1] * len(board.kinds)
        got_win = [False] * len(board.kinds)
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                pos = r * board.w + c
                if cell & self.player_ids:
                    got_p = pos
                for oid in cell:
                    form = self.forms.get(oid)
                    if form is not None:
                        got[order[form[0]]] = pos
                        got_win[order[form[0]]] = form[1]
        want_win = [board.targets.get(cell) == board.shape[i]
                    for i, cell in enumerate(state[1])]
        if (got_p, tuple(got)) != state:
            return state, (f"after {d!r}: engine (player={got_p}, "
                           f"pieces={tuple(got)}) vs model (player={state[0]},"
                           f" pieces={state[1]})")
        if got_win != want_win:
            return state, (f"after {d!r}: engine win forms {got_win} vs model "
                           f"{want_win}")
        if eng.check_win() != board.won(state):
            return state, f"after {d!r}: win disagreement"
        return state, None

    def report(self) -> str:
        return "[" + " ".join(f"{k} {v}" for k, v in self.cover.items()) + "]"


def _fuzz(episodes: int = 40, steps: int = 60, seed: int = 0) -> int:
    """Assert `_Board` reproduces the interpreter, cell for cell.

    This is what licenses planning natively at all: the model may only stand in
    for the interpreter for as long as it agrees with it. Two suites:

    * **Every level**, from its start and from every PREFIX of its own
      solution, then random presses. The prefixes matter because the interesting
      states -- pieces parked on targets, dominoes lined up against a target
      block -- are ones random play from the start almost never builds.
    * **`_SCENARIOS`**, hand-built boards for the turns even that does not
      reach reliably: every way a rigid push is refused, a domino pushed
      broadside, and a piece won and then un-won.

    The coverage counts are printed for the same reason the scenarios exist: a
    run whose ``domino`` or ``unwon`` column is zero has not tested those rules,
    whatever its verdict says."""
    import random as _random

    game = FullCircleSolver().make_game(0)
    expert = FullCircleExpert(game, node_cap=FullCircleSolver.node_cap)
    rng = _random.Random(seed)
    bad = 0

    ref = _Referee(game)
    for level in range(game.n_levels):
        game.set_level(level)
        board = _Board.from_engine(game._engine, game._game)
        plan = expert.plan(game._engine, level) or []
        starts = list(range(len(plan) + 1))   # the start, and every prefix
        mismatches = 0
        for ep in range(episodes):
            game.set_level(level)
            eng = game._engine
            state = board.start
            presses = (list(plan[:starts[ep % len(starts)]])
                       + [rng.choice(_DIRS) for _ in range(steps)])
            for d in presses:
                state, complaint = ref.press(eng, board, state, d)
                if complaint:
                    mismatches += 1
                    print(f"  level {level}: MISMATCH {complaint}")
                    break
                if board.won(state):
                    break                     # the engine stops accepting input
        bad += mismatches
        print(f"level {level}: {episodes} runs from {len(starts)} start states"
              f" -- {'OK' if not mismatches else str(mismatches) + ' MISMATCH'}")
    print(f"   levels coverage {ref.report()}")

    scen = _Referee(game)
    for name, rows, presses in _SCENARIOS:
        board, state, eng = _load_scenario(game, rows)
        complaint = None
        for d in presses.split():
            state, complaint = scen.press(eng, board, state, d)
            if complaint:
                break
            if board.won(state):
                break
        bad += complaint is not None
        print(f"scenario {name:32s} -- "
              f"{'OK' if not complaint else 'MISMATCH ' + complaint}")
    print(f"   scenario coverage {scen.report()}")
    missing = [k for k, v in scen.cover.items() if not v and not ref.cover[k]]
    if missing:
        print(f"WARNING: never exercised: {', '.join(missing)}")
    print("model matches the interpreter" if not bad
          else f"FUZZ FAILED: {bad} mismatching runs")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# Render audit
# ---------------------------------------------------------------------------

def _audit() -> int:
    """Assert every cell COMPOSITION a frame can show is distinct, at every
    cell size the boards use.

    Composition, not object: what an agent has to read off a cell here is a
    *stack* -- which quarter-circle, in which colour, standing on which target
    (or on none) -- and this game's whole visual language is transparency, so
    the failure mode to look for is a piece whose opaque body hides the target
    it is standing on. The eight ``_Win`` forms are audited only over their own
    matching target because that is the only place they can ever be seen (a
    piece stepping off its target is adjacent to it and reverts on the same
    turn), and the eight base forms only over the three targets that are NOT
    theirs, for the same reason.

    Whole 64x64 frames of uniform boards are compared rather than one cell out
    of a mixed board: `_render_frame` centre-pads a non-square board, so
    indexing a cell by ``cell_px * r`` silently reads the wrong pixels on the
    wide levels. Two uniform boards render identically iff their cells do."""
    import itertools

    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    game = FullCircleSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    comps: dict = {
        "floor": (), "wall_v": ("wall_v",), "wall_h": ("wall_h",),
        "outside": ("outside",), "arrow_r": ("arrow_r",),
        "arrow_d": ("arrow_d",), "player": ("player",),
        "player_on_arrow_r": ("arrow_r", "player"),
    }
    for tname, tshape in _TARGETS.items():
        comps[tname] = (tname,)
        comps[f"player_on_{tname}"] = (tname, "player")
    for name, shape, _co in _PIECES:
        comps[name] = (name,)
        comps[f"{name}_on_arrow_r"] = ("arrow_r", name)
        for tname, tshape in _TARGETS.items():
            if tshape == shape:
                comps[f"{name}{_WIN_SUFFIX}_on_{tname}"] = (
                    tname, name + _WIN_SUFFIX)
            else:
                comps[f"{name}_on_{tname}"] = (tname, name)

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for name, objs in comps.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{len(comps)} compositions -- "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    if "--ties" in sys.argv:
        sys.exit(_ties([int(a) for a in sys.argv[sys.argv.index("--ties") + 1:]
                        if a.isdigit()]))
    if "--bfs" in sys.argv:
        sys.exit(_bfs([int(a) for a in sys.argv[sys.argv.index("--bfs") + 1:]
                       if a.isdigit()]))
    sys.exit(FullCircleSolver.main())
