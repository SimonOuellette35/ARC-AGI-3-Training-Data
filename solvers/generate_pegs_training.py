"""Generate Phase-1 training data for the PuzzleScript game ps:pegs
("Pegs", the TI-83+ PuzzPack puzzle ported to PuzzleScript).

The harness -- the engine-blackbox A* expert, the rotation contract, the
trajectory recorder and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other search-the-interpreter ps:
generators. This file is only the game-specific part: the macro set its
replacement-selection mode needs, the elimination heuristic that guides the
search, and two edits to the shipped .txt that the mandatory augmentation
forced.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_pegs",
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
Every expert step also carries the full set of equally-optimal presses.

The game
--------
You push pegs into each other and the board has to end EMPTY (``No Block``).
Which pegs you shove together is the whole puzzle, because the four types do
four different things when they meet themselves:

    circle  + circle    -> both vanish
    square  + square    -> both vanish
    triangle + triangle -> one CreatedSolidBlock, i.e. a new wall
    cross   + cross     -> one vanishes, the other becomes a SelectedBlock and
                           the player enters SELECTION MODE

Two pegs of DIFFERENT types refuse to combine and the push is cancelled, so a
peg is never moved by anything but a deliberate matching shove. A hole is the
other way out: a square pushed into one FILLS it (the hole becomes floor), any
other peg falls in and is gone, and the hole survives -- so one reachable hole
can swallow every circle, triangle and cross on the board. The player falls in
too, which ends the level.

SELECTION MODE is the mechanic that makes the game more than a sokoban. While a
SelectedBlock is on the board the player cannot move at all: an arrow advances
the selection one step round ``triangle -> cross -> square -> circle ->`` and
ACTION5 stamps the current choice down as a real peg. So a pair of crosses is a
WILDCARD -- it is how level 1 gets the second square its lone square needs, and
how level 3 gets the second triangle. That is also why the search cannot be a
plain `PSSokobanExpert`: there is no target to match pieces to, the piece
INVENTORY changes mid-level, and one of the moves is not a push.

The levels
----------
The shipped file holds 5. They cover the four combine rules, but they leave two
thirds of the game unexercised: only ONE has a hole, and the replacement is only
ever asked for a Triangle (the default, no arrows) or a Square. A corpus built
from those five teaches "press ACTION the moment the selection appears" and
almost nothing about the hole, so the file now carries 9, four of them authored
here and each winnable only the intended way:

    level  pegs                 plan push ties  search  what it is
     0     2o 1q 1< 2+, 1 hole    17   8  12%    8.0s   SHIPPED -- every rule
                                                        once, in one board
     1     2o 1q 4< 2+            32  16  22%   42.3s   SHIPPED -- the wildcard
                                                        IS the second square
     2     2o 6<                  25   6  16%   11.0s   SHIPPED -- a plus-shaped
                                                        maze of triangles
     3     2o 2q 1< 2+            37   9   5%   13.1s   SHIPPED -- corridors,
                                                        wildcard is a triangle
     4     4<                     52  15   0%   39.5s   SHIPPED -- four triangles
                                                        down a two-room shaft
     5     1o 1<, 1 hole          16   7   6%    0.8s   the hole eats whatever is
                                                        not a square
     6     1o 2+                  20  13  20%    0.6s   the crosses have to
                                                        become a CIRCLE (3
                                                        arrows): a lone triangle
                                                        has no out
     7     1o 3+                  25  15  24%    3.4s   three crosses, so the
                                                        first replacement is
                                                        another CROSS and gets
                                                        forged again
     8     2o 1q, 1 hole          14   6   0%    0.1s   the square is a BRIDGE --
                                                        filling the hole is the
                                                        only way across

``push`` counts the presses that move a peg (the rest is walking and the
selection presses), ``ties`` the share of steps with more than one optimal
press. 238 moves over 9 levels, ~119s of one-time search written to
``data/pegs_plans.json`` so `parallelize_generator`'s shards do not each
re-derive it. Delete the file to re-derive. Every board is 12x8, so they all
render at the same cell size.

Expert solver
-------------
`PegsExpert` -- `PSPushExpert`'s walk-and-push macro A* over the real
interpreter, with three game-specific pieces:

* **The selection macros.** While a SelectedBlock is up, the walk-and-push
  enumeration is meaningless (the player is frozen) and actively harmful: three
  of its four first presses CYCLE THE SELECTION and then fail, so the search
  would queue a garbage state per macro. `_analyze` returns the four real
  choices instead -- ``n arrows then ACTION``, n in 0..3 -- which is also the
  whole of the mode's branching.

* **The heuristic is an ELIMINATION SCHEDULE, not a distance.** Nothing has to
  end up anywhere here, so there is no target to measure to; what is left to do
  is pair the surviving pegs off. `_elim` costs the cheapest ORDER in which
  that could happen if pegs passed through each other -- "walk to a peg, then
  either shove it into one it matches, drop it down a hole, or forge a
  replacement out of two crosses" -- over shortest paths through the level's
  permanent walls, not over manhattan distance (on the maze levels that is the
  difference between a gradient and a guess). Ordering the jobs is what lets
  the WALK between them be charged, and the walking is most of a plan: level 0
  is 8 pushes out of 17 presses, so a push-only estimate is short by nine and
  A* has nothing to steer with. `_push_field` adds the other half of it -- a
  peg only moves with the player directly behind it, so every TURN in a peg's
  route costs a trip round it, and level 1 went from not finishing in 25
  minutes to 42 seconds when that was charged.

  It is also the deadlock test, for free: when no schedule exists at all -- a
  lone circle with no hole and no crosses left to forge a partner -- the state
  is provably unwinnable and `dead` drops it un-expanded. That is not a rare
  corner, it is what a wrong shove does.

* **Exact dedup.** `PSPushExpert` merges states by the player's REACHABLE
  REGION, which is a documented mis-cost of up to the region's diameter and
  cost ps:entrepotphage_demake real optimality. `_region_key` here is the full
  canonical grid, packed one byte per cell.

``weight = 1`` and ``exact_goal_test`` is on. Drop `_elim`'s walk terms and
what is left is a pure elimination matching that is admissible outright, which
makes the plan shortest by construction; the walk terms are a sharpening whose
ordering argument assumes a peg is dealt with in one stretch, so they are a
search GUIDE and ``--optimal`` is the certificate -- it re-solves every level
under the matching-only bound and requires the same length. All nine levels
come back identical, so every shipped plan is proved shortest. (An earlier
version
of `_elim` was quietly inadmissible for a different reason -- it could not
express "the wildcard the crosses forge is the partner the lone triangle needs"
-- and level 0 came back as 19 moves, reported as proved, where 17 exists.)

Optimal-action sets
-------------------
Exact, from one bounded layered BFS per level (`_label`), not from
`PSPushExpert`'s per-press re-solve. Once the macro A* has proved ``d*``, a
layered forward sweep over PRIMITIVE presses keeps only what ``g + h <= d*``
allows -- i.e. exactly the states a shortest path can contain -- and a backward
pass over the edges it kept gives every one of them an exact distance to the
win. The optimal set at each step is then just the presses that go one closer.
On level 0 that is ~7s against the 135s `exact_optsets` spent re-solving, and
it is not subject to the re-solve's own blind spot (a candidate its heuristic
prunes is a candidate it never measures).

`PSPushExpert.annotate_walks` could not do this job at all. A selection press
changes the board, so it would be labelled forced -- but all FOUR arrows
advance the selection identically, so all four are optimal and labelling one of
them is labelling the augmentation instead of the game. And a peg shoved out of
the way to reach another one is frequently just how the player walks. No step
ever ships unlabelled (the always-emit-optimal-targets rule).

The two .txt edits
------------------
Both are forced by the mandatory presentation augmentation, and both are
documented at the point of change in ``data/puzzlescript_games/Pegs.txt``.

* **The selection cycle is now direction-free.** It used to run forward on DOWN
  and backward on UP (with left/right dead), i.e. the mechanic asked which
  button is "down" -- a fact about the board's hidden rotation, not about the
  picture, and the picture is all a policy has. Any arrow now advances it by
  one. No shipped level's optimal length moves: they need Triangle (0 presses,
  the default) and Square (2 presses, which cost 2 either way under the old
  rules as well).

* **The pegs are recoloured.** The original HUSL pastels all collapse onto the
  same two ARC palette indices -- outline 4, fill 10 -- so circle, square and
  triangle rendered in identical colours and circle vs square differed in FOUR
  corner pixels out of 25. Peg type is the entire puzzle, so each type now owns
  an index: circle yellow, square blue, triangle green, cross red, and the
  selected variants swap their outline to the purple the selecting player
  wears. ``--audit`` is the regression test, and it is a WHOLE-FRAME audit: the
  per-cell crop that the older generators in this tree use is wrong for any
  board whose render is upscaled, which this one is (60x40 -> 64x42).

  The same edit also gave InvisibleDeadPlayer a sprite. It shipped as five rows
  of nothing, so a player who had fallen down a hole rendered as bare floor --
  a state the frame simply could not show, and one the RESET exploration prefix
  walks into on its own.

Augmentation
------------
Pegs' engine state after reset is identical for every seed (the levels are
fixed ASCII maps), so the only per-(seed, level) variable is the presentation:
the frame rotation (rotation_k in {0,1,2,3}) with the matching directional
action remap. The expert plan is therefore seed-independent -- solved once per
level, cached, and replayed per seed with that seed's remapped screen actions.

Pegs is also in `PuzzleScriptAdapter._FLIP_GAMES`, so the presentation group is
the board's full 8 elements rather than 4 -- which on a five-level game is the
difference between 20 trajectories and 40. ``--symmetry`` is the evidence:
every level is random-walked and each ``(board, press)`` it visits is replayed
INDEPENDENTLY on all eight turned and mirrored copies of that same board, which
is the version of the check that caught ps:lovendpieces where a plan replay did
not. 25200 transitions over the nine levels, all exactly symmetric. The one
thing worth arguing rather than measuring is the TRIANGLE sprite, which is
chiral: it is Hungry Kitty's answer -- the shape encodes no facing, the game has
one triangle object and no direction states, so a mirrored triangle is a
triangle -- and since the recolor, peg type is read off the colour anyway.

Usage (run from the repo root):
    python solvers/generate_pegs_training.py --episodes 200 \
        --out data/training_multi_level/pegs

    python solvers/generate_pegs_training.py --plans     # level report
    python solvers/generate_pegs_training.py --optimal   # optimality proof
    python solvers/generate_pegs_training.py --verify    # replay every plan
    python solvers/generate_pegs_training.py --symmetry  # 8-fold mechanic check
    python solvers/generate_pegs_training.py --audit     # rendering audit
"""

from __future__ import annotations

import sys
import heapq
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (PSAStarSolver, PSPushExpert,   # noqa: E402
                                     Plan, _DELTA, restore, snapshot)

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Pegs"

#: Disk cache of each level's start plan and its optimal-action sets. The
#: searches are seed-independent, so without a file on disk every shard
#: `parallelize_generator` starts would re-derive all of them. Delete to
#: re-derive.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "pegs_plans.json"

#: Heuristic charge for a state the elimination relaxation cannot empty at all
#: -- a peg with no partner, no hole and no crosses left to forge one. Large
#: enough to sink the node, finite so the search stays complete. `dead` drops
#: these before they are ever queued; the constant is what makes the same fact
#: visible to `_label`, whose forward sweep asks the heuristic directly.
UNWINNABLE = 10_000

#: `_label`'s sentinel for "this press wins outright", distinct from any state.
_WIN = object()


def cycles(peg_type: int, wanted: int) -> int:
    """Arrow presses a WILD peg needs before it can be ``wanted``.

    A wildcard is a selection still sitting on Triangle (that is where a cross
    pair always leaves it), so stamping it as anything else costs its distance
    round the ring. Concrete pegs cost nothing, and two wildcards cost nothing
    because both can be stamped Triangle and walled together. Real presses that
    nothing else in `_elim` counts -- level 1's whole point is that its lone
    square needs a wildcard turned into a SQUARE, which is two arrows."""
    if peg_type != WILD or wanted == WILD:
        return 0
    return (wanted - TRIANGLE) % len(CYCLE)

#: The selection ring, in the order one arrow press advances it. The index into
#: this tuple IS a peg's type id everywhere below.
CYCLE = ("triangle", "cross", "square", "circle")
TRIANGLE, CROSS, SQUARE, CIRCLE = 0, 1, 2, 3
#: A cross pair leaves a peg whose type the player has not chosen yet. In the
#: relaxation it pairs with anything, which is exactly what it can become.
WILD = 4


class PegsExpert(PSPushExpert):
    """Walk-and-push macro A* over the real interpreter, guided by how cheaply
    the remaining pegs could be paired off.

    WHY NOT `PSSokobanExpert`. Pegs has no targets. Its win condition is ``No
    Block``: pieces do not have to arrive anywhere, they have to STOP EXISTING,
    and which pairs you make is the puzzle. So the sokoban machinery -- per-target
    push-distance tables, a greedy piece-to-target matching, the corner-deadlock
    test that falls out of them -- has nothing to measure. Worse, its deadlock
    rule would be backwards: a peg shoved into a corner is fine here as long as
    something can still be pushed INTO it.

    THE MACRO SET is `PSPushExpert`'s ``walk to the push cell, then push``,
    except while a SelectedBlock is on the board. In that mode the player is
    frozen (``[ > SelectingPlayer ] -> [ SelectingPlayer ]`` eats the force), so
    every walk-and-push macro fails -- but three of the four fail AFTER their
    first press has cycled the selection, so leaving them in would queue one
    junk state per macro and let the search wander round the ring. `_analyze`
    returns the four genuine choices instead: n arrows then ACTION, n in 0..3.

    THE HEURISTIC (`_elim`) is the cost of the cheapest scheme that empties the
    board if pegs could pass through each other and through the player:

        pair two pegs of the same type      d(a, b)
        drop one peg down a hole            min over holes of d(a, hole)
        combine two crosses                 d(a, b) + 1 + <cost of the wildcard>

    where ``d`` is the shortest path through the level's PERMANENT walls (all
    pairs, one BFS per cell, computed once per board). Distances ignore the
    other pegs and any CreatedSolidBlock a triangle pair has since built, which
    only ever makes the estimate smaller.

    The ``+ 1`` on a cross pair is the ACTION press the mode forces; the
    wildcard's own cost is whatever it costs to eliminate the peg the pair
    leaves behind, so the estimate does not forget that a replacement still has
    to be dealt with.

    ADMISSIBILITY, and the honest bit. Drop the walk terms and what is left is
    a pure elimination MATCHING (``walk_bound = False``), which is admissible
    outright: a peg's own pushes can never total less than the distance its
    scheme sends it, and the schemes cover everything the game can do. Adding
    the walks makes the estimate far sharper -- most of a plan is walking, and
    without them A* has nothing to steer with -- but the ordering argument
    behind them assumes each peg is dealt with in one stretch, and a plan that
    shoved a peg halfway, went elsewhere and came back could in principle be
    charged for a walk it does not make. So the walk terms are a search GUIDE,
    not a certificate, and ``--optimal`` is the certificate: it re-solves every
    level under the matching-only heuristic, where ``weight = 1`` +
    ``exact_goal_test`` + exact dedup does prove shortest, and requires the same
    length. (This is the same discipline `PSSokobanExpert` documents for its own
    greedy matching.)

    AND IT IS THE DEADLOCK TEST. When no scheme exists -- an odd circle with no
    hole to drop it down and no crosses left to forge a partner from -- the
    state can never reach ``No Block`` by any sequence, because the relaxation
    already allows strictly more than the game does. `dead` drops those
    un-expanded. On level 1, where the single square is only winnable through
    the crosses, that is most of the search space.

    An earlier version of `_elim` walked the pegs in a fixed sorted order and
    only ever considered the FIRST one next. That is complete for a plain
    matching and is NOT complete here, because a cross pair mints a new peg: by
    the time the recursion reached the crosses, the lone triangle they were
    supposed to partner had already been priced against a hole on the other side
    of the board. The estimate came out too high, which is inadmissible, and it
    cost level 0 two moves (19 against 17) with the search reporting the 19 as
    proved. Hence the loop over every remaining peg.

    EXACT DEDUP. `_region_key` is the full canonical grid rather than
    `PSPushExpert`'s ``(pieces, player region)``. The region merge is a
    documented mis-cost of up to the region's diameter and it cost
    ps:entrepotphage_demake a proved-shortest plan; here the state is small and
    the exact key is what the plan memo uses anyway.

    OPTIMAL SETS come from `_label`, not from `PSPushExpert.optimal_sets`. See
    that method for why re-solving per candidate press is the wrong shape here.
    """

    pushable_names = ("block",)
    #: A hole is a blocker for the PLAYER -- walking into one is death -- while
    #: still being a perfectly good destination for a push, which is exactly what
    #: `_analyze` does with it (a push's landing cell is never tested).
    blocker_names = ("solidblock", "hole")

    #: ACTION5 is a real move in this game (it stamps the selection down), so it
    #: has to be back in the action space `_label` sweeps over. `_astar`
    #: branches on macros and is unaffected.
    directions = ["up", "down", "left", "right", "action"]

    #: A macro is ``walk + act``, so macros cost different amounts and the first
    #: win GENERATED is not the shortest one. Run to the bound instead.
    exact_goal_test = True

    #: Charge the player's walk between jobs in `_elim`. Off is the provably
    #: admissible matching-only bound `--optimal` certifies against; see the
    #: class docstring.
    walk_bound: bool = True

    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        super().setup()
        g = self.g
        self.bg_id = g.obj_name_to_idx["background"]
        #: Hole boundaries are drawn by the game's own rules from the hole set,
        #: so they are a deterministic function of the rest of the grid and
        #: carrying them in the state key would only make it bigger.
        self.boundary_ids = set(g.resolve_object_name("holeboundary"))
        self.hole_ids = set(g.resolve_object_name("hole"))
        self.selected_ids = set(g.resolve_object_name("selectedblock"))
        #: The permanent walls -- the level's own NormalSolidBlock. A
        #: CreatedSolidBlock is deliberately NOT in here: it appears mid-level,
        #: and a distance field that ignored it stays a lower bound while a
        #: field that included it would have to be rebuilt per node.
        self.static_wall_ids = set(g.resolve_object_name("normalsolidblock"))
        self.type_of: dict[int, int] = {}
        for t, name in enumerate(CYCLE):
            self.type_of[g.obj_name_to_idx[name]] = t
            self.type_of[g.obj_name_to_idx["selected" + name]] = t
        self._dist: dict = {}          # per-board all-pairs walk distances
        self._push: dict = {}          # per-board all-pairs PUSH costs
        self._elim_memo: dict = {}
        self._h_memo: dict = {}
        self._cell_code: dict = {}     # cell contents -> byte, for `_pack`
        self._code_cell: list = []

    # -- canonical state ------------------------------------------------------
    def _key(self, eng) -> frozenset:
        """Every non-background, non-boundary cell.

        `PSPushExpert`'s key is pieces + player, which assumes the scenery is
        static. Pegs' is not: a triangle pair BUILDS a wall and a square FILLS a
        hole, and both of those decide what is reachable for the rest of the
        level."""
        skip = self.boundary_ids
        bg = self.bg_id
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in cell
            if o != bg and o not in skip
        )

    def _region_key(self, eng, region) -> bytes:
        return self._pack(eng)

    # -- packed state (one byte per cell) -------------------------------------
    def _pack(self, eng) -> bytes:
        """`_key`, as ``H * W`` bytes.

        The same information -- one code per distinct cell CONTENT, assigned as
        they are met -- at a fortieth of the size. It matters twice: `_astar`
        keeps one of these per generated node, and `_label`'s layered sweep
        keeps one per state on (or one step off) a shortest path, which is where
        a frozenset-of-triples key turns a 100k-state level into gigabytes.
        `_unpack` rebuilds a grid the interpreter treats identically -- hole
        boundaries are re-derived by the game's own rules on the next step, and
        Background is on every parsed cell by construction."""
        skip = self.boundary_ids
        bg = self.bg_id
        code = self._cell_code
        out = bytearray()
        for row in eng.grid:
            for cell in row:
                content = frozenset(o for o in cell
                                    if o != bg and o not in skip)
                got = code.get(content)
                if got is None:
                    got = code[content] = len(self._code_cell)
                    self._code_cell.append(content)
                    assert got < 256, "more cell contents than a byte holds"
                out.append(got)
        return bytes(out)

    def _unpack(self, eng, key: bytes) -> None:
        """Load a `_pack`ed state back into the engine."""
        w = len(eng.grid[0])
        bg = self.bg_id
        eng.grid = [[{bg} | set(self._code_cell[key[r * w + c]])
                     for c in range(w)]
                    for r in range(len(key) // w)]
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()

    # -- board scan -----------------------------------------------------------
    def _scan(self, eng) -> tuple:
        """``(pegs, holes, player, selected)`` for the current grid.

        ``pegs`` is a sorted tuple of ``(type, row, col)``; ``selected`` is the
        type currently showing in selection mode, or None."""
        pegs, holes = [], []
        player = selected = None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                if cell & self.player_ids:
                    player = (r, c)
                if cell & self.hole_ids:
                    holes.append((r, c))
                for o in cell:
                    t = self.type_of.get(o)
                    if t is None:
                        continue
                    pegs.append((t, r, c))
                    if o in self.selected_ids:
                        selected = t
        return tuple(sorted(pegs)), tuple(sorted(holes)), player, selected

    # -- macros ---------------------------------------------------------------
    def _analyze(self, eng) -> tuple:
        """`PSPushExpert._analyze`, except in selection mode.

        There the four choices ARE the branching: press an arrow n times to
        walk the ring round to the type you want, then ACTION to stamp it. Any
        arrow does, so only one is enumerated -- `_label` sweeps all four and
        finds the tie itself."""
        if self._selected_cell(eng) is not None:
            return 0, [["down"] * n + ["action"] for n in range(len(CYCLE))]
        return super()._analyze(eng)

    def _selected_cell(self, eng):
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & self.selected_ids:
                    return (r, c)
        return None

    # -- distances ------------------------------------------------------------
    def _distances(self, eng) -> dict:
        """``{cell: {cell: steps}}`` over the level's permanent walls, memoized
        on the wall map (which no rule in this game changes).

        One BFS per cell of a <=96-cell board, so building it is free next to a
        single interpreter step -- and it is the difference between a heuristic
        that guides and one that does not on the maze levels, where two pegs a
        manhattan 4 apart are a 12-step walk around a wall."""
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        wall = tuple(tuple(bool(cell & self.static_wall_ids) for cell in row)
                     for row in grid)
        got = self._dist.get(wall)
        if got is not None:
            return got
        field: dict = {}
        for r in range(h):
            for c in range(w):
                if wall[r][c]:
                    continue
                dist = {(r, c): 0}
                queue = deque([(r, c)])
                while queue:
                    cur = queue.popleft()
                    step = dist[cur] + 1
                    for dr, dc in _DELTA.values():
                        nxt = (cur[0] + dr, cur[1] + dc)
                        if (0 <= nxt[0] < h and 0 <= nxt[1] < w
                                and not wall[nxt[0]][nxt[1]] and nxt not in dist):
                            dist[nxt] = step
                            queue.append(nxt)
                field[(r, c)] = dist
        self._dist[wall] = field
        return field

    def _push_field(self, eng) -> dict:
        """``{from_cell: {to_cell: presses}}`` to shove ONE peg from one cell to
        another, memoized on the wall map beside `_distances`.

        Not the walk distance: a peg only moves when the player is directly
        behind it, so every TURN in a peg's route costs the player a trip round
        it -- two moves to swap to a perpendicular side, three to get to the
        opposite one -- and those trips are the single biggest thing a plain
        distance misses. Level 1's four triangles all have to be shoved round a
        corner to meet, and without the turn charge the estimate is short by
        most of a dozen moves there, which is the difference between a search
        that closes and one that runs for an hour.

        It stays a lower bound because the trips are disjoint from the pushes
        and from each other: the player really is behind the peg at each push,
        and really does have to get from one side of it to another between two
        pushes in different directions. Both are counted at their minimum, over
        the permanent walls only."""
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        wall = tuple(tuple(bool(cell & self.static_wall_ids) for cell in row)
                     for row in grid)
        got = self._push.get(wall)
        if got is not None:
            return got

        deltas = list(_DELTA.values())
        #: presses the player spends getting from behind a peg to the side the
        #: next push needs: 0 straight on, 2 to a perpendicular side (round one
        #: corner), 3 to the opposite side (round two).
        turn = {(i, j): (0 if i == j else
                         3 if deltas[i] == (-deltas[j][0], -deltas[j][1])
                         else 2)
                for i in range(4) for j in range(4)}
        field: dict = {}
        for r in range(h):
            for c in range(w):
                if wall[r][c]:
                    continue
                best: dict = {(r, c): 0}
                seen: dict = {}
                queue = []
                for i, (dr, dc) in enumerate(deltas):
                    nxt = (r + dr, c + dc)
                    if (0 <= nxt[0] < h and 0 <= nxt[1] < w
                            and not wall[nxt[0]][nxt[1]]):
                        heapq.heappush(queue, (1, nxt, i))
                while queue:
                    cost, cell, i = heapq.heappop(queue)
                    if seen.get((cell, i), 1 << 30) <= cost:
                        continue
                    seen[(cell, i)] = cost
                    if cost < best.get(cell, 1 << 30):
                        best[cell] = cost
                    for j, (dr, dc) in enumerate(deltas):
                        nxt = (cell[0] + dr, cell[1] + dc)
                        if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                            continue
                        if wall[nxt[0]][nxt[1]]:
                            continue
                        step = cost + 1 + turn[(i, j)]
                        if step < seen.get((nxt, j), 1 << 30):
                            heapq.heappush(queue, (step, nxt, j))
                field[(r, c)] = best
        self._push[wall] = field
        return field

    # -- the elimination matching --------------------------------------------
    def _elim(self, pegs, holes, field, push, anchor, slop) -> int:
        """Cheapest ORDERED scheme that empties ``pegs``, in presses.

        The player is somewhere within ``slop`` of ``anchor``; ``pegs`` is what
        is left. The recursion chooses which elimination happens NEXT and what
        it costs:

            walk to it     max(0, walk(anchor, a) - 1 - slop)
            + then either  push(a, hole)       drop ``a`` down a hole
                           push(a, b)          shove ``a`` into a peg it matches
                           push(a, b) + 1      forge a replacement out of two
                                               crosses -- the +1 is the ACTION
                                               press the selection mode forces,
                                               and a WILD peg is left standing
                                               at ``b`` for the rest of the
                                               scheme to deal with
            + `cycles`     the arrows a WILD needs before it can be what the
                           job wants it to be

        and recurses with the anchor moved to where that job finished. ``walk``
        is `_distances` (the player's own shortest path) and ``push`` is
        `_push_field` (the peg's, with the player's trips round it at every
        turn charged).

        WHY IT IS ORDERED. An unordered matching bounds only the PUSHES, and on
        these boards the pushes are a minority of the plan -- level 0 is 8
        pushes out of 17 presses, so a push-only estimate is short by nine and
        A* has nothing to steer with. Sequencing the jobs is what lets the walk
        BETWEEN them be charged, and it is the whole difference between a search
        that closes in seconds and one that does not close.

        The pushes alone are a bound outright, and that is what ``walk_bound =
        False`` leaves: each peg's own pushes cannot total less than the
        distance its job sends it, so the sum over jobs is a lower bound on the
        plan's pushes. The walk terms sharpen it by sequencing the jobs -- after
        a job the player stands one cell short of where it finished (two, for a
        cross pair, whose combining press does not move the player at all), so
        the next walk is charged from there with that much ``slop`` forgiven --
        and that part assumes each peg is handled in ONE stretch. See the class
        docstring for what that costs and how ``--optimal`` settles it.
        Distances themselves ignore the other pegs and any wall a triangle pair
        has built, which only ever makes the estimate smaller.

        And when it comes back `UNWINNABLE` there is no scheme at all, which is
        a proof rather than a guess: the relaxation permits strictly more than
        the game does."""
        if not pegs:
            return 0
        key = (pegs, holes, anchor, slop)
        got = self._elim_memo.get(key)
        if got is not None:
            return got
        from_anchor = field.get(anchor, {})
        best = UNWINNABLE
        for j, a in enumerate(pegs):
            reach = from_anchor.get((a[1], a[2]))
            if reach is None:
                continue                      # walled off from the player
            walk = max(0, reach - 1 - slop) if self.walk_bound else 0
            if walk >= best:
                continue
            rest = pegs[:j] + pegs[j + 1:]
            here = push.get((a[1], a[2]), {})
            for hole in holes:
                drop = here.get(hole)
                if drop is not None:
                    # A wildcard dropped down a hole is stamped with
                    # whatever is cheapest, i.e. Triangle: no arrows.
                    best = min(best, walk + drop
                               + self._elim(rest, holes, field, push, hole, 1))
            for i, b in enumerate(rest):
                # A cross never annihilates -- it FORGES, which is the branch
                # below. Everything else pairs with its own type, and a WILD
                # pairs with anything because that is the choice the player has
                # not made yet.
                if CROSS in (a[0], b[0]):
                    continue
                if a[0] != b[0] and WILD not in (a[0], b[0]):
                    continue
                step = here.get((b[1], b[2]))
                if step is not None:
                    # A WILD is a selection still sitting on Triangle, so
                    # making it match its partner costs that many arrows.
                    best = min(best, walk + step
                               + cycles(a[0], b[0]) + cycles(b[0], a[0])
                               + self._elim(rest[:i] + rest[i + 1:], holes,
                                            field, push, (b[1], b[2]), 1))
            # Forging: a cross shoved into a cross leaves a WILD standing where
            # the second one was, plus the ACTION press that stamps it. A WILD
            # counts as a cross here (one arrow away), which is what makes the
            # CHAIN expressible -- level 7 has three crosses, so the first
            # replacement has to be stamped as another cross and forged again.
            # Without that this reads as unwinnable and `dead` throws the whole
            # level away.
            if a[0] in (CROSS, WILD):
                for i, b in enumerate(rest):
                    if b[0] not in (CROSS, WILD) or CROSS not in (a[0], b[0]):
                        continue
                    step = here.get((b[1], b[2]))
                    if step is None:
                        continue
                    left = tuple(sorted(rest[:i] + rest[i + 1:]
                                        + ((WILD, b[1], b[2]),)))
                    best = min(best, walk + step + 1
                               + cycles(a[0], CROSS) + cycles(b[0], CROSS)
                               + self._elim(left, holes, field, push,
                                            (b[1], b[2]), 2))
        best = min(best, UNWINNABLE)
        self._elim_memo[key] = best
        return best

    def heuristic(self, eng) -> int:
        pegs, holes, player, selected = self._scan(eng)
        if not pegs:
            return 0
        key = (pegs, holes, player, selected)
        got = self._h_memo.get(key)
        if got is not None:
            return got
        field = self._distances(eng)
        push = self._push_field(eng)
        if player is None:                     # consumed by a rule
            got = UNWINNABLE
        elif selected is not None:
            # The selection is a WILDCARD the player has not spent yet, so the
            # estimate is a minimum over what it could become -- the arrows to
            # walk the ring there, plus the scheme that type allows, plus the
            # one ACTION press the mode always costs. Charging the CURRENT type
            # instead would be inadmissible: one arrow press can turn a board
            # with no scheme at all into one that is nearly solved.
            cell = self._selected_cell(eng)
            others = tuple(p for p in pegs if (p[1], p[2]) != cell)
            best = UNWINNABLE
            for t in range(len(CYCLE)):
                cand = tuple(sorted(others + ((t, cell[0], cell[1]),)))
                best = min(best, (t - selected) % len(CYCLE)
                           + self._elim(cand, holes, field, push, player, 0))
            got = min(UNWINNABLE, 1 + best)
        else:
            got = min(UNWINNABLE,
                      self._elim(pegs, holes, field, push, player, 0))
        self._h_memo[key] = got
        return got

    def dead(self, eng) -> bool:
        """Drop states that can never win: the player has fallen down a hole,
        or the elimination relaxation has no scheme left at all."""
        for row in eng.grid:
            for cell in row:
                if cell & self.dead_player_ids:
                    return True
        return self.heuristic(eng) >= UNWINNABLE

    @property
    def dead_player_ids(self) -> set:
        got = getattr(self, "_dead_ids", None)
        if got is None:
            got = self._dead_ids = set(
                self.g.resolve_object_name("invisibledeadplayer"))
        return got

    # -- optimal-action sets --------------------------------------------------
    def _search(self, eng):
        """`_astar`, then `_label`. (`_astar` leaves the grid dirty, hence the
        explicit restore.)"""
        start = snapshot(eng)
        found = self._astar(eng)
        restore(eng, start)
        if found is None:
            return None
        sets = self._label(eng, found)
        restore(eng, start)
        return Plan(found, sets)

    def _label(self, eng, plan) -> list:
        """Exact per-step optimal SETS for ``plan``, from one bounded layered
        BFS over PRIMITIVE presses. Engine at the plan's start state.

        WHY NOT `PSPushExpert.optimal_sets`. That one answers the same question
        by re-solving from every candidate press at every step -- 5 searches per
        step, each a full `_astar` -- and on level 0 alone that is 135s against
        the 10s the plan itself cost. It also cannot be more exact than the
        heuristic that prunes it: a press it prunes as hopeless because ``h``
        says so is a press that never gets measured.

        THE SWEEP instead builds the level's whole SHORTEST-PATH SUBGRAPH once.
        A layered forward BFS from the start keeps a successor only while
        ``g + 1 + h <= d*``, which (for an admissible ``h``) throws away exactly
        the states no shortest path can contain, and records the edges it kept.
        A backward pass over those edges then gives the exact distance-to-win of
        every state that survived, and the optimal set at each step of the plan
        is simply the presses whose successor is one closer. Two facts make the
        pruned subgraph the right object rather than an approximation: a state
        on a shortest path always has ``g + h <= d*`` (so it is kept), and a
        state that is not on one can only be assigned a distance that is too
        BIG by a subgraph that is missing edges (so it is still, correctly, not
        optimal).

        This is ps:fractured_identity's bounded layered BFS, on the plan the
        macro A* has already proved."""
        target = len(plan)
        start = self._pack(eng)
        layers: list = [[start]]
        depth = {start: 0}
        edges: dict = {}
        WIN = _WIN                             # sentinel child: the goal
        for g in range(target):
            nxt = []
            for key in layers[g]:
                out = []
                for press in self.directions:
                    self._unpack(eng, key)
                    eng.step(press)
                    if eng.check_win():
                        out.append((press, WIN))
                        continue
                    child = self._pack(eng)
                    if child == key:
                        continue               # a press the board ignored
                    seen = depth.get(child)
                    if seen is None:
                        if g + 1 + self.heuristic(eng) > target:
                            continue
                        depth[child] = g + 1
                        nxt.append(child)
                    elif seen != g + 1:
                        continue               # not on a shortest path via here
                    out.append((press, child))
                edges[key] = out
            layers.append(nxt)

        to_win: dict = {}
        for g in range(target - 1, -1, -1):
            for key in layers[g]:
                best = None
                for _press, child in edges[key]:
                    cost = 1 if child is WIN else (
                        None if child not in to_win else 1 + to_win[child])
                    if cost is not None and (best is None or cost < best):
                        best = cost
                if best is not None:
                    to_win[key] = best
        assert to_win.get(start) == target, (
            f"labelling sweep found {to_win.get(start)} where the search "
            f"proved {target}")

        # Read the sets off the plan's own states.
        self._unpack(eng, start)
        out = []
        for i, taken in enumerate(plan):
            key = self._pack(eng)
            want = to_win[key] - 1
            best = [press for press, child in edges[key]
                    if (0 if child is WIN else to_win.get(child, 1 << 30))
                    == want]
            # The recorded step ties with itself by construction; a
            # disagreement would mean the sweep and the search disagree, so
            # fall back to labelling what the expert actually did.
            out.append(best if taken in best else [taken])
            eng.step(taken)
        return out


class PegsSolver(PSAStarSolver):
    game_id = "puzzlescript_pegs"
    game_name = GAME_NAME
    expert_cls = PegsExpert

    #: Record against the GAME FOLDER's adapter -- that file is what
    #: `game_envs` hands a live agent, so a sprite patch or step cap added to it
    #: later must not silently diverge from what this generator tapes.
    game_module_id = "ps:pegs"

    #: Unweighted, with `exact_goal_test`, so plans are shortest (``--optimal``
    #: is the certificate). The whole search is a few minutes once and cached to
    #: disk, so there is nothing to buy by trading optimality away.
    weight = 1
    #: A memory budget, not a time one: every queued node holds a board
    #: snapshot plus its macro list. No level comes near this.
    node_cap = 2_000_000
    #: The longest plan is 52 moves; the rest is room for the RESET exploration
    #: prefix and the re-plan after it, under the adapter's own per-level cap.
    max_steps = 150

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time cost visible as startup and
        fills the disk cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _levels(game) -> list:
    """Levels named as bare integers on the command line, else all of them.

    The heavy modes (`_optimal` above all) are minutes per level, so being able
    to say ``--optimal 7`` rather than re-proving the whole game is the
    difference between a check that gets run and one that does not."""
    picked = [int(a) for a in sys.argv[1:] if a.isdigit()]
    return picked or list(range(game.n_levels))


def _peg_census(expert, eng) -> str:
    pegs, holes, _player, _sel = expert._scan(eng)
    mark = {TRIANGLE: "<", CROSS: "+", SQUARE: "q", CIRCLE: "o"}
    counts = {t: sum(1 for p in pegs if p[0] == t) for t in mark}
    out = " ".join(f"{counts[t]}{mark[t]}" for t in (CIRCLE, SQUARE, TRIANGLE,
                                                     CROSS) if counts[t])
    return out + (f", {len(holes)} hole" if holes else "")


def _report() -> int:
    """Per-level peg census, plan length, push share and tie coverage.

    Also the admissibility check the optimality claim rests on: replay the plan
    and require ``h`` at every step to be no more than the moves remaining. An
    inadmissible heuristic does not crash, it just quietly returns a plan that
    is not shortest."""
    import time

    game = PegsSolver().make_game(0)
    expert = PegsExpert(game, node_cap=PegsSolver.node_cap)
    total = 0
    bad = 0
    levels = _levels(game)
    for level in levels:
        game.set_level(level)
        eng = game._engine
        census = _peg_census(expert, eng)
        t0 = time.time()
        found = expert.plan(eng, level)
        dt = time.time() - t0
        if found is None:
            print(f"level {level:2d}: NO PLAN")
            bad += 1
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)

        # Replay: count pushes, and check the heuristic never over-estimates.
        start = snapshot(eng)
        pushes = 0
        slack = []
        for i, direction in enumerate(found):
            est = expert.heuristic(eng)
            slack.append(len(found) - i - est)
            before = [[set(cell) for cell in row] for row in eng.grid]
            eng.step(direction)
            pegs_of = expert.push_ids
            moved = any(
                (before[r][c] & pegs_of) != (eng.grid[r][c] & pegs_of)
                for r in range(len(before)) for c in range(len(before[0])))
            pushes += bool(moved)
        won = eng.check_win()
        restore(eng, start)
        if not won or min(slack) < 0:
            bad += 1
        total += len(found)
        print(f"level {level:2d}: {eng.height:2d}x{eng.width:2d}  "
              f"{census:16s} {len(found):3d} moves, {pushes:3d} push, "
              f"{ties:3d} ties ({ties / max(1, len(found)):3.0%}), "
              f"h<=d* {'ok' if min(slack) >= 0 else 'VIOLATED'}, "
              f"{'WIN' if won else 'NO WIN'}, {dt:6.1f}s")
    print(f"total {total} moves over {len(levels)} levels"
          + ("" if not bad else f" -- {bad} LEVEL(S) BAD"))
    return 0 if not bad else 1


def _verify() -> int:
    """Drive every cached plan through the ADAPTER (not the raw engine) at a
    handful of seeds, so the rotation contract is exercised end to end: a plan
    is only real if the screen presses it is recorded as reproduce the win."""
    from arcengine import ActionInput, GameState

    from solvers.common.ps_astar import screen_action

    solver = PegsSolver()
    game = solver.make_game(0)
    expert = PegsExpert(game, node_cap=PegsSolver.node_cap)
    bad = 0
    for seed in (0, 1, 2, 3):
        game_s = solver.make_game(seed)
        for level in _levels(game_s):
            game.set_level(level)
            plan = expert.plan(game._engine, level)
            game_s.set_level(level)
            # AFTER set_level, never before: `PuzzleScriptAdapter.set_level`
            # re-seeds from ``seed + idx``, so the presentation is per
            # (seed, LEVEL). Reading it once per seed silently replays three
            # quarters of the levels through the wrong remap -- which is what
            # this mode exists to catch, and it caught itself first.
            rot = (game_s._rotation_k, game_s._hflip, game_s._vflip)
            for direction in plan:
                act = screen_action(direction, *rot)
                game_s.perform_action(ActionInput(id=act))
            ok = game_s._state == GameState.WIN
            bad += not ok
            print(f"seed {seed} level {level}: rot {rot[0]}"
                  f"{' h' if rot[1] else ''}{' v' if rot[2] else ''}, "
                  f"{len(plan):3d} presses -> {'WIN' if ok else 'FAILED'}")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad}")
    return 0 if not bad else 1


def _optimal() -> int:
    """The optimality certificate: re-solve every level under the provably
    admissible matching-only heuristic and require the same plan length.

    `PegsExpert.heuristic`'s walk terms sharpen the estimate by assuming each
    peg is dealt with in one stretch, which is a search GUIDE rather than a
    proof (see the class docstring). Dropping them leaves a pure elimination
    matching, which is a lower bound outright -- and with ``weight = 1``,
    ``exact_goal_test`` and exact dedup, an admissible heuristic makes
    `_astar`'s answer shortest by construction. Agreement between the two is
    therefore the certificate; a disagreement would mean the shipped plan is
    not shortest.

    It is slower than the guided search (the estimate is weaker, so A* expands
    more), which is exactly why it is a separate mode rather than the default.
    """
    import time

    class _Admissible(PegsExpert):
        walk_bound = False
        plan_cache_path = None          # never poison the shipped cache

    solver = PegsSolver()
    game = solver.make_game(0)
    guided = PegsExpert(game, node_cap=PegsSolver.node_cap)
    proof = _Admissible(game, node_cap=PegsSolver.node_cap)
    bad = 0
    for level in _levels(game):
        game.set_level(level)
        want = guided.plan(game._engine, level)
        game.set_level(level)
        t0 = time.time()
        got = proof.plan(game._engine, level)
        dt = time.time() - t0
        same = (want is not None and got is not None
                and len(want) == len(got))
        bad += not same
        print(f"level {level:2d}: shipped {len(want) if want else None:>4}, "
              f"matching-only {len(got) if got else None:>4} "
              f"({dt:6.1f}s) -- {'PROVED SHORTEST' if same else 'MISMATCH'}")
    print("optimality clean" if not bad else f"NOT PROVED: {bad} level(s)")
    return 0 if not bad else 1


#: The 8 presentations of a square board, as (quarter turns, mirror).
_GROUP = [(k, m) for k in range(4) for m in (False, True)]


def _turn_grid(grid):
    """One quarter turn clockwise: ``(r, c) -> (c, H - 1 - r)``."""
    h, w = len(grid), len(grid[0])
    return [[grid[h - 1 - c][r] for c in range(h)] for r in range(w)]


def _mirror_grid(grid):
    return [list(reversed(row)) for row in grid]


#: The direction relabelling that goes with `_turn_grid` / `_mirror_grid`.
_TURN_DIR = {"up": "right", "right": "down", "down": "left", "left": "up",
             "action": "action"}
_MIRROR_DIR = {"left": "right", "right": "left", "up": "up", "down": "down",
               "action": "action"}


def _transform(grid, k, mirror):
    for _ in range(k):
        grid = _turn_grid(grid)
    return _mirror_grid(grid) if mirror else grid


def _transform_dir(direction, k, mirror):
    for _ in range(k):
        direction = _TURN_DIR[direction]
    return _MIRROR_DIR[direction] if mirror else direction


def _symmetry(rounds: int = 400) -> int:
    """Is the 8-element presentation group a symmetry of the MECHANIC?

    This is the question `PuzzleScriptAdapter._FLIP_GAMES` membership turns on,
    and the recipe is ps:icecrates' plus ps:lovendpieces' correction to it:
    replaying a shortest PLAN on the eight copies of its board is not enough,
    because no shortest plan need ever traverse an order-sensitive transition.
    So the board states come from a RANDOM WALK, and each recorded
    ``(board, press)`` is replayed INDEPENDENTLY on all eight presentations of
    that same board.

    Boards are transformed as LAYOUTS and reloaded, so the interpreter re-runs
    the level-start rules and re-derives the hole-boundary art itself; that art
    is directional on purpose and is excluded from the comparison, as are the
    engine's own bookkeeping objects. Everything the player can move or destroy
    is compared, plus ``check_win``."""
    import random

    solver = PegsSolver()
    game = solver.make_game(0)
    eng, g = game._engine, game._game
    expert = PegsExpert(game, node_cap=PegsSolver.node_cap)
    boundary = expert.boundary_ids
    dirs = ["up", "down", "left", "right", "action"]

    def pieces(grid):
        return frozenset((r, c, o)
                         for r, row in enumerate(grid)
                         for c, cell in enumerate(row)
                         for o in cell
                         if o != expert.bg_id and o not in boundary)

    def load(layout):
        eng.load_level([[set(cell) for cell in row] for row in layout])

    bad = 0
    checked = 0
    rng = random.Random(0)
    for level in _levels(game):
        base = g.levels[level]
        load(base)
        walk = []
        for _ in range(rounds):
            layout = [[set(cell) for cell in row] for row in eng.grid]
            direction = rng.choice(dirs)
            walk.append((layout, direction))
            eng.step(direction)
            if eng.check_win():
                load(base)

        for layout, direction in walk:
            load(layout)
            eng.step(direction)
            ref, ref_win = pieces(eng.grid), eng.check_win()
            for k, mirror in _GROUP[1:]:
                load(_transform(layout, k, mirror))
                eng.step(_transform_dir(direction, k, mirror))
                checked += 1
                want = pieces(_transform(
                    [[set(cell) for cell in row] for row in
                     _grid_of(ref, len(layout), len(layout[0]),
                              expert.bg_id)], k, mirror))
                if pieces(eng.grid) != want or eng.check_win() != ref_win:
                    bad += 1
                    if bad <= 5:
                        print(f"  level {level} rot {k} mirror {mirror}: "
                              f"press {direction} disagrees")
        print(f"level {level}: {len(walk)} boards x 7 presentations "
              f"-- {'SYMMETRIC' if not bad else str(bad) + ' MISMATCH'}")
    print(f"{checked} transitions checked; "
          + ("symmetry clean" if not bad else f"NOT SYMMETRIC: {bad}"))
    return 0 if not bad else 1


def _grid_of(cells, h, w, bg):
    """Rebuild a grid of object sets from a `pieces` frozenset."""
    grid = [[set() for _ in range(w)] for _ in range(h)]
    for r, c, o in cells:
        grid[r][c].add(o)
    return grid


def _audit() -> int:
    """Assert every cell COMPOSITION the game can show is distinct.

    WHOLE-FRAME, not the per-cell crop the older audits in this tree use: that
    crop is arithmetic on ``64 // max(h, w)`` and is simply wrong for any board
    whose render `_render_frame` upscales, which this one is (60x40 -> 64x42).
    Filling the whole board with one composition and comparing whole frames is
    the same test with no geometry to get wrong, since rendering is per-cell
    independent.

    The three collisions it reports as EXPECTED are all FilledHole, which is
    drawn in the game's own near-white and therefore reads as floor. That is
    correct rather than a defect: a filled hole IS floor -- nothing in the rule
    set can tell the two apart, so a frame that showed them differently would be
    showing state the game does not have."""
    import itertools

    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    game = PegsSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    boundary = ("holeboundaryup", "holeboundaryright",
                "holeboundarydown", "holeboundaryleft")
    comps = {
        "floor": (), "wall": ("normalsolidblock",),
        "created_wall": ("createdsolidblock",),
        "player": ("normalplayer",), "selecting": ("selectingplayer",),
        "dead": ("invisibledeadplayer",),
        "circle": ("circle",), "square": ("square",),
        "triangle": ("triangle",), "cross": ("cross",),
        "sel_circle": ("selectedcircle",), "sel_square": ("selectedsquare",),
        "sel_triangle": ("selectedtriangle",), "sel_cross": ("selectedcross",),
        "hole": ("emptyhole",), "hole_walled": ("emptyhole",) + boundary,
        "filled": ("filledhole",),
        "player_on_filled": ("filledhole", "normalplayer"),
        "circle_on_filled": ("filledhole", "circle"),
    }
    #: FilledHole is floor, by the rules as well as by the picture; see above.
    expected = {frozenset(pair) for pair in
                (("floor", "filled"), ("player", "player_on_filled"),
                 ("circle", "circle_on_filled"))}

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
                   if np.array_equal(shots[a], shots[b])
                   and frozenset((a, b)) not in expected]
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (levels {','.join(str(x) for x in levels)}): "
              f"{len(comps)} compositions, "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--optimal" in sys.argv:
        sys.exit(_optimal())
    sys.exit(PegsSolver.main())
