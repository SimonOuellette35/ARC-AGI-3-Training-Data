"""Generate Phase-1 training data for the PuzzleScript game ps:circuit_breaker
(Zithral's "Circuit Breaker").

The harness -- the rotation contract, the trajectory recorder and the BaseSolver
plumbing -- lives in `solvers/common/ps_astar.py`, shared with the other ps:
generators. This file is the game-specific part: the goal model, a native model
of the mechanic to search in, the heuristic built from the goal model, and the
optimal-action tie sets the plan ships as training targets.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_circuit_breaker",
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
Sokoban with a RELATIONAL goal. Crates are circuit components: each carries some
subset of four green connector stubs (``EdgeU/D/L/R``), drawn from the middle of
its cell out to the cell wall on that side. The push rules are textbook --

    [ > Player | Crate ] -> [ > Player | > Crate ]
    [ > Crate | Blocker ] -> [ Crate | Blocker ]
    [ > Crate Edge ]      -> [ > Crate > Edge ]

-- one crate is shoved one cell, never a chain, and it carries its stubs with
it. ``noaction`` is in the prelude, so the four arrows are the whole action
space.

What makes it not-Sokoban is the win condition. There are no target cells:

    all EdgeU on MatchU / EdgeD on MatchD / EdgeL on MatchL / EdgeR on MatchR

and the late rules stamp a Match only where two stubs meet across a cell wall
(``late down [ EdgeD | EdgeU ]``, ``late right [ EdgeR | EdgeL ]``). So the goal
is "every connector is plugged into its opposite number on the neighbouring
crate" -- a shape to be ASSEMBLED, at no particular place on the board.
``FullMatch`` (a white pip in the middle of a crate) marks a crate whose every
stub is plugged, which is the rendered feedback the mechanic gives: a win is
"every crate wears a white pip", and the stubs visibly join into unbroken green
lines across the assembled block.

Five levels, 3-8 crates on 5x5..8x8 interiors. The last two put the player
INSIDE a ring of crates, so even the first move has to be a push.

Solving for the goal (this is the whole solver)
-----------------------------------------------
"Assemble a shape" is only a search-friendly goal once you know WHICH shape, so
the expert derives it up front instead of groping for it with a heuristic.

A winning layout is exactly a pairing: every D stub is plugged into some crate's
U stub sitting one cell BELOW it, every R stub into some L stub one cell to the
RIGHT. Choose those two bijections and the crates' relative positions are forced
-- propagate ``pos[u] = pos[d] + (1, 0)`` and ``pos[l] = pos[r] + (0, 1)`` through
a union-find over offsets, and a choice either contradicts itself or lays every
crate out at a fixed offset from its component's origin. Keep the layouts that
survive the mechanic's own test, applied cell by cell: crate ``i`` has a stub on
side ``e`` **if and only if** the neighbour on that side exists and has the
opposite stub. (Both halves matter. A stub with no neighbour is unmatched, and so
is a neighbour's stub pointing back at a blank side.)

`_goal_shapes` runs that over ``|D|! x |R|!`` pairings -- at most 2880 here -- and
on all five shipped levels it returns **exactly one shape, in one connected
piece**. `_placements` then slides it over the board, and what comes out is the
COMPLETE, explicit list of winning positions: 6-14 of them per level. Two things
fall out of having it.

  * The heuristic is a Sokoban one again (below).
  * **The win test is the heuristic.** A layout wins iff it is one of those
    placements, iff every crate sits on its own target cell for that placement,
    iff the push-distance sum below is 0. So the search never needs a separate
    goal test, which matters because it runs one per generated node.

Searching a native model
------------------------
The other ps: generators search the interpreter itself. Here that is too slow to
finish: one `step` costs **1.1 ms** -- 40x the BFS, the heuristic, the state key
and the snapshot in a node put together -- and the packed levels need millions of
them. Level 3's answer is 62 presses, and the interpreter search was still at
f = 26 after 20k nodes and 22 seconds. So the search runs on a native model
instead -- crate cells, their connector patterns, the player, and the one push
rule -- at about 10 us a node.

The model is only as trustworthy as its checks, so there are two:

  * ``--verify-model`` fuzzes it against the real interpreter, walking each level
    through thousands of random presses and comparing the crate layout, the
    player and the win flag after every single one.
  * Every plan this generator emits is REPLAYED through the real interpreter in
    `_annotate` (which has to walk it anyway to find the tie sets), and asserts
    the interpreter agrees it won. A model that drifted would abort the run, not
    quietly ship losing trajectories.

The search itself is A* over ``walk to the push cell, then push`` MACROS, so its
depth is the number of pushes and each walk is one BFS rather than four levels of
branching. Costs are counted in primitive MOVES, the unit the agent pays, and a
state keeps the player's EXACT cell rather than its reachable region -- see
`_model_astar` for why the usual Sokoban canonicalisation is the wrong trade on
these boards.

The heuristic, minimised over every placement, is

    sum over crates of push-distance(crate's cell -> its cell in that placement)

matched exactly within each connector-pattern class (crates with identical stub
sets are interchangeable, so a class of k is minimised over its k! assignments),
plus the player's walk to the nearest crate. Push-distance is the usual per-cell
reverse BFS over push edges -- a crate reaches ``X`` from ``X - d`` only when
``X - d`` and the player's stand cell ``X - 2d`` are both on the board and not
walls -- so the tables double as the **deadlock test** they do in
`generate_bad_example_training.py`: a crate wedged where two walls meet has no
outgoing push, appears in no table, and its subtree is dropped outright.

Every term is a lower bound (push-distance ignores the other crates; the min runs
over the complete set of winning layouts), so a plan rung 1 returns is SHORTEST.
Where the estimate is too loose to close there -- level 3 packs 8 crates round the
player, and rung 1 runs out of nodes on it -- `_WEIGHTS` escalates; a plan from a
higher rung is still engine-verified and winning, just no longer certified
shortest. ``--report`` prints the rung each level used.

Optimal-action tie sets
-----------------------
A macro plan is mostly WALK, and a shortest walk to a stand cell is never unique
-- any interleaving of the two axes is exactly as good, and training a single
arbitrary one as the only right answer teaches an ordering the game does not
have. `_annotate` replays the finished plan, splits it at the steps that actually
moved a crate, and for each walk step emits every direction that still lies on a
shortest walk to the same stand cell (one BFS per walk segment). Push steps are
labelled with the push. So every expert step ships an ``optimal`` set and none of
them is a guess -- see [[always-emit-optimal-targets]].

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the per-(seed, level) variables are purely presentation: the frame
rotation (rotation_k in {0,1,2,3}) plus independent horizontal and vertical flips
with the matching directional action remap (`PuzzleScriptAdapter._FLIP_GAMES`),
which re-sample the board's 8-element symmetry group -- 5 levels x 8 = 40
presentations instead of 20.

The flips are exact here for a reason worth stating, because this is the one game
in that set with genuinely DIRECTIONAL sprites. A connector's meaning is its
geometry: ``EdgeU`` is a stub drawn to the top wall of its cell, and it plugs into
the crate above. Mirroring the frame mirrors the stub art along with the board, so
a pair that met across a wall still meets, and the remapped arrows push them
together the same way. The U/D and L/R rules are each other's transpose, there is
no gravity, and input is screen-relative.

The expert plan is therefore seed-independent: solved once per level, cached on
disk (see `_PLAN_CACHE`), and replayed per seed with that seed's remapped screen
actions.

Rendering note
--------------
Crate was ``#555555``/``#666666``, and BOTH quantize to ARC index 3 -- the same
index as Wall's DARKGRAY, so every crate rendered as a solid block identical to
the walls it sits between (see [[ps-palette-collisions]]). Repainted Gray (2)
border / LightBlue (10) body in `data/puzzlescript_games/Circuit_Breaker.txt`,
which are unused by every other object here. The Match markers stay invisible on
purpose: they are #00ff00 where the stubs are #00aa00 and ARC has one green, and
each Match sprite is a strict sub-segment of the stub it sits on, so they could
add nothing to the frame. Nothing is lost -- FullMatch's white pip reports the
same fact per crate, and it is the only per-crate signal the ORIGINAL game shows
either.

Usage (run from the repo root):
    python solvers/generate_circuit_breaker_training.py --report       # warm the
    python solvers/generate_circuit_breaker_training.py --episodes 200 \
        --out data/training_multi_level/circuit_breaker
    python solvers/generate_circuit_breaker_training.py --verify-model

``--report`` first is worth the habit before a `parallelize_generator.py` run: it
leaves `_PLAN_CACHE` on disk, and shards that find it warm search nothing at all
rather than each repeating the same ~90s of A* at the same moment.
"""

from __future__ import annotations

import heapq
import json
import random
import sys
from collections import deque
from itertools import permutations, product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import PSAStarSolver, PSPushExpert  # noqa: E402

GAME_NAME = "Circuit_Breaker"

#: Connector bits. A crate's "pattern" is the OR of the sides it has a stub on.
U, D, L, R = 1, 2, 4, 8
_SIDES = (U, D, L, R)
_SIDE_DELTA = {U: (-1, 0), D: (1, 0), L: (0, -1), R: (0, 1)}
_OPPOSITE = {U: D, D: U, L: R, R: L}

#: The four presses, in the engine's own names.
_MOVES = ("up", "down", "left", "right")
_MOVE_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Heuristic charge for a state that can no longer win -- a crate parked where no
#: placement's cell is reachable from (a corner, a pocket the player cannot get
#: behind). Nodes that score it are dropped, not queued.
_DEAD = 1 << 30

#: Ceiling on the placements the heuristic minimises over. One connected shape
#: gives at most a board's worth of translations (tens), so this only ever bites
#: on a hypothetical many-component shape, where the product of per-component
#: translations could explode. Past it the placement list is truncated, which can
#: only make the heuristic larger, so plans stay winning -- they just stop being
#: certified shortest.
_MAX_PLACEMENTS = 4096

#: Exact assignment is over k! permutations within a connector-pattern class;
#: past this a greedy nearest-first matching stands in (still a search guide, no
#: longer a lower bound). The shipped levels top out at k = 2.
_EXACT_ASSIGN = 6

#: Weight ladder for the macro A*. Rung 1 is admissible, so its plans are
#: shortest; the packed levels that cannot close there fall through to a greedier
#: rung, whose plans are still engine-verified wins. `_report` names the rung each
#: level used.
_WEIGHTS = (1, 2, 3, 5)

#: Nodes generated before a rung gives up and the next one starts. This is a
#: MEMORY bound as much as a time one -- every generated node leaves an entry in
#: the cost table -- and the ladder is there precisely so a rung is allowed to
#: fail cheaply. 2M peaks around 400 MB and builds all five levels in ~90s.
#:
#: The one level it costs anything is level 3, which falls through rung 1 and is
#: answered by rung 3 in 62 presses; at 8M (about 1.5 GB, ~4 minutes) rung 2
#: reaches it and returns 56. That is not worth making the default, because
#: `parallelize_generator.py` starts every shard at once and each one that finds
#: `_PLAN_CACHE` cold pays this in parallel with the others. Warm the cache with
#: ``--report`` before a sharded run and the shards search nothing at all.
_NODE_CAP = 2_000_000

#: Solved plans, keyed by level. The searches are seed-independent (levels are
#: fixed ASCII maps and only the PRESENTATION is augmented), but they are the
#: whole cost of this generator, and `parallelize_generator.py` would otherwise
#: repeat them in every shard. Cached entries are re-verified against the
#: interpreter on load, so a stale file cannot survive a level edit.
_PLAN_CACHE = (Path(__file__).resolve().parent.parent
               / "data" / "circuit_breaker_plans.json")


class Plan(list):
    """A plan that carries a per-step optimal-action SET (`_annotate`)."""

    optsets: list[list[str]] | None = None
    #: `_WEIGHTS` rung that found it, for `_report`.
    weight: int = 0


# ---------------------------------------------------------------------------
# The goal model: which layouts win
# ---------------------------------------------------------------------------

def _offsets(n: int, constraints) -> list[tuple[int, tuple[int, int]]] | None:
    """Solve ``pos[b] = pos[a] + delta`` for every ``(a, b, delta)``.

    Union-find over offsets: returns ``[(component root, offset from it)]`` per
    crate, or None if two constraints contradict each other."""
    parent = list(range(n))
    off = [(0, 0)] * n          # offset from the parent; (0, 0) at a root

    def find(x):
        path = []
        node = x
        while parent[node] != node:
            path.append(node)
            node = parent[node]
        # Re-point each node on the path straight at the root, nearest first, so
        # by the time a node is rewritten its own parent already holds an offset
        # measured FROM the root.
        for step in reversed(path):
            up = off[parent[step]]
            off[step] = (off[step][0] + up[0], off[step][1] + up[1])
            parent[step] = node
        return node, off[x] if x != node else (0, 0)

    for a, b, (dr, dc) in constraints:
        ra, oa = find(a)
        rb, ob = find(b)
        if ra == rb:
            if (ob[0] - oa[0], ob[1] - oa[1]) != (dr, dc):
                return None
        else:
            parent[rb] = ra
            off[rb] = (oa[0] + dr - ob[0], oa[1] + dc - ob[1])
    return [find(i) for i in range(n)]


def _valid_layout(cells: dict) -> bool:
    """The mechanic's own win test on a concrete ``{cell: pattern}`` layout: a
    crate has a stub on side ``e`` IFF the neighbour there has the opposite one.

    Checking all four sides of every crate covers both failure modes -- a stub
    pointing at nothing, and a blank side facing someone else's stub."""
    for (r, c), pattern in cells.items():
        for side in _SIDES:
            dr, dc = _SIDE_DELTA[side]
            other = cells.get((r + dr, c + dc))
            plugged = other is not None and bool(other & _OPPOSITE[side])
            if bool(pattern & side) != plugged:
                return False
    return True


def _goal_shapes(patterns: list[int]) -> list[tuple]:
    """Every winning arrangement of these crates, as relative geometry.

    Returns a sorted list of shapes; a shape is a tuple of connected COMPONENTS
    and a component is a sorted tuple of ``((dr, dc), pattern)`` normalised so its
    top-left corner is the origin. Absolute position is not part of it -- a
    layout wins wherever it is put -- and neither is crate identity, since two
    crates with the same stub pattern are interchangeable.

    See the module docstring: choosing which D plugs into which U and which R
    into which L determines the whole geometry, so this enumerates those two
    bijections and keeps the ones the mechanic accepts."""
    n = len(patterns)
    downs = [i for i, p in enumerate(patterns) if p & D]
    ups = [i for i, p in enumerate(patterns) if p & U]
    rights = [i for i, p in enumerate(patterns) if p & R]
    lefts = [i for i, p in enumerate(patterns) if p & L]
    if len(downs) != len(ups) or len(rights) != len(lefts):
        return []                    # a stub with nothing of its kind to plug into

    shapes = set()
    for pu in permutations(ups):
        vertical = [(downs[i], pu[i], (1, 0)) for i in range(len(downs))]
        for pl in permutations(lefts):
            placed = _offsets(n, vertical + [(rights[i], pl[i], (0, 1))
                                             for i in range(len(rights))])
            if placed is None:
                continue
            groups: dict = {}
            for i, (root, delta) in enumerate(placed):
                groups.setdefault(root, {})[delta] = patterns[i]
            if sum(len(cells) for cells in groups.values()) != n:
                continue             # two crates solved to the same cell
            if not all(_valid_layout(cells) for cells in groups.values()):
                continue
            shape = []
            for cells in groups.values():
                top = min(r for r, _ in cells)
                left = min(c for _, c in cells)
                shape.append(tuple(sorted(((r - top, c - left), p)
                                          for (r, c), p in cells.items())))
            shapes.add(tuple(sorted(shape)))
    return sorted(shapes)


def _placements(shapes, free: set, cap: int = _MAX_PLACEMENTS) -> list[dict]:
    """Every winning layout that fits on this board, as ``{cell: pattern}``.

    One per (shape, translation of each of its components) whose cells are all
    free of walls, that does not stack two components, and that still passes
    `_valid_layout` once the components sit together -- components placed side by
    side can plug into, or block, each other, which the per-component test could
    not see."""
    out: dict[tuple, dict] = {}
    for shape in shapes:
        options = []
        for component in shape:
            here = []
            for (r0, c0) in free:
                cells = {(r0 + dr, c0 + dc): p for (dr, dc), p in component}
                if all(cell in free for cell in cells):
                    here.append(cells)
            if not here:
                break
            options.append(here)
        else:
            for combo in product(*options):
                merged: dict = {}
                for cells in combo:
                    merged.update(cells)
                if len(merged) != sum(len(cells) for cells in combo):
                    continue                       # components overlap
                if not _valid_layout(merged):
                    continue
                out[tuple(sorted(merged.items()))] = merged
                if len(out) >= cap:
                    return list(out.values())
    return list(out.values())


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class CircuitBreakerExpert(PSPushExpert):
    """Macro A* over a native model of the push mechanic, aimed at the explicit
    list of winning layouts. See the module docstring.

    Inherits `PSPushExpert` for its plan memo, its level-scoped dynamic-object
    state key and its object-id resolution; the search itself is `_search`, over
    the model rather than the interpreter.

    STATE ENCODING. Cells are ``row * width + col`` ints -- the hot loops index
    sets and dicts with them millions of times, and tuple keys cost several times
    as much. Every level is walled all the way round (`_bind` checks it), so
    ``code - 1`` can never wrap from column 0 onto the previous row: the cell it
    would land on is a wall and is not in `_free`.

    A model state is ``(cells, player)`` where ``cells`` is a tuple of crate cells
    laid out in `_spans` order -- one contiguous span per connector pattern, each
    span sorted. Crates with the same pattern are interchangeable, so sorting
    inside a span is what makes two orderings of them ONE state."""

    pushable_names = ("crate",)
    blocker_names = ("wall",)

    def setup(self) -> None:
        super().setup()
        g = self.g
        self.crate = g.obj_name_to_idx["crate"]
        self.side_ids = {side: g.obj_name_to_idx[name] for side, name in
                         ((U, "edgeu"), (D, "edged"),
                          (L, "edgel"), (R, "edger"))}
        self._board_cache: dict = {}

    # -- reading the interpreter ----------------------------------------------
    def _crates(self, eng) -> dict:
        """``{(row, col): connector pattern}`` for every crate on the board."""
        crate, side_ids = self.crate, self.side_ids
        return {(r, c): sum(side for side, o in side_ids.items() if o in cell)
                for r, row in enumerate(eng.grid)
                for c, cell in enumerate(row) if crate in cell}

    def _player_cell(self, eng) -> tuple[int, int] | None:
        return next(((r, c)
                     for r, row in enumerate(eng.grid)
                     for c, cell in enumerate(row) if cell & self.player_ids),
                    None)

    def _read(self, eng) -> tuple[tuple, int]:
        """The interpreter's board as a model state (see the class docstring)."""
        width = self._width
        crates = self._crates(eng)
        by_pattern: dict = {}
        for (r, c), pattern in crates.items():
            by_pattern.setdefault(pattern, []).append(r * width + c)
        cells = []
        for pattern, _start, _end in self._spans:
            cells.extend(sorted(by_pattern[pattern]))
        player = self._player_cell(eng)
        return tuple(cells), player[0] * width + player[1]

    # -- per-level static data -------------------------------------------------
    def plan(self, eng, level: int | None = None):
        """Bind this board's winning layouts and distance tables, then plan.

        The walls never move and no rule creates or destroys a crate or a stub,
        so both depend only on the LAYOUT and are built once per level here --
        `_estimate` runs on every generated node and must not re-derive static
        facts."""
        self._bind(eng)
        return super().plan(eng, level)

    def _bind(self, eng) -> None:
        """Resolve everything the model and the heuristic need for this board:
        the free cells, the winning placements, the push-distance tables and the
        pattern spans of the state encoding."""
        width = len(eng.grid[0])
        crates = self._crates(eng)
        walls = {(r, c)
                 for r, row in enumerate(eng.grid)
                 for c, cell in enumerate(row) if cell & self.blocker_ids}
        free = {(r, c)
                for r in range(len(eng.grid))
                for c in range(width) if (r, c) not in walls}
        # Width is in the key because it is the cell encoding's base: two levels
        # with the same interior on differently sized grids are different boards.
        signature = (len(eng.grid), width, frozenset(free),
                     tuple(sorted(crates.values())))
        cached = self._board_cache.get(signature)
        if cached is None:
            cached = self._build(free, sorted(crates.values()), width,
                                 len(eng.grid))
            self._board_cache[signature] = cached
        (self._width, self._free, self._steps, self._spans, self._span_of,
         self._places, self._tables) = cached

    def _build(self, free: set, patterns: list[int], width: int, height: int):
        if any(r in (0, height - 1) or c in (0, width - 1) for r, c in free):
            raise ValueError("Circuit Breaker level is not walled all the way "
                             "round; the int cell encoding assumes it is")
        code = {cell: cell[0] * width + cell[1] for cell in free}

        # One contiguous span of the state tuple per connector pattern.
        spans, span_of, at = [], [], 0
        for pattern in sorted(set(patterns)):
            count = patterns.count(pattern)
            spans.append((pattern, at, at + count))
            span_of.extend([(at, at + count)] * count)
            at += count

        places_raw = _placements(_goal_shapes(patterns), free)
        places = []
        for cells in places_raw:
            by_pattern: dict = {}
            for cell, pattern in cells.items():
                by_pattern.setdefault(pattern, []).append(code[cell])
            places.append(tuple(tuple(sorted(by_pattern[pattern]))
                                for pattern, _s, _e in spans))
        tables = {code[cell]: self._push_table(free, cell, code)
                  for cells in places_raw for cell in cells}
        steps = tuple((name, dr * width + dc)
                      for name, (dr, dc) in _MOVE_DELTA.items())
        return (width, frozenset(code.values()), steps, tuple(spans),
                tuple(span_of), tuple(places), tables)

    @staticmethod
    def _push_table(free: set, target: tuple[int, int], code: dict) -> dict:
        """``{cell: pushes to get a crate from cell onto target}``.

        Reverse BFS over push edges only, ignoring the other crates (the standard
        Sokoban relaxation): a crate arrives at ``X`` from ``X - d`` only when
        ``X - d`` is on the board and the player's stand cell ``X - 2d`` is too.
        A cell in NO table is a cell a crate can never leave -- see the deadlock
        note in the module docstring."""
        steps = {target: 0}
        queue = deque([target])
        while queue:
            r, c = queue.popleft()
            cost = steps[(r, c)]
            for dr, dc in _MOVE_DELTA.values():
                came = (r - dr, c - dc)           # where the crate was pushed from
                stand = (r - 2 * dr, c - 2 * dc)  # where the player stood
                if came in free and stand in free and came not in steps:
                    steps[came] = cost + 1
                    queue.append(came)
        return {code[cell]: cost for cell, cost in steps.items()}

    # -- heuristic (and, at 0, the win test) -----------------------------------
    def _estimate(self, cells: tuple, player: int | None = None) -> int:
        """Lower bound on the moves left, in the model's encoding.

        Zero EXACTLY at a win: the placements are the complete list of winning
        layouts, and a placement costs 0 iff every crate already sits on its cell
        in it. `_model_astar` leans on that and never runs a separate goal test.
        """
        tables, spans = self._tables, self._spans
        best = _DEAD
        for place in self._places:
            total = 0
            for targets, (_pattern, start, end) in zip(place, spans):
                if end - start == 1:
                    total += tables[targets[0]].get(cells[start], _DEAD)
                else:
                    total += self._assign(cells[start:end], targets, tables)
                if total >= best:
                    break
            else:
                best = total
                if best == 0:
                    return 0
        if best >= _DEAD or player is None:
            return best
        # The walk to the first push: the player pushes from the cell BESIDE a
        # crate, hence the -1. Some crate has to move, so this is a lower bound on
        # the moves before any push happens.
        width = self._width
        pr, pc = divmod(player, width)
        return best + max(0, min(abs(pr - c // width) + abs(pc - c % width)
                                 for c in cells) - 1)

    @staticmethod
    def _assign(sources, targets, tables) -> int:
        """Cheapest total push-distance matching one connector-pattern class of
        crates onto that class's cells in a placement.

        Exact (over k! permutations) for the small classes this game has; greedy
        nearest-first past `_EXACT_ASSIGN`, where exact stops being worth a
        per-node cost."""
        if len(targets) <= _EXACT_ASSIGN:
            best = _DEAD
            for order in permutations(sources):
                total = 0
                for target, src in zip(targets, order):
                    total += tables[target].get(src, _DEAD)
                    if total >= best:
                        break
                else:
                    best = total
            return best
        remaining = list(sources)
        total = 0
        for target in targets:
            table = tables[target]
            pick = min(range(len(remaining)),
                       key=lambda i: table.get(remaining[i], _DEAD))
            total += table.get(remaining.pop(pick), _DEAD)
        return min(total, _DEAD)

    def heuristic(self, eng) -> int:
        """`PSExpert`'s hook, reading the interpreter. The search uses
        `_estimate` on model states directly; this is what reports and any
        interpreter-side caller see."""
        return self._estimate(*self._read(eng))

    # -- the model -------------------------------------------------------------
    def _walks(self, cells: tuple, player: int) -> dict:
        """Shortest walks from the player over the cells the crates leave open,
        as a BFS tree of parent pointers -- materialised by `_walk_to` only for
        the handful of cells a push actually uses, since this runs once per
        expansion and building a path list per reachable cell would allocate the
        whole board's worth of them to use a few."""
        free, steps = self._free, self._steps
        blocked = set(cells)
        parent = {player: None}
        queue = deque([player])
        while queue:
            cur = queue.popleft()
            for name, delta in steps:
                nxt = cur + delta
                if nxt in free and nxt not in blocked and nxt not in parent:
                    parent[nxt] = (cur, name)
                    queue.append(nxt)
        return parent

    @staticmethod
    def _walk_to(parent: dict, cell: int) -> list[str]:
        out = []
        while parent[cell] is not None:
            cell, name = parent[cell]
            out.append(name)
        out.reverse()
        return out

    def _pushed(self, cells: tuple, index: int, dest: int) -> tuple:
        """``cells`` with crate ``index`` moved to ``dest``, re-sorted inside its
        connector-pattern span so interchangeable crates keep one encoding."""
        start, end = self._span_of[index]
        out = list(cells)
        out[index] = dest
        out[start:end] = sorted(out[start:end])
        return tuple(out)

    def _model_astar(self, cells: tuple, player: int,
                     weight: int, node_cap: int) -> list[str] | None:
        """A* over ``walk to the push cell, then push`` macros on the model.

        ``g`` counts primitive MOVES, the unit the agent pays, and a state is
        ``(crate layout, the player's exact cell)``.

        NOT the player's reachable REGION, which is the usual Sokoban
        canonicalisation and is what `PSPushExpert` uses. It is exact when moves
        are counted in pushes and an approximation when they are counted in
        moves: merging two states that differ only in where in the region the
        player stands mis-costs the next walk by up to the region's diameter.
        These boards are open enough for that to bite hard -- level 2 comes back
        69 presses under the region key against a true 59, and level 4 is 78
        against 76 -- for 1.6x the search time. So the player's cell stays in the
        key.

        The goal is tested when a node is POPPED, not when it is generated. The
        cheaper generation-time test returns whichever win the expansion order
        reached first, which is not the shortest -- it cost level 0 two presses.

        Paths are kept as a trace of (parent, segment) rather than a list per
        queued node: at these node counts, copying a growing plan into every
        child is most of the memory."""
        if self._estimate(cells) == 0:
            return []
        free, steps = self._free, self._steps
        trace: list[tuple[int, tuple]] = [(-1, ())]
        queue = [(weight * self._estimate(cells, player), 0, 0, cells, player,
                  0, 1)]
        best: dict = {(cells, player): 0}
        counter = 0
        nodes = 0
        while queue:
            _f, cost, _c, cells, player, node, left = heapq.heappop(queue)
            if left == 0:                         # `_estimate` is 0 only at a win
                return self._unroll(trace, node)
            if best.get((cells, player), 1 << 30) < cost:
                continue                          # superseded by a cheaper route
            parent = self._walks(cells, player)
            for index, cell in enumerate(cells):
                for name, delta in steps:
                    stand = cell - delta
                    dest = cell + delta
                    if (stand not in parent or dest not in free
                            or dest in cells):
                        continue     # unreachable, or the crate has nowhere to go
                    moved = self._pushed(cells, index, dest)
                    walk = self._walk_to(parent, stand)
                    step = cost + len(walk) + 1
                    if best.get((moved, cell), 1 << 30) <= step:
                        continue
                    guess = self._estimate(moved, cell)
                    if guess >= _DEAD:
                        continue                  # crate pushed somewhere terminal
                    best[(moved, cell)] = step
                    trace.append((node, tuple(walk) + (name,)))
                    nodes += 1
                    if nodes >= node_cap:
                        return None
                    counter += 1
                    heapq.heappush(queue, (step + weight * guess, step, counter,
                                           moved, cell, len(trace) - 1, guess))
        return None

    @staticmethod
    def _unroll(trace, node: int) -> list[str]:
        out: list[str] = []
        while node > 0:
            parent, segment = trace[node]
            out.extend(reversed(segment))
            node = parent
        out.reverse()
        return out

    # -- plan = search the model, then prove it on the interpreter -------------
    def _search(self, eng):
        """`PSExpert.plan`'s strategy hook. Search the model, then hand the plan
        to `_annotate`, which replays it through the REAL interpreter and refuses
        to return one the interpreter does not agree wins."""
        cells, player = self._read(eng)
        for weight in _WEIGHTS:
            found = self._model_astar(cells, player, weight, _NODE_CAP)
            if found is not None:
                plan = self._annotate(eng, found)
                plan.weight = weight
                return plan
        return None

    def _annotate(self, eng, found: list) -> Plan:
        """Replay ``found`` on the interpreter: verify the win, and attach a
        per-step optimal SET.

        A macro plan is walk, walk, ..., push, and the walks are where the ties
        are: any shortest route to the same stand cell leaves the board in the
        same state for the same cost, so every first step of one is equally
        optimal. Cut the plan at the steps that actually moved a crate, and run
        one BFS per walk segment from the stand cell it ends on.

        Pushes are labelled with the push taken. (A different push could in
        principle tie, but proving that needs a distance oracle over states this
        search does not have -- so the set stays a proven subset, never a guess.)

        The replay is also the model's per-plan audit; see the module docstring.
        """
        plan = Plan(found)
        crates = self._crates(eng)
        walk: list[int] = []                 # indices of the current walk segment
        players = []                         # player cell before each of them
        optsets: list[list[str]] = [[d] for d in found]
        for i, direction in enumerate(found):
            player = self._player_cell(eng)
            eng.step(direction)
            moved = self._crates(eng)
            if moved == crates:
                walk.append(i)
                players.append(player)
                continue
            self._label_walk(optsets, walk, players, player, crates)
            crates, walk, players = moved, [], []
        if not eng.check_win():
            raise AssertionError(
                "model plan did not win on the real interpreter -- the native "
                "model has drifted from the engine; run --verify-model")
        plan.optsets = optsets
        return plan

    def _label_walk(self, optsets, walk, players, stand, crates) -> None:
        """Fill in the tie sets for one walk segment ending at ``stand``."""
        if not walk:
            return
        width = self._width
        open_cells = self._free - {r * width + c for r, c in crates}
        dist = {stand[0] * width + stand[1]: 0}
        queue = deque(dist)
        while queue:
            cur = queue.popleft()
            for _name, delta in self._steps:
                nxt = cur + delta
                if nxt in open_cells and nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
        for step, (i, player) in enumerate(zip(walk, players)):
            togo = len(walk) - step
            here = player[0] * width + player[1]
            optsets[i] = [name for name, delta in self._steps
                          if dist.get(here + delta, 1 << 30) == togo - 1]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class CircuitBreakerSolver(PSAStarSolver):
    game_id = "puzzlescript_circuit_breaker"
    game_name = GAME_NAME
    expert_cls = CircuitBreakerExpert

    #: The rung the expert STARTS at; `_WEIGHTS` owns the escalation.
    weight = 1
    #: The model search has its own `_NODE_CAP`; this one is unused.
    node_cap = 0
    #: Plans are well inside the adapter's 200-press budget; the ceiling only has
    #: to cover a re-plan after an epsilon detour, and `epsilon` is 0 here.
    max_steps = 200

    def prepare_expert(self, game, expert) -> None:
        """Seed the expert's plan memo from `_PLAN_CACHE`, and refill the file
        with anything it was missing.

        Every cached plan is REPLAYED through the interpreter as it is loaded
        (`_annotate` does the walk anyway, to rebuild the tie sets), so a file
        left behind by an older version of a level cannot be used: it fails the
        win assertion and the level is searched again from scratch."""
        cache = {}
        if _PLAN_CACHE.exists():
            cache = json.loads(_PLAN_CACHE.read_text())
        fresh = dict(cache)

        def search(level, eng):
            found = expert.plan(eng, level)
            if found is not None:
                fresh[str(level)] = {"presses": list(found),
                                     "weight": found.weight}

        for level in range(game.n_levels):
            game.set_level(level)
            eng = game._engine
            expert._bind(eng)
            entry = cache.get(str(level))
            if not isinstance(entry, dict):
                fresh.pop(str(level), None)      # absent, or an older format
                search(level, eng)
                continue
            try:
                plan = expert._annotate(eng, entry["presses"])
                plan.weight = entry["weight"]
            except Exception:      # noqa: BLE001 -- see below
                # Deliberately broad, and only over the CACHED path: any way at
                # all in which a file on disk fails to replay into a win on this
                # level is a stale file, and the answer to all of them is the
                # same -- drop it and search the level again. A fresh search's
                # own `_annotate` is not wrapped, so a real bug still raises.
                game.set_level(level)
                fresh.pop(str(level), None)
                search(level, eng)
                continue
            game.set_level(level)
            expert.cache[(level, expert._key(eng))] = plan
        if fresh != cache:
            _PLAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
            _PLAN_CACHE.write_text(json.dumps(fresh, indent=1, sort_keys=True))

    def optimal_for(self, expert, level: int, plan, pi: int):
        """Every press that is equally optimal at this step -- see `_annotate`.
        Falls back to the press about to be taken, so no expert step ever ships
        unlabelled (the always-emit-optimal-targets rule)."""
        sets = getattr(plan, "optsets", None)
        if sets is not None and pi < len(sets) and sets[pi]:
            return sets[pi]
        return [plan[pi]] if pi < len(plan) else None


def _verify_model(presses: int = 4000, seed: int = 0) -> int:
    """Fuzz the native model against the real interpreter.

    The search never touches the interpreter, so "the model IS the mechanic" has
    to be checked rather than argued. This walks each level through random
    presses and, after every one, compares what the model predicts against what
    the engine did: the crate layout with its connector patterns, the player's
    cell, and the win flag."""
    solver = CircuitBreakerSolver()
    game = solver.make_game(0)
    expert = CircuitBreakerExpert(game)
    rng = random.Random(seed)
    checked = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        expert._bind(eng)
        width = expert._width
        cells, player = expert._read(eng)
        for _ in range(presses):
            name = rng.choice(_MOVES)
            delta = dict(expert._steps)[name]

            # What the model says.
            ahead = player + delta
            if ahead not in expert._free:
                model = (cells, player)                     # into a wall
            elif ahead in cells:
                index = cells.index(ahead)
                dest = ahead + delta
                if dest in expert._free and dest not in cells:
                    model = (expert._pushed(cells, index, dest), ahead)
                else:
                    model = (cells, player)                 # the push is refused
            else:
                model = (cells, ahead)

            eng.step(name)
            truth = expert._read(eng)
            if model != truth:
                print(f"MISMATCH level {level} press {name}: "
                      f"model {model} != engine {truth}")
                return 1
            if (expert._estimate(truth[0]) == 0) != bool(eng.check_win()):
                print(f"WIN MISMATCH level {level} at {truth}")
                return 1
            cells, player = truth
            checked += 1
        # A level cannot be re-entered mid-fuzz once it is won, and random play
        # will not win these, but re-seat it anyway so the next level starts
        # clean.
        game.set_level(level)
    print(f"ok: {checked} presses over {game.n_levels} levels -- the model "
          f"matched the interpreter on every one (layout, player and win)")
    return 0


def _report() -> int:
    """Per-level report: how many winning layouts the goal model found, the
    heuristic's opening estimate against the plan it actually took, the `_WEIGHTS`
    rung that found it, and how many steps carry a real tie set."""
    solver = CircuitBreakerSolver()
    game = solver.make_game(0)
    expert = CircuitBreakerExpert(game, weight=CircuitBreakerSolver.weight)
    solver.prepare_expert(game, expert)
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        expert._bind(game._engine)         # `heuristic` reads this level's tables
        crates = expert._crates(game._engine)
        shapes = _goal_shapes(sorted(crates.values()))
        opening = expert.heuristic(game._engine)
        plan = expert.plan(game._engine, level)
        if plan is None:
            print(f"level {level}: {len(crates)} crates, NO PLAN")
            continue
        ties = sum(1 for s in (plan.optsets or []) if len(s) > 1)
        total += len(plan)
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"level {level}: {len(crates)} crates, {len(shapes)} shape(s) x "
              f"{len(expert._places)} placement(s), h0 {opening:3d} -> "
              f"{len(plan):3d} presses at weight {plan.weight or '?'} "
              f"(budget {game._max_steps}, {room}), "
              f"{ties:3d} steps with a tie set ({ties / max(1, len(plan)):.0%})")
    print(f"total {total} presses over {game.n_levels} levels")
    return 0


if __name__ == "__main__":
    if "--verify-model" in sys.argv:
        sys.exit(_verify_model())
    if "--report" in sys.argv:
        sys.exit(_report())
    sys.exit(CircuitBreakerSolver.main())
