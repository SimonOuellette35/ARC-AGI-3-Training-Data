"""Generate Phase-1 training data for the PuzzleScript game ps:l_a_s_e_r
("L.A.S.E.R.", Franklin P. Dyer).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
its disk cache, and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: a native model of the mechanic (fuzz-verified against
the interpreter), a walk-and-push MACRO search over it, the optimal-action
oracle, and the level work the shipped file needed.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_l.a.s.e.r",
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
exactly. Every expert step also carries the full set of equally-optimal presses.

The game
--------
Sokoban with laser emitters. ``All target on box`` + ``No player on zap`` +
``Some player``: cover every target with a box, and do not be standing in a
beam when you do it. The whole rule set is eight lines, and four facts fall out
of them that a source-read gets wrong:

  * **The emitters are PUSHABLE.** ``[> Player|Laser] -> [> Player|> Laser]``
    sits right beside the box-pushing rule, and ``Laser`` is on the same
    collision layer as ``Wall, Player, Box``. So an emitter shoves exactly like
    a crate -- one cell, never in a chain -- and shoving it moves its whole
    beam onto another row or column. Half the later levels are about that, and
    a solver that treats the emitters as scenery cannot win level 2 onwards.
  * **Beams pass THROUGH the player and stop at everything else.** The
    propagation rules are guarded by ``No Obstacle`` with
    ``Obstacle = Wall or Laser or Box`` -- the player is not in it. So a beam
    is a plain ray cast from the emitter that halts *before* the first wall,
    box or other emitter, and standing in one does not interrupt it. Boxes and
    emitters are therefore the only shields.
  * **Standing in a beam kills you on the NEXT press, not this one.** The zaps
    are recomputed from scratch in the late rules of every turn (``[Zap] -> []``
    then the six propagation rules), and the very first main rule of the next
    turn is ``[Player Zap] -> [Zap]``, which deletes the player before anything
    it pressed can matter. There is no escaping and no blocking the beam from
    inside it. Since ``No player on zap`` is also a win condition, the rule for
    a solver is simply: **no press may end with the player on a zap**, ever --
    which makes every walk a shortest path through the *unlit* cells rather
    than through the free ones. ``--fuzz`` checks it in both directions at every
    one of 17441 presses; it is the one mechanic that is invisible in a single
    frame's rule list and fatal to miss.
  * **A blocked press is a total no-op** -- the engine reports no movement, the
    adapter does not count the step, and the zaps are unchanged. There is no
    wait move, so a level's timing cannot be adjusted by bumping a wall. (It
    does not matter here, because nothing in this game moves on its own, but it
    is worth knowing before assuming it does.)

``run_rules_on_level_start on`` means the beams are already lit on the level's
first frame, and no shipped level starts the player inside one.

The levels
----------
Twenty, in three named stages, and **all twenty are winnable and recorded**:

    level  size   boxes targets emitters  plan  ties  search  states
    0       5x5     1      1       0         5   20%  field       13
    1       7x6     3      2       1        22    9%  field      269
    2       7x7     2      2       1        15    0%  field      638
    3       7x7     2      2       1        25   12%  field    20478
    4       7x7     3      1       2        18   11%  field     3696
    5       6x4     1      1       1         7    0%  field        8
    6       8x8     3      3       2        46    4%  A*      235815   <- stage 2
    7       8x8     1      1       2        55    3%  field     1135
    8       8x9     2      2       1        26    0%  field       50
    9       9x10    4      2       2        39    7%  field     5658
    10      7x9     5      2       2        20   10%  field      136
    11      7x8     5      2       2        10    0%  field     2631
    12      9x10    4      2       3        32    3%  field      651
    13      9x10    4      3       3        33    3%  field    20014
    14      9x9     3      2       4        47    4%  A*      520800   <- stage 3
    15      7x8     2      2       1        29    3%  field      446
    16      8x6     3      3       1        30    0%  field    11487
    17     11x11    5      3       2       184    8%  A*      347277
    18     11x13    7      4       3        68    1%  A*      874414
    19      9x9     5      3       1        56    3%  A*       72005

``plan`` is proved SHORTEST -- see "Expert solver" -- ``ties`` is the share of
plan steps with more than one equally-optimal press, and ``states`` is the macro
states the search enumerated or settled. Level 17's 184 presses is over the
adapter's stock 200-step budget once an agent wanders at all, and level 18's 68
is uncomfortably close, so `games/ps:l_a_s_e_r` raises the limit
on those two the way `games/ps:cancel` and `games/ps:bridge` do; this generator
records through that wrapper (``game_module_id``), i.e. against exactly the
adapter a live agent plays.

The art
-------
`data/puzzlescript_games/L.A.S.E.R.txt` ships a monochrome dark-red palette in
which wall, player, box, target and emitter ALL quantize to one ARC index, and
1-pixel-thick sprites that vanish at the 4px-per-cell the biggest level renders
at. Three frames the game is *about* were pixel-identical: player-in-a-beam vs
player-safe (the death frame), box-on-target vs box-on-floor (the win frame),
and target vs bare floor at 4px. The sprites and colours are re-drawn in that
file, with the reasoning in a comment beside them; ``--audit`` re-checks all 21
stacking compositions at all eight cell sizes the twenty levels use.

Expert solver
-------------
A native model (`_Board`), because the interpreter is ~500 presses/s and these
searches need millions. It is the whole rule set as a state machine over
``(player cell, sorted box cells, sorted emitter cells+facings)``: walls and
targets are static, and the zaps are a pure function of the boxes and emitters,
so nothing else is carried. ``--fuzz`` proves it against the interpreter -- 17441
random presses across all twenty levels (291 of which end in a death),
comparing the pieces, both zap layers, the win predicate and the death rule,
with zero divergences.

Search is over **walk-and-push macros**, not primitive presses. A press that
does not push changes nothing but the player's cell (beams pass through the
player), so a shortest primitive plan is exactly a sequence of "walk the cheapest
unlit route to a pushing stance, then push", and a macro edge costs
``walk length + 1``. That turns level 17's 184-press search into a 42-push one.
The equivalence is not assumed: every level's macro answer was checked against an
independent primitive breadth-first search over the same model, and all twenty
agree exactly (5, 22, 15, 25, 18, 7, 46, 55, 26, 39, 20, 10, 32, 33, 47, 29, 30,
184, 68, 56 = 767 presses).

Two tiers, because the levels differ by three orders of magnitude:

  * **The exact field** (`_Board.field`), for the fifteen levels whose macro
    graph closes inside `FIELD_CAP`: enumerate every reachable macro state and
    run a backward Dijkstra from the winning ones. That labels the true
    distance-to-win of *every* state, which gives a proved-shortest plan and,
    for free, the EXACT optimal-press set at every step (see below).
  * **A-star** (`_Board.macro_astar`), for levels 6, 14, 17, 18 and 19, whose macro
    graphs run to hundreds of thousands of states. The heuristic is a lower
    bound on the number of pushes still owed: a minimum-cost assignment of each
    target to a distinct box, priced by a per-target reverse push-distance table
    that ignores everything but walls. It is admissible (one push is one press),
    it prunes cornered boxes outright (a box with no route to any target is
    scored infinite), and every plan it returns is therefore shortest.

Optimal-action sets
-------------------
Never None -- every recorded step ships a set.

On the field levels they are exact: the distance-to-win of any state is
recovered from the macro field by one multi-source Dijkstra over the unlit cells
of its configuration (`_Board.config_field`), and a press is optimal iff it
lands on distance ``d - 1``. A refused press maps the state to itself and is
excluded by construction; a press into a beam has no distance at all and is
excluded too.

On the five A* levels they are the walk sets: every direction that keeps the
player on *a* shortest unlit route to the stance the plan is walking to is
labelled, and the push itself is labelled alone. That is sound (all of those
presses lie on a shortest plan of the same length) but it under-labels -- a
genuinely different push ORDER of the same length is not found. `--verify`
re-derives every label on the field levels from the other end, with an
independent forward search from each successor that shares no bookkeeping with
the backward sweep.

Augmentation
------------
This game's engine state after reset is identical for every seed (levels are
fixed ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`, which this game was added to). 20 levels x
16 presentations = 320.

The flips are sound and, because the six propagation rules are written out per
DIRECTION (a ``LATE RIGHT`` block, a ``LATE HORIZONTAL`` block, ...) rather than
with the relative ``>`` force -- the exact shape that hid a chirality in
ps:gobble_rush -- they are MEASURED rather than argued: ``--symmetry`` replays
each level's plan on all eight turned and mirrored copies of its own board and
requires every press to land the pieces where the transform says. There is
deliberately no colour augmentation: the re-drawn palette is what separates box
from target from player from beam, and a flattening recolor would put two of
them back on the same index.

Usage (run from the repo root):
    python solvers/generate_l_a_s_e_r_training.py \
        --episodes 200 --out data/training_multi_level/puzzlescript_l.a.s.e.r

    python solvers/generate_l_a_s_e_r_training.py --plans
    python solvers/generate_l_a_s_e_r_training.py --verify
    python solvers/generate_l_a_s_e_r_training.py --fuzz
    python solvers/generate_l_a_s_e_r_training.py --symmetry
    python solvers/generate_l_a_s_e_r_training.py --audit
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame                # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert, Plan      # noqa: E402

GAME_NAME = "L.A.S.E.R"
GAME_MODULE = "ps:l_a_s_e_r"

#: Fixed direction order, so a plan (and its tie sets) is reproducible.
_ORDER: tuple[str, ...] = ("up", "down", "left", "right")
#: Index into `_ORDER` -> (dr, dc). Emitter facings use the same numbering.
_DELTA: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
#: Index of the opposite direction (which is where the pusher has to stand).
_OPP: tuple[int, ...] = (1, 0, 3, 2)
#: PuzzleScript object name of each emitter facing, in `_ORDER` order.
_LASERS: tuple[str, ...] = ("laserup", "laserdown", "laserleft", "laserright")

#: Enumerate a level's whole macro graph only when it closes inside this many
#: states; past it `_Board.plan` falls back to A*. It is a MEMORY budget -- the
#: closure holds every state's edge list -- and the split it produces is sharp:
#: the fifteen field levels top out at 20 478 states, and the next one up is
#: 930 845.
FIELD_CAP = 40_000

#: A* budget for the five levels past `FIELD_CAP`. The worst of them (level 18)
#: settles ~890k states, so this is a guard against an edited level rather than
#: a live limit; `--plans` prints what each one actually used.
ASTAR_CAP = 1_500_000

_INF = float("inf")


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """The whole of L.A.S.E.R.'s rule set, as a state machine over

        ``(player cell, sorted box cells, sorted (emitter cell, facing) pairs)``

    Cells are flat ``r * w + c`` integers: the searches hold millions of states
    and a tuple of ints is a fraction of the memory of a tuple of pairs.

    Walls and targets are static -- no rule in the game creates or destroys
    either -- so they live on the board, not in the state. Neither zap layer is
    carried: ``[Zap] -> []`` wipes both at the top of every turn and the late
    rules rebuild them from the emitters, so they are a pure function of the
    boxes and the emitters and would only alias the key.
    """

    def __init__(self, h, w, walls, targets, player, boxes, lasers):
        self.h, self.w = h, w
        self.walls = walls
        self.targets = targets
        self.start = (player, tuple(sorted(boxes)), tuple(sorted(lasers)))
        # Neighbour tables: ``self.nbr[d][cell]`` is the cell one step in
        # direction d, or -1 off the board. Built once; the inner loops of the
        # search do nothing but index them.
        self.nbr = []
        for dr, dc in _DELTA:
            table = [-1] * (h * w)
            for r in range(h):
                for c in range(w):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        table[r * w + c] = rr * w + cc
            self.nbr.append(table)
        self.around = [[t[p] for t in self.nbr if t[p] >= 0] for p in range(h * w)]
        self._cfg: dict = {}

    @classmethod
    def read(cls, eng, ids) -> "_Board":
        """Build the model from the interpreter's current grid."""
        wall_id, target_id, box_id, player_id, laser_ids = ids
        w = eng.width
        walls, targets, boxes, lasers, player = set(), set(), [], [], None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                p = r * w + c
                if wall_id in cell:
                    walls.add(p)
                if target_id in cell:
                    targets.add(p)
                if box_id in cell:
                    boxes.append(p)
                if player_id in cell:
                    player = p
                for d, lid in enumerate(laser_ids):
                    if lid in cell:
                        lasers.append((p, d))
        return cls(eng.height, w, frozenset(walls), frozenset(targets),
                   player, boxes, lasers)

    # -- derived state --------------------------------------------------------
    def cfg(self, boxes, lasers) -> tuple[frozenset, frozenset]:
        """``(blocked cells, zapped cells)`` for one box/emitter configuration.

        ``blocked`` is the shared ``Wall, Player, Laser, Box`` collision layer
        minus the player, i.e. what the player cannot walk into and what a beam
        stops before. ``zapped`` is every cell either zap layer reaches: a plain
        ray cast per emitter, since ``Obstacle`` does not name the player and
        the two layers never interfere (they are separate collision layers, so
        a horizontal and a vertical beam simply share the cell).

        Memoized on the configuration because a walk visits many player cells
        without changing it, and the searches ask for it constantly."""
        key = (boxes, lasers)
        got = self._cfg.get(key)
        if got is None:
            blocked = self.walls | frozenset(boxes) | frozenset(p for p, _ in lasers)
            zap = set()
            for p, d in lasers:
                table = self.nbr[d]
                q = table[p]
                while q >= 0 and q not in blocked:
                    zap.add(q)
                    q = table[q]
            got = (blocked, frozenset(zap))
            self._cfg[key] = got
        return got

    def won(self, state) -> bool:
        """``All target on box`` and ``No player on zap`` and ``Some player``.

        A state whose player is on a zap is a corpse one press later, and every
        caller keeps those out of the search entirely (`settle` and `macros`
        refuse them, `distance` gives them no value), so by the time anything
        asks whether a state is won the box test is the whole of it."""
        return self.targets.issubset(state[1])

    # -- dynamics -------------------------------------------------------------
    def step(self, state, d):
        """One press, in direction index ``d``. Returns the settled state, or
        ``state`` itself when the press is refused (a no-op the engine does not
        even count).

        The push is the plain sokoban one -- the destination has to be clear of
        the whole ``Wall, Player, Laser, Box`` layer, and there is no chaining,
        because the rule only ever matches ONE object next to the player. What
        is not plain sokoban is that an emitter is pushed by the same rule as a
        box, so the two cases differ only in which part of the state is
        rewritten."""
        player, boxes, lasers = state
        nxt = self.nbr[d][player]
        if nxt < 0 or nxt in self.walls:
            return state
        pushed_box = nxt in boxes
        pushed_laser = not pushed_box and any(p == nxt for p, _ in lasers)
        if pushed_box or pushed_laser:
            dest = self.nbr[d][nxt]
            if dest < 0:
                return state
            blocked, _zap = self.cfg(boxes, lasers)
            if dest in blocked:
                return state
            if pushed_box:
                boxes = tuple(sorted([x for x in boxes if x != nxt] + [dest]))
            else:
                lasers = tuple(sorted((dest, fd) if p == nxt else (p, fd)
                                      for p, fd in lasers))
        return (nxt, boxes, lasers)

    def lethal(self, state) -> bool:
        """True when this settled state deletes the player on the next press.

        ``[Player Zap] -> [Zap]`` is the FIRST main rule of every turn, so the
        press that would escape never runs. It is also why the state is not a
        win: ``No player on zap`` fails. Either way it is a dead end, and the
        searches drop it rather than modelling the corpse."""
        _blocked, zap = self.cfg(state[1], state[2])
        return state[0] in zap

    def settle(self, state, d):
        """`step` plus the lethality guard: the successor a search may take, or
        None for "refused, or fatal"."""
        nxt = self.step(state, d)
        if nxt == state or self.lethal(nxt):
            return None
        return nxt

    # -- walks and macros -----------------------------------------------------
    def walk(self, player, boxes, lasers) -> dict:
        """``{cell: presses}`` for every cell the player can reach WITHOUT
        pushing anything, over the unlit free cells of this configuration.

        Every step of a walk ends a turn, so an unlit route is not a nicety --
        one lit cell anywhere along it is a death. The beams cannot change while
        the player only walks (nothing it steps on is an ``Obstacle``), so the
        forbidden set is fixed for the whole walk and this is one plain BFS."""
        blocked, zap = self.cfg(boxes, lasers)
        dist = {player: 0}
        queue = deque([player])
        while queue:
            p = queue.popleft()
            step = dist[p] + 1
            for q in self.around[p]:
                if q in blocked or q in zap or q in dist:
                    continue
                dist[q] = step
                queue.append(q)
        return dist

    def macros(self, state) -> list:
        """Every ``(cost, next state, stance cell, direction)`` reachable from
        ``state`` by walking and then pushing once.

        This is the search's whole branching factor. It is exact rather than an
        abstraction: a press that pushes nothing leaves the board identical (the
        beams pass through the player), so any shortest primitive plan
        decomposes into exactly these, and the cheapest walk to a stance is what
        `walk` returns."""
        player, boxes, lasers = state
        blocked, _zap = self.cfg(boxes, lasers)
        dist = self.walk(player, boxes, lasers)
        out = []
        lmap = dict(lasers)
        for cell in itertools.chain(boxes, lmap):
            for d in range(4):
                stance = self.nbr[_OPP[d]][cell]
                if stance < 0 or stance not in dist:
                    continue
                dest = self.nbr[d][cell]
                if dest < 0 or dest in blocked:
                    continue
                if cell in lmap:
                    nxt = (cell, boxes,
                           tuple(sorted((dest, fd) if p == cell else (p, fd)
                                        for p, fd in lasers)))
                else:
                    nxt = (cell,
                           tuple(sorted([x for x in boxes if x != cell] + [dest])),
                           lasers)
                # The push leaves the player in the cell the object vacated; if
                # a beam now crosses it the player is dead next press.
                _nb, nzap = self.cfg(nxt[1], nxt[2])
                if cell in nzap:
                    continue
                out.append((dist[stance] + 1, nxt, stance, d))
        return out

    # -- the exact distance-to-win field --------------------------------------
    def field(self, cap: int = FIELD_CAP):
        """``({macro state: presses to the win}, stats)``, or ``(None, stats)``
        when the macro graph does not close inside ``cap`` states.

        Two passes. The first enumerates every macro state reachable from the
        start (a macro state is the board right after a push, plus the start),
        keeping each one's edge list. The second is a backward Dijkstra from the
        winning states over those edges -- Dijkstra rather than BFS because a
        macro edge costs its walk, not 1.

        The result is the exact distance-to-win of every macro state, which is
        what makes both the plan and the optimal-press sets exact; `config_field`
        extends it to the walk states in between."""
        adjacency = {self.start: None}
        stack = [self.start]
        edges = 0
        while stack:
            state = stack.pop()
            if adjacency[state] is not None:
                continue
            out = self.macros(state)
            adjacency[state] = out
            edges += len(out)
            for _w, nxt, _s, _d in out:
                if nxt not in adjacency:
                    adjacency[nxt] = None
                    stack.append(nxt)
            if len(adjacency) > cap:
                return None, {"states": len(adjacency), "edges": edges,
                              "capped": True}

        preds: dict = {}
        for state, out in adjacency.items():
            for w, nxt, _s, _d in out:
                preds.setdefault(nxt, []).append((w, state))
        dist = {s: 0 for s in adjacency if self.won(s)}
        queue = [(0, i, s) for i, s in enumerate(dist)]
        heapq.heapify(queue)
        tie = len(queue)
        while queue:
            d, _t, state = heapq.heappop(queue)
            if d > dist[state]:
                continue
            for w, prev in preds.get(state, ()):
                if d + w < dist.get(prev, _INF):
                    dist[prev] = d + w
                    tie += 1
                    heapq.heappush(queue, (dist[prev], tie, prev))
        return dist, {"states": len(adjacency), "edges": edges,
                      "wins": sum(1 for s in adjacency if self.won(s)),
                      "capped": False}

    def config_field(self, boxes, lasers, macro_dist: dict) -> dict:
        """``{player cell: presses to the win}`` for one box/emitter
        configuration, given the macro field.

        A state that is not already won is worth ``walk to some stance`` + 1 +
        ``the macro state that push leads to``. Pricing every stance first and
        then running ONE multi-source Dijkstra outward over the unlit cells
        answers that for every player position in the configuration at once,
        which is what makes the per-step optimal sets cheap."""
        blocked, zap = self.cfg(boxes, lasers)
        best: dict = {}
        lmap = dict(lasers)
        for cell in itertools.chain(boxes, lmap):
            for d in range(4):
                stance = self.nbr[_OPP[d]][cell]
                if stance < 0 or stance in blocked or stance in zap:
                    continue
                dest = self.nbr[d][cell]
                if dest < 0 or dest in blocked:
                    continue
                if cell in lmap:
                    nxt = (cell, boxes,
                           tuple(sorted((dest, fd) if p == cell else (p, fd)
                                        for p, fd in lasers)))
                else:
                    nxt = (cell,
                           tuple(sorted([x for x in boxes if x != cell] + [dest])),
                           lasers)
                _nb, nzap = self.cfg(nxt[1], nxt[2])
                if cell in nzap:
                    continue
                value = macro_dist.get(nxt)
                if value is None:
                    continue
                if 1 + value < best.get(stance, _INF):
                    best[stance] = 1 + value

        dist = dict(best)
        queue = [(v, k) for k, v in best.items()]
        heapq.heapify(queue)
        while queue:
            d, p = heapq.heappop(queue)
            if d > dist[p]:
                continue
            for q in self.around[p]:
                if q in blocked or q in zap:
                    continue
                if d + 1 < dist.get(q, _INF):
                    dist[q] = d + 1
                    heapq.heappush(queue, (d + 1, q))
        return dist

    def distance(self, state, macro_dist: dict, cache: dict):
        """Exact presses-to-win for ANY state (walking or just-pushed), or None.

        The cache is keyed on the configuration, so the four successors of a
        plan step share one `config_field`."""
        if self.lethal(state):
            return None
        if self.won(state):
            return 0
        key = (state[1], state[2])
        got = cache.get(key)
        if got is None:
            got = self.config_field(state[1], state[2], macro_dist)
            cache[key] = got
        value = got.get(state[0])
        return None if value is None or value == _INF else value

    def plan_from_field(self, macro_dist: dict):
        """``(presses, optsets)`` walked downhill from the start, or
        ``(None, None)`` when the level cannot be won.

        At a state of distance ``d`` the optimal set is exactly the presses
        reaching ``d - 1``. A refused press maps the state to itself (distance
        ``d``, excluded); a press into a beam has no distance at all (excluded);
        a press that throws the level away lands outside the field (excluded).
        So the set never needs a special case and never ships empty."""
        cache: dict = {}
        state = self.start
        here = self.distance(state, macro_dist, cache)
        if here is None:
            return None, None
        presses, optsets = [], []
        while here:
            best = []
            for d in range(4):
                nxt = self.step(state, d)
                if nxt == state:
                    continue
                if self.distance(nxt, macro_dist, cache) == here - 1:
                    best.append(d)
            if not best:                       # unreachable: the field is exact
                raise AssertionError("field is inconsistent at distance "
                                     f"{here}")
            presses.append(best[0])
            optsets.append(best)
            state = self.step(state, best[0])
            here -= 1
        return presses, optsets

    # -- A*, for the levels whose macro graph does not close ------------------
    def push_tables(self) -> dict:
        """``{target: {cell: pushes to get a box from cell onto target}}``,
        ignoring every box, emitter and beam -- only walls.

        A reverse BFS from each target: a box reaches ``p`` from ``src = p - d``
        if the pusher's cell ``src - d`` is also not a wall. Relaxing away the
        other pieces is what makes it a LOWER bound, and it prunes corner
        deadlocks for free -- a box in a corner is in no table at all, because
        every way out needs the pusher inside a wall."""
        tables = {}
        for target in self.targets:
            dist = {target: 0}
            queue = deque([target])
            while queue:
                p = queue.popleft()
                for d in range(4):
                    src = self.nbr[_OPP[d]][p]
                    if src < 0 or src in self.walls or src in dist:
                        continue
                    stance = self.nbr[_OPP[d]][src]
                    if stance < 0 or stance in self.walls:
                        continue
                    dist[src] = dist[p] + 1
                    queue.append(src)
            tables[target] = dist
        return tables

    def make_heuristic(self):
        """An admissible lower bound on the presses still owed, from the boxes
        alone: the minimum-cost way of assigning each target a DISTINCT box,
        priced by `push_tables`.

        Admissible because one push is one press and the tables ignore every
        obstacle but walls, so the true plan cannot be shorter. It says nothing
        about walking, which is most of the cost on the big levels -- but it is
        exactly what keeps the A* answers provably shortest, and it returns
        infinity (a hard prune) as soon as some target has no box that can still
        reach it."""
        tables = self.push_tables()
        targets = sorted(self.targets)
        n = len(targets)
        full = 1 << n
        big = 1 << 20

        def heuristic(boxes) -> int:
            if self.targets.issubset(boxes):
                return 0
            cost = [[tables[t].get(b, big) for b in boxes] for t in targets]
            row = [big] * full
            row[0] = 0
            for j in range(len(boxes)):
                nxt = row[:]
                for mask in range(full):
                    base = row[mask]
                    if base >= big:
                        continue
                    for i in range(n):
                        if mask >> i & 1:
                            continue
                        value = base + cost[i][j]
                        if value < nxt[mask | 1 << i]:
                            nxt[mask | 1 << i] = value
                row = nxt
            return row[full - 1]

        return heuristic, big

    def macro_astar(self, cap: int = ASTAR_CAP):
        """``(macro path, stats)`` -- the shortest walk-and-push sequence to a
        win, or ``(None, stats)`` if the level cannot be won inside ``cap``
        states.

        Plain A* with `make_heuristic`, reopening on a better ``g`` so an
        inconsistent bound cannot cost optimality, and testing the goal when a
        state is POPPED rather than when it is discovered -- discovering a win
        only bounds it from above, and doing it the wrong way round quietly
        returned 48 instead of 46 on level 6."""
        heuristic, big = self.make_heuristic()
        start = self.start
        if self.won(start):
            return [], {"states": 0, "pops": 0}
        best = {start: 0}
        parent: dict = {}
        queue = [(heuristic(start[1]), 0, start)]
        tie = pops = 0
        while queue:
            _f, _t, state = heapq.heappop(queue)
            g = best[state]
            if self.won(state):
                path = []
                while state in parent:
                    prev, stance, d = parent[state]
                    path.append((stance, d, state))
                    state = prev
                path.reverse()
                return path, {"states": len(best), "pops": pops}
            pops += 1
            for w, nxt, stance, d in self.macros(state):
                step = g + w
                if step < best.get(nxt, 1 << 30):
                    h = heuristic(nxt[1])
                    if h >= big:                  # a target is now unreachable
                        continue
                    best[nxt] = step
                    parent[nxt] = (state, stance, d)
                    tie += 1
                    heapq.heappush(queue, (step + h, tie, nxt))
            if len(best) > cap:
                return None, {"states": len(best), "pops": pops, "capped": True}
        return None, {"states": len(best), "pops": pops}

    def flatten(self, path, start=None):
        """``(presses, optsets, owners)`` for a macro path: the walks spelled
        out, each step labelled with every direction that keeps the player on A
        shortest unlit route to the same stance, and ``owners[i]`` the index of
        the macro press ``i`` belongs to.

        Sound but not complete -- it cannot see that a different push ORDER of
        the same length exists, which is why the levels that can afford the
        field use it instead. Nothing here is guessed from the macro boundary:
        each label is re-derived by measuring the distance to the stance from
        the state actually reached.

        ``start`` re-enters the path from an arbitrary state (only the player
        may differ -- the pushes still have to be the ones the path names); it
        is how `--verify` rebuilds the plan a labelled alternative leads to,
        without re-running the search."""
        state = self.start if start is None else start
        presses, optsets, owners = [], [], []
        for index, (stance, d, expect) in enumerate(path):
            to_stance = self.walk(stance, state[1], state[2])
            while state[0] != stance:
                left = to_stance[state[0]]
                best = []
                for cand in range(4):
                    nxt = self.step(state, cand)
                    if nxt == state or nxt[1] != state[1] or nxt[2] != state[2]:
                        continue          # refused, or a push: not this walk
                    if to_stance.get(nxt[0], _INF) == left - 1:
                        best.append(cand)
                presses.append(best[0])
                optsets.append(best)
                owners.append(index)
                state = self.step(state, best[0])
            presses.append(d)
            optsets.append([d])
            owners.append(index)
            state = self.step(state, d)
            assert state == expect, "macro path diverged from the model"
        return presses, optsets, owners

    # -- the one entry point --------------------------------------------------
    def solve(self):
        """``(presses, optsets, stats)``. Exact field where it closes, A* where
        it does not; both give a shortest plan.

        ``stats`` also carries the macro ``path`` and per-press ``owners`` on
        the A* tier, which is all `--verify` needs to rebuild a variant plan."""
        macro_dist, stats = self.field()
        if macro_dist is not None:
            presses, optsets = self.plan_from_field(macro_dist)
            stats["mode"] = "field"
            return presses, optsets, stats
        path, stats = self.macro_astar()
        stats["mode"] = "astar"
        if path is None:
            return None, None, stats
        presses, optsets, owners = self.flatten(path)
        stats["path"], stats["owners"] = path, owners
        return presses, optsets, stats


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class LaserExpert(PSExpert):
    """Plans on `_Board` -- built off the interpreter's grid, then never
    stepping the interpreter again.

    `PSExpert` still owns everything around that: the in-memory memo, the level
    scoping, the snapshot/restore discipline and the on-disk plan cache. The
    only override is `_search`, so `heuristic` (the shared primitive A*'s hook)
    is unreachable here by construction."""

    #: `_key` below is dynamic-objects-only, canonical only WITHIN a level.
    scope_by_level = True
    #: No rule in the game reads ACTION5 -- it is a pure no-op, measured in
    #: ``--fuzz`` -- so the four moves are the whole action space.
    directions = ["up", "down", "left", "right"]
    #: Level 18's search settles ~890k macro states; caching the twenty START
    #: plans keeps every later process (and every `parallelize_generator` shard)
    #: at ~0.1s instead of ~2 minutes.
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "l_a_s_e_r_plans.json")

    def setup(self) -> None:
        names = self.g.obj_name_to_idx
        self.ids = (names["wall"], names["target"], names["box"],
                    names["player"], tuple(names[n] for n in _LASERS))
        self.zap_ids = (names["horizap"], names["vertzap"])

    def _key(self, eng) -> frozenset:
        """Player, boxes and emitters (with the facing). Walls and targets are
        static per level -- hence ``scope_by_level`` -- and the two zap layers
        are recomputed from scratch every turn, so including them would only
        make the same board look like several."""
        _wall_id, _target_id, box_id, player_id, laser_ids = self.ids
        keep = {box_id: 1, player_id: 2}
        keep.update({lid: 3 + d for d, lid in enumerate(laser_ids)})
        return frozenset(
            (r, c, tag)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for obj, tag in keep.items()
            if obj in cell
        )

    def heuristic(self, eng) -> int:
        raise AssertionError("this expert never runs the shared primitive A*")

    def board(self, eng) -> _Board:
        return _Board.read(eng, self.ids)

    def _search(self, eng) -> list | None:
        presses, optsets, _stats = self.board(eng).solve()
        if presses is None:
            return None
        return Plan([_ORDER[d] for d in presses],
                    [[_ORDER[d] for d in s] for s in optsets])


class LaserSolver(PSAStarSolver):
    game_id = "puzzlescript_l.a.s.e.r"
    game_name = GAME_NAME
    expert_cls = LaserExpert
    #: `games/ps:l_a_s_e_r` raises the per-level step budget on the two levels
    #: whose shortest plan does not comfortably fit the adapter's stock 200, so
    #: the generator has to record against THAT adapter, not a bare one.
    game_module_id = GAME_MODULE

    #: The longest plan is 184 presses; the rest is room for the exploration
    #: prefix and the re-plan after it.
    max_steps = 400


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _levels():
    """``(solver, game, expert)`` with the adapter parked at level 0."""
    solver = LaserSolver()
    game = solver.make_game(0)
    return solver, game, LaserExpert(game)


def _pieces(eng, expert) -> tuple:
    """The dynamic pieces, read off the interpreter: ``(player, boxes,
    emitters)``, in (row, col) form so a presentation transform can be applied
    to it."""
    _wall_id, _target_id, box_id, player_id, laser_ids = expert.ids
    boxes, lasers, player = set(), set(), None
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if box_id in cell:
                boxes.add((r, c))
            if player_id in cell:
                player = (r, c)
            for d, lid in enumerate(laser_ids):
                if lid in cell:
                    lasers.add(((r, c), d))
    return player, frozenset(boxes), frozenset(lasers)


def _zaps(eng, expert) -> tuple:
    """Both zap layers as ``(horizontal cells, vertical cells)``."""
    hz_id, vz_id = expert.zap_ids
    hz, vz = set(), set()
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if hz_id in cell:
                hz.add((r, c))
            if vz_id in cell:
                vz.add((r, c))
    return frozenset(hz), frozenset(vz)


def _report() -> int:
    """Per-level board size, piece counts, plan length, tie coverage, search
    size -- and CERTIFY every plan by replaying it through the interpreter."""
    _solver, game, expert = _levels()
    eng = game._engine
    total = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        clock = time.time()
        presses, optsets, stats = board.solve()
        took = time.time() - clock
        if presses is None:
            print(f"level {level:2d}: UNSOLVABLE ({stats})")
            bad += 1
            continue
        # Certification: the interpreter must win on the LAST press and no
        # earlier -- an earlier win would mean the plan is not shortest.
        won_at = None
        for i, d in enumerate(presses):
            eng.step(_ORDER[d])
            if eng.check_win():
                won_at = i
                break
        ok = won_at == len(presses) - 1
        bad += not ok
        total += len(presses)
        ties = sum(1 for s in optsets if len(s) > 1)
        budget = game._max_steps
        room = "ok" if len(presses) <= budget else "OVER BUDGET"
        print(f"level {level:2d}: {board.h:2d}x{board.w:2d} "
              f"{len(board.start[1])} boxes {len(board.targets)} targets "
              f"{len(board.start[2])} emitters, "
              f"{len(presses):3d} presses (budget {budget}, {room}), "
              f"{ties:3d} tie steps ({ties / len(presses):3.0%}), "
              f"{stats['mode']:5s} {stats['states']:7d} states in {took:6.2f}s, "
              f"{'CERTIFIED' if ok else 'PLAN REJECTED BY THE INTERPRETER'}")
    print(f"total {total} presses over {game.n_levels} levels")
    return 0 if not bad else 1


def _verify() -> int:
    """Double-entry check of the plan lengths AND the tie sets.

    The field levels are re-derived from the other end: an independent
    breadth-first search over PRIMITIVE presses (sharing nothing with the macro
    closure, its backward Dijkstra or `config_field`) measures the true distance
    to the win from the start and from every successor of every plan step, and
    the optimal set has to be exactly the successors one press closer.

    The A* levels get the check their labels actually claim, and it is not a
    model-only one: for every labelled alternative, the variant plan "take that
    press, then rejoin the macro path" is rebuilt and REPLAYED THROUGH THE
    INTERPRETER, which must win on its last press and on no earlier one, at
    exactly the length of the plan. Re-running the search from each successor
    instead would be the stronger statement but costs a minute a press on level
    18; this says the same thing about every press that is labelled."""
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        presses, optsets, stats = board.solve()
        if presses is None:
            print(f"level {level:2d}: unsolvable, nothing to verify")
            bad += 1
            continue

        if stats["mode"] == "field":
            truth = _primitive_field(board)
            note = "exact"
            if truth.get(board.start) != len(presses):
                print(f"level {level:2d}: LENGTH MISMATCH plan {len(presses)} "
                      f"vs primitive BFS {truth.get(board.start)}")
                bad += 1
                continue
            state = board.start
            for i, d in enumerate(presses):
                here = truth[state]
                want = set()
                for cand in range(4):
                    nxt = board.settle(state, cand)
                    if nxt is not None and truth.get(nxt, -1) == here - 1:
                        want.add(cand)
                    if nxt is not None and board.won(nxt) and here == 1:
                        want.add(cand)
                if want != set(optsets[i]):
                    print(f"level {level:2d} step {i}: OPTSET MISMATCH "
                          f"{sorted(optsets[i])} vs {sorted(want)}")
                    bad += 1
                    break
                state = board.step(state, d)
        else:
            note = "sound"
            path, owners = stats["path"], stats["owners"]
            state = board.start
            for i, d in enumerate(presses):
                for cand in optsets[i]:
                    nxt = board.settle(state, cand)
                    if nxt is None:
                        print(f"level {level:2d} step {i}: labelled press "
                              f"{_ORDER[cand]} is refused or fatal")
                        bad += 1
                        break
                    # A press that COMPLETED its macro (the push) has to rejoin
                    # at the next one; a walk press is still inside its own.
                    done = (i + 1 == len(presses) or owners[i + 1] != owners[i])
                    rest = path[owners[i] + 1:] if done else path[owners[i]:]
                    tail, _sets, _own = board.flatten(rest, start=nxt)
                    variant = presses[:i] + [cand] + tail
                    if len(variant) != len(presses):
                        print(f"level {level:2d} step {i}: labelled press "
                              f"{_ORDER[cand]} leads to a {len(variant)}-press "
                              f"plan, not {len(presses)}")
                        bad += 1
                        break
                    if not _replay_wins(game, level, variant):
                        print(f"level {level:2d} step {i}: the variant plan "
                              f"through {_ORDER[cand]} does not win on its "
                              f"last press")
                        bad += 1
                        break
                state = board.step(state, d)
            game.set_level(level)

        ties = sum(1 for s in optsets if len(s) > 1)
        print(f"level {level:2d}: {len(presses):3d} presses, {ties:3d} tie steps "
              f"re-derived ({note})")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} levels")
    return 0 if not bad else 1


def _primitive_field(board: _Board, cap: int = 8_000_000) -> dict:
    """``{state: presses to the win}`` by breadth-first search over PRIMITIVE
    presses from the start, then a backward sweep -- the independent witness
    `--verify` prices the macro machinery against.

    Only used on the fifteen levels the macro graph closes for, whose primitive
    spaces top out at ~160k states."""
    order = {board.start: 0}
    queue = deque([board.start])
    preds: dict = {}
    wins = []
    while queue:
        state = queue.popleft()
        if board.won(state):
            wins.append(state)
            continue
        for d in range(4):
            nxt = board.settle(state, d)
            if nxt is None:
                continue
            preds.setdefault(nxt, []).append(state)
            if nxt not in order:
                order[nxt] = order[state] + 1
                queue.append(nxt)
                if len(order) > cap:
                    raise MemoryError("primitive space exceeded the cap")
    dist = {w: 0 for w in wins}
    queue = deque(wins)
    while queue:
        state = queue.popleft()
        step = dist[state] + 1
        for prev in preds.get(state, ()):
            if prev not in dist:
                dist[prev] = step
                queue.append(prev)
    return dist


def _replay_wins(game, level: int, presses) -> bool:
    """Replay a press sequence on the real interpreter from the level's start:
    True iff it wins on its LAST press and on no earlier one."""
    game.set_level(level)
    eng = game._engine
    for i, d in enumerate(presses):
        eng.step(_ORDER[d])
        if eng.check_win():
            return i == len(presses) - 1
    return False


def _fuzz(trials: int = 30, depth: int = 40, seed: int = 1234) -> int:
    """Differentially fuzz `_Board` against the real interpreter.

    Every level, from a COLD state (no warm-up press -- a warm-up hides exactly
    the turn-one bookkeeping that has bitten other ps: generators), random
    presses including ACTION5, comparing after every one:

      * the pieces (player, boxes, emitters with their facings),
      * BOTH zap layers, cell for cell -- the model ray-casts them and the
        interpreter grows them one rule application at a time, so this is the
        real test of "beams pass through the player and stop before the first
        obstacle",
      * ``check_win``,
      * and the death rule in both directions: a state whose player is on a zap
        must delete the player on the next press, and a state whose player is
        not must not.

    A model that invents a death silently loses solutions; one that misses a
    death ships plans that walk the agent into a beam."""
    _solver, game, expert = _levels()
    eng = game._engine
    rng = random.Random(seed)
    actions = list(_ORDER) + ["action"]
    steps = deaths = checked = bad = 0

    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        if board.lethal(board.start):
            print(f"level {level:2d}: STARTS IN A BEAM")
            bad += 1
        if _pieces_of(board, board.start) != _pieces(eng, expert):
            print(f"level {level:2d}: START MISMATCH")
            bad += 1
        if _zaps_of(board, board.start) != _zaps(eng, expert):
            print(f"level {level:2d}: START ZAP MISMATCH")
            bad += 1

        for _trial in range(trials):
            game.set_level(level)
            state = board.start
            for _t in range(depth):
                name = rng.choice(actions)
                lethal = board.lethal(state)
                eng.step(name)
                steps += 1
                live = _pieces(eng, expert)[0] is not None
                checked += 1
                if live == lethal:
                    print(f"level {level:2d}: DEATH MISMATCH -- model said "
                          f"{'lethal' if lethal else 'safe'}, interpreter "
                          f"{'kept' if live else 'deleted'} the player")
                    bad += 1
                    break
                if not live:
                    deaths += 1
                    break
                nxt = (state if name == "action"
                       else board.step(state, _ORDER.index(name)))
                if _pieces_of(board, nxt) != _pieces(eng, expert):
                    print(f"level {level:2d}: STEP MISMATCH after {name}")
                    bad += 1
                    break
                if _zaps_of(board, nxt) != _zaps(eng, expert):
                    print(f"level {level:2d}: ZAP MISMATCH after {name}")
                    bad += 1
                    break
                model_win = board.won(nxt) and not board.lethal(nxt)
                if model_win != eng.check_win():
                    print(f"level {level:2d}: WIN MISMATCH after {name}")
                    bad += 1
                    break
                state = nxt
                if model_win:
                    break
    print(f"{steps} presses on {game.n_levels} levels, {checked} death checks, "
          f"{deaths} deaths observed: "
          + ("model exact" if not bad else f"{bad} DIVERGENCES"))
    return 0 if not bad else 1


def _pieces_of(board: _Board, state) -> tuple:
    """`_pieces`, from a model state instead of from the interpreter."""
    w = board.w
    return ((state[0] // w, state[0] % w),
            frozenset((p // w, p % w) for p in state[1]),
            frozenset(((p // w, p % w), d) for p, d in state[2]))


def _zaps_of(board: _Board, state) -> tuple:
    """`_zaps`, from a model state: the horizontal layer is what the left/right
    emitters light, the vertical layer what the up/down ones do."""
    w = board.w
    blocked, _zap = board.cfg(state[1], state[2])
    layers = []
    for wanted in ((2, 3), (0, 1)):
        cells = set()
        for p, d in state[2]:
            if d not in wanted:
                continue
            table = board.nbr[d]
            q = table[p]
            while q >= 0 and q not in blocked:
                cells.add((q // w, q % w))
                q = table[q]
        layers.append(frozenset(cells))
    return tuple(layers)


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

    dmap = {}
    for i, (dr, dc) in enumerate(_DELTA):
        if mirror:
            dc = -dc
        for _ in range(k):
            dr, dc = dc, -dr
        dmap[i] = _DELTA.index((dr, dc))
    return cell, dims, dmap


def _symmetry() -> int:
    """Prove ON THE INTERPRETER that the 8-element presentation group is an
    exact symmetry of the mechanic, which is what entitles this game to the
    `PuzzleScriptAdapter._FLIP_GAMES` augmentation.

    The six beam-propagation rules are written out per direction (a
    ``LATE RIGHT`` block, a ``LATE HORIZONTAL`` block, ...) rather than with the
    relative ``>`` force, and it is that per-direction loop that has hidden a
    chirality in other ps: games -- the rule ORDER, not the mechanic, decides
    who wins a contested cell. Reading the six blocks and declaring them copies
    of each other is exactly the argument that was already wrong there, so
    instead each level's plan is replayed on all eight turned and mirrored
    copies of its own board and every press must leave the pieces (and both
    beams) where the transform of the reference run put them.

    The copies are built by transforming the LEVEL LAYOUT rather than the loaded
    grid, so the interpreter re-runs ``run_rules_on_level_start`` and lights the
    beams itself on each one. Moving the CELLS is not enough: an emitter's
    facing is an object identity here (``LaserRight`` is a different object from
    ``LaserDown``), so a turned copy has to SUBSTITUTE each emitter for the one
    facing the transformed direction. Leaving them alone builds a different
    puzzle and reports the whole game chiral, plain rotations included."""
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        presses, _optsets, _stats = board.solve()
        layout = g.levels[level]
        hw = (len(layout), len(layout[0]))

        eng.load_level(layout)
        reference = [(_pieces(eng, expert), _zaps(eng, expert))]
        for d in presses:
            eng.step(_ORDER[d])
            reference.append((_pieces(eng, expert), _zaps(eng, expert)))

        notes = []
        for k, mirror in itertools.product(range(4), (False, True)):
            if (k, mirror) == (0, False):
                continue
            cell, dims, dmap = _transform(k, mirror)
            th, tw = dims(hw)
            laser_ids = expert.ids[4]
            swap = {laser_ids[d]: laser_ids[dmap[d]] for d in range(4)}
            turned = [[set() for _ in range(tw)] for _ in range(th)]
            for r, row in enumerate(layout):
                for c, objs in enumerate(row):
                    tr, tc = cell((r, c), hw)
                    turned[tr][tc] = {swap.get(o, o) for o in objs}
            eng.load_level(turned)
            for i, d in enumerate(presses):
                eng.step(_ORDER[dmap[d]])
                (player, boxes, lasers), (hz, vz) = reference[i + 1]
                want = ((cell(player, hw),
                         frozenset(cell(x, hw) for x in boxes),
                         frozenset((cell(p, hw), dmap[fd]) for p, fd in lasers)),
                        (frozenset(cell(x, hw) for x in hz),
                         frozenset(cell(x, hw) for x in vz)))
                got = (_pieces(eng, expert), _zaps(eng, expert))
                # A turn by 90 degrees swaps which LAYER a beam lives on, so
                # compare the layers under the same swap the cells got.
                if k % 2:
                    want = (want[0], (want[1][1], want[1][0]))
                if got != want:
                    notes.append(f"rot{k}{'m' if mirror else ''}@press{i}")
                    break
        bad += len(notes)
        print(f"level {level:2d}: {len(presses):3d} presses x 7 presentations: "
              + ("SYMMETRIC" if not notes else "CHIRAL " + ", ".join(notes[:3])))
    print("symmetry clean -- the flip augmentation is exact" if not bad
          else f"SYMMETRY FAILED: {bad} chiral presentations")
    return 0 if not bad else 1


#: Every stack of objects the game can put in one cell, named. The zaps never
#: share a cell with a box, an emitter or a wall (all three are ``Obstacle``, and
#: a beam stops before one), so those combinations are absent on purpose.
_COMPOSITIONS = {
    "floor": (), "wall": ("wall",), "target": ("target",), "box": ("box",),
    "box_on_target": ("target", "box"),
    "player": ("player",), "player_on_target": ("target", "player"),
    "hz": ("horizap",), "vz": ("vertzap",), "hz+vz": ("horizap", "vertzap"),
    "player_on_hz": ("horizap", "player"),
    "player_on_vz": ("vertzap", "player"),
    "player_on_hz+vz": ("horizap", "vertzap", "player"),
    "player_on_target_hz": ("target", "horizap", "player"),
    "target_on_hz": ("target", "horizap"),
    "target_on_vz": ("target", "vertzap"),
    "laser_up": ("laserup",), "laser_down": ("laserdown",),
    "laser_left": ("laserleft",), "laser_right": ("laserright",),
    "laser_up_on_target": ("target", "laserup"),
}


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    Three frames the game is about were identical in the shipped art, and all
    three were in the STACK rather than in one sprite: the player covered the
    beam it was standing in (the frame that kills you), the box covered the
    target under it (the frame that wins), and at 4px the target and both beams
    disappeared into the background entirely. An object-by-object colour check
    catches none of them.

    Whole frames are compared, not cell crops: `_render_frame` upscales the
    board to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window."""
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for name, objs in _COMPOSITIONS.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        # The two frames that must not merely differ but be READABLE get their
        # margins reported rather than just their distinctness.
        cell = h * w
        win_px = int((shots["box_on_target"] != shots["box"]).sum()) / cell
        die_px = int((shots["player_on_hz"] != shots["player"]).sum()) / cell
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"box-on-target reads {win_px:4.1f}px/cell, "
              f"player-in-a-beam reads {die_px:4.1f}px/cell, "
              + ("OK" if not clashes else "IDENTICAL " + str(clashes)))
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(LaserSolver.main())
