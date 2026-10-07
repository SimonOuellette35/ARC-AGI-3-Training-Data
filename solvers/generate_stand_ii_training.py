"""ps:stand_ii -- Connorses' "Stand II": the crowd that walks on ONE key, now
with two kinds of crate in the way.

    [ >  Player | Crate ]    -> [ >  Player | > Crate ]
    [ >  Player | BigCrate ] -> [    Player | > BigCrate ]
    late [Target1 Player] -> [Target2 Player]
    late [Target2 no Player] -> [Target1]
    ==============
    WINCONDITIONS
    ==============
    No Target1

Eleven levels, all eleven WON, ten of them PROVED SHORTEST -- 8, 13, 4, 7, 18,
17, 6, 12, 31 and 20 presses -- and proved TWICE: once off an exact
distance-to-win field, and once by a heuristic-free breadth-first press search
that shares none of the field's machinery (`--bfs`).  Part of the
`solvers/common/ps_astar.py` family, on the native-model side of it
(`plan_cache_path` + a `_search` override): the whole mechanic is
re-implemented as three integer bitmasks, and the interpreter's job is to
REFEREE the model (`--verify`) and to CERTIFY its answers (`--plans`).

WHAT THIS GAME IS: ps:stand PLUS A SOKOBAN, ON ONE KEY

`ps:stand` is the same author's earlier game and the same core rule -- an arrow
key assigns its force to EVERY Player sprite, so the puzzle is BREAKING
FORMATION: two people three cells apart stay three cells apart forever unless a
wall stops one of them and not the other, and the win is one body on each light
AT THE SAME INSTANT (`No Target1`, a light being lit only while somebody stands
on it).  See `solvers/generate_stand_training.py`; everything that generator
says about the crowd applies here unchanged, including the heuristic.

What Stand II adds is the two crate rules above, and they are NOT the same
rule twice:

* **Crate** is an ordinary sokoban crate.  The pusher keeps its own force, so
  player and crate advance together.
* **BigCrate** is the inverted one: the RHS DROPS the player's force
  (``[ > Player | BigCrate] -> [ Player | > BigCrate ]``).  You shove it one
  cell and you do not move -- "you can push metal crates but they're heavy and
  slow you down".  That makes a BigCrate a WAIT BUTTON for one person, which is
  the only device in either game that lets one member of the crowd stand still
  while the others walk, and it is why levels 6, 7, 8 and 10 are built around
  one.  It also propagates: a person queued up BEHIND the shover has their move
  eaten too, because the shover never vacates the cell.
* **Neither rule chains.**  There is no ``[ > Crate | Crate ]``, so a crate
  pushed into anything solid simply does not move -- and then neither does its
  pusher, nor anybody queued behind them.  The game says so out loud between
  levels 4 and 5 ("You can't push two crates in a row though").

`CrateTop` / `BigCrateTop` are pure decoration -- the author's trick for drawing
a crate one square taller (`late up [crate|no crateTop] -> [Crate|Cratetop]`,
re-derived from scratch every turn).  They sit in their own collision layer, so
they never block anything and they are not part of the state.

`Hole` and `Lose` (fall in a hole and the level restarts) are declared, and NO
LEVEL PLACES A HOLE -- `StandIIBoard` asserts that at construction, so if the
file ever gains one the model fails loudly instead of quietly ignoring a death.
`noaction` is in the prelude, so ACTION assigns no force and cannot change a
cell; like ps:stand it is left out of `directions` rather than searched and
discarded, and `--verify` MEASURES that it is inert rather than trusting the
prelude.

THE MODEL: THREE BITMASKS, AND ONE FRONT-FIRST PASS

Walls never move and a light is a pure function of whether a body is on it, so
a state is exactly ``(players, crates, bigcrates)`` -- three ``int`` masks over
cell indices.  `StandIIBoard.step` is:

1. every Player takes the pressed force;
2. a Player facing a Crate lends the force to it (rule 1);
3. a Player facing a BigCrate gives the force away entirely -- the BigCrate
   gets it and the Player loses it (rule 2);
4. one FRONT-FIRST pass settles the moves.

Step 4 is exact in one pass for the same reason it is in ps:stand: every force
a press assigns points the SAME way, so there is exactly one chain per queue
and the front-most member decides for everybody behind.  What is new is that a
queue can now contain solids with NO force -- an unpushed crate, a BigCrate
nobody is shoving, or a Player whose force rule 2 just confiscated -- and those
are blockers, exactly as walls are.  Scanning ascending row-major for
`up`/`left` and descending for `down`/`right` visits the cell ahead first in
every case, so "blocked" is decided before it is read.

THE HEURISTIC: ps:stand's DIRECTION-DEBT BOUND, WITH THE CRATES RELAXED AWAY

From the one-key rule: a press is ONE direction, and a `down` press advances
each person by at most one `down`-step, so a plan with `n_down` down-presses
gives EVERY person at most `n_down` down-steps.  Writing `req_d(p, t)` for the
fewest `d`-steps on any walk from cell `p` to cell `t`,

    n_d  >=  max over people of req_d(person, its light)      for each d,
    L    >=  SUM over the four directions of those four maxima,

minimised over the perfect matchings of people to lights.  `req_d` is a 0-1 BFS
out of each light (a step in direction `d` costs 1, the other three cost 0), so
it is wall-aware -- a person forced to walk around a block pays for the
detour's `down`s even where the row difference is zero.

The crates are RELAXED AWAY: `req_d` is computed on the walls-only board, where
crates are floor.  That keeps it admissible -- every real trajectory is a walk
on the relaxed board, so it cannot contain fewer `d`-steps than `req_d` -- and
it keeps `h` a function of the PLAYER mask alone, which is what makes the memo
pay for itself (level 10 visits 616374 states but far fewer player layouts).
It also keeps it CONSISTENT: prepending an `e`-step to a walk adds no `d`-steps
for any other `d`, so one press retires at most one press's worth of one
direction's debt, and `h(s) <= 1 + h(s')` across every edge.  `--verify`
MEASURES both properties over every state of every exact field rather than
resting on this paragraph.

What it does NOT know is that crates have to be shoved out of the way, so on
the crate-heavy boards the bound is loose (level 4: h0 = 3 against d* = 18) and
the search pays for it.  That is the whole story of level 9; see below.

THE FIELD: A* FOR THE BOUND, THEN ONE SWEEP FOR EVERYTHING ELSE

The `solvers/generate_spiders_hollow_training.py` pattern.  `astar` proves `d*`
(weight 1, the admissible heuristic above).  `field` then sweeps forward from
the level start keeping only states with ``g + h <= d*`` -- with a consistent
`h` that is a superset of every state on every shortest path -- and runs a
backward BFS from the winning edges over exactly those edges.  Out of the
resulting exact distance-to-win field come, in one pass:

* a plan whose length is checked against `d*`, i.e. provably shortest;
* the EXACT optimal-action SET at every step: every press that still finishes
  in the same number of moves.

LEVEL 9 IS THE ONE THAT IS NOT PROVED

Level 9 is 7 people, 7 lights and 11 crates on a 10x9 board, and it is the only
level where the exact search does not terminate.  Weight-1 A* was still at
``f = 30`` after 338655 states and 445 seconds, against a win that is at most
48 presses away; weight-2 A* was at ``f = 44`` after 115659 states and 228
seconds.  The relaxed heuristic is the reason -- it costs the walk and knows
nothing about the eleven crates that have to be shoved out of the corridors
before anybody can walk it.  (Nor is it that the level is far from the
heuristic's reach in general: ``h0 = 20`` here, and the other ten levels close
their gap in seconds.)

So level 9 is planned at ``weight = 3`` (`StandIIExpert.level_weight`), which
finds a 48-press WIN in 86 s.  That plan is a genuine win -- `--plans` replays
it through the real interpreter -- it is simply not proved shortest, so its
steps are labelled with themselves alone rather than with a measured optimal
set.  Declaring the weight per level rather than discovering it with a time
limit is deliberate: a wall-clock fallback would make the disk cache
machine-dependent, and the whole point of `plan_cache_path` is that two
processes derive byte-identical plans.

WHAT WAS INVISIBLE (two things, both in the frame and neither in the rules)

1. **The winning square.**  `Target2` -- what a light becomes while somebody is
   standing on it -- shipped `Darkgreen lightgreen`, and the ARC palette has
   exactly one green, so both of them and the Background's own `GREEN` are
   index 14.  A person standing on a lit light was therefore pixel-identical to
   a person standing on grass, on a game whose entire win condition is "all of
   them at once".  It is now `Yellow Yellow` (11), the author's sprite geometry
   untouched.  This is the same bug, and the same fix, as ps:stand.

2. **A light under a BigCrate.**  Levels 5 and 7 are 14 cells wide, so they
   render at 4 px per cell, and the renderer's centred nearest sampling of a
   5x5 sprite into 4x4 keeps sprite rows/cols 0,1,3,4 and DROPS row/col 2.
   `Target1` paints rows 1-3, of which only 1 and 3 survive, and `BigCrate`'s
   rows 1 and 3 were both fully opaque -- so on level 7, the one 4-px level
   with a BigCrate, shoving it onto a light ERASED the light from the frame.
   One pixel of BigCrate's row 3 is now transparent, at a column the 4-px
   sampling keeps.

Both live in `data/puzzlescript_games/Stand_II.txt`, both are sprite-only, and
`--audit` is the regression test for both -- it FAILED on both before the
change.  What `--audit` still reports as identical, deliberately, is
`wall1 + cratetop` vs `wall2 + cratetop` (and the same pair with
`bigcratetop`): `Wall1` is a solid brown cell and `Wall2` is the same brown
with its bottom row clear, so a crate's bottom decal fills exactly the gap that
told them apart.  Neither object appears in any rule -- they are two paint jobs
for one behaviour -- so those two pictures MEAN the same thing, which is the
property `--audit` actually enforces: two compositions may render alike only if
they say the same thing about the cell.

`Stand_II` is in `PuzzleScriptAdapter._FLIP_GAMES`, so it is drawn at all 16
presentations rather than the 4 rotations every ps: game gets.  The argument is
ESCAPE!'s -- gravity-free, screen-relative input, a win condition that names no
direction (`No Target1`), and no sprite that encodes a facing.  Two things here
are worth measuring rather than asserting, and `--symmetry` measures them: a
press moves up to SEVEN bodies at once (the shape that hid Gobble Rush's
rule-order chirality -- safe here because every force a press assigns points the
same way, so two chains can never contest a cell), and the one rule in the file
that names a direction, `late up [crate|no crateTop]`, draws a decal on the
engine-up side.  That decal is deleted and re-derived every turn, sits in its
own collision layer and is read by nothing, so a flipped presentation simply
draws crate tops on the flipped side -- consistently, and as an exact transform
of the unflipped frame.

VERIFIED.  11/11 levels, 184 presses, every one of the eleven plans replayed on
the real interpreter to a WIN (`--plans`), and the ten weight-1 plans each
their level's proven ``d*``.

* ``--bfs``: a heuristic-free breadth-first press search re-derives 8, 13, 4,
  7, 18, 17, 6, 12 and 31 for levels 0-8 out of 43 / 984 / 8 / 99 / 57057 /
  1421 / 23 / 4474 / 2727 reachable states, and ``--bfs 10`` re-derives 20 out
  of 12759045.  So every level this generator calls shortest is proved shortest
  TWICE, by two searches sharing no machinery.
* ``--verify``: 35200 presses of random and CROWDED play with the model and the
  interpreter stepped side by side, 0 mismatches; ACTION pressed 4840 times and
  inert every time; and over the 333555 states of the ten exact fields, 0
  inadmissible states and 0 inconsistent edges.
* ``--audit``: 30 reachable compositions at each of the ten cell sizes the
  levels use, no two that MEAN different things sharing a picture under any of
  the eight rotations and mirrors, and a fuzz confirming no light/solid pair
  outside the declared reachable set ever appears.  It FAILED on both sprite
  bugs before the fix.
* ``--symmetry``: 11 levels x 16 presentations, every plan a WIN and every
  frame of both the plan and a 200-press random walk exactly the transform of
  the unaugmented one.
* Generation: 6 seeds x 11 levels in 5.4 s warm (about 110 s from an empty plan
  cache, of which level 9 is 87 s and level 10 is 17 s).  All 66 levels WIN,
  all 1241 recorded steps replay frame-exact through the adapter from the
  recorded SCREEN actions, 1110 of them labelled (1.12 optimal presses per
  step, the taken press always inside its own set; the 131 unlabelled are the
  exploration prefixes and each level's opening RESET), every index in 0..6,
  and two processes at different hash seeds emit byte-identical episodes --
  and, from a cold cache, a byte-identical ``data/stand_ii_plans.json``.  The
  longest plan is 48 presses against the adapter's 200-press per-level cap.

RECOVERY is the family's RESET prefix (``recovery_mode = "reset"``).  It has to
be: both halves of this game are irreversible -- once a wall has eaten one
person's step the crowd's shape has changed, and a crate shoved against a wall
can never be pulled back -- so a perturbed board cannot be re-planned from, and
one RESET lands on exactly the state the cached field was built from.

CLI: ``--verify`` (the model against the interpreter over random AND crowded
play, ACTION's inertness, and the heuristic's admissibility/consistency against
the exact fields), ``--audit`` (every composition the game can draw, pixel-wise
distinct-or-synonymous at every cell size the levels use and under the whole
8-element symmetry group, plus a fuzz that no composition outside the declared
set ever appears), ``--symmetry`` (every presentation is an exact transform of
the unaugmented frames and every plan still wins through the adapter),
``--bfs`` (a heuristic-free exhaustive re-proof of the levels whose whole space
fits), ``--plans`` (every level's plan, replayed through the interpreter),
otherwise the BaseSolver CLI.
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

from arcengine import ActionInput, GameState                      # noqa: E402
from adapters.puzzlescript_adapter import (                       # noqa: E402
    PuzzleScriptAdapter, _render_frame,
)
from solvers.base_solver import _ID_TO_GAMEACTION                 # noqa: E402
from solvers.common.ps_astar import (                             # noqa: E402
    PSAStarSolver, PSExpert, Plan, screen_action,
)
from utils.rotation import inverse_remap_action_full              # noqa: E402

GAME_ID = "ps:stand_ii"
GAME_NAME = "Stand_II"

#: The presses the search branches on.  ACTION is deliberately absent: the file
#: declares `noaction` and no rule reads it, so it assigns no force and changes
#: no cell, and a press that cannot change the board can never be on a shortest
#: path.  `verify` measures that rather than trusting the prelude.
DIRS = ("up", "down", "left", "right")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Everything on these boards.  Wall1/Wall2 are two paint jobs for one
#: behaviour (neither appears in a rule; both merely occupy the pushables'
#: collision layer) and Target1/Target2 are the unlit and lit faces of one
#: light.  CrateTop/BigCrateTop are decals in a layer of their own and are not
#: read here at all; Hole/Lose are declared by the game and placed by no level.
_OBJECTS = ("wall1", "wall2", "target1", "target2", "player", "crate",
            "bigcrate", "hole", "lose", "cratetop", "bigcratetop")

_INF = float("inf")

#: Sentinel successor for "this press wins".  A `str` can never collide with a
#: board, which is a 3-tuple of `int`.
_WIN = "win"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class StandIIBoard:
    """One level, and the whole mechanic, as bit twiddling over cell indices.

    A state is ``(players, crates, bigcrates)``: three ``int`` masks in which
    bit ``r * W + c`` is set when that cell holds one.  Nothing else varies --
    walls never move, a light is a pure function of whether a body is on it,
    and the crate-top decals are re-derived from the crates every turn -- so
    this is an exact, canonical encoding, and level 10's 121-cell board costs
    three small ints per state instead of the ~64 KB a
    `solvers.common.ps_astar.snapshot` of the same grid needs.
    """

    def __init__(self, eng, ids: dict):
        grid = eng.grid
        self.H, self.W = len(grid), len(grid[0])
        wall_ids = {ids["wall1"], ids["wall2"]}
        light_ids = {ids["target1"], ids["target2"]}
        walls, lights, people, crates, bigs = set(), [], [], [], []
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                i = r * self.W + c
                if cell & wall_ids:
                    walls.add((r, c))
                if cell & light_ids:
                    lights.append(i)
                if ids["player"] in cell:
                    people.append(i)
                if ids["crate"] in cell:
                    crates.append(i)
                if ids["bigcrate"] in cell:
                    bigs.append(i)
                # The hole rules (`late [falls hole] -> [hole]`, `late [player
                # hole] -> [lose hole]`, `[lose hole] -> restart`) are the only
                # part of this game the model does not implement, on the
                # grounds that no level places a hole.  Check it rather than
                # assume it.
                if ids["hole"] in cell or ids["lose"] in cell:
                    raise AssertionError(
                        f"Stand_II level places a hole at {(r, c)}: the model "
                        "does not implement falling and would be wrong")
        # `nxt[d][i]` is the cell a body on `i` moves to when `d` is pressed,
        # or -1 when the board edge or a WALL stops it.  Walls are baked in
        # because they never move; crates and people are not, because they do.
        self.nxt = {}
        for d, (dr, dc) in _DELTA.items():
            arr = [-1] * (self.H * self.W)
            for r in range(self.H):
                for c in range(self.W):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < self.H and 0 <= nc < self.W
                            and (nr, nc) not in walls):
                        arr[r * self.W + c] = nr * self.W + nc
            self.nxt[d] = arr
        self.walls = walls
        self.lights = tuple(lights)
        self.goal = sum(1 << i for i in lights)
        self.start = (sum(1 << i for i in people),
                      sum(1 << i for i in crates),
                      sum(1 << i for i in bigs))
        self._h_cache: dict[int, float] = {}
        self._fields = self._dir_fields()

    # -- dynamics ------------------------------------------------------------
    @staticmethod
    def _cells(mask: int) -> list[int]:
        """The occupied cells of ``mask``, in ASCENDING index order."""
        out = []
        while mask:
            low = mask & -mask
            out.append(low.bit_length() - 1)
            mask ^= low
        return out

    def step(self, state: tuple, d: str) -> tuple:
        """The board after pressing ``d``.

        The two push rules, then one FRONT-FIRST pass over the solids.

        The pass is exact because every force this press assigns points the
        same way, so there is exactly one chain per queue and the front-most
        member decides for everybody behind.  Ascending row-major order visits
        the cell ahead first for `up` and `left` (its reverse does for `down`
        and `right`), so a cell's blocked-ness is always decided before it is
        read.  Cross-row order is irrelevant: two bodies in different rows can
        never block each other on a horizontal press.

        A solid with NO force is a blocker exactly as a wall is -- an unpushed
        crate, a BigCrate nobody is shoving, and a Player whose force rule 2
        just confiscated are all in that state, which is what makes a BigCrate
        stop the whole queue standing behind its shover.
        """
        players, crates, bigs = state
        nxt = self.nxt[d]
        occupied = players | crates | bigs
        forced_p, forced_c, forced_b = players, 0, 0
        for cell in self._cells(players):
            ahead = nxt[cell]
            if ahead < 0:
                continue
            if (crates >> ahead) & 1:
                forced_c |= 1 << ahead          # [ > Player | Crate ]
            elif (bigs >> ahead) & 1:
                forced_b |= 1 << ahead          # [ > Player | BigCrate ]
                forced_p &= ~(1 << cell)        # ... and the pusher stays put
        forced = forced_p | forced_c | forced_b

        order = self._cells(occupied)
        if d == "down" or d == "right":
            order.reverse()
        blocked = 0
        for cell in order:
            if not (forced >> cell) & 1:
                blocked |= 1 << cell            # no force: a wall, for today
                continue
            ahead = nxt[cell]
            if ahead < 0 or ((occupied >> ahead) & 1
                             and (blocked >> ahead) & 1):
                blocked |= 1 << cell

        out = []
        for mask in (players, crates, bigs):
            moved = 0
            for cell in self._cells(mask):
                moved |= 1 << (cell if (blocked >> cell) & 1 else nxt[cell])
            out.append(moved)
        return (out[0], out[1], out[2])

    def won(self, state: tuple) -> bool:
        """Every light lit AT ONCE -- the engine's `No Target1`."""
        return state[0] & self.goal == self.goal

    # -- the heuristic -------------------------------------------------------
    def _dir_fields(self) -> dict:
        """``{(light, d): [fewest d-steps on any walk from cell to light]}``.

        A 0-1 BFS backwards out of each light in which a step in direction
        ``d`` costs 1 and the other three cost 0.  Unlike a plain distance this
        is the DEBT a plan owes one direction, which is the quantity the
        one-key rule bounds.  Crates are floor here (see the module docstring):
        relaxing them is what keeps `h` admissible AND a function of the player
        mask alone.
        """
        n = self.H * self.W
        fields = {}
        for d in DIRS:
            # Reverse adjacency: `pred[v]` holds every (u, cost) with u --e--> v.
            pred: list[list[tuple[int, int]]] = [[] for _ in range(n)]
            for e in DIRS:
                cost = 1 if e == d else 0
                for u, v in enumerate(self.nxt[e]):
                    if v >= 0:
                        pred[v].append((u, cost))
            for light in self.lights:
                dist = [_INF] * n
                dist[light] = 0
                queue = deque([light])
                while queue:
                    v = queue.popleft()
                    dv = dist[v]
                    for u, cost in pred[v]:
                        nd = dv + cost
                        if nd < dist[u]:
                            dist[u] = nd
                            (queue.appendleft if cost == 0
                             else queue.append)(u)
                fields[(light, d)] = dist
        return fields

    def heuristic(self, state: tuple) -> float:
        """Presses still owed, as a lower bound: the smallest over
        person-to-light matchings of the sum over the four directions of the
        largest debt any matched person owes that direction.

        `_INF` when some light is unreachable for every free person -- the
        state can never win and the search drops it.
        """
        players = state[0]
        got = self._h_cache.get(players)
        if got is not None:
            return got
        people = self._cells(players)
        lights = self.lights
        n = len(lights)
        # One row per direction: rows[d][person][light] is that person's debt.
        rows = [[[self._fields[(t, d)][p] for t in lights] for p in people]
                for d in DIRS]
        best = _INF
        for pick in itertools.permutations(range(len(people)), n):
            total = 0
            for row in rows:
                worst = 0
                for j in range(n):
                    debt = row[pick[j]][j]
                    if debt > worst:
                        worst = debt
                if worst == _INF:
                    total = _INF
                    break
                total += worst
            if total < best:
                best = total
                if best == 0:
                    break
        self._h_cache[players] = best
        return best

    # -- search --------------------------------------------------------------
    def astar(self, node_cap: int = 1_500_000,
              weight: int = 1) -> "list[str] | None":
        """A press sequence to a win, or None if there is none within the cap.

        At ``weight == 1`` the heuristic is admissible and consistent, so the
        first win popped is optimal -- and unlike a macro search every edge
        here costs exactly one press, so generating a win IS reaching it.  A
        larger weight trades that guarantee for reach; see
        `StandIIExpert.level_weight` for the one level that needs it.
        """
        start = self.start
        if self.won(start):
            return []
        queue = [(weight * self.heuristic(start), 0, 0, start, ())]
        best_g = {start: 0}
        counter = 0
        while queue:
            _f, g, _c, state, path = heapq.heappop(queue)
            for d in DIRS:
                nxt = self.step(state, d)
                if nxt == state:
                    continue            # pressed into a wall, as a group
                if self.won(nxt):
                    return list(path) + [d]
                ng = g + 1
                if best_g.get(nxt, 1 << 30) <= ng:
                    continue
                hv = self.heuristic(nxt)
                if hv == _INF:
                    continue                    # a light nobody can ever reach
                best_g[nxt] = ng
                counter += 1
                heapq.heappush(queue, (ng + weight * hv, ng, counter, nxt,
                                       path + (d,)))
            if len(best_g) > node_cap:
                return None
        return None

    def field(self, dstar: int) -> tuple:
        """``(succ, dist)`` -- the exact distance-to-win field within ``d*`` of
        the start, over the states a shortest plan can pass through.

        Forward sweep keeping a successor only when ``g + h <= d*``.  With an
        admissible, consistent ``h`` that keeps every state on every shortest
        path: for a state ``u`` on one, ``g(u) + h(u) <= g(u) + dist(u) = d*``,
        and the same holds of each of its prefixes -- so the backward BFS that
        follows measures the true distance-to-win for all of them.
        ``dist[start] == d*`` at the end is the check that it did.
        """
        start = self.start
        gval = {start: 0}
        succ: dict = {}
        queue = deque([start])
        while queue:
            state = queue.popleft()
            g = gval[state]
            edges: dict = {}
            for d in DIRS:
                nxt = self.step(state, d)
                if nxt == state:
                    continue
                if self.won(nxt):
                    edges[d] = _WIN
                    continue
                if nxt in gval:
                    edges[d] = nxt             # keep the edge for the sweep
                    continue
                if g + 1 + self.heuristic(nxt) > dstar:
                    continue                   # cannot be on a shortest path
                gval[nxt] = g + 1
                edges[d] = nxt
                queue.append(nxt)
            succ[state] = edges
        return succ, _backward(succ)

    def plan(self, node_cap: int = 1_500_000,
             weight: int = 1) -> "Plan | None":
        """The plan and its per-step optimal SETS.

        At ``weight == 1`` the sets are MEASURED off the exact field, and the
        plan's length is checked against `d*` -- it is provably shortest.  At a
        larger weight there is no such guarantee and no field to read sets off,
        so every step is labelled with itself alone: a genuine win, honestly
        labelled, which is what level 9 gets.

        Ties are broken by `DIRS` order, which is what makes a re-derived plan
        byte-identical across processes (and therefore what lets the disk cache
        be trusted between `parallelize_generator` shards).
        """
        found = self.astar(node_cap, weight)
        if found is None:
            return None
        if weight != 1:
            return Plan(found, [[d] for d in found])
        dstar = len(found)
        succ, dist = self.field(dstar)
        if dist.get(self.start) != dstar:
            return None
        presses, optsets = [], []
        state = self.start
        while True:
            best = [d for d in DIRS
                    if _cost(succ, dist, state, d) == dist[state]]
            presses.append(best[0])
            optsets.append(best)
            nxt = succ[state][best[0]]
            if nxt == _WIN:
                return Plan(presses, optsets)
            state = nxt


def _cost(succ: dict, dist: dict, state, d) -> "int | None":
    """Presses to win if ``d`` is pressed at ``state``, or None when that press
    is unavailable (it changed nothing) or leads nowhere."""
    nxt = succ[state].get(d)
    if nxt is None:
        return None
    if nxt == _WIN:
        return 1
    rest = dist.get(nxt)
    return None if rest is None else rest + 1


def _backward(succ: dict) -> dict:
    """Presses-to-win for every state, by backward BFS from the winning edges.
    States absent from the result cannot win within the swept subgraph."""
    rev: dict = {}
    dist: dict = {}
    frontier = []
    for state, edges in succ.items():
        for nxt in edges.values():
            if nxt == _WIN:
                if state not in dist:
                    dist[state] = 1
                    frontier.append(state)
            else:
                rev.setdefault(nxt, []).append(state)
    queue = deque(frontier)
    while queue:
        state = queue.popleft()
        for prev in rev.get(state, ()):
            if prev not in dist:
                dist[prev] = dist[state] + 1
                queue.append(prev)
    return dist


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class StandIIExpert(PSExpert):
    """`PSExpert`'s plan memo, snapshot discipline and disk cache around the
    native search.  `heuristic` is never called on the interpreter -- the model
    owns it -- so it asserts rather than returning a number nothing would use.
    """

    directions = list(DIRS)
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "stand_ii_plans.json")

    #: Levels planned at a weight above 1, i.e. WON but not proved shortest.
    #: Level 9 is 7 people, 7 lights and 11 crates: weight-1 A* was still at
    #: f = 29 after 117455 states and 200 s against a win at most 48 presses
    #: away, because the heuristic relaxes the crates away and so charges
    #: nothing for shoving eleven of them out of the corridors.  Declared per
    #: level rather than discovered with a wall-clock fallback so that two
    #: processes derive byte-identical plans and the disk cache stays portable.
    level_weight: dict[int, int] = {9: 3}

    def setup(self) -> None:
        self.ids = {n: self.g.obj_name_to_idx[n] for n in _OBJECTS}
        self.stats: dict[int, dict] = {}      # level -> what the search cost
        self._last: dict | None = None
        self._level: int | None = None

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "StandIIExpert plans on its native model; heuristic is unused")

    def board(self, eng) -> StandIIBoard:
        return StandIIBoard(eng, self.ids)

    def weight_for(self, level: "int | None") -> int:
        return self.level_weight.get(level, 1)

    def _search(self, eng) -> "Plan | None":
        board = self.board(eng)
        weight = self.weight_for(self._level)
        t0 = time.time()
        found = board.plan(node_cap=self.node_cap, weight=weight)
        self._last = {
            "people": len(StandIIBoard._cells(board.start[0])),
            "crates": len(StandIIBoard._cells(board.start[1])),
            "bigcrates": len(StandIIBoard._cells(board.start[2])),
            "lights": len(board.lights),
            "h0": board.heuristic(board.start),
            "weight": weight,
            "states": len(board._h_cache),
            "seconds": time.time() - t0,
        }
        return found

    def plan(self, eng, level=None):
        self._level, self._last = level, None
        got = super().plan(eng, level)
        if level is not None and self._last is not None:
            self.stats[level] = self._last
        return got

    def describe(self, level: int) -> str:
        got = self.stats.get(level)
        if got is None:
            return "cached"
        note = "" if got["weight"] == 1 else f", weight {got['weight']}"
        return (f"{got['states']} player layouts, h0 {got['h0']}, "
                f"{got['seconds']:.1f}s{note}")


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class StandIISolver(PSAStarSolver):
    game_id = GAME_ID
    game_name = GAME_NAME
    game_module_id = GAME_ID
    expert_cls = StandIIExpert
    #: Level 10's exact field alone touches 616374 states; the cap is the
    #: runaway guard, not the budget.
    node_cap = 1_500_000
    max_steps = 150                    # the longest plan is 48 presses


# ---------------------------------------------------------------------------
# --verify: the model against the interpreter, and the heuristic's two claims
# ---------------------------------------------------------------------------

def _engine_state(eng, ids: dict, width: int) -> tuple:
    """``(players, crates, bigcrates)`` read off the interpreter's grid."""
    out = [0, 0, 0]
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            bit = 1 << (r * width + c)
            for k, name in enumerate(("player", "crate", "bigcrate")):
                if ids[name] in cell:
                    out[k] |= bit
    return (out[0], out[1], out[2])


def verify(trials: int = 40, presses: int = 80, verbose: bool = True) -> int:
    """Four measurements, none of which the docstring is allowed to assert.

    1. **The model IS the game.**  Random play on every level, with the model
       and the interpreter stepped side by side and all three masks compared
       after every press, plus the win predicate.  Half the trials run on
       CROWDED boards -- extra Players, Crates and BigCrates scattered over the
       free cells before play starts.  That mode is not decoration: the only
       interesting parts of `step` are the queue rule (a chain moves iff its
       front can) and the two ways a queue stalls (an unpushed crate ahead, or
       a BigCrate eating its shover's force), and the shipped levels place
       their pieces far too sparsely to hit those often under random play.  It
       is the lesson ps:sokobaiogenesis learned the hard way -- a clean fuzz
       can be blind to exactly the configuration a shortest plan manufactures
       on purpose.

    2. **ACTION is inert.**  The file declares `noaction`; this presses it at
       every state of the fuzz and requires the grid not to change, which is
       what justifies leaving it out of `DIRS`.

    3. **The heuristic is ADMISSIBLE**: `h(s) <= dist(s)` at every state of
       every exact field.

    4. **The heuristic is CONSISTENT**: `h(s) <= 1 + h(s')` across every edge
       of every exact field.  This is the property the field sweep rests on (an
       inconsistent `h` could prune a state that is on a shortest path), so it
       is measured rather than argued.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    ids = {n: game._game.obj_name_to_idx[n] for n in _OBJECTS}
    rng = random.Random(f"{GAME_NAME}:verify")
    bad = checked = action_checked = 0

    for level in range(game.n_levels):
        game.set_level(level)
        board = StandIIBoard(eng, ids)
        free = [(r, c) for r in range(board.H) for c in range(board.W)
                if (r, c) not in board.walls]
        for trial in range(trials):
            game.set_level(level)
            crowded = trial >= trials // 2
            if crowded:
                spots = rng.sample(free, min(len(free), rng.randint(3, 14)))
                for r, c in spots:
                    cell = eng.grid[r][c]
                    cell -= {ids["player"], ids["crate"], ids["bigcrate"]}
                    cell.add(ids[rng.choice(("player", "player", "crate",
                                             "crate", "bigcrate"))])
                eng._position_index_dirty = True
                eng._rule_noop_cache.clear()
            state = _engine_state(eng, ids, board.W)
            for i in range(presses):
                if i % 7 == 3:                       # (2) ACTION is inert
                    before = [set(cell) for row in eng.grid for cell in row]
                    eng.step("action")
                    after = [set(cell) for row in eng.grid for cell in row]
                    action_checked += 1
                    if before != after:
                        bad += 1
                        if bad < 4:
                            print(f"  ACTION changed the grid on L{level}")
                d = rng.choice(DIRS)
                eng.step(d)
                got = _engine_state(eng, ids, board.W)
                want = board.step(state, d)
                checked += 1
                if got != want or eng.check_win() != board.won(got):
                    bad += 1
                    if bad < 4:
                        where = "crowded" if crowded else "plain"
                        print(f"  MISMATCH L{level} {where} press {d}: "
                              f"model {want} engine {got}")
                state = got
    if verbose:
        print(f"  model vs interpreter: {checked} presses, {bad} mismatches")
        print(f"  ACTION pressed {action_checked} times, always inert")

    inadmissible = inconsistent = states = fields = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = StandIIBoard(eng, ids)
        if StandIIExpert.level_weight.get(level, 1) != 1:
            continue                  # no exact field to measure against
        plan = board.plan(node_cap=StandIISolver.node_cap)
        if plan is None:                        # `--plans` reports the level
            continue
        succ, dist = board.field(len(plan))
        states += len(succ)
        fields += 1
        for state, edges in succ.items():
            here = dist.get(state)
            hv = board.heuristic(state)
            if here is not None and hv > here:
                inadmissible += 1
            for nxt in edges.values():
                other = 0 if nxt == _WIN else board.heuristic(nxt)
                if hv > 1 + other:
                    inconsistent += 1
    if verbose:
        print(f"  heuristic over {states} states of {fields} exact fields: "
              f"{inadmissible} inadmissible, {inconsistent} inconsistent edges")
    return bad + inadmissible + inconsistent


# ---------------------------------------------------------------------------
# --audit: every cell composition the game can draw
# ---------------------------------------------------------------------------

#: The lights layer, the solids layer and the decal layer.  A cell is one of
#: each, subject to `_reachable` below.
_LIGHTS = ("", "target1", "target2")
_SOLIDS = ("", "player", "wall1", "wall2", "crate", "bigcrate")
_DECALS = ("", "cratetop", "bigcratetop")


def _reachable(light: str, solid: str) -> bool:
    """Can a rendered frame ever show this light-and-solid pair?

    The two late rules are each other's inverse and both run before the frame
    is drawn (`late [Target1 Player] -> [Target2 Player]` /
    `late [Target2 no Player] -> [Target1]`), so a light shows its LIT face if
    and only if somebody is standing on it: `target2` never appears alone and
    `target1` never appears under a player.  And no legend character and no
    rule ever puts a light under a wall.  `audit` FUZZES this claim as well as
    using it."""
    if light == "target2":
        return solid == "player"
    if light == "target1":
        return solid not in ("player", "wall1", "wall2")
    return True


def _cases() -> dict:
    """``{name: (objects, meaning)}`` for every composition the game can draw.

    ``meaning`` is what the cell SAYS: whether a light is here, and which of
    the five behaviours occupies it.  `Wall1` and `Wall2` collapse to one
    behaviour -- neither appears in any rule, both merely occupy the pushables'
    collision layer -- and the crate-top decals say nothing at all, being
    re-derived from the crate below them every turn.  Two compositions may
    render identically only if their meanings match; see `audit`.
    """
    out = {}
    for light in _LIGHTS:
        for solid in _SOLIDS:
            if not _reachable(light, solid):
                continue
            for decal in _DECALS:
                objs = tuple(o for o in (light, solid, decal) if o)
                name = "+".join(objs) if objs else "floor"
                meaning = (light != "",
                           "wall" if solid in ("wall1", "wall2") else solid)
                out[name] = (objs, meaning)
    return out


def audit(verbose: bool = True) -> int:
    """Assert every cell COMPOSITION that means something different renders
    differently, at every cell size the levels use -- and that no rotation or
    mirror of one is another with a different meaning.

    Composition, not object: both bugs this is the regression test for were in
    the STACK.  `player on target2` -- the winning square of the game -- was
    the Background's own green under a body, so it was one picture with `player`
    on grass; and `bigcrate on target1` lost the light entirely at the 4 px per
    cell that levels 5 and 7 render at.

    MEANING, not identity: `wall1+cratetop` and `wall2+cratetop` are one
    picture and are meant to be.  Wall1 is a solid brown cell, Wall2 is the
    same brown with its bottom row clear, and a crate's bottom decal fills
    exactly the gap that told them apart -- but the two objects are two paint
    jobs for one behaviour, so the frame is not hiding anything.  What would be
    a bug is two compositions that SAY different things sharing a picture.

    The board is FILLED with the composition and whole frames are compared,
    rather than one cell being sliced out by ``cell_px`` arithmetic:
    `_render_frame` upscales a sub-64 render to fill the frame and then
    letterboxes it, so the cell grid in the output is not ``cell_px``-aligned
    and the crop lands in the wrong window.

    The eight-way pass is what the presentation augmentation needs: this game
    takes rotations (every ps: game does) and, per `_FLIP_GAMES`, mirrors too,
    so a transform of one composition that equalled a composition MEANING
    something else would make the corpus say two different things about one
    picture.

    Finally the declared reachable set is FUZZED: random play on every level,
    with every composition that appears on the board checked against
    `_reachable`.  That is what keeps `player on target1` out of the audit
    honest -- it is excluded because the late rules make it unrenderable, and
    this measures that rather than believing it."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, parsed = game._engine, game._game
    idx = parsed.obj_name_to_idx
    cases = _cases()
    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots, meaning = {}, {}
        for name, (objs, says) in cases.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, parsed))
            meaning[name] = says
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if meaning[a] != meaning[b]
                   and np.array_equal(shots[a], shots[b])]
        turned = [(a, b, k, hf)
                  for a, b in itertools.permutations(shots, 2)
                  for k in range(4) for hf in (False, True)
                  if meaning[a] != meaning[b]
                  and np.array_equal(_transform(shots[a], k, hf, False),
                                     shots[b])]
        synonyms = [(a, b) for a, b in itertools.combinations(shots, 2)
                    if meaning[a] == meaning[b]
                    and np.array_equal(shots[a], shots[b])]
        bad += len(clashes) + len(turned)
        if verbose:
            verdict = ("all distinct" if not clashes
                       else "IDENTICAL " + str(clashes))
            turn_note = ("" if not turned
                         else ", TRANSFORM CLASH " + str(turned))
            syn_note = ("" if not synonyms
                        else f", {len(synonyms)} synonymous pairs "
                             f"({', '.join(a + '==' + b for a, b in synonyms)})")
            print(f"  {h}x{w} (cell {min(64 // h, 64 // w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): "
                  f"{len(cases)} compositions, {verdict}{turn_note}{syn_note}")

    # The reachable set, fuzzed rather than believed.
    rng = random.Random(f"{GAME_NAME}:audit")
    unexpected: set = set()
    seen: set = set()
    solids = {idx[o]: o for o in _SOLIDS if o}
    lights = {idx[o]: o for o in _LIGHTS if o}
    for level in range(game.n_levels):
        for _ in range(6):
            game.set_level(level)
            for _ in range(60):
                for row in eng.grid:
                    for cell in row:
                        light = "".join(lights[o] for o in cell if o in lights)
                        solid = "".join(solids[o] for o in cell if o in solids)
                        seen.add((light, solid))
                        if not _reachable(light, solid):
                            unexpected.add((light, solid))
                eng.step(rng.choice(DIRS))
    bad += len(unexpected)
    if verbose:
        print(f"  reachability fuzz: {len(seen)} light/solid pairs seen, "
              f"{len(unexpected) or 'none'} outside the declared set"
              + (f" {sorted(unexpected)}" if unexpected else ""))
    return bad


def _transform(frame, k: int, hflip: bool, vflip: bool):
    if k:
        frame = np.rot90(frame, k=k)
    if hflip:
        frame = np.fliplr(frame)
    if vflip:
        frame = np.flipud(frame)
    return np.ascontiguousarray(frame)


# ---------------------------------------------------------------------------
# --symmetry: the presentation augmentation is a true symmetry of the mechanic
# ---------------------------------------------------------------------------

def symmetry(walk_presses: int = 200, verbose: bool = True) -> int:
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unaugmented ones.

    This is the evidence for putting the game in `_FLIP_GAMES`.  The structural
    argument is the ESCAPE! one -- gravity-free, screen-relative input, a win
    condition that names no direction (`No Target1`), and no sprite that
    encodes a facing.  Two things here are worth measuring anyway.  A press
    moves up to SEVEN bodies at once, which is the shape that hid Gobble Rush's
    rule-order chirality; it is safe because every force a press assigns points
    the SAME way, so two chains can never claim one cell and the order
    `_resolve_forces` traced them in cannot decide anything.  And one rule in
    the file names a direction -- `late up [crate|no crateTop]`, which paints a
    decal on the engine-up side of every crate -- but that decal is deleted and
    re-derived every turn, lives in its own collision layer and is read by
    nothing, so a flipped presentation just draws crate tops on the flipped
    side.

    Both the PLANS and a seeded random walk are replayed -- the walk is what
    reaches the boards a plan never visits (crates jammed into corners, whole
    queues pressed into a wall, the unbound ACTION key), and a plan is what
    puts seven bodies on seven lights at once."""
    solver = StandIISolver()
    game, expert, _ = solver._ensure(0)
    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        plans[level] = expert.plan(game._engine, level)

    def drive(g, level, presses):
        g.set_level(level)
        out = [np.asarray(g._current_frame)]
        for act in presses:
            fd = g.perform_action(ActionInput(id=act))
            out.append(np.asarray(fd.frame[-1] if fd.frame
                                  else g._current_frame))
        return out

    ref, seen, bad = {}, set(), 0
    for seed in range(400):
        if len(seen) == 16 and seed > 40:
            break
        g = solver.make_game(seed)
        for level in range(g.n_levels):
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            rng = random.Random(f"{GAME_NAME}:symmetry:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(walk_presses)]
            plan = plans[level] or []
            for tag, presses in (("plan", [screen_action(d, *k)
                                           for d in plan]),
                                 ("walk", [inverse_remap_action_full(a, *k)
                                           for a in walk])):
                frames = drive(g, level, presses)
                if tag == "plan" and g._state != GameState.WIN:
                    print(f"  seed {seed} L{level} {k}: plan did not win")
                    bad += 1
                key = (level, tag)
                if key not in ref:
                    if k != (0, False, False):
                        continue          # nothing to compare against yet
                    ref[key] = frames
                    continue
                if any(not np.array_equal(_transform(a, *k), b)
                       for a, b in zip(ref[key], frames)):
                    print(f"  seed {seed} L{level} {k}: {tag} frames are not "
                          f"the transform of the unaugmented ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
    return bad


# ---------------------------------------------------------------------------
# --bfs: the independent optimality proof, for the levels whose space fits
# ---------------------------------------------------------------------------

def bfs_report(levels=(0, 1, 2, 3, 4, 5, 6, 7, 8), verbose: bool = True) -> int:
    """Exhaustive breadth-first press search to the depth of the first win, and
    the shortest win it finds compared against the planned one.

    This is the check that the heuristic is not quietly losing a shorter
    answer: the BFS has no notion of a matching, a debt or a bound, it just
    presses all four keys breadth-first and stops expanding once it is as deep
    as the best win so far -- which is exhaustive for the question being asked,
    since nothing beyond that depth can be shorter.  The nine levels in the
    default cost 43 / 984 / 8 / 99 / 57057 / 1421 / 23 / 4474 / 2727 states
    between them and all nine come back PROVED SHORTEST.  Level 10 does too --
    ``--bfs 10``, measured -- but on its own it is 12759045 states and several
    GB, so it is opt-in rather than part of the default sweep.  Level 9 is the
    only level not proved by anything (see the module docstring)."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    ids = {n: game._game.obj_name_to_idx[n] for n in _OBJECTS}
    bad = 0
    for level in levels:
        game.set_level(level)
        board = StandIIBoard(eng, ids)
        plan = board.plan(node_cap=StandIISolver.node_cap)
        if plan is None:
            print(f"  L{level}: UNSOLVED")
            bad += 1
            continue
        dist = {board.start: 0}
        queue = deque([board.start])
        best = None
        while queue:
            state = queue.popleft()
            if best is not None and dist[state] >= best:
                continue
            for d in DIRS:
                nxt = board.step(state, d)
                if nxt == state:
                    continue
                if board.won(nxt):
                    if best is None or dist[state] + 1 < best:
                        best = dist[state] + 1
                    continue
                if nxt not in dist:
                    dist[nxt] = dist[state] + 1
                    queue.append(nxt)
        ok = best == len(plan)
        bad += 0 if ok else 1
        if verbose:
            print(f"  L{level}: {len(dist)} reachable states, shortest win "
                  f"{best} presses, plan {len(plan)} -- "
                  f"{'PROVED SHORTEST' if ok else 'MISMATCH'}")
    return bad


# ---------------------------------------------------------------------------
# --plans
# ---------------------------------------------------------------------------

def plan_report() -> None:
    """Every level's plan, replayed through the real interpreter -- the
    end-to-end test of the model, the search and the tie labelling at once."""
    solver = StandIISolver()
    game, expert, solvable = solver._ensure(0)
    eng = game._engine
    total = ties = proved = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level}: UNSOLVED")
            continue
        for d in plan:
            eng.step(d)
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        step_ties = sum(len(s) - 1 for s in sets)
        weight = expert.weight_for(level)
        proved += weight == 1
        total += len(plan)
        ties += step_ties
        print(f"  L{level}: {eng.height}x{eng.width}, "
              f"{len(StandIIBoard._cells(board.start[0]))} people / "
              f"{len(board.lights)} lights / "
              f"{len(StandIIBoard._cells(board.start[1]))} crates / "
              f"{len(StandIIBoard._cells(board.start[2]))} big, "
              f"h0 {board.heuristic(board.start)}, "
              f"{len(plan):3d} presses  win={eng.check_win()}  "
              f"{'shortest' if weight == 1 else 'WON only'}  "
              f"{step_ties:2d} tie-presses  [{expert.describe(level)}]")
    print(f"  solvable levels: {solvable}")
    print(f"  {proved} of {len(solvable)} plans proved shortest")
    print(f"  {total} presses, {ties} of them with a second right answer "
          f"({(total + ties) / max(1, total):.2f} optimal presses per step)")


if __name__ == "__main__":
    if "--verify" in sys.argv:
        violations = verify()
        print(f"verify: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--bfs" in sys.argv:
        args = [int(a) for a in sys.argv[sys.argv.index("--bfs") + 1:]
                if a.isdigit()]
        violations = bfs_report(tuple(args) if args
                                else (0, 1, 2, 3, 4, 5, 6, 7, 8))
        print(f"bfs: {violations} mismatches")
        sys.exit(1 if violations else 0)
    if "--plans" in sys.argv:
        plan_report()
        sys.exit(0)
    sys.exit(StandIISolver.main())
