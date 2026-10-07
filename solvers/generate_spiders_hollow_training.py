"""ps:spiders_hollow -- John Thyer's "Spider's Hollow": a fairy walks into a
tree, and the way out of every room is a HOLE in the floor.

    [ > Player | Hole no Block ] -> [ | Player Hole ] Sfx0 Win

Eleven levels, all eleven won, and every plan PROVED SHORTEST -- not "A* with a
weight of 1 and a heuristic that looked admissible", but read off an EXACT
distance-to-win field (see THE FIELD below).  Part of the
`solvers/common/ps_astar.py` family, and unusual in it for having NO native
model at all: the interpreter is fast enough here, and everything that would
have needed a model -- optimal-action sets, a proof of optimality -- comes out
of the field instead, which cannot drift from the game because it IS the game.

THE THREE MECHANICS, and the one thing each of them is not

* **A WEB COSTS THREE PRESSES, and any three will do.**

      [ > Player Web1 ] -> [ Player Web2 ]
      [ > Player Web2 ] -> [ Player Web3 ]

  Neither rule moves you; each just advances the web under you and eats the
  press.  So standing on a fresh web, press one charges it to Web2, press two
  to Web3, and press three -- the first press no rule cancels -- finally moves
  you.  Two consequences the rules do not say out loud.  The *direction* of the
  first two presses is free: `>` with no direction prefix expands to all four,
  so any movement key charges the web, INCLUDING one pressed straight into a
  wall (the force is assigned before collisions are resolved, so a refused move
  still charges).  That is most of what the optimal-action sets in this corpus
  are: on a charging press, every direction ties.  And a web you step onto is
  always FRESH, because

      [ Player | Web2 ] -> [ Player | Web1 ]
      [ Player | Web3 ] -> [ Player | Web1 ]

  resets every web ORTHOGONALLY ADJACENT to the player, and you were adjacent
  to that cell on the turn you moved into it.  You can never bank a charge, and
  a web you leave is Web1 again by the time you could come back.  Both facts
  are what make the heuristic below exactly tight rather than merely
  admissible.

* **A SPIDER WALKS A SHORTEST PATH TO YOU, THROUGH ANY GEOMETRY.**  The chase
  is Maadball's three-colour flood written out in PuzzleScript: seed `Path1`
  around the spider, cycle `Path1 -> Path2 -> Path3 -> Path1` outward in a
  `startloop` until the board is coloured by distance mod 3, mark the player's
  cell `RightPath`, walk that marker back down the colour cycle to the spider,
  and give the spider one step along it.  `Obstacle = Wall or Block` is the
  only thing the flood will not cross -- so a crate is cover and a web is not,
  and a spider is never confused by a dead end.  Reaching you is fatal:

      [ > Spider | Player ] -> [ | SpiderRed ]

  deletes the player outright.  There is no lose condition; the board simply
  has no player on it any more, which is why `dead` is "no player" and not any
  cleverer predicate.

* **THE SPIDER IS ON A FUSE, and the fuse is four bookkeeping objects.**
  `Timer` marches one cell right per turn (`[ Timer ] -> [ Right Timer ]`)
  until it is beside a `TimerAlarm`, and

      late [ Timer | TimerAlarm ][ SpiderSleep ] -> [ | ][ Spider ]

  drops a spider onto the `SpiderSleep` cell -- which on every level that has
  one is the cell the PLAYER STARTED ON.  Levels 5-8 give you 6 to 10 presses
  of quiet, and the last level chains two fuses (`Timer2` wakes a `TimerSleep`
  into a second `Timer`) for 21.  `_fuse` reads that countdown off the board
  arithmetically and `--verify` checks it against the interpreter's own answer
  on every level.

THE LAST LEVEL IS WON BY DYING

Level 10 has no hole.  Its win is the other one:

    late [ TrappedFairy ][ SpiderRed ] -> [ TrappedFairy ][ SpiderRed ] Win

`SpiderRed` is the corpse marker the death rule leaves behind, so the only way
to satisfy it is to LET THE SPIDER EAT YOU, in the room where the four fairies
from the opening are hanging in webs.  ("And she'll never be apart from them
again.")  The catch is that the spawn cell is the player's own, and a spider
spawned into an occupied cell REPLACES the player -- same collision layer --
leaving a board with no player, no corpse and no way to win.  So the level is:
get off your own square within 21 presses, then let it come.  Every square is a
web, so getting off costs exactly 3, and 22 is the whole of it -- 21 for the
fuse and 1 for the spider's step.

That makes the last level's LABELS unlike anything else here: past those first
three presses the fuse decides the length and nothing the player does can
shorten it, so nearly every press ties.  21 of its 22 steps carry a second
right answer (4.59 optimal presses per step) and ACTION is one of them on all
21 -- it is the only level in the game where waiting is ever optimal, and the
plan the tie-break happens to emit spends none of its presses on it, wandering
the web room instead.  Both readings are labelled equally right, which is
exactly what the corpus should say.

THE HEURISTIC: a walk length that already knows what the webs cost

`Terrain` runs one Dijkstra out from the hole in which entering a cell costs 1
and LEAVING a web cell costs 2 more, and `heuristic` subtracts the charge
already on the web the player is standing on.  It is admissible because every
press either moves the player one cell, charges the web under it, or does
nothing at all, and it is TIGHT because of the reset rule above: a web is
always Web1 when you arrive, so 2 is the exact toll, not a guess.  Blocks are
counted as floor (a push costs the same one press as a step, and a push the
board refuses costs more), spiders are ignored entirely, and both relaxations
only ever make the estimate smaller.  Consistency -- needed for the field
sweep, not just for A* -- follows from the same three cases.  On level 10 the
estimate is the fuse instead: at least `_fuse` presses before a spider exists,
plus at least one more for it to reach you.  `--verify` MEASURES both claims
(`h <= dist` at every state of every field, `h(s) <= 1 + h(s')` across every
edge) rather than resting on this paragraph.

THE FIELD: A* for the bound, then one sweep for everything else

`_astar` proves `d*` (weight 1, admissible heuristic).  `_field_plan` then
sweeps forward from the level start keeping only states with `g + h <= d*` --
which, with a consistent `h`, is a superset of every state on every shortest
path -- and runs a backward BFS from the winning edges over exactly those
edges.  That gives the exact distance-to-win for every state a shortest plan
can pass through, and from it, in one pass and with no re-solving:

* a plan that is provably shortest (its length is checked against `d*`);
* the EXACT optimal-action set at every step -- every press that still finishes
  in the same number of moves -- instead of the inferred sets a walk annotator
  would produce.  This game needs them badly: 93 of the 233 presses in the
  eleven plans have a second right answer, most of them because a press spent
  charging a web is the same press whichever direction it is.

The bound and the sweep are the whole cost of the generator: 4678 states and
23390 interpreter presses over the eleven levels, 498 s cold, with level 8
(2154 states) and the last level (1374) between them more than half of it.  It
is seed-independent, so it lands in `data/spiders_hollow_plans.json` and only
the first process to run pays for it -- and a rebuild under a different
PYTHONHASHSEED reproduces that file byte for byte, which is what lets
`parallelize_generator` shards start warm.

WHAT WAS INVISIBLE.  Five of this game's objects shipped with a sprite of five
blank rows, the three web states shipped with the SAME sprite and the same two
greys, and the wood floor of the opening scenes was pixel-identical to the wall
-- so the charge under the player's feet, the spider's nest, the whole
countdown and the difference between floor and wall were all absent from the
frame.  Nine objects in `data/puzzlescript_games/Spider's_Hollow.txt` are
repainted or redrawn (its header has the detail); `--audit` is the regression
test.  No rule, level, object or collision layer was touched, and since this
expert plans by stepping the interpreter there is no model that could have been
tuned to the change.

VERIFIED.  11/11 levels, 233 presses, every plan its level's proven `d*`.
`--verify`: the fuse matches the interpreter on all five levels that have one
(levels 5-8 at 10, 7, 9 and 6 presses, and the last level's chained 21), and
`--plans` re-derives every plan; over all 4678 field states
the heuristic is 0 times inadmissible and 0 edges inconsistent.  `--audit`: 66
compositions, all pixel-distinct at cell_px 5, the only size the levels use.
`--symmetry`: 11 levels x 4 presentations, every plan a WIN and every frame of
both the plan and a 120-press random walk exactly the transform of the
unaugmented one.  6 seeds x 11 levels recorded in 14 s: every level a WIN, all
1469 steps replay frame-exact through the adapter from the recorded SCREEN
actions, 1398/1398 expert steps labelled (2.21 optimal presses/step, the taken
press always inside its own set), indices in 0..5, and two processes at
different hash seeds emit byte-identical episodes.  The longest plan is 45
presses against the adapter's 200-press cap.

RECOVERY is the family's RESET prefix (`recovery_mode = "reset"`).  It has to
be: the fuse cannot be rewound, a spider cannot be un-spawned, and being eaten
is unrecoverable on ten of the eleven levels -- so one RESET back to the level
start is the only sound undo, and it lands on exactly the state the cached
field was built from.

CLI: ``--verify`` (fuse against the interpreter, admissibility and consistency
against the exact field), ``--audit`` (every cell composition the game can
draw, pixel-distinct at the cell size the levels use), ``--symmetry`` (every
presentation is an exact transform of the unaugmented frames and every plan
still wins through the adapter), ``--plans`` (every level's plan, replayed
through the interpreter), otherwise the BaseSolver CLI.
"""

from __future__ import annotations

import heapq
import itertools
import random
import struct
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameState                      # noqa: E402
from adapters.puzzlescript_adapter import (                       # noqa: E402
    PuzzleScriptAdapter, _render_cell_sprite,
)
from solvers.base_solver import _ID_TO_GAMEACTION                 # noqa: E402
from solvers.common.ps_astar import (                             # noqa: E402
    DIRECTIONS, PSAStarSolver, PSExpert, Plan, restore, screen_action,
    snapshot,
)
from utils.rotation import inverse_remap_action_full              # noqa: E402

GAME_ID = "ps:spiders_hollow"
GAME_NAME = "Spider's_Hollow"

#: Engine direction -> (dr, dc).  ACTION is a pure WAIT in this game and has no
#: delta; it is still in `PSExpert.directions` because on the last level the
#: shortest plan is mostly waiting.
_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}

#: Sentinel successor for "this press wins".  A `str` can never collide with a
#: packed board, which is `bytes`.
_WIN = "win"

#: Heuristic for a board whose hole no walk can reach.  Finite so the search
#: stays complete, large enough that such a state sinks.
_UNREACHABLE = 10_000

#: Objects the expert resolves by name.  Everything else on the board (the
#: sky, the leaves, the wood, the entrance) is scenery no rule reads.
_OBJECTS = ("wall", "player", "block", "hole", "web1", "web2", "web3",
            "spider", "spiderred", "spidersleep", "trappedfairy",
            "timer", "timer2", "timeralarm", "timersleep")


# ---------------------------------------------------------------------------
# The static half of a level
# ---------------------------------------------------------------------------

class Terrain:
    """Walls, webs, the hole -- and the walk-cost field out of the hole.

    Every one of those is static for the whole of a level: no rule in this game
    creates or destroys a wall, moves a hole, or turns a web cell into anything
    but another web.  So the field is built ONCE per search instead of inside
    `HollowExpert.heuristic`, which runs on every node of it.
    """

    def __init__(self, eng, ids: dict[str, int]):
        grid = eng.grid
        self.H, self.W = len(grid), len(grid[0])
        webs = {ids["web1"], ids["web2"], ids["web3"]}
        self.walls: set[tuple[int, int]] = set()
        self.webs: set[tuple[int, int]] = set()
        self.hole: tuple[int, int] | None = None
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                if ids["wall"] in cell:
                    self.walls.add((r, c))
                if cell & webs:
                    self.webs.add((r, c))
                if ids["hole"] in cell:
                    self.hole = (r, c)
        self.dist = {} if self.hole is None else self._walk_field()

    def _walk_field(self) -> dict[tuple[int, int], int]:
        """``{cell: presses to walk from cell into the hole}``, webs included.

        Dijkstra out from the hole where the edge ``v -> u`` costs ``1`` for
        the step plus ``2`` when ``v`` is a web -- the two presses the web eats
        before it lets go.  The toll is charged on LEAVING, so the hole itself
        (which is never a web, and which the player only ever enters) is free
        and the cell the player is standing on pays.
        """
        dist = {self.hole: 0}
        queue = [(0, self.hole)]
        while queue:
            d, cell = heapq.heappop(queue)
            if d > dist[cell]:
                continue
            r, c = cell
            for dr, dc in _DELTA.values():
                nxt = (r + dr, c + dc)
                if not (0 <= nxt[0] < self.H and 0 <= nxt[1] < self.W):
                    continue
                if nxt in self.walls:
                    continue
                nd = d + 1 + (2 if nxt in self.webs else 0)
                if nd < dist.get(nxt, 1 << 30):
                    dist[nxt] = nd
                    heapq.heappush(queue, (nd, nxt))
        return dist


# ---------------------------------------------------------------------------
# The fuse
# ---------------------------------------------------------------------------

def _reach(pos, alarms) -> "tuple[int, tuple[int, int]] | None":
    """``(presses until a timer at ``pos`` ends a turn beside an alarm, that
    alarm)``, or None if it never does.

    A timer only ever moves RIGHT, one cell per turn, and it shares a collision
    layer with the alarms -- so it can reach a same-row alarm only from the
    left, and a same-column alarm only by landing on its column.
    """
    if pos is None:
        return None
    r, c = pos
    best = None
    for alarm in alarms:
        ar, ac = alarm
        if ar == r:
            k = ac - 1 - c
        elif abs(ar - r) == 1:
            k = ac - c
        else:
            continue
        if k >= 0 and (best is None or k < best[0]):
            best = (k, alarm)
    return best


def _fuse(timer, timer2, sleep, alarms) -> "int | None":
    """Presses until a `Spider` appears, from the timekeeping objects alone.

    Two shapes, and the second is only on the last level.  A live `Timer` is
    read straight off `_reach`, because the spawn is a LATE rule and so fires
    on the very turn the timer lands beside its alarm.  A `Timer2` instead
    wakes a `TimerSleep` into a second `Timer`, and that rule is a MAIN one --
    it matches the board BEFORE the turn's movement -- so it fires one turn
    later than the arithmetic suggests, and the timer it creates has already
    missed that turn's move.  Hence ``k + 1 + rest``.  The alarm the first fuse
    consumes is struck off for the second: both are spent by their rules.
    """
    if timer is not None:
        got = _reach(timer, alarms)
        return None if got is None else got[0]
    if timer2 is not None and sleep is not None:
        got = _reach(timer2, alarms)
        if got is None:
            return None
        k, used = got
        rest = _reach(sleep, [a for a in alarms if a != used])
        return None if rest is None else k + 1 + rest[0]
    return None


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class HollowExpert(PSExpert):
    """A* over the real interpreter for the bound, then one exact field sweep
    for the plan and its optimal-action sets.

    `PSExpert` supplies the plan memo, the snapshot discipline and the on-disk
    cache; `_search` is replaced the way the enumerating experts in this family
    replace it, because the strategy underneath is not "return the first
    winning path A* finds" but "prove `d*`, then measure everything".
    """

    directions = list(DIRECTIONS)          # up / down / left / right / action
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "spiders_hollow_plans.json")

    def setup(self) -> None:
        g = self.g
        self.ids = {n: g.obj_name_to_idx[n] for n in _OBJECTS}
        self.i_player = self.ids["player"]
        self.i_web2, self.i_web3 = self.ids["web2"], self.ids["web3"]
        n_objects = 1 + max(g.obj_idx_to_name)
        if n_objects > 32:
            raise AssertionError(
                f"{n_objects} objects will not fit the 32-bit cell mask "
                "`pack` uses")
        self.terrain: Terrain | None = None
        self._mask_cache: dict[int, tuple[int, ...]] = {}
        self._shape: tuple[int, int] | None = None
        self.stats: dict[int, dict] = {}       # level -> what the sweep cost
        self._last: dict | None = None         # the most recent field sweep

    # -- board packing (so the sweep keeps no snapshots) ---------------------
    #
    # A `snapshot` is a fresh `set` per cell -- tens of KB apiece -- so a sweep
    # that keeps one per state is bounded by RAM rather than by time, and how
    # far the `g + h <= d*` prune reaches is not known before it runs.  `pack`
    # is instead a LOSSLESS spelling of the whole grid in four bytes per cell,
    # one bit per object index, so a state is re-seated by decoding its own
    # key and nothing but the key is stored.  It is deliberately not `_key`:
    # that one is a set of `(r, c, obj)` triples because `PSExpert.plan` writes
    # it out as the disk cache's level signature.

    def pack(self, eng) -> bytes:
        h, w = len(eng.grid), len(eng.grid[0])
        self._shape = (h, w)
        return struct.pack(
            f"<{h * w}I",
            *[sum(1 << o for o in cell) for row in eng.grid for cell in row])

    def unpack(self, eng, key: bytes) -> None:
        h, w = self._shape
        masks = struct.unpack(f"<{h * w}I", key)
        cache = self._mask_cache
        grid = []
        i = 0
        for _r in range(h):
            row = []
            for _c in range(w):
                m = masks[i]
                i += 1
                objs = cache.get(m)
                if objs is None:
                    objs = cache[m] = tuple(o for o in range(32) if m >> o & 1)
                row.append(set(objs))
            grid.append(row)
        eng.grid = grid
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()

    # -- the state of the world, in one pass ---------------------------------
    def read(self, eng) -> dict:
        """Everything `heuristic` and `dead` need, from a single grid scan."""
        ids = self.ids
        out = {"player": None, "spider": None, "timer": None, "timer2": None,
               "sleep": None, "alarms": [], "charge": 0}
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                if ids["player"] in cell:
                    out["player"] = (r, c)
                    out["charge"] = (2 if self.i_web3 in cell else
                                     1 if self.i_web2 in cell else 0)
                if ids["spider"] in cell:
                    out["spider"] = (r, c)
                if ids["timer"] in cell:
                    out["timer"] = (r, c)
                if ids["timer2"] in cell:
                    out["timer2"] = (r, c)
                if ids["timersleep"] in cell:
                    out["sleep"] = (r, c)
                if ids["timeralarm"] in cell:
                    out["alarms"].append((r, c))
        return out

    def dead(self, eng) -> bool:
        """No player on the board.  A spider that reaches you deletes you
        outright (`[ > Spider | Player ] -> [ | SpiderRed ]`) and a spider
        spawned onto your own cell replaces you, so this is the game's only
        failure -- and on the last level it is checked AFTER the win, which is
        exactly that corpse."""
        pid = self.i_player
        return not any(pid in cell for row in eng.grid for cell in row)

    def heuristic(self, eng) -> int:
        terrain = self.terrain
        seen = self.read(eng)
        player = seen["player"]
        if player is None:
            return 0
        if terrain.hole is not None:
            d = terrain.dist.get(player)
            # The walk field charges every web 2 presses; the one under the
            # player has already been charged `seen["charge"]` of them.
            return _UNREACHABLE if d is None else d - seen["charge"]
        # No hole: the last level, whose win is being eaten.
        spider = seen["spider"]
        if spider is not None:
            # They close on each other by at most two cells a turn, and the
            # kill is the spider's step INTO the player.  Manhattan (never
            # more than the walk distance) keeps it a lower bound.
            gap = abs(spider[0] - player[0]) + abs(spider[1] - player[1])
            return gap // 2 + 1
        left = _fuse(seen["timer"], seen["timer2"], seen["sleep"],
                     seen["alarms"])
        return 1 if left is None else left + 1

    # -- search --------------------------------------------------------------
    def plan(self, eng, level=None):
        """Bind this level's static terrain, then plan as usual.

        Ahead of `super()` rather than inside `_search`, because
        `PSExpert.plan` reaches the disk cache first and a hit must still leave
        a terrain behind for anything that asks the heuristic afterwards
        (`--verify` does)."""
        self.terrain = Terrain(eng, self.ids)
        self._last = None
        got = super().plan(eng, level)
        if level is not None and self._last is not None:
            self.stats[level] = {k: self._last[k]
                                 for k in ("states", "steps", "dstar")}
        return got

    def _search(self, eng) -> "Plan | None":
        if self.terrain is None:
            self.terrain = Terrain(eng, self.ids)
        start = snapshot(eng)
        found = self._astar(eng)               # leaves the grid dirty
        restore(eng, start)
        if found is None:
            return None
        return self._field_plan(eng, len(found))

    def _field_plan(self, eng, dstar: int) -> "Plan | None":
        """The exact distance-to-win field within ``d*`` of the start, and the
        plan and optimal-action sets read off it.  Engine restored.

        The forward sweep keeps a successor only when ``g + h <= d*``.  With an
        admissible, consistent ``h`` that keeps every state on every shortest
        path: for a state ``u`` on one, ``g(u) + h(u) <= g(u) + dist(u) = d*``,
        and consistency makes the same true of each of its prefixes -- so the
        backward BFS that follows measures the true distance-to-win for all of
        them.  ``dist(start) == d*`` at the end is the check that it did.
        """
        start = snapshot(eng)
        key = self.pack(eng)
        gval = {key: 0}
        succ: dict[bytes, dict[str, object]] = {}
        queue = deque([key])
        steps = 0
        while queue:
            k = queue.popleft()
            g = gval[k]
            edges: dict[str, object] = {}
            for direction in self.directions:
                self.unpack(eng, k)
                eng.step(direction)
                steps += 1
                if eng.check_win():
                    edges[direction] = _WIN
                    continue
                if self.dead(eng):
                    continue
                nxt = self.pack(eng)
                if nxt == k:
                    continue                   # a press that did nothing
                if nxt in gval:
                    edges[direction] = nxt     # keep the edge for the sweep
                    continue
                if g + 1 + self.heuristic(eng) > dstar:
                    continue                   # cannot be on a shortest path
                gval[nxt] = g + 1
                edges[direction] = nxt
                queue.append(nxt)
            succ[k] = edges
        restore(eng, start)

        dist = _backward(succ)
        if dist.get(key) != dstar:
            return None

        presses, optsets = [], []
        k = key
        while True:
            best = [d for d in self.directions
                    if _cost(succ, dist, k, d) == dist[k]]
            presses.append(best[0])
            optsets.append(best)
            nxt = succ[k][best[0]]
            if nxt == _WIN:
                break
            k = nxt
        self._last = {"states": len(succ), "steps": steps, "dstar": dstar,
                      "field": dist, "succ": succ}
        return Plan(presses, optsets)

    def describe(self, level: int) -> str:
        got = self.stats.get(level)
        return "cached" if got is None else (
            f"{got['states']} states, {got['steps']} presses")


def _cost(succ, dist, k, direction) -> "int | None":
    """Presses to win if ``direction`` is pressed at ``k``, or None when that
    press is unavailable (it did nothing, it died) or leads nowhere."""
    nxt = succ[k].get(direction)
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
    for k, edges in succ.items():
        for nxt in edges.values():
            if nxt == _WIN:
                if k not in dist:
                    dist[k] = 1
                    frontier.append(k)
            else:
                rev.setdefault(nxt, []).append(k)
    queue = deque(frontier)
    while queue:
        k = queue.popleft()
        for prev in rev.get(k, ()):
            if prev not in dist:
                dist[prev] = dist[k] + 1
                queue.append(prev)
    return dist


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class SpidersHollowSolver(PSAStarSolver):
    game_id = GAME_ID
    game_name = GAME_NAME
    game_module_id = GAME_ID
    expert_cls = HollowExpert
    max_steps = 300


# ---------------------------------------------------------------------------
# --verify: the fuse, and the two properties the optimality proof rests on
# ---------------------------------------------------------------------------

def verify(verbose: bool = True) -> int:
    """Three measurements, none of which the docstring is allowed to assert.

    1. **The fuse.**  `_fuse` reads a countdown off the timekeeping objects.
       Pressing ACTION from the level start advances nothing but that
       countdown, so the turn a `Spider` first appears on the board IS the
       ground truth, and it is compared against `_fuse` for every level.
    2. **Admissibility.**  ``h(s) <= dist(s)`` at every state of every exact
       field.  This is what makes ``d*`` a proof rather than a plan length.
    3. **Consistency.**  ``h(s) <= 1 + h(s')`` across every edge of every
       field.  This is what makes the ``g + h <= d*`` prune of the forward
       sweep lossless, and so the optimal-action sets exact.
    """
    solver = SpidersHollowSolver()
    game, expert, _ = solver._ensure(0)
    bad = 0

    # 1. the fuse against the interpreter
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        seen = expert.read(eng)
        claim = _fuse(seen["timer"], seen["timer2"], seen["sleep"],
                      seen["alarms"])
        if seen["spider"] is not None:
            # Level 4 hatches with its spider already on the board, so there is
            # no countdown to read and none to check.
            if verbose:
                print(f"  L{level:2d}: spider on the board from the start")
            continue
        spider_id = expert.ids["spider"]
        truth = None
        for turn in range(1, 200):
            eng.step("action")
            if any(spider_id in cell for row in eng.grid for cell in row):
                truth = turn
                break
        if claim != truth:
            print(f"  L{level:2d}: fuse says {claim}, interpreter "
                  f"says {truth}")
            bad += 1
        elif verbose:
            print(f"  L{level:2d}: fuse {claim} -- confirmed"
                  if claim is not None else f"  L{level:2d}: no fuse")

    # 2 + 3. the field, per level
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level:2d}: no plan")
            bad += 1
            continue
        got = getattr(expert, "_last", None)
        if got is None or got["dstar"] != len(plan):
            # A disk-cached plan carries no field; rebuild it here so the
            # properties are measured on this run rather than trusted.
            expert.terrain = Terrain(eng, expert.ids)
            probe = snapshot(eng)
            found = expert._astar(eng)
            restore(eng, probe)
            if found is None:
                print(f"  L{level:2d}: cached plan but A* found none")
                bad += 1
                continue
            expert._field_plan(eng, len(found))
            got = expert._last
        field, succ = got["field"], got["succ"]
        adm = incons = 0
        hcache = {}

        def h_at(k):
            if k not in hcache:
                expert.unpack(eng, k)
                hcache[k] = expert.heuristic(eng)
            return hcache[k]

        for k, remaining in field.items():
            if h_at(k) > remaining:
                adm += 1
        for k, edges in succ.items():
            if k not in field:
                continue
            for nxt in edges.values():
                target = 0 if nxt == _WIN else h_at(nxt)
                if h_at(k) > 1 + target:
                    incons += 1
        bad += adm + incons
        if verbose:
            print(f"  L{level:2d}: d*={len(plan):3d}  {len(succ):6d} states  "
                  f"{adm} inadmissible  {incons} inconsistent edges")
    return bad


# ---------------------------------------------------------------------------
# --audit: every composition the game can draw is pixel-distinct
# ---------------------------------------------------------------------------

#: Compositions the audit forgives, with the reason.  Both are decorative: the
#: three skies and the two leaves are backdrop on the two opening scenes, where
#: every cell is walkable and no rule reads any of them.
_AUDIT_IGNORE = ("sky1", "sky2", "sky3", "leaf1", "leaf2")


def audit(trials: int = 40, steps: int = 40, verbose: bool = True) -> int:
    """Collect every cell composition the game actually produces -- by walking
    each level randomly, not by enumerating the object powerset -- and require
    them to render differently at the cell size the levels use.

    Composition, not object: the three web charges were identical as objects,
    but so were `player over Web1` and `player over Web3`, and it is the second
    of those that decides whether the next press moves you.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    parsed = game._game
    layers = game._engine._obj_layers
    players = set(game._engine._player_indices)
    names = parsed.obj_idx_to_name

    comps: set[tuple[str, ...]] = set()
    sizes: set[int] = set()
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        sizes.add(max(1, min(64 // len(eng.grid), 64 // len(eng.grid[0]))))
        for trial in range(trials):
            game.set_level(level)
            eng = game._engine
            rng = random.Random(f"spiders_hollow:audit:{level}:{trial}")
            for _ in range(steps):
                for row in eng.grid:
                    for cell in row:
                        comps.add(tuple(sorted(names[o] for o in cell)))
                if eng.check_win():
                    break
                eng.step(rng.choice(DIRECTIONS))

    def stack(comp):
        body, player = [], []
        for name in comp:
            idx = parsed.obj_name_to_idx[name]
            (player if idx in players else body).append(
                (layers.get(idx, -1), parsed.objects[name]))
        body.sort(key=lambda x: x[0])
        player.sort(key=lambda x: x[0])
        return body + player

    def decorative(a, b):
        """True when a and b differ only in backdrop objects."""
        keep = tuple(n for n in a if n not in _AUDIT_IGNORE)
        return keep == tuple(n for n in b if n not in _AUDIT_IGNORE)

    bad = 0
    for px in sorted(sizes):
        blocks = {c: _render_cell_sprite(stack(c), px) for c in comps}
        clashes = [(a, b) for a, b in itertools.combinations(sorted(comps), 2)
                   if bool((blocks[a] == blocks[b]).all())
                   and not decorative(a, b)]
        for a, b in clashes:
            print(f"  cell_px={px}: {a} and {b} render identically")
        bad += len(clashes)
        if verbose:
            print(f"  cell_px={px}: {len(comps)} compositions, "
                  f"{'all distinct' if not clashes else 'COLLISIONS'}")
    return bad


# ---------------------------------------------------------------------------
# --symmetry: the presentation augmentation is exactly a transform
# ---------------------------------------------------------------------------

def symmetry(walk_presses: int = 120, verbose: bool = True) -> int:
    """Replay every level through the ADAPTER at every presentation it draws
    and require the frames to be exactly the transform of the unaugmented ones.

    The structural claim is stronger here than in most of this family and the
    check is correspondingly narrow: a ps: game's ENGINE is never rotated --
    the adapter turns the rendered frame and forward-remaps the input -- so the
    dynamics under every presentation are the same dynamics.  What this
    measures is therefore the `screen_action` contract (get the inverse remap
    backwards and the adapter executes a different direction than the expert
    planned, silently, on three of every four orientations) and that no sprite
    is drawn from anything but the grid.

    It is also the argument for leaving this game OUT of `_FLIP_GAMES` and
    taking only the four rotations every ps: game gets.  The chase resolves
    ties by the scan order of the flood's backtrack, which is chiral, and
    `[ Timer ] -> [ Right Timer ]` names an absolute direction outright -- so a
    mirrored presentation would show a spider stepping the way this game never
    steps.  The rotations are safe for the same reason they are safe
    everywhere here: the frames are exact transforms and the engine underneath
    is untouched.

    A seeded random WALK is replayed beside each plan; it is what reaches the
    boards a plan never visits (a spider on top of the player, a crate wedged
    on the hole) and it presses ACTION too.
    """
    solver = SpidersHollowSolver()
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

    def transform(frame, k, hflip, vflip):
        if k:
            frame = np.rot90(frame, k=k)
        if hflip:
            frame = np.fliplr(frame)
        if vflip:
            frame = np.flipud(frame)
        return np.ascontiguousarray(frame)

    ref, seen, bad = {}, set(), 0
    for seed in range(200):
        if len(seen) == 4 and seed > 20:
            break
        g = solver.make_game(seed)
        for level in range(g.n_levels):
            g.set_level(level)
            pres = (g._rotation_k, g._hflip, g._vflip)
            seen.add(pres)
            rng = random.Random(f"spiders_hollow:symmetry:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(walk_presses)]
            for tag, presses in (
                    ("plan", [screen_action(d, *pres) for d in plans[level]]),
                    ("walk", [inverse_remap_action_full(a, *pres)
                              for a in walk])):
                frames = drive(g, level, presses)
                if tag == "plan" and g._state != GameState.WIN:
                    print(f"  seed {seed} L{level} {pres}: plan did not win")
                    bad += 1
                slot = (level, tag)
                if slot not in ref:
                    if pres != (0, False, False):
                        continue           # nothing to compare against yet
                    ref[slot] = frames
                    continue
                if any(not np.array_equal(transform(a, *pres), b)
                       for a, b in zip(ref[slot], frames)):
                    print(f"  seed {seed} L{level} {pres}: {tag} frames are "
                          "not the transform of the unaugmented ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
    return bad


# ---------------------------------------------------------------------------
# --plans
# ---------------------------------------------------------------------------

def plan_report() -> None:
    """Every level's plan, replayed through the real interpreter -- the
    end-to-end test of the search, the field and the tie labelling at once."""
    solver = SpidersHollowSolver()
    game, expert, solvable = solver._ensure(0)
    total = ties = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level:2d}: UNSOLVED")
            continue
        for direction in plan:
            eng.step(direction)
            if eng.check_win():
                break
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        tied = sum(1 for s in sets if len(s) > 1)
        waitable = sum(1 for s in sets if "action" in s)
        total += len(plan)
        ties += tied
        print(f"  L{level:2d}: {len(plan):3d} presses  win={eng.check_win()}  "
              f"{tied:3d} steps with a second right answer  "
              f"({sum(len(s) for s in sets) / len(sets):.2f} optimal/step, "
              f"{waitable:3d} where waiting is one of them)  "
              f"{expert.describe(level)}")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second right answer")


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
    if "--plans" in sys.argv:
        plan_report()
        sys.exit(0)
    sys.exit(SpidersHollowSolver.main())
