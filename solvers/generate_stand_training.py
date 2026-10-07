"""ps:stand -- Connorses' "Stand": a crowd of little people who all walk on the
SAME key, and a set of lights that only counts as lit while somebody is
standing on it.

    late [Target1 Player] -> [Target2 Player]
    late [Target2 no Player] -> [Target1]
    ==============
    WINCONDITIONS
    ==============
    No Target1

Five levels, all five won, and every plan PROVED SHORTEST -- 9, 15, 11, 28 and
37 presses, read off an EXACT distance-to-win field rather than off "A* with a
weight of 1 and a heuristic that looked admissible".  Part of the
`solvers/common/ps_astar.py` family, on the native-model side of it: the
interpreter runs at ~3.8k presses/s here and the searches need millions, so the
mechanic is re-implemented as one line of bit twiddling and the interpreter's
job is to CERTIFY the answer (`--plans`) and to referee the model (`--verify`).

THE MECHANIC, and the two things it is NOT (a sokoban, and a game with a wait)

* **ONE KEY MOVES EVERYBODY.**  There is no "select a character" press: the
  turn's force is assigned to every Player sprite on the board, so an arrow key
  moves all of them one cell in that direction at once.  The whole puzzle is
  therefore *breaking formation*: two people who start three cells apart stay
  three cells apart forever unless a wall stops one of them and not the other.
  That is why the plans are full of presses INTO walls -- `left left left
  right` in level 4's plan is not slack, it is three presses spent grinding one
  person into a corner while the others still had room, and one press taking
  the group back.

* **A CHAIN OF PEOPLE MOVES AS ONE, AND A BLOCKED CHAIN MOVES NOT AT ALL.**
  Players share their collision layer with the walls and with each other, so a
  person can be blocked by a person.  The adapter resolves a press by tracing
  each pushed chain to its end and moving it only if the cell past the end is
  free (`PSEngine._resolve_forces`), and since every force in this game points
  the SAME way there is exactly one chain per queue of people: the front-most
  decides for all of them.  `StandBoard.step` is that rule, evaluated
  front-first so one pass settles it.  Nothing else can happen -- two chains
  can never contest a cell when every force is parallel, which is why the model
  has no conflict handling and why the presentation flips are safe (see below).

* **IT IS NOT A SOKOBAN.**  Crate and BigCrate exist in the OBJECTS section
  (and BigCrate has the nice inverted rule `[ > Player | BigCrate] -> [ Player
  | > BigCrate ]` -- you shove it without moving yourself), but no level in the
  file places one.  The five boards are people, walls and lights, and nothing
  else.

* **ACTION DOES NOTHING.**  No rule reads it, so pressing X assigns no force
  and changes no cell.  It is left out of `directions` rather than being
  searched and discarded: a press that cannot change the board can never be on
  a shortest path, and branching on it would have cost a fifth of every search
  for nothing.

THE WIN IS SIMULTANEOUS, WHICH IS WHY THE LATE RULES ARE THERE

`No Target1` reads as "every light is lit", and a light is lit only while a
person is standing on it -- step off and `late [Target2 no Player] ->
[Target1]` puts it straight back.  So the win is not "visit all the lights", it
is "have one body on each light AT THE SAME INSTANT", with as many bodies as
lights on every level (1, 2, 2, 6 and 3).  In model terms the board state is
nothing but the SET of occupied cells and the win is `targets subset of
players` -- the lights themselves carry no state, which is what makes a
121-cell board fit in a 121-bit integer.

The message between levels 3 and 4 says it out loud ("you'll have to stand on
all of them at once. Yup."), and it is also the reason a lit light has to be
VISIBLE; see WHAT WAS INVISIBLE.

THE HEURISTIC: count the presses each DIRECTION owes, not the distance

The obvious lower bound -- the largest walk-distance in the best
person-to-light matching -- is far too weak here, because the constraint is not
that each person walks far, it is that the group has to be pumped back and
forth to break formation.  The bound this uses instead comes from the one-key
rule:

  every press is ONE direction, and a press of `down` advances each person by
  at most one `down`-step, so a plan with `n_down` down-presses gives EVERY
  person at most `n_down` down-steps.  Hence, writing `req_d(p, t)` for the
  fewest `d`-steps on any walk from cell `p` to cell `t`,

      n_d  >=  max over people of req_d(person, its light)     for each d,
      L    =   n_up + n_down + n_left + n_right
           >=  SUM over the four directions of those four maxima.

`h` is that sum, minimised over the perfect matchings of people to lights (n!
of them, and n is at most 6).  `req_d` is a 0-1 BFS out of each light -- a step
in direction `d` costs 1 and the other three cost 0 -- so it is wall-aware: a
person who has to go around a block pays for the detour's `down`s even where
the row difference is zero.

It is ADMISSIBLE by the argument above and CONSISTENT because one press can
only retire one press's worth of one direction's debt: for a fixed matching,
moving a person one cell in direction `e` lowers `req_e` by at most 1 and
lowers no other `req_d` at all (prepending an `e`-step to a walk adds no
`d`-steps), so `h(s) <= 1 + h(s')` across every edge.  `--verify` MEASURES both
claims over every state of every field rather than resting on this paragraph: 0
inadmissible states and 0 inconsistent edges over 5962 of them.

How much it buys: the whole of this game -- search and sweep together --
touches 34 / 70 / 125 / 8129 / 3317 distinct boards, of which 19 / 48 / 75 /
3015 / 2805 are the field (every state a shortest plan can pass through).  A
plain BFS on level 3 passes 300000 states by depth 15 and the answer is at 28;
its space grows about 1.7x per press, so the unguided version of this search is
not a slower program, it is an impossible one.

THE FIELD: A* for the bound, then one sweep for everything else

`StandBoard.astar` proves `d*` (weight 1, the admissible heuristic above).
`StandBoard.field` then sweeps forward from the level start keeping only
states with `g + h <= d*` -- which, with a consistent `h`, is a superset of
every state on every shortest path -- and runs a backward BFS from the winning
edges over exactly those edges.  That yields the exact distance-to-win for
every state a shortest plan can pass through, and from it, in one pass and with
no re-solving:

* a plan that is provably shortest (its length is checked against `d*`);
* the EXACT optimal-action set at every step -- every press that still finishes
  in the same number of moves.  14 of this game's 100 presses have a second
  right answer (1.14 optimal presses per step), which is what a corpus should
  say about a board where two people are walking down parallel corridors and
  the order of the two axes does not matter yet.

WHAT WAS INVISIBLE

`Target2` -- the object a light becomes while somebody is standing on it --
shipped as `Darkgreen lightgreen`, and the ARC palette has exactly one green,
so both of them and the Background's own `GREEN` are index 14.  The WINNING
composition of this game, a person standing on a lit light, was therefore
pixel-identical to a person standing on grass: an agent could see where the
unlit lights were and could never see one light up, on the one game whose win
condition is "all of them at once".  `data/puzzlescript_games/Stand.txt` now
paints Target2 `Yellow` (11), keeping the author's sprite geometry -- rows 1-3
of the 5x5, exactly where the Player sprite is transparent -- so the lit tile
shows through the body standing on it.  Nothing else in the file is touched: no
rule, no level, no legend character, no other sprite.  `--audit` is the
regression test, and it FAILED before the change (`target2` == `floor`, `player
on target2` == `player`) at both cell sizes the levels use.

VERIFIED.  5/5 levels, 100 presses, every plan its level's proven `d*` and
every one of them replayed on the real interpreter to a WIN (`--plans`).
`--verify`: the model agrees with the interpreter on 24000 presses across
random play and CROWDED boards (up to 12 extra people, which is what actually
exercises the chain rule -- the shipped levels rarely queue three people in a
line), 0 inadmissible states and 0 inconsistent edges over the five fields
(5962 states). `--bfs`: a heuristic-free breadth-first press search of levels
0-2 (36, 100 and 314 states within the answers' depth) finds the same 9, 15 and
11.  `--audit`: 8 compositions, pairwise distinct at both cell sizes (9 px and
5 px) and under all eight rotations and mirrors of each -- and FAILING before
the Target2 repaint. `--symmetry`: 5 levels x 16 presentations, every plan a
WIN and every frame of both the plan and a 200-press random walk exactly the
transform of the unaugmented one.  6 seeds x 5 levels recorded in 7.7 s cold
(2.2 s warm, the searches being the whole of the difference): every level a
WIN, all 671 recorded steps replay frame-exact through the adapter from the
recorded SCREEN actions, 606/606 expert steps labelled (1.14 optimal presses
per step, the taken press always inside its own set), indices in 0..6, and two
processes at different hash seeds emit byte-identical episodes -- and, from a
cold cache, a byte-identical `data/stand_plans.json`.  The longest plan is 37
presses against the adapter's 200-press per-level cap.

RECOVERY is the family's RESET prefix (`recovery_mode = "reset"`).  It has to
be: formation-breaking is IRREVERSIBLE -- once a wall has eaten one person's
step the group's shape has changed and no press sequence can restore it in
general -- so a perturbed board cannot be re-planned from, and one RESET back
to the level start lands on exactly the state the cached field was built from.

CLI: ``--verify`` (the model against the interpreter, and the heuristic's two
properties against the exact fields), ``--audit`` (every cell composition the
game can draw, pixel-distinct at every cell size the levels use and under the
whole 8-element symmetry group), ``--symmetry`` (every presentation is an exact
transform of the unaugmented frames and every plan still wins through the
adapter), ``--bfs`` (an independent exhaustive proof of the levels whose whole
space fits), ``--plans`` (every level's plan, replayed through the
interpreter), otherwise the BaseSolver CLI.
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

GAME_ID = "ps:stand"
GAME_NAME = "Stand"

#: The presses the search branches on.  ACTION is deliberately absent: no rule
#: in this game reads it, so it assigns no force and changes no cell, and a
#: press that cannot change the board can never be on a shortest path.
DIRS = ("up", "down", "left", "right")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Everything on these boards.  Wall1 and Wall2 are two paint jobs for one
#: behaviour (neither appears in any rule; both merely occupy the players'
#: collision layer), and Target1/Target2 are the unlit and lit faces of one
#: light.  Crate and BigCrate are declared by the game but placed by no level.
_OBJECTS = ("wall1", "wall2", "target1", "target2", "player")

_INF = float("inf")

#: Sentinel successor for "this press wins".  A `str` can never collide with a
#: board, which is an `int`.
_WIN = "win"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class StandBoard:
    """One level, and the whole mechanic, as bit twiddling over cell indices.

    A state is an ``int``: bit ``r * W + c`` is set when somebody is standing
    on that cell.  Nothing else varies -- the walls never move, and a light is
    a pure function of whether a body is on it -- so this is an exact,
    canonical encoding of the board, and a 121-cell level costs 121 bits per
    state instead of the ~64 KB a `solvers.common.ps_astar.snapshot` of the
    same grid needs. That is the difference between a search that fits in RAM
    and one that does not: level 3's answer is 28 presses deep and the
    reachable space grows about 1.7x per press.
    """

    def __init__(self, eng, ids: dict):
        grid = eng.grid
        self.H, self.W = len(grid), len(grid[0])
        wall_ids = {ids["wall1"], ids["wall2"]}
        light_ids = {ids["target1"], ids["target2"]}
        player_id = ids["player"]
        walls, lights, people = set(), [], []
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell & wall_ids:
                    walls.add((r, c))
                if cell & light_ids:
                    lights.append(r * self.W + c)
                if player_id in cell:
                    people.append(r * self.W + c)
        # `nxt[d][i]` is the cell a body on `i` walks to when `d` is pressed,
        # or -1 when the board edge or a wall stops it.  Precomputing it is
        # what makes `step` four array lookups per person instead of a bounds
        # check and a set lookup.
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
        self.start = sum(1 << i for i in people)
        self._h_cache: dict[int, int] = {}
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

    def step(self, mask: int, d: str) -> int:
        """The board after pressing ``d``.  Everybody moves one cell that way
        unless the chain they are in is blocked.

        Evaluated FRONT-FIRST, which is what makes one pass exact: a person is
        stopped only by the board edge, by a wall, or by a person who is
        themselves stopped, and the person in front of them has already been
        decided.  Row-major index order is front-first for `up` and `left` (and
        its reverse is, for `down` and `right`) within every row and column
        that matters -- two people in different rows can never block each other
        on a horizontal press, so the cross-row order of the scan is
        irrelevant.
        """
        nxt = self.nxt[d]
        cells = self._cells(mask)
        if d == "down" or d == "right":
            cells.reverse()
        blocked = 0
        for cell in cells:
            ahead = nxt[cell]
            if ahead < 0 or (blocked >> ahead) & 1:
                blocked |= 1 << cell
        new = 0
        for cell in cells:
            new |= 1 << (cell if (blocked >> cell) & 1 else nxt[cell])
        return new

    def won(self, mask: int) -> bool:
        """Every light lit AT ONCE -- the engine's `No Target1`."""
        return mask & self.goal == self.goal

    # -- the heuristic -------------------------------------------------------
    def _dir_fields(self) -> dict:
        """``{(light, d): [fewest d-steps on any walk from cell to light]}``.

        A 0-1 BFS backwards out of each light in which a step in direction
        ``d`` costs 1 and the other three cost 0.  Unlike a plain distance this
        is the DEBT the plan owes to one direction, which is the quantity the
        one-key rule bounds (see the module docstring).
        """
        n = self.H * self.W
        fields = {}
        for d in DIRS:
            # Reverse adjacency: `pred[v]` holds every (u, cost) with
            # u --e--> v.
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

    def heuristic(self, mask: int) -> float:
        """Presses still owed, as a lower bound: the smallest over
        person-to-light matchings of the sum over the four directions of the
        largest debt any matched person owes that direction.

        `_INF` when some light is unreachable for every free person -- the
        state can never win and the search drops it.
        """
        got = self._h_cache.get(mask)
        if got is not None:
            return got
        people = self._cells(mask)
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
        self._h_cache[mask] = best
        return best

    # -- search --------------------------------------------------------------
    def astar(self, node_cap: int = 2_000_000) -> "list[str] | None":
        """A shortest press sequence, or None if there is none within the cap.

        Weight 1 over an admissible, consistent heuristic, so the first win
        popped is optimal -- and unlike a macro search every edge here costs
        exactly one press, so generating a win IS reaching it.
        """
        start = self.start
        if self.won(start):
            return []
        queue = [(self.heuristic(start), 0, 0, start, ())]
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
                heapq.heappush(queue, (ng + hv, ng, counter, nxt,
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
        succ: dict[int, dict[str, object]] = {}
        queue = deque([start])
        while queue:
            state = queue.popleft()
            g = gval[state]
            edges: dict[str, object] = {}
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

    def plan(self, node_cap: int = 2_000_000) -> "Plan | None":
        """The shortest plan and its exact per-step optimal SETS.

        Ties are broken by `DIRS` order, which is what makes a re-derived plan
        byte-identical across processes (and therefore what lets the disk cache
        be trusted between `parallelize_generator` shards)."""
        found = self.astar(node_cap)
        if found is None:
            return None
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

class StandExpert(PSExpert):
    """`PSExpert`'s plan memo, snapshot discipline and disk cache around the
    native search.  `heuristic` is never called on the interpreter -- the model
    owns it -- so it asserts rather than returning a number nothing would use.
    """

    directions = list(DIRS)
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "stand_plans.json")

    def setup(self) -> None:
        self.ids = {n: self.g.obj_name_to_idx[n] for n in _OBJECTS}
        self.stats: dict[int, dict] = {}      # level -> what the search cost
        self._last: dict | None = None

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "StandExpert plans on its native model; heuristic is unused")

    def board(self, eng) -> StandBoard:
        return StandBoard(eng, self.ids)

    def _search(self, eng) -> "Plan | None":
        board = self.board(eng)
        t0 = time.time()
        found = board.plan(node_cap=self.node_cap)
        self._last = {
            "people": len(StandBoard._cells(board.start)),
            "lights": len(board.lights),
            "h0": board.heuristic(board.start),
            "states": len(board._h_cache),
            "seconds": time.time() - t0,
        }
        return found

    def plan(self, eng, level=None):
        self._last = None
        got = super().plan(eng, level)
        if level is not None and self._last is not None:
            self.stats[level] = self._last
        return got

    def describe(self, level: int) -> str:
        got = self.stats.get(level)
        return "cached" if got is None else (
            f"{got['states']} states, h0 {got['h0']}, {got['seconds']:.1f}s")


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class StandSolver(PSAStarSolver):
    game_id = GAME_ID
    game_name = GAME_NAME
    game_module_id = GAME_ID
    expert_cls = StandExpert
    max_steps = 120                    # the longest plan is 37 presses


# ---------------------------------------------------------------------------
# --verify: the model against the interpreter, and the heuristic's two claims
# ---------------------------------------------------------------------------

def _engine_people(eng, player_id: int, width: int) -> int:
    return sum(1 << (r * width + c)
               for r, row in enumerate(eng.grid)
               for c, cell in enumerate(row) if player_id in cell)


def verify(trials: int = 60, presses: int = 80, verbose: bool = True) -> int:
    """Three measurements, none of which the docstring is allowed to assert.

    1. **The model IS the game.**  Random play on every level, with the model
       and the interpreter stepped side by side and the whole occupied-cell set
       compared after every press, plus the win predicate.  Half the trials run
       on CROWDED boards -- extra Player sprites scattered over the free cells
       before play starts.  That mode is not decoration: the chain rule (a
       queue of people moves iff the front one can) is the only part of `step`
       that random play on the shipped levels barely exercises, because those
       levels place their people far apart.  It is the same lesson
       ps:sokobaiogenesis learned the hard way -- a clean fuzz can be blind to
       exactly the configuration a shortest plan manufactures on purpose.

    2. **The heuristic is ADMISSIBLE**: `h(s) <= dist(s)` at every state of
       every level's exact field.

    3. **The heuristic is CONSISTENT**: `h(s) <= 1 + h(s')` across every
       edge of every field.  This is the property the field sweep rests on (an
       inconsistent `h` could prune a state that is on a shortest path), so it
       is measured rather than argued.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    ids = {n: game._game.obj_name_to_idx[n] for n in _OBJECTS}
    player_id = ids["player"]
    rng = random.Random(f"{GAME_NAME}:verify")
    bad = checked = 0

    for level in range(game.n_levels):
        game.set_level(level)
        board = StandBoard(eng, ids)
        for trial in range(trials):
            game.set_level(level)
            crowded = trial >= trials // 2
            if crowded:
                free = [(r, c) for r in range(board.H) for c in range(board.W)
                        if (r, c) not in board.walls]
                for r, c in rng.sample(free, min(len(free),
                                                 rng.randint(3, 12))):
                    eng.grid[r][c].add(player_id)
                eng._position_index_dirty = True
                eng._rule_noop_cache.clear()
            state = _engine_people(eng, player_id, board.W)
            for _ in range(presses):
                d = rng.choice(DIRS)
                eng.step(d)
                got = _engine_people(eng, player_id, board.W)
                want = board.step(state, d)
                checked += 1
                if got != want or eng.check_win() != board.won(got):
                    bad += 1
                    if bad < 4:
                        where = "crowded" if crowded else "plain"
                        print(f"  MISMATCH L{level} {where} press {d}: "
                              f"model {want:x} engine {got:x}")
                state = got
    if verbose:
        print(f"  model vs interpreter: {checked} presses, {bad} mismatches")

    inadmissible = inconsistent = states = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = StandBoard(eng, ids)
        plan = board.plan()
        if plan is None:                        # `--plans` reports the level
            continue
        succ, dist = board.field(len(plan))
        states += len(succ)
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
        print(f"  heuristic over {states} field states: {inadmissible} "
              f"inadmissible, {inconsistent} inconsistent edges")
    return bad + inadmissible + inconsistent


# ---------------------------------------------------------------------------
# --audit: every cell composition the game can draw
# ---------------------------------------------------------------------------

#: Every stack a square of this game can hold.  Player, Wall1 and Wall2 share
#: one collision layer, so no two of them ever meet; the lights are on their
#: own layer underneath.  `player on target1` is transient by construction (the
#: late
#: rule lights the tile at the end of the very turn a body arrives) but it is
#: audited anyway -- it is what a level whose start places somebody on a light
#: would draw before the first press.
_AUDIT_CASES = {
    "floor": (),
    "wall1": ("wall1",),
    "wall2": ("wall2",),
    "target1": ("target1",),
    "target2": ("target2",),
    "player": ("player",),
    "player on target1": ("player", "target1"),
    "player on target2": ("player", "target2"),
}


def audit(verbose: bool = True) -> int:
    """Assert every cell COMPOSITION renders differently at every cell size the
    levels use -- and that no rotation or mirror of one is another.

    Composition, not object: the bug this is the regression test for was in the
    STACK.  Target2 was the Background's own green, so `player on target2` --
    the winning square of the game -- and `player` were one picture.

    The board is FILLED with the composition and whole frames are compared,
    rather than one cell being sliced out by ``cell_px`` arithmetic:
    `_render_frame` upscales a sub-64 render to fill the frame and then
    letterboxes it, so the cell grid in the output is not ``cell_px``-aligned
    and the crop lands in the wrong window.

    The eight-way pass is what the presentation augmentation needs: this game
    takes rotations (every ps: game does) and, per `_FLIP_GAMES`, mirrors too,
    so a transform of one composition that equalled another composition would
    make the corpus say two different things about the same picture."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, parsed = game._engine, game._game
    idx = parsed.obj_name_to_idx
    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for name, objs in _AUDIT_CASES.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, parsed))
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        turned = [(a, b, k, hf)
                  for a, b in itertools.permutations(shots, 2)
                  for k in range(4) for hf in (False, True)
                  if np.array_equal(_transform(shots[a], k, hf, False),
                                    shots[b])]
        bad += len(clashes) + len(turned)
        if verbose:
            verdict = ("all distinct" if not clashes
                       else "IDENTICAL " + str(clashes))
            turn_note = ("" if not turned
                         else ", TRANSFORM CLASH " + str(turned))
            print(f"  {h}x{w} (cell {64 // max(h, w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): "
                  f"{len(_AUDIT_CASES)} compositions, "
                  f"{verdict}{turn_note}")
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
    encodes a facing.  What is different here is that a press moves up to SIX
    bodies at once, which is the shape that hid Gobble Rush's rule-order
    chirality, so that half is measured rather than argued.  The reason to
    expect it clean is structural too: every force a press assigns points the
    SAME way, so two chains can never claim one cell and the order
    `_resolve_forces` traced them in cannot decide anything.

    Both the PLANS and a seeded random walk are replayed -- the walk is what
    reaches the boards a plan never visits (whole crowds jammed into a corner,
    presses into walls, the unbound ACTION key), and a plan is what puts six
    bodies on six lights at once."""
    solver = StandSolver()
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
            for tag, presses in (("plan", [screen_action(d, *k)
                                           for d in plans[level]]),
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

def bfs_report(levels=(0, 1, 2), verbose: bool = True) -> int:
    """Exhaustive breadth-first press search to the depth of the first win, and
    the shortest win it finds compared against the planned one.

    This is the check that the heuristic is not quietly losing a shorter
    answer: the BFS has no notion of a matching, a debt or a bound, it just
    presses all four keys breadth-first and stops expanding once it is as deep
    as the best win so far -- which is exhaustive for the question being asked,
    since nothing beyond that depth can be shorter.  Only levels 0-2 are in
    reach (36, 100 and 314 states inside their answers' depth); levels 3 and 4
    grow about 1.7x per press and pass 300000 states by depth 15 against
    answers at 28 and 37, which is exactly why they get a heuristic search and
    an admissibility
    measurement instead."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    ids = {n: game._game.obj_name_to_idx[n] for n in _OBJECTS}
    bad = 0
    for level in levels:
        game.set_level(level)
        board = StandBoard(eng, ids)
        plan = board.plan()
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
    solver = StandSolver()
    game, expert, solvable = solver._ensure(0)
    eng = game._engine
    total = ties = 0
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
        total += len(plan)
        ties += step_ties
        print(f"  L{level}: {eng.height}x{eng.width}, "
              f"{len(StandBoard._cells(board.start))} people / "
              f"{len(board.lights)} lights, "
              f"h0 {board.heuristic(board.start)}, "
              f"{len(plan):3d} presses  win={eng.check_win()}  "
              f"{step_ties:2d} tie-presses  [{expert.describe(level)}]")
    print(f"  solvable levels: {solvable}")
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
        violations = bfs_report(tuple(args) if args else (0, 1, 2))
        print(f"bfs: {violations} mismatches")
        sys.exit(1 if violations else 0)
    if "--plans" in sys.argv:
        plan_report()
        sys.exit(0)
    sys.exit(StandSolver.main())
