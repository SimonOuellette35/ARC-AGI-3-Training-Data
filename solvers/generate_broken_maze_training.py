"""Generate Phase-1 training data for the PuzzleScript game ps:broken_maze
("Broken Maze" by David Ding).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, the native model of the turn, the exact search over it, and the
optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_broken_maze",
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
A wall cross splits every board into four sealed quadrants, each holding one
walker: ``NW`` (orange, the ``Player``) plus ``NE``, ``SE`` and ``SW`` (all dark
grey). One arrow key drives all four at once, each in its own MIRROR of the
pressed direction:

    RIGID HORIZONTAL [> NW][NE][SE][SW] -> [> NW][< NE][< SE][> SW]
    RIGID VERTICAL   [> NW][NE][SE][SW] -> [> NW][> NE][< SE][< SW]

so NE is NW reflected in the vertical axis, SW is NW reflected in the horizontal
one, and SE is NW turned through 180 degrees -- exactly what the compass names
say. The win is ``all players on Target``, and every level puts one target in
each quadrant. Three consequences drive the whole solver:

  * **The quadrants are four DIFFERENT copies of one maze.** That is the
    "broken" in the title: the walls are near-mirror-images of each other with a
    few cells changed, so a route that works in one quadrant walks its twin into
    a wall. The puzzle is to find one key sequence that is legal in all four.
  * **Movement is strictly all-or-nothing** (``RIGID`` cancels the whole group,
    and the third rule cancels the Player whenever an Other is stationary). A
    single blocked walker freezes all four -- there is no way to "bump a wall to
    break lockstep", which is the standard trick in this genre and is exactly the
    tool this game withholds. Verified by fuzzing, not read off the rules: 37k
    random presses across all 8 levels never once moved a strict subset.
    Therefore **the four positions are a pure function of NW's position**:
    ``p_i = p_i(0) + M_i * (p_nw - p_nw(0))`` with ``M_i`` the sign pair above,
    an invariant that holds for every reachable state and is what makes the
    search small.
  * **Crates are the one thing that does NOT lock step.** ``[> Players | Crate]
    -> [> Players | > Crate]`` gives the crate its own force, and that force is
    NOT part of the rigid group -- so when some other quadrant blocks the turn,
    the walkers all stay put and the pushed crates still move. A press that
    "does nothing" therefore routinely rearranges up to four crates for free, and
    the crate levels are built on it. A crate is stopped by a wall or by another
    crate (crates do not push each other -- nothing but a Player appears on the
    left of that rule), and a stopped crate blocks its pusher, which freezes
    everybody.

There is no ACTION key (the game is declared ``noaction``), so the whole action
space is the four arrows.

Model + search
--------------
`_step` re-implements one turn natively; it is ~5 us against the interpreter's
~1.3 ms, and `--selfcheck` drives thousands of random presses through BOTH the
engine and the model comparing every walker and every crate after every press.
That is the guard that lets the search trust it.

The state is ``(4 walker cells, frozenset of crate cells)`` -- the walkers are
redundant given the invariant above, but they are four cheap ints and keeping
them makes the transition read like the rules.

The heuristic is the exact distance in the **combined maze**: project every cell
into NW's frame, keep the cells whose four images are all floor, and run one
backward BFS from the cells whose four images are all targets. That is the game
with the crates deleted, so it is an admissible AND consistent lower bound (a
crate can only ever block), and it is a *perfect* one on the four crate-free
levels -- A* walks straight down it without expanding a wrong node. Cells the
field cannot reach are dead outright, which prunes the corners of the crate
levels. With the goal test on POP, plans are provably shortest.

Cost is nil: the deepest level needs ~3.4k model steps, so every level is
searched from scratch at startup (no stored plan table) and the whole run is
dominated by rendering frames.

Optimal-action targets and recovery
-----------------------------------
Because the model is exact and the search is instant, `optimal_for` emits the
TRUE optimal set rather than a reordering heuristic: it reads the LIVE engine
state, re-solves each of the four successors and keeps every press whose
remaining optimal distance is one less. That is worth doing here -- these boards
are full of order-free stretches (any interleaving of "three left, two up"
reaches the same cell in the same number of presses), and on the crate levels it
also credits the genuinely different routes a crate rearrangement opens up.
Training one arbitrary interleaving as the only right answer is a lie the policy
has to unlearn.

Reading the live state (rather than replaying the plan from the level start, as
the irreversible games in this family must) is also what lets the generator take
real detours: `epsilon` is non-zero here, so roughly one press in eight is a
random legal alternative and the expert then re-plans from wherever it landed.
Those steps record the mistake as the action taken and the recovery as
``optimal``, which is the signal a policy needs to get back on route after its
own error -- on top of the explore-then-RESET prefix in front of every level.

Augmentation
------------
Per (seed, level) the adapter draws a frame rotation (0..3) and, since
``Broken_Maze`` is in `PuzzleScriptAdapter._FLIP_GAMES`, an independent
horizontal and vertical flip -- 16 presentations of each of the 8 levels. The
flips are an exact symmetry of this game rather than a plausible one: reflecting
the board relabels the walkers (hflip swaps NW<->NE and SW<->SE, vflip swaps
NW<->SW and NE<->SE) and, once the reflected axis is negated, both rigid rules
map onto themselves under that relabelling. Every sprite is mirror-symmetric
(the walkers are squares with the four corners cut, the crate is a ring), so a
flipped frame is a frame the game could really have shipped, with a different
quadrant playing the orange reference part -- which the rotation augmentation
already does anyway.

Usage (run from the repo root):
    python solvers/generate_broken_maze_training.py --episodes 200 \
        --out data/training_multi_level/broken_maze
    python solvers/generate_broken_maze_training.py --selfcheck
"""

from __future__ import annotations

import heapq
import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import PuzzleScriptAdapter   # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert     # noqa: E402

GAME_NAME = "Broken_Maze"

#: The four walkers, in the order `_Geometry.start` / the state tuple use.
NAMES = ("nw", "ne", "se", "sw")

#: Engine direction -> (dr, dc) for NW. The other three walkers apply `_SIGN`.
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Per-walker sign pair: NE mirrors the column, SW the row, SE both. This IS the
#: pair of RIGID rules, rewritten as arithmetic.
_SIGN = {"nw": (1, 1), "ne": (1, -1), "se": (-1, -1), "sw": (-1, 1)}

#: The whole action space -- the game is declared ``noaction``.
_DIRS = ("up", "down", "left", "right")

#: Heuristic charge for a walker the combined maze cannot route to a target. The
#: search drops those states instead (see `_solve`); this is only what
#: `PSExpert.heuristic` reports for one.
_UNREACHABLE = 999


class _Geometry:
    """Everything about a level that never changes, in NW's coordinate frame.

    ``free`` / ``goals`` / ``dist`` are the COMBINED maze: a cell counts only if
    all four walkers' images of it are legal, which is the whole puzzle stated as
    one grid. Built once per level and shared by the search, the heuristic and
    the optimal-set labeller.
    """

    __slots__ = ("walls", "targets", "free", "goals", "dist")

    def __init__(self, walls, targets, free, goals, dist):
        self.walls = walls          # set[(r, c)] -- wall cells, all quadrants
        self.targets = targets      # set[(r, c)]
        self.free = free            # set[(r, c)] -- NW cells legal for all four
        self.goals = goals          # set[(r, c)] -- NW cells that win
        self.dist = dist            # {(r, c): presses to a goal, crates ignored}


def _step(geom, pos: tuple, crates: frozenset, d: str) -> tuple:
    """One turn of the real game, natively. Returns the new ``(pos, crates)``.

    Order of business matches the interpreter (see the module docstring): every
    walker's push is resolved against the state BEFORE the turn, the pushes that
    are legal happen regardless of whether anyone walks, and the walkers move
    only if not one of the four was blocked.
    """
    walls = geom.walls
    dr0, dc0 = _DELTA[d]
    blocked = False
    pushes = []
    for i, name in enumerate(NAMES):
        sr, sc = _SIGN[name]
        dr, dc = sr * dr0, sc * dc0
        front = (pos[i][0] + dr, pos[i][1] + dc)
        if front in walls:
            blocked = True
        elif front in crates:
            dest = (front[0] + dr, front[1] + dc)
            # A crate is stopped by a wall, by another crate (crates do not push
            # each other) or by a walker; a stopped crate stops its pusher, and a
            # stopped pusher freezes all four.
            if dest in walls or dest in crates or dest in pos:
                blocked = True
            else:
                pushes.append((front, dest))
    if pushes:
        crates = (crates - {a for a, _ in pushes}) | {b for _, b in pushes}
    if blocked:
        return pos, crates
    return tuple((pos[i][0] + _SIGN[n][0] * dr0, pos[i][1] + _SIGN[n][1] * dc0)
                 for i, n in enumerate(NAMES)), crates


def _solve(geom, pos: tuple, crates: frozenset, node_cap: int) -> list | None:
    """Shortest press sequence from ``(pos, crates)`` to a win, or None.

    A* over `_step` with `_Geometry.dist` as the heuristic. That field is the
    crate-free distance, so it is consistent (one press changes it by at most
    one) and admissible (deleting crates can only shorten a route); with the goal
    test on POP rather than on generation, the plan returned is provably
    shortest, which is what `BrokenMazeSolver.optimal_for` relies on.

    A state whose walker has NO entry in the field is dead -- it cannot reach a
    target even with every crate deleted -- so it is dropped rather than queued.
    """
    if pos[0] in geom.goals:
        return []
    h0 = geom.dist.get(pos[0])
    if h0 is None:
        return None
    counter = 0
    pq = [(h0, 0, counter, pos, crates, ())]
    best = {(pos, crates): 0}
    nodes = 0
    while pq:
        _f, g, _c, cpos, ccrates, path = heapq.heappop(pq)
        if cpos[0] in geom.goals:
            return list(path)
        if best[(cpos, ccrates)] < g:
            continue                       # stale duplicate
        for d in _DIRS:
            npos, ncrates = _step(geom, cpos, ccrates, d)
            nodes += 1
            if npos == cpos and ncrates == ccrates:
                continue                   # the turn changed nothing at all
            h = geom.dist.get(npos[0])
            if h is None:
                continue                   # unreachable even without crates
            key = (npos, ncrates)
            ng = g + 1
            if best.get(key, 1 << 30) <= ng:
                continue
            best[key] = ng
            counter += 1
            heapq.heappush(pq, (ng + h, ng, counter, npos, ncrates, path + (d,)))
        if nodes >= node_cap:
            return None
    return None


class BrokenMazeExpert(PSExpert):
    """Exact planner over the native model (see the module docstring).

    `PSExpert` supplies the plan memo, the restore discipline and the level
    scoping; `_search` swaps in the native A*, so the interpreter is only ever
    stepped by the recorder -- which is also what verifies every plan, since a
    level is kept only when the engine reports WIN.
    """

    directions = list(_DIRS)

    #: `_key` is walkers + crates, canonical only WITHIN a level (the walls and
    #: targets that complete the state are static per level but differ between
    #: levels).
    scope_by_level = True

    def setup(self) -> None:
        g = self.g
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.crate_ids = set(g.resolve_object_name("crate"))
        self.target_ids = set(g.resolve_object_name("target"))
        self.walker_ids = {n: set(g.resolve_object_name(n)) for n in NAMES}
        self._geom: dict[int, _Geometry] = {}
        self._solved: dict[tuple, list | None] = {}
        self._cur: tuple[int | None, _Geometry] | None = None

    # -- reading the engine ---------------------------------------------------
    def read(self, eng) -> tuple[tuple, frozenset]:
        """``(walker cells in NAMES order, crate cells)`` from the engine grid."""
        pos: list = [None] * 4
        crates = set()
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                if cell & self.crate_ids:
                    crates.add((r, c))
                for i, n in enumerate(NAMES):
                    if cell & self.walker_ids[n]:
                        pos[i] = (r, c)
        return tuple(pos), frozenset(crates)

    def geometry(self, eng, level: int | None) -> _Geometry:
        """The level's `_Geometry`, built once and cached.

        The projection is state-independent: ``p_i - M_i * p_nw`` is invariant
        over every reachable state (see the module docstring), so whichever
        snapshot happens to build it defines the same combined maze.
        """
        geom = self._geom.get(level)
        if geom is not None:
            return geom
        pos, crates = self.read(eng)
        h, w = len(eng.grid), len(eng.grid[0])
        walls, targets = set(), set()
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & self.wall_ids:
                    walls.add((r, c))
                if cell & self.target_ids:
                    targets.add((r, c))

        def images(cell):
            """The four walkers' positions when NW stands on ``cell``."""
            return [(pos[i][0] + _SIGN[n][0] * (cell[0] - pos[0][0]),
                     pos[i][1] + _SIGN[n][1] * (cell[1] - pos[0][1]))
                    for i, n in enumerate(NAMES)]

        free, goals = set(), set()
        for r in range(h):
            for c in range(w):
                img = images((r, c))
                if any(not (0 <= q[0] < h and 0 <= q[1] < w) or q in walls
                       for q in img):
                    continue
                free.add((r, c))
                if all(q in targets for q in img):
                    goals.add((r, c))

        # Backward BFS from every winning cell over the combined maze. Goal cells
        # the walkers can never reach are harmless seeds: they only ever lower an
        # estimate, so the field stays admissible.
        dist = {p: 0 for p in goals}
        queue = deque(goals)
        while queue:
            p = queue.popleft()
            for dr, dc in _DELTA.values():
                nxt = (p[0] + dr, p[1] + dc)
                if nxt in free and nxt not in dist:
                    dist[nxt] = dist[p] + 1
                    queue.append(nxt)

        geom = _Geometry(walls, targets, free, goals, dist)
        self._geom[level] = geom
        return geom

    # -- planning -------------------------------------------------------------
    def _key(self, eng):
        # Walkers + crates. Cheaper than the base's whole-grid key and exact:
        # nothing else on these boards ever changes.
        return self.read(eng)

    def plan(self, eng, level: int | None = None) -> list | None:
        self._cur = (level, self.geometry(eng, level))
        return super().plan(eng, level)

    def heuristic(self, eng) -> int:
        """Presses to a win with every crate deleted -- see `_Geometry`."""
        _level, geom = self._cur
        pos, _crates = self.read(eng)
        return geom.dist.get(pos[0], _UNREACHABLE)

    def _search(self, eng) -> list | None:
        level, geom = self._cur
        pos, crates = self.read(eng)
        return self.solve(level, geom, pos, crates)

    def solve(self, level, geom, pos, crates) -> list | None:
        """`_solve`, memoized per (level, state). The memo is what makes the
        optimal-set labeller cheap: it re-solves the same successors the plan
        walks through, one step apart."""
        key = (level, pos, crates)
        if key not in self._solved:
            self._solved[key] = _solve(geom, pos, crates, self.node_cap)
        return self._solved[key]

    def optimal_dirs(self, level, geom, pos, crates) -> list | None:
        """Every press that keeps the state on a SHORTEST route to the win.

        Exact, not heuristic: each successor is re-solved and kept when its
        optimal distance is one less than this state's."""
        base = self.solve(level, geom, pos, crates)
        if base is None:
            return None
        best: list[str] = []
        for d in _DIRS:
            npos, ncrates = _step(geom, pos, crates, d)
            if npos == pos and ncrates == crates:
                continue                   # a press with no effect is never optimal
            if npos[0] in geom.goals:
                if len(base) == 1:
                    best.append(d)
                continue
            sub = self.solve(level, geom, npos, ncrates)
            if sub is not None and len(sub) + 1 == len(base):
                best.append(d)
        return best


class BrokenMazeSolver(PSAStarSolver):
    game_id = "puzzlescript_broken_maze"
    game_name = GAME_NAME
    expert_cls = BrokenMazeExpert

    #: The native search is exact and instant (~3.4k model steps on the worst
    #: level), so there is no stored plan table and no weighting: every level is
    #: solved from scratch, shortest-first, at startup.
    node_cap = 2_000_000
    weight = 1
    #: The adapter GAME_OVERs at 200 actions per level; the longest plan here is
    #: 21 presses, leaving the whole budget to the exploration prefix and the
    #: detours.
    max_steps = 200

    #: Non-zero, unlike the rest of this family. The `PSAStarSolver` default is 0
    #: because those games are IRREVERSIBLE -- a detour can strand them, so their
    #: recovery data has to come from the explore-then-RESET prefix alone. Broken
    #: Maze is not: the planner is exact and instant from any state, and
    #: `record_level`'s detour guard already refuses any alternative that leaves
    #: the level unwinnable (which here is a crate shoved into a corner). So one
    #: press in eight is a random legal alternative, after which the expert
    #: re-plans from wherever it landed -- the taken action is the mistake and
    #: `optimal` is what should have been done, which is the recovery signal the
    #: policy actually needs. The RESET prefix still runs in front of it.
    epsilon = 0.12

    def prepare_expert(self, game, expert) -> None:
        """Build every level's `_Geometry` up front.

        Pure front-loading -- the projection does not depend on which state
        builds it -- but it means the per-level BFS is paid once, at startup,
        rather than inside the first search that needs it."""
        for level in range(game.n_levels):
            game.set_level(level)
            expert.geometry(game._engine, level)

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """Every press that keeps the game on a SHORTEST route to the win, read
        off the LIVE engine state.

        `record_level` asks for this before executing the press, so the engine is
        still at the state being labelled -- which is what makes the label right
        after an epsilon detour too, where the state is off the plan entirely and
        replaying the plan from the level start would label the wrong states.
        The set is exact (`BrokenMazeExpert.optimal_dirs` re-solves each
        successor), so a step whose plan is one of several equally short routes
        is labelled with all of them instead of with one arbitrary interleaving.

        Falls back to the press about to be taken if the oracle has nothing to
        say -- no expert step may ship unlabelled."""
        eng = expert.game._engine
        best = expert.optimal_dirs(level, expert.geometry(eng, level),
                                   *expert.read(eng))
        if best:
            return best
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Model self-check
# ---------------------------------------------------------------------------

def selfcheck(trials: int = 40, steps: int = 80, verbose: bool = True) -> int:
    """Drive random presses through BOTH the interpreter and `_step`, comparing
    every walker and every crate after every press. This is the guard that lets
    the search trust the native model; it also re-asserts the two invariants the
    whole design rests on -- that the walkers never move as a strict subset, and
    that their positions stay a pure function of NW's.

    Returns the number of mismatches."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    expert = BrokenMazeExpert(game)
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        geom = expert.geometry(game._engine, level)
        rng = random.Random(f"broken_maze:selfcheck:{level}")
        bad = 0
        for _ in range(trials):
            game.set_level(level)
            eng = game._engine
            pos, crates = expert.read(eng)
            for _ in range(steps):
                d = rng.choice(_DIRS)
                mpos, mcrates = _step(geom, pos, crates, d)
                eng.step(d)
                epos, ecrates = expert.read(eng)
                if (mpos, mcrates) != (epos, ecrates):
                    bad += 1
                    break
                moved = [mpos[i] != pos[i] for i in range(4)]
                if any(moved) and not all(moved):
                    bad += 1                     # lockstep broken
                    break
                pos, crates = mpos, mcrates
                if eng.check_win():
                    break
        total += bad
        if verbose:
            print(f"  L{level}: {'OK' if not bad else f'{bad} MISMATCHES'} "
                  f"({trials} rollouts x {steps} presses)")
    return total


def _plan_report() -> None:
    """Print the shortest plan for every level (engine-verified) -- the quick
    "is this game still fully solved" check."""
    solver = BrokenMazeSolver()
    game, expert, solvable = solver._ensure(0)
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(game._engine, level)
        if plan is None:
            print(f"  L{level}: UNSOLVED")
            continue
        eng = game._engine
        for d in plan:
            eng.step(d)
        print(f"  L{level}: {len(plan):3d} presses  win={eng.check_win()}  "
              f"{' '.join(d[0] for d in plan)}")
    print(f"  solvable levels: {solvable}")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        bad = selfcheck()
        print(f"selfcheck: {bad} mismatches")
        _plan_report()
        sys.exit(1 if bad else 0)
    sys.exit(BrokenMazeSolver.main())
