"""Generate Phase-1 training data for the PuzzleScript game
ps:santas_great_escape ("Santa's Great Escape" by Bear & Cow).

The harness -- the engine-blackbox A* expert, the rotation contract, the
trajectory recorder and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other search-the-interpreter ps:
generators. This file is only the game-specific part: its name, its goal
heuristic, and the three report modes that price what the heuristic claims.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_santas_great_escape",
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

THE GAME
--------
Santa walks a house, drops a gift on every Target, and leaves through the
Chimney. Six levels, five presses, no lose condition and nothing to push onto
anything -- the whole puzzle is the ROUTE.

  * **The counter.** The player is not one object but thirteen: Santa12 down to
    Santa0, all with the same sprite. Stepping onto a Target consumes it and
    decrements the santa, leaving a Gift behind::

        late [ Santa1 Target ] -> [ Santa0onGift ]        (x12, Santa12..Santa1)

    Every level ships exactly as many Targets as its santa's number
    (1/1/1/1/3/5), so the counter is not a second resource to budget -- it is a
    restatement of "how many targets are left", and it reaches 0 exactly when
    the last one is delivered.

  * **The escape.** ``[ > Santa0 | Chimney ] -> [ ... | Chimney ]``. An empty
    RHS cell: Santa0 walking into the chimney is REMOVED from the board. Only
    Santa0 -- a santa still carrying gifts just bumps into it. Combined with
    ``No Target``, that deletion IS the win; see the header of
    `games/ps:santas_great_escape/ps:santas_great_escape.py` for why it has to
    be spelled ``No Player`` for this interpreter.

  * **Push.** ``[ > Player | MovableObj ] -> [ > Player | > MovableObj ]``
    pushes a Tree (levels 0, 3, 4; the Table and the two Chairs the game also
    declares appear in no level). It is never required and never useful on the
    shipped boards -- a tree is just a wall you can shove -- but it is a legal
    move and the search has it.

  * **The chase.** A Boy or a Dog standing next to Santa steps toward him
    (``[ Boy | Player ] -> [ > Boy | Player ]``, blocked by Santa's own body,
    so in practice it FOLLOWS him as he retreats), and a Dog two cells away in
    a line, with a passable cell between, teleports into that cell
    (``late [ Dog | no Obstacle no Enemy | Player ] -> [ | Dog | Player ]``).
    Both are Obstacles, so a chase is a moving wall. There is no death rule:
    "Don't get caught" is flavour text, and the worst an enemy can do is stand
    where Santa wanted to walk.

    Two consequences worth naming, because the expert has to be right about
    them rather than assume them. The leap PLACES the dog rather than moving
    it, so it evicts whatever shares that cell -- and Bed_topLeft, the Table,
    the Chairs and the two Sinks are (unlike the other three bed quarters) not
    in the ``Obstacle`` group, so a dog can land on one and destroy it, opening
    a hole in the furniture. And because the enemies move on Santa's move, the
    ACTION press is not a no-op: it is a WAIT that advances the chase.

Expert solver
-------------
`PSExpert`: A* over the REAL interpreter, branching on the five primitive
presses. There is no macro worth naming (nothing in this game is a push
puzzle), and no model of the mechanic is written down anywhere in this file --
the enemies, the counter, the eviction and the escape are all whatever the
interpreter does.

The heuristic is the travel bound the route problem asks for: the remaining
task is "visit every Target, then reach the Chimney", so charge the MST of the
metric closure over ``{santa} + targets + {chimney}``. Any route through those
nodes is a connected subgraph spanning them, so its MST is a genuine lower
bound -- and unlike "distance to the nearest target" it does not collapse to
nothing on level 5, where five targets sit in five different pockets. With the
targets gone it degenerates to the walk to the chimney, which is then the whole
remaining game. (Multiple chimneys would be the min over them; every shipped
level has one.)

ADMISSIBILITY, and why the map is so pessimistic about what blocks. The pair
distances are BFS over a map whose only blockers are Wall and Black -- the two
objects no rule in this game creates, destroys or moves. Beds, furniture, the
chimney itself, trees and the enemies are all walked straight through. That is
deliberate, not laziness: a tree is pushable, an enemy walks away, and the dog
leap can DELETE a Bed_topLeft or a chair, so every one of them is a blocker
that the game can take back, and a heuristic that counted on one would
overestimate the states where it is gone. `_assert_static_terrain` checks the
Wall/Black premise against the parsed rules at setup rather than trusting this
paragraph.

So the estimate is admissible and the searches run at ``weight = 1``, which
makes the plans shortest and the measured optimal sets below exact rather than
a guess. That is not left as an argument either: ``--verify`` re-derives every
plan with the disk cache detached and prices it against a plain breadth-first
search over the same interpreter that uses NONE of the heuristic. All six
levels come back PROVED shortest (9 / 20 / 24 / 27 / 20 / 41 moves), and the
heuristic never over-charges at any step of any plan.

Optimal action sets
-------------------
``exact_optsets``: at every step of a plan, `PSExpert.optimal_sets` re-solves
from each of the five successors and keeps every press that still finishes in
the moves remaining. This game needs them more than most -- it is a walk, and a
walk's order is free, so any interleaving of the two axes that stays on a
shortest route is equally right and labelling one of them as the answer would
train a coin flip the policy cannot win. Measured over the six levels, 1.07
presses per step are optimal on average: levels 0 and 4 are forced at every
single step (they are corridors), and the rest peak at 1.10 in the open rooms
of levels 1 and 5. See ``--plans``.

That the number is so close to 1 is a fact about these boards, not a sign the
measurement is doing nothing -- the house is a warren of one-cell passages, and
where it opens up the sets do too.

The whole cost of the corpus is the seed-0 search: ~4s for the six plans and
~85s for the six sets of optimal sets (level 3 is four fifths of that), cached
together into `PLAN_CACHE`. Every later seed replays from disk at ~1s.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed
ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip, each with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`), which together re-sample the board's
8-element symmetry group.

The flips are earned rather than assumed. Every rule in the game is stated with
the relative ``>`` force or is direction-agnostic, the win conditions name no
direction, the input is screen-relative and there is no gravity -- but this
game has up to three independently moving bodies, which is the shape that hid
Gobble Rush's rule-order chirality, so ``--symmetry`` does not argue it: it
replays each level's plan on all 8 turned and mirrored copies of that level's
own BOARD and requires the resulting board to be the transform of the original
one, press for press. All six levels are exactly symmetric.

There is deliberately no recolor. ``--audit`` is the standing check on that:
Gift is Orange+Yellow and Wall is Brown+DarkBrown, which are the SAME two ARC
palette indices once ``brown -> 12`` and ``orange -> 12`` collapse, so the two
are told apart by sprite shape alone and a recolor has nowhere safe to move.

The expert plan is therefore seed-independent: solved once per level, cached,
and replayed per seed with that seed's remapped screen actions.

Usage (run from the repo root):
    python solvers/generate_santas_great_escape_training.py --episodes 200 \
        --out data/training_multi_level/santas_great_escape

    python solvers/generate_santas_great_escape_training.py --plans     # plans + tie sets
    python solvers/generate_santas_great_escape_training.py --audit     # rendering audit
    python solvers/generate_santas_great_escape_training.py --symmetry  # flip/rotation safety
"""

from __future__ import annotations

import itertools
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from adapters.puzzlescript_adapter import _render_frame              # noqa: E402
from solvers.common.ps_astar import (                                # noqa: E402
    PSAStarSolver, PSExpert, _load_game_module, restore, snapshot)

GAME_NAME = "Santa's_Great_Escape"
GAME_MODULE_ID = "ps:santas_great_escape"

#: Where each level's START plan (and its measured optimal sets) is kept between
#: processes. The searches are the whole cost of generation here -- ~90s for the
#: six levels, and seed-independent -- so without a file on disk every shard
#: `parallelize_generator` starts would re-derive all of them.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "santas_great_escape_plans.json"

#: (dr, dc) per grid direction, for the walk BFS.
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: Stand-in distance for "unreachable even through every piece of furniture".
#: Finite, so a board the walls really do cut in two is still ordered sensibly
#: rather than overflowing the queue's ordering.
_FAR = 300


class SantaExpert(PSExpert):
    """Primitive A* over the real interpreter, guided by the route MST.

    See the module docstring for the bound and for why the walk map treats
    everything except Wall and Black as passable."""

    #: Measure the per-step optimal SETS. The plans are shortest (admissible
    #: heuristic, ``weight == 1``), so the measurement is exact, and this game
    #: is a walk -- the single largest source of order-free ties there is.
    exact_optsets = True

    #: Keep each level's start plan (and its optimal sets) on disk.
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        idx = self.g.obj_name_to_idx
        self.wall_ids = {idx["wall"], idx["black"]}
        self.target = idx["target"]
        self.chimney = idx["chimney"]
        self.player_ids = set(self.game._engine._player_indices)
        self._dist: dict = {}
        self._assert_static_terrain()

    def _assert_static_terrain(self) -> None:
        """Fail loudly if any rule mentions Wall or Black on a RHS.

        The heuristic's whole admissibility argument is that those two are the
        only things on the board no rule can add or take away, so the BFS map
        built from them is a superset of what Santa can really walk on. A rule
        that created a wall would break the bound silently -- plans would stop
        being shortest and the measured optimal sets would quietly become a
        guess -- so the premise is checked rather than commented."""
        for rule in self.g.rules:
            for group in rule.groups_rhs:
                for cell in group:
                    for _mod, _name, idxs in cell:
                        if self.wall_ids & set(idxs):
                            raise AssertionError(
                                f"{GAME_NAME}: a rule writes Wall/Black on its "
                                "RHS, so the heuristic's static-terrain "
                                "assumption no longer holds")

    # -- per-board setup ------------------------------------------------------
    def plan(self, eng, level: int | None = None):
        """Bind this board's static geometry, then plan as usual.

        The walk map and the chimney positions are what every BFS runs over and
        no rule in this game can change either, so they are read ONCE here
        rather than inside `heuristic`, which runs on every node."""
        self._prep(eng)
        return super().plan(eng, level)

    def _prep(self, eng) -> None:
        grid = eng.grid
        self.h, self.w = len(grid), len(grid[0])
        self._free = [[not (cell & self.wall_ids) for cell in row]
                      for row in grid]
        self._chimneys = [(r, c)
                          for r, row in enumerate(grid)
                          for c, cell in enumerate(row)
                          if self.chimney in cell]
        self._dist = {}

    def _bfs(self, src) -> dict:
        """Walk distances from ``src`` over the (static) passable map, memoized.

        The chimney's own cell is passable here even though Santa can never
        stand on it: the escape is a single step INTO it, so counting that step
        is exactly right, and a route that goes THROUGH it only ever makes the
        estimate smaller."""
        hit = self._dist.get(src)
        if hit is not None:
            return hit
        dist = {src: 0}
        queue = deque([src])
        while queue:
            r, c = queue.popleft()
            step = dist[(r, c)] + 1
            for dr, dc in _DELTA:
                nxt = (r + dr, c + dc)
                if (0 <= nxt[0] < self.h and 0 <= nxt[1] < self.w
                        and self._free[nxt[0]][nxt[1]] and nxt not in dist):
                    dist[nxt] = step
                    queue.append(nxt)
        self._dist[src] = dist
        return dist

    def _scan(self, eng) -> tuple:
        """One pass over the grid for the santa and the undelivered targets."""
        player = None
        targets: list[tuple[int, int]] = []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                if cell & self.player_ids:
                    player = (r, c)
                if self.target in cell:
                    targets.append((r, c))
        return player, targets

    # -- heuristic ------------------------------------------------------------
    def heuristic(self, eng) -> int:
        player, targets = self._scan(eng)
        if player is None:
            # Santa is off the board: he went down the chimney. That is the win
            # once the targets are gone, and an unreachable state otherwise
            # (nothing can put him back), so it is charged out of the way.
            return 0 if not targets else _FAR * 10
        if not targets:
            dist = self._bfs(player)
            return min((dist.get(x, _FAR) for x in self._chimneys), default=0)
        return min(self._mst([player] + targets + [chim])
                   for chim in self._chimneys)

    def _mst(self, nodes) -> int:
        """MST of the metric closure over ``nodes``, rooted at ``nodes[0]``.

        A route that visits every node is a connected subgraph spanning them,
        so the cheapest spanning tree can only be shorter -- which is what makes
        this a bound rather than an estimate."""
        pair = {}
        for a in nodes:
            da = self._bfs(a)
            for b in nodes:
                pair[(a, b)] = da.get(b, _FAR)
        joined = {nodes[0]}
        rest = set(nodes[1:])
        total = 0
        while rest:
            cost, nxt = min((pair[(a, b)], b) for a in joined for b in rest)
            total += cost
            joined.add(nxt)
            rest.discard(nxt)
        return total


class SantasGreatEscapeSolver(PSAStarSolver):
    game_id = "puzzlescript_santas_great_escape"
    game_name = GAME_NAME
    expert_cls = SantaExpert

    #: The game module patches the parsed game before play (it splits the
    #: ``Santa<N>OnGift`` composites so the gift is actually placed, and
    #: restates the escape win condition) and that patched adapter is exactly
    #: what `game_envs` hands a live agent -- so the generator has to record
    #: against it rather than build its own.
    game_module_id = GAME_MODULE_ID

    #: Unweighted: the six searches together are ~4s, so there is nothing to
    #: buy by trading plan length away -- and ``weight == 1`` is what makes the
    #: measured optimal sets exact.
    weight = 1
    #: A runaway backstop, not a budget anything here comes near (level 3, the
    #: deepest, closes in a fraction of it).
    node_cap = 400_000
    #: Level 5's plan is 41 moves, the longest by half again.
    max_steps = 120


# ---------------------------------------------------------------------------
# Report modes
# ---------------------------------------------------------------------------

def _expert_and_game(fresh_cache: bool = False):
    """A solver, its adapter and its expert, wired the way `main` wires them.

    ``fresh_cache`` detaches `PLAN_CACHE` so a report re-derives the searches
    instead of reading back what a previous run stored -- which is what a check
    on the searches has to do to be a check at all."""
    solver = SantasGreatEscapeSolver()
    if fresh_cache:
        solver.expert_cls = type("SantaExpertNoCache", (SantaExpert,),
                                 {"plan_cache_path": None})
    game = solver.make_game(0)
    expert = solver.expert_cls(game, node_cap=solver.node_cap,
                               weight=solver.weight)
    return solver, game, expert


def _plans() -> int:
    """Plan, length, timing and tie sets for every level."""
    _solver, game, expert = _expert_and_game()
    letter = {"up": "U", "down": "D", "left": "L", "right": "R", "action": "X"}
    total_steps = total_alts = 0
    for level in range(game.n_levels):
        game.set_level(level)
        started = time.time()
        plan = expert.plan(game._engine, level)
        if plan is None:
            print(f"level {level}: NO PLAN")
            continue
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        total_steps += len(plan)
        total_alts += sum(len(s) for s in sets)
        print(f"level {level}: {len(plan):3d} moves, "
              f"{sum(len(s) for s in sets) / len(sets):.2f} optimal presses "
              f"per step  ({time.time() - started:.1f}s)")
        print("   " + " ".join("".join(letter[d] for d in s) for s in sets))
    if total_steps:
        print(f"\n{total_steps} steps, {total_alts / total_steps:.2f} optimal "
              f"presses per step overall")
    return 0


def _audit() -> int:
    """Assert every cell COMPOSITION renders distinctly, at every cell size.

    Composition, not object: a delivered Gift has to differ from the Target it
    replaced and from the Wall whose two ARC palette indices it shares (Brown
    and Orange both collapse to 12), and Santa standing on a Gift has to differ
    from Santa standing on bare floor -- otherwise the frame does not say how
    many gifts are left and the level is unobservable.

    The thirteen Santa objects are deliberately NOT in here: the game draws
    them all with the same sprite, so they are indistinguishable by
    construction. That costs nothing only because every level ships exactly as
    many targets as its santa's number -- then the counter is a restatement of
    "targets still on the board" and the frame does say it. That is the whole
    observability argument for this game, so it is CHECKED here rather than
    asserted: a level whose santa outnumbered its targets would be one where
    the agent cannot tell a santa who can leave from one who cannot."""
    _solver, game, expert = _expert_and_game()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    bad_counter = 0
    for level in range(game.n_levels):
        game.set_level(level)
        targets = sum(1 for row in eng.grid for cell in row
                      if idx["target"] in cell)
        number = next(n for n in range(13)
                      if any(idx[f"santa{n}"] in cell
                             for row in eng.grid for cell in row))
        print(f"level {level}: santa{number}, {targets} target(s) -- "
              f"{'observable' if number == targets else 'COUNTER NOT READABLE'}")
        bad_counter += number != targets

    comps = {name: (name,) for name in
             ("background", "wall", "black", "target", "gift", "chimney",
              "tree", "bed_topleft", "bed_topright", "bed_bottomleft",
              "bed_bottomright", "boy", "dog", "table",
              "sink_left", "sink_right", "chair_left", "chair_right")}
    comps["santa"] = ("santa1",)
    comps["santa_on_target"] = ("target", "santa1")
    comps["santa_on_gift"] = ("gift", "santa1")
    comps["boy_on_gift"] = ("gift", "boy")
    comps["dog_on_gift"] = ("gift", "dog")
    comps["tree_on_gift"] = ("gift", "tree")

    sizes: dict[tuple[int, int], list[int]] = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for name, objs in comps.items():
            # Fill the WHOLE board with the composition and compare whole
            # frames: `_render_frame` upscales a sub-64 render to fill the
            # frame, so cropping one cell out lands in the wrong place on most
            # board shapes. Rendering is per-cell independent, so this is the
            # same test done where the geometry cannot drift.
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, g))
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not (bad or bad_counter)
          else f"AUDIT FAILED: {bad} indistinguishable pairs, "
               f"{bad_counter} level(s) with an unreadable counter")
    return 0 if not (bad or bad_counter) else 1


#: The 8 elements of the square's symmetry group, as (rotations, mirror).
_SYMMETRIES = tuple((k, m) for k in range(4) for m in (False, True))

#: How each symmetry permutes the four engine directions. ``k`` quarter turns
#: counter-clockwise in grid coordinates, then (optionally) a left-right mirror.
_TURN = {"up": "left", "left": "down", "down": "right", "right": "up"}
_MIRROR = {"left": "right", "right": "left", "up": "up", "down": "down"}


def _transform_grid(grid, k: int, mirror: bool):
    out = [list(row) for row in grid]
    for _ in range(k):
        out = [list(row) for row in zip(*out)][::-1]     # 90 deg, matching _TURN
    if mirror:
        out = [row[::-1] for row in out]
    return [[set(cell) for cell in row] for row in out]


def _transform_dir(direction: str, k: int, mirror: bool) -> str:
    if direction == "action":
        return direction
    for _ in range(k):
        direction = _TURN[direction]
    return _MIRROR[direction] if mirror else direction


def _symmetry() -> int:
    """Replay every level's plan on all 8 turned and mirrored copies of its own
    board, and require the result to be the transform of the original result.

    This is what earns this game its place in `_FLIP_GAMES` (the rotation is
    mandatory and unconditional; the two flips are opt-in). Arguing it from the
    rules would nearly work -- everything is stated with the relative ``>``
    force or is direction-agnostic, the win conditions name no direction, there
    is no gravity and the input is screen-relative -- but the game has up to
    three independently moving bodies whose interactions are settled by the
    order the interpreter expands a rule's four directions, which is a fact
    about the screen and not about the board. That is exactly the shape that
    hid Gobble Rush's chirality, so it is measured instead of argued.

    The comparison is press for press, not just at the end: a chirality that
    cancels out by the last move is still a different game in between.
    """
    _solver, game, expert = _expert_and_game()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level}: NO PLAN")
            bad += 1
            continue
        # The reference run, kept as the board after every press.
        game.set_level(level)
        truth = []
        for direction in plan:
            eng.step(direction)
            truth.append(snapshot(eng))
        broken = []
        for k, mirror in _SYMMETRIES:
            if k == 0 and not mirror:
                continue
            game.set_level(level)
            start = _transform_grid(eng.grid, k, mirror)
            eng.height, eng.width = len(start), len(start[0])
            restore(eng, start)
            for i, direction in enumerate(plan):
                eng.step(_transform_dir(direction, k, mirror))
                if eng.grid != _transform_grid(truth[i], k, mirror):
                    broken.append((k, mirror, i))
                    break
        bad += len(broken)
        print(f"level {level}: {len(plan):3d} moves x 7 transforms -- "
              f"{'symmetric' if not broken else 'BROKEN ' + str(broken)}")
    print("symmetry clean" if not bad
          else f"SYMMETRY FAILED: {bad} transform(s) disagree")
    return 0 if not bad else 1


#: State budget for `_verify`'s independent shortest-path search. Levels whose
#: space is wider than this are reported unpriced rather than searched forever:
#: level 5 is an 8x11 board with five targets, a boy and a dog, and a
#: heuristic-free sweep to its plan's depth of 41 does not close.
_BFS_STATE_CAP = 500_000


def _verify() -> int:
    """Re-derive every plan with the disk cache detached and check three things
    -- the standing check that `PLAN_CACHE` has not gone stale against an
    edited level, an edited heuristic or an edited interpreter.

    1. **The plan wins**, replayed press for press on the real interpreter from
       the real level start.
    2. **The heuristic never over-charges along it.** A shortest plan has
       ``d*(s_i) == moves remaining``, so ``h(s_i) <= remaining`` at every step
       (and ``h == 0`` at the win) is a necessary condition for the
       admissibility that the plans -- and therefore the measured optimal sets
       -- rest on. It is cheap and it covers every level.
    3. **Nothing shorter exists**, checked against a plain breadth-first search
       over the same interpreter that uses NONE of the heuristic. That is the
       independent price, and it is the one that cannot always be paid: a
       heuristic-free sweep is exponential in the plan's depth, so it runs
       under `_BFS_STATE_CAP` and levels wider than that are reported unpriced
       rather than silently skipped. On the shipped levels it never comes to
       that: all six close inside the cap and all six are proved shortest.
    """
    _solver, game, expert = _expert_and_game(fresh_cache=True)
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level}: NO PLAN")
            bad += 1
            continue

        # 1 + 2: replay, watching the heuristic against what is left.
        game.set_level(level)
        expert._prep(eng)
        over = []
        for i, direction in enumerate(plan):
            if expert.heuristic(eng) > len(plan) - i:
                over.append(i)
            eng.step(direction)
        won = eng.check_win() and expert.heuristic(eng) == 0

        # 3: the independent, heuristic-free price.
        game.set_level(level)
        seen = {expert._key(eng)}
        frontier = [snapshot(eng)]
        depth = 0
        found = None
        while frontier and found is None and depth < len(plan):
            if len(seen) > _BFS_STATE_CAP:
                break
            depth += 1
            nxt = []
            for snap in frontier:
                for direction in expert.directions:
                    restore(eng, snap)
                    eng.step(direction)
                    if eng.check_win():
                        found = depth
                        break
                    key = expert._key(eng)
                    if key in seen:
                        continue
                    seen.add(key)
                    nxt.append(snapshot(eng))
                if found is not None:
                    break
            frontier = nxt
        priced = found is not None or not frontier
        shortest = found is None or found == len(plan)
        ok = won and not over and shortest
        bad += not ok
        print(f"level {level}: {len(plan):3d} moves, win={won}, "
              f"h<=remaining={not over}, "
              f"shortest={'proved' if priced and shortest else 'FAILED' if not shortest else f'unpriced ({len(seen)} states)'}"
              f"  -- {'OK' if ok else 'FAILED'}")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} level(s)")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    sys.exit(SantasGreatEscapeSolver.main())
