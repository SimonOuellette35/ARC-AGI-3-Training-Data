"""Generate Phase-1 training data for the PuzzleScript game ps:icecrates
("IceCrates", Tyler Glaiel).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: a native model of the
slide, the exhaustive reachable-state oracle built on it, the exact
optimal-action sets that oracle hands over for free, and the reports that
certify all of it against the interpreter.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_icecrates",
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
frames exactly. Every expert step carries the full set of equally-optimal
presses (see "Optimal-action sets" below).

The game
--------
You are on ice. A press sends you sliding until something stops you, and you win
by coming to REST on the goal -- the game's own level-2 message ("You must land
ON the goal") is the whole puzzle, because the goal itself does not stop you.
Crates get shoved along the way and can be dropped into water to make new floor.

Everything below was measured against the interpreter, not read off the .txt:

* **One press is a whole SLIDE.** The ``[ PlayerL ] -> [ LEFT PlayerL ] again``
  loop runs the run to completion inside a single ``eng.step``, so the branching
  factor is four and a plan is a handful of presses, not a handful of cells.
* **You stop only at an OBSTACLE**, and ``Obstacle = Wall or StillCrate or
  Water``. The Goal is on its own collision layer and is not one, so landing on
  it means finding a slide whose stopping wall is directly beyond it.
* **Water is a wall until you fill it.** The player can never enter water; the
  only thing that changes that is a crate.
* **Crates do not slide on their own.** ``LEFT [ PlayerL | Crate ] ->
  [ PlayerL | LEFT Crate ] again`` re-applies the push force every tick, so a
  crate moves only while the player is directly behind it: the player and the
  whole chain of crates ahead of it travel in lockstep and stop together. A
  crate is never left coasting across the board.
* **A crate pushed into water is destroyed and the water becomes FilledWater**
  (``LEFT [ > Crate | Water ] -> [ | FilledWater]``), which is a separate
  collision layer, i.e. ordinary floor. One crate fills one water cell; the
  crates behind it immediately move into the space it left, and the player
  slides on over the patch it made. Crates are the scarce resource, and spending
  one on the wrong hole is the only irreversible move in the game.
* **StillCrate is not a piece, it is a one-tick marker.** ``LEFT [ > Crate |
  Obstacle ] -> [StillCrate | Obstacle]`` stops a crate and makes it an obstacle
  for the same tick, which is what cascades the stop back down the chain and
  then stops the player; ``[StillCrate]->[Crate]`` turns it back at the end of
  the rule pass, before anything moves.
* **There is no death, and no wait.** Nothing kills the player, no rule
  restarts the level, and pressing into a wall is a complete no-op (rule 6 turns
  PlayerL straight back into PlayerStill before rule 7 can set ``again``), so a
  bump cannot be used to pass a turn either. The only way to lose a level is to
  strand or waste the crates it needs.
* ACTION5 does nothing (the game declares ``noaction`` and no rule reads it), so
  `IceCratesExpert.directions` is the four moves and the search never branches
  on it.
* The six ``Water1..Water6`` objects are a purely cosmetic animation that
  advances on every tick, including every ``again`` tick inside a press. It is
  state the model deliberately does not carry -- `IceCratesExpert._key`
  collapses all six to one id, or two identical boards half a slide apart would
  key as different states.

Level indices here and in every report are 0-BASED, i.e. one less than the
number in the game's own "Level N" messages.

Why an exhaustive oracle, and not the shared A*
-----------------------------------------------
Measured first, per the ps:entrepotphage_demake lesson: the interpreter runs at
~460 ``eng.step``/s here (a press is a whole slide, and the water animation
rescans the board on every tick of it), which is too slow to search through. But
there is very little to search: the player can only ever come to rest against an
obstacle, so the reachable state space of a whole level is 13 to 1247 states and
`_Board.field` enumerates all of it in under 20ms.

So this generator does not search. It enumerates the reachable component with
`_Board.step` -- a native model of the rules above, held to the interpreter by
``--fuzz`` -- and runs a backward BFS from the winning states over the recorded
predecessors, which gives the exact distance-to-win of every state. A forward
enumeration plus a predecessor map is used rather than ps:futuristic_block_
pushing_game's reverse pull-field because this game's transition relation is not
invertible: a crate that fell in the water cannot be un-drowned, so there is no
backward rule to run.

That buys three things a heuristic search would not:

* plans that are provably SHORTEST: 4, 5, 12, 29, 8 and 32 presses;
* exact optimal-action SETS -- ``dist(successor) == dist - 1`` -- rather than
  ones inferred from the shape of the plan;
* a real proof of solvability (or, on an edited level, of unsolvability).

Because the whole thing costs milliseconds there is no ``plan_cache_path``:
every `parallelize_generator` shard re-derives the plans, which is cheaper than
the staleness risk a cached entry would carry.

Optimal-action sets
-------------------
The field gives these exactly and for free: at a state on a shortest path, a
press is optimal iff it reaches a state whose distance-to-win is one less. A
refused press (into a wall) leaves the state unchanged, so it is never optimal,
and a press that throws the level away lands outside the field. No step ships
unlabelled (the always-emit-optimal-targets rule), and a step with a unique
optimal press ships a one-element set. ``--verify`` re-derives every one of them
from an independent BFS.

The rendering fixes
-------------------
Three, all in ``data/puzzlescript_games/IceCrates.txt``, and all about the same
failure: an object that MATTERS being invisible under the object standing on it.

  * **Player**: the original sprite was opaque everywhere except its four
    corners, so a player standing on the goal was pixel-identical to a player
    standing on bare ice -- the frame the win condition is about was not in the
    observation at all. It is now a 3x3 blob inside a transparent ring.
  * **Crate**: solid, so crate-on-goal and crate-on-a-filled-hole were both
    pixel-identical to crate-on-ice. It is now a frame with a 3x3 window.
  * **Goal**: a flag drawn on part of the cell, in two greens that the ARC
    palette collapses to one index anyway. It is now a solid green square, so
    it reads through both windows above.

The windows span sprite rows/cols 1-3, not 1-2: the two biggest levels render
at 4px per cell, where `_render_cell_sprite`'s centered nearest sampling keeps
sprite rows/cols 0,1,3,4 and drops the MIDDLE one. A window inset to rows/cols
1-2 survives as a single pixel; measured, that is 1.1 differing pixels per cell
against 4.4 for the shipped shape.

``--audit`` is the check, and it compares WHOLE FRAMES of a board filled with
each cell composition rather than slicing one cell out by ``cell_px``
arithmetic: `_render_frame` upscales the board to fill 64x64, so the cell grid
in the output is not ``cell_px``-aligned and the arithmetic crop reads the wrong
window. The eleven compositions are checked at every cell size the six levels
render at (4, 5, 6, 7 and 9 px).

Augmentation
------------
This game's engine state after reset is identical for every seed (levels are
fixed ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip, each with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`, which this game was added to). Sliding on
ice is gravity-free, the win condition names no direction, input is
screen-relative, and the four per-direction rule blocks are structural copies of
each other, so every element of the 8-element symmetry group is an exact
symmetry of the mechanic. ``--symmetry`` proves it on the interpreter rather
than by reading the rules: each level's plan is replayed on all eight turned and
mirrored copies of its own board and every press must land the pieces exactly
where the transform of the reference run puts them. 6 levels x 16 presentations
= 96.

The expert plan is therefore seed-independent: solved once per level and
replayed per seed with that seed's remapped screen actions.

Usage (run from the repo root):
    python solvers/generate_icecrates_training.py \
        --episodes 200 --out data/training_multi_level/icecrates

    python solvers/generate_icecrates_training.py --plans
    python solvers/generate_icecrates_training.py --verify
    python solvers/generate_icecrates_training.py --fuzz
    python solvers/generate_icecrates_training.py --symmetry
    python solvers/generate_icecrates_training.py --audit
"""

from __future__ import annotations

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

GAME_NAME = "IceCrates"

#: Engine direction -> (dr, dc). The four moves are the whole action space.
_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}
#: Fixed iteration order, so a plan (and its tie sets) is reproducible.
_ORDER: tuple[str, ...] = ("up", "down", "left", "right")

#: Refuse to enumerate more than this. It is a MEMORY budget -- each state is a
#: dict entry holding two frozensets -- and it exists so an edited level that
#: opens the board up fails loudly instead of swapping. The largest level here
#: reaches 1247 states, so this is four orders of magnitude of headroom.
STATE_CAP = 5_000_000


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """The whole of IceCrates' rule set, as a state machine over

        ``(player cell, frozenset of crate cells, frozenset of water cells)``

    Walls and the goal are static -- no rule in the game creates or destroys
    either -- so they live on the board, not in the state. Filled-in water is
    not carried either: a filled cell is exactly a cell that has left ``waters``
    and it behaves as ordinary floor, so "was there ever a crate here" is not a
    distinction the dynamics can see. The six water-animation phases are
    likewise absent; they are cosmetic.
    """

    def __init__(self, h, w, walls, goals, crates, waters, player):
        self.h, self.w = h, w
        self.walls = walls
        self.goals = goals
        self.start = (player, frozenset(crates), frozenset(waters))

    @classmethod
    def read(cls, eng, ids) -> "_Board":
        """Build the model from the interpreter's current grid."""
        wall, water, player_ids, crate, goal = ids
        walls, goals, crates, waters, player = set(), set(), set(), set(), None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & wall:
                    walls.add((r, c))
                if cell & water:
                    waters.add((r, c))
                if cell & crate:
                    crates.add((r, c))
                if cell & goal:
                    goals.add((r, c))
                if cell & player_ids:
                    player = (r, c)
        return cls(eng.height, eng.width, frozenset(walls), frozenset(goals),
                   crates, waters, player)

    def won(self, state) -> bool:
        """``Some PlayerStill on Goal``. Every state here is a SETTLED board --
        the slide has finished, so the player object is PlayerStill -- which is
        why standing on the goal is the whole test."""
        return state[0] in self.goals

    # -- dynamics -------------------------------------------------------------
    def step(self, state, direction):
        """One press: the entire slide. Returns ``(state, offgrid)``.

        The loop body is one ``again`` tick, in the interpreter's rule order:

          1. the run of crates directly ahead of the player all take the force
             (``[ PlayerL | Crate ]`` then ``[ > Crate | Crate ]``);
          2. a moving crate facing WATER is destroyed and the water becomes
             floor -- only the front one of a run can be, since the others face
             crates;
          3. a moving crate facing an obstacle stops, which stops the whole run
             behind it and, being an obstacle itself for this tick, the player
             too;
          4. otherwise the player stops iff IT faces an obstacle;
          5. if nothing stopped, everything moves one cell and the tick repeats.

        ``offgrid`` flags the one board shape this game has no rule for: sliding
        at the edge of the map. ``[ PlayerL | Obstacle ]`` needs a CELL to match
        an obstacle in, so at the boundary nothing stops the run, the move
        silently fails, and ``again`` re-arms forever until the interpreter's
        50-iteration cap leaves a PlayerL on the board that no later press can
        turn back into a PlayerStill -- a soft-lock. Every level here walls its
        play area in (``--plans`` reports the count, and it is 0 everywhere), so
        this is a guard against an edited level, not a live case."""
        p, crates, waters = state
        crates, waters = set(crates), set(waters)
        dr, dc = _DELTA[direction]
        offgrid = False
        while True:
            chain = []
            q = (p[0] + dr, p[1] + dc)
            while q in crates:
                chain.append(q)
                q = (q[0] + dr, q[1] + dc)

            stopped = False
            if chain:
                ahead = (chain[-1][0] + dr, chain[-1][1] + dc)
                if ahead in waters:                    # rule 4: fill the hole
                    crates.discard(chain.pop())
                    waters.discard(ahead)
                if chain:                              # rule 5: front crate jams
                    ahead = (chain[-1][0] + dr, chain[-1][1] + dc)
                    if not self._on(ahead):
                        offgrid = stopped = True
                    elif ahead in self.walls or ahead in waters:
                        stopped = True
            if not stopped and not chain:              # rule 6: the player jams
                ahead = (p[0] + dr, p[1] + dc)
                if not self._on(ahead):
                    offgrid = stopped = True
                elif ahead in self.walls or ahead in waters:
                    stopped = True
            if stopped:
                break

            for cell in chain:
                crates.discard(cell)
            for cell in chain:
                crates.add((cell[0] + dr, cell[1] + dc))
            p = (p[0] + dr, p[1] + dc)
        return (p, frozenset(crates), frozenset(waters)), offgrid

    def _on(self, cell) -> bool:
        return 0 <= cell[0] < self.h and 0 <= cell[1] < self.w

    # -- the exact distance-to-win field --------------------------------------
    def field(self, cap: int = STATE_CAP):
        """``({state: presses to the win}, stats)`` over the whole reachable
        component.

        Two passes. The first enumerates every state reachable from the start,
        recording each state's PREDECESSORS; winning states are recorded but not
        expanded, because the episode ends there. The second is a BFS backwards
        over those predecessors from every winning state, which gives the exact
        distance-to-win.

        The predecessor map is what stands in for a reverse rule here: drowning
        a crate destroys it, so `step` has no inverse to run backwards (compare
        ps:futuristic_block_pushing_game, whose pushes invert to pulls). The
        price is that the whole component has to be enumerated first -- which
        for this game is milliseconds."""
        start = self.start
        seen = {start}
        preds: dict = {}
        queue = deque([start])
        wins, offgrid, edges = [], 0, 0
        while queue:
            state = queue.popleft()
            if self.won(state):
                wins.append(state)
                continue
            for direction in _ORDER:
                nxt, off = self.step(state, direction)
                if off:
                    offgrid += 1
                    continue
                edges += 1
                preds.setdefault(nxt, []).append(state)
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
                    if len(seen) > cap:
                        raise MemoryError(
                            f"reachable space exceeded {cap} states on a "
                            f"{self.h}x{self.w} board")

        dist = {w: 0 for w in wins}
        queue = deque(wins)
        while queue:
            state = queue.popleft()
            d = dist[state] + 1
            for prev in preds.get(state, ()):
                if prev not in dist:
                    dist[prev] = d
                    queue.append(prev)
        return dist, {"states": len(seen), "wins": len(wins),
                      "edges": edges, "offgrid": offgrid}

    def plan(self, dist: dict):
        """``(presses, optsets)`` from the start, or ``(None, None)``.

        Walk the field downhill. At every state the OPTIMAL SET is exactly the
        presses reaching distance ``d - 1``: a press into a wall leaves the
        state (and so the distance) unchanged and is never in it, and a press
        onto a state the field does not hold is a press that threw the level
        away."""
        state = self.start
        if state not in dist:
            return None, None
        presses, optsets = [], []
        while dist[state]:
            want = dist[state] - 1
            best = [d for d in _ORDER
                    if dist.get(self.step(state, d)[0], -1) == want]
            presses.append(best[0])
            optsets.append(best)
            state = self.step(state, best[0])[0]
        return presses, optsets


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class IceCratesExpert(PSExpert):
    """Plans by enumerating `_Board`'s reachable component off the engine grid
    and walking the resulting field downhill; never steps the interpreter.

    `PSExpert` still owns everything around that -- the in-memory memo, the
    level scoping and the snapshot/restore discipline -- so the only override is
    `_search`. `heuristic` is unreachable by construction: nothing here runs A*.
    """

    #: `_key` below is dynamic-objects-only, canonical only WITHIN a level.
    scope_by_level = True
    #: ``noaction`` in the header and no rule reads ACTION5, so it is not a move.
    directions = ["up", "down", "left", "right"]

    def setup(self) -> None:
        g = self.g
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.water_ids = set(g.resolve_object_name("water"))
        self.crate_ids = set(g.resolve_object_name("crate"))
        self.goal_ids = set(g.resolve_object_name("goal"))
        self.filled_ids = set(g.resolve_object_name("filledwater"))
        self.player_ids = set(self.game._engine._player_indices)

    def _key(self, eng) -> frozenset:
        """Player, crates and water, with every object class collapsed to one
        id per class.

        Two collapses matter. The six ``Water1..Water6`` phases advance on every
        tick, so keying on the concrete object would give the same board up to
        six different keys and defeat the memo. The five ``Player*`` facings are
        one id for the same reason (only PlayerStill can be seen at rest, but
        the key should not depend on that). Walls and the goal are static per
        level, hence ``scope_by_level``; FilledWater is exactly "a cell that
        used to be water", so the water set already carries it."""
        classes = ((self.player_ids, 0), (self.crate_ids, 1), (self.water_ids, 2))
        return frozenset(
            (r, c, tag)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for ids, tag in classes
            if cell & ids
        )

    def heuristic(self, eng) -> int:
        raise AssertionError("the field is exact; no search runs here")

    def board(self, eng) -> _Board:
        return _Board.read(eng, (self.wall_ids, self.water_ids, self.player_ids,
                                 self.crate_ids, self.goal_ids))

    def _search(self, eng) -> list | None:
        board = self.board(eng)
        dist, _stats = board.field()
        presses, optsets = board.plan(dist)
        return None if presses is None else Plan(presses, optsets)


class IceCratesSolver(PSAStarSolver):
    game_id = "puzzlescript_icecrates"
    game_name = GAME_NAME
    expert_cls = IceCratesExpert

    #: The longest plan is 32 presses; the rest is room for the exploration
    #: prefix and the re-plan after it. Stays well under the adapter's 200-step
    #: per-level budget, which would otherwise flip a level to GAME_OVER
    #: mid-plan.
    max_steps = 120


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _levels():
    """``(solver, game, expert)`` with the adapter parked at level 0."""
    solver = IceCratesSolver()
    game = solver.make_game(0)
    return solver, game, IceCratesExpert(game)


def _dynamic(eng, expert) -> tuple:
    """The pieces, read off the interpreter: ``(player, crates, waters,
    filled)``. Phase-free, so it can be compared across runs and across
    presentations."""
    crates, waters, filled, player = set(), set(), set(), None
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if cell & expert.crate_ids:
                crates.add((r, c))
            if cell & expert.water_ids:
                waters.add((r, c))
            if cell & expert.filled_ids:
                filled.add((r, c))
            if cell & expert.player_ids:
                player = (r, c)
    return player, frozenset(crates), frozenset(waters), frozenset(filled)


def _report() -> int:
    """Per-level board size, piece count, plan length, tie coverage and the size
    of the reachable space -- and CERTIFY each plan by replaying it through the
    interpreter."""
    _solver, game, expert = _levels()
    total = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = expert.board(eng)
        t = time.time()
        dist, stats = board.field()
        presses, optsets = board.plan(dist)
        took = time.time() - t
        if stats["offgrid"]:
            print(f"level {level:2d}: {stats['offgrid']} OFF-GRID SLIDES "
                  f"(the play area is not walled in; see _Board.step)")
            bad += 1
        if presses is None:
            print(f"level {level:2d}: UNSOLVABLE "
                  f"({stats['states']} states, no winning one)")
            bad += 1
            continue
        # Certification: the interpreter must win on the LAST press and no
        # earlier (an earlier win would mean the plan is not shortest).
        won_at = None
        for i, direction in enumerate(presses):
            eng.step(direction)
            if eng.check_win():
                won_at = i
                break
        ok = won_at == len(presses) - 1
        bad += not ok
        total += len(presses)
        ties = sum(1 for s in optsets if len(s) > 1)
        room = "ok" if len(presses) < game._max_steps else "OVER BUDGET"
        print(f"level {level:2d}: {board.h:2d}x{board.w:2d} "
              f"{len(board.start[1])} crates {len(board.start[2])} water, "
              f"{len(presses):3d} presses (budget {game._max_steps}, {room}), "
              f"{ties:3d} tie steps ({ties / len(presses):3.0%}), "
              f"space {stats['states']:5d} states / {stats['wins']:2d} winning "
              f"in {took:5.3f}s, {stats['offgrid']} off-grid, "
              f"{'CERTIFIED' if ok else 'PLAN REJECTED BY THE INTERPRETER'}")
    print(f"total {total} presses over {game.n_levels} levels")
    return 0 if not bad else 1


def _verify() -> int:
    """Double-entry check of the plan lengths AND the tie sets.

    `_Board.field` derives both from one backward sweep over a predecessor map
    built during the forward enumeration, so a bug in that bookkeeping would be
    invisible to `_report` (whose certification only says the plan wins on its
    last press, not that no shorter one exists) and to `--fuzz` (which only ever
    exercises `step`).

    So the answer is recomputed from the other end, sharing nothing with the
    backward pass: a plain forward BFS to the nearest winning state, run
    independently from the start and from each of the four successors of every
    plan step. The plan length must equal the start's distance, and the optimal
    set at step ``i`` must be exactly the presses whose successor's distance is
    one less than the current state's."""
    _solver, game, expert = _levels()
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(game._engine)
        dist, _stats = board.field()
        presses, optsets = board.plan(dist)

        def to_win(state, _b=board):
            """Shortest press count from ``state`` to a win, by plain BFS."""
            if _b.won(state):
                return 0
            seen, queue = {state}, deque([(state, 0)])
            while queue:
                cur, d = queue.popleft()
                for direction in _ORDER:
                    nxt, off = _b.step(cur, direction)
                    if off or nxt in seen:
                        continue
                    if _b.won(nxt):
                        return d + 1
                    seen.add(nxt)
                    queue.append((nxt, d + 1))
            return None

        t = time.time()
        notes = []
        star = to_win(board.start)
        if star != len(presses):
            notes.append(f"LENGTH {star} != {len(presses)}")
        state = board.start
        here = star
        for i, direction in enumerate(presses):
            want = []
            for alt in _ORDER:
                nxt, off = board.step(state, alt)
                if off:
                    continue
                d = to_win(nxt)
                if d is not None and d + 1 == here:
                    want.append(alt)
            if want != optsets[i]:
                notes.append(f"step {i}: {optsets[i]} != {want}")
            state = board.step(state, direction)[0]
            here -= 1
        bad += len(notes)
        print(f"level {level:2d}: d*={star:3d} by an independent BFS, "
              f"{sum(len(s) for s in optsets):3d} labelled presses checked in "
              f"{time.time() - t:5.2f}s: "
              f"{'AGREES' if not notes else '; '.join(notes[:3])}")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} disagreements")
    return 0 if not bad else 1


def _fuzz(trials: int = 25, walk: int = 25) -> int:
    """Assert `_Board` reproduces the interpreter exactly.

    Random play from the level START is not enough: the interesting rules are
    the crate ones, and a random walker on these boards mostly slides around
    without ever lining a crate up against water. So the fuzz restarts from
    EVERY PREFIX of each level's own solution, which puts the model in the crate
    and water configurations the plans actually visit, and prints COVERAGE
    COUNTERS -- a run reporting agreement while having drowned no crate has
    checked nothing about the rule the game is named for."""
    _solver, game, expert = _levels()
    rng = random.Random(20260814)
    bad = 0
    tot = {"slide": 0, "push": 0, "drown": 0, "refused": 0, "chain": 0}
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        layout = game._game.levels[level]
        board = expert.board(eng)
        dist, _stats = board.field()
        presses, _optsets = board.plan(dist)
        cnt = {k: 0 for k in tot}
        for prefix in range(len(presses) + 1):
            for _ in range(trials):
                eng.load_level(layout)
                state = board.start
                for direction in presses[:prefix]:
                    eng.step(direction)
                    state = board.step(state, direction)[0]
                for _ in range(walk):
                    direction = rng.choice(_ORDER)
                    before = state
                    eng.step(direction)
                    state, off = board.step(state, direction)
                    if off:
                        break            # the interpreter soft-locks here
                    player, crates, waters, filled = _dynamic(eng, expert)
                    if (state[0], state[1], state[2]) != (player, crates, waters):
                        bad += 1
                        if bad < 5:
                            print(f"  level {level} MISMATCH after {direction}: "
                                  f"model {state} engine {(player, crates, waters)}")
                        break
                    if filled != board.start[2] - state[2]:
                        bad += 1
                        print(f"  level {level} FILLED-WATER MISMATCH")
                        break
                    if eng.check_win() != board.won(state):
                        bad += 1
                        print(f"  level {level} WIN MISMATCH")
                        break
                    if state == before:
                        cnt["refused"] += 1
                    elif len(state[2]) < len(before[2]):
                        cnt["drown"] += 1
                    elif state[1] != before[1]:
                        cnt["push"] += 1
                        moved = len(state[1] - before[1])
                        cnt["chain"] += moved > 1
                    else:
                        cnt["slide"] += 1
        for k in cnt:
            tot[k] += cnt[k]
        print(f"level {level:2d}: {cnt['slide']:6d} slides {cnt['push']:5d} pushes "
              f"({cnt['chain']:4d} multi-crate) {cnt['drown']:4d} drowned crates "
              f"{cnt['refused']:6d} refused")
    n = sum(tot.values())
    print(f"{n} transitions "
          f"({tot['push']} pushes, {tot['drown']} drowned crates): "
          f"{'model matches the interpreter' if not bad else f'{bad} MISMATCHES'}")
    if not tot["drown"] or not tot["chain"]:
        print("FUZZ TOO WEAK: it never drowned a crate or never pushed a run of them")
        return 1
    return 0 if not bad else 1


# -- the symmetry group -----------------------------------------------------

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
        if k % 2:
            h, w = w, h
        return (h, w)

    dmap = {}
    for name, (dr, dc) in _DELTA.items():
        if mirror:
            dc = -dc
        for _ in range(k):
            dr, dc = dc, -dr
        dmap[name] = next(n for n, d in _DELTA.items() if d == (dr, dc))
    return cell, dims, dmap


def _symmetry() -> int:
    """Prove ON THE INTERPRETER that the 8-element presentation group is an
    exact symmetry of the mechanic, which is what entitles this game to the
    `PuzzleScriptAdapter._FLIP_GAMES` augmentation.

    The rules are written out per direction (a LEFT block, a RIGHT block, ...)
    rather than with the relative ``>`` force, and it is that per-direction loop
    that has hidden a chirality in other ps: games -- the rule ORDER, not the
    mechanic, decides who wins a contested cell. Reading the four blocks and
    declaring them copies of each other is exactly the argument that was wrong
    there, so instead each level's plan is replayed on all eight turned and
    mirrored copies of its own board (built by transforming the LEVEL LAYOUT, so
    the interpreter re-runs the level-start rules and re-derives the wall art
    itself), and every press must leave the pieces exactly where the transform
    of the reference run put them.

    Only the pieces are compared: the sixteen ``Wall_xxxx`` objects encode which
    neighbours a wall has, so they are supposed to differ between presentations,
    and the water animation is a phase nobody should be asserting on."""
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        dist, _stats = board.field()
        presses, _optsets = board.plan(dist)
        layout = g.levels[level]
        hw = (len(layout), len(layout[0]))

        eng.load_level(layout)
        ref = [_dynamic(eng, expert)]
        for direction in presses:
            eng.step(direction)
            ref.append(_dynamic(eng, expert))

        notes = []
        for k, mirror in itertools.product(range(4), (False, True)):
            if (k, mirror) == (0, False):
                continue
            cell, dims, dmap = _transform(k, mirror)
            th, tw = dims(hw)
            turned = [[set() for _ in range(tw)] for _ in range(th)]
            for r, row in enumerate(layout):
                for c, objs in enumerate(row):
                    tr, tc = cell((r, c), hw)
                    turned[tr][tc] = set(objs)
            eng.load_level(turned)
            for i, direction in enumerate(presses):
                eng.step(dmap[direction])
                player, crates, waters, filled = ref[i + 1]
                want = (cell(player, hw),
                        frozenset(cell(x, hw) for x in crates),
                        frozenset(cell(x, hw) for x in waters),
                        frozenset(cell(x, hw) for x in filled))
                if _dynamic(eng, expert) != want:
                    notes.append(f"rot{k}{'m' if mirror else ''}@press{i}")
                    break
        bad += len(notes)
        print(f"level {level:2d}: {len(presses):3d} presses x 7 presentations: "
              f"{'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:3])}")
    print("symmetry clean -- the flip augmentation is exact" if not bad
          else f"SYMMETRY FAILED: {bad} chiral presentations")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    Three bugs this exists for all shipped in the original art, and all were in
    the STACK rather than in one sprite: the Player covered the Goal completely
    (so the win frame was unreadable), and the Crate covered both the Goal and a
    filled-in water hole. An object-by-object colour check catches none of them.

    Whole frames are compared, not cell crops: `_render_frame` upscales the
    board to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and the arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor)."""
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {
        "floor": (), "wall": ("wall_1111",),
        "water": ("water1",), "water_phase3": ("water3",),
        "filled": ("filledwater",), "goal": ("goal",),
        "crate": ("crate",), "player": ("playerstill",),
        "crate_on_goal": ("goal", "crate"),
        "player_on_goal": ("goal", "playerstill"),
        "crate_on_filled": ("filledwater", "crate"),
        "player_on_filled": ("filledwater", "playerstill"),
    }

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
        # The win frame is the one that must not merely differ but be READABLE,
        # so its margin is reported rather than just its distinctness.
        win_px = int((shots["player_on_goal"] != shots["player"]).sum()) / (h * w)
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"player-on-goal reads {win_px:4.1f}px/cell against player, "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
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
    sys.exit(IceCratesSolver.main())
