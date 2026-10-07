"""Generate Phase-1 training data for the PuzzleScript game ps:one_way_street
("One Way Street", Franklin P. Dyer).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
its disk cache, and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: a native model of the four rules, the exact field that
plans over it, and the two rendering fixes the game needed.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_one_way_street",
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
exactly. Each step also carries the full set of equally-optimal presses (see
"Optimal-action sets" below).

The game
--------
Sokoban with CHAINED pushes and one-way tiles. The whole rule set is::

    [> Player|Box] -> [> Player|> Box]
    [> Box|Box]    -> [> Box|> Box]

    VERTICAL[> Movable|Lefttile]   -> [Movable|Lefttile]
    RIGHT   [> Movable|Lefttile]   -> [Movable|Lefttile]
    ... and the same two cancels for each of the other three tiles

and the win condition is ``All goal on box`` -- every GOAL must carry a box, NOT
every box a goal. Level 8 has four boxes and one goal, so the two readings are
genuinely different here and the field's seed set has to be "every configuration
that covers the goals", not one fixed target configuration.

Read the cancels in the direction they are stated and a one-way tile is simply a
cell you may only ENTER while moving the way its arrow points; leaving it is
unrestricted, and standing on one restricts nothing. Three consequences, and the
third is the one a plausible-looking model gets wrong:

  * A cancel that lands on the last box of a chain kills the whole push. The box
    keeps its cell, Player/Box/Wall share a collision layer, so the box behind it
    is blocked, and so on back to the player: the turn is a total no-op.

  * The cancel is about the cell being ENTERED, whatever is standing in it. A
    tile with a box parked on it still cancels a player walking in from the
    wrong side.

  * **A push can move the box and leave the player behind.** The push rules run
    BEFORE the cancels, so a player pressing into a tile cell that holds a box
    hands the box its force first and only then loses its own: the box moves one
    cell and the player does not. That is the only way to get a box OFF a
    one-way tile in a direction the tile forbids, which makes it a mechanic and
    not a curiosity: level 1 cannot be solved without it. It is also RARE -- the
    342k-transition fuzz below reaches it 92 times, over five of the ten levels,
    because it needs a box parked on a tile with the player beside it on the
    wrong side -- so the coverage counter for it is printed separately, and it is
    level 1's engine-certified plan (``--plans``, and again under all eight
    presentations in ``--symmetry``) that really pins the behaviour down.

ACTION5 is bound to nothing, so the four directions are the whole action space.

The levels
----------
The shipped file held five boards. Five more were authored here around them (the
same treatment as ps:futuristic_block_pushing_game), and the whole set is
ordered by plan length:

    level  size    boxes goals  plan  ties  states   what it is
    0       7x9    1     1        10   0%      675   two rooms, one rightward door
    1       7x9    1     1        10   0%      696   a box shoved OFF an up-tile
    2       7x9    1     1        18   0%      286   ring corridor, shortcut closed
    3       9x9    2     2        22   0%     2285   SHIPPED
    4       6x9    1     1        23   0%      165   SHIPPED
    5      8x11    2     2        27   4%    32333   two boxes, one door, one chain push
    6      9x11    3     3        33  12%   401951   SHIPPED
    7       9x9    2     2        36   0%    10080   SHIPPED
    8       9x9    4     1        37   3%    33568   SHIPPED -- 4 boxes, ONE goal
    9      8x12    2     2        59   5%    26274   two boxes round a solid block

``plan`` is proved SHORTEST (not merely found), ``ties`` is the share of its
steps with more than one equally-optimal press, and ``states`` is the size of the
level's reachable state space. The longest plan is 59 presses against the
adapter's 200-step per-level budget, so this game needs no `games/` step-limit
wrapper the way ps:count_mover does.

Expert solver
-------------
Not a search over the interpreter, and not the family's macro A* either: the
mechanic collapses to a small native model (`_Board` -- boxes as a bitmask over
the free cells, the player as an index), so `_Board.field` computes the EXACT
distance-to-win of every reachable state and `_Board.plan` walks it downhill.

The field is built the other way round from the one in
ps:futuristic_block_pushing_game, and the reason is worth stating because it is
the reusable part. That game reverse-BFSes from the win over PULL-moves, which
needs (a) a hand-derived inverse of the forward rule and (b) a UNIQUE winning
configuration to seed from. Neither holds here: chained pushes and per-cell entry
permissions make the inverse relation fiddly enough to be a real risk, and with
four boxes for one goal the winning configurations number in the thousands. So
instead:

  1. a forward BFS from the start closes the reachable set, recording each
     state's predecessors as it goes;
  2. a backward BFS from every reachable winning state over those predecessors
     gives the exact distance-to-win.

That is exact for the same reason the reverse field is -- every shortest path
from the start stays inside the reachable set -- and it costs one pass over a
space that the one-way tiles keep small (the largest level closes at 402k states
in ~5s; all ten together are ~6s). What it gives up is the reverse field's free
deadlock pruning: a box shoved into a corner still enters the closure here, it
simply never gets a distance.

Three things this buys over a forward A* on macros:

  * **Optimality is proved, not hoped for.** `PSPushExpert` merges states by the
    player's REACHABLE REGION, which mis-costs a move-counted plan; and walk-and-
    push macros are not even sound in this game, since a press that pushes
    nothing can still be refused by a tile and a press that moves a box need not
    move the player.
  * **The optimal-action SETS are measured.** At distance ``d``, a press is
    optimal iff it reaches ``d - 1``. `PSPushExpert.annotate_walks` can only ever
    find ties between WALK steps and calls every push forced; here a tie between
    two different pushes is labelled too.
  * **Speed.** ~6s for all ten fields, cached to disk thereafter.

The interpreter is then only asked to CERTIFY: `--plans` replays each field plan
through the real engine and requires the win on the LAST press and no earlier,
and the recorder drives that same engine for every taped frame, so a model that
disagreed with PuzzleScript could not produce a WIN episode.

Checks
------
``--fuzz`` asserts `_Board` reproduces the interpreter exactly, fuzzing from
EVERY PREFIX of every level's own solution rather than from the level start --
random play from the start on these boards wanders and almost never lines a push
up, so the rules the game is made of would go unexercised. Coverage counters are
printed for that reason.

``--verify`` is the independent check of the SEARCH (the fuzz only ever reaches
`step`, the forward rule). Three parts, and the third is the one that leaves the
model behind entirely:

  * the backward field is recomputed WITHOUT the predecessor map, by scanning the
    whole reachable set once per BFS layer and keeping the states with a
    successor one step closer. It must agree everywhere, which is what certifies
    the predecessor bookkeeping in `field`.
  * a forward BFS from the start must put the nearest winning state at exactly
    the plan length, and each step's optimal set must equal
    ``{a : fwd(next) == i + 1 and back(next) == d* - i - 1}`` -- the double-entry
    formula, using both passes.
  * every LABELLED press is spliced into a variant plan (prefix + that press + a
    fresh optimal continuation) and replayed on the real INTERPRETER, which must
    win on press ``d*`` and not before. So the tie labels this generator ships
    are certified by PuzzleScript, not by `_Board`.

``--symmetry`` proves on the interpreter that the 8-element presentation group is
an exact symmetry of the mechanic, which is what entitles the game to
`PuzzleScriptAdapter._FLIP_GAMES`. It matters here more than usual: the tile
cancels are stated as eight separate direction-qualified rules rather than with
the relative ``>`` force, and reading them off as copies of each other is exactly
the argument that was wrong in ps:gobble_rush. Each level's plan is replayed on
all eight turned and mirrored copies of its own board, with the four TILE OBJECTS
RELABELLED by the transform -- transforming only the cells would leave a LeftTile
pointing left in a board where left has become up, i.e. would build a different
puzzle and report the game chiral (the trap `--symmetry` hit in ps:l_a_s_e_r).

The rendering fixes
-------------------
On the original art ``--audit`` reports SEVEN indistinguishable composition pairs
at both cell sizes in use, and all seven have one cause: the Player sprite was
opaque everywhere except its four OUTER corners, where the Goal has nothing to
show and all four tiles are the same blue. So ``player_on_goal`` was
pixel-identical to ``player`` -- the object the win condition names vanished
under the player -- and the four ``player_on_<tile>`` compositions were identical
to EACH OTHER, i.e. a player standing on a one-way tile did not show which way it
pointed. Two changes in ``data/puzzlescript_games/One_Way_Street.txt``:

  * **Player** gets four pixels punched out at rows/cols 1 and 3. That is
    precisely where the four tile sprites differ from one another and from bare
    floor (white/blue/white/blue for a LeftTile, blue/white/blue/white for a
    RightTile, white/white/blue/blue up, blue/blue/white/white down,
    white/white/white/white floor), and it is also where the Goal's ring sits.
    This one change clears all seven pairs.
  * **Goal** goes from ``darkblue`` to ``red``, and gains the four CELL CORNERS
    beside its ring. The colour is the fix that the audit does not catch, because
    it is not a pixel-identity: ``darkblue`` and the tiles' ``blue`` quantize to
    the SAME ARC index (9), so the goal was drawn in the tiles' own colour and
    told apart from an arrow only by shape. The corners make it readable under
    the player from its own art rather than only through the Player's new holes,
    and the ring is what shows through the Box's open middle.

``--audit`` is the check, over all nineteen compositions the game can produce
(floor, wall, goal, the four tiles, and box/player alone, on a goal and on each
tile) at every cell size the ten levels render at (5 and 7 px). It compares WHOLE
FRAMES of a board filled with each composition rather than slicing one cell out
by ``cell_px`` arithmetic: `_render_frame` upscales the board to fill 64x64 and
letterboxes it, so the output's cell grid is not ``cell_px``-aligned and the
arithmetic crop reads the wrong window.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the presentation
augmentations: rotation_k in {0,1,2,3} plus an independent horizontal and
vertical flip, each with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`, which this game was added to). 10 levels x 16
presentations = 160. There is deliberately no colour augmentation: box-on-goal is
read as the red goal ring showing through the box's open middle, which a
flattening recolor would erase.

The expert plan is therefore seed-independent: solved once per level, cached to
``data/one_way_street_plans.json``, and replayed per seed with that seed's
remapped screen actions.

Usage (run from the repo root):
    python solvers/generate_one_way_street_training.py \
        --episodes 200 --out data/training_multi_level/one_way_street

    python solvers/generate_one_way_street_training.py --plans
    python solvers/generate_one_way_street_training.py --verify
    python solvers/generate_one_way_street_training.py --fuzz
    python solvers/generate_one_way_street_training.py --symmetry
    python solvers/generate_one_way_street_training.py --audit
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

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "One_Way_Street"

#: Disk cache of each level's start plan AND its optimal-action sets. The fields
#: are seed-independent and cost ~6s together, which every shard of
#: `parallelize_generator` would otherwise repeat on every core. Delete the file
#: to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "one_way_street_plans.json"

#: Engine direction -> (dr, dc). The four moves are the whole action space.
_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}
#: Fixed iteration order, so a plan (and its tie sets) is reproducible.
_ORDER: tuple[str, ...] = ("up", "down", "left", "right")

#: PuzzleScript object name -> the one direction it admits entry from.
_TILE_NAMES: dict[str, str] = {
    "lefttile": "left", "righttile": "right",
    "uptile": "up", "downtile": "down",
}

#: Refuse to close a reachable set larger than this. It is a MEMORY budget, not a
#: time one -- each state is a dict entry plus its predecessor list -- and it
#: exists so an edited level that opens a board up fails loudly instead of
#: swapping. The largest level here closes at 402k states.
STATE_CAP = 4_000_000


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One Way Street over the level's free cells.

    A state is ``(box_mask, player_index)``: boxes as a bitmask over the
    free-cell index, the player as an index into the same list. Walls are not
    represented at all -- they are exactly the cells absent from ``free``, so
    "off the board" and "into a wall" are the same lookup miss.

    Two precomputed tables carry the whole rule set:

      * ``nxt[d][i]`` -- the index of cell ``i``'s neighbour in direction ``d``,
        or -1 when that is a wall or off the board;
      * ``ok[d][i]``  -- may an object ENTER cell ``i`` while moving ``d``?
        False only for a one-way tile whose arrow points elsewhere.

    `step` is the game; `field` inverts it.
    """

    def __init__(self, walls, tiles, goals, boxes, player, h, w):
        self.h, self.w = h, w
        self.free = [(r, c) for r in range(h) for c in range(w)
                     if (r, c) not in walls]
        self.idx = {cell: i for i, cell in enumerate(self.free)}
        self.n = len(self.free)
        self.gmask = self._mask(goals)
        self.start = (self._mask(boxes), self.idx[player])
        self.nxt = {}
        self.ok = {}
        for d, (dr, dc) in _DELTA.items():
            self.nxt[d] = [self.idx.get((r + dr, c + dc), -1)
                           for (r, c) in self.free]
            self.ok[d] = [tiles.get(cell, d) == d for cell in self.free]

    @classmethod
    def read(cls, eng, ids) -> "_Board":
        """Build the model from the interpreter's current grid."""
        wall, tile_ids, goal, box, players = ids
        walls, tiles, goals, boxes, player = set(), {}, [], [], None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & wall:
                    walls.add((r, c))
                    continue                 # a wall cell can hold nothing else
                for obj, direction in tile_ids.items():
                    if obj in cell:
                        tiles[(r, c)] = direction
                if cell & goal:
                    goals.append((r, c))
                if cell & box:
                    boxes.append((r, c))
                if cell & players:
                    player = (r, c)
        return cls(walls, tiles, goals, boxes, player, eng.height, eng.width)

    def _mask(self, cells) -> int:
        m = 0
        for cell in cells:
            m |= 1 << self.idx[cell]
        return m

    def win(self, state) -> bool:
        """``All goal on box``: every GOAL carries a box. Note the direction --
        level 8 has four boxes for one goal, so this is strictly weaker than
        "every box is on a goal"."""
        return state[0] & self.gmask == self.gmask

    # -- dynamics -------------------------------------------------------------
    def step(self, state, direction):
        """One press. Returns the new state, or ``state`` itself when the
        interpreter would cancel the turn.

        The turn resolves in the order PuzzleScript applies the rules. First the
        push rules hand a force to the player and, through it, to the whole
        contiguous run of boxes ahead of it. Then each of those objects loses its
        force if the cell it would enter is a one-way tile pointing elsewhere.
        Finally the forces that survived are resolved: an object moves only if
        the object ahead of it moved (or there was none and the cell beyond is
        free), because Player, Box and Wall share a collision layer.

        Since the surviving-force test is per-object and the collision cascade
        runs from the front, the movers are always a SUFFIX of the run: the
        objects from the first index whose force survived, ``t``, up to the last
        box. ``t == 0`` is the ordinary case where the player moves too; ``t == 1``
        is the one where the player's own force was cancelled by a tile in the
        cell it was pushing INTO, and the box moves without it."""
        mask, p = state
        nxt, ok = self.nxt[direction], self.ok[direction]
        chain = [p]
        x = nxt[p]
        while x >= 0 and (mask >> x) & 1:
            chain.append(x)
            x = nxt[x]
        if x < 0:
            return state                 # the far end is a wall / off the board
        k = len(chain) - 1
        t = k + 1
        for i in range(k, -1, -1):
            if not ok[chain[i + 1] if i < k else x]:
                break                    # this object's force is cancelled
            t = i
        if t > k:
            return state                 # nothing survived: a no-op turn
        # Every mover shifts one cell along the run, so the net effect on the
        # mask is always "the first moving box leaves, the far end gains one".
        first_box = max(t, 1)
        if first_box <= k:
            mask = mask ^ (1 << chain[first_box]) | (1 << x)
        return (mask, (chain[1] if k else x) if t == 0 else p)

    # -- the exact distance-to-win field --------------------------------------
    def field(self, cap: int = STATE_CAP) -> tuple[dict, dict]:
        """``(dist, fwd)`` -- distance to the win, and distance from the start,
        for every state reachable from the level start.

        A forward closure that records predecessors as it goes, then a backward
        BFS from every reachable WINNING state over those predecessors. Exact in
        PRIMITIVE PRESSES (the unit the agent pays) and with no canonicalisation
        applied: every shortest path out of the start stays inside the reachable
        set, so restricting the backward pass to that set cannot shorten one.

        Why not the usual reverse BFS from the win over pull-moves: it needs a
        hand-derived inverse of the forward rule (awkward with chained pushes and
        per-cell entry permissions) and a unique goal configuration to seed from,
        which ``All goal on box`` does not give when a level has spare boxes.

        A refused press maps a state to itself; those self-loops are dropped
        rather than recorded, so no state is ever its own predecessor.

        Deadlocks need no test: a box shoved somewhere no press can retrieve it
        from still enters the closure, it just never receives a distance, and
        `plan` reports the level unsolvable if the START is one of those."""
        start = self.start
        pred: dict = {}
        fwd = {start: 0}
        queue: deque = deque([start])
        while queue:
            state = queue.popleft()
            d = fwd[state] + 1
            for direction in _ORDER:
                nxt = self.step(state, direction)
                if nxt == state:
                    continue                          # refused: a self-loop
                pred.setdefault(nxt, []).append(state)
                if nxt not in fwd:
                    fwd[nxt] = d
                    queue.append(nxt)
            if len(fwd) > cap:
                raise MemoryError(
                    f"reachable set exceeded {cap} states on a {self.h}x{self.w} "
                    f"board with {bin(self.start[0]).count('1')} boxes")
        dist: dict = {}
        queue = deque()
        for state in fwd:
            if self.win(state):
                dist[state] = 0
                queue.append(state)
        while queue:
            state = queue.popleft()
            d = dist[state] + 1
            for prev in pred.get(state, ()):
                if prev not in dist:
                    dist[prev] = d
                    queue.append(prev)
        return dist, fwd

    def plan(self, dist: dict):
        """``(presses, optsets)`` from the start, or ``(None, None)``.

        Walk the field downhill. At every state the OPTIMAL SET is exactly the
        presses reaching distance ``d - 1``: a refused press leaves the state
        (and so the distance) unchanged and is never in it, and a press onto a
        state the field does not hold is a press that threw the win away."""
        state = self.start
        if state not in dist:
            return None, None
        presses, optsets = [], []
        while dist[state]:
            want = dist[state] - 1
            best = [a for a in _ORDER
                    if dist.get(self.step(state, a), -1) == want]
            presses.append(best[0])
            optsets.append(best)
            state = self.step(state, best[0])
        return presses, optsets


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class OneWayStreetExpert(PSExpert):
    """Plans by building `_Board`'s exact field off the engine grid and walking
    it downhill; never steps the interpreter.

    `PSExpert` still owns everything around that -- the in-memory memo, the
    on-disk plan cache with its staleness check, the level scoping and the
    snapshot/restore discipline -- so the only override is `_search`. `heuristic`
    is unreachable by construction: nothing here runs A*.
    """

    #: `_key` below is dynamic-objects-only, canonical only WITHIN a level.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        g = self.g
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.goal_ids = set(g.resolve_object_name("goal"))
        self.box_ids = set(g.resolve_object_name("box"))
        self.player_ids = set(self.game._engine._player_indices)
        self.tile_ids = {g.resolve_object_name(name)[0]: direction
                         for name, direction in _TILE_NAMES.items()}
        self.dyn_ids = self.box_ids | self.player_ids

    def _key(self, eng) -> frozenset:
        # Boxes + player. Walls, goals and tiles are static per level -- no rule
        # in this game creates or destroys any of them -- hence ``scope_by_level``.
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn)
        )

    def heuristic(self, eng) -> int:
        raise AssertionError("the field is exact; no search runs here")

    def board(self, eng) -> _Board:
        return _Board.read(eng, (self.wall_ids, self.tile_ids, self.goal_ids,
                                 self.box_ids, self.player_ids))

    def _search(self, eng) -> list | None:
        board = self.board(eng)
        dist, _fwd = board.field()
        presses, optsets = board.plan(dist)
        return None if presses is None else Plan(presses, optsets)


class OneWayStreetSolver(PSAStarSolver):
    game_id = "puzzlescript_one_way_street"
    game_name = GAME_NAME
    expert_cls = OneWayStreetExpert

    #: The longest plan is 59 presses; the rest is room for the exploration
    #: prefix and the replay after it. Stays under the adapter's 200-step
    #: per-level budget, which would otherwise flip a long level to GAME_OVER
    #: mid-plan.
    max_steps = 150

    def prepare_expert(self, game, expert) -> None:
        """Build every level's field before `discover_solvable` asks for it --
        the same work either way, but it makes the one-time cost visible as
        startup rather than as a mysteriously slow first seed, and it fills the
        disk cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _levels():
    """``(solver, game, expert)`` with the adapter parked at level 0."""
    solver = OneWayStreetSolver()
    game = solver.make_game(0)
    return solver, game, OneWayStreetExpert(game)


def _report() -> int:
    """Per-level board size, piece count, plan length, tie coverage and state
    count -- and CERTIFY each plan by replaying it through the interpreter."""
    _solver, game, expert = _levels()
    total = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = expert.board(eng)
        if eng.check_win():
            print(f"level {level:2d}: STARTS ALREADY WON")
            bad += 1
            continue
        t = time.time()
        dist, fwd = board.field()
        presses, optsets = board.plan(dist)
        took = time.time() - t
        if presses is None:
            print(f"level {level:2d}: UNSOLVABLE")
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
        print(f"level {level:2d}: {board.h:2d}x{board.w:2d} free={board.n:2d} "
              f"{bin(board.start[0]).count('1')} boxes "
              f"{bin(board.gmask).count('1')} goals, {len(presses):3d} presses "
              f"(budget {game._max_steps}, {room}), "
              f"{ties:3d} tie steps ({ties / len(presses):3.0%}), "
              f"{len(fwd):7d} reachable ({len(dist):7d} scored) in "
              f"{took:5.2f}s, "
              f"{'CERTIFIED' if ok else 'PLAN REJECTED BY THE INTERPRETER'}")
    print(f"total {total} presses over {game.n_levels} levels")
    return 0 if not bad else 1


def _verify() -> int:
    """Independent check of the SEARCH -- the part `--fuzz` cannot reach.

    The fuzz only ever exercises `step`, the forward rule. Everything the plans
    actually rest on is downstream of that: the predecessor bookkeeping in
    `field`, the backward pass, and the claim that a labelled press is genuinely
    one of the shortest. Three checks, in increasing independence:

      * **the backward field, recomputed without the predecessor map.** One scan
        of the whole reachable set per BFS layer, keeping the states with a
        successor at distance ``d - 1``. Slower (O(states x d*)) and completely
        separate bookkeeping; it must agree on every state.
      * **the double-entry formula.** A forward BFS from the start must put the
        nearest winning state at exactly the plan length, and step ``i``'s
        optimal set must be ``{a : fwd(next) == i + 1 and back(next) ==
        d* - i - 1}`` -- which uses both passes and so fails if either is wrong.
      * **every labelled press replayed on the INTERPRETER.** Each press in each
        optimal set is spliced into a variant plan (the prefix, that press, then
        a fresh optimal continuation from wherever it lands) and driven through
        the real engine, which must reach ``check_win`` on press ``d*`` and not
        before. This one leaves `_Board` behind entirely: it is PuzzleScript
        certifying the tie labels this generator ships.

    Slower than everything else here (the layer scan on level 6 is 402k states x
    33 layers), so it is opt-in rather than part of ``--plans``."""
    _solver, game, expert = _levels()
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        layout = game._game.levels[level]
        board = expert.board(eng)
        back, fwd = board.field()
        presses, optsets = board.plan(back)
        star = len(presses)
        notes = []

        # 1. the backward field again, by layer scan, with no predecessor map.
        t = time.time()
        scan = {s: 0 for s in fwd if board.win(s)}
        frontier = set(scan)
        d = 0
        while frontier:
            d += 1
            frontier = {s for s in fwd if s not in scan
                        and any(board.step(s, a) in frontier for a in _ORDER)}
            for s in frontier:
                scan[s] = d
        if scan != back:
            notes.append(f"LAYER SCAN differs on {len(set(scan) ^ set(back))} states")
        scan_t = time.time() - t

        # 2. the double-entry formula, over both passes.
        wins = [fwd[s] for s in fwd if board.win(s)]
        if min(wins) != star:
            notes.append(f"LENGTH {min(wins)} != {star}")
        state = board.start
        for i, direction in enumerate(presses):
            want = [a for a in _ORDER
                    if fwd.get(board.step(state, a), -1) == i + 1
                    and back.get(board.step(state, a), -1) == star - i - 1]
            if want != optsets[i]:
                notes.append(f"step {i}: {optsets[i]} != {want}")
            state = board.step(state, direction)

        # 3. every labelled press, spliced into a variant plan and replayed on
        #    the real interpreter.
        replays = 0
        state = board.start
        for i, alts in enumerate(optsets):
            for alt in alts:
                nxt = board.step(state, alt)
                tail, _ = _variant(board, nxt).plan(back)
                variant = presses[:i] + [alt] + tail
                replays += 1
                eng.load_level(layout)
                won_at = None
                for j, direction in enumerate(variant):
                    eng.step(direction)
                    if eng.check_win():
                        won_at = j
                        break
                if won_at != star - 1 or len(variant) != star:
                    notes.append(f"step {i} press {alt}: interpreter won at "
                                 f"{won_at} of {len(variant)}")
            state = board.step(state, presses[i])

        bad += len(notes)
        print(f"level {level:2d}: {len(fwd):7d} reachable, layer scan {scan_t:6.2f}s, "
              f"d*={star:3d}, {replays:3d} labelled presses replayed: "
              f"{'AGREES' if not notes else '; '.join(notes[:3])}")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} disagreements")
    return 0 if not bad else 1


def _variant(board: _Board, state):
    """``board`` with its start moved to ``state`` -- so `_Board.plan` can be
    reused to build an optimal CONTINUATION from mid-plan without recomputing
    the field."""
    clone = object.__new__(_Board)
    clone.__dict__ = dict(board.__dict__)
    clone.start = state
    return clone


def _fuzz(trials: int = 40, walk: int = 30) -> int:
    """Assert `_Board` reproduces the interpreter exactly.

    Random play from the level START is not enough: on these boards it wanders
    and almost never lines a push up, so the rules the whole game is made of
    would go unexercised. Instead fuzz from EVERY PREFIX of each level's own
    solution, which puts the model in the box configurations the plans actually
    visit. Coverage counters are printed for the same reason -- a run reporting
    agreement while having pushed nothing has checked nothing -- and they are
    split out by the three shapes a turn can take, including the one this game
    is named for: a press that moves a box and leaves the PLAYER standing. That
    last one is rare on purpose-built boards (92 of 342k transitions) -- see the
    module docstring for why it is level 1's plan, not the fuzz, that pins it
    down."""
    _solver, game, expert = _levels()
    rng = random.Random(20260814)
    bad = pushes = walks = blocked = solo = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        layout = game._game.levels[level]
        board = expert.board(eng)
        dist, _fwd = board.field()
        presses, _ = board.plan(dist)
        lvl = [0, 0, 0, 0]
        for prefix in range(len(presses) + 1):
            for _ in range(trials):
                eng.load_level(layout)
                state = board.start
                for direction in presses[:prefix]:
                    eng.step(direction)
                    state = board.step(state, direction)
                for _ in range(walk):
                    direction = rng.choice(_ORDER)
                    before = state
                    eng.step(direction)
                    state = board.step(state, direction)
                    mask, p = state
                    mine = (frozenset(board.free[i] for i in range(board.n)
                                      if (mask >> i) & 1), board.free[p])
                    boxes, player = set(), None
                    for r, row in enumerate(eng.grid):
                        for c, cell in enumerate(row):
                            if cell & expert.box_ids:
                                boxes.add((r, c))
                            if cell & expert.player_ids:
                                player = (r, c)
                    if mine != (frozenset(boxes), player):
                        bad += 1
                        if bad < 5:
                            print(f"  level {level} MISMATCH after {direction}: "
                                  f"model {mine} engine {(boxes, player)}")
                    if eng.check_win() != board.win(state):
                        bad += 1
                        print(f"  level {level} WIN MISMATCH")
                    if state == before:
                        lvl[2] += 1
                    elif state[0] == before[0]:
                        lvl[1] += 1
                    elif state[1] == before[1]:
                        lvl[3] += 1          # the box moved, the player did not
                    else:
                        lvl[0] += 1
        pushes += lvl[0]
        walks += lvl[1]
        blocked += lvl[2]
        solo += lvl[3]
        print(f"level {level:2d}: {lvl[0]:6d} pushes  {lvl[1]:6d} walks  "
              f"{lvl[2]:6d} refused  {lvl[3]:5d} box-without-player")
    print(f"{pushes + walks + blocked + solo} transitions ({pushes} pushes, "
          f"{walks} walks, {blocked} refused, {solo} box-without-player): "
          f"{'model matches the interpreter' if not bad else f'{bad} MISMATCHES'}")
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


def _dynamic(eng, expert):
    """``(player, boxes)`` off the interpreter grid -- the only things a
    presentation transform is allowed to be asserted on."""
    player, boxes = None, set()
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if cell & expert.player_ids:
                player = (r, c)
            if cell & expert.box_ids:
                boxes.add((r, c))
    return player, frozenset(boxes)


def _symmetry() -> int:
    """Prove ON THE INTERPRETER that the 8-element presentation group is an exact
    symmetry of the mechanic, which is what entitles this game to
    `PuzzleScriptAdapter._FLIP_GAMES`.

    It matters more here than in a plain sokoban. The two push rules use the
    relative ``>`` force and are symmetric by inspection, but the tile cancels are
    eight separate DIRECTION-QUALIFIED rules (``VERTICAL[...Lefttile]``,
    ``RIGHT[...Lefttile]``, ...), and "read the blocks and declare them copies of
    each other" is exactly the argument that was wrong in ps:gobble_rush, where
    the ORDER the interpreter expands a rule's directions decided who won a
    contested cell.

    So each level's plan is replayed on all eight turned and mirrored copies of
    its own board, built by transforming the LEVEL LAYOUT (the interpreter then
    re-runs the level-start rules itself), and every press must leave the pieces
    exactly where the transform of the reference run put them.

    **The tiles are RELABELLED by the transform**, and that is the whole subtlety:
    a tile's direction is baked into object IDENTITY (four distinct objects), so
    moving only the cells would leave a LeftTile admitting leftward entry on a
    board where left has become up -- a different puzzle, which would report the
    game chiral even though it is not. That is the trap `--symmetry` hit in
    ps:l_a_s_e_r with its four laser facings."""
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    relabel = {g.resolve_object_name(name)[0]: direction
               for name, direction in _TILE_NAMES.items()}
    by_dir = {direction: obj for obj, direction in relabel.items()}
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        dist, _fwd = board.field()
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
                    turned[tr][tc] = {by_dir[dmap[relabel[o]]] if o in relabel
                                      else o for o in objs}
            eng.load_level(turned)
            for i, direction in enumerate(presses):
                eng.step(dmap[direction])
                player, boxes = ref[i + 1]
                want = (cell(player, hw),
                        frozenset(cell(x, hw) for x in boxes))
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

    Both bugs this exists for shipped in the original art, and both were in the
    STACK rather than in one sprite: Goal quantized onto the tiles' blue AND was
    drawn exactly under the Player's opaque region, and all four tiles are
    identical at the four corners the Player leaves transparent -- so a player
    standing on a one-way tile did not show which way it pointed. An
    object-by-object colour check catches neither.

    Whole frames are compared, not cell crops: `_render_frame` upscales the board
    to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and the arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor)."""
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    tiles = list(_TILE_NAMES)
    comps = {"floor": (), "wall": ("wall",), "goal": ("goal",),
             "box": ("box",), "box_on_goal": ("goal", "box"),
             "player": ("player",), "player_on_goal": ("goal", "player")}
    for tile in tiles:
        comps[tile] = (tile,)
        comps[f"box_on_{tile}"] = (tile, "box")
        comps[f"player_on_{tile}"] = (tile, "player")

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
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {len(comps)} compositions, "
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
    sys.exit(OneWayStreetSolver.main())
