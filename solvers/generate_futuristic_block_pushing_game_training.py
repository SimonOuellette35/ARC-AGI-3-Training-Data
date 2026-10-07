"""Generate Phase-1 training data for the PuzzleScript game
ps:futuristic_block_pushing_game ("Futuristic Block Pushing Game", Jazzy
Williams' one-board Sokoban).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
its disk cache, and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: a native model of the one rule, the exact reverse-BFS
field that plans over it, and the two rendering fixes the game needed.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_futuristic_block_pushing_game",
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
Textbook Sokoban, and nothing else. One rule --

    [> Player | Crate] -> [> Player | > Crate]

-- and one win condition, ``All Target on Crate``. Player, Wall and Crate share
a collision layer, so a crate shoved into a wall or into a second crate simply
cancels the turn and pushes never chain; Target sits on its own layer below
them. ACTION5 is bound to nothing, so the four directions are the whole action
space. Note the win condition is stated the other way round from the usual
``All Crate on Target`` -- every TARGET must carry a crate. Every level here has
as many crates as targets, so the two readings coincide, but a level with a
spare crate would not be won by parking it anywhere.

The levels
----------
The shipped file held ONE board -- level 9 below, 43 moves. One level is not a
corpus: this game takes no colour augmentation, so its whole presentation space
would be 1 level x 4 rotations x 4 flips = 16 boards, and a policy would be
memorising one maze rather than learning what a push is. The file now carries
13, twelve of them authored here around the original and ordered by plan length:

    level  size    crates  plan  ties  states   what it is
    0       7x9    1         9    0%      816   one push per axis, open room
    1       7x9    2        16   25%    2 912   two crates around a pillar
    2       8x9    2        18   17%    8 917   two crates, L-shaped room
    3       7x8    1        19    0%      442   the crate is behind a wall
    4      7x10    4        23   39%  602 434   a comb of four, pushed up a rank
    5      9x11    2        24   21%   75 876   open room, opposite corners
    6      9x11    2        27    0%   15 050   zigzag corridors
    7      9x10    2        28   11%   12 186   staircase
    8       9x9    3        32   16%   77 572   three crates, targets far side
    9       9x9    4        43    2%  231 033   THE SHIPPED BOARD
    10     9x10    3        52   10%   96 092   one gap in a full-width wall
    11    10x12    3        62   18%  837 716   two rooms, one door
    12     9x10    3        73   16%   52 705   a diagonal barrier

``plan`` is proved SHORTEST (not merely found) -- see the expert below --
``ties`` is the share of its steps with more than one equally-optimal press, and
``states`` is the size of that level's reverse-BFS field. The longest plan is 73
moves against the adapter's 200-step per-level budget, so this game needs no
`games/` step-limit wrapper the way ps:count_mover does.

Every level is engine-verified: the field's plan is replayed through the real
interpreter and must reach ``check_win`` on its last press and no earlier
(``--plans``). None starts already won, and each has as many crates as targets.
The three shapes that quietly make a Sokoban board impossible were all hit while
authoring these and are worth naming, because none of them raises anything -- the
level just reports unsolvable:

  * a crate on the row or column NEXT TO the outer wall cannot be pushed away
    from it, because the player would have to stand inside the wall;
  * a 2x2 block of crates is frozen -- every push of every member needs a cell
    another member occupies;
  * a crate in a one-cell-wide corridor can only travel ALONG it, so a target
    around a bend is unreachable.

Expert solver
-------------
Not a search over the interpreter: this game's mechanic collapses to plain
Sokoban, so `_Board` is a native model (crates as a bitmask over the free cells,
the player as an index) and `_Board.field` is an exact **reverse BFS from the
win**, over pull-moves rather than pushes. That gives the true distance-to-win
of every state that can reach the win at all, which buys three things a forward
A* over macros does not:

  * **Optimality is proved, not hoped for.** `PSSokobanExpert`'s push-distance
    heuristic is a greedy matching, i.e. a guide rather than a certificate, and
    `PSPushExpert` additionally merges states by the player's REACHABLE REGION,
    which can mis-cost a move-counted plan by the region's diameter. The field
    counts primitive moves exactly and merges nothing.
  * **The optimal-action SETS are measured, not inferred.** See below.
  * **Speed.** All 13 fields together take ~3s; the interpreter macro A* needs
    ~24s for level 9 alone. Deadlocked configurations never enter the field at
    all -- they cannot reach the win, so the backward search never generates
    them -- which is what keeps the state counts in the table small.

The interpreter is then only asked to CERTIFY: `plan()` replays the field's
answer through the real engine (``--plans``), and the recorder drives that same
engine for every taped frame, so a model that disagreed with PuzzleScript could
not produce a WIN episode.

The model was fuzz-checked against the interpreter before being trusted: 40
random rollouts from every prefix of every level's own solution, comparing the
crate set, the player cell and ``check_win`` after each press. That covers what
random play alone does not -- ~2.8k pushes and ~18k refused moves on the shipped
board, where uniform random play from the start mostly wanders. ``--fuzz``
re-runs it.

Optimal-action sets
-------------------
The field gives these exactly and for free: at a state ``s`` on a shortest path,
a press is optimal iff it reaches a state whose distance-to-win is one less.
That is stronger than `PSPushExpert.annotate_walks`, which infers ties by
classifying each step as a walk or a push and labels every push as forced --
correct in a Sokoban, but it can only ever find ties BETWEEN walk steps, and it
never notices that two different pushes finish in the same number of moves. Here
both kinds are labelled. No step ships unlabelled (the always-emit-optimal
-targets rule), and a step with a unique optimal press ships a one-element set.

The rendering fixes
-------------------
Two, both in ``data/puzzlescript_games/Futuristic_Block_Pushing_Game.txt`` and
both about the Target, which is the object the win condition names:

  * ``Target`` was ``blue``, and Wall is ``blue darkblue`` -- ARC quantization
    sends all three to index 9, so a target beside a wall was the wall's own
    colour. It is now ``red`` (8), which nothing else in the game uses.
  * ``Target`` was drawn as a hollow ring on rows/cols 1-3, and the Player
    sprite is OPAQUE across exactly those eight cells. So a player standing on a
    target was pixel-identical to a player standing on bare floor -- the game's
    goal state was unreadable at the moment it mattered. It is now a SOLID
    square, which shows through the Crate's open middle AND through the Player's
    transparent corners.

``--audit`` is the check, and it compares WHOLE FRAMES of a board filled with
each cell composition rather than slicing one cell out by ``cell_px``
arithmetic: `_render_frame` upscales the board to fill 64x64 and letterboxes it,
so the cell grid in the output is not ``cell_px``-aligned and the arithmetic
crop reads the wrong window. The compositions are floor, wall, target, crate,
crate-on-target, player and player-on-target, checked at every cell size the 13
levels actually render at (5, 6, 7 and 8 px).

Augmentation
------------
This game's engine state after reset is identical for every seed (levels are
fixed ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip, each with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`, which this game was added to), which
together re-sample the board's 8-element symmetry group. Sokoban is
gravity-free, its one rule is stated with the relative ``>`` force, its win
condition names no direction and its input is screen-relative, so a flip is an
exact symmetry of the mechanic; no sprite is chiral in the sense that matters --
a mirrored Player is still the only object shaped like a Player, and Wall,
Crate and Target are all symmetric under the whole group. There is deliberately
no colour augmentation: crate-on-target is read as "the red target showing
through the crate's open middle", which a flattening recolor would erase.
13 levels x 16 presentations = 208.

The expert plan is therefore seed-independent: solved once per level, cached to
``data/futuristic_block_pushing_game_plans.json``, and replayed per seed with
that seed's remapped screen actions.

Usage (run from the repo root):
    python solvers/generate_futuristic_block_pushing_game_training.py \
        --episodes 200 --out data/training_multi_level/futuristic_block_pushing_game

    python solvers/generate_futuristic_block_pushing_game_training.py --plans
    python solvers/generate_futuristic_block_pushing_game_training.py --verify
    python solvers/generate_futuristic_block_pushing_game_training.py --audit
    python solvers/generate_futuristic_block_pushing_game_training.py --fuzz
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

GAME_NAME = "Futuristic_Block_Pushing_Game"

#: Disk cache of each level's start plan AND its optimal-action sets. The fields
#: are seed-independent and cost ~3s together, which every shard of
#: `parallelize_generator` would otherwise repeat on every core. Delete the file
#: to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "futuristic_block_pushing_game_plans.json"

#: Engine direction -> (dr, dc). The four moves are the whole action space.
_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}
#: Fixed iteration order, so a plan (and its tie sets) is reproducible.
_ORDER: tuple[str, ...] = ("up", "down", "left", "right")

#: Refuse to build a field larger than this. It is a MEMORY budget, not a time
#: one -- each state is a dict entry keyed by a (mask, player) pair -- and it
#: exists so an edited level that opens the board up fails loudly instead of
#: swapping. The largest level here builds 838k states.
STATE_CAP = 4_000_000


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """Plain Sokoban over the level's free cells.

    A state is ``(crate_mask, player_index)``: crates as a bitmask over the
    free-cell index, the player as an index into the same list. Walls are not
    represented at all -- they are exactly the cells absent from ``free``, so
    "off the board" and "into a wall" are the same lookup miss.

    The whole of this game's rule set is `step`. `field` inverts it.
    """

    def __init__(self, walls, targets, crates, player, h, w):
        self.h, self.w = h, w
        self.free = [(r, c) for r in range(h) for c in range(w)
                     if (r, c) not in walls]
        self.idx = {cell: i for i, cell in enumerate(self.free)}
        self.n = len(self.free)
        self.tmask = self._mask(targets)
        self.start = (self._mask(crates), self.idx[player])

    @classmethod
    def read(cls, eng, ids) -> "_Board":
        """Build the model from the interpreter's current grid."""
        wall, target, crate, players = ids
        walls, targets, crates, player = set(), [], [], None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & wall:
                    walls.add((r, c))
                    continue                 # a wall cell can hold nothing else
                if cell & target:
                    targets.append((r, c))
                if cell & crate:
                    crates.append((r, c))
                if cell & players:
                    player = (r, c)
        return cls(walls, targets, crates, player, eng.height, eng.width)

    def _mask(self, cells) -> int:
        m = 0
        for cell in cells:
            m |= 1 << self.idx[cell]
        return m

    # -- dynamics -------------------------------------------------------------
    def step(self, state, direction):
        """One press. Returns the new state, or ``state`` itself when the
        interpreter would cancel the turn.

        Three cases, and they are the whole game: walking into a wall or off the
        board does nothing; walking into a crate pushes it, unless the cell
        beyond is a wall or another crate (Player, Wall and Crate share one
        collision layer, so a blocked crate blocks the player too and pushes
        never chain); anything else is a bare move."""
        mask, p = state
        dr, dc = _DELTA[direction]
        r, c = self.free[p]
        j = self.idx.get((r + dr, c + dc))
        if j is None:
            return state                                   # wall / off-board
        if not (mask >> j) & 1:
            return (mask, j)                               # bare move
        k = self.idx.get((r + 2 * dr, c + 2 * dc))
        if k is None or (mask >> k) & 1:
            return state                                   # crate blocked
        return (mask ^ (1 << j) | (1 << k), j)             # push

    # -- the exact distance-to-win field --------------------------------------
    def field(self, cap: int = STATE_CAP) -> dict:
        """``{state: moves to the win}`` for every state that can reach the win.

        A reverse BFS from the goal, so the distances are exact in PRIMITIVE
        MOVES -- the unit the agent pays -- and no canonicalisation is applied.
        The goal crate configuration is unique (this game has as many crates as
        targets), so the seeds are ``(targets, p)`` for every cell ``p`` the
        player could be standing on, and the search runs outward from there.

        The reverse of a press is a PULL, and there are two ways a state could
        have been reached from the cell ``q`` behind the player:

          * a bare move -- the crates are unchanged, and ``q`` must have been
            free of them (the player cannot have been standing on a crate);
          * a push -- the crate now one step AHEAD of the player was on the
            player's own cell, so put it back and take it off the cell ahead.

        Deadlocks need no test: a crate parked where no pull can reach it simply
        never appears in the field, and `plan` reports the level unsolvable if
        the start is one of those states."""
        dist: dict = {}
        queue: deque = deque()
        tmask, free, idx = self.tmask, self.free, self.idx
        for p in range(self.n):
            if (tmask >> p) & 1:
                continue                       # the player cannot stand on a crate
            dist[(tmask, p)] = 0
            queue.append((tmask, p))
        while queue:
            state = queue.popleft()
            mask, p = state
            d = dist[state] + 1
            r, c = free[p]
            for dr, dc in _DELTA.values():
                q = idx.get((r - dr, c - dc))          # where the player came from
                if q is None or (mask >> q) & 1:
                    continue                           # wall, or it held a crate
                if not (mask >> p) & 1:                # (the player's own cell)
                    prev = (mask, q)                   # ...it was a bare move
                    if prev not in dist:
                        dist[prev] = d
                        queue.append(prev)
                k = idx.get((r + dr, c + dc))          # ...or it was a push
                if k is not None and (mask >> k) & 1 and not (mask >> p) & 1:
                    prev = (mask ^ (1 << k) | (1 << p), q)
                    if prev not in dist:
                        dist[prev] = d
                        queue.append(prev)
            if len(dist) > cap:
                raise MemoryError(
                    f"field exceeded {cap} states on a {self.h}x{self.w} board "
                    f"with {bin(self.start[0]).count('1')} crates")
        return dist

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

class FuturisticBlockPushingExpert(PSExpert):
    """Plans by building `_Board`'s exact reverse-BFS field off the engine grid
    and walking it downhill; never steps the interpreter.

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
        self.target_ids = set(g.resolve_object_name("target"))
        self.crate_ids = set(g.resolve_object_name("crate"))
        self.player_ids = set(self.game._engine._player_indices)
        self.dyn_ids = self.crate_ids | self.player_ids

    def _key(self, eng) -> frozenset:
        # Crates + player. Walls and targets are static per level -- no rule in
        # this game creates or destroys either -- hence ``scope_by_level``.
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
        return _Board.read(eng, (self.wall_ids, self.target_ids,
                                 self.crate_ids, self.player_ids))

    def _search(self, eng) -> list | None:
        board = self.board(eng)
        presses, optsets = board.plan(board.field())
        return None if presses is None else Plan(presses, optsets)


class FuturisticBlockPushingSolver(PSAStarSolver):
    game_id = "puzzlescript_futuristic_block_pushing_game"
    game_name = GAME_NAME
    expert_cls = FuturisticBlockPushingExpert

    #: The longest plan is 73 moves; the rest is room for the exploration prefix
    #: and the re-plan after it. Stays under the adapter's 200-step per-level
    #: budget, which would otherwise flip a long level to GAME_OVER mid-plan.
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
    solver = FuturisticBlockPushingSolver()
    game = solver.make_game(0)
    return solver, game, FuturisticBlockPushingExpert(game)


def _report() -> int:
    """Per-level board size, piece count, plan length, tie coverage and field
    size -- and CERTIFY each plan by replaying it through the interpreter."""
    _solver, game, expert = _levels()
    total = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = expert.board(eng)
        crates = bin(board.start[0]).count("1")
        t = time.time()
        dist = board.field()
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
              f"{crates} crates, {len(presses):3d} moves "
              f"(budget {game._max_steps}, {room}), "
              f"{ties:3d} tie steps ({ties / len(presses):3.0%}), "
              f"field {len(dist):7d} states in {took:5.2f}s, "
              f"{'CERTIFIED' if ok else 'PLAN REJECTED BY THE INTERPRETER'}")
    print(f"total {total} moves over {game.n_levels} levels")
    return 0 if not bad else 1


def _verify() -> int:
    """Double-entry check of the plans AND the tie sets, by a FORWARD BFS.

    `_Board.field` runs backwards from the win over pull-moves, which is the one
    piece of hand-derived logic in this file that the interpreter fuzz cannot
    reach (the fuzz only exercises `step`, the forward rule). So the answer is
    recomputed here from the other end: a plain forward BFS from the level start
    over `step`, which shares no code with the reverse pass.

    Two things are then asserted, and each uses BOTH fields:

      * ``d* = min forward distance to a winning state`` must equal the reverse
        field's distance at the start, i.e. the plan length. Either field being
        wrong shows up here.
      * a press at plan step ``i`` lies on SOME shortest path iff its successor
        is at forward distance ``i + 1`` AND at reverse distance ``d* - i - 1``.
        That set must be exactly the one shipped in the training labels.

    Slower than everything else here (~20s, and level 11's forward BFS is 4.4M
    states against the reverse field's 838k -- the forward search has to walk
    into every deadlock, which is precisely what the reverse one gets to skip),
    so it is opt-in rather than part of `--plans`."""
    _solver, game, expert = _levels()
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(game._engine)
        back = board.field()
        presses, optsets = board.plan(back)

        t = time.time()
        fwd = {board.start: 0}
        queue = deque([board.start])
        while queue:
            state = queue.popleft()
            d = fwd[state] + 1
            for direction in _ORDER:
                nxt = board.step(state, direction)
                if nxt not in fwd:
                    fwd[nxt] = d
                    queue.append(nxt)
        wins = [d for (mask, _p), d in fwd.items() if mask == board.tmask]
        star = min(wins) if wins else None

        notes = []
        if star != len(presses):
            notes.append(f"LENGTH {star} != {len(presses)}")
        state = board.start
        for i, direction in enumerate(presses):
            want = [a for a in _ORDER
                    if fwd.get(board.step(state, a), -1) == i + 1
                    and back.get(board.step(state, a), -1) == star - i - 1]
            if want != optsets[i]:
                notes.append(f"step {i}: {optsets[i]} != {want}")
            state = board.step(state, direction)
        bad += len(notes)
        print(f"level {level:2d}: forward {len(fwd):7d} states in {time.time()-t:5.2f}s, "
              f"d*={star:3d}, {sum(len(s) for s in optsets):3d} labelled presses: "
              f"{'AGREES' if not notes else '; '.join(notes[:3])}")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} disagreements")
    return 0 if not bad else 1


def _fuzz(trials: int = 40, walk: int = 30) -> int:
    """Assert `_Board` reproduces the interpreter exactly.

    Random play from the level START is not enough: on these boards it wanders
    and almost never lines a push up, so the rule the whole game is made of
    would go unexercised. Instead fuzz from EVERY PREFIX of each level's own
    solution, which puts the model in the crate configurations the plans
    actually visit. Coverage counters are printed for the same reason -- a run
    reporting agreement while having pushed nothing has checked nothing.
    """
    _solver, game, expert = _levels()
    rng = random.Random(20260813)
    bad = pushes = walks = blocked = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        layout = game._game.levels[level]
        board = expert.board(eng)
        presses, _ = board.plan(board.field())
        lvl_push = lvl_walk = lvl_block = 0
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
                    crates, player = set(), None
                    for r, row in enumerate(eng.grid):
                        for c, cell in enumerate(row):
                            if cell & expert.crate_ids:
                                crates.add((r, c))
                            if cell & expert.player_ids:
                                player = (r, c)
                    if mine != (frozenset(crates), player):
                        bad += 1
                        if bad < 5:
                            print(f"  level {level} MISMATCH after {direction}: "
                                  f"model {mine} engine {(crates, player)}")
                    if eng.check_win() != (mask == board.tmask):
                        bad += 1
                        print(f"  level {level} WIN MISMATCH")
                    if state == before:
                        lvl_block += 1
                    elif state[0] != before[0]:
                        lvl_push += 1
                    else:
                        lvl_walk += 1
        pushes += lvl_push
        walks += lvl_walk
        blocked += lvl_block
        print(f"level {level:2d}: {lvl_push:6d} pushes  {lvl_walk:6d} walks  "
              f"{lvl_block:6d} refused")
    print(f"{pushes + walks + blocked} transitions "
          f"({pushes} pushes, {walks} walks, {blocked} refused): "
          f"{'model matches the interpreter' if not bad else f'{bad} MISMATCHES'}")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    Two bugs this exists for both shipped in the original art, and both were in
    the STACK rather than in one sprite: Target quantized onto Wall's index, and
    the Player sprite covered the Target's ring exactly, so the frame the win
    condition is about was pixel-identical to the player standing on floor. An
    object-by-object colour check catches neither.

    Whole frames are compared, not cell crops: `_render_frame` upscales the
    board to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and the arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor)."""
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {"floor": (), "wall": ("wall",), "target": ("target",),
             "crate": ("crate",), "crate_on_target": ("target", "crate"),
             "player": ("player",), "player_on_target": ("target", "player")}

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
              f"{','.join(str(x) for x in levels)}): "
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
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(FuturisticBlockPushingSolver.main())
