"""Generate Phase-1 training data for the PuzzleScript game ps:tile_tiler
("Tile Tiler" by Sky Chan).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, the native model of the turn, the exact distance field over the
whole state space, and the optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_tile_tiler",
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
action (post rotation remap), i.e. the button an agent presses in the presented
view, so replaying the recorded actions reproduces the recorded frames exactly.

(Levels are numbered from 1 in this prose, the way the game presents them. The
reports and the corpus's ``level_id`` count from 0, so "level 4" below is the one
``--plans`` prints as L3.)

The game
--------
You are a gardener raking a sand garden, and the whole puzzle is that YOU CANNOT
WALK OVER YOUR OWN WORK. Four arrow keys, no ACTION, one Player per region, and
three rules that all key off ONE thing -- what is in the cell you are stepping
into:

  * **Bare ground** (no tile, no wall): you step in and a fresh ``UnsteppedTile``
    is laid under you. This is the only way a tile is ever created, and it is
    what the win is made of.
  * **An UnsteppedTile**: you step in and it is SCUFFED -- it becomes
    ``HSteppedTile`` if you walked in sideways, ``VSteppedTile`` if you walked in
    up or down. (The two are mechanically identical; only the art differs, which
    matters for rendering and for nothing else.)
  * **A stepped tile**: the press is CANCELLED -- you do not move -- and the tile
    is smoothed back to ``UnsteppedTile``. This is the game's only repair move,
    it costs a whole turn, and it can only be aimed at a cell you are standing
    NEXT to.
  * **A wall** stops you dead, changing nothing.

The win is ``all Background on (Wall or UnsteppedTile)`` plus ``no VSteppedTile``
plus ``no HSteppedTile``. Background is in every cell, so read together that is:
**every non-wall cell holds an UnsteppedTile, including the one you are standing
on.** Note what that makes of the square you START on -- it holds no tile at all,
so you have to leave and come back to it, and the level is never a plain
"visit everything once" walk.

The shape of the whole thing follows from one observation, which is also the
reason the search space is as small as it is: *once no bare ground is left, the
tile under the Player is stepped forever.* A repair leaves your own cell alone,
and a move scuffs the cell you land on -- so with nothing left to lay a fresh
tile on, no press can ever put an Unstepped tile under you again. Every win
therefore has the same last act: step onto the LAST piece of bare ground (which
lays your own clean tile), then spend one press per leftover stepped tile,
smoothing each of the neighbours you can reach without moving. If a scuffed tile
is not adjacent to that final square, the board is already lost and the field
says so -- which is why between 6.6% and 48.7% of the states each level's field
covers are dead, and why the epsilon detours below have to be guarded.

Model + search
--------------
`_Region.step_dense` re-implements one turn natively; ``--selfcheck`` drives 72k
random presses through BOTH the interpreter and the model across all four levels
and compares the full board (every cell's tile, the Player cells) and the win
flag after every one. That is the guard that lets the search trust it.

A level's free cells split into connected REGIONS, each holding exactly one
Player, and a press is delivered to all of them at once. Level 4 has two (a 1x5
corridor and a 2x5 room, either side of a solid wall row) and must win BOTH ON
THE SAME PRESS, which is the only thing that makes it harder than its 15 cells
suggest. So a region is enumerated on its own and a level state is the TUPLE of
its regions' states; the distance field lives on the product.

Inside a region the state is (tile per cell, Player cell) and the tiles move
monotonically -- bare ground is laid on once and never returns -- so a cell that
starts bare has three values and a cell that ships with a tile has two. That
mixed-radix code is dense and decodable, so `_Region.enumerate` needs no
dictionary: it numbers the code space, walks one vectorised forward BFS from the
level start for the reachable subset, and comes out with a compact successor
table. The reachable sets are 9779 / 68745 / 3172609 states and, on level 4,
248 x 158924 = 39.4M product states.

The field is then a straight BFS by layers, run as a Bellman sweep in numpy
(`_Board.enumerate` over `_backup`): after k sweeps the array holds "presses to a
win using at most k presses", so at convergence it is the exact distance and
nothing about it is heuristic. The shortest routes are 8, 11, 12 and 14 presses.

Building all four fields costs ~9s and peaks around 450 MB, level 4's 35 sweeps
over 39.4M cells being ~6.5s of it. That is the WHOLE cost of generation: the
engine state after a reset is seed-independent (only the presentation is
augmented), so it is paid once per process and every seed after it is a replay
plus a re-render -- ~0.27s an episode, measured over 100. Budget for it when
sharding with `parallelize_generator`: the field is per-process, not per-seed.

A plan is a descent of the field, a tie set is a lookup, and both answer from ANY
state, which is what makes the epsilon detours labelable.

Optimal-action targets and recovery
-----------------------------------
`optimal_for` reads the LIVE engine state and returns every press that lowers the
exact distance -- true tie sets, not a reordering heuristic. This game needs them
more than most: the opening of every level is a walk over bare ground where the
ORDER the ground is covered is largely free (level 1 is a ring, and going round
it clockwise or anticlockwise are the same 8 presses), so labelling one
arbitrary route as the single right answer would train a coin flip the policy
cannot win.

``epsilon = 0.12`` is on, but this game is NOT reversible -- scuffing a tile you
cannot get back beside is permanent -- so the recorder's detour guard is doing
real work here rather than being a formality: `record_level` only keeps a random
alternative that both changes the board AND leaves the level winnable, and with
an exact field that test is a single array lookup. The step then records the
mistake as the action taken and the recovery as ``optimal``, on top of the
explore-then-RESET prefix that opens every episode.

Rendering
---------
Four sprites redrawn, in the game FILE (see the comment at the top of
data/puzzlescript_games/Tile_Tiler.txt): the Player hid the tile it was standing
on, so ``player on unsteppedtile`` and ``player on vsteppedtile`` -- a won board
and a board with one scuff left -- were PIXEL-IDENTICAL. The Player is now a 3x3
figure with a transparent border and the three tiles carry what distinguishes
them on that border. ``--audit`` is the check, and it compares WHOLE FRAMES
rather than cell crops (`_render_frame` upscales the board to fill 64x64 and
letterboxes it, so an arithmetic cell crop reads the wrong window).

Augmentation
------------
Per (seed, level) the adapter draws a frame rotation (0..3); ``Tile_Tiler`` is
not in `PuzzleScriptAdapter._FLIP_GAMES`, so there is no flip. ``--symmetry``
measures both halves of what that presentation needs. The mechanic: replay each
plan and a 400-press random walk on all three turned copies of the layout, with
HSteppedTile and VSteppedTile relabelled, because they are not scenery -- they
are the record of which AXIS the player walked in, so a quarter turn maps one
onto the other. And the art, which is the constraint the sprite redraw had to
respect: ``rot90`` of the V sprite must BE the H sprite, or a turned view would
show tile glyphs the unturned view never does. It was true of the original art
and it still is.

Usage (run from the repo root):
    python solvers/generate_tile_tiler_training.py --episodes 1600 \
        --out data/training_multi_level/tile_tiler
    python solvers/generate_tile_tiler_training.py --selfcheck
    python solvers/generate_tile_tiler_training.py --plans
    python solvers/generate_tile_tiler_training.py --verify
    python solvers/generate_tile_tiler_training.py --audit
    python solvers/generate_tile_tiler_training.py --symmetry
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import (                     # noqa: E402
    PuzzleScriptAdapter, _render_frame)
from solvers.common.ps_astar import (                           # noqa: E402
    PSAStarSolver, PSExpert, restore, snapshot)

GAME_NAME = "Tile_Tiler"

#: Engine direction -> (dr, dc).
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: The action space the search branches on, and the tie-break order, so a
#: re-derived plan is byte-identical across processes. ``action`` parses (the
#: game does not declare ``noaction``) but no rule reads it; `selfcheck` asserts
#: it is a pure no-op against the interpreter rather than reading that off the
#: rules section.
_DIRS = ("up", "down", "left", "right")

#: Tile values. BARE is "no tile at all", the state every cell of levels 1, 2 and
#: 4 starts in and the only one a press can leave permanently.
BARE, CLEAN, SCUFF = 0, 1, 2

#: Sentinel for "cannot win from here" in the int16 distance field.
_INF = 32767

#: How many dense codes `_Region.enumerate` steps at a time. `_Region.step_dense`
#: holds roughly a dozen int64 temporaries the width of its input, so stepping
#: level 3's 3.17M-state table in one call costs ~300 MB of transient for a 13 MB
#: answer. Chunking caps that at ~100 MB and costs nothing measurable in time --
#: which matters because `parallelize_generator` runs one of these per shard.
_CHUNK = 1 << 20


def _backup(dist, moves, out) -> None:
    """One Bellman backup of the whole distance field, written into ``out``:
    ``out[s] = 1 + min over the four presses of dist[successor of s]``, with
    _INF absorbing and the caller pinning the winning states back to 0.

    ``moves`` is one list of successor columns per direction -- one column per
    region -- so a press re-indexes the field along every axis at once.

    Two things here are about MEMORY, not about the arithmetic, and level 4 is
    why: its field is 39.4M int16 cells and this runs 35 times over it, once per
    process, in a generator `parallelize_generator` may be running eight copies
    of.

      * The gather is one `np.take` per axis rather than one `np.ix_`. They
        compute the same thing, but `np.ix_` hands numpy a broadcast pair of
        index arrays and the advanced-index machinery materialises them at the
        FULL product shape in int64 -- 315 MB against the 79 MB the int16 result
        itself costs.
      * It runs in blocks of leading-axis rows (`_CHUNK` cells at a time), so the
        temporaries are megabytes rather than another two copies of the field.

    ``min(x, _INF - 1) + 1`` is how the press is charged: it costs one everywhere
    and leaves _INF at _INF, which is also what stops the int16 wrapping at the
    sentinel.
    """
    rows = max(1, _CHUNK // max(1, out.size // out.shape[0]))
    for lo in range(0, out.shape[0], rows):
        block = None
        for cols in moves:
            landed = dist[cols[0][lo:lo + rows]]
            for axis, col in enumerate(cols[1:], 1):
                landed = np.take(landed, col, axis=axis)
            if block is None:
                block = landed
            else:
                np.minimum(block, landed, out=block)
        np.minimum(block, _INF - 1, out=block)
        block += 1
        out[lo:lo + rows] = block


# ---------------------------------------------------------------------------
# One connected region: its state code space, its turn, its reachable set
# ---------------------------------------------------------------------------

class _Region:
    """One 4-connected component of a level's free cells, with the single Player
    standing in it, plus the complete set of states it can reach.

    A state is (the tile in each cell, which cell the Player is on). Tiles only
    ever go BARE -> CLEAN -> SCUFF <-> CLEAN, so a cell that starts bare takes
    three values and a cell that ships with a tile takes two: the state packs
    into a MIXED-RADIX integer that is dense (every code is a syntactically
    valid board) and decodable, which is what lets `enumerate` run as a
    vectorised BFS over integers with no dictionary anywhere.

    `succ` is the compact successor table the field is built on; `dense_to_id`
    maps a code back to its compact id and is how a LIVE engine grid is read.
    """

    __slots__ = ("cells", "n", "radix", "stride", "n_codes", "n_dense",
                 "ahead", "win_code", "start_dense",
                 "codes", "succ", "win", "dense_to_id")

    def __init__(self, cells, tiles0, player0: int):
        self.cells = list(cells)
        n = self.n = len(self.cells)
        where = {p: i for i, p in enumerate(self.cells)}

        self.radix = np.array([3 if t == BARE else 2 for t in tiles0],
                              dtype=np.int64)
        self.stride = np.ones(n, dtype=np.int64)
        for i in range(1, n):
            self.stride[i] = self.stride[i - 1] * self.radix[i - 1]
        self.n_codes = int(self.stride[-1] * self.radix[-1])
        self.n_dense = self.n_codes * n

        # ``ahead[d][i]`` is the cell one step along d, or -1 when that is a Wall
        # or off the board -- the interpreter cancels the press, it does not
        # refuse it, so a -1 is a self-loop and not a missing edge. Regions are
        # separated by walls, so a neighbour outside this region is always -1.
        self.ahead = {
            d: np.array([where.get((r + dr, c + dc), -1) for (r, c) in self.cells],
                        dtype=np.int64)
            for d, (dr, dc) in _DELTA.items()}

        self.win_code = self.encode([CLEAN] * n, 0) // n
        self.start_dense = self.encode(tiles0, player0)

        self.codes = None                  # compact id -> dense code
        self.succ = {}                     # direction -> int32[n_states]
        self.win = None                    # bool[n_states]
        self.dense_to_id = None            # int32[n_dense], -1 where unreached

    # -- the code -------------------------------------------------------------
    def _digit(self, i: int, tile: int) -> int:
        """The mixed-radix digit cell ``i`` stores for ``tile``. A cell that
        cannot be BARE stores CLEAN as 0 and SCUFF as 1."""
        return tile if self.radix[i] == 3 else tile - 1

    def encode(self, tiles, player: int) -> int:
        """One board -- a tile per cell plus the Player's cell -- as its dense
        code. The only place a board becomes an integer, so the level start, the
        winning code and every LIVE read off the engine grid agree by
        construction."""
        return int(sum(int(self.stride[i]) * self._digit(i, tiles[i])
                       for i in range(self.n))) * self.n + player

    # -- one turn, vectorised over dense codes --------------------------------
    def step_dense(self, x, direction: str):
        """One press, natively, for a whole array of dense codes at once.

        The three rules of the game, in the order the interpreter resolves them,
        distinguished purely by the tile in the cell ahead: bare ground is
        tiled and entered, a clean tile is scuffed and entered, a scuffed tile
        is smoothed and the press is cancelled."""
        n = self.n
        player = x % n
        code = x // n
        target = self.ahead[direction][player]
        movable = target >= 0
        cell = np.where(movable, target, 0)

        stride, radix = self.stride[cell], self.radix[cell]
        digit = (code // stride) % radix
        tile = digit + (radix == 2)

        # BARE -> CLEAN, CLEAN -> SCUFF, SCUFF -> CLEAN.
        new_tile = np.where(tile == SCUFF, CLEAN, tile + 1)
        new_digit = new_tile - (radix == 2)
        new_code = code + (new_digit - digit) * stride
        # The Player enters unless the press was cancelled by a scuffed tile.
        new_player = np.where(tile == SCUFF, player, cell)
        return np.where(movable, new_code * n + new_player, x)

    # -- the reachable set ----------------------------------------------------
    def enumerate(self) -> int:
        """Number every state reachable from the level start and build the
        compact successor table. Returns the state count.

        A layered forward BFS, one vectorised `step_dense` per (layer,
        direction, `_CHUNK`). The result is CLOSED under every press by
        construction, so any state the agent can reach by any sequence -- plan,
        exploration prefix, epsilon detour, or a RESET back to the start -- is in
        it, and the field below can answer for all of them.

        Winning states are NOT treated as terminal. In a one-region level they
        are (the adapter ends the level there), but level 4 wins only when both
        regions are clean at once, so a region that is momentarily clean has to
        be able to keep playing.
        """
        if self.codes is not None:
            return len(self.codes)

        d2i = np.full(self.n_dense, -1, dtype=np.int32)
        d2i[self.start_dense] = 0
        layers = [np.array([self.start_dense], dtype=np.int64)]
        total = 1
        frontier = layers[0]
        while len(frontier):
            found = np.unique(np.concatenate(
                [self._step_all(frontier, d) for d in _DIRS]))
            fresh = found[d2i[found] < 0]
            if not len(fresh):
                break
            d2i[fresh] = np.arange(total, total + len(fresh), dtype=np.int32)
            total += len(fresh)
            layers.append(fresh)
            frontier = fresh

        self.codes = np.concatenate(layers)
        self.dense_to_id = d2i
        self.succ = {}
        for d in _DIRS:
            row = np.empty(len(self.codes), dtype=np.int32)
            for lo in range(0, len(self.codes), _CHUNK):
                block = self.codes[lo:lo + _CHUNK]
                row[lo:lo + _CHUNK] = d2i[self.step_dense(block, d)]
            self.succ[d] = row
        self.win = (self.codes // self.n == self.win_code)
        return len(self.codes)

    def _step_all(self, x, direction: str):
        """`step_dense` over a whole array, `_CHUNK` codes at a time."""
        if len(x) <= _CHUNK:
            return self.step_dense(x, direction)
        return np.concatenate([self.step_dense(x[lo:lo + _CHUNK], direction)
                               for lo in range(0, len(x), _CHUNK)])

    def id_of_dense(self, code: int) -> int:
        i = int(self.dense_to_id[code])
        if i < 0:                                            # pragma: no cover
            raise AssertionError("a state outside the enumerated closure")
        return i


# ---------------------------------------------------------------------------
# A level: the product of its regions, and the exact distance field on it
# ---------------------------------------------------------------------------

class _Board:
    """A level's regions plus the EXACT distance-to-win field over their product.

    A level state is one compact id per region and the field is an array of that
    shape, so a level with a single region indexes it directly and level 4 -- two
    regions that must be clean on the SAME press -- indexes the 248 x 158924
    product. Nothing here is heuristic: `enumerate` is a BFS by layers spelled as
    a Bellman sweep, so after k sweeps the array holds "presses to a win using at
    most k presses" and at convergence that is the distance itself.

    `plan` is a descent of the field, `optimal` a lookup, and both work from ANY
    state -- which is what lets the recording take real detours and label the
    recovery.
    """

    __slots__ = ("regions", "shape", "start", "dist", "_win", "sweeps", "dead")

    def __init__(self, regions):
        self.regions = list(regions)
        for region in self.regions:
            region.enumerate()
        self.shape = tuple(len(r.codes) for r in self.regions)
        self.start = tuple(r.id_of_dense(r.start_dense) for r in self.regions)
        self.dist = None
        self._win = None
        self.sweeps = 0
        # Counted once, when the field is built. `distance` is called millions of
        # times per episode and goes through `enumerate`'s early return, so
        # recomputing it there would put a 39.4M-cell scan behind every lookup.
        self.dead = 0

    # -- one turn -------------------------------------------------------------
    def step(self, state, direction: str):
        """One press delivered to every region at once."""
        return tuple(int(r.succ[direction][s])
                     for r, s in zip(self.regions, state))

    def won(self, state) -> bool:
        """Every non-wall cell of every region holds a clean tile."""
        return all(bool(r.win[s]) for r, s in zip(self.regions, state))

    # -- the field ------------------------------------------------------------
    def enumerate(self) -> tuple[int, int]:
        """Build the exact distance-to-win field. Returns ``(states, dead)``.

        Sweep k+1 is ``1 + min over the four presses of sweep k``, pinned to 0 on
        the winning states, floored against the previous sweep so it is monotone.
        `_backup` is the sweep; this loop only pins the winning states,
        keeps the result monotone against the previous array, and stops when a
        sweep changes nothing.
        """
        if self.dist is not None:
            return int(np.prod(self.shape)), self.dead

        win = self.regions[0].win
        for region in self.regions[1:]:
            win = win[..., None] & region.win
        self._win = win

        moves = [[r.succ[d] for r in self.regions] for d in _DIRS]
        dist = np.full(self.shape, _INF, dtype=np.int16)
        dist[win] = 0
        best = np.empty(self.shape, dtype=np.int16)
        while True:
            self.sweeps += 1
            _backup(dist, moves, best)
            np.minimum(best, dist, out=best)
            best[win] = 0
            if np.array_equal(best, dist):
                break
            dist, best = best, dist
        self.dist = dist
        self.dead = int((dist >= _INF).sum())
        return int(np.prod(self.shape)), self.dead

    # -- reading the field ----------------------------------------------------
    def distance(self, state) -> int | None:
        """Presses to a win from ``state``, or None if it cannot win."""
        self.enumerate()
        d = int(self.dist[state])
        return None if d >= _INF else d

    def optimal(self, state) -> list:
        """Every press that keeps the game on a SHORTEST route to the win.

        Exact, straight off the field. A press into a wall maps the state to
        itself, whose distance is unchanged rather than one less, so no-ops are
        excluded for free."""
        here = self.distance(state)
        if not here:                       # a win (0) or a dead state (None)
            return []
        return [d for d in _DIRS
                if self.distance(self.step(state, d)) == here - 1]

    def plan(self, state) -> list | None:
        """A SHORTEST press sequence from ``state`` to a win, or None.

        A descent of the field taking the first tied direction in ``_DIRS``
        order, so the plan is a pure function of the board and does not vary
        with the interpreter's hash seed."""
        if self.distance(state) is None:
            return None
        out = []
        while self.distance(state):
            best = self.optimal(state)
            if not best:                                     # pragma: no cover
                raise AssertionError("distance field has no descent")
            out.append(best[0])
            state = self.step(state, best[0])
        return out


def _regions_of(eng, ids) -> list:
    """Split a loaded level into `_Region`s: one per 4-connected component of the
    non-Wall cells, each carrying its own Player.

    Every component of every level here holds exactly one Player, which is
    asserted rather than assumed -- a component with none is frozen (nothing can
    ever lay a tile in it) and one with two would make the per-region state
    incomplete."""
    free = [(r, c) for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row) if not cell & ids["wall"]]
    todo = set(free)
    out = []
    for cell in free:
        if cell not in todo:
            continue
        blob, queue = [], deque([cell])
        todo.discard(cell)
        while queue:
            r, c = queue.popleft()
            blob.append((r, c))
            for dr, dc in _DELTA.values():
                nxt = (r + dr, c + dc)
                if nxt in todo:
                    todo.discard(nxt)
                    queue.append(nxt)
        blob.sort()
        tiles, players = [], []
        for i, (r, c) in enumerate(blob):
            occupied = eng.grid[r][c]
            tiles.append(CLEAN if occupied & ids["clean"] else
                         SCUFF if occupied & ids["scuff"] else BARE)
            if occupied & ids["player"]:
                players.append(i)
        assert len(players) == 1, f"region of {len(blob)} cells has " \
                                  f"{len(players)} players"
        out.append(_Region(blob, tiles, players[0]))
    return out


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class TileTilerExpert(PSExpert):
    """Exact planner over the enumerated state space (see `_Board`).

    `PSExpert` supplies the plan memo, the restore discipline and the level
    scoping; `_search` swaps in the field descent, so the interpreter is only
    ever stepped by the recorder -- which is also what verifies every plan, since
    a level is kept only when the engine reports WIN.
    """

    directions = list(_DIRS)

    #: `_key` is the tile board plus the Player cells, canonical only WITHIN a
    #: level (the walls that complete the state are static per level yet differ
    #: between them, and the compact ids are per-region numberings).
    scope_by_level = True

    def setup(self) -> None:
        g = self.g
        self.ids = {
            "wall": set(g.resolve_object_name("wall")),
            "clean": set(g.resolve_object_name("unsteppedtile")),
            "scuff": (set(g.resolve_object_name("hsteppedtile"))
                      | set(g.resolve_object_name("vsteppedtile"))),
            "player": set(self.game._engine._player_indices),
        }
        self._boards: dict[int | None, _Board] = {}
        self._cur: _Board | None = None

    # -- reading the engine ---------------------------------------------------
    def board(self, eng, level: int | None) -> _Board:
        """The level's `_Board`, built (and enumerated) once and cached.

        The walls are static -- no rule creates or destroys one -- so whichever
        state happens to build it describes the same level -- but the TILES are
        not, and `_Region` reads the board it is handed as the level's START:
        that is where each cell's radix comes from (a cell that ships bare has
        three values, one that ships tiled has two). Building from a mid-game
        grid would therefore build a code space with no room for the bare cells
        that grid has already covered, and the corruption would be silent. Every
        caller does set the level first -- `prepare_expert` builds all four up
        front -- and the assertion below is what keeps that true."""
        board = self._boards.get(level)
        if board is None:
            board = _Board(_regions_of(eng, self.ids))
            board.enumerate()
            assert self.read(eng, board) == board.start, \
                "a board was built from a mid-game grid, not a level start"
            self._boards[level] = board
        return board

    def read(self, eng, board: _Board) -> tuple:
        """The level's state -- one compact id per region -- off the engine grid.

        The read is exact: every non-Wall cell contributes its tile and the
        Player cells are located, so nothing about the board is inferred."""
        ids = self.ids
        out = []
        for region in board.regions:
            tiles = []
            player = -1
            for i, (r, c) in enumerate(region.cells):
                cell = eng.grid[r][c]
                tiles.append(CLEAN if cell & ids["clean"] else
                             SCUFF if cell & ids["scuff"] else BARE)
                if cell & ids["player"]:
                    player = i
            assert player >= 0, "a region lost its Player"
            out.append(region.id_of_dense(region.encode(tiles, player)))
        return tuple(out)

    def state(self, eng, level: int | None) -> tuple:
        return self.read(eng, self.board(eng, level))

    # -- planning -------------------------------------------------------------
    def _key(self, eng):
        assert self._cur is not None, "plan() sets the board before keying"
        return self.read(eng, self._cur)

    def plan(self, eng, level: int | None = None) -> list | None:
        self._cur = self.board(eng, level)
        return super().plan(eng, level)

    def heuristic(self, eng) -> int:
        """The EXACT remaining press count (the base `_astar` is never used here,
        but the contract is that this is 0 at a win and admissible, and the field
        is both)."""
        d = self._cur.distance(self.read(eng, self._cur))
        return 0 if d is None else d

    def _search(self, eng) -> list | None:
        return self._cur.plan(self.read(eng, self._cur))

    def optimal_dirs(self, eng, level: int | None) -> list:
        """Every press on a shortest route, from the engine's CURRENT state."""
        board = self.board(eng, level)
        return board.optimal(self.read(eng, board))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class TileTilerSolver(PSAStarSolver):
    game_id = "puzzlescript_tile_tiler"
    game_name = GAME_NAME
    expert_cls = TileTilerExpert

    #: Unused -- `TileTilerExpert._search` never calls the base A* -- but left at
    #: the family default so a future subclass that does is not silently starved.
    node_cap = 2_000_000
    weight = 1
    #: The adapter GAME_OVERs at 200 actions per level; the longest plan here is
    #: 14 presses, leaving essentially the whole budget to the exploration prefix
    #: and the detours.
    max_steps = 200

    #: Non-zero even though this game is IRREVERSIBLE, which is the opposite of
    #: the usual reasoning in this family. Scuffing a tile you cannot get back
    #: beside is permanent, and between 6.6% and 48.7% of the states each level's
    #: field covers are dead -- but `record_level`'s detour guard only keeps a random
    #: alternative that both changes the board AND leaves the level winnable, and
    #: with an exact field that test is one array lookup rather than a re-solve.
    #: So the detours are free and cannot strand the recording: the taken action
    #: is the mistake and ``optimal`` is the recovery, which is the signal a
    #: policy needs after its own error. The RESET prefix still runs in front of
    #: it.
    epsilon = 0.12

    def prepare_expert(self, game, expert) -> None:
        """Build every level's field up front.

        Pure front-loading -- the field does not depend on which state builds it
        -- but it means level 4's 39.4M-state sweep is paid once, at startup,
        rather than inside the first query that needs it."""
        for level in range(game.n_levels):
            game.set_level(level)
            expert.board(game._engine, level)

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """Every press that keeps the game on a SHORTEST route to the win, read
        off the LIVE engine state.

        `record_level` asks for this before executing the press, so the engine is
        still at the state being labelled -- which is what makes the label right
        after an epsilon detour too, where the state is off the plan entirely and
        replaying the plan from the level start would label the wrong states.

        Falls back to the press about to be taken if the field has nothing to say
        -- no expert step may ship unlabelled."""
        best = expert.optimal_dirs(expert.game._engine, level)
        if best:
            return best
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _levels():
    """``(solver, game, expert)`` with every level's field built."""
    solver = TileTilerSolver()
    game, expert, _solvable = solver._ensure(0)
    return solver, game, expert


def selfcheck(trials: int = 300, steps: int = 60, verbose: bool = True) -> int:
    """Drive random presses through BOTH the interpreter and `_Region.step_dense`,
    comparing the WHOLE board (every cell's tile plus the Player cells, via the
    state id) and the win flag after every one. This is the guard that lets the
    planner trust the native model.

    ``action`` is included in the press alphabet even though the search excludes
    it: the claim that it is a pure no-op is exactly the kind of thing that
    should be measured against the interpreter rather than read off a rules
    section.

    Returns the number of mismatches."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    expert = TileTilerExpert(game)
    alphabet = _DIRS + ("action",)
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = expert.board(eng, level)
        rng = random.Random(f"tile_tiler:selfcheck:{level}")
        bad = presses = 0
        for _ in range(trials):
            game.set_level(level)
            state = expert.read(eng, board)
            if state != board.start:
                bad += 1
                break
            for _ in range(steps):
                d = rng.choice(alphabet)
                want = state if d == "action" else board.step(state, d)
                eng.step(d)
                presses += 1
                if (expert.read(eng, board) != want
                        or board.won(want) != eng.check_win()):
                    bad += 1
                    break
                state = want
                if eng.check_win():
                    break
        total += bad
        if verbose:
            print(f"  L{level}: {'OK' if not bad else f'{bad} MISMATCHES'} "
                  f"({presses} presses vs the interpreter)")
    return total


def _walk(board: _Board, plan) -> list:
    """The states ``plan`` passes through, starting at the level start."""
    out, state = [], board.start
    for d in plan:
        out.append(state)
        state = board.step(state, d)
    return out


def _report() -> int:
    """Print each level's space, its shortest plan and the engine's verdict --
    the quick "is this game still fully solved" check."""
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        n_states, n_dead = board.enumerate()
        plan = board.plan(board.start)
        if plan is None:
            bad += 1
            print(f"  L{level}: UNSOLVED")
            continue
        for d in plan:
            eng.step(d)
        won = eng.check_win()                   # read BEFORE anything reloads
        ties = sum(len(board.optimal(s)) for s in _walk(board, plan))
        regions = "x".join(str(len(r.codes)) for r in board.regions)
        cells = "+".join(str(r.n) for r in board.regions)
        print(f"  L{level}: {eng.height}x{eng.width}  {cells} free cells  "
              f"{len(plan):3d} presses  win={won}  "
              f"{regions} = {n_states} states "
              f"({100.0 * n_dead / n_states:.1f}% dead, "
              f"{board.sweeps} sweeps)  "
              f"{ties / len(plan):.2f} optimal presses/step")
        if not won:
            bad += 1
    print("plans clean -- every level still wins on the interpreter" if not bad
          else f"PLANS FAILED: {bad} levels")
    return 1 if bad else 0


def _verify(sample_size: int = 200_000) -> int:
    """Three checks of the distance field and of every tie set it ships.

    The field is built by ONE routine (`_Board.enumerate`'s Bellman sweep) and
    the training labels ARE its tie sets, so "the plan wins" is nowhere near
    enough of a check -- a field that is uniformly one too large still descends
    to a win, and still hands out tie sets that are quietly wrong. Three passes,
    which fail differently:

      * **The Bellman equation, checked one state at a time.** Take every state
        where the level has few enough of them and a random sample where it does
        not, and re-derive ``d(s) = 1 + min d(succ(s))`` through the SCALAR
        path -- `_Board.step` and `_Board.distance`, plain per-region lookups --
        rather than through `_backup`'s blocked, transposed, multi-axis gather.
        That asymmetry is the point: the vectorised backup is where an indexing
        bug would hide, and it would hide well, because a field that is wrong by
        a consistent permutation is still a self-consistent fixpoint of ITSELF.
        Also checks, over the whole array at once, that ``d == 0`` holds at
        exactly the winning states.
      * **Independent forward BFS.** Walk FORWARD from the level start, layer by
        layer, and report the layer at which a winning state first appears. It
        shares no code path with the backward sweep -- different direction,
        different termination, plain Python -- and it is the value the corpus
        depends on, so the two must agree on ``d*``.
      * **Executable labels.** Walk each level's plan on the INTERPRETER and, at
        every step, actually press each direction the label calls optimal: the
        board the engine lands on must be the native successor, and its field
        distance must be exactly one less. That takes the labels out of the model
        and puts them through the real thing.
    """
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)

        # -- pass 1: the Bellman equation, one sampled state at a time ---------
        if not np.array_equal(board.dist == 0, board._win):
            bad += 1
            print(f"  L{level}: distance 0 is not exactly the winning states")
        total = int(np.prod(board.shape))
        if total <= sample_size:
            sample = list(itertools.product(*(range(n) for n in board.shape)))
            how = "exhaustive"
        else:
            rng = random.Random(f"tile_tiler:verify:{level}")
            sample = [board.start] + [
                tuple(rng.randrange(n) for n in board.shape)
                for _ in range(sample_size)]
            how = "sampled"
        wrong = 0
        for state in sample:
            here = board.distance(state)
            if board.won(state):
                if here != 0:
                    wrong += 1
                continue
            onward = [board.distance(board.step(state, d)) for d in _DIRS]
            live = [x for x in onward if x is not None]
            want = 1 + min(live) if live else None
            if want != here:
                wrong += 1
        if wrong:
            bad += 1
            print(f"  L{level}: {wrong} of {len(sample)} states break the "
                  f"Bellman equation")

        # -- pass 2: an independent forward BFS from the level start -----------
        seen = {board.start}
        frontier = [board.start]
        forward = None
        depth = 0
        while frontier and forward is None:
            depth += 1
            nxt = []
            for state in frontier:
                for d in _DIRS:
                    landed = board.step(state, d)
                    if landed in seen:
                        continue
                    if board.won(landed):
                        forward = depth
                        break
                    seen.add(landed)
                    nxt.append(landed)
                if forward is not None:
                    break
            frontier = nxt
        if forward != board.distance(board.start):
            bad += 1
            print(f"  L{level}: the forward BFS reaches a win in {forward} "
                  f"presses, the field says {board.distance(board.start)}")

        # -- pass 3: every shipped label, pressed on the interpreter -----------
        plan = board.plan(board.start)
        state = board.start
        labels = 0
        for i, press in enumerate(plan):
            here = board.distance(state)
            best_dirs = board.optimal(state)
            if press not in best_dirs:
                bad += 1
                print(f"  L{level}: step {i} is not in its own optimal set")
            before = snapshot(eng)
            for alt in best_dirs:
                eng.step(alt)
                landed = expert.read(eng, board)
                labels += 1
                if (landed != board.step(state, alt)
                        or board.distance(landed) != here - 1):
                    bad += 1
                    print(f"  L{level}: step {i} labels {alt} optimal, but the "
                          f"interpreter does not land one press closer")
                restore(eng, before)
            eng.step(press)
            state = board.step(state, press)
        if not eng.check_win():
            bad += 1
            print(f"  L{level}: the interpreter does not report a win")
        print(f"  L{level}: {len(sample)}/{total} states relaxed scalar-wise "
              f"({how}), "
              f"forward BFS d*={forward} in {len(seen)} states, "
              f"{labels} labels pressed on the interpreter -- "
              f"{'OK' if not bad else 'see above'}")
    print(f"verify: {bad} problems")
    return 1 if bad else 0


def _turn(k: int):
    """``(cell_fn, dims_fn, direction_map)`` for ``k`` clockwise quarter turns --
    the group the adapter's frame rotation presents this game under.

    The direction map is DERIVED from the same linear part that moves the cells,
    so the two cannot drift apart."""
    def cell(rc, hw):
        r, c = rc
        h, w = hw
        for _ in range(k):
            r, c, h, w = c, h - 1 - r, w, h
        return (r, c)

    def dims(hw):
        h, w = hw
        return (w, h) if k % 2 else (h, w)

    dmap = {}
    for name, (dr, dc) in _DELTA.items():
        for _ in range(k):
            dr, dc = dc, -dr
        dmap[name] = next(n for n, d in _DELTA.items() if d == (dr, dc))
    return cell, dims, dmap


def _turn_layout(layout, k: int, swap: dict):
    """``layout`` (rows of object-index sets) rotated ``k`` quarter turns, with
    HSteppedTile and VSteppedTile RELABELLED through ``swap`` on the odd turns.

    That relabel is the whole point: the two stepped tiles are not scenery, they
    are the record of WHICH AXIS the player walked in, so a quarter turn maps one
    onto the other. A rotation check that skipped it would be checking a
    different game."""
    hw = (len(layout), len(layout[0]))
    cell, dims, _ = _turn(k)
    th, tw = dims(hw)
    out = [[set() for _ in range(tw)] for _ in range(th)]
    for r, row in enumerate(layout):
        for c, objs in enumerate(row):
            tr, tc = cell((r, c), hw)
            out[tr][tc] = {swap.get(o, o) if k % 2 else o for o in objs}
    return out


def _symmetry(walk: int = 400) -> int:
    """Prove ON THE INTERPRETER that the quarter-turn group is an exact symmetry
    of this game, in both of the ways the adapter's rotation augmentation relies
    on.

      * **The mechanic.** Rebuild each level's LAYOUT turned (with H/V swapped on
        the odd turns), reload it, and replay the level's own plan plus a seeded
        random walk with the presses turned to match. The whole board -- tiles
        and Player -- must land where the turn says after every press. The plan
        is the sequence the corpus actually records; the random walk is there
        because a shortest plan never presses into a wall, which is the one thing
        the interpreter has to decide here.
      * **The tile art**, on the parsed 5x5 sprite matrices in ARC palette
        indices: ``rot90`` of the V sprite must BE the H sprite, and back, and U
        must be turn-invariant. This is the part that had to survive the sprite
        redraw. A screen-horizontal press under a quarter turn is an engine
        VERTICAL press and lays a VSteppedTile, which the turned frame draws
        turned; unless the glyphs map onto each other, a turned presentation
        would show the agent tile pictures the unturned one never does, and the
        three tiles would look like six.

    Two things are deliberately NOT claimed. The Wall and Player sprites are not
    turn-invariant, so a whole rendered LEVEL is not equal to the turn of its own
    frame -- which is fine and nothing depends on it: the adapter turns the
    entire picture, so those two simply appear turned, as in any rotated
    photograph. And the FRAME-level equality would not hold even for a tile-only
    board, because `_render_frame` upscales by a non-integer NEAREST factor and
    floors the letterbox pad (``(64 - rh) // 2``), so a 45x64 play area is padded
    9 rows above and 10 below and its turn is padded 9 columns left and 10 right
    -- a parity artefact of the presentation, not of this game. Measuring the
    sprites avoids inheriting it.
    """
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    swap = {idx["hsteppedtile"]: idx["vsteppedtile"],
            idx["vsteppedtile"]: idx["hsteppedtile"]}
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        layout = g.levels[level]
        hw = (len(layout), len(layout[0]))
        rng = random.Random(f"tile_tiler:symmetry:{level}")
        runs = {"plan": board.plan(board.start),
                "walk": [rng.choice(_DIRS) for _ in range(walk)]}

        notes = []
        for kind, presses in runs.items():
            eng.load_level(layout)
            ref = []
            for d in presses:
                eng.step(d)
                ref.append([[set(cell) for cell in row] for row in eng.grid])
            for k in (1, 2, 3):
                dmap = _turn(k)[2]
                eng.load_level(_turn_layout(layout, k, swap))
                for i, d in enumerate(presses):
                    eng.step(dmap[d])
                    want = _turn_layout(ref[i], k, swap)
                    if want != [[set(c) for c in row] for row in eng.grid]:
                        notes.append(f"{kind}:rot{k}@press{i}")
                        break

        bad += len(notes)
        print(f"  L{level}: {hw[0]}x{hw[1]}, "
              f"({len(runs['plan'])} plan + {walk} random) presses x 3 turns: "
              f"{'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:3])}")

    # -- the tile art, on the sprite matrices themselves -----------------------
    art = {}
    for name in ("unsteppedtile", "hsteppedtile", "vsteppedtile"):
        obj = g.objects[name]
        art[name] = np.array([[obj.colors[ci] if 0 <= ci < len(obj.colors)
                               else -1 for ci in row] for row in obj.sprite],
                             dtype=np.int16)
    for src, want in (("vsteppedtile", "hsteppedtile"),
                      ("hsteppedtile", "vsteppedtile"),
                      ("unsteppedtile", "unsteppedtile")):
        turned = np.rot90(art[src], -1)
        if np.array_equal(turned, art[want]):
            print(f"  art: rot90({src}) == {want}")
        else:
            bad += 1
            print(f"  art: rot90({src}) != {want} "
                  f"({int((turned != art[want]).sum())} of 25 sprite pixels)")

    print("symmetry clean -- the rotation augmentation is exact" if not bad
          else f"SYMMETRY FAILED: {bad} broken presentations")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION the game can show is distinct in the frame,
    at every cell size the four levels render at.

    The pair this exists for is ``player+unsteppedtile`` against
    ``player+vsteppedtile``: the win condition is "every free cell holds an
    UnsteppedTile and no stepped tile survives", the last cell to be resolved is
    always the one under the Player, and the shipped Player painted over exactly
    the pixels that told those two apart -- so a won board and a board with one
    scuff left were pixel-identical. See the header of
    data/puzzlescript_games/Tile_Tiler.txt.

    Whole frames are compared, not cell crops: `_render_frame` upscales the board
    to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor)."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    # Player and Wall share a collision layer, so player+wall is not
    # representable; every other stack of one tile and an optional Player is.
    comps = [(), ("wall",), ("unsteppedtile",), ("hsteppedtile",),
             ("vsteppedtile",), ("player",), ("player", "unsteppedtile"),
             ("player", "hsteppedtile"), ("player", "vsteppedtile")]
    clashes_total = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for objs in comps:
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[objs] = np.asarray(_render_frame(eng, g)).copy()
        pairs = [(a, b, int((shots[a] != shots[b]).sum()))
                 for a, b in itertools.combinations(shots, 2)]
        clashes = [(a, b) for a, b, n in pairs if not n]
        clashes_total += len(clashes)
        name = lambda t: "+".join(t) if t else "floor"           # noqa: E731
        print(f"{h}x{w} board (level{'s' if len(levels) > 1 else ''} "
              f"{', '.join(str(x) for x in levels)}), "
              f"cell {max(1, min(64 // h, 64 // w))}px")
        for objs in comps:
            print(f"  {name(objs):24s} "
                  f"{int((shots[objs] != shots[()]).sum()):5d} px differ from "
                  f"bare floor")
        a, b, n = min(pairs, key=lambda t: t[2])
        print(f"  closest pair: {name(a)} vs {name(b)}, {n} px apart")
        for a, b in clashes:
            print(f"  IDENTICAL: {name(a)} == {name(b)}")
    print("audit clean" if not clashes_total
          else f"AUDIT FAILED: {clashes_total} indistinguishable pairs")
    return 0 if not clashes_total else 1


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        bad = selfcheck()
        print(f"selfcheck: {bad} mismatches")
        sys.exit(1 if bad else _report())
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    sys.exit(TileTilerSolver.main())
