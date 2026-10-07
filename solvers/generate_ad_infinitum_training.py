r"""Generate Phase-1 training data for the PuzzleScript game
ps:ad_infinitum_v0_4_2_implemented_highfloor_finished.

The harness -- the trajectory recorder, the rotation contract and the BaseSolver
plumbing -- lives in `solvers/common/ps_astar.py`, shared with the other ps:
generators. This file is the game-specific part: a pure-Python MODEL of the
mechanic, a macro A* over that model, and the escalating-weight ladder that
decides how hard to try per level.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_ad_infinitum",
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
action (post rotation-remap), i.e. the button an agent presses in the rotated
view, so replaying the recorded actions reproduces the recorded frames exactly.

The game
--------
"You have a bunch of mirrors. Create infinities." -- a sokoban-shaped puzzle by
Tom Hermans. The Player walks the grid and PUSHES mirrors; a push moves the
whole contiguous chain of mirrors ahead of it if the cell past the chain is
free. Walls block. HighFloor (the grey inlay in levels 10 and 18) is the one
asymmetry: the player may walk onto it, mirrors may not be pushed onto it.

Every turn the engine re-derives the lasers from scratch. Two mirrors in the
same column shoot at each other, and so do two in the same row -- through walls,
at any distance, because the firing rules are ``up [Shootable | ... | Shootable]``
with an unconstrained ellipsis. So each mirror gets four bits: is there another
mirror somewhere above / below / left / right of it, in its own column / row.
A mirror is SATISFIED (becomes a gold VictoryMirror and stays lit) when those
bits are exactly the two its diagonal reflects between, or all four:

    ``/``  needs (up, left)  or (down, right)  or all four
    ``\``  needs (up, right) or (down, left)   or all four

Anything else -- one bit, three bits, a mirror alone in its row -- leaves it a
plain white NormalMirror. The beams are then drawn outward from the satisfied
mirrors only, each running until it hits the next mirror.

The three win conditions are ``No BulbOff`` (vacuous -- no level ships a bulb),
``No NormalMirror`` (every mirror satisfied) and ``All Player on Laser`` (the
player standing in a beam). So the puzzle is: shove every mirror into a closed
circuit of beams, then step into one. The corollary the search leans on is that
every row and column holding a mirror must hold at least two of them.

The expert: a model, not the interpreter
----------------------------------------
The family default is to search the real interpreter (`PSExpert`), and its macro
variant `PSPushExpert` is exactly the right SHAPE for this game. It is far too
slow for it: an interpreter step costs ~1.6 ms here (the laser rules rebuild
every beam on the board every turn), the hard levels expand up to ~800k nodes,
and expanding ONE node means trying every push on the board -- ~30 of them, each
of which `PSPushExpert` would have to walk through the interpreter step by step.
That is days per level, not minutes.

So the mechanic is re-implemented in `Board` -- push chains, the HighFloor
exclusion, the four-bit satisfaction test and the beam trace. It expands ~4k
nodes/s, i.e. ~120k engine-equivalent steps/s against the interpreter's ~600.
The model is not trusted on faith:

  * it was differentially fuzzed against the interpreter -- random walks on all
    28 levels, plus ~4000 hand-planted mirror layouts (random, random-rectangle
    and 3x3 all-four-bits configurations, 56 of them genuine engine wins) --
    with zero disagreements on either the resulting grid or ``check_win``;
  * and every plan this file emits is still executed against the interpreter by
    `ps_astar.record_level`, which only keeps the level if the ENGINE reports
    `GameState.WIN`. A model that drifted would lose seeds, never fake them.

`AdInfinitumExpert.plan` therefore searches the model and returns primitive
directions, so everything downstream (`record_level`, the plan cache, the
epsilon detour) is unchanged.

Macro A*
--------
Successors are ``walk to the push cell, then push`` macros, so search depth is
the number of PUSHES rather than the number of key presses, and each walk is a
BFS shortest path instead of something the search has to rediscover. ``g``
counts primitive MOVES, so the cost metric stays the one the agent pays.

States are deduped by (mirror layout, player REACHABLE REGION) -- the standard
sokoban canonicalisation. Two states with the same mirrors and the same player
region admit exactly the same futures, so merging them cannot lose a solution.

The goal test is not "the layout is right": the player must also be standing in
a beam. At every node whose layout is fully satisfied the search takes the
shortest walk to a reachable beam cell and stops -- the win fires the moment the
player steps in.

The weight ladder
-----------------
``heuristic`` is the number of unsatisfied mirrors. It is informative but NOT
admissible (one push that completes a rectangle satisfies four mirrors at once),
so w=1 is "as short as this heuristic gets", not a proof of optimality. Levels
that do not fall to it are retried at increasing weight, which trades length for
reach; each rung has a NODE budget (not a time limit -- which levels solve must
not depend on how busy the machine is), and a rung that EXHAUSTS its frontier
ends the ladder early: an empty frontier is a proof of unsolvability, not a
budget failure.

Levels solved, at the first rung that reaches them (plan lengths in primitive
moves):

    w=1   0(11) 1(27) 3(16) 4(46) 5(50) 6(15) 7(3) 8(10) 9(10) 10(25) 11(3)
          12(3) 15(25) 16(27) 17(23) 18(31) 19(3) 20(2) 21(2) 24(33) 25(1)
    w=4   2(50) 13(43) 14(57) 23(42)

25 of the 28 levels. The one-to-three-move levels are not a bug: levels 7, 11,
12, 19, 20, 21 and 25 are the author's shipped sketches, whose mirrors are
already all satisfied, so the whole puzzle is stepping into the nearest beam.

Level 22 is UNSOLVABLE and proved so: its frontier empties in well under a
second, and the ladder stops there rather than re-running the same dead search
at every weight. Levels 26 and 27 are the author's unlabelled "(Testing)"
scratch boards (16 and 24 mirrors); nothing found a win for them, greedy w=40
included, so they are in ``skip_levels`` rather than burning the ladder at every
startup.

The ladder costs ~12 minutes at the first `_ensure` -- level 14 alone is ~5 of
them -- and it is paid ONCE, not once per run: the results go to `PLAN_CACHE` on
disk, and within a process the engine state after reset is seed-independent, so
every later seed replays from the in-memory plan cache. Delete
``data/ad_infinitum_plans.json`` to force a fresh search.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), and this game is not in `PuzzleScriptAdapter._RECOLOR_GAMES` or
``_FLIP_GAMES``, so the only presentation variable is the frame rotation
(rotation_k in {0,1,2,3}) with its matching directional action remap.

Four presentations would make thin data for a cached, seed-independent plan, so
the per-seed variety comes from the trajectory instead:

  * the RESET-recovery prefix (`BaseSolver._reset_prefix`) opens every episode
    with an epsilon-decayed random flail, which for this game is real
    exploration -- pushes are not undoable, so the prefix genuinely wrecks the
    board before the RESET restores it; and
  * the cached plan is stored as MACROS, not key presses, and re-materialised
    per seed: each walk is drawn uniformly from that state's shortest-path tree
    (`Board.region` shuffles its BFS order from the seed's RNG) and the final
    beam cell is drawn from the nearest ties. Same pushes, same length, a
    different route between them on every seed.

Usage (run from the repo root):
    python solvers/generate_ad_infinitum_training.py --episodes 200 \
        --out data/training_multi_level/puzzlescript_ad_infinitum
"""

from __future__ import annotations

import heapq
import json
import os
import random
import sys
from collections import deque
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solvers.common.ps_astar import PSAStarSolver, PSExpert   # noqa: E402

GAME_NAME = "Ad_Infinitum_v0.4.2_[Implemented_HighFloor,_finished]"

#: Engine direction -> (dr, dc). The game declares ``noaction`` in its prelude,
#: so ACTION5 is dead and the four moves are the whole input space.
_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}
_MOVES: tuple[tuple[str, int, int], ...] = tuple(
    (d, dr, dc) for d, (dr, dc) in _DELTA.items())
_OPPOSITE: dict[str, str] = {"up": "down", "down": "up",
                             "left": "right", "right": "left"}

#: Mirror types, as stored in a layout dict.
_SLASH = 0        # ``/`` -- Mirror1 / VictoryMirror1
_BACKSLASH = 1    # ``\`` -- Mirror2 / VictoryMirror2

#: (up, down, left, right) bit patterns that satisfy a mirror. The all-four case
#: satisfies both types; the two-bit cases are the pair the diagonal reflects
#: between. See the module docstring.
_ALL_FOUR = (True, True, True, True)
_SATISFIED: dict[int, frozenset] = {
    _SLASH: frozenset({_ALL_FOUR,
                       (True, False, True, False),      # up + left
                       (False, True, False, True)}),    # down + right
    _BACKSLASH: frozenset({_ALL_FOUR,
                           (True, False, False, True),  # up + right
                           (False, True, True, False)}),  # down + left
}

#: Beam directions a satisfied mirror emits: exactly the bits that satisfied it.
_BEAM_DIRS: dict[tuple, tuple[tuple[str, int, int], ...]] = {
    _ALL_FOUR: _MOVES,
    (True, False, True, False): (("up", -1, 0), ("left", 0, -1)),
    (False, True, False, True): (("down", 1, 0), ("right", 0, 1)),
    (True, False, False, True): (("up", -1, 0), ("right", 0, 1)),
    (False, True, True, False): (("down", 1, 0), ("left", 0, -1)),
}


# ---------------------------------------------------------------------------
# The mechanic, re-implemented (see "The expert: a model, not the interpreter")
# ---------------------------------------------------------------------------

class Board:
    """One level's static geometry plus the pure-Python mechanic.

    Walls and HighFloor never move (the ``WallTop`` / ``WallBottom`` cosmetic
    rules swap one wall object for another but every one of them is a ``Wall``),
    so the static layer can be read off ANY grid of the level, mid-game
    included. The dynamic layer -- player cell and mirror layout -- is read
    fresh with `read`.
    """

    def __init__(self, eng, wall_ids: set, high_id: int,
                 slash_ids: set, backslash_ids: set, player_ids: set):
        self.h, self.w = len(eng.grid), len(eng.grid[0])
        self._slash, self._backslash = slash_ids, backslash_ids
        self._player = player_ids
        self.walls: set = set()
        self.high: set = set()
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & wall_ids:
                    self.walls.add((r, c))
                if high_id in cell:
                    self.high.add((r, c))

    # -- state ---------------------------------------------------------------
    def read(self, eng) -> tuple:
        """``(player cell, {cell: type})`` from the engine's current grid."""
        player = None
        mirrors: dict = {}
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                if cell & self._slash:
                    mirrors[(r, c)] = _SLASH
                elif cell & self._backslash:
                    mirrors[(r, c)] = _BACKSLASH
                if cell & self._player:
                    player = (r, c)
        return player, mirrors

    def inside(self, cell) -> bool:
        return 0 <= cell[0] < self.h and 0 <= cell[1] < self.w

    # -- movement ------------------------------------------------------------
    def region(self, mirrors: dict, player: tuple,
               rng: random.Random | None = None) -> dict:
        """Parent-pointer BFS over the cells the player can walk to.

        Mirrors block: walking into one is a PUSH, which is a macro, not a walk.
        HighFloor does not block -- the player may stand on it, only mirrors may
        not be pushed onto it.

        ``rng`` shuffles the neighbour order, which re-roots the shortest-path
        tree: every walk stays shortest, but WHICH shortest walk comes out
        varies per seed. That is the per-seed trajectory variety (see the module
        docstring); leave it None inside the search, where only lengths matter.
        """
        moves = list(_MOVES)
        parent: dict = {player: None}
        queue = deque((player,))
        while queue:
            cur = queue.popleft()
            r, c = cur
            if rng is not None:
                rng.shuffle(moves)
            for d, dr, dc in moves:
                nxt = (r + dr, c + dc)
                if nxt in parent or not self.inside(nxt):
                    continue
                if nxt in self.walls or nxt in mirrors:
                    continue
                parent[nxt] = (cur, d)
                queue.append(nxt)
        return parent

    @staticmethod
    def walk(parent: dict, target: tuple) -> list:
        """The directions from the BFS root to ``target``."""
        out: list = []
        cur = target
        while parent[cur] is not None:
            cur, d = parent[cur]
            out.append(d)
        out.reverse()
        return out

    @staticmethod
    def walk_len(parent: dict, target: tuple) -> int:
        """``len(walk(parent, target))`` without materialising the list."""
        n = 0
        cur = target
        while parent[cur] is not None:
            cur = parent[cur][0]
            n += 1
        return n

    def push(self, mirrors: dict, cell: tuple, d: str):
        """Apply the push of the chain starting at ``cell`` in direction ``d``.

        Returns the new layout, or None if the chain cannot move. The player
        ends on ``cell``; the caller is responsible for having walked it to
        ``cell - delta`` first.
        """
        dr, dc = _DELTA[d]
        chain = []
        r, c = cell
        while (r, c) in mirrors:
            chain.append((r, c))
            r += dr
            c += dc
        if not self.inside((r, c)) or (r, c) in self.walls or (r, c) in self.high:
            return None
        moved = dict(mirrors)
        for pos in reversed(chain):
            moved[(pos[0] + dr, pos[1] + dc)] = moved.pop(pos)
        return moved

    # -- lasers --------------------------------------------------------------
    @staticmethod
    def _bits(mirrors: dict) -> dict:
        """cell -> (up, down, left, right): is there another mirror that way.

        Distance and walls are irrelevant -- the firing rules pair ANY two
        mirrors sharing a row or column (see the module docstring) -- so the
        extremes of each row and column are all this needs.
        """
        rows: dict = {}
        cols: dict = {}
        for r, c in mirrors:
            if r in rows:
                lo, hi = rows[r]
                rows[r] = (min(lo, c), max(hi, c))
            else:
                rows[r] = (c, c)
            if c in cols:
                lo, hi = cols[c]
                cols[c] = (min(lo, r), max(hi, r))
            else:
                cols[c] = (r, r)
        return {(r, c): (cols[c][0] < r, cols[c][1] > r,
                         rows[r][0] < c, rows[r][1] > c)
                for r, c in mirrors}

    def unsatisfied(self, mirrors: dict) -> int:
        """How many mirrors are still plain white NormalMirrors."""
        bits = self._bits(mirrors)
        return sum(1 for cell, t in mirrors.items()
                   if bits[cell] not in _SATISFIED[t])

    def beams(self, mirrors: dict):
        """Every cell lit by a beam, or None if some mirror is unsatisfied.

        Only satisfied mirrors emit (the engine deletes every laser not sitting
        on a VictoryMirror before propagating), and a beam runs until the next
        mirror -- through walls, which stop nothing.
        """
        bits = self._bits(mirrors)
        lit: set = set()
        for cell, t in mirrors.items():
            b = bits[cell]
            if b not in _SATISFIED[t]:
                return None
            for _d, dr, dc in _BEAM_DIRS[b]:
                r, c = cell
                while True:
                    lit.add((r, c))
                    nxt = (r + dr, c + dc)
                    if not self.inside(nxt) or nxt in mirrors:
                        break
                    r, c = nxt
        return lit


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

#: Where the ladder's results are kept between runs, in the repo's usual
#: ``data/<game>_plans.json`` shape (see `generate_lf52_training.build_plans`).
#: The plans are seed-independent, so this turns the one-off search cost into a
#: one-off-EVER cost -- which matters most for `parallelize_generator.py`, where
#: every shard would otherwise redo it. Each entry records the start layout it
#: was solved from and is ignored if that no longer matches, so editing the
#: level in the .txt cannot silently serve a stale plan.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "ad_infinitum_plans.json"

#: ``(weight, expanded-node budget)`` rungs, in order. See "The weight ladder".
#: The budget counts EXPANDED nodes, not seconds, so which levels solve -- and
#: which plan each one gets -- is a property of the code and not of how busy the
#: machine was. Measured need at the rung that solves each level: 140k (level 4)
#: at w=1, ~790k (level 14) at w=4; the caps sit above those. The w=12 rung has
#: never been reached and is there so a level that needs it fails cheaply rather
#: than not at all.
LADDER: tuple[tuple[int, int], ...] = ((1, 175_000), (4, 900_000),
                                       (12, 300_000))


class AdInfinitumExpert(PSExpert):
    """Macro A* over `Board`, materialised into primitive directions.

    The plan CACHE holds macros -- ``("push", cell, direction)`` steps and a
    final ``("walk", beam cells)`` -- rather than key presses, so the same cached
    solution can be re-materialised with a different (still shortest) walk
    between each pair of pushes, and onto a different one of the equally-near
    beam cells, on every seed. Set `rng` to make that per-seed; leave it None for
    the deterministic route.
    """

    directions = ["up", "down", "left", "right"]

    #: `_key` is dynamic-objects-only, canonical only WITHIN a level (walls and
    #: HighFloor are static per level but differ between them).
    scope_by_level = True

    def setup(self) -> None:
        g = self.g
        ids = {name: set(g.resolve_object_name(name))
               for name in ("wall", "highfloor", "m/", "m\\", "player")}
        self._wall_ids = ids["wall"]
        self._high_id = g.obj_name_to_idx["highfloor"]
        self._slash_ids = ids["m/"]            # Mirror1 or VictoryMirror1
        self._backslash_ids = ids["m\\"]       # Mirror2 or VictoryMirror2
        self._player_ids = ids["player"]
        self._dynamic = self._slash_ids | self._backslash_ids | self._player_ids
        self._boards: dict[int, Board] = {}
        #: Per-seed route randomiser; see the class docstring.
        self.rng: random.Random | None = None
        #: `PLAN_CACHE` contents, level -> {"start": layout, "macros": ...}.
        self._disk = self._load_disk()

    def _key(self, eng) -> frozenset:
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & self._dynamic)
        )

    def board(self, eng, level: int | None) -> Board:
        """The level's `Board`, built once. Static geometry only, so building it
        from a mid-game grid is safe."""
        board = self._boards.get(level)
        if board is None:
            board = Board(eng, self._wall_ids, self._high_id,
                          self._slash_ids, self._backslash_ids,
                          self._player_ids)
            self._boards[level] = board
        return board

    def heuristic(self, eng) -> int:
        """Never called: `plan` is overridden, so `PSExpert._astar` -- the only
        caller -- never runs. The model-side heuristic is
        `Board.unsatisfied`."""
        raise NotImplementedError

    # -- search --------------------------------------------------------------
    def _astar_model(self, board: Board, player: tuple, mirrors: dict,
                     weight: int, node_cap: int):
        """``(macros, exhausted)``: a macro plan, or None with ``exhausted``
        telling whether the frontier emptied (a proof that this state cannot be
        won) or the node budget ran out."""
        # Nodes are stored as (parent index, macro) so a 1M-node search does not
        # also copy a 40-element path at every push.
        store: list = [(None, None)]
        start = frozenset(mirrors.items())
        heap = [(weight * board.unsatisfied(mirrors), 0, 0, player, start)]
        best = {(start, min(board.region(mirrors, player))): 0}
        expanded = 0

        def rebuild(idx: int) -> list:
            out: list = []
            while store[idx][0] is not None:
                idx, macro = store[idx][0], store[idx][1]
                out.append(macro)
            out.reverse()
            return out

        while heap:
            _f, g, idx, pos, layout = heapq.heappop(heap)
            mirrors = dict(layout)
            parent = board.region(mirrors, pos)

            lit = board.beams(mirrors)
            if lit is not None:
                # Every mirror is satisfied, so all that is left is to step into
                # a beam. Keep the whole nearest TIE SET: they are the same
                # number of moves apart, so which one a seed walks to is free
                # variety (see `materialise`).
                reach = [(board.walk_len(parent, cell), cell)
                         for cell in lit if cell in parent]
                if reach:
                    nearest = min(reach)[0]
                    ties = tuple(sorted(c for n, c in reach if n == nearest))
                    return rebuild(idx) + [("walk", ties)], False

            expanded += 1
            if expanded > node_cap:
                return None, False

            for cell in mirrors:
                r, c = cell
                for d, dr, dc in _MOVES:
                    stand = (r - dr, c - dc)
                    if stand not in parent:
                        continue
                    moved = board.push(mirrors, cell, d)
                    if moved is None:
                        continue
                    cost = g + board.walk_len(parent, stand) + 1
                    layout2 = frozenset(moved.items())
                    key = (layout2, min(board.region(moved, cell)))
                    if best.get(key, 1 << 30) <= cost:
                        continue
                    best[key] = cost
                    store.append((idx, ("push", cell, d)))
                    heapq.heappush(
                        heap,
                        (cost + weight * board.unsatisfied(moved), cost,
                         len(store) - 1, cell, layout2))
        return None, True

    def _search(self, board: Board, player: tuple, mirrors: dict):
        """Run `LADDER` until a rung wins or proves the state unsolvable."""
        for weight, node_cap in LADDER:
            macros, exhausted = self._astar_model(board, player, mirrors,
                                                  weight, node_cap)
            if macros is not None or exhausted:
                return macros
        return None

    # -- disk plan cache -----------------------------------------------------
    @staticmethod
    def _layout_json(player: tuple, mirrors: dict) -> list:
        return [list(player),
                sorted([r, c, t] for (r, c), t in mirrors.items())]

    @staticmethod
    def _macros_json(macros: list | None) -> list | None:
        if macros is None:
            return None
        out = []
        for macro in macros:
            if macro[0] == "walk":
                out.append(["walk", [list(cell) for cell in macro[1]]])
            else:
                out.append(["push", list(macro[1]), macro[2]])
        return out

    @staticmethod
    def _macros_from_json(raw: list | None) -> list | None:
        if raw is None:
            return None
        out = []
        for tag, arg, *rest in raw:
            if tag == "walk":
                out.append(("walk", tuple(tuple(cell) for cell in arg)))
            else:
                out.append(("push", tuple(arg), rest[0]))
        return out

    def _load_disk(self) -> dict:
        """``{level: {"start": layout, "macros": ...}}``, or empty if unreadable.
        A cache that cannot be parsed is a cache miss, never a crash."""
        try:
            raw = json.loads(PLAN_CACHE.read_text())
            return {int(k): v for k, v in raw.items()}
        except Exception:                                  # noqa: BLE001
            return {}

    def _save_disk(self) -> None:
        """Write the cache atomically -- shards started together would otherwise
        interleave into a truncated file (see `parallelize_generator`)."""
        try:
            PLAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = PLAN_CACHE.with_suffix(f".json.tmp{os.getpid()}")
            tmp.write_text(json.dumps(
                {str(k): self._disk[k] for k in sorted(self._disk)}, indent=1))
            os.replace(tmp, PLAN_CACHE)
        except OSError:
            pass                                           # cache is optional

    # -- macros -> key presses ----------------------------------------------
    def materialise(self, board: Board, player: tuple, mirrors: dict,
                    macros: list) -> list:
        """Expand a macro plan into primitive directions, drawing each walk --
        and the beam cell it ends on -- from `rng` (see the class docstring)."""
        mirrors = dict(mirrors)
        out: list = []
        for macro in macros:
            parent = board.region(mirrors, player, self.rng)
            if macro[0] == "walk":
                ties = macro[1]
                target = ties[0] if self.rng is None else self.rng.choice(ties)
                out += board.walk(parent, target)
                break
            _tag, cell, d = macro
            dr, dc = _DELTA[d]
            out += board.walk(parent, (cell[0] - dr, cell[1] - dc))
            out.append(d)
            mirrors = board.push(mirrors, cell, d)
            player = cell
        return out

    def stall(self, board: Board, player: tuple, mirrors: dict) -> list:
        """One turn that changes nothing, for the state that is ALREADY won on
        paper but has not been ticked yet.

        Levels start with no lasers on the board -- the rules that draw them run
        on a turn -- so a level shipped with every mirror satisfied AND the
        player already standing where a beam will be needs a key press before
        ``check_win`` can see it, and the search would hand back an empty plan.
        A move into a wall is a full turn (the game does not set
        ``require_player_movement``), which redraws the beams under a player who
        never left the cell. None of the 28 shipped levels needs this; it is here
        so that a level that did would lose a key press, not the whole seed."""
        for d, dr, dc in _MOVES:
            nxt = (player[0] + dr, player[1] + dc)
            if (not board.inside(nxt) or nxt in board.walls
                    or (nxt in mirrors and board.push(mirrors, nxt, d) is None)):
                return [d]
        # Nothing blocks the player, so step onto an empty neighbour and back --
        # the win fires on the return. Never onto a mirror: that is a push, and
        # it would break the very layout that is already won.
        for d, dr, dc in _MOVES:
            nxt = (player[0] + dr, player[1] + dc)
            if nxt not in mirrors:
                return [d, _OPPOSITE[d]]
        return []                     # boxed in by movable mirrors on all sides

    # -- PSExpert interface --------------------------------------------------
    def plan(self, eng, level: int | None = None) -> list | None:
        """A WIN sequence of primitive directions from the engine's current
        state, or None. Leaves the engine untouched (the search never runs it)."""
        key = (level, self._key(eng))
        board = self.board(eng, level)
        player, mirrors = board.read(eng)
        if key not in self.cache:
            if player is None:                   # pushed off the board by a rule
                self.cache[key] = None
            else:
                layout = self._layout_json(player, mirrors)
                entry = self._disk.get(level)
                if entry is not None and entry["start"] == layout:
                    self.cache[key] = self._macros_from_json(entry["macros"])
                else:
                    macros = self._search(board, player, mirrors)
                    self.cache[key] = macros
                    # Only the level's START state is worth keeping: it is the
                    # one state every seed re-plans from.
                    if entry is None or entry["start"] == layout:
                        self._disk[level] = {"start": layout,
                                             "macros": self._macros_json(macros)}
                        self._save_disk()
        macros = self.cache[key]
        if macros is None:
            return None
        moves = self.materialise(board, player, mirrors, macros)
        if not moves and not eng.check_win():
            moves = self.stall(board, player, mirrors)
        return moves


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class AdInfinitumSolver(PSAStarSolver):
    game_id = "puzzlescript_ad_infinitum"
    game_name = GAME_NAME
    expert_cls = AdInfinitumExpert

    #: The author's unlabelled "(Testing)" scratch boards -- 16 and 24 mirrors,
    #: no win found by anything up to greedy w=40. Skipped so the ladder is not
    #: burned on them at every startup. See the module docstring.
    skip_levels = frozenset({26, 27})

    #: Longest plan is 78 moves (level 14); the rest is exploration-prefix room.
    max_steps = 200

    def solve_episode(self, seed: int, explore: bool = True):
        # Re-materialise the cached macro plans along this seed's own shortest
        # walks (see the module docstring's Augmentation section).
        self._ensure(seed)[1].rng = random.Random(f"ad_infinitum:{seed}")
        return super().solve_episode(seed, explore)


if __name__ == "__main__":
    sys.exit(AdInfinitumSolver.main())
