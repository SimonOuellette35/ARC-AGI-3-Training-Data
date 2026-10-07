"""Generate Phase-1 training data for the PuzzleScript game ps:a_knights_tour.

The harness -- the trajectory recorder, the rotation contract and the BaseSolver
plumbing -- lives in `solvers/common/ps_astar.py`, shared with the other
search-the-interpreter ps: generators. This file is the game-specific part: its
adapter (the game ships its own, with per-level step limits), its state key, its
goal heuristic and -- the one real deviation from the family -- a MACRO-level A*
in place of the family's primitive-action A*.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_a_knights_tour",
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
A chessboard with a Knight, some Pawns, some Rooks and some Holes. You do NOT
move the knight directly: you move a red CURSOR (the PuzzleScript ``Player``,
which starts on the knight) one cell per arrow press, and the eight legal knight
destinations are highlighted each turn. Pressing ACTION (X / ACTION5) while the
cursor sits on a highlighted cell teleports the knight there, capturing whatever
piece stood on it. Holes are never highlighted, so the knight cannot land on
one; the cursor itself walks over anything and is stopped only by the board edge.
Win condition: ``no pawn`` and ``no rook`` -- every piece captured.

Rooks make it a timed puzzle. On EVERY turn (cursor moves included) a rook that
shares a row or column with the knight charges toward it until something blocks
it, and a rook that ends up orthogonally adjacent to the knight captures it --
the knight is deleted from the board and the level becomes unwinnable. On top of
that the game ships its own adapter (``games/ps:a_knights_tour``) imposing a
per-level step limit of ``50 + 10 * level``; exceeding it is a GAME_OVER. This
generator drives that same adapter, so a demonstration is only recorded if it
wins inside the budget an agent actually gets.

Expert solver: macro A*
-----------------------
The family's primitive-action A* (branch on up/down/left/right/action) does solve
this game, but it wastes its whole search budget re-discovering the cursor
walk: three of every four turns are pure cursor transport with a forced answer.
The observation that collapses it:

  * The cursor is on the knight at the start of the level and again after every
    capture, and every knight destination is a (+-1, +-2) / (+-2, +-1) offset,
    i.e. exactly 3 cursor moves away with no obstacle able to block them.
  * The cursor appears in no rule except "capture at the cursor", and it is
    alone in its collision layer -- so WHICH 3-move path it takes changes
    nothing. In particular the rooks only ever react to the KNIGHT, which does
    not move during cursor transport.

So a "jump the knight to cell X" macro is exactly ``manhattan(cursor, X)``
cursor presses (3 from a capture) plus one ACTION, and the search branches on
the <= 8 destinations instead of on 5 primitives at every one of those turns.
Macros are still executed by *stepping the real interpreter*, so every rook
charge, blocked charge and knight capture is the engine's own -- only the
branching is coarsened, never the dynamics. Any returned plan is a genuine WIN
path.

Two things make the search sharp:

  * ``heuristic`` is 4x the minimum spanning tree of {knight} u {pieces} under
    knight-move distance (BFS on the level's hole layout, cached per level). Any
    capture order is a walk visiting every piece, so the MST lower-bounds the
    number of jumps, and each jump costs at least 3 + 1 turns. Admissible from
    every state the search generates.
  * The search is bounded by the level's own step limit: a node whose cost has
    already passed ``50 + 10 * level`` cannot lead to a recordable win, so it is
    dropped. That makes "unsolvable" mean "unsolvable *within the budget the
    agent has*", which is the only sense that matters here.

Nodes whose knight has been captured are dropped outright -- with no knight the
win condition can never be met again, so they are dead ends, not just bad ones.

All 11 levels win, at 40, 40, 64, 64, 92, 24, 60, 116, 64, 88 and 64 turns --
optimal, since the heuristic is admissible. The searches cost roughly 6 minutes
in total (level 4 alone is ~4 of them) and are paid ONCE per process: the engine
state after reset is seed-independent, so every later seed replays from the plan
cache. A sharded run (``parallelize_generator.py``) pays it once per shard.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps); only the presentation varies per (seed, level): the frame rotation
(rotation_k in {0,1,2,3}) and the recolor of the eight surfaces listed in
``PuzzleScriptAdapter._recolor_surfaces`` -- the two checkerboard shades, the
holes, the knight, the pawns, the rooks, the move highlights and the cursor. This
game has no flip augmentation. The expert plan is therefore seed-independent:
solved once per level, cached, and replayed per seed with that seed's
rotation-remapped screen actions and recolored frames.

That recolor is also what makes the game OBSERVABLE. Shipped, the cursor (`red`)
and the move highlights (`lightred`) both map to ARC palette 8, and the cursor's
sprite is a strict subset of the highlight's -- so a cursor standing on a legal
destination rendered pixel-identical to that destination with no cursor on it,
making the single state in which ACTION matters invisible. Because the recolor
draws every surface from a shared "already used" set, cursor and highlight are
now guaranteed distinct on every (seed, level).

Usage (run from the repo root):
    python solvers/generate_a_knights_tour_training.py --episodes 200 \
        --out data/training_multi_level/a_knights_tour
"""

from __future__ import annotations

import heapq
import importlib.util
import sys
from collections import deque
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from solvers.common.ps_astar import (                   # noqa: E402
    PSAStarSolver, PSExpert, restore, snapshot,
)

GAME_NAME = "A_Knight's_Tour"
_GAME_ID = "ps:a_knights_tour"

# The game id carries a ':' namespace, so the module cannot be imported by name
# (``games.ps:a_knights_tour`` is not a legal identifier) -- load it by path, the
# same way ``game_envs._load_puzzlescript_env`` does for live play. Going through
# the game's own ``make_game`` is what gives the generator the SAME per-level step
# limits a live agent plays under, so a demo is never recorded for a plan the
# agent would be cut off in the middle of.
_GAME_PATH = _REPO_ROOT / "games" / _GAME_ID / f"{_GAME_ID}.py"
_spec = importlib.util.spec_from_file_location("ps_game_a_knights_tour", _GAME_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
make_knights_tour_game = _module.make_game

#: The eight knight offsets, as (d_row, d_col). Every one of them is exactly 3
#: cursor presses away (|dr| + |dc| == 3), which is what fixes the macro cost.
KNIGHT_OFFSETS = (
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
)

#: Cost floor of one capture: 3 cursor presses + 1 ACTION.
JUMP_COST = 4

#: Stand-in for "no knight route between these two cells" in the distance table.
#: Larger than any real knight distance on a 9x9 board, small enough that the MST
#: sum cannot overflow into a plan-length comparison.
UNREACHABLE = 999


class KnightsTourExpert(PSExpert):
    """Macro A* over knight jumps, executed on the real interpreter.

    Successors are "walk the cursor to knight-destination X, press ACTION",
    i.e. ``manhattan(cursor, X)`` direction presses followed by ``action``. The
    macro is only accepted if the engine really did put the knight on X, so a
    destination the game refuses (a hole is never highlighted, and the capture
    rule needs the highlight) prunes itself.
    """

    #: `_key` is dynamic-objects-only, which is canonical only WITHIN a level --
    #: holes and background differ between levels. See `PSExpert.scope_by_level`.
    scope_by_level = True

    def setup(self) -> None:
        g = self.g
        self.knight = g.obj_name_to_idx["knight"]
        self.pawn = g.obj_name_to_idx["pawn"]
        self.hole = g.obj_name_to_idx["hole"]
        self.player = g.obj_name_to_idx["player"]
        self.rooks = frozenset(g.obj_name_to_idx[n] for n in g.or_groups["rook"])
        #: Everything that moves or disappears. Holes / background are static per
        #: level; the Option highlights are a pure function of the knight cell and
        #: are recomputed every turn, so neither belongs in the state key.
        self.dynamic = self.rooks | {self.knight, self.pawn, self.player}
        #: level hole-layout -> all-pairs knight-move distances.
        self._dist_cache: dict[tuple, dict] = {}
        #: Step budget of the level currently being planned (set by `plan`).
        self._budget = 1 << 30

    # ── state ────────────────────────────────────────────────────────────────
    def _key(self, eng) -> frozenset:
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & self.dynamic)
        )

    def _read(self, eng) -> tuple:
        """(knight cell, cursor cell, [piece cells]) -- one pass over the grid."""
        knight = cursor = None
        pieces = []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if self.knight in cell:
                    knight = (r, c)
                if self.player in cell:
                    cursor = (r, c)
                if self.pawn in cell or (cell & self.rooks):
                    pieces.append((r, c))
        return knight, cursor, pieces

    # ── knight-move geometry (static per level) ──────────────────────────────
    def _distances(self, eng) -> dict:
        """All-pairs knight-move distance over the level's non-hole cells.

        Holes are excluded as *destinations* only, which is exactly the game's
        rule (the highlight is deleted on a hole). Cached by hole layout, so the
        11 levels pay for this once each.
        """
        h, w = eng.height, eng.width
        holes = frozenset((r, c) for r in range(h) for c in range(w)
                          if self.hole in eng.grid[r][c])
        cached = self._dist_cache.get((h, w, holes))
        if cached is not None:
            return cached
        table = {}
        for source in ((r, c) for r in range(h) for c in range(w)):
            dist = {source: 0}
            queue = deque([source])
            while queue:
                r, c = queue.popleft()
                for dr, dc in KNIGHT_OFFSETS:
                    nxt = (r + dr, c + dc)
                    if (0 <= nxt[0] < h and 0 <= nxt[1] < w
                            and nxt not in holes and nxt not in dist):
                        dist[nxt] = dist[(r, c)] + 1
                        queue.append(nxt)
            table[source] = dist
        self._dist_cache[(h, w, holes)] = table
        return table

    # ── heuristic ────────────────────────────────────────────────────────────
    def heuristic(self, eng) -> int:
        """4 x MST({knight} u pieces) under knight-move distance.

        Capturing every piece means the knight walks a path through all of them,
        and any such walk contains a spanning tree of that node set, so the MST
        weight lower-bounds the number of jumps. Each jump costs at least
        ``JUMP_COST`` turns (the cursor is on the knight after a capture, and the
        nearest highlight is 3 presses away), so the product is admissible from
        every state this search generates. Returns ``None`` for a dead state --
        no knight left, so the win condition can never be met again.
        """
        knight, cursor, pieces = self._read(eng)
        if knight is None:
            return None
        if not pieces:
            return 0
        dist = self._distances(eng)
        return JUMP_COST * _mst_weight(dist, knight, pieces)

    # ── macros ───────────────────────────────────────────────────────────────
    def _macros(self, eng, knight, cursor) -> list[tuple[tuple[int, int], list[str]]]:
        """(destination, primitive presses) for every knight destination on the
        board. The cursor route is vertical-then-horizontal; any route of the
        same length behaves identically (the cursor is inert and the knight,
        which is all the rooks react to, does not move during transport)."""
        out = []
        for dr, dc in KNIGHT_OFFSETS:
            dest = (knight[0] + dr, knight[1] + dc)
            if not (0 <= dest[0] < eng.height and 0 <= dest[1] < eng.width):
                continue
            if self.hole in eng.grid[dest[0]][dest[1]]:
                continue                       # never highlighted -> not a move
            vr, vc = dest[0] - cursor[0], dest[1] - cursor[1]
            path = (["up" if vr < 0 else "down"] * abs(vr)
                    + ["left" if vc < 0 else "right"] * abs(vc)
                    + ["action"])
            out.append((dest, path))
        return out

    # ── search ───────────────────────────────────────────────────────────────
    def plan(self, eng, level: int | None = None) -> list | None:
        """As `PSExpert.plan`, but bounded by the level's own step limit: the
        adapter flips to GAME_OVER at ``50 + 10 * level`` actions, so a longer
        plan is not a plan. ``_max_steps`` is read off the adapter, which
        ``set_level`` has already primed for this level."""
        self._budget = int(getattr(self.game, "_max_steps", 1 << 30))
        return super().plan(eng, level)

    def _astar(self, eng) -> list | None:
        if eng.check_win():
            return []
        start_h = self.heuristic(eng)
        if start_h is None:
            return None
        start = snapshot(eng)
        start_key = self._key(eng)
        best_g = {start_key: 0}
        counter = 0
        nodes = 0
        heap = [(self.weight * start_h, 0, counter, start_key, start, [])]
        while heap:
            _f, g, _c, key, snap, path = heapq.heappop(heap)
            if g > best_g.get(key, -1):
                continue                       # stale duplicate, already improved
            restore(eng, snap)
            knight, cursor, _pieces = self._read(eng)
            for dest, macro in self._macros(eng, knight, cursor):
                if g + len(macro) > self._budget:
                    continue                   # cannot finish inside the level's limit
                restore(eng, snap)
                for press in macro:
                    eng.step(press)
                nodes += 1
                if eng.check_win():
                    return path + macro
                new_knight, _cur, _p = self._read(eng)
                if new_knight != dest:
                    continue                   # jump refused, or the knight died
                new_key = self._key(eng)
                new_g = g + len(macro)
                if best_g.get(new_key, 1 << 30) <= new_g:
                    continue
                h = self.heuristic(eng)
                if h is None:
                    continue                   # knight captured -> dead end
                best_g[new_key] = new_g
                counter += 1
                heapq.heappush(heap, (new_g + self.weight * h, new_g, counter,
                                      new_key, snapshot(eng), path + macro))
            if nodes >= self.node_cap:
                return None
        return None


def _mst_weight(dist: dict, knight: tuple, pieces: list) -> int:
    """Prim's MST over {knight} u pieces in the knight-move metric."""
    nodes = [knight] + pieces
    n = len(nodes)
    edge = [0] + [UNREACHABLE] * (n - 1)
    used = [False] * n
    total = 0
    for _ in range(n):
        u = min((i for i in range(n) if not used[i]), key=lambda i: edge[i])
        used[u] = True
        total += edge[u]
        row = dist[nodes[u]]
        for v in range(n):
            if not used[v]:
                w = row.get(nodes[v], UNREACHABLE)
                if w < edge[v]:
                    edge[v] = w
    return total


#: Process-wide adapter + plan cache + solvable-level set, shared by every
#: `KnightsTourSolver` instance. See `KnightsTourSolver._ensure`.
_SHARED: dict = {}


class KnightsTourSolver(PSAStarSolver):
    game_id = "puzzlescript_a_knights_tour"
    game_name = GAME_NAME
    expert_cls = KnightsTourExpert

    node_cap = 400_000
    weight = 1
    #: Recorder cap only; the binding limit is the adapter's own per-level budget
    #: (50..150), which `KnightsTourExpert.plan` already bounds the search by.
    max_steps = 200

    def make_game(self, seed: int):
        # The game's own adapter subclass, so the generator plays under the same
        # per-level step limits as a live agent (see the module docstring).
        return make_knights_tour_game(seed=seed)

    def _ensure(self, seed: int):
        """As `PSAStarSolver._ensure`, but the adapter / expert / solvable set are
        cached per PROCESS rather than per solver instance.

        The 11 searches cost ~6 minutes and their results depend on neither the
        seed nor the instance (fixed ASCII levels, deterministic interpreter), so
        a tool that builds a fresh solver per run -- ``test_datagen.py`` does,
        once per ``--runs`` -- would otherwise re-pay all six minutes every run.
        """
        if self._game is None and "game" in _SHARED:
            self._game = _SHARED["game"]
            self._expert = _SHARED["expert"]
            self._solvable = _SHARED["solvable"]
        game, expert, solvable = super()._ensure(seed)
        _SHARED.update(game=game, expert=expert, solvable=solvable)
        return game, expert, solvable


if __name__ == "__main__":
    sys.exit(KnightsTourSolver.main())
