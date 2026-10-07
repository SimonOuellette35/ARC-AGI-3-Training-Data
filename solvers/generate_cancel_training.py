"""Generate Phase-1 training data for the PuzzleScript game ps:cancel
("Cancel" by Dean Huff -- carry the four elements into each other until the
board is empty).

The harness -- the rotation contract, the trajectory recorder, the
RESET-recovery prefix and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: the mechanic, a native model of it, the pair-distance
tables that make the search trivial, and the optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_cancel",
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
Four arrow keys, no ACTION (nothing binds it, so it is dropped from the search),
and four win conditions: ``No Earth``, ``No Water``, ``No Fire``, ``No Air``.
Two late rules delete pieces in pairs --

    late [ Fire | Water ] -> [ | ]
    late [ Earth | Air  ] -> [ | ]

-- so the whole game is "bring each Earth next to an Air and each Fire next to a
Water". The rest of the file is how you carry a piece, and the four elements
each answer that differently:

    [ > Player | Earth ] -> [ > Player | > Earth ]      push
    [ < Player | Water ] -> [ < Player | < Water ]      pull
    [ > Player | Fire  ] -> [ Fire | Player ]           swap
    [ V Player | Air   ] -> [ V Player | < Air ]        drag
    [ ^ Player | Air   ] -> [ ^ Player | < Air ]        drag

Read them as "where does the player have to stand, and which way does it press,
to move this piece one cell in direction ``d``":

  * **Earth** is a sokoban crate: stand at ``X - d`` and press ``d``. It is the
    only piece the player pushes AWAY from itself, so it is the only one that
    needs a free cell on the far side -- and the only one that can be shoved
    somewhere it can never be recovered from.
  * **Water** is pulled: stand at ``X + d`` and press ``d``, i.e. walk directly
    away from it and it follows into the cell you left. Needs ``X + 2d`` free.
  * **Fire** swaps with the player: stand at ``X + d`` and press ``-d``, walking
    into it. Needs nothing free at all, which makes fire the cheapest piece to
    move and the reason the fire/water levels are about the WATER.
  * **Air** is dragged sideways: stand at ``X + d`` and press either direction
    PERPENDICULAR to ``d``. The air lands in the cell the player just left.

The unifying fact, and the one worth keeping in mind while reading the search:
water, fire and air all move INTO the cell the player vacates; only earth moves
away from it. So for every piece except earth the player has to reach the
piece's DESTINATION first, and a piece can never be carried into a pocket the
player cannot already stand in.

Two consequences that shape the levels:

  * **Walking is not free.** A player stepping past an air drags it, and a
    player stepping away from a water pulls it. There is no such thing as a
    side-effect-free route across "Blustery day" (level 7) -- the airs are
    scattered exactly where you have to walk. This is why the search below is
    over PRIMITIVE presses and not over the walk-to-and-push macros the other
    sokoban-shaped ps: generators use (`PSPushExpert`): a macro search has to
    assume its walks are inert, and here they are not.
  * **Two airs cancel each other's drag.** Both drag rules fire, both airs are
    forced into the one cell the player vacated, and the engine refuses both --
    so a player squeezing between two airs disturbs neither. That is a genuine
    mechanic (it is how you cross the middle of "Barometric pressure"), not an
    engine quirk, and `_Board.step` models it exactly.

Level notes (all 8 are solved; 4 tutorials and 4 real puzzles):

  * **0-3** teach one element each: 25, 26, 58 and 34 presses, and those four
    numbers are provably optimal (with a single pair on the board the pair table
    below IS the exact distance, so the A* runs on a perfect heuristic).
  * **4 "Barometric pressure"** -- 10 earths ringing a 10-air blob. Every earth
    is walled in on three sides, so nine of them cannot be pushed anywhere
    useful and several cannot be pushed at all: the puzzle is to walk the AIR
    out to them, one at a time, which is what makes the drag rule the whole
    level.
  * **5 "Bucket brigade"** and **6 "Boiling point"** are the fire/water levels.
    Fire is free to move and water is not, so both are about routing water --
    and level 6's bottom-right water sits in a pocket whose only free neighbour
    is reachable only THROUGH the fire train, which is why the greedy
    "cancel the closest pair" ordering has to be allowed to backtrack.
  * **7 "Blustery day"** -- 5 airs, 5 earths pinned in the right-hand column.
    The airs have to be walked the width of the board, through a maze that keeps
    putting other airs beside the player.

Expert solver
-------------
Planning happens on `_Board`, a native re-implementation of the five rules and
the two late rules, and every plan is then certified on the real interpreter
(`_certify`) before it is returned. The engine step is only 0.15 ms, so the
model is not about raw speed per press -- it is about the PAIR TABLES, which
need a few million transitions each and would take minutes through the
interpreter. `_fuzz_check` replays random press sequences through both and
compares the full board after every single press, including on randomly
scattered high-density boards that random play from a level start would never
reach; it is what earns the right to plan on the model at all.

The search is two-tiered:

  * `PairTable` is the EXACT number of presses to cancel one piece of kind
    ``ka`` against one of kind ``kb`` on an otherwise empty board, for every
    ``(player cell, piece cell, piece cell)`` triple -- a fixpoint over the whole
    3-body state graph, computed once per level (a few seconds) and cached on
    disk with the plan. Two tables cover the game: Earth/Air and Fire/Water.
  * `_Planner.next_cancels` is an A* over primitive presses on the FULL board
    whose goal is "the piece count drops" and whose heuristic is the minimum
    pair-table value over the pairs still on the board. That heuristic is a
    lower bound (the table's board has strictly fewer obstacles and the real one
    can only cost more) and it is a very tight one, so each cancellation falls
    out in milliseconds where the naive "distance between the two cells"
    heuristic could not find one at all inside 120k nodes.

The table also supplies the DEAD test for free: a state where every surviving
pair scores infinity is one where nothing can ever cancel again, so the
successor is dropped instead of queued. Earth is easy to strand (push one into a
corner and it is out of the game), and this is what stops the search from
exploring behind that mistake.

`_Planner.solve` then beams over the ORDER of the cancellations -- which pair to
clear next is a real choice, and the greedy "closest first" answer is not the
best one -- keeping the shortest plan found across a ladder of beam widths.
Nothing here claims the resulting plan is globally shortest; each cancellation
SEGMENT is shortest, and the ladder is what buys back most of the rest (level 6
goes 262 -> 212 presses, level 4 104 -> 88).

The step limit (why this generator goes through the game folder)
---------------------------------------------------------------
`PuzzleScriptAdapter` gives every level 200 actions and flips to GAME_OVER on the
201st, and Cancel's last three boards need 180, 212 and 287 presses -- so as
shipped they cannot be finished by anyone, expert or agent. It is a silent
failure and worth recognising in the next ps: game that behaves this way: the
plan replays correctly, the frames are all fine, and then the level simply never
reaches WIN, so the seed is discarded and the game generates zero episodes with
no error anywhere. `games/ps:cancel/ps:cancel.py` raises the per-level limit (the
same treatment `games/ps:bridge` and `games/ps:a_knights_tour` already get), and
``game_module_id`` points this generator at THAT adapter, so it records against
the game a live agent is actually given.

Optimal-action targets
----------------------
`_optimal_sets` labels every recorded press. A plan splits into stretches where
only the player moves and single presses that move a piece; within a stretch,
every shortest route to the cell the stretch ends on over the SIDE-EFFECT-FREE
walk graph (a step is in that graph exactly when the model says it moves the
player and nothing else) reaches the identical board for the identical cost, so
all of them are labelled optimal. Piece-moving presses are labelled with
themselves, as is any stretch whose recorded route is not a shortest one. The
claim is interchangeability with the step actually taken, not that the plan is
the shortest possible -- see `always-emit-optimal-targets`: every expert step
ships with a target.

Augmentation
------------
Cancel's engine state after reset is identical for every seed (the levels are
fixed ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip, each with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`). Every rule in the game is stated over
relative directions and is symmetric in the two perpendiculars, and its input is
screen-relative with no gravity, so the board's whole 8-element symmetry group
is a presentation the game could really have shipped. That takes 8 levels to 64
presentations. No colour augmentation: the six objects already land on six
distinct ARC indices (background 5, wall 12, player 3, earth 14, water 9,
fire 8, air 0) and telling the four elements apart IS the game, so a flattening
recolor could only take information away.

The expert plan is therefore seed-independent: solved once per level, cached in
memory AND on disk (`data/cancel_plans.json`), and replayed per seed with that
seed's remapped screen actions. A cold run pays ~80s for the tables and beams of
all eight levels; a warm one generates six episodes in 14s. Delete the file to
re-derive it.

Usage (run from the repo root):
    python solvers/generate_cancel_training.py --episodes 200 \
        --out data/training_multi_level/cancel

    python solvers/generate_cancel_training.py --plans   # per-level report
    python solvers/generate_cancel_training.py --fuzz    # model vs interpreter
"""

from __future__ import annotations

import heapq
import json
import os
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (PSAStarSolver, PSExpert,  # noqa: E402
                                     restore, snapshot)

GAME_NAME = "Cancel"

_REPO_ROOT = Path(__file__).resolve().parent.parent
#: Per-level start plan + optimal sets. Without it every shard of
#: `parallelize_generator` would rebuild all sixteen pair tables on every core.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "cancel_plans.json"

_DIRS = ("up", "down", "left", "right")
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: For a player press ``m``, the two cells whose Air is forced into the cell the
#: player leaves, in the order the two drag rules are written. Both are checked:
#: when BOTH hold an air the engine refuses both moves (they contend for the one
#: vacated cell), which is why the count matters and not the order.
_AIR_OFFSETS = {
    "down":  ((0, 1), (0, -1)),
    "up":    ((0, -1), (0, 1)),
    "right": ((-1, 0), (1, 0)),
    "left":  ((1, 0), (-1, 0)),
}

#: ``late [ Fire | Water ]`` then ``late [ Earth | Air ]``, each expanded over
#: ``rule.directions`` in the engine's order. The order is only observable when
#: one piece is adjacent to two partners at once, and then it decides which of
#: them is consumed -- so it is part of the mechanic, not a detail.
_CANCELS = (("F", "W"), ("E", "A"))

#: Which kind annihilates which.
_PARTNER = {"E": "A", "A": "E", "F": "W", "W": "F"}

#: Heuristic value of "these two can never meet". Also the DEAD test: a state
#: where every surviving pair scores this is unwinnable.
_INF = 1 << 20


# ---------------------------------------------------------------------------
# Native model of the mechanic
# ---------------------------------------------------------------------------

class _Board:
    """The static walls of one level plus the game's dynamics.

    A state is the pair ``(player_cell, {cell: kind})`` with kind in ``EWFA``;
    `step` is the whole mechanic. It is verified against the interpreter by
    `_fuzz_check` -- run that after ANY edit here."""

    def __init__(self, eng, game):
        idx = game.obj_name_to_idx
        grid = eng.grid
        self.h, self.w = len(grid), len(grid[0])
        self.player_id = idx["player"]
        self.kind_of = {idx["earth"]: "E", idx["water"]: "W",
                        idx["fire"]: "F", idx["air"]: "A"}
        self.walls = frozenset(
            (r, c) for r in range(self.h) for c in range(self.w)
            if idx["wall"] in grid[r][c])
        self.open_cells = [(r, c) for r in range(self.h) for c in range(self.w)
                           if (r, c) not in self.walls]

    def read(self, eng):
        """Engine grid -> ``(player_cell, {cell: kind})``."""
        player = None
        pieces = {}
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                for o in cell:
                    kind = self.kind_of.get(o)
                    if kind is not None:
                        pieces[(r, c)] = kind
                if self.player_id in cell:
                    player = (r, c)
        return player, pieces

    def inb(self, cell) -> bool:
        return 0 <= cell[0] < self.h and 0 <= cell[1] < self.w

    def step(self, player, pieces, direction):
        """One turn. Returns ``(player, pieces)``.

        ``pieces`` is handed back as the SAME object whenever no piece moved, so
        callers can test ``result is pieces`` to tell a plain walk (or a refused
        press) from a press that changed the board -- which is how `_Planner`
        prunes no-ops and how `_optimal_sets` finds its walk stretches."""
        delta = _DELTA.get(direction)
        if delta is None:                       # ACTION is bound to nothing
            return player, pieces
        dr, dc = delta
        p = player
        q = (p[0] + dr, p[1] + dc)
        if not self.inb(q) or q in self.walls:
            return player, pieces
        occ = pieces.get(q)
        if occ == "E":                          # push: needs the far side free
            far = (q[0] + dr, q[1] + dc)
            if (not self.inb(far) or far in self.walls or far in pieces):
                return player, pieces
            new = dict(pieces)
            del new[q]
            new[far] = "E"
            vacated = p
        elif occ == "F":                        # swap: the fire takes p
            new = dict(pieces)
            del new[q]
            new[p] = "F"
            vacated = None
        elif occ is not None:                   # water / air block head-on
            return player, pieces
        else:
            new = None                          # nothing has moved yet
            vacated = p

        if vacated is not None:
            # Exactly one piece can follow into the cell the player left. The
            # water rule is written before the two air rules and wins outright;
            # two contending airs cancel each other and neither moves.
            back = (p[0] - dr, p[1] - dc)
            src = new if new is not None else pieces
            if self.inb(back) and src.get(back) == "W":
                if new is None:
                    new = dict(pieces)
                del new[back]
                new[vacated] = "W"
            else:
                airs = [a for a in
                        ((p[0] + ar, p[1] + ac)
                         for ar, ac in _AIR_OFFSETS[direction])
                        if self.inb(a) and src.get(a) == "A"]
                if len(airs) == 1:
                    if new is None:
                        new = dict(pieces)
                    del new[airs[0]]
                    new[vacated] = "A"

        if new is None:                         # the player walked, nothing else
            return q, pieces
        return q, self._cancel(new)

    def _cancel(self, pieces):
        """Both late rules to fixpoint, in the engine's rule-then-direction
        order. Mutates and returns ``pieces``.

        Matches within one direction pass never overlap (the partner cell is a
        fixed offset from the anchor, so both ends of every match are distinct
        objects), which is why the engine's "apply every match found in one
        scan" is reproduced here by simply applying them all."""
        for a, b in _CANCELS:
            for dr, dc in _D4:
                while True:
                    hits = [((r, c), (r + dr, c + dc))
                            for (r, c), k in pieces.items()
                            if k == a and pieces.get((r + dr, c + dc)) == b]
                    if not hits:
                        break
                    for x, y in hits:
                        pieces.pop(x, None)
                        pieces.pop(y, None)
        return pieces

    def walk_step(self, cell, pieces, direction):
        """The cell reached by a press that moves the PLAYER AND NOTHING ELSE,
        or None. This is the edge relation of the side-effect-free walk graph
        `_optimal_sets` labels ties over, and it is defined by `step` itself so
        the two can never drift apart."""
        npl, npcs = self.step(cell, pieces, direction)
        return npl if (npcs is pieces and npl != cell) else None


# ---------------------------------------------------------------------------
# Fuzz: the model against the real interpreter
# ---------------------------------------------------------------------------

def _fuzz_check(trials: int = 2500, max_len: int = 40, seed: int = 11,
                verbose: bool = True) -> int:
    """Replay random press sequences through the interpreter AND `_Board` and
    compare the full board after every single press. Returns the mismatch count.

    Half the rollouts start from a level's real start position; the other half
    scatter 2-22 random pieces over that level's walls. The scattered boards are
    the point: the contended cases -- an air on each side of the player, a water
    behind AND an air beside, a chain of pieces against a push -- are what a
    model gets wrong, and random play from a start position essentially never
    reaches them (the first fix this found, "two airs contend and NEITHER
    moves", showed up only on the scattered boards).

    Run it after any edit to `_Board.step`:
        python solvers/generate_cancel_training.py --fuzz
    """
    from adapters.puzzlescript_adapter import PuzzleScriptAdapter

    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    idx = game._game.obj_name_to_idx
    inv = {"E": idx["earth"], "W": idx["water"], "F": idx["fire"],
           "A": idx["air"]}
    rng = random.Random(seed)
    presses = list(_DIRS) + ["action"]
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = _Board(eng, game._game)
        start = snapshot(eng)
        origin = board.read(eng)
        for trial in range(trials):
            if trial % 2:
                cells = list(board.open_cells)
                rng.shuffle(cells)
                n = rng.randint(2, min(rng.choice([6, 12, 22]), len(cells) - 1))
                pieces = board._cancel(
                    {cells[i]: rng.choice("EWFA") for i in range(n)})
                player = cells[n]
                if player in pieces:
                    continue
                grid = [[{idx["background"]} for _ in range(board.w)]
                        for _ in range(board.h)]
                for cell in board.walls:
                    grid[cell[0]][cell[1]] = {idx["wall"]}
                for cell, kind in pieces.items():
                    grid[cell[0]][cell[1]] = {inv[kind]}
                grid[player[0]][player[1]] = {idx["player"]}
                restore(eng, grid)
            else:
                restore(eng, start)
                player, pieces = origin[0], dict(origin[1])
            for _ in range(rng.randint(1, max_len)):
                press = rng.choice(presses)
                eng.step(press)
                player, pieces = board.step(player, pieces, press)
                if (board.read(eng) != (player, pieces)
                        or eng.check_win() != (not pieces)):
                    bad += 1
                    if verbose and bad <= 8:
                        print(f"MISMATCH level {level} trial {trial} "
                              f"press {press!r}")
                        print("  engine:", board.read(eng), eng.check_win())
                        print("  model :", (player, pieces), not pieces)
                    break
                if eng.check_win():
                    break
        restore(eng, start)
    print(f"fuzz: {trials * game.n_levels} rollouts, {bad} mismatches")
    return bad


# ---------------------------------------------------------------------------
# Exact pair-distance tables
# ---------------------------------------------------------------------------

class _PairTable:
    """``get(player, x, y)`` = the exact number of presses to cancel a piece of
    kind ``ka`` at ``x`` against one of kind ``kb`` at ``y``, on a board holding
    nothing else.

    Built as a fixpoint over the WHOLE 3-body state graph rather than by a
    search per query: there are only ``|open cells|^3`` states (a few hundred
    thousand), every query on the real board is then a table lookup, and the
    result is admissible for the real board -- the real board has strictly more
    obstacles (the other pieces) and more obstacles can only cost presses, never
    save them.

    States where ``x`` and ``y`` are already 4-adjacent are not states: they
    would have cancelled the moment they arose.
    """

    def __init__(self, board: _Board, ka: str, kb: str):
        cells = board.open_cells
        self.n = n = len(cells)
        self.idx = {cell: i for i, cell in enumerate(cells)}
        size = n * n * n
        # -1: this press cancels the pair.  -2: no such transition (the press is
        # a no-op, or the triple is not a state at all).
        succ = np.full((size, 4), -2, dtype=np.int32)
        step = board.step
        for ip, p in enumerate(cells):
            for ix, x in enumerate(cells):
                if x == p:
                    continue
                base = (ip * n + ix) * n
                for iy, y in enumerate(cells):
                    if y == p or y == x:
                        continue
                    if abs(x[0] - y[0]) + abs(x[1] - y[1]) == 1:
                        continue
                    s = base + iy
                    pieces = {x: ka, y: kb}
                    for d, direction in enumerate(_DIRS):
                        npl, npcs = step(p, pieces, direction)
                        if len(npcs) < 2:
                            succ[s, d] = -1
                        elif npcs is pieces and npl == p:
                            succ[s, d] = s
                        else:
                            nx = ny = None
                            for cell, kind in npcs.items():
                                if kind == ka and nx is None:
                                    nx = cell
                                else:
                                    ny = cell
                            succ[s, d] = ((self.idx[npl] * n + self.idx[nx]) * n
                                          + self.idx[ny])
        self.dist = self._fixpoint(succ, size)

    @staticmethod
    def _fixpoint(succ, size):
        """Distance-to-cancellation for every state, by relaxing
        ``d[s] = 1 + min_d d[succ(s, d)]`` until it stops moving.

        Vectorised over all states at once: the alternative (invert the edge
        list and BFS backwards) needs the reverse graph materialised, and this
        converges in as many passes as the deepest state is deep -- tens, each
        one a handful of numpy operations."""
        # Two sentinel slots past the real states: `size` is "cancels here"
        # (distance 0) and `size + 1` is "no transition" (stays unreachable).
        dist = np.full(size + 2, _INF, dtype=np.int32)
        dist[size] = 0
        work = np.where(succ == -1, size,
                        np.where(succ == -2, size + 1, succ))
        for _ in range(600):
            best = dist[work].min(axis=1)
            np.minimum(best, _INF - 1, out=best)
            best += 1
            new = np.minimum(dist[:size], best)
            if np.array_equal(new, dist[:size]):
                break
            dist[:size] = new
        return dist

    def get(self, player, x, y) -> int:
        idx = self.idx
        ip, ix, iy = idx.get(player), idx.get(x), idx.get(y)
        if ip is None or ix is None or iy is None:
            return _INF
        return int(self.dist[(ip * self.n + ix) * self.n + iy])


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

#: ``(beam width, cancellations tried per beam entry)`` rungs. Every rung is run
#: and the SHORTEST plan wins: a wider beam is not monotonically better here
#: (its extra candidates can crowd a good line out of the next layer), and the
#: whole ladder costs seconds against a plan that is then cached on disk.
_BEAMS = ((8, 6), (24, 12), (64, 24))


class _Planner:
    """Beam over the ORDER of the cancellations; A* over presses inside each.

    See the module docstring. The pair tables are the expensive part and are
    built once per level here."""

    def __init__(self, board: _Board, node_cap: int = 200_000):
        self.b = board
        self.node_cap = node_cap
        self.tab = {pair: _PairTable(board, *pair) for pair in _CANCELS}

    # -- heuristic / dead test ---------------------------------------------
    def h(self, player, pieces) -> int:
        """Presses to the NEXT cancellation, lower-bounded by the cheapest pair
        left. ``_INF`` means no pair can ever meet again -- the state is dead."""
        best = _INF
        for c1, k1 in pieces.items():
            if k1 == "A" or k1 == "W":
                continue                       # each pair is visited from E / F
            table = self.tab[(k1, _PARTNER[k1])]
            for c2, k2 in pieces.items():
                if k2 != _PARTNER[k1]:
                    continue
                value = table.get(player, c1, c2)
                if value < best:
                    best = value
        return best

    def matching_bound(self, player, pieces) -> int:
        """Optimistic presses to clear the WHOLE board: greedily match each
        Earth/Fire to its cheapest free partner and sum. Only used to rank beam
        entries against each other, so a greedy matching (rather than a real
        assignment) is enough."""
        total = 0
        taken = set()
        for c1, k1 in pieces.items():
            if k1 == "A" or k1 == "W":
                continue
            table = self.tab[(k1, _PARTNER[k1])]
            best, best_cell = _INF, None
            for c2, k2 in pieces.items():
                if k2 != _PARTNER[k1] or c2 in taken:
                    continue
                value = table.get(player, c1, c2)
                if value < best:
                    best, best_cell = value, c2
            if best_cell is None:
                return _INF
            taken.add(best_cell)
            total += best
        return total

    # -- inner: the cheapest ways to cancel ONE pair ------------------------
    def next_cancels(self, player, pieces, want=3):
        """Up to ``want`` distinct cheapest press sequences that reduce the piece
        count, shortest first. Plain A*: the first is optimal, the rest are the
        alternatives the outer beam branches on."""
        board = self.b
        n0 = len(pieces)
        h0 = self.h(player, pieces)
        if h0 >= _INF:
            return []
        pq = [(h0, 0, 0, player, pieces, [])]
        best = {(player, frozenset(pieces.items())): 0}
        counter = nodes = 0
        out, seen_goal = [], set()
        while pq:
            _f, g, _c, pl, pcs, path = heapq.heappop(pq)
            for direction in _DIRS:
                npl, npcs = board.step(pl, pcs, direction)
                nodes += 1
                if npcs is pcs and npl == pl:
                    continue                    # the engine refused the press
                key = (npl, frozenset(npcs.items()))
                if len(npcs) < n0:
                    if key not in seen_goal:
                        seen_goal.add(key)
                        out.append(path + [direction])
                        if len(out) >= want:
                            return out
                    continue
                ng = g + 1
                if best.get(key, 1 << 30) <= ng:
                    continue
                hv = self.h(npl, npcs)
                if hv >= _INF:
                    continue                    # dead: nothing can cancel again
                best[key] = ng
                counter += 1
                heapq.heappush(pq, (ng + hv, ng, counter, npl, npcs,
                                    path + [direction]))
            if nodes >= self.node_cap:
                break
        return out

    # -- outer: which pair to clear next ------------------------------------
    def solve(self, player, pieces):
        """The shortest plan found over the `_BEAMS` ladder, or None."""
        shortest = None
        for width, fanout in _BEAMS:
            found = self._beam(player, pieces, width, fanout)
            if found is not None and (shortest is None
                                      or len(found) < len(shortest)):
                shortest = found
        return shortest

    def _beam(self, player, pieces, width, fanout):
        board = self.b
        frontier = [(0, player, dict(pieces), [])]
        done = None
        while frontier:
            kids = []
            for g, pl, pcs, path in frontier:
                if not pcs:
                    if done is None or g < len(done):
                        done = path
                    continue
                for sub in self.next_cancels(pl, pcs, want=fanout):
                    npl, npcs = pl, pcs
                    for direction in sub:
                        npl, npcs = board.step(npl, npcs, direction)
                    kids.append((g + len(sub), npl, npcs, path + sub))
            if not kids:
                break
            kids.sort(key=lambda kid: kid[0] + self.matching_bound(kid[1],
                                                                  kid[2]))
            frontier = kids[:width]
        return done

    # -- optimal-action targets --------------------------------------------
    def optimal_sets(self, player, pieces, plan):
        """The optimal press SET for each step of ``plan``.

        A plan alternates stretches where only the player moves with single
        presses that move a piece. Every shortest route across a stretch's
        side-effect-free walk graph ends on the same cell with the same board
        for the same cost, so all of them are equally optimal and labelling one
        arbitrary interleaving as "the" answer would train against the truth. A
        piece-moving press has no such freedom, and a stretch whose recorded
        route is not a shortest one is labelled literally."""
        board = self.b
        sets: list[list[str]] = []
        i, n = 0, len(plan)
        pieces = dict(pieces)
        while i < n:
            # Scan forward to the first press that moves a piece.
            j, pl, pcs = i, player, pieces
            while j < n:
                npl, npcs = board.step(pl, pcs, plan[j])
                if npcs is not pcs:
                    break
                pl, pcs, j = npl, npcs, j + 1
            walk, end = plan[i:j], pl
            if walk:
                dist = self._walk_dist(pieces, end)
                here = player
                literal = dist.get(here) != len(walk)
                for direction in walk:
                    best = [direction]
                    if not literal:
                        best = [d for d in _DIRS
                                if dist.get(board.walk_step(here, pieces, d),
                                            1 << 30) == dist[here] - 1]
                        if direction not in best:
                            best = [direction]
                    sets.append(best)
                    here = board.walk_step(here, pieces, direction)
            if j < n:
                sets.append([plan[j]])
            for k in range(i, min(j + 1, n)):
                player, pieces = board.step(player, pieces, plan[k])
                pieces = dict(pieces)
            i = j + 1
        return sets

    def _walk_dist(self, pieces, target):
        """Side-effect-free walk distances TO ``target``.

        Backwards over the same edge relation `_Board.walk_step` defines, which
        is directed: whether a step disturbs an air or a water depends on which
        way it is taken, so ``u -> v`` being inert does not make ``v -> u``
        inert."""
        board = self.b
        dist = {target: 0}
        queue = deque([target])
        while queue:
            cell = queue.popleft()
            for direction in _DIRS:
                dr, dc = _DELTA[direction]
                prev = (cell[0] - dr, cell[1] - dc)
                if (prev in dist or not board.inb(prev)
                        or prev in board.walls or prev in pieces):
                    # `step` does not police the player's own cell -- it is the
                    # caller's job -- so without the ``prev in pieces`` test the
                    # BFS would happily route a walk THROUGH a piece and report
                    # a distance shorter than any real route, which reads as
                    # "the recorded walk was not shortest" and silently drops
                    # every tie set on the level.
                    continue
                if board.walk_step(prev, pieces, direction) == cell:
                    dist[prev] = dist[cell] + 1
                    queue.append(prev)
        return dist


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class _Plan(list):
    """The press sequence, carrying the optimal SET for each of its steps, so a
    plan restored from disk still labels every step without rebuilding the pair
    tables."""

    def __init__(self, presses, optsets):
        super().__init__(presses)
        self.optsets = optsets


class CancelExpert(PSExpert):
    """Plans on `_Board` and certifies every plan on the interpreter.

    The plan memo, the level scoping and the snapshot/restore discipline are
    `PSExpert`'s; the search strategy and the disk cache are here."""

    #: `_key` is `PSExpert`'s full-grid canonical key, so plans are safe to share
    #: across levels and this stays False.
    scope_by_level = False
    #: ACTION is bound to nothing in this game.
    directions = list(_DIRS)

    def setup(self) -> None:
        self._disk = self._load_disk()

    def heuristic(self, eng) -> int:
        """Unused: `_search` plans on the native model, so `PSExpert._astar` --
        the only caller -- never runs."""
        raise NotImplementedError("CancelExpert plans on the native model")

    # -- planning ----------------------------------------------------------
    def _search(self, eng):
        board = _Board(eng, self.g)
        player, pieces = board.read(eng)
        if player is None:
            return None
        planner = _Planner(board, node_cap=self.node_cap)
        found = planner.solve(player, pieces)
        if found is None:
            return None
        found = self._certify(eng, found)
        if found is None:
            return None
        return _Plan(found, planner.optimal_sets(player, pieces, found))

    @staticmethod
    def _certify(eng, plan):
        """Replay ``plan`` through the real interpreter and return it truncated
        at the press that wins, or None if it never does. This is what makes the
        native model safe to plan on: a modelling slip costs a level, never a
        recorded trajectory that does not win."""
        snap = snapshot(eng)
        try:
            for i, direction in enumerate(plan):
                eng.step(direction)
                if eng.check_win():
                    return plan[:i + 1]
            return None
        finally:
            restore(eng, snap)

    def plan(self, eng, level: int | None = None):
        """`PSExpert.plan` (memo + restore discipline), with the level's START
        plan also kept on disk -- it is the only state any seed ever plans from,
        recovery being a RESET back to it, and re-deriving it costs a minute of
        table building in every process."""
        if eng.check_win():
            return []
        sig = sorted([r, c, o] for (r, c, o) in self._key(eng))
        cached = self._disk.get(level)
        if cached is not None and cached["start"] == sig:
            if cached["plan"] is None:
                return None
            return _Plan(cached["plan"], cached["optsets"])
        found = super().plan(eng, level)
        if level is not None and cached is None:
            self._disk[level] = {
                "start": sig,
                "plan": None if found is None else list(found),
                "optsets": None if found is None else found.optsets,
            }
            self._save_disk()
        return found

    # -- disk cache ---------------------------------------------------------
    @staticmethod
    def _load_disk() -> dict:
        """``{level: {"start": sig, "plan": [...] | None, "optsets": [...]}}``,
        or empty if unreadable -- an unparseable cache is a miss, never a
        crash."""
        try:
            raw = json.loads(PLAN_CACHE.read_text())
            return {int(k): v for k, v in raw.items()}
        except Exception:                                    # noqa: BLE001
            return {}

    def _save_disk(self) -> None:
        """Write atomically -- shards started together would otherwise interleave
        into a truncated file (see `parallelize_generator`)."""
        try:
            PLAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = PLAN_CACHE.with_suffix(f".json.tmp{os.getpid()}")
            tmp.write_text(json.dumps(
                {str(k): self._disk[k] for k in sorted(self._disk)}, indent=1))
            os.replace(tmp, PLAN_CACHE)
        except OSError:
            pass                                             # cache is optional


class CancelSolver(PSAStarSolver):
    game_id = "puzzlescript_cancel"
    game_name = GAME_NAME
    expert_cls = CancelExpert

    #: Record against the GAME FOLDER's adapter, not a bare `PuzzleScriptAdapter`.
    #: `games/ps:cancel/ps:cancel.py` raises the per-level step limit, and three
    #: levels need it: the adapter's 200-step default flips levels 5-7 to
    #: GAME_OVER partway through the expert's own plan, so a generator building
    #: its own adapter would tape six of eight levels and silently drop the rest.
    game_module_id = "ps:cancel"

    #: A* nodes per cancellation segment. The shipped levels solve inside a few
    #: thousand -- the pair-table heuristic is that tight -- so this is a
    #: runaway guard, not a tuning dial.
    node_cap = 200_000
    #: The longest plan is 290 presses; the ceiling only has to cover a re-plan
    #: after an epsilon detour, and `epsilon` is 0 for this family.
    max_steps = 400

    def prepare_expert(self, game, expert) -> None:
        """Build every level's plan before `discover_solvable` asks for it --
        the same work either way, but it makes the one-time cost visible as
        startup rather than as a mysteriously slow first seed."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """The full set of interchangeable presses at this step -- see
        `_Planner.optimal_sets`. Falls back to the press about to be taken, so
        no expert step ever ships unlabelled (the always-emit-optimal-targets
        rule)."""
        sets = getattr(plan, "optsets", None)
        if sets is not None and pi < len(sets):
            return sets[pi]
        return [plan[pi]] if pi < len(plan) else None


def _report() -> int:
    """Per-level plan report: length, the level's step budget, and how many
    steps carry a tie set. The budget column is the point of the report -- three
    levels need `games/ps:cancel`'s raised limit, and a plan that crept back
    over it would otherwise only show up as a level quietly going missing."""
    game = CancelSolver().make_game(0)
    expert = CancelExpert(game, node_cap=CancelSolver.node_cap)
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        found = expert.plan(game._engine, level)
        if found is None:
            print(f"level {level}: NO PLAN")
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(found)
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"level {level}: {len(found):4d} presses "
              f"(budget {game._max_steps}, {room}), "
              f"{ties:4d} steps with a tie set ({ties / max(1, len(found)):.0%})")
    print(f"total {total} presses over {game.n_levels} levels")
    return 0


if __name__ == "__main__":
    if "--fuzz" in sys.argv:
        sys.exit(1 if _fuzz_check() else 0)
    if "--plans" in sys.argv:
        sys.exit(_report())
    sys.exit(CancelSolver.main())
