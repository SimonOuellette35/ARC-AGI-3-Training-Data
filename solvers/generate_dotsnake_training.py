"""Generate Phase-1 training data for the PuzzleScript game ps:dotsnake
("Dotsnake" by Franklin P. Dyer -- eat every dot, then leave through the exit,
without ever crossing the tail you leave behind you).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, a native model of the interpreter, the search over it, and the
optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_dotsnake",
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
Win is ``No Dot`` AND ``All Player on Exit``: eat every dot on the board and
finish standing on the exit square. Four arrow keys; ACTION is bound to nothing
and is a literal no-op, so `DotsnakeExpert.directions` drops it.

  * **THE TAIL IS PERMANENT.** Every square the head leaves grows a Tail piece
    (``[VERTICAL Player No Tail] -> [VERTICAL Player Tail1]`` and the eight
    corner rules beside it), and ``[> Player|Tail] -> cancel`` refuses to walk
    into one. Nothing ever removes a tail. So a level is a **self-avoiding walk**
    from the head's start square to the exit that has to pass through every dot:
    no backtracking, no waiting (a refused press cancels the whole turn and
    changes nothing at all), and no second visit to any square.

  * **The exit must be the LAST square.** Tail, Exit and Dot share one collision
    layer, so the Tail dropped on a square the head leaves *replaces* the Exit
    that was there. Walking over the exit early does not merely waste it, it
    DELETES it, and ``All Player on Exit`` can never be met again. A self-avoiding
    walk visits the exit once, and that visit is the win -- which is why the
    search can treat the exit as an absorbing endpoint rather than as a square
    with a "do not step here yet" flag.

  * **Dots are eaten on arrival**, by the late rule ``[Player Dot] -> [Player]``,
    with no stomach, no cost and no choice. Dots are therefore pure *required
    vertices* of the walk, not a resource to manage.

  * **The six tail sprites are bookkeeping, not mechanics.** Tail1/Tail2 are the
    straight-through pieces, Tail3-6 the four corners, and which one is drawn is
    chosen by the eight rules that read the connecting neighbour's group
    (``UppyTail`` = connects up, etc.). Every one of them blocks identically, so
    the piece identity is invisible to the search -- `Model` reproduces it anyway,
    because the fuzz check compares whole grids.

  * **A blocked press is NOT a cancelled press.** Only a Wall or a Tail ahead
    cancels. Stepping into a ``Black`` filler square (the ``-`` regions that pad
    the ragged levels) or off the board edge is refused by the collision layer
    *after* the rules have run, so the head turns to face that way and drops a
    tail on its own square while staying put. It is never useful -- but it is a
    real state change, so the model has to reproduce it or the fuzz check fails
    on the cancel/no-cancel decision.

Expert solver
-------------
The mechanic above collapses to one clean combinatorial problem:

    a minimum-length simple path in the grid graph, from the head's square to
    the exit square, passing through every dot square.

`Search` solves it EXACTLY by IDA* over that abstraction (`Board`), with three
prunes that do all the work:

  * **Connectivity.** Flood the still-unvisited squares from the head; if any
    remaining dot -- or the exit -- is outside that flood, the walk has sealed
    itself off and the branch dies. This is what almost every wrong move in the
    game does. The flood is run on bit-parallel row masks (shift left/right for
    the column neighbours, shift by the row stride for the row ones), so a whole
    BFS layer is four big-int operations, and it stops the moment every terminal
    is covered.

  * **Degree.** A dot is entered and left, so it needs TWO still-usable
    neighbours (the head's own square counts as one when it is adjacent); the
    exit is the endpoint and needs one. A dot at the end of a corridor the walk
    has just sealed fails this before the flood does, for a fraction of the cost.

  * **The bound.** ``h`` is the larger of ``max_dot d(head,dot) + d(dot,exit)``
    and the MST of the metric closure over ``{head} u dots u {exit}``, both over
    the level's static walls-only distances. Both are admissible: the walk visits
    every terminal, so its length is at least the weight of a Hamiltonian path on
    the terminal set, and that is at least the MST's. Distances that ignore the
    tail can only be too short, never too long.

Grid graphs are bipartite, so every walk from the head to the exit has the same
length parity and the IDA* bound steps by 2. **All 12 levels solve, optimally, in
15 to 65 presses (423 total, ~1.5 s cold)**, and every plan is replayed through
the real interpreter by ``--plans`` before it ships.

Steps are labelled with the full optimal SET, computed exactly: at each step,
every legal alternative direction is re-searched at the plan's own total length,
and kept if a completion of that length exists. Only 17 of the 423 presses have a
second right answer -- which is the mechanic, not a gap in the labeller. A
self-avoiding walk admits no distance-to-goal field: the tail is state, so two
routes to the same square are never interchangeable, and the levels are laid out
as corridors precisely so the line of play is forced. Compare
[[collect-gnocchi-solver]], 38 of 939 on the same grounds.

Two art fixes ship with it, in ``data/puzzlescript_games/Dotsnake.txt`` (the
``--audit`` mode is the regression test; see `audit`):

  1. **The exit was invisible.** It shipped with the Background sprite AND the
     Background colours -- pixel-identical to bare floor at every cell size these
     boards render at. The one square the game asks you to reach was not in the
     frame. It is now a blue portal ring on light-blue corner studs.
  2. **The four heads were redrawn as a symmetry orbit.** Their single eye pixel
     sat off the facing axis, so a rotated or mirrored presentation (this game is
     in `_FLIP_GAMES`) drew a head that is not any sprite the game owns; and the
     eye sat on the sprite's middle row/column, which ``cell_px = 4`` does not
     sample, so it vanished on the two biggest boards. The heads now carry a
     symmetric PAIR of eyes on the leading edge, rot90 maps Up->Right->Down->Left
     exactly, the top-bottom mirror swaps Up and Down, and all four corners are
     transparent -- which is what lets the exit show through the head standing on
     it, i.e. through the winning frame.
"""

from __future__ import annotations

import itertools
import random
import sys
import time
from collections import deque
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from adapters.puzzlescript_adapter import (                    # noqa: E402
    PuzzleScriptAdapter, _render_cell_sprite)
from solvers.common.ps_astar import (                          # noqa: E402
    Plan, PSAStarSolver, PSExpert)

#: PuzzleScript game name (``data/puzzlescript_games/Dotsnake.txt``).
GAME_NAME = "Dotsnake"

#: Where each level's start plan is kept between processes. The searches are the
#: whole cost of generation and are seed-independent (the levels are fixed ASCII
#: maps; only the presentation is augmented), so without a file on disk every
#: shard `parallelize_generator` starts re-derives all twelve.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "dotsnake_plans.json"

#: The four engine directions, in the order the labeller reports them. ACTION is
#: bound to no rule in this game and does nothing at all, so it is not here.
DIRS = ("up", "down", "left", "right")

DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


# ---------------------------------------------------------------------------
# Native model of the interpreter (for the fuzz check)
# ---------------------------------------------------------------------------

#: The four tail groups from the LEGEND, as the piece numbers in each. A piece is
#: in a group when it CONNECTS that way: Tail1 is the vertical straight (up and
#: down), Tail2 the horizontal one, Tail3-6 the four corners.
LEFTY, RIGHTY, UPPY, DOWNY = {2, 3, 5}, {2, 4, 6}, {1, 3, 4}, {1, 5, 6}

#: The eight corner rules, keyed by the direction the head is leaving in, each as
#: ``(neighbour offset, the group that neighbour must be in, the piece to draw)``
#: and **listed in the .txt's own order**, because two of them can match at once
#: (the head's square can have a connecting tail on both sides) and the engine
#: lets the later rule overwrite the earlier one -- Tail, Exit and Dot are one
#: collision layer, so drawing a second piece replaces the first.
PIECE_RULES: dict[str, list] = {
    "left":  [((-1, 0), DOWNY, 3), ((1, 0), UPPY, 5)],
    "up":    [((0, -1), RIGHTY, 3), ((0, 1), LEFTY, 4)],
    "right": [((-1, 0), DOWNY, 4), ((1, 0), UPPY, 6)],
    "down":  [((0, -1), RIGHTY, 5), ((0, 1), LEFTY, 6)],
}


class Model:
    """Dotsnake's whole turn, natively.

    State is ``(head, facing, {square: tail piece}, frozenset of dots)`` -- the
    same four things `read_state` pulls out of the interpreter's grid, so the two
    can be compared cell for cell. `step` returns the next state, or ``None``
    when the turn is CANCELLED (the engine leaves the grid untouched).

    It exists for `selfcheck`. The search does not use it: `Board` is a further
    abstraction (the walk, without the sprites), and this model is what licenses
    that abstraction by showing the interpreter has nothing else in it.
    """

    def __init__(self, h: int, w: int, walls: set, blocked: set):
        self.h, self.w = h, w
        #: Squares holding a Wall -- the only thing that CANCELS a press.
        self.walls = walls
        #: Squares the head cannot stand on: walls plus the Black filler. Moving
        #: into Black is refused by the collision layer, not by a rule, so it is
        #: a blocked press and not a cancelled one.
        self.blocked = blocked

    def step(self, st, di: int):
        head, _facing, tails, dots = st
        d = DIRS[di]
        dr, dc = DELTA[d]
        r, c = head
        ahead = (r + dr, c + dc)
        onboard = 0 <= ahead[0] < self.h and 0 <= ahead[1] < self.w
        if onboard and (ahead in tails or ahead in self.walls):
            return None                       # [> Player|Tail] / [> Player|Wall]

        # The tail dropped on the square being left. Later rules overwrite
        # earlier ones; the No-Tail fallback only fires when none matched AND the
        # square is bare (a square can already hold a tail after a blocked press).
        piece = None
        for (nr, nc), group, p in PIECE_RULES[d]:
            q = tails.get((r + nr, c + nc))
            if q is not None and q in group:
                piece = p
        tails = dict(tails)
        if piece is not None:
            tails[head] = piece
        elif head not in tails:
            tails[head] = 1 if d in ("up", "down") else 2

        moved = onboard and ahead not in self.blocked
        head = ahead if moved else head
        nxt = (head, di, tails, dots - {head} if head in dots else dots)
        # A press that changes literally nothing leaves the engine's grid equal
        # to what it was, which the fuzz check reads as a cancelled turn.
        return None if nxt == st else nxt


def read_state(game, eng) -> tuple:
    """``(head, facing, {square: piece}, dots)`` off the interpreter's grid."""
    idx = game.obj_name_to_idx
    heads = {idx["headup"]: 0, idx["headdown"]: 1,
             idx["headleft"]: 2, idx["headright"]: 3}
    pieces = {idx[f"tail{i}"]: i for i in range(1, 7)}
    dot = idx["dot"]
    head = facing = None
    tails, dots = {}, set()
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            for o in cell:
                if o in heads:
                    head, facing = (r, c), heads[o]
                elif o in pieces:
                    tails[(r, c)] = pieces[o]
                elif o == dot:
                    dots.add((r, c))
    return head, facing, tails, frozenset(dots)


def model_from_engine(eng, game) -> tuple:
    """``(Model, state)`` for the level currently loaded."""
    idx = game.obj_name_to_idx
    wall, black = idx["wall"], idx["black"]
    walls, blocked = set(), set()
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if wall in cell:
                walls.add((r, c))
                blocked.add((r, c))
            elif black in cell:
                blocked.add((r, c))
    return Model(len(eng.grid), len(eng.grid[0]), walls, blocked), \
        read_state(game, eng)


# ---------------------------------------------------------------------------
# The search abstraction: a minimum-length covering walk
# ---------------------------------------------------------------------------

class Board:
    """A level as the only thing the search needs it to be: a grid graph, a
    start square, an exit square and a set of squares the walk must pass
    through.

    Squares are flat ``r * w + c`` indices and sets of them are big-int BITMASKS,
    which is what makes the connectivity prune cheap: one BFS layer of the
    still-free space is ``(M<<1) | (M>>1) | (M<<w) | (M>>w)``, masked to keep the
    column shifts from wrapping into the next row.
    """

    def __init__(self, h: int, w: int, free: int, start: int, exit_: int,
                 dots: int):
        self.h, self.w = h, w
        self.free = free            # squares the head may stand on
        self.start = start
        self.exit = exit_
        self.dots = dots
        n = h * w
        self.full = (1 << n) - 1
        self.col0 = sum(1 << (r * w) for r in range(h))
        self.colw = sum(1 << (r * w + w - 1) for r in range(h))
        # Free 4-neighbours of every square, and the same as (index, direction)
        # pairs in DIRS order so move generation never re-derives coordinates.
        self.nbr = [0] * n
        self.moves: list[list] = [[] for _ in range(n)]
        for r in range(h):
            for c in range(w):
                i = r * w + c
                for d in DIRS:
                    dr, dc = DELTA[d]
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    j = rr * w + cc
                    if (free >> j) & 1:
                        self.nbr[i] |= 1 << j
                        self.moves[i].append((d, j))
        # Walls-only distances from every terminal, for the admissible bound.
        self.terminals = bits(dots) + [exit_]
        self.dist = {t: self._bfs(t) for t in self.terminals}

    def _bfs(self, src: int) -> list:
        inf = 1 << 20
        dist = [inf] * (self.h * self.w)
        dist[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            m = self.nbr[u]
            while m:
                b = m & -m
                m ^= b
                v = b.bit_length() - 1
                if dist[v] == inf:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    def expand(self, m: int) -> int:
        """Every square 4-adjacent to the set ``m`` (plus ``m`` itself's
        neighbours only -- the caller ORs ``m`` back in)."""
        w = self.w
        return (((m << 1) & ~self.col0) | ((m >> 1) & ~self.colw)
                | (m << w) | (m >> w)) & self.full


def bits(m: int) -> list:
    """The set bit indices of ``m``, ascending."""
    out = []
    while m:
        b = m & -m
        m ^= b
        out.append(b.bit_length() - 1)
    return out


def board_from_engine(eng, game) -> Board:
    """Read the loaded level into a `Board`. Walls and Black are both simply
    "not free" here: the search never wants to step on either, and the one way
    they differ (Wall cancels the turn, Black merely refuses the move) matters
    only to `Model`."""
    idx = game.obj_name_to_idx
    wall, black, exit_i, dot_i = (idx["wall"], idx["black"],
                                  idx["exit"], idx["dot"])
    heads = {idx[n] for n in ("headup", "headdown", "headleft", "headright")}
    h, w = len(eng.grid), len(eng.grid[0])
    free = dots = 0
    start = exit_ = None
    for r in range(h):
        for c in range(w):
            cell = eng.grid[r][c]
            if wall in cell or black in cell:
                continue
            i = r * w + c
            free |= 1 << i
            if dot_i in cell:
                dots |= 1 << i
            if exit_i in cell:
                exit_ = i
            if cell & heads:
                start = i
    if start is None or exit_ is None:
        return None
    return Board(h, w, free, start, exit_, dots)


class Search:
    """Exact minimum-length covering walk, by IDA*.

    ``node_cap`` bounds a single bound-iteration, not the whole ladder: a level
    that blows it is reported unsolved rather than silently truncated (none of
    the twelve comes near it -- the worst is ~3000 nodes).
    """

    def __init__(self, board: Board, node_cap: int = 2_000_000):
        self.b = board
        self.node_cap = node_cap
        self.nodes = 0
        self.total_nodes = 0

    # -- the admissible bound --------------------------------------------------
    def h(self, pos: int, rem: int) -> int:
        """A lower bound on the presses left: the walk still has to reach the
        exit through every remaining dot.

        Two bounds, both over the level's static walls-only distances (ignoring
        the tail, which can only make a route longer):

          * ``d(pos, dot) + d(dot, exit)`` for the worst remaining dot -- the walk
            passes through it, so it is at least that long;
          * the MST of the metric closure over ``{pos} u remaining dots u
            {exit}``. The walk visits those terminals in some order, so its
            length is at least the weight of that Hamiltonian path on the
            terminal set, which is at least the MST's.
        """
        b = self.b
        best = b.dist[b.exit][pos]
        rest = bits(rem)
        if not rest:
            return best
        for t in rest:
            v = b.dist[t][pos] + b.dist[t][b.exit]
            if v > best:
                best = v
        nodes = rest + [b.exit]                     # Prim's, rooted at ``pos``
        key = [b.dist[t][pos] for t in nodes]
        used = [False] * len(nodes)
        total = 0
        for _ in range(len(nodes)):
            mi, mv = -1, 1 << 20
            for i, k in enumerate(key):
                if not used[i] and k < mv:
                    mi, mv = i, k
            used[mi] = True
            total += mv
            row = b.dist[nodes[mi]]
            for i, t in enumerate(nodes):
                if not used[i] and row[t] < key[i]:
                    key[i] = row[t]
        return max(best, total)

    # -- the prunes ------------------------------------------------------------
    def feasible(self, pos: int, visited: int, rem: int) -> bool:
        """False when this state provably cannot be completed.

        Degree first (it is a popcount per remaining dot and catches the same
        sealed corridors sooner), then the flood."""
        b = self.b
        avail = (b.free & ~visited) | (1 << pos)
        m = rem
        while m:
            bit = m & -m
            m ^= bit
            # A dot is passed THROUGH: one neighbour to come in by, another to
            # leave by. The head's own square counts, it is where a walk can
            # come from.
            if bin(b.nbr[bit.bit_length() - 1] & avail).count("1") < 2:
                return False
        if not b.nbr[b.exit] & avail:
            return False
        need = rem | (1 << b.exit)
        reach = 1 << pos
        while True:
            if reach & need == need:
                return True
            nxt = (reach | b.expand(reach)) & avail
            if nxt == reach:
                return False
            reach = nxt

    # -- depth-first search under a length bound --------------------------------
    def dfs(self, pos: int, visited: int, rem: int, g: int, bound: int):
        """The shortest completion of length ``bound - g`` from here, as a list
        of directions, or None. ``pos`` is never the exit: arriving there ends
        the walk, and the caller returns before recursing."""
        self.nodes += 1
        self.total_nodes += 1
        if self.nodes > self.node_cap:
            raise _Overrun
        b = self.b
        if g + self.h(pos, rem) > bound:
            return None
        if not self.feasible(pos, visited, rem):
            return None
        cand = []
        for d, j in b.moves[pos]:
            if (visited >> j) & 1:
                continue
            if j == b.exit:
                if rem == 0 and g + 1 <= bound:
                    return [d]
                continue                    # stepping on the exit early KILLS it
            nrem = rem & ~(1 << j)
            est = self.h(j, nrem)
            if g + 1 + est <= bound:
                cand.append((est, d, j, nrem))
        cand.sort()                         # most promising child first
        for _est, d, j, nrem in cand:
            found = self.dfs(j, visited | (1 << j), nrem, g + 1, bound)
            if found is not None:
                return [d] + found
        return None

    def solve(self, verbose: bool = False):
        """The optimal plan, or None. Bounds step by 2: the grid graph is
        bipartite, so every walk from the start to the exit has the parity of
        their distance and the odd bounds cannot hold a solution."""
        b = self.b
        bound = self.h(b.start, b.dots)
        if (bound & 1) != (b.dist[b.exit][b.start] & 1):
            bound += 1
        limit = bin(b.free).count("1")            # a simple path cannot be longer
        while bound <= limit:
            self.nodes = 0
            try:
                found = self.dfs(b.start, 1 << b.start, b.dots, 0, bound)
            except _Overrun:
                return None
            if verbose:
                print(f"    bound {bound}: {self.nodes} nodes")
            if found is not None:
                return found
            bound += 2
        return None

    # -- optimal-action sets ---------------------------------------------------
    def optsets(self, plan: list) -> list:
        """Every equally-shortest press at each step of ``plan``.

        Exact, and it can afford to be: the plan's total length ``L`` is optimal,
        so an alternative at step ``i`` is optimal exactly when a completion of
        length ``L - i - 1`` exists from the square it leads to -- one more
        bounded DFS, at a bound so tight the prunes close it immediately.

        Ties are rare here (17 of 423 presses) and that is the mechanic: a
        self-avoiding walk has no distance-to-goal field, because the tail is
        state and two routes to the same square leave different boards behind
        them. The plan's own press is always in the set, so no step ever ships
        unlabelled."""
        b = self.b
        total = len(plan)
        pos, visited, rem = b.start, 1 << b.start, b.dots
        out = []
        for i, taken in enumerate(plan):
            best = []
            for d, j in b.moves[pos]:
                if (visited >> j) & 1:
                    continue
                if j == b.exit:
                    if rem == 0 and i + 1 == total:
                        best.append(d)
                    continue
                if d == taken:
                    best.append(d)          # optimal by construction
                    continue
                self.nodes = 0
                try:
                    ok = self.dfs(j, visited | (1 << j), rem & ~(1 << j),
                                  i + 1, total) is not None
                except _Overrun:
                    ok = False              # unproven is not optimal
                if ok:
                    best.append(d)
            out.append([d for d in DIRS if d in best])
            dr, dc = DELTA[taken]
            r, c = divmod(pos, b.w)
            pos = (r + dr) * b.w + (c + dc)
            visited |= 1 << pos
            rem &= ~(1 << pos)
        return out


class _Overrun(Exception):
    """A single IDA* iteration blew the node cap."""


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class DotsnakeExpert(PSExpert):
    """`PSExpert` with the whole search replaced: the mechanic is exactly a
    minimum-length covering walk (see the module docstring), so planning happens
    on `Board` rather than by stepping the interpreter, and `heuristic` -- the
    hook a stepping A* would call -- is never reached.

    What is inherited is everything around the search: the in-memory memo, the
    on-disk start-plan cache with its staleness check, and the snapshot/restore
    discipline `PSExpert.plan` wraps `_search` in."""

    directions = list(DIRS)          # ACTION is bound to no rule in this game
    plan_cache_path = PLAN_CACHE

    def __init__(self, game, node_cap: int = 400_000, weight: int = 1):
        super().__init__(game, node_cap=node_cap, weight=weight)
        self._how: dict = {}         # level -> one line for `describe`
        self._level: int | None = None

    def heuristic(self, eng) -> int:                        # pragma: no cover
        raise AssertionError("DotsnakeExpert plans on Board, not on the engine")

    def _search(self, eng):
        board = board_from_engine(eng, self.g)
        if board is None:
            return None
        started = time.time()
        search = Search(board)
        plan = search.solve()
        if plan is None:
            return None
        sets = search.optsets(plan)
        self._how[self._level] = (f"{search.total_nodes} nodes "
                                  f"{time.time() - started:.2f}s")
        return Plan(plan, sets)

    def plan(self, eng, level: int | None = None):
        """`PSExpert.plan`, remembering which level `describe` is reporting on."""
        self._level = level
        return super().plan(eng, level)

    def describe(self, level) -> str:
        """How this level's plan was found, for ``--plans``; "cached" when it
        came off disk and nothing ran."""
        return self._how.get(level, "cached")


# ---------------------------------------------------------------------------
# BaseSolver harness
# ---------------------------------------------------------------------------

class DotsnakeSolver(PSAStarSolver):
    game_id = "puzzlescript_dotsnake"
    game_name = GAME_NAME
    expert_cls = DotsnakeExpert

    #: Plans run to 65 presses, so the adapter's 200-press default leaves a 3x
    #: margin for a policy that wanders and no level needs a raised limit (the
    #: RESET in the recovery prefix zeroes that counter, so only the post-reset
    #: plan is charged against it).
    max_steps = 200

    def optimal_for(self, expert, level: int, plan, pi: int):
        """The optimal press set at this step, off the plan's own optsets.
        Falls back to the press about to be taken so no expert step ever ships
        unlabelled -- `train_policy` v2 supervises ``optimal`` only, so a step
        without one contributes nothing to the loss."""
        sets = getattr(plan, "optsets", None)
        if sets is not None and pi < len(sets) and sets[pi]:
            return sets[pi]
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Model self-check (fuzz against the real interpreter)
# ---------------------------------------------------------------------------

def selfcheck(trials: int = 60, steps: int = 60, verbose: bool = True) -> int:
    """Audit the claim the solver rests on: `Model` reproduces the interpreter
    EXACTLY, so `Board` -- which throws even the tail sprites away -- is throwing
    away nothing that can bite.

    On random rollouts from every level start, the settled state (head, facing,
    every tail piece, every dot), the cancel/no-cancel decision and the win flag
    all have to agree. Random play is the fuzz: it walks into walls and into its
    own tail (the two cancels), presses into the Black filler and off the board
    edge (the blocked-but-not-cancelled press), and runs long enough to seal
    itself into a corner -- corners a plan never visits."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        cancels = wins = 0
        for t in range(trials):
            game.set_level(level)
            model, state = model_from_engine(eng, game._game)
            rng = random.Random(f"dotsnake:selfcheck:{level}:{t}")
            for _ in range(steps):
                before = [[set(c) for c in cell] for cell in eng.grid]
                di = rng.randrange(len(DIRS))
                eng.step(DIRS[di])
                frozen = eng.grid == before
                predicted = model.step(state, di)
                expected = state if predicted is None else predicted
                actual = read_state(game._game, eng)
                if actual != expected:
                    bad += 1
                    print(f"  L{level}: state divergence on {DIRS[di]}")
                    for i, (a, b) in enumerate(zip(expected, actual)):
                        if a != b:
                            print(f"    field {i}: model {a} engine {b}")
                    break
                if (predicted is None) != frozen:
                    bad += 1
                    print(f"  L{level}: cancel divergence on {DIRS[di]} "
                          f"(engine frozen={frozen}, model cancel="
                          f"{predicted is None})")
                    break
                if eng.check_win() != won(game._game, eng, expected):
                    bad += 1
                    print(f"  L{level}: win divergence on {DIRS[di]}")
                    break
                if predicted is None:
                    cancels += 1
                else:
                    state = predicted
                if eng.check_win():
                    wins += 1
                    break
        if verbose:
            print(f"  L{level}: {'OK' if not bad else 'VIOLATIONS'} "
                  f"({trials} rollouts, {cancels} cancelled turns, "
                  f"{wins} accidental wins)")
        if bad:
            break
    return bad


def won(game, eng, state) -> bool:
    """The game's own win condition read off a `Model` state: no dot left, and
    the head standing on an Exit. The Exit is static as long as it exists -- and
    the moment the walk leaves it, the tail replaces it -- so this reads the
    exits from the live grid rather than carrying them in the state."""
    head, _facing, _tails, dots = state
    if dots:
        return False
    exits = {(r, c)
             for r, row in enumerate(eng.grid)
             for c, cell in enumerate(row)
             if game.obj_name_to_idx["exit"] in cell}
    return head in exits


# ---------------------------------------------------------------------------
# Render audit
# ---------------------------------------------------------------------------

#: Every cell composition a level can show, as the object stack that draws it.
#: Tail-on-exit and dot-on-exit are absent on purpose: Tail, Exit and Dot are one
#: collision layer, so those stacks cannot occur (and the head-on-exit case, which
#: CAN, is the winning frame).
_AUDIT_CASES = {
    "floor": ["background"],
    "wall": ["background", "wall"],
    "black filler": ["background", "black"],
    "exit": ["background", "exit"],
    "dot": ["background", "dot"],
    "head up": ["background", "headup"],
    "head down": ["background", "headdown"],
    "head left": ["background", "headleft"],
    "head right": ["background", "headright"],
    "head up on exit": ["background", "exit", "headup"],
    "head down on exit": ["background", "exit", "headdown"],
    "head left on exit": ["background", "exit", "headleft"],
    "head right on exit": ["background", "exit", "headright"],
    "tail vertical": ["background", "tail1"],
    "tail horizontal": ["background", "tail2"],
    "tail up-left": ["background", "tail3"],
    "tail up-right": ["background", "tail4"],
    "tail down-left": ["background", "tail5"],
    "tail down-right": ["background", "tail6"],
    "head up on own tail": ["background", "tail1", "headup"],
    "head down on own tail": ["background", "tail1", "headdown"],
    "head left on own tail": ["background", "tail2", "headleft"],
    "head right on own tail": ["background", "tail2", "headright"],
}

#: The one collision class this game is allowed to have, and it is deliberate: a
#: head sharing its square with a tail is drawn as the bare head.
#:
#: That stack exists only after a BLOCKED press -- stepping into the Black filler
#: or off the board edge, where the rules run (so a tail is dropped on the head's
#: own square) but the collision layer then refuses the move. It carries NO
#: information: the cancel rule ``[> Player|Tail]`` reads the square AHEAD, so a
#: tail underfoot blocks nothing, and its only other effect is to suppress the
#: ``No Tail`` fallback later, i.e. to change which of six interchangeable tail
#: SPRITES is drawn when the head finally leaves. Nothing an agent could do
#: differently depends on seeing it, and buying it back would cost the corner
#: transparency that makes the exit visible under the head -- which is the
#: winning frame. Asserted to still collide, so that if the art ever changes this
#: exception is reported as stale instead of quietly covering a new bug.
_AUDIT_ALLOWED = {frozenset({h, f"{h} on own tail"})
                  for h in ("head up", "head down", "head left", "head right")}


def audit(verbose: bool = True) -> int:
    """Check that every cell composition is pixel-distinct at every cell size the
    levels use.

    This is the regression test for the two art fixes in the .txt. The exit
    shipped as a byte-for-byte copy of the background sprite, in the background's
    own colours, so "floor" and "exit" collided at every size and the game's goal
    was not in the frame at all; and every head is transparent at the four corners
    the exit paints, which is what keeps "head on exit" distinct from "head".
    Composition, not object: these bugs live in the stack. Returns the number of
    colliding pairs."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    parsed = game._game
    layers = game._engine._obj_layers
    sizes = set()
    for level in range(game.n_levels):
        game.set_level(level)
        h, w = len(game._engine.grid), len(game._engine.grid[0])
        sizes.add(max(1, min(64 // h, 64 // w)))

    def stack(names):
        objs = [(layers.get(parsed.obj_name_to_idx[n], -1), parsed.objects[n])
                for n in names]
        objs.sort(key=lambda x: x[0])
        return objs

    bad = 0
    for px in sorted(sizes):
        blocks = {k: _render_cell_sprite(stack(v), px)
                  for k, v in _AUDIT_CASES.items()}
        for a, b in itertools.combinations(_AUDIT_CASES, 2):
            same = bool((blocks[a] == blocks[b]).all())
            if frozenset({a, b}) in _AUDIT_ALLOWED:
                if not same:                 # the exception has gone stale
                    bad += 1
                    print(f"  cell_px={px}: {a} and {b} are now distinct -- "
                          f"drop them from _AUDIT_ALLOWED")
            elif same:
                bad += 1
                print(f"  cell_px={px}: {a} and {b} render identically")
        if verbose:
            print(f"  cell_px={px}: {len(_AUDIT_CASES)} cell types, "
                  f"{len(_AUDIT_ALLOWED)} allowed collisions, "
                  f"{'all distinct' if not bad else 'COLLISIONS'}")
    return bad


# ---------------------------------------------------------------------------
# Plan report
# ---------------------------------------------------------------------------

def _plan_report() -> None:
    """Print every level's plan, how it was found and how many of its steps have
    more than one right answer -- the quick "is this game still fully solved"
    check. Every plan is replayed through the real interpreter, so this is also
    the search's end-to-end test."""
    solver = DotsnakeSolver()
    game, expert, solvable = solver._ensure(0)
    total = ties = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level}: UNSOLVED")
            continue
        for direction in plan:                          # interpreter-verify it
            eng.step(direction)
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        step_ties = sum(len(s) - 1 for s in sets)
        total += len(plan)
        ties += step_ties
        print(f"  L{level:2d}: {len(plan):3d} presses  win={eng.check_win()}  "
              f"{expert.describe(level):>20}  {step_ties:3d} tie-presses")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second right answer")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        violations = selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = audit()
        print(f"audit: {collisions} colliding cell types")
        sys.exit(1 if collisions else 0)
    if "--plans" in sys.argv:
        _plan_report()
        sys.exit(0)
    sys.exit(DotsnakeSolver.main())
