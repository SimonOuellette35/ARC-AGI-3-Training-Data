"""Generate Phase-1 training data for the PuzzleScript game ps:variations
("Variations", arrogant.gamer, www.arrogantgamer.com).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, a native model of them, the exact distance FIELD that model is
searched with, the proof that the plans are shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_variations",
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
action (post rotation remap), i.e. the button an agent presses in the augmented
view, so replaying the recorded actions reproduces the recorded frames exactly.
Every expert step carries the full set of equally-optimal presses.

THE GAME: a sokoban where the crates are TELEPATHIC, and the ACTION key
re-wires which of them can hear each other
-----------------------------------------------------------------------------
Every square on the board is one of four KINDS, drawn as four rotations of the
same little glyph, and the win is ``all A on Z`` -- every square standing on a
goal. Four rules, and between them they make a game that looks like sokoban for
one press and then stops being one:

    [ > player | A ] -> [ > player | > A ]              ( an ordinary push )
    [ action player | A_k ] -> [ action player | action A_k ]
    [ action A_k ] -> [ A_{k+1} ]                       ( ACTION turns a kind )
    [ > A_k ] [ A_k ] -> [ > A_k ] [ > A_k ]            ( ...and the game )

* **Pushing one square pushes EVERY square of its kind, anywhere on the board.**
  Not "every adjacent one", not "every one in the same room" -- every one, with
  no reference to distance, walls or line of sight. Five of the nine levels have
  the player sealed in a field with no goal in it at all, and are solved
  entirely by shoving a square you *can* reach in order to move the squares you
  cannot.
* **The remote copies DIRECTION, not displacement.** All of them step the same
  way at the same time, so a group of one kind keeps its shape -- until one of
  them is blocked, which is what shears the group and is what every level is
  built around.
* **A blocked push still moves the remotes.** The chain that starts at the
  player is resolved on its own: if the square the player leans on cannot move,
  the player does not move either, and every OTHER square of that kind moves
  anyway. Level 2 is the tutorial for exactly that ("you can push a stationary
  block to move a neighbour").
* **ACTION turns every square ORTHOGONALLY ADJACENT to a player one step round
  the cycle** A_1 -> A_2 -> A_3 -> A_4 -> A_1, and nothing else happens that
  turn. That is the only way to change which squares are linked -- so ACTION is
  a real move here, and the search branches on five presses, not four.
* **A press into a wall with nothing adjacent is a total no-op**, because the
  remote rule needs a square that is already being pushed to have something to
  copy. This game has no "wait" button.
* **One level has TWO players.** Level 6 draws a second one inside the maze;
  both receive the input force, both push, and both turn their own neighbours on
  ACTION. It is also the one level with no macro decomposition -- see
  `_Board.macro_edges`.

Nine levels, 7x6 to 11x11, two to eight squares each and exactly as many goals
as squares. The last two ("RC quad", "RC double") are the pure form: a wall
cross splits the board into rooms, the player is shut in one of them, and the
squares in the other rooms are steered blind -- and because the wall runs the
full width, a remote square's KIND can never be changed again, so which of them
move together is fixed for the whole level.

Two edits to `data/puzzlescript_games/Variations.txt` were needed to make this
game playable at all, both documented in a header comment in that file:

1. **The remote rule did not fire.** The adapter's multi-bracket "overlap
   preference" bound both brackets of ``[ > A_k ] [ A_k ]`` to the one square
   already being pushed, so the mechanic above did not exist: levels 2-6 were
   unwinnable and levels 0 and 1 -- which differ ONLY in whether their two
   squares share a kind -- enumerated to byte-identical state graphs. Asking the
   second bracket for a ``stationary`` square removes the overlap and reaches
   the same fixpoint reference PuzzleScript reaches by looping the original
   rule. See gotcha 1 in `ps-engine-render-gotchas`; the alternative, narrowing
   the heuristic in the adapter, would touch 28+ other corpus games.
2. **Nothing on the board was distinguishable.** All art was black-on-white,
   black is ARC 5, and 5 is also the colour `_render_frame` letterboxes with --
   so the walls were the picture frame. The player and all four square kinds
   were 5x5 sprites with no transparent pixel, and the goal was a white square
   with one dark pixel at row/col 2, so a square standing on a goal (the win
   condition) rendered identically to one standing on floor. On the 17-wide
   level, which renders at cell_px 3 and samples sprite rows/cols {0, 2, 4}, all
   four kinds, the player and the walls were the SAME solid block. See
   `--audit`.

THE MODEL
---------
`_Board` is the mechanic as one function of ``(state, press)``, over flat
``r * w + c`` cells. A state is ``(player cells..., square codes...)`` where a
code is ``cell * 4 + (kind - 1)``; walls and goals never change and live on the
board. Every force in a turn points the same way -- there is no rule here that
produces a second direction -- which collapses the interpreter's general
chain-and-conflict resolution into: forced bodies form maximal runs along the
press direction, and a run moves iff the cell past its head is on the board and
empty. `--selfcheck` fuzzes that against the real interpreter.

THE SEARCH
----------
A ladder, because the nine levels differ by four orders of magnitude in size:

1. **The exact FIELD** (`_Board.field`) -- a forward BFS by LAYER, stopped the
   moment a win appears, then a backward pass down the layers marking every
   state from which a win is still exactly ``d_star - depth`` presses away. The
   backward pass re-derives the edges instead of storing predecessors: no press
   in this game has an enumerable inverse (it turns kinds, shears a group and
   silently drops the squares that were blocked), so a backward sweep must run
   over forward edges either way, and re-deriving them costs one pass while
   storing them costs a list per state -- which is the difference between
   fitting in memory and not. The field gives a provably shortest plan, the
   EXACT set of equally-optimal presses at every step, and a proof of deadness
   for `record_level`'s recovery path.
2. **A* over MACROS** (`_Board.macro_edges`) for the levels whose closure does
   not fit. A press either pushes, or turns kinds, or moves nothing but the
   player -- so every plan is ``walk* press walk* press ...`` and the unit worth
   searching is "walk somewhere, then press once". The player's position between
   two presses stops being a state at all, which is what makes these boards
   searchable.
3. **The bound that makes rung 2 work is `_Rooms`.** Walls split six of the nine
   levels into regions no body can leave, and a region with no player in it
   cannot see WHICH square was pushed -- only that some kind was shoved some way.
   Its own evolution is therefore a tiny automaton over the 16 ``(kind,
   direction)`` events, so its exact distance-to-solved is simply enumerated, and
   the plan is at least the largest of them. On level 8 that is 16 against the 4
   a position-only bound gives, and it is the difference between a beam's guess
   of 47 presses and a PROOF that 29 is the shortest there is. The room holding
   the player is measured the same way under the real press dynamics.
   Rooms are not independent of EACH OTHER, though -- they hear the same event,
   so level 7's two upper rooms both want their kind-4 square moved and in
   opposite directions. Tabling PAIRS of rooms is what sees that, and it is what
   took level 7 from unsolved to solved.
4. **Exact tie sets without a field** (`measured_optsets`). Knowing a plan is
   shortest turns "which presses are equally good" into a decision -- can this
   successor still finish in the presses that are left -- which `_macro_within`
   answers by pruning every node the admissible bound says cannot. Measured from
   the END of the plan backwards under one clock, because that is where the bound
   is tight and the answers are cheap; whatever the clock does not reach falls
   back to `rejoin_optsets`, a sound floor.
5. **A weighted macro A* / macro beam**, for whatever rung 2 cannot close. Those
   levels get a win and no claim about its length, `shorten` splices out whatever
   a bounded BFS can cross faster, and their tie sets are the rejoin floor.

WHERE IT LANDS
--------------
All nine levels win, 264 presses. Eight are provably shortest with COMPLETE
optimal-action sets: levels 0-4 and 6 off the exact field, levels 5 (48 presses)
and 8 (29) off macro A* at weight 1 with the sets measured. Level 7 -- "RC quad",
eight squares in four rooms -- takes 56 presses off a weighted macro A*: a win,
with no claim about its length, and rejoin tie sets. Its three remote rooms ARE
jointly solvable in 20 events (measured by BFS over the 216M-state product of the
three room automata), so the level is winnable and the gap is the BOUND: most of
its cost is producing each event -- rotating a handle to the right kind and
walking round it -- which no room-based bound can see.

Modes: ``--plans`` (per-level report, replayed through the interpreter),
``--selfcheck`` (model vs interpreter, plans re-proved, tie sets brute-forced,
recovery boards), ``--audit`` (render every cell composition at every cell size
the levels use), ``--symmetry`` (every plan replayed under all four rotations).
"""

from __future__ import annotations

import heapq
import random
import sys
import time
from collections import deque
from itertools import permutations as _permutations, product as _product
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from arcengine import ActionInput, GameAction, GameState        # noqa: E402
from adapters.puzzlescript_adapter import PuzzleScriptAdapter   # noqa: E402
from solvers.common.ps_astar import (                           # noqa: E402
    Plan, PSAStarSolver, PSExpert, screen_action,
)
from utils.rotation import inverse_remap_action_full            # noqa: E402

GAME_NAME = "Variations"

#: Engine presses, in the order ties are broken. ACTION is a REAL move here (it
#: is the only way to change which squares are linked), so unlike most of the
#: sokoban-shaped ps: games this search branches on five, not four.
DIRS: tuple[str, ...] = ("up", "down", "left", "right", "action")
_ACTION = 4

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Disk cache of every level's start plan AND its optimal-action sets. Level 5's
#: field alone is minutes and tens of millions of states, and it is
#: seed-independent (the engine state after reset is the same for every seed --
#: only the presentation is augmented), so without a file on disk every shard a
#: `parallelize_generator` fan-out starts would re-derive all nine. Delete the
#: file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "variations_plans.json"

#: The adapter cuts a level off at 200 presses, and `record_level`'s exploration
#: prefix ends in a `set_level` that resets that counter -- so a plan longer than
#: this could not be replayed even from a clean start.
MAX_PLAN = 190

#: States the exact field may hold before it gives up and the ladder falls
#: through to the macro search. The field is tried FIRST because it is the only
#: rung that produces COMPLETE optimal-action sets rather than a sound floor, and
#: it is cheap exactly when it works: every level it closes here does so in under
#: four seconds. This ceiling is what bounds the wasted sweep on the ones it
#: cannot -- at roughly 220 bytes a state that is a few hundred MB and a few tens
#: of seconds, once, cached to disk.
FIELD_CAP = 3_000_000
#: What a RE-plan (an epsilon detour, a recovery board) is allowed to spend. A
#: re-plan happens inside the recording loop, per seed, so it must not cost what
#: the one-off startup sweep costs.
FIELD_CAP_FAST = 250_000

#: ``(weight, edge cap, seconds)`` rungs, smallest weight first: weight 1 is a
#: PROOF of shortest, above 1 is only a win. Each rung carries its OWN clock --
#: a shared one is spent entirely by rung 1 on the levels rung 1 cannot close,
#: which is exactly where the later rungs were needed.
#:
#: ``_MACRO_*`` searches "walk somewhere, then press once" (`_Board.macro_edges`)
#: and is the main ladder; ``_ASTAR_*`` is the primitive fallback for the
#: two-player board, which has no macro decomposition.
#: The weight-1 macro cap is deliberately huge. With `_Rooms` in the bound the
#: hard levels close at 30-40 million macro edges and a few hundred MB, and every
#: one of them is the difference between a plan that is PROVED shortest with
#: exact tie sets and a beam's guess -- so the budget is spent there rather than
#: on the weighted rungs, which only ever return a win.
_MACRO_FULL = ((1, 40_000_000, 900.0), (2, 4_000_000, 120.0),
               (5, 4_000_000, 120.0))
_MACRO_FAST = ((3, 30_000, 6.0),)
_ASTAR_FULL = ((1, 4_000_000, 600.0), (2, 2_000_000, 240.0),
               (4, 1_500_000, 240.0))
_ASTAR_FAST = ((3, 100_000, 6.0),)
#: ``(width, seconds)`` rungs for the macro beam that follows A*. Every one is
#: run and the shortest answer kept.
_BEAM_FULL = ((200, 120.0), (2_000, 300.0), (10_000, 600.0))
_BEAM_FAST = ((200, 8.0),)
#: Stop a beam deduping past this many states.
BEAM_SEEN_CAP = 3_000_000

#: Budget for ONE distance query inside `measured_optsets`, which is how a
#: ``macro:w1`` level gets tie sets as complete as a field level's.
MEASURE_CAP = 3_000_000
MEASURE_SECONDS = 120.0
#: ...and for a whole plan's worth of them. Spent from the END of the plan
#: backwards; whatever is left over falls back to `rejoin_optsets`, which is a
#: sound floor.
MEASURE_TOTAL_SECONDS = 1_200.0

#: How far `shorten` looks for a faster crossing between two states of a plan
#: that is not known to be shortest. Five presses branch to at most 5**8 states,
#: which is a fraction of a second per plan position.
SHORTEN_HORIZON = 8

#: How far `rejoin_optsets` looks for an alternative press to rejoin the plan.
#: Calibrated against the levels that DO have a field: see `--selfcheck`, which
#: compares the rejoin sets against the exact ones on every field step.
REJOIN_HORIZON = 9

_INF = 1 << 20


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic, plus the searches.

    Cells are flat ``r * w + c``. A STATE is a single flat tuple

        (player cell, ..., square code, ...)

    with the players first (there are one or two, fixed per level) and then the
    squares, each encoded as ``cell * 4 + (kind - 1)``. Both halves are kept
    sorted, which is canonical because two bodies can never share a cell -- they
    are all on one collision layer. Walls and goals are never created or
    destroyed by any rule, so they live on the board rather than in the state.

    THE MECHANIC, as one press:

      1. Every player is forced in the pressed direction.
      2. If a player is stepping into a square, that square is forced -- the
         ordinary sokoban rule, and the ONLY way a force ever enters the game.
      3. Every square whose KIND matches one forced in step 2 is forced, wherever
         it is. This is the remote control, and it reads nothing about distance,
         walls or line of sight.
      4. Resolution. Every force this turn points the same way, so the
         interpreter's chain tracing collapses to: the forced bodies form maximal
         RUNS along the press direction, and a run moves iff the cell past its
         head is on the board and holds no body. A run that cannot move does not
         move at all -- which is what leaves the player standing still while its
         remote twins walk away.

    ACTION is separate and moves nothing: every square orthogonally adjacent to
    a player advances one kind. The four rotate rules in the .txt are written
    A_4 -> A_1 first, so the kind produced by one is never re-matched by the
    next and the whole set turns exactly once.
    """

    __slots__ = ("h", "w", "np", "wall", "goal", "_edge", "_goalmask", "_int")

    def __init__(self, h: int, w: int, n_players: int, walls, goals):
        self.h, self.w = h, w
        self.np = n_players
        self.wall = bytearray(h * w)
        for i in walls:
            self.wall[i] = 1
        self.goal = frozenset(goals)
        self._goalmask = bytearray(h * w)
        for i in goals:
            self._goalmask[i] = 1
        #: Every value a state element can take, INTERNED. A state is a tuple of
        #: cells and codes, and codes run past CPython's small-int cache (up to
        #: ``h * w * 4``), so freshly computed elements would allocate a 28-byte
        #: int apiece -- on a sweep holding millions of states that is most of the
        #: memory. Building each element through this table makes every state
        #: tuple 8 bytes per element and nothing else.
        self._int = list(range(h * w * 4 + 4))
        #: ``_edge[cell][di]`` is the neighbour cell or -1 off the board, built
        #: once so the inner loops never do bounds arithmetic. Index 4 (ACTION)
        #: is absent: it is not a direction.
        self._edge = []
        for i in range(h * w):
            r, c = divmod(i, w)
            row = []
            for d in DIRS[:4]:
                dr, dc = _DELTA[d]
                nr, nc = r + dr, c + dc
                row.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1)
            self._edge.append(tuple(row))

    # -- the mechanic --------------------------------------------------------
    def step(self, state, di: int):
        """The state after pressing ``DIRS[di]``, or ``state`` itself when the
        press changes nothing at all -- which is a non-move: it spends a press to
        reach the board it started from, so no shortest path contains one."""
        n = self.np
        edge = self._edge
        ints = self._int

        if di == _ACTION:
            players = set(state[:n])
            out = []
            changed = False
            for code in state[n:]:
                cell = code >> 2
                e = edge[cell]
                if (e[0] in players or e[1] in players
                        or e[2] in players or e[3] in players):
                    out.append(ints[(cell << 2) | ((code + 1) & 3)])
                    changed = True
                else:
                    out.append(code)
            if not changed:
                return state
            return state[:n] + tuple(sorted(out))

        wall = self.wall
        squares = state[n:]
        at = {code >> 2: code for code in squares}
        players = set(state[:n])

        # (2) the sokoban push -- the only place a force is born
        kinds = set()
        for p in state[:n]:
            q = edge[p][di]
            if q >= 0:
                code = at.get(q)
                if code is not None:
                    kinds.add(code & 3)

        # (3) the remote control
        forced = set(players)
        if kinds:
            for code in squares:
                if (code & 3) in kinds:
                    forced.add(code >> 2)

        # (4) resolution
        memo: dict = {}

        def _can(pos: int) -> bool:
            run = []
            cur = pos
            while True:
                known = memo.get(cur)
                if known is not None:
                    ok = known
                    break
                run.append(cur)
                q = edge[cur][di]
                if q < 0 or wall[q]:
                    ok = False
                    break
                if q in forced:
                    cur = q                 # same run; acyclic, one direction
                    continue
                ok = q not in at and q not in players
                break
            for x in run:
                memo[x] = ok
            return ok

        movers = {p for p in forced if _can(p)}
        if not movers:
            return state
        new_players = tuple(sorted(
            (ints[edge[p][di]] if p in movers else p) for p in state[:n]))
        new_squares = tuple(sorted(
            ints[(edge[code >> 2][di] << 2) | (code & 3)]
            if (code >> 2) in movers else code
            for code in squares))
        return new_players + new_squares

    def won(self, state) -> bool:
        """``all A on Z``: every square stands on a goal.

        A subset test, not "every goal is covered" -- which is what the .txt
        literally says, and the same thing on all nine shipped levels because
        each has exactly as many squares as goals."""
        gm = self._goalmask
        for code in state[self.np:]:
            if not gm[code >> 2]:
                return False
        return True

    # -- rooms ---------------------------------------------------------------
    def components(self):
        """``(list of cell lists, cell -> component id)`` for the connected
        regions of non-wall cells.

        Nothing in this game ever crosses a wall, so a body stays in the region
        it started in for the whole level and the regions are static. That is
        what `_Rooms` is built on."""
        owner: dict = {}
        comps: list = []
        for i in range(self.h * self.w):
            if self.wall[i] or i in owner:
                continue
            cid = len(comps)
            cells = []
            owner[i] = cid
            queue = deque([i])
            while queue:
                u = queue.popleft()
                cells.append(u)
                for v in self._edge[u]:
                    if v >= 0 and not self.wall[v] and v not in owner:
                        owner[v] = cid
                        queue.append(v)
            comps.append(cells)
        return comps, owner

    def push_squares(self, codes, kind: int, di: int):
        """``codes`` after an EXTERNAL event forces every square of ``kind`` in
        direction ``di``, with no player present.

        This is what a room with no player in it sees: it never learns which
        square the player leaned on, only that some kind was shoved some way.
        Resolution is the same run rule as `step` -- a room is bounded by walls,
        so no run can reach out of it and nothing outside it can block one."""
        edge, wall, ints = self._edge, self.wall, self._int
        occupied = {c >> 2 for c in codes}
        forced = {c >> 2 for c in codes if (c & 3) == kind}
        if not forced:
            return codes
        memo: dict = {}

        def _can(pos: int) -> bool:
            run = []
            cur = pos
            while True:
                known = memo.get(cur)
                if known is not None:
                    ok = known
                    break
                run.append(cur)
                q = edge[cur][di]
                if q < 0 or wall[q]:
                    ok = False
                    break
                if q in forced:
                    cur = q
                    continue
                ok = q not in occupied
                break
            for x in run:
                memo[x] = ok
            return ok

        movers = {p for p in forced if _can(p)}
        if not movers:
            return codes
        return tuple(sorted(
            ints[(edge[c >> 2][di] << 2) | (c & 3)] if (c >> 2) in movers else c
            for c in codes))

    # -- macros: "walk somewhere, then press once" ---------------------------
    def macro_edges(self, state):
        """``[(cost, presses, next_state), ...]``: every "walk to a cell, then
        press once" a single player can perform, cheapest first per destination.

        WHY THIS IS THE RIGHT UNIT, and why it loses nothing. A press does one
        of three things: it moves squares (a push, which the remote rule then
        broadcasts), it turns the kinds beside the player (ACTION), or it moves
        only the player. Nothing but a push can create a force -- the remote rule
        needs a square that is ALREADY being shoved to have a direction to copy
        -- so a press that is not a push and not an ACTION beside a square is a
        pure walk step. Every plan is therefore
        ``walk* press walk* press ...``, and in a SHORTEST plan each walk
        stretch is a shortest body-free walk, because nothing moves during it.

        So enumerating (shortest walk to c, one press at c) over the cells ``c``
        that have a square beside them reaches exactly the same states an
        optimal primitive search would, at a fraction of the nodes: the player's
        position between two presses stops being a state at all. On the levels
        where the player is sealed in a room with two squares that is a ~40x cut,
        and it is what turns "a beam found a win" into "this is the shortest
        plan there is".

        Two clauses are load-bearing:

        * the walk is body-free, so it really does move nothing -- a step INTO a
          square is a push, and that is a macro of its own;
        * a press into a wall, and a push where every square of the kind is
          blocked, change nothing at all and are dropped here, which is right:
          they spend a press to reach the board they started from.

        Only defined for a ONE-player board. With two players (level 6) a walk
        is not free-form -- both bodies move on every press and either can push
        -- so that level is searched at the primitive level instead.
        """
        assert self.np == 1
        edge, wall = self._edge, self.wall
        squares = state[1:]
        occupied = {code >> 2 for code in squares}
        #: The only cells worth ending a walk on: a press from anywhere else
        #: moves nothing but the player.
        touch = set()
        for cell in occupied:
            for v in edge[cell]:
                if v >= 0 and not wall[v] and v not in occupied:
                    touch.add(v)

        start = state[0]
        dist = {start: 0}
        parent: dict = {start: None}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for di in range(4):
                v = edge[u][di]
                if v < 0 or wall[v] or v in occupied or v in dist:
                    continue
                dist[v] = dist[u] + 1
                parent[v] = (u, di)
                queue.append(v)

        best: dict = {}
        for c in touch:
            k = dist.get(c)
            if k is None:
                continue
            walk = []
            cur = c
            while parent[cur] is not None:
                prev, di = parent[cur]
                walk.append(di)
                cur = prev
            walk.reverse()
            seated = (c,) + squares
            options = [_ACTION]
            options += [di for di in range(4)
                        if edge[c][di] >= 0 and edge[c][di] in occupied]
            for di in options:
                nxt = self.step(seated, di)
                if nxt == seated:
                    continue                # spends a press, changes nothing
                cost = k + 1
                have = best.get(nxt)
                if have is None or cost < have[0]:
                    best[nxt] = (cost, tuple(walk) + (di,))
        return [(cost, presses, nxt) for nxt, (cost, presses) in best.items()]

    # -- the exact field -----------------------------------------------------
    def field(self, state, cap: int):
        """``(good, d_star, status)`` -- the exact distance field as a LAYERED
        DAG, the length of a shortest win, and one of ``"exact"`` / ``"dead"`` /
        ``"capped"``.

        ``good[d]`` is the set of states at forward depth ``d`` from which a win
        is exactly ``d_star - d`` presses away, i.e. every state on some shortest
        start-to-win path. That is all the field is ever asked for, and asking
        for no more is what makes it affordable here.

        Sweep 1 is a forward BFS by LAYER, stopped the moment a win appears --
        which both fixes ``d_star`` and collects every state within ``d_star``
        presses of the start. Sweep 2 walks the layers back down: a state in
        layer ``d`` is good iff one of its successors is good in layer ``d + 1``.

        Two things make the layered form the right one for this game rather than
        the usual backward BFS over recorded predecessors:

        * **No press here has an enumerable inverse.** A press turns the kind of
          every square beside a player, shears a linked group by leaving its
          blocked members behind, and can move seven squares at once. So the
          backward sweep must run over edges the forward one produced -- and
          re-deriving them by calling `step` again costs one extra pass, while
          STORING them costs a list per state, which is the difference between
          fitting in memory and not on the levels this is used for.
        * **A successor of a good state is always exactly one layer deeper.** If
          ``s`` is good at depth ``d`` and pressing ``p`` leaves ``d_star - d -
          1`` presses to win, then ``s'`` cannot have been seen shallower: its
          depth plus its remaining distance would be under ``d_star``, and
          ``d_star`` is minimal. So testing membership of ``good[d + 1]`` is not
          an approximation, it is the whole answer.

        ``"dead"`` is a PROOF -- the frontier ran dry with no win reachable at
        all -- which is what `record_level` needs to hear to fall back to a RESET
        rather than keep pressing.
        """
        if self.won(state):
            return [{state}], 0, "exact"
        depth = {state: 0}
        layers = [[state]]
        d_star = None
        while d_star is None:
            here = layers[-1]
            if not here:
                return None, None, "dead"
            d = len(layers)
            new = []
            for cur in here:
                for di in range(5):
                    nxt = self.step(cur, di)
                    if nxt == cur or nxt in depth:
                        continue
                    depth[nxt] = d
                    new.append(nxt)
                    if d_star is None and self.won(nxt):
                        d_star = d
            if len(depth) > cap:
                return None, None, "capped"
            layers.append(new)
        # The deepest layer is where the first win appeared, so nothing in it was
        # ever expanded -- a won board is terminal. `depth` has done its job
        # (deduplication) and is the single biggest structure here, so drop it
        # before the backward pass allocates the good sets.
        del depth
        good = [None] * (d_star + 1)
        good[d_star] = {s for s in layers[d_star] if self.won(s)}
        for d in range(d_star - 1, -1, -1):
            ahead = good[d + 1]
            layer = set()
            for cur in layers[d]:
                for di in range(5):
                    nxt = self.step(cur, di)
                    if nxt != cur and nxt in ahead:
                        layer.add(cur)
                        break
            good[d] = layer
        return good, d_star, "exact"

    def field_optimal(self, good, depth: int, state):
        """``[(press index, successor), ...]`` for every press on a shortest path
        from ``state``, which is at forward depth ``depth``. Ties come out in
        `DIRS` order, which is what makes a re-derived plan byte-identical across
        processes."""
        if depth >= len(good) - 1:
            return []
        ahead = good[depth + 1]
        out = []
        for di in range(5):
            nxt = self.step(state, di)
            if nxt != state and nxt in ahead:
                out.append((di, nxt))
        return out


# ---------------------------------------------------------------------------
# The heuristic
# ---------------------------------------------------------------------------

def _min_sum(cost) -> int:
    """``min over perfect matchings of the SUM`` on a rectangular matrix whose
    rows are at most as many as its columns (Jonker-Volgenant, O(n^3) on the
    padded square). N is at most 8 here, so this is microseconds."""
    rows = len(cost)
    if rows == 0:
        return 0
    cols = len(cost[0])
    n = max(rows, cols)
    big = [[cost[i][j] if i < rows else 0 for j in range(n)] for i in range(n)]
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = big[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return int(round(-v[0]))


def _bottleneck(cost) -> int:
    """``min over perfect matchings of max cost`` on a rectangular matrix whose
    rows (squares) are at most as many as its columns (goals).

    Binary-searches the threshold and runs Kuhn's augmenting-path matching on the
    edges below it. N is at most 8 here, so this is microseconds. Returns
    `_INF` when no perfect matching exists at any threshold, which is a PROOF
    that the board is dead -- some square can no longer reach any goal that the
    others do not need."""
    rows = len(cost)
    if rows == 0:
        return 0
    cols = len(cost[0])
    if cols < rows:
        return _INF
    values = sorted({v for row in cost for v in row})
    if not values:
        return _INF

    def feasible(limit: int) -> bool:
        match = [-1] * cols

        def try_row(r: int, seen: list) -> bool:
            for c in range(cols):
                if cost[r][c] > limit or seen[c]:
                    continue
                seen[c] = True
                if match[c] < 0 or try_row(match[c], seen):
                    match[c] = r
                    return True
            return False

        for r in range(rows):
            if not try_row(r, [False] * cols):
                return False
        return True

    lo, hi = 0, len(values) - 1
    if not feasible(values[hi]):
        return _INF
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(values[mid]):
            hi = mid
        else:
            lo = mid + 1
    return values[lo]


#: Configurations one room's exact table may hold before it is abandoned. The
#: expensive one is a room holding the player and two squares (level 8's: 36
#: cells, so 36*35*34 body placements times 16 kind assignments), which is under
#: a million; the ceiling is here so an edited level cannot turn the heuristic
#: into the search.
ROOM_CAP = 4_000_000
#: How many distinct room tables to keep. One per (room, what is standing in it),
#: so the shipped levels need one or two each -- the limit exists because
#: `--selfcheck` re-seats bodies at random and would otherwise build a table per
#: re-seating.
ROOM_TABLES = 24


class _RoomTable:
    """``get(per-room configs) -> presses-to-solved``, or None if unreachable.

    A mixed-radix index into a flat `bytearray` rather than a dict of tuples:
    the whole point of a room table is that it is small enough to keep, and 255
    is the "cannot be solved from here" sentinel."""

    __slots__ = ("index", "mult", "dist")

    def __init__(self, index, mult, dist):
        self.index = index
        self.mult = mult
        self.dist = dist

    def __len__(self):
        return len(self.dist)

    def get(self, cfgs):
        idx = 0
        for ind, m, c in zip(self.index, self.mult, cfgs):
            i = ind.get(c)
            if i is None:
                return None
            idx += i * m
        d = self.dist[idx]
        return None if d == 255 else d


#: The answer for a room that cannot hold what is standing in it -- nothing is
#: solvable, which is what an empty table says. Two spellings because the
#: player-room table is a plain dict keyed by the whole restricted state.
_EMPTY_TABLE = _RoomTable((), (), bytearray(b"\xff"))
_EMPTY_DICT: dict = {}


class _Rooms:
    """An EXACT distance for every room of the board, in isolation -- and, taken
    together, by far the strongest admissible bound this game admits.

    THE OBSERVATION. Walls split six of the nine levels into regions that no
    body can ever leave, and on five of them the player is sealed in one of them
    with the goals somewhere else entirely. A room with no player in it cannot
    see WHICH square was pushed; all it ever learns is "some kind was shoved some
    way". So its own evolution is a tiny automaton over the 16 possible
    ``(kind, direction)`` events -- for the shipped levels, a few hundred
    configurations -- and its exact distance-to-solved can simply be enumerated.

    WHY THAT IS A BOUND. Every press of a plan induces at most one event in each
    room, so the plan is at least as long as any single room's own event
    distance. Allowing all 16 events (rather than only the ones the player can
    actually produce) only makes each room's answer smaller, which keeps it a
    bound. The room holding the player is measured with the real press dynamics
    instead, restricted to the bodies in it -- also exact, because nothing
    outside the room can block or push anything inside it.

    The result is ``max`` over rooms, and it is what turns the remote-control
    levels from "a beam found a win" into "this is the shortest plan there is":
    the position-only bound in `_Heur` scores level 5 at 3 against a plan of
    dozens, because it cannot see that a square must be steered by a handle four
    rooms away.

    A room whose table has NO solved configuration reachable proves the whole
    board dead, which is exactly what `record_level` needs to hear.

    Skipped for a board whose players are in DIFFERENT rooms (level 6): a press
    then forces kinds from two places at once, so neither room's evolution is a
    function of its own state and the press alone.
    """

    __slots__ = ("board", "owner", "comps", "tables", "stored")

    def __init__(self, board: _Board):
        self.board = board
        self.comps, self.owner = board.components()
        self.tables: dict = {}
        self.stored = 0

    # -- the tables ----------------------------------------------------------
    def _remote_table(self, cids: tuple, kinds_per: tuple):
        """``_RoomTable`` over a GROUP of player-free rooms, exact in events.

        One table per group, not per room, because the rooms are not independent
        of each other even though each is independent of the board: they hear the
        SAME event, so "shove kind 4 right" moves a square in every room that has
        one. Level 7's two upper rooms each want their kind-4 square in a
        different direction, and only a table over the PAIR can see that the two
        wishes cost more than either alone.

        The joint state is a mixed-radix index over the rooms' own configuration
        lists, and the backward sweep uses each room's PREIMAGE under an event --
        the predecessors of a joint state are the product of the per-room
        predecessors -- so nothing ever materialises the joint edge list.
        """
        board = self.board
        gm = board._goalmask
        ints = board._int
        per_cfgs, per_index = [], []
        total = 1
        for cid, kinds in zip(cids, kinds_per):
            cells = self.comps[cid]
            n = len(kinds)
            if len(cells) < n:
                return _EMPTY_TABLE
            cfgs = sorted({tuple(sorted(ints[(c << 2) | k]
                                        for c, k in zip(spots, kinds)))
                           for spots in _permutations(cells, n)})
            total *= len(cfgs)
            if total > ROOM_CAP:
                return None
            per_cfgs.append(cfgs)
            per_index.append({c: i for i, c in enumerate(cfgs)})

        mult, acc = [], 1
        for cfgs in reversed(per_cfgs):
            mult.append(acc)
            acc *= len(cfgs)
        mult.reverse()

        #: ``pre[r][e][j]`` -- the configs of room ``r`` that event ``e`` maps to
        #: ``j``. A room that no event moves out of ``j`` still lists ``j``
        #: itself, which is right: the event happened, this room ignored it.
        pre = []
        for cfgs, index in zip(per_cfgs, per_index):
            room = [[[] for _ in cfgs] for _ in range(16)]
            for i, c in enumerate(cfgs):
                for kind in range(4):
                    for di in range(4):
                        room[kind * 4 + di][index[
                            board.push_squares(c, kind, di)]].append(i)
            pre.append(room)

        dist = bytearray(b"\xff" * total)
        frontier = []
        solved_per = [[i for i, c in enumerate(cfgs)
                       if all(gm[x >> 2] for x in c)] for cfgs in per_cfgs]
        for combo in _product(*solved_per):
            idx = sum(i * m for i, m in zip(combo, mult))
            dist[idx] = 0
            frontier.append(idx)
        d = 0
        while frontier and d < 254:
            d += 1
            nxt = []
            for idx in frontier:
                js, rest = [], idx
                for m in mult:
                    js.append(rest // m)
                    rest %= m
                for e in range(16):
                    for combo in _product(*(pre[r][e][j]
                                            for r, j in enumerate(js))):
                        k = sum(i * m for i, m in zip(combo, mult))
                        if dist[k] == 255:
                            dist[k] = d
                            nxt.append(k)
            frontier = nxt
        return _RoomTable(tuple(per_index), tuple(mult), dist)

    def _player_table(self, cid: int, n_players: int, n_squares: int):
        """``(players..., codes...) -> presses-to-solved`` for the room holding
        the player, over the REAL press dynamics restricted to it."""
        cells = self.comps[cid]
        board = self.board
        gm = board._goalmask
        ints = board._int
        n = n_players + n_squares
        if len(cells) < n:
            return _EMPTY_DICT
        if len(cells) ** n * (4 ** n_squares) > 4 * ROOM_CAP:
            return None                      # cheap size test before enumerating
        configs = set()
        for spots in _permutations(cells, n):
            players = tuple(sorted(ints[c] for c in spots[:n_players]))
            for kinds in _product(range(4), repeat=n_squares):
                configs.add(players + tuple(sorted(
                    ints[(c << 2) | k]
                    for c, k in zip(spots[n_players:], kinds))))
            if len(configs) > ROOM_CAP:
                return None
        rev: dict = {}
        dist: dict = {}
        frontier = deque()
        for cfg in configs:
            if all(gm[c >> 2] for c in cfg[n_players:]):
                dist[cfg] = 0
                frontier.append(cfg)
        for cfg in configs:
            for di in range(5):
                nxt = board.step(cfg, di)
                if nxt != cfg:
                    rev.setdefault(nxt, []).append(cfg)
        while frontier:
            cur = frontier.popleft()
            d = dist[cur]
            for prev in rev.get(cur, ()):
                if prev not in dist:
                    dist[prev] = d + 1
                    frontier.append(prev)
        return dist

    def _table(self, key):
        table = self.tables.get(key)
        if table is not None or key in self.tables:
            return table
        if self.stored > ROOM_CAP or len(self.tables) >= ROOM_TABLES:
            self.tables[key] = None
            return None
        cids, n_players, payload = key
        table = (self._player_table(cids[0], n_players, payload) if n_players
                 else self._remote_table(cids, payload))
        self.tables[key] = table
        if table is not None:
            self.stored += len(table)
        return table

    # -- the bound -----------------------------------------------------------
    def bound(self, state) -> int:
        """``max`` over the room groups whose exact tables exist, or `_INF` when
        one of them proves the board can never be solved.

        Groups, cheapest-covering first: every player-free room together (the
        strongest, when the product fits), then every PAIR of them, then the
        singles. Each is a max over an independent lower bound, so taking the
        largest available is sound however the tables happen to fall out -- a
        group that is too big to enumerate simply contributes nothing."""
        board = self.board
        owner = self.owner
        n = board.np
        rooms: dict = {}
        for p in state[:n]:
            rooms.setdefault(owner[p], ([], []))[0].append(p)
        for code in state[n:]:
            rooms.setdefault(owner[code >> 2], ([], []))[1].append(code)
        if sum(1 for players, _ in rooms.values() if players) > 1:
            return 0                         # see the class docstring
        best = 0
        free = []
        for cid, (players, codes) in sorted(rooms.items()):
            if not codes:
                continue
            if players:
                table = self._table(((cid,), len(players), len(codes)))
                if table is None:
                    continue
                d = table.get(tuple(sorted(players)) + tuple(sorted(codes)))
                if d is None:
                    return _INF
                best = max(best, d)
            else:
                free.append((cid, tuple(sorted(codes))))

        def group(items):
            nonlocal best
            key = (tuple(cid for cid, _ in items), 0,
                   tuple(tuple(sorted(c & 3 for c in codes))
                         for _cid, codes in items))
            table = self._table(key)
            if table is None:
                return False                 # too big to enumerate: no bound
            d = table.get([codes for _cid, codes in items])
            if d is None:
                best = _INF
                return True
            best = max(best, d)
            return True

        if len(free) > 1 and not group(free):
            for i in range(len(free)):
                for j in range(i + 1, len(free)):
                    if group([free[i], free[j]]) and best >= _INF:
                        return _INF
        for item in free:
            group([item])
            if best >= _INF:
                return _INF
        return best


class _Heur:
    """Two scores over the same floor distances: an admissible BOUND for the A*
    that has to prove a plan shortest, and a RANKING for the beam that only has
    to find one.

    ``bound`` is the bottleneck assignment: every square must finish on a goal of
    its own, a press moves any one square by at most one cell, so the plan is at
    least as long as the largest distance in the cheapest assignment. It is a MAX
    and not a sum on purpose -- one press moves an entire kind at once, so seven
    squares can each close a cell in the same press and a sum would not be a
    bound at all.

    ``rank`` adds the other two things that obviously cost presses -- the total
    assignment cost, and the walk a player still owes before it can touch
    anything -- and is admissible in neither, which is fine for a beam.

    Distances are floor walks that ignore every body, so they never over-estimate
    (a square in the way only ever makes the real walk longer)."""

    __slots__ = ("board", "dist", "goals", "rooms")

    def __init__(self, board: _Board):
        self.board = board
        self.rooms = _Rooms(board)
        self.goals = sorted(board.goal)
        n = board.h * board.w
        #: ``dist[g][cell]``: the walk from every cell to goal ``g``. Sweeping
        #: from the GOALS rather than all-pairs keeps this at |goals| BFS runs
        #: (at most 8) instead of |cells|.
        self.dist = []
        for g in self.goals:
            d = [_INF] * n
            d[g] = 0
            q = deque([g])
            while q:
                u = q.popleft()
                for v in board._edge[u]:
                    if v >= 0 and not board.wall[v] and d[v] > d[u] + 1:
                        d[v] = d[u] + 1
                        q.append(v)
            self.dist.append(d)

    def _cost(self, state):
        D = self.dist
        return [[D[j][code >> 2] for j in range(len(self.goals))]
                for code in state[self.board.np:]]

    def bound(self, state) -> int:
        """``max(per-room exact distance, bottleneck, cheapest travel / squares)``.

        The last two are the position-only halves and neither dominates the
        other: a press moves at most every square by one cell, so the plan is at
        least the cheapest total travel divided by how many squares there are;
        and it is at least the longest single journey in the cheapest bottleneck
        assignment. They are minimised over DIFFERENT assignments, which is
        legal -- the actual final assignment is a perfect matching, so it costs
        at least the cheapest one under either measure.

        `_Rooms.bound` is the one that does the work on the remote-control
        levels, where the position-only halves score a thirty-press board at 3.
        All three are admissible, so their max is."""
        if self.board.won(state):
            return 0
        rooms = self.rooms.bound(state)
        if rooms:
            # On every board that HAS room tables this dominates the other two by
            # a wide margin (level 8: 16 against 4), and both of them are an
            # O(n^3) assignment solve per node, so it is worth not paying for
            # them. Dropping a smaller admissible term keeps the result
            # admissible. A zero here is never "solved" -- that is the `won`
            # branch above -- it means no table was available.
            return rooms
        cost = self._cost(state)
        h = _bottleneck(cost)
        if h >= _INF:
            return _INF
        n = len(cost)
        return max(h, -(-_min_sum(cost) // n))

    def rank(self, state) -> int:
        """The BEAM's score: the room distance weighted up, so that closing a
        room outranks shuffling a square one cell nearer to a goal, plus the
        position-only terms as a tie-break within a room-distance class.

        Inadmissible on purpose, and never read by anything that claims a plan is
        shortest."""
        if self.board.won(state):
            return 0
        cost = self._cost(state)
        h = _bottleneck(cost)
        if h >= _INF:
            return _INF
        rooms = self.rooms.bound(state)
        if rooms >= _INF:
            return _INF
        gm = self.board._goalmask
        h = 8 * rooms + 4 * h + _min_sum(cost)
        loose = [code >> 2 for code in state[self.board.np:]
                 if not gm[code >> 2]]
        if loose:
            reach = _INF
            for p in state[:self.board.np]:
                for cell in loose:
                    r0, c0 = divmod(p, self.board.w)
                    r1, c1 = divmod(cell, self.board.w)
                    reach = min(reach, abs(r0 - r1) + abs(c0 - c1))
            if reach < _INF:
                h += reach
        return h


# ---------------------------------------------------------------------------
# The heuristic searches
# ---------------------------------------------------------------------------

def _astar(board: _Board, heur: _Heur, state, weight: int, cap: int, deadline):
    """Weighted A* on `_Heur.bound`. Returns a press-index list, or None if the
    cap, the deadline or the reachable space ran out first.

    At ``weight == 1`` the bound is admissible and a returned plan is provably
    shortest; above 1 it is only a win."""
    if board.won(state):
        return []
    counter = 0
    pq = [(weight * heur.bound(state), 0, 0, state, ())]
    best_g = {state: 0}
    nodes = 0
    while pq:
        _f, g, _c, cur, path = heapq.heappop(pq)
        if g > best_g.get(cur, 1 << 30):
            continue
        if g >= MAX_PLAN:
            continue
        for di in range(5):
            nxt = board.step(cur, di)
            if nxt == cur:
                continue
            nodes += 1
            if board.won(nxt):
                return list(path) + [di]
            ng = g + 1
            if best_g.get(nxt, 1 << 30) <= ng:
                continue
            h = heur.bound(nxt)
            if h >= _INF:
                continue                    # proved dead: no assignment exists
            best_g[nxt] = ng
            counter += 1
            heapq.heappush(pq, (ng + weight * h, ng, counter, nxt, path + (di,)))
        if nodes >= cap or time.monotonic() > deadline:
            return None
    return None


def _macro_astar(board: _Board, heur: _Heur, state, cap: int, deadline,
                 weight: int = 1):
    """A* over `_Board.macro_edges`, whose edge costs are the presses the macro
    spends. Returns a flat PRESS list, or None if the cap, the deadline or the
    reachable space ran out first.

    At ``weight == 1`` the heuristic is admissible and the goal is returned when
    it is POPPED, not when it is generated -- edges here cost more than one, so
    the first win to be REACHED is not the cheapest -- which makes a returned
    plan provably the shortest press sequence there is."""
    if board.won(state):
        return []
    counter = 0
    pq = [(weight * heur.bound(state), 0, 0, state, ())]
    best_g = {state: 0}
    nodes = 0
    while pq:
        _f, g, _c, cur, path = heapq.heappop(pq)
        if g > best_g.get(cur, 1 << 30):
            continue
        if board.won(cur):
            return list(path)
        for cost, presses, nxt in board.macro_edges(cur):
            ng = g + cost
            if ng > MAX_PLAN:
                continue
            nodes += 1
            if best_g.get(nxt, 1 << 30) <= ng:
                continue
            h = 0 if board.won(nxt) else heur.bound(nxt)
            if h >= _INF:
                continue                    # proved dead: no assignment exists
            best_g[nxt] = ng
            counter += 1
            heapq.heappush(pq, (ng + weight * h, ng, counter, nxt,
                                path + presses))
        if nodes >= cap or time.monotonic() > deadline:
            return None
    return None


def _macro_within(board: _Board, heur: _Heur, state, limit: int,
                  cap: int, deadline):
    """``True`` / ``False`` / ``None``: is a win reachable from ``state`` within
    ``limit`` presses? ``None`` means the cap or the deadline ran out before the
    answer -- a budget failure, not a "no".

    The DECISION is what `measured_optsets` needs, and it is far cheaper than the
    distance it would otherwise ask for: because the bound is admissible, every
    node with ``g + h > limit`` is dropped instead of expanded, and one press off
    a shortest path the bound is already tight against the budget. Asking
    `_macro_astar` for the exact distance instead searches the whole ball."""
    if board.won(state):
        return True
    if limit < 1 or heur.bound(state) > limit:
        return False
    counter = 0
    pq = [(heur.bound(state), 0, 0, state)]
    best_g = {state: 0}
    nodes = 0
    while pq:
        _f, g, _c, cur = heapq.heappop(pq)
        if g > best_g.get(cur, 1 << 30):
            continue
        if board.won(cur):
            return True
        for cost, _presses, nxt in board.macro_edges(cur):
            ng = g + cost
            if ng > limit:
                continue
            nodes += 1
            if best_g.get(nxt, 1 << 30) <= ng:
                continue
            h = 0 if board.won(nxt) else heur.bound(nxt)
            if h >= _INF or ng + h > limit:
                continue
            best_g[nxt] = ng
            counter += 1
            heapq.heappush(pq, (ng + h, ng, counter, nxt))
        if nodes >= cap or time.monotonic() > deadline:
            return None
    return False


def _beam(board: _Board, heur: _Heur, state, width: int, deadline):
    """Beam over MACROS (or over primitive presses on a two-player board),
    ranked by ``presses spent so far + _Heur.rank``, deduped against everything
    already seen. Returns a flat press list or None.

    Ranking on ``g + rank`` rather than on ``rank`` alone matters here because
    macro edges cost between 1 and a dozen presses: a beam that ignored what it
    had already spent would happily walk the length of the board for a score it
    could have bought next door."""
    if board.won(state):
        return []
    macro = board.np == 1
    frontier = [(heur.rank(state), 0, state, ())]
    seen = {state}
    recording = True
    best = None
    for _ in range(MAX_PLAN):
        nxt = []
        for _score, g, cur, path in frontier:
            edges = (board.macro_edges(cur) if macro else
                     [(1, (di,), board.step(cur, di)) for di in range(5)])
            for cost, presses, s2 in edges:
                if s2 == cur or s2 in seen:
                    continue
                ng = g + cost
                if ng > MAX_PLAN:
                    continue
                p2 = path + tuple(presses)
                if board.won(s2):
                    if best is None or len(p2) < len(best):
                        best = list(p2)
                    continue
                if recording:
                    seen.add(s2)
                score = heur.rank(s2)
                if score >= _INF:
                    continue
                nxt.append((ng + score, ng, s2, p2))
        if recording and len(seen) > BEAM_SEEN_CAP:
            # Stop deduping rather than let a wide beam spend more memory on the
            # `seen` set than the search is worth. Re-visiting a state costs
            # nothing but a slot in the frontier.
            recording = False
        if best is not None or not nxt or time.monotonic() > deadline:
            return best
        nxt.sort(key=lambda x: x[0])
        frontier = nxt[:width]
    return best


def measured_optsets(board: _Board, heur: _Heur, state, presses,
                     cap: int = MEASURE_CAP, seconds: float = MEASURE_SECONDS,
                     total_seconds: float = MEASURE_TOTAL_SECONDS):
    """EXACT per-step optimal sets for a plan already known to be SHORTEST,
    measured by re-solving from each successor.

    ``(sets, unresolved)``. This is what the field would have produced, bought a
    different way: off a field level there is no distance table, but there is a
    search that returns the exact distance, so ask it. A press is optimal at step
    ``i`` when the plan has ``r`` presses left and that press leaves ``r - 1``.

    Two prunes make it affordable and both are sound rather than heuristic -- a
    press that leaves the board untouched cannot be on a shortest path (it spends
    a press to reach the board it started from), and neither can one whose
    ADMISSIBLE bound already exceeds what is left. On these levels `_Heur.bound`
    is the per-room exact distance, which is sharp enough that the second prune
    answers most alternatives outright.

    A query that runs out of budget leaves that step ``unresolved``; the caller
    falls back to `rejoin_optsets` for it rather than shipping a set that might
    be missing a genuine tie. The recorded press is always in its own set.

    Note the test is a DECISION, "can this successor still finish in ``r - 1``",
    not a distance. It is equivalent here: the plan is shortest, so every
    successor of a state on it needs at least ``r - 1``, and no successor can do
    better.

    STEPS ARE MEASURED FROM THE END BACKWARDS, under one shared clock. Cost rises
    steeply with what is left -- at ``r`` small the admissible bound is tight
    against the budget and `_macro_within` refuses almost every node, while at
    ``r`` near the plan's whole length the question is as hard as the original
    search. Working backwards therefore buys the most labelled steps per second,
    and stopping when the clock runs out leaves the EXPENSIVE end unresolved
    rather than a random half."""
    states = [state]
    for di in presses:
        states.append(board.step(states[-1], di))
    sets: list = [None] * len(presses)
    unresolved = []
    stop = time.monotonic() + total_seconds
    for i in range(len(presses) - 1, -1, -1):
        if time.monotonic() > stop:
            unresolved.extend(range(i + 1))
            break
        remaining = len(presses) - i
        cur = states[i]
        best, gave_up = [], False
        for alt in range(5):
            nxt = board.step(cur, alt)
            if nxt == cur:
                continue                     # spends a press, changes nothing
            if board.won(nxt):
                if remaining == 1:
                    best.append(alt)
                continue
            got = _macro_within(board, heur, nxt, remaining - 1, cap,
                                min(time.monotonic() + seconds, stop))
            if got is None:
                gave_up = True
                break
            if got:
                best.append(alt)
        if gave_up or presses[i] not in best:
            unresolved.append(i)
        else:
            sets[i] = [DIRS[a] for a in best]
    return sets, sorted(unresolved)


def shorten(board: _Board, state, presses, horizon: int = SHORTEN_HORIZON):
    """Splice out of ``presses`` any stretch a bounded BFS can cross faster.

    Only ever applied to a plan that is NOT known to be shortest (a beam's, or a
    weighted A*'s). It is the cheap half of what the field would have proved: a
    beam commits to its ranking at every layer and habitually pays two or three
    presses to undo a turn it took for the score, and those are exactly the
    detours a small BFS between two states the plan already visits finds. It also
    catches the degenerate case for free -- a plan that returns to a state it has
    already been in is spliced at the repeat.

    Sound by construction: the replacement is a real press sequence between two
    states of the plan itself, so the tail after the splice still applies."""
    presses = list(presses)
    while True:
        states = [state]
        for di in presses:
            states.append(board.step(states[-1], di))
        #: LAST occurrence, so a repeated state splices away as much as possible.
        index = {s: i for i, s in enumerate(states)}
        cut = None
        for i in range(len(presses)):
            budget = min(horizon, len(presses) - i - 1)
            if budget <= 0:
                continue
            frontier = [(states[i], ())]
            seen = {states[i]}
            for d in range(1, budget + 1):
                nxt = []
                for cur, path in frontier:
                    for di in range(5):
                        s2 = board.step(cur, di)
                        if s2 == cur or s2 in seen:
                            continue
                        seen.add(s2)
                        p2 = path + (di,)
                        j = index.get(s2)
                        if j is not None and j > i + d:
                            cut = (i, j, p2)
                            break
                        nxt.append((s2, p2))
                    if cut:
                        break
                if cut or not nxt:
                    break
                frontier = nxt
            if cut:
                break
        if not cut:
            return presses
        i, j, path = cut
        presses[i:j] = list(path)


def rejoin_optsets(board: _Board, state, presses, horizon: int = REJOIN_HORIZON):
    """Per-step optimal SETS for a plan whose length is not known to be optimal.

    A press ``alt`` is in step ``i``'s set when, taken instead of the plan's own,
    some ``k <= horizon`` further presses land exactly on the state the plan
    reaches after ``k + 1`` presses -- because then "alt, those k, and the plan's
    own tail" is a plan of the SAME length. That is sound with no reference to
    optimality: every press in a set demonstrably costs what the recorded one
    costs. It is not complete -- a tie that rejoins later than the horizon, or
    never, is missed -- so off a field level these sets are a floor rather than
    the whole truth. `--selfcheck` calibrates the horizon against the levels that
    DO have a field, where both answers exist."""
    states = [state]
    for di in presses:
        states.append(board.step(states[-1], di))
    out = []
    for i, di in enumerate(presses):
        best = {di}
        for alt in range(5):
            if alt == di:
                continue
            s = board.step(states[i], alt)
            if s == states[i]:
                continue
            if board.won(s):
                if len(presses) - i == 1:
                    best.add(alt)
                continue
            if s == states[i + 1]:
                best.add(alt)
                continue
            frontier = {s}
            for k in range(1, horizon + 1):
                if i + 1 + k >= len(states):
                    break
                target = states[i + 1 + k]
                grown = set()
                hit = False
                for cur in frontier:
                    for d2 in range(5):
                        s2 = board.step(cur, d2)
                        if s2 == cur:
                            continue
                        if s2 == target:
                            hit = True
                            break
                        grown.add(s2)
                    if hit:
                        break
                if hit:
                    best.add(alt)
                    break
                if not grown:
                    break
                frontier = grown
        out.append([DIRS[d] for d in sorted(best)])
    return out


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class VariationsExpert(PSExpert):
    """`PSExpert`'s plan memo, level scoping and disk cache around the ladder.

    The base class keeps everything except the strategy, which is replaced the
    way `PSEnumExpert` replaces it -- so `heuristic` is never called and asserts
    rather than returning a number nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state (a recovery prefix, an epsilon detour) is answered from that
    state -- including "this board is now dead", which an exploration prefix
    really can produce here: shove a square into a pocket it cannot be steered
    out of and the level is over."""

    directions = list(DIRS)
    #: `_key` is the dynamic objects only, which is canonical WITHIN a level but
    #: not across them -- the walls and the goals are static per level and differ
    #: between levels.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    #: Whether this expert may spend the FULL ladder budget. `prepare_expert`
    #: turns it on for the one startup pass over the shipped level starts and off
    #: again afterwards, so a mid-episode re-plan cannot cost what a cold
    #: derivation costs.
    full_budget = False

    def setup(self) -> None:
        g = self.g
        self.player_ids = set(self.game._engine._player_indices)
        self.wall_ids = set(g.resolve_object_name("w"))
        self.goal_ids = set(g.resolve_object_name("omega"))
        #: The four kinds, in cycle order, so index+1 IS the kind.
        self.kind_ids = [set(g.resolve_object_name(f"a_{k}")) for k in (1, 2, 3, 4)]
        #: `_Board`s by STATIC signature (see `read`), not by level index, so the
        #: edge table is built once per level and shared by every state of it.
        self._boards: dict = {}
        self._last_how = ""
        #: Steps of the last ``macro:w1`` plan whose exact tie set could not be
        #: measured in budget and fell back to `rejoin_optsets`. Reported by
        #: `--plans`; it is a quality number, not a correctness one -- a rejoin
        #: set is sound either way.
        self._last_unresolved = 0

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "VariationsExpert reads an exact distance field (or an admissible "
            "A* bound); PSExpert.heuristic is unused")

    def _key(self, eng) -> frozenset:
        """``(row, col, tag)`` triples for the players (tag 0) and the squares
        (tag = kind).

        Built from the MODEL's reading rather than from raw object ids, so the
        disk cache's stored signature is exactly what the model plans from."""
        _board, _heur, state = self.read(eng)
        if state is None:
            return frozenset()
        w = eng.width
        n = _board.np
        return frozenset(
            {(p // w, p % w, 0) for p in state[:n]}
            | {((c >> 2) // w, (c >> 2) % w, (c & 3) + 1) for c in state[n:]})

    def plan(self, eng, level: int | None = None):
        """`PSExpert.plan`, plus the VERDICT carried through the disk cache.

        ``how`` is what `--plans` reports and it is derived by `solve`, which a
        cache hit never calls -- so without this a cached run would say "cached"
        about every level and the report would stop meaning anything. Recorded
        only against the entry whose stored start layout is the board in front
        of us, so a re-plan from a perturbed state cannot overwrite the level's
        verdict with its own."""
        self._last_how = ""
        self._last_unresolved = 0
        got = super().plan(eng, level)
        if level is None or self.plan_cache_path is None:
            return got
        entry = self._disk.get(level)
        if entry is None:
            return got
        sig = sorted([r, c, o] for (r, c, o) in self._key(eng))
        if entry.get("start") != sig:
            return got
        if self._last_how:
            if entry.get("how") != self._last_how:
                entry["how"] = self._last_how
                entry["unresolved"] = self._last_unresolved
                self._save_disk()
        else:
            self._last_how = entry.get("how", "cached")
            self._last_unresolved = entry.get("unresolved", 0)
        return got

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, heuristic, state)`` for the engine's current grid, with
        ``state`` None when there is no player on it.

        Boards are cached by their STATIC signature -- dimensions, player count,
        walls, goals -- rather than by level index, so the edge table and the
        goal-distance sweeps are built once per level and shared by every state
        of it, including the re-seated boards `--selfcheck` invents."""
        h, w = eng.height, eng.width
        walls, goals, players, squares = [], [], [], []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.goal_ids:
                    goals.append(i)
                if cell & self.player_ids:
                    players.append(i)
                for k, ids in enumerate(self.kind_ids):
                    if cell & ids:
                        squares.append((i << 2) | k)
                        break
        sig = (h, w, len(players), tuple(walls), tuple(goals))
        entry = self._boards.get(sig)
        if entry is None:
            board = _Board(h, w, len(players), walls, goals)
            entry = self._boards[sig] = (board, _Heur(board))
        board, heur = entry
        if not players:
            return board, heur, None
        ints = board._int
        return board, heur, (tuple(ints[p] for p in sorted(players))
                             + tuple(ints[c] for c in sorted(squares)))

    # -- the ladder ----------------------------------------------------------
    def solve(self, board: _Board, heur: _Heur, state, full: bool):
        """``(Plan, how)`` or ``(None, how)``.

        ``how`` is one of

        * ``"won"`` -- nothing to do;
        * ``"field"`` -- provably shortest, with COMPLETE tie sets;
        * ``"macro:w1"`` / ``"astar:w1"`` -- provably shortest (the bound is
          admissible and the goal is taken off the heap, not off the frontier),
          with tie sets by `rejoin_optsets`, i.e. a sound floor;
        * ``"macro:w<n>"`` / ``"astar:w<n>"`` / ``"beam:<width>"`` -- a win, no
          claim about its length, tie sets by rejoin;
        * ``"dead"`` -- PROVED unwinnable, which is what `record_level` needs to
          hear to fall back to a RESET;
        * ``"unsolved"`` -- every rung ran out."""
        self._last_unresolved = 0
        if board.won(state):
            return Plan([], []), "won"
        if heur.bound(state) >= _INF:
            # No square-to-goal assignment exists at all, so no press sequence
            # can finish. Cheap, and the only deadness test a capped level gets.
            return None, "dead"
        good, d_star, status = board.field(
            state, FIELD_CAP if full else FIELD_CAP_FAST)
        if status == "dead":
            return None, "dead"
        if status == "exact":
            if d_star > MAX_PLAN:
                return None, "unsolved"     # cannot be replayed in 200 presses
            presses, optsets = [], []
            cur = state
            for i in range(d_star):
                best = board.field_optimal(good, i, cur)
                if not best:                # unreachable: good[i] has a step
                    return None, "unsolved"
                presses.append(DIRS[best[0][0]])
                optsets.append([DIRS[di] for di, _ in best])
                cur = best[0][1]
            return Plan(presses, optsets), "field"

        found, how = None, "unsolved"
        if board.np == 1:
            # Macros first: the same answer with far fewer nodes, because the
            # player's position between two presses stops being a state. The
            # two-player board has no macro decomposition (both bodies move on
            # every press and either can push) and goes straight to the
            # primitive rungs.
            for weight, cap, secs in (_MACRO_FULL if full else _MACRO_FAST):
                got = _macro_astar(board, heur, state, cap,
                                   time.monotonic() + secs, weight)
                if got is not None:
                    found, how = got, f"macro:w{weight}"
                    break
        if found is None:
            for weight, cap, secs in (_ASTAR_FULL if full else _ASTAR_FAST):
                got = _astar(board, heur, state, weight, cap,
                             time.monotonic() + secs)
                if got is not None:
                    found, how = got, f"astar:w{weight}"
                    break
        if how in ("macro:w1", "astar:w1"):
            # The bound is admissible and the goal was returned when it was
            # POPPED, so this IS a shortest plan: no beam can beat it and
            # `shorten` has nothing to find. Knowing the plan is shortest is
            # also what lets the tie sets be MEASURED exactly -- "which presses
            # still leave r - 1" only means anything when r is the truth.
            cheap = rejoin_optsets(board, state, found)
            sets = cheap
            if full and how == "macro:w1":
                exact, unresolved = measured_optsets(board, heur, state, found)
                sets = [cheap[i] if s is None else s for i, s in enumerate(exact)]
                self._last_unresolved = len(unresolved)
            return Plan([DIRS[di] for di in found], sets), how
        # Every beam rung is run and the SHORTEST answer kept, rather than
        # stopping at the first that wins: a narrow beam on this game reliably
        # finds a win and reliably pays several presses for it, and the plan
        # length is what the corpus teaches. Each candidate is `shorten`ed
        # BEFORE the comparison, so the rungs are ranked on what they are
        # actually worth rather than on how much slack they happened to leave.
        if found is not None:
            found = shorten(board, state, found)
        for width, secs in (_BEAM_FULL if full else _BEAM_FAST):
            got = _beam(board, heur, state, width, time.monotonic() + secs)
            if got is None:
                continue
            got = shorten(board, state, got)
            if found is None or len(got) < len(found):
                found, how = got, f"beam:{width}"
        if found is None or len(found) > MAX_PLAN:
            return None, "unsolved"
        return (Plan([DIRS[di] for di in found],
                     rejoin_optsets(board, state, found)), how)

    def _search(self, eng) -> "Plan | None":
        board, heur, state = self.read(eng)
        if state is None:                    # no player on the board
            return None
        plan, how = self.solve(board, heur, state, self.full_budget)
        self._last_how = how
        return plan


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class VariationsSolver(PSAStarSolver):
    game_id = "puzzlescript_variations"
    game_name = GAME_NAME
    expert_cls = VariationsExpert

    #: `games/ps:variations/ps:variations.py` is a plain passthrough (it
    #: constructs the adapter and nothing else), so there is nothing to gain by
    #: routing through it -- but it IS what a live agent is handed, so if that
    #: wrapper ever grows a patch this must be set to ``"ps:variations"``.
    game_module_id = ""

    #: Unused: the expert runs its own ladder rather than `PSExpert._astar`.
    #: Left at the base values so nothing reads a lie off them.
    weight = 1
    node_cap = 400_000

    max_steps = 200
    epsilon = 0.0

    def prepare_expert(self, game, expert) -> None:
        """Plan every level's START with the FULL ladder budget before
        `discover_solvable` asks for it -- the same work either way, but it fills
        the disk cache in one pass, makes the startup cost visible as startup,
        and leaves the expert on the fast budget for the re-plans that happen
        inside the recording loop."""
        expert.full_budget = True
        try:
            for level in range(game.n_levels):
                if level in self.skip_levels:
                    continue
                game.set_level(level)
                expert.plan(game._engine, level)
        finally:
            expert.full_budget = False


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new():
    solver = VariationsSolver()
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


#: The verdicts that mean "this plan is the shortest one there is".
_PROVED = ("won", "field", "macro:w1", "astar:w1")


def _exact_sets(expert: VariationsExpert) -> bool:
    """Did the last `VariationsExpert.plan` produce COMPLETE optimal-action sets
    rather than `rejoin_optsets`' floor? True off a field, and off a
    ``macro:w1`` plan every one of whose steps `measured_optsets` reached."""
    return (expert._last_how in ("won", "field")
            or (expert._last_how in ("macro:w1", "astar:w1")
                and not expert._last_unresolved))


def _plans() -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the ladder and the tie labelling at
    once."""
    _solver, game, expert, solvable = _new()
    eng = game._engine
    total = ties = bad = 0
    shortest = skipped = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, _heur, state = expert.read(eng)
        plan = expert.plan(eng, level)
        how = expert._last_how
        if plan is None:
            # A level with no plan is DROPPED from every episode by
            # `discover_solvable`, not emitted broken, so it is not a failure of
            # this report -- but which KIND of no-plan it is matters: "dead" is a
            # proof that the board cannot be won, "unsolved" is the ladder
            # running out of budget and is worth another look.
            print(f"  L{level:2d}: {eng.height:2d}x{eng.width:2d}  "
                  f"{'PROVED UNWINNABLE' if how == 'dead' else 'UNSOLVED'} "
                  f"({how}) -- dropped from every episode")
            skipped += 1
            continue
        for direction in plan:
            eng.step(direction)
        won = eng.check_win()
        bad += not won
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        tie_steps = sum(1 for s in sets if len(s) > 1)
        total += len(plan)
        ties += tie_steps
        proved = how in _PROVED
        shortest += proved
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        exact_sets = _exact_sets(expert)
        print(f"  L{level:2d}: {eng.height:2d}x{eng.width:2d}  "
              f"{len(state) - board.np} squares / {len(board.goal)} goals  "
              f"{len(plan):3d} presses  win={won}  "
              f"{'shortest' if proved else 'a win  '} via {how:<12s} "
              f"(budget {game._max_steps}, {room})  "
              f"{tie_steps:3d} steps with a tie set"
              f"{'' if exact_sets else ' (a sound floor, not exact)'}")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print(f"  {shortest}/{game.n_levels - skipped} solved levels provably "
          f"shortest ({skipped} level(s) have no plan and are dropped)")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


#: `_brute` ran out of states before it could answer -- neither a distance nor
#: "no win within the limit". Callers skip; treating it as a "no" would turn a
#: budget into a false accusation.
_CAPPED = object()


def _brute(board: _Board, state, limit: int, cap: int = 3_000_000):
    """Presses to a win from ``state``, searched fresh; None if there is none
    within ``limit``; `_CAPPED` if the state cap ran out first.

    Shares nothing with `_Board.field`, `_macro_astar` or `_Heur` but
    `_Board.step` itself, which is the point: it is the independent answer every
    optimality claim below is checked against."""
    if board.won(state):
        return 0
    seen = {state}
    queue = deque([(state, 0)])
    while queue:
        cur, d = queue.popleft()
        if d >= limit:
            continue
        for di in range(5):
            nxt = board.step(cur, di)
            if nxt == cur or nxt in seen:
                continue
            if board.won(nxt):
                return d + 1
            if len(seen) >= cap:
                return _CAPPED
            seen.add(nxt)
            queue.append((nxt, d + 1))
    return None


def _selfcheck(walk_presses: int = 720, samples: int = 45,
               brute_limit: int = 9, verbose: bool = True) -> int:
    """Nine independent checks, all of which must report zero.

    1. The MODEL against the interpreter: a seeded random walk on every level,
       comparing the model's state with the engine's grid AND the model's win
       test with the engine's after every single press.
    2. The same on randomly re-seated boards, so the comparison is not confined
       to the states a shipped level happens to reach.
    3. Every FIELD plan re-proved shortest by `_brute`, an estimate-free BFS that
       shares nothing with the field but `step`.
    4. Every field TIE SET brute-forced: each press in the set must reach a win
       in exactly the presses the plan has left, and each press outside it must
       not.
    5. `rejoin_optsets` calibrated against the exact sets on the field levels --
       it may be a subset, that is what "a floor" means, but it is never wrong.
    6. `measured_optsets` against the FIELD's sets on the levels that have both.
       Two independent exact answers to the same question, so they must agree
       EXACTLY rather than one containing the other.
    7. The ROOM decomposition the admissible bound is built on -- see
       `_check_rooms`.
    8. The MACRO decomposition, which is what makes a ``macro:w1`` plan a proof:
       on every state near enough to a win for `_brute` to answer, `_macro_astar`
       must return exactly that distance.
    9. Every recovery board answered with a plan that really wins, or a truthful
       None.
    """
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    rng = random.Random(20260822)
    bad = 0

    # -- 1 & 2: the model against the interpreter ---------------------------
    presses = 0
    for level in range(game.n_levels):
        for trial in range(4):
            game.set_level(level)
            board, _heur, state = expert.read(eng)
            if trial:
                # Re-seat every body on a random free cell: the same mechanic,
                # on boards the shipped level never reaches.
                free = [i for i in range(board.h * board.w) if not board.wall[i]]
                rng.shuffle(free)
                n = board.np
                cells = free[:len(state)]
                state = (tuple(sorted(cells[:n]))
                         + tuple(sorted((c << 2) | rng.randrange(4)
                                        for c in cells[n:])))
                _seat(eng, expert, board, state)
            for _ in range(walk_presses // 4):
                di = rng.randrange(5)
                eng.step(DIRS[di])
                state = board.step(state, di)
                presses += 1
                _b, _h, live = expert.read(eng)
                if live != state:
                    bad += 1
                    if bad <= 3:
                        print(f"    L{level} t{trial}: model {state} "
                              f"!= engine {live} after {DIRS[di]}")
                    state = live
                if board.won(state) != eng.check_win():
                    bad += 1
                    print(f"    L{level} t{trial}: win test disagrees")
                if eng.check_win():
                    break
    if verbose:
        print(f"  model vs interpreter: {presses} presses, {bad} violations")

    # -- 3 & 4: the plans that CLAIM to be shortest, re-proved ---------------
    checked_plans = checked_ties = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, _heur, state = expert.read(eng)
        plan = expert.plan(eng, level)
        if plan is None or not _exact_sets(expert):
            continue
        # The whole length, where an estimate-free BFS can still reach a win.
        # `_brute` is given the plan's OWN length as its depth budget, so a
        # shorter answer would come back rather than being cut off.
        d = _brute(board, state, len(plan))
        if d is not _CAPPED:
            checked_plans += 1
            if d != len(plan):
                bad += 1
                print(f"    L{level}: the plan is {len(plan)} presses "
                      f"({expert._last_how}), brute force says {d}")
        cur = state
        for i, direction in enumerate(plan):
            remaining = len(plan) - i
            if remaining <= brute_limit:
                for di, d_name in enumerate(DIRS):
                    nxt = board.step(cur, di)
                    if nxt == cur:
                        in_set = False
                    elif board.won(nxt):
                        in_set = remaining == 1
                    else:
                        sub = _brute(board, nxt, remaining - 1)
                        if sub is _CAPPED:
                            continue
                        in_set = sub is not None and sub + 1 == remaining
                    if in_set != (d_name in plan.optsets[i]):
                        bad += 1
                        print(f"    L{level} step {i}: {d_name} in-set "
                              f"{d_name in plan.optsets[i]}, brute says {in_set}")
                checked_ties += 1
            cur = board.step(cur, DIRS.index(direction))
    if verbose:
        print(f"  whole plans re-proved shortest by brute force: "
              f"{checked_plans} (the rest are out of BFS range); "
              f"exact tie sets brute-forced: {checked_ties}")

    # -- 5: the rejoin sets against the exact ones --------------------------
    compared = subset_ok = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, _heur, state = expert.read(eng)
        plan = expert.plan(eng, level)
        if plan is None or not _exact_sets(expert) or len(plan) > 60:
            continue
        cheap = rejoin_optsets(board, state,
                              [DIRS.index(d) for d in plan])
        for i, (exact, guess) in enumerate(zip(plan.optsets, cheap)):
            compared += 1
            if not set(guess) <= set(exact):
                bad += 1
                print(f"    L{level} step {i}: rejoin labelled {guess}, "
                      f"exact is {exact}")
            elif set(guess) == set(exact):
                subset_ok += 1
    if verbose:
        print(f"  rejoin sets vs exact: {compared} steps compared, "
              f"{subset_ok} identical, none wrong")

    # -- 6: the two tie-set derivations against each other ------------------
    # `measured_optsets` labels the levels that have no field, so the only place
    # it can be checked against a second opinion is a level that has one. Both
    # sides are exact, so they must agree EXACTLY -- not "a subset", which is all
    # the rejoin floor promises.
    agreed = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, heur, state = expert.read(eng)
        plan = expert.plan(eng, level)
        if (plan is None or expert._last_how != "field" or len(plan) > 30
                or board.np != 1):
            # `measured_optsets` searches MACROS, which only exist on a
            # one-player board -- so level 6, the two-player one, is out of
            # scope here exactly as it is out of scope for the ladder rung this
            # check exists to validate.
            continue
        presses = [DIRS.index(d) for d in plan]
        sets, unres = measured_optsets(board, heur, state, presses,
                                       cap=2_000_000, seconds=60.0,
                                       total_seconds=300.0)
        for i, exact in enumerate(plan.optsets):
            if i in unres:
                continue
            agreed += 1
            if sorted(sets[i]) != sorted(exact):
                bad += 1
                print(f"    L{level} step {i}: measured {sets[i]}, "
                      f"field {exact}")
    if verbose:
        print(f"  measured tie sets vs the field's: {agreed} steps agreed")

    # -- 7: the ROOM decomposition the bound is built on --------------------
    bad += _check_rooms(game, expert, rng, verbose=verbose)

    # -- 8: the MACRO decomposition loses nothing ---------------------------
    # The claim behind every ``macro:w1`` plan is that "walk somewhere, then
    # press once" reaches the same states an optimal primitive search would.
    # Checked the only way that settles it: on states near enough to a win for
    # an estimate-free BFS to answer, `_macro_astar` must return exactly the
    # same distance -- and its plan must really win.
    macro_states = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, heur, start = expert.read(eng)
        if board.np != 1:
            continue
        plan = expert.plan(eng, level)
        pool = set()
        if plan is not None:
            chain = [start]
            for d in plan:
                chain.append(board.step(chain[-1], DIRS.index(d)))
            for s in chain[-(brute_limit + 2):]:
                if board.won(s):
                    continue
                pool.add(s)
                for _ in range(20):
                    t = s
                    for _ in range(rng.randrange(1, 3)):
                        t = board.step(t, rng.randrange(5))
                    if not board.won(t):
                        pool.add(t)
        for s in list(pool)[:samples // 2]:
            d = _brute(board, s, brute_limit)
            if d is None or d is _CAPPED:
                continue
            got = _macro_astar(board, heur, s, 3_000_000,
                               time.monotonic() + 60.0, 1)
            macro_states += 1
            if got is None or len(got) != d:
                bad += 1
                print(f"    L{level}: brute says {d}, macro says "
                      f"{None if got is None else len(got)}")
                continue
            cur = s
            for di in got:
                cur = board.step(cur, di)
            if not board.won(cur):
                bad += 1
                print(f"    L{level}: a macro plan does not win")
    if verbose:
        print(f"  macro vs primitive distance: {macro_states} states compared")

    # -- 9: recovery from a perturbed board ---------------------------------
    recovered = dead = gave_up = 0
    for level in range(game.n_levels):
        for _ in range(samples // game.n_levels + 1):
            game.set_level(level)
            for _ in range(rng.randrange(1, 12)):
                eng.step(DIRS[rng.randrange(5)])
                if eng.check_win():
                    break
            if eng.check_win():
                continue
            plan = expert.plan(eng, level)
            if plan is None:
                # "dead" is a PROOF the board can never be won again; anything
                # else is the fast re-plan budget running out, which is not a
                # correctness problem (the episode falls back to a RESET) but is
                # worth counting separately rather than calling it a proof.
                dead += expert._last_how == "dead"
                gave_up += expert._last_how != "dead"
                continue
            for direction in plan:
                eng.step(direction)
            if not eng.check_win():
                bad += 1
                print(f"    L{level}: recovery plan did not win")
            recovered += 1
    if verbose:
        print(f"  recovery: {recovered} perturbed boards re-planned to a WIN, "
              f"{dead} PROVED dead, {gave_up} out of re-plan budget")
    return bad


def _check_rooms(game, expert: VariationsExpert, rng: random.Random,
                 trials: int = 400, verbose: bool = True) -> int:
    """The claim `_Rooms` is built on, checked against the model it bounds.

    A press's effect on a room with no player in it must be exactly
    ``push_squares(that room's squares, the kind the player shoved, the
    direction)`` -- nothing else about the board may reach it. That is what
    makes a room's own distance a lower bound on the whole plan, so it is worth
    measuring rather than arguing: every disagreement here would be a heuristic
    that can over-estimate, i.e. a "provably shortest" plan that is not.

    Also asserts the two cases the room automaton assumes have no effect at all:
    an ACTION press, and a press whose player is not stepping into a square.
    """
    eng = game._engine
    bad = seen = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, _heur, _s = expert.read(eng)
        comps, owner = board.components()
        for _ in range(trials // game.n_levels + 1):
            game.set_level(level)
            board, _heur, state = expert.read(eng)
            for _ in range(rng.randrange(0, 25)):
                nxt = board.step(state, rng.randrange(5))
                if nxt == state or board.won(nxt):
                    break
                state = nxt
            if board.won(state):
                continue
            player_rooms = {owner[p] for p in state[:board.np]}
            for di in range(5):
                after = board.step(state, di)
                kind = None
                if di != _ACTION:
                    at = {c >> 2: c for c in state[board.np:]}
                    for p in state[:board.np]:
                        q = board._edge[p][di]
                        if q >= 0 and q in at:
                            kind = at[q] & 3
                for cid, cells in enumerate(comps):
                    if cid in player_rooms:
                        continue
                    inside = frozenset(cells)
                    before = tuple(sorted(c for c in state[board.np:]
                                          if (c >> 2) in inside))
                    got = tuple(sorted(c for c in after[board.np:]
                                       if (c >> 2) in inside))
                    want = (before if kind is None else
                            board.push_squares(before, kind, di))
                    seen += 1
                    if got != tuple(sorted(want)):
                        bad += 1
                        if bad <= 3:
                            print(f"    L{level} room {cid} {DIRS[di]}: "
                                  f"model {got}, room automaton {want}")
    if verbose:
        print(f"  room independence: {seen} (press, player-free room) pairs "
              f"checked, {bad} violations")
    return bad


def _seat(eng, expert, board: _Board, state) -> None:
    """Put ``state`` on the ENGINE's grid, so the interpreter and the model start
    a fuzz trial from the same board. Walls and goals are left where they are."""
    bg = expert.bg_id
    keep = expert.wall_ids | expert.goal_ids | ({bg} if bg is not None else set())
    for r in range(eng.height):
        for c in range(eng.width):
            eng.grid[r][c] &= keep
    w = eng.width
    pid = sorted(expert.player_ids)[0]
    for p in state[:board.np]:
        eng.grid[p // w][p % w].add(pid)
    for code in state[board.np:]:
        cell, kind = code >> 2, code & 3
        eng.grid[(cell) // w][(cell) % w].add(sorted(expert.kind_ids[kind])[0])
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


# ---------------------------------------------------------------------------
# Render audit
# ---------------------------------------------------------------------------

#: Every cell composition the nine levels can show. A wall never shares a cell
#: with a goal (no level draws one and no rule creates one), and the player can
#: never share a cell with a square or a wall -- they are all on one collision
#: layer -- so those pairs are deliberately absent rather than missing.
_COMPOSITIONS = (
    (),
    ("w",),
    ("omega",),
    ("player",), ("player", "omega"),
    ("a_1",), ("a_1", "omega"),
    ("a_2",), ("a_2", "omega"),
    ("a_3",), ("a_3", "omega"),
    ("a_4",), ("a_4", "omega"),
)


def _audit(verbose: bool = True) -> int:
    """Render every composition, at every board SHAPE the nine levels use, and
    assert the frames are pairwise distinct.

    Whole frames, never a cell crop: `_render_frame` upscales the rendered board
    to fill 64x64 and then letterboxes it, so the cell grid in the output is not
    ``cell_px``-aligned and slicing one cell out of it compares the wrong window
    (see gotcha in `ps-palette-collisions`). Each composition is painted into
    EVERY cell of a board of the given shape instead.

    Also asserts no composition renders as a uniform field of the pad colour 5 --
    the check a pairwise matrix structurally cannot make, because the letterbox
    is not a composition."""
    from adapters.puzzlescript_adapter import _render_frame       # noqa: PLC0415
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    g = game._game
    eng = game._engine
    shapes = {}
    for level in range(game.n_levels):
        game.set_level(level)
        shapes[(eng.height, eng.width)] = level
    bad = 0
    for (h, w), level in sorted(shapes.items()):
        cell_px = max(1, min(64 // h, 64 // w))
        game.set_level(level)
        frames = {}
        for comp in _COMPOSITIONS:
            ids = [g.obj_name_to_idx[n] for n in comp]
            eng.grid = [[set(ids) | {g.obj_name_to_idx["background"]}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            frame = np.asarray(_render_frame(eng, g))
            if len(np.unique(frame)) == 1 and frame.flat[0] == 5:
                bad += 1
                print(f"    {h}x{w} cell_px {cell_px}: {comp or 'floor'} renders "
                      f"as a uniform field of the LETTERBOX colour")
            for other, ref in frames.items():
                if np.array_equal(frame, ref):
                    bad += 1
                    print(f"    {h}x{w} cell_px {cell_px}: "
                          f"{comp or 'floor'} == {other or 'floor'}")
            frames[comp] = frame
        if verbose:
            print(f"  {h:2d}x{w:2d} (cell_px {cell_px}): "
                  f"{len(_COMPOSITIONS)} compositions pairwise distinct"
                  if not bad else f"  {h}x{w}: CLASHES")
    return bad


def _orbit(a, b) -> bool:
    """Is sprite ``b`` sprite ``a`` turned 90 degrees clockwise?"""
    return all(b[r][c] == a[4 - c][r] for r in range(5) for c in range(5))


def _rotation_orbit(verbose: bool = True) -> int:
    """The four square kinds must be a closed rotation ORBIT in the SAME
    direction the ACTION key walks, or the rotation augmentation shows art the
    game cannot produce -- and the mirror must NOT be one, which is why this game
    stays out of `_FLIP_GAMES`."""
    g = PuzzleScriptAdapter(GAME_NAME, seed=0)._game
    art = [g.objects[f"a_{k}"].sprite for k in (1, 2, 3, 4)]
    bad = 0
    for k in range(4):
        if not _orbit(art[k], art[(k + 1) % 4]):
            bad += 1
            print(f"    a_{k + 1} turned 90 degrees is not a_{(k + 1) % 4 + 1}")
    for name in ("player", "omega", "w", "background"):
        s = g.objects[name].sprite
        if not _orbit(s, s):
            bad += 1
            print(f"    {name} is not rotation-symmetric")
    if verbose:
        print(f"  square art is a closed clockwise rotation orbit: {not bad}")
    return bad


def _symmetry(verbose: bool = True) -> int:
    """Replay every plan under all four rotations and assert (a) it still wins --
    which is what tests the `screen_action` remap -- and (b) the frames are
    exactly the rotation of the unaugmented ones, which is what tests that the
    square art is an orbit."""
    _solver, ref_game, expert, _solvable = _new()
    plans = {}
    for level in range(ref_game.n_levels):
        ref_game.set_level(level)
        plan = expert.plan(ref_game._engine, level)
        if plan is not None:
            plans[level] = list(plan)
    walk = [GameAction.ACTION1, GameAction.ACTION5, GameAction.ACTION4,
            GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION5]
    bad = 0
    seen = set()
    ref: dict = {}

    def drive(g, level, presses):
        g.set_level(level)
        out = [np.asarray(g._current_frame)]
        for act in presses:
            fd = g.perform_action(ActionInput(id=act))
            out.append(np.asarray(fd.frame[-1] if fd.frame else g._current_frame))
        return out

    for seed in range(24):
        g = PuzzleScriptAdapter(GAME_NAME, seed=seed)
        for level in sorted(plans):
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            for tag, presses in (
                    ("plan", [screen_action(d, *k) for d in plans[level]]),
                    ("walk", [inverse_remap_action_full(a, *k) for a in walk])):
                frames = drive(g, level, presses)
                if tag == "plan" and g._state != GameState.WIN:
                    print(f"    seed {seed} L{level} {k}: plan did not win")
                    bad += 1
                key = (level, tag)
                if key not in ref:
                    if k != (0, False, False):
                        continue          # nothing to compare against yet
                    ref[key] = frames
                    continue
                if any(not np.array_equal(np.rot90(a, k[0]), b)
                       for a, b in zip(ref[key], frames)):
                    print(f"    seed {seed} L{level} {k}: {tag} frames are not "
                          f"the rotation of the unaugmented ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
    return bad


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--selfcheck" in sys.argv:
        violations = _selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = _audit() + _rotation_orbit()
        print(f"audit: {collisions} problems")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = _symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    sys.exit(VariationsSolver.main())
