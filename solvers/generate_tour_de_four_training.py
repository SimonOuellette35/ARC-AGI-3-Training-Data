"""ps:tour_de_four -- Chris Pickel's "Tour de Four": a sokoban with NO WALLS
whose crates VANISH four at a time, so the board is a machine for destroying
itself and the only thing that can trap you is the edge of the screen.

THE GAME IN THREE RULES

    [ > Movable | Movable ] -> [ > Movable | > Movable ]     push a whole LINE
    (a startloop that floods each mono-coloured component and counts to Four)
    late [Four] [Checking] -> [Four] [Cleared] again          the component dies

`Movable = Player or Block`, `Block = Red or Blue or Yellow`, and the win is
`No Block`. There is no wall object, no target, and no ACTION binding. So one
press does exactly one thing: it shoves the contiguous run of blocks in front of
the player one cell along, and afterwards every 4-connected same-coloured group
of FOUR OR MORE disappears. Nothing else moves, nothing falls, and no rule reads
the X key -- the expert searches four presses, not five.

WHAT THE COUNTING LOOP ACTUALLY COMPUTES. It looks stateful and random and is
neither. The loop marks every block `Unchecked`, picks a `random` unchecked one
as the origin, floods `Checking` outward across like-coloured neighbours while a
Number counter walks One->Two->Three->Four, and if Four is reached converts the
whole `Checking` set to `Cleared` -- which, `Cleared` sharing a collision layer
with the blocks, is how the blocks are deleted. Two consequences:

* The counter stops at Four but the DELETION does not: the three counting rules
  each fire once (the RHS destroys their own LHS Number), and then the unnumbered
  `[Checking Red | Unchecked Red] -> [Checking Red | Checking Red]` runs to
  fixpoint over the rest, so a component of nine dies as one component of nine.
  Four is a THRESHOLD, not a quota.
* The `random` origin decides nothing -- ONCE THE FLOOD IS A FLOOD. Each loop
  pass consumes one whole component and leaves it neither Unchecked nor Checking,
  so the passes partition the board into its components regardless of order, and
  components are disjoint. The model is therefore `remove every mono component of
  size >= 4`, deterministic, and `--selfcheck` measures it: 25k presses over the
  thirteen levels plus 3000 synthetic three-colour block fields (half of them
  with a clear) against the interpreter, 0 divergences.

  That fuzz is what found the ONE thing about this game that is not in its rules,
  and it is a bug in the interpreter rather than in the game: `_execute_late_single`
  runs each of an undirected rule's four expanded directions to a fixpoint but
  never re-runs the direction LIST, so the shipped spread rule only reached as far
  as one up/down/left/right sweep carried it. A winding component was abandoned
  half-marked, its unvisited cells were re-examined later as components of their
  own, came up short of four and SURVIVED the clear -- and which cell survived
  depended on the `random` origin, so a press was not even deterministic. Joining
  the twelve flood rules into one `+` group (which is a no-op in real PuzzleScript,
  where a rule is re-applied until it stops matching) makes the closure a closure
  again; the header of data/puzzlescript_games/_Tour_de_Four.txt has the whole
  story. Before the fix, a 7-cell blob kept one cell and 8% of the fuzzed fields
  diverged; after it, none do.

There is no cascade to model either. A clear only ever REMOVES blocks, and
removing blocks cannot join two components, so the `again` tick that sweeps the
`Cleared` markers away finds nothing new to do. One settle pass is exact.

THE EDGE IS THE WALL, AND THAT IS THE WHOLE PUZZLE

With no wall object the only thing that refuses a push is the board edge: the
run of blocks in front of you moves iff the cell past the last one exists.
Turned around, that is a conservation law the levels are BUILT on, and the
solver would not get past level 12 without it:

    to push a block DOWN you must stand ABOVE it, so a block in ROW 0 can
    never leave row 0. Likewise row h-1, column 0 and column w-1 -- and a
    block in a CORNER can never move again, ever.

So every block carries a permanent LOCUS: a corner block is pinned to its
square, an edge block is pinned to its edge line, an interior block is free
(interior blocks can be pushed onto an edge and are pinned from then on, which
is what makes an edge a trap and not just a boundary). `--confine` fuzzes that
claim against the interpreter over 48k presses and 222k edge-line colour counts:
no block ever leaves the line it is on, and no corner block ever moves.

`_Board.dead` turns the locus into a sound feasibility test. Every block must
end up inside some component of >= 4 of its colour, so for one colour with `m`
blocks: if `0 < m < 4` the level is lost outright, and two blocks whose loci are
`d` apart can only share a component if that component has at least `1 + d`
cells -- a connected set touching both loci is that long. Blocks that pairwise
fail `1 + d <= m` must therefore land in DIFFERENT components, each needing four
blocks of its own, so `m < 4 * omega` is dead, where `omega` is a clique of
mutually-exclusive blocks (a greedy clique, i.e. a LOWER bound on omega, which
is what keeps the prune sound). Level 12 is unsolvable without it: it seeds
yellows in three corners of a 6x6 and the beam spends its whole width on boards
where two of them are already stranded 10 apart with five blocks left to bridge
them. With the prune it wins in 4 seconds. It pays everywhere -- level 7's exact
search goes 449k states -> 120k, and level 2 becomes affordable at all.

THE SEARCH IS OVER MACROS: BEAM FIRST, THEN A BOUND THAT MAKES IT A THEOREM

Nothing in this game happens except when the player presses, and the only press
with an effect on the board is a push, so a shortest press sequence is a
shortest sequence of pushes with shortest walks in between -- exactly a path in
the macro graph `walk to the square behind a run of blocks, then shove it`,
costing `walk + 1`.

The plan comes from a width-capped layered BEAM over those macros, ordered by
`g + W*(h + A*blocks)`, where `h` sums, per colour, an MST over that colour's
components with edge weights `(gap between them) + a constant per merge` -- the
cost of gathering what is left into clumps of four. A sweep of five
`(width, W, A)` settings runs on every level and the shortest win wins. Finding
a win does not end a beam: costs vary inside a layer because the walks do, so it
keeps going with everything that cannot beat the best win pruned off.

Then -- and this is the ordering that matters -- that win's cost becomes the
BOUND for one uniform-cost macro Dijkstra (no heuristic, so there is no
admissibility to get wrong). `search(bound = cost - 1)` never expands a node the
unbounded search would not have, and it lands in one of three places instead of
two: `"win"` means something shorter exists and, Dijkstra popping in cost order,
the thing it found IS the shortest; `"exhausted"` means nothing shorter exists,
so the BEAM's plan is proved shortest without being re-derived; `"capped"` means
the 8M-state budget ran out and nothing is proved. Attempting the exact search
on all thirteen levels is affordable only because the bound is there first.

The result is 8 of 13 provably shortest -- levels 0-2 and 4-8, at 10 / 14 / 19 /
34 / 11 / 23 / 31 / 21 presses -- all of them by exhaustion, because on every
one of those eight the BEAM had already found the optimum by itself and the
Dijkstra's job was only to prove it. The five it cannot reach (3, 9, 10, 11, 12,
at 43 / 30 / 36 / 46 / 50) are 13 to 24 blocks on a 6x6, 6x8 or 7x7 and do not
enumerate: a plain Dijkstra on level 12 was still going at 6M states. Their
plans are measured wins -- `--plans` replays every one through the real
interpreter -- but NOT proved shortest, and every report says which is which.
That the beam matched the exact answer on all eight levels where the truth is
known is the evidence for the other five; `--sweep` re-runs the whole thing per
level so the spread is visible rather than asserted.

OPTIMAL SETS, exact on the proved levels and honest on the rest.

* proved levels: for every step and every one of the four presses, the press is
  offered iff a bounded macro re-solve from the state it lands on still wins in
  the presses that remain. That is a proof in both directions -- nothing optimal
  omitted, nothing second-best offered -- and `--ties` re-derives all of it with
  an independent primitive-press BFS that has no notion of a macro.
* beam levels: the WALK ties, which are exact for what they claim. A walk
  changes nothing but the player, so every shortest route to the square the next
  push is taken from reaches it in the same presses and leaves the identical
  state; those presses are exactly as good as the recorded one. What is unknown
  there is only whether some other push order would be shorter, so the push
  steps carry the single press the plan takes.

Never `optimal=None`: every recorded expert step has a set.

RECOVERY is the family's RESET prefix (`recovery_mode = "reset"`). Clearing is
irreversible and a block shoved into a corner is irreversible on its own, so a
flailed board cannot be re-planned from in general -- and re-planning a beam
level costs minutes, which is why `epsilon` stays 0 (the nuevo_asylum
argument). The exploration prefix flails freely and ONE RESET restores the state
the cached plan was solved from. Plans live in `data/tour_de_four_plans.json`.

ART: nothing had to be redrawn, which is not the usual outcome. Player is
White-on-Black, Red is red/darkred, Blue is blue/purple, Yellow is yellow/brown,
and those land on ARC indices {0,5}, {8,13}, {9,15}, {11,12} -- four flat-out
different pictures. `--audit` checks all six compositions (the three blocks, the
player, bare floor, and the `Cleared` marker the again-tick sweeps before any
frame is drawn) at all five board shapes the levels use, down to 8px cells.

SYMMETRY: the game takes the full 16-presentation group (`_FLIP_GAMES`). It is
gravity-free, the input is screen-relative (`>` on the push rule), the win names
no direction (`No Block`), and -- unlike sokoban_match3 -- the clearing rule has
no engine-grid rule order to argue around at all: the flood is undirected and
runs to a FIXPOINT over components, so what dies does not depend on which of the
four directions the interpreter expands first. The Blue and Yellow sprites are
not rotation-symmetric, but a dot pattern is a picture and not a heading; the
adapter transforms the rendered picture and remaps the input and nothing else.
`--symmetry` replays every level's plan AND a seeded 200-press random walk at
all 16 presentations and requires exact frame equality with the transform.

VERIFIED. 13 of 13 levels, 368 presses, every one of them replayed through the
real interpreter to a WIN, and 8 of the 13 provably shortest. `--selfcheck`:
24960 presses fuzzed over the thirteen levels (5560 exact no-ops, 3091 pushes,
764 blocks cleared) plus 3000 random three-colour block fields (1504 of them
with a clear), 0 divergences from the interpreter -- and it is what found the
flood bug the .txt now fixes. `--confine`: 48666 presses, 222147 edge-line
colour counts and every corner block watched, 0 violations of the law `dead`
prunes with. `--audit`: 6 cell compositions pairwise distinct at all five board
shapes, down to level 3's 8px cells. `--ties`: 333 presses re-derived by a
primitive-press BFS that knows nothing about macros, 0 disagreements in either
direction (1 press left unjudged at the 6M-state budget, and reported as such).
`--symmetry`: all 16 presentations drawn, every level's plan a WIN at every one
of them and every frame of both the plan and a seeded 200-press random walk
exactly the transform of the unaugmented one, 0 violations. Recorded in-process:
3 seeds x 13 levels, 39/39 WIN, 1104 expert steps of which 1104 carry an optimal
set (mean 1.06 presses, the taken press always inside its own set), the only
unlabelled steps being the 39 RESET-prefix explore presses the closing RESET
discards; indices in 0..5; longest level 51 actions against the adapter's
200-press cap. Cold build of the plan cache is ~50 minutes, all of it the
bounded exact searches; after that every seed is a cache hit.

CLI
---
    --selfcheck   the native model fuzzed against the interpreter
    --confine     the edge-confinement law the dead-prune rests on, measured
    --audit       every cell composition renders distinctly, at every cell size
    --symmetry    every presentation is an exact transform of the unaugmented
    --plans       every level's plan, replayed through the interpreter
    --ties        every optimal-action label re-derived by a primitive BFS
    --sweep       the beam parameter sweep, per level (what the plans came from)
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import (        # noqa: E402
    PuzzleScriptAdapter, _render_frame)
from arcengine import ActionInput, GameState        # noqa: E402
from solvers.base_solver import _ID_TO_GAMEACTION   # noqa: E402
from solvers.common.ps_astar import (              # noqa: E402
    PSAStarSolver, PSExpert, Plan, screen_action)
from utils.rotation import inverse_remap_action_full  # noqa: E402

#: The adapter's name for data/puzzlescript_games/_Tour_de_Four.txt. The leading
#: underscore is the file's, not a typo.
GAME_NAME = "_Tour_de_Four"
GAME_ID = "ps:tour_de_four"

#: Engine directions, in the order the model indexes them.
DIRNAMES = ("up", "down", "left", "right")
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: A mono-coloured component this big or bigger dies. Four, as the title says --
#: and the whole of it dies, not four of it (see the module docstring).
MATCH = 4

#: The block colours, as the model codes them. 0 is an empty square.
_COLOURS = (1, 2, 3)

_INF = 1 << 30

#: Beam configurations swept per level, as ``(width, weight, block_bonus)``. The
#: shortest win across the whole sweep is what `solve` keeps.
#:
#: Five, and not one tuned pair, because which configuration works is a property
#: of the LEVEL and not of the game. ``block_bonus`` pays for clearing blocks
#: NOW, which is what cracks a board whose colours are nearly stranded (level 12
#: is 51 presses at ``(1500, 3, 4)`` and NO WIN AT ALL at ``(1500, 1, 0)``);
#: leaving it at 0 is a plain shortest-first search, which is what finds the
#: cheap answer on a board with room to manoeuvre (level 3 is 48 presses there
#: and level 10 41). Running all five costs a couple of minutes on the worst
#: level and the results are cached to disk, so there is no reason to guess.
_BEAM_SWEEP = ((1500, 1, 0), (1500, 3, 4), (6000, 1, 0), (6000, 1, 2),
               (6000, 3, 4))

#: Pushes a beam layer will go to before giving the configuration up. Every plan
#: found is under 25 pushes; this only stops a hopeless configuration wandering.
_BEAM_MAX_PUSH = 60


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level, as a model the search can step in microseconds.

    A state is ``(player, board)`` -- a flat cell index and a ``bytes`` of one
    colour code per cell. There is nothing else: no walls, no targets, no
    counters (the Number/Checking/Unchecked objects exist only inside a single
    `late` loop and are gone before the tick ends), and the player's facing is
    not drawn. `selfcheck` is the proof that this is the whole state.
    """

    def __init__(self, eng, ids):
        grid = eng.grid
        self.h, self.w = len(grid), len(grid[0])
        self.n = n = self.h * self.w
        code = {ids["red"]: 1, ids["blue"]: 2, ids["yellow"]: 3}
        cells = bytearray(n)
        player = None
        for r in range(self.h):
            for c in range(self.w):
                cell, i = grid[r][c], r * self.w + c
                if ids["player"] in cell:
                    player = i
                for oid, col in code.items():
                    if oid in cell:
                        cells[i] = col
        self.state = (player, bytes(cells))

        # Neighbour table. ``None`` off the board -- and off the board is the
        # ONLY obstruction this game has.
        self.nb = [[None] * 4 for _ in range(n)]
        for r in range(self.h):
            for c in range(self.w):
                i = r * self.w + c
                for k, (dr, dc) in enumerate(_DELTA):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.h and 0 <= cc < self.w:
                        self.nb[i][k] = rr * self.w + cc
        self._build_loci()

    # -- the edge-confinement law ------------------------------------------
    def _build_loci(self) -> None:
        """Per-cell LOCUS: every square a block standing here could ever reach.

        A push down needs a pusher above, so a block in row 0 never leaves row
        0; the same on the other three edges, and a corner block never moves at
        all. An interior block is unconstrained (it can be pushed onto an edge
        and pinned there, but that only ever ADDS constraints later, so "any
        square" is a sound over-approximation now). `confine` measures the law
        against the interpreter; `dead` is what spends it.

        ``locdist[i][j]`` is the closest two blocks sitting at ``i`` and ``j``
        could ever be brought -- 0 whenever either of them is free.
        """
        h, w, n = self.h, self.w, self.n
        loci: list[frozenset | None] = []
        for i in range(n):
            r, c = divmod(i, w)
            edge_r, edge_c = r in (0, h - 1), c in (0, w - 1)
            if edge_r and edge_c:
                loci.append(frozenset({i}))                    # a corner: pinned
            elif edge_r:
                loci.append(frozenset(r * w + cc for cc in range(w)))
            elif edge_c:
                loci.append(frozenset(rr * w + c for rr in range(h)))
            else:
                loci.append(None)                              # free
        self.pinned = [loci[i] is not None for i in range(n)]
        dist = [[0] * n for _ in range(n)]
        for i in range(n):
            if loci[i] is None:
                continue
            for j in range(i + 1, n):
                if loci[j] is None:
                    continue
                best = _INF
                for a in loci[i]:
                    ra, ca = divmod(a, w)
                    for b in loci[j]:
                        rb, cb = divmod(b, w)
                        d = abs(ra - rb) + abs(ca - cb)
                        if d < best:
                            best = d
                dist[i][j] = dist[j][i] = best
        self.locdist = dist

    def dead(self, cells) -> bool:
        """True when no sequence of presses can ever clear this board.

        SOUND, never merely plausible -- the exact searches prune with it and
        still claim shortest. Two independent counting arguments, per colour:

        * a clear takes a whole component of four or more, so a colour left with
          one, two or three blocks can never be finished;
        * two blocks whose loci are ``d`` apart can only ever share a component
          if that component spans them, which takes ``1 + d`` cells of that
          colour. Blocks that pairwise fail ``1 + d <= m`` must land in
          different components, and each of those components needs four blocks
          of its own -- so ``m >= 4 * omega`` for any set of ``omega`` pairwise
          incompatible blocks.

        ``omega`` is a greedy clique, which is a LOWER bound on the largest one;
        under-estimating it can only make the prune weaker, never wrong.
        """
        locd = self.locdist
        for col in _COLOURS:
            cs = [i for i in range(self.n) if cells[i] == col]
            m = len(cs)
            if m == 0:
                continue
            if m < MATCH:
                return True
            pinned = [i for i in cs if self.pinned[i]]
            if len(pinned) < 2:
                continue
            omega = 1
            for seed in pinned:
                clique = [seed]
                for x in pinned:
                    if x == seed:
                        continue
                    if all(1 + locd[x][y] > m for y in clique):
                        clique.append(x)
                if len(clique) > omega:
                    omega = len(clique)
            if m < MATCH * omega:
                return True
        return False

    # -- dynamics -----------------------------------------------------------
    def settle(self, cells):
        """The `late` loop, in one pass: delete every 4-connected mono component
        of `MATCH` or more.

        One pass is exact and a second would change nothing, because a clear only
        ever REMOVES blocks and removing blocks cannot connect two components.
        Whole components die, not four of them: the counter's three rules fire
        once each and the unnumbered spread rule then runs to fixpoint over the
        rest. Both facts are measured by `selfcheck`.
        """
        nb, n = self.nb, self.n
        seen = bytearray(n)
        out = None
        for i in range(n):
            col = cells[i]
            if col == 0 or seen[i]:
                continue
            comp = [i]
            seen[i] = 1
            qi = 0
            while qi < len(comp):
                x = comp[qi]
                qi += 1
                for y in nb[x]:
                    if y is not None and not seen[y] and cells[y] == col:
                        seen[y] = 1
                        comp.append(y)
            if len(comp) >= MATCH:
                if out is None:
                    out = bytearray(cells)
                for x in comp:
                    out[x] = 0
        return cells if out is None else bytes(out)

    def step(self, st, k):
        """One press. Returns the SAME tuple when the press does nothing -- off
        the board, or into a run of blocks that reaches the edge. ACTION is not
        modelled because no rule reads it.

        The push is TRANSITIVE (`[> Movable | Movable]`), so the whole contiguous
        run in front of the player travels, however long and whatever colours it
        mixes; it travels iff the square past its far end exists.
        """
        p, cells = st
        nxt = self.nb[p][k]
        if nxt is None:
            return st
        if cells[nxt] == 0:
            return (nxt, cells)
        run = [nxt]
        x = nxt
        while True:
            y = self.nb[x][k]
            if y is None:                       # the run reaches the edge
                return st
            if cells[y] == 0:
                break
            run.append(y)
            x = y
        moved = bytearray(cells)
        moved[nxt] = 0
        for c in run:
            moved[self.nb[c][k]] = cells[c]
        return (nxt, self.settle(bytes(moved)))

    def won(self, st) -> bool:
        """`No Block` (and `No Cleared`, which the again-tick has already swept
        by the time a press returns). No level starts won."""
        return not any(st[1])

    # -- macros -------------------------------------------------------------
    def walk_tree(self, st):
        """BFS parent pointers over the squares the player may WALK on: empty and
        on the board. Stepping into a block is a push, which is a macro."""
        p, cells = st
        parent = {p: None}
        q = deque([p])
        while q:
            x = q.popleft()
            for k in range(4):
                y = self.nb[x][k]
                if y is None or cells[y] or y in parent:
                    continue
                parent[y] = (x, k)
                q.append(y)
        return parent

    @staticmethod
    def route(parent, cell) -> list:
        out = []
        while parent[cell] is not None:
            cell, k = parent[cell]
            out.append(k)
        out.reverse()
        return out

    def macros(self, st):
        """Every ``walk to a square, then shove what is next to it``, as
        ``(walk presses, push direction, resulting state)``.

        This is what makes the search affordable and it costs nothing in
        optimality: no press in this game does anything except move the player,
        so a shortest press sequence is a shortest sequence of pushes with
        shortest walks between them -- which is a shortest macro path.
        """
        parent = self.walk_tree(st)
        cells = st[1]
        out = []
        for stance in parent:
            for k in range(4):
                nxt = self.nb[stance][k]
                if nxt is None or cells[nxt] == 0:
                    continue                       # not a push
                ns = self.step((stance, cells), k)
                if ns[0] == stance:                # the run hit the edge
                    continue
                out.append((self.route(parent, stance), k, ns))
        return out

    # -- exact search -------------------------------------------------------
    def search(self, st, bound=None, state_cap=_INF):
        """Uniform-cost macro Dijkstra. Returns ``(macros, cost, status)`` with
        ``status`` one of ``"win"`` / ``"exhausted"`` / ``"capped"``.

        The status is the point: ``"exhausted"`` is a PROOF that no win exists
        within ``bound``, which is what both the optimality claim and the
        optimal-set measurement are made of, while ``"capped"`` proves nothing.
        A win is returned when its node is POPPED, never when it is generated:
        macros have different costs, so a 2-press macro off a g=30 node beats a
        9-press one off a g=24.
        """
        if self.won(st):
            return [], 0, "win"
        dist = {st: 0}
        parent = {st: None}
        pq = [(0, 0, st)]
        counter = 0
        while pq:
            g, _c, s = heapq.heappop(pq)
            if g > dist.get(s, _INF):
                continue
            if bound is not None and g > bound:
                return None, None, "exhausted"
            if self.won(s):
                out = []
                while parent[s] is not None:
                    s, walk, push = parent[s]
                    out.append((walk, push))
                out.reverse()
                return out, g, "win"
            if len(dist) > state_cap:
                return None, None, "capped"
            for walk, push, ns in self.macros(s):
                ng = g + len(walk) + 1
                if bound is not None and ng > bound:
                    continue
                if ng >= dist.get(ns, _INF):
                    continue
                if self.dead(ns[1]):
                    continue
                dist[ns] = ng
                parent[ns] = (s, walk, push)
                counter += 1
                heapq.heappush(pq, (ng, counter, ns))
        return None, None, "exhausted"

    # -- beam ---------------------------------------------------------------
    def heuristic(self, st, per_merge=2) -> int:
        """What is left to do, as ``gather the survivors into clumps of four``.

        Per colour: its blocks form components (none of size >= MATCH, or they
        would have cleared), those components have to be merged until every
        surviving clump holds four, and a merge costs at least the gap between
        the two components. So take an MST over the components under
        ``gap = (nearest cells' Manhattan distance) - 1``, drop the most
        expensive edges down to the largest number of clumps the block count can
        support, and charge ``per_merge`` on top for the walking each merge costs.

        Inadmissible on purpose (one push can close two gaps at once when it
        shoves a run toward two different clumps), so it steers the beam and is
        never used by `search`, which is uniform-cost.
        """
        cells = st[1]
        total = 0
        for col in _COLOURS:
            comps = self.components(cells, col)
            j = len(comps)
            if j < 2:
                continue
            m = sum(len(c) for c in comps)
            clumps = max(1, min(m // MATCH, j // 2))
            edges = []
            for a in range(j):
                for b in range(a + 1, j):
                    edges.append((self.gap(comps[a], comps[b]), a, b))
            edges.sort()
            par = list(range(j))

            def find(x):
                while par[x] != x:
                    par[x] = par[par[x]]
                    x = par[x]
                return x

            kept = []
            for wgt, a, b in edges:
                ra, rb = find(a), find(b)
                if ra != rb:
                    par[ra] = rb
                    kept.append(wgt)
            kept.sort()
            total += sum(w + per_merge for w in kept[:max(0, j - clumps)])
        return total

    def components(self, cells, col) -> list:
        seen = set()
        out = []
        for i in range(self.n):
            if cells[i] != col or i in seen:
                continue
            comp = [i]
            seen.add(i)
            qi = 0
            while qi < len(comp):
                x = comp[qi]
                qi += 1
                for y in self.nb[x]:
                    if y is not None and y not in seen and cells[y] == col:
                        seen.add(y)
                        comp.append(y)
            out.append(comp)
        return out

    def gap(self, a, b) -> int:
        """Presses' worth of distance between two components: how far one of
        them has to travel before they touch."""
        w = self.w
        best = _INF
        for x in a:
            rx, cx = divmod(x, w)
            for y in b:
                ry, cy = divmod(y, w)
                d = abs(rx - ry) + abs(cx - cy)
                if d < best:
                    best = d
        return best - 1

    def beam(self, st, width, weight, block_bonus, max_push=_BEAM_MAX_PUSH):
        """Layered beam over the macros, one layer per PUSH, ordered by
        ``g + weight * (heuristic + block_bonus * blocks left)``.

        Layering on pushes rather than presses is what keeps a layer comparable:
        every state in it has had the same number of chances to change the
        board. Costs still vary INSIDE a layer, though (the walks between pushes
        do), so finding a win does not end the search -- a later layer can hold
        a cheaper one. It keeps going instead, with everything that could not
        possibly beat the best win pruned away: a state at ``g`` needs at least
        one more press, so it is worth expanding only while ``g + 1 < best``.
        That collapses the frontier within a layer or two of the first win and
        costs almost nothing.

        Any win it returns is a genuine win (`--plans` replays it through the
        interpreter) but NOT a proved shortest one; `solve` is what tries to
        prove or beat it.
        """
        if self.won(st):
            return [], 0
        layer = {st: (0, None)}
        best = None
        for _ in range(max_push):
            nxt = {}
            for s, (g, node) in layer.items():
                for walk, push, ns in self.macros(s):
                    ng = g + len(walk) + 1
                    if self.won(ns):
                        if best is None or ng < best[0]:
                            best = (ng, (node, walk, push))
                        continue
                    if best is not None and ng >= best[0] - 1:
                        continue           # cannot reach a shorter win than best
                    if self.dead(ns[1]):
                        continue
                    old = nxt.get(ns)
                    if old is None or ng < old[0]:
                        nxt[ns] = (ng, (node, walk, push))
            if not nxt:
                break
            ranked = sorted(nxt.items(),
                            key=lambda kv: (kv[1][0] + weight *
                                            (self.heuristic(kv[0]) + block_bonus *
                                             sum(1 for x in kv[0][1] if x))))
            layer = dict(ranked[:width])
        if best is None:
            return None, None
        out = []
        node = best[1]
        while node is not None:
            prev, walk, push = node
            out.append((walk, push))
            node = prev
        out.reverse()
        return out, best[0]

    # -- the plan -----------------------------------------------------------
    def solve(self, state_cap, sweep=_BEAM_SWEEP, verbose=False):
        """``(macros, cost, proved)`` -- the plan, and whether it is a THEOREM.

        The beam runs FIRST, every configuration of the sweep, and the shortest
        win it finds becomes the bound for a single exact macro Dijkstra. That
        ordering is what makes the exact search affordable to attempt on all
        thirteen levels instead of six: `search(bound=cost-1)` never expands a
        node the unbounded search would not have, and it ends in one of three
        useful places rather than two.

        * ``"win"`` -- there is something shorter, and because Dijkstra pops in
          cost order the thing it found is the shortest there is. The beam plan
          is thrown away and this is proved. No shipped level takes this branch
          -- the beam matches the optimum on all eight it can prove -- but it is
          what catches a level where the beam falls short, and it costs nothing
          to leave in: the bounded search has to run either way.
        * ``"exhausted"`` -- nothing shorter exists, so the BEAM plan is proved
          shortest without ever being re-derived.
        * ``"capped"`` -- the state budget ran out and nothing is proved; the
          beam plan stands as a measured win (`--plans` replays it through the
          interpreter) that may not be shortest.
        """
        best, best_cost = None, _INF
        for width, weight, bonus in sweep:
            got, got_cost = self.beam(self.state, width, weight, bonus)
            if verbose:
                print(f"    beam w={width} W={weight} A={bonus}: "
                      f"{'no win' if got is None else str(got_cost) + ' presses'}",
                      flush=True)
            if got is not None and got_cost < best_cost:
                best, best_cost = got, got_cost
        if best is None:
            # The beam found nothing at all: either the level is genuinely lost
            # (`dead` proves several of them are, from states a rollout reaches)
            # or the sweep is too narrow. An unbounded exact search settles it.
            plan, cost, status = self.search(self.state, state_cap=state_cap)
            if verbose:
                print(f"    no beam win; unbounded exact: {status}", flush=True)
            return (plan, cost, True) if status == "win" else (None, None, False)
        plan, cost, status = self.search(self.state, bound=best_cost - 1,
                                         state_cap=state_cap)
        if verbose:
            print(f"    exact under {best_cost - 1}: {status}"
                  f"{'' if plan is None else f' at {cost}'}", flush=True)
        if status == "win":
            return plan, cost, True
        return best, best_cost, status == "exhausted"

    # -- optimal sets -------------------------------------------------------
    @staticmethod
    def flatten(macros) -> list:
        return [d for walk, push in macros for d in walk + [push]]

    def walk_optsets(self, macros) -> list:
        """Per-press optimal sets that are exact for what they claim: every
        shortest route to the square the next push is taken from.

        A walk changes nothing but the player, so two shortest routes to the same
        stance reach it in the same presses and leave the identical state -- each
        of their presses is exactly as good as the recorded one. The push press
        itself carries only the press the plan takes, because whether some other
        push order is shorter is precisely what a beam plan does not know.
        """
        st = self.state
        out = []
        for walk, push in macros:
            cells = st[1]
            stance = st[0]
            for k in walk:                     # where this macro is heading
                stance = self.nb[stance][k]
            # distance from every free square to the stance, on this fixed board
            dist = {stance: 0}
            q = deque([stance])
            while q:
                x = q.popleft()
                for y in self.nb[x]:
                    if y is None or cells[y] or y in dist:
                        continue
                    dist[y] = dist[x] + 1
                    q.append(y)
            p = st[0]
            for k in walk:
                here = dist[p]
                best = []
                for kk in range(4):
                    y = self.nb[p][kk]
                    if y is None or cells[y]:
                        continue
                    if dist.get(y, _INF) == here - 1:
                        best.append(DIRNAMES[kk])
                assert DIRNAMES[k] in best
                out.append(best)
                p = self.nb[p][k]
            out.append([DIRNAMES[push]])
            # the shove is taken FROM the stance, not from where the macro began
            st = self.step((stance, cells), push)
        return out

    def exact_optsets(self, presses, state_cap) -> list | None:
        """Per-press optimal sets MEASURED, exact in both directions -- or None
        if a bounded re-solve ran out of state budget.

        A press is offered at step ``i`` iff the state it lands on can still be
        won in ``len(presses) - i - 1`` presses, which a bounded macro Dijkstra
        either exhibits or refutes by exhausting its frontier. Legitimate only
        because the plan is shortest: that is what makes "one press cheaper" the
        only thing a successor could possibly manage.
        """
        st = self.state
        out = []
        for i, taken in enumerate(presses):
            left = len(presses) - i - 1
            best = []
            for k in range(4):
                ns = self.step(st, k)
                if ns == st:                       # a press that does nothing
                    continue
                if self.won(ns):
                    if left == 0:
                        best.append(DIRNAMES[k])
                    continue
                if not left or self.dead(ns[1]):
                    continue
                _p, _c, status = self.search(ns, bound=left, state_cap=state_cap)
                if status == "capped":
                    return None
                if status == "win":
                    best.append(DIRNAMES[k])
            assert DIRNAMES[taken] in best, (i, taken, best)
            out.append(best)
            st = self.step(st, taken)
        return out


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class TourExpert(PSExpert):
    """`PSExpert`'s plan memo, restore discipline and disk cache around the
    native search. `heuristic` is never called -- tier 1 is uniform-cost and
    tier 2 uses `_Board.heuristic` on the native board."""

    directions = list(DIRNAMES)                    # ACTION is dead: no rule reads it
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "tour_de_four_plans.json")

    OBJECTS = ("player", "red", "blue", "yellow")

    #: Macro states ONE bounded exact search may enumerate before it gives the
    #: level up as unproved. Level 6 spends 7.5M of them (~3 minutes, ~2.5 GB)
    #: to exhaust everything under 23 presses, which is what turns its beam plan
    #: from a win into the shortest win; drop this and five more levels lose
    #: their proof while keeping the identical plan.
    state_cap: int = 8_000_000

    #: The budget for ONE bounded re-solve inside `exact_optsets`, which runs up
    #: to three of them per press of the plan. Deliberately far smaller than
    #: `state_cap`: a level whose plan search costs minutes would cost hours to
    #: label this way, and the fallback (`walk_optsets`) is exact for what it
    #: claims, so the trade is a weaker LABEL on a plan that is still proved
    #: shortest -- never a weaker plan.
    optset_cap: int = 800_000

    def setup(self):
        self.ids = {n: self.g.obj_name_to_idx[n] for n in self.OBJECTS}
        #: level -> {"kind", "proved", "exact_sets", "pushes"}; filled by `plan`
        #: from the disk entry, so a cache hit in a later process reports the
        #: same thing the search that found it did.
        self.stats = {}
        self.verbose = False
        self._last = None

    def heuristic(self, eng):
        raise AssertionError(
            "TourExpert plans on its native model; heuristic is unused")

    def board(self, eng) -> _Board:
        return _Board(eng, self.ids)

    def _search(self, eng):
        """Tier 1 (exact macro Dijkstra) if it fits the budget, tier 2 (the beam
        sweep) if it does not; then the strongest optimal-set labelling the
        remaining budget can pay for. `self._last` carries the provenance out to
        `plan`, which is what writes it into the disk cache."""
        board = self.board(eng)
        macros, cost, proved = board.solve(self.state_cap, verbose=self.verbose)
        if macros is None:
            self._last = {"kind": "unsolved", "proved": False,
                          "exact_sets": False, "pushes": 0}
            return None
        presses = board.flatten(macros)
        assert len(presses) == cost
        sets = (board.exact_optsets(presses, self.optset_cap)
                if proved else None)
        exact_sets = sets is not None
        if sets is None:
            sets = board.walk_optsets(macros)
        self._last = {"kind": "exact" if proved else "beam", "proved": proved,
                      "exact_sets": exact_sets, "pushes": len(macros)}
        return Plan([DIRNAMES[k] for k in presses], sets)

    def plan(self, eng, level: int | None = None):
        """`PSExpert.plan`, plus the provenance of the answer.

        The disk cache stores only ``start`` / ``plan`` / ``optsets``, so a later
        process that hits it would otherwise have no idea whether the plan it is
        replaying was proved shortest or beamed. The extra keys ride along in the
        same entry and are filled in the first time a level is searched."""
        self._last = None
        found = super().plan(eng, level)
        if level is None:
            return found
        entry = self._disk.get(level)
        if entry is None:
            return found
        if "proved" not in entry and self._last is not None:
            entry.update(self._last)
            self._save_disk()
        self.stats[level] = {
            "kind": entry.get("kind", "?"),
            "proved": bool(entry.get("proved")),
            "exact_sets": bool(entry.get("exact_sets")),
            "pushes": entry.get("pushes", 0),
        }
        return found


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class TourSolver(PSAStarSolver):
    game_id = GAME_ID
    game_name = GAME_NAME
    game_module_id = GAME_ID
    expert_cls = TourExpert
    #: The longest plan is 50 presses (level 12) and the longest level recorded
    #: is 51 actions; this bounds the replay only, well inside the adapter's own
    #: 200-press level budget.
    max_steps = 140


# ---------------------------------------------------------------------------
# --selfcheck: the model against the interpreter
# ---------------------------------------------------------------------------

def _read_engine(eng, ids, w):
    """The interpreter's board, in the model's own terms."""
    player, cells = None, bytearray(len(eng.grid) * w)
    code = {ids["red"]: 1, ids["blue"]: 2, ids["yellow"]: 3}
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            i = r * w + c
            if ids["player"] in cell:
                player = i
            for oid, col in code.items():
                if oid in cell:
                    cells[i] = col
    return (player, bytes(cells))


def selfcheck(trials=12, steps=80, boards=3000, verbose=True):
    """Fuzz the model against the real interpreter, two ways.

    ROLLOUTS from every level start are the honest fuzz: random play shoves runs
    of blocks into the edge (which must be an exact no-op, the game's only
    refusal), presses ACTION (which must do nothing at all), strands blocks in
    corners and lines four of them up by accident. Guided play draws mostly from
    presses the model believes move a block so the rollouts actually reach the
    clearing rule instead of spending themselves walking; uniform play stays in
    because it is what covers the no-op classification.

    But thirteen hand-made levels barely exercise `settle`, which is the half of
    this game that is not sokoban. So the second mode loads RANDOM three-colour
    block fields, with the player sealed in a one-square pocket so its press can
    move nothing, and compares the board the `late` loop leaves. That is where
    the two facts the model turns on are measured: that a component of five or
    nine dies WHOLE (a model deleting only four of it misses), and that the
    `random` origin decides nothing.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    ids = {n: game._game.obj_name_to_idx[n] for n in TourExpert.OBJECTS}
    bad = presses = noops = moved = cleared = wins = 0

    for level in range(game.n_levels):
        for guided in (False, True):
            for t in range(trials):
                game.set_level(level)
                board = _Board(eng, ids)
                st = board.state
                rng = random.Random(f"{GAME_ID}:selfcheck:{level}:{guided}:{t}")
                for _ in range(steps):
                    nexts = [board.step(st, k) for k in range(4)]
                    shove = [k for k in range(4) if nexts[k][1] != st[1]]
                    live = [k for k in range(4) if nexts[k] != st]
                    roll = rng.random()
                    if guided and shove and roll < 0.6:
                        k = rng.choice(shove)
                    elif guided and live and roll < 0.95:
                        k = rng.choice(live)
                    else:
                        k = rng.randrange(5)          # 5 = ACTION, a no-op
                    before = st
                    eng.step("action" if k == 4 else DIRNAMES[k])
                    st = st if k == 4 else board.step(st, k)
                    presses += 1
                    if st == before:
                        noops += 1
                    else:
                        if st[1] != before[1]:
                            moved += 1
                        cleared += (sum(1 for x in before[1] if x) -
                                    sum(1 for x in st[1] if x))
                    if (_read_engine(eng, ids, board.w) != st
                            or eng.check_win() != board.won(st)):
                        bad += 1
                        if bad < 4:
                            print(f"  L{level} t{t} guided={guided}: model {st}"
                                  f" != engine {_read_engine(eng, ids, board.w)}")
                        break
                    if board.won(st):
                        wins += 1
                        break

    # -- the clearing rule on its own, over random block fields -------------
    h, w = 8, 8
    bg = game._game.obj_name_to_idx["background"]
    codes = [ids["red"], ids["blue"], ids["yellow"]]
    field = [(r, c) for r in range(2, 8) for c in range(0, 8)]
    board = None
    rng = random.Random(f"{GAME_ID}:selfcheck:fields")
    fields = fclear = 0
    for _ in range(boards):
        # Two colours much of the time: three colours on 48 squares rarely makes
        # a component of five, and components of five are the point.
        palette = codes[:rng.choice((2, 2, 3))]
        placed = {(r, c): rng.choice(palette)
                  for (r, c) in rng.sample(field, rng.randint(8, 34))}
        # Rows 0 and 1 stay empty and the player sits in the corner, so its one
        # press is a guaranteed no-op and what is measured is the LATE loop
        # alone, on a field no plan would ever reach.
        grid = [[{bg} for _ in range(w)] for _ in range(h)]
        grid[0][0] = {bg, ids["player"]}
        for (r, c), col in placed.items():
            grid[r][c].add(col)
        eng.load_level(grid)
        if board is None:
            board = _Board(eng, ids)
        eng.step("up")                            # a guaranteed no-op press
        flat = bytearray(h * w)
        for (r, c), col in placed.items():
            flat[r * w + c] = {ids["red"]: 1, ids["blue"]: 2, ids["yellow"]: 3}[col]
        mine = board.settle(bytes(flat))
        live = _read_engine(eng, ids, w)[1]
        fields += 1
        if live != bytes(flat):
            fclear += 1
        if live != mine:
            bad += 1
            if bad < 8:
                print(f"  field fuzz: model != engine")
    if verbose:
        print(f"  {presses} presses fuzzed over {game.n_levels} levels "
              f"({noops} no-ops, {moved} pushes, {cleared} blocks cleared, "
              f"{wins} accidental wins)")
        print(f"  {fields} random block fields settled "
              f"({fclear} of them with a clear)")
    return bad


# ---------------------------------------------------------------------------
# --confine: the edge law the dead-prune rests on
# ---------------------------------------------------------------------------

def confine(trials=25, steps=150, verbose=True):
    """Measure the claim `_Board._build_loci` is built on, on the INTERPRETER: a
    block on an edge never leaves that edge, and a corner block never moves.

    This is the one thing the solver believes that is not a direct reading of a
    rule -- it is an argument about how far a `>` force can reach -- and `dead`
    is what spends it, so an exact "provably shortest" claim rests on it.

    What makes it decidable without tracking block identity is the clear: on a
    press that clears NOTHING the total block count is unchanged, so a colour's
    count on an edge line can only fall if a block of that colour left the line,
    which is exactly the violation. Presses that DO clear are counted and
    skipped, since a clear may legitimately empty part of a line. The corners get
    the stronger check: while a corner holds a block nothing can push another one
    into it (the run would have to start off the board), so its colour must be
    constant until a clear.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    ids = {n: game._game.obj_name_to_idx[n] for n in TourExpert.OBJECTS}
    bad = presses = clean = skipped = watched = 0

    def lines(h, w):
        yield [c for c in range(w)]                                # row 0
        yield [(h - 1) * w + c for c in range(w)]                  # row h-1
        yield [r * w for r in range(h)]                            # column 0
        yield [r * w + (w - 1) for r in range(h)]                  # column w-1

    for level in range(game.n_levels):
        for t in range(trials):
            game.set_level(level)
            board = _Board(eng, ids)
            h, w = board.h, board.w
            corners = (0, w - 1, (h - 1) * w, h * w - 1)
            rng = random.Random(f"{GAME_ID}:confine:{level}:{t}")
            for _ in range(steps):
                before = _read_engine(eng, ids, w)[1]
                if not any(before):
                    break
                eng.step(DIRNAMES[rng.randrange(4)])
                after = _read_engine(eng, ids, w)[1]
                presses += 1
                if sum(1 for x in after if x) != sum(1 for x in before if x):
                    skipped += 1                    # a clear: not decidable here
                    continue
                clean += 1
                for line in lines(h, w):
                    for col in _COLOURS:
                        b = sum(1 for i in line if before[i] == col)
                        if b == 0:
                            continue
                        watched += 1
                        if sum(1 for i in line if after[i] == col) < b:
                            bad += 1
                            if bad < 5:
                                print(f"  L{level}: a block of colour {col} LEFT "
                                      f"the edge line {line[0]}..{line[-1]}")
                for i in corners:
                    if before[i] and after[i] != before[i]:
                        bad += 1
                        if bad < 5:
                            print(f"  L{level}: the CORNER block at "
                                  f"{divmod(i, w)} moved")
    if verbose:
        print(f"  {presses} presses ({clean} with no clear, {skipped} skipped "
              f"because a clear made the line counts undecidable)")
        print(f"  {watched} edge-line colour counts checked, {bad} violations")
    return bad


# ---------------------------------------------------------------------------
# --audit: every cell composition, pixel-distinct at every cell size
# ---------------------------------------------------------------------------

#: Every stack of objects a square in this game can hold. Player, the three
#: blocks and Cleared all share ONE collision layer, so no two of them ever
#: meet and there is nothing else on the board.
_AUDIT_CASES = {
    "floor": (),
    "player": ("player",),
    "red": ("red",),
    "blue": ("blue",),
    "yellow": ("yellow",),
    "cleared": ("cleared",),
}


def audit(verbose=True):
    """Assert every cell COMPOSITION renders differently at every cell size the
    levels use.

    `cleared` is in the list although the again-tick sweeps it before any frame
    is drawn: it is drawn in the Player's own white, and the check is what would
    catch a future change that let it survive a press.

    The board is FILLED with the composition and whole frames are compared,
    rather than one cell being sliced out by ``cell_px`` arithmetic:
    `_render_frame` upscales a sub-64 render to fill the frame and then
    letterboxes it, so the cell grid in the output is not ``cell_px``-aligned.
    Rendering is per-cell independent, so a whole-frame comparison is the same
    test done where the geometry cannot drift."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, parsed = game._engine, game._game
    idx = parsed.obj_name_to_idx
    sizes = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for name, objs in _AUDIT_CASES.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, parsed))
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        if verbose:
            print(f"  {h}x{w} (cell {64 // max(h, w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): "
                  f"{len(_AUDIT_CASES)} compositions, "
                  f"{'all distinct' if not clashes else 'IDENTICAL ' + str(clashes)}")
    return bad


# ---------------------------------------------------------------------------
# --symmetry: the presentation augmentation is a true symmetry of the mechanic
# ---------------------------------------------------------------------------

def symmetry(walk_presses=200, verbose=True):
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unaugmented ones.

    This is the evidence for putting the game in `_FLIP_GAMES`. The structural
    argument is the ESCAPE! one -- gravity-free, screen-relative input (`>` on
    the push rule), a win condition that names no direction (`No Block`), and no
    sprite that encodes a facing. Unlike sokoban_match3 there is not even an
    engine-grid rule ORDER to argue around: the clearing flood is undirected and
    runs to a fixpoint over whole components, so which blocks die cannot depend
    on which of the four directions the interpreter expands first. What this
    measures is exactly that: the same engine directions driven at all 16
    presentations, frames compared against the unaugmented run.

    Both the PLANS and a seeded random walk are replayed -- the walk reaches the
    boards a plan never visits (runs of blocks jammed against the edge, blocks
    stranded in corners, components of six cleared at once) and it presses the
    unbound ACTION key too."""
    solver = TourSolver()
    game, expert, _ = solver._ensure(0)
    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        plans[level] = expert.plan(game._engine, level)

    def drive(g, level, presses):
        g.set_level(level)
        out = [np.asarray(g._current_frame)]
        for act in presses:
            fd = g.perform_action(ActionInput(id=act))
            out.append(np.asarray(fd.frame[-1] if fd.frame
                                  else g._current_frame))
        return out

    def transform(frame, k, hflip, vflip):
        if k:
            frame = np.rot90(frame, k=k)
        if hflip:
            frame = np.fliplr(frame)
        if vflip:
            frame = np.flipud(frame)
        return np.ascontiguousarray(frame)

    ref, seen, bad = {}, set(), 0
    for seed in range(400):
        if len(seen) == 16 and seed > 40:
            break
        g = solver.make_game(seed)
        for level in range(g.n_levels):
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            rng = random.Random(f"{GAME_ID}:symmetry:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(walk_presses)]
            plan = plans[level] or []
            for tag, presses in (("plan", [screen_action(d, *k) for d in plan]),
                                 ("walk", [inverse_remap_action_full(a, *k)
                                           for a in walk])):
                frames = drive(g, level, presses)
                if tag == "plan" and plan and g._state != GameState.WIN:
                    print(f"  seed {seed} L{level} {k}: plan did not win")
                    bad += 1
                key = (level, tag)
                if key not in ref:
                    if k != (0, False, False):
                        continue          # nothing to compare against yet
                    ref[key] = frames
                    continue
                if any(not np.array_equal(transform(a, *k), b)
                       for a, b in zip(ref[key], frames)):
                    print(f"  seed {seed} L{level} {k}: {tag} frames are not "
                          f"the transform of the unaugmented ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
    return bad


# ---------------------------------------------------------------------------
# --ties: optimal-action labels re-derived by an independent primitive BFS
# ---------------------------------------------------------------------------

def ties_report(levels=None, state_cap=6_000_000, verbose=True):
    """Re-derive every step's optimal SET with a bounded primitive-press BFS and
    require it to equal the label the plan carries.

    Only meaningful on the levels whose plan is PROVED shortest, and only those
    are checked: there the label claims to be every press that still finishes in
    the presses that remain, and this BFS -- which has no notion of a macro, a
    stance or a walk -- answers the same question by pressing all four keys
    breadth-first. Agreement in both directions is the claim: a press the labels
    omit and the BFS finds would be a right answer taught as wrong, and one the
    labels offer and the BFS refutes would be a second-best press taught as
    optimal.

    Beam levels are LISTED and skipped, because their labels deliberately claim
    something weaker (see `_Board.walk_optsets`).
    """
    solver = TourSolver()
    game, expert, _ = solver._ensure(0)
    eng = game._engine
    bad = checked = inconclusive = 0
    skipped = []

    def wins_within(board, st, limit):
        """True / False / None -- None meaning the BFS ran out of state budget
        and this press could not be adjudicated, which is reported rather than
        silently counted as agreement."""
        if board.won(st):
            return True
        seen, frontier = {st}, [st]
        for _ in range(limit):
            nxt = []
            for s in frontier:
                for k in range(4):
                    ns = board.step(s, k)
                    if ns in seen:
                        continue
                    if board.won(ns):
                        return True
                    seen.add(ns)
                    nxt.append(ns)
            if len(seen) > state_cap:
                return None
            frontier = nxt
            if not frontier:
                break
        return False

    for level in (levels if levels is not None else range(game.n_levels)):
        game.set_level(level)
        board = expert.board(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            continue
        if not expert.stats.get(level, {}).get("exact_sets"):
            skipped.append(level)
            continue
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        st = board.state
        for i, taken in enumerate(plan):
            left = len(plan) - i - 1
            truth, unknown = [], False
            for k in range(4):
                ns = board.step(st, k)
                if ns == st:                       # a press that does nothing
                    continue
                checked += 1
                got = wins_within(board, ns, left)
                if got is None:
                    unknown = True
                    inconclusive += 1
                elif got:
                    truth.append(DIRNAMES[k])
            if not unknown and sorted(truth) != sorted(sets[i]):
                bad += 1
                print(f"  L{level} step {i}: labels {sorted(sets[i])} != "
                      f"BFS truth {sorted(truth)}")
            st = board.step(st, DIRNAMES.index(taken))
        if verbose:
            print(f"  L{level}: {len(plan)} steps re-derived")
    if verbose:
        print(f"  {checked} presses re-derived by primitive BFS, "
              f"{bad} disagreements, {inconclusive} over the "
              f"{state_cap} state budget and left unjudged")
        if skipped:
            print(f"  labels are walk-ties rather than exact sets, so not "
                  f"checked: {skipped}")
    return bad


# ---------------------------------------------------------------------------
# --sweep / --plans
# ---------------------------------------------------------------------------

def sweep_report():
    """Every beam configuration on every level the exact search cannot reach --
    what the plans in the cache actually came from, and how sensitive they are.

    A beam whose answer is the same at every width and weight is evidence the
    plan is not junk even where it is not proved; one that swings is a level to
    distrust. Slow (it re-runs the whole sweep, ignoring the plan cache)."""
    solver = TourSolver()
    game, expert, _ = solver._ensure(0)
    eng = game._engine
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        t0 = time.time()
        _p, cost, status = board.search(board.state, state_cap=expert.state_cap)
        if status == "win":
            print(f"  L{level:2d}: exact {cost} presses "
                  f"({time.time() - t0:.0f}s) -- PROVED SHORTEST")
            continue
        print(f"  L{level:2d}: exact search {status} after "
              f"{time.time() - t0:.0f}s; beam sweep:")
        for width, weight, bonus in _BEAM_SWEEP:
            t1 = time.time()
            got, got_cost = board.beam(board.state, width, weight, bonus)
            print(f"        width {width:5d} W={weight} A={bonus}: "
                  f"{'no win' if got is None else str(got_cost) + ' presses'}"
                  f"  ({time.time() - t1:.0f}s)", flush=True)


def _plan_report():
    """Every level's plan, replayed through the real interpreter -- the
    end-to-end test of the model, the search and the tie labelling at once."""
    solver = TourSolver()
    game, expert, solvable = solver._ensure(0)
    total = ties = proved = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = expert.board(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level:2d}: UNSOLVED")
            continue
        blocks = sum(1 for x in board.state[1] if x)
        for direction in plan:
            eng.step(direction)
        left = sum(1 for x in _read_engine(eng, expert.ids, board.w)[1] if x)
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        step_ties = sum(len(s) - 1 for s in sets)
        info = expert.stats.get(level, {})
        tag = ("shortest" if info.get("proved") else "beam    ")
        sets_tag = "exact" if info.get("exact_sets") else "walk "
        proved += 1 if info.get("proved") else 0
        total += len(plan)
        ties += step_ties
        print(f"  L{level:2d}: {board.h}x{board.w}, {blocks:2d} blocks -> "
              f"{left} left, {len(plan):3d} presses in {info.get('pushes', 0):3d} "
              f"pushes [{tag} / {sets_tag} sets]  win={eng.check_win()}  "
              f"{step_ties:3d} tie-presses")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second right answer, "
          f"{proved}/{len(solvable)} levels provably shortest")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        violations = selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--confine" in sys.argv:
        violations = confine()
        print(f"confine: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--ties" in sys.argv:
        args = [int(a) for a in sys.argv[sys.argv.index("--ties") + 1:]
                if a.isdigit()]
        violations = ties_report(args or None)
        print(f"ties: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--sweep" in sys.argv:
        sweep_report()
        sys.exit(0)
    if "--plans" in sys.argv:
        _plan_report()
        sys.exit(0)
    sys.exit(TourSolver.main())
