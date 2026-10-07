"""ps:sokolor_ex -- ncrecc's "Sokolor EX": a sokoban whose win condition is
never a PLACE. It is a WIRING, and it is checked by a coin flip.

THE ONE RULE THAT CHANGES EVERYTHING

    [Active] -> []

is the FIRST rule in the file, so every turn begins by blowing away every lit
block on the board. Nothing about "lit" is stored between turns. Lighting is
re-derived from scratch, every press, in three steps:

    random [> Block] [Blue] -> [> Block] [Blue Active]     (and Red/Yellow/Green)
    startloop
      late [Active Block|Connector]  -> [Active Block|Active Connector]
      late [Active Connector|Block]  -> [Active Connector|Active Block]
      late [Active Blue|Blue] -> [Active Blue|Active Blue]   (and the others,
      late [Active Difficult|Difficult] -> [... Active Difficult]  + Difficult)
    endloop

so: **if and only if some block moved this turn**, ONE UNIFORMLY RANDOM block of
each colour lights up, and the light floods outward through same-colour
adjacency, through Difficult-Difficult adjacency, and through a Connector to a
block of ANY type. The win is `All Target on Block` and `All Color on Active`.

FOUR CONSEQUENCES, and the whole solver is built on them

* **The win is a property of the CONNECTIVITY GRAPH, not of any square.** Define
  one undirected graph on the occupied cells: an edge between 4-neighbours iff
  they are the same type, or either of them is a Connector. Lighting is exactly
  "the union of the components that contain a seed", because every spread rule
  is symmetric and monotone, so its fixpoint is the component closure and the
  order the interpreter runs the loop in cannot matter. `--selfcheck` measures
  that: 4310 random Connector/Difficult fields carrying exactly ONE colour block
  (so the seed is KNOWN, not inferred) reproduce the interpreter's lit set
  exactly, while a model that drops Difficult-Difficult misses 1326 of them and
  one that drops Connector-Connector misses 1734.

* **Therefore the goal the solver plans for is stricter than the goal the
  interpreter checks.** Only one block per colour is seeded, so a colour whose
  blocks lie in two components wins only when some OTHER colour's die happens to
  land in the second one. A plan that needs that is a plan that loses. The
  search's `won` is the LUCK-FREE condition -- *for every colour present, all of
  its blocks lie in ONE component* -- which makes every seeding outcome a win.
  `--rng` is the proof: every level's plan replayed under 200 different global
  seeds, 1000 replays, every one a WIN.

* **A press that moves no block cannot win.** The seeding rules are gated on
  `[> Block]`, so a turn spent walking lights nothing and `All Color on Active`
  is false however the pieces stand. The board can sit in a winning SHAPE for
  twenty presses without the game noticing; it is the push that collects. So the
  model carries the positional predicate and the search is required to end on a
  push -- which a macro search does for free, and which is also why a re-plan
  from an already-shaped board returns a one-push plan rather than the empty one.

* **Nothing else happens at all.** One block moves per press and only if the
  player walks into it; `no Pin` blocks are never pushed; a block refuses to
  enter a Recession (the player walks on them freely); there is no chain push,
  no pull, no gravity; `noaction` makes ACTION dead and `require_player_movement`
  makes a blocked press an exact no-op. The expert searches four directions.

THE SEARCH: macro A*, and the plans are PROVABLY SHORTEST

Because nothing in this game happens except when a block moves, a shortest press
sequence is a shortest sequence of pushes with shortest walks between them. A
macro is `walk to the square behind a block, then push it` at cost `walk + 1`,
and macro A* over those is exact. Four of the five levels fall out in under two
seconds of uniform-cost search.

Level 3 does not: its reachable space is 12.5M primitive states and 3.7M macro
states, and plain Dijkstra needs 78 s and 2.3 GB to reach the 40-press answer.
The heuristic that fixes it is admissible for a reason worth writing down. When
a level has NO Connector on it every component is monochromatic, so "all of a
colour in one component" means literally "those blocks are 4-adjacent"; for a
colour with exactly TWO blocks that is a two-body problem, and

    pushbound(u, v) = the exact minimum number of PUSHES to bring two blocks
                      alone on the board (walls and recessions only, no other
                      blocks, player free to teleport) into adjacency

is a lower bound on the pushes that colour still costs. Those push sets are
DISJOINT across colours -- a press moves one block, and that block has one
colour -- so the SUM over colours is admissible, not just the max.

And the same table answers a second question the search needs more: a pair it
never reached cannot be brought together on an EMPTY board, so it cannot be
brought together on this one, so that state is provably LOST and the successor is
dropped rather than queued. Sokoban flailing spends most of its branching on
blocks it has just ruined, and this is what stops paying for it -- level 3 is
118 s of search with the bound alone and 5 s with the dead-pair test, on 104002
macro states instead of 987k. Both cost one BFS over cell pairs, per level.
All five plans are the ones the exhaustive primitive BFS finds (`--bfs`).

OPTIMAL SETS ARE MEASURED, exactly. For every step of every plan and every one
of the four presses, the press is offered iff a bounded macro re-solve from the
state it lands on still wins in the presses that remain -- a proof in both
directions, nothing optimal omitted and nothing second-best offered. The bound
shrinks by one per step, so the cost collapses as the plan runs: level 3's first
step costs 11 s, its tenth under a second, and everything past its twentieth is
free. `--ties` re-derives the labels with a primitive-press BFS that knows
nothing about macros.

RECOVERY is the family's RESET prefix (`recovery_mode = "reset"`). The expert
itself re-plans from anywhere -- the search reads the live board and knows
nothing about a start state -- but the GAME does not always allow it: `--recover`
flails the levels with random walks and 75 of 100 landing boards can still be
planned to a win, the other 25 having wedged a block into a corner or split a
colour behind a wall it can no longer cross. One RESET restores exactly the state
the cached plan was solved from. The plans (and their optimal sets) live in
`data/sokolor_ex_plans.json`: the cold build is 1 m 33 s and 203 MB, almost all
of it level 3, and every seed and every shard would otherwise pay it again.

STOCHASTIC GAME, DETERMINISTIC CORPUS. The interpreter draws its four seeds from
the module-global `random`, so the same presses render different pink twice
running. `SokolorSolver.solve_episode` pins that global once per episode, which
makes a recording a pure function of its seed without making the WIN depend on
one; every frame-level check here (`--symmetry`, `--plans`) pins it per drive,
and `--rng` is the one that deliberately re-rolls.

ART. Nothing needed fixing, which is worth recording because this game had two
ways to hide its own mechanic and takes neither. Every block sprite has
transparent holes and `Inactive` (black) sits under `Active` (pink) beneath the
block layer, so "lit" shows through all six block types at every cell size the
levels use. And Lightgreen and Green collapse to one ARC index, so the Green
block is a flat green square -- distinguishable anyway, by its holes.
`--audit` is the regression test. The one composition pair that DOES collide is
`X` vs `X on Target`: Target is a hollow ring that anything on top of it covers
completely. No level in this game places a single Target (`All Target on Block`
is vacuous on all five), so it is excluded rather than repainted -- see
`_AUDIT_CASES`.

VERIFIED. 5 levels, 184 presses (25, 61, 45, 40, 13), every one provably
shortest -- and the proof is independent of the search that found them. `--bfs`
enumerates each level's reachable space with a primitive-press BFS that knows
nothing about macros (134647, 98528, 30440, 7915852 and 12290 states) and its
shortest wins are exactly those five numbers. `--rng`: 1000 replays, 5 levels x
200 global seeds, 0 lost to the dice. `--selfcheck`: 47842 presses fuzzed over
the five levels (2511 pushes, 2 accidental wins, none of them lucky) plus 4310
known-seed spread fields, 0 divergences from the interpreter. `--ties`
re-derives 456 labelled presses on levels 0, 1, 2 and 4 with a primitive BFS: 0
disagreements, in both directions (level 3's bound of 39 primitive presses
reaches most of its 12.5M states, so it has to be asked for by number).
`--dead` enumerates levels 1 and 2 whole (116820 and 109108 states) and proves
every one of the 73860 and 57908 states the heuristic prunes genuinely
unwinnable, by backward reachability from every state that has a winning press.
`--audit`: 17 reachable cell compositions pairwise distinct at all three cell
sizes the five boards use. `--symmetry`: 16 presentations x 5 levels, every plan
a WIN and every frame of both the plan and a 200-press random walk exactly the
transform of the unaugmented one. Recorded in-process: 5/5 levels per seed, and
all 780 recorded presses of a four-seed corpus replay to a WIN through the
adapter under a fresh RNG stream -- which is the point of the luck-free win condition, since the pink they
draw is not the pink that was taped. Three processes at different
PYTHONHASHSEEDs write byte-identical episodes. Longest level record is 62
actions, against the adapter's 200-press cap.

CLI
---
    --selfcheck   the native model fuzzed against the interpreter: rollouts on
                  the real levels, plus random block fields whose lit set is
                  predicted from a KNOWN seed
    --rng         every plan replayed under 200 global seeds -- the proof that
                  the luck-free win condition is the right one
    --audit       every reachable cell composition renders distinctly
    --symmetry    every presentation is an exact transform of the unaugmented
    --plans       every level's plan, replayed through the interpreter
    --ties        every optimal-action label re-derived by a primitive BFS
    --dead        every state the heuristic prunes as lost, proved lost over a
                  whole enumerated state space
    --recover     how much of a flailed board the expert can still plan out of
                  -- the number behind recovery_mode = "reset"
    --bfs         exhaustive primitive BFS over a whole state space -- the
                  independent optimality proof (level 3 is 12.5M states)
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
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

GAME_NAME = "Sokolor_EX"
GAME_ID = "ps:sokolor_ex"

#: Engine directions, in the order the model indexes them. ``k ^ 1`` is the
#: opposite direction, which is what makes a push's stance one xor away.
DIRNAMES = ("up", "down", "left", "right")
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: The six pushable object classes, in the order the model numbers them. The
#: first four are the game's ``Color`` group -- the ones that get seeded and the
#: only ones the win condition counts. Connector and Difficult are scenery that
#: conducts.
TYPES = ("blue", "red", "yellow", "green", "connector", "difficult")
COLORS = frozenset(range(4))
CONNECTOR = TYPES.index("connector")

_INF = 1 << 30
#: What `SokolorBoard.heuristic` returns for a state no relaxation can finish --
#: a pair of same-coloured blocks that cannot be brought together even with the
#: board emptied of everything else. Real pushes are a SUBSET of the relaxed
#: ones, so that is a proof the state is lost, not a guess, and `search` drops
#: the successor instead of queueing it.
_DEAD = _INF


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class SokolorBoard:
    """One level, as a model the search can step in microseconds.

    A state is ``(player, blocks)`` -- a flat cell index and a sorted tuple of
    ``(cell, type)``. Walls, recessions, targets and pins are static (a pinned
    block is never pushed, so it never leaves the cell its Pin is drawn on), and
    nothing else in the game persists between turns: ``Active`` is deleted by the
    first rule of every turn and re-derived from the positions. So the pair is
    exact, and `selfcheck` is the proof.
    """

    def __init__(self, eng, ids):
        grid = eng.grid
        self.h, self.w = eng.height, eng.width
        n = self.h * self.w
        self.wall = [False] * n
        self.recess = [False] * n
        targets, pinned, blocks, player = [], [], [], None
        for r in range(self.h):
            for c in range(self.w):
                cell, i = grid[r][c], r * self.w + c
                if ids["wall"] in cell:
                    self.wall[i] = True
                if ids["recession"] in cell:
                    self.recess[i] = True
                if ids["target"] in cell:
                    targets.append(i)
                if ids["pin"] in cell:
                    pinned.append(i)
                if ids["player"] in cell:
                    player = i
                for ti, t in enumerate(TYPES):
                    if ids[t] in cell:
                        blocks.append((i, ti))
        self.targets = frozenset(targets)
        self.pinned = frozenset(pinned)
        self.state = (player, tuple(sorted(blocks)))

        #: Which colours the win condition actually quantifies over. Blocks are
        #: never created or destroyed, so this is a property of the LEVEL.
        self.colors_present = frozenset(t for _c, t in blocks if t in COLORS)
        self.has_connector = any(t == CONNECTOR for _c, t in blocks)

        # Neighbour table. ``None`` off the board -- not every level is ringed
        # by wall (level 0 leaves bare Background outside its walls), so "is it a
        # wall?" is not the same question as "is it on the board?".
        self.nb = [[None] * 4 for _ in range(n)]
        for r in range(self.h):
            for c in range(self.w):
                i = r * self.w + c
                for k, (dr, dc) in enumerate(_DELTA):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.h and 0 <= cc < self.w:
                        self.nb[i][k] = rr * self.w + cc

        self._pairs = self._pair_table()
        self._twos = self._two_block_colors()

    # -- dynamics -----------------------------------------------------------
    def step(self, st, k):
        """One press. Returns ``(state, pushed)``, the SAME state value when the
        press does nothing -- into a wall, off the board, into a pinned block, or
        into a block the board refuses to move.

        `require_player_movement` is what makes all of those exact no-ops rather
        than turns in which only the lighting changes: the interpreter rolls the
        whole turn back when the player did not end up somewhere new. ACTION is
        not modelled at all -- the prelude declares `noaction`.
        """
        p, blocks = st
        nxt = self.nb[p][k]
        if nxt is None or self.wall[nxt]:
            return st, False
        occ = dict(blocks)
        if nxt not in occ:
            return (nxt, blocks), False
        if nxt in self.pinned:                     # [> Player|Block NO PIN]
            return st, False
        beyond = self.nb[nxt][k]
        if (beyond is None or self.wall[beyond] or self.recess[beyond]
                or beyond in occ):                 # [> Block|Recession] cancels
            return st, False
        moved = [(c, t) for (c, t) in blocks if c != nxt]
        moved.append((beyond, occ[nxt]))
        return (nxt, tuple(sorted(moved))), True

    # -- the lighting graph --------------------------------------------------
    def components(self, blocks) -> dict:
        """Cell -> component id, over the graph the light floods along: an edge
        between 4-neighbours iff they are the same type or either is a Connector.

        Every spread rule in the file is symmetric (`[Active X|X]` with no
        direction expands to all four) and only ever ADDS Active, so its fixpoint
        is exactly this component closure and no rule ordering inside the
        ``startloop`` can change it."""
        occ = dict(blocks)
        root = {c: c for c in occ}

        def find(x):
            while root[x] != x:
                root[x] = root[root[x]]
                x = root[x]
            return x

        for c, t in blocks:
            for k in (1, 3):                       # down, right: each edge once
                y = self.nb[c][k]
                if y is None:
                    continue
                u = occ.get(y)
                if u is None:
                    continue
                if t == u or t == CONNECTOR or u == CONNECTOR:
                    a, b = find(c), find(y)
                    if a != b:
                        root[a] = b
        return {c: find(c) for c in occ}

    def lit(self, blocks, seeds) -> frozenset:
        """The set the interpreter lights when ``seeds`` are the blocks its four
        `random` rules picked: the union of their components. Used only by
        `selfcheck`; the search never needs it, because it plans for the outcome
        that holds whatever the seeds are."""
        comp = self.components(blocks)
        roots = {comp[c] for c in seeds}
        return frozenset(c for c in comp if comp[c] in roots)

    def won(self, st) -> bool:
        """The LUCK-FREE win: every Target covered, and every colour's blocks all
        in ONE component.

        This is strictly stronger than `All Target on Block and All Color on
        Active`, and deliberately. Exactly one block per colour is seeded, and it
        is drawn uniformly, so a colour split across two components is lit in full
        only when a DIFFERENT colour's draw happens to fall in the second one --
        a win the plan cannot be replayed for. When every colour is whole, every
        draw lights every colour block and the press wins with certainty; `--rng`
        measures that over 1000 replays.

        Note this is a predicate on POSITIONS only. The game also requires the
        press to have moved a block (see the module docstring), which is a
        property of the press, not of the board -- `search` carries it.
        """
        occ = dict(st[1])
        if not self.targets.issubset(occ):
            return False
        comp = self.components(st[1])
        for ti in self.colors_present:
            roots = {comp[c] for (c, t) in st[1] if t == ti}
            if len(roots) > 1:
                return False
        return True

    # -- heuristic ----------------------------------------------------------
    def _pair_table(self) -> dict:
        """``{(u, v): pushes}`` -- the exact minimum number of pushes to bring two
        blocks ALONE on this board into adjacency, for every reachable pair.

        Backward BFS from every adjacent pair over the reverse push (a block at
        ``a`` came from ``src`` with the pusher standing beyond it), on a board
        holding nothing but walls and recessions. Dropping the other blocks and
        the player's walk can only make the real problem harder, so this is a
        lower bound on the pushes that pair still costs -- and being exact on the
        relaxation is what makes it worth computing rather than guessing
        Manhattan (which on level 3 leaves the search 40% bigger)."""
        free = [i for i in range(self.h * self.w)
                if not self.wall[i] and not self.recess[i]]
        dist, q = {}, deque()
        for u in free:
            for k in range(4):
                v = self.nb[u][k]
                if v is None or self.wall[v] or self.recess[v]:
                    continue
                key = (u, v) if u < v else (v, u)
                if key not in dist:
                    dist[key] = 0
                    q.append(key)
        while q:
            u, v = q.popleft()
            d = dist[(u, v)]
            for a, other in ((u, v), (v, u)):
                for k in range(4):
                    src = self.nb[a][k]            # a arrived from src ...
                    if (src is None or self.wall[src] or self.recess[src]
                            or src == other):
                        continue
                    # ... which means it was pushed in direction k^1, so the
                    # pusher stood one further along k, on the far side of src.
                    stance = self.nb[src][k]
                    if stance is None or self.wall[stance]:
                        continue
                    key = (src, other) if src < other else (other, src)
                    if key not in dist:
                        dist[key] = d + 1
                        q.append(key)
        return dist

    def _two_block_colors(self) -> tuple:
        """The colours the heuristic can speak about: exactly two blocks, on a
        board with no Connector.

        Both conditions are load-bearing. With a Connector anywhere, two blocks of
        a colour can share a component while sitting far apart, so "must become
        adjacent" is simply false. With three or more blocks of a colour, one
        component means a connected polyomino, whose diameter is not 1, so the
        pairwise bound is not a bound at all. Level 0 (twelve blues) and level 4
        (connectors) therefore search with h == 0, and both are seconds anyway."""
        if self.has_connector:
            return ()
        counts = {}
        for _c, t in self.state[1]:
            counts[t] = counts.get(t, 0) + 1
        return tuple(sorted(t for t in self.colors_present if counts[t] == 2))

    def heuristic(self, st) -> int:
        """An admissible lower bound on the presses left, for a state that is not
        yet won.

        The sum over colours is legitimate, not just the max: a press moves one
        block and that block has one colour, so the presses each term counts are
        disjoint. The floor of 1 is the push the game still owes -- a board that
        is already in the winning SHAPE has not won until something moves.

        A pair the relaxed table never reached is `_DEAD`: those two blocks
        cannot be brought together on an otherwise EMPTY board, so they cannot be
        brought together on this one. Reporting that rather than the 0 a missing
        key would default to is most of what this heuristic is worth -- level 3
        is 118 s of search without it and 5 s with it, because a sokoban's
        flailing spends most of its branching on blocks it has just ruined."""
        occ = {}
        for c, t in st[1]:
            if t in self._twos:
                occ.setdefault(t, []).append(c)
        total = 0
        for t in self._twos:
            u, v = occ[t]
            d = self._pairs.get((u, v) if u < v else (v, u))
            if d is None:
                return _DEAD
            total += d
        return total if total else 1

    # -- macros -------------------------------------------------------------
    def walk_tree(self, st) -> dict:
        """BFS parent pointers over the squares the player may walk on: on the
        board, not a wall, not a block. Recessions and Targets are floor -- they
        are on their own collision layer and stop blocks, never the player."""
        p, blocks = st
        occ = {c for c, _t in blocks}
        parent = {p: None}
        q = deque([p])
        while q:
            x = q.popleft()
            for k in range(4):
                y = self.nb[x][k]
                if y is None or self.wall[y] or y in parent or y in occ:
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

    def macros(self, st) -> list:
        """Every ``walk to the square behind a block, then push it``, as
        ``(presses, resulting state)``.

        A macro path is a shortest press sequence because no press in this game
        does anything except move the player, and the seeding a push triggers
        depends only on where the pieces end up -- so there is never a reason to
        take a press that neither advances a walk nor moves a block. The player
        ends on the square the block left, which is why the state needs no
        separate record of where it is standing."""
        parent = self.walk_tree(st)
        occ = {c for c, _t in st[1]}
        out = []
        for cell, t in st[1]:
            if cell in self.pinned:
                continue
            for k in range(4):
                stance = self.nb[cell][k ^ 1]
                dest = self.nb[cell][k]
                if stance is None or stance not in parent:
                    continue
                if (dest is None or self.wall[dest] or self.recess[dest]
                        or dest in occ):
                    continue
                moved = [(c, u) for (c, u) in st[1] if c != cell]
                moved.append((dest, t))
                out.append((self.route(parent, stance) + [k],
                            (cell, tuple(sorted(moved)))))
        return out

    # -- search -------------------------------------------------------------
    def search(self, st, bound=None, node_cap=_INF):
        """Macro A*. Returns ``(presses, cost)`` for a shortest winning press
        sequence, or ``(None, None)``.

        The start is NEVER a goal, however its blocks stand: the module
        docstring's third consequence is that a board already in the winning
        shape has not won until something moves, so the frontier is seeded with
        the start's MACROS rather than with the start itself. That is also why
        ``st`` is left out of ``dist`` -- it leaves a macro path free to come back
        to the shape it started in and collect on it, which a search that had
        already claimed distance 0 for that state would silently prune.

        ``bound`` stops the search once every remaining node must cost more than
        it, which is what makes the optimal-set measurement affordable: an
        alternative press only ever has to be shown NOT to finish in the presses
        that are left. A win is returned when its node is POPPED, never when it is
        generated -- macros have different costs, so a 3-press macro found from a
        g=45 node can beat an 11-press one found from g=40.
        """
        if bound is not None and self.heuristic(st) > bound:
            return None, None
        dist, parent, pq, counter = {}, {}, [], 0

        def offer(prev, presses, ns, g):
            nonlocal counter
            nh = 0 if self.won(ns) else self.heuristic(ns)
            if nh >= _DEAD:                     # provably lost -- see heuristic
                return
            if bound is not None and g + nh > bound:
                return
            if g < dist.get(ns, _INF):
                dist[ns] = g
                parent[ns] = (prev, presses)
                counter += 1
                heapq.heappush(pq, (g + nh, g, counter, ns))

        for presses, ns in self.macros(st):
            offer(None, presses, ns, len(presses))
        while pq:
            f, g, _c, s = heapq.heappop(pq)
            if g > dist.get(s, _INF):
                continue
            if bound is not None and f > bound:
                return None, None
            if self.won(s):
                out = []
                while s is not None:
                    s, taken = parent[s]
                    out = taken + out
                return out, g
            if len(dist) > node_cap:
                return None, None
            for presses, ns in self.macros(s):
                offer(s, presses, ns, g + len(presses))
        return None, None

    def optimal_sets(self, plan, progress=None) -> list:
        """Per-step optimal press SETS, MEASURED and exact in both directions.

        A press is offered at step ``i`` iff the state it lands on can still win
        in ``len(plan) - i - 1`` presses, which a bounded re-solve either exhibits
        or refutes by exhausting the frontier. Nothing optimal is omitted and
        nothing offered is second best. The bound also proves that nothing longer
        needs checking: the plan is shortest, so no successor can do better than
        one press cheaper.

        A press that both moves a block AND leaves a won shape ends the game
        there and then. It can only appear at the last step -- an earlier one
        would be a shorter plan than the shortest -- which is why that branch
        asserts rather than deciding."""
        st = self.state
        out = []
        for i, taken in enumerate(plan):
            left = len(plan) - i - 1
            best = []
            for k in range(4):
                ns, pushed = self.step(st, k)
                if ns == st:                       # a press that does nothing
                    continue
                if pushed and self.won(ns):
                    assert left == 0, (i, k, left)
                    best.append(k)
                    continue
                if left and self.search(ns, bound=left)[0] is not None:
                    best.append(k)
            assert taken in best, (i, taken, best)
            out.append(best)
            if progress is not None:
                progress(i, best)
            st = self.step(st, taken)[0]
        return out


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class SokolorExpert(PSExpert):
    """`PSExpert`'s plan memo, restore discipline and disk cache around the
    native macro A*. `heuristic` (the ENGINE-state one the base class declares)
    is never called -- the search runs on `SokolorBoard`."""

    directions = list(DIRNAMES)
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "sokolor_ex_plans.json")

    OBJECTS = ("wall", "recession", "target", "player", "pin") + TYPES

    def setup(self):
        self.ids = {n: self.g.obj_name_to_idx[n] for n in self.OBJECTS}

    def heuristic(self, eng):
        raise AssertionError(
            "SokolorExpert plans on its native model; the engine-state "
            "heuristic is unused")

    def board(self, eng) -> SokolorBoard:
        return SokolorBoard(eng, self.ids)

    def _search(self, eng):
        board = self.board(eng)
        plan, _cost = board.search(board.state, node_cap=self.node_cap)
        if plan is None:
            return None
        return Plan([DIRNAMES[k] for k in plan],
                    [[DIRNAMES[k] for k in s]
                     for s in board.optimal_sets(plan)])


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class SokolorSolver(PSAStarSolver):
    game_id = GAME_ID
    game_name = GAME_NAME
    game_module_id = GAME_ID
    expert_cls = SokolorExpert
    #: The longest plan is level 1's 61 presses. This bounds the exploration
    #: prefix plus the replay; the adapter's own budget is 200 presses per level
    #: and its RESET (``set_level``) hands the replay a fresh one.
    max_steps = 150
    #: Headroom, not a requirement: level 3 is the biggest search from a level
    #: START at 104k macro states and the other four are under 43k, so the family
    #: default of 400k would do for planning. It is raised because running OUT of
    #: it is SILENT -- an unsolvable level is legitimately dropped from the corpus
    #: rather than failing the run, so a cap that bites looks exactly like a level
    #: the game does not offer -- and because a re-plan from a FLAILED board is
    #: the expensive case, not a level start: `--recover` finds level 4 boards
    #: needing over a million (the dead-pair prune that shrinks level 3 does
    #: nothing there, since a board with Connectors has no `_twos` colour).
    node_cap = 2_000_000

    def solve_episode(self, seed: int, explore: bool = True):
        """Pin the interpreter's dice, then record the episode as the family does.

        This is the one generator here whose GAME is stochastic: the four
        `random` seeding rules draw from the module-global `random`, so which
        blocks light up -- and therefore the pink in every recorded frame -- is
        drawn from whatever state that global happens to be in. Nothing else in
        the recording touches it (`BaseSolver` explores from its own
        ``self.rng``, and the adapter's presentation augmentation has its own
        stream keyed on the seed), so one seeding here is enough to make the
        whole episode a pure function of ``seed`` -- which is what lets a shard
        be re-run, or run at a different ``PYTHONHASHSEED``, and produce the same
        bytes.

        The WIN never depended on this: `SokolorBoard.won` plans for the outcome
        that holds whatever the dice say, and `--rng` measures that over 1000
        replays. What depended on it was only whether the same corpus comes back
        twice."""
        random.seed(f"{GAME_ID}:{seed}")
        return super().solve_episode(seed, explore)


# ---------------------------------------------------------------------------
# --selfcheck: the model against the interpreter
# ---------------------------------------------------------------------------

def _read_engine(eng, board, ids):
    """The interpreter's board, in the model's own terms."""
    player, blocks = None, []
    for r in range(board.h):
        for c in range(board.w):
            cell, i = eng.grid[r][c], r * board.w + c
            if ids["player"] in cell:
                player = i
            for ti, t in enumerate(TYPES):
                if ids[t] in cell:
                    blocks.append((i, ti))
    return (player, tuple(sorted(blocks)))


def _read_active(eng, board, active_id) -> frozenset:
    return frozenset(r * board.w + c
                     for r in range(board.h) for c in range(board.w)
                     if active_id in eng.grid[r][c])


def selfcheck(trials=40, steps=120, fields=6000, verbose=True):
    """Fuzz the model against the real interpreter, two ways.

    ROLLOUTS from every level start check the three things the search believes:
    the positions after a press (walls, board edges, pinned blocks, a block
    refusing a Recession, and ACTION -- which must do nothing at all, the prelude
    says `noaction`), that the interpreter's lit set is closed under the model's
    component graph, and that `check_win` is exactly "every Target covered and
    every Color block lit". It also asserts the direction that matters most: a
    state the model calls `won`, entered by a PUSH, must be a win for the
    interpreter whatever its four dice said.

    But rollouts cannot pin the SPREAD down, because a lit set is closed under a
    graph that is too FINE just as happily as under the right one. So the second
    mode loads random Connector/Difficult fields carrying exactly ONE colour
    block -- which makes the seed KNOWN rather than inferred, since the only
    colour on the board has to be the one the `random` rules picked -- and
    compares the interpreter's lit set against the closure from that block. That
    is fully discriminating in both directions, and it is where the two
    non-obvious edges are measured: a model without Difficult-Difficult misses
    1326 of these fields and one without Connector-Connector misses 1734.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    ids = {n: game._game.obj_name_to_idx[n] for n in SokolorExpert.OBJECTS}
    active_id = game._game.obj_name_to_idx["active"]
    bad = presses = noops = pushes = wins = lucky = 0

    for level in range(game.n_levels):
        for guided in (False, True):
            for t in range(trials):
                game.set_level(level)
                board = SokolorBoard(eng, ids)
                st = board.state
                rng = random.Random(f"{GAME_NAME}:selfcheck:{level}:{guided}:{t}")
                for _ in range(steps):
                    nexts = [board.step(st, k) for k in range(4)]
                    pushy = [k for k in range(4) if nexts[k][1]]
                    live = [k for k in range(4) if nexts[k][0] != st]
                    roll = rng.random()
                    if guided and pushy and roll < 0.6:
                        k = rng.choice(pushy)
                    elif guided and live and roll < 0.95:
                        k = rng.choice(live)
                    else:
                        k = rng.randrange(5)          # 5 = ACTION, a no-op
                    before = st
                    eng.step("action" if k == 4 else DIRNAMES[k])
                    st, pushed = (st, False) if k == 4 else board.step(st, k)
                    presses += 1
                    noops += st == before
                    pushes += pushed
                    if _read_engine(eng, board, ids) != st:
                        bad += 1
                        if bad < 4:
                            print(f"  L{level} t{t}: model {st} != engine "
                                  f"{_read_engine(eng, board, ids)}")
                        break
                    # the lit set must be a union of whole components ...
                    av = _read_active(eng, board, active_id)
                    if av != board.lit(st[1], av):
                        bad += 1
                        if bad < 8:
                            print(f"  L{level} t{t}: lit set {sorted(av)} is not "
                                  f"component-closed")
                        break
                    # ... and the win must be exactly what the file says it is
                    occ = dict(st[1])
                    engine_win = eng.check_win()
                    expect = (board.targets.issubset(occ)
                              and all(c in av for c, u in st[1] if u in COLORS))
                    if engine_win != expect:
                        bad += 1
                        if bad < 8:
                            print(f"  L{level} t{t}: check_win {engine_win} != "
                                  f"{expect}")
                        break
                    if board.won(st) and pushed and not engine_win:
                        bad += 1
                        print(f"  L{level} t{t}: a LUCK-FREE win the engine "
                              f"refused -- the win model is wrong")
                        break
                    lucky += engine_win and not (board.won(st) and pushed)
                    if engine_win:
                        wins += 1
                        break

    # -- the spread on its own, over random fields with a KNOWN seed ---------
    h, w = 9, 9
    bg = game._game.obj_name_to_idx["background"]
    inactive = game._game.obj_name_to_idx["inactive"]
    green = TYPES.index("green")
    rng = random.Random(f"{GAME_NAME}:selfcheck:fields")
    board = None
    drawn = 0
    for _ in range(fields):
        grid = [[{bg} for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if r in (0, h - 1) or c in (0, w - 1):
                    grid[r][c].add(ids["wall"])
        cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        rng.shuffle(cells)
        # a player, a block it can definitely shove, and a landing square: the
        # seeding rules are gated on [> Block], so a field where nothing moves
        # lights nothing and measures nothing.
        (pr, pc), k = cells[0], rng.randrange(4)
        dr, dc = _DELTA[k]
        br, bc, fr, fc = pr + dr, pc + dc, pr + 2 * dr, pc + 2 * dc
        if not (1 <= br < h - 1 and 1 <= bc < w - 1
                and 1 <= fr < h - 1 and 1 <= fc < w - 1):
            continue
        grid[pr][pc].add(ids["player"])
        used = {(pr, pc), (fr, fc), (br, bc)}
        grid[br][bc] |= {ids[TYPES[rng.choice((CONNECTOR, 5))]], inactive}
        rest = [x for x in cells[1:] if x not in used]
        grid[rest[0][0]][rest[0][1]] |= {ids["green"], inactive}   # the ONE colour
        for (r, c) in rest[1:]:
            if rng.random() < 0.5:
                grid[r][c] |= {ids[TYPES[rng.choice((CONNECTOR, 5))]], inactive}
        eng.load_level(grid)
        if board is None:
            board = SokolorBoard(eng, ids)
        eng.step(DIRNAMES[k])
        blocks = _read_engine(eng, board, ids)[1]
        av = _read_active(eng, board, active_id)
        if not av:
            continue
        drawn += 1
        seed = [c for c, t in blocks if t == green]
        if board.lit(blocks, seed) != av:
            bad += 1
            if bad < 8:
                print(f"  field fuzz: model {sorted(board.lit(blocks, seed))} "
                      f"!= engine {sorted(av)}")
    if verbose:
        print(f"  {presses} presses fuzzed over {game.n_levels} levels "
              f"({noops} no-ops, {pushes} pushes, {wins} accidental wins, "
              f"{lucky} of them LUCKY)")
        print(f"  {drawn} random fields lit from a known seed")
    return bad


# ---------------------------------------------------------------------------
# --rng: the luck-free win condition, measured
# ---------------------------------------------------------------------------

def rng_report(seeds=200, verbose=True):
    """Replay every plan through the interpreter under ``seeds`` different global
    RNG states and require every one of them to WIN.

    This is the evidence for `SokolorBoard.won` being stricter than the file's
    own win condition. The four `random` rules draw from the module-global
    `random`, so re-seeding it is re-rolling the dice that decide which block of
    each colour lights up; a plan that leaned on a lucky draw would fail here at
    a rate this many replays cannot miss."""
    solver = SokolorSolver()
    game, expert, _ = solver._ensure(0)
    eng = game._engine
    bad = replays = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(eng, level)
        losses = 0
        for s in range(seeds):
            random.seed(f"{GAME_NAME}:rng:{level}:{s}")
            game.set_level(level)
            for direction in plan:
                eng.step(direction)
            replays += 1
            if not eng.check_win():
                losses += 1
        bad += losses
        if verbose:
            print(f"  L{level}: {len(plan)} presses x {seeds} seeds -- "
                  f"{seeds - losses}/{seeds} WIN")
    if verbose:
        print(f"  {replays} replays, {bad} of them lost to the dice")
    return bad


# ---------------------------------------------------------------------------
# --audit: every reachable cell composition, pixel-distinct at every cell size
# ---------------------------------------------------------------------------

#: Every stack of objects a square in these levels can hold. Player, Wall and
#: the six block classes share one collision layer, so no two of them ever meet;
#: Inactive rides under every block and Active rides over Inactive, which is
#: what makes "lit" a different picture from "not lit".
#:
#: TARGET IS DELIBERATELY ABSENT. Its sprite is a hollow ring that a Player or a
#: block covers completely, so `X` and `X on Target` are one picture -- but not
#: one level of this game places a Target (`All Target on Block` is vacuous on
#: all five), so there is no composition to repaint and no mechanic hidden. If a
#: level ever gains one, this is where it fails first.
_AUDIT_CASES = dict(
    [("floor", ()),
     ("recession", ("recession",)),
     ("wall", ("wall",)),
     ("player", ("player",)),
     ("player on recession", ("player", "recession"))]
    + [(t, (t, "inactive")) for t in TYPES]
    + [(t + " lit", (t, "inactive", "active")) for t in TYPES])


def audit(verbose=True):
    """Assert every reachable cell COMPOSITION renders differently at every cell
    size the levels use.

    Composition, not object: what a policy has to read here is not "there is a
    block" but "there is a LIT block", which lives in five transparent pixels of
    the sprite showing the Active square underneath. The board is FILLED with the
    composition and whole frames are compared rather than one cell being sliced
    out by ``cell_px`` arithmetic: `_render_frame` upscales a sub-64 render to
    fill the frame and then letterboxes it, so the cell grid in the output is not
    ``cell_px``-aligned and a crop lands in the wrong window."""
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
    argument is the ESCAPE! one -- gravity-free, screen-relative input, a win
    condition that names no direction (`All Target on Block`, `All Color on
    Active`), one moving body per press so nothing can contest a cell, and no
    sprite that encodes a facing (the six blocks have asymmetric holes, but a
    hole pattern is a picture, not a heading: it maps to its own mirror exactly
    as the motion does). The spread rules are written without a direction and
    reach a fixpoint, so there is no rule-order chirality to argue around either.

    The RANDOMNESS is the one thing that needs care, and it needs less than it
    looks. The adapter transforms the rendered PICTURE and remaps the input; the
    engine grid is never transformed, so the interpreter draws its four seeds
    from the same global stream in the same order whatever the presentation is.
    Pinning `random.seed` before each drive therefore makes the comparison exact
    rather than making it possible -- and a seeded random WALK is replayed
    alongside the plans because it is what reaches the boards a plan never
    visits (blocks wedged into corners, every colour split apart) and it presses
    ACTION too."""
    solver = SokolorSolver()
    game, expert, _ = solver._ensure(0)
    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        plans[level] = expert.plan(game._engine, level)

    def drive(g, level, presses, tag):
        g.set_level(level)
        random.seed(f"{GAME_NAME}:symmetry:{tag}")     # pin the four dice
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

    # Enumerate the presentations FIRST and drive one seed per (level, k). Every
    # drive here is 261 presses through the adapter, so scanning 400 seeds and
    # replaying each would be half a million presses to re-measure four
    # pictures; ``set_level`` is what draws the augmentation, so reading it is
    # free.
    reps = {}
    for seed in range(400):
        g = solver.make_game(seed)
        for level in range(g.n_levels):
            g.set_level(level)
            reps.setdefault(
                (level, (g._rotation_k, g._hflip, g._vflip)), seed)

    ref, seen, bad = {}, {k for _lvl, k in reps}, 0
    for (level, k), seed in sorted(reps.items(),
                                   key=lambda x: (x[0][0], x[0][1] != (0, False, False))):
        g = solver.make_game(seed)
        rng = random.Random(f"{GAME_NAME}:symmetry:walk:{level}")
        walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                for _ in range(walk_presses)]
        for tag, presses in (("plan", [screen_action(d, *k)
                                       for d in plans[level]]),
                             ("walk", [inverse_remap_action_full(a, *k)
                                       for a in walk])):
            frames = drive(g, level, presses, f"{level}:{tag}")
            if tag == "plan" and g._state != GameState.WIN:
                print(f"  seed {seed} L{level} {k}: plan did not win")
                bad += 1
            key = (level, tag)
            if key not in ref:
                if k != (0, False, False):
                    print(f"  L{level}: no unaugmented presentation to compare "
                          f"against -- nothing measured")
                    bad += 1
                    continue
                ref[key] = frames
                continue
            if any(not np.array_equal(transform(a, *k), b)
                   for a, b in zip(ref[key], frames)):
                print(f"  seed {seed} L{level} {k}: {tag} frames are not "
                      f"the transform of the unaugmented ones")
                bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
        print(f"  {len(reps)} (level, presentation) pairs driven, "
              f"{walk_presses} random presses each")
    return bad


# ---------------------------------------------------------------------------
# --bfs: the independent optimality proof
# ---------------------------------------------------------------------------

def bfs_report(levels=(0, 1, 2, 3, 4), verbose=True):
    """Exhaustive primitive-press BFS over a whole reachable state space, and the
    shortest win it finds compared against the macro plan.

    This is the check that the macro reformulation is not quietly losing a
    shorter answer -- it presses single keys and knows nothing about a macro, a
    stance or a walk. The win test is where this game differs from an ordinary
    sokoban: a state is not a win, a PRESS is, so what is measured is
    ``min over reachable s of dist(s) + 1`` over the states that have a push
    landing on a won shape. That is exact without doubling the space on an
    "arrived by pushing" bit, because ``dist(s)`` is already the cheapest way to
    stand there and the winning press is one more.

    All five levels are affordable, so all five are the default. Four are under
    140k states and seconds; level 3's whole reachable space is 12.5M, of which
    this visits 7.9M in ~2 minutes and ~3 GB -- the difference is the cut that
    stops expanding past the depth the first win was found at, which BFS order
    makes safe.
    """
    solver = SokolorSolver()
    game, expert, _ = solver._ensure(0)
    eng = game._engine
    bad = 0
    for level in levels:
        game.set_level(level)
        board = expert.board(eng)
        plan = expert.plan(eng, level)
        dist = {board.state: 0}
        q = deque([board.state])
        best = None
        while q:
            s = q.popleft()
            d = dist[s]
            if best is not None and d + 1 > best:
                continue
            for k in range(4):
                ns, pushed = board.step(s, k)
                if ns == s:
                    continue
                if pushed and board.won(ns) and (best is None or d + 1 < best):
                    best = d + 1
                if ns not in dist:
                    dist[ns] = d + 1
                    q.append(ns)
        ok = best == len(plan)
        bad += 0 if ok else 1
        if verbose:
            print(f"  L{level}: {len(dist)} reachable states, shortest win "
                  f"{best} presses, plan {len(plan)} -- "
                  f"{'PROVED SHORTEST' if ok else 'MISMATCH'}")
    return bad


# ---------------------------------------------------------------------------
# --dead: the pruning that makes the search cheap, proved on a whole space
# ---------------------------------------------------------------------------

def dead_report(levels=(1, 2), verbose=True):
    """Enumerate a whole reachable space and require every state the heuristic
    calls `_DEAD` to be genuinely unwinnable.

    The argument is short -- the relaxed push the pair table is built from drops
    the other blocks and the player's reachability, so every REAL push is a
    relaxed push, so a pair the relaxed table never reached is a pair no play can
    reach -- but it is the one place where a wrong sign in `_pair_table` would not
    show up as a slow search or a long plan. It would show up as a level the
    solver declares unsolvable, or (worse) a plan that is short because the
    shorter route was pruned. So it is measured: forward BFS over the whole
    space, then backward reachability from every state that has a winning press,
    and the two sets must not intersect.

    Levels 1 and 2 are the two that have both a `_twos` colour and a space small
    enough to hold twice -- 116820 and 109108 states, of which 73860 and 57908 are
    pruned, all of them correctly. (Those are bigger than the counts `--bfs`
    prints for the same levels, which stop expanding past the depth the answer
    was found at.) Level 3 has the first and 12.5M of the second; levels 0 and 4
    have no `_twos` colour at all, so their heuristic never says `_DEAD`."""
    solver = SokolorSolver()
    game, expert, _ = solver._ensure(0)
    eng = game._engine
    bad = 0
    for level in levels:
        game.set_level(level)
        board = expert.board(eng)
        if not board._twos:
            print(f"  L{level}: no two-block colour -- the heuristic never says "
                  f"_DEAD here, nothing to prove")
            continue
        rev, seeds = {}, []
        dist = {board.state: 0}
        q = deque([board.state])
        while q:
            s = q.popleft()
            for k in range(4):
                ns, pushed = board.step(s, k)
                if ns == s:
                    continue
                if pushed and board.won(ns):
                    seeds.append(s)
                rev.setdefault(ns, []).append(s)
                if ns not in dist:
                    dist[ns] = dist[s] + 1
                    q.append(ns)
        alive = set(seeds)
        stack = list(seeds)
        while stack:
            x = stack.pop()
            for y in rev.get(x, ()):
                if y not in alive:
                    alive.add(y)
                    stack.append(y)
        dead = [s for s in dist if board.heuristic(s) >= _DEAD]
        wrong = [s for s in dead if s in alive]
        bad += len(wrong)
        if verbose:
            print(f"  L{level}: {len(dist)} states, {len(alive)} can still win, "
                  f"{len(dead)} called _DEAD, {len(wrong)} of them WRONGLY")
    return bad


# ---------------------------------------------------------------------------
# --ties: every optimal-action label re-derived by an independent primitive BFS
# ---------------------------------------------------------------------------

def ties_report(levels=(0, 1, 2, 4), node_cap=4_000_000, verbose=True):
    """Re-derive every step's optimal SET with a bounded primitive-press BFS and
    require it to equal the label the plan carries.

    `SokolorBoard.optimal_sets` measures the same thing through the macro search,
    so this is the check that the macro reformulation has not changed what
    "optimal" means -- the BFS here just presses all four keys breadth-first and
    asks whether a win is still ``left`` presses away. Agreement in BOTH
    directions is the claim: a press the labels omit and the BFS finds would be a
    target the corpus teaches is wrong, and one the labels offer and the BFS
    refutes would be a second-best press taught as optimal.

    Unlike the macro version this pays the full breadth at every step, so the
    default is the four levels where that is seconds to a minute; level 3 has to
    be asked for by number and takes far longer, since a bound of 39 primitive
    presses reaches most of its 12.5M states at every one of its early steps.
    ``node_cap`` bounds one step's frontier. A step where any press ran into it
    is UNDECIDED, not disagreeing: the BFS neither found a win nor exhausted the
    space, so its empty answer is silence rather than a refutation. Undecided
    steps are counted and printed separately and they fail the check, because a
    label nothing re-derived is a label nothing checked. Level 3's early steps
    are all undecided at the default cap, which is what "ask for it by number"
    means here -- raise ``node_cap`` past its 12.5M states to decide them."""
    solver = SokolorSolver()
    game, expert, _ = solver._ensure(0)
    eng = game._engine
    bad = checked = capped = undecided = 0

    def wins_within(board, st, limit):
        """Can a win be collected within ``limit`` further presses from ``st``?"""
        seen, frontier = {st}, [st]
        for _ in range(limit):
            nxt = []
            for s in frontier:
                for k in range(4):
                    ns, pushed = board.step(s, k)
                    if ns == s:
                        continue
                    if pushed and board.won(ns):
                        return True, False
                    if ns in seen:
                        continue
                    seen.add(ns)
                    nxt.append(ns)
            frontier = nxt
            if not frontier:
                break
            if len(seen) > node_cap:
                return False, True
        return False, False

    for level in levels:
        game.set_level(level)
        board = expert.board(eng)
        plan = expert.plan(eng, level)
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        st = board.state
        for i, taken in enumerate(plan):
            left = len(plan) - i - 1
            truth, blind = [], False
            for k in range(4):
                ns, pushed = board.step(st, k)
                if ns == st:                       # a press that does nothing
                    continue
                checked += 1
                if pushed and board.won(ns):
                    if left == 0:
                        truth.append(DIRNAMES[k])
                    continue
                ok, hit = wins_within(board, ns, left)
                capped += hit
                blind |= hit
                if ok:
                    truth.append(DIRNAMES[k])
            if blind:
                undecided += 1
                print(f"  L{level} step {i}: UNDECIDED -- the frontier hit the "
                      f"{node_cap} cap before it could refute anything")
            elif sorted(truth) != sorted(sets[i]):
                bad += 1
                print(f"  L{level} step {i}: labels {sorted(sets[i])} != "
                      f"BFS truth {sorted(truth)}")
            st = board.step(st, DIRNAMES.index(taken))[0]
        if verbose:
            print(f"  L{level}: {len(plan)} steps re-derived")
    if verbose:
        print(f"  {checked} presses re-derived by primitive BFS, "
              f"{bad} disagreements, {undecided} steps undecided "
              f"({capped} frontier caps)")
    return bad + undecided


# ---------------------------------------------------------------------------
# --recover: how much of a flailed board the expert can still plan out of
# ---------------------------------------------------------------------------

def recover_report(trials=25, verbose=True):
    """Flail each level with a random walk and measure how often the expert can
    still plan a WIN from where it lands.

    This is the number behind ``recovery_mode = "reset"``. The expert genuinely
    re-plans from any state (`SokolorBoard.search` reads the live board and has
    no notion of a "plan from the start"), so the recovery mode is not a
    limitation of the solver -- it is a fact about the GAME: a block shoved into
    a corner, or a colour split by a wall it can no longer cross, is a board no
    amount of planning gets out of. About a quarter of these random boards are
    that, which is exactly the case a RESET prefix is for and the case an
    epsilon detour must avoid, and `record_level`'s detour guard (keep the
    alternative only if the level stays winnable) is what avoids it.

    Level 3 is left out: it is minutes per re-plan, and it is the same
    measurement."""
    solver = SokolorSolver()
    game, expert, _ = solver._ensure(0)
    eng = game._engine
    ok = dead = 0
    for level in range(game.n_levels):
        if level == 3:
            continue
        wins = 0
        for t in range(trials):
            game.set_level(level)
            board = expert.board(eng)
            st = board.state
            rng = random.Random(f"{GAME_NAME}:recover:{level}:{t}")
            for _ in range(rng.randint(5, 40)):
                st = board.step(st, rng.randrange(4))[0]
            plan, _cost = board.search(st, node_cap=solver.node_cap)
            if plan is None:
                dead += 1
                continue
            s = st
            for k in plan:
                s = board.step(s, k)[0]
            assert board.won(s), (level, t)
            ok += 1
            wins += 1
        if verbose:
            print(f"  L{level}: {wins}/{trials} flailed boards re-planned")
    if verbose:
        print(f"  {ok} perturbed states re-planned to a win, "
              f"{dead} wedged past recovery")
    return 0


# ---------------------------------------------------------------------------
# --plans
# ---------------------------------------------------------------------------

def _plan_report():
    """Every level's plan, replayed through the real interpreter -- the
    end-to-end test of the model, the search and the tie labelling at once."""
    solver = SokolorSolver()
    game, expert, solvable = solver._ensure(0)
    total = ties = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = expert.board(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level:2d}: UNSOLVED")
            continue
        random.seed(f"{GAME_NAME}:plans:{level}")
        for direction in plan:
            eng.step(direction)
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        step_ties = sum(len(s) - 1 for s in sets)
        total += len(plan)
        ties += step_ties
        kinds = {}
        for _c, t in board.state[1]:
            kinds[TYPES[t]] = kinds.get(TYPES[t], 0) + 1
        print(f"  L{level:2d}: {eng.height}x{eng.width}, "
              f"{' '.join(f'{v}{k[0]}' for k, v in sorted(kinds.items()))} -> "
              f"{len(plan):3d} presses  win={eng.check_win()}  "
              f"{step_ties:3d} tie-presses")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second right answer")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        violations = selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--rng" in sys.argv:
        violations = rng_report()
        print(f"rng: {violations} lost replays")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--bfs" in sys.argv:
        args = [int(a) for a in sys.argv[sys.argv.index("--bfs") + 1:]
                if a.isdigit()]
        violations = bfs_report(tuple(args) if args else (0, 1, 2, 3, 4))
        print(f"bfs: {violations} mismatches")
        sys.exit(1 if violations else 0)
    if "--dead" in sys.argv:
        args = [int(a) for a in sys.argv[sys.argv.index("--dead") + 1:]
                if a.isdigit()]
        violations = dead_report(tuple(args) if args else (1, 2))
        print(f"dead: {violations} states wrongly pruned")
        sys.exit(1 if violations else 0)
    if "--ties" in sys.argv:
        args = [int(a) for a in sys.argv[sys.argv.index("--ties") + 1:]
                if a.isdigit()]
        violations = ties_report(tuple(args) if args else (0, 1, 2, 4))
        print(f"ties: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--recover" in sys.argv:
        recover_report()
        sys.exit(0)
    if "--plans" in sys.argv:
        _plan_report()
        sys.exit(0)
    sys.exit(SokolorSolver.main())
