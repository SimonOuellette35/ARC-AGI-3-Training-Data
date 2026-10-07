"""Generate Phase-1 training data for the PuzzleScript game
ps:ims445_puzzlescript_game_buoy_deploy (Breton Ballas' "Buoy Deploy").

The harness -- the rotation contract, the trajectory recorder, the plan memo and
its disk cache, and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: a native model of the five live rules, the macro A* that
plans over it, and the level/rendering fixes the game needed.

The game
--------
A sokoban whose crates can be PULLED as well as pushed, over a board mined with
two kinds of hazard::

    [ >  Player | Buoy ] -> [ >  Player | > Buoy ]      push
    [ <  Player | Buoy ] -> [ <  Player | < Buoy ]      pull
    [ > Player | Bomb ]     -> restart
    [ > Player | Octopus ]  -> restart
    [ > Buoy | Bomb ]       -> [ | ]                    both destroyed
    [ Stationary Octopus ]  -> [ randomDir Octopus ]    (see below)

Win: ``All Target on Buoy`` -- every target cell must carry a buoy. Levels ship
with MORE buoys than targets, and the surplus is the currency: the only way to
clear a bomb is to shove a buoy into it, which destroys the buoy too.

Four consequences that shape every plan, none of them visible in the rule list:

  * **The pull is what makes corners survivable.** To move a buoy at ``X`` one
    step along ``d`` you may either stand at ``X - d`` and push, or stand at
    ``X + d`` and step to ``X + 2d`` and drag. A buoy in a corner has no push
    cell but usually still has a pull cell, so the classic sokoban deadlock is
    mostly absent here -- and the deadlocks that remain are the ones a BOMB
    creates by sealing a region.
  * **There is no free walk.** The pull rule fires on any step taken away from a
    buoy, so a player walking past a buoy DRAGS it. A "bare walk" step is one
    whose destination is free *and* whose opposite neighbour holds no buoy; the
    walk graph is therefore directed and depends on where the buoys are.
  * **A push and a pull can fire on the SAME press** (buoys on both sides of the
    player along the axis of travel), so one press can move two buoys.
  * **Bombs and octopi both block movement** -- they share the player's
    collision layer -- so stepping "into" one never moves anything. See the
    restart trap below for why that is not the harmless no-op it looks like.

The two engine facts this generator is built around
---------------------------------------------------
**1. The octopi never move.** ``randomDir`` is not a token
`PuzzleScriptAdapter`'s rule parser knows (`_parse_rule_side` recognises the
four directions, ``perpendicular``/``parallel``, ``horizontal``/``vertical``,
``moving``/``stationary``/``action``/``no``/``random`` and nothing else), so
``[Stationary Octopus] -> [randomDir Octopus]`` parses as
``[Stationary Octopus] -> [<unresolvable> Octopus]``, resolves to zero object
indices, and is classified *trivial* -- an identity substitution the fast apply
path skips outright. Verified on the grid: 30 presses on level 1 leave all four
octopi on their starting cells.

So in the adapted game an octopus is a second, PERMANENT kind of hazard: lethal
to touch like a bomb, but immune to buoys, where a bomb can be cleared by
spending one. That is a coherent mechanic and this generator plans against it
rather than against the .txt's stated intent. Implementing ``randomDir`` instead
was rejected deliberately: it would make an otherwise fully deterministic puzzle
stochastic (the wa30 category), which costs the optimal-action targets that are
the point of the corpus, and it would change behaviour in every other corpus
game that uses the keyword.

It does cost one shipped level -- see "The level fix" below.

**2. ``restart`` is executed by the ADAPTER, not by ``eng.step``.** A rule with
``restart`` sets `PSEngine._rule_restart` and nothing else; it is
`PuzzleScriptAdapter.perform_action` that reads the flag and reloads the level.
An expert that plans by stepping the engine (or, as here, a native model checked
against it) therefore sees a bomb-bump as a plain no-op, and a plan containing
one would silently RELOAD THE LEVEL on replay and desynchronise every frame
after it. `_Board.step` returns an explicit ``restart`` flag for exactly those
presses and the search refuses them; ``--fuzz`` asserts the flag agrees with
``_rule_restart`` on every press it takes.

The levels
----------
Seven ship with the game, all of them 17 wide, and all seven are solved. ``plan``
is the expert's press count and ``ties`` the share of its steps with more than
one equally-optimal press::

    level  size   buoys  bombs  octopi  targets  plan  ties  search
    0      11x17    5      4      0       1       136   1%   A*
    1      11x17    3      0      4       3        30   7%   A*
    2      12x17    5      3      2       2       158   3%   beam
    3      12x17    2      4      3       1        60   2%   A*
    4      12x17    2      6      3       1        67   4%   A*
    5      12x17    5      5      4       2       177   2%   beam
    6      12x17   10     36      6       6       126   6%   A* (w=20)

754 expert presses in all. They are long for their board size because a PULL
sokoban charges two or three presses per cell of buoy travel: after each push
the player has to walk back around to the far side. Four of the seven therefore
do not fit the adapter's 200-press default, so
``games/ps:ims445_puzzlescript_game_buoy_deploy`` raises those four to twice
their plan -- without that they would be unwinnable as shipped for any agent.

Ties are thin (1-7%) because these boards are corridors: a walk step is labelled
with every bare-walk direction that keeps it on a shortest route to the cell the
run ends on, and in a one-cell corridor there is only ever one.

Level 6 is the one worth reading. Its ten buoys are exactly six targets plus the
four bombs that seal the four corner rooms, so every buoy has exactly one job and
any waste loses the board. Level 2 has the same shape at a smaller scale (five
buoys = two targets + three sealing bombs).

Every level is engine-verified: ``--report`` replays the plan through the real
interpreter via ``perform_action`` -- the same path the recorder drives -- and
requires ``GameState.WIN`` on the last press and no earlier.

The level fix
-------------
**Level 2 was unwinnable and is now fixed in
``data/puzzlescript_games/IMS445_Puzzlescript_Game__Buoy_Deploy_.txt``.** Its
opening row is a corridor ``#P....M.........#`` whose only exit is the cell the
octopus stands on, so with static octopi the player's reachable region is the
five cells ``(1,1)..(1,5)`` and NOT ONE buoy, target or bomb on the board can
ever be touched. The level is built around the octopus wandering off, which this
adapter's octopi never do. Verified by flood fill before the fix: 5 of 87
free cells reachable, 0 of 2 targets, 0 of 5 buoys.

Usage (run from the repo root):
    python solvers/generate_ims445_buoy_deploy_training.py \\
        --episodes 200 --out data/training_multi_level/ims445_buoy_deploy

    python solvers/generate_ims445_buoy_deploy_training.py --report
    python solvers/generate_ims445_buoy_deploy_training.py --fuzz
    python solvers/generate_ims445_buoy_deploy_training.py --audit
    python solvers/generate_ims445_buoy_deploy_training.py --verify
"""

from __future__ import annotations

import heapq
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame                # noqa: E402
from solvers.base_solver import _ID_TO_GAMEACTION                     # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert, Plan      # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "IMS445_Puzzlescript_Game__Buoy_Deploy_"
GAME_MODULE_ID = "ps:ims445_puzzlescript_game_buoy_deploy"

#: Disk cache of each level's start plan AND its optimal-action sets. The
#: searches are seed-independent and cost the whole of generation's startup,
#: which every shard of `parallelize_generator` would otherwise repeat.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "ims445_buoy_deploy_plans.json"

#: Engine direction -> (dr, dc). The four moves are the whole action space:
#: ACTION5 is bound to nothing in this game.
_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}
#: Fixed iteration order, so a plan (and its tie sets) is reproducible.
_ORDER: tuple[str, ...] = ("up", "down", "left", "right")


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """Buoy Deploy over the level's cells, with the interpreter left out of it.

    Static: ``walls`` and ``octo`` (octopi never move -- see the module
    docstring), which together are ``blocked``, plus ``targets``, which no rule
    in this game creates or destroys.

    Dynamic, and so the state: ``(buoys, bombs, player)`` with the two piece
    sets as frozensets of cells. Bombs are in the state because a buoy shoved
    into one destroys them both, and which bombs are left decides which regions
    the player can walk between.
    """

    def __init__(self, walls, octo, targets, buoys, bombs, player, h, w):
        self.h, self.w = h, w
        self.walls = frozenset(walls)
        self.octo = frozenset(octo)
        self.targets = frozenset(targets)
        self.blocked = self.walls | self.octo
        self.start = (frozenset(buoys), frozenset(bombs), player)

    @classmethod
    def read(cls, eng, ids) -> "_Board":
        """Build the model from the interpreter's current grid."""
        wall, target, buoy, bomb, octo, player = ids
        walls, octos, targets, buoys, bombs, p = set(), set(), set(), set(), set(), None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                if cell & wall:
                    walls.add((r, c))
                if cell & octo:
                    octos.add((r, c))
                if cell & target:
                    targets.add((r, c))
                if cell & buoy:
                    buoys.add((r, c))
                if cell & bomb:
                    bombs.add((r, c))
                if cell & player:
                    p = (r, c)
        return cls(walls, octos, targets, buoys, bombs, p, eng.height, eng.width)

    def inb(self, cell) -> bool:
        return 0 <= cell[0] < self.h and 0 <= cell[1] < self.w

    # -- dynamics -------------------------------------------------------------
    def step(self, state, direction):
        """One press. Returns ``(new_state, flag)`` with ``flag`` one of:

          ``ok``       the press changed something;
          ``noop``     the interpreter refused it and the grid is untouched;
          ``restart``  the press bumped a bomb or an octopus. The GRID is
                       untouched -- both share the player's collision layer, so
                       nothing moves -- but `PuzzleScriptAdapter.perform_action`
                       reloads the level when it sees the flag, so this is the
                       one refused press that is not a no-op for a recording.
                       Never emitted into a plan; see the module docstring.

        The order below is the rule order, and it matters: the push rule gives
        the buoy ahead its force, the pull rule gives the buoy behind its own,
        and only then does ``[> Buoy | Bomb]`` get to consume a MOVING buoy. A
        press can therefore move two buoys at once, and the pulled one always
        lands on the cell the player is vacating, so it can never be refused on
        its own."""
        buoys, bombs, p = state
        dr, dc = _DELTA[direction]
        ahead = (p[0] + dr, p[1] + dc)
        behind = (p[0] - dr, p[1] - dc)
        if not self.inb(ahead) or ahead in self.walls:
            return state, "noop"
        if ahead in self.octo or ahead in bombs:
            return state, "restart"
        pulled = behind in buoys

        if ahead in buoys:
            beyond = (ahead[0] + dr, ahead[1] + dc)
            if self.inb(beyond) and beyond in bombs:
                # [> Buoy | Bomb] -> [ | ]: the pushed buoy and the bomb both go.
                nb = set(buoys)
                nb.discard(ahead)
                if pulled:
                    nb.discard(behind)
                    nb.add(p)
                return (frozenset(nb), bombs - {beyond}, ahead), "ok"
            if (not self.inb(beyond) or beyond in self.blocked
                    or beyond in buoys or beyond in bombs):
                # A blocked buoy blocks the player, which blocks the pull too.
                return state, "noop"
            nb = set(buoys)
            nb.discard(ahead)
            nb.add(beyond)
            if pulled:
                nb.discard(behind)
                nb.add(p)
            return (frozenset(nb), bombs, ahead), "ok"

        if pulled:
            nb = set(buoys)
            nb.discard(behind)
            nb.add(p)
            return (frozenset(nb), bombs, ahead), "ok"
        return (buoys, bombs, ahead), "ok"

    def won(self, state) -> bool:
        """``All Target on Buoy``. The quantifier is vacuously true with no
        targets at all, but no rule here can delete one -- ``[> Buoy | Bomb]``
        empties the two cells only on the collision layer it matched, and
        ``--fuzz`` asserts the target set never changes."""
        return bool(self.targets) and self.targets <= state[0]


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------

class _Planner:
    """Macro A* over `_Board`.

    A macro is ``walk to a cell, then one press that moves a buoy``. That
    decomposition is COMPLETE here even though the walk is constrained: every
    press either moves at least one buoy (a macro's last step) or moves only the
    player (a bare walk step), and a press that drags a buoy is itself a
    one-press macro rather than part of a walk. ``g`` counts primitive presses,
    which is the unit the agent pays.

    Dedup is on the exact ``(buoys, bombs, player)`` state. The usual sokoban
    trick of merging states by the player's reachable REGION is unsound here:
    the walk graph is DIRECTED (an edge exists only when the cell opposite the
    step holds no buoy), so two cells in one reachable set are not
    interchangeable.
    """

    #: Presses charged for each bomb a buoy's route has to cross. Crossing one
    #: is not a move at all -- it is a whole SECOND buoy being walked into the
    #: bomb and destroyed with it -- so the charge has to dominate the walking
    #: or the search happily routes every buoy through the minefield. Only the
    #: ranking matters, not the number.
    BOMB_TOLL = 12

    def __init__(self, board: _Board):
        self.b = board
        self.pass_ = {(r, c) for r in range(board.h) for c in range(board.w)
                      if (r, c) not in board.blocked}
        self._tables: dict = {}

    def tables(self, bombs) -> dict:
        """``{target: {cell: (bombs to clear, buoy-moves)}}`` for a bomb set.

        Reverse Dijkstra over the one-buoy transport relation, ignoring every
        other buoy. A buoy at ``X`` can move to ``X + d`` when ``X + d`` is
        passable and EITHER ``X - d`` is passable (the player pushes from
        there) or ``X + 2d`` is (the player drags from ``X + d``).

        A bomb cell is passable at a price. It has to be *passable*: a bomb can
        be cleared, so walling them off would call reachable targets dead and
        prune winning boards. And it has to have a *price*: clearing one costs
        a whole buoy, which on the exact-fit levels (2 and 6, where the buoy
        count is the target count plus the number of sealing bombs) is the only
        thing the plan is really choosing. Cost is the lexicographic pair, so
        the bomb count is minimised first and never traded against walking.

        Keyed by the bomb set and memoized: bombs change only when one is
        destroyed, so a whole search sees a handful of distinct sets."""
        cached = self._tables.get(bombs)
        if cached is not None:
            return cached
        P = self.pass_
        out = {}
        for target in self.b.targets:
            dist = {target: (0, 0)}
            heap = [(0, 0, target)]
            while heap:
                nb, nm, y = heapq.heappop(heap)
                if (nb, nm) != dist.get(y):
                    continue                       # a stale heap entry
                for dr, dc in _DELTA.values():
                    x = (y[0] - dr, y[1] - dc)     # the buoy came from here
                    if x not in P:
                        continue
                    if not ((x[0] - dr, x[1] - dc) in P            # push cell
                            or (y[0] + dr, y[1] + dc) in P):       # drag cell
                        continue
                    cost = (nb + (y in bombs), nm + 1)
                    if cost < dist.get(x, (1 << 30, 0)):
                        dist[x] = cost
                        heapq.heappush(heap, (cost[0], cost[1], x))
            out[target] = dist
        self._tables[bombs] = out
        return out

    def h(self, state):
        """Estimated presses to the win, or ``None`` when the state is DEAD.

        The estimate is a greedy matching of bare targets to distinct buoys over
        the tables (moves, plus `BOMB_TOLL` per bomb the route must cross) and
        the player's walk to the nearest buoy less one -- it pushes from beside
        it, not from on top of it.

        The DEAD test is the half that makes these boards tractable, because
        the one irreversible act in this game is spending a buoy:

          * a bare target no surviving buoy can reach at any price is a lost
            board -- the greedy matching runs out of buoys;
          * and so is one whose CHEAPEST route still crosses more bombs than
            the spare buoys can pay for. Every bomb on that route needs its own
            buoy driven into it, and those buoys are gone, so the budget is
            ``len(buoys) - len(bare targets)``. The bound is the WORST single
            target rather than the sum, because two routes can share a cleared
            bomb and a sum would call winnable boards dead."""
        buoys, bombs, p = state
        bare = [t for t in self.b.targets if t not in buoys]
        if not bare:
            return 0
        if len(buoys) < len(bare):
            return None
        tables = self.tables(bombs)
        loose = list(buoys)
        total = 0
        worst_tolls = 0
        taken: set[int] = set()
        for target in sorted(bare):
            table = tables[target]
            best = best_i = None
            for i, cell in enumerate(loose):
                if i in taken:
                    continue
                d = table.get(cell)
                if d is not None and (best is None or d < best):
                    best, best_i = d, i
            if best is None:
                return None                     # no buoy can ever reach it
            taken.add(best_i)
            worst_tolls = max(worst_tolls, best[0])
            total += best[1] + self.BOMB_TOLL * best[0]
        if worst_tolls > len(buoys) - len(bare):
            return None                         # not enough buoys to pay
        walk = min(abs(p[0] - r) + abs(p[1] - c) for r, c in loose)
        return total + max(0, walk - 1)

    # -- macros ---------------------------------------------------------------
    def walk_tree(self, state) -> dict:
        """BFS over BARE-WALK steps from the player: ``{cell: (prev, dir)}``.

        An edge ``P -> P + d`` exists when ``P + d`` is free of everything on
        the player's collision layer AND ``P - d`` holds no buoy -- a step away
        from a buoy drags it, which is a macro, not a walk."""
        buoys, bombs, p = state
        blocked = self.b.blocked
        parent: dict = {p: None}
        queue = deque([p])
        while queue:
            cur = queue.popleft()
            for direction, (dr, dc) in _DELTA.items():
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in parent or not self.b.inb(nxt):
                    continue
                if nxt in blocked or nxt in buoys or nxt in bombs:
                    continue
                if (cur[0] - dr, cur[1] - dc) in buoys:
                    continue                     # this step would drag
                parent[nxt] = (cur, direction)
                queue.append(nxt)
        return parent

    @staticmethod
    def walk_to(parent, cell) -> list:
        out = []
        while parent[cell] is not None:
            cell, direction = parent[cell]
            out.append(direction)
        out.reverse()
        return out

    def macros(self, state):
        """``[(presses, new_state)]`` -- every ``walk + buoy press`` available.

        Presses the model flags ``restart`` are dropped outright (see the module
        docstring), and so are the ones that leave the pieces where they were:
        a bare walk is not a macro, and a refused push wastes a press to reach
        the state it started from."""
        parent = self.walk_tree(state)
        buoys, bombs, _p = state
        out = []
        for cell in parent:
            for direction in _ORDER:
                new, flag = self.b.step((buoys, bombs, cell), direction)
                if flag != "ok" or (new[0] == buoys and new[1] == bombs):
                    continue
                out.append((self.walk_to(parent, cell) + [direction], new))
        return out

    def astar(self, state, weight: int = 1, node_cap: int = 400_000):
        """``(presses, nodes)`` -- a winning plan, or ``(None, nodes)``.

        ``weight`` above 1 inflates the heuristic: the plan is still a genuine
        win (the model is exact and `--verify` replays it through the real
        interpreter) but is no longer argued shortest. See `_WEIGHTS` for which
        level needs what and why."""
        if self.b.won(state):
            return [], 0
        h0 = self.h(state)
        if h0 is None:
            return None, 0
        counter = nodes = 0
        pq = [(weight * h0, 0, counter, state, [])]
        best_g = {state: 0}
        while pq:
            _f, g, _c, cur, path = heapq.heappop(pq)
            if g > best_g.get(cur, 1 << 30):
                continue                        # a stale heap entry
            for macro, new in self.macros(cur):
                nodes += 1
                ng = g + len(macro)
                if self.b.won(new):
                    return path + macro, nodes
                if best_g.get(new, 1 << 30) <= ng:
                    continue
                hh = self.h(new)
                if hh is None:
                    continue                    # dead: a target went unreachable
                best_g[new] = ng
                counter += 1
                heapq.heappush(pq, (ng + weight * hh, ng, counter, new,
                                    path + macro))
            if nodes >= node_cap:
                return None, nodes
        return None, nodes

    def beam(self, state, width: int = 400, depth: int = 60,
             node_cap: int = 600_000):
        """A width-capped breadth-first beam over the same macros.

        WHY, and when to prefer it. A* and the beam pay the same thing per node
        -- one walk BFS plus one macro simulation -- and A* spends that budget
        where its heuristic points. That is the right call only while the
        heuristic can tell good boards from bad, and on the long levels here it
        cannot: their plans are four or five INDEPENDENT deliveries in sequence
        (open this bomb, drag that buoy across the board, then the next), and
        over the dozens of presses spent repositioning the player between them
        the estimate is flat. A* then degenerates to uniform-cost search at a
        depth of two hundred presses, which is hopeless. A beam spends its
        budget on breadth at every depth instead and only uses the heuristic to
        pick who survives, which is all a flat heuristic is good for.

        The trade is optimality -- a beam plan wanders where A* would not -- so
        `_WEIGHTS` sends a level here only when A* cannot reach it at all.
        Memory is bounded by construction (``width`` survivors per layer, plus
        the dedup set), which A* with a big node cap is not."""
        if self.b.won(state):
            return [], 0
        if self.h(state) is None:
            return None, 0
        frontier = [(state, [])]
        nodes = 0
        for _layer in range(depth):
            kids = []
            seen = set()
            for cur, path in frontier:
                for macro, new in self.macros(cur):
                    nodes += 1
                    if self.b.won(new):
                        return path + macro, nodes
                    # Dedup WITHIN the layer only. A search-wide `seen` set
                    # starves this beam: these boards are a corridor of states
                    # so narrow that after ~70 layers every child of every
                    # survivor has already been generated, the frontier
                    # collapses to one and the search reports failure with the
                    # goal still twenty presses away (measured on level 2).
                    # Per-layer dedup keeps the breadth without that.
                    if new in seen:
                        continue
                    hh = self.h(new)
                    if hh is None:
                        continue
                    seen.add(new)
                    # Rank on the heuristic alone, with the running cost as the
                    # tie-break; the tuple must not fall through to comparing
                    # states, which are frozensets and have no order.
                    kids.append((hh, len(path) + len(macro), new, path + macro))
                if nodes >= node_cap:
                    return None, nodes
            if not kids:
                return None, nodes              # the reachable space closed
            kids.sort(key=lambda kid: (kid[0], kid[1]))
            frontier = [(new, path) for _h, _g, new, path in kids[:width]]
        return None, nodes

    # -- optimal-action sets --------------------------------------------------
    def optsets(self, plan) -> list[list[str]]:
        """Per-step optimal-press SETS for ``plan``, from the board's start.

        Most of a plan is the player WALKING to the next buoy, and a walk's
        order is free: any interleaving that stays on a shortest route to the
        same cell costs the same and leaves an identical board, because a bare
        walk moves nothing. Labelling one arbitrary interleaving as the only
        right answer trains the policy to a coin flip. So each walk step is
        labelled with every bare-walk direction that keeps it on a shortest
        route to the stand cell the run ends on; buoy presses are labelled with
        themselves alone -- which buoy to shove where is the puzzle, and a
        sibling press is a different plan, not a reordering of this one.

        The run's walkable map is constant (a walk moves nothing), so one BFS
        per run over the DIRECTED walk graph gives the distance field the
        alternatives are read off."""
        state = self.b.start
        steps = []                    # (kind, state_before, direction)
        for direction in plan:
            new, flag = self.b.step(state, direction)
            assert flag == "ok", f"plan press {direction!r} is a {flag}"
            kind = "walk" if (new[0] == state[0] and new[1] == state[1]) else "press"
            steps.append((kind, state, direction))
            state = new

        out: list = [None] * len(plan)
        i = 0
        while i < len(steps):
            if steps[i][0] != "walk":
                out[i] = [steps[i][2]]
                i += 1
                continue
            run = i
            while i < len(steps) and steps[i][0] == "walk":
                i += 1
            if i >= len(steps):
                # A trailing walk stands the player nowhere in particular; keep
                # the recorded choice as the only label rather than invent ties.
                for j in range(run, i):
                    out[j] = [steps[j][2]]
                continue
            stand = steps[i][1][2]
            dist = self._walk_distances(steps[run][1], stand)
            for j in range(run, i):
                here = steps[j][1][2]
                d0 = dist.get(here)
                alts = [] if d0 is None else [
                    d for d in _ORDER
                    if dist.get(self._walk_dest(steps[j][1], d)) == d0 - 1]
                # The recorded step is on a shortest route by construction, so a
                # disagreement means the reconstruction drifted -- fall back to
                # labelling what the expert actually did.
                out[j] = alts if steps[j][2] in alts else [steps[j][2]]
        return out

    def _walk_dest(self, state, direction):
        """The cell a BARE walk in ``direction`` reaches, or None if that press
        is not a bare walk from ``state``."""
        new, flag = self.b.step(state, direction)
        if flag != "ok" or new[0] != state[0] or new[1] != state[1]:
            return None
        return new[2]

    def _walk_distances(self, state, target) -> dict:
        """BFS distance to ``target`` over BARE-WALK steps, computed on the
        REVERSED walk graph (the forward one is directed)."""
        buoys, bombs, _p = state
        blocked = self.b.blocked
        dist = {target: 0}
        queue = deque([target])
        while queue:
            cur = queue.popleft()
            for dr, dc in _DELTA.values():
                prev = (cur[0] - dr, cur[1] - dc)     # prev --(dr,dc)--> cur
                if prev in dist or not self.b.inb(prev):
                    continue
                if prev in blocked or prev in buoys or prev in bombs:
                    continue
                if (prev[0] - dr, prev[1] - dc) in buoys:
                    continue                          # that step would drag
                dist[prev] = dist[cur] + 1
                queue.append(prev)
        return dist


# ---------------------------------------------------------------------------
# Per-level search settings
# ---------------------------------------------------------------------------

#: ``level -> ("astar", weight) | ("beam", width)``.
#:
#: Unweighted A* is used wherever it closes in seconds, and those plans are
#: argued shortest as far as this model goes -- ``g`` counts primitive presses,
#: nothing is merged by player region, and the heuristic's two terms (buoy moves
#: and the walk to the first buoy) are charged against disjoint presses. The one
#: place it is not a certificate is the `BOMB_TOLL`, which is a rank, not a cost;
#: level 0 is the only A* level with bombs on a buoy's route, so its plan is
#: shortest only among plans that clear the same bombs.
#:
#: Levels 2 and 5 go to the beam because the heuristic goes FLAT on them. Their
#: plans are four or five INDEPENDENT deliveries in sequence (open this bomb,
#: drag that buoy across the board, then the next), and the dozens of presses
#: spent walking the player between deliveries move the estimate not at all. A*
#: degenerates to uniform-cost search at a depth of a hundred-odd presses and
#: does not return; the beam finds both in about a second. Those plans are
#: winning and engine-verified, and NOT argued shortest.
#:
#: Level 6 is the opposite case and worth the note, because it looks like the
#: beam's kind of level and is not. Its heuristic is not flat, it is BLIND in
#: one particular way: the four bombs its plan must clear are there to let the
#: PLAYER into the corner rooms, not to let a buoy out, and each corner room
#: already contains the buoy for its own target. So the transport tables charge
#: no toll at all and h(start) is 14 against a 126-press plan. What saves it is
#: that the board is a nest of small independent puzzles rather than one long
#: chain, so a heavily weighted A* -- which is close to greedy best-first -- goes
#: at them one at a time and closes in seconds, where a beam of any width spends
#: its whole budget on the cross product of the six rooms' openings.
_STRATEGY: dict[int, tuple[str, int]] = {
    0: ("astar", 1),
    1: ("astar", 1),
    2: ("beam", 200),
    3: ("astar", 1),
    4: ("astar", 1),
    5: ("beam", 200),
    6: ("astar", 20),
}
#: A level added past the table gets A* first and the beam as a fallback, which
#: is also what a level whose configured search comes back empty gets.
_DEFAULT_STRATEGY = ("astar", 1)

#: A* node budget. This is a MEMORY bound as much as a time one -- every queued
#: node holds two frozensets and its path -- so it is set where the whole search
#: stays inside a few hundred MB.
NODE_CAP = 400_000
#: Beam layers. One layer is one buoy-moving press, and the longest plan here
#: spends about ninety of them.
BEAM_DEPTH = 140


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class BuoyDeployExpert(PSExpert):
    """Plans over `_Board` with `_Planner`; never steps the interpreter.

    `PSExpert` owns everything around that -- the in-memory memo, the on-disk
    plan cache with its staleness check, the level scoping and the
    snapshot/restore discipline -- so `_search` is the only override.
    `heuristic` is unreachable: no A* runs on the interpreter here.

    The base `_key` -- every non-background cell -- is deliberately NOT narrowed
    to the cheaper dynamic-objects-only key this family usually uses. That key
    is canonical within a level, but `PSExpert.plan` also uses it as the ON-DISK
    CACHE'S STALENESS CHECK, and the static objects it leaves out are exactly
    the ones this game's level fix moved: an octopus relocation (or a wall, or a
    target) would not change the signature, so a cache written before the edit
    would be served against the edited board -- silently, as a plan that no
    longer wins. The key is computed once per `plan` call and never inside a
    search (that runs on `_Board`, which does not touch it), so the wider key
    costs nothing.
    """

    #: Harmless with the whole-board key above, and kept so the memo stays
    #: level-scoped if that key is ever narrowed after all.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        g = self.g
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.target_ids = set(g.resolve_object_name("target"))
        self.buoy_ids = set(g.resolve_object_name("buoy"))
        self.bomb_ids = set(g.resolve_object_name("bomb"))
        self.octo_ids = set(g.resolve_object_name("octopus"))
        self.player_ids = set(self.game._engine._player_indices)

    def heuristic(self, eng) -> int:
        raise AssertionError("the model plans; PSExpert's A* is unused here")

    def board(self, eng) -> _Board:
        return _Board.read(eng, (self.wall_ids, self.target_ids, self.buoy_ids,
                                 self.bomb_ids, self.octo_ids, self.player_ids))

    def solve(self, eng, level: int | None = None):
        """``(presses, optsets)`` from the engine's current grid, or
        ``(None, None)`` if neither search reaches a win.

        The configured search runs first and the OTHER one is the fallback (see
        `_STRATEGY`). That matters for recovery, not for the level starts: an
        exploration prefix can leave the board in a state its level's configured
        search is the wrong tool for, and a generator that returned None there
        would drop the level rather than re-plan out of it."""
        board = self.board(eng)
        planner = _Planner(board)
        kind, size = _STRATEGY.get(level, _DEFAULT_STRATEGY)
        order = [(kind, size)] + ([("beam", 200)] if kind == "astar"
                                  else [("astar", 8)])
        for kind, size in order:
            if kind == "astar":
                presses, _n = planner.astar(board.start, weight=size,
                                            node_cap=NODE_CAP)
            else:
                presses, _n = planner.beam(board.start, width=size,
                                           depth=BEAM_DEPTH)
            if presses is not None:
                return presses, planner.optsets(presses)
        return None, None

    def _search(self, eng) -> list | None:
        # `plan` scopes the memo by level, but `_search` is not told which one;
        # the level only picks the search WEIGHT, and the start states the disk
        # cache serves are the only ones where that choice matters, so read it
        # off the adapter rather than threading it through PSExpert.
        level = self.game._current_level_index
        presses, sets = self.solve(eng, level)
        return None if presses is None else Plan(presses, sets)


class BuoyDeploySolver(PSAStarSolver):
    game_id = "puzzlescript_ims445_puzzlescript_game_buoy_deploy"
    game_name = GAME_NAME
    game_module_id = GAME_MODULE_ID
    expert_cls = BuoyDeployExpert

    #: Room for the longest plan plus the exploration prefix and the re-plan
    #: after it. The per-level engine budget is raised to match in
    #: ``games/ps:ims445_puzzlescript_game_buoy_deploy``.
    max_steps = 400

    def prepare_expert(self, game, expert) -> None:
        """Solve every level before `discover_solvable` asks, so the one-time
        cost shows up as startup and the disk cache fills in one pass."""
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
    solver = BuoyDeploySolver()
    game = solver.make_game(0)
    return solver, game, BuoyDeployExpert(game)


def _report() -> int:
    """Per-level shape, plan length, tie coverage -- and CERTIFY each plan by
    replaying it through the real interpreter."""
    from solvers.common.ps_astar import screen_action                  # noqa: E402
    from arcengine import ActionInput, GameState                       # noqa: E402
    _solver, game, expert = _levels()
    bad = 0
    total = 0
    print(f"{'lvl':>3} {'size':>6} {'buoy':>4} {'bomb':>4} {'oct':>3} "
          f"{'tgt':>3} {'plan':>5} {'ties':>5} {'sec':>6}  verdict")
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = expert.board(eng)
        buoys, bombs, _p = board.start
        t0 = time.time()
        presses, sets = expert.solve(eng, level)
        dt = time.time() - t0
        if presses is None:
            print(f"{level:>3} {board.h:>2}x{board.w:<3} {len(buoys):>4} "
                  f"{len(bombs):>4} {len(board.octo):>3} {len(board.targets):>3} "
                  f"{'-':>5} {'-':>5} {dt:>6.1f}  UNSOLVED")
            bad += 1
            continue
        ties = sum(1 for s in sets if len(s) > 1) / max(1, len(sets))
        # Certify: replay through the adapter, which is what the recorder drives.
        game.set_level(level)
        rot = (game._rotation_k, game._hflip, game._vflip)
        state = None
        for i, direction in enumerate(presses):
            act = screen_action(direction, *rot)
            game.perform_action(ActionInput(id=act))
            state = game._state
            if state == GameState.WIN and i != len(presses) - 1:
                state = "EARLY WIN"
                break
        ok = state == GameState.WIN
        bad += not ok
        total += len(presses)
        print(f"{level:>3} {board.h:>2}x{board.w:<3} {len(buoys):>4} "
              f"{len(bombs):>4} {len(board.octo):>3} {len(board.targets):>3} "
              f"{len(presses):>5} {ties:>5.0%} {dt:>6.1f}  "
              f"{'WIN' if ok else 'FAIL ' + str(state)}")
    print(f"total expert presses: {total}")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# Fuzz: the model against the interpreter
# ---------------------------------------------------------------------------

def _fuzz(trials: int = 12, steps: int = 60, seed: int = 7) -> int:
    """Random play on every level, asserting the model and the interpreter agree
    on the board, on ``check_win`` AND on the restart flag after every press.

    The restart half is the one that cannot be skipped: `eng.step` leaves the
    grid untouched for a bomb bump, so a model that called it a plain no-op
    would agree on every cell and still be wrong about the only thing that
    matters -- `perform_action` reloads the level when the flag is set."""
    _solver, game, expert = _levels()
    eng = game._engine
    rng = random.Random(seed)
    ids = (expert.wall_ids, expert.target_ids, expert.buoy_ids,
           expert.bomb_ids, expert.octo_ids, expert.player_ids)
    counts = dict(ok=0, noop=0, restart=0, push=0, pull=0, both=0,
                  destroy=0, win=0)
    bad = 0

    def read():
        buoys, bombs, targets, p = set(), set(), set(), None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                if cell & expert.buoy_ids:
                    buoys.add((r, c))
                if cell & expert.bomb_ids:
                    bombs.add((r, c))
                if cell & expert.target_ids:
                    targets.add((r, c))
                if cell & expert.player_ids:
                    p = (r, c)
        return (frozenset(buoys), frozenset(bombs), p), frozenset(targets)

    for level in range(game.n_levels):
        for trial in range(trials):
            game.set_level(level)
            board = _Board.read(eng, ids)
            state = board.start
            live, targets0 = read()
            if live != state:
                print(f"level {level}: model start != engine start")
                bad += 1
                continue
            for t in range(steps):
                direction = rng.choice(_ORDER)
                new, flag = board.step(state, direction)
                counts[flag] += 1
                if flag == "ok":
                    dr, dc = _DELTA[direction]
                    push = (state[2][0] + dr, state[2][1] + dc) in state[0]
                    pull = (state[2][0] - dr, state[2][1] - dc) in state[0]
                    counts["push"] += push and not pull
                    counts["pull"] += pull and not push
                    counts["both"] += push and pull
                    counts["destroy"] += len(new[1]) < len(state[1])
                eng.step(direction)
                live, targets = read()
                if eng._rule_restart != (flag == "restart"):
                    print(f"level {level} trial {trial} step {t} {direction}: "
                          f"restart flag {flag!r} vs engine "
                          f"{eng._rule_restart}")
                    bad += 1
                    break
                if live != new:
                    print(f"level {level} trial {trial} step {t} {direction}: "
                          f"model {new} != engine {live}")
                    bad += 1
                    break
                if targets != targets0:
                    print(f"level {level}: the TARGET set changed "
                          f"({sorted(targets0 - targets)}) -- the win "
                          f"condition could go vacuous")
                    bad += 1
                    break
                if board.won(new) != eng.check_win():
                    print(f"level {level} trial {trial} step {t}: win "
                          f"{board.won(new)} vs {eng.check_win()}")
                    bad += 1
                    break
                state = new
                if eng.check_win():
                    counts["win"] += 1
                    break
    print("fuzz:", counts)
    print("mismatches:", bad)
    return 1 if bad else 0


def _fuzz_plans(seed: int = 11) -> int:
    """The other half of the fuzz: random rollouts from every PREFIX of every
    level's own solution, which is where the interesting boards are.

    Uniform random play from a level start mostly wanders the opening room; the
    buoy-into-bomb destructions, the simultaneous push+pull and the boards with
    a buoy already parked on a target only show up along a real solution."""
    _solver, game, expert = _levels()
    eng = game._engine
    rng = random.Random(seed)
    ids = (expert.wall_ids, expert.target_ids, expert.buoy_ids,
           expert.bomb_ids, expert.octo_ids, expert.player_ids)
    bad = 0
    checked = 0
    for level in range(game.n_levels):
        game.set_level(level)
        presses, _sets = expert.solve(eng, level)
        if presses is None:
            continue
        for cut in range(0, len(presses), max(1, len(presses) // 12)):
            game.set_level(level)
            board = _Board.read(eng, ids)
            state = board.start
            for direction in presses[:cut]:
                state, flag = board.step(state, direction)
                eng.step(direction)
                assert flag == "ok"
            for _ in range(25):
                direction = rng.choice(_ORDER)
                new, flag = board.step(state, direction)
                eng.step(direction)
                checked += 1
                live = _Board.read(eng, ids).start
                if eng._rule_restart != (flag == "restart") or live != new:
                    print(f"level {level} cut {cut}: model {new} flag {flag} "
                          f"vs engine {live} restart {eng._rule_restart}")
                    bad += 1
                    break
                state = new
                if eng.check_win():
                    break
    print(f"prefix fuzz: {checked} presses, mismatches: {bad}")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# Render audit
# ---------------------------------------------------------------------------

def _audit() -> int:
    """Assert every cell COMPOSITION the game can show is distinguishable, at
    every cell size the levels render at.

    Whole frames, not a cell crop: `_render_frame` upscales the board to fill
    64x64 and letterboxes it, so the cell grid in the output is not
    ``cell_px``-aligned and slicing one cell out by arithmetic reads the wrong
    window (see solvers/generate_explod_training.py)."""
    _solver, game, _expert = _levels()
    g = game._game
    names = ("background", "target", "wall", "player", "buoy", "bomb", "octopus")
    idx = {n: g.obj_name_to_idx[n] for n in names}
    comps = {
        "floor": (),
        "wall": ("wall",),
        "target": ("target",),
        "player": ("player",),
        "player-on-target": ("target", "player"),
        "buoy": ("buoy",),
        "buoy-on-target": ("target", "buoy"),          # the WIN composition
        "bomb": ("bomb",),
        "bomb-on-target": ("target", "bomb"),
        "octopus": ("octopus",),
        "octopus-on-target": ("target", "octopus"),
    }
    sizes = set()
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        sizes.add(min(64 // eng.height, 64 // eng.width))
    bad = 0
    for cell_px in sorted(sizes):
        # A board just big enough to render at this cell size.
        side = 64 // cell_px
        game.set_level(0)
        eng = game._engine
        frames = {}
        for name, stack in comps.items():
            eng.height = eng.width = side
            eng.grid = [[{idx["background"]} | {idx[o] for o in stack}
                         for _ in range(side)] for _ in range(side)]
            eng._position_index_dirty = True
            frames[name] = np.asarray(_render_frame(eng, g)).copy()
        keys = sorted(frames)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if np.array_equal(frames[a], frames[b]):
                    print(f"cell_px {cell_px}: {a} and {b} render IDENTICALLY")
                    bad += 1
        print(f"cell_px {cell_px}: {len(keys)} compositions, "
              f"{'OK' if not bad else 'COLLISIONS'}")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# Verify: an episode end to end, in memory
# ---------------------------------------------------------------------------

def _verify(seeds: int = 6) -> int:
    """Solve several seeds in process and check three things. Writes nothing.

      * every level of every seed reaches ``GameState.WIN``;
      * every recorded EXPERT step carries an optimal-action target (the
        always-emit-optimal-targets rule);
      * **replaying the recorded action indices on a fresh adapter reproduces
        the recorded frames exactly, and wins.** That is the check that pins the
        rotation contract, and it needs its own replay rather than trusting the
        recording loop: `record_level` re-plans whenever the plan runs out, so
        an expert whose presses were remapped the wrong way round would keep
        re-planning from the states it did not intend and could still stumble
        into a win (it would just be slow) -- and the taped actions would then
        not reproduce the taped frames for the agent. Six seeds is enough to
        draw several distinct (rotation, hflip, vflip) presentations out of the
        16 this game is augmented over.
    """
    from arcengine import ActionInput, GameAction, GameState           # noqa: E402
    solver = BuoyDeploySolver()
    bad = 0
    for seed in range(seeds):
        ok, levels = solver.solve_episode(seed, explore=(seed % 2 == 1))
        gaps = sum(1 for lv in levels for a in lv["actions"][1:]
                   if a.get("phase", "expert") == "expert"
                   and not a.get("optimal"))
        resets = sum(1 for lv in levels for a in lv["actions"][1:]
                     if a["index"] == int(GameAction.RESET.value))

        replay = solver.make_game(seed)
        drift = 0
        for lv in levels:
            replay.set_level(lv["level_id"])
            frames = [np.asarray(replay._current_frame)]
            for act in lv["actions"][1:]:
                fd = replay.perform_action(
                    ActionInput(id=_ID_TO_GAMEACTION[act["index"]]))
                frames.append(np.asarray(fd.frame[-1] if fd.frame
                                         else replay._current_frame))
            if replay._state != GameState.WIN:
                drift += 1
            elif not all(np.array_equal(a, b) for a, b in
                         zip(frames, np.asarray(lv["observations"]))):
                drift += 1
        rot = (replay._rotation_k, replay._hflip, replay._vflip)
        print(f"seed {seed}: ok={ok} levels={[lv['level_id'] for lv in levels]} "
              f"steps={sum(len(lv['actions']) for lv in levels)} "
              f"unlabelled-expert-steps={gaps} resets={resets} "
              f"replay-mismatches={drift} rot={rot}")
        bad += (not ok) or gaps or drift
    return 1 if bad else 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--report" in argv or "--plans" in argv:
        return _report()
    if "--fuzz" in argv:
        return _fuzz() or _fuzz_plans()
    if "--audit" in argv:
        return _audit()
    if "--verify" in argv:
        return _verify()
    return BuoyDeploySolver.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
