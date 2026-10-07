"""ps:snake_crate -- Joshua Rigsby's "Snake Crate": a sokoban whose player is a
SNAKE, so every square it walks on is burned behind it forever.

THE MECHANIC (four rules, and three of them are one sentence each)

    [ > Player | Crate ]  -> [ > Player | > Crate ]      push
    [ < Player | Crate ]  -> [ < Player | < Crate ]      PULL
    LATE [ player no start ] -> [ player Snake ]         the trail
    [ > player | Snake ]  -> cancel                      you may not re-enter it

`All Crate on Target` is the win. Four consequences shape the whole solver, and
none of them is visible in the rules as written:

* **The player both pushes and pulls, in the same press.** `<` is the rule's
  direction reversed, so the second rule reads "a crate BEHIND you comes with
  you". A press can therefore shove the crate in front and drag the one behind
  at the same time -- which is also why this file's heuristic is a search guide
  and not an optimality certificate (see `SnakeCrateBoard.heuristic`).

* **A pull always leaves its crate ON the trail, and a crate on the trail can
  never be pushed again.** The crate lands in the square the player has just
  vacated, and that square grew a Snake the moment the player entered it. Every
  push needs the player to step INTO the crate's square, and stepping into a
  Snake cancels the turn -- from any direction. So a crate is pushable until the
  first time it is pulled and pull-only for ever after, which the distance
  tables model as two separate reverse BFS passes (`_table`).

* **The start square is the one square that never burns.** The trail rule is
  gated on `no start`, so the player may walk back onto its starting square as
  often as it likes, and a crate parked there stays PUSHABLE. It is also what
  keeps the state graph acyclic: every press either grows the trail or steps
  onto start, and from start every press grows the trail.

* **A blocked press and a cancelled press are the same thing to the board.**
  Nothing distinguishes them: the LATE rule is idempotent (the player already
  stands on its own Snake, or on start, which it refuses to mark). ACTION is a
  no-op for the same reason -- no rule reads it, and the trail rule has nothing
  left to do. `--selfcheck` fuzzes all of that against the interpreter.

THE SEARCH: A* over ``walk to a stance, then press the same direction k times``

A primitive-move A* drowns here. The trail makes every distinct walk a distinct
state, so there is no player-region canonicalisation to collapse them with (the
standard sokoban dedup `PSPushExpert` uses is unsound in this game -- two routes
to one square leave different boards), and a 100-press solution is a depth-100
search over branching 4. It solves levels 0-3 and dies on level 4.

Macros fix the depth: a successor is "walk the shortest route to a square from
which pressing d moves a crate, then press d, k times". Pushes and pulls come
out of the same enumeration -- the stance for a push of crate c in direction d
is `c - d` and for a pull it is `c + d`, and the press is `d` either way -- so
the branching is (crates x 4 directions x 2 sides x run length) and the depth is
the number of direction CHANGES, ~10 rather than ~100.

The heuristic is what makes it work at all. `PSSokobanExpert`'s (push-distance
tables + the walk to the first push) solves 8 of the 11 levels and hits the node
cap on the three biggest -- on boards this open the crate distances are almost
Manhattan and carry no information, and what the plan actually spends its
presses on is the player walking BETWEEN crates. `heuristic` therefore charges
a greedy TOUR: match crates to targets over the distance tables, then serve them
in nearest-first order, paying the walk from where the previous crate was
DELIVERED to the next crate. That is the whole difference between 8/11 and 11/11.

Two prunes carry the rest. A state is dead when a loose crate has no neighbour
in the CRATE-PERMEABLE flood from the player (walls and trail only): the trail
only ever grows, so a square outside that flood is a square the player can never
reach again, and a crate none of whose neighbours is reachable can never move.
And a crate that no distance table can reach at all -- the pull-only table for a
crate on the trail -- scores `DEAD` and sinks.

Plans are winning and interpreter-verified. They are NOT certified shortest:
the walk inside a macro is one arbitrary shortest route, the tour heuristic
overestimates by construction, and levels the ladder solves at weight 2 are
weighted A*. Levels 0-3 do match a primitive A*'s proven optimum (12/19/26/36).

OPTIMAL SETS ARE PROVED, not inferred. `optimal_sets` offers an alternative
press only when it can EXHIBIT an equally-short winning completion: reroute the
rest of the current walk run to the same stance square and replay the plan's
whole suffix on the model. If the reroute still wins in the same number of
presses, the alternative is as good as the expert's own choice -- which is the
only sound claim available here, because a walk in this game is NOT inert
(`PSPushExpert.annotate_walks`, which assumes it is, would label presses that
burn the wrong square). Manipulation presses are labelled with themselves: which
crate to move where is the puzzle.

ART. Four sprites and one collision-layer line in
`data/puzzlescript_games/Snake_Crate.txt` -- the trail drew OVER the crates, the
start square drew over them too and was the player's own colour, the crate was
painted the wall's colour, and the target hid under every body. The file's own
header has the detail; `--audit` is the regression test.

VERIFIED. 11/11 levels, 621 presses, cold build 31 s and 138 MB (the plan
cache is `data/snake_crate_plans.json`, so every later shard starts warm), and a
rebuild under a different PYTHONHASHSEED reproduces the cache byte for byte.
`--selfcheck` fuzzes 33k presses over both modes with 0 divergences (505 crate
moves, 169 of them landing a crate on the trail and 133 pushes the trail
refused, which is the mechanic that needed covering). `--audit`: 15 compositions
pairwise distinct at cell_px 3 and 4, the only two sizes the levels use.
`--symmetry`: 11 levels x 16 presentations, every plan a WIN and every frame of
both the plan and a 200-press random walk exactly the transform of the
unaugmented one. 8 seeds x 11 levels recorded: every level a WIN, all 5062 steps
replay frame-exact through the adapter from the recorded SCREEN actions,
4968/4968 expert steps labelled (1.11 optimal presses/step, the taken press
always in its own set), indices in 0..5, no zero-expert level, and two processes
at different hash seeds write byte-identical episodes. Worst plan is 119 presses
against the adapter's 200-press cap.

RECOVERY is the family's RESET prefix (`recovery_mode = "reset"`): the trail is
irreversible, so a perturbed board cannot be re-planned from in general, and one
RESET restores exactly the state the cached plan was solved from.

CLI: ``--selfcheck`` (model vs interpreter fuzz), ``--audit`` (every cell
composition pixel-distinct at every cell size the levels use), ``--symmetry``
(every presentation is an exact transform of the unaugmented frames), ``--plans``
(every level's plan, replayed through the interpreter), otherwise the BaseSolver
CLI.
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
    PuzzleScriptAdapter, _render_cell_sprite)
from arcengine import ActionInput, GameState        # noqa: E402
from solvers.base_solver import _ID_TO_GAMEACTION   # noqa: E402
from solvers.common.ps_astar import (              # noqa: E402
    PSAStarSolver, PSExpert, Plan, screen_action)
from utils.rotation import inverse_remap_action_full  # noqa: E402

GAME_NAME = "Snake_Crate"
GAME_ID = "ps:snake_crate"

#: Engine directions, in the order the model indexes them. ``k ^ 1`` is the
#: reverse of ``k`` -- which is how one enumeration produces both the push
#: stance (``c - d``) and the pull stance (``c + d``).
DIRNAMES = ("up", "down", "left", "right")
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: Heuristic charge for a board that can no longer be won. Large enough to sink
#: the node, finite so the search stays complete.
DEAD = 10_000

#: Longest run of one direction a single macro may press. No level's board is
#: 40 squares across, so this only bounds a pathological loop.
MAX_RUN = 40


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class SnakeCrateBoard:
    """One level, as a model the search can step in microseconds.

    A state is ``(player, trail, crates)``: a flat cell index, a bitmask of the
    burned squares, and a sorted tuple of crate cells. It is exact -- walls,
    targets and the start square are static, and nothing else in the game moves
    -- and `--selfcheck` is the proof.
    """

    def __init__(self, eng, ids):
        grid = eng.grid
        self.h, self.w = len(grid), len(grid[0])
        n = self.h * self.w
        wall_id, crate_id, target_id = ids["wall"], ids["crate"], ids["target"]
        snake_id, start_id, player_id = ids["snake"], ids["start"], ids["player"]
        self.wall = [False] * n
        targets, crates, trail = [], [], 0
        player = self.start = None
        for r in range(self.h):
            for c in range(self.w):
                cell, i = grid[r][c], r * self.w + c
                if wall_id in cell: self.wall[i] = True
                if target_id in cell: targets.append(i)
                if start_id in cell: self.start = i
                if crate_id in cell: crates.append(i)
                if snake_id in cell: trail |= 1 << i
                if player_id in cell: player = i
        self.targets = tuple(sorted(targets))
        self.target_set = frozenset(self.targets)
        self.state = (player, trail, tuple(sorted(crates)))

        self.nb = [[None] * 4 for _ in range(n)]
        for r in range(self.h):
            for c in range(self.w):
                i = r * self.w + c
                for k, (dr, dc) in enumerate(_DELTA):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.h and 0 <= cc < self.w:
                        self.nb[i][k] = rr * self.w + cc

        # Static tables. `tab_all` is for a crate that can still be pushed,
        # `tab_pull` for one standing on the trail -- see the module docstring.
        self.tab_all = {t: self._table(t, False) for t in self.targets}
        self.tab_pull = {t: self._table(t, True) for t in self.targets}
        self.walkfield = {t: self._walk_field(t) for t in self.targets}

    # -- dynamics -----------------------------------------------------------
    def step(self, st, k):
        """One press. Returns the same state object when the press does nothing
        -- a wall or a crate the board refuses to move (blocked), or a Snake in
        the square ahead (cancelled). The two are indistinguishable on the
        board, which is why the model does not distinguish them either."""
        p, trail, crates = st
        nxt = self.nb[p][k]
        if nxt is None or self.wall[nxt] or (trail >> nxt) & 1:
            return st
        if nxt in crates:
            beyond = self.nb[nxt][k]
            if beyond is None or self.wall[beyond] or beyond in crates:
                return st
            crates = tuple(sorted([x for x in crates if x != nxt] + [beyond]))
        back = self.nb[p][k ^ 1]
        if back is not None and back in crates:      # the pull, same press
            crates = tuple(sorted([x for x in crates if x != back] + [p]))
        if nxt != self.start:
            trail |= 1 << nxt
        return (nxt, trail, crates)

    def won(self, st):
        return self.target_set.issuperset(st[2])

    # -- static tables ------------------------------------------------------
    def _table(self, target, pull_only):
        """``{cell: crate-moves to bring a crate from cell onto target}``.

        Reverse BFS over crate moves, ignoring the trail and the other crates
        (the standard sokoban relaxation). A crate goes from ``c`` to ``c + d``
        when ``c + d`` is clear and either the player can stand BEHIND it at
        ``c - d`` (a push) or IN FRONT at ``c + d`` and step on to ``c + 2d`` (a
        pull). ``pull_only`` drops the push half, which is what a crate already
        standing on the trail is limited to.
        """
        dist = {target: 0}
        q = deque([target])
        while q:
            x = q.popleft()
            for k in range(4):
                c = self.nb[x][k ^ 1]              # the crate came from here
                if c is None or self.wall[c] or c in dist:
                    continue
                ahead = self.nb[x][k]              # c + 2d, where a pull lands
                ok = ahead is not None and not self.wall[ahead]
                if not ok and not pull_only:
                    behind = self.nb[c][k ^ 1]     # c - d, where a push stands
                    ok = behind is not None and not self.wall[behind]
                if ok:
                    dist[c] = dist[x] + 1
                    q.append(c)
        return dist

    def _walk_field(self, src):
        """Walk distance from ``src`` over the walls alone -- the trail is not
        in it, because this measures travel the plan has not made yet."""
        dist = {src: 0}
        q = deque([src])
        while q:
            x = q.popleft()
            for k in range(4):
                y = self.nb[x][k]
                if y is not None and not self.wall[y] and y not in dist:
                    dist[y] = dist[x] + 1
                    q.append(y)
        return dist

    # -- search guidance ----------------------------------------------------
    def flood(self, st, permeable=False):
        """Walk distances from the player. ``permeable`` lets the flood run
        THROUGH crates, which turns it into an over-approximation of every
        square the player can ever reach -- sound because the trail only grows,
        and that is what makes `dead` a proof rather than a guess."""
        p, trail, crates = st
        blocked = frozenset() if permeable else frozenset(crates)
        dist = {p: 0}
        q = deque([p])
        while q:
            x = q.popleft()
            for k in range(4):
                y = self.nb[x][k]
                if (y is None or self.wall[y] or y in dist
                        or (trail >> y) & 1 or y in blocked):
                    continue
                dist[y] = dist[x] + 1
                q.append(y)
        return dist

    def dead(self, st):
        """True when a loose crate has no reachable neighbour left, i.e. no
        press will ever touch it again."""
        seen = self.flood(st, permeable=True)
        for c in st[2]:
            if c in self.target_set:
                continue
            if not any(self.nb[c][k] in seen for k in range(4)
                       if self.nb[c][k] is not None):
                return True
        return False

    def heuristic(self, st):
        """Crate work + a greedy player TOUR over the crates still to serve.

        The crate half is a min-cost assignment over the distance tables (the
        levels have at most four crates, so it is exact by enumeration). The
        tour half is what the plans actually spend their presses on: walk to the
        nearest loose crate, deliver it, walk from THAT target to the next
        nearest, and so on.

        Deliberately not admissible -- the tour is greedy, and a single press
        can move two crates (push in front, pull behind) -- so this is a search
        guide, in the same sense as `PSSokobanExpert`'s. See the module
        docstring on what that costs.
        """
        p, trail, crates = st
        if self.target_set.issuperset(crates):
            return 0
        cost = []
        for c in crates:
            table = self.tab_pull if (trail >> c) & 1 else self.tab_all
            cost.append([table[t].get(c) for t in self.targets])
        best = bestperm = None
        for perm in itertools.permutations(range(len(self.targets)),
                                           len(crates)):
            total = 0
            for ci, ti in enumerate(perm):
                d = cost[ci][ti]
                if d is None:
                    break
                total += d
            else:
                if best is None or total < best:
                    best, bestperm = total, perm
        if best is None:
            return DEAD
        assign = {crates[ci]: self.targets[ti]
                  for ci, ti in enumerate(bestperm)}
        todo = [c for c in crates if c != assign[c]]
        if not todo:
            return best
        field = self.flood(st)
        travel = 0
        while todo:
            pick = pd = None
            for c in todo:
                d = min((field[y] for k in range(4)
                         if (y := self.nb[c][k]) is not None and y in field),
                        default=None)
                if d is not None and (pd is None or d < pd):
                    pick, pd = c, d
            if pick is None:
                return DEAD
            travel += pd
            todo.remove(pick)
            field = self.walkfield[assign[pick]]
        return best + travel

    # -- macros -------------------------------------------------------------
    def walk_tree(self, st):
        """BFS parent pointers from the player over the squares it may walk on:
        no wall, no trail, no crate (walking into a crate is a press that moves
        it, which is a macro, not a walk)."""
        p, trail, crates = st
        cr = frozenset(crates)
        parent = {p: None}
        q = deque([p])
        while q:
            x = q.popleft()
            for k in range(4):
                y = self.nb[x][k]
                if (y is None or self.wall[y] or y in parent
                        or (trail >> y) & 1 or y in cr):
                    continue
                parent[y] = (x, k)
                q.append(y)
        return parent

    @staticmethod
    def route(parent, cell):
        out = []
        while parent[cell] is not None:
            cell, k = parent[cell]
            out.append(k)
        out.reverse()
        return out

    def macros(self, st):
        """Every ``walk to a stance, then press d up to k times``, as
        ``(presses, resulting state)`` pairs -- one per run length, so a run
        that overshoots its target is still available as a shorter one."""
        parent = self.walk_tree(st)
        stances = set()
        for c in st[2]:
            for k in range(4):
                for side in (k ^ 1, k):            # push stance, pull stance
                    s = self.nb[c][side]
                    if s is not None and s in parent:
                        stances.add((s, k))
        out = []
        for (stance, k) in sorted(stances):
            walk = self.route(parent, stance)
            cur = st
            for wk in walk:
                cur = self.step(cur, wk)
            if cur[0] != stance:                   # a crate moved under the walk
                continue
            for n in range(1, MAX_RUN + 1):
                nxt = self.step(cur, k)
                if nxt is cur or nxt == cur:
                    break
                cur = nxt
                out.append((walk + [k] * n, cur))
                if self.won(cur):
                    break
        return out


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------

def astar(board, weight, node_cap):
    """Macro A*, memory-light: the heap carries a node INDEX, never a path, so
    a search that runs to its cap costs a few hundred bytes a node instead of
    the whole prefix. Returns a flat list of press indices, or None."""
    st = board.state
    if board.won(st):
        return []
    nodes = [(st, -1, ())]                     # (state, parent, presses taken)
    pq = [(weight * board.heuristic(st), 0, 0)]
    best_g = {st: 0}
    expanded = 0

    def rebuild(idx, tail):
        out = list(tail)
        while idx >= 0:
            _s, parent, presses = nodes[idx]
            out = list(presses) + out
            idx = parent
        return out

    while pq:
        _f, g, idx = heapq.heappop(pq)
        st = nodes[idx][0]
        if best_g.get(st, 1 << 30) < g:
            continue
        for presses, ns in board.macros(st):
            expanded += 1
            if board.won(ns):
                return rebuild(idx, presses)
            ng = g + len(presses)
            if best_g.get(ns, 1 << 30) <= ng:
                continue
            h = board.heuristic(ns)
            if h >= DEAD or board.dead(ns):
                continue
            best_g[ns] = ng
            nodes.append((ns, idx, tuple(presses)))
            heapq.heappush(pq, (ng + weight * h, ng, len(nodes) - 1))
        if expanded >= node_cap:
            return None
    return None


def optimal_sets(board, plan):
    """Per-step optimal press SETS for ``plan``, PROVED by exhibiting an
    equally-short winning completion for every alternative offered.

    A walk run inside a macro ends on the stance square of the press that
    follows it, and any other shortest route to that square costs the same --
    but in this game it does NOT leave the same board, because the route it
    takes is burned behind it. So an alternative first step is offered only
    when rerouting the rest of the run through it and then replaying the plan's
    entire suffix still wins in the same number of presses. Manipulation
    presses (anything that moved a crate) are labelled with themselves alone.
    """
    states = [board.state]
    for k in plan:
        states.append(board.step(states[-1], k))
    kinds = ["move" if states[i + 1][2] == states[i][2] else "crate"
             for i in range(len(plan))]

    def replay(st, presses):
        for k in presses:
            nxt = board.step(st, k)
            if nxt == st:                       # a press that did nothing
                return None
            st = nxt
            if board.won(st):
                return st
        return st if board.won(st) else None

    out = [[k] for k in plan]
    i = 0
    while i < len(plan):
        if kinds[i] == "crate":
            i += 1
            continue
        run = i
        while i < len(plan) and kinds[i] == "move":
            i += 1
        stance = states[i][0]                   # where the run has to end up
        for j in range(run, i):
            left = i - j - 1                    # presses left in the run after j
            for k in range(4):
                if k == plan[j]:
                    continue
                nxt = board.step(states[j], k)
                if nxt == states[j] or nxt[2] != states[j][2]:
                    continue                    # a no-op, or a different plan
                parent = board.walk_tree(nxt)
                if stance not in parent:
                    continue
                detour = board.route(parent, stance)
                if len(detour) != left:
                    continue
                end = replay(nxt, detour + plan[i:])
                if end is not None:
                    out[j].append(k)
            out[j].sort()
    return out


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class SnakeCrateExpert(PSExpert):
    """`PSExpert`'s plan memo, restore discipline and disk cache around the
    native search. `heuristic` is never called -- the model carries its own."""

    directions = list(DIRNAMES)
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "snake_crate_plans.json")

    #: ``(weight, node cap)`` tried in order. Weight 1 is the plan we want and
    #: solves 10 of the 11 levels inside the cap; level 10 (three crates on the
    #: most open board in the game) needs weight 2, where it lands in a second.
    #: The cap is deliberately modest: an uncapped weight-1 search of that level
    #: does finish, at four million macros and several GB, and no plan is worth
    #: swapping the machine out.
    ladder = ((1, 400_000), (2, 400_000), (3, 400_000))

    OBJECTS = ("wall", "player", "crate", "target", "snake", "start")

    def setup(self):
        g = self.g
        self.ids = {n: g.obj_name_to_idx[n] for n in self.OBJECTS}
        self.found_at = {}                       # level -> weight that solved it

    def heuristic(self, eng):
        raise AssertionError(
            "SnakeCrateExpert plans on its native model; heuristic is unused")

    def board(self, eng):
        return SnakeCrateBoard(eng, self.ids)

    def _search(self, eng):
        board = self.board(eng)
        for weight, cap in self.ladder:
            plan = astar(board, weight, cap)
            if plan is not None:
                self._last_weight = weight
                return Plan([DIRNAMES[k] for k in plan],
                            [[DIRNAMES[k] for k in s]
                             for s in optimal_sets(board, plan)])
        return None

    def plan(self, eng, level=None):
        self._last_weight = None
        got = super().plan(eng, level)
        if level is not None and self._last_weight is not None:
            self.found_at[level] = self._last_weight
        return got

    def describe(self, level):
        w = self.found_at.get(level)
        return "cached" if w is None else f"A* weight {w}"


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class SnakeCrateSolver(PSAStarSolver):
    game_id = GAME_ID
    game_name = GAME_NAME
    game_module_id = GAME_ID
    expert_cls = SnakeCrateExpert
    max_steps = 300


# ---------------------------------------------------------------------------
# --selfcheck: the model against the interpreter
# ---------------------------------------------------------------------------

def _read_engine(eng, ids):
    """The interpreter's board, in the model's own terms."""
    player, crates, trail = None, [], 0
    w = len(eng.grid[0])
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            i = r * w + c
            if ids["player"] in cell: player = i
            if ids["crate"] in cell: crates.append(i)
            if ids["snake"] in cell: trail |= 1 << i
    return (player, trail, tuple(sorted(crates)))


def selfcheck(trials=25, steps=60, verbose=True):
    """Fuzz the model against the real interpreter from every level start.

    Random play is the fuzz: it walks into walls, into crates it cannot move and
    into its own trail (the blocked press and the cancelled press, which must
    both leave the board alone), it presses ACTION (which must do nothing at
    all), it pulls crates onto the trail and then tries to push them, and it
    runs long enough to seal itself into a corner. After every press the whole
    board -- player, every crate, every trail square -- and the win flag have to
    agree.

    It runs in two modes, and it needs both. UNIFORM play is the honest fuzz,
    but a snake that presses at random walks into its own trail almost at once
    and spends the rest of the rollout pressing against it -- three quarters of
    those presses are no-ops, which tests the cancel and nothing else. GUIDED
    play draws (most of the time) from the presses the MODEL believes are live,
    so the rollouts keep moving and actually push, pull and wedge crates. On its
    own that would be circular -- a press the model wrongly calls dead would
    never be tried -- so the uniform mode, which is what covers the no-op
    classification, stays in and both are compared against the engine the same
    way.

    It is also the proof that the collision-layer move in the .txt is inert: a
    rule that read layer order would show up here as a divergence."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    ids = {n: game._game.obj_name_to_idx[n] for n in SnakeCrateExpert.OBJECTS}
    bad = presses = moved = cancels = wins = 0
    wedged = pullonly = 0
    for level in range(game.n_levels):
        for guided in (False, True):
            for t in range(trials):
                game.set_level(level)
                board = SnakeCrateBoard(eng, ids)
                st = board.state
                rng = random.Random(
                    f"snake_crate:selfcheck:{level}:{guided}:{t}")
                for _ in range(steps):
                    nexts = [board.step(st, k) for k in range(4)]
                    crateful = [k for k in range(4)
                                if nexts[k][2] != st[2]]
                    live_presses = [k for k in range(4) if nexts[k] != st]
                    roll = rng.random()
                    if guided and crateful and roll < 0.7:
                        k = rng.choice(crateful)
                    elif guided and live_presses and roll < 0.95:
                        k = rng.choice(live_presses)
                    else:
                        k = rng.randrange(5)     # 5 = ACTION, a no-op
                    before = st
                    eng.step("action" if k == 4 else DIRNAMES[k])
                    st = st if k == 4 else board.step(st, k)
                    presses += 1
                    if st == before:
                        cancels += 1
                        if before[0] is not None and k < 4:
                            ahead = board.nb[before[0]][k]
                            if ahead is not None and ahead in before[2]:
                                wedged += 1      # a push the trail refused
                    elif st[2] != before[2]:
                        moved += 1
                        pullonly += sum(1 for c in st[2]
                                        if (st[1] >> c) & 1)
                    live = _read_engine(eng, ids)
                    if live != st or eng.check_win() != board.won(st):
                        bad += 1
                        if bad < 4:
                            print(f"  level {level} trial {t} guided={guided}: "
                                  f"model {st} != engine {live} "
                                  f"(win {board.won(st)}/{eng.check_win()})")
                        break
                    if board.won(st):
                        wins += 1
                        break
    if verbose:
        print(f"  {presses} presses fuzzed, {cancels} of them no-ops, "
              f"{moved} crate moves ({pullonly} landing a crate on the trail, "
              f"{wedged} pushes the trail refused), {wins} accidental wins")
    return bad


# ---------------------------------------------------------------------------
# --audit: every cell composition, pixel-distinct at every cell size
# ---------------------------------------------------------------------------

#: Every stack of objects a square in this game can hold. The ones that cannot
#: occur are left out on purpose: the start square never grows a Snake (the
#: trail rule refuses to mark it) and no level puts a Target on it, and the
#: player is never off both -- it starts on the start square and every square it
#: enters afterwards grows a Snake under it.
_AUDIT_CASES = {
    "floor": ["background"],
    "wall": ["background", "wall"],
    "grass": ["background", "grass"],
    "target": ["background", "target"],
    "trail": ["background", "snake"],
    "trail on target": ["background", "snake", "target"],
    "start": ["background", "start"],
    "crate": ["background", "crate"],
    "crate on target": ["background", "crate", "target"],
    "crate on trail": ["background", "snake", "crate"],
    "crate on trail on target": ["background", "snake", "crate", "target"],
    "crate on start": ["background", "crate", "start"],
    "player": ["background", "snake", "player"],
    "player on target": ["background", "snake", "player", "target"],
    "player on start": ["background", "player", "start"],
}


def audit(verbose=True):
    """Assert every composition above renders differently at every cell size
    the levels use. Composition, not object: all four bugs the .txt header
    describes lived in the STACK, not in a single sprite.

    The stack is built the way `_render_frame` builds it -- non-player objects
    by collision layer, then the player appended LAST unconditionally -- which
    is exactly why the start square had to become a bottom-right corner block:
    it draws above the crate (so it used to erase it) but below the player."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    parsed = game._game
    layers = game._engine._obj_layers
    players = set(game._engine._player_indices)
    sizes = set()
    for level in range(game.n_levels):
        game.set_level(level)
        h, w = len(game._engine.grid), len(game._engine.grid[0])
        sizes.add(max(1, min(64 // h, 64 // w)))

    def stack(names):
        body, player = [], []
        for n in names:
            idx = parsed.obj_name_to_idx[n]
            (player if idx in players else body).append(
                (layers.get(idx, -1), parsed.objects[n]))
        body.sort(key=lambda x: x[0])
        player.sort(key=lambda x: x[0])
        return body + player

    bad = 0
    for px in sorted(sizes):
        blocks = {k: _render_cell_sprite(stack(v), px)
                  for k, v in _AUDIT_CASES.items()}
        clashes = [(a, b) for a, b in itertools.combinations(_AUDIT_CASES, 2)
                   if bool((blocks[a] == blocks[b]).all())]
        for a, b in clashes:
            print(f"  cell_px={px}: {a!r} and {b!r} render identically")
        bad += len(clashes)
        if verbose:
            print(f"  cell_px={px}: {len(_AUDIT_CASES)} compositions, "
                  f"{'all distinct' if not clashes else 'COLLISIONS'}")
    return bad


# ---------------------------------------------------------------------------
# --symmetry: the presentation augmentation is a true symmetry of the mechanic
# ---------------------------------------------------------------------------

def symmetry(walk_presses=200, verbose=True):
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unaugmented ones.

    This is the evidence for putting the game in `_FLIP_GAMES` (rotation is
    mandatory and is checked here too). The structural argument is that every
    rule in the game is written with the RELATIVE forces `>` and `<`, so not one
    of them names an axis, and the win condition is positional -- but this game
    moves two crates in a single press (the one in front and the one behind), and
    a multi-body press is exactly the shape that hid Gobble Rush's chirality. So
    it is measured rather than argued: the two bodies a press moves land on
    `p + 2d` and `p`, which can never be the same square, so no two moves can
    contest a cell and the order the interpreter expands a rule's four directions
    cannot decide anything.

    Both the PLANS and a seeded random walk are replayed -- the walk is what
    reaches the boards a plan never visits (crates wedged against the trail, the
    player sealed into a pocket), and it presses ACTION too."""
    solver = SnakeCrateSolver()
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
        if k: frame = np.rot90(frame, k=k)
        if hflip: frame = np.fliplr(frame)
        if vflip: frame = np.flipud(frame)
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
            rng = random.Random(f"snake_crate:symmetry:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(walk_presses)]
            for tag, presses in (("plan", [screen_action(d, *k)
                                           for d in plans[level]]),
                                 ("walk", [inverse_remap_action_full(a, *k)
                                           for a in walk])):
                frames = drive(g, level, presses)
                if tag == "plan" and g._state != GameState.WIN:
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
        print(f"  {len(seen)} presentations drawn: "
              f"{sorted(seen)}")
    return bad


# ---------------------------------------------------------------------------
# --plans
# ---------------------------------------------------------------------------

def _plan_report():
    """Every level's plan, replayed through the real interpreter -- the
    end-to-end test of the model, the search and the tie labelling at once."""
    solver = SnakeCrateSolver()
    game, expert, solvable = solver._ensure(0)
    total = ties = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level:2d}: UNSOLVED")
            continue
        for direction in plan:
            eng.step(direction)
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        step_ties = sum(len(s) - 1 for s in sets)
        total += len(plan)
        ties += step_ties
        print(f"  L{level:2d}: {len(plan):3d} presses  win={eng.check_win()}  "
              f"{expert.describe(level):>12}  {step_ties:3d} tie-presses")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second right answer")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        violations = selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--plans" in sys.argv:
        _plan_report()
        sys.exit(0)
    sys.exit(SnakeCrateSolver.main())
