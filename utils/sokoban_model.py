"""Sokoban expert: push-level A* over ``(player, boxes)``, counted in MOVES.

The model is pure geometry + dynamics -- it never touches the engine -- so the
generator (`solvers/generate_sokoban_training.py`) can rebuild it from whatever
board the LIVE level happens to be showing and re-plan from there. One instance
serves every state of one level geometry (walls + targets) and caches what it
learns across calls.

WHY PUSH-LEVEL AND NOT MOVE-LEVEL
---------------------------------
``utils/push_puzzle.PushModel`` (the pb01/pb02/pb03 expert) searches MOVES: 4-way
branching over ``(player, crates)``. That is fine for a 10x10 board with two
crates and a ~30-move answer; ``games/sokoban`` runs to 19x17 with five crates,
so move-level A* never terminates there.

So the search here is over PUSHES. Between two pushes the player only walks, and
the cheapest walk is a BFS on the static board, so one search node = one push and
its edge weight is ``walk_distance + 1``. Path cost is therefore still counted in
real moves (a push is a move too), which keeps answers MOVE-optimal -- not merely
push-optimal -- while collapsing the branching factor to ``4 * n_boxes`` and the
depth to the number of pushes. Two details keep that sound:

* the player position is kept EXACT in the state key. The usual "normalise the
  player to its reachable region's canonical cell" trick is only valid when
  optimising pushes: which side of the box the player is standing on changes the
  next walk, hence the move cost.
* the heuristic is a min-cost box->target assignment over TRUE push distances
  (`_target_fields`, a reverse "pull" BFS per target), a lower bound on the
  remaining pushes and hence on the remaining moves.

OPTIMALITY vs TIME
------------------
With that (push-only) heuristic, exact move-optimal search is affordable on the
small levels and NOT on the big ones: the bound ignores walking, which is ~2/3 of
the real cost, so A* degenerates towards Dijkstra (level 6 of seed 0: 270k
expansions, ~90 s -- and levels the exact search cannot finish at all inside 400k).
`plan` therefore escalates a weight schedule (`WEIGHTS`): each attempt runs
weighted A* (``f = g + w*h``) under an expansion cap and the first one that
succeeds wins. ``w = 1`` comes first, so whenever the exact optimum is cheap that
is exactly what comes back (`Plan.exact` records it); the later attempts trade a
bounded amount of optimality -- a ``w``-weighted answer is at most ``w`` times
optimal -- for a search that terminates.

Measured over the 42 boards of seeds 0-5 (the tuning run behind the constants
below): total plan length +1.2% against the most patient schedule tried, in 55%
of its search time -- and on every board where the exact search DID terminate,
the shipped schedule returned that same exact optimum.

DEADLOCKS
---------
Two sound tests, used both to prune the search and to answer "is this state still
winnable?" for the generator's recovery path:

* **dead squares** -- cells from which no sequence of pushes can reach ANY target
  (the pull-BFS above never labelled them). Covers corners, wall-hugging lanes
  with no target on them, and pockets behind interior walls.
* **freeze deadlock** -- the classic mutual-blocking test: a box is frozen when it
  is immovable on both axes, counting walls, dead squares, and OTHER boxes that
  are themselves frozen on the perpendicular axis. A frozen box off a target means
  the state is lost.

Both are one-sided -- they can only ever declare a LOST state lost -- so pruning
with them cannot make the expert give up on a winnable board.

INCREMENTAL RE-PLANNING
-----------------------
`BaseSolver.record_level` re-invokes ``solve_from`` after every step with more
than one co-optimal action (and after every exploratory detour), so a naive expert
would run a full A* per move. `plan` therefore keeps the last answer and first
tries to CONTINUE it: if the previous plan, re-costed from the new state, comes to
exactly ``previous_cost - 1``, it is still optimal (one move cannot buy more than
one move of progress) and no search runs. Walking down a plan and executing its
pushes both hit that path, so a whole level costs ~one search.
"""
from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import dataclass

#: action id -> (dx, dy), matching ``games/sokoban``'s ``_DELTAS``
#: (ACTION1 up, ACTION2 down, ACTION3 left, ACTION4 right).
DELTAS: dict[int, tuple[int, int]] = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
ACTION_IDS: tuple[int, ...] = (1, 2, 3, 4)

_INF = 1 << 30

#: ``(weight, expansion cap)`` attempts, in order. The first entry is the exact
#: (admissible) search, so small boards always come back move-optimal; the rest
#: only ever run when that one hits its cap.
WEIGHTS: tuple[tuple[float, int], ...] = (
    (1.0, 2_000), (1.5, 4_000), (2.0, 10_000), (4.0, 40_000), (8.0, 150_000),
)


@dataclass(frozen=True)
class Push:
    """One push of a plan, plus what is needed to re-cost it from a new state.

    Cells are integer indices (``y * width + x``). ``suffix_cost`` is the number
    of MOVES from standing on ``stand`` to the win: this push (1) + the walk to
    the next push's ``stand`` + that push's suffix. So the cost of a whole plan
    from an arbitrary player cell is just
    ``walk(player, pushes[0].stand) + pushes[0].suffix_cost``.
    """

    stand: int                      # cell the player must stand on to push
    action: int                     # the action id that performs the push
    box: int                        # the box being pushed (== stand + delta)
    dest: int                       # where it lands (== box + delta)
    boxes_after: tuple              # sorted box tuple once the push is made
    suffix_cost: int


@dataclass(frozen=True)
class Plan:
    """An optimal (or, when ``exact`` is False, ``weight``-bounded) plan."""

    pushes: tuple[Push, ...]
    cost: int                       # total moves from the state it was made for
    exact: bool = True


class SokobanModel:
    """Push dynamics + a move-count planner for ONE board geometry.

    Cells are ``(x, y)`` pairs across the public API and integer indices inside;
    `idx` / `cell` convert. Boxes are passed as iterables of cells and are
    normalised to a sorted tuple of indices, which is the state key.
    """

    def __init__(self, width: int, height: int, walls, targets) -> None:
        self.width = w = int(width)
        self.height = h = int(height)
        n = w * h
        self.n = n
        walls = {self.idx(c) for c in walls}
        self.targets = frozenset(self.idx(c) for c in targets)
        self.floor = [i for i in range(n) if i not in walls]
        open_ = bytearray(n)
        for i in self.floor:
            open_[i] = 1
        self._open = open_

        # adjacency + push table, precomputed so the hot loops are index lookups
        self.adj: list[tuple[int, ...]] = [()] * n
        #: cell -> ((action, stand, dest), ...) for every direction whose stand
        #: AND destination cells are floor (i.e. the push is geometrically legal
        #: ignoring the other boxes)
        self.push_tab: list[tuple[tuple[int, int, int], ...]] = [()] * n
        #: cell -> the cells a box could have been pushed FROM to land here.
        #: The exact inverse of `push_tab`, which is what makes the pull-BFS in
        #: `_target_fields` agree with the push rule (a pull needs the cell
        #: BEHIND the box to be floor, not the one in front).
        pull: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            if not open_[i]:
                continue
            x, y = self.cell(i)
            nbrs, pushes = [], []
            for action, (dx, dy) in DELTAS.items():
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w and 0 <= ny < h) or not open_[ny * w + nx]:
                    continue
                nbrs.append(ny * w + nx)
                sx, sy = x - dx, y - dy
                if 0 <= sx < w and 0 <= sy < h and open_[sy * w + sx]:
                    pushes.append((action, sy * w + sx, ny * w + nx))
            self.adj[i] = tuple(nbrs)
            self.push_tab[i] = tuple(pushes)
            for _action, _stand, dest in pushes:
                pull[dest].append(i)
        self.pull_tab: list[tuple[int, ...]] = [tuple(p) for p in pull]

        self._fields = self._target_fields()
        self.dead_cells = frozenset(
            i for i in self.floor
            if all(f[i] >= _INF for f in self._fields))
        self._h_cache: dict[tuple, int] = {}
        self._frozen_cache: dict[tuple, bool] = {}
        self._plan: Plan | None = None          # last answer, for incremental replan
        self._answers: dict[tuple, Plan | None] = {}   # state -> answer (or None)
        # BFS scratch: `_wd` holds distances, `_ws` stamps them with `_epoch`, so
        # a walk costs no allocation and no clearing.
        self._wd = [0] * n
        self._ws = [0] * n
        self._epoch = 0

    # ── index helpers ────────────────────────────────────────────────────────
    def idx(self, cell) -> int:
        return int(cell[1]) * self.width + int(cell[0])

    def cell(self, i: int) -> tuple[int, int]:
        return (i % self.width, i // self.width)

    def normalize(self, boxes) -> tuple:
        """Boxes (cells or indices) -> the canonical sorted index tuple."""
        return tuple(sorted(b if isinstance(b, int) else self.idx(b)
                            for b in boxes))

    # ── geometry ─────────────────────────────────────────────────────────────
    def _target_fields(self) -> list[list[int]]:
        """One field per target: cell -> minimum pushes to get a box from that
        cell onto the target, ignoring the other boxes.

        A reverse BFS over `pull_tab`: a box goes from ``c`` to ``c + d`` only if
        ``c + d`` (where it lands) and ``c - d`` (where the player must stand)
        are both floor, so the predecessors of a cell are exactly the cells that
        can be pushed onto it. Distances are in PUSHES and ignore the other boxes
        and the player's walking, which is what makes the derived heuristic a
        lower bound rather than an estimate.
        """
        fields = []
        for target in sorted(self.targets):
            dist = [_INF] * self.n
            dist[target] = 0
            q = deque([target])
            while q:
                cur = q.popleft()
                d1 = dist[cur] + 1
                for src in self.pull_tab[cur]:
                    if dist[src] > d1:
                        dist[src] = d1
                        q.append(src)
            fields.append(dist)
        return fields

    # ── dynamics ─────────────────────────────────────────────────────────────
    def is_goal(self, boxes) -> bool:
        targets = self.targets
        return all(b in targets for b in boxes)

    def step(self, player, boxes, action_id: int):
        """One player move, in CELLS. Returns ``(player, boxes, outcome)`` with
        outcome ``"blocked"`` / ``"moved"`` / ``"push"``. Mirrors
        ``Sokoban.step`` (minus the step counter and the undo branch)."""
        dx, dy = DELTAS[action_id]
        ahead = (player[0] + dx, player[1] + dy)
        boxes = tuple(boxes)
        if not self._is_open(ahead):
            return player, boxes, "blocked"
        if ahead in boxes:
            behind = (ahead[0] + dx, ahead[1] + dy)
            if not self._is_open(behind) or behind in boxes:
                return player, boxes, "blocked"
            moved = tuple(sorted([b for b in boxes if b != ahead] + [behind]))
            return ahead, moved, "push"
        return ahead, boxes, "moved"

    def _is_open(self, cell) -> bool:
        x, y = cell
        return (0 <= x < self.width and 0 <= y < self.height
                and bool(self._open[y * self.width + x]))

    # ── walk distances (boxes are static while the player walks) ─────────────
    def _walk(self, origin: int, boxes) -> None:
        """Flood the board from ``origin`` with ``boxes`` blocking, into the
        ``_wd``/``_ws`` scratch. Read the result with `_wdist` BEFORE the next
        `_walk` call -- the buffer is reused.

        This is the hot loop of the whole search (one flood per expansion), hence
        the epoch-stamped scratch arrays (no allocation, no clearing), the
        hand-rolled queue and the local rebinding. Stopping the flood early once
        the pushable cells are all labelled was measured SLOWER than finishing it
        (the per-cell membership test costs more than the tail of a 200-cell
        flood saves), so it floods the whole reachable region."""
        self._epoch += 1
        epoch, wd, ws, adj = self._epoch, self._wd, self._ws, self.adj
        if origin in boxes:
            return
        # `boxes` is a handful of ints: scanning the tuple beats building a set.
        blocked = boxes if len(boxes) < 8 else set(boxes)
        wd[origin] = 0
        ws[origin] = epoch
        q = [origin]
        head = 0
        while head < len(q):
            cur = q[head]
            head += 1
            d1 = wd[cur] + 1
            for nxt in adj[cur]:
                if ws[nxt] != epoch and nxt not in blocked:
                    ws[nxt] = epoch
                    wd[nxt] = d1
                    q.append(nxt)

    def _wdist(self, i: int) -> int | None:
        return self._wd[i] if self._ws[i] == self._epoch else None

    def walk_field(self, origin, boxes) -> dict:
        """cell -> walking moves from ``origin`` (public, dict form)."""
        boxes = self.normalize(boxes)
        self._walk(self.idx(origin), boxes)
        return {self.cell(i): self._wd[i] for i in self.floor
                if self._ws[i] == self._epoch}

    def _walk_moves(self, origin: int, goal: int, boxes) -> list[int] | None:
        """A shortest walk ``origin -> goal`` as action ids, or None if the goal
        is unreachable without moving a box. Deterministic (fixed action order)."""
        if origin == goal:
            return []
        self._walk(goal, boxes)                   # distances TO the goal
        if self._ws[origin] != self._epoch:
            return None
        wd, ws, epoch = self._wd, self._ws, self._epoch
        w = self.width
        moves, cur = [], origin
        while cur != goal:
            x, y = cur % w, cur // w
            for action in ACTION_IDS:
                dx, dy = DELTAS[action]
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w and 0 <= ny < self.height):
                    continue
                nxt = ny * w + nx
                if ws[nxt] == epoch and wd[nxt] == wd[cur] - 1:
                    moves.append(action)
                    cur = nxt
                    break
            else:                                 # pragma: no cover -- BFS invariant
                return None
        return moves

    # ── deadlock tests ───────────────────────────────────────────────────────
    def _frozen(self, boxes: tuple) -> bool:
        """Is some off-target box immovable on both axes? (freeze deadlock)"""
        hit = self._frozen_cache.get(boxes)
        if hit is not None:
            return hit
        box_set = set(boxes)
        w, hgt, open_, dead = self.width, self.height, self._open, self.dead_cells

        def side(i: int, dx: int, dy: int) -> int | None:
            x, y = i % w, i // w
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < hgt):
                return None
            j = ny * w + nx
            return j if open_[j] else None

        def blocked(i: int, axis: int, seen: frozenset) -> bool:
            """Can the box on ``i`` never move along ``axis`` (0 = x, 1 = y)?"""
            dx, dy = (1, 0) if axis == 0 else (0, 1)
            a = side(i, -dx, -dy)
            b = side(i, dx, dy)
            if a is None or b is None:
                return True                       # a wall on either side pins it
            if a in dead and b in dead:
                return True                       # nowhere useful to go on this axis
            for nb in (a, b):
                if nb in box_set:
                    if nb in seen or blocked(nb, 1 - axis, seen | {i}):
                        return True               # mutually blocking neighbours
            return False

        out = False
        for box in boxes:
            if box in self.targets:
                continue
            if blocked(box, 0, frozenset((box,))) and blocked(box, 1, frozenset((box,))):
                out = True
                break
        self._frozen_cache[boxes] = out
        return out

    def is_dead(self, boxes) -> bool:
        """A one-sided "this state is lost" test: cheap, never wrong about a
        winnable state (but it may miss a lost one -- the search settles those)."""
        boxes = self.normalize(boxes)
        if any(b in self.dead_cells for b in boxes):
            return True
        return self._frozen(boxes)

    # ── heuristic ────────────────────────────────────────────────────────────
    def _heuristic(self, boxes: tuple) -> int:
        """Min-cost box->target assignment over true push distances: a lower
        bound on the remaining PUSHES, hence on the remaining moves. ``_INF``
        means no assignment exists at all -- a dead state."""
        hit = self._h_cache.get(boxes)
        if hit is not None:
            return hit
        nb = len(boxes)
        cost = [[f[b] for f in self._fields] for b in boxes]
        # Bitmask DP over targets -- <= 5 boxes here, so a few hundred ops.
        best = [_INF] * (1 << nb)
        best[0] = 0
        for mask in range(1 << nb):
            cur = best[mask]
            if cur >= _INF:
                continue
            i = bin(mask).count("1")             # box i is assigned next
            if i == nb:
                continue
            row = cost[i]
            for j in range(nb):
                bit = 1 << j
                if mask & bit or row[j] >= _INF:
                    continue
                if cur + row[j] < best[mask | bit]:
                    best[mask | bit] = cur + row[j]
        out = best[(1 << nb) - 1]
        self._h_cache[boxes] = out
        return out

    # ── the search ───────────────────────────────────────────────────────────
    def _search(self, player: int, boxes: tuple, weight: float,
                max_expansions: int) -> Plan | None:
        """Weighted push-level A*. ``weight == 1`` is exact. Returns None when the
        state is provably unwinnable OR the expansion cap was hit (the caller
        escalates; a caller out of attempts treats it as "cannot help")."""
        h0 = self._heuristic(boxes)
        if h0 >= _INF:
            return None
        counter = itertools.count()
        start = (player, boxes)
        heap = [(h0 * weight, 0, next(counter), start)]
        best_g = {start: 0}
        came: dict[tuple, tuple] = {}
        goal_state = None
        expansions = 0
        targets, dead = self.targets, self.dead_cells
        push_tab, heuristic = self.push_tab, self._heuristic
        while heap:
            _f, g, _c, cur = heapq.heappop(heap)
            if g > best_g.get(cur, g):
                continue                          # stale heap entry
            cur_player, cur_boxes = cur
            if all(b in targets for b in cur_boxes):
                goal_state = cur
                break
            expansions += 1
            if expansions > max_expansions:
                return None
            self._walk(cur_player, cur_boxes)
            wd, ws, epoch = self._wd, self._ws, self._epoch
            box_set = set(cur_boxes)
            for box in cur_boxes:
                for action, stand, dest in push_tab[box]:
                    if ws[stand] != epoch or stand in box_set:
                        continue                  # cannot get behind the box
                    if dest in box_set or dest in dead:
                        continue
                    nxt_boxes = tuple(sorted(
                        [b for b in cur_boxes if b != box] + [dest]))
                    ng = g + wd[stand] + 1
                    nxt = (box, nxt_boxes)        # player ends where the box was
                    if ng >= best_g.get(nxt, _INF):
                        continue
                    if self._frozen(nxt_boxes):
                        continue
                    h = heuristic(nxt_boxes)
                    if h >= _INF:
                        continue
                    best_g[nxt] = ng
                    came[nxt] = (cur, (stand, action, box, dest, nxt_boxes))
                    heapq.heappush(heap,
                                   (ng + h * weight, ng, next(counter), nxt))
        if goal_state is None:
            return None

        # Rebuild the push list back-to-front, filling in each suffix cost.
        raw: list[tuple] = []
        cur = goal_state
        while cur != start:
            prev, rec = came[cur]
            raw.append(rec)
            cur = prev
        raw.reverse()
        pushes: list[Push] = []
        for stand, action, box, dest, boxes_after in reversed(raw):
            if pushes:
                # walk from where this push leaves the player (`box`) to the NEXT
                # push's stand cell, over the board as it is AFTER this push
                nxt = pushes[0]
                self._walk(nxt.stand, boxes_after)
                walk = self._wdist(box)
                suffix = 1 + walk + nxt.suffix_cost
            else:
                suffix = 1
            pushes.insert(0, Push(stand, action, box, dest, boxes_after, suffix))
        self._walk(pushes[0].stand, boxes)
        first_walk = self._wdist(player)
        if first_walk is None:                    # pragma: no cover -- start was reachable
            first_walk = 0
        return Plan(tuple(pushes), first_walk + pushes[0].suffix_cost,
                    exact=weight == 1.0)

    # ── public planning API ──────────────────────────────────────────────────
    def _cost_of(self, pushes: tuple[Push, ...], player: int,
                 boxes: tuple) -> int | None:
        """Cost in moves of executing ``pushes`` from ``(player, boxes)``, or
        None if that skeleton no longer applies to this board."""
        if not pushes:
            return 0 if self.is_goal(boxes) else None
        head = pushes[0]
        if head.box not in boxes:
            return None
        if head.boxes_after != tuple(sorted(
                [b for b in boxes if b != head.box] + [head.dest])):
            return None                           # a different board than planned
        self._walk(head.stand, boxes)
        walk = self._wdist(player)
        if walk is None:
            return None
        return walk + head.suffix_cost

    def plan(self, player, boxes) -> Plan | None:
        """An optimal plan from ``(player, boxes)``, or None if no plan is found.

        Continues the previous answer where possible (see the module docstring):
        one move cannot reduce the distance-to-win by more than one, so a carried
        plan that re-costs to exactly ``previous_cost - 1`` is provably still as
        good as the plan it came from, and no search runs.

        Answers (including "no plan": a dead state stays dead) are memoised per
        state, so the generator's recovery scan over the undo history re-asks
        about the same board for free."""
        player = player if isinstance(player, int) else self.idx(player)
        boxes = self.normalize(boxes)
        state = (player, boxes)
        if state in self._answers:
            return self._answers[state]
        prev = self._plan
        if prev is not None:
            for pushes in (prev.pushes, prev.pushes[1:]):
                cost = self._cost_of(pushes, player, boxes)
                if cost is not None and cost == prev.cost - 1:
                    found = Plan(pushes, cost, prev.exact)
                    self._remember(state, found)
                    return found
        found = None
        if self.is_goal(boxes):
            found = Plan((), 0)
        elif not self.is_dead(boxes):
            for weight, cap in WEIGHTS:
                found = self._search(player, boxes, weight, cap)
                if found is not None:
                    break
        self._remember(state, found)
        return found

    def _remember(self, state: tuple, found: Plan | None) -> None:
        """Memoise an answer; keep it as the anchor for the next incremental
        re-plan when it is a real plan (a probe that found nothing must not
        clobber the anchor -- the caller is usually still walking a live plan)."""
        if len(self._answers) > 50_000:           # unbounded-growth guard
            self._answers.clear()
        self._answers[state] = found
        if found is not None:
            self._plan = found

    def moves(self, player, boxes) -> list[int]:
        """The plan as a flat action-id list ([] = already won, or no plan)."""
        player = player if isinstance(player, int) else self.idx(player)
        boxes = self.normalize(boxes)
        found = self.plan(player, boxes)
        if found is None:
            return []
        out: list[int] = []
        cur_player, cur_boxes = player, boxes
        for push in found.pushes:
            walk = self._walk_moves(cur_player, push.stand, cur_boxes)
            if walk is None:                      # pragma: no cover -- plan invariant
                return []
            out.extend(walk)
            out.append(push.action)
            cur_player, cur_boxes = push.box, push.boxes_after
        return out

    def optimal_moves(self, player, boxes) -> list[int]:
        """Every first move that keeps the win exactly one move closer.

        The plan's next push is fixed, so the tie set is every first step of a
        SHORTEST walk to that push's stand cell -- which is where the bulk of
        Sokoban's co-optimality lives (open floor has many equally short routes).
        Alternative push ORDERS of equal total cost are not enumerated, so this is
        a sound subset: everything returned really is as good as the plan."""
        player = player if isinstance(player, int) else self.idx(player)
        boxes = self.normalize(boxes)
        found = self.plan(player, boxes)
        if found is None or not found.pushes:
            return []
        head = found.pushes[0]
        if player == head.stand:
            return [head.action]
        self._walk(head.stand, boxes)
        here = self._wdist(player)
        if here is None:                          # pragma: no cover -- plan invariant
            return []
        wd, ws, epoch, w = self._wd, self._ws, self._epoch, self.width
        x, y = player % w, player // w
        out = []
        for action in ACTION_IDS:
            dx, dy = DELTAS[action]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < self.height):
                continue
            nxt = ny * w + nx
            if ws[nxt] == epoch and wd[nxt] == here - 1:
                out.append(action)
        return out
