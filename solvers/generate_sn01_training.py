"""Generate Phase-1 training data for the sn01 "Snake Collect" game.

sn01 (games/sn01/sn01.py): a 12x12 snake. ACTION1..4 move the head up/down/
left/right; eating a yellow food grows the snake by one segment; the level is won
when every food is eaten. A move into a wall or off the board is a NO-OP (the
action is consumed, nothing changes); a move into the snake's own body -- except
into the cell the tail is about to vacate -- is GAME_OVER.

Expert
------
The objective is a shortest-tour problem with a growing obstacle: visit every
remaining food, never self-intersecting. ``solve_from`` runs A* over the exact
state ``(body, remaining-food mask)`` -- ``body`` is the full segment tuple, head
last, exactly as the engine's ``_body`` deque -- with the transition mirroring
``Sn01.step`` (tail-vacation rule and grow-on-eat included).

The heuristic is the exact open-tour cost over the remaining foods computed on
the WALL-ONLY grid (the snake body ignored): a Held-Karp table
``tsp[mask][i]`` = cheapest route that starts at food ``i`` and covers ``mask``,
combined with a BFS distance field per food, so
``h = min_i d(head, food_i) + tsp[mask][i]``. Ignoring the body can only
UNDER-estimate (the body adds constraints, never removes them), so h is
admissible -- and on a 12x12 board with a <=6-segment snake it is nearly exact,
which is what keeps A* to a handful of expansions per call. Plain BFS over the
same state space is not viable here: level 4's optimum is 19 moves, and the
19-ball of snake configurations is in the millions.

Recovery (supports_recovery = True, recovery_mode = "replan")
------------------------------------------------------------
``solve_from`` reads the LIVE game -- body segments from ``game._body``, the
remaining foods from ``game._foods``, the walls from the level's sprites -- so it
re-plans from any reachable state: after the exploratory prefix, mid-burst, or
after an off-plan food has been eaten out of order (which merely shrinks the
mask and lengthens the snake). States it cannot win from -- the snake boxed in by
its own body -- return ``[]``, which the base rolls back (burst UNDO) or recovers
from with a RESET; a burst that walks into the body is GAME_OVER, likewise
handled by the base. ``optimal_set_from`` returns the full co-optimal tie set
(every first move that starts a shortest tour), so stochastic-optimal play draws
a genuinely different route through the foods each episode.

Usage (run from the repo root):
    python solvers/generate_sn01_training.py --episodes 1000 \
        --out data/training_multi_level/sn01
"""

from __future__ import annotations

import heapq
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.sn01.sn01 import Sn01                            # noqa: E402
from solvers.base_solver import BaseSolver                  # noqa: E402

#: (action id, dx, dy) in the order ``Sn01.step`` tests them.
_MOVES = ((1, 0, -1), (2, 0, 1), (3, -1, 0), (4, 1, 0))

_INF = float("inf")


class _Ctx:
    """Everything about one level's static geometry + the search caches.

    Built per ``solve_from``/``optimal_set_from`` call from the LIVE board but
    memoised on the solver by ``(grid, walls, remaining foods)``, so the BFS
    fields, the Held-Karp table and previously solved states are reused across
    the steps of a level (the food set only ever shrinks, and each distinct set
    gets its own context)."""

    def __init__(self, gw: int, gh: int, walls: frozenset,
                 foods: tuple[tuple[int, int], ...]) -> None:
        self.gw, self.gh, self.walls = gw, gh, walls
        self.foods = foods                       # index i <-> bit i of the mask
        self.full_mask = (1 << len(foods)) - 1
        self._fields = {f: self._bfs_field(f) for f in foods}
        self._tsp = self._held_karp()
        self._hmemo: dict = {}
        self.cache: dict = {}                    # (body, mask) -> (cost, plan)

    # ── wall-only geometry ───────────────────────────────────────────────────
    def free(self, x: int, y: int) -> bool:
        return (0 <= x < self.gw and 0 <= y < self.gh
                and (x, y) not in self.walls)

    def _bfs_field(self, src: tuple[int, int]) -> dict:
        """Shortest-path distance from ``src`` to every free cell (walls only)."""
        dist = {src: 0}
        q = deque([src])
        while q:
            x, y = q.popleft()
            d = dist[(x, y)] + 1
            for _aid, dx, dy in _MOVES:
                nxt = (x + dx, y + dy)
                if nxt not in dist and self.free(*nxt):
                    dist[nxt] = d
                    q.append(nxt)
        return dist

    def _held_karp(self) -> list:
        """``tsp[mask][i]``: cheapest wall-only route starting at food ``i`` that
        visits every food in ``mask`` (``i`` in ``mask``). Masks are walked in
        increasing value order, which is enough since ``mask ^ (1<<i) < mask``."""
        n = len(self.foods)
        tsp = [[_INF] * n for _ in range(1 << n)]
        for i in range(n):
            tsp[1 << i][i] = 0
        for mask in range(1 << n):
            for i in range(n):
                if not mask >> i & 1 or mask == 1 << i:
                    continue
                rest = mask ^ (1 << i)
                best = _INF
                fi = self._fields[self.foods[i]]
                for j in range(n):
                    if not rest >> j & 1:
                        continue
                    d = fi.get(self.foods[j], _INF)
                    if d == _INF or tsp[rest][j] == _INF:
                        continue
                    cand = d + tsp[rest][j]
                    if cand < best:
                        best = cand
                tsp[mask][i] = best
        return tsp

    # ── admissible heuristic ─────────────────────────────────────────────────
    def h(self, head: tuple[int, int], mask: int) -> float:
        """Exact open-tour cost from ``head`` over the foods in ``mask``, on the
        wall-only grid. Admissible: the snake body can only lengthen a route."""
        if mask == 0:
            return 0
        memo = self._hmemo
        key = (head, mask)
        got = memo.get(key)
        if got is not None:
            return got
        best = _INF
        for i, food in enumerate(self.foods):
            if not mask >> i & 1:
                continue
            d = self._fields[food].get(head, _INF)
            t = self._tsp[mask][i]
            if d == _INF or t == _INF:
                continue
            cand = d + t
            if cand < best:
                best = cand
        memo[key] = best
        return best

    # ── the engine transition, exactly (see ``Sn01.step``) ───────────────────
    def successors(self, body: tuple, mask: int):
        """Yield ``(action_id, new_body, new_mask)`` for every move that neither
        no-ops nor kills. A move into a wall / off-board is dropped (a wasted
        action is never optimal); a move into a body segment other than the
        vacating tail is a loss and dropped too."""
        hx, hy = body[-1]
        tail = body[0]
        for aid, dx, dy in _MOVES:
            nxt = (hx + dx, hy + dy)
            if not self.free(*nxt):
                continue
            if nxt in body and nxt != tail:
                continue
            bit = 0
            for i, food in enumerate(self.foods):
                if food == nxt and mask >> i & 1:
                    bit = 1 << i
                    break
            if bit:                              # grow: the tail stays put
                yield aid, body + (nxt,), mask & ~bit
            else:
                yield aid, body[1:] + (nxt,), mask


class Sn01Solver(BaseSolver):
    game_id = "sn01"
    #: solve_from re-plans from the live snake, and the only unrecoverable states
    #: (self-trapped / dead) are handled by the base's burst-undo + RESET.
    supports_recovery = True
    recovery_mode = "replan"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._ctxs: dict = {}

    def make_game(self, seed: int):
        # sn01 takes no seed (its five levels are fixed); cross-episode variety
        # comes from stochastic-optimal routing + the exploration prefix.
        return Sn01()

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    # ── live state ───────────────────────────────────────────────────────────
    def _read(self, game):
        """``(ctx, body, mask)`` for the LIVE board: body segments (head last),
        the remaining foods, and the wall layout, all read off the game."""
        level = game.current_level
        gw, gh = level.grid_size
        walls = frozenset((s.x, s.y) for s in level.get_sprites_by_tag("wall"))
        body = tuple((int(x), int(y)) for x, y in game._body)
        foods = tuple(sorted((int(f.x), int(f.y)) for f in game._foods))
        key = (gw, gh, walls, foods)
        ctx = self._ctxs.get(key)
        if ctx is None:
            ctx = self._ctxs[key] = _Ctx(gw, gh, walls, foods)
        return ctx, body, ctx.full_mask

    # ── A* over (body, remaining-food mask) ──────────────────────────────────
    def _astar(self, ctx: _Ctx, body: tuple, mask: int, limit: int | None = None):
        """Cheapest eat-every-remaining-food plan from ``(body, mask)``.

        Returns ``(cost, plan)`` -- ``plan`` a list of action ids -- or
        ``(None, None)`` when there is none (self-trapped) or when ``limit`` is
        set and the optimum exceeds it. A cost found under a ``limit`` is still
        the true optimum (A* with an admissible heuristic), so it is cached."""
        if mask == 0:
            return 0, []
        hit = ctx.cache.get((body, mask))
        if hit is not None:
            cost, plan = hit
            if limit is not None and cost > limit:
                return None, None
            return cost, list(plan)

        h0 = ctx.h(body[-1], mask)
        if h0 == _INF or (limit is not None and h0 > limit):
            return None, None

        start = (body, mask)
        best_g = {start: 0}
        parent: dict = {start: None}
        pq = [(h0, 0, 0, start)]
        tick = 0
        while pq:
            _f, g, _t, state = heapq.heappop(pq)
            if g > best_g.get(state, _INF):
                continue                         # stale queue entry
            cbody, cmask = state
            if cmask == 0:
                plan = []
                node = state
                while parent[node] is not None:
                    prev, aid = parent[node]
                    plan.append(aid)
                    node = prev
                plan.reverse()
                ctx.cache[(body, mask)] = (g, tuple(plan))
                return g, plan
            for aid, nbody, nmask in ctx.successors(cbody, cmask):
                ns = (nbody, nmask)
                ng = g + 1
                if ng >= best_g.get(ns, _INF):
                    continue
                nh = ctx.h(nbody[-1], nmask)
                if nh == _INF:
                    continue
                nf = ng + nh
                if limit is not None and nf > limit:
                    continue
                best_g[ns] = ng
                parent[ns] = (state, aid)
                tick += 1
                heapq.heappush(pq, (nf, ng, tick, ns))
        return None, None

    # ── BaseSolver hooks ─────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int):
        """Optimal plan from the game's CURRENT state (see module docstring).
        Empty list == no way to finish from here (self-trapped)."""
        ctx, body, mask = self._read(game)
        _cost, plan = self._astar(ctx, body, mask)
        return plan or []

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-optimal next move at the live state: the moves ``a`` whose
        successor still admits a ``cost - 1`` tour. Each candidate is checked with
        its own bounded A* (bounded at ``cost - 1``, so a sub-optimal branch is
        abandoned as soon as its f-value exceeds that), which is cheap because the
        tour heuristic is near-exact. The canonical ``solve_from`` head is always
        in the set, and the shared per-context cache means the first search here is
        the one ``solve_from`` just did."""
        ctx, body, mask = self._read(game)
        cost, _plan = self._astar(ctx, body, mask)
        if cost is None or cost == 0:
            return None
        opts = []
        for aid, nbody, nmask in ctx.successors(body, mask):
            if nmask == 0:
                if cost == 1:
                    opts.append(aid)
                continue
            c2, _p2 = self._astar(ctx, nbody, nmask, limit=cost - 1)
            if c2 == cost - 1:
                opts.append(aid)
        return opts or None


if __name__ == "__main__":
    sys.exit(Sn01Solver.main())
