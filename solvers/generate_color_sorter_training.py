"""Generate Phase-1 training data for color_sorter (games/color_sorter/color_sorter.py).

color_sorter: one or two FIXED axes split a 64x64 board into 2 or 4 regions;
coloured tokens sit mirrored across every axis; a cursor rides the axes and
ACTION5 swaps the token pair / group straddling the axis at the cursor's current
station.  Win when every region is one colour and the region colours are all
distinct.

The expert reads only the LIVE, fully-observable board -- each token's colour and
grid position and the cursor (mode / station index) -- exactly what a viewer of
the frame sees; nothing hidden (no reverse-scramble list, no region-target
table).  It then:

  1. builds a min-SWAP distance field over the swap orbit (BFS from the solved
     colourings, on a colour-canonicalised grid so it is computed once per
     level layout and cached for the whole run), and
  2. A*s the joint (colour-grid, cursor) state with that field as an admissible
     heuristic to a minimum-ACTION plan.

``optimal_set_from`` returns the co-optimal first-action SET from the same
search, so stochastic-optimal sampling stays on an optimal trajectory.  Every
swap is an involution and the game has no lose state, so any exploratory detour
is recoverable by replanning -> supports_recovery.

Usage (from the repo root):
    python solvers/generate_color_sorter_training.py --episodes 1000 \
        --out data/training_multi_level/color_sorter
"""
from __future__ import annotations

import heapq
import sys
from collections import deque
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.color_sorter.color_sorter import ColorSorter          # noqa: E402
from solvers.base_solver import BaseSolver                        # noqa: E402

_ACTS = (1, 2, 3, 4, 5)


# ── layout + swap model (all derived from observable board structure) ───────
class _Layout:
    __slots__ = ("nx", "ny", "axc", "axr", "region_of", "regions",
                 "nV", "nH", "key")

    def __init__(self, game) -> None:
        xcols, yrows = list(game._xcols), list(game._yrows)
        self.nx, self.ny = len(xcols), len(yrows)
        self.axc, self.axr = game._axc, game._axr
        self.region_of: list[tuple[int, int]] = []
        for i in range(self.nx):
            for j in range(self.ny):
                kx = 0 if self.axc is None else (-1 if xcols[i] < self.axc else 1)
                ky = 0 if self.axr is None else (-1 if yrows[j] < self.axr else 1)
                self.region_of.append((kx, ky))
        self.regions = sorted(set(self.region_of))
        self.nV = self.ny if self.axc is not None else 0   # vertical-axis stations
        self.nH = self.nx if self.axr is not None else 0   # horizontal-axis stations
        self.key = (tuple(xcols), tuple(yrows), self.axc, self.axr)

    def swap(self, grid: tuple, mode: str, idx: int) -> tuple:
        """Mirror-swap the token pairs straddling the axis at station ``idx`` --
        the exact effect of ColorSorter._apply_swap."""
        a = list(grid)
        ny = self.ny
        if mode == "V":
            j = idx
            for i in range(self.nx // 2):
                i2 = self.nx - 1 - i
                a[i * ny + j], a[i2 * ny + j] = a[i2 * ny + j], a[i * ny + j]
        else:
            i = idx
            for j in range(ny // 2):
                j2 = ny - 1 - j
                a[i * ny + j], a[i * ny + j2] = a[i * ny + j2], a[i * ny + j]
        return tuple(a)

    def goal_grids(self, n: int) -> list[tuple]:
        """Every monochrome-per-region colouring using colours ``0..n-1``."""
        out = []
        for perm in permutations(range(n)):
            cmap = dict(zip(self.regions, perm))
            out.append(tuple(cmap[r] for r in self.region_of))
        return out


_FIELD_CACHE: dict = {}


def _dist_field(layout: _Layout) -> dict:
    """grid -> min swaps to a solved colouring, over the colour-canonical orbit.

    Keyed on the layout only (colours are canonicalised to 0..n-1), so it is
    built once per level and reused for every seed of the run."""
    f = _FIELD_CACHE.get(layout.key)
    if f is not None:
        return f
    n = len(layout.regions)
    dist: dict = {}
    q: deque = deque()
    for g in layout.goal_grids(n):
        if g not in dist:
            dist[g] = 0
            q.append(g)
    vsts = range(layout.nV)
    hsts = range(layout.nH)
    while q:
        g = q.popleft()
        d = dist[g] + 1
        for idx in vsts:
            ng = layout.swap(g, "V", idx)
            if ng not in dist:
                dist[ng] = d
                q.append(ng)
        for idx in hsts:
            ng = layout.swap(g, "H", idx)
            if ng not in dist:
                dist[ng] = d
                q.append(ng)
    _FIELD_CACHE[layout.key] = dist
    return dist


def _canon(grid: tuple) -> tuple:
    """Relabel colours to their rank (0..n-1) among the distinct colours present."""
    order = sorted(set(grid))
    rank = {c: i for i, c in enumerate(order)}
    return tuple(rank[c] for c in grid)


# ── joint (grid, cursor) transitions -- a faithful mirror of ColorSorter.step ─
def _succ(layout: _Layout, state: tuple, a: int):
    grid, mode, vi, hi = state
    if a == 1:
        return None if not layout.nV else (grid, "V", max(0, vi - 1), hi)
    if a == 2:
        return None if not layout.nV else (grid, "V", min(layout.nV - 1, vi + 1), hi)
    if a == 3:
        return None if not layout.nH else (grid, "H", vi, max(0, hi - 1))
    if a == 4:
        return None if not layout.nH else (grid, "H", vi, min(layout.nH - 1, hi + 1))
    # a == 5: swap at the current station
    if mode == "V":
        if not layout.nV:
            return None
        return (layout.swap(grid, "V", vi), mode, vi, hi)
    if not layout.nH:
        return None
    return (layout.swap(grid, "H", hi), mode, vi, hi)


def _optcost(layout: _Layout, D: dict, s: tuple, limit: int) -> int:
    """Min actions from ``s`` to a solved state, or ``limit + 1`` if it exceeds
    ``limit``.  A* with the swap-distance heuristic; explores only the optimal
    cone."""
    if D.get(s[0], 1) == 0:
        return 0
    h = D.get(s[0])
    if h is None or h > limit:
        return limit + 1
    pq = [(h, 0, s)]
    best = {s: 0}
    while pq:
        f, g, cur = heapq.heappop(pq)
        if f > limit:
            return limit + 1
        if g > best.get(cur, 1 << 30):
            continue
        if D[cur[0]] == 0:
            return g
        for a in _ACTS:
            ns = _succ(layout, cur, a)
            if ns is None:
                continue
            hn = D.get(ns[0])
            if hn is None:
                continue
            ng = g + 1
            if ng + hn > limit or ng >= best.get(ns, 1 << 30):
                continue
            best[ns] = ng
            heapq.heappush(pq, (ng + hn, ng, ns))
    return limit + 1


_PLAN_CACHE: dict = {}


def _plan(layout: _Layout, D: dict, state: tuple):
    """(optimal action plan, co-optimal first-action set) from ``state``."""
    ck = (layout.key, state)
    hit = _PLAN_CACHE.get(ck)
    if hit is not None:
        return hit
    if len(_PLAN_CACHE) > 200_000:
        _PLAN_CACHE.clear()

    grid0 = state[0]
    if D.get(grid0, 1) == 0 or grid0 not in D:
        _PLAN_CACHE[ck] = ([], set())
        return [], set()

    pq = [(D[grid0], 0, state)]
    best_g = {state: 0}
    parent: dict = {state: (None, None)}
    goal = None
    while pq:
        f, g, s = heapq.heappop(pq)
        if g > best_g.get(s, 1 << 30):
            continue
        if D[s[0]] == 0:
            goal = s
            break
        for a in _ACTS:
            ns = _succ(layout, s, a)
            if ns is None:
                continue
            hn = D.get(ns[0])
            if hn is None:
                continue
            ng = g + 1
            if ng < best_g.get(ns, 1 << 30):
                best_g[ns] = ng
                parent[ns] = (s, a)
                heapq.heappush(pq, (ng + hn, ng, ns))

    if goal is None:
        _PLAN_CACHE[ck] = ([], set())
        return [], set()

    plan: list[int] = []
    s = goal
    while parent[s][0] is not None:
        ps, pa = parent[s]
        plan.append(pa)
        s = ps
    plan.reverse()

    C = best_g[goal]
    opt_first: set[int] = set()
    for a in _ACTS:
        ns = _succ(layout, state, a)
        if ns is None:
            continue
        if _optcost(layout, D, ns, C - 1) == C - 1:
            opt_first.add(a)
    if not opt_first and plan:
        opt_first.add(plan[0])

    _PLAN_CACHE[ck] = (plan, opt_first)
    return plan, opt_first


def _read(game):
    """Live board -> (_Layout, distance field, joint state).  Returns None on the
    (generator-invariant-violating) case where region count != colour count."""
    layout = _Layout(game)
    grid = tuple(game._find(x, y)._cs_color
                 for x in game._xcols for y in game._yrows)
    if len(set(grid)) != len(layout.regions):
        return None
    D = _dist_field(layout)
    state = (_canon(grid), game._mode, game._vi, game._hi)
    return layout, D, state


class ColorSorterSolver(BaseSolver):
    game_id = "color_sorter"
    #: Every swap is its own inverse and the game has no lose state, so
    #: solve_from re-plans correctly from any perturbed board -> exploration
    #: prefix + bursts are safe.
    supports_recovery = True

    def make_game(self, seed: int):
        return ColorSorter(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4, 5]

    def solve_from(self, game, level_idx: int, seed: int):
        got = _read(game)
        if got is None:
            return []
        layout, D, state = got
        plan, _ = _plan(layout, D, state)
        return plan

    def optimal_set_from(self, game, level_idx: int, seed: int):
        got = _read(game)
        if got is None:
            return []
        layout, D, state = got
        _, opt_first = _plan(layout, D, state)
        return sorted(opt_first)


if __name__ == "__main__":
    sys.exit(ColorSorterSolver.main())
