"""Shared expert + data generator for the tw01..tw06 family (The Witness panels).

The six ``tw*`` games are one mechanic wearing six constraint hats. Every one of
them draws a LINE along the edges of a small node grid: ACTION1..4 push the line
one node up/down/left/right, ACTION5 submits it. The submit either wins the level
(the line ends on the end node *and* the panel's constraint holds) or wipes the
line back to the start node. What differs between the games is only the predicate:

  tw01  PathDots    the line must pass through every marked node
  tw02  ColorSplit  the regions the line cuts must be colour-pure
  tw03  ShapeFill   the polyominoes in a region must tile it exactly
  tw04  SymDraw     TWO lines, the second a mirror of the first; both must land
  tw05  StarPair    every colour present in a region must appear exactly twice
  tw06  TriCount    a cell with N triangles must have N of its edges on the line

So this module owns the whole family: the state model (`Panel`), the expert, the
per-seed puzzle generator, and the `BaseSolver` glue. A per-game generator is a
dozen lines: the constraint reader and the random-level recipe.

The expert
----------
The state IS the line: a tuple of nodes. Only three things can happen to it --
push a node on the tail, pop the tail (walking back down the line), or submit,
which either wins or resets it to ``(start,)``. That makes the distance to a
WIN exactly computable against a set of target solutions ``S``::

    cost(P) = 1                                     if P already wins
    cost(P) = min over S of  (|P|-k) + (|S|-k) + 1  , k = |common prefix(P,S)|
              and also  1 + cost((start,))          , the ACTION5 wipe

-- because reaching S from P must walk back to their common prefix and then out
along S, and the wipe is a one-action teleport to ``(start,)``. So the expert
needs no search at plan time: the solutions are enumerated ONCE per level, folded
into a prefix table (`prefix_minima`), and `_cost` becomes a handful of dict
lookups. `optimal_set_from` then falls out for free -- an action is optimal iff
it drops the cost by one -- which is the real tie set, not one canonical route,
and it makes `solve_from` fully state-driven. Recovery from an exploration prefix
or a perturbation burst is the SAME code path: whatever the line looks like,
`_cost` measures it. Nothing about the plan is cached against the level's initial
state.

The enumeration (`enumerate_solutions`) is a randomised DFS over simple paths
with a reachability prune. On the 3x3/4x4 panels it exhausts the space outright,
and the expert is then EXACTLY optimal -- verified against a brute-force BFS over
the real state space, from every reachable line, not just the opening one. On 5x5
it is budget-capped, so the plan is optimal with respect to the solutions it
found (and the ACTION5 wipe is always in the candidate set, so recovery never
gets stuck).

Per-seed puzzles
----------------
The shipped games are static -- no ``seed`` reaches the level layout, and
``levels/tw0*_levels.json`` does not exist, so every seed replays the same five
hand-written panels. That makes a 2000-episode corpus 2000 copies of one puzzle.
So the generator SYNTHESISES a fresh panel per (seed, level) and feeds it in
through the games' own JSON-level hook (`_load_json_levels`), leaving the game
files untouched. Puzzles are built BACKWARDS from a random line -- draw a random
simple path, then derive constraints that the path satisfies (dots on it, region
colours from the regions it cuts, triangle counts from the edges it uses, ...) --
so every generated panel is solvable by construction and comes with a known
solution seeded into the expert. ``--no-augment`` falls back to the shipped
panels.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from solvers.base_solver import BaseSolver  # noqa: E402
from utils.witness_grid import (  # noqa: E402
    SQUARE_A, SQUARE_B, SQUARE_C, WitnessGrid)

INF = float("inf")

#: action id -> (dcol, drow), matching every tw game's ``step()``.
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
SUBMIT = 5

#: Cap on solutions kept per panel. The expert wants EVERY solution, not the
#: short ones: a long solution that shares more prefix with a perturbed line is
#: genuinely cheaper to reach than a short one that shares none, so filtering by
#: length makes `WitnessSolver._cost` over-estimate on exactly the states
#: recovery has to handle. The cap only ever bites on 5x5 panels.
MAX_SOLUTIONS = 20_000
#: Length slack used by the FIRST enumeration pass, which is looking for a short
#: solution rather than for coverage. See `enumerate_solutions`.
LENGTH_SLACK = 4
#: DFS node budget for one level's enumeration. 3x3/4x4 panels are exhausted
#: well inside it (the whole space is 2k/150k nodes); 5x5 panels are truncated.
DFS_BUDGET = 400_000


_GRIDS = {}


def grid_for(cols, rows):
    """A cached ``WitnessGrid`` -- the games' own geometry/region helper."""
    grid = _GRIDS.get((cols, rows))
    if grid is None:
        grid = _GRIDS[(cols, rows)] = WitnessGrid(cols, rows)
    return grid


def _edge(a, b):
    return (a, b) if a <= b else (b, a)


# ---------------------------------------------------------------------------
# Panel: the state model shared by all six games
# ---------------------------------------------------------------------------
class Panel:
    """One level's geometry + constraint, and the exact line-state transition.

    ``step`` is a faithful re-implementation of the games' ``step()``, including
    the quirks the expert has to plan around: a move onto the previous node POPS
    (walks the line back) rather than pushing, a move onto any other node already
    on the line is silently ignored, and ACTION5 on a non-winning line wipes it
    back to the start. Getting these exactly right is what makes
    `optimal_set_from` a real tie set rather than a guess.
    """

    def __init__(self, cols, rows, breakpoints=()):
        self.cols = cols
        self.rows = rows
        self.breakpoints = frozenset(_edge(tuple(a), tuple(b)) for a, b in breakpoints)

    # -- geometry ---------------------------------------------------------
    def in_bounds(self, node):
        return 0 <= node[0] <= self.cols and 0 <= node[1] <= self.rows

    def passable(self, a, b):
        return self.in_bounds(b) and _edge(a, b) not in self.breakpoints

    def max_nodes(self):
        return (self.cols + 1) * (self.rows + 1)

    # -- subclass API -----------------------------------------------------
    def roots(self):
        """Node(s) a line can be rooted at."""
        raise NotImplementedError

    def reset_root(self):
        """Where an ACTION5 wipe leaves the line."""
        raise NotImplementedError

    def step(self, path, aid):
        raise NotImplementedError

    def extensions(self, path):
        """``(action_id, longer_path)`` for every legal push. Enumeration only."""
        raise NotImplementedError

    def is_solution(self, path):
        raise NotImplementedError

    def goal_node(self, path):
        """The node the line must finish on (for the reachability prune)."""
        raise NotImplementedError

    def signature(self):
        raise NotImplementedError


class SinglePathPanel(Panel):
    """tw01 / tw02 / tw03 / tw05 / tw06 -- one line, one constraint predicate."""

    def __init__(self, cols, rows, starts, end, breakpoints, check, sig):
        super().__init__(cols, rows, breakpoints)
        self.starts = [tuple(s) for s in starts]
        self.end = tuple(end)
        self._check = check
        self._sig = sig

    def roots(self):
        return list(self.starts)

    def reset_root(self):
        return self.starts[0]

    def goal_node(self, path):
        return self.end

    def _valid_move(self, a, b):
        return (self.in_bounds(b)
                and abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                and _edge(a, b) not in self.breakpoints)

    def step(self, path, aid):
        if aid == SUBMIT:
            return path if self.is_solution(path) else (self.starts[0],)
        dc, dr = DIRS[aid]
        cur = path[-1]
        tgt = (cur[0] + dc, cur[1] + dr)
        if not self._valid_move(cur, tgt):
            # Multi-start panels auto-select a start on the FIRST move: the game
            # scans ``starts`` in order for one that can move this way and
            # re-roots the line there. Single-start panels never take this path.
            if len(path) != 1 or len(self.starts) <= 1:
                return path
            for alt in self.starts:
                alt_tgt = (alt[0] + dc, alt[1] + dr)
                if self._valid_move(alt, alt_tgt):
                    path, cur, tgt = (alt,), alt, alt_tgt
                    break
            else:
                return path
        if len(path) >= 2 and tgt == path[-2]:
            return path[:-1]
        if tgt not in path:
            return path + (tgt,)
        return path

    def extensions(self, path):
        cur = path[-1]
        out = []
        for aid, (dc, dr) in DIRS.items():
            tgt = (cur[0] + dc, cur[1] + dr)
            if self._valid_move(cur, tgt) and tgt not in path:
                out.append((aid, path + (tgt,)))
        return out

    def is_solution(self, path):
        return path[-1] == self.end and self._check(path)

    def signature(self):
        return self._sig


class SymmetryPanel(Panel):
    """tw04 -- the player drives the blue line; the yellow one mirrors it.

    The yellow line is a pure function of the blue one, so the STATE is still
    just the blue path: ``yellow[i] = yellow_start + M(blue[i] - blue_start)``
    for the level's symmetry map M. A move is legal only if it is legal for
    both lines, neither revisits a node, and the two do not land on the same
    node in the same step.
    """

    _MIRROR = {"horizontal": (-1, 1), "vertical": (1, -1), "rotational": (-1, -1)}

    def __init__(self, cols, rows, symmetry, blue_start, blue_end,
                 yellow_start, yellow_end, blue_dots, yellow_dots,
                 breakpoints, sig):
        super().__init__(cols, rows, breakpoints)
        self.symmetry = symmetry
        self.blue_start = tuple(blue_start)
        self.blue_end = tuple(blue_end)
        self.yellow_start = tuple(yellow_start)
        self.yellow_end = tuple(yellow_end)
        self.blue_dots = [tuple(d) for d in blue_dots]
        self.yellow_dots = [tuple(d) for d in yellow_dots]
        self._sig = sig

    def mirror_delta(self, dc, dr):
        sc, sr = self._MIRROR.get(self.symmetry, (1, 1))
        return (dc * sc, dr * sr)

    def mate(self, node):
        d = self.mirror_delta(node[0] - self.blue_start[0],
                              node[1] - self.blue_start[1])
        return (self.yellow_start[0] + d[0], self.yellow_start[1] + d[1])

    def yellow_of(self, path):
        return tuple(self.mate(n) for n in path)

    def roots(self):
        return [self.blue_start]

    def reset_root(self):
        return self.blue_start

    def goal_node(self, path):
        return self.blue_end

    def _push(self, path, tgt):
        """The push half of ``step``: the pair-legality test, then append."""
        cur = path[-1]
        ycur, ytgt = self.mate(cur), self.mate(tgt)
        if not (self.in_bounds(tgt) and _edge(cur, tgt) not in self.breakpoints):
            return None
        if not (self.in_bounds(ytgt) and _edge(ycur, ytgt) not in self.breakpoints):
            return None
        if tgt in path or ytgt in self.yellow_of(path) or tgt == ytgt:
            return None
        return path + (tgt,)

    def step(self, path, aid):
        if aid == SUBMIT:
            return path if self.is_solution(path) else (self.blue_start,)
        dc, dr = DIRS[aid]
        cur = path[-1]
        tgt = (cur[0] + dc, cur[1] + dr)
        ycur, ytgt = self.mate(cur), self.mate(tgt)
        # Both lines must be able to make the move at all -- an out-of-bounds or
        # severed YELLOW edge blocks the blue line too, even for a walk-back.
        if not (self.in_bounds(tgt) and _edge(cur, tgt) not in self.breakpoints):
            return path
        if not (self.in_bounds(ytgt) and _edge(ycur, ytgt) not in self.breakpoints):
            return path
        if len(path) >= 2 and tgt == path[-2]:
            return path[:-1]
        pushed = self._push(path, tgt)
        return pushed if pushed is not None else path

    def extensions(self, path):
        cur = path[-1]
        out = []
        for aid, (dc, dr) in DIRS.items():
            nxt = self._push(path, (cur[0] + dc, cur[1] + dr))
            if nxt is not None:
                out.append((aid, nxt))
        return out

    def is_solution(self, path):
        if path[-1] != self.blue_end:
            return False
        yellow = self.yellow_of(path)
        if yellow[-1] != self.yellow_end:
            return False
        blue_set, yellow_set = set(path), set(yellow)
        return (all(d in blue_set for d in self.blue_dots)
                and all(d in yellow_set for d in self.yellow_dots))

    def signature(self):
        return self._sig


# ---------------------------------------------------------------------------
# Solution enumeration
# ---------------------------------------------------------------------------
def enumerate_solutions(panel, root, *, rng, limit=MAX_SOLUTIONS,
                        slack=LENGTH_SLACK, budget=DFS_BUDGET, known=()):
    """Every solution rooted at ``root`` the budget can reach, up to ``limit``.

    A randomised DFS over simple paths. Two prunes carry it: the line STOPS at
    the goal node (any continuation has consumed the goal and can never end on
    it), and the goal must stay reachable through nodes the line has not eaten
    yet. Together those exhaust the 3x3/4x4 panels well inside the budget, which
    is what makes the expert exactly optimal there; bigger panels are truncated,
    which costs optimality but not correctness -- every returned path is a real,
    verified solution.

    The search runs in TWO passes, because "find a short answer" and "find every
    answer" want opposite orderings and opposite prunes:

      1. length-capped, so the first thing found is a SHORT solution. This is
         what bounds how long the recorded trajectory is on a truncated panel.
      2. uncapped, for coverage: `WitnessSolver._cost` measures distance to the
         nearest solution, and a long solution sharing a long prefix with a
         perturbed line is often the nearest one.

    ``known`` seeds solutions found elsewhere (the line a synthesised puzzle was
    built from), so a budget-truncated search always has at least one target and
    pass 1 starts with a real length cap instead of the whole grid.
    """
    found, seen = [], set()
    for s in known:
        s = tuple(tuple(n) for n in s)
        if s and s[0] == root and s not in seen and panel.is_solution(s):
            found.append(s)
            seen.add(s)

    best = min((len(s) for s in found), default=None)
    cap = panel.max_nodes()
    budget = [budget]

    def reachable(path, goal):
        """Is ``goal`` still reachable from the line's tail through free nodes?"""
        cur = path[-1]
        if cur == goal:
            return True
        blocked = set(path)
        stack, seen_n = [cur], {cur}
        while stack:
            node = stack.pop()
            for dc, dr in DIRS.values():
                nxt = (node[0] + dc, node[1] + dr)
                if nxt in seen_n or nxt in blocked:
                    continue
                if not panel.passable(node, nxt):
                    continue
                if nxt == goal:
                    return True
                seen_n.add(nxt)
                stack.append(nxt)
        return False

    tighten = True          # pass 1 narrows `cap` as it finds shorter answers

    def dfs(path):
        nonlocal best, cap
        if budget[0] <= 0 or len(found) >= limit:
            return
        budget[0] -= 1
        goal = panel.goal_node(path)
        if path[-1] == goal:
            # Terminal: the goal node is consumed, so nothing longer can win.
            if panel.is_solution(path) and path not in seen:
                found.append(path)
                seen.add(path)
                if best is None or len(path) < best:
                    best = len(path)
                    if tighten:
                        cap = min(cap, best + slack)
            return
        dist = abs(path[-1][0] - goal[0]) + abs(path[-1][1] - goal[1])
        if len(path) + dist > cap:
            return
        if not reachable(path, goal):
            return
        children = [p for _, p in panel.extensions(path)]
        rng.shuffle(children)
        if len(children) > 1:
            # Warnsdorff order: step into the most CORNERED node first. The
            # region/tiling panels are only satisfied by long, space-filling
            # lines (the shipped 5x5 tw02 needs 27 of 36 nodes), and a line that
            # strands a free node can never be one -- so exploring the cramped
            # side first reaches real solutions orders of magnitude sooner. It
            # is an ordering only: nothing is pruned, so the search stays
            # exhaustive whenever the budget allows.
            children.sort(key=lambda p: sum(
                1 for dc, dr in DIRS.values()
                if (p[-1][0] + dc, p[-1][1] + dr) not in p
                and panel.passable(p[-1], (p[-1][0] + dc, p[-1][1] + dr))))
        for child in children:
            dfs(child)

    # Pass 1 -- length-capped, hunting a SHORT solution. With a `known` line the
    # cap is real from the first node; without one it only tightens once
    # something is found, which is still enough to stop a truncated search from
    # returning nothing but near-Hamiltonian lines.
    if best is not None:
        cap = min(cap, best + slack)
    reserve, budget[0] = budget[0] * 2 // 3, budget[0] // 3
    dfs((root,))

    # Pass 2 -- uncapped, hunting COVERAGE. Restart from the root with the full
    # grid available again; `seen` dedupes against pass 1.
    tighten = False
    cap = panel.max_nodes()
    budget[0] += reserve
    dfs((root,))

    return tuple(found[:limit])


def prefix_minima(solutions):
    """``{line prefix: shortest solution extending it}`` over a solution set.

    This is the whole expert, compressed. `WitnessSolver._cost` needs, for each
    prefix of the live line, the length of the shortest solution that keeps that
    prefix -- scanning the raw solution list for it is O(|solutions|) per query,
    which on an exhausted 4x4 panel means 8k comparisons six times a step. As a
    prefix table it is one dict lookup per prefix, so a cost query is O(|line|)
    however many solutions there are, and nothing has to be dropped to keep the
    lookup cheap.
    """
    out = {}
    for solution in solutions:
        length = len(solution)
        for j in range(1, length + 1):
            prefix = solution[:j]
            if out.get(prefix, 1 << 30) > length:
                out[prefix] = length
    return out


# ---------------------------------------------------------------------------
# Puzzle synthesis helpers (shared by the per-game recipes)
# ---------------------------------------------------------------------------
def random_line(cols, rows, start, end, rng, *, min_nodes, breakpoints=()):
    """A random simple path ``start -> end`` with at least ``min_nodes`` nodes.

    Puzzles are built from the answer outwards, so this is the seed of every
    synthesised panel. The search biases AWAY from the end node until the length
    target is met -- stepping onto the end early terminates the path (it can
    never be re-entered), which would otherwise collapse most draws to the
    beeline.
    """
    bps = frozenset(_edge(tuple(a), tuple(b)) for a, b in breakpoints)
    budget = [40_000]
    out = []

    def dfs(path):
        if out or budget[0] <= 0:
            return
        budget[0] -= 1
        cur = path[-1]
        if cur == end:
            if len(path) >= min_nodes:
                out.append(tuple(path))
            return
        nbrs = []
        for dc, dr in DIRS.values():
            nxt = (cur[0] + dc, cur[1] + dr)
            if not (0 <= nxt[0] <= cols and 0 <= nxt[1] <= rows):
                continue
            if nxt in path or _edge(cur, nxt) in bps:
                continue
            nbrs.append(nxt)
        rng.shuffle(nbrs)
        if len(path) < min_nodes:
            nbrs.sort(key=lambda n: n == end)      # try the end LAST
        for nxt in nbrs:
            dfs(path + (nxt,))

    dfs((start,))
    return out[0] if out else None


def node_mirror(node, cols, rows, symmetry):
    """The node a symmetry maps ``node`` to -- tw04's natural yellow start."""
    c, r = node
    if symmetry == "horizontal":
        return (cols - c, r)
    if symmetry == "vertical":
        return (c, rows - r)
    return (cols - c, rows - r)                    # rotational


def random_mirror_line(cols, rows, symmetry, blue_start, yellow_start, rng,
                       *, min_nodes, tries=60):
    """A random blue line whose mirrored twin stays legal the whole way (tw04).

    Both lines have to remain in bounds, stay self-avoiding and never share a
    node in the same step, so this walks the PAIR rather than the blue line
    alone. Where the walk ends is where the level's two end nodes go -- the
    puzzle is defined by its answer, so no endpoint pair is ever unreachable.
    """
    proto = SymmetryPanel(cols, rows, symmetry, blue_start, blue_start,
                          yellow_start, yellow_start, (), (), (), None)
    for _ in range(tries):
        path = (blue_start,)
        while True:
            options = [nxt for _, nxt in proto.extensions(path)]
            if not options:
                break
            path = rng.choice(options)
        if len(path) >= min_nodes:
            # Any PREFIX of a legal pair-walk is itself legal, so truncating
            # picks a different (still reachable) endpoint pair each draw.
            return proto, path[:rng.randint(min_nodes, len(path))]
    return proto, None


def path_edges(path):
    return {_edge(path[i], path[i + 1]) for i in range(len(path) - 1)}


def split_regions(grid, path):
    """The cell regions the line cuts the panel into.

    Delegates to the games' OWN ``WitnessGrid.path_splits_regions`` and returns
    its `set` objects untouched. That is deliberate and load-bearing for tw03:
    the engine's tiling check reads the shapes in region-ITERATION order, so a
    re-implementation that returned sorted lists would hand the checker a
    different shape order and could disagree with the engine about whether a
    board is solved. Callers that need a stable order for sampling must
    ``sorted()`` it themselves; callers that must mirror the engine (tw03) must
    not.
    """
    return grid.path_splits_regions(list(path))


def cell_edge_count(cell, edges):
    c, r = cell
    sides = (((c, r), (c + 1, r)),
             ((c, r + 1), (c + 1, r + 1)),
             ((c, r), (c, r + 1)),
             ((c + 1, r), (c + 1, r + 1)))
    return sum(1 for a, b in sides if _edge(a, b) in edges)


def corners(cols, rows):
    return [(0, 0), (cols, 0), (0, rows), (cols, rows)]


def partition_region(region, rng, max_piece=4):
    """Chop a region into random connected polyomino pieces (a tiling of it).

    tw03 asks for the reverse of the usual puzzle: instead of finding a tiling,
    build one and then hand the pieces to the player.
    """
    free = set(region)
    pieces = []
    while free:
        seed_cell = rng.choice(sorted(free))
        piece = {seed_cell}
        free.discard(seed_cell)
        target = rng.randint(1, max_piece)
        while len(piece) < target:
            frontier = sorted({(c + dc, r + dr)
                               for c, r in piece
                               for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1))}
                              & free)
            if not frontier:
                break
            grow = rng.choice(frontier)
            piece.add(grow)
            free.discard(grow)
        pieces.append(sorted(piece))
    return pieces


def normalise_shape(cells):
    min_c = min(c for c, _ in cells)
    min_r = min(r for _, r in cells)
    return sorted((c - min_c, r - min_r) for c, r in cells)


# ---------------------------------------------------------------------------
# The BaseSolver glue
# ---------------------------------------------------------------------------
class _ConfigProbe:
    """Just enough of a game for `read_panel` to read a config that has not been
    instantiated yet. The tw games copy their JSON level config into
    ``Level._data`` essentially verbatim, so the config IS the level data."""

    class _Level:
        def __init__(self, data):
            self._data = data

    def __init__(self, config):
        self.current_level = self._Level(config)


class WitnessSolver(BaseSolver):
    """Generator base for tw01..tw06. A subclass supplies the game class, the
    live-state constraint reader (`read_panel`) and the per-seed level recipe
    (`random_config`); everything else -- the expert, recovery, augmentation
    plumbing and the CLI -- is here."""

    game_cls = None
    #: Node grids per level for the SYNTHESISED panels, as ``(cols, rows)``.
    #: 4x4 is the practical ceiling for the region-constraint games: the
    #: enumeration is exhaustive there and budget-truncated at 5x5.
    level_shapes = ((3, 3), (3, 3), (4, 4), (4, 4), (4, 4))

    supports_recovery = True      # solve_from reads the LIVE line, never a cache
    recovery_mode = "replan"
    stochastic_optimal = True
    max_resets = 5

    #: DFS node budget per panel, the multiplier for the one retry a panel that
    #: yields nothing gets, and how many panels' solution sets to keep.
    dfs_budget = DFS_BUDGET
    budget_escalation = 100
    model_cache_size = 64

    def __init__(self, *args, augment=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = augment
        self._models = {}
        self._known = {}

    # -- subclass API -----------------------------------------------------
    def read_panel(self, game) -> Panel:
        """Build a `Panel` from the game's LIVE current level."""
        raise NotImplementedError

    def random_config(self, rng, level_idx, cols, rows):
        """A synthesised level config in the game's own JSON-level schema, plus
        the line it was built from: ``(config, known_solution)``."""
        raise NotImplementedError

    def read_path(self, game):
        return tuple(tuple(n) for n in game._path)

    # -- game construction -------------------------------------------------
    def _configs(self, seed):
        rng = random.Random(f"{self.game_id}:levels:{seed}")
        out, known = [], []
        for idx, (cols, rows) in enumerate(self.level_shapes):
            for _ in range(400):
                made = self.random_config(rng, idx, cols, rows)
                if made is None:
                    continue
                cfg, line = made
                # Puzzles are built backwards from ``line``, but "the line
                # satisfies the constraints I derived from it" is an assumption,
                # not a theorem -- tw03's tiling check, for one, is order
                # sensitive. So run the panel's REAL predicate over the answer
                # before shipping the level; a panel that fails here would be an
                # unsolvable board handed to the recorder.
                if self.read_panel(_ConfigProbe(cfg)).is_solution(tuple(line)):
                    break
            else:
                raise RuntimeError(
                    f"{self.game_id}: no level {idx} config for seed {seed}")
            out.append(cfg)
            known.append(line)
        return out, known

    def make_game(self, seed):
        # The model cache is keyed by panel signature, so with --no-augment
        # (every seed replaying the same panels) it is computed once for the
        # whole run. Synthesised panels are all distinct, so bound it -- and
        # only ever drop it BETWEEN episodes, so nothing is evicted mid-level.
        if len(self._models) > self.model_cache_size:
            self._models.clear()
        self._known = {}
        if not self.augment:
            return self.game_cls(seed)
        configs, known = self._configs(seed)
        self._known = {i: (line,) for i, line in enumerate(known)}
        entries = [{"config": c, "validated": True} for c in configs]
        # The games already know how to take levels from JSON (`_create_levels`
        # prefers `_load_json_levels()` over its hardcoded fallback), so feeding
        # synthesised panels in needs NO change to any game file -- just answer
        # that hook for the duration of the constructor.
        original = self.game_cls.__dict__.get("_load_json_levels")
        self.game_cls._load_json_levels = staticmethod(lambda: entries)
        try:
            return self.game_cls(seed)
        finally:
            if original is not None:
                setattr(self.game_cls, "_load_json_levels", original)
            else:                                   # pragma: no cover
                delattr(self.game_cls, "_load_json_levels")

    def set_level(self, game, level_idx):
        # ``ARCBaseGame.set_level`` does NOT restore sprites, and these games
        # repaint the panel by swapping the background sprite in place -- so a
        # level entered twice would render the PREVIOUS line. Re-clone first,
        # exactly as the engine's own ``level_reset`` does.
        clean = getattr(game, "_clean_levels", None)
        if clean is not None and 0 <= level_idx < len(clean):
            game._levels[level_idx] = clean[level_idx].clone()
        super().set_level(game, level_idx)

    # -- the expert --------------------------------------------------------
    def _model(self, game, level_idx, seed):
        """``(panel, prefix_minima)`` for the live level, computed once."""
        panel = self.read_panel(game)
        sig = panel.signature()
        cached = self._models.get(sig)
        if cached is not None:
            return cached
        known = self._known.get(level_idx, ())
        table = {}
        for budget in (self.dfs_budget, self.dfs_budget * self.budget_escalation):
            rng = random.Random(f"{self.game_id}:sols:{seed}:{level_idx}")
            solutions = [s for root in panel.roots()
                         for s in enumerate_solutions(panel, root, rng=rng,
                                                      known=known, budget=budget)]
            table = prefix_minima(solutions)
            if table:
                break
            # Nothing found -- either the panel really is unsolvable, or it is
            # one of the sparse ones (the shipped 5x5 tw02 has ~380 solutions
            # among 1.26M candidate lines) where the default budget runs out
            # first. Pay for one exhaustive pass before giving up; the result is
            # cached by panel signature, so a --no-augment run pays it once.
        self._models[sig] = (panel, table)
        return panel, table

    @staticmethod
    def _cost(panel, table, path):
        """Actions still needed to win from ``path`` (see the module docstring).

        Walking back to a shared prefix and out again is the ONLY way to get
        from one line to another, so the distance to the nearest solution is a
        minimum over the live line's own prefixes -- plus the ACTION5 wipe,
        which is a one-action teleport back to the start node.
        """
        if panel.is_solution(path):
            return 1
        best = INF
        for j in range(1, len(path) + 1):
            shortest = table.get(path[:j])
            if shortest is None:
                break            # no solution keeps this prefix, nor any longer
            best = min(best, (len(path) - j) + (shortest - j) + 1)
        root = panel.reset_root()
        if path != (root,):
            shortest = table.get((root,))
            if shortest is not None:
                best = min(best, 1 + shortest)
        return best

    def _optimal(self, panel, table, path):
        cost = self._cost(panel, table, path)
        if cost == INF:
            return []
        if cost == 1:
            return [SUBMIT]
        out = []
        for aid in (1, 2, 3, 4, SUBMIT):
            nxt = panel.step(path, aid)
            if nxt == path:                       # no-op action
                continue
            if self._cost(panel, table, nxt) == cost - 1:
                out.append(aid)
        return out

    def solve_from(self, game, level_idx, seed):
        panel, table = self._model(game, level_idx, seed)
        path = self.read_path(game)
        plan = []
        for _ in range(self.step_guard):
            options = self._optimal(panel, table, path)
            if not options:
                return []
            aid = options[0] if len(options) == 1 else self.rng.choice(options)
            plan.append(aid)
            if aid == SUBMIT and panel.is_solution(path):
                return plan
            path = panel.step(path, aid)
        return []

    def optimal_set_from(self, game, level_idx, seed):
        panel, table = self._model(game, level_idx, seed)
        return self._optimal(panel, table, self.read_path(game)) or None

    # -- CLI ---------------------------------------------------------------
    @classmethod
    def build_argparser(cls):
        p = super().build_argparser()
        p.add_argument("--no-augment", action="store_true",
                       help="Play the game's SHIPPED panels instead of "
                            "synthesising a fresh one per (seed, level). The "
                            "shipped panels are identical for every seed, so a "
                            "corpus generated this way has no board variety.")
        return p

    @classmethod
    def main(cls, argv=None):
        args = cls.build_argparser().parse_args(argv)
        solver = cls(rng=random.Random(args.seed), burst_prob=args.noise,
                     burst_mean=args.burst, augment=not args.no_augment)
        out = args.out or (Path("data/training_multi_level") / cls.game_id)
        return solver.run(args.episodes, out, start_seed=args.start_seed,
                          max_attempts=args.max_attempts,
                          progress_every=args.progress_every,
                          explore=not args.no_exploration)


# ---------------------------------------------------------------------------
# Shared level-reading helpers for the five single-line games
# ---------------------------------------------------------------------------
def read_common(game):
    """``(data, cols, rows, starts, end, breakpoints)`` from the live level."""
    data = game.current_level._data
    starts = ([tuple(s) for s in data["starts"]] if "starts" in data
              else [tuple(data["start"])])
    bps = [(tuple(a), tuple(b)) for a, b in data.get("breakpoints", [])]
    return data, data["cols"], data["rows"], starts, tuple(data["end"]), bps


def parse_cell_map(raw):
    """``{"c,r": v}`` (the games' JSON cell-key form) -> ``{(c, r): v}``."""
    out = {}
    for key, value in raw.items():
        c, r = key.split(",")
        out[(int(c), int(r))] = value
    return out


def pick_endpoints(cols, rows, rng):
    """A start/end pair for a synthesised panel: two distinct grid corners."""
    a, b = rng.sample(corners(cols, rows), 2)
    return a, b


SQUARE_COLORS = (SQUARE_A, SQUARE_B, SQUARE_C)
