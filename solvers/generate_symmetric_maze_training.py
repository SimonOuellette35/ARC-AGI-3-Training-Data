"""Generate Phase-1 training data for symmetric_maze (Symmetric Cursor Maze).

symmetric_maze (games/symmetric_maze/symmetric_maze.py) is a mirror-coupled maze:
four cursors sit at the four mirror images of one cell, an arrow key moves ALL of
them (cursor 1 with its left/right flipped, cursor 2 with up/down flipped, cursor
3 with both), and the level is won when all four occupy the SAME cell. ACTION5
toggles single-cursor mode, which is how the mirror lock-step can be broken.
Each *seed* is a 7-level game, so every WIN seed yields one multi-level episode.

Action set (simple actions only):
    ACTION1 (up)  ACTION2 (down)  ACTION3 (left)  ACTION4 (right)
    ACTION5 (toggle single-cursor mode, advancing the selected cursor)
    ACTION7 (undo) -- never used here; the exploration policy skips it too.

THE MIRROR FOLD: FOUR COUPLED CURSORS BECOME FOUR FREE TOKENS
-------------------------------------------------------------
The naive state is four cursor cells, which on a 15x15 board is ~110^4 = 1.5e8
configurations -- hopeless to search directly, and the coupling between the
cursors makes a per-cursor plan wrong. But the coupling is exactly a mirror, so it
folds away. Push every cursor through its own mirror::

    T_0 = identity                      T_1(x, y) = (w-1-x, y)
    T_2(x, y) = (x, h-1-y)              T_3(x, y) = (w-1-x, h-1-y)

and track ``u_i = T_i(pos_i)``. Because ``_MIRROR_DIRS[A][i]`` is precisely
``_MIRROR_DIRS[A][0]`` sign-flipped by ``T_i``, every cursor's *mirrored* move
becomes the SAME nominal (cursor-0) step in u-space; and because the maze is
symmetric under all four ``T_i``, a cursor is wall-blocked exactly when its token
is. So in u-space the game is plain and uncoupled:

  * an arrow key in all-cursor mode steps EVERY token by the nominal direction,
  * in single-cursor mode it steps ONLY the selected token, and
  * a level starts (and stays, under all-cursor play) with all four tokens
    STACKED on one cell -- which is why the group can only ever meet at the
    center, where the four mirrors coincide.

Winning ``pos_i == p`` for all i is ``u_i == T_i(p)``, so the u-space goal is
"tokens stacked at the center" for ``p = center`` and a mirror-spread tuple
otherwise. (This fold is verified against the engine by ``--self-test``.)

THE PLAN
--------
``solve_from`` reads the live tokens and splits on whether the mirror still holds:

  * **stacked + all-cursor mode** (the common case: every level start, and every
    state optimal play passes through) -- shortest path to the center over the
    precomputed distance-to-center field, a pure lookup. ``optimal_set_from``
    returns EVERY direction that descends that field, so the recorded target is
    the full optimal tie set and the base samples the taken action from it: a
    different equally-perfect route each episode.
  * **desynchronised** (exploration or a burst pressed ACTION5 and broke the
    lock-step) -- `_gather` builds a guaranteed plan by pricing every rendezvous
    cell over the all-pairs distance table (gather the strays with single-cursor
    moves, then walk the stack in as a group; or meet at a non-center cell when
    cursors already share one), including the exact ACTION5 overhead, which is
    lumpy because the selection only ever advances cyclically: 1/3/5/7 presses.
    Then `_search` searches for the true optimum, bounded by that plan's length,
    and the shorter of the two wins. The search is what finds the WALL FUNNEL --
    walk the group into a wall and the cursors it blocks stand still while the
    others close up, re-forming the mirror for zero ACTION5 presses -- which in
    these corridor mazes is usually the cheapest re-sync and which no
    hand-written plan family reliably contains. On a 7x7 that funnel is a 6-move
    recovery where the gather pays 16.

Measured: against an exhaustive BFS over the true state space, every one of 360
sampled 7x7 states is planned exactly optimally; and over 25 instrumented episodes
(175 levels, 948 desynchronised plans) the exact search never once ran out of
budget, so every recovery the generator emits is provably optimal. Desyncs that
exploration and bursts actually produce are mild -- 86% have the four tokens on
just two cells -- which is why that holds; the node cap only bites under
artificial four-way scatter, and there the constructive gather still carries it.

Recovery is therefore real rather than nominal: the whole per-level table set is
built once (all-pairs BFS over at most 225 cells), the exploration prefix costs
0-3 extra expert steps over pure optimal play, and RESET is never needed -- the
maze is fully reversible, so ``recovery_mode`` stays ``"replan"``.

NOT PRIVILEGED: the plan is built from the maze walls, the four cursor positions,
and the control mode -- all of which the frame shows (the walls and cursors
directly, the mode via the HUD pips added for exactly this reason). The game's
stashed ``solution_steps`` is never read.

Usage (run from the repo root):
    python solvers/generate_symmetric_maze_training.py --episodes 1000 \
        --out data/training_multi_level/symmetric_maze
    python solvers/generate_symmetric_maze_training.py --self-test
"""

from __future__ import annotations

import heapq
import itertools
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction                                  # noqa: E402
from games.symmetric_maze.symmetric_maze import (                 # noqa: E402
    _CONFIGS, CELL, SymmetricMaze)
from solvers.base_solver import BaseSolver                         # noqa: E402

# Nominal (cursor-0) deltas, index-aligned with ``_MOVE`` and with the game's own
# ``_MIRROR_DIRS[ACTIONn][0]``. In u-space EVERY cursor steps by these, whichever
# cursor the action actually drives -- that is the whole point of the mirror fold.
_NOMINAL = ((0, -1), (0, 1), (-1, 0), (1, 0))
_MOVE = (GameAction.ACTION1, GameAction.ACTION2,
         GameAction.ACTION3, GameAction.ACTION4)
_TOGGLE = GameAction.ACTION5

_INF = 1 << 20


def _select_presses(single: bool, active: int, target: int) -> int:
    """ACTION5 presses needed to be steering cursor ``target``.

    ACTION5 alternates all-cursor <-> single-cursor and advances the selection by
    one on every entry into single-cursor mode, so from ``(all-cursor, a)`` the
    reachable single-cursor selections are ``a+1, a+2, ...`` at 1, 3, 5, 7 presses
    -- cursor ``a`` itself is the EXPENSIVE one (a full lap, 7 presses), not the
    cheap one.
    """
    if single and active == target:
        return 0
    lap = (target - active) % 4
    if single:
        return 2 * lap                      # one press back to all-cursor, then 2*lap-1
    return 2 * (lap or 4) - 1


class _Maze:
    """Per-level geometry: walls, mirror maps, and all-pairs distances.

    Cells are flat ``y * w + x`` ids. The board is at most 15x15, so the whole
    all-pairs table is ~225 BFS runs -- built once per (seed, level) and then only
    read, which is what keeps `solve_from` cheap enough to call every step.
    """

    def __init__(self, grid, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.n = n = w * h
        self.free = [c for c in range(n) if grid[c // w][c % w] == 0]

        # nbr[d][c] = the cell one nominal step d from c, or -1 (wall / off-board).
        nbr = [[-1] * n for _ in range(4)]
        for c in self.free:
            x, y = c % w, c // w
            for d, (dx, dy) in enumerate(_NOMINAL):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and grid[ny][nx] == 0:
                    nbr[d][c] = ny * w + nx
        self.nbr = nbr
        self.center = (h // 2) * w + (w // 2)

        # The four mirrors, as cell -> cell maps. Each is an involution, so the
        # same table folds a cursor into u-space and unfolds a target back out.
        self.mirror = []
        for i in range(4):
            m = [0] * n
            for c in range(n):
                x, y = c % w, c // w
                if i in (1, 3):
                    x = w - 1 - x
                if i in (2, 3):
                    y = h - 1 - y
                m[c] = y * w + x
            self.mirror.append(m)

        self.dist = {s: self._bfs(s) for s in self.free}

    def _bfs(self, src: int) -> list[int]:
        d = [_INF] * self.n
        d[src] = 0
        queue = deque([src])
        while queue:
            c = queue.popleft()
            for dd in range(4):
                t = self.nbr[dd][c]
                if t >= 0 and d[t] == _INF:
                    d[t] = d[c] + 1
                    queue.append(t)
        return d

    def descending(self, field: list[int], cell: int) -> list[int]:
        """Every nominal direction from ``cell`` that strictly descends ``field``
        -- i.e. the equally-optimal steps toward that field's source."""
        here = field[cell]
        out = []
        for d in range(4):
            t = self.nbr[d][cell]
            if t >= 0 and field[t] == here - 1:
                out.append(d)
        return out

    def path(self, src: int, dst: int, rng) -> list[int]:
        """One shortest path ``src -> dst`` as nominal directions, tie-broken at
        random so repeated recoveries do not all take the same corridor."""
        field = self.dist[dst]
        dirs, cell = [], src
        while cell != dst:
            opts = self.descending(field, cell)
            if not opts:                       # unreachable -- caller filters these
                return []
            d = opts[0] if len(opts) == 1 else rng.choice(opts)
            dirs.append(d)
            cell = self.nbr[d][cell]
        return dirs


class SymmetricMazeSolver(BaseSolver):
    """Data generator for symmetric_maze. See the module docstring."""

    game_id = "symmetric_maze"
    supports_recovery = True          # solve_from re-plans from ANY live state
    recovery_mode = "replan"          # the maze is fully reversible
    stochastic_optimal = True         # tie set comes from `optimal_set_from`

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._geom: dict[int, _Maze] = {}       # level_idx -> geometry
        self._geom_seed: int | None = None

    # ── engine plumbing ──────────────────────────────────────────────────────
    def make_game(self, seed: int):
        return SymmetricMaze(seed=seed)

    # ── cheap burst snapshots ────────────────────────────────────────────────
    def _game_snapshot(self, game):
        """Capture exactly what a perturbation burst can change.

        The base's default is ``deepcopy(game)``, which for this game copies all
        seven levels' ~200 immovable wall sprites twice over (``_levels`` and
        ``_clean_levels``) on every burst -- measured at ~7x the cost of the entire
        rest of the episode. Inside a level only the four cursors, the control
        mode, the undo history and the budget/termination bookkeeping move; the
        walls, floor and goal sprites are built once by ``_build_level`` and never
        touched. So snapshot those and nothing else. ``--self-test`` cross-checks
        this against a real ``deepcopy`` round-trip.
        """
        level = game.current_level
        cursors = [level.get_sprites_by_name(f"cursor_{i}")[0] for i in range(4)]
        return ("symmetric_maze",
                [(c.x, c.y) for c in cursors],
                bool(game._single_mode), int(game._active_cursor),
                list(game._history), game._step_counter.steps_remaining,
                game._state, bool(getattr(game, "_next_level", False)),
                game._action_count, game._score)

    def _game_restore(self, game, snap) -> None:
        (_, positions, single, active, history, steps, state, next_level,
         action_count, score) = snap
        level = game.current_level
        for i, (x, y) in enumerate(positions):
            level.get_sprites_by_name(f"cursor_{i}")[0].set_position(x, y)
        game._single_mode = single
        game._active_cursor = active
        game._history = list(history)        # fresh list: a snapshot may be reused
        game._step_counter.steps_remaining = steps
        game._state = state
        game._next_level = next_level
        game._action_count = action_count
        game._score = score
        game._sync_hud()

    # ── live state, read from what the frame shows ────────────────────────────
    def _maze(self, game, level_idx: int, seed: int) -> _Maze:
        if self._geom_seed != seed:             # one seed's tables at a time
            self._geom.clear()
            self._geom_seed = seed
        mz = self._geom.get(level_idx)
        if mz is None:
            level = game.current_level
            gw, gh = level.grid_size
            w, h = gw // CELL, gh // CELL
            grid = [[0] * w for _ in range(h)]
            for sprite in level.get_sprites():
                name = sprite.name or ""
                if name.startswith("wall_"):
                    _, sx, sy = name.split("_")
                    grid[int(sy)][int(sx)] = 1
            mz = _Maze(grid, w, h)
            self._geom[level_idx] = mz
        return mz

    def _read(self, game, level_idx: int, seed: int):
        """``(maze, tokens, single_mode, active_cursor)`` at the live state.

        ``tokens[i]`` is cursor i folded into u-space. The control mode is read off
        the game rather than re-decoded from the HUD pixels it draws -- the same
        shortcut as reading wall sprites instead of thresholding wall pixels. Both
        are on screen (see the game's HUD), so nothing here is hidden state.
        """
        mz = self._maze(game, level_idx, seed)
        level = game.current_level
        tokens = []
        for i in range(4):
            sprite = level.get_sprites_by_name(f"cursor_{i}")[0]
            cell = (sprite.y // CELL) * mz.w + (sprite.x // CELL)
            tokens.append(mz.mirror[i][cell])
        return (mz, tuple(tokens),
                bool(game._single_mode), int(game._active_cursor))

    # ── the expert ───────────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int) -> list:
        mz, tokens, single, active = self._read(game, level_idx, seed)
        if tokens[0] == tokens[1] == tokens[2] == tokens[3]:
            # Mirror-locked: the four cursors can only meet at the center, so walk
            # the stack down the distance-to-center field. Every level start is
            # here, and so is every state optimal play passes through.
            stack = tokens[0]
            if stack == mz.center:
                return []                       # already won; unreachable in practice
            return self._emit(mz, tokens, single, active, (stack,) * 4,
                              mz.path(stack, mz.center, self.rng))
        # Desynchronised. The constructive gather is a guaranteed, cheap plan; the
        # A* refines it to the true optimum whenever it can do so within its node
        # budget (which, for the near-mirror states exploration actually produces,
        # is essentially always).
        plan = self._gather(mz, tokens, single, active)
        if not plan:
            return []
        # A cheap GREEDY pass first (same search, heuristic over-weighted so it
        # dives at the goal instead of proving optimality). Its length is usually
        # within a step or two of optimal, and handing THAT to the exact pass as
        # the bound -- rather than the gather's, which is ~40% longer -- is what
        # decides whether the exact pass finishes inside its node budget.
        quick = self._search(mz, tokens, single, active, limit=len(plan),
                             weight=5, max_nodes=20_000)
        if quick:
            plan = quick
        exact = self._search(mz, tokens, single, active, limit=len(plan))
        return exact or plan

    def _search(self, mz: _Maze, tokens, single: bool, active: int, *,
                limit: int, weight: int = 1,
                max_nodes: int = 60_000) -> list | None:
        """Exact shortest route back to "all four stacked on the center", or None if
        nothing beats ``limit`` within the node budget.

        WHY A SEARCH AND NOT JUST THE GATHER. `_gather` assumes single-cursor moves
        do the gathering and group moves only translate the stack -- but the
        cheapest re-sync in a corridor maze is usually neither: walk the GROUP into
        a wall, and the cursors it blocks stand still while the others close up, so
        the mirror re-forms for zero ACTION5 presses. In a 7x7 that funnel is a
        6-move recovery where the gather pays 16. Any plan family hand-written
        around one such trick will miss the next one, so the desync branch searches
        instead.

        A* over ``(tokens, single_mode)`` with ``h = max_i dist(token_i, center)``:
        no action moves any token more than one cell, so h never overestimates and
        never changes by more than 1 per step -- admissible AND consistent, so the
        first pop of the goal is optimal and no state needs re-expanding. ``limit``
        prunes every branch that cannot beat the plan we already hold, which is what
        keeps the search small; the prune uses the unweighted ``g + h``, so it stays
        valid whatever ``weight`` is.

        ``weight > 1`` over-weights h in the priority only: the search then dives at
        the goal and returns a good plan fast, but NOT a provably optimal one. It is
        used to obtain a tight ``limit`` for the subsequent exact (``weight == 1``)
        pass, and as the answer if that pass runs out of budget.

        THE SELECTION IS FOLDED AWAY TOO. The state carries no ``active_cursor``:
        the tuple is kept ROTATED so the selected cursor is index 0. Rotating the
        tokens and the selection together is an exact symmetry here -- the mirror
        fold already made every action's u-space effect independent of WHICH cursor
        it drives, ACTION5 advances the selection cyclically (which is just another
        rotation), and the goal is rotation-invariant -- so the four selections
        collapse onto one and the search shrinks 4x. The emitted plan needs no
        un-rotating, because the action ids never referred to a cursor index.

        The goal is the CENTER meeting point only, because that is what h bounds.
        Meeting at some other cell -- worth it once three cursors already share one
        -- stays `_gather`'s job, and the caller keeps whichever came out shorter.
        """
        field = mz.dist[mz.center]
        nbr = mz.nbr
        goal = (mz.center,) * 4
        start = (tokens[active:] + tokens[:active], single)
        h0 = max(field[t] for t in tokens)
        if h0 >= limit:                             # cannot beat what we already have
            return None
        best_g = {start: 0}
        parent: dict = {}
        heap = [(weight * h0, 0, start)]
        nodes = 0
        while heap:
            _, g, state = heapq.heappop(heap)
            if g > best_g.get(state, _INF):
                continue                            # stale heap entry
            toks, sing = state
            if toks == goal:
                plan = []
                while state in parent:
                    state, aid = parent[state]
                    plan.append(_TOGGLE if aid == 5 else _MOVE[aid - 1])
                plan.reverse()
                return plan
            nodes += 1
            if nodes > max_nodes:
                return None                         # give up; caller uses the gather
            for aid in (1, 2, 3, 4, 5):
                if aid == 5:
                    # Leaving single-cursor mode keeps the selection; entering it
                    # advances -- one rotation of the canonical tuple.
                    nxt = ((toks, False) if sing
                           else (toks[1:] + toks[:1], True))
                else:
                    d = aid - 1
                    moved = list(toks)
                    for i in ((0,) if sing else (0, 1, 2, 3)):
                        step = nbr[d][toks[i]]
                        if step >= 0:
                            moved[i] = step
                    moved = tuple(moved)
                    if moved == toks:
                        continue                    # wall on every mover: a no-op
                    nxt = (moved, sing)
                ng = g + 1
                nh = max(field[t] for t in nxt[0])
                if ng + nh >= limit:                # cannot beat the plan we hold
                    continue
                if ng < best_g.get(nxt, _INF):
                    best_g[nxt] = ng
                    parent[nxt] = (state, aid)
                    heapq.heappush(heap, (ng + weight * nh, ng, nxt))
        return None

    def optimal_set_from(self, game, level_idx: int, seed: int) -> list | None:
        """The full optimal tie set, but only for the mirror-locked states where it
        is exactly known: every direction that descends the distance-to-center
        field. Off the mirror (mid-recovery) return None, so the base keeps the
        single canonical head of the gather plan as the target."""
        mz, tokens, single, active = self._read(game, level_idx, seed)
        if not tokens[0] == tokens[1] == tokens[2] == tokens[3]:
            return None
        stack = tokens[0]
        if stack == mz.center:
            return None
        if single:
            return [_TOGGLE]                    # back to all-cursor mode first
        field = mz.dist[mz.center]
        return [_MOVE[d] for d in mz.descending(field, stack)] or None

    # ── planning ─────────────────────────────────────────────────────────────
    def _gather(self, mz: _Maze, tokens, single: bool, active: int) -> list:
        """Cheapest way back to a win from a desynchronised (post-exploration)
        state, over two families of plan:

          (a) gather every token onto a rendezvous cell with single-cursor moves,
              then walk the stack to the center as a group -- this is the family
              that covers "one cursor drifted off the mirror", and it degenerates
              to the pure group walk when the tokens are already stacked;
          (b) meet at a non-center cell ``p`` outright, i.e. drive each token to
              ``T_i(p)`` with single-cursor moves and no group walk -- the cheap
              option once several cursors already share a cell.

        Each candidate is priced exactly (moves + ACTION5 presses) and the best is
        emitted. This is a fast, always-available plan, not the optimum: it cannot
        see the wall-funnel re-syncs that `_search` finds. `solve_from` runs it
        first anyway, because its length is the bound that makes that search cheap,
        and it is the fallback if the search runs out of budget.
        """
        dist = mz.dist
        best = None                             # (cost, targets, tail_dirs)
        for c in mz.free:
            legs = [dist[u][c] for u in tokens]
            if max(legs) >= _INF:
                continue
            tail = dist[c][mz.center]
            if tail >= _INF:
                continue
            targets = (c,) * 4
            cost = (sum(legs) + tail
                    + self._presses(targets, tokens, single, active, tail > 0)[0])
            if best is None or cost < best[0]:
                best = (cost, targets, c)
        for p in mz.free:
            if p == mz.center:
                continue                        # already covered, with a group tail
            targets = tuple(mz.mirror[i][p] for i in range(4))
            legs = [dist[tokens[i]][targets[i]] for i in range(4)]
            if max(legs) >= _INF:
                continue
            cost = (sum(legs)
                    + self._presses(targets, tokens, single, active, False)[0])
            if best is None or cost < best[0]:
                best = (cost, targets, None)

        if best is None:                        # no reachable meeting point at all
            return []
        _, targets, rendezvous = best
        tail = ([] if rendezvous is None
                else mz.path(rendezvous, mz.center, self.rng))
        return self._emit(mz, tokens, single, active, targets, tail)

    def _presses(self, targets, tokens, single: bool, active: int,
                 group_tail: bool):
        """``(total ACTION5 presses, cursor visit order)`` for driving each stray
        token to its target one cursor at a time.

        The selection only ever advances cyclically, so the order matters; with at
        most four strays the exact minimum is a 24-permutation scan.
        """
        stray = [i for i in range(4) if tokens[i] != targets[i]]
        best = None
        for order in itertools.permutations(stray):
            total, cur_single, cur_active = 0, single, active
            for i in order:
                total += _select_presses(cur_single, cur_active, i)
                cur_single, cur_active = True, i
            if group_tail and cur_single:
                total += 1                      # back to all-cursor mode for the walk
            if best is None or total < best[0]:
                best = (total, order)
        return best                             # >= 1 candidate: () when none stray

    def _emit(self, mz: _Maze, tokens, single: bool, active: int, targets,
              tail_dirs: list[int]) -> list:
        """Turn a (targets, group-tail) plan into the action list."""
        _, order = self._presses(targets, tokens, single, active, bool(tail_dirs))
        plan: list = []
        cur_single, cur_active = single, active
        for i in order:
            plan += [_TOGGLE] * _select_presses(cur_single, cur_active, i)
            cur_single, cur_active = True, i
            plan += [_MOVE[d] for d in mz.path(tokens[i], targets[i], self.rng)]
        if tail_dirs:
            if cur_single:
                plan.append(_TOGGLE)
            plan += [_MOVE[d] for d in tail_dirs]
        return plan


# ---------------------------------------------------------------------------
# Self-test: the two claims the whole solver rests on
# ---------------------------------------------------------------------------
def _self_test(seeds: int = 8) -> int:
    """Check, against the real engine, that

      1. the MIRROR FOLD holds -- after any action, every cursor's new position is
         exactly its u-space token stepped by the action's nominal direction (or
         held by a wall), for both control modes; and
      2. RECOVERY works -- from a state randomly perturbed off the mirror (the
         situation the exploration prefix and the bursts create), the plan
         ``solve_from`` returns actually wins when driven; and

      3. the cheap burst SNAPSHOT is faithful -- restoring it leaves the same
         frame, the same live state, and the same future as the base's generic
         ``deepcopy`` round-trip would.

    Failure prints the offending (seed, level, state) and returns non-zero.
    """
    import copy
    import random

    from utils.explore import Action

    solver = SymmetricMazeSolver(rng=random.Random(12345))
    rng = random.Random(999)
    fold_checks = recoveries = snapshots = 0

    def tokens_of(game, mz):
        out = []
        for i in range(4):
            s = game.current_level.get_sprites_by_name(f"cursor_{i}")[0]
            out.append(mz.mirror[i][(s.y // CELL) * mz.w + (s.x // CELL)])
        return tuple(out)

    for seed in range(seeds):
        game = solver.make_game(seed)
        for level_idx in range(solver.num_levels(game)):
            solver.set_level(game, level_idx)
            solver._lift_step_limit(game)
            mz = solver._maze(game, level_idx, seed)

            # (1) the fold, over a random walk that mixes moves and mode toggles.
            for _ in range(120):
                aid = rng.choice([1, 2, 3, 4, 5])
                before = tokens_of(game, mz)
                single, active = bool(game._single_mode), int(game._active_cursor)
                res = solver.drive(game, Action(aid))
                after = tokens_of(game, mz)
                if aid == 5:
                    want = after == before
                else:
                    d = _NOMINAL[aid - 1]
                    want = True
                    for i in range(4):
                        step = mz.nbr[aid - 1][before[i]]
                        moves = (not single) or i == active
                        expect = step if (moves and step >= 0) else before[i]
                        want = want and after[i] == expect
                fold_checks += 1
                if not want:
                    print(f"FOLD BROKEN seed={seed} level={level_idx} action={aid} "
                          f"single={single} active={active} d={_NOMINAL[aid-1] if aid<5 else None} "
                          f"before={before} after={after}")
                    return 1
                if res.solved or res.dead:
                    solver.reset_level(game, level_idx, seed)   # restart this level

            # (2) recovery from perturbed states, including desynchronised ones.
            for trial in range(6):
                solver.reset_level(game, level_idx, seed)
                for _ in range(rng.randint(1, 14)):
                    aid = rng.choice([1, 2, 3, 4, 5])
                    if solver.drive(game, Action(aid)).solved:
                        break
                else:
                    state = solver._read(game, level_idx, seed)
                    plan = solver.solve_from(game, level_idx, seed)
                    if not plan:
                        print(f"NO PLAN seed={seed} level={level_idx} state={state[1:]}")
                        return 1
                    won = False
                    for act in plan:
                        res = solver.drive(game, Action(int(act.value)))
                        if res.solved:
                            won = True
                            break
                        if res.dead:
                            break
                    if not won:
                        print(f"RECOVERY FAILED seed={seed} level={level_idx} "
                              f"tokens={state[1]} single={state[2]} active={state[3]} "
                              f"plan={[int(a.value) for a in plan]}")
                        return 1
                    recoveries += 1
                del trial

            # (3) the cheap snapshot vs the base's generic deepcopy round-trip.
            for _ in range(4):
                solver.reset_level(game, level_idx, seed)
                for _ in range(rng.randint(1, 10)):
                    if solver.drive(game, Action(rng.choice([1, 2, 3, 4, 5]))).solved:
                        break
                probe = [Action(rng.choice([1, 2, 3, 4, 5])) for _ in range(6)]
                fast = solver._game_snapshot(game)
                generic = ("native", copy.deepcopy(game))

                def replay():
                    trace = [solver.render(game).tolist()]
                    for act in probe:
                        trace.append(solver.drive(game, act).frame.tolist())
                    return trace

                solver._game_restore(game, fast)
                cheap_trace = replay()
                BaseSolver._game_restore(solver, game, generic)
                if replay() != cheap_trace:
                    print(f"SNAPSHOT MISMATCH seed={seed} level={level_idx}")
                    return 1
                snapshots += 1
    print(f"self-test OK: {fold_checks} fold checks, {recoveries} driven "
          f"recoveries, {snapshots} snapshot round-trips over {seeds} seeds "
          f"x {len(_CONFIGS)} levels")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(SymmetricMazeSolver.main())
