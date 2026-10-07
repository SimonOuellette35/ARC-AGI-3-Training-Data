"""Generate Phase-1 training data for the Gem Mine game.

Gem Mine (games/gem_mine/gem_mine.py): guide the miner through a cave, pick up
gems one at a time (ACTION5 adjacent to a gem) and deposit them at the depot
cart (ACTION5 on the depot). The miner carries one gem at a time; the level wins
when every gem is deposited. Each *seed* is a full 7-cave game, so each WIN seed
yields one multi-level episode.

Gem Mine action set:
    ACTION1 up   ACTION2 down   ACTION3 left   ACTION4 right
    ACTION5 pick up adjacent gem / deposit carried gem at the depot
    ACTION7 undo  (index 7 -- NOT emitted; the BFS never needs it)

The expert is a BFS over (miner_x, miner_y, carried_gem, deposited_mask) in grid
coordinates. ACTION5 is modelled exactly: it picks up the FIRST adjacent on-floor
gem in up/down/left/right order, and deposits the carried gem when on the depot.
``solve_from`` reads the LIVE game (miner cell, carried gem colour, deposited
gems via sprite visibility) for its start state; the static cave/depot/gem layout
comes from ``game._level_data[level_idx]``.

Reversible / always-solvable: from any reachable state the miner can still finish
(deposit the carried gem, then collect the rest), and ``solve_from`` reads the
live state, so ``supports_recovery = True`` -- the exploration prefix + bursts
are safe.

Usage (run from the repo root):
    python solvers/generate_gem_mine_training.py --episodes 1000 \
        --out data/training_multi_level/gem_mine
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction                            # noqa: E402
from games.gem_mine.gem_mine import GemMine, CELL           # noqa: E402
from solvers.base_solver import BaseSolver                  # noqa: E402

# Grid-space move deltas, in the SAME order the game scans directions.
_MOVE_ACTIONS = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}
# Adjacency scan order used by the game's ACTION5 pickup (up, down, left, right).
_PICKUP_ADJ = [(0, -1), (0, 1), (-1, 0), (1, 0)]


class GemMineSolver(BaseSolver):
    game_id = "gem_mine"
    # Always solvable from any reachable state; solve_from reads live state.
    supports_recovery = True

    def make_game(self, seed: int):
        return GemMine(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4, 5]

    def set_level(self, game, level_idx: int) -> None:
        # Clear the engine's ``_next_level`` flag the base breaks on (set_level
        # does not). gem_mine's on_set_level ADDS the miner/gem sprites without
        # removing old ones, and __init__ already built level 0, so re-setting
        # level 0 would double-add -- skip it (the constructed game is at 0).
        game._next_level = False
        if level_idx != 0:
            game.set_level(level_idx)

    def solve_from(self, game, level_idx: int, seed: int):
        """BFS depositing every gem, from the LIVE miner state on ``level_idx``.

        Reads the live game (miner cell, carried-gem colour, which gems are
        already deposited via visibility) as the BFS start; the static cave/depot/
        gem layout comes from ``game._level_data``. Models ACTION5 exactly (first
        adjacent on-floor gem in up/down/left/right order; deposit at depot).
        Returns a simple-action plan (moves + ACTION5)."""
        cave, _miner_start, depot, gem_positions, gem_colors = \
            game._level_data[level_idx]
        height = len(cave)
        width = len(cave[0])
        n_gems = len(gem_positions)

        tunnels = frozenset(
            (x, y) for y in range(height) for x in range(width) if cave[y][x] == 1
        )
        gem_pos_to_idx = {tuple(gp): i for i, gp in enumerate(gem_positions)}
        depot_grid = tuple(depot)
        all_deposited = (1 << n_gems) - 1
        NONE = n_gems  # sentinel for "not carrying"

        # --- Live start state (grid coords) ---
        miner = game._miner()
        mx, my = miner.x // CELL, miner.y // CELL
        carried = NONE
        if game._carried_gem is not None:
            carried = gem_colors.index(game._carried_gem)
        deposited = 0
        gems = game._gems()
        for i in range(n_gems):
            gs = [s for s in gems if f"gem_{i}" in s.tags]
            invisible = gs and not gs[0].is_visible
            if invisible and i != carried:
                deposited |= (1 << i)
        start = (mx, my, carried, deposited)

        parent = {start: None}
        action_rec = {start: None}
        q = deque([start])
        goal = None

        while q and goal is None:
            state = q.popleft()
            mx, my, carried, deposited = state

            def _add(ns, act):
                if ns not in parent:
                    parent[ns] = state
                    action_rec[ns] = act
                    q.append(ns)

            # --- Movement ---
            for action, (dx, dy) in _MOVE_ACTIONS.items():
                nx, ny = mx + dx, my + dy
                if (nx, ny) in tunnels:
                    _add((nx, ny, carried, deposited), action)

            # --- ACTION5: deposit ---
            if (mx, my) == depot_grid and carried != NONE:
                new_dep = deposited | (1 << carried)
                ns = (mx, my, NONE, new_dep)
                if new_dep == all_deposited:
                    if ns not in parent:
                        parent[ns] = state
                        action_rec[ns] = GameAction.ACTION5
                    goal = ns
                    break
                _add(ns, GameAction.ACTION5)

            # --- ACTION5: pickup (first adjacent on-floor gem, game order) ---
            if carried == NONE:
                for adx, ady in _PICKUP_ADJ:
                    gi = gem_pos_to_idx.get((mx + adx, my + ady))
                    if gi is not None and not (deposited >> gi) & 1:
                        _add((mx, my, gi, deposited), GameAction.ACTION5)
                        break

        if goal is None:
            return []                              # unsolvable from here

        actions: list = []
        node = goal
        while action_rec[node] is not None:
            actions.append(action_rec[node])
            node = parent[node]
        actions.reverse()
        return actions

    #: Cap on the abstract state-space size ``tunnels x (n_gems+1) x 2**n_gems``
    #: for which ``optimal_set_from`` computes the full co-optimal set. gem_mine's
    #: state is exponential in the gem count (up to ~950k cells for the 9-gem
    #: caves), and the base RE-SOLVES every step once a tie set is returned, so a
    #: per-step full-state BFS on the big caves would make generation impractically
    #: slow. Above the cap ``optimal_set_from`` returns ``None`` -> the base replays
    #: the single canonical head (baseline speed, still perfectly optimal, just no
    #: extra diversity). The early caves (<=5 gems) stay well under the cap and get
    #: full stochastic-optimal diversity.
    optset_state_cap: int = 60_000

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-optimal next action at the LIVE state: each move/ACTION5
        that begins a shortest deposit-every-gem solution.

        Distance is over the SAME abstract state ``(mx, my, carried, deposited)``
        the BFS uses, so a pickup is returned only when collecting THAT gem next is
        co-optimal for the full remaining objective (gem order + routing), never
        just because a gem is adjacent. Uses ONE BFS from the live state with
        first-move label propagation (the set of first actions that can start a
        shortest path to each state), so the whole tie set costs a single BFS
        rather than one-per-candidate. The transition mirrors ``solve_from``
        exactly, so the canonical head is always in the set. Returns ``None`` on the
        large caves (see ``optset_state_cap``) -> the base replays the single
        canonical head at baseline speed."""
        cave, _miner_start, depot, gem_positions, gem_colors = \
            game._level_data[level_idx]
        height = len(cave)
        width = len(cave[0])
        n_gems = len(gem_positions)

        tunnels = frozenset(
            (x, y) for y in range(height) for x in range(width) if cave[y][x] == 1
        )
        # Too big to compute the full set per step (see optset_state_cap).
        if len(tunnels) * (n_gems + 1) * (1 << n_gems) > self.optset_state_cap:
            return None
        gem_pos_to_idx = {tuple(gp): i for i, gp in enumerate(gem_positions)}
        depot_grid = tuple(depot)
        all_deposited = (1 << n_gems) - 1
        NONE = n_gems

        def neighbors(state):
            mx, my, carried, deposited = state
            for action, (dx, dy) in _MOVE_ACTIONS.items():
                nx, ny = mx + dx, my + dy
                if (nx, ny) in tunnels:
                    yield action, (nx, ny, carried, deposited)
            # ACTION5: deposit at the depot, else pick up the first adjacent gem
            # (mutually exclusive: deposit needs carried, pickup needs empty hands).
            if (mx, my) == depot_grid and carried != NONE:
                yield GameAction.ACTION5, (mx, my, NONE,
                                           deposited | (1 << carried))
            if carried == NONE:
                for adx, ady in _PICKUP_ADJ:
                    gi = gem_pos_to_idx.get((mx + adx, my + ady))
                    if gi is not None and not (deposited >> gi) & 1:
                        yield GameAction.ACTION5, (mx, my, gi, deposited)
                        break

        miner = game._miner()
        mx, my = miner.x // CELL, miner.y // CELL
        carried = NONE
        if game._carried_gem is not None:
            carried = gem_colors.index(game._carried_gem)
        deposited = 0
        gems = game._gems()
        for i in range(n_gems):
            gs = [s for s in gems if f"gem_{i}" in s.tags]
            invisible = gs and not gs[0].is_visible
            if invisible and i != carried:
                deposited |= (1 << i)
        s0 = (mx, my, carried, deposited)
        if s0[3] == all_deposited:                 # already solved
            return []

        # Single BFS from s0; ``first[state]`` = the set of first actions out of s0
        # that can begin a shortest path to ``state`` (FIFO order guarantees each
        # state's label set is complete before it is expanded).
        dist = {s0: 0}
        first: dict = {s0: frozenset()}
        q = deque([s0])
        best_goal = None
        while q:
            u = q.popleft()
            du = dist[u]
            if best_goal is not None and du > best_goal:
                break                              # goal layer fully settled
            if u[3] == all_deposited and best_goal is None:
                best_goal = du
            for act, v in neighbors(u):
                contrib = frozenset((act,)) if u == s0 else first[u]
                if v not in dist:
                    dist[v] = du + 1
                    first[v] = contrib
                    q.append(v)
                elif dist[v] == du + 1:
                    first[v] = first[v] | contrib
        if best_goal is None:
            return []                              # unreachable from here
        res: frozenset = frozenset()
        for st, d in dist.items():
            if d == best_goal and st[3] == all_deposited:
                res |= first[st]
        return list(res)


if __name__ == "__main__":
    sys.exit(GemMineSolver.main())
