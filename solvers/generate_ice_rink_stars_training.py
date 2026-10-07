"""Generate Phase-1 training data for the Ice Rink Stars game.

Ice Rink Stars (games/ice_rink_stars/ice_rink_stars.py): the skater slides in a
direction until it hits a wall/board, collecting every star it passes through.
The level generator guarantees each rink is winnable by sliding alone -- no ice
block (ACTION5) or undo (ACTION7) needed -- so the expert is a plain BFS over
(skater_gx, skater_gy, collected_star_mask) using ACTION1..4. Each *seed* is a
full 7-rink game, so each WIN seed yields one multi-level episode.

Ice Rink Stars action set:
    ACTION1 (up)   ACTION2 (down)   ACTION3 (left)   ACTION4 (right)
    ACTION5 place ice block          ACTION7 undo (NOT emitted)

``solve_from`` reads the LIVE game (skater cell, collected stars via sprite
visibility) for its BFS start; the static walls/stars layout comes from
``game._level_data[level_idx]``.

NOT reversible for recovery: a slide can strand the skater in a spot from which
the remaining stars are no longer slide-reachable, and exploratory ACTION5 places
permanent-ish ice blocks, so an off-plan detour can't be re-planned from.
Recovery is RESET-mode (``supports_recovery = True``, ``recovery_mode =
"reset"``): explore the prefix, then a single RESET to the rink's initial state,
then replay the BFS slide plan from there.

Usage (run from the repo root):
    python solvers/generate_ice_rink_stars_training.py --episodes 1000 \
        --out data/training_multi_level/ice_rink_stars
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction                            # noqa: E402
from games.ice_rink_stars.ice_rink_stars import (           # noqa: E402
    IceRinkStars, CELL)
from solvers.base_solver import BaseSolver                  # noqa: E402

# Slide directions in grid coords, paired with the launching simple action.
_SLIDE_DIRS = [
    ((0, -1), GameAction.ACTION1),  # up
    ((0,  1), GameAction.ACTION2),  # down
    ((-1, 0), GameAction.ACTION3),  # left
    ((1,  0), GameAction.ACTION4),  # right
]


class IceRinkStarsSolver(BaseSolver):
    game_id = "ice_rink_stars"
    # solve_from BFSes the slide-graph over the LIVE skater cell + star mask, so it
    # re-plans from any perturbed-but-solvable state -> replan mode. A slide that
    # strands the skater (making the rink slide-unsolvable) is handled by the
    # burst-undo rollback (the stranding burst is erased), with RESET as last resort.
    supports_recovery = True
    recovery_mode = "replan"

    def make_game(self, seed: int):
        return IceRinkStars(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    def set_level(self, game, level_idx: int) -> None:
        # Clear the engine's ``_next_level`` flag the base breaks on (set_level
        # does not). on_set_level removes+re-adds the dynamic skater/star/cursor/
        # block sprites, so re-setting a level (incl. level 0) is idempotent.
        game._next_level = False
        game.set_level(level_idx)

    def solve_from(self, game, level_idx: int, seed: int):
        """BFS collecting every star from the LIVE skater state on ``level_idx``.

        Reads the live game (skater cell, collected stars via which star sprites
        remain) as the BFS start; the static walls/stars layout comes from
        ``game._level_data``. Slide-only (no ice blocks). Returns a simple-action
        plan of ACTION1..4 launches."""
        walls_set, stars, _skater_start, width, height = \
            game._level_data[level_idx]
        stars_list = list(stars)
        n_stars = len(stars_list)
        all_collected = (1 << n_stars) - 1

        # --- Live start state (grid coords) ---
        level = game.current_level
        skater = level.get_sprites_by_tag("skater")[0]
        skx, sky = skater.x // CELL, skater.y // CELL
        # A collected star stays in the level as an INVISIBLE sprite (the engine
        # calls set_visible(False), never remove_sprite), so "still to collect"
        # == the star sprite is still visible. Filtering on is_visible is required
        # for a correct live collected-mask whenever solve_from is re-invoked
        # MID-level (which the stochastic-optimal tie set triggers).
        present = {(s.x // CELL, s.y // CELL)
                   for s in level.get_sprites_by_tag("star") if s.is_visible}
        start_mask = 0
        for i, (stx, sty) in enumerate(stars_list):
            if (stx, sty) not in present:
                start_mask |= (1 << i)

        def _slide_collect(sx, sy, dx, dy, mask):
            x, y = sx, sy
            while True:
                nx, ny = x + dx, y + dy
                if (nx, ny) in walls_set or not (0 <= nx < width and 0 <= ny < height):
                    break
                x, y = nx, ny
                for i, (stx, sty) in enumerate(stars_list):
                    if x == stx and y == sty and not (mask >> i) & 1:
                        mask |= (1 << i)
            return x, y, mask

        start = (skx, sky, start_mask)
        parent = {start: None}
        action_rec = {start: None}
        q = deque([start])
        goal = None
        if start_mask == all_collected:
            return []

        while q and goal is None:
            sx, sy, mask = q.popleft()
            for (dx, dy), action in _SLIDE_DIRS:
                nx, ny, new_mask = _slide_collect(sx, sy, dx, dy, mask)
                if nx == sx and ny == sy and new_mask == mask:
                    continue
                ns = (nx, ny, new_mask)
                if ns not in parent:
                    parent[ns] = (sx, sy, mask)
                    action_rec[ns] = action
                    q.append(ns)
                    if new_mask == all_collected:
                        goal = ns
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

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-optimal next SLIDE at the LIVE state.

        Builds the exact same slide graph over ``(x, y, collected_mask)`` as
        ``solve_from``, then computes a distance-to-goal field by backward BFS
        from the all-stars-collected states over the forward slide edges. The
        optimal set = every launch action whose resulting state is exactly one
        slide closer to collecting all stars (``dist == dist(start) - 1``).

        This feeds the base's STOCHASTIC OPTIMAL sampling so each episode takes a
        different shortest slide route. Correct because ``solve_from`` re-plans
        from the LIVE skater cell + star mask, so after the base samples a
        non-canonical (but co-optimal) slide it recovers exactly. O(states+edges),
        no game copy. Returns [] when already solved / no goal (base falls back to
        the plan head)."""
        walls_set, stars, _skater_start, width, height = \
            game._level_data[level_idx]
        stars_list = list(stars)
        n_stars = len(stars_list)
        all_collected = (1 << n_stars) - 1

        level = game.current_level
        skater = level.get_sprites_by_tag("skater")[0]
        skx, sky = skater.x // CELL, skater.y // CELL
        present = {(s.x // CELL, s.y // CELL)
                   for s in level.get_sprites_by_tag("star") if s.is_visible}
        start_mask = 0
        for i, (stx, sty) in enumerate(stars_list):
            if (stx, sty) not in present:
                start_mask |= (1 << i)
        start = (skx, sky, start_mask)
        if start_mask == all_collected:
            return []

        def _slide_collect(sx, sy, dx, dy, mask):
            x, y = sx, sy
            while True:
                nx, ny = x + dx, y + dy
                if (nx, ny) in walls_set or not (0 <= nx < width and 0 <= ny < height):
                    break
                x, y = nx, ny
                for i, (stx, sty) in enumerate(stars_list):
                    if x == stx and y == sty and not (mask >> i) & 1:
                        mask |= (1 << i)
            return x, y, mask

        # Forward BFS: enumerate the reachable slide graph + its edges.
        edges: dict = {}
        goals = set()
        seen = {start}
        q = deque([start])
        while q:
            s = q.popleft()
            sx, sy, mask = s
            if mask == all_collected:
                goals.add(s)
                continue
            out = []
            for (dx, dy), action in _SLIDE_DIRS:
                nx, ny, new_mask = _slide_collect(sx, sy, dx, dy, mask)
                if nx == sx and ny == sy and new_mask == mask:
                    continue                       # no-op slide (blocked)
                ns = (nx, ny, new_mask)
                out.append((action, ns))
                if ns not in seen:
                    seen.add(ns)
                    q.append(ns)
            edges[s] = out

        # Backward BFS for the distance-to-goal field over the forward edges.
        rev: dict = {}
        for s, out in edges.items():
            for _action, ns in out:
                rev.setdefault(ns, []).append(s)
        dist = {g: 0 for g in goals}
        dq = deque(goals)
        while dq:
            s = dq.popleft()
            for p in rev.get(s, ()):  # noqa: B905
                if p not in dist:
                    dist[p] = dist[s] + 1
                    dq.append(p)

        d0 = dist.get(start)
        if d0 is None:
            return []                              # unsolvable (base uses head)
        return [action for action, ns in edges.get(start, ())
                if dist.get(ns, 1 << 30) == d0 - 1]


if __name__ == "__main__":
    sys.exit(IceRinkStarsSolver.main())
