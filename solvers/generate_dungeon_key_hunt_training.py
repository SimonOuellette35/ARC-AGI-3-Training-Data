"""Generate Phase-1 training data for the Dungeon Key Hunt game.

Dungeon Key Hunt (games/dungeon_key_hunt/dungeon_key_hunt.py): navigate a maze,
collect coloured keys (carry up to 3), unlock matching doors, reach the chest.
Each *seed* is a full 7-dungeon game, so each WIN seed yields one multi-level
episode. Grids are procedurally derived from the seed.

Dungeon Key Hunt action set (simple actions only; ACTION7=undo is NOT used):
    ACTION1 (up)  ACTION2 (down)  ACTION3 (left)  ACTION4 (right)
    ACTION5       pick up key on/adjacent to hero, or unlock adjacent door

The expert is a BFS over the game's internal level state keyed by
(hero_x, hero_y, collected_mask, locked_mask, inventory_tuple), in grid (cell)
coordinates. ``solve_from`` reads the LIVE game (hero position, key visibility,
door lock tags, inventory) for its start state; the static geometry (walls, key/
door positions, chest, colours) comes from ``game._level_data[level_idx]``.

NOT reversible: unlocking a door permanently consumes a key (ACTION7 undoes moves
and pickups but NOT door unlocks) and the hero carries at most 3 keys, so a
perturbed detour can strand the level and can't be re-planned from. Recovery is
RESET-mode (``supports_recovery = True``, ``recovery_mode = "reset"``): explore
the prefix, then a single RESET to the dungeon's initial state, then replay the
BFS plan from there.

Usage (run from the repo root):
    python solvers/generate_dungeon_key_hunt_training.py --episodes 1000 \
        --out data/training_multi_level/dungeon_key_hunt
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction                            # noqa: E402
from games.dungeon_key_hunt.dungeon_key_hunt import (       # noqa: E402
    DungeonKeyHunt, _CARRY_LIMIT, CELL)
from solvers.base_solver import BaseSolver                  # noqa: E402

# Grid-unit movement deltas (BFS works in cell coordinates, not pixels).
_MOVE_DELTAS = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}
# Same adjacency order the game uses in _get_adjacent_key / _get_adjacent_door.
_ADJ = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]


class DungeonKeyHuntSolver(BaseSolver):
    game_id = "dungeon_key_hunt"
    # solve_from BFSes over the LIVE abstract state (hx, hy, collected, locked, inv),
    # so it re-plans from any perturbed-but-solvable state -> replan mode. The
    # irreversible mechanics (door unlock consumes a key; carry limit 3) that could
    # strand a detour are handled by the burst-undo rollback (a stranding burst is
    # erased) with RESET as the last resort.
    supports_recovery = True
    recovery_mode = "replan"
    # The irreversible key/door mechanic means the random exploration prefix strands
    # the level fairly often; each strand costs one RESET, so give extra headroom.
    max_resets = 6

    def make_game(self, seed: int):
        return DungeonKeyHunt(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4, 5]

    def set_level(self, game, level_idx: int) -> None:
        # The base drives one game instance across all levels, breaking on the
        # engine's ``_next_level`` flag (which set_level does NOT clear). Clear it
        # so the next level's first action isn't mistaken for an instant solve.
        # on_set_level here removes+re-adds hero/key sprites, so re-setting a
        # level (incl. level 0 after construction) is idempotent.
        game._next_level = False
        game.set_level(level_idx)

    def solve_from(self, game, level_idx: int, seed: int):
        """BFS to the chest from the LIVE hero state on ``level_idx``.

        Reads the live game (hero cell, collected keys via visibility, locked
        doors via the ``door_closed`` tag, held inventory) as the BFS start; the
        static geometry comes from ``game._level_data``. ACTION5 mirrors step():
        priority 1 picks up the first adjacent uncollected key (inventory not
        full), priority 2 unlocks the first adjacent locked door whose colour is
        held. Returns a simple-action plan (moves + ACTION5)."""
        data = game._level_data[level_idx]
        grid = data["grid"]
        width, height = data["width"], data["height"]
        key_positions = [tuple(kp) for kp in data["key_positions"]]
        door_positions = [tuple(dp) for dp in data["door_positions"]]
        chest_pos = tuple(data["chest_pos"])
        colors = data["colors"]
        num_keys = len(key_positions)

        key_set = {kp: i for i, kp in enumerate(key_positions)}
        door_set = {dp: i for i, dp in enumerate(door_positions)}
        wall_set = frozenset(
            (x, y) for y in range(height) for x in range(width) if grid[y][x] == 0
        )

        # --- Live start state (grid coords) ---
        hero = game._hero()
        hx, hy = hero.x // CELL, hero.y // CELL
        collected = 0
        for i in range(num_keys):
            ks = [s for s in game._keys() if f"key_{i}" in s.tags]
            if ks and not ks[0].is_visible:
                collected |= (1 << i)
        locked = 0
        for i in range(len(door_positions)):
            ds = [s for s in game._doors() if f"door_{i}" in s.tags]
            if ds and "door_closed" in ds[0].tags:
                locked |= (1 << i)
        inv = tuple(sorted(game._inventory))
        start_state = (hx, hy, collected, locked, inv)

        parent = {start_state: None}
        action_rec = {start_state: None}
        q = deque([start_state])
        goal = None

        while q and goal is None:
            state = q.popleft()
            hx, hy, coll, locked, inv = state
            inv_list = list(inv)

            def _add(new_state, act):
                if new_state not in parent:
                    parent[new_state] = state
                    action_rec[new_state] = act
                    q.append(new_state)

            # --- Movement ---
            for action, (dx, dy) in _MOVE_DELTAS.items():
                nx, ny = hx + dx, hy + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if (nx, ny) in wall_set:
                    continue
                if (nx, ny) in door_set and (locked >> door_set[(nx, ny)]) & 1:
                    continue
                new_state = (nx, ny, coll, locked, inv)
                _add(new_state, action)
                if (nx, ny) == chest_pos:
                    goal = new_state
                    break
            if goal is not None:
                break

            # --- ACTION5: pick up first adjacent uncollected key (inv not full) ---
            if len(inv) < _CARRY_LIMIT:
                picked = False
                for adx, ady in _ADJ:
                    kpos = (hx + adx, hy + ady)
                    if kpos in key_set:
                        ki = key_set[kpos]
                        if not (coll >> ki) & 1:
                            new_coll = coll | (1 << ki)
                            new_inv = tuple(sorted(inv_list + [colors[ki]]))
                            _add((hx, hy, new_coll, locked, new_inv),
                                 GameAction.ACTION5)
                            picked = True
                            break
                if picked:
                    continue

            # --- ACTION5: unlock the FIRST adjacent locked door, iff its colour
            # is held. The game's ``_get_adjacent_door`` returns that one door and
            # step() then gives up if its colour isn't in the inventory -- it never
            # looks at a second adjacent door. So the scan must stop at the first
            # LOCKED door either way (an already-open door is skipped, as
            # ``_door_at`` requires the ``door_closed`` tag). Scanning on would add
            # unlock edges the engine cannot make, and those phantom shortcuts
            # corrupt every distance computed from this graph.
            inv_set = set(inv)
            for adx, ady in _ADJ:
                dpos = (hx + adx, hy + ady)
                if dpos in door_set:
                    di = door_set[dpos]
                    if (locked >> di) & 1:
                        door_color = colors[di]
                        if door_color in inv_set:
                            new_locked = locked & ~(1 << di)
                            new_inv_list = list(inv_list)
                            new_inv_list.remove(door_color)
                            _add((hx, hy, coll, new_locked, tuple(new_inv_list)),
                                 GameAction.ACTION5)
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
        """Every equally-optimal next action at the LIVE state: each move/ACTION5
        whose successor lies on a shortest reach-the-chest solution
        (``dist == d0 - 1``).

        Distance is over the SAME abstract state
        ``(hx, hy, collected, locked, inv)`` the BFS uses, so an irreversible
        ACTION5 (a key pickup that fills the carry limit, or a door unlock that
        consumes a key) is returned ONLY when it keeps a solution of the same
        minimal length -- never merely because a key/door is adjacent. A BFS per
        candidate first move; the transition mirrors ``solve_from`` exactly
        (pickup takes priority over unlock), so the canonical head is always in
        the set."""
        data = game._level_data[level_idx]
        grid = data["grid"]
        width, height = data["width"], data["height"]
        key_positions = [tuple(kp) for kp in data["key_positions"]]
        door_positions = [tuple(dp) for dp in data["door_positions"]]
        chest_pos = tuple(data["chest_pos"])
        colors = data["colors"]
        num_keys = len(key_positions)

        key_set = {kp: i for i, kp in enumerate(key_positions)}
        door_set = {dp: i for i, dp in enumerate(door_positions)}
        wall_set = frozenset(
            (x, y) for y in range(height) for x in range(width) if grid[y][x] == 0
        )

        def neighbors(state):
            hx, hy, coll, locked, inv = state
            inv_list = list(inv)
            # --- Movement ---
            for action, (dx, dy) in _MOVE_DELTAS.items():
                nx, ny = hx + dx, hy + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if (nx, ny) in wall_set:
                    continue
                if (nx, ny) in door_set and (locked >> door_set[(nx, ny)]) & 1:
                    continue
                yield action, (nx, ny, coll, locked, inv)
            # --- ACTION5: pick up first adjacent uncollected key (inv not full) ---
            if len(inv) < _CARRY_LIMIT:
                for adx, ady in _ADJ:
                    kpos = (hx + adx, hy + ady)
                    if kpos in key_set:
                        ki = key_set[kpos]
                        if not (coll >> ki) & 1:
                            new_inv = tuple(sorted(inv_list + [colors[ki]]))
                            yield GameAction.ACTION5, (hx, hy, coll | (1 << ki),
                                                       locked, new_inv)
                            return                 # pickup takes priority
            # --- ACTION5: unlock the FIRST adjacent locked door, iff its colour is
            # held (see solve_from: the game never looks past that first door, so
            # the scan stops there whether or not the unlock succeeds). ---
            inv_set = set(inv)
            for adx, ady in _ADJ:
                dpos = (hx + adx, hy + ady)
                if dpos in door_set:
                    di = door_set[dpos]
                    if (locked >> di) & 1:
                        if colors[di] in inv_set:
                            new_inv_list = list(inv_list)
                            new_inv_list.remove(colors[di])
                            yield GameAction.ACTION5, (hx, hy, coll,
                                                       locked & ~(1 << di),
                                                       tuple(new_inv_list))
                        return

        def dist_to_goal(s0):
            q = deque([s0])
            dist = {s0: 0}
            while q:
                st = q.popleft()
                if (st[0], st[1]) == chest_pos:
                    return dist[st]
                for _act, ns in neighbors(st):
                    if ns in dist:
                        continue
                    dist[ns] = dist[st] + 1
                    q.append(ns)
            return None

        hero = game._hero()
        hx, hy = hero.x // CELL, hero.y // CELL
        collected = 0
        for i in range(num_keys):
            ks = [s for s in game._keys() if f"key_{i}" in s.tags]
            if ks and not ks[0].is_visible:
                collected |= (1 << i)
        locked = 0
        for i in range(len(door_positions)):
            ds = [s for s in game._doors() if f"door_{i}" in s.tags]
            if ds and "door_closed" in ds[0].tags:
                locked |= (1 << i)
        s0 = (hx, hy, collected, locked, tuple(sorted(game._inventory)))

        d0 = dist_to_goal(s0)
        if not d0:                                 # already solved / unreachable
            return []
        opt: list = []
        for act, ns in neighbors(s0):
            d = dist_to_goal(ns)
            if d is not None and d == d0 - 1:
                opt.append(act)
        return opt


if __name__ == "__main__":
    sys.exit(DungeonKeyHuntSolver.main())
