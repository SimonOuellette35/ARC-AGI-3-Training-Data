"""Generate Phase-1 training data for the g50t game.

Mirrors generate_flood_training.py / generate_maze_training.py, but for the
local g50t game (games/g50t/g50t.py). g50t is a grid navigation / record-replay
puzzle: the player (moved in 6-cell steps by ACTION1..4) must reach the tile
diagonally adjacent to the goal before a scrolling timer bar runs out. The path
is usually blocked by movable blocks that are gated by pressure switches, and
the player alone cannot both hold a switch and reach the goal. ACTION5 commits a
replaying "ghost": it rewinds the player to spawn and spawns a clone that, in all
later phases, replays the exact recorded moves and comes to rest on its final
cell -- so routing a ghost to rest on a switch holds that switch open while the
real player walks past the (now moved) block. Up to (num_doors - 1) ghosts can be
committed.

There is NO reliable hand-written solver for this game, so this generator embeds
a fresh engine-as-oracle solver:

  * a breadth-first navigation search (moves 1..4) over the real engine that,
    from a given state, either reaches the win tile or enumerates the reachable
    cells that press a switch; and
  * a recursive "phase" search that, when the player is trapped, routes a ghost
    to each reachable switch-pressing cell (nav + ACTION5) and recurses.

Every candidate action is applied to a deepcopy of the real engine, and any plan
returned was executed on that clone lineage all the way to a genuine engine win
(game._next_level / GameState.WIN), so the plan is always valid. Because the
display orientation is randomised per set_level (random_rotation_k with no seed),
a plan is only valid on the instance it was found on -- so each level is searched
and recorded on the SAME game instance.

Output schema (identical to maze/flood):

    {
      "game_id": "g50t",
      "levels": [
        {"level_id": 0,
         "observations": [[[c, ...], ...], ...],   # [T, H, W] palette idx
         "actions":      [{"type":"simple","index":k}, ...]},  # length T
        ...
      ]
    }

actions[0] is the RESET that produced obs[0] (each level recorded from a fresh
set_level, so obs[0] is that level's initial frame); actions[i] for i>=1 is the
simple action (index 1..5) that took the agent from obs[i-1] to obs[i].

g50t's RENDERED frames vary across seeds (per-seed random colour remap and a
random display rotation), so each WIN seed yields one complete multi-level
episode: episode_{k:05d}_seed{seed}.json.

Usage (run from the repo root):
    python solvers/generate_g50t_training.py --episodes 1000 \
        --out data/training_multi_level/g50t
"""

from __future__ import annotations

import copy
import sys
from collections import deque
from pathlib import Path

# Repo root (parent of solvers/) so the engine + game imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameAction, GameState  # noqa: E402
from games.g50t.g50t import G50t  # noqa: E402
from solvers.base_solver import BaseSolver  # noqa: E402

GAME_ID = "g50t"
_STEP_GUARD = 4000

# g50t exposes simple actions 1..5 (ACTION1..ACTION5). RESET is 0.
_ACTMAP = {
    1: GameAction.ACTION1,
    2: GameAction.ACTION2,
    3: GameAction.ACTION3,
    4: GameAction.ACTION4,
    5: GameAction.ACTION5,
}
_STEP = 6  # jarvstobjt: the player moves in 6-cell increments.


def _fast_apply(game, action: GameAction):
    """Apply one simple action to `game` (no rendering) and settle it.

    Returns (won, lost). Mirrors the engine's own action loop but stops the
    instant a level solve is queued so we never spill into the next level.
    """
    game._full_reset = False
    game._set_action(ActionInput(id=action))
    guard = 0
    while not game.is_action_complete():
        guard += 1
        if guard > _STEP_GUARD:
            break
        if game._next_level:
            break
        game.step()
    won = game._next_level or game._state == GameState.WIN
    lost = game._state == GameState.GAME_OVER
    return won, lost


def _nav_key(game):
    """Hashable key for the navigation search (moves 1..4 only, no ACTION5).

    Captures everything that makes future navigation dynamics differ: the player
    cell + dead flag, visible block positions, active ghost positions and guards.
    Switch states are a deterministic function of these positions, so they need
    not be keyed separately. The timer is monotonic in move count, so a shorter
    path to the same key dominates -- BFS visits shortest first.

    The phase move-index (len(areahjypvy)) only affects dynamics through ghost
    replay -- ghosts step indexed by it until their recorded path is exhausted,
    after which they rest permanently. So we only key the move-index while a
    ghost is still replaying, clamped to the longest ghost path; once every ghost
    is at rest the index no longer matters and those states correctly merge (this
    is the crucial collapse that keeps the search tractable)."""
    c = game.vgwycxsxjz
    p = c.dzxunlkwxt
    if c.rloltuowth:
        maxlen = max((len(v) for v in c.rloltuowth.values()), default=0)
        move_index = min(len(c.areahjypvy), maxlen)
    else:
        move_index = 0
    return (
        (p.x, p.y, p.amjlzzsesf),
        move_index,
        tuple(sorted((b.x, b.y) for b in c.uwxkstolmf if b.is_visible)),
        tuple(sorted((g.x, g.y) for g in c.rloltuowth)),
        tuple(sorted((g.x, g.y, g.wzgvpxcawd) for g in c.kgvnkyaimw)),
    )


def _pressed_switches(game):
    """Frozenset of indices of the switches the player's current cell presses."""
    c = game.vgwycxsxjz
    p = c.dzxunlkwxt
    return frozenset(i for i, s in enumerate(c.hamayflsib) if c.xvkyljflji(p, s))


def _nav_search(game, target, max_nodes):
    """BFS over moves 1..4 from `game`.

    If `target` is None: explore and return (win_path or None, candidates) where
    `candidates` maps each distinct set-of-pressed-switches to a (cell, path) that
    achieves it -- one representative per switch combination, so a ghost can be
    routed to hold those switches. If `target` is a cell: return (path or None,
    {}), also returning a winning path if the win tile is hit en route. The
    search mutates only clones."""
    start = copy.deepcopy(game)
    q = deque([(start, [])])
    seen = {_nav_key(start)}
    candidates = {}
    nodes = 0
    while q:
        node, path = q.popleft()
        nodes += 1
        if nodes > max_nodes:
            break
        for a in (1, 2, 3, 4):
            child = copy.deepcopy(node)
            won, lost = _fast_apply(child, _ACTMAP[a])
            if lost:
                continue
            npath = path + [a]
            if won:
                return npath, candidates
            cp = child.vgwycxsxjz.dzxunlkwxt
            cell = (cp.x, cp.y)
            if target is not None and cell == target:
                return npath, candidates
            k = _nav_key(child)
            if k in seen:
                continue
            seen.add(k)
            if target is None:
                pressed = _pressed_switches(child)
                if pressed and pressed not in candidates:
                    candidates[pressed] = (cell, npath)
            q.append((child, npath))
    return None, candidates


def solve_plan(game, max_ghosts=None, max_nodes=8000):
    """Structured record-replay solver.

    Recursively: try to navigate the (live) player to the win tile; if trapped,
    route a ghost to each reachable switch-pressing cell (nav + ACTION5) and
    recurse with one fewer ghost budget. Returns a simple-action plan (indices
    1..5) that wins from `game`'s current state, or None. `game` is untouched.
    """
    if max_ghosts is None:
        # Committing a ghost advances the phase counter; when it reaches the
        # number of doors the engine wipes all ghosts, so at most (doors-1) hold.
        max_ghosts = max(0, len(game.vgwycxsxjz.drofvwhbxb) - 1)

    def rec(gm, ghosts_left, used):
        win_path, candidates = _nav_search(gm, None, max_nodes)
        if win_path is not None:
            return win_path
        if ghosts_left <= 0:
            return None
        for pressed, (cell, to_cell) in candidates.items():
            if pressed in used:
                continue
            gm2 = copy.deepcopy(gm)
            for a in to_cell:
                _fast_apply(gm2, _ACTMAP[a])
            won, lost = _fast_apply(gm2, _ACTMAP[5])  # commit ghost
            if lost:
                continue
            sub = rec(gm2, ghosts_left - 1, used | {pressed})
            if sub is not None:
                return to_cell + [5] + sub
        return None

    return rec(copy.deepcopy(game), max_ghosts, frozenset())


class G50tSolver(BaseSolver):
    game_id = GAME_ID
    step_guard = _STEP_GUARD
    # This is a decrementing-timer game (moves drain a budget) and the record-
    # replay ghost commits are one-way, so an exploratory detour can strand the
    # player irrecoverably. ``solve_plan`` also relies on the current live state
    # being a level's fresh start -> RESET recovery: explore the epsilon prefix,
    # then RESET to the fresh start and replay the engine-verified plan. solve_from
    # re-reads the live rotation on the restored instance, so the cached plan stays
    # valid. Timer/ghost one-way-ness needs a couple of resets to recover.
    supports_recovery = True
    recovery_mode = "reset"
    max_resets = 5

    # A plan is valid for any seed sharing the same (level_idx, rotation_k): the
    # game-coordinate dynamics (movement, collision, switches, ghosts) are identical
    # across seeds -- only the colour remap and display rotation differ, and the
    # rotation is captured by rotation_k. So each (level, rotation) is solved once
    # and reused across seeds (mirrors the original module-level _PLAN_CACHE).
    _plan_cache: dict = {}

    def make_game(self, seed: int):
        return G50t(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4, 5]                             # 4 moves + commit ghost

    def solve_from(self, game, level_idx: int, seed: int):
        """Engine-as-oracle record-replay plan (simple-action indices 1..5) for
        the level ``game`` is currently on. Cached by (level_idx, rotation_k) since
        the plan is a pure function of the game-coordinate dynamics + rotation."""
        key = (level_idx, game._rotation_k)
        plan = self._plan_cache.get(key)
        if plan is None:
            plan = solve_plan(game)
            if plan is None:
                return []                                  # unsolvable -> fail episode
            self._plan_cache[key] = plan
        return list(plan)


if __name__ == "__main__":
    sys.exit(G50tSolver.main())
