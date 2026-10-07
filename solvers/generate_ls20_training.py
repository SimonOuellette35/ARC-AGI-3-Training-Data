"""Generate Phase-1 training data for the LS20 game (games/ls20/ls20.py).

Mirrors generate_flood_training.py / generate_maze_training.py but for LS20.
Each episode is one multi-level game (7 levels) written in the schema expected
by the encoder / dynamics stack:

    {
      "game_id": "ls20",
      "levels": [
        {"level_id": 0,
         "observations": [[[c, ...], ...], ...],   # [T, H, W] palette idx frames
         "actions":      [{"type":"simple","index":k}, ...]},  # length T
        ...
      ]
    }

actions[0] is the RESET that produced obs[0]; actions[i] (i>=1) is the simple
action that took the agent from obs[i-1] to obs[i]. Every action is a simple
directional action, index in 0..5 (RESET=0, ACTION1..4 = 1..4).

LS20 is a puzzle where a cursor carries a "held" shape/colour/rotation. Stepping
on cycler tiles (shape / colour / rotation) mutates the held attributes; the goal
is to drive the cursor onto every target pad while the held attributes match that
pad's goal, all within a decrementing step budget (refillable at reset tiles).
The win condition depends on internal state (held attributes + collected flags +
step counter + moving-cycler positions), NOT sprite positions alone, so a
position-keyed BFS fails. The embedded solver below is an engine-as-oracle A*
(deep-copying the real engine per candidate move) whose visited key includes the
full relevant state (cursor, held shape/colour/rotation, collected flags, moving
cycler + pusher positions, remaining reset tiles). It is "phased": it searches for
the next target to collect, then continues from that state, which keeps each
search small enough to be fast.

LS20's ``Ls20`` class takes no seed argument -- out of the box every play is a
fresh random instance (palette, decor, goals and display rotation are randomised
at set_level time from the global RNG and an unseeded instance RNG). This
generator instead makes the augmentation DETERMINISTIC per (seed, level), exactly
like the seeded games (flood, klotski, ...): before recording each level it seeds
the game's instance RNG with ``ls20:<seed>:<level>`` (driving the colour / shape /
goal-rotation draws in ``_randomize_level``) and pins the display rotation to
``random_rotation_k(seed, level)``. So one episode == one seed, each level's board
is reproducible byte-for-byte from (seed, level), and a later live run of the same
seed renders every level at the same orientation and palette.

Because a given (seed, level) is now fixed, a seed whose moving-cycler level the
solver cannot crack within budget is simply SKIPPED and the next seed tried (as
flood skips unsolvable seeds), rather than re-rolling fresh random instances.

LS20 action set: only directional moves are used.
    ACTION1 = up      ACTION2 = down
    ACTION3 = left     ACTION4 = right
(The engine internally re-maps these for the level's random rotation, so the raw
action index we record is exactly what a player would press.)

Usage (run from the repo root):
    python solvers/generate_ls20_training.py --episodes 1000 \
        --out data/training_multi_level/ls20
"""

from __future__ import annotations

import copy
import heapq
import random
import sys
import time
from collections import deque
from pathlib import Path

# Repo root (parent of solvers/) -- that's where the games/ package + engine live.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameAction, GameState  # noqa: E402
from games.ls20.ls20 import Ls20  # noqa: E402
from solvers.base_solver import BaseSolver  # noqa: E402

GAME_ID = "ls20"
_STEP_GUARD = 6000
_CELL = 5

# Raw directional actions (engine re-maps these for the level rotation internally).
_AMAP = {
    1: GameAction.ACTION1,
    2: GameAction.ACTION2,
    3: GameAction.ACTION3,
    4: GameAction.ACTION4,
}

# Sprites that never move / never mutate during a play -- shared (not copied) when
# we deep-copy the engine for search, which makes each copy ~10x cheaper.
_STATIC_TAGS = {
    "ihdgageizm",   # walls
    "xfmluydglp",   # cycler patrol rails
    "eqatonpohu",
    "ghizzeqtoh",
    "hoswmpiqkw",
    "vjotnebuqo",
}


# ── Engine helpers ────────────────────────────────────────────────────────────
def _strip_static(game):
    """Drop the (large) other-level data the engine keeps around so per-move
    deep-copies during search are cheap. Only the current level is retained; the
    engine treats it as the sole level, so a level win sets GameState.WIN instead
    of _next_level -- both are detected as "solved"."""
    lvl = game.current_level
    game._clean_levels = None
    game._levels = [lvl]
    game._current_level_index = 0
    return game


def _static_ids(game):
    """id()->sprite map of never-mutated sprites, to pre-seed deepcopy's memo so
    they are shared rather than copied."""
    return {
        id(s): s
        for s in game.current_level._sprites
        if s.tags and any(t in _STATIC_TAGS for t in s.tags)
    }


def _fast_step(game, action: GameAction):
    """Apply one simple action against the engine WITHOUT rendering. Returns
    (solved, dead)."""
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
    solved = game._next_level or game._state == GameState.WIN
    dead = game._state == GameState.GAME_OVER
    return solved, dead


def _state_key(game):
    """Visited key: the full mutable state that the win/lose logic depends on.

    Cursor cell, held shape/colour/rotation index, collected-target flags, the
    positions of the moving cyclers and pusher bars, and which reset tiles remain.
    (Step counter and lives are deliberately excluded: within the shortest path to
    a given key they are always at their best, so keeping the first arrival is
    safe -- and every search edge uses the real engine, so any returned plan is
    valid regardless.)"""
    cyclers = tuple((c._sprite.x, c._sprite.y) for c in game.wsoslqeku)
    pushers = tuple((p.sprite.x, p.sprite.y) for p in game.hasivfwip)
    resets = tuple(sorted(
        (s.x, s.y) for s in game.current_level.get_sprites_by_tag("npxgalaybz")
    ))
    return (
        game.gudziatsk.x, game.gudziatsk.y,
        game.fwckfzsyc, game.hiaauhahz, game.cklxociuu,
        tuple(game.lvrnuajbl),
        cyclers, pushers, resets,
    )


def _target_dmaps(game):
    """Static-wall BFS distance (in cells) from each target pad, for the A*
    heuristic."""
    walls = {(s.x, s.y) for s in game.current_level.get_sprites_by_tag("ihdgageizm")}
    w = game.qlgmdayuo
    h = game.drtdqwdbc
    dmaps = []
    for t in game.plrpelhym:
        d = {(t.x, t.y): 0}
        dq = deque([(t.x, t.y)])
        while dq:
            x, y = dq.popleft()
            for dx, dy in ((_CELL, 0), (-_CELL, 0), (0, _CELL), (0, -_CELL)):
                nx, ny = x + dx, y + dy
                if (nx, ny) in d or nx < 0 or ny < 0 or nx >= w or ny >= h \
                        or (nx, ny) in walls:
                    continue
                d[(nx, ny)] = d[(x, y)] + 1
                dq.append((nx, ny))
        dmaps.append(d)
    return dmaps


def _heuristic(game, dmaps, k_mismatch=6):
    """Estimated cost to collect the nearest still-uncollected target: cell
    distance to the pad plus a penalty per mismatched held attribute (each needs a
    cycler detour)."""
    best = 0
    cx, cy = game.gudziatsk.x, game.gudziatsk.y
    for i in range(len(game.plrpelhym)):
        if game.lvrnuajbl[i]:
            continue
        goal = (game.ldxlnycps[i], game.yjdexjsoa[i], game.ehwheiwsk[i])
        d = dmaps[i].get((cx, cy), 40)
        mism = ((game.fwckfzsyc != goal[0])
                + (game.hiaauhahz != goal[1])
                + (game.cklxociuu != goal[2]))
        e = d + mism * k_mismatch
        best = e if best == 0 else min(best, e)
    return best


def _phase_search(game, sids, dmaps, node_cap, time_cap, weight):
    """A* until one more target is collected (or the whole level is solved).

    Returns (subplan, resulting_game, nodes). subplan is a list of raw action
    indices (1..4). Deep-copies the engine per candidate move (memo-sharing the
    static sprites)."""
    base = sum(game.lvrnuajbl)
    cnt = 0
    pq = [(0, 0, cnt, game, [])]
    seen = {_state_key(game): 0}
    nodes = 0
    t0 = time.time()
    while pq:
        _f, g, _, gm, path = heapq.heappop(pq)
        nodes += 1
        if nodes > node_cap or time.time() - t0 > time_cap:
            return None, None, nodes
        for a in (1, 2, 3, 4):
            child = copy.deepcopy(gm, dict(sids))
            solved, dead = _fast_step(child, _AMAP[a])
            if dead:
                continue
            if solved or sum(child.lvrnuajbl) > base:
                return path + [a], child, nodes
            key = _state_key(child)
            ng = g + 1
            if key in seen and seen[key] <= ng:
                continue
            seen[key] = ng
            cnt += 1
            heapq.heappush(
                pq,
                (ng + weight * _heuristic(child, dmaps), ng, cnt, child, path + [a]),
            )
    return None, None, nodes


def _replay_on_real(root, plan) -> bool:
    """Does ``plan`` actually win on an UNSTRIPPED copy of ``root``?

    The search runs on ``_strip_static(deepcopy(root))``, and that stripped game
    does NOT behave like the real one: setting ``_levels=[lvl]`` /
    ``_current_level_index=0`` is enough to change the outcome on its own
    (nulling ``_clean_levels`` alone is not). Observed on seed 0 level 4: the two
    worlds are byte-identical for 50 moves, then the real game kills the player on
    move 51 while the stripped sim walks on and "wins" at 52. So a plan that the
    search considers engine-verified can be invalid in the engine the recorder
    actually drives, and the level fails at replay -- silently, because
    ``solve_episode`` swallows it and just drops the seed.

    This replays the candidate on an unstripped copy, mirroring
    ``BaseSolver.drive`` exactly (same break conditions, and rendering after every
    engine step, since this engine advances animations on render). ``root`` is not
    mutated. Turning a bad plan into an honest ``None`` costs one extra replay and
    makes the failure legible instead of seed loss."""
    game = copy.deepcopy(root)
    for idx in plan:
        game._full_reset = False
        game._set_action(ActionInput(id=_AMAP[int(idx)]))
        guard = 0
        while not game.is_action_complete():
            guard += 1
            if guard > _STEP_GUARD or game._next_level:
                break
            game.step()
            game.camera.render(game.current_level.get_sprites())
        if game._state == GameState.GAME_OVER:
            return False
        if game._next_level or game._state == GameState.WIN:
            return True
    return bool(game._next_level) or game._state == GameState.WIN


def solve(root, node_cap=80000, time_cap=25, weight=4, verify=True):
    """Return a full simple-action plan (list of raw action indices 1..4) that
    solves the level `root` is currently on, or None. `root` is NOT mutated.

    The plan is verified on the REAL (unstripped) engine before being returned --
    see `_replay_on_real` for why the search's own "engine-verified" is not
    sufficient. ``verify=False`` skips that check (useful to measure how often the
    stripped sim and the real engine disagree).

    Budget note: every genuine per-phase clean solve observed across all 7 levels
    finishes in <530 nodes / <1.5s, so these caps never bind for generation (which
    is optimal-only, see ``Ls20Solver.__init__``). They stay generous for the LIVE
    recovery path: ``solve_from`` re-planning from an arbitrary perturbed state
    during eval gets the full budget to find a genuinely harder route."""
    plan = _search_plan(root, node_cap, time_cap, weight)
    if plan is None or not verify:
        return plan
    return plan if _replay_on_real(root, plan) else None


def _search_plan(root, node_cap, time_cap, weight):
    """The phase-by-phase A* itself, on the stripped (fast) copy. Callers should
    go through `solve`, which validates the result against the real engine."""
    game = _strip_static(copy.deepcopy(root))
    sids = _static_ids(game)
    dmaps = _target_dmaps(game)
    plan = []
    while sum(game.lvrnuajbl) < len(game.plrpelhym):
        sub, nxt, _nodes = _phase_search(game, sids, dmaps, node_cap, time_cap, weight)
        if sub is None:
            return None
        plan += sub
        game = nxt
        if game._next_level or game._state == GameState.WIN:
            return plan
    return plan if (game._next_level or game._state == GameState.WIN) else None


# ── BaseSolver wiring ─────────────────────────────────────────────────────────
class Ls20Solver(BaseSolver):
    game_id = GAME_ID
    step_guard = _STEP_GUARD
    supports_recovery = True
    # RESET-recovery, NOT replan. ``solve()`` is a deep-copy-heavy engine-oracle A*
    # that plans from a level's DETERMINISTIC fresh start; it cannot reliably (or
    # cheaply) re-plan from an arbitrary perturbed state, and ls20's tight decrementing
    # step budget + 3 lives means an exploratory detour usually strands the cursor.
    # So reset mode is the right shape: the exploration prefix runs first (recorded as
    # recovery-training data), then ONE RESET restores the (seed,level) fresh start and
    # the engine-verified optimal plan replays from there. Reset mode also, for free,
    # (a) disables the perturbation bursts -- they only fire in replan mode -- and
    # (b) skips the per-step stochastic-optimal lookahead (it drives every alternative
    # move + a full solve_from each; ~180 A* solves for a 60-move level here), both of
    # which were prohibitively slow for this solver. So the exploratory phase is kept,
    # bursts are off, and generation stays fast.
    recovery_mode = "reset"
    max_resets = 5

    def make_game(self, seed: int):
        self._seed = seed
        return Ls20(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]                                # directional moves only

    def set_level(self, game, level_idx: int) -> None:
        """Make level ``level_idx`` DETERMINISTIC in (seed, level_idx): seed the
        instance RNG (colours / shapes / goal rotations, consumed by
        on_set_level -> _randomize_level) BEFORE set_level; the display rotation is
        drawn by AugmentedGame.set_level as ``random_rotation_k(seed, level_idx)``."""
        game._rng = random.Random(f"ls20:{self._seed}:{level_idx}")
        game.set_level(level_idx)

    def solve_from(self, game, level_idx: int, seed: int):
        """Engine-as-oracle A* plan (raw directional indices 1..4) that solves the
        level ``game`` is currently on, or [] if it can't be cracked in budget."""
        plan = solve(game)
        return list(plan) if plan else []


if __name__ == "__main__":
    sys.exit(Ls20Solver.main())
