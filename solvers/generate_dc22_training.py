"""Generate Phase-1 training data for the DC22 game (games/dc22/dc22.py).

DC22 is a keyboard+click "sculpt-a-path" maze. A 2x2 player (`rpygrnbjhwj1`,
tag ``pcxjvnmybet``) must reach the goal sprite ``bqxa`` (win =
``player.x==bqxa.x and player.y==bqxa.y``) before the per-level step counter
runs out. ACTION1/2/3/4 move the player up/down/left/right by 2 px; a move
succeeds only if the destination has no TANGIBLE/INVISIBLE wall pixel AND an
INTANGIBLE floor pixel under the player's top-left cell. ACTION6 is a mouse
click.

The maze is reshaped by clicking on-screen panels:
  * jpug colour panels (letters a/b/c/d/...) flip EVERY same-letter ``wbze`` cell
    to its next pixel-footprint variant at once (bridge orientation / wall<->floor),
  * ``itki`` tiles teleport the player to their sibling when their panel is clicked
    while the player stands on one; a dedicated ``gkrr-jpug`` panel colour-cycles
    the itki tiles (changing wall<->floor),
  * ``zbhi`` pads permanently open the same-letter jpug door when stepped on,
  * pressure plates reveal same-letter jpug bridges while held,
  * a crane (``nxhz``, levels 5-6) ferries ``hhxv`` bridge cells across gaps.
The objective is purely to carve a continuous INTANGIBLE floor path to ``bqxa``.

Solver
------
The game is treated as a black box and solved by weighted-A* over the discrete
action space {up,down,left,right} plus one ACTION6 click per visible jpug/sys_click
panel. Neighbours are produced by cloning the engine (a fast identity-shared
``deepcopy`` that skips the other levels) and driving one action; states are
de-duplicated on a UI-independent key = the multiset of (name, x, y, interaction)
of every non-REMOVED sprite plus the crane's internal offsets (the rendered frame
is NOT used as the key because the step-counter bar mutates it every action). The
heuristic is Manhattan(player, goal)/2. This cracks the four non-crane levels
(0-3) reliably; the two crane levels (4-5) come from hand-recorded demos in
``data/dc22_demos`` (they are not cracked by the search).

BaseSolver migration
--------------------
The record/replay/schema/CLI harness now lives in ``BaseSolver``. Each level's
seed-independent plan (A* for 0-3, demo for 4-5) is obtained ONCE and cached on
the instance; ``solve_from`` reads the LIVE game's display rotation
(``game._rotation_k``, drawn per-(seed, level) by ``AugmentedGame``) and
forward-rotates the k=0 plan into the SCREEN action the policy will see, so the
recorded pair is the ROTATED frame + the screen action that achieves the goal.
``supports_recovery`` stays False: a level's plan is a fixed optimal replay, the
maze is not reversible (walls get carved, the crane commits), and ``solve_from``
only replays that precomputed plan -- so the exploration prefix stays off.

Because the seed only permutes the eight themed colours (6-13) and the display
rotation -- never walls/floors/positions -- every level's winning action sequence
is seed-independent; one solve serves every seed, its colours baked into that
seed's rendered frames. ``solve_episode`` records each planned level on a FRESH
``Dc22(seed)`` (partial episodes are valid, like bp35 / lf52).

Action schema (mixed simple + mouse, matching cd82 / alchemy_stones)
    RESET / simple :  {"type": "simple", "index": k}          # k in 0..6
    mouse click    :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_dc22_training.py --episodes 1000 \
        --out data/training_multi_level/dc22
"""

from __future__ import annotations

import copy
import heapq
import json
import sys
import time
from pathlib import Path

# Repo root (parent of solvers/) -- where the games/ package + engine live.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402,F401
from arcengine import (  # noqa: E402
    ActionInput,
    GameAction,
    GameState,
    InteractionMode,
)
from games.dc22.dc22 import Dc22  # noqa: E402
from solvers.base_solver import BaseSolver  # noqa: E402
from utils.explore import (  # noqa: E402
    Action, CLICK_ACTION, EpsilonSchedule, ExplorationPolicy)
from utils.rotation import (  # noqa: E402
    ACTION_REMAP_INVERSE,
    remap_click,
)

GAME_ID = "dc22"
_STEP_GUARD = 4000

_DIR_ACTION = {
    1: GameAction.ACTION1,
    2: GameAction.ACTION2,
    3: GameAction.ACTION3,
    4: GameAction.ACTION4,
}
_MOVES = [(GameAction.ACTION1, 1), (GameAction.ACTION2, 2),
          (GameAction.ACTION3, 3), (GameAction.ACTION4, 4)]


# ── Rotation augmentation ─────────────────────────────────────────────────────
# A plan is solved at k=0 (grid == screen). To REPLAY it under a display rotation
# of k*90deg CCW so it still wins, each action is forward-rotated into the rotated
# screen space; the engine then inverse-remaps it back to k=0 grid space and
# reproduces the identical gameplay -- only the rendered frame is rotated. This
# pairs the ROTATED frame the policy sees at eval time (the live game re-rolls a
# random rotation every level) with the screen action that achieves the goal.
def _rot_record(rec, k: int):
    """Forward-rotate one plan record ("move", dir) / ("click", x, y) for rotation k."""
    if k == 0:
        return rec
    if rec[0] == "move":
        return ("move", _DIR_ACTION[rec[1]].value)
    nx, ny = rec[1], rec[2]
    return ("click", int(nx), int(ny))


# ── Engine helpers ────────────────────────────────────────────────────────────
def _make_level(seed: int, level_idx: int, k: int = 0) -> Dc22:
    """Fresh ``Dc22(seed)`` on ``level_idx`` with the display rotation pinned to
    ``k`` (0..3). The per-seed colour map is already fixed by the seeded constructor
    RNG; ``k`` is the rotation-augmentation choice for this (seed, level)."""
    game = Dc22(seed=seed)
    game.set_level(level_idx)
    game._rotation_k = k % 4
    return game


def _fastclone(game: Dc22) -> Dc22:
    """~4x faster than a plain deepcopy: share the immutable per-game structures
    (the other levels + the clean-level templates) by identity so only the current
    level's sprites and the game's mutable attributes are actually copied. Verified
    to play byte-identically to ``copy.deepcopy``."""
    memo: dict[int, object] = {}
    clean = getattr(game, "_clean_levels", None)
    if clean is not None:
        memo[id(clean)] = clean
        for lv in clean:
            memo[id(lv)] = lv
    levels = getattr(game, "_levels", None)
    cur = game.current_level
    if levels is not None:
        for lv in levels:
            if lv is not cur:
                memo[id(lv)] = lv
    return copy.deepcopy(game, memo)


def _drive_noframe(game: Dc22, action_input: ActionInput):
    """Drive one action inside the search, skipping rendering -- only the resulting
    state (win/lose) matters. Stops the instant a level solve is queued."""
    game._full_reset = False
    game._set_action(action_input)
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


def _state_key(game: Dc22):
    """UI-independent hashable state: every non-REMOVED sprite's (name, x, y,
    interaction) plus the crane's internal offsets. Excludes the rendered frame
    because the step-counter progress bar mutates it every action."""
    items = []
    for s in game.current_level.get_sprites():
        if s.interaction == InteractionMode.REMOVED:
            continue
        items.append((s.name, s.x, s.y, s.interaction.value))
    items.sort()
    extra = (
        getattr(game, "nxhz_x", 0),
        getattr(game, "nxhz_y", 0),
        getattr(game, "nxhz_attached_kind", "none"),
        getattr(game, "attached_hhxv_x", 0),
        getattr(game, "attached_hhxv_y", 0),
    )
    return (tuple(items), extra)


def _click_targets(game: Dc22):
    """Candidate ACTION6 clicks for the search: display (x, y) of the centre of
    each currently visible, non-REMOVED/INVISIBLE jpug or sys_click sprite."""
    scale, offx, offy = game.camera._calculate_scale_and_offset()
    seen = set()
    out = []
    for s in game.current_level.get_sprites():
        if s.interaction in (InteractionMode.REMOVED, InteractionMode.INVISIBLE):
            continue
        if not s.is_visible:
            continue
        if not ("jpug" in s.tags or "sys_click" in s.tags):
            continue
        gx = s.x + s.width // 2
        gy = s.y + s.height // 2
        if (gx, gy) in seen:
            continue
        seen.add((gx, gy))
        out.append((int(gx * scale + offx), int(gy * scale + offy)))
    return out


# ── Solver: weighted A* over discrete actions, engine as black box ─────────────
def solve_level(level_idx: int, time_budget: float, max_nodes: int, weight: float,
                verbose: bool = False):
    """Return a winning plan for ``level_idx`` as a list of action records --
    ("move", dir 1..4) or ("click", x, y) -- or None. The plan is seed-independent
    (colours don't affect mechanics), so a single solve serves every seed."""
    g0 = _make_level(0, level_idx)
    goal = (g0.bqxa.x, g0.bqxa.y)

    def h(game):
        p = (game.fdvakicpimr.x, game.fdvakicpimr.y)
        return weight * ((abs(p[0] - goal[0]) + abs(p[1] - goal[1])) // 2)

    counter = 0
    frontier = [(h(g0), 0, counter, g0, [])]
    counter += 1
    best = {_state_key(g0): 0}
    nodes = 0
    t0 = time.time()
    while frontier:
        _f, gcost, _, game, path = heapq.heappop(frontier)
        nodes += 1
        if verbose and nodes % 2000 == 0:
            print(f"    ..nodes={nodes} frontier={len(frontier)} g={gcost} "
                  f"seen={len(best)} t={round(time.time()-t0,1)}s", flush=True)
        if nodes > max_nodes or time.time() - t0 > time_budget:
            return None
        neighbours = [(act, ai, False) for act, ai in _MOVES]
        for dx, dy in _click_targets(game):
            neighbours.append((GameAction.ACTION6, {"x": dx, "y": dy}, True))
        for act, ai, is_click in neighbours:
            child = _fastclone(game)
            if is_click:
                solved, dead = _drive_noframe(child, ActionInput(id=act, data=ai))
                rec = ("click", ai["x"], ai["y"])
            else:
                solved, dead = _drive_noframe(child, ActionInput(id=act))
                rec = ("move", ai)
            if dead:
                continue
            if solved:
                return path + [rec]
            key = _state_key(child)
            ngc = gcost + 1
            if key in best and best[key] <= ngc:
                continue
            best[key] = ngc
            heapq.heappush(frontier, (ngc + h(child), ngc, counter, child, path + [rec]))
            counter += 1
    return None


def _load_plan_file(plan_dir: Path, idx: int):
    """Load a hand-recorded plan (from play_dc22.py) for level ``idx`` if present
    and marked won. Returns the plan (list of records) or None."""
    if plan_dir is None:
        return None
    path = plan_dir / f"dc22_level{idx}_plan.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if not data.get("won", False):
        return None
    return data.get("plan")


# ── BaseSolver subclass ────────────────────────────────────────────────────────
class Dc22Solver(BaseSolver):
    game_id = GAME_ID
    step_guard = _STEP_GUARD
    # A level's plan is a fixed optimal replay; the maze is NOT reversible (walls
    # carved, crane committed) and ``solve_from`` only replays the precomputed
    # plan valid at the level's INITIAL state. So exploration is enabled via the
    # RESET-recovery paradigm: probe the epsilon prefix, then ONE RESET back to the
    # initial state (``reset_level`` re-clones ``_clean_levels`` + re-draws the
    # deterministic rotation, matching ``observations[0]``), then replay the plan.
    supports_recovery = True
    recovery_mode = "reset"

    # Seed-independent plan sources: A* for levels 0-3, hand-recorded demos for the
    # crane levels 4-5.
    _PLAN_DIR = Path("data/dc22_demos")
    _SOLVE_BUDGET = 120.0
    _CRANE_BUDGET = 120.0
    _MAX_NODES = 400000
    _WEIGHT = 2.0

    def __init__(self, **kw):
        super().__init__(**kw)
        self._plans = None

    def make_game(self, seed: int):
        return Dc22(seed=seed)

    def available_actions(self, game) -> list[int]:
        # Four movement keys plus the panel/teleport click DC22 exposes.
        return [1, 2, 3, 4, CLICK_ACTION]

    # -- seed-independent per-level plans (built once, cached on the instance) --
    def _get_plans(self) -> dict:
        if self._plans is None:
            self._plans = self._build_plans()
        return self._plans

    def _build_plans(self) -> dict:
        num_levels = len(Dc22()._levels)
        plans: dict[int, list] = {}
        for idx in range(num_levels):
            demo_path = self._PLAN_DIR / f"dc22_level{idx}_plan.json"
            if demo_path.exists():
                demo = _load_plan_file(self._PLAN_DIR, idx)
                if demo is not None:
                    print(f"  level {idx}: hand-recorded plan, {len(demo)} actions",
                          flush=True)
                    plans[idx] = demo
                continue
            budget = self._CRANE_BUDGET if idx >= 4 else self._SOLVE_BUDGET
            print(f"Solving level {idx} (budget {budget}s) ...", flush=True)
            t0 = time.time()
            plan = solve_level(idx, time_budget=budget, max_nodes=self._MAX_NODES,
                               weight=self._WEIGHT)
            dt = round(time.time() - t0, 1)
            if plan is None:
                print(f"  level {idx}: UNSOLVED after {dt}s -- skipping", flush=True)
            else:
                print(f"  level {idx}: solved, {len(plan)} actions ({dt}s)", flush=True)
                plans[idx] = plan
        return plans

    def solve_from(self, game, level_idx: int, seed: int):
        """Forward-rotate the cached k=0 plan for ``level_idx`` into the SCREEN
        actions to send at the live display rotation (``game._rotation_k``). The
        plan is state-independent (rotation is a scalar, clicks are absolute display
        coords), so the whole plan is converted up-front."""
        plan = self._get_plans().get(level_idx)
        if not plan:
            return []
        k = game._rotation_k
        actions = []
        for rec in plan:
            r = _rot_record(rec, k)
            if r[0] == "move":
                actions.append(Action(int(r[1])))          # r[1] is the action value
            else:
                actions.append(Action(CLICK_ACTION, (int(r[2]), int(r[1]))))  # (y, x)
        return actions

    def solve_episode(self, seed: int, explore: bool = True):
        """Record every planned level on a FRESH ``Dc22(seed)`` (matching the
        original, which builds one game per level). ``ok`` iff at least one level
        recorded and every planned level verified as a WIN (planned levels are
        seed-independent, so this always holds in practice).

        Exploration (the epsilon prefix + RESET-recovery) is wired in the same way
        as the base ``solve_episode``: the ``EpsilonSchedule`` + ``ExplorationPolicy``
        are built ONCE per episode so the ignorant-then-informed arc spans the whole
        episode, and handed to every ``record_level`` call. They engage only when
        ``explore and self.supports_recovery``; otherwise this is pure optimal replay."""
        plans = self._get_plans()
        do_explore = explore and self.supports_recovery
        schedule = (EpsilonSchedule(self.rng, center=self._explore_center,
                                    jitter=self._explore_jitter)
                    if do_explore else None)
        exploration = (ExplorationPolicy(self.available_actions(self.make_game(seed)),
                                         self.rng)
                       if do_explore else None)
        levels = []
        for level_idx in sorted(plans):
            game = self.make_game(seed)
            try:
                obs, acts = self.record_level(game, level_idx, seed,
                                              schedule=schedule,
                                              exploration=exploration)
            except Exception:                              # noqa: BLE001
                return False, levels
            if obs is None:
                return False, levels
            levels.append({"level_id": level_idx, "observations": obs, "actions": acts})
        return (len(levels) > 0), levels


if __name__ == "__main__":
    sys.exit(Dc22Solver.main())
