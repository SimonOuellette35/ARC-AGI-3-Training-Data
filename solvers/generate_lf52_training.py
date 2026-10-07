"""Generate Phase-1 training data for the LF52 game (games/lf52/lf52.py).

LF52 is a *peg-solitaire-with-a-conveyor* puzzle played over ten hand-designed
levels on an 8x8-ish grid.  The pieces:

* ``fozwvlovdui`` -- **pegs** (the peg-solitaire stones).  Some carry a colour
  suffix (``_red`` / ``_blue``); the per-seed palette is randomised but the
  suffixes are fixed by the level.
* ``hupkpseyuim`` -- the board / hole background (the playable region).
* ``hupkpseyuim2`` -- a **transporter** that rides the rails.
* ``kraubslpehi`` -- **rails** the transporter slides along.
* ``dgxfozncuiz`` -- inert **target markers** you may jump over.

Mechanics (verified against the engine)
---------------------------------------
* **Move** (ACTION1..4 = up / down / left / right): every transporter slides one
  cell along the rails in that direction, carrying any peg sharing its cell.
  (Also pans the board a few pixels -- a purely cosmetic scroll that does not
  change any piece's grid coordinate.)
* **Jump** (two ACTION6 clicks: the peg, then the landing cell two away): the
  selected peg hops two cells over an occupied neighbour into an empty hole.
  The jumped peg is **captured** (removed) *only* when it is a non-blue peg of
  the *same* name as the mover; otherwise the jump merely repositions the mover.
* A level is **won** the instant the number of non-blue pegs drops to 1
  (to 2 for levels 6 and 7, whose boards cannot reach a single peg).

Solver
------
The logical board (peg layout, rails, targets) is *identical across seeds* -- a
seed only re-rolls the colour palette and a per-level display rotation.  So each
level is solved **once**, offline, by a best-first search producing a list of
seed-independent logical primitives:

    ("M", (dx, dy))            # a directional move, game-space
    ("J", src_cell, land_cell) # a jump: click src, then click land
    ("C", cell)                # a single click (hand-recorded, play_lf52.py)

**Camera-awareness.**  The live board is only a 64x64 window onto a wider grid,
and clicks *outside* that window are invalid.  Some levels (2, 3) have pegs on
the far side of the board only clickable after scrolling the view (a carried-peg
move pans the board).  So the search is two-tier (:func:`solve_level`): a fast
**offset-blind** search (:func:`_solve_fast`) produces a candidate; if it replays
with every click inside the window it is kept, otherwise the level is re-solved
by a slower **camera-aware** search (:func:`_solve_camera_aware`) that drives the
real engine per node so the scroll offset is exact.

Coverage: the search reliably solves **levels 1-5** (0-4); levels 6-9 come from
hand-recorded demos in ``--demo-dir`` (``play_lf52.py``), which override the
solver per level.  Level 10 (idx 9) remains unsolved and is skipped.  Partial
episodes are valid (dc22 / bp35 style).

BaseSolver migration
--------------------
The record/replay/schema/CLI harness now lives in ``BaseSolver``. Because a
jump's click coordinate keys on the **live scroll offset** -- which the carried-
peg moves change mid-plan -- ``solve_from`` replays the cached logical plan on a
deepcopy of the LIVE game, reading each step's offset + rotation to resolve the
concrete SCREEN-space clicks/moves, then the base drives those exact actions on
the real game (identical actions -> identical offsets) to capture frames. The
base ``drive`` already renders each step (which is how the offset settles) so it
is used as-is with a raised ``step_guard``. ``supports_recovery`` stays False: a
non-capturing hop can strand pegs (the board can become unsolvable), and
``solve_from`` only replays a precomputed plan.

Action schema (mixed simple + mouse, matching cd82 / cn04 / gp01 / bp35)
    RESET / move   :  {"type": "simple", "index": k}                      # k in 0..5
    mouse click    :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_lf52_training.py --episodes 1000 \
        --out data/training_multi_level/lf52
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

import numpy as np  # noqa: E402
from arcengine import ActionInput, GameAction, GameState  # noqa: E402
from games.lf52.lf52 import Lf52  # noqa: E402
from solvers.base_solver import BaseSolver  # noqa: E402
from utils.explore import (  # noqa: E402
    Action, CLICK_ACTION, EpsilonSchedule, ExplorationPolicy)
from utils.rotation import remap_action, remap_click  # noqa: E402

GAME_ID = "lf52"
NUM_LEVELS = 10
_RESET_ACTION = {"type": "simple", "index": int(GameAction.RESET.value)}
_ACTION6 = int(GameAction.ACTION6.value)
_STEP_GUARD = 6000

# game-space directional deltas <-> the ACTION that produces them.
_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
_DELTA_TO_ACTION = {
    (0, -1): GameAction.ACTION1,
    (0, 1): GameAction.ACTION2,
    (-1, 0): GameAction.ACTION3,
    (1, 0): GameAction.ACTION4,
}
# Levels whose win condition is "two non-blue pegs left" instead of one.
_WIN_AT_TWO = {6, 7}


# ── Engine state helpers (operate on the inner game `game.ikhhdzfmarl`) ────────
def _eng(game):
    return game.ikhhdzfmarl


def _peg_objs(e):
    return [p for p in e.hncnfaqaddg.kdsncymzyeb if "fozwvlovdui" in p.yrxvacxlgrf]


def _peg_map(e):
    return {(p.grid_x, p.grid_y): p.yrxvacxlgrf for p in _peg_objs(e)}


def _transporters(e):
    return tuple(sorted((p.grid_x, p.grid_y)
                        for p in e.hncnfaqaddg.kdsncymzyeb
                        if p.yrxvacxlgrf == "hupkpseyuim2"))


def _nonblue(peg_map):
    return sum(1 for n in peg_map.values() if "blue" not in n)


def _spread(peg_map):
    """Sum of pairwise Manhattan distances between the non-blue pegs -- lower
    means more clustered, hence closer to a capturing jump."""
    cells = [c for c, n in peg_map.items() if "blue" not in n]
    total = 0
    for i in range(len(cells)):
        ax, ay = cells[i]
        for j in range(i + 1, len(cells)):
            bx, by = cells[j]
            total += abs(ax - bx) + abs(ay - by)
    return total


def _hval(peg_map):
    """Best-first priority: minimise non-blue peg count, then clustering."""
    return _nonblue(peg_map) * 10000 + _spread(peg_map)


def _offset(e):
    return tuple(e.hncnfaqaddg.cdpcbbnfdp)


def _state_key(e):
    # The scroll offset is part of the state: two boards with identical pegs but
    # different scroll positions expose different cells to the camera.
    return (tuple(sorted(_peg_map(e).items())), _transporters(e), _offset(e))


def _cell_onscreen(e, cell):
    """True iff ``cell``'s centre currently falls inside the real 64x64 camera
    window at the live scroll offset -- i.e. it can actually be clicked."""
    grid = e.hncnfaqaddg
    ox, oy = grid.cdpcbbnfdp
    tw, th = grid.tile_size
    px = ox + cell[0] * tw + tw // 2
    py = oy + cell[1] * th + th // 2
    return 0 <= px < 64 and 0 <= py < 64


def _onscreen_jumps(e):
    """(src, (dx, dy), land) for every engine-legal jump whose src peg AND land
    hole both fall inside the camera window."""
    out = []
    for p in _peg_objs(e):
        src = (p.grid_x, p.grid_y)
        if not _cell_onscreen(e, src):
            continue
        for d in _DIRS:
            if e.qikmikecdf(src, d):
                land = (src[0] + 2 * d[0], src[1] + 2 * d[1])
                if _cell_onscreen(e, land):
                    out.append((src, d, land))
    return out


def _click_data(game, cell):
    """ACTION6 data (screen == game space at k=0) that clicks board ``cell`` at
    the live scroll offset."""
    grid = _eng(game).hncnfaqaddg
    ox, oy = grid.cdpcbbnfdp
    tw, th = grid.tile_size
    return {"x": int(ox + cell[0] * tw + tw // 2),
            "y": int(oy + cell[1] * th + th // 2)}


def _search_step(game, action_input):
    """Apply one action to the real engine during search, draining the animation
    frames so the scroll offset settles. Returns (solved, dead)."""
    game._full_reset = False
    game._set_action(action_input)
    guard = 0
    while not game.is_action_complete():
        guard += 1
        if guard > _STEP_GUARD or game._next_level:
            break
        game.step()
        game.camera.render(game.current_level.get_sprites())
    solved = game._next_level or game._state == GameState.WIN
    dead = game._state == GameState.GAME_OVER
    return solved, dead


# ── Fast solver (offset-blind: mutates a deepcopied grid, no rendering) ────────
def _legal_jumps(e):
    """(src_cell, (dx, dy)) for every jump the engine deems legal from a peg."""
    out = []
    for p in _peg_objs(e):
        src = (p.grid_x, p.grid_y)
        for d in _DIRS:
            if e.qikmikecdf(src, d):
                out.append((src, d))
    return out


def _apply_jump(e, src, d):
    """Mutate the logical grid exactly as ``cfilhtifcb`` would."""
    grid = e.hncnfaqaddg
    dx, dy = d
    mid = (src[0] + dx, src[1] + dy)
    land = (src[0] + 2 * dx, src[1] + 2 * dy)
    sel = next(p for p in _peg_objs(e) if (p.grid_x, p.grid_y) == src)
    cap = None
    for q in grid.ijpoqzvnjt(mid[0], mid[1]):
        if "fozwvlovdui" in q.yrxvacxlgrf and "blue" not in q.yrxvacxlgrf:
            cap = q
            break
    if cap is not None and cap.yrxvacxlgrf == sel.yrxvacxlgrf:
        grid.faretcdgmc(cap)
    before = sel.abvcoxnskr
    sel.porbskbertu, sel.igmurzpcaud = land
    grid.ievgclzwme(sel, before, sel.abvcoxnskr)


def _solve_fast(level_idx, node_cap=150000, time_cap=100.0):
    """Original offset-blind best-first search over jumps + moves. Returns a plan
    (possibly containing off-screen clicks) or ``None``."""
    internal = level_idx + 1
    win_at = 2 if internal in _WIN_AT_TWO else 1
    start = time.time()
    root = Lf52(seed=0)
    root.set_level(level_idx)
    re = _eng(root)

    seen = {_state_key(re)}
    counter = 0
    cid = 0
    pq = [(_hval(_peg_map(re)), 0, 0, copy.deepcopy(re), [])]
    while pq and counter < node_cap and time.time() - start < time_cap:
        _, depth, _, e, path = heapq.heappop(pq)
        for src, d in _legal_jumps(e):
            counter += 1
            e2 = copy.deepcopy(e)
            _apply_jump(e2, src, d)
            land = (src[0] + 2 * d[0], src[1] + 2 * d[1])
            step = ("J", src, land)
            if _nonblue(_peg_map(e2)) <= win_at:
                return path + [step]
            k = _state_key(e2)
            if k in seen:
                continue
            seen.add(k)
            cid += 1
            heapq.heappush(pq, (_hval(_peg_map(e2)), depth + 1, cid, e2, path + [step]))
        cur = _state_key(e)
        for dx, dy in _DIRS:
            counter += 1
            e2 = copy.deepcopy(e)
            e2.tmhxwcojkh(dx, dy)
            k = _state_key(e2)
            if k == cur or k in seen:
                continue
            seen.add(k)
            cid += 1
            heapq.heappush(pq, (_hval(_peg_map(e2)), depth + 1, cid,
                                e2, path + [("M", (dx, dy), None)]))
    return None


def _plan_all_onscreen(level_idx, plan):
    """Replay ``plan`` on ``Lf52(seed=0)`` and return True iff it reaches a
    verified WIN with *every* ACTION6 click inside the 64x64 window."""
    if plan is None:
        return False
    try:
        obs, acts = replay_level(0, level_idx, plan)
    except Exception:
        return False
    if obs is None:
        return False
    return all(0 <= a["data"]["x"] < 64 and 0 <= a["data"]["y"] < 64
               for a in acts if a.get("type") == "mouse")


# ── Public entry point: fast search, camera-aware fallback for off-screen plans ─
def solve_level(level_idx, node_cap=150000, time_cap=100.0):
    """Return a seed-independent plan solving ``level_idx`` (0-based), or ``None``."""
    fast = _solve_fast(level_idx, node_cap, time_cap)
    if _plan_all_onscreen(level_idx, fast):
        return fast
    if fast is not None:
        print(f"    level {level_idx}: fast plan has off-screen clicks -- "
              f"re-solving camera-aware")
    return _solve_camera_aware(level_idx, node_cap, time_cap)


# ── Camera-aware solver (slow: drives the real engine so the scroll is exact) ──
def _solve_camera_aware(level_idx, node_cap=150000, time_cap=100.0):
    """Return a seed-independent plan solving ``level_idx`` (0-based), or ``None``.
    Drives the REAL engine per node (deepcopy of the whole game) so the board's
    scroll offset is always exact; a jump is expanded only when *both* clicks land
    inside the live 64x64 camera window."""
    internal = level_idx + 1
    win_at = 2 if internal in _WIN_AT_TWO else 1
    start = time.time()
    root = Lf52(seed=0)
    root.set_level(level_idx)
    # Pin rotation on the SEARCH instance only (screen == game space here). Rotation
    # is a pure display transform, so a plan of logical cells/deltas is
    # rotation-invariant; RECORDING instances stay at their natural rotation.
    root._rotation_k = 0
    re = _eng(root)

    seen = {_state_key(re)}
    counter = 0
    cid = 0
    pq = [(_hval(_peg_map(re)), 0, 0, root, [])]
    while pq and counter < node_cap and time.time() - start < time_cap:
        _, depth, _, g, path = heapq.heappop(pq)
        e = _eng(g)
        for src, d, land in _onscreen_jumps(e):
            counter += 1
            g2 = copy.deepcopy(g)
            _, dead = _search_step(g2, ActionInput(id=GameAction.ACTION6,
                                                   data=_click_data(g2, src)))
            if dead:
                continue
            solved, dead = _search_step(g2, ActionInput(id=GameAction.ACTION6,
                                                        data=_click_data(g2, land)))
            if dead:
                continue
            e2 = _eng(g2)
            step = ("J", src, land)
            if solved or _nonblue(_peg_map(e2)) <= win_at:
                return path + [step]
            k = _state_key(e2)
            if k in seen:
                continue
            seen.add(k)
            cid += 1
            heapq.heappush(pq, (_hval(_peg_map(e2)), depth + 1, cid, g2, path + [step]))
        cur = _state_key(e)
        for dx, dy in _DIRS:
            counter += 1
            g2 = copy.deepcopy(g)
            _, dead = _search_step(g2, ActionInput(id=_DELTA_TO_ACTION[(dx, dy)]))
            if dead:
                continue
            k = _state_key(_eng(g2))
            if k == cur or k in seen:
                continue
            seen.add(k)
            cid += 1
            heapq.heappush(pq, (_hval(_peg_map(_eng(g2))), depth + 1, cid,
                                g2, path + [("M", (dx, dy), None)]))
    return None


def _plan_to_json(plan):
    out = []
    for p in plan:
        if p[0] == "J":
            out.append(["J", list(p[1]), list(p[2])])
        else:                                     # "M" delta or "C" cell
            out.append([p[0], list(p[1])])
    return out


def _plan_from_json(raw):
    out = []
    for p in raw:
        if p[0] == "J":
            out.append(("J", tuple(p[1]), tuple(p[2])))
        elif p[0] == "C":
            out.append(("C", tuple(p[1]), None))
        else:
            out.append(("M", tuple(p[1]), None))
    return out


def load_demos(demo_dir):
    """Load hand-recorded plans written by ``play_lf52.py`` -- files named
    ``lf52_level{idx}_plan.json`` with a ``won`` winning ``plan``. Returns
    {level_idx: plan}. These take precedence over the automatic solver."""
    demos = {}
    if not demo_dir or not demo_dir.exists():
        return demos
    for path in sorted(demo_dir.glob("lf52_level*_plan.json")):
        try:
            data = json.loads(path.read_text())
            if data.get("won") and data.get("plan"):
                demos[int(data["level"])] = _plan_from_json(data["plan"])
        except Exception:
            continue
    return demos


def build_plans(node_cap, time_cap, cache_path=None, demo_dir=None):
    """Solve every level once (seed-independent); return {level_idx: plan} for the
    solved levels. Precedence per level: hand-recorded demo, else cached result,
    else a fresh automatic solve (cached to ``cache_path``)."""
    demos = load_demos(demo_dir)

    cache = {}  # level_idx -> plan | None  (presence == "already attempted")
    if cache_path and cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text())
            cache = {int(k): (_plan_from_json(v) if v is not None else None)
                     for k, v in raw.items()}
        except Exception:
            cache = {}

    plans = {}
    for idx in range(NUM_LEVELS):
        if idx in demos:
            plan = demos[idx]
            print(f"  level {idx}: hand-demo plan len {len(plan)}")
        elif idx in cache:
            plan = cache[idx]
            print(f"  level {idx}: "
                  f"{'cached plan len ' + str(len(plan)) if plan else 'cached UNSOLVED'}")
        else:
            plan = solve_level(idx, node_cap, time_cap)
            cache[idx] = plan
            print(f"  level {idx}: "
                  f"{'plan len ' + str(len(plan)) if plan else 'UNSOLVED'}")
        if plan is not None:
            plans[idx] = plan

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(
            {str(i): (_plan_to_json(cache[i]) if cache[i] is not None else None)
             for i in sorted(cache)}))
    return plans


# ── Replay a cached plan on the real engine (rotation / offset aware) ──────────
def _render(game):
    return np.asarray(game.camera.render(game.current_level.get_sprites())).tolist()


def _drive(game, action_input):
    """Perform one action, draining the animation frames (which only advance when
    the camera renders). Stops the instant a level solve is queued."""
    game._full_reset = False
    game._set_action(action_input)
    last = None
    guard = 0
    while not game.is_action_complete():
        guard += 1
        if guard > _STEP_GUARD:
            break
        if game._next_level:
            break
        game.step()
        last = game.camera.render(game.current_level.get_sprites())
    if last is None:
        last = game.camera.render(game.current_level.get_sprites())
    solved = game._next_level or game._state == GameState.WIN
    dead = game._state == GameState.GAME_OVER
    return np.asarray(last).tolist(), solved, dead


def _click_for_cell(game, cell):
    """Display (x, y) for a click that lands on board cell ``cell``, using the live
    board offset (which the cosmetic scroll changes) and display rotation."""
    grid = _eng(game).hncnfaqaddg
    ox, oy = grid.cdpcbbnfdp
    tw, th = grid.tile_size
    wx = ox + cell[0] * tw + tw // 2
    wy = oy + cell[1] * th + th // 2
    k = game._rotation_k
    # inverse of the engine's remap_click(display, k) == world:
    return (wx, wy)


def _screen_move_action(game, delta):
    """Screen ACTION whose in-engine remap (for the live rotation) is the move that
    produces game-space ``delta``."""
    game_act = _DELTA_TO_ACTION[delta]
    k = game._rotation_k
    for s in (GameAction.ACTION1, GameAction.ACTION2,
              GameAction.ACTION3, GameAction.ACTION4):
        if remap_action(s, k) == game_act:
            return s
    return game_act  # k == 0 identity


def replay_level(seed, level_idx, plan):
    """Replay ``plan`` for ``level_idx`` on a fresh ``Lf52(seed)``. Returns
    (observations, actions) on a verified level solve, else (None, None).

    Retained for the SOLVER's on-screen validation (:func:`_plan_all_onscreen`);
    the training capture path is the BaseSolver harness (``solve_from`` + base
    ``drive``)."""
    game = Lf52(seed=seed)
    game.set_level(level_idx)

    observations = [_render(game)]
    actions = [dict(_RESET_ACTION)]

    for prim in plan:
        if prim[0] == "M":                       # move
            screen = _screen_move_action(game, prim[1])
            frame, solved, dead = _drive(game, ActionInput(id=screen))
            observations.append(frame)
            actions.append({"type": "simple", "index": int(screen.value)})
            if dead:
                return None, None
        else:
            cells = (prim[1], prim[2]) if prim[0] == "J" else (prim[1],)
            for cell in cells:
                bx, by = _click_for_cell(game, cell)
                frame, solved, dead = _drive(
                    game, ActionInput(id=GameAction.ACTION6,
                                      data={"x": int(bx), "y": int(by)}))
                observations.append(frame)
                actions.append({"type": "mouse", "index": _ACTION6,
                                "data": {"x": int(bx), "y": int(by)}})
                if dead:
                    return None, None
                if solved:
                    return observations, actions
    return None, None  # plan exhausted without a verified solve


# ── BaseSolver subclass ────────────────────────────────────────────────────────
class Lf52Solver(BaseSolver):
    game_id = GAME_ID
    step_guard = _STEP_GUARD
    # A non-capturing hop can strand pegs (the board can become unsolvable), and
    # ``solve_from`` only replays a precomputed plan valid at the level's INITIAL
    # state. Exploration is enabled via RESET-recovery: probe the epsilon prefix,
    # then ONE RESET back to the initial state (``reset_level`` re-clones
    # ``_clean_levels`` + re-draws the deterministic rotation), then replay the plan.
    supports_recovery = True
    recovery_mode = "reset"

    _CACHE_PATH = Path("data/lf52_plans.json")
    _DEMO_DIR = Path("data/lf52_demos")
    _NODE_CAP = 400000
    _TIME_CAP = 200.0

    def __init__(self, **kw):
        super().__init__(**kw)
        self._plans = None

    def make_game(self, seed: int):
        return Lf52(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4, CLICK_ACTION]

    def _get_plans(self) -> dict:
        if self._plans is None:
            self._plans = build_plans(self._NODE_CAP, self._TIME_CAP,
                                      self._CACHE_PATH, self._DEMO_DIR)
        return self._plans

    def solve_from(self, game, level_idx: int, seed: int):
        """Resolve the cached logical plan into SCREEN-space ``Action`` list. A
        jump's click coordinate keys on the live scroll offset (which carried-peg
        moves change), so the plan is replayed on a deepcopy of ``game`` to read
        each step's offset + rotation; the base then drives the same actions on the
        real game (identical actions -> identical offsets)."""
        plan = self._get_plans().get(level_idx)
        if not plan:
            return []
        sim = copy.deepcopy(game)
        actions = []
        for prim in plan:
            if prim[0] == "M":
                screen = _screen_move_action(sim, prim[1])
                act = Action(int(screen.value))
                actions.append(act)
                res = self.drive(sim, act)
                if res.dead:
                    return []
                if res.solved:
                    break
            else:
                cells = (prim[1], prim[2]) if prim[0] == "J" else (prim[1],)
                done = False
                for cell in cells:
                    bx, by = _click_for_cell(sim, cell)
                    act = Action(CLICK_ACTION, (int(by), int(bx)))    # click_rc = (y, x)
                    actions.append(act)
                    res = self.drive(sim, act)
                    if res.dead:
                        return []
                    if res.solved:
                        done = True
                        break
                if done:
                    break
        return actions

    def solve_episode(self, seed: int, explore: bool = True):
        """Record every solvable level for one seed (bp35-style partial episode).
        A level that fails to replay is skipped; ``ok`` iff >=1 level recorded.

        Exploration (the epsilon prefix + RESET-recovery) is wired in the same way
        as the base ``solve_episode``: the ``EpsilonSchedule`` + ``ExplorationPolicy``
        are built ONCE per episode (episode-wide arc) and handed to every
        ``record_level`` call. They engage only when ``explore and
        self.supports_recovery``; otherwise this is pure optimal replay."""
        plans = self._get_plans()
        do_explore = explore and self.supports_recovery
        schedule = (EpsilonSchedule(self.rng, center=self._explore_center,
                                    jitter=self._explore_jitter)
                    if do_explore else None)
        exploration = (ExplorationPolicy(self.available_actions(self.make_game(seed)),
                                         self.rng)
                       if do_explore else None)
        levels = []
        for level_idx in range(NUM_LEVELS):
            if level_idx not in plans:
                continue
            game = self.make_game(seed)
            try:
                obs, acts = self.record_level(game, level_idx, seed,
                                              schedule=schedule,
                                              exploration=exploration)
            except Exception:                              # noqa: BLE001
                continue
            if obs is None:
                continue
            levels.append({"level_id": level_idx, "observations": obs, "actions": acts})
        return bool(levels), levels


if __name__ == "__main__":
    sys.exit(Lf52Solver.main())
