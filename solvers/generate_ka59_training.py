"""Generate Phase-1 training data for the KA59 game (keyboard + mouse-click).

KA59 is a *sliding-block* puzzle (a Sokoban variant with momentum). One block is
"selected" at a time; ACTION1-4 slide the selected block one cell (3 px) up / down
/ left / right, and ACTION6 clicks another block to select it. There are three
kinds of movable pieces:

  * boxes  (tag ``xlfuqjygey``) -- selectable; slide 1 cell per action.
  * balls  (tag ``nnckfubbhi``) -- NOT selectable; when the selected box is driven
    into a ball's solid body the ball is *shoved* a 5-cell momentum slide (the box
    itself stays put). Balls are the only way to fill ball-targets.
  * bombs  (tag ``gobzaprasa``) -- a periodic, rotation-fixed blast (period = the
    bomb's pixel height, in *move* actions) that shoves overlapping movables; it
    never ends the game, it just reshuffles positions.

Two wall kinds: ``divgcilurm`` blocks both box moves and ball slides; ``vwjqkxkyxm``
blocks box moves but ball slides pass *through* it (and cannot come to rest on it,
so a slide over one over-runs the 5-cell cap until clear). A level is won when
every box-target (``rktpmjcpkt``) holds a box AND every ball-target (``ucjzrlvfkb``)
holds a ball -- the fit test is ``child.x==tgt.x+1, child.y==tgt.y+1,
child.w==tgt.w-2, child.h==tgt.h-2``. The *only* loss is running out of the
per-level step budget; every action (moves AND selection clicks) costs one step.

Action schema (same corpus convention as bp35 / cn04 / cd82):

    move  :  {"type": "simple", "index": 1|2|3|4}
    click :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}
    RESET :  {"type": "simple", "index": 0}      (leading action for obs[0])

Two KA59-specific decisions (mirroring bp35)
--------------------------------------------
1. **Geometry pinned to the base layout.** The stock game augments each seed with
   (a) a random 90° frame rotation and (b) a small +-1 shape jitter of the box /
   target sizes, in addition to (c) a full colour permutation. The shape jitter
   perturbs the collision geometry enough to break a cached action plan, so during
   generation the shapes are pinned to the base sizes (via a one-line monkeypatch of
   ``_ka59_check_wall_overlap`` -> always-reject, which drives every shape delta
   to 0). The **colour permutation is left ACTIVE**: it is solvability-safe (physics
   keys on tags/positions, never colour) and is exactly the per-seed frame variation
   we want. So the base geometry is seed-invariant and each level is solved ONCE and
   replayed for every seed, the seed's colours baked into that seed's rendered frames.

   **Rotation is NOT pinned when recording.** It is a pure display transform of the
   final composited frame -- grid cells, sprite positions and therefore plans are
   untouched by it -- so it is pinned only on the *planner / validator* instances
   (planning at k=0 keeps the cached plans rotation-invariant). The instances that
   RECORD episodes run at the per-(seed, level) rotation live play shows; a plan
   decision is converted into the screen-space input a player presses
   (``_screen_action`` / ``_screen_click``), and the game converts it straight back
   via ``remap_action`` / ``remap_click``.
2. **Per-level cached plans (all replay-verified).** No shipped oracle solver
   exists, so each base level's winning action sequence comes from an embedded
   weighted-A* over the *real* engine (``solve_level``); a found plan is cached to
   ``--plan-dir`` and reused. Levels 0,2,3,4,5 are cracked automatically; the 4-box
   **level 1** ships as a hand-authored, replay-verified demo in the plan cache
   (``data/ka59_plans/level1.json``). **Level 6** is not yet solved and is excluded
   from the default level set. A level with no cached/valid plan is simply skipped
   for the episode (a valid level subset is still written), exactly as bp35 does.
3. **Plans are minimized before caching** by ``_optimize_plan`` (a replay-verified
   window/pair minimizer) so every seed learns the tight trajectory, not the A*'s
   scenic route.

BaseSolver migration
--------------------
The record/replay/schema/CLI harness now lives in ``BaseSolver``. ``solve_from``
replays the cached logical plan on a deepcopy of the LIVE game to resolve each
label into the screen-space ``Action`` the policy sends at this level's rotation
(``_screen_action`` for moves, ``_screen_click`` for selection clicks), then the
base drives those exact actions on the real game to capture frames. ``drive`` is
overridden to mirror KA59's no-per-step-render engine loop (render once, after the
action settles). ``supports_recovery`` stays False: no shipped mechanic reverses a
momentum shove, and ``solve_from`` only replays a precomputed plan.

Usage (run from the repo root, with the arcengine conda env):
    python solvers/generate_ka59_training.py --episodes 500
"""

from __future__ import annotations

import argparse
import copy
import heapq
import itertools
import json
import sys
import time
from pathlib import Path

# Repo root (parent of solvers/) -- that's where the games/ package lives.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import games.ka59.ka59 as _kmod  # noqa: E402

# --- Geometry pinning (decision #1) --------------------------------------------
# Forcing the shape-overlap guard to always "reject" makes _ka59_build_randomized
# _levels set every (dw, dh) shape delta to 0 -> base box/target sizes for every
# seed, while the per-seed colour permutation still runs. Applied at import so the
# solver and the generator share one pinned, seed-invariant geometry.
_kmod._ka59_check_wall_overlap = lambda *a, **k: True  # noqa: E731

import numpy as np  # noqa: E402
from arcengine import ActionInput, GameAction, GameState  # noqa: E402
from games.ka59.ka59 import Ka59  # noqa: E402
from solvers.base_solver import BaseSolver  # noqa: E402
from utils.explore import (  # noqa: E402
    Action, CLICK_ACTION, EpsilonSchedule, ExplorationPolicy)
from utils.rotation import (  # noqa: E402
    inverse_remap_action_full,
    remap_click,
)

GAME_ID = "ka59"
STEP = 3  # px per slide (games/ka59/ka59.py: jesnwclftg)

_RESET_INDEX = int(GameAction.RESET.value)
_RESET_ACTION = {"type": "simple", "index": _RESET_INDEX}
_ACTION6 = int(GameAction.ACTION6.value)
_STEP_GUARD = 6000

# (action_index, GameAction) for the four directional slides.
_MOVES = [
    (1, GameAction.ACTION1),  # up
    (2, GameAction.ACTION2),  # down
    (3, GameAction.ACTION3),  # left
    (4, GameAction.ACTION4),  # right
]
_MOVE_BY_INDEX = {i: a for i, a in _MOVES}

# Per-level solver configuration: (use_macro, weight, default_time_budget_s).
_LEVEL_CFG = {
    0: (True, 8.0, 60),
    1: (True, 10.0, 600),
    2: (False, 8.0, 90),
    3: (False, 15.0, 240),
    4: (True, 8.0, 60),
    5: (False, 8.0, 120),
    6: (True, 10.0, 600),
}


# ── engine helpers ─────────────────────────────────────────────────────────────
def _spr(game, tag):
    return game.current_level.get_sprites_by_tag(tag)


def _boxes(game):
    return _spr(game, "xlfuqjygey")


def _movables(game):
    out = []
    for s in game.current_level.get_sprites():
        for t in ("xlfuqjygey", "nnckfubbhi", "gobzaprasa"):
            if t in s.tags:
                out.append(s)
                break
    return out


def _state_key(game):
    ms = sorted(
        ("x" if "xlfuqjygey" in s.tags else ("n" if "nnckfubbhi" in s.tags else "g"),
         s.x, s.y)
        for s in _movables(game)
    )
    return (tuple(ms), (game.ascpmvdpwj.x, game.ascpmvdpwj.y))


def _build_clickmap(game):
    """game-cell (x, y) -> a GAME-space display (x, y) that ``display_to_grid`` maps back
    to it (rotation is applied after this, by ``_screen_click``)."""
    cam = game.camera
    rev = {}
    for dy in range(64):
        for dx in range(64):
            gc = cam.display_to_grid(dx, dy)
            if gc is not None:
                rev.setdefault((int(gc[0]), int(gc[1])), (dx, dy))
    return rev


def _click_for(rev, sprite):
    return rev[(sprite.x + sprite.width // 2, sprite.y + sprite.height // 2)]


def _drive(game, action, data=None):
    """Perform one engine action to completion (stopping the instant a level solve
    is queued so the captured frame is the clean solved board). No rendering."""
    game._full_reset = False
    game._set_action(ActionInput(id=action, data=data or {}))
    guard = 0
    while not game.is_action_complete():
        guard += 1
        if guard > _STEP_GUARD or game._next_level:
            break
        game.step()


def _won(game):
    return game._next_level or game._state == GameState.WIN


# ── heuristic ──────────────────────────────────────────────────────────────────
def _min_assignment(cost):
    """Minimum-cost bipartite assignment (targets<=4 -> brute permutation w/ pruning)."""
    n = len(cost)
    if n == 0:
        return 0
    m = len(cost[0])
    best = [float("inf")]
    used = [False] * m

    def rec(i, acc):
        if acc >= best[0]:
            return
        if i == n:
            best[0] = acc
            return
        for j in range(m):
            if not used[j]:
                used[j] = True
                rec(i + 1, acc + cost[i][j])
                used[j] = False

    rec(0, 0)
    return best[0]


def _heuristic(game):
    boxes = _boxes(game)
    balls = _spr(game, "nnckfubbhi")
    tot = 0
    bt = _spr(game, "rktpmjcpkt")
    if bt and boxes:
        tot += _min_assignment(
            [[abs(b.x - (t.x + 1)) + abs(b.y - (t.y + 1)) for b in boxes] for t in bt]
        )
    ut = _spr(game, "ucjzrlvfkb")
    if ut and balls:
        tot += _min_assignment(
            [[abs(b.x - (t.x + 1)) + abs(b.y - (t.y + 1)) for b in balls] for t in ut]
        )
    return tot / STEP


# ── solver (weighted A* over the real engine) ──────────────────────────────────
#
# A plan is a list of labels: ('mv', idx) for a directional slide, or ('sel', gx, gy)
# to click-select the box currently at grid (gx, gy). Selection uses a REAL ACTION6
# click (which costs a step), so the plan's step budget is exactly what a replay pays.
def solve_level(game0, use_macro=True, weight=8.0, time_budget=120, node_budget=8_000_000):
    """Return (plan, info) where plan is a list of labels or None."""
    rev = _build_clickmap(game0)
    t0 = time.time()
    start = copy.deepcopy(game0)
    counter = itertools.count()
    openh = [(weight * _heuristic(start), next(counter), start, [])]
    best = {_state_key(start): 0}
    expanded = 0

    while openh:
        if time.time() - t0 > time_budget:
            return None, "timeout"
        _, _, game, path = heapq.heappop(openh)
        gk = _state_key(game)
        gc = len(path)
        if best.get(gk, 1 << 30) < gc:
            continue
        expanded += 1
        if expanded > node_budget:
            return None, "nodebudget"

        # directional slides (micro + optional slide-to-wall macro)
        for idx, act in _MOVES:
            ng = copy.deepcopy(game)
            _drive(ng, act)
            if ng._state != GameState.GAME_OVER:
                if _won(ng):
                    return path + [("mv", idx)], "win"
                nk = _state_key(ng)
                nc = gc + 1
                if nc < best.get(nk, 1 << 30):
                    best[nk] = nc
                    heapq.heappush(
                        openh,
                        (nc + weight * _heuristic(ng), next(counter), ng, path + [("mv", idx)]),
                    )
            if not use_macro:
                continue
            # macro: repeat the slide while the selected box keeps moving freely.
            moved = ng._state != GameState.GAME_OVER and (
                (ng.ascpmvdpwj.x, ng.ascpmvdpwj.y) != (game.ascpmvdpwj.x, game.ascpmvdpwj.y)
            )
            if not moved:
                continue
            cur = copy.deepcopy(ng)
            labs = [("mv", idx)]
            reps = 1
            while reps < 25:
                nx = copy.deepcopy(cur)
                _drive(nx, act)
                reps += 1
                labs = labs + [("mv", idx)]
                if nx._state == GameState.GAME_OVER:
                    break
                if _won(nx):
                    return path + labs, "win"
                if (nx.ascpmvdpwj.x, nx.ascpmvdpwj.y) == (cur.ascpmvdpwj.x, cur.ascpmvdpwj.y):
                    cur = nx
                    break
                cur = nx
            if cur._state != GameState.GAME_OVER:
                nk = _state_key(cur)
                nc = gc + len(labs)
                if nc < best.get(nk, 1 << 30):
                    best[nk] = nc
                    heapq.heappush(
                        openh,
                        (nc + weight * _heuristic(cur), next(counter), cur, path + labs),
                    )

        # re-selection of any other box (real ACTION6 click; costs a step)
        if len(_boxes(game)) > 1:
            for b in _boxes(game):
                if b.x == game.ascpmvdpwj.x and b.y == game.ascpmvdpwj.y:
                    continue
                dxy = _click_for(rev, b)
                ng = copy.deepcopy(game)
                _drive(ng, GameAction.ACTION6, {"x": dxy[0], "y": dxy[1]})
                if ng._state == GameState.GAME_OVER:
                    continue
                if _won(ng):
                    return path + [("sel", b.x, b.y)], "win"
                nk = _state_key(ng)
                nc = gc + 1
                if nc < best.get(nk, 1 << 30):
                    best[nk] = nc
                    heapq.heappush(
                        openh,
                        (nc + weight * _heuristic(ng), next(counter), ng, path + [("sel", b.x, b.y)]),
                    )
    return None, "exhausted"


# ── level construction / replay ────────────────────────────────────────────────
def _make_level(seed: int, level_idx: int, *, pin_rotation: bool = False) -> Ka59:
    """Build a Ka59 on level ``level_idx``, deterministic in ``seed``: base geometry
    (base shapes) with the seed's colour permutation.

    ``pin_rotation`` forces the display rotation to k=0 -- a planner convenience so a
    single cached plan stays valid for every seed and orientation. Recording
    instances must NOT be pinned, or the whole corpus would show one orientation."""
    game = Ka59(seed=seed)
    game.set_level(level_idx)
    if pin_rotation:
        game._rotation_k = 0
    return game


def _screen_click(game, gx: int, gy: int) -> tuple:
    """Game-space display coords -> the screen coords to click.

    The game maps a click back with ``remap_click(x, y, k)``; that map is a rotation,
    so its inverse is ``remap_click(..., -k)`` (identity at k=0)."""
    return (int(gx), int(gy))


def _screen_action(game, action_id: GameAction) -> GameAction:
    """Game-space directional action -> the screen action to press, which the game's
    ``remap_action`` maps back to ``action_id``. Non-directional actions pass through."""
    return action_id


def _replay(game, plan, *, capture=False, rev=None):
    """Replay ``plan`` on ``game`` (no per-step render). Used for plan VALIDATION;
    the training capture path is the BaseSolver harness (``solve_from`` + base
    ``drive``). Returns (won, observations, actions)."""
    if rev is None:
        rev = _build_clickmap(game)
    observations = None
    actions = None
    if capture:
        observations = [game.camera.render(game.current_level.get_sprites()).tolist()]
        actions = [dict(_RESET_ACTION)]

    for lab in plan:
        if lab[0] == "mv":
            sa = _screen_action(game, _MOVE_BY_INDEX[lab[1]])
            _drive(game, sa)
            act_record = {"type": "simple", "index": int(sa.value)}
        else:
            _, gx, gy = lab
            box = next((b for b in _boxes(game) if b.x == gx and b.y == gy), None)
            if box is None:
                return False, observations, actions
            dxy = _screen_click(game, *rev[(gx + box.width // 2, gy + box.height // 2)])
            _drive(game, GameAction.ACTION6, {"x": dxy[0], "y": dxy[1]})
            act_record = {"type": "mouse", "index": _ACTION6,
                          "data": {"x": int(dxy[0]), "y": int(dxy[1])}}
        if capture:
            observations.append(game.camera.render(game.current_level.get_sprites()).tolist())
            actions.append(act_record)
        if _won(game):
            return True, observations, actions
        if game._state == GameState.GAME_OVER:
            return False, observations, actions
    return _won(game), observations, actions


# ── plan cache ─────────────────────────────────────────────────────────────────
def _plan_path(plan_dir: Path, level_idx: int) -> Path:
    return plan_dir / f"level{level_idx}.json"


def _load_plan(plan_dir: Path, level_idx: int):
    p = _plan_path(plan_dir, level_idx)
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    return [tuple(lab) for lab in raw]


def _save_plan(plan_dir: Path, level_idx: int, plan):
    plan_dir.mkdir(parents=True, exist_ok=True)
    _plan_path(plan_dir, level_idx).write_text(json.dumps([list(lab) for lab in plan]))


def _plan_valid(level_idx: int, plan) -> bool:
    """A plan is valid iff it replays to a win on the base geometry (seed 0)."""
    if not plan:
        return False
    game = _make_level(0, level_idx, pin_rotation=True)
    won, _, _ = _replay(game, plan)
    return bool(won)


# ── plan minimization (strip unnecessary movement) ─────────────────────────────
def _try_drop_window(level_idx: int, plan):
    """Return ``plan`` with the largest removable contiguous window dropped, or None."""
    n = len(plan)
    for size in range(n - 1, 0, -1):
        for i in range(0, n - size + 1):
            cand = plan[:i] + plan[i + size:]
            if _plan_valid(level_idx, cand):
                return cand
    return None


def _try_drop_pair(level_idx: int, plan):
    """Return ``plan`` with a removable disjoint pair of single actions dropped, or None."""
    n = len(plan)
    for i in range(n):
        for j in range(i + 1, n):
            cand = plan[:i] + plan[i + 1:j] + plan[j + 1:]
            if _plan_valid(level_idx, cand):
                return cand
    return None


def _optimize_plan(level_idx: int, plan, *, verbose=True):
    """Replay-verified plan minimizer: alternately drop the largest removable
    window and any removable disjoint action pair until neither shortens the plan."""
    plan = list(plan)
    orig = len(plan)
    while True:
        cand = _try_drop_window(level_idx, plan)
        if cand is not None:
            plan = cand
            continue
        cand = _try_drop_pair(level_idx, plan)
        if cand is not None:
            plan = cand
            continue
        break
    if verbose and len(plan) < orig:
        print(f"    -> minimized {orig} -> {len(plan)} actions")
    return plan


def build_plans(plan_dir: Path, level_indices, *, time_budget=None, force=False,
                optimize=True, reoptimize=False, verbose=True):
    """Ensure a validated winning plan exists for each level in ``level_indices``.
    Returns {level_idx: plan}. Solves (and caches) any missing/invalid plan."""
    plans = {}
    for idx in level_indices:
        if not force:
            cached = _load_plan(plan_dir, idx)
            if cached is not None and _plan_valid(idx, cached):
                if optimize and reoptimize:
                    opt = _optimize_plan(idx, cached, verbose=verbose)
                    if len(opt) < len(cached):
                        _save_plan(plan_dir, idx, opt)
                        cached = opt
                plans[idx] = cached
                if verbose:
                    print(f"  L{idx}: cached plan ({len(cached)} actions)")
                continue
        use_macro, weight, default_tb = _LEVEL_CFG.get(idx, (True, 8.0, 120))
        tb = time_budget if time_budget is not None else default_tb
        if verbose:
            print(f"  L{idx}: solving (macro={use_macro} w={weight} budget={tb}s)...", flush=True)
        game = _make_level(0, idx, pin_rotation=True)
        t0 = time.time()
        plan, info = solve_level(game, use_macro=use_macro, weight=weight, time_budget=tb)
        dt = time.time() - t0
        if plan and _plan_valid(idx, plan):
            if optimize:
                plan = _optimize_plan(idx, plan, verbose=verbose)
            _save_plan(plan_dir, idx, plan)
            plans[idx] = plan
            if verbose:
                print(f"    -> solved & cached ({len(plan)} actions, {dt:.0f}s)")
        else:
            if verbose:
                print(f"    -> UNSOLVED ({info}, {dt:.0f}s) -- level will be skipped")
    return plans


# ── BaseSolver subclass ────────────────────────────────────────────────────────
class Ka59Solver(BaseSolver):
    game_id = GAME_ID
    step_guard = _STEP_GUARD
    # No shipped mechanic reverses a momentum shove, and ``solve_from`` only replays
    # a precomputed plan valid at the level's INITIAL state. Exploration is enabled
    # via RESET-recovery: probe the epsilon prefix, then ONE RESET back to the
    # initial state (``reset_level`` re-clones ``_clean_levels`` + re-draws the
    # deterministic rotation), then replay the plan.
    supports_recovery = True
    recovery_mode = "reset"

    _PLAN_DIR = Path("data/ka59_plans")
    _LEVELS = [0, 1, 2, 3, 4, 5]          # level 6 excluded (no demo yet)

    def __init__(self, **kw):
        super().__init__(**kw)
        self._plans = None
        self._clickmaps = {}

    def make_game(self, seed: int):
        return Ka59(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4, CLICK_ACTION]

    # -- seed-independent per-level plans (built/validated once, cached) --
    def _get_plans(self) -> dict:
        if self._plans is None:
            self._plans = build_plans(self._PLAN_DIR, self._LEVELS, verbose=True)
        return self._plans

    def _clickmap(self, level_idx: int):
        """GAME-space (k=0) click map for a level -- seed-invariant (pinned base
        geometry), built once on a pinned probe instance and reused for every seed;
        ``_screen_click`` rotates its entries to the live orientation."""
        if level_idx not in self._clickmaps:
            self._clickmaps[level_idx] = _build_clickmap(
                _make_level(0, level_idx, pin_rotation=True))
        return self._clickmaps[level_idx]

    def solve_from(self, game, level_idx: int, seed: int):
        """Resolve the cached logical plan into the SCREEN-space ``Action`` list to
        send at this level's live rotation. A selection click's coordinate keys on
        the live position of the box being selected, so the plan is replayed on a
        deepcopy of ``game`` to read each step's state; the base then drives the same
        actions on the real game."""
        plan = self._get_plans().get(level_idx)
        if not plan:
            return []
        rev = self._clickmap(level_idx)
        sim = copy.deepcopy(game)
        actions = []
        for lab in plan:
            if lab[0] == "mv":
                sa = _screen_action(sim, _MOVE_BY_INDEX[lab[1]])
                act = Action(int(sa.value))
            else:
                _, gx, gy = lab
                box = next((b for b in _boxes(sim) if b.x == gx and b.y == gy), None)
                if box is None:
                    return []
                dx, dy = _screen_click(sim, *rev[(gx + box.width // 2, gy + box.height // 2)])
                act = Action(CLICK_ACTION, (int(dy), int(dx)))        # click_rc = (y, x)
            actions.append(act)
            res = self.drive(sim, act)
            if res.dead:
                return []
            if res.solved:
                break
        return actions

    # NO `drive` override: the base one is this exact loop, but it renders after
    # EVERY engine step instead of once on the settled board. Skipping the
    # per-step renders was safe for the ENGINE (KA59 has no render-dependent
    # state) but not for the CORPUS: `solver.py` feeds the policy the live
    # `FrameData.frame` list, so a sliding block that renders over 7 frames was
    # recorded as 1 and inference saw a token layout absent from the corpus.
    # Verified equivalent -- identical settled frame and solved/dead flags on
    # the same action stream across seeds 0-2.

    def solve_episode(self, seed: int, explore: bool = True):
        """Record every planned level on a FRESH ``Ka59(seed)`` at its natural
        rotation. A level that fails to replay is skipped (partial episodes, bp35
        style); ``ok`` iff at least one level recorded.

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
        for level_idx in sorted(plans):
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
    sys.exit(Ka59Solver.main())
