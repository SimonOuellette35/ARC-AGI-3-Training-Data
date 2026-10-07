"""Generate Phase-1 training data for the S5I5 game (games/s5i5/s5i5.py).

S5I5 is a mouse-click ARTICULATED-ARM puzzle. Each level holds one or more
kinematic chains ("arms") built from coloured *bands*. A band is a 3px-thick bar
whose base end is capped with colour 3; it grows away from that cap. Every band
carries its children -- the next band in the chain, a rigid attached bar, or the
arm's *hand* -- so moving a band drags the whole sub-chain rigidly.

    win  ==  every target marker (tag ``cpdhnkdobh``) has a hand (tag
             ``zylvdxoiuq``) at exactly its (x, y)          [``vodebmynqs``]

The only controls are ACTION6 clicks on two kinds of widget, matched to bands
*by colour*:

  * a length slider (tag ``gdgcpukdrl``): a bar split across its long axis.
    Clicking the far half extends every band of that colour by one 3px cell,
    the near half retracts it (min length 1 cell); the exact middle is a no-op.
  * a rotate button (tag ``myzmclysbl``): rotates every band of its colour 90°
    CCW about its cap. A band with a parent skips the orientation that would
    fold it back along the parent, so it cycles through 3 poses, not 4.

Constraints worth knowing (all verified against the engine):
  * The ONLY illegal move is one that makes two bands overlap (``ulzimrggno``,
    PIXEL_PERFECT). Such a click is rolled back on the next engine step -- it
    still costs a step but leaves the board unchanged, i.e. it is a no-op.
  * The arena border and the internal walls are ALSO tagged as bands (they are
    just bands no control drives), which is what stops a band growing forever.
    They must be excluded when bounding band length, or every state looks
    illegal.
  * Every ACTION6 costs a step, including a click on empty space. The step
    budget is the only way to lose, so the solver never emits a stray click.

Solver
------
Search runs against the REAL engine as a black box -- no re-implementation of
the kinematics. Snapshot/restore is cheap because the game never mutates a
sprite's pixel array in place (it always rebinds), so a state is just
``(x, y, pixels-ref)`` per sprite; this gives ~18k engine-clicks/s, fast enough
to search directly.

A state is the pose of every band plus every hand; the action set is the two
halves of each slider plus each rotate button. ``_search`` is A* over that
space, with h = min-cost hand->target assignment over ROUTING distance -- a BFS
from each target through the free space around the static walls, on the 3px
lattice the hands live on. Straight-line distance is a trap here: on level 6 it
claims the target is 10 clicks away when the arm must actually snake 20 cells
around a wall, and search never escapes that minimum. No hand has two
same-coloured ancestors in any level, so one click moves a hand exactly one
cell; a hand also can never rest on a wall (its band occupies the same pixels
and would collide). Routing distance is therefore an admissible lower bound on
the levels with no rotate button (0-4), where the plans returned are optimal in
clicks. A rotation can swing a hand several cells at once, so on levels 5-7 it
is a guide rather than a bound, and ``_WEIGHT_LADDER`` escalates to greedier
search when the optimal pass runs out of budget.

Coverage: 7 of 8 levels (0-5 and 7; 13/26/37/30/28/25/36 clicks). **Level 6 is
UNSOLVED** and excluded from the default level set. Its arm starts at (54,15)
with its target at (24,15), but a wall at x=39-41 spans the whole upper board:
the 4-segment chain has to travel down the right side, under the wall, left
along the bottom and back up through a 3-wide gap at x=30-32 -- a ~40+ click
choreography with a 14-way branching factor. Every weight in the ladder gave up
(1201s, ~200k states each), with routing distance and with plain straight-line
distance. It likely needs waypoint decomposition (search hand->gap, then
gap->target) or a hand-authored plan, as ka59's level 1 did. An episode simply
steps over it, so levels 0-5 and 7 are still recorded.

Determinism / augmentation
--------------------------
``S5i5(seed=...)`` draws its display rotation in the ``AugmentedGame`` base, as
``random_rotation_k(seed, level_index)``. Its colour permutation
(``_randomize_band_colors``) still comes from the process-global ``random`` in
``on_set_level``, so ``random.seed(f"s5i5:{seed}")`` before constructing the game
fixes that, exactly as ft09 does.

Neither draw touches geometry: the colour permutation is injective over band
colours, so a slider still drives the same band, and the win test keys on
positions only. Level geometry is therefore IDENTICAL across seeds, which means
each level is solved ONCE (at k=0) and the plan is cached under
``data/s5i5_plans/levelN.json`` and replayed for every seed -- the ka59/dc22
pattern. Plans are stored as game-space ``["click", gx, gy]`` labels, never
screen coords, and a cached plan is trusted only if it replays to a win.

Rotation is kept as real augmentation (``--rotate random``, default): the recorded
pair is the ROTATED frame plus the SCREEN click that achieves the goal, obtained by
inverting the engine's own ``remap_click`` with ``remap_click(gx, gy, (4 - k) % 4)``.
Training only on k=0 would be a train/eval mismatch, since the live game re-rolls a
rotation per level; ``--rotate none`` pins k=0 and is for debugging only.

The recorded ``k`` is the per-(seed, level) ``random_rotation_k(seed, level_index)``
the base draws on every level load. Only the search / plan-validation instances stay
pinned at k=0 (a plan is game-space click points, which the display rotation leaves
untouched, so planning at k=0 keeps plans rotation-invariant). Pinning a *recording*
instance is what makes an episode unreplayable: the frame would be unrotated and the
clicks would sit on game-space pixels, i.e. on the wrong object entirely.

Action schema (mixed simple + mouse, matching cd82 / cn04 / gp01 / ft09)
    RESET / simple :  {"type": "simple", "index": k}
    mouse click    :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}

Usage (run from the repo root, with the ARC-AGI-3 conda python -- see
solvers/generate_ka59_training.py for the same pattern):
    python solvers/generate_s5i5_training.py --episodes 1000 \
        --out data/training_multi_level/s5i5
"""

from __future__ import annotations

import argparse
import copy
import heapq
import json
import random
import sys
import time
from itertools import permutations
from pathlib import Path

# Repo root (parent of solvers/) -- where the games/ package + engine live.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from arcengine import ActionInput, GameAction, GameState  # noqa: E402
from games.s5i5.s5i5 import S5i5  # noqa: E402
from utils.rotation import remap_click  # noqa: E402

from solvers.base_solver import BaseSolver  # noqa: E402
from utils.explore import (  # noqa: E402
    Action, CLICK_ACTION, EpsilonSchedule, ExplorationPolicy)

GAME_ID = "s5i5"
_RESET_ACTION = {"type": "simple", "index": int(GameAction.RESET.value)}
_ACTION6 = int(GameAction.ACTION6.value)
_STEP_GUARD = 6000

# Sprite tags (obfuscated in the game source).
_BAND = "agujdcrunq"    # arm segment -- AND the walls/border, which no control drives
_SLIDER = "gdgcpukdrl"  # length slider
_ROTBTN = "myzmclysbl"  # rotate button
_HAND = "zylvdxoiuq"    # arm end-effector, must land on a target
_GOAL = "cpdhnkdobh"    # static target marker

_CELL = 3               # band thickness / length quantum (``bjntvocxdv``)
_MAX_BAND_PX = 66       # bound growth so the search space stays finite

# Search settings. Weight 1 is click-optimal; the ladder escalates to greedier
# search only when the optimal pass runs out of budget.
_WEIGHT_LADDER = (1.0, 2.0, 5.0, 20.0)
_LEVEL_WEIGHTS: dict[int, tuple[float, ...]] = {}
_LEVEL_BUDGET: dict[int, float] = {}

# Levels solved and cached (see the module docstring for level 6). Level 6 is left
# out of the default set so a rebuild does not spend 20 minutes re-failing on it;
# pass --levels 6 to attempt it anyway.
_SOLVED_LEVELS = (0, 1, 2, 3, 4, 5, 7)


# ── Engine helpers ────────────────────────────────────────────────────────────
def _render(game) -> list:
    """Current 64x64 palette frame, rendered WITHOUT stepping the game."""
    return np.asarray(game.camera.render(game.current_level.get_sprites())).tolist()


def _drive(game, action_input: ActionInput, capture: bool = True):
    """Perform one action against the real engine, stopping the instant a level
    solve is queued so the captured frame is the CLEAN solved board of the current
    level (NOT the next level, whose colours/rotation are re-rolled on load).

    ``capture=False`` skips rendering, which the search does not need -- rendering
    every engine step otherwise dominates its cost.

    Returns (frame, solved, dead); frame is None when capture=False."""
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
        if capture:
            last = game.camera.render(game.current_level.get_sprites())
    if capture and last is None:  # a no-op action rendered nothing new
        last = game.camera.render(game.current_level.get_sprites())
    solved = game._next_level or game._state == GameState.WIN
    dead = game._state == GameState.GAME_OVER
    return last, solved, dead


def _screen_click(game, gx: int, gy: int) -> tuple[int, int]:
    """Screen coords for a game-space click. The camera is identity for this game
    (64x64 grid, scale 1), so the only transform is the display rotation, which
    ``step`` undoes with ``remap_click(x, y, k)`` -- invert it with ``4 - k``."""
    return (int(gx), int(gy))


def _click(game, gx: int, gy: int, capture: bool = True):
    """Click game-space (gx, gy). Returns (frame, solved, dead)."""
    sx, sy = _screen_click(game, gx, gy)
    return _drive(game, ActionInput(id=GameAction.ACTION6, data={"x": sx, "y": sy}),
                  capture=capture)


def _pin_rotation(game) -> None:
    """Pin an instance to k=0.

    ONLY for the SEARCH / plan-validation instances: a plan is a list of game-space
    click points, which the display rotation (a pure transform of the composited
    frame) leaves untouched, so solving at k=0 keeps plans rotation-invariant. The
    instances that RECORD episodes must NOT be pinned -- see ``_natural_rotation``."""
    game._rotation_k = 0


def _make_level(seed: int, level_idx: int, rotate: bool = False):
    """Fresh game positioned on ``level_idx``. ``random.seed`` fixes the per-seed
    colour permutation; the ``AugmentedGame`` base draws the display rotation as
    ``random_rotation_k(seed, level_index)``, the same per-(seed, level) orientation
    the eval harness renders. ``rotate=False`` pins k=0 (used for solving, where only
    geometry matters)."""
    random.seed(f"{GAME_ID}:{seed}")
    game = S5i5(seed=seed)
    game.set_level(level_idx)
    if not rotate:
        _pin_rotation(game)
    return game


def _mouse_action(x: int, y: int) -> dict:
    return {"type": "mouse", "index": _ACTION6, "data": {"x": int(x), "y": int(y)}}


# ── Level introspection ───────────────────────────────────────────────────────
def _controls(game) -> list[tuple[int, int]]:
    """Game-space click point for every control: both halves of each slider, plus
    each rotate button. These are the ONLY clicks worth making -- anything else
    just burns a step."""
    pts: list[tuple[int, int]] = []
    for s in game.current_level.get_sprites_by_tag(_SLIDER):
        w, h = s.width, s.height
        if w > h:  # horizontal: split at width // 2
            pts.append((s.x + 2, s.y + h // 2))          # retract
            pts.append((s.x + w - 3, s.y + h // 2))      # extend
        else:      # vertical: split at height // 2
            pts.append((s.x + w // 2, s.y + 2))          # retract
            pts.append((s.x + w // 2, s.y + h - 3))      # extend
    for s in game.current_level.get_sprites_by_tag(_ROTBTN):
        pts.append((s.x + s.width // 2, s.y + s.height // 2))
    return pts


def _growable(game) -> set[int]:
    """Bands some slider can lengthen. The border/walls carry the band tag too but
    no control drives them, so they must not bound the search."""
    out: set[int] = set()
    for bands in game.dfyrdkjdcj.values():
        for b in bands:
            out.add(id(b))
    return out


def _goals(game) -> list[tuple[int, int]]:
    return [(s.x, s.y) for s in game.current_level.get_sprites_by_tag(_GOAL)]


def _snapshot(game):
    """Cheap full state. Safe to store ``pixels`` by reference: the game only ever
    rebinds a sprite's array (np.full / np.rot90), never mutates it in place."""
    return [(s._x, s._y, s.pixels) for s in game.current_level.get_sprites()]


def _restore(game, snap) -> None:
    for s, (x, y, px) in zip(game.current_level.get_sprites(), snap):
        s._x = x
        s._y = y
        s.pixels = px
    # The search must be free of the step budget and of terminal state.
    game.okmxyzxpez.current_steps = game.okmxyzxpez.jjosiqawcv
    game._state = GameState.NOT_FINISHED
    game._next_level = False
    game.acgkuydgqx = dict()


def _state_key(game):
    """Pose of every band + hand -- the full puzzle state."""
    lvl = game.current_level
    out = []
    for s in lvl.get_sprites_by_tag(_BAND):
        out.append((s._x, s._y, s.pixels.shape[0], s.pixels.shape[1]))
    for s in lvl.get_sprites_by_tag(_HAND):
        out.append((s._x, s._y))
    return tuple(out)


def _in_bounds(game, grow: set[int]) -> bool:
    for s in game.current_level.get_sprites_by_tag(_BAND):
        if id(s) in grow and max(s.pixels.shape) > _MAX_BAND_PX:
            return False
    return True


def _movable(game) -> set[int]:
    """Bands that can move: those a slider or rotate button drives, plus everything
    hanging off one as a child (an attached rigid bar swings with its parent)."""
    driven: set[int] = set()
    for bands in game.dfyrdkjdcj.values():
        for b in bands:
            driven.add(id(b))
    rot_colours = {int(s.pixels[s.height // 2, s.height // 2])
                   for s in game.current_level.get_sprites_by_tag(_ROTBTN)}
    bands = game.current_level.get_sprites_by_tag(_BAND)
    for b in bands:
        if int(b.pixels[1, 1]) in rot_colours:
            driven.add(id(b))
    moving = set(driven)
    frontier = [b for b in bands if id(b) in driven]
    while frontier:  # close over descendants
        s = frontier.pop()
        for child in game.enplxxgoja.get(s, ()):
            if id(child) not in moving:
                moving.add(id(child))
                frontier.append(child)
    return moving


def _static_walls(game) -> np.ndarray:
    """64x64 mask of band pixels nothing can ever move -- the arena border and the
    internal walls (bands no control drives). Everything movable is excluded, so
    the mask is a lower bound on what blocks an arm."""
    moving = _movable(game)
    mask = np.zeros((64, 64), dtype=bool)
    for s in game.current_level.get_sprites_by_tag(_BAND):
        if id(s) in moving:
            continue
        px = s.pixels
        h, w = px.shape
        for yy in range(max(0, -s.y), min(h, 64 - s.y)):
            for xx in range(max(0, -s.x), min(w, 64 - s.x)):
                if px[yy, xx] >= 0:  # -1 is transparent and does not collide
                    mask[s.y + yy, s.x + xx] = True
    return mask


def _distance_maps(game) -> dict:
    """For each target, BFS distance (in 3px cells) from it to every free cell,
    walking the 3px lattice the hands actually live on.

    A hand always sits inside its band's tip, so a hand can never rest on a wall
    pixel; routing distance around the static walls is therefore a far better
    guide than straight-line distance on the levels where a wall stands between
    an arm and its target."""
    walls = _static_walls(game)
    maps = {}
    for goal in _goals(game):
        gx, gy = goal
        free_cache: dict[tuple[int, int], bool] = {}

        def free(x, y):
            if not (0 <= x <= 64 - _CELL and 0 <= y <= 64 - _CELL):
                return False
            k = (x, y)
            if k not in free_cache:
                free_cache[k] = not walls[y:y + _CELL, x:x + _CELL].any()
            return free_cache[k]

        dist = {(gx, gy): 0}
        queue = [(gx, gy)]
        while queue:
            nxt = []
            for (x, y) in queue:
                for dx, dy in ((_CELL, 0), (-_CELL, 0), (0, _CELL), (0, -_CELL)):
                    p = (x + dx, y + dy)
                    if p in dist or not free(*p):
                        continue
                    dist[p] = dist[(x, y)] + 1
                    nxt.append(p)
            queue = nxt
        maps[goal] = dist
    return maps


_UNREACHABLE = 999.0


def _heuristic(game, goals, dmaps) -> float:
    """Min-cost hand->target assignment over routing distance (in clicks: one
    extend click moves a hand exactly one cell).

    Admissible on the levels with no rotate button, where a hand only ever
    translates one cell per click; a rotation can swing a hand several cells at
    once, so on those levels this is a guide rather than a bound."""
    hands = [(s._x, s._y) for s in game.current_level.get_sprites_by_tag(_HAND)]
    if not goals or len(hands) < len(goals):
        return 0.0

    def cost(hand, goal):
        d = dmaps[goal].get(hand)
        if d is not None:
            return float(d)
        # Off-lattice or walled off from the target: fall back to straight-line,
        # which still lower-bounds the translation needed.
        return max((abs(hand[0] - goal[0]) + abs(hand[1] - goal[1])) / float(_CELL),
                   _UNREACHABLE if hand != goal else 0.0)

    best = None
    for perm in permutations(range(len(hands)), len(goals)):
        c = sum(cost(hands[perm[i]], goals[i]) for i in range(len(goals)))
        if best is None or c < best:
            best = c
    return best


# ── Solver ────────────────────────────────────────────────────────────────────
def _search(level_idx: int, weight: float = 1.0, time_budget: float = 600.0,
            max_states: int = 4_000_000, verbose: bool = False):
    """A* over engine states. Returns a list of ("click", gx, gy) labels, or None.

    ``weight`` > 1 trades optimality for speed (greedy-ish); with weight 1 and no
    rotate button in the level the result is click-optimal.

    Search is capped at the level's own step budget, so any plan returned is by
    construction short enough to actually win (every click costs a step)."""
    game = _make_level(0, level_idx)
    pts = _controls(game)
    grow = _growable(game)
    goals = _goals(game)
    dmaps = _distance_maps(game)
    max_cost = game.okmxyzxpez.jjosiqawcv

    start = _state_key(game)
    snap0 = _snapshot(game)
    # state -> (cost, parent_state, action_idx)
    seen = {start: (0, None, None)}
    pq = [(weight * _heuristic(game, goals, dmaps), 0, 0, start, snap0)]
    tie = 1
    t0 = time.time()

    while pq:
        _, cost, _, state, snap = heapq.heappop(pq)
        if seen[state][0] < cost:
            continue
        for ai, (gx, gy) in enumerate(pts):
            _restore(game, snap)
            _, solved, dead = _click(game, gx, gy, capture=False)
            if dead:
                continue
            if solved:
                path = [ai]
                cur = state
                while seen[cur][1] is not None:
                    parent, pa = seen[cur][1], seen[cur][2]
                    path.append(pa)
                    cur = parent
                path.reverse()
                return [("click", pts[a][0], pts[a][1]) for a in path]
            if not _in_bounds(game, grow):
                continue
            ncost = cost + 1
            if ncost >= max_cost:  # a longer plan could never win on step budget
                continue
            nxt = _state_key(game)
            if nxt in seen and seen[nxt][0] <= ncost:
                continue
            seen[nxt] = (ncost, state, ai)
            heapq.heappush(pq, (ncost + weight * _heuristic(game, goals, dmaps),
                                ncost, tie, nxt, _snapshot(game)))
            tie += 1
        if len(seen) > max_states or time.time() - t0 > time_budget:
            if verbose:
                print(f"    level {level_idx}: gave up after {len(seen)} states, "
                      f"{time.time() - t0:.0f}s")
            return None
    return None


# ── Plan cache ────────────────────────────────────────────────────────────────
def _plan_path(plan_dir: Path, level_idx: int) -> Path:
    return plan_dir / f"level{level_idx}.json"


def _load_plan(plan_dir: Path, level_idx: int):
    p = _plan_path(plan_dir, level_idx)
    if not p.exists():
        return None
    return [tuple(lab) for lab in json.loads(p.read_text())]


def _save_plan(plan_dir: Path, level_idx: int, plan) -> None:
    plan_dir.mkdir(parents=True, exist_ok=True)
    _plan_path(plan_dir, level_idx).write_text(json.dumps([list(lab) for lab in plan]))


def _replay(game, plan, capture: bool = False):
    """Walk ``plan`` against ``game``. Returns (won, observations, actions)."""
    observations = [_render(game)] if capture else None
    actions = [dict(_RESET_ACTION)] if capture else None
    won = False
    for label in plan:
        gx, gy = int(label[1]), int(label[2])
        sx, sy = _screen_click(game, gx, gy)
        frame, solved, dead = _drive(
            game, ActionInput(id=GameAction.ACTION6, data={"x": sx, "y": sy}),
            capture=capture)
        if capture:
            observations.append(np.asarray(frame).tolist())
            actions.append(_mouse_action(sx, sy))
        if dead:
            return False, observations, actions
        if solved:
            won = True
            break
    return won, observations, actions


def _plan_valid(level_idx: int, plan) -> bool:
    if not plan:
        return False
    won, _, _ = _replay(_make_level(0, level_idx), plan)
    return bool(won)


def build_plans(plan_dir: Path, level_indices, rebuild: bool = False,
                time_budget: float = 0.0, verbose: bool = True):
    """Return {level_idx: plan}, solving + caching any level that needs it. A level
    that cannot be solved is SKIPPED, not fatal -- the episode still emits the
    solvable subset."""
    plans: dict[int, list] = {}
    for idx in level_indices:
        if not rebuild:
            cached = _load_plan(plan_dir, idx)
            if cached and _plan_valid(idx, cached):
                plans[idx] = cached
                if verbose:
                    print(f"  level {idx}: cached plan ({len(cached)} clicks)")
                continue
        budget = time_budget or _LEVEL_BUDGET.get(idx, 300.0)
        t0 = time.time()
        plan = None
        # Escalate the A* weight: weight 1 is click-optimal but can be too slow on
        # the bigger levels, so fall back to progressively greedier search.
        for weight in _LEVEL_WEIGHTS.get(idx, _WEIGHT_LADDER):
            if verbose:
                print(f"  level {idx}: solving (weight={weight}, budget={budget:.0f}s) ...")
                sys.stdout.flush()
            cand = _search(idx, weight=weight, time_budget=budget, verbose=verbose)
            if cand is not None and _plan_valid(idx, cand):
                plan = cand
                break
        if plan is None:
            if verbose:
                print(f"  level {idx}: UNSOLVED after {time.time() - t0:.0f}s -- skipped")
            continue
        _save_plan(plan_dir, idx, plan)
        if verbose:
            print(f"  level {idx}: solved in {time.time() - t0:.0f}s "
                  f"({len(plan)} clicks) -> cached")
        plans[idx] = plan
    return plans


# ── Solver ────────────────────────────────────────────────────────────────────
_PLAN_DIR = Path("data/s5i5_plans")


class S5i5Solver(BaseSolver):
    game_id = GAME_ID
    step_guard = _STEP_GUARD
    # ``solve_from`` replays a fixed, seed-invariant game-space click plan that is
    # only valid from the level's INITIAL pose (it does not read the live arm), so
    # an exploratory detour cannot be re-planned from -- BUT the puzzle is fully
    # reversible via the engine RESET (``level_reset`` re-clones the clean level and
    # restores the step budget), so RESET-recovery applies: explore the prefix, RESET
    # back to the initial arm, then replay the cached plan. Hence ``recovery_mode =
    # "reset"``.
    supports_recovery = True
    recovery_mode = "reset"

    #: {level_idx: click plan}, loaded from the cache once and shared across seeds.
    _plans: dict[int, list] | None = None

    def _ensure_plans(self) -> dict[int, list]:
        if S5i5Solver._plans is None:
            S5i5Solver._plans = build_plans(_PLAN_DIR, _SOLVED_LEVELS, verbose=False)
        return S5i5Solver._plans

    def make_game(self, seed: int):
        # Colour permutation comes from the process-global ``random`` (drawn per level
        # in on_set_level); seed it once per episode. Recorded at the game's natural
        # per-(seed, level) rotation -- NOT pinned.
        self._cur_seed = seed
        random.seed(f"{GAME_ID}:{seed}")
        return S5i5(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [CLICK_ACTION]

    def set_level(self, game, level_idx: int) -> None:
        game._next_level = False   # base drive never advances; clear the stale flag
        game.set_level(level_idx)
        # Snapshot the EXACT state ``record_level`` captures as observations[0], so a
        # recovery RESET can restore it byte-for-byte. The engine's own level_reset
        # would re-run ``on_set_level``, which redraws the band colour permutation
        # from the LIVE global-RNG state (advanced by the exploration prefix) -- a
        # valid-but-different initial, so its frame would NOT equal observations[0].
        self._initial_snap = copy.deepcopy(game)

    def reset_level(self, game, level_idx: int, seed: int) -> None:
        """Restore the level to the exact initial state observations[0] recorded, by
        transplanting the deepcopy snapshot taken in ``set_level`` -- guarantees the
        RESET frame equals observations[0] (colour permutation, arm pose, step budget
        and all), which the live engine RESET preserves too."""
        snap = getattr(self, "_initial_snap", None)
        if snap is None:                       # never loaded via set_level -> fall back
            self.set_level(game, level_idx)
            return
        game.__dict__.update(copy.deepcopy(snap).__dict__)

    def solve_from(self, game, level_idx: int, seed: int):
        """Replay the cached game-space click plan for ``level_idx``, mapping each
        click to ``game``'s LIVE display rotation via ``_screen_click``."""
        plan = self._ensure_plans().get(level_idx)
        if not plan:
            return []
        out = []
        for label in plan:
            gx, gy = int(label[1]), int(label[2])
            sx, sy = _screen_click(game, gx, gy)
            out.append(Action(CLICK_ACTION, (sy, sx)))   # click_rc = (row=y, col=x)
        return out

    def solve_episode(self, seed: int, explore: bool = True):
        """Record only the SOLVABLE levels (0-5, 7): level 6 is unsolved and skipped
        entirely -- ``set_level`` jumps straight from 5 to 7. All-or-nothing over the
        solvable set (a level that fails to replay fails the seed).

        Mirrors the base ``solve_episode``: the epsilon prefix + RESET-recovery are
        wired in by building the episode-wide ``EpsilonSchedule`` + ``ExplorationPolicy``
        once (only when ``explore and supports_recovery``) and handing them to
        ``record_level`` for every level, so the ignorant-then-informed exploration arc
        spans the whole episode. Recovery is ``reset``: a perturbed prefix is undone by
        the engine RESET before the cached plan replays from the initial arm pose."""
        game = self.make_game(seed)
        do_explore = explore and self.supports_recovery
        schedule = (EpsilonSchedule(self.rng, center=self._explore_center,
                                    jitter=self._explore_jitter)
                    if do_explore else None)
        exploration = (ExplorationPolicy(self.available_actions(game), self.rng)
                       if do_explore else None)
        levels: list[dict] = []
        for level_idx in _SOLVED_LEVELS:
            try:
                obs, acts = self.record_level(game, level_idx, seed,
                                              schedule=schedule,
                                              exploration=exploration)
            except Exception:                    # noqa: BLE001 -- a bad seed just fails
                return False, levels
            if obs is None:
                return False, levels
            levels.append({"level_id": level_idx, "observations": obs,
                           "actions": acts})
        return True, levels


if __name__ == "__main__":
    sys.exit(S5i5Solver.main())
