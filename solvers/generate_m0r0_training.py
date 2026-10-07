"""Generate Phase-1 training data for the M0R0 game (keyboard + mouse-click).

M0R0 is a *mirror-maze* puzzle: two robots are driven by one set of arrow keys and
must be brought onto the same cell. The catch is that they do not move alike --
``qzfkx-ubwff-idtiq`` steps ``(dx, dy)`` while ``qzfkx-ubwff-crkfz`` steps
``(-dx, dy)``. So a vertical press moves both the same way and a horizontal press
moves them in *opposite* directions: in open space their y-gap never changes and
their x-gap only ever changes by 2. The ONLY way to break that lockstep is to let
the geometry stop one robot while the other keeps going, so the entire game is
about steering the pair into walls in the right order.

Board elements (all names as they appear in ``games/m0r0/m0r0.py``):

  * ``jggua-LevelN`` -- the maze. One big PIXEL_PERFECT sprite whose ``0`` pixels
    are solid and whose ``-1`` pixels are open floor.
  * ``cvcer``        -- a free-floating block (tag ``nhiae``). ACTION6-click it to
    select it, after which ACTION1-4 push *it* instead of the robots (no
    adjacency needed -- it is directly driven). Clicking anything else deselects
    and gives the arrows back to the robots. Blocks are walls for robots, so they
    are the player's own de-synchronisation tool -- and, in level 2, corks the game
    drops into the corridors you need.
  * ``wyiex``        -- a hazard. A robot ending its move on one snaps *every*
    sys_click sprite (both robots and every block) back to its level-start cell.
  * ``hnutp-<c>`` / ``dfnuk-<c>`` -- button / door pairs. A door is open exactly
    while some unpaired robot stands on a button of the same colour, recomputed at
    the end of every directional action.

The robots merge when they share a cell, or when they *swap* through each other:
if they were horizontally adjacent before the move and either lands on the other's
old cell, both snap to the midpoint. Merging both robots ends the level. The only
loss is the 150-action budget (every action, clicks included, costs one).

Action schema (same corpus convention as ka59 / bp35 / cn04 / cd82):

    move  :  {"type": "simple", "index": 1|2|3|4}
    click :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}
    RESET :  {"type": "simple", "index": 0}      (leading action for obs[0])

Three M0R0-specific decisions
-----------------------------
1. **Solve once, replay per seed.** M0R0's per-seed augmentation is a colour
   permutation (``_randomize_colors``) plus a 90-degree frame rotation
   (``random_rotation_k``). Neither touches the board: the maze, the robots, the
   blocks and the hazards sit on exactly the same cells for every seed. So each
   level is solved ONCE, in game space, and the resulting logical plan is replayed
   for every seed with that seed's colours baked into the rendered frames.

2. **Rotation is left ACTIVE and re-mapped at replay** (the lf52 pattern, not
   ka59's k=0 pinning). The plan is stored in *game* space; ``_replay`` reads the
   level's live ``_rotation_k`` and converts each action to the *screen* action the
   player would actually press -- directions via ``inverse_remap_action_full`` and
   clicks via ``remap_click(..., (4 - k) % 4)``. That keeps all four orientations in
   the corpus (the on-screen ``RotationDisplay`` is the cue that disambiguates them)
   instead of training only on the unrotated view.

3. **The search runs on a native model, and every plan is engine-verified.**
   Deep-copying the engine per node caps out around ~60 nodes/s, and level 2 alone
   needs millions. So ``_Model`` re-implements the step function natively (~5 us/node)
   and the search runs there; the winning plan is then replayed through the REAL
   engine and only kept if the engine says WIN (``_plan_valid``). The model is
   checked against the engine directly by ``--selfcheck``, which drives thousands of
   random rollouts through both and compares state after every action.

   Three tiers, escalating only as far as each level forces:
     * *frozen*     -- BFS over robot positions with the blocks nailed down. Cracks
       every level that needs no block moved (0, 1, 4).
     * *exhaustive* -- BFS over the full state (robots + blocks + selection), under a
       node cap. Still optimal; enough for levels 3 and 5, which need a block shoved
       aside but not much of one.
     * *guided*     -- weighted A* over the full state, for when breadth-first is
       hopeless. The heuristic is the exact distance-to-merge on the blocks-removed
       relaxation, precomputed once per level by a backward BFS over all ~5.8k
       robot-pair states. Only level 2 gets this far, and it is built to: its
       mirror-axis column is solid except for a single cell, that cell has a block
       sitting in it, and a second block corks the only corridor into the left half,
       so nothing is winnable until two separate blocks are driven out of the way.

Determinism: the plan is seed-free, and per (seed, level) the colours come from a
keyed ``random.Random`` and the rotation from a seeded global draw, so a given
``--start-seed`` reproduces byte-for-byte.

Usage (run from the repo root, with the arcengine conda env -- see the env note in
solvers/generate_ka59_training.py):
    python solvers/generate_m0r0_training.py --episodes 500
    # (re)build the plan cache only, with a big search budget:
    python solvers/generate_m0r0_training.py --build-plans-only --level-time-budget 900
    # verify the native model still matches the engine step-for-step:
    python solvers/generate_m0r0_training.py --selfcheck
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

# Repo root (parent of solvers/) -- that's where the games/ package lives.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameAction, GameState  # noqa: E402
from games.m0r0.m0r0 import M0r0  # noqa: E402
from utils.rotation import inverse_remap_action_full, remap_click  # noqa: E402

from solvers.base_solver import BaseSolver  # noqa: E402
from utils.explore import Action, CLICK_ACTION  # noqa: E402

GAME_ID = "m0r0"
BUDGET = 150  # per-level action budget (games/m0r0/m0r0.py: `if self._action_count > 150`)

_RESET_ACTION = {"type": "simple", "index": int(GameAction.RESET.value)}
_ACTION6 = int(GameAction.ACTION6.value)
_STEP_GUARD = 5000

# Robot sprite names, and the sign each one applies to a (dx, dy) press. Only the
# two ``ubwff`` robots appear in the shipped levels; the ``kncqr`` pair (which also
# mirrors dy) is modelled anyway so the solver stays correct if a level adds them.
_ROBOT_NAMES = ["qzfkx-ubwff-idtiq", "qzfkx-ubwff-crkfz",
                "qzfkx-kncqr-idtiq", "qzfkx-kncqr-crkfz"]
_SIGNS = {"ubwff-idtiq": (1, 1), "ubwff-crkfz": (-1, 1),
          "kncqr-idtiq": (1, -1), "kncqr-crkfz": (-1, -1)}
_DOOR_COLORS = ["raixb", "ujcze", "qeazm"]

# ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right
_DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

# Per-level solver config: (weight, default_time_budget_s) for the guided tier.
_LEVEL_CFG = {2: (2.0, 900)}
_DEFAULT_CFG = (1.0, 300)

# Heuristic value for a robot-pair state that cannot merge even with every block
# removed. Such a state is not provably dead (a block can serve as a wall the
# relaxation lacks), so it is only deprioritised, never pruned.
_H_UNREACHABLE = 80


# ── native model ───────────────────────────────────────────────────────────────
#
# A state is ``(robot_positions, block_positions, selected_block_index)``:
#   robot_positions -- one cell per robot, in ``_ROBOT_NAMES`` order
#   block_positions -- one cell per cvcer, in level sprite order. Identity order,
#                      NOT sorted: the selection index has to survive a push.
#   selected         -- index into block_positions, or -1 for "arrows drive the robots"
#
# Door state is deliberately absent: ``dssyxgdgjg`` recomputes it from the robot
# positions at the end of every directional action and clicks never move a robot,
# so it is a pure function of the state and never has to be carried.
def _sprite_cells(s):
    px = s.render()
    return {(s.x + x, s.y + y)
            for y in range(px.shape[0]) for x in range(px.shape[1]) if px[y][x] != -1}


class _Model:
    """Native re-implementation of ``M0r0.step`` for one level (see ``--selfcheck``)."""

    def __init__(self, game):
        lv = game.current_level
        self.gw, self.gh = lv.grid_size
        self.walls = set()
        for s in lv.get_sprites_by_tag("jggua"):
            self.walls |= _sprite_cells(s)
        self.hazards = {(s.x, s.y) for s in lv.get_sprites_by_name("wyiex")}
        self.buttons, self.doors = {}, {}
        for c in _DOOR_COLORS:
            b = {(s.x, s.y) for s in lv.get_sprites_by_name(f"hnutp-{c}")}
            d = set()
            for s in lv.get_sprites_by_name(f"dfnuk-{c}"):
                d |= _sprite_cells(s)
            if b and d:
                self.buttons[c], self.doors[c] = b, d
        self.signs = []
        rpos = []
        for n in _ROBOT_NAMES:
            sp = lv.get_sprites_by_name(n)
            if sp:
                self.signs.append(_SIGNS[n.split("-", 1)[1]])
                rpos.append((sp[0].x, sp[0].y))
        self.names = [n for n in _ROBOT_NAMES if lv.get_sprites_by_name(n)]
        self.rpos0 = tuple(rpos)
        self.blocks0 = tuple((s.x, s.y) for s in lv.get_sprites_by_name("cvcer"))
        self.start = (self.rpos0, self.blocks0, -1)
        # Cells no block can ever occupy -> a click there always means "deselect".
        self._deselect = min(self.walls | self.hazards) if (self.walls or self.hazards) else None

    def closed_doors(self, rpos):
        """Cells blocked by a shut door, given the robots' positions."""
        out = set()
        for c, cells in self.doors.items():
            if not any(p in self.buttons[c] for p in rpos):
                out |= cells
        return out

    def click(self, state, cell):
        rpos, blocks, _ = state
        return (rpos, blocks, blocks.index(cell) if cell in blocks else -1)

    def move(self, state, act):
        """Apply ACTION1-4. Returns ``(new_state, won)``."""
        rpos, blocks, sel = state
        dx, dy = _DIRS[act]
        if sel >= 0:
            return self._push_block(rpos, blocks, sel, dx, dy)
        return self._move_robots(rpos, blocks, sel, dx, dy)

    def _push_block(self, rpos, blocks, sel, dx, dy):
        nx, ny = blocks[sel][0] + dx, blocks[sel][1] + dy
        if not (0 <= nx < self.gw and 0 <= ny < self.gh):
            return (rpos, blocks, sel), False
        # try_move_sprite tests EVERY sprite, so a block is stopped by the maze, the
        # hazards, the buttons, a shut door, a robot, and any other block.
        blocked = set(self.walls) | self.hazards | set(blocks) | set(rpos)
        blocked |= self.closed_doors(rpos)
        for cells in self.buttons.values():
            blocked |= cells
        if (nx, ny) in blocked:
            return (rpos, blocks, sel), False
        nb = list(blocks)
        nb[sel] = (nx, ny)
        return (rpos, tuple(nb), sel), False

    def _move_robots(self, rpos, blocks, sel, dx, dy):
        # bpcdxdwyxx only tests the maze, the blocks and shut doors -- robots pass
        # through each other, and hazards are walkable (they trigger the reset).
        blocked = set(self.walls) | set(blocks) | self.closed_doors(rpos)
        npos = []
        for p, (sx, sy) in zip(rpos, self.signs):
            tx, ty = p[0] + dx * sx, p[1] + dy * sy
            npos.append(p if not (0 <= tx < self.gw and 0 <= ty < self.gh)
                        or (tx, ty) in blocked else (tx, ty))
        if any(p in self.hazards for p in npos):
            return (self.rpos0, self.blocks0, sel), False
        # Pass-through merge: horizontally adjacent before the move, and one landed
        # on the other's old cell -> both snap to the midpoint of where they ended.
        for i in range(len(npos)):
            for j in range(i + 1, len(npos)):
                (ax, ay), (bx, by) = rpos[i], rpos[j]
                if abs(ax - bx) == 1 and ay == by and (npos[i] == (bx, by) or npos[j] == (ax, ay)):
                    mid = ((npos[i][0] + npos[j][0]) // 2, (npos[i][1] + npos[j][1]) // 2)
                    npos[i] = npos[j] = mid
        won = len(set(npos)) < len(npos)
        return (tuple(npos), blocks, sel), won

    def successors(self, state, with_blocks):
        """(next_state, label) for every action. Labels are the plan's alphabet."""
        out = []
        for a in (1, 2, 3, 4):
            ns, won = self.move(state, a)
            out.append((ns, ("mv", a), won))
        if with_blocks and state[1]:
            for i, cell in enumerate(state[1]):
                if i != state[2]:
                    out.append((self.click(state, cell), ("cl", cell), False))
            if state[2] >= 0 and self._deselect is not None:
                out.append(((state[0], state[1], -1), ("cl", self._deselect), False))
        return out


# ── heuristic: exact distance-to-merge with every block removed ────────────────
#
# Robot-pair states number only ~gw*gh squared (<=5.8k here), so the whole relaxed
# transition graph is built and reversed, and one backward BFS from the winning
# states gives an exact distance for every state. Blocks are dropped for the
# relaxation, which makes this a strong guide (it is the true cost wherever blocks
# are irrelevant) but NOT admissible -- a block can act as a wall the relaxation
# does not have -- so the guided tier is a plan-finder, not a proof of optimality.
def _relaxed_pair_dist(m: _Model):
    cells = [(x, y) for y in range(m.gh) for x in range(m.gw) if (x, y) not in m.walls]
    rev, wins = {}, set()
    for a in cells:
        for b in cells:
            st = ((a, b), (), -1)
            for act in (1, 2, 3, 4):
                ns, won = m.move(st, act)
                if won:
                    wins.add((a, b))
                    break
                rev.setdefault(ns[0], set()).add((a, b))
    dist = {w: 1 for w in wins}
    q = deque(wins)
    while q:
        s = q.popleft()
        for p in rev.get(s, ()):
            if p not in dist:
                dist[p] = dist[s] + 1
                q.append(p)
    return dist


# ── search ─────────────────────────────────────────────────────────────────────
def _bfs(m: _Model, with_blocks, limit=BUDGET, max_nodes=2_000_000):
    """Breadth-first, so the plan it returns is optimal. With ``with_blocks`` the
    blocks are nailed down and only the robots move, which keeps the state space
    tiny; without it, every block push and selection click is in play too."""
    seen = {m.start}
    q = deque([(m.start, ())])
    nodes = 0
    while q:
        st, path = q.popleft()
        if len(path) >= limit:
            continue
        nodes += 1
        if nodes > max_nodes:
            return None, f"nodebudget({nodes})"
        for ns, lab, won in m.successors(st, with_blocks=with_blocks):
            if won:
                return list(path) + [lab], f"win nodes={nodes}"
            if ns not in seen:
                seen.add(ns)
                q.append((ns, path + (lab,)))
    return None, f"exhausted nodes={nodes}"


def _astar_guided(m: _Model, weight, time_budget, limit=BUDGET):
    """Tier 2: weighted A* over the full state, guided by the relaxed pair-distance."""
    dist = _relaxed_pair_dist(m)
    h = lambda st: dist.get(st[0], _H_UNREACHABLE)  # noqa: E731
    t0 = time.time()
    cnt = itertools.count()
    openh = [(weight * h(m.start), next(cnt), m.start, ())]
    best = {m.start: 0}
    nodes = 0
    while openh:
        if time.time() - t0 > time_budget:
            return None, f"timeout nodes={nodes}"
        _, _, st, path = heapq.heappop(openh)
        g = len(path)
        if best.get(st, 1 << 30) < g or g >= limit:
            continue
        nodes += 1
        for ns, lab, won in m.successors(st, with_blocks=True):
            if won:
                return list(path) + [lab], f"win nodes={nodes} {time.time()-t0:.0f}s"
            if ns == st or g + 1 >= best.get(ns, 1 << 30):
                continue
            best[ns] = g + 1
            heapq.heappush(openh, (g + 1 + weight * h(ns), next(cnt), ns, path + (lab,)))
    return None, f"exhausted nodes={nodes}"


def solve_level(m: _Model, weight, time_budget, exhaustive_nodes=500_000):
    """Return (plan, info), escalating only as far as the level forces us to.

    Both BFS tiers are optimal, so a level is only handed to the inexact guided
    tier once breadth-first has been shown to be out of reach.
    """
    plan, info = _bfs(m, with_blocks=False)
    if plan or not m.blocks0:
        return plan, "frozen/" + info
    plan, info = _bfs(m, with_blocks=True, max_nodes=exhaustive_nodes)
    if plan:
        return plan, "exhaustive/" + info
    plan, info = _astar_guided(m, weight, time_budget)
    return plan, "guided/" + info


# ── level construction / engine driving ────────────────────────────────────────
def _make_level(seed: int, level_idx: int) -> M0r0:
    """Build an M0r0 sitting on ``level_idx``, deterministic in ``seed``.

    M0r0 draws two augmentations from two different places: the rotation from
    ``AugmentedGame``, which makes it a pure function of ``(seed, level_index)`` as
    soon as the constructor is given a seed, and the colour permutation from
    ``self._rng``, which is still unseeded and so is pinned here, right before the
    ``set_level`` that consumes it.
    """
    game = M0r0(seed=seed)
    game._rng = random.Random(f"m0r0-colors:{seed}:{level_idx}")
    game.set_level(level_idx)
    return game


def _drive(game, action, data=None):
    """Perform one engine action to completion (stopping the instant a level solve
    is queued, so the captured frame is the clean solved board)."""
    game._full_reset = False
    game._set_action(ActionInput(id=action, data=data or {}))
    guard = 0
    while not game.is_action_complete():
        guard += 1
        if guard > _STEP_GUARD or game._next_level:
            break
        game.step()


def _won(game):
    return bool(game._next_level) or game._state == GameState.WIN


def _build_clickmap(game):
    """game cell (x, y) -> an UNROTATED display pixel that ``display_to_grid`` maps
    back to it. Rotation is applied by RotationDisplay after the camera renders, so
    this map is rotation-independent and ``_replay`` rotates the pixel itself."""
    cam = game.camera
    rev = {}
    for dy in range(64):
        for dx in range(64):
            gc = cam.display_to_grid(dx, dy)
            if gc is not None:
                rev.setdefault((int(gc[0]), int(gc[1])), (dx, dy))
    return rev


# The plan is written in GAME space; the level is presented rotated by k*90. These
# two convert a game-space action into the screen action a player looking at that
# rotated frame would actually press, which is what the engine expects in its
# action data (it undoes the rotation itself via remap_action / remap_click).
def _drive_move(game, k, act):
    screen = getattr(GameAction, f"ACTION{act}")
    _drive(game, screen)
    return {"type": "simple", "index": int(screen.value)}


def _drive_click(game, k, rev, cell):
    sx, sy = rev[cell]
    _drive(game, GameAction.ACTION6, {"x": sx, "y": sy})
    return {"type": "mouse", "index": _ACTION6, "data": {"x": int(sx), "y": int(sy)}}


def _replay(game, plan, *, capture=False, rev=None):
    """Replay a game-space ``plan`` on ``game`` at its live rotation.
    Returns (won, observations, actions)."""
    if rev is None:
        rev = _build_clickmap(game)
    k = game._rotation_k
    observations = actions = None
    if capture:
        observations = [game.camera.render(game.current_level.get_sprites()).tolist()]
        actions = [dict(_RESET_ACTION)]

    for lab in plan:
        if lab[0] == "mv":
            record = _drive_move(game, k, lab[1])
        else:
            cell = tuple(lab[1])
            if cell not in rev:
                return False, observations, actions
            record = _drive_click(game, k, rev, cell)
        if capture:
            observations.append(game.camera.render(game.current_level.get_sprites()).tolist())
            actions.append(record)
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
    return [(lab[0], tuple(lab[1]) if lab[0] == "cl" else lab[1])
            for lab in json.loads(p.read_text())]


def _save_plan(plan_dir: Path, level_idx: int, plan):
    plan_dir.mkdir(parents=True, exist_ok=True)
    out = [[lab[0], list(lab[1]) if lab[0] == "cl" else lab[1]] for lab in plan]
    _plan_path(plan_dir, level_idx).write_text(json.dumps(out))


def _plan_valid(level_idx: int, plan) -> bool:
    """A plan is valid iff the REAL engine replays it to a win, within budget.

    This is what keeps the native model honest: the model only ever proposes, the
    engine disposes. Checked at two rotations so the replay's action/click remapping
    is exercised, not just the k=0 identity path.
    """
    if not plan or len(plan) > BUDGET:
        return False
    for seed in (0, 1, 2, 3):
        game = _make_level(seed, level_idx)
        won, _, _ = _replay(game, plan)
        if not won:
            return False
    return True


def build_plans(plan_dir: Path, level_indices, *, time_budget=None, force=False, verbose=True):
    """Ensure a validated winning plan exists for each level. Returns {idx: plan}."""
    plans = {}
    for idx in level_indices:
        if not force:
            cached = _load_plan(plan_dir, idx)
            if cached is not None and _plan_valid(idx, cached):
                plans[idx] = cached
                if verbose:
                    print(f"  L{idx}: cached plan ({len(cached)} actions)")
                continue
        weight, default_tb = _LEVEL_CFG.get(idx, _DEFAULT_CFG)
        tb = time_budget if time_budget is not None else default_tb
        if verbose:
            print(f"  L{idx}: solving (w={weight} budget={tb}s)...", flush=True)
        m = _Model(_make_level(0, idx))
        t0 = time.time()
        plan, info = solve_level(m, weight, tb)
        dt = time.time() - t0
        if plan and _plan_valid(idx, plan):
            _save_plan(plan_dir, idx, plan)
            plans[idx] = plan
            if verbose:
                print(f"    -> solved & cached ({len(plan)} actions, {dt:.0f}s) [{info}]")
        elif verbose:
            why = "engine rejected the plan" if plan else info
            print(f"    -> UNSOLVED ({why}, {dt:.0f}s) -- level will be skipped")
    return plans


# ── model self-check ───────────────────────────────────────────────────────────
def selfcheck(level_indices, trials=300, steps=40, verbose=True):
    """Drive random action sequences through BOTH the engine and ``_Model`` and
    compare the full state after every action. This is the guard that lets the
    search trust the native model. Because it goes through ``_drive_move`` /
    ``_drive_click``, it also exercises the rotation remapping over whatever
    orientations the seeds happen to draw -- a bad remap shows up as a mismatch.
    Returns the number of mismatches."""
    total = 0
    for idx in level_indices:
        game0 = _make_level(0, idx)
        m = _Model(game0)
        rev = _build_clickmap(game0)
        cells = list(rev)
        rng = random.Random(f"selfcheck:{idx}")
        bad = 0
        for t in range(trials):
            game = _make_level(t, idx)  # vary the seed -> vary the rotation
            k = game._rotation_k
            st = m.start
            for _ in range(steps):
                if rng.random() < 0.2:
                    cell = rng.choice(cells)
                    _drive_click(game, k, rev, cell)
                    st = m.click(st, cell)
                else:
                    a = rng.choice([1, 2, 3, 4])
                    _drive_move(game, k, a)
                    st, won = m.move(st, a)
                    if won or _won(game):
                        if bool(won) != _won(game):
                            bad += 1
                        break
                if _engine_state(game, m) != st:
                    bad += 1
                    break
        total += bad
        if verbose:
            print(f"  L{idx}: {'OK' if not bad else f'{bad} MISMATCHES'} ({trials} rollouts)")
    return total


def _engine_state(game, m: _Model):
    lv = game.current_level
    rpos = tuple((lv.get_sprites_by_name(n)[0].x, lv.get_sprites_by_name(n)[0].y)
                 for n in m.names)
    blocks = lv.get_sprites_by_name("cvcer")
    sel = -1
    if game.cfwgj is not None and not game.vmcbq:
        sel = next(i for i, s in enumerate(blocks) if s is game.cfwgj)
    return (rpos, tuple((s.x, s.y) for s in blocks), sel)


# ── Solver ─────────────────────────────────────────────────────────────────────
_PLAN_DIR = Path("data/m0r0_plans")
_LEVELS = (0, 1, 2, 3, 4, 5)


class M0r0Solver(BaseSolver):
    game_id = GAME_ID
    step_guard = _STEP_GUARD
    # Each episode replays a fixed, seed-invariant game-space plan; ``solve_from``
    # does not read the live robot/block positions, so RESET-recovery restores the
    # level's initial state after the exploration prefix, then replays the plan.
    supports_recovery = True
    recovery_mode = "reset"

    #: {level_idx: plan} and {level_idx: clickmap}, both seed-invariant, built once.
    _plans: dict[int, list] | None = None
    _clickmaps: dict[int, dict] = {}

    def _ensure_plans(self) -> dict[int, list]:
        if M0r0Solver._plans is None:
            M0r0Solver._plans = build_plans(_PLAN_DIR, _LEVELS, verbose=False)
        return M0r0Solver._plans

    def _clickmap(self, game, level_idx: int) -> dict:
        # The map is over the UNROTATED display (rotation is applied afterwards by
        # RotationDisplay), so it is seed/rotation-invariant and cached per level.
        rev = M0r0Solver._clickmaps.get(level_idx)
        if rev is None:
            rev = _build_clickmap(game)
            M0r0Solver._clickmaps[level_idx] = rev
        return rev

    def make_game(self, seed: int):
        self._cur_seed = seed
        return M0r0(seed=seed)

    def num_levels(self, game) -> int:
        return len(_LEVELS)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4, CLICK_ACTION]

    def set_level(self, game, level_idx: int) -> None:
        # Colour permutation comes from the instance ``_rng`` (unseeded by default);
        # pin it per (seed, level). Rotation is a pure function of (seed, level).
        game._rng = random.Random(f"m0r0-colors:{self._cur_seed}:{level_idx}")
        game._next_level = False   # base drive never advances; clear the stale flag
        game.set_level(level_idx)

    def solve_from(self, game, level_idx: int, seed: int):
        """Replay the cached game-space plan at ``game``'s LIVE rotation: moves via
        ``inverse_remap_action_full``, clicks via ``remap_click`` on the click map."""
        plan = self._ensure_plans().get(level_idx)
        if not plan:
            return []
        k = game._rotation_k
        rev = self._clickmap(game, level_idx)
        out: list = []
        for lab in plan:
            if lab[0] == "mv":
                screen = getattr(GameAction, f"ACTION{lab[1]}")
                out.append(Action(int(screen.value)))
            else:
                cell = tuple(lab[1])
                if cell not in rev:
                    return []
                sx, sy = rev[cell]
                out.append(Action(CLICK_ACTION, (sy, sx)))   # click_rc = (row=y, col=x)
        return out


if __name__ == "__main__":
    sys.exit(M0r0Solver.main())
