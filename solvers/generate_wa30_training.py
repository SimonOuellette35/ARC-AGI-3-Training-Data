"""Generate Phase-1 training data for the WA30 game (keyboard-only).

WA30 is a **drag-Sokoban with helper/adversary AI movers**. On a 64x64 board (everything is
grid-aligned to a 4px lattice) the avatar (tag ``wbmdvjhthc``) walks one cell per action and can
**grab and drag a single 4x4 block** (tag ``geezpjgiyd``) at a time. The level is won the instant
**every block sits on a "win" target cell** (the ``fsjjayjoeg`` sprites) and no block is currently
held (``ymzfopzgbq``). Two families of autonomous movers run after every player action:

  * **Helper movers** (``kdweefinfi``) grab a *free* block and shuttle it toward the nearest **win**
    target, releasing it once it lands there. Cooperative, but too slow to beat the budget alone.
  * **Adversary movers** (``ysysltqlke``) grab a free block -- *including a settled one* -- and drag
    it toward a **decoy** target (NOT a win cell). A surviving adversary makes multi-block levels
    unwinnable. The avatar destroys one by standing one cell away, facing it, and pressing ACTION5.

Mechanics (obfuscated ``games/wa30/wa30.py``; all verified against the engine):

  * **One action = one cell (4px).** ACTION1/2/3/4 = up/down/left/right; ACTION5 = grab / release /
    destroy-adversary. Every action costs one step whether or not the avatar moved.
  * **Grab needs facing.** ACTION1-4 set rotation even when the move is blocked; ACTION5 grabs the
    block faced. **Drag is a rigid pair** -- both destination cells must be free. Dots block the
    avatar but a dragged block may be pushed onto one.
  * **Static obstacles + dots partition the board.** On dot-maze levels the win targets sit in a
    region the avatar cannot enter, so those blocks can only be delivered by the helper movers.
  * **Budget.** ``StepCounter`` per level; the win test precedes it.

Per-seed geometry, so per-seed plans
-------------------------------------
``on_set_level`` randomises block starting positions, the palette, and the frame rotation, each a
pure function of ``(seed, level)`` (this generator patched the game to accept ``Wa30(seed=S)``).
The layout changes with the seed, so a plan is solved **on the fly** per (seed, level) -- never
cached across seeds. The solver is engine-blackbox: search nodes hold real ``Wa30`` states driven
one action at a time, so every returned plan is correct by construction.

The solver (``solve_level``): a subgoal-decomposed greedy planner
-----------------------------------------------------------------
  1. **Clear adversaries first** via single-step-MPC pursuit, capped.
  2. **Deliver one block per round** -- an exact BFS over the abstract drag model ``(avatar, block,
     holding?, facing)`` seats the most-constrained block on a free win cell; executed on the real
     engine and committed the instant the on-target count rises.
  3. **Wait / kill when nothing is draggable** -- a verified walk-up kill of a parked adversary, else
     idle against a wall to let the helper movers make progress.
  4. **Randomised restarts** (``solve_level_restarts``) escape greedy target-assignment dead-ends.

Rotation: a plan is game-space (the search pins its copy to k=0); the recorded actions are replayed
at the seed's LIVE rotation via ``inverse_remap_action_full`` (rotation only remaps ACTION1-4;
ACTION5 passes through) -- the m0r0/tu93 convention. In this BaseSolver port, ``solve_from`` returns
the *screen* plan directly, so the driven and recorded action indices are already in screen space.

Episode model: **per-level independent, subset-allowed (NOT all-or-nothing).** Each level is solved
and recorded on its OWN fresh ``Wa30(seed)`` instance (layout is a pure function of ``(seed,
level)``); an unsolved level is simply SKIPPED and the solved subset is still written. A seed's
episode "wins" if it solves at least ``min_levels`` levels. This is why ``solve_episode`` is
overridden -- the base's single-instance all-or-nothing loop does not fit. ``supports_recovery`` is
False: a dragged block or a stolen block can become permanently unwinnable, so a perturbed state is
not recoverable and exploration stays off.

Demos + priors. L7/L8 rarely auto-solve; human demos (from ``play_wa30.py``, in ``data/wa30_demos``)
cover them. ``mine_demo_priors`` distils demos into SEED-INDEPENDENT per-level target priors (which
win cells to fill, in what order), which steer the greedy solver on the big levels; paths are still
re-derived per seed against the live engine, so every plan stays engine-verified.

Action schema (same corpus convention as ka59 / m0r0 / tu93):

    RESET :  {"type": "simple", "index": 0}      (leading action for obs[0])
    move  :  {"type": "simple", "index": 1|2|3|4}
    act5  :  {"type": "simple", "index": 5}

Usage (run from the repo root, with the arcengine conda env):

    python solvers/generate_wa30_training.py --episodes 1000 --out data/training_multi_level/wa30
"""

from __future__ import annotations

import copy
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

# Repo root (parent of solvers/) -- that's where the games/ package lives.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameAction, GameState  # noqa: E402
from games.wa30.wa30 import Wa30  # noqa: E402
from utils.rotation import inverse_remap_action_full  # noqa: E402
from solvers.base_solver import BaseSolver, DriveResult, _ID_TO_GAMEACTION  # noqa: E402
from utils.explore import Action, EpsilonSchedule, ExplorationPolicy  # noqa: E402

GAME_ID = "wa30"
STEP = 4  # celomdfhbh: the board lattice / one cell
_STEP_GUARD = 3000

# Sprite tags (obfuscated names kept verbatim so they can be grepped in the game source).
_TAG_PLAYER = "wbmdvjhthc"
_TAG_BLOCK = "geezpjgiyd"
_TAG_HELPER = "kdweefinfi"    # cooperative: drags free blocks onto WIN targets
_TAG_ADVERSARY = "ysysltqlke"  # steals settled blocks onto DECOY targets; destroyable

# action index -> (dx, dy) and the rotation `pjedoipwee` assigns to that move.
_DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
# facing rotation -> the cell offset of the block the avatar faces (from vwiozbtqgi).
_FACE = {0: (0, -STEP), 180: (0, STEP), 90: (STEP, 0), 270: (-STEP, 0)}


def _rot_of(dx: int, dy: int) -> int:
    """Mirror of `pjedoipwee`: which rotation a (dx, dy) press sets."""
    if dy < 0:
        return 0
    if dx > 0:
        return 90
    if dy > 0:
        return 180
    return 270


# ── engine helpers ─────────────────────────────────────────────────────────────
def _game_action(index: int) -> GameAction:
    """``GameAction`` member for an action index (reverse lookup by ``.value`` is broken)."""
    return getattr(GameAction, f"ACTION{index}")


def _fast_copy(game):
    """Deep-copy the game, SHARING the untouched levels and the clean-level templates (sc25 trick)."""
    memo = {id(game._clean_levels): game._clean_levels}
    for i, level in enumerate(game._levels):
        if i != game._current_level_index:
            memo[id(level)] = level
    return copy.deepcopy(game, memo)


def _drive_engine(game, index: int, *, render: bool = False):
    """Run one *screen* action to completion. Returns (frame_or_None, solved, dead)."""
    game._full_reset = False
    game._set_action(ActionInput(id=_game_action(index)))
    guard = 0
    while not game.is_action_complete():
        if guard > _STEP_GUARD or game._next_level:
            break
        game.step()
        guard += 1
    frame = None
    if render:
        frame = np.asarray(game.camera.render(game.current_level.get_sprites())).tolist()
    solved = bool(game._next_level) or game._state == GameState.WIN
    return frame, solved, game._state == GameState.GAME_OVER


def _blocks(game):
    return game.current_level.get_sprites_by_tag(_TAG_BLOCK)


def _player(game):
    return game.current_level.get_sprites_by_tag(_TAG_PLAYER)[0]


def _grid_targets(game):
    """Win-target cells a 4-grid block can actually rest on (lattice-aligned only)."""
    return set(c for c in game.wyzquhjerd if c[0] % STEP == 0 and c[1] % STEP == 0)


def _on_target_count(game):
    """Blocks currently seated on a win cell and not held -- the win progress measure."""
    gt = _grid_targets(game)
    return sum(1 for b in _blocks(game) if (b.x, b.y) in gt and b not in game.zmqreragji)


# ── abstract drag model (planning only; every plan is engine-verified before commit) ────
def _bfs_deliver(game, block_xy, targets):
    """Shortest action sequences to seat ``block_xy`` on each cell in ``targets`` and release."""
    p = _player(game)
    pk = set(game.pkbufziase)
    dots = set(game.qthdiggudy)
    pk_static = pk - {(p.x, p.y), block_xy}
    targetset = set(targets)
    start = (p.x, p.y, block_xy[0], block_xy[1], 0, p.rotation)
    came = {start: None}
    q = deque([start])
    results = {}

    def blocked_ungrab(cx, cy, bx, by):
        return (cx, cy) in pk_static or (cx, cy) in dots or (cx, cy) == (bx, by)

    while q:
        st = q.popleft()
        px, py, bx, by, gr, rot = st
        if gr == 0 and (bx, by) in targetset and (bx, by) not in results:
            acts, cur = [], st
            while came[cur] is not None:
                prev, a = came[cur]
                acts.append(a)
                cur = prev
            results[(bx, by)] = acts[::-1]
            if len(results) == len(targetset):
                break
        for a, (dx, dy) in _DIRS.items():
            nrot = _rot_of(dx, dy)
            if gr == 0:
                npx, npy = px + dx, py + dy
                ns = (px, py, bx, by, 0, nrot) if blocked_ungrab(npx, npy, bx, by) \
                    else (npx, npy, bx, by, 0, nrot)
            else:
                npx, npy = px + dx, py + dy
                nbx, nby = bx + dx, by + dy
                okp = ((npx, npy) not in pk_static or (npx, npy) == (bx, by)) and (npx, npy) not in dots
                okb = ((nbx, nby) not in pk_static or (nbx, nby) == (px, py))
                ns = (npx, npy, nbx, nby, 1, rot) if (okp and okb) else (px, py, bx, by, 1, rot)
            if ns not in came:
                came[ns] = (st, a)
                q.append(ns)
        if gr == 0:  # ACTION5 grab if facing the block
            fdx, fdy = _FACE[rot]
            if (px + fdx, py + fdy) == (bx, by):
                ns = (px, py, bx, by, 1, rot)
                if ns not in came:
                    came[ns] = (st, 5)
                    q.append(ns)
        else:  # ACTION5 release
            ns = (px, py, bx, by, 0, rot)
            if ns not in came:
                came[ns] = (st, 5)
                q.append(ns)
    return results


def _pin_barrier_dir(game):
    """An action index that keeps the avatar pinned (moves into a barrier), else None."""
    p = _player(game)
    pk = set(game.pkbufziase)
    dots = set(game.qthdiggudy)
    for a, (dx, dy) in _DIRS.items():
        c = (p.x + dx, p.y + dy)
        if c in pk or c in dots:
            return a
    return None


def _bfs_face_adversary(game):
    """BFS (avatar-only) to a cell facing a ``ysysltqlke`` mover, then ACTION5 to destroy it."""
    advs = game.current_level.get_sprites_by_tag(_TAG_ADVERSARY)
    if not advs:
        return None
    p = _player(game)
    pk = set(game.pkbufziase)
    dots = set(game.qthdiggudy)
    pk_static = pk - {(p.x, p.y)}
    advpos = set((s.x, s.y) for s in advs)
    start = (p.x, p.y, p.rotation)
    came = {start: None}
    q = deque([start])
    goal = None

    def blocked(cx, cy):
        return (cx, cy) in pk_static or (cx, cy) in dots

    while q:
        st = q.popleft()
        px, py, rot = st
        fdx, fdy = _FACE[rot]
        if (px + fdx, py + fdy) in advpos:
            goal = st
            break
        for a, (dx, dy) in _DIRS.items():
            nrot = _rot_of(dx, dy)
            npx, npy = px + dx, py + dy
            ns = (px, py, nrot) if blocked(npx, npy) else (npx, npy, nrot)
            if ns not in came:
                came[ns] = (st, a)
                q.append(ns)
    if goal is None:
        return None
    acts, cur = [], goal
    while came[cur] is not None:
        prev, a = came[cur]
        acts.append(a)
        cur = prev
    return acts[::-1] + [5]


def _try_kill_adversary(game, cap=45):
    """Single-step MPC toward the nearest adversary, but only COMMIT if it destroys one."""
    n0 = len(game.current_level.get_sprites_by_tag(_TAG_ADVERSARY))
    if n0 == 0:
        return None
    gg = _fast_copy(game)
    acts = []
    for _ in range(cap):
        path = _bfs_face_adversary(gg)
        if path is None:
            return None
        step_a = path[0]
        _, _, dead = _drive_engine(gg, step_a)
        acts.append(step_a)
        if dead:
            return None
        if len(gg.current_level.get_sprites_by_tag(_TAG_ADVERSARY)) < n0:
            return gg, acts
    return None


def _execute_until_progress(game, acts, base):
    """Drive ``acts`` on a copy; stop the instant the on-target count exceeds ``base`` or the level
    is won. Returns (game2, gained, dead, used)."""
    gg = _fast_copy(game)
    for i, a in enumerate(acts):
        _, s, d = _drive_engine(gg, a)
        if d:
            return gg, False, True, i + 1
        if s or _on_target_count(gg) > base:
            return gg, True, False, i + 1
    return gg, _on_target_count(gg) > base, False, len(acts)


def solve_level(game, *, verbose=False, max_rounds=None, wait_k=14, rng=None, prior=None,
                assign=True):
    """Greedy subgoal solver over the real engine. Returns a game-space plan or None."""
    rank = _prior_rank(prior) if prior else {}
    _UNRANKED = 1 << 30
    _ASSIGN_INF = 1e9
    g = _fast_copy(game)
    g._rotation_k = 0
    budget = g.kuncbnslnm.current_steps
    nb = len(_blocks(g))
    if max_rounds is None:
        max_rounds = nb * 8 + 30
    plan = []

    # Phase 0: destroy adversaries via single-step MPC pursuit (they move every action).
    def pursue(g, plan, cap):
        used, best, since = 0, None, 0
        while g.current_level.get_sprites_by_tag(_TAG_ADVERSARY) and used < cap:
            acts = _bfs_face_adversary(g)
            if acts is None:
                return g, False
            d_now = len(acts)
            if best is None or d_now < best:
                best, since = d_now, 0
            else:
                since += 1
            if since >= 6:
                return g, False
            _, _, dead = _drive_engine(g, acts[0])
            plan.append(acts[0])
            used += 1
            if dead:
                return g, True
        return g, False

    if g.current_level.get_sprites_by_tag(_TAG_ADVERSARY):
        g, dead = pursue(g, plan, cap=min(30, max(12, budget // 4)))
        if dead:
            if verbose:
                print("   died pursuing adversary")
            return None
        if len(plan) > budget:
            return None

        # Clear EVERY remaining adversary before delivering. Alternate verified walk-up kills (which
        # commit ONLY on success) with a single idle step to let the adversary reposition.
        kill_cap = min(budget // 2, 80)
        spent_here = 0
        while g.current_level.get_sprites_by_tag(_TAG_ADVERSARY) and spent_here < kill_cap:
            kill = _try_kill_adversary(g)
            if kill is not None:
                gk, kacts = kill
                if len(plan) + len(kacts) > budget:
                    break
                plan += kacts
                g = gk
                spent_here += len(kacts)
                if verbose:
                    print(f"   opening kill ({len(kacts)}) -> total {len(plan)}/{budget}")
                continue
            pin = _pin_barrier_dir(g)
            _, solved, dead = _drive_engine(g, pin if pin is not None else 5)
            plan.append(pin if pin is not None else 5)
            spent_here += 1
            if dead or len(plan) > budget:
                return None
            if solved:
                return plan

    rounds = 0
    while _on_target_count(g) < nb:
        rounds += 1
        if rounds > max_rounds:
            if verbose:
                print(f"   too many rounds at {_on_target_count(g)}/{nb}")
            return None
        gt = _grid_targets(g)
        occ = set((b.x, b.y) for b in _blocks(g) if (b.x, b.y) in gt and b not in g.zmqreragji)
        free_t = [c for c in gt if c not in occ]
        base = _on_target_count(g)

        # per-block reachable free targets, most-constrained-first
        perblock = []
        for b in _blocks(g):
            if (b.x, b.y) in gt and b not in g.zmqreragji:
                continue
            res = _bfs_deliver(g, (b.x, b.y), free_t)
            if res:
                perblock.append(((b.x, b.y), res))
        # Global block->target matching (``assign``, on by default). A matching reserves a distinct
        # cell per block instead of the locally-cheapest greedy delivery.
        assigned = {}
        if assign and perblock:
            cols = sorted({t for _, res in perblock for t in res})
            col_ix = {t: i for i, t in enumerate(cols)}
            cost = np.full((len(perblock), len(cols)), _ASSIGN_INF, dtype=float)
            for r, (_, res) in enumerate(perblock):
                for t, acts in res.items():
                    c = len(acts) * 64 + min(rank.get(t, 63), 63)
                    if rng is not None:
                        c += rng.randint(0, 64)
                    cost[r, col_ix[t]] = c
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] < _ASSIGN_INF:
                    # key by BLOCK CELL, not row index -- perblock is shuffled/sorted below
                    assigned[perblock[r][0]] = cols[c]
        if rng is not None:
            rng.shuffle(perblock)
        if rank:
            # Honour the demo's fill ORDER, then most-constrained-first as before.
            perblock.sort(key=lambda pr: (min(rank.get(t, _UNRANKED) for t in pr[1]),
                                          len(pr[1]), min(len(a) for a in pr[1].values())))
        else:
            perblock.sort(key=lambda pr: (len(pr[1]), min(len(a) for a in pr[1].values())))

        committed = False
        for bxy, res in perblock:
            cands = sorted(res.items(), key=lambda kv: (rank.get(kv[0], _UNRANKED), len(kv[1])))
            if bxy in assigned:
                # try this block's reserved cell first, keep the rest as a recovery path
                cands.sort(key=lambda kv: kv[0] != assigned[bxy])
            for txy, acts in cands:
                gg, gained, dead, used = _execute_until_progress(g, acts, base)
                if dead or not gained:
                    continue
                plan += acts[:used]
                g = gg
                committed = True
                if verbose:
                    print(f"   {bxy}->{txy} used {used}/{len(acts)} -> "
                          f"{_on_target_count(g)}/{nb} total {len(plan)}/{budget}")
                break
            if committed:
                break

        if not committed and g.current_level.get_sprites_by_tag(_TAG_ADVERSARY):
            # Deliveries stalled with an adversary alive -- try a verified walk-up kill.
            kill = _try_kill_adversary(g)
            if kill is not None:
                gk, kacts = kill
                if len(plan) + len(kacts) <= budget:
                    plan += kacts
                    g = gk
                    committed = True
                    if verbose:
                        print(f"   killed adversary ({len(kacts)}) -> total {len(plan)}/{budget}")

        if not committed:
            # WAIT: let helper movers make progress. Idle for as long as the budget allows; a no-gain
            # result is discarded (not committed), so it spends nothing.
            pin = _pin_barrier_dir(g)
            burst = max(wait_k, budget - len(plan))
            wacts = [pin if pin is not None else 5] * burst
            gg, gained, dead, used = _execute_until_progress(g, wacts, base)
            if gained and not dead:
                plan += wacts[:used]
                g = gg
                committed = True
                if verbose:
                    print(f"   WAIT used {used} -> {_on_target_count(g)}/{nb} total {len(plan)}/{budget}")

        if not committed:
            if verbose:
                print(f"   no progress at {_on_target_count(g)}/{nb}")
            return None
        if len(plan) > budget:
            if verbose:
                print(f"   over budget {len(plan)} > {budget}")
            return None
        if g._next_level or g._state == GameState.WIN:
            return plan
    return plan


def solve_level_restarts(game, *, attempts=4, time_limit=15.0, prior=None, assign=True,
                         verbose=False):
    """``solve_level`` with randomised restarts to escape greedy target-assignment dead-ends.

    With a ``prior``, even attempts follow the demo's fill order and odd attempts drop it -- so the
    prior can only ever ADD coverage.
    """
    t0 = time.time()
    for i in range(attempts):
        if i > 0 and time.time() - t0 > time_limit:
            break
        rng = None if i == 0 else random.Random(9973 + i)
        use = prior if (prior and i % 2 == 0) else None
        plan = solve_level(game, verbose=(verbose and i == 0), rng=rng, prior=use, assign=assign)
        if plan is not None:
            return plan
    return None


# ── demos + priors ───────────────────────────────────────────────────────────────
def load_demos(demo_dir):
    """Load human demonstrations -> {(seed, level): game-space plan} for WON demos only."""
    demos = {}
    if not demo_dir:
        return demos
    demo_dir = Path(demo_dir)
    if not demo_dir.exists():
        return demos
    import json as _json
    for path in sorted(demo_dir.glob("wa30_seed*_level*_plan.json")):
        try:
            d = _json.loads(path.read_text())
        except Exception:
            continue
        if d.get("game_id") == GAME_ID and d.get("won") and "plan" in d:
            demos[(int(d["seed"]), int(d["level"]))] = [int(a) for a in d["plan"]]
    return demos


def mine_demo_priors(demos):
    """Distil per-(seed, level) demos into SEED-INDEPENDENT per-level target priors.

    Replays each demo at its LIVE rotation and, for the demos that still win, records the win cells
    occupied at the win plus the first step each was filled. Returns ``(priors, stale)`` where each
    level maps to ``{"core": [...], "extra": [...], "n_demos": k}`` (core = the subset EVERY demo
    agreed on; extra = the rest of the union). ``stale`` lists demos that no longer replay to a win.
    """
    per_level = {}
    stale = []
    for (seed, lvl), plan in sorted(demos.items()):
        g = Wa30(seed=seed)
        g.set_level(lvl)
        k = g._rotation_k
        gt = _grid_targets(g)
        first, occ, won = {}, set(), False
        for i, a in enumerate(plan):
            _, solved, dead = _drive_engine(g, int(_game_action(a).value))
            if dead:
                break
            cur = set((b.x, b.y) for b in _blocks(g)
                      if (b.x, b.y) in gt and b not in g.zmqreragji)
            for c in cur - occ:
                first.setdefault(c, i)   # cells churn (a mover can steal one back); keep the first
            occ = cur
            if solved:
                won = True
                break
        if won:
            per_level.setdefault(lvl, []).append((occ, first))
        else:
            stale.append((seed, lvl))

    priors = {}
    for lvl, samples in per_level.items():
        finals = [s[0] for s in samples]
        core = set.intersection(*finals)
        union = set().union(*finals)

        def mean_first(c, samples=samples):
            ts = [s[1][c] for s in samples if c in s[1]]
            return sum(ts) / len(ts) if ts else float("inf")

        priors[lvl] = {
            "core": sorted(core, key=mean_first),
            "extra": sorted(union - core, key=mean_first),
            "n_demos": len(samples),
        }
    return priors, stale


def _prior_rank(prior):
    """Win cell -> fill-order rank. Cells no demo used are absent (they sort last, not excluded)."""
    rank = {}
    for i, c in enumerate(prior.get("core", ())):
        rank[tuple(c)] = i
    off = len(rank)
    for i, c in enumerate(prior.get("extra", ())):
        rank[tuple(c)] = off + i
    return rank


# ── the solver ───────────────────────────────────────────────────────────────────
class Wa30Solver(BaseSolver):
    """Drag-Sokoban with helper/adversary AI movers. Per-(seed, level) engine-blackbox greedy
    planner; every level solved on the fly (layout is seed-specific). Subset-allowed: an unsolved
    level is skipped and the solved subset is still written.

    ``supports_recovery`` stays False: a dragged/stolen block can become permanently unwinnable, so
    the state is not recoverable from an arbitrary perturbation -- exploration stays off.
    """

    game_id = GAME_ID
    step_guard = _STEP_GUARD
    # RESET-recovery: exploration may drag/steal a block into a permanently
    # unwinnable configuration, so recovery is NOT replan -- it is one RESET back to
    # the level's initial state (``reset_level`` -> the AugmentedGame ``set_level``
    # re-clones the level and ``on_set_level`` re-seeds block placement AND refills
    # the StepCounter), after which the engine-verified plan replays to the win.
    supports_recovery = True
    recovery_mode = "reset"
    # The explore prefix can spend the whole step budget before the reset commits;
    # allow a couple of extra RESETs so a lethal opening never fails the level.
    max_resets = 5

    def __init__(self, *args, attempts: int = 4, time_limit: float = 20.0, min_levels: int = 1,
                 demo_dir: Path | None = Path("data/wa30_demos"), use_demos: bool = True,
                 use_priors: bool = True, level_filter=None, verbose: bool = False,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._attempts = attempts
        self._time_limit = time_limit
        self._min_levels = min_levels
        self._demo_dir = demo_dir
        self._use_demos = use_demos
        self._use_priors = use_priors
        self._level_filter = None if level_filter is None else set(level_filter)
        self._verbose = verbose
        self._demos = None            # lazy: {(seed, level): game-space plan}
        self._priors = None           # lazy: {level: prior dict}

    def make_game(self, seed: int):
        return Wa30(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4, 5]

    def _ensure_demos(self) -> None:
        """Load demos and mine seed-independent priors once (dropping stale demos)."""
        if self._demos is not None:
            return
        demos = load_demos(self._demo_dir) if self._use_demos else {}
        priors = {}
        if demos and self._use_priors:
            priors, stale = mine_demo_priors(demos)
            for key in stale:
                del demos[key]
        self._demos, self._priors = demos, priors

    def solve_from(self, game, level_idx: int, seed: int):
        """Return the SCREEN-space plan for ``game``'s current (fresh) level.

        A matching human demo (game-space) is replayed as-is; otherwise the greedy solver runs on
        the live game (via non-mutating copies) with any mined target prior. The game-space plan is
        then rotated to the seed's live screen space so the driven and recorded action indices match
        the original ``record_level``. Empty list = no plan -> this level is skipped by
        ``solve_episode``.
        """
        self._ensure_demos()
        plan = self._demos.get((seed, level_idx))
        if plan is None:
            plan = solve_level_restarts(game, attempts=self._attempts, time_limit=self._time_limit,
                                        prior=self._priors.get(level_idx), verbose=self._verbose)
        if plan is None:
            return []
        k = game._rotation_k
        return [Action(int(_game_action(a).value)) for a in plan]

    def drive(self, game, action: Action) -> DriveResult:
        """One screen action against the real engine, rendered AFTER the action completes."""
        game._full_reset = False
        game._set_action(ActionInput(id=_ID_TO_GAMEACTION[action.action_id]))
        guard = 0
        while not game.is_action_complete():
            if guard > self.step_guard or game._next_level:
                break
            game.step()
            guard += 1
        frame = np.asarray(game.camera.render(game.current_level.get_sprites()))
        solved = bool(game._next_level) or game._state == GameState.WIN
        dead = game._state == GameState.GAME_OVER
        return DriveResult(frame, solved, dead)

    def solve_episode(self, seed: int, explore: bool = True):
        """Play one seed level-by-level on INDEPENDENT fresh instances, keeping the solved subset.

        Each level is solved+recorded on its own ``Wa30(seed)`` (layout is a pure function of
        ``(seed, level)``), so a skipped level does not block the rest -- unlike the base's
        single-instance all-or-nothing loop. A level not in ``level_filter`` is attempted only if a
        human demo covers it (a demo costs no solve time). The episode "wins" (ok=True) if at least
        ``min_levels`` levels solved.

        RESET-recovery is wired in the base-``solve_episode`` way: the epsilon schedule + coverage
        policy are built ONCE (when ``explore and supports_recovery``) so the exploration arc spans
        the whole episode, and the base ``record_level`` runs the loop per level -- explore prefix,
        ONE RESET to the level's initial state (blocks re-seeded, StepCounter refilled), then the
        engine-verified plan replays to the win. The custom ``drive`` override is what
        ``record_level`` uses to step the engine, so the driven/recorded action indices stay in the
        seed's live screen space, exactly as before.
        """
        self._ensure_demos()
        template = self.make_game(seed)
        n_levels = self.num_levels(template)
        do_explore = explore and self.supports_recovery
        schedule = (EpsilonSchedule(self.rng, center=self._explore_center,
                                    jitter=self._explore_jitter)
                    if do_explore else None)
        exploration = (ExplorationPolicy(self.available_actions(template), self.rng)
                       if do_explore else None)
        out_levels: list[dict] = []
        for idx in range(n_levels):
            if self._level_filter is not None and idx not in self._level_filter \
                    and (seed, idx) not in self._demos:
                continue
            game = self.make_game(seed)
            try:
                obs, acts = self.record_level(game, idx, seed,
                                              schedule=schedule, exploration=exploration)
            except Exception:                    # noqa: BLE001 -- a bad level just skips
                obs, acts = None, None
            if obs is None:
                continue
            out_levels.append({"level_id": idx, "observations": obs, "actions": acts})
        return (len(out_levels) >= self._min_levels), out_levels

    # ── CLI: add the WA30-specific knobs on top of the base argparser ────────────────
    @classmethod
    def build_argparser(cls):
        p = super().build_argparser()
        p.add_argument("--attempts", type=int, default=4,
                       help="Randomised solver restarts per level.")
        p.add_argument("--time-limit", type=float, default=20.0,
                       help="Soft per-level wall-clock budget for restarts (seconds).")
        p.add_argument("--min-levels", type=int, default=1,
                       help="Skip a seed whose episode solves fewer than this many levels.")
        p.add_argument("--demo-dir", type=Path, default=Path("data/wa30_demos"),
                       help="Directory of human demos from play_wa30.py (default data/wa30_demos).")
        p.add_argument("--no-demos", action="store_true",
                       help="Ignore --demo-dir and solve every level with the auto-solver.")
        p.add_argument("--no-priors", action="store_true",
                       help="Do not distil demos into seed-independent target priors.")
        p.add_argument("--levels", type=str, default=None,
                       help="Comma-separated level indices to ATTEMPT (default: all). Demo-covered "
                            "levels are always included regardless.")
        return p

    @classmethod
    def main(cls, argv=None) -> int:
        args = cls.build_argparser().parse_args(argv)
        level_filter = None
        if args.levels:
            level_filter = [int(x) for x in args.levels.split(",") if x.strip() != ""]
        solver = cls(rng=random.Random(args.seed), burst_prob=args.noise, burst_mean=args.burst,
                     attempts=args.attempts, time_limit=args.time_limit,
                     min_levels=args.min_levels, demo_dir=args.demo_dir,
                     use_demos=not args.no_demos, use_priors=not args.no_priors,
                     level_filter=level_filter)
        out = args.out or (Path("data/training_multi_level") / cls.game_id)
        return solver.run(args.episodes, out, start_seed=args.start_seed,
                          max_attempts=args.max_attempts,
                          progress_every=args.progress_every,
                          explore=not args.no_exploration)


if __name__ == "__main__":
    sys.exit(Wa30Solver.main())
