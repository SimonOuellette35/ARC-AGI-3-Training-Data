"""Generate Phase-1 training data for the R11L game (games/r11l/r11l.py).

R11L is a *mouse-click* "centre-of-mass" puzzle. Each coloured **body** (``bdkaz-*``,
a 5x5 disc) is rigidly pinned to the **centroid of its own legs** (``bdkazLeg-*``,
5x5 diamonds); the body itself is never touched directly. One leg is *selected* at
a time (it renders with colour 0 instead of 3). An ACTION6 click:

  * inside ANY leg's 5x5 bounding box  -> selects that leg          (1 action)
  * anywhere else                      -> teleports the selected leg
                                          so its CENTRE lands on the clicked cell
                                          (1 action)

The click coordinate IS the new leg centre, and ``body_centre = floor(mean(leg
centres))`` per axis, so moving one of *n* legs by *d* drags the body by *d/n*.
Movement is a teleport, not a slide: only the DESTINATION is validated, there is
no path.

Level is won when every group that owns a target (``kzeze-*``, a 7x7 ring) has its
body overlapping that target -- pixel-perfect, and the ring's interior counts, so
the tolerance is roughly +/-4 cells of centre-to-centre offset. Groups whose name
contains ``anqcf`` are decoys and are excluded from the win check.

Constraints the solver models exactly:
  * a leg's destination must not overlap a ``bvzgd-*`` wall (else the click is
    consumed with no move);
  * a destination click must not land inside another leg's bbox (that would
    *select* that leg instead of moving);
  * after every move NO body may overlap a ``qtwnv-*`` hazard -- that costs a
    strike (5 strikes = loss) and rewinds the move;
  * 60 actions per level.

Levels 5-6 (idx 4, 5) add **paint**: their bodies (``bdkaz-yukft*``) start blank and
stamp any ``xigcb-*`` piece they come to rest on (pieces overwrite index-aligned,
and are consumed). Such a body has no target of its own; instead a target with no
body is satisfied by ANY blank-body whose *colour set* equals the target's colour
set. So the solver picks an ordered piece recipe per target, then routes the body
to rest on each piece in turn -- never resting on a piece it does not want, since
pickups are irreversible. Only a few dozen (body -> target, recipe) assignments are
viable, so all of them are routed and the cheapest is kept.

Solver
------
Geometry is IDENTICAL across seeds (``on_set_level`` randomises only colours and the
display rotation -- never a position), so each level is solved ONCE and the plan is
cached in ``data/r11l_plans/levelN.json`` as rotation-free GRID clicks, then replayed
for every seed. Colour randomisation is a *bijection*, so colour-set matching -- and
therefore the piece recipe -- is seed-invariant. Per seed the cached grid clicks are
inverse-rotated into that level's display space.

Routing is an A* over leg-centre tuples driven by precomputed 64x64 masks (leg-vs-
wall, body-vs-hazard, body-vs-target, body-vs-piece); the resulting plan is REPLAYED
against the real engine and only an engine-verified win is ever recorded.

The win only needs the body to *touch* its target, so most landings are a wide tie the
router must break. It breaks them towards the most CENTRED landing (``_centre_cost``),
which reads as deliberate rather than as a lucky graze -- but strictly among landings
that already cost the same number of moves. Two landings still finish off-centre
because the grid runs out: no click can drag the body any closer in one move (level 2's
``kpaac``), or in two (level 6's ``kpaaczjgrppgkne``); centring those would cost clicks.

Action schema (mixed simple + mouse, matching gp01 / cd82 / ka59)
    RESET / simple :  {"type": "simple", "index": k}
    mouse click    :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}

Usage (run from the repo root, with the ARC-AGI-3 conda python -- see the
`arc-agi-3-run-env` note):
    python solvers/generate_r11l_training.py --episodes 1000 \
        --out data/training_multi_level/r11l
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from arcengine import ActionInput, GameAction, GameState  # noqa: E402

from games.r11l.r11l import R11l  # noqa: E402

from solvers.base_solver import BaseSolver  # noqa: E402
from utils.explore import Action, CLICK_ACTION  # noqa: E402

GAME_ID = "r11l"
_RESET_ACTION = {"type": "simple", "index": int(GameAction.RESET.value)}
_ACTION6 = int(GameAction.ACTION6.value)
_STEP_GUARD = 4000
_NUM_LEVELS = 6
_PLAN_DIR = Path(__file__).resolve().parent.parent / "data" / "r11l_plans"

# Padded occupancy frame: level sprites (walls/hazards) hang off the 64x64 grid.
PAD = 12
PSIZE = 64 + 2 * PAD

# A leg centre is a click coordinate, so it is always in [0, 63]; therefore
# body_centre = floor(mean(leg centres)) is always in [0, 63] too -- no clipping.
_GRID = np.arange(64)
_PX = np.broadcast_to(_GRID[None, :], (64, 64))  # _PX[y, x] == x
_PY = np.broadcast_to(_GRID[:, None], (64, 64))  # _PY[y, x] == y


# ── mask helpers ──────────────────────────────────────────────────────────────
def _solid(sprite) -> np.ndarray:
    """Rendered non-transparent mask. -2 (ring interior) counts as solid, exactly
    as ``Sprite.collides_with`` does."""
    return np.asarray(sprite.render()) != -1


def _stamp(occ: np.ndarray, sprite) -> None:
    """OR a sprite's solid mask into a padded occupancy frame."""
    m = _solid(sprite)
    h, w = m.shape
    x0, y0 = sprite.x + PAD, sprite.y + PAD
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(PSIZE, x0 + w), min(PSIZE, y0 + h)
    if xs0 >= xs1 or ys0 >= ys1:
        return
    occ[ys0:ys1, xs0:xs1] |= m[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]


def _occ_of(sprites) -> np.ndarray:
    occ = np.zeros((PSIZE, PSIZE), bool)
    for s in sprites:
        _stamp(occ, s)
    return occ


def _hit_map(shape: np.ndarray, occ: np.ndarray) -> np.ndarray:
    """bool[64, 64] indexed [gy][gx]: a sprite with `shape` whose CENTRE sits on grid
    (gx, gy) overlaps `occ`."""
    h, w = shape.shape
    cy, cx = h // 2, w // 2
    res = np.zeros((64, 64), bool)
    for dy in range(h):
        for dx in range(w):
            if not shape[dy, dx]:
                continue
            ys = _GRID - cy + dy + PAD
            xs = _GRID - cx + dx + PAD
            res |= occ[ys[:, None], xs[None, :]]
    return res


def _centre_cost(sprite) -> np.ndarray:
    """int[64, 64] indexed [gy][gx]: squared distance from a body centred on (gx, gy)
    to being concentric with `sprite`.

    Every cell of a `_hit_map` is an equally valid landing as far as the game is
    concerned (any overlap wins), so the router uses this to break that tie towards a
    centred landing -- it never costs a move.
    """
    h, w = _solid(sprite).shape
    cx, cy = sprite.x + w // 2, sprite.y + h // 2
    return (_PX - cx) ** 2 + (_PY - cy) ** 2


def _cheb_dist_to(mask: np.ndarray) -> np.ndarray:
    """Chebyshev distance from every cell to the nearest True cell of `mask`."""
    INF = 10**6
    dist = np.full((64, 64), INF, np.int32)
    frontier = list(zip(*np.where(mask)))
    for y, x in frontier:
        dist[y, x] = 0
    d = 0
    while frontier:
        nxt = []
        for y, x in frontier:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < 64 and 0 <= nx < 64 and dist[ny, nx] > d + 1:
                        dist[ny, nx] = d + 1
                        nxt.append((ny, nx))
        frontier = nxt
        d += 1
    return dist


# ── level model ───────────────────────────────────────────────────────────────
class LevelModel:
    """Static, colour-independent geometry of one level, as 64x64 boolean masks.

    Every collection here is ordered deterministically (by name / position). The game
    derives ``brdck``/``fwqwj`` from a *set* of group names, so their key order varies
    with PYTHONHASHSEED across processes -- relying on it would make a cached plan
    depend on the process that solved it.
    """

    def __init__(self, game: R11l):
        lvl = game.current_level
        self.legs: list[tuple[int, int]] = []      # centres, sorted by position
        self.leg_group: list[str] = []
        self.legok: list[np.ndarray] = []          # leg centre is wall-free
        wall = _occ_of([s for s in lvl.get_sprites() if s.name.startswith("bvzgd-")])
        for leg in sorted(game.ftmaz, key=lambda s: (s.x, s.y)):
            self.legs.append((leg.x + 2, leg.y + 2))
            self.leg_group.append(game.mfrvmbaujm(leg))
            self.legok.append(~_hit_map(_solid(leg), wall))

        haz = _occ_of([s for s in lvl.get_sprites() if s.name.startswith("qtwnv")])
        self.groups: dict[str, dict] = {}
        for gname, data in sorted(game.brdck.items()):
            body, tgt = data["kignw"], data["xwdrv"]
            legs = [i for i, g in enumerate(self.leg_group) if g == gname]
            self.groups[gname] = {
                "legs": legs,
                "body": body.name if body else None,
                "shape": _solid(body) if body else None,
                "target": tgt.name if tgt else None,
                "hazok": (~_hit_map(_solid(body), haz)) if body else None,
            }

        # A body-shaped disc is identical for every bdkaz sprite; use any body's
        # shape for target/piece hit maps (paint never changes a body's silhouette).
        any_shape = next(
            (g["shape"] for g in self.groups.values() if g["shape"] is not None), None
        )
        self.tgt_hit: dict[str, np.ndarray] = {}
        self.tgt_cost: dict[str, np.ndarray] = {}
        for gname, data in sorted(game.brdck.items()):
            tgt = data["xwdrv"]
            if tgt is None:
                continue
            shape = self.groups[gname]["shape"]
            shape = any_shape if shape is None else shape
            self.tgt_hit[gname] = _hit_map(shape, _occ_of([tgt]))
            self.tgt_cost[gname] = _centre_cost(tgt)

        self.piece_hit: dict[str, np.ndarray] = {}
        self.piece_cost: dict[str, np.ndarray] = {}
        for p in sorted(game.nxahg, key=lambda s: s.name):
            self.piece_hit[p.name] = _hit_map(any_shape, _occ_of([p]))
            self.piece_cost[p.name] = _centre_cost(p)

        self.paint_bodies = sorted(game.fwqwj)

    def body_centre(self, centres: list[tuple[int, int]]) -> tuple[int, int]:
        n = len(centres)
        return (sum(c[0] for c in centres) // n, sum(c[1] for c in centres) // n)


def _bbox_free(all_centres: list[tuple[int, int]]) -> np.ndarray:
    """bool[64, 64]: clicking this cell is a MOVE, not a leg selection."""
    free = np.ones((64, 64), bool)
    for cx, cy in all_centres:
        free[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3] = False
    return free


# ── router ────────────────────────────────────────────────────────────────────
def route(model: LevelModel, gname: str, centres: list[tuple[int, int]],
          other_legs: list[tuple[int, int]], goal: np.ndarray, rest: np.ndarray,
          cost: np.ndarray, max_moves: int = 5, beam: int = 48,
          max_expand: int = 4000):
    """A* over this group's leg-centre tuple.

    `goal`/`rest`/`cost` are 64x64 maps over the BODY CENTRE: `rest` is where the body
    may come to rest at all, `goal` is where we want it, and `cost` ranks the cells of
    `goal` (lower = better centred). Returns (moves, final_centres) with
    moves = [(local_leg_idx, gx, gy)], or (None, None).

    The A* minimises MOVES; `cost` only ranks landings that are already tied on move
    count, so a tidier landing is never bought with an extra click. Ranking *every*
    same-length landing instead of only those of the first state to reach one is not
    worth it: it finds nothing tidier on any level, and a tidier intermediate rest can
    leave the legs somewhere that costs the NEXT routing step a move.
    """
    gl = model.groups[gname]["legs"]   # local leg index -> index into model.legs
    n = len(centres)
    if n == 0:
        return None, None
    start = tuple(centres)
    bx, by = model.body_centre(list(start))
    if goal[by, bx]:
        return [], list(start)

    hmap = _cheb_dist_to(goal)
    reach = max(1, 63 // n)  # most a single move can shift the body centre

    def h_of(state):
        bx, by = model.body_centre(list(state))
        return hmap[by, bx] / reach

    seen = {start: 0}
    pq = [(h_of(start), 0, start, [])]
    expand = 0
    while pq and expand < max_expand:
        _, g, state, path = heapq.heappop(pq)
        if g >= max_moves:
            continue
        expand += 1
        free = _bbox_free(list(state) + other_legs)
        Sx = sum(c[0] for c in state)
        Sy = sum(c[1] for c in state)
        best = None   # (cost, leg, gx, gy) of the tidiest landing at g + 1 moves
        for j in range(n):
            cx, cy = state[j]
            Dx = (Sx - cx + _PX) // n
            Dy = (Sy - cy + _PY) // n
            ok = model.legok[gl[j]] & free & rest[Dy, Dx]
            if not ok.any():
                continue
            hit = ok & goal[Dy, Dx]
            if hit.any():
                # Every hit cell finishes the job in the same single move, so take the
                # one that leaves the body most concentric with the goal sprite. Any
                # leg may be the one to move, so score them all before committing.
                c = np.where(hit, cost[Dy, Dx], 10**9)
                py, px = (int(v) for v in np.unravel_index(np.argmin(c), c.shape))
                if best is None or int(c[py, px]) < best[0]:
                    best = (int(c[py, px]), j, px, py)
                continue
            # Keep the best `beam` distinct resulting body centres.
            cand = np.argwhere(ok)
            keys = hmap[Dy[ok], Dx[ok]]
            order = np.argsort(keys, kind="stable")
            picked: set[tuple[int, int]] = set()
            for oi in order:
                py, px = int(cand[oi][0]), int(cand[oi][1])
                d = (int(Dx[py, px]), int(Dy[py, px]))
                if d in picked:
                    continue
                picked.add(d)
                ns = list(state)
                ns[j] = (px, py)
                ns = tuple(ns)
                ng = g + 1
                if seen.get(ns, 10**9) <= ng:
                    continue
                seen[ns] = ng
                heapq.heappush(pq, (ng + h_of(ns), ng, ns, path + [(j, px, py)]))
                if len(picked) >= beam:
                    break
        if best is not None:
            _, j, px, py = best
            ns = list(state)
            ns[j] = (px, py)
            return path + [(j, px, py)], ns
    return None, None


# ── plan assembly ─────────────────────────────────────────────────────────────
def _select_click(legs: list[tuple[int, int]], j: int) -> tuple[int, int] | None:
    """A click that selects leg `j`: inside its bbox and inside NO OTHER leg's bbox.

    The game scans its own ``ftmaz`` list and takes the FIRST bbox match, and that
    list's order is not reproducible across processes -- so require the cell to be
    unambiguous, which makes the click resolve to leg `j` under ANY scan order.
    """
    cx, cy = legs[j]
    cells = [(cx, cy)] + [(cx + dx, cy + dy)
                          for dy in range(-2, 3) for dx in range(-2, 3)]
    for x, y in cells:
        if not (0 <= x < 64 and 0 <= y < 64):
            continue
        if any(abs(x - legs[i][0]) <= 2 and abs(y - legs[i][1]) <= 2
               for i in range(len(legs)) if i != j):
            continue
        return x, y
    return None


class PlanBuilder:
    """Turns (leg index, destination) moves into a click list, tracking which leg is
    currently selected.

    The level starts with ``ftmaz[0]`` selected, but which leg that is depends on the
    game's own hash-seed-dependent list order, so assume NOTHING: `sel` starts as None
    so the first move of the plan always emits an explicit select click.
    """

    def __init__(self, model: LevelModel):
        self.legs = list(model.legs)
        self.sel: int | None = None
        self.clicks: list[tuple[int, int]] = []

    def move(self, j: int, gx: int, gy: int) -> bool:
        if self.sel != j:
            c = _select_click(self.legs, j)
            if c is None:
                return False
            self.clicks.append(c)
            self.sel = j
        self.clicks.append((gx, gy))
        self.legs[j] = (gx, gy)
        return True


# ── paint recipes ─────────────────────────────────────────────────────────────
def _colour_set(px: np.ndarray) -> frozenset:
    return frozenset(int(c) for c in np.unique(px) if c > 0)


def _stamped(base: np.ndarray, pieces: list[np.ndarray]) -> np.ndarray:
    """Apply `iciufrewti`'s index-aligned overwrite for an ordered piece list."""
    out = base.copy()
    for p in pieces:
        m = (p != -1) & (p != 0)
        out[m] = p[m]
    return out


def _find_recipes(game: R11l, target_cs: frozenset, max_len: int = 3):
    """Ordered piece-name tuples whose stamped colour set equals `target_cs`."""
    body_name = next(iter(game.fwqwj))
    blank = np.asarray(game.current_level.get_sprites_by_name(body_name)[0].pixels)
    blank = np.where(blank != -1, 0, -1)  # a pristine, unpainted body
    pieces = {p.name: np.asarray(p.pixels)
              for p in sorted(game.nxahg, key=lambda s: s.name)}
    out = []
    for k in range(1, max_len + 1):
        for combo in itertools.permutations(pieces, k):
            if _colour_set(_stamped(blank, [pieces[c] for c in combo])) == target_cs:
                out.append(combo)
    return out


# ── level solvers ─────────────────────────────────────────────────────────────
def _solve_simple(model: LevelModel) -> list[tuple[int, int]] | None:
    """Levels with a body per target: park each body on its own target."""
    pb = PlanBuilder(model)
    todo = [g for g, d in model.groups.items()
            if d["target"] is not None and d["body"] is not None and "anqcf" not in g]
    for gname in todo:
        gl = model.groups[gname]["legs"]
        hazok = model.groups[gname]["hazok"]
        goal = model.tgt_hit[gname] & hazok
        centres = [pb.legs[i] for i in gl]
        others = [c for i, c in enumerate(pb.legs) if i not in gl]
        moves, _ = route(model, gname, centres, others, goal, hazok,
                         model.tgt_cost[gname])
        if moves is None:
            return None
        for j, gx, gy in moves:
            if not pb.move(gl[j], gx, gy):
                return None
    return pb.clicks


def _solve_paint(model: LevelModel, game: R11l) -> list[tuple[int, int]] | None:
    """Levels 5-6: blank bodies collect pieces to match a bodyless target's colours."""
    targets = [g for g, d in model.groups.items()
               if d["target"] is not None and d["body"] is None and "anqcf" not in g]
    bodies = [g for g, d in model.groups.items()
              if d["body"] in model.paint_bodies]
    lvl = game.current_level
    recipes = {}
    for t in targets:
        tgt = lvl.get_sprites_by_name(model.groups[t]["target"])[0]
        recipes[t] = _find_recipes(game, _colour_set(np.asarray(tgt.pixels)))
        if not recipes[t]:
            return None

    # Score every (body -> target) assignment against every disjoint recipe pairing and
    # keep the shortest, breaking ties on how centred the final landings are. There are
    # only a few dozen candidates, and which one comes first is arbitrary -- it is not
    # even the shortest -- so it is worth pricing them all.
    best = None
    for perm in itertools.permutations(bodies, len(targets)):
        for combo in itertools.product(*(recipes[t] for t in targets)):
            used = [p for r in combo for p in r]
            if len(set(used)) != len(used):
                continue  # two bodies want the same piece
            got = _try_paint_plan(model, list(zip(perm, targets, combo)))
            if got is None:
                continue
            clicks, offcentre = got
            if best is None or (len(clicks), offcentre) < best[0]:
                best = ((len(clicks), offcentre), clicks)
    return best[1] if best else None


def _try_paint_plan(model: LevelModel, assign):
    """(clicks, offcentre) for one body/target/recipe assignment, or None if unroutable.
    `offcentre` sums each body's squared miss on its final target landing."""
    pb = PlanBuilder(model)
    remaining = set(model.piece_hit)
    offcentre = 0
    for bgroup, tgroup, recipe in assign:
        gl = model.groups[bgroup]["legs"]
        hazok = model.groups[bgroup]["hazok"]
        for step in list(recipe) + [None]:
            # Never rest on a piece we don't want: pickups are irreversible.
            forbid = remaining - ({step} if step else set())
            rest = hazok.copy()
            for p in forbid:
                rest &= ~model.piece_hit[p]
            goal = rest & (model.piece_hit[step] if step else model.tgt_hit[tgroup])
            cost = model.piece_cost[step] if step else model.tgt_cost[tgroup]
            centres = [pb.legs[i] for i in gl]
            others = [c for i, c in enumerate(pb.legs) if i not in gl]
            moves, fin = route(model, bgroup, centres, others, goal, rest, cost)
            if moves is None:
                return None
            for j, gx, gy in moves:
                if not pb.move(gl[j], gx, gy):
                    return None
            if step:
                remaining.discard(step)
            else:
                bx, by = model.body_centre(fin)
                offcentre += int(cost[by, bx])
    return pb.clicks, offcentre


# ── engine driving ────────────────────────────────────────────────────────────
def _render(game) -> list:
    return np.asarray(game.camera.render(game.current_level.get_sprites())).tolist()


def _inv_rot_click(gx: int, gy: int, k: int) -> tuple[int, int]:
    """The grid click, unconverted.

    Solvers plan and emit in the core game's UPRIGHT space; `BaseSolver` does the
    one conversion to screen space when it records. Converting here as well
    rotated twice and put the click on the wrong cell at k != 0."""
    return gx, gy



def _drive(game, dx: int, dy: int):
    """One ACTION6 click in display space. Returns (frame, solved, dead)."""
    game._full_reset = False
    game._set_action(ActionInput(id=GameAction.ACTION6, data={"x": dx, "y": dy}))
    last = None
    guard = 0
    while not game.is_action_complete():
        if game._next_level or guard > _STEP_GUARD:
            break
        guard += 1
        game.step()
        last = game.camera.render(game.current_level.get_sprites())
    if last is None:
        last = game.camera.render(game.current_level.get_sprites())
    solved = game._next_level or game._state == GameState.WIN
    dead = game._state == GameState.GAME_OVER
    return last, solved, dead


def _make_level(seed: int, level_idx: int, force_k0: bool = False) -> R11l:
    """A fresh game pinned on `level_idx`. The display rotation comes from the
    `AugmentedGame` base and is a pure function of (seed, level_index); the colour
    randomisation still runs off the GLOBAL `random` module, so determinism for it
    comes from seeding `random` immediately before each `set_level` -- the ls20/mm01
    no-seed-ctor pattern. `force_k0` pins the solver's instance to k=0 (plans are
    rotation-free grid clicks, so solving at k=0 keeps them seed-invariant)."""
    random.seed(f"{GAME_ID}:{seed}:{level_idx}")
    game = R11l(seed=seed)
    random.seed(f"{GAME_ID}:{seed}:{level_idx}")
    game.set_level(level_idx)
    if force_k0:
        game._rotation_k = 0
    return game


# ── plan cache ────────────────────────────────────────────────────────────────
def _solve_level(level_idx: int) -> list[tuple[int, int]] | None:
    """Solve `level_idx` ONCE in rotation-free grid space. Geometry is identical for
    every seed, and colour randomisation is a bijection (so colour-SET matching is
    seed-invariant), hence one plan serves all seeds."""
    game = _make_level(0, level_idx, force_k0=True)
    model = LevelModel(game)
    if game.fwqwj:
        plan = _solve_paint(model, game)
    else:
        plan = _solve_simple(model)
    if plan is None:
        return None
    if not _replay(_make_level(0, level_idx, force_k0=True), plan, 0):
        return None
    return plan


def _replay(game: R11l, plan, k: int) -> bool:
    """Replay grid-space clicks against the real engine. True iff the level is won."""
    for gx, gy in plan:
        dx, dy = _inv_rot_click(gx, gy, k)
        _, solved, dead = _drive(game, dx, dy)
        if dead:
            return False
        if solved:
            return True
    return False


def build_plans(levels, force: bool = False) -> dict[int, list]:
    _PLAN_DIR.mkdir(parents=True, exist_ok=True)
    plans: dict[int, list] = {}
    for i in levels:
        path = _PLAN_DIR / f"level{i}.json"
        if path.exists() and not force:
            plans[i] = [tuple(c) for c in json.loads(path.read_text())]
            continue
        print(f"  solving level {i} ...", flush=True)
        plan = _solve_level(i)
        if plan is None:
            print(f"  level {i}: NO PLAN")
            continue
        path.write_text(json.dumps([list(c) for c in plan]))
        print(f"  level {i}: {len(plan)} actions -> {path}")
        plans[i] = plan
    return plans


# ── Solver ──────────────────────────────────────────────────────────────────
class R11lSolver(BaseSolver):
    game_id = GAME_ID
    step_guard = _STEP_GUARD
    # Recovery is RESET-mode: the mask-A* is too slow to re-run per solve_from call
    # (a live re-plan is ~50s/episode), so ``solve_from`` replays a seed-invariant
    # cached grid-click plan valid only from the level's initial state. Exploration
    # explores the prefix, RESETs back to the initial state, then replays the plan.
    # (A cost-guarded replan -- cached-at-pristine + budget-capped live re-plan --
    # was prototyped but the paint levels 5-6 resist re-planning and it was fragile,
    # so RESET stays.)
    supports_recovery = True
    recovery_mode = "reset"

    #: {level_idx: grid-click plan}, solved (or loaded from cache) once and shared.
    _plans: dict[int, list] | None = None

    def _ensure_plans(self) -> dict[int, list]:
        if R11lSolver._plans is None:
            R11lSolver._plans = build_plans(list(range(_NUM_LEVELS)))
        return R11lSolver._plans

    def make_game(self, seed: int):
        # Colour randomisation runs off the GLOBAL ``random`` module; seed it right
        # before the constructor's implicit ``set_level(0)``. The rotation is a pure
        # function of (seed, level_index) via the AugmentedGame base.
        self._cur_seed = seed
        random.seed(f"{GAME_ID}:{seed}:0")
        return R11l(seed=seed)

    def num_levels(self, game) -> int:
        return _NUM_LEVELS

    def available_actions(self, game) -> list[int]:
        return [CLICK_ACTION]

    def set_level(self, game, level_idx: int) -> None:
        random.seed(f"{GAME_ID}:{self._cur_seed}:{level_idx}")
        game._next_level = False   # base drive never advances; clear the stale flag
        game.set_level(level_idx)

    def solve_from(self, game, level_idx: int, seed: int):
        """Replay the cached grid-click plan for ``level_idx`` (solved once by
        ``build_plans`` -- the mask geometry is seed-/rotation-invariant), mapping
        each grid cell to a centre-of-cell screen click at ``game``'s LIVE rotation.
        Valid only from the level's initial state, hence RESET-recovery."""
        plan = self._ensure_plans().get(level_idx)
        if not plan:
            return []
        k = game._rotation_k
        out = []
        for gx, gy in plan:
            dx, dy = _inv_rot_click(gx, gy, k)
            out.append(Action(CLICK_ACTION, (dy, dx)))   # click_rc = (row=y, col=x)
        return out


if __name__ == "__main__":
    raise SystemExit(R11lSolver.main())
