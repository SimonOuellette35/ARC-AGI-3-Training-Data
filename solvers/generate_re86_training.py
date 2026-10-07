"""Generate Phase-1 training data for the RE86 game (BaseSolver subclass).

An expert solver clears the levels of a seed and the trajectory is written as one
multi-level episode JSON in the encoder/dynamics schema, via ``BaseSolver``:

    {
      "game_id": "re86",
      "levels": [
        {"level_id": 0,
         "observations": [[[c,...],...],...],       # [T, 64, 64]
         "actions": [a0, a1, ...]},                  # [T]
        ...
      ]
    }

actions[0] is the RESET that produced obs[0] (game start for level 0, the level
transition for the rest); actions[i>=1] is the action that took the agent from
obs[i-1] to obs[i]. All actions are simple: {"type": "simple", "index": int}
(carrying the shared phase/changed/optimal fields from ``_encode_step``).

--------------------------------------------------------------------------------
The game (decoded from the obfuscated source)
--------------------------------------------------------------------------------
RE86 is a "reconstruct the coloured constellation" puzzle. A hidden 64x64
REFERENCE sprite (tag ``vzuwsebntu``) encodes target DOTS -- 3x3 blocks whose
border is ``_bg_dot_border`` (ignored) and whose centre pixel ``C`` is the
required colour. The level is WON (engine predicate ``cdjxpfqest``) when the
union of the movable SHAPES (tag ``vfaeucgcyr``), stamped onto a blank canvas,
covers every dot centre with a pixel of the matching colour ``C``.

The four direction actions move the currently-selected shape one 3px cell
(rotation-remapped by the level's ``_rotation_k``); ACTION5 cycles the
selection. A per-move StepCounter drains and hitting zero loses the level, so
the plan must be movement-efficient.

Shapes come in three geometries (by tag): a CROSS (full plus), an X (diagonals,
tag ``ldkpywfara``) and a BOX (rectangle outline, tag ``nogegkgqgd``). The
selected shape's centre pixel carries the selection marker colour, which would
corrupt the win check if it rests on a required dot -- so after placement we
cycle the selection until a non-conflicting shape rests selected.

Three sub-mechanics appear across the eight levels:
  * REPOSITION -- shape colours already match the goal, just slide them.
  * PAINT      -- a shape's colour is wrong; drive it into a colour-matched
                  STATION (tag ``ozhohpbjxz``) to repaint it (the first station
                  collided wins; collisions are pixel-perfect). Crossing a
                  station of the SAME colour is harmless.
  * RESHAPE    -- a shape can be pushed into an OBSTACLE (tag ``miqpqafylc``) to
                  deform. A BOX changes aspect ratio: a horizontal push does
                  (h+3, w-3), a vertical push does (h-3, w+3), so h+w is
                  invariant. A CROSS has its crossbar shifted (the crossing point
                  moves on the 3-lattice, the arms staying full-length rows /
                  columns). Both are handled by the same engine-driven search.

--------------------------------------------------------------------------------
The solver (engine-as-oracle, no hand-coded geometry beyond the above)
--------------------------------------------------------------------------------
Per level:
  1. CONFIG -- coordinate descent over (colour, deformed-geometry, position) per
     shape, using the goal-dot template as the objective, until zero dots are
     uncovered. With an obstacle present, boxes may take any h+w-preserving
     rectangle and crosses any crossbar-shifted variant; this fixes the colour
     assignment and confirms feasibility.
  2. EXECUTE -- for each shape: select it. Simple levels (no obstacle) paint it
     (BFS to a state colliding ONLY a matching station) and walk it to its target
     along a station/obstacle-avoiding BFS path. Deformable levels hand the shape
     to a best-first ENGINE SEARCH: it drives the four moves on a deep copy
     (snapshot/restore per node, no rendering) until the shape covers all of its
     colour's dots, discovering the reshape / crossbar-deform / repaint maneuvers
     automatically because the real engine computes every move's effect. Finally
     the selection is cycled to a resting shape that doesn't corrupt the win
     check.

Every action is applied through the real ``perform_action`` (ground truth) and the
settled board is recorded. The paint-spread flood-fill an action can trigger is drained
in-engine but NOT emitted frame-by-frame (see ``_LevelRunner.press``), so a repaint reads as
an INSTANT colour change and the recorded (obs, action) is the settled transition -- the bare
action list therefore no longer replays the animation tick-for-tick, by design.

--------------------------------------------------------------------------------
Episode model (PARTIAL clears, unlike the all-or-nothing base default)
--------------------------------------------------------------------------------
Levels the solver cannot clear (config infeasible or a plan step fails) end the
episode. Because a partial clear is still useful data, this generator OVERRIDES
``solve_episode``: it returns the levels it DID clear (``levels[:cleared]``) and
reports success when at least ``--min-levels`` levels are cleared. ``--require-win``
restricts output to full eight-level clears (GameState.WIN). Everything else --
the {"game_id", "levels": [...]} schema, ``_encode_step``/``_reset_step`` action
format, the WIN-filter write loop -- comes from ``BaseSolver``.

RE86 action set (logical, before rotation remap):
    ACTION1 up (y-3)  ACTION2 down (y+3)  ACTION3 left (x-3)  ACTION4 right (x+3)
    ACTION5 cycle selected shape

Usage (run from the repo root, with an env that has arcengine installed):
    python solvers/generate_re86_training.py --episodes 500 \
        --out data/training_multi_level/re86
"""

from __future__ import annotations

import argparse
import copy
import heapq
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

# Repo root (parent of solvers/) so the engine + game imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import games.re86.re86 as re86mod
from arcengine import ActionInput, GameAction, GameState
from games.re86.re86 import Re86, wfftxovxaa
from utils.rotation import ACTION_REMAP_INVERSE
from utils.explore import (
    Action, EpsilonSchedule, ExplorationPolicy, RESET_ACTION)
from solvers.base_solver import BaseSolver

GAME_ID = "re86"
STEP = re86mod.ilmaurgzng  # 3

A1, A2, A3, A4, A5 = (
    GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
    GameAction.ACTION4, GameAction.ACTION5,
)

# Exploration policy action id (1..5) -> the raw GameAction to send. Exploration
# has no logical intent, so ids map straight to keys (no rotation remap) and the
# recorded index equals the key actually pressed -- self-consistent, matching the
# native ``BaseSolver.drive`` which sends the exploration id directly too.
_EXPLORE_ACTIONS = {1: A1, 2: A2, 3: A3, 4: A4, 5: A5}

# Sprite tags (decoded).
TAG_SHAPE = "vfaeucgcyr"      # movable shape
TAG_STATION = "ozhohpbjxz"    # paint station
TAG_OBSTACLE = "miqpqafylc"   # obstacle (reshapes shapes on collision)
TAG_REF = "vzuwsebntu"        # goal reference constellation
TAG_BOX = "nogegkgqgd"        # box (rectangle-outline) shape
TAG_X = "ldkpywfara"          # X (diagonal) shape


def _raw(logical: GameAction, k: int) -> GameAction:
    """Raw action to send so the level's rotation remap (k) yields the logical move."""
    k %= 4
    return logical if k == 0 else ACTION_REMAP_INVERSE[k][logical]


def _fill_color(s) -> int | None:
    """Robust shape colour = mode of its non-empty pixels (dominates the single
    selection-marker/stray pixel)."""
    px = s.pixels[s.pixels != -1]
    if px.size == 0:
        return None
    vals, counts = np.unique(px, return_counts=True)
    return int(vals[np.argmax(counts)])


def _rel_cells(s):
    """Occupied cells of a sprite as (dx, dy) offsets from its top-left."""
    return [(int(dx), int(dy)) for dy, dx in np.argwhere(s.pixels != -1)]


def _abs_cells(sp):
    """Occupied board cells of a placed sprite as a set of (x, y)."""
    return set((int(sp.x + dx), int(sp.y + dy)) for dy, dx in np.argwhere(sp.pixels != -1))


def _goal_groups(g):
    """Required dots as {colour: set((y, x))} (border pixels excluded)."""
    b = g._bg_dot_border
    d = defaultdict(set)
    for r in g.current_level.get_sprites_by_tag(TAG_REF):
        ys, xs = np.where((r.pixels != -1) & (r.pixels != b))
        for y, x in zip(ys, xs):
            d[int(r.pixels[y, x])].add((int(y), int(x)))
    return d


# ── Reshaped box variants (h + w invariant) ──────────────────────────────────

def _box_outline_cells(h, w):
    c = set()
    for x in range(w):
        c.add((x, 0)); c.add((x, h - 1))
    for y in range(h):
        c.add((0, y)); c.add((w - 1, y))
    return c


def _box_variants(h0, w0):
    """All rectangle-outline variants reachable by obstacle pushes: h+w constant,
    each dimension >= 4 and on the same 3-lattice residue as the start."""
    s = h0 + w0
    out = []
    h = 4
    while h <= s - 4:
        w = s - h
        if h % 3 == h0 % 3 and w % 3 == w0 % 3 and w >= 4:
            out.append((h, w))
        h += 3
    return out


def _cross_bars(h, w):
    """Current (row, col) of a cross's horizontal and vertical bar (the full
    row / full column of its pixel grid)."""
    return h // 2, w // 2


def _cross_variant_cells(h, w, rH, cV):
    """Cells (dx, dy) of a cross whose horizontal bar is at row rH and vertical
    bar at column cV (arms span the whole hxw grid)."""
    c = set()
    for x in range(w):
        c.add((x, rH))
    for y in range(h):
        c.add((cV, y))
    return c


def _shape_variants(s, can_reshape):
    """(cells, h, w) geometry options. With an obstacle to push against, boxes
    expand to every h+w-preserving rectangle and crosses expand to every
    crossbar-shifted variant (the deform moves the crossing point on the
    3-lattice); otherwise every shape keeps its fixed geometry."""
    if can_reshape and TAG_BOX in s.tags:
        return [(_box_outline_cells(h, w), h, w) for (h, w) in _box_variants(s.height, s.width)]
    if can_reshape and TAG_X not in s.tags and TAG_BOX not in s.tags:
        h, w = s.height, s.width
        rH0, cV0 = _cross_bars(h, w)
        out = []
        for rH in range(rH0 % STEP, h, STEP):
            for cV in range(cV0 % STEP, w, STEP):
                out.append((_cross_variant_cells(h, w, rH, cV), h, w))
        return out
    cl = set((int(dx), int(dy)) for dy, dx in np.argwhere(s.pixels != -1))
    return [(cl, s.height, s.width)]


# ── Phase 1: config search (colour + geometry + position per shape) ───────────

def _solve_config(g, restarts=60, iters=60):
    """Coordinate descent for a winning configuration. Returns a list, one entry
    per movable shape, of (colour, frozenset(cells), x, y), or None. Restores the
    shapes' selection markers before returning (descent normalises them)."""
    mov = g.current_level.get_sprites_by_tag(TAG_SHAPE)
    snap = [s.pixels.copy() for s in mov]
    g.tgawbgxjra()  # normalise centres so stamped geometry matches the win check
    groups = _goal_groups(g)
    colors = list(groups)
    goal_cells = [(pos, c) for c, grp in groups.items() for pos in grp]

    # A shape can only END as its own colour or a colour it can PAINT to (a
    # station of that colour exists). Restricting the colour choices keeps the
    # descent from proposing impossible recolourings (e.g. no-station levels).
    station_colors = set(int(st.pixels[1, 1]) for st in g.current_level.get_sprites_by_tag(TAG_STATION))
    can_reshape = len(g.current_level.get_sprites_by_tag(TAG_OBSTACLE)) > 0

    # Candidate (colour, cells, x, y) per shape: every geometry/colour placed so
    # it covers at least one dot of that colour.
    opts = []
    for s in mov:
        o = []
        deformable = can_reshape and TAG_X not in s.tags  # box or cross
        rx = None if deformable else s.x % STEP
        ry = None if deformable else s.y % STEP
        allowed = (station_colors | {_fill_color(s)})
        shape_colors = [c for c in colors if c in allowed]
        for (cl, h, w) in _shape_variants(s, can_reshape):
            fcl = frozenset(cl)
            for c in shape_colors:
                ps = set()
                for (gy, gx) in groups[c]:
                    for (cx, cy) in cl:
                        x = gx - cx; y = gy - cy
                        if ((rx is None or x % STEP == rx) and (ry is None or y % STEP == ry)
                                and 0 <= x + w // 2 < 64 and 0 <= y + h // 2 < 64):
                            ps.add((x, y))
                for (x, y) in ps:
                    o.append((c, fcl, x, y))
        opts.append(o)
    if any(not o for o in opts):
        for s, px in zip(mov, snap):
            s.pixels = px
        return None

    def mism(cfg):
        canvas = {}
        for (c, cl, x, y) in cfg:
            for (dx, dy) in cl:
                canvas[(y + dy, x + dx)] = c
        return sum(1 for pos, c in goal_cells if canvas.get(pos) != c)

    rng = random.Random(0)
    result = None
    for r in range(restarts):
        cfg = [rng.choice(opts[i]) for i in range(len(mov))]
        prev = None
        for _ in range(iters):
            for i in range(len(mov)):
                best = (mism(cfg), cfg[i])
                for opt in opts[i]:
                    cfg[i] = opt
                    u = mism(cfg)
                    if u < best[0]:
                        best = (u, opt)
                cfg[i] = best[1]
            u = mism(cfg)
            if u == 0:
                result = cfg
                break
            if u == prev:
                break
            prev = u
        if result:
            break

    for s, px in zip(mov, snap):
        s.pixels = px
    return result


# ── Phase 2: execution driver ────────────────────────────────────────────────

class _LevelRunner:
    """Drives one level on the real engine, recording (frame, action) pairs.

    Actions are emitted through ``BaseSolver._encode_step`` so the on-disk step
    format (``type``/``index``/``phase``/``changed``/``optimal``) matches every
    other generator. This is an expert replay rather than an oracle tie-set, so an
    ``phase="expert"`` step is supervised with the single action it took
    (``optimal=[taken]``) -- never ``None``, which would train nothing at all.
    ``changed`` compares consecutive recorded frames."""

    def __init__(self, game, level_index, observations, actions, schedule=None,
                 solver=None):
        # `_encode_step` is an INSTANCE method (it converts the upright plan to the
        # screen action it records, which needs the live rotation), so the runner
        # needs the solver it belongs to -- not `BaseSolver` unbound.
        self.solver = solver
        self.g = game
        self.li = level_index
        self.obs = observations
        self.acts = actions
        self.schedule = schedule           # episode-wide EpsilonSchedule (or None)
        self.k = game._rotation_k
        self.mov = game.current_level.get_sprites_by_tag(TAG_SHAPE)
        self.stations = game.current_level.get_sprites_by_tag(TAG_STATION)
        self.obstacles = game.current_level.get_sprites_by_tag(TAG_OBSTACLE)
        self.obs_cells = set()
        for o in self.obstacles:
            self.obs_cells |= _abs_cells(o)
        self.failed = False

    # -- primitives --
    def advanced(self):
        return self.g._state == GameState.WIN or self.g.level_index > self.li

    def lost(self):
        return self.g._state == GameState.GAME_OVER

    def press(self, a, phase="expert"):
        """One action -> ONE recorded (frame, action). Any paint-spread flood-fill the
        action triggers is drained SILENTLY in-engine: the flood A5 ticks advance the board to
        its settled state but are NOT emitted as observation frames, so a repaint reads as an
        INSTANT colour change -- the animation is not something the recorded observations (or the
        dynamics) need to see. The flood ticks don't drain the step counter, so gameplay is
        unchanged, and the engine is left in the settled state so the next action follows from it.

        ``phase`` labels the step for the loss mask: expert moves are ``"expert"``;
        an exploration-prefix move passes ``"explore"`` so it is excluded from the
        policy loss (its ``optimal`` target stays None). Each recorded step advances
        the episode-wide epsilon schedule so the exploration horizon is measured in
        TOTAL steps (front-loaded across the whole episode), exactly as the base
        ``record_level`` does.
        """
        if self.failed or self.advanced() or self.lost():
            return
        fd = self.g.perform_action(ActionInput(id=a))
        guard = 0
        while self.g.zrermyobpw is not None and guard < 120 and not self.advanced() and not self.lost():
            fd = self.g.perform_action(ActionInput(id=A5))     # drain the flood; emit NO frame
            guard += 1
        adv = self.advanced()
        frame = fd.frame[0] if adv else fd.frame[-1]           # the settled board
        changed = not np.array_equal(np.asarray(frame), np.asarray(self.obs[-1]))
        self.obs.append(frame)
        taken = Action(int(a.value))
        # An EXPERT step is supervised with its OWN taken action. v2 supervises
        # `optimal` ONLY, so the previous `optimal=None` left re86 with 100%
        # unlabelled expert steps -- in every context, contributing nothing to the
        # loss -- which is why re86 went 4/12 -> 0/12. Explore steps stay
        # unsupervised: they are deliberately random and cloning them teaches
        # flailing. See the `always-emit-optimal-targets` rule.
        optimal = [taken] if phase == "expert" else None
        self.acts.append(self.solver._encode_step(
            taken, optimal=optimal, phase=phase, changed=changed))
        if self.schedule is not None:
            self.schedule.advance()

    def mv(self, dx, dy):
        if dx:
            self.press(_raw(A4 if dx > 0 else A3, self.k))
        if dy:
            self.press(_raw(A2 if dy > 0 else A1, self.k))

    def explore_prefix(self, schedule, exploration):
        """Drive the episode-wide epsilon prefix as random key presses, recorded as
        ``phase="explore"`` steps. Stops when the schedule stops exploring or the
        level advances/dies (a random walk can drain the step counter -> GAME_OVER,
        which the caller then undoes with the engine RESET). Returns True if any
        exploratory action was taken (so the caller knows a RESET is needed)."""
        perturbed = False
        while schedule.explore():
            if self.advanced() or self.lost():
                break
            act = exploration.action(np.asarray(self.obs[-1]))
            a = _EXPLORE_ACTIONS.get(act.action_id)
            if a is None:
                schedule.advance()
                continue
            self.press(a, phase="explore")     # press() advances the schedule
            perturbed = True
        return perturbed

    def sel_index(self):
        for i, s in enumerate(self.mov):
            if wfftxovxaa(s):
                return i
        return None

    def select(self, i):
        guard = 0
        while self.sel_index() != i and guard < len(self.mov) + 2 and not self.advanced():
            self.press(A5)
            guard += 1

    # -- station-avoiding lattice BFS --
    def _bfs(self, s, forb, is_goal, max_expand=4000):
        rel = _rel_cells(s)
        w, h = s.width, s.height
        start = (s.x, s.y)

        def valid(x, y):
            if not (0 <= x + w // 2 < 64 and 0 <= y + h // 2 < 64):
                return False
            for dx, dy in rel:
                if (x + dx, y + dy) in forb:
                    return False
            return True

        if is_goal(*start):
            return []
        q = deque([start])
        prev = {start: None}
        expanded = 0
        while q and expanded < max_expand:
            cur = q.popleft()
            expanded += 1
            for dx, dy in ((STEP, 0), (-STEP, 0), (0, STEP), (0, -STEP)):
                nx, ny = cur[0] + dx, cur[1] + dy
                if (nx, ny) in prev:
                    continue
                if not (0 <= nx + w // 2 < 64 and 0 <= ny + h // 2 < 64):
                    continue
                if is_goal(nx, ny):
                    prev[(nx, ny)] = cur
                    path = []
                    nd = (nx, ny)
                    while prev[nd] is not None:
                        p = prev[nd]
                        path.append((nd[0] - p[0], nd[1] - p[1]))
                        nd = p
                    return path[::-1]
                if valid(nx, ny):
                    prev[(nx, ny)] = cur
                    q.append((nx, ny))
        return None

    def _other_station_cells(self, tc):
        f = set()
        for st in self.stations:
            if int(st.pixels[1, 1]) != tc:
                f |= _abs_cells(st)
        return f

    def _walk(self, path):
        for dx, dy in path:
            self.mv(int(np.sign(dx)) if dx else 0, int(np.sign(dy)) if dy else 0)
            if self.failed or self.advanced() or self.lost():
                return

    # -- paint a shape to colour tc via a matching station --
    def paint(self, s, tc):
        sts = [st for st in self.stations if int(st.pixels[1, 1]) == tc]
        if not sts:
            return False
        forb = self._other_station_cells(tc) | self.obs_cells
        st_cells = set()
        for st in sts:
            st_cells |= _abs_cells(st)
        rel = _rel_cells(s)

        def is_goal(x, y):
            hit = False
            for dx, dy in rel:
                c = (x + dx, y + dy)
                if c in forb:          # would collide a wrong station first
                    return False
                if c in st_cells:
                    hit = True
            return hit

        for _ in range(2):
            if _fill_color(s) == tc:
                return True
            path = self._bfs(s, forb, is_goal)
            if path is None:
                return False
            self._walk(path)
        return _fill_color(s) == tc

    @staticmethod
    def _sim(gg, action):
        """Apply one action to a planning game WITHOUT rendering frames (the
        render/tolist is the dominant per-node cost). Mirrors the engine's
        perform_action loop (step until the action completes)."""
        if gg._state == GameState.WIN or gg._state == GameState.GAME_OVER:
            return
        gg._full_reset = False
        gg._set_action(ActionInput(id=action))
        guard = 0
        while not gg.is_action_complete() and guard < 300:
            if gg._next_level:
                gg._really_set_next_level()
            else:
                gg.step()
            guard += 1

    @staticmethod
    def _bars(px):
        """(row, col) of a cross's horizontal and vertical bar, or (-1, -1)."""
        r = np.where((px != -1).all(axis=1))[0]
        c = np.where((px != -1).all(axis=0))[0]
        return (int(r[0]) if len(r) else -1, int(c[0]) if len(c) else -1)

    # -- cover a colour's dots via engine-simulated best-first search --
    def search_cover(self, shape_idx, dots, tc, target, max_expand=9000):
        """Best-first over the four moves to drive the (already-selected,
        already-painted) shape until it covers every dot of colour ``tc``.
        Handles reshape (box) and crossbar-deform (cross) uniformly because the
        real engine computes each move's effect. Planning runs on a single deep
        copy with cheap snapshot/restore of the shape state per node and no frame
        rendering. The heuristic pulls the shape toward the winning geometry that
        the config search already found (``target`` = (tx, ty, cells)), which is
        far stronger than a raw dot-distance. Moves that would repaint the shape
        the wrong colour are pruned. Returns a list of (dx, dy) unit steps, or
        None.

        NB: the selection-marker colour lives in a MODULE global
        (re86mod.gigkddryzx) that is overwritten when a simulated move advances a
        level (re-randomisation); it is saved and restored so planning never
        poisons the live game's selection detection."""
        dot_xy = [(int(x), int(y)) for (y, x) in dots]
        tx, ty, cells = target
        dys = [dy for _, dy in cells]
        dxs = [dx for dx, _ in cells]
        t_rH = max(set(dys), key=dys.count)   # target horizontal-bar row
        t_cV = max(set(dxs), key=dxs.count)   # target vertical-bar col
        is_box = TAG_BOX in self.mov[shape_idx].tags
        t_h = 1 + max(dys)
        moves = ((0, -1), (0, 1), (-1, 0), (1, 0))
        n_dots = len(dot_xy)
        # Cells of stations of the right colour -- when the shape drifts to the
        # wrong colour (having crossed a mismatched station), the search is pulled
        # back to a matching station to repaint. Crossing stations is allowed
        # (not pruned) because the LAST station touched sets the final colour.
        tc_station_cells = set()
        for st in self.stations:
            if int(st.pixels[1, 1]) == tc:
                tc_station_cells |= _abs_cells(st)

        # Simulated moves that advance a level re-randomise, consuming both the
        # module marker global AND the global ``random`` module (via
        # random_rotation_k). Both are saved and restored so planning cannot
        # perturb the live game's later real level transitions.
        saved_marker = re86mod.gigkddryzx
        saved_rng = random.getstate()
        pg = copy.deepcopy(self.g)
        s = pg.current_level.get_sprites_by_tag(TAG_SHAPE)[shape_idx]
        hud = pg.gimylktpny

        def snapshot():
            return (s.pixels.copy(), s.x, s.y, hud.current_steps)

        def restore(snap):
            px, x, y, steps = snap
            s.pixels = px.copy()
            s.set_position(x, y)
            hud.current_steps = steps
            pg.zrermyobpw = None
            pg.gwwpnjmvvo = None

        def heur():
            occ = _abs_cells(s)
            if _fill_color(s) != tc:
                # Wrong colour: strictly worse than any correctly-coloured state;
                # steer toward the nearest matching station to repaint.
                if tc_station_cells:
                    d = min(min(abs(ox - px2) + abs(oy - py2) for (px2, py2) in tc_station_cells)
                            for (ox, oy) in occ)
                else:
                    d = 999
                return n_dots + 1, d
            unc = sum(1 for p in dot_xy if p not in occ)
            if unc == 0:
                return 0, 0
            geom = abs(s.height - t_h) if is_box else sum(abs(b - t) for b, t in zip(self._bars(s.pixels), (t_rH, t_cV)))
            if geom > 0 and self.obs_cells:
                # The shape still needs RESHAPING (geometry != the winning target),
                # which only happens by colliding with the deform obstacle. Steer to
                # the obstacle FIRST, then to the target: pulling toward (tx, ty) --
                # where geometry can never change -- makes the search wander around
                # hunting the obstacle (the winding level-5 reshape path).
                d_obs = min(min(abs(ox - obx) + abs(oy - oby) for (obx, oby) in self.obs_cells)
                            for (ox, oy) in occ)
                return unc, d_obs + geom
            return unc, abs(s.x - tx) + abs(s.y - ty) + geom

        try:
            hu, hd = heur()
            if hu == 0:
                return []
            counter = 0
            start = snapshot()
            # A*, not greedy best-first: the secondary key is g + hd (moves-so-far +
            # distance-to-target), so among states with the same #uncovered dots the search
            # prefers the one reached in FEWER moves -- this is what stops it taking winding,
            # non-minimal paths (the ~2x frame bloat). #uncovered stays primary so the
            # colour-repaint steering (heur returns n_dots+1 when the shape is the wrong
            # colour) is untouched. g = len(path).
            pq = [(hu, hd, counter, start, [])]
            seen = {(start[1], start[2], start[0].tobytes())}
            expanded = 0
            while pq and expanded < max_expand:
                _, _, _, snap, path = heapq.heappop(pq)
                expanded += 1
                for (dx, dy) in moves:
                    restore(snap)
                    # LOGICAL action, not the screen one: `_sim` drives via
                    # ``_set_action``, which bypasses the rotation wrapper, so the core
                    # game wants the un-rotated direction. (`press` goes through
                    # ``perform_action``, which DOES de-rotate, which is why the plan is
                    # converted with `_raw` there and not here. Feeding the screen action
                    # to `_sim` searched the wrong dynamics at k != 0 and lost levels
                    # 5-7.)
                    a = (A4 if dx > 0 else A3) if dx else (A2 if dy > 0 else A1)
                    self._sim(pg, a)
                    # Covering the last shape completes the level -> engine advances.
                    if pg._state == GameState.WIN or pg.level_index > self.li:
                        return path + [(dx, dy)]
                    g2 = 0
                    while pg.zrermyobpw is not None and g2 < 120:
                        self._sim(pg, A5)
                        g2 += 1
                    kk = (s.x, s.y, s.pixels.tobytes())
                    if kk in seen:
                        continue
                    seen.add(kk)
                    hu, hd = heur()
                    if hu == 0:
                        return path + [(dx, dy)]
                    counter += 1
                    heapq.heappush(pq, (hu, (len(path) + 1) + hd, counter, snapshot(),
                                        path + [(dx, dy)]))
            return None
        finally:
            re86mod.gigkddryzx = saved_marker
            random.setstate(saved_rng)

    # -- full per-shape plan --
    def run(self, cfg, groups):
        has_obstacle = len(self.obstacles) > 0
        # Solve shapes in SELECTION-RING order, starting from the one currently
        # selected: select, solve, cycle once to the next, solve, ... There is no
        # inter-shape blocking, so any other order only wastes ACTION5 cycles (and
        # reads as the solver arbitrarily skipping a shape and coming back to it).
        # A safe shape is rested selected by the explicit final cycling below, not
        # by the solve order.
        n = len(self.mov)
        start = self.sel_index() or 0
        order = [(start + k) % n for k in range(n)]
        for i in order:
            if self.failed or self.advanced() or self.lost():
                break
            s = self.mov[i]
            tc, cells, tx, ty = cfg[i]
            self.select(i)
            deformable = has_obstacle and TAG_X not in s.tags  # box or cross
            # Paint to the target colour first. For deformable shapes a failure is
            # not fatal -- the (colour-aware) cover search can still repaint by
            # crossing a matching station en route.
            if _fill_color(s) != tc:
                if not self.paint(s, tc) and not deformable:
                    self.failed = True
                    break
            if deformable:
                # Reshape/deform/repaint + position handled uniformly by search.
                path = self.search_cover(i, groups[tc], tc, (tx, ty, list(cells)))
                if path is None:
                    self.failed = True
                    break
                self._walk(path)
                if self.advanced() or self.lost():
                    break  # covering the last shape completed the level
                occ = _abs_cells(s)
                if any((int(x), int(y)) not in occ for (y, x) in groups[tc]):
                    self.failed = True
                    break
            else:
                forb = self._other_station_cells(tc) | self.obs_cells
                path = self._bfs(s, forb, lambda x, y, tx=tx, ty=ty: (x, y) == (tx, ty))
                if path is None:
                    self.failed = True
                    break
                self._walk(path)
                if (s.x, s.y) != (tx, ty) or _fill_color(s) != tc:
                    self.failed = True
                    break
        # Cycle selection to a non-conflicting resting shape (win check runs each
        # action) unless already advanced.
        if not self.advanced() and not self.lost() and not self.failed:
            for _ in range(len(self.mov) + 1):
                if self.advanced():
                    break
                self.press(A5)
        return self.advanced()


# ── Episode assembly ─────────────────────────────────────────────────────────

def _new_game(seed):
    random.seed(seed)
    g = Re86(seed=seed)
    g._rng = random.Random(seed)
    g.perform_action(ActionInput(id=GameAction.RESET))
    return g


class Re86Solver(BaseSolver):
    """RE86 generator. Reuses the base run()/main() output loop and action schema,
    but OVERRIDES ``solve_episode`` for the partial-clear episode model: an episode
    is written as long as it clears at least ``--min-levels`` levels (``levels``
    returned = the cleared prefix), and ``--require-win`` restricts to full clears.

    Paint-spread is IRREVERSIBLE within a level, but the engine RESET
    (``level_reset``: re-clone the level from ``_clean_levels`` and restore the step
    counter + NOT_FINISHED state) fully undoes any detour, so RESET-recovery still
    applies -- exactly like a human who flails, gets stuck/dies, hits reset, and only
    THEN solves. The custom record loop below replicates the base "reset" recovery:
    run the episode-wide epsilon prefix, RESET back to the level's initial state, then
    run the expert. Hence ``supports_recovery = True`` / ``recovery_mode = "reset"``."""
    #: Drives the engine through ``game.perform_action`` -- the AGENT-facing entry
    #: point, which de-rotates whatever it is handed. So this generator's plan is
    #: already in SCREEN space (it converts it itself), and the base must NOT
    #: convert again when recording. Generators that use the base's ``drive``
    #: (``_set_action``, which bypasses the wrapper) leave this False.
    plans_in_screen_space = True

    game_id = GAME_ID
    supports_recovery = True
    recovery_mode = "reset"

    def __init__(self, *, min_levels: int = 1, require_win: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._min_levels = min_levels
        self._require_win = require_win

    def make_game(self, seed: int):
        return _new_game(seed)

    def solve_episode(self, seed: int, explore: bool = True):
        """Play a seed from RESET, clearing as many levels as possible, and return
        ``(ok, cleared_levels)``. ``ok`` means ">= min_levels cleared" (and a full
        WIN when ``--require-win``); ``cleared_levels`` is ``levels[:cleared]`` --
        the failed/partial level that ended the run is dropped, matching the
        original generator.

        RESET-recovery (``recovery_mode="reset"``) is wired into this custom loop:
        the episode-wide ``EpsilonSchedule`` + ``ExplorationPolicy`` are built once
        (only when ``explore and supports_recovery``) and each level opens with the
        epsilon prefix (random keys, recorded ``phase="explore"``). If the prefix
        perturbed the board -- or drained the step counter to GAME_OVER -- a single
        engine RESET restores the level's initial state (``level_reset``), recorded
        as a ``phase="reset"`` action, before the expert (config search + placement)
        runs from that clean state. Because the expert re-solves the restored board,
        the cleared-level count under explore matches pure-optimal replay."""
        g = _new_game(seed)
        num_levels = len(g._levels)
        do_explore = explore and self.supports_recovery
        schedule = (EpsilonSchedule(self.rng, center=self._explore_center,
                                    jitter=self._explore_jitter)
                    if do_explore else None)
        exploration = (ExplorationPolicy(self.available_actions(g), self.rng)
                       if do_explore else None)
        levels_out: list[dict] = []
        cleared = 0

        for li in range(num_levels):
            initial_frame = g.camera.render(g.current_level.get_sprites()).tolist()
            observations = [initial_frame]
            actions = [self._reset_step()]

            # ── epsilon exploration prefix + RESET recovery ──────────────────
            if exploration is not None and schedule is not None:
                # Snapshot the EXACT observations[0] state. The engine RESET
                # (level_reset) would re-run on_set_level and redraw the colour
                # permutation + selection marker from the LIVE global-RNG state, so
                # its restored frame would NOT equal observations[0]; a deepcopy
                # transplant does, which is what the live engine RESET preserves.
                init_snap = copy.deepcopy(g)
                init_marker = re86mod.gigkddryzx
                ex = _LevelRunner(g, li, observations, actions, schedule=schedule, solver=self)
                perturbed = ex.explore_prefix(schedule, exploration)
                if ex.advanced():           # random walk happened to clear the level
                    levels_out.append({"level_id": li, "observations": observations,
                                       "actions": actions})
                    cleared += 1
                    if g._state == GameState.WIN:
                        break
                    continue
                if perturbed:               # undo the detour: restore initial state
                    prev = observations[-1]
                    g.__dict__.update(copy.deepcopy(init_snap).__dict__)
                    re86mod.gigkddryzx = init_marker
                    frame = g.camera.render(g.current_level.get_sprites())
                    observations.append(frame.tolist())
                    actions.append(self._encode_step(
                        Action(RESET_ACTION), optimal=[Action(RESET_ACTION)],
                        phase="reset",
                        changed=not np.array_equal(np.asarray(frame),
                                                   np.asarray(prev))))

            cfg = _solve_config(g)
            if cfg is None:
                levels_out.append({"level_id": li, "observations": observations,
                                   "actions": actions})
                break

            groups = _goal_groups(g)
            runner = _LevelRunner(g, li, observations, actions, schedule=schedule, solver=self)
            ok = runner.run(cfg, groups)
            levels_out.append({"level_id": li, "observations": observations,
                               "actions": actions})
            if not ok or runner.lost():
                break
            cleared += 1
            if g._state == GameState.WIN:
                break

        state = g._state
        good = cleared >= self._min_levels and cleared > 0
        if self._require_win and state != GameState.WIN:
            good = False
        return good, levels_out[:cleared]

    def available_actions(self, game) -> list[int]:
        """The five re86 keys: four moves + the selection cycle (ACTION5)."""
        return [1, 2, 3, 4, 5]

    # ── CLI: add re86's partial-clear knobs on top of the base parser ─────────
    @classmethod
    def build_argparser(cls) -> argparse.ArgumentParser:
        p = super().build_argparser()
        p.add_argument("--min-levels", type=int, default=1,
                       help="Write an episode only if it clears at least this "
                            "many levels (partial clears are kept).")
        p.add_argument("--require-win", action="store_true",
                       help="Only write full-clear (GameState.WIN) episodes.")
        return p

    @classmethod
    def main(cls, argv=None) -> int:
        args = cls.build_argparser().parse_args(argv)
        solver = cls(rng=random.Random(args.seed),
                     burst_prob=args.noise, burst_mean=args.burst,
                     min_levels=args.min_levels, require_win=args.require_win)
        out = args.out or (Path("data/training_multi_level") / cls.game_id)
        return solver.run(args.episodes, out, start_seed=args.start_seed,
                          max_attempts=args.max_attempts,
                          progress_every=args.progress_every,
                          explore=not args.no_exploration)


if __name__ == "__main__":
    sys.exit(Re86Solver.main())
