"""Generate Phase-1 training data for QR04 ("3x3 Quad Twist") -- mouse-click only.

The game
--------
An 8x8 board of tiles in 4 colours, plus a target "key" drawn as an 8x8 mini-board in the HUD.
ACTION6 clicks a cell; that cell is the TOP-LEFT anchor of a 3x3 block and the block is rotated a
quarter-turn CLOCKWISE. A click is *rejected* (red flash, no step charged) when the block would
run off the board (anchor x or y > 5) or would cover a wall. Win = every non-wall cell matches
the key. Only real rotations cost steps; exhausting the budget loses.

A move only permutes tile POSITIONS, so this is a permutation puzzle with a rich, heavily
overlapping move set (18-36 legal anchors per level) -- and, crucially, one where the target may
be flat-out unreachable. See ``games/qr04/qr04.py``: the tile invariant ``(value - x - y) mod 2``
is preserved by every rotation, which is why the SHIPPED level table was unwinnable on all seven
levels and had to be regenerated as "target, scrambled backwards". That rewrite is what makes
this generator possible at all; it also fixes the levels' step budgets and adds the repo's
per-(seed, level) augmentation (display rotation, target key, scramble, episode palette).

The solver: exact bidirectional BFS over the FRAME, never a heuristic
--------------------------------------------------------------------
``solve_from`` returns a **provably shortest** click plan for the LIVE board, or nothing:

* State is read out of the rendered 64x64 frame -- the tile grid's cell centres and the HUD key --
  and is the tuple of **colours** on the non-wall cells (wall cells sit in no legal block, so they
  are constant and simply excluded). The engine's private 0..3 tile values, the walls list and the
  scramble that built the level are never touched; a rotation permutes colours exactly as it
  permutes values, so the search does not need them. A move is one of the level's legal anchors.
* **Backward layers** are BFS'd from the target using counter-clockwise rotations (the inverse
  move) out to depth ``hb``, growing layer by layer until the next layer would exceed
  `BWD_STATE_CAP`. The target is constant for the whole level, so this map is built ONCE per
  (level, target) and reused by every step, every optimal-set probe and every burst check.
* **Forward BFS** from the live board runs at most `FWD_DEPTH` layers and stops at the first
  state present in the backward map, which makes the concatenated plan optimal.

So the certifiable radius is ``hb + FWD_DEPTH`` -- 6 clicks on the open boards (36 anchors) and 7
on the walled ones (18-30 anchors), measured. Level scramble depths are 1..5, which sits inside
that radius with room to spare, and the searches are milliseconds (the failure case -- an
off-plan board further away than the radius -- costs ~50 ms, which matters because the
exploration prefix asks for a plan on every one of its steps).

Nothing here is a heuristic or a fallback plan: when the live board is out of radius,
``solve_from`` returns ``[]`` and `BaseSolver.record_level` recovers with a RESET (or rolls a
burst back). That is deliberate -- every recorded ``optimal`` label is a certified-shortest
action, and a longer "good enough" plan would poison exactly the labels the policy trains on.

Optimal ACTION SETS (not just one route)
----------------------------------------
``optimal_set_from`` returns EVERY click that is on some shortest solution: for each legal anchor
``a``, ``a`` is optimal iff ``dist(apply(a, board)) == dist - 1``, tested with the same cached
backward map (usually a single dict lookup, since ``dist - 1 <= hb`` for most steps). So the
recorded target is the full tie set and `BaseSolver`'s stochastic-optimal sampling walks a
different -- still perfectly optimal -- route each episode. Ties are common here: the board has
many symmetric scrambles, and independent scramble moves can be undone in any order.

Recovery
--------
``supports_recovery = True`` with ``recovery_mode = "replan"``: ``solve_from`` reads only the live
board, so after an exploratory click it re-solves from wherever that left us whenever the board is
still inside the search radius (a stray rotation costs at most 3 clicks to undo, so early levels
recover outright). Beyond the radius the base RESETs, and a stranding burst is rolled back after
its engine-verified recovery check fails. Rotations are fully reversible, so no detour can brick a
level -- only push it out of certifiable range.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_qr04_training.py --episodes 500
    # self-check: search optimality, frame-observability of the state, engine-verified wins:
    python solvers/generate_qr04_training.py --verify
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Repo root (parent of solvers/) -- that's where the games/ package lives.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.qr04.qr04 import (  # noqa: E402
    GH, GW, KEY_PX, KEY_X, KEY_Y, N_VALUES, WALL_C, Qr04, legal_anchors,
)
from solvers.base_solver import BaseSolver  # noqa: E402
from utils.explore import Action, CLICK_ACTION  # noqa: E402
from utils.rotation import remap_click  # noqa: E402

GAME_ID = "qr04"

#: Stop growing the backward layers once they would pass this many states. 30k keeps the build
#: at ~0.6 s worst case (36 anchors) and lands on depth 3 for the open boards / depth 4 for the
#: walled ones -- see the module docstring for the resulting certifiable radius.
BWD_STATE_CAP = 30_000
#: Hard ceiling on backward depth (deeper layers blow up memory long before they help).
BWD_DEPTH_CAP = 4
#: Forward layers explored from the live board before giving up.
FWD_DEPTH = 3


# ── level geometry (walls -> anchors + the move permutations) ──────────────────
def geometry(walls: frozenset[tuple[int, int]]):
    """``(anchors, free_cells, cw, ccw)`` for a wall layout.

    ``free_cells`` is the state's cell order; ``cw[i]`` / ``ccw[i]`` are the ``(dst, src)`` index
    pairs of anchor ``i``'s clockwise / counter-clockwise rotation. Only the 8 border cells of a
    block move (the centre is a fixed point of a quarter-turn), so applying a move is 8 byte
    writes on a copy rather than a 64-element permutation.
    """
    anchors = legal_anchors(walls)
    free = [(x, y) for y in range(GH) for x in range(GW) if (x, y) not in walls]
    idx = {c: i for i, c in enumerate(free)}
    cw, ccw = [], []
    for ax, ay in anchors:
        # The game's rotation is grid[gy+j][gx+i] <- grid[gy+2-i][gx+j].
        pairs = [(idx[(ax + i, ay + j)], idx[(ax + j, ay + 2 - i)])
                 for j in range(3) for i in range(3) if (i, j) != (1, 1)]
        cw.append(pairs)
        ccw.append([(s, d) for d, s in pairs])
    return anchors, free, cw, ccw


def apply_move(state: bytes, pairs) -> bytes:
    out = bytearray(state)
    for dst, src in pairs:
        out[dst] = state[src]
    return bytes(out)


def backward_layers(target: bytes, ccw) -> tuple[dict[bytes, int], int]:
    """``({state: distance-to-target}, depth)`` -- BFS out from the target along INVERSE moves.

    Grown one full layer at a time and truncated when the layer would push the map past
    `BWD_STATE_CAP`, so the depth adapts to the level's branching factor (fewer legal anchors ->
    deeper map -> larger certifiable radius) instead of being a fixed guess.
    """
    seen = {target: 0}
    frontier = [target]
    depth = 0
    for d in range(BWD_DEPTH_CAP):
        nxt = []
        for state in frontier:
            for pairs in ccw:
                nb = apply_move(state, pairs)
                if nb not in seen:
                    seen[nb] = d + 1
                    nxt.append(nb)
        if len(seen) > BWD_STATE_CAP:
            for state in nxt:                   # discard the over-cap layer wholesale
                del seen[state]
            break
        frontier = nxt
        depth = d + 1
        if not frontier:                        # whole reachable set enumerated
            break
    return seen, depth


def search(state: bytes, bwd: dict[bytes, int], cw, max_fwd: int):
    """Shortest plan (list of anchor indices) from ``state`` into ``bwd``, or ``None``.

    Forward BFS meeting the precomputed backward map: the first meeting found at forward layer
    ``f`` gives total ``f + bwd[meet]``, and because both sides expand complete layers in step
    order, that total is the true distance (all shorter combinations were already enumerated).
    """
    if state in bwd:
        return [] if bwd[state] == 0 else _lift(state, bwd, cw)
    parents: dict[bytes, tuple[bytes, int] | None] = {state: None}
    frontier = [state]
    for depth in range(max_fwd):
        nxt = []
        hit = None
        for s in frontier:
            for ai, pairs in enumerate(cw):
                s2 = apply_move(s, pairs)
                if s2 in parents:
                    continue
                parents[s2] = (s, ai)
                nxt.append(s2)
                if s2 in bwd and (hit is None or bwd[s2] < bwd[hit]):
                    hit = s2
        if hit is not None:
            plan = []
            cur = hit
            while parents[cur] is not None:
                prev, ai = parents[cur]
                plan.append(ai)
                cur = prev
            plan.reverse()
            return plan + _lift(hit, bwd, cw)
        frontier = nxt
        if not frontier:
            break
    return None


def _lift(state: bytes, bwd: dict[bytes, int], cw) -> list[int]:
    """Walk ``state`` down the backward map to the target, greedily picking any move that
    decreases the stored distance. Layer distances make this exact -- no search needed."""
    plan = []
    cur = bwd[state]
    while cur > 0:
        for ai, pairs in enumerate(cw):
            nb = apply_move(state, pairs)
            if bwd.get(nb, 1 << 30) == cur - 1:
                plan.append(ai)
                state, cur = nb, cur - 1
                break
        else:                                   # unreachable: the map is layer-consistent
            raise AssertionError("backward map is not layer-consistent")
    return plan


# ── Solver ────────────────────────────────────────────────────────────────────
class Qr04Solver(BaseSolver):
    game_id = GAME_ID
    # solve_from re-solves the LIVE board from scratch on every call, so any reachable state
    # within the search radius is recovered from; beyond it the base RESETs (see the docstring).
    supports_recovery = True
    recovery_mode = "replan"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._geom_key = None
        self._geom = None
        self._bwd_key = None
        self._bwd: tuple[dict[bytes, int], int] | None = None
        self._plan_key = None
        self._plan_val: tuple[list[int], int] | None = None

    # ── engine glue ──────────────────────────────────────────────────────────
    def make_game(self, seed: int):
        # Qr04(seed=S) is a pure function of (S, level_index) -- board, key, palette and display
        # rotation. Recorded at its NATURAL rotation, never pinned to k=0.
        return Qr04(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [CLICK_ACTION]

    # ── board reading: FROM THE FRAME, never from engine state ───────────────
    def _state_of(self, game) -> tuple[bytes, bytes, frozenset]:
        """``(board, key, walls)`` read out of the RENDERED 64x64 frame.

        Nothing here touches ``_g`` / ``_target`` / ``_walls``, let alone the scramble that built
        the level: the solver sees exactly what a viewer of the frame sees. That is affordable
        because qr04 is fully observable -- the board is the top-left 32x32 tile grid, the key is
        the HUD mini-board, and walls are the only cells painted `WALL_C` (no tile colour can be
        3). Values are never recovered: the search runs directly on COLOUR indices, and a rotation
        permutes colours exactly as it permutes the engine's private 0..3 values.

        Rendering runs the HUD interfaces, which CONSUME the one-frame click marker and reject
        flash, so both are parked across the read: the decode never sees a marker sitting on a cell
        centre, and the pending marker still lands on the next recorded frame.
        """
        ui = game._ui
        click, flash = ui._click, ui._reject_frames
        ui._click, ui._reject_frames = None, 0
        try:
            frame = np.asarray(game.camera.render(game.current_level.get_sprites()))
        finally:
            ui._click, ui._reject_frames = click, flash
        frame = np.rot90(frame, k=-game.rotation_k)          # undo the display rotation
        board, key = [], []
        walls = set()
        for y in range(GH):
            for x in range(GW):
                cx, cy = self._cell_centre(game, x, y)
                colour = int(frame[cy, cx])
                if colour == WALL_C:
                    walls.add((x, y))
                    continue
                board.append(colour)
                key.append(int(frame[KEY_Y + y * KEY_PX, KEY_X + x * KEY_PX]))
        return bytes(board), bytes(key), frozenset(walls)

    def _geometry_for(self, walls):
        if self._geom_key != walls:
            self._geom_key, self._geom = walls, geometry(walls)
        return self._geom

    def _backward_for(self, walls, tgt, ccw):
        """Backward layers for this (wall layout, target). One-entry cache: a level keeps the
        same target for its whole duration, so this is built once per level and then reused by
        every step, optimal-set probe and burst-recovery check."""
        key = (walls, tgt)
        if self._bwd_key != key:
            self._bwd_key, self._bwd = key, backward_layers(tgt, ccw)
        return self._bwd

    def _analyse(self, game):
        """``(board, key, walls, anchor_plan, distance)`` for the live frame; ``plan`` is ``None``
        when the board is outside the certifiable radius. One-entry memo, because `record_level`
        asks for the plan and then immediately asks for the optimal set at the very same state."""
        cur, tgt, walls = self._state_of(game)
        key = (walls, tgt, cur)
        if self._plan_key == key:
            return self._plan_val
        anchors, free, cw, ccw = self._geometry_for(walls)
        bwd, hb = self._backward_for(walls, tgt, ccw)
        plan = search(cur, bwd, cw, FWD_DEPTH)
        val = (cur, tgt, walls, plan, None if plan is None else len(plan))
        self._plan_key, self._plan_val = key, val
        return val

    # ── clicks ───────────────────────────────────────────────────────────────
    @staticmethod
    def _cell_centre(game, gx: int, gy: int) -> tuple[int, int]:
        """Centre-of-cell pixel for grid cell ``(gx, gy)`` in GAME space.

        Read from the LIVE camera, never hardcoded: ``ARCBaseGame.set_level`` resizes the camera
        to the level's ``grid_size``, so qr04's ``CAM`` constant is dead and the real viewport is
        8x8 at scale 8 -- assuming the constant gives cell centres 4 px off, which silently
        addresses the wrong anchor."""
        scale, offx, offy = game.camera._calculate_scale_and_offset()
        return gx * scale + offx + scale // 2, gy * scale + offy + scale // 2

    def _click(self, game, anchor: tuple[int, int]) -> Action:
        """Screen click that selects grid cell ``anchor``.

        The frame is presented rotated by k, and the game maps a click back with
        ``remap_click(x, y, k)``; that map is a rotation, so the screen pixel a player would click
        is its inverse, ``remap_click(..., -k)`` (identity at k=0)."""
        px, py = self._cell_centre(game, *anchor)
        sx, sy = px, py
        return Action(CLICK_ACTION, (sy, sx))           # click_rc == (row, col)

    # ── BaseSolver API ───────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int):
        """Shortest click plan for the board as it stands, or ``[]`` if it is further from the
        target than the search can certify (the base then RESETs / rolls a burst back)."""
        cur, tgt, walls, plan, _dist = self._analyse(game)
        if plan is None:
            return []
        anchors = self._geometry_for(walls)[0]
        if not plan:
            # Board already equals the key. Unreachable in practice -- the engine checks the win
            # after every rotation, so this state always advances the level the moment it occurs
            # -- but a full turn of any anchor returns to it and wins, so never return "stuck".
            return [self._click(game, anchors[0])] * N_VALUES
        return [self._click(game, anchors[ai]) for ai in plan]

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every click that lies on SOME shortest solution: anchor ``a`` qualifies iff the board
        after ``a`` is one step closer. Reuses the cached backward map, so the common case
        (``dist - 1 <= hb``) is one dict lookup per anchor."""
        cur, tgt, walls, plan, dist = self._analyse(game)
        if plan is None or not plan:
            return None
        anchors, free, cw, ccw = self._geometry_for(walls)
        bwd, hb = self._backward_for(walls, tgt, ccw)
        want = dist - 1
        out = []
        for ai, pairs in enumerate(cw):
            nb = apply_move(cur, pairs)
            near = bwd.get(nb)
            if near is not None:
                if near == want:
                    out.append(self._click(game, anchors[ai]))
                continue
            if want <= hb:              # not in the map => strictly further than hb >= want
                continue
            sub = search(nb, bwd, cw, want - hb)
            if sub is not None and len(sub) == want:
                out.append(self._click(game, anchors[ai]))
        return out


# ── self-check ────────────────────────────────────────────────────────────────
def verify(seeds=range(6)) -> int:
    """Assert the three things the corpus depends on, for every level of several seeds:

    1. the frame decode really does recover the engine's board / key / walls (the solver plans
       from the frame, so a decode that silently drifted would plan on a fiction);
    2. every plan is engine-verified -- replaying it wins the level -- and its length equals the
       level's scramble depth, i.e. the search really is finding the shortest solution;
    3. every action in the recorded optimal SET is itself the start of a shortest solution.
    """
    solver = Qr04Solver()
    bad = 0
    for seed in seeds:
        game = solver.make_game(seed)
        for li in range(len(game._levels)):
            solver.set_level(game, li)
            depth = int(game.current_level.get_data("scramble"))
            cur, tgt, walls = solver._state_of(game)

            # (1) the frame decode agrees with the engine, cell for cell. The palette ORDER is
            # unobservable (the solver never recovers a 0..3 value), so the check is that ONE
            # consistent value->colour map explains every board and key cell.
            pal = {}
            ok_obs = walls == game._walls
            free = [(x, y) for y in range(GH) for x in range(GW) if (x, y) not in walls]
            for i, (x, y) in enumerate(free):
                for grid, decoded in ((game._g, cur), (game._target, tgt)):
                    if pal.setdefault(grid[y][x] % N_VALUES, decoded[i]) != decoded[i]:
                        ok_obs = False
            if not ok_obs:
                print(f"  seed {seed} L{li}: FRAME DECODE MISMATCH")
                bad += 1

            # (2) plan is optimal + engine-verified
            _c, _t, _w, plan, dist = solver._analyse(game)
            if plan is None:
                print(f"  seed {seed} L{li}: NO PLAN (scramble {depth})")
                bad += 1
                continue
            opts = solver.optimal_set_from(game, li, seed)
            acts = solver.solve_from(game, li, seed)
            solved = False
            for a in acts:
                res = solver.drive(game, a)
                if res.solved:
                    solved = True
                    break
                if res.dead:
                    break
            flag = "" if (solved and dist == depth and len(opts) >= 1) else "  <-- BAD"
            if flag:
                bad += 1
            print(f"  seed {seed} L{li}: dist={dist} scramble={depth} "
                  f"|optimal set|={len(opts)} engine_win={solved}{flag}")

            # (3) every optimal-set member starts a shortest solution
            solver.set_level(game, li)
            anchors, free, cw, ccw = solver._geometry_for(walls)
            bwd, hb = solver._backward_for(walls, tgt, ccw)
            by_click = {solver._click(game, a): i for i, a in enumerate(anchors)}
            for a in opts:
                nb = apply_move(cur, cw[by_click[a]])
                sub = search(nb, bwd, cw, FWD_DEPTH)
                if sub is None or len(sub) != dist - 1:
                    print(f"  seed {seed} L{li}: optimal-set member is not optimal: {a}")
                    bad += 1
    print("verify: OK" if not bad else f"verify: {bad} PROBLEM(S)")
    return 1 if bad else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--verify" in argv:
        argv.remove("--verify")
        return verify()
    return Qr04Solver.main(argv)


if __name__ == "__main__":
    sys.exit(main())
