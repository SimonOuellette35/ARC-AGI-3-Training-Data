"""Generate Phase-1 training data for the CN04 game (games/cn04/cn04.py).

CN04 is a *mouse-click* geometric "connect the ports" puzzle. Each level has a
handful of sprites, every one carrying RED connection points (palette colour 8)
on its border. A sprite is selected by ACTION6-clicking it (layer raised, the
previously-selected one dropped), moved one cell at a time with ACTION1..4, and
rotated 90 deg clockwise with ACTION5. Two red points from DIFFERENT sprites that
land on the SAME grid cell "connect" (rendered as colour 3). The level is won
(``exlcvhdjsf``) once every red point of every sprite is connected -- i.e. the
sprites interlock so their red ports pair up exactly two-per-cell.

Randomisation (``_build_cn04_randomized_levels``) perturbs only each sprite's
BODY cells (never the red points) and its STARTING position; sprite shapes and
rotations are preserved. Because the red ports are fixed, the *solved relative
arrangement* is IDENTICAL across seeds -- only where the pieces begin differs.

Solver
------
1. Geometry search (``_search_arrangement``): fix the most-connected sprite as an
   anchor, then place the rest one at a time, each aligning one of its red ports
   onto an existing unpaired port, pruning any cell that would exceed two ports.
   Seed-independent -> cached.
2. Translate the arrangement into the grid, choosing the offset that MINIMISES
   total movement from the (per-seed) start positions.
3. ``solve_from`` drives the real engine ONE action at a time from the LIVE state:
   it finds the first sprite not yet at its target pose (preferring the currently
   active one), selects it if needed, rotates it to its target orientation (first
   nudging it to a rotation-safe cell so the bounding box never leaves the grid,
   since moves are bounds-clamped), then walks it to its target cell. The final
   sprite's last move completes every port and triggers ``next_level``. Because the
   phase (select / rotate / move) is recovered from each sprite's live pose vs its
   target, no plan state is carried across steps -- the base harness re-invokes
   ``solve_from`` after every action. Only an engine-verified solve is recorded.

Determinism / augmentation
--------------------------
``Cn04(seed=S)`` fixes the per-level bodies + start positions from ``S``. Episodes
are recorded at the NATURAL display rotation (``on_set_level`` -> per-level
``random_rotation_k``), never pinned to k=0. Rotation is a pure DISPLAY transform:
grid cells, sprite positions and the geometry search are untouched, so only the
*inputs* need converting. The game maps a screen click back with
``remap_click(x, y, k)`` and a screen direction with ``remap_action(id, k)``, so
``_grid_to_display`` / ``_screen_action`` apply the inverses and the recorded
action is the SCREEN-space one actually issued. Each level is recorded from a
fresh ``Cn04(seed)`` position, all-or-nothing.

Recovery is OFF: ``solve_from`` reads live state and self-corrects, but random
exploratory rotations near a boundary are not guaranteed reversible, so the
exploration prefix is left disabled.

Action schema (mixed simple + mouse, matching alchemy_stones / bp35 / cd82)
    RESET / simple :  {"type": "simple", "index": k}          # k in 0..5
    mouse click    :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_cn04_training.py --episodes 1000 \
        --out data/training_multi_level/cn04
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                          # noqa: E402
from arcengine import GameAction                            # noqa: E402
from games.cn04.cn04 import Cn04                            # noqa: E402
from solvers.base_solver import BaseSolver                  # noqa: E402
from utils.explore import Action, CLICK_ACTION              # noqa: E402
from utils.rotation import inverse_remap_action_full, remap_click  # noqa: E402

_ROTS = (0, 90, 180, 270)

# Simple directional / rotate actions.
_AMAP = {
    1: GameAction.ACTION1,   # up    (y-1)
    2: GameAction.ACTION2,   # down  (y+1)
    3: GameAction.ACTION3,   # left  (x-1)
    4: GameAction.ACTION4,   # right (x+1)
    5: GameAction.ACTION5,   # rotate 90 CW
}


# ── Geometry ──────────────────────────────────────────────────────────────────
def _rot_pixels(P: np.ndarray, rot: int) -> np.ndarray:
    """Sprite pixels rendered at ``rot`` degrees, matching Sprite.render()
    (``np.rot90(k=(-rot)//90)`` -- clockwise)."""
    k = int((-rot % 360) // 90)
    return np.rot90(P, k=k) if k else P


def _red_offsets(P: np.ndarray, rot: int):
    """(frozenset of (col,row) red-port offsets, (w, h)) for pixels ``P`` at ``rot``."""
    r = _rot_pixels(P, rot)
    ys, xs = np.where(r == 8)
    return frozenset(zip(xs.tolist(), ys.tolist())), (int(r.shape[1]), int(r.shape[0]))


def _search_arrangement(sprite_reds: list):
    """Find a winning relative arrangement.

    ``sprite_reds[i]`` maps rot -> (offsets, (w, h)). Returns {i: (rot, X, Y)} in
    a virtual (untranslated) coordinate frame, or None. Ports pair exactly two per
    occupied cell; each newly placed sprite must touch an existing unpaired port.
    Candidate order is fully sorted for byte-for-byte determinism."""
    n = len(sprite_reds)
    order = sorted(range(n), key=lambda i: -len(sprite_reds[i][0][0]))
    sol: dict[int, tuple[int, int, int]] = {}

    def place(depth: int, counts: dict) -> bool:
        if depth == n:
            return all(v == 2 for v in counts.values()) and len(counts) > 0
        placed = set(sol)
        for i in order:
            if i in placed:
                continue
            for r in _ROTS:
                offs, _wh = sprite_reds[i][r]
                if not placed:
                    cells = list(offs)
                    nc = dict(counts)
                    for (c, rw) in cells:
                        nc[(c, rw)] = nc.get((c, rw), 0) + 1
                    sol[i] = (r, 0, 0)
                    if place(depth + 1, nc):
                        return True
                    del sol[i]
                else:
                    unpaired = sorted(c for c, v in counts.items() if v == 1)
                    cand = sorted({
                        (ec[0] - c, ec[1] - rw)
                        for (c, rw) in offs for ec in unpaired
                    })
                    for (X, Y) in cand:
                        placed_cells = [(X + c, Y + rw) for (c, rw) in offs]
                        nc = dict(counts)
                        ok = touch = False
                        ok = True
                        for cc in placed_cells:
                            if counts.get(cc, 0) == 1:
                                touch = True
                            v = nc.get(cc, 0) + 1
                            if v > 2:
                                ok = False
                                break
                            nc[cc] = v
                        if not ok or not touch:
                            continue
                        sol[i] = (r, X, Y)
                        if place(depth + 1, nc):
                            return True
                        del sol[i]
        return False

    if place(0, {}):
        return dict(sol)
    return None


_ARRANGEMENT_CACHE: dict[int, list] = {}


def _level_arrangement(game, level_idx: int):
    """Per-sprite (rot, Xrel, Yrel) for the current level, cached by index
    (seed-independent). Returns a list aligned with get_sprites() order, or None."""
    if level_idx in _ARRANGEMENT_CACHE:
        return _ARRANGEMENT_CACHE[level_idx]
    sprites = game.current_level.get_sprites()
    sprite_reds = [
        {r: _red_offsets(game.npwwu[s.name], r) for r in _ROTS} for s in sprites
    ]
    sol = _search_arrangement(sprite_reds)
    result = None if sol is None else [sol[i] for i in range(len(sprites))]
    _ARRANGEMENT_CACHE[level_idx] = result
    return result


def _targets(game, arrangement):
    """Absolute (rot, x, y) targets: translate the arrangement into the grid,
    choosing the offset that minimises total movement from current positions."""
    sprites = game.current_level.get_sprites()
    gw, gh = game.current_level.grid_size

    dims = []  # (w, h) at each sprite's target rotation
    for i, s in enumerate(sprites):
        r = arrangement[i][0]
        dims.append(_red_offsets(game.npwwu[s.name], r)[1])

    def axis_offset(rel, dim, start, size):
        lo = -min(rel)
        hi = size - max(rel[i] + dim[i] for i in range(len(rel)))
        if lo > hi:
            return None
        deltas = sorted(start[i] - rel[i] for i in range(len(rel)))
        med = deltas[(len(deltas) - 1) // 2]
        return max(lo, min(hi, med))

    xr = [arrangement[i][1] for i in range(len(sprites))]
    yr = [arrangement[i][2] for i in range(len(sprites))]
    wd = [dims[i][0] for i in range(len(sprites))]
    hd = [dims[i][1] for i in range(len(sprites))]
    sx = [s.x for s in sprites]
    sy = [s.y for s in sprites]
    ox = axis_offset(xr, wd, sx, gw)
    oy = axis_offset(yr, hd, sy, gh)
    if ox is None or oy is None:
        return None
    return [(arrangement[i][0], xr[i] + ox, yr[i] + oy) for i in range(len(sprites))]


def _find_click_cell(game, sprite):
    """A grid cell where clicking selects ``sprite`` (topmost, pixel-perfect), or
    None if the sprite is fully occluded."""
    rendered = sprite.render()
    for row in range(rendered.shape[0]):
        for col in range(rendered.shape[1]):
            if rendered[row, col] == -1:
                continue
            gx, gy = sprite.x + col, sprite.y + row
            if game.current_level.get_sprite_at(gx, gy, ignore_collidable=True) is sprite:
                return gx, gy
    return None


class Cn04Solver(BaseSolver):
    game_id = "cn04"
    step_guard = 6000
    # ``solve_from`` reads the LIVE sprite poses and returns ONE step toward the
    # arrangement, so it re-plans correctly from an arbitrary perturbed state.
    supports_recovery = True
    recovery_mode = "replan"

    def make_game(self, seed: int):
        return Cn04(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4, 5, CLICK_ACTION]

    def _grid_to_display(self, game, gx: int, gy: int) -> Action:
        """Centre-of-cell SCREEN click Action for grid cell (gx, gy). The game
        maps a click back with ``remap_click(x, y, k)`` (a rotation), so its
        inverse is ``remap_click(..., -k)`` (identity at k=0)."""
        scale, offx, offy = game.camera._calculate_scale_and_offset()
        gpx = int(gx * scale + offx + scale // 2)
        gpy = int(gy * scale + offy + scale // 2)
        sx, sy = gpx, gpy
        return Action(CLICK_ACTION, (sy, sx))              # click_rc = (row=y, col=x)

    def _screen_action(self, game, idx: int) -> GameAction:
        """SCREEN action to press so the game's ``remap_action`` turns it back into
        ``_AMAP[idx]``. ACTION5 (rotate) is non-directional and passes through."""
        return _AMAP[idx]

    def _step_toward(self, game, sprite, tx, ty) -> GameAction:
        """One screen-space directional key that moves ``sprite`` toward (tx, ty);
        only called when it is not already there."""
        if sprite.x < tx:
            a = 4
        elif sprite.x > tx:
            a = 3
        elif sprite.y < ty:
            a = 2
        else:
            a = 1
        return self._screen_action(game, a)

    def _steps_toward(self, game, sprite, tx, ty) -> list:
        """EVERY co-optimal screen-space directional key that moves ``sprite`` one
        cell closer to (tx, ty). Movement in cn04 is pure grid translation with NO
        inter-sprite collision (``step`` only bounds-clamps the moving sprite's
        bounding box; see games/cn04/cn04.py), so reaching (tx, ty) is a free
        Manhattan walk: any interleaving of the still-misaligned axes costs the
        same ``|dx|+|dy|`` moves. Both endpoints are in-bounds (targets are placed
        to fit and the sprite got here by valid moves), and the walk is monotone
        per axis, so every intermediate cell stays in-bounds too -- so each returned
        direction is a valid, equally-optimal step. This is the maze open-field tie,
        and the ONLY diversity source here: select/rotate phases stay single-headed
        (occlusion / rotation-safety make those genuinely ordered)."""
        opts = []
        if sprite.x != tx:
            opts.append(self._screen_action(game, 4 if sprite.x < tx else 3))
        if sprite.y != ty:
            opts.append(self._screen_action(game, 2 if sprite.y < ty else 1))
        return opts

    def solve_from(self, game, level_idx: int, seed: int):
        """ONE action toward completing the arrangement, from the LIVE state.

        Finds the first sprite not at its target pose (preferring the currently
        selected one so its highlight is not wasted), then emits the next select /
        rotate / move step for it. The base re-invokes this after every action, so
        the select-rotate-walk phases reproduce the original online driver; the
        last sprite's final move triggers the win."""
        arrangement = _level_arrangement(game, level_idx)
        if arrangement is None:
            return []
        targets = _targets(game, arrangement)
        if targets is None:
            return []
        sprites = game.current_level.get_sprites()
        gw, gh = game.current_level.grid_size

        def at_target(i):
            s = sprites[i]
            tr, tx, ty = targets[i]
            return s.rotation == tr and (s.x, s.y) == (tx, ty)

        pending = [i for i in range(len(sprites)) if not at_target(i)]
        if not pending:
            return []   # every sprite placed but no verified win -> fail

        # Prefer the active sprite so its "free" selection is spent on a sprite
        # that still needs to move rather than being dropped and re-selected.
        active = next((i for i in pending if sprites[i] is game.weqid), None)
        i = active if active is not None else pending[0]
        s = sprites[i]
        target_rot, tx, ty = targets[i]

        # 1) select if not the active sprite.
        if game.weqid is not s:
            cell = _find_click_cell(game, s)
            if cell is None:
                return []
            return [self._grid_to_display(game, *cell)]

        # 2) rotate to target orientation, from a rotation-safe cell so the
        #    bounding box never leaves the grid (moves are bounds-clamped).
        if s.rotation != target_rot:
            M = int(max(game.npwwu[s.name].shape))
            safe = (min(tx, gw - M), min(ty, gh - M))
            if (s.x, s.y) != safe:
                return [self._step_toward(game, s, *safe)]
            return [GameAction.ACTION5]   # non-directional -> no screen remap

        # 3) walk to the target cell (the final sprite's last step wins).
        return [self._step_toward(game, s, tx, ty)]

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """The equally-optimal next actions at the LIVE state (STOCHASTIC OPTIMAL).

        Mirrors ``solve_from``'s phase selection EXACTLY -- same target sprite,
        same select/rotate/walk phase -- so the head it would return is always a
        member. The only phases with a genuine tie are the two MOVEMENT phases
        (nudge-to-rotation-safe-cell and walk-to-target): cn04 moves are collision-
        free grid translations, so getting a sprite from A to B is a free Manhattan
        walk and any misaligned-axis step is equally optimal (``_steps_toward``).
        Select and rotate are left single-headed: selection order is occlusion-
        sensitive (sprites overlap at their interlocked targets, so a wrong order
        can strand a piece) and the rotate itself is a unique action -- neither is a
        safe tie. Returns None only to defer to the base (never here; ``[]`` means
        no move, matching ``solve_from``)."""
        arrangement = _level_arrangement(game, level_idx)
        if arrangement is None:
            return []
        targets = _targets(game, arrangement)
        if targets is None:
            return []
        sprites = game.current_level.get_sprites()
        gw, gh = game.current_level.grid_size

        def at_target(i):
            s = sprites[i]
            tr, tx, ty = targets[i]
            return s.rotation == tr and (s.x, s.y) == (tx, ty)

        pending = [i for i in range(len(sprites)) if not at_target(i)]
        if not pending:
            return []
        active = next((i for i in pending if sprites[i] is game.weqid), None)
        i = active if active is not None else pending[0]
        s = sprites[i]
        target_rot, tx, ty = targets[i]

        # 1) select -- single canonical click (occlusion-sensitive, not a safe tie).
        if game.weqid is not s:
            cell = _find_click_cell(game, s)
            if cell is None:
                return []
            return [self._grid_to_display(game, *cell)]

        # 2) rotate: nudge to a rotation-safe cell (free Manhattan tie), then the
        #    unique ACTION5.
        if s.rotation != target_rot:
            M = int(max(game.npwwu[s.name].shape))
            safe = (min(tx, gw - M), min(ty, gh - M))
            if (s.x, s.y) != safe:
                return self._steps_toward(game, s, *safe)
            return [GameAction.ACTION5]

        # 3) walk to the target cell -- every misaligned-axis step is co-optimal.
        return self._steps_toward(game, s, tx, ty)


if __name__ == "__main__":
    sys.exit(Cn04Solver.main())
