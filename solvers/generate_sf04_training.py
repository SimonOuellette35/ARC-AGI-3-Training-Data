"""Generate Phase-1 training data for the SF04 game (games/sf04/sf04.py).

SF04 ("Stencil Rotate Paint") is a *mouse-click* stencil puzzle on a 16x16 grid.
Each level shows a hint pattern (``goal``); ACTION5 rotates a 3-cell L stencil
(increment-only, 0->1->2->3->0) and an ACTION6 click stamps that stencil at the
clicked *anchor* cell. A stamp is rejected wholesale if any of its 3 cells is out
of bounds or a wall. The level is solved the instant ``goal <= painted``.

Solver
------
Paint is additive and the win test is a *superset* test, so overpainting is free
and the puzzle is a pure covering problem: pick a set of legal stamps whose union
contains every hint cell.

We solve that cover exactly, minimising the number of *stamps* (tie-broken by
rotations) -- deliberately NOT the number of steps. Every hint shape is a union of
L footprints plus the odd stray singleton, and an L footprint is by definition a
stencil footprint at some rotation, so a minimum-stamp cover lands exactly one
stamp on each L at that L's own orientation: rotate to match, stamp once, which is
the concept the game teaches. (Minimising steps degenerates into "never rotate,
spam the default stencil" -- see ``_plan_level``.)

Rotation is increment-only (0->1->2->3->0), so a plan using rotation set Q is
played by walking the stencil forward through the needed q's. ``solve_from``
returns a flat action stream: rotations interleaved with the anchor clicks.
Legality (``placements_covering``) is imported from the game module so it cannot
drift from ``Sf04.step``; plans are still REPLAYED against the real engine and only
an engine-verified level solve is recorded.

Recovery is ON via the REPLAN paradigm (``supports_recovery = True``,
``recovery_mode = "replan"``): paint is additive (a stamp can never be undone) and
the goal/walls are fixed, so NO perturbation can make a level unsolvable -- the
cover is always reachable. ``solve_from`` reads the live goal/walls AND the live
stencil rotation ``game._rotq`` each call, then replays the cover in cyclic order
from that current rotation (stamps sorted by ``(q - q0) % 4``, the increment-only
distance). Cells the detour already painted are harmless (overpaint is free), so
the same stream is valid from the clean start (q0=0, blank) and from any perturbed
board. Exploration bursts therefore recover in place without a RESET. The shared
``Sf04UI`` overlay carries mutable click-feedback state that ``on_set_level`` does
NOT clear, so the solver's ``set_level`` wipes it -- otherwise a leftover click
marker would make a re-seated level frame differ from ``observations[0]`` (see
``Sf04Solver.set_level``).

Determinism / augmentation
--------------------------
``Sf04(seed=S)`` bakes a deterministic per-seed board *and* a per-seed colour
permutation; there is no display rotation, so recorded click coordinates need no
inverse-rotation and each seed yields a distinct, fully deterministic episode.
One episode == one seed, all-or-nothing over the levels.

Action schema (mixed simple + mouse, matching gp01 / cd82 / cn04 / bp35)
    RESET / simple :  {"type": "simple", "index": k}                      # k in 0..5
    mouse click    :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_sf04_training.py --episodes 1000 \
        --out data/training_multi_level/sf04
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction                            # noqa: E402
from games.sf04.sf04 import (                               # noqa: E402
    Sf04, placements_covering, stencil_cells)
from solvers.base_solver import BaseSolver                  # noqa: E402
from utils.explore import Action, CLICK_ACTION              # noqa: E402

Cell = tuple[int, int]
Placement = tuple[Cell, int]  # (anchor, rotation q)


# ── Cover solver (minimise stamps, tie-broken by rotations) ───────────────────
def _candidates(goal: list[Cell], walls: set[Cell]) -> dict[Placement, int]:
    """Every legal stamp covering at least one hint cell, as a bitmask over
    ``goal``. Legality comes from the game module, so it cannot drift from
    ``Sf04.step``."""
    index = {cell: i for i, cell in enumerate(goal)}
    masks: dict[Placement, int] = {}
    for cell in goal:
        for anchor, q in placements_covering(cell, walls):
            if (anchor, q) in masks:
                continue
            mask = 0
            for c in stencil_cells(anchor[0], anchor[1], q):
                if c in index:
                    mask |= 1 << index[c]
            masks[(anchor, q)] = mask
    return masks


def _min_stamps(goal: list[Cell], masks: dict[Placement, int]) -> list[Placement] | None:
    """Fewest stamps from ``masks`` whose union covers all of ``goal``, or None.

    DP over covered-cell bitmasks. Candidates are visited in sorted order and ties
    are never overwritten, so the chosen cover is a deterministic function of the
    board."""
    full = (1 << len(goal)) - 1
    best: dict[int, tuple[int, Placement | None, int]] = {0: (0, None, 0)}
    ordered = sorted(masks.items())
    for covered in range(full + 1):
        if covered not in best:
            continue
        cost = best[covered][0]
        for placement, mask in ordered:
            nxt = covered | mask
            if nxt == covered:  # stamp adds nothing
                continue
            if nxt not in best or cost + 1 < best[nxt][0]:
                best[nxt] = (cost + 1, placement, covered)
    if full not in best:
        return None
    chosen: list[Placement] = []
    node = full
    while node:
        _, placement, prev = best[node]
        chosen.append(placement)
        node = prev
    return chosen


def _plan_level(goal: list[Cell], walls: set[Cell]) -> list[Placement] | None:
    """Fewest stamps, tie-broken by fewest rotations; ordered ascending-q for play.

    The objective is lexicographic ``(stamps, max q)`` -- NOT total steps. Pricing
    rotations against stamps produces bad demonstrations: an ACTION5 costs the same
    step as an ACTION6, so a step-minimising cover buys back each rotation by
    overpainting with extra q=0 stamps and degenerates into "never rotate". A
    minimum-stamp cover instead lands exactly one stamp per hint shape, at that
    shape's own orientation: rotate to match, stamp once. Paying for that in
    rotations is free (the budget is 40-100 against a worst case of ~5 steps).

    Rotation is increment-only from q=0, so a plan using rotations up to ``qmax``
    pays exactly ``qmax`` ACTION5s when played in ascending q; scanning ``qmax``
    upward and only accepting a strictly smaller stamp count yields the fewest
    stamps at the lowest rotation ceiling achieving them."""
    if not goal:
        return None
    all_masks = _candidates(goal, walls)
    best_plan: list[Placement] | None = None
    for qmax in range(4):
        masks = {p: m for p, m in all_masks.items() if p[1] <= qmax}
        stamps = _min_stamps(goal, masks)
        if stamps is None:
            continue
        if best_plan is None or len(stamps) < len(best_plan):
            best_plan = stamps
    if best_plan is None:
        return None
    # Ascending q is what makes `max(q)` rotations sufficient; (y, x) within a
    # rotation just fixes a stable reading order.
    return sorted(best_plan, key=lambda p: (p[1], p[0][1], p[0][0]))


class Sf04Solver(BaseSolver):
    game_id = "sf04"
    step_guard = 6000
    # REPLAN recovery: paint is additive and rotation is a one-way cycle, so no
    # perturbation can brick a level. ``solve_from`` reads the LIVE stencil rotation
    # (``game._rotq``) and re-covers the live goal/walls, emitting cyclic-ascending
    # rotations from wherever the detour left the stencil -- valid from any
    # perturbed-but-solvable board, so bursts recover without a RESET.
    supports_recovery = True
    recovery_mode = "replan"

    def make_game(self, seed: int):
        return Sf04(seed=seed)

    def set_level(self, game, level_idx: int) -> None:
        """Seat ``level_idx`` and wipe the ``Sf04UI`` click-feedback overlay.

        Sf04 has no display rotation and its board + colour LUT are a pure
        function of the seed, so a re-seated level would render byte-identically
        to its first capture -- EXCEPT that the ``Sf04UI`` instance persists across
        levels and holds mutable click-feedback markers (a coloured plus stamped at
        the last-clicked cell for 8 render frames). ``on_set_level`` never clears
        them, so after the exploration prefix a recovery RESET (base ``reset_level``
        -> ``set_level``) would redraw a leftover marker and the reset frame would
        differ from ``observations[0]`` by ~5-10 cells. Clearing the feedback here
        -- the single choke point for both the initial capture and every RESET --
        makes the RESET frame reproduce ``observations[0]`` exactly."""
        super().set_level(game, level_idx)
        ui = getattr(game, "_ui", None)
        if ui is not None:
            ui._click_pos = None
            ui._click_frames = 0
            ui._click_ok = False

    def available_actions(self, game) -> list[int]:
        return [int(GameAction.ACTION5.value), CLICK_ACTION]

    def _goal_cells(self, game) -> list[Cell]:
        raw = game.current_level.get_data("goal") or []
        return sorted(tuple(int(t) for t in p) for p in raw)

    def _walls(self, game) -> set[Cell]:
        return {(s.x, s.y) for s in game.current_level.get_sprites_by_tag("wall")}

    def _click(self, game, gx: int, gy: int) -> Action:
        """A click Action targeting the centre of grid cell ``(gx, gy)`` (whose
        ``display_to_grid`` maps back to ``(gx, gy)``), in the engine's
        display-pixel space via the camera's live scale/offset."""
        scale, offx, offy = game.camera._calculate_scale_and_offset()
        px = int(gx * scale + scale // 2 + offx)
        py = int(gy * scale + scale // 2 + offy)
        return Action(CLICK_ACTION, (py, px))              # click_rc = (row=y, col=x)

    def solve_from(self, game, level_idx: int, seed: int):
        """Increment-only rotations + anchor clicks covering every hint cell, valid
        from the LIVE state (any stencil rotation, any partial paint).

        Read the cover from the live goal/walls (both perturbation-invariant), then
        replay it from the LIVE stencil rotation ``game._rotq``. Rotation is a one-way
        cycle (0->1->2->3->0), so the stamps are visited in CYCLIC order starting at
        the current rotation -- sorted by ``(q - q0) % 4`` -- and each ACTION5 walks
        the stencil forward to the next stamp's q. Paint is additive and overpaint is
        free, so any cells the detour already painted are harmless and the cover never
        needs re-solving for them: this same flat stream solves from the clean start
        (q0=0, blank board) and from any perturbed-but-solvable board alike."""
        goal = self._goal_cells(game)
        if not goal:
            return []
        plan = _plan_level(goal, self._walls(game))
        if plan is None:
            return []
        q0 = int(getattr(game, "_rotq", 0)) % 4
        # Visit stamp rotations in increment-only cyclic order from the LIVE q0.
        ordered = sorted(plan, key=lambda p: ((p[1] - q0) % 4, p[0][1], p[0][0]))
        seq: list = []
        rotq = q0
        for (ax, ay), q in ordered:
            while rotq != q:                 # increment-only; cyclic-ascending from q0
                seq.append(GameAction.ACTION5)
                rotq = (rotq + 1) % 4
            seq.append(self._click(game, ax, ay))
        return seq


if __name__ == "__main__":
    sys.exit(Sf04Solver.main())
