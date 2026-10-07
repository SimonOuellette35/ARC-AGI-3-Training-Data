"""Generate Phase-1 training data for the GP01 game (games/gp01/gp01.py).

GP01 ("Grid Paint") is a mouse-click pattern-matching puzzle on an 8x8 grid: an
ACTION6 click toggles a yellow paint mark on a cell, and a level is solved the
instant the painted set equals the gray hint set (``goal_cells``). Walls (color 3)
can't be painted; goal cells are never walls; ACTION1-4 are no-ops.

The harness (record/replay, schema, CLI, exploration prefix + bursts) lives in
``BaseSolver``. Because painting toggles, every click is reversible, so the
exploration prefix is safe and ``solve_from`` recovers from any painted state by
clicking the symmetric difference between the current paint and the goal.

Per-seed augmentation: ``Gp01(seed=S)`` builds 5 levels with random goal patterns
(and disjoint walls); no display rotation, so click coords need no inverse-rotation.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_gp01_training.py --episodes 1000 \
        --out data/training_multi_level/gp01
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.gp01.gp01 import Gp01                           # noqa: E402
from solvers.base_solver import BaseSolver                 # noqa: E402
from utils.explore import Action, CLICK_ACTION             # noqa: E402


class Gp01Solver(BaseSolver):
    game_id = "gp01"
    supports_recovery = True                               # clicks toggle -> reversible

    def make_game(self, seed: int):
        return Gp01(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [CLICK_ACTION]                              # only clicks do anything

    def _click(self, game, gx: int, gy: int) -> Action:
        """A click Action targeting the centre of grid cell ``(gx, gy)``, in the
        engine's display-pixel space (via the camera's live scale/offset)."""
        scale, offx, offy = game.camera._calculate_scale_and_offset()
        px = int(gx * scale + scale // 2 + offx)
        py = int(gy * scale + scale // 2 + offy)
        return Action(CLICK_ACTION, (py, px))              # click_rc = (row=y, col=x)

    def solve_from(self, game, level_idx: int, seed: int):
        """Click every cell where the current paint disagrees with the goal (the
        symmetric difference). From the blank start this paints each goal cell
        once; after an exploratory detour it also un-paints stray marks."""
        goal = set(game._goal)
        painted = set(game._painted())
        todo = sorted(goal.symmetric_difference(painted))
        return [self._click(game, gx, gy) for gx, gy in todo]

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """STOCHASTIC OPTIMAL: painting is an ORDER-INDEPENDENT toggle, so every
        cell in the symmetric difference between the current paint and the goal is
        an equally-optimal next click -- each click resolves exactly one
        disagreeing cell and reduces the remaining work by one, independent of
        order. Returning the whole set makes the base sample a fresh permutation
        of these clicks every episode (each a distinct, still-minimal route). This
        is the SAME live-state symmetric-difference `solve_from` computes, so when
        the base takes one click and re-plans, the set shrinks by exactly that
        cell -- length is always |symmetric difference| and never grows. Safe as a
        set only because gp01 is replan-mode and `solve_from` reads the LIVE
        painted state (a click never makes another symdiff click unnecessary)."""
        goal = set(game._goal)
        painted = set(game._painted())
        todo = sorted(goal.symmetric_difference(painted))
        return [self._click(game, gx, gy) for gx, gy in todo]


if __name__ == "__main__":
    sys.exit(Gp01Solver.main())
