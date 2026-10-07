"""Generate Phase-1 training data for Gym-Gridworlds/Barrier-5x5-v0.

The goal is ringed by one-way arrow tiles, so it can only be entered from the
one side whose arrow does not point away. Modelling an arrow tile as a cell
with a single outgoing edge is what makes this fall out of the shared search
with no special case. The adapter re-rolls the barrier each level.

All of the solving is in ``common/gymgw.py`` -- one weighted shortest-path
field over the live tile grid (walls and pits impassable, quicksand ~10
actions to leave, penalty tiles weighted, arrow tiles reduced to a single
exit), re-derived every step, which is what makes the epsilon exploration
prefix and the perturbation bursts recoverable (``supports_recovery = True``).
One env reset per level, ``levels_per_episode`` levels per episode.

Absolute action set: ACTION1 up, ACTION2 down, ACTION3 left, ACTION4 right (+
ACTION5 stay where the env has it). The adapter rotates the frame per level
and inverse-remaps directional input, so the expert plans in grid space and
converts to the screen-space key to press.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_gymgw_barrier_5x5_training.py --episodes 1000 \
        --out data/training_multi_level/gymgridworlds_barrier_5x5
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.gymgw import GymgwSolver              # noqa: E402


class GymgwBarrier5x5Solver(GymgwSolver):
    env_id = "Gym-Gridworlds/Barrier-5x5-v0"


if __name__ == "__main__":
    sys.exit(GymgwBarrier5x5Solver.main())
