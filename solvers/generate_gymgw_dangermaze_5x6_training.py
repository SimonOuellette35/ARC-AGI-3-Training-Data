"""Generate Phase-1 training data for Gym-Gridworlds/DangerMaze-5x6-v0.

A small maze of walls, penalty tiles and PITS -- stepping into a pit is
instant death, so the expert treats them as impassable rather than expensive.
Re-randomised by the adapter on every level reset.

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
    python solvers/generate_gymgw_dangermaze_5x6_training.py --episodes 1000 \
        --out data/training_multi_level/gymgridworlds_dangermaze_5x6
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.gymgw import GymgwSolver              # noqa: E402


class GymgwDangerMaze5x6Solver(GymgwSolver):
    env_id = "Gym-Gridworlds/DangerMaze-5x6-v0"


if __name__ == "__main__":
    sys.exit(GymgwDangerMaze5x6Solver.main())
