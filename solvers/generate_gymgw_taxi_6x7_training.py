"""Generate Phase-1 training data for Gym-Gridworlds/Taxi-6x7-v0.

Taxi delivery, and the one env in the family whose state is more than a
position: three passengers must be picked up (by driving over them) before the
destination is worth anything. Arriving with none pays 0 and ends the episode
as a LOSS, and passengers cannot be dropped once picked, so the expert
searches over (cell, passengers-collected) and its terminal is 'on the goal
with all three aboard' -- worth 15 rather than 1 or 3.

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
    python solvers/generate_gymgw_taxi_6x7_training.py --episodes 1000 \
        --out data/training_multi_level/gymgridworlds_taxi_6x7
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.gymgw import GymgwSolver              # noqa: E402


class GymgwTaxi6x7Solver(GymgwSolver):
    env_id = "Gym-Gridworlds/Taxi-6x7-v0"
    collect_first = True


if __name__ == "__main__":
    sys.exit(GymgwTaxi6x7Solver.main())
