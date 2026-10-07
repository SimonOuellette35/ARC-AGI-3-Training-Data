"""Generate Phase-1 training data for Gym-Gridworlds/Maze-12x12-v0.

A fresh recursive-backtracker maze every level: the adapter's
``_randomize_maze_12x12`` re-carves the walls and re-rolls the frame rotation on
each reset, so the expert has to read the whole board rather than memorise a
route. The maze is a spanning tree, so a path to the green goal always exists and
it is UNIQUE -- which makes the stochastic-optimal tie set a single action almost
everywhere here.

All of the solving is in ``common/gymgw.py`` -- one weighted shortest-path field
over the live tile grid (walls and pits impassable, quicksand ~10 actions to
leave, penalty tiles weighted, arrow tiles reduced to a single exit), re-derived
every step, which is what makes the epsilon exploration prefix and the
perturbation bursts recoverable (``supports_recovery = True``). One env reset per
level, ``levels_per_episode`` levels per episode.

Absolute action set: ACTION1 up, ACTION2 down, ACTION3 left, ACTION4 right
(+ ACTION5 stay where the env has it). The adapter rotates the frame per level and
inverse-remaps directional input, so the expert plans in grid space and converts
to the screen-space key to press.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_gymgw_maze12x12_training.py --episodes 1000 \
        --out data/training_multi_level/gymgridworlds_maze_12x12
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.gymgw import GymgwSolver              # noqa: E402


class GymgwMaze12x12Solver(GymgwSolver):
    env_id = "Gym-Gridworlds/Maze-12x12-v0"
    game_id = "gymgridworlds_maze_12x12"       # matches GymGridworldsAdapter.game_id


if __name__ == "__main__":
    sys.exit(GymgwMaze12x12Solver.main())
