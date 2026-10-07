"""Generate Phase-1 training data for MiniGrid-Dynamic-Obstacles-8x8-v0.

Reach the green goal square while balls drift randomly around the room.
Walking into an obstacle -- or a wall -- ends the episode, and the collision
is judged against the layout as it stood when the action was chosen, so the
expert simply never steps into an occupied cell and re-plans against the moved
obstacles every single step. When the obstacles temporarily seal the route it
turns on the spot, which is a free wait (``stall_if_stuck``).

All of the solving is in ``common/minigrid.py`` -- one exact shortest-path
field over (cell, heading, carried object, cleared blockers), re-derived from
the live grid every step, which is what makes the epsilon exploration prefix
and the perturbation bursts recoverable (``supports_recovery = True``). One
maze per level, ``levels_per_episode`` mazes per episode.

Egocentric action set: ACTION1 forward, ACTION3 turn-left, ACTION4 turn-right.
Frame rotation is a per-level render augmentation and does not touch the
egocentric actions.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_minigrid_dynamic_obstacles_8x8_training.py --episodes 1000 \
        --out data/training_multi_level/minigrid_dynamic_obstacles_8x8
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.minigrid import MinigridSolver          # noqa: E402


class MinigridDynamicObstacles8x8Solver(MinigridSolver):
    env_id = "MiniGrid-Dynamic-Obstacles-8x8-v0"
    action_set = (1, 3, 4)
    stall_if_stuck = True


if __name__ == "__main__":
    sys.exit(MinigridDynamicObstacles8x8Solver.main())
