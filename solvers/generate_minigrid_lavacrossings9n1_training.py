"""Generate Phase-1 training data for MiniGrid-LavaCrossingS9N1-v0.

Reach the green goal square. Pure navigation: turn and step, no objects. Lava
crossings with one gap each.

All of the solving is in ``common/minigrid.py`` -- one exact shortest-path
field over (cell, heading, carried object, cleared blockers), re-derived from
the live grid every step, which is what makes the epsilon exploration prefix
and the perturbation bursts recoverable (``supports_recovery = True``). One
maze per level, ``levels_per_episode`` mazes per episode.

Egocentric action set: ACTION1 forward, ACTION3 turn-left, ACTION4 turn-right.
Frame rotation is a per-level render augmentation and does not touch the
egocentric actions.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_minigrid_lavacrossings9n1_training.py --episodes 1000 \
        --out data/training_multi_level/minigrid_lavacrossings9n1
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.minigrid import MinigridSolver          # noqa: E402


class MinigridLavaCrossingS9N1Solver(MinigridSolver):
    env_id = "MiniGrid-LavaCrossingS9N1-v0"
    action_set = (1, 3, 4)


if __name__ == "__main__":
    sys.exit(MinigridLavaCrossingS9N1Solver.main())
