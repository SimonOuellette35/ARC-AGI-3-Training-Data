"""Generate Phase-1 training data for MiniGrid-Unlock-v0.

Open the locked door -- that IS the win, there is no goal square. The planner
routes to the key, picks it up, and the terminal edge of its field is the
ACTION5 that turns the lock.

All of the solving is in ``common/minigrid.py`` -- one exact shortest-path
field over (cell, heading, carried object, cleared blockers), re-derived from
the live grid every step, which is what makes the epsilon exploration prefix
and the perturbation bursts recoverable (``supports_recovery = True``). One
maze per level, ``levels_per_episode`` mazes per episode.

Egocentric action set: ACTION1 forward, ACTION3 turn-left, ACTION4 turn-right,
ACTION5 interact. Frame rotation is a per-level render augmentation and does
not touch the egocentric actions.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_minigrid_unlock_training.py --episodes 1000 \
        --out data/training_multi_level/minigrid_unlock
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.minigrid import MinigridSolver          # noqa: E402


class MinigridUnlockSolver(MinigridSolver):
    env_id = "MiniGrid-Unlock-v0"
    action_set = (1, 3, 4, 5)
    objective = "open_door"


if __name__ == "__main__":
    sys.exit(MinigridUnlockSolver.main())
