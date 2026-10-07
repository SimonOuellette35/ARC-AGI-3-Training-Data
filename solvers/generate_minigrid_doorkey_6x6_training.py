"""Generate Phase-1 training data for MiniGrid-DoorKey-6x6-v0.

Pick up the key, unlock the door with it and reach the green goal. The shared
planner carries the key in its state vector and remembers that the door stays
unlocked, so it also handles a level where the key has to be fetched from the
far side of the room.

All of the solving is in ``common/minigrid.py`` -- one exact shortest-path
field over (cell, heading, carried object, cleared blockers), re-derived from
the live grid every step, which is what makes the epsilon exploration prefix
and the perturbation bursts recoverable (``supports_recovery = True``). One
maze per level, ``levels_per_episode`` mazes per episode.

Egocentric action set: ACTION1 forward, ACTION3 turn-left, ACTION4 turn-right,
ACTION5 interact. Frame rotation is a per-level render augmentation and does
not touch the egocentric actions.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_minigrid_doorkey_6x6_training.py --episodes 1000 \
        --out data/training_multi_level/minigrid_doorkey_6x6
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.minigrid import MinigridSolver          # noqa: E402


class MinigridDoorKey6x6Solver(MinigridSolver):
    env_id = "MiniGrid-DoorKey-6x6-v0"
    action_set = (1, 3, 4, 5)


if __name__ == "__main__":
    sys.exit(MinigridDoorKey6x6Solver.main())
