"""Generate Phase-1 training data for MiniGrid-UnlockPickup-v0.

Pick up the box in the next room, which is behind a locked door. The agent
carries one thing at a time, so the key has to be put down again (ACTION2)
before the box can be picked up -- the planner's carried-object state and drop
edges are what make that fall out automatically.

All of the solving is in ``common/minigrid.py`` -- one exact shortest-path
field over (cell, heading, carried object, cleared blockers), re-derived from
the live grid every step, which is what makes the epsilon exploration prefix
and the perturbation bursts recoverable (``supports_recovery = True``). One
maze per level, ``levels_per_episode`` mazes per episode.

Egocentric action set: ACTION1 forward, ACTION2 drop, ACTION3 turn-left,
ACTION4 turn-right, ACTION5 interact. Frame rotation is a per-level render
augmentation and does not touch the egocentric actions.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_minigrid_unlockpickup_training.py --episodes 1000 \
        --out data/training_multi_level/minigrid_unlockpickup
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.minigrid import MinigridSolver          # noqa: E402


class MinigridUnlockPickupSolver(MinigridSolver):
    env_id = "MiniGrid-UnlockPickup-v0"
    action_set = (1, 2, 3, 4, 5)
    objective = "carry"
    target = ("box", None)


if __name__ == "__main__":
    sys.exit(MinigridUnlockPickupSolver.main())
