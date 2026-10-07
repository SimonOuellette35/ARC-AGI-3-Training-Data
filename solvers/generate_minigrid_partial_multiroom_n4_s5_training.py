"""Generate Phase-1 training data for partial:MiniGrid-MultiRoom-N4-S5-v0.

The partially-observed twin of ``minigrid_multiroom_n4_s5``: reach the green
goal square in a chain of rooms behind shut doors, seen only through the
agent's 7x7 egocentric cone.

The expert holds no map it was given -- it builds one out of what it has
actually seen, plans over that, and when the thing it needs is not in it yet,
walks to the edge of the known region to look. All of that is in
``common/minigrid_partial.py``, which reads nothing but the observation; this
file only says which game it is.

The recorded target is the best action GIVEN WHAT THE AGENT KNOWS, which is
what an optimal target has to mean under partial observation. Exploration
prefix and perturbation bursts stay on (``supports_recovery = True``); one
level per env reset, ``levels_per_episode`` levels per episode.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_minigrid_partial_multiroom_n4_s5_training.py --episodes 1000 \
        --out data/training_multi_level/minigrid_partial_multiroom_n4_s5
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.minigrid_partial import MinigridPartialSolver  # noqa: E402


class MinigridPartialMultiRoomN4S5Solver(MinigridPartialSolver):
    env_id = "MiniGrid-MultiRoom-N4-S5-v0"
    action_set = (1, 3, 4, 5)


if __name__ == "__main__":
    sys.exit(MinigridPartialMultiRoomN4S5Solver.main())
