"""Generate Phase-1 training data for MiniGrid-MultiRoom-N6-v0.

Six rooms in a chain joined by (unlocked) doors, goal in the last room. A BFS+turn
expert routes through the rooms, opening each closed door (ACTION5) before
stepping through. Fully reversible navigation, so it uses the ordinary BaseSolver
multilevel model with the epsilon exploration prefix + bursts enabled
(``supports_recovery = True``) -- one maze per level, ``levels_per_episode`` mazes
per episode. See ``common/minigrid.py``.

Egocentric action set: ACTION1 forward, ACTION3 turn-left, ACTION4 turn-right,
ACTION5 interact (facing a door, that toggles it). Frame rotation is a per-level
render augmentation and does not touch the egocentric actions.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_minigrid_multiroom_n6_training.py --episodes 1000 \
        --out data/training_multi_level/minigrid_multiroom_n6
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.minigrid import MinigridSolver          # noqa: E402


class MinigridMultiRoomN6Solver(MinigridSolver):
    env_id = "MiniGrid-MultiRoom-N6-v0"
    game_id = "minigrid_multiroom_n6"
    action_set = (1, 3, 4, 5)                  # + toggle-door


if __name__ == "__main__":
    sys.exit(MinigridMultiRoomN6Solver.main())
