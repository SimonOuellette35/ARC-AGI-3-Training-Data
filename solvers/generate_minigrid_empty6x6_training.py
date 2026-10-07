"""Generate Phase-1 training data for MiniGrid-Empty-Random-6x6-v0.

A 6x6 empty room: goal fixed at the bottom-right, random agent start cell + start
heading. A BFS+turn expert reaches the goal. This is a standard, fully reversible
gridworld, so it uses the ordinary BaseSolver multilevel model with the epsilon
exploration prefix + bursts enabled (``supports_recovery = True``) -- one maze per
level, ``levels_per_episode`` mazes per episode. See ``common/minigrid.py``.

Egocentric action set: ACTION1 forward, ACTION3 turn-left, ACTION4 turn-right
(no doors or objects in an empty room, so no ACTION5 interact). Frame rotation is
a per-level render augmentation and does not touch the egocentric actions.

NOTE: this file predates the ``generate_minigrid_<game_id>_training.py`` naming
the other 42 MiniGrid generators use -- its ``game_id`` is
``minigrid_empty_random_6x6`` (the env is the RANDOM-start 6x6), and
``generate_minigrid_empty_6x6_training.py`` is the separate, fixed-start
``MiniGrid-Empty-6x6-v0``.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_minigrid_empty6x6_training.py --episodes 1000 \
        --out data/training_multi_level/minigrid_empty6x6
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.minigrid import MinigridSolver          # noqa: E402


class MinigridEmpty6x6Solver(MinigridSolver):
    env_id = "MiniGrid-Empty-Random-6x6-v0"
    game_id = "minigrid_empty_random_6x6"      # matches MinigridAdapter.game_id
    action_set = (1, 3, 4)                     # empty room: no toggle-door


if __name__ == "__main__":
    sys.exit(MinigridEmpty6x6Solver.main())
