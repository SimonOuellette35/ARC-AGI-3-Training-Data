"""Generate Phase-1 training data for MiniGrid-ObstructedMaze-1Q-v0.

Pick up the blue ball at the far corner of a room maze whose doors are locked,
whose keys are HIDDEN INSIDE BOXES and whose doorways are blocked by balls.
The expert never reads a box's contents (it can't -- they are not in the
frame); when it needs a key it cannot see, it goes and opens the nearest box
and re-plans against what turns up. Fewer levels per episode than the rest of
the family: the search is the biggest here, and the shipped v0 levels are not
all solvable, so the WIN filter discards more seeds.

All of the solving is in ``common/minigrid.py`` -- one exact shortest-path
field over (cell, heading, carried object, cleared blockers), re-derived from
the live grid every step, which is what makes the epsilon exploration prefix
and the perturbation bursts recoverable (``supports_recovery = True``). One
maze per level, ``levels_per_episode`` mazes per episode.

Egocentric action set: ACTION1 forward, ACTION2 drop, ACTION3 turn-left,
ACTION4 turn-right, ACTION5 interact. Frame rotation is a per-level render
augmentation and does not touch the egocentric actions.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_minigrid_obstructedmaze_1q_training.py --episodes 1000 \
        --out data/training_multi_level/minigrid_obstructedmaze_1q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.minigrid import MinigridSolver          # noqa: E402


class MinigridObstructedMaze1QSolver(MinigridSolver):
    env_id = "MiniGrid-ObstructedMaze-1Q-v0"
    action_set = (1, 2, 3, 4, 5)
    objective = "carry"
    target = ("ball", "blue")
    boxes_contain_keys = True
    levels_per_episode = 3


if __name__ == "__main__":
    sys.exit(MinigridObstructedMaze1QSolver.main())
