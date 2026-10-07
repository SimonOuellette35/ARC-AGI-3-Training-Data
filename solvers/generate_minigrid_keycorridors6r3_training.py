"""Generate Phase-1 training data for MiniGrid-KeyCorridorS6R3-v0.

Pick up the ball hidden behind the locked door at the end of the corridor. The
key is in one of the side rooms, so the expert opens the ordinary doors to
find it, unlocks the far room, puts the key back down to free its hands, and
only then picks the ball up.

All of the solving is in ``common/minigrid.py`` -- one exact shortest-path
field over (cell, heading, carried object, cleared blockers), re-derived from
the live grid every step, which is what makes the epsilon exploration prefix
and the perturbation bursts recoverable (``supports_recovery = True``). One
maze per level, ``levels_per_episode`` mazes per episode.

Egocentric action set: ACTION1 forward, ACTION2 drop, ACTION3 turn-left,
ACTION4 turn-right, ACTION5 interact. Frame rotation is a per-level render
augmentation and does not touch the egocentric actions.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_minigrid_keycorridors6r3_training.py --episodes 1000 \
        --out data/training_multi_level/minigrid_keycorridors6r3
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.minigrid import MinigridSolver          # noqa: E402


class MinigridKeyCorridorS6R3Solver(MinigridSolver):
    env_id = "MiniGrid-KeyCorridorS6R3-v0"
    action_set = (1, 2, 3, 4, 5)
    objective = "carry"
    target = ("ball", None)


if __name__ == "__main__":
    sys.exit(MinigridKeyCorridorS6R3Solver.main())
