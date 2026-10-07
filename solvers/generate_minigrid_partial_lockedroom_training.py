"""Generate Phase-1 training data for partial:MiniGrid-LockedRoom-v0.

The partially-observed twin of ``minigrid_lockedroom``: reach the green goal
square in six rooms off a corridor, one locked, seen only through the agent's
7x7 egocentric cone.

Almost all of the work is search: the key is in one of five rooms and nothing
about the level tells the agent which, so it opens doors and looks until the
key comes into view.

Blind, the search is most of the game and it does not always fit inside the
step budget, so this bundles fewer levels per episode than the rest of the
family (a level the expert runs out of budget on is discarded by the WIN
filter).

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
    python solvers/generate_minigrid_partial_lockedroom_training.py --episodes 1000 \
        --out data/training_multi_level/minigrid_partial_lockedroom
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.minigrid_partial import MinigridPartialSolver  # noqa: E402


class MinigridPartialLockedRoomSolver(MinigridPartialSolver):
    env_id = "MiniGrid-LockedRoom-v0"
    action_set = (1, 3, 4, 5)
    levels_per_episode = 2


if __name__ == "__main__":
    sys.exit(MinigridPartialLockedRoomSolver.main())
