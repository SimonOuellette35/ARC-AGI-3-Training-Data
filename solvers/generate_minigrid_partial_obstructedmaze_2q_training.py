"""Generate Phase-1 training data for partial:MiniGrid-ObstructedMaze-2Q-v0.

The partially-observed twin of ``minigrid_obstructedmaze_2q``: pick up the
blue ball in a room maze of locked doors, hidden keys and blocked doorways,
seen only through the agent's 7x7 egocentric cone.

The hardest of the family to play blind: the keys are inside boxes that have
to be found first, and the doorways they open are blocked by balls that have
to be found and carried away.

By a wide margin the most expensive game in the family to generate: blind, a
level runs to the far end of its 1584-step budget and every one of those steps
re-plans over a map that just grew, which measures at several minutes per
level. It is bundled ONE level per episode for that reason -- and if you only
want the corpus, generate this one last.

The expert holds no map it was given -- it builds one out of what it has
actually seen, plans over that, and when the thing it needs is not in it yet,
walks to the edge of the known region to look. All of that is in
``common/minigrid_partial.py``, which reads nothing but the observation; this
file only says which game it is.

The recorded target is the best action GIVEN WHAT THE AGENT KNOWS, which is
what an optimal target has to mean under partial observation. One level per env
reset, ``levels_per_episode`` levels per episode.

BURSTS ARE OFF HERE (``default_noise = 0``), alone in the family, because this is
the one game whose post-burst solvability check cannot be afforded: it plays the
rest of the level out re-planning at every step, and 2Q's fields are the most
expensive in the file by an order of magnitude -- ~0.25s each over ~23k
enumerated states, against ~20ms on keycorridors6r3. Profiled, twelve bursts took
586s of a 647s level.

Turning them off does NOT make this game reliably cheap, and nothing here should
be read as claiming it does. 2Q's levels are BIMODAL: with bursts off, one
measured level solved in 15s and the next was still running after 25 minutes.
The tail, not the median, is what a 1000-episode run costs, and no amount of
burst tuning touches it -- it is a maze whose field has to be rebuilt from the
live pose at every step. Budget this corpus in the low hundreds of episodes,
generate it last, and expect a long tail of seeds that never finish.

The exploration prefix still runs (``supports_recovery`` stays True), so an
episode still opens off-policy and the expert still has to recover from it; what
is lost is the mid-level perturbation data. ``--noise 0.1`` restores it, at
roughly 30x on the levels that are fast without it (15s -> 490s, measured).

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_minigrid_partial_obstructedmaze_2q_training.py --episodes 1000 \
        --out data/training_multi_level/minigrid_partial_obstructedmaze_2q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.minigrid_partial import MinigridPartialSolver  # noqa: E402


class MinigridPartialObstructedMaze2QSolver(MinigridPartialSolver):
    env_id = "MiniGrid-ObstructedMaze-2Q-v0"
    action_set = (1, 2, 3, 4, 5)
    objective = "carry"
    target = ("ball", "blue")
    boxes_contain_keys = True
    levels_per_episode = 1
    default_noise = 0.0                  # see the note above: bursts cost 10x here


if __name__ == "__main__":
    sys.exit(MinigridPartialObstructedMaze2QSolver.main())
