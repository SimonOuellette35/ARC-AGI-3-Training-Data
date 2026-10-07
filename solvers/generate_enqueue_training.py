"""Generate Phase-1 training data for the PuzzleScript game ps:enqueue.

The whole harness -- the engine-blackbox A* expert, the rotation contract, the
trajectory recorder and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other search-the-interpreter ps:
generators. This file is only the game-specific part: its name, its goal
heuristic, and its search budget.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_enqueue",
      "levels": [
        {
          "level_id": 0,
          "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
          "actions":      [a_0, a_1, ..., a_{T-1}], # length T
        },
        ...
      ]
    }

actions[0] is the RESET that produced obs[0]; actions[i] for i>=1 is the action
that took the agent from obs[i-1] to obs[i]. The recorded index is the *screen*
action (post rotation-remap), i.e. the button an agent presses in the rotated
view, so replaying the recorded actions reproduces the recorded frames exactly.

The game
--------
Enqueue is a queue-manipulation puzzle by Allen Webster. Coloured blocks sit in
a queue; the Player walks the queue and presses ACTION to enqueue a block from a
Filler / dequeue a block to a Taker, dropping blocks onto matching coloured pads.
The win condition is "No Pad" (every pad covered by its matching block). The
meaningful actions are the four directions plus ACTION (ACTION5).

Solvable levels
---------------
The single-queue levels (indices 0-7) are fully simulated by the pure-Python
PuzzleScript interpreter and are solved + verified here. The later two-queue
levels (indices 8+) use a three-bracket-group filler/taker transfer rule that
the interpreter does not reproduce -- an exhaustive BFS shows no action sequence
ever removes a pad -- so those levels are cleanly skipped by
`ps_astar.discover_solvable` (they simply fail to plan) rather than emitted as
broken data.

Augmentation
------------
Enqueue's engine state after reset is identical for every seed, so the only
per-(seed, level) variables are the presentation augmentations: the frame
rotation (rotation_k in {0,1,2,3}) plus an independent horizontal flip (fliplr)
and vertical flip (flipud), each with the matching directional action remap.
Enqueue is gravity-free and its inputs are screen-relative moves, so a flip is
an exact symmetry (mirror the frame, swap that axis of the input). The expert
plan is therefore identical across seeds: it is solved once per level and
replayed from cache, and only the presented frames and the recorded screen-action
encoding differ between seeds. The four rotations x 2 hflip x 2 vflip re-sample
the board's 8-element symmetry group.

Usage (run from the repo root):
    python solvers/generate_enqueue_training.py --episodes 200 \
        --out data/training_multi_level/enqueue
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import PSAStarSolver, PSExpert  # noqa: E402

GAME_NAME = "Enqueue"


class EnqueueExpert(PSExpert):
    """Goal heuristic: pads still present (win => No Pad).

    Admissible -- one action clears at most one pad here -- so plain A* (weight 1)
    returns shortest WIN paths.
    """

    def setup(self) -> None:
        self.pad_ids = set(self.g.resolve_object_name("pad"))

    def heuristic(self, eng) -> int:
        pad = self.pad_ids
        n = 0
        for row in eng.grid:
            for cell in row:
                if cell & pad:
                    n += 1
        return n


class EnqueueSolver(PSAStarSolver):
    game_id = "puzzlescript_enqueue"
    game_name = GAME_NAME
    expert_cls = EnqueueExpert

    node_cap = 400_000
    max_steps = 200


if __name__ == "__main__":
    sys.exit(EnqueueSolver.main())
