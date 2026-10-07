"""Generate Phase-1 training data for the PuzzleScript game ps:everything_antimatters.

The whole harness -- the engine-blackbox A* expert, the rotation contract, the
trajectory recorder and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other search-the-interpreter ps:
generators. This file is only the game-specific part: its name, its goal
heuristic, its skipped levels and its search budget.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_everything_antimatters",
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
You steer a Cursor around a walled board. Pressing ACTION (X / ACTION5) "aims":
a Positive (matter) ghost and a Negative (antimatter) ghost both spawn on the
cursor cell, and the cursor freezes in place. Each subsequent direction press
moves the Positive ghost that way and the Negative ghost the OPPOSITE way (a
point reflection through the frozen cursor). Pressing ACTION again solidifies the
ghosts into a Positive and a Negative block at their current cells (the move is
cancelled if a ghost would land on a wall or on matching matter). Solidified
Positive+Negative annihilate if they share a cell; matter dropped on a Pit is
destroyed. Win condition: every TargetP cell holds a Positive AND every TargetN
cell holds a Negative. The meaningful actions are the four directions plus ACTION.

Solvable levels
---------------
The weighted-A* expert (weight 8) wins 9 of the 11 levels: {0, 1, 2, 3, 5, 6, 7,
8, 9}. Two levels resist the budget and are skipped up front via ``skip_levels``
rather than emitted as broken data:
  * level 4  -- a single column of ALTERNATING TargetN/TargetP over a pit. The
    mirror mechanic forces the unwanted ghost toward the pit for disposal, and
    filling the far targets needs the cursor threaded past already-placed matter;
    greedy primitive-action A* explodes here.
  * level 10 -- a large board with an irregular multi-pit layout.
Both are genuinely solvable by the mechanic but need a maneuver-level (macro)
planner this primitive-action A* does not implement. Because each level is an
independent trajectory in the training schema, an episode holding only the solved
levels is valid; level_id preserves the true (seed, level) provenance. Clear
``skip_levels`` (with a bigger ``node_cap`` / ``weight``) to force-attempt them.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps); only the presentation augmentations vary per (seed, level): the frame
rotation (rotation_k in {0,1,2,3}) and the border/background/cursor/block-class
recolor (see puzzlescript_adapter._recolor_surfaces). Antimatters has no flip
augmentation. The expert plan is therefore seed-independent: solved once per
level, cached, and replayed per seed with that seed's rotation-remapped screen
actions and recolored frames.

Usage (run from the repo root):
    python solvers/generate_antimatters_training.py --episodes 200 \
        --out data/training_multi_level/everything_antimatters
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import PSAStarSolver, PSExpert  # noqa: E402

GAME_NAME = "Everything_Antimatters"


class AntimattersExpert(PSExpert):
    """Goal heuristic: unsatisfied targets, plus targets blocked by the WRONG
    matter type (a Negative on a TargetP / a Positive on a TargetN must be
    cleared, costing an extra maneuver). Goal-aware and cheap; not admissible
    once weighted, which is the point -- weight 8 is what makes the 9 solvable
    levels land inside the node budget."""

    def setup(self) -> None:
        g = self.g
        self.pos_ids = set(g.resolve_object_name("positive"))
        self.neg_ids = set(g.resolve_object_name("negative"))
        self.tp_ids = set(g.resolve_object_name("targetp"))
        self.tn_ids = set(g.resolve_object_name("targetn"))

    def heuristic(self, eng) -> int:
        pos, neg, tp, tn = self.pos_ids, self.neg_ids, self.tp_ids, self.tn_ids
        h = 0
        for row in eng.grid:
            for cell in row:
                if cell & tp:
                    if not (cell & pos):
                        h += 1
                    if cell & neg:
                        h += 1
                if cell & tn:
                    if not (cell & neg):
                        h += 1
                    if cell & pos:
                        h += 1
        return h


class AntimattersSolver(PSAStarSolver):
    game_id = "puzzlescript_everything_antimatters"
    game_name = GAME_NAME
    expert_cls = AntimattersExpert

    #: See "Solvable levels" above -- skipped up front so discovery does not burn
    #: ~node_cap nodes per level fruitlessly at every startup.
    skip_levels = frozenset({4, 10})

    node_cap = 400_000
    weight = 8
    max_steps = 300


if __name__ == "__main__":
    sys.exit(AntimattersSolver.main())
