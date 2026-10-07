"""Generate Phase-1 training data for the PuzzleScript game ps:drop_maze.

The harness -- the engine-blackbox A* expert, the trajectory recorder and the
BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other search-the-interpreter ps: generators. This file is only the game-specific
part: its name, its state key, its goal heuristic and the three ways it deviates
from the family (two directions, no action remap, all-levels-must-win).

Each solved seed yields one multi-level episode JSON in the format the
encoder/dynamics stack expects:

    {
      "game_id": "puzzlescript_drop_maze",
      "levels": [
        {
          "level_id": 0,
          "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
          "actions":      [a_0, a_1, ..., a_{T-1}]  # length T
        },
        ...
      ]
    }

actions[0] is the RESET that produced obs[0]; actions[i] for i>=1 is the action
that took the agent from obs[i-1] to obs[i]. ONE Drop_Maze seed is already a
multi-level game (5 levels), so each solved seed yields one complete multi-level
episode -- no cross-seed bundling.

The game
--------
Drop_Maze is a "rotate the maze, drop the ball" puzzle. Each level is a 2x2 block
of quadrants holding four pre-rotated copies of the maze whose balls are kept in
sync; pressing LEFT / RIGHT rotates gravity (walks the Player one quadrant) and
every ball drops. The win condition is "all balls on their targets". The only
meaningful actions are LEFT (ACTION3) and RIGHT (ACTION4) -- UP/DOWN are
cancelled by the game and ACTION does nothing. Each perform_action returns a
single settled frame (the internal fall animation is drained inside the step),
so the recorded (obs, action) granularity matches how an agent actually sees the
game.

Rotation
--------
Drop_Maze is the one PuzzleScript game in
`PuzzleScriptAdapter._NO_ACTION_REMAP_GAMES`: its rotation is a NATIVE in-engine
mechanic, so the adapter neither rot90s the frame nor remaps the input, and
left/right must mean the same thing on screen and in the engine
(``remap_actions = False``). The augmentation is instead a random rotation-START
(a random number of native rotate-steps at reset), plus an optional horizontal
flip and independent recolors of background / walls / ball / goal per
(seed, level).

That rotation-start is the reason for ``require_all_levels``: it is the one
engine-relevant per-seed variable, so a level winnable at one seed can be
stranded at another (gravity is irreversible -- a ball can drop into a dead end).
Solvability is therefore NOT seed-independent here, and the family's one-shot
"discover the solvable levels at seed 0" would wrongly drop a level for every
later seed. Instead every level is attempted and a seed counts only if all of
them win; seeds stranded by their rotation-start are simply skipped by the CLI.

Expert solver
-------------
A* over the REAL engine dynamics, with two family deviations. The state key is
narrowed to the DYNAMIC cells (balls + Player) because walls / targets /
background genuinely are static here, which makes the key much cheaper -- and
that in turn requires ``scope_by_level``, since two levels can share a ball
layout while having different walls. The search branches on LEFT/RIGHT only.
Any returned plan is a genuine WIN path.

Usage (run from the repo root):
    python solvers/generate_drop_maze_training.py --episodes 500 \
        --out data/training_multi_level/drop_maze
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import PSAStarSolver, PSExpert  # noqa: E402

GAME_NAME = "Drop_Maze"


class DropMazeExpert(PSExpert):
    """Dynamic-state-keyed A* over LEFT/RIGHT, heuristic = balls off target."""

    #: UP/DOWN are cancelled by the game and ACTION does nothing, so branching on
    #: them would only burn nodes on self-loops.
    directions = ["left", "right"]

    #: `_key` below is dynamic-objects-only, which is canonical only WITHIN a
    #: level -- see `PSExpert.scope_by_level`.
    scope_by_level = True

    def setup(self) -> None:
        g = self.g
        self.ball_ids = {g.obj_name_to_idx[n] for n in g.or_groups.get("ball", [])
                         if n in g.obj_name_to_idx}
        self.player_ids = set(self.game._engine._player_indices)
        self.dyn_ids = self.ball_ids | self.player_ids
        self.target_id = g.obj_name_to_idx.get("target")

    def _key(self, eng) -> frozenset:
        # Only balls + Player move; walls / target / background are static per
        # level, so keying on the dynamic cells is a complete and much cheaper
        # canonical state (scoped by level via ``scope_by_level``).
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & self.dyn_ids)
        )

    def heuristic(self, eng) -> int:
        # Balls not co-located with a target (win => every ball on a target).
        # Not strictly admissible -- one rotation can settle several balls at
        # once -- so plans are near-optimal, which is fine for demonstrations.
        t = self.target_id
        off = 0
        for row in eng.grid:
            for cell in row:
                if (cell & self.ball_ids) and t not in cell:
                    off += 1
        return off


class DropMazeSolver(PSAStarSolver):
    game_id = "puzzlescript_drop_maze"
    game_name = GAME_NAME
    expert_cls = DropMazeExpert

    #: Native in-engine rotation -- the adapter passes input through untouched.
    remap_actions = False
    #: Solvability is seed-dependent (rotation-start); see the docstring.
    require_all_levels = True

    node_cap = 400_000
    max_steps = 200


if __name__ == "__main__":
    sys.exit(DropMazeSolver.main())
