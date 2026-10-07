"""Generate training demonstrations for ps:wrap (Wrap, Joseph Mansfield).

Reach the goal by walking, wrapping across a room's border, and teleporting
between rooms. Wrap rules can move goal/teleport markers, and the Play/Immune
distinction affects teleportation, so position alone is not a sufficient state.

The shared PSEnumExpert enumerates the real interpreter's complete board
states, preserving these details without duplicating the rules. Backward BFS
from winning transitions gives shortest plans and every equally optimal action.
The seven shipped levels have 8, 11, 13, 11, 22, 21, and 24 reachable nonterminal
states; shortest solutions take 8, 7, 10, 7, 6, 17, and 14 presses respectively.

PSAStarSolver handles seed-dependent rendering, screen-action remapping, plan
caching, and exploration followed by RESET recovery. All seven levels must win
for an episode to be accepted. Planning uses the full engine board; recordings
retain the game's normal flickscreen view.

Usage from the repository root, with the ARC-AGI-3 environment's Python:
    python solvers/generate_wrap_training.py --episodes 1000
    python master_data_generator.py --game_list list.csv --episodes 1000
where list.csv may contain ps:wrap.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import PSAStarSolver, PSEnumExpert  # noqa: E402


class WrapExpert(PSEnumExpert):
    # No rule reads ACTION; only directional presses can change the board.
    directions = ["up", "down", "left", "right"]


class WrapSolver(PSAStarSolver):
    game_id = "puzzlescript_wrap"
    game_name = "Wrap"
    game_module_id = "ps:wrap"
    expert_cls = WrapExpert
    require_all_levels = True
    node_cap = 2_000
    max_steps = 200


if __name__ == "__main__":
    sys.exit(WrapSolver.main())
