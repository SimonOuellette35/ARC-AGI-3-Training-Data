"""Generate Phase-1 training data for the PuzzleScript game ps:bad_example
("Bad Example", the Sokoban clone the PuzzleScript tutorial ships as its
worked example).

The harness -- the engine-blackbox A* expert, the rotation contract, the
trajectory recorder and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other search-the-interpreter ps:
generators. This file is only the game-specific part: its name, the
walk-and-push search it uses, and its goal heuristic.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_bad_example",
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
action (post rotation/flip remap), i.e. the button an agent presses in the
augmented view, so replaying the recorded actions reproduces the recorded frames
exactly.

The game
--------
Textbook Sokoban, and nothing else. One rule --

    [> Player | Box] -> [> Player | > Box]

-- and one win condition, ``All Box on Target``. Wall, Box and Player share a
collision layer, so a box shoved into a wall or into a second box simply cancels
the turn; Target sits on its own layer above them, and its sprite is a hollow
white ring with a transparent middle, so a box ON a target renders as that ring
with the purple box showing through it and is distinguishable from a bare box.
ACTION5 is bound to nothing, so the four directions are the whole action space
(`PSPushExpert.directions` already drops it from the search).

Three levels, all winnable, none of them subtle:

  * **Level 0** -- the player starts hemmed in by four boxes in a plus, with the
    four targets on the diagonals. Every box has to be walked around and pushed
    on both axes. 25 moves.
  * **Level 1** -- a 4x3 checkerboard block of six boxes and six targets, which
    has to be shifted onto the opposite colour. 26 moves.
  * **Level 2** -- two boxes, two targets, an L-shaped board. The narrow one: the
    player starts in a pocket whose only exit is *through* a box, so the opening
    move is forced, and the far target is reachable only by pushing a box right
    along row 3 and then twice down column 4.

Expert solver
-------------
`PSPushExpert`: A* over the REAL engine dynamics whose successors are
``walk to the push cell, then push`` MACROS rather than single key presses. A
push is the only move with any effect in a Sokoban, so a primitive search burns
its budget re-deriving the walks between pushes; branching on macros makes the
search depth the number of pushes and finds each walk with one BFS. Costs stay in
primitive MOVES (the unit the agent pays) and the emitted plan is a flat list of
directions, so the recorder is unchanged.

The heuristic is a per-target PUSH-DISTANCE table (reverse BFS from the target
over push edges: a box reaches ``X`` from ``X - d`` only when ``X - d`` and the
player's stand cell ``X - 2d`` are both on the board and not walls), summed over
the bare targets by a greedy nearest-box matching, plus the player's walk to the
nearest box. Two things this buys beyond "count the uncovered targets":

  * The boards are corridors and pockets, so what a move costs is the detour, not
    the target count -- level 2's far target is 8 pushes away from a box that is
    2 cells from it as the crow flies.
  * **Deadlock detection for free.** A cell with two adjacent walls has no
    outgoing push at all, so the reverse BFS never reaches it and it is in no
    target's table. A box pushed into a corner therefore scores `_DEAD` and the
    subtree sinks to the back of the queue instead of being explored -- which is
    most of what makes level 1's checkerboard tractable, since almost every early
    push that is not part of the solution buries a box in a corner.

The greedy matching can overcount (it commits each target to its nearest free box
in board order rather than solving the assignment), so like the rest of this
family the heuristic is a search guide, not an optimality certificate. At
``weight = 1`` it still returns the shortest plan on all three levels here:
re-running the same search under a fully admissible variant (drop the matching,
take the nearest box per target with reuse allowed) returns byte-identical plans
on all three, for 283s of search instead of 7.4s.

Augmentation
------------
Bad Example's engine state after reset is identical for every seed (levels are
fixed ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip, each with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`), which together re-sample the board's
8-element symmetry group. Sokoban is gravity-free and its inputs are
screen-relative moves, so a flip is an exact symmetry of the mechanic -- and the
game takes no color augmentation, so without the flips its seeds would differ by
nothing but one of four rotations, over only three levels. There is deliberately
no recolor: box-on-target is read as "white ring with purple showing through",
which the flattening recolor surfaces would erase.

The expert plan is therefore seed-independent: solved once per level, cached, and
replayed per seed with that seed's remapped screen actions. Seed 0 pays ~7s for
all three searches; every later seed replays them from cache.

Usage (run from the repo root):
    python solvers/generate_bad_example_training.py --episodes 200 \
        --out data/training_multi_level/bad_example
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import PSAStarSolver, PSSokobanExpert  # noqa: E402

GAME_NAME = "Bad_Example"


class BadExampleExpert(PSSokobanExpert):
    """Walk-and-push A* over the real interpreter, guided by per-target
    push-distance tables.

    The whole expert is `PSSokobanExpert` -- the shared plain-sokoban heuristic,
    which is where the tables, the greedy matching and the deadlock argument now
    live (this game and ps:count_mover ran byte-identical copies of them). All
    that is game-specific is which object is pushed onto which."""

    pushable_names = ("box",)
    target_names = ("target",)
    blocker_names = ("wall",)


class BadExampleSolver(PSAStarSolver):
    game_id = "puzzlescript_bad_example"
    game_name = GAME_NAME
    expert_cls = BadExampleExpert

    #: Unweighted: all three boards fall out in ~7s together, so there is nothing
    #: to buy by trading optimality away.
    weight = 1
    node_cap = 400_000
    #: Plans are 25-30 moves; the ceiling only has to cover a re-plan after an
    #: epsilon detour, and `epsilon` is 0 for this family.
    max_steps = 120


if __name__ == "__main__":
    sys.exit(BadExampleSolver.main())
