"""Generate Phase-1 training data for the PuzzleScript game ps:count_mover
("Count Mover", Jonah Ostroff's one-joke moving-company Sokoban).

The harness -- the engine-blackbox A* expert, the rotation contract, the
trajectory recorder and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other search-the-interpreter ps:
generators. This file is only the game-specific part: its name, its levels, and
the two lines that say which object is pushed onto which.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_count_mover",
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
exactly. Each step also carries the full set of equally-optimal presses (see
"Optimal-action sets" below).

The game
--------
Textbook Sokoban and nothing else -- the same one-rule mechanic as
ps:bad_example, which is why both experts are two attribute lines on
`PSSokobanExpert`:

    [> Player | Crate] -> [> Player | > Crate]

with the win condition ``All Crate on Goal``. Wall, Player and Crate share a
collision layer, so a crate shoved into a wall or into a second crate simply
cancels the turn (pushes never chain); Goal sits on its own layer below them.
ACTION5 is bound to nothing, so the four directions are the whole action space
(`PSPushExpert.directions` already drops it from the search).

The levels
----------
The shipped file holds ONE level -- the game is a joke about counting, and its
single board is the punchline. One level is not a corpus: with no colour
augmentation the whole game would be 16 presentations (1 level x 4 rotations x
4 flips), and a policy would be memorising one board rather than learning what a
push is. So the file now carries 13, twelve of them authored here around the
original, ordered by expert plan length:

    level  size    crates  plan  budget  search   what it is
    0      8x9     1        9      60    0.0s     one crate, one push per axis
    1      6x9     1       11      60    0.0s     a wall between player and crate
    2      7x9     2       13      60    0.0s     two crates around a pillar
    3      7x9     2       15      60    0.1s     two crates, open room
    4      8x10    4       20      60    2.1s     a rank of four, pushed up
    5      7x9     3       22      60    0.6s     three crates, three columns
    6      11x11   4       31      60   46.8s     pinwheel: 4 crates, 4 corners
    7      9x10    4       33      60   14.7s     crates and goals interleaved
    8      9x9     7       35      60   14.6s     THE SHIPPED LEVEL (7 crates)
    9      9x11    2       42      84    0.3s     warehouse aisles: the routing
                                                  is the puzzle, two pushes each
    10     9x11    3       45      90    8.9s     three alcoves, one open floor
    11     9x11    3       61     122   20.1s     one door between the halves,
                                                  so crates cross one at a time
    12     10x9    3       69     138   12.9s     a single gap in a full-width
                                                  wall: everything funnels

``budget`` is the level's step limit -- this game runs TIGHTER than the
adapter's 200-step default, because efficiency is its whole joke, and
`games/ps:count_mover/ps:count_mover.py` is where that lives. The shipped level
was given 60 for a 35-move solution; the levels added here keep 60 wherever it
holds at least that margin and get 2x their plan where it does not (the last
four). This generator therefore records against THAT adapter
(``game_module_id``), not a bare `PuzzleScriptAdapter`: levels 11 and 12 cannot
be finished inside 60 moves at all, so a generator that built its own
200-step adapter would tape two episodes no agent could ever reproduce.

Every level is engine-verified solvable (that is what the ``plan`` column is --
an A* plan the interpreter actually walked to a WIN), none starts already won,
and each has as many goals as crates. The two shapes that make a Sokoban level
quietly impossible are avoided by construction and would have been caught
anyway: a crate in a corner cannot move at all, and a crate against the board's
outer wall can only ever slide ALONG that wall -- the first draft of level 12
put its crates on the top row and the expert reported it unsolvable in 0.6s.

Expert solver
-------------
`PSSokobanExpert` -- the shared plain-sokoban expert: A* over the REAL engine
dynamics whose successors are ``walk to the push cell, then push`` MACROS,
guided by per-target push-distance tables. See its docstring for the heuristic
and for why those tables are also the deadlock test. Nothing here is
game-specific except ``crate`` and ``goal``.

``weight = 1``, so plans are shortest up to the heuristic's greedy matching (a
guide, not a certificate -- again see `PSSokobanExpert`). The 13 searches cost
~121s together, all of it on the three levels whose boards are wide open; that
is a one-time cost at seed 0, and it is also written to
``data/count_mover_plans.json`` so it is not re-paid by every shard
`parallelize_generator` starts. Warm, a run starts in about a second. Delete the
file to re-derive it.

Optimal-action sets
-------------------
`PSPushExpert.annotate_walks` (via ``annotate``) labels every step: a push with
itself, and each step of a walk with every direction that keeps it on a shortest
route to the same push cell. The ties are the common case here -- most of a
Sokoban solution is the player walking to the next push cell, and any
interleaving of the two axes reaches it in the same number of moves and leaves
an identical board, since no rule fires on a bare move. Labelling one of them as
"the" answer would train against the truth on most steps. No step ever ships
unlabelled (the always-emit-optimal-targets rule).

The palette fix
---------------
The original art does not survive quantization onto the 16-colour ARC palette:
Crate's ``#630 #951`` and Wall's ``#842 #420`` ALL land on index 13, so every
crate rendered as a solid block of exactly the wall's colour -- indistinguishable
from the walls it sits between and from the letterbox border, with the game's
only movable object therefore absent from the frame. ``Crate`` is now
``Purple LightBlue`` (15 / 10); no rule, sprite shape or level was touched by
that edit. ``--audit`` is the check: it renders every cell COMPOSITION the game
can show (floor, wall, goal, crate, crate-on-goal, player, player-on-goal) at
every cell size the 13 levels actually render at and asserts they are pairwise
distinct. Composition, not object -- a crate ON a goal has to be readable as
both, which it is because the crate sprite is a 3x3 centre block that leaves the
goal's border ring showing.

Augmentation
------------
Count Mover's engine state after reset is identical for every seed (levels are
fixed ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip, each with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`), which together re-sample the board's
8-element symmetry group. Sokoban is gravity-free, its one rule is stated with
the relative ``>`` force, its win condition is positional in no way and its input
is screen-relative, so a flip is an exact symmetry of the mechanic; no sprite is
chiral, so no mirror can land one object's art on another's. There is
deliberately no colour augmentation: crate-on-goal is read as "the goal's red
ring with the purple crate showing through it", which a flattening recolor would
erase. 13 levels x 16 presentations = 208.

The expert plan is therefore seed-independent: solved once per level, cached,
and replayed per seed with that seed's remapped screen actions.

Usage (run from the repo root):
    python solvers/generate_count_mover_training.py --episodes 200 \
        --out data/training_multi_level/count_mover

    python solvers/generate_count_mover_training.py --plans   # level report
    python solvers/generate_count_mover_training.py --audit   # rendering audit
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame               # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSSokobanExpert    # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Count_Mover"

#: Disk cache of the per-level start plan AND its optimal-action sets. The
#: searches are seed-independent but cost ~121s of interpreter steps, which every
#: shard of `parallelize_generator` would otherwise repeat on every core. Delete
#: the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "count_mover_plans.json"


class CountMoverExpert(PSSokobanExpert):
    """The shared plain-sokoban expert: walk-and-push A* over the real
    interpreter, guided by per-target push-distance tables that double as the
    corner-deadlock test. See `PSSokobanExpert`; the mechanic here is the
    textbook one, so there is nothing else to say."""

    pushable_names = ("crate",)
    target_names = ("goal",)
    blocker_names = ("wall",)

    #: Label every step with its optimal SET (most steps are walks, whose axis
    #: order is free), and keep each level's start plan on disk.
    annotate = True
    plan_cache_path = PLAN_CACHE


class CountMoverSolver(PSAStarSolver):
    game_id = "puzzlescript_count_mover"
    game_name = GAME_NAME
    expert_cls = CountMoverExpert

    #: Record against the GAME FOLDER's adapter, not a bare `PuzzleScriptAdapter`.
    #: `games/ps:count_mover/ps:count_mover.py` runs TIGHTER than the adapter's
    #: 200-step default -- this game is about efficiency, so a level ends in
    #: GAME_OVER after 60 moves (more for the four long ones) -- and that is the
    #: game a live agent is handed. A generator building its own adapter would
    #: tape an episode nobody can reproduce.
    game_module_id = "ps:count_mover"

    #: Unweighted: 121s for all 13 boards, paid once and then cached to disk, so
    #: there is nothing worth buying by trading optimality away.
    weight = 1
    #: The slowest level settles well inside this; the cap is a runaway guard,
    #: not a tuning dial.
    node_cap = 400_000
    #: The longest plan is 69 moves; the rest is room for the exploration prefix
    #: and a re-plan after it. Stays under the adapter's own 200-step per-level
    #: budget, which would otherwise flip a long level to GAME_OVER mid-plan.
    max_steps = 150

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time cost visible as startup
        rather than as a mysteriously slow first seed, and it fills the disk
        cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _report() -> int:
    """Per-level plan length, board size, piece count and tie coverage."""
    game = CountMoverSolver().make_game(0)
    expert = CountMoverExpert(game, node_cap=CountMoverSolver.node_cap)
    crate = game._game.obj_name_to_idx["crate"]
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        crates = sum(1 for row in eng.grid for cell in row if crate in cell)
        found = expert.plan(eng, level)
        if found is None:
            print(f"level {level:2d}: NO PLAN")
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(found)
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"level {level:2d}: {eng.height:2d}x{eng.width:2d} "
              f"{crates} crates, {len(found):3d} moves "
              f"(budget {game._max_steps}, {room}), "
              f"{ties:3d} steps with a tie set ({ties / max(1, len(found)):.0%})")
    print(f"total {total} moves over {game.n_levels} levels")
    return 0


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    The bug this exists for shipped in the original art: Crate and Wall both
    quantized to ARC 13, so the only movable object in the game was invisible.
    An object-by-object colour check would not have caught the other half of it
    either -- what has to stay readable is the STACK (a crate on a goal is a
    different thing from a crate and from a goal), and the sprite detail that
    separates them is a few pixels wide, so it has to be checked at the cell
    size each board actually renders at."""
    game = CountMoverSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {"floor": (), "wall": ("wall",), "goal": ("goal",),
             "crate": ("crate",), "crate_on_goal": ("goal", "crate"),
             "player": ("player",), "player_on_goal": ("goal", "player")}

    sizes = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for name, objs in comps.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            frame = np.asarray(_render_frame(eng, g))
            cell = 64 // max(h, w)
            r, c = h // 2, w // 2
            shots[name] = frame[cell * r:cell * (r + 1),
                                cell * c:cell * (c + 1)].copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(CountMoverSolver.main())
