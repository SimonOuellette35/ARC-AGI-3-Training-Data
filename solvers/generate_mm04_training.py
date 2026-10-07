"""Generate Phase-1 training data for the MM04 game (games/mm04/mm04.py).

MM04 ("Memory Peek") is [mm01](generate_mm01_training.py) -- the same 7 levels of
2..8 colour pairs, the same 60-step budget -- plus one extra move: **ACTION5 peeks**,
turning every unmatched tile face-up. It is usable once per level and the reveal
lasts 5 steps, during which clicks are swallowed. Measured cost: the peek action
itself plus 4 dead steps; on the 5th the board collapses and that step's click does
land.

Solver
------
Peek first, then execute. The peek is the game's whole point and it dominates blind
play: 5 + 2n = 21 steps at n = 8 pairs against 59 usable, with zero mismatches.

Crucially this stays honest. The solver reads the board out of the **rendered peek
frame** -- never from ``game._slot_colors`` -- so every subsequent click is grounded
in an observation the model also saw. That is exactly the lesson the game is built to
teach: look, memorise, execute. (For a hidden tile no such frame exists, which is why
the other four mm games must probe instead; see ``common.mm``.)

The peek needs no special-casing in the play loop. ``_click_until_landed`` decides a
click landed only when the revealed set equals ``matched | flipped | {slot}``; while
the peek is up *every* tile is revealed, which visibly is not that set, so it simply
keeps re-clicking until the board collapses. Perfect knowledge then makes ``_choose``
open every turn with a known-complete pair.

Determinism / augmentation
--------------------------
``Mm04`` takes ``seed=``, and every random thing about it is drawn from that
seed through ``AugmentedGame``: the slot shuffle (``level_rng("layout")``) hides
which colour where per (seed, level), the background + ordered tile palette are
per seed (so one episode's colour scheme is shared by all its levels), and each
level is DISPLAYED at a per-(seed, level) rotation k. So the generator just
constructs with the seed (see ``common.mm.make_level``) -- never hand-seeding
``_rng``/``_color_rng``, which ``on_set_level`` overwrites anyway.

Because of the rotation, the solver works in SCREEN space: ``_read_board``
un-turns the frame before reading tiles, and recorded clicks are the screen
coordinates the tile appears at, so they replay through ``perform_action``.

Action schema (mixed simple + mouse, matching cd82 / cn04 / gp01)
    RESET / simple :  {"type": "simple", "index": k}      # k in 0..5; peek = 5
    mouse click    :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_mm04_training.py --episodes 1000 \
        --out data/training_multi_level/mm04
    python solvers/generate_mm04_training.py --verify
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.mm04.mm04 import Mm04  # noqa: E402
from solvers.common.mm import MmSolver, MmSpec  # noqa: E402

SPEC = MmSpec(game_id="mm04", game_cls=Mm04, group_size=2, peek=True)

if __name__ == "__main__":
    sys.exit(MmSolver.main(SPEC))
