"""Generate Phase-1 training data for the MM02 game (games/mm02/mm02.py).

MM02 ("Memory Triples") is concentration with TRIPLETS, *mouse-click* only. Each of
the 5 levels lays out 2..4 colours face-down, each appearing exactly three times
(2x3, 3x3, 3x3, 3x4, 3x4); an ACTION6 click flips a tile. The engine only scores at
the third flip: three of one colour clear permanently, otherwise the board locks and
all three turn back over. Clear every triplet to advance.

The step budget is 90 per level. **MM02's flip-back is free**: ``step`` returns on
``_waiting_for_flip_back`` *before* decrementing the counter, so the 2 locked steps
cost actions but not budget -- a mismatched turn costs 3. That single line is the
only thing separating mm02 from [mm03](generate_mm03_training.py), where the same
turn costs 5.

Solver
------
Honest and non-privileged: tile colours reach the solver only by reading the
rendered frame -- never from ``game._slot_colors``. A hidden tile's colour is not in
the observation, so an oracle plan would be unlearnable by construction; see
``common.mm`` for the reasoning and the strategy, which is shared with the pair
games -- the click-by-click rule generalises to triplets untouched. Note a doomed
turn cannot be aborted (re-clicking a face-up tile just burns a step), so the solver
spends its forced third click probing an unknown tile. ~32 of 89 usable steps.

Determinism / augmentation
--------------------------
``Mm02`` takes ``seed=``, and every random thing about it is drawn from that
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
    RESET / simple :  {"type": "simple", "index": k}                      # k in 0..5
    mouse click    :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_mm02_training.py --episodes 1000 \
        --out data/training_multi_level/mm02
    python solvers/generate_mm02_training.py --verify
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.mm02.mm02 import Mm02  # noqa: E402
from solvers.common.mm import MmSolver, MmSpec  # noqa: E402

SPEC = MmSpec(game_id="mm02", game_cls=Mm02, group_size=3)

if __name__ == "__main__":
    sys.exit(MmSolver.main(SPEC))
