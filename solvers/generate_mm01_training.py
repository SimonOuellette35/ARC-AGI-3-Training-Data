"""Generate Phase-1 training data for the MM01 game (games/mm01/mm01.py).

MM01 ("Memory Match") is the classic concentration game, *mouse-click* only. Each of
the 7 levels lays out 2..8 colour PAIRS face-down (2x2, 2x3, 2x4, 2x5, 3x4, 2x7,
4x4); an ACTION6 click flips a tile face-up. Flip two of the same colour and the pair
stays up forever; flip two that differ and the board locks for 2 steps before both
turn back over. Clear every pair to advance. The step budget is 60 per level and a
mismatch costs 4 of them (2 clicks + 2 locked steps), so mismatches are the currency.

Solver
------
Honest and non-privileged: tile colours reach the solver only by reading the
rendered frame -- never from ``game._slot_colors``. A hidden tile's colour is not in
the observation, so an oracle plan would be unlearnable by construction; see
``common.mm`` for the reasoning and the strategy. Worst case is 6n = 48 steps at
n = 8 pairs against 59 usable, so every level fits. Each recorded level is a real,
engine-verified solve.

Determinism / augmentation
--------------------------
``Mm01`` takes ``seed=``, and every random thing about it is drawn from that
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
    python solvers/generate_mm01_training.py --episodes 1000 \
        --out data/training_multi_level/mm01
    python solvers/generate_mm01_training.py --verify
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.mm01.mm01 import Mm01  # noqa: E402
from solvers.common.mm import MmSolver, MmSpec  # noqa: E402

SPEC = MmSpec(game_id="mm01", game_cls=Mm01, group_size=2)

if __name__ == "__main__":
    sys.exit(MmSolver.main(SPEC))
