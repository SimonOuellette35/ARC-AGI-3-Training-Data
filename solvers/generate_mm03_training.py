"""Generate Phase-1 training data for the MM03 game (games/mm03/mm03.py).

MM03 ("Memory Triples (variant)") is byte-for-byte [mm02](generate_mm02_training.py)
-- same 5 triplet layouts (2x3, 3x3, 3x3, 3x4, 3x4), same 90-step budget, same
mouse-click-only action space -- with **one** functional difference:

    mm02.step():  if self._waiting_for_flip_back: ...; return   <-- BEFORE the
                  self._steps_remaining -= 1                         decrement
    mm03.step():  self._steps_remaining -= 1                    <-- AFTER
                  if self._waiting_for_flip_back: ...; return

So mm03 CHARGES for the 2 locked steps after a mismatched triplet: a bad turn costs
5 steps against mm02's 3 (measured, not inferred). That makes mm03 the strictly
tighter budget variant and the only reason it is a separate game.

Solver
------
Identical to mm02's -- honest, non-privileged, tile colours read only from the
rendered frame. See ``common.mm`` for the reasoning and the strategy. The wait-cost
difference never reaches the solver: it re-clicks until a flip lands rather than
modelling either game's lock timing, so one code path covers both. ~32 of 89 usable
steps, so the tighter charging is not binding.

Determinism / augmentation
--------------------------
``Mm03`` takes ``seed=``, and every random thing about it is drawn from that
seed through ``AugmentedGame``: the slot shuffle (``level_rng("layout")``) hides
which colour where per (seed, level), the background + ordered tile palette are
per seed (so one episode's colour scheme is shared by all its levels), and each
level is DISPLAYED at a per-(seed, level) rotation k. So the generator just
constructs with the seed (see ``common.mm.make_level``) -- never hand-seeding
``_rng``/``_color_rng``, which ``on_set_level`` overwrites anyway.

Because of the rotation, the solver works in SCREEN space: ``_read_board``
un-turns the frame before reading tiles, and recorded clicks are the screen
coordinates the tile appears at, so they replay through ``perform_action``.

Note ``mm03.py`` builds its ``levels`` list at MODULE level and hands the same list
to every ``Mm03()``. That is harmless here -- the base ctor clones into
``_clean_levels``, and ``_randomize_level`` rebuilds from the module-level
LEVEL_LAYOUTS rather than from a level's own data -- but it is worth knowing if that
file is ever edited.

Action schema (mixed simple + mouse, matching cd82 / cn04 / gp01)
    RESET / simple :  {"type": "simple", "index": k}                      # k in 0..5
    mouse click    :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_mm03_training.py --episodes 1000 \
        --out data/training_multi_level/mm03
    python solvers/generate_mm03_training.py --verify
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.mm03.mm03 import Mm03  # noqa: E402
from solvers.common.mm import MmSolver, MmSpec  # noqa: E402

SPEC = MmSpec(game_id="mm03", game_cls=Mm03, group_size=3)

if __name__ == "__main__":
    sys.exit(MmSolver.main(SPEC))
