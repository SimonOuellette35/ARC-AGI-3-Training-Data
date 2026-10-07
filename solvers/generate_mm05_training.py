"""Generate Phase-1 training data for the MM05 game (games/mm05/mm05.py).

MM05 ("Memory Sticky") is meant to be [mm01](generate_mm01_training.py) -- the same 7
levels of 2..8 colour pairs, the same 60-step budget -- plus a "sticky" rule: tiles
orthogonally adjacent to a matched pair cannot be flipped.

**The sticky rule is dead code: mm05 is currently mm01.** ``_flip_slot`` only
considers matched pairs at Manhattan distance 1 and then blocks cells in
``orth_neighbors(a) & orth_neighbors(b)``. On a square grid two orthogonally adjacent
cells have NO common orthogonal neighbour, so that intersection is always empty and
the guard never fires -- verified over 60 seeds x 7 levels, zero flips blocked. The
metadata's rule ("adjacent to a matched pair") is the UNION of the endpoints'
neighbours, not the intersection.

Fixing it is a game-design call, not a solver one, and it is not obviously safe:
under union semantics blocked tiles stay blocked forever, so match ORDER starts to
matter and some boards may become unwinnable -- which would turn this generator's
strategy into a search problem and make solvability a per-seed question.

So this solver behaves as the shipped game does, but does not assume the rule is
dead: ``_click_until_landed`` retries a flip and reports failure if it never lands,
so if the rule is ever fixed, blocked tiles surface as skipped seeds rather than
silently corrupt data. Should that happen, revisit the strategy here.

Solver
------
Honest and non-privileged: tile colours reach the solver only by reading the
rendered frame -- never from ``game._slot_colors``. A hidden tile's colour is not in
the observation, so an oracle plan would be unlearnable by construction; see
``common.mm`` for the reasoning and the strategy. Worst case 6n = 48 steps at n = 8
pairs against 59 usable.

Determinism / augmentation
--------------------------
``Mm05`` takes ``seed=``, and every random thing about it is drawn from that
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
    python solvers/generate_mm05_training.py --episodes 1000 \
        --out data/training_multi_level/mm05
    python solvers/generate_mm05_training.py --verify
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.mm05.mm05 import Mm05  # noqa: E402
from solvers.common.mm import MmSolver, MmSpec  # noqa: E402

SPEC = MmSpec(game_id="mm05", game_cls=Mm05, group_size=2)

if __name__ == "__main__":
    sys.exit(MmSolver.main(SPEC))
