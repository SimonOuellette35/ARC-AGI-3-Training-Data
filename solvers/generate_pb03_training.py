"""Generate Phase-1 training data for pb03 (Decoy Push).

One crate, the real goal pad, and a decoy pad within two cells of it that ends the
game the moment the crate is pushed onto it -- so the puzzle is a colour judgement
(the bottom-left HUD tile is painted in the decoy's colour as the key) on top of the
push planning. The expert forbids crate-onto-decoy transitions outright, so no
recorded trajectory ever demonstrates the losing push; the losses that DO appear
come from exploration, and are followed by the recorded RESET recovery.

The expert, the recovery story and the rotation handling are shared with pb01/pb02
in ``solvers/common/push_blocks.py``.

Usage (repo root, ARC-AGI-3 conda python):
    python solvers/generate_pb03_training.py --episodes 1000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.pb03.pb03 import Pb03                          # noqa: E402
from solvers.common.push_blocks import PushBlocksSolver   # noqa: E402


class Pb03Solver(PushBlocksSolver):
    game_id = "pb03"
    game_cls = Pb03


if __name__ == "__main__":
    sys.exit(Pb03Solver.main())
