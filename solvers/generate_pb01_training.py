"""Generate Phase-1 training data for pb01 (One-Box Push).

Push the single crate onto the goal pad. Five levels of rising difficulty (open
floor -> scattered blocks -> corridor barrier -> 10x10 with a stub -> multi-gap
barrier); every level's layout, walls, colours and display rotation are drawn per
(seed, level), so each WIN seed is a genuinely different multi-level game.

The expert, the recovery story and the rotation handling are shared with pb02/pb03
in ``solvers/common/push_blocks.py``.

Usage (repo root, ARC-AGI-3 conda python):
    python solvers/generate_pb01_training.py --episodes 1000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.pb01.pb01 import Pb01                          # noqa: E402
from solvers.common.push_blocks import PushBlocksSolver   # noqa: E402


class Pb01Solver(PushBlocksSolver):
    game_id = "pb01"
    game_cls = Pb01


if __name__ == "__main__":
    sys.exit(Pb01Solver.main())
