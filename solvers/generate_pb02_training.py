"""Generate Phase-1 training data for pb02 (Two-Box Push).

Two crates, two goal pads, every pad must be covered -- so ORDER matters: parking
the first crate can wall off the lane the second one needs, and the expert's A*
searches over both crates jointly rather than solving them one at a time. Layout,
walls, colours and display rotation are drawn per (seed, level).

The expert, the recovery story and the rotation handling are shared with pb01/pb03
in ``solvers/common/push_blocks.py``.

Usage (repo root, ARC-AGI-3 conda python):
    python solvers/generate_pb02_training.py --episodes 1000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.pb02.pb02 import Pb02                          # noqa: E402
from solvers.common.push_blocks import PushBlocksSolver   # noqa: E402


class Pb02Solver(PushBlocksSolver):
    game_id = "pb02"
    game_cls = Pb02


if __name__ == "__main__":
    sys.exit(Pb02Solver.main())
