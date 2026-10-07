"""Generate Phase-1 training data for tw01 (PathDots).

The Witness's mandatory-waypoint panel: draw a line from the start node to the
end node so that it passes through every marked node. Levels are synthesised per
(seed, level) -- a random line first, then dots sampled off it -- so each WIN
seed is a genuinely different set of panels; ``--no-augment`` plays the shipped
five instead.

Expert, recovery and augmentation plumbing are shared with tw02..tw06 in
``solvers/common/witness.py``.

Usage (repo root, ARC-AGI-3 conda python):
    python solvers/generate_tw01_training.py --episodes 1000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.tw01.tw01 import Tw01                                    # noqa: E402
from solvers.common.witness import (                                # noqa: E402
    SinglePathPanel, WitnessSolver, pick_endpoints, random_line, read_common)


class Tw01Solver(WitnessSolver):
    game_id = "tw01"
    game_cls = Tw01
    level_shapes = ((3, 3), (3, 3), (4, 4), (4, 4), (5, 5))
    dot_counts = (1, 2, 2, 3, 4)

    def read_panel(self, game):
        data, cols, rows, starts, end, bps = read_common(game)
        dots = frozenset(tuple(d) for d in data["dots"])
        sig = ("tw01", cols, rows, tuple(starts), end, tuple(sorted(bps)),
               tuple(sorted(dots)))
        return SinglePathPanel(cols, rows, starts, end, bps,
                               lambda path: dots <= set(path), sig)

    def random_config(self, rng, level_idx, cols, rows):
        start, end = pick_endpoints(cols, rows, rng)
        min_nodes = max(4, (cols + 1) * (rows + 1) // 2)
        line = random_line(cols, rows, start, end, rng, min_nodes=min_nodes)
        if line is None:
            return None
        # Dots come off the line, never off its endpoints: a dot on the start or
        # end node is satisfied by every candidate answer and constrains nothing.
        pool = list(line[1:-1])
        n_dots = self.dot_counts[level_idx]
        if len(pool) < n_dots:
            return None
        dots = rng.sample(pool, n_dots)
        cfg = {"cols": cols, "rows": rows,
               "starts": [list(start)], "end": list(end),
               "dots": [list(d) for d in dots]}
        return cfg, line


if __name__ == "__main__":
    sys.exit(Tw01Solver.main())
