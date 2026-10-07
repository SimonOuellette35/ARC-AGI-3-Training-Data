"""Generate Phase-1 training data for tw02 (ColorSplit).

The Witness's separation panel: the line cuts the board into regions, and no
region may hold two different square colours. Levels are synthesised per
(seed, level) -- a random line first, then each region it cuts gets a colour and
a few squares of that colour -- so every panel is solvable by construction;
``--no-augment`` plays the shipped five.

Panels stay at 4x4: the expert's solution enumeration is exhaustive there (so
its play is exactly optimal), and only budget-truncated at 5x5.

Expert, recovery and augmentation plumbing are shared with tw01/tw03..tw06 in
``solvers/common/witness.py``.

Usage (repo root, ARC-AGI-3 conda python):
    python solvers/generate_tw02_training.py --episodes 1000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.tw02.tw02 import Tw02                                    # noqa: E402
from solvers.common.witness import (                                # noqa: E402
    SQUARE_COLORS, SinglePathPanel, WitnessSolver, grid_for, parse_cell_map,
    pick_endpoints, random_line, read_common, split_regions)


class Tw02Solver(WitnessSolver):
    game_id = "tw02"
    game_cls = Tw02
    level_shapes = ((3, 3), (3, 3), (4, 4), (4, 4), (4, 4))

    def read_panel(self, game):
        data, cols, rows, starts, end, bps = read_common(game)
        squares = parse_cell_map(data["squares"])
        grid = grid_for(cols, rows)

        def check(path):
            for region in split_regions(grid, path):
                seen = {squares[cell] for cell in region if cell in squares}
                if len(seen) > 1:
                    return False
            return True

        sig = ("tw02", cols, rows, tuple(starts), end, tuple(sorted(bps)),
               tuple(sorted(squares.items())))
        return SinglePathPanel(cols, rows, starts, end, bps, check, sig)

    def random_config(self, rng, level_idx, cols, rows):
        start, end = pick_endpoints(cols, rows, rng)
        min_nodes = max(4, (cols + 1) * (rows + 1) // 2)
        line = random_line(cols, rows, start, end, rng, min_nodes=min_nodes)
        if line is None:
            return None
        regions = [sorted(r) for r in split_regions(grid_for(cols, rows), line)]
        if len(regions) < 2:
            # A line that separates nothing makes the squares vacuous: every
            # path that reaches the end would win.
            return None
        regions.sort()                     # set order is stable but not obvious
        palette = list(SQUARE_COLORS)
        rng.shuffle(palette)
        squares = {}
        for i, region in enumerate(regions):
            colour = palette[i % len(palette)]
            for cell in rng.sample(region, rng.randint(1, min(3, len(region)))):
                squares[f"{cell[0]},{cell[1]}"] = colour
        if len(set(squares.values())) < 2:
            return None
        cfg = {"cols": cols, "rows": rows,
               "starts": [list(start)], "end": list(end), "squares": squares}
        return cfg, line


if __name__ == "__main__":
    sys.exit(Tw02Solver.main())
