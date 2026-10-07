"""Generate Phase-1 training data for tw05 (StarPair).

The Witness's star panel: within a region, any star colour that appears at all
must appear exactly twice. Levels are synthesised per (seed, level) -- a random
line first, then each region it cuts gets one or two colours placed as exact
pairs -- so every panel is solvable by construction; ``--no-augment`` plays the
shipped two.

Panels stay at 4x4 so the expert's solution enumeration is exhaustive (see
``solvers/common/witness.py``, which also owns the expert and the recovery /
augmentation plumbing shared with tw01..tw06).

Usage (repo root, ARC-AGI-3 conda python):
    python solvers/generate_tw05_training.py --episodes 1000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.tw05.tw05 import Tw05                                    # noqa: E402
from solvers.common.witness import (                                # noqa: E402
    SQUARE_COLORS, SinglePathPanel, WitnessSolver, grid_for, parse_cell_map,
    pick_endpoints, random_line, read_common, split_regions)


class Tw05Solver(WitnessSolver):
    game_id = "tw05"
    game_cls = Tw05
    level_shapes = ((3, 3), (3, 3), (4, 4), (4, 4), (4, 4))

    def read_panel(self, game):
        data, cols, rows, starts, end, bps = read_common(game)
        stars = parse_cell_map(data["stars"])
        grid = grid_for(cols, rows)

        def check(path):
            for region in split_regions(grid, path):
                counts = {}
                for cell in region:
                    if cell in stars:
                        counts[stars[cell]] = counts.get(stars[cell], 0) + 1
                if any(n != 2 for n in counts.values()):
                    return False
            return True

        sig = ("tw05", cols, rows, tuple(starts), end, tuple(sorted(bps)),
               tuple(sorted(stars.items())))
        return SinglePathPanel(cols, rows, starts, end, bps, check, sig)

    def random_config(self, rng, level_idx, cols, rows):
        start, end = pick_endpoints(cols, rows, rng)
        min_nodes = max(4, (cols + 1) * (rows + 1) // 2)
        line = random_line(cols, rows, start, end, rng, min_nodes=min_nodes)
        if line is None:
            return None
        regions = sorted(sorted(r) for r in split_regions(grid_for(cols, rows), line))
        if len(regions) < 2:
            return None
        stars, starred_regions = {}, 0
        for region in regions:
            n_colours = min(len(region) // 2, rng.randint(1, 2))
            if n_colours < 1:
                continue
            cells = rng.sample(region, 2 * n_colours)
            for i, colour in enumerate(rng.sample(SQUARE_COLORS, n_colours)):
                for cell in cells[2 * i:2 * i + 2]:
                    stars[f"{cell[0]},{cell[1]}"] = colour
            starred_regions += 1
        # One starred region is satisfiable without cutting anything apart, so
        # the line would be unconstrained; insist the pairing spans the split.
        if starred_regions < 2:
            return None
        cfg = {"cols": cols, "rows": rows,
               "starts": [list(start)], "end": list(end), "stars": stars}
        return cfg, line


if __name__ == "__main__":
    sys.exit(Tw05Solver.main())
