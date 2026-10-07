"""Generate Phase-1 training data for tw06 (TriCount).

The Witness's triangle panel: a cell showing N triangles must have exactly N of
its four edges walked by the line. Levels are synthesised per (seed, level) --
a random line first, then the triangle counts READ OFF that line -- so every
panel is solvable by construction; ``--no-augment`` plays the shipped two.

Expert, recovery and augmentation plumbing are shared with tw01..tw05 in
``solvers/common/witness.py``.

Usage (repo root, ARC-AGI-3 conda python):
    python solvers/generate_tw06_training.py --episodes 1000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.tw06.tw06 import Tw06                                    # noqa: E402
from solvers.common.witness import (                                # noqa: E402
    SinglePathPanel, WitnessSolver, cell_edge_count, parse_cell_map,
    path_edges, pick_endpoints, random_line, read_common)


class Tw06Solver(WitnessSolver):
    game_id = "tw06"
    game_cls = Tw06
    level_shapes = ((3, 3), (3, 3), (4, 4), (4, 4), (5, 5))
    triangle_counts = (1, 2, 3, 3, 4)

    def read_panel(self, game):
        data, cols, rows, starts, end, bps = read_common(game)
        triangles = tuple(sorted(parse_cell_map(data["triangles"]).items()))

        def check(path):
            edges = path_edges(path)
            return all(cell_edge_count(cell, edges) == n for cell, n in triangles)

        sig = ("tw06", cols, rows, tuple(starts), end, tuple(sorted(bps)),
               triangles)
        return SinglePathPanel(cols, rows, starts, end, bps, check, sig)

    def random_config(self, rng, level_idx, cols, rows):
        start, end = pick_endpoints(cols, rows, rng)
        min_nodes = max(4, (cols + 1) * (rows + 1) // 2)
        line = random_line(cols, rows, start, end, rng, min_nodes=min_nodes)
        if line is None:
            return None
        edges = path_edges(line)
        # Only 1..3 are drawable (and a 0 would say "keep the line off this cell
        # entirely", which the renderer has no glyph for), so cells the line
        # misses or fully encircles are not candidates.
        counts = {}
        for r in range(rows):
            for c in range(cols):
                n = cell_edge_count((c, r), edges)
                if 1 <= n <= 3:
                    counts[(c, r)] = n
        want = self.triangle_counts[level_idx]
        if len(counts) < want:
            return None
        cells = rng.sample(sorted(counts), want)
        cfg = {"cols": cols, "rows": rows,
               "starts": [list(start)], "end": list(end),
               "triangles": {f"{c},{r}": counts[(c, r)] for c, r in cells}}
        return cfg, line


if __name__ == "__main__":
    sys.exit(Tw06Solver.main())
