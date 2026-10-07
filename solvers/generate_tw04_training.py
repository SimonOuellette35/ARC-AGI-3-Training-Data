"""Generate Phase-1 training data for tw04 (SymDraw).

The Witness's Symmetry Island panel: the player drives the blue line and a
yellow one mirrors every move; both must land on their own end node with their
own waypoints covered. Levels are synthesised per (seed, level) by walking the
PAIR of lines at random and then declaring where they stopped to be the two end
nodes -- so every panel is solvable by construction; ``--no-augment`` plays the
shipped five.

Because the yellow line is a pure function of the blue one, the solver's state
is still just the blue path; see ``SymmetryPanel`` in
``solvers/common/witness.py``, which also owns the expert and the recovery /
augmentation plumbing shared with tw01..tw06.

Usage (repo root, ARC-AGI-3 conda python):
    python solvers/generate_tw04_training.py --episodes 1000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.tw04.tw04 import Tw04                                    # noqa: E402
from solvers.common.witness import (                                # noqa: E402
    SymmetryPanel, WitnessSolver, corners, node_mirror, random_mirror_line)


class Tw04Solver(WitnessSolver):
    game_id = "tw04"
    game_cls = Tw04
    level_shapes = ((4, 3), (4, 3), (3, 4), (4, 4), (5, 4))
    symmetries = ("horizontal", "horizontal", "vertical", "rotational",
                  "horizontal")
    dot_counts = (0, 1, 1, 2, 2)

    def read_path(self, game):
        return tuple(tuple(n) for n in game._blue_path)

    def read_panel(self, game):
        data = game.current_level._data
        bps = [(tuple(a), tuple(b)) for a, b in data.get("breakpoints", [])]
        panel = SymmetryPanel(
            data["cols"], data["rows"], data["symmetry"],
            tuple(data["blue_start"]), tuple(data["blue_end"]),
            tuple(data["yellow_start"]), tuple(data["yellow_end"]),
            [tuple(d) for d in data["blue_dots"]],
            [tuple(d) for d in data["yellow_dots"]],
            bps, None)
        panel._sig = ("tw04", panel.cols, panel.rows, panel.symmetry,
                      panel.blue_start, panel.blue_end, panel.yellow_start,
                      panel.yellow_end, tuple(sorted(panel.blue_dots)),
                      tuple(sorted(panel.yellow_dots)), tuple(sorted(bps)))
        return panel

    def random_config(self, rng, level_idx, cols, rows):
        symmetry = self.symmetries[level_idx]
        blue_start = rng.choice(corners(cols, rows))
        yellow_start = node_mirror(blue_start, cols, rows, symmetry)
        if yellow_start == blue_start:
            return None                    # the two lines would be one line
        min_nodes = max(4, (cols + 1) * (rows + 1) // 3)
        proto, line = random_mirror_line(cols, rows, symmetry, blue_start,
                                         yellow_start, rng, min_nodes=min_nodes)
        if line is None:
            return None
        yellow = proto.yellow_of(line)
        n_dots = self.dot_counts[level_idx]
        if n_dots and len(line) - 2 < n_dots:
            return None
        blue_dots = rng.sample(list(line[1:-1]), n_dots) if n_dots else []
        yellow_dots = rng.sample(list(yellow[1:-1]), n_dots) if n_dots else []
        cfg = {"cols": cols, "rows": rows, "symmetry": symmetry,
               "blue_start": list(blue_start), "blue_end": list(line[-1]),
               "yellow_start": list(yellow_start), "yellow_end": list(yellow[-1]),
               "blue_dots": [list(d) for d in blue_dots],
               "yellow_dots": [list(d) for d in yellow_dots]}
        return cfg, line


if __name__ == "__main__":
    sys.exit(Tw04Solver.main())
