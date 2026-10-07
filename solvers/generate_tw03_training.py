"""Generate Phase-1 training data for tw03 (ShapeFill).

The Witness's polyomino panel: the shapes sitting inside a region must tile that
region exactly -- no gaps, no overlaps, no spillover. Levels are synthesised per
(seed, level) by running the puzzle backwards: draw a random line, then CHOP
each region it cuts into random polyominoes and hand those pieces back as the
clues. So every panel has a tiling by construction; ``--no-augment`` plays the
shipped single level.

The region predicate is imported from the game itself (``_exact_cover``) rather
than reimplemented, so the expert can never drift from what the engine accepts.

Panels stay at 4x4 so the expert's solution enumeration is exhaustive (see
``solvers/common/witness.py``, which owns the expert and the recovery /
augmentation plumbing shared with tw01..tw06).

Usage (repo root, ARC-AGI-3 conda python):
    python solvers/generate_tw03_training.py --episodes 1000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.tw03.tw03 import Tw03, _exact_cover                      # noqa: E402
from solvers.common.witness import (                                # noqa: E402
    SinglePathPanel, WitnessSolver, grid_for, normalise_shape,
    partition_region, pick_endpoints, random_line, read_common, split_regions)


def _anchor_order(pieces):
    """Sort a tiling into the order ``_exact_cover`` will consume it.

    The engine's cover is greedy by index rather than a real exact-cover search:
    at step i it takes ``shapes[i]`` and only ever tries placements that cover
    ``min(uncovered)``. So a correct tiling presented in the wrong order is
    REJECTED. Emitting the pieces in anchor order makes the greedy walk follow
    the tiling we built.
    """
    remaining = {cell for piece in pieces for cell in piece}
    todo = list(pieces)
    out = []
    while todo:
        anchor = min(remaining)
        piece = next(p for p in todo if anchor in p)
        todo.remove(piece)
        remaining -= set(piece)
        out.append(piece)
    return out


class Tw03Solver(WitnessSolver):
    game_id = "tw03"
    game_cls = Tw03
    level_shapes = ((3, 3), (3, 3), (4, 4), (4, 4), (4, 4))
    max_piece = (2, 3, 3, 4, 4)

    def read_panel(self, game):
        data, cols, rows, starts, end, bps = read_common(game)
        tetris = {}
        for key, value in data["tetris"].items():
            c, r = key.split(",")
            tetris[(int(c), int(r))] = {
                "cells": sorted(tuple(s) for s in value["shape"]),
                "rotated": value.get("rotated", False),
                "negative": value.get("negative", False),
            }

        grid = grid_for(cols, rows)

        def check(path):
            for region in split_regions(grid, path):
                # Region-ITERATION order, exactly as ``Tw03._check_solution``
                # builds it: ``_exact_cover`` walks the shapes in this order and
                # forces shape i to cover the i-th uncovered anchor, so it is
                # order-sensitive and a sorted copy could disagree with the
                # engine.
                shapes = [tetris[cell] for cell in region if cell in tetris]
                if not shapes:
                    continue
                area = sum((-1 if s["negative"] else 1) * len(s["cells"])
                           for s in shapes)
                if area != len(region):
                    return False
                positive = [s for s in shapes if not s["negative"]]
                if not _exact_cover(positive, set(region), set(), 0):
                    return False
            return True

        sig = ("tw03", cols, rows, tuple(starts), end, tuple(sorted(bps)),
               tuple(sorted((cell, tuple(s["cells"]), s["rotated"], s["negative"])
                            for cell, s in tetris.items())))
        return SinglePathPanel(cols, rows, starts, end, bps, check, sig)

    def random_config(self, rng, level_idx, cols, rows):
        start, end = pick_endpoints(cols, rows, rng)
        min_nodes = max(4, (cols + 1) * (rows + 1) // 2)
        line = random_line(cols, rows, start, end, rng, min_nodes=min_nodes)
        if line is None:
            return None
        regions = split_regions(grid_for(cols, rows), line)
        if len(regions) < 2:
            return None
        tetris = {}
        for region in regions:
            cells = sorted(region)
            pieces = partition_region(cells, rng,
                                      max_piece=self.max_piece[level_idx])
            # ``_exact_cover`` is greedy by index: shape i must cover the i-th
            # remaining anchor (the lexicographically smallest uncovered cell).
            # Order the pieces that way, then hand out the symbol cells in the
            # region's own ITERATION order -- which is the order the engine will
            # read the shapes back in -- so the checker walks the tiling in
            # exactly the order that solves it.
            pieces = _anchor_order(pieces)
            holders = rng.sample(cells, len(pieces))
            holders = [cell for cell in region if cell in set(holders)]
            for cell, piece in zip(holders, pieces):
                tetris[f"{cell[0]},{cell[1]}"] = {
                    "shape": [list(s) for s in normalise_shape(piece)],
                    "rotated": False,
                    "negative": False,
                }
        cfg = {"cols": cols, "rows": rows,
               "starts": [list(start)], "end": list(end), "tetris": tetris}
        return cfg, line


if __name__ == "__main__":
    sys.exit(Tw03Solver.main())
