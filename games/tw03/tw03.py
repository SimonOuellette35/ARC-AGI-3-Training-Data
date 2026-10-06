"""
tw03_shapefill.py — ShapeFill: Polyomino exact-cover puzzles

The Witness polyomino mechanic: after the line splits the panel into regions,
the polyominoes in each region must tile it exactly.
Trains the Agent's spatial reasoning and exact-cover abilities.

Core Knowledge: Objectness + Geometry — shape matching and placement
ARC-AGI insight: jigsaw / tiling operations
"""

import sys
import os
try:
    _code_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
except NameError:
    _code_dir = os.getcwd()
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

from utils.arc_game import AugmentedGame
from arcengine import (
    ARCBaseGame, BlockingMode, Camera, GameAction,
    InteractionMode, Level, Sprite,
)
from utils.witness_grid import (
    WitnessGrid, GRID_BG, PATH_COLOR, CURSOR_COLOR,
    START_COLOR, END_COLOR, ERROR_COLOR, SUCCESS_COLOR,
    COLOR_BLACK, POLY_COLOR,
)
from typing import List, Tuple, Set, Dict, Optional


def _rotations(shape):
    """Generate all rotations of a shape."""
    shapes = [shape]
    current = shape
    for _ in range(3):
        current = [(y, -x) for x, y in current]
        min_x = min(x for x, y in current)
        min_y = min(y for x, y in current)
        current = [(x - min_x, y - min_y) for x, y in current]
        current.sort()
        if current not in shapes:
            shapes.append(current)
    return shapes


def _exact_cover(shapes, region_cells, placed, idx):
    """Backtracking check whether shapes can exactly cover the region."""
    remaining = region_cells - placed
    if not remaining:
        return idx >= len(shapes)
    if idx >= len(shapes):
        return len(remaining) == 0

    shape_info = shapes[idx]
    cells = shape_info["cells"]
    is_rotated = shape_info["rotated"]
    is_negative = shape_info["negative"]

    if is_negative:
        return _exact_cover(shapes, region_cells, placed, idx + 1)

    variants = _rotations(cells) if is_rotated else [cells]
    anchor = min(remaining)

    for variant in variants:
        for ref in variant:
            offset_x = anchor[0] - ref[0]
            offset_y = anchor[1] - ref[1]
            placed_cells = [(x + offset_x, y + offset_y) for x, y in variant]
            placed_set = set(placed_cells)

            if all(c in region_cells and c not in placed for c in placed_cells):
                new_placed = placed | placed_set
                if _exact_cover(shapes, region_cells, new_placed, idx + 1):
                    return True

    return False


class Tw03(AugmentedGame):
    """ShapeFill — Polyomino exact-cover puzzles

    Rules: Draw a line from start to end; the path partitions the panel.
    The polyominoes in each region must tile it exactly (no gaps, no overlaps).
    """

    def __init__(self, seed: int = 0):
        self._init_augmentation(seed)   # FIRST: the base rotates the board on set_level
        self._seed = seed
        self._path: List[Tuple[int, int]] = []
        self._grid: Optional[WitnessGrid] = None
        self._starts: List[Tuple[int, int]] = [(0, 0)]
        self._start: Tuple[int, int] = (0, 0)
        self._end: Tuple[int, int] = (0, 0)
        self._tetris: Dict[Tuple[int, int], dict] = {}
        self._breakpoints: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

        levels = self._create_levels()
        camera = Camera(background=GRID_BG, letter_box=COLOR_BLACK, interfaces=[])
        super().__init__(
            game_id="tw03",
            levels=levels,
            camera=camera,
            win_score=min(len(levels), 254),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )

    @staticmethod
    def _parse_starts(cfg: dict) -> List[Tuple[int, int]]:
        """Parse the list of start points from config. Supports 'starts' (multiple) and 'start' (single)."""
        if "starts" in cfg:
            return [tuple(s) for s in cfg["starts"]]
        return [tuple(cfg["start"])]

    @staticmethod
    def _load_json_levels() -> Optional[list]:
        """Load levels from a JSON file. Returns a list of dicts with 'config' and 'validated' fields."""
        try:
            import json
            levels_path = os.path.join(_code_dir, "levels", "tw03_levels.json")
            if os.path.exists(levels_path):
                with open(levels_path) as f:
                    data = json.load(f)
                return [{"config": entry["config"], "validated": entry.get("validated", True)} for entry in data["levels"]]
        except Exception:
            pass
        return None

    def _create_levels(self) -> List[Level]:
        json_entries = self._load_json_levels()
        if json_entries:
            level_configs = []
            for entry in json_entries:
                cfg = entry["config"]
                tetris = {}
                for k, v in cfg["tetris"].items():
                    parts = k.split(",")
                    cell = (int(parts[0]), int(parts[1]))
                    tetris[cell] = {
                        "shape": [tuple(s) for s in v["shape"]],
                        "rotated": v.get("rotated", False),
                        "negative": v.get("negative", False),
                    }
                config = {
                    "cols": cfg["cols"],
                    "rows": cfg["rows"],
                    "starts": self._parse_starts(cfg),
                    "end": tuple(cfg["end"]),
                    "tetris": tetris,
                    "validated": entry.get("validated", True),
                }
                if "breakpoints" in cfg:
                    config["breakpoints"] = [
                        (tuple(bp[0]), tuple(bp[1])) for bp in cfg["breakpoints"]
                    ]
                level_configs.append(config)
        else:
            # Hardcoded fallback: simple 2x2 square
            level_configs = [
                {
                    "cols": 3, "rows": 3,
                    "starts": [(0, 0)], "end": (3, 3),
                    "tetris": {
                        (0, 0): {"shape": [(0, 0), (1, 0), (0, 1), (1, 1)], "rotated": False, "negative": False},
                        (2, 0): {"shape": [(0, 0), (0, 1)], "rotated": False, "negative": False},
                    },
                },
            ]

        levels = []
        for i, config in enumerate(level_configs):
            grid = WitnessGrid(config["cols"], config["rows"])
            frame = grid.render_grid()

            for s in config["starts"]:
                grid.draw_start(frame, s)
            grid.draw_end(frame, config["end"])

            for cell, t in config["tetris"].items():
                grid.draw_polyomino(frame, cell, t["shape"], POLY_COLOR)

            # Draw broken edges
            for bp in config.get("breakpoints", []):
                grid.draw_breakpoint(frame, bp[0], bp[1])

            if not config.get("validated", True):
                grid.draw_unvalidated_indicator(frame)

            bg_sprite = Sprite(
                pixels=frame, name="grid_bg",
                x=0, y=0, layer=-10,
                blocking=BlockingMode.NOT_BLOCKED,
                interaction=InteractionMode.INTANGIBLE,
                tags=["sys_static"],
            )

            sx, sy = grid.node_to_pixel(*config["starts"][0])
            cursor_sprite = Sprite(
                pixels=[[CURSOR_COLOR]], name="cursor",
                x=sx, y=sy, layer=10,
                blocking=BlockingMode.NOT_BLOCKED,
                interaction=InteractionMode.TANGIBLE,
            )

            # Serialize tetris for level data
            tetris_data = {}
            for cell, t in config["tetris"].items():
                tetris_data[f"{cell[0]},{cell[1]}"] = {
                    "shape": [list(s) for s in t["shape"]],
                    "rotated": t["rotated"],
                    "negative": t["negative"],
                }

            level_data = {
                    "cols": config["cols"],
                    "rows": config["rows"],
                    "starts": config["starts"],
                    "end": config["end"],
                    "tetris": tetris_data,
                    "validated": config.get("validated", True),
            }
            if "breakpoints" in config:
                level_data["breakpoints"] = config["breakpoints"]

            level = Level(
                sprites=[bg_sprite, cursor_sprite],
                grid_size=(64, 64),
                data=level_data,
                name=f"Level {i + 1}",
            )
            levels.append(level)

        return levels

    def on_set_level(self, level: Level) -> None:
        data = level._data
        self._grid = WitnessGrid(data["cols"], data["rows"])
        # Support multiple start points
        if "starts" in data:
            self._starts = [tuple(s) for s in data["starts"]]
        else:
            self._starts = [tuple(data["start"])]
        self._start = self._starts[0]
        self._end = tuple(data["end"])
        self._tetris = {}
        for k, v in data["tetris"].items():
            parts = k.split(",")
            cell = (int(parts[0]), int(parts[1]))
            self._tetris[cell] = {
                "shape": [tuple(s) for s in v["shape"]],
                "rotated": v.get("rotated", False),
                "negative": v.get("negative", False),
            }
        self._breakpoints = set()
        for bp in data.get("breakpoints", []):
            n1, n2 = tuple(bp[0]), tuple(bp[1])
            self._breakpoints.add((min(n1, n2), max(n1, n2)))
        self._path = [self._start]

    def _try_auto_select_start(self, dc: int, dr: int) -> bool:
        """With multiple start points, try to auto-select one based on the first move direction.

        Only triggered when the path has just the initial start (len==1) and the
        current start cannot move. Iterates over all start points and selects the
        first one that can move in the (dc, dr) direction.
        """
        if len(self._path) != 1 or len(self._starts) <= 1:
            return False
        for alt in self._starts:
            alt_target = (alt[0] + dc, alt[1] + dr)
            if self._is_valid_move(alt, alt_target):
                self._start = alt
                self._path = [alt]
                return True
        return False

    def step(self) -> None:
        if not self._grid:
            self.complete_action()
            return

        action = self.action.id
        current = self._path[-1] if self._path else self._start

        if action == GameAction.ACTION5:
            self._check_solution()
        elif action in (GameAction.ACTION1, GameAction.ACTION2,
                        GameAction.ACTION3, GameAction.ACTION4):
            dc, dr = 0, 0
            if action == GameAction.ACTION1: dr = -1
            elif action == GameAction.ACTION2: dr = 1
            elif action == GameAction.ACTION3: dc = -1
            elif action == GameAction.ACTION4: dc = 1

            target = (current[0] + dc, current[1] + dr)

            # Validate move legality
            if not self._is_valid_move(current, target):
                # Multiple start points: try auto-switching start
                if self._try_auto_select_start(dc, dr):
                    current = self._path[-1]
                    target = (current[0] + dc, current[1] + dr)
                else:
                    self.complete_action()
                    return

            if len(self._path) >= 2 and target == self._path[-2]:
                self._path.pop()
            elif target not in self._path:
                self._path.append(target)
            self._update_display()

        self.complete_action()

    def _is_valid_move(self, from_node, to_node):
        if not self._grid:
            return False
        tc, tr = to_node
        if not (0 <= tc <= self._grid.cols and 0 <= tr <= self._grid.rows):
            return False
        fc, fr = from_node
        if abs(fc - tc) + abs(fr - tr) != 1:
            return False
        edge = (min(from_node, to_node), max(from_node, to_node))
        if edge in self._breakpoints:
            return False
        return True

    def _check_solution(self) -> None:
        if not self._grid:
            return

        if self._path[-1] != self._end:
            self._start = self._starts[0]
            self._path = [self._start]
            self._update_display()
            return

        # Region tiling check
        regions = self._grid.path_splits_regions(self._path)
        for region in regions:
            shapes_in_region = []
            total_positive_area = 0
            total_negative_area = 0

            for cell in region:
                if cell in self._tetris:
                    t = self._tetris[cell]
                    is_negative = t.get("negative", False)
                    shapes_in_region.append({
                        "cells": sorted(t["shape"]),
                        "rotated": t.get("rotated", False),
                        "negative": is_negative,
                    })
                    if is_negative:
                        total_negative_area += len(t["shape"])
                    else:
                        total_positive_area += len(t["shape"])

            if not shapes_in_region:
                continue

            expected_area = total_positive_area - total_negative_area
            if expected_area != len(region):
                self._start = self._starts[0]
                self._path = [self._start]
                self._update_display()
                return

            positive_shapes = [s for s in shapes_in_region if not s["negative"]]
            if not _exact_cover(positive_shapes, set(region), set(), 0):
                self._start = self._starts[0]
                self._path = [self._start]
                self._update_display()
                return

        self._update_display(path_color=SUCCESS_COLOR)
        self.next_level()

    def _update_display(self, path_color: int = PATH_COLOR) -> None:
        if not self._grid:
            return

        frame = self._grid.render_grid()
        for s in self._starts:
            self._grid.draw_start(frame, s)
        self._grid.draw_end(frame, self._end)

        for cell, t in self._tetris.items():
            self._grid.draw_polyomino(frame, cell, t["shape"], POLY_COLOR)

        # Draw broken edges
        for bp in self._breakpoints:
            self._grid.draw_breakpoint(frame, bp[0], bp[1])

        for i in range(len(self._path) - 1):
            self._grid.draw_path_segment(frame, self._path[i], self._path[i + 1], path_color)

        if self._path:
            self._grid.draw_dot(frame, self._path[-1], CURSOR_COLOR)

        # Redraw starts so they always appear green above the cursor
        for s in self._starts:
            self._grid.draw_start(frame, s)

        # Unvalidated indicator
        if not self.current_level._data.get("validated", True):
            self._grid.draw_unvalidated_indicator(frame)

        bg_sprites = self.current_level.get_sprites_by_name("grid_bg")
        if bg_sprites:
            self.current_level.remove_sprite(bg_sprites[0])

        new_bg = Sprite(
            pixels=frame, name="grid_bg",
            x=0, y=0, layer=-10,
            blocking=BlockingMode.NOT_BLOCKED,
            interaction=InteractionMode.INTANGIBLE,
            tags=["sys_static"],
        )
        self.current_level.add_sprite(new_bg)
