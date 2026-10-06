"""
tw01_pathdots.py — PathDots: Mandatory waypoint (path-through dot) puzzles

One of The Witness's core mechanics: draw a line on a grid whose path must
pass through every marked dot.
This is the most basic constraint type, training the Agent's path-planning
and spatial reasoning abilities.

Core Knowledge: Objectness — mandatory waypoints on the path
ARC-AGI insight: "preserve specific elements" rule

Level design (progressive tutorial):
  Level 1: 3x3 grid, 1 dot on a straight path (naturally traversed)
  Level 2: 3x3 grid, 1 dot requiring a detour
  Level 3: 4x4 grid, 2 dots
  Level 4: 4x4 grid, 3 dots (requires precise path planning)
  Level 5: 5x5 grid, 4 dots (assumption-breaking: dots on edges)
"""

import sys
import os
# Ensure this directory is in sys.path (compatible with both SDK exec() and direct execution)
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
    START_COLOR, END_COLOR, DOT_COLOR, ERROR_COLOR, SUCCESS_COLOR,
    COLOR_BLACK,
)
from typing import List, Tuple, Set, Optional


class Tw01(AugmentedGame):
    """PathDots — Mandatory waypoint puzzles

    Rules: Draw a line from start to end; the path must pass through all
    marked yellow dots.
    """

    def __init__(self, seed: int = 0):
        self._init_augmentation(seed)   # FIRST: the base rotates the board on set_level
        self._seed = seed

        # Game state — must be initialised before super().__init__(),
        # because super().__init__() calls on_set_level() which sets these values
        self._path: List[Tuple[int, int]] = []
        self._grid: Optional[WitnessGrid] = None
        self._starts: List[Tuple[int, int]] = [(0, 0)]
        self._start: Tuple[int, int] = (0, 0)  # currently selected start
        self._end: Tuple[int, int] = (0, 0)
        self._dots: List[Tuple[int, int]] = []
        self._breakpoints: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
        self._drawing = False

        levels = self._create_levels()
        camera = Camera(background=GRID_BG, letter_box=COLOR_BLACK, interfaces=[])
        super().__init__(
            game_id="tw01",
            levels=levels,
            camera=camera,
            win_score=min(len(levels), 254),
            available_actions=[1, 2, 3, 4, 5],  # up/down/left/right + confirm
            seed=seed,
        )

    @staticmethod
    def _load_json_levels() -> Optional[list]:
        """Try to load levels from a JSON file. Returns [{"config": ..., "validated": bool}]."""
        try:
            import json
            levels_path = os.path.join(_code_dir, "levels", "tw01_levels.json")
            if os.path.exists(levels_path):
                with open(levels_path) as f:
                    data = json.load(f)
                return [{"config": entry["config"],
                         "validated": entry.get("validated", True)}
                        for entry in data["levels"]]
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_starts(cfg: dict) -> List[Tuple[int, int]]:
        """Parse the list of start points from config. Supports 'starts' (multiple) and 'start' (single)."""
        if "starts" in cfg:
            return [tuple(s) for s in cfg["starts"]]
        return [tuple(cfg["start"])]

    def _create_levels(self) -> List[Level]:
        """Create all levels. Loads from JSON first, falls back to hardcoded levels."""
        json_entries = self._load_json_levels()
        if json_entries:
            level_configs = []
            for entry in json_entries:
                cfg = entry["config"]
                starts = self._parse_starts(cfg)
                config = {
                    "cols": cfg["cols"],
                    "rows": cfg["rows"],
                    "starts": starts,
                    "end": tuple(cfg["end"]),
                    "dots": [tuple(d) for d in cfg["dots"]],
                    "validated": entry.get("validated", True),
                }
                if "breakpoints" in cfg:
                    config["breakpoints"] = [
                        (tuple(bp[0]), tuple(bp[1])) for bp in cfg["breakpoints"]
                    ]
                level_configs.append(config)
        else:
            # Hardcoded fallback
            level_configs = [
                {
                    "cols": 3, "rows": 3,
                    "starts": [(0, 0)], "end": (3, 0),
                    "dots": [(2, 0)],
                },
                {
                    "cols": 3, "rows": 3,
                    "starts": [(0, 0)], "end": (3, 0),
                    "dots": [(1, 1)],
                },
                {
                    "cols": 4, "rows": 4,
                    "starts": [(0, 0)], "end": (4, 4),
                    "dots": [(2, 0), (2, 4)],
                },
                {
                    "cols": 4, "rows": 4,
                    "starts": [(0, 0)], "end": (4, 4),
                    "dots": [(0, 2), (4, 2), (2, 4)],
                },
                {
                    "cols": 5, "rows": 5,
                    "starts": [(0, 0)], "end": (5, 5),
                    "dots": [(1, 0), (5, 1), (3, 5), (0, 3)],
                },
            ]

        levels = []
        for i, config in enumerate(level_configs):
            grid = WitnessGrid(config["cols"], config["rows"])
            frame = grid.render_grid()

            # Draw all start points and end point
            for s in config["starts"]:
                grid.draw_start(frame, s)
            grid.draw_end(frame, config["end"])

            # Draw mandatory waypoint dots
            for dot in config["dots"]:
                grid.draw_dot(frame, dot, DOT_COLOR)

            # Draw broken edges
            for bp in config.get("breakpoints", []):
                grid.draw_breakpoint(frame, bp[0], bp[1])

            # Unvalidated indicator
            if not config.get("validated", True):
                grid.draw_unvalidated_indicator(frame)

            # Create background sprite
            bg_sprite = Sprite(
                pixels=frame,
                name="grid_bg",
                x=0, y=0,
                layer=-10,
                blocking=BlockingMode.NOT_BLOCKED,
                interaction=InteractionMode.INTANGIBLE,
                tags=["sys_static"],
            )

            # Create cursor sprite (at the first start point)
            sx, sy = grid.node_to_pixel(*config["starts"][0])
            cursor_sprite = Sprite(
                pixels=[[CURSOR_COLOR]],
                name="cursor",
                x=sx, y=sy,
                layer=10,
                blocking=BlockingMode.NOT_BLOCKED,
                interaction=InteractionMode.TANGIBLE,
            )

            level_data = {
                "cols": config["cols"],
                "rows": config["rows"],
                "starts": config["starts"],
                "end": config["end"],
                "dots": config["dots"],
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
        """Reset state when switching levels."""
        data = level._data
        self._grid = WitnessGrid(data["cols"], data["rows"])
        # Support multiple start points
        if "starts" in data:
            self._starts = [tuple(s) for s in data["starts"]]
        else:
            self._starts = [tuple(data["start"])]
        self._start = self._starts[0]
        self._end = tuple(data["end"])
        self._dots = [tuple(d) for d in data["dots"]]
        self._breakpoints = set()
        for bp in data.get("breakpoints", []):
            n1, n2 = tuple(bp[0]), tuple(bp[1])
            self._breakpoints.add((min(n1, n2), max(n1, n2)))
        self._path = [self._start]
        self._drawing = True

    def _current_node(self) -> Tuple[int, int]:
        """Current node at the end of the path."""
        return self._path[-1] if self._path else self._start

    def _pixel_to_nearest_node(self, px: int, py: int) -> Optional[Tuple[int, int]]:
        """Convert pixel coordinates to the nearest grid node."""
        if not self._grid:
            return None

        best_node = None
        best_dist = float('inf')

        for row in range(self._grid.rows + 1):
            for col in range(self._grid.cols + 1):
                nx, ny = self._grid.node_to_pixel(col, row)
                dist = abs(px - nx) + abs(py - ny)
                if dist < best_dist:
                    best_dist = dist
                    best_node = (col, row)

        return best_node

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
        """Core game logic."""
        if not self._grid or not self._drawing:
            self.complete_action()
            return

        action = self.action.id
        current = self._current_node()

        if action == GameAction.ACTION5:
            # Submit path
            self._check_solution()
        elif action in (GameAction.ACTION1, GameAction.ACTION2,
                        GameAction.ACTION3, GameAction.ACTION4):
            # Directional movement
            dc, dr = 0, 0
            if action == GameAction.ACTION1:
                dr = -1  # up
            elif action == GameAction.ACTION2:
                dr = 1   # down
            elif action == GameAction.ACTION3:
                dc = -1  # left
            elif action == GameAction.ACTION4:
                dc = 1   # right

            target = (current[0] + dc, current[1] + dr)

            # Validate move legality
            if not self._is_valid_move(current, target):
                # Multiple start points: try auto-switching start
                if self._try_auto_select_start(dc, dr):
                    current = self._current_node()
                    target = (current[0] + dc, current[1] + dr)
                else:
                    self.complete_action()
                    return

            # If backtracking to the previous node
            if len(self._path) >= 2 and target == self._path[-2]:
                self._path.pop()
            elif target not in self._path:
                self._path.append(target)
            # else: target already in path (would create loop), ignore

            self._update_display()

        self.complete_action()

    def _is_valid_move(self, from_node: Tuple[int, int],
                       to_node: Tuple[int, int]) -> bool:
        """Check whether moving from one node to another is valid."""
        if not self._grid:
            return False

        fc, fr = from_node
        tc, tr = to_node

        # Check target is within grid bounds
        if not (0 <= tc <= self._grid.cols and 0 <= tr <= self._grid.rows):
            return False

        # Must be an adjacent node (Manhattan distance == 1)
        if abs(fc - tc) + abs(fr - tr) != 1:
            return False

        # Check breakpoints
        edge = (min(from_node, to_node), max(from_node, to_node))
        if edge in self._breakpoints:
            return False

        return True

    def _check_solution(self) -> None:
        """Check whether the current path satisfies all constraints."""
        # Check 1: path must reach the end point
        if self._current_node() != self._end:
            self._show_error()
            return

        # Check 2: path must pass through all mandatory waypoints
        path_set = set(self._path)
        for dot in self._dots:
            if dot not in path_set:
                self._show_error()
                return

        # All checks passed!
        self._show_success()
        self.next_level()

    def _show_error(self) -> None:
        """Show error feedback (brief red flash)."""
        # Reset path to the first start point
        self._start = self._starts[0]
        self._path = [self._start]
        self._update_display()

    def _show_success(self) -> None:
        """Show success feedback."""
        # Path turns green
        self._update_display(path_color=SUCCESS_COLOR)

    def _update_display(self, path_color: int = PATH_COLOR) -> None:
        """Update the display."""
        if not self._grid:
            return

        data = self.current_level._data
        frame = self._grid.render_grid()

        # Draw all start points and end point
        for s in self._starts:
            self._grid.draw_start(frame, s)
        self._grid.draw_end(frame, self._end)

        # Draw mandatory waypoint dots
        for dot in self._dots:
            covered = dot in set(self._path)
            color = SUCCESS_COLOR if covered else DOT_COLOR
            self._grid.draw_dot(frame, dot, color)

        # Draw broken edges
        for bp in self._breakpoints:
            self._grid.draw_breakpoint(frame, bp[0], bp[1])

        # Draw path
        for i in range(len(self._path) - 1):
            self._grid.draw_path_segment(frame, self._path[i], self._path[i + 1], path_color)

        # Draw cursor
        if self._path:
            cursor_node = self._path[-1]
            self._grid.draw_dot(frame, cursor_node, CURSOR_COLOR)

        # Redraw starts so they always appear green above the cursor
        for s in self._starts:
            self._grid.draw_start(frame, s)

        # Unvalidated indicator
        if not self.current_level._data.get("validated", True):
            self._grid.draw_unvalidated_indicator(frame)

        # Update background sprite
        bg_sprites = self.current_level.get_sprites_by_name("grid_bg")
        if bg_sprites:
            self.current_level.remove_sprite(bg_sprites[0])

        new_bg = Sprite(
            pixels=frame,
            name="grid_bg",
            x=0, y=0,
            layer=-10,
            blocking=BlockingMode.NOT_BLOCKED,
            interaction=InteractionMode.INTANGIBLE,
            tags=["sys_static"],
        )
        self.current_level.add_sprite(new_bg)

        # Move cursor sprite
        if self._path:
            cx, cy = self._grid.node_to_pixel(*self._path[-1])
            cursor_sprites = self.current_level.get_sprites_by_name("cursor")
            if cursor_sprites:
                cursor_sprites[0].set_position(cx, cy)
