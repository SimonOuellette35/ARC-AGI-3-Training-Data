"""
tw04_symdraw.py — SymDraw: Symmetric line-drawing puzzles

Core mechanic from The Witness's Symmetry Island: the player controls one
line while a second line moves as its mirror image automatically.
Both lines must reach their respective endpoints simultaneously.

Core Knowledge: Geometry & Topology — symmetry transformations
ARC-AGI insight: rotation, reflection, mirroring operations; mental simulation

Level design:
  Level 1: 3x3, horizontal mirror, two starts and two ends
  Level 2: 3x3, horizontal mirror + 1 mandatory waypoint
  Level 3: 4x4, vertical mirror
  Level 4: 4x4, rotational symmetry (180 degrees)
  Level 5: 5x5, horizontal mirror + colored mandatory waypoints
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
    START_COLOR, END_COLOR, DOT_COLOR, ERROR_COLOR, SUCCESS_COLOR,
    COLOR_BLACK, COLOR_BLUE, COLOR_YELLOW,
)
from typing import List, Tuple, Set, Optional


class Tw04(AugmentedGame):
    """SymDraw — Symmetric line-drawing puzzles

    Rules: Control the blue line; the yellow line moves symmetrically.
    Both lines must travel from their respective starts to their respective ends.
    """

    def __init__(self, seed: int = 0):
        self._init_augmentation(seed)   # FIRST: the base rotates the board on set_level
        self._seed = seed

        # Game state — must be initialised before super().__init__()
        self._blue_path: List[Tuple[int, int]] = []
        self._yellow_path: List[Tuple[int, int]] = []
        self._grid: Optional[WitnessGrid] = None
        self._blue_start: Tuple[int, int] = (0, 0)
        self._blue_end: Tuple[int, int] = (0, 0)
        self._yellow_start: Tuple[int, int] = (0, 0)
        self._yellow_end: Tuple[int, int] = (0, 0)
        self._symmetry: str = "horizontal"  # horizontal, vertical, rotational
        self._blue_dots: List[Tuple[int, int]] = []
        self._yellow_dots: List[Tuple[int, int]] = []
        self._breakpoints: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

        levels = self._create_levels()
        camera = Camera(background=GRID_BG, letter_box=COLOR_BLACK, interfaces=[])
        super().__init__(
            game_id="tw04",
            levels=levels,
            camera=camera,
            win_score=min(len(levels), 254),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )

    def _mirror(self, node: Tuple[int, int], cols: int, rows: int) -> Tuple[int, int]:
        """Compute the mirrored node based on the symmetry type."""
        c, r = node
        if self._symmetry == "horizontal":
            return (cols - c, r)  # horizontal flip
        elif self._symmetry == "vertical":
            return (c, rows - r)  # vertical flip
        elif self._symmetry == "rotational":
            return (cols - c, rows - r)  # 180-degree rotation
        return node

    @staticmethod
    def _load_json_levels():
        """Try to load levels from a JSON file. Returns list[dict], each with 'config' and 'validated' fields."""
        try:
            import json
            levels_path = os.path.join(_code_dir, "levels", "tw04_levels.json")
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
                config = {
                    "cols": cfg["cols"],
                    "rows": cfg["rows"],
                    "symmetry": cfg["symmetry"],
                    "blue_start": tuple(cfg["blue_start"]),
                    "blue_end": tuple(cfg["blue_end"]),
                    "yellow_start": tuple(cfg["yellow_start"]),
                    "yellow_end": tuple(cfg["yellow_end"]),
                    "blue_dots": [tuple(d) for d in cfg["blue_dots"]],
                    "yellow_dots": [tuple(d) for d in cfg["yellow_dots"]],
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
                    "cols": 4, "rows": 3,
                    "symmetry": "horizontal",
                    "blue_start": (0, 0), "blue_end": (0, 3),
                    "yellow_start": (4, 0), "yellow_end": (4, 3),
                    "blue_dots": [], "yellow_dots": [],
                },
                {
                    "cols": 4, "rows": 3,
                    "symmetry": "horizontal",
                    "blue_start": (0, 0), "blue_end": (0, 3),
                    "yellow_start": (4, 0), "yellow_end": (4, 3),
                    "blue_dots": [(1, 1)], "yellow_dots": [(3, 1)],
                },
                {
                    "cols": 3, "rows": 4,
                    "symmetry": "vertical",
                    "blue_start": (0, 0), "blue_end": (3, 0),
                    "yellow_start": (0, 4), "yellow_end": (3, 4),
                    "blue_dots": [], "yellow_dots": [],
                },
                {
                    "cols": 4, "rows": 4,
                    "symmetry": "rotational",
                    "blue_start": (0, 0), "blue_end": (2, 0),
                    "yellow_start": (4, 4), "yellow_end": (2, 4),
                    "blue_dots": [(1, 1)], "yellow_dots": [(3, 3)],
                },
                {
                    "cols": 6, "rows": 4,
                    "symmetry": "horizontal",
                    "blue_start": (0, 0), "blue_end": (0, 4),
                    "yellow_start": (6, 0), "yellow_end": (6, 4),
                    "blue_dots": [(1, 0), (1, 2)],
                    "yellow_dots": [(5, 0), (5, 2)],
                },
            ]

        levels = []
        for i, config in enumerate(level_configs):
            grid = WitnessGrid(config["cols"], config["rows"])
            frame = grid.render_grid()

            # Blue start and end points
            grid.draw_dot(frame, config["blue_start"], COLOR_BLUE)
            grid.draw_dot(frame, config["blue_end"], COLOR_BLUE)

            # Yellow start and end points
            grid.draw_dot(frame, config["yellow_start"], COLOR_YELLOW)
            grid.draw_dot(frame, config["yellow_end"], COLOR_YELLOW)

            # Blue mandatory waypoints
            for dot in config["blue_dots"]:
                grid.draw_dot(frame, dot, COLOR_BLUE)
            # Yellow mandatory waypoints
            for dot in config["yellow_dots"]:
                grid.draw_dot(frame, dot, COLOR_YELLOW)

            # Draw broken edges
            for bp in config.get("breakpoints", []):
                grid.draw_breakpoint(frame, bp[0], bp[1])

            if not config.get("validated", True):
                grid.draw_unvalidated_indicator(frame)

            bg_sprite = Sprite(
                pixels=frame,
                name="grid_bg",
                x=0, y=0, layer=-10,
                blocking=BlockingMode.NOT_BLOCKED,
                interaction=InteractionMode.INTANGIBLE,
                tags=["sys_static"],
            )

            level_data = {
                    "cols": config["cols"],
                    "rows": config["rows"],
                    "symmetry": config["symmetry"],
                    "blue_start": config["blue_start"],
                    "blue_end": config["blue_end"],
                    "yellow_start": config["yellow_start"],
                    "yellow_end": config["yellow_end"],
                    "blue_dots": config["blue_dots"],
                    "yellow_dots": config["yellow_dots"],
                    "validated": config.get("validated", True),
            }
            if "breakpoints" in config:
                level_data["breakpoints"] = config["breakpoints"]

            level = Level(
                sprites=[bg_sprite],
                grid_size=(64, 64),
                data=level_data,
                name=f"Level {i + 1}",
            )
            levels.append(level)

        return levels

    def on_set_level(self, level: Level) -> None:
        data = level._data
        self._grid = WitnessGrid(data["cols"], data["rows"])
        self._symmetry = data["symmetry"]
        self._blue_start = tuple(data["blue_start"])
        self._blue_end = tuple(data["blue_end"])
        self._yellow_start = tuple(data["yellow_start"])
        self._yellow_end = tuple(data["yellow_end"])
        self._blue_dots = [tuple(d) for d in data["blue_dots"]]
        self._yellow_dots = [tuple(d) for d in data["yellow_dots"]]
        self._breakpoints = set()
        for bp in data.get("breakpoints", []):
            n1, n2 = tuple(bp[0]), tuple(bp[1])
            self._breakpoints.add((min(n1, n2), max(n1, n2)))
        self._blue_path = [self._blue_start]
        self._yellow_path = [self._yellow_start]

    def step(self) -> None:
        if not self._grid:
            self.complete_action()
            return

        action = self.action.id
        data = self.current_level._data

        if action == GameAction.ACTION5:
            self._check_solution()
        elif action in (GameAction.ACTION1, GameAction.ACTION2,
                        GameAction.ACTION3, GameAction.ACTION4):
            dc, dr = 0, 0
            if action == GameAction.ACTION1: dr = -1
            elif action == GameAction.ACTION2: dr = 1
            elif action == GameAction.ACTION3: dc = -1
            elif action == GameAction.ACTION4: dc = 1

            # Compute blue line target
            blue_current = self._blue_path[-1]
            blue_target = (blue_current[0] + dc, blue_current[1] + dr)

            # Compute yellow line target (symmetric)
            yellow_current = self._yellow_path[-1]
            yellow_delta = self._mirror_delta(dc, dr)
            yellow_target = (yellow_current[0] + yellow_delta[0],
                           yellow_current[1] + yellow_delta[1])

            # Both lines must make valid moves
            cols, rows = data["cols"], data["rows"]
            blue_valid = self._is_valid_node(blue_target, cols, rows)
            yellow_valid = self._is_valid_node(yellow_target, cols, rows)

            # breakpoint check for both paths
            if blue_valid:
                b_edge = (min(blue_current, blue_target), max(blue_current, blue_target))
                if b_edge in self._breakpoints:
                    blue_valid = False
            if yellow_valid:
                y_edge = (min(yellow_current, yellow_target), max(yellow_current, yellow_target))
                if y_edge in self._breakpoints:
                    yellow_valid = False

            if blue_valid and yellow_valid:
                # Check for backtracking
                if (len(self._blue_path) >= 2 and
                    blue_target == self._blue_path[-2]):
                    self._blue_path.pop()
                    self._yellow_path.pop()
                elif (blue_target not in self._blue_path and
                      yellow_target not in self._yellow_path):
                    # The two lines cannot cross onto the same node
                    if blue_target != yellow_target:
                        self._blue_path.append(blue_target)
                        self._yellow_path.append(yellow_target)

                self._update_display()

        self.complete_action()

    def _mirror_delta(self, dc: int, dr: int) -> Tuple[int, int]:
        """Compute the movement delta for the symmetric direction."""
        if self._symmetry == "horizontal":
            return (-dc, dr)
        elif self._symmetry == "vertical":
            return (dc, -dr)
        elif self._symmetry == "rotational":
            return (-dc, -dr)
        return (dc, dr)

    def _is_valid_node(self, node: Tuple[int, int],
                       cols: int, rows: int) -> bool:
        c, r = node
        return 0 <= c <= cols and 0 <= r <= rows

    def _check_solution(self) -> None:
        # Blue line reaches blue endpoint
        if self._blue_path[-1] != self._blue_end:
            self._reset_paths()
            return
        # Yellow line reaches yellow endpoint
        if self._yellow_path[-1] != self._yellow_end:
            self._reset_paths()
            return
        # Blue mandatory waypoints
        blue_set = set(self._blue_path)
        for dot in self._blue_dots:
            if dot not in blue_set:
                self._reset_paths()
                return
        # Yellow mandatory waypoints
        yellow_set = set(self._yellow_path)
        for dot in self._yellow_dots:
            if dot not in yellow_set:
                self._reset_paths()
                return

        self._update_display(path_color=SUCCESS_COLOR)
        self.next_level()

    def _reset_paths(self):
        self._blue_path = [self._blue_start]
        self._yellow_path = [self._yellow_start]
        self._update_display()

    def _update_display(self, path_color: int = PATH_COLOR) -> None:
        if not self._grid:
            return

        data = self.current_level._data
        frame = self._grid.render_grid()

        # Start and end points
        self._grid.draw_dot(frame, self._blue_start, COLOR_BLUE)
        self._grid.draw_dot(frame, self._blue_end, COLOR_BLUE)
        self._grid.draw_dot(frame, self._yellow_start, COLOR_YELLOW)
        self._grid.draw_dot(frame, self._yellow_end, COLOR_YELLOW)

        # Mandatory waypoints
        for dot in self._blue_dots:
            self._grid.draw_dot(frame, dot, COLOR_BLUE)
        for dot in self._yellow_dots:
            self._grid.draw_dot(frame, dot, COLOR_YELLOW)

        # Draw broken edges
        for bp in self._breakpoints:
            self._grid.draw_breakpoint(frame, bp[0], bp[1])

        # Blue path
        for i in range(len(self._blue_path) - 1):
            self._grid.draw_path_segment(
                frame, self._blue_path[i], self._blue_path[i + 1], COLOR_BLUE)

        # Yellow path
        for i in range(len(self._yellow_path) - 1):
            self._grid.draw_path_segment(
                frame, self._yellow_path[i], self._yellow_path[i + 1], COLOR_YELLOW)

        # Cursors
        if self._blue_path:
            self._grid.draw_dot(frame, self._blue_path[-1], COLOR_BLUE)
        if self._yellow_path:
            self._grid.draw_dot(frame, self._yellow_path[-1], COLOR_YELLOW)

        # Unvalidated indicator
        if not self.current_level._data.get("validated", True):
            self._grid.draw_unvalidated_indicator(frame)

        bg_sprites = self.current_level.get_sprites_by_name("grid_bg")
        if bg_sprites:
            self.current_level.remove_sprite(bg_sprites[0])

        new_bg = Sprite(
            pixels=frame,
            name="grid_bg",
            x=0, y=0, layer=-10,
            blocking=BlockingMode.NOT_BLOCKED,
            interaction=InteractionMode.INTANGIBLE,
            tags=["sys_static"],
        )
        self.current_level.add_sprite(new_bg)
