"""
witness_grid.py — Shared grid rendering utility for The Witness-style panel puzzles

All Witness-inspired ARC-AGI-3 games share this utility class to render panel grids.

Core concepts:
- Render an N×M grid panel within a 64×64 pixel space
- The grid consists of "nodes" (line intersections) and "edges" (lines connecting nodes)
- Paths walk along grid edges
- Symbols are placed at cell centers

Color indices (16-color palette):
  0=white, 1=light_gray, 2=gray, 3=dark_gray, 4=near_black, 5=black,
  6=magenta, 7=light_magenta, 8=red, 9=blue, 10=light_blue, 11=yellow,
  12=orange, 13=maroon, 14=green, 15=purple
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from arcengine import Sprite


# === Color constants ===
COLOR_WHITE = 0
COLOR_LIGHT_GRAY = 1
COLOR_GRAY = 2
COLOR_DARK_GRAY = 3
COLOR_NEAR_BLACK = 4
COLOR_BLACK = 5
COLOR_MAGENTA = 6
COLOR_LIGHT_MAGENTA = 7
COLOR_RED = 8
COLOR_BLUE = 9
COLOR_LIGHT_BLUE = 10
COLOR_YELLOW = 11
COLOR_ORANGE = 12
COLOR_MAROON = 13
COLOR_GREEN = 14
COLOR_PURPLE = 15

# === Semantic colors ===
GRID_BG = COLOR_DARK_GRAY      # Grid background (panel base color)
GRID_LINE = COLOR_BLACK        # Grid lines (black, high contrast against dark gray background)
PATH_COLOR = COLOR_BLUE        # Drawn path
CURSOR_COLOR = COLOR_YELLOW    # Cursor / active position
START_COLOR = COLOR_GREEN      # Start point
END_COLOR = COLOR_RED          # End point
DOT_COLOR = COLOR_YELLOW       # Hexagon dot
ERROR_COLOR = COLOR_RED        # Error feedback
SUCCESS_COLOR = COLOR_GREEN    # Success feedback
CELL_BG = COLOR_WHITE          # Cell background
SQUARE_A = COLOR_MAGENTA       # Colored square A
SQUARE_B = COLOR_LIGHT_BLUE    # Colored square B
SQUARE_C = COLOR_ORANGE        # Colored square C
STAR_COLOR = COLOR_YELLOW      # Star
POLY_COLOR = COLOR_PURPLE      # Polyomino
TRI_COLOR = COLOR_ORANGE       # Triangle
ERASER_COLOR = COLOR_NEAR_BLACK  # Elimination symbol (Y-shape, must contrast with white cell bg)


class WitnessGrid:
    """The Witness-style grid panel.

    Grid coordinate system:
    - Node coordinates: (col, row), range [0, cols] x [0, rows]
    - Edge: a line connecting two adjacent nodes
    - Cell coordinates: (col, row), range [0, cols-1] x [0, rows-1]

    Rendering to 64x64 pixel space:
    - Outer margin: margin pixels
    - Node: node_size x node_size pixels
    - Edge: edge_length x line_width pixels
    - Cell: cell_size x cell_size pixels
    """

    def __init__(self, cols: int, rows: int, margin: int = 4):
        """
        Args:
            cols: Number of grid columns (cell count)
            rows: Number of grid rows (cell count)
            margin: Outer margin in pixels
        """
        self.cols = cols
        self.rows = rows
        self.margin = margin

        # Calculate pixel dimensions
        # Available space = 64 - 2*margin
        avail = 64 - 2 * margin
        # Use the larger dimension to calculate cell_size, ensuring both directions fit
        self.node_size = 1
        self.line_width = 1
        max_dim = max(cols, rows)
        self.cell_size = (avail - (max_dim + 1) * self.node_size) // max_dim

        # Check if it fits
        total_w = (cols + 1) * self.node_size + cols * self.cell_size
        total_h = (rows + 1) * self.node_size + rows * self.cell_size
        assert total_w <= avail, f"Grid too wide: {total_w} > {avail}"
        assert total_h <= avail, f"Grid too tall: {total_h} > {avail}"

        # Centering offset
        self.offset_x = margin + (avail - total_w) // 2
        self.offset_y = margin + (avail - total_h) // 2

    def node_to_pixel(self, col: int, row: int) -> Tuple[int, int]:
        """Convert node coordinates to pixel coordinates (top-left corner)."""
        px = self.offset_x + col * (self.node_size + self.cell_size)
        py = self.offset_y + row * (self.node_size + self.cell_size)
        return (px, py)

    def cell_center_pixel(self, col: int, row: int) -> Tuple[int, int]:
        """Convert cell coordinates to center pixel coordinates."""
        px = self.offset_x + col * (self.node_size + self.cell_size) + self.node_size + self.cell_size // 2
        py = self.offset_y + row * (self.node_size + self.cell_size) + self.node_size + self.cell_size // 2
        return (px, py)

    def render_grid(self) -> List[List[int]]:
        """Render an empty grid as a 64x64 pixel array.

        Returns:
            64x64 int array (color indices)
        """
        frame = [[GRID_BG] * 64 for _ in range(64)]

        # Draw grid lines (nodes + connections)
        for row in range(self.rows + 1):
            for col in range(self.cols + 1):
                # Draw node
                nx, ny = self.node_to_pixel(col, row)
                if 0 <= nx < 64 and 0 <= ny < 64:
                    frame[ny][nx] = GRID_LINE

                # Draw horizontal edge (right of node)
                if col < self.cols:
                    for dx in range(1, self.cell_size + 1):
                        px = nx + dx
                        if 0 <= px < 64 and 0 <= ny < 64:
                            frame[ny][px] = GRID_LINE

                # Draw vertical edge (below node)
                if row < self.rows:
                    for dy in range(1, self.cell_size + 1):
                        py = ny + dy
                        if 0 <= nx < 64 and 0 <= py < 64:
                            frame[py][nx] = GRID_LINE

        # Fill cell backgrounds
        for row in range(self.rows):
            for col in range(self.cols):
                cx = self.offset_x + col * (self.node_size + self.cell_size) + self.node_size
                cy = self.offset_y + row * (self.node_size + self.cell_size) + self.node_size
                for dy in range(self.cell_size):
                    for dx in range(self.cell_size):
                        px, py = cx + dx, cy + dy
                        if 0 <= px < 64 and 0 <= py < 64:
                            frame[py][px] = CELL_BG

        return frame

    def render_to_sprite(self, extra_pixels: Optional[List[Tuple[int, int, int]]] = None) -> Sprite:
        """Render the grid as a Sprite object.

        Args:
            extra_pixels: List of extra pixels [(x, y, color), ...]

        Returns:
            64x64 Sprite object
        """
        frame = self.render_grid()

        if extra_pixels:
            for x, y, color in extra_pixels:
                if 0 <= x < 64 and 0 <= y < 64:
                    frame[y][x] = color

        return Sprite(
            pixels=frame,
            name="grid_bg",
            x=0, y=0,
            layer=-10,
            tags=["sys_static"],
        )

    def draw_path_segment(self, frame: List[List[int]],
                          node1: Tuple[int, int], node2: Tuple[int, int],
                          color: int = PATH_COLOR):
        """Draw a path segment between two adjacent nodes on the frame."""
        x1, y1 = self.node_to_pixel(*node1)
        x2, y2 = self.node_to_pixel(*node2)

        if x1 == x2:  # Vertical
            for y in range(min(y1, y2), max(y1, y2) + 1):
                if 0 <= x1 < 64 and 0 <= y < 64:
                    frame[y][x1] = color
        elif y1 == y2:  # Horizontal
            for x in range(min(x1, x2), max(x1, x2) + 1):
                if 0 <= x < 64 and 0 <= y1 < 64:
                    frame[y1][x] = color

    def draw_dot(self, frame: List[List[int]],
                 node: Tuple[int, int], color: int = DOT_COLOR):
        """Draw a dot marker at a node position."""
        nx, ny = self.node_to_pixel(*node)
        # Draw a 3x3 cross shape (if space allows)
        for dx, dy in [(-1, 0), (0, -1), (0, 0), (1, 0), (0, 1)]:
            px, py = nx + dx, ny + dy
            if 0 <= px < 64 and 0 <= py < 64:
                frame[py][px] = color

    def draw_start(self, frame: List[List[int]], node: Tuple[int, int]):
        """Draw a start point marker."""
        self.draw_dot(frame, node, START_COLOR)

    def draw_end(self, frame: List[List[int]], node: Tuple[int, int]):
        """Draw an end point marker."""
        self.draw_dot(frame, node, END_COLOR)

    def draw_cell_symbol(self, frame: List[List[int]],
                         cell: Tuple[int, int], color: int,
                         size: int = 3):
        """Draw a square symbol at the cell center."""
        cx, cy = self.cell_center_pixel(*cell)
        half = size // 2
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                px, py = cx + dx, cy + dy
                if 0 <= px < 64 and 0 <= py < 64:
                    frame[py][px] = color

    def get_adjacent_nodes(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get all adjacent nodes for a given node."""
        col, row = node
        neighbors = []
        for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nc, nr = col + dc, row + dr
            if 0 <= nc <= self.cols and 0 <= nr <= self.rows:
                neighbors.append((nc, nr))
        return neighbors

    def path_to_edges(self, path: List[Tuple[int, int]]) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Convert a path node sequence to a set of edges."""
        edges = set()
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            edges.add((min(a, b), max(a, b)))
        return edges

    def path_splits_regions(self, path: List[Tuple[int, int]]) -> List[Set[Tuple[int, int]]]:
        """Split the panel into regions based on the path.

        Returns:
            List of regions, where each region is a set of cell coordinates
        """
        path_edges = self.path_to_edges(path)

        # Find connected regions using BFS
        visited = set()
        regions = []

        for row in range(self.rows):
            for col in range(self.cols):
                cell = (col, row)
                if cell in visited:
                    continue

                # BFS starting from this cell
                region = set()
                queue = [cell]
                while queue:
                    c = queue.pop(0)
                    if c in visited:
                        continue
                    visited.add(c)
                    region.add(c)

                    cc, cr = c
                    # Check adjacent cells in four directions
                    for dc, dr, edge_n1, edge_n2 in [
                        (-1, 0, (cc, cr), (cc, cr + 1)),    # Left
                        (1, 0, (cc + 1, cr), (cc + 1, cr + 1)),  # Right
                        (0, -1, (cc, cr), (cc + 1, cr)),    # Up
                        (0, 1, (cc, cr + 1), (cc + 1, cr + 1)),  # Down
                    ]:
                        nc, nr = cc + dc, cr + dr
                        if 0 <= nc < self.cols and 0 <= nr < self.rows:
                            edge = (min(edge_n1, edge_n2), max(edge_n1, edge_n2))
                            if edge not in path_edges and (nc, nr) not in visited:
                                queue.append((nc, nr))

                regions.append(region)

        return regions

    def draw_star(self, frame: List[List[int]],
                  cell: Tuple[int, int], color: int = STAR_COLOR):
        """Draw a star symbol (diamond shape) at the cell center, scaled to fit cell_size."""
        cx, cy = self.cell_center_pixel(*cell)
        half = (self.cell_size - 1) // 2
        if half >= 2:
            # Full 5x5 diamond
            offsets = [(-2, 0), (-1, -1), (-1, 0), (-1, 1),
                       (0, -2), (0, -1), (0, 0), (0, 1), (0, 2),
                       (1, -1), (1, 0), (1, 1), (2, 0)]
        else:
            # Small 3x3 cross for tiny cells
            offsets = [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
        for dx, dy in offsets:
            px, py = cx + dx, cy + dy
            if 0 <= px < 64 and 0 <= py < 64:
                frame[py][px] = color

    def draw_triangle(self, frame: List[List[int]],
                      cell: Tuple[int, int], count: int,
                      color: int = TRI_COLOR):
        """Draw 1-3 small triangle markers in a cell, scaled to fit cell_size."""
        cx, cy = self.cell_center_pixel(*cell)
        half = (self.cell_size - 1) // 2

        if count == 1:
            offsets = [0]
        elif count == 2:
            # Leave room for triangle body (±1) on each side
            step = max(1, min(2, half - 1))
            offsets = [-step, step]
        else:  # count == 3
            step = max(1, min(3, half - 1))
            offsets = [-step, 0, step]

        for ox in offsets[:count]:
            px_base = cx + ox
            # Per-triangle half-width: clamp so it stays within cell
            tri_hw = min(1, half - abs(ox))
            tri_hw = max(0, tri_hw)
            # Tip pixel (top of triangle)
            if 0 <= px_base < 64 and 0 <= cy - 1 < 64:
                frame[cy - 1][px_base] = color
            # Base row
            for dx in range(-tri_hw, tri_hw + 1):
                ppx, ppy = px_base + dx, cy
                if 0 <= ppx < 64 and 0 <= ppy < 64:
                    frame[ppy][ppx] = color

    def draw_polyomino(self, frame: List[List[int]],
                       cell: Tuple[int, int], shape: list,
                       color: int = POLY_COLOR):
        """Draw a polyomino shape preview in a cell, centered and scaled to fit cell_size."""
        cx, cy = self.cell_center_pixel(*cell)
        half = (self.cell_size - 1) // 2

        # Shape extent (handle non-zero-based shapes)
        xs = [s[0] for s in shape]
        ys = [s[1] for s in shape]
        min_sx, max_sx = min(xs), max(xs)
        min_sy, max_sy = min(ys), max(ys)
        extent_x = max_sx - min_sx + 1
        extent_y = max_sy - min_sy + 1
        extent = max(extent_x, extent_y)

        # Block size: each polyomino cell rendered as block×block pixels
        # Must fit: extent * block <= 2 * half + 1
        block = min(2, (2 * half + 1) // extent) if extent > 0 else 2
        block = max(1, block)

        # Center the shape within the cell
        total_w = extent_x * block
        total_h = extent_y * block
        start_x = cx - total_w // 2
        start_y = cy - total_h // 2

        for sx, sy in shape:
            for dx in range(block):
                for dy in range(block):
                    px = start_x + (sx - min_sx) * block + dx
                    py = start_y + (sy - min_sy) * block + dy
                    if 0 <= px < 64 and 0 <= py < 64:
                        frame[py][px] = color

    def draw_eraser(self, frame: List[List[int]],
                    cell: Tuple[int, int], color: int = ERASER_COLOR):
        """Draw an elimination symbol (Y shape) at the cell center, scaled to fit cell_size."""
        cx, cy = self.cell_center_pixel(*cell)
        half = (self.cell_size - 1) // 2
        if half >= 2:
            # Full Y shape: stem + two branches
            offsets = [(0, 0), (0, 1), (0, 2), (-1, -1), (1, -1), (-2, -2), (2, -2)]
        elif half >= 1:
            # Small Y: 3x3
            offsets = [(0, 0), (0, 1), (-1, -1), (1, -1)]
        else:
            # Minimal: single pixel
            offsets = [(0, 0)]
        for dx, dy in offsets:
            px, py = cx + dx, cy + dy
            if 0 <= px < 64 and 0 <= py < 64:
                frame[py][px] = color

    def draw_unvalidated_indicator(self, frame: List[List[int]]) -> None:
        """Draw an orange '?' marker in the top-right corner of the frame, indicating this level has not been validated by the solver."""
        # Draw a 5x5 pixel '?' shape at position (58,2)
        COLOR = 12  # COLOR_ORANGE
        # ? pattern:
        #  .XX.
        #  ...X
        #  ..X.
        #  ....
        #  ..X.
        pixels = [
            (59, 2), (60, 2),
            (61, 3),
            (60, 4),
            (60, 6),
        ]
        for px, py in pixels:
            if 0 <= px < 64 and 0 <= py < 64:
                frame[py][px] = COLOR

    def draw_breakpoint(self, frame: List[List[int]],
                        node1: Tuple[int, int], node2: Tuple[int, int]):
        """Draw a gap (broken edge) on the grid line between two adjacent nodes.

        Erase ~60% of pixels in the middle of the edge, keep short stubs at both ends, and fill the gap with GRID_BG.
        """
        x1, y1 = self.node_to_pixel(*node1)
        x2, y2 = self.node_to_pixel(*node2)

        if x1 == x2:  # Vertical edge
            lo, hi = min(y1, y2), max(y1, y2)
            length = hi - lo
            gap_start = lo + max(1, length * 2 // 10)
            gap_end = hi - max(1, length * 2 // 10)
            for y in range(gap_start, gap_end + 1):
                if 0 <= x1 < 64 and 0 <= y < 64:
                    frame[y][x1] = GRID_BG
        elif y1 == y2:  # Horizontal edge
            lo, hi = min(x1, x2), max(x1, x2)
            length = hi - lo
            gap_start = lo + max(1, length * 2 // 10)
            gap_end = hi - max(1, length * 2 // 10)
            for x in range(gap_start, gap_end + 1):
                if 0 <= x < 64 and 0 <= y1 < 64:
                    frame[y1][x] = GRID_BG

    def cell_edge_count(self, cell: Tuple[int, int],
                        path_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]]) -> int:
        """Count the number of cell boundary edges traversed by the path."""
        col, row = cell
        edges = [
            ((col, row), (col + 1, row)),       # Top
            ((col, row + 1), (col + 1, row + 1)),  # Bottom
            ((col, row), (col, row + 1)),         # Left
            ((col + 1, row), (col + 1, row + 1)),  # Right
        ]
        count = 0
        for e in edges:
            normalized = (min(e[0], e[1]), max(e[0], e[1]))
            if normalized in path_edges:
                count += 1
        return count
