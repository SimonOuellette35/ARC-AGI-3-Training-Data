"""Frame-only primitives shared by generated ``is_win`` programs.

The functions in this module deliberately depend only on NumPy and their frame
arguments.  ``goal_generators.test_goals`` exposes the module to verifier blocks
as ``gv``; generated programs must not import it themselves.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return four-connected components of a boolean 2-D mask as ``(row, col)`` arrays."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("connected_components expects a 2-D mask")
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    rows, cols = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue = deque([(r, c)])
        cells = []
        while queue:
            y, x = queue.popleft()
            cells.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        components.append(np.asarray(cells, dtype=int))
    return components


def component_bbox(component: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inclusive-exclusive ``(r0, r1, c0, c1)`` bounding box."""
    cells = np.asarray(component, dtype=int)
    if cells.size == 0:
        raise ValueError("component_bbox needs at least one cell")
    return (int(cells[:, 0].min()), int(cells[:, 0].max()) + 1,
            int(cells[:, 1].min()), int(cells[:, 1].max()) + 1)


def is_solid_rect(component: np.ndarray) -> bool:
    """Whether cells fill their bounding rectangle without holes."""
    r0, r1, c0, c1 = component_bbox(component)
    return len(component) == (r1 - r0) * (c1 - c0)


def component_stats(frame: np.ndarray, color: int | None = None) -> list[dict[str, int | bool]]:
    """Describe four-connected same-colour components, optionally for one colour."""
    frame = np.asarray(frame)
    colors = [int(color)] if color is not None else [int(v) for v in np.unique(frame)]
    out = []
    for value in colors:
        for cells in connected_components(frame == value):
            r0, r1, c0, c1 = component_bbox(cells)
            out.append({"color": value, "area": len(cells), "r0": r0, "r1": r1,
                        "c0": c0, "c1": c1, "h": r1 - r0, "w": c1 - c0,
                        "solid": is_solid_rect(cells)})
    return out


def touches_color(frame: np.ndarray, component: np.ndarray, color: int) -> bool:
    """Whether any four-neighbour of a component has ``color``."""
    frame = np.asarray(frame)
    rows, cols = frame.shape
    for r, c in np.asarray(component, dtype=int):
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and frame[nr, nc] == color:
                return True
    return False


def color_counts(frame: np.ndarray) -> dict[int, int]:
    """Map every visible palette value to its pixel count."""
    values, counts = np.unique(np.asarray(frame), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def rarest_colors(frame: np.ndarray, n: int = 1, *, exclude: tuple[int, ...] = ()) -> list[int]:
    """Return up to ``n`` visible colours sorted by count, then colour value."""
    excluded = set(exclude)
    return [color for color, _ in sorted(
        ((color, count) for color, count in color_counts(frame).items() if color not in excluded),
        key=lambda item: (item[1], item[0]))[:n]]


def dominant_color(frame: np.ndarray, *, exclude: tuple[int, ...] = ()) -> int | None:
    """Return the most frequent non-excluded colour, or ``None`` for none."""
    excluded = set(exclude)
    choices = [(count, color) for color, count in color_counts(frame).items() if color not in excluded]
    return max(choices)[1] if choices else None


def unique_color_with_pixel_count(frame: np.ndarray, count: int) -> int | None:
    """Return the unique colour with exactly ``count`` pixels, else ``None``."""
    matches = [color for color, observed in color_counts(frame).items() if observed == count]
    return matches[0] if len(matches) == 1 else None


def centroid_of_color(frame: np.ndarray, color: int) -> tuple[float, float] | None:
    """Return the ``(row, col)`` centroid of a colour, or ``None`` if absent."""
    cells = np.argwhere(np.asarray(frame) == color)
    if not len(cells):
        return None
    return (float(cells[:, 0].mean()), float(cells[:, 1].mean()))


def non_letterbox_bbox(frame: np.ndarray, background: int | None = None) -> tuple[int, int, int, int]:
    """Bounding box of rendered content; the dominant colour is padding by default."""
    frame = np.asarray(frame)
    if background is None:
        background = dominant_color(frame)
    cells = np.argwhere(frame != background)
    if not len(cells):
        return (0, 0, 0, 0)
    return component_bbox(cells)


def infer_grid_geometry(frame: np.ndarray, background: int | None = None) -> tuple[int, int, int, int, int]:
    """Infer ``(r0, r1, c0, c1, cell_size)`` from a letterboxed block grid."""
    r0, r1, c0, c1 = non_letterbox_bbox(frame, background)
    if r1 == r0 or c1 == c0:
        return r0, r1, c0, c1, 1
    runs = []
    for line in (np.asarray(frame)[r0:r1, c0], np.asarray(frame)[r0, c0:c1]):
        changes = np.flatnonzero(line[1:] != line[:-1]) + 1
        edges = np.r_[0, changes, len(line)]
        runs.extend(np.diff(edges).tolist())
    cell_size = int(np.gcd.reduce(np.asarray([run for run in runs if run > 0], dtype=int))) if runs else 1
    return r0, r1, c0, c1, max(1, cell_size)


def sample_logical_cell(frame: np.ndarray, row: int, col: int, geometry: tuple[int, int, int, int, int]) -> np.ndarray:
    """Return pixels for logical grid cell ``(row, col)`` in inferred geometry."""
    r0, r1, c0, c1, size = geometry
    y0, x0 = r0 + row * size, c0 + col * size
    return np.asarray(frame)[y0:min(y0 + size, r1), x0:min(x0 + size, c1)]


def decode_rotated_grid(frame: np.ndarray, grid_size: int, rotation: int) -> np.ndarray:
    """Undo a quarter-turn display rotation and sample its uniformly scaled grid.

    ARC's standard camera centres the largest integer scale of a logical square
    grid in the 64x64 display. ``rotation`` uses ``numpy.rot90`` convention.
    """
    frame = np.asarray(frame)
    if frame.shape != (64, 64) or grid_size <= 0:
        raise ValueError("decode_rotated_grid expects a 64x64 frame and positive grid size")
    scale = 64 // grid_size
    if scale <= 0:
        raise ValueError("grid_size must not exceed 64")
    offset = (64 - grid_size * scale) // 2
    raw = np.rot90(frame, -rotation)
    return raw[offset:offset + grid_size * scale:scale,
               offset:offset + grid_size * scale:scale]


def labeled_uniform_shapes(grid: np.ndarray, shapes: list | tuple) -> np.ndarray | None:
    """Label fixed shapes whose filled cells are each one non-hole colour.

    ``shapes`` is ``[((x, y), [[0|1, ...], ...]), ...]``. A shape's filled
    cells must share a value distinct from every declared hole; the returned
    grid contains its input-order label at filled cells and ``-1`` elsewhere.
    ``None`` means a shape did not match or lies outside the grid.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError("labeled_uniform_shapes expects a 2-D grid")
    labels = np.full(grid.shape, -1, dtype=int)
    rows, cols = grid.shape
    for label, ((x, y), shape) in enumerate(shapes):
        filled, holes = [], []
        for dy, row in enumerate(shape):
            for dx, cell in enumerate(row):
                xx, yy = x + dx, y + dy
                if not (0 <= xx < cols and 0 <= yy < rows):
                    return None
                (filled if cell else holes).append(grid[yy, xx])
                if cell:
                    labels[yy, xx] = label
        if not filled or not holes or len(set(map(int, filled))) != 1 or filled[0] == holes[0]:
            return None
    return labels


def any_pixel_transition(current: np.ndarray, predicted: np.ndarray, from_color: int, to_color: int) -> bool:
    """Whether a same-position pixel changed from one specified colour to another."""
    return bool(((np.asarray(current) == from_color) & (np.asarray(predicted) == to_color)).any())


def is_real_transition(current: np.ndarray, predicted: np.ndarray, max_changed_pixels: int | None = None) -> bool:
    """Check that frames differ, optionally bounding the size of the change."""
    changed = int((np.asarray(current) != np.asarray(predicted)).sum())
    return changed > 0 and (max_changed_pixels is None or changed <= max_changed_pixels)


def goal_mask_from_frame(frame0: np.ndarray, goal_color: int) -> np.ndarray:
    """Return the pixels carrying the static goal colour in the opening frame."""
    return np.asarray(frame0) == goal_color


def mask_fully_recolored(frame: np.ndarray, mask: np.ndarray, color: int) -> bool:
    """Whether a non-empty mask is entirely covered by ``color`` in a frame."""
    mask = np.asarray(mask, dtype=bool)
    return bool(mask.any() and np.all(np.asarray(frame)[mask] == color))


def vanished_color(current: np.ndarray, predicted: np.ndarray, color: int) -> bool:
    """Whether a colour visible before an action is entirely absent afterwards."""
    return bool(np.any(np.asarray(current) == color) and not np.any(np.asarray(predicted) == color))


def uniform_overlay_color(frame: np.ndarray, mask: np.ndarray) -> int | None:
    """Return the sole colour covering a non-empty mask, else ``None``."""
    values = np.unique(np.asarray(frame)[np.asarray(mask, dtype=bool)])
    return int(values[0]) if len(values) == 1 else None


def win_by_covering_static_goal(frame0: np.ndarray, predicted: np.ndarray, goal_color: int, occupant_color: int) -> bool:
    """Whether every opening-frame goal pixel is now occupied by one colour."""
    return mask_fully_recolored(predicted, goal_mask_from_frame(frame0, goal_color), occupant_color)


def win_by_goal_color_vanishing(current: np.ndarray, predicted: np.ndarray, goal_color: int) -> bool:
    """Whether the visible goal colour disappeared during this action."""
    return vanished_color(current, predicted, goal_color)


def win_by_transition(current: np.ndarray, predicted: np.ndarray, from_color: int, to_color: int,
                      max_changed_pixels: int | None = None) -> bool:
    """Whether a bounded real transition contains a specified colour transition."""
    return (is_real_transition(current, predicted, max_changed_pixels)
            and any_pixel_transition(current, predicted, from_color, to_color))


def win_by_no_remaining_colors(predicted: np.ndarray, colors: tuple[int, ...] | list[int] | set[int]) -> bool:
    """Whether none of the supplied colours remain visible."""
    return not np.isin(np.asarray(predicted), list(colors)).any()


def win_by_all_pixels_near_target(source: np.ndarray, target: np.ndarray, max_manhattan: int) -> bool:
    """Whether every source pixel lies within Manhattan distance of a target pixel."""
    source_cells = np.argwhere(np.asarray(source, dtype=bool))
    target_cells = np.argwhere(np.asarray(target, dtype=bool))
    if not len(source_cells) or not len(target_cells):
        return False
    distances = np.abs(source_cells[:, None, :] - target_cells[None, :, :]).sum(axis=2)
    return bool(np.all(distances.min(axis=1) <= max_manhattan))
