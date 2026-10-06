"""Engine-free serialization shared by goal probing and hypothesizer B inference."""
from __future__ import annotations
import numpy as np

COLOR_NAMES = {
    0: "white",
    1: "light_gray",
    2: "gray",
    3: "dark_gray",
    4: "charcoal",
    5: "black",
    6: "magenta",
    7: "pink",
    8: "red",
    9: "blue",
    10: "cyan",
    11: "yellow",
    12: "orange",
    13: "maroon",
    14: "green",
    15: "purple",
}


def color_name(idx: int) -> str:
    return COLOR_NAMES.get(int(idx), f"color{int(idx)}")


HEX = "0123456789abcdef"


DIFF_FULL_MAX = 30           # <= this many changed cells -> full coord list; else count+bbox


DIFFS_PER_LINE = 4           # full-list layout: transitions per output line


def header(avail: list[int]) -> str:
    # The game is ANONYMOUS: its name never appears in the model input (the hypothesizer
    # must infer the goal from evidence, not recall it from a memorised game id). Grid
    # size / value range / palette are constants of the whole benchmark; the ONE thing
    # that varies per game is the valid-action subset, reported by the env itself.
    palette = " ".join(f"{i} {COLOR_NAMES[i]}" for i in sorted(COLOR_NAMES))
    names = []
    for a in sorted(avail):
        if a == 0:
            names.append("RESET")
        elif a == 6:
            names.append("ACTION6 = mouse click at (row,col)")
        else:
            names.append(f"ACTION{a}")
    return (f"grid 64x64, ints 0-15 | coords (row,col), row 0 = top\n"
            f"palette: {palette}\n"
            f"valid actions for this game: {'; '.join(names)}")


def hex_rows(grid: np.ndarray) -> str:
    g = np.clip(grid, 0, 15).astype(int)
    return "\n".join("".join(HEX[v] for v in row) for row in g)


def diff_text(prev: np.ndarray, cur: np.ndarray) -> str:
    """Cell-exact diff, (row, col) convention. Array comparison only -- the diff is never
    grouped into or attributed to objects."""
    rc = np.argwhere(prev != cur)
    n = len(rc)
    if n == 0:
        return " no cells changed"
    if n > DIFF_FULL_MAX:
        (r0, c0), (r1, c1) = rc.min(axis=0), rc.max(axis=0)
        return f" {n} cells changed; bbox rows {r0}..{r1} cols {c0}..{c1}"
    parts = [f"({r},{c}) {color_name(prev[r, c])}->{color_name(cur[r, c])}" for r, c in rc]
    lines = ["  ".join(parts[i:i + DIFFS_PER_LINE]) for i in range(0, n, DIFFS_PER_LINE)]
    return "\n".join(" " + ln for ln in lines)


TASK_SPEC = (
    "Write the Python function:\n"
    "    def is_win(frame0: np.ndarray, current_state: np.ndarray,\n"
    "               action: dict | None, next_state: np.ndarray) -> bool\n"
    "All grids are 64x64 ints 0-15. `frame0` is the initial frame above;\n"
    "`current_state` is the frame before the action; `action` is either None or\n"
    "`{'id': int, 'data': dict}`; and `next_state` is the frame after that action.\n"
    "The states are exactly ONE action apart. Return True exactly when this\n"
    "transition wins. Use `frame0` for static layout that may later be occluded,\n"
    "and use `current_state`, `action`, and `next_state` wherever the game's real\n"
    "win rule requires them. numpy is available as np. The frame-only `goal_verifier`\n"
    "library is already available as `gv`; do not import it. Reuse `gv.<method>`\n"
    "whenever it covers generic component, colour, mask, or transition logic."
)


def build_sequence(frames, labels, status, avail) -> str:
    out = [header(avail), "", "FRAME 0 (initial):", hex_rows(frames[0]), ""]
    for i, lab in enumerate(labels, start=1):
        out.append(f"STEP {i}  {lab}")
        out.append(diff_text(frames[i - 1], frames[i]))
        out.append("")
    if status:
        out.append(f"[episode ended: {status}]")
        out.append("")
    out.append(TASK_SPEC)
    return "\n".join(out)

