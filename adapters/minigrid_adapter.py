"""minigrid_adapter.py — Wraps Minigrid environments as ARCBaseGame-compatible objects.

Why Minigrid is a better fit than ALE for ARC-AGI-3 training:
  - Turn-based by design: env waits for discrete action, then returns next state.
    Identical interaction model to ARC-AGI-3 (no frameskip, no timer, no NOOP spam).
  - Grid-native observations: cells are (object_type, color, state) semantic tuples,
    not compressed sprite pixels. Scaling to 64×64 preserves all information.
  - Goal inference required: agent must infer "reach the green goal" / "bring key to
    door" from visual state alone — same cognitive demand as ARC-AGI-3.
  - 70+ environments covering navigation, manipulation, memory, and constraint tasks.

Frame pipeline:
  Minigrid obs (H×W×3 uint8: object_idx, color_idx, state)
  → map each cell to ARC palette index via semantic table
  → nearest-neighbour scale to 64×64
  → 64×64 uint8 ndarray of palette indices

ARC palette mapping (semantic, not pixel):
  Object type determines base color; Minigrid's color field used for multi-color
  objects (keys, balls, boxes, doors) so color-coded puzzles remain solvable.

Action mapping (see _ACTION_MAP for the full rationale):
  GameAction.ACTION1 → forward     (most-used; move in facing direction)
  GameAction.ACTION2 → drop        (place the carried object on the cell ahead)
  GameAction.ACTION3 → turn left
  GameAction.ACTION4 → turn right
  GameAction.ACTION5 → interact    (toggle a door / open a full box / pick up)
  GameAction.ACTION6 → pickup      (legacy; ACTION5 subsumes it)
  GameAction.ACTION7 → reserved by BaseAdapter for UNDO

WIN / GAME_OVER semantics:
  terminated=True AND reward > 0  →  GameState.WIN    (reached goal)
  truncated=True                  →  GameState.GAME_OVER  (step limit exceeded)

Lava handling (discoverability-friendly):
  Stepping into lava does NOT end the game. Instead, the agent is teleported
  back to the starting position and incurs a 5-step penalty to the action
  count. This allows exploration and learning of lava layouts without the
  frustration of instant death.

Level concept:
  set_level(idx) resets with seed = base_seed + idx, giving varied starting
  configurations (random goal/key/agent placement) per "level".

Usage:
    from adapters import MinigridAdapter
    from arcengine import GameAction, ActionInput, GameState

    game = MinigridAdapter("MiniGrid-DoorKey-5x5-v0", seed=0)
    game.set_level(0)
    result = game.perform_action(ActionInput(id=GameAction.ACTION1), raw=True)
    print(result.state)           # GameState.NOT_FINISHED
    print(result.frame[0].shape)  # (64, 64)
"""

from __future__ import annotations

import random as _random_module

import numpy as np
from PIL import Image

import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX

from arcengine import ActionInput, GameAction, GameState, FrameDataRaw

import copy as _copy
from adapters.base import BaseAdapter

# ---------------------------------------------------------------------------
# Semantic mapping: Minigrid cell → ARC palette index
#
# ARC 16-color palette (index → meaning):
#   0  white        7  light gray
#   1  blue         8  red-orange (danger/cursor)
#   2  red          9  blue (player)
#   3  green        10 mid gray
#   4  dark gray    11 yellow
#   5  black        12 orange
#   6  pink/magenta 13 brown
#                   14 green (target/goal)
#                   15 purple
# ---------------------------------------------------------------------------

# Minigrid color index → ARC palette index
# Verified against actual frame_to_rgb_array output:
#   8=(249,60,49)=red  9=(30,147,255)=blue  14=(79,204,48)=green
#   2=(153,153,153)=mid-grey  11=yellow  15=purple
_MINIGRID_COLOR_TO_ARC: dict[int, int] = {
    COLOR_TO_IDX["red"]:    8,   # ARC red      (249, 60, 49)
    COLOR_TO_IDX["green"]:  14,  # ARC green    (79, 204, 48)
    COLOR_TO_IDX["blue"]:   9,   # ARC blue     (30, 147, 255)
    COLOR_TO_IDX["purple"]: 15,  # ARC purple
    COLOR_TO_IDX["yellow"]: 11,  # ARC yellow
    COLOR_TO_IDX["grey"]:   2,   # ARC mid-grey (153, 153, 153)
}

# Fixed-color objects that ignore Minigrid's color field
_FIXED_OBJECT_ARC: dict[int, int] = {
    OBJECT_TO_IDX["unseen"]: 5,   # black — unexplored
    OBJECT_TO_IDX["empty"]:  0,   # white — empty space
    OBJECT_TO_IDX["wall"]:   4,   # dark gray — wall
    OBJECT_TO_IDX["floor"]:  1,   # light grey — floor tile (index 1, not 7)
    OBJECT_TO_IDX["goal"]:   14,  # green (target) — goal cell
    OBJECT_TO_IDX["lava"]:   8,   # red — deadly
    OBJECT_TO_IDX["agent"]:  9,   # blue (player) — agent
}

# Objects that use Minigrid's color field (including doors — same color as matching key)
_COLOR_CODED_OBJECTS: frozenset[int] = frozenset({
    OBJECT_TO_IDX["key"],
    OBJECT_TO_IDX["ball"],
    OBJECT_TO_IDX["box"],
    OBJECT_TO_IDX["door"],
})


def _cell_to_arc(obj_idx: int, color_idx: int, state: int) -> int:
    """Map a single Minigrid cell (object, color, state) → ARC palette index."""
    if obj_idx in _FIXED_OBJECT_ARC:
        return _FIXED_OBJECT_ARC[obj_idx]
    if obj_idx == OBJECT_TO_IDX["door"] and state == STATE_TO_IDX["open"]:
        # Opened doors always render as light-blue, regardless of the door's
        # original color.  Guarantees a visible color change on opening even
        # when cells are too small for _draw_door_shape to run (e.g. the
        # 25×25 MultiRoom grid where cell_px=2).  Light blue is unused
        # elsewhere in the MiniGrid palette so it uniquely signals "opened".
        return 10  # light blue — opened/passable
    if obj_idx in _COLOR_CODED_OBJECTS:
        return _MINIGRID_COLOR_TO_ARC.get(color_idx, 12)  # default: orange
    return 0  # fallback: white


# Precompute full lookup table for all (object, color, state) combinations.
# Avoids per-pixel Python dispatch during frame conversion.
_CELL_LUT: np.ndarray = np.zeros((11, 6, 3), dtype=np.uint8)
for _obj in range(11):
    for _col in range(6):
        for _st in range(3):
            _CELL_LUT[_obj, _col, _st] = _cell_to_arc(_obj, _col, _st)


# ---------------------------------------------------------------------------
# Frame utilities
# ---------------------------------------------------------------------------

def _draw_agent_direction(
    frame: np.ndarray,
    r_center: int,
    c_center: int,
    cell_px: int,
    direction: int,
) -> None:
    """Draw a red directional indicator strictly within the agent cell.

    For cells ≥ 4 px: filled triangle pointing in the facing direction.
    For cells < 4 px: single-pixel leading-edge bar (one row or column of red
    at the facing side) — at 4× display scale this becomes a 4-pixel strip,
    which is small but clearly directional without overflowing into neighbours.

    Uses ARC palette index 8 (249, 60, 49) = red — verified against
    frame_to_rgb_array.

    direction: 0=east(▶)  1=south(▽)  2=west(◀)  3=north(▲)
    r_center, c_center: pixel centre of the agent cell in the 64×64 frame.
    """
    half = cell_px // 2
    r1 = max(0, r_center - half)
    c1 = max(0, c_center - half)
    r2 = min(64, r1 + cell_px)
    c2 = min(64, c1 + cell_px)
    ch, cw = r2 - r1, c2 - c1

    if cell_px >= 4:
        # Filled triangle contained within the cell
        for dr in range(ch):
            for dc in range(cw):
                nr = (dr + 0.5) / ch   # 0 = top,  1 = bottom
                nc = (dc + 0.5) / cw   # 0 = left, 1 = right
                if direction == 3:     # north ▲: apex at top, base at bottom
                    in_tri = abs(nc - 0.5) < nr * 0.45
                elif direction == 1:   # south ▽: apex at bottom, base at top
                    in_tri = abs(nc - 0.5) < (1.0 - nr) * 0.45
                elif direction == 0:   # east  ▶: apex at right, base at left
                    in_tri = abs(nr - 0.5) < (1.0 - nc) * 0.45
                else:                  # west  ◀: apex at left, base at right
                    in_tri = abs(nr - 0.5) < nc * 0.45
                if in_tri:
                    frame[r1 + dr, c1 + dc] = 8   # red
    else:
        # Leading-edge bar: one row/col of red at the facing side of the cell
        if direction == 3:    frame[r1,    c1:c2] = 8   # north: top row
        elif direction == 1:  frame[r2-1,  c1:c2] = 8   # south: bottom row
        elif direction == 0:  frame[r1:r2, c2-1 ] = 8   # east:  right col
        else:                 frame[r1:r2, c1   ] = 8   # west:  left col


def _draw_key_shape(
    frame: np.ndarray,
    r1: int,
    c1: int,
    cell_px: int,
    arc_color: int,
) -> None:
    """Draw a pixel-art key inside the cell at [r1:r1+cell_px, c1:c1+cell_px].

    Layout: black background, rectangular ring (the "bow") in the top half,
    a 1-pixel vertical shaft down the center, and a short horizontal tooth
    near the shaft's bottom. The key color matches the corresponding door color.

    Only called when cell_px >= 3.
    """
    r2 = r1 + cell_px
    c2 = c1 + cell_px

    # Black background so the colored key art stands out
    frame[r1:r2, c1:c2] = 5

    cx = c1 + cell_px // 2  # center column

    # --- Bow (rectangular ring, top ~45% of cell) ---
    bow_size = max(2, int(cell_px * 0.45))
    br1 = r1
    br2 = r1 + bow_size
    bc1 = cx - bow_size // 2
    bc2 = bc1 + bow_size
    # Clamp to cell bounds
    bc1 = max(bc1, c1)
    bc2 = min(bc2, c2)
    for rr in range(br1, min(br2, r2)):
        for cc in range(bc1, bc2):
            on_edge = (rr == br1 or rr == br2 - 1 or
                       cc == bc1 or cc == bc2 - 1)
            if on_edge:
                frame[rr, cc] = arc_color

    # --- Shaft (1px wide, from bow bottom to ~85% of cell height) ---
    shaft_top = br2
    shaft_bot = r1 + max(br2 - r1, int(cell_px * 0.85))
    for rr in range(shaft_top, min(shaft_bot, r2)):
        if c1 <= cx < c2:
            frame[rr, cx] = arc_color

    # --- Tooth (short horizontal notch at ~70% of cell height) ---
    tooth_r = r1 + int(cell_px * 0.70)
    tooth_len = max(1, cell_px // 4)
    if r1 <= tooth_r < r2:
        for cc in range(cx, min(cx + tooth_len + 1, c2)):
            frame[tooth_r, cc] = arc_color


def _draw_door_shape(
    frame: np.ndarray,
    r1: int,
    c1: int,
    cell_px: int,
    arc_color: int,
    door_state: int,
) -> None:
    """Draw a pixel-art door inside the cell at [r1:r1+cell_px, c1:c1+cell_px].

    Visually reads as a gate / passage between rooms:
    Closed (state=1): colored left & right posts + dark grey interior + top bar.
    Locked (state=2): same as closed + red center dot (keyhole).
    Open (state=0):   clearly de-colored — white passage fill with only a thin
                      1-pixel border of the original door color for identity.
                      The swap from saturated colored gate → white-filled frame
                      makes the "opened" state unambiguous at a glance.

    Distinct from boxes (solid colored fill) and balls (circle on black).
    Only called when cell_px >= 3.
    """
    r2 = r1 + cell_px
    c2 = c1 + cell_px

    OPEN   = STATE_TO_IDX["open"]
    LOCKED = STATE_TO_IDX["locked"]

    if door_state == OPEN:
        # Opened door: light-blue passage fill (uniquely signals "opened" — no
        # other MiniGrid object uses light blue) with a thin 1-pixel border in
        # the original door color so the player can still tell which door was
        # opened when multiple colored doors exist in a puzzle.
        frame[r1:r2, c1:c2] = 10          # light blue = opened passage
        frame[r1,      c1:c2] = arc_color  # top edge
        frame[r2 - 1,  c1:c2] = arc_color  # bottom edge
        frame[r1:r2,   c1]    = arc_color  # left edge
        frame[r1:r2,   c2 - 1] = arc_color # right edge
    else:
        # Closed/locked: gate shape — colored vertical posts + top bar, dark center
        frame[r1:r2, c1:c2] = 4          # dark grey fill
        frame[r1:r2, c1] = arc_color      # left post
        frame[r1:r2, c2 - 1] = arc_color  # right post
        frame[r1, c1:c2] = arc_color      # top lintel

        if door_state == LOCKED:
            # Red keyhole dot in center
            kh_r = r1 + cell_px // 2
            kh_c = c1 + cell_px // 2
            if r1 <= kh_r < r2 and c1 <= kh_c < c2:
                frame[kh_r, kh_c] = 8     # red keyhole


def _draw_ball_shape(
    frame: np.ndarray,
    r1: int,
    c1: int,
    cell_px: int,
    arc_color: int,
) -> None:
    """Draw a filled circle (ball) inside the cell.

    Black background, colored filled circle centered in the cell.
    At cell_px=3-4: a diamond approximation. At cell_px>=5: proper circle.
    """
    r2 = r1 + cell_px
    c2 = c1 + cell_px

    # Black background
    frame[r1:r2, c1:c2] = 5

    cy = r1 + cell_px / 2.0
    cx = c1 + cell_px / 2.0
    radius = cell_px * 0.38  # slightly inset from cell edge

    for rr in range(r1, r2):
        for cc in range(c1, c2):
            dist = ((rr + 0.5 - cy) ** 2 + (cc + 0.5 - cx) ** 2) ** 0.5
            if dist <= radius:
                frame[rr, cc] = arc_color


def _draw_box_shape(
    frame: np.ndarray,
    r1: int,
    c1: int,
    cell_px: int,
    arc_color: int,
) -> None:
    """Draw a box (solid colored square with white center mark) inside the cell.

    Solid colored fill with a small white center dot. Visually reads as a
    solid object/container — distinct from doors (vertical bars/gate) and
    balls (circle on black).
    """
    r2 = r1 + cell_px
    c2 = c1 + cell_px

    # Solid colored fill
    frame[r1:r2, c1:c2] = arc_color
    # White center dot
    mid_r = r1 + cell_px // 2
    mid_c = c1 + cell_px // 2
    frame[mid_r, mid_c] = 0


def _draw_goal_shape(
    frame: np.ndarray,
    r1: int,
    c1: int,
    cell_px: int,
) -> None:
    """Draw a green diamond/star on white background to mark the goal cell.

    Distinct from green keys (key shape on black) and green balls (circle on black).
    Uses white (0) background + green (14) diamond.
    """
    r2 = r1 + cell_px
    c2 = c1 + cell_px

    # White background
    frame[r1:r2, c1:c2] = 0

    cy = cell_px / 2.0
    cx = cell_px / 2.0
    radius = cell_px * 0.42

    for dr in range(cell_px):
        for dc in range(cell_px):
            # Diamond: |dx| + |dy| <= radius  (L1 / Manhattan distance)
            dist = abs(dr + 0.5 - cy) + abs(dc + 0.5 - cx)
            if dist <= radius:
                frame[r1 + dr, c1 + dc] = 14  # green


def _grid_to_arc_frame(obs_image: np.ndarray, partial_obs: bool = False) -> np.ndarray:
    """Convert Minigrid (W, H, 3) cell observation to (64, 64) ARC palette frame.

    FullyObsWrapper returns obs["image"] as (width, height, 3) — column-first.
    Transpose to (H, W, 3) = (row, col, 3) first so PIL renders correctly.

    Uses the largest integer cell size that fits in 64×64 to avoid uneven
    NEAREST-scaling artefacts (e.g. a 19×19 grid at non-integer scale gives
    cells with alternating 3-px and 4-px widths, which looks noisy). The
    rendered grid is center-padded to 64×64 with a black border.
    """
    obs_image = obs_image.transpose(1, 0, 2)   # (W,H,3) → (H,W,3) = (row,col,3)
    H, W, _ = obs_image.shape

    obj   = np.clip(obs_image[:, :, 0], 0, 10)
    color = np.clip(obs_image[:, :, 1], 0, 5)
    state = np.clip(obs_image[:, :, 2], 0, 2)
    arc_grid = _CELL_LUT[obj, color, state]

    # Integer cell size → uniform cells, no aliasing
    cell_px = max(1, 64 // max(H, W))
    rh, rw  = H * cell_px, W * cell_px   # rendered size, always ≤ 64×64

    img = Image.fromarray(arc_grid, mode="L")
    img = img.resize((rw, rh), Image.NEAREST)   # PIL resize takes (width, height)

    # Center-pad to 64×64 with black (palette 5)
    pad_r = (64 - rh) // 2
    pad_c = (64 - rw) // 2
    frame = np.full((64, 64), 5, dtype=np.uint8)
    frame[pad_r:pad_r + rh, pad_c:pad_c + rw] = np.asarray(img, dtype=np.uint8)

    # Draw pixel-art shapes for all interactive objects (cells >= 3px)
    if cell_px >= 3:
        KEY_IDX  = OBJECT_TO_IDX["key"]
        DOOR_IDX = OBJECT_TO_IDX["door"]
        BALL_IDX = OBJECT_TO_IDX["ball"]
        BOX_IDX  = OBJECT_TO_IDX["box"]
        GOAL_IDX = OBJECT_TO_IDX["goal"]

        for gr in range(H):
            for gc in range(W):
                o = int(obj[gr, gc])
                pr = pad_r + gr * cell_px
                pc = pad_c + gc * cell_px
                if o == KEY_IDX:
                    ac = _MINIGRID_COLOR_TO_ARC.get(int(color[gr, gc]), 12)
                    _draw_key_shape(frame, pr, pc, cell_px, ac)
                elif o == DOOR_IDX:
                    ac = _MINIGRID_COLOR_TO_ARC.get(int(color[gr, gc]), 12)
                    _draw_door_shape(frame, pr, pc, cell_px, ac, int(state[gr, gc]))
                elif o == BALL_IDX:
                    ac = _MINIGRID_COLOR_TO_ARC.get(int(color[gr, gc]), 12)
                    _draw_ball_shape(frame, pr, pc, cell_px, ac)
                elif o == BOX_IDX:
                    ac = _MINIGRID_COLOR_TO_ARC.get(int(color[gr, gc]), 12)
                    _draw_box_shape(frame, pr, pc, cell_px, ac)
                elif o == GOAL_IDX:
                    _draw_goal_shape(frame, pr, pc, cell_px)

    # Overlay directional triangle on the agent cell
    agent_locs = np.argwhere(obs_image[:, :, 0] == OBJECT_TO_IDX["agent"])
    if len(agent_locs) > 0:
        gr = int(agent_locs[0, 0])
        gc = int(agent_locs[0, 1])
        direction = int(obs_image[gr, gc, 2])   # state field encodes agent_dir
        ar1 = pad_r + gr * cell_px
        ac1 = pad_c + gc * cell_px

        r_center = ar1 + cell_px // 2
        c_center = ac1 + cell_px // 2
        _draw_agent_direction(frame, r_center, c_center, cell_px, direction)
    elif partial_obs:
        # In egocentric (partial) mode the agent is always at the bottom-center
        # of the 7×7 view and is the observer — it doesn't appear as an object
        # in the image.  Draw a red upward triangle there so the player can see
        # where they are standing.
        gr, gc = H - 1, W // 2
        r_center = pad_r + gr * cell_px + cell_px // 2
        c_center = pad_c + gc * cell_px + cell_px // 2
        _draw_agent_direction(frame, r_center, c_center, cell_px, direction=3)  # 3 = north ▲

    return frame


# ---------------------------------------------------------------------------
# Available Minigrid environments (non-BabyAI, non-WFC)
# ---------------------------------------------------------------------------

_AVAILABLE_MINIGRID_ENVS: tuple[str, ...] = (
    # Navigation — empty rooms
    "MiniGrid-Empty-5x5-v0",
    "MiniGrid-Empty-6x6-v0",
    "MiniGrid-Empty-8x8-v0",
    "MiniGrid-Empty-16x16-v0",
    "MiniGrid-Empty-Random-5x5-v0",
    "MiniGrid-Empty-Random-6x6-v0",
    # Navigation — multi-room
    "MiniGrid-FourRooms-v0",
    "MiniGrid-MultiRoom-N2-S4-v0",
    "MiniGrid-MultiRoom-N4-S5-v0",
    "MiniGrid-MultiRoom-N6-v0",
    # Key + Door puzzles
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-DoorKey-6x6-v0",
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-DoorKey-16x16-v0",
    "MiniGrid-Unlock-v0",
    "MiniGrid-UnlockPickup-v0",
    "MiniGrid-BlockedUnlockPickup-v0",
    "MiniGrid-KeyCorridorS3R1-v0",
    "MiniGrid-KeyCorridorS3R2-v0",
    "MiniGrid-KeyCorridorS3R3-v0",
    "MiniGrid-KeyCorridorS4R3-v0",
    "MiniGrid-KeyCorridorS5R3-v0",
    "MiniGrid-KeyCorridorS6R3-v0",
    "MiniGrid-LockedRoom-v0",
    # NOTE: Fetch / PutNear / GoToObject / GoToDoor families are excluded
    # because their target is specified only via a natural-language `mission`
    # string ("go get a blue key"), with no visual indicator on the grid.
    # They are unsolvable from pixels alone and therefore don't fit the
    # ARC-AGI-3 "infer the goal visually" premise.
    # Obstacle / lava navigation
    "MiniGrid-LavaGapS5-v0",
    "MiniGrid-LavaGapS6-v0",
    "MiniGrid-LavaGapS7-v0",
    "MiniGrid-SimpleCrossingS9N1-v0",
    "MiniGrid-SimpleCrossingS9N2-v0",
    "MiniGrid-SimpleCrossingS9N3-v0",
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-LavaCrossingS9N2-v0",
    "MiniGrid-LavaCrossingS9N3-v0",
    "MiniGrid-LavaCrossingS11N5-v0",
    # Dynamic obstacles
    "MiniGrid-Dynamic-Obstacles-5x5-v0",
    "MiniGrid-Dynamic-Obstacles-6x6-v0",
    "MiniGrid-Dynamic-Obstacles-8x8-v0",
    "MiniGrid-Dynamic-Obstacles-16x16-v0",
    "MiniGrid-Dynamic-Obstacles-Random-5x5-v0",
    "MiniGrid-Dynamic-Obstacles-Random-6x6-v0",
    # Obstructed maze
    "MiniGrid-ObstructedMaze-1Q-v0",
    "MiniGrid-ObstructedMaze-2Q-v0",
    # Distributional shift
    "MiniGrid-DistShift1-v0",
    "MiniGrid-DistShift2-v0",
)

# GameAction → Minigrid action integer
# Chosen to match solver_client.py's _KEY_MAP:
#   ACTION1 = up/w      → forward
#   ACTION2 = down/s    → drop  (put the carried object on the cell ahead)
#   ACTION3 = left/a    → turn left  (counter-clockwise)
#   ACTION4 = right/d   → turn right (clockwise)
#   ACTION5 = e/space   → INTERACT — context-sensitive, resolved in `_apply`:
#                         toggle a door, open a full box, else pick up
#   ACTION6 = f         → pickup (legacy; ACTION5 subsumes it)
#   ACTION7             → NOT available: BaseAdapter reserves it for UNDO
#
# Only five ids are usable as game actions here (ACTION6 is the mouse click in
# ARC-AGI-3 and ACTION7 is UNDO), but MiniGrid needs six primitives — forward,
# turn x2, pickup, drop, toggle. ACTION5 therefore merges pickup and toggle into
# one "interact with what I'm facing" key, which is unambiguous because a cell
# holds at most one object. That is what lets the whole MiniGrid family — keys,
# doors, boxes, blocking balls — be played inside ACTION1..ACTION5. Minigrid's
# `done` action is dropped: no supported env uses it.
_ACTION_MAP: dict[GameAction, int] = {
    GameAction.ACTION1: 2,  # forward       (up / w)
    GameAction.ACTION2: 4,  # drop          (down / s)
    GameAction.ACTION3: 0,  # turn left     (left / a)
    GameAction.ACTION4: 1,  # turn right    (right / d)
    GameAction.ACTION5: 5,  # interact      (e / space)  ← toggle / open / pickup
    GameAction.ACTION6: 3,  # pickup        (f)
    GameAction.ACTION7: 4,  # (unreachable: intercepted as UNDO)
}


# ---------------------------------------------------------------------------
# Step budget tracker
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MinigridAdapter
# ---------------------------------------------------------------------------

class MinigridAdapter(BaseAdapter):
    """Wraps a Minigrid environment as an ARCBaseGame-compatible object.

    Args:
        env_id:      Gymnasium environment ID, e.g. "MiniGrid-DoorKey-5x5-v0".
        seed:        Base random seed. set_level(idx) uses seed + idx.
        partial_obs: If False (default), wraps with FullyObsWrapper to expose the
                     entire grid. If True, uses the native 7×7 egocentric view so
                     the agent must navigate with limited visibility. Unseen cells
                     render as black (ARC palette index 5). The game_id gains a
                     "partial_" prefix to distinguish it from the full-obs variant.
    """

    def __init__(self, env_id: str, seed: int = 0, partial_obs: bool = False) -> None:
        self._env_id = env_id
        self._partial_obs = partial_obs
        # Derive a clean game_id string (strip "MiniGrid-" prefix and "-v0" suffix)
        clean = env_id.replace("MiniGrid-", "").replace("-v0", "").replace("-v1", "")
        prefix = "minigrid_partial_" if partial_obs else "minigrid_"
        self._game_id = f"{prefix}{clean.lower().replace('-', '_')}"
        self._seed = seed

        raw_env = gym.make(env_id)
        # FullyObsWrapper exposes the complete grid; omit it for partial-obs mode
        # where only a 7×7 egocentric cone is visible.
        self._env = raw_env if partial_obs else FullyObsWrapper(raw_env)

        # Rotation augmentation (randomized per level reset)
        self._rotation_k: int = 0

        # Shared adapter scaffolding (state, step counter, undo stack) + 1st frame.
        self._init_base(self._game_id, max_steps=self._env.unwrapped.max_steps)
        self._reset_env(self._seed)

    # ------------------------------------------------------------------
    # Internal reset (BaseAdapter hook)
    # ------------------------------------------------------------------

    def _reset_env(self, seed: int) -> None:
        obs, _info = self._env.reset(seed=seed)
        self._rotation_k = _random_module.randint(0, 3)
        self._current_frame = self._rotate_frame(
            _grid_to_arc_frame(obs["image"], partial_obs=self._partial_obs)
        )
        self._state = GameState.NOT_FINISHED
        self._action_count = 0
        self._step_counter.reset(self._env.unwrapped.max_steps)
        u = self._env.unwrapped
        pos = u.agent_pos
        self._start_pos = pos.copy() if hasattr(pos, 'copy') else np.array(pos)
        self._start_dir = u.agent_dir

    def _rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply the current rotation augmentation to a frame."""
        if self._rotation_k == 0:
            return frame
        return np.rot90(frame, k=self._rotation_k).copy()

    def _is_front_lava(self) -> bool:
        """Check if the cell in front of the agent is lava."""
        u = self._env.unwrapped
        fwd = u.front_pos
        if fwd is None:
            return False
        cell = u.grid.get(*fwd)
        return cell is not None and cell.type == "lava"

    def _obs_image(self) -> np.ndarray:
        """The observation image for the CURRENT env state.

        In full-obs mode the whole-grid view comes from the FullyObsWrapper; in
        partial-obs mode ``self._env`` is the bare env (behind gym's
        OrderEnforcing wrapper, which has no ``observation`` method), so the
        egocentric view has to be taken straight off ``gen_obs``."""
        u = self._env.unwrapped
        obs = u.gen_obs()
        if not self._partial_obs:
            obs = self._env.observation(obs)
        return obs["image"]

    def _handle_lava_reset(self) -> None:
        """Teleport agent back to start and apply a 5-step penalty."""
        u = self._env.unwrapped
        u.agent_pos = self._start_pos.copy()
        u.agent_dir = self._start_dir
        self._action_count += 5
        self._step_counter.steps_remaining = max(0, self._step_counter.max_steps - self._action_count)
        self._current_frame = self._rotate_frame(
            _grid_to_arc_frame(self._obs_image(), partial_obs=self._partial_obs)
        )
        if self._step_counter.steps_remaining == 0:
            self._state = GameState.GAME_OVER

    # ------------------------------------------------------------------
    # BaseAdapter hooks (game_id/set_level/perform_action/_make_frame_data are
    # provided by BaseAdapter; RESET + ACTION7-undo + terminal short-circuit too).
    # ------------------------------------------------------------------

    def _apply(self, action_input: ActionInput) -> None:
        """Perform one egocentric action against the live env.

        WIN: terminated AND reward>0. GAME_OVER: terminated AND reward==0
        (lava/death), or truncated / step budget exhausted."""
        mg_action = _ACTION_MAP.get(action_input.id, 2)  # default: forward

        # ACTION5 = INTERACT: resolve against whatever is in front of the agent.
        # A cell holds at most one object, so the intent is never ambiguous —
        # doors get toggled, a box that *contains* something gets opened (so the
        # contents are revealed rather than trapped inside), and anything else
        # pickable gets picked up.
        if action_input.id == GameAction.ACTION5:
            u = self._env.unwrapped
            fwd = u.front_pos
            front = u.grid.get(*fwd) if fwd is not None else None
            if front is not None and front.type != "door" and front.can_pickup():
                # empty box / key / ball → pick it up; full box → toggle (below)
                if not (front.type == "box" and front.contains is not None):
                    mg_action = 3

        # Box disambiguation for the legacy ACTION6 = raw pickup: a pickup on a
        # box that *contains* another object should toggle (5) instead. If the
        # box is empty (UnlockPickup, where picking the box up IS the goal), let
        # the pickup pass.
        if mg_action == 3 and action_input.id == GameAction.ACTION6:
            u = self._env.unwrapped
            fwd = u.front_pos
            front = u.grid.get(*fwd) if fwd is not None else None
            if front is not None and front.type == "box" and front.contains is not None:
                mg_action = 5  # toggle instead of pickup

        # Lava safety: forward into lava resets to start with 5-step penalty
        if mg_action == 2 and self._is_front_lava():
            self._handle_lava_reset()
            return

        obs, reward, terminated, truncated, _info = self._env.step(mg_action)

        self._current_frame = self._rotate_frame(
            _grid_to_arc_frame(obs["image"], partial_obs=self._partial_obs)
        )
        self._action_count += 1
        self._step_counter.steps_remaining = max(0, self._step_counter.max_steps - self._action_count)

        if terminated:
            self._state = GameState.WIN if reward > 0 else GameState.GAME_OVER
        elif truncated or self._step_counter.steps_remaining == 0:
            self._state = GameState.GAME_OVER
        else:
            self._state = GameState.NOT_FINISHED

    # ── snapshot / restore for generic UNDO ──────────────────────────────────
    #: Env attributes that hold REFERENCES to objects living in `grid` (or in the
    #: agent's hands). They must be copied in the SAME deepcopy pass as the grid,
    #: or the copy breaks the identity link they encode -- and several envs decide
    #: the WIN by identity: UnlockPickup / BlockedUnlockPickup / KeyCorridor /
    #: ObstructedMaze test `self.carrying == self.obj`, Unlock tests
    #: `self.door.is_open`, DynamicObstacles moves the balls in `self.obstacles`.
    #: Copy them separately (or not at all) and every post-undo pickup of the
    #: mission object silently stops counting as a win.
    _REF_ATTRS: tuple[str, ...] = ("carrying", "obj", "door", "obstacles",
                                   "room_grid", "rooms")

    def _snapshot(self):
        """Full mutable state: the minigrid env (grid/agent/carrying/step_count)
        plus the adapter bookkeeping. The grid and every env attribute that points
        into it are deep-copied TOGETHER (one shared memo), so reopening a door or
        picking up a key is reversible without severing object identity."""
        u = self._env.unwrapped
        refs = {"grid": u.grid}
        for attr in self._REF_ATTRS:
            if hasattr(u, attr):
                refs[attr] = getattr(u, attr)
        return (_copy.deepcopy(refs), np.array(u.agent_pos), int(u.agent_dir),
                int(u.step_count), self._action_count,
                self._step_counter.steps_remaining,
                self._state, self._current_frame)

    def _restore(self, snap) -> None:
        u = self._env.unwrapped
        (refs, apos, adir, step_count, self._action_count,
         self._step_counter.steps_remaining,
         self._state, self._current_frame) = snap
        # Copy again on the way out so a snapshot stays reusable (the base
        # solver's burst-rollback may restore the same anchor more than once).
        for attr, val in _copy.deepcopy(refs).items():
            setattr(u, attr, val)
        u.agent_pos = np.array(apos)
        u.agent_dir = adir
        u.step_count = step_count

    def absolute_move(self, target_dir: int) -> FrameDataRaw:
        """Move one step in an absolute grid direction without exposing turn actions.

        Silently issues the minimum number of turn actions to face target_dir,
        then one forward action. Intermediate turn frames are suppressed so the
        caller only sees the result of the movement.

        target_dir:
            0 = east  (right in grid)
            1 = south (down in grid)
            2 = west  (left in grid)
            3 = north (up in grid)
        """
        dummy = ActionInput(id=GameAction.ACTION1)

        if self._state in (GameState.WIN, GameState.GAME_OVER):
            return self._make_frame_data(dummy)

        # snapshot pre-move so ACTION7 undo reverses a whole directional move
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self.max_undo:
            self._undo_stack.pop(0)

        current_dir = self._env.unwrapped.agent_dir
        turns_right = (target_dir - current_dir) % 4
        turns_left  = (current_dir - target_dir) % 4

        if turns_right <= turns_left:
            turn_int, n_turns = 1, turns_right   # Minigrid: 1 = turn right
        else:
            turn_int, n_turns = 0, turns_left    # Minigrid: 0 = turn left

        # Execute turns silently; episode could end on max_steps during a turn
        for _ in range(n_turns):
            obs, reward, terminated, truncated, _ = self._env.step(turn_int)
            self._action_count += 1
            self._step_counter.steps_remaining = max(0, self._step_counter.max_steps - self._action_count)
            if terminated or truncated or self._step_counter.steps_remaining == 0:
                self._current_frame = self._rotate_frame(
                    _grid_to_arc_frame(obs["image"], partial_obs=self._partial_obs)
                )
                self._state = GameState.WIN if (terminated and reward > 0) else GameState.GAME_OVER
                return self._make_frame_data(dummy)

        # Step forward — lava check first
        if self._is_front_lava():
            self._handle_lava_reset()
            return self._make_frame_data(dummy)

        obs, reward, terminated, truncated, _ = self._env.step(2)  # forward
        self._current_frame = self._rotate_frame(
            _grid_to_arc_frame(obs["image"], partial_obs=self._partial_obs)
        )
        self._action_count += 1
        self._step_counter.steps_remaining = max(0, self._step_counter.max_steps - self._action_count)

        if terminated:
            self._state = GameState.WIN if reward > 0 else GameState.GAME_OVER
        elif truncated or self._step_counter.steps_remaining == 0:
            self._state = GameState.GAME_OVER
        else:
            self._state = GameState.NOT_FINISHED

        return self._make_frame_data(dummy)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def list_available_envs() -> list[str]:
        """Return supported Minigrid environment IDs (full-observation variants)."""
        return list(_AVAILABLE_MINIGRID_ENVS)

    @staticmethod
    def list_available_partial_envs() -> list[str]:
        """Return supported Minigrid environment IDs (partial-observation variants).

        The env IDs are identical to the full-obs list; pass partial_obs=True when
        constructing the adapter to enable the 7×7 egocentric view.
        """
        return list(_AVAILABLE_MINIGRID_ENVS)

    def close(self) -> None:
        """Release the underlying gymnasium environment."""
        self._env.close()

    def __repr__(self) -> str:
        return (
            f"MinigridAdapter(env={self._env_id!r}, partial_obs={self._partial_obs}, "
            f"level={self._current_level_index}, state={self._state})"
        )
