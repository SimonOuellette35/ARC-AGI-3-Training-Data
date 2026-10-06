"""Shared rotation augmentation utilities for ARC-AGI-3 games.

Provides frame rotation (via a RenderableUserDisplay) and directional
action / click-coordinate remapping so that any game can be randomly
rotated at level-construction time without per-game boilerplate.

Rotation convention:
    k=0  →   0° (identity)
    k=1  →  90° counter-clockwise  (np.rot90 k=1)
    k=2  → 180°
    k=3  → 270° counter-clockwise  (= 90° clockwise)
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from arcengine import GameAction, RenderableUserDisplay


# ── Directional action remapping ─────────────────────────────────────────────
#
# When the rendered frame is rotated by k*90° CCW, the player's directional
# input must be inverse-rotated so that "screen-up" still maps to the correct
# game-coordinate direction.
#
# ACTION1 = up, ACTION2 = down, ACTION3 = left, ACTION4 = right

ACTION_REMAP: dict[int, dict[GameAction, GameAction]] = {
    1: {
        GameAction.ACTION1: GameAction.ACTION4,   # up    → right
        GameAction.ACTION2: GameAction.ACTION3,   # down  → left
        GameAction.ACTION3: GameAction.ACTION1,   # left  → up
        GameAction.ACTION4: GameAction.ACTION2,   # right → down
    },
    2: {
        GameAction.ACTION1: GameAction.ACTION2,   # up    → down
        GameAction.ACTION2: GameAction.ACTION1,   # down  → up
        GameAction.ACTION3: GameAction.ACTION4,   # left  → right
        GameAction.ACTION4: GameAction.ACTION3,   # right → left
    },
    3: {
        GameAction.ACTION1: GameAction.ACTION3,   # up    → left
        GameAction.ACTION2: GameAction.ACTION4,   # down  → right
        GameAction.ACTION3: GameAction.ACTION2,   # left  → down
        GameAction.ACTION4: GameAction.ACTION1,   # right → up
    },
}

# Inverse remap: converts a game-coordinate direction to the screen direction.
# Used when the game produces a direction (e.g. arrow rendering) that needs to
# be displayed in rotated screen space.
ACTION_REMAP_INVERSE: dict[int, dict[GameAction, GameAction]] = {
    1: {
        GameAction.ACTION1: GameAction.ACTION3,
        GameAction.ACTION2: GameAction.ACTION4,
        GameAction.ACTION3: GameAction.ACTION2,
        GameAction.ACTION4: GameAction.ACTION1,
    },
    2: {
        GameAction.ACTION1: GameAction.ACTION2,
        GameAction.ACTION2: GameAction.ACTION1,
        GameAction.ACTION3: GameAction.ACTION4,
        GameAction.ACTION4: GameAction.ACTION3,
    },
    3: {
        GameAction.ACTION1: GameAction.ACTION4,
        GameAction.ACTION2: GameAction.ACTION3,
        GameAction.ACTION3: GameAction.ACTION1,
        GameAction.ACTION4: GameAction.ACTION2,
    },
}


def remap_action(action_id: GameAction, k: int) -> GameAction:
    """Remap a player directional action for rotation *k*.

    Non-directional actions (ACTION5, ACTION6, ACTION7, RESET, …) pass through
    unchanged.
    """
    if k % 4 == 0:
        return action_id
    return ACTION_REMAP[k % 4].get(action_id, action_id)


# ── Directional action flip remapping ────────────────────────────────────────
#
# A frame flip mirrors one screen axis, so the matching directional input must
# be swapped along that axis for a screen press to have the intended effect:
#   - horizontal flip (np.fliplr) mirrors the x-axis  → swap left ↔ right
#   - vertical   flip (np.flipud) mirrors the y-axis  → swap up   ↔ down
# Each is an involution (its own inverse), and because they touch disjoint axes
# they commute, so hflip and vflip can be composed in either order.

FLIP_HORIZONTAL: dict[GameAction, GameAction] = {
    GameAction.ACTION3: GameAction.ACTION4,   # left  → right
    GameAction.ACTION4: GameAction.ACTION3,   # right → left
}

FLIP_VERTICAL: dict[GameAction, GameAction] = {
    GameAction.ACTION1: GameAction.ACTION2,   # up    → down
    GameAction.ACTION2: GameAction.ACTION1,   # down  → up
}


def _flip_action(action_id: GameAction, hflip: bool, vflip: bool) -> GameAction:
    """Apply the horizontal / vertical flip direction swaps to an action.

    Non-directional actions (ACTION5, …) pass through unchanged.
    """
    if hflip:
        action_id = FLIP_HORIZONTAL.get(action_id, action_id)
    if vflip:
        action_id = FLIP_VERTICAL.get(action_id, action_id)
    return action_id


def remap_action_full(action_id: GameAction, k: int,
                      hflip: bool = False, vflip: bool = False) -> GameAction:
    """Screen action → game/engine action for the full presentation transform.

    The frame is presented as ``flip(rot90(engine_frame, k))`` — rotation first,
    then the flip(s) — so a screen press is inverted by first undoing the flip
    (an involution) and then undoing the rotation. Non-directional actions pass
    through unchanged.
    """
    return remap_action(_flip_action(action_id, hflip, vflip), k)


def inverse_remap_action_full(action_id: GameAction, k: int,
                              hflip: bool = False, vflip: bool = False) -> GameAction:
    """Game/engine action → the screen action to press so that
    :func:`remap_action_full` maps it back to ``action_id``.

    Inverse of :func:`remap_action_full`: undo the rotation via
    ``ACTION_REMAP_INVERSE`` first, then apply the flip swap(s).
    """
    k = k % 4
    if k != 0:
        action_id = ACTION_REMAP_INVERSE[k].get(action_id, action_id)
    return _flip_action(action_id, hflip, vflip)


def remap_click(x: int, y: int, k: int, size: int = 64) -> tuple[int, int]:
    """Inverse-rotate display click coordinates back to game coordinates.

    ``size`` is the frame edge length (default 64).
    """
    k = k % 4
    if k == 0:
        return (x, y)
    if k == 1:
        return (size - 1 - y, x)
    if k == 2:
        return (size - 1 - x, size - 1 - y)
    return (y, size - 1 - x)


def random_rotation_k(seed: int | None = None, level_index: int = 0) -> int:
    """Return a rotation index (0, 1, 2, or 3).

    With `seed` given, the draw is DETERMINISTIC in (seed, level_index): it comes from an
    isolated RNG, so the same (seed, level) always yields the same rotation regardless of any
    global-RNG consumption or in-level play. This lets a recorded episode and a later live
    run of the same seed render each level at the SAME orientation (so per-(seed, level)
    demonstration matching works byte-for-byte). With `seed` None it falls back to a global
    random draw (legacy nondeterministic behaviour)."""
    if seed is None:
        return random.randint(0, 3)
    return random.Random(f"rot:{int(seed)}:{int(level_index)}").randint(0, 3)


# ── Display rotator ─────────────────────────────────────────────────────────

class RotationDisplay(RenderableUserDisplay):
    """RenderableUserDisplay that rotates the entire frame by k*90° CCW.

    Add an instance to the Camera's ``interfaces`` list (typically last so
    it runs after other HUD overlays).  Call ``set_k(k)`` on level load.
    """

    def __init__(self) -> None:
        self._k: int = 0

    def set_k(self, k: int) -> None:
        self._k = k % 4

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self._k == 0:
            return frame
        return np.rot90(frame, k=self._k).copy()
