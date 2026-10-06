"""Shared colour-permutation augmentation for ARC-AGI-3 games.

Provides a :class:`ColorRemapDisplay` (a ``RenderableUserDisplay``) that applies
a per-pixel colour lookup to the final composited frame, plus
:func:`random_color_lut` to draw a deterministic permutation from a seed.

Because the remap runs on the *rendered frame* (last in the camera's interface
chain) it recolours everything uniformly -- sprites, background and letterbox --
without touching any game logic. Gameplay and solvers key on sprite *names* and
cell positions, never on colour, so a colour permutation is always
solvability-safe (unlike rotation, which remaps controls).

Convention (mirrors ``random_rotation_k``):
    seed given  → deterministic permutation from an isolated RNG.
    seed None   → a fresh random permutation (legacy per-run behaviour).

Scope it EPISODE-wide (draw once, reuse for every level) by seeding
level-independently and caching the LUT, so an episode's frames share one
colour scheme.
"""

from __future__ import annotations

import random

import numpy as np
from arcengine import RenderableUserDisplay

NUM_COLORS = 16


def random_color_lut(seed: object = None, num_colors: int = NUM_COLORS) -> np.ndarray:
    """Return a length-``num_colors`` lookup array remapping ``old → lut[old]``.

    The result is a random *permutation* of ``range(num_colors)``, so it is a
    bijection: every originally-distinct colour maps to a distinct colour and no
    two elements collide. With ``seed`` given the draw is deterministic in the
    seed; with ``seed`` None it is a fresh per-call random permutation.
    """
    rng = random.Random(seed)
    perm = list(range(num_colors))
    rng.shuffle(perm)
    return np.asarray(perm, dtype=np.int64)


class ColorRemapDisplay(RenderableUserDisplay):
    """RenderableUserDisplay that permutes the whole frame's colours via a LUT.

    Add an instance to the Camera's ``interfaces`` list and call
    :meth:`set_lut` (typically once per episode). With no LUT set it is the
    identity. Order relative to a rotation display does not matter -- a per-pixel
    recolour and a spatial rotation commute.
    """

    def __init__(self) -> None:
        self._lut: np.ndarray | None = None

    def set_lut(self, lut: np.ndarray | None) -> None:
        self._lut = None if lut is None else np.asarray(lut, dtype=np.int64)

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self._lut is None:
            return frame
        return self._lut[frame].astype(frame.dtype, copy=False)
