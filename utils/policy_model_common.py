"""Shared board encoding, decoding, action heads and configuration for policy models."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# Constants describing the data / task
# ----------------------------------------------------------------------------
BOARD_H = 64
BOARD_W = 64
NUM_COLORS = 16       # observation cell values are in [0, 15]

# --- Action space ----------------------------------------------------------
# Discrete action vocabulary: index 0..6 == RESET, ACTION1..5, and ACTION6 (the
# mouse click). CLICK_INDEX actions additionally carry a pixel coordinate (x, y).
CLICK_INDEX = 6            # ACTION6 == mouse click; carries a (x, y) coordinate
NUM_ACTION_TYPES = 7       # discrete type head size: indices 0..6 inclusive
NUM_ACTIONS = NUM_ACTION_TYPES  # backwards-compatible alias for older importers
# Default click-distribution resolution. Must divide BOARD_W/BOARD_H so pixels
# bin cleanly. >=16 so the pointer can actually represent per-cell answers (see
# module docstring); tune with --pointer-grid.
POINTER_GRID = 16
NO_COORD = 255             # uint8 sentinel in the cache: action has no (x, y)
NO_ACTION = -1             # int16 sentinel: padding slot in an optimal-action set

# --- Patch tokenisation ----------------------------------------------------
# Boards are encoded to a PATCH_GRID x PATCH_GRID conv map, one token per cell.
# Module-level because the dataset/collate side reasons about token counts too.
PATCH_GRID = 4
PATCHES_PER_FRAME = PATCH_GRID * PATCH_GRID


def coord_to_bin(xy: torch.Tensor, grid: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel coords -> (bx, by) grid-bin indices (both long, same leading shape).

    ``xy`` is ``(..., 2)`` holding ``(x, y)`` pixel coordinates (the ``NO_COORD``
    sentinel and any out-of-range value are clamped into the valid grid; callers
    mask these away for non-click steps). Binning is y-major to match
    ``coord_to_bin_flat``.
    """
    scale_x = BOARD_W // grid
    scale_y = BOARD_H // grid
    x = xy[..., 0].clamp(0, BOARD_W - 1).long() // scale_x
    y = xy[..., 1].clamp(0, BOARD_H - 1).long() // scale_y
    return x.clamp(0, grid - 1), y.clamp(0, grid - 1)


def coord_to_bin_flat(xy: torch.Tensor, grid: int) -> torch.Tensor:
    """Pixel coords -> flat grid-cell index ``by * grid + bx`` (y-major)."""
    bx, by = coord_to_bin(xy, grid)
    return by * grid + bx


def bin_flat_to_coord(idx: int, grid: int) -> tuple[int, int]:
    """Flat grid-cell index -> (x, y) pixel coordinate at the cell CENTRE.

    Inverse of ``coord_to_bin_flat`` used at inference to turn a pointer-head cell
    into a concrete click. Decoding to the cell centre round-trips the corpus's
    clicks exactly whenever they fall on cell centres (e.g. grid=16 -> cell 4 ->
    bin 4 -> centre 4*4+2 == pixel 18, the value the demos store)."""
    cell_x = BOARD_W // grid
    cell_y = BOARD_H // grid
    bx = idx % grid
    by = idx // grid
    return int(bx * cell_x + cell_x // 2), int(by * cell_y + cell_y // 2)


def _span_keep(k: int, max_frames: int) -> np.ndarray:
    """Which frames of a ``k``-frame animation to keep under a ``max_frames`` cap.

    Evenly spaced and ALWAYS including both endpoints: the last frame is the
    settled state the next action was decided from (dropping it would break the
    state/target correspondence), and the first is the animation's onset. Returns
    the in-span offsets in increasing order; a no-op ``arange(k)`` when the span
    already fits.

    Lives here (not in the dataset) because ``policy_runtime`` caps live
    animation spans with the exact same rule the trainer used."""
    if max_frames <= 0 or k <= max_frames:
        return np.arange(k)
    if max_frames == 1:                       # only room for the settled frame
        return np.asarray([k - 1])
    return np.unique(np.linspace(0, k - 1, max_frames).round().astype(np.int64))


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class PatchEncoder(nn.Module):
    """Conv a 64x64 board (16 colors) to a grid x grid map -> grid*grid patch tokens.

    Color-embed each cell to a small channel dim, then a strided conv stack
    downsamples 64 by a factor of two per layer until it reaches grid x grid
    (grid=4 -> 64->32->16->8->4). Unlike a single-token encoder
    this keeps every spatial cell as its own ``d_model`` token, so absolute
    position survives into the transformer (see the module docstring).
    """

    def __init__(self, d_model: int, color_dim: int = 16,
                 grid: int = PATCH_GRID):
        super().__init__()
        self.color_embed = nn.Embedding(NUM_COLORS, color_dim)
        # The conv stack is built FROM the requested grid: each layer halves the
        # board, so reaching a grid x grid map takes log2(64/grid) stride-2 convs
        # (grid 4 -> 4 convs, grid 8 -> 3, grid 16 -> 2). This used to be a fixed
        # 4-layer stack, so passing grid=8 silently still produced a 4x4 map and
        # the model's patch count no longer matched the encoder's output.
        if grid < 1 or BOARD_H % grid or (BOARD_H // grid) & (BOARD_H // grid - 1):
            raise ValueError(f"patch grid {grid} must divide {BOARD_H} by a power "
                             f"of two (valid: 1, 2, 4, 8, 16, 32, 64)")
        n_down = max(1, int(math.log2(BOARD_H // grid))) if grid < BOARD_H else 0
        mid = [32, 64, 128][-(n_down - 1):] if n_down > 1 else []
        ch = [color_dim] + mid + [d_model]
        layers: list[nn.Module] = []
        for cin, cout in zip(ch[:-1], ch[1:]):
            layers += [
                nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
                nn.GELU(),
            ]
        self.conv = nn.Sequential(*layers)         # (N, d_model, grid, grid)
        self.grid = grid

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        # boards: (N,64,64) int64 -> (N, grid*grid, d_model)
        x = self.color_embed(boards)                 # (N,64,64,C)
        x = x.permute(0, 3, 1, 2).contiguous()       # (N,C,64,64)
        x = self.conv(x)                             # (N,d,grid,grid)
        return x.flatten(2).transpose(1, 2)          # (N, grid*grid, d)


class BoardDecoder(nn.Module):
    """A ``grid x grid`` d_model feature map -> per-cell logits on the full 64x64.

    The mirror image of ``PatchEncoder``: ``log2(64 / grid)`` stride-2 transposed
    convs double the map each step until it is 64x64, then a 1x1 conv emits
    ``out_ch`` logits per cell (16 colour logits + 1 change logit for the
    dynamics head). Only used when ``--dynamics`` is on.
    """

    def __init__(self, d_model: int, grid: int, out_ch: int):
        super().__init__()
        if grid < 1 or BOARD_H % grid or (BOARD_H // grid) & (BOARD_H // grid - 1):
            raise ValueError(f"patch grid {grid} must divide {BOARD_H} by a power "
                             f"of two (valid: 1, 2, 4, 8, 16, 32, 64)")
        n_up = int(math.log2(BOARD_H // grid)) if grid < BOARD_H else 0
        # Channel schedule shrinks d_model -> ... -> 32 over the up-convs, padded
        # with 32 when there are more up-convs than the base list has entries.
        base = [d_model, 128, 64, 32]
        chans = base[:n_up + 1] + [32] * max(0, n_up + 1 - len(base))
        layers: list[nn.Module] = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            layers += [
                nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
                nn.GELU(),
            ]
        self.deconv = nn.Sequential(*layers)          # (N, chans[-1], 64, 64)
        self.head = nn.Conv2d(chans[-1] if n_up else d_model, out_ch,
                              kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, d_model, grid, grid) -> (N, out_ch, 64, 64)
        return self.head(self.deconv(x))


class ActionIO(nn.Module):
    """Shared action embedding + dual output heads for a click-capable policy.

    Encapsulates everything that changes between the simple-only and mouse-click
    action spaces, so both the baseline and Option-C policies reuse ONE
    implementation:

      * ``action_token(act, xy)`` -> a d-dim token per action that embeds the
        discrete type AND (for clicks) the chosen ``(x, y)`` bin, so a later
        state can condition on where the previous click landed. Non-click actions
        add a learned ``no_click`` vector instead.
      * ``predict(ctx)`` -> ``(type_logits, ptr_logits)`` where ``type_logits`` is
        over the 7 discrete actions and ``ptr_logits`` is a flat
        ``grid*grid`` distribution over click locations.
    """

    def __init__(self, d_model: int, grid: int):
        super().__init__()
        self.grid = grid
        self.action_embed = nn.Embedding(NUM_ACTION_TYPES, d_model)
        # Click coordinate embedding at pointer-grid resolution (separable x / y).
        self.x_embed = nn.Embedding(grid, d_model)
        self.y_embed = nn.Embedding(grid, d_model)
        self.no_click = nn.Parameter(torch.zeros(d_model))
        self.type_head = nn.Linear(d_model, NUM_ACTION_TYPES)
        self.pointer_head = nn.Linear(d_model, grid * grid)

    def action_token(self, act: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """act (B,L) long, xy (B,L,2) long -> action tokens (B,L,d)."""
        tok = self.action_embed(act)
        bx, by = coord_to_bin(xy, self.grid)               # (B,L) each
        click_vec = self.x_embed(bx) + self.y_embed(by)    # (B,L,d)
        is_click = (act == CLICK_INDEX).unsqueeze(-1)      # (B,L,1)
        coord = torch.where(is_click, click_vec, self.no_click.expand_as(click_vec))
        return tok + coord

    def predict(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ctx (..., d) -> (type_logits (..., NUM_ACTION_TYPES),
        ptr_logits (..., grid*grid))."""
        return self.type_head(ctx), self.pointer_head(ctx)


@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ff_mult: int = 4
    dropout: float = 0.1
    color_dim: int = 16
    max_states: int = 100
    pointer_grid: int = POINTER_GRID
    patch_grid: int = PATCH_GRID  # board -> patch_grid x patch_grid tokens
    max_frames: int = 4          # cap on frames per step's animation span
    dynamics: bool = False       # build the auxiliary next-board dynamics head
    dynamics_change_threshold: float = -1.0  # raw V1 default; V2 overrides to .25

    def __post_init__(self):
        from utils.dynamics_decoding import change_threshold
        change_threshold(self.dynamics_change_threshold)


def frame_metadata(frames, act, mask, nframes=None, frame_step=None,
                   frame_slot=None, frame_mask=None):
    """Normalize optional single-frame inputs to the packed animation format."""
    if nframes is None:
        if frames.shape[1] != act.shape[1]:
            raise ValueError("Pass frame-span metadata for multi-frame inputs")
        nframes = mask.long()
        frame_step = torch.arange(act.shape[1], device=frames.device).expand_as(act)
        frame_slot = torch.zeros_like(act)
        frame_mask = mask
    elif any(x is None for x in (frame_step, frame_slot, frame_mask)):
        raise ValueError("nframes requires frame_step, frame_slot and frame_mask")
    return nframes, frame_step, frame_slot, frame_mask


def packed_layout(nframes, mask, frame_step, frame_slot, tokens_per_frame,
                  extra_tokens):
    """Token offsets shared by the original and spatial/temporal policies.

    Real tokens are left-packed; causal attention cannot reach right padding.
    Readout and action offsets for padded steps are clamped for safe gathers.
    """
    sizes = (nframes * tokens_per_frame + extra_tokens) * mask.long()
    start = sizes.cumsum(1) - sizes
    length = int(sizes.sum(1).max().item())
    if length == 0:
        raise ValueError("A policy batch must contain at least one valid step")
    readout = (start + nframes * tokens_per_frame).clamp(max=length - 1)
    action = (readout + 1).clamp(max=length - 1)
    frame = start.gather(1, frame_step.clamp(min=0)) + frame_slot * tokens_per_frame
    return length, readout, action, frame
