"""A residual pointer that scores locations in the current spatial feature grid."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class SpatialPointer(nn.Module):
    def __init__(self, d_model, patch_grid, pointer_grid, dim=64):
        super().__init__()
        self.patch_grid = patch_grid
        self.pointer_grid = pointer_grid
        self.dim = dim
        self.upscale = max(1, pointer_grid // patch_grid)
        if pointer_grid >= patch_grid and pointer_grid % patch_grid:
            raise ValueError('pointer_grid must be an integer multiple of patch_grid')
        self.policy_norm = nn.LayerNorm(d_model)
        self.spatial_norm = nn.LayerNorm(d_model)
        self.query = nn.Linear(d_model, dim)
        self.keys = nn.Linear(d_model, dim * self.upscale ** 2)
        # An existing checkpoint's pointer predictions are unchanged at startup.
        nn.init.zeros_(self.keys.weight)
        nn.init.zeros_(self.keys.bias)

    def forward(self, policy, spatial):
        """policy (N,D), spatial (N,P*P,D) -> residual logits (N,G*G)."""
        keys = self.keys(self.spatial_norm(spatial)).transpose(1, 2)
        keys = keys.reshape(-1, self.dim * self.upscale ** 2, self.patch_grid, self.patch_grid)
        if self.upscale > 1:
            keys = F.pixel_shuffle(keys, self.upscale)
        if keys.shape[-1] != self.pointer_grid:
            keys = F.adaptive_avg_pool2d(keys, self.pointer_grid)
        query = self.query(self.policy_norm(policy))
        return torch.einsum('nd,ndhw->nhw', query, keys).flatten(1) / math.sqrt(self.dim)
