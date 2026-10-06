"""Pack model history without importing game-engine action types."""
import numpy as np
import torch

from utils.policy_model_common import NO_COORD, _span_keep


def policy_inputs(cfg, spans, actions, action_coords, device):
    """Pack a history with a dummy final action, shared by policy and search."""
    L = len(spans)
    if not L or len(actions) != L - 1 or len(action_coords) != L - 1:
        raise ValueError("Expected one more span than actions and coordinates")
    actions, action_coords = list(actions), list(action_coords)
    if L > cfg.max_states:                     # keep the most recent window
        keep = cfg.max_states
        spans = spans[-keep:]
        actions = actions[-(keep - 1):] if keep > 1 else []
        action_coords = action_coords[-(keep - 1):] if keep > 1 else []
        L = keep

    # Cap each animation the way the dataset does, then flatten to one stream.
    capped = [s[_span_keep(len(s), cfg.max_frames)] for s in spans]
    nfr = np.asarray([len(s) for s in capped], dtype=np.int64)     # (L,)
    frames = np.concatenate(capped, axis=0)                        # (F,64,64)
    frame_step = np.repeat(np.arange(L, dtype=np.int64), nfr)      # (F,)
    frame_slot = np.concatenate([np.arange(k) for k in nfr])       # (F,)

    def _b(a, dtype=torch.long):                # -> (1, ...) batch of one
        return torch.as_tensor(a, dtype=dtype, device=device).unsqueeze(0)

    act = actions + [0]                                          # dummy for last
    coords = action_coords + [(NO_COORD, NO_COORD)]             # dummy for last
    mask = torch.ones(1, L, dtype=torch.bool, device=device)

    return dict(frames=_b(frames), act=_b(act), xy=_b(coords), mask=mask,
                nframes=_b(nfr), frame_step=_b(frame_step),
                frame_slot=_b(frame_slot),
                frame_mask=torch.ones(1, frames.shape[0], dtype=torch.bool,
                                      device=device))
