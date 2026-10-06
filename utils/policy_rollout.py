"""Imagined continuations shared by V1/V2 training and fixed-depth evaluation."""
import torch

from utils.policy_model_common import BOARD_H, BOARD_W, NO_COORD
from utils.dynamics_decoding import configured_change_threshold, decode_dynamics_board


def rollout_starts(batch, mask, depth, generator=None):
    """Sample same-level windows; a terminal action may be their final hop."""
    dyn_ok = batch["dyn_ok"].to(mask.device) & mask
    continuation = batch.get("dyn_continue", batch["dyn_ok"]).to(mask.device) & mask
    window_ok = dyn_ok.clone()
    for shift in range(1, depth):
        shifted = torch.zeros_like(dyn_ok)
        if shift < mask.shape[1]:
            shifted[:, :-shift] = dyn_ok[:, shift:]
        window_ok &= shifted
    for shift in range(depth - 1):
        shifted = torch.zeros_like(dyn_ok)
        if shift == 0:
            shifted = continuation
        elif shift < mask.shape[1]:
            shifted[:, :-shift] = continuation[:, shift:]
        window_ok &= shifted
    scores = torch.rand(mask.shape, device=mask.device, generator=generator)
    scores[~window_ok] = -1
    rows = window_ok.any(1).nonzero(as_tuple=False).flatten()
    return rows, scores.argmax(1)[rows]


def imagined_inputs(inp, rows, starts, predictions):
    """Keep real history through the root, then one frame per imagined step.

    Repacking removes future animation pixels AND their lengths. Recorded
    actions specify interventions; no observations of their outcomes are used.
    The source tensors are never mutated.
    """
    parts = []
    for i, (row, start) in enumerate(zip(rows.tolist(), starts.tolist())):
        observed_steps = start + 1
        nf = int(inp["nframes"][row, :observed_steps].sum())
        frames = inp["frames"][row, :nf]
        if predictions:
            frames = torch.cat([frames, torch.stack([p[i] for p in predictions])])
        nfr = torch.cat([inp["nframes"][row, :observed_steps],
                         inp["nframes"].new_ones(len(predictions))])
        parts.append((row, frames, nfr))
    B = len(parts)
    L = max(len(p[2]) for p in parts)
    F = max(len(p[1]) for p in parts)
    out = {"frames": inp["frames"].new_zeros(B, F, BOARD_H, BOARD_W),
           "act": inp["act"].new_zeros(B, L),
           "xy": inp["xy"].new_full((B, L, 2), NO_COORD),
           "mask": inp["mask"].new_zeros(B, L),
           "nframes": inp["nframes"].new_zeros(B, L),
           "frame_step": inp["frame_step"].new_zeros(B, F),
           "frame_slot": inp["frame_slot"].new_zeros(B, F),
           "frame_mask": inp["frame_mask"].new_zeros(B, F)}
    for i, (row, frames, nfr) in enumerate(parts):
        length, nf = len(nfr), len(frames)
        out["frames"][i, :nf] = frames
        out["act"][i, :length] = inp["act"][row, :length]
        out["xy"][i, :length] = inp["xy"][row, :length]
        out["mask"][i, :length] = True
        out["nframes"][i, :length] = nfr
        out["frame_mask"][i, :nf] = True
        steps = torch.arange(length, device=nfr.device)
        out["frame_step"][i, :nf] = torch.repeat_interleave(steps, nfr)
        span_starts = nfr.cumsum(0) - nfr
        out["frame_slot"][i, :nf] = (torch.arange(nf, device=nfr.device)
                                     - torch.repeat_interleave(span_starts, nfr))
    return out


def dynamics_rollout_predictions(model, batch, inp, depth, generator=None, *,
                                 return_policy=False, return_prediction=False):
    """Final-hop logits and original batch indices, or None if no window fits.

    Intermediate predictions are detached. Only the final forward preserves
    the caller's gradient mode, so training and evaluation use the same path.
    With ``return_policy``, return ``(dynamics_result, policy_result)`` where
    policy_result contains final-forward type/pointer logits, original batch
    rows, and a mask selecting every imagined input step (excluding the root).
    With ``return_prediction``, append the decoded final board to the dynamics
    result for fidelity metrics; the first three outputs remain raw logits and
    original batch indices. The checkpoint's change gate controls roll-in.
    """
    if depth < 1:
        raise ValueError("rollout depth must be positive")
    rows, starts = rollout_starts(batch, inp["mask"], depth, generator)
    if rows.numel() == 0:
        return None
    predictions = []
    threshold = configured_change_threshold(getattr(model, "cfg", None))
    for hop in range(depth):
        roll_inp = imagined_inputs(inp, rows, starts, predictions)
        dm = torch.zeros_like(roll_inp["mask"])
        dm[torch.arange(len(rows), device=rows.device), starts + hop] = True
        # Repeated forwards mix no-grad roll-in and checkpointed training.
        # Do not reuse autocast weight casts across those gradient contexts;
        # recomputation must see the same graph as the final forward.
        device_type = inp["frames"].device.type
        with torch.set_grad_enabled(torch.is_grad_enabled() and hop == depth - 1), \
                torch.autocast(device_type=device_type,
                               enabled=torch.is_autocast_enabled(device_type),
                               cache_enabled=False):
            types, pointers, colors, changes, decoded = model(
                **roll_inp, return_dynamics=True, dyn_decode=dm)
        if hop < depth - 1:
            board = decode_dynamics_board(colors, changes, roll_inp, decoded, threshold)
            # Predictions fed to imagined_inputs are in rollout-batch order.
            ordered = torch.empty_like(board)
            ordered[decoded[:, 0]] = board
            predictions.append(ordered.detach().to(inp["frames"].dtype))
    original_decoded = torch.stack([rows[decoded[:, 0]], decoded[:, 1]], dim=1)
    result = colors, changes, original_decoded
    if return_prediction:
        result += (decode_dynamics_board(colors, changes, roll_inp, decoded, threshold),)
    if return_policy:
        steps = torch.arange(roll_inp["mask"].shape[1], device=rows.device)
        imagined_mask = roll_inp["mask"] & (steps[None] > starts[:, None])
        return result, (types, pointers, rows, imagined_mask)
    return result
