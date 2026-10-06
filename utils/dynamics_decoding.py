"""Board decoding shared by imagined training, evaluation and planning."""
import torch


def change_threshold(value):
    """A probability threshold, or -1 to use the raw color decoder."""
    value = float(value)
    if value != -1 and not 0 <= value <= 1:
        raise ValueError("change threshold must be -1 (raw) or between 0 and 1")
    return value


def configured_change_threshold(cfg):
    return change_threshold(getattr(cfg, "dynamics_change_threshold", -1))


def decode_dynamics_board(colors, changes, inp, decoded, threshold):
    """Copy low-change pixels from each decoded step's own settled input.

    ``decoded`` maps decoder rows to (batch, step), including ragged animation
    spans. During a rollout this input is already imagined; future real boards
    and training targets never participate. Raw logits remain untouched.
    """
    threshold = change_threshold(threshold)
    predicted = colors.argmax(1)
    if threshold == -1:
        return predicted
    bi, li = decoded.unbind(1)
    settled = inp["nframes"].cumsum(1) - 1
    current = inp["frames"][bi, settled[bi, li]]
    return torch.where(changes.float().sigmoid() > threshold, predicted, current)


def decoder_metadata(cfg):
    threshold = configured_change_threshold(cfg)
    return {"decoder": "change_gate" if threshold >= 0 else "color_argmax",
            "change_threshold": threshold}
