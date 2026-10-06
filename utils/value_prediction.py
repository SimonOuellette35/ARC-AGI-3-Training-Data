"""Frozen V1/V2 dynamics augmentation for labelled value transitions."""
import random

import torch

from utils.policy_checkpoint import policy_from_checkpoint
from utils.policy_model_common import NO_COORD
from utils.dynamics_decoding import configured_change_threshold, decode_dynamics_board


class PredictedTransitions:
    """Replace observations, never labels, actions, reference frames or pairs.

    A sampled target transition ends a 1..max_depth imagined suffix, rooted in
    its own recorded level. Only settled observations at/before the root are
    visible. Subsequent inputs use the checkpoint's board decoder, as in MCTS.
    """

    def __init__(self, model, cfg, device, fraction=.25, max_depth=3,
                 batch_size=8, amp=False, context=0, cache_features=True):
        if not 0 <= fraction <= 1:
            raise ValueError("predicted fraction must be between 0 and 1")
        if max_depth < 1 or batch_size < 1:
            raise ValueError("prediction depth and batch size must be positive")
        if context < 0:
            raise ValueError("prediction context must be nonnegative (0 = checkpoint limit)")
        if not cfg.dynamics or cfg.max_states < 1:
            raise ValueError("A dynamics-enabled checkpoint with context is required")
        self.model = model.eval().requires_grad_(False)
        self.cfg, self.device = cfg, torch.device(device)
        self.fraction, self.max_depth = fraction, max_depth
        self.batch_size = batch_size
        self.amp = amp and self.device.type == "cuda"
        self.context = min(context, cfg.max_states) if context else cfg.max_states
        self.cache_features = cache_features and hasattr(model, 'encode_frames')

    @classmethod
    def from_checkpoint(cls, path, device, **kwargs):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model, cfg = policy_from_checkpoint(checkpoint, device)
        return cls(model, cfg, device, **kwargs)

    def _inputs(self, histories, actions, coords):
        """Batch single-frame spans, truncating the context exactly as planning."""
        lengths = [min(len(h), self.context) for h in histories]
        b, length = len(histories), max(lengths)
        frames = torch.zeros(b, length, 64, 64, dtype=torch.long, device=self.device)
        act = torch.zeros(b, length, dtype=torch.long, device=self.device)
        xy = torch.full((b, length, 2), NO_COORD, dtype=torch.long, device=self.device)
        mask = torch.zeros(b, length, dtype=torch.bool, device=self.device)
        for i, n in enumerate(lengths):
            frames[i, :n] = torch.stack(histories[i][-n:])
            act[i, :n] = actions[i][-n:]
            xy[i, :n] = coords[i][-n:]
            mask[i, :n] = True
        steps = torch.arange(length, device=self.device).expand(b, -1)
        return dict(frames=frames, act=act, xy=xy, mask=mask,
                    nframes=mask.long(), frame_step=steps * mask,
                    frame_slot=torch.zeros_like(act), frame_mask=mask)

    def _encode(self, frames):
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            features = self.model.encode_frames(frames)
        return features if isinstance(features, tuple) else (features,)

    @staticmethod
    def _pack_features(features, inp):
        packed = []
        b, length = inp['mask'].shape
        for component in features:
            value = component[0].new_zeros((b, length, *component[0].shape[1:]))
            for row, history in enumerate(component):
                n = min(len(history), length)
                value[row, :n] = history[-n:]
            packed.append(value)
        return packed[0] if len(packed) == 1 else tuple(packed)

    @torch.no_grad()
    def __call__(self, batch, rng=None):
        rng = rng if rng is not None else random
        n = len(batch["act"])
        expected = self.fraction * n
        # Stochastic rounding also gives tiny batches a 25% long-run mixture.
        take = int(expected) + int(rng.random() < expected - int(expected))
        depths = torch.zeros(n, dtype=torch.long)
        if not take:
            return batch, depths
        groups = {}
        for index in rng.sample(range(n), take):
            li, target = batch["sources"][index].tolist()
            depth = rng.randint(1, min(self.max_depth, target))
            groups.setdefault(depth, []).append((index, li, target))
            depths[index] = depth

        out = dict(batch)
        out["current"] = batch["current"].to(self.device).clone()
        out["next"] = batch["next"].to(self.device).clone()
        self.model.eval()
        for depth, requests in groups.items():
            # Similar context lengths reduce padded transformer work. Selection
            # and rollout depths have already been sampled, so labels stay put.
            requests.sort(key=lambda request: min(request[2] - depth + 1, self.context))
            for offset in range(0, len(requests), self.batch_size):
                chunk = requests[offset:offset + self.batch_size]
                histories, actions, coords = [], [], []
                for _, li, target in chunk:
                    level = batch["levels"][li]
                    root = target - depth
                    lo = max(0, root + 1 - self.context)
                    # No real board after root is copied into the rollout.
                    observed = torch.as_tensor(level["settled"][lo:root + 1],
                                               device=self.device, dtype=torch.long)
                    histories.append(list(observed.unbind()))
                    actions.append(torch.as_tensor(level["act"][lo + 1:target + 1],
                                                   device=self.device, dtype=torch.long))
                    coords.append(torch.as_tensor(level["axy"][lo + 1:target + 1],
                                                  device=self.device, dtype=torch.long))
                roots = [len(h) for h in histories]
                features = None
                if self.cache_features:
                    # The CNN (and V2's spatial attention) is independent per
                    # frame. Encode the real prefix once, not once per hop.
                    encoded = self._encode(torch.cat([torch.stack(h) for h in histories]))
                    features = [list(part.split(roots)) for part in encoded]
                for hop in range(depth):
                    inp = self._inputs(histories,
                                       [a[:r + hop] for a, r in zip(actions, roots)],
                                       [c[:r + hop] for c, r in zip(coords, roots)])
                    decode = torch.zeros_like(inp["mask"])
                    rows = torch.arange(len(chunk), device=self.device)
                    decode[rows, inp["mask"].sum(1) - 1] = True
                    if features is not None:
                        inp['frame_features'] = self._pack_features(features, inp)
                    with torch.autocast(device_type=self.device.type, enabled=self.amp):
                        _, _, colors, changes, decoded = self.model(
                            **inp, return_dynamics=True, dyn_decode=decode)
                    # Respect the decoder's row mapping instead of relying on order.
                    predicted = decode_dynamics_board(
                        colors, changes, inp, decoded, configured_change_threshold(self.cfg))
                    for prediction, (row, _) in zip(predicted, decoded.tolist()):
                        if hop == depth - 1:
                            index = chunk[row][0]
                            out["current"][index] = histories[row][-1]
                            out["next"][index] = prediction
                        histories[row].append(prediction)
                    if features is not None and hop < depth - 1:
                        ordered = torch.empty_like(predicted)
                        ordered[decoded[:, 0]] = predicted
                        encoded = self._encode(ordered)
                        features = [[torch.cat([history, new[row:row + 1]])
                                     for row, history in enumerate(component)]
                                    for component, new in zip(features, encoded)]
        return out, depths
