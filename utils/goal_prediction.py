"""Offline imagined evidence for both goal-program training formats."""
import hashlib
import random
import re

import numpy as np

from utils.goal_input_format import COLOR_NAMES, diff_text
from utils.dynamics_decoding import (
    configured_change_threshold, decode_dynamics_board, decoder_metadata,
)


def add_prediction_arguments(parser):
    parser.add_argument('--dynamics-ckpt', help='Frozen V1/V2 checkpoint for imagined evidence')
    parser.add_argument('--predicted-fraction', type=float, default=.25,
                        help='Fraction of prepared prompts with imagined suffixes (default .25)')
    parser.add_argument('--predicted-max-depth', type=int, default=3,
                        help='Maximum recorded-action rollout depth (default 3)')
    parser.add_argument('--prediction-device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--prediction-batch-size', type=int, default=8,
                        help='Maximum simultaneous imagined A rollouts (default 8; 1 uses less memory)')


def make_predictor(args):
    fraction = getattr(args, 'predicted_fraction', .25)
    depth = getattr(args, 'predicted_max_depth', 3)
    batch_size = getattr(args, 'prediction_batch_size', 8)
    if not 0 <= fraction <= 1 or depth < 1 or batch_size < 1:
        raise ValueError('Require 0 <= predicted-fraction <= 1, predicted-max-depth >= 1 '
                         'and prediction-batch-size >= 1')
    if not getattr(args, 'dynamics_ckpt', None) or fraction == 0:
        return None
    import torch
    from utils.policy_checkpoint import policy_from_checkpoint

    device = getattr(args, 'prediction_device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.dynamics_ckpt, map_location='cpu', weights_only=False)
    model, cfg = policy_from_checkpoint(checkpoint, device)
    print(f'[prepare] frozen dynamics {args.dynamics_ckpt}; {fraction:.0%} imagined, '
          f'1..{depth} steps on {device}', flush=True)
    return GoalRollouts(model, cfg, device, args.seed, fraction, depth, batch_size)


class GoalRollouts:
    def __init__(self, model, cfg, device, seed=42, fraction=.25, max_depth=3, batch_size=8):
        if not cfg.dynamics:
            raise ValueError('Imagined evidence requires a dynamics-enabled checkpoint')
        if batch_size < 1:
            raise ValueError('prediction-batch-size must be positive')
        self.model = model.eval().requires_grad_(False)
        self.cfg, self.device = cfg, device
        self.seed, self.fraction, self.max_depth = seed, fraction, max_depth
        self.batch_size = batch_size

    def selected_rng(self, identity):
        # Independent from prefix sampling, so enabling augmentation cannot
        # change which clean examples or game splits are used for validation.
        seed = hashlib.sha256(f'{self.seed}:{identity}'.encode()).digest()
        rng = random.Random(int.from_bytes(seed, 'big'))
        return rng if rng.random() < self.fraction else None

    def rollout(self, spans, incoming, future):
        """Real spans through root; only generated single-frame spans after it."""
        return self.rollout_many([(spans, incoming, future)])[0]

    def rollout_many(self, requests):
        """Batch independent continuations, returning spans in request order.

        Each hop decodes only the latest valid step, with the same animation
        cap and FIFO history as singleton inference. Finished rows leave the
        batch; future real observations never enter it.
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence
        from utils.policy_inputs import policy_inputs

        results = []
        with torch.no_grad():
            for offset in range(0, len(requests), self.batch_size):
                chunk = requests[offset:offset + self.batch_size]
                histories, taken, futures = [], [], []
                for spans, incoming, future in chunk:
                    if not spans or len(spans) != len(incoming) or not future:
                        raise ValueError('Need observed spans, incoming actions and a nonempty continuation')
                    histories.append(list(spans[-self.cfg.max_states:]))
                    taken.append(list(incoming[-self.cfg.max_states:]))
                    futures.append(future)
                predictions = [[] for _ in chunk]
                for hop in range(max(map(len, futures))):
                    active = [i for i, future in enumerate(futures) if hop < len(future)]
                    inputs = []
                    for i in active:
                        actions = [int(a['index']) for a in taken[i][1:]]
                        coords = [self._coords(a) for a in taken[i][1:]]
                        # Pack on CPU, then transfer whole padded tensors once per hop.
                        inp = policy_inputs(self.cfg, histories[i], actions, coords, 'cpu')
                        action = futures[i][hop]
                        inp['act'][0, -1] = int(action['index'])
                        inp['xy'][0, -1] = torch.tensor(self._coords(action))
                        inputs.append(inp)
                    inp = {key: pad_sequence([part[key][0] for part in inputs], batch_first=True,
                                             padding_value=255 if key == 'xy' else 0).to(self.device)
                           for key in inputs[0]}
                    decode = torch.zeros_like(inp['mask'])
                    decode[torch.arange(len(active), device=self.device), inp['mask'].sum(1) - 1] = True
                    with torch.autocast(device_type=torch.device(self.device).type,
                                        enabled=torch.device(self.device).type == 'cuda'):
                        _, _, colors, changes, decoded = self.model(**inp, return_dynamics=True, dyn_decode=decode)
                    frames = decode_dynamics_board(
                        colors, changes, inp, decoded, configured_change_threshold(self.cfg)
                    ).to('cpu', torch.uint8).numpy()
                    for frame, (row, _) in zip(frames, decoded.tolist()):
                        i = active[row]
                        span = frame[None]
                        predictions[i].append(span)
                        histories[i] = (histories[i] + [span])[-self.cfg.max_states:]
                        taken[i] = (taken[i] + [futures[i][hop]])[-self.cfg.max_states:]
                results.extend(predictions)
        return results

    @staticmethod
    def _coords(action):
        if int(action['index']) != 6:
            return (255, 255)
        return (int(action['data']['x']), int(action['data']['y']))


def prediction_counts(stats, split, record):
    details = record.get('imagined_rollout')
    stats[f'{split}_imagined' if details else f'{split}_real'] += 1
    if details:
        stats[f'{split}_imagined_steps'] += details['depth']


def prediction_metadata(predictor):
    return {'enabled': predictor is not None,
            **decoder_metadata(getattr(predictor, 'cfg', None)),
            'targets': 'unchanged_real_is_win_program', 'stage': 'prepare'}


_STEP = re.compile(r'^STEP (\d+)  (.+)\r?$', re.M)
_CLICK = re.compile(r'ACTION6 click\(row=(\d+),col=(\d+)\)')
_CELL = re.compile(r'\((\d+),(\d+)\) ([a-z_]+)->([a-z_]+)')


def parse_probe(text):
    """Read actions and only the losslessly reconstructable prefix of B.

    A count/bounding-box summary does not identify pixels. Stop reconstructing
    there; later action labels can still specify imagined interventions.
    """
    matches = list(_STEP.finditer(text))
    marker = text.index('FRAME 0 (initial):') + len('FRAME 0 (initial):')
    rows = text[marker:].lstrip('\r\n').splitlines()[:64]
    if len(rows) != 64 or any(re.fullmatch('[0-9a-f]{64}', r) is None for r in rows):
        raise ValueError('Imagined B inputs require a complete 64x64 initial hex frame')
    frames = [np.array([[int(c, 16) for c in r] for r in rows], dtype=np.uint8)]
    if not matches:
        return frames, [], [], text, None, ''
    task_start = text.index('Write the Python function:', matches[-1].end())
    ending = re.search(r'^\[episode ended: ([^\]\r\n]+)\]', text, re.M)
    end_steps = ending.start() if ending else task_start
    actions, labels = [], []
    colors = {name: index for index, name in COLOR_NAMES.items()}
    for i, match in enumerate(matches):
        if int(match[1]) != i + 1:
            raise ValueError('Probe STEP indices must be consecutive')
        label = match[2].strip()
        click = _CLICK.fullmatch(label)
        if click:
            r, c = map(int, click.groups())
            if r not in range(64) or c not in range(64):
                raise ValueError('Probe click outside board')
            action = dict(index=6, data=dict(x=c, y=r))
        elif re.fullmatch(r'ACTION[0-57]|RESET', label):
            action = dict(index=0 if label == 'RESET' else int(label[-1]))
        else:
            raise ValueError(f'Unsupported probe action: {label}')
        actions.append(action)
        labels.append(label)
        if len(frames) != i + 1:
            continue  # an earlier lossy diff prevents further reconstruction
        diff = text[match.end():matches[i + 1].start() if i + 1 < len(matches) else end_steps].strip()
        if re.fullmatch(r'\d+ cells changed; bbox rows \d+\.\.\d+ cols \d+\.\.\d+', diff):
            continue
        frame = frames[-1].copy()
        if diff != 'no cells changed':
            cells = list(_CELL.finditer(diff))
            if not cells or _CELL.sub('', diff).strip():
                raise ValueError('Unrecognized probe diff; refusing to invent observations')
            visited = set()
            for cell in cells:
                r, c = map(int, cell.group(1, 2))
                old, new = cell.group(3, 4)
                if (r not in range(64) or c not in range(64) or (r, c) in visited
                        or old not in colors or new not in colors or frame[r, c] != colors[old]):
                    raise ValueError('Inconsistent probe pixel diff')
                frame[r, c] = colors[new]
                visited.add((r, c))
        frames.append(frame)
    return frames, actions, labels, text[:matches[0].start()], ending, text[task_start:]


def imagined_probe(record, predictor):
    rng = predictor.selected_rng(f"B:{record['game']}:{record['source']}")
    if rng is None:
        return record
    frames, actions, labels, prefix, ending, task = parse_probe(record['input'])
    if not actions:
        return record  # reset-only evidence has no recorded action to imagine
    root = rng.randrange(min(len(frames), len(actions)))
    depth = rng.randint(1, min(predictor.max_depth, len(actions) - root))
    end = root + depth
    predicted = predictor.rollout([f[None] for f in frames[:root + 1]],
                                  [dict(index=0)] + actions[:root], actions[root:end])
    combined = frames[:root + 1] + [span[-1] for span in predicted]
    lines = [prefix.rstrip(), '']
    for i in range(end):
        lines.extend([f'STEP {i + 1}  {labels[i]}', diff_text(combined[i], combined[i + 1]), ''])
    # Terminal status remains real evidence, only at its original endpoint.
    if ending and end == len(actions):
        lines.extend([ending[0], ''])
    lines.append(task)
    return {**record, 'input': '\n'.join(lines),
            'imagined_rollout': {'root_decision': root, 'depth': depth, 'end_decision': end}}
