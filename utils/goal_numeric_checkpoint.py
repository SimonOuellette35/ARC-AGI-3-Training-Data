"""Tensor + JSON training state, including optimizer/RNG state on Torch 2.5."""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def save_state(value, prefix):
    prefix = Path(prefix)
    tensors = {}

    def encode(item):
        if torch.is_tensor(item):
            key = str(len(tensors))
            tensors[key] = item.detach().cpu().contiguous().clone()
            return {'tensor': key}
        if isinstance(item, dict):
            return {'dict': [[encode(k), encode(v)] for k, v in item.items()]}
        if isinstance(item, (list, tuple)):
            return {'tuple' if isinstance(item, tuple) else 'list': [encode(v) for v in item]}
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError(f'Unsupported checkpoint value: {type(item)}')

    structure = encode(value)
    save_file(tensors, str(prefix.with_suffix('.safetensors')))
    prefix.with_suffix('.json').write_text(json.dumps(structure) + '\n')


def load_state(prefix):
    prefix = Path(prefix)
    tensors = load_file(str(prefix.with_suffix('.safetensors')))

    def decode(item):
        if not isinstance(item, dict):
            return item
        if 'tensor' in item:
            return tensors[item['tensor']]
        if 'dict' in item:
            return {decode(k): decode(v) for k, v in item['dict']}
        if 'tuple' in item:
            return tuple(decode(v) for v in item['tuple'])
        return [decode(v) for v in item['list']]

    return decode(json.loads(prefix.with_suffix('.json').read_text()))
