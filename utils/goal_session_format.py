"""Lossless session serialization shared by hypothesizer A training/inference."""
from __future__ import annotations
from functools import lru_cache
import json
import numpy as np
from utils.goal_hypothesizer import TASK

FORMAT = """Anonymous session; grid 64x64; colors are hex 0-f; coordinates (row,column).
Actions: 0 RESET, 1-5 ACTION1-5, 6 click (x=column,y=row).
FULL: 64 hex rows. RLE: each row is space-separated hex_color:decimal_count.
DELTA: space-separated row,column:new_hex_color relative to the preceding frame.
SAME: identical to the preceding frame. Every encoding is lossless.
Each STATE lists the animation frames produced by its incoming action, ending
in the settled frame. The last STATE is the current decision point.
"""


def cap_span(span, maximum):
    if maximum <= 0 or len(span) <= maximum:
        return span
    keep = [len(span) - 1] if maximum == 1 else np.unique(
        np.linspace(0, len(span) - 1, maximum).round().astype(int))
    return span[keep]


def encode_frame(frame, previous=None):
    frame = np.asarray(frame)
    if frame.shape != (64, 64) or np.any((frame < 0) | (frame > 15)) or np.any(frame != frame.astype(int)):
        raise ValueError('Expected a 64x64 integer frame in 0..15')
    return _encode_frame(frame.astype(np.uint8, copy=False).tobytes(),
                         None if previous is None else np.asarray(previous).astype(np.uint8).tobytes())


_HEX_TABLE = bytes.maketrans(bytes(range(16)), b'0123456789abcdef')


@lru_cache(maxsize=4096)
def _encode_frame(pixels, previous_pixels):
    # Content keys also work across overlapping prefixes and separate episodes;
    # the bounded cache cannot retain observation arrays or grow with the corpus.
    if pixels == previous_pixels:
        return 'SAME'
    frame = np.frombuffer(pixels, dtype=np.uint8).reshape(64, 64)
    hex_pixels = pixels.translate(_HEX_TABLE).decode('ascii')
    full = 'FULL\n' + '\n'.join(hex_pixels[i:i + 64] for i in range(0, 4096, 64))
    rows = []
    for row in frame:
        edges = [0, *(np.flatnonzero(row[1:] != row[:-1]) + 1).tolist(), 64]
        rows.append(' '.join(f'{int(row[a]):x}:{b-a}' for a, b in zip(edges, edges[1:])))
    choices = [full, 'RLE\n' + '\n'.join(rows)]
    if previous_pixels is not None:
        previous = np.frombuffer(previous_pixels, dtype=np.uint8).reshape(64, 64)
        cells = np.argwhere(frame != previous)
        choices.append('DELTA ' + ' '.join(f'{r},{c}:{int(frame[r,c]):x}' for r, c in cells) if len(cells) else 'SAME')
    return min(choices, key=len)


def encode_state(span, action, *, max_frames=4, new_level=False):
    lines = ['STATE' + (' NEW LEVEL' if new_level else '')]
    if action is not None:
        lines.append('incoming action ' + json.dumps(action_dict(action), separators=(',', ':')))
    previous = None
    for frame in cap_span(span, max_frames):
        lines.append(encode_frame(frame, previous))
        previous = frame
    return '\n'.join(lines)


def action_dict(action):
    index = int(action['index'])
    if index not in range(7):
        raise ValueError(f'Invalid action index {index}')
    data = {}
    if index == 6:
        data = {key: int(action['data'][key]) for key in ('x', 'y')}
        if any(value not in range(64) for value in data.values()):
            raise ValueError('Click outside board')
    return {'id': index, 'data': data}


def action_id_remap(rng):
    """One random permutation of action ids 1-5, as a dict; 0 and 6 map to themselves.

    Draw ONCE per training example and reuse it (via `apply_action_remap`) on
    every action list -- real prefix AND any imagined continuation -- that
    belongs to that same session, so the relabelling stays consistent for the
    whole thing a mapping that changed partway through would carry no
    learnable signal at all. See `shuffle_action_ids` for why this exists.
    """
    ids = list(range(1, 6))
    shuffled = ids[:]
    rng.shuffle(shuffled)
    remap = dict(zip(ids, shuffled))
    remap[0], remap[6] = 0, 6
    return remap


def apply_action_remap(actions, remap):
    """Relabel each action's ``index`` through `remap`; ``None`` passes through."""
    return [None if a is None else {**a, 'index': remap[int(a['index'])]} for a in actions]


def shuffle_action_ids(actions, rng):
    """Relabel indices 1-5 through one random permutation; 0 and 6 are fixed.

    Every corpus game is recorded under its one true action mapping, so a
    text model can memorize "ACTION3 usually turns right" across games
    instead of reading a session's own transitions to learn what its actions
    do. Drawing ONE permutation per training example (held fixed across that
    example's whole session, since a mapping that changed mid-session would
    carry no signal at all) breaks that shortcut -- the same trick as
    utils.policy_training's action-identity shuffle, generalised from the
    rotation augmentation's 4-element remap. RESET (0) always means reset and
    a click's effect is the (x, y) it carries, not a discrete id, so neither
    is shuffled. ``None`` entries (a leading no-action placeholder) pass
    through unchanged.
    """
    return apply_action_remap(actions, action_id_remap(rng))


def session_record(frame0, spans, actions, *, max_states=32, max_frames=4, boundaries=()):
    """Serialize observed spans and their incoming actions; metadata is excluded.

    For policy_runtime inputs use runtime_record below. Every STATE starts with
    a full/RLE frame, allowing FIFO eviction without dangling delta references.
    """
    if len(spans) != len(actions) or not spans:
        raise ValueError('Need one incoming action per nonempty span')
    first = max(0, len(spans) - max_states) if max_states > 0 else 0
    blocks = []
    for i in range(first, len(spans)):
        blocks.append(encode_state(spans[i], actions[i], max_frames=max_frames,
                                   new_level=i in boundaries))
    return {'prefix': FORMAT + '\nCURRENT LEVEL FRAME0\n' + encode_frame(frame0), 'states': blocks}


def runtime_record(frame0, spans, actions, action_coords, **kwargs):
    """Accept the same spans/actions/action_coords as policy_runtime.

    frame0 must be retained from the current level's initial settled frame.
    actions contains len(spans)-1 taken actions; there is no future action.
    """
    if len(actions) != len(spans) - 1 or len(action_coords) != len(actions):
        raise ValueError('Runtime history requires one fewer actions than spans')
    incoming = [None] + [dict(index=index, data=dict(zip(('x', 'y'), xy)))
                         for index, xy in zip(actions, action_coords)]
    return session_record(frame0, spans, incoming, **kwargs)


def prompt_text(record, drop=0):
    return record['prefix'] + '\n\n' + '\n\n'.join(record['states'][drop:]) + '\n\n' + TASK
