"""Compact, lossless codec for multi-level training episodes.

Replaces the verbose per-episode JSON (64x64 integer grids stored as ASCII)
with a single zstd-compressed container per episode:

    [4 bytes: little-endian uint32 header length]
    [header: JSON bytes -> game_id, per-level metadata + frame offsets]
    [payload: raw uint8 observation frames, concatenated in level/frame order]

Observations are stored as uint8 (values are small, 0..255); the whole thing is
compressed with zstd. Measured ~300-500x smaller than the original JSON.

The `labels` field (legacy encoder-labelling data) is intentionally dropped.

Public API:
    from utils import episode_codec

    save_episode(path, episode_dict)   # dict shaped like the original JSON
    load_episode(path)                 # -> dict, observations as np.ndarray[k,H,W]
    json_to_episode(json_dict)         # normalize a loaded JSON dict (drops labels)
"""

import io
import json
import struct

import numpy as np
import zstandard as zstd

MAGIC = b"EPZ1"
ZSTD_LEVEL = 19

# Fields on a level that we drop entirely (legacy encoder-labelling data).
DROP_LEVEL_FIELDS = ("labels",)
# Fields we handle explicitly; everything else on a level is preserved as-is.
KNOWN_LEVEL_FIELDS = ("level_id", "observations", "actions") + DROP_LEVEL_FIELDS


def _level_frames(level):
    """Return (frames_uint8[k,H,W], height, width) for one level's observations."""
    obs = level["observations"]
    if not obs:
        return np.zeros((0, 0, 0), dtype=np.uint8), 0, 0
    arr = np.asarray(obs, dtype=np.int64)
    if arr.ndim != 3:
        raise ValueError(
            f"expected observations shaped [k,H,W], got ndim={arr.ndim} shape={arr.shape}"
        )
    lo, hi = int(arr.min()), int(arr.max())
    if lo < 0 or hi > 255:
        raise ValueError(f"observation values out of uint8 range: min={lo} max={hi}")
    return arr.astype(np.uint8), arr.shape[1], arr.shape[2]


def json_to_episode(d):
    """Normalize a JSON-loaded episode dict: drop labels, keep everything else."""
    out = {k: v for k, v in d.items() if k != "levels"}
    out_levels = []
    for lvl in d["levels"]:
        clean = {k: v for k, v in lvl.items() if k not in DROP_LEVEL_FIELDS}
        out_levels.append(clean)
    out["levels"] = out_levels
    return out


def save_episode(path, episode):
    """Serialize an episode dict (original-JSON shape) to a compact .ep.zst file."""
    payload = io.BytesIO()
    level_meta = []
    offset = 0
    for lvl in episode["levels"]:
        frames, h, w = _level_frames(lvl)
        raw = frames.tobytes()
        payload.write(raw)
        meta = {k: v for k, v in lvl.items() if k not in KNOWN_LEVEL_FIELDS}
        meta.update(
            level_id=lvl.get("level_id"),
            actions=lvl.get("actions", []),
            n_frames=int(frames.shape[0]),
            height=h,
            width=w,
            byte_offset=offset,
            byte_len=len(raw),
        )
        level_meta.append(meta)
        offset += len(raw)

    header = {
        k: v for k, v in episode.items() if k != "levels"
    }
    header["levels"] = level_meta
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    blob = io.BytesIO()
    blob.write(MAGIC)
    blob.write(struct.pack("<I", len(header_bytes)))
    blob.write(header_bytes)
    blob.write(payload.getvalue())

    compressed = zstd.ZstdCompressor(level=ZSTD_LEVEL).compress(blob.getvalue())
    with open(path, "wb") as f:
        f.write(compressed)


def _read(path):
    """``(header dict, payload bytes)`` for a .ep.zst file."""
    with open(path, "rb") as f:
        blob = zstd.ZstdDecompressor().decompress(f.read())

    if blob[:4] != MAGIC:
        raise ValueError(f"bad magic in {path!r}: {blob[:4]!r}")
    (hlen,) = struct.unpack("<I", blob[4:8])
    return json.loads(blob[8 : 8 + hlen].decode("utf-8")), blob[8 + hlen :]


def load_header(path):
    """The episode's header only: ``game_id`` plus per-level metadata
    (``level_id``, ``actions``, frame count and shape) with no frames attached.

    For callers that want to know what an episode COVERS without paying to
    materialise it -- ``run_tests.py`` reads one episode per game just to learn
    which levels the corpus has, and the frames are the whole size of the file.
    """
    return _read(path)[0]


def load_episode(path):
    """Load a .ep.zst file back into an episode dict.

    Observations are returned as np.ndarray[k,H,W] uint8 (not nested lists).
    Call .tolist() on them if a downstream consumer needs plain Python lists.
    """
    header, payload = _read(path)

    episode = {k: v for k, v in header.items() if k != "levels"}
    levels = []
    for meta in header["levels"]:
        k, h, w = meta["n_frames"], meta["height"], meta["width"]
        start = meta["byte_offset"]
        raw = payload[start : start + meta["byte_len"]]
        if k and h and w:
            frames = np.frombuffer(raw, dtype=np.uint8).reshape(k, h, w)
        else:
            frames = np.zeros((k, h, w), dtype=np.uint8)
        lvl = {
            key: val
            for key, val in meta.items()
            if key
            not in ("n_frames", "height", "width", "byte_offset", "byte_len")
        }
        lvl["observations"] = frames
        levels.append(lvl)
    episode["levels"] = levels
    return episode
