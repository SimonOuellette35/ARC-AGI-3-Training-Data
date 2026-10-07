"""Convert verbose training-episode JSON into compact zstd-compressed .ep.zst.

Each `episode_*.json` under data/training_multi_level/<game>/ becomes an
`episode_*.ep.zst` holding the same data (minus the legacy `labels` field),
typically ~300-500x smaller.

MULTI-FRAME AWARE. One action can own a SPAN of frames (its animation); the
corpus stores `observations` as a FLAT frame stream and tags each step with
`n_obs` (absent => 1). The codec payload is already a flat per-level frame
stack and `actions` ride along in the header verbatim, so nothing is lost --
but this converter now *verifies* it: the spans must tile the frame stream
exactly, in the JSON source and again in the reloaded .ep.zst. A file whose
spans don't tile is reported and neither converted nor deleted.

By default this is SAFE:
  * originals are NOT deleted (pass --delete to remove a .json only after its
    round-trip has been verified);
  * every file is verified: observations, actions (including `n_obs`) and all
    other level fields decoded from the new file must exactly match the JSON
    source (excluding dropped `labels`).

Usage:
    python3 utils/convert_training_data.py --dry-run               # report only
    python3 utils/convert_training_data.py --game aquarium_sorter  # one game
    python3 utils/convert_training_data.py                         # keep JSON
    python3 utils/convert_training_data.py --delete                # verified deletion
    python3 utils/convert_training_data.py --game enqueue --delete
    python3 utils/convert_training_data.py --delete --jobs 8       # parallel files

Also supports package execution: python3 -m utils.convert_training_data.
"""

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

if __package__:
    from . import episode_codec as codec
else:
    import episode_codec as codec

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "training_multi_level")


def _spans_ok(actions, n_frames):
    """True if the per-action `n_obs` spans tile a stream of `n_frames` exactly.

    Mirrors common_utils.action_spans (inlined: that module pulls in torch,
    which the system python3 running this script does not have)."""
    return sum(int(a.get("n_obs", 1) or 1) for a in actions) == n_frames


def _multiframe_stats(levels):
    """(n_multiframe_levels, max_n_obs) over a list of level dicts."""
    n_mf = 0
    mx = 1
    for lvl in levels:
        for a in lvl.get("actions", []):
            k = int(a.get("n_obs", 1) or 1)
            if k > mx:
                mx = k
            if k > 1:
                n_mf += 1
                break
    return n_mf, mx


def _check_spans(levels, where):
    """Raise if any level's action spans don't tile its observation stream."""
    for lvl in levels:
        obs = lvl["observations"]
        n_frames = len(obs)
        acts = lvl.get("actions", [])
        if not _spans_ok(acts, n_frames):
            got = sum(int(a.get("n_obs", 1) or 1) for a in acts)
            raise ValueError(
                f"{where}: level {lvl.get('level_id')} n_obs spans cover {got} frames "
                f"but observations has {n_frames}"
            )


def _verify(json_dict, reloaded):
    """Assert the reloaded episode matches the JSON source (ignoring dropped labels)."""
    jtop = {k: v for k, v in json_dict.items() if k != "levels"}
    rtop = {k: v for k, v in reloaded.items() if k != "levels"}
    assert jtop == rtop, f"episode-level fields differ: {jtop} != {rtop}"
    jl, rl = json_dict["levels"], reloaded["levels"]
    assert len(jl) == len(rl), f"level count {len(jl)} != {len(rl)}"
    for a, b in zip(jl, rl):
        assert a.get("level_id") == b.get("level_id"), "level_id mismatch"
        # actions carry `n_obs`; exact equality also proves the spans survived.
        assert a.get("actions", []) == b.get("actions", []), "actions mismatch"
        # any other per-level field (rotation, seed, ...) must round-trip too
        extra_j = {k: v for k, v in a.items() if k not in ("level_id", "actions", "observations", "labels")}
        extra_r = {k: v for k, v in b.items() if k not in ("level_id", "actions", "observations")}
        assert extra_j == extra_r, f"level fields differ: {extra_j} != {extra_r}"
        ja = np.asarray(a["observations"], dtype=np.uint8)
        rb = np.asarray(b["observations"], dtype=np.uint8)
        assert ja.shape == rb.shape, f"obs shape {ja.shape} != {rb.shape}"
        assert np.array_equal(ja, rb), "observation pixels differ"
    # spans must still tile after the round-trip
    _check_spans(reloaded["levels"], "reloaded")


def convert_file(json_path, delete=False, dry_run=False):
    """Convert one JSON file. Returns (json_bytes, out_bytes, had_labels, n_mf, max_n_obs)."""
    with open(json_path, "rb") as f:
        raw = f.read()
    d = json.loads(raw)
    had_labels = any("labels" in lvl for lvl in d["levels"])
    n_mf, max_n_obs = _multiframe_stats(d["levels"])

    # Multi-frame integrity of the SOURCE, before we treat it as disposable.
    _check_spans(d["levels"], "source")

    if dry_run:
        return len(raw), None, had_labels, n_mf, max_n_obs

    out_path = json_path[: -len(".json")] + ".ep.zst"
    episode = codec.json_to_episode(d)
    codec.save_episode(out_path, episode)

    # Verify round-trip before we consider the original disposable.
    reloaded = codec.load_episode(out_path)
    _verify(d, reloaded)

    out_bytes = os.path.getsize(out_path)
    if delete:
        os.remove(json_path)
    return len(raw), out_bytes, had_labels, n_mf, max_n_obs


def _worker(job):
    """Process-pool entry point: returns (path, result_tuple | None, error_str | None)."""
    jp, delete, dry_run = job
    try:
        return jp, convert_file(jp, delete=delete, dry_run=dry_run), None
    except Exception as e:  # noqa: BLE001 - surface which file failed, keep going
        return jp, None, f"{type(e).__name__}: {e}"


def convert_game(game_dir, delete=False, dry_run=False, jobs=1):
    files = sorted(glob.glob(os.path.join(game_dir, "*.json")))
    tot_in = tot_out = n = n_labels = n_mf_levels = n_failed = 0
    max_n_obs = 1
    jobs_list = [(jp, delete, dry_run) for jp in files]

    if jobs > 1 and len(jobs_list) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = ex.map(_worker, jobs_list, chunksize=1)
            outs = list(results)
    else:
        outs = [_worker(j) for j in jobs_list]

    for jp, res, err in outs:
        if err is not None:
            print(f"  !! FAILED {os.path.basename(jp)}: {err}", file=sys.stderr)
            n_failed += 1
            continue
        in_b, out_b, had_labels, n_mf, mx = res
        tot_in += in_b
        n += 1
        n_labels += int(had_labels)
        n_mf_levels += n_mf
        max_n_obs = max(max_n_obs, mx)
        if out_b is not None:
            tot_out += out_b
    return n, tot_in, tot_out, n_labels, n_mf_levels, max_n_obs, n_failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", help="convert only this game folder (default: all)")
    ap.add_argument("--delete", action="store_true", help="remove each .json after its round-trip verifies")
    ap.add_argument("--dry-run", action="store_true", help="report sizes only; write/delete nothing")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes (default 1)")
    ap.add_argument("--root", default=ROOT, help="training_multi_level root")
    args = ap.parse_args()

    if args.game:
        games = [args.game]
    else:
        games = sorted(
            d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))
        )

    grand_in = grand_out = grand_n = grand_labels = grand_failed = 0
    hdr = f"{'game':28s} {'files':>6} {'json':>10} {'out':>10} {'ratio':>7}  {'multi-frame':>18}  labels"
    print(hdr)
    print("-" * len(hdr))
    for g in games:
        gd = os.path.join(args.root, g)
        if not os.path.isdir(gd):
            print(f"  !! no such game folder: {gd}", file=sys.stderr)
            continue
        n, tin, tout, nlab, nmf, mx, nfail = convert_game(
            gd, delete=args.delete, dry_run=args.dry_run, jobs=args.jobs
        )
        grand_in += tin
        grand_out += tout
        grand_n += n
        grand_labels += nlab
        grand_failed += nfail
        ratio = (tin / tout) if tout else float("nan")
        lab = f"{nlab} dropped" if nlab else "-"
        mf = f"{nmf} lvls max n_obs={mx}" if nmf else "1:1"
        fail = f"  !! {nfail} FAILED" if nfail else ""
        print(f"{g:28s} {n:6d} {tin/1e9:9.3f}G {tout/1e9:9.3f}G {ratio:6.1f}x  {mf:>18}  {lab}{fail}", flush=True)

    print("-" * len(hdr))
    ratio = (grand_in / grand_out) if grand_out else float("nan")
    print(f"{'TOTAL':28s} {grand_n:6d} {grand_in/1e9:9.3f}G {grand_out/1e9:9.3f}G {ratio:6.1f}x"
          f"  {grand_labels} files had labels, {grand_failed} failed")
    if args.dry_run:
        print("\n(dry run: nothing written or deleted)")
    elif not args.delete:
        print("\n(originals kept; re-run with --delete once you've confirmed the new files)")


if __name__ == "__main__":
    main()
