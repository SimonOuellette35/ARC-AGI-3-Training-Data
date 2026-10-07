#!/usr/bin/env python3
"""master_data_generator.py -- drive many per-game training-data generators in parallel.

Given a CSV list of games, this resolves each name to its
``solvers/generate_<...>_training.py`` script, runs the generators concurrently
(one subprocess per game, ``--n_workers`` at a time), and then COMPRESSES each
game's output and DELETES the uncompressed JSON -- i.e. per game::

    <gen python>  solvers/generate_<x>_training.py --episodes N --out data/training_multi_level/<game_id>
    <compress python> utils/convert_training_data.py --game <game_id> --root data/training_multi_level --delete

The generators require ``arcengine``, while ``utils/convert_training_data.py``
requires ``numpy`` and ``zstandard``. The generator and compression interpreters
can use the same environment when it has all dependencies. Both are auto-detected and can be
overridden with ``--python`` / ``--compress-python`` (or ``$ARC_PYTHON`` /
``$SYSTEM_PYTHON``). This launcher itself is stdlib-only, so it runs under
either.

Usage::

    python3 master_data_generator.py --game_list list.csv --episodes 1000 --n_workers 6

    python3 master_data_generator.py --game_list list.csv --episodes 200 \
        --n_workers 8 --skip-existing --timeout 7200
    python3 master_data_generator.py --game_list list.csv --episodes 10 --dry-run
    python3 master_data_generator.py --list-games          # what names are resolvable

The game list
-------------
One game per line; blank lines and ``#`` comments are ignored. A name may be a
``game_id`` (``ar25``, ``sokoban``, ``gymgridworlds_barrier_5x5``,
``puzzlescript_drop_maze``), a generator stem (``gymgw_barrier_5x5``), a
``play.py`` name (``ps:vrps``, ``gymgw:Gym-Gridworlds/Full-4x5-v0``,
``MiniGrid-LockedRoom-v0``, ``partial:MiniGrid-LavaGapS5-v0``), or a path
to the generator script itself. A matching generator must exist in ``solvers/``.
With a header row, these columns are honoured
per game and override the global CLI values::

    game,episodes,start_seed,seed,args
    ar25,1000,0,0,
    sokoban,2000,,,--noise 0.2
    minigrid_empty_8x8,,,,

Without a header, the first column is the game, an all-digits second column is
the episode count, and anything after that is appended as extra generator args.

Notes
-----
* Generators number their output ``episode_00000...`` from scratch on every run,
  so re-running a game with the same ``--start-seed`` overwrites/duplicates its
  corpus. Use ``--skip-existing`` to leave already-populated games alone, or
  ``--start-seed`` to extend one.
* A generator that falls short of ``--episodes`` exits 1 but still leaves valid
  episodes behind; those are compressed and the game is reported ``partial``.
* Compression only deletes a ``.json`` after its round-trip has verified, so a
  leftover ``.json`` means that file failed to convert -- it is reported, not
  silently dropped.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
GEN_DIR = REPO / "solvers"
DEFAULT_OUT_ROOT = REPO / "data" / "training_multi_level"
CONVERTER = REPO / "utils" / "convert_training_data.py"
INDEX_CACHE = REPO / ".master_gen_index.json"

#: Where the generators' interpreter usually lives on this machine. Only used
#: when it actually exists; otherwise we fall back to the running interpreter.
CONDA_PYTHON = Path("/home/simon/anaconda3/envs/ARC-AGI-3/bin/python")

_PRINT_LOCK = threading.Lock()
_LIVE_PROCS: set[subprocess.Popen] = set()
_ABORT = threading.Event()


def log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Interpreters
# ---------------------------------------------------------------------------
def default_gen_python() -> str:
    env = os.environ.get("ARC_PYTHON")
    if env:
        return env
    if CONDA_PYTHON.exists():
        return str(CONDA_PYTHON)
    return sys.executable


def default_compress_python() -> str:
    env = os.environ.get("SYSTEM_PYTHON")
    if env:
        return env
    return shutil.which("python3") or sys.executable


# ---------------------------------------------------------------------------
# Resolving a game name -> (generator script, game_id)
# ---------------------------------------------------------------------------
# The output directory has to be the generator's own ``game_id`` (that is what
# the corpus is keyed by, and what ``utils/convert_training_data.py --game`` takes), so
# the launcher must know it before the generator runs. Most generators declare it
# as a literal; the minigrid/gymgw families DERIVE it from ``env_id`` at class
# definition time, which no regex can see -- those are resolved by importing the
# module in a subprocess (once, cached in ``.master_gen_index.json``).
_RE_GAME_ID = re.compile(r'\bgame_id\s*[:=]\s*["\']([\w.:/\-]+)["\']')
_RE_GAME_ID_CONST = re.compile(r'^GAME_ID\s*=\s*["\']([\w.:/\-]+)["\']', re.M)
_RE_ENV_ID = re.compile(r'^\s*env_id\s*(?::\s*str\s*)?=\s*["\']([^"\']+)["\']', re.M)
# Match complete Python string literals so titles containing apostrophes,
# punctuation, or escaped quotes keep their full spelling.
_RE_GAME_ALIAS = re.compile(
    r'''^\s*(?:GAME_NAME|game_name|GAME_MODULE|GAME_MODULE_ID|game_module_id)\s*(?::\s*str\s*)?=\s*'''
    r'''("(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')''', re.M)
#: The ``sys.exit(XxxSolver.main(SPEC))`` line -- names the object whose
#: ``game_id`` the generator actually writes under.
_RE_MAIN_CALL = re.compile(r'(?:sys\.exit|raise\s+SystemExit)\(\s*([\w.]+)\s*\(([^)]*)\)')

_PROBE_SRC = r"""
import importlib.util, json, sys
path, names = sys.argv[1], sys.argv[2].split(",") if sys.argv[2] else []
spec = importlib.util.spec_from_file_location("_gen_probe", path)
mod = importlib.util.module_from_spec(spec)
sys.modules["_gen_probe"] = mod
spec.loader.exec_module(mod)

def gid(obj):
    v = getattr(obj, "game_id", None)
    return v if isinstance(v, str) and v else None

found = None
for n in names:                      # the main() target, then its arguments
    obj = mod
    try:
        for part in n.split("."):
            obj = getattr(obj, part)
    except AttributeError:
        continue
    found = gid(obj)
    if found:
        break
if not found:                        # anything module-level that declares one
    for n, obj in vars(mod).items():
        if n.startswith("_"):
            continue
        found = gid(obj)
        if found:
            break
print(json.dumps({"game_id": found}))
"""


def _normalize(name: str) -> str:
    """Collapse the cosmetic differences between a game_id and a file stem.

    ``gymgridworlds_fourrooms_8x7`` / ``gymgw_fourrooms8x7`` /
    ``puzzlescript_drop_maze`` / ``drop_maze`` all have to find each other."""
    s = name.strip().lower()
    # Accept client names that accidentally repeat the PuzzleScript namespace.
    while s.startswith("ps:"):
        s = s[len("ps:"):]
    for pre in ("puzzlescript_", "ps_", "ps:", "gymgridworlds_", "gymgw_", "gymgw:"):
        if s.startswith(pre):
            s = s[len(pre):]
    return re.sub(r"[^a-z0-9]", "", s)


class GeneratorIndex:
    """Maps user-supplied game names to ``(script, game_id)``."""

    def __init__(self, gen_dir: Path = GEN_DIR) -> None:
        self.scripts = sorted(gen_dir.glob("generate_*_training.py"))
        self.by_name: dict[str, Path] = {}       # exact alias -> script
        self.by_norm: dict[str, list[Path]] = {}  # normalized alias -> scripts
        self.game_ids: dict[Path, str] = {}      # script -> statically-known id
        self.main_names: dict[Path, list[str]] = {}
        self._probe_cache: dict[str, dict] = {}
        self._probed: set[Path] = set()
        self._lock = threading.Lock()
        self._build()

    # ── static scan ──────────────────────────────────────────────────────────
    def _alias(self, alias: str, script: Path) -> None:
        self.by_name.setdefault(alias, script)
        self.by_norm.setdefault(_normalize(alias), []).append(script)

    def _build(self) -> None:
        for s in self.scripts:
            stem = s.stem[len("generate_"):-len("_training")]
            self._alias(s.stem, s)
            self._alias(stem, s)
            src = s.read_text(errors="replace")
            # Client IDs and source titles can differ substantially from a
            # generator's shorter filename/corpus ID. Register declared aliases
            # without importing the generator or guessing by substring.
            for literal in _RE_GAME_ALIAS.findall(src):
                try:
                    alias = ast.literal_eval(literal)
                except (SyntaxError, ValueError):
                    continue
                if alias:
                    self._alias(alias, s)
            # Register the actual environment ID, including its version. Partial
            # MiniGrid generators share env_id with full-observation generators,
            # so only register their prefixed client name.
            for env_id in dict.fromkeys(_RE_ENV_ID.findall(src)):
                if env_id.startswith("MiniGrid-"):
                    prefix = "partial:" if stem.startswith("minigrid_partial_") else ""
                    self._alias(prefix + env_id, s)
                elif env_id.startswith("Gym-Gridworlds/"):
                    self._alias(env_id, s)
                    self._alias("gymgw:" + env_id, s)
            ids = _RE_GAME_ID.findall(src) + _RE_GAME_ID_CONST.findall(src)
            if ids:
                self.game_ids[s] = ids[0]
                for gid in dict.fromkeys(ids):
                    self._alias(gid, s)
            m = _RE_MAIN_CALL.search(src)
            if m:
                target = m.group(1)
                target = target[:-len(".main")] if target.endswith(".main") else target
                names = [target] + [a.strip() for a in m.group(2).split(",") if a.strip()]
                self.main_names[s] = [n for n in names if re.fullmatch(r"[\w.]+", n)]
        self._load_cache()
        for s in self.scripts:
            gid = self.cached_id(s)
            if gid:
                self._alias(gid, s)

    # ── probe cache (import the module and ask it) ───────────────────────────
    def _load_cache(self) -> None:
        try:
            self._probe_cache = json.loads(INDEX_CACHE.read_text())
        except Exception:                        # noqa: BLE001 -- cache is advisory
            self._probe_cache = {}

    def save(self) -> None:
        """Persist whatever ids have been probed this run."""
        self._save_cache()

    def _save_cache(self) -> None:
        try:
            INDEX_CACHE.write_text(json.dumps(self._probe_cache, indent=1, sort_keys=True))
        except Exception:                        # noqa: BLE001
            pass

    @staticmethod
    def _stamp(script: Path) -> str:
        st = script.stat()
        return f"{int(st.st_mtime)}:{st.st_size}"

    def probe(self, script: Path, python: str) -> str | None:
        """``game_id`` by importing the generator module (result cached on disk).

        A FAILED probe is recorded with its error but is NOT treated as an
        answer: it is re-run on the next invocation, so fixing the environment
        (installing a package, pointing --python at the right interpreter) is
        enough -- no --refresh-index needed."""
        key = str(script.relative_to(REPO))
        stamp = self._stamp(script)
        hit = self._probe_cache.get(key)
        if hit and hit.get("stamp") == stamp and hit.get("game_id"):
            return hit.get("game_id")
        cmd = [python, "-c", _PROBE_SRC, str(script),
               ",".join(self.main_names.get(script, []))]
        gid, err = None, None
        try:
            r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=300)
            try:
                gid = json.loads(r.stdout.strip().splitlines()[-1])["game_id"]
            except Exception:                    # noqa: BLE001 -- it did not get that far
                tail = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
                err = tail[-1] if tail else f"exit {r.returncode}, no output"
            if gid is None and err is None:
                err = "imported cleanly but no object declared a game_id"
        except subprocess.TimeoutExpired:
            err = "import timed out after 300s"
        except Exception as exc:                 # noqa: BLE001 -- could not even launch it
            err = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._probe_cache[key] = {"stamp": stamp, "game_id": gid,
                                      **({"error": err} if err else {})}
        return gid

    def probe_error(self, script: Path) -> str | None:
        """Why the last probe of ``script`` failed (None if it succeeded)."""
        hit = self._probe_cache.get(str(script.relative_to(REPO)))
        return hit.get("error") if hit else None

    def probe_all(self, python: str, workers: int = 4) -> None:
        """Fill the cache for every generator whose id isn't statically visible."""
        # NB: filter on the CACHE, not `probe()` -- probing here would run every
        # import serially and defeat the pool below.
        todo = [s for s in self.scripts
                if s not in self.game_ids and s not in self._probed
                and not self._is_cached(s)]
        if not todo:
            self._save_cache()
            return
        log(f"Indexing {len(todo)} generator(s) whose game_id is computed at import "
            f"time (one-off, cached in {INDEX_CACHE.name}) ...")
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = {pool.submit(self.probe, s, python): s for s in todo}
            for f in as_completed(futs):
                s = futs[f]
                self._probed.add(s)
                gid = f.result()
                if gid:
                    self._alias(gid, s)
        self._save_cache()

    def _is_cached(self, script: Path) -> bool:
        """Has this script been probed SUCCESSFULLY at its CURRENT mtime/size?
        A failed probe does not count -- it is retried every run, so a fixed
        environment takes effect without --refresh-index."""
        hit = self._probe_cache.get(str(script.relative_to(REPO)))
        return bool(hit and hit.get("stamp") == self._stamp(script) and hit.get("game_id"))

    def cached_id(self, script: Path) -> str | None:
        hit = self._probe_cache.get(str(script.relative_to(REPO)))
        if hit and hit.get("stamp") == self._stamp(script):
            return hit.get("game_id")
        return None

    def game_id_of(self, script: Path, python: str) -> str | None:
        return self.game_ids.get(script) or self.cached_id(script) or self.probe(script, python)

    def _why(self, script: Path, python: str) -> str:
        """The reason a probe produced no game_id, for the user-facing error."""
        err = self.probe_error(script) or "no game_id and no error reported"
        return f"probing it with {python} failed: {err}"

    # ── resolution ───────────────────────────────────────────────────────────
    def resolve(self, name: str, python: str) -> tuple[Path, str]:
        """(script, game_id) for a user-supplied name. Raises ``KeyError``."""
        raw = name.strip()
        # 1. an explicit path to a generator
        if raw.endswith(".py"):
            p = Path(raw)
            p = p if p.is_absolute() else (REPO / p)
            if not p.exists():
                raise KeyError(f"{name!r}: no such generator script")
            gid = self.game_id_of(p, python)
            if not gid:
                raise KeyError(f"{name!r}: could not determine its game_id "
                               f"-- {self._why(p, python)}")
            return p, gid
        # 2. exact alias (file stem or a statically-declared game_id)
        script = self.by_name.get(raw)
        # 3. normalized alias (gymgw_ vs gymgridworlds_, puzzlescript_ prefix, ...)
        if script is None:
            cands = list(dict.fromkeys(self.by_norm.get(_normalize(raw), [])))
            if len(cands) > 1:
                raise KeyError(f"{name!r} is ambiguous: {[c.name for c in cands]}")
            script = cands[0] if cands else None
        # 4. import-probe the families whose game_id is derived from env_id
        if script is None:
            self.probe_all(python)
            script = self.by_name.get(raw)
            if script is None:
                cands = list(dict.fromkeys(self.by_norm.get(_normalize(raw), [])))
                if len(cands) > 1:
                    raise KeyError(f"{name!r} is ambiguous: {[c.name for c in cands]}")
                script = cands[0] if cands else None
        if script is None:
            near = sorted({s.stem[len("generate_"):-len("_training")]
                           for k, v in self.by_norm.items() for s in v
                           if _normalize(raw) in k or k in _normalize(raw)})[:6]
            hint = f" (did you mean: {', '.join(near)}?)" if near else ""
            raise KeyError(f"{name!r}: no generator found{hint}")
        gid = self.game_id_of(script, python)
        if not gid:
            raise KeyError(f"{name!r}: {script.name} -- {self._why(script, python)}")
        return script, gid


# ---------------------------------------------------------------------------
# The game list
# ---------------------------------------------------------------------------
@dataclass
class Entry:
    name: str
    episodes: int | None = None
    start_seed: int | None = None
    seed: int | None = None
    out: str | None = None
    args: list[str] = field(default_factory=list)


_HEADER_KEYS = {"game", "game_id", "name", "games"}


def read_game_list(path: Path) -> list[Entry]:
    """Parse the CSV. Header optional; blank lines and ``#`` comments skipped."""
    rows = []
    with Path(path).open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            cells = [c.strip() for c in row]
            if not cells[0] or cells[0].startswith("#"):
                continue
            rows.append(cells)
    if not rows:
        raise SystemExit(f"{path}: no games listed")

    header = None
    if rows[0][0].lower() in _HEADER_KEYS:
        header = [c.lower() for c in rows[0]]
        rows = rows[1:]

    def as_int(v):
        v = (v or "").strip()
        return int(v) if v.isdigit() else None

    entries: list[Entry] = []
    for cells in rows:
        if header:
            rec = {header[i]: cells[i] for i in range(min(len(header), len(cells)))}
            name = (rec.get("game") or rec.get("game_id") or rec.get("name")
                    or rec.get("games") or "")
            if not name:
                continue
            extra = (rec.get("args") or rec.get("extra_args") or "").strip()
            entries.append(Entry(
                name=name,
                episodes=as_int(rec.get("episodes")),
                start_seed=as_int(rec.get("start_seed")),
                seed=as_int(rec.get("seed")),
                out=(rec.get("out") or None),
                args=extra.split() if extra else []))
        else:
            name, rest = cells[0], cells[1:]
            episodes = as_int(rest[0]) if rest else None
            if episodes is not None:
                rest = rest[1:]
            extra = " ".join(c for c in rest if c).strip()
            entries.append(Entry(name=name, episodes=episodes,
                                 args=extra.split() if extra else []))
    return entries


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
@dataclass
class Job:
    name: str
    script: Path
    game_id: str
    out_dir: Path
    episodes: int
    start_seed: int
    seed: int
    extra: list[str]
    log_path: Path

    def gen_cmd(self, python: str) -> list[str]:
        cmd = [python, str(self.script.relative_to(REPO)),
               "--episodes", str(self.episodes),
               "--out", str(self.out_dir),
               "--start-seed", str(self.start_seed),
               "--seed", str(self.seed)]
        return cmd + self.extra

    def compress_cmd(self, python: str, delete: bool, jobs: int) -> list[str]:
        cmd = [python, str(CONVERTER.relative_to(REPO)),
               "--game", self.out_dir.name,
               "--root", str(self.out_dir.parent),
               "--jobs", str(max(1, jobs))]
        return cmd + (["--delete"] if delete else [])


@dataclass
class Result:
    job: Job
    status: str = "pending"      # ok | partial | failed | timeout | skipped | compress-failed
    gen_rc: int | None = None
    comp_rc: int | None = None
    written: int = 0             # episode_*.json produced by the generator
    compressed: int = 0          # episode_*.ep.zst present afterwards
    leftover: int = 0            # episode_*.json that survived compression
    compress_ok: bool = True
    seconds: float = 0.0
    note: str = ""

    def add_note(self, msg: str) -> None:
        self.note = f"{self.note}; {msg}" if self.note else msg


def _count(out_dir: Path, pattern: str) -> int:
    return sum(1 for _ in out_dir.glob(pattern)) if out_dir.is_dir() else 0


def _run(cmd: list[str], log_fh, timeout: int | None) -> int:
    """Run a child process with its output tee'd into ``log_fh``. Returns the
    exit code; -9 on timeout, -2 when the run was aborted (Ctrl-C)."""
    if _ABORT.is_set():
        return -2
    log_fh.write(f"\n$ {' '.join(cmd)}\n")
    log_fh.flush()
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=log_fh,
                            stderr=subprocess.STDOUT, text=True)
    _LIVE_PROCS.add(proc)
    try:
        return proc.wait(timeout=timeout or None)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        log_fh.write(f"\n!! timed out after {timeout}s -- killed\n")
        return -9
    finally:
        _LIVE_PROCS.discard(proc)


def run_job(job: Job, opts) -> Result:
    res = Result(job=job)
    t0 = time.time()

    have_zst = _count(job.out_dir, "episode_*.ep.zst")
    have_json = _count(job.out_dir, "episode_*.json")
    if opts.skip_existing and (have_zst + have_json) >= job.episodes:
        res.status = "skipped"
        res.compressed, res.written = have_zst, have_json
        res.note = f"already has {have_zst + have_json} episodes"
        log(f"[skip] {job.game_id}: {res.note}")
        return res
    if _ABORT.is_set():
        res.status = "failed"
        res.note = "aborted"
        return res

    job.out_dir.mkdir(parents=True, exist_ok=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"[start] {job.game_id}  ({job.episodes} episodes)  log -> "
        f"{job.log_path.relative_to(REPO) if job.log_path.is_relative_to(REPO) else job.log_path}")

    with job.log_path.open("w") as fh:
        fh.write(f"# {job.name} -> {job.script.name} (game_id={job.game_id})\n")
        fh.write(f"# started {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        res.gen_rc = _run(job.gen_cmd(opts.python), fh, opts.timeout)
        res.written = _count(job.out_dir, "episode_*.json") - have_json
        produced = _count(job.out_dir, "episode_*.json")

        if res.gen_rc == 0:
            res.status = "ok"
        elif res.gen_rc == -9:
            res.status = "timeout"
        elif res.gen_rc == -2:
            res.status = "failed"
            res.note = "aborted"
        elif produced:
            res.status = "partial"          # generator fell short but wrote episodes
            res.add_note(f"generator rc={res.gen_rc}")
        else:
            res.status = "failed"
            res.add_note(f"generator rc={res.gen_rc}, no episodes written")

        # COMPRESS + DELETE. Whatever the generator's exit code, the episodes it
        # did write are valid, so they get converted; `--delete` only removes a
        # .json once its round-trip through the codec has verified.
        if produced and not opts.no_compress and not _ABORT.is_set():
            res.comp_rc = _run(job.compress_cmd(opts.compress_python,
                                                delete=not opts.keep_json,
                                                jobs=opts.compress_jobs), fh, opts.timeout)
            res.leftover = _count(job.out_dir, "episode_*.json")
            if res.comp_rc != 0:
                res.compress_ok = False
                res.add_note(f"convert rc={res.comp_rc}")
            elif res.leftover and not opts.keep_json:
                # A .json only survives --delete when its round-trip did NOT
                # verify. After a kill (timeout / Ctrl-C) that is usually just the
                # half-written file the generator was in the middle of; otherwise
                # it is a real conversion failure. Either way it is KEPT, not
                # silently dropped -- the log names the file.
                res.compress_ok = False
                res.add_note(f"{res.leftover} .json failed to convert (kept)")
            if not res.compress_ok and res.status in ("ok", "partial"):
                res.status = "compress-failed"
        res.compressed = _count(job.out_dir, "episode_*.ep.zst")
        fh.write(f"\n# finished {datetime.now():%Y-%m-%d %H:%M:%S} status={res.status}\n")

    res.seconds = time.time() - t0
    log(f"[{res.status}] {job.game_id}  +{res.written} episodes  "
        f"{res.compressed} .ep.zst  {res.seconds/60:.1f} min"
        + (f"  ({res.note})" if res.note else ""))
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game_list", "--game-list", dest="game_list", type=Path,
                   help="CSV listing the games to generate (one per line).")
    p.add_argument("--episodes", type=int, default=1000,
                   help="WIN episodes per game (a CSV `episodes` column wins).")
    p.add_argument("--n_workers", "--n-workers", dest="n_workers", type=int, default=4,
                   help="Games generated concurrently (one subprocess each).")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                   help="Corpus root; each game lands in <root>/<game_id>.")
    p.add_argument("--start-seed", type=int, default=0,
                   help="First seed each generator tries (extend a corpus with this).")
    p.add_argument("--seed", type=int, default=0,
                   help="Generator RNG seed (exploration / stochastic-optimal).")
    p.add_argument("--extra-args", default="",
                   help="Extra args appended to EVERY generator, e.g. \"--noise 0.2\".")
    p.add_argument("--timeout", type=int, default=0,
                   help="Per-game wall-clock limit in seconds (0 = none).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip a game that already holds >= --episodes files.")
    p.add_argument("--no-compress", action="store_true",
                   help="Leave the JSON uncompressed (default: compress + delete).")
    p.add_argument("--keep-json", action="store_true",
                   help="Compress but KEEP the .json originals (no --delete).")
    p.add_argument("--compress-jobs", type=int, default=1,
                   help="Worker processes inside utils/convert_training_data.py, per game.")
    p.add_argument("--python", default=default_gen_python(),
                   help="Interpreter for the generators (needs arcengine).")
    p.add_argument("--compress-python", default=default_compress_python(),
                   help="Interpreter for utils/convert_training_data.py (needs numpy and zstandard).")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="Where per-game logs go (default logs/master_data_generator/<ts>).")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve everything and print the commands; run nothing.")
    p.add_argument("--list-games", action="store_true",
                   help="Print every generator and its game_id, then exit.")
    p.add_argument("--refresh-index", action="store_true",
                   help="Re-probe every generator's game_id (rebuilds the cache).")
    return p


def _install_sigint(pool: ThreadPoolExecutor | None = None) -> None:
    def handler(signum, frame):                  # noqa: ARG001
        if _ABORT.is_set():
            raise KeyboardInterrupt
        _ABORT.set()
        log("\n!! interrupt: killing running generators (Ctrl-C again to force)")
        for proc in list(_LIVE_PROCS):
            try:
                proc.terminate()
            except Exception:                    # noqa: BLE001
                pass
    signal.signal(signal.SIGINT, handler)


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    index = GeneratorIndex()

    if args.refresh_index:
        INDEX_CACHE.unlink(missing_ok=True)
        index = GeneratorIndex()
        index.probe_all(args.python, workers=max(1, args.n_workers))

    if args.list_games:
        index.probe_all(args.python, workers=max(1, args.n_workers))
        for s in index.scripts:
            gid = index.game_id_of(s, args.python) or "?"
            print(f"{gid:44s} {s.relative_to(REPO)}")
        print(f"\n{len(index.scripts)} generators")
        return 0

    if not args.game_list:
        build_argparser().error("--game_list is required (or use --list-games)")
    if not args.game_list.exists():
        build_argparser().error(f"no such game list: {args.game_list}")

    entries = read_game_list(args.game_list)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir or (REPO / "logs" / "master_data_generator" / stamp)

    jobs: list[Job] = []
    unresolved: list[str] = []
    for e in entries:
        try:
            script, gid = index.resolve(e.name, args.python)
        except KeyError as err:
            unresolved.append(err.args[0])        # str(KeyError) re-quotes the message
            continue
        out_dir = Path(e.out) if e.out else (args.out_root / gid)
        if not out_dir.is_absolute():
            out_dir = REPO / out_dir
        jobs.append(Job(
            name=e.name, script=script, game_id=gid, out_dir=out_dir,
            episodes=e.episodes if e.episodes is not None else args.episodes,
            start_seed=e.start_seed if e.start_seed is not None else args.start_seed,
            seed=e.seed if e.seed is not None else args.seed,
            extra=(args.extra_args.split() if args.extra_args else []) + e.args,
            log_path=log_dir / f"{gid}.log"))
    index.save()

    if unresolved:
        log("Unresolved games:")
        for u in unresolved:
            log(f"  !! {u}")
        if not jobs:
            return 2

    dup = {j.out_dir for j in jobs}
    if len(dup) != len(jobs):
        seen, clash = set(), set()
        for j in jobs:
            (clash if j.out_dir in seen else seen).add(j.out_dir)
        log(f"!! two entries target the same output dir: "
            f"{sorted(str(c) for c in clash)} -- they would overwrite each other")
        return 2

    log(f"{len(jobs)} game(s), {args.n_workers} worker(s), "
        f"{args.episodes} episodes each (unless overridden)")
    log(f"Generator python : {args.python}")
    log(f"Compress python  : {args.compress_python}"
        + ("  (compression DISABLED)" if args.no_compress else ""))
    log(f"Output root      : {args.out_root}")
    log(f"Logs             : {log_dir}\n")

    if args.dry_run:
        for j in jobs:
            print(f"# {j.name} -> {j.game_id}")
            print("  " + " ".join(j.gen_cmd(args.python)))
            if not args.no_compress:
                print("  " + " ".join(j.compress_cmd(args.compress_python,
                                                     delete=not args.keep_json,
                                                     jobs=args.compress_jobs)))
        print("\n(dry run: nothing executed)")
        return 2 if unresolved else 0

    log_dir.mkdir(parents=True, exist_ok=True)
    _install_sigint()

    t0 = time.time()
    results: list[Result] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.n_workers)) as pool:
        futs = {pool.submit(run_job, j, args): j for j in jobs}
        for f in as_completed(futs):
            done += 1
            try:
                results.append(f.result())
            except Exception as err:             # noqa: BLE001 -- one game must not sink the sweep
                j = futs[f]
                results.append(Result(job=j, status="failed", note=repr(err)))
                log(f"[failed] {j.game_id}: {err!r}")
            log(f"  ... {done}/{len(jobs)} games finished "
                f"({(time.time() - t0)/60:.1f} min elapsed)")

    # ── summary ──────────────────────────────────────────────────────────────
    order = {"failed": 0, "compress-failed": 1, "timeout": 2, "partial": 3,
             "skipped": 4, "ok": 5}
    results.sort(key=lambda r: (order.get(r.status, 9), r.job.game_id))
    hdr = f"\n{'game_id':36s} {'status':16s} {'new':>6} {'.ep.zst':>8} {'min':>7}  note"
    print(hdr)
    print("-" * (len(hdr) + 10))
    for r in results:
        print(f"{r.job.game_id:36s} {r.status:16s} {r.written:6d} {r.compressed:8d} "
              f"{r.seconds/60:7.1f}  {r.note}")
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print("-" * (len(hdr) + 10))
    print(f"{len(results)} games in {(time.time() - t0)/60:.1f} min: "
          + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if unresolved:
        print(f"{len(unresolved)} unresolved name(s) -- see above")
    keeping_json = args.keep_json or args.no_compress
    leftover = [r for r in results if r.leftover and not keeping_json]
    if leftover:
        print("Uncompressed .json left behind (conversion did not verify): "
              + ", ".join(f"{r.job.game_id}({r.leftover})" for r in leftover)
              + " -- after an interrupted run these are usually the truncated "
                "file the kill landed on; delete them and re-run that game.")
    print(f"Logs: {log_dir}")

    bad = sum(1 for r in results
              if r.status in ("failed", "timeout") or not r.compress_ok)
    return 1 if (bad or unresolved) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
