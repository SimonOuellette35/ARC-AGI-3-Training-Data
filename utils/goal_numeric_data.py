"""Goal A: a small index over the policy cache; observations stay numeric."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import time

import numpy as np

from utils.goal_hypothesizer import read_programs, split_for_game


FORMAT = 'policy-input-goal-v1'
INPUT_KEYS = ('frames', 'act', 'xy', 'mask', 'nframes', 'frame_step', 'frame_slot', 'frame_mask')


def prepare(args):
    from filelock import FileLock
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output_dir / 'numeric.prepare.lock')):
        _prepare(args)


def _prepare(args):
    from train_goal_hypothesizer_A import resolve_labels
    from utils.policy_training import CACHE_VERSION

    if not 0 <= args.val_fraction < 1 or args.samples_per_level < 1 or args.max_episodes_per_game < 0:
        raise ValueError('Invalid sampling/split settings')
    root = args.cache_dir.resolve()
    games = sorted(p for p in root.iterdir() if p.is_dir())
    if args.games:
        wanted = set(args.games)
        games = [p for p in games if p.name in wanted]
        if wanted - {p.name for p in games}:
            raise ValueError('Requested games are absent from the policy cache')
    if not games:
        raise ValueError('No policy cache: run train_policy_dynamics.py preprocess first')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'numeric.sqlite3'
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ('resume', 'command', 'output_dir')}
    settings['cache_dir'] = str(root)
    settings['format'] = FORMAT
    with closing(sqlite3.connect(path)) as db:
        db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS programs (id INTEGER PRIMARY KEY, source TEXT UNIQUE);
            CREATE TABLE IF NOT EXISTS episodes (path TEXT PRIMARY KEY, size INTEGER, mtime INTEGER);
            CREATE TABLE IF NOT EXISTS samples (id INTEGER PRIMARY KEY, split TEXT,
                path TEXT, level INTEGER, decisions TEXT, target INTEGER,
                sample_start INTEGER, sample_end INTEGER);
            CREATE INDEX IF NOT EXISTS sample_lookup ON samples(split, sample_end);
        ''')
        encoded = json.dumps(settings, sort_keys=True)
        previous = db.execute("SELECT value FROM metadata WHERE key='settings'").fetchone()
        if previous and previous[0] != encoded:
            raise ValueError('Numeric index settings changed; use a different output directory')
        if previous and not args.resume:
            raise ValueError('Numeric index exists; use --resume')
        db.execute("INSERT OR IGNORE INTO metadata VALUES ('settings', ?)", (encoded,))
        db.execute("DELETE FROM metadata WHERE key='complete'")
        counts = {s: db.execute('SELECT COALESCE(MAX(sample_end),0) FROM samples WHERE split=?',
                               (s,)).fetchone()[0] for s in ('train', 'validation')}
        db.commit()
        report = time.monotonic()
        processed = 0
        for game in games:
            label = resolve_labels(game.name, args.goal_root, args.demos_root)
            if label is None:
                print(f'[index] {game.name}: no program labels; skipped', flush=True)
                continue
            digest = hashlib.sha256(label.read_bytes()).hexdigest()
            key = 'labels:' + game.name
            old = db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
            if old and old[0] != digest:
                raise ValueError(f'Labels changed for {game.name}')
            db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', (key, digest))
            programs, universal = read_programs(label)
            targets = {}
            for lid, source in programs.items():
                db.execute('INSERT OR IGNORE INTO programs(source) VALUES (?)', (source,))
                targets[lid] = db.execute('SELECT id FROM programs WHERE source=?', (source,)).fetchone()[0]
            db.commit()
            split = split_for_game(game.name, args.seed, args.val_fraction)
            paths = sorted(game.glob('*.npz'))
            if args.max_episodes_per_game:
                paths = paths[:args.max_episodes_per_game]
            for file in paths:
                relative = str(file.relative_to(root))
                stat = file.stat()
                old = db.execute('SELECT size,mtime FROM episodes WHERE path=?', (relative,)).fetchone()
                if old:
                    if old != (stat.st_size, stat.st_mtime_ns):
                        raise ValueError(f'Policy cache changed: {file}')
                    continue
                # Read only tiny metadata/action-count arrays, never obs_*.
                with np.load(file, allow_pickle=False) as z:
                    if int(z['cache_version']) != CACHE_VERSION:
                        raise ValueError(f'Unsupported policy cache version: {file}')
                    for li in range(int(z['n_levels'])):
                        lid = int(z[f'lid_{li}'])
                        target = next(iter(targets.values())) if universal else targets.get(lid)
                        eligible = len(z[f'nobs_{li}']) - 1
                        if target is None or eligible < 1:
                            continue
                        rng = random.Random(f'{args.seed}:{relative}:{li}')
                        decisions = sorted(rng.sample(range(eligible), min(eligible, args.samples_per_level)))
                        start = counts[split]
                        counts[split] += len(decisions)
                        db.execute('INSERT INTO samples(split,path,level,decisions,target,sample_start,sample_end) '
                                   'VALUES (?,?,?,?,?,?,?)',
                                   (split, relative, li, json.dumps(decisions), target, start, counts[split]))
                db.execute('INSERT INTO episodes VALUES (?,?,?)', (relative, stat.st_size, stat.st_mtime_ns))
                processed += 1
                if processed % 128 == 0:
                    db.commit()  # bounded resume batches, without an fsync per episode
                if time.monotonic() - report >= 30:
                    print(f'[index] {processed:,} new episodes; examples={counts}; '
                          f'index={path.stat().st_size / 1e6:.1f} MB', flush=True)
                    report = time.monotonic()
            db.commit()
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('complete','true')")
        db.commit()
    print(f'[index] complete: {counts}; {path.stat().st_size / 1e6:.1f} MB; '
          'observations reused from policy cache; no input tokenization', flush=True)


class NumericGoalDataset:
    def __init__(self, directory, split, cfg, *, predictor=None, cache_dir=None):
        self.path = Path(directory) / 'numeric.sqlite3'
        if not self.path.exists():
            raise ValueError('A now requires numeric.sqlite3. Run A prepare with --cache-dir data/cache_policy; '
                             'legacy text/token caches are not numeric model inputs.')
        self.split, self.cfg, self.predictor = split, cfg, predictor
        with self.connect() as db:
            if not db.execute("SELECT value FROM metadata WHERE key='complete'").fetchone():
                raise ValueError('Index is incomplete; resume preparation first')
            self.settings = json.loads(db.execute("SELECT value FROM metadata WHERE key='settings'").fetchone()[0])
            self.length = db.execute('SELECT COALESCE(MAX(sample_end),0) FROM samples WHERE split=?',
                                     (split,)).fetchone()[0]
            self.programs = dict(db.execute('SELECT id,source FROM programs'))
        self.root = Path(cache_dir or self.settings['cache_dir'])

    def connect(self):
        return closing(sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        from utils.policy_inputs import policy_inputs
        if index < 0 or index >= self.length:
            raise IndexError(index)
        with self.connect() as db:
            path, li, decisions, target, start = db.execute(
                'SELECT path,level,decisions,target,sample_start FROM samples '
                'WHERE split=? AND sample_end>? ORDER BY sample_end LIMIT 1', (self.split, index)).fetchone()
            size, mtime = db.execute('SELECT size,mtime FROM episodes WHERE path=?', (path,)).fetchone()
        file = self.root / path
        stat = file.stat()
        if (stat.st_size, stat.st_mtime_ns) != (size, mtime):
            raise ValueError(f'Policy cache changed since indexing: {file}')
        end = json.loads(decisions)[index - start]
        spans, incoming = [], []
        with np.load(file, allow_pickle=False) as z:
            # Work backwards so only levels needed by the FIFO are decompressed.
            future = []
            for level in range(li, -1, -1):
                nobs = z[f'nobs_{level}'].astype(np.int64)
                count = end + 1 if level == li else len(nobs) - 1
                first = max(0, count - (self.cfg.max_states - len(spans)))
                offsets = np.concatenate(([0], nobs.cumsum()))
                obs, act, xy = z[f'obs_{level}'], z[f'act_{level}'], z[f'axy_{level}']
                def action(k):
                    return {'index': int(act[k]), 'data': dict(x=int(xy[k, 0]), y=int(xy[k, 1]))}
                parts = [obs[offsets[k]:offsets[k + 1]] for k in range(first, count)]
                spans = parts + spans
                incoming = [action(k) for k in range(first, count)] + incoming
                if level == li and self.predictor is not None:
                    future = [action(k) for k in range(end + 1, min(len(act), end + 1 + self.predictor.max_depth))]
                if len(spans) >= self.cfg.max_states:
                    break
        if not self.settings['no_action_shuffle']:
            from utils.goal_session_format import action_id_remap, apply_action_remap
            remap = action_id_remap(random.Random(f'{self.settings["seed"]}:{path}:{li}:{end}:actions'))
            incoming, future = apply_action_remap(incoming, remap), apply_action_remap(future, remap)
        if self.predictor is not None and future:
            rng = self.predictor.selected_rng(f'numeric:{path}:{li}:{end}')
            if rng is not None:
                future = future[:rng.randint(1, len(future))]
                spans += self.predictor.rollout(spans, incoming, future)
                incoming += future
        acts = [a['index'] for a in incoming[1:]]
        coords = [(a['data']['x'], a['data']['y']) if a['index'] == 6 else (255, 255)
                  for a in incoming[1:]]
        packed = policy_inputs(self.cfg, spans, acts, coords, 'cpu')
        return {**{key: value[0] for key, value in packed.items()}, 'program': self.programs[target]}


def collate_numeric(examples):
    from torch.nn.utils.rnn import pad_sequence
    return {**{key: pad_sequence([e[key] for e in examples], batch_first=True,
                                padding_value=255 if key == 'xy' else 0) for key in INPUT_KEYS},
            'programs': [e['program'] for e in examples]}
