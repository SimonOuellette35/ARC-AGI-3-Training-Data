"""Losslessly compressed prepared records with bounded, atomic append batches.

SQLite stores compressed evidence per row and references shared target programs.
No expanded copy is needed to resume preparation or stream into the tokenizer.
Only the current
transaction (at most 128 records) can be lost when preparation is interrupted.
"""
from contextlib import closing
from collections import OrderedDict
import json
from pathlib import Path
import sqlite3
import time
import zlib


SPLITS = ('train', 'validation', 'validation_mixed')
APPLICATION_ID = 0x474F414C


class CompressedRecords:
    def __init__(self, path, *, writable=False):
        self.path = Path(path)
        uri = self.path.resolve().as_uri() + ('?mode=rwc' if writable else '?mode=ro')
        self.db = sqlite3.connect(uri, uri=True)
        try:
            if writable and self.db.execute('PRAGMA application_id').fetchone()[0] == 0:
                if self.db.execute('SELECT name FROM sqlite_master').fetchone():
                    raise ValueError(f'Not a prepared-record database: {path}')
                self.db.execute('BEGIN')
                self.db.execute('CREATE TABLE targets (id INTEGER PRIMARY KEY, program BLOB NOT NULL UNIQUE)')
                self.db.execute('CREATE TABLE records (id INTEGER PRIMARY KEY, payload BLOB NOT NULL, '
                                'target_id INTEGER NOT NULL REFERENCES targets(id))')
                self.db.execute(f'PRAGMA application_id={APPLICATION_ID}')
                self.db.execute('PRAGMA user_version=1')
                self.db.commit()
            if (self.db.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID or
                    self.db.execute('PRAGMA user_version').fetchone()[0] != 1):
                raise ValueError(f'Unknown prepared-record database format: {path}')
            # Default rollback journal and FULL sync keep committed rows recoverable
            # after process termination or disk-full errors, without a growing WAL.
            self.db.execute('PRAGMA foreign_keys=ON')
            self.count = self.db.execute('SELECT COALESCE(MAX(id), 0) FROM records').fetchone()[0]
            self.position = 0
            self.pending = 0
            self.target_ids = OrderedDict()
            self.programs = OrderedDict()
        except BaseException:
            self.db.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                self.db.commit()
            else:
                self.db.rollback()
        finally:
            self.db.close()

    def read_next(self):
        if self.position >= self.count:
            return None
        self.position += 1
        row = self.db.execute('SELECT payload, target_id FROM records WHERE id=?', (self.position,)).fetchone()
        if row is None:
            raise ValueError(f'Missing record {self.position} in {self.path}')
        return self.decode(*row)

    @staticmethod
    def _remember(cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > 256:
            cache.popitem(last=False)

    def decode(self, payload, target_id):
        if target_id not in self.programs:
            row = self.db.execute('SELECT program FROM targets WHERE id=?', (target_id,)).fetchone()
            if row is None:
                raise ValueError(f'Missing target {target_id} in {self.path}')
            self._remember(self.programs, target_id, zlib.decompress(row[0]).decode('utf-8'))
        self.programs.move_to_end(target_id)
        record = json.loads(zlib.decompress(payload))
        record['completion'] = self.programs[target_id]
        return record

    def append(self, record):
        evidence = dict(record)
        program = evidence.pop('completion')
        if program not in self.target_ids:
            compressed = zlib.compress(program.encode('utf-8'))
            self.db.execute('INSERT OR IGNORE INTO targets(program) VALUES (?)', (compressed,))
            target_id = self.db.execute('SELECT id FROM targets WHERE program=?', (compressed,)).fetchone()[0]
            self._remember(self.target_ids, program, target_id)
        self.target_ids.move_to_end(program)
        payload = zlib.compress(json.dumps(evidence, separators=(',', ':')).encode('utf-8'))
        self.db.execute('INSERT INTO records(payload, target_id) VALUES (?, ?)',
                        (payload, self.target_ids[program]))
        self.pending += 1
        if self.pending >= 128:
            self.db.commit()
            self.pending = 0


def prepared_path(directory, split):
    paths = [Path(directory) / f'{split}{suffix}' for suffix in ('.sqlite3', '.jsonl')]
    existing = [path for path in paths if path.exists()]
    if len(existing) > 1:
        raise ValueError(f'Both storage formats exist for {split} in {directory}; '
                         'move the JSONL backup elsewhere, or run compact --remove-source')
    return existing[0] if existing else None


def iter_prepared_records(path):
    if Path(path).suffix != '.jsonl':
        with CompressedRecords(path) as store:
            for row in store.db.execute('SELECT payload, target_id FROM records ORDER BY id'):
                yield store.decode(*row)
    else:
        with Path(path).open(encoding='utf-8') as source:
            for line in source:
                yield json.loads(line)


def compact_prepared(directory, *, remove_source=False):
    """Convert complete legacy rows, preserving saved predictions and row order.

    Interrupted conversions replay/validate the committed prefix. The original is
    removed only on explicit request, after verifying all rows and closing the DB.
    An unterminated final JSONL row is discarded just as in preparation resume.
    """
    found = False
    for split in SPLITS:
        source = Path(directory) / f'{split}.jsonl'
        if not source.exists():
            continue
        found = True
        destination = source.with_suffix('.sqlite3')
        temporary = source.with_suffix('.sqlite3.incomplete')
        published = destination.exists()
        target = destination if published else temporary
        before = source.stat()
        count = 0
        report_at = time.monotonic() + 30
        print(f'[compact] {source} -> {destination}', flush=True)
        with CompressedRecords(target, writable=True) as store, source.open('rb') as stream:
            for line in stream:
                if not line.endswith(b'\n'):
                    print(f'[compact] ignoring incomplete final row in {source}', flush=True)
                    break
                record = json.loads(line)
                saved = store.read_next()
                if saved is None:
                    if published:
                        raise ValueError(f'Existing {destination} is shorter than {source}')
                    store.append(record)
                elif saved != record:
                    raise ValueError(f'Conversion mismatch in {source} at row {count + 1}')
                count += 1
                if time.monotonic() >= report_at:
                    print(f'[compact] {split}: {count} rows checked', flush=True)
                    report_at = time.monotonic() + 30
            if store.read_next() is not None:
                raise ValueError(f'Unconsumed converted rows in {target}')
        # Read back every stored row against the original before publishing/deleting.
        print(f'[compact] verifying {count} rows in {target}', flush=True)
        report_at = time.monotonic() + 30
        with source.open('rb') as stream, closing(iter_prepared_records(target)) as records:
            verified = 0
            for saved in records:
                if saved != json.loads(stream.readline()):
                    raise ValueError(f'Conversion verification failed for {source}')
                verified += 1
                if time.monotonic() >= report_at:
                    print(f'[compact] {split}: {verified}/{count} rows verified', flush=True)
                    report_at = time.monotonic() + 30
            if verified != count:
                raise ValueError(f'Conversion row count mismatch for {source}')
        after = source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f'Source changed during conversion: {source}; stop preparation first')
        if target == temporary:
            temporary.replace(destination)
        size = destination.stat().st_size
        print(f'[compact] {split}: {count} rows; {before.st_size:,} -> {size:,} bytes '
              f'({before.st_size / max(1, size):.1f}x smaller)', flush=True)
        if remove_source:
            source.unlink()
    if not found:
        raise ValueError(f'No prepared JSONL files found in {directory}')
