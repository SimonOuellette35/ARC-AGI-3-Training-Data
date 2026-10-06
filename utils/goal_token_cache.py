"""Parallel, resumable tokenization; only compact token IDs are stored on disk.

Workers read disjoint source ranges directly, avoiding large records over IPC.
An Arrow file is committed by publishing its JSON receipt last. Cache identity
includes source metadata and the actual tokenizer, not just a model name.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import closing
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import time

from utils.goal_prepared_data import CompressedRecords


_WORKER = None


def _initialize(tokenizer_dir, encoder_module, encoder_name, max_length):
    global _WORKER
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    encoder = getattr(importlib.import_module(encoder_module), encoder_name)
    _WORKER = tokenizer, encoder, max_length


def _records(path, start, stop):
    if path.suffix == '.sqlite3':
        with CompressedRecords(path) as store:
            for row in store.db.execute(
                    'SELECT payload, target_id FROM records WHERE id>? AND id<=? ORDER BY id',
                    (start, stop)):
                yield store.decode(*row)
    else:
        with path.open('rb') as stream:
            stream.seek(start)
            while stream.tell() < stop:
                yield json.loads(stream.readline())


def _atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value) + '\n')
    temporary.replace(path)


def _build_chunk(job, context=None):
    from datasets import Features, Sequence, Value
    from datasets.arrow_writer import ArrowWriter
    path, cache, index, start, stop, expected = job
    path, cache = Path(path), Path(cache)
    tokenizer, encoder, max_length = context or _WORKER
    arrow = cache / f'{index:07d}.arrow'
    temporary = arrow.with_suffix('.arrow.tmp')
    stats = dict(processed=0, kept=0, dropped=0, imagined=0, dropped_imagined=0, tokens=0)
    features = Features({'input_ids': Sequence(Value('int32')), 'prompt_length': Value('int32')})
    with ArrowWriter(path=str(temporary), features=features, writer_batch_size=32) as writer:
        with closing(_records(path, start, stop)) as records:
            for record in records:
                encoded = encoder(record, tokenizer, max_length)
                imagined = bool(record.get('imagined_rollout'))
                stats['processed'] += 1
                if encoded is None:
                    stats['dropped'] += 1
                    stats['dropped_imagined'] += imagined
                    continue
                # All current encoders mask a contiguous prompt, then supervise
                # the complete target. Reconstruct redundant columns at collation.
                labels = encoded['labels']
                prompt_length = next((i for i, token in enumerate(labels) if token != -100), len(labels))
                if (labels != [-100] * prompt_length + encoded['input_ids'][prompt_length:]
                        or encoded['attention_mask'] != [1] * len(labels)):
                    raise ValueError('Token cache requires completion-only labels and unpadded inputs')
                writer.write(dict(input_ids=encoded['input_ids'], prompt_length=prompt_length))
                stats['kept'] += 1
                stats['imagined'] += imagined
                stats['tokens'] += len(labels)
        writer.finalize()
    if stats['processed'] != expected:
        raise ValueError(f'Source range changed: expected {expected}, read {stats["processed"]}')
    temporary.replace(arrow)
    stats['bytes'] = arrow.stat().st_size
    _atomic_json(arrow.with_suffix('.json'), stats)
    return index, stats


def _plan(path, chunk_size):
    if path.suffix == '.sqlite3':
        with CompressedRecords(path) as store:
            count = store.count
        return [(start, min(start + chunk_size, count), min(chunk_size, count - start))
                for start in range(0, count, chunk_size)]
    # JSONL needs a one-time byte-offset index; no parsing/tokenization here.
    print(f'[tokenize] indexing JSONL byte ranges: {path}', flush=True)
    chunks, start, count = [], 0, 0
    report_at = time.monotonic() + 30
    with path.open('rb') as stream:
        while stream.readline():
            count += 1
            if count == chunk_size:
                stop = stream.tell()
                chunks.append((start, stop, count))
                start, count = stop, 0
            if time.monotonic() >= report_at:
                print(f'[tokenize] indexed {stream.tell():,}/{path.stat().st_size:,} bytes', flush=True)
                report_at = time.monotonic() + 30
        if count:
            chunks.append((start, stream.tell(), count))
    return chunks


def tokenized_dataset(path, tokenizer, encode_record, max_length, *, cache_dir,
                      identity, workers=0, chunk_size=2000):
    """Return a memory-mapped dataset; reuse completed chunks on every restart."""
    from datasets import Dataset, concatenate_datasets
    from filelock import FileLock
    if workers < 0 or chunk_size < 1:
        raise ValueError('workers must be >= 0 and chunk_size must be positive')
    if not workers:
        workers = min(8, len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1)
    path = Path(path)
    before = path.stat()
    stamp = (before.st_size, before.st_mtime_ns)
    backend = (tokenizer.backend_tokenizer.to_str() if getattr(tokenizer, 'is_fast', False)
               else repr(tokenizer))
    signature = json.dumps([str(path.resolve()), stamp, identity, max_length, chunk_size,
                            backend, tokenizer.chat_template, tokenizer.special_tokens_map,
                            'compact-token-cache-v1'], sort_keys=True)
    cache = Path(cache_dir) / hashlib.sha256(signature.encode()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    # Also serializes distributed ranks sharing this cache. CUDA is initialized
    # only after preprocessing by the caller; workers use spawn regardless.
    with FileLock(str(cache / 'build.lock')):
        plan_file = cache / 'plan.json'
        if not plan_file.exists():
            _atomic_json(plan_file, _plan(path, chunk_size))
        chunks = json.loads(plan_file.read_text())
        total = sum(chunk[2] for chunk in chunks)
        receipts, jobs = {}, []
        for index, (start, stop, expected) in enumerate(chunks):
            arrow = cache / f'{index:07d}.arrow'
            receipt = arrow.with_suffix('.json')
            if arrow.exists() and receipt.exists():
                stats = json.loads(receipt.read_text())
                if stats['processed'] != expected or stats['bytes'] != arrow.stat().st_size:
                    raise ValueError(f'Invalid token cache chunk: {arrow}')
                receipts[index] = stats
            else:
                jobs.append((str(path), str(cache), index, start, stop, expected))
        reused = sum(s['processed'] for s in receipts.values())
        print(f'[tokenize] {path.name}: {total:,} source rows; {reused:,} cached; '
              f'workers={workers}; cache={cache}', flush=True)
        started, last_report = time.monotonic(), 0

        def report(force=False):
            nonlocal last_report
            now = time.monotonic()
            if not force and now - last_report < 30:
                return
            last_report = now
            done = sum(s['processed'] for s in receipts.values())
            rate = (done - reused) / max(now - started, .001)
            eta = ('0.00h' if done == total else
                   f'{(total - done) / rate / 3600:.2f}h' if rate else 'pending first chunk')
            dropped = sum(s['dropped'] for s in receipts.values())
            size = sum(s['bytes'] for s in receipts.values()) / 1e9
            print(f'[tokenize] {path.name}: {done:,}/{total:,} source rows; '
                  f'dropped={dropped:,}; {rate:.1f} rows/s; ETA={eta}; cache={size:.2f} GB', flush=True)

        if jobs and workers == 1:
            for job in jobs:
                index, stats = _build_chunk(job, (tokenizer, encode_record, max_length))
                receipts[index] = stats
                report()
        elif jobs:
            tokenizer_dir = cache / 'tokenizer'
            tokenizer.save_pretrained(tokenizer_dir)
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs)),
                    mp_context=multiprocessing.get_context('spawn'), initializer=_initialize,
                    initargs=(str(tokenizer_dir), encode_record.__module__, encode_record.__name__, max_length)) as pool:
                queue = iter(jobs)
                pending = {pool.submit(_build_chunk, job) for job in
                           [next(queue, None) for _ in range(min(workers, len(jobs)))] if job is not None}
                while pending:
                    finished, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                    for future in finished:
                        index, stats = future.result()
                        receipts[index] = stats
                        job = next(queue, None)
                        if job is not None:
                            pending.add(pool.submit(_build_chunk, job))
                    report()
        report(force=True)
        after = path.stat()
        if stamp != (after.st_size, after.st_mtime_ns):
            raise ValueError('Prepared data changed during tokenization; stop preparation before training')
        totals = {key: sum(s[key] for s in receipts.values())
                  for key in ('kept', 'imagined', 'dropped', 'dropped_imagined', 'tokens')}
        print(f'[tokenize] {path.name} complete: {totals}', flush=True)
        parts = [Dataset.from_file(str(cache / f'{index:07d}.arrow'))
                 for index in range(len(chunks)) if receipts[index]['kept']]
        return concatenate_datasets(parts) if parts else None
