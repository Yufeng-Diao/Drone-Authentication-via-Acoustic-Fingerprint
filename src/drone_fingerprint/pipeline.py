"""A verified recording index and shared, immutable NumPy feature arrays.

Two cache levels separate raw segment-local MFCCs from selected feature vectors.
Keys exclude the estimator; raw MFCC keys also exclude derivative/fraction choices.
Writers serialize on one Ubuntu advisory lock and publish complete directories.
"""

from contextlib import contextmanager
from dataclasses import replace
import fcntl
import importlib.metadata
import json
import os
import re
from pathlib import Path
import shutil

import numpy as np
import toolbox.mfcc_base

from .artifacts import digest
from .data import file_signature, inspect_recording, labelled_intervals, recording_path, sha256, PCMReader
from .features import feature_metadata, frame_geometry, iter_base_frames, iter_selected_frames
from .noise import NoisyReader
from .storage import atomic_json, external_path, sync_directory


@contextmanager
def cache_lock(cache_root: Path, audio_root: Path):
    """One writer at a time; valid cache hits never rewrite cached files."""
    cache_root = external_path(cache_root, audio_root)
    if audio_root.resolve().is_relative_to(cache_root):
        raise ValueError('Cache root must not contain the input audio directory')
    cache_root.mkdir(parents=True, exist_ok=True)
    with (cache_root / '.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            # A dead process can leave staging directories, never published hits.
            for path in cache_root.glob('.building-*'):
                if re.fullmatch(r'\.building-[0-9a-f]{64}', path.name) and path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
            yield cache_root
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def recording_index(records, audio_root, cache_root, verify=False):
    """Rehash changed files, or every file on request; unchanged entries use stat."""
    path = cache_root / ('recordings-' + digest(str(audio_root.resolve())) + '.json')
    index = json.loads(path.read_text()) if path.exists() else {'schema_version': 1, 'recordings': {}}
    if (index.get('schema_version') != 1 or not isinstance(index.get('recordings'), dict)
            or (path.exists() and index.get('content_sha256') != digest(index['recordings']))):
        raise ValueError('Invalid recording index; remove it and rebuild')
    changed, checked = False, 0
    for record in records:
        previous = index['recordings'].get(record['path'])
        signature = file_signature(recording_path(audio_root, record))
        if (verify or previous is None or previous['signature'] != signature
                or previous['sha256'] != record['sha256'] or previous['frames'] != record['frames']
                or signature[2] != record['bytes']):
            index['recordings'][record['path']] = inspect_recording(audio_root, record)
            changed, checked = True, checked + 1
    if changed:
        index['content_sha256'] = digest(index['recordings'])
        atomic_json(path, index)
    return {r['path']: index['recordings'][r['path']] for r in records}, checked


def build_recording_index(manifest, audio_root, cache_root):
    """One directory walk compares the full inventory; subsequent tasks use paths."""
    records = manifest['recordings']
    if manifest['schema_version'] != 1 or len({r['path'] for r in records}) != len(records):
        raise ValueError('Invalid or duplicate recording manifest')
    actual = {str((Path(directory) / name).relative_to(audio_root))
              for directory, _, names in os.walk(audio_root) for name in names if name.lower().endswith('.wav')}
    if actual != {r['path'] for r in records}:
        raise ValueError('WAV inventory differs from manifest; missing or unexpected recording')
    with cache_lock(cache_root, audio_root) as root:
        _, checked = recording_index(records, audio_root, root, verify=True)
    return {'status': 'passed', 'recordings': len(records), 'directory_walks': 1, 'hashed_recordings': checked}


def cache_identity(records, segments, settings, split):
    relevant = {s['recording'] for s in segments}
    clean = replace(settings, blocks=1, portion=3)
    files = [Path(__file__).parent / name for name in ('data.py', 'features.py', 'noise.py', 'pipeline.py')]
    core = Path(toolbox.mfcc_base.__file__).parent
    files += [core/'mfcc_base.py', core/'mfcc_sigproc.py']
    noise = None if split == 'train' or settings.test_snr_quarter_db is None else {'seed': settings.seed, 'snr_quarter_db': settings.test_snr_quarter_db}
    return {'schema_version': 1, 'representation': 'raw MFCC including coefficient zero',
            'recordings': [r for r in records if r['path'] in relevant], 'segments': segments,
            'acoustics': feature_metadata(clean), 'noise': noise,
            'dependencies': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'scikit-learn')},
            'source': {p.name: sha256(p) for p in files}}


def load_array(directory, identity, *, freshly_published=False):
    """Validate before mmap; new entries were already hashed under the same lock."""
    if directory.is_symlink():
        raise ValueError('Feature cache directory must not be a symlink')
    metadata = json.loads((directory/'metadata.json').read_text())
    path = directory/'features.npy'
    if (metadata.get('identity') != identity or metadata.get('status') != 'complete'
            or path.is_symlink() or path.stat().st_size != metadata['bytes']
            or (not freshly_published and sha256(path) != metadata['sha256'])):
        raise ValueError(f'Invalid feature cache: {directory}; remove this entry to rebuild')
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if list(array.shape) != metadata['shape'] or array.dtype != np.dtype('float64'):
        raise ValueError(f'Invalid cached feature shape or dtype: {directory}')
    return array, metadata


@contextmanager
def array_writer(root, key, identity, shape):
    """Flush data and metadata, then publish together with an atomic rename."""
    if not re.fullmatch(r'[0-9a-f]{64}', key):
        raise ValueError('Cache key must be a lowercase SHA-256')
    estimated_bytes = int(np.prod(shape)) * 8 + 16384
    if shutil.disk_usage(root).free < estimated_bytes:
        raise ValueError(f'Insufficient free cache space: need at least {estimated_bytes} bytes')
    temporary = root / ('.building-' + key)
    temporary.mkdir()
    array = None
    try:
        array = np.lib.format.open_memmap(temporary/'features.npy', mode='w+', dtype=np.float64, shape=shape)
        metadata = {'identity': identity, 'shape': list(shape), 'status': 'complete'}
        yield array, metadata
        array.flush()
        array._mmap.close()
        array = None
        with (temporary/'features.npy').open('rb') as stream:
            os.fsync(stream.fileno())
        metadata.update(bytes=(temporary/'features.npy').stat().st_size, sha256=sha256(temporary/'features.npy'))
        atomic_json(temporary/'metadata.json', metadata)
        os.replace(temporary, root/key)
        sync_directory(root)
    finally:
        if array is not None:
            array._mmap.close()
        if temporary.exists():
            shutil.rmtree(temporary)


def prepare_features(records, audio_root, settings, cache_root, *, verify_inputs=False, batch_frames=32):
    """Return read-only train/test mmaps, row provenance, noise records and counters."""
    if type(batch_frames) is not int or not 1 <= batch_frames <= 256:
        raise ValueError('batch_frames must be an integer from 1 through 256')
    segments = []
    for record in records:
        for split, start, end, label in labelled_intervals(record, settings.partition):
            segments.append({'recording': record['path'], 'identity': record['identity'], 'classifier_label': label,
                             'split': split, 'start_sample': start, 'end_sample': end,
                             'feature_rows': frame_geometry(end-start, settings)[2]})
    arrays, noise_records = {}, []
    stats = {'directory_walks': 0, 'hashed_recordings': 0, 'base_hits': 0, 'base_misses': 0,
             'selected_hits': 0, 'selected_misses': 0, 'batch_frames': batch_frames, 'cache_keys': {}}
    with cache_lock(cache_root, audio_root) as root:
        print(f'Feature arrays: {sum(s["feature_rows"] for s in segments)} rows, '
              f'{feature_metadata(settings)["columns"]} selected columns, float64; '
              f'{sum(s["feature_rows"] for s in segments) * (settings.n_filters + feature_metadata(settings)["columns"]) * 8} data bytes for base + selected arrays', flush=True)
        index, stats['hashed_recordings'] = recording_index(records, audio_root, root, verify_inputs)
        for split in ('train', 'test'):
            rows = [s for s in segments if s['split'] == split]
            count = sum(s['feature_rows'] for s in rows)
            if not count:
                raise ValueError(f'No {split} intervals selected')
            base_identity = cache_identity(records, rows, settings, split)
            base_key = digest(base_identity)
            selected_identity = {'base_key': base_key, 'features': feature_metadata(settings)}
            selected_key = digest(selected_identity)
            stats['cache_keys'][split] = {'base': base_key, 'selected': selected_key}
            if (root/selected_key).exists():
                arrays[split], metadata = load_array(root/selected_key, selected_identity)
                stats['selected_hits'] += 1
                print(f'Feature cache hit: {split}, {count} rows', flush=True)
            else:
                if (root/base_key).exists():
                    base, base_meta = load_array(root/base_key, base_identity)
                    stats['base_hits'] += 1
                else:
                    print(f'Extracting {split}: {count} rows in batches of {batch_frames} frames', flush=True)
                    stats['base_misses'] += 1
                    with array_writer(root, base_key, base_identity, (count, settings.n_filters)) as (target, base_meta):
                        offset, noises = 0, []
                        for record in records:
                            intervals = [s for s in rows if s['recording'] == record['path']]
                            if not intervals:
                                continue
                            print(f"Recording: {record['path']} ({split})", flush=True)
                            reader = PCMReader(audio_root, record, index[record['path']])
                            noisy = NoisyReader(reader, record['sha256'], settings.seed, settings.test_snr_quarter_db) if base_identity['noise'] else None
                            try:
                                for segment in intervals:
                                    for block in iter_base_frames(noisy or reader, segment['start_sample'], segment['end_sample'], settings, batch_frames):
                                        target[offset:offset+len(block)] = block
                                        offset += len(block)
                                if noisy is not None:
                                    noises.append({'recording': record['path'], **noisy.finish()})
                            finally:
                                reader.close()
                        if offset != count:
                            raise ValueError('Extracted row count differs from segment index')
                        base_meta['noise'] = noises
                    base, base_meta = load_array(root/base_key, base_identity, freshly_published=True)
                stats['selected_misses'] += 1
                columns = feature_metadata(settings)['columns']
                with array_writer(root, selected_key, selected_identity, (count, columns)) as (target, metadata):
                    offset = 0
                    for segment in rows:
                        end = offset + segment['feature_rows']
                        cursor = offset
                        for block in iter_selected_frames(base[offset:end], settings):
                            target[cursor:cursor+len(block)] = block
                            cursor += len(block)
                        offset = end
                    metadata['noise'] = base_meta['noise']
                del base
                arrays[split], metadata = load_array(root/selected_key, selected_identity, freshly_published=True)
            if arrays[split].shape != (count, feature_metadata(settings)['columns']):
                raise ValueError('Feature cache shape differs from the requested task')
            noise_records.extend(metadata['noise'])
        for record in records:
            if file_signature(recording_path(audio_root, record)) != index[record['path']]['signature']:
                raise ValueError('Recording changed during task preparation')
    labels = {split: np.concatenate([np.full(s['feature_rows'], s['classifier_label'], dtype=np.int64)
                                    for s in segments if s['split'] == split]) for split in ('train', 'test')}
    original = np.concatenate([np.full(s['feature_rows'], s['identity'], dtype=np.int64) for s in segments if s['split'] == 'test'])
    return arrays, labels, original, segments, noise_records, stats
