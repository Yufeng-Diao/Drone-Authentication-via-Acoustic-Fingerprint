"""External directories and atomic completion records on Ubuntu filesystems."""

import json
import os
from pathlib import Path
import tempfile


def external_path(path: Path, audio_root: Path | None = None) -> Path:
    path = path.resolve()
    if any((p / '.git').exists() or (p / 'pyproject.toml').exists() for p in (path, *path.parents)):
        raise ValueError('Output must be outside every Git checkout or Python source project')
    if audio_root is not None and path.is_relative_to(audio_root.resolve()):
        raise ValueError('Output must not be inside the input audio directory')
    return path


def create_output(path: Path, audio_root: Path | None = None) -> Path:
    path = external_path(path, audio_root)
    path.mkdir(parents=True, exist_ok=False)
    return path


def sync_directory(path: Path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value):
    """Flush once, then atomically replace; a partial JSON is never completion."""
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)
