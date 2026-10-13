"""Small command-line interface with explicit inputs and external outputs."""

import argparse
import json
from pathlib import Path
import sys
import wave

from . import __version__
from .config import Settings, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Acoustic drone authentication: validated research workflow")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one declared closed/open-set experiment")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--audio-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True, help="New directory outside every source checkout")
    run.add_argument("--resume", action="store_true", help="Reuse only a matching, intact completed task")
    run.add_argument("--save-features", action="store_true", help="Save a binary baseline snapshot for validation")
    run.add_argument('--cache-root', type=Path, help='External shared cache; default: sibling feature-cache')
    run.add_argument('--batch-frames', type=int, default=32, help='MFCC extraction batch, 1..256 frames; default 32')
    run.add_argument('--verify-inputs', action='store_true', help='Rehash WAVs even when indexed file metadata is unchanged')
    index = commands.add_parser('index', help='Verify all manifest WAVs with one directory walk and build the recording index')
    index.add_argument('--manifest', type=Path, required=True)
    index.add_argument('--audio-root', type=Path, required=True)
    index.add_argument('--cache-root', type=Path, required=True)
    synthetic = commands.add_parser("synthetic", help="Check installation with labelled synthetic WAV fixtures")
    synthetic.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print("Initializing the numerical stack; startup may take time.", flush=True)
        from .data import select_recordings, select_room_noise, sha256
        from .workflow import create_output, run_experiment
        if args.command == 'index':
            from .pipeline import build_recording_index
            result = build_recording_index(json.loads(args.manifest.read_text(encoding='utf-8')), args.audio_root, args.cache_root)
            print(json.dumps(result, indent=2), flush=True)
            return 0
        if args.command == "synthetic":
            from .synthetic import create_fixture
            output = create_output(args.output)
            manifest, subset = create_fixture(output / "fixture")
            settings = Settings()
            audio_root, run_output = manifest.parent, output / "run"
            provenance = {"dataset": "synthetic installation fixture; no paper accuracy claim"}
        else:
            settings, manifest, subset = load_config(args.config)
            audio_root, run_output = args.audio_root, args.output
            provenance = {"dataset": "Explicit identity projection of the frozen validation subset",
                          "configuration_sha256": sha256(args.config)}
        records = select_recordings(manifest, subset, settings.identities)
        if settings.partition is not None:
            records.append(select_room_noise(manifest, subset))
        provenance.update(recordings_manifest_sha256=sha256(manifest), subset_manifest_sha256=sha256(subset))
        run_experiment(records, audio_root, settings, run_output, provenance,
                       resume=getattr(args, "resume", False), save_features=getattr(args, "save_features", False),
                       cache_root=getattr(args, "cache_root", None), verify_inputs=getattr(args, "verify_inputs", False), batch_frames=getattr(args, "batch_frames", 32))
        return 0
    except (ValueError, TypeError, KeyError, OSError, EOFError, wave.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
