"""Small command-line interface with explicit inputs and external outputs."""

import argparse
from pathlib import Path
import sys
import wave

from . import __version__
from .config import Settings, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Acoustic drone authentication: P1 closed-set workflow")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run clean base MFCCs on a declared recording subset")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--audio-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True, help="New directory outside every source checkout")
    synthetic = commands.add_parser("synthetic", help="Check installation with labelled synthetic WAV fixtures")
    synthetic.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print("Initializing the numerical stack; startup may take time.", flush=True)
        from .data import select_recordings, sha256
        from .workflow import create_output, run_closed_set
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
        provenance.update(recordings_manifest_sha256=sha256(manifest), subset_manifest_sha256=sha256(subset))
        run_closed_set(records, audio_root, settings, run_output, provenance)
        return 0
    except (ValueError, TypeError, KeyError, OSError, EOFError, wave.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
