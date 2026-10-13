"""Validate the frozen paper protocol without importing the training stack.

JSON results go to stdout; progress goes to stderr. Redirect both outside the
repository. This checks data and specifications, not numerical model behavior.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import wave


CONFIG = Path(__file__).resolve().parents[1] / "config" / "paper"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def split_intervals(frames):
    """Return half-open sample intervals; paired recordings share boundaries."""
    require(isinstance(frames, int) and frames >= 100, "Recording is too short")
    boundaries = [frames * percent // 100 for percent in (0, 20, 35, 65, 80, 100)]
    return {"train": [[boundaries[i], boundaries[i + 1]] for i in (0, 2, 4)],
            "test": [[boundaries[i], boundaries[i + 1]] for i in (1, 3)]}


def validate_partitions(partitions):
    require([p["trial"] for p in partitions] == list(range(1, 11)), "Expected ten ordered trials")
    for p in partitions:
        groups = [p[k] for k in ("registered", "background", "attack")]
        require(all(len(g) == len(set(g)) == 8 for g in groups), "Each group must contain eight unique identities")
        require(sorted(sum(groups, [])) == list(range(1, 25)), "Partition groups overlap or omit an identity")


def validate_recordings(manifest):
    records = manifest["recordings"]
    require(len(records) == 257, "Expected all 257 published recordings")
    require(len({r["path"] for r in records}) == len(records), "Duplicate recording paths")
    require(len({r["sha256"] for r in records}) == len(records), "Duplicate recording content")
    pairs = defaultdict(list)
    for r in records:
        path = PurePosixPath(r["path"])
        require(not path.is_absolute() and ".." not in path.parts and "\\" not in r["path"], "Unsafe recording path")
        require(re.fullmatch(r"[0-9a-f]{64}", r["sha256"]) is not None, "Invalid SHA-256")
        require(r["frames"] > 0 and r["bytes"] >= r["frames"] * 2, "Invalid recording size")
        match = re.fullmatch(r"_uav_(\d{8}|2022noise)_(d\d+|n)_(hover|noise)_(1|5)m_(.*)\.wav", path.name.lower())
        require(match is not None, f"Unrecognized filename: {path}")
        date, label, state, distance, suffix = match.groups()
        identity = None if label == "n" else int(label[1:])
        require((r["date"], r["identity"], r["distance_m"], r["pair_key"]) ==
                (date, identity, int(distance), f"{date}_{label}_{state}_{suffix}"), f"Filename metadata mismatch: {path}")
        require(r["legacy_date_selected"] == (date in manifest["legacy_dates"]), "Legacy date selection mismatch")
        if identity is not None:
            require(1 <= identity <= 24 and state == "hover", "Invalid drone identity or state")
            pairs[r["pair_key"]].append(r)
        else:
            require(state == "noise" and date == "2022noise", "Invalid room-noise metadata")
    for key, pair in pairs.items():
        require(len(pair) == 2 and {r["distance_m"] for r in pair} == {1, 5}, f"Incomplete distance pair: {key}")
        require(len({r["frames"] for r in pair}) == 1, f"Pair frame counts differ: {key}")
    paper = [r for r in records if r["identity"] is not None and r["legacy_date_selected"]]
    require(Counter(r["identity"] for r in paper) == Counter({i: 10 for i in range(1, 25)}), "Paper coverage must be ten files per identity")
    require(sum(r["identity"] is None for r in records) == 1, "Expected one room-noise recording")
    require(Counter(r["date"] for r in records if not r["legacy_date_selected"]) == {"20220227": 16}, "Unexpected excluded recordings")
    require((manifest["sample_rate_hz"], manifest["channels"], manifest["sample_width_bytes"]) == (44100, 1, 2), "Unexpected audio format")
    return paper, pairs


def validate_subset(subset, manifest, manifest_bytes):
    require(subset["recordings_manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest(), "Subset references a different manifest")
    records = manifest["recordings"]
    expected = []
    for identity in range(1, 25):
        candidates = [r for r in records if r["identity"] == identity and r["legacy_date_selected"]]
        key = min(r["pair_key"] for r in candidates)
        expected.extend(r["path"] for r in candidates if r["pair_key"] == key)
    require(subset["drone_recordings"] == sorted(expected), "Subset must contain the first complete pair per identity")
    noise = next(r for r in records if r["identity"] is None)
    require(subset["room_noise_recording"] == noise["path"] and
            subset["room_noise_interval_samples"] == [0, 60 * 44100] and noise["frames"] >= 60 * 44100,
            "Subset must use the first 60 seconds of room noise")


def validate_experiments(protocol):
    require(protocol["models"] == ["QDA", "LDA", "LSVM", "RBF-SVM", "KNN", "DT", "RF", "GNB"], "Unexpected model list")
    require(protocol["split"]["boundary_percent"] == [0, 20, 35, 65, 80, 100] and
            protocol["split"]["interval_roles"] == ["train", "test", "train", "test", "train"], "Unexpected split")
    experiments = {e["id"]: e for e in protocol["experiments"]}
    require(len(experiments) == len(protocol["experiments"]) == 6, "Expected six unique experiment presets")
    require(set(experiments) == {"figure2", "figure3", "figure4", "figure5", "table2", "tables4_5"}, "Unexpected paper outputs")
    require(protocol["datasets"]["DS1"] == list(range(1, 9)) and protocol["datasets"]["DS2"] == list(range(1, 25)), "Incorrect DS1/DS2 identities")
    for name, dataset in (("figure2", "DS1"), ("figure3", "DS1"), ("figure4", "DS1N"),
                          ("table2", "DS2"), ("figure5", "DS2N"), ("tables4_5", "DS2-open-set")):
        require(experiments[name]["dataset"] == dataset, f"Incorrect dataset: {name}")
    for name, filters in (("figure2", 50), ("table2", 201), ("tables4_5", 201)):
        e = experiments[name]
        require(e["n_filters"] == [filters] and e["blocks"] == [1] and e["fractions"] == [[1, 1]], f"Incorrect feature settings: {name}")
    require(all(experiments[name]["frame_ms"] == [1000] for name in ("figure3", "table2", "tables4_5")), "Expected one-second frames")
    require(experiments["tables4_5"]["partition_trials"] == list(range(1, 11)), "Expected all ten attack trials")
    require(protocol["features"]["overlap_fraction"] == 0.5, "Expected 50% overlap")
    require(experiments["figure2"]["frame_ms"] == list(range(20, 2521, 50)), "Frame sweep differs from the paper")
    require(experiments["figure3"]["n_filters"] == list(range(26, 272, 5)), "Filter sweep differs from the paper")
    require(experiments["figure3"]["blocks"] == [1, 2, 3] and
            experiments["figure3"]["fractions"] == [[1, 3], [2, 3], [1, 1]], "Expected nine feature variants")
    for name, parent, snrs in (("figure4", "figure3", [0]), ("figure5", "table2", list(range(-32, 61)))):
        require(experiments[name]["reuse_models_from"] == parent and experiments[name]["snr_quarter_db"] == snrs,
                "Noise evaluation must reuse the designated clean models and exact SNR grid")
    coverage = []
    for e in experiments.values():
        fits = 0
        if "reuse_models_from" not in e:
            fits = 8 * len(e["frame_ms"]) * len(e["n_filters"]) * len(e["blocks"]) * len(e["fractions"]) * len(e.get("partition_trials", [1]))
        require(fits == e["new_fits"], f"Incorrect fit count: {e['id']}")
        coverage.append({"id": e["id"], "paper": e["paper"], "new_fits": fits, "status": "specified; not run"})
    require(sum(e["new_fits"] for e in coverage) == 4096, "Expected 4096 planned fits")
    return coverage


def verify_audio(root, record):
    root = root.resolve()
    path = (root / record["path"]).resolve()
    require(path.is_relative_to(root), f"Recording escapes audio root: {record['path']}")
    require(path.is_file(), f"Missing recording: {record['path']}")
    require(path.stat().st_size == record["bytes"], f"File size mismatch: {record['path']}")
    with path.open("rb") as stream:
        require(hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"], f"Checksum mismatch: {record['path']}")
    with wave.open(str(path), "rb") as audio:
        require((audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes(), audio.getcomptype()) ==
                (1, 2, 44100, record["frames"], "NONE"), f"Audio format mismatch: {record['path']}")


def validate(audio_root=None):
    def load(name):
        return json.loads((CONFIG / name).read_text())

    manifest = load("recordings.json")
    paper, pairs = validate_recordings(manifest)
    subset = load("validation_subset.json")
    validate_subset(subset, manifest, (CONFIG / "recordings.json").read_bytes())
    partitions = load("attack_partitions.json")["partitions"]
    validate_partitions(partitions)
    coverage = validate_experiments(load("protocol.json"))
    printed = load("table1_durations.json")
    durations = []
    for row in printed["rows"] + [dict(identity="total", **printed["total"])]:
        selected = [r for r in paper if row["identity"] == "total" or r["identity"] == row["identity"]]
        values = {f"{d}m_s": sum(r["frames"] for r in selected if r["distance_m"] == d) / 44100 for d in (1, 5)}
        values["total_s"] = sum(r["frames"] for r in selected) / 44100
        errors = {key: values[key] - row[key] for key in values}
        require(all(abs(error) <= printed["rounding_tolerance_s"] for error in errors.values()), f"Table 1 duration mismatch: {row['identity']}")
        durations.append(dict(identity=row["identity"], measured=values, printed=row, difference_s=errors))
    if audio_root is not None:
        require(audio_root.is_dir(), "Audio root does not exist")
        found = {p.relative_to(audio_root).as_posix() for p in audio_root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"}
        require(found == {r["path"] for r in manifest["recordings"]}, "Missing or unindexed WAV files; refusing to silently skip files")
        for i, record in enumerate(manifest["recordings"], 1):
            verify_audio(audio_root, record)
            if i % 32 == 0 or i == len(manifest["recordings"]):
                print(f"Verified {i}/{len(manifest['recordings'])} WAV files", file=sys.stderr, flush=True)
    subset_records = [r for r in paper if r["path"] in subset["drone_recordings"]]
    example = subset_records[0]
    return {"status": "passed", "scope": "P0 protocol and data only; no training or feature extraction",
            "audio_verification": "passed" if audio_root is not None else "not run",
            "config_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(CONFIG.glob("*.json"))},
            "published_wav_count": len(manifest["recordings"]), "paper_drone_count": len(paper),
            "excluded_drone_count": 16, "paired_recordings": len(pairs), "paper_pairs": len(paper) // 2,
            "pairing_limit": "Names, format and frame counts checked; acoustic synchronization not independently verified.",
            "subset_drone_count": len(subset_records), "subset_drone_duration_s": sum(r["frames"] for r in subset_records) / 44100,
            "subset_room_noise_duration_s": 60, "room_noise_duration_s": next(r["frames"] for r in manifest["recordings"] if r["identity"] is None) / 44100,
            "table1_comparison": durations, "attack_trials": len(partitions), "experiment_coverage": coverage,
            "timeline_example": {"pair_key": example["pair_key"], "frames": example["frames"], "sample_rate_hz": 44100,
                                 "intervals": split_intervals(example["frames"])}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, help="Extracted dataset root; omit for specification-only checks")
    args = parser.parse_args()
    try:
        result = validate(args.audio_root)
    except (ValueError, KeyError, TypeError, OSError, wave.Error) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        sys.exit(1)
    print(json.dumps(result, indent=2))
