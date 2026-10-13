"""Regression checks for protocol validation; no training dependencies."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import validate_p0 as protocol
import report_p0


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((protocol.CONFIG / "recordings.json").read_text())

    def test_frozen_protocol(self):
        result = protocol.validate()
        self.assertEqual(result["audio_verification"], "not run")
        self.assertEqual(result["paper_pairs"], 120)

    def test_duplicate_attack_identity_rejected(self):
        rows = json.loads((protocol.CONFIG / "attack_partitions.json").read_text())["partitions"]
        rows[0]["attack"][0] = rows[0]["registered"][0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            protocol.validate_partitions(rows)

    def test_pair_frame_mismatch_rejected(self):
        self.manifest["recordings"][0]["frames"] -= 1
        with self.assertRaisesRegex(ValueError, "frame counts"):
            protocol.validate_recordings(self.manifest)

    def test_missing_recording_rejected(self):
        self.manifest["recordings"].pop()
        with self.assertRaisesRegex(ValueError, "257"):
            protocol.validate_recordings(self.manifest)

    def test_report_rejects_source_directory(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            report_p0.external_directory(protocol.CONFIG)

    def test_incorrect_paper_features_rejected(self):
        config = json.loads((protocol.CONFIG / "protocol.json").read_text())
        config["experiments"][3]["n_filters"] = [200]
        with self.assertRaisesRegex(ValueError, "feature settings"):
            protocol.validate_experiments(config)

    def test_unsafe_path_rejected(self):
        self.manifest["recordings"][0]["path"] = "../outside.wav"
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            protocol.validate_recordings(self.manifest)

    def test_incomplete_subset_rejected(self):
        subset = json.loads((protocol.CONFIG / "validation_subset.json").read_text())
        subset["drone_recordings"].pop()
        with self.assertRaisesRegex(ValueError, "complete pair"):
            protocol.validate_subset(subset, self.manifest, (protocol.CONFIG / "recordings.json").read_bytes())

    def test_split_covers_every_sample_once(self):
        for frames in (100, 101, 100003, 5379696):
            intervals = protocol.split_intervals(frames)
            ordered = sorted(intervals["train"] + intervals["test"])
            self.assertEqual(ordered[0][0], 0)
            self.assertEqual(ordered[-1][1], frames)
            self.assertTrue(all(a[1] == b[0] for a, b in zip(ordered, ordered[1:])))
            self.assertLessEqual(abs(sum(b-a for a, b in intervals["train"]) - frames * 0.7), 2)

    def test_corrupted_audio_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.wav"
            with wave.open(str(path), "wb") as audio:
                audio.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
                audio.writeframes(b"\x01\x00" * 100)
            data = path.read_bytes()
            record = {"path": path.name, "bytes": len(data), "frames": 100,
                      "sha256": hashlib.sha256(data).hexdigest()}
            protocol.verify_audio(Path(directory), record)
            path.write_bytes(data[:-1] + b"\xff")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                protocol.verify_audio(Path(directory), record)


if __name__ == "__main__":
    unittest.main()
