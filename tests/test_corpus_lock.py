"""Fail-closed exact-content lock tests independent of the network."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from forge_core.neural.prepare import prepare
from forge_core.neural.lock import verify_corpus_lock


class TestFrozenPreviewLock(unittest.TestCase):
    def test_lock_accepts_unchanged_demo_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            manifest = prepare("demo", path, 12, 8)
            manifest_path = path / "manifest.json"
            lock_path = path / "lock.json"
            lock = {
                "schema": 1, "name": "test",
                "dataset": manifest["source"],
                "license": manifest["license"],
                "train_rows": 12, "valid_rows": 8,
                "train_sha256": manifest["train_sha256"],
                "valid_sha256": manifest["valid_sha256"],
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            }
            lock_path.write_text(json.dumps(lock))
            self.assertEqual(verify_corpus_lock(manifest_path, lock_path)["status"], "verified")
            lock["valid_sha256"] = "0" * 64
            lock_path.write_text(json.dumps(lock))
            with self.assertRaisesRegex(ValueError, "lock mismatch"):
                verify_corpus_lock(manifest_path, lock_path)

    def test_lock_detects_noncontent_manifest_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            manifest = prepare("demo", path, 12, 8)
            manifest_path = path / "manifest.json"
            lock_path = path / "lock.json"
            lock = {
                "schema": 1, "name": "test",
                "dataset": manifest["source"], "license": manifest["license"],
                "train_rows": 12, "valid_rows": 8,
                "train_sha256": manifest["train_sha256"], "valid_sha256": manifest["valid_sha256"],
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            }
            lock_path.write_text(json.dumps(lock))
            manifest["warning"] = "mutated"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                verify_corpus_lock(manifest_path, lock_path)


if __name__ == "__main__":
    unittest.main()
