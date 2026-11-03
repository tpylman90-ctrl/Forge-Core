"""Safeguard checks for archival validation of original Forge Seed v0.7."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from forge_core.neural.archive_v07 import audit
from forge_core.neural.seed_v07 import train


class TestArchiveV07(unittest.TestCase):
    def test_reject_corrupt_public_data_and_preserve_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / "source"
            source.mkdir()
            train_text = ("A small fox went home and found a toy. " * 200 + "\n\n") * 2
            valid_text = ("A happy rabbit met a kind friend today. " * 200 + "\n\n") * 2
            (source / "train.txt").write_text(train_text)
            (source / "valid.txt").write_text(valid_text)
            manifest = {
                "schema": 1, "dataset": "roneneldan/TinyStories",
                "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
                "license": "cdla-sharing-1.0",
                "train_document_count": 100, "valid_document_count": 100,
                "train_sha256": hashlib.sha256(train_text.encode()).hexdigest(),
                "valid_sha256": hashlib.sha256(valid_text.encode()).hexdigest(),
            }
            (source / "manifest.json").write_text(json.dumps(manifest))
            target = folder / "checkpoint"
            train(source=source, out=target, model_size="tiny", steps=1,
                  batch=2, seq_len=32, vocab=512, eval_every=1)
            shutil.copyfile(source / "train.txt", target / "train.txt")
            shutil.copyfile(source / "valid.txt", target / "valid.txt")
            (target / "source-manifest.json").write_text(json.dumps(manifest))
            (target / "SOURCE-NOTICE.txt").write_text(
                "TinyStories CDLA-Sharing-1.0 public synthetic research source")
            report = audit(target)
            self.assertEqual(report["optimizer_step"], 1)
            self.assertIn("tokenizer.json", report["artifacts_sha256"])
            self.assertTrue((target / "archive-integrity.json").exists())
            (target / "valid.txt").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "valid data hash mismatch"):
                audit(target)


if __name__ == "__main__":
    unittest.main()
