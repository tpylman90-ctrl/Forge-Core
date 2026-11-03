"""Draft checkpoint archive must preserve the exact approved public inputs."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from forge_core.neural.archive import verify_archive
from forge_core.neural.experiment import run
from forge_core.neural.prepare import prepare


class TestArchiveAudit(unittest.TestCase):
    def test_matching_checkpoint_data_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare("demo", root, 12, 8)
            out = root / "archive"
            out.mkdir()
            run(train_file=root / "train.txt", valid_file=root / "valid.txt",
                manifest_file=root / "manifest.json", output=out / "checkpoint.pt",
                report=out / "metrics.json", model_size="tiny", steps=1,
                batch_size=2, seq_len=16, eval_batches=2)
            for filename, renamed in (("train.txt", "train.txt"),
                                      ("valid.txt", "valid.txt"),
                                      ("manifest.json", "dataset-manifest.json")):
                shutil.copyfile(root / filename, out / renamed)
            (out / "SOURCE-NOTICE.txt").write_text("Original project demo data")
            report = verify_archive(out, require_public_tinystories=False)
            self.assertEqual(report["optimizer_steps"], 1)
            self.assertTrue((out / "archive-audit.json").is_file())
            with self.assertRaisesRegex(ValueError, "TinyStories"):
                verify_archive(out)
            (out / "valid.txt").write_text("corrupted")
            with self.assertRaisesRegex(ValueError, "corpus differs"):
                verify_archive(out, require_public_tinystories=False)


if __name__ == "__main__":
    unittest.main()
