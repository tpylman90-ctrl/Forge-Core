import json
import tempfile
import unittest
from pathlib import Path

import torch

from forge_core.neural.prepare import prepare
from forge_core.neural.experiment import run


class TestScratchLanguageLoop(unittest.TestCase):
    def test_demo_manifest_and_heldout_split(self):
        with tempfile.TemporaryDirectory() as temp:
            files = Path(temp)
            manifest = prepare("demo", files, train_rows=24, valid_rows=10)
            self.assertEqual(manifest["accepted_train_rows"], 24)
            self.assertEqual(manifest["accepted_valid_rows"], 10)
            self.assertNotEqual(manifest["train_sha256"], manifest["valid_sha256"])
            self.assertTrue((files / "manifest.json").exists())
            self.assertEqual(
                json.loads((files / "manifest.json").read_text())["train_sha256"],
                manifest["train_sha256"],
            )

    def test_training_evaluation_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepare("demo", root, train_rows=12, valid_rows=8)
            train, valid = root / "train.txt", root / "valid.txt"
            first = root / "one.pt"
            second = root / "three-resumed.pt"
            direct = root / "three-direct.pt"
            kwargs = dict(train_file=train, valid_file=valid, model_size="tiny",
                          batch_size=2, seq_len=16, eval_batches=1, seed=22)
            a = run(**kwargs, output=first, steps=1, report=root / "a.json")
            self.assertEqual(a["global_step"], 1)
            self.assertGreater(a["parameter_count"], 0)
            self.assertGreater(a["heldout_loss_before"], 0)
            b = run(**kwargs, output=second, steps=2, resume=first)
            c = run(**kwargs, output=direct, steps=3)
            self.assertEqual(b["global_step"], 3)
            self.assertEqual(c["global_step"], 3)
            recovered = torch.load(second, weights_only=True)
            uninterrupted = torch.load(direct, weights_only=True)
            self.assertIn("optimizer", recovered)
            self.assertIn("sample_rng_state", recovered)
            self.assertEqual(len(recovered["history"]), 2)
            for key, tensor in recovered["model"].items():
                self.assertTrue(torch.equal(tensor, uninterrupted["model"][key]), key)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                run(**{**kwargs, "seed": 23}, output=root / "bad.pt",
                    resume=first, steps=1)

    def test_reject_identical_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "same.txt"
            p.write_text("An original sentence for a very small test. " * 4)
            with self.assertRaisesRegex(ValueError, "identical"):
                run(train_file=p, valid_file=p, output=Path(temp) / "bad.pt",
                    model_size="tiny", steps=1, seq_len=16)


if __name__ == "__main__":
    unittest.main()
