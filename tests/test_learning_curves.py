"""Intermediate language-progress evaluation does not corrupt exact resume."""
import tempfile
import unittest
from pathlib import Path
import torch

from forge_core.neural.experiment import run
from forge_core.neural.prepare import prepare


class TestLearningCurves(unittest.TestCase):
    def test_progress_curve_and_expanded_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare("demo", root, 12, 8)
            args = dict(train_file=root / "train.txt", valid_file=root / "valid.txt",
                        manifest_file=root / "manifest.json", model_size="tiny",
                        batch_size=2, seq_len=16, seed=14, eval_batches=2,
                        eval_every=2, audit_batches=4)
            data = run(**args, steps=5, output=root / "curve.pt")
            self.assertEqual([x["global_step"] for x in data["evaluation_curve"]], [0, 2, 4, 5])
            self.assertEqual(data["audit_validation_windows"], 8)
            self.assertIsNotNone(data["audit_heldout_loss_after"])
            self.assertEqual(data["training_tokens_this_run"], 160)
            resumed = run(**args, steps=2, resume=root / "curve.pt", output=root / "resume.pt")
            fresh = run(**args, steps=7, output=root / "fresh.pt")
            self.assertEqual([x["global_step"] for x in resumed["evaluation_curve"]], [5, 7])
            left = torch.load(root / "resume.pt", weights_only=True)["model"]
            right = torch.load(root / "fresh.pt", weights_only=True)["model"]
            for key in left:
                self.assertTrue(torch.equal(left[key], right[key]), key)

    def test_negative_audit_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare("demo", root, 12, 8)
            with self.assertRaisesRegex(ValueError, "evaluation interval"):
                run(train_file=root / "train.txt", valid_file=root / "valid.txt",
                    output=root / "bad.pt", audit_batches=-2)


if __name__ == "__main__":
    unittest.main()
