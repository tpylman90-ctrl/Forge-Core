"""Controlled linguistic-why A/B tests, no personal data or self-labels."""
from dataclasses import asdict
import hashlib
from pathlib import Path
import tempfile
import unittest
import torch

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.why_experiment import build_examples, shuffled_control, run_experiment


class TestLinguisticWhyExperiment(unittest.TestCase):
    def test_choices_balanced_but_not_fixed_answer_and_unseen_vocabulary(self):
        tr, ev = build_examples("train"), build_examples("eval")
        self.assertEqual((len(tr), len(ev)), (30, 30))
        for skill in {x.skill for x in tr}:
            self.assertEqual({x.answer for x in tr if x.skill == skill}, {"A", "B"})
            self.assertEqual({x.answer for x in ev if x.skill == skill}, {"A", "B"})
        null = shuffled_control(tr, 23)
        self.assertEqual([x.prompt for x in tr], [x.prompt for x in null])
        self.assertEqual(sorted(x.answer for x in tr), sorted(x.answer for x in null))
        self.assertNotEqual([x.answer for x in tr], [x.answer for x in null])
        self.assertFalse({x.id for x in tr} & {x.id for x in ev})

    def test_same_checkpoint_and_validation_hash_and_equal_update_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = ModelConfig(n_layer=1, n_head=4, n_embd=32, max_seq_len=256)
            torch.manual_seed(5)
            model = ByteTransformer(cfg)
            raw = ("The fox followed the rabbit. It was a sunny day. " * 8).encode()
            (root / "valid.txt").write_bytes(raw)
            language_text = ("A young fox followed an otter through the forest. " * 8).encode()
            (root / "train.txt").write_bytes(language_text)
            torch.save({
                "schema": 2, "config": asdict(cfg), "model": model.state_dict(),
                "valid_sha256": hashlib.sha256(raw).hexdigest(),
                "train_sha256": hashlib.sha256(language_text).hexdigest(),
                "step": 7,
            }, root / "checkpoint.pt")
            report = run_experiment(checkpoint_path=root / "checkpoint.pt",
                                    validation_path=root / "valid.txt",
                                    steps=2, batch=2, validation_windows=2,
                                    output_dir=root / "out")
            self.assertTrue(report["identical_initial_weights"])
            self.assertEqual(report["steps_per_arm"], 2)
            self.assertEqual(report["eval_gold"]["total"], 30)
            self.assertEqual(report["eval_shuffled"]["total"], 30)
            self.assertEqual(set(report["heldout_next_byte_loss"]),
                             {"frozen", "verified_labels", "shuffled_labels"})
            self.assertTrue((root / "out" / "why-experiment.json").exists())
            self.assertTrue((root / "out" / "verified-labels.pt").exists())
            regularized = run_experiment(
                checkpoint_path=root / "checkpoint.pt", validation_path=root / "valid.txt",
                steps=2, batch=2, validation_windows=2,
                language_train_path=root / "train.txt", language_weight=0.4)
            self.assertEqual(regularized["public_language_loss_weight"], 0.4)
            self.assertTrue(regularized["identical_language_minibatches"])
            (root / "train.txt").write_text("changed")
            with self.assertRaisesRegex(ValueError, "training text differs"):
                run_experiment(checkpoint_path=root / "checkpoint.pt",
                               validation_path=root / "valid.txt",
                               steps=1, language_train_path=root / "train.txt",
                               language_weight=0.4)
            (root / "valid.txt").write_text("altered")
            with self.assertRaisesRegex(ValueError, "differs"):
                run_experiment(checkpoint_path=root / "checkpoint.pt",
                               validation_path=root / "valid.txt", steps=1)

    def test_invalid_bounds(self):
        with self.assertRaisesRegex(ValueError, "Invalid"):
            run_experiment(checkpoint_path=Path("missing"), validation_path=Path("missing"),
                           steps=0)


if __name__ == "__main__":
    unittest.main()
