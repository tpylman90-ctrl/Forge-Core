"""Trainability gates for the original language model."""
import json
import tempfile
import unittest
from pathlib import Path

from forge_core.neural.experiment import configuration, run
from forge_core.neural.model import ByteTransformer
from forge_core.neural.prepare import prepare


class TestTrainingGates(unittest.TestCase):
    def test_intermediate_model_is_original_and_cpu_feasible(self):
        tiny = ByteTransformer(configuration("tiny")).num_parameters
        seed = ByteTransformer(configuration("seed")).num_parameters
        base = ByteTransformer(configuration("base")).num_parameters
        self.assertLess(tiny, 100_000)
        self.assertGreater(seed, 1_000_000)
        self.assertLess(seed, base)

    def test_manifest_gates_and_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepare("demo", root, train_rows=12, valid_rows=8)
            args = dict(
                train_file=root / "train.txt", valid_file=root / "valid.txt",
                manifest_file=root / "manifest.json", output=root / "ckpt.pt",
                model_size="tiny", steps=1, batch_size=2, seq_len=16,
                eval_batches=2, sample_tokens=3, report=root / "metrics.json",
            )
            result = run(**args)
            self.assertEqual(result["validation_windows"], 4)
            self.assertIn("sample_text", result)
            self.assertIsNotNone(result["manifest_sha256"])
            data = json.loads((root / "manifest.json").read_text())
            data["train_sha256"] = "0" * 64
            (root / "manifest.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                run(**{**args, "output": root / "invalid.pt"})


if __name__ == "__main__":
    unittest.main()
