"""Fixed-prompt qualitative inspection is deterministic and local-only."""
from dataclasses import asdict
import tempfile
import unittest
from pathlib import Path

import torch

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.qualitative import PROMPTS, inspect


class TestQualitative(unittest.TestCase):
    def test_fixed_samples_and_no_global_rng_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = ModelConfig(n_layer=1, n_head=4, n_embd=32, max_seq_len=128)
            torch.manual_seed(12)
            model = ByteTransformer(cfg)
            filename = root / "model.pt"
            torch.save({"config": asdict(cfg), "model": model.state_dict(), "step": 2}, filename)
            state = torch.get_rng_state().clone()
            a = inspect(filename, generated_bytes=4)
            b = inspect(filename, generated_bytes=4)
            self.assertEqual(a, b)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertEqual(len(a["samples"]), len(PROMPTS))
            self.assertEqual(a["checkpoint_step"], 2)
            self.assertTrue(all(len(row["continuation"]) > 0 for row in a["samples"]))

    def test_invalid_limits(self):
        with self.assertRaisesRegex(ValueError, "limits"):
            inspect(Path("missing.pt"), generated_bytes=0)


if __name__ == "__main__":
    unittest.main()
