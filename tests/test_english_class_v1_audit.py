"""Tamper-resistant English Class lineage and comparison checks."""
import copy
from pathlib import Path
import unittest

from forge_core.neural.english_class_v1_audit import check_lineage
from forge_core.neural.english_class_v1 import ARMS, comparable_story_loss


class TestResearchArtifactIntegrity(unittest.TestCase):
    def setUp(self):
        self.source = "a" * 64
        self.tokenizer = "b" * 64
        self.config = {"vocab_size": 256, "max_seq_len": 256,
                       "n_layer": 1, "n_head": 2, "n_embd": 32,
                       "dropout": 0.0}
        self.record = {
            "schema": 1, "research_only": True,
            "arm": "english_class",
            "source_checkpoint_sha256": self.source,
            "source_tokenizer_sha256": self.tokenizer,
            "configuration": self.config,
            "best_weights": {"dummy": 1},
        }

    def verify(self, record):
        return check_lineage(
            record, arm="english_class", original_sha=self.source,
            tokenizer_sha=self.tokenizer, configuration=self.config)

    def test_valid_source_and_research_only_lineage(self):
        self.verify(self.record)
        self.assertEqual(len(ARMS), 3)

    def test_any_substituted_model_or_tokenizer_is_rejected(self):
        tests = [
            ("schema", 2),
            ("research_only", False),
            ("arm", "corrupted_answers"),
            ("source_checkpoint_sha256", "c" * 64),
            ("source_tokenizer_sha256", "d" * 64),
            ("configuration", {"vocab_size": 123}),
            ("best_weights", None),
        ]
        for name, wrong in tests:
            with self.subTest(field=name):
                altered = copy.deepcopy(self.record)
                if name == "best_weights":
                    del altered[name]
                else:
                    altered[name] = wrong
                with self.assertRaisesRegex(ValueError, "lineage"):
                    self.verify(altered)

    def test_source_retention_scoring_is_48_identical_windows(self):
        import torch
        from unittest.mock import patch
        with patch("forge_core.neural.english_class_v1.heldout",
                   return_value=2.123456) as fake:
            a = comparable_story_loss(object(), torch.zeros(100))
            b = comparable_story_loss(object(), torch.zeros(100))
            self.assertEqual(a, b)
            self.assertEqual(fake.call_count, 2)
            for call in fake.call_args_list:
                self.assertEqual(call.kwargs,
                                 {"windows": 48, "seq_len": 64})


if __name__ == "__main__":
    unittest.main()
