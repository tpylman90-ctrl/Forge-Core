"""Strict tests for balanced preschool QA curriculum and research-only models."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from forge_core.neural.preschool_qa import (
    TRAIN_NAMES, TRANSFER_NAMES, training_tasks, transfer_tasks, shuffled_labels,
    question_prompt, research,
)
from forge_core.neural.seed_v07 import train


class TestPreschoolQA(unittest.TestCase):
    def test_balanced_choices_deterministic_and_unseen_entities(self):
        tr = training_tasks(160, seed=7)
        self.assertEqual(tr, training_tasks(160, seed=7))
        self.assertEqual(len(tr), 160)
        self.assertEqual(len({x.id for x in tr}), 160)
        for skill in ("color", "recipient", "location", "count"):
            rows = [x for x in tr if x.skill == skill]
            self.assertEqual(len(rows), 40)
            self.assertEqual(sum(x.answer == "A" for x in rows), 20)
            self.assertEqual(sum(x.answer == "B" for x in rows), 20)
        null = shuffled_labels(tr, seed=20)
        self.assertEqual([x.prompt for x in tr], [x.prompt for x in null])
        self.assertNotEqual([x.answer for x in tr], [x.answer for x in null])
        for skill in ("color", "recipient", "location", "count"):
            self.assertEqual(
                sorted(x.answer for x in tr if x.skill == skill),
                sorted(x.answer for x in null if x.skill == skill)
            )
        transfer = transfer_tasks()
        self.assertEqual(len(transfer), 40)
        self.assertEqual(sum(x.answer == "A" for x in transfer), 20)
        self.assertEqual(sum(x.answer == "B" for x in transfer), 20)
        self.assertFalse(set(TRAIN_NAMES) &
                         {name for pair in TRANSFER_NAMES for name in pair})

    def test_mirror_answers_and_invalid_inputs(self):
        prompt_a, answer_a = question_prompt(
            "Ava handed a toy to Jon.", "Who got it?", "Jon", "Ava", False)
        prompt_b, answer_b = question_prompt(
            "Ava handed a toy to Jon.", "Who got it?", "Jon", "Ava", True)
        self.assertEqual((answer_a, answer_b), ("A", "B"))
        self.assertNotEqual(prompt_a, prompt_b)
        with self.assertRaisesRegex(ValueError, "identical"):
            question_prompt("x", "y", "same", "same", False)
        with self.assertRaisesRegex(ValueError, "Count"):
            training_tasks(17)

    def test_tiny_reproducible_research_and_hash_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus = root / "corpus"
            corpus.mkdir()
            tr = ("Once upon a time, a little girl met a friend and was happy. " * 150 +
                  "\n\nThe little fox went home after playing all day.\n\n") * 2
            va = ("A gentle rabbit carried a blue flower in the garden. " * 140 +
                  "\n\nThe teacher smiled at the little bird.\n\n") * 2
            (corpus / "train.txt").write_text(tr)
            (corpus / "valid.txt").write_text(va)
            manifest = {
                "schema": 1,
                "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
                "license": "cdla-sharing-1.0", "train_document_count": 100,
                "valid_document_count": 100,
                "train_sha256": hashlib.sha256(tr.encode()).hexdigest(),
                "valid_sha256": hashlib.sha256(va.encode()).hexdigest(),
            }
            (corpus / "manifest.json").write_text(json.dumps(manifest))
            out = root / "baseline"
            train(source=corpus, out=out, steps=1, model_size="tiny",
                  batch=2, seq_len=32, vocab=512, eval_every=1)
            (out / "source-manifest.json").write_text(json.dumps(manifest))
            (out / "train.txt").write_text(tr)
            (out / "valid.txt").write_text(va)
            result = research(folder=out, out=root / "qa", steps=2,
                              count=16, batch_size=2, eval_every=1)
            self.assertEqual(result["adaptation_steps_per_arm"], 2)
            self.assertTrue(result["same_initial_weights"])
            self.assertEqual(result["baseline"]["legacy_40"]["total"], 40)
            self.assertEqual(result["verified_teaching"]["transfer_40"]["total"], 40)
            self.assertTrue((root / "qa" / "results.json").is_file())
            self.assertTrue((root / "qa" / "verified-research.pt").is_file())
            (out / "train.txt").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "Modified train"):
                research(folder=out, out=root / "bad", steps=1, count=16)


if __name__ == "__main__":
    unittest.main()
