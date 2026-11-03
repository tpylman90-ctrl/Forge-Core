"""Forge v3 factorized question/event learning safeguards (no real user data)."""
from dataclasses import replace
from types import SimpleNamespace
import tempfile
import time
import unittest
from pathlib import Path

import torch

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.role_curriculum_v1 import make_rows
from forge_core.neural.role_contrast_v2 import Feature, features, groups, question_grams
from forge_core.neural.role_factorized_v3 import (
    SEEDS, TwoStageHead, audit_exam_vocab, auxiliary_loss, extended_train,
    make_new_exam, source_facts, verify_new_exam, train, experiment,
)


class TestCharCodec:
    def encode(self, text):
        return SimpleNamespace(
            ids=[ord(c) % 127 for c in text],
            offsets=[(i, i + 1) for i in range(len(text))],
        )


class TestFactorizedData(unittest.TestCase):
    def test_sealed_exam_and_independent_gold(self):
        audit_exam_vocab()
        rows = make_new_exam()
        self.assertEqual(rows, make_new_exam())
        self.assertEqual(len(rows), 288)
        self.assertEqual(len({r.prompt for r in rows}), 288)
        self.assertEqual({r.seed for r in rows}, {12311, 12312})
        self.assertEqual(len({r.template for r in rows}), 3)
        self.assertEqual(len({r.answer for r in rows}) > 5, True)
        verify_new_exam(rows)
        for row in rows:
            truth = source_facts(row.template, row.story)
            self.assertEqual(row.answer, truth[row.role])

    def test_gold_and_query_tampering_fail_closed(self):
        row = make_new_exam()[0]
        with self.assertRaisesRegex(ValueError, "gold"):
            verify_new_exam([replace(row, answer=row.recipient)] + make_new_exam()[1:])
        with self.assertRaisesRegex(ValueError, "Question"):
            verify_new_exam([replace(row, prompt=row.prompt.replace(
                "Which person initially", "Which person accidentally"))] + make_new_exam()[1:])
        with self.assertRaisesRegex(ValueError, "grammar"):
            verify_new_exam([replace(row, story=row.story.replace(
                "put", "destroyed"))] + make_new_exam()[1:])

    def test_training_augmentation_is_paired_and_disjoint(self):
        rows = extended_train()
        self.assertEqual(len(rows), 1600)
        self.assertEqual(len(set(r.prompt for r in rows)), 1600)
        self.assertEqual(len(groups([
            Feature(SimpleNamespace(seed=r.seed, template=r.template,
                                    trial=r.trial, swap=r.swap, role=r.role,
                                    answer=r.answer),
                    torch.empty(1)) for r in rows
        ])), 400)
        self.assertFalse(set(r.prompt for r in rows) & {
            r.prompt for r in make_new_exam()})


class TestFactorizedNeuralHead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        cls.backbone = ByteTransformer(
            ModelConfig(vocab_size=128, max_seq_len=384,
                        n_layer=1, n_head=2, n_embd=16))
        cls.backbone.eval()
        for parameter in cls.backbone.parameters():
            parameter.requires_grad_(False)
        cls.codec = TestCharCodec()

    def test_giver_and_recipient_questions_are_separate_network_inputs(self):
        sample = features(self.backbone, self.codec, make_rows("train")[:4])
        a, b = sample[:2]
        self.assertTrue(torch.equal(a.slot.entity, b.slot.entity))
        self.assertFalse(torch.equal(a.grams, b.grams))
        head = TwoStageHead(16)
        giver_story, giver_question, giver_answer = head(a)
        receiver_story, receiver_question, receiver_answer = head(b)
        self.assertTrue(torch.equal(giver_story, receiver_story))
        self.assertTrue(torch.allclose(giver_answer.exp().sum(), torch.tensor(1.), atol=1e-5))
        self.assertTrue(torch.allclose(receiver_answer.exp().sum(), torch.tensor(1.), atol=1e-5))
        self.assertEqual(giver_question.shape, (2,))
        self.assertEqual(giver_answer.shape, (2,))
        # Different gold labels may NEVER change network input or output.
        corrupted = replace(make_rows("train")[0], answer=make_rows("train")[0].recipient)
        from forge_core.neural.role_contrast_v2 import features as extract
        modified = extract(self.backbone, self.codec, [corrupted])[0]
        self.assertTrue(torch.equal(a.slot.entity, modified.slot.entity))
        self.assertTrue(torch.equal(a.grams, modified.grams))
        self.assertTrue(torch.equal(head(a)[-1], head(modified)[-1]))

    def test_question_ablation_cannot_distinguish_opposing_questions(self):
        a, b = features(self.backbone, self.codec, make_rows("train")[:2])
        head = TwoStageHead(16)
        self.assertTrue(torch.equal(head(a, ablate=True)[-1],
                                    head(b, ablate=True)[-1]))
        self.assertTrue(torch.equal(head(a, ablate=True)[1],
                                    head(b, ablate=True)[1]))

    def test_only_head_gets_training_gradients_and_wrong_story_changes_loss(self):
        sample = features(self.backbone, self.codec, make_rows("train")[:4])
        head = TwoStageHead(16)
        normal = auxiliary_loss(head, sample[0])
        wrong = auxiliary_loss(head, sample[0], false_giver=True)
        self.assertGreater(abs(normal.item() - wrong.item()), 1e-7)
        normal.backward()
        self.assertTrue(any(p.grad is not None for p in head.parameters()))
        self.assertTrue(all(p.grad is None for p in self.backbone.parameters()))

    def test_two_step_smoke_all_four_arms(self):
        subset = features(self.backbone, self.codec, extended_train()[:16])
        dev = features(self.backbone, self.codec, make_rows("dev")[:8])
        exam = features(self.backbone, self.codec, make_new_exam()[:8])
        with tempfile.TemporaryDirectory() as folder:
            results = train(seed=SEEDS[0], training=subset, dev=dev,
                            exam=exam, steps=2, eval_every=1,
                            max_seconds=30, checkpoint_dir=Path(folder))
            self.assertEqual(results["steps"], 2)
            self.assertTrue((Path(folder) / f"seed-{SEEDS[0]}-research.pt").exists())
            self.assertEqual(set(results["arms"]), {
                "factorized", "wrong_story", "question_off", "monolithic"})
            for arm in results["arms"].values():
                self.assertEqual(arm["unseen_synthetic_exam"]["total"], 8)
                self.assertEqual(arm["unseen_synthetic_exam"][
                    "both_questions_correct"]["total"], 4)

    def test_bounded_time_checks(self):
        with self.assertRaisesRegex(ValueError, "bounded CPU pilot"):
            experiment(source=Path("."), parent=Path("."), pointer=Path("."),
                       out=Path("."), max_minutes=10)
        with self.assertRaisesRegex(ValueError, "bounded CPU pilot"):
            experiment(source=Path("."), parent=Path("."), pointer=Path("."),
                       out=Path("."), steps=20000)


if __name__ == "__main__":
    unittest.main()
