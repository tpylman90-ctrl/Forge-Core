"""Unit checks for role-contrast v2; no archived checkpoint downloads."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import time
import unittest

import torch
from torch.nn import functional as F

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.role_curriculum_v1 import make_rows
from forge_core.neural.role_contrast_v2 import (
    EXAM_SEEDS, SEEDS, ContrastRoleHead, audit_exam, audit_exam_vocab,
    features, group_flip_null, groups, make_exam, measure, pair_loss,
    question_grams, story_only_slots, train, validate_budget,
)


class TestCharCodec:
    def encode(self, text):
        return SimpleNamespace(
            ids=[ord(x) % 127 for x in text],
            offsets=[(i, i + 1) for i in range(len(text))],
        )


class TestRoleContrastData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.final = make_exam()

    def test_workflow_budget_and_python_ceiling_agree(self):
        validate_budget(36, 50000, 500)
        validate_budget(1, 1, 1)
        for minutes, steps, cadence in (
                (37, 50000, 500), (36, 50001, 500),
                (36, 50000, 50001), (0, 100, 10)):
            with self.subTest(minutes=minutes, steps=steps, cadence=cadence):
                with self.assertRaises(ValueError):
                    validate_budget(minutes, steps, cadence)

    def test_unique_new_exam_and_verified_gold(self):
        audit_exam_vocab()
        self.assertEqual(len(self.final), 360)
        self.assertEqual(self.final, make_exam())
        self.assertEqual(len(set(row.prompt for row in self.final)), 360)
        self.assertEqual({row.seed for row in self.final}, set(EXAM_SEEDS))
        self.assertEqual(len({row.template for row in self.final}), 3)
        for row in self.final:
            self.assertEqual(row.answer, row.giver if row.role == "giver" else row.recipient)
        audit_exam(self.final)

    def test_poisoned_final_answers_cannot_pass(self):
        x = self.final.copy()
        x[0] = replace(x[0], answer=x[0].recipient)
        with self.assertRaisesRegex(ValueError, "gold"):
            audit_exam(x)
        x = self.final.copy()
        x[0] = replace(x[0], prompt=x[0].prompt.replace("Who originally", "Who accidentally"))
        with self.assertRaisesRegex(ValueError, "question"):
            audit_exam(x)
        x = self.final.copy()
        x[0] = replace(x[0], story=x[0].story.replace("passed", "lost"))
        with self.assertRaisesRegex(ValueError, "grammar"):
            audit_exam(x)

    def test_same_story_has_both_questions_and_counterfactuals(self):
        tests = make_rows("train")[:12]
        self.assertEqual(len(tests), 12)
        allrows = tests + make_rows("dev")[:8] + self.final
        self.assertEqual(
            len({row.prompt for row in allrows}), len(allrows))
        for rows in (tests, make_rows("dev")[:8], self.final[:12]):
            for i in range(0, len(rows), 4):
                batch = rows[i:i + 4]
                self.assertEqual(batch[0].story, batch[1].story)
                self.assertEqual(batch[2].story, batch[3].story)
                self.assertNotEqual(batch[0].answer, batch[1].answer)
                self.assertNotEqual(batch[0].answer, batch[2].answer)
                self.assertNotEqual(batch[1].answer, batch[3].answer)


class TestRoleContrastModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(14)
        cls.backbone = ByteTransformer(
            ModelConfig(vocab_size=128, max_seq_len=384,
                        n_layer=1, n_head=2, n_embd=16))
        cls.backbone.eval()
        for p in cls.backbone.parameters():
            p.requires_grad_(False)
        cls.codec = TestCharCodec()

    def test_question_ablation_no_leak_and_gold_not_a_feature(self):
        giver, recipient = make_rows("train")[:2]
        f_g = story_only_slots(self.backbone, self.codec, giver)
        f_r = story_only_slots(self.backbone, self.codec, recipient)
        self.assertTrue(torch.equal(f_g.entity, f_r.entity))
        self.assertFalse(torch.equal(f_g.query, f_r.query))
        self.assertNotEqual(
            question_grams(giver.prompt).tolist(),
            question_grams(recipient.prompt).tolist())
        model = ContrastRoleHead(16)
        from forge_core.neural.role_contrast_v2 import Feature
        a = Feature(f_g, question_grams(giver.prompt))
        b = Feature(f_r, question_grams(recipient.prompt))
        self.assertTrue(torch.equal(
            model(a, ablate_question=True),
            model(b, ablate_question=True)))
        self.assertFalse(torch.equal(model(a), model(b)))
        corrupted = replace(giver, answer=giver.recipient)
        other = story_only_slots(self.backbone, self.codec, corrupted)
        self.assertTrue(torch.equal(f_g.entity, other.entity))
        self.assertTrue(torch.equal(f_g.query, other.query))
        self.assertNotEqual(f_g.gold, other.gold)

    def test_controls_different_and_matched_paired_structure(self):
        sample = features(self.backbone, self.codec, make_rows("train")[:40])
        pairs = groups(sample)
        self.assertEqual(len(pairs), 10)
        targets = group_flip_null(sample, pairs, seed=7638)
        self.assertGreater(
            sum(a != b.slot.gold for a, b in zip(targets, sample)), 0)
        for group in pairs:
            for a, b in ((0, 1), (2, 3)):
                self.assertNotEqual(targets[group[a]], targets[group[b]])
        head = ContrastRoleHead(16)
        optim = torch.optim.AdamW(head.parameters(), lr=0.001)
        loss = pair_loss(head, sample, pairs[0], [f.slot.gold for f in sample],
                         ablate=False, contrastive=True)
        loss.backward()
        optim.step()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(any(p.grad is not None for p in head.parameters()))
        self.assertTrue(all(p.grad is None for p in self.backbone.parameters()))

    def test_quick_fully_labeled_pilot_and_guarded_budget(self):
        train_features = features(self.backbone, self.codec, make_rows("train")[:40])
        dev_features = features(self.backbone, self.codec, make_rows("dev")[:16])
        exam_features = features(self.backbone, self.codec, make_exam()[:16])
        report = train(SEEDS[0], rows=train_features,
                       development=dev_features, exam=exam_features,
                       max_steps=2, eval_every=1,
                       deadline=time.monotonic() + 30)
        self.assertEqual(report["updates_per_arm"], 2)
        for arm in ("paired", "cross_entropy", "question_ablated", "shuffled_pairs"):
            r = report["heads"][arm]["final_synthetic_exam"]
            self.assertEqual(r["total"], 16)
            self.assertEqual(r["both_questions_correct"]["total"], 8)
            self.assertEqual(r["both_entity_swaps_correct"]["total"], 8)
        self.assertTrue(report["original_scratch_backbone_unchanged"])


if __name__ == "__main__":
    unittest.main()
