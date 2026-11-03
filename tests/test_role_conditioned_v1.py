"""Source-verified, no-gold-leak role-conditioned neural research tests."""
from dataclasses import replace
from types import SimpleNamespace
from collections import defaultdict
import unittest

import torch
from torch.nn import functional as F

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.span_pointer_lab import WordPointer
from forge_core.neural.role_curriculum_v1 import (
    EXAM_NAMES, NAMES, SOURCES, SEEDS, TRIALS, candidate_spans,
    make_rows, source_roles, verify_disjoint, verify_rows,
)
from forge_core.neural.role_conditioned_v1 import (
    RoleHead, evaluate, score_rows, shuffled_labels, slot_features, Slots,
    old_pointer_restricted,
)


class CharCodec:
    """Deterministic test-only token IDs, not the pretrained Forge BPE."""
    def encode(self, text):
        return SimpleNamespace(ids=[ord(c) % 127 for c in text],
                               offsets=[(i, i + 1) for i in range(len(text))])


class TestRoleCurriculum(unittest.TestCase):
    def test_separate_vocabulary_families_and_counts(self):
        verify_disjoint()
        lengths = {"train": 320, "dev": 80, "exam": 288}
        combined = []
        for split, count in lengths.items():
            rows = make_rows(split)
            self.assertEqual(rows, make_rows(split))
            self.assertEqual(len(rows), count)
            self.assertEqual(len({r.prompt for r in rows}), count)
            self.assertEqual(len({r.answer for r in rows}) > 1, True)
            for r in rows:
                self.assertEqual(r.answer, r.giver if r.role == "giver" else r.recipient)
                names = candidate_spans(r.prompt)
                self.assertEqual(set(x[0] for x in names), {r.giver, r.recipient})
            combined.extend(rows)
        self.assertEqual(len({r.prompt for r in combined}), len(combined))

    def test_gold_and_question_tampering_are_rejected(self):
        row = make_rows("train")[0]
        with self.assertRaisesRegex(ValueError, "Gold answer"):
            verify_rows([replace(row, answer=row.recipient)])
        with self.assertRaisesRegex(ValueError, "Question does not match"):
            verify_rows([replace(
                row, prompt=row.prompt.replace("Who gave", "Who received"))])
        with self.assertRaisesRegex(ValueError, "Duplicate|duplicate"):
            verify_rows([row, row])
        with self.assertRaisesRegex(ValueError, "grammar"):
            source_roles(row.template, row.story.replace("offered", "forgot"))
        with self.assertRaisesRegex(ValueError, "two independently"):
            candidate_spans("Story: no entity here.\nQuestion: Why?\nAnswer:")

    def test_counterfactual_cases_have_both_roles(self):
        rows = make_rows("exam")
        look = {(r.seed, r.template, r.trial, r.swap, r.role): r for r in rows}
        for seed in SEEDS["exam"]:
            for name, _, _ in SOURCES["exam"]:
                for trial in range(TRIALS["exam"]):
                    a = look[(seed, name, trial, 0, "giver")]
                    b = look[(seed, name, trial, 1, "giver")]
                    self.assertEqual(a.giver, b.recipient)
                    self.assertEqual(a.recipient, b.giver)
                    self.assertNotEqual(a.answer, b.answer)
                    rec = look[(seed, name, trial, 0, "recipient")]
                    self.assertEqual(a.story, rec.story)
                    self.assertNotEqual(a.answer, rec.answer)

    def test_role_shuffle_is_not_illusory(self):
        rows = make_rows("train")
        width = 8
        feats = [Slots(r.id, r.role, r.template, r.seed, r.trial, r.swap,
                       r.answer, tuple(x[0] for x in candidate_spans(r.prompt)),
                       torch.empty(width), torch.empty(2, 2 * width + 9),
                       tuple(x[0] for x in candidate_spans(r.prompt)).index(r.answer))
                 for r in rows]
        wrong = shuffled_labels(feats, 7723)
        self.assertGreater(sum(a != b.gold for a, b in zip(wrong, feats)), 50)
        for role in ("giver", "recipient"):
            idx = [i for i, r in enumerate(feats) if r.role == role]
            self.assertEqual(sum(wrong[i] for i in idx), sum(feats[i].gold for i in idx))


class TestRoleHead(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(102)
        self.model = ByteTransformer(ModelConfig(
            vocab_size=128, max_seq_len=384, n_layer=1, n_head=2, n_embd=16))
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def test_independent_name_slots_and_gradient_only_in_head(self):
        row = make_rows("train")[0]
        feature = slot_features(self.model, CharCodec(), row)
        self.assertEqual(feature.entity.shape, (2, 2 * 16 + 9))
        self.assertEqual(feature.query.shape, (16,))
        self.assertEqual(feature.names[feature.gold], row.answer)
        self.assertFalse(feature.entity.requires_grad)
        head = RoleHead(16)
        logits = head(feature.query, feature.entity)
        self.assertEqual(logits.shape, (2,))
        F.cross_entropy(logits[None, :], torch.tensor([feature.gold])).backward()
        self.assertTrue(any(p.grad is not None for p in head.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        other = row.recipient if row.answer == row.giver else row.giver
        changed = slot_features(self.model, CharCodec(), replace(row, answer=other))
        self.assertEqual(feature.names, changed.names)
        self.assertTrue(torch.equal(feature.entity, changed.entity))
        self.assertTrue(torch.equal(feature.query, changed.query))
        self.assertNotEqual(feature.gold, changed.gold)

    def test_legacy_pointer_baseline_uses_same_detected_name_slots(self):
        rows = make_rows("dev")[:4]
        report = old_pointer_restricted(WordPointer(16), self.model, CharCodec(), rows)
        self.assertEqual(report["total"], 4)
        self.assertEqual(report["both_questions_correct"]["total"], 2)
        self.assertEqual(report["both_entity_swaps_correct"]["total"], 2)

    def test_pair_metrics_detect_constant_answer_failure(self):
        rows = make_rows("dev")
        slots = [Slots(r.id, r.role, r.template, r.seed, r.trial, r.swap,
                       r.answer, tuple(n for n, _, _ in candidate_spans(r.prompt)),
                       torch.empty(0), torch.empty(0), -1) for r in rows]
        perfect, _ = score_rows(slots, lambda s: s.answer)
        self.assertEqual(perfect["correct"], len(rows))
        self.assertEqual(perfect["both_questions_correct"]["correct"], len(rows) // 2)
        self.assertEqual(perfect["both_entity_swaps_correct"]["correct"], len(rows) // 2)
        constant, _ = score_rows(slots, lambda s: s.names[0])
        self.assertEqual(constant["correct"], len(rows) // 2)
        self.assertEqual(constant["both_questions_correct"]["correct"], 0)
        self.assertEqual(constant["both_entity_swaps_correct"]["correct"], len(rows) // 4)
        with self.assertRaisesRegex(ValueError, "not an independently"):
            score_rows(slots, lambda s: "unknown")


if __name__ == "__main__":
    unittest.main()
