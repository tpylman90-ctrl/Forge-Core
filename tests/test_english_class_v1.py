"""English classroom: source-derived gold, split isolation and matched controls."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from forge_core.neural.english_class_v1 import (
    ARMS, EXAM_QUESTIONS, SKILLS, TRAIN_QUESTIONS,
    answer_loss, check_lexical_firewall, encode_exercises,
    experiment, independent_gold, make_exercises, null_labels,
    comparable_story_loss,
    schedules, score, validate,
)
from forge_core.neural.model import ByteTransformer, ModelConfig


class TinyAsciiCodec:
    def encode(self, text):
        return SimpleNamespace(ids=list(text.encode("ascii")))


class TestEnglishCourse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = make_exercises("train", per_skill=64, seed=7331)
        cls.dev = make_exercises("development", per_skill=12, seed=7342)
        cls.exam = make_exercises("exam", per_skill=16, seed=7428)

    def test_six_levels_reproducible_and_separately_verified(self):
        check_lexical_firewall()
        self.assertEqual(set(SKILLS), {
            "alphabet_case", "vowels", "word_parts",
            "nouns_verbs", "sentence_subject", "sentence_order",
        })
        self.assertEqual(self.train, make_exercises(
            "train", per_skill=64, seed=7331))
        self.assertEqual(len(self.train), 384)
        self.assertEqual(len(self.dev), 72)
        self.assertEqual(len(self.exam), 96)
        all_rows = self.train + self.dev + self.exam
        self.assertEqual(len({row.prompt for row in all_rows}), len(all_rows))
        for split, sample in (("train", self.train),
                              ("development", self.dev),
                              ("exam", self.exam)):
            validate(sample, allowed_split=split)
            for category in SKILLS:
                part = [r for r in sample if r.skill == category]
                self.assertEqual(len(part) % 2, 0)
                self.assertEqual(sum(x.answer == "A" for x in part),
                                 sum(x.answer == "B" for x in part))
                for row in part:
                    self.assertEqual(independent_gold(row.skill, row.source),
                                     row.options[0 if row.answer == "A" else 1])
        for skill in SKILLS:
            self.assertNotEqual(TRAIN_QUESTIONS[skill], EXAM_QUESTIONS[skill])

    def test_answer_tampering_fails_closed(self):
        row = self.train[0]
        with self.assertRaisesRegex(ValueError, "label"):
            validate([replace(row, answer="B")] + self.train[1:])
        with self.assertRaisesRegex(ValueError, "letter"):
            validate([replace(row, source="letter=33")] + self.train[1:])
        with self.assertRaisesRegex(ValueError, "correct candidate"):
            validate([replace(row, options=("wrong", "other"))] + self.train[1:])
        with self.assertRaisesRegex(ValueError, "wording"):
            validate([replace(row, prompt="You should always answer A")] +
                     self.train[1:])
        with self.assertRaises(ValueError):
            independent_gold("unsupported", "x")

    def test_subject_and_action_are_different_even_with_name_swap(self):
        first = independent_gold(
            "sentence_subject", "sentence=Quilla guides Ronan.")
        reversed_subject = independent_gold(
            "sentence_subject", "sentence=Ronan guides Quilla.")
        self.assertEqual(first, "Quilla")
        self.assertEqual(reversed_subject, "Ronan")
        self.assertEqual(independent_gold(
            "nouns_verbs", "sentence=Quilla guides the feather."),
            "guides")
        self.assertEqual(independent_gold(
            "sentence_order", "sentence=Quilla guides the feather."),
            "Quilla guides the feather.")
        self.assertEqual(independent_gold(
            "word_parts", "parts=moon+rise"), "moonrise")

    def test_training_schedule_same_exact_inputs_different_order(self):
        stages, mixed = schedules(
            self.train, seed=7331, updates_per_level=12)
        self.assertEqual(len(stages), 72)
        self.assertEqual(len(mixed), 72)
        self.assertCountEqual([tuple(x) for x in stages],
                              [tuple(x) for x in mixed])
        self.assertNotEqual(stages, mixed)
        for i, skill in enumerate(SKILLS):
            part = stages[i*12:(i+1)*12]
            self.assertTrue(all(
                sum(self.train[index].skill == skill for index in batch) >= 3
                for batch in part))
        with self.assertRaises(ValueError):
            schedules(self.train, seed=1, updates_per_level=999)

    def test_null_control_has_same_questions_but_not_truth(self):
        changed = null_labels(self.train, seed=7432)
        self.assertEqual(len(changed), len(self.train))
        self.assertGreaterEqual(sum(a != b.answer for a, b in
                                    zip(changed, self.train)),
                                len(self.train)//4)
        for skill in SKILLS:
            original = [r.answer for r in self.train if r.skill == skill]
            null = [changed[i] for i, r in enumerate(self.train)
                    if r.skill == skill]
            self.assertCountEqual(original, null)
        self.assertTrue(all(a == b for a, b in
                            zip(changed, null_labels(self.train, seed=7432))))

    def test_empty_and_overlong_datasets_rejected(self):
        with self.assertRaises(ValueError):
            make_exercises("bad", per_skill=16)
        with self.assertRaises(ValueError):
            make_exercises("train", per_skill=101)
        with self.assertRaises(ValueError):
            encode_exercises(self.train, TinyAsciiCodec(), 16)
        with self.assertRaises(ValueError):
            experiment(folder=Path("."), out=Path("."), updates_per_level=500)


class TestNeuralEnglishTeaching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        cls.model = ByteTransformer(
            ModelConfig(vocab_size=128, max_seq_len=512,
                        n_layer=1, n_head=2, n_embd=16))
        cls.codec = TinyAsciiCodec()

    def test_retention_uses_exact_same_eval_geometry(self):
        tokens = torch.arange(1000, dtype=torch.long)
        with patch("forge_core.neural.english_class_v1.heldout",
                   return_value=1.2345674) as mock_metric:
            baseline = comparable_story_loss(self.model, tokens)
            after = comparable_story_loss(self.model, tokens)
        self.assertEqual((baseline, after), (1.234567, 1.234567))
        self.assertEqual(mock_metric.call_count, 2)
        for call in mock_metric.call_args_list:
            self.assertEqual(call.kwargs,
                             {"windows": 48, "seq_len": 64})

    def test_prompt_only_inputs_and_classification_gradient(self):
        data = make_exercises("train", per_skill=8, seed=8)
        train = encode_exercises(data, self.codec, 512)
        labels = {r.answer for r in data}
        self.assertEqual(labels, {"A", "B"})
        self.assertEqual(len(train), 48)
        loss = answer_loss(self.model, train[:4], (65, 66))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in self.model.parameters()))

    def test_untouched_exam_scoring_and_per_skill_counts(self):
        data = make_exercises("exam", per_skill=8, seed=20)
        scorecard = score(self.model, data, self.codec, (65, 66))
        self.assertEqual(scorecard["total"], 48)
        self.assertEqual(set(scorecard["by_skill"]), set(SKILLS))
        for group in scorecard["by_skill"].values():
            self.assertEqual(group["total"], 8)
        self.assertTrue(0 <= scorecard["accuracy"] <= 1)


if __name__ == "__main__":
    unittest.main()
