"""Verify source/challenge isolation and counterbalanced synthetic facts."""
import unittest
from pathlib import Path
from forge_core.neural.preschool_compositional import (
    CHALLENGE_OBJECTS, CHALLENGE_PEOPLE, TRAIN_OBJECTS, TRAIN_PEOPLE,
    SKILL_NAMES, make_examples, shuffle_answers_generic, run,
)

class TestCompositionalQA(unittest.TestCase):
    def test_disjoint_names_objects_prompts_and_reproducibility(self):
        self.assertFalse(set(TRAIN_PEOPLE) & set(CHALLENGE_PEOPLE))
        self.assertFalse(set(TRAIN_OBJECTS) & set(CHALLENGE_OBJECTS))
        train = make_examples("train", 1000, seed=707)
        challenge = make_examples("challenge", 200, seed=708)
        preview = make_examples("challenge", 50, seed=784)
        self.assertEqual(train, make_examples("train", 1000, seed=707))
        self.assertEqual((len(train),len(challenge),len(preview)), (1000,200,50))
        self.assertFalse({r.prompt for r in train} & {r.prompt for r in challenge})
        self.assertFalse({r.prompt for r in challenge} & {r.prompt for r in preview})
        self.assertFalse({r.prompt for r in train} & {r.prompt for r in preview})
        for row in train+challenge:
            self.assertIn(row.answer.casefold(), row.prompt.casefold())
            self.assertEqual(row.prompt.count("Answer:"), 1)
            self.assertNotIn("\nA) ", row.prompt)
        for skill in SKILL_NAMES:
            self.assertEqual(sum(x.skill == skill for x in train), 200)
            self.assertEqual(sum(x.skill == skill for x in challenge), 40)

    def test_expansion_keeps_original_questions(self):
        self.assertEqual(make_examples("train",160,707),
                         make_examples("train",800,707)[:160])
        self.assertEqual(make_examples("train",800,707),
                         make_examples("train",3200,707)[:800])

    def test_shuffled_control_has_identical_prompts_and_skill_counts(self):
        data=make_examples("train",160,92)
        null=shuffle_answers_generic(data,543)
        self.assertEqual([x.prompt for x in data],[x.prompt for x in null])
        self.assertNotEqual([x.answer for x in data],[x.answer for x in null])
        for skill in SKILL_NAMES:
            self.assertEqual(sorted(x.answer for x in data if x.skill==skill),
                             sorted(x.answer for x in null if x.skill==skill))

    def test_evaluation_supports_a_new_skill_name(self):
        # Regression: direct evaluator previously hardcoded only four skills,
        # so 'giver' crashed before experimental training even started.
        from types import SimpleNamespace
        import torch
        from forge_core.neural.preschool_direct import (
            DirectExample, evaluate_direct,
        )
        class DummyTokenizer:
            def encode(self, text):
                return SimpleNamespace(ids=[1, 2])
            def decode(self, ids):
                return "Ada."
        class DummyModel:
            cfg = SimpleNamespace(max_seq_len=256)
            def eval(self):
                return self
            def __call__(self, x):
                logits = torch.zeros((1, x.shape[1], 4))
                logits[0, -1, 2] = 5.0
                return logits, None
        rows=[DirectExample("giver-test","giver",
                            "Story: Ada gave a toy to Ben.\nQuestion: Who gave it?\nAnswer:",
                            "Ada")]
        score=evaluate_direct(DummyModel(),DummyTokenizer(),rows,max_tokens=2)
        self.assertEqual(score["correct"], 1)
        self.assertEqual(score["by_skill"]["giver"]["total"], 1)

    def test_bounds_prevent_unsafe_runs(self):
        with self.assertRaisesRegex(ValueError,"Bounded"):
            run(folder=Path("missing"),out=Path("missing"),
                stages=((40,5001,0.2,0.0001),))
        with self.assertRaisesRegex(ValueError,"Challenge"):
            run(folder=Path("missing"),out=Path("missing"),challenge_count=43)

if __name__=="__main__":
    unittest.main()
