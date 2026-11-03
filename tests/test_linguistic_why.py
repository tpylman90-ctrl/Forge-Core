"""Gold-supervised linguistic WHY probes; never use candidate claims as labels."""
import json
import tempfile
import unittest
from pathlib import Path

from forge_core.learning.linguistic_why import (
    build_probes, validate_splits, write_probes,
)


class TestLinguisticWhy(unittest.TestCase):
    def test_distinct_splits_and_determinism(self):
        train = build_probes("train")
        evaluation = build_probes("eval")
        self.assertEqual(train, build_probes("train"))
        self.assertEqual(len(train), 15)
        self.assertEqual(len(evaluation), 15)
        validate_splits(train, evaluation)
        self.assertEqual({p.skill for p in train}, {
            "semantic_roles", "subject_verb_agreement", "negation"
        })

    def test_swapping_roles_changes_meaning_not_grammar(self):
        probe = next(p for p in build_probes("train")
                     if p.skill == "semantic_roles")
        self.assertTrue(probe.gold["both_grammatical"])
        self.assertTrue(probe.gold["roles_reversed"])
        self.assertEqual(probe.gold["a"]["agent"], probe.gold["b"]["patient"])
        self.assertEqual(probe.gold["b"]["agent"], probe.gold["a"]["patient"])
        self.assertNotEqual(probe.sentence_a, probe.sentence_b)

    def test_agreement_and_negation_are_different_questions(self):
        train = build_probes("train")
        agreement = next(p for p in train if p.skill == "subject_verb_agreement")
        negation = next(p for p in train if p.skill == "negation")
        self.assertEqual(agreement.gold["grammatical"], "a")
        self.assertEqual(agreement.gold["ungrammatical"], "b")
        self.assertEqual(negation.gold["a"]["event_polarity"], "affirmed")
        self.assertEqual(negation.gold["b"]["event_polarity"], "negated")
        self.assertTrue(negation.gold["both_grammatical"])

    def test_hidden_eval_gold_and_digests(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            manifest = write_probes(out)
            self.assertEqual(manifest["train_count"], 15)
            self.assertEqual(manifest["eval_count"], 15)
            questions = [json.loads(x) for x in (out / "eval-questions.jsonl").read_text().splitlines()]
            answers = [json.loads(x) for x in (out / "eval-gold.jsonl").read_text().splitlines()]
            self.assertEqual({x["id"] for x in questions}, {x["id"] for x in answers})
            self.assertTrue(all("gold" not in x and "verified_rule" not in x
                                for x in questions))
            self.assertTrue(all("gold" in x for x in answers))
            self.assertEqual(manifest, write_probes(out))
            self.assertEqual({p["skill"] for p in questions},
                             {"semantic_roles", "subject_verb_agreement", "negation"})

    def test_invalid_split(self):
        with self.assertRaisesRegex(ValueError, "Split"):
            build_probes("private")


if __name__ == "__main__":
    unittest.main()
