"""Gold-supervised linguistic WHY probes: balanced options and unseen nouns/verbs."""
import json
import tempfile
import unittest
from pathlib import Path

from forge_core.learning.linguistic_why import (
    build_probes, validate_splits, write_probes,
)


class TestLinguisticWhy(unittest.TestCase):
    def test_distinct_splits_and_determinism(self):
        train, evaluation = build_probes("train"), build_probes("eval")
        self.assertEqual(train, build_probes("train"))
        self.assertEqual((len(train), len(evaluation)), (15, 15))
        validate_splits(train, evaluation)
        self.assertEqual({p.skill for p in train},
                         {"semantic_roles", "subject_verb_agreement", "negation"})
        self.assertFalse({p.verb_root for p in train} &
                         {p.verb_root for p in evaluation})

    def test_role_swaps_preserve_grammar(self):
        p = next(p for p in build_probes("train") if p.skill == "semantic_roles")
        self.assertTrue(p.gold["both_grammatical"])
        self.assertEqual(p.gold["a"]["agent"], p.gold["b"]["patient"])
        self.assertEqual(p.gold["b"]["agent"], p.gold["a"]["patient"])
        self.assertNotEqual(p.sentence_a, p.sentence_b)

    def test_grammaticality_and_negation_labels_are_not_always_a(self):
        data = build_probes("train")
        agreement = [p for p in data if p.skill == "subject_verb_agreement"]
        negation = [p for p in data if p.skill == "negation"]
        self.assertEqual({p.gold["grammatical"] for p in agreement}, {"a", "b"})
        self.assertEqual(
            {side for p in negation for side in ("a", "b")
             if p.gold[side]["event_polarity"] == "negated"}, {"a", "b"})
        self.assertTrue(all(p.gold["both_grammatical"] for p in negation))

    def test_hidden_gold_and_deterministic_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            manifest = write_probes(out)
            self.assertEqual(manifest["schema"], 2)
            self.assertEqual((manifest["train_count"], manifest["eval_count"]), (15, 15))
            questions = [json.loads(x) for x in (out / "eval-questions.jsonl").read_text().splitlines()]
            gold = [json.loads(x) for x in (out / "eval-gold.jsonl").read_text().splitlines()]
            self.assertEqual({x["id"] for x in questions}, {x["id"] for x in gold})
            self.assertTrue(all("gold" not in x and "verified_rule" not in x for x in questions))
            self.assertTrue(all("gold" in x for x in gold))
            self.assertEqual(manifest, write_probes(out))

    def test_invalid_split(self):
        with self.assertRaisesRegex(ValueError, "Split"):
            build_probes("private")


if __name__ == "__main__":
    unittest.main()
