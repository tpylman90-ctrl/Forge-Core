"""Sealed research-only role-binding diagnostic: deterministic truth and contrasts."""
from dataclasses import replace
import unittest

from forge_core.neural.role_binding_diagnostic import (
    OBJECTS, PEOPLE, ROLES, SEEDS, TEMPLATES,
    check_novel_lexicon, make_cases, parse_roles, story_for, summarize, verify_cases,
)


class TestRoleBindingDiagnostic(unittest.TestCase):
    def test_no_overlap_with_earlier_pointer_splits(self):
        check_novel_lexicon()
        self.assertEqual(len(PEOPLE), len(set(PEOPLE)))
        self.assertEqual(len(OBJECTS), len(set(OBJECTS)))

    def test_independent_grammar_verification_across_voice_and_order(self):
        for template in TEMPLATES:
            for a, b in (("Nola", "Quinn"), ("Quinn", "Nola")):
                with self.subTest(template=template, giver=a):
                    story = story_for(template, a, b, "magnet")
                    self.assertEqual(parse_roles(template, story), (a, b, "magnet"))
                    with self.assertRaises(ValueError):
                        parse_roles(template, story.replace("magnet", "MAGNET"))

    def test_sealed_contrast_cases_and_cross_seed_uniqueness(self):
        all_rows = []
        for seed in SEEDS:
            rows = make_cases(seed)
            self.assertEqual(rows, make_cases(seed))
            self.assertEqual(len(rows), len(TEMPLATES) * 8 * 2 * 2)
            verify_cases(rows)
            all_rows.extend(rows)
            self.assertEqual(set(r.skill for r in rows), set(ROLES))
            for row in rows:
                self.assertIn(row.answer, (row.giver, row.recipient))
                self.assertEqual(row.answer, row.giver if row.skill == "giver" else row.recipient)
                self.assertEqual(row.prompt.count(row.giver), 1)
                self.assertEqual(row.prompt.count(row.recipient), 1)
                self.assertTrue(row.prompt.endswith("\nAnswer:"))
        self.assertEqual(len(set(x.prompt for x in all_rows)), len(all_rows))
        self.assertEqual(len(set(x.id for x in all_rows)), len(all_rows))

    def test_gold_poisoning_detected(self):
        row = make_cases(SEEDS[0], per_template=2)[0]
        with self.assertRaisesRegex(ValueError, "do not match"):
            verify_cases([replace(row, answer=row.recipient)])
        with self.assertRaisesRegex(ValueError, "Unknown source structure"):
            verify_cases([replace(row, prompt=row.prompt.replace("Story:", "Prompt:"))])
        with self.assertRaisesRegex(ValueError, "Role query"):
            verify_cases([replace(row, prompt=row.prompt.replace("Who gave", "Who received"))])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            verify_cases([row, row])

    def test_pair_metrics_cannot_be_faked_by_constant_copy(self):
        cases = make_cases(SEEDS[0], per_template=2)
        perfect = summarize(cases, {c.id: c.answer for c in cases})
        expected_pairs = len(TEMPLATES) * 2 * 2
        self.assertEqual(perfect["correct"], len(cases))
        self.assertEqual(perfect["role_question_pairs"]["both_correct"], expected_pairs)
        self.assertEqual(perfect["entity_swap_pairs"]["both_correct"], expected_pairs)
        self.assertEqual(perfect["selected_one_of_two_people"], len(cases))
        constant = summarize(cases, {c.id: c.giver for c in cases})
        self.assertEqual(constant["correct"], len(cases) // 2)
        self.assertEqual(constant["role_question_pairs"]["both_correct"], 0)
        self.assertEqual(constant["entity_swap_pairs"]["both_correct"], expected_pairs // 2)
        with self.assertRaisesRegex(ValueError, "exactly one prediction"):
            summarize(cases, {})

    def test_budget_validation(self):
        with self.assertRaisesRegex(ValueError, "bounds"):
            make_cases(SEEDS[0], per_template=1)
        with self.assertRaisesRegex(ValueError, "bounds"):
            make_cases(9999)


if __name__ == "__main__":
    unittest.main()
