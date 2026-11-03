"""Original, deterministic direct-answer tasks must preserve explicit truth."""
import unittest

from forge_core.neural.paired_contrasts import paired_tasks
from forge_core.neural.preschool_direct import (
    direct_examples, shuffle_answers, normalize_answer, encode,
)
from forge_core.neural.preschool_eval import build_prompts
from forge_core.neural.preschool_qa import Task


class TestDirectAnswers(unittest.TestCase):
    def test_gold_and_counterfactual_question_conversion(self):
        choices = paired_tasks(128, seed=410)
        rows = direct_examples(choices)
        self.assertEqual(len(rows), 64)
        self.assertEqual(rows, direct_examples(choices))
        self.assertEqual(len({x.id for x in rows}), 64)
        self.assertEqual(len({x.prompt for x in rows}), 64)
        for skill in ("color", "recipient", "location", "count"):
            self.assertEqual(len([x for x in rows if x.skill == skill]), 16)
        # In two opposing states of the same semantic case, answer must change.
        for i in range(0, len(rows), 2):
            self.assertNotEqual(rows[i].answer, rows[i + 1].answer)

    def test_shuffled_control_same_questions_and_answer_histograms(self):
        gold = direct_examples(paired_tasks(128))
        shuffled = shuffle_answers(gold, 42)
        self.assertEqual([x.prompt for x in gold],
                         [x.prompt for x in shuffled])
        self.assertNotEqual([x.answer for x in gold],
                            [x.answer for x in shuffled])
        for skill in ("color", "recipient", "location", "count"):
            self.assertEqual(
                sorted(x.answer for x in gold if x.skill == skill),
                sorted(x.answer for x in shuffled if x.skill == skill))

    def test_holdout_mirrors_become_one_question_each(self):
        rows = [
            Task(x["id"], x["skill"], x["prompt"], x["gold"])
            for x in build_prompts()
        ]
        derived = direct_examples(rows)
        self.assertEqual(len(derived), 20)
        self.assertEqual({x.skill for x in derived},
                         {"color", "recipient", "location", "count"})

    def test_normalize_answer_and_reject_bad_examples(self):
        self.assertEqual(normalize_answer("  BLUE.  "), "blue")
        self.assertEqual(normalize_answer(" Ada.\nMore text"), "ada")
        with self.assertRaisesRegex(ValueError, "Missing options"):
            direct_examples([
                Task("x:mirror0", "color", "not a prompt", "A"),
                Task("x:mirror1", "color", "not a prompt", "B"),
            ])


if __name__ == "__main__":
    unittest.main()
