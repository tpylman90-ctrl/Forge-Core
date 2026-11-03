"""No shortcut by answer position or unchanged counterfactual meaning."""
import unittest

from forge_core.neural.paired_contrasts import paired_tasks


class TestPairedContrasts(unittest.TestCase):
    def test_four_tasks_per_case_and_balanced(self):
        data = paired_tasks(512, 410)
        self.assertEqual(data, paired_tasks(512, 410))
        self.assertEqual(len(data), 512)
        self.assertEqual(len(set(row.id for row in data)), 512)
        self.assertEqual(data[:128], paired_tasks(128, 410))
        for skill in ("color", "recipient", "location", "count"):
            subset = [x for x in data if x.skill == skill]
            self.assertEqual(len(subset), 128)
            self.assertEqual(sum(x.answer == "A" for x in subset), 64)
            self.assertEqual(sum(x.answer == "B" for x in subset), 64)

    def test_both_choice_orders_and_counterfactual_values(self):
        data = paired_tasks(32, 41)
        for i in range(0, len(data), 4):
            first, second, third, fourth = data[i:i+4]
            self.assertEqual([x.answer for x in (first, second, third, fourth)],
                             ["A", "B", "A", "B"])
            self.assertNotEqual(first.prompt, second.prompt)
            self.assertNotEqual(third.prompt, fourth.prompt)
            # Counterfactual should change the answer being described.
            # The A option in the first and third states must differ.
            a1 = first.prompt.split("\nA) ")[1].split("\n")[0]
            a2 = third.prompt.split("\nA) ")[1].split("\n")[0]
            self.assertNotEqual(a1, a2)
            # Both variants of one state carry the same story and question.
            self.assertEqual(first.prompt.split("\nA) ")[0],
                             second.prompt.split("\nA) ")[0])
            self.assertEqual(third.prompt.split("\nA) ")[0],
                             fourth.prompt.split("\nA) ")[0])

    def test_invalid_bounds(self):
        with self.assertRaisesRegex(ValueError, "multiple"):
            paired_tasks(17)
        with self.assertRaisesRegex(ValueError, "multiple"):
            paired_tasks(0)


if __name__ == "__main__":
    unittest.main()
