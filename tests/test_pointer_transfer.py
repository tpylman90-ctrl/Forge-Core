"""Novel pointer exam must be source-grounded, disjoint, and immutable."""
import unittest
from pathlib import Path

from forge_core.neural.pointer_transfer import (
    EXAM_PEOPLE, EXAM_WORDS, make_exam, examine, SKILLS,
)
from forge_core.neural.preschool_copy import (
    TRAIN_PEOPLE, EVAL_PEOPLE, TRAIN_WORDS, EVAL_WORDS,
)


class TestPointerTransfer(unittest.TestCase):
    def test_novel_people_objects_and_gold_in_source(self):
        self.assertFalse(set(EXAM_PEOPLE) & (set(TRAIN_PEOPLE)|set(EVAL_PEOPLE)))
        self.assertFalse(set(EXAM_WORDS) & (set(TRAIN_WORDS)|set(EVAL_WORDS)))
        rows=make_exam(5101)
        self.assertEqual(rows,make_exam(5101))
        self.assertEqual(len(rows),120)
        self.assertEqual({x.skill for x in rows},set(SKILLS))
        for skill in SKILLS:
            self.assertEqual(sum(x.skill==skill for x in rows),20)
        for row in rows:
            self.assertTrue(row.prompt.endswith("\nAnswer:"))
            self.assertIn(row.answer.lower(),row.prompt.lower())
        combined=[r.prompt for seed in (5101,5102,5103)
                  for r in make_exam(seed)]
        self.assertEqual(len(combined),len(set(combined)))

    def test_sanity_boundaries_before_io(self):
        with self.assertRaisesRegex(ValueError,"budget"):
            make_exam(5101,per_skill=1)
        with self.assertRaises(Exception):
            examine(source=Path("missing"),parent=Path("missing"),
                    pointer=Path("missing"),output=Path("missing/out.json"))


if __name__=="__main__":
    unittest.main()
