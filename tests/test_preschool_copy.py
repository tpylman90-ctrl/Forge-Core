"""Copy/select/role instruction curriculum: reproducible and leakage-safe."""
import unittest
from pathlib import Path
from forge_core.neural.preschool_copy import (
    SKILLS, TRAIN_PEOPLE, EVAL_PEOPLE, TRAIN_WORDS, EVAL_WORDS,
    tasks, shuffle_within_skill, experiment,
)

class TestCopyInstructionCurriculum(unittest.TestCase):
    def test_splits_are_disjoint_and_each_skill_present(self):
        self.assertFalse(set(TRAIN_PEOPLE)&set(EVAL_PEOPLE))
        self.assertFalse(set(TRAIN_WORDS)&set(EVAL_WORDS))
        tr=tasks("train",2400,900)
        ev=tasks("challenge",120,901)
        self.assertEqual(len(tr),2400)
        self.assertEqual(len(ev),120)
        self.assertEqual(tr,tasks("train",2400,900))
        self.assertFalse({x.prompt for x in tr}&{x.prompt for x in ev})
        self.assertEqual(len({x.prompt for x in tr}),2400)
        self.assertEqual(len({x.prompt for x in ev}),120)
        for skill in SKILLS:
            self.assertEqual(sum(x.skill==skill for x in tr),400)
            self.assertEqual(sum(x.skill==skill for x in ev),20)
        for item in tr+ev:
            self.assertTrue(item.prompt.endswith("\nAnswer:"))
            self.assertIn(item.answer.casefold(),item.prompt.casefold())

    def test_stages_are_cumulative_and_control_is_matched(self):
        full=tasks("train",2400,900)
        self.assertEqual(full[:120],tasks("train",120,900))
        self.assertEqual(full[:600],tasks("train",600,900))
        control=shuffle_within_skill(full[:120],93)
        self.assertEqual([x.prompt for x in control],
                         [x.prompt for x in full[:120]])
        self.assertNotEqual([x.answer for x in control],
                            [x.answer for x in full[:120]])
        for skill in SKILLS:
            self.assertEqual(
                sorted(x.answer for x in control if x.skill==skill),
                sorted(x.answer for x in full[:120] if x.skill==skill))

    def test_bounded_configs_before_io(self):
        with self.assertRaisesRegex(ValueError,"bounded"):
            experiment(source=Path("missing"),candidate=Path("missing"),
                       out=Path("missing"),stages=((120,5001,0.2,0.0001),))
        with self.assertRaisesRegex(ValueError,"Unknown"):
            tasks("private",120)
        with self.assertRaisesRegex(ValueError,"divisible"):
            tasks("train",71)

if __name__=="__main__":
    unittest.main()
