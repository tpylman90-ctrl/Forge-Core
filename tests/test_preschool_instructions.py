"""Unseen one-step instructions: deterministic, unique, safely isolated."""
import unittest
from pathlib import Path
from forge_core.neural.preschool_instructions import (
    AUDIT_SKILLS, instruction_examples, examine,
)

class TestInstructionAudit(unittest.TestCase):
    def test_exactly_fifty_verifiable_gold_instructions(self):
        tasks=instruction_examples(233)
        self.assertEqual(len(tasks),50)
        self.assertEqual(tasks,instruction_examples(233))
        self.assertEqual(len({x.prompt for x in tasks}),50)
        for skill in AUDIT_SKILLS:
            self.assertEqual(len([x for x in tasks if x.skill==skill]),10)
        for task in tasks:
            self.assertTrue(task.answer)
            self.assertIn("Instruction:",task.prompt)
            self.assertTrue(task.prompt.endswith("\nAnswer:"))
            self.assertIn(task.answer.lower(),task.prompt.lower())
            self.assertNotIn("\nA) ",task.prompt)

    def test_different_seeds_change_questions_without_changing_task_count(self):
        a=instruction_examples(233)
        b=instruction_examples(234)
        self.assertEqual(len(b),50)
        self.assertNotEqual([x.prompt for x in a],
                            [x.prompt for x in b])

    def test_missing_unapproved_source_is_rejected(self):
        with self.assertRaises(ValueError):
            examine(Path("missing"),Path("missing"),Path("missing/out.json"))

if __name__=="__main__":
    unittest.main()
