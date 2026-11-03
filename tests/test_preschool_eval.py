"""Preschool-capability task construction is reproducible and balanced."""
import unittest
from forge_core.neural.preschool_eval import build_prompts, questions, MANUAL_PROMPTS


class TestPreschoolEval(unittest.TestCase):
    def test_scoring_tasks_answer_position_balanced(self):
        tasks=build_prompts()
        self.assertEqual(len(questions()),20)
        self.assertEqual(len(tasks),40)
        self.assertEqual([x["gold"] for x in tasks].count("A"),20)
        self.assertEqual([x["gold"] for x in tasks].count("B"),20)
        self.assertEqual(set(x["skill"] for x in tasks),
                         {"color","recipient","location","count"})
        self.assertEqual(len({x["id"] for x in tasks}),40)
        self.assertEqual(len(MANUAL_PROMPTS),15)
        for i in range(0,len(tasks),2):
            self.assertEqual(tasks[i]["gold"],"A")
            self.assertEqual(tasks[i+1]["gold"],"B")
            self.assertEqual(tasks[i]["skill"],tasks[i+1]["skill"])
        self.assertEqual(tasks,build_prompts())


if __name__=="__main__":
    unittest.main()
