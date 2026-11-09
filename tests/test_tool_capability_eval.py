"""Fresh specialist benchmark construction stays balanced and isolated."""
import unittest

from forge_core.neural.multilane_v1 import SOURCES as TRAIN_SOURCES
from forge_core.neural.tool_capability_eval import HOLDOUTS, QUESTIONS, rotated_tasks


class TestToolCapabilityEval(unittest.TestCase):
    def test_fresh_pages_are_disjoint_from_training_pages(self):
        for lane, spec in HOLDOUTS.items():
            self.assertTrue(set(spec["urls"]).isdisjoint(TRAIN_SOURCES[lane]["urls"]))
            self.assertEqual(len(spec["urls"]), 8)

    def test_each_lane_has_balanced_four_position_rotations(self):
        for lane in ("blender", "godot"):
            tasks = rotated_tasks(lane)
            self.assertEqual(len(QUESTIONS[lane]) * 4, len(tasks))
            self.assertEqual(len({task["id"] for task in tasks}), len(tasks))
            self.assertEqual([task["gold"] for task in tasks].count("A"), len(QUESTIONS[lane]))
            self.assertEqual([task["gold"] for task in tasks].count("B"), len(QUESTIONS[lane]))
            self.assertEqual([task["gold"] for task in tasks].count("C"), len(QUESTIONS[lane]))
            self.assertEqual([task["gold"] for task in tasks].count("D"), len(QUESTIONS[lane]))
            for i in range(0, len(tasks), 4):
                self.assertEqual([tasks[i + j]["gold"] for j in range(4)],
                                 ["A", "B", "C", "D"])
                self.assertEqual(len({tasks[i + j]["base_id"] for j in range(4)}), 1)


if __name__ == "__main__":
    unittest.main()
