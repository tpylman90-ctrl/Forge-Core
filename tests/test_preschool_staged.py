"""Staged curriculum constraints: same-source, no promotion, bounded compute."""
import unittest
from pathlib import Path

from forge_core.neural.preschool_staged import DEFAULT_STAGES, CONTRAST_STAGES, curriculum


class TestPreschoolStaged(unittest.TestCase):
    def test_stages_start_small_and_increase(self):
        sizes = [x[0] for x in DEFAULT_STAGES]
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(sizes[0], 16)
        self.assertEqual(sizes[-1], 512)
        self.assertEqual(sum(x[1] for x in DEFAULT_STAGES), 2600)

    def test_counterfactual_stage_sizes_and_bounded_budget(self):
        sizes = [row[0] for row in CONTRAST_STAGES]
        self.assertEqual(sizes, [32, 64, 128, 256, 512])
        self.assertTrue(all(size % 32 == 0 for size in sizes))
        self.assertEqual(sum(row[1] for row in CONTRAST_STAGES), 3250)
        with self.assertRaisesRegex(ValueError, "Paired stages"):
            curriculum(folder=Path("missing"), output=Path("missing"),
                       paired=True, stages=((16, 10, 0, 0.0003),))

    def test_invalid_bounds_refuse_before_reading_files(self):
        with self.assertRaisesRegex(ValueError, "Invalid"):
            curriculum(folder=Path("missing"), output=Path("nowhere"),
                       stages=((16, 0, 0.0, 0.0003),))
        with self.assertRaisesRegex(ValueError, "Too many"):
            curriculum(folder=Path("missing"), output=Path("nowhere"),
                       stages=((16, 1800, 0.0, 0.0003),
                               (64, 1800, 0.1, 0.0003),
                               (256, 1800, 0.2, 0.0003)))


if __name__ == "__main__":
    unittest.main()
