"""Staged curriculum constraints: same-source, no promotion, bounded compute."""
import unittest
from pathlib import Path

from forge_core.neural.preschool_staged import DEFAULT_STAGES, curriculum


class TestPreschoolStaged(unittest.TestCase):
    def test_stages_start_small_and_increase(self):
        sizes = [x[0] for x in DEFAULT_STAGES]
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(sizes[0], 16)
        self.assertEqual(sizes[-1], 512)
        self.assertEqual(sum(x[1] for x in DEFAULT_STAGES), 2600)

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
