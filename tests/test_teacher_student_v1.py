"""Teacher/student model study should reject unsafe budgets and absent artifacts."""
import unittest
from pathlib import Path
from forge_core.neural.teacher_student_v1 import student_study


class TestTeacherStudentV1(unittest.TestCase):
    def test_bounds_enforced_before_source_reads(self):
        for params in (
            {"updates":0}, {"updates":2001}, {"seed":0},
            {"batch_size":7}, {"language_weight":0.91},
            {"learning_rate":-0.05},
        ):
            with self.subTest(params=params):
                with self.assertRaisesRegex(ValueError, "Out-of-bounds"):
                    student_study(
                        original_dir=Path("missing"),candidate_dir=Path("missing"),
                        lesson_dir=Path("missing"),output=Path("missing"),
                        **params)

    def test_absent_source_refused_without_model_promotion(self):
        with self.assertRaises(Exception):
            student_study(
                original_dir=Path("missing"),candidate_dir=Path("missing"),
                lesson_dir=Path("missing"),output=Path("missing"),
                updates=1)


if __name__ == "__main__":
    unittest.main()
