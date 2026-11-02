import subprocess
import tempfile
import unittest
from pathlib import Path

from forge_core.language.local_teacher import LocalTeacher


class TestLocalTeacher(unittest.TestCase):
    def test_local_subprocess_only_and_no_automatic_saving(self):
        with tempfile.TemporaryDirectory() as temp:
            binary = Path(temp) / "llama-cli"
            model = Path(temp) / "model.gguf"
            binary.touch()
            model.touch()
            captures = []
            def runner(argv, **kwargs):
                captures.append((argv, kwargs))
                return subprocess.CompletedProcess(argv, 0, "Why did you choose that path?", "")
            teacher = LocalTeacher(binary, model, run=runner)
            question = teacher.question("Rejected design B", ["Proportions before details"])
            self.assertEqual(question, "Why did you choose that path?")
            argv, kwargs = captures[0]
            self.assertEqual(argv[0], str(binary))
            self.assertIn(str(model), argv)
            self.assertIn("Rejected design B", argv[argv.index("-p") + 1])
            self.assertNotIn("shell", kwargs)
            self.assertTrue(kwargs["check"])
            self.assertEqual(kwargs["timeout"], 180)

    def test_missing_model_and_empty_observation(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "existing"
            p.touch()
            with self.assertRaises(FileNotFoundError):
                LocalTeacher(p, Path(temp) / "missing.gguf")
            teacher = LocalTeacher(p, p)
            with self.assertRaises(ValueError):
                teacher.question("   ")


if __name__ == "__main__":
    unittest.main()
