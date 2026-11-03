"""Memorization diagnostic must report rather than claim ability."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from forge_core.neural.preschool_sanity import probe


class TestPreschoolSanity(unittest.TestCase):
    def test_bounds_fail_before_loading_any_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "bounded"):
                probe(Path(tmp), Path(tmp)/"out.json", steps=0)
            with self.assertRaisesRegex(ValueError, "bounded"):
                probe(Path(tmp), Path(tmp)/"out.json", lr=-0.01)


if __name__ == "__main__":
    unittest.main()
