"""Copy v7 length-generalization experiment integrity gates."""
import unittest
from pathlib import Path
from forge_core.neural.english_length_v7 import (
    mixed_training, experiment, SEEDS,
)
from forge_core.neural.english_learned_pointer_v6 import (
    assert_firewall, generate_pairs, validate_pairs,
)


class TestLengthV7(unittest.TestCase):
    def test_mixed_lengths_reproducible_and_partitioned(self):
        a=mixed_training(80,13741)
        self.assertEqual(a,mixed_training(80,13741))
        validate_pairs(a)
        self.assertEqual(sum(len(x.original)>8 for x in a),40)
        self.assertEqual(sum(len(x.original)<=8 for x in a),40)
        self.assertTrue(all(x.split=="train" for x in a))
        assert_firewall(a,generate_pairs("long",24,20261107),
                        generate_pairs("exam",24,20261106),
                        generate_pairs("dev",24,20261105))

    def test_invalid_budget_fails_before_source_io(self):
        for kw in ({"updates":0},{"updates":1001},
                   {"seeds":()},{"seeds":(13741,13741)},
                   {"seeds":(123,)}):
            with self.subTest(kw=kw),self.assertRaisesRegex(
                    ValueError,"Outside preregistered"):
                experiment(source=Path("missing"),out=Path("missing"),**kw)


if __name__=="__main__":
    unittest.main()
