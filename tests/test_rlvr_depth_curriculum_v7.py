"""Forge v7 proof curriculum: exact depth, split safety and controlled training."""
import tempfile
import unittest
from pathlib import Path

import torch

from forge_core.neural.rlvr_proof_generator_v5 import (
    STOP, make_problem, oracle_trace, verify_trace,
)
from forge_core.neural.rlvr_depth_curriculum_v7 import (
    ARMS, SEEDS, evaluate_depth, experiment, problem_at_depth,
    proof_depth, staged_task, train_arm,
)
from forge_core.neural.rlvr_verified_traces_v6 import proof_lessons


class TestExactDifficulty(unittest.TestCase):
    def test_all_three_depths_have_real_complete_proofs(self):
        for split in ("train", "development", "shifted"):
            for i in range(80):
                for required in (1, 2, 3):
                    p = problem_at_depth(321, i, split, required)
                    original = make_problem(321, i, split)
                    old_depth = proof_depth(original)
                    if old_depth is None:
                        self.assertIsNone(proof_depth(p))
                        self.assertEqual(oracle_trace(p), (STOP,))
                        self.assertTrue(verify_trace(
                            p, oracle_trace(p))["sound_refusal"])
                    else:
                        self.assertEqual(proof_depth(p), required)
                        script = oracle_trace(p)
                        self.assertEqual(len(script), required)
                        self.assertTrue(verify_trace(p, script)["proved"])
                        self.assertTrue(all(
                            STOP not in choice
                            for _, _, choice in proof_lessons(p)))

    def test_phase_one_two_three_retain_old_short_goals(self):
        seed = 531
        n = 90
        p_first = [staged_task(seed, i, 1, n) for i in range(80)]
        p_middle = [staged_task(seed, i, 50, n) for i in range(80)]
        p_final = [staged_task(seed, i, 90, n) for i in range(80)]
        self.assertEqual({phase for _, phase in p_first}, {1})
        self.assertEqual({phase for _, phase in p_middle}, {2})
        self.assertEqual({phase for _, phase in p_final}, {3})
        self.assertEqual({proof_depth(p) for p, _ in p_first}, {1, None})
        self.assertEqual({proof_depth(p) for p, _ in p_middle},
                         {1, 2, None})
        self.assertEqual({proof_depth(p) for p, _ in p_final},
                         {1, 2, 3, None})
        with self.assertRaises(ValueError):
            staged_task(seed, 1, 0, n)
        with self.assertRaises(ValueError):
            problem_at_depth(seed, 1, "train", 4)

    def test_unknown_goal_does_not_become_solvable(self):
        p = problem_at_depth(5321, 2, "shifted", 1)
        self.assertIsNone(proof_depth(p))
        self.assertEqual(oracle_trace(p), (STOP,))
        self.assertTrue(verify_trace(p, (STOP,))["sound_refusal"])

    def test_balanced_metrics_do_not_merge_refusal_and_proof(self):
        torch.manual_seed(11)
        from forge_core.neural.rlvr_proof_generator_v5 import ProposalPolicy
        m = ProposalPolicy()
        for d in (1, 2, 3):
            score = evaluate_depth(m, seed=3131,
                                   split="development", depth=d, n=40)
            self.assertEqual(score["feasible"], 20)
            self.assertEqual(score["infeasible"], 20)
            self.assertEqual(score["total"], 40)
            self.assertLessEqual(score["proved"], 20)
            self.assertLessEqual(score["sound_refusal"], 20)
            self.assertLessEqual(score["proved"] +
                                 score["sound_refusal"], 40)
            self.assertGreater(score["nonstop_total"], 0)


class TestCurriculumTraining(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(98)

    def test_each_training_arm_has_proper_dev_selection_and_updates(self):
        for arm in ARMS:
            model, state = train_arm(
                arm, seed=SEEDS[0], updates=2,
                tasks_per_update=4, eval_every=1)
            self.assertEqual(state["updates"], 2)
            self.assertIn(state["best_step"], (1, 2))
            self.assertEqual(len(state["curve"]), 2)
            self.assertIn("best_weights", state)
            self.assertIn("last_weights", state)
            self.assertIn("optimizer_state", state)
            self.assertIn("sampler_rng_state", state)
            score = evaluate_depth(
                model, seed=491, split="development", depth=1, n=8)
            self.assertEqual(score["total"], 8)

    def test_experiment_output_flags_no_language_model_promotion(self):
        with tempfile.TemporaryDirectory() as d:
            report = experiment(Path(d), updates=10, seeds=SEEDS[:1])
            self.assertFalse(report["promoted"])
            self.assertTrue(
                report["original_forge_seed_language_checkpoint_unchanged"])
            self.assertEqual(report["updates_per_arm_seed"], 10)
            self.assertEqual(set(report["arms"]), set(ARMS))
            self.assertTrue((Path(d) / "rlvr-depth-curriculum-v7.json").exists())
            for arm in ARMS:
                result = report["results"][0]["arms"][arm]
                self.assertEqual(set(result["shifted_final_by_depth"]),
                                 {"1", "2", "3"})
                for level in ("1", "2", "3"):
                    self.assertEqual(result[
                        "shifted_final_by_depth"][level]["total"], 240)

    def test_bounded_compute_and_seed_checks(self):
        with self.assertRaises(ValueError):
            train_arm(ARMS[0], seed=1, updates=2000)
        with self.assertRaises(ValueError):
            train_arm("unknown", seed=1, updates=2)
        with self.assertRaises(ValueError):
            experiment(Path("/tmp/notwritten-v7"), updates=9)
        with self.assertRaises(ValueError):
            experiment(Path("/tmp/notwritten-v7"), updates=12, seeds=())


if __name__ == "__main__":
    unittest.main()
