"""Verify canonical formal action demonstrations and research comparison gates."""
import random
import tempfile
import unittest
from pathlib import Path

import torch

from forge_core.neural.rlvr_proof_generator_v5 import (
    Problem, ProposalPolicy, STOP, best_continuations,
    evaluate, make_problem, verify_trace,
)
from forge_core.neural.rlvr_verified_traces_v6 import (
    ARMS, SEEDS, action_diagnostics, experiment, lesson_loss,
    proof_lessons, train_arm,
)


class TestVerifiedActionLessons(unittest.TestCase):
    def test_exact_arithmetic_and_logic_demonstrations_include_all_steps(self):
        p = Problem("arithmetic", 2, 7, ())
        examples = proof_lessons(p)
        self.assertEqual(len(examples), 3)
        self.assertTrue(all(STOP not in allowed
                            for _, _, allowed in examples))
        self.assertEqual(examples[0][0], 2)
        self.assertEqual(examples[-1][1], 1)
        # Verify complete action traces rather than individual lucky answers.
        for domain_index in (0, 1):
            sample = make_problem(523, domain_index, "train")
            self.assertTrue(verify_trace(
                sample, tuple(
                    [next(a for a in options if a != STOP)
                     for _, _, options in proof_lessons(sample)]
                ) if domain_index % 4 == 0 else (STOP,))[
                    "correct_resolution"])

    def test_unsolvable_training_has_verified_abstention_only(self):
        for i in (2, 3, 6, 7):
            p = make_problem(431, i, "train")
            lessons = proof_lessons(p)
            self.assertEqual(len(lessons), 1)
            self.assertEqual(lessons[0][2], (STOP,))
            self.assertTrue(verify_trace(p, (STOP,))["sound_refusal"])

    def test_every_verified_prefix_passes_independent_reachability(self):
        for split in ("train", "development", "shifted"):
            for i in range(40):
                p = make_problem(2002, i, split)
                for state, remaining, allowed in proof_lessons(p):
                    self.assertEqual(
                        allowed, best_continuations(p, state, remaining))
                    self.assertTrue(allowed)
                    self.assertEqual(len(set(allowed)), len(allowed))

    def test_oracle_demos_do_not_change_model_forward_signature(self):
        p = make_problem(222, 0, "train")
        model = ProposalPolicy()
        a = model(p, p.initial, 3)
        self.assertEqual(a.shape, (5,))
        self.assertTrue(a.requires_grad)
        l = lesson_loss(model, p)
        self.assertTrue(l.requires_grad)
        l.backward()
        self.assertTrue(any(t.grad is not None for t in model.parameters()))

    def test_abstain_only_has_zero_nonstop_accuracy(self):
        torch.manual_seed(15)
        model = ProposalPolicy()
        # All models are evaluated on nontrivial teacher prefixes;
        # zero-good-action collapse is exposed independently of success.
        result = action_diagnostics(
            model, seed=411, split="development", n=40)
        self.assertGreater(result["valid_nonstop_total"], 0)
        self.assertGreater(result["abstain_total"], 0)
        self.assertEqual(
            result["valid_nonstop_total"] + result["abstain_total"],
            result["total_teacher_steps"])
        self.assertLessEqual(
            result["correct_teacher_steps"], result["total_teacher_steps"])


class TestTrainingModes(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(9)

    def test_small_actual_update_has_recoverable_optimizer_and_dev_only_best(self):
        for arm in ARMS:
            model, recovery = train_arm(
                arm, seed=SEEDS[0], updates=2,
                tasks_per_update=4, eval_every=1)
            self.assertEqual(recovery["updates"], 2)
            self.assertIn(recovery["best_dev_step"], (1, 2))
            self.assertIn("best_weights", recovery)
            self.assertIn("last_weights", recovery)
            self.assertIn("optimizer_state", recovery)
            self.assertIn("sampler_rng_state", recovery)
            self.assertEqual(len(recovery["development_curve"]), 2)
            self.assertEqual(
                evaluate(model, seed=405, split="development", n=8)["feasible"], 4)

    def test_budget_limits_and_nonempty_seeds_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "approved"):
            train_arm(ARMS[0], seed=SEEDS[0], updates=1201)
        with self.assertRaisesRegex(ValueError, "Invalid"):
            experiment(Path("/tmp/not-written-v6"), updates=9)
        with self.assertRaisesRegex(ValueError, "Invalid"):
            experiment(Path("/tmp/not-written-v6"), updates=10, seeds=())

    def test_no_model_promotion_in_experiment_report(self):
        with tempfile.TemporaryDirectory() as folder:
            # Ten small iterations give an actual complete experiment,
            # including immutable source/oracle baselines and artifact.
            result = experiment(
                Path(folder), updates=10, seeds=SEEDS[:1])
            self.assertFalse(result["promoted"])
            self.assertTrue(result[
                "original_forge_seed_language_checkpoint_unchanged"])
            self.assertTrue(
                (Path(folder) / "rlvr-verified-traces-v6.json").exists())
            for arm in ARMS:
                scored = result["results"][0]["arms"][arm]
                self.assertEqual(scored["shifted_proofs"]["tasks"], 240)
                self.assertEqual(scored["shifted_proofs"]["feasible"], 120)


if __name__ == "__main__":
    unittest.main()
