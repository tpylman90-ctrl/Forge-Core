"""Verify formal rewards, process labels, policy gradients and no leakage."""
import tempfile
import unittest
from pathlib import Path
import random

import torch

from forge_core.neural.rlvr_process_v1 import (
    Task, VerifiablePolicy, ARMS, encode, evaluate, expected_step, experiment,
    generate_task, options, rollout, train, verify_step, verify_trace,
)


class TestExactOracles(unittest.TestCase):
    def test_arithmetic_chain_distinguishes_answer_from_proof(self):
        problem = Task("math", 3, (4, -2), ((0, 1), (1, 2)), 5)
        self.assertEqual(expected_step(problem, 0, 3), 7)
        self.assertEqual(expected_step(problem, 1, 7), 5)
        self.assertEqual(verify_trace(problem, (7, 5)), {
            "process": [True, True], "final": True,
            "complete_verified_proof": True,
        })
        # A lucky final answer does not count as a valid proof.
        self.assertEqual(verify_trace(problem, (6, 5)), {
            "process": [False, False], "final": True,
            "complete_verified_proof": False,
        })
        self.assertFalse(verify_step(problem, 0, 3, 8))
        with self.assertRaises(ValueError):
            expected_step(problem, 2, 3)

    def test_symbolic_implication_chain_is_checked_locally(self):
        # A -> B; B -> C; antecedents must already be established.
        p = Task("logic", 1 << 0, (0, 0), ((0, 1), (1, 2)), 7)
        self.assertEqual(expected_step(p, 0, 1), 3)
        self.assertEqual(expected_step(p, 1, 3), 7)
        self.assertEqual(verify_trace(p, (3, 7))["complete_verified_proof"], True)
        self.assertEqual(expected_step(p, 1, 1), 1)
        self.assertFalse(verify_trace(p, (1, 7))["complete_verified_proof"])
        self.assertEqual(verify_trace(p, (1, 1))["process"], [False, True])
        with self.assertRaises(ValueError):
            Task("logic", 1, (0, 0), ((0, 0), (1, 2)), 0)

    def test_candidate_choices_both_include_exactly_one_valid_next_state(self):
        for split in ("train", "exam"):
            for i in range(70):
                task = generate_task(90, i, split)
                state = task.initial
                for step in range(2):
                    for k in range(4):
                        candidates = options(task, step, state, random.Random(i + 7 * k))
                        labels = [verify_step(task, step, state, choice)
                                  for choice in candidates]
                        self.assertEqual(sum(labels), 1)
                    state = expected_step(task, step, state)
                    self.assertEqual(encode(task, step, task.initial, state).shape, (32,))
        with self.assertRaises(ValueError):
            Task("invalid", 0, (0, 0), ((0, 1), (1, 2)), 0)

    def test_proof_verifier_rejects_missing_or_extra_steps(self):
        p = generate_task(12, 0)
        for t in ((1,), (1, 2, 3), ()):
            with self.assertRaises(ValueError):
                verify_trace(p, t)


class TestPolicyResearch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        torch.set_num_threads(2)

    def test_reward_and_process_labels_never_enter_input_features(self):
        problem = Task("math", 2, (3, -1), ((0, 1), (1, 2)), 4)
        x = encode(problem, 0, 2, 5)
        y = encode(problem, 0, 2, 6)
        self.assertEqual(x.shape[0], 32)
        self.assertFalse(torch.equal(x, y))
        head = VerifiablePolicy()
        sampled = rollout(head, problem, random.Random(99))
        self.assertEqual(len(sampled["trace"]), 2)
        self.assertEqual(len(sampled["verifier_labels"]), 4)
        self.assertEqual(sum(sampled["verifier_labels"]), 2)
        self.assertIsInstance(sampled["logp"].requires_grad, bool)
        self.assertTrue(sampled["logp"].requires_grad)
        self.assertTrue(sampled["process_loss"].requires_grad)
        result = sampled["verdict"]
        self.assertEqual(sampled["reward_outcome"], float(result["final"]))
        self.assertEqual(sampled["reward_process"], float(result["final"]) +
                         0.25 * sum(result["process"]))

    def test_reproducible_model_training_and_exact_trace_metric(self):
        policy, curve = train(arm="process_supervised", steps=2, seed=42,
                              examples_per_update=2, rollouts_per_example=2)
        self.assertEqual(len(curve), 2)
        report = evaluate(policy, seed=777, n=12)
        self.assertEqual(report["total"], 12)
        self.assertEqual(report["total_local_steps"], 24)
        self.assertLessEqual(report["all_steps_and_final"], report["outcome_correct"])
        self.assertLessEqual(report["all_steps_and_final"] * 2,
                             report["locally_valid_steps"])
        with self.assertRaises(ValueError):
            train(arm="unknown", steps=1, seed=3)
        with self.assertRaises(ValueError):
            evaluate(policy, n=0)

    def test_pilot_metrics_and_checkpoint_not_promoted(self):
        # Minimal genuine on-policy training, not expensive CI.
        with tempfile.TemporaryDirectory() as folder:
            report = experiment(Path(folder), seed=109, steps=10)
            self.assertEqual(report["status"], "research_only")
            self.assertFalse(report["original_seed_model_updated"])
            self.assertEqual(set(report["arms"]), set(ARMS))
            self.assertTrue((Path(folder) / "rlvr-process-v1.json").exists())
            for arm, output in report["arms"].items():
                self.assertEqual(output["exam"]["total"], 200)
                self.assertEqual(output["curve"][-1]["update"], 10)
            with self.assertRaises(ValueError):
                experiment(Path(folder), steps=400)


if __name__ == "__main__":
    unittest.main()
