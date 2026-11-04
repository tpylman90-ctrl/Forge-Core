"""Verifier-class balance, stage-2 coverage and development-only rejection."""
import random
import tempfile
import unittest
from pathlib import Path

import torch

from forge_core.neural.rlvr_abstention_v3 import (
    AbstentionPolicy, SEEDS, case_for, candidates, evaluate, gold_action, rollout,
)
from forge_core.neural.rlvr_process_v1 import expected_step
from forge_core.neural.rlvr_step_verifier_v4 import (
    ARMS, THRESHOLDS, balanced_gate, candidate_metrics,
    canonical_step_examples, fit, select_dev_threshold, experiment,
)


class AlwaysInvalid(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([0., 0., 0.]), torch.full((2,), -10.)


class TestVerifierData(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_training_includes_both_steps_of_impossible_prefixes(self):
        for index in range(64):
            case = case_for(SEEDS[0], index, "train")
            steps = canonical_step_examples(SEEDS[0], index, "train")
            self.assertEqual(len(steps), 2)
            for step, (features, target, labels) in enumerate(steps):
                self.assertEqual(features.shape, (2, 32))
                self.assertEqual(labels.shape, (2,))
                self.assertEqual(int(labels.sum()), int(case.availability[step]))
                self.assertEqual(target == 2, not case.availability[step])
            if not case.availability[0] and case.availability[1]:
                # This second step was still generated even though an
                # acting policy would have had to refuse at step zero.
                self.assertLess(steps[1][1], 2)

    def test_one_positive_per_four_unlabeled_proposals(self):
        positives = 0
        proposals = 0
        for i in range(240):
            for _, _, labels in canonical_step_examples(301, i, "train"):
                positives += int(labels.sum())
                proposals += labels.numel()
        self.assertEqual(proposals, 960)
        self.assertEqual(positives, 240)

    def test_always_negative_is_not_75_percent_evidence(self):
        observed = candidate_metrics(AlwaysInvalid(), seed=311,
                                     split="development", n=80)
        self.assertEqual(observed["candidate_count"], 320)
        self.assertEqual(observed["true_positive"], 0)
        self.assertEqual(observed["false_positive"], 0)
        self.assertEqual(observed["false_negative"], 80)
        self.assertEqual(observed["true_negative"], 240)
        self.assertEqual(observed["raw_accuracy"], .75)
        self.assertEqual(observed["balanced_accuracy"], .5)
        self.assertEqual(observed["valid_recall"], 0.)
        self.assertEqual(observed["always_invalid_accuracy"], .75)

    def test_negative_checker_rejects_even_feasible_proofs(self):
        case = next(case_for(311, i, "train") for i in range(24)
                    if case_for(311, i, "train").availability[0])
        outcome = rollout(AlwaysInvalid(), case, random.Random(81),
                          mode="checker", threshold=.5)
        self.assertEqual(outcome["false_reject"], 1)
        score = evaluate(AlwaysInvalid(), seed=311, split="shifted",
                         n=80, mode="checker", threshold=.5)
        self.assertEqual(score["resolved"], 40)
        self.assertEqual(score["correct_feasible"], 0)
        self.assertEqual(score["invalid_accepted"], 0)
        with self.assertRaises(ValueError):
            rollout(AlwaysInvalid(), case, random.Random(81),
                    mode="checker", threshold=1.1)


class TestLearning(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(301)

    def test_development_gate_penalizes_zero_coverage(self):
        none = {"feasible_proof_rate": 0., "infeasible_safe_rate": .7,
                "resolved_rate": .52, "accepted_invalid_rate": 0.}
        both = {"feasible_proof_rate": .4, "infeasible_safe_rate": .5,
                "resolved_rate": .47, "accepted_invalid_rate": .1}
        self.assertEqual(balanced_gate(none)[0], 0)
        self.assertGreater(balanced_gate(both)[0], 0)

    def test_threshold_selection_uses_development_grid_only(self):
        model = AbstentionPolicy()
        threshold, report = select_dev_threshold(model, seed=301)
        self.assertIn(threshold, THRESHOLDS)
        self.assertEqual(len(report["development_grid"]), len(THRESHOLDS))
        self.assertIn("harmonic", report["criterion"])

    def test_three_actual_training_objectives_and_recovery_metadata(self):
        for arm in ARMS:
            model, curve, checkpoint = fit(
                arm, seed=311, updates=2, tasks_per_update=8)
            self.assertEqual(curve[-1]["update"], 2)
            self.assertEqual(checkpoint["completed_updates"], 2)
            self.assertEqual(checkpoint["arm"], arm)
            self.assertIn("optimizer_state", checkpoint)
            self.assertIn("torch_rng_state", checkpoint)
            observed = candidate_metrics(model, seed=16001,
                                         split="shifted", n=8)
            self.assertEqual(observed["candidate_count"], 32)
            self.assertGreaterEqual(observed["valid_recall"], 0)
            self.assertLessEqual(observed["valid_recall"], 1)

    def test_bound_training_and_seed_validation(self):
        with self.assertRaises(ValueError):
            fit(ARMS[0], seed=311, updates=201)
        with self.assertRaises(ValueError):
            fit("unrecognized", seed=311, updates=2)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                experiment(Path(d), updates=999)
            with self.assertRaises(ValueError):
                experiment(Path(d), updates=10, seeds=())


if __name__ == "__main__":
    unittest.main()
