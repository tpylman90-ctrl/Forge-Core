"""Formal rejection, confidence and out-of-distribution proof safeguards."""
import random
import tempfile
import unittest
from pathlib import Path

import torch

from forge_core.neural.rlvr_process_v1 import encode, verify_step
from forge_core.neural.rlvr_abstention_v3 import (
    ARMS, REJECT, SEEDS, Case, AbstentionPolicy, case_for,
    candidates, evaluate, fit_arm, gold_action, rollout, experiment,
)


class TestFormalAbstention(unittest.TestCase):
    def test_balanced_regimes_and_shifted_numeric_difficulty(self):
        for split in ("train", "development", "shifted"):
            domain_regimes = {}
            for i in range(240):
                case = case_for(313, i, split)
                key = (case.domain, case.availability)
                domain_regimes[key] = domain_regimes.get(key, 0) + 1
                self.assertIn(case.domain, ("math", "logic"))
                self.assertEqual(case.split, split)
            self.assertEqual(len(domain_regimes), 8)
            self.assertEqual(set(domain_regimes.values()), {30})
        train = case_for(313, 0, "train").task
        shifted = case_for(313, 0, "shifted").task
        self.assertEqual(train.kind, shifted.kind)
        self.assertLessEqual(abs(train.initial), 5)
        self.assertGreaterEqual(abs(shifted.initial), 12)

    def test_neither_candidate_valid_requires_reject(self):
        for seed in (211, 313):
            for split in ("train", "shifted"):
                for i in range(32):
                    case = case_for(seed, i, split)
                    for step in range(2):
                        rng = random.Random(i * 41 + step)
                        offered = candidates(case, step, case.task.initial, rng)
                        truth, labels = gold_action(
                            case.task, step, case.task.initial, offered)
                        self.assertEqual(sum(labels), int(case.availability[step]))
                        self.assertEqual(truth == REJECT,
                                         not case.availability[step])
                        self.assertEqual(len(set(offered)), 2)
                        self.assertEqual(
                            tuple(verify_step(case.task, step,
                                              case.task.initial, value)
                                  for value in offered), labels)
                        for value in offered:
                            self.assertEqual(
                                encode(case.task, step,
                                       case.task.initial, value).shape, (32,))

    def test_choice_order_not_a_gold_label_shortcut(self):
        positions = []
        for i in range(200):
            case = case_for(319, i, "train")
            if case.availability[0]:
                selected, _ = gold_action(
                    case.task, 0, case.task.initial,
                    candidates(case, 0, case.task.initial,
                               random.Random(11 + i * 17)))
                positions.append(selected)
        self.assertGreater(positions.count(0), 25)
        self.assertGreater(positions.count(1), 25)

    def test_reject_only_reference_and_exact_checker(self):
        for split in ("development", "shifted"):
            oracle = evaluate(None, seed=317, split=split,
                              n=80, mode="oracle")
            reject = evaluate(None, seed=317, split=split,
                              n=80, mode="always_reject")
            self.assertEqual(oracle["resolved"], 80)
            self.assertEqual(oracle["correct_feasible"], 20)
            self.assertEqual(oracle["correct_infeasible"], 60)
            self.assertEqual(oracle["invalid_accepted"], 0)
            self.assertEqual(oracle["false_reject"], 0)
            self.assertEqual(reject["resolved"], 40)
            self.assertEqual(reject["correct_feasible"], 0)
            self.assertEqual(reject["correct_infeasible"], 40)
            self.assertEqual(set(reject["by_offer_regime"]),
                             {"VV", "VN", "NV", "NN"})

    def test_positive_and_negative_rejects_are_distinct(self):
        rng = random.Random(99)
        has_available = next(case_for(317, i, "train")
                             for i in range(24)
                             if case_for(317, i, "train").availability[0])
        has_none = next(case_for(317, i, "train")
                        for i in range(24)
                        if not case_for(317, i, "train").availability[0])
        wrong_reject = rollout(None, has_available, rng,
                               mode="always_reject")
        correct_reject = rollout(None, has_none, rng,
                                 mode="always_reject")
        self.assertEqual(wrong_reject["false_reject"], 1)
        self.assertFalse(wrong_reject["resolved"])
        self.assertEqual(correct_reject["correct_reject"], 1)
        self.assertTrue(correct_reject["resolved"])

    def test_invalid_or_ambiguous_formal_candidate_set_rejected(self):
        case = case_for(411, 0, "train")
        current = case.task.initial
        with self.assertRaisesRegex(ValueError, "distinct"):
            gold_action(case.task, 0, current, (current, current))
        with self.assertRaisesRegex(ValueError, "two distinct"):
            gold_action(case.task, 0, current, (current,))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            rollout(None, case, random.Random(1), mode="arbitrary")
        with self.assertRaisesRegex(ValueError, "requires"):
            rollout(None, case, random.Random(1), mode="greedy")


class TestLearnedAbstention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(2)

    def test_network_outputs_three_actions_with_learned_reject(self):
        case = case_for(301, 2, "train")
        offered = candidates(case, 0, case.task.initial, random.Random(98))
        x = torch.stack([encode(case.task, 0, case.task.initial, v)
                         for v in offered])
        policy = AbstentionPolicy()
        logits, validity = policy(x)
        self.assertEqual(logits.shape, (3,))
        self.assertEqual(validity.shape, (2,))
        self.assertEqual(logits.argmax().ndim, 0)
        # The model is not passed verifier labels or availability.
        alt = Case(case.task, (not case.availability[0], True),
                   case.domain, case.split)
        self.assertTrue(torch.equal(
            policy(x)[0],
            policy(torch.stack([encode(alt.task, 0, alt.task.initial, v)
                                for v in offered]))[0]))
        with self.assertRaises(ValueError):
            policy(torch.zeros(3, 32))

    def test_true_stochastic_rl_and_process_supervision_have_gradients(self):
        case = case_for(315, 0, "train")
        policy = AbstentionPolicy()
        r = rollout(policy, case, random.Random(991), mode="sample")
        self.assertTrue(r["logp"].requires_grad)
        self.assertTrue(r["actor_ce"].requires_grad)
        self.assertTrue(r["process_bce"].requires_grad)
        self.assertLessEqual(r["checked_candidates"], 4)
        (r["logp"] + r["actor_ce"] + r["process_bce"]).backward()
        self.assertTrue(any(p.grad is not None for p in policy.parameters()))

    def test_short_actual_training_and_proof_metrics(self):
        for arm in ARMS:
            policy, curve, best, recovery = fit_arm(
                arm, SEEDS[0], updates=2, tasks_per_update=2,
                rollouts_per_task=2, development_every=1)
            self.assertEqual(len(curve), 2)
            self.assertIn(best, (1, 2))
            self.assertEqual(recovery["completed_updates"], 2)
            self.assertIn("last_optimizer_state", recovery)
            self.assertIn("best_model_state", recovery)
            self.assertNotEqual(recovery["last_model_state"], {})
            result = evaluate(policy, seed=12911, split="shifted", n=16)
            self.assertEqual(result["total_tasks"], 16)
            self.assertEqual(result["feasible"], 4)
            self.assertEqual(result["infeasible"], 12)
            self.assertGreaterEqual(result["invalid_accepted"], 0)
            self.assertLessEqual(result["complete_proof"], result["feasible"])
            self.assertLessEqual(result["correct_infeasible"],
                                 result["infeasible"])
            self.assertLessEqual(result["resolved"], 16)

    def test_training_budget_rejects_oversized_jobs(self):
        with self.assertRaisesRegex(ValueError, "limits"):
            fit_arm(ARMS[0], SEEDS[0], updates=999)
        with self.assertRaisesRegex(ValueError, "configuration"):
            experiment(Path("/tmp/unreachable"), updates=999)
        with self.assertRaisesRegex(ValueError, "configuration"):
            experiment(Path("/tmp/unreachable"), updates=12, seeds=())


if __name__ == "__main__":
    unittest.main()
