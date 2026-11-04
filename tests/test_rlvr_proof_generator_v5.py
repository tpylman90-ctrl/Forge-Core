"""Proof-action generation: exact verifiers, controls and scratch RL gradients."""
import random
import unittest

import torch

from forge_core.neural.rlvr_proof_generator_v5 import (
    ACTION_NAMES, ARMS, HORIZON, SEEDS, STOP, Problem, ProposalPolicy,
    achieved, balanced_gate, best_continuations, evaluate, experiment,
    features, make_problem, oracle_trace, reachable_states, rollout,
    train_arm, transition, verify_trace,
)


class TestFormalProofActions(unittest.TestCase):
    def test_arithmetic_action_sequence_is_generated_not_offered(self):
        p = Problem("arithmetic", 2, 7, ())
        self.assertEqual(transition(p, 2, 0), 3)
        self.assertEqual(transition(p, 3, 2), 6)
        self.assertEqual(transition(p, 6, 0), 7)
        self.assertIsNone(transition(p, 3, 3))
        self.assertIsNone(transition(p, 40, 2))
        self.assertTrue(verify_trace(p, (0, 2, 0))["proved"])
        self.assertFalse(verify_trace(p, (2, 0, 3))["proved"])
        self.assertFalse(verify_trace(p, (STOP,))["correct_resolution"])
        self.assertTrue(achieved(p, 7))
        self.assertEqual(best_continuations(p, 2, HORIZON) != (STOP,), True)
        self.assertLessEqual(len(oracle_trace(p)), HORIZON)

    def test_formal_implication_chain_and_proven_refusal(self):
        rules = ((0, 1), (1, 2), (2, 3), (5, 6))
        p = Problem("implication", 1 << 0, 3, rules)
        self.assertEqual(transition(p, 1, 0), 3)
        self.assertEqual(transition(p, 3, 1), 7)
        self.assertEqual(transition(p, 7, 2), 15)
        self.assertIsNone(transition(p, 1, 2))
        self.assertTrue(verify_trace(p, (0, 1, 2))["proved"])
        self.assertFalse(verify_trace(p, (0, 2))["correct_resolution"])
        no_way = Problem("implication", 1 << 0, 6, rules)
        self.assertEqual(best_continuations(no_way, 1, HORIZON), (STOP,))
        self.assertTrue(verify_trace(no_way, (STOP,))["sound_refusal"])
        self.assertFalse(verify_trace(p, (STOP,))["sound_refusal"])

    def test_wrong_intermediate_actions_are_not_credited(self):
        p = Problem("arithmetic", 2, 7, ())
        illegal = verify_trace(p, (3,))
        self.assertEqual(illegal["invalid_actions"], 0)  # 2 / 2 is legal
        self.assertFalse(illegal["proved"])
        broken = verify_trace(p, (0, 3, 0))
        self.assertEqual(broken["invalid_actions"], 1)
        self.assertFalse(broken["correct_resolution"])
        with self.assertRaises(ValueError):
            verify_trace(p, (0, 1, 2, 3))
        with self.assertRaises(ValueError):
            verify_trace(p, (9,))
        with self.assertRaises(ValueError):
            best_continuations(p, 2, 4)

    def test_generator_balances_domain_and_solvability_in_every_split(self):
        for split in ("train", "development", "shifted"):
            counts = {}
            for index in range(80):
                p = make_problem(521, index, split)
                success = bool(best_continuations(
                    p, p.initial, HORIZON) != (STOP,))
                key = p.domain, success
                counts[key] = counts.get(key, 0) + 1
                script = oracle_trace(p)
                outcome = verify_trace(p, script)
                self.assertTrue(outcome["correct_resolution"])
                self.assertEqual(outcome["proved"], success)
                self.assertEqual(outcome["sound_refusal"], not success)
                self.assertEqual(outcome["invalid_actions"], 0)
            self.assertEqual(counts, {
                ("arithmetic", True): 20, ("arithmetic", False): 20,
                ("implication", True): 20, ("implication", False): 20,
            })

    def test_shifted_arithmetic_is_unseen_start_magnitude(self):
        for index in range(0, 40, 2):
            train = make_problem(500, index, "train")
            shifted = make_problem(500, index, "shifted")
            self.assertLessEqual(abs(train.initial), 7)
            self.assertGreaterEqual(abs(shifted.initial), 18)

    def test_exact_reference_vs_always_abstain(self):
        for split in ("development", "shifted"):
            oracle = evaluate(None, seed=407, split=split,
                              n=40, mode="oracle")
            none = evaluate(None, seed=407, split=split,
                            n=40, mode="always_stop")
            self.assertEqual(oracle["correct_resolution"], 40)
            self.assertEqual(oracle["proved"], 20)
            self.assertEqual(oracle["sound_refusal"], 20)
            self.assertEqual(oracle["invalid_proposals"], 0)
            self.assertEqual(none["correct_resolution"], 20)
            self.assertEqual(none["proved"], 0)
            self.assertEqual(none["sound_refusal"], 20)
            self.assertEqual(balanced_gate(none)[0], 0)
            self.assertGreater(balanced_gate(oracle)[0], 0)


class TestNeuralProposalPolicy(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_neural_policy_generates_five_actions_and_has_no_gold_arg(self):
        for i in (0, 1):
            p = make_problem(312, i, "train")
            neural = ProposalPolicy()
            logits = neural(p, p.initial, 3)
            self.assertEqual(logits.shape, (len(ACTION_NAMES),))
            self.assertTrue(logits.requires_grad)
            for a in range(STOP + 1):
                self.assertEqual(features(p, p.initial, 3, a).shape, (48,))
            with self.assertRaises(ValueError):
                features(p, p.initial, 3, 8)

    def test_policy_gradient_and_exact_process_labels_are_separate(self):
        p = make_problem(313, 0, "train")
        neural = ProposalPolicy()
        example = rollout(neural, p, random.Random(15))
        self.assertGreaterEqual(example["steps"], 1)
        self.assertIsInstance(example["trace"], tuple)
        self.assertTrue(example["logp"].requires_grad)
        self.assertTrue(example["oracle_process_loss"].requires_grad)
        self.assertLessEqual(example["invalid_proposals"], example["steps"])
        self.assertEqual(example["verdict"]["invalid_actions"],
                         example["invalid_proposals"])
        loss = example["logp"] + example["oracle_process_loss"]
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in neural.parameters()))
        with self.assertRaises(ValueError):
            rollout(neural, p, random.Random(1), mode="uncertified")

    def test_random_and_untrained_are_distinct_from_exact_oracle(self):
        p = make_problem(319, 0, "train")
        neural = ProposalPolicy()
        baseline = rollout(None, p, random.Random(11), mode="random")
        verified = rollout(None, p, random.Random(11), mode="oracle")
        self.assertTrue(verified["verdict"]["correct_resolution"])
        self.assertLessEqual(baseline["invalid_proposals"], baseline["steps"])
        with self.assertRaises(ValueError):
            rollout(None, p, random.Random(11), mode="greedy")

    def test_small_real_policy_update_and_development_selection(self):
        for arm in ARMS:
            model, recovery = train_arm(
                arm, seed=SEEDS[0], updates=2, tasks_per_update=2,
                rollouts_per_task=2, dev_every=1)
            self.assertEqual(recovery["last_update"], 2)
            self.assertIn(recovery["best_dev_step"], (1, 2))
            self.assertIn("last_weights", recovery)
            self.assertIn("best_weights", recovery)
            result = evaluate(model, seed=540, split="development", n=8)
            self.assertEqual(result["feasible"], 4)
            self.assertEqual(result["infeasible"], 4)
            self.assertLessEqual(result["proved"], 4)
            self.assertLessEqual(result["sound_refusal"], 4)

    def test_strict_training_budgets(self):
        with self.assertRaises(ValueError):
            train_arm("unknown", seed=1, updates=1)
        with self.assertRaises(ValueError):
            train_arm(ARMS[0], seed=1, updates=251)
        with self.assertRaises(ValueError):
            experiment(None, updates=9)
        with self.assertRaises(ValueError):
            experiment(None, updates=20, seeds=())


if __name__ == "__main__":
    unittest.main()
