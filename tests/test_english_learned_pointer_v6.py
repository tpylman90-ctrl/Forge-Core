"""Regression tests for original trainable neural copier and honest controls."""
from collections import Counter
from types import SimpleNamespace
import unittest

import torch

from forge_core.neural.english_learned_pointer_v6 import (
    ALPHABET, ARMS, BOS, EOS, MASK, PAD, MAX_OUTPUT, NeuralCopyDecoder,
    Pair, assert_firewall, batch_tensors, corrupted_map, edit_distance,
    evaluate, experiment, generate_pairs, greedy_generate, ids,
    objective, sequence_bucket, training_batches, validate_pairs, word_pairs,
)


class ExactPythonOracle:
    """Deterministic reference system, never a learned model or a baseline arm."""
    def eval(self):
        return self

    def __call__(self, src, prev):
        # This exact index is deliberately hardcoded only in the test oracle.
        # The actual NeuralCopyDecoder NEVER indexes src at the target slot.
        position = prev.shape[1] - 1
        b = src.shape[0]
        logit = torch.full((b,prev.shape[1],EOS+1),-50.)
        for i in range(b):
            next_id = int(src[i,position]) if position < src.shape[1] else EOS
            if next_id == PAD:
                next_id = EOS
            logit[i,-1,next_id] = 100.
        return logit,None


class DataIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = generate_pairs("train",120,13631)
        cls.dev = generate_pairs("dev",24,10219)
        cls.exam = generate_pairs("exam",28,12347)
        cls.long = generate_pairs("long",24,17193)
        cls.real = word_pairs()

    def test_all_splits_and_twins_hold_exact_independent_truth(self):
        assert_firewall(self.train,self.dev,self.exam,self.long,self.real)
        for pairs,split in ((self.train,"train"),(self.dev,"dev"),
                            (self.exam,"exam"),(self.long,"long"),
                            (self.real,"words")):
            validate_pairs(pairs)
            self.assertTrue(all(x.split == split for x in pairs))
            for p in pairs:
                self.assertEqual(len(p.original),len(p.twin))
                self.assertEqual(sum(a != b for a,b in
                                     zip(p.original,p.twin)),1)
                self.assertNotEqual(p.original[p.changed_index],
                                    p.twin[p.changed_index])
                if split != "words":
                    self.assertEqual(sequence_bucket(p.original),split)
                if split == "long":
                    self.assertTrue(9 <= len(p.original) <= 12)
                elif split != "words":
                    self.assertTrue(3 <= len(p.original) <= 8)
        self.assertEqual(self.exam,generate_pairs("exam",28,12347))
        self.assertEqual(len(self.real),36)
        self.assertEqual(len(set(p.original for p in self.real)),36)

    def test_uniform_random_characters_not_old_alternating_vowels(self):
        # The previous v4 result suffered a systematic vowel at index 2.
        second = [p.original[1] for p in self.train]
        self.assertGreater(sum(c in "aeiou" for c in second),10)
        self.assertGreater(sum(c not in "aeiou" for c in second),80)

    def test_source_tampering_and_duplicate_pairs_rejected(self):
        with self.assertRaises(ValueError):
            validate_pairs([Pair("cat","dog",1,"train")])
        with self.assertRaises(ValueError):
            validate_pairs([Pair("cat","cxt",1,"exam")])
        with self.assertRaisesRegex(ValueError,"Repeated"):
            validate_pairs([self.train[0],self.train[0]])
        with self.assertRaisesRegex(ValueError,"collision"):
            assert_firewall(self.train,self.train)
        with self.assertRaises(ValueError):
            ids("BAD!")
        with self.assertRaises(ValueError):
            sequence_bucket("a"*25)

    def test_negative_target_derangement_matches_length_without_self_labels(self):
        corrupted=corrupted_map(self.train,seed=1123)
        self.assertEqual(len(corrupted),2*len(self.train))
        self.assertTrue(all(w != target for w,target in corrupted.items()))
        self.assertTrue(all(len(w) == len(target)
                            for w,target in corrupted.items()))
        original_by_length={}
        corrupted_by_length={}
        for w,t in corrupted.items():
            original_by_length.setdefault(len(w),[]).append(w)
            corrupted_by_length.setdefault(len(w),[]).append(t)
        for length in original_by_length:
            self.assertCountEqual(original_by_length[length],
                                  corrupted_by_length[length])
        self.assertEqual(corrupted,corrupted_map(self.train,seed=1123))

    def test_same_training_pairs_used_by_all_arms(self):
        batches=training_batches(self.train,updates=20,seed=19)
        self.assertEqual(len(batches),20)
        self.assertTrue(all(len(row)==8 for row in batches))
        sources=set(w for p in self.train for w in (p.original,p.twin))
        self.assertTrue(all(w in sources for batch in batches for w in batch))
        self.assertEqual(len(ARMS),5)
        with self.assertRaises(ValueError):
            training_batches(self.train,updates=999999,seed=2)


class NeuralCopyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(45)
        torch.set_num_threads(2)
        self.net=NeuralCopyDecoder(width=32)

    def test_model_is_original_small_trainable_attention_not_direct_copy(self):
        self.assertTrue(sum(p.numel() for p in self.net.parameters()) > 1000)
        self.assertLess(sum(p.numel() for p in self.net.parameters()),250000)
        source,prev,target,alignment=batch_tensors(["abc","fox"])
        logits,attn=self.net(source,prev)
        self.assertEqual(logits.shape,(2,4,27))
        self.assertEqual(attn.shape,(2,4,4))
        self.assertTrue(torch.allclose(attn.sum(-1),
                                       torch.ones(2,4),atol=1e-6))
        self.assertEqual(target[0,:4].tolist(),
                         [ALPHABET.index("a"),ALPHABET.index("b"),
                          ALPHABET.index("c"),EOS])
        self.assertEqual(alignment[0,:4].tolist(),[0,1,2,3])
        self.assertEqual(prev[0,0].item(),BOS)
        self.assertEqual(prev[0,1].item(),ALPHABET.index("a"))
        self.assertNotEqual(prev[0,0].item(),ALPHABET.index("a"))

    def test_variable_width_source_padding_and_learned_eos_marker(self):
        source,prev,target,alignment=batch_tensors(["cat","sunshine"])
        self.assertEqual(source[0,3].item(),EOS)
        self.assertEqual(source[0,4].item(),PAD)
        self.assertEqual(source[1,8].item(),EOS)
        self.assertEqual(target[0,3].item(),EOS)
        self.assertEqual(target[0,4].item(),-100)
        _,attn=self.net(source,prev)
        self.assertLess(attn[0,:,4:].max().item(),1e-6)

    def test_source_hidden_control_cannot_read_the_answer(self):
        source,prev,target,align=batch_tensors(
            ["abca","zebr"],mask_source=True)
        self.assertTrue((source[:,:4] == MASK).all())
        self.assertTrue((source[:,4] == EOS).all())
        original_source,*_=batch_tensors(["abca","zebr"])
        self.assertFalse(torch.equal(source,original_source))
        self.assertEqual(target[0,0].item(),ALPHABET.index("a"))
        self.assertEqual(target[1,0].item(),ALPHABET.index("z"))

    def test_teacher_loss_has_gradients_and_alignment_is_only_auxiliary(self):
        loss=objective(self.net,["abca","zebr"],alignment_weight=.35)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(any(p.grad is not None and
                            p.grad.abs().sum().item()>0
                            for p in self.net.parameters()))
        with self.assertRaises(ValueError):
            objective(self.net,["abca"],["zzzz"],alignment_weight=.35)
        with self.assertRaises(ValueError):
            objective(NeuralCopyDecoder(width=32,pointer=False),
                      ["abca"],alignment_weight=.35)

    def test_true_greedy_generation_oracle_perfect_and_model_unassisted(self):
        words=["abc","gold","mystery","sunshine"]
        oracle=ExactPythonOracle()
        self.assertEqual(greedy_generate(oracle,words),words)
        # This next output must be a real model prediction, not the
        # Python oracle or an externally indexed assembly.
        result=greedy_generate(self.net,words)
        self.assertEqual(len(result),len(words))
        self.assertTrue(all(set(w)<=set(ALPHABET) for w in result))
        self.assertTrue(all(len(w)<=MAX_OUTPUT for w in result))

    def test_exact_changed_character_twins_are_independently_checked(self):
        pairs=[Pair("dove","dole",2,"words"),
               Pair("mango","mango",1,"words")]
        with self.assertRaises(ValueError):
            validate_pairs(pairs)
        valid=[pairs[0],Pair("mango","mango",1,"words")]
        with self.assertRaises(ValueError):
            evaluate(ExactPythonOracle(),valid)
        valid=[pairs[0],Pair("mango","mango",1,"words")]
        good=Pair("mango","mengo",1,"words")
        result=evaluate(ExactPythonOracle(),[pairs[0],good])
        self.assertEqual(result["exact"],4)
        self.assertEqual(result["pair_exact"],2)
        self.assertEqual(result["changed_letter_both_correct"],2)
        self.assertEqual(result["changed_response"],2)
        self.assertEqual(result["character_error_rate"],0.)

    def test_edit_distance_and_bounded_input(self):
        self.assertEqual(edit_distance("cat","cut"),1)
        self.assertEqual(edit_distance("cat","cats"),1)
        self.assertEqual(edit_distance("","cat"),3)
        with self.assertRaises(ValueError):
            batch_tensors(["abc"],["cat","dog"])
        with self.assertRaises(ValueError):
            NeuralCopyDecoder(width=17)
        with self.assertRaises(ValueError):
            experiment(source=None,out=None,updates=1900)
        with self.assertRaises(ValueError):
            experiment(source=None,out=None,seeds=())


if __name__=="__main__":
    unittest.main()
