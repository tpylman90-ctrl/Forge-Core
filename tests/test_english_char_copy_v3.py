"""Unit tests for source-sealed character micro-literacy and model supervision."""
from dataclasses import replace
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch

from forge_core.neural.english_char_copy_v3 import (
    ARMS, SKILLS, SEEDS, evaluate, expected, levenshtein, make_lessons,
    normalized, random_letters, schedules, shuffle_answers, split_bucket,
    study, validate,
)
from forge_core.neural.english_production_v2 import encode_samples, train_loss
from forge_core.neural.model import ByteTransformer, ModelConfig


class PlainCodec:
    def encode(self, text):
        return SimpleNamespace(ids=list(text.encode("ascii")))
    def decode(self, tokens):
        return bytes(tokens).decode("ascii",errors="replace")


class NewlineOnly:
    cfg = SimpleNamespace(max_seq_len=256,vocab_size=128)
    def eval(self):
        return self
    def __call__(self,x):
        logits=torch.zeros((*x.shape,128))
        logits[:,-1,10]=10.
        return logits,None


class TestCharTruth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=make_lessons("train",per_skill=100,seed=SEEDS[0])
        cls.dev=make_lessons("development",per_skill=10,seed=77333)
        cls.exam=make_lessons("exam",per_skill=24,seed=99173)

    def test_independent_truth_for_all_four_copy_operations(self):
        self.assertEqual(expected("copy_letter","letter=k;context=cat"),"k")
        self.assertEqual(expected("copy_spaced","letters=c a t"),"c a t")
        self.assertEqual(expected("join_letters","letters=c a t"),"cat")
        self.assertEqual(expected("append_letter","letters=c a t;append=s"),"cats")
        for invalid in ("letters=abc","letters=c  a t","letters=c a"):
            with self.assertRaises(ValueError):
                expected("join_letters",invalid)

    def test_all_four_skills_and_hash_splits_reproducible(self):
        self.assertEqual(len(self.train),400)
        self.assertEqual(len(self.dev),40)
        self.assertEqual(len(self.exam),96)
        self.assertEqual(set(r.skill for r in self.train),set(SKILLS))
        self.assertEqual(self.exam,make_lessons("exam",per_skill=24,seed=99173))
        for split,rows in (("train",self.train),
                           ("development",self.dev),
                           ("exam",self.exam)):
            validate(rows,split)
        self.assertEqual(len(set(x.prompt for x in
                             self.train+self.dev+self.exam)),536)
        # A whole target string (other than single letter targets) is
        # assigned to exactly one train/development/exam partition.
        keys={split:{
            r.answer.replace(" ","") for r in rows
            if r.skill!="copy_letter"}
              for split,rows in (("train",self.train),
                                 ("development",self.dev),
                                 ("exam",self.exam))}
        self.assertFalse(keys["train"]&keys["development"])
        self.assertFalse(keys["train"]&keys["exam"])
        self.assertFalse(keys["development"]&keys["exam"])
        with self.assertRaises(ValueError):
            split_bucket("Hi!")

    def test_label_and_question_tampering_fails_closed(self):
        altered=replace(self.train[0],answer="WRONG")
        with self.assertRaisesRegex(ValueError,"answer"):
            validate([altered]+self.train[1:])
        altered=replace(self.train[0],prompt=self.train[0].prompt+" bad")
        with self.assertRaisesRegex(ValueError,"question"):
            validate([altered]+self.train[1:])
        altered=replace(self.train[0],split="exam")
        with self.assertRaises(ValueError):
            validate([altered]+self.train[1:])

    def test_staged_vs_mixed_equal_training_samples(self):
        staged,mixed=schedules(self.train,SEEDS[0],updates_per_level=8)
        self.assertEqual(len(staged),len(SKILLS)*8)
        self.assertCountEqual([tuple(x) for x in staged],
                              [tuple(x) for x in mixed])
        self.assertNotEqual(staged,mixed)
        for level,skill in enumerate(SKILLS):
            for row in staged[level*8:(level+1)*8]:
                self.assertEqual(sum(self.train[j].skill==skill
                                     for j in row[:3]),3)
        with self.assertRaises(ValueError):
            schedules(self.train,42,updates_per_level=400)

    def test_shuffled_null_has_identical_inputs_and_answer_marginals(self):
        shuffled=shuffle_answers(self.train,seed=2101)
        self.assertEqual([x.prompt for x in shuffled],
                         [x.prompt for x in self.train])
        self.assertGreaterEqual(sum(a.answer!=b.answer for a,b in
                                    zip(shuffled,self.train)),200)
        for skill in SKILLS:
            self.assertEqual(
                Counter(r.answer for r in self.train if r.skill==skill),
                Counter(r.answer for r in shuffled if r.skill==skill))

    def test_character_errors_and_normalization(self):
        self.assertEqual(levenshtein("cat","cats"),1)
        self.assertEqual(levenshtein("cats","cat"),1)
        self.assertEqual(levenshtein("abc","axc"),1)
        self.assertEqual(normalized(" C A T\\n"),"c a t\\n")
        self.assertEqual(normalized(" C A T\n"),"c a t")
        with self.assertRaises(ValueError):
            make_lessons("train",per_skill=300,seed=11)


class TestNeuralCharGeneration(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(22)
        self.model=ByteTransformer(ModelConfig(
            vocab_size=128,max_seq_len=256,n_layer=1,n_head=2,n_embd=16))
        self.codec=PlainCodec()

    def test_generative_teacher_forcing_trains_only_written_answer(self):
        rows=make_lessons("train",per_skill=4,seed=23)
        examples=encode_samples(rows,self.codec,256)
        self.assertEqual(len(examples),16)
        for (input_ids,labels),item in zip(examples,rows):
            start=len(self.codec.encode(item.prompt).ids)-1
            self.assertEqual(labels[:start],[-100]*start)
            self.assertEqual(len(labels),len(input_ids))
        loss=train_loss(self.model,examples[:4])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in self.model.parameters()))

    def test_exact_edit_first_char_and_termination_are_distinct(self):
        data=make_lessons("exam",per_skill=4,seed=87)
        report=evaluate(NewlineOnly(),data,self.codec,max_tokens=5)
        self.assertEqual(report["total"],16)
        self.assertEqual(report["exact"],0)
        self.assertEqual(set(report["by_skill"]),set(SKILLS))
        for skill in SKILLS:
            item=report["by_skill"][skill]
            self.assertEqual(item["total"],4)
            self.assertEqual(item["newline_terminated_rate"],1.)
            self.assertEqual(item["first_character_rate"],0.)
            self.assertGreater(item["character_error_rate"],0)

    def test_strict_training_limits(self):
        with self.assertRaises(ValueError):
            study(folder=None,out=None,updates_per_level=999)
        with self.assertRaises(ValueError):
            study(folder=None,out=None,updates_per_level=120,seeds=())


if __name__=="__main__":
    unittest.main()
