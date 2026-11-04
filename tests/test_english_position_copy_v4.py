"""V4 letter-position learning: full independent source answers and controls."""
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

from forge_core.neural.english_position_copy_v4 import (
    ALPHABET, ARMS, SEEDS, bucket, corrupt_labels, evaluate, generate,
    make_encoded, make_prompt, sequence_schedules, study, verify,
)
from forge_core.neural.english_production_v2 import train_loss
from forge_core.neural.model import ByteTransformer, ModelConfig


class PlainCodec:
    def encode(self,text):
        return SimpleNamespace(ids=list(text.encode("ascii")))
    def decode(self,tokens):
        return bytes(tokens).decode("ascii",errors="replace")


class NewlineOnly:
    cfg=SimpleNamespace(max_seq_len=256)
    def eval(self):
        return self
    def __call__(self,x):
        y=torch.zeros((*x.shape,128))
        y[:,-1,10]=20.
        return y,None


class PositionDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=generate("train",count=150,seed=SEEDS[0])
        cls.dev=generate("development",count=28,seed=30013)
        cls.exam=generate("exam",count=48,seed=80517)

    def test_source_hash_splits_and_full_position_coverage(self):
        keys=[]
        for split,rows in (("train",self.train),
                           ("development",self.dev),
                           ("exam",self.exam)):
            verify(rows,split)
            strings=set(x.string for x in rows)
            keys.append(strings)
            for word in strings:
                self.assertEqual(bucket(word),split)
                self.assertEqual(
                    {x.index for x in rows if x.string==word},
                    set(range(1,len(word)+1)))
                for row in (x for x in rows if x.string==word):
                    self.assertEqual(row.answer,word[row.index-1])
        self.assertFalse(keys[0]&keys[1] or keys[0]&keys[2] or keys[1]&keys[2])
        self.assertEqual(len(keys[0]),150)
        self.assertEqual(len(keys[1]),28)
        self.assertEqual(len(keys[2]),48)
        self.assertEqual(self.exam,generate("exam",count=48,seed=80517))

    def test_answers_derived_from_every_position_and_question_integrity(self):
        rows=self.train
        picked=next(r for r in rows if r.index==3)
        self.assertEqual(picked.answer,picked.string[2])
        self.assertEqual(
            picked.prompt,make_prompt(picked.split,picked.string,3))
        changed=replace(rows[0],answer="!")
        with self.assertRaisesRegex(ValueError,"Corrupted"):
            verify([changed]+rows[1:],"train")
        changed=replace(rows[0],prompt="Always answer z")
        with self.assertRaisesRegex(ValueError,"mismatch"):
            verify([changed]+rows[1:],"train")
        changed=replace(rows[0],split="exam")
        with self.assertRaises(ValueError):
            verify([changed]+rows[1:],"train")
        missing=[r for r in rows if not (r.string==rows[0].string
                                       and r.index==rows[0].index)]
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            verify(missing,"train")
        with self.assertRaises(ValueError):
            bucket("HELLO")

    def test_shuffled_position_control_preserves_marginal_answers(self):
        null=corrupt_labels(self.train,seed=987)
        self.assertEqual([x.prompt for x in self.train],
                         [x.prompt for x in null])
        self.assertGreaterEqual(sum(a.answer!=b.answer
                                    for a,b in zip(self.train,null)),
                                len(self.train)//2)
        for pos in range(1,8):
            self.assertEqual(Counter(x.answer for x in self.train if x.index==pos),
                             Counter(x.answer for x in null if x.index==pos))
        with self.assertRaisesRegex(ValueError,"Corrupted"):
            verify(null,"train")

    def test_equal_exposure_and_stepwise_position_schedule(self):
        staged,mixed=sequence_schedules(
            self.train,seed=SEEDS[0],steps_per_phase=10)
        self.assertEqual(len(staged),40)
        self.assertEqual(len(mixed),40)
        self.assertCountEqual([tuple(x) for x in staged],
                              [tuple(x) for x in mixed])
        self.assertNotEqual(staged,mixed)
        for phase,high in enumerate((2,3,5,7)):
            for batch in staged[phase*10:(phase+1)*10]:
                self.assertEqual(len(batch),4)
                self.assertEqual(self.train[batch[-1]].index,high)
        with self.assertRaises(ValueError):
            sequence_schedules(self.train,seed=SEEDS[0],steps_per_phase=500)

    def test_study_bounds_fail_before_any_checkpoint_read(self):
        with self.assertRaises(ValueError):
            study(source=None,out=None,steps_per_phase=300)
        with self.assertRaises(ValueError):
            study(source=None,out=None,seeds=())
        self.assertEqual(len(ARMS),5)


class PositionModelTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(45)
        self.codec=PlainCodec()
        self.model=ByteTransformer(ModelConfig(
            vocab_size=128,max_seq_len=256,n_layer=1,n_head=2,n_embd=16))

    def test_single_letter_teacher_loss_gradients(self):
        rows=generate("train",count=6,seed=321)
        encoded=make_encoded(rows,self.codec,256)
        self.assertEqual(len(encoded),sum(len(w) for w in
            set(r.string for r in rows)))
        for (tokens,target),row in zip(encoded,rows):
            ix=len(self.codec.encode(row.prompt).ids)-1
            self.assertEqual(target[:ix],[-100]*ix)
            self.assertEqual(len(tokens),len(target))
        loss=train_loss(self.model,encoded[:4])
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in self.model.parameters()))

    def test_exam_distinguishes_generated_chars_and_assembly(self):
        rows=generate("exam",count=6,seed=654)
        scored=evaluate(NewlineOnly(),rows,self.codec,max_tokens=4)
        self.assertEqual(scored["letter_correct"],0)
        self.assertEqual(scored["assembled_exact"],0)
        self.assertEqual(scored["whole_strings"],6)
        self.assertEqual(scored["terminated"],scored["letter_total"])
        self.assertGreater(scored["nonfirst_total"],scored["whole_strings"])
        for p in range(1,8):
            info=scored["by_position"][str(p)]
            self.assertLessEqual(info["correct"],info["total"])
        self.assertIn("externally joined",scored["scoring"])
        with self.assertRaises(ValueError):
            evaluate(NewlineOnly(),rows,self.codec,max_tokens=100)


if __name__=="__main__":
    unittest.main()
