"""Formal controls for v5 counterfactual, position-unbiased character learning."""
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

from forge_core.neural.english_unbiased_v5 import (
    ALPHABET, ARMS, SEEDS, bucket, evaluate, experiment, make_encoded,
    make_tasks, permute_targets, prompt_for, source_strings,
    train_batches, validate,
)
from forge_core.neural.english_production_v2 import train_loss
from forge_core.neural.model import ByteTransformer, ModelConfig


class AsciiCodec:
    def encode(self, text):
        return SimpleNamespace(ids=list(text.encode("ascii")))

    def decode(self, tokens):
        return bytes(tokens).decode("ascii", errors="replace")


class AllNewlines:
    cfg = SimpleNamespace(max_seq_len=256)

    def eval(self):
        return self

    def __call__(self, x):
        logits = torch.zeros((*x.shape, 128))
        logits[:,-1,10] = 100.
        return logits,None


class ExactIndexedCharacterOracle:
    """Test fixture: independently reads the prompt; NOT a learned baseline."""
    cfg = SimpleNamespace(max_seq_len=256)

    def eval(self):
        return self

    def __call__(self, x):
        text = bytes(x[0].tolist()).decode("ascii")
        src = text.split("Letters: ",1)[1].split("\n",1)[0].replace(" ","")
        pos = int(text.split("Position: ",1)[1].split("\n",1)[0])
        suffix = text.rsplit("\nAnswer:",1)[1]
        if suffix:
            letter = "\n"
        else:
            letter = src[pos-1]
        logits = torch.zeros((*x.shape, 128))
        logits[:,-1,ord(letter)] = 100.
        return logits,None


class TestV5CounterfactualData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=make_tasks("train",families=90,seed=SEEDS[0])
        cls.dev=make_tasks("development",families=12,seed=7811)
        cls.exam=make_tasks("exam",families=16,seed=10413)

    def test_data_is_independently_uniform_not_vowel_position_synthetic(self):
        # In v4 every character at index 2 was an easy vowel. This
        # generation must include consonants AND vowels in that slot.
        base_rows=[r for r in self.train if r.variant==0]
        index2=[r.answer for r in base_rows if r.index==2]
        self.assertGreater(sum(c not in "aeiou" for c in index2), 65)
        self.assertGreater(sum(c in "aeiou" for c in index2), 5)
        self.assertEqual(set(ARMS),
                         {"paired_verified","unpaired_verified",
                          "permuted_labels","source_masked"})
        self.assertEqual(len(ALPHABET),26)

    def test_two_source_variants_differ_at_exactly_queried_location(self):
        for split,rows,count in (
            ("train",self.train,90),
            ("development",self.dev,12),
            ("exam",self.exam,16),
        ):
            validate(rows,split=split)
            self.assertEqual(len(set(r.family for r in rows)),count)
            for family in set(r.family for r in rows):
                matching=[r for r in rows if r.family==family]
                a=next(r.text for r in matching if r.variant==0)
                b=next(r.text for r in matching if r.variant==1)
                changed=next(r.changed_position for r in matching)
                self.assertEqual(sum(x!=y for x,y in zip(a,b)),1)
                self.assertNotEqual(a[changed-1],b[changed-1])
                for variant in (0,1):
                    present=[r for r in matching if r.variant==variant]
                    self.assertEqual({r.index for r in present},
                                     set(range(1,len(a)+1)))
                    for row in present:
                        self.assertEqual(row.answer,row.text[row.index-1])
                        self.assertEqual(row.prompt,prompt_for(
                            split,row.text,row.index))
        self.assertFalse(source_strings(self.train)&source_strings(self.dev))
        self.assertFalse(source_strings(self.train)&source_strings(self.exam))
        self.assertFalse(source_strings(self.dev)&source_strings(self.exam))
        self.assertEqual(self.exam,make_tasks("exam",families=16,seed=10413))

    def test_corrupted_answers_fail_closed(self):
        altered=replace(self.train[0],answer="!")
        with self.assertRaisesRegex(ValueError,"Wrong character"):
            validate([altered]+self.train[1:])
        altered=replace(self.train[0],changed_position=3)
        with self.assertRaises(ValueError):
            validate([altered]+self.train[1:])
        altered=replace(self.train[0],prompt="Answer: q")
        with self.assertRaisesRegex(ValueError,"Prompt alteration"):
            validate([altered]+self.train[1:])
        altered=replace(self.train[0],split="exam")
        with self.assertRaises(ValueError):
            validate([altered]+self.train[1:])
        with self.assertRaises(ValueError):
            bucket("B8AD")

    def test_source_blind_control_has_no_letters_and_keeps_positions(self):
        row=self.exam[0]
        masked=prompt_for(row.split,row.text,row.index,hide_source=True)
        self.assertNotEqual(masked,row.prompt)
        self.assertIn("Letters: "+" ".join("?" for _ in row.text),masked)
        self.assertIn(f"Position: {row.index}",masked)
        self.assertNotIn("Letters: "+" ".join(row.text),masked)

    def test_label_shuffle_preserves_position_marginals_and_prompts(self):
        wrong=permute_targets(self.train,seed=4381)
        self.assertEqual([x.prompt for x in wrong],
                         [x.prompt for x in self.train])
        self.assertGreater(sum(a.answer!=b.answer
                               for a,b in zip(wrong,self.train)),
                           int(.70*len(wrong)))
        for pos in range(1,9):
            before=Counter(r.answer for r in self.train if r.index==pos)
            after=Counter(r.answer for r in wrong if r.index==pos)
            self.assertEqual(before,after)
        with self.assertRaisesRegex(ValueError,"Wrong character"):
            validate(wrong)

    def test_paired_vs_unpaired_has_exact_sample_multiset(self):
        paired,unpaired=train_batches(self.train,seed=139,updates=30)
        self.assertEqual(len(paired),len(unpaired))
        self.assertCountEqual([i for b in paired for i in b],
                              [i for b in unpaired for i in b])
        self.assertNotEqual(paired,unpaired)
        self.assertTrue(all(len(b)==4 for b in paired))
        for batch in paired:
            for start in (0,2):
                a,b=[self.train[i] for i in batch[start:start+2]]
                self.assertEqual(a.family,b.family)
                self.assertEqual(a.index,b.index)
                self.assertEqual((a.variant,b.variant),(0,1))
        with self.assertRaises(ValueError):
            train_batches(self.train,seed=12,updates=5000)

    def test_budget_cap_precedes_any_checkpoint_or_dataset_read(self):
        with self.assertRaises(ValueError):
            experiment(source=None,out=None,updates=9000)
        with self.assertRaises(ValueError):
            experiment(source=None,out=None,seeds=())
        with self.assertRaises(ValueError):
            make_tasks("unknown",families=18,seed=1)


class TestV5NeuralInterfaces(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(43)
        self.codec=AsciiCodec()
        self.model=ByteTransformer(ModelConfig(
            vocab_size=128,max_seq_len=256,n_layer=1,n_head=2,n_embd=16))

    def test_only_generated_letter_answer_tokens_are_supervised(self):
        rows=make_tasks("train",families=6,seed=445)
        items=make_encoded(rows,self.codec,256)
        blind=make_encoded(rows,self.codec,256,hide_source=True)
        self.assertEqual(len(items),len(rows))
        for (tokens,targets),(hidden,masked_targets),row in zip(
                items,blind,rows):
            count=len(self.codec.encode(row.prompt).ids)
            self.assertEqual(targets[:count-1],[-100]*(count-1))
            self.assertEqual(len(tokens),len(targets))
            self.assertEqual(
                self.codec.decode(hidden).split("\nLetters: ",1)[1]
                    .split("\n",1)[0].replace(" ",""),
                "?"*len(row.text))
            self.assertEqual(masked_targets[-2:],targets[-2:])
        loss=train_loss(self.model,items[:4])
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in self.model.parameters()))

    def test_exact_external_reference_yields_perfect_counterfactual_pairs(self):
        rows=make_tasks("exam",families=6,seed=9)
        result=evaluate(ExactIndexedCharacterOracle(),rows,self.codec)
        self.assertEqual(result["letters_correct"],result["letter_total"])
        self.assertEqual(result["paired_both_correct"],6)
        self.assertEqual(result["paired_predictions_differ"],6)
        self.assertEqual(result["externally_assembled_exact"],12)
        self.assertEqual(result["externally_assembled_total"],12)
        self.assertEqual(result["terminated"],result["letter_total"])
        self.assertGreater(result["nonfirst_total"],0)

    def test_refusal_or_empty_response_is_not_a_letter(self):
        rows=make_tasks("exam",families=6,seed=9)
        result=evaluate(AllNewlines(),rows,self.codec)
        self.assertEqual(result["letters_correct"],0)
        self.assertEqual(result["paired_both_correct"],0)
        self.assertEqual(result["externally_assembled_exact"],0)
        self.assertEqual(result["terminated"],result["letter_total"])
        self.assertTrue(all(v["correct"]==0 for v in
                            result["by_position"].values()))
        with self.assertRaises(ValueError):
            evaluate(AllNewlines(),rows,self.codec,max_tokens=100)


if __name__=="__main__":
    unittest.main()
