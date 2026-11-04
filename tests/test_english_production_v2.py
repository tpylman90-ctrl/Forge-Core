"""Independent English production evaluation: grammar, firewalls, null and gradients."""
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

from forge_core.neural.english_production_v2 import (
    ARMS, LEX, SEEDS, SKILLS, check_firewall, encode_samples, evaluate,
    make_lessons, normalize, past, plural, present, prompt_for, schedules,
    shuffle_teaching, study, train_loss, truth, validate,
)
from forge_core.neural.model import ByteTransformer, ModelConfig


class PlainCodec:
    def encode(self, text):
        return SimpleNamespace(ids=list(text.encode("ascii")))

    def decode(self, tokens):
        return bytes(tokens).decode("ascii", errors="replace")


class RepeatedNewlineModel:
    cfg = SimpleNamespace(max_seq_len=256, vocab_size=128)

    def eval(self):
        return self

    def __call__(self, x):
        y = torch.full((*x.shape, 128), -10.)
        y[:,-1,10] = 100.0
        return y, None


class TestEnglishProductionGrammar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=make_lessons("train",per_skill=64,seed=SEEDS[0])
        cls.development=make_lessons("development",per_skill=8,seed=33441)
        cls.exam=make_lessons("exam",per_skill=16,seed=99001)

    def test_regular_english_rules_cover_suffix_variations(self):
        examples=[
            (plural("daisy"),"daisies"),
            (plural("fox"),"foxes"),
            (plural("button"),"buttons"),
            (present("carry"),"carries"),
            (present("wash"),"washes"),
            (present("enjoy"),"enjoys"),
            (present("bake"),"bakes"),
            (past("try"),"tried"),
            (past("close"),"closed"),
            (past("paint"),"painted"),
            (truth("join_parts","parts=moon+light"),"moonlight"),
            (truth("write_sentence","sentence=Amina|carry|kite"),
             "Amina carries the kite."),
            (truth("negate_sentence","sentence=Amina|carry|kite"),
             "Amina does not carry the kite."),
            (truth("actor_role",
                   "event=Amina|give|kite|Benton;ask=receiver"),"Benton"),
            (truth("actor_role",
                   "event=Benton|give|kite|Amina;ask=giver"),"Benton"),
        ]
        for actual,expected in examples:
            self.assertEqual(actual,expected)

    def test_teacher_gold_derives_only_from_source(self):
        check_firewall()
        self.assertEqual(len(SKILLS),7)
        self.assertEqual(len(self.train),448)
        self.assertEqual(len(self.development),56)
        self.assertEqual(len(self.exam),112)
        for split,items in (("train",self.train),
                            ("development",self.development),
                            ("exam",self.exam)):
            validate(items,split)
            self.assertEqual(set(x.skill for x in items),set(SKILLS))
            for item in items:
                self.assertEqual(item.answer,truth(item.skill,item.source))
                self.assertEqual(item.prompt,
                                 prompt_for(item.split,item.skill,item.source))
        self.assertEqual(len({x.prompt for x in
                             self.train+self.development+self.exam}),
                         448+56+112)
        for field in ("people","nouns","verbs","prefix","suffix"):
            self.assertFalse(set(LEX["train"][field]) &
                             set(LEX["development"][field]))
            self.assertFalse(set(LEX["train"][field]) &
                             set(LEX["exam"][field]))
            self.assertFalse(set(LEX["development"][field]) &
                             set(LEX["exam"][field]))

    def test_tampering_cannot_change_solved_answer_or_source(self):
        bad=replace(self.train[0],answer="this is false")
        with self.assertRaisesRegex(ValueError,"Wrong supplied"):
            validate([bad]+self.train[1:],"train")
        bad=replace(self.train[0],source="parts=<BAD>")
        with self.assertRaises(ValueError):
            validate([bad]+self.train[1:])
        bad=replace(self.train[0],prompt=self.train[0].prompt+" FLAG")
        with self.assertRaisesRegex(ValueError,"wording"):
            validate([bad]+self.train[1:])
        with self.assertRaisesRegex(ValueError,"Ambiguous"):
            truth("actor_role","event=Amina|give|kite|Amina;ask=giver")
        with self.assertRaises(ValueError):
            truth("unknown","x")

    def test_counterfactual_roles_track_which_person_did_what(self):
        a=truth("actor_role","event=Amina|give|kite|Benton;ask=giver")
        b=truth("actor_role","event=Benton|give|kite|Amina;ask=giver")
        c=truth("actor_role","event=Amina|give|kite|Benton;ask=receiver")
        self.assertEqual((a,b,c),("Amina","Benton","Benton"))

    def test_shuffled_control_keeps_all_prompts_and_answer_marginals(self):
        wrong=shuffle_teaching(self.train,seed=9383)
        self.assertEqual([x.prompt for x in wrong],
                         [x.prompt for x in self.train])
        self.assertGreaterEqual(sum(a.answer!=b.answer for a,b in
                                    zip(wrong,self.train)),len(self.train)//2)
        for skill in SKILLS:
            original=Counter(x.answer for x in self.train if x.skill==skill)
            shuffled=Counter(x.answer for x in wrong if x.skill==skill)
            self.assertEqual(original,shuffled)
        self.assertEqual(wrong,shuffle_teaching(self.train,seed=9383))

    def test_equal_exposure_ordered_and_mixed_schedules(self):
        staged,mixed=schedules(self.train,seed=8123,updates_per_level=8)
        self.assertEqual(len(staged),56)
        self.assertEqual(len(mixed),56)
        self.assertCountEqual([tuple(b) for b in staged],
                              [tuple(b) for b in mixed])
        self.assertNotEqual(staged,mixed)
        for i,skill in enumerate(SKILLS):
            for batch in staged[i*8:(i+1)*8]:
                self.assertGreaterEqual(
                    sum(self.train[k].skill==skill for k in batch),3)
        with self.assertRaises(ValueError):
            schedules(self.train,seed=1,updates_per_level=500)

    def test_input_rejects_oversized_or_corrupt_tasks(self):
        with self.assertRaises(ValueError):
            make_lessons("train",per_skill=200,seed=1)
        with self.assertRaises(ValueError):
            make_lessons("wrong",per_skill=12,seed=1)
        with self.assertRaises(ValueError):
            encode_samples(self.train,PlainCodec(),32)
        with self.assertRaises(ValueError):
            study(folder=None,out=None,updates_per_level=300)
        self.assertEqual(normalize(" Apple. \n"),"apple")


class TestNeuralGeneration(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(73)
        self.model=ByteTransformer(
            ModelConfig(vocab_size=128,max_seq_len=256,
                        n_layer=1,n_head=2,n_embd=16))
        self.codec=PlainCodec()

    def test_teacher_forcing_targets_only_answer_not_the_question(self):
        rows=make_lessons("train",per_skill=4,seed=32)
        pairs=encode_samples(rows,self.codec,256)
        self.assertEqual(len(pairs),28)
        self.assertTrue(any(y==-100 for y in pairs[0][1]))
        self.assertTrue(any(y!=-100 for y in pairs[0][1]))
        for (tokens,targets),row in zip(pairs,rows):
            start=len(self.codec.encode(row.prompt).ids)-1
            self.assertEqual(targets[:start],[-100]*start)
            self.assertEqual(len(tokens),len(targets))
        loss=train_loss(self.model,pairs[:4])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in self.model.parameters()))

    def test_scoring_uses_generated_text_not_options(self):
        rows=make_lessons("exam",per_skill=4,seed=812)
        scored=evaluate(RepeatedNewlineModel(),rows,self.codec,max_new_tokens=4)
        self.assertEqual(scored["correct"],0)
        self.assertEqual(scored["total"],28)
        self.assertEqual(set(scored["by_skill"]),set(SKILLS))
        self.assertIn("not A/B",scored["scoring"])
        for data in scored["by_skill"].values():
            self.assertEqual(data["total"],4)
            self.assertEqual(data["nonempty"],0)
        with self.assertRaises(ValueError):
            evaluate(RepeatedNewlineModel(),rows,self.codec,max_new_tokens=200)


if __name__=="__main__":
    unittest.main()
