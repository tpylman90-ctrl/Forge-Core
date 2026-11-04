"""Challenge integrity, independence and research-only lineage tests."""
from copy import deepcopy
from collections import defaultdict
from types import SimpleNamespace
import unittest

import torch

from forge_core.neural.english_v5_word_challenge import (
    WORDS, CHALLENGE_SCHEMA, changed_twin, challenge_rows,
    check_unseen_challenge, lineage,
)
from forge_core.neural.english_unbiased_v5 import evaluate


class AsciiCodec:
    def encode(self,text):
        return SimpleNamespace(ids=list(text.encode("ascii")))
    def decode(self,tokens):
        return bytes(tokens).decode("ascii",errors="replace")


class FormulaicExactOracle:
    """An explicit deterministic source checker, never a trained model."""
    cfg=SimpleNamespace(max_seq_len=256)
    def eval(self):
        return self
    def __call__(self,x):
        s=bytes(x[0].tolist()).decode("ascii")
        chars=s.split("Letters: ",1)[1].split("\n",1)[0].replace(" ","")
        pos=int(s.split("Position: ",1)[1].split("\n",1)[0])
        next_char=("\n" if s.split("\nAnswer:",1)[1]
                   else chars[pos-1])
        logits=torch.zeros((*x.shape,128))
        logits[:,-1,ord(next_char)]=100.
        return logits,None


class ChallengeSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=challenge_rows()

    def test_curated_words_and_one_letter_edits_are_fixed(self):
        self.assertEqual(CHALLENGE_SCHEMA,1)
        self.assertEqual(len(WORDS),36)
        self.assertEqual(len(set(WORDS)),36)
        pairs=defaultdict(lambda:defaultdict(dict))
        for word in WORDS:
            self.assertTrue(4<=len(word)<=8)
            altered,index=changed_twin(word)
            self.assertEqual(len(altered),len(word))
            self.assertNotEqual(altered[index-1],word[index-1])
            self.assertEqual(sum(a!=b for a,b in zip(word,altered)),1)
        for task in self.rows:
            pairs[task.family][task.variant][task.index]=task
            self.assertEqual(task.answer,task.text[task.index-1])
        self.assertEqual(set(pairs),set(WORDS))
        for original,twin in pairs.items():
            self.assertEqual(set(twin),{0,1})
            length=len(original)
            self.assertEqual(set(twin[0]),set(range(1,length+1)))
            self.assertEqual(set(twin[1]),set(range(1,length+1)))
            for variant in (0,1):
                self.assertEqual(len(twin[variant]),length)

    def test_original_v5_random_train_dev_exam_cannot_include_word_sources(self):
        # Checked using the precise original v5 generators/seeds.
        check_unseen_challenge(self.rows)

    def test_exact_external_oracle_yields_perfect_assembly(self):
        scored=evaluate(FormulaicExactOracle(),self.rows,AsciiCodec())
        self.assertEqual(scored["letters_correct"],len(self.rows))
        self.assertEqual(scored["paired_both_correct"],len(WORDS))
        self.assertEqual(scored["paired_predictions_differ"],len(WORDS))
        self.assertEqual(scored["externally_assembled_exact"],2*len(WORDS))

    def test_wrong_checkpoint_source_or_neural_arm_fails_closed(self):
        record={
            "schema":1,"research_only":True,
            "arm":"paired_verified","seed":11631,
            "original_checkpoint_sha256":"a"*64,
            "source_tokenizer_sha256":"b"*64,
            "configuration":{"vocab_size":256},"best_weights":{},
        }
        args={
            "arm":"paired_verified","seed":11631,
            "original_hash":"a"*64,"tokenizer_hash":"b"*64,
            "config":{"vocab_size":256},
        }
        lineage(record,**args)
        for key,invalid in [
            ("research_only",False),
            ("original_checkpoint_sha256","c"*64),
            ("source_tokenizer_sha256","d"*64),
            ("arm","unpaired_verified"),
            ("seed",11632),
            ("configuration",{"vocab_size":512}),
        ]:
            with self.subTest(field=key):
                edited=deepcopy(record)
                edited[key]=invalid
                with self.assertRaisesRegex(ValueError,"wrong original lineage"):
                    lineage(edited,**args)


if __name__=="__main__":
    unittest.main()
