"""Neural pointer research: source spans, frozen representation and honest bounds."""
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.span_pointer_lab import (
    WordPointer, features_for, frozen_hidden, experiment,
)


@dataclass
class Example:
    id: str
    skill: str
    prompt: str
    answer: str


class CharacterCodec:
    """Offset-faithful test codec only; real pilot uses original BPE."""
    def encode(self, text):
        return SimpleNamespace(
            ids=[ord(x) % 93 + 1 for x in text],
            offsets=[(i,i+1) for i in range(len(text))])


class TestSpanPointerV1(unittest.TestCase):
    def test_copies_unseen_name_from_source_word_spans(self):
        torch.manual_seed(11)
        model=ByteTransformer(ModelConfig(
            vocab_size=128,max_seq_len=256,n_layer=1,n_head=2,n_embd=16))
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        codec=CharacterCodec()
        text="Instruction: Ada handed the toy to Ben. Who received it?\nAnswer:"
        row=Example("test:001","recipient",text,"Ben")
        item=features_for(model,codec,row)
        self.assertEqual(item.candidates[item.gold_index],"Ben")
        self.assertFalse(item.words.requires_grad)
        self.assertEqual(item.query.shape[0],16)
        self.assertGreater(len(item.candidates),10)
        head=WordPointer(16)
        logits=head(item.query,item.words)
        self.assertEqual(logits.shape,(len(item.candidates),))
        loss=F.cross_entropy(logits[None,:],
                             torch.tensor([item.gold_index]))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in head.parameters()))
        self.assertFalse(any(p.grad is not None for p in model.parameters()))

    def test_reject_gold_missing_from_source(self):
        model=ByteTransformer(ModelConfig(
            vocab_size=128,max_seq_len=256,n_layer=1,n_head=2,n_embd=16))
        with self.assertRaisesRegex(ValueError,"cannot be copied"):
            features_for(model,CharacterCodec(),
                Example("test:002","recipient",
                        "Instruction: Ada gave a toy to Ben.\nAnswer:","Zelda"))

    def test_invalid_budgets_before_file_io(self):
        for kwargs in ({"steps":0},{"seed":0},{"steps":2001},{"lr":0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError,"Out-of-bounds"):
                    experiment(source=Path("missing"),parent=Path("missing"),
                               lessons=Path("missing"),out=Path("missing"),
                               **kwargs)


if __name__=="__main__":
    unittest.main()
