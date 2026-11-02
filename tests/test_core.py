import unittest

import torch

from forge_core.agent.authorization import ActionRequest, Approval, require_approval
from forge_core.learning.curiosity import CuriosityEngine, CuriositySettings, ObservationSignal
from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.tokenizer import ByteTokenizer


class TestTokenizer(unittest.TestCase):
    def test_utf8_roundtrip(self):
        tok = ByteTokenizer()
        original = "Forge: why? ⚙"
        self.assertEqual(tok.decode(tok.encode(original)), original)

    def test_invalid_byte(self):
        with self.assertRaises(ValueError):
            ByteTokenizer().decode([256])


class TestModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.model = ByteTransformer(
            ModelConfig(n_layer=1, n_head=4, n_embd=32, max_seq_len=16)
        )
        self.model.eval()

    def test_forward_and_loss(self):
        tokens = torch.randint(0, 256, (2, 8))
        logits, loss = self.model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (2, 8, 256))
        self.assertTrue(torch.isfinite(loss))

    def test_causal_mask(self):
        a = torch.randint(0, 256, (1, 8))
        b = a.clone()
        b[:, 5:] = (b[:, 5:] + 1) % 256
        with torch.no_grad():
            first, _ = self.model(a)
            second, _ = self.model(b)
        self.assertTrue(torch.allclose(first[:, :5], second[:, :5], atol=1e-5))

    def test_generate(self):
        prefix = torch.tensor([[70, 111, 114, 103, 101]], dtype=torch.long)
        generated = self.model.generate(prefix, 3)
        self.assertEqual(tuple(generated.shape), (1, 8))

    def test_gradients(self):
        self.model.train()
        x = torch.randint(0, 256, (2, 8))
        _, loss = self.model(x, x)
        loss.backward()
        self.assertIsNotNone(self.model.token.weight.grad)


class TestSafety(unittest.TestCase):
    def test_actions_deny_by_default(self):
        request = ActionRequest("send", "example-message")
        with self.assertRaises(PermissionError):
            require_approval(request, None)
        with self.assertRaises(PermissionError):
            require_approval(request, Approval(ActionRequest("delete", "example-message"), True))
        with self.assertRaises(PermissionError):
            require_approval(request, Approval(request, False))
        require_approval(request, Approval(request, True))

    def test_curiosity_requires_confirmation(self):
        engine = CuriosityEngine()
        candidate = engine.propose("Changed a design", "Why?", "Prefer clarity")
        self.assertFalse(candidate.confirmed)
        self.assertFalse(engine.confirm(candidate, False).confirmed)
        self.assertTrue(engine.confirm(candidate, True).confirmed)
        self.assertFalse(candidate.confirmed)

    def test_high_curiosity_asks_often_with_safety_gates(self):
        engine = CuriosityEngine()
        self.assertEqual(engine.settings.mode, "high")
        self.assertTrue(engine.should_ask(ObservationSignal(True, novel=True)))
        self.assertTrue(engine.should_ask(ObservationSignal(True, motivation_unclear=True)))
        self.assertFalse(engine.should_ask(ObservationSignal(False, novel=True)))
        self.assertFalse(engine.should_ask(ObservationSignal(True, novel=True, sensitive_context=True)))
        self.assertFalse(engine.should_ask(ObservationSignal(True, novel=True, unsafe_to_interrupt=True)))
        self.assertFalse(engine.should_ask(ObservationSignal(True, novel=True, already_asked=True)))
        self.assertFalse(CuriosityEngine(CuriositySettings("paused")).should_ask(
            ObservationSignal(True, novel=True, motivation_unclear=True)))
        self.assertFalse(CuriosityEngine(CuriositySettings("quiet")).should_ask(
            ObservationSignal(True, novel=True, motivation_unclear=True)))
        balanced = CuriosityEngine(CuriositySettings("balanced"))
        self.assertFalse(balanced.should_ask(ObservationSignal(True, novel=True)))
        self.assertTrue(balanced.should_ask(ObservationSignal(True, novel=True, motivation_unclear=True)))
        with self.assertRaises(ValueError):
            CuriositySettings("unknown")


if __name__ == "__main__":
    unittest.main()
