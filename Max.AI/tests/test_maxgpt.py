"""Tests for the tokenizer, the data pipeline and the model.

Run from the Max.AI folder:  python -m unittest discover -s tests
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maxgpt.data import Dataset, encode_conversation, format_prompt, parse_conversations  # noqa: E402
from maxgpt.model import GPT, GPTConfig  # noqa: E402
from maxgpt.optim import AdamW  # noqa: E402
from maxgpt.tokenizer import Tokenizer  # noqa: E402

SAMPLE = """===
User: What is Python?
Max: Python is a programming language.
===
User: Show code
Max: ```python
print("hi")
```
User: Thanks
Max: You're welcome!
"""


class TokenizerTest(unittest.TestCase):
    def test_roundtrip_and_specials(self):
        tok = Tokenizer.train("hello world hello code héllo 🙂 " * 20, vocab_size=300)
        for text in ["hello world", "héllo 🙂 new words", "<|user|>hi<|end|>"]:
            self.assertEqual(tok.decode(tok.encode(text)), text)
        self.assertIn(tok.special["<|end|>"], tok.encode("<|end|>"))
        self.assertNotIn(tok.special["<|end|>"], tok.encode("<|end|>", allow_special=False))

    def test_save_load(self):
        tok = Tokenizer.train("abc abd abe " * 30, vocab_size=280)
        with tempfile.TemporaryDirectory() as d:
            tok.save(os.path.join(d, "t.json"))
            again = Tokenizer.load(os.path.join(d, "t.json"))
        self.assertEqual(again.encode("abc abd"), tok.encode("abc abd"))
        self.assertEqual(again.vocab_size, tok.vocab_size)


class DataTest(unittest.TestCase):
    def test_parse(self):
        convs = parse_conversations(SAMPLE)
        self.assertEqual(len(convs), 2)
        self.assertEqual(convs[1][1], ("assistant", '```python\nprint("hi")\n```'))
        self.assertEqual(len(convs[1]), 4)

    def test_mask_only_assistant(self):
        tok = Tokenizer.train(SAMPLE * 5, vocab_size=400)
        ids, mask = encode_conversation(tok, parse_conversations(SAMPLE)[0])
        learned = tok.decode([i for i, m in zip(ids, mask) if m])
        self.assertEqual(learned, "Python is a programming language.<|end|>")
        self.assertTrue(format_prompt([("user", "hi")]).endswith("<|assistant|>"))

    def test_batches(self):
        tok = Tokenizer.train(SAMPLE * 5, vocab_size=400)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(SAMPLE)
        try:
            ds = Dataset(tok, [f.name], n_ctx=64)
            x, y = ds.batch(4)
            self.assertEqual(x.shape, y.shape)
            self.assertTrue((y != -1).any())
        finally:
            os.unlink(f.name)


class GuardTest(unittest.TestCase):
    def setUp(self):
        from maxgpt.guard import KnowledgeGuard
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(SAMPLE + "===\nUser: How do I merge two dictionaries?\nMax: Use a | b.\n"
                             "===\nUser: What is JavaScript?\nMax: A language.\n")
        self.path = f.name
        self.guard = KnowledgeGuard([f.name], threshold=0.5)

    def tearDown(self):
        os.unlink(self.path)

    def test_known_and_unknown(self):
        self.assertTrue(self.guard.knows("what is python"))
        self.assertTrue(self.guard.knows("merge two dicts"))  # synonym + rewording
        self.assertTrue(self.guard.knows("thanks!"))           # small talk always passes
        self.assertFalse(self.guard.knows("What is Kubernetes?"))
        self.assertFalse(self.guard.knows("What is a closure in JavaScript?"))  # new topic word

    def test_follow_up_uses_previous_question(self):
        self.assertTrue(self.guard.knows("show me another example"))  # conversational follow-up
        self.assertFalse(self.guard.knows("what about kubernetes"))
        self.assertTrue(self.guard.knows("in python", previous="How do I merge two dictionaries?"))


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.cfg = GPTConfig(vocab_size=50, n_ctx=16, n_layer=2, n_head=2, n_embd=16)

    def test_shapes_and_save_load(self):
        model = GPT(self.cfg)
        x = np.random.default_rng(0).integers(0, 50, (2, 10))
        logits = model.forward(x)
        self.assertEqual(logits.shape, (2, 10, 50))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            model.save(path)
            again = GPT.load(path)
        np.testing.assert_allclose(again.forward(x).data, logits.data, rtol=1e-5, atol=1e-6)

    def test_kv_cache_matches_full_forward(self):
        model = GPT(self.cfg, seed=3)
        ids = list(np.random.default_rng(1).integers(0, 50, 12))
        cache = [None] * self.cfg.n_layer
        cached = [model._infer(np.array(ids[:5]), cache)]  # prompt in one go
        for t in ids[5:]:
            cached.append(model._infer(np.array([t]), cache))  # then one token at a time
        full = model.forward(np.array([ids])).data[0]
        for j, logits in enumerate(cached):
            np.testing.assert_allclose(logits, full[4 + j], rtol=1e-4, atol=1e-5)

    def test_generation_past_context_window(self):
        model = GPT(self.cfg)
        out = list(model.generate([1, 2, 3], max_new_tokens=40, temperature=0.8, rng=np.random.default_rng(0)))
        self.assertEqual(len(out), 40)  # longer than n_ctx=16: the window slides without errors

    def test_learns_a_sequence(self):
        """A model should memorise a short repeating pattern in a few dozen steps."""
        model = GPT(self.cfg)
        seq = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]])
        opt = AdamW(model.params, lr=1e-2)
        first = None
        for _ in range(60):
            loss = model.loss(seq[:, :-1], seq[:, 1:])
            first = first if first is not None else float(loss.data)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(float(loss.data), first * 0.2)
        out = list(model.generate([1, 2, 3], max_new_tokens=4, temperature=0))
        self.assertEqual(out, [4, 5, 6, 7])


if __name__ == "__main__":
    unittest.main()
