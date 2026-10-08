"""Finite-difference checks for every autograd op MaxGPT uses.

Run from the Max.AI folder:  python -m unittest discover -s tests
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maxgpt import autograd as ag  # noqa: E402


def numeric_grad(f, t, eps=1e-6):
    g = np.zeros_like(t.data)
    it = np.nditer(t.data, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = t.data[i]
        t.data[i] = old + eps
        hi = float(f().data)
        t.data[i] = old - eps
        lo = float(f().data)
        t.data[i] = old
        g[i] = (hi - lo) / (2 * eps)
    return g


class GradCheck(unittest.TestCase):
    def setUp(self):
        self._dtype = ag.DTYPE
        ag.DTYPE = np.float64  # double precision for accurate checks
        self.rng = np.random.default_rng(0)

    def tearDown(self):
        ag.DTYPE = self._dtype

    def p(self, *shape):
        return ag.Tensor(self.rng.normal(size=shape), requires_grad=True)

    def check(self, f, *params, tol=1e-5):
        for t in params:
            t.grad = None
        f().backward()
        for t in params:
            num = numeric_grad(f, t)
            err = np.max(np.abs(num - t.grad)) / (np.max(np.abs(num)) + 1e-8)
            self.assertLess(err, tol, f"gradient mismatch for {t.shape}: {err}")

    def readout(self, out, seed=1):
        """Random weighted readout so every output element affects the loss."""
        r = np.random.default_rng(seed).normal(size=out.shape)
        flat = ag.reshape(ag.mul(out, r), (1, -1))
        return ag.cross_entropy(flat, np.zeros(1, dtype=int))

    def test_matmul_add_broadcast(self):
        a, w, b = self.p(2, 3, 4), self.p(4, 5), self.p(5)
        self.check(lambda: self.readout(a @ w + b), a, w, b)

    def test_batched_matmul(self):
        a, b = self.p(2, 3, 4, 5), self.p(2, 3, 5, 2)
        self.check(lambda: self.readout(a @ b), a, b)

    def test_reshape_transpose(self):
        a = self.p(2, 3, 4)
        self.check(lambda: self.readout(a.reshape(2, 3, 2, 2).transpose(0, 2, 1, 3)), a)

    def test_gelu(self):
        a = self.p(3, 4)
        self.check(lambda: self.readout(ag.gelu(a)), a)

    def test_layer_norm(self):
        x, w, b = self.p(2, 3, 6), self.p(6), self.p(6)
        self.check(lambda: self.readout(ag.layer_norm(x, w, b)), x, w, b)

    def test_attention(self):
        q, k, v = self.p(1, 2, 4, 3), self.p(1, 2, 4, 3), self.p(1, 2, 4, 3)
        self.check(lambda: self.readout(ag.causal_attention(q, k, v)), q, k, v)

    def test_embedding(self):
        w = self.p(7, 3)
        idx = np.array([[1, 3, 3], [0, 6, 1]])
        self.check(lambda: self.readout(ag.embedding(w, idx)), w)

    def test_cross_entropy_ignore(self):
        logits = self.p(5, 6)
        t = np.array([1, -1, 5, 0, -1])
        self.check(lambda: ag.cross_entropy(logits, t), logits)

    def test_attention_is_causal(self):
        q, k, v = self.p(1, 1, 4, 3), self.p(1, 1, 4, 3), self.p(1, 1, 4, 3)
        out1 = ag.causal_attention(q, k, v).data.copy()
        v.data[0, 0, 3] += 10.0  # changing the last token must not affect earlier ones
        out2 = ag.causal_attention(q, k, v).data
        np.testing.assert_allclose(out1[0, 0, :3], out2[0, 0, :3])


if __name__ == "__main__":
    unittest.main()
