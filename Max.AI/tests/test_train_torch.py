"""The PyTorch trainer must produce exactly the network the app runs with NumPy."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:  # the app itself doesn't need PyTorch
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class ParityTest(unittest.TestCase):
    def test_exported_weights_match_numpy_model(self):
        from maxgpt.model import GPT, GPTConfig
        from maxgpt.train_torch import TorchGPT
        cfg = GPTConfig(vocab_size=97, n_ctx=32, n_layer=2, n_head=4, n_embd=32)
        tm = TorchGPT(cfg, seed=3)
        with torch.no_grad():                      # make biases and norms non-trivial too
            for p in tm.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        ids = np.random.default_rng(0).integers(0, 97, (2, 20))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            tm.export(path, half=False)
            nm = GPT.load(path)
            again = TorchGPT.from_maxgpt(path)
        expected = tm(torch.from_numpy(ids)).detach().numpy()
        np.testing.assert_allclose(nm.forward(ids).data, expected, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(again(torch.from_numpy(ids)).detach().numpy(), expected, rtol=1e-5, atol=1e-5)
        # the fast KV-cache inference path agrees as well
        cache = [None] * cfg.n_layer
        np.testing.assert_allclose(nm._infer(ids[0], cache), expected[0, -1], rtol=1e-4, atol=1e-4)

    def test_half_precision_export_loads(self):
        from maxgpt.model import GPT, GPTConfig
        from maxgpt.train_torch import TorchGPT
        tm = TorchGPT(GPTConfig(vocab_size=50, n_ctx=16, n_layer=1, n_head=2, n_embd=16))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            tm.export(path)                         # float16 on disk
            nm = GPT.load(path)
        self.assertEqual(nm.params["wte"].data.dtype, np.float32)
        ids = np.arange(10)[None]
        np.testing.assert_allclose(nm.forward(ids).data, tm(torch.from_numpy(ids)).detach().numpy(), atol=2e-3)


if __name__ == "__main__":
    unittest.main()
