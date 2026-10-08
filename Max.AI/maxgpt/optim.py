"""AdamW optimizer, gradient clipping and a warmup + cosine learning-rate schedule."""
import math

import numpy as np


class AdamW:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.params = params  # dict name -> Tensor
        self.lr, self.betas, self.eps, self.wd = lr, betas, eps, weight_decay
        self.t = 0
        self.m = {k: np.zeros_like(p.data) for k, p in params.items()}
        self.v = {k: np.zeros_like(p.data) for k, p in params.items()}

    def zero_grad(self):
        for p in self.params.values():
            p.grad = None

    def clip_grad_norm(self, max_norm=1.0):
        total = math.sqrt(sum(float((p.grad ** 2).sum()) for p in self.params.values() if p.grad is not None))
        if total > max_norm:
            scale = max_norm / (total + 1e-6)
            for p in self.params.values():
                if p.grad is not None:
                    p.grad *= scale
        return total

    def step(self, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        b1, b2 = self.betas
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for k, p in self.params.items():
            if p.grad is None:
                continue
            m, v = self.m[k], self.v[k]
            m *= b1
            m += (1 - b1) * p.grad
            v *= b2
            v += (1 - b2) * p.grad * p.grad
            if p.data.ndim >= 2:  # decay matrices, not biases / norm gains
                p.data -= lr * self.wd * p.data
            p.data -= lr * (m / c1) / (np.sqrt(v / c2) + self.eps)

    def state_dict(self):
        return {"t": np.array(self.t), **{f"m.{k}": v for k, v in self.m.items()},
                **{f"v.{k}": v for k, v in self.v.items()}}

    def load_state_dict(self, state):
        self.t = int(state["t"])
        for k in self.m:
            if f"m.{k}" in state and state[f"m.{k}"].shape == self.m[k].shape:
                self.m[k] = state[f"m.{k}"].copy()
                self.v[k] = state[f"v.{k}"].copy()


def cosine_lr(step, max_steps, base_lr, warmup=100, min_ratio=0.1):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
