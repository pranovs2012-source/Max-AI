"""A tiny reverse-mode autograd engine on top of NumPy.

Only the operations MaxGPT needs are implemented, and the expensive ones
(attention, layer norm, cross-entropy) are fused so they stay fast and
numerically stable. Every op is verified against finite differences in
tests/test_gradcheck.py.
"""
from contextlib import contextmanager

import numpy as np

DTYPE = np.float32
_GRAD_ENABLED = [True]


@contextmanager
def no_grad():
    """Disable graph recording (inference / generation)."""
    prev = _GRAD_ENABLED[0]
    _GRAD_ENABLED[0] = False
    try:
        yield
    finally:
        _GRAD_ENABLED[0] = prev


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward")

    def __init__(self, data, requires_grad=False, _parents=(), _backward=None):
        self.data = data if isinstance(data, np.ndarray) and data.dtype == DTYPE else np.asarray(data, dtype=DTYPE)
        self.grad = None
        self.requires_grad = requires_grad
        self._parents = _parents
        self._backward = _backward

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    def _accumulate(self, g):
        if not self.requires_grad:
            return
        g = _unbroadcast(g, self.data.shape)
        self.grad = g if self.grad is None else self.grad + g

    def backward(self, grad=None):
        """Backpropagate from this tensor (usually a scalar loss)."""
        topo, seen = [], set()
        stack = [(self, False)]
        while stack:  # iterative DFS, avoids recursion limits on deep graphs
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if p.requires_grad and id(p) not in seen:
                    stack.append((p, False))
        self.grad = np.ones_like(self.data) if grad is None else grad.astype(DTYPE)
        for node in reversed(topo):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)
                if node is not self:
                    node.grad = None  # free intermediate gradients as we go

    def __add__(self, other):
        return add(self, other)

    def __matmul__(self, other):
        return matmul(self, other)

    def reshape(self, *shape):
        return reshape(self, shape)

    def transpose(self, *axes):
        return transpose(self, axes)


def _make(data, parents, backward):
    """Create an op result, recording the graph only when needed."""
    if _GRAD_ENABLED[0] and any(p.requires_grad for p in parents):
        return Tensor(data, True, parents, backward)
    return Tensor(data)


def _unbroadcast(g, shape):
    """Sum a broadcast gradient back down to `shape`."""
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def as_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x)


# ---------------------------------------------------------------------------
# Elementary ops
# ---------------------------------------------------------------------------
def add(a, b):
    a, b = as_tensor(a), as_tensor(b)

    def backward(g):
        a._accumulate(g)
        b._accumulate(g)
    return _make(a.data + b.data, (a, b), backward)


def mul(a, b):
    a, b = as_tensor(a), as_tensor(b)

    def backward(g):
        a._accumulate(g * b.data)
        b._accumulate(g * a.data)
    return _make(a.data * b.data, (a, b), backward)


def matmul(a, b):
    a, b = as_tensor(a), as_tensor(b)

    def backward(g):
        if a.requires_grad:
            a._accumulate(g @ np.swapaxes(b.data, -1, -2))
        if b.requires_grad:
            if b.data.ndim == 2 and a.data.ndim > 2:
                # (..., m, k) @ (k, n): fold batch dims into one big matmul
                k, n = b.data.shape
                b._accumulate(a.data.reshape(-1, k).T @ g.reshape(-1, n))
            else:
                b._accumulate(np.swapaxes(a.data, -1, -2) @ g)
    return _make(a.data @ b.data, (a, b), backward)


def reshape(a, shape):
    old = a.data.shape

    def backward(g):
        a._accumulate(g.reshape(old))
    return _make(a.data.reshape(shape), (a,), backward)


def transpose(a, axes):
    axes = tuple(axes) if axes else tuple(reversed(range(a.data.ndim)))
    inv = tuple(np.argsort(axes))

    def backward(g):
        a._accumulate(g.transpose(inv))
    return _make(a.data.transpose(axes), (a,), backward)


def embedding(weight, idx):
    """Row lookup: weight[idx]. idx is an int array of any shape."""
    idx = np.asarray(idx)

    def backward(g):
        if weight.requires_grad:
            gw = np.zeros_like(weight.data)
            np.add.at(gw, idx.reshape(-1), g.reshape(-1, weight.data.shape[1]))
            weight._accumulate(gw)
    return _make(weight.data[idx], (weight,), backward)


# ---------------------------------------------------------------------------
# Fused neural-network ops
# ---------------------------------------------------------------------------
_GELU_C = np.float32(np.sqrt(2.0 / np.pi))


def gelu(x):
    """GELU, tanh approximation (as used by GPT-2)."""
    xd = x.data
    t = np.tanh(_GELU_C * (xd + 0.044715 * xd ** 3))
    out = 0.5 * xd * (1.0 + t)

    def backward(g):
        dt = (1.0 - t * t) * _GELU_C * (1.0 + 3 * 0.044715 * xd * xd)
        x._accumulate(g * (0.5 * (1.0 + t) + 0.5 * xd * dt))
    return _make(out, (x,), backward)


def layer_norm(x, w, b, eps=1e-5):
    xd = x.data
    mu = xd.mean(-1, keepdims=True)
    var = xd.var(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = (xd - mu) * rstd
    out = xhat * w.data + b.data

    def backward(g):
        if w.requires_grad:
            w._accumulate((g * xhat).reshape(-1, xd.shape[-1]).sum(0))
        if b.requires_grad:
            b._accumulate(g.reshape(-1, xd.shape[-1]).sum(0))
        if x.requires_grad:
            dxhat = g * w.data
            x._accumulate(rstd * (dxhat - dxhat.mean(-1, keepdims=True)
                                  - xhat * (dxhat * xhat).mean(-1, keepdims=True)))
    return _make(out.astype(DTYPE), (x, w, b), backward)


def causal_attention(q, k, v):
    """softmax(q·kᵀ/√d + causal mask)·v for tensors shaped (B, H, T, d)."""
    T, d = q.data.shape[-2], q.data.shape[-1]
    scale = np.float32(1.0 / np.sqrt(d))
    s = (q.data @ np.swapaxes(k.data, -1, -2)) * scale
    mask = np.triu(np.ones((T, T), dtype=bool), 1)
    s[..., mask] = -1e9
    s -= s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    out = p @ v.data

    def backward(g):
        if v.requires_grad:
            v._accumulate(np.swapaxes(p, -1, -2) @ g)
        dp = g @ np.swapaxes(v.data, -1, -2)
        ds = p * (dp - (dp * p).sum(-1, keepdims=True)) * scale
        if q.requires_grad:
            q._accumulate(ds @ k.data)
        if k.requires_grad:
            k._accumulate(np.swapaxes(ds, -1, -2) @ q.data)
    return _make(out, (q, k, v), backward)


def cross_entropy(logits, targets, ignore_index=-1):
    """Mean cross-entropy over positions whose target != ignore_index.

    logits: (N, V) Tensor, targets: (N,) int array.
    """
    ld = logits.data
    targets = np.asarray(targets)
    valid = targets != ignore_index
    n = max(int(valid.sum()), 1)
    z = ld - ld.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    safe_t = np.where(valid, targets, 0)
    picked = logp[np.arange(len(targets)), safe_t]
    loss = -(picked * valid).sum() / n

    def backward(g):
        grad = np.exp(logp)
        grad[np.arange(len(targets)), safe_t] -= 1.0
        grad *= (valid[:, None] / n) * g
        logits._accumulate(grad.astype(DTYPE))
    return _make(np.asarray(loss, dtype=DTYPE), (logits,), backward)
