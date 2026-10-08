"""MaxGPT: a decoder-only transformer (GPT architecture) in pure NumPy."""
import json
from dataclasses import asdict, dataclass

import numpy as np

from . import autograd as ag
from .autograd import Tensor


@dataclass
class GPTConfig:
    vocab_size: int = 2048
    n_ctx: int = 256      # maximum context length in tokens
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128

    @property
    def head_dim(self):
        return self.n_embd // self.n_head


class GPT:
    def __init__(self, config, seed=1337):
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.config = c = config
        rng = np.random.default_rng(seed)

        def w(*shape, std=0.02):
            return Tensor(rng.normal(0.0, std, size=shape), requires_grad=True)

        def const(value, *shape):
            return Tensor(np.full(shape, value), requires_grad=True)

        proj_std = 0.02 / np.sqrt(2 * c.n_layer)  # GPT-2 residual scaling
        self.params = {"wte": w(c.vocab_size, c.n_embd), "wpe": w(c.n_ctx, c.n_embd, std=0.01)}
        for i in range(c.n_layer):
            p = f"h{i}."
            self.params.update({
                p + "ln1.w": const(1.0, c.n_embd), p + "ln1.b": const(0.0, c.n_embd),
                p + "attn.wq": w(c.n_embd, c.n_embd), p + "attn.bq": const(0.0, c.n_embd),
                p + "attn.wk": w(c.n_embd, c.n_embd), p + "attn.bk": const(0.0, c.n_embd),
                p + "attn.wv": w(c.n_embd, c.n_embd), p + "attn.bv": const(0.0, c.n_embd),
                p + "attn.wo": w(c.n_embd, c.n_embd, std=proj_std), p + "attn.bo": const(0.0, c.n_embd),
                p + "ln2.w": const(1.0, c.n_embd), p + "ln2.b": const(0.0, c.n_embd),
                p + "mlp.w1": w(c.n_embd, 4 * c.n_embd), p + "mlp.b1": const(0.0, 4 * c.n_embd),
                p + "mlp.w2": w(4 * c.n_embd, c.n_embd, std=proj_std), p + "mlp.b2": const(0.0, c.n_embd),
            })
        self.params.update({"lnf.w": const(1.0, c.n_embd), "lnf.b": const(0.0, c.n_embd)})

    def num_params(self):
        return sum(p.data.size for p in self.params.values())

    # -- forward ---------------------------------------------------------------
    def forward(self, idx):
        """idx: int array (B, T) -> logits Tensor (B, T, vocab)."""
        c, P = self.config, self.params
        B, T = idx.shape
        assert T <= c.n_ctx, f"sequence length {T} exceeds context {c.n_ctx}"
        x = ag.embedding(P["wte"], idx) + ag.embedding(P["wpe"], np.arange(T))

        def heads(t):
            return t.reshape(B, T, c.n_head, c.head_dim).transpose(0, 2, 1, 3)

        for i in range(c.n_layer):
            p = f"h{i}."
            h = ag.layer_norm(x, P[p + "ln1.w"], P[p + "ln1.b"])
            q = heads(h @ P[p + "attn.wq"] + P[p + "attn.bq"])
            k = heads(h @ P[p + "attn.wk"] + P[p + "attn.bk"])
            v = heads(h @ P[p + "attn.wv"] + P[p + "attn.bv"])
            a = ag.causal_attention(q, k, v).transpose(0, 2, 1, 3).reshape(B, T, c.n_embd)
            x = x + (a @ P[p + "attn.wo"] + P[p + "attn.bo"])
            h = ag.layer_norm(x, P[p + "ln2.w"], P[p + "ln2.b"])
            h = ag.gelu(h @ P[p + "mlp.w1"] + P[p + "mlp.b1"])
            x = x + (h @ P[p + "mlp.w2"] + P[p + "mlp.b2"])
        x = ag.layer_norm(x, P["lnf.w"], P["lnf.b"])
        return x @ P["wte"].transpose(1, 0)  # weight tying with the token embedding

    def loss(self, idx, targets):
        logits = self.forward(idx)
        return ag.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))

    # -- fast inference with a key/value cache -------------------------------------
    def _infer(self, new_ids, cache):
        """Plain-NumPy forward pass for new tokens only, reusing cached keys/values.

        cache: list (one per layer) of [k, v] arrays shaped (H, T_so_far, d), or None.
        Returns logits for the last new token.
        """
        c, P = self.config, {k: v.data for k, v in self.params.items()}
        start = 0 if cache[0] is None else cache[0][0].shape[1]
        T = len(new_ids)
        x = P["wte"][new_ids] + P["wpe"][start:start + T]

        def ln(t, w, b):
            mu = t.mean(-1, keepdims=True)
            return (t - mu) / np.sqrt(t.var(-1, keepdims=True) + 1e-5) * w + b

        def split(t):
            return t.reshape(T, c.n_head, c.head_dim).transpose(1, 0, 2)

        for i in range(c.n_layer):
            p = f"h{i}."
            h = ln(x, P[p + "ln1.w"], P[p + "ln1.b"])
            q = split(h @ P[p + "attn.wq"] + P[p + "attn.bq"])
            k = split(h @ P[p + "attn.wk"] + P[p + "attn.bk"])
            v = split(h @ P[p + "attn.wv"] + P[p + "attn.bv"])
            if cache[i] is not None:
                k = np.concatenate([cache[i][0], k], axis=1)
                v = np.concatenate([cache[i][1], v], axis=1)
            cache[i] = [k, v]
            s = q @ k.transpose(0, 2, 1) / np.sqrt(c.head_dim)
            total = k.shape[1]
            # new token j (absolute position start+j) may see keys 0..start+j
            mask = np.arange(total)[None, :] > (start + np.arange(T))[:, None]
            s = np.where(mask, -1e9, s)
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            a = (s @ v).transpose(1, 0, 2).reshape(T, c.n_embd)
            x = x + a @ P[p + "attn.wo"] + P[p + "attn.bo"]
            h = ln(x, P[p + "ln2.w"], P[p + "ln2.b"])
            h = h @ P[p + "mlp.w1"] + P[p + "mlp.b1"]
            h = 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h ** 3)))
            x = x + h @ P[p + "mlp.w2"] + P[p + "mlp.b2"]
        x = ln(x[-1], P["lnf.w"], P["lnf.b"])
        return x @ P["wte"].T

    # -- generation --------------------------------------------------------------
    def generate(self, ids, max_new_tokens=200, temperature=0.8, top_k=40, top_p=0.95,
                 stop_ids=(), repetition_penalty=1.1, rng=None):
        """Sample tokens one at a time. Yields each new token id."""
        rng = rng or np.random.default_rng()
        ids = list(ids)[-self.config.n_ctx:]
        cache, pending = [None] * self.config.n_layer, list(ids)
        with ag.no_grad():
            for _ in range(max_new_tokens):
                if len(ids) > self.config.n_ctx:  # context full: slide window, rebuild cache
                    ids = ids[-self.config.n_ctx:]
                    cache, pending = [None] * self.config.n_layer, list(ids)
                logits = self._infer(np.array(pending), cache).astype(np.float64)
                if repetition_penalty != 1.0:
                    recent = list(set(ids[-64:]))
                    r = logits[recent]
                    logits[recent] = np.where(r > 0, r / repetition_penalty, r * repetition_penalty)
                if temperature <= 0:
                    nxt = int(np.argmax(logits))
                else:
                    logits /= temperature
                    if top_k and top_k < len(logits):
                        kth = np.partition(logits, -top_k)[-top_k]
                        logits[logits < kth] = -np.inf
                    probs = np.exp(logits - logits.max())
                    probs /= probs.sum()
                    if top_p and top_p < 1.0:
                        order = np.argsort(-probs)
                        cum = np.cumsum(probs[order])
                        probs[order[cum - probs[order] > top_p]] = 0.0
                        probs /= probs.sum()
                    nxt = int(rng.choice(len(probs), p=probs))
                if nxt in stop_ids:
                    return
                ids.append(nxt)
                pending = [nxt]
                yield nxt

    # -- persistence -------------------------------------------------------------
    def save(self, path):
        np.savez_compressed(path, __config__=np.array(json.dumps(asdict(self.config))),
                            **{k: v.data for k, v in self.params.items()})

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            model = cls(GPTConfig(**json.loads(str(f["__config__"]))))
            for k in model.params:
                model.params[k].data = f[k].astype(ag.DTYPE)
        return model
