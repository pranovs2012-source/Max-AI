"""Fast training for MaxGPT with PyTorch (used on GitHub's machines for the big training runs).

The network is exactly MaxGPT's architecture (maxgpt/model.py): the trained weights are exported
in MaxGPT's own format, and the app keeps running them with NumPy only. PyTorch is just a faster
engine for the training maths.

    python -m maxgpt.train_torch --corpus corpus/ --out maxgpt/checkpoints --minutes 120
    python -m maxgpt.train_torch --corpus corpus/ --out ckpt --resume --minutes 180 --total-minutes 480

Training can be split into stages (GitHub jobs last at most 6 hours): --resume continues from the
weights and optimizer state of the previous stage, and the learning-rate schedule follows the
overall --total-minutes budget.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import GPTConfig

PRESETS = {
    # name: (n_layer, n_head, n_embd, n_ctx)
    "base": (6, 6, 192, 512),
    "large": (8, 8, 256, 512),
    "xl": (10, 8, 320, 512),
}


class TorchGPT(torch.nn.Module):
    """MaxGPT with PyTorch tensors. Parameter names and shapes match maxgpt/model.py."""

    def __init__(self, cfg, seed=1337):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        E, L = cfg.n_embd, cfg.n_layer
        proj_std = 0.02 / math.sqrt(2 * L)

        def w(*shape, std=0.02):
            return torch.nn.Parameter(torch.randn(*shape, generator=g) * std)

        def const(value, n):
            return torch.nn.Parameter(torch.full((n,), float(value)))

        p = {"wte": w(cfg.vocab_size, E), "wpe": w(cfg.n_ctx, E, std=0.01)}
        for i in range(L):
            h = f"h{i}."
            p.update({
                h + "ln1.w": const(1, E), h + "ln1.b": const(0, E),
                h + "attn.wq": w(E, E), h + "attn.bq": const(0, E),
                h + "attn.wk": w(E, E), h + "attn.bk": const(0, E),
                h + "attn.wv": w(E, E), h + "attn.bv": const(0, E),
                h + "attn.wo": w(E, E, std=proj_std), h + "attn.bo": const(0, E),
                h + "ln2.w": const(1, E), h + "ln2.b": const(0, E),
                h + "mlp.w1": w(E, 4 * E), h + "mlp.b1": const(0, 4 * E),
                h + "mlp.w2": w(4 * E, E, std=proj_std), h + "mlp.b2": const(0, E),
            })
        p.update({"lnf.w": const(1, E), "lnf.b": const(0, E)})
        # ParameterDict keys can't contain "."; keep the MaxGPT names in a mapping
        self.names = list(p)
        self.params = torch.nn.ParameterList([p[n] for n in self.names])
        self.index = {n: i for i, n in enumerate(self.names)}

    def P(self, name):
        return self.params[self.index[name]]

    def forward(self, idx):
        c = self.cfg
        B, T = idx.shape
        x = self.P("wte")[idx] + self.P("wpe")[:T]
        for i in range(c.n_layer):
            h = f"h{i}."
            a = F.layer_norm(x, (c.n_embd,), self.P(h + "ln1.w"), self.P(h + "ln1.b"), eps=1e-5)

            def heads(t):
                return t.view(B, T, c.n_head, c.head_dim).transpose(1, 2)
            q = heads(a @ self.P(h + "attn.wq") + self.P(h + "attn.bq"))
            k = heads(a @ self.P(h + "attn.wk") + self.P(h + "attn.bk"))
            v = heads(a @ self.P(h + "attn.wv") + self.P(h + "attn.bv"))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).reshape(B, T, c.n_embd)
            x = x + y @ self.P(h + "attn.wo") + self.P(h + "attn.bo")
            a = F.layer_norm(x, (c.n_embd,), self.P(h + "ln2.w"), self.P(h + "ln2.b"), eps=1e-5)
            a = F.gelu(a @ self.P(h + "mlp.w1") + self.P(h + "mlp.b1"), approximate="tanh")
            x = x + a @ self.P(h + "mlp.w2") + self.P(h + "mlp.b2")
        x = F.layer_norm(x, (c.n_embd,), self.P("lnf.w"), self.P("lnf.b"), eps=1e-5)
        return x @ self.P("wte").t()

    # ── MaxGPT checkpoint format ────────────────────────────────────────────
    def export(self, path, half=True):
        arrays = {n: self.P(n).detach().cpu().numpy() for n in self.names}
        if half:
            arrays = {n: a.astype(np.float16) for n, a in arrays.items()}
        from dataclasses import asdict
        np.savez_compressed(path, __config__=np.array(json.dumps(asdict(self.cfg))), **arrays)

    @classmethod
    def from_maxgpt(cls, path):
        with np.load(path) as f:
            cfg = GPTConfig(**json.loads(str(f["__config__"])))
            model = cls(cfg)
            with torch.no_grad():
                for n in model.names:
                    model.P(n).copy_(torch.from_numpy(f[n].astype(np.float32)))
        return model


# ── data ─────────────────────────────────────────────────────────────────────
class PackedData:
    """Examples (token ids + per-token loss weights) packed whole into fixed windows.

    corpus/<split>.npz holds `ids` (uint16), `weights` (uint8: 0 none, 1 context, 2 answer)
    and `offsets`, as written by maxgpt.corpus.
    """

    def __init__(self, path, n_ctx, pad_id, seed=0):
        with np.load(path) as f:
            ids, wts, off = f["ids"].astype(np.int64), f["weights"], f["offsets"]
        self.examples = [(ids[a:b], wts[a:b]) for a, b in zip(off[:-1], off[1:]) if b - a > 1]
        self.n_ctx, self.pad, self.rng = n_ctx, pad_id, np.random.default_rng(seed)
        self.tokens = int(sum(len(e[0]) for e in self.examples))

    def windows(self):
        """One epoch of windows: examples in random order, packed greedily without splitting."""
        order = self.rng.permutation(len(self.examples))
        size = self.n_ctx + 1
        cur_ids, cur_w = [], []
        for i in order:
            ids, w = self.examples[i]
            for s in range(0, len(ids), self.n_ctx):        # very long examples are split
                pi, pw = ids[s:s + size], w[s:s + size]
                if len(pi) < 2:
                    continue
                if sum(len(x) for x in cur_ids) + len(pi) > size and cur_ids:
                    yield self._finish(cur_ids, cur_w)
                    cur_ids, cur_w = [], []
                cur_ids.append(pi)
                cur_w.append(pw)
        if cur_ids:
            yield self._finish(cur_ids, cur_w)

    def _finish(self, ids_list, w_list):
        ids = np.concatenate(ids_list)[: self.n_ctx + 1]
        w = np.concatenate(w_list)[: self.n_ctx + 1]
        x = np.full(self.n_ctx, self.pad, dtype=np.int64)
        y = np.full(self.n_ctx, self.pad, dtype=np.int64)
        wt = np.zeros(self.n_ctx, dtype=np.int64)          # 0 no loss, 1 context, 2 answer
        n = len(ids) - 1
        x[:n], y[:n], wt[:n] = ids[:-1], ids[1:], w[1:]
        # don't learn to predict the first token of the next packed example from the previous one
        starts = np.cumsum([len(a) for a in ids_list])[:-1] - 1
        wt[starts[starts < n]] = 0
        return x, y, wt

    def batches(self, batch_size, ctx_weight):
        weight_of = np.array([0.0, ctx_weight, 1.0], dtype=np.float32)
        while True:
            buf = []
            for x, y, codes in self.windows():
                buf.append((x, y, weight_of[codes]))
                if len(buf) == batch_size:
                    xs, ys, ws = map(np.stack, zip(*buf))
                    yield torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ws)
                    buf = []


def weighted_loss(model, x, y, w):
    logits = model(x)
    nll = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="none")
    w = w.view(-1)
    return (nll * w).sum() / w.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, data, batches=20, batch_size=16):
    model.eval()
    total, n = 0.0, 0
    for i, (x, y, w) in enumerate(data.batches(batch_size, ctx_weight=0.0)):  # answer tokens only
        if i >= batches:
            break
        total += float(weighted_loss(model, x, y, w))
        n += 1
    model.train()
    return total / max(n, 1)


def lr_at(progress, peak, warmup=0.02, floor=0.1):
    """Warmup, then cosine decay over the whole multi-stage run (progress 0..1)."""
    if progress < warmup:
        return peak * progress / warmup
    t = min(1.0, (progress - warmup) / (1 - warmup))
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def sample(model, tok, prompt, max_new=80):
    from .data import format_prompt
    ids = tok.encode(format_prompt([("user", prompt)]))
    end = tok.special["<|end|>"]
    model.eval()
    out = []
    for _ in range(max_new):
        x = torch.tensor([ids[-model.cfg.n_ctx:]])
        nxt = int(torch.argmax(model(x)[0, -1]))
        if nxt == end:
            break
        ids.append(nxt)
        out.append(nxt)
    model.train()
    return tok.decode(out)


def main(argv=None):
    from .tokenizer import Tokenizer
    ap = argparse.ArgumentParser(description="Train MaxGPT with PyTorch")
    ap.add_argument("--corpus", required=True, help="folder written by maxgpt.corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="large", choices=PRESETS)
    ap.add_argument("--minutes", type=float, default=60, help="training time for this stage")
    ap.add_argument("--total-minutes", type=float, default=None, help="planned time of all stages (LR schedule)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ctx-weight", type=float, default=0.2, help="loss weight of non-answer tokens")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    total_minutes = args.total_minutes or args.minutes
    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, "maxgpt.npz")
    state_path = os.path.join(args.out, "train_state.pt")
    meta_path = os.path.join(args.out, "meta.json")

    tok = Tokenizer.load(os.path.join(args.corpus, "tokenizer.json"))
    tok.save(os.path.join(args.out, "tokenizer.json"))
    meta = {"minutes": 0.0, "steps": 0, "tokens": 0}
    if args.resume and os.path.exists(model_path):
        model = TorchGPT.from_maxgpt(model_path)
        if os.path.exists(meta_path):
            meta.update(json.load(open(meta_path)))
        print(f"resumed {sum(p.numel() for p in model.parameters()):,} parameters after {meta['minutes']:.0f} min")
    else:
        L, H, E, T = PRESETS[args.preset]
        model = TorchGPT(GPTConfig(vocab_size=tok.vocab_size, n_ctx=T, n_layer=L, n_head=H, n_embd=E))
        print(f"new model: {sum(p.numel() for p in model.parameters()):,} parameters, {model.cfg}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
    if args.resume and os.path.exists(state_path):
        opt.load_state_dict(torch.load(state_path)["opt"])

    pad = tok.special["<|pad|>"]
    train = PackedData(os.path.join(args.corpus, "train.npz"), model.cfg.n_ctx, pad, seed=meta["steps"])
    val = PackedData(os.path.join(args.corpus, "val.npz"), model.cfg.n_ctx, pad, seed=1)
    print(f"train: {len(train.examples):,} examples, {train.tokens:,} tokens · val: {len(val.examples):,} examples")
    batches = train.batches(args.batch, args.ctx_weight)

    def save(tag=""):
        model.export(model_path)
        torch.save({"opt": opt.state_dict()}, state_path)
        meta.update(val_loss=evaluate(model, val), config=model.cfg.__dict__, grounded=True)
        json.dump(meta, open(meta_path, "w"), indent=1)
        print(f"  saved {tag} · validation loss (answers) {meta['val_loss']:.3f}", flush=True)

    start = last_save = last_log = time.time()
    before = meta["minutes"]          # minutes trained in earlier stages
    stage_tokens = 0
    model.train()
    while True:
        elapsed = (time.time() - start) / 60
        if elapsed >= args.minutes:
            break
        lr = lr_at((before + elapsed) / total_minutes, args.lr)
        for group in opt.param_groups:
            group["lr"] = lr
        x, y, w = next(batches)
        loss = weighted_loss(model, x, y, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        meta["steps"] += 1
        stage_tokens += int((w > 0).sum())
        if time.time() - last_log > 60:
            rate = stage_tokens / (time.time() - start)
            print(f"step {meta['steps']:6d} | {elapsed:6.1f} min | loss {float(loss):.3f} | lr {lr:.2e} | "
                  f"{rate:,.0f} tok/s", flush=True)
            last_log = time.time()
        if time.time() - last_save > 30 * 60:
            meta["minutes"] = before + (time.time() - start) / 60
            meta["tokens_this_stage"] = stage_tokens
            save("checkpoint")
            for q in ("Who are you?", "What is photosynthesis?"):
                print(f"  sample: {q} -> {sample(model, tok, q)!r}", flush=True)
            last_save = time.time()
    meta["minutes"] = before + (time.time() - start) / 60
    meta["tokens"] = meta.get("tokens", 0) + stage_tokens
    save("final")
    for q in ("Who are you?", "Who created you?", "What is photosynthesis?", "Write a short poem about rain"):
        print(f"  sample: {q} -> {sample(model, tok, q)!r}", flush=True)


if __name__ == "__main__":
    main()
