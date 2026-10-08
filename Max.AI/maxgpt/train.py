"""Train MaxGPT from scratch (or continue training) on your own data.

Run from the Max.AI folder:
    python -m maxgpt.train                       # small model on maxgpt/data (default)
    python -m maxgpt.train --preset base --steps 6000
    python -m maxgpt.train --resume --steps 2000 # continue from the last checkpoint
"""
import argparse
import os
import time

import numpy as np

from .data import Dataset, default_files, format_prompt
from .model import GPT, GPTConfig
from .optim import AdamW, cosine_lr
from .tokenizer import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))

PRESETS = {
    # name: (n_layer, n_head, n_embd, n_ctx, vocab)
    "tiny": (2, 2, 64, 128, 1024),      # for quick tests
    "small": (4, 4, 128, 256, 2048),    # ~1M params, trains on any laptop CPU
    "base": (6, 6, 192, 256, 4096),     # ~3.5M params, better answers, slower
    "large": (8, 8, 256, 384, 8192),    # ~8.5M params, needs more data and time
}


def evaluate(model, ds, batches=4, batch_size=8):
    losses = [float(model.loss(*ds.batch(batch_size, "val")).data) for _ in range(batches)]
    return sum(losses) / len(losses)


def sample(model, tok, question, max_new_tokens=60):
    ids = tok.encode(format_prompt([("user", question)]))
    out = model.generate(ids, max_new_tokens, temperature=0.7, stop_ids={tok.special["<|end|>"]})
    return tok.decode(list(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train MaxGPT")
    ap.add_argument("--data", default=os.path.join(HERE, "data"), help="folder with chat/ and text/ subfolders")
    ap.add_argument("--out", default=os.path.join(HERE, "checkpoints"), help="where to save the model")
    ap.add_argument("--preset", default="small", choices=PRESETS)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--resume", action="store_true", help="continue from the checkpoint in --out")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, "maxgpt.npz")
    tok_path = os.path.join(args.out, "tokenizer.json")
    opt_path = os.path.join(args.out, "optimizer.npz")
    chat_files, text_files = default_files(args.data)
    print(f"data: {len(chat_files)} chat file(s), {len(text_files)} text file(s)")

    if args.resume and os.path.exists(model_path):
        tok = Tokenizer.load(tok_path)
        model = GPT.load(model_path)
        print(f"resumed {model.num_params():,}-parameter model from {model_path}")
    else:
        n_layer, n_head, n_embd, n_ctx, vocab = PRESETS[args.preset]
        corpus = ""
        for path in chat_files + text_files:
            with open(path, encoding="utf-8") as f:
                corpus += f.read() + "\n"
        print(f"training tokenizer (vocab {vocab}) on {len(corpus):,} characters...")
        tok = Tokenizer.train(corpus, vocab_size=vocab)
        tok.save(tok_path)
        cfg = GPTConfig(vocab_size=tok.vocab_size, n_ctx=n_ctx, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
        model = GPT(cfg, seed=args.seed)
        print(f"new model: {model.num_params():,} parameters ({cfg})")

    ds = Dataset(tok, chat_files, text_files, n_ctx=model.config.n_ctx, seed=args.seed)
    print(f"examples: {len(ds.train)} train / {len(ds.val)} validation")
    opt = AdamW(model.params, lr=args.lr)
    if args.resume and os.path.exists(opt_path):
        with np.load(opt_path) as f:
            opt.load_state_dict(dict(f))
    best = float("inf")
    t0 = time.time()
    for step in range(args.steps):
        lr = cosine_lr(step, args.steps, args.lr, warmup=min(100, max(1, args.steps // 10)))
        x, y = ds.batch(args.batch)
        loss = model.loss(x, y)
        opt.zero_grad()
        loss.backward()
        opt.clip_grad_norm(1.0)
        opt.step(lr)
        if step % 25 == 0:
            rate = (step + 1) / (time.time() - t0)
            print(f"step {step:5d}/{args.steps} | loss {float(loss.data):.3f} | lr {lr:.2e} | {rate:.2f} steps/s", flush=True)
        last = step == args.steps - 1
        if (step + 1) % args.eval_every == 0 or last:
            val = evaluate(model, ds)
            print(f"  validation loss {val:.3f}")
            print(f"  sample: Who are you? -> {sample(model, tok, 'Who are you?')!r}", flush=True)
            if val < best or last:
                best = min(best, val)
                model.save(model_path)
                np.savez(opt_path, **opt.state_dict())
                print(f"  saved {model_path}")
    print(f"done in {time.time() - t0:.0f}s. Chat with it:  python -m maxgpt.chat")


if __name__ == "__main__":
    main()
