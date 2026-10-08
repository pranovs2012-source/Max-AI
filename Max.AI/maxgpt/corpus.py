"""Build MaxGPT's big training corpus.

Sources (downloaded when the corpus is built, not stored in the repository):

* Databricks Dolly 15k (CC BY-SA 3.0) — 15,000 human-written instructions and answers:
  open questions, explanations, brainstorming, writing, summaries, questions about a passage.
* SQuAD v1.1 (CC BY-SA 4.0) — 87,000 questions about Wikipedia paragraphs. Turned into
  "answer from search results" lessons: Max reads one or more results (sometimes with an
  unrelated distractor, sometimes with no answer at all) and replies with a complete sentence.
* Max's own conversations in maxgpt/data/chat (identity, coding, world facts, everyday help),
  repeated so they stay strong.

    python -m maxgpt.corpus --out corpus --vocab 8192

Writes corpus/train.npz, corpus/val.npz (token ids, loss-weight codes, offsets),
corpus/tokenizer.json and corpus/stats.json.
"""
import argparse
import json
import os
import random
import re
import urllib.request

import numpy as np

from .data import NO_ANSWER, SYSTEM_PROMPT, augment_conversation, default_files, grounded_question, parse_conversations
from .tokenizer import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DOLLY_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
SQUAD_URL = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json"

NO_ANSWER_VARIANTS = [
    NO_ANSWER,
    "The search results don't say, so I'd rather not guess.",
    "I couldn't find that in the search results. Could you rephrase the question?",
]


def fetch(src, cache_dir):
    if not src.startswith("http"):
        return src
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, os.path.basename(src.split("?")[0]))
    if not os.path.exists(path):
        print(f"downloading {src}", flush=True)
        req = urllib.request.Request(src, headers={"User-Agent": "MaxAI-trainer/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.replace(path + ".part", path)
    return path


def clean(text):
    text = re.sub(r"\[\d+\]|\[citation needed\]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def sentences(text):
    out, pos = [], 0
    for part in _SENT.split(text):
        start = text.find(part, pos)
        out.append((start, part))
        pos = start + len(part)
    return out


def trim_passage(text, keep_at=None, limit=700):
    """At most `limit` characters of whole sentences, keeping the sentence at `keep_at`."""
    sents = sentences(text)
    if len(text) <= limit:
        return text
    if keep_at is None:
        idx = 0
    else:
        idx = max(i for i, (start, _) in enumerate(sents) if start <= keep_at)
    chosen, lo, hi = [sents[idx][1]], idx, idx
    size = len(sents[idx][1])
    while True:
        grew = False
        for j in (lo - 1, hi + 1):
            if 0 <= j < len(sents) and size + len(sents[j][1]) + 1 <= limit:
                if j < lo:
                    chosen.insert(0, sents[j][1]); lo = j
                else:
                    chosen.append(sents[j][1]); hi = j
                size += len(sents[j][1]) + 1
                grew = True
        if not grew:
            break
    return " ".join(chosen)


def squad_examples(path, rng, max_per_paragraph=5):
    data = json.load(open(path, encoding="utf-8"))["data"]
    paragraphs = [(a["title"], p) for a in data for p in a["paragraphs"]]
    out = []
    for title, p in paragraphs:
        ctx = clean(p["context"])
        qas = p["qas"][:max_per_paragraph]
        for qa in qas:
            ans = qa["answers"][0]
            start = ctx.find(ans["text"])
            if start < 0:
                continue
            sent = next((s for st, s in reversed(sentences(ctx)) if st <= start), None)
            if not sent or len(sent) > 400 or ans["text"] not in sent:
                continue
            answer = sent.strip()
            if not answer.endswith((".", "!", "?")):
                answer += "."
            passage = trim_passage(ctx, start)
            q = qa["question"].strip()
            roll = rng.random()
            if roll < 0.08:                               # nothing relevant: say so honestly
                others = [trim_passage(clean(rng.choice(paragraphs)[1]["context"]), None, 400) for _ in range(2)]
                out.append(("squad-none", [("user", grounded_question(q, others)), ("assistant", rng.choice(NO_ANSWER_VARIANTS))]))
                continue
            passages = [passage]
            if roll < 0.55:                               # a distractor result to ignore
                other = rng.choice(paragraphs)
                if other[0] != title:
                    passages.append(trim_passage(clean(other[1]["context"]), None, 350))
                    rng.shuffle(passages)
            out.append(("squad", [("user", grounded_question(q, passages)), ("assistant", answer)]))
    return out


def dolly_examples(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            inst, ctx, resp = clean(r["instruction"]), clean(r.get("context") or ""), clean(r["response"])
            if not inst or not resp or len(resp) > 1800 or len(ctx) > 1800:
                continue
            cat = r.get("category", "")
            if ctx and cat in ("closed_qa", "information_extraction"):
                user = grounded_question(inst, [ctx])
            elif ctx:
                user = f"{inst}\n\n{ctx}"
            else:
                user = inst
            out.append((f"dolly-{cat}", [("user", user), ("assistant", resp)]))
    return out


def max_examples(rng, repeat=4, augment=2):
    out = []
    chat_files, _ = default_files(os.path.join(HERE, "data"))
    for path in chat_files:
        for conv in parse_conversations(open(path, encoding="utf-8").read()):
            for _ in range(repeat):
                out.append(("max", conv))
                for _ in range(augment):
                    out.append(("max", augment_conversation(conv, np.random.default_rng(rng.randrange(1 << 30)))))
    return out


def encode(tok, turns):
    """Token ids and loss-weight codes: 2 = Max's answer, 1 = everything else, 0 = role markers."""
    ids = tok.encode(f"<|system|>{SYSTEM_PROMPT}<|end|>")
    codes = [1] * len(ids)
    for role, text in turns:
        head = tok.encode(f"<|{role}|>")
        # Answers start with a space, so their first word is the same token as in the passage it
        # comes from (" Space", not "Space") — copying facts from search results gets much easier.
        body = tok.encode((" " + text) if role == "assistant" else text, allow_special=False) + [tok.special["<|end|>"]]
        ids += head + body
        codes += [0] * len(head) + [2 if role == "assistant" else 1] * len(body)
    return ids, codes


def save_split(path, encoded):
    lengths = [len(i) for i, _ in encoded]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    ids = np.concatenate([np.array(i, dtype=np.uint16) for i, _ in encoded])
    codes = np.concatenate([np.array(c, dtype=np.uint8) for _, c in encoded])
    np.savez_compressed(path, ids=ids, weights=codes, offsets=offsets)
    return int(offsets[-1])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the MaxGPT training corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dolly", default=DOLLY_URL, help="path or URL ('' to skip)")
    ap.add_argument("--squad", default=SQUAD_URL, help="path or URL ('' to skip)")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tokenizer", default="", help="reuse this tokenizer.json (to continue training a model)")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    cache = os.path.join(args.out, "downloads")

    examples = max_examples(rng)
    if args.squad:
        examples += squad_examples(fetch(args.squad, cache), rng)
    if args.dolly:
        examples += dolly_examples(fetch(args.dolly, cache))
    rng.shuffle(examples)
    kinds = {}
    for kind, _ in examples:
        kinds[kind] = kinds.get(kind, 0) + 1
    print("examples:", json.dumps(kinds, indent=1), flush=True)

    sample_text = "\n".join(t for _, turns in examples[:60000] for _, t in turns)[:30_000_000]
    print(f"training tokenizer (vocab {args.vocab}) on {len(sample_text):,} characters", flush=True)
    os.makedirs(args.out, exist_ok=True)
    if args.tokenizer:
        tok = Tokenizer.load(args.tokenizer)
    else:
        tok = Tokenizer.train(SYSTEM_PROMPT + "\n" + sample_text, vocab_size=args.vocab, verbose=True)
    tok.save(os.path.join(args.out, "tokenizer.json"))

    encoded = [encode(tok, turns) for _, turns in examples]
    n_val = min(1500, len(encoded) // 50)
    stats = {"examples": kinds, "vocab": tok.vocab_size,
             "val_tokens": save_split(os.path.join(args.out, "val.npz"), encoded[:n_val]),
             "train_tokens": save_split(os.path.join(args.out, "train.npz"), encoded[n_val:])}
    json.dump(stats, open(os.path.join(args.out, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
