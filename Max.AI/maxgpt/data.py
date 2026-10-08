"""Training data: chat transcripts and plain text, turned into token windows.

Chat file format (UTF-8 .txt in data/chat/), conversations separated by a line `===`:

    ===
    User: What is recursion?
    Max: Recursion is when a function calls itself ...
    User: Show me an example
    Max: ```python
    def fact(n):
        return 1 if n <= 1 else n * fact(n - 1)
    ```

A turn continues until the next line starting with `User:` or `Max:`.
Plain-text files in data/text/ are used as-is (every token is learned).
"""
import glob
import os

import numpy as np

SYSTEM_PROMPT = "You are Max AI, a friendly coding assistant made by Pranov. Give brief, correct answers and code when useful."


def parse_conversations(text):
    convs = []
    for block in text.split("\n===\n"):
        block = block.strip()
        if block.startswith("==="):
            block = block[3:]
        turns, role, buf = [], None, []
        for line in block.strip("\n").split("\n"):
            if line.startswith("User:") or line.startswith("Max:"):
                if role:
                    turns.append((role, "\n".join(buf).strip()))
                role = "user" if line.startswith("User:") else "assistant"
                buf = [line.split(":", 1)[1].strip()]
            elif role:
                buf.append(line)
        if role:
            turns.append((role, "\n".join(buf).strip()))
        if any(r == "assistant" for r, _ in turns):
            convs.append(turns)
    return convs


def format_prompt(history, system=SYSTEM_PROMPT):
    """history: list of (role, text). Returns the prompt string, ready for Max's reply."""
    s = f"<|system|>{system}<|end|>"
    for role, text in history:
        s += f"<|{role}|>{text}<|end|>"
    return s + "<|assistant|>"


def encode_conversation(tok, turns, system=SYSTEM_PROMPT):
    """Token ids plus a mask that is True only where Max's words should be learned."""
    ids = tok.encode(f"<|system|>{system}<|end|>")
    mask = [False] * len(ids)
    for role, text in turns:
        head = tok.encode(f"<|{role}|>")
        body = tok.encode(text, allow_special=False) + [tok.special["<|end|>"]]
        ids += head + body
        mask += [False] * len(head) + [role == "assistant"] * len(body)
    return ids, mask


_PREFIXES = ["", "", "", "Can you tell me: ", "Please explain: ", "Quick question: ", "Hey Max, ", "Max, ", "I need help. "]


def rephrase(text, rng):
    """Cheap paraphrases of a user question so a small model copes with different wording."""
    t = text.strip()
    choice = rng.integers(0, 5)
    if choice == 1:
        t = t.lower()
    elif choice == 2:
        t = t.rstrip("?!. ")
    elif choice == 3:
        t = t[:1].lower() + t[1:]
        t = _PREFIXES[rng.integers(3, len(_PREFIXES))] + t
    elif choice == 4:
        t = t.rstrip("?!. ") + rng.choice(["?", "??", " pls", " please", ""])
    return t


def augment_conversation(turns, rng):
    return [(role, rephrase(text, rng) if role == "user" else text) for role, text in turns]


def windows(ids, mask, size):
    """Split a long sequence into overlapping windows of at most `size` tokens."""
    if len(ids) <= size:
        return [(ids, mask)]
    step = size // 2
    return [(ids[i:i + size], mask[i:i + size]) for i in range(0, len(ids) - size // 2, step)]


class Dataset:
    def __init__(self, tok, chat_files=(), text_files=(), n_ctx=256, seed=0, augment=3):
        """augment: how many extra reworded copies of each conversation to add."""
        self.tok, self.n_ctx = tok, n_ctx
        self.pad = tok.special["<|pad|>"]
        rng = np.random.default_rng(seed)
        examples = []
        for path in chat_files:
            with open(path, encoding="utf-8") as f:
                for conv in parse_conversations(f.read()):
                    for variant in [conv] + [augment_conversation(conv, rng) for _ in range(augment)]:
                        ids, mask = encode_conversation(tok, variant)
                        examples += windows(ids, mask, n_ctx + 1)
        for path in text_files:
            with open(path, encoding="utf-8") as f:
                ids = tok.encode(f.read(), allow_special=False)
            for i in range(0, max(1, len(ids) - 1), n_ctx):
                chunk = ids[i:i + n_ctx + 1]
                if len(chunk) > 8:
                    examples.append((chunk, [True] * len(chunk)))
        examples = [e for e in examples if any(e[1][1:])]
        if not examples:
            raise ValueError("no training examples found — check your data files")
        rng.shuffle(examples)
        n_val = max(1, len(examples) // 20) if len(examples) > 20 else 0
        self.val, self.train = examples[:n_val], examples[n_val:]
        self.rng = rng

    def batch(self, size, split="train"):
        pool = self.train if split == "train" else (self.val or self.train)
        picks = [pool[i] for i in self.rng.integers(0, len(pool), size)]
        T = max(len(ids) for ids, _ in picks) - 1
        x = np.full((size, T), self.pad, dtype=np.int64)
        y = np.full((size, T), -1, dtype=np.int64)
        for b, (ids, mask) in enumerate(picks):
            n = len(ids) - 1
            x[b, :n] = ids[:-1]
            tgt = np.array(ids[1:])
            tgt[~np.array(mask[1:], dtype=bool)] = -1
            y[b, :n] = tgt
        return x, y


def default_files(data_dir):
    chat = sorted(glob.glob(os.path.join(data_dir, "chat", "*.txt")))
    text = sorted(glob.glob(os.path.join(data_dir, "text", "*.txt")))
    return chat, text
