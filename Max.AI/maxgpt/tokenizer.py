"""Byte-level BPE tokenizer, trained from scratch on your own text.

Any UTF-8 text can be encoded (unknown characters fall back to raw bytes),
so the model never sees an "unknown token".
"""
import heapq
import json
import re
from collections import Counter, defaultdict

# Splits text into word-ish chunks before merging (similar in spirit to GPT-2).
PATTERN = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d{1,3}| ?[^\sA-Za-z\d]+|\s+(?!\S)|\s+")

SPECIAL_TOKENS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|pad|>"]


class Tokenizer:
    def __init__(self, merges=None):
        self.merges = {}  # (id_a, id_b) -> new id, in rank order
        self.vocab = {i: bytes([i]) for i in range(256)}
        for a, b in merges or []:
            self._add_merge(a, b)
        self._init_special()
        self._cache = {}

    def _add_merge(self, a, b):
        new_id = 256 + len(self.merges)
        self.merges[(a, b)] = new_id
        self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    def _init_special(self):
        base = 256 + len(self.merges)
        self.special = {tok: base + i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.special_inv = {v: k for k, v in self.special.items()}
        self._special_re = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(SPECIAL_TOKENS)

    @classmethod
    def train(cls, text, vocab_size=2048, verbose=False):
        """Learn merges until the vocabulary reaches `vocab_size`.

        Incremental: after each merge only the words containing that pair are updated, and a
        heap finds the most frequent pair, so even very large corpora train in seconds.
        """
        n_merges = max(0, vocab_size - 256 - len(SPECIAL_TOKENS))
        words = Counter(PATTERN.findall(text) if isinstance(text, str) else text)
        seqs = [list(w.encode("utf-8")) for w in words]
        freqs = list(words.values())
        counts, where = defaultdict(int), defaultdict(set)
        for i, ids in enumerate(seqs):
            for pair in zip(ids, ids[1:]):
                counts[pair] += freqs[i]
                where[pair].add(i)
        heap = [(-c, pair) for pair, c in counts.items()]
        heapq.heapify(heap)
        tok = cls()
        for step in range(n_merges):
            best = None
            while heap:
                neg, pair = heapq.heappop(heap)
                if counts.get(pair, 0) == -neg:
                    best = pair
                    break
            if best is None or counts[best] < 2:
                break
            tok._add_merge(*best)
            new_id = tok.merges[best]
            touched = set()
            for i in where.pop(best, ()):
                old = seqs[i]
                merged = _merge(old, best, new_id)
                if len(merged) == len(old):
                    continue
                f = freqs[i]
                for pair in zip(old, old[1:]):
                    counts[pair] -= f
                    touched.add(pair)
                for pair in zip(merged, merged[1:]):
                    counts[pair] += f
                    where[pair].add(i)
                    touched.add(pair)
                seqs[i] = merged
            counts.pop(best, None)
            for pair in touched:
                if counts.get(pair, 0) > 0:
                    heapq.heappush(heap, (-counts[pair], pair))
            if verbose and (step + 1) % 1000 == 0:
                print(f"  tokenizer: {step + 1}/{n_merges} merges", flush=True)
        tok._init_special()
        return tok

    def _encode_chunk(self, chunk):
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            # apply the earliest-learned merge available
            best = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            ids = _merge(ids, best, self.merges[best])
        if len(self._cache) < 100_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text, allow_special=True):
        out = []
        parts = self._special_re.split(text) if allow_special else [text]
        for part in parts:
            if not part:
                continue
            if allow_special and part in self.special:
                out.append(self.special[part])
                continue
            for chunk in PATTERN.findall(part):
                out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids):
        buf, out = b"", []
        for i in ids:
            if i in self.special_inv:
                out.append(buf.decode("utf-8", errors="replace"))
                buf = b""
                out.append(self.special_inv[i])
            else:
                buf += self.vocab.get(i, b"")
        out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "maxgpt-bpe", "merges": [list(p) for p in self.merges]}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls([tuple(p) for p in data["merges"]])


def _merge(ids, pair, new_id):
    out, i, a, b = [], 0, pair[0], pair[1]
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
