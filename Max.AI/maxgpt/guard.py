"""Knowledge guard: does MaxGPT know about this topic?

A small model answers questions similar to its training data well, but produces
nonsense on topics it never saw. The guard compares a question with every
question in the training data (TF-IDF cosine similarity). If nothing is close
enough, Max says so honestly instead of guessing, and the question is logged to
data/unanswered.txt so you can teach Max the answer and retrain.
"""
import math
import os
import re
from collections import Counter
from datetime import datetime

from .data import parse_conversations

_WORD = re.compile(r"[a-z0-9_#+.]+")
# Words that say nothing about the topic.
_STOP = set("""a an the is are was were be to of in on for and or with my your i you me it this that
what how do does can could would should please pls tell explain about give show make write max hey hi
quick question need help some any use using way get who where when why which ok okay cool nice great yes no
yeah yep lol thanks thank bye good morning night sure alright hmm wow example examples more another again
instead also simpler shorter detail details same other one""".split())


# Different spellings of the same topic.
_SYNONYMS = {"js": "javascript", "py": "python", "dict": "dictionary", "dicts": "dictionary",
             "dictionaries": "dictionary", "lists": "list", "arrays": "array", "funcs": "function",
             "func": "function", "functions": "function", "repo": "repository", "db": "database",
             "errors": "error", "bug": "error", "bugs": "error", "html5": "html", "css3": "css"}


def _terms(text):
    words = (w.strip(".") for w in _WORD.findall(text.lower()))
    return [_SYNONYMS.get(w, w) for w in words if w and w not in _STOP]


class KnowledgeGuard:
    def __init__(self, chat_files, threshold=0.5, log_path=None):
        self.threshold, self.log_path = threshold, log_path
        questions = []
        for path in chat_files:
            with open(path, encoding="utf-8") as f:
                for conv in parse_conversations(f.read()):
                    questions += [text for role, text in conv if role == "user"]
        docs = [Counter(_terms(q)) for q in questions]
        df = Counter(t for d in docs for t in d)
        n = len(docs) or 1
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        # words never seen in training weigh more than any known word: they signal a new topic
        self.unknown_idf = 1.5 * (math.log(1 + n) + 1)
        self.vecs = [self._vec(d) for d in docs]
        self.questions = questions

    def _vec(self, counts):
        v = {t: c * self.idf.get(t, self.unknown_idf) for t, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def score(self, question):
        """Best cosine similarity (0..1) to any training question, and that question."""
        terms = _terms(question)
        if not terms:  # greetings, "thanks", "ok"... let the model answer
            return 1.0, None
        q = self._vec(Counter(terms))
        if not any(t in self.idf for t in q):
            return 0.0, None
        best, best_q = 0.0, None
        for vec, text in zip(self.vecs, self.questions):
            s = sum(w * vec.get(t, 0.0) for t, w in q.items())
            if s > best:
                best, best_q = s, text
        return best, best_q

    def knows(self, question, previous=None):
        """True if the question (or, for short follow-ups, question + previous one) is familiar."""
        if self.score(question)[0] >= self.threshold:
            return True
        return bool(previous) and self.score(previous + " " + question)[0] >= self.threshold

    def log_unknown(self, question):
        if not self.log_path:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now():%Y-%m-%d %H:%M}\t{question.replace(chr(10), ' ')}\n")
        except OSError:
            pass
