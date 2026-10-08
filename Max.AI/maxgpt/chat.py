"""Chat with a trained MaxGPT model.

    python -m maxgpt.chat                 # interactive terminal chat
    python -m maxgpt.chat --temp 0.5      # more focused answers
"""
import argparse
import os
import threading

from .data import SYSTEM_PROMPT, default_files, format_prompt
from .guard import KnowledgeGuard
from .model import GPT
from .tokenizer import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, "checkpoints")
DATA_DIR = os.path.join(HERE, "data")

UNKNOWN_REPLY = ("I haven't learned about that yet, so I'd rather not guess. I'm a small model that knows "
                 "Python, JavaScript, HTML, CSS, SQL, Git and computer science basics. Try asking about one of "
                 "those, or rephrase your question with more detail.")


class MaxGPTEngine:
    """Loads a checkpoint once and answers chat histories. Thread-safe."""

    def __init__(self, checkpoint_dir=DEFAULT_DIR, temperature=0.7, top_k=40, top_p=0.95, max_new_tokens=200,
                 guard=True, guard_threshold=0.5):
        model_path = os.path.join(checkpoint_dir, "maxgpt.npz")
        tok_path = os.path.join(checkpoint_dir, "tokenizer.json")
        if not (os.path.exists(model_path) and os.path.exists(tok_path)):
            raise FileNotFoundError(
                f"No trained MaxGPT model in {checkpoint_dir}. Train one first:  python -m maxgpt.train")
        self.model = GPT.load(model_path)
        self.tok = Tokenizer.load(tok_path)
        self.end_id = self.tok.special["<|end|>"]
        self.settings = dict(temperature=temperature, top_k=top_k, top_p=top_p, max_new_tokens=max_new_tokens)
        self._lock = threading.Lock()
        chat_files, _ = default_files(DATA_DIR)
        self.guard = KnowledgeGuard(chat_files, guard_threshold, os.path.join(DATA_DIR, "unanswered.txt")) \
            if guard and chat_files else None

    def _prompt_ids(self, history, system, max_new):
        """Encode the conversation, dropping the oldest turns if it doesn't fit."""
        budget = self.model.config.n_ctx - max_new
        turns = list(history)
        while True:
            ids = self.tok.encode(format_prompt(turns, system))
            if len(ids) <= budget or len(turns) <= 1:
                return ids[-budget:]
            turns = turns[2:] if len(turns) > 2 else turns[1:]

    def stream(self, history, system=SYSTEM_PROMPT, **overrides):
        """history: list of (role, text), role 'user' or 'assistant'. Yields text pieces."""
        opts = {**self.settings, **overrides}
        asked = [text for role, text in history if role == "user"]
        question = asked[-1] if asked else ""
        previous = asked[-2] if len(asked) > 1 else None
        if self.guard and not self.guard.knows(question, previous):
            self.guard.log_unknown(question)
            yield UNKNOWN_REPLY
            return
        max_new = min(opts["max_new_tokens"], self.model.config.n_ctx // 2)
        ids = self._prompt_ids(history, system, max_new)
        with self._lock:
            pending = []
            for tid in self.model.generate(ids, max_new, opts["temperature"], opts["top_k"],
                                           opts["top_p"], stop_ids={self.end_id}):
                pending.append(tid)
                text = self.tok.decode(pending)
                if "�" not in text:  # wait until multi-byte characters are complete
                    yield text
                    pending = []
            if pending:
                yield self.tok.decode(pending)

    def reply(self, history, system=SYSTEM_PROMPT, **overrides):
        return "".join(self.stream(history, system, **overrides)).strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Chat with MaxGPT")
    ap.add_argument("--checkpoint", default=DEFAULT_DIR)
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args(argv)
    engine = MaxGPTEngine(args.checkpoint, temperature=args.temp)
    print(f"MaxGPT ({engine.model.num_params():,} parameters). Type 'exit' to quit, 'reset' to clear history.\n")
    history = []
    while True:
        try:
            msg = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if msg.lower() in ("exit", "quit"):
            break
        if msg.lower() == "reset":
            history = []
            continue
        if not msg:
            continue
        history.append(("user", msg))
        print("max > ", end="", flush=True)
        parts = []
        for piece in engine.stream(history):
            parts.append(piece)
            print(piece, end="", flush=True)
        print("\n")
        history.append(("assistant", "".join(parts).strip()))


if __name__ == "__main__":
    main()
