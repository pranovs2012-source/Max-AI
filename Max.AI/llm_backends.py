"""Language-model backends for Max AI. No external API is used.

Choose one with the MAX_AI_BACKEND environment variable:

  maxgpt (default)  Max's own model, trained from scratch with `python -m maxgpt.train`.
                    Needs only NumPy and runs on any CPU.
  gguf              Any open-weight model in GGUF format (Llama, Qwen, Mistral, ...) running
                    locally through llama.cpp. Much smarter, still no API.
                    pip install llama-cpp-python, then set MAX_AI_GGUF_PATH=/path/to/model.gguf
"""
import os

SYSTEM_PROMPT = (
    "You are Max AI, developed by Pranov. You know a lot about computer science, give brief answers, "
    "talk naturally and can give code snippets for anything. You are a helpful assistant for coders."
)


class MaxGPTBackend:
    name = "MaxGPT"

    def __init__(self):
        from maxgpt.chat import DEFAULT_DIR, MaxGPTEngine
        self.engine = MaxGPTEngine(os.environ.get("MAXGPT_CHECKPOINT", DEFAULT_DIR),
                                   temperature=float(os.environ.get("MAXGPT_TEMPERATURE", "0.6")))

    def reply(self, history):
        # MaxGPT was trained with its own short system prompt, so we keep that one.
        return self.engine.reply(history) or "Hmm, I'm not sure. Could you rephrase that?"


class GGUFBackend:
    name = "Local GGUF model"

    def __init__(self):
        from llama_cpp import Llama  # optional dependency
        path = os.environ.get("MAX_AI_GGUF_PATH")
        if not path or not os.path.exists(path):
            raise FileNotFoundError("Set MAX_AI_GGUF_PATH to a downloaded .gguf model file")
        self.llm = Llama(model_path=path, n_ctx=int(os.environ.get("MAX_AI_GGUF_CTX", "4096")), verbose=False)

    def reply(self, history):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [{"role": role, "content": text} for role, text in history]
        out = self.llm.create_chat_completion(messages=messages, max_tokens=1024, temperature=0.6)
        return out["choices"][0]["message"]["content"].strip()


BACKENDS = {"maxgpt": MaxGPTBackend, "gguf": GGUFBackend}
_backend, _error = None, None


def get_backend():
    """Load the configured backend once. Returns (backend or None, error message or None)."""
    global _backend, _error
    if _backend is None and _error is None:
        choice = os.environ.get("MAX_AI_BACKEND", "maxgpt").lower()
        try:
            _backend = BACKENDS[choice]()
        except KeyError:
            _error = f"Unknown MAX_AI_BACKEND '{choice}'. Use one of: {', '.join(BACKENDS)}."
        except Exception as e:  # missing checkpoint, missing package, bad path...
            _error = f"{choice} backend is not ready: {e}"
    return _backend, _error
