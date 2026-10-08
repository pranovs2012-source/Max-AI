# Max-AI 🤖

An AI coding assistant built with Flask, powered by **MaxGPT — Max's own language model**.
No OpenAI, Groq or any other external API: the model is written from scratch, trained on
Max's own data, and runs on your own machine.

---

## 🚀 Features
- **MaxGPT**: a GPT-style transformer written from scratch in Python + NumPy
  (own tokenizer, own autograd engine, own training loop, own weights)
- **Knowledge guard**: Max says honestly when a topic is outside what it has learned,
  instead of making things up, and logs the question so you can teach it
- Exact answers for arithmetic ("what is 7 * 8") from a safe built-in calculator
- Chat with memory of the recent conversation, saved per user in SQLite
- Code blocks with one-click copy, voice input, chat history restore / clear
- Email + password login and optional Google sign-in
- Optional **local GGUF backend**: run any open-weight model (Llama, Qwen, Mistral…) on your
  own computer through llama.cpp — still no API

---

## 🛠️ Tech Stack
- **Backend:** Flask, Python
- **AI model:** MaxGPT (NumPy transformer, trained from scratch)
- **Database:** SQLite (users and chat history)
- **Frontend:** HTML, CSS, JavaScript (Jinja2 templates)

---

## ⚡ Quick start
```bash
cd Max.AI
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

python -m maxgpt.train              # only if maxgpt/checkpoints/ is missing (~40-60 min on a laptop CPU)
python MAX_AI.py                    # open http://127.0.0.1:5000
```

The trained model lives in `Max.AI/maxgpt/checkpoints/` (`maxgpt.npz` + `tokenizer.json`).
The **Train MaxGPT** GitHub Action trains it on GitHub's servers and commits the weights
automatically whenever the model code or training data changes — or run it yourself from the
**Actions** tab → *Train MaxGPT* → *Run workflow*.

---

## ▲ Deploying on Vercel
The repository is ready for Vercel: `api/index.py` serves the Flask app, `vercel.json` routes
every request to it and the root `requirements.txt` lists what Vercel installs. Every push to
`main` redeploys automatically.

In the Vercel project settings → **Environment Variables**, add:
- `FLASK_SECRET_KEY` — any long random string (otherwise users get logged out between requests)

> Vercel's disk is read-only except `/tmp`, which is wiped when the app goes idle. On Vercel the
> app works on a copy of `maxlog.db` in `/tmp`, so **new accounts and chat history are temporary**.
> For permanent accounts, host the app somewhere with a persistent disk (Render, Railway, a VPS)
> or point `MAX_AI_DB` at a database file on persistent storage.

---

## 🧠 How MaxGPT works
| File | What it does |
|---|---|
| `maxgpt/autograd.py` | A small automatic-differentiation engine (backpropagation) on NumPy |
| `maxgpt/tokenizer.py` | Byte-level BPE tokenizer learned from the training text |
| `maxgpt/model.py` | Decoder-only transformer: token + position embeddings, causal self-attention, MLP, layer norm, weight tying, KV-cache generation |
| `maxgpt/optim.py` | AdamW optimizer, gradient clipping, warmup + cosine learning rate |
| `maxgpt/data.py` | Loads chat transcripts and text, learns only Max's replies, rewords questions for robustness |
| `maxgpt/guard.py` | Knowledge guard: checks whether a question is close to something Max learned |
| `maxgpt/calc.py` | Safe calculator for arithmetic questions |
| `maxgpt/train.py` | Training script with presets, validation and checkpoints |
| `maxgpt/chat.py` | Chat engine (used by the app) and a terminal chat: `python -m maxgpt.chat` |
| `llm_backends.py` | Picks the backend: `maxgpt` (default) or `gguf` |
| `tests/` | Gradient checks for every operation, tokenizer, data, guard, calculator and model tests |

Run the tests with `python -m unittest discover -s tests` (from the `Max.AI` folder).

### Model sizes
```bash
python -m maxgpt.train --preset small              # ~1.1M parameters (default)
python -m maxgpt.train --preset base --steps 8000  # ~3.5M parameters, better answers
python -m maxgpt.train --resume --steps 2000       # keep training the current model
```

### Teach Max new things
Add conversations to any `.txt` file in `Max.AI/maxgpt/data/chat/`:
```
===
User: What is a closure?
Max: A closure is a function that remembers variables from the scope where it was created...
```
Plain articles or notes can go in `Max.AI/maxgpt/data/text/`. Then retrain (or push to GitHub
and let the workflow retrain). Questions Max couldn't answer are saved in
`Max.AI/maxgpt/data/unanswered.txt` — a ready-made to-do list of what to teach next.
More (and more varied) data is the single best way to make MaxGPT smarter.

> **Honest expectations:** MaxGPT is tiny compared with models like Llama 3 70B. It answers the
> kinds of questions in its training data well and admits when it doesn't know something, but
> it can't reason about brand-new topics. For much stronger answers *without* an API, use the
> GGUF backend below.

---

## 🔌 Optional: a bigger local model (still no API)
```bash
pip install llama-cpp-python
# download a GGUF model, e.g. a small "Qwen2.5-Coder-1.5B-Instruct" or "Llama-3.2-3B-Instruct" Q4 file
export MAX_AI_BACKEND=gguf
export MAX_AI_GGUF_PATH=/path/to/model.gguf
python MAX_AI.py
```

---

## 🔑 Environment variables (all optional)
```bash
FLASK_SECRET_KEY=a_long_random_string    # keeps logins valid across restarts
GOOGLE_CLIENT_ID=your_google_client_id   # for Google sign-in
MAX_AI_BACKEND=maxgpt                    # or gguf
MAXGPT_TEMPERATURE=0.6                   # lower = more focused answers
MAX_AI_DB=/path/to/maxlog.db             # where users and chat history are stored
```
Never commit API keys or secrets to the repository.
