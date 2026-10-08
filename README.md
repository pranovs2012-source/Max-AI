# Max-AI 🤖

Max is an AI assistant created by **Pranov**, powered by **MaxGPT**, a transformer language
model built and trained for Max. It helps with programming, computer science, science,
geography, history and everyday questions.

---

## 🚀 Features
- **MaxGPT**: a GPT-style transformer written from scratch in Python + NumPy
  (own tokenizer, own autograd engine, own training loop, own weights)
- **Thinks before answering**: understands the question, picks a strategy, gathers facts,
  double-checks its own answer and shows every step ("🧠 Thought for 0.8s")
- **Live knowledge from Wikipedia**: summaries of any topic, follow-up questions, rankings read
  from Wikipedia tables ("top 10 fastest cars"), today's world news, word definitions (Wiktionary),
  questions in Hindi, Tamil and other languages (answered from that language's Wikipedia),
  knowledge cards with pictures and source links — like a search results page
- **Voice**: answers read aloud with an ElevenLabs voice (browser voice as a fallback); ask with
  the microphone and Max answers out loud
- General knowledge it was trained on: countries and capitals, planets, chemical elements,
  inventions, science, history, health, money, study tips, writing help and more
- **Knowledge guard**: Max says honestly when a topic is outside what it has learned,
  instead of making things up, and logs the question so you can teach it
- Exact answers for arithmetic ("what is 7 * 8") from a safe built-in calculator
- Chat with memory of the recent conversation, saved per user in SQLite
- Code blocks with one-click copy, voice input, chat history restore / clear
- Email + password login and optional Google sign-in
- Optional **local GGUF backend**: run any open-weight model (Llama, Qwen, Mistral…) on your
  own computer through llama.cpp

---

## 🛠️ Tech Stack
- **Backend:** Flask, Python
- **AI model:** MaxGPT (NumPy transformer, trained from scratch)
- **Database:** PostgreSQL — Neon or Supabase (users and chat history)
- **Knowledge:** Wikipedia, Wiktionary and Wikipedia's Current events portal (no key needed)
- **Voice:** ElevenLabs text-to-speech
- **Frontend:** HTML, CSS, JavaScript (Jinja2 templates)

---

## ⚡ Quick start
```bash
cd Max.AI
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

python -m maxgpt.knowledge          # build the world-facts training file
python -m maxgpt.train              # only if maxgpt/checkpoints/ is missing (~40-60 min on a laptop CPU)
python MAX_AI.py                    # open http://127.0.0.1:5000
```

The trained model lives in `Max.AI/maxgpt/checkpoints/` (`maxgpt.npz` + `tokenizer.json`).
The **Train MaxGPT** GitHub Action trains it on GitHub's servers and commits the weights
automatically whenever the model code or training data changes — or run it yourself from the
**Actions** tab → *Train MaxGPT* → *Run workflow*.

---

## 🐘 Database: PostgreSQL (Neon or Supabase)
Max stores accounts and chat history in Postgres. Create a free database and copy its
**connection string**:

- **Neon** — [neon.tech](https://neon.tech) → New project → *Connection string* (use the *pooled* one).
  Or in Vercel: *Storage* → *Create database* → Neon, which sets `DATABASE_URL` for you.
- **Supabase** — [supabase.com](https://supabase.com) → New project → *Connect* → *Transaction pooler*
  (port 6543) connection string.

Set it as `DATABASE_URL` (in Vercel: *Settings* → *Environment Variables*; locally: `export DATABASE_URL=...`).
Tables are created automatically on first start. To bring over accounts from the old SQLite file:
```bash
cd Max.AI
DATABASE_URL=postgresql://... python migrate_to_postgres.py      # copies maxlog.db users + history
```
Without `DATABASE_URL` the app falls back to a local SQLite file (on Vercel a temporary one), so
connect Postgres for permanent accounts.

---

## 🔊 Voice (ElevenLabs)
1. Create an API key at [elevenlabs.io](https://elevenlabs.io) → *Developers* → *API keys*.
2. Set `ELEVENLABS_API_KEY` in Vercel's environment variables.
3. Optional: `ELEVENLABS_VOICE_ID` — any voice from your ElevenLabs Voice Library (the default is
   the premade British voice "Daniel", calm and precise like an AI butler), and `ELEVENLABS_MODEL`
   (default `eleven_flash_v2_5`, fast and multilingual).

Turn on **🔈 Voice** in the chat header to hear every answer, press **🔊 Listen** under any answer,
or ask with the 🎤 microphone and Max answers out loud. Without a key, the browser's own voice is used.

---

## ▲ Deploying on Vercel
The repository is ready for Vercel: `api/index.py` serves the Flask app, `vercel.json` routes
every request to it (with up to 30 s per request for web lookups) and the root `requirements.txt`
lists what Vercel installs. Every push to `main` redeploys automatically.

In the Vercel project settings → **Environment Variables**, add:
- `FLASK_SECRET_KEY` — any long random string (keeps users logged in)
- `DATABASE_URL` — your Neon / Supabase Postgres connection string
- `ELEVENLABS_API_KEY` — for Max's voice (optional)

---

## 🧠 How MaxGPT works
| File | What it does |
|---|---|
| `maxgpt/autograd.py` | A small automatic-differentiation engine (backpropagation) on NumPy |
| `maxgpt/tokenizer.py` | Byte-level BPE tokenizer learned from the training text |
| `maxgpt/model.py` | Decoder-only transformer: token + position embeddings, causal self-attention, MLP, layer norm, weight tying, KV-cache generation |
| `maxgpt/optim.py` | AdamW optimizer, gradient clipping, warmup + cosine learning rate |
| `maxgpt/data.py` | Loads chat transcripts and text, learns only Max's replies, rewords questions for robustness |
| `maxgpt/knowledge.py` | Fact tables (countries, elements, planets, inventions…) turned into training conversations |
| `maxgpt/think.py` | Max's reasoning loop: understands the question, picks a strategy (recall, look up, rank, news, define, calculate), gathers facts, self-checks and records every step |
| `maxgpt/web.py` | Wikipedia / Wiktionary client: search, article summaries, ranking tables, news, definitions, other languages |
| `database.py` | PostgreSQL (Neon / Supabase) storage for users and chat history |
| `voice.py` | ElevenLabs text-to-speech |
| `maxgpt/guard.py` | Knowledge guard: checks whether a question is close to something Max learned |
| `maxgpt/calc.py` | Safe calculator for arithmetic questions |
| `maxgpt/train.py` | Training script with presets, validation and checkpoints |
| `maxgpt/chat.py` | Chat engine (used by the app) and a terminal chat: `python -m maxgpt.chat` |
| `llm_backends.py` | Picks the backend: `maxgpt` (default) or `gguf` |
| `tests/` | Gradient checks for every operation, tokenizer, data, guard, calculator and model tests |

Run the tests with `python -m unittest discover -s tests` (from the `Max.AI` folder); add
`TEST_DATABASE_URL=postgresql://...` to test Postgres too. `python tools/live_check.py` asks Max real
questions against live Wikipedia. Both run on GitHub in the **Max AI checks** workflow.

### Model sizes
```bash
python -m maxgpt.train --preset small              # ~1.1M parameters, quick
python -m maxgpt.train --preset base --steps 8000  # ~3.5M parameters (used by the GitHub workflow)
python -m maxgpt.train --resume --steps 2000       # keep training the current model
```

### Teach Max new things
Add conversations to any `.txt` file in `Max.AI/maxgpt/data/chat/`:
```
===
User: What is a closure?
Max: A closure is a function that remembers variables from the scope where it was created...
```
Plain articles or notes can go in `Max.AI/maxgpt/data/text/`, and new facts can be added to the
tables in `maxgpt/knowledge.py`. Then retrain (or push to GitHub and let the workflow retrain).
Questions Max couldn't answer are saved in `Max.AI/maxgpt/data/unanswered.txt` — a ready-made
to-do list of what to teach next. More (and more varied) data is the single best way to make
MaxGPT smarter.

> **Honest expectations:** MaxGPT is small compared with models like Llama 3 70B. It answers the
> kinds of questions in its training data well and admits when it doesn't know something, but
> it can't reason about brand-new topics. For much stronger answers, use the GGUF backend below.

---

## 🔌 Optional: a bigger local model
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
DATABASE_URL=postgresql://...             # Neon / Supabase Postgres (users and chat history)
ELEVENLABS_API_KEY=...                   # Max's voice
ELEVENLABS_VOICE_ID=onwK4e9ZLuTAKqWW03F9 # optional: any ElevenLabs voice
MAX_AI_WEB=1                             # 0 = answer only from the trained model, no Wikipedia
```
Never commit API keys or secrets to the repository.
