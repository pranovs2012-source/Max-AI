"""Max AI — Flask chat app powered by MaxGPT, Max's own language model (no external API)."""
import os
import secrets
import shutil
import sqlite3

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from llm_backends import get_backend

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(HERE, "Templates" if os.path.isdir(os.path.join(HERE, "Templates")) else "templates")
DATABASE = os.environ.get("MAX_AI_DB") or os.path.join(HERE, "maxlog.db")
if os.environ.get("VERCEL") and not os.environ.get("MAX_AI_DB"):
    # Vercel's file system is read-only except /tmp, and /tmp is wiped when the function
    # goes cold. Work on a copy there; for permanent accounts use a hosted database.
    tmp_db = "/tmp/maxlog.db"
    if not os.path.exists(tmp_db) and os.path.exists(DATABASE):
        shutil.copy(DATABASE, tmp_db)
    DATABASE = tmp_db
HISTORY_TURNS = 10  # how many recent messages are sent to the model as context

app = Flask(__name__, template_folder=TEMPLATE_DIR)
# Set FLASK_SECRET_KEY in production so logins survive restarts.
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)


def db():
    return sqlite3.connect(DATABASE)


def init_db():
    with db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS usersdb (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")


def save_message(email, role, content):
    with db() as conn:
        conn.execute("INSERT INTO chat_history (email, role, content) VALUES (?, ?, ?)", (email, role, content))


def load_messages(email, limit=None):
    with db() as conn:
        rows = conn.execute("SELECT role, content FROM chat_history WHERE email = ? ORDER BY id", (email,)).fetchall()
    rows = [{"role": r, "content": c} for r, c in rows]
    return rows[-limit:] if limit else rows


init_db()


@app.route("/")
def index():
    return redirect(url_for("dashboard" if "email" in session else "login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        if not email or not password:
            return render_template("login.html", error="Email and password required")
        with db() as conn:
            row = conn.execute("SELECT password FROM usersdb WHERE email = ?", (email,)).fetchone()
        if row and check_password_hash(row[0], password):
            session["email"] = email
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid email or password")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        if not email or not password or not confirm_password:
            return render_template("register.html", error="All fields required")
        if password != confirm_password:
            return render_template("register.html", error="Passwords do not match")
        try:
            with db() as conn:
                conn.execute("INSERT INTO usersdb (email, password) VALUES (?, ?)",
                             (email, generate_password_hash(password)))
            return render_template("register.html", success="Account created! Please login.")
        except sqlite3.IntegrityError:
            return render_template("register.html", error="Email already exists")
    return render_template("register.html")


@app.route("/google_auth", methods=["POST"])
def google_auth():
    token = (request.json or {}).get("token")
    if not token:
        return jsonify({"error": "No token provided"}), 400
    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token
        idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), os.environ.get("GOOGLE_CLIENT_ID"))
    except ValueError:
        return jsonify({"error": "Invalid token"}), 401
    except ImportError:
        return jsonify({"error": "Google login needs: pip install google-auth"}), 500
    email = idinfo.get("email")
    with db() as conn:
        if not conn.execute("SELECT id FROM usersdb WHERE email = ?", (email,)).fetchone():
            # random password: Google users can't log in with a guessable one
            conn.execute("INSERT INTO usersdb (email, password) VALUES (?, ?)",
                         (email, generate_password_hash(secrets.token_urlsafe(32))))
    session["email"] = email
    return jsonify({"success": True, "redirect": url_for("dashboard")})


@app.route("/logout")
def logout():
    session.pop("email", None)
    return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    if "email" not in session:
        return redirect(url_for("login"))
    return render_template("MAX_AI.html")


@app.route("/chat", methods=["POST"])
def chat():
    if "email" not in session:
        return jsonify({"error": "Not logged in"}), 401
    user_input = ((request.json or {}).get("message") or "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    backend, error = get_backend()
    if backend is None:
        return jsonify({"reply": f"⚠️ {error}"}), 503

    email = session["email"]
    history = [(m["role"], m["content"]) for m in load_messages(email, HISTORY_TURNS)]
    history.append(("user", user_input))
    try:
        reply = backend.reply(history)
    except Exception as e:
        return jsonify({"reply": f"⚠️ The model failed to answer: {e}"}), 500
    save_message(email, "user", user_input)
    save_message(email, "assistant", reply)
    return jsonify({"reply": reply, "model": backend.name})


@app.route("/restore_history")
def restore_history():
    if "email" not in session:
        return jsonify({"error": "not logged in"}), 401
    return jsonify({"history": load_messages(session["email"])})


@app.route("/clear_history", methods=["POST"])
def clear_history():
    if "email" not in session:
        return jsonify({"error": "not logged in"}), 401
    with db() as conn:
        conn.execute("DELETE FROM chat_history WHERE email = ?", (session["email"],))
    return jsonify({"ok": True})


@app.route("/model_info")
def model_info():
    backend, error = get_backend()
    return jsonify({"ready": backend is not None, "model": backend.name if backend else None, "error": error})


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=int(os.environ.get("PORT", "5000")))
