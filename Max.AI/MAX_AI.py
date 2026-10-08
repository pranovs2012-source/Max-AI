"""Max AI — Flask chat app powered by MaxGPT, with live knowledge, a voice and a Postgres database."""
import os
import secrets
import time

from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

import database
import voice
from llm_backends import get_backend

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(HERE, "Templates" if os.path.isdir(os.path.join(HERE, "Templates")) else "templates")
HISTORY_TURNS = 10  # how many recent messages are sent to the model as context

app = Flask(__name__, template_folder=TEMPLATE_DIR)
# Set FLASK_SECRET_KEY in production so logins survive restarts.
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)

_db_error, _db_checked = None, 0.0


def _init_database():
    global _db_error, _db_checked
    _db_checked = time.time()
    try:
        database.init()
        _db_error = None
    except Exception as e:  # wrong DATABASE_URL, database waking up, network...
        _db_error = f"Database is not reachable: {e}"


_init_database()


@app.before_request
def require_database():
    if _db_error and time.time() - _db_checked > 10:
        _init_database()          # retry: hosted databases can be waking up from sleep
    if _db_error and request.endpoint not in ("model_info", "static"):
        if request.endpoint in ("login", "register"):
            return render_template(f"{request.endpoint}.html", error=_db_error), 503
        return jsonify({"error": _db_error}), 503


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
        stored = database.get_password_hash(email)
        if stored and check_password_hash(stored, password):
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
        if database.create_user(email, generate_password_hash(password)):
            return render_template("register.html", success="Account created! Please login.")
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
    if not database.user_exists(email):
        # random password: Google users can't log in with a guessable one
        database.create_user(email, generate_password_hash(secrets.token_urlsafe(32)))
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
    history = [(m["role"], m["content"]) for m in database.load_messages(email, HISTORY_TURNS)]
    history.append(("user", user_input))
    try:
        result = backend.answer(history)
    except Exception as e:
        return jsonify({"reply": f"⚠️ The model failed to answer: {e}"}), 500
    database.save_message(email, "user", user_input)
    database.save_message(email, "assistant", result["reply"])
    return jsonify(dict(result, model=backend.name))


@app.route("/tts", methods=["POST"])
def tts():
    """Max's voice (ElevenLabs). 404 tells the page to use the browser's own speech instead."""
    if "email" not in session:
        return jsonify({"error": "Not logged in"}), 401
    if not voice.enabled():
        return jsonify({"error": "ElevenLabs voice is not configured"}), 404
    text = ((request.json or {}).get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text"}), 400
    try:
        audio = voice.synthesize(text)
    except voice.VoiceError as e:
        return jsonify({"error": str(e)}), 502
    return Response(audio, mimetype="audio/mpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.route("/restore_history")
def restore_history():
    if "email" not in session:
        return jsonify({"error": "not logged in"}), 401
    return jsonify({"history": database.load_messages(session["email"])})


@app.route("/clear_history", methods=["POST"])
def clear_history():
    if "email" not in session:
        return jsonify({"error": "not logged in"}), 401
    database.clear_messages(session["email"])
    return jsonify({"ok": True})


@app.route("/model_info")
def model_info():
    backend, error = get_backend()
    return jsonify({"ready": backend is not None, "model": backend.name if backend else None, "error": error,
                    "voice": "elevenlabs" if voice.enabled() else "browser", "database": database.KIND,
                    "database_error": _db_error})


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=int(os.environ.get("PORT", "5000")))
