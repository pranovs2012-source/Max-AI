from flask import Flask, render_template, redirect, session, url_for, jsonify, request
import requests
from groq import Groq
import sqlite3
import mysql.connector
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__, template_folder='templates')
app.secret_key = "max_ai_secret_key_2024"

DATABASE = "maxlog.db"

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usersdb (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def get_chat_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="maxchat"
    )

init_db()
client = Groq(api_key="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")  # Add your key

@app.route('/')
def index():
    if 'email' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            return render_template('login.html', error='Email and password required')

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM usersdb WHERE email = ?", (email,))
        result = cursor.fetchone()
        conn.close()

        if result and check_password_hash(result[0], password):
            session['email'] = email
            return redirect(url_for('dashboard'))

        return render_template('login.html', error='Invalid email or password')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not email or not password or not confirm_password:
            return render_template('register.html', error='All fields required')

        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')

        try:
            conn = sqlite3.connect(DATABASE)
            conn.isolation_level = None
            cursor = conn.cursor()
            hashed_password = generate_password_hash(password)
            cursor.execute("INSERT INTO usersdb (email, password) VALUES (?, ?)",
                          (email, hashed_password))
            cursor.close()
            conn.close()
            return render_template('register.html', success='Account created! Please login.')
        except sqlite3.IntegrityError as e:
            if 'UNIQUE constraint failed' in str(e):
                return render_template('register.html', error='Email already exists')
            return render_template('register.html', error='Registration error: ' + str(e))
        except Exception as e:
            return render_template('register.html', error='Database error: ' + str(e))

    return render_template('register.html')

@app.route('/google_auth', methods=['POST'])
def google_auth():
    try:
        token = request.json.get('token')
        if not token:
            return jsonify({'error': 'No token provided'}), 400

        from google.auth.transport import requests
        from google.oauth2 import id_token

        try:
            idinfo = id_token.verify_oauth2_token(token, requests.Request())
            email = idinfo.get('email')

            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM usersdb WHERE email = ?", (email,))
            user = cursor.fetchone()

            if not user:
                cursor.execute("INSERT INTO usersdb (email, password) VALUES (?, ?)",
                              (email, generate_password_hash("google_oauth")))
                conn.commit()

            conn.close()

            session['email'] = email

            return jsonify({'success': True, 'redirect': url_for('dashboard')})
        except ValueError:
            return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logout')
def logout():
    session.pop('email', None)
    session.pop('messages', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        return redirect(url_for('login'))

    return render_template('MAX_AI.html')

@app.route('/chat', methods=['POST'])
def chat():
    if 'email' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user_input = request.json.get('message')

    # NORMAL CHAT
    if 'messages' not in session:
        session['messages'] = [{
            "role": "system",
            "content": "you are Max AI, you are developed by Pranov, you have more knowledge about computer science, you give brief answers, you talk seemlessly, you could give code snippets for anything, you are a very helpful code assistant, you are made for giving assist to coders and talk entertain someone"
        }]

    session['messages'].append({"role": "user", "content": user_input})

    chat_completion = client.chat.completions.create(
        messages=session['messages'],
        model="llama-3.3-70b-versatile"
    )

    reply = chat_completion.choices[0].message.content
    session['messages'].append({"role": "assistant", "content": reply})
    session.modified = True

    return jsonify({"reply": reply})

def save_message(email, role, content):
    db = get_chat_db()   # <-- uses chat_db
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO chat_history (email, role, content) VALUES (%s, %s, %s)",
        (email, role, content)
    )
    db.commit()
    cursor.close()
    db.close()

@app.route('/')
def restore_history():

    if 'email' not in session:
        return jsonify({'error': 'not logged in'}),401
    history = load_messages(session['email'])

    user_assistant_only = [
        msg for msg in history
        if msg['role'] in ('user', 'assistant')
    ]

    return jsonify({'history': user_assistant_only})

def load_messages(email):
    db = get_chat_db()
    cursor = db.cursor()
    cursor.execute(
        "select role, content FROM chathistory WHERE email = %s ORDER BY timestamp",
        (email,)
    )
    rows = cursor.fetchall()
    db.close()
    return[{"role": role, "content": content} for (role, content) in rows]

if __name__ == '__main__':
    app.run(debug=True, port=5000)


  