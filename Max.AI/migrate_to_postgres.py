"""Copy users and chat history from the old SQLite file (maxlog.db) into Postgres.

    DATABASE_URL=postgresql://... python migrate_to_postgres.py [path/to/maxlog.db]

Safe to run more than once: users already in Postgres (and their messages) are skipped.
"""
import os
import sqlite3
import sys

if not os.environ.get("DATABASE_URL"):
    sys.exit("Set DATABASE_URL to your Neon or Supabase connection string first.")

import database  # noqa: E402  (reads DATABASE_URL)

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "maxlog.db")
src = sqlite3.connect(path)
database.init()
tables = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
users = src.execute("SELECT email, password FROM usersdb").fetchall() if "usersdb" in tables else []
new_users = {email for email, pw in users if database.create_user(email, pw)}
messages = 0
if "chat_history" in tables:
    with database.connect() as db:
        for email, role, content in src.execute("SELECT email, role, content FROM chat_history ORDER BY id"):
            if email in new_users:  # history of users copied before is already there
                db.execute("INSERT INTO chat_history (email, role, content) VALUES (?, ?, ?)", (email, role, content))
                messages += 1
print(f"Users: {len(new_users)} added, {len(users) - len(new_users)} already there. Messages copied: {messages}.")
