"""Max AI database: PostgreSQL (Neon, Supabase or any Postgres) for users and chat history.

Set DATABASE_URL to your Postgres connection string, for example

    Neon:      postgresql://user:password@ep-xxx-pooler.region.aws.neon.tech/neondb?sslmode=require
    Supabase:  postgresql://postgres.xxxx:password@aws-0-region.pooler.supabase.com:6543/postgres

Vercel's Neon and Supabase integrations set DATABASE_URL / POSTGRES_URL automatically.
Without one (local development, or before you connect a database) a SQLite file is used so the
app still starts; on Vercel that file is temporary, so connect Postgres for permanent accounts.
"""
import os
import sqlite3
from contextlib import contextmanager

HERE = os.path.dirname(os.path.abspath(__file__))
URL = next((os.environ[k] for k in ("DATABASE_URL", "POSTGRES_URL", "SUPABASE_DB_URL")
            if os.environ.get(k)), None)
SQLITE_PATH = os.environ.get("MAX_AI_DB") or os.path.join(HERE, "maxlog.db")
if os.environ.get("VERCEL") and not os.environ.get("MAX_AI_DB"):
    # Until DATABASE_URL is set, keep the site working on Vercel's temporary disk (wiped when idle).
    SQLITE_PATH = "/tmp/maxlog.db"

if URL:
    import psycopg
    from psycopg.errors import UniqueViolation
    KIND = "postgres"
    IntegrityError = (UniqueViolation, sqlite3.IntegrityError)
else:
    KIND = "sqlite"
    IntegrityError = (sqlite3.IntegrityError,)


def _postgres_url(url):
    # Hosted Postgres needs TLS; add it when the URL doesn't say otherwise.
    if "sslmode=" not in url and not any(h in url for h in ("@localhost", "@127.0.0.1", "@postgres:")):
        url += ("&" if "?" in url else "?") + "sslmode=require"
    return url


@contextmanager
def connect():
    """A connection that commits on success. Use `?` placeholders in SQL for both databases."""
    if KIND == "postgres":
        conn = psycopg.connect(_postgres_url(URL), connect_timeout=10)
    else:
        conn = sqlite3.connect(SQLITE_PATH)
    try:
        yield _Conn(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class _Conn:
    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql, params=()):
        if KIND == "postgres":
            sql = sql.replace("?", "%s")
            cur = self.conn.cursor()
            cur.execute(sql, params)
            return cur
        return self.conn.execute(sql, params)


def init():
    with connect() as db:
        if KIND == "postgres":
            db.execute("""CREATE TABLE IF NOT EXISTS usersdb (
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now())""")
            db.execute("""CREATE TABLE IF NOT EXISTS chat_history (
                id BIGSERIAL PRIMARY KEY,
                email TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT now())""")
            db.execute("CREATE INDEX IF NOT EXISTS chat_history_email_id ON chat_history (email, id)")
        else:
            db.execute("""CREATE TABLE IF NOT EXISTS usersdb (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")


def get_password_hash(email):
    with connect() as db:
        row = db.execute("SELECT password FROM usersdb WHERE email = ?", (email,)).fetchone()
    return row[0] if row else None


def user_exists(email):
    return get_password_hash(email) is not None


def create_user(email, password_hash):
    """Returns False if the email is already registered."""
    try:
        with connect() as db:
            db.execute("INSERT INTO usersdb (email, password) VALUES (?, ?)", (email, password_hash))
        return True
    except IntegrityError:
        return False


def save_message(email, role, content):
    with connect() as db:
        db.execute("INSERT INTO chat_history (email, role, content) VALUES (?, ?, ?)", (email, role, content))


def load_messages(email, limit=None):
    with connect() as db:
        if limit:
            rows = db.execute("SELECT role, content FROM (SELECT id, role, content FROM chat_history "
                              "WHERE email = ? ORDER BY id DESC LIMIT ?) recent ORDER BY id", (email, limit)).fetchall()
        else:
            rows = db.execute("SELECT role, content FROM chat_history WHERE email = ? ORDER BY id", (email,)).fetchall()
    return [{"role": r, "content": c} for r, c in rows]


def clear_messages(email):
    with connect() as db:
        db.execute("DELETE FROM chat_history WHERE email = ?", (email,))
