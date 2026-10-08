"""Database layer tests: SQLite always, PostgreSQL too when TEST_DATABASE_URL is set.

    TEST_DATABASE_URL=postgresql://postgres@localhost:5432/postgres python -m unittest discover -s tests
"""
import importlib
import os
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_database(url=None, sqlite_path=None):
    for key in ("DATABASE_URL", "POSTGRES_URL", "SUPABASE_DB_URL", "VERCEL"):
        os.environ.pop(key, None)
    if url:
        os.environ["DATABASE_URL"] = url
    if sqlite_path:
        os.environ["MAX_AI_DB"] = sqlite_path
    import database
    return importlib.reload(database)


class DatabaseContract:
    """The same checks for every database kind."""

    def test_users(self):
        db = self.db
        email = f"{uuid.uuid4().hex}@test.dev"
        self.assertFalse(db.user_exists(email))
        self.assertTrue(db.create_user(email, "hash1"))
        self.assertFalse(db.create_user(email, "hash2"))       # duplicate email is refused
        self.assertEqual(db.get_password_hash(email), "hash1")
        self.assertTrue(db.user_exists(email))

    def test_messages(self):
        db = self.db
        email = f"{uuid.uuid4().hex}@test.dev"
        for i in range(5):
            db.save_message(email, "user" if i % 2 == 0 else "assistant", f"message {i} with 'quotes' and ? marks")
        everything = db.load_messages(email)
        self.assertEqual([m["content"][:9] for m in everything], [f"message {i}" for i in range(5)])
        recent = db.load_messages(email, limit=2)
        self.assertEqual([m["content"][:9] for m in recent], ["message 3", "message 4"])  # oldest first
        db.clear_messages(email)
        self.assertEqual(db.load_messages(email), [])


class SQLiteTest(DatabaseContract, unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = load_database(sqlite_path=os.path.join(self.tmp.name, "test.db"))
        self.assertEqual(self.db.KIND, "sqlite")
        self.db.init()

    def tearDown(self):
        os.environ.pop("MAX_AI_DB", None)
        self.tmp.cleanup()


@unittest.skipUnless(os.environ.get("TEST_DATABASE_URL"), "set TEST_DATABASE_URL to test PostgreSQL")
class PostgresTest(DatabaseContract, unittest.TestCase):
    def setUp(self):
        self.db = load_database(url=os.environ["TEST_DATABASE_URL"])
        self.assertEqual(self.db.KIND, "postgres")
        self.db.init()
        self.db.init()  # creating tables twice is fine

    def tearDown(self):
        os.environ.pop("DATABASE_URL", None)


class UrlTest(unittest.TestCase):
    def test_ssl_added_for_hosted_databases(self):
        db = load_database()
        self.assertEqual(db._postgres_url("postgresql://u:p@ep-x.neon.tech/neondb"),
                         "postgresql://u:p@ep-x.neon.tech/neondb?sslmode=require")
        self.assertEqual(db._postgres_url("postgresql://u:p@h/db?sslmode=disable"), "postgresql://u:p@h/db?sslmode=disable")
        self.assertEqual(db._postgres_url("postgresql://u@localhost:5432/db"), "postgresql://u@localhost:5432/db")


if __name__ == "__main__":
    unittest.main()
