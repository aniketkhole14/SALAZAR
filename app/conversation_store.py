"""SQLite-backed persistent conversation storage for SALAZAR."""

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("SALAZAR_ROOT", Path(__file__).resolve().parents[1])).resolve()
DB_PATH = ROOT / "data" / "database" / "conversations.db"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class ConversationStore:
    def __init__(self, db_path=DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self):
        with self.connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    agent TEXT,
                    task_type TEXT,
                    sources TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id)
                        REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            db.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, id)
            """)

    def create(self, title="New conversation"):
        conversation_id = str(uuid.uuid4())
        now = timestamp()
        with self.connect() as db:
            db.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?)",
                (conversation_id, title, now, now),
            )
        return conversation_id

    def list(self):
        with self.connect() as db:
            rows = db.execute("""
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
            """).fetchall()
        return [dict(row) for row in rows]

    def get_messages(self, conversation_id):
        with self.connect() as db:
            exists = db.execute(
                "SELECT 1 FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if not exists:
                return None
            rows = db.execute("""
                SELECT id, role, content, agent, task_type, sources, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id
            """, (conversation_id,)).fetchall()

        import json
        result = []
        for row in rows:
            item = dict(row)
            item["sources"] = json.loads(item["sources"])
            result.append(item)
        return result

    def add_message(self, conversation_id, role, content,
                    agent=None, task_type=None, sources=None):
        import json
        now = timestamp()
        with self.connect() as db:
            exists = db.execute(
                "SELECT 1 FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if not exists:
                raise KeyError("Conversation not found.")

            db.execute("""
                INSERT INTO messages
                (conversation_id, role, content, agent, task_type, sources, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                conversation_id, role, content, agent, task_type,
                json.dumps(sources or []), now,
            ))

            if role == "user":
                count = db.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()[0]
                if count == 1:
                    title = " ".join(content.split())[:60]
                    db.execute(
                        "UPDATE conversations SET title = ? WHERE id = ?",
                        (title or "New conversation", conversation_id),
                    )

            db.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )

    def delete(self, conversation_id):
        with self.connect() as db:
            cursor = db.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            return cursor.rowcount > 0
