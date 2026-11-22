"""Private Command/Edit sessions; SQLite serializes GUI/daemon writers."""

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .assistant_session import (
    append_private_messages,
    assistant_mode,
    context_messages,
    pending_executions,
    recent_messages,
    record_private_execution,
    retain_execution,
    validate_execution,
    validate_executions,
    validate_messages,
    validated_session,
)
from .config import state_dir

__all__ = [
    "AssistantHistory",
    "append_private_messages",
    "context_messages",
    "pending_executions",
    "record_private_execution",
    "validate_execution",
    "validate_executions",
    "validate_messages",
]

MAX_SESSIONS = 50


def _execution_events(
    db: sqlite3.Connection, item_id: str, message_count: int
) -> list[dict[str, Any]]:
    events = [
        {"message_count": event["message_count"], "result": json.loads(event["result"])}
        for event in db.execute(
            "SELECT message_count, result FROM executions WHERE session_id=? ORDER BY id",
            (item_id,),
        )
    ]
    return validate_executions(events, message_count)


def _load_session(db: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    messages = validate_messages(json.loads(row["messages"]))
    return {
        **dict(row),
        "messages": messages,
        "executions": _execution_events(db, row["id"], len(messages)),
    }


def _store_executions(db: sqlite3.Connection, item_id: str, events: list[dict[str, Any]]) -> None:
    db.executemany(
        "INSERT INTO executions(session_id,message_count,result) VALUES (?,?,?)",
        (
            (item_id, event["message_count"], json.dumps(event["result"], ensure_ascii=False))
            for event in events
        ),
    )


def _prune(db: sqlite3.Connection) -> None:
    db.execute(
        "DELETE FROM sessions WHERE id NOT IN "
        "(SELECT id FROM sessions ORDER BY updated DESC, rowid DESC LIMIT ?)",
        (MAX_SESSIONS,),
    )


def _new_session(db: sqlite3.Connection, mode: str) -> dict[str, Any]:
    session = {
        "id": str(uuid.uuid4()),
        "mode": mode,
        "title": "New chat",
        "updated": time.time(),
        "messages": [],
        "executions": [],
    }
    db.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
        (session["id"], session["mode"], session["title"], session["updated"], "[]"),
    )
    _set_current(db, session)
    _prune(db)
    return session


def _set_current(db: sqlite3.Connection, session: dict[str, Any]) -> None:
    db.execute("INSERT OR REPLACE INTO current VALUES (?, ?)", (session["mode"], session["id"]))


def _find_session(db: sqlite3.Connection, item_id: str) -> sqlite3.Row:
    row: sqlite3.Row | None = db.execute("SELECT * FROM sessions WHERE id=?", (item_id,)).fetchone()
    if row is None:
        raise ValueError("Assistant session no longer exists")
    return row


class AssistantHistory:
    def __init__(self) -> None:
        self.path = state_dir() / "assistant.sqlite3"
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, mode TEXT NOT NULL, title TEXT NOT NULL,
                updated REAL NOT NULL, messages TEXT NOT NULL)""")
            db.execute("CREATE TABLE IF NOT EXISTS current (mode TEXT PRIMARY KEY, id TEXT)")
            db.execute("""CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                message_count INTEGER NOT NULL, result TEXT NOT NULL)""")
        self.path.chmod(0o600)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=3)
        db.execute("PRAGMA foreign_keys=ON")
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def rows(self, mode: str | None = None) -> list[dict[str, Any]]:
        if mode is not None:
            assistant_mode(mode)
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT id, mode, title, updated FROM sessions "
                    "WHERE (? IS NULL OR mode=?) ORDER BY updated DESC, rowid DESC",
                    (mode, mode),
                )
            ]

    def get(self, item_id: str) -> dict[str, Any]:
        uuid.UUID(item_id)
        with self.connect() as db:
            db.execute("BEGIN")
            return _load_session(db, _find_session(db, item_id))

    def current(self, mode: str) -> dict[str, Any]:
        assistant_mode(mode)
        with self.connect() as db:
            # Recheck and create within one transaction so two callers agree on the current chat.
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT sessions.* FROM current JOIN sessions ON sessions.id = current.id "
                "WHERE current.mode = ?",
                (mode,),
            ).fetchone()
            if row:
                try:
                    return _load_session(db, row)
                except ValueError:
                    pass  # An unreadable saved chat is not the current conversation.
            return _new_session(db, mode)

    def new(self, mode: str) -> dict[str, Any]:
        assistant_mode(mode)
        with self.connect() as db:
            return _new_session(db, mode)

    def select(self, item_id: str) -> dict[str, Any]:
        uuid.UUID(item_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            session = _load_session(db, _find_session(db, item_id))
            _set_current(db, session)
            return session

    def rename(self, item_id: str, title: object) -> None:
        if not isinstance(title, str) or not title.strip() or len(title) > 100:
            raise ValueError("Chat title must contain 1–100 characters")
        self.get(item_id)
        with self.connect() as db:
            db.execute("UPDATE sessions SET title=? WHERE id=?", (title.strip(), item_id))

    def delete(self, item_id: str) -> None:
        uuid.UUID(item_id)
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE id=?", (item_id,))
            db.execute("DELETE FROM current WHERE id=?", (item_id,))

    def append(self, item_id: str, messages: object) -> None:
        messages = validate_messages(messages)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM sessions WHERE id=?", (item_id,)).fetchone()
            if row is None:
                raise ValueError("Assistant session no longer exists")
            previous = validate_messages(json.loads(row["messages"]))
            combined, removed = recent_messages(previous, messages)
            if removed:
                db.execute(
                    "UPDATE executions SET message_count=MAX(0,message_count-?) WHERE session_id=?",
                    (removed, item_id),
                )
            title = row["title"]
            if title == "New chat":
                title = next((m["content"][:80] for m in messages if m["role"] == "user"), title)
            db.execute(
                "UPDATE sessions SET title=?, updated=?, messages=? WHERE id=?",
                (title, time.time(), json.dumps(combined, ensure_ascii=False), item_id),
            )

    def record_execution(self, item_id: str, result: object) -> None:
        receipt = validate_execution(result)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT mode,messages FROM sessions WHERE id=?", (item_id,)).fetchone()
            if row is None or row["mode"] != "command":
                raise ValueError("Command session no longer exists")
            count = len(validate_messages(json.loads(row["messages"])))
            events = retain_execution(_execution_events(db, item_id, count), receipt, count)
            db.execute("DELETE FROM executions WHERE session_id=?", (item_id,))
            _store_executions(db, item_id, events)
            db.execute("UPDATE sessions SET updated=? WHERE id=?", (time.time(), item_id))

    def export(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            db.execute("BEGIN")
            return [
                _load_session(db, row)
                for row in db.execute("SELECT * FROM sessions ORDER BY updated DESC, rowid DESC")
            ]

    @staticmethod
    def validate_entries(rows: object) -> list[dict[str, Any]]:
        if not isinstance(rows, list) or len(rows) > MAX_SESSIONS:
            raise ValueError("Invalid assistant sessions")
        prepared = []
        ids = set()
        for raw in rows:
            row = validated_session(raw)
            if row["id"] in ids:
                raise ValueError("Duplicate assistant session")
            ids.add(row["id"])
            prepared.append(row)
        return prepared

    def restore(self, rows: object) -> None:
        rows = self.validate_entries(rows)
        with self.connect() as db:
            for row in rows:
                db.execute(
                    "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?)",
                    (
                        row["id"],
                        row["mode"],
                        row["title"],
                        row["updated"],
                        json.dumps(row["messages"], ensure_ascii=False),
                    ),
                )
                _store_executions(db, row["id"], row["executions"])
            _prune(db)
