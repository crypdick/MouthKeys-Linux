"""Bounded private Command/Edit sessions; SQLite serializes GUI/daemon writers."""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager

from .config import state_dir

MAX_SESSIONS = 50
MAX_MESSAGES = 200
MAX_SESSION_BYTES = 1024 * 1024
MAX_EXECUTIONS = 100
MAX_EXECUTION_BYTES = 512 * 1024


def validate_execution(result):
    fields = {"command", "working_directory", "returncode", "output", "truncated", "timed_out"}
    if not isinstance(result, dict) or not fields <= set(result) <= fields | {"cancelled"}:
        raise ValueError("Invalid command execution receipt")
    for name, limit in (("command", 16384), ("working_directory", 4096), ("output", 262144)):
        if not isinstance(result[name], str) or len(result[name]) > limit:
            raise ValueError("Invalid command execution receipt text")
    if (
        not result["command"].strip()
        or "\0" in result["command"]
        or "\0" in result["working_directory"]
        or (result["working_directory"] and not result["working_directory"].startswith("/"))
        or type(result["returncode"]) is not int
        or not -(2**31) <= result["returncode"] < 2**31
        or any(type(result[key]) is not bool for key in ("truncated", "timed_out"))
        or ("cancelled" in result and type(result["cancelled"]) is not bool)
    ):
        raise ValueError("Invalid command execution receipt values")
    try:
        size = len(json.dumps(result, ensure_ascii=False).encode())
    except UnicodeEncodeError:
        raise ValueError("Invalid command execution receipt Unicode") from None
    if size > MAX_EXECUTION_BYTES - 128:
        raise ValueError("Command execution receipt exceeds 512 KiB")
    return dict(result)


def validate_executions(events, message_count):
    if not isinstance(events, list) or len(events) > MAX_EXECUTIONS:
        raise ValueError("Invalid command execution events")
    for event in events:
        if (
            not isinstance(event, dict)
            or set(event) != {"message_count", "result"}
            or type(event["message_count"]) is not int
            or not 0 <= event["message_count"] <= message_count
            or event["message_count"] % 2
        ):
            raise ValueError("Invalid command execution event position")
        validate_execution(event["result"])
    if len(json.dumps(events, ensure_ascii=False).encode()) > MAX_EXECUTION_BYTES:
        raise ValueError("Command execution events exceed 512 KiB")
    return events


def pending_executions(session):
    messages = validate_messages(session["messages"])
    events = validate_executions(session.get("executions", []), len(messages))
    return [dict(event["result"]) for event in events if event["message_count"] >= len(messages)]


def record_private_execution(session, result):
    """Record an observed outcome in bounded runtime memory, without opening SQLite."""
    validate_messages(session["messages"])
    events = [
        *validate_executions(session.get("executions", []), len(session["messages"])),
        {"message_count": len(session["messages"]), "result": validate_execution(result)},
    ]
    while (
        len(events) > MAX_EXECUTIONS
        or len(json.dumps(events, ensure_ascii=False).encode()) > MAX_EXECUTION_BYTES
    ):
        events.pop(0)
    session["executions"] = events
    return session


def append_private_messages(session, messages):
    previous = validate_messages(session["messages"])
    events = validate_executions(session.get("executions", []), len(previous))
    combined = [*previous, *validate_messages(messages)]
    removed = max(0, len(combined) - MAX_MESSAGES)
    session["messages"] = validate_messages(combined[-MAX_MESSAGES:])
    session["executions"] = [
        {"message_count": max(0, event["message_count"] - removed), "result": dict(event["result"])}
        for event in events
    ]
    return session


def validate_messages(messages):
    if not isinstance(messages, list) or len(messages) > MAX_MESSAGES:
        raise ValueError("Invalid assistant messages")
    if len(messages) % 2:
        raise ValueError("Assistant history must contain completed turns")
    for index, message in enumerate(messages):
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in ("user", "assistant")
            or message["role"] != ("user" if index % 2 == 0 else "assistant")
            or not isinstance(message["content"], str)
        ):
            raise ValueError("Invalid assistant message")
    if len(json.dumps(messages, ensure_ascii=False).encode()) > MAX_SESSION_BYTES:
        raise ValueError("Assistant session exceeds 1 MiB; start a new chat")
    return messages


def context_messages(messages):
    """Keep complete recent messages within the provider context budget."""
    result, size = [], 0
    for message in reversed(validate_messages(messages)[-32:]):
        count = len(json.dumps(message, ensure_ascii=False).encode())
        if size + count > 240 * 1024:
            break
        result.insert(0, dict(message))
        size += count
    while result and result[0]["role"] != "user":
        result.pop(0)
    if result and result[-1]["role"] != "assistant":
        result.pop()
    return result


class AssistantHistory:
    def __init__(self):
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
    def connect(self):
        db = sqlite3.connect(self.path, timeout=3)
        db.execute("PRAGMA foreign_keys=ON")
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _mode(mode):
        if mode not in ("command", "edit"):
            raise ValueError("Invalid assistant mode")

    def rows(self, mode=None):
        if mode is not None:
            self._mode(mode)
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT id, mode, title, updated FROM sessions "
                    "WHERE (? IS NULL OR mode=?) ORDER BY updated DESC, rowid DESC",
                    (mode, mode),
                )
            ]

    def get(self, item_id):
        uuid.UUID(item_id)
        with self.connect() as db:
            db.execute("BEGIN")
            row = db.execute("SELECT * FROM sessions WHERE id=?", (item_id,)).fetchone()
            events = [
                {"message_count": event["message_count"], "result": json.loads(event["result"])}
                for event in db.execute(
                    "SELECT message_count, result FROM executions WHERE session_id=? ORDER BY id",
                    (item_id,),
                )
            ]
        if row is None:
            raise ValueError("Assistant session no longer exists")
        messages = validate_messages(json.loads(row["messages"]))
        return {
            **dict(row),
            "messages": messages,
            "executions": validate_executions(events, len(messages)),
        }

    def current(self, mode):
        self._mode(mode)
        with self.connect() as db:
            row = db.execute("SELECT id FROM current WHERE mode=?", (mode,)).fetchone()
        if row:
            try:
                return self.get(row["id"])
            except ValueError:
                pass
        return self.new(mode)

    def new(self, mode):
        self._mode(mode)
        item_id = str(uuid.uuid4())
        with self.connect() as db:
            db.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
                (item_id, mode, "New chat", time.time(), "[]"),
            )
            db.execute("INSERT OR REPLACE INTO current VALUES (?, ?)", (mode, item_id))
            db.execute(
                "DELETE FROM sessions WHERE id NOT IN "
                "(SELECT id FROM sessions ORDER BY updated DESC, rowid DESC LIMIT ?)",
                (MAX_SESSIONS,),
            )
        return self.get(item_id)

    def select(self, item_id):
        row = self.get(item_id)
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO current VALUES (?, ?)", (row["mode"], item_id))
        return row

    def rename(self, item_id, title):
        if not isinstance(title, str) or not title.strip() or len(title) > 100:
            raise ValueError("Chat title must contain 1–100 characters")
        self.get(item_id)
        with self.connect() as db:
            db.execute("UPDATE sessions SET title=? WHERE id=?", (title.strip(), item_id))

    def delete(self, item_id):
        uuid.UUID(item_id)
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE id=?", (item_id,))
            db.execute("DELETE FROM current WHERE id=?", (item_id,))

    def append(self, item_id, messages):
        validate_messages(messages)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM sessions WHERE id=?", (item_id,)).fetchone()
            if row is None:
                raise ValueError("Assistant session no longer exists")
            all_messages = json.loads(row["messages"]) + messages
            combined = all_messages[-MAX_MESSAGES:]
            removed = len(all_messages) - len(combined)
            if removed:
                db.execute(
                    "UPDATE executions SET message_count=MAX(0,message_count-?) WHERE session_id=?",
                    (removed, item_id),
                )
            validate_messages(combined)
            title = row["title"]
            if title == "New chat":
                title = next((m["content"][:80] for m in messages if m["role"] == "user"), title)
            db.execute(
                "UPDATE sessions SET title=?, updated=?, messages=? WHERE id=?",
                (title, time.time(), json.dumps(combined, ensure_ascii=False), item_id),
            )

    def record_execution(self, item_id, result):
        receipt = validate_execution(result)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT mode,messages FROM sessions WHERE id=?", (item_id,)).fetchone()
            if row is None or row["mode"] != "command":
                raise ValueError("Command session no longer exists")
            session = {
                "messages": json.loads(row["messages"]),
                "executions": [
                    {"message_count": event["message_count"], "result": json.loads(event["result"])}
                    for event in db.execute(
                        "SELECT message_count,result FROM executions "
                        "WHERE session_id=? ORDER BY id",
                        (item_id,),
                    )
                ],
            }
            record_private_execution(session, receipt)
            db.execute("DELETE FROM executions WHERE session_id=?", (item_id,))
            db.executemany(
                "INSERT INTO executions(session_id,message_count,result) VALUES (?,?,?)",
                [
                    (
                        item_id,
                        event["message_count"],
                        json.dumps(event["result"], ensure_ascii=False),
                    )
                    for event in session["executions"]
                ],
            )
            db.execute("UPDATE sessions SET updated=? WHERE id=?", (time.time(), item_id))

    def export(self):
        return [self.get(row["id"]) for row in self.rows()]

    @staticmethod
    def validate_entries(rows):
        import math

        if not isinstance(rows, list) or len(rows) > MAX_SESSIONS:
            raise ValueError("Invalid assistant sessions")
        ids = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) - {"executions"} != {
                "id",
                "mode",
                "title",
                "updated",
                "messages",
            }:
                raise ValueError("Invalid assistant session")
            uuid.UUID(row["id"])
            if row["id"] in ids:
                raise ValueError("Duplicate assistant session")
            ids.add(row["id"])
            AssistantHistory._mode(row["mode"])
            if (
                not isinstance(row["title"], str)
                or len(row["title"]) > 100
                or type(row["updated"]) not in (int, float)
                or not math.isfinite(row["updated"])
            ):
                raise ValueError("Invalid assistant session metadata")
            validate_messages(row["messages"])
            validate_executions(row.get("executions", []), len(row["messages"]))
            if row["mode"] != "command" and row.get("executions"):
                raise ValueError("Only command sessions may contain execution events")
        return rows

    def restore(self, rows):
        self.validate_entries(rows)
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
                db.executemany(
                    "INSERT INTO executions(session_id,message_count,result) VALUES (?,?,?)",
                    [
                        (
                            row["id"],
                            event["message_count"],
                            json.dumps(event["result"], ensure_ascii=False),
                        )
                        for event in row.get("executions", [])
                    ],
                )
            db.execute(
                "DELETE FROM sessions WHERE id NOT IN "
                "(SELECT id FROM sessions ORDER BY updated DESC, rowid DESC LIMIT ?)",
                (MAX_SESSIONS,),
            )
