"""Private, bounded file-transcription results, independent of microphone history."""

import json
import math
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .config import state_dir

MAX_RESULT = 20 * 1024 * 1024


def validate_result(result):
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise ValueError("Invalid file transcription result")
    for name in ("duration", "processing_seconds"):
        value = result.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid result {name}")
    segments = result.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("Invalid file segments")
    for segment in segments:
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise ValueError("Invalid file segment")
        start, end = segment.get("start"), segment.get("end")
        if (
            type(start) not in (int, float)
            or type(end) not in (int, float)
            or not math.isfinite(start)
            or not math.isfinite(end)
            or not 0 <= start <= end
        ):
            raise ValueError("Invalid segment timestamps")
    if len(json.dumps(result, allow_nan=False).encode()) > MAX_RESULT:
        raise ValueError("File result exceeds 20 MiB")
    return result


class FileHistory:
    def __init__(self):
        self.path = state_dir() / "files.sqlite3"
        with self.connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS results (
                id TEXT PRIMARY KEY, created REAL NOT NULL, name TEXT NOT NULL,
                result TEXT NOT NULL)""")
        self.path.chmod(0o600)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=3)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def add(self, filename, result, limit=50):
        validate_result(result)
        if type(limit) is not int or not 1 <= limit <= 10000:
            raise ValueError("File history limit must be 1–10000")
        item_id = str(uuid.uuid4())
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO results VALUES (?, ?, ?, ?)",
                (
                    item_id,
                    time.time(),
                    Path(filename).name,
                    json.dumps(result, ensure_ascii=False, allow_nan=False),
                ),
            )
            connection.execute(
                "DELETE FROM results WHERE id NOT IN "
                "(SELECT id FROM results ORDER BY created DESC, rowid DESC LIMIT ?)",
                (limit,),
            )
        return item_id

    def rows(self, query=""):
        with self.connect() as connection:
            rows = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM results ORDER BY created DESC, rowid DESC"
                )
            ]
        folded = query.casefold()
        return [
            {**row, "result": json.loads(row["result"])}
            for row in rows
            if folded in (row["name"] + row["result"]).casefold()
        ]

    def delete(self, item_id):
        uuid.UUID(item_id)
        with self.connect() as connection:
            connection.execute("DELETE FROM results WHERE id=?", (item_id,))

    def clear(self):
        with self.connect() as connection:
            connection.execute("DELETE FROM results")

    @staticmethod
    def validate_entries(rows):
        if not isinstance(rows, list) or len(rows) > 10000:
            raise ValueError("Invalid file history entries")
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"id", "created", "name", "result"}:
                raise ValueError("Invalid file history entry")
            uuid.UUID(row["id"])
            if (
                type(row["created"]) not in (int, float)
                or not math.isfinite(row["created"])
                or not isinstance(row["name"], str)
                or Path(row["name"]).name != row["name"]
            ):
                raise ValueError("Invalid file history metadata")
            try:
                datetime.fromtimestamp(row["created"])
            except (ValueError, OverflowError, OSError):
                raise ValueError("Invalid file history date") from None
            validate_result(row["result"])
        return rows

    def restore(self, rows):
        self.validate_entries(rows)
        with self.connect() as connection:
            for row in rows:
                connection.execute(
                    "INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?)",
                    (
                        row["id"],
                        row["created"],
                        row["name"],
                        json.dumps(row["result"], ensure_ascii=False, allow_nan=False),
                    ),
                )
