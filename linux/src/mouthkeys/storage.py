import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .config import private_dir, state_dir


class History:
    """Private local history. Audio recovery is bounded by a storage budget."""

    def __init__(self, root: Path | None = None):
        self.root = private_dir(root or state_dir())
        self.audio = private_dir(self.root / "recordings")
        self.db = self.root / "history.sqlite3"
        with self.connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS history (
                id TEXT PRIMARY KEY, created REAL NOT NULL, text TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL, error TEXT NOT NULL DEFAULT '')""")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(history)")}
            for name, definition in METADATA.items():
                if name not in columns:
                    conn.execute(f"ALTER TABLE history ADD COLUMN {name} {definition}")
        self.db.chmod(0o600)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db, timeout=3)
        conn.row_factory = sqlite3.Row
        conn.create_function("casefold", 1, lambda value: value.casefold())
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def begin(self, item_id: str):
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO history(id, created, status, was_ai_processed, ai_processing_error, "
                "ai_processing_model) VALUES (?, ?, 'recording', 0, '', '')",
                (item_id, time.time()),
            )

    def update(self, item_id: str, status: str, text: str = "", error: str = "", **metadata):
        unknown = metadata.keys() - METADATA.keys()
        if unknown:
            raise ValueError(f"Unknown history metadata: {sorted(unknown)}")
        extra = "".join(f", {name}=?" for name in metadata)
        with self.connect() as conn:
            conn.execute(
                f"UPDATE history SET status=?, text=?, error=?{extra} WHERE id=?",
                (status, text, error, *metadata.values(), item_id),
            )

    def rows(self, limit=100, offset=0, query=""):
        if not 1 <= int(limit) <= 10000 or int(offset) < 0:
            raise ValueError("Invalid history page")
        with self.connect() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM history WHERE instr("
                    "casefold(text || ' ' || app || ' ' || window), "
                    "casefold(?)) > 0 ORDER BY created DESC, rowid DESC LIMIT ? OFFSET ?",
                    (query, limit, offset),
                )
            ]

    def get(self, item_id):
        self.recording(item_id)  # Validate UUID before using it across file operations.
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM history WHERE id=?", (item_id,)).fetchone()
        if row is None:
            raise ValueError("History entry not found")
        return dict(row)

    def forget(self, item_id):
        self.recording(item_id).unlink(missing_ok=True)
        with self.connect() as conn:
            conn.execute("DELETE FROM history WHERE id=?", (item_id,))

    def delete_audio(self, item_id=None):
        paths = [self.recording(item_id)] if item_id else self.audio.glob("*.wav")
        for path in paths:
            path.unlink(missing_ok=True)

    def audio_usage(self):
        files = list(self.audio.glob("*.wav"))
        return {"bytes": sum(path.stat().st_size for path in files), "recordings": len(files)}

    def all_rows(self):
        with self.connect() as conn:
            return [
                dict(row) for row in conn.execute("SELECT * FROM history ORDER BY created DESC")
            ]

    def latest_text(self):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT text FROM history WHERE text != '' ORDER BY created DESC LIMIT 1"
            ).fetchone()
        if row is None:
            raise ValueError("No transcript yet")
        return row["text"]

    def recover(self):
        with self.connect() as conn:
            conn.execute("""UPDATE history SET status='failed', error='Interrupted; reprocess audio'
                WHERE status IN ('recording', 'transcribing')""")

    def prune_audio(self, keep: str):
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id FROM history WHERE status IN "
                "('copied', 'paste_requested', 'insert_requested', 'pasted', "
                "'transcribed', 'send_requested', 'cancelled')"
            )
            for row in rows:
                if row["id"] != keep:
                    self.recording(row["id"]).unlink(missing_ok=True)

    def prune_budget(self, budget_bytes):
        """Prune oldest inactive audio, including failures; preserve transcript rows."""
        with self.connect() as conn:
            active = {
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM history WHERE status IN ('recording', 'transcribing')"
                )
            }
            created = {
                row["id"]: row["created"] for row in conn.execute("SELECT id, created FROM history")
            }
        files = []
        for path in self.audio.glob("*.wav"):
            if path.stem not in active:
                stat = path.stat()
                files.append((created.get(path.stem, stat.st_mtime), path, stat.st_size))
        total = sum(size for _, _, size in files)
        for _, path, size in sorted(files):
            if total <= budget_bytes:
                break
            path.unlink(missing_ok=True)
            total -= size

    def recording(self, item_id: str):
        # IDs arrive over IPC; never let a history request escape the recordings directory.
        import uuid

        return self.audio / f"{uuid.UUID(item_id)}.wav"

    def clear(self):
        with self.connect() as conn:
            conn.execute("DELETE FROM history")
        for path in self.audio.glob("*.wav"):
            path.unlink()


# Additive schema migration keeps existing transcripts and recordings untouched.
METADATA = {
    "raw_text": "TEXT NOT NULL DEFAULT ''",
    "duration": "REAL NOT NULL DEFAULT 0",
    "model": "TEXT NOT NULL DEFAULT ''",
    "language": "TEXT NOT NULL DEFAULT ''",
    "app": "TEXT NOT NULL DEFAULT ''",
    "window": "TEXT NOT NULL DEFAULT ''",
    # NULL distinguishes legacy entries from a recorded negative/empty outcome.
    "was_ai_processed": "INTEGER",
    "ai_processing_error": "TEXT",
    "ai_processing_model": "TEXT",
}
