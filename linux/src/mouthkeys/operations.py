"""History, stats, and portable data tools shared by CLI, GUI, and local API."""

import json
import math
import os
import tempfile
import uuid
import zipfile
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

from .config import Config


def write_private(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def statistics(history, config=None, today=None):
    config = config or Config.load()
    rows = [r for r in history.all_rows() if r["text"]]
    now = datetime.now().timestamp()
    days = Counter()
    hours = Counter()
    apps = Counter()
    for row in rows:
        when = datetime.fromtimestamp(row["created"])
        days[when.date()] += len(row["text"].split())
        hours[when.hour] += 1
        if row.get("app"):
            apps[row["app"]] += 1
    today = today or date.today()
    cursor = today if today in days else today - timedelta(days=1)
    streak = 0
    while days:
        if cursor in days:
            streak += 1
        elif config.weekends_break_streak == "on" or cursor.weekday() < 5:
            break
        cursor -= timedelta(days=1)
    words = sum(days.values())
    duration = sum(float(r.get("duration", 0)) for r in rows)
    windows = {}
    for label, seconds in (
        ("6 hours", 21600),
        ("24 hours", 86400),
        ("7 days", 604800),
        ("30 days", 2592000),
        ("All time", float("inf")),
    ):
        selected = [r for r in rows if now - r["created"] <= seconds]
        windows[label] = {
            "transcriptions": len(selected),
            "words": sum(len(r["text"].split()) for r in selected),
        }
    return {
        "words": words,
        "transcriptions": len(rows),
        "recording_seconds": duration,
        "time_saved_seconds": max(0, words / config.typing_wpm * 60 - duration),
        "current_streak": streak,
        "windows": windows,
        "daily_words": {str(k): v for k, v in sorted(days.items())},
        "top_apps": apps.most_common(5),
        "peak_hour": hours.most_common(1),
        "longest_words": max((len(r["text"].split()) for r in rows), default=0),
        "most_words_in_day": max(days.values(), default=0),
        "average_words": round(words / len(rows), 1) if rows else 0,
    }


def export_audio(history, path, item_id=None):
    rows = [history.get(item_id)] if item_id else history.all_rows()
    with tempfile.NamedTemporaryFile(dir=Path(path).parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            manifest = []
            for row in rows:
                audio = history.recording(row["id"])
                if audio.exists():
                    archive.write(audio, f"audio/{audio.name}")
                    archive.writestr(f"text/{row['id']}.txt", row["text"])
                    manifest.append({**row, "audio": f"audio/{audio.name}"})
            archive.writestr("manifest.jsonl", "\n".join(json.dumps(r) for r in manifest))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def backup_export(history, path):
    from .dictionary import Dictionary
    from .profiles import load_profiles

    return write_private(
        path,
        json.dumps(
            {
                "version": 1,
                "config": asdict(Config.load()),
                "dictionary": Dictionary.load(),
                "profiles": load_profiles(),
                "history": history.all_rows(),
            },
            indent=2,
        ).encode(),
    )


def backup_import(history, path):
    from .dictionary import Dictionary
    from .profiles import save_profiles, validate_profiles

    path = Path(path)
    if path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("Backup exceeds 100 MiB")
    data = json.loads(path.read_text())
    if data.get("version") != 1:
        raise ValueError("Unsupported backup version")
    config = Config(**data["config"]).validate()
    rows = data["history"]
    if not isinstance(rows, list) or len(rows) > 1000000:
        raise ValueError("Invalid backup history")
    fields = {
        "id",
        "created",
        "text",
        "status",
        "error",
        "raw_text",
        "duration",
        "model",
        "language",
        "app",
        "window",
    }
    for row in rows:
        if (
            not isinstance(row, dict)
            or not {"id", "created", "text", "status", "error"} <= row.keys()
        ):
            raise ValueError("Invalid history entry")
        uuid.UUID(row["id"])
        if (
            set(row) - fields
            or type(row["created"]) not in (int, float)
            or not math.isfinite(row["created"])
        ):
            raise ValueError("Invalid history fields")
        try:
            datetime.fromtimestamp(row["created"])
        except (ValueError, OverflowError, OSError):
            raise ValueError("History timestamp is out of range") from None
        for name in fields - {"created", "duration"}:
            if name in row and not isinstance(row[name], str):
                raise ValueError("Invalid history text")
        if "duration" in row and (
            type(row["duration"]) not in (int, float)
            or not math.isfinite(row["duration"])
            or row["duration"] < 0
        ):
            raise ValueError("Invalid history duration")
    # Validate dictionary before touching current data. Its saver supplies stricter validation.
    dictionary = Dictionary.validate(data["dictionary"])
    profiles = validate_profiles(data.get("profiles", []))
    if config.ai_profile and config.ai_profile not in {profile["id"] for profile in profiles}:
        raise ValueError("Backup selects a missing AI profile")
    # Merge by UUID, never delete existing recordings or entries during restore.
    with history.connect() as connection:
        for row in rows:
            names = list(row)
            placeholders = ",".join("?" for _ in names)
            connection.execute(
                f"INSERT OR REPLACE INTO history ({','.join(names)}) VALUES ({placeholders})",
                list(row.values()),
            )
    Dictionary.save(dictionary)
    save_profiles(profiles)
    config.save()
    return {"imported": len(rows), "mode": "merge"}


def handle_action(history, action, data, allow_mutation=True):
    if action == "history":
        return history.rows(
            limit=int(data.get("limit", 100)),
            offset=int(data.get("offset", 0)),
            query=data.get("query", ""),
        )
    if action == "stats":
        return statistics(history)
    if action == "audio_usage":
        return history.audio_usage()
    if action not in {"delete", "delete_audio", "export_audio", "backup_export", "backup_import"}:
        return NotImplemented
    if not allow_mutation:
        raise ValueError("Wait until dictation finishes before changing or exporting history")
    if action == "delete":
        history.get(data["id"])
        history.forget(data["id"])
        return {"deleted": data["id"]}
    if action == "delete_audio":
        history.delete_audio(data.get("id"))
        return history.audio_usage()
    if action == "export_audio":
        return export_audio(history, data["path"], data.get("id"))
    if action == "backup_export":
        return backup_export(history, data["path"])
    return backup_import(history, data["path"])
