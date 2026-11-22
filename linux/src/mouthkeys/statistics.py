"""Summarize local transcript history without writing to it."""

import time
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from typing import cast

from .config import Config
from .storage import History


def statistics(
    history: History, config: Config | None = None, today: date | None = None
) -> dict[str, object]:
    config = config or Config.load()
    rows = [row for row in history.all_rows() if row["text"]]
    now = time.time()
    days: Counter[date] = Counter()
    hours: Counter[int] = Counter()
    apps: Counter[str] = Counter()
    for row in rows:
        when = datetime.fromtimestamp(cast("float", row["created"]), UTC).astimezone()
        days[when.date()] += len(cast("str", row["text"]).split())
        hours[when.hour] += 1
        if row.get("app"):
            apps[cast("str", row["app"])] += 1
    today = today or datetime.now(UTC).astimezone().date()
    cursor = today if today in days else today - timedelta(days=1)
    streak = 0
    while days:
        if cursor in days:
            streak += 1
        elif config.weekends_break_streak == "on" or cursor.weekday() < 5:
            break
        cursor -= timedelta(days=1)
    words = sum(days.values())
    duration = sum(float(cast("float | str", r.get("duration", 0))) for r in rows)
    best_streak = run = 0
    previous = None
    for day in sorted(days):
        gap = (day - previous).days if previous else 0
        connected = gap == 1 or (
            previous is not None
            and config.weekends_break_streak == "off"
            and gap <= 3
            and all((previous + timedelta(days=step)).weekday() >= 5 for step in range(1, gap))
        )
        run = run + 1 if connected else 1
        best_streak = max(best_streak, run)
        previous = day
    milestones = {}
    for label, reached, targets in (
        ("Words", words, (1000, 10000, 50000, 100000, 500000, 1000000)),
        ("Dictations", len(rows), (50, 100, 500, 1000, 5000, 10000)),
        ("Streak", best_streak, (7, 14, 30, 60, 100, 365)),
    ):
        milestones[label] = [
            {"target": target, "achieved": reached >= target} for target in targets
        ]
    windows = {}
    for label, seconds in (
        ("6 hours", 21600),
        ("24 hours", 86400),
        ("7 days", 604800),
        ("30 days", 2592000),
        ("All time", float("inf")),
    ):
        selected = [r for r in rows if now - cast("float", r["created"]) <= seconds]
        windows[label] = {
            "transcriptions": len(selected),
            "words": sum(len(cast("str", r["text"]).split()) for r in selected),
        }
    return {
        "words": words,
        "transcriptions": len(rows),
        "recording_seconds": duration,
        "time_saved_seconds": max(0, words / config.typing_wpm * 60 - duration),
        "current_streak": streak,
        "best_streak": best_streak,
        "milestones": milestones,
        "windows": windows,
        "daily_words": {str(k): v for k, v in sorted(days.items())},
        "top_apps": apps.most_common(5),
        "peak_hour": hours.most_common(1),
        "longest_words": max((len(cast("str", r["text"]).split()) for r in rows), default=0),
        "most_words_in_day": max(days.values(), default=0),
        "average_words": round(words / len(rows), 1) if rows else 0,
    }
