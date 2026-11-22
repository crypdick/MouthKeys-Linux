"""Persist and validate private, bounded benchmark measurements."""

import json
import sys
import uuid
from pathlib import Path
from typing import Any

from .config import private_dir, state_dir
from .private_files import write_private

MAX_REPORTS = 100
MAX_REPORT_BYTES = 256 * 1024


def _latest_paths(directory: Path) -> list[Path]:
    timestamps = []
    for path in directory.glob("*.json"):
        try:
            timestamps.append((path.stat().st_mtime_ns, path))
        except OSError:
            continue
    return [path for _, path in sorted(timestamps, reverse=True)]


def save_measurement(report: dict[str, Any]) -> None:
    directory = private_dir(state_dir() / "benchmarks")
    write_private(directory / (uuid.uuid4().hex + ".json"), json.dumps(report).encode())
    for old in _latest_paths(directory)[MAX_REPORTS:]:
        old.unlink(missing_ok=True)


def _duration(value: object) -> bool:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    # A large JSON integer can overflow float conversion in display formatting.
    return 0 <= value <= sys.float_info.max


def _parse_report(raw: bytes) -> dict[str, Any] | None:
    report: object = json.loads(raw)
    if not isinstance(report, dict) or report.get("version") != 1:
        return None
    if not all(
        isinstance(report.get(key), dict)
        for key in ("hardware", "software", "artifact", "decode_settings")
    ):
        return None
    accuracy = report.get("accuracy")
    if accuracy is not None and not isinstance(accuracy, dict):
        return None
    if not all(
        isinstance(report.get(key), str)
        for key in (
            "engine",
            "model",
            "revision",
            "device",
            "compute_type",
            "timestamp_utc",
            "fixture_sha256",
        )
    ):
        return None
    if not all(
        _duration(report.get(key))
        for key in ("real_time_factor", "decode_seconds", "audio_seconds")
    ):
        return None
    return report


def measurements() -> list[dict[str, Any]]:
    """Return valid reports among the newest records, skipping any concurrently removed."""
    directory = state_dir() / "benchmarks"
    reports = []
    for path in _latest_paths(directory)[:MAX_REPORTS]:
        try:
            with path.open("rb") as stream:
                raw = stream.read(MAX_REPORT_BYTES + 1)
            if len(raw) > MAX_REPORT_BYTES:
                continue
            report = _parse_report(raw)
        except (OSError, ValueError, RecursionError):
            continue
        if report is not None:
            reports.append(report)
    return reports
