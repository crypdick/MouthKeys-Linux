"""Validate a complete backup before writing any local stores."""

import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .assistant_history import AssistantHistory
from .config import Config
from .desktop_entry import apply_startup
from .dictionary import Dictionary, DictionaryData
from .file_history import FileHistory, FileHistoryEntry
from .private_files import write_private
from .profiles import PromptProfile, load_profiles, save_profiles, validate_profiles
from .pronunciation import PronunciationStore
from .pronunciation_spans import PronunciationProfile
from .routing import PromptRoute, load_routes, save_routes, validate_routes
from .storage import History
from .upstream_backup import load_upstream_backup, read_backup
from .vocabulary import VocabularyEntry, VocabularyStore

_REQUIRED_HISTORY_TEXT = {"id", "text", "status", "error"}
_TEXT_FIELDS = _REQUIRED_HISTORY_TEXT | {"raw_text", "model", "language", "app", "window"}
_REQUIRED_HISTORY_FIELDS = _REQUIRED_HISTORY_TEXT | {"created"}
_HISTORY_FIELDS = _TEXT_FIELDS | {
    "created",
    "duration",
    "was_ai_processed",
    "ai_processing_error",
    "ai_processing_model",
}


@dataclass(frozen=True)
class _RestorePlan:
    config: Config
    dictionary: DictionaryData
    profiles: list[PromptProfile]
    rows: list[dict[str, object]]
    routes: list[PromptRoute]
    file_rows: list[FileHistoryEntry] | None
    assistant_rows: list[dict[str, Any]] | None
    vocabulary: list[VocabularyEntry] | None
    pronunciations: list[PronunciationProfile] | None


def backup_export(history: History, path: str | Path) -> str:
    document = {
        "version": 1,
        "config": asdict(Config.load()),
        "dictionary": Dictionary.load(),
        "profiles": load_profiles(),
        "history": history.all_rows(),
        "file_history": FileHistory().rows(),
        "assistant_history": AssistantHistory().export(),
        "prompt_routes": load_routes(),
        "vocabulary": VocabularyStore.entries(),
        "pronunciations": PronunciationStore.load(),
    }
    return write_private(path, json.dumps(document, indent=2).encode())


def _finite_number(value: object, message: str) -> float:
    if not isinstance(value, int | float) or type(value) not in (int, float):
        raise ValueError(message)
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(message) from None
    if not math.isfinite(result):
        raise ValueError(message)
    return result


def _validate_outcome(row: dict[str, object]) -> None:
    processed = row.get("was_ai_processed")
    if processed is not None and (type(processed) not in (bool, int) or processed not in (0, 1)):
        raise ValueError("Invalid history AI outcome")
    for field, label in (("ai_processing_error", "error"), ("ai_processing_model", "model")):
        value = row.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Invalid history AI {label}")


def _history_row(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict) or not _REQUIRED_HISTORY_FIELDS.issubset(raw):
        raise ValueError("Invalid history entry")
    if raw.keys() - _HISTORY_FIELDS:
        raise ValueError("Invalid history fields")
    if any(not isinstance(raw[name], str) for name in _TEXT_FIELDS & raw.keys()):
        raise ValueError("Invalid history text")
    uuid.UUID(raw["id"])

    created = _finite_number(raw["created"], "Invalid history fields")
    try:
        datetime.fromtimestamp(created, UTC).astimezone()
    except (ValueError, OverflowError, OSError):
        raise ValueError("History timestamp is out of range") from None
    row = {**raw, "created": created}
    _validate_outcome(row)
    if "duration" in row:
        duration = _finite_number(row["duration"], "Invalid history duration")
        if duration < 0:
            raise ValueError("Invalid history duration")
        row["duration"] = duration
    return row


def _history_rows(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, list) or len(raw) > 1_000_000:
        raise ValueError("Invalid backup history")
    return [_history_row(row) for row in raw]


def _restore_config(raw: object) -> Config:
    if not isinstance(raw, dict):
        raise TypeError("Configuration must be a JSON object")
    try:
        current = Config.load()
    except (OSError, ValueError, TypeError):
        current = Config()
    return Config.from_mapping({**asdict(current), **raw})


def _restore_profiles(document: dict[str, object], config: Config) -> list[PromptProfile]:
    try:
        current = load_profiles()
    except (OSError, ValueError):
        if "profiles" not in document:
            raise
        current = []
    by_id = {profile["id"]: profile for profile in current}
    imported = validate_profiles(document.get("profiles", []))
    by_id.update({profile["id"]: profile for profile in imported})
    profiles = validate_profiles(list(by_id.values()))
    if config.ai_profile and config.ai_profile not in by_id:
        raise ValueError("Backup selects a missing AI profile")
    for selection in (config.edit_profile, config.command_profile):
        if selection not in {"default", "off"} and selection not in by_id:
            raise ValueError("Backup selects a missing mode profile")
    return profiles


def _prepare_restore(document: dict[str, object]) -> _RestorePlan:
    if document.get("version") != 1:
        raise ValueError("Unsupported backup version")
    config = _restore_config(document.get("config"))
    rows = _history_rows(document.get("history"))
    dictionary = Dictionary.validate(document.get("dictionary"))
    profiles = _restore_profiles(document, config)
    return _RestorePlan(
        config=config,
        dictionary=dictionary,
        profiles=profiles,
        rows=rows,
        routes=(
            validate_routes(document["prompt_routes"], profiles)
            if "prompt_routes" in document
            else load_routes(profiles)
        ),
        file_rows=(
            FileHistory.validate_entries(document["file_history"])
            if "file_history" in document
            else None
        ),
        assistant_rows=(
            AssistantHistory.validate_entries(document["assistant_history"])
            if "assistant_history" in document
            else None
        ),
        vocabulary=(
            VocabularyStore.validate_entries(document["vocabulary"])
            if "vocabulary" in document
            else None
        ),
        pronunciations=(
            PronunciationStore.validate(document["pronunciations"])
            if "pronunciations" in document
            else None
        ),
    )


def _read_import(
    path: str | Path, app_map: object, expected_sha256: str | None = None
) -> tuple[dict[str, object], dict[str, object] | None, str]:
    document, digest = read_backup(path, with_digest=True)
    if expected_sha256 is not None and expected_sha256 != digest:
        raise ValueError("Backup changed since preview; preview it again")
    if "schemaVersion" in document:
        converted, report = load_upstream_backup(path, app_map=app_map, document=document)
        return converted, {**report, "file_sha256": digest}, digest
    return document, None, digest


def backup_preview(path: str | Path, *, app_map: object = None) -> dict[str, object]:
    # Preview validates files/snapshots without constructing history databases.
    document, report, digest = _read_import(path, app_map)
    plan = _prepare_restore(document)
    if report is not None:
        return report
    profiles = document.get("profiles")
    return {
        "format": "native",
        "counts": {
            "history": len(plan.rows),
            "profiles": len(profiles) if isinstance(profiles, list) else 0,
            "routes": len(plan.routes),
            "replacements": len(plan.dictionary["replacements"]),
        },
        "changes": document["config"],
        "warnings": [],
        "unmapped": {},
        "file_sha256": digest,
    }


def _apply_restore(history: History, plan: _RestorePlan, *, native: bool) -> None:
    # Merge by UUID; never delete existing recordings or entries during restore.
    with history.connect() as connection:
        for row in plan.rows:
            names = list(row)
            placeholders = ",".join("?" for _ in names)
            connection.execute(
                f"INSERT OR REPLACE INTO history ({','.join(names)}) VALUES ({placeholders})",  # noqa: S608
                list(row.values()),
            )
    Dictionary.save(plan.dictionary)
    save_profiles(plan.profiles)
    save_routes(plan.routes)
    plan.config.save()
    if native:
        apply_startup(plan.config)
    if plan.file_rows is not None:
        FileHistory().restore(plan.file_rows)
    if plan.assistant_rows is not None:
        AssistantHistory().restore(plan.assistant_rows)
    if plan.vocabulary is not None:
        VocabularyStore.save_entries(plan.vocabulary)
    if plan.pronunciations is not None:
        PronunciationStore.save(plan.pronunciations)


def backup_import(
    history: History | None,
    path: str | Path,
    *,
    app_map: object = None,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    document, migration, _ = _read_import(path, app_map, expected_sha256)
    plan = _prepare_restore(document)
    if history is None:
        raise ValueError("Backup restore requires a history store")
    _apply_restore(history, plan, native=migration is None)
    result: dict[str, object] = {"imported": len(plan.rows), "mode": "merge"}
    if migration is not None:
        result["migration"] = migration
    return result
