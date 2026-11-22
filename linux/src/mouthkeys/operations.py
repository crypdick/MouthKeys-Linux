"""Dispatch local history and backup requests."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from .backup import backup_export, backup_import, backup_preview
from .private_files import write_private
from .statistics import statistics
from .storage import History

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "backup_export",
    "backup_import",
    "backup_preview",
    "handle_action",
    "statistics",
    "write_private",
]

_MUTATIONS = {"delete", "delete_audio", "backup_export", "backup_import"}
_READS = {"history", "stats", "audio_usage"}


def _read_history(history: History, action: str, data: Mapping[str, object]) -> object:
    if action == "history":
        return history.rows(
            limit=int(cast("int | float | str", data.get("limit", 100))),
            offset=int(cast("int | float | str", data.get("offset", 0))),
            query=cast("str", data.get("query", "")),
        )
    if action == "stats":
        return statistics(history)
    return history.audio_usage()


def _write_history(history: History, action: str, data: Mapping[str, object]) -> object:
    if action == "delete":
        item = cast("str", data["id"])
        history.get(item)
        history.forget(item)
        return {"deleted": item}
    if action == "delete_audio":
        history.delete_audio(cast("str | None", data.get("id")))
        return history.audio_usage()
    path = cast("str | Path", data["path"])
    if action == "backup_export":
        return backup_export(history, path)
    return backup_import(
        history,
        path,
        app_map=data.get("app_map"),
        expected_sha256=cast("str | None", data.get("expected_sha256")),
    )


def handle_action(
    history: History | None,
    action: str,
    data: Mapping[str, object],
    *,
    allow_mutation: bool = True,
) -> object:
    if action == "backup_preview":
        return backup_preview(cast("str | Path", data["path"]), app_map=data.get("app_map"))
    if action not in _READS | _MUTATIONS:
        return NotImplemented
    if history is None:
        raise ValueError("History request requires a history store")
    if action in _READS:
        return _read_history(history, action, data)
    if not allow_mutation:
        raise ValueError("Wait until dictation finishes before changing or exporting history")
    return _write_history(history, action, data)
