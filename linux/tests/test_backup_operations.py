import json
import os
import uuid

import pytest

from mouthkeys.config import Config, config_path, private_dir
from mouthkeys.dictionary import Dictionary
from mouthkeys.operations import backup_import, backup_preview, handle_action, statistics
from mouthkeys.private_files import write_private
from mouthkeys.profiles import load_profiles, profiles_path
from mouthkeys.routing import load_routes, save_routes
from mouthkeys.storage import History


def archive(tmp_path, **overrides):
    document = {
        "version": 1,
        "config": {},
        "dictionary": Dictionary.defaults(),
        "history": [],
        **overrides,
    }
    path = tmp_path / "archive.json"
    path.write_text(json.dumps(document))
    return path


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"version": 2}, ValueError, "Unsupported backup version"),
        ({"config": []}, TypeError, "Configuration must be a JSON object"),
        ({"history": {}}, ValueError, "Invalid backup history"),
        ({"history": [{}]}, ValueError, "Invalid history entry"),
    ],
)
def test_native_preview_rejects_invalid_documents(tmp_path, overrides, error, message):
    with pytest.raises(error, match=message):
        backup_preview(archive(tmp_path, **overrides))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"unsupported_column": "ignored?"}, "Invalid history fields"),
        ({"text": ["not a string"]}, "Invalid history text"),
        ({"ai_processing_model": []}, "Invalid history AI model"),
    ],
)
def test_invalid_history_rows_cannot_partially_apply_backup(tmp_path, changes, message):
    history = History()
    item = str(uuid.uuid4())
    history.begin(item)
    history.update(item, "copied", "original")
    row = {"id": item, "created": 1, "text": "imported", "status": "copied", "error": ""}
    path = archive(tmp_path, history=[row, {**row, **changes}], config={"spoken_send": "on"})
    with pytest.raises(ValueError, match=message):
        backup_import(history, path)
    assert Config.load().spoken_send == "off"
    assert history.get(item)["text"] == "original"


@pytest.mark.parametrize(
    ("setting", "message"),
    [("ai_profile", "missing AI profile"), ("edit_profile", "missing mode profile")],
)
def test_preview_detects_unresolved_profile_references(tmp_path, setting, message):
    path = archive(tmp_path, config={setting: str(uuid.uuid4())})
    with pytest.raises(ValueError, match=message):
        backup_preview(path)


def test_explicit_sections_can_repair_corrupt_live_settings(tmp_path):
    private_dir(config_path().parent)
    routes = [{"app": "Editor", "profile": "default"}]
    save_routes(routes)
    config_path().write_text("{")
    profiles_path().write_text("{")
    path = archive(tmp_path, config={"spoken_send": "on"})
    history = History()

    # When profiles are absent, a legacy backup cannot replace an unreadable profile store.
    with pytest.raises(json.JSONDecodeError):
        backup_import(history, path)
    assert config_path().read_text() == "{"
    assert profiles_path().read_text() == "{"

    path = archive(tmp_path, config={"spoken_send": "on"}, profiles=[])
    assert backup_import(history, path) == {"imported": 0, "mode": "merge"}
    assert Config.load().spoken_send == "on"
    assert load_profiles() == []
    assert load_routes() == routes


def test_legacy_row_without_optional_metadata_restores_with_defaults(tmp_path):
    history = History()
    item = str(uuid.uuid4())
    row = {"id": item, "created": 1, "text": "old", "status": "copied", "error": ""}
    path = archive(tmp_path, history=[row])
    assert backup_preview(path)["counts"]["history"] == 1
    backup_import(history, path)
    assert history.get(item)["duration"] == 0
    assert history.get(item)["was_ai_processed"] is None


def test_preview_can_run_without_history_but_apply_cannot(tmp_path):
    path = archive(tmp_path)
    assert handle_action(None, "backup_preview", {"path": path})["format"] == "native"
    with pytest.raises(ValueError, match="history store"):
        backup_import(None, path)
    with pytest.raises(ValueError, match="history store"):
        handle_action(None, "stats", {})


@pytest.mark.parametrize("action", ["delete", "delete_audio", "backup_export", "backup_import"])
def test_busy_controller_rejects_mutating_history_actions_before_touching_data(action, tmp_path):
    history = History()
    item = str(uuid.uuid4())
    history.begin(item)
    audio = history.recording(item)
    audio.write_bytes(b"private")
    path = tmp_path / "not-written.json"
    with pytest.raises(ValueError, match="Wait until dictation finishes"):
        handle_action(history, action, {"id": item, "path": path}, allow_mutation=False)
    assert audio.read_bytes() == b"private"
    assert history.get(item)["id"] == item
    assert not path.exists()


def test_history_dispatch_keeps_read_search_and_audio_only_removal():
    history = History()
    item = str(uuid.uuid4())
    history.begin(item)
    history.update(item, "copied", "exact text")
    audio = history.recording(item)
    audio.write_bytes(b"audio")
    assert handle_action(history, "history", {"query": "EXACT"})[0]["id"] == item
    assert handle_action(history, "stats", {}, allow_mutation=False)["words"] == 2
    assert handle_action(history, "audio_usage", {}) == {"recordings": 1, "bytes": 5}
    assert handle_action(history, "delete_audio", {"id": item}) == {"recordings": 0, "bytes": 0}
    assert history.get(item)["text"] == "exact text"
    assert handle_action(history, "delete", {"id": item}) == {"deleted": item}
    assert history.all_rows() == []


def test_dispatch_exports_and_restores_the_reviewed_backup(tmp_path):
    source, destination = History(tmp_path / "source"), History(tmp_path / "destination")
    item = str(uuid.uuid4())
    source.begin(item)
    source.update(item, "copied", "backed up")
    path = tmp_path / "archive.json"

    assert handle_action(source, "backup_export", {"path": path}) == str(path)
    report = handle_action(destination, "backup_preview", {"path": path})
    assert handle_action(
        destination, "backup_import", {"path": path, "expected_sha256": report["file_sha256"]}
    ) == {"imported": 1, "mode": "merge"}
    assert destination.get(item)["text"] == "backed up"


def test_transcript_without_app_counts_in_activity_but_not_top_apps():
    history = History()
    item = str(uuid.uuid4())
    history.begin(item)
    history.update(item, "copied", "words without app")
    totals = statistics(history)
    assert totals["words"] == 3
    assert totals["top_apps"] == []


def test_private_write_cleans_up_and_preserves_target_if_sync_fails(tmp_path, monkeypatch):
    path = tmp_path / "target"
    path.write_bytes(b"original")

    def fail_sync(_descriptor):
        raise OSError("cannot sync")

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OSError, match="cannot sync"):
        write_private(path, b"replacement")
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]


def test_private_write_reports_missing_parent_without_creating_it(tmp_path):
    path = tmp_path / "missing" / "file"
    with pytest.raises(FileNotFoundError):
        write_private(path, b"new")
    assert list(tmp_path.iterdir()) == []
