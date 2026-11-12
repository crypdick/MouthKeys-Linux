import json
import sqlite3
import uuid
import zipfile
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta

import pytest

from mouthkeys.config import CHOICES, RANGES, Config
from mouthkeys.dictionary import Dictionary
from mouthkeys.operations import backup_export, backup_import, export_audio, statistics
from mouthkeys.storage import History


def entry(history, text="hello world", day=None):
    item = str(uuid.uuid4())
    history.begin(item)
    history.update(item, "copied", text, duration=2.5, raw_text="raw words", app="Editor")
    if day:
        with history.connect() as connection:
            connection.execute(
                "UPDATE history SET created=? WHERE id=?",
                (datetime.combine(day, datetime.min.time()).timestamp(), item),
            )
    return item


def test_additive_schema_preserves_legacy_history(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    item = str(uuid.uuid4())
    with sqlite3.connect(root / "history.sqlite3") as connection:
        connection.execute(
            "CREATE TABLE history(id TEXT PRIMARY KEY, created REAL NOT NULL, "
            "text TEXT NOT NULL DEFAULT '', status TEXT NOT NULL, error TEXT NOT NULL DEFAULT '')"
        )
        connection.execute("INSERT INTO history VALUES (?, 1, 'original', 'copied', '')", (item,))
    history = History(root)
    assert history.get(item)["text"] == "original"
    assert history.get(item)["was_ai_processed"] is None
    assert history.get(item)["ai_processing_error"] is None
    history.update(item, "copied", "edited", model="small.en", duration=3)
    assert History(root).get(item)["duration"] == 3


def test_history_records_ai_outcome_and_backup_preserves_legacy_unknowns(tmp_path):
    history = History()
    item = entry(history, "AI output")
    assert history.get(item)["was_ai_processed"] == 0
    assert history.get(item)["ai_processing_error"] == ""
    history.update(item, "copied", "AI output", was_ai_processed=True)
    legacy = entry(history, "old transcript")
    history.update(
        legacy, "copied", "old transcript", was_ai_processed=None, ai_processing_error=None
    )
    path = tmp_path / "backup.json"
    backup_export(history, path)
    history.clear()
    backup_import(history, path)
    assert history.get(item)["was_ai_processed"] == 1
    assert history.get(legacy)["was_ai_processed"] is None
    assert history.get(legacy)["ai_processing_error"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("was_ai_processed", 2),
        ("was_ai_processed", 1.0),
        ("was_ai_processed", "true"),
        ("ai_processing_error", []),
    ],
)
def test_backup_rejects_invalid_ai_outcomes_before_mutation(tmp_path, field, value):
    history = History()
    item = entry(history, "original")
    path = tmp_path / "backup.json"
    backup_export(history, path)
    data = json.loads(path.read_text())
    data["history"][0][field] = value
    data["history"][0]["text"] = "must not be stored"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        backup_import(history, path)
    assert history.get(item)["text"] == "original"


def test_search_paging_deletion_and_audio_only_cleanup():
    history = History()
    first = entry(history, "first café")
    second = entry(history, "second café")
    third = entry(history, "third unrelated")
    assert [r["id"] for r in history.rows(query="CAFÉ", limit=1)] == [second]
    assert history.rows(query="café", limit=1, offset=1)[0]["id"] == first
    assert len(history.rows(query="editor")) == 3
    history.recording(first).write_bytes(b"private audio")
    history.delete_audio()
    assert history.get(first)["text"] == "first café"
    history.forget(third)
    with pytest.raises(ValueError, match="not found"):
        history.get(third)
    with pytest.raises(ValueError):
        history.recording("../../outside")


def test_backup_roundtrip_merges_without_deleting_audio_and_excludes_secrets(tmp_path, monkeypatch):
    history = History()
    item = entry(history)
    history.recording(item).write_bytes(b"keep")
    monkeypatch.setenv("MOUTHKEYS_AI_API_KEY", "secret-not-exported")
    Config(spoken_send="on", ai_enabled="on", ai_model="test").save()
    dictionary = Dictionary.defaults()
    dictionary["replacements"] = [{"triggers": ["mouth keys"], "replacement": "MouthKeys"}]
    Dictionary.save(dictionary)
    path = tmp_path / "backup.json"
    backup_export(history, path)
    assert path.stat().st_mode & 0o777 == 0o600
    assert "secret-not-exported" not in path.read_text()
    second = entry(history, "newer")
    Config().save()
    Dictionary.save(Dictionary.defaults())
    backup_import(history, path)
    assert Config.load().spoken_send == "on"
    assert Dictionary.apply("mouth keys") == "MouthKeys"
    assert history.get(second)["text"] == "newer"
    assert history.recording(item).read_bytes() == b"keep"
    broken = json.loads(path.read_text())
    broken["dictionary"] = {"bad": []}
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError):
        backup_import(history, path)
    assert len(history.all_rows()) == 2


def test_audio_archive_manifest_is_portable_and_private(tmp_path):
    history = History()
    item = entry(history, "transcript")
    history.recording(item).write_bytes(b"audio")
    path = tmp_path / "audio.zip"
    export_audio(history, path)
    assert path.stat().st_mode & 0o777 == 0o600
    with zipfile.ZipFile(path) as archive:
        assert archive.read(f"audio/{item}.wav") == b"audio"
        assert archive.read(f"text/{item}.txt") == b"transcript"
        assert json.loads(archive.read("manifest.jsonl"))["id"] == item


def test_stats_streak_skips_weekends_only_if_configured():
    history = History()
    monday = date(2026, 11, 9)
    entry(history, "one two three", monday)
    entry(history, "four five", monday - timedelta(days=3))
    stats = statistics(history, Config(weekends_break_streak="off"), monday)
    assert stats["current_streak"] == 2
    assert stats["words"] == 5
    assert stats["recording_seconds"] == 5
    assert statistics(history, Config(), monday)["current_streak"] == 1


@pytest.mark.parametrize("key", RANGES)
def test_numeric_config_ranges_apply_to_all_entrypoints(key):
    with pytest.raises(ValueError):
        replace(Config(), **{key: RANGES[key][1] + 1}).validate()
    with pytest.raises(ValueError):
        replace(Config(), **{key: True}).validate()


@pytest.mark.parametrize("key", CHOICES)
def test_enum_configuration_rejects_unknown_values(key):
    with pytest.raises(ValueError):
        replace(Config(), **{key: "unknown-choice"}).validate()


def test_legacy_config_loads_new_defaults(tmp_path):
    from mouthkeys.config import config_path, private_dir

    private_dir(config_path().parent)
    config_path().write_text('{"model":"small.en","language":"es"}')
    loaded = Config.load()
    assert loaded.model == "small.en"
    assert loaded.spoken_send == "off"
    assert loaded.ai_enabled == "off"
    assert len(asdict(loaded)) > 11


@pytest.mark.parametrize(
    "field,value",
    [
        ("created", float("nan")),
        ("created", float("inf")),
        ("created", 1e100),
        ("duration", -1),
        ("duration", True),
        ("duration", float("nan")),
    ],
)
def test_invalid_backup_numbers_rejected_before_mutation(tmp_path, field, value):
    history = History()
    item = entry(history)
    path = tmp_path / "backup.json"
    backup_export(history, path)
    data = json.loads(path.read_text())
    data["history"][0][field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        backup_import(history, path)
    assert history.get(item)["duration"] == 2.5
    assert len(history.all_rows()) == 1


def test_direct_insert_success_audio_prunes_like_other_delivery_statuses():
    history = History()
    first, second = entry(history), entry(history)
    history.update(first, "insert_requested", "first")
    history.recording(first).write_bytes(b"old")
    history.recording(second).write_bytes(b"latest")
    history.prune_audio(second)
    assert not history.recording(first).exists()
    assert history.recording(second).exists()
