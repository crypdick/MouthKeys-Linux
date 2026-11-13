import copy
import hashlib
import json
import os
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from mouthkeys.config import Config
from mouthkeys.dictionary import Dictionary
from mouthkeys.operations import backup_import, backup_preview, handle_action
from mouthkeys.profiles import PROFILE_FIELDS, load_profiles
from mouthkeys.routing import load_routes, select_prompt
from mouthkeys.storage import History
from mouthkeys.upstream_backup import convert_upstream_backup, read_backup
from mouthkeys.vocabulary import VocabularyStore

PROFILE_ID = "a90dbe73-4ab7-42dc-acfc-f51d3293ab13"
HISTORY_ID = "c66ae138-d0a2-47c1-aeab-6bd1f23f81aa"


def upstream_document():
    # Shapes/keys from upstream BackupService.swift and TranscriptionHistoryEntry.
    return {
        "schemaVersion": {"major": 1, "minor": 0},
        "appVersion": "0.1.5",
        "exportedAt": "2026-11-01T19:40:09Z",
        "settings": {
            "selectedProviderID": "ollama",
            "selectedModelByProvider": {"ollama": "existing-local-model"},
            "savedProviders": [],
            "selectedSpeechModel": "whisper-medium",
            "customDictionaryEntries": [
                {
                    "id": "df59fc30-6fe5-4578-804a-137e052c3bf8",
                    "triggers": ["mouth keys"],
                    "replacement": "MouthKeys",
                },
            ],
            "fillerWords": ["um", "uh"],
            "removeFillerWordsEnabled": True,
            "overlayBottomOffset": 50.0,
            "overlayPosition": "top",
            "transcriptionSoundVolume": 0.25,
            "transcriptionStartSound": "fluid_sfx_2",
            "weekendsDontBreakStreak": True,
            "saveTranscriptionHistory": True,
            "defaultDictationPromptOverride": "Clean only; do not answer.",
            "selectedDictationPromptID": PROFILE_ID.upper(),
            "selectedEditPromptID": PROFILE_ID,
            "dictationPromptRoutingScope": "selectedAppsOnly",
            "hotkeyShortcut": {
                "kind": "keyboard",
                "keyCode": 49,
                "modifierFlagsRawValue": 1048576,
                "modifierKeyCodes": [],
            },
            "preferredInputDeviceUID": "AppleUSBAudioEngine:some-device",
            "shareAnonymousAnalytics": True,
            "autoUpdateCheckEnabled": True,
            "commandModeConfirmBeforeExecute": False,
            "textInsertionMode": "reliablePaste",
            "pronunciationMatchingEnabled": False,
        },
        "promptProfiles": [
            {
                "id": PROFILE_ID.upper(),
                "name": "Concise",
                "prompt": "Shorten selected text.",
                "mode": "edit",
                "includeContext": True,
                "createdAt": "2026-10-01T01:02:03Z",
                "updatedAt": "2026-10-02T01:02:03Z",
            }
        ],
        "appPromptBindings": [
            {
                "id": "e0d5d6aa-bb9a-4c2d-aecc-857ec57e93fa",
                "mode": "edit",
                "appBundleID": "com.example.Editor",
                "appName": "Editor",
                "promptID": PROFILE_ID,
            }
        ],
        "transcriptionHistory": [
            {
                "id": HISTORY_ID.upper(),
                "timestamp": "2026-11-01T12:00:00-08:00",
                "rawText": "raw private words",
                "processedText": "Final private words.",
                "appName": "Editor",
                "windowTitle": "notes",
                "characterCount": 20,
                "wordCount": 3,
                "wasAIProcessed": True,
                "processingModel": "actual-cleanup-model",
                "aiProcessingError": None,
                "audio": {
                    "fileName": "not-present.wav",
                    "durationMilliseconds": 2500,
                    "byteCount": 80000,
                    "sampleRate": 16000,
                    "channels": 1,
                    "model": "whisper-medium",
                },
            }
        ],
        "pronunciationProfiles": [{"modelKey": "parakeet-v3", "hiddenSize": 1024}],
    }


def source_file(tmp_path, document=None):
    path = tmp_path / "upstream.json"
    path.write_text(json.dumps(upstream_document() if document is None else document))
    return path


def xdg_files():
    return {
        (kind, str(path.relative_to(root))): path.read_bytes()
        for kind in ("CONFIG", "STATE", "CACHE", "DATA")
        for root in [Path(os.environ[f"XDG_{kind}_HOME"])]
        for path in root.rglob("*")
        if path.is_file()
    }


def test_preview_is_read_only_transcript_free_and_reports_unsupported_platform_fields(
    tmp_path, monkeypatch
):
    path = source_file(tmp_path)
    before = xdg_files()
    monkeypatch.setattr(History, "__init__", lambda *args, **kwargs: pytest.fail("history opened"))
    report = backup_preview(path)
    assert xdg_files() == before
    assert report["format"] == "upstream"
    assert report["counts"] == {
        "history": 1,
        "profiles": 1,
        "routes": 0,
        "pending_routes": 0,
        "replacements": 1,
    }
    assert report["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "raw private words" not in json.dumps(report)
    assert "Final private words" not in json.dumps(report)
    assert "ai_enabled" not in report["changes"]
    assert report["pending_selections"]["selectedEditPromptID"]["profile"] == PROFILE_ID
    for field in (
        "hotkeyShortcut",
        "preferredInputDeviceUID",
        "pronunciationProfiles",
        "shareAnonymousAnalytics",
        "autoUpdateCheckEnabled",
        "commandModeConfirmBeforeExecute",
        "appPromptBindings[0]",
    ):
        assert field in report["unmapped"]


def test_upstream_import_preserves_history_metadata_and_merges_existing_data(tmp_path, monkeypatch):
    existing_id = str(uuid.uuid4())
    history = History()
    history.begin(existing_id)
    history.update(existing_id, "copied", "Existing transcript")
    data = Dictionary.defaults()
    data["replacements"] = [{"triggers": ["old"], "replacement": "Existing"}]
    Dictionary.save(data)
    VocabularyStore.save_entries([{"text": "Existing vocabulary"}])
    previous_words = VocabularyStore.path().read_bytes()
    # Upstream acoustic embeddings must not overwrite or clear this independent store.
    pronunciation_path = Dictionary.path().with_name("pronunciations.json")
    pronunciation_path.write_text("existing Linux data")
    Config(delivery="clipboard", command_confirm="on", launch_at_startup="on").save()
    monkeypatch.setattr(
        "mouthkeys.cli.apply_startup", lambda config: pytest.fail("startup touched")
    )
    monkeypatch.setattr(
        "mouthkeys.file_history.FileHistory.__init__",
        lambda *args, **kwargs: pytest.fail("absent file history opened"),
    )
    monkeypatch.setattr(
        "mouthkeys.assistant_history.AssistantHistory.__init__",
        lambda *args, **kwargs: pytest.fail("absent assistant history opened"),
    )
    path = source_file(tmp_path)
    app_map = {"com.example.Editor": "linux-editor"}
    report = backup_preview(path, app_map=app_map)
    result = backup_import(history, path, app_map=app_map, expected_sha256=report["file_sha256"])
    row = history.get(HISTORY_ID)
    assert row["created"] == datetime.fromisoformat("2026-11-01T20:00:00+00:00").timestamp()
    assert row["raw_text"] == "raw private words"
    assert row["text"] == "Final private words."
    assert row["app"] == "Editor" and row["window"] == "notes"
    assert row["was_ai_processed"] == 1
    assert row["ai_processing_model"] == "actual-cleanup-model"
    assert row["ai_processing_error"] is None
    assert row["duration"] == 2.5 and row["model"] == "whisper-medium"
    assert not history.recording(HISTORY_ID).exists()
    assert history.get(existing_id)["text"] == "Existing transcript"
    assert Dictionary.apply("old mouth keys") == "Existing MouthKeys"
    assert VocabularyStore.path().read_bytes() == previous_words
    assert pronunciation_path.read_text() == "existing Linux data"
    assert Config.load().model == "medium"
    assert Config.load().overlay_bottom_offset == 50
    assert Config.load().sound_volume == 25
    assert Config.load().command_confirm == "on"
    assert Config.load().launch_at_startup == "on"
    assert Config.load().restore_clipboard == "on"
    assert Config.load().insertion_mode == "clipboard"
    assert Config.load().pronunciation_matching == "off"
    assert load_profiles()[0]["ai_prompt"] == "Shorten selected text."
    assert load_profiles()[0]["ai_model"] == "existing-local-model"
    assert load_routes() == []
    assert result["migration"]["counts"]["routes"] == 0
    assert result["migration"]["counts"]["pending_routes"] == 1
    assert result["migration"]["pending_bindings"][0]["profile"] == PROFILE_ID
    assert select_prompt(Config.load(), app="linux-editor", mode="edit").ai_enabled == "off"
    backup_import(history, path, app_map=app_map)
    assert len(history.all_rows()) == 2


def test_import_without_audio_does_not_mislabel_ai_model_as_speech_model(tmp_path):
    document = upstream_document()
    document["transcriptionHistory"][0].pop("audio")
    history = History()
    backup_import(history, source_file(tmp_path, document))
    row = history.get(HISTORY_ID)
    assert row["model"] == "" and row["duration"] == 0
    assert row["ai_processing_model"] == "actual-cleanup-model"


def test_apple_engine_device_shortcuts_and_credentials_are_not_guessed(tmp_path):
    document = upstream_document()
    document["settings"].update(
        selectedSpeechModel="apple-speech-analyzer",
        selectedProviderID="custom-private",
        selectedModelByProvider={"custom:custom-private": "real-model"},
        savedProviders=[
            {
                "id": "custom-private",
                "baseURL": "http://localhost:1234/v1",
                "apiKey": "MUST-NOT-IMPORT",
                "name": "Private",
                "models": ["real-model"],
            }
        ],
    )
    report = backup_preview(source_file(tmp_path, document))
    assert "selectedSpeechModel" in report["unmapped"]
    assert "engine" not in report["changes"] and "model" not in report["changes"]
    assert "MUST-NOT-IMPORT" not in json.dumps(report)
    assert "ai_endpoint" not in report["changes"]
    assert report["pending_profiles"][0]["endpoint"] == "http://localhost:1234/v1/chat/completions"


def test_unresolved_provider_profiles_reported_without_model_placeholders(tmp_path):
    document = upstream_document()
    document["settings"]["selectedModelByProvider"] = {}
    report = backup_preview(source_file(tmp_path, document))
    assert report["counts"]["profiles"] == 0
    assert "promptProfiles" in report["unmapped"]
    assert "ai_model" not in report["changes"]


@pytest.mark.parametrize("damage", ["version", "date", "history", "settings", "model-type"])
def test_invalid_migration_rejected_before_mutation(tmp_path, damage):
    document = upstream_document()
    if damage == "version":
        document["schemaVersion"]["major"] = 2
    elif damage == "date":
        document["transcriptionHistory"][0]["timestamp"] = "2026-11-01T12:00:00"
    elif damage == "history":
        document["transcriptionHistory"][0]["wasAIProcessed"] = "true"
    elif damage == "settings":
        document["settings"]["removeFillerWordsEnabled"] = "true"
    else:
        document["transcriptionHistory"][0]["processingModel"] = []
    history = History()
    path = source_file(tmp_path, document)
    before = xdg_files()
    with pytest.raises(ValueError):
        backup_import(history, path)
    assert xdg_files() == before


def test_hash_mismatch_refuses_changed_file_before_writes(tmp_path):
    path = source_file(tmp_path)
    report = backup_preview(path)
    document = upstream_document()
    document["settings"]["fillerWords"] = ["changed"]
    path.write_text(json.dumps(document))
    before = xdg_files()
    with pytest.raises(ValueError, match="changed since preview"):
        backup_import(None, path, expected_sha256=report["file_sha256"])
    assert xdg_files() == before


def test_preview_service_action_allowed_while_recording_without_touching_history(tmp_path):
    report = handle_action(
        None, "backup_preview", {"path": source_file(tmp_path)}, allow_mutation=False
    )
    assert report["format"] == "upstream"


def test_read_bounds_actual_bytes_and_pure_conversion_does_not_mutate_arguments(
    tmp_path, monkeypatch
):
    from mouthkeys import upstream_backup

    monkeypatch.setattr(upstream_backup, "MAX_BACKUP_BYTES", 4)
    with pytest.raises(ValueError, match="100 MiB"):
        read_backup(source_file(tmp_path))
    document = upstream_document()
    original = copy.deepcopy(document)
    dictionary = Dictionary.defaults()
    before = copy.deepcopy(dictionary)
    convert_upstream_backup(document, Config(), dictionary, [], [])
    assert document == original and dictionary == before


def test_imported_drafts_do_not_change_active_profiles_providers_or_app_routes():
    base = Config(
        ai_enabled="on",
        ai_endpoint="http://localhost:9999/v1/chat/completions",
        ai_model="old-local-model",
        ai_profile=PROFILE_ID,
        edit_profile=PROFILE_ID,
        ai_key_env="OLD_KEY",
        ai_models_endpoint="http://localhost:9999/v1/models",
        ai_reasoning_parameter="reasoning_effort",
        ai_reasoning_value="low",
    )
    existing = {
        "id": PROFILE_ID,
        "name": "Existing",
        "shortcut": "",
        **{field: getattr(base, field) for field in PROFILE_FIELDS},
    }
    routes = [{"app": "linux-editor", "profile": PROFILE_ID, "mode": "edit"}]
    converted, report = convert_upstream_backup(
        upstream_document(),
        base,
        Dictionary.defaults(),
        [existing],
        routes,
        app_map={"com.example.Editor": "linux-editor"},
    )
    draft = converted["profiles"][0]
    assert draft["id"] != PROFILE_ID
    assert report["profile_id_map"][PROFILE_ID] == draft["id"]
    assert converted["prompt_routes"] == routes
    effective = replace(base, **converted["config"])
    for key in (
        "ai_enabled",
        "ai_endpoint",
        "ai_model",
        "ai_profile",
        "edit_profile",
        "ai_key_env",
        "ai_models_endpoint",
        "ai_reasoning_parameter",
        "ai_reasoning_value",
    ):
        assert getattr(effective, key) == getattr(base, key)
    selected = select_prompt(
        effective,
        app="linux-editor",
        mode="edit",
        profiles=[existing, draft],
        routes=converted["prompt_routes"],
    )
    assert selected.ai_endpoint == base.ai_endpoint and selected.ai_model == base.ai_model
    again, _ = convert_upstream_backup(
        upstream_document(),
        base,
        Dictionary.defaults(),
        [existing, draft],
        routes,
        app_map={"com.example.Editor": "linux-editor"},
    )
    assert again["profiles"][0]["id"] == draft["id"]


@pytest.mark.parametrize(
    "endpoint,credential_source",
    [
        ("https://new-provider.example/v1", "secret-service"),
        ("http://localhost:1234/v1", "environment"),
    ],
)
def test_imported_endpoint_never_reuses_environment_key_or_unrelated_discovery(
    monkeypatch, endpoint, credential_source
):
    from mouthkeys import credentials
    from mouthkeys.ai_providers import request_headers, request_json

    monkeypatch.setenv("MOUTHKEYS_AI_API_KEY", "unrelated-secret")
    document = upstream_document()
    document["settings"].update(
        selectedProviderID="custom",
        selectedModelByProvider={"custom:custom": "real-model"},
        savedProviders=[{"id": "custom", "baseURL": endpoint, "apiKey": "exported-secret"}],
    )
    base = Config(
        ai_endpoint="https://old-provider.example/v1/chat/completions",
        ai_models_endpoint="https://old-provider.example/v1/models",
        ai_reasoning_parameter="reasoning_effort",
        ai_reasoning_value="high",
    )
    converted, _ = convert_upstream_backup(document, base, Dictionary.defaults(), [], [])
    draft = converted["profiles"][0]
    assert draft["ai_key_env"] == ""
    assert draft["ai_credential_source"] == credential_source
    assert (
        draft["ai_models_endpoint"]
        == draft["ai_reasoning_parameter"]
        == draft["ai_reasoning_value"]
        == ""
    )
    effective = select_prompt(base, override=draft["id"], mode="edit", profiles=[draft], routes=[])
    if credential_source == "secret-service":
        lookups = []

        def absent_key(config, action, **kwargs):
            lookups.append(credentials._attributes(config))
            return b""

        monkeypatch.setattr(credentials, "_run", absent_key)
        with pytest.raises(RuntimeError, match="No API key saved"):
            request_json(
                effective.ai_endpoint,
                effective,
                body={},
                opener=lambda *args, **kwargs: pytest.fail("HTTP contacted"),
            )
        assert lookups[0][-1] == endpoint + "/chat/completions"
    else:
        assert request_headers(effective, effective.ai_provider) == {
            "Content-Type": "application/json"
        }


def test_upstream_gib_budget_never_rounds_down_retention():
    document = upstream_document()
    document["settings"]["audioHistoryBudgetGB"] = 4
    converted, _ = convert_upstream_backup(document, Config(), Dictionary.defaults(), [], [])
    assert converted["config"]["audio_budget_mb"] == 4295
    assert converted["config"]["audio_budget_mb"] * 1_000_000 >= 4 * 1_073_741_824


@pytest.mark.parametrize("aliases", [[], ["", " ", "\t", " \n "]])
def test_disabled_empty_spoken_action_disables_existing_aliases(aliases):
    from mouthkeys.formatting import format_text

    document = upstream_document()
    document["settings"]["spokenFormattingActionRules"] = [
        {"action": "newLine", "aliases": aliases, "isEnabled": False},
        {"action": "newLine", "aliases": ["try to re-enable"], "isEnabled": True},
    ]
    converted, _ = convert_upstream_backup(document, Config(), Dictionary.defaults(), [], [])
    actions = converted["dictionary"]["actions"]
    assert all(not row.get("enabled", True) for row in actions if row["symbol"] == "\n")
    assert not any("try to re-enable" in row["aliases"] for row in actions)
    assert (
        format_text(
            "before literal new line after",
            Config(spoken_formatting="on"),
            dictionary=converted["dictionary"],
        )
        == "before literal new line after"
    )


def test_disabled_empty_action_without_native_row_is_reported_not_invented():
    document = upstream_document()
    document["settings"]["spokenFormattingActionRules"] = [
        {"action": "newLine", "aliases": [], "isEnabled": False},
    ]
    dictionary = Dictionary.defaults()
    dictionary["actions"] = []
    converted, report = convert_upstream_backup(document, Config(), dictionary, [], [])
    assert converted["dictionary"]["actions"] == []
    assert "spokenFormattingActionRules[0]" in report["unmapped"]
