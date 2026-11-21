from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from mouthkeys.config import Config
from mouthkeys.dictionary import Dictionary
from mouthkeys.profiles import PROFILE_FIELDS
from mouthkeys.upstream_backup import convert_upstream_backup, read_backup

from .upstream_backup_fixture import HISTORY_ID, PROFILE_ID, upstream_document


def convert(document, *, config=None, profiles=None, app_map=None):
    return convert_upstream_backup(
        document, config or Config(), Dictionary.defaults(), profiles or [], [], app_map=app_map
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"overlayBottomOffset": float("nan")}, "Invalid upstream overlayBottomOffset"),
        ({"continuousDictationModeEnabled": 1}, "contextAwareCapitalizationEnabled|SpacingEnabled"),
        ({"weekendsDontBreakStreak": "yes"}, "weekendsDontBreakStreak must be boolean"),
        ({"transcriptionSoundVolume": 1.1}, "Invalid upstream recording sound volume"),
        ({"fillerWords": {}}, "Invalid upstream fillerWords"),
        ({"fillerWords": [3]}, "Upstream filler word must be text"),
        (
            {"spokenFormattingActionRules": [{"action": "newLine", "aliases": {}}]},
            "aliases must be a list",
        ),
        (
            {"spokenFormattingActionRules": [{"action": "newLine", "aliases": [], "isEnabled": 1}]},
            "enabled flag must be boolean",
        ),
        ({"editPromptOff": 1}, "editPromptOff must be boolean"),
    ],
)
def test_invalid_upstream_settings_are_not_coerced(changes, message):
    document = upstream_document()
    document["settings"].update(changes)
    with pytest.raises(ValueError, match=message):
        convert(document)


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"exportedAt": 42}, "dates must be ISO8601 strings"),
        ({"settings": []}, "Invalid upstream backup settings"),
        ({"schemaVersion": {"major": 1, "minor": -1}}, "Invalid upstream backup schema version"),
        ({"transcriptionHistory": [None]}, "Invalid upstream backup fields"),
    ],
)
def test_bad_backup_header_or_row_is_rejected(patch, message):
    with pytest.raises(ValueError, match=message):
        convert(upstream_document() | patch)


@pytest.mark.parametrize(
    "audio", [[], {"durationMilliseconds": -1}, {"durationMilliseconds": False}]
)
def test_invalid_audio_metadata_cannot_become_history(audio):
    document = upstream_document()
    document["transcriptionHistory"][0]["audio"] = audio
    with pytest.raises(ValueError, match="Invalid upstream audio metadata"):
        convert(document)


def test_history_duplicate_ids_and_nontext_ai_errors_are_rejected():
    document = upstream_document()
    row = document["transcriptionHistory"][0]
    document["transcriptionHistory"].append(row | {"id": HISTORY_ID.upper()})
    with pytest.raises(ValueError, match="Duplicate upstream history ID"):
        convert(document)
    document["transcriptionHistory"].pop()
    row["aiProcessingError"] = []
    with pytest.raises(ValueError, match="Upstream AI processing error must be text"):
        convert(document)


@pytest.mark.parametrize(
    ("key", "delivery", "shortcut"),
    [("enter", "standard", "return"), ("shiftEnter", "reliablePaste", "shift+return")],
)
def test_mapped_settings_preserve_explicit_choices_and_report_unavailable_apple_features(
    key, delivery, shortcut
):
    document = upstream_document()
    document.pop("exportedAt")
    document["settings"].update(
        gaavModeEnabled=True,
        gaavLowercaseFirstLetterEnabled=False,
        continuousDictationModeEnabled=True,
        spokenSendKey=key,
        textInsertionMode=delivery,
        pronunciationMatchingEnabled=True,
        dictationPromptOff=False,
        punctuationDictionaryRules=[
            {"aliases": ["  double  bang ", "Double bang"], "symbol": " ‼ "},
            {"aliases": [" "], "symbol": "x"},
        ],
        spokenFormattingActionRules=[
            {"action": "newLine", "aliases": ["paragraph divider"], "isEnabled": False}
        ],
    )
    document["transcriptionHistory"][0]["aiProcessingError"] = "Timed out"
    converted, report = convert(document)
    settings = converted["config"]
    assert settings["lowercase_first"] == "off"
    assert settings["remove_trailing_period"] == "on"
    assert settings["space_between_dictations"] == "on"
    assert settings["smart_capitalization"] == "on"
    assert settings["spoken_send_key"] == shortcut
    if delivery == "standard":
        assert settings["insertion_mode"] == "direct"
        assert "restore_clipboard" not in settings
    else:
        assert settings["insertion_mode"] == "clipboard"
        assert settings["restore_clipboard"] == "on"
    assert "pronunciation_matching" not in settings
    assert report["pending_selections"]["dictationPromptOff"] is False
    assert "pronunciationMatchingEnabled" in report["unmapped"]
    assert "punctuationDictionaryRules[1]" in report["unmapped"]
    assert {"aliases": ["double bang"], "symbol": "‼"} in converted["dictionary"]["punctuation"]
    assert {"aliases": ["paragraph divider"], "symbol": "\n", "enabled": False} in converted[
        "dictionary"
    ]["actions"]
    assert converted["history"][0]["ai_processing_error"] == "Timed out"
    assert "Timed out" not in str(report)


def test_empty_backup_without_provider_or_speech_keeps_existing_config():
    document = {"schemaVersion": {"major": 1, "minor": 0}, "settings": {}}
    converted, report = convert(document, config=Config(model="large-v3", ai_enabled="on"))
    assert converted["config"] == {}
    assert converted["history"] == []
    assert report["warnings"] == []
    assert report["unmapped"] == {}


def test_migration_keeps_enabled_action_aliases_and_unknown_model_metadata():
    document = upstream_document()
    document["promptProfiles"] = []
    document["transcriptionHistory"][0].pop("processingModel")
    document["settings"]["spokenFormattingActionRules"] = [{"action": "tab", "aliases": ["indent"]}]
    converted, report = convert(document)
    assert converted["profiles"] == []
    assert report["pending_profiles"] == []
    assert converted["history"][0]["ai_processing_model"] is None
    assert {"aliases": ["indent"], "symbol": "\t", "enabled": True} in converted["dictionary"][
        "actions"
    ]


@pytest.mark.parametrize("value", [[], {"com.example.Editor": " "}, {12: "editor"}])
def test_invalid_app_map_is_not_used_for_pending_bindings(value):
    with pytest.raises(ValueError, match="Application mapping must map"):
        convert(upstream_document(), app_map=value)


def test_valid_profile_defaults_without_imported_selections_do_not_change_the_base():
    document = upstream_document()
    document["settings"]["selectedEditPromptID"] = None
    document["settings"]["selectedDictationPromptID"] = "bd09ebde-212d-4156-a64a-ba56b8d3c3f5"
    document["settings"].pop("pronunciationMatchingEnabled")
    document["appPromptBindings"][0]["mode"] = "unrecognized"
    converted, report = convert(document, app_map={"com.example.Editor": "editor"})
    assert converted["prompt_routes"] == []
    assert report["pending_bindings"] == []
    assert "selectedEditPromptID" not in report["pending_selections"]
    assert "selectedDictationPromptID" in report["unmapped"]
    assert "pronunciationMatchingEnabled" not in report["unmapped"]


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("duplicate", "Duplicate upstream prompt profile ID"),
        ("mode", "Invalid upstream prompt mode"),
    ],
)
def test_invalid_upstream_profiles_are_rejected_before_install(damage, message):
    document = upstream_document()
    draft = document["promptProfiles"][0]
    if damage == "duplicate":
        document["promptProfiles"].append(deepcopy(draft))
    else:
        draft["mode"] = "command"
    with pytest.raises(ValueError, match=message):
        convert(document)


def test_migration_never_overwrites_a_conflicting_derived_profile():
    base = replace(Config(), ai_model="original")
    data = asdict(base)
    existing = {"id": PROFILE_ID, "name": "Existing", **{key: data[key] for key in PROFILE_FIELDS}}
    converted, _ = convert(upstream_document(), config=base, profiles=[existing])
    draft = converted["profiles"][0]
    assert draft["id"] != PROFILE_ID
    other = draft | {"name": "Existing different draft"}
    with pytest.raises(ValueError, match="Imported prompt profile ID conflicts"):
        convert(upstream_document(), config=base, profiles=[existing, other])


def test_invalid_provider_configuration_is_reported_without_activating_profiles():
    document = upstream_document()
    document["settings"]["selectedModelByProvider"] = []
    converted, report = convert(document)
    assert converted["profiles"] == []
    assert "endpoint/model unsupported" in report["unmapped"]["provider"]


def test_explicit_loopback_endpoint_is_not_rewritten_when_already_complete():
    document = upstream_document()
    endpoint = "http://[::1]:11434/v1/chat/completions"
    document["settings"]["savedProviders"] = [{"id": "ollama", "baseURL": endpoint}]
    converted, _ = convert(document)
    assert converted["profiles"][0]["ai_endpoint"] == endpoint
    assert converted["profiles"][0]["ai_credential_source"] == "environment"
    assert converted["profiles"][0]["ai_key_env"] == ""


def test_reader_rejects_json_array(tmp_path):
    path = tmp_path / "backup.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="Backup must be a JSON object"):
        read_backup(path)
