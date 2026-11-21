"""Parse the settings and transcript records in offline upstream backups."""

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeGuard


@dataclass
class MigrationNotes:
    handled: set[str] = field(default_factory=set)
    unmapped: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def skip(self, name: str, reason: str) -> None:
        self.handled.add(name)
        self.unmapped[name] = reason
        self.warnings.append(f"{name}: {reason}")


def read_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("Invalid upstream backup fields")
    return value


def read_list(document: dict[str, object], name: str, limit: int) -> list[object]:
    value = document.get(name, [])
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"Invalid upstream {name}")
    return value


def read_date(value: object) -> float:
    if not isinstance(value, str):
        raise ValueError("Upstream dates must be ISO8601 strings")  # noqa: TRY004 - imported field
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Upstream dates must include a timezone")
    return parsed.timestamp()


def read_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Upstream {name} must be text")  # noqa: TRY004 - invalid imported field
    return value


def finite_number(value: object) -> TypeGuard[int | float]:
    return (type(value) is int or type(value) is float) and math.isfinite(value)


BOOL_SETTINGS = {
    "hideFromDockAndAppSwitcher": "hide_from_taskbar",
    "showMainWindowAtLoginLaunch": "show_window_at_login",
    "showThinkingTokens": "ai_show_thinking",
    "enableStreamingPreview": "live_preview",
    "skipSilentRecordingsEnabled": "skip_silent_recordings",
    "enableAIStreaming": "ai_streaming",
    "copyTranscriptionToClipboard": "copy_to_clipboard",
    "overlayOuterBorderEnabled": "overlay_border",
    "saveTranscriptionHistory": "save_history",
    "notifyAIProcessingFailures": "show_ai_alerts",
    "showMicrophoneChangeAlerts": "show_microphone_alerts",
    "showPasteCheckAlerts": "show_paste_alerts",
    "returnDictationToStartingField": "return_to_start",
    "spokenSendEnabled": "spoken_send",
    "spokenSendImmediatelyEnabled": "send_after_pause",
    "questionMarkShortcutEnabled": "question_mark_shortcut",
    "removeFillerWordsEnabled": "remove_filler_words",
    "autoConvertPunctuationEnabled": "spoken_formatting",
    "literalDictationFormattingEnabled": "literal_formatting",
    "pauseMediaDuringTranscription": "pause_media",
    "automaticDictionaryLearningEnabled": "automatic_learning",
    "vocabularyBoostingEnabled": "vocabulary_boosting",
    "fileTranscriptionSpeakerLabelsEnabled": "file_speaker_labels",
}
DIRECT_SETTINGS = {
    "overlayPosition": "overlay_position",
    "overlayBottomOffset": "overlay_bottom_offset",
    "transcriptionPreviewCharLimit": "preview_chars",
    "userTypingWPM": "typing_wpm",
    "spokenSendPhrase": "spoken_send_phrase",
    "punctuationDictionaryPrefix": "literal_prefix",
    "fileTranscriptionExpectedSpeakerCount": "file_speaker_count",
    "defaultDictationPromptOverride": "ai_prompt",
    "defaultEditPromptOverride": "edit_prompt",
    "hotkeyMode": "hotkey_mode",
    "transcriptionStartSound": "sound_preset",
}


def _mapped_settings(settings: dict[str, object], notes: MigrationNotes) -> dict[str, object]:
    changes: dict[str, object] = {}
    for source, destination in BOOL_SETTINGS.items():
        if settings.get(source) is None:
            continue
        if type(settings[source]) is not bool:
            raise ValueError(f"Upstream {source} must be boolean")
        changes[destination] = "on" if settings[source] else "off"
        notes.handled.add(source)
    for source, destination in DIRECT_SETTINGS.items():
        if settings.get(source) is not None:
            value = settings[source]
            if destination in (
                "overlay_bottom_offset",
                "preview_chars",
                "typing_wpm",
                "file_speaker_count",
            ):
                if not finite_number(value):
                    raise ValueError(f"Invalid upstream {source}")
                value = round(value)
            changes[destination] = value
            notes.handled.add(source)
    for source, fallback, destination in (
        ("gaavLowercaseFirstLetterEnabled", "gaavModeEnabled", "lowercase_first"),
        ("gaavRemoveTrailingPeriodEnabled", "gaavModeEnabled", "remove_trailing_period"),
        (
            "continuousDictationSpacingEnabled",
            "continuousDictationModeEnabled",
            "space_between_dictations",
        ),
        (
            "contextAwareCapitalizationEnabled",
            "continuousDictationModeEnabled",
            "smart_capitalization",
        ),
    ):
        value = settings.get(source, settings.get(fallback))
        if value is not None:
            if type(value) is not bool:
                raise ValueError(f"Upstream {source} must be boolean")
            changes[destination] = "on" if value else "off"
            notes.handled.update((source, fallback))
    return changes


def _special_settings(
    settings: dict[str, object], notes: MigrationNotes, changes: dict[str, object]
) -> None:
    if "weekendsDontBreakStreak" in settings:
        if type(settings["weekendsDontBreakStreak"]) is not bool:
            raise ValueError("Upstream weekendsDontBreakStreak must be boolean")
        changes["weekends_break_streak"] = "off" if settings["weekendsDontBreakStreak"] else "on"
        notes.handled.add("weekendsDontBreakStreak")
    if "fillerWords" in settings:
        words = read_list(settings, "fillerWords", 10000)
        changes["filler_words"] = ",".join(read_text(word, "filler word") for word in words)
        notes.handled.add("fillerWords")
    if "transcriptionSoundVolume" in settings:
        value = settings["transcriptionSoundVolume"]
        if not finite_number(value) or not 0 <= value <= 1:
            raise ValueError("Invalid upstream recording sound volume")
        changes["sound_volume"] = round(value * 100)
        notes.handled.add("transcriptionSoundVolume")
    send_key = settings.get("spokenSendKey")
    if isinstance(send_key, str) and send_key in ("enter", "shiftEnter"):
        changes["spoken_send_key"] = {"enter": "return", "shiftEnter": "shift+return"}[send_key]
        notes.handled.add("spokenSendKey")
    insertion = settings.get("textInsertionMode")
    if insertion in ("standard", "reliablePaste"):
        changes["insertion_mode"] = "direct" if insertion == "standard" else "clipboard"
        if insertion == "reliablePaste":
            changes["restore_clipboard"] = "on"
        notes.handled.add("textInsertionMode")


def _speech_route(
    settings: dict[str, object], notes: MigrationNotes, changes: dict[str, object]
) -> None:
    speech = settings.get("selectedSpeechModel")
    whisper = {
        "whisper-tiny": "tiny",
        "whisper-base": "base",
        "whisper-small": "small",
        "whisper-medium": "medium",
        "whisper-large-turbo": "large-v3-turbo",
        "whisper-large": "large-v3",
    }
    if isinstance(speech, str) and speech in whisper:
        changes.update(engine="whisper", model=whisper[speech], compute_type="default")
        notes.handled.add("selectedSpeechModel")
        notes.warnings.append(
            "Speech selection saves a Linux route only; weights are never downloaded or loaded."
        )
    elif speech is not None:
        notes.skip(
            "selectedSpeechModel",
            "Apple APIs/artifacts and non-Whisper routes require explicit Linux model selection",
        )


def convert_settings(settings: dict[str, object], notes: MigrationNotes) -> dict[str, object]:
    changes = _mapped_settings(settings, notes)
    _special_settings(settings, notes, changes)
    _speech_route(settings, notes, changes)
    return changes


def read_audio_metadata(audio: object) -> tuple[int | float, str]:
    if audio is None:
        return 0, ""
    if isinstance(audio, dict):
        milliseconds = audio.get("durationMilliseconds")
        if type(milliseconds) is int and milliseconds >= 0:
            return milliseconds / 1000, read_text(audio.get("model") or "", "audio model")
    raise ValueError("Invalid upstream audio metadata")


def convert_history(document: dict[str, object], notes: MigrationNotes) -> list[dict[str, object]]:
    history: list[dict[str, object]] = []
    seen_history: set[str] = set()
    has_audio = False
    for value in read_list(document, "transcriptionHistory", 1000000):
        row = read_object(value)
        identifier = str(uuid.UUID(read_text(row["id"], "history ID")))
        if identifier in seen_history:
            raise ValueError("Duplicate upstream history ID")
        seen_history.add(identifier)
        flag = row["wasAIProcessed"]
        if type(flag) is not bool:
            raise ValueError("Upstream wasAIProcessed must be boolean")
        audio = row.get("audio")
        duration, model = read_audio_metadata(audio)
        has_audio |= audio is not None
        error = row.get("aiProcessingError")
        if error is not None:
            read_text(error, "AI processing error")
        if row.get("processingModel") is not None:
            read_text(row["processingModel"], "AI processing model")
        history.append(
            {
                "id": identifier,
                "created": read_date(row["timestamp"]),
                "text": read_text(row["processedText"], "processed text"),
                "raw_text": read_text(row["rawText"], "raw text"),
                "app": read_text(row["appName"], "application name"),
                "window": read_text(row["windowTitle"], "window title"),
                "was_ai_processed": flag,
                "ai_processing_error": error,
                "ai_processing_model": row.get("processingModel"),
                "status": "transcribed",
                "error": "",
                "duration": duration,
                "model": model,
            }
        )
    if has_audio:
        notes.warnings.append(
            "Audio metadata/duration imported, not files; backup JSON has no audio bytes."
        )
    return history


def read_aliases(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("Upstream aliases must be a list")  # noqa: TRY004 - invalid imported field
    return list(
        dict.fromkeys(
            alias
            for item in value
            if (alias := " ".join(read_text(item, "alias").casefold().split()))
        )
    )


def read_app_mapping(app_map: object) -> dict[str, str]:
    app_map = {} if app_map is None else app_map
    if not isinstance(app_map, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or not v.strip()
        for k, v in app_map.items()
    ):
        raise ValueError("Application mapping must map macOS bundle IDs to Linux application names")
    return app_map


def backup_header(document: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    version = document.get("schemaVersion")
    if (
        not isinstance(version, dict)
        or type(version.get("major")) is not int
        or version["major"] != 1
    ):
        raise ValueError("Unsupported upstream backup schema version")
    if type(version.get("minor")) is not int or version["minor"] < 0:
        raise ValueError("Invalid upstream backup schema version")
    settings = document.get("settings")
    if not isinstance(settings, dict) or not all(isinstance(key, str) for key in settings):
        raise ValueError("Invalid upstream backup settings")
    if "exportedAt" in document:
        read_date(document["exportedAt"])
    return version, settings
