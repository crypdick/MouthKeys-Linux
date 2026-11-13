"""Offline migration of upstream macOS JSON backups, with explicit loss reporting.

Source contract: BenevolentFutures/MouthKeys d50ad008 BackupService.swift.
Conversion never opens devices, resolves credentials, loads models or contacts providers.
"""

import copy
import hashlib
import ipaddress
import json
import math
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from .config import Config
from .dictionary import Dictionary
from .dictionary_transfer import _merge_rules, merge_replacements, replacement_entries
from .profiles import PROFILE_FIELDS, load_profiles, validate_profiles
from .routing import load_routes, validate_routes

MAX_BACKUP_BYTES = 100 * 1024 * 1024

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
    "saveAudioWithTranscriptionHistory": "save_audio",
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
# URLs match upstream ModelRepository, not discovery results or model guesses.
PROVIDERS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "xai": "https://api.x.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
}


def read_backup(path, *, with_digest=False):
    with Path(path).open("rb") as source:
        raw = source.read(MAX_BACKUP_BYTES + 1)
    if len(raw) > MAX_BACKUP_BYTES:
        raise ValueError("Backup exceeds 100 MiB")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Backup must be a JSON object")
    return (data, hashlib.sha256(raw).hexdigest()) if with_digest else data


def _list(document, name, limit):
    value = document.get(name, [])
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"Invalid upstream {name}")
    return value


def _date(value):
    if not isinstance(value, str):
        raise ValueError("Upstream dates must be ISO8601 strings")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Upstream dates must include a timezone")
    result = parsed.timestamp()
    if not math.isfinite(result):
        raise ValueError("Invalid upstream date")
    return result


def _text(value, name):
    if not isinstance(value, str):
        raise ValueError(f"Upstream {name} must be text")
    return value


def _aliases(value):
    if not isinstance(value, list):
        raise ValueError("Upstream aliases must be a list")
    return list(
        dict.fromkeys(
            alias for item in value if (alias := " ".join(_text(item, "alias").casefold().split()))
        )
    )


def _provider(settings):
    provider_id = settings.get("selectedProviderID")
    if not isinstance(provider_id, str):
        return None
    providers = _list(settings, "savedProviders", 100)
    saved = next((p for p in providers if isinstance(p, dict) and p.get("id") == provider_id), None)
    base = saved.get("baseURL") if saved else PROVIDERS.get(provider_id)
    models = settings.get("selectedModelByProvider", {})
    if not isinstance(models, dict):
        raise ValueError("Invalid upstream selectedModelByProvider")
    model = models.get(provider_id) or models.get("custom:" + provider_id)
    if not isinstance(base, str) or not isinstance(model, str) or not model.strip():
        return None
    anthropic = provider_id == "anthropic" or urlsplit(base).hostname == "api.anthropic.com"
    path = "/messages" if anthropic else "/chat/completions"
    endpoint = base.rstrip("/")
    if not endpoint.endswith(path):
        endpoint += path
    host = urlsplit(endpoint).hostname
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = host == "localhost"
    config = Config(
        ai_provider="anthropic" if anthropic else "compatible",
        ai_endpoint=endpoint,
        ai_model=model,
        ai_key_env="",
        ai_credential_source="environment" if local else "secret-service",
    )
    config.validate()
    return config


def _convert_upstream_backup(
    document, base_config, current_dictionary, current_profiles, current_routes, *, app_map=None
):
    """Build native import data plus transcript-free report; do not write any state."""
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
    if not isinstance(settings, dict):
        raise ValueError("Invalid upstream backup settings")
    if "exportedAt" in document:
        _date(document["exportedAt"])
    warnings, unmapped, changes = [], {}, {}
    handled = set()

    def skip(field, reason):
        handled.add(field)
        unmapped[field] = reason
        warnings.append(f"{field}: {reason}")

    for source, destination in BOOL_SETTINGS.items():
        if settings.get(source) is None:
            continue
        if type(settings[source]) is not bool:
            raise ValueError(f"Upstream {source} must be boolean")
        changes[destination] = "on" if settings[source] else "off"
        handled.add(source)
    for source, destination in DIRECT_SETTINGS.items():
        if settings.get(source) is not None:
            value = settings[source]
            if destination in (
                "overlay_bottom_offset",
                "preview_chars",
                "typing_wpm",
                "file_speaker_count",
            ):
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f"Invalid upstream {source}")
                value = round(value)
            changes[destination] = value
            handled.add(source)
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
            handled.update((source, fallback))
    if "weekendsDontBreakStreak" in settings:
        if type(settings["weekendsDontBreakStreak"]) is not bool:
            raise ValueError("Upstream weekendsDontBreakStreak must be boolean")
        changes["weekends_break_streak"] = "off" if settings["weekendsDontBreakStreak"] else "on"
        handled.add("weekendsDontBreakStreak")
    if "fillerWords" in settings:
        words = _list(settings, "fillerWords", 10000)
        changes["filler_words"] = ",".join(_text(word, "filler word") for word in words)
        handled.add("fillerWords")
    if "transcriptionSoundVolume" in settings:
        value = settings["transcriptionSoundVolume"]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid upstream recording sound volume")
        changes["sound_volume"] = round(value * 100)
        handled.add("transcriptionSoundVolume")
    if settings.get("audioHistoryBudgetGB") is not None:
        value = settings["audioHistoryBudgetGB"]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid upstream audio budget")
        # Upstream GiB; Linux pruning uses decimal MB. Never round retention down.
        changes["audio_budget_mb"] = math.ceil(value * 1_073_741_824 / 1_000_000)
        handled.add("audioHistoryBudgetGB")
    if settings.get("spokenSendKey") in ("enter", "shiftEnter"):
        changes["spoken_send_key"] = {"enter": "return", "shiftEnter": "shift+return"}[
            settings["spokenSendKey"]
        ]
        handled.add("spokenSendKey")
    insertion = settings.get("textInsertionMode")
    if insertion in ("standard", "reliablePaste"):
        changes["insertion_mode"] = "direct" if insertion == "standard" else "clipboard"
        if insertion == "reliablePaste":
            changes["restore_clipboard"] = "on"
        handled.add("textInsertionMode")
    speech = settings.get("selectedSpeechModel")
    whisper = {
        "whisper-tiny": "tiny",
        "whisper-base": "base",
        "whisper-small": "small",
        "whisper-medium": "medium",
        "whisper-large-turbo": "large-v3-turbo",
        "whisper-large": "large-v3",
    }
    if speech in whisper:
        changes.update(engine="whisper", model=whisper[speech], compute_type="default")
        handled.add("selectedSpeechModel")
        warnings.append(
            "Speech selection saves a Linux route only; weights are never downloaded or loaded."
        )
    elif speech is not None:
        skip(
            "selectedSpeechModel",
            "Apple APIs/artifacts and non-Whisper routes require explicit Linux model selection",
        )

    dictionary = copy.deepcopy(Dictionary.validate(current_dictionary))
    replacements = replacement_entries(_list(settings, "customDictionaryEntries", 10000))
    dictionary["replacements"] = merge_replacements(dictionary["replacements"], replacements)
    handled.add("customDictionaryEntries")
    if settings.get("punctuationDictionaryRules") is not None:
        rows = _list(settings, "punctuationDictionaryRules", 10000)
        rules = []
        for index, row in enumerate(rows):
            aliases = _aliases(row["aliases"])
            symbol = _text(row["symbol"], "symbol").strip()
            if not aliases or not symbol:
                skip(
                    f"punctuationDictionaryRules[{index}]",
                    "Empty punctuation row omitted like upstream normalization",
                )
                continue
            rules.append({"aliases": aliases, "symbol": symbol})
        dictionary["punctuation"] = _merge_rules(dictionary["punctuation"], rules)
        handled.add("punctuationDictionaryRules")
    if settings.get("spokenFormattingActionRules") is not None:
        symbols = {"newLine": "\n", "newParagraph": "\n\n", "tab": "\t", "space": " "}
        seen_actions = set()
        for index, row in enumerate(_list(settings, "spokenFormattingActionRules", 10000)):
            action = row["action"]
            symbol = symbols[action]
            if action in seen_actions:
                continue  # Upstream keeps the first row for each action.
            seen_actions.add(action)
            aliases = _aliases(row["aliases"])
            enabled = row.get("isEnabled", True)
            if type(enabled) is not bool:
                raise ValueError("Upstream action enabled flag must be boolean")
            enabled = enabled and bool(aliases)
            if not enabled:
                matching = [entry for entry in dictionary["actions"] if entry["symbol"] == symbol]
                for entry in matching:
                    entry["enabled"] = False
                if not aliases and not matching:
                    skip(
                        f"spokenFormattingActionRules[{index}]",
                        "Disabled action has no native aliases; remains unavailable",
                    )
            if aliases:
                dictionary["actions"] = _merge_rules(
                    dictionary["actions"],
                    [
                        {
                            "aliases": aliases,
                            "symbol": symbol,
                            "enabled": enabled,
                        }
                    ],
                )
        handled.add("spokenFormattingActionRules")
    Dictionary.validate(dictionary)

    profiles, profile_id_map, pending_profiles = [], {}, []
    current_by_id = {profile["id"]: profile for profile in current_profiles}
    reserved = {base_config.ai_profile, base_config.edit_profile, base_config.command_profile}
    reserved.update(rule["profile"] for rule in current_routes)
    try:
        provider = _provider(settings)
    except (ValueError, TypeError):
        provider = None
        skip(
            "provider",
            "Provider endpoint/model unsupported; credentials and endpoint not imported",
        )
    upstream_profiles = _list(document, "promptProfiles", 100)
    if provider is not None:
        handled.update(("selectedProviderID", "selectedModelByProvider", "savedProviders"))
        warnings.append(
            "Imported AI profiles stay inactive; existing Linux providers/selections unchanged. "
            "Review credentials and explicitly select a profile/add a route in Settings."
        )
        for profile in upstream_profiles:
            identifier = str(uuid.UUID(profile["id"]))
            if identifier in profile_id_map:
                raise ValueError("Duplicate upstream prompt profile ID")
            mode = {"dictate": "dictation", "edit": "edit", "write": "edit", "rewrite": "edit"}.get(
                profile.get("mode", "dictate")
            )
            if mode is None:
                raise ValueError("Invalid upstream prompt mode")
            draft = {
                "id": identifier,
                "name": profile["name"],
                "shortcut": "",
                **{key: getattr(provider, key) for key in PROFILE_FIELDS},
                "ai_prompt": profile["prompt"],
            }
            existing = current_by_id.get(identifier)
            if existing != draft and (existing is not None or identifier in reserved):
                serialized = json.dumps(draft, sort_keys=True, ensure_ascii=False)
                draft["id"] = str(
                    uuid.uuid5(uuid.NAMESPACE_URL, "mouthkeys:upstream:" + serialized)
                )
                if draft["id"] in current_by_id and current_by_id[draft["id"]] != draft:
                    raise ValueError("Imported prompt profile ID conflicts with existing data")
            profile_id_map[identifier] = draft["id"]
            profiles.append(draft)
            pending_profiles.append(
                {
                    "source_id": identifier,
                    "id": draft["id"],
                    "name": profile["name"],
                    "mode": mode,
                    "provider": provider.ai_provider,
                    "endpoint": provider.ai_endpoint,
                    "model": provider.ai_model,
                    "credential_source": provider.ai_credential_source,
                }
            )
        profiles = validate_profiles(profiles)
        if profiles:
            warnings.append(
                "Prompts use global provider/model; backup omits per-profile provider pins."
            )
            warnings.append(
                "Profile dates/context flags omitted; source selections/routes shown as pending."
            )
    elif upstream_profiles:
        skip(
            "promptProfiles",
            "No supported selected provider/model; prompt profiles remain in original backup",
        )
    pending_selections = {}
    for source in ("selectedDictationPromptID", "selectedEditPromptID"):
        identifier = settings.get(source)
        if identifier and str(uuid.UUID(identifier)) in profile_id_map:
            identifier = str(uuid.UUID(identifier))
            pending_selections[source] = {
                "source_id": identifier,
                "profile": profile_id_map[identifier],
            }
            handled.add(source)
    for source in ("editPromptOff", "dictationPromptOff"):
        if settings.get(source) is not None:
            if type(settings[source]) is not bool:
                raise ValueError(f"Upstream {source} must be boolean")
            pending_selections[source] = settings[source]
            handled.add(source)
    scope = settings.get("dictationPromptRoutingScope")
    if scope in ("allApps", "selectedAppsOnly"):
        pending_selections["dictationPromptRoutingScope"] = scope
        handled.add("dictationPromptRoutingScope")

    app_map = {} if app_map is None else app_map
    if not isinstance(app_map, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or not v.strip()
        for k, v in app_map.items()
    ):
        raise ValueError("Application mapping must map macOS bundle IDs to Linux application names")
    routes = copy.deepcopy(current_routes)
    pending_bindings = []
    for index, binding in enumerate(_list(document, "appPromptBindings", 1000)):
        bundle = binding.get("appBundleID", "")
        mode = {"dictate": "dictation", "edit": "edit", "write": "edit", "rewrite": "edit"}.get(
            binding.get("mode", "dictate")
        )
        selected = binding.get("promptID")
        source_id = str(uuid.UUID(selected)) if selected else "default"
        identifier = profile_id_map.get(source_id, "default" if not selected else None)
        app = app_map.get(bundle)
        if not app or mode is None or identifier is None:
            skip(
                f"appPromptBindings[{index}]",
                "Requires explicit Linux application mapping and an imported prompt",
            )
            continue
        pending_bindings.append(
            {
                "source_app": bundle,
                "source_profile": source_id,
                "app": app,
                "profile": identifier,
                "mode": mode,
            }
        )
    combined = {p["id"]: p for p in current_profiles}
    combined.update({p["id"]: p for p in profiles})
    validate_routes(routes, validate_profiles(list(combined.values())))

    history, seen_history = [], set()
    for row in _list(document, "transcriptionHistory", 1000000):
        identifier = str(uuid.UUID(row["id"]))
        if identifier in seen_history:
            raise ValueError("Duplicate upstream history ID")
        seen_history.add(identifier)
        flag = row["wasAIProcessed"]
        if type(flag) is not bool:
            raise ValueError("Upstream wasAIProcessed must be boolean")
        audio = row.get("audio")
        duration, model = 0, ""
        if audio is not None:
            if (
                not isinstance(audio, dict)
                or type(audio.get("durationMilliseconds")) is not int
                or audio["durationMilliseconds"] < 0
            ):
                raise ValueError("Invalid upstream audio metadata")
            duration = audio["durationMilliseconds"] / 1000
            model = audio.get("model") or ""
        error = row.get("aiProcessingError")
        if error is not None:
            _text(error, "AI processing error")
        if row.get("processingModel") is not None:
            _text(row["processingModel"], "AI processing model")
        history.append(
            {
                "id": identifier,
                "created": _date(row["timestamp"]),
                "text": _text(row["processedText"], "processed text"),
                "raw_text": _text(row["rawText"], "raw text"),
                "app": _text(row["appName"], "application name"),
                "window": _text(row["windowTitle"], "window title"),
                "was_ai_processed": flag,
                "ai_processing_error": error,
                "ai_processing_model": row.get("processingModel"),
                "status": "transcribed",
                "error": "",
                "duration": duration,
                "model": _text(model, "audio model"),
            }
        )
    if any(row.get("audio") is not None for row in document.get("transcriptionHistory", [])):
        warnings.append(
            "Audio metadata/duration imported, not files; backup JSON has no audio bytes."
        )
    if document.get("pronunciationProfiles"):
        skip(
            "pronunciationProfiles",
            "Apple embeddings cannot become Linux acoustic templates; existing profiles unchanged",
        )
    if settings.get("pronunciationMatchingEnabled"):
        skip(
            "pronunciationMatchingEnabled",
            "Re-enroll Linux pronunciation takes before enabling matching",
        )
    elif settings.get("pronunciationMatchingEnabled") is False:
        changes["pronunciation_matching"] = "off"
        handled.add("pronunciationMatchingEnabled")
    for field, value in settings.items():
        if field not in handled and value is not None:
            skip(field, "No equivalent safe automatic mapping; existing Linux setting retained")
    # Config validation enforces native ranges without truncating or inventing settings.
    replace(base_config, **changes).validate()
    backup = {
        "version": 1,
        "config": changes,
        "dictionary": dictionary,
        "profiles": profiles,
        "history": history,
        "prompt_routes": routes,
    }
    report = {
        "format": "upstream",
        "schema_version": version,
        "counts": {
            "history": len(history),
            "profiles": len(profiles),
            "routes": 0,
            "pending_routes": len(pending_bindings),
            "replacements": len(replacements),
        },
        "changes": changes,
        "warnings": warnings,
        "unmapped": unmapped,
        "pending_profiles": pending_profiles,
        "pending_selections": pending_selections,
        "pending_bindings": pending_bindings,
        "profile_id_map": profile_id_map,
    }
    return backup, report


def convert_upstream_backup(
    document, base_config, current_dictionary, current_profiles, current_routes, *, app_map=None
):
    try:
        return _convert_upstream_backup(
            document,
            base_config,
            current_dictionary,
            current_profiles,
            current_routes,
            app_map=app_map,
        )
    except (KeyError, TypeError, AttributeError, OverflowError):
        raise ValueError("Invalid upstream backup fields") from None


def load_upstream_backup(path, *, app_map=None, document=None):
    """Read snapshots only: unlike store helpers, never create lock files or databases."""
    dictionary_path = Dictionary.path()
    dictionary = read_backup(dictionary_path) if dictionary_path.exists() else Dictionary.defaults()
    return convert_upstream_backup(
        read_backup(path) if document is None else document,
        Config.load(),
        dictionary,
        load_profiles(),
        load_routes(),
        app_map=app_map,
    )


def preview_upstream_backup(path, *, app_map=None):
    from .operations import _prepare_backup

    document, digest = read_backup(path, with_digest=True)
    converted, report = load_upstream_backup(path, app_map=app_map, document=document)
    _prepare_backup(converted)
    return {**report, "file_sha256": digest}
