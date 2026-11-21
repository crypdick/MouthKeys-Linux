"""Offline migration of upstream macOS JSON backups, with explicit loss reporting.

Source contract: BenevolentFutures/MouthKeys d50ad008 BackupService.swift.
Conversion never opens devices, resolves credentials, loads models or contacts providers.
"""

import copy
import hashlib
import ipaddress
import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal, overload
from urllib.parse import urlsplit

from .config import Config
from .dictionary import Dictionary, DictionaryData
from .dictionary_transfer import _merge_rules, merge_replacements, replacement_entries
from .profiles import PROFILE_FIELDS, PromptProfile, load_profiles, validate_profiles
from .routing import load_routes, validate_routes
from .upstream_backup_sections import (
    MigrationNotes,
    backup_header,
    convert_history,
    convert_settings,
    read_aliases,
    read_app_mapping,
    read_list,
    read_object,
    read_text,
)

MAX_BACKUP_BYTES = 100 * 1024 * 1024

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


@overload
def read_backup(path: str | Path, *, with_digest: Literal[False] = False) -> dict[str, object]: ...


@overload
def read_backup(
    path: str | Path, *, with_digest: Literal[True]
) -> tuple[dict[str, object], str]: ...


def read_backup(
    path: str | Path, *, with_digest: bool = False
) -> dict[str, object] | tuple[dict[str, object], str]:
    with Path(path).open("rb") as source:
        raw = source.read(MAX_BACKUP_BYTES + 1)
    if len(raw) > MAX_BACKUP_BYTES:
        raise ValueError("Backup exceeds 100 MiB")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Backup must be a JSON object")
    return (data, hashlib.sha256(raw).hexdigest()) if with_digest else data


def _provider(settings: dict[str, object]) -> Config | None:
    provider_id = settings.get("selectedProviderID")
    if not isinstance(provider_id, str):
        return None
    providers = read_list(settings, "savedProviders", 100)
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
        local = host is not None and ipaddress.ip_address(host).is_loopback
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


def _convert_dictionary(
    settings: dict[str, object], current_dictionary: object, notes: MigrationNotes
) -> tuple[DictionaryData, int]:
    dictionary = copy.deepcopy(Dictionary.validate(current_dictionary))
    replacements = replacement_entries(read_list(settings, "customDictionaryEntries", 10000))
    dictionary["replacements"] = merge_replacements(dictionary["replacements"], replacements)
    notes.handled.add("customDictionaryEntries")
    if settings.get("punctuationDictionaryRules") is not None:
        rows = read_list(settings, "punctuationDictionaryRules", 10000)
        rules = []
        for index, value in enumerate(rows):
            row = read_object(value)
            aliases = read_aliases(row["aliases"])
            symbol = read_text(row["symbol"], "symbol").strip()
            if not aliases or not symbol:
                notes.skip(
                    f"punctuationDictionaryRules[{index}]",
                    "Empty punctuation row omitted like upstream normalization",
                )
                continue
            rules.append({"aliases": aliases, "symbol": symbol})
        dictionary["punctuation"] = _merge_rules(dictionary["punctuation"], rules)
        notes.handled.add("punctuationDictionaryRules")
    if settings.get("spokenFormattingActionRules") is not None:
        symbols = {"newLine": "\n", "newParagraph": "\n\n", "tab": "\t", "space": " "}
        seen_actions = set()
        for index, value in enumerate(read_list(settings, "spokenFormattingActionRules", 10000)):
            row = read_object(value)
            action = read_text(row["action"], "action")
            symbol = symbols[action]
            if action in seen_actions:
                continue  # Upstream keeps the first row for each action.
            seen_actions.add(action)
            aliases = read_aliases(row["aliases"])
            enabled = row.get("isEnabled", True)
            if type(enabled) is not bool:
                raise ValueError("Upstream action enabled flag must be boolean")
            enabled = enabled and bool(aliases)
            if not enabled:
                matching = [entry for entry in dictionary["actions"] if entry["symbol"] == symbol]
                for entry in matching:
                    entry["enabled"] = False
                if not aliases and not matching:
                    notes.skip(
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
        notes.handled.add("spokenFormattingActionRules")
    Dictionary.validate(dictionary)
    return dictionary, len(replacements)


def _selected_provider(settings: dict[str, object], notes: MigrationNotes) -> Config | None:
    try:
        return _provider(settings)
    except (ValueError, TypeError):
        notes.skip(
            "provider",
            "Provider endpoint/model unsupported; credentials and endpoint not imported",
        )
        return None


def _mode(row: dict[str, object]) -> str | None:
    value = row.get("mode", "dictate")
    modes = {"dictate": "dictation", "edit": "edit", "write": "edit", "rewrite": "edit"}
    return modes.get(value) if isinstance(value, str) else None


def _draft_profiles(
    upstream_profiles: list[object],
    provider: Config | None,
    current_by_id: dict[str, dict[str, object]],
    reserved: set[str],
    notes: MigrationNotes,
) -> tuple[list[PromptProfile], dict[str, str], list[dict[str, object]]]:
    if provider is None:
        if upstream_profiles:
            notes.skip(
                "promptProfiles",
                "No supported selected provider/model; prompt profiles remain in original backup",
            )
        return [], {}, []
    profiles: list[dict[str, object]] = []
    profile_id_map: dict[str, str] = {}
    pending_profiles: list[dict[str, object]] = []
    notes.handled.update(("selectedProviderID", "selectedModelByProvider", "savedProviders"))
    notes.warnings.append(
        "Imported AI profiles stay inactive; existing Linux providers/selections unchanged. "
        "Review credentials and explicitly select a profile/add a route in Settings."
    )
    for value in upstream_profiles:
        profile = read_object(value)
        identifier = str(uuid.UUID(read_text(profile["id"], "prompt profile ID")))
        if identifier in profile_id_map:
            raise ValueError("Duplicate upstream prompt profile ID")
        mode = _mode(profile)
        if mode is None:
            raise ValueError("Invalid upstream prompt mode")
        draft: dict[str, object] = {
            "id": identifier,
            "name": profile["name"],
            "shortcut": "",
            **{key: getattr(provider, key) for key in PROFILE_FIELDS},
            "ai_prompt": profile["prompt"],
        }
        draft_id = identifier
        existing = current_by_id.get(identifier)
        if existing != draft and (existing is not None or identifier in reserved):
            serialized = json.dumps(draft, sort_keys=True, ensure_ascii=False)
            draft_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "mouthkeys:upstream:" + serialized))
            draft["id"] = draft_id
            if draft_id in current_by_id and current_by_id[draft_id] != draft:
                raise ValueError("Imported prompt profile ID conflicts with existing data")
        profile_id_map[identifier] = draft_id
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
    validated = validate_profiles(profiles)
    if validated:
        notes.warnings.append(
            "Prompts use global provider/model; backup omits per-profile provider pins."
        )
        notes.warnings.append(
            "Profile dates/context flags omitted; source selections/routes shown as pending."
        )
    return validated, profile_id_map, pending_profiles


def _pending_selections(
    settings: dict[str, object], profile_id_map: dict[str, str], notes: MigrationNotes
) -> dict[str, object]:
    pending_selections: dict[str, object] = {}
    for source in ("selectedDictationPromptID", "selectedEditPromptID"):
        identifier = settings.get(source)
        if not identifier:
            continue
        identifier = str(uuid.UUID(read_text(identifier, "selected prompt ID")))
        if identifier in profile_id_map:
            pending_selections[source] = {
                "source_id": identifier,
                "profile": profile_id_map[identifier],
            }
            notes.handled.add(source)
    for source in ("editPromptOff", "dictationPromptOff"):
        if settings.get(source) is not None:
            if type(settings[source]) is not bool:
                raise ValueError(f"Upstream {source} must be boolean")
            pending_selections[source] = settings[source]
            notes.handled.add(source)
    scope = settings.get("dictationPromptRoutingScope")
    if scope in ("allApps", "selectedAppsOnly"):
        pending_selections["dictationPromptRoutingScope"] = scope
        notes.handled.add("dictationPromptRoutingScope")
    return pending_selections


def _pending_bindings(
    bindings: list[object],
    app_map: dict[str, str],
    profile_id_map: dict[str, str],
    notes: MigrationNotes,
) -> list[dict[str, object]]:
    pending_bindings: list[dict[str, object]] = []
    for index, value in enumerate(bindings):
        binding = read_object(value)
        bundle = read_text(binding.get("appBundleID", ""), "application ID")
        mode = _mode(binding)
        selected = binding.get("promptID")
        source_id = (
            str(uuid.UUID(read_text(selected, "binding prompt ID"))) if selected else "default"
        )
        identifier = profile_id_map.get(source_id, "default" if not selected else None)
        app = app_map.get(bundle)
        if not app or mode is None or identifier is None:
            notes.skip(
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
    return pending_bindings


def _convert_upstream_backup(
    document: object,
    base_config: Config,
    current_dictionary: object,
    current_profiles: Sequence[Mapping[str, object]],
    current_routes: Sequence[Mapping[str, object]],
    *,
    app_map: object = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build native import data plus transcript-free report; do not write any state."""
    document = read_object(document)
    version, settings = backup_header(document)
    notes = MigrationNotes()
    changes = convert_settings(settings, notes)

    dictionary, replacement_count = _convert_dictionary(settings, current_dictionary, notes)
    current_by_id = {
        read_text(profile["id"], "profile ID"): dict(profile) for profile in current_profiles
    }
    reserved = {base_config.ai_profile, base_config.edit_profile, base_config.command_profile}
    reserved.update(read_text(rule["profile"], "route profile") for rule in current_routes)
    provider = _selected_provider(settings, notes)
    profiles, profile_id_map, pending_profiles = _draft_profiles(
        read_list(document, "promptProfiles", 100), provider, current_by_id, reserved, notes
    )
    pending_selections = _pending_selections(settings, profile_id_map, notes)
    mapping = read_app_mapping(app_map)
    pending_bindings = _pending_bindings(
        read_list(document, "appPromptBindings", 1000), mapping, profile_id_map, notes
    )
    routes = [copy.deepcopy(dict(route)) for route in current_routes]
    combined = dict(current_by_id)
    combined.update({p["id"]: dict(p) for p in profiles})
    validate_routes(routes, validate_profiles(list(combined.values())))

    history = convert_history(document, notes)
    if document.get("pronunciationProfiles"):
        notes.skip(
            "pronunciationProfiles",
            "Apple embeddings cannot become Linux acoustic templates; existing profiles unchanged",
        )
    if settings.get("pronunciationMatchingEnabled"):
        notes.skip(
            "pronunciationMatchingEnabled",
            "Re-enroll Linux pronunciation takes before enabling matching",
        )
    elif settings.get("pronunciationMatchingEnabled") is False:
        changes["pronunciation_matching"] = "off"
        notes.handled.add("pronunciationMatchingEnabled")
    for field, value in settings.items():
        if field not in notes.handled and value is not None:
            notes.skip(
                field, "No equivalent safe automatic mapping; existing Linux setting retained"
            )
    # Config validation enforces native ranges without truncating or inventing settings.
    Config.from_mapping(asdict(base_config) | changes)
    backup = {
        "version": 1,
        "config": changes,
        "dictionary": dictionary,
        "profiles": profiles,
        "history": history,
        "prompt_routes": routes,
    }
    report: dict[str, object] = {
        "format": "upstream",
        "schema_version": version,
        "counts": {
            "history": len(history),
            "profiles": len(profiles),
            "routes": 0,
            "pending_routes": len(pending_bindings),
            "replacements": replacement_count,
        },
        "changes": changes,
        "warnings": notes.warnings,
        "unmapped": notes.unmapped,
        "pending_profiles": pending_profiles,
        "pending_selections": pending_selections,
        "pending_bindings": pending_bindings,
        "profile_id_map": profile_id_map,
    }
    return backup, report


def convert_upstream_backup(
    document: object,
    base_config: Config,
    current_dictionary: object,
    current_profiles: Sequence[Mapping[str, object]],
    current_routes: Sequence[Mapping[str, object]],
    *,
    app_map: object = None,
) -> tuple[dict[str, object], dict[str, object]]:
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


def load_upstream_backup(
    path: str | Path, *, app_map: object = None, document: object = None
) -> tuple[dict[str, object], dict[str, object]]:
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
