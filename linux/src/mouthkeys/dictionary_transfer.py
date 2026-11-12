"""Portable upstream dictionaries alongside lossless Linux dictionary transfers."""

import copy
import json
import uuid
from pathlib import Path

from .data_lock import dictionary_lock
from .dictionary import Dictionary
from .operations import write_private
from .vocabulary import VocabularyStore

MAX_TRANSFER_BYTES = 16 * 1024 * 1024


def phrase_key(value):
    return " ".join(value.casefold().split())


def merge_replacements(existing, incoming):
    """Upsert by replacement; incoming triggers move from conflicting rules."""
    result = copy.deepcopy(existing)
    for entry in incoming:
        same = next(
            (
                old
                for old in result
                if phrase_key(old["replacement"]) == phrase_key(entry["replacement"])
            ),
            None,
        )
        triggers = list(
            dict.fromkeys(
                phrase_key(value)
                for value in [*(same["triggers"] if same else []), *entry["triggers"]]
            )
        )
        keys = set(triggers)
        retained = []
        for old in result:
            if old is same:
                continue
            remaining = [value for value in old["triggers"] if phrase_key(value) not in keys]
            if remaining:
                retained.append({**old, "triggers": remaining})
        merged = {
            **entry,
            "triggers": triggers,
            "replacement": same["replacement"] if same else entry["replacement"],
        }
        if same and "id" in same:
            merged["id"] = same["id"]
        result = retained + [merged]
    return result


def replacement_entries(values):
    if not isinstance(values, list) or len(values) > 10000:
        raise ValueError("Replacement import must contain at most 10000 entries")
    result = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Invalid replacement entry")
        triggers = value.get("from", value.get("triggers"))
        if isinstance(triggers, str):
            triggers = [triggers]
        entry = {
            "triggers": triggers,
            "replacement": value.get("to", value.get("replacement")),
        }
        if "enabled" in value:
            entry["enabled"] = value["enabled"]
        if value.get("id") is not None:
            try:
                entry["id"] = str(uuid.UUID(value["id"]))
            except (ValueError, TypeError, AttributeError):
                raise ValueError("Invalid replacement ID") from None
        # Validate each entry before resolving cross-entry conflicts like upstream.
        Dictionary.validate({**Dictionary.defaults(), "replacements": [entry]})
        result.append(entry)
    return merge_replacements([], result)


def custom_word_entries(values):
    if not isinstance(values, list):
        raise ValueError("Custom words must be a list")
    entries = []
    for value in values:
        if isinstance(value, str):
            value = {"text": value}
        if not isinstance(value, dict) or set(value) - {"text", "weight", "aliases"}:
            raise ValueError("Invalid custom word entry")
        entries.append({"weight": None, "aliases": [], **value})
    return VocabularyStore.validate_entries(entries)


def decode_document(data):
    """Return supplied native sections and optional vocabulary; absent means untouched."""
    if not isinstance(data, dict):
        raise ValueError("Dictionary transfer must be a JSON object")
    native = "punctuation" in data or "actions" in data
    if native:
        if set(data) - {"replacements", "punctuation", "actions", "customWords"}:
            raise ValueError("Unknown native dictionary transfer fields")
        sections = {key: value for key, value in data.items() if key != "customWords"}
        Dictionary.validate(sections)
    else:
        sections = {}
        if "replacements" in data:
            sections["replacements"] = replacement_entries(data["replacements"])
    words = None
    for key in ("customWords", "terms"):
        if key in data:
            words = custom_word_entries(data[key])
            break
    if not native and not sections and words is None:
        for key in ("items", "entries"):
            if key not in data:
                continue
            values = data[key]
            if (
                isinstance(values, list)
                and values
                and isinstance(values[0], dict)
                and ("triggers" in values[0] or "from" in values[0])
            ):
                sections["replacements"] = replacement_entries(values)
            else:
                words = custom_word_entries(values)
            break
    if not sections and words is None:
        raise ValueError("Expected replacements, customWords, terms, or API items/entries")
    return sections, words


def _merge_rules(existing, incoming):
    result = copy.deepcopy(existing)
    for entry in incoming:
        keys = {phrase_key(value) for value in entry["aliases"]}
        result = [
            {**old, "aliases": remaining}
            for old in result
            if (remaining := [value for value in old["aliases"] if phrase_key(value) not in keys])
        ]
        result.append(copy.deepcopy(entry))
    return result


def import_file(path, mode="replace"):
    if mode not in ("merge", "replace"):
        raise ValueError("Dictionary import mode must be merge or replace")
    path = Path(path)
    # Bound reads too: a file can grow between stat and read.
    with path.open("rb") as source:
        raw = source.read(MAX_TRANSFER_BYTES + 1)
    if len(raw) > MAX_TRANSFER_BYTES:
        raise ValueError("Dictionary transfer exceeds 16 MiB")
    sections, words = decode_document(json.loads(raw))
    with dictionary_lock():
        current = Dictionary.load()
        updated = copy.deepcopy(current)
        for section, entries in sections.items():
            updated[section] = (
                entries
                if mode == "replace"
                else (
                    merge_replacements(current[section], entries)
                    if section == "replacements"
                    else _merge_rules(current[section], entries)
                )
            )
        Dictionary.validate(updated)
        if words is not None:
            old_words = VocabularyStore.entries()
            if mode == "merge":
                keys = {entry["text"].casefold() for entry in old_words}
                words = old_words + [
                    entry for entry in words if entry["text"].casefold() not in keys
                ]
            words = VocabularyStore.validate_entries(words)
        # Both stores validate before writes. Restore vocabulary if second write fails.
        previous = None
        if words is not None:
            previous = (
                VocabularyStore.path().read_bytes() if VocabularyStore.path().exists() else None
            )
            VocabularyStore.save_entries(words)
        try:
            Dictionary.save(updated)
        except Exception:
            if words is not None:
                if previous is None:
                    VocabularyStore.path().unlink(missing_ok=True)
                else:
                    write_private(VocabularyStore.path(), previous)
            raise
        return updated


def export_file(path, format="native"):
    if format not in ("native", "upstream"):
        raise ValueError("Dictionary export format must be native or upstream")
    with dictionary_lock():
        data = Dictionary.load()
        words = VocabularyStore.entries()
        if format == "upstream":
            data = {
                "replacements": [
                    {"from": entry["triggers"], "to": entry["replacement"]}
                    for entry in data["replacements"]
                    if entry.get("enabled", True)
                ]
            }
        data["customWords"] = words
        encoded = json.dumps(data, ensure_ascii=False, indent=2).encode()
        if len(encoded) > MAX_TRANSFER_BYTES:
            raise ValueError("Dictionary transfer exceeds 16 MiB")
    write_private(path, encoded)
