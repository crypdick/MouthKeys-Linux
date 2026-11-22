"""Atomic dictionary mutations and response formats for the loopback API."""

import json
import uuid
from collections.abc import Mapping

from .data_lock import dictionary_lock
from .dictionary import Dictionary, Replacement
from .dictionary_transfer import merge_replacements, phrase_key, replacement_entries
from .vocabulary import VocabularyStore


def _replacement_id(entry: Replacement) -> str:
    if "id" in entry:
        return str(uuid.UUID(entry["id"]))
    value = json.dumps([entry["triggers"], entry["replacement"]], ensure_ascii=False)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "mouthkeys:replacement:" + value))


def _replace_requested(data: Mapping[str, object]) -> bool:
    mode = data.get("mode", "append")
    if mode not in ("append", "replace"):
        raise ValueError("Invalid dictionary write mode")
    return mode == "replace"


def custom_words(update: Mapping[str, object] | None) -> dict[str, object]:
    with dictionary_lock():
        entries = VocabularyStore.entries()
        if update is not None:
            replace_all = _replace_requested(update)
            incoming = update.get("entries")
            if "entries" not in update and "text" in update:
                incoming = [{key: value for key, value in update.items() if key != "mode"}]
            if incoming is None:
                raise ValueError("Expected custom word entries")
            words = VocabularyStore.validate_entries(incoming)
            keys = {item["text"].casefold() for item in words}
            retained = (
                []
                if replace_all
                else [item for item in entries if item["text"].casefold() not in keys]
            )
            entries = VocabularyStore.save_entries(words + retained)
    return {"count": len(entries), "items": entries}


def replacements(update: Mapping[str, object] | None, *, native: bool) -> object:
    with dictionary_lock():
        dictionary = Dictionary.load()
        if update is not None:
            dictionary["replacements"] = _updated_replacements(dictionary["replacements"], update)
            Dictionary.save(dictionary)
        rules = dictionary["replacements"]
    if native:
        return rules
    items = [{"id": _replacement_id(entry), **entry} for entry in rules]
    return {"count": len(items), "items": items}


def _updated_replacements(
    existing: list[Replacement], data: Mapping[str, object]
) -> list[Replacement]:
    replace_all = _replace_requested(data)
    if "replacements" in data:
        return replacement_entries(data["replacements"])
    values = data.get("entries")
    if values is None and "triggers" in data:
        values = [{key: value for key, value in data.items() if key != "mode"}]
    if values is None:
        raise ValueError("Expected replacement entries")
    incoming = replacement_entries(values)
    if replace_all:
        return incoming
    identifiers = {entry["id"] for entry in incoming if "id" in entry}
    targets = {phrase_key(entry["replacement"]) for entry in incoming}
    retained = [
        entry
        for entry in existing
        if _replacement_id(entry) not in identifiers
        and phrase_key(entry["replacement"]) not in targets
    ]
    return merge_replacements(retained, incoming)
