"""Private vocabulary with native prompt biasing and explicit alias normalization.

Whisper/Qwen accept textual hints, not weighted finite-state decoding. Positive
numeric weights are preserved for native decoders. Textual hints use at most
five repetitions and a 4096-byte budget, not a calibrated probability or
guaranteed recognition. Aliases are user-authored
literal replacements; they never guess similar-sounding words.
"""

import json
import math
import re

from .config import config_path
from .data_lock import dictionary_lock
from .dictionary import Dictionary, phrase_pattern
from .operations import write_private

MAX_TERMS = 256
MAX_PROMPT_BYTES = 4096


def _phrase(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError("Vocabulary entries must contain 1–128 characters")
    if any(not char.isprintable() for char in value):
        raise ValueError("Vocabulary entries must be printable text")
    return " ".join(value.split())


class VocabularyStore:
    @staticmethod
    def path():
        return config_path().with_name("vocabulary.json")

    @staticmethod
    def validate(words):
        if not isinstance(words, list):
            raise ValueError("Vocabulary must be a list")
        entries = VocabularyStore.validate_entries(
            [{"text": word, "weight": 1, "aliases": []} for word in words]
        )
        return [entry["text"] for entry in entries]

    @staticmethod
    def validate_entries(entries):
        if not isinstance(entries, list) or len(entries) > MAX_TERMS:
            raise ValueError("Vocabulary must contain at most 256 words or phrases")
        cleaned, seen, aliases = [], set(), set()
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or "text" not in entry
                or set(entry) - {"text", "weight", "aliases"}
            ):
                raise ValueError("Vocabulary entry requires text, weight, and aliases")
            word = _phrase(entry["text"])
            weight = entry.get("weight")
            if weight is not None and (
                type(weight) not in (int, float) or not 0 < weight <= 3.4028235e38
            ):
                raise ValueError("Vocabulary weight must be a positive finite float or null")
            if word.casefold() in seen:
                raise ValueError("Duplicate vocabulary entry")
            seen.add(word.casefold())
            entry_aliases = entry.get("aliases", [])
            if entry_aliases is None:
                entry_aliases = []
            if not isinstance(entry_aliases, list) or len(entry_aliases) > 20:
                raise ValueError("Each vocabulary entry supports at most 20 aliases")
            variants = []
            for alias in entry_aliases:
                alias = _phrase(alias)
                if alias.casefold() in aliases or alias.casefold() == word.casefold():
                    raise ValueError("Duplicate or self-referencing vocabulary alias")
                aliases.add(alias.casefold())
                variants.append(alias)
            cleaned.append({"text": word, "weight": weight, "aliases": variants})
        if aliases & seen:
            raise ValueError("Vocabulary aliases must not override another vocabulary word")
        if len(json.dumps(cleaned).encode()) > 256 * 1024:
            raise ValueError("Vocabulary entries exceed 256 KiB")
        return cleaned

    @classmethod
    def entries(cls):
        with dictionary_lock():
            path = cls.path()
            if not path.exists():
                return []
            if path.stat().st_size > 256 * 1024:
                raise ValueError("Vocabulary file exceeds 256 KiB")
            data = json.loads(path.read_text())
            if isinstance(data, list) and all(isinstance(item, str) for item in data):
                data = [{"text": word, "weight": 1, "aliases": []} for word in data]
            return cls.validate_entries(data)

    @classmethod
    def load(cls):
        return [entry["text"] for entry in cls.entries()]

    @classmethod
    def save_entries(cls, entries):
        from .config import private_dir

        with dictionary_lock():
            entries = cls.validate_entries(entries)
            path = cls.path()
            private_dir(path.parent)
            write_private(path, json.dumps(entries, ensure_ascii=False, indent=2).encode())
            return entries

    @classmethod
    def save(cls, words):
        """Legacy string-list editing preserves weights/aliases for surviving words."""
        with dictionary_lock():
            words = cls.validate(words)
            previous = {entry["text"].casefold(): entry for entry in cls.entries()}
            entries = [
                {**previous[word.casefold()], "text": word}
                if word.casefold() in previous
                else {"text": word, "weight": 1, "aliases": []}
                for word in words
            ]
            cls.save_entries(entries)
            return words

    @classmethod
    def add(cls, word):
        with dictionary_lock():
            word = _phrase(word)
            words = cls.load()
            if word.casefold() not in {item.casefold() for item in words}:
                words.append(word)
            return cls.save(words)

    @classmethod
    def remove(cls, word):
        with dictionary_lock():
            return cls.save([item for item in cls.load() if item.casefold() != word.casefold()])

    @classmethod
    def effective_entries(cls):
        """Merge enabled replacements for decoding, without rewriting user vocabulary.

        Upstream uses weight 8 for dictionary terms and keeps highest-priority
        terms when its 256-term decoder budget is exceeded. Aliases here are
        decoder metadata only; explicit replacement application stays one-pass.
        """
        with dictionary_lock():
            merged = {entry["text"].casefold(): dict(entry) for entry in cls.entries()}
            for replacement in Dictionary.load()["replacements"]:
                if not replacement.get("enabled", True):
                    continue
                try:
                    text = _phrase(replacement["replacement"])
                except ValueError:
                    continue  # Formatting actions/long prose are not vocabulary terms.
                key = text.casefold()
                aliases = []
                for trigger in replacement["triggers"]:
                    try:
                        alias = _phrase(trigger)
                    except ValueError:
                        continue
                    if alias.casefold() != key:
                        aliases.append(alias)
                previous = merged.get(key)
                if previous:
                    aliases = [*previous["aliases"], *aliases]
                merged[key] = {
                    "text": previous["text"] if previous else text,
                    "weight": max(previous["weight"] or 0, 8) if previous else 8,
                    "aliases": list(dict.fromkeys(aliases)),
                }
            return sorted(
                merged.values(),
                key=lambda entry: (-(entry["weight"] or 0), entry["text"].casefold()),
            )[:MAX_TERMS]

    @classmethod
    def apply_aliases(cls, text):
        aliases = {
            " ".join(alias.casefold().split()): entry["text"]
            for entry in cls.entries()
            for alias in entry["aliases"]
        }
        if not aliases:
            return text
        pattern = "|".join(
            phrase_pattern(alias) for alias in sorted(aliases, key=len, reverse=True)
        )
        return re.sub(
            pattern, lambda match: aliases[" ".join(match[0].casefold().split())], text, flags=re.I
        )


def _prompt_words(config):
    if getattr(config, "vocabulary_boosting", "off") != "on":
        return []
    words, size = [], 0
    for entry in VocabularyStore.effective_entries():
        repetitions = math.ceil(min(5, entry["weight"] or 1))
        for _ in range(repetitions):
            added = len(entry["text"].encode()) + (2 if words else 0)
            if size + added + len("Vocabulary: ") > MAX_PROMPT_BYTES:
                break
            words.append(entry["text"])
            size += added
    return words


def context_prompt(config):
    words = _prompt_words(config)
    return "Vocabulary: " + ", ".join(words) if words else ""


def whisper_options(config):
    words = _prompt_words(config)
    return {"hotwords": ", ".join(words)} if words else {}
