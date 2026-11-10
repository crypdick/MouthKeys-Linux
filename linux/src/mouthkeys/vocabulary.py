"""Private vocabulary with native prompt biasing and explicit alias normalization.

Whisper/Qwen accept textual hints, not weighted finite-state decoding. A weight
of 1–5 repeats that term in their bounded prompt. This is a contextual hint, not
a calibrated probability or guaranteed recognition. Aliases are user-authored
literal replacements; they never guess similar-sounding words.
"""

import json
import re
import threading

from .config import config_path
from .dictionary import phrase_pattern
from .operations import write_private

_LOCK = threading.RLock()


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
        if not isinstance(entries, list) or len(entries) > 100:
            raise ValueError("Vocabulary must contain at most 100 words or phrases")
        cleaned, seen, aliases = [], set(), set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"text", "weight", "aliases"}:
                raise ValueError("Vocabulary entry requires text, weight, and aliases")
            word = _phrase(entry["text"])
            weight = entry["weight"]
            if type(weight) is not int or not 1 <= weight <= 5:
                raise ValueError("Vocabulary prompt weight must be an integer from 1 to 5")
            if word.casefold() in seen:
                raise ValueError("Duplicate vocabulary entry")
            seen.add(word.casefold())
            if not isinstance(entry["aliases"], list) or len(entry["aliases"]) > 20:
                raise ValueError("Each vocabulary entry supports at most 20 aliases")
            variants = []
            for alias in entry["aliases"]:
                alias = _phrase(alias)
                if alias.casefold() in aliases or alias.casefold() == word.casefold():
                    raise ValueError("Duplicate or self-referencing vocabulary alias")
                aliases.add(alias.casefold())
                variants.append(alias)
            cleaned.append({"text": word, "weight": weight, "aliases": variants})
        if aliases & seen:
            raise ValueError("Vocabulary aliases must not override another vocabulary word")
        prompt = ", ".join(entry["text"] for entry in cleaned for _ in range(entry["weight"]))
        if len(prompt.encode()) > 4096:
            raise ValueError("Weighted vocabulary prompt exceeds 4096 UTF-8 bytes")
        if len(json.dumps(cleaned).encode()) > 256 * 1024:
            raise ValueError("Vocabulary entries exceed 256 KiB")
        return cleaned

    @classmethod
    def entries(cls):
        with _LOCK:
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

        with _LOCK:
            entries = cls.validate_entries(entries)
            path = cls.path()
            private_dir(path.parent)
            write_private(path, json.dumps(entries, ensure_ascii=False, indent=2).encode())
            return entries

    @classmethod
    def save(cls, words):
        """Legacy string-list editing preserves weights/aliases for surviving words."""
        with _LOCK:
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
        with _LOCK:
            word = _phrase(word)
            words = cls.load()
            if word.casefold() not in {item.casefold() for item in words}:
                words.append(word)
            return cls.save(words)

    @classmethod
    def remove(cls, word):
        with _LOCK:
            return cls.save([item for item in cls.load() if item.casefold() != word.casefold()])

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
    return [entry["text"] for entry in VocabularyStore.entries() for _ in range(entry["weight"])]


def context_prompt(config):
    words = _prompt_words(config)
    return "Vocabulary: " + ", ".join(words) if words else ""


def whisper_options(config):
    words = _prompt_words(config)
    return {"hotwords": ", ".join(words)} if words else {}
