"""Local replacement and spoken-formatting dictionary; no engine or desktop access."""

import copy
import json
import os
import re
import tempfile
from pathlib import Path

from .config import config_path, private_dir

# Keep aliases grouped: dot/period, hyphen/dash, and quote variants have different spacing.
DEFAULT_PUNCTUATION_RULES = [
    {"aliases": ["comma"], "symbol": ","},
    {"aliases": ["period", "full stop"], "symbol": "."},
    {"aliases": ["dot"], "symbol": "."},
    {"aliases": ["question mark", "questionmark"], "symbol": "?"},
    {"aliases": ["exclamation mark", "exclamation point", "bang"], "symbol": "!"},
    {"aliases": ["colon"], "symbol": ":"},
    {"aliases": ["semicolon", "semi colon"], "symbol": ";"},
    {"aliases": ["ellipsis", "dot dot dot", "three dots"], "symbol": "..."},
    {"aliases": ["slash", "forward slash", "forwardslash"], "symbol": "/"},
    {"aliases": ["backslash", "back slash"], "symbol": "\\"},
    {"aliases": ["hyphen"], "symbol": "-"},
    {"aliases": ["dash", "minus sign"], "symbol": "-"},
    {"aliases": ["em dash", "long dash"], "symbol": "—"},
    {"aliases": ["en dash"], "symbol": "–"},
    {
        "aliases": [
            "open parenthesis",
            "open parentheses",
            "left parenthesis",
            "left parentheses",
            "open paren",
            "left paren",
        ],
        "symbol": "(",
    },
    {
        "aliases": [
            "close parenthesis",
            "close parentheses",
            "right parenthesis",
            "right parentheses",
            "close paren",
            "right paren",
        ],
        "symbol": ")",
    },
    {
        "aliases": ["open bracket", "left bracket", "open square bracket", "left square bracket"],
        "symbol": "[",
    },
    {
        "aliases": [
            "close bracket",
            "right bracket",
            "close square bracket",
            "right square bracket",
        ],
        "symbol": "]",
    },
    {
        "aliases": [
            "open brace",
            "left brace",
            "open curly brace",
            "left curly brace",
            "open curly bracket",
            "left curly bracket",
        ],
        "symbol": "{",
    },
    {
        "aliases": [
            "close brace",
            "right brace",
            "close curly brace",
            "right curly brace",
            "close curly bracket",
            "right curly bracket",
        ],
        "symbol": "}",
    },
    {"aliases": ["open angle bracket", "left angle bracket", "less than sign"], "symbol": "<"},
    {"aliases": ["close angle bracket", "right angle bracket", "greater than sign"], "symbol": ">"},
    {"aliases": ["quote", "quotes", "quotation mark", "double quote"], "symbol": '"'},
    {
        "aliases": ["open quote", "opening quote", "open double quote", "opening double quote"],
        "symbol": '"',
    },
    {
        "aliases": ["close quote", "closing quote", "close double quote", "closing double quote"],
        "symbol": '"',
    },
    {"aliases": ["single quote"], "symbol": "'"},
    {"aliases": ["apostrophe"], "symbol": "'"},
    {"aliases": ["at the rate", "at sign", "commercial at"], "symbol": "@"},
    {"aliases": ["ampersand", "and sign"], "symbol": "&"},
    {"aliases": ["plus sign", "plus"], "symbol": "+"},
    {"aliases": ["equals sign", "equal sign", "equal", "equals"], "symbol": "="},
    {"aliases": ["percent sign", "percentage sign", "percent"], "symbol": "%"},
    {"aliases": ["dollar sign", "dollar"], "symbol": "$"},
    {"aliases": ["hash", "hash sign", "hashtag", "pound sign", "number sign"], "symbol": "#"},
    {"aliases": ["asterisk", "star symbol"], "symbol": "*"},
    {"aliases": ["underscore"], "symbol": "_"},
    {"aliases": ["pipe", "vertical bar"], "symbol": "|"},
    {"aliases": ["tilde"], "symbol": "~"},
    {"aliases": ["caret"], "symbol": "^"},
    {"aliases": ["backtick", "back tick"], "symbol": "`"},
]


def phrase_pattern(phrase):
    return r"(?<!\w)" + r"\s+".join(re.escape(word) for word in phrase.split()) + r"(?!\w)"


class Dictionary:
    @staticmethod
    def path():
        return config_path().with_name("dictionary.json")

    @staticmethod
    def defaults():
        return {
            "replacements": [],
            "punctuation": copy.deepcopy(DEFAULT_PUNCTUATION_RULES),
            "actions": [
                {"aliases": ["new line", "next line", "newline"], "symbol": "\n"},
                {"aliases": ["new paragraph", "next paragraph"], "symbol": "\n\n"},
                {"aliases": ["tab"], "symbol": "\t"},
                {"aliases": ["space"], "symbol": " "},
            ],
        }

    @staticmethod
    def validate(data):
        if not isinstance(data, dict) or set(data) != {"replacements", "punctuation", "actions"}:
            raise ValueError("Dictionary requires replacements, punctuation, and actions lists")
        for section, key, value_key in (
            ("replacements", "triggers", "replacement"),
            ("punctuation", "aliases", "symbol"),
            ("actions", "aliases", "symbol"),
        ):
            if not isinstance(data[section], list) or len(data[section]) > 10000:
                raise ValueError(f"{section} must be a list of at most 10000 entries")
            seen = set()
            for entry in data[section]:
                if not isinstance(entry, dict) or set(entry) - {key, value_key, "enabled"}:
                    raise ValueError(f"Invalid {section} entry")
                aliases = entry.get(key)
                if (
                    not isinstance(aliases, list)
                    or not aliases
                    or any(not isinstance(a, str) or not a.strip() or len(a) > 256 for a in aliases)
                    or not isinstance(entry.get(value_key), str)
                    or len(entry[value_key]) > 10000
                    or type(entry.get("enabled", True)) is not bool
                ):
                    raise ValueError(f"Invalid {section} entry")
                for alias in aliases:
                    normalized = " ".join(alias.casefold().split())
                    if normalized in seen:
                        raise ValueError(f"Duplicate {section} phrase: {alias}")
                    seen.add(normalized)
        return data

    @classmethod
    def load(cls):
        path = cls.path()
        if not path.exists():
            return cls.defaults()
        return cls.validate(json.loads(path.read_text()))

    @classmethod
    def save(cls, data):
        cls.validate(data)
        path = cls.path()
        private_dir(path.parent)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
            json.dump(data, output, ensure_ascii=False, indent=2)
            temporary = Path(output.name)
        temporary.replace(path)

    @classmethod
    def import_file(cls, path):
        data = cls.validate(json.loads(Path(path).read_text()))
        cls.save(data)
        return data

    @classmethod
    def export_file(cls, path):
        data = cls.load()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as output:
            os.fchmod(output.fileno(), 0o600)
            json.dump(data, output, ensure_ascii=False, indent=2)

    @classmethod
    def apply(cls, text, data=None):
        data = cls.load() if data is None else cls.validate(data)
        # One pass: replacement text never becomes another trigger in this dictation.
        replacements = {
            " ".join(alias.casefold().split()): entry["replacement"]
            for entry in data["replacements"]
            if entry.get("enabled", True)
            for alias in entry["triggers"]
        }
        if not replacements:
            return text
        pattern = "|".join(phrase_pattern(a) for a in sorted(replacements, key=len, reverse=True))
        return re.sub(
            pattern,
            lambda match: replacements[" ".join(match.group().casefold().split())],
            text,
            flags=re.IGNORECASE,
        )
