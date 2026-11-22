"""Match prefixed dictionary phrases and render generated punctuation separately."""

import re
from typing import Literal, NamedTuple

from .dictionary import DictionaryData, FormattingRule
from .format_whitespace import NEWLINES, horizontal, rstrip_horizontal

_Spacing = Literal[
    "text",
    "joined",
    "left",
    "right",
    "around",
    "double",
    "single",
    "line",
    "paragraph",
    "tab",
    "space",
]
_ACTIONS: dict[str, _Spacing] = {"\n": "line", "\n\n": "paragraph", "\t": "tab", " ": "space"}
_STATIC_SPACING: dict[str, _Spacing] = {
    **dict.fromkeys((",", "?", "!", ":", ";", "...", ")", "]", "}", ">", "%"), "right"),
    **dict.fromkeys(("(", "[", "{", "<", "$"), "left"),
    **dict.fromkeys(("+", "=", "&", "—", "–"), "around"),
}
_PUNCTUATION_PAIRS = frozenset("+=%-—–/\\@#$&*_|~^<>()[]{}\"'\x60.?!:;")
_TOKENS = re.compile(r"[^\W_]+|[\W_]+")


class _Part(NamedTuple):
    text: str
    kind: _Spacing


class _Rule(NamedTuple):
    phrase: tuple[str, ...]
    part: _Part


_COMMA = _Part(",", "right")
_PERCENT = _Part("%", "right")


def _double_quote(aliases: set[str]) -> _Spacing:
    if any(alias.startswith(("open ", "opening ")) for alias in aliases):
        return "left"
    if any(alias.startswith(("close ", "closing ")) for alias in aliases):
        return "right"
    return "double"


def _punctuation_spacing(rule: FormattingRule) -> _Spacing:
    symbol = rule["symbol"]
    aliases = {" ".join(alias.casefold().split()) for alias in rule["aliases"]}
    if symbol == ".":
        return "joined" if "dot" in aliases else "right"
    if symbol == "-":
        return "around" if aliases & {"dash", "minus sign"} else "joined"
    if symbol == '"':
        return _double_quote(aliases)
    if symbol == "'":
        return "joined" if "apostrophe" in aliases else "single"
    return _STATIC_SPACING.get(symbol, "joined")


def _compile_rules(data: DictionaryData) -> dict[str, list[_Rule]]:
    claimed: set[tuple[str, ...]] = set()
    by_first_word: dict[str, list[_Rule]] = {}
    for section, rows in (("punctuation", data["punctuation"]), ("actions", data["actions"])):
        for row in rows:
            if not row.get("enabled", True):
                continue
            spacing = (
                _ACTIONS.get(row["symbol"], "joined")
                if section == "actions"
                else _punctuation_spacing(row)
            )
            for alias in row["aliases"]:
                phrase = tuple(alias.casefold().split())
                if phrase in claimed:
                    continue
                claimed.add(phrase)
                by_first_word.setdefault(phrase[0], []).append(
                    _Rule(phrase, _Part(row["symbol"], spacing))
                )
    for rules in by_first_word.values():
        rules.sort(key=lambda rule: (len(rule.phrase), len(" ".join(rule.phrase))), reverse=True)
    return by_first_word


def _phrase_end(tokens: list[str], start: int, phrase: tuple[str, ...]) -> int | None:
    cursor = start
    for index, word in enumerate(phrase):
        if index:
            if cursor >= len(tokens) or not horizontal(tokens[cursor]):
                return None
            cursor += 1
        if cursor >= len(tokens) or tokens[cursor].casefold() != word:
            return None
        cursor += 1
    return cursor


def _generated_at(
    tokens: list[str],
    index: int,
    prefix: tuple[str, ...],
    rules: dict[str, list[_Rule]],
) -> tuple[_Part, int] | None:
    end = _phrase_end(tokens, index, prefix)
    if end is None or end + 1 >= len(tokens) or not horizontal(tokens[end]):
        return None
    start = end + 1
    for rule in rules.get(tokens[start].casefold(), []):
        if (stop := _phrase_end(tokens, start, rule.phrase)) is not None:
            return rule.part, stop
    return None


def _parts(text: str, prefix: tuple[str, ...], data: DictionaryData) -> list[_Part]:
    rules = _compile_rules(data)
    # Punctuation grouped with whitespace must not be mistaken for a phrase gap.
    tokens = _TOKENS.findall(text)
    parts: list[_Part] = []
    index = 0
    while index < len(tokens):
        match = _generated_at(tokens, index, prefix, rules)
        if match is None:
            parts.append(_Part(tokens[index], "text"))
            index += 1
        else:
            part, index = match
            parts.append(part)
    return parts


def _neighbor(parts: list[_Part], at: int, direction: int) -> _Part | None:
    at += direction
    while 0 <= at < len(parts):
        part = parts[at]
        if part.kind != "text" or not horizontal(part.text):
            return part
        at += direction
    return None


def _decorative_comma(parts: list[_Part], index: int) -> bool:
    before, after = _neighbor(parts, index, -1), _neighbor(parts, index, 1)
    if before is None or after is None:
        return False
    if before.kind == "text":
        return before.text[-1:] in "0123456789" and after == _PERCENT
    return (
        after.kind != "text"
        and before.text[:1] in _PUNCTUATION_PAIRS
        and after.text[:1] in _PUNCTUATION_PAIRS
    )


def _skip_horizontal(text: str) -> str:
    start = 0
    while start < len(text) and horizontal(text[start]):
        start += 1
    return text[start:]


def _trim_action_neighbors(parts: list[_Part]) -> None:
    for index, (_, kind) in enumerate(parts):
        if kind not in {"line", "paragraph", "tab", "space"}:
            continue
        if index and parts[index - 1].kind == "text":
            before = rstrip_horizontal(parts[index - 1].text)
            if before.endswith(".") and not before.endswith(".."):
                parts[index - 1] = _Part(before[:-1], "text")
        if index + 1 < len(parts) and parts[index + 1].kind == "text":
            tail = _skip_horizontal(parts[index + 1].text)
            allowed = ".," if kind in {"line", "paragraph"} else "."
            if tail and tail[0] in allowed and not tail.startswith(tail[0] * 2):
                parts[index + 1] = _Part(_skip_horizontal(tail[1:]), "text")


def _render(parts: list[_Part]) -> str:
    result = ""
    quote_open: dict[_Spacing, bool] = {"double": True, "single": True}
    index = 0
    while index < len(parts):
        symbol, kind = parts[index]
        index += 1
        if kind == "text":
            result += symbol
            continue
        if kind in quote_open:
            toggle = kind
            kind = "left" if quote_open[toggle] else "right"
            quote_open[toggle] = not quote_open[toggle]
        if kind != "left":
            result = rstrip_horizontal(result)
        if kind == "around" and result and result[-1] not in NEWLINES:
            result += " "
        result += symbol
        if kind != "right":
            while (
                index < len(parts) and parts[index].kind == "text" and horizontal(parts[index].text)
            ):
                index += 1
        if kind == "around" and any(
            part.kind != "text" or (part.text and not horizontal(part.text))
            for part in parts[index:]
        ):
            result += " "
    return result


def apply_spoken_punctuation(text: str, data: DictionaryData, prefix: str = "literal") -> str:
    """Render generated punctuation without treating adjacent ASR text as a command."""
    words = tuple(prefix.casefold().split())
    if not words or words[0] not in text.casefold():
        return text
    parts = _parts(text, words, data)
    parts = [
        part
        for index, part in enumerate(parts)
        if part != _COMMA or not _decorative_comma(parts, index)
    ]
    _trim_action_neighbors(parts)
    return _render(parts)
