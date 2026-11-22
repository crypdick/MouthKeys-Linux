"""Deterministic dictation formatting, including conservative Spoken Send parsing.

Behavior adapted from BenevolentFutures/MouthKeys d50ad008:
SpokenSendParser, QuestionMarkShortcut, and ASRService spoken/literal formatting.
No desktop effects here.
"""

import re
import unicodedata

from .dictionary import Dictionary, phrase_pattern
from .format_whitespace import NEWLINES, horizontal, rstrip_horizontal
from .literal_formatting import apply_literal_formatting as apply_literal_formatting
from .literal_formatting import literal_autocomplete_spacing as literal_autocomplete_spacing
from .spoken_send import SendPlan as SendPlan
from .spoken_send import parse_send as parse_send


def enabled(config: object, key: str) -> bool:
    return getattr(config, key, "off") == "on"


_DOUBLED_Q = re.compile(
    r"(?<![\w'’\-/\\.@#$=~])(?:qq|(?:q|queue|cue)[\s.,;:!?]+(?:q|queue|cue))(?![\w'’])",
    re.I,
)
_TRAILING_Q = re.compile(r"(?<![\w'’\-/\\.@#$=~])(?:q|cue)[\s.,;:!?]*$", re.I)
_REPLACED_ENDINGS = set(",.;:!…")


def _ending_question(text):
    end = len(text)
    while end and (text[end - 1].isspace() or text[end - 1] in _REPLACED_ENDINGS):
        end -= 1
    text = text[:end]
    return text if text.endswith("?") else text + "?"


def apply_question_mark_shortcut(text, literal_prefix="literal"):
    """Upstream's final-text Q shortcut; not idempotent after removing a literal prefix."""
    words = literal_prefix.split()
    literal = (
        re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"[\s,:;]*$", re.I)
        if words
        else None
    )
    output, rest, ends_literal = "", text, False
    while match := _DOUBLED_Q.search(rest):
        before, after = rest[: match.start()], rest[match.end() :]
        escaped = literal.search(before) if literal else None
        if escaped:
            output += before[: escaped.start()] + match[0]
            ends_literal = all(
                c.isspace() or unicodedata.category(c).startswith("P") for c in after
            )
            rest = after
            continue
        output += _ending_question(before)
        after = after.lstrip("".join(_REPLACED_ENDINGS | {"?"}))
        spacing = re.match(r"\s*", after)[0]
        following = after[len(spacing) :]
        if not following:
            rest = ""
            break
        output += spacing if any(c in NEWLINES for c in spacing) else " "
        word = re.match(r"\S+", following)[0]
        rest = following[0].upper() + following[1:] if word == word.lower() else following
    result = output + rest
    if ends_literal:
        return result
    if trailing := _TRAILING_Q.search(result):
        before = result[: trailing.start()]
        escaped = literal.search(before) if literal else None
        return before[: escaped.start()] + trailing[0] if escaped else _ending_question(before)
    return result


def format_text(
    text, config, dictionary=None, context=None, *, app_name=None, app_id=None, window_title=None
):
    app_context = dict(app_name=app_name, app_id=app_id, window_title=window_title)
    if enabled(config, "vocabulary_boosting"):
        from .vocabulary import VocabularyStore

        text = VocabularyStore.apply_aliases(text)
    data = Dictionary.load() if dictionary is None else Dictionary.validate(dictionary)
    text = Dictionary.apply(text, data)
    if enabled(config, "remove_filler_words"):
        fillers = [word.strip() for word in config.filler_words.split(",") if word.strip()]
        if fillers:
            pattern = "|".join(
                phrase_pattern(word) for word in sorted(fillers, key=len, reverse=True)
            )
            text = re.sub(r"(?:" + pattern + r")[,;:]?\s*", "", text, flags=re.I)
    if enabled(config, "spoken_formatting"):
        text = apply_spoken_punctuation(text, data, config.literal_prefix)
    if enabled(config, "literal_formatting"):
        text = apply_literal_formatting(text, **app_context)
    if enabled(config, "question_mark_shortcut"):
        text = apply_question_mark_shortcut(text, config.literal_prefix)
    if enabled(config, "lowercase_first") and text[:1].isupper():
        text = text[0].lower() + text[1:]
    if enabled(config, "remove_trailing_period") and text.endswith("."):
        text = text[:-1]
    if enabled(config, "smart_capitalization") and context is not None:
        boundary = None
        for character in reversed(context):
            if character in NEWLINES:
                break
            if horizontal(character) or character in "\"'”’»›)]}」』":
                continue
            boundary = character
            break
        text = _first_letter_case(text, upper=boundary is None or boundary in ".!?")
    if enabled(config, "space_between_dictations") and context is not None and text:
        if context and not context[-1].isspace() and not text[0].isspace():
            text = " " + text
        if not text[-1].isspace():
            text += " "
    if enabled(config, "literal_formatting"):
        text = literal_autocomplete_spacing(text, **app_context)
    return text


def _first_letter_case(text, *, upper):
    for index, character in enumerate(text):
        if character.isalpha():
            changed = character.upper() if upper else character.lower()
            return text[:index] + changed + text[index + 1 :]
    return text


def _punctuation_spacing(rule):
    symbol, aliases = rule["symbol"], {" ".join(a.casefold().split()) for a in rule["aliases"]}
    if symbol == ".":
        return "joined" if "dot" in aliases else "right"
    if symbol in {",", "?", "!", ":", ";", "...", ")", "]", "}", ">", "%"}:
        return "right"
    if symbol in {"(", "[", "{", "<", "$"}:
        return "left"
    if symbol in {"+", "=", "&", "—", "–"}:
        return "around"
    if symbol == "-":
        return "around" if aliases & {"dash", "minus sign"} else "joined"
    if symbol == '"':
        if any(a.startswith(("open ", "opening ")) for a in aliases):
            return "left"
        if any(a.startswith(("close ", "closing ")) for a in aliases):
            return "right"
        return "double"
    if symbol == "'":
        return "joined" if "apostrophe" in aliases else "single"
    return "joined"


def apply_spoken_punctuation(text, data, prefix="literal"):
    """Render prefixed dictionary rules, preserving untouched ASR punctuation.

    Parts retain generated-vs-existing identity: only generated commas between
    generated symbols and ASR punctuation adjacent to actions may be removed.
    Upstream's reachable dictionary path has no bare-word/context inference.
    """
    prefix_words = prefix.casefold().split()
    if not prefix_words or prefix.casefold() not in text.casefold():
        return text
    rules = []
    claimed = set()
    actions = {"\n": "line", "\n\n": "paragraph", "\t": "tab", " ": "space"}
    for section in ("punctuation", "actions"):
        for row in data[section]:
            if not row.get("enabled", True):
                continue
            spacing = (
                actions.get(row["symbol"], "joined")
                if section == "actions"
                else _punctuation_spacing(row)
            )
            for alias in row["aliases"]:
                words = alias.casefold().split()
                normalized = tuple(words)
                if normalized in claimed:
                    continue
                claimed.add(normalized)
                rules.append((prefix_words + words, row["symbol"], spacing))
    rules.sort(key=lambda row: (len(row[0]), len(" ".join(row[0]))), reverse=True)
    # Like upstream, group consecutive word and nonword characters. In particular,
    # punctuation grouped with whitespace must not be mistaken for a phrase gap.
    tokens = re.findall(r"[^\W_]+|[\W_]+", text)
    parts = []
    index = 0
    while index < len(tokens):
        for words, symbol, spacing in rules:
            cursor = index
            for word_index, word in enumerate(words):
                if word_index:
                    if cursor >= len(tokens) or not horizontal(tokens[cursor]):
                        break
                    cursor += 1
                if cursor >= len(tokens) or tokens[cursor].casefold() != word:
                    break
                cursor += 1
            else:
                parts.append([symbol, spacing])
                index = cursor
                break
        else:
            parts.append([tokens[index], "text"])
            index += 1
    punctuation_pairs = set("+=%-—–/\\@#$&*_|~^<>()[]{}\"'`.?!:;")

    def neighbor(at, direction):
        at += direction
        while 0 <= at < len(parts):
            value, kind = parts[at]
            if kind != "text" or not horizontal(value):
                return parts[at]
            at += direction
        return None

    cleaned = []
    for index, part in enumerate(parts):
        if part == [",", "right"]:
            before, after = neighbor(index, -1), neighbor(index, 1)
            pair = (
                before
                and after
                and before[1] != "text"
                and after[1] != "text"
                and before[0][:1] in punctuation_pairs
                and after[0][:1] in punctuation_pairs
            )
            percent = (
                before
                and before[1] == "text"
                and before[0][-1:] in "0123456789"
                and after == ["%", "right"]
            )
            if pair or percent:
                continue
        cleaned.append(part.copy())
    parts = cleaned
    for index, (_, kind) in enumerate(parts):
        if kind not in {"line", "paragraph", "tab", "space"}:
            continue
        if index and parts[index - 1][1] == "text":
            before = rstrip_horizontal(parts[index - 1][0])
            if before.endswith(".") and not before.endswith(".."):
                parts[index - 1][0] = before[:-1]
        if index + 1 < len(parts) and parts[index + 1][1] == "text":
            after = parts[index + 1][0]
            stripped = after
            while stripped and horizontal(stripped[0]):
                stripped = stripped[1:]
            allowed = ".," if kind in {"line", "paragraph"} else "."
            if stripped and stripped[0] in allowed and not stripped.startswith(stripped[0] * 2):
                stripped = stripped[1:]
                while stripped and horizontal(stripped[0]):
                    stripped = stripped[1:]
                parts[index + 1][0] = stripped
    result = ""
    quote_open = {"double": True, "single": True}
    index = 0
    while index < len(parts):
        symbol, kind = parts[index]
        if kind == "text":
            result += symbol
            index += 1
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
        index += 1
        if kind != "right":
            while index < len(parts) and parts[index][1] == "text" and horizontal(parts[index][0]):
                index += 1
        if kind == "around" and any(
            kind != "text" or (value and not horizontal(value)) for value, kind in parts[index:]
        ):
            result += " "
    return result
