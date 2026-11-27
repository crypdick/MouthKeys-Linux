"""Deterministic dictation formatting, including conservative Spoken Send parsing.

Behavior adapted from BenevolentFutures/MouthKeys d50ad008:
SpokenSendParser, QuestionMarkShortcut, and ASRService spoken/literal formatting.
No desktop effects here.
"""

import re
import unicodedata
from typing import TypedDict

from .config import Config
from .dictionary import Dictionary, phrase_pattern
from .format_whitespace import NEWLINES, horizontal
from .literal_formatting import apply_literal_formatting, literal_autocomplete_spacing
from .spoken_punctuation import apply_spoken_punctuation
from .spoken_send import SendPlan, parse_send
from .vocabulary import VocabularyStore

__all__ = [
    "FormattingApp",
    "SendPlan",
    "apply_literal_formatting",
    "apply_question_mark_shortcut",
    "apply_spoken_punctuation",
    "enabled",
    "format_text",
    "literal_autocomplete_spacing",
    "parse_send",
]


class FormattingApp(TypedDict, total=False):
    app_name: str | None
    app_id: str | None
    window_title: str | None


def enabled(config: object, key: str) -> bool:
    return getattr(config, key, "off") == "on"


_DOUBLED_Q = re.compile(
    r"(?<![\w'’\-/\\.@#$=~])(?:qq|(?:q|queue|cue)[\s.,;:!?]+(?:q|queue|cue))(?![\w'’])",
    re.I,
)
_TRAILING_Q = re.compile(r"(?<![\w'’\-/\\.@#$=~])(?:q|cue)[\s.,;:!?]*$", re.I)
_REPLACED_ENDINGS = set(",.;:!…")


def _ending_question(text: str) -> str:
    end = len(text)
    while end and (text[end - 1].isspace() or text[end - 1] in _REPLACED_ENDINGS):
        end -= 1
    text = text[:end]
    return text if text.endswith("?") else text + "?"


def apply_question_mark_shortcut(text: str, literal_prefix: str = "literal") -> str:
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
        following = after.lstrip()
        spacing = after[: len(after) - len(following)]
        if not following:
            rest = ""
            break
        output += spacing if any(c in NEWLINES for c in spacing) else " "
        word = following.split(maxsplit=1)[0]
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
    text: str,
    config: Config,
    dictionary: object = None,
    context: str | None = None,
    *,
    app: FormattingApp | None = None,
) -> str:
    app_context: FormattingApp = app if app is not None else {}
    if enabled(config, "vocabulary_boosting"):
        text = VocabularyStore.apply_aliases(text)
    data = Dictionary.load() if dictionary is None else Dictionary.validate(dictionary)
    text = Dictionary.apply(text, data)
    if enabled(config, "remove_filler_words"):
        text = _without_fillers(text, config.filler_words)
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
    if context is not None and enabled(config, "smart_capitalization"):
        text = _first_letter_case(text, upper=_starts_sentence(context))
    if enabled(config, "space_between_dictations") and text:
        if context and not context[-1].isspace() and not text[0].isspace():
            text = " " + text
        if not text[-1].isspace():
            text += " "
    if enabled(config, "literal_formatting"):
        text = literal_autocomplete_spacing(text, **app_context)
    return text


def _first_letter_case(text: str, *, upper: bool) -> str:
    for index, character in enumerate(text):
        if character.isalpha():
            changed = character.upper() if upper else character.lower()
            return text[:index] + changed + text[index + 1 :]
    return text


def _without_fillers(text: str, filler_words: str) -> str:
    fillers = [word.strip() for word in filler_words.split(",") if word.strip()]
    if not fillers:
        return text
    pattern = "|".join(phrase_pattern(word) for word in sorted(fillers, key=len, reverse=True))
    return re.sub(r"(?:" + pattern + r")[,;:]?\s*", "", text, flags=re.I)


def _starts_sentence(context: str) -> bool:
    for character in reversed(context):
        if character in NEWLINES:
            break
        if not horizontal(character) and character not in "\"'”’»›)]}」』":
            return character in ".!?"
    return True
