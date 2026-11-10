"""Deterministic dictation formatting, including conservative Spoken Send parsing.

Behavior adapted from BenevolentFutures/MouthKeys d50ad008, by Atin Woodard,
SpokenSendParser, QuestionMarkShortcut, and ASRService spoken/literal formatting.
No desktop effects here.
"""

import re
import unicodedata
from dataclasses import dataclass

from .dictionary import Dictionary, phrase_pattern


def enabled(config, key):
    return getattr(config, key, "off") == "on"


@dataclass(frozen=True)
class SendPlan:
    text: str
    should_send: bool = False


_SENTENCE_WORDS = set(
    "not never cannot dont doesnt didnt wont cant shouldnt wouldnt couldnt isnt arent wasnt "
    "werent havent hasnt mustnt i you we they he she i'll you'll we'll they'll he'll she'll "
    "i'd you'd we'd they'd will would can could should shall may might must do does did gonna "
    "wanna gotta to".split()
)
_PASSED_WORDS = set(
    (
        "just already also really actually probably definitely still ever even only then now please"
    ).split()
)
_OBJECT_MARKERS = set("thank thanks to for with at from about by of without".split())


def _sentence_owns_phrase(prefix):
    end = len(prefix)
    for _ in range(3):
        match = re.search(r"([\w'’]+)\s*$", prefix[:end])
        if not match:
            return False
        word = match[1].casefold().replace("’", "'")
        if word in {"i", "you", "we", "they", "he", "she"}:
            preceding = re.search(r"([\w'’]+)\s*$", prefix[: match.start()])
            return not (preceding and preceding[1].casefold() in _OBJECT_MARKERS)
        if word in _SENTENCE_WORDS or word.endswith("n't"):
            return True
        if word not in _PASSED_WORDS:
            return False
        end = match.start()
    return False


def parse_send(text, phrase="send it", active=False, literal_prefix="literal"):
    if not active or not phrase.strip():
        return SendPlan(text)
    parts = re.findall(r"\w+", phrase)
    if not parts:
        return SendPlan(text)
    pattern = r"[\s,.:;!?]+".join(re.escape(word) for word in parts)
    suffix = r"[\s,.:;!?]*$"
    literal = re.search(
        phrase_pattern(literal_prefix) + r"[\s,:;]+(" + pattern + ")" + suffix,
        text,
        re.IGNORECASE,
    )
    if literal:
        return SendPlan(text[: literal.start()] + literal[1])
    tail = re.search(r"[\W_]*$", text)
    if tail and any(q in tail[0] for q in ("?", "？")):
        return SendPlan(text)
    # A repeated phrase can be a command even if its first occurrence was a sentence.
    matches = list(re.finditer(r"(?<!\w)" + pattern + r"(?!\w)", text, re.IGNORECASE))
    for match in matches:
        rest = text[match.end() :]
        if not re.fullmatch(r"(?:[\s,.:;!?]+" + pattern + r")*" + suffix, rest, re.I):
            continue
        prefix = text[: match.start()]
        if _sentence_owns_phrase(prefix):
            continue
        prefix = re.sub(
            r"(?i)(?:\bgo ahead and|\band then|\band|\blet's|\bjust)\s*[,;:]?\s*$", "", prefix
        ).rstrip(" ,;:")
        if re.fullmatch(r"(?i)(?:um|uh|er|please|just)[\s,]*", prefix):
            prefix = ""
        return SendPlan(prefix, True)
    return SendPlan(text)


_DOUBLED_Q = re.compile(
    r"(?<![\w'’\-/\\.@#$=~])(?:qq|(?:q|queue|cue)[\s.,;:!?]+(?:q|queue|cue))(?![\w'’])",
    re.I,
)
_TRAILING_Q = re.compile(r"(?<![\w'’\-/\\.@#$=~])(?:q|cue)[\s.,;:!?]*$", re.I)
_REPLACED_ENDINGS = set(",.;:!…")
_NEWLINES = set("\n\r\v\f\u0085\u2028\u2029")


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
        output += spacing if any(c in _NEWLINES for c in spacing) else " "
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


def format_text(text, config, dictionary=None, context=None):
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
        rules = {}
        for entry in [*data["punctuation"], *data["actions"]]:
            if entry.get("enabled", True):
                for alias in entry["aliases"]:
                    rules[" ".join(alias.casefold().split())] = entry["symbol"]
        if rules:
            aliases = "|".join(
                phrase_pattern(alias) for alias in sorted(rules, key=len, reverse=True)
            )
            prefix = phrase_pattern(config.literal_prefix)
            text = re.sub(
                r"[ ]*" + prefix + r"[\s,:;]+(" + aliases + ")",
                lambda m: rules[" ".join(m[1].casefold().split())],
                text,
                flags=re.I,
            )
            text = re.sub(r"([\n\t])[ ]+", r"\1", text)
    if enabled(config, "literal_formatting"):
        # Explicit syntax only. Do not turn an ordinary 'at work' into an @mention.
        text = re.sub(r"(?i)\b(?:forward\s+slash|slash)\s+([a-z][a-z0-9_-]{1,39})\b", r"/\1", text)
        text = re.sub(r"(?i)\b(?:at\s+sign|mention|tag)\s+([a-z][a-z0-9_.-]*)", r"@\1", text)
    if enabled(config, "question_mark_shortcut"):
        text = apply_question_mark_shortcut(text, config.literal_prefix)
    text = re.sub(r" {2,}", " ", text)
    if text.strip():
        text = text.strip(" ")
    if enabled(config, "smart_capitalization") and context is not None:
        preceding = context.rstrip(" \t")
        upper = not preceding or preceding[-1] in ".!?\n"
        text = _first_letter_case(text, upper=upper)
    if enabled(config, "lowercase_first"):
        text = _first_letter_case(text, upper=False)
    if enabled(config, "remove_trailing_period"):
        text = re.sub(r"(?<=\w)\.$", "", text)
    if (
        enabled(config, "space_between_dictations")
        and context
        and text
        and not context[-1].isspace()
        and not text[0].isspace()
        and text[0] not in ",.;:!?)]}"
    ):
        text = " " + text
    return text


def _first_letter_case(text, *, upper):
    for index, character in enumerate(text):
        if character.isalpha():
            changed = character.upper() if upper else character.lower()
            return text[:index] + changed + text[index + 1 :]
    return text
