"""Deterministic dictation formatting, including conservative Spoken Send parsing.

Behavior adapted from BenevolentFutures/MouthKeys d50ad008:
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
            if character in _NEWLINES:
                break
            if _horizontal(character) or character in "\"'”’»›)]}」』":
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


def _horizontal(text):
    return bool(text) and all(c.isspace() and c not in _NEWLINES for c in text)


def _rstrip_horizontal(text):
    end = len(text)
    while end and _horizontal(text[end - 1]):
        end -= 1
    return text[:end]


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
                    if cursor >= len(tokens) or not _horizontal(tokens[cursor]):
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
            if kind != "text" or not _horizontal(value):
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
            before = _rstrip_horizontal(parts[index - 1][0])
            if before.endswith(".") and not before.endswith(".."):
                parts[index - 1][0] = before[:-1]
        if index + 1 < len(parts) and parts[index + 1][1] == "text":
            after = parts[index + 1][0]
            stripped = after
            while stripped and _horizontal(stripped[0]):
                stripped = stripped[1:]
            allowed = ".," if kind in {"line", "paragraph"} else "."
            if stripped and stripped[0] in allowed and not stripped.startswith(stripped[0] * 2):
                stripped = stripped[1:]
                while stripped and _horizontal(stripped[0]):
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
            result = _rstrip_horizontal(result)
        if kind == "around" and result and result[-1] not in _NEWLINES:
            result += " "
        result += symbol
        index += 1
        if kind != "right":
            while index < len(parts) and parts[index][1] == "text" and _horizontal(parts[index][0]):
                index += 1
        if kind == "around" and any(
            kind != "text" or (value and not _horizontal(value)) for value, kind in parts[index:]
        ):
            result += " "
    return result


_SLASH_REJECTED = set(
    "a an and as at back backslash be been being bin by comma desktop documents dot downloads "
    "etc for forward from home in is library local mark of on or period private question quote "
    "quotes semicolon slash slashes source sources src the tmp to user users usr var volumes "
    "was were with without".split()
)
_SLASH_LEAD_INS = set(
    (
        "call choose do enter execute open pick press run say select send start try type use write"
    ).split()
)
_MENTION_REJECTED = set(
    "a an airport breakfast brunch class dinner home hotel house lunch meeting night noon office "
    "place restaurant school shop store the today tomorrow work yesterday".split()
)
_MENTION_LEAD_INS = set(
    "add ask cc dm hello hey hi invite message notify ping send tag tell".split()
)


def _lead_in(text, start, allowed):
    prefix = text[:start].strip()
    if not prefix or prefix[-1] in ".!?:;([{":
        return True
    words = re.findall(r"[a-zA-Z0-9_-]+", prefix)
    return not words or words[-1].casefold() in allowed


def apply_literal_formatting(text, *, app_name=None, app_id=None, window_title=None):
    """Upstream's opt-in slash commands and explicit/contextual mentions."""
    haystack = " ".join(value or "" for value in (app_name, app_id, window_title)).casefold()
    relaxed = any(app in haystack for app in ("slack", "discord", "teams"))
    token = r"([A-Za-z][A-Za-z0-9_-]{1,39})(?![A-Za-z0-9_-])"
    for pattern, spoken in (
        (r"(?<!\w)/\s+" + token, False),
        (r"(?<!\w)(?:forward\s+slash|slash)\s+" + token, True),
    ):

        def slash(match, spoken=spoken, source=text):
            command = match[1].lower()
            if command in _SLASH_REJECTED or not re.search(r"[a-z0-9_-]$", command):
                return match[0]
            if spoken and not _lead_in(source, match.start(), _SLASH_LEAD_INS):
                return match[0]
            return "/" + command

        text = re.sub(pattern, slash, text, flags=re.I)
    name = r"([A-Za-z][A-Za-z0-9_.-]*(?:\s+[A-Z][A-Za-z0-9_.-]*){0,2})(?![A-Za-z0-9_.-])"
    for pattern, is_relaxed in (
        (r"(?<![\w@])(?i:at\s+(?:sign|the\s+rate)|tag|mention)\s+" + name, False),
        (r"(?<![\w@])[Aa]t\s+" + name, True),
    ):
        if is_relaxed and not relaxed:
            continue

        def mention(match, is_relaxed=is_relaxed, source=text):
            words = match[1].split(" ")
            if any(word.lower() in _MENTION_REJECTED for word in words):
                return match[0]
            if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", word) for word in words):
                return match[0]
            if is_relaxed and (
                any(not word[:1].isupper() for word in words)
                or not _lead_in(source, match.start(), _MENTION_LEAD_INS)
            ):
                return match[0]
            if source[match.end() : match.end() + 1] in {"'", "’"}:
                return match[0]
            return "@" + match[1].strip()

        text = re.sub(pattern, mention, text)
    return text


def literal_autocomplete_spacing(text, *, app_name=None, app_id=None, window_title=None):
    haystack = " ".join(value or "" for value in (app_name, app_id, window_title)).casefold()
    if not text or not _horizontal(text[-1]):
        return text
    stripped = _rstrip_horizontal(text)
    if any(app in haystack for app in ("codex", "chatgpt", "claude", "cursor", "windsurf")):
        if re.fullmatch(r"/[A-Za-z][A-Za-z0-9_-]{1,39}", stripped):
            return stripped
    if any(app in haystack for app in ("slack", "discord", "teams")):
        if re.search(
            r"(?<![\w@])@[A-Za-z][A-Za-z0-9_.-]*(?:\s+[A-Z][A-Za-z0-9_.-]*){0,2}$", stripped
        ):
            return stripped
    return text
