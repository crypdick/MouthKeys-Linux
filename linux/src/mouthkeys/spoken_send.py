"""Recognize explicit, trailing Spoken Send phrases without desktop effects."""

import re
from dataclasses import dataclass

from .dictionary import phrase_pattern


@dataclass(frozen=True)
class SendPlan:
    text: str
    should_send: bool = False


_SENTENCE_WORDS = {
    "not",
    "never",
    "cannot",
    "dont",
    "doesnt",
    "didnt",
    "wont",
    "cant",
    "shouldnt",
    "wouldnt",
    "couldnt",
    "isnt",
    "arent",
    "wasnt",
    "werent",
    "havent",
    "hasnt",
    "mustnt",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
    "i'll",
    "you'll",
    "we'll",
    "they'll",
    "he'll",
    "she'll",
    "i'd",
    "you'd",
    "we'd",
    "they'd",
    "will",
    "would",
    "can",
    "could",
    "should",
    "shall",
    "may",
    "might",
    "must",
    "do",
    "does",
    "did",
    "gonna",
    "wanna",
    "gotta",
    "to",
}
_PASSED_WORDS = {
    "just",
    "already",
    "also",
    "really",
    "actually",
    "probably",
    "definitely",
    "still",
    "ever",
    "even",
    "only",
    "then",
    "now",
    "please",
}
_OBJECT_MARKERS = {
    "thank",
    "thanks",
    "to",
    "for",
    "with",
    "at",
    "from",
    "about",
    "by",
    "of",
    "without",
}


def _sentence_owns_phrase(prefix: str) -> bool:
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


def parse_send(
    text: str,
    phrase: str = "send it",
    active: bool = False,  # noqa: FBT001, FBT002 - retain positional parser compatibility
    literal_prefix: str = "literal",
) -> SendPlan:
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
        return SendPlan(prefix, should_send=True)
    return SendPlan(text)
