"""Anchor actual ASR word timings to exact transcript characters, without guessing.

Whitespace may differ between an ASR word token and the rendered transcript.
Every other character, including punctuation and case, must agree in order.
"""

import math
from typing import NewType, TypedDict, TypeGuard

MAX_WORDS = 10000


CharacterOffset = NewType("CharacterOffset", int)


class AlignedWord(TypedDict):
    word: str
    start: float
    end: float
    char_start: CharacterOffset
    char_end: CharacterOffset


def _valid_time(value: object) -> TypeGuard[float]:
    # Exact numeric types: bool is not a timestamp.
    if not isinstance(value, (int, float)) or type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def align_words(text: str, words: object, *, duration: float | None = None) -> list[AlignedWord]:
    if not isinstance(text, str) or not isinstance(words, list) or len(words) > MAX_WORDS:
        raise ValueError("Invalid word alignment")
    if duration is not None and (not _valid_time(duration) or duration < 0):
        raise ValueError("Invalid alignment duration")
    output: list[AlignedWord] = []
    cursor, previous_start, previous_end = 0, 0.0, 0.0
    for word in words:
        if not isinstance(word, dict) or not isinstance(word.get("word"), str):
            raise ValueError("Invalid aligned word")  # noqa: TRY004 - one invalid alignment contract
        token = word["word"].strip()
        start, end = word.get("start"), word.get("end")
        if (
            not token
            or len(token) > 1000
            or not _valid_time(start)
            or not _valid_time(end)
            or not 0 <= start <= end
            or start < previous_start
            or end < previous_end
            or (duration is not None and end > duration + 0.001)
        ):
            raise ValueError("Invalid aligned word timing")
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        first = cursor
        # Multiword corrected spans retain a single real time interval. Match
        # their lexical characters exactly, allowing only whitespace differences.
        for character in token:
            if character.isspace():
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
            elif cursor >= len(text) or text[cursor] != character:
                raise ValueError("Word alignment does not match transcript")
            else:
                cursor += 1
        aligned: AlignedWord = {
            **word,  # Preserve provider/provenance metadata.
            "word": word["word"],
            "start": start,
            "end": end,
            "char_start": CharacterOffset(first),
            "char_end": CharacterOffset(cursor),
        }
        output.append(aligned)
        previous_start, previous_end = start, end
    if text[cursor:].strip():
        raise ValueError("Word alignment does not cover transcript")
    return output


def lexical_span(
    text: str, first: AlignedWord, last: AlignedWord
) -> tuple[CharacterOffset, CharacterOffset]:
    """Exclude edge punctuation so a replacement cannot swallow surrounding syntax."""
    start, end = int(first["char_start"]), int(last["char_end"])
    while start < end and not (text[start].isalnum() or text[start] == "_"):
        start += 1
    while end > start and not (text[end - 1].isalnum() or text[end - 1] == "_"):
        end -= 1
    return CharacterOffset(start), CharacterOffset(end)
