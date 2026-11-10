"""Anchor actual ASR word timings to exact transcript characters, without guessing.

Whitespace may differ between an ASR word token and the rendered transcript.
Every other character, including punctuation and case, must agree in order.
"""

import math

MAX_WORDS = 10000


def align_words(text, words, *, duration=None):
    if not isinstance(text, str) or not isinstance(words, list) or len(words) > MAX_WORDS:
        raise ValueError("Invalid word alignment")
    if duration is not None and (
        type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0
    ):
        raise ValueError("Invalid alignment duration")
    output, cursor, previous_start, previous_end = [], 0, 0.0, 0.0
    for word in words:
        if not isinstance(word, dict) or not isinstance(word.get("word"), str):
            raise ValueError("Invalid aligned word")
        token = word["word"].strip()
        start, end = word.get("start"), word.get("end")
        if (
            not token
            or len(token) > 1000
            or type(start) not in (int, float)
            or type(end) not in (int, float)
            or not math.isfinite(start)
            or not math.isfinite(end)
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
        output.append({**word, "char_start": first, "char_end": cursor})
        previous_start, previous_end = float(start), float(end)
    if text[cursor:].strip():
        raise ValueError("Word alignment does not cover transcript")
    return output


def lexical_span(text, first, last):
    """Exclude edge punctuation so a replacement cannot swallow surrounding syntax."""
    start, end = first["char_start"], last["char_end"]
    while start < end and not (text[start].isalnum() or text[start] == "_"):
        start += 1
    while end > start and not (text[end - 1].isalnum() or text[end - 1] == "_"):
        end -= 1
    return start, end
