"""Whitespace shared by dictation formatting rules."""

NEWLINES = set("\n\r\v\f\u0085\u2028\u2029")


def horizontal(text: str) -> bool:
    return bool(text) and all(c.isspace() and c not in NEWLINES for c in text)


def rstrip_horizontal(text: str) -> str:
    end = len(text)
    while end and horizontal(text[end - 1]):
        end -= 1
    return text[:end]
