"""Bounded display-only decoding of the root Command explanation."""

import json
from collections.abc import Callable
from typing import Literal

MAX_RESULT_BYTES = 128 * 1024
MAX_TOKEN_WORK = 1024 * 1024
FENCE_START = "```json\n"
FENCE_END = "\n```"
_WHITESPACE = " \t\r\n"
_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


def _after_space(text: str, offset: int) -> int:
    while offset < len(text) and text[offset] in _WHITESPACE:
        offset += 1
    return offset


def _codepoint(digits: str) -> int:
    if any(digit not in "0123456789abcdefABCDEF" for digit in digits):
        raise ValueError("Invalid JSON Unicode escape")
    return int(digits, 16)


def _escaped_character(text: str, offset: int) -> tuple[str, int] | None:
    if offset + 1 == len(text):
        return None
    escape = text[offset + 1]
    if escape in _ESCAPES:
        return _ESCAPES[escape], offset + 2
    if escape != "u":
        raise ValueError("Invalid JSON escape")
    end = offset + 6
    if end > len(text):
        return None
    code = _codepoint(text[offset + 2 : end])
    if 0xD800 <= code <= 0xDBFF:
        if end + 6 > len(text):
            return None
        if text[end : end + 2] != "\\u":
            raise ValueError("Missing low surrogate")
        low = _codepoint(text[end + 2 : end + 6])
        if not 0xDC00 <= low <= 0xDFFF:
            raise ValueError("Invalid low surrogate")
        code = 0x10000 + ((code - 0xD800) << 10) + low - 0xDC00
        end += 6
    elif 0xDC00 <= code <= 0xDFFF:
        raise ValueError("Unexpected low surrogate")
    return chr(code), end


class CommandExplanationPreview:
    """Display a root explanation string without parsing an actionable command."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        self.emit = emit
        self.buffer = ""
        self.bytes = 0
        self.cursor = 0
        self.phase: Literal[
            "start", "object", "key", "colon", "value", "separator", "text", "done"
        ] = "start"
        self.key: str | None = None
        self.work = 0
        self.decoder = json.JSONDecoder()

    def feed(self, piece: str) -> None:
        if self.phase == "done":
            return
        try:
            self.bytes += len(piece.encode())
        except UnicodeEncodeError:
            self.phase = "done"
            return
        if self.bytes > MAX_RESULT_BYTES:
            self.phase = "done"
            return
        self.buffer += piece
        while self.cursor < len(self.buffer):
            if self.phase == "text":
                self._read_text()
                break
            self.cursor = _after_space(self.buffer, self.cursor)
            if self.cursor == len(self.buffer) or not self._advance():
                break

    def _advance(self) -> bool:
        char = self.buffer[self.cursor]
        if self.phase in ("start", "object"):
            return self._read_start(char)
        if self.phase in ("key", "value"):
            return self._read_token(char)
        if self.phase == "colon" and char == ":":
            self.phase = "value"
        elif self.phase == "separator" and char == ",":
            self.phase = "key"
        else:
            self.phase = "done"
        self.cursor += 1
        return self.phase != "done"

    def _read_start(self, char: str) -> bool:
        if self.phase == "start" and char == "`":
            pending = self.buffer[self.cursor :]
            if pending.startswith(FENCE_START):
                self.cursor += len(FENCE_START)
                self.phase = "object"
                return True
            if not FENCE_START.startswith(pending):
                self.phase = "done"
            return False
        if char != "{":
            self.phase = "done"
            return False
        self.phase = "key"
        self.cursor += 1
        return True

    def _read_token(self, char: str) -> bool:
        if self.phase == "value" and self.key == "explanation":
            self.cursor += 1
            self.phase = "text" if char == '"' else "done"
            return self.phase != "done"

        # Raw decode skips nested lookalikes. Bound repeated attempts on incomplete values.
        self.work += len(self.buffer) - self.cursor
        if self.work > MAX_TOKEN_WORK:
            self.phase = "done"
            return False
        try:
            value, end = self.decoder.raw_decode(self.buffer, self.cursor)
        except json.JSONDecodeError:
            return False
        except RecursionError:
            self.phase = "done"
            return False
        if self.phase != "key":
            return self._read_separator(end)
        if isinstance(value, str):
            self.key, self.cursor, self.phase = value, end, "colon"
        else:
            self.phase = "done"
        return self.phase != "done"

    def _read_separator(self, end: int) -> bool:
        # A preceding value is not complete until its root delimiter arrives.
        end = _after_space(self.buffer, end)
        if end == len(self.buffer):
            return False
        if self.buffer[end] not in ",}":
            self.phase = "done"
            return False
        self.cursor, self.phase = end, "separator"
        return True

    def _read_text(self) -> None:
        output: list[str] = []
        while self.cursor < len(self.buffer):
            char = self.buffer[self.cursor]
            if char == '"' or ord(char) < 32:
                self.phase = "done"
                break
            if char != "\\":
                output.append(char)
                self.cursor += 1
                continue
            try:
                escaped = _escaped_character(self.buffer, self.cursor)
            except ValueError:
                self.phase = "done"
                break
            if escaped is None:
                break
            char, self.cursor = escaped
            output.append(char)
        if output:
            self.emit("".join(output))
