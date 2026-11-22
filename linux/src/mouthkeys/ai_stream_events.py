"""Normalize streaming provider events without making requests or invoking callbacks."""

import json
from dataclasses import dataclass
from typing import Any

from .ai_providers import thinking_text

_TRUNCATED = "AI completion truncated; increase AI maximum output tokens"
_INCOMPLETE = "AI stream did not complete normally"
_INVALID_EVENT = "AI stream returned an invalid event"


@dataclass(frozen=True)
class StreamDelta:
    text: str = ""
    thinking: str = ""
    done: bool = False


def _ollama(data: dict[str, Any]) -> tuple[object, bool]:
    if data.get("done_reason") == "length":
        raise ValueError(_TRUNCATED)
    return data.get("message", {}).get("content", ""), data.get("done") is True


def _anthropic(data: dict[str, Any]) -> tuple[object, bool]:
    kind = data.get("type")
    if kind == "content_block_start":
        block = data.get("content_block", {})
        if block.get("type") == "text":
            return block.get("text", ""), False
    if kind == "content_block_delta":
        delta = data.get("delta", {})
        if delta.get("type") == "text_delta":
            return delta.get("text", ""), False
    if kind == "message_delta" and data.get("delta", {}).get("stop_reason") == "max_tokens":
        raise ValueError(_TRUNCATED)
    return "", kind == "message_stop"


def _gemini(data: dict[str, Any]) -> tuple[object, bool]:
    candidates = data.get("candidates", [])
    if not candidates:
        return "", False
    candidate = candidates[0]
    reason = candidate.get("finishReason")
    if reason == "MAX_TOKENS":
        raise ValueError(_TRUNCATED)
    if reason and reason != "STOP":
        raise ValueError(_INCOMPLETE)
    text = "".join(
        part["text"]
        for part in candidate.get("content", {}).get("parts", [])
        if "text" in part and not part.get("thought")
    )
    return text, reason == "STOP"


def _compatible(data: dict[str, Any]) -> tuple[object, bool]:
    choices = data.get("choices", [])
    if not choices:
        return "", False
    choice = choices[0]
    reason = choice.get("finish_reason")
    if reason == "length":
        raise ValueError("AI completion truncated; increase provider output token limit")
    if reason and reason != "stop":
        raise ValueError(_INCOMPLETE)
    return choice.get("delta", {}).get("content") or "", reason == "stop"


_DECODERS = {"ollama": _ollama, "anthropic": _anthropic, "gemini": _gemini}


def read_delta(value: str, provider: str, *, include_thinking: bool) -> StreamDelta:
    if value == "[DONE]" and provider == "compatible":
        return StreamDelta(done=True)
    try:
        data = json.loads(value)
        if not isinstance(data, dict) or data.get("error") or data.get("type") == "error":
            raise ValueError("AI stream reported an error")
        text, done = _DECODERS.get(provider, _compatible)(data)
        thinking = thinking_text(data, provider, streaming=True) if include_thinking else ""
    except (KeyError, IndexError, TypeError, AttributeError, json.JSONDecodeError, RecursionError):
        raise ValueError(_INVALID_EVENT) from None
    if not isinstance(text, str):
        raise ValueError("AI stream returned invalid text content")  # noqa: TRY004 - protocol error
    try:
        text.encode()
        thinking.encode()
    except UnicodeEncodeError:
        raise ValueError("AI stream returned invalid Unicode") from None
    return StreamDelta(text, thinking, done)
