"""Native SSE/NDJSON AI streams, bounded output and explicit cancellation.

Only text deltas enter completions; explicit provider thinking has an opt-in
separate callback, never conversation history. Tool payloads remain private. No partial
completion is returned after cancellation, truncation, or premature stream EOF.
"""

import http.client
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .ai_provider_settings import endpoint_url
from .ai_providers import MAX_RESPONSE, CompletionCancelledError, NoRedirect, request_headers
from .ai_stream_events import read_delta

CompletionCancelled = CompletionCancelledError


@runtime_checkable
class _StreamConfig(Protocol):
    @property
    def ai_provider(self) -> str: ...

    @property
    def ai_timeout(self) -> float: ...


@runtime_checkable
class _Response(Protocol):
    def readline(self, limit: int) -> bytes: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class _StreamLimit:
    deadline: float
    cancelled: threading.Event | None

    def check(self) -> None:
        if self.cancelled is not None and self.cancelled.is_set():
            raise CompletionCancelledError("AI cleanup cancelled")
        if time.monotonic() >= self.deadline:
            raise RuntimeError("AI cleanup stream timed out")


def _abort(response: _Response) -> None:
    # HTTPResponse.close can block on a buffered read in another thread.
    connection = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
    with suppress(OSError):
        if connection is not None:
            connection.shutdown(socket.SHUT_RDWR)
        else:
            response.close()


@contextmanager
def _interrupt_blocked_reads(response: _Response, limit: _StreamLimit) -> Iterator[None]:
    finished = threading.Event()

    def monitor() -> None:
        while not finished.wait(0.05):
            try:
                limit.check()
            except RuntimeError:
                _abort(response)
                return

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        yield
    finally:
        finished.set()
        watcher.join(timeout=0.2)


def _events(response: _Response, provider: str, limit: _StreamLimit) -> Iterator[str]:
    total = 0
    lines: list[str] = []
    while True:
        limit.check()
        line = response.readline(MAX_RESPONSE + 1)
        limit.check()  # A blocked read may return data after cancellation or timeout.
        total += len(line)
        if total > MAX_RESPONSE:
            raise ValueError("AI cleanup response exceeds 1 MB")
        if not line:
            if lines:
                yield "\n".join(lines)
            return
        try:
            value = line.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError:
            raise ValueError("AI stream returned invalid UTF-8") from None
        if provider == "ollama":
            if value.strip():
                yield value
        elif not value:
            if lines:
                yield "\n".join(lines)
                lines.clear()
        elif value.startswith("data:"):
            lines.append(value[5:].lstrip(" "))
        # SSE comments, event names, and IDs are transport metadata, not model text.


def _gemini_stream_url(endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.path.endswith(":generateContent"):
        parsed = parsed._replace(
            path=parsed.path.removesuffix(":generateContent") + ":streamGenerateContent"
        )
    elif not parsed.path.endswith(":streamGenerateContent"):
        raise ValueError("Gemini streaming requires a generateContent endpoint")
    query = dict(urllib.parse.parse_qsl(parsed.query))
    query["alt"] = "sse"
    return urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(query)))


def _stream_request(
    endpoint: str, payload: Mapping[str, object], config: _StreamConfig
) -> urllib.request.Request:
    provider = config.ai_provider
    if provider == "gemini":
        endpoint = _gemini_stream_url(endpoint)
    else:
        payload = {**payload, "stream": True}
    body = json.dumps(dict(payload)).encode()
    if len(body) > MAX_RESPONSE:
        raise ValueError("AI cleanup request exceeds 1 MB")
    return urllib.request.Request(  # noqa: S310
        endpoint_url(endpoint), data=body, headers=request_headers(config, provider), method="POST"
    )


def _read_completion(
    events: Iterable[str],
    provider: str,
    limit: _StreamLimit,
    on_delta: Callable[[str], None] | None,
    on_thinking: Callable[[str], None] | None,
) -> tuple[list[str], bool]:
    pieces: list[str] = []
    complete = False
    for value in events:
        delta = read_delta(value, provider, include_thinking=on_thinking is not None)
        limit.check()
        if delta.text:
            pieces.append(delta.text)
            if on_delta is not None:
                on_delta(delta.text)
        if delta.thinking and on_thinking is not None:
            limit.check()
            on_thinking(delta.thinking)
        complete = complete or delta.done
        if delta.done and (value == "[DONE]" or provider in ("ollama", "anthropic", "gemini")):
            break
    return pieces, complete


def stream_completion(  # noqa: PLR0913 - stream callbacks and request context are independent
    endpoint: str,
    payload: Mapping[str, object],
    config: _StreamConfig,
    *,
    opener: Callable[..., Any] | None = None,
    on_delta: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    on_thinking: Callable[[str], None] | None = None,
) -> str:
    if cancel_event is not None and cancel_event.is_set():
        raise CompletionCancelledError("AI cleanup cancelled")
    request = _stream_request(endpoint, payload, config)
    open_request = opener or urllib.request.build_opener(NoRedirect()).open
    timeout = config.ai_timeout
    limit = _StreamLimit(time.monotonic() + timeout, cancel_event)
    try:
        with (
            open_request(request, timeout=timeout) as response,
            _interrupt_blocked_reads(response, limit),
        ):
            events = _events(response, config.ai_provider, limit)
            pieces, complete = _read_completion(
                events, config.ai_provider, limit, on_delta, on_thinking
            )
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"AI cleanup endpoint returned HTTP {exc.code}") from None
    except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as exc:
        limit.check()
        if isinstance(exc, ValueError):
            raise
        raise RuntimeError("AI cleanup endpoint unavailable") from None
    limit.check()
    if not complete:
        raise ValueError("AI stream ended before completion; partial text discarded")
    result = "".join(pieces).strip()
    if not result:
        raise ValueError("AI endpoint returned an empty completion")
    return result
