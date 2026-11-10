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

from .ai_providers import MAX_RESPONSE, NoRedirect, endpoint_url, request_headers, thinking_text


class CompletionCancelled(RuntimeError):
    pass


def _abort(response):
    # HTTPResponse.close alone can wait on a buffered read in another thread.
    # Shutdown its owned socket first, then let the reading thread close normally.
    connection = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
    if connection is not None:
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    else:
        response.close()  # In-memory/test transports have no socket.


def _events(response, provider, cancelled, deadline):
    total = 0
    lines = []
    while True:
        if cancelled is not None and cancelled.is_set():
            raise CompletionCancelled("AI cleanup cancelled")
        if time.monotonic() > deadline:
            raise RuntimeError("AI cleanup stream timed out")
        line = response.readline(MAX_RESPONSE + 1)
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


def _delta(data, provider):
    if not isinstance(data, dict) or data.get("error") or data.get("type") == "error":
        raise ValueError("AI stream reported an error")
    text, done = "", False
    if provider == "ollama":
        if data.get("done_reason") == "length":
            raise ValueError("AI completion truncated; increase AI maximum output tokens")
        text = data.get("message", {}).get("content", "")
        done = data.get("done") is True
    elif provider == "anthropic":
        kind = data.get("type")
        if kind == "content_block_start" and data.get("content_block", {}).get("type") == "text":
            text = data["content_block"].get("text", "")
        elif kind == "content_block_delta" and data.get("delta", {}).get("type") == "text_delta":
            text = data["delta"].get("text", "")
        elif kind == "message_delta" and data.get("delta", {}).get("stop_reason") == "max_tokens":
            raise ValueError("AI completion truncated; increase AI maximum output tokens")
        done = kind == "message_stop"
    elif provider == "gemini":
        candidates = data.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            reason = candidate.get("finishReason")
            if reason == "MAX_TOKENS":
                raise ValueError("AI completion truncated; increase AI maximum output tokens")
            if reason and reason != "STOP":
                raise ValueError("AI stream did not complete normally")
            text = "".join(
                part["text"]
                for part in candidate.get("content", {}).get("parts", [])
                if "text" in part and not part.get("thought")
            )
            done = reason == "STOP"
    else:
        choices = data.get("choices", [])
        if choices:
            choice = choices[0]
            reason = choice.get("finish_reason")
            if reason == "length":
                raise ValueError("AI completion truncated; increase provider output token limit")
            if reason and reason != "stop":
                raise ValueError("AI stream did not complete normally")
            text = choice.get("delta", {}).get("content") or ""
            done = reason == "stop"
    if not isinstance(text, str):
        raise ValueError("AI stream returned invalid text content")
    return text, done


def stream_completion(
    endpoint, payload, config, *, opener=None, on_delta=None, cancel_event=None, on_thinking=None
):
    provider = config.ai_provider
    if cancel_event is not None and cancel_event.is_set():
        raise CompletionCancelled("AI cleanup cancelled")
    if provider == "gemini":
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.path.endswith(":generateContent"):
            parsed = parsed._replace(
                path=parsed.path.removesuffix(":generateContent") + ":streamGenerateContent"
            )
        elif not parsed.path.endswith(":streamGenerateContent"):
            raise ValueError("Gemini streaming requires a generateContent endpoint")
        query = dict(urllib.parse.parse_qsl(parsed.query))
        query["alt"] = "sse"
        endpoint = urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(query)))
    else:
        payload = {**payload, "stream": True}
    body = json.dumps(payload).encode()
    if len(body) > MAX_RESPONSE:
        raise ValueError("AI cleanup request exceeds 1 MB")
    request = urllib.request.Request(
        endpoint_url(endpoint), data=body, headers=request_headers(config, provider), method="POST"
    )
    open_request = opener or urllib.request.build_opener(NoRedirect()).open
    finished = threading.Event()
    timed_out = threading.Event()
    timeout = config.ai_timeout
    deadline = time.monotonic() + timeout
    pieces, complete = [], False
    try:
        with open_request(request, timeout=timeout) as response:

            def monitor():
                while not finished.wait(0.05):
                    if cancel_event is not None and cancel_event.is_set():
                        _abort(response)
                        return
                    if time.monotonic() >= deadline:
                        timed_out.set()
                        _abort(response)
                        return

            watcher = threading.Thread(target=monitor, daemon=True)
            watcher.start()
            try:
                for value in _events(response, provider, cancel_event, deadline):
                    if value == "[DONE]" and provider == "compatible":
                        complete = True
                        break
                    try:
                        data = json.loads(value)
                        text, done = _delta(data, provider)
                        thinking = (
                            thinking_text(data, provider, streaming=True)
                            if on_thinking is not None
                            else ""
                        )
                    except (KeyError, IndexError, TypeError, AttributeError, json.JSONDecodeError):
                        raise ValueError("AI stream returned an invalid event") from None
                    if text:
                        pieces.append(text)
                        if on_delta is not None:
                            on_delta(text)
                    if thinking:
                        on_thinking(thinking)
                    complete = complete or done
                    if done and provider in ("ollama", "anthropic", "gemini"):
                        break
            finally:
                finished.set()
                watcher.join(timeout=0.2)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"AI cleanup endpoint returned HTTP {exc.code}") from None
    except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as exc:
        if cancel_event is not None and cancel_event.is_set():
            raise CompletionCancelled("AI cleanup cancelled") from None
        if timed_out.is_set():
            raise RuntimeError("AI cleanup stream timed out") from None
        if isinstance(exc, ValueError):
            raise
        raise RuntimeError("AI cleanup endpoint unavailable") from None
    if cancel_event is not None and cancel_event.is_set():
        raise CompletionCancelled("AI cleanup cancelled")
    if timed_out.is_set():
        raise RuntimeError("AI cleanup stream timed out")
    if not complete:
        raise ValueError("AI stream ended before completion; partial text discarded")
    result = "".join(pieces).strip()
    if not result:
        raise ValueError("AI endpoint returned an empty completion")
    return result
