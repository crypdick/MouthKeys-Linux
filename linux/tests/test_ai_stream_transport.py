"""SSE/NDJSON transport through the public completion path, without networking."""

import io
import socket
import threading
import urllib.error
from types import SimpleNamespace

import pytest

from mouthkeys import ai_providers, ai_stream
from mouthkeys.ai_providers import request_completion
from mouthkeys.ai_stream import CompletionCancelled

from .test_ai_stream import config, sse


def test_sse_multiline_and_terminal_event_at_eof_strip_metadata():
    wire = (
        b": ping\nid: example\nevent: metadata\n\n"
        b'trailer: ignored\ndata: {"choices":[{\n'
        b'data: "delta":{"content":"Hello"},"finish_reason":"stop"}]}'
    )
    deltas = []
    result = request_completion(
        "Raw", config(), opener=lambda *_a, **_k: io.BytesIO(wire), on_delta=deltas.append
    )
    assert result == "Hello"
    assert deltas == ["Hello"]


def test_ollama_ignores_blank_ndjson_lines():
    wire = b'  \n\n{"message":{"content":"Hello"},"done":true}\n'
    assert (
        request_completion("Raw", config("ollama"), opener=lambda *_a, **_k: io.BytesIO(wire))
        == "Hello"
    )


@pytest.mark.parametrize(
    ("wire", "error"),
    [
        (b"data: \xff\n\n", "invalid UTF-8"),
        (b"data: {invalid\n\n", "invalid event"),
        (sse("[DONE]"), "empty completion"),
        (sse({"choices": [{"delta": [1]}]}), "invalid event"),
    ],
)
def test_invalid_or_empty_stream_does_not_emit(wire, error):
    def unexpected(_):
        pytest.fail("Unvalidated text reached a callback")

    with pytest.raises(ValueError, match=error):
        request_completion(
            "Raw", config(), opener=lambda *_a, **_k: io.BytesIO(wire), on_delta=unexpected
        )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (urllib.error.HTTPError("https://example.test", 503, "private", {}, None), "HTTP 503"),
        (urllib.error.URLError("private"), "endpoint unavailable"),
    ],
)
def test_transport_error_does_not_expose_provider_payload(error, expected):
    def opener(*_args, **_options):
        raise error

    with pytest.raises(RuntimeError, match=expected) as raised:
        request_completion("Raw", config(), opener=opener)
    assert "private" not in str(raised.value)


def test_cancel_between_provider_selection_and_transport_never_opens(monkeypatch):
    cancelled = threading.Event()
    validate_endpoint = ai_providers.endpoint_url

    def validate(value):
        cancelled.set()
        return validate_endpoint(value)

    monkeypatch.setattr(ai_providers, "endpoint_url", validate)
    with pytest.raises(CompletionCancelled):
        request_completion(
            "Raw",
            config(),
            cancel_event=cancelled,
            opener=lambda *_a, **_k: pytest.fail("Cancelled request reached provider"),
        )


def test_oversized_request_fails_before_contacting_provider():
    with pytest.raises(ValueError, match="request exceeds 1 MB"):
        request_completion(
            "x" * 1024 * 1024,
            config(),
            opener=lambda *_a, **_k: pytest.fail("Oversized request reached provider"),
        )


def test_gemini_preserves_existing_stream_endpoint_and_query_options():
    cfg = config("gemini")
    cfg.ai_endpoint = "https://example.test/model:streamGenerateContent?other=1&alt=old"
    observed = []
    wire = sse({"candidates": [{"content": {"parts": [{"text": "x"}]}, "finishReason": "STOP"}]})

    def opener(request, **_options):
        observed.append(request.full_url)
        return io.BytesIO(wire)

    assert request_completion("Raw", cfg, opener=opener) == "x"
    assert observed == ["https://example.test/model:streamGenerateContent?other=1&alt=sse"]


def test_gemini_rejects_endpoint_without_generation_method():
    cfg = config("gemini")
    cfg.ai_endpoint = "https://example.test/model:otherMethod"
    with pytest.raises(ValueError, match="requires a generateContent endpoint"):
        request_completion(
            "Raw", cfg, opener=lambda *_a, **_k: pytest.fail("Invalid endpoint used")
        )


@pytest.mark.parametrize("stop_reason", ["cancel", "timeout"])
def test_blocked_transport_shuts_down_its_socket(monkeypatch, stop_reason):
    cancelled, entered, release = threading.Event(), threading.Event(), threading.Event()
    elapsed = [0.0]
    monkeypatch.setattr(ai_stream, "time", SimpleNamespace(monotonic=lambda: elapsed[0]))
    reached = []

    def shutdown(direction):
        reached.append(direction)
        release.set()
        raise OSError("Already disconnected")

    class Blocked(io.BytesIO):
        fp = SimpleNamespace(raw=SimpleNamespace(_sock=SimpleNamespace(shutdown=shutdown)))

        def readline(self, limit=-1):
            entered.set()
            assert release.wait(2), "Monitor did not interrupt read"
            raise OSError("Transport disconnected")

    errors = []

    def run():
        try:
            request_completion(
                "Raw", config(), cancel_event=cancelled, opener=lambda *_a, **_k: Blocked()
            )
        except RuntimeError as error:
            errors.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(1)
        if stop_reason == "cancel":
            cancelled.set()
        else:
            elapsed[0] = 31.0
        worker.join(2)
        assert not worker.is_alive()
    finally:
        release.set()
        worker.join(2)
    assert reached == [socket.SHUT_RDWR]
    (error,) = errors
    if stop_reason == "cancel":
        assert isinstance(error, CompletionCancelled)
    else:
        assert str(error) == "AI cleanup stream timed out"
