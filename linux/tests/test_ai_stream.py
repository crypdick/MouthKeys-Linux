import io
import json
import threading
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from mouthkeys.ai_providers import PROVIDER_ENDPOINTS, request_completion
from mouthkeys.ai_stream import CompletionCancelled
from mouthkeys.config import Config


def config(provider="compatible"):
    return SimpleNamespace(
        **{
            **asdict(Config()),
            "ai_enabled": "on",
            "ai_streaming": "on",
            "ai_model": "test",
            "ai_provider": provider,
            "ai_endpoint": PROVIDER_ENDPOINTS[provider],
        }
    )


def sse(*events):
    return b"".join(
        b"data: "
        + (event.encode() if isinstance(event, str) else json.dumps(event).encode())
        + b"\n\n"
        for event in events
    )


@pytest.mark.parametrize(
    "provider,wire",
    [
        (
            "compatible",
            sse(
                {"choices": [{"delta": {"reasoning_content": "private thought"}}]},
                {"choices": [{"delta": {"content": "Hello "}}]},
                {"choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}]},
                "[DONE]",
            ),
        ),
        (
            "anthropic",
            sse(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "private thought"},
                },
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello "}},
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "world"}},
                {"type": "message_stop"},
            ),
        ),
        (
            "gemini",
            sse(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"thought": True, "text": "private thought"},
                                    {"text": "Hello "},
                                ]
                            }
                        }
                    ]
                },
                {
                    "candidates": [
                        {"content": {"parts": [{"text": "world"}]}, "finishReason": "STOP"}
                    ]
                },
            ),
        ),
        (
            "ollama",
            b'{"message":{"thinking":"private thought","content":"Hello "},"done":false}\n'
            b'{"message":{"content":"world"},"done":true}\n',
        ),
    ],
)
def test_native_streams_emit_only_text_deltas_and_return_complete_text(provider, wire):
    calls, deltas = [], []

    def opener(request, **kwargs):
        calls.append(request)
        return io.BytesIO(wire)

    assert (
        request_completion("Raw", config(provider), opener=opener, on_delta=deltas.append)
        == "Hello world"
    )
    assert deltas == ["Hello ", "world"]
    body = json.loads(calls[0].data)
    if provider == "gemini":
        assert calls[0].full_url.endswith(":streamGenerateContent?alt=sse")
        assert "stream" not in body
    else:
        assert body["stream"] is True


def test_partial_stream_eof_never_returns_transcript():
    wire = sse({"choices": [{"delta": {"content": "partial"}}]})
    with pytest.raises(ValueError, match="partial text discarded"):
        request_completion("Raw", config(), opener=lambda *a, **kw: io.BytesIO(wire))


@pytest.mark.parametrize(
    "provider,event",
    [
        ("compatible", {"choices": [{"delta": {"content": "cut"}, "finish_reason": "length"}]}),
        ("anthropic", {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}}),
        ("gemini", {"candidates": [{"finishReason": "MAX_TOKENS"}]}),
        ("ollama", {"done": True, "done_reason": "length"}),
    ],
)
def test_stream_truncation_rejected(provider, event):
    wire = json.dumps(event).encode() + b"\n" if provider == "ollama" else sse(event)
    with pytest.raises(ValueError, match="truncated"):
        request_completion("Raw", config(provider), opener=lambda *a, **kw: io.BytesIO(wire))


def test_cancel_before_request_never_opens_network():
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(CompletionCancelled):
        request_completion(
            "Raw",
            config(),
            cancel_event=cancelled,
            opener=lambda *a, **kw: pytest.fail("Network request"),
        )


def test_cancel_interrupts_blocked_stream_and_closes_transport():
    cancelled, entered, release = threading.Event(), threading.Event(), threading.Event()
    closed = []

    class Blocked:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def readline(self, limit):
            entered.set()
            assert release.wait(2), "Cancellation did not interrupt blocked read"
            return b""

        def close(self):
            closed.append(True)
            release.set()

    result = []

    def run():
        try:
            request_completion(
                "Raw", config(), opener=lambda *a, **kw: Blocked(), cancel_event=cancelled
            )
        except Exception as error:
            result.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    assert entered.wait(1)
    cancelled.set()
    worker.join(2)
    assert not worker.is_alive()
    assert isinstance(result[0], CompletionCancelled)
    assert closed


def test_stream_errors_and_oversized_payloads_do_not_echo_provider_content():
    wire = sse({"type": "error", "error": {"message": "private secret"}})
    with pytest.raises(ValueError) as error:
        request_completion("Raw", config(), opener=lambda *a, **kw: io.BytesIO(wire))
    assert str(error.value) == "AI stream reported an error"
    with pytest.raises(ValueError, match="exceeds 1 MB"):
        request_completion(
            "Raw", config(), opener=lambda *a, **kw: io.BytesIO(b"x" * (1024 * 1024 + 1))
        )


@pytest.mark.parametrize(
    "provider,event,answer,stop",
    [
        (
            "compatible",
            {"choices": [{"delta": {"reasoning_content": "External thought"}}]},
            {"choices": [{"delta": {"content": "Answer"}}]},
            "[DONE]",
        ),
        (
            "anthropic",
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "External thought"},
            },
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Answer"}},
            {"type": "message_stop"},
        ),
        (
            "gemini",
            {
                "candidates": [
                    {"content": {"parts": [{"thought": True, "text": "External thought"}]}}
                ]
            },
            {"candidates": [{"content": {"parts": [{"text": "Answer"}]}}]},
            {"candidates": [{"finishReason": "STOP"}]},
        ),
        (
            "ollama",
            {"message": {"thinking": "External thought"}, "done": False},
            {"message": {"content": "Answer"}, "done": False},
            {"done": True},
        ),
    ],
)
def test_stream_history_and_thinking_channels_are_separate(provider, event, answer, stop):
    wire = (
        b"".join(json.dumps(part).encode() + b"\n" for part in (event, answer, stop))
        if provider == "ollama"
        else sse(event, answer, stop)
    )
    history = [
        {"role": "user", "content": "First request"},
        {"role": "assistant", "content": "First response"},
    ]
    thoughts, deltas, requests = [], [], []

    def opener(request, **kwargs):
        requests.append(json.loads(request.data))
        return io.BytesIO(wire)

    assert (
        request_completion(
            "Follow up",
            config(provider),
            opener=opener,
            history=history,
            on_thinking=thoughts.append,
            on_delta=deltas.append,
        )
        == "Answer"
    )
    assert thoughts == ["External thought"]
    assert deltas == ["Answer"]
    assert "External thought" not in json.dumps(requests)
    body = requests[0]
    conversation = body["contents"] if provider == "gemini" else body["messages"]
    assert len(conversation) == (4 if provider in ("compatible", "ollama") else 3)


def test_invalid_thinking_event_never_becomes_answer_or_error_content():
    wire = sse({"choices": [{"delta": {"reasoning_content": {"secret": "private"}}}]}, "[DONE]")
    with pytest.raises(ValueError, match="invalid thinking content") as error:
        request_completion(
            "Raw",
            config(),
            opener=lambda *a, **kw: io.BytesIO(wire),
            on_thinking=lambda text: pytest.fail("Invalid callback"),
        )
    assert "private" not in str(error.value)


def test_nonstream_cancel_suppresses_late_thinking_and_completion():
    cancelled = threading.Event()
    cfg = config()
    cfg.ai_streaming = "off"

    def opener(*args, **kwargs):
        cancelled.set()
        return io.BytesIO(
            b'{"choices":[{"message":{"content":"late","reasoning_content":"late"}}]}'
        )

    with pytest.raises(CompletionCancelled):
        request_completion(
            "Raw",
            cfg,
            opener=opener,
            cancel_event=cancelled,
            on_thinking=lambda text: pytest.fail("Late callback"),
        )
