import io
import urllib.error
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest

from mouthkeys import ai_providers
from mouthkeys.config import Config

from .test_provider_tools import opener


def enabled_config(**overrides):
    return replace(Config(ai_enabled="on", ai_model="test-model"), **overrides)


def no_network(*_args, **_kwargs):
    pytest.fail("Invalid request contacted the provider")


@pytest.mark.parametrize(
    ("text", "overrides", "message"),
    [
        ("", {}, "nonempty text"),
        ("raw", {"ai_model": " "}, "Choose an AI cleanup model"),
        ("raw", {"ai_provider": "unknown"}, "Unknown AI provider protocol"),
        ("x" * ai_providers.MAX_RESPONSE, {}, "request exceeds 1 MB"),
    ],
)
def test_nonstreaming_completion_validates_before_contacting_provider(text, overrides, message):
    with pytest.raises(ValueError, match=message):
        ai_providers.request_completion(text, enabled_config(**overrides), opener=no_network)


def test_already_cancelled_completion_does_not_contact_provider():
    event = Event()
    event.set()
    with pytest.raises(ai_providers.CompletionCancelledError, match="cancelled"):
        ai_providers.request_completion(
            "raw", enabled_config(), opener=no_network, cancel_event=event
        )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"malformed JSON", "invalid JSON"),
        (b"[]", "no text completion"),
        (b'{"choices":[{"message":{"content":" "}}]}', "empty completion"),
    ],
)
def test_nonstreaming_completion_rejects_a_malformed_or_empty_reply(body, message):
    with pytest.raises(ValueError, match=message):
        ai_providers.request_completion(
            "raw", enabled_config(), opener=lambda *_args, **_kwargs: io.BytesIO(body)
        )


def test_transport_failure_keeps_private_diagnostic_out_of_error():
    def unavailable(*_args, **_kwargs):
        raise urllib.error.URLError("private credentials and transcript")

    with pytest.raises(RuntimeError, match=r"^AI cleanup endpoint unavailable$"):
        ai_providers.request_completion("private transcript", enabled_config(), opener=unavailable)


def test_nonstreaming_reasoning_is_delivered_separately_from_text():
    payload = {"choices": [{"message": {"content": " clean ", "reasoning_content": "Separate"}}]}
    calls, thoughts = [], []
    reply = ai_providers.request_completion(
        "raw", enabled_config(), opener=opener(payload, calls), on_thinking=thoughts.append
    )
    assert reply == "clean"
    assert thoughts == ["Separate"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("a" * (ai_providers.MAX_HISTORY_BYTES + 1), "exceeds 256 KiB"),
        ("surrogate-\ud800", "valid Unicode"),
    ],
)
def test_history_rejects_unbounded_or_unencodable_text(content, message):
    history = [{"role": "user", "content": content}, {"role": "assistant", "content": "reply"}]
    with pytest.raises(ValueError, match=message):
        ai_providers.request_completion("raw", enabled_config(), history=history, opener=no_network)


def test_reasoning_rejects_oversized_values_and_conflicting_request_shapes():
    oversized = SimpleNamespace(ai_reasoning_parameter="think", ai_reasoning_value="x" * 257)
    with pytest.raises(ValueError, match="JSON scalar or short string"):
        ai_providers.validate_reasoning(oversized)
    conflict = SimpleNamespace(ai_reasoning_parameter="options.think", ai_reasoning_value="true")
    with pytest.raises(ValueError, match="conflicts with request structure"):
        ai_providers._apply_reasoning({"options": "other value"}, conflict)
