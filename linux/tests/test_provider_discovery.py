import io
import json
from dataclasses import replace
from urllib.parse import parse_qsl, urlsplit

import pytest

from mouthkeys import provider_tools
from mouthkeys.config import Config

from .test_provider_tools import opener


@pytest.mark.parametrize(
    ("provider", "endpoint", "page", "cursor"),
    [
        (
            "anthropic",
            "https://server/v1/messages",
            {"data": [], "has_more": True, "last_id": "page+2"},
            "after_id",
        ),
        (
            "gemini",
            "https://server/v1/models/{model}:generateContent",
            {"models": [], "nextPageToken": "page+2"},
            "pageToken",
        ),
    ],
)
def test_model_pagination_keeps_duplicate_and_blank_deployment_parameters(
    provider, endpoint, page, cursor
):
    config = replace(
        Config(),
        ai_provider=provider,
        ai_endpoint=endpoint + f"?deployment=v1&flag=&deployment=v2&{cursor}=initial",
    )
    received = []

    def read(request, **_):
        received.append(parse_qsl(urlsplit(request.full_url).query, keep_blank_values=True))
        payload = page if len(received) == 1 else {"data": [], "models": []}
        return io.BytesIO(json.dumps(payload).encode())

    assert provider_tools.list_models(config, opener=read) == []
    original = [("deployment", "v1"), ("flag", ""), ("deployment", "v2")]
    assert received == [[*original, (cursor, "initial")], [*original, (cursor, "page+2")]]


def test_gemini_rejects_malformed_capabilities_instead_of_leaking_python_error():
    config = replace(
        Config(),
        ai_provider="gemini",
        ai_endpoint="https://server/v1/models/{model}:generateContent",
    )
    payload = {"models": [{"name": "models/example", "supportedGenerationMethods": None}]}
    with pytest.raises(ValueError, match="invalid model capabilities"):
        provider_tools.list_models(config, opener=opener(payload, []))


def test_connection_probe_reports_malformed_provider_completion_as_validation_error():
    config = replace(
        Config(), ai_model="test-model", ai_endpoint="https://server/v1/chat/completions"
    )
    with pytest.raises(ValueError, match="no text completion"):
        provider_tools.verify(config, opener=opener({"choices": None}, []))


@pytest.mark.parametrize(
    ("provider", "body", "message"),
    [
        ("compatible", [], "no model list"),
        ("compatible", {"data": None}, "no model list"),
        ("compatible", {"data": ["not an object"]}, "invalid model"),
        ("compatible", {"data": [{"id": "bad\nidentifier"}]}, "invalid model identifier"),
        (
            "gemini",
            {"models": [{"name": 4, "supportedGenerationMethods": ["generateContent"]}]},
            "invalid model identifier",
        ),
        ("gemini", {"models": [], "nextPageToken": 3}, "invalid page cursor"),
        ("anthropic", {"data": [], "has_more": True, "last_id": ""}, "invalid page cursor"),
    ],
)
def test_discovery_rejects_malformed_catalog_entries_and_cursors(provider, body, message):
    config = replace(
        Config(),
        ai_provider=provider,
        ai_endpoint="https://server/completion",
        ai_models_endpoint="https://server/catalog",
    )
    with pytest.raises(ValueError, match=message):
        provider_tools.list_models(config, opener=opener(body, []))


def test_catalog_with_more_than_1000_distinct_ids_is_rejected():
    payload = {"data": [{"id": f"model-{index}"} for index in range(1001)]}
    with pytest.raises(ValueError, match="exceeds 1000 models"):
        provider_tools.list_models(Config(), opener=opener(payload, []))


def test_catalog_pagination_is_bounded_even_when_each_cursor_is_unique():
    config = replace(Config(), ai_provider="anthropic", ai_endpoint="https://server/v1/messages")
    requests = []

    def next_page(request, **_):
        requests.append(request.full_url)
        payload = {"data": [], "has_more": True, "last_id": f"page-{len(requests)}"}
        return io.BytesIO(json.dumps(payload).encode())

    with pytest.raises(ValueError, match="exceeds ten pages"):
        provider_tools.list_models(config, opener=next_page)
    assert len(requests) == 10
    assert len(set(requests)) == 10


def test_duplicate_ids_keep_first_server_order_and_ollama_accepts_model_field():
    config = replace(Config(), ai_provider="ollama", ai_endpoint="http://localhost:11434/api/chat")
    result = provider_tools.list_models(
        config, opener=opener({"models": [{"model": "one"}, {"name": "two"}, {"name": "one"}]}, [])
    )
    assert result == ["one", "two"]


def test_verification_rejects_empty_model_before_network():
    with pytest.raises(ValueError, match="Choose an AI cleanup model"):
        provider_tools.verify(
            replace(Config(), ai_model="models/"),
            opener=lambda *_args, **_kwargs: pytest.fail("Verification contacted provider"),
        )


@pytest.mark.parametrize("text", ["", None, "OK"])
def test_native_verification_uses_the_native_limit_and_requires_text(text):
    config = replace(
        Config(),
        ai_provider="ollama",
        ai_endpoint="http://localhost:11434/api/chat",
        ai_model="local",
    )
    calls = []
    probe = opener({"message": {"content": text}}, calls)
    if text:
        assert provider_tools.verify(config, opener=probe)["ok"]
    else:
        with pytest.raises(ValueError, match="no text completion"):
            provider_tools.verify(config, opener=probe)
    request, _ = calls[0]
    assert json.loads(request.data)["options"] == {"num_predict": 32}
