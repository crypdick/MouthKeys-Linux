import io
import json
import urllib.error
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mouthkeys import ai_providers, provider_tools
from mouthkeys.config import Config


def opener(payload, calls):
    def request(req, **kwargs):
        calls.append((req, kwargs))
        return io.BytesIO(json.dumps(payload).encode())

    return request


@pytest.mark.parametrize(
    "provider,endpoint,payload,expected,path",
    [
        (
            "compatible",
            "https://server/v1/chat/completions",
            {"data": [{"id": "custom"}]},
            ["custom"],
            "/v1/models",
        ),
        (
            "anthropic",
            "https://server/v1/messages",
            {"data": [{"id": "claude"}]},
            ["claude"],
            "/v1/models",
        ),
        (
            "ollama",
            "http://localhost:11434/api/chat",
            {"models": [{"name": "local:latest"}]},
            ["local:latest"],
            "/api/tags",
        ),
        (
            "gemini",
            "https://server/v1beta/models/{model}:generateContent",
            {
                "models": [
                    {
                        "name": "models/text-model",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {"name": "models/embed", "supportedGenerationMethods": ["embedContent"]},
                ]
            },
            ["text-model"],
            "/v1beta/models",
        ),
    ],
)
def test_explicit_model_discovery_uses_native_routes_and_filters_gemini(
    provider, endpoint, payload, expected, path, monkeypatch
):
    config = replace(Config(), ai_provider=provider, ai_endpoint=endpoint)
    monkeypatch.setenv(config.ai_key_env, "private-secret")
    calls = []
    assert provider_tools.list_models(config, opener=opener(payload, calls)) == expected
    request, options = calls[0]
    assert request.full_url.endswith(path)
    assert request.method == "GET"
    assert options == {"timeout": 30}
    assert "private-secret" not in request.full_url


def test_custom_discovery_route_requires_explicit_same_origin_before_network():
    config = replace(Config(), ai_endpoint="https://server/custom/route")

    def fail(*a, **kw):
        pytest.fail("Unexpected network")

    with pytest.raises(ValueError, match="explicit"):
        provider_tools.list_models(config, opener=fail)
    with pytest.raises(ValueError, match="origin"):
        provider_tools.list_models(
            replace(config, ai_models_endpoint="https://other/models"), opener=fail
        )
    calls = []
    provider_tools.list_models(
        replace(config, ai_models_endpoint="https://server/catalog"),
        opener=opener({"data": []}, calls),
    )
    assert calls[0][0].full_url == "https://server/catalog"


def test_pagination_is_same_origin_and_repeated_cursors_fail_closed():
    config = replace(Config(), ai_provider="anthropic", ai_endpoint="https://server/v1/messages")
    calls = []
    responses = iter(
        [{"data": [{"id": "one"}], "has_more": True, "last_id": "one"}, {"data": [{"id": "two"}]}]
    )

    def request(req, **kwargs):
        calls.append(req.full_url)
        return io.BytesIO(json.dumps(next(responses)).encode())

    assert provider_tools.list_models(config, opener=request) == ["one", "two"]
    assert calls == ["https://server/v1/models", "https://server/v1/models?after_id=one"]
    with pytest.raises(ValueError, match="repeated"):
        provider_tools.list_models(
            config, opener=opener({"data": [], "has_more": True, "last_id": "same"}, [])
        )


def test_verify_sends_fixed_test_no_saved_prompt_or_transcript_and_bounds_output(monkeypatch):
    config = replace(Config(), ai_model="model", ai_prompt="private saved instruction")
    monkeypatch.setenv(config.ai_key_env, "secret")
    calls = []
    summary = provider_tools.verify(
        config,
        opener=opener(
            {"choices": [{"message": {"content": "Untrusted provider reply secret"}}]}, calls
        ),
    )
    assert summary == {"ok": True, "provider": "compatible", "model": "model"}
    body = json.loads(calls[0][0].data)
    assert body["max_tokens"] == 32
    assert "private saved instruction" not in calls[0][0].data.decode()
    assert "secret" not in json.dumps(summary)


def test_discovery_errors_do_not_echo_secrets():
    def request(*a, **kw):
        raise urllib.error.HTTPError("https://secret", 401, "secret", {}, io.BytesIO(b"secret"))

    with pytest.raises(RuntimeError) as error:
        provider_tools.list_models(Config(), opener=request)
    assert str(error.value) == "AI cleanup endpoint returned HTTP 401"


@pytest.mark.parametrize(
    "parameter,value,expected",
    [
        ("reasoning_effort", "high", "high"),
        ("enable_thinking", "false", False),
        ("think", '"low"', "low"),
        ("generationConfig.thinkingConfig.thinkingBudget", "0", 0),
        ("thinking.type", '"adaptive"', "adaptive"),
    ],
)
def test_reasoning_parameters_support_controlled_scalars_without_overwriting_requests(
    parameter, value, expected
):
    config = SimpleNamespace(ai_reasoning_parameter=parameter, ai_reasoning_value=value)
    assert ai_providers.validate_reasoning(config) == (parameter, expected)
    body = ai_providers._apply_reasoning({"model": "original", "messages": ["safe"]}, config)
    assert body["model"] == "original"
    assert body["messages"] == ["safe"]
    actual = body
    for part in parameter.split("."):
        actual = actual[part]
    assert actual == expected


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("model", "other"),
        ("messages", "[]"),
        ("Authorization", "secret"),
        ("reasoning_effort", "{}"),
        ("thinking", "NaN"),
        ("", "high"),
    ],
)
def test_reserved_parameters_and_non_scalar_values_rejected(parameter, value):
    with pytest.raises(ValueError):
        ai_providers.validate_reasoning(
            SimpleNamespace(ai_reasoning_parameter=parameter, ai_reasoning_value=value)
        )
