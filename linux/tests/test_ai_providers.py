import io
import json
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, replace

import pytest

from mouthkeys import ai_providers, profiles
from mouthkeys.config import Config


@pytest.mark.parametrize(
    "provider,response,header",
    [
        ("compatible", {"choices": [{"message": {"content": " Cleaned."}}]}, "Authorization"),
        (
            "anthropic",
            {
                "content": [
                    {"type": "thinking", "thinking": "Private"},
                    {"type": "text", "text": "Cleaned."},
                ]
            },
            "X-api-key",
        ),
        (
            "gemini",
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"thought": True, "text": "Private"}, {"text": "Cleaned."}]
                        }
                    }
                ]
            },
            "X-goog-api-key",
        ),
        ("ollama", {"message": {"content": "Cleaned."}, "done": True}, "Authorization"),
    ],
)
def test_native_protocols(provider, response, header, monkeypatch):
    config = replace(
        Config(),
        ai_enabled="on",
        ai_provider=provider,
        ai_endpoint=ai_providers.PROVIDER_ENDPOINTS[provider],
        ai_model="test-model",
        ai_max_tokens=1234,
        ai_timeout=17,
    )
    monkeypatch.setenv(config.ai_key_env, "test-secret")
    requests = []

    def opener(request, **kwargs):
        requests.append((request, kwargs))
        return io.BytesIO(json.dumps(response).encode())

    assert ai_providers.request_completion("Raw words.", config, opener=opener) == "Cleaned."
    request, options = requests[0]
    assert request.get_header(header) == (
        "test-secret" if provider in ("anthropic", "gemini") else "Bearer test-secret"
    )
    assert "test-secret" not in request.full_url
    assert options == {"timeout": 17}
    body = json.loads(request.data)
    if provider == "anthropic":
        assert body["max_tokens"] == 1234
        assert body["system"] == config.ai_prompt
        assert request.get_header("Anthropic-version") == "2023-06-01"
        assert body["messages"] == [{"role": "user", "content": "Raw words."}]
    elif provider == "gemini":
        assert "/models/test-model:generateContent" in request.full_url
        assert body["contents"][0]["parts"][0]["text"] == "Raw words."
        assert body["generationConfig"]["maxOutputTokens"] == 1234
        assert "model" not in body
    else:
        assert body["messages"][1]["content"] == "Raw words."
        assert body["stream"] is False
        if provider == "ollama":
            assert body["options"]["num_predict"] == 1234


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/chat",
        "https://user:secret@example.com/chat",
        "https://:secret@example.com/chat",
        "file:///tmp/socket",
        "https://example.com/chat#fragment",
        "https://example.com/chat?key=secret",
        "https://example.com/chat?access_token=secret",
        "https://example.com:bad/chat",
        "https://example.com/\nchat",
    ],
)
def test_bad_endpoint_rejected_before_request(url):
    config = replace(Config(), ai_enabled="on", ai_model="model", ai_endpoint=url)
    with pytest.raises(ValueError):
        ai_providers.request_completion(
            "Raw", config, opener=lambda *a, **kw: pytest.fail("Network access")
        )


def test_loopback_alternatives_and_https_custom_endpoint():
    for url in ("http://127.0.0.2:1234/chat", "http://[::1]:1234/chat", "https://remote/chat"):
        assert ai_providers.endpoint_url(url) == url


def test_redirect_never_forwards_credentials():
    handler = ai_providers.NoRedirect()
    request = urllib.request.Request(
        "https://first/chat", headers={"Authorization": "Bearer secret"}
    )
    with pytest.raises(ValueError, match="redirected"):
        handler.redirect_request(request, None, 307, "redirect", {}, "https://second/chat")


def test_errors_never_echo_response_or_key(monkeypatch):
    config = replace(Config(), ai_enabled="on", ai_model="model")
    monkeypatch.setenv(config.ai_key_env, "secret-token")

    def opener(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://secret-token", 401, "secret-token", {}, io.BytesIO(b"secret-token")
        )

    with pytest.raises(RuntimeError) as error:
        ai_providers.request_completion("private transcript", config, opener=opener)
    assert str(error.value) == "AI cleanup endpoint returned HTTP 401"
    assert error.value.__cause__ is None
    monkeypatch.setenv(config.ai_key_env, "secret\nheader")
    with pytest.raises(ValueError, match="invalid characters") as error:
        ai_providers.request_completion("private", config, opener=opener)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "provider,payload",
    [
        ("anthropic", {"stop_reason": "max_tokens", "content": [{"type": "text", "text": "Cut"}]}),
        (
            "gemini",
            {
                "candidates": [
                    {"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "Cut"}]}}
                ]
            },
        ),
        ("compatible", {"choices": [{"finish_reason": "length", "message": {"content": "Cut"}}]}),
        ("ollama", {"done_reason": "length", "message": {"content": "Cut"}}),
    ],
)
def test_truncated_completions_not_silently_delivered(provider, payload):
    config = replace(Config(), ai_enabled="on", ai_provider=provider, ai_model="model")
    with pytest.raises(ValueError, match="truncated"):
        ai_providers.request_completion(
            "Raw", config, opener=lambda *a, **kw: io.BytesIO(json.dumps(payload).encode())
        )


def test_bounded_response_and_disabled_no_network():
    config = replace(Config(), ai_enabled="on", ai_model="model")
    with pytest.raises(ValueError, match="exceeds 1 MB"):
        ai_providers.request_completion(
            "Raw", config, opener=lambda *a, **kw: io.BytesIO(b"x" * (1024 * 1024 + 1))
        )
    assert (
        ai_providers.request_completion(
            "Raw", Config(), opener=lambda *a, **kw: pytest.fail("Network access")
        )
        == "Raw"
    )


def profile():
    config = replace(Config(), ai_model="example-model")
    values = asdict(config)
    return {
        "id": str(uuid.uuid4()),
        "name": "Example",
        **{k: values[k] for k in profiles.PROFILE_FIELDS},
    }


def test_profiles_roundtrip_and_resolution():
    first = profile()
    first["ai_prompt"] = "Translate into Spanish."
    profiles.save_profiles([first])
    assert profiles.load_profiles() == [first]
    assert profiles.profiles_path().stat().st_mode & 0o777 == 0o600
    selected = replace(Config(), ai_profile=first["id"], ai_enabled="on")
    resolved = profiles.resolve_profile(selected)
    assert resolved.ai_prompt == first["ai_prompt"]
    assert resolved.ai_model == "example-model"
    assert resolved.ai_enabled == "on"
    assert profiles.resolve_profile(Config()) == Config()


@pytest.mark.parametrize(
    "field,value",
    [
        ("api_key", "literal-secret"),
        ("ai_key_env", "sk-literal-secret"),
        ("ai_timeout", True),
        ("ai_provider", "unknown"),
        ("ai_endpoint", "http://remote"),
        ("id", "../../escape"),
    ],
)
def test_profile_validation_precedes_write(field, value):
    original = profile()
    profiles.save_profiles([original])
    invalid = {**original, field: value}
    with pytest.raises(ValueError):
        profiles.save_profiles([invalid])
    assert profiles.load_profiles() == [original]


def test_missing_selected_profile_does_not_fall_back_to_another_provider():
    config = replace(Config(), ai_profile=str(uuid.uuid4()))
    with pytest.raises(ValueError, match="no longer exists"):
        profiles.resolve_profile(config)


def test_local_profile_can_explicitly_omit_key_environment():
    import uuid
    from dataclasses import asdict

    from mouthkeys.config import Config
    from mouthkeys.profiles import PROFILE_FIELDS, resolve_profile, save_profiles

    values = asdict(Config(ai_key_env="", ai_model="local-model"))
    profile = {
        "id": str(uuid.uuid4()),
        "name": "Local without credentials",
        **{key: values[key] for key in PROFILE_FIELDS},
    }
    save_profiles([profile])
    resolved = resolve_profile(Config(ai_profile=profile["id"]))
    assert resolved.ai_key_env == ""
