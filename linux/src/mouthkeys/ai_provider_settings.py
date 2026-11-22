"""Pure validation of provider endpoints, routes and reasoning options."""

import ipaddress
import json
import math
import re
import urllib.parse

ReasoningValue = str | int | float | bool | None
PROVIDER_ENDPOINTS = {
    "compatible": "http://localhost:11434/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    "ollama": "http://localhost:11434/api/chat",
}


def _parse_endpoint(value: str) -> urllib.parse.SplitResult:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or any(character.isspace() for character in value)
    ):
        raise ValueError("AI endpoint must be an HTTP(S) URL without embedded credentials")
    try:
        loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = parsed.hostname.lower() == "localhost"
    if parsed.scheme == "http" and not loopback:
        raise ValueError("Remote AI endpoints require HTTPS")
    if parsed.fragment:
        raise ValueError("AI endpoint must not contain a fragment")
    if any(
        name.lower() in {"key", "api_key", "api-key", "apikey", "token", "access_token"}
        for name, _ in urllib.parse.parse_qsl(parsed.query)
    ):
        raise ValueError("Keep AI API keys in an environment variable, not the endpoint URL")
    try:
        _ = parsed.port
    except ValueError:
        raise ValueError("AI endpoint has an invalid port") from None
    return parsed


def endpoint_url(value: str) -> str:
    _parse_endpoint(value)
    return value


def _origin(parsed: urllib.parse.SplitResult) -> tuple[str, str | None, int]:
    port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
    return parsed.scheme, parsed.hostname, port


def models_endpoint(completion: str, provider: str, *, explicit: str = "") -> str:
    parsed = _parse_endpoint(completion)
    if explicit:
        if _origin(_parse_endpoint(explicit)) != _origin(parsed):
            raise ValueError("Model discovery endpoint must share completion endpoint origin")
        return explicit

    path = parsed.path
    if provider == "compatible" and path.endswith("/chat/completions"):
        path = path.removesuffix("/chat/completions") + "/models"
    elif provider == "anthropic" and path.endswith("/messages"):
        path = path.removesuffix("/messages") + "/models"
    elif provider == "ollama" and path.endswith("/api/chat"):
        path = path.removesuffix("/api/chat") + "/api/tags"
    elif provider == "gemini" and "/models/" in path and path.endswith(":generateContent"):
        path = path.rsplit("/models/", 1)[0] + "/models"
    else:
        raise ValueError("Custom completion route requires an explicit model discovery endpoint")
    return urllib.parse.urlunsplit(parsed._replace(path=path))


def reasoning_option(parameter: object, value: object) -> tuple[str, ReasoningValue] | None:
    if not parameter:
        if value:
            raise ValueError("Choose a reasoning parameter before setting its value")
        return None
    nested = {
        "chat_template_kwargs.enable_thinking",
        "generationConfig.thinkingConfig.thinkingBudget",
        "generationConfig.thinkingConfig.thinkingLevel",
        "generationConfig.thinkingConfig.includeThoughts",
        "output_config.effort",
        "options.think",
        "thinking.type",
        "thinking.budget_tokens",
    }
    if not isinstance(parameter, str) or (
        parameter not in nested
        and not re.fullmatch(
            r"(?:reasoning|thinking|think|enable_thinking)(?:_[A-Za-z0-9]+)*", parameter
        )
    ):
        raise ValueError("Reasoning parameter must name a supported thinking/reasoning field")
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError("Reasoning value must contain a JSON scalar or short string")
    try:
        parsed: object = json.loads(value)
    except ValueError:
        parsed = value
    if not isinstance(parsed, str | int | float | bool) and parsed is not None:
        raise ValueError("Reasoning value must be a JSON scalar, never an object or array")
    if isinstance(parsed, float) and not math.isfinite(parsed):
        raise ValueError("Reasoning value must be finite")
    return parameter, parsed
