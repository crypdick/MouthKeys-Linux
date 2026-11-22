"""Explicit provider verification and same-origin model discovery; no background I/O.

Protocol references:
https://developers.openai.com/api/reference/resources/models/methods/list
https://platform.claude.com/docs/en/api/models/list
https://ai.google.dev/api/models#method:-models.list
https://docs.ollama.com/api/tags
"""

import urllib.parse
from collections.abc import Iterator
from dataclasses import replace

from .ai_provider_settings import endpoint_url, models_endpoint
from .ai_providers import RequestOpener, _completion, _request_payload, request_json
from .config import Config
from .profiles import resolve_profile


def _model_identifier(item: dict[str, object], provider: str) -> str:
    if provider == "gemini":
        name = item.get("name")
        if isinstance(name, str):
            name = name.removeprefix("models/")
    elif provider == "ollama":
        name = item.get("name", item.get("model"))
    else:
        name = item.get("id")
    if not isinstance(name, str) or not name or len(name) > 500 or not name.isprintable():
        raise ValueError("Model discovery returned an invalid model identifier")
    return name


def _page_models(page: dict[str, object], provider: str) -> Iterator[str]:
    items = page.get("models" if provider in ("gemini", "ollama") else "data")
    if not isinstance(items, list):
        raise ValueError("Model discovery returned no model list")  # noqa: TRY004 - provider payload
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Model discovery returned an invalid model")  # noqa: TRY004 - provider payload
        if provider == "gemini":
            methods = item.get("supportedGenerationMethods", [])
            if not isinstance(methods, list):
                raise ValueError("Model discovery returned invalid model capabilities")
            if "generateContent" not in methods:
                continue
        yield _model_identifier(item, provider)


def _next_cursor(page: dict[str, object], provider: str) -> tuple[str, str] | None:
    if provider == "gemini":
        cursor = page.get("nextPageToken")
        if cursor is None or cursor == "":
            return None
        if not isinstance(cursor, str) or len(cursor) > 2048:
            raise ValueError("Model discovery returned an invalid page cursor")
        return "pageToken", cursor
    if provider == "anthropic" and page.get("has_more"):
        cursor = page.get("last_id")
        if not isinstance(cursor, str) or not cursor or len(cursor) > 500:
            raise ValueError("Model discovery returned an invalid page cursor")
        return "after_id", cursor
    return None


def _page_url(endpoint: str, key: str, cursor: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    query = [
        (name, value)
        for name, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if name != key
    ]
    query.append((key, cursor))
    return urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(query)))


def list_models(config: Config, *, opener: RequestOpener | None = None) -> list[str]:
    """List server-visible models on explicit request, at most 1000 IDs/ten pages."""
    config = resolve_profile(config)
    provider = config.ai_provider
    base = models_endpoint(config.ai_endpoint, provider, explicit=config.ai_models_endpoint)
    endpoint = base
    found: dict[str, None] = {}
    seen: set[str] = set()
    for _ in range(10):
        if endpoint in seen:
            raise ValueError("Model discovery returned a repeated page cursor")
        seen.add(endpoint)
        page = request_json(endpoint, config, opener=opener)
        if not isinstance(page, dict):
            raise ValueError("Model discovery returned no model list")  # noqa: TRY004 - provider payload
        for name in _page_models(page, provider):
            found[name] = None
            if len(found) > 1000:
                raise ValueError("Model discovery exceeds 1000 models")
        cursor = _next_cursor(page, provider)
        if cursor is None:
            return list(found)
        endpoint = _page_url(base, *cursor)
    raise ValueError("Model discovery exceeds ten pages")


def verify(config: Config, *, opener: RequestOpener | None = None) -> dict[str, str | bool]:
    """Explicit tiny completion checks credentials, selected model, and protocol.

    Only a fixed test prompt leaves the process. Provider content is discarded; the
    result never quotes untrusted provider text, diagnostic payloads, or credentials.
    """
    config = replace(
        resolve_profile(config),
        ai_enabled="on",
        ai_prompt="Return only the word OK.",
        ai_max_tokens=32,
        ai_reasoning_parameter="",
        ai_reasoning_value="",
    )
    model = urllib.parse.quote(config.ai_model.removeprefix("models/"), safe="")
    if not model:
        raise ValueError("Choose an AI cleanup model first")
    provider = config.ai_provider
    payload = _request_payload("Connection test", config, provider)
    if provider == "compatible":
        payload["max_tokens"] = 32
    endpoint = endpoint_url(config.ai_endpoint.replace("{model}", model))
    data = request_json(endpoint, config, body=payload, opener=opener)
    try:
        result = _completion(data, provider)
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ValueError("AI endpoint returned no text completion") from None
    if not isinstance(result, str) or not result.strip():
        raise ValueError("AI endpoint returned no text completion")
    return {"ok": True, "provider": provider, "model": config.ai_model}
