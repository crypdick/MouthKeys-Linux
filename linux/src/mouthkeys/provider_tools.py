"""Explicit provider verification and same-origin model discovery; no background I/O.

Protocol references:
https://developers.openai.com/api/reference/resources/models/methods/list
https://platform.claude.com/docs/en/api/models/list
https://ai.google.dev/api/models#method:-models.list
https://docs.ollama.com/api/tags
"""

import urllib.parse
from dataclasses import replace

from .ai_providers import _completion, _request_payload, endpoint_url, request_json
from .profiles import resolve_profile


def _origin(url):
    parsed = urllib.parse.urlsplit(endpoint_url(url))
    return (
        parsed.scheme,
        parsed.hostname.lower(),
        parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80),
    )


def models_endpoint(config):
    endpoint = endpoint_url(config.ai_endpoint)
    explicit = getattr(config, "ai_models_endpoint", "")
    if explicit:
        explicit = endpoint_url(explicit)
        if _origin(explicit) != _origin(endpoint):
            raise ValueError("Model discovery endpoint must share completion endpoint origin")
        return explicit
    parsed = urllib.parse.urlsplit(endpoint)
    path = parsed.path
    provider = config.ai_provider
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
    # Keep an explicit deployment/API-version query rather than inventing another provider.
    return urllib.parse.urlunsplit(parsed._replace(path=path))


def _with_query(endpoint, **values):
    parsed = urllib.parse.urlsplit(endpoint)
    query = dict(urllib.parse.parse_qsl(parsed.query))
    query.update(values)
    return urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(query)))


def list_models(config, *, opener=None):
    """List server-visible models on explicit request, at most 1000 IDs/ten pages."""
    config = resolve_profile(config)
    endpoint = models_endpoint(config)
    provider = config.ai_provider
    found = []
    seen_pages = set()
    for _ in range(10):
        if endpoint in seen_pages:
            raise ValueError("Model discovery returned a repeated page cursor")
        seen_pages.add(endpoint)
        data = request_json(endpoint, config, opener=opener)
        if not isinstance(data, dict):
            raise ValueError("Model discovery returned no model list")
        key = "models" if provider in ("gemini", "ollama") else "data"
        items = data.get(key)
        if not isinstance(items, list):
            raise ValueError("Model discovery returned no model list")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Model discovery returned an invalid model")
            if provider == "gemini":
                if "generateContent" not in item.get("supportedGenerationMethods", []):
                    continue
                name = item.get("name", "")
                if isinstance(name, str):
                    name = name.removeprefix("models/")
            elif provider == "ollama":
                name = item.get("name", item.get("model", ""))
            else:
                name = item.get("id", "")
            if (
                not isinstance(name, str)
                or not name
                or len(name) > 500
                or any(not c.isprintable() for c in name)
            ):
                raise ValueError("Model discovery returned an invalid model identifier")
            if name not in found:
                found.append(name)
            if len(found) > 1000:
                raise ValueError("Model discovery exceeds 1000 models")
        if provider == "gemini" and data.get("nextPageToken"):
            cursor = data["nextPageToken"]
            if not isinstance(cursor, str) or len(cursor) > 2048:
                raise ValueError("Model discovery returned an invalid page cursor")
            endpoint = _with_query(models_endpoint(config), pageToken=cursor)
        elif provider == "anthropic" and data.get("has_more"):
            cursor = data.get("last_id")
            if not isinstance(cursor, str) or not cursor or len(cursor) > 500:
                raise ValueError("Model discovery returned an invalid page cursor")
            endpoint = _with_query(models_endpoint(config), after_id=cursor)
        else:
            return found
    raise ValueError("Model discovery exceeds ten pages")


def verify(config, *, opener=None):
    """Explicit tiny completion checks credentials, selected model, and protocol.

    Only a fixed test prompt leaves the process. Provider content is discarded; the
    result never quotes untrusted provider text, diagnostic payloads, or credentials.
    """
    config = resolve_profile(config)
    config = replace(
        config,
        ai_enabled="on",
        ai_prompt="Return only the word OK.",
        ai_max_tokens=32,
        ai_reasoning_parameter="",
        ai_reasoning_value="",
    )
    model = urllib.parse.quote(config.ai_model.removeprefix("models/"), safe="")
    if not model:
        raise ValueError("Choose an AI cleanup model first")
    payload = _request_payload("Connection test", config, config.ai_provider)
    if config.ai_provider == "compatible":
        payload["max_tokens"] = 32
    endpoint = endpoint_url(config.ai_endpoint.replace("{model}", model))
    result = _completion(
        request_json(endpoint, config, body=payload, opener=opener), config.ai_provider
    )
    if not isinstance(result, str) or not result.strip():
        raise ValueError("AI endpoint returned no text completion")
    return {"ok": True, "provider": config.ai_provider, "model": config.ai_model}
