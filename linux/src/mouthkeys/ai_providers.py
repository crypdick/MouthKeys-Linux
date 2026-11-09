"""Small native AI cleanup adapters; requests occur only after explicit opt-in."""

import http.client
import ipaddress
import json
import os
import urllib.error
import urllib.parse
import urllib.request

MAX_RESPONSE = 1024 * 1024
PROVIDER_ENDPOINTS = {
    "compatible": "http://localhost:11434/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    "ollama": "http://localhost:11434/api/chat",
}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("AI endpoint redirected; configure its final URL explicitly")


def endpoint_url(value):
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
    return value


def _request_payload(text, config, provider):
    messages = [
        {"role": "system", "content": config.ai_prompt},
        {"role": "user", "content": text},
    ]
    limit = getattr(config, "ai_max_tokens", 4096)
    if provider == "anthropic":
        return {
            "model": config.ai_model,
            "system": config.ai_prompt,
            "messages": [messages[1]],
            "max_tokens": limit,
            "stream": False,
        }
    if provider == "gemini":
        return {
            "systemInstruction": {"parts": [{"text": config.ai_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"maxOutputTokens": limit},
        }
    body = {"model": config.ai_model, "messages": messages, "stream": False}
    if provider == "ollama":
        body["options"] = {"num_predict": limit}
    # Compatible providers differ on output-token parameter names. Leave it out
    # rather than sending a parameter that breaks otherwise compatible endpoints.
    return body


def _completion(data, provider):
    if provider == "anthropic":
        if data.get("stop_reason") == "max_tokens":
            raise ValueError("AI completion truncated; increase AI maximum output tokens")
        return "".join(block["text"] for block in data["content"] if block.get("type") == "text")
    if provider == "gemini":
        candidate = data["candidates"][0]
        if candidate.get("finishReason") == "MAX_TOKENS":
            raise ValueError("AI completion truncated; increase AI maximum output tokens")
        return "".join(
            part["text"]
            for part in candidate["content"]["parts"]
            if "text" in part and not part.get("thought")
        )
    if provider == "ollama":
        if data.get("done_reason") == "length":
            raise ValueError("AI completion truncated; increase AI maximum output tokens")
        return data["message"]["content"]
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("AI completion truncated; increase provider output token limit")
    return choice["message"]["content"]


def request_completion(text, config, *, opener=None):
    """Call selected protocol; never echo provider responses or secrets in errors."""
    if config.ai_enabled != "on":
        return text
    if not config.ai_model.strip():
        raise ValueError("Choose an AI cleanup model first")
    provider = getattr(config, "ai_provider", "compatible")
    if provider not in PROVIDER_ENDPOINTS:
        raise ValueError("Unknown AI provider protocol")
    model = urllib.parse.quote(config.ai_model.removeprefix("models/"), safe="")
    endpoint = endpoint_url(config.ai_endpoint.replace("{model}", model))
    body = json.dumps(_request_payload(text, config, provider)).encode()
    if len(body) > MAX_RESPONSE:
        raise ValueError("AI cleanup request exceeds 1 MB")
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(config.ai_key_env, "") if config.ai_key_env else ""
    if len(key) > 4096 or any(not 33 <= ord(character) <= 126 for character in key):
        raise ValueError("AI API key contains invalid characters")
    if provider == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
        if key:
            headers["x-api-key"] = key
    elif provider == "gemini":
        if key:
            headers["x-goog-api-key"] = key
    elif key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    open_request = opener or urllib.request.build_opener(NoRedirect()).open
    try:
        with open_request(request, timeout=getattr(config, "ai_timeout", 30)) as response:
            raw = response.read(MAX_RESPONSE + 1)
    except urllib.error.HTTPError as exc:
        # Error bodies can echo keys, prompts, or transcripts. Never include them.
        raise RuntimeError(f"AI cleanup endpoint returned HTTP {exc.code}") from None
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        raise RuntimeError("AI cleanup endpoint unavailable") from None
    if len(raw) > MAX_RESPONSE:
        raise ValueError("AI cleanup response exceeds 1 MB")
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        raise ValueError("AI endpoint returned invalid JSON") from None
    try:
        result = _completion(data, provider)
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ValueError("AI endpoint returned no text completion") from None
    if not isinstance(result, str) or not result.strip():
        raise ValueError("AI endpoint returned an empty completion")
    return result.strip()
