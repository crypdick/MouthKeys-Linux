"""Small native AI cleanup adapters; requests occur only after explicit opt-in."""

import http.client
import ipaddress
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request

MAX_RESPONSE = 1024 * 1024
MAX_HISTORY_MESSAGES = 32
MAX_HISTORY_BYTES = 256 * 1024
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


def validate_reasoning(config):
    parameter = getattr(config, "ai_reasoning_parameter", "")
    value = getattr(config, "ai_reasoning_value", "")
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
        parsed = json.loads(value)
    except ValueError:
        parsed = value
    if not isinstance(parsed, str | int | float | bool) and parsed is not None:
        raise ValueError("Reasoning value must be a JSON scalar, never an object or array")
    if isinstance(parsed, float) and not math.isfinite(parsed):
        raise ValueError("Reasoning value must be finite")
    return parameter, parsed


def _apply_reasoning(body, config):
    option = validate_reasoning(config)
    if option:
        path, value = option
        destination = body
        parts = path.split(".")
        for part in parts[:-1]:
            destination = destination.setdefault(part, {})
            if not isinstance(destination, dict):
                raise ValueError("Reasoning parameter conflicts with request structure")
        destination[parts[-1]] = value
    return body


def request_headers(config, provider):
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
    return headers


def request_json(endpoint, config, *, body=None, opener=None):
    provider = getattr(config, "ai_provider", "compatible")
    payload = json.dumps(body).encode() if body is not None else None
    if payload is not None and len(payload) > MAX_RESPONSE:
        raise ValueError("AI cleanup request exceeds 1 MB")
    request = urllib.request.Request(
        endpoint_url(endpoint),
        data=payload,
        headers=request_headers(config, provider),
        method="POST" if body is not None else "GET",
    )
    open_request = opener or urllib.request.build_opener(NoRedirect()).open
    try:
        with open_request(request, timeout=getattr(config, "ai_timeout", 30)) as response:
            raw = response.read(MAX_RESPONSE + 1)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"AI cleanup endpoint returned HTTP {exc.code}") from None
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        raise RuntimeError("AI cleanup endpoint unavailable") from None
    if len(raw) > MAX_RESPONSE:
        raise ValueError("AI cleanup response exceeds 1 MB")
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        raise ValueError("AI endpoint returned invalid JSON") from None


def validate_history(history):
    """Copy bounded completed text-only turns, never tools or provider reasoning."""
    if history is None:
        return []
    if not isinstance(history, list | tuple) or len(history) > MAX_HISTORY_MESSAGES:
        raise ValueError("AI history must contain at most 32 messages")
    messages, size = [], 0
    for index, item in enumerate(history):
        if (
            not isinstance(item, dict)
            or set(item) != {"role", "content"}
            or item["role"] != ("user" if index % 2 == 0 else "assistant")
            or not isinstance(item["content"], str)
            or not item["content"].strip()
        ):
            raise ValueError("AI history must contain completed user/assistant text pairs only")
        if len(item["content"]) > MAX_HISTORY_BYTES:
            raise ValueError("AI history exceeds 256 KiB")
        try:
            size += len(item["content"].encode("utf-8"))
        except UnicodeEncodeError:
            raise ValueError("AI history must contain valid Unicode text") from None
        if size > MAX_HISTORY_BYTES:
            raise ValueError("AI history exceeds 256 KiB")
        messages.append({"role": item["role"], "content": item["content"]})
    if len(messages) % 2:
        raise ValueError("AI history must contain completed user/assistant text pairs only")
    return messages


def _base_payload(text, config, provider, history=None):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("AI input must be nonempty text")
    conversation = [*validate_history(history), {"role": "user", "content": text}]
    messages = [{"role": "system", "content": config.ai_prompt}, *conversation]
    limit = getattr(config, "ai_max_tokens", 4096)
    if provider == "anthropic":
        return {
            "model": config.ai_model,
            "system": config.ai_prompt,
            "messages": conversation,
            "max_tokens": limit,
            "stream": False,
        }
    if provider == "gemini":
        return {
            "systemInstruction": {"parts": [{"text": config.ai_prompt}]},
            "contents": [
                {
                    "role": "model" if item["role"] == "assistant" else "user",
                    "parts": [{"text": item["content"]}],
                }
                for item in conversation
            ],
            "generationConfig": {"maxOutputTokens": limit},
        }
    body = {"model": config.ai_model, "messages": messages, "stream": False}
    if provider == "ollama":
        body["options"] = {"num_predict": limit}
    # Compatible providers differ on output-token parameter names. Leave it out
    # rather than sending a parameter that breaks otherwise compatible endpoints.
    return body


def _request_payload(text, config, provider, history=None):
    return _apply_reasoning(_base_payload(text, config, provider, history), config)


def thinking_text(data, provider, *, streaming=False):
    """Only explicit external-provider reasoning fields; never infer from text."""
    if provider == "anthropic":
        if streaming:
            if data.get("type") == "content_block_start":
                block = data.get("content_block", {})
                values = [block.get("thinking", "")] if block.get("type") == "thinking" else []
            else:
                delta = data.get("delta", {})
                values = (
                    [delta.get("thinking", "")] if delta.get("type") == "thinking_delta" else []
                )
        else:
            values = [
                block.get("thinking", "")
                for block in data.get("content", [])
                if block.get("type") == "thinking"
            ]
    elif provider == "gemini":
        candidates = data.get("candidates", [])
        values = (
            [
                part.get("text", "")
                for part in candidates[0].get("content", {}).get("parts", [])
                if part.get("thought") is True
            ]
            if candidates
            else []
        )
    elif provider == "ollama":
        values = [data.get("message", {}).get("thinking", "")]
    else:
        choices = data.get("choices", [])
        message = choices[0].get("delta" if streaming else "message", {}) if choices else {}
        # OpenAI-compatible services (e.g. local vLLM) explicitly tag these
        # fields. Ordinary content and <think> strings are never parsed here.
        values = [message.get("reasoning_content", message.get("reasoning", "")) or ""]
    if any(not isinstance(value, str) for value in values):
        raise ValueError("AI endpoint returned invalid thinking content")
    return "".join(values)


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


def request_completion(
    text, config, *, opener=None, on_delta=None, cancel_event=None, history=None, on_thinking=None
):
    """Return final text; optional history is completed text-only conversation pairs.

    on_thinking receives only explicitly tagged external-provider text, separately
    from the completion. It is never appended to history or parsed from ordinary
    output. The caller decides whether to display this opt-in channel.
    """
    if config.ai_enabled != "on":
        return text
    if cancel_event is not None and cancel_event.is_set():
        from .ai_stream import CompletionCancelled

        raise CompletionCancelled("AI cleanup cancelled")
    if not config.ai_model.strip():
        raise ValueError("Choose an AI cleanup model first")
    provider = getattr(config, "ai_provider", "compatible")
    if provider not in PROVIDER_ENDPOINTS:
        raise ValueError("Unknown AI provider protocol")
    model = urllib.parse.quote(config.ai_model.removeprefix("models/"), safe="")
    endpoint = endpoint_url(config.ai_endpoint.replace("{model}", model))
    if getattr(config, "ai_streaming", "off") == "on":
        from .ai_stream import stream_completion

        return stream_completion(
            endpoint,
            _request_payload(text, config, provider, history),
            config,
            opener=opener,
            on_delta=on_delta,
            cancel_event=cancel_event,
            on_thinking=on_thinking,
        )
    data = request_json(
        endpoint, config, body=_request_payload(text, config, provider, history), opener=opener
    )
    if cancel_event is not None and cancel_event.is_set():
        from .ai_stream import CompletionCancelled

        raise CompletionCancelled("AI cleanup cancelled")
    try:
        result = _completion(data, provider)
        thinking = thinking_text(data, provider) if on_thinking is not None else ""
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ValueError("AI endpoint returned no text completion") from None
    if not isinstance(result, str) or not result.strip():
        raise ValueError("AI endpoint returned an empty completion")
    if thinking:
        on_thinking(thinking)
    return result.strip()
