"""Validate and bound Command/Edit data without opening a history store."""

import json
import math
import uuid
from typing import Any, cast

MAX_MESSAGES = 200
MAX_SESSION_BYTES = 1024 * 1024
MAX_EXECUTIONS = 100
MAX_EXECUTION_BYTES = 512 * 1024


def validate_execution(result: object) -> dict[str, Any]:
    fields = {"command", "working_directory", "returncode", "output", "truncated", "timed_out"}
    if not isinstance(result, dict) or not fields <= set(result) <= fields | {"cancelled"}:
        raise ValueError("Invalid command execution receipt")
    for name, limit in (("command", 16384), ("working_directory", 4096), ("output", 262144)):
        if not isinstance(result[name], str) or len(result[name]) > limit:
            raise ValueError("Invalid command execution receipt text")
    if (
        not result["command"].strip()
        or "\0" in result["command"]
        or "\0" in result["working_directory"]
        or (result["working_directory"] and not result["working_directory"].startswith("/"))
        or type(result["returncode"]) is not int
        or not -(2**31) <= result["returncode"] < 2**31
        or any(type(result[key]) is not bool for key in ("truncated", "timed_out"))
        or ("cancelled" in result and type(result["cancelled"]) is not bool)
    ):
        raise ValueError("Invalid command execution receipt values")
    try:
        size = len(json.dumps(result, ensure_ascii=False).encode())
    except UnicodeEncodeError:
        raise ValueError("Invalid command execution receipt Unicode") from None
    if size > MAX_EXECUTION_BYTES - 128:
        raise ValueError("Command execution receipt exceeds 512 KiB")
    return dict(result)


def validate_executions(events: object, message_count: int) -> list[dict[str, Any]]:
    if not isinstance(events, list) or len(events) > MAX_EXECUTIONS:
        raise ValueError("Invalid command execution events")
    for event in events:
        if (
            not isinstance(event, dict)
            or set(event) != {"message_count", "result"}
            or type(event["message_count"]) is not int
            or not 0 <= event["message_count"] <= message_count
            or event["message_count"] % 2
        ):
            raise ValueError("Invalid command execution event position")
        validate_execution(event["result"])
    if len(json.dumps(events, ensure_ascii=False).encode()) > MAX_EXECUTION_BYTES:
        raise ValueError("Command execution events exceed 512 KiB")
    return events


def pending_executions(session: dict[str, Any]) -> list[dict[str, Any]]:
    messages = validate_messages(session["messages"])
    events = validate_executions(session.get("executions", []), len(messages))
    return [dict(event["result"]) for event in events if event["message_count"] >= len(messages)]


def retain_execution(
    events: list[dict[str, Any]], receipt: dict[str, Any], message_count: int
) -> list[dict[str, Any]]:
    events = [*events, {"message_count": message_count, "result": receipt}]
    while (
        len(events) > MAX_EXECUTIONS
        or len(json.dumps(events, ensure_ascii=False).encode()) > MAX_EXECUTION_BYTES
    ):
        events.pop(0)
    return events


def record_private_execution(session: dict[str, Any], result: object) -> dict[str, Any]:
    """Record an observed outcome in bounded runtime memory, without opening SQLite."""
    count = len(validate_messages(session["messages"]))
    events = validate_executions(session.get("executions", []), count)
    session["executions"] = retain_execution(events, validate_execution(result), count)
    return session


def recent_messages(
    previous: list[dict[str, str]], incoming: list[dict[str, str]]
) -> tuple[list[dict[str, str]], int]:
    combined = [*previous, *incoming]
    recent = validate_messages(combined[-MAX_MESSAGES:])
    return recent, len(combined) - len(recent)


def append_private_messages(session: dict[str, Any], messages: object) -> dict[str, Any]:
    previous = validate_messages(session["messages"])
    events = validate_executions(session.get("executions", []), len(previous))
    recent, removed = recent_messages(previous, validate_messages(messages))
    session["messages"] = recent
    session["executions"] = [
        {"message_count": max(0, event["message_count"] - removed), "result": dict(event["result"])}
        for event in events
    ]
    return session


def validate_messages(messages: object) -> list[dict[str, str]]:
    if not isinstance(messages, list) or len(messages) > MAX_MESSAGES:
        raise ValueError("Invalid assistant messages")
    if len(messages) % 2:
        raise ValueError("Assistant history must contain completed turns")
    for index, message in enumerate(messages):
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in ("user", "assistant")
            or message["role"] != ("user" if index % 2 == 0 else "assistant")
            or not isinstance(message["content"], str)
        ):
            raise ValueError("Invalid assistant message")
    try:
        size = len(json.dumps(messages, ensure_ascii=False).encode())
    except UnicodeEncodeError:
        raise ValueError("Assistant messages must contain valid Unicode") from None
    if size > MAX_SESSION_BYTES:
        raise ValueError("Assistant session exceeds 1 MiB; start a new chat")
    return messages


def context_messages(messages: object) -> list[dict[str, str]]:
    """Keep complete recent turns within the provider context budget."""
    recent = validate_messages(messages)[-32:]
    remaining = 240 * 1024
    start = len(recent)
    for index in range(len(recent) - 2, -1, -2):
        size = sum(
            len(json.dumps(message, ensure_ascii=False).encode())
            for message in recent[index : index + 2]
        )
        if size > remaining:
            break
        remaining -= size
        start = index
    return [dict(message) for message in recent[start:]]


def assistant_mode(mode: object) -> None:
    if mode not in ("command", "edit"):
        raise ValueError("Invalid assistant mode")


def _session_timestamp(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("Invalid assistant session metadata")
    try:
        updated = float(cast("int | float", value))
    except OverflowError:
        raise ValueError("Invalid assistant session metadata") from None
    if not math.isfinite(updated):
        raise ValueError("Invalid assistant session metadata")
    return updated


def validated_session(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) - {"executions"} != {
        "id",
        "mode",
        "title",
        "updated",
        "messages",
    }:
        raise ValueError("Invalid assistant session")
    item_id, title = raw["id"], raw["title"]
    if not isinstance(item_id, str) or not isinstance(title, str) or len(title) > 100:
        raise ValueError("Invalid assistant session metadata")
    uuid.UUID(item_id)
    try:
        title.encode()
    except UnicodeEncodeError:
        raise ValueError("Invalid assistant session metadata") from None
    mode = raw["mode"]
    assistant_mode(mode)
    updated = _session_timestamp(raw["updated"])
    messages = validate_messages(raw["messages"])
    events = validate_executions(raw.get("executions", []), len(messages))
    if mode != "command" and events:
        raise ValueError("Only command sessions may contain execution events")
    return {
        "id": item_id,
        "mode": mode,
        "title": title,
        "updated": updated,
        "messages": messages,
        "executions": events,
    }
