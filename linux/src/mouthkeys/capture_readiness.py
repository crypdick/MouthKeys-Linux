"""Decide whether a Bluetooth loopback has a running physical input."""

import json
import subprocess
from typing import Any


def _run(*args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed audio inspection commands
        list(args), capture_output=True, text=True, check=True, timeout=0.5
    ).stdout


def _rows(raw: str) -> list[dict[str, Any]]:
    data: object = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Expected an array of audio objects")  # noqa: TRY004 - malformed snapshot
    return [item for item in data if isinstance(item, dict)]


def _properties(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Invalid audio properties")  # noqa: TRY004 - malformed snapshot
    return value


def _enabled(value: object) -> bool:
    return value is True or value == "true"


def _loopback_ready(source: str) -> bool:
    graph = _rows(_run("pw-dump", "Node"))
    nodes = [
        (info.get("state"), _properties(info.get("props", {})))
        for row in graph
        if isinstance(info := row.get("info"), dict)
    ]
    selected = next((props for _, props in nodes if props.get("node.name") == source), None)
    if selected is None:
        return False
    if not _enabled(selected.get("bluez5.loopback")):
        return True
    device = selected.get("device.id")
    return device is not None and any(
        str(props.get("device.id")) == str(device)
        and _enabled(props.get("bluez5.loopback-target"))
        and state == "running"
        for state, props in nodes
    )


def bluetooth_capture_ready(source: str) -> bool:
    """PipeWire's Bluetooth loopback can run before its real microphone exists."""
    if source == "default":
        try:
            source = _run("pactl", "get-default-source").strip()
        except (OSError, subprocess.SubprocessError):
            return True  # No source discovery: retain normal PulseAudio readiness.
    if not source.startswith("bluez_input."):
        return True
    try:
        sources = _rows(_run("pactl", "-f", "json", "list", "sources"))
        selected = next((row for row in sources if row.get("name") == source), None)
        if selected is None:
            return False
        props = _properties(selected.get("properties", {}))
        # Native PulseAudio capture does not need a PipeWire graph.
        return not _enabled(props.get("bluez5.loopback")) or _loopback_ready(source)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError("Cannot verify Bluetooth microphone readiness with pw-dump") from exc
