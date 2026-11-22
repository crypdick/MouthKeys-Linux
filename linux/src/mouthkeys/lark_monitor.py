"""Opt-in receiver discovery and generation-scoped battery polling for the overlay."""

import json
import shlex
import subprocess
import threading
import time
from contextlib import suppress
from typing import Any, cast

from .config import Config
from .lark import PRODUCT, VENDOR, BatteryStatus, MicBattery, UnavailableBattery, UnavailableReason
from .processes import python_command

POLL_SECONDS = 15
MAX_SOURCE_CHAIN = 16


def _run_pactl(*args: str) -> str:
    return subprocess.run(  # noqa: S603
        ["pactl", *args],  # noqa: S607 - fixed PulseAudio client on PATH
        capture_output=True,
        text=True,
        check=True,
        timeout=1,
    ).stdout


def _pactl_records(kind: str) -> list[dict[str, Any]]:
    records = json.loads(_run_pactl("-f", "json", "list", kind))
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError("Invalid PulseAudio source information")
    return records


def _usb_id(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        with suppress(ValueError):
            return int(value, 16)
    return None


def _module_masters(modules: list[dict[str, Any]], owner: object) -> list[str]:
    for module in modules:
        if str(module.get("index")) == str(owner) and module.get("name") == "module-remap-source":
            arguments = module.get("argument", "")
            if isinstance(arguments, str):
                return [
                    word.removeprefix("master=")
                    for word in shlex.split(arguments)
                    if word.startswith("master=")
                ]
    return []


def input_is_receiver(source: str) -> bool:
    if source == "default":
        source = _run_pactl("get-default-source").strip()
    sources = {
        item.get("name"): item
        for item in _pactl_records("sources")
        if isinstance(item.get("name"), str)
    }
    pending = [source]
    visited: set[str] = set()
    modules: list[dict[str, Any]] | None = None
    while pending and len(visited) < MAX_SOURCE_CHAIN:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        item = sources.get(name)
        if item is None:
            continue
        props = item.get("properties")
        if isinstance(props, dict):
            if (
                _usb_id(props.get("device.vendor.id")) == VENDOR
                and _usb_id(props.get("device.product.id")) == PRODUCT
            ):
                return True
            master = props.get("device.master_device")
            if isinstance(master, str) and master:
                pending.append(master)
        if (owner := item.get("owner_module")) is not None:
            if modules is None:
                modules = _pactl_records("modules")
            pending.extend(_module_masters(modules, owner))
    return False


def _parse_mic(value: object) -> MicBattery:
    if not isinstance(value, dict) or type(value.get("linked")) is not bool:
        raise ValueError("Invalid receiver battery status")
    linked, percent = value["linked"], value.get("percent")
    if percent is not None and (type(percent) is not int or not linked or not 0 <= percent <= 100):
        raise ValueError("Invalid receiver battery status")
    return {"linked": linked, "percent": percent}


def _helper_status(value: object) -> BatteryStatus:
    if not isinstance(value, dict):
        raise ValueError("Invalid receiver battery status")  # noqa: TRY004 - helper protocol error
    result = value.get("result")
    if result == "ok":
        return {
            "result": "ok",
            "mic1": _parse_mic(value.get("mic1")),
            "mic2": _parse_mic(value.get("mic2")),
        }
    if result not in ("no-device", "permission-denied", "unavailable"):
        raise ValueError("Invalid receiver battery status")
    status: UnavailableBattery = {"result": cast("UnavailableReason", result)}
    if isinstance(error := value.get("error"), str):
        status["error"] = error
    return status


def microphone_labels(source: str, battery: BatteryStatus | None) -> tuple[str, str]:
    if battery and battery["result"] == "ok":
        values = [
            f"{percent}%"
            for mic in (battery["mic1"], battery["mic2"])
            if (percent := mic["percent"]) is not None
        ]
        text = source + (" " + "/".join(values) if values else "")
        return text, "Lark A1 battery · " + (" / ".join(values) or "No linked mic")
    return source, "Choose microphone · " + (battery.get("result", "") if battery else "")


class BatteryMonitor:
    def __init__(self) -> None:
        self.value: BatteryStatus | None = None
        self.polled_at = float("-inf")
        self.worker: threading.Thread | None = None
        self.source: str | None = None
        self._lock = threading.Lock()
        self._generation = object()

    def poll(self, config: Config, source: str) -> BatteryStatus | None:
        selected = source if config.mic_battery == "on" else None
        with self._lock:
            if selected != self.source:
                self.value = None
                self.polled_at = float("-inf")
                self.source = selected
                self._generation = object()
            if selected is None:
                return None
            now = time.monotonic()
            if now - self.polled_at >= POLL_SECONDS and (
                self.worker is None or not self.worker.is_alive()
            ):
                self.polled_at = now
                self.worker = threading.Thread(
                    target=self._poll, args=(selected, self._generation), daemon=True
                )
                self.worker.start()
            return self.value

    def _poll(self, source: str, generation: object) -> None:
        try:
            value: BatteryStatus = {"result": "not-receiver"}
            if input_is_receiver(source):
                with self._lock:
                    if self._generation is not generation:
                        return
                result = subprocess.run(  # noqa: S603
                    python_command("mouthkeys.lark"),
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=2,
                )
                value = _helper_status(json.loads(result.stdout))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            value = {"result": "unavailable", "error": str(exc)}
        with self._lock:
            if self._generation is generation:
                self.value = value
