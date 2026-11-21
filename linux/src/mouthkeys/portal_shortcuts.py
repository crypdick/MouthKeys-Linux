"""Run the isolated GlobalShortcuts helper and dispatch its events."""

import importlib.util
import json
import os
import select
import subprocess
import time
import typing
from collections.abc import Iterator
from contextlib import suppress
from math import isfinite
from pathlib import Path

from .config import Config
from .control import request
from .hotkeys import Activation, bindings
from .portal_helper import MAX_MESSAGE_BYTES, ShortcutEntry
from .processes import python_command
from .x11_shortcuts import ShortcutEvent


def preferred_trigger(mask: int, key: str) -> str:
    if key.startswith("button"):
        raise ValueError(
            "Portal supports keyboard shortcuts only; use X11 or desktop mouse bindings"
        )
    modifiers = [
        name for bit, name in ((4, "CTRL"), (8, "ALT"), (1, "SHIFT"), (64, "LOGO")) if mask & bit
    ]
    return "+".join([*modifiers, key])


def _entries(config: Config) -> list[ShortcutEntry]:
    return [
        {"id": f"{action}_{index}", "action": action, "trigger": preferred_trigger(mask, key)}
        for index, (action, mask, key) in enumerate(bindings(config))
    ]


def _helper_command() -> list[str]:
    system_python = Path("/usr/bin/python3")
    if importlib.util.find_spec("gi") is None and system_python.is_file():
        helper = Path(__file__).resolve().with_name("portal_helper.py")
        return [str(system_python), "-I", str(helper)]
    return python_command("mouthkeys.portal_helper")


def _frames(fd: int) -> Iterator[list[bytes]]:
    buffered = b""
    while True:
        if not select.select([fd], [], [], 0.25)[0]:
            yield []
            continue
        chunk = os.read(fd, 4096)
        if not chunk:
            if buffered:
                raise RuntimeError("Incomplete shortcut event")
            return
        *lines, buffered = (buffered + chunk).split(b"\n")
        if len(buffered) > MAX_MESSAGE_BYTES or any(
            len(line) > MAX_MESSAGE_BYTES for line in lines
        ):
            raise RuntimeError("Oversized shortcut event")
        yield lines


def _event(line: bytes, actions: dict[str, str]) -> ShortcutEvent:
    data = json.loads(line)
    if not isinstance(data, dict):
        raise TypeError("Invalid portal shortcut event")
    match data:
        case {"error": str(error)}:
            raise RuntimeError(error)
    identity, action, down, timestamp = (
        data.get("id"),
        data.get("action"),
        data.get("down"),
        data.get("at"),
    )
    if (
        not isinstance(identity, str)
        or not isinstance(action, str)
        or actions.get(identity) != action
        or not isinstance(down, bool)
        or isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
    ):
        raise ValueError("Invalid portal shortcut event")
    try:
        at = float(timestamp)
    except OverflowError:
        raise ValueError("Invalid portal shortcut timestamp") from None
    if not isfinite(at):
        raise ValueError("Invalid portal shortcut timestamp")
    return action, down, at, identity


def _monitor(
    child: subprocess.Popen[bytes],
    config: Config,
    activation: Activation,
    entries: list[ShortcutEntry],
) -> Config | None:
    # The child was created with both streams set to PIPE.
    stdin = typing.cast("typing.BinaryIO", child.stdin)
    stdout = typing.cast("typing.BinaryIO", child.stdout)
    payload = (json.dumps(entries) + "\n").encode()
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("Oversized portal shortcut configuration")
    stdin.write(payload)
    stdin.close()
    actions = {entry["id"]: entry["action"] for entry in entries}
    checked_at = 0.0
    for batch in _frames(stdout.fileno()):
        for line in batch:
            activation.event(*_event(line, actions))
        now = time.monotonic()
        if now - checked_at >= 1:
            checked_at = now
            refreshed = Config.load()
            if refreshed.shortcut_backend != "portal" or (
                not activation.pressed
                and (refreshed.hotkey_mode != config.hotkey_mode or _entries(refreshed) != entries)
            ):
                return refreshed
    if child.wait(timeout=2):
        raise RuntimeError("GlobalShortcuts portal unavailable, cancelled, or denied")
    return None


def _stop(child: subprocess.Popen[bytes]) -> None:
    # The child was created with both streams set to PIPE.
    stdin = typing.cast("typing.BinaryIO", child.stdin)
    stdout = typing.cast("typing.BinaryIO", child.stdout)
    try:
        with stdout, suppress(BrokenPipeError):
            stdin.close()
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2)


def launch() -> int:
    config = Config.load()
    activation = Activation(config.hotkey_mode, request)
    while config.shortcut_backend == "portal":
        entries = _entries(config)
        child = subprocess.Popen(  # noqa: S603 - fixed packaged helper command
            _helper_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        try:
            updated = _monitor(child, config, activation, entries)
        finally:
            try:
                activation.finish()
            finally:
                _stop(child)
        if updated is None:
            break
        config = updated
        activation.mode = config.hotkey_mode
    return 0
