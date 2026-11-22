"""Optional native shortcuts. Desktop/i3 ownership remains the default.

X11 uses passive grabs, never polling every key. Portal uses system-managed
GlobalShortcuts; running that backend explicitly may open its consent dialog.
"""

import fcntl
import logging
import os
import pathlib
import queue
import re
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from types import FrameType
from typing import Protocol, runtime_checkable

from . import profiles
from .config import Config, runtime_dir, state_dir
from .control import ensure_daemon, request
from .processes import python_command
from .x11_shortcuts import ShortcutBinding, ShortcutEvent, X11Grabs

logger = logging.getLogger(__name__)

MODIFIERS = {"shift": 1, "ctrl": 4, "control": 4, "alt": 8, "super": 64, "meta": 64, "win": 64}


def parse_chord(value: str) -> tuple[int, str]:
    if value.endswith("++"):
        value = value[:-1] + "plus"
    parts = [part.strip().lower() for part in value.split("+")]
    if not parts[-1] or any(part not in MODIFIERS for part in parts[:-1]):
        raise ValueError(f"Invalid shortcut: {value}")
    key = parts[-1]
    if not re.fullmatch(r"[a-z0-9_-]+", key):
        raise ValueError(f"Invalid shortcut key: {key}")
    mask = 0
    for part in parts[:-1]:
        mask |= MODIFIERS[part]
    if not mask and (len(key) == 1 or key in ("button1", "button2", "button3")):
        raise ValueError("Ordinary typing/click shortcuts require a modifier")
    return mask, {
        "escape": "Escape",
        "esc": "Escape",
        "enter": "Return",
        "return": "Return",
        "tab": "Tab",
        "backspace": "BackSpace",
        "delete": "Delete",
        "del": "Delete",
        "insert": "Insert",
        "ins": "Insert",
        "pgup": "Prior",
        "page_up": "Prior",
        "pgdown": "Next",
        "page_down": "Next",
        "left": "Left",
        "right": "Right",
        "up": "Up",
        "down": "Down",
        "home": "Home",
        "end": "End",
        "capslock": "Caps_Lock",
        "numlock": "Num_Lock",
    }.get(key, key.upper() if re.fullmatch(r"f\d+", key) else key)


def bindings(config: Config) -> list[ShortcutBinding]:
    configured = [
        ("toggle", config.toggle_shortcut),
        ("cancel", config.cancel_shortcut),
        ("paste", config.paste_shortcut_key),
        ("command", config.command_shortcut_key),
        ("edit", config.edit_shortcut_key),
        *(
            (f"profile:{profile['id']}", profile["shortcut"])
            for profile in profiles.load_profiles()
        ),
    ]
    result = [
        (action, *parse_chord(chord))
        for action, value in configured
        for chord in value.split(",")
        if chord.strip()
    ]
    keys = {(mask, key) for _, mask, key in result}
    if len(keys) != len(result):
        raise ValueError("Two actions cannot share the same shortcut")
    return result


@runtime_checkable
class ShortcutSender(Protocol):
    def __call__(self, action: str, **fields: object) -> object: ...


class Activation:
    """Serial commands preserve start-before-stop even when microphone startup is slow."""

    def __init__(self, mode: str, send: ShortcutSender) -> None:
        self.mode, self.send = mode, send
        self.pressed: set[str] = set()
        self.primary: set[str] = set()
        self.primary_at = 0.0
        self.started = False
        self.recording_id: str | None = None

    def event(self, action: str, down: bool, when: float, token: str | None = None) -> None:  # noqa: FBT001 - ordered shortcut event fields
        key = token or action
        if down:
            self._press(action, when, key)
        elif key in self.pressed:
            self.pressed.remove(key)
            if key in self.primary:
                self.primary.remove(key)
                if not self.primary:
                    self._release(when)

    def _press(self, action: str, when: float, token: str) -> None:
        if token in self.pressed:
            return
        self.pressed.add(token)
        if action not in ("toggle", "command", "edit") and not action.startswith("profile:"):
            self.send(action)
            return
        self.primary.add(token)
        if len(self.primary) > 1:
            return
        self.primary_at = when
        self.started = False
        self.recording_id = None
        options: dict[str, object] = {}
        if action.startswith("profile:"):
            options["profile"] = action.split(":", 1)[1]
        elif action in ("command", "edit"):
            options["mode"] = action
        if self.mode in ("hold", "automatic"):
            options["held"] = True
        result = self.send("start" if self.mode == "hold" else "toggle", **options)
        if not isinstance(result, Mapping):
            raise TypeError("Invalid recording response")
        self.started = result.get("phase") == "recording"
        identity = result.get("id")
        self.recording_id = identity if isinstance(identity, str) else None

    def _release(self, when: float) -> None:
        if not self.started:
            return
        options = {"id": self.recording_id} if self.recording_id else {}
        duration = when - self.primary_at
        if self.mode == "hold" or (self.mode == "automatic" and duration >= 0.25):
            self.send("stop", **options)
            self.started = False
        elif self.mode == "automatic":
            self.send("release_hold", **options)

    def finish(self) -> None:
        try:
            if self.started and self.primary:
                self.send(
                    "stop",
                    automatic=True,
                    **({"id": self.recording_id} if self.recording_id else {}),
                )
        finally:
            self.primary.clear()
            self.pressed.clear()
            self.started = False
            self.recording_id = None


def _consume(events: queue.Queue[ShortcutEvent | str | None], activation: Activation) -> None:
    while True:
        value = events.get()
        try:
            if value is None:
                activation.finish()
            elif isinstance(value, str):
                activation.mode = value
            else:
                activation.event(*value)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Shortcut action failed", extra={"error_type": type(exc).__name__})
        if value is None:
            return


def _read_x11(config: Config, events: queue.Queue[ShortcutEvent | str | None]) -> None:
    while config.shortcut_backend == "x11":
        active_bindings = bindings(config)
        backend = X11Grabs(active_bindings)
        checked_at = 0.0
        try:
            for event in backend.events():
                if event is not None:
                    events.put(event, timeout=1)
                now = time.monotonic()
                if backend.down or now - checked_at < 1:
                    continue
                checked_at = now
                refreshed = Config.load()
                if (
                    refreshed.shortcut_backend != config.shortcut_backend
                    or refreshed.hotkey_mode != config.hotkey_mode
                    or bindings(refreshed) != active_bindings
                ):
                    config = refreshed
                    events.put(config.hotkey_mode, timeout=1)
                    break
        finally:
            backend.close()


def _run_x11(config: Config) -> int:
    if not os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        raise ValueError("Native X11 shortcuts require an X11 session; use portal on Wayland")
    events: queue.Queue[ShortcutEvent | str | None] = queue.Queue(maxsize=64)
    worker = threading.Thread(
        target=_consume, args=(events, Activation(config.hotkey_mode, request)), daemon=True
    )
    worker.start()
    try:
        _read_x11(config, events)
    finally:
        events.put(None, timeout=1)
        worker.join(timeout=12)
    return 0


def _terminate(_signal: int, _frame: FrameType | None) -> None:
    raise SystemExit(0)


def launch() -> int:
    config = Config.load()
    if config.shortcut_backend == "desktop":
        raise ValueError("Desktop owns shortcuts; choose x11 or portal explicitly")
    with (runtime_dir() / "hotkeys.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        previous_handler = None
        if threading.current_thread() is threading.main_thread():
            previous_handler = signal.signal(signal.SIGTERM, _terminate)
        try:
            ensure_daemon()
            if config.shortcut_backend == "portal":
                from .portal_shortcuts import launch as portal

                return portal()
            return _run_x11(config)
        finally:
            if previous_handler is not None:
                signal.signal(signal.SIGTERM, previous_handler)


def ensure_hotkeys() -> None:
    """Only explicit X11 configuration autostarts. Portal consent requires manual launch."""
    config = Config.load()
    if (
        config.shortcut_backend != "x11"
        or not os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
    ):
        return
    with (runtime_dir() / "hotkeys.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
    spawn_hotkeys()


def spawn_hotkeys() -> None:
    """Start the configured listener; the child owns the backend and singleton lock."""
    with (state_dir() / "hotkeys.log").open("a") as output:
        pathlib.Path(output.name).chmod(0o600)
        subprocess.Popen(  # noqa: S603
            python_command("mouthkeys", "hotkeys"),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            start_new_session=True,
        )
