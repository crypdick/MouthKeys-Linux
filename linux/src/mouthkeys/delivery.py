"""Guard transcript insertion and Spoken Send with the captured destination."""

import subprocess
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from threading import Event
from typing import TypedDict

from . import accessibility, atspi_helper, clipboard, desktop_input
from .config import Config
from .desktop import terminal_name


class CapturedField(TypedDict, total=False):
    window: str | None
    field: dict[str, object]


Target = str | CapturedField | None


def _temporary_clipboard(text: str, config: Config) -> Callable[[bool], bool] | None:
    """Set temporary text and return the callback to restore or retain it."""
    # Snapshot failure falls back to ordinary, recoverable clipboard delivery.
    with suppress(OSError, RuntimeError, subprocess.SubprocessError):
        return clipboard.ClipboardPaste(
            text,
            delay_ms=config.clipboard_restore_delay_ms,
            exclude_history=config.clipboard_exclude_history == "on",
        ).finish
    return None


def _verify_field_paste(field: dict[str, object], text: str) -> dict[str, object] | None:
    before = field.get("text")
    field_id = field.get("field_id")
    start, end = field.get("selection_start"), field.get("selection_end")
    if not isinstance(before, str) or not isinstance(field_id, str):
        return None
    offset = field.get("text_start", 0)
    count = field.get("character_count", len(before))
    if (
        type(start) is not int
        or type(end) is not int
        or type(count) is not int
        or type(offset) is not int
    ):
        return None
    # Use the existing AT-SPI replacement rules to compare against the same field region.
    expected: str = atspi_helper.expected_text(field, text)
    expected_count = count + len(text) - (end - start)
    deadline = time.monotonic() + 0.8
    for _ in range(3):
        current = accessibility.read_snapshot(field)
        if (
            current
            and current.get("field_id") == field_id
            and current.get("focused")
            and current.get("text") == expected
            and current.get("text_start", 0) == offset
            and current.get("character_count", len(expected)) == expected_count
        ):
            return current
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    return None


class Delivery:
    send_key_guarded = True
    last_verified: dict[str, object] | None = None

    def capture_target(self, _config: Config | None = None) -> Target:
        before = self.focus()
        field = (
            accessibility.read_snapshot(include_protected=True)
            if desktop_input.session() != "headless"
            else None
        )
        if field and field.get("protected"):
            return None
        if field and field.get("foreground") is not True:
            field = None
        after = self.focus()
        if before != after:
            return None
        return {"window": after, "field": field} if field else after

    def _field_current(
        self, target: CapturedField, *, unchanged: bool = False
    ) -> dict[str, object] | None:
        field = target.get("field")
        if not field:
            return None
        current = accessibility.read_snapshot(field)
        if (
            not current
            or current.get("field_id") != field.get("field_id")
            or not current.get("focused")
            or current.get("foreground") is not True
        ):
            return None
        if unchanged and not atspi_helper.unchanged(field, current):
            return None
        return current

    def prepare_target(self, config: Config, target: Target) -> bool:
        if not isinstance(target, dict) or not target.get("field"):
            return False
        window = target.get("window")
        if (window is None or self.focus() == window) and self._field_current(
            target, unchanged=True
        ):
            return True
        if getattr(config, "return_to_start", "off") != "on":
            return False
        field = target["field"]
        current = accessibility.read_snapshot(field)
        if not current or current.get("text") != field.get("text"):
            return False
        try:
            window_ready = not window or (
                desktop_input.activate_window(window) and self.focus() == window
            )
            return (
                window_ready
                and accessibility.focus_target(field)
                and bool(self._field_current(target, unchanged=True))
            )
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return False

    def replace_selection(
        self, text: str, config: Config, target: Target, *, cancel_event: Event | None = None
    ) -> str:
        """Explicit edit: replace only the unchanged selected range, preserving other rich text."""
        self.last_verified = None
        if cancel_event is not None and cancel_event.is_set():
            return "cancelled"
        if (
            not isinstance(target, dict)
            or not target.get("field")
            or target["field"].get("terminal")
            or not self.prepare_target(config, target)
        ):
            return self._copy_unless_cancelled(text, cancel_event)
        return self._deliver_field(
            text,
            replace(config, delivery=desktop_input.session(), insertion_mode="direct"),
            target,
            cancel_event=cancel_event,
        )

    def _native_insert(
        self, text: str, config: Config, target: CapturedField, *, cancel_event: Event | None
    ) -> str:
        if cancel_event is not None and cancel_event.is_set():
            return "cancelled"
        current = accessibility.insert_text(target["field"], text)
        if current:
            self.last_verified = current
            if getattr(config, "copy_to_clipboard", "on") == "on":
                self.copy(text)
            return "pasted"
        # Do not paste a second time: a failed verification could follow an accepted write.
        self.copy(text)
        return "insert_requested"

    def _copy_unless_cancelled(self, text: str, cancel_event: Event | None) -> str:
        if cancel_event is not None and cancel_event.is_set():
            return "cancelled"
        self.copy(text)
        return "copied"

    def _deliver_field(
        self, text: str, config: Config, target: CapturedField, *, cancel_event: Event | None = None
    ) -> str:
        self.last_verified = None
        if cancel_event is not None and cancel_event.is_set():
            return "cancelled"
        if config.delivery != desktop_input.session() or not self.prepare_target(config, target):
            return self._copy_unless_cancelled(text, cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            return "cancelled"
        field = target["field"]
        if (
            config.insertion_mode == "direct"
            and not self.terminal(target)
            and field["selection_start"] == field["selection_end"]
        ):
            return self._native_insert(text, config, target, cancel_event=cancel_event)
        finish_paste = (
            _temporary_clipboard(text, config) if config.restore_clipboard == "on" else None
        )
        if finish_paste is None:
            if cancel_event is not None and cancel_event.is_set():
                return "cancelled"
            self.copy(text)
        return self._paste_field(text, config, target, finish_paste, cancel_event=cancel_event)

    def _paste_field(
        self,
        text: str,
        config: Config,
        target: CapturedField,
        finish_paste: Callable[[bool], bool] | None,
        *,
        cancel_event: Event | None,
    ) -> str:
        verified = None
        try:
            if not self._field_current(target, unchanged=True):
                return "copied"
            if cancel_event is not None and cancel_event.is_set():
                return "cancelled"
            self.paste_key(config)
            if config.verify_paste == "on":
                verified = _verify_field_paste(target["field"], text)
                self.last_verified = verified
            return "pasted" if verified is not None else "paste_requested"
        finally:
            if finish_paste:
                finish_paste(verified is not None)

    def target_metadata(self, target: Target) -> dict[str, str]:
        if isinstance(target, dict):
            metadata = desktop_input.window_metadata(target.get("window"))
            field = target.get("field")
            app = field.get("app") if field else None
            if not metadata.get("app") and isinstance(app, str) and app:
                metadata["app"] = app
            return metadata
        return desktop_input.window_metadata(target)

    def terminal(self, target: Target) -> bool | None:
        if isinstance(target, dict):
            field = target.get("field", {})
            app = field.get("app")
            if field.get("terminal") or (isinstance(app, str) and terminal_name(app)):
                return True
            return self.terminal(target.get("window"))
        value = (
            self.target_metadata(target).get("app", "").lower()
            if desktop_input.session() == "wayland"
            else desktop_input.window_class(target)
        )
        if value is None:
            return None
        return terminal_name(value)

    def focus(self) -> str | None:
        return desktop_input.focus()

    def copy(self, text: str) -> None:
        desktop_input.copy(text)

    def paste_key(self, config: Config) -> None:
        desktop_input.paste_key(config)

    def deliver(self, text: str, config: Config, target: Target) -> str:
        if isinstance(target, dict):
            return self._deliver_field(text, config, target)
        if config.destination_guard == "field":
            # A window ID cannot establish which tab, terminal pane, or field owns input.
            # Explicit focus-now paste remains available through the manual service action.
            self.copy(text)
            return "copied"
        direct = getattr(config, "insertion_mode", "clipboard") == "direct"
        if (
            direct
            and config.delivery == "x11"
            and desktop_input.session() == "x11"
            and target
            and all(char.isprintable() for char in text)
            and self.focus() == target
            and self.terminal(target) is False
        ) and self.focus() == target:
            desktop_input.type_x11(text)
            if getattr(config, "copy_to_clipboard", "on") == "on":
                self.copy(text)
            return "insert_requested"
        desktop = desktop_input.session()
        if config.delivery == desktop and (
            desktop == "x11"
            or (desktop == "wayland" and target and target.startswith(("sway:", "hypr:")))
        ):
            return self._paste_window(text, config, target, desktop=desktop)
        self.copy(text)
        return "copied"

    def _paste_window(
        self, text: str, config: Config, target: str | None, *, desktop: desktop_input.Session
    ) -> str:
        finish_paste = None
        if (
            config.restore_clipboard == "on"
            and desktop_input.session() == desktop
            and target
            and self.focus() == target
        ):
            finish_paste = _temporary_clipboard(text, config)
        if finish_paste is None:
            self.copy(text)
        requested = False
        try:
            if desktop_input.session() != desktop or target is None or self.focus() != target:
                return "copied"
            # Let the selection owner become ready, then recheck without activating any window.
            time.sleep(0.15)
            if self.focus() != target:
                return "copied"
            self.paste_key(config)
            requested = True
            # Key injection cannot prove the destination consumed the text.
            return "paste_requested"
        finally:
            if finish_paste:
                finish_paste(requested)

    def send_key(
        self,
        config: Config,
        target: Target,
        before_send: Callable[[], bool] | None = None,
        *,
        allow_without_insert: bool = False,
    ) -> bool:
        """Require captured field identity unless weaker window checks were explicitly chosen."""
        if config.spoken_send != "on":
            return False
        if isinstance(target, dict):
            return self._send_with_field(
                config, target, before_send, allow_without_insert=allow_without_insert
            )
        if config.destination_guard == "field":
            return False
        if (
            desktop_input.session() == "wayland"
            and config.delivery == "wayland"
            and isinstance(target, str)
            and target.startswith(("sway:", "hypr:"))
            and self.focus() == target
        ):
            terminal = self.terminal(target)
            return self.focus() == target and desktop_input.send_wayland(
                config, terminal=terminal, before_send=before_send
            )
        if desktop_input.session() != "x11" or not target or self.focus() != target:
            return False
        terminal = self.terminal(target)
        return self.focus() == target and desktop_input.send_x11(
            config, terminal=terminal, before_send=before_send
        )

    def _send_with_field(
        self,
        config: Config,
        target: CapturedField,
        before_send: Callable[[], bool] | None,
        *,
        allow_without_insert: bool,
    ) -> bool:
        terminal = self.terminal(target)
        if target.get("window") and self.focus() != target["window"]:
            return False
        # Field/caret probe must be last: metadata and compositor queries can stall.
        current = self._field_current(target)
        verified = target["field"] if allow_without_insert else self.last_verified
        if (
            not current
            or not verified
            or current.get("protected") is not False
            or verified.get("protected") is not False
            or type(verified.get("caret")) is not int
            or current["field_id"] != verified["field_id"]
            or current["text"] != verified["text"]
            or current.get("caret") != verified.get("caret")
            or current.get("selection_start") != verified.get("caret")
            or current.get("selection_end") != verified.get("caret")
            or verified.get("selection_start") != verified.get("caret")
            or verified.get("selection_end") != verified.get("caret")
            or any(
                current.get(key) != verified.get(key)
                for key in ("text_start", "character_count", "complete")
            )
        ):
            return False
        if desktop_input.session() == "wayland":
            return desktop_input.send_wayland(config, terminal=terminal, before_send=before_send)
        if desktop_input.session() == "x11":
            return desktop_input.send_x11(config, terminal=terminal, before_send=before_send)
        return False
