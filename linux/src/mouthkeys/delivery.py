import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import replace

from .config import Config

TERMINALS = (
    "ghostty",
    "c11",
    "kitty",
    "alacritty",
    "wezterm",
    "xterm",
    "urxvt",
    "gnome-terminal",
    "org.gnome.terminal",
    "gnome terminal",
    "konsole",
    "tilix",
    "terminator",
    "foot",
    "st-256color",
    "xfce4-terminal",
    "qterminal",
    "lxterminal",
)


def terminal_name(value):
    return any(
        re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", value.lower())
        for name in TERMINALS
    )


def session():
    if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland":
        return "wayland"
    return "x11" if os.environ.get("DISPLAY") else "headless"


def run(args, **kwargs):
    try:
        return subprocess.run(args, check=True, timeout=3, capture_output=True, **kwargs)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Install {args[0]} (see linux/README.md)") from exc
    except subprocess.CalledProcessError as exc:
        detail = (
            exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        )
        raise RuntimeError(f"{args[0]} failed: {(detail or '').strip()}") from exc


class Delivery:
    send_key_guarded = True

    def capture_target(self, config=None):
        from .accessibility import read_snapshot

        before = self.focus()
        field = read_snapshot() if session() != "headless" else None
        if field and field.get("foreground") is not True:
            field = None
        after = self.focus()
        if before != after:
            return None
        return {"window": after, "field": field} if field else after

    def _field_current(self, target, *, unchanged=False):
        from .accessibility import read_snapshot
        from .atspi_helper import unchanged as same_contents

        field = target.get("field")
        current = read_snapshot(field) if field else None
        if (
            not current
            or current.get("field_id") != field.get("field_id")
            or not current.get("focused")
            or current.get("foreground") is not True
        ):
            return None
        if unchanged and not same_contents(field, current):
            return None
        return current

    def _activate_window(self, window):
        if not window:
            return False
        if session() == "x11" and re.fullmatch(r"(?:0x[\da-fA-F]+|\d+)", window):
            run(["xdotool", "windowactivate", "--sync", window])
        elif window.startswith("sway:") and re.fullmatch(r"\d+", window[5:]):
            run(["swaymsg", f"[con_id={window[5:]}]", "focus"])
        elif window.startswith("hypr:") and re.fullmatch(r"0x[\da-fA-F]+", window[5:]):
            run(["hyprctl", "dispatch", "focuswindow", "address:" + window[5:]])
        else:
            return False
        return self.focus() == window

    def prepare_target(self, config, target):
        if not isinstance(target, dict) or not target.get("field"):
            return False
        window = target.get("window")
        if (window is None or self.focus() == window) and self._field_current(
            target, unchanged=True
        ):
            return True
        if getattr(config, "return_to_start", "off") != "on":
            return False
        from .accessibility import focus_target, read_snapshot

        field = target["field"]
        current = read_snapshot(field)
        if not current or current.get("text") != field.get("text"):
            return False
        try:
            if window and not self._activate_window(window):
                return False
            return focus_target(field) and bool(self._field_current(target, unchanged=True))
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return False

    def replace_selection(self, text, config, target, *, cancel_event=None):
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
            if cancel_event is not None and cancel_event.is_set():
                return "cancelled"
            self.copy(text)
            return "copied"
        return self._deliver_field(
            text,
            replace(config, delivery=session(), insertion_mode="direct"),
            target,
            cancel_event=cancel_event,
        )

    def _native_insert(self, text, config, target, *, cancel_event=None):
        from .accessibility import insert_text

        if cancel_event is not None and cancel_event.is_set():
            return "cancelled"
        current = insert_text(target["field"], text)
        if current:
            self.last_verified = current
            if getattr(config, "copy_to_clipboard", "on") == "on":
                self.copy(text)
            return "pasted"
        # Do not paste a second time: a failed verification could follow an accepted write.
        self.copy(text)
        return "insert_requested"

    def _deliver_field(self, text, config, target, *, cancel_event=None):
        from .accessibility import read_snapshot
        from .atspi_helper import expected_text

        self.last_verified = None
        if cancel_event is not None and cancel_event.is_set():
            return "cancelled"
        if config.delivery != session() or not self.prepare_target(config, target):
            if cancel_event is not None and cancel_event.is_set():
                return "cancelled"
            self.copy(text)
            return "copied"
        if cancel_event is not None and cancel_event.is_set():
            return "cancelled"
        field = target["field"]
        if (
            config.insertion_mode == "direct"
            and not self.terminal(target)
            and field["selection_start"] == field["selection_end"]
        ):
            return self._native_insert(text, config, target, cancel_event=cancel_event)
        clipboard = None
        if config.restore_clipboard == "on":
            from .clipboard import ClipboardPaste

            try:
                clipboard = ClipboardPaste(
                    text,
                    delay_ms=config.clipboard_restore_delay_ms,
                    exclude_history=getattr(config, "clipboard_exclude_history", "on") == "on",
                )
            except (OSError, RuntimeError, subprocess.SubprocessError):
                pass
        if clipboard is None:
            if cancel_event is not None and cancel_event.is_set():
                return "cancelled"
            self.copy(text)
        if not self._field_current(target, unchanged=True):
            if clipboard:
                clipboard.finish(False)
            return "copied"
        if cancel_event is not None and cancel_event.is_set():
            if clipboard:
                clipboard.finish(False)
            return "cancelled"
        try:
            self.paste_key(config)
        except Exception:
            if clipboard:
                clipboard.finish(False)
            raise
        result = "paste_requested"
        if getattr(config, "verify_paste", "on") == "on":
            expected = expected_text(field, text)
            deadline = time.monotonic() + 0.8
            for _ in range(3):
                current = read_snapshot(field)
                if (
                    current
                    and current.get("field_id") == field["field_id"]
                    and current.get("focused")
                    and current.get("text") == expected
                    and current.get("text_start", 0) == field.get("text_start", 0)
                    and current.get("character_count", len(current["text"]))
                    == (
                        field.get("character_count", len(field["text"]))
                        + len(text)
                        - (field["selection_end"] - field["selection_start"])
                    )
                ):
                    self.last_verified = current
                    result = "pasted"
                    break
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
        if clipboard:
            clipboard.finish(result == "pasted")  # Never restore before text consumption is proven.
        return result

    def target_metadata(self, target):
        if isinstance(target, dict):
            metadata = self.target_metadata(target.get("window"))
            if not metadata.get("app") and target.get("field", {}).get("app"):
                metadata["app"] = target["field"]["app"]
            return metadata
        if session() == "wayland" and isinstance(target, str):
            try:
                if target.startswith("sway:") and re.fullmatch(r"\d+", target[5:]):
                    pending = [json.loads(run(["swaymsg", "-t", "get_tree", "-r"]).stdout)]
                    while pending:
                        node = pending.pop()
                        if node.get("id") == int(target[5:]):
                            return {
                                "app": str(
                                    node.get("app_id")
                                    or node.get("window_properties", {}).get("class")
                                    or ""
                                )[:200],
                                "window": str(node.get("name") or "")[:1000],
                            }
                        pending.extend(node.get("nodes", []) + node.get("floating_nodes", []))
                elif target.startswith("hypr:") and re.fullmatch(r"0x[\da-fA-F]+", target[5:]):
                    clients = json.loads(run(["hyprctl", "clients", "-j"]).stdout)
                    node = next((node for node in clients if node.get("address") == target[5:]), {})
                    return {
                        "app": str(node.get("class") or "")[:200],
                        "window": str(node.get("title") or "")[:1000],
                    }
            except (
                OSError,
                RuntimeError,
                ValueError,
                KeyError,
                TypeError,
                subprocess.SubprocessError,
            ):
                return {}
        if session() != "x11" or not target or not re.fullmatch(r"(?:0x[\da-fA-F]+|\d+)", target):
            return {}
        try:
            output = run(["xprop", "-id", target, "WM_CLASS", "_NET_WM_NAME", "WM_NAME"])
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return {}
        values = {}
        for line in output.stdout.decode(errors="replace").splitlines():
            key = line.split("(", 1)[0]
            quoted = re.findall(r'"((?:\\.|[^"\\])*)"', line)
            if quoted:
                values[key] = quoted[-1].replace(r"\"", '"').replace(r"\\", "\\")
        return {
            "app": values.get("WM_CLASS", "")[:200],
            "window": values.get("_NET_WM_NAME", values.get("WM_NAME", ""))[:1000],
        }

    def window_class(self, target):
        if session() != "x11" or not target:
            return None
        try:
            value = run(["xprop", "-id", str(target), "WM_CLASS"]).stdout.decode().lower()
            return value if "=" in value else None
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return None

    def terminal(self, target):
        if isinstance(target, dict):
            field = target.get("field", {})
            if field.get("terminal") or terminal_name(field.get("app", "")):
                return True
            return self.terminal(target.get("window"))
        value = (
            self.target_metadata(target).get("app", "").lower()
            if session() == "wayland"
            else self.window_class(target)
        )
        if value is None:
            return None
        return terminal_name(value)

    def focus(self):
        if session() == "wayland":
            try:
                if os.environ.get("SWAYSOCK"):
                    tree = json.loads(run(["swaymsg", "-t", "get_tree", "-r"]).stdout)
                    pending = [tree]
                    while pending:
                        node = pending.pop()
                        if node.get("focused") and node.get("type") == "con":
                            return f"sway:{int(node['id'])}"
                        pending.extend(node.get("nodes", []) + node.get("floating_nodes", []))
                elif os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
                    window = json.loads(run(["hyprctl", "activewindow", "-j"]).stdout)
                    address = window.get("address", "")
                    if re.fullmatch(r"0x[0-9a-fA-F]+", address) and address != "0x0":
                        return f"hypr:{address}"
            except (
                OSError,
                RuntimeError,
                ValueError,
                KeyError,
                TypeError,
                subprocess.SubprocessError,
            ):
                pass
            return None
        if session() != "x11":
            return None
        try:
            return run(["xdotool", "getwindowfocus"]).stdout.decode().strip() or None
        except (RuntimeError, subprocess.TimeoutExpired):
            return None

    def copy(self, text: str):
        if session() == "wayland":
            # wl-copy forks to own the selection; never use --paste-once (XWayland reads twice).
            run(["wl-copy", "--type", "text/plain;charset=utf-8"], input=text.encode())
        elif session() == "x11":
            # xclip's selection owner inherits stdout unless redirected; do not capture it.
            try:
                subprocess.run(
                    ["xclip", "-selection", "clipboard", "-in"],
                    input=text.encode(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                    timeout=3,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("Install xclip (see linux/README.md)") from exc
        else:
            raise RuntimeError("No desktop session; transcript saved in history")

    def paste_key(self, config: Config):
        backend = config.paste_backend
        if backend == "auto":
            if session() != "x11":
                raise ValueError("Wayland: use your normal paste key, or configure wtype/ydotool")
            backend = "xdotool"
        shifted = config.paste_shortcut == "ctrl+shift+v"
        if backend == "xdotool":
            if session() != "x11":
                raise ValueError("xdotool cannot paste into native Wayland apps")
            run(["xdotool", "key", "--clearmodifiers", config.paste_shortcut])
        elif backend == "wtype":
            args = ["wtype", "-M", "ctrl"]
            if shifted:
                args += ["-M", "shift"]
            args += ["-k", "v"]
            if shifted:
                args += ["-m", "shift"]
            run(args + ["-m", "ctrl"])
        else:
            keys = ["29:1"] + (["42:1"] if shifted else []) + ["47:1", "47:0"]
            run(["ydotool", "key", *keys, *(["42:0"] if shifted else []), "29:0"])

    def deliver(self, text: str, config: Config, target: str | dict | None):
        if isinstance(target, dict):
            return self._deliver_field(text, config, target)
        direct = getattr(config, "insertion_mode", "clipboard") == "direct"
        if (
            direct
            and config.delivery == "x11"
            and session() == "x11"
            and target
            and all(char.isprintable() for char in text)
            and self.focus() == target
            and self.terminal(target) is False
        ):
            if self.focus() == target:
                run(
                    ["xdotool", "type", "--clearmodifiers", "--delay", "0", "--file", "-"],
                    input=text.encode(),
                )
                if getattr(config, "copy_to_clipboard", "on") == "on":
                    self.copy(text)
                return "insert_requested"
        if (
            config.delivery == "wayland"
            and session() == "wayland"
            and target
            and target.startswith(("sway:", "hypr:"))
        ):
            clipboard = None
            if config.restore_clipboard == "on" and self.focus() == target:
                from .clipboard import ClipboardPaste

                try:
                    clipboard = ClipboardPaste(
                        text,
                        delay_ms=config.clipboard_restore_delay_ms,
                        exclude_history=config.clipboard_exclude_history == "on",
                    )
                except (OSError, RuntimeError, subprocess.SubprocessError):
                    pass
            if clipboard is None:
                self.copy(text)
            if self.focus() != target:
                if clipboard:
                    clipboard.finish(False)
                return "copied"
            time.sleep(0.15)
            if self.focus() != target:
                if clipboard:
                    clipboard.finish(False)
                return "copied"
            try:
                self.paste_key(config)
            except Exception:
                if clipboard:
                    clipboard.finish(False)
                raise
            if clipboard:
                clipboard.finish(True)
            return "paste_requested"
        clipboard = None
        if (
            getattr(config, "restore_clipboard", "off") == "on"
            and config.delivery == "x11"
            and session() == "x11"
            and target
            and self.focus() == target
        ):
            from .clipboard import ClipboardPaste

            try:
                clipboard = ClipboardPaste(
                    text,
                    delay_ms=getattr(config, "clipboard_restore_delay_ms", 500),
                    exclude_history=getattr(config, "clipboard_exclude_history", "on") == "on",
                )
            except (OSError, RuntimeError, subprocess.SubprocessError):
                pass  # Snapshot failure falls back to ordinary, recoverable clipboard delivery.
        if clipboard is None:
            self.copy(text)
        if config.delivery != "x11":
            return "copied"
        if session() != "x11" or target is None or self.focus() != target:
            if clipboard:
                clipboard.finish(False)
            return "copied"
        # Let the selection owner become ready, then recheck without activating any window.
        time.sleep(0.15)
        if self.focus() != target:
            if clipboard:
                clipboard.finish(False)
            return "copied"
        try:
            self.paste_key(config)
        except Exception:
            if clipboard:
                clipboard.finish(False)
            raise
        if clipboard:
            clipboard.finish(True)
        return "paste_requested"  # Key injection cannot prove the destination consumed the text.

    def send_key(
        self,
        config: Config,
        target: str | dict | None,
        before_send=None,
        allow_without_insert=False,
    ):
        """Send only to the same captured destination, with field identity when available."""
        if config.spoken_send != "on":
            return False
        if isinstance(target, dict):
            terminal = self.terminal(target)
            if target.get("window") and self.focus() != target["window"]:
                return False
            # Field/caret probe must be last: metadata and compositor queries can stall.
            current = self._field_current(target)
            verified = (
                target["field"] if allow_without_insert else getattr(self, "last_verified", None)
            )
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
            if session() == "wayland":
                return self._wayland_send(config, terminal=terminal, before_send=before_send)
            if session() == "x11":
                return self._x11_send(config, terminal=terminal, before_send=before_send)
            return False
        if (
            session() == "wayland"
            and config.delivery == "wayland"
            and isinstance(target, str)
            and target.startswith(("sway:", "hypr:"))
            and self.focus() == target
        ):
            terminal = self.terminal(target)
            if self.focus() != target:
                return False
            return self._wayland_send(config, terminal=terminal, before_send=before_send)
        if session() != "x11" or not target or self.focus() != target:
            return False
        terminal = self.terminal(target)
        if self.focus() != target:
            return False
        return self._x11_send(config, terminal=terminal, before_send=before_send)

    def _x11_send(self, config, *, terminal=False, before_send=None):
        key = "return" if terminal else config.spoken_send_key.lower()
        keys = {"return": "Return", "shift+return": "shift+Return", "ctrl+return": "ctrl+Return"}
        if key not in keys:
            raise ValueError("Unsupported Spoken Send key")
        if before_send is not None and not before_send():
            return False
        run(["xdotool", "key", "--clearmodifiers", keys[key]])
        return True

    def _wayland_send(self, config, *, terminal=False, before_send=None):
        key = "return" if terminal else config.spoken_send_key
        modifier = {"return": None, "shift+return": "shift", "ctrl+return": "ctrl"}[key]
        if config.paste_backend == "wtype":
            args = ["wtype"] + (["-M", modifier] if modifier else []) + ["-k", "Return"]
            if before_send is not None and not before_send():
                return False
            run(args + (["-m", modifier] if modifier else []))
        elif config.paste_backend == "ydotool":
            code = {"shift": "42", "ctrl": "29"}.get(modifier)
            keys = ([code + ":1"] if code else []) + ["28:1", "28:0"]
            if before_send is not None and not before_send():
                return False
            run(["ydotool", "key", *keys, *([code + ":0"] if code else [])])
        else:
            return False
        return True


def doctor(config: Config):
    desktop = session()
    required = ["ffmpeg"]
    if desktop == "wayland":
        required += ["wl-copy"]
    elif desktop == "x11":
        required += ["xclip"]
        if config.delivery == "x11" or config.paste_backend in ("auto", "xdotool"):
            required += ["xdotool"]
    if config.paste_backend in ("wtype", "ydotool"):
        required += [config.paste_backend]
    if getattr(config, "insertion_mode", "clipboard") == "direct":
        required += ["xprop"]
    if getattr(config, "pause_media", "off") == "on":
        required += ["playerctl"]
    if (
        getattr(config, "sound_file", "")
        or getattr(config, "sound_stop_file", "")
        or getattr(config, "sound_preset", "custom") == "mouthkeys"
    ):
        required += ["paplay"]
    return {
        "session": desktop,
        "commands": {name: shutil.which(name) for name in required},
        "audio_source": config.audio_source,
        "model": config.model,
        "delivery": config.delivery,
    }
