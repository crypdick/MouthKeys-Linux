import json
import os
import re
import shutil
import subprocess
import time

from .config import Config


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
    def target_metadata(self, target):
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
        value = self.window_class(target)
        if value is None:
            return None
        return any(
            name in value
            for name in (
                "ghostty",
                "c11",
                "kitty",
                "alacritty",
                "wezterm",
                "xterm",
                "urxvt",
                "gnome-terminal",
                "konsole",
                "tilix",
                "terminator",
                "foot",
                "st-256color",
            )
        )

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

    def deliver(self, text: str, config: Config, target: str | None):
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
            self.copy(text)
            if self.focus() != target:
                return "copied"
            time.sleep(0.15)
            if self.focus() != target:
                return "copied"
            self.paste_key(config)
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
                    text, delay_ms=getattr(config, "clipboard_restore_delay_ms", 500)
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

    def send_key(self, config: Config, target: str | None):
        """Spoken Send requires X11 target continuity; generic Wayland cannot prove it."""
        if session() != "x11" or not target or self.focus() != target:
            return False
        key = getattr(config, "spoken_send_key", "return").lower()
        if self.terminal(target):
            key = "return"
        keys = {"return": "Return", "shift+return": "shift+Return", "ctrl+return": "ctrl+Return"}
        if key not in keys:
            raise ValueError("Unsupported Spoken Send key")
        if self.focus() != target:
            return False
        run(["xdotool", "key", "--clearmodifiers", keys[key]])
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
    if getattr(config, "sound_file", ""):
        required += ["paplay"]
    return {
        "session": desktop,
        "commands": {name: shutil.which(name) for name in required},
        "audio_source": config.audio_source,
        "model": config.model,
        "delivery": config.delivery,
    }
