"""Linux compositor probes and input commands; delivery policy belongs to Delivery."""

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator
from typing import Literal, TypedDict

from .config import Config

Session = Literal["x11", "wayland", "headless"]

_X11_ID = re.compile(r"(?:0x[\da-fA-F]+|\d+)")
_SWAY_ID = re.compile(r"sway:(\d+)")
_HYPR_ID = re.compile(r"hypr:(0x[\da-fA-F]+)")
_PROBE_ERRORS = (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError)


def session() -> Session:
    if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland":
        return "wayland"
    return "x11" if os.environ.get("DISPLAY") else "headless"


def run(args: list[str], *, stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(  # noqa: S603
            args, input=stdin, check=True, timeout=3, capture_output=True
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Install {args[0]} (see linux/README.md)") from exc
    except subprocess.CalledProcessError as exc:
        detail = (
            exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        )
        raise RuntimeError(f"{args[0]} failed: {(detail or '').strip()}") from exc


def _sway_nodes() -> Iterator[dict[str, object]]:
    pending: list[object] = [json.loads(run(["swaymsg", "-t", "get_tree", "-r"]).stdout)]
    while pending:
        node = pending.pop()
        if not isinstance(node, dict):
            raise TypeError("Invalid Sway window tree")
        yield node
        for key in ("nodes", "floating_nodes"):
            children = node.get(key, [])
            if not isinstance(children, list):
                raise TypeError("Invalid Sway window children")
            pending.extend(children)


def focus() -> str | None:
    desktop = session()
    try:
        if desktop == "wayland":
            return _wayland_focus()
        if desktop == "x11":
            return run(["xdotool", "getwindowfocus"]).stdout.decode().strip() or None
    except _PROBE_ERRORS:
        return None
    return None


def _wayland_focus() -> str | None:
    if os.environ.get("SWAYSOCK"):
        for node in _sway_nodes():
            identity = node.get("id")
            if node.get("focused") is True and node.get("type") == "con" and type(identity) is int:
                return f"sway:{identity}"
    elif os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        window = json.loads(run(["hyprctl", "activewindow", "-j"]).stdout)
        address = window.get("address") if isinstance(window, dict) else None
        if (
            isinstance(address, str)
            and re.fullmatch(r"0x[\da-fA-F]+", address)
            and address != "0x0"
        ):
            return f"hypr:{address}"
    return None


def activate_window(window: str) -> bool:
    """Request activation; the caller must explicitly authorize it and verify focus."""
    if session() == "x11" and _X11_ID.fullmatch(window):
        run(["xdotool", "windowactivate", "--sync", window])
    elif sway := _SWAY_ID.fullmatch(window):
        run(["swaymsg", f"[con_id={sway[1]}]", "focus"])
    elif hypr := _HYPR_ID.fullmatch(window):
        run(["hyprctl", "dispatch", "focuswindow", f"address:{hypr[1]}"])
    else:
        return False
    return True


def window_metadata(window: str | None) -> dict[str, str]:
    if not window:
        return {}
    desktop = session()
    try:
        if desktop == "wayland":
            return _wayland_metadata(window)
        if desktop == "x11" and _X11_ID.fullmatch(window):
            return _x11_metadata(window)
    except _PROBE_ERRORS:
        return {}
    return {}


def _wayland_metadata(window: str) -> dict[str, str]:
    if sway := _SWAY_ID.fullmatch(window):
        identity = int(sway[1])
        for node in _sway_nodes():
            if node.get("id") == identity:
                properties = node.get("window_properties")
                xwayland_class = properties.get("class") if isinstance(properties, dict) else None
                return {
                    "app": str(node.get("app_id") or xwayland_class or "")[:200],
                    "window": str(node.get("name") or "")[:1000],
                }
    elif hypr := _HYPR_ID.fullmatch(window):
        clients = json.loads(run(["hyprctl", "clients", "-j"]).stdout)
        if not isinstance(clients, list):
            return {}
        for node in clients:
            if isinstance(node, dict) and node.get("address") == hypr[1]:
                return {
                    "app": str(node.get("class") or "")[:200],
                    "window": str(node.get("title") or "")[:1000],
                }
        return {"app": "", "window": ""}
    return {}


def _x11_metadata(window: str) -> dict[str, str]:
    output = run(["xprop", "-id", window, "WM_CLASS", "_NET_WM_NAME", "WM_NAME"])
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


def window_class(window: str | None) -> str | None:
    if session() != "x11" or not window or not _X11_ID.fullmatch(window):
        return None
    try:
        value = run(["xprop", "-id", window, "WM_CLASS"]).stdout.decode().lower()
    except _PROBE_ERRORS:
        return None
    return value if "=" in value else None


def type_x11(text: str) -> None:
    run(
        ["xdotool", "type", "--clearmodifiers", "--delay", "0", "--file", "-"],
        stdin=text.encode(),
    )


def copy(text: str) -> None:
    desktop = session()
    if desktop == "wayland":
        # wl-copy forks to own the selection; never use --paste-once (XWayland reads twice).
        args = ["wl-copy", "--type", "text/plain;charset=utf-8"]
    elif desktop == "x11":
        args = ["xclip", "-selection", "clipboard", "-in"]
    else:
        raise RuntimeError("No desktop session; transcript saved in history")
    # The owner outlives the launcher. An inherited pipe can keep run() waiting for EOF.
    try:
        subprocess.run(  # noqa: S603
            args,
            input=text.encode(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=3,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Install {args[0]} (see linux/README.md)") from exc


def paste_key(config: Config) -> None:
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
        run([*args, "-m", "ctrl"])
    else:
        keys = ["29:1"] + (["42:1"] if shifted else []) + ["47:1", "47:0"]
        run(["ydotool", "key", *keys, *(["42:0"] if shifted else []), "29:0"])


def send_x11(
    config: Config, *, terminal: bool | None, before_send: Callable[[], bool] | None
) -> bool:
    key = "return" if terminal else config.spoken_send_key.lower()
    keys = {"return": "Return", "shift+return": "shift+Return", "ctrl+return": "ctrl+Return"}
    if key not in keys:
        raise ValueError("Unsupported Spoken Send key")
    if before_send is not None and not before_send():
        return False
    run(["xdotool", "key", "--clearmodifiers", keys[key]])
    return True


def send_wayland(
    config: Config, *, terminal: bool | None, before_send: Callable[[], bool] | None
) -> bool:
    key = "return" if terminal else config.spoken_send_key
    modifier = {"return": None, "shift+return": "shift", "ctrl+return": "ctrl"}[key]
    if config.paste_backend == "wtype":
        args = [
            "wtype",
            *(["-M", modifier] if modifier else []),
            "-k",
            "Return",
            *(["-m", modifier] if modifier else []),
        ]
    elif config.paste_backend == "ydotool":
        code = {"shift": "42", "ctrl": "29"}[modifier] if modifier else None
        args = [
            "ydotool",
            "key",
            *([code + ":1"] if code else []),
            "28:1",
            "28:0",
            *([code + ":0"] if code else []),
        ]
    else:
        return False
    if before_send is not None and not before_send():
        return False
    run(args)
    return True


def automatic_delivery_warning(config: Config) -> str | None:
    """Check configuration and tools without probing focus or sending input."""
    if config.delivery == "clipboard":
        return None
    desktop = session()
    if config.delivery != desktop:
        if config.delivery == "x11" and desktop == "wayland":
            return (
                "X11 auto-paste needs an X11 login session; this session is Wayland "
                "(even if DISPLAY is set). Choose Wayland auto-paste or Clipboard only."
            )
        return (
            f"Delivery is set to {config.delivery}, but the detected session is {desktop}. "
            "Choose delivery for your desktop session or Clipboard only."
        )
    backend = config.paste_backend
    if desktop == "x11" and backend == "auto":
        backend = "xdotool"
    warning = None
    if desktop == "wayland" and backend in ("auto", "xdotool"):
        warning = "Wayland paste keys need a compatible wtype or configured ydotool backend."
    elif not shutil.which(backend):
        warning = f"Paste backend {backend} not found. Install it or choose another backend."
    elif desktop == "x11" and backend != "xdotool" and not shutil.which("xdotool"):
        warning = "Install xdotool for X11 window detection and focus checks."
    return warning


class DeliveryDoctor(TypedDict):
    session: str
    commands: dict[str, str | None]
    audio_source: str
    model: str
    delivery: str
    delivery_warning: str | None


def doctor(config: Config) -> DeliveryDoctor:
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
        "delivery_warning": automatic_delivery_warning(config),
    }
