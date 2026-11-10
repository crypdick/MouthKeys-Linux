"""Optional native shortcuts. Desktop/i3 ownership remains the default.

X11 uses passive grabs, never polling every key. Portal uses system-managed
GlobalShortcuts; running that backend explicitly may open its consent dialog.
"""

import ctypes
import ctypes.util
import fcntl
import os
import queue
import re
import select
import signal
import subprocess
import threading
import time

from .config import Config, runtime_dir, state_dir
from .processes import python_command

MODIFIERS = {"shift": 1, "ctrl": 4, "control": 4, "alt": 8, "super": 64, "meta": 64, "win": 64}


def parse_chord(value):
    if value.endswith("++"):
        value = value[:-1] + "plus"
    parts = [part.strip().lower() for part in value.split("+")]
    if not parts or not parts[-1] or any(part not in MODIFIERS for part in parts[:-1]):
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


def bindings(config):
    result = []
    for action, value in (
        ("toggle", config.toggle_shortcut),
        ("cancel", config.cancel_shortcut),
        ("paste", config.paste_shortcut_key),
        ("reprocess", config.reprocess_shortcut_key),
        ("command", getattr(config, "command_shortcut_key", "")),
        ("edit", getattr(config, "edit_shortcut_key", "")),
    ):
        for chord in value.split(","):
            if chord.strip():
                result.append((action, *parse_chord(chord)))
    from .profiles import load_profiles

    for profile in load_profiles():
        for chord in profile.get("shortcut", "").split(","):
            if chord.strip():
                result.append(("profile:" + profile["id"], *parse_chord(chord)))
    keys = [(mask, key) for _, mask, key in result]
    if len(set(keys)) != len(keys):
        raise ValueError("Two actions cannot share the same shortcut")
    return result


class Activation:
    """Serial commands preserve start-before-stop even when microphone startup is slow."""

    def __init__(self, mode, send):
        self.mode, self.send = mode, send
        self.pressed = {}
        self.started = False
        self.primary = set()
        self.primary_at = 0
        self.recording_id = None

    def event(self, action, down, when, token=None):
        token = token or action
        if down:
            if token in self.pressed:
                return
            self.pressed[token] = when
            recording = action in ("toggle", "command", "edit") or action.startswith("profile:")
            if not recording:
                self.send(action)
            else:
                self.primary.add(token)
                if len(self.primary) > 1:
                    return
                self.primary_at = when
                options = {}
                if action.startswith("profile:"):
                    options["profile"] = action.split(":", 1)[1]
                elif action in ("command", "edit"):
                    options["mode"] = action
                if self.mode in ("hold", "automatic"):
                    options["held"] = True
                result = self.send("start" if self.mode == "hold" else "toggle", **options)
                self.started = result.get("phase") == "recording"
                self.recording_id = result.get("id")
        elif token in self.pressed:
            self.pressed.pop(token)
            duration = when - self.primary_at
            self.primary.discard(token)
            recording = action in ("toggle", "command", "edit") or action.startswith("profile:")
            if recording and self.primary:
                return
            if (
                recording
                and self.started
                and (self.mode == "hold" or (self.mode == "automatic" and duration >= 0.25))
            ):
                self.send("stop", **({"id": self.recording_id} if self.recording_id else {}))
                self.started = False
            elif recording and self.started and self.mode == "automatic":
                self.send(
                    "release_hold", **({"id": self.recording_id} if self.recording_id else {})
                )

    def finish(self):
        if self.started and self.primary:
            self.send(
                "stop", automatic=True, **({"id": self.recording_id} if self.recording_id else {})
            )
        self.primary.clear()
        self.pressed.clear()
        self.started = False


class KeyEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("serial", ctypes.c_ulong),
        ("send_event", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("window", ctypes.c_ulong),
        ("root", ctypes.c_ulong),
        ("subwindow", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("x_root", ctypes.c_int),
        ("y_root", ctypes.c_int),
        ("state", ctypes.c_uint),
        ("keycode", ctypes.c_uint),
        ("same_screen", ctypes.c_int),
    ]


class ModifierMap(ctypes.Structure):
    _fields_ = [("max_keypermod", ctypes.c_int), ("modifiermap", ctypes.POINTER(ctypes.c_ubyte))]


class X11:
    def __init__(self, config):
        name = ctypes.util.find_library("X11")
        if not name:
            raise RuntimeError("libX11 unavailable")
        self.lib = ctypes.CDLL(name)
        for method, restype, args in (
            ("XOpenDisplay", ctypes.c_void_p, [ctypes.c_char_p]),
            ("XDefaultRootWindow", ctypes.c_ulong, [ctypes.c_void_p]),
            ("XStringToKeysym", ctypes.c_ulong, [ctypes.c_char_p]),
            ("XKeysymToKeycode", ctypes.c_ubyte, [ctypes.c_void_p, ctypes.c_ulong]),
            ("XConnectionNumber", ctypes.c_int, [ctypes.c_void_p]),
            ("XPending", ctypes.c_int, [ctypes.c_void_p]),
            ("XNextEvent", ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]),
            ("XCloseDisplay", ctypes.c_int, [ctypes.c_void_p]),
            ("XSync", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
            ("XSelectInput", ctypes.c_int, [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_long]),
            (
                "XGrabKey",
                ctypes.c_int,
                [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_uint,
                    ctypes.c_ulong,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                ],
            ),
            (
                "XGrabButton",
                ctypes.c_int,
                [
                    ctypes.c_void_p,
                    ctypes.c_uint,
                    ctypes.c_uint,
                    ctypes.c_ulong,
                    ctypes.c_int,
                    ctypes.c_uint,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_ulong,
                    ctypes.c_ulong,
                ],
            ),
            (
                "XkbSetDetectableAutoRepeat",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p],
            ),
        ):
            function = getattr(self.lib, method)
            function.restype, function.argtypes = restype, args
        self.display = self.lib.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError("Cannot open X11 display")
        self.errors = []
        callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
        self.error_handler = callback_type(lambda *_: self.errors.append(True) or 0)
        self.lib.XSetErrorHandler.argtypes = [callback_type]
        self.lib.XSetErrorHandler.restype = ctypes.c_void_p
        self.lib.XSetErrorHandler(self.error_handler)
        self.root = self.lib.XDefaultRootWindow(self.display)
        self.actions = {}
        self.down = {}
        self.ignored = 2
        try:
            # Do not select ButtonPress on root: window managers can own that mask.
            # Passive button grabs supply their own press/release event mask below.
            self.lib.XSelectInput(self.display, self.root, 1 | 2)
            supported = ctypes.c_int()
            self.lib.XkbSetDetectableAutoRepeat(self.display, 1, ctypes.byref(supported))
            if not supported.value:
                raise RuntimeError("X11 server lacks detectable auto-repeat")
            self.lib.XGetModifierMapping.argtypes = [ctypes.c_void_p]
            self.lib.XGetModifierMapping.restype = ctypes.POINTER(ModifierMap)
            self.lib.XFreeModifiermap.argtypes = [ctypes.POINTER(ModifierMap)]
            mapping = self.lib.XGetModifierMapping(self.display)
            try:
                numlock = self.lib.XKeysymToKeycode(
                    self.display, self.lib.XStringToKeysym(b"Num_Lock")
                )
                for modifier in range(8):
                    for index in range(mapping.contents.max_keypermod):
                        if (
                            numlock
                            and mapping.contents.modifiermap[
                                modifier * mapping.contents.max_keypermod + index
                            ]
                            == numlock
                        ):
                            self.ignored |= 1 << modifier
            finally:
                self.lib.XFreeModifiermap(mapping)
            for action, mask, key in bindings(config):
                mouse = re.fullmatch(r"button(\d+)", key)
                code = (
                    int(mouse[1])
                    if mouse
                    else self.lib.XKeysymToKeycode(
                        self.display, self.lib.XStringToKeysym(key.encode())
                    )
                )
                if not code or (mouse and not 1 <= code <= 32):
                    raise ValueError(f"Unavailable shortcut key: {key}")
                for locks in {0, 2, self.ignored & ~2, self.ignored}:
                    if mouse:
                        self.lib.XGrabButton(
                            self.display, code, mask | locks, self.root, 0, 4 | 8, 1, 1, 0, 0
                        )
                    else:
                        self.lib.XGrabKey(self.display, code, mask | locks, self.root, 0, 1, 1)
                self.actions[(bool(mouse), code, mask)] = action
            self.lib.XSync(self.display, 0)
            if self.errors:
                raise RuntimeError("Shortcut unavailable or already owned by desktop/i3")
        except Exception:
            self.close()
            raise

    def events(self):
        fd = self.lib.XConnectionNumber(self.display)
        while True:
            if not self.lib.XPending(self.display):
                if not select.select([fd], [], [], 0.25)[0]:
                    yield None
                    continue
            raw = (ctypes.c_long * 24)()
            self.lib.XNextEvent(self.display, ctypes.byref(raw))
            event = ctypes.cast(raw, ctypes.POINTER(KeyEvent)).contents
            if event.type in (2, 3, 4, 5):
                token = (event.type in (4, 5), event.keycode)
                down = event.type in (2, 4)
                if down:
                    if token in self.down:
                        continue
                    action = self.actions.get((*token, (event.state & 255) & ~self.ignored))
                    if action:
                        self.down[token] = action
                else:
                    action = self.down.pop(token, None)
                if action:
                    yield action, down, time.monotonic(), str(token)

    def close(self):
        if self.display:
            self.lib.XCloseDisplay(self.display)
            self.display = None


def launch():
    from .service import ensure_daemon, request

    config = Config.load()
    if config.shortcut_backend == "desktop":
        raise ValueError("Desktop owns shortcuts; choose x11 or portal explicitly")
    with (runtime_dir() / "hotkeys.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        if threading.current_thread() is threading.main_thread():

            def terminate(*_):
                raise SystemExit(0)

            signal.signal(signal.SIGTERM, terminate)
        ensure_daemon()
        if config.shortcut_backend == "portal":
            from .portal_shortcuts import launch as portal

            return portal()
        if not os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            raise ValueError("Native X11 shortcuts require an X11 session; use portal on Wayland")
        events = queue.Queue(maxsize=64)
        activation = Activation(config.hotkey_mode, request)

        def consume():
            while True:
                value = events.get()
                if value is None:
                    return
                try:
                    if value[0] == "_mode":
                        activation.mode = value[1]
                    elif value[0] == "_finish":
                        activation.finish()
                    else:
                        activation.event(*value)
                except Exception as exc:
                    print(f"Shortcut: {exc}", flush=True)

        worker = threading.Thread(target=consume, daemon=True)
        worker.start()
        try:
            while True:
                active_bindings = bindings(config)
                backend = X11(config)
                checked_at = 0
                try:
                    for event in backend.events():
                        if event:
                            events.put(event, timeout=1)
                        if not backend.down and time.monotonic() - checked_at >= 1:
                            checked_at = time.monotonic()
                            refreshed = Config.load()
                            if (
                                bindings(refreshed),
                                refreshed.hotkey_mode,
                                refreshed.shortcut_backend,
                            ) != (active_bindings, config.hotkey_mode, config.shortcut_backend):
                                config = refreshed
                                events.put(("_mode", config.hotkey_mode), timeout=1)
                                break
                finally:
                    backend.close()
                if config.shortcut_backend != "x11":
                    return 0  # Switching to portal still requires an explicit setup request.
        finally:
            events.put(("_finish",), timeout=1)
            events.put(None, timeout=1)
            worker.join(timeout=12)


def ensure_hotkeys():
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
    with (state_dir() / "hotkeys.log").open("a") as output:
        os.chmod(output.name, 0o600)
        subprocess.Popen(
            python_command("mouthkeys", "hotkeys"),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            start_new_session=True,
        )
