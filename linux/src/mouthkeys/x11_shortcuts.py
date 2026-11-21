"""Own an X11 display connection and its passive keyboard and mouse grabs."""

import ctypes
import ctypes.util
import re
import select
import time
from collections.abc import Iterator, Sequence

ShortcutBinding = tuple[str, int, str]
ShortcutEvent = tuple[str, bool, float, str]
ErrorHandler = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)


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


def _load_library() -> ctypes.CDLL:
    name = ctypes.util.find_library("X11")
    if not name:
        raise RuntimeError("libX11 unavailable")
    lib = ctypes.CDLL(name)
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
        ("XSetErrorHandler", ctypes.c_void_p, [ctypes.c_void_p]),
        ("XGetModifierMapping", ctypes.POINTER(ModifierMap), [ctypes.c_void_p]),
        ("XFreeModifiermap", ctypes.c_int, [ctypes.POINTER(ModifierMap)]),
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
        function = getattr(lib, method)
        function.restype, function.argtypes = restype, args
    return lib


class X11Grabs:
    def __init__(self, bindings: Sequence[ShortcutBinding]) -> None:
        self.lib = _load_library()
        self.display: int | None = self.lib.XOpenDisplay(None)
        if self.display is None:
            raise RuntimeError("Cannot open X11 display")
        self.actions: dict[tuple[bool, int, int], str] = {}
        self.down: dict[tuple[bool, int], str] = {}
        self.failed_grab = False
        self.previous_handler: ctypes.c_void_p | None = None
        self.error_handler = ErrorHandler(self._report_error)
        try:
            self.previous_handler = ctypes.c_void_p(
                self.lib.XSetErrorHandler(ctypes.cast(self.error_handler, ctypes.c_void_p))
            )
            self.root = self.lib.XDefaultRootWindow(self.display)
            self._install(bindings)
        except BaseException:
            self.close()
            raise

    def _report_error(self, _display: int | None, _event: int | None) -> int:
        self.failed_grab = True
        return 0

    def _install(self, bindings: Sequence[ShortcutBinding]) -> None:
        # Do not select ButtonPress on root; passive button grabs supply their own event mask.
        self.lib.XSelectInput(self.display, self.root, 1 | 2)
        supported = ctypes.c_int()
        self.lib.XkbSetDetectableAutoRepeat(self.display, 1, ctypes.byref(supported))
        if not supported.value:
            raise RuntimeError("X11 server lacks detectable auto-repeat")
        self.ignored = self._lock_modifiers()
        for action, mask, key in bindings:
            self._grab(action, mask, key)
        self.lib.XSync(self.display, 0)
        if self.failed_grab:
            raise RuntimeError("Shortcut unavailable or already owned by desktop/i3")

    def _lock_modifiers(self) -> int:
        mapping = self.lib.XGetModifierMapping(self.display)
        if not mapping:
            raise RuntimeError("Cannot read X11 modifier mapping")
        try:
            ignored = 2  # Caps Lock
            numlock = self.lib.XKeysymToKeycode(self.display, self.lib.XStringToKeysym(b"Num_Lock"))
            if not numlock:
                return ignored
            size = mapping.contents.max_keypermod
            keys = mapping.contents.modifiermap
            for modifier in range(8):
                if numlock in keys[modifier * size : (modifier + 1) * size]:
                    ignored |= 1 << modifier
            return ignored
        finally:
            self.lib.XFreeModifiermap(mapping)

    def _grab(self, action: str, mask: int, key: str) -> None:
        mouse = re.fullmatch(r"button(\d+)", key)
        code = (
            int(mouse[1])
            if mouse
            else self.lib.XKeysymToKeycode(self.display, self.lib.XStringToKeysym(key.encode()))
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

    def events(self) -> Iterator[ShortcutEvent | None]:
        fd = self.lib.XConnectionNumber(self.display)
        while True:
            if not self.lib.XPending(self.display) and not select.select([fd], [], [], 0.25)[0]:
                yield None
                continue
            raw = (ctypes.c_long * 24)()
            self.lib.XNextEvent(self.display, ctypes.byref(raw))
            event = ctypes.cast(raw, ctypes.POINTER(KeyEvent)).contents
            if decoded := self._decode(event):
                yield decoded

    def _decode(self, event: KeyEvent) -> ShortcutEvent | None:
        if event.type not in (2, 3, 4, 5):
            return None
        token = (event.type in (4, 5), event.keycode)
        down = event.type in (2, 4)
        if down:
            if token in self.down:
                return None
            action = self.actions.get((*token, event.state & 255 & ~self.ignored))
            if action:
                self.down[token] = action
        else:
            action = self.down.pop(token, None)
        return (action, down, time.monotonic(), str(token)) if action else None

    def close(self) -> None:
        if self.display is None:
            return
        display, self.display = self.display, None
        try:
            self.lib.XCloseDisplay(display)
        finally:
            # The error callback is global to libX11 and must not outlive this owner.
            if self.previous_handler is not None:
                self.lib.XSetErrorHandler(self.previous_handler)
                self.previous_handler = None
