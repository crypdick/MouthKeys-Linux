import ctypes
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from mouthkeys import x11_shortcuts


@pytest.fixture
def server(monkeypatch):
    lib = Mock()
    lib.__class__ = ctypes.CDLL  # libX11 functions are resolved dynamically.
    lib.XOpenDisplay.return_value = 11
    lib.XDefaultRootWindow.return_value = 22
    lib.XConnectionNumber.return_value = 33
    lib.XStringToKeysym.side_effect = lambda key: {b"space": 32, b"Num_Lock": 77}.get(key, 0)
    lib.XKeysymToKeycode.side_effect = lambda _display, symbol: symbol
    original = 44
    state = SimpleNamespace(handler=original, autorepeat=True, events=deque())

    def repeat(_display, _enable, result):
        ctypes.cast(result, ctypes.POINTER(ctypes.c_int))[0] = int(state.autorepeat)

    def set_handler(handler):
        previous = state.handler
        state.handler = handler.value
        return previous

    def read_event(_display, pointer):
        event = state.events.popleft()
        ctypes.memmove(pointer, ctypes.byref(event), ctypes.sizeof(event))
        return 0

    keys = (ctypes.c_ubyte * 16)()
    keys[9] = 77  # Num Lock occupies Mod2, second key in the group.
    mapping = x11_shortcuts.ModifierMap(2, keys)
    lib.XGetModifierMapping.return_value = ctypes.pointer(mapping)
    lib.XkbSetDetectableAutoRepeat.side_effect = repeat
    lib.XSetErrorHandler.side_effect = set_handler
    lib.XPending.side_effect = lambda _display: len(state.events)
    lib.XNextEvent.side_effect = read_event
    monkeypatch.setattr(x11_shortcuts.ctypes.util, "find_library", Mock(return_value="fake-xlib"))
    monkeypatch.setattr(x11_shortcuts.ctypes, "CDLL", Mock(return_value=lib))
    return lib, state, original


def test_native_grabs_handle_locks_but_release_by_original_key(server, monkeypatch):
    lib, state, original = server
    state.events.extend(
        x11_shortcuts.KeyEvent(type=kind, keycode=code, state=mask)
        for kind, code, mask in (
            (9, 32, 4),  # Not a key or button.
            (2, 32, 0),  # Unmodified key does not match.
            (3, 32, 0),  # Release with no recorded press does nothing.
            (2, 32, 4 | 2 | 16 | 256),  # Lock and mouse buttons ignored.
            (2, 32, 4),  # Repeat is ignored even if delivered.
            (4, 8, 64 | 16),
            (3, 32, 0),  # Release works after Ctrl is lifted.
            (5, 8, 0),
        )
    )
    select = Mock(return_value=([], [], []))
    monkeypatch.setattr(x11_shortcuts, "select", SimpleNamespace(select=select))
    monkeypatch.setattr(x11_shortcuts, "time", SimpleNamespace(monotonic=Mock(return_value=4.0)))
    native = x11_shortcuts.X11Grabs([("toggle", 4, "space"), ("cancel", 64, "button8")])
    assert state.handler != original
    assert {grab.args[2] for grab in lib.XGrabKey.call_args_list} == {4, 6, 20, 22}
    assert {grab.args[2] for grab in lib.XGrabButton.call_args_list} == {64, 66, 80, 82}
    lib.XSelectInput.assert_called_once_with(11, 22, 3)
    lib.XFreeModifiermap.assert_called_once()
    events = native.events()
    assert next(events) == ("toggle", True, 4.0, "(False, 32)")
    assert native.down == {(False, 32): "toggle"}
    assert next(events) == ("cancel", True, 4.0, "(True, 8)")
    assert next(events) == ("toggle", False, 4.0, "(False, 32)")
    assert next(events) == ("cancel", False, 4.0, "(True, 8)")
    assert next(events) is None
    assert next(events) is None
    assert not native.down
    assert select.call_args_list == [call([33], [], [], 0.25)] * 2
    native.close()
    native.close()
    lib.XCloseDisplay.assert_called_once_with(11)
    assert state.handler == original


def test_without_num_lock_only_caps_combinations_are_grabbed(server, monkeypatch):
    lib, state, _ = server
    lib.XStringToKeysym.side_effect = lambda key: 32 if key == b"space" else 0
    native = x11_shortcuts.X11Grabs([("toggle", 4, "space")])
    assert {grab.args[2] for grab in lib.XGrabKey.call_args_list} == {4, 6}

    def ready(*_args):
        state.events.append(x11_shortcuts.KeyEvent(type=2, keycode=32, state=6))
        return [33], [], []

    monkeypatch.setattr(x11_shortcuts, "select", SimpleNamespace(select=ready))
    assert next(native.events())[:2] == ("toggle", True)
    native.close()


def test_unavailable_x11_library_never_opens_display(server, monkeypatch):
    lib, _, _ = server
    monkeypatch.setattr(x11_shortcuts.ctypes.util, "find_library", Mock(return_value=None))
    with pytest.raises(RuntimeError, match="libX11 unavailable"):
        x11_shortcuts.X11Grabs([])
    lib.XOpenDisplay.assert_not_called()


def test_unavailable_display_never_installs_handler(server):
    lib, _, _ = server
    lib.XOpenDisplay.return_value = None
    with pytest.raises(RuntimeError, match="Cannot open X11 display"):
        x11_shortcuts.X11Grabs([])
    lib.XSetErrorHandler.assert_not_called()
    lib.XCloseDisplay.assert_not_called()


@pytest.mark.parametrize(
    ("failure", "exception", "message"),
    [
        ("repeat", RuntimeError, "lacks detectable auto-repeat"),
        ("mapping", RuntimeError, "Cannot read X11 modifier mapping"),
        ("conflict", RuntimeError, "already owned by desktop/i3"),
        ("missing-key", ValueError, "Unavailable shortcut key"),
        ("button33", ValueError, "Unavailable shortcut key"),
        ("button0", ValueError, "Unavailable shortcut key"),
    ],
)
def test_setup_failure_releases_display_and_restores_previous_handler(
    server, failure, exception, message
):
    lib, state, original = server
    bindings = [("toggle", 4, "space")]
    if failure == "repeat":
        state.autorepeat = False
    elif failure == "mapping":
        lib.XGetModifierMapping.return_value = None
    elif failure == "conflict":

        def conflict(*_):
            callback = x11_shortcuts.ErrorHandler(state.handler)
            assert callback(None, None) == 0

        lib.XSync.side_effect = conflict
    else:
        bindings = [("toggle", 4, failure)]

    with pytest.raises(exception, match=message):
        x11_shortcuts.X11Grabs(bindings)
    lib.XCloseDisplay.assert_called_once_with(11)
    assert state.handler == original
    if failure in ("repeat", "mapping"):
        lib.XFreeModifiermap.assert_not_called()


def test_handler_restoration_survives_display_close_error(server):
    lib, state, original = server
    native = x11_shortcuts.X11Grabs([])
    lib.XCloseDisplay.side_effect = OSError("Display disconnected")
    with pytest.raises(OSError, match="Display disconnected"):
        native.close()
    assert state.handler == original
    native.close()
    assert lib.XCloseDisplay.call_args_list == [call(11)]


def test_interrupted_handler_install_still_closes_display(server):
    lib, state, original = server
    lib.XSetErrorHandler.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        x11_shortcuts.X11Grabs([])
    lib.XCloseDisplay.assert_called_once_with(11)
    lib.XSetErrorHandler.assert_called_once()
    assert state.handler == original
