import fcntl
import signal
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from mouthkeys import hotkeys, portal_shortcuts
from mouthkeys.config import Config, runtime_dir, state_dir


@pytest.fixture
def listener(monkeypatch):
    daemon = Mock()
    request = Mock(return_value={"phase": "recording", "id": "session"})
    grabs = Mock()
    popen = Mock()
    monkeypatch.setattr(hotkeys, "ensure_daemon", daemon)
    monkeypatch.setattr(hotkeys, "request", request)
    monkeypatch.setattr(hotkeys, "X11Grabs", grabs)
    monkeypatch.setattr(hotkeys, "subprocess", SimpleNamespace(Popen=popen, DEVNULL=-3))
    return SimpleNamespace(daemon=daemon, request=request, grabs=grabs, popen=popen)


def test_desktop_listener_cannot_be_started_explicitly(listener):
    with pytest.raises(ValueError, match="Desktop owns shortcuts"):
        hotkeys.launch()
    listener.daemon.assert_not_called()
    listener.grabs.assert_not_called()


@pytest.mark.parametrize("has_display", [False, True])
def test_native_listener_rejects_non_x11_sessions(monkeypatch, listener, has_display):
    Config(shortcut_backend="x11").save()
    if has_display:
        monkeypatch.setenv("DISPLAY", ":test")
        monkeypatch.setenv("WAYLAND_DISPLAY", "test")
    previous_handler = signal.getsignal(signal.SIGTERM)
    with pytest.raises(ValueError, match="require an X11 session"):
        hotkeys.launch()
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    listener.grabs.assert_not_called()
    hotkeys.ensure_hotkeys()
    listener.popen.assert_not_called()


def test_portal_requires_explicit_launch_and_does_not_register_x11_grabs(monkeypatch, listener):
    Config(shortcut_backend="portal").save()
    monkeypatch.setenv("DISPLAY", ":test")
    launch_portal = Mock(return_value=0)
    monkeypatch.setattr(portal_shortcuts, "launch", launch_portal)
    hotkeys.ensure_hotkeys()
    listener.popen.assert_not_called()
    previous_handler = signal.getsignal(signal.SIGTERM)
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(hotkeys.launch).result(timeout=3) == 0
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    listener.daemon.assert_called_once_with()
    launch_portal.assert_called_once_with()
    listener.grabs.assert_not_called()


def test_signal_handler_is_restored_after_listener_interruption(listener):
    Config(shortcut_backend="portal").save()
    previous_handler = signal.getsignal(signal.SIGTERM)

    def interrupt():
        handler = signal.getsignal(signal.SIGTERM)
        assert handler is not previous_handler
        handler(signal.SIGTERM, None)

    listener.daemon.side_effect = interrupt
    with pytest.raises(SystemExit) as result:
        hotkeys.launch()
    assert result.value.code == 0
    assert signal.getsignal(signal.SIGTERM) == previous_handler


def test_native_startup_respects_active_listener_lock_and_private_logs(monkeypatch, listener):
    Config(shortcut_backend="x11").save()
    monkeypatch.setenv("DISPLAY", ":test")
    lock_path = runtime_dir() / "hotkeys.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert hotkeys.launch() == 0
        hotkeys.ensure_hotkeys()
    listener.popen.assert_not_called()
    listener.daemon.assert_not_called()

    hotkeys.ensure_hotkeys()
    assert listener.popen.call_args.args[0][-3:] == ["-m", "mouthkeys", "hotkeys"]
    assert listener.popen.call_args.kwargs["start_new_session"]
    log = state_dir() / "hotkeys.log"
    assert listener.popen.call_args.kwargs["stdout"].name == str(log)
    assert log.stat().st_mode & 0o777 == 0o600


def test_native_reload_waits_for_release_and_reuses_ordered_dispatch(monkeypatch, listener):
    initial = Config(shortcut_backend="x11", hotkey_mode="toggle", toggle_shortcut="ctrl+space")
    changed = replace(initial, toggle_shortcut="alt+space")
    held = replace(changed, hotkey_mode="hold")
    desktop = replace(held, shortcut_backend="desktop")
    load = Mock(side_effect=[initial, initial, changed, held, desktop])
    monkeypatch.setattr(Config, "load", load)
    monkeypatch.setenv("DISPLAY", ":test")
    times = iter([2, 3, 3.5, 5, 7, 9, 11, 13])
    monkeypatch.setattr(hotkeys, "time", SimpleNamespace(monotonic=lambda: next(times)))

    def stream(backend):
        backend.down = {"key": "toggle"}
        assert not backend.close.called
        yield "toggle", True, 1.0, "key"
        backend.down.clear()
        yield "toggle", False, 2.0, "key"
        yield None
        yield None
        pytest.fail("configuration change did not retire the older grabbings")

    backends = [Mock() for _ in range(3)]
    for backend in backends:
        backend.events.side_effect = lambda backend=backend: stream(backend)
    listener.grabs.side_effect = backends
    assert hotkeys.launch() == 0
    assert listener.request.call_args_list == [
        call("toggle"),
        call("toggle"),
        call("start", held=True),
        call("stop", id="session"),
    ]
    registered = [grab.args[0] for grab in listener.grabs.call_args_list]
    assert [actions[0] for actions in registered] == [
        ("toggle", 4, "space"),
        ("toggle", 8, "space"),
        ("toggle", 8, "space"),
    ]
    for backend in backends:
        backend.close.assert_called_once_with()
    assert load.call_count == 5


def test_listener_failure_finishes_held_recording_and_closes_backend(monkeypatch, listener):
    Config(shortcut_backend="x11", hotkey_mode="hold").save()
    monkeypatch.setenv("DISPLAY", ":test")
    backend = listener.grabs.return_value
    backend.down = {"key": "toggle"}

    def stream():
        yield "cancel", True, 0.0, "other"
        yield "toggle", True, 1.0, "key"
        raise OSError("X11 connection lost")

    def request(command, **_options):
        if command == "cancel":
            raise OSError("Control socket unavailable for cancel")
        return {"phase": "recording", "id": "owned"}

    backend.events.side_effect = stream
    listener.request.side_effect = request
    with pytest.raises(OSError, match="X11 connection lost"):
        hotkeys.launch()
    backend.close.assert_called_once_with()
    assert listener.request.call_args_list == [
        call("cancel"),
        call("start", held=True),
        call("stop", automatic=True, id="owned"),
    ]


@pytest.mark.parametrize("failure", [OSError("Control socket lost"), None])
def test_failed_press_cannot_stop_the_previous_recording(failure):
    send = Mock(side_effect=[{"phase": "recording", "id": "old"}, None, failure])
    activation = hotkeys.Activation("automatic", send)
    activation.event("toggle", True, 1.0)
    activation.event("toggle", False, 1.1)
    if isinstance(failure, OSError):
        with pytest.raises(OSError, match="Control socket lost"):
            activation.event("toggle", True, 2.0)
    else:
        with pytest.raises(TypeError, match="Invalid recording response"):
            activation.event("toggle", True, 2.0)

    activation.event("toggle", False, 3.0)
    activation.finish()
    assert send.call_args_list == [
        call("toggle", held=True),
        call("release_hold", id="old"),
        call("toggle", held=True),
    ]


def test_release_before_press_is_ignored_for_non_recording_action():
    send = Mock()
    activation = hotkeys.Activation("hold", send)
    activation.event("paste", False, 0.0, "key")
    activation.event("paste", True, 1.0, "key")
    activation.event("paste", False, 2.0, "key")
    activation.finish()
    send.assert_called_once_with("paste")
    assert not activation.pressed


def test_cleanup_clears_pressed_state_when_daemon_stop_fails():
    send = Mock(side_effect=[{"phase": "recording", "id": "held"}, OSError("Control socket lost")])
    activation = hotkeys.Activation("hold", send)
    activation.event("toggle", True, 1.0)
    with pytest.raises(OSError, match="Control socket lost"):
        activation.finish()
    activation.finish()
    assert not activation.pressed
    assert send.call_args_list == [
        call("start", held=True),
        call("stop", automatic=True, id="held"),
    ]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "Invalid shortcut"),
        ("hyper+space", "Invalid shortcut"),
        ("ctrl+@", "Invalid shortcut key"),
        ("button1", "require a modifier"),
    ],
)
def test_invalid_chords_do_not_register(value, message):
    with pytest.raises(ValueError, match=message):
        hotkeys.parse_chord(value)


def test_plus_key_and_function_keys_have_native_names():
    assert hotkeys.parse_chord("ctrl++") == (4, "plus")
    assert hotkeys.parse_chord("F4") == (0, "F4")
