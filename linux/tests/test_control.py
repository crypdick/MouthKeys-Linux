import fcntl
import io
import json
import os
import socket
import stat
import subprocess
from contextlib import contextmanager
from pathlib import Path
from subprocess import Popen
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mouthkeys import control
from mouthkeys.config import Config, runtime_dir, state_dir


@pytest.mark.parametrize(
    ("response", "failure"),
    [
        (b'{"ok":false,"error":"Busy"}\n', "Busy"),
        (b'{"ok":false,"error":7}\n', "Invalid daemon response"),
        (b'{"ok":true}\n', "Invalid daemon response"),
        (b"[]\n", "Invalid daemon response"),
        (b"x" * 257, "Response too large"),
    ],
)
def test_control_exchange_uses_private_socket_and_validates_response(
    monkeypatch, response, failure
):
    path = str(runtime_dir() / "control.sock")
    messages = []

    @contextmanager
    def open_client(family, kind):
        assert (family, kind) == (socket.AF_UNIX, socket.SOCK_STREAM)
        yield SimpleNamespace(
            settimeout=messages.append,
            connect=messages.append,
            sendall=messages.append,
            makefile=lambda mode: io.BytesIO(response),
        )

    monkeypatch.setattr(control, "MAX_MESSAGE", 256)
    monkeypatch.setattr(socket, "socket", open_client)
    with pytest.raises(RuntimeError, match=failure):
        control.request("select_microphone", source="test input")
    assert messages == [20, path, b'{"action": "select_microphone", "source": "test input"}\n']


@pytest.fixture
def launches(monkeypatch):
    launched = []
    child = Mock(spec=Popen)

    def launch(command, **options):
        assert command[1:4] == ["-P", "-m", "mouthkeys"]
        assert options["stdin"] == subprocess.DEVNULL
        assert options["start_new_session"]
        output = options["stdout"]
        assert output is options["stderr"]
        path = Path(output.name)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        launched.append((command[-1], path))
        return child

    monkeypatch.setattr(subprocess, "Popen", launch)
    return launched


@pytest.mark.parametrize("ready", [False, True])
def test_daemon_retry_is_bounded_and_starts_only_one_child(monkeypatch, launches, ready):
    attempts = []
    clock = iter([10, 10, 10.05, 15])
    sleeps = []
    monkeypatch.setattr(
        control, "time", SimpleNamespace(monotonic=lambda: next(clock), sleep=sleeps.append)
    )

    def status(action):
        attempts.append(action)
        if ready and len(attempts) == 3:
            return {"phase": "idle"}
        raise ConnectionRefusedError

    monkeypatch.setattr(control, "request", status)
    if ready:
        control.ensure_daemon()
    else:
        with pytest.raises(RuntimeError, match="Daemon did not start"):
            control.ensure_daemon()
    assert attempts == ["status"] * 3
    assert sleeps == [0.05] * (1 if ready else 2)
    assert launches == [("daemon", state_dir() / "daemon.log")]


def test_ready_daemon_needs_no_child_process(monkeypatch, launches):
    monkeypatch.setattr(control, "request", lambda action: {"phase": "idle"})
    control.ensure_daemon()
    assert not launches


@pytest.mark.parametrize(
    ("display", "enabled", "qt"), [(False, "on", True), (True, "off", True), (True, "on", False)]
)
def test_daemon_skips_tray_without_desktop_support(monkeypatch, launches, display, enabled, qt):
    if display:
        monkeypatch.setenv("WAYLAND_DISPLAY", "fake-wayland")
    Config(tray=enabled).save()
    monkeypatch.setattr(control.importlib.util, "find_spec", lambda _: object() if qt else None)
    assert control.launch_tray() is None
    assert launches == []


@pytest.mark.parametrize("available", [True, False])
def test_daemon_launch_passes_private_tray_lifeline(monkeypatch, caplog, available):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(control.importlib.util, "find_spec", lambda _: object())
    child = Mock(spec=Popen)
    inherited = []

    def launch(command, **options):
        (fd,) = options["pass_fds"]
        assert command[4:] == ["tray", "--service-fd", str(fd)]
        if not available:
            raise OSError("could not spawn")
        inherited.append(socket.socket(fileno=os.dup(fd)))
        return child

    monkeypatch.setattr(subprocess, "Popen", launch)
    result = control.launch_tray()
    if not available:
        assert result is None
        assert "could not spawn" in caplog.text
        return
    service, process = result
    (tray,) = inherited
    with service, tray:
        assert process is child
        service.settimeout(1)
        tray.sendall(b"1")
        assert service.recv(1) == b"1"
        tray.close()
        assert service.recv(1) == b""


def test_enabled_panel_starts_with_gui_support_but_not_for_an_existing_panel(monkeypatch, launches):
    monkeypatch.setenv("WAYLAND_DISPLAY", "fake-wayland")
    monkeypatch.setattr(control.importlib.util, "find_spec", lambda name: None)
    control.ensure_overlay()
    assert launches == []
    monkeypatch.setattr(control.importlib.util, "find_spec", lambda name: object())
    Config(recording_panel="off").save()
    control.ensure_overlay()
    assert launches == []
    Config(recording_panel="on").save()
    with (runtime_dir() / "overlay.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Socket not yet created by the running overlay; don't start a second.
        assert not (runtime_dir() / "overlay-wake.sock").exists()
        control.ensure_overlay()
        assert launches == []
    control.ensure_overlay()
    assert launches == [("overlay", state_dir() / "overlay.log")]


def test_control_client_import_does_not_load_capture_or_gui(tmp_path):
    command = control.python_command()[:2]
    result = subprocess.run(  # noqa: S603
        [
            *command,
            "-c",
            (
                "import sys; import mouthkeys.control; print(sorted(name for name in sys.modules "
                "if name.startswith(('mouthkeys.service', 'mouthkeys.audio', 'PySide6'))))"
            ),
        ],
        cwd=tmp_path,
        timeout=3,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == []
