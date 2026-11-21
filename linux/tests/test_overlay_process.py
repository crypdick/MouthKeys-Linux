import fcntl
import socket
import stat
from types import SimpleNamespace

import pytest

from mouthkeys import overlay_process
from mouthkeys.config import runtime_dir


def test_panel_launcher_refuses_duplicate_process(monkeypatch):
    monkeypatch.setattr(
        overlay_process, "QApplication", lambda argv: pytest.fail("Started duplicate panel")
    )
    with (runtime_dir() / "overlay.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert overlay_process.launch() == 0


@pytest.mark.parametrize("aborted", [False, True])
def test_panel_wake_socket_polls_once_and_is_cleaned_up(monkeypatch, aborted):
    root = runtime_dir()
    path = root / "overlay-wake.sock"
    path.touch()  # Remnant from a terminated panel.
    settings = []
    polls = []
    activations = []
    intervals = []
    enabled = []
    panel = SimpleNamespace(
        poll=lambda: polls.append("poll"),
        timer=SimpleNamespace(setInterval=intervals.append),
    )
    read = overlay_process.QSocketNotifier.Type.Read

    def notifier(descriptor, mode):
        assert descriptor >= 0
        assert mode == read
        return SimpleNamespace(
            activated=SimpleNamespace(connect=activations.append),
            setEnabled=enabled.append,
        )

    def event_loop():
        assert path.is_socket()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
            client.settimeout(1)
            client.sendto(b"wake", str(path))
            client.sendto(b"wake", str(path))
        activations[0]()
        assert polls == ["poll", "poll"]
        if aborted:
            raise RuntimeError("fake Qt loop failure")
        return 17

    app = SimpleNamespace(
        setApplicationName=settings.append,
        setQuitOnLastWindowClosed=settings.append,
        exec=event_loop,
    )
    monkeypatch.setattr(overlay_process, "QApplication", lambda argv: app)
    monkeypatch.setattr(overlay_process, "Overlay", lambda: panel)
    notifier.Type = SimpleNamespace(Read=read)
    monkeypatch.setattr(overlay_process, "QSocketNotifier", notifier)
    if aborted:
        with pytest.raises(RuntimeError, match="fake Qt loop failure"):
            overlay_process.launch()
    else:
        assert overlay_process.launch() == 17
    assert settings == ["MouthKeys", False]
    assert enabled == [False]
    assert intervals == [200]
    assert not path.exists()
