import io
import json
import signal
import socket
import threading
from types import SimpleNamespace

import pytest

from mouthkeys import daemon
from mouthkeys.config import runtime_dir
from mouthkeys.control import request


class FakeCommands:
    def __init__(self):
        self.interrupted = False
        self.errors = []

    def command(self, action, **data):
        if action == "toggle":
            raise ValueError("Busy")
        return {"action": action, "data": data}

    def status(self):
        return {"interrupted": self.interrupted}

    def interrupt(self):
        self.interrupted = True

    def tick(self):
        pass

    def report_error(self, message):
        self.errors.append(message)


@pytest.fixture
def daemon_socket():
    controller = FakeCommands()
    path = str(runtime_dir() / "control.sock")
    with daemon.Server(path, controller) as server:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        try:
            yield path, controller
        finally:
            if thread.is_alive():
                server.schedule_shutdown()
                thread.join(2)
            assert not thread.is_alive()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"x" * 100 + b"\n", "Request too large or incomplete"),
        (b'{"action":"status"}', "Request too large or incomplete"),
        (b"[]\n", "Request must be a JSON object"),
        (b'{"action":[]}\n', "Request must name an action"),
    ],
)
def test_control_socket_refuses_incomplete_or_invalid_command(
    daemon_socket, monkeypatch, payload, message
):
    path, controller = daemon_socket
    monkeypatch.setattr(daemon, "MAX_MESSAGE", 64)
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(2)
        client.connect(path)
        client.sendall(payload)
        client.shutdown(socket.SHUT_WR)
        with client.makefile("rb") as stream:
            assert json.loads(stream.readline(128)) == {"ok": False, "error": message}
    assert controller.errors == []
    assert request("history", limit=10) == {"action": "history", "data": {"limit": 10}}


def test_daemon_shutdown_reply_observes_interrupted_controller(daemon_socket):
    _, controller = daemon_socket
    assert request("shutdown") == {"interrupted": True}
    assert controller.interrupted


@pytest.mark.parametrize("client_gone", [False, True])
def test_shutdown_acknowledges_before_stopping_socket(client_gone):
    events = []
    controller = FakeCommands()

    def write(line):
        assert json.loads(line) == {"ok": True, "result": {"interrupted": True}}
        events.append("reply")
        if client_gone:
            raise BrokenPipeError

    handler = SimpleNamespace(
        connection=SimpleNamespace(settimeout=lambda _: None),
        rfile=io.BytesIO(b'{"action":"shutdown"}\n'),
        wfile=SimpleNamespace(write=write),
        server=SimpleNamespace(
            controller=controller,
            schedule_shutdown=lambda: events.append("shutdown"),
        ),
    )
    daemon.Handler.handle(handler)
    assert events == ["reply", "shutdown"]


@pytest.mark.parametrize("api_available", [False, True])
def test_daemon_binds_private_socket_reaps_stale_files_and_closes_api(monkeypatch, api_available):
    root = runtime_dir()
    control_socket = root / "control.sock"
    control_socket.touch()
    recordings = root / "private-recordings"
    recordings.mkdir()
    stale = recordings / "stale.wav"
    stale.touch()
    controller = FakeCommands()
    actions = []
    handlers = {}
    monkeypatch.setattr(daemon.os, "umask", actions.append)
    monkeypatch.setattr(signal, "signal", lambda sig, handler: handlers.update({sig: handler}))

    def start(instance):
        assert instance is controller
        if not api_available:
            raise RuntimeError("unsupported configuration")
        return SimpleNamespace(
            shutdown=lambda: actions.append("stop API"),
            server_close=lambda: actions.append("close API"),
        )

    def serve_forever(server, *, poll_interval):
        assert control_socket.is_socket()
        assert not stale.exists()
        assert poll_interval == 0.025
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        assert controller.interrupted

    monkeypatch.setattr(daemon, "start_local_api", start)
    monkeypatch.setattr(daemon.Server, "serve_forever", serve_forever)
    monkeypatch.setattr(daemon.Server, "schedule_shutdown", lambda self: actions.append("stop IPC"))
    daemon.serve(lambda: controller)
    assert not control_socket.exists()
    assert set(handlers) == {signal.SIGTERM, signal.SIGINT}
    if api_available:
        assert actions == [0o077, "stop IPC", "stop API", "close API"]
        assert controller.errors == []
    else:
        assert actions == [0o077, "stop IPC"]
        assert controller.errors == ["Local API unavailable: unsupported configuration"]
