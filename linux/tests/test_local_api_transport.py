"""Exercise shutdown and request-slot cleanup without external networking."""

import socket
from types import SimpleNamespace

import pytest

from mouthkeys import local_api
from mouthkeys.local_api import Handler, Server


@pytest.fixture
def server():
    with Server(("127.0.0.1", 0), Handler, bind_and_activate=False) as server:
        yield server


def test_read_deadline_ignores_connection_already_closed():
    handler = object.__new__(Handler)

    def shutdown(_direction):
        raise OSError("Closed")

    handler.connection = SimpleNamespace(shutdown=shutdown)
    handler.expire_read()


def test_server_closes_connection_when_all_request_slots_are_busy(server, monkeypatch):
    for _ in range(8):
        assert server.request_slots.acquire(blocking=False)
    closed = []
    monkeypatch.setattr(server, "shutdown_request", closed.append)
    try:
        with socket.socket() as request:
            server.process_request(request, ("127.0.0.1", 12))
            assert closed == [request]
    finally:
        for _ in range(8):
            server.request_slots.release()


def test_server_releases_slot_when_worker_does_not_start(server, monkeypatch):
    def fail(_server, _request, _address):
        raise RuntimeError("Cannot start worker")

    monkeypatch.setattr(local_api.ThreadingHTTPServer, "process_request", fail)
    with socket.socket() as request, pytest.raises(RuntimeError, match="Cannot start worker"):
        server.process_request(request, ("127.0.0.1", 12))
    for _ in range(8):
        assert server.request_slots.acquire(blocking=False)
    for _ in range(8):
        server.request_slots.release()


def test_client_disconnect_during_reply_does_not_retry(monkeypatch):
    handler = object.__new__(Handler)
    replies = []
    monkeypatch.setattr(handler, "_read_request", lambda: None)
    monkeypatch.setattr(handler, "_handle", lambda _request: {"status": "ok"})

    def answer(status, _body):
        replies.append(status)
        raise BrokenPipeError

    monkeypatch.setattr(handler, "answer", answer)
    handler.dispatch()
    assert replies == [200]
