import ctypes.util
import io
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from mouthkeys import native_worker
from mouthkeys.config import Config


def test_kernel_network_isolation_preserves_local_ipc_in_child():
    if not ctypes.util.find_library("seccomp"):
        pytest.skip("Optional native worker requires libseccomp")
    script = """
import socket
from mouthkeys.native_worker import isolate_network
isolate_network()
for family in (socket.AF_INET, socket.AF_INET6):
    try:
        socket.socket(family)
    except PermissionError:
        pass
    else:
        raise AssertionError('Internet socket allowed')
left, right = socket.socketpair()
left.send(b'local')
assert right.recv(5) == b'local'
left.close()
right.close()
print('isolated')
"""
    result = subprocess.run(
        [sys.executable, "-P", "-c", script],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "isolated"


class Process:
    def __init__(self, responses):
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO()
        for response in responses:
            native_worker._write_message(self.stdout, response)
        self.stdout.seek(0)
        self.returncode = None
        self.killed = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout):
        return self.returncode


def test_model_worker_env_is_private_and_rpc_preserves_sessions(monkeypatch):
    process = Process([{"result": "ready"}, {"result": "partial"}, {"result": "final"}])
    calls = []
    previous = os.environ.get("HF_HUB_OFFLINE")
    monkeypatch.setattr(native_worker.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        native_worker.subprocess, "Popen", lambda *args, **kw: calls.append((args, kw)) or process
    )
    monkeypatch.setattr(
        native_worker.select, "select", lambda read, write, *args: (read, write, [])
    )
    config = replace(Config(), engine="nemo")
    backend = native_worker.NativeWorkerBackend(config)
    assert backend.stream([1, 2], config, "session") == "partial"
    assert backend.stream([1, 2, 3], config, "session", final=True) == "final"
    assert calls[0][1]["env"]["HF_HUB_OFFLINE"] == "1"
    assert calls[0][1]["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert calls[0][1]["stderr"] == subprocess.DEVNULL
    assert calls[0][1]["close_fds"] is True
    assert os.environ.get("HF_HUB_OFFLINE") == previous
    messages = io.BytesIO(process.stdin.getvalue())
    assert native_worker._read_message(messages)[0]["config"]["engine"] == "nemo"
    request, samples = native_worker._read_message(messages, allow_audio=True)
    assert request["method"] == "stream"
    assert request["session_key"] == "session"
    assert request["final"] is False
    assert list(samples) == [1, 2]
    directory = native_worker.Path(backend.directory.name)
    backend.close()
    assert process.killed
    assert not directory.exists()


def test_native_protocol_rejects_unexpected_audio_response_before_allocation():
    import struct

    stream = io.BytesIO()
    native_worker._write_message(stream, {"result": "fake"}, [1, 2])
    stream.seek(0)
    with pytest.raises(ValueError, match="Unexpected audio"):
        native_worker._read_message(stream)
    with pytest.raises(ValueError, match="message size"):
        native_worker._read_message(io.BytesIO(struct.pack("!I", native_worker.MAX_HEADER + 1)))


def test_native_timeout_kills_worker_not_truncated_success(monkeypatch):
    backend = native_worker.NativeWorkerBackend.__new__(native_worker.NativeWorkerBackend)
    backend.process = Process([])
    backend.timeout = 10
    monkeypatch.setattr(native_worker.select, "select", lambda *args: ([], [], []))
    with pytest.raises(RuntimeError, match="worker stopped"):
        backend._response()
    assert backend.process.killed
    assert not backend.alive()


def test_unexpected_worker_exit_is_actionable(monkeypatch):
    backend = native_worker.NativeWorkerBackend.__new__(native_worker.NativeWorkerBackend)
    backend.process = Process([])
    backend.timeout = 10
    monkeypatch.setattr(
        native_worker.select, "select", lambda read, write, *args: (read, write, [])
    )
    with pytest.raises(RuntimeError, match="audio can be reprocessed"):
        backend._response()
    assert backend.process.killed


def test_long_file_rpc_uses_bounded_speech_and_restores_timestamps(monkeypatch):
    import numpy as np

    from mouthkeys import native_asr
    from mouthkeys.config import Config
    from mouthkeys.native_worker import NativeWorkerBackend

    backend = object.__new__(NativeWorkerBackend)
    backend.timeout = 30
    calls = []
    monkeypatch.setattr("mouthkeys.native_worker.FILE_WINDOW_SAMPLES", 100)
    monkeypatch.setattr(
        native_asr,
        "speech_spans",
        lambda audio: [{"start": 10, "end": 30}, {"start": 100, "end": 130}],
    )

    def call(request, samples):
        calls.append((request, len(samples)))
        return [
            {
                "start": 0.0,
                "end": len(samples) / 16000,
                "text": "word",
                "words": [{"word": "word", "start": 0, "end": len(samples) / 16000}],
            }
        ]

    backend._call = call
    result = backend.segments(np.zeros(200), Config())
    assert [n for _, n in calls] == [20, 30]
    assert all(req["presegmented"] for req, _ in calls)
    assert [(s.start, s.end) for s in result] == [
        (10 / 16000, 30 / 16000),
        (100 / 16000, 130 / 16000),
    ]
    assert [(word["start"], word["end"]) for word in backend.words] == [
        (10 / 16000, 30 / 16000),
        (100 / 16000, 130 / 16000),
    ]


def test_unresponsive_worker_input_is_bounded_and_killed(monkeypatch):
    backend = object.__new__(native_worker.NativeWorkerBackend)
    backend.process = Process([])
    backend.timeout = 0.01
    backend.lock = native_worker.threading.Lock()
    monkeypatch.setattr(native_worker.select, "select", lambda *args: ([], [], []))
    with pytest.raises(RuntimeError, match="input timed out"):
        backend._call({"method": "segments"}, [0] * 100)
    assert backend.process.killed
