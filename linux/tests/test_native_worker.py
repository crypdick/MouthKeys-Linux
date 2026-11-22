import io
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mouthkeys import native_asr, native_protocol, native_worker
from mouthkeys.config import Config


class Process:
    def __init__(self, responses):
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO()
        for response in responses:
            native_protocol.write_message(self.stdout, response)
        self.stdout.seek(0)
        self.returncode = None
        self.killed = False
        self.waits = []

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout):
        self.waits.append(timeout)
        return self.returncode


def ready_worker(monkeypatch, *responses):
    process = Process([{"result": "ready", "native_streaming": False}, *responses])
    config = Config(engine="nemo", native_timeout_seconds=10)
    monkeypatch.setattr(native_worker.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(native_worker.subprocess, "Popen", lambda *_args, **_options: process)
    monkeypatch.setattr(native_protocol.select, "select", lambda read, write, *_: (read, write, []))
    return native_worker.NativeWorkerBackend(config), process, config


def test_model_worker_env_is_private_and_rpc_preserves_sessions(monkeypatch):
    process = Process([{"result": "ready"}, {"result": "partial"}, {"result": "final"}])
    calls = []
    previous = os.environ.get("HF_HUB_OFFLINE")
    monkeypatch.setattr(native_worker.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        native_worker.subprocess, "Popen", lambda *args, **kw: calls.append((args, kw)) or process
    )
    monkeypatch.setattr(
        native_protocol.select, "select", lambda read, write, *args: (read, write, [])
    )
    config = replace(Config(), engine="nemo")
    backend = native_worker.NativeWorkerBackend(config)
    assert backend.stream([1, 2], config, "session") == "partial"
    assert backend.stream([1, 2, 3], config, (Path("recording.wav"), 7), final=True) == "final"
    assert calls[0][1]["env"]["HF_HUB_OFFLINE"] == "1"
    assert calls[0][1]["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert calls[0][1]["stderr"] == subprocess.DEVNULL
    assert calls[0][1]["close_fds"] is True
    assert os.environ.get("HF_HUB_OFFLINE") == previous
    messages = io.BytesIO(process.stdin.getvalue())
    assert native_protocol.read_message(messages)[0]["config"]["engine"] == "nemo"
    request, samples = native_protocol.read_message(messages, allow_audio=True)
    assert request["method"] == "stream"
    assert request["session_key"] == "session"
    assert request["final"] is False
    assert list(samples) == [1, 2]
    final, samples = native_protocol.read_message(messages, allow_audio=True)
    assert final["session_key"] == ["recording.wav", 7]
    assert final["final"] is True
    assert list(samples) == [1, 2, 3]
    directory = Path(backend.directory.name)
    backend.close()
    assert process.killed
    assert not directory.exists()
    backend.close()
    assert process.waits == [3]


def test_startup_input_uses_timeout_and_reaps_stalled_worker(monkeypatch):
    process = Process([])
    monkeypatch.setattr(native_worker.importlib.util, "find_spec", lambda _: object())
    directory = None

    def spawn(*_args, **options):
        nonlocal directory
        directory = Path(options["env"]["MOUTHKEYS_NATIVE_STATE"])
        return process

    monkeypatch.setattr(native_worker.subprocess, "Popen", spawn)
    monkeypatch.setattr(native_protocol.select, "select", lambda *_: ([], [], []))
    with pytest.raises(RuntimeError, match="input timed out"):
        native_worker.NativeWorkerBackend(Config(engine="nemo", native_timeout_seconds=10))
    assert process.killed
    assert process.stdin.closed
    assert process.stdout.closed
    assert directory is not None
    assert not directory.exists()


def test_launch_error_cleans_temporary_directory_before_return(monkeypatch):
    monkeypatch.setattr(native_worker.importlib.util, "find_spec", lambda _: object())
    directory = None

    def spawn(*_args, **options):
        nonlocal directory
        directory = Path(options["env"]["MOUTHKEYS_NATIVE_STATE"])
        assert directory.is_dir()
        raise OSError("Cannot start worker")

    monkeypatch.setattr(native_worker.subprocess, "Popen", spawn)
    with pytest.raises(OSError, match="Cannot start worker") as failure:
        native_worker.NativeWorkerBackend(Config(engine="nemo"))
    assert failure.traceback
    assert directory is not None
    assert not directory.exists()


@pytest.mark.parametrize(
    "reply", [{"words": []}, {"result": 99}, {"error": {}}, {"result": "words", "words": False}]
)
def test_invalid_reply_reaps_worker(monkeypatch, reply):
    backend, process, config = ready_worker(monkeypatch, reply)
    try:
        with pytest.raises(RuntimeError, match="worker exited"):
            backend.stream([0], config, "test")
        assert not backend.alive()
        assert process.killed
    finally:
        backend.close()


def test_native_timeout_kills_worker_not_truncated_success(monkeypatch):
    backend, process, config = ready_worker(monkeypatch)
    monkeypatch.setattr(native_protocol.select, "select", lambda read, write, *_: ([], write, []))
    with pytest.raises(RuntimeError, match="exceeded 10s; worker stopped"):
        backend.stream([0], config, "test")
    assert process.killed
    assert not backend.alive()


def test_unexpected_worker_exit_is_actionable(monkeypatch):
    backend, process, config = ready_worker(monkeypatch)
    with pytest.raises(RuntimeError, match="record again"):
        backend.stream([0], config, "test")
    assert process.killed
    with pytest.raises(RuntimeError, match="reload model"):
        backend.stream([0], config, "test")


def test_long_file_rpc_uses_bounded_speech_and_restores_timestamps(monkeypatch):
    def reply(frames):
        return {
            "result": [
                {
                    "start": 0.0,
                    "end": frames / 16000,
                    "text": "word",
                    "words": [{"word": "word", "start": 0, "end": frames / 16000}],
                }
            ],
        }

    backend, process, config = ready_worker(monkeypatch, reply(20), reply(30))
    monkeypatch.setattr(native_worker, "FILE_WINDOW_SAMPLES", 100)
    monkeypatch.setattr(
        native_asr,
        "speech_spans",
        lambda audio: [{"start": 10, "end": 30}, {"start": 100, "end": 130}],
    )
    try:
        result = backend.segments(np.zeros(200), config)
        sent = io.BytesIO(process.stdin.getvalue())
        native_protocol.read_message(sent)
        requests = [native_protocol.read_message(sent, allow_audio=True) for _ in range(2)]
        assert [len(samples) for _, samples in requests] == [20, 30]
        assert all(req["presegmented"] for req, _ in requests)
        timestamps = [(10 / 16000, 30 / 16000), (100 / 16000, 130 / 16000)]
        assert [(s.start, s.end) for s in result] == timestamps
        assert [(word["start"], word["end"]) for word in backend.words] == timestamps
    finally:
        backend.close()


def test_unresponsive_worker_input_is_bounded_and_killed(monkeypatch):
    backend, process, config = ready_worker(monkeypatch)
    monkeypatch.setattr(native_protocol.select, "select", lambda *_: ([], [], []))
    with pytest.raises(RuntimeError, match="input timed out"):
        backend.segments([0] * 100, config)
    assert process.killed


def test_ready_handshake_reports_actual_native_streaming_capability(monkeypatch):
    backend, _, _ = ready_worker(monkeypatch)
    try:
        assert backend.native_streaming is False
    finally:
        backend.close()


def test_close_workers_continues_after_an_unreapable_worker(monkeypatch):
    reaped = []

    def stuck():
        raise subprocess.TimeoutExpired("worker", 3)

    monkeypatch.setattr(
        native_worker,
        "_WORKERS",
        [
            SimpleNamespace(close=stuck),
            SimpleNamespace(close=lambda: reaped.append(True)),
        ],
    )
    native_worker.close_workers()
    assert reaped == [True]


def test_worker_cannot_launch_without_optional_dependency(monkeypatch):
    monkeypatch.setattr(native_worker.importlib.util, "find_spec", lambda _: None)
    monkeypatch.setattr(
        native_worker.subprocess, "Popen", lambda *_args, **_kw: pytest.fail("spawn")
    )
    with pytest.raises(RuntimeError, match="Install the transformers optional dependency"):
        native_worker.NativeWorkerBackend(Config(engine="transformers"))


def test_invalid_ready_handshake_reaps_worker(monkeypatch):
    process = Process([{"result": "not ready"}])
    monkeypatch.setattr(native_worker.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(native_worker.subprocess, "Popen", lambda *_args, **_options: process)
    monkeypatch.setattr(native_protocol.select, "select", lambda read, write, *_: (read, write, []))
    with pytest.raises(RuntimeError, match="did not finish loading"):
        native_worker.NativeWorkerBackend(Config(engine="nemo"))
    assert process.killed
    assert process.waits == [3]


def test_worker_error_can_be_followed_by_reset_and_retry(monkeypatch):
    backend, process, config = ready_worker(
        monkeypatch, {"error": "Cannot decode"}, {"result": None}, {"result": "Recovered"}
    )
    try:
        with pytest.raises(RuntimeError, match="Cannot decode"):
            backend.stream([0], config, "retry")
        assert backend.alive()
        backend.reset_stream()
        assert backend.stream([0], config, "retry") == "Recovered"
        wire = io.BytesIO(process.stdin.getvalue())
        native_protocol.read_message(wire)
        native_protocol.read_message(wire, allow_audio=True)
        assert native_protocol.read_message(wire)[0] == {"method": "reset_stream"}
    finally:
        backend.close()
    backend.reset_stream()


@pytest.mark.parametrize("failure", ["missing", "broken", "exited"])
def test_unavailable_worker_is_not_reused(monkeypatch, failure):
    backend, process, config = ready_worker(monkeypatch)
    original_writer = process.stdin

    class BrokenPipe(io.BytesIO):
        def write(self, _data):
            raise BrokenPipeError("No reader")

    if failure == "exited":
        process.returncode = 1
    elif failure == "missing":
        process.stdin = None
    else:
        process.stdin = BrokenPipe()
    try:
        with pytest.raises(
            RuntimeError, match="reload model" if failure == "exited" else "worker exited"
        ):
            backend.stream([0], config, "test")
        assert not backend.alive()
    finally:
        backend.close()
        original_writer.close()
    assert process.killed == (failure != "exited")
    assert process.stdout.closed


@pytest.mark.parametrize("result", ["bad list", [{"text": "missing time"}], ["invalid segment"]])
def test_invalid_segment_reply_reaps_worker(monkeypatch, result):
    backend, process, config = ready_worker(monkeypatch, {"result": result})
    try:
        with pytest.raises(RuntimeError, match="worker exited"):
            backend.segments([0], config)
        assert process.killed
    finally:
        backend.close()
