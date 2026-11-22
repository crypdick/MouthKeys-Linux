"""Recover from unconfirmed recorder shutdown without opening another capture."""

import io
import subprocess
import threading
import wave

import pytest

from mouthkeys.audio import CaptureSession, CaptureTeardownError
from mouthkeys.config import Config

from .pipeline_support import FakeRecorder, finish, make_controller, write_audio


def test_unknown_old_capture_teardown_does_not_open_replacement(tmp_path):
    opened = []
    may_stop = threading.Event()

    class Segment:
        def __init__(self, config, path, **kwargs):
            opened.append(config.audio_source)
            write_audio(path)

        def stop(self):
            if not may_stop.is_set():
                raise RuntimeError("Capture shutdown not confirmed")

        def finished(self):
            return may_stop.is_set()

    path = tmp_path / "recording.wav"
    session = CaptureSession(Config(audio_source="old"), path, recorder=Segment)
    try:
        with pytest.raises(CaptureTeardownError, match="Cannot finish microphone old") as failure:
            session.switch_source("healthy")
        assert failure.value.source == "old"
        assert opened == ["old"]
        assert session.capture is not None
        with wave.open(str(path), "rb") as wav:
            assert wav.getnframes() == 8000
    finally:
        may_stop.set()
        session.stop()
    assert session.capture is None
    assert session.closed


def test_service_does_not_quarantine_new_source_when_old_teardown_is_unknown(monkeypatch):
    class UnknownStop(FakeRecorder):
        def switch_source(self, source, *, timeout):
            raise CaptureTeardownError("old", RuntimeError("Cannot confirm microphone shutdown"))

    Config(audio_source="old").save()
    app = make_controller(UnknownStop)
    app.command("start")
    monkeypatch.setattr(
        "mouthkeys.microphone.source_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Must not open fallback")),
    )
    app.command("select_microphone", source="healthy")
    finish(app)
    assert "old" in app.microphone.quarantine
    assert "healthy" not in app.microphone.quarantine
    assert app.delivery.calls[0][2] is None


@pytest.mark.parametrize("retry_action", ["stop", "cancel", "toggle"])
def test_unconfirmed_recorder_shutdown_blocks_new_capture_until_retry(monkeypatch, retry_action):
    processes = []

    class Process:
        def __init__(self, args, **kwargs):
            self.stdin = io.BytesIO()
            self.pid = 999999
            self.confirmed = False
            write_audio(args[-1])
            processes.append(self)

        def poll(self):
            return 0 if self.confirmed else None

        def wait(self, timeout):
            if not self.confirmed:
                raise subprocess.TimeoutExpired("synthetic capture", timeout)
            return 0

    def cannot_confirm_kill(*args):
        raise PermissionError("Synthetic capture shutdown not confirmed")

    monkeypatch.setattr("mouthkeys.capture_recorder.subprocess.Popen", Process)
    monkeypatch.setattr("mouthkeys.capture_recorder.os.killpg", cannot_confirm_kill)
    Config(audio_source="old").save()
    app = make_controller(CaptureSession)
    app.command("start")
    item_id = app.item_id
    capture = app.recorder
    app.command("select_microphone", source="healthy")
    app.worker.join(2)
    assert not app.worker.is_alive()
    assert app.phase == "capture_failed"
    assert app.recorder is capture
    assert not app.job.done.is_set()
    assert len(processes) == 1
    assert app.job.audio_path.exists()
    assert app.history.rows()[0]["status"] == "recording"
    assert not app.transcriber.entered.is_set()
    assert not app.delivery.calls
    with pytest.raises(ValueError, match="capture_failed"):
        app.command("start")
    assert len(processes) == 1
    processes[0].confirmed = True
    app.command(retry_action)
    finish(app)
    assert app.item_id == item_id
    assert app.recorder is None
    assert app.job.done.is_set()
    assert not app.job.audio_path.exists()
    assert capture.closed
    if retry_action == "cancel":
        assert not app.delivery.calls
        assert app.history.rows()[0]["status"] == "cancelled"
    else:
        assert app.delivery.calls[0][2] is None
        assert app.history.rows()[0]["status"] == "copied"


def test_shutdown_discards_segment_even_when_capture_teardown_fails():
    class Segment:
        def __init__(self, config, path, **kwargs):
            write_audio(path)

        def stop(self):
            raise RuntimeError("Cannot confirm capture shutdown")

        def finished(self):
            return False

    app = make_controller(lambda config, path: CaptureSession(config, path, recorder=Segment))
    app.command("start")
    capture = app.recorder
    segment = capture.segment
    assert segment.exists()
    try:
        app.interrupt()
        assert not segment.exists()
        assert not app.job.audio_path.exists()
        assert app.history.rows()[0]["status"] == "failed"
    finally:
        # Release fake capture's writer and thread without exercising real devices.
        capture.capture = None
        capture.segment = None
        capture.stop()
