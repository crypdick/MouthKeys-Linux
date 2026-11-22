"""Headless tests of PCM relay ownership, cancellation, and file cleanup."""

import threading
import wave
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from mouthkeys import audio
from mouthkeys.audio import CaptureSession, CaptureTeardownError
from mouthkeys.config import Config

from .pipeline_support import write_audio


def open_capture(tmp_path, *, cancel_event=None):
    opened = []
    devices = []

    def factory(config, path, **_options):
        opened.append(config.audio_source)
        write_audio(path)
        device = SimpleNamespace(path=path, stop=lambda: None, finished=lambda: False)
        devices.append(device)
        return device

    session = CaptureSession(
        Config(audio_source="old"),
        tmp_path / "audio.wav",
        recorder=factory,
        cancel_event=cancel_event,
    )
    return session, opened, devices


@pytest.mark.parametrize("timing", ["before", "during"])
def test_switch_cancel_preserves_pcm_without_opening_replacement(tmp_path, timing):
    cancelled = threading.Event()
    session, opened, devices = open_capture(tmp_path, cancel_event=cancelled)
    if timing == "before":
        cancelled.set()
    else:
        devices[0].stop = cancelled.set
    try:
        with pytest.raises(RuntimeError, match="Recording is stopping"):
            session.switch_source("replacement")
        assert opened == ["old"]
    finally:
        session.stop()
    with wave.open(str(session.path), "rb") as recording:
        assert recording.getnframes() == 8000


def test_failed_stop_with_unreadable_status_keeps_ownership(tmp_path):
    session, opened, devices = open_capture(tmp_path)

    def stop():
        raise RuntimeError("Unconfirmed")

    def finished():
        raise OSError("Device state unavailable")

    devices[0].stop = stop
    devices[0].finished = finished
    try:
        with pytest.raises(CaptureTeardownError, match="Cannot finish microphone old"):
            session.switch_source("new")
        assert opened == ["old"]
        assert session.capture is devices[0]
    finally:
        devices[0].stop = lambda: None
        session.stop()


def test_bad_final_segment_blocks_switch_but_preserves_committed_audio(tmp_path):
    session, opened, devices = open_capture(tmp_path)

    def stop():
        with wave.open(str(devices[0].path), "wb") as part:
            part.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
            part.writeframes(bytes(640))

    devices[0].stop = stop
    try:
        with pytest.raises(CaptureTeardownError, match="Expected mono 16 kHz PCM"):
            session.switch_source("new")
        assert opened == ["old"]
        assert not devices[0].path.exists()
    finally:
        session.stop()
    with wave.open(str(session.path), "rb") as recording:
        assert recording.getnframes() == 8000


def test_stop_reports_capture_error_after_draining_final_frames(tmp_path):
    session, _, devices = open_capture(tmp_path)

    def stop():
        with wave.open(str(devices[0].path), "wb") as part:
            part.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            part.writeframes(bytes(32000))
        raise RuntimeError("Input unplugged")

    devices[0].stop = stop
    devices[0].finished = lambda: True
    with pytest.raises(RuntimeError, match="Input unplugged"):
        session.stop()
    assert session.closed
    assert not devices[0].path.exists()
    with wave.open(str(session.path), "rb") as recording:
        assert recording.getnframes() == 16000


def test_failed_initial_drain_stops_capture_and_cleans_private_segment(tmp_path):
    stopped = []

    def factory(_config, path, **_options):
        return SimpleNamespace(stop=lambda: stopped.append(path))

    canonical = tmp_path / "audio.wav"
    with pytest.raises(EOFError):
        CaptureSession(Config(), canonical, recorder=factory)
    assert len(stopped) == 1
    assert not stopped[0].exists()
    with wave.open(str(canonical), "rb") as recording:
        assert recording.getnframes() == 0


def test_relay_retries_short_reads_without_advancing_pcm_cursor(monkeypatch, tmp_path):
    opens = []
    read = audio._capture_wav

    def no_bytes(frames):
        opens.append(frames)
        return b""

    @contextmanager
    def transient_read(path):
        if not opens:
            yield SimpleNamespace(setpos=lambda _: None, readframes=no_bytes), 8000
        else:
            with read(path) as original:
                yield original

    monkeypatch.setattr(audio, "_capture_wav", transient_read)
    session, _, _ = open_capture(tmp_path)
    session.stop()
    assert opens == [8000]
    with wave.open(str(session.path), "rb") as recording:
        assert recording.getnframes() == 8000


def test_relay_failure_reaches_stop_without_losing_existing_audio(monkeypatch, tmp_path):
    session, _, _ = open_capture(tmp_path)
    drain = session._drain

    def transient_failure():
        if threading.current_thread() is session.thread:
            raise OSError("Transient private file error")
        drain()

    monkeypatch.setattr(session, "_drain", transient_failure)
    try:
        assert session.stopping.wait(2)
        assert session.finished()
    finally:
        with pytest.raises(RuntimeError, match="Capture relay failed: Transient"):
            session.stop()
    with wave.open(str(session.path), "rb") as recording:
        assert recording.getnframes() == 8000


def test_relay_start_failure_closes_recording_and_stops_device(monkeypatch, tmp_path):
    stopped = []

    def factory(_config, path, **_options):
        write_audio(path)
        return SimpleNamespace(stop=lambda: stopped.append(path))

    def no_start():
        raise RuntimeError("Thread could not start")

    monkeypatch.setattr(
        audio,
        "threading",
        SimpleNamespace(
            Event=threading.Event,
            RLock=threading.RLock,
            Thread=lambda **_options: SimpleNamespace(start=no_start),
        ),
    )
    canonical = tmp_path / "audio.wav"
    with pytest.raises(RuntimeError, match="Thread could not start"):
        CaptureSession(Config(), canonical, recorder=factory)
    assert len(stopped) == 1
    assert not stopped[0].exists()
    with wave.open(str(canonical), "rb") as recording:
        assert recording.getnframes() == 8000


def test_stop_reports_relay_that_does_not_join_within_budget(monkeypatch, tmp_path):
    waits = []

    def join(*, timeout):
        waits.append(timeout)

    thread = SimpleNamespace(start=lambda: None, join=join, is_alive=lambda: True)
    monkeypatch.setattr(
        audio,
        "threading",
        SimpleNamespace(
            Event=threading.Event, RLock=threading.RLock, Thread=lambda **_options: thread
        ),
    )
    session, _, devices = open_capture(tmp_path)
    with pytest.raises(RuntimeError, match="Capture relay did not stop"):
        session.stop()
    assert waits == [1]
    assert session.closed
    assert not devices[0].path.exists()
