import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mouthkeys import recording_destination, recording_preview
from mouthkeys.config import Config
from mouthkeys.recording_preview import PreviewWork, RecordingPreview, reset_owned_stream

from .pipeline_support import SwitchRecorder, finish, make_controller


def test_monitor_waits_for_pause_and_requires_a_new_phrase_after_speech_resumes(
    monkeypatch, tmp_path
):
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(recording_preview, "time", SimpleNamespace(monotonic=lambda: clock.now))
    reading = {"frame": 12, "level": 0.8}
    original_meter = recording_preview.live_meter
    monkeypatch.setattr(recording_preview, "live_meter", lambda _: reading)
    path = tmp_path / "capture.wav"
    config = Config(live_preview="off", spoken_send="on", send_after_pause="on")
    monitor = RecordingPreview(threading.RLock(), threading.Lock())
    assert not monitor.due(config)
    assert not monitor.should_stop(config, held=False)
    published = []
    heard = SimpleNamespace(text="First send it")
    transcriber = SimpleNamespace(preview=lambda *_: heard.text)
    work = PreviewWork(path, config, transcriber, (), lambda: True, lambda: False, published.append)

    def decode():
        monitor.schedule(work)
        monitor.worker.join(2)
        assert not monitor.worker.is_alive()
        assert published[-1] == heard.text

    clock.now = 13.0
    assert monitor.due(config)
    decode()
    assert monitor.armed_at == 13.0
    clock.now = 13.125
    decode()
    assert monitor.armed_at == 13.0
    assert not monitor.should_stop(config, held=False)

    monitor.sample(path, heard.text)
    assert monitor.meter == {**reading, "at": 13.125}
    clock.now = 13.5
    monitor.sample(path, heard.text)
    assert monitor.meter["at"] == monitor.last_voice_at == 13.125
    assert not monitor.should_stop(config, held=False)  # Voice is still too recent.
    reading["frame"] = 13
    monitor.sample(path, heard.text)
    assert monitor.armed_at is None
    decode()  # A new decode of the old phrase must not rearm delivery.
    assert monitor.armed_at is None

    heard.text = "Second send it"
    decode()
    assert monitor.armed_at == 13.5
    reading.update(frame=14, level=0)
    clock.now = 14.0
    monitor.sample(path, heard.text)
    assert monitor.last_voice_at == 13.5
    assert monitor.should_stop(config, held=False)
    assert not monitor.should_stop(config, held=True)
    assert not monitor.should_stop(replace(config, spoken_send="off"), held=False)
    assert not monitor.should_stop(replace(config, send_after_pause="off"), held=False)
    assert not monitor.due(replace(config, send_after_pause="off"))

    heard.text = "No suffix here"
    decode()
    assert not monitor.should_stop(config, held=False)
    monkeypatch.setattr(recording_preview, "live_meter", original_meter)
    monitor.sample(path, heard.text)  # The path can disappear before the next meter read.
    assert monitor.meter == {}


def test_superseded_work_never_enters_decoder(tmp_path):
    lock = threading.Lock()
    monitor = RecordingPreview(threading.RLock(), lock)
    path = tmp_path / "capture.wav"
    decoded = []
    transcriber = SimpleNamespace(preview=lambda *_: decoded.append(True))
    work = PreviewWork(path, Config(), transcriber, (), lambda: True, lambda: False, lambda _: None)
    with lock:
        monitor.schedule(work)
        monitor.suspend("")
    monitor.worker.join(2)
    assert not monitor.worker.is_alive()
    assert decoded == []


def test_native_stream_cleanup_requires_path_ownership(tmp_path):
    audio = tmp_path / "current.wav"
    resets = []
    transcriber = SimpleNamespace(stream_key=(audio,), reset_stream=lambda: resets.append(audio))
    reset_owned_stream(transcriber, tmp_path / "older.wav")
    assert resets == []
    reset_owned_stream(transcriber, audio)
    assert resets == [audio]
    del transcriber.reset_stream
    reset_owned_stream(transcriber, audio)


def test_preview_started_before_microphone_switch_cannot_publish_afterward():
    Config(audio_source="original", spoken_send="on", send_after_pause="on").save()
    app = make_controller(SwitchRecorder)
    entered, release = threading.Event(), threading.Event()

    def preview(*_):
        entered.set()
        assert release.wait(3)
        return "Stale words send it"

    app.transcriber.preview = preview
    app.command("start")
    app.capture.activated_at -= 1
    app.live_preview.preview_at = 0
    app.tick()
    try:
        assert entered.wait(2)
        worker = app.live_preview.worker
        app.tick()
        assert app.live_preview.worker is worker
        status = app.command("select_microphone", source="usb")
        assert status["phase"] == "recording"
        assert status["microphone"] == "usb"
        release.set()
        app.live_preview.worker.join(2)
        assert not app.live_preview.worker.is_alive()
        assert app.status()["preview"] == ""
        assert app.live_preview.armed_at is None
        assert app.live_preview.preview_at == 0

        app.transcriber.preview = lambda *_: "Fresh words send it"
        app.tick()
        app.live_preview.worker.join(2)
        assert not app.live_preview.worker.is_alive()
        assert app.status()["preview"] == "Fresh words send it"
        assert app.live_preview.armed_at is not None
    finally:
        release.set()
        app.live_preview.worker.join(2)
        app.command("cancel")
        finish(app)


@pytest.mark.parametrize(("spoken", "expired"), [(True, False), (True, True), (False, False)])
def test_automatic_stop_keeps_daemon_available_during_focus_lookup(monkeypatch, spoken, expired):
    Config(
        delivery="x11",
        spoken_send="on",
        send_after_pause="on" if spoken else "off",
    ).save()
    app = make_controller()
    app.command("start")
    app.capture.activated_at -= 1
    app.live_preview.preview_at = 0  # This transcriber has no preview adapter.
    app.tick()  # Prime the meter before simulating a pause with no new frames.
    app.live_preview.armed_at = app.live_preview.last_voice_at = time.monotonic() - 1
    app.recorder.ended = not spoken
    entered, release = threading.Event(), threading.Event()
    destinations = []
    clock = SimpleNamespace(offset=0.0)
    monkeypatch.setattr(
        recording_destination,
        "time",
        SimpleNamespace(monotonic=lambda: time.monotonic() + clock.offset),
    )

    def focus():
        entered.set()
        assert release.wait(3)
        return "focused"

    app.delivery.focus = focus
    app.delivery.deliver = lambda _text, _config, target: destinations.append(target) or "copied"
    tick = threading.Thread(target=app.tick, daemon=True)
    tick.start()
    try:
        if spoken:
            assert entered.wait(2)
            tick.join(1)
            assert not tick.is_alive()  # service_actions can accept another command.
            assert app.status()["phase"] == "stopping"
            if expired:
                clock.offset = recording_destination.FOCUS_TIMEOUT + 1
    finally:
        release.set()
        tick.join(3)
    assert not tick.is_alive()
    finish(app)
    assert entered.is_set() == spoken
    assert destinations == ["focused" if spoken and not expired else None]
