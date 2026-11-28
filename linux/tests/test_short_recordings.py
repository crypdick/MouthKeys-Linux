"""Fast re-toggles and short silent takes should not reach speech recognition."""

import threading
import wave
from types import SimpleNamespace

import pytest

from mouthkeys import recording_capture
from mouthkeys.config import Config
from mouthkeys.hotkeys import Activation

from .pipeline_support import FakeRecorder, finish, make_controller, write_audio


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(recording_capture, "time", SimpleNamespace(monotonic=lambda: clock.now))
    return clock


@pytest.mark.parametrize(
    ("elapsed", "frames", "sample", "delivered"),
    [
        (0.125, 8000, 500, False),  # Even noisy audio is dropped after a fast re-toggle.
        (0.2, 8000, 500, True),  # The cutoff does not swallow a short spoken word.
        (0.25, 8000, 6, False),  # Ignore a little microphone noise.
        (0.25, 15999, 0, False),
        (0.25, 16000, 0, True),  # Default silence rule is strictly sub-second.
        (0.25, 800, 0, False),  # Suppress the old "too short" error on silence.
        (0.25, 0, 0, False),
    ],
)
def test_short_recordings_only_deliver_audible_intentional_takes(
    clock, monkeypatch, elapsed, frames, sample, delivered
):
    sounds = []
    monkeypatch.setattr(
        recording_capture, "play_sound", lambda _, *, stop=False: sounds.append(stop)
    )
    app = make_controller()
    app.transcriber.text = "Yes"
    item = app.command("toggle")["id"]
    path = app.job.audio_path
    write_audio(path, frames=frames, sample=sample)
    clock.now += elapsed
    app.command("toggle")
    finish(app)

    assert not path.exists()
    assert app.status()["notice"] is None
    assert app.status()["pending_jobs"] == 0
    assert app.transcriber.entered.is_set() is delivered
    assert bool(app.delivery.calls) is delivered
    assert sounds == ([False, True] if delivered else [False])
    if delivered:
        assert app.history.get(item)["text"] == "Yes"
    else:
        assert app.history.rows() == []
        assert app.status()["message"] == "Cancelled"


def test_explicit_hold_release_keeps_short_speech(clock):
    app = make_controller()
    app.command("start", held=True)
    write_audio(app.job.audio_path, sample=500)
    clock.now += 0.125
    app.command("stop")
    finish(app)
    assert app.delivery.calls


def test_fast_toggle_does_not_wait_for_focus(clock):
    Config(delivery="x11").save()
    app = make_controller()
    app.delivery.focus = lambda: pytest.fail("False start probed focus")
    app.command("toggle")
    clock.now += 0.125
    app.command("toggle")
    finish(app)
    assert not app.transcriber.entered.is_set()


@pytest.mark.parametrize("shutdown", [False, True])
def test_cancelled_microphone_start_leaves_no_history(shutdown):
    entered = threading.Event()
    release = threading.Event()

    def open_capture(config, path):
        entered.set()
        assert release.wait(3)
        return FakeRecorder(config, path)

    app = make_controller(open_capture)
    start = threading.Thread(target=lambda: app.command("toggle"))
    start.start()
    try:
        assert entered.wait(2)
        app.command("toggle")
        if shutdown:
            app.interrupt()
    finally:
        release.set()
        start.join(3)
    assert not start.is_alive()
    if shutdown:
        assert app.status()["phase"] == "interrupted"
    else:
        assert app.status()["phase"] == "idle"
        assert app.status()["message"] == "Cancelled"
    assert app.history.rows() == []
    assert not app.job.audio_path.exists()
    assert not app.transcriber.entered.is_set()


@pytest.mark.parametrize(
    ("elapsed", "sample", "previewed"), [(0.125, 500, False), (0.25, 0, False), (0.25, 500, True)]
)
def test_short_preview_waits_past_toggle_window_and_requires_sound(
    clock, elapsed, sample, previewed
):
    Config(preview_interval_ms=100).save()
    app = make_controller()
    app.transcriber.preview = lambda *_: "Yes"
    app.command("start")
    write_audio(app.job.audio_path, sample=sample)
    app.live_preview.preview_at = 0
    clock.now += elapsed
    app.tick()
    try:
        worker = app.live_preview.worker
        if worker is not None:
            worker.join(2)
            assert not worker.is_alive()
        assert (app.status()["preview"] == "Yes") is previewed
    finally:
        app.command("cancel")
        finish(app)


def test_optional_silence_filter_still_skips_longer_recordings():
    Config(skip_silent_recordings="on").save()
    app = make_controller()
    app.command("start")
    write_audio(app.job.audio_path, frames=16000, sample=0)
    app.command("stop")
    finish(app)
    assert app.message == "Silent recording skipped"
    assert not app.transcriber.entered.is_set()
    assert not app.delivery.calls


def test_short_silent_take_is_discarded_without_waiting_for_an_earlier_job():
    Config(save_history="off").save()
    app = make_controller()
    app.delivery.deliver = lambda *_: "copied"
    app.transcriber.release.clear()
    app.command("start")
    app.command("stop")
    prior = app.worker
    try:
        assert app.transcriber.entered.wait(2)
        app.command("start")
        write_audio(app.job.audio_path, sample=0)
        app.command("stop")
        finish(app)
        assert prior.is_alive()
        assert not app.job.audio_path.exists()
        assert app.status()["message"] == "Cancelled"
        assert app.history.rows() == []
    finally:
        app.transcriber.release.set()
        prior.join(3)
    assert not prior.is_alive()


@pytest.mark.parametrize("mode", ["toggle", "automatic"])
def test_fast_native_retoggle_uses_event_time_through_slow_start(clock, mode):
    def slow_open(config, path):
        clock.now += 0.4
        return FakeRecorder(config, path)

    app = make_controller(slow_open)
    shortcut = Activation(mode, app.command)
    shortcut.event("toggle", True, 10.0)
    shortcut.event("toggle", False, 10.06)
    shortcut.event("toggle", True, 10.12)
    finish(app)

    assert app.history.rows() == []
    assert app.status()["message"] == "Cancelled"
    assert not app.transcriber.entered.is_set()


@pytest.mark.parametrize("sample", [0, 500])
def test_slow_focus_probe_does_not_extend_a_short_take(clock, sample):
    Config(delivery="x11").save()
    app = make_controller()
    app.command("start")
    write_audio(app.job.audio_path, sample=sample)
    clock.now += 0.5

    def focus():
        clock.now += 0.6
        write_audio(app.job.audio_path, frames=17600, sample=sample)
        return "123"

    app.delivery.focus = focus
    app.command("stop")
    finish(app)
    assert app.transcriber.entered.is_set() is bool(sample)
    assert bool(app.history.rows()) is bool(sample)


@pytest.mark.parametrize("sample", [0, 500])
def test_short_take_checks_meter_windows_not_spikes_or_only_the_tail(clock, sample):
    app = make_controller()
    app.command("start")
    with wave.open(str(app.job.audio_path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        # An audible burst near the start must survive a quiet tail. A tiny spike must not.
        wav.writeframes(sample.to_bytes(2, "little", signed=True) * 3200)
        wav.writeframes(bytes(4799 * 2) + b"\x64\0")
    clock.now += 0.5
    app.command("stop")
    finish(app)
    assert app.transcriber.entered.is_set() is bool(sample)
    assert bool(app.history.rows()) is bool(sample)
