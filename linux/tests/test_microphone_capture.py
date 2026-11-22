import threading
import wave

import pytest

from mouthkeys.audio import CaptureSession
from mouthkeys.config import Config

from .pipeline_support import FakeRecorder, SwitchRecorder, finish, make_controller


def test_capture_failure_tries_alternate_and_quarantines_failed_source(monkeypatch):
    Config(audio_source="usb").save()
    calls = []

    def record(config, path):
        calls.append(config.audio_source)
        if config.audio_source == "usb":
            raise RuntimeError("USB readiness timed out")
        return FakeRecorder(config, path)

    monkeypatch.setattr(
        "mouthkeys.microphone.source_candidates",
        lambda config, excluded: [name for name in ("usb", "backup") if name not in excluded],
    )
    app = make_controller(record)
    app.command("start")
    assert calls == ["usb", "backup"]
    assert app.status()["microphone"] == "backup"
    assert Config.load().audio_source == "usb"
    assert app.job.config.audio_source == "backup"
    assert "fallback microphone" in app.message
    app.command("stop")
    finish(app)
    app.command("start")
    assert calls == ["usb", "backup", "backup"]
    app.command("cancel")
    finish(app)


def test_startup_status_and_cancel_do_not_wait_for_capture(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def record(config, path):
        entered.set()
        assert release.wait(2)
        return FakeRecorder(config, path)

    app = make_controller(record)
    result = []
    worker = threading.Thread(target=lambda: result.append(app.command("start")))
    worker.start()
    try:
        assert entered.wait(1)
        assert app.command("status")["phase"] == "starting"
        assert app.command("cancel")["phase"] == "starting"
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert result[0]["phase"] == "idle"
    assert app.history.rows()[0]["status"] == "cancelled"
    assert app.delivery.calls == []


def test_failed_sources_have_total_time_budget(monkeypatch):
    Config(audio_source="one").save()
    elapsed = [0.0]
    calls = []

    def record(config, path):
        calls.append(config.audio_source)
        elapsed[0] += 5
        raise RuntimeError("not ready")

    monkeypatch.setattr("mouthkeys.recording_capture.time.monotonic", lambda: elapsed[0])
    monkeypatch.setattr(
        "mouthkeys.microphone.source_candidates",
        lambda config, excluded: [name for name in ("one", "two", "three") if name not in excluded],
    )
    app = make_controller(record)
    with pytest.raises(RuntimeError, match="not ready"):
        app.command("start")
    assert calls == ["one", "two"]
    assert app.phase == "idle"


def test_temporary_microphone_pick_switches_now_without_changing_saved_order():
    Config(audio_source="original", microphone_preferred="original,usb").save()
    app = make_controller(SwitchRecorder)
    app.command("start")
    capture = app.recorder
    item_id = app.item_id
    started = app.capture.started
    app.command("select_microphone", source="usb")
    assert capture.sources == ["original", "usb"]
    assert app.item_id == item_id
    assert app.capture.started == started
    assert app.status()["microphone"] == "usb"
    assert app.status()["session_microphone"] == "usb"
    assert "capture gap" in app.message
    assert Config.load().audio_source == "original"
    assert Config.load().microphone_preferred == "original,usb"
    app.command("stop")
    finish(app)
    app.command("start")
    assert app.recorder.sources == ["usb"]
    app.command("cancel")
    finish(app)


def test_persistent_pick_preserves_other_settings_and_never_list():
    Config(audio_source="original", microphone_never="bad", ai_model="original").save()
    app = make_controller(SwitchRecorder)
    app.command("select_microphone", source="usb", persist=True)
    assert Config.load().audio_source == "usb"
    assert app.status()["session_microphone"] is None
    assert Config.load().ai_model == "original"
    with pytest.raises(ValueError, match="Never"):
        app.command("select_microphone", source="bad", persist=True)
    assert Config.load().audio_source == "usb"


def test_switch_failure_recovers_previous_source_and_keeps_audio(monkeypatch):
    Config(audio_source="original").save()
    app = make_controller(SwitchRecorder)
    app.command("start")
    capture = app.recorder
    capture.reject.add("bad")
    monkeypatch.setattr(
        "mouthkeys.microphone.source_candidates",
        lambda config, excluded: [name for name in ("original",) if name not in excluded],
    )
    app.command("select_microphone", source="bad")
    assert capture.sources == ["original", "bad", "original"]
    assert app.phase == "recording"
    assert app.config.audio_source == "original"
    assert app.status()["session_microphone"] is None
    assert "Selected microphone failed" in app.notice["message"]
    assert app.job.audio_path.exists()
    app.command("cancel")
    finish(app)


def test_capture_session_preserves_exact_pcm_prefix_and_removes_private_segments(tmp_path):
    segments = {"first": b"\x11\x22" * 8000, "second": b"\x33\x44" * 8000}
    opened = []

    class Segment:
        def __init__(self, config, path, *, timeout, cancelled):
            assert timeout > 0
            assert not cancelled()
            self.path = path
            opened.append(path)
            with wave.open(str(path), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(segments[config.audio_source])

        def stop(self):
            pass

        def finished(self):
            return False

    path = tmp_path / "canonical.wav"
    session = CaptureSession(Config(audio_source="first"), path, recorder=Segment)
    with wave.open(str(path), "rb") as wav:
        assert wav.readframes(8000) == segments["first"]
    session.switch_source("second")
    session.stop()
    assert not session.thread.is_alive()
    assert all(not part.exists() for part in opened)
    assert path.stat().st_mode & 0o777 == 0o600
    with wave.open(str(path), "rb") as wav:
        assert wav.readframes(16000) == segments["first"] + segments["second"]
    assert list(tmp_path.glob("*.wav")) == [path]


def test_cancel_during_microphone_switch_suppresses_late_delivery():
    entered, released = threading.Event(), threading.Event()

    class DelayedSwitch(SwitchRecorder):
        def switch_source(self, source, *, timeout):
            entered.set()
            assert released.wait(2)
            super().switch_source(source, timeout=timeout)

    Config(audio_source="original").save()
    app = make_controller(DelayedSwitch)
    app.command("start")
    worker = threading.Thread(
        target=lambda: app.command("select_microphone", source="usb", persist=True)
    )
    worker.start()
    try:
        assert entered.wait(1)
        result = app.command("cancel")
        assert result["phase"] in ("stopping", "queued", "idle")
    finally:
        released.set()
        worker.join(2)
    assert not worker.is_alive()
    finish(app)
    assert not app.delivery.calls
    assert app.status()["session_microphone"] is None
    assert app.history.rows()[0]["status"] == "cancelled"
    assert Config.load().audio_source == "original"


def test_all_switch_candidates_fail_recovers_prior_audio_without_target(monkeypatch):
    app = make_controller(SwitchRecorder)
    app.command("start")
    app.recorder.reject.add("bad")
    monkeypatch.setattr("mouthkeys.microphone.source_candidates", lambda config, excluded: [])
    app.command("select_microphone", source="bad")
    finish(app)
    assert app.history.rows()[0]["status"] == "copied"
    assert app.delivery.calls[0][2] is None


def test_capture_session_failed_switch_keeps_original_audio(tmp_path):
    original = b"\x12\x34" * 8000

    class Segment:
        def __init__(self, config, path, **kwargs):
            if config.audio_source == "bad":
                raise RuntimeError("source unplugged")
            with wave.open(str(path), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(original)

        def stop(self):
            pass

        def finished(self):
            return False

    path = tmp_path / "canonical.wav"
    session = CaptureSession(Config(), path, recorder=Segment)
    with pytest.raises(RuntimeError, match="source unplugged"):
        session.switch_source("bad")
    session.stop()
    with wave.open(str(path), "rb") as wav:
        assert wav.readframes(8000) == original
    assert list(tmp_path.glob("*.wav")) == [path]


def test_release_hold_during_start_is_not_relatching_after_readiness():
    entered, release = threading.Event(), threading.Event()

    def record(config, path):
        entered.set()
        assert release.wait(2)
        return FakeRecorder(config, path)

    app = make_controller(record)
    worker = threading.Thread(target=lambda: app.command("start", held=True))
    worker.start()
    try:
        assert entered.wait(1)
        app.command("release_hold", id=app.job.item_id)
        assert not app.job.held_activation
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert app.job.phase == "recording"
    assert not app.job.held_activation
    app.command("cancel")
    finish(app)


def test_real_synthetic_ffmpeg_switch_keeps_both_sources(monkeypatch, tmp_path):
    import shutil
    import subprocess

    from mouthkeys.audio import Recorder

    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is not installed")
    raw = {}
    for name, sample in (("first", b"\x11\x22"), ("second", b"\x33\x44")):
        raw[name] = tmp_path / f"{name}.pcm"
        raw[name].write_bytes(sample * (4 * 16000))
    popen = subprocess.Popen

    def synthetic(args, **kwargs):
        args = list(args)
        source = args[args.index("-i") + 1]
        args[args.index("-f") + 1] = "s16le"
        args[args.index("-sample_rate")] = "-ar"
        args[args.index("-channels")] = "-ac"
        index = args.index("-fragment_size")
        del args[index : index + 2]
        args.insert(args.index("-i"), "-re")
        args[args.index("-i") + 1] = str(raw[source])
        return popen(args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", synthetic)
    monkeypatch.setattr("mouthkeys.capture_recorder.bluetooth_capture_ready", lambda _: True)
    path = tmp_path / "recording.wav"
    session = CaptureSession(Config(audio_source="first"), path, recorder=Recorder)
    try:
        session.switch_source("second")
    finally:
        session.stop()
    session.stop()  # Lifecycle cleanup is repeatable, including file handles.
    with wave.open(str(path), "rb") as wav:
        samples = wav.readframes(wav.getnframes())
    assert samples.startswith(b"\x11\x22" * 320)
    assert samples.endswith(b"\x33\x44" * 320)
    assert samples.count(b"\x33\x44") >= 320


def test_shutdown_during_start_discards_audio_without_delivery():
    entered, release = threading.Event(), threading.Event()

    def record(config, path):
        capture = FakeRecorder(config, path)
        entered.set()
        assert release.wait(2)
        return capture

    app = make_controller(record)
    result = []
    worker = threading.Thread(target=lambda: result.append(app.command("start")))
    worker.start()
    try:
        assert entered.wait(1)
        app.interrupt()
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert result[0]["phase"] == "interrupted"
    assert app.history.rows()[0]["status"] == "failed"
    assert not app.job.audio_path.exists()
    assert not app.delivery.calls


def test_unplugged_old_microphone_does_not_quarantine_untried_replacement():
    original, replacement = b"\x11\x22" * 8000, b"\x33\x44" * 8000
    opened = []

    class Segment:
        def __init__(self, config, path, **kwargs):
            self.source = config.audio_source
            self.ended = False
            opened.append(self.source)
            with wave.open(str(path), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(original if self.source == "old" else replacement)

        def stop(self):
            self.ended = True
            if self.source == "old":
                raise RuntimeError("Old microphone unplugged")

        def finished(self):
            return self.ended

    Config(audio_source="old").save()
    app = make_controller(lambda config, path: CaptureSession(config, path, recorder=Segment))
    app.command("start")
    app.command("select_microphone", source="healthy")
    assert opened == ["old", "healthy"]
    assert app.config.audio_source == "healthy"
    assert app.phase == "recording"
    assert "old" in app.capture.microphone.quarantine
    assert "healthy" not in app.capture.microphone.quarantine
    assert "Old microphone unplugged" in app.notice["message"]

    def transcribe(path, config):
        with wave.open(str(path), "rb") as wav:
            assert wav.readframes(16000) == original + replacement
        return "recognized"

    app.transcriber.transcribe = transcribe
    app.command("stop")
    finish(app)
    assert not app.job.audio_path.exists()
    assert app.history.rows()[0]["text"] == "recognized"
