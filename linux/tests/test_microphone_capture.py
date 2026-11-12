import threading
import wave

import pytest
from test_pipeline import FakeDelivery, FakeRecorder, FakeTranscriber, finish, write_audio

from mouthkeys.audio import CaptureSession
from mouthkeys.config import Config
from mouthkeys.service import Controller
from mouthkeys.storage import History


def controller(recorder=FakeRecorder):
    history = History()
    return Controller(history, FakeTranscriber(), FakeDelivery(history), recorder)


def test_capture_failure_tries_alternate_and_quarantines_failed_source(monkeypatch):
    Config(audio_source="usb").save()
    calls = []

    def record(config, path):
        calls.append(config.audio_source)
        if config.audio_source == "usb":
            raise RuntimeError("USB readiness timed out")
        return FakeRecorder(config, path)

    monkeypatch.setattr(
        "mouthkeys.service.source_candidates",
        lambda config, excluded: [name for name in ("usb", "backup") if name not in excluded],
    )
    app = controller(record)
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

    app = controller(record)
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

    monkeypatch.setattr("mouthkeys.service.time.monotonic", lambda: elapsed[0])
    monkeypatch.setattr(
        "mouthkeys.service.source_candidates",
        lambda config, excluded: [name for name in ("one", "two", "three") if name not in excluded],
    )
    app = controller(record)
    with pytest.raises(RuntimeError, match="not ready"):
        app.command("start")
    assert calls == ["one", "two"]
    assert app.phase == "idle"


class SwitchRecorder(FakeRecorder):
    def __init__(self, config, path):
        super().__init__(config, path)
        self.sources = [config.audio_source]
        self.reject = set()

    def switch_source(self, source, *, timeout):
        assert 0 < timeout <= 5
        self.sources.append(source)
        if source in self.reject:
            raise RuntimeError(f"{source} unavailable")


def test_temporary_microphone_pick_switches_now_without_changing_saved_order():
    Config(audio_source="original", microphone_preferred="original,usb").save()
    app = controller(SwitchRecorder)
    app.command("start")
    capture = app.recorder
    item_id = app.item_id
    started = app.started
    app.command("select_microphone", source="usb")
    assert capture.sources == ["original", "usb"]
    assert app.item_id == item_id
    assert app.started == started
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
    app = controller(SwitchRecorder)
    app.command("select_microphone", source="usb", persist=True)
    assert Config.load().audio_source == "usb"
    assert app.session_source is None
    assert Config.load().ai_model == "original"
    with pytest.raises(ValueError, match="Never"):
        app.command("select_microphone", source="bad", persist=True)
    assert Config.load().audio_source == "usb"


def test_switch_failure_recovers_previous_source_and_keeps_audio(monkeypatch):
    Config(audio_source="original").save()
    app = controller(SwitchRecorder)
    app.command("start")
    capture = app.recorder
    capture.reject.add("bad")
    monkeypatch.setattr(
        "mouthkeys.service.source_candidates",
        lambda config, excluded: [name for name in ("original",) if name not in excluded],
    )
    app.command("select_microphone", source="bad")
    assert capture.sources == ["original", "bad", "original"]
    assert app.phase == "recording"
    assert app.config.audio_source == "original"
    assert app.session_source is None
    assert "Selected microphone failed" in app.notice["message"]
    assert app.job.audio_path.exists()
    app.command("cancel")
    finish(app)


def test_capture_session_preserves_exact_pcm_prefix_and_removes_private_segments(tmp_path):
    segments = {"first": b"\x11\x22" * 8000, "second": b"\x33\x44" * 8000}
    opened = []

    class Segment:
        def __init__(self, config, path, *, timeout, cancelled):
            assert timeout > 0 and not cancelled()
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

    app = controller(DelayedSwitch)
    app.command("start")
    worker = threading.Thread(target=lambda: app.command("select_microphone", source="usb"))
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
    assert app.session_source is None
    assert app.history.rows()[0]["status"] == "cancelled"


def test_all_switch_candidates_fail_recovers_prior_audio_without_target(monkeypatch):
    app = controller(SwitchRecorder)
    app.command("start")
    app.recorder.reject.add("bad")
    monkeypatch.setattr("mouthkeys.service.source_candidates", lambda config, excluded: [])
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

    app = controller(record)
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
    monkeypatch.setattr("mouthkeys.audio.bluetooth_capture_ready", lambda _: True)
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


def test_shutdown_during_start_retains_recovery_audio_without_delivery():
    entered, release = threading.Event(), threading.Event()

    def record(config, path):
        capture = FakeRecorder(config, path)
        entered.set()
        assert release.wait(2)
        return capture

    app = controller(record)
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
    assert app.job.audio_path.exists()
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
    app = controller(lambda config, path: CaptureSession(config, path, recorder=Segment))
    app.command("start")
    app.command("select_microphone", source="healthy")
    assert opened == ["old", "healthy"]
    assert app.config.audio_source == "healthy"
    assert app.phase == "recording"
    assert "old" in app.source_quarantine
    assert "healthy" not in app.source_quarantine
    assert "Old microphone unplugged" in app.notice["message"]
    app.command("stop")
    finish(app)
    with wave.open(str(app.job.audio_path), "rb") as wav:
        assert wav.readframes(16000) == original + replacement


def test_unknown_old_capture_teardown_does_not_open_replacement(tmp_path):
    from mouthkeys.audio import CaptureTeardownError

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
    from mouthkeys.audio import CaptureTeardownError

    class UnknownStop(SwitchRecorder):
        def switch_source(self, source, *, timeout):
            raise CaptureTeardownError("old", RuntimeError("Cannot confirm microphone shutdown"))

    Config(audio_source="old").save()
    app = controller(UnknownStop)
    app.command("start")
    monkeypatch.setattr(
        "mouthkeys.service.source_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Must not open fallback")),
    )
    app.command("select_microphone", source="healthy")
    finish(app)
    assert "old" in app.source_quarantine
    assert "healthy" not in app.source_quarantine
    assert app.delivery.calls[0][2] is None


@pytest.mark.parametrize("retry_action", ["stop", "cancel", "toggle"])
def test_unconfirmed_recorder_shutdown_blocks_new_capture_until_retry(monkeypatch, retry_action):
    import io
    import subprocess

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

    monkeypatch.setattr("mouthkeys.audio.subprocess.Popen", Process)
    monkeypatch.setattr("mouthkeys.audio.os.killpg", cannot_confirm_kill)
    Config(audio_source="old").save()
    app = controller(lambda config, path: CaptureSession(config, path))
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
    assert capture.closed
    if retry_action == "cancel":
        assert not app.delivery.calls
        assert app.history.rows()[0]["status"] == "cancelled"
    else:
        assert app.delivery.calls[0][2] is None
        assert app.history.rows()[0]["status"] == "copied"
