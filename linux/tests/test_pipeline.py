import threading
import uuid
import wave

import pytest

from mouthkeys.config import Config
from mouthkeys.service import Controller
from mouthkeys.storage import History


def write_audio(path):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 8000)


class FakeRecorder:
    def __init__(self, config, path):
        self.path = path
        self.stopped = False
        self.ended = False
        write_audio(path)

    def stop(self):
        self.stopped = True

    def finished(self):
        return self.ended


class FakeTranscriber:
    def __init__(self, text="Hello, world! café 日本語"):
        self.text = text
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.error = None

    def load(self, config):
        pass

    def transcribe(self, path, config):
        self.entered.set()
        assert self.release.wait(3), "Timed out waiting for test inference gate"
        if self.error:
            raise self.error
        return self.text


class FakeDelivery:
    def __init__(self, history):
        self.calls = []
        self.error = None
        self.history = history

    def focus(self):
        return "123"

    def deliver(self, text, config, target):
        assert self.history.rows()[0]["text"] == text, "Transcript must be durable before delivery"
        self.calls.append((text, config, target))
        if self.error:
            raise self.error
        return "copied"

    def copy(self, text):
        self.calls.append(text)

    def paste_key(self, config):
        self.calls.append("paste")


@pytest.fixture
def controller():
    history = History()
    return Controller(history, FakeTranscriber(), FakeDelivery(history), FakeRecorder)


def finish(controller):
    controller.worker.join(timeout=4)
    assert not controller.worker.is_alive()
    assert controller.status()["phase"] == "idle"


def test_dictation_persists_before_delivery_and_discards_audio(controller):
    first = controller.command("toggle")["id"]
    controller.command("toggle")
    finish(controller)
    assert controller.history.rows()[0]["status"] == "copied"
    assert controller.status()["notice"] is None
    assert not controller.history.recording(first).exists()
    second = controller.command("start")["id"]
    controller.command("stop")
    finish(controller)
    assert not controller.history.recording(first).exists()
    assert not controller.history.recording(second).exists()
    assert list(controller.job.audio_path.parent.glob("*.wav")) == []
    assert len(controller.history.rows()) == 2


@pytest.mark.parametrize(
    "desktop,expected",
    [("wayland", "needs an X11 login session"), ("x11", "identify the focused field")],
)
def test_auto_paste_fallback_explains_why(controller, monkeypatch, desktop, expected):
    from mouthkeys import delivery

    monkeypatch.setattr(delivery, "session", lambda: desktop)
    monkeypatch.setattr(delivery.shutil, "which", lambda name: f"/fake/{name}")
    Config(delivery="x11", destination_guard="field").save()
    controller.command("start")
    controller.command("stop")
    finish(controller)
    status = controller.status()
    assert status["notice"]["kind"] == "paste"
    assert expected in status["notice"]["message"]
    assert expected in status["message"]
    assert controller.history.rows()[0]["status"] == "copied"


def test_cancel_recording_stops_capture_without_delivery(controller):
    item = controller.command("start")["id"]
    recorder = controller.recorder
    controller.command("cancel")
    finish(controller)
    assert recorder.stopped
    assert not controller.history.recording(item).exists()
    assert controller.history.rows()[0]["status"] == "cancelled"
    assert not controller.delivery.calls


def test_cancel_during_inference_suppresses_late_paste(controller):
    controller.transcriber.release.clear()
    controller.command("start")
    controller.command("stop")
    assert controller.transcriber.entered.wait(2)
    controller.command("cancel")
    with pytest.raises(ValueError, match="Busy"):
        controller.command("clear")
    controller.transcriber.release.set()
    finish(controller)
    assert not controller.delivery.calls
    assert controller.history.rows()[0]["status"] == "cancelled"


def test_delivery_failure_keeps_text_but_discards_audio(controller):
    controller.delivery.error = RuntimeError("Clipboard unavailable")
    controller.command("start")
    path = controller.job.audio_path
    assert path.exists()
    controller.command("stop")
    finish(controller)
    row = controller.history.rows()[0]
    assert row["status"] == "failed"
    assert row["text"] == controller.transcriber.text
    assert not path.exists()


@pytest.mark.parametrize("error", [RuntimeError("Download model first"), None])
def test_model_failure_and_empty_speech_discard_audio(controller, error):
    controller.transcriber.error = error
    controller.transcriber.text = ""
    controller.command("start")
    path = controller.job.audio_path
    controller.command("stop")
    finish(controller)
    assert controller.history.rows()[0]["status"] == "failed"
    assert not path.exists()
    assert not controller.delivery.calls


def test_source_and_config_frozen_at_start_target_at_stop(controller):
    Config(audio_source="mic-A", delivery="x11").save()
    controller.command("start")
    Config(audio_source="mic-B").save()
    controller.command("stop")
    finish(controller)
    _, config, target = controller.delivery.calls[0]
    assert config.audio_source == "mic-A"
    assert target == "123"


def test_auto_stop_at_capture_limit(controller):
    controller.command("start")
    recorder = controller.recorder
    recorder.ended = True
    controller.tick()
    finish(controller)
    assert recorder.stopped


def test_capture_wall_clock_limit_stops_even_a_stalled_source(controller):
    controller.command("start")
    recorder = controller.recorder
    controller.started -= controller.config.max_seconds + 1
    controller.tick()
    finish(controller)
    assert recorder.stopped


def test_capture_start_failure_does_not_leave_busy(controller):
    def fail(*_):
        raise RuntimeError("Source unplugged")

    controller.recorder_factory = fail
    with pytest.raises(RuntimeError, match="Source unplugged"):
        controller.command("start")
    assert controller.phase == "idle"
    assert controller.history.rows()[0]["status"] == "failed"


def test_capture_stop_failure_retains_audio(controller):
    controller.command("start")

    def fail():
        raise RuntimeError("Capture hung")

    controller.recorder.stop = fail
    controller.command("stop")
    finish(controller)
    assert "Capture hung" in controller.message
    assert controller.phase == "idle"
    assert controller.recorder is None
    assert not controller.job.audio_path.exists()


def test_restart_deletes_legacy_audio_but_preserves_transcripts(controller):
    item = str(uuid.uuid4())
    controller.history.begin(item)
    controller.history.update(item, "transcribing", "already recognized")
    path = controller.history.recording(item)
    write_audio(path)
    restarted = Controller(controller.history, controller.transcriber, controller.delivery)
    assert restarted.history.rows()[0]["status"] == "failed"
    assert restarted.history.rows()[0]["text"] == "already recognized"
    assert not path.exists()


def test_history_clear_removes_recordings_and_transcripts(controller):
    controller.command("start")
    controller.command("stop")
    finish(controller)
    controller.command("clear")
    assert controller.history.rows() == []
    assert list(controller.history.audio.iterdir()) == []


def test_history_rejects_path_traversal(controller):
    with pytest.raises(ValueError):
        controller.command("reprocess", id="../../outside")


def test_private_files(controller):
    assert controller.history.root.stat().st_mode & 0o777 == 0o700
    assert controller.history.db.stat().st_mode & 0o777 == 0o600


def test_config_validation_and_atomic_roundtrip():
    config = Config(model="small", language="auto", max_seconds=60)
    config.save()
    assert Config.load() == config
    with pytest.raises(ValueError):
        Config(max_seconds=-1).save()
    assert Config.load() == config


def test_preview_is_live_but_never_persisted_or_delivered(controller):
    controller.transcriber.preview = lambda path, config: "provisional words"
    controller.command("start")
    controller.preview_at = 0
    controller.tick()
    controller.preview_worker.join(3)
    assert not controller.preview_worker.is_alive()
    assert controller.status()["preview"] == "provisional words"
    assert controller.history.rows()[0]["text"] == ""
    assert not controller.delivery.calls
    controller.command("stop")
    finish(controller)
    assert controller.status()["preview"] == controller.transcriber.text
    assert controller.delivery.calls[0][0] == controller.transcriber.text


def test_cancelled_preview_cannot_leak_into_next_recording(controller):
    entered, release = threading.Event(), threading.Event()

    def preview(path, config):
        entered.set()
        assert release.wait(3)
        return "stale words"

    controller.transcriber.preview = preview
    controller.command("start")
    controller.preview_at = 0
    controller.tick()
    assert entered.wait(2)
    controller.command("cancel")
    finish(controller)
    controller.command("start")
    release.set()
    controller.preview_worker.join(3)
    assert not controller.preview_worker.is_alive()
    assert controller.status()["preview"] == ""
    assert not controller.delivery.calls
    controller.command("cancel")


def test_final_inference_waits_for_preview_and_cancel_suppresses_delivery(controller):
    entered, release = threading.Event(), threading.Event()

    def preview(path, config):
        entered.set()
        assert release.wait(3)
        return "stale words"

    controller.transcriber.preview = preview
    controller.command("start")
    controller.preview_at = 0
    controller.tick()
    assert entered.wait(2)
    controller.command("stop")
    assert not controller.transcriber.entered.is_set()
    controller.command("cancel")
    release.set()
    finish(controller)
    controller.preview_worker.join(3)
    assert controller.status()["preview"] == ""
    assert not controller.transcriber.entered.is_set()
    assert not controller.delivery.calls


def test_preview_failure_does_not_prevent_final_transcription(controller):
    def preview(path, config):
        raise RuntimeError("Preview unavailable")

    controller.transcriber.preview = preview
    controller.command("start")
    controller.preview_at = 0
    controller.tick()
    controller.preview_worker.join(3)
    controller.command("stop")
    finish(controller)
    assert controller.delivery.calls


@pytest.mark.parametrize("empty", [False, True])
def test_failed_final_never_exposes_provisional_text_as_copyable(controller, empty):
    controller.transcriber.preview = lambda path, config: "unfinished words"
    controller.command("start")
    controller.preview_at = 0
    controller.tick()
    controller.preview_worker.join(3)
    assert controller.status()["preview"] == "unfinished words"
    if empty:
        controller.transcriber.text = ""
    else:
        controller.transcriber.error = RuntimeError("Final inference failed")
    controller.command("stop")
    finish(controller)
    assert controller.status()["preview"] == ""
    assert not controller.delivery.calls


@pytest.mark.parametrize("wall_clock", [False, True])
def test_automatic_stop_never_snapshots_a_paste_destination(controller, wall_clock):
    Config(delivery="x11").save()
    controller.command("start")

    def unexpected():
        raise AssertionError("An automatic stop must not select the focused window")

    controller.delivery.focus = unexpected
    if wall_clock:
        controller.started -= controller.config.max_seconds + 1
    else:
        controller.recorder.ended = True
    controller.tick()
    finish(controller)
    assert controller.delivery.calls[0][2] is None


def test_stop_remains_responsive_and_cancel_suppresses_inference(controller):
    entered, release = threading.Event(), threading.Event()
    controller.command("start")

    def stop():
        entered.set()
        assert release.wait(3)

    controller.recorder.stop = stop
    controller.command("stop")
    assert entered.wait(1)
    # A separate thread catches regressions that hold the controller lock during stop.
    result = []
    probe = threading.Thread(target=lambda: result.append(controller.command("cancel")))
    probe.start()
    try:
        probe.join(0.5)
        assert not probe.is_alive()
        assert result[0]["phase"] == "stopping"
    finally:
        release.set()
        probe.join(2)
        finish(controller)
    assert not controller.transcriber.entered.is_set()
    assert not controller.delivery.calls


def test_status_responsive_during_committed_delivery(controller):
    entered, release = threading.Event(), threading.Event()

    def deliver(*args):
        entered.set()
        assert release.wait(3)
        return "copied"

    controller.delivery.deliver = deliver
    controller.command("start")
    controller.command("stop")
    assert entered.wait(2)
    result = []
    probe = threading.Thread(target=lambda: result.append(controller.status()))
    probe.start()
    try:
        probe.join(0.5)
        assert not probe.is_alive()
        assert result[0]["phase"] == "delivering"
        with pytest.raises(ValueError, match="Delivery already started"):
            controller.command("cancel")
    finally:
        release.set()
        probe.join(2)
        finish(controller)


@pytest.mark.parametrize("during_inference", [False, True])
def test_shutdown_discards_audio_without_late_delivery(controller, during_inference):
    controller.command("start")
    if during_inference:
        controller.transcriber.release.clear()
        controller.command("stop")
        assert controller.transcriber.entered.wait(2)
    else:
        recorder = controller.recorder
    controller.interrupt()
    if during_inference:
        controller.transcriber.release.set()
        controller.worker.join(3)
        assert not controller.worker.is_alive()
    else:
        assert recorder.stopped
    assert not controller.job.audio_path.exists()
    assert controller.history.rows()[0]["status"] == "failed"
    assert not controller.delivery.calls


def test_explicit_stop_snapshots_focus_before_acknowledging(controller):
    Config(delivery="x11").save()
    controller.command("start")
    entered, release = threading.Event(), threading.Event()

    def focus():
        entered.set()
        assert release.wait(3)
        return "original-window"

    controller.delivery.focus = focus
    responses = []
    caller = threading.Thread(target=lambda: responses.append(controller.command("stop")))
    caller.start()
    try:
        assert entered.wait(1)
        assert caller.is_alive()  # No acknowledgment before the snapshot.
        assert controller.status()["phase"] == "stopping"
    finally:
        release.set()
        caller.join(2)
        finish(controller)
    assert responses
    assert controller.delivery.calls[0][2] == "original-window"


def test_shutdown_during_manual_copy_preserves_completed_history(controller):
    controller.command("start")
    controller.command("stop")
    finish(controller)
    original = controller.history.rows()[0]
    entered, release = threading.Event(), threading.Event()

    def copy(text):
        entered.set()
        assert release.wait(3)

    controller.delivery.copy = copy
    caller = threading.Thread(target=lambda: controller.command("copy"))
    caller.start()
    try:
        assert entered.wait(1)
        controller.interrupt()
        assert controller.history.rows()[0] == original
    finally:
        release.set()
        caller.join(2)
    assert not caller.is_alive()


def test_error_identifiers_are_distinct_across_daemon_restarts(controller):
    controller.report_error("Busy (loading)")
    first = controller.status()["notice"]["serial"]
    restarted = Controller(controller.history, controller.transcriber, controller.delivery)
    restarted.report_error("Source unplugged")
    assert restarted.status()["notice"]["serial"] != first


def test_audio_cleanup_survives_native_reset_failure(controller):
    controller.command("start")
    job = controller.job
    controller.command("cancel")
    finish(controller)
    write_audio(job.audio_path)
    controller.transcriber.stream_key = (job.audio_path,)

    def fail_reset():
        raise RuntimeError("Native worker unavailable")

    controller.transcriber.reset_stream = fail_reset
    job.done.clear()
    with pytest.raises(RuntimeError, match="Native worker unavailable"):
        controller._finish(job, None)
    assert job.done.is_set()
    assert not job.audio_path.exists()


def test_old_config_cannot_reenable_audio_retention():
    import json

    from mouthkeys.config import config_path

    Config().save()
    path = config_path()
    values = json.loads(path.read_text())
    values.update(save_audio="on", audio_budget_mb=4000, reprocess_shortcut_key="super+r")
    path.write_text(json.dumps(values))
    loaded = Config.load()
    loaded.save()
    assert (
        not {"save_audio", "audio_budget_mb", "reprocess_shortcut_key"}
        & json.loads(path.read_text()).keys()
    )
