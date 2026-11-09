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


def test_dictation_persists_before_delivery_and_keeps_latest_audio(controller):
    first = controller.command("toggle")["id"]
    controller.command("toggle")
    finish(controller)
    assert controller.history.rows()[0]["status"] == "copied"
    assert controller.history.recording(first).exists()
    second = controller.command("start")["id"]
    controller.command("stop")
    finish(controller)
    assert not controller.history.recording(first).exists()
    assert controller.history.recording(second).exists()
    assert len(controller.history.rows()) == 2


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


def test_delivery_failure_keeps_text_and_audio_for_reprocess(controller):
    controller.delivery.error = RuntimeError("Clipboard unavailable")
    item = controller.command("start")["id"]
    controller.command("stop")
    finish(controller)
    row = controller.history.rows()[0]
    assert row["status"] == "failed"
    assert row["text"] == controller.transcriber.text
    assert controller.history.recording(item).exists()
    controller.delivery.error = None
    controller.command("reprocess", id=item)
    finish(controller)
    assert controller.delivery.calls[-1][2] is None
    assert controller.history.rows()[0]["status"] == "copied"


def test_model_failure_and_empty_speech_are_recoverable(controller):
    controller.transcriber.error = RuntimeError("Download model first")
    item = controller.command("start")["id"]
    controller.command("stop")
    finish(controller)
    assert "Download model" in controller.history.rows()[0]["error"]
    assert controller.history.recording(item).exists()
    controller.transcriber.error = None
    controller.transcriber.text = ""
    controller.command("reprocess", id=item)
    finish(controller)
    assert "No speech" in controller.history.rows()[0]["error"]
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
    assert controller.history.recording(controller.item_id).exists()


def test_restart_marks_interrupted_audio_recoverable(controller):
    item = str(uuid.uuid4())
    controller.history.begin(item)
    write_audio(controller.history.recording(item))
    restarted = Controller(controller.history, controller.transcriber, controller.delivery)
    assert restarted.history.rows()[0]["status"] == "failed"
    restarted.command("reprocess")
    finish(restarted)
    assert restarted.history.rows()[0]["status"] == "copied"


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


@pytest.mark.parametrize("outcome", ["failure", "cancel", "restart"])
def test_reprocess_never_destroys_original_transcript_or_audio(controller, outcome):
    original_id = controller.command("start")["id"]
    controller.command("stop")
    finish(controller)
    original_row = controller.history.rows()[0]
    original_audio = controller.history.recording(original_id).read_bytes()
    controller.transcriber.entered.clear()
    controller.transcriber.release.clear()
    if outcome == "failure":
        controller.transcriber.error = RuntimeError("Model unavailable")
    retry_id = controller.command("reprocess", id=original_id)["id"]
    assert retry_id != original_id
    assert controller.transcriber.entered.wait(2)
    if outcome == "cancel":
        controller.command("cancel")
    if outcome == "restart":
        controller.history.recover()
        controller.command("cancel")
    controller.transcriber.release.set()
    finish(controller)
    rows = {row["id"]: row for row in controller.history.rows()}
    assert rows[original_id] == original_row
    assert controller.history.recording(original_id).read_bytes() == original_audio


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
def test_shutdown_preserves_audio_without_late_delivery(controller, during_inference):
    item = controller.command("start")["id"]
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
    assert controller.history.recording(item).exists()
    assert controller.history.rows()[0]["status"] == "failed"
    assert not controller.delivery.calls
    restarted = Controller(controller.history, controller.transcriber, controller.delivery)
    restarted.command("reprocess", id=item)
    finish(restarted)
    assert restarted.history.rows()[0]["status"] == "copied"


def test_audio_budget_prunes_old_failures_but_keeps_text_and_active_audio(controller):
    history = controller.history
    ids = [str(uuid.uuid4()) for _ in range(3)]
    for index, item in enumerate(ids):
        history.begin(item)
        history.recording(item).write_bytes(b"x" * 100)
        if index < 2:
            history.update(item, "failed", text=f"Transcript {index}")
        with history.connect() as conn:
            conn.execute("UPDATE history SET created=? WHERE id=?", (index, item))
    history.prune_budget(100)
    assert not history.recording(ids[0]).exists()
    assert history.recording(ids[1]).exists()
    assert history.recording(ids[2]).exists()  # Active capture is outside the recovery budget.
    assert len(history.rows()) == 3
    assert history.latest_text() == "Transcript 1"


def test_idle_maintenance_enforces_changed_audio_budget(controller, monkeypatch):
    monkeypatch.setattr("mouthkeys.service.time.monotonic", lambda: 5.0)
    Config(audio_budget_mb=1).save()
    for index in range(2):
        item = str(uuid.uuid4())
        controller.history.begin(item)
        controller.history.recording(item).write_bytes(b"x" * 700000)
        controller.history.update(item, "failed", text=f"Transcript {index}")
    controller.tick()
    assert sum(p.stat().st_size for p in controller.history.audio.glob("*.wav")) <= 1000000
    assert len(controller.history.rows()) == 2


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
