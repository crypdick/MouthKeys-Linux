import threading
import time
from dataclasses import replace

from test_pipeline import FakeDelivery, FakeRecorder, FakeTranscriber, finish

from mouthkeys.config import Config
from mouthkeys.service import Controller
from mouthkeys.storage import History


def controller(text="Hello world"):
    history = History()
    return Controller(history, FakeTranscriber(text), FakeDelivery(history), FakeRecorder)


def test_spoken_send_requires_explicit_paste_and_strips_phrase():
    Config(delivery="x11", spoken_send="on").save()
    app = controller("Hello world send it")
    sent = []
    app.delivery.deliver = lambda text, config, target: (
        app.delivery.calls.append((text, target)) or "paste_requested"
    )
    app.delivery.send_key = lambda config, target: sent.append((config.spoken_send_key, target))
    app.command("start")
    app.command("stop")
    finish(app)
    assert app.delivery.calls == [("Hello world", "123")]
    assert sent == [("return", "123")]
    assert app.history.rows()[0]["raw_text"] == "Hello world send it"
    assert app.history.rows()[0]["text"] == "Hello world"


def test_clipboard_only_does_not_swallow_send_phrase_or_send_key():
    Config(spoken_send="on").save()
    app = controller("Hello world send it")
    app.delivery.send_key = lambda *_: (_ for _ in ()).throw(AssertionError("Unexpected send"))
    app.command("start")
    app.command("stop")
    finish(app)
    assert app.history.rows()[0]["text"] == "Hello world send it"


def test_copy_fallback_reports_spoken_send_was_not_dispatched():
    Config(delivery="x11", spoken_send="on").save()
    app = controller("Hello world send it")
    app.delivery.send_key = lambda *_: (_ for _ in ()).throw(AssertionError("Unexpected send"))
    app.command("start")
    app.command("stop")
    finish(app)
    assert "Spoken Send skipped: text copied" in app.message
    row = app.history.rows()[0]
    assert row["text"] == "Hello world"
    assert "Spoken Send skipped" in row["error"]


def test_cancel_send_countdown_leaves_inserted_text_without_return():
    Config(delivery="x11", spoken_send="on").save()
    app = controller("Hello world send it")
    app.delivery.deliver = lambda *_: "paste_requested"
    sent = []
    app.delivery.send_key = lambda *_: sent.append(True)
    app.command("start")
    app.command("stop")
    deadline = time.monotonic() + 2
    while not app.send_pending and time.monotonic() < deadline:
        time.sleep(0.01)
    assert app.send_pending
    app.command("cancel")
    finish(app)
    assert not sent
    assert app.history.rows()[0]["text"] == "Hello world"


def test_ai_failure_falls_back_to_formatted_transcript_without_losing_raw(monkeypatch):
    Config(ai_enabled="on", ai_model="model", lowercase_first="on").save()
    app = controller()

    def fail(*_):
        raise RuntimeError("offline")

    monkeypatch.setattr("mouthkeys.service.process", fail)
    app.command("start")
    app.command("stop")
    finish(app)
    assert app.delivery.calls[0][0] == "hello world"
    assert app.history.rows()[0]["raw_text"] == "Hello world"
    assert "AI cleanup failed" in app.status()["message"]
    row = app.history.rows()[0]
    assert row["was_ai_processed"] == 0
    assert "offline" in row["ai_processing_error"]
    assert "AI cleanup failed" in app.status()["notice"]["message"]


def test_ai_history_records_actual_success_even_when_text_unchanged(monkeypatch):
    Config(ai_enabled="on", ai_model="model").save()
    app = controller("Unchanged")
    monkeypatch.setattr("mouthkeys.service.process", lambda text, config: text)
    app.command("start")
    app.command("stop")
    finish(app)
    row = app.history.rows()[0]
    assert row["raw_text"] == row["text"] == "Unchanged"
    assert row["was_ai_processed"] == 1
    assert row["ai_processing_error"] == ""


def test_edit_delivery_failure_is_not_recorded_as_ai_failure(monkeypatch):
    Config(ai_enabled="on", ai_model="model", edit_delivery="replace").save()
    app = controller("Edit instruction")
    monkeypatch.setattr(app, "_propose_assistant", lambda *a: {"mode": "edit", "text": "Edited"})
    monkeypatch.setattr(app, "_edit_target_matches", lambda job: True)

    def failed_delivery(*args, **kwargs):
        raise RuntimeError("Desktop helper missing")

    app.delivery.replace_selection = failed_delivery
    app.command("start", mode="edit")
    app.command("stop")
    finish(app)
    row = app.history.rows()[0]
    assert row["status"] == "failed"
    assert row["was_ai_processed"] == 1
    assert row["ai_processing_error"] == ""
    assert row["error"] == "Desktop helper missing"


def test_cancelled_ai_result_does_not_leave_success_or_raw_transcript(monkeypatch):
    Config(ai_enabled="on", ai_model="model").save()
    app = controller("Unchanged")
    entered, release = threading.Event(), threading.Event()

    def process(text, config):
        entered.set()
        assert release.wait(2)
        return text

    monkeypatch.setattr("mouthkeys.service.process", process)
    app.command("start")
    app.command("stop")
    try:
        assert entered.wait(1)
        app.command("cancel")
    finally:
        release.set()
        finish(app)
    row = app.history.rows()[0]
    assert row["was_ai_processed"] == 0
    assert row["ai_processing_error"] == row["raw_text"] == row["text"] == ""


def test_cancel_during_ai_suppresses_delivery(monkeypatch):
    Config(ai_enabled="on", ai_model="model").save()
    app = controller()
    entered, release = threading.Event(), threading.Event()

    def processing(text, config):
        assert app.history.rows()[0]["raw_text"] == text
        entered.set()
        assert release.wait(2)
        return "Edited"

    monkeypatch.setattr("mouthkeys.service.process", processing)
    app.command("start")
    app.command("stop")
    assert entered.wait(2)
    app.command("cancel")
    release.set()
    finish(app)
    assert not app.delivery.calls
    assert app.history.rows()[0]["status"] == "cancelled"
    assert app.history.rows()[0]["raw_text"] == ""


def test_privacy_opt_out_never_writes_transcript_history():
    Config(save_history="off").save()
    app = controller()
    item = app.command("start")["id"]
    path = app.job.audio_path
    assert "private-recordings" in str(path)
    assert app.history.rows() == []

    def deliver(text, config, target):
        assert app.history.rows() == []
        app.delivery.calls.append((text, config, target))
        return "copied"

    app.delivery.deliver = deliver
    app.command("stop")
    finish(app)
    assert app.delivery.calls[0][0] == "Hello world"
    assert app.history.rows() == []
    assert not path.exists()
    assert not app.history.recording(item).exists()
    # Paste last remains available in memory for this service session.
    app.command("copy")
    assert app.delivery.calls[-1] == "Hello world"


def test_save_audio_preserves_previous_success_with_budget_bound():
    Config(save_audio="on").save()
    app = controller()
    first = app.command("start")["id"]
    app.command("stop")
    finish(app)
    second = app.command("start")["id"]
    app.command("stop")
    finish(app)
    assert all(app.history.recording(item).exists() for item in (first, second))


def test_skip_silent_recordings_is_explicit_and_does_not_load_model():
    Config(skip_silent_recordings="on").save()
    app = controller()
    app.command("start")
    app.command("stop")
    finish(app)
    assert app.message == "Silent recording skipped"
    assert not app.transcriber.entered.is_set()
    assert not app.delivery.calls


def test_spoken_pause_stops_but_duration_limit_does_not_send():
    config = Config(delivery="x11", spoken_send="on", send_after_pause="on")
    config.save()
    app = controller("Hello send it")
    app.delivery.deliver = lambda *_: "paste_requested"
    sends = []
    app.delivery.send_key = lambda *_: sends.append(True)
    app.command("start")
    app.send_armed_at = time.monotonic() - 1
    app.last_voice_at = time.monotonic() - 1
    app.tick()
    finish(app)
    assert sends == [True]
    replace(config, send_after_pause="off").save()
    app.command("start")
    app.recorder.ended = True
    app.tick()
    finish(app)
    assert sends == [True]
    assert app.history.rows()[0]["text"] == "Hello send it"


def test_training_records_actual_recognized_phrase_without_any_delivery():
    Config(lowercase_first="on", spoken_send="on", ai_enabled="on", ai_model="model").save()
    app = controller("Mouth keys send it")
    app.command("start", mode="training")
    app.command("stop")
    finish(app)
    assert not app.delivery.calls
    assert app.status()["assistant_proposal"] == {"mode": "training", "text": "Mouth keys send it"}
    assert app.history.rows()[0]["raw_text"] == "Mouth keys send it"
    assert app.history.rows()[0]["text"] == "Mouth keys send it"


def test_assistant_mode_returns_proposal_without_executing_or_inserting(monkeypatch):
    app = controller("List files")
    calls = []

    def propose(text, config, mode, selected_text, **kwargs):
        calls.append((text, mode, selected_text))
        return {"mode": mode, "instruction": text, "text": "ls -la", "command": "ls -la"}

    monkeypatch.setattr("mouthkeys.assistance.propose", propose)
    app.command("start", mode="command", selected_text="chosen")
    app.command("stop")
    finish(app)
    assert calls == [("List files", "command", "chosen")]
    assert app.status()["assistant_proposal"]["command"] == "ls -la"
    assert app.history.rows()[0]["raw_text"] == "List files"
    assert app.history.rows()[0]["text"] == "ls -la"
    assert not app.delivery.calls


def test_new_recording_suppresses_older_pending_spoken_send():
    Config(delivery="x11", spoken_send="on").save()
    app = controller("Hello send it")
    app.delivery.deliver = lambda *_: "paste_requested"
    sends = []
    app.delivery.send_key = lambda *_: sends.append(True)
    first = app.command("start")["id"]
    app.command("stop")
    old_worker = app.worker
    deadline = time.monotonic() + 2
    while not app.send_pending and time.monotonic() < deadline:
        time.sleep(0.005)
    assert app.send_pending
    app.command("start")
    app.command("cancel")
    finish(app)
    old_worker.join(2)
    assert not old_worker.is_alive()
    assert sends == []
    row = next(row for row in app.history.rows() if row["id"] == first)
    assert "newer recording" in row["error"]


def test_old_preview_cannot_arm_a_new_recording(monkeypatch):
    Config(delivery="x11", spoken_send="on", send_after_pause="on").save()
    app = controller()
    entered, release = threading.Event(), threading.Event()

    def preview(*_):
        entered.set()
        assert release.wait(2)
        return "Hello send it"

    app.transcriber.preview = preview
    app.delivery.deliver = lambda *_: "copied"
    app.command("start")
    app.preview_at = 0
    app.tick()
    preview_worker = app.preview_worker
    assert entered.wait(2)
    app.command("stop")
    old_worker = app.worker
    deadline = time.monotonic() + 2
    while app.phase == "stopping" and time.monotonic() < deadline:
        time.sleep(0.005)
    app.command("start")
    app.preview_at = -123
    release.set()
    preview_worker.join(2)
    assert not preview_worker.is_alive()
    assert app.send_armed_at is None
    assert app.preview_at == -123
    assert app.preview_text == ""
    app.command("cancel")
    finish(app)
    old_worker.join(2)
    assert not old_worker.is_alive()


def test_history_disabled_latest_copy_does_not_resurrect_older_saved_text():
    import uuid

    Config(save_history="off").save()
    app = controller("New private text")
    old = str(uuid.uuid4())
    app.history.begin(old)
    app.history.update(old, "copied", "Old saved text")
    app.command("start")
    app.command("stop")
    finish(app)
    app.command("copy")
    assert app.delivery.calls[-1] == "New private text"
    app.command("copy", id=old)
    assert app.delivery.calls[-1] == "Old saved text"


def test_destination_metadata_is_persisted_without_enabling_clipboard_paste():
    Config(delivery="clipboard").save()
    app = controller()
    targets = []
    app.delivery.target_metadata = lambda target: (
        targets.append(target) or {"app": "Editor", "window": "Project"}
    )
    app.command("start")
    app.command("stop")
    finish(app)
    assert targets == ["123"]
    assert app.delivery.calls[0][2] is None
    assert app.history.rows()[0]["app"] == "Editor"
    assert app.history.rows()[0]["window"] == "Project"


def test_cancel_during_final_send_probe_prevents_key_dispatch():
    Config(delivery="x11", spoken_send="on").save()
    app = controller("Hello send it")
    app.delivery.deliver = lambda *_: "paste_requested"
    app.delivery.send_key_guarded = True
    entered, release = threading.Event(), threading.Event()
    sent = []

    def send(config, target, *, before_send, allow_without_insert):
        assert not allow_without_insert
        entered.set()
        assert release.wait(2)
        if before_send():
            sent.append(True)
            return True
        return False

    app.delivery.send_key = send
    app.command("start")
    app.command("stop")
    assert entered.wait(2)
    assert app.command("cancel")["phase"] != "error"
    release.set()
    finish(app)
    assert sent == []
    assert app.history.rows()[0]["text"] == "Hello"


def test_queued_recording_preview_preserves_predecessor_native_cache():
    Config().save()
    app = controller()
    app.command("start")
    previous = threading.Event()
    app.job.predecessors = (previous,)
    calls = []
    app.transcriber.preview = lambda *_: calls.append(True) or "Preview"
    app._preview(app.item_id, Config())
    assert calls == []
    previous.set()
    app._preview(app.item_id, Config())
    assert calls == [True]
    assert app.preview_text == "Preview"
    app.command("cancel")
    finish(app)
