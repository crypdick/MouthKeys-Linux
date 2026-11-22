import threading

import pytest

from mouthkeys.config import Config

from .pipeline_support import FakeRecorder, SwitchRecorder, finish, make_controller


def test_standard_capture_factory_receives_job_cancel_event_and_readiness_budget(monkeypatch):
    class Standard(FakeRecorder):
        def __init__(self, config, path, *, cancel_event, timeout):
            super().__init__(config, path)
            self.cancel_event = cancel_event
            self.timeout = timeout

    monkeypatch.setattr("mouthkeys.recording_capture.CaptureSession", Standard)
    app = make_controller(Standard)
    assert app.command("start")["phase"] == "recording"
    assert app.recorder.cancel_event is app.job.cancel_event
    assert 0 < app.recorder.timeout <= 5
    app.command("cancel")
    finish(app)


def test_adapter_without_switch_support_keeps_current_recording():
    Config(audio_source="original").save()
    app = make_controller()
    app.command("start")
    capture = app.recorder
    with pytest.raises(ValueError, match="does not support microphone switching"):
        app.command("select_microphone", source="other", persist=True)
    assert app.recorder is capture
    assert app.status()["phase"] == "recording"
    assert Config.load().audio_source == "original"
    app.command("cancel")
    finish(app)


def test_persisting_current_microphone_does_not_trigger_switch():
    Config(audio_source="original").save()
    app = make_controller(SwitchRecorder)
    app.command("select_microphone", source="session")
    app.command("start")
    capture = app.recorder
    app.command("select_microphone", source="session", persist=True)
    assert capture.sources == ["session"]
    assert app.status()["session_microphone"] is None
    assert Config.load().audio_source == "session"
    app.command("cancel")
    finish(app)


def test_live_microphone_preference_saves_against_latest_settings():
    class UpdatesSettings(SwitchRecorder):
        def switch_source(self, source, *, timeout):
            Config(audio_source="original", language="fr").save()
            super().switch_source(source, timeout=timeout)

    Config(audio_source="original", language="en").save()
    app = make_controller(UpdatesSettings)
    app.command("start")
    assert app.command("select_microphone", source="usb", persist=True)["phase"] == "recording"
    assert app.job.config.language == "en"  # The running transcription keeps its captured options.
    assert Config.load().language == "fr"
    assert Config.load().audio_source == "usb"
    assert app.status()["session_microphone"] is None
    app.command("cancel")
    finish(app)


def test_live_microphone_save_failure_keeps_working_choice_for_session(monkeypatch):
    Config(audio_source="original").save()
    app = make_controller(SwitchRecorder)
    app.command("start")

    def fail(_config):
        raise OSError("Settings are read only")

    monkeypatch.setattr(Config, "save", fail)
    status = app.command("select_microphone", source="usb", persist=True)
    assert status["phase"] == "recording"
    assert status["microphone"] == status["session_microphone"] == "usb"
    assert Config.load().audio_source == "original"
    assert "preference not saved" in status["notice"]["message"]
    app.command("cancel")
    finish(app)


def test_microphone_selection_rejects_startup_and_shutdown_without_saving():
    entered, release = threading.Event(), threading.Event()

    def record(config, path):
        entered.set()
        assert release.wait(2)
        return FakeRecorder(config, path)

    Config(audio_source="original").save()
    app = make_controller(record)
    worker = threading.Thread(target=lambda: app.command("start"))
    worker.start()
    try:
        assert entered.wait(1)
        with pytest.raises(ValueError, match=r"Busy \(starting\)"):
            app.command("select_microphone", source="usb", persist=True)
        app.interrupt()
        with pytest.raises(ValueError, match="Service is shutting down"):
            app.command("select_microphone", source="usb", persist=True)
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert Config.load().audio_source == "original"
    assert not app.delivery.calls


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"mode": "unknown"}, "Unknown recording mode"),
        ({"selected_text": 10}, "Selected text exceeds"),
        ({"mode": "pronunciation", "label": "not\nprintable"}, "Pronunciation label"),
        ({"assistant_history": []}, "Inline assistant history requires private"),
        ({"assistant_executions": []}, "Execution context requires inline assistant history"),
    ],
)
def test_invalid_start_does_not_open_or_persist_a_recording(data, message):
    app = make_controller(lambda *_: pytest.fail("Capture started from invalid input"))
    with pytest.raises(ValueError, match=message):
        app.command("start", **data)
    assert app.status()["phase"] == "idle"
    assert app.history.rows() == []
    assert app.job is None


def test_invalid_profile_leaves_no_job_and_valid_pronunciation_still_starts():
    app = make_controller()
    with pytest.raises(ValueError, match="Prompt override must name a profile"):
        app.command("start", profile=123)
    assert app.job is None
    assert app.status()["pending_jobs"] == 0
    assert app.history.rows() == []
    result = app.command("start", mode="pronunciation", label="  Café  ", profile="off")
    assert result["phase"] == "recording"
    assert app.job.label == "Café"
    assert app.job.prompt_override == "off"
    app.command("cancel")
    finish(app)


def test_stop_during_readiness_cancels_even_when_capture_then_fails(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def failed_capture(config, path):
        FakeRecorder(config, path)
        entered.set()
        assert release.wait(3)
        raise RuntimeError("Input disconnected")

    app = make_controller(failed_capture)
    pauses, resumes, result = [], [], []
    monkeypatch.setattr(app.capture.media, "pause", pauses.append)
    monkeypatch.setattr(app.capture.media, "resume", lambda: resumes.append(True))
    worker = threading.Thread(target=lambda: result.append(app.command("start")))
    worker.start()
    try:
        assert entered.wait(2)
        active = app.status()
        assert active["phase"] == "starting"
        assert app.command("stop", id="older-job") == active
        assert app.command("stop", id=active["id"])["message"] == "Cancelling microphone start"
        assert app.job.cancel_event.is_set()
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert result[0]["message"] == "Microphone start cancelled"
    assert len(pauses) == 1
    assert resumes == [True]
    assert app.history.get(active["id"])["status"] == "cancelled"
    assert app.history.get(active["id"])["error"] == ""
    assert not app.job.audio_path.exists()
    assert not app.delivery.calls


def test_shutdown_rejects_new_start_without_opening_capture():
    app = make_controller(lambda *_: pytest.fail("Capture started after shutdown"))
    app.interrupt()
    with pytest.raises(ValueError, match="Service is shutting down"):
        app.command("start")
    assert app.status()["phase"] == "interrupted"
    assert app.history.rows() == []


@pytest.mark.parametrize("unavailable", [True, False])
def test_edit_start_never_uses_incomplete_or_unavailable_selected_text(unavailable):
    app = make_controller()

    def capture(_config):
        if unavailable:
            raise OSError("Desktop snapshot unavailable")
        return {
            "field": {
                "complete": False,
                "text": "private excerpt",
                "selection_start": 0,
                "selection_end": 7,
            }
        }

    app.delivery.capture_target = capture
    app.command("start", mode="edit")
    assert app.job.selected_text == ""
    assert (app.job.start_target is None) is unavailable
    app.command("cancel")
    finish(app)
