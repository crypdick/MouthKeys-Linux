from pathlib import Path
from threading import Event, Thread

import pytest

from mouthkeys import recording_finalizer
from mouthkeys.pronunciation import PronunciationStore

from .pipeline_support import finish, make_controller


def test_private_audio_cleanup_failure_does_not_stall_later_jobs(monkeypatch):
    app = make_controller()
    app.command("start")
    job = app.job
    unlink = Path.unlink

    def fail_discard(path, *args, **kwargs):
        if path == job.audio_path:
            raise PermissionError("private recording path")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as flaky_storage:
        flaky_storage.setattr(Path, "unlink", fail_discard)
        app.command("stop")
        finish(app)
        assert job.done.is_set()
        assert app.history.get(job.item_id)["status"] == "copied"
        assert app.status()["notice"]["message"] == "Could not discard temporary recording"
        assert "private recording path" not in str(app.status())

        # A committed job must never remain listed as predecessor after cleanup.
        assert app.command("start")["phase"] == "recording"
        assert job.done not in app.job.predecessors
        app.command("stop")
        finish(app)
        assert len(app.delivery.calls) == 2
    job.audio_path.unlink()


def test_cancel_after_recognition_does_not_enroll_or_leave_a_transcribing_row(monkeypatch):
    app = make_controller()
    resolve = recording_finalizer.routed_config

    def cancel_before_review(job):
        app.command("cancel", id=job.item_id)
        return resolve(job)

    monkeypatch.setattr(recording_finalizer, "routed_config", cancel_before_review)
    monkeypatch.setattr(
        PronunciationStore, "enroll", lambda *_: pytest.fail("Cancelled audio was enrolled")
    )
    item_id = app.command("start", mode="pronunciation", label="MouthKeys")["id"]
    app.command("stop")
    finish(app)

    row = app.history.get(item_id)
    assert row["status"] == "cancelled"
    assert row["raw_text"] == ""
    assert app.status()["assistant_proposal"] is None
    assert not app.job.audio_path.exists()


def test_assistant_result_returned_after_cancel_is_never_published(monkeypatch):
    app = make_controller()

    def late_proposal(job, *_):
        app.command("cancel", id=job.item_id)
        return {"mode": "command", "text": "Too late"}

    monkeypatch.setattr(app.recording_assistant, "propose", late_proposal)
    item_id = app.command("start", mode="command")["id"]
    app.command("stop")
    finish(app)

    assert app.history.get(item_id)["status"] == "cancelled"
    assert app.status()["assistant_proposal"] is None
    assert app.status()["preview"] == ""
    assert not app.delivery.calls


@pytest.mark.parametrize("transcript", ["", "Earlier transcript"])
def test_old_recognition_cannot_clear_current_preview_or_post_old_routing_warning(
    monkeypatch, transcript
):
    app = make_controller()
    app.transcriber.text = transcript
    app.transcriber.release.clear()

    def warning(job):
        return job.config, "Configuration problem in earlier job"

    monkeypatch.setattr(recording_finalizer, "routed_config", warning)
    old = app.command("start")["id"]

    def verify_durable_older_transcript(text, *_):
        assert app.history.get(old)["text"] == text
        return "copied"

    monkeypatch.setattr(app.delivery, "deliver", verify_durable_older_transcript)
    app.command("stop")
    old_worker = app.worker
    try:
        assert app.transcriber.entered.wait(2)
        current = app.command("start")["id"]
        with app.lock:
            app.state.preview_text = "Live preview for newer recording"
    finally:
        app.transcriber.release.set()
        old_worker.join(3)
    assert not old_worker.is_alive()

    status = app.status()
    assert status["id"] == current
    assert status["phase"] == "recording"
    assert status["preview"] == "Live preview for newer recording"
    assert status["notice"] is None
    assert app.history.get(old)["status"] == ("copied" if transcript else "failed")
    app.command("cancel")
    finish(app)


def test_logout_during_delivery_leaves_interrupted_history_untouched(monkeypatch):
    app = make_controller()
    deliver = app.delivery.deliver

    def logout_after_delivery(*args):
        result = deliver(*args)
        app.interrupt()
        return result

    monkeypatch.setattr(app.delivery, "deliver", logout_after_delivery)
    item_id = app.command("start")["id"]
    app.command("stop")
    app.worker.join(3)
    assert not app.worker.is_alive()
    assert app.job.done.is_set()
    assert app.status()["phase"] == "interrupted"
    row = app.history.get(item_id)
    assert row["status"] == "failed"
    assert row["error"] == "Interrupted; audio discarded"
    assert row["text"] == app.transcriber.text


def test_retiring_older_job_cannot_replace_current_notice(monkeypatch):
    app = make_controller()
    started, release = Event(), Event()
    deliver, unlink = app.delivery.deliver, Path.unlink

    def deliver_then_wait(*args):
        result = deliver(*args)
        started.set()
        assert release.wait(3)
        return result

    monkeypatch.setattr(app.delivery, "deliver", deliver_then_wait)
    app.command("start")
    old = app.job
    app.command("stop")
    worker = app.worker

    def fail_private_cleanup(path, *args, **kwargs):
        if path == old.audio_path:
            raise OSError("Earlier recording could not be removed")
        return unlink(path, *args, **kwargs)

    try:
        assert started.wait(2)
        app.command("start")
        app.report_error("New recording notice")
        monkeypatch.setattr(Path, "unlink", fail_private_cleanup)
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert old.done.is_set()
    assert app.status()["notice"]["message"] == "New recording notice"
    assert app.status()["phase"] == "recording"
    app.command("cancel")
    finish(app)
    unlink(old.audio_path)


def test_idle_command_cannot_start_while_cleanup_still_owns_prior_job(monkeypatch):
    app = make_controller()
    entered, release = Event(), Event()

    def wait_to_retire(*_):
        entered.set()
        assert release.wait(3)

    monkeypatch.setattr(recording_finalizer, "reset_cancelled_stream", wait_to_retire)
    app.command("start")
    app.command("stop")
    try:
        assert entered.wait(2)
        assert app.status()["phase"] == "idle"
        with pytest.raises(ValueError, match=r"Busy \(queued\)"):
            app.command("warmup")
    finally:
        release.set()
        finish(app)


@pytest.mark.parametrize("fails", [False, True])
def test_logout_during_capture_stop_never_decodes_or_overwrites_interrupted_state(
    monkeypatch, fails
):
    app = make_controller()
    stopping, logging_out, release = Event(), Event(), Event()
    app.command("start")
    job = app.job
    wait = app.capture.stop_done.wait

    def blocked_stop():
        stopping.set()
        assert release.wait(3)
        if fails:
            raise OSError("Capture stopped with an error")

    def wait_for_capture(timeout):
        logging_out.set()
        return wait(timeout)

    monkeypatch.setattr(app.recorder, "stop", blocked_stop)
    monkeypatch.setattr(app.capture.stop_done, "wait", wait_for_capture)
    app.command("stop")
    logout = Thread(target=app.interrupt, daemon=True)
    try:
        assert stopping.wait(2)
        logout.start()
        assert logging_out.wait(2)
    finally:
        release.set()
        app.worker.join(3)
        if logout.ident is not None:
            logout.join(3)
    assert not logout.is_alive()
    assert not app.worker.is_alive()
    assert job.done.is_set()
    assert app.status()["phase"] == "interrupted"
    assert app.history.get(job.item_id)["error"] == "Interrupted; audio discarded"
    assert not app.transcriber.entered.is_set()


def test_cancel_before_any_recording_is_a_noop_for_history():
    app = make_controller()
    assert app.command("cancel")["phase"] == "idle"
    assert app.history.all_rows() == []


def test_logout_during_model_warmup_cannot_restore_idle_status(monkeypatch):
    app = make_controller()
    entered, release = Event(), Event()

    def wait_for_warmup(_config):
        entered.set()
        assert release.wait(3)

    monkeypatch.setattr(app.transcriber, "load", wait_for_warmup)
    app.command("warmup")
    try:
        assert entered.wait(2)
        app.interrupt()
    finally:
        release.set()
        app.worker.join(3)
    assert not app.worker.is_alive()
    assert app.status()["phase"] == "interrupted"


@pytest.mark.parametrize("spoken", [False, True])
def test_tick_only_propagates_stop_request_failures_for_spoken_send(monkeypatch, spoken):
    app = make_controller()
    app.command("start")
    app.recorder.ended = True

    def fail_request(*_args, **_kwargs):
        raise RuntimeError("Cannot stop yet")

    monkeypatch.setattr(app.live_preview, "should_stop", lambda *_args, **_kwargs: spoken)
    monkeypatch.setattr(app.capture, "request_stop", fail_request)
    if spoken:
        with pytest.raises(RuntimeError, match="Cannot stop yet"):
            app.tick()
    else:
        app.tick()
    assert app.status()["phase"] == "recording"
    app.command("cancel")
    finish(app)
