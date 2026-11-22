"""Race regressions for the transition from capture to final inference."""

import threading

import pytest

from mouthkeys import recording_inference
from mouthkeys.config import Config

from .pipeline_support import make_controller


@pytest.mark.parametrize(
    ("action", "silent"), [("cancel", True), ("logout", True), ("cancel", False)]
)
def test_cancel_or_logout_takes_precedence_over_late_silence_result(monkeypatch, action, silent):
    Config(skip_silent_recordings="on").save()
    app = make_controller()
    entered, release = threading.Event(), threading.Event()

    def silence(_path):
        entered.set()
        assert release.wait(3)
        return silent

    monkeypatch.setattr(recording_inference, "is_short_silence", silence)
    item_id = app.command("start")["id"]
    app.command("stop")
    job = app.job
    try:
        assert entered.wait(2)
        if action == "cancel":
            app.command("cancel")
        else:
            app.interrupt()
    finally:
        release.set()
        app.worker.join(3)
    assert not app.worker.is_alive()
    assert job.done.is_set()
    assert not job.audio_path.exists()
    assert not app.transcriber.entered.is_set()
    assert not app.delivery.calls
    row = app.history.get(item_id)
    if action == "cancel":
        assert (row["status"], app.status()["message"]) == ("cancelled", "Cancelled")
    else:
        assert (row["status"], row["error"]) == ("failed", "Interrupted; audio discarded")


def test_cancelled_queue_item_skips_audio_validation_after_predecessor(monkeypatch):
    app = make_controller()
    checked = []
    monkeypatch.setattr(recording_inference, "check_audio", checked.append)
    app.delivery.deliver = lambda *_: "copied"
    app.transcriber.release.clear()
    prior_id = app.command("start")["id"]
    prior = app.job
    timed_wait = threading.Event()
    wait = prior.done.wait

    def observe_wait(timeout):
        finished = wait(timeout)
        if not finished:
            timed_wait.set()
        return finished

    monkeypatch.setattr(prior.done, "wait", observe_wait)
    app.command("stop")
    prior_worker = app.worker
    later_worker = None
    try:
        assert app.transcriber.entered.wait(2)
        later_id = app.command("start")["id"]
        app.command("stop")
        later_worker = app.worker
        assert timed_wait.wait(2)
        assert app.status()["phase"] == "queued"
        app.command("cancel")
    finally:
        app.transcriber.release.set()
        prior_worker.join(3)
        if later_worker is not None:
            later_worker.join(3)
    assert not prior_worker.is_alive()
    assert not later_worker.is_alive()
    assert checked == [prior.audio_path]
    assert app.history.get(prior_id)["status"] == "copied"
    assert app.history.get(later_id)["status"] == "cancelled"


def test_cancel_after_audio_validation_does_not_start_silence_probe(monkeypatch):
    Config(skip_silent_recordings="on").save()
    app = make_controller()
    entered, release = threading.Event(), threading.Event()

    def validate(_path):
        entered.set()
        assert release.wait(3)

    monkeypatch.setattr(recording_inference, "check_audio", validate)
    monkeypatch.setattr(
        recording_inference,
        "is_short_silence",
        lambda _: pytest.fail("Cancelled audio reached silence probe"),
    )
    item_id = app.command("start")["id"]
    app.command("stop")
    try:
        assert entered.wait(2)
        app.command("cancel")
    finally:
        release.set()
        app.worker.join(3)
    assert not app.worker.is_alive()
    assert not app.transcriber.entered.is_set()
    assert app.history.get(item_id)["status"] == "cancelled"


def test_unusable_alignment_preserves_recognized_text(monkeypatch):
    Config(pronunciation_matching="on").save()
    app = make_controller()

    class BrokenAlignment:
        def __deepcopy__(self, _memo):
            raise ValueError("Adapter diagnostic")

    app.transcriber.last_alignment = BrokenAlignment()
    monkeypatch.setattr(
        "mouthkeys.recording_transcript.PronunciationStore.correct", lambda text, _audio: text
    )
    item_id = app.command("start")["id"]
    app.command("stop")
    app.worker.join(3)
    assert not app.worker.is_alive()
    row = app.history.get(item_id)
    assert row["text"] == row["raw_text"] == app.transcriber.text
    assert row["status"] == "copied"
    assert app.status()["message"].endswith("Word timing metadata unavailable")


def test_logout_releases_queued_worker_without_waiting_for_model(monkeypatch):
    app = make_controller()
    app.transcriber.release.clear()
    app.delivery.deliver = lambda *_: pytest.fail("Delivered after logout")
    app.command("start")
    waiting = threading.Event()
    wait = app.job.done.wait

    def observe_wait(timeout):
        waiting.set()
        return wait(timeout)

    monkeypatch.setattr(app.job.done, "wait", observe_wait)
    app.command("stop")
    prior = app.worker
    later = None
    try:
        assert app.transcriber.entered.wait(2)
        item_id = app.command("start")["id"]
        app.command("stop")
        later = app.worker
        assert waiting.wait(2)
        assert app.status()["phase"] == "queued"
        app.interrupt()
        later.join(2)
        assert not later.is_alive()
        assert prior.is_alive()
        assert app.history.get(item_id)["status"] == "failed"
    finally:
        app.transcriber.release.set()
        prior.join(3)
        if later is not None:
            later.join(3)
    assert not prior.is_alive()
