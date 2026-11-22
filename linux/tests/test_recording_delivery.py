import threading
import wave

import pytest

from mouthkeys import desktop_input, recording_cleanup, recording_delivery
from mouthkeys.audio import is_short_silence
from mouthkeys.config import Config

from .pipeline_support import finish, make_controller


@pytest.mark.parametrize(
    ("width", "frames", "sample", "silent"),
    [
        (2, 64000, -1, True),
        (2, 64001, 0, False),
        (2, 0, 0, False),
        (2, 10, -32768, False),
        (2, 10, 2, False),
        (1, 10, 0, False),
    ],
)
def test_silence_check_requires_bounded_near_zero_pcm(tmp_path, width, frames, sample, silent):
    path = tmp_path / "capture.wav"
    with wave.open(str(path), "wb") as output:
        output.setframerate(16000)
        output.setnchannels(1)
        output.setsampwidth(width)
        output.writeframes(sample.to_bytes(width, "little", signed=True) * frames)
    assert is_short_silence(path) is silent


@pytest.mark.parametrize(
    ("target", "guard", "reason"),
    [
        (None, "field_then_window", "No destination was captured."),
        ("123", "field", "Could not identify the focused field."),
        ("123", "field_then_window", "Could not verify the original destination"),
    ],
)
def test_automatic_copy_reports_the_actual_missing_destination_evidence(
    monkeypatch, target, guard, reason
):
    Config(delivery="x11", destination_guard=guard).save()
    monkeypatch.setattr(desktop_input, "automatic_delivery_warning", lambda _: None)
    app = make_controller()
    app.delivery.capture_target = lambda _: target
    app.command("start")
    app.command("stop")
    finish(app)
    assert app.history.rows()[0]["status"] == "copied"
    assert app.status()["notice"]["kind"] == "paste"
    assert reason in app.status()["notice"]["message"]


def test_guarded_send_only_reserves_one_key_without_inserting_empty_text(monkeypatch):
    Config(delivery="x11", spoken_send="on").save()
    monkeypatch.setattr(recording_delivery, "SEND_REVIEW_SECONDS", 0)
    app = make_controller()
    app.transcriber.text = "send it"
    app.delivery.send_key_guarded = True
    reservations = []

    def send(_config, target, *, before_send, allow_without_insert):
        assert target == "123"
        assert allow_without_insert
        reservations.extend([before_send(), before_send()])
        return reservations[0]

    app.delivery.send_key = send
    app.command("start")
    app.command("stop")
    finish(app)
    assert reservations == [True, False]
    assert app.history.rows()[0]["status"] == "send_requested"
    assert not app.delivery.calls
    assert not app.job.send_pending


@pytest.mark.parametrize("status", ["pasted", "transcribed"])
def test_send_without_supported_insertion_or_key_adapter_does_not_claim_success(
    monkeypatch, status
):
    Config(delivery="x11", spoken_send="on").save()
    monkeypatch.setattr(recording_delivery, "SEND_REVIEW_SECONDS", 0)
    app = make_controller()
    app.transcriber.text = "Hello send it"
    app.delivery.deliver = lambda *_: status
    app.command("start")
    app.command("stop")
    finish(app)
    row = app.history.rows()[0]
    assert row["status"] == status
    assert row["text"] == "Hello"
    assert row["error"] == (
        "Spoken Send skipped: destination could not be verified" if status == "pasted" else ""
    )
    assert not app.job.send_dispatched


def test_cancel_wakes_the_spoken_send_review_window(monkeypatch):
    Config(delivery="x11", spoken_send="on").save()
    monkeypatch.setattr(recording_delivery, "SEND_REVIEW_SECONDS", 5)
    app = make_controller()
    app.transcriber.text = "Hello send it"
    app.delivery.deliver = lambda *_: "paste_requested"
    app.delivery.send_key = lambda *_: pytest.fail("Cancelled Send cannot enqueue a key")
    app.command("start")
    job = app.job
    entered = threading.Event()
    wait = job.cancel_event.wait

    def observed_wait(timeout):
        entered.set()
        return wait(timeout)

    monkeypatch.setattr(job.cancel_event, "wait", observed_wait)
    app.command("stop")
    try:
        assert entered.wait(2)
        assert job.send_pending
        assert "Send cancelled" in app.command("cancel")["message"]
        assert job.done.wait(2), "Cancellation should wake the review wait"
    finally:
        job.cancel_event.set()
        finish(app)
    row = app.history.rows()[0]
    assert row["status"] == "paste_requested"
    assert row["text"] == "Hello"
    assert not job.send_pending
    assert not job.send_dispatched


def test_streaming_cleanup_does_not_publish_or_deliver_after_cancellation(monkeypatch):
    Config(ai_enabled="on", ai_model="model", ai_streaming="on").save()
    app = make_controller()
    received, release = threading.Event(), threading.Event()

    def process(text, _config, *, on_delta, cancel_event):
        assert app.history.rows()[0]["raw_text"] == text
        on_delta("Partial cleanup")
        received.set()
        assert release.wait(3)
        assert cancel_event.is_set()
        on_delta(" Stale text")
        return "No longer wanted"

    monkeypatch.setattr(recording_cleanup, "process", process)
    app.command("start")
    app.command("stop")
    try:
        assert received.wait(2)
        assert app.status()["preview"] == "Partial cleanup"
        app.command("cancel")
    finally:
        release.set()
        finish(app)
    assert app.status()["preview"] == ""
    assert app.history.rows()[0]["raw_text"] == ""
    assert app.history.rows()[0]["status"] == "cancelled"
    assert not app.delivery.calls


def test_cleanup_for_an_older_job_does_not_overwrite_a_new_recording_preview(monkeypatch):
    Config(ai_enabled="on", ai_model="model", ai_streaming="on").save()
    app = make_controller()
    entered, publish_late, published = (threading.Event() for _ in range(3))

    def process(_text, _config, *, on_delta, cancel_event):
        on_delta("Earlier recording")
        entered.set()
        assert publish_late.wait(3)
        assert not cancel_event.is_set()
        on_delta(" stale update")
        published.set()
        return "Earlier transcript"

    monkeypatch.setattr(recording_cleanup, "process", process)
    first = app.command("start")["id"]
    app.command("stop")
    old_worker = app.worker
    try:
        assert entered.wait(2)
        assert app.status()["preview"] == "Earlier recording"
        assert app.command("start")["id"] != first
        publish_late.set()
        assert published.wait(2)
        assert app.status()["phase"] == "recording"
        assert app.status()["preview"] == ""
    finally:
        publish_late.set()
        old_worker.join(3)
        if app.status()["phase"] == "recording":
            app.command("cancel")
            finish(app)
    assert not old_worker.is_alive()
    first_row = next(row for row in app.history.rows() if row["id"] == first)
    assert first_row["text"] == "Earlier transcript"


def test_prompt_routing_failure_delivers_the_local_transcript_without_contacting_a_provider(
    monkeypatch,
):
    Config(ai_enabled="on", ai_model="model", lowercase_first="on").save()
    app = make_controller()
    app.transcriber.text = "Raw transcript"

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("Route unavailable")

    def unexpected_provider(*_args, **_kwargs):
        pytest.fail("A failed prompt route must not make an AI request")

    monkeypatch.setattr("mouthkeys.routing.select_prompt", unavailable)
    monkeypatch.setattr(recording_cleanup, "process", unexpected_provider)
    app.command("start")
    app.command("stop")
    finish(app)
    row = app.history.rows()[0]
    assert row["raw_text"] == "Raw transcript"
    assert row["text"] == app.delivery.calls[0][0] == "raw transcript"
    assert row["was_ai_processed"] == 0
    assert "Prompt routing failed (Route unavailable)" in row["ai_processing_error"]
    assert app.status()["notice"]["kind"] == "ai"
