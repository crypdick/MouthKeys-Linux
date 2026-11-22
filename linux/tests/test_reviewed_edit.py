import threading
import time

import pytest

from mouthkeys import assistance, delivery, desktop_input
from mouthkeys.config import Config
from mouthkeys.recording_edit import EditReplacement
from mouthkeys.service import Controller
from mouthkeys.storage import History

from .pipeline_support import FakeDelivery, FakeRecorder, FakeTranscriber, finish


def edit_controller():
    history = History()
    app = Controller(history, FakeTranscriber("Shorten"), FakeDelivery(history), FakeRecorder)
    target = {
        "window": "123",
        "field": {
            "field_id": "bus|field",
            "text": "Before original after",
            "selection_start": 7,
            "selection_end": 15,
            "editable": True,
            "protected": False,
            "complete": True,
        },
    }
    app.delivery.capture_target = lambda _: target
    return app, target


def captured_edit(monkeypatch):
    Config(ai_enabled="on", ai_model="test").save()
    app, target = edit_controller()
    monkeypatch.setattr(assistance, "request_completion", lambda *a, **kw: "Edited")
    app.command("start", mode="edit")
    app.command("stop")
    finish(app)
    assert not app.delivery.calls
    assert app.status()["assistant_proposal"]["can_accept_edit"]
    return app, target


@pytest.mark.parametrize("return_to_target", [False, True])
def test_review_accepts_exact_captured_voice_edit_once_without_state_lock(
    monkeypatch, return_to_target
):
    app, target = captured_edit(monkeypatch)
    calls = []

    def replace_selection(text, config, captured, *, cancel_event):
        statuses = []
        worker = threading.Thread(target=lambda: statuses.append(app.status()))
        worker.start()
        worker.join(1)
        assert not worker.is_alive(), "Desktop adapter must not own controller state lock"
        calls.append((text, config.return_to_start, captured, cancel_event))
        return "insert_requested"

    app.delivery.replace_selection = replace_selection
    result = app.command("accept_edit", id=app.item_id, return_to_target=return_to_target)
    assert calls == [("Edited", "on" if return_to_target else "off", target, app.job.cancel_event)]
    assert result["assistant_proposal"]["can_accept_edit"] is False
    assert "do not retry" in result["message"]
    with pytest.raises(ValueError, match="No current captured"):
        app.command("accept_edit", id=app.item_id, return_to_target=return_to_target)


def test_review_rejects_stale_id_cancelled_and_missing_capture(monkeypatch):
    app, _ = captured_edit(monkeypatch)
    app.delivery.replace_selection = lambda *a, **kw: pytest.fail("Unexpected desktop delivery")
    with pytest.raises(ValueError, match="No current captured"):
        app.command("accept_edit", id="older-job")
    with pytest.raises(ValueError, match="explicit boolean"):
        app.command("accept_edit", id=app.item_id, return_to_target=1)
    app.command("cancel")
    with pytest.raises(ValueError, match="No current captured"):
        app.command("accept_edit", id=app.item_id)


@pytest.mark.parametrize("interrupting_action", ["cancel", "start"])
def test_cancel_or_supersede_review_waiting_for_model_lock(monkeypatch, interrupting_action):
    app, _ = captured_edit(monkeypatch)
    app.delivery.replace_selection = lambda *a, **kw: pytest.fail("Cancelled edit was delivered")
    app.model_lock.acquire()
    results = []
    worker = threading.Thread(
        target=lambda: results.append(app.command("accept_edit", id=app.item_id))
    )
    worker.start()
    try:
        deadline = time.monotonic() + 1
        while app.status()["phase"] != "queued" and time.monotonic() < deadline:
            time.sleep(0.001)
        assert app.status()["phase"] == "queued"
        app.command(interrupting_action)
    finally:
        app.model_lock.release()
        worker.join(2)
        if app.status()["phase"] == "recording":
            app.command("cancel")
            finish(app)
    assert not worker.is_alive()
    expected = "recording" if interrupting_action == "start" else "idle"
    assert results[0]["phase"] == expected


@pytest.mark.parametrize("boundary", ["provider", "handoff", "desktop"])
def test_new_recording_supersedes_automatic_edit_before_insertion(monkeypatch, boundary):
    Config(ai_enabled="on", ai_model="test", edit_delivery="replace").save()
    app, _ = edit_controller()
    entered, release = threading.Event(), threading.Event()
    inserted = []

    def propose(*_args, **_options):
        if boundary == "provider":
            entered.set()
            assert release.wait(3)
        return {"mode": "edit", "text": "Edited"}

    def replace_selection(text, _config, _target, *, cancel_event):
        if boundary == "desktop":
            entered.set()
            assert release.wait(3)
        if cancel_event.is_set():
            return "cancelled"
        inserted.append(text)
        return "pasted"

    if boundary == "handoff":
        apply = EditReplacement.apply

        def paused_apply(edit, adapter):
            entered.set()
            assert release.wait(3)
            return apply(edit, adapter)

        monkeypatch.setattr(EditReplacement, "apply", paused_apply)

    monkeypatch.setattr(assistance, "propose", propose)
    app.delivery.replace_selection = replace_selection
    first = app.command("start", mode="edit")["id"]
    app.command("stop")
    worker = app.worker
    try:
        assert entered.wait(2)
        assert app.command("start")["id"] != first
    finally:
        release.set()
        worker.join(3)
        if app.status()["phase"] == "recording":
            app.command("cancel")
            finish(app)
    assert not worker.is_alive()
    assert not inserted
    row = next(row for row in app.history.rows() if row["id"] == first)
    assert row["text"] == "Edited"
    assert row["status"] == "transcribed"
    assert row["was_ai_processed"]


@pytest.mark.parametrize(
    ("field_patch", "expected"),
    [
        (None, ""),
        ({"protected": True}, ""),
        ({"complete": False}, ""),
        ({"selection_start": "bad"}, ""),
        ({"selection_end": False}, ""),
        ({"selection_end": 7}, ""),
        ({"selection_start": -1}, ""),
        ({"selection_end": 22}, ""),
        ({"text": None}, ""),
        pytest.param(
            {"text": "x" * 100001, "selection_start": 0, "selection_end": 100001},
            "",
            id="oversize",
        ),
        ({"editable": False}, "original"),
        ({"terminal": True}, "original"),
        ({"field_id": ""}, "original"),
    ],
)
def test_edit_uses_readable_capture_but_only_replaces_verified_selection(
    monkeypatch, field_patch, expected
):
    Config(ai_enabled="on", ai_model="test", edit_delivery="replace").save()
    app, target = edit_controller()
    target["field"] = {**target["field"], **field_patch} if field_patch is not None else None
    selected = []

    def propose(_text, _config, mode, selection, **_options):
        selected.append(selection)
        return {"mode": mode, "text": "Generated"}

    monkeypatch.setattr(assistance, "propose", propose)
    app.delivery.replace_selection = lambda *a, **kw: pytest.fail("Unverified target")
    app.command("start", mode="edit")
    app.command("stop")
    finish(app)
    assert selected == [expected]
    assert not app.status()["assistant_proposal"]["can_accept_edit"]


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        ("pasted", "Edit inserted"),
        ("copied", "Edit copied; captured selection changed or unavailable"),
        ("paste_requested", "Edit insertion requested; do not retry automatically"),
    ],
)
def test_automatic_edit_commits_before_desktop_and_reports_outcome(monkeypatch, outcome, message):
    Config(ai_enabled="on", ai_model="test", edit_delivery="replace").save()
    app, target = edit_controller()
    monkeypatch.setattr(assistance, "request_completion", lambda *a, **kw: "Edited")

    def replace_selection(text, _config, captured, *, cancel_event):
        assert text == "Edited"
        assert captured == target
        assert not cancel_event.is_set()
        row = app.history.rows()[0]
        assert (row["status"], row["text"]) == ("transcribed", "Edited")
        return outcome

    app.delivery.replace_selection = replace_selection
    app.command("start", mode="edit")
    app.command("stop")
    finish(app)
    assert app.history.rows()[0]["status"] == outcome
    assert app.status()["message"] == message
    assert not app.status()["assistant_proposal"]["can_accept_edit"]
    with pytest.raises(ValueError, match="No current captured"):
        app.command("accept_edit", id=app.item_id)


def test_logout_cancels_reviewed_edit_before_desktop_insertion(monkeypatch):
    app, _ = captured_edit(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    cancelled = []

    def replace_selection(_text, _config, _target, *, cancel_event):
        entered.set()
        assert release.wait(3)
        cancelled.append(cancel_event.is_set())
        return "cancelled" if cancelled[-1] else "pasted"

    app.delivery.replace_selection = replace_selection
    worker = threading.Thread(target=lambda: app.command("accept_edit", id=app.item_id))
    worker.start()
    try:
        assert entered.wait(2)
        app.interrupt()
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert cancelled == [True]
    assert app.status()["phase"] == "interrupted"
    assert app.history.rows()[0]["status"] == "transcribed"


def test_invalid_review_proposal_does_not_claim_or_deliver(monkeypatch):
    app, _ = captured_edit(monkeypatch)
    app.assistant_proposal["text"] = None
    with pytest.raises(ValueError, match="No current captured"):
        app.command("accept_edit", id=app.item_id)
    assert not app.job.edit_consumed


def test_review_cannot_retry_if_desktop_adapter_lacks_replacement(monkeypatch):
    app, _ = captured_edit(monkeypatch)
    with pytest.raises(TypeError, match="Desktop adapter cannot replace"):
        app.command("accept_edit", id=app.item_id)
    assert app.status()["phase"] == "idle"
    assert not app.status()["assistant_proposal"]["can_accept_edit"]


@pytest.mark.parametrize("boundary", ["prepare", "copy", "field"])
def test_edit_cancellation_at_desktop_boundaries_never_sends_key(monkeypatch, boundary):
    adapter = delivery.Delivery()
    cancelled = threading.Event()
    target = {"field": {"selection_start": 0, "selection_end": 3}}
    calls = []
    monkeypatch.setattr(desktop_input, "session", lambda: "x11")

    def prepare(*_):
        if boundary == "prepare":
            cancelled.set()
        return True

    def copy(_):
        calls.append("copy")
        if boundary == "copy":
            cancelled.set()

    def field(*_, **__):
        if boundary == "field":
            cancelled.set()
        return True

    monkeypatch.setattr(adapter, "prepare_target", prepare)
    monkeypatch.setattr(adapter, "copy", copy)
    monkeypatch.setattr(adapter, "_field_current", field)
    monkeypatch.setattr(adapter, "paste_key", lambda _: pytest.fail("Cancelled key was sent"))
    config = Config(delivery="x11", restore_clipboard="off")
    assert (
        adapter.replace_selection("Edited", config, target, cancel_event=cancelled) == "cancelled"
    )
    assert calls == ([] if boundary == "prepare" else ["copy"])
