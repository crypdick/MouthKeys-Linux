import threading
import time

import pytest
from test_pipeline import FakeDelivery, FakeRecorder, FakeTranscriber, finish

from mouthkeys import assistance, delivery
from mouthkeys.config import Config
from mouthkeys.service import Controller
from mouthkeys.storage import History


def captured_edit(monkeypatch):
    Config(ai_enabled="on", ai_model="test").save()
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
    monkeypatch.setattr(assistance, "request_completion", lambda *a, **kw: "Edited")
    app.command("start", mode="edit")
    app.command("stop")
    finish(app)
    assert not app.delivery.calls
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
    app.command("cancel")
    with pytest.raises(ValueError, match="No current captured"):
        app.command("accept_edit", id=app.item_id)


def test_cancel_while_review_waits_for_mutex_suppresses_delivery(monkeypatch):
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
        app.command("cancel")
    finally:
        app.model_lock.release()
    worker.join(1)
    assert not worker.is_alive()
    assert results[0]["phase"] == "idle"
    assert app.status()["phase"] == "idle"


@pytest.mark.parametrize("boundary", ["prepare", "copy", "field"])
def test_edit_cancellation_at_desktop_boundaries_never_sends_key(monkeypatch, boundary):
    adapter = delivery.Delivery()
    cancelled = threading.Event()
    target = {"field": {"selection_start": 0, "selection_end": 3}}
    calls = []
    monkeypatch.setattr(delivery, "session", lambda: "x11")

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
