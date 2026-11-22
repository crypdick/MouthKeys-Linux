"""Headless manual copy and paste behavior when recording state changes."""

import threading
import uuid
from types import SimpleNamespace

import pytest

from mouthkeys import recording_delivery, service

from .pipeline_support import make_controller


def add_history(app, text):
    item = str(uuid.uuid4())
    app.history.begin(item)
    app.history.update(item, "copied" if text else "failed", text)
    return item


@pytest.mark.parametrize(
    ("action", "adapter_fails"),
    [("paste", False), ("copy", False), ("copy", True)],
)
def test_replay_does_not_paste_or_overwrite_state_after_new_recording(action, adapter_fails):
    app = make_controller()
    add_history(app, "Earlier transcript")
    copying, release = threading.Event(), threading.Event()
    statuses, errors = [], []

    def copy(text):
        assert text == "Earlier transcript"
        copying.set()
        assert release.wait(3)
        if adapter_fails:
            raise RuntimeError("Clipboard unavailable")

    app.delivery.copy = copy

    def replay():
        try:
            statuses.append(app.command(action))
        except RuntimeError as exc:
            errors.append(str(exc))

    worker = threading.Thread(target=replay)
    worker.start()
    try:
        try:
            assert copying.wait(2)
            started = app.command("start")
            assert started["phase"] == "recording"
        finally:
            release.set()
            worker.join(3)
        assert not worker.is_alive()
        assert not app.delivery.calls
        status = app.status()
        assert status["phase"] == "recording"
        assert status["id"] == started["id"]
        assert status["message"] == started["message"]
        if adapter_fails:
            assert errors == ["Clipboard unavailable"]
        else:
            assert statuses[0]["phase"] == "recording"
    finally:
        app.interrupt()


def test_explicit_empty_history_item_reports_itself_without_latest_transcript():
    app = make_controller()
    empty = add_history(app, "")
    with pytest.raises(ValueError, match="No transcript for this entry"):
        app.command("copy", id=empty)
    assert app.status()["phase"] == "idle"
    assert not app.delivery.calls


def test_replay_uses_explicit_history_entry_before_latest():
    app = make_controller()
    earlier = add_history(app, "Selected transcript")
    add_history(app, "Newer transcript")
    assert app.command("paste", id=earlier)["message"] == "Paste requested"
    assert app.delivery.calls == ["Selected transcript", "paste"]
    assert app.command("copy")["message"] == "Copied"
    assert app.delivery.calls[-1] == "Newer transcript"


def test_replay_rejects_bad_history_id_and_busy_or_interrupted_service():
    app = make_controller()
    with pytest.raises(ValueError, match="Invalid history entry ID"):
        app.command("copy", id=["not-an-id"])
    app.command("start")
    try:
        with pytest.raises(ValueError, match=r"Busy \(recording\)"):
            app.command("paste")
    finally:
        app.interrupt()
    with pytest.raises(ValueError, match="Service is shutting down"):
        app.command("copy")
    assert not app.delivery.calls


def test_replay_failure_returns_service_to_idle_without_losing_transcript():
    app = make_controller()
    item = add_history(app, "Prior transcript")

    def fail(_text):
        raise RuntimeError("Clipboard unavailable")

    app.delivery.copy = fail
    with pytest.raises(RuntimeError, match="Clipboard unavailable"):
        app.command("copy")
    assert app.status()["phase"] == "idle"
    assert app.status()["message"] == "Clipboard unavailable"
    assert app.history.get(item)["text"] == "Prior transcript"


@pytest.mark.parametrize("boundary", ["before-copy", "before-paste"])
def test_replay_rechecks_ownership_at_desktop_boundaries(monkeypatch, boundary):
    app = make_controller()
    add_history(app, "Prior transcript")
    reached, release = threading.Event(), threading.Event()

    def wait():
        reached.set()
        assert release.wait(3)

    if boundary == "before-copy":
        replay = service.replay_transcript

        def pause(*args):
            wait()
            return replay(*args)

        monkeypatch.setattr(service, "replay_transcript", pause)
    else:
        monkeypatch.setattr(recording_delivery, "time", SimpleNamespace(sleep=lambda _: wait()))

    worker = threading.Thread(target=lambda: app.command("paste"))
    worker.start()
    try:
        try:
            assert reached.wait(2)
            started = app.command("start")
        finally:
            release.set()
            worker.join(3)
        assert not worker.is_alive()
        assert app.delivery.calls == ([] if boundary == "before-copy" else ["Prior transcript"])
        assert app.status()["id"] == started["id"]
        assert app.status()["phase"] == "recording"
    finally:
        app.interrupt()
