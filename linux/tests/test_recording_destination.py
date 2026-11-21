from threading import Event
from types import SimpleNamespace

import pytest

from mouthkeys import accessibility, recording_destination, service
from mouthkeys.config import Config

from .pipeline_support import finish, make_controller


@pytest.mark.parametrize(
    "config",
    [
        Config(target_policy="start"),
        Config(return_to_start="on"),
    ],
)
def test_explicit_start_destination_is_not_replaced_at_stop(config):
    calls = []
    start = {"window": "at-start", "field": {"text": "private"}}
    delivery = SimpleNamespace(
        capture_target=lambda _: pytest.fail("Unexpected focus probe"),
        target_metadata=lambda target: calls.append(target) or {"app": "Editor"},
    )
    sample = recording_destination.snapshot_for_stop(
        delivery, config, "capture-focus", start_target=start, mode="dictation"
    )
    assert sample.automatic
    assert sample.target is start
    assert not calls  # Metadata is deferred until recording has stopped.
    assert sample.metadata(delivery) == {"app": "Editor"}
    assert calls == [start]


def test_private_plain_dictation_does_not_probe_optional_desktop_state(monkeypatch):
    def unexpected(*_args):
        pytest.fail("Unexpected optional desktop lookup")

    delivery = SimpleNamespace(capture_target=unexpected, target_metadata=unexpected)
    monkeypatch.setattr(accessibility, "read_context", unexpected)
    sample = recording_destination.snapshot_for_stop(
        delivery, Config(save_history="off"), None, start_target=None, mode="dictation"
    )
    assert sample.target is None
    assert not sample.automatic
    assert sample.context is None
    assert sample.metadata(delivery) is None


@pytest.mark.parametrize("mode", ["edit", "dictation"])
def test_optional_metadata_does_not_become_automatic_delivery_target(monkeypatch, mode):
    looks = []
    delivery = SimpleNamespace(
        focus=lambda: "metadata-only",
        target_metadata=lambda target: looks.append(target) or {"window": target},
    )
    monkeypatch.setattr(accessibility, "read_context", lambda: "before caret")
    config = Config(
        save_history="off",
        ai_enabled="off" if mode == "edit" else "on",
        space_between_dictations="on",
    )
    sample = recording_destination.snapshot_for_stop(
        delivery, config, None, start_target=None, mode=mode
    )
    assert sample.target is None
    assert not sample.automatic
    assert sample.context == "before caret"
    assert sample.metadata(delivery) == {"window": "metadata-only"}
    assert looks == ["metadata-only"]


def test_focus_or_context_failure_falls_back_without_changing_capture_destination(monkeypatch):
    tries = []

    def snapshot(_config):
        tries.append("snapshot")
        if len(tries) == 1:
            raise RuntimeError("Window vanished")
        return "metadata-only"

    def failure(*_args):
        raise RuntimeError("Stale desktop address")

    delivery = SimpleNamespace(capture_target=snapshot, target_metadata=failure)
    monkeypatch.setattr(accessibility, "read_context", failure)
    sample = recording_destination.snapshot_for_stop(
        delivery,
        Config(smart_capitalization="on"),
        "capture-focus",
        start_target=None,
        mode="dictation",
    )
    assert sample.automatic
    assert sample.target is None
    assert sample.metadata_target == "metadata-only"
    assert sample.context is None
    assert sample.metadata(delivery) is None


def test_unavailable_adapters_and_non_mapping_metadata_leave_no_desktop_state():
    args = (Config(), "capture-focus")
    sample = recording_destination.snapshot_for_stop(
        object(), *args, start_target=None, mode="dictation"
    )
    assert sample.target is None
    delivery = SimpleNamespace(target_metadata=lambda _: ["not a mapping"])
    updated = recording_destination.snapshot_for_stop(
        delivery, *args, start_target=None, mode="dictation"
    )
    assert updated.metadata(delivery) is None
    del delivery.target_metadata
    assert updated.metadata(delivery) is None


def test_focus_ack_timeout_prevents_paste_but_retains_metadata_after_capture(monkeypatch):
    Config(delivery="x11", target_policy="stop").save()
    app = make_controller()
    app.command("start")
    recorder = app.recorder
    entered, released = Event(), Event()
    events = []

    def focus(_config):
        assert not recorder.stopped
        entered.set()
        assert released.wait(2)
        events.append("focus")
        return "late-window"

    def metadata(target):
        assert recorder.stopped
        assert target == "late-window"
        events.append("metadata")
        return {"app": "Editor"}

    app.delivery.capture_target = focus
    app.delivery.target_metadata = metadata
    gate = service.FocusGate()

    def elapsed(*, timeout):
        assert timeout == 4
        assert entered.wait(1)
        return False

    monkeypatch.setattr(gate.ready, "wait", elapsed)
    monkeypatch.setattr(service, "FocusGate", lambda: gate)
    try:
        assert app.command("stop")["phase"] == "stopping"
        assert not gate.valid
        assert not recorder.stopped
    finally:
        released.set()
        finish(app)
    assert events == ["focus", "metadata"]
    assert app.history.rows()[0]["app"] == "Editor"
    assert app.delivery.calls[0][2] is None
