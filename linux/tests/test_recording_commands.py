import threading

import pytest

from mouthkeys.config import Config
from mouthkeys.service import Controller
from mouthkeys.storage import History

from .pipeline_support import FakeDelivery, FakeRecorder, FakeTranscriber, finish


def app(text="Speech"):
    history = History()
    return Controller(history, FakeTranscriber(text), FakeDelivery(history), FakeRecorder)


def test_stop_and_hold_release_do_not_affect_a_different_recording():
    controller = app()
    identity = controller.command("start", held=True)["id"]
    assert controller.command("release_hold", id="retired")["phase"] == "recording"
    assert controller.job.held_activation
    assert controller.command("stop", id="retired")["phase"] == "recording"
    assert controller.command("release_hold", id=identity)["phase"] == "recording"
    assert not controller.job.held_activation
    controller.command("stop", id=identity)
    finish(controller)
    assert controller.command("stop", id="retired")["phase"] == "idle"
    with pytest.raises(ValueError, match="Not recording"):
        controller.command("stop")


def test_status_remains_available_after_interrupt_but_warmup_is_rejected():
    controller = app()
    controller.interrupt()
    assert controller.command("release_hold", id="retired")["phase"] == "interrupted"
    assert controller.command("status")["phase"] == "interrupted"
    with pytest.raises(ValueError, match="Service is shutting down"):
        controller.command("warmup")


def test_cancel_cannot_claim_success_after_send_has_been_reserved():
    Config(delivery="x11", spoken_send="on").save()
    controller = app("Hello world send it")
    controller.delivery.deliver = lambda *_: "pasted"
    controller.delivery.send_key_guarded = True
    entered, release = threading.Event(), threading.Event()

    def send(_config, _target, *, before_send, allow_without_insert):
        assert not allow_without_insert
        assert before_send()
        entered.set()
        assert release.wait(2)
        return True

    controller.delivery.send_key = send
    controller.command("start")
    controller.command("stop")
    try:
        assert entered.wait(2)
        with pytest.raises(ValueError, match="Send already dispatched"):
            controller.command("cancel")
    finally:
        release.set()
        finish(controller)
    assert controller.history.rows()[0]["text"] == "Hello world"


def test_targeted_cancel_ignores_retired_recording():
    controller = app()
    identity = controller.command("start")["id"]
    assert controller.command("cancel", id="retired")["phase"] == "recording"
    assert not controller.job.cancel_event.is_set()
    controller.command("cancel", id=identity)
    finish(controller)
    assert controller.history.rows()[0]["status"] == "cancelled"
    assert not controller.delivery.calls
