import threading
import time

import pytest
from test_pipeline import FakeRecorder

from mouthkeys.config import Config
from mouthkeys.service import Controller
from mouthkeys.storage import History


class OrderedTranscriber:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def transcribe(self, path, config):
        self.calls.append((path.stem, config.model))
        if len(self.calls) == 1:
            self.entered.set()
            assert self.release.wait(3)
        return f"{config.model}: {path.stem}"


class Delivery:
    def __init__(self, history):
        self.calls = []
        self.history = history

    def focus(self):
        return "123"

    def deliver(self, text, config, target):
        assert any(row["text"] == text for row in self.history.rows())
        self.calls.append((text, config.model, target))
        return "copied"


def app():
    history = History()
    return Controller(history, OrderedTranscriber(), Delivery(history), FakeRecorder)


def join(*workers):
    for worker in workers:
        worker.join(4)
        assert not worker.is_alive()


def test_back_to_back_capture_keeps_config_audio_order_and_visible_session_separate():
    Config(model="first", delivery="x11", save_audio="on").save()
    controller = app()
    first = controller.command("start")["id"]
    controller.command("stop")
    first_worker = controller.worker
    assert controller.transcriber.entered.wait(2)
    Config(model="second", delivery="clipboard", save_audio="on").save()
    second = controller.command("start")["id"]
    assert second != first
    assert controller.status()["phase"] == "recording"
    assert controller.status()["pending_jobs"] == 2
    controller.command("stop")
    second_worker = controller.worker
    assert controller.status()["id"] == second
    deadline = time.monotonic() + 2
    while controller.status()["phase"] == "stopping" and time.monotonic() < deadline:
        time.sleep(0.005)
    assert controller.status()["phase"] == "queued"
    assert len(controller.transcriber.calls) == 1
    controller.transcriber.release.set()
    join(first_worker, second_worker)
    assert controller.transcriber.calls == [(first, "first"), (second, "second")]
    assert [(model, target) for _, model, target in controller.delivery.calls] == [
        ("first", "123"),
        ("second", None),
    ]
    assert controller.status()["id"] == second
    assert controller.status()["pending_jobs"] == 0
    assert controller.status()["phase"] == "idle"
    assert controller.history.recording(first).exists()
    assert controller.history.recording(second).exists()


def test_cancel_newest_does_not_cancel_or_overwrite_previous_job():
    controller = app()
    first = controller.command("start")["id"]
    controller.command("stop")
    worker = controller.worker
    assert controller.transcriber.entered.wait(2)
    second = controller.command("start")["id"]
    controller.command("cancel")
    join(controller.worker)
    assert controller.status()["message"] == "Cancelled"
    controller.transcriber.release.set()
    join(worker)
    assert len(controller.delivery.calls) == 1
    assert first in controller.delivery.calls[0][0]
    assert controller.status()["id"] == second
    assert controller.status()["message"] == "Cancelled"
    assert not controller.history.recording(second).exists()


def test_queue_bound_rejects_without_replacing_pending_job():
    Config(queue_limit=1).save()
    controller = app()
    first = controller.command("start")["id"]
    controller.command("stop")
    assert controller.transcriber.entered.wait(2)
    with pytest.raises(ValueError, match="queue is full"):
        controller.command("start")
    assert controller.status()["id"] == first
    controller.transcriber.release.set()
    join(controller.worker)


def test_cancelled_middle_job_cannot_allow_third_to_overtake_first():
    controller = app()
    first = controller.command("start")["id"]
    controller.command("stop")
    first_worker = controller.worker
    assert controller.transcriber.entered.wait(2)
    controller.command("start")
    controller.command("cancel")
    join(controller.worker)
    third = controller.command("start")["id"]
    controller.command("stop")
    third_worker = controller.worker
    assert controller.transcriber.calls == [(first, "base.en")]
    controller.transcriber.release.set()
    join(first_worker, third_worker)
    assert controller.transcriber.calls == [(first, "base.en"), (third, "base.en")]
