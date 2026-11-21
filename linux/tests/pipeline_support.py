"""Fake recording, inference and delivery for headless Controller tests."""

import threading
import wave

from mouthkeys.service import Controller
from mouthkeys.storage import History


def write_audio(path):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 8000)


class FakeRecorder:
    def __init__(self, config, path):
        self.path = path
        self.stopped = False
        self.ended = False
        write_audio(path)

    def stop(self):
        self.stopped = True

    def finished(self):
        return self.ended


class FakeTranscriber:
    def __init__(self, text="Hello, world! café 日本語"):
        self.text = text
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.error = None

    def load(self, config):
        pass

    def transcribe(self, path, config):
        self.entered.set()
        assert self.release.wait(3), "Timed out waiting for test inference gate"
        if self.error:
            raise self.error
        return self.text


class FakeDelivery:
    def __init__(self, history):
        self.calls = []
        self.error = None
        self.history = history

    def focus(self):
        return "123"

    def deliver(self, text, config, target):
        assert self.history.rows()[0]["text"] == text, "Transcript must be durable before delivery"
        self.calls.append((text, config, target))
        if self.error:
            raise self.error
        return "copied"

    def copy(self, text):
        self.calls.append(text)

    def paste_key(self, config):
        self.calls.append("paste")


def make_controller(recorder=FakeRecorder):
    history = History()
    return Controller(history, FakeTranscriber(), FakeDelivery(history), recorder)


def finish(controller):
    controller.worker.join(timeout=4)
    assert not controller.worker.is_alive()
    assert controller.status()["phase"] == "idle"
