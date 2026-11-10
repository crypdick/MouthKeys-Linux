import shutil
import threading
from pathlib import Path

import pytest
from test_learning import snapshot
from test_pipeline import FakeDelivery, FakeRecorder, FakeTranscriber, finish

from mouthkeys.config import Config
from mouthkeys.dictionary import Dictionary
from mouthkeys.learning import CorrectionLearner
from mouthkeys.pronunciation import PronunciationStore
from mouthkeys.service import Controller
from mouthkeys.storage import History


def app(text="wurld", recorder=FakeRecorder):
    history = History()
    return Controller(history, FakeTranscriber(text), FakeDelivery(history), recorder)


class SpeechRecorder(FakeRecorder):
    def __init__(self, config, path):
        super().__init__(config, path)
        shutil.copyfile(Path(__file__).parent / "resources" / "dictation_fixture.wav", path)


class FastLearner(CorrectionLearner):
    def __init__(self, config):
        super().__init__(config, clock=lambda: 0)
        self.now = 0

    def observe(self, value, now=None):
        self.now += 4
        return super().observe(value, now=self.now)


def confirmed_delivery(controller, result="pasted"):
    def deliver(text, config, target):
        controller.delivery.last_verified = snapshot(text)
        controller.delivery.calls.append(text)
        return result

    controller.delivery.deliver = deliver


def test_confirmed_insertion_starts_read_only_learning_and_review_commands(monkeypatch):
    Config(automatic_learning="on", learning_min_occurrences=1).save()
    controller = app()
    confirmed_delivery(controller)
    monkeypatch.setattr("mouthkeys.service.CorrectionLearner", FastLearner)
    monkeypatch.setattr("mouthkeys.service.read_snapshot", lambda target: snapshot("world"))
    controller.command("start")
    controller.command("stop")
    finish(controller)
    controller.learning_worker.join(3)
    assert not controller.learning_worker.is_alive()
    pending = controller.command("learning_pending")
    assert len(pending) == 1
    assert pending[0]["heard"] == "wurld"
    assert Dictionary.load()["replacements"] == []
    assert controller.status()["learning_pending"] == pending
    controller.command("learning_accept", id=pending[0]["id"])
    assert Dictionary.apply("wurld") == "world"
    assert controller.command("learning_pending") == []


@pytest.mark.parametrize(
    "result,enabled", [("paste_requested", "on"), ("copied", "on"), ("pasted", "off")]
)
def test_unconfirmed_clipboard_and_opt_out_never_observe_fields(monkeypatch, result, enabled):
    Config(automatic_learning=enabled).save()
    controller = app()
    confirmed_delivery(controller, result)
    monkeypatch.setattr(
        "mouthkeys.service.read_snapshot", lambda *_: pytest.fail("Unexpected field read")
    )
    controller.command("start")
    controller.command("stop")
    finish(controller)
    assert controller.learning_worker is None


def test_new_recording_invalidates_slow_old_observer_without_blocking_status(monkeypatch):
    Config(automatic_learning="on", learning_min_occurrences=1).save()
    controller = app()
    confirmed_delivery(controller)
    entered, release = threading.Event(), threading.Event()

    def read(target):
        entered.set()
        assert release.wait(3)
        return snapshot("world")

    monkeypatch.setattr("mouthkeys.service.read_snapshot", read)
    controller.command("start")
    controller.command("stop")
    finish(controller)
    observer = controller.learning_worker
    assert entered.wait(2)
    status_done = threading.Event()
    threading.Thread(target=lambda: (controller.status(), status_done.set()), daemon=True).start()
    try:
        assert status_done.wait(1), "AT-SPI probe held controller state lock"
        controller.command("start")
        controller.command("cancel")
        finish(controller)
    finally:
        release.set()
        observer.join(3)
    assert not observer.is_alive()
    assert controller.command("learning_pending") == []
    assert Dictionary.load()["replacements"] == []


def test_pronunciation_training_uses_actual_audio_without_delivering_and_preserves_raw():
    Config().save()
    controller = app("recognized phrase", SpeechRecorder)
    for count in (1, 2):
        controller.command("start", mode="pronunciation", label="Intended spelling")
        controller.command("stop")
        finish(controller)
        proposal = controller.status()["assistant_proposal"]
        assert proposal["mode"] == "pronunciation"
        assert proposal["text"] == "Intended spelling"
        assert proposal["samples"] == count
        assert proposal["ready"] is (count == 2)
    assert len(PronunciationStore.load()[0]["samples"]) == 2
    assert not controller.delivery.calls
    assert controller.history.rows()[0]["raw_text"] == "recognized phrase"


def test_pronunciation_no_speech_does_not_enroll():
    controller = app("", SpeechRecorder)
    controller.command("start", mode="pronunciation", label="phrase")
    controller.command("stop")
    finish(controller)
    assert PronunciationStore.load() == []
    assert controller.history.rows()[0]["error"] == "No speech detected"


def test_optional_target_capture_failure_does_not_leak_pending_job():
    Config(target_policy="start").save()
    controller = app()
    controller.delivery.capture_target = lambda *_: (_ for _ in ()).throw(RuntimeError("No AT-SPI"))
    controller.command("start")
    assert controller.job.start_target is None
    controller.command("stop")
    finish(controller)
    assert controller.status()["pending_jobs"] == 0


def test_acoustic_correction_runs_before_format_and_never_rewrites_raw(monkeypatch):
    Config(pronunciation_matching="on", lowercase_first="on").save()
    controller = app("Raw heard phrase")
    monkeypatch.setattr(PronunciationStore, "correct", lambda text, path: "CorrectSpelling")
    controller.command("start")
    controller.command("stop")
    finish(controller)
    assert controller.history.rows()[0]["raw_text"] == "Raw heard phrase"
    assert controller.history.rows()[0]["text"] == "correctSpelling"


def test_actual_alignment_uses_span_matcher_and_keeps_raw_on_budget_limit(monkeypatch):
    Config(pronunciation_matching="on", pronunciation_span_budget=321).save()
    controller = app("Heard phrase")
    original = {
        "text": "Heard phrase",
        "source": "whisper",
        "words": [
            {"word": "Heard", "start": 0, "end": 0.2},
            {"word": "phrase", "start": 0.2, "end": 0.5},
        ],
    }
    controller.transcriber.last_alignment = original

    def correct(text, path, alignment, *, budget):
        assert text == alignment["text"] == "Heard phrase"
        assert alignment == original and alignment is not original
        assert alignment["words"] is not original["words"]
        assert budget == 321
        return {"text": "Correct phrase", "truncated": True}

    monkeypatch.setattr(PronunciationStore, "correct_spans", correct)
    monkeypatch.setattr(
        PronunciationStore, "correct", lambda *_: pytest.fail("Whole phrase fallback")
    )
    controller.command("start")
    controller.command("stop")
    finish(controller)
    row = controller.history.rows()[0]
    assert row["raw_text"] == "Heard phrase"
    assert row["text"] == "Correct phrase"
    assert "budget reached" in controller.status()["message"]


def test_edit_replacement_requires_exact_captured_selection():
    from types import SimpleNamespace

    field = {**snapshot("Original text"), "selection_start": 0, "selection_end": 8}
    assert Controller._edit_target_matches(
        SimpleNamespace(start_target={"field": field}, selected_text="Original")
    )
    assert not Controller._edit_target_matches(
        SimpleNamespace(start_target={"field": field}, selected_text="Different")
    )
    assert not Controller._edit_target_matches(
        SimpleNamespace(
            start_target={"field": {**field, "protected": True}}, selected_text="Original"
        )
    )
