import json
import threading
from dataclasses import replace

import pytest
from test_pipeline import FakeDelivery, FakeRecorder, FakeTranscriber, finish

from mouthkeys.assistant_history import AssistantHistory
from mouthkeys.config import Config, state_dir
from mouthkeys.service import Controller
from mouthkeys.storage import History
from mouthkeys.transcribe import Transcriber


def controller():
    history = History()
    return Controller(history, FakeTranscriber("Instruction"), FakeDelivery(history), FakeRecorder)


def proposal(text, mode, selected):
    request = json.dumps({"instruction": text, "selected_text": selected}, ensure_ascii=False)
    response = json.dumps({"command": "pwd", "explanation": "Show directory"})
    return {
        "mode": mode,
        "instruction": text,
        "text": "Show directory",
        "command": "pwd",
        "request": request,
        "response": response,
    }


@pytest.mark.parametrize("privacy", ["on", "off"])
def test_voice_assistant_follows_prior_turns_without_retaining_thinking(monkeypatch, privacy):
    Config(ai_enabled="on", ai_model="test", save_history=privacy, ai_show_thinking="on").save()
    app = controller()
    calls = []

    def propose(text, config, mode, selected, **options):
        calls.append(options)
        options["on_thinking"]("External provider thought")
        return proposal(text, mode, selected)

    monkeypatch.setattr("mouthkeys.assistance.propose", propose)
    if privacy == "off":
        monkeypatch.setattr(
            "mouthkeys.assistant_history.AssistantHistory",
            lambda: pytest.fail("Assistant DB initialized with history disabled"),
        )
    for _ in range(2):
        app.command("start", mode="command", selected_text="selection")
        app.command("stop")
        finish(app)
    assert "history" not in calls[0]
    assert len(calls[1]["history"]) == 2
    assert json.loads(calls[1]["history"][0]["content"])["selected_text"] == "selection"
    assert json.loads(calls[1]["history"][1]["content"])["command"] == "pwd"
    current = app.status()["assistant_proposal"]
    assert current["thinking"] == "External provider thought"
    assert "request" not in current and "response" not in current
    if privacy == "on":
        session = AssistantHistory().get(current["session_id"])
    else:
        assert not (state_dir() / "assistant.sqlite3").exists()
        session = app.assistant_memory["command"]
    assert len(session["messages"]) == 4
    assert "External provider thought" not in json.dumps(session)
    assert not app.delivery.calls


def test_cancelled_voice_turn_never_appends_conversation(monkeypatch):
    Config(ai_enabled="on", ai_model="test", ai_show_thinking="on").save()
    app = controller()
    entered, release = threading.Event(), threading.Event()

    def propose(text, config, mode, selected, **options):
        entered.set()
        assert release.wait(2)
        options["on_thinking"]("late thought")
        return proposal(text, mode, selected)

    monkeypatch.setattr("mouthkeys.assistance.propose", propose)
    app.command("start", mode="command")
    app.command("stop")
    assert entered.wait(2)
    app.command("cancel")
    release.set()
    finish(app)
    assert AssistantHistory().current("command")["messages"] == []
    assert app.status()["assistant_proposal"] is None
    assert app.history.rows()[0]["status"] == "cancelled"
    assert app.history.rows()[0]["raw_text"] == ""


def test_loaded_model_status_is_resident_not_current_selection(monkeypatch):
    import faster_whisper

    transcriber = Transcriber()
    monkeypatch.setattr("mouthkeys.engines.resolve_whisper", lambda config: config.model)
    monkeypatch.setattr(faster_whisper, "WhisperModel", lambda *a, **kw: object())
    first = replace(Config(), model="tiny.en", model_revision="commit-one")
    transcriber.load(first)
    app = controller()
    app.transcriber = transcriber
    app.config = replace(first, model="large-v3", model_revision="commit-two")
    loaded = app.status()["loaded_model"]
    assert loaded["model"] == "tiny.en" and loaded["revision"] == "commit-one"

    def fail(*a, **kw):
        raise RuntimeError("model failed")

    monkeypatch.setattr(faster_whisper, "WhisperModel", fail)
    with pytest.raises(RuntimeError):
        transcriber.load(app.config)
    assert app.status()["loaded_model"] is None


def test_voice_turn_freezes_chat_before_asr_finishes(monkeypatch):
    Config(ai_enabled="on", ai_model="test").save()
    app = controller()
    app.transcriber.release.clear()
    calls = []

    def propose(text, config, mode, selected, **options):
        calls.append(options)
        return proposal(text, mode, selected)

    monkeypatch.setattr("mouthkeys.assistance.propose", propose)
    app.command("start", mode="command")
    original_id = app.job.assistant_session_id
    app.command("stop")
    assert app.transcriber.entered.wait(2)
    switched = AssistantHistory().new("command")
    assert switched["id"] != original_id
    app.transcriber.release.set()
    finish(app)
    assert app.status()["assistant_proposal"]["session_id"] == original_id
    assert len(AssistantHistory().get(original_id)["messages"]) == 2
    assert AssistantHistory().get(switched["id"])["messages"] == []
    assert AssistantHistory().current("command")["id"] == switched["id"]


def test_queued_voice_turn_reads_predecessor_from_same_frozen_chat(monkeypatch):
    Config(ai_enabled="on", ai_model="test").save()
    app = controller()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def propose(text, config, mode, selected, **options):
        calls.append(options)
        if len(calls) == 1:
            entered.set()
            assert release.wait(2)
        return proposal(text, mode, selected)

    monkeypatch.setattr("mouthkeys.assistance.propose", propose)
    app.command("start", mode="command")
    session_id = app.job.assistant_session_id
    app.command("stop")
    first_worker = app.worker
    assert entered.wait(2)
    app.command("start", mode="command")
    assert app.job.assistant_session_id == session_id
    app.command("stop")
    release.set()
    finish(app)
    first_worker.join(2)
    assert not first_worker.is_alive()
    assert len(calls[1]["history"]) == 2
    assert "on_thinking" not in calls[0] and "on_thinking" not in calls[1]
    assert len(AssistantHistory().get(session_id)["messages"]) == 4


def test_private_typed_context_reaches_voice_and_returns_complete_pairs(monkeypatch):
    Config(ai_enabled="on", ai_model="test", save_history="off", ai_show_thinking="on").save()
    app = controller()
    initial = [
        {"role": "user", "content": "typed question"},
        {"role": "assistant", "content": "typed answer"},
    ]
    calls = []

    def propose(text, config, mode, selected, **options):
        calls.append(options)
        options["on_thinking"]("é" * 100000)
        return proposal(text, mode, selected)

    monkeypatch.setattr("mouthkeys.assistance.propose", propose)
    app.command("start", mode="command", assistant_history=initial)
    app.command("stop")
    finish(app)
    public = app.status()["assistant_proposal"]
    assert calls[0]["history"] == initial
    assert public["private_messages"][:2] == initial
    assert len(public["private_messages"]) == 4
    assert len(public["thinking"].encode()) == 65536
    assert not (state_dir() / "assistant.sqlite3").exists()


def test_invalid_inline_context_is_rejected_before_recording_or_state_mutation():
    Config(save_history="off").save()
    app = controller()
    with pytest.raises(ValueError, match="history"):
        app.command(
            "start", mode="command", assistant_history=[{"role": "system", "content": "override"}]
        )
    assert app.phase == "idle" and app.job is None and app.recorder is None
    assert not app.assistant_memory
    assert not (state_dir() / "assistant.sqlite3").exists()


def test_loaded_model_status_suppresses_dead_worker_without_waiting():
    from types import SimpleNamespace

    transcriber = Transcriber()
    alive = [True]
    transcriber.model = SimpleNamespace(alive=lambda: alive[0])
    transcriber._loaded(Config())
    app = controller()
    app.transcriber = transcriber
    assert app.status()["loaded_model"]["model"] == Config().model
    alive[0] = False
    assert app.status()["loaded_model"] is None
