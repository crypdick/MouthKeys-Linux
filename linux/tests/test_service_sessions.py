import json
import threading
from dataclasses import replace

import pytest

from mouthkeys import assistance, recording_assistant
from mouthkeys.assistant_history import AssistantHistory, pending_executions
from mouthkeys.config import Config, state_dir
from mouthkeys.recording_job import DictationJob

from .pipeline_support import finish
from .test_assistant_history import receipt, turn
from .test_assistant_sessions import controller, proposal


@pytest.mark.parametrize("fail", [False, True])
def test_voice_turn_observes_receipt_and_only_success_consumes_it(monkeypatch, fail):
    Config(ai_enabled="on", ai_model="test").save()
    store = AssistantHistory()
    session = store.current("command")
    store.append(session["id"], turn("Earlier proposal"))
    store.record_execution(session["id"], receipt())
    calls = []

    def propose(text, config, mode, selected, **options):
        calls.append(json.loads(text))
        if fail:
            raise RuntimeError("Provider unavailable")
        return proposal(text, mode, selected)

    monkeypatch.setattr("mouthkeys.assistance.propose", propose)
    app = controller()
    app.command("start", mode="command")
    app.command("stop")
    finish(app)
    assert calls[0]["observed_command_results"] == [receipt()]
    assert bool(pending_executions(store.get(session["id"]))) is fail


def test_private_execution_handoff_is_observed_without_database(monkeypatch):
    Config(ai_enabled="on", ai_model="test", save_history="off").save()
    app = controller()
    messages = turn("Private earlier proposal")
    events = [{"message_count": 2, "result": receipt()}]
    seen = []

    def propose(text, config, mode, selected, **options):
        seen.append(json.loads(text))
        return proposal(text, mode, selected)

    monkeypatch.setattr("mouthkeys.assistance.propose", propose)
    app.command("start", mode="command", assistant_history=messages, assistant_executions=events)
    app.command("stop")
    finish(app)
    assert seen[0]["observed_command_results"] == [receipt()]
    public = app.status()["assistant_proposal"]
    assert public["private_executions"] == events
    assert len(public["private_messages"]) == 4
    assert not pending_executions(
        {"messages": public["private_messages"], "executions": public["private_executions"]}
    )
    assert not (state_dir() / "assistant.sqlite3").exists()


def test_invalid_execution_context_is_rejected_before_recording():
    Config(save_history="off").save()
    app = controller()
    with pytest.raises(ValueError):
        app.command(
            "start",
            mode="command",
            assistant_history=[],
            assistant_executions=[{"message_count": 100, "result": receipt()}],
        )
    assert app.job is None
    assert app.recorder is None


def test_execution_context_requires_inline_history():
    Config(save_history="off").save()
    app = controller()
    with pytest.raises(ValueError, match="requires inline assistant history"):
        app.command("start", mode="command", assistant_executions=[])
    assert app.job is None
    assert not (state_dir() / "assistant.sqlite3").exists()


def test_voice_stream_status_is_a_stable_snapshot(monkeypatch):
    Config(ai_enabled="on", ai_model="test", ai_streaming="on").save()
    app = controller()
    snapshots = []

    def propose(text, _config, mode, selected, *, on_delta, **_options):
        on_delta("First")
        snapshots.append(app.status()["assistant_stream"])
        on_delta(" second")
        snapshots.append(app.status()["assistant_stream"])
        return proposal(text, mode, selected)

    monkeypatch.setattr(assistance, "propose", propose)
    app.command("start", mode="edit")
    app.command("stop")
    finish(app)
    assert [snapshot["text"] for snapshot in snapshots] == ["First", "First second"]
    assert app.status()["assistant_stream"] is None


def test_interrupt_hides_voice_stream_and_cannot_publish_late_output(monkeypatch):
    Config(ai_enabled="on", ai_model="test", ai_streaming="on").save()
    app = controller()
    entered, release = threading.Event(), threading.Event()

    def propose(text, _config, mode, selected, *, on_delta, **_options):
        on_delta("Partial voice result")
        entered.set()
        assert release.wait(3)
        on_delta("Late voice result")
        return proposal(text, mode, selected)

    monkeypatch.setattr(assistance, "propose", propose)
    app.command("start", mode="edit")
    app.command("stop")
    job, worker = app.job, app.worker
    try:
        assert entered.wait(2)
        assert app.status()["assistant_stream"]["text"] == "Partial voice result"
        app.interrupt()
        assert app.status()["assistant_stream"] is None
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert app.status()["phase"] == "interrupted"
    assert app.status()["assistant_proposal"] is None
    assert app.status()["assistant_stream"] is None
    assert AssistantHistory().get(job.assistant_session_id)["messages"] == []


def test_late_update_from_a_completed_round_cannot_leak_into_the_next(monkeypatch):
    Config(ai_enabled="on", ai_model="test", ai_streaming="on", command_confirm="off").save()
    app = controller()
    callbacks = []
    snapshots = []
    executed = []

    def propose(text, _config, mode, selected, *, on_delta, **_options):
        callbacks.append(on_delta)
        result = proposal(text, mode, selected)
        if len(callbacks) == 1:
            on_delta("First round")
        else:
            callbacks[0]("Late update")
            on_delta("Second round")
            snapshots.append(app.status()["assistant_stream"]["text"])
            result["command"] = ""
        return result

    def execute(command, **_options):
        executed.append(command)
        return {"returncode": 0, "output": "Receipt", "timed_out": False, "truncated": False}

    monkeypatch.setattr(assistance, "propose", propose)
    monkeypatch.setattr(assistance, "execute_command", execute)
    app.command("start", mode="command")
    app.command("stop")
    finish(app)
    assert snapshots == ["Second round"]
    assert len(executed) == 1
    assert app.status()["assistant_proposal"]["execution_count"] == 1


def test_expanding_json_characters_fit_in_the_voice_status_budget(monkeypatch):
    Config(ai_enabled="on", ai_model="test", ai_streaming="on", ai_show_thinking="on").save()
    app = controller()
    snapshots = []

    def propose(text, _config, mode, selected, *, on_delta, on_thinking, **_options):
        on_delta("🌿" * 40000)
        snapshots.append(app.status()["assistant_stream"])
        on_thinking("\n" * 90000)
        snapshots.append(app.status()["assistant_stream"])
        return proposal(text, mode, selected)

    monkeypatch.setattr(assistance, "propose", propose)
    app.command("start", mode="edit")
    app.command("stop")
    finish(app)
    text_only, combined = snapshots
    for snapshot in snapshots:
        assert len(json.dumps(snapshot).encode()) <= recording_assistant.STREAM_LIMIT
        assert snapshot["text"]
        assert set(snapshot["text"]) == {"🌿"}
    assert text_only["thinking"] == ""
    assert combined["thinking"]
    thinking = app.status()["assistant_proposal"]["thinking"]
    assert thinking == "\n" * recording_assistant.THINKING_LIMIT
    entry = AssistantHistory().get(app.job.assistant_session_id)
    assert all(message["content"] != thinking for message in entry["messages"])


def test_large_private_proposal_trims_wire_history_without_mutating_the_session(monkeypatch):
    Config(ai_enabled="on", ai_model="test", save_history="off").save()
    app = controller()
    earlier = turn("Earlier instruction " + "x" * 650)
    keep = turn("Keep this round")
    events = [{"message_count": 2, "result": {**receipt(), "output": "x" * 500}}]
    limit = 620
    monkeypatch.setattr(recording_assistant, "PROPOSAL_LIMIT", limit)

    def propose(*_args, **_options):
        return {
            "mode": "edit",
            "text": "Latest",
            "request": "New instruction",
            "response": "Latest",
        }

    monkeypatch.setattr(assistance, "propose", propose)
    app.command(
        "start", mode="edit", assistant_history=[*earlier, *keep], assistant_executions=events
    )
    app.command("stop")
    finish(app)
    result = app.status()["assistant_proposal"]
    assert result["private_messages"] == [
        *keep,
        {"role": "user", "content": "New instruction"},
        {"role": "assistant", "content": "Latest"},
    ]
    assert not result["private_executions"]
    assert len(json.dumps({"assistant_proposal": result, "preview": result["text"]})) <= limit
    session = app.recording_assistant.memory["edit"]
    assert session["messages"][:2] == earlier
    assert session["executions"] == events
    assert not (state_dir() / "assistant.sqlite3").exists()


def test_proposal_that_cannot_fit_status_does_not_publish_partial_output(monkeypatch):
    Config(ai_enabled="on", ai_model="test", save_history="off", ai_streaming="on").save()
    app = controller()
    monkeypatch.setattr(recording_assistant, "PROPOSAL_LIMIT", 350)

    def propose(*_args, on_delta, **_options):
        on_delta("Partial update")
        return {"mode": "edit", "text": "x" * 350, "request": "Instruction", "response": "Result"}

    monkeypatch.setattr(assistance, "propose", propose)
    app.command("start", mode="edit")
    app.command("stop")
    finish(app)
    assert app.status()["assistant_stream"] is None
    assert app.status()["assistant_proposal"] is None
    assert "exceeds status limit" in app.status()["notice"]["message"]
    assert app.recording_assistant.memory["edit"]["messages"][-1]["content"] == "Result"


def test_superseded_voice_job_can_finish_but_cannot_publish_a_stream(monkeypatch):
    Config(ai_enabled="on", ai_model="test", ai_streaming="on").save()
    app = controller()
    app.transcriber.release.clear()
    snapshots = []

    def propose(text, _config, mode, selected, *, on_delta, **_options):
        on_delta("Prior recording output")
        snapshots.append(app.status())
        return proposal(text, mode, selected)

    monkeypatch.setattr(assistance, "propose", propose)
    first = app.command("start", mode="edit")["id"]
    app.command("stop")
    worker = app.worker
    try:
        assert app.transcriber.entered.wait(2)
        second = app.command("start")["id"]
        assert second != first
    finally:
        app.transcriber.release.set()
        worker.join(3)
        if app.status()["phase"] == "recording":
            app.command("cancel")
            finish(app)
    assert not worker.is_alive()
    assert snapshots[0]["id"] == second
    assert snapshots[0]["assistant_stream"] is None
    assert snapshots[0]["preview"] == ""
    old = next(row for row in app.history.rows() if row["id"] == first)
    assert old["status"] == "transcribed"


def test_cancel_between_proposal_and_command_does_not_call_executor(monkeypatch):
    Config(ai_enabled="on", ai_model="test", command_confirm="off").save()
    app = controller()
    append = AssistantHistory.append

    def cancel_after_save(store, item_id, messages):
        append(store, item_id, messages)
        app.command("cancel")

    def unexpected_executor(*_args, **_options):
        pytest.fail("Cancelled request must not call the command executor")

    monkeypatch.setattr(AssistantHistory, "append", cancel_after_save)
    monkeypatch.setattr(
        assistance,
        "propose",
        lambda text, _, mode, selected, **_options: proposal(text, mode, selected),
    )
    monkeypatch.setattr(assistance, "execute_command", unexpected_executor)
    app.command("start", mode="command")
    app.command("stop")
    finish(app)
    assert app.history.rows()[0]["status"] == "cancelled"
    assert not AssistantHistory().get(app.job.assistant_session_id)["executions"]


@pytest.mark.parametrize("history", ["on", "off"])
def test_unbound_voice_job_fails_before_requesting_a_proposal(history, tmp_path):
    config = replace(Config(), save_history=history)
    app = controller()
    job = DictationJob("fixture", config, (), tmp_path / "recording.wav", mode="edit")
    with pytest.raises(ValueError, match="Assistant session no longer exists"):
        app.recording_assistant.propose(job, config, "Instruction", lambda _: False)
    assert not (state_dir() / "assistant.sqlite3").exists()


def test_bound_voice_job_cancelled_before_inference_never_reaches_provider(monkeypatch, tmp_path):
    config = Config(save_history="off")
    app = controller()
    job = DictationJob("fixture", config, (), tmp_path / "recording.wav", mode="edit")
    app.recording_assistant.bind_session(job, None)
    job.cancel_event.set()

    def unexpected(*_args, **_options):
        pytest.fail("Cancelled recording must not call a provider")

    monkeypatch.setattr(assistance, "propose", unexpected)
    with pytest.raises(ValueError, match="Assistant request cancelled"):
        app.recording_assistant.propose(job, config, "Instruction", lambda _: False)
    assert app.recording_assistant.stream_status() is None
    assert not (state_dir() / "assistant.sqlite3").exists()
