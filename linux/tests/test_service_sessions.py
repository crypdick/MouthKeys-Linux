import json

import pytest
from test_assistant_history import receipt, turn
from test_assistant_sessions import controller, proposal
from test_pipeline import finish

from mouthkeys.assistant_history import AssistantHistory, pending_executions
from mouthkeys.config import Config, state_dir


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
    assert app.job is None and app.recorder is None
