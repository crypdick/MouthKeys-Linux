import json

import pytest
from test_pipeline import FakeDelivery, FakeRecorder, FakeTranscriber, finish

from mouthkeys import assistance
from mouthkeys.assistant_history import AssistantHistory
from mouthkeys.config import Config, state_dir
from mouthkeys.service import Controller
from mouthkeys.storage import History


@pytest.mark.parametrize(
    "command,policy,expected",
    [
        ("ls -la", "on", True),
        ("rm -rf example", "off", False),
        ("ls -la", "destructive", False),
        ("rm file", "destructive", True),
        ("printf test && sudo command", "destructive", True),
        ("mv\told new", "destructive", True),
        ("echo rm -rf", "destructive", True),
    ],
)
def test_upstream_command_confirmation_heuristic(command, policy, expected):
    assert assistance.requires_confirmation(command, policy) is expected


def automatic(monkeypatch, *, policy="off", continuation="on", history="on", command="ls"):
    Config(
        ai_enabled="on",
        ai_model="frozen-model",
        command_confirm=policy,
        command_continue=continuation,
        save_history=history,
    ).save()
    store = History()
    app = Controller(store, FakeTranscriber("Do work"), FakeDelivery(store), FakeRecorder)
    proposals, executed = [], []

    def completion(text, config, **kwargs):
        proposals.append((json.loads(text), config.ai_model, kwargs.get("history", [])))
        return json.dumps({"command": command, "explanation": "Proposed work"})

    def execute(text, **kwargs):
        executed.append((text, kwargs))
        return {
            "returncode": 0,
            "output": "Observed output",
            "timed_out": False,
            "truncated": False,
        }

    monkeypatch.setattr(assistance, "request_completion", completion)
    monkeypatch.setattr(assistance, "execute_command", execute)
    return app, proposals, executed


@pytest.mark.parametrize(
    "policy,command,runs",
    [("on", "ls", 0), ("destructive", "rm a", 0), ("destructive", "ls", 1), ("off", "rm a", 1)],
)
def test_voice_automatic_policy_and_single_step_receipt(monkeypatch, policy, command, runs):
    app, proposals, executed = automatic(
        monkeypatch, policy=policy, continuation="off", command=command
    )
    app.command("start", mode="command")
    app.command("stop")
    finish(app)
    assert len(executed) == runs
    assert len(proposals) == 1
    session = AssistantHistory().get(app.job.assistant_session_id)
    assert len(session["executions"]) == runs
    if runs:
        assert app.assistant_proposal["command"] == ""
        assert session["executions"][0]["result"]["command"] == command
    assert not app.delivery.calls


def test_voice_automatic_continuation_frozen_provider_receipts_and_step_bound(monkeypatch):
    app, proposals, executed = automatic(monkeypatch)
    app.command("start", mode="command")
    Config(ai_enabled="on", ai_model="changed", command_confirm="on").save()
    app.command("stop")
    finish(app)
    assert len(executed) == len(proposals) == 20
    assert all(model == "frozen-model" for _, model, _ in proposals)
    assert proposals[1][0]["instruction"].startswith('{"instruction":')
    observed = json.loads(proposals[1][0]["instruction"])["observed_command_results"]
    assert observed[0]["output"] == "Observed output"
    assert "20-step limit" in app.assistant_proposal["text"]
    assert len(AssistantHistory().get(app.job.assistant_session_id)["executions"]) == 20


@pytest.mark.parametrize("private", [False, True])
def test_execution_receipt_survives_continuation_failure(monkeypatch, private):
    app, proposals, executed = automatic(monkeypatch, history="off" if private else "on")

    def completion(text, config, **kwargs):
        if executed:
            raise RuntimeError("Provider failed after command")
        return json.dumps({"command": "ls", "explanation": "List files"})

    monkeypatch.setattr(assistance, "request_completion", completion)
    app.command("start", mode="command")
    app.command("stop")
    finish(app)
    session = (
        app.job.assistant_memory_session
        if private
        else AssistantHistory().get(app.job.assistant_session_id)
    )
    assert len(session["executions"]) == 1
    assert session["executions"][0]["result"]["output"] == "Observed output"
    assert "Provider failed" in app.status()["message"]
    assert app.status()["notice"]["kind"] == "ai"
    if private:
        assert not (state_dir() / "assistant.sqlite3").exists()


def test_cancelled_automatic_execution_still_keeps_receipt(monkeypatch):
    app, _, executed = automatic(monkeypatch)

    def execute(command, **kwargs):
        app.command("cancel")
        assert kwargs["cancel_event"].is_set()
        return {
            "returncode": -9,
            "output": "Partial",
            "timed_out": False,
            "truncated": False,
            "cancelled": True,
        }

    monkeypatch.setattr(assistance, "execute_command", execute)
    app.command("start", mode="command")
    app.command("stop")
    finish(app)
    session = AssistantHistory().get(app.job.assistant_session_id)
    assert session["executions"][0]["result"]["cancelled"] is True
    assert app.assistant_proposal is None


def test_malformed_streamed_command_explanation_never_executes(monkeypatch):
    app, _, executed = automatic(monkeypatch)

    def completion(text, config, **kwargs):
        output = '{"explanation":"Partial answer","command":42}'
        for char in output:
            kwargs["on_delta"](char)
        assert app.status()["assistant_stream"]["text"] == "Partial answer"
        return output

    Config(ai_enabled="on", ai_model="test", ai_streaming="on", command_confirm="off").save()
    monkeypatch.setattr(assistance, "request_completion", completion)
    app.command("start", mode="command")
    app.command("stop")
    finish(app)
    assert not executed
    assert app.assistant_proposal is None
    assert app.status()["assistant_stream"] is None
    assert "no valid command proposal" in app.status()["message"]
