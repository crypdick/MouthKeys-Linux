import json
import threading
import time

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from mouthkeys.assistant_history import AssistantHistory
from mouthkeys.assistant_ui import AssistantPage
from mouthkeys.config import Config, state_dir
from mouthkeys.profiles import load_profiles, save_profiles

from .test_mode_routing import profiles


def page():
    app = QApplication.instance() or QApplication([])
    window = QWidget()
    window.jobs = set()
    window.send = lambda *args, **kwargs: None

    def send(action, data, callback):
        window.send(action, data, callback)

    return app, window, AssistantPage(window, send, window.jobs)


def process_until(app, predicate):
    deadline = time.monotonic() + 2
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.001)
    assert predicate(), "Qt operation exceeded bounded wait"


def test_voice_edit_review_button_requires_capture_and_explicit_return(monkeypatch):

    _app, window, assistant = page()
    sent = []
    window.send = lambda action, data, callback: (
        sent.append((action, data)) or callback({"result": {"message": "Edit inserted"}})
    )
    assistant.show_proposal({"mode": "edit", "text": "Typed rewrite"})
    assert not assistant.editor.replace_button.isEnabled()
    assert not assistant.editor.return_replace_button.isEnabled()
    assistant.show_proposal(
        {"mode": "edit", "text": "Voice rewrite", "edit_id": "job", "can_accept_edit": True}
    )
    assert assistant.editor.replace_button.isEnabled()
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Yes)
    assistant.accept_edit(return_to_target=True)
    assert sent == [("accept_edit", {"id": "job", "return_to_target": True})]
    assert assistant.feedback.text() == "Edit inserted"
    assert not assistant.editor.replace_button.isEnabled()
    assistant.accept_edit(return_to_target=False)
    assert len(sent) == 1
    window.close()


def test_voice_stream_clears_old_actionable_proposal_and_cancels_service():
    _app, window, assistant = page()
    sent = []
    window.send = lambda action, _data, _callback: sent.append(action)
    assistant.show_proposal({"mode": "command", "text": "Old result", "command": "old"})
    assert assistant.editor.run_button.isEnabled()
    assistant.update_status(
        {
            "phase": "transcribing",
            "assistant_stream": {"id": "new", "mode": "edit", "text": "New partial"},
        }
    )
    assert assistant.proposal is None
    assert not assistant.editor.run_button.isEnabled()
    assert assistant.editor.result.toPlainText() == "New partial"
    assistant.cancel_request()
    assert sent == ["cancel"]
    window.close()


def test_typed_automatic_chain_freezes_resolved_profile_and_voice_clears_it(monkeypatch):

    profile_id = profiles()[0]
    Config(command_profile=profile_id, command_confirm="off", command_continue="on").save()
    app, window, assistant = page()
    seen = []

    def propose(instruction, config, mode, selection, **kwargs):
        seen.append((config.ai_model, kwargs["resolved_config"]))
        rows = load_profiles()
        rows[0]["ai_model"] = "changed-model"
        save_profiles(rows)
        return {"mode": "command", "text": "Result", "command": "ls" if len(seen) == 1 else ""}

    monkeypatch.setattr("mouthkeys.assistant_ui.propose", propose)
    monkeypatch.setattr(
        "mouthkeys.assistant_ui.execute_command",
        lambda *a, **kw: {
            "returncode": 0,
            "output": "Fake",
            "timed_out": False,
            "truncated": False,
        },
    )
    assistant.editor.instruction.setText("List files")
    assistant.generate()
    process_until(app, lambda: len(seen) == 2 and not assistant.jobs)
    assert seen == [("model-0", True), ("model-0", True)]
    assert assistant.chain_config is not None
    assistant.show_proposal({"mode": "command", "text": "Voice proposal", "command": "review"})
    assert assistant.chain_config is None
    assert assistant.proposal_config is None
    window.close()


@pytest.mark.parametrize("policy", ["off", "destructive"])
def test_typed_automatic_command_respects_policy_without_confirmation(monkeypatch, policy):
    Config(ai_enabled="on", ai_model="test", command_confirm=policy, command_continue="off").save()
    app, window, assistant = page()
    executed = []
    monkeypatch.setattr(
        "mouthkeys.assistant_ui.propose",
        lambda *a, **kw: {"mode": "command", "text": "List files", "command": "ls"},
    )

    def execute(command, **kwargs):
        executed.append(command)
        return {"returncode": 0, "output": "Fake output", "timed_out": False, "truncated": False}

    monkeypatch.setattr("mouthkeys.assistant_ui.execute_command", execute)
    monkeypatch.setattr(
        assistant.editor,
        "confirm_command",
        lambda _: (_ for _ in ()).throw(AssertionError("Unexpected confirmation")),
    )
    assistant.editor.instruction.setText("List files")
    assistant.generate()
    process_until(app, lambda: executed and not assistant.jobs)
    assert executed == ["ls"]
    assert assistant.pending_result["output"] == "Fake output"
    assert not assistant.proposal
    window.close()


def test_typed_edit_cancel_signals_worker_and_discards_late_deltas_and_result(monkeypatch):
    app, window, assistant = page()
    entered = threading.Event()
    events = []

    def propose(instruction, config, mode, selection, *, cancel_event, on_delta, resolved_config):
        assert resolved_config is True
        events.append(cancel_event)
        on_delta("Partial")
        entered.set()
        assert cancel_event.wait(2)
        on_delta(" late output")
        return {"mode": "edit", "text": "Late completed result"}

    monkeypatch.setattr("mouthkeys.assistant_ui.propose", propose)
    assistant.mode.setCurrentIndex(1)
    assistant.generate()
    assert entered.wait(1)
    process_until(app, lambda: assistant.editor.result.toPlainText() == "Partial")
    task = assistant.request_task
    assert assistant.editor.cancel_request_button.isEnabled()
    assistant.cancel_request()
    assert events[0].is_set()
    assert task.wait(2000)
    process_until(app, lambda: not assistant.jobs)
    assert assistant.proposal is None
    assert assistant.editor.result.toPlainText() == "Partial"
    assert assistant.feedback.text() == "Request cancelled"
    assert not assistant.editor.cancel_request_button.isEnabled()
    assert not window.jobs
    assistant.close()
    window.close()


def test_typed_edit_streams_successfully_without_cancel_or_direct_desktop_delivery(monkeypatch):
    app, window, assistant = page()

    def propose(instruction, config, mode, selection, *, cancel_event, on_delta, resolved_config):
        assert resolved_config is True
        assert not cancel_event.is_set()
        on_delta("First ")
        on_delta("second")
        return {"mode": "edit", "text": "First second"}

    monkeypatch.setattr("mouthkeys.assistant_ui.propose", propose)
    assistant.mode.setCurrentIndex(1)
    assistant.generate()
    task = assistant.request_task
    assert task.wait(2000)
    process_until(app, lambda: not assistant.jobs)
    assert assistant.editor.result.toPlainText() == "First second"
    assert assistant.proposal == {"mode": "edit", "text": "First second"}
    assert not assistant.editor.run_button.isEnabled()
    assert not assistant.editor.cancel_request_button.isEnabled()
    assistant.close()
    window.close()


def test_edit_followup_uses_previous_result_not_original_selection(monkeypatch):

    app, window, assistant = page()
    calls = []

    def propose(instruction, config, mode, selection, **kwargs):
        calls.append((instruction, selection, kwargs.get("history", [])))
        request = json.dumps({"instruction": instruction, "selected_text": selection})
        output = "First rewrite" if len(calls) == 1 else "Second rewrite"
        return {"mode": mode, "text": output, "request": request, "response": output}

    monkeypatch.setattr("mouthkeys.assistant_ui.propose", propose)
    assistant.mode.setCurrentIndex(1)
    assistant.editor.selection.setPlainText("Original")
    assistant.editor.instruction.setText("Shorten")
    assistant.generate()
    process_until(app, lambda: not assistant.jobs)
    assistant.editor.instruction.setText("Make friendlier")
    assistant.generate()
    process_until(app, lambda: not assistant.jobs)
    assert calls[0][1] == "Original"
    assert calls[1][1] == ""
    assert calls[1][2][-1] == {"role": "assistant", "content": "First rewrite"}
    assert assistant.editor.result.toPlainText() == "Second rewrite"
    assert len(assistant.chats.session["messages"]) == 4
    sent = []
    window.send = lambda _action, data, _callback: sent.append(data)
    assistant.record()
    assert sent[0]["selected_text"] == ""
    window.close()


def test_private_voice_to_typed_and_back_preserves_context_without_db(monkeypatch):

    Config(save_history="off").save()
    app, window, assistant = page()
    private = [
        {"role": "user", "content": "Voice instruction"},
        {"role": "assistant", "content": "Voice result"},
    ]
    assistant.mode.setCurrentIndex(1)
    assistant.show_proposal(
        {
            "mode": "edit",
            "text": "Voice result",
            "session_id": "ephemeral",
            "private_messages": private,
        }
    )
    calls = []

    def propose(instruction, config, mode, selection, **kwargs):
        calls.append(kwargs["history"])
        return {
            "mode": mode,
            "text": "Typed result",
            "request": json.dumps({"instruction": instruction}),
            "response": "Typed result",
        }

    monkeypatch.setattr("mouthkeys.assistant_ui.propose", propose)
    assistant.editor.instruction.setText("Follow up")
    assistant.generate()
    process_until(app, lambda: not assistant.jobs)
    assert calls == [private]
    sent = []
    window.send = lambda action, data, _callback: sent.append((action, data))
    assistant.record()
    assert sent[0][0] == "start"
    assert sent[0][1]["assistant_history"][-1]["content"] == "Typed result"
    assert not (state_dir() / "assistant.sqlite3").exists()
    window.close()


def test_reviewed_result_survives_restart_without_automatic_continuation(monkeypatch, tmp_path):

    Config(command_continue="off").save()
    app, window, assistant = page()
    request = json.dumps({"instruction": "Inspect", "selected_text": ""})
    response = json.dumps({"command": "suggested command", "explanation": "Inspect"})
    assistant.request_context = (dict(assistant.chats.session), True, "")
    assistant.show_proposal(
        {
            "mode": "command",
            "text": "Inspect",
            "command": "suggested command",
            "request": request,
            "response": response,
        }
    )
    session_id = assistant.chats.session["id"]
    assistant.editor.command.setPlainText("reviewed command")
    assistant.editor.directory.setText(str(tmp_path))
    confirmations, executions = [], []
    monkeypatch.setattr(
        assistant.editor, "confirm_command", lambda text: confirmations.append(text) or True
    )

    def execute(command, **options):
        executions.append((command, options["working_directory"]))
        return {
            "returncode": 0,
            "output": "Observed output",
            "truncated": False,
            "timed_out": False,
        }

    monkeypatch.setattr("mouthkeys.assistant_ui.execute_command", execute)
    assistant.run_command()
    process_until(app, lambda: not assistant.jobs)
    assert executions == [("reviewed command", str(tmp_path))]
    assert confirmations == [f"Working directory: {tmp_path}\n\nreviewed command"]
    assert not assistant.editor.run_button.isEnabled()
    receipt = AssistantHistory().get(session_id)["executions"][0]["result"]
    assert receipt["command"] == "reviewed command"
    assert receipt["output"] == "Observed output"
    assistant.chats.new()
    AssistantHistory().select(session_id)
    assistant.chats.reload()
    requests = []

    def propose(instruction, config, mode, selection, **kwargs):
        requests.append(json.loads(instruction))
        return {"mode": "command", "text": "Finished", "command": ""}

    monkeypatch.setattr("mouthkeys.assistant_ui.propose", propose)
    assistant.editor.instruction.setText("What happened?")
    assistant.generate()
    process_until(app, lambda: not assistant.jobs)
    assert requests[0]["observed_command_results"][0]["output"] == "Observed output"
    assert len(executions) == 1
    window.close()


def test_voice_command_adopts_mode_and_chat_before_followup(monkeypatch):
    app, window, assistant = page()
    assistant.mode.setCurrentIndex(1)
    store = AssistantHistory()
    chat = store.new("command")
    messages = [
        {"role": "user", "content": "Voice request"},
        {"role": "assistant", "content": "Voice proposal"},
    ]
    store.append(chat["id"], messages)
    assistant.show_proposal(
        {
            "mode": "command",
            "session_id": chat["id"],
            "text": "Voice proposal",
            "command": "proposed command",
        }
    )
    assert assistant.mode.currentData() == "command"
    assert assistant.chats.picker.currentData() == chat["id"]
    calls = []

    def propose(instruction, config, mode, selection, **kwargs):
        calls.append((mode, kwargs["history"]))
        return {"mode": mode, "text": "Finished", "command": ""}

    monkeypatch.setattr("mouthkeys.assistant_ui.propose", propose)
    assistant.editor.instruction.setText("Next step")
    assistant.generate()
    process_until(app, lambda: not assistant.jobs)
    assert calls == [("command", messages)]
    window.close()


def test_repeated_equal_voice_proposals_use_job_identity():
    _app, window, assistant = page()
    proposal = {"mode": "command", "text": "Same answer", "command": "same command"}
    assistant.update_status({"id": "first-job", "phase": "idle", "assistant_proposal": proposal})
    assert assistant.editor.run_button.isEnabled()
    assistant.proposal = None
    assistant.editor.run_button.setEnabled(False)
    assistant.update_status({"id": "first-job", "phase": "idle", "assistant_proposal": proposal})
    assert not assistant.editor.run_button.isEnabled()
    assistant.update_status({"id": "second-job", "phase": "idle", "assistant_proposal": proposal})
    assert assistant.editor.run_button.isEnabled()
    window.close()


def test_disabling_history_before_run_keeps_new_outcome_private(monkeypatch):
    app, window, assistant = page()
    saved_id = assistant.chats.session["id"]
    assistant.show_proposal({"mode": "command", "text": "Inspect", "command": "reviewed command"})
    Config(save_history="off", command_continue="off").save()
    monkeypatch.setattr(assistant.editor, "confirm_command", lambda text: True)
    monkeypatch.setattr(
        "mouthkeys.assistant_ui.execute_command",
        lambda *a, **kw: {
            "returncode": 0,
            "output": "Private observed output",
            "timed_out": False,
            "truncated": False,
        },
    )
    assistant.run_command()
    process_until(app, lambda: not assistant.jobs)
    assert AssistantHistory().get(saved_id)["executions"] == []
    assert assistant.chats.session["id"] is None
    assert assistant.chats.session["executions"][0]["result"]["output"] == "Private observed output"
    window.close()


def test_disabling_history_before_typed_result_returns_keeps_the_turn_private(monkeypatch):
    app, window, assistant = page()
    entered, release = threading.Event(), threading.Event()

    def propose(*_args, **_kwargs):
        entered.set()
        assert release.wait(2)
        return {
            "mode": "edit",
            "text": "Private result",
            "request": "Private instruction",
            "response": "Private result",
        }

    monkeypatch.setattr("mouthkeys.assistant_ui.propose", propose)
    assistant.mode.setCurrentIndex(1)
    session_id = assistant.chats.session["id"]
    assistant.editor.instruction.setText("Rewrite")
    assistant.generate()
    try:
        assert entered.wait(1)
        Config(save_history="off").save()
    finally:
        release.set()
        process_until(app, lambda: not assistant.jobs)
    try:
        assert AssistantHistory().get(session_id)["messages"] == []
        assert assistant.chats.session["id"] is None
        assert assistant.chats.session["messages"][-1]["content"] == "Private result"
        assert assistant.editor.result.toPlainText() == "Private result"
        assert not window.isVisible()
    finally:
        window.close()
