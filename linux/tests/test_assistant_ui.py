import threading
import time

from PySide6.QtWidgets import QApplication, QWidget

from mouthkeys.assistant_ui import AssistantPage
from mouthkeys.config import Config, state_dir


def page():
    app = QApplication.instance() or QApplication([])
    window = QWidget()
    window.jobs = set()
    window.send = lambda *args, **kwargs: None
    return app, window, AssistantPage(window)


def process_until(app, predicate):
    deadline = time.monotonic() + 2
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.001)
    assert predicate(), "Qt operation exceeded bounded wait"


def test_typed_edit_cancel_signals_worker_and_discards_late_deltas_and_result(monkeypatch):
    app, window, assistant = page()
    entered = threading.Event()
    events = []

    def propose(instruction, config, mode, selection, *, cancel_event, on_delta):
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
    process_until(app, lambda: assistant.result.toPlainText() == "Partial")
    task = assistant.request_task
    assert assistant.cancel_request_button.isEnabled()
    assistant.cancel_request()
    assert events[0].is_set()
    assert task.wait(2000)
    process_until(app, lambda: not assistant.jobs)
    assert assistant.proposal is None
    assert assistant.result.toPlainText() == "Partial"
    assert assistant.feedback.text() == "Request cancelled"
    assert not assistant.cancel_request_button.isEnabled()
    assert not window.jobs
    assistant.close()
    window.close()


def test_typed_edit_streams_successfully_without_cancel_or_direct_desktop_delivery(monkeypatch):
    app, window, assistant = page()

    def propose(instruction, config, mode, selection, *, cancel_event, on_delta):
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
    assert assistant.result.toPlainText() == "First second"
    assert assistant.proposal == {"mode": "edit", "text": "First second"}
    assert not assistant.run_button.isEnabled()
    assert not assistant.cancel_request_button.isEnabled()
    assistant.close()
    window.close()


def test_edit_followup_uses_previous_result_not_original_selection(monkeypatch):
    import json

    app, window, assistant = page()
    calls = []

    def propose(instruction, config, mode, selection, **kwargs):
        calls.append((instruction, selection, kwargs.get("history", [])))
        request = json.dumps({"instruction": instruction, "selected_text": selection})
        output = "First rewrite" if len(calls) == 1 else "Second rewrite"
        return {"mode": mode, "text": output, "request": request, "response": output}

    monkeypatch.setattr("mouthkeys.assistant_ui.propose", propose)
    assistant.mode.setCurrentIndex(1)
    assistant.selection.setPlainText("Original")
    assistant.instruction.setText("Shorten")
    assistant.generate()
    process_until(app, lambda: not assistant.jobs)
    assistant.instruction.setText("Make friendlier")
    assistant.generate()
    process_until(app, lambda: not assistant.jobs)
    assert calls[0][1] == "Original"
    assert calls[1][1] == ""
    assert calls[1][2][-1] == {"role": "assistant", "content": "First rewrite"}
    assert assistant.result.toPlainText() == "Second rewrite"
    assert len(assistant.session["messages"]) == 4
    sent = []
    window.send = lambda action, data: sent.append(data)
    assistant.record()
    assert sent[0]["selected_text"] == ""
    window.close()


def test_private_voice_to_typed_and_back_preserves_context_without_db(monkeypatch):
    import json

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
    assistant.instruction.setText("Follow up")
    assistant.generate()
    process_until(app, lambda: not assistant.jobs)
    assert calls == [private]
    sent = []
    window.send = lambda action, data: sent.append((action, data))
    assistant.record()
    assert sent[0][0] == "start"
    assert sent[0][1]["assistant_history"][-1]["content"] == "Typed result"
    assert not (state_dir() / "assistant.sqlite3").exists()
    window.close()


def test_reviewed_result_survives_restart_without_automatic_continuation(monkeypatch, tmp_path):
    import json

    from mouthkeys.assistant_history import AssistantHistory

    Config(command_continue="off").save()
    app, window, assistant = page()
    request = json.dumps({"instruction": "Inspect", "selected_text": ""})
    response = json.dumps({"command": "suggested command", "explanation": "Inspect"})
    assistant.request_context = (dict(assistant.session), True, "")
    assistant.show_proposal(
        {
            "mode": "command",
            "text": "Inspect",
            "command": "suggested command",
            "request": request,
            "response": response,
        }
    )
    session_id = assistant.session["id"]
    assistant.command.setPlainText("reviewed command")
    assistant.directory.setText(str(tmp_path))
    confirmations, executions = [], []
    monkeypatch.setattr(
        assistant, "confirm_command", lambda text: confirmations.append(text) or True
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
    assert not assistant.run_button.isEnabled()
    receipt = AssistantHistory().get(session_id)["executions"][0]["result"]
    assert receipt["command"] == "reviewed command"
    assert receipt["output"] == "Observed output"
    assistant.new_chat()
    AssistantHistory().select(session_id)
    assistant.reload_chat()
    requests = []

    def propose(instruction, config, mode, selection, **kwargs):
        requests.append(json.loads(instruction))
        return {"mode": "command", "text": "Finished", "command": ""}

    monkeypatch.setattr("mouthkeys.assistant_ui.propose", propose)
    assistant.instruction.setText("What happened?")
    assistant.generate()
    process_until(app, lambda: not assistant.jobs)
    assert requests[0]["observed_command_results"][0]["output"] == "Observed output"
    assert len(executions) == 1
    window.close()


def test_voice_command_adopts_mode_and_chat_before_followup(monkeypatch):
    from mouthkeys.assistant_history import AssistantHistory

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
    assert assistant.chats.currentData() == chat["id"]
    calls = []

    def propose(instruction, config, mode, selection, **kwargs):
        calls.append((mode, kwargs["history"]))
        return {"mode": mode, "text": "Finished", "command": ""}

    monkeypatch.setattr("mouthkeys.assistant_ui.propose", propose)
    assistant.instruction.setText("Next step")
    assistant.generate()
    process_until(app, lambda: not assistant.jobs)
    assert calls == [("command", messages)]
    window.close()


def test_repeated_equal_voice_proposals_use_job_identity():
    app, window, assistant = page()
    proposal = {"mode": "command", "text": "Same answer", "command": "same command"}
    assistant.update_status({"id": "first-job", "phase": "idle", "assistant_proposal": proposal})
    assert assistant.run_button.isEnabled()
    assistant.proposal = None
    assistant.run_button.setEnabled(False)
    assistant.update_status({"id": "first-job", "phase": "idle", "assistant_proposal": proposal})
    assert not assistant.run_button.isEnabled()
    assistant.update_status({"id": "second-job", "phase": "idle", "assistant_proposal": proposal})
    assert assistant.run_button.isEnabled()
    window.close()


def test_disabling_history_before_run_keeps_new_outcome_private(monkeypatch):
    from mouthkeys.assistant_history import AssistantHistory

    app, window, assistant = page()
    saved_id = assistant.session["id"]
    assistant.show_proposal({"mode": "command", "text": "Inspect", "command": "reviewed command"})
    Config(save_history="off", command_continue="off").save()
    monkeypatch.setattr(assistant, "confirm_command", lambda text: True)
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
    assert assistant.session["id"] is None
    assert assistant.session["executions"][0]["result"]["output"] == "Private observed output"
    window.close()
