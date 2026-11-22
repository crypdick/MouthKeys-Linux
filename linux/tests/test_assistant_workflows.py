"""Assistant review and worker failure behavior without live desktop effects."""

import json
from threading import Event
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QMessageBox

from mouthkeys import assistant_ui
from mouthkeys.assistant_history import AssistantHistory
from mouthkeys.config import Config

from .test_assistant_ui import page, process_until


def test_return_confirmation_does_not_replace_a_different_voice_edit(monkeypatch):
    _, window, assistant = page()
    window.send = Mock()
    assistant.show_proposal(
        {"mode": "edit", "text": "Reviewed edit", "edit_id": "reviewed", "can_accept_edit": True}
    )

    def confirm(*args):
        # Native confirmation runs a nested event loop; the daemon may post a new proposal.
        assistant.update_status(
            {
                "id": "new",
                "phase": "idle",
                "assistant_proposal": {
                    "mode": "edit",
                    "text": "Different edit",
                    "edit_id": "new",
                    "can_accept_edit": True,
                },
            }
        )
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", confirm)
    assistant.accept_edit(return_to_target=True)
    window.send.assert_not_called()
    assert assistant.editor.result.toPlainText() == "Different edit"
    assert assistant.editor.replace_button.isEnabled()


def test_command_confirmation_cannot_consume_replacement_proposal(monkeypatch):
    _, _, assistant = page()
    assistant.show_proposal({"mode": "command", "text": "Reviewed", "command": "printf old"})

    def confirm(*args):
        assistant.show_proposal({"mode": "command", "text": "Replaced", "command": "printf new"})
        return True

    task = Mock()
    monkeypatch.setattr(assistant.editor, "confirm_command", confirm)
    monkeypatch.setattr(assistant, "_task", task)
    assistant.run_command()
    task.assert_not_called()
    assert assistant.editor.result.toPlainText() == "Replaced"
    assert assistant.editor.run_button.isEnabled()


def test_worker_failure_releases_controls_and_blocks_new_requests_while_busy(monkeypatch):
    app, window, assistant = page()
    window.send = Mock()
    entered, release = Event(), Event()

    def fail(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        raise OSError("Provider offline")

    monkeypatch.setattr(assistant_ui, "propose", fail)
    assistant.editor.instruction.setText("Review these notes")
    assistant.editor.generate_button.click()
    try:
        assert entered.wait(1)
        assert not assistant.editor.generate_button.isEnabled()
        assistant.record()
        assistant.generate()
        assistant.update_status(
            {"phase": "idle", "assistant_proposal": {"mode": "edit", "text": "Stale result"}}
        )
        window.send.assert_not_called()
        assert assistant.proposal is None
    finally:
        release.set()
        process_until(app, lambda: not assistant.jobs)
    assert not window.jobs
    assert assistant.editor.generate_button.isEnabled()
    assert assistant.feedback.text() == "Provider offline"


def test_no_worker_starts_when_chat_cannot_be_reloaded(monkeypatch):
    _, _, assistant = page()
    launch = Mock()
    monkeypatch.setattr(assistant.chats, "reload", lambda: None)
    monkeypatch.setattr(assistant, "_task", launch)
    assistant.generate()
    launch.assert_not_called()
    assert assistant.editor.generate_button.isEnabled()


def test_unreadable_profile_is_reported_before_worker_start(monkeypatch):
    _, _, assistant = page()
    launch = Mock()
    monkeypatch.setattr(assistant, "_task", launch)

    def unavailable(*args, **kwargs):
        raise OSError("Profile unavailable")

    monkeypatch.setattr(assistant_ui, "select_prompt", unavailable)
    assistant.generate()
    assert assistant.feedback.text() == "Profile unavailable"
    launch.assert_not_called()


def test_result_remains_available_after_typed_conversation_write_fails(monkeypatch):
    app, _, assistant = page()
    assistant.mode.setCurrentIndex(1)
    session_id = assistant.chats.session["id"]

    def write_failed(*args, **kwargs):
        raise OSError("History database full")

    monkeypatch.setattr(assistant.chats, "append_turn", write_failed)
    monkeypatch.setattr(
        assistant_ui,
        "propose",
        lambda *args, **kwargs: {
            "mode": "edit",
            "text": "Finished edit",
            "request": "Edit these notes",
            "response": "Finished edit",
        },
    )
    assistant.generate()
    process_until(app, lambda: not assistant.jobs)
    assert (
        assistant.feedback.text()
        == "Result ready; could not save conversation: History database full"
    )
    assert assistant.editor.result.toPlainText() == "Finished edit"
    assert AssistantHistory().get(session_id)["messages"] == []


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ({"error": "Original selection changed"}, "Original selection changed"),
        ({"result": None}, "Edit replacement finished"),
    ],
)
def test_edit_replacement_reports_errors_and_fallback_messages(reply, message):
    _, window, assistant = page()
    window.send = Mock(side_effect=lambda _action, _data, callback: callback(reply))
    assistant.show_proposal(
        {"mode": "edit", "text": "Rewrite", "edit_id": "current", "can_accept_edit": True}
    )
    assistant.editor.replace_button.click()
    action, payload, _callback = window.send.call_args.args
    assert (action, payload) == (
        "accept_edit",
        {"id": "current", "return_to_target": False},
    )
    assert assistant.feedback.text() == message
    assert not assistant.editor.replace_button.isEnabled()


def test_declining_return_confirmation_leaves_edit_available(monkeypatch):
    _, window, assistant = page()
    window.send = Mock()
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No)
    assistant.show_proposal(
        {"mode": "edit", "text": "Proposed edit", "edit_id": "current", "can_accept_edit": True}
    )
    assistant.editor.return_replace_button.click()
    window.send.assert_not_called()
    assert assistant.editor.replace_button.isEnabled()


def test_command_receipt_failure_keeps_output_and_requires_explicit_followup(monkeypatch):
    Config(command_continue="on").save()
    app, _, assistant = page()
    assistant.show_proposal({"mode": "command", "text": "Reviewed", "command": "printf result"})

    def write_failed(*args, **kwargs):
        raise OSError("History database full")

    propose = Mock(return_value={"mode": "command", "text": "Followup", "command": ""})
    monkeypatch.setattr(AssistantHistory, "record_execution", write_failed)
    monkeypatch.setattr(assistant.editor, "confirm_command", lambda _text: True)
    monkeypatch.setattr(assistant_ui, "propose", propose)
    monkeypatch.setattr(
        assistant_ui,
        "execute_command",
        lambda *args, **kwargs: {
            "returncode": -9,
            "output": "observed",
            "timed_out": True,
            "truncated": True,
        },
    )
    assistant.editor.run_button.click()
    process_until(app, lambda: not assistant.jobs)
    assert assistant.editor.result.toPlainText() == "observed"
    assert (
        assistant.feedback.text()
        == "Command finished; result could not be saved: History database full"
    )
    assert not assistant.editor.run_button.isEnabled()
    propose.assert_not_called()
    assistant.editor.instruction.setText("What happened?")
    assistant.generate()
    process_until(app, lambda: not assistant.jobs)
    assert json.loads(propose.call_args.args[0])["observed_command_results"] == [
        assistant.pending_result
    ]


def test_command_continuation_stops_at_cap_even_without_saved_conversation(monkeypatch):
    Config(command_continue="on").save()
    app, _, assistant = page()
    assistant.show_proposal({"mode": "command", "text": "Reviewed", "command": "printf result"})
    assistant.command_steps = 19
    assistant.chats.session = None
    propose = Mock()
    monkeypatch.setattr(assistant_ui, "propose", propose)
    monkeypatch.setattr(assistant.editor, "confirm_command", lambda _text: True)
    monkeypatch.setattr(
        assistant_ui,
        "execute_command",
        lambda *args, **kwargs: {
            "returncode": -9,
            "output": "partial",
            "timed_out": True,
            "truncated": False,
        },
    )
    assistant.run_command()
    process_until(app, lambda: not assistant.jobs)
    assert assistant.feedback.text() == "20-step limit reached; review results before continuing."
    assert assistant.editor.result.toPlainText() == "partial"
    propose.assert_not_called()


def test_idle_assistant_ignores_run_cancel_and_late_thinking_signal(monkeypatch):
    _, window, assistant = page()
    window.send = Mock()
    launch = Mock()
    monkeypatch.setattr(assistant, "_task", launch)
    assistant.run_command()
    assistant.cancel_request()
    assistant.show_thinking("previous-request", "Late private thinking")
    launch.assert_not_called()
    window.send.assert_not_called()
    assert assistant.editor.thinking.toPlainText() == ""


def test_cancelled_command_retains_receipt_but_does_not_continue(monkeypatch):
    Config(command_continue="on").save()
    app, _, assistant = page()
    session_id = assistant.chats.session["id"]
    assistant.show_proposal({"mode": "command", "text": "Reviewed", "command": "printf partial"})
    entered = Event()

    def execute(*args, cancel_event, **kwargs):
        entered.set()
        assert cancel_event.wait(2)
        return {
            "returncode": -9,
            "output": "partial",
            "truncated": False,
            "timed_out": False,
            "cancelled": True,
        }

    propose = Mock()
    monkeypatch.setattr(assistant_ui, "propose", propose)
    monkeypatch.setattr(assistant_ui, "execute_command", execute)
    monkeypatch.setattr(assistant.editor, "confirm_command", lambda _text: True)
    assistant.run_command()
    try:
        assert entered.wait(1)
    finally:
        assistant.cancel_request()
        process_until(app, lambda: not assistant.jobs)
    assert assistant.feedback.text() == "Request cancelled"
    assert assistant.editor.result.toPlainText() == "partial"
    assert AssistantHistory().get(session_id)["executions"][0]["result"]["cancelled"] is True
    propose.assert_not_called()
