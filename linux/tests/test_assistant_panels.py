import threading
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QInputDialog, QMessageBox, QPushButton

from mouthkeys.assistant_history import AssistantHistory
from mouthkeys.config import Config, state_dir

from .test_assistant_ui import page, process_until


def click(assistant, title):
    next(button for button in assistant.findChildren(QPushButton) if button.text() == title).click()


def test_chat_picker_supports_rename_and_confirmed_deletion(monkeypatch):
    _, window, assistant = page()
    store = AssistantHistory()
    original = assistant.chats.session["id"]
    responses = iter([("Discarded title", False), (" ", True), ("Renamed chat", True)])
    monkeypatch.setattr(QInputDialog, "getText", lambda *_args, **_kwargs: next(responses))
    decisions = iter([QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes])
    monkeypatch.setattr(QMessageBox, "question", lambda *_: next(decisions))
    try:
        click(assistant, "Rename")
        assert store.get(original)["title"] == "New chat"
        click(assistant, "Rename")
        assert "Chat title must contain 1–100 characters" in assistant.feedback.text()
        click(assistant, "Rename")
        assert store.get(original)["title"] == "Renamed chat"
        assert assistant.chats.picker.currentText() == "Renamed chat"
        click(assistant, "Delete")
        assert assistant.chats.session["id"] == original
        click(assistant, "Delete")
        with pytest.raises(ValueError, match="no longer exists"):
            store.get(original)
        assert assistant.chats.session["id"] != original
        assert not window.isVisible()
    finally:
        window.close()


@pytest.mark.parametrize("action", ["Rename", "Delete"])
def test_modal_chat_action_cannot_change_a_newly_selected_conversation(monkeypatch, action):
    _, window, assistant = page()
    store = AssistantHistory()
    first = assistant.chats.session
    second = store.new("command")
    assistant.chats.reload()
    picker = assistant.chats.picker
    picker.setCurrentIndex(picker.findData(first["id"]))

    def switch():
        picker.setCurrentIndex(picker.findData(second["id"]))

    def rename(*_args, **_kwargs):
        switch()
        return "Wrong title", True

    def delete(*_):
        switch()
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QInputDialog, "getText", rename)
    monkeypatch.setattr(QMessageBox, "question", delete)
    try:
        click(assistant, action)
        assert "Conversation changed" in assistant.feedback.text()
        assert assistant.chats.session["id"] == second["id"]
        assert store.get(first["id"])["title"] == first["title"]
        assert store.get(second["id"])["title"] == second["title"]
        assert not window.isVisible()
    finally:
        window.close()


def test_private_chat_can_be_replaced_without_opening_the_history_database(monkeypatch):
    Config(save_history="off").save()
    _, window, assistant = page()
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes)

    def unexpected_rename(*_args, **_kwargs):
        pytest.fail("Private chats cannot be renamed or saved")

    monkeypatch.setattr(QInputDialog, "getText", unexpected_rename)
    private_messages = [
        {"role": "user", "content": "Private request"},
        {"role": "assistant", "content": "Private answer"},
    ]
    try:
        assistant.show_proposal(
            {"mode": "edit", "text": "Private answer", "private_messages": private_messages}
        )
        assert "Private answer" in assistant.chats.transcript.toPlainText()
        click(assistant, "Rename")
        click(assistant, "Delete")
        assert not assistant.chats.transcript.toPlainText()
        assistant.show_proposal(
            {"mode": "edit", "text": "Malformed history", "private_messages": {}}
        )
        assert assistant.feedback.text() == "Invalid assistant messages"
        assert assistant.editor.result.toPlainText() == "Malformed history"
        click(assistant, "New")
        assert assistant.chats.session["id"] is None
        assert not (state_dir() / "assistant.sqlite3").exists()
        assert not window.isVisible()
    finally:
        window.close()


def test_chat_reload_failure_does_not_send_an_instruction_with_stale_history(monkeypatch):
    _, window, assistant = page()

    def fail(_store, _mode):
        raise OSError("History unavailable")

    def unexpected_request(*_args, **_kwargs):
        pytest.fail("No request should be sent when the context failed to load")

    monkeypatch.setattr(AssistantHistory, "current", fail)
    monkeypatch.setattr("mouthkeys.assistant_ui.propose", unexpected_request)
    try:
        assistant.editor.instruction.setText("Do not send this")
        click(assistant, "Generate proposal")
        assert assistant.feedback.text() == "History unavailable"
        assert not assistant.jobs
    finally:
        window.close()


def test_deleted_chat_cannot_become_the_current_conversation_from_an_old_picker():
    _, window, assistant = page()
    store = AssistantHistory()
    original = assistant.chats.session["id"]
    missing = store.new("command")["id"]
    store.select(original)
    assistant.chats.reload()
    picker = assistant.chats.picker
    try:
        picker.setCurrentIndex(-1)
        assert assistant.chats.session["id"] == original
        store.delete(missing)
        picker.setCurrentIndex(picker.findData(missing))
        assert "no longer exists" in assistant.feedback.text()
        assert picker.currentData() == original
        assert assistant.chats.session["id"] == original
    finally:
        window.close()


@pytest.mark.parametrize("history", ["on", "off"])
def test_voice_result_does_not_switch_to_a_chat_from_the_other_history_mode(history):
    Config(save_history=history).save()
    _, window, assistant = page()
    original = assistant.chats.session
    metadata = (
        {"session_id": "50c55f36-ac48-4b67-bcdb-ea1a1116ec11"}
        if history == "off"
        else {"private_messages": []}
    )
    try:
        assistant.show_proposal({"mode": "command", "text": "Voice result", **metadata})
        assert assistant.editor.result.toPlainText() == "Voice result"
        assert assistant.chats.session is original
        assert original["messages"] == []
        assert (state_dir() / "assistant.sqlite3").exists() == (history == "on")
    finally:
        window.close()


def test_deleted_chat_shows_result_without_saving_to_another_chat(monkeypatch):
    app, window, assistant = page()
    original = assistant.chats.session["id"]
    entered, release = threading.Event(), threading.Event()

    def propose(*_args, **_kwargs):
        entered.set()
        assert release.wait(2)
        return {"mode": "command", "text": "Result", "request": "Request", "response": "Result"}

    def unexpected_dialog(*_):
        pytest.fail("A pending request cannot prompt to delete its chat")

    monkeypatch.setattr(QMessageBox, "question", unexpected_dialog)
    monkeypatch.setattr("mouthkeys.assistant_ui.propose", propose)
    click(assistant, "Generate proposal")
    try:
        assert entered.wait(1)
        # The busy editor cannot switch contexts or create another chat.
        click(assistant, "New")
        click(assistant, "Delete")
        assert assistant.chats.reload() is None
        assert assistant.chats.session["id"] == original
        assert len(AssistantHistory().rows()) == 1
        AssistantHistory().delete(original)
    finally:
        release.set()
        process_until(app, lambda: not assistant.jobs)
    try:
        assert assistant.editor.result.toPlainText() == "Result"
        assert "could not save conversation" in assistant.feedback.text()
        assert AssistantHistory().rows() == []
    finally:
        window.close()


def test_thinking_stream_is_display_only_and_copy_uses_an_explicit_action(monkeypatch):
    Config(save_history="off", ai_show_thinking="on").save()
    app, window, assistant = page()
    copied = []
    monkeypatch.setattr(
        QGuiApplication, "clipboard", lambda: SimpleNamespace(setText=copied.append)
    )
    entered, release = threading.Event(), threading.Event()

    def propose(*_args, **options):
        options["on_thinking"]("Temporary reasoning")
        entered.set()
        assert release.wait(2)
        return {
            "mode": "edit",
            "text": "Final",
            "request": "Edit request",
            "response": "Final",
            "thinking": "Final reasoning",
        }

    monkeypatch.setattr("mouthkeys.assistant_ui.propose", propose)
    try:
        click(assistant, "Copy result")
        assert not copied
        assistant.mode.setCurrentIndex(assistant.mode.findData("edit"))
        click(assistant, "Generate proposal")
        try:
            assert entered.wait(1)
            process_until(
                app, lambda: assistant.editor.thinking.toPlainText() == "Temporary reasoning"
            )
            assert not assistant.editor.thinking.isHidden()
        finally:
            release.set()
            process_until(app, lambda: not assistant.jobs)
        assert assistant.editor.thinking.toPlainText() == "Final reasoning"
        assert assistant.chats.session["messages"][-1]["content"] == "Final"
        assert not copied
        click(assistant, "Copy result")
        assert copied == ["Final"]
        assert assistant.feedback.text() == "Copied result"
        assistant.update_status(
            {
                "phase": "transcribing",
                "assistant_stream": {
                    "mode": "edit",
                    "text": "Partial",
                    "thinking": "Voice thoughts",
                },
            }
        )
        assert assistant.editor.thinking.toPlainText() == "Voice thoughts"
        assert not window.isVisible()
    finally:
        window.close()


@pytest.mark.parametrize("allowed", [False, True])
def test_review_dialog_shows_literal_text_and_defaults_to_no(monkeypatch, allowed):
    _, window, assistant = page()

    def examine(dialog):
        assert dialog.textFormat() == Qt.TextFormat.PlainText
        assert dialog.text() == "Literal <example>"
        assert dialog.defaultButton() is dialog.button(QMessageBox.StandardButton.No)
        assert not dialog.isVisible()
        return QMessageBox.StandardButton.Yes if allowed else QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "exec", examine)
    try:
        assert assistant.editor.confirm_command("Literal <example>") == allowed
    finally:
        window.close()
