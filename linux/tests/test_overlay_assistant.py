from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette

from mouthkeys.assistant_history import AssistantHistory
from mouthkeys.config import Config
from mouthkeys.overlay import Overlay
from mouthkeys.overlay_assistant import OverlayAssistant


def panel(monkeypatch):
    widget = Overlay()
    widget.timer.stop()
    widget.animation.stop()
    monkeypatch.setattr(widget, "show", lambda: None)
    return widget


def test_passive_panel_shows_stream_and_retains_final_without_opening_review(monkeypatch):
    widget = panel(monkeypatch)
    widget.set_status(
        {
            "id": "edit",
            "phase": "transcribing",
            "mode": "edit",
            "preview": "Spoken request",
            "assistant_stream": {"id": "edit", "mode": "edit", "text": "Partial answer"},
        }
    )
    assert widget.display_text() == "Partial answer"
    assert not widget.assistant_button.isHidden()
    widget.set_status(
        {
            "id": "edit",
            "phase": "idle",
            "mode": "edit",
            "message": "Recording ready for review; nothing inserted",
            "assistant_proposal": {"mode": "edit", "text": "Final answer"},
        }
    )
    assert widget.display_text() == "Final answer"
    assert widget.dismiss_at is None
    assert widget.assistant_window is None
    assert widget.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
    assert widget.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    widget.action("cancel")
    widget.set_status(widget.data)
    assert widget.dismiss_at == 0  # Repeated status cannot undo explicit dismissal.


def test_explicit_review_reuses_assistant_page_and_preserves_panel_focus_flags(monkeypatch):
    widget = panel(monkeypatch)
    shown = []
    for method in ("show", "raise_", "activateWindow"):
        monkeypatch.setattr(
            OverlayAssistant, method, lambda self, method=method: shown.append(method)
        )
    proposal = {"mode": "command", "text": "Inspect files", "command": "pwd"}
    widget.set_status({"id": "command", "phase": "idle", "assistant_proposal": proposal})
    widget.assistant_button.click()
    host = widget.assistant_window
    assert shown == ["show", "raise_", "activateWindow"]
    assert not host.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
    assert widget.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
    assistant = host.assistant_page
    assert assistant.result.toPlainText() == "Inspect files"
    assert assistant.command.toPlainText() == "pwd"
    assert assistant.run_button.isEnabled()
    assert not assistant.command.isHidden()
    widget.open_assistant()
    assert widget.assistant_window is host
    assert not host.jobs  # Opening a proposal cannot run it.
    Config(theme="light").save()
    widget.set_status(widget.data)
    assert host.palette().color(QPalette.ColorRole.Base).name() == "#ffffff"


def test_expanded_review_has_recent_new_followup_and_live_status(monkeypatch):
    Config(save_history="on").save()
    store = AssistantHistory()
    first = store.current("edit")
    store.append(
        first["id"],
        [{"role": "user", "content": "Draft"}, {"role": "assistant", "content": "First draft"}],
    )
    second = store.new("edit")
    store.rename(second["id"], "Second chat")
    widget = panel(monkeypatch)
    host = OverlayAssistant(widget)
    widget.assistant_window = host
    assistant = host.assistant_page
    assistant.mode.setCurrentIndex(assistant.mode.findData("edit"))
    assert assistant.chats.count() == 2
    assistant.chats.setCurrentIndex(assistant.chats.findData(first["id"]))
    assert "First draft" in assistant.conversation.toPlainText()
    assistant.instruction.setText("Make it shorter")
    assert assistant.instruction.text() == "Make it shorter"
    assistant.new_chat()
    assert assistant.chats.count() == 3
    assert assistant.session["id"] not in (first["id"], second["id"])
    widget.set_status(
        {
            "id": "voice",
            "phase": "idle",
            "assistant_proposal": {
                "mode": "edit",
                "text": "Live final",
                "session_id": first["id"],
            },
        }
    )
    assert assistant.result.toPlainText() == "Live final"
    assert assistant.session["id"] == first["id"]
    assert not host.isVisible()
    assert not host.jobs


def test_disabled_passive_panel_still_updates_explicit_review(monkeypatch):
    widget = panel(monkeypatch)
    host = OverlayAssistant(widget)
    widget.assistant_window = host
    Config(recording_panel="off").save()
    status = {
        "id": "edit",
        "phase": "idle",
        "assistant_proposal": {"mode": "edit", "text": "Review remains available"},
    }
    widget.set_status(status)
    assert host.assistant_page.result.toPlainText() == "Review remains available"
    assert widget.data == status
    assert not widget.isVisible()
