"""Manage the assistant conversation picker and its persistent or in-memory session."""

import json
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QWidget,
)

from .assistant_history import (
    AssistantHistory,
    append_private_messages,
    record_private_execution,
    validate_messages,
)
from .config import Config

ChatSession = dict[str, Any]


class AssistantChats:
    def __init__(
        self,
        parent: QWidget,
        mode: QComboBox,
        is_busy: Callable[[], bool],
        reset_proposal: Callable[[], None],
        show_status: Callable[[str], None],
    ) -> None:
        self.parent = parent
        self.mode = mode
        self.is_busy = is_busy
        self.reset_proposal = reset_proposal
        self.show_status = show_status
        self.session: ChatSession | None = None
        self.private_sessions: dict[str, ChatSession] = {}
        self.picker = QComboBox()
        self.picker.currentIndexChanged.connect(self._switch)
        self.controls = QHBoxLayout()
        self.controls.addWidget(self.picker, 1)
        for label, handler in (
            ("New", self.new),
            ("Rename", self._rename),
            ("Delete", self._delete),
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, handler=handler: handler())
            self.controls.addWidget(button)
        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setMaximumHeight(130)
        self.transcript.setPlaceholderText("Conversation context appears here.")

    def reload(self) -> ChatSession | None:
        if self.is_busy():
            return None
        mode = self.mode.currentData()
        try:
            if Config.load().save_history == "on":
                store = AssistantHistory()
                session = store.current(mode)
                rows = store.rows(mode)
            else:
                session = self.private_sessions.setdefault(
                    mode, {"id": None, "mode": mode, "title": "Private chat", "messages": []}
                )
                rows = [session]
            with QSignalBlocker(self.picker):
                self.picker.clear()
                for row in rows:
                    self.picker.addItem(row["title"], row["id"])
                self.picker.setCurrentIndex(max(0, self.picker.findData(session["id"])))
            self.session = session
            self._render()
        # Prevent a new request from using stale conversation state after a storage failure.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            self.show_status(str(exc))
            return None
        return session

    def _render(self) -> None:
        messages = self.session["messages"] if self.session else []
        self.transcript.setPlainText(
            "\n\n".join(
                f"{message['role'].capitalize()}: {message['content']}" for message in messages
            )
        )

    def _switch(self, _index: int) -> None:
        item_id = self.picker.currentData()
        if self.is_busy() or not item_id:
            return
        try:
            self.session = AssistantHistory().select(item_id)
            self.reset_proposal()
            self._render()
        # Report failures from a deleted or unavailable chat without logging its contents.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            selected = self.session["id"] if self.session else None
            with QSignalBlocker(self.picker):
                self.picker.setCurrentIndex(self.picker.findData(selected))
            self.show_status(str(exc))

    def new(self) -> None:
        if self.is_busy():
            return
        if Config.load().save_history == "on":
            AssistantHistory().new(self.mode.currentData())
        else:
            self.private_sessions.pop(self.mode.currentData(), None)
        self.reset_proposal()
        self.reload()

    def _rename(self) -> None:
        session = self.session
        if self.is_busy() or not session or not session["id"]:
            return
        title, accepted = QInputDialog.getText(
            self.parent, "Rename chat", "Title", text=session["title"]
        )
        if not accepted or not self._still_current(session):
            return
        try:
            AssistantHistory().rename(session["id"], title)
            self.reload()
        # A closed or deleted chat cannot be renamed; report the storage error in the UI.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            self.show_status(str(exc))

    def _delete(self) -> None:
        session = self.session
        if self.is_busy() or not session:
            return
        confirmed = QMessageBox.question(self.parent, "Delete chat", "Delete this conversation?")
        if confirmed != QMessageBox.StandardButton.Yes or not self._still_current(session):
            return
        if session["id"]:
            AssistantHistory().delete(session["id"])
        else:
            self.private_sessions.pop(session["mode"], None)
        self.reset_proposal()
        self.reload()

    def _still_current(self, session: ChatSession) -> bool:
        # A modal dialog runs the event loop; a new voice result may select another chat.
        if not self.is_busy() and self.session is session:
            return True
        self.show_status("Conversation changed; choose it again")
        return False

    def remember_private(self, session: ChatSession) -> None:
        session.update(id=None, title="Private chat")
        self.session = self.private_sessions[session["mode"]] = session

    def append_turn(
        self, session: ChatSession, request: str, response: str, *, persistent: bool
    ) -> None:
        messages = [
            {"role": "user", "content": request},
            {"role": "assistant", "content": response},
        ]
        # Never persist a private request or a turn finished after the user disabled history.
        if persistent and Config.load().save_history == "on":
            store = AssistantHistory()
            store.append(session["id"], messages)
            self.session = store.get(session["id"])
        else:
            self.remember_private(append_private_messages(session, messages))
        self._render()

    def record_execution(
        self, session: ChatSession | None, receipt: dict[str, Any], *, persistent: bool
    ) -> str | None:
        if session is None:
            return None
        try:
            if persistent and Config.load().save_history == "on":
                store = AssistantHistory()
                store.record_execution(session["id"], receipt)
                self.session = store.get(session["id"])
            else:
                record_private_execution(session, receipt)
                self.remember_private(session)
        # Keep the output available without putting private command results in logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            return f"Command finished; result could not be saved: {exc}"
        return None

    def adopt_voice(self, proposal: dict[str, Any]) -> str | None:
        if not (proposal.get("session_id") or "private_messages" in proposal):
            return None
        try:
            if Config.load().save_history == "on":
                if proposal.get("session_id"):
                    self.session = AssistantHistory().select(proposal["session_id"])
                    self.reload()
            elif "private_messages" in proposal:
                messages = validate_messages(proposal.pop("private_messages"))
                self.remember_private(
                    {
                        "mode": proposal["mode"],
                        "messages": messages,
                        "executions": proposal.pop("private_executions", []),
                    }
                )
                self.reload()
        # Keep the result available even when a chat cannot be loaded.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            return str(exc)
        return None

    def previous_selection(self) -> str | None:
        if not self.session or not self.session["messages"]:
            return None
        try:
            request = json.loads(self.session["messages"][-2]["content"])
        except (ValueError, TypeError, IndexError):
            return None
        selection = request.get("selected_text") if isinstance(request, dict) else None
        return selection if isinstance(selection, str) else None
