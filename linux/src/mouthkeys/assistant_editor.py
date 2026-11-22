"""Build the assistant editor and render proposals without running their actions."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from .assistant_chats import AssistantChats


@dataclass(frozen=True)
class EditorActions:
    generate: Callable[[], None]
    record: Callable[[], None]
    stop_recording: Callable[[], None]
    cancel_recording: Callable[[], None]
    cancel_request: Callable[[], None]
    run_command: Callable[[], None]
    accept_edit: Callable[[], None]
    return_and_replace: Callable[[], None]


class AssistantEditor:
    def __init__(
        self, layout: QVBoxLayout, chats: AssistantChats, feedback: QLabel, actions: EditorActions
    ) -> None:
        self.parent = chats.parent
        self.chats = chats
        self.feedback = feedback
        self._build_inputs(layout, chats, actions)
        self._build_output(layout, chats, actions)

    def _build_inputs(
        self, layout: QVBoxLayout, chats: AssistantChats, actions: EditorActions
    ) -> None:
        hint = QLabel(
            "Uses the AI provider configured in App Settings. Command proposes a shell command "
            "for review unless automatic commands are explicitly enabled; Edit rewrites pasted "
            "text or generates new text. Captured selections support reviewed replacement."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        form.addRow("Mode", chats.mode)
        form.addRow("Conversation", chats.controls)
        self.selection = QPlainTextEdit()
        self.selection.setPlaceholderText(
            "Paste text to edit here. Leave empty to generate content."
        )
        self.selection.setMaximumHeight(110)
        form.addRow("Selected text", self.selection)
        self.instruction = QLineEdit()
        self.instruction.setPlaceholderText("Type an instruction, or record one below")
        form.addRow("Instruction", self.instruction)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.generate_button = QPushButton("Generate proposal")
        self.generate_button.clicked.connect(actions.generate)
        buttons.addWidget(self.generate_button)
        record = QPushButton("Record instruction")
        record.clicked.connect(actions.record)
        buttons.addWidget(record)
        stop = QPushButton("Stop recording")
        stop.clicked.connect(actions.stop_recording)
        buttons.addWidget(stop)
        cancel = QPushButton("Cancel recording")
        cancel.clicked.connect(actions.cancel_recording)
        buttons.addWidget(cancel)
        self.cancel_request_button = QPushButton("Cancel request")
        self.cancel_request_button.setEnabled(False)
        self.cancel_request_button.clicked.connect(actions.cancel_request)
        buttons.addWidget(self.cancel_request_button)
        layout.addLayout(buttons)

    def _build_output(
        self, layout: QVBoxLayout, chats: AssistantChats, actions: EditorActions
    ) -> None:
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        self.result = QPlainTextEdit()
        self.result.setReadOnly(True)
        self.result.setPlaceholderText("Proposal or edited text appears here.")
        layout.addWidget(self.result, 1)
        layout.addWidget(chats.transcript)
        self.thinking = QPlainTextEdit()
        self.thinking.setReadOnly(True)
        self.thinking.setMaximumHeight(90)
        self.thinking.setPlaceholderText("External provider reasoning (display only)")
        self.thinking.hide()
        layout.addWidget(self.thinking)
        self.command = QPlainTextEdit()
        self.command.setPlaceholderText("Review and optionally edit proposed shell command here.")
        self.command.setMaximumHeight(90)
        self.command.hide()
        layout.addWidget(self.command)
        self.directory = QLineEdit()
        self.directory.setPlaceholderText("Working directory (empty uses MouthKeys directory)")
        self.directory.hide()
        layout.addWidget(self.directory)
        footer = QHBoxLayout()
        copy = QPushButton("Copy result")
        copy.clicked.connect(self.copy_result)
        footer.addWidget(copy)
        self.run_button = QPushButton("Run reviewed command…")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(actions.run_command)
        footer.addWidget(self.run_button)
        self.replace_button = QPushButton("Replace captured selection")
        self.replace_button.setEnabled(False)
        self.replace_button.clicked.connect(lambda _checked=False: actions.accept_edit())
        footer.addWidget(self.replace_button)
        self.return_replace_button = QPushButton("Return to target and replace…")
        self.return_replace_button.setEnabled(False)
        self.return_replace_button.clicked.connect(
            lambda _checked=False: actions.return_and_replace()
        )
        footer.addWidget(self.return_replace_button)
        layout.addLayout(footer)

    def display(
        self, proposal: Mapping[str, Any], *, show_thinking: bool, history_notice: str | None
    ) -> None:
        self.result.setPlainText(proposal["text"])
        self.command.setPlainText(proposal.get("command", ""))
        self.directory.setText(proposal.get("working_directory", ""))
        command_mode = proposal["mode"] == "command"
        self.command.setVisible(command_mode)
        self.directory.setVisible(command_mode)
        self._display_thinking(proposal.get("thinking", ""), visible=show_thinking)
        can_replace = bool(not command_mode and proposal.get("can_accept_edit"))
        self.set_review_actions(
            run=command_mode and bool(proposal.get("command")), replace=can_replace
        )
        outcome = (
            "Review command and its effects before running."
            if command_mode and proposal.get("command")
            else "Command result ready."
            if command_mode
            else "Edited text ready. Review before replacing the captured selection."
            if can_replace
            else "Edited text ready. Copy it when satisfied."
        )
        self.feedback.setText(history_notice or outcome)

    def set_review_actions(self, *, run: bool = False, replace: bool = False) -> None:
        self.run_button.setEnabled(run)
        self.replace_button.setEnabled(replace)
        self.return_replace_button.setEnabled(replace)

    def set_request_pending(
        self, *, pending: bool, proposal: Mapping[str, Any] | None = None
    ) -> None:
        self.generate_button.setEnabled(not pending)
        self.chats.mode.setEnabled(not pending)
        self.chats.picker.setEnabled(not pending)
        self.set_review_actions(
            run=bool(proposal and proposal.get("command")),
            replace=bool(proposal and proposal.get("can_accept_edit")),
        )

    def clear_proposal(self) -> None:
        self.result.clear()
        self.command.clear()
        self.selection.clear()
        self.set_review_actions()

    def begin_generation(self, *, show_thinking: bool) -> None:
        self.feedback.setText("Generating proposal…")
        self.command.clear()
        self.result.clear()
        self.thinking.clear()
        self.thinking.setVisible(show_thinking)

    def display_voice_stream(self, stream: dict[str, Any], *, show_thinking: bool) -> None:
        self.result.setPlainText(stream.get("text", ""))
        self.feedback.setText("Generating voice instruction result…")
        self._display_thinking(stream.get("thinking", ""), visible=show_thinking)

    def _display_thinking(self, text: object, *, visible: bool) -> None:
        if visible and isinstance(text, str) and text:
            self.thinking.setPlainText(text)
            self.thinking.show()
        else:
            self.thinking.hide()

    def append_stream(self, piece: str, *, thinking: bool = False) -> None:
        view = self.thinking if thinking else self.result
        cursor = view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(piece)
        view.setTextCursor(cursor)

    def copy_result(self, *_: object) -> None:
        text = self.result.toPlainText()
        if text:
            QGuiApplication.clipboard().setText(text)
            self.feedback.setText("Copied result")

    def confirm_command(self, command: str) -> bool:
        dialog = QMessageBox(self.parent)
        dialog.setWindowTitle("Run this shell command?")
        dialog.setTextFormat(Qt.TextFormat.PlainText)
        dialog.setText(command)
        dialog.setInformativeText(
            "Runs as your user through /bin/sh. Review paths, destructive effects, and privileges."
        )
        dialog.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        dialog.setDefaultButton(QMessageBox.StandardButton.No)
        return dialog.exec() == QMessageBox.StandardButton.Yes
