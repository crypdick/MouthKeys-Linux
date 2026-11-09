"""Explicitly reviewed AI Command/Edit tools; no automatic execution or insertion."""

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .assistance import execute_command, propose
from .config import Config


class AssistantTask(QThread):
    done = Signal(object)

    def __init__(self, function, parent):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            self.done.emit({"result": self.function()})
        except Exception as exc:
            self.done.emit({"error": str(exc)})


class AssistantPage(QWidget):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.jobs = set()
        self.last_proposal = None
        self.proposal = None
        layout = QVBoxLayout(self)
        hint = QLabel(
            "Uses the AI provider configured in App Settings. Command proposes a shell command "
            "for your review; Edit rewrites pasted text or generates new text. Nothing runs "
            "or gets inserted into another app automatically."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem("Command", "command")
        self.mode.addItem("Edit / Generate", "edit")
        form.addRow("Mode", self.mode)
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
        actions = QHBoxLayout()
        self.generate_button = QPushButton("Generate proposal")
        self.generate_button.clicked.connect(self.generate)
        actions.addWidget(self.generate_button)
        record = QPushButton("Record instruction")
        record.clicked.connect(self.record)
        actions.addWidget(record)
        stop = QPushButton("Stop recording")
        stop.clicked.connect(lambda: window.send("stop"))
        actions.addWidget(stop)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(lambda: window.send("cancel"))
        actions.addWidget(cancel)
        layout.addLayout(actions)
        self.feedback = QLabel("AI tools require an enabled provider and model.")
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        self.result = QPlainTextEdit()
        self.result.setReadOnly(True)
        self.result.setPlaceholderText("Proposal or edited text appears here.")
        layout.addWidget(self.result, 1)
        self.command = QPlainTextEdit()
        self.command.setPlaceholderText("Review and optionally edit proposed shell command here.")
        self.command.setMaximumHeight(90)
        self.command.hide()
        layout.addWidget(self.command)
        footer = QHBoxLayout()
        copy = QPushButton("Copy result")
        copy.clicked.connect(self.copy_result)
        footer.addWidget(copy)
        self.run_button = QPushButton("Run reviewed command…")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_command)
        footer.addWidget(self.run_button)
        layout.addLayout(footer)

    def _task(self, function, callback):
        task = AssistantTask(function, self)
        self.jobs.add(task)
        # Main window must not destroy workers while a provider or command runs.
        self.window.jobs.add(task)
        self.generate_button.setEnabled(False)
        self.run_button.setEnabled(False)

        def done(response):
            if "error" in response:
                self.feedback.setText(response["error"])
            else:
                callback(response["result"])

        def finished():
            self.jobs.discard(task)
            self.window.jobs.discard(task)
            self.generate_button.setEnabled(True)
            self.run_button.setEnabled(bool(self.proposal and self.proposal.get("command")))

        task.done.connect(done)
        task.finished.connect(finished)
        task.finished.connect(task.deleteLater)
        task.start()

    def record(self):
        self.window.send(
            "start",
            {"mode": self.mode.currentData(), "selected_text": self.selection.toPlainText()},
        )
        self.feedback.setText("Recording instruction; press Stop when finished.")

    def generate(self):
        if self.jobs:
            return
        try:
            config = Config.load()
            instruction = self.instruction.text()
            mode = self.mode.currentData()
            selection = self.selection.toPlainText()
        except Exception as exc:
            self.feedback.setText(str(exc))
            return
        self.feedback.setText("Generating proposal…")
        self._task(lambda: propose(instruction, config, mode, selection), self.show_proposal)

    def show_proposal(self, proposal):
        self.proposal = proposal
        self.result.setPlainText(proposal["text"])
        self.command.setPlainText(proposal.get("command", ""))
        command_mode = proposal["mode"] == "command"
        self.command.setVisible(command_mode)
        self.run_button.setEnabled(command_mode and bool(proposal.get("command")))
        self.feedback.setText(
            "Review command and its effects before running."
            if command_mode
            else "Edited text ready. Copy it when satisfied."
        )

    def update_status(self, status):
        proposal = status.get("assistant_proposal")
        if proposal and proposal != self.last_proposal:
            self.last_proposal = proposal
            self.show_proposal(proposal)

    def copy_result(self):
        text = self.result.toPlainText()
        if text:
            QGuiApplication.clipboard().setText(text)
            self.feedback.setText("Copied result")

    def confirm_command(self, command):
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Run this shell command?")
        dialog.setTextFormat(Qt.TextFormat.PlainText)
        dialog.setText(command)
        dialog.setInformativeText(
            "Runs as your user through /bin/sh. Review paths, destructive effects, and privileges."
        )
        dialog.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        dialog.setDefaultButton(QMessageBox.StandardButton.No)
        return dialog.exec() == QMessageBox.StandardButton.Yes

    def run_command(self):
        if self.jobs or not self.proposal or self.proposal["mode"] != "command":
            return
        command = self.command.toPlainText().strip()
        if not command or not self.confirm_command(command):
            return
        self.feedback.setText("Running reviewed command…")

        def complete(result):
            self.result.setPlainText(result["output"])
            suffix = " — timed out" if result["timed_out"] else ""
            if result["truncated"]:
                suffix += " — output limit reached"
            self.feedback.setText(f"Exit status {result['returncode']}{suffix}")

        self._task(lambda: execute_command(command), complete)
