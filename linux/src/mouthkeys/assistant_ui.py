"""Command/Edit review, configured automatic commands, and captured-selection acceptance."""

import copy
import json
import threading
import uuid
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .assistance import execute_command, propose, requires_confirmation
from .assistant_history import (
    AssistantHistory,
    append_private_messages,
    context_messages,
    pending_executions,
    record_private_execution,
    validate_messages,
)
from .config import Config


class AssistantTask(QThread):
    done = Signal(object)

    def __init__(self, function, parent, cancel_event=None):
        super().__init__(parent)
        self.function = function
        self.cancel_event = cancel_event or threading.Event()

    def run(self):
        try:
            self.done.emit({"result": self.function()})
        except Exception as exc:
            self.done.emit({"error": str(exc)})


class AssistantPage(QWidget):
    streamed = Signal(str, str)
    thought_streamed = Signal(str, str)

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.jobs = set()
        self.last_proposal = None
        self.proposal = None
        self.request_task = None
        self.request_id = None
        self.session = None
        self.memory_sessions = {}
        self.request_context = None
        self.pending_result = None
        self.command_steps = 0
        self.last_selection = None
        self.next_request = None
        self.waiting_voice = False
        self.chain_config = None
        self.proposal_config = None
        self.streamed.connect(self.show_delta)
        self.thought_streamed.connect(self.show_thinking)
        layout = QVBoxLayout(self)
        hint = QLabel(
            "Uses the AI provider configured in App Settings. Command proposes a shell command "
            "for review unless automatic commands are explicitly enabled; Edit rewrites pasted "
            "text or generates new text. Captured selections support reviewed replacement."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem("Command", "command")
        self.mode.addItem("Edit / Generate", "edit")
        form.addRow("Mode", self.mode)
        chats = QHBoxLayout()
        self.chats = QComboBox()
        self.chats.currentIndexChanged.connect(self.switch_chat)
        chats.addWidget(self.chats, 1)
        for label, handler in (
            ("New", self.new_chat),
            ("Rename", self.rename_chat),
            ("Delete", self.delete_chat),
        ):
            button = QPushButton(label)
            button.clicked.connect(handler)
            chats.addWidget(button)
        form.addRow("Conversation", chats)
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
        cancel = QPushButton("Cancel recording")
        cancel.clicked.connect(lambda: window.send("cancel"))
        actions.addWidget(cancel)
        self.cancel_request_button = QPushButton("Cancel request")
        self.cancel_request_button.setEnabled(False)
        self.cancel_request_button.clicked.connect(self.cancel_request)
        actions.addWidget(self.cancel_request_button)
        layout.addLayout(actions)
        self.feedback = QLabel("AI tools require an enabled provider and model.")
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        self.result = QPlainTextEdit()
        self.result.setReadOnly(True)
        self.result.setPlaceholderText("Proposal or edited text appears here.")
        layout.addWidget(self.result, 1)
        self.conversation = QPlainTextEdit()
        self.conversation.setReadOnly(True)
        self.conversation.setMaximumHeight(130)
        self.conversation.setPlaceholderText("Conversation context appears here.")
        layout.addWidget(self.conversation)
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
        self.run_button.clicked.connect(self.run_command)
        footer.addWidget(self.run_button)
        self.replace_button = QPushButton("Replace captured selection")
        self.replace_button.setEnabled(False)
        self.replace_button.clicked.connect(lambda: self.accept_edit(False))
        footer.addWidget(self.replace_button)
        self.return_replace_button = QPushButton("Return to target and replace…")
        self.return_replace_button.setEnabled(False)
        self.return_replace_button.clicked.connect(lambda: self.accept_edit(True))
        footer.addWidget(self.return_replace_button)
        layout.addLayout(footer)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.reload_chat()

    def mode_changed(self):
        self.reset_proposal()
        self.reload_chat()

    def reload_chat(self):
        if self.jobs:
            return
        mode = self.mode.currentData()
        try:
            if Config.load().save_history == "on":
                store = AssistantHistory()
                self.session = store.current(mode)
                rows = store.rows(mode)
            else:
                self.session = self.memory_sessions.setdefault(
                    mode, {"id": None, "mode": mode, "title": "Private chat", "messages": []}
                )
                rows = [self.session]
            self.chats.blockSignals(True)
            self.chats.clear()
            for row in rows:
                self.chats.addItem(row["title"], row["id"])
            self.chats.setCurrentIndex(max(0, self.chats.findData(self.session["id"])))
            self.chats.blockSignals(False)
            self.render_conversation()
        except Exception as exc:
            self.feedback.setText(str(exc))

    def render_conversation(self):
        self.conversation.setPlainText(
            "\n\n".join(
                f"{message['role'].capitalize()}: {message['content']}"
                for message in self.session["messages"]
            )
            if self.session
            else ""
        )

    def switch_chat(self, *_):
        if self.jobs or not self.chats.currentData():
            return
        try:
            self.session = AssistantHistory().select(self.chats.currentData())
            self.reset_proposal()
            self.render_conversation()
        except Exception as exc:
            self.feedback.setText(str(exc))

    def reset_proposal(self):
        self.proposal = None
        self.request_context = None
        self.pending_result = None
        self.command_steps = 0
        self.last_selection = None
        self.result.clear()
        self.command.clear()
        self.selection.clear()
        self.run_button.setEnabled(False)
        self.replace_button.setEnabled(False)
        self.return_replace_button.setEnabled(False)
        self.chain_config = None
        self.proposal_config = None

    def new_chat(self):
        if self.jobs:
            return
        if Config.load().save_history == "on":
            AssistantHistory().new(self.mode.currentData())
        else:
            self.memory_sessions.pop(self.mode.currentData(), None)
        self.reset_proposal()
        self.reload_chat()

    def rename_chat(self):
        if self.jobs or not self.session or not self.session["id"]:
            return
        title, accepted = QInputDialog.getText(
            self, "Rename chat", "Title", text=self.session["title"]
        )
        if accepted:
            try:
                AssistantHistory().rename(self.session["id"], title)
                self.reload_chat()
            except Exception as exc:
                self.feedback.setText(str(exc))

    def delete_chat(self):
        if self.jobs or not self.session:
            return
        if (
            QMessageBox.question(self, "Delete chat", "Delete this conversation?")
            != QMessageBox.StandardButton.Yes
        ):
            return
        if self.session["id"]:
            AssistantHistory().delete(self.session["id"])
        else:
            self.memory_sessions.pop(self.mode.currentData(), None)
        self.reset_proposal()
        self.reload_chat()

    def _task(self, function, callback, cancel_event=None, *, accept_cancelled=False):
        task = AssistantTask(function, self, cancel_event)
        self.jobs.add(task)
        # Main window must not destroy workers while a provider or command runs.
        self.window.jobs.add(task)
        self.generate_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.replace_button.setEnabled(False)
        self.return_replace_button.setEnabled(False)
        self.mode.setEnabled(False)
        self.chats.setEnabled(False)

        def done(response):
            if task.cancel_event.is_set() and not accept_cancelled:
                return
            if "error" in response:
                self.feedback.setText(response["error"])
                self.request_context = None
            else:
                callback(response["result"])

        def finished():
            self.jobs.discard(task)
            self.window.jobs.discard(task)
            self.generate_button.setEnabled(True)
            self.mode.setEnabled(True)
            self.chats.setEnabled(True)
            self.run_button.setEnabled(bool(self.proposal and self.proposal.get("command")))
            can_replace = bool(self.proposal and self.proposal.get("can_accept_edit"))
            self.replace_button.setEnabled(can_replace)
            self.return_replace_button.setEnabled(can_replace)
            if self.request_task is task:
                self.request_task = None
                self.request_id = None
                self.cancel_request_button.setEnabled(False)
                if task.cancel_event.is_set():
                    self.request_context = None
                    self.feedback.setText("Request cancelled")
            continuation, self.next_request = self.next_request, None
            if continuation is not None and not task.cancel_event.is_set():
                continuation()

        task.done.connect(done)
        task.finished.connect(finished)
        task.finished.connect(task.deleteLater)
        task.start()
        return task

    def record(self):
        if self.jobs:
            return
        data = {"mode": self.mode.currentData(), "selected_text": self.current_selection()}
        if Config.load().save_history == "off" and self.session:
            data["assistant_history"] = context_messages(self.session["messages"])
            data["assistant_executions"] = [
                {"message_count": len(data["assistant_history"]), "result": result}
                for result in pending_executions(self.session)
            ]
        self.window.send(
            "start",
            data,
        )
        self.proposal = None
        self.run_button.setEnabled(False)
        self.replace_button.setEnabled(False)
        self.return_replace_button.setEnabled(False)
        self.waiting_voice = True
        self.feedback.setText("Recording instruction; press Stop when finished.")

    def current_selection(self):
        selection = self.selection.toPlainText()
        if (
            self.mode.currentData() == "edit"
            and self.session
            and self.session["messages"]
            and selection == self.last_selection
        ):
            return ""
        return selection

    def generate(self, checked=False, *, continuation=False):
        if self.jobs or self.waiting_voice:
            return
        try:
            mode = self.mode.currentData()
            if continuation and self.chain_config:
                config = self.chain_config
            else:
                from .routing import select_prompt

                config = select_prompt(Config.load(), mode=mode, routes=[])
                self.chain_config = config
            self.proposal_config = config
            instruction = self.instruction.text()
            selection = self.current_selection()
            self.reload_chat()
            session = dict(self.session)
            history = context_messages(session["messages"])
        except Exception as exc:
            self.feedback.setText(str(exc))
            return
        self.feedback.setText("Generating proposal…")
        self.proposal = None
        self.command.clear()
        self.result.clear()
        self.thinking.clear()
        self.thinking.setVisible(config.ai_show_thinking == "on")
        if not continuation:
            self.command_steps = 0
        observed = pending_executions(session)
        if self.pending_result is not None and self.pending_result not in observed:
            observed.append(self.pending_result)
        if observed:
            instruction = json.dumps(
                {
                    "instruction": "Assess the observed result; propose the next step or finish."
                    if continuation
                    else instruction,
                    "observed_command_results": observed,
                },
                ensure_ascii=False,
            )
        self.request_context = (session, config.save_history == "on", selection)
        cancelled = threading.Event()
        request_id = self.request_id = uuid.uuid4().hex

        def on_delta(piece):
            self.streamed.emit(request_id, piece)

        options = {}
        if history:
            options["history"] = history
        if config.ai_show_thinking == "on":
            options["on_thinking"] = lambda piece: self.thought_streamed.emit(request_id, piece)
        self.request_task = self._task(
            lambda: propose(
                instruction,
                config,
                mode,
                selection,
                resolved_config=True,
                cancel_event=cancelled,
                on_delta=on_delta,
                **options,
            ),
            self.show_proposal,
            cancelled,
        )
        self.cancel_request_button.setEnabled(True)

    def show_delta(self, request_id, piece):
        if (
            request_id != self.request_id
            or self.request_task is None
            or self.request_task.cancel_event.is_set()
        ):
            return
        cursor = self.result.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(piece)
        self.result.setTextCursor(cursor)

    def show_thinking(self, request_id, piece):
        if (
            request_id != self.request_id
            or self.request_task is None
            or self.request_task.cancel_event.is_set()
        ):
            return
        cursor = self.thinking.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(piece)
        self.thinking.setTextCursor(cursor)

    def cancel_request(self):
        if self.request_task is None and self.waiting_voice:
            self.window.send("cancel")
            self.cancel_request_button.setEnabled(False)
            self.feedback.setText("Voice request cancelled")
            return
        if self.request_task is not None:
            self.request_task.cancel_event.set()
            self.request_id = None
            self.cancel_request_button.setEnabled(False)
            self.feedback.setText("Request cancelled; waiting for connection to close")

    def show_proposal(self, proposal):
        proposal = dict(proposal)
        typed = self.request_context is not None
        if not typed:
            self.chain_config = None
            self.proposal_config = None
        incoming_mode = proposal.get("mode")
        if incoming_mode in ("command", "edit") and incoming_mode != self.mode.currentData():
            self.mode.blockSignals(True)
            self.mode.setCurrentIndex(self.mode.findData(incoming_mode))
            self.mode.blockSignals(False)
            self.request_context = None
            self.last_selection = None
            self.pending_result = None
        request = proposal.pop("request", None)
        response = proposal.pop("response", None)
        if self.request_context is not None and request is not None and response is not None:
            session, persistent, selection = self.request_context
            self.last_selection = selection or self.last_selection
            messages = [
                {"role": "user", "content": request},
                {"role": "assistant", "content": response},
            ]
            try:
                if persistent:
                    AssistantHistory().append(session["id"], messages)
                    self.session = AssistantHistory().get(session["id"])
                else:
                    append_private_messages(session, messages)
                    self.session = self.memory_sessions[session["mode"]] = session
                self.render_conversation()
                self.pending_result = None
            except Exception as exc:
                self.feedback.setText(f"Result ready; could not save conversation: {exc}")
            self.request_context = None
        elif proposal.get("session_id") and Config.load().save_history == "on":
            try:
                self.session = AssistantHistory().select(proposal["session_id"])
                self.reload_chat()
            except Exception as exc:
                self.feedback.setText(str(exc))
        if "private_messages" in proposal and Config.load().save_history == "off":
            try:
                messages = validate_messages(proposal.pop("private_messages"))
                self.session = {
                    "id": None,
                    "mode": proposal["mode"],
                    "title": "Private chat",
                    "messages": messages,
                    "executions": proposal.pop("private_executions", []),
                }
                self.memory_sessions[proposal["mode"]] = self.session
                self.reload_chat()
            except ValueError as exc:
                self.feedback.setText(str(exc))
        self.request_context = None
        self.proposal = proposal
        if self.session and self.session["messages"]:
            try:
                previous = json.loads(self.session["messages"][-2]["content"])
                self.last_selection = previous.get("selected_text") or self.last_selection
            except (ValueError, TypeError, AttributeError, IndexError):
                pass
        self.result.setPlainText(proposal["text"])
        self.command.setPlainText(proposal.get("command", ""))
        self.directory.setText(proposal.get("working_directory", ""))
        command_mode = proposal["mode"] == "command"
        self.command.setVisible(command_mode)
        self.directory.setVisible(command_mode)
        if proposal.get("thinking") and Config.load().ai_show_thinking == "on":
            self.thinking.setPlainText(proposal["thinking"])
            self.thinking.show()
        else:
            self.thinking.hide()
        self.run_button.setEnabled(command_mode and bool(proposal.get("command")))
        can_replace = bool(not command_mode and proposal.get("can_accept_edit"))
        self.replace_button.setEnabled(can_replace)
        self.return_replace_button.setEnabled(can_replace)
        self.feedback.setText(
            "Review command and its effects before running."
            if command_mode and proposal.get("command")
            else "Command result ready."
            if command_mode
            else "Edited text ready. Review before replacing the captured selection."
            if can_replace
            else "Edited text ready. Copy it when satisfied."
        )
        if (
            typed
            and command_mode
            and proposal.get("command")
            and self.proposal_config
            and not requires_confirmation(proposal["command"], self.proposal_config.command_confirm)
        ):
            self.next_request = lambda: self.run_command(automatic=True)

    def update_status(self, status):
        if self.jobs:
            return
        stream = status.get("assistant_stream")
        if stream and stream.get("mode") in ("edit", "command"):
            self.proposal = None
            self.request_context = None
            self.chain_config = None
            self.proposal_config = None
            self.run_button.setEnabled(False)
            self.replace_button.setEnabled(False)
            self.return_replace_button.setEnabled(False)
            self.waiting_voice = True
            self.cancel_request_button.setEnabled(True)
            self.result.setPlainText(stream.get("text", ""))
            self.feedback.setText("Generating voice instruction result…")
            if stream.get("thinking") and Config.load().ai_show_thinking == "on":
                self.thinking.setPlainText(stream["thinking"])
                self.thinking.show()
            else:
                self.thinking.hide()
        proposal = status.get("assistant_proposal")
        if status.get("phase") == "idle" and not status.get("pending_jobs"):
            self.waiting_voice = False
            self.cancel_request_button.setEnabled(False)
        identity = (status.get("id"), proposal)
        if (
            proposal
            and proposal.get("mode") in ("command", "edit")
            and identity != self.last_proposal
        ):
            self.last_proposal = identity
            self.show_proposal(proposal)

    def accept_edit(self, return_to_target):
        if self.jobs or not self.proposal or not self.proposal.get("can_accept_edit"):
            return
        if (
            return_to_target
            and QMessageBox.question(
                self,
                "Return to captured destination?",
                "Activate the captured destination and replace only its unchanged selected text?",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        edit_id = self.proposal.get("edit_id")
        self.proposal["can_accept_edit"] = False
        self.replace_button.setEnabled(False)
        self.return_replace_button.setEnabled(False)
        self.feedback.setText("Replacing reviewed selection…")

        def complete(response):
            if "error" in response:
                self.feedback.setText(response["error"])
            else:
                status = response.get("result", {})
                self.feedback.setText(status.get("message", "Edit replacement finished"))

        self.window.send(
            "accept_edit", {"id": edit_id, "return_to_target": return_to_target}, callback=complete
        )

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

    def run_command(self, checked=False, *, automatic=False):
        if (
            self.jobs
            or self.waiting_voice
            or not self.proposal
            or self.proposal["mode"] != "command"
        ):
            return
        command = self.command.toPlainText().strip()
        directory = self.directory.text().strip() or str(Path.cwd())
        if directory:
            # Confirm directory together with the exact editable command.
            reviewed = f"Working directory: {directory}\n\n{command}"
        else:
            reviewed = command
        config = self.proposal_config or Config.load()
        if not command or (
            not (automatic and not requires_confirmation(command, config.command_confirm))
            and not self.confirm_command(reviewed)
        ):
            return
        execution_session = self.session
        persistent = config.save_history == "on" and execution_session and execution_session["id"]
        if execution_session and not persistent:
            execution_session = copy.deepcopy(execution_session)
            execution_session.update(id=None, title="Private chat")
            self.session = self.memory_sessions[execution_session["mode"]] = execution_session
        self.feedback.setText(
            "Running configured automatic command…" if automatic else "Running reviewed command…"
        )

        def complete(result):
            self.command_steps += 1
            self.pending_result = {"command": command, "working_directory": directory, **result}
            self.proposal = None
            saved = True
            try:
                if persistent and Config.load().save_history == "on":
                    store = AssistantHistory()
                    store.record_execution(execution_session["id"], self.pending_result)
                    self.session = store.get(execution_session["id"])
                elif execution_session:
                    execution_session.update(id=None, title="Private chat")
                    record_private_execution(execution_session, self.pending_result)
                    self.session = self.memory_sessions[execution_session["mode"]] = (
                        execution_session
                    )
            except Exception as exc:
                saved = False
                self.feedback.setText(f"Command finished; result could not be saved: {exc}")
            self.result.setPlainText(result["output"])
            suffix = " — timed out" if result["timed_out"] else ""
            if result["truncated"]:
                suffix += " — output limit reached"
            if saved:
                self.feedback.setText(f"Exit status {result['returncode']}{suffix}")
            # Replaying a finished proposal must never execute the same command twice.
            self.proposal = None
            if (
                saved
                and not cancelled.is_set()
                and config.command_continue == "on"
                and self.command_steps < 20
            ):
                self.next_request = lambda: self.generate(continuation=True)
            elif self.command_steps >= 20:
                self.feedback.setText("20-step limit reached; review results before continuing.")

        options = {"working_directory": directory} if directory else {}
        cancelled = threading.Event()
        options["cancel_event"] = cancelled
        self.request_task = self._task(
            lambda: execute_command(command, **options), complete, cancelled, accept_cancelled=True
        )
        self.cancel_request_button.setEnabled(True)
