"""Command/Edit review, configured automatic commands, and captured-selection acceptance."""

import copy
import json
import threading
import uuid
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, TypeVar, cast

from PySide6.QtCore import QSignalBlocker, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from .assistance import execute_command, propose, requires_confirmation
from .assistant_chats import AssistantChats, ChatSession
from .assistant_editor import AssistantEditor, EditorActions
from .assistant_history import context_messages, pending_executions
from .config import Config
from .qt_control import Response, Send, Task
from .routing import select_prompt
from .tray_status import status_text

_Result = TypeVar("_Result")


class AssistantPage(QWidget):
    streamed = Signal(str, str)
    thought_streamed = Signal(str, str)

    def __init__(self, parent: QWidget, send: Send, owner_jobs: set[QThread]) -> None:
        super().__init__(parent)
        self.send = send
        self.owner_jobs = owner_jobs
        self.jobs: set[Task] = set()
        self.last_proposal: tuple[object, object] | None = None
        self.proposal: dict[str, Any] | None = None
        self.request_task: Task | None = None
        self.request_id: str | None = None
        self.request_context: tuple[ChatSession, bool, str] | None = None
        self.pending_result: dict[str, Any] | None = None
        self.command_steps = 0
        self.last_selection: str | None = None
        self.next_request: Callable[[], None] | None = None
        self.waiting_voice = False
        self.chain_config: Config | None = None
        self.proposal_config: Config | None = None
        self.streamed.connect(self.show_delta)
        self.thought_streamed.connect(self.show_thinking)
        layout = QVBoxLayout(self)
        self.mode = QComboBox()
        self.mode.addItem("Command", "command")
        self.mode.addItem("Edit / Generate", "edit")
        self.feedback = QLabel("AI tools require an enabled provider and model.")
        self.chats = AssistantChats(
            self, self.mode, lambda: bool(self.jobs), self.reset_proposal, self.feedback.setText
        )
        self.editor = AssistantEditor(
            layout,
            self.chats,
            self.feedback,
            EditorActions(
                generate=self.generate,
                record=self.record,
                stop_recording=lambda: send("stop", None, None),
                cancel_recording=lambda: send("cancel", None, None),
                cancel_request=self.cancel_request,
                run_command=self.run_command,
                accept_edit=partial(self.accept_edit, return_to_target=False),
                return_and_replace=partial(self.accept_edit, return_to_target=True),
            ),
        )
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.chats.reload()

    def mode_changed(self, _index: int) -> None:
        self.reset_proposal()
        self.chats.reload()

    def reset_proposal(self) -> None:
        self.proposal = None
        self.request_context = None
        self.pending_result = None
        self.command_steps = 0
        self.last_selection = None
        self.editor.clear_proposal()
        self.chain_config = None
        self.proposal_config = None

    def _task(
        self,
        function: Callable[[], _Result],
        callback: Callable[[_Result], None],
        cancel_event: threading.Event,
        *,
        accept_cancelled: bool = False,
    ) -> None:
        task = Task(function, self, cancel_event)
        self.request_task = task
        self.jobs.add(task)
        # Main window must not destroy workers while a provider or command runs.
        self.owner_jobs.add(task)
        self.editor.set_request_pending(pending=True)
        self.editor.cancel_request_button.setEnabled(True)

        def done(response: Response) -> None:
            if task.cancel_event.is_set() and not accept_cancelled:
                return
            if "error" in response:
                self.feedback.setText(str(response["error"]))
                self.request_context = None
            else:
                callback(cast("_Result", response["result"]))

        def finished() -> None:
            self.jobs.discard(task)
            self.owner_jobs.discard(task)
            self.editor.set_request_pending(pending=False, proposal=self.proposal)
            self.request_task = None
            self.request_id = None
            self.editor.cancel_request_button.setEnabled(False)
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

    def record(self) -> None:
        if self.jobs:
            return
        data: dict[str, object] = {
            "mode": self.mode.currentData(),
            "selected_text": self.current_selection(),
        }
        if Config.load().save_history == "off" and self.chats.session:
            history = context_messages(self.chats.session["messages"])
            data["assistant_history"] = history
            data["assistant_executions"] = [
                {"message_count": len(history), "result": result}
                for result in pending_executions(self.chats.session)
            ]
        self.send("start", data, None)
        self.proposal = None
        self.editor.set_review_actions()
        self.waiting_voice = True
        self.feedback.setText("Recording instruction; press Stop when finished.")

    def current_selection(self) -> str:
        selection = self.editor.selection.toPlainText()
        if (
            self.mode.currentData() == "edit"
            and self.chats.session
            and self.chats.session["messages"]
            and selection == self.last_selection
        ):
            return ""
        return selection

    def generate(self, *, continuation: bool = False) -> None:
        if self.jobs or self.waiting_voice:
            return
        try:
            mode = self.mode.currentData()
            if continuation and self.chain_config:
                config = self.chain_config
            else:
                config = select_prompt(Config.load(), mode=mode, routes=[])
                self.chain_config = config
            self.proposal_config = config
            instruction = self.editor.instruction.text()
            selection = self.current_selection()
            chat = self.chats.reload()
            if chat is None:
                return
            session = dict(chat)
            history = context_messages(session["messages"])
        # Report to the assistant UI without duplicating private data in logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            self.feedback.setText(str(exc))
            return
        self.proposal = None
        self.editor.begin_generation(show_thinking=config.ai_show_thinking == "on")
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

        options: dict[str, Any] = {}
        if history:
            options["history"] = history
        if config.ai_show_thinking == "on":
            options["on_thinking"] = partial(self.thought_streamed.emit, request_id)
        self._task(
            lambda: propose(
                instruction,
                config,
                mode,
                selection,
                resolved_config=True,
                cancel_event=cancelled,
                on_delta=partial(self.streamed.emit, request_id),
                **options,
            ),
            self.show_proposal,
            cancelled,
        )

    def _owns_request(self, request_id: str) -> bool:
        return (
            request_id == self.request_id
            and self.request_task is not None
            and not self.request_task.cancel_event.is_set()
        )

    def show_delta(self, request_id: str, piece: str) -> None:
        if self._owns_request(request_id):
            self.editor.append_stream(piece)

    def show_thinking(self, request_id: str, piece: str) -> None:
        if self._owns_request(request_id):
            self.editor.append_stream(piece, thinking=True)

    def cancel_request(self) -> None:
        if self.request_task is None and self.waiting_voice:
            self.send("cancel", None, None)
            self.editor.cancel_request_button.setEnabled(False)
            self.feedback.setText("Voice request cancelled")
            return
        if self.request_task is not None:
            self.request_task.cancel_event.set()
            self.request_id = None
            self.editor.cancel_request_button.setEnabled(False)
            self.feedback.setText("Request cancelled; waiting for connection to close")

    def show_proposal(self, payload: Mapping[str, object]) -> None:
        proposal: dict[str, Any] = dict(payload)
        typed = self.request_context is not None
        if not typed:
            self.chain_config = None
            self.proposal_config = None
        incoming_mode = proposal.get("mode")
        if incoming_mode in ("command", "edit") and incoming_mode != self.mode.currentData():
            with QSignalBlocker(self.mode):
                self.mode.setCurrentIndex(self.mode.findData(incoming_mode))
            self.request_context = None
            self.last_selection = None
            self.pending_result = None
        history_notice = self._save_turn(proposal)
        self.request_context = None
        self.proposal = proposal
        self.last_selection = self.chats.previous_selection() or self.last_selection
        command_mode = proposal["mode"] == "command"
        self.editor.display(
            proposal,
            show_thinking=bool(proposal.get("thinking")) and Config.load().ai_show_thinking == "on",
            history_notice=history_notice,
        )
        if (
            typed
            and command_mode
            and proposal.get("command")
            and self.proposal_config
            and not requires_confirmation(proposal["command"], self.proposal_config.command_confirm)
        ):
            self.next_request = lambda: self.run_command(automatic=True)

    def _save_turn(self, proposal: dict[str, Any]) -> str | None:
        request = proposal.pop("request", None)
        response = proposal.pop("response", None)
        if self.request_context is None or request is None or response is None:
            return self.chats.adopt_voice(proposal)
        session, persistent, selection = self.request_context
        self.last_selection = selection or self.last_selection
        try:
            self.chats.append_turn(session, request, response, persistent=persistent)
        # Keep the result available without logging private conversation data.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            return f"Result ready; could not save conversation: {exc}"
        self.pending_result = None
        return None

    def update_status(self, status: Mapping[str, object]) -> None:
        if self.jobs:
            return
        stream = status.get("assistant_stream")
        if isinstance(stream, dict) and stream.get("mode") in ("edit", "command"):
            self.proposal = None
            self.request_context = None
            self.chain_config = None
            self.proposal_config = None
            self.editor.set_review_actions()
            self.waiting_voice = True
            self.editor.cancel_request_button.setEnabled(True)
            self.editor.display_voice_stream(
                stream,
                show_thinking=bool(stream.get("thinking"))
                and Config.load().ai_show_thinking == "on",
            )
        proposal = status.get("assistant_proposal")
        if status.get("phase") == "idle" and not status.get("pending_jobs"):
            self.waiting_voice = False
            self.editor.cancel_request_button.setEnabled(False)
        identity = (status.get("id"), proposal)
        if (
            isinstance(proposal, dict)
            and proposal.get("mode") in ("command", "edit")
            and identity != self.last_proposal
        ):
            self.last_proposal = identity
            self.show_proposal(proposal)

    def accept_edit(self, *, return_to_target: bool) -> None:
        proposal = self.proposal
        if self.jobs or not proposal or not proposal.get("can_accept_edit"):
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
        if self.proposal is not proposal:
            return
        proposal["can_accept_edit"] = False
        self.editor.set_review_actions()
        self.feedback.setText("Replacing reviewed selection…")

        def complete(response: Response) -> None:
            if "error" in response:
                self.feedback.setText(str(response["error"]))
            else:
                status = response.get("result")
                message = "Edit replacement finished"
                self.feedback.setText(
                    status_text(status, "message", message) if isinstance(status, dict) else message
                )

        self.send(
            "accept_edit",
            {"id": proposal.get("edit_id"), "return_to_target": return_to_target},
            complete,
        )

    def run_command(self, *, automatic: bool = False) -> None:
        proposal = self.proposal
        if self.jobs or self.waiting_voice or not proposal or proposal["mode"] != "command":
            return
        command = self.editor.command.toPlainText().strip()
        directory = self.editor.directory.text().strip() or str(Path.cwd())
        # Confirm directory together with the exact editable command.
        reviewed = f"Working directory: {directory}\n\n{command}"
        config = self.proposal_config or Config.load()
        if not command or (
            not (automatic and not requires_confirmation(command, config.command_confirm))
            and not self.editor.confirm_command(reviewed)
        ):
            return
        if self.proposal is not proposal:
            return
        execution_session = self.chats.session
        persistent = config.save_history == "on" and bool(
            execution_session and execution_session["id"]
        )
        if execution_session and not persistent:
            execution_session = copy.deepcopy(execution_session)
            self.chats.remember_private(execution_session)
        self.feedback.setText(
            "Running configured automatic command…" if automatic else "Running reviewed command…"
        )

        def complete(result: dict[str, int | str | bool]) -> None:
            self.command_steps += 1
            self.pending_result = {"command": command, "working_directory": directory, **result}
            # Replaying a finished proposal must never execute the same command twice.
            self.proposal = None
            notice = self.chats.record_execution(
                execution_session, self.pending_result, persistent=persistent
            )
            self.editor.result.setPlainText(str(result["output"]))
            suffix = " — timed out" if result["timed_out"] else ""
            if result["truncated"]:
                suffix += " — output limit reached"
            self.feedback.setText(notice or f"Exit status {result['returncode']}{suffix}")
            if (
                notice is None
                and not cancelled.is_set()
                and config.command_continue == "on"
                and self.command_steps < 20
            ):
                self.next_request = lambda: self.generate(continuation=True)
            elif self.command_steps >= 20:
                self.feedback.setText("20-step limit reached; review results before continuing.")

        cancelled = threading.Event()
        self._task(
            lambda: execute_command(command, working_directory=directory, cancel_event=cancelled),
            complete,
            cancelled,
            accept_cancelled=True,
        )
