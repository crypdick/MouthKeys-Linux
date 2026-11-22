"""Voice Command/Edit sessions, bounded progress, and completed proposals."""

import json
import uuid
from _thread import RLock
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from . import assistance, assistant_history
from .ai_providers import validate_history
from .config import Config
from .control import MAX_MESSAGE
from .recording_job import DictationJob

STREAM_LIMIT = 131072
THINKING_LIMIT = 65536
PROPOSAL_LIMIT = MAX_MESSAGE - STREAM_LIMIT
STEP_LIMIT = 20
CONTINUATION = "Assess the observed result; propose the next step or finish."


@dataclass(frozen=True)
class InlineContext:
    messages: list[dict[str, str]]
    executions: list[dict[str, Any]]


def inline_context(data: Mapping[str, object], mode: str, config: Config) -> InlineContext | None:
    if "assistant_history" not in data:
        if "assistant_executions" in data:
            raise ValueError("Execution context requires inline assistant history")
        return None
    messages = validate_history(data["assistant_history"])
    if mode not in ("command", "edit") or config.save_history != "off":
        raise ValueError("Inline assistant history requires private Command/Edit mode")
    executions = assistant_history.validate_executions(
        data.get("assistant_executions", []), len(messages)
    )
    return InlineContext(messages, executions)


@dataclass
class _Session:
    entry: dict[str, Any]
    history: assistant_history.AssistantHistory | None = None

    @classmethod
    def open(cls, job: DictationJob, config: Config) -> "_Session":
        if config.save_history == "on" and job.assistant_session_id is not None:
            history = assistant_history.AssistantHistory()
            return cls(history.get(job.assistant_session_id), history)
        if config.save_history == "off" and job.assistant_memory_session is not None:
            return cls(job.assistant_memory_session)
        raise ValueError("Assistant session no longer exists")

    def append(self, turns: list[dict[str, str]]) -> None:
        if self.history is None:
            assistant_history.append_private_messages(self.entry, turns)
        else:
            self.history.append(self.entry["id"], turns)
            self.entry = self.history.get(self.entry["id"])

    def record_execution(self, receipt: dict[str, object]) -> None:
        if self.history is None:
            assistant_history.record_private_execution(self.entry, receipt)
        else:
            self.history.record_execution(self.entry["id"], receipt)
            self.entry = self.history.get(self.entry["id"])

    def attach_to(self, proposal: dict[str, Any]) -> None:
        proposal["session_id"] = self.entry["id"]
        if self.history is not None:
            return
        messages = assistant_history.context_messages(self.entry["messages"])
        removed = len(self.entry["messages"]) - len(messages)
        proposal["private_messages"] = messages
        proposal["private_executions"] = _recent_receipts(self.entry.get("executions", []), removed)


def _recent_receipts(events: list[dict[str, Any]], removed: int) -> list[dict[str, Any]]:
    return [
        {"message_count": event["message_count"] - removed, "result": dict(event["result"])}
        for event in events
        if event["message_count"] >= removed
    ]


def _bound_proposal(proposal: dict[str, Any]) -> None:
    def size() -> int:
        # Status includes a separate copy of the proposal text as its preview.
        return len(json.dumps({"assistant_proposal": proposal, "preview": proposal["text"]}))

    while proposal.get("private_executions") and size() > PROPOSAL_LIMIT:
        del proposal["private_executions"][0]
    while proposal.get("private_messages") and size() > PROPOSAL_LIMIT:
        del proposal["private_messages"][:2]
        proposal["private_executions"] = _recent_receipts(proposal["private_executions"], 2)
    if size() > PROPOSAL_LIMIT:
        raise ValueError("Assistant proposal exceeds status limit; request a shorter response")


def _bounded_append(text: str, piece: str, limit: int) -> str:
    return (text + piece).encode()[:limit].decode(errors="ignore")


@dataclass
class _Progress:
    data: dict[str, str]
    thinking: str = ""
    closed: bool = False


class RecordingAssistant:
    def __init__(
        self,
        lock: RLock,
        owns_job: Callable[[DictationJob], bool],
        interrupted: Callable[[], bool],
        command_started: Callable[[DictationJob], None],
    ) -> None:
        self.lock = lock
        self.owns_job = owns_job
        self.interrupted = interrupted
        self.command_started = command_started
        self.memory: dict[str, dict[str, Any]] = {}
        self._progress: _Progress | None = None

    def bind_session(self, job: DictationJob, context: InlineContext | None) -> None:
        """Freeze the session reference before recording or waiting behind earlier jobs."""
        if job.mode not in ("command", "edit"):
            return
        if job.config.save_history == "on":
            job.assistant_session_id = assistant_history.AssistantHistory().current(job.mode)["id"]
            return
        previous = self.memory.get(job.mode)
        if context is not None and (
            previous is None
            or previous["messages"] != context.messages
            or previous.get("executions", []) != context.executions
        ):
            self.memory[job.mode] = {
                "id": uuid.uuid4().hex,
                "messages": context.messages,
                "executions": context.executions,
            }
        session = self.memory.setdefault(job.mode, {"id": uuid.uuid4().hex, "messages": []})
        job.assistant_session_id = session["id"]
        job.assistant_memory_session = session

    def stream_status(self) -> dict[str, str] | None:
        with self.lock:
            return dict(self._progress.data) if self._progress is not None else None

    def clear_stream(self) -> None:
        with self.lock:
            self._progress = None

    def _cancelled(self, job: DictationJob) -> bool:
        return job.cancel_event.is_set() or self.interrupted()

    def _stream(
        self,
        job: DictationJob,
        progress: _Progress,
        channel: Literal["text", "thinking"],
        piece: str,
    ) -> None:
        with self.lock:
            if progress.closed or self._cancelled(job):
                return
            if channel == "thinking":
                progress.thinking = _bounded_append(progress.thinking, piece, THINKING_LIMIT)
            if self._progress is not progress or not self.owns_job(job):
                return
            limit = THINKING_LIMIT if channel == "thinking" else STREAM_LIMIT
            data = progress.data
            data[channel] = _bounded_append(data[channel], piece, limit)
            # JSON escapes can expand control characters and non-ASCII text.
            while len(json.dumps(data)) > STREAM_LIMIT:
                data[channel] = data[channel][: len(data[channel]) // 2]

    def _request(
        self, job: DictationJob, config: Config, instruction: str, session: _Session
    ) -> dict[str, Any]:
        progress = _Progress({"id": job.item_id, "mode": job.mode, "text": "", "thinking": ""})
        with self.lock:
            if self.owns_job(job):
                self._progress = progress

        options: dict[str, Any] = {"resolved_config": True, "cancel_event": job.cancel_event}
        conversation = assistant_history.context_messages(session.entry["messages"])
        if conversation:
            options["history"] = conversation
        if config.ai_show_thinking == "on":
            options["on_thinking"] = lambda piece: self._stream(job, progress, "thinking", piece)
        if config.ai_streaming == "on":
            options["on_delta"] = lambda piece: self._stream(job, progress, "text", piece)
        try:
            proposal: dict[str, Any] = assistance.propose(
                instruction, config, job.mode, job.selected_text, **options
            )
        finally:
            with self.lock:
                progress.closed = True
        if self._cancelled(job):
            raise ValueError("Assistant request cancelled")
        request = proposal.pop(
            "request", json.dumps({"instruction": instruction, "selected_text": job.selected_text})
        )
        response = proposal.pop("response", proposal.get("command", proposal["text"]))
        session.append(
            [{"role": "user", "content": request}, {"role": "assistant", "content": response}]
        )
        if progress.thinking:
            proposal["thinking"] = progress.thinking
        return proposal

    def _execute(
        self, job: DictationJob, proposal: dict[str, Any], command: str, session: _Session
    ) -> None:
        directory = proposal.get("working_directory", "") or str(Path.cwd())
        with self.lock:
            if self._cancelled(job):
                raise ValueError("Assistant request cancelled before command execution")
            self.command_started(job)
        result = assistance.execute_command(
            command, working_directory=directory, cancel_event=job.cancel_event
        )
        # An attempted command may already have run, so keep its receipt even on cancellation.
        session.record_execution({"command": command, "working_directory": directory, **result})
        proposal["command"] = ""
        proposal["text"] = (
            result["output"] or f"Command finished with exit status {result['returncode']}"
        )
        proposal["executed"] = True
        if self._cancelled(job):
            raise ValueError("Assistant command cancelled; execution receipt retained")

    def propose(
        self,
        job: DictationJob,
        config: Config,
        text: str,
        can_replace: Callable[[DictationJob], bool],
    ) -> dict[str, Any]:
        session = _Session.open(job, config)
        executions = 0
        for step in range(STEP_LIMIT):
            if self._cancelled(job):
                raise ValueError("Assistant request cancelled")
            instruction = text if step == 0 else CONTINUATION
            observed = assistant_history.pending_executions(session.entry)
            if observed:
                instruction = json.dumps(
                    {"instruction": instruction, "observed_command_results": observed},
                    ensure_ascii=False,
                )
            proposal = self._request(job, config, instruction, session)
            command = proposal.get("command", "")
            if (
                job.mode != "command"
                or assistance.requires_confirmation(command, config.command_confirm)
                or not command
            ):
                break
            self._execute(job, proposal, command, session)
            executions += 1
            if config.command_continue != "on":
                break
            if step == STEP_LIMIT - 1:
                proposal["text"] = (
                    "20-step limit reached; review results before continuing.\n" + proposal["text"]
                )
        session.attach_to(proposal)
        if executions:
            proposal["execution_count"] = executions
        if job.mode == "edit":
            proposal.update(edit_id=job.item_id, can_accept_edit=can_replace(job))
        _bound_proposal(proposal)
        return proposal
