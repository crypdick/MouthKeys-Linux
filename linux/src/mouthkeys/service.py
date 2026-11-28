import copy
import threading
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol, cast

from . import __version__
from .audio import CaptureSession, Recording
from .config import Config
from .delivery import Delivery, Target
from .operations import handle_action
from .recording_assistant import RecordingAssistant
from .recording_capture import CaptureFactory, RecordingCapture
from .recording_delivery import (
    RecordingDelivery,
    TranscriptDelivery,
    replay_text,
    replay_transcript,
)
from .recording_destination import FocusGate
from .recording_edit import (
    claim_edit,
    reviewed_return_target,
)
from .recording_finalizer import RecordingFinalizer
from .recording_job import DictationJob
from .recording_learning import LearningCommand, RecordingLearning
from .recording_preview import RecordingPreview
from .recording_setup import prepare_recording
from .recording_state import RecordingState
from .storage import History
from .transcribe import Transcriber

if TYPE_CHECKING:
    from .recording_inference import SpeechModel

    class _WarmupModel(Protocol):
        def load(self, config: Config) -> None: ...

    class _ReplayAdapter(Protocol):
        def copy(self, text: str) -> None: ...

        def paste_key(self, config: Config) -> None: ...


class Controller:
    def __init__(
        self,
        history: History | None = None,
        transcriber: object | None = None,
        delivery: TranscriptDelivery | None = None,
        recorder: CaptureFactory = CaptureSession,
    ) -> None:
        self.history = history or History()
        self.history.recover()
        self.transcriber = transcriber or Transcriber()
        self.delivery = delivery or Delivery()
        self.state = RecordingState(self.history)
        self.lock = self.state.lock
        self.model_lock = threading.Lock()
        self.live_preview = RecordingPreview(self.lock, self.model_lock)
        self.recording_assistant = RecordingAssistant(
            self.lock,
            lambda job: self.state.job is job,
            lambda: self.state.interrupted,
            self._automatic_command_started,
        )
        self.learning = RecordingLearning(
            self.lock, self._owns_learning, lambda: self.state.interrupted, self.report_error
        )
        self.recording_delivery = RecordingDelivery(
            self.delivery, self.lock, self.learning, lambda: self.state.interrupted
        )
        self.finalizer = RecordingFinalizer(
            self.state, self.recording_assistant, self.recording_delivery, self.model_lock
        )
        self.capture = RecordingCapture(
            self.state,
            self.delivery,
            recorder,
            self.live_preview,
            lambda job, target, automatic: self._finish(job, target, auto_paste_expected=automatic),
        )

    @property
    def worker(self) -> threading.Thread | None:
        return self.state.worker

    @property
    def recorder(self) -> Recording | None:
        return self.capture.recorder

    @property
    def config(self) -> Config:
        return self.capture.config

    @config.setter
    def config(self, config: Config) -> None:
        self.capture.config = config

    @property
    def phase(self) -> str:
        return self.state.phase

    @phase.setter
    def phase(self, phase: str) -> None:
        self.state.phase = phase

    @property
    def message(self) -> str:
        return self.state.message

    @property
    def item_id(self) -> str | None:
        return self.state.item_id

    @property
    def preview_text(self) -> str:
        return self.state.preview_text

    @property
    def notice(self) -> dict[str, object] | None:
        return self.state.notice

    @property
    def job(self) -> DictationJob | None:
        return self.state.job

    @property
    def assistant_proposal(self) -> dict[str, Any] | None:
        return self.state.assistant_proposal

    def report_error(self, message: str, kind: str = "error") -> None:
        self.state.report_error(message, kind)

    def status(self) -> dict[str, object]:
        with self.lock:
            return {
                "version": __version__,
                "pending_jobs": sum(not job.done.is_set() for job in self.state.jobs),
                "can_record": self.state.phase
                in ("idle", "recording", "transcribing", "queued", "delivering"),
                "assistant_proposal": self.state.assistant_proposal,
                "assistant_stream": self.recording_assistant.stream_status(),
                "mode": self.state.job.mode if self.state.job else "dictation",
                "loaded_model": copy.deepcopy(getattr(self.transcriber, "loaded_model", None)),
                "learning_pending": self.learning.pending(),
                "notice": self.state.notice,
                "preview": self.state.preview_text,
                "meter": self.live_preview.meter if self.state.phase == "recording" else {},
                "microphone": self.config.audio_source,
                "session_microphone": self.capture.microphone.session_source,
                "phase": self.state.phase,
                "message": self.state.message,
                "id": self.state.item_id,
                "seconds": self.capture.seconds(),
            }

    def command(self, action: str, **data: object) -> object:
        gate = FocusGate()
        result = self._command(action, _focus_gate=gate, **data)
        if (
            action in ("stop", "toggle")
            and cast("dict[str, object]", result).get("phase") == "stopping"
        ):
            gate.wait()
        return result

    def _command(self, action: str, _focus_gate: FocusGate, **data: object) -> object:
        if action == "select_microphone":
            return self.capture.select(data, self.status)
        if action == "accept_edit":
            return self._accept_edit(data)
        if action in ("copy", "paste"):
            return self._manual_delivery(action, data)
        if action in ("learning_pending", "learning_accept", "learning_dismiss"):
            return self.learning.command(cast("LearningCommand", action), data)
        with self.lock:
            if action == "toggle" and self.state.phase not in (
                "starting",
                "recording",
                "switching",
                "capture_failed",
            ):
                action = "start"
            if action != "start":
                return self._locked_command(action, data, _focus_gate)
        return self._start_recording(data)

    def _locked_command(
        self, action: str, data: Mapping[str, object], focus_gate: FocusGate
    ) -> object:
        if action in ("status", "release_hold"):
            if (
                action == "release_hold"
                and self.state.job
                and self.state.phase in ("starting", "recording", "switching")
                and data.get("id") == self.state.job.item_id
            ):
                self.state.job.held_activation = False
            return self.status()
        result = handle_action(
            self.history,
            action,
            data,
            allow_mutation=self.state.phase == "idle" and not self.pending_jobs(),
        )
        if result is not NotImplemented:
            return result
        if self.state.interrupted:
            raise ValueError("Service is shutting down")
        if action in ("cancel", "stop", "toggle"):
            if data.get("id") is not None and (
                not self.state.job or data["id"] != self.state.job.item_id
            ):
                return self.status()
            if action == "cancel" or (action == "toggle" and self.capture.is_false_start()):
                focus_gate.resolved()
                return self._cancel_command(quiet=action == "toggle" or data.get("quiet") is True)
            self.capture.stop_command(data, focus_gate)
            return self.status()
        return self._idle_command(action)

    def _cancel_command(self, *, quiet: bool = False) -> dict[str, object]:
        self.learning.cancel()
        if self.state.phase == "delivering" and self.state.job and self.state.job.send_pending:
            if self.state.job.send_dispatched:
                raise ValueError("Send already dispatched; cannot withdraw the key")
            self.state.cancelled = True
            self.state.job.cancelled = True
            self.state.job.cancel_event.set()
            self.state.message = "Send cancelled; text already inserted"
            return self.status()
        if self.state.phase == "delivering":
            raise ValueError("Delivery already started; waiting for clipboard/paste to finish")
        self.state.preview_text = ""
        self.state.cancelled = True
        if self.state.job:
            self.state.job.cancelled = True
            self.state.job.quiet_discard = self.state.job.quiet_discard or quiet
            self.state.job.cancel_event.set()
        if self.state.phase in ("recording", "switching", "capture_failed"):
            self.capture.stop(None)
        elif self.state.phase in ("starting", "stopping", "transcribing", "queued", "loading"):
            self.state.message = "Cancelling; waiting for capture/model to finish"
        return self.status()

    def _idle_command(self, action: str) -> dict[str, object]:
        if self.state.phase != "idle":
            raise ValueError(f"Busy ({self.state.phase}); wait or cancel before {action}")
        if self.pending_jobs():
            raise ValueError(f"Busy (queued); wait or cancel before {action}")
        if action == "warmup":
            self.config = Config.load()
            self.state.cancelled = False
            self.state.phase, self.state.message = "loading", "Loading local model"
            worker = threading.Thread(target=self._warmup, daemon=True)
            self.state.worker = worker
            worker.start()
            return self.status()
        if action == "clear":
            self.learning.clear()
            self.state.preview_text = ""
            self.state.last_text = ""
            self.state.assistant_proposal = None
            self.history.clear()
            self.state.item_id = None
            self.state.message = "History and recordings cleared"
            return self.status()
        raise ValueError(f"Unknown command: {action}")

    def _start_recording(self, data: Mapping[str, object]) -> dict[str, object]:
        with self.lock:
            if self.state.interrupted:
                raise ValueError("Service is shutting down")
            if self.state.phase not in ("idle", "transcribing", "queued", "delivering"):
                raise ValueError(f"Busy ({self.state.phase}); wait or cancel before start")
            pending = self.pending_jobs()
            setup = prepare_recording(
                data, Config.load(), pending, self.capture.microphone, self.delivery
            )
            job = setup.job
            self.recording_assistant.bind_session(job, setup.context)
            if job.config.save_history == "on":
                self.history.begin(job.item_id)

            # Preparation and history can fail; do not supersede the old job until both succeed.
            if self.state.job and self.state.job.edit_consumed:
                self.state.job.cancel_event.set()
            for previous in pending:
                previous.suppress_send = True
            self.state.item_id, self.state.job = job.item_id, job
            self.state.jobs = [*pending, job]
            self.config = setup.capture_config
            self.state.preview_text = ""
            self.live_preview.reset()
            self.learning.cancel()
            self.state.assistant_proposal = None
            self.recording_assistant.clear_stream()
            self.state.cancelled = False
            self.state.phase, self.state.message = "starting", "Opening microphone"
        return self.capture.open(job, setup.capture_config, self.status)

    def _owns_learning(self, recording_id: str) -> bool:
        # RecordingLearning calls this under the controller state lock.
        job = self.state.job
        return (
            not self.state.interrupted
            and job is not None
            and job.item_id == recording_id
            and not job.cancelled
        )

    def _accept_edit(self, data: Mapping[str, object]) -> dict[str, object]:
        return_to_target = reviewed_return_target(data)
        with self.lock:
            job = self.state.job
            edit = None
            if (
                not self.state.interrupted
                and self.state.phase == "idle"
                and job is not None
                and data.get("id") == job.item_id
                and not self.pending_jobs()
            ):
                config = replace(job.config, return_to_start="on" if return_to_target else "off")
                edit = claim_edit(job, self.state.assistant_proposal, config)
            if edit is None:
                raise ValueError("No current captured Edit selection available for replacement")
            self.state.phase, self.state.message = "queued", "Waiting to replace reviewed selection"
        try:
            with self.model_lock:
                with self.lock:
                    if self.state.interrupted or self.state.job is not edit.job:
                        return self.status()
                    if edit.job.cancelled:
                        self.state.mark_job(edit.job, "idle", "Edit cancelled")
                        return self.status()
                self.finalizer.apply_edit(edit)
        finally:
            with self.lock:
                if self.state.job is edit.job and not self.state.interrupted:
                    self.state.phase = "idle"
        return self.status()

    def pending_jobs(self) -> list[DictationJob]:
        return self.state.pending_jobs()

    def interrupt(self) -> None:
        """Stop on logout without interpreting it as the user's discard command."""
        with self.lock:
            self.state.interrupted = True
            self.recording_assistant.clear_stream()
            self.learning.cancel()
            self.state.cancelled = True
            recorder = self.capture.detach()
            if self.state.job and self.state.job.edit_consumed:
                self.state.job.cancel_event.set()
            for job in self.pending_jobs():
                job.cancelled = True
                job.cancel_event.set()
                if job.quiet_discard:
                    self.state.discard(job)
                    continue
                rows = self.history.rows(limit=10000)
                row = next((r for r in rows if r["id"] == job.item_id), {})
                saved_text = row.get("text")
                self.state.update_job(
                    job,
                    "failed",
                    saved_text if isinstance(saved_text, str) else "",
                    "Interrupted; audio discarded",
                )
            self.state.phase, self.state.message = "interrupted", "Interrupted; audio discarded"
        self.capture.discard_after_logout(recorder)

    def _manual_delivery(self, action: str, data: Mapping[str, object]) -> dict[str, object]:
        with self.lock:
            if self.state.interrupted:
                raise ValueError("Service is shutting down")
            if self.state.phase != "idle" or self.pending_jobs():
                raise ValueError(f"Busy ({self.state.phase}); wait or cancel before {action}")
            text = replay_text(self.history, self.state.last_text, data.get("id"))
            config = Config.load()
            owner = self.state.job
            message = "Delivering transcript"
            self.state.phase, self.state.message = "delivering", message

        def active() -> bool:
            with self.lock:
                return (
                    not self.state.interrupted
                    and self.state.job is owner
                    and self.state.phase == "delivering"
                )

        try:
            adapter = cast("_ReplayAdapter", self.delivery)
            paste = (lambda: adapter.paste_key(config)) if action == "paste" else None
            message = replay_transcript(text, adapter.copy, paste, active)
        except Exception as exc:
            message = str(exc)
            raise
        finally:
            with self.lock:
                if active():
                    self.state.phase, self.state.message = "idle", message
        return self.status()

    def _warmup(self) -> None:
        try:
            with self.model_lock:
                cast("_WarmupModel", self.transcriber).load(self.config)
            message = "Model ready"
        # The job or IPC response reports the failure; avoid duplicating private data in logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            message = f"Model unavailable: {exc}. Run mouthkeys download first."
        with self.lock:
            if not self.state.interrupted:
                self.state.phase, self.state.message = (
                    "idle",
                    "Cancelled" if self.state.cancelled else message,
                )

    def _automatic_command_started(self, job: DictationJob) -> None:
        self.state.mark_job(job, "transcribing", "Running configured automatic command")

    def _finish(
        self, job: DictationJob, target: Target, *, auto_paste_expected: bool = False
    ) -> None:
        self.finalizer.finish(
            job,
            target,
            cast("SpeechModel", self.transcriber),
            auto_paste_expected=auto_paste_expected,
        )

    def tick(self) -> None:
        self.capture.tick(self.transcriber)
