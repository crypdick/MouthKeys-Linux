import copy
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, cast

from . import __version__
from .audio import CaptureSession, CaptureTeardownError
from .config import Config, runtime_dir
from .delivery import Delivery, Target
from .desktop import MediaSession, choose_source, play_sound
from .microphone import MicrophoneSelection, Recording, SwitchOutcome, selection
from .pronunciation import PronunciationStore
from .recording_assistant import RecordingAssistant
from .recording_cleanup import cleanup_transcript
from .recording_delivery import DeliveryRequest, RecordingDelivery, delivery_message
from .recording_destination import snapshot_for_stop
from .recording_edit import (
    EditReplacement,
    claim_edit,
    edit_message,
    reviewed_return_target,
)
from .recording_inference import Recognition, recognize_audio, reset_cancelled_stream
from .recording_job import DictationJob, wait_for_predecessors
from .recording_learning import RecordingLearning
from .recording_preview import PreviewWork, RecordingPreview
from .recording_setup import prepare_recording, routed_config
from .recording_transcript import format_dictation
from .storage import History
from .transcribe import Transcriber

FOCUS_TIMEOUT = 4


@dataclass
class FocusGate:
    ready: threading.Event = field(default_factory=threading.Event)
    valid: bool = True
    deadline: float | None = None

    def wait(self) -> None:
        # A stop acknowledgment must not precede its focus snapshot.
        if not self.ready.wait(timeout=FOCUS_TIMEOUT):
            self.valid = False

    def resolved(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.valid = False
        self.ready.set()


class Controller:
    def __init__(self, history=None, transcriber=None, delivery=None, recorder=CaptureSession):
        self.history = history or History()
        self.history.recover()
        self.transcriber = transcriber or Transcriber()
        self.delivery = delivery or Delivery()
        self.recorder_factory = recorder
        self.lock = threading.RLock()
        self.microphone = MicrophoneSelection(self.lock)
        self.phase = "idle"
        self.message = "Ready"
        self.item_id = None
        self.recorder = None
        self.cancelled = False
        self.started = None
        self.worker = None
        self.model_lock = threading.Lock()
        self.live_preview = RecordingPreview(self.lock, self.model_lock)
        self.preview_text = ""
        self.interrupted = False
        self.dictation_delivery = False
        self.stop_done = threading.Event()
        self.stop_done.set()
        self.notice = None
        self.notice_serial = 0
        self.notice_token = uuid.uuid4().hex
        self.media = MediaSession()
        self.recorded_seconds = 0
        self.ai_warning = ""
        self.last_text = ""
        self.job = None
        self.jobs: list[DictationJob] = []
        self.assistant_proposal = None
        self.recording_assistant = RecordingAssistant(
            self.lock,
            lambda job: self.job is job,
            lambda: self.interrupted,
            self._automatic_command_started,
        )
        self.learning = RecordingLearning(
            self.lock, self._owns_learning, lambda: self.interrupted, self.report_error
        )
        self.recording_delivery = RecordingDelivery(
            self.delivery, self.lock, self.learning, lambda: self.interrupted
        )

    def report_error(self, message: str, kind: str = "error") -> None:
        with self.lock:
            self.notice_serial += 1
            self.notice = {
                "serial": f"{self.notice_token}:{self.notice_serial}",
                "message": message,
                "at": time.time(),
                "kind": kind,
            }

    def status(self) -> dict[str, object]:
        with self.lock:
            return {
                "version": __version__,
                "pending_jobs": sum(not job.done.is_set() for job in self.jobs),
                "can_record": self.phase
                in ("idle", "recording", "transcribing", "queued", "delivering"),
                "assistant_proposal": self.assistant_proposal,
                "assistant_stream": self.recording_assistant.stream_status(),
                "mode": self.job.mode if self.job else "dictation",
                "loaded_model": copy.deepcopy(getattr(self.transcriber, "loaded_model", None)),
                "learning_pending": self.learning.pending(),
                "notice": self.notice,
                "preview": self.preview_text,
                "meter": self.live_preview.meter if self.phase == "recording" else {},
                "microphone": self.config.audio_source if hasattr(self, "config") else "default",
                "session_microphone": self.microphone.session_source,
                "phase": self.phase,
                "message": self.message,
                "id": self.item_id,
                "seconds": round(time.monotonic() - self.started, 1) if self.started else 0,
            }

    def command(self, action, **data):
        gate = FocusGate()
        result = self._command(action, _focus_gate=gate, **data)
        if action in ("stop", "toggle") and result.get("phase") == "stopping":
            gate.wait()
        return result

    def _command(self, action, _focus_gate, **data):
        if action == "toggle":
            with self.lock:
                action = (
                    "cancel"
                    if self.phase == "starting"
                    else (
                        "stop"
                        if self.phase in ("recording", "switching", "capture_failed")
                        else "start"
                    )
                )
        if action == "start":
            return self._start_recording(data)
        if action == "select_microphone":
            return self._select_microphone(data)
        if action == "accept_edit":
            return self._accept_edit(data)
        if action in ("copy", "paste"):
            return self._manual_delivery(action, data)
        if action in ("learning_pending", "learning_accept", "learning_dismiss"):
            return self.learning.command(action, data)
        with self.lock:
            return self._locked_command(action, data, _focus_gate)

    def _locked_command(
        self, action: str, data: Mapping[str, object], focus_gate: FocusGate
    ) -> object:
        if action == "status":
            return self.status()
        if action == "release_hold":
            if (
                self.job
                and self.phase in ("starting", "recording", "switching")
                and data.get("id") == self.job.item_id
            ):
                self.job.held_activation = False
            return self.status()
        from .operations import handle_action

        result = handle_action(
            self.history,
            action,
            data,
            allow_mutation=self.phase == "idle" and not self.pending_jobs(),
        )
        if result is not NotImplemented:
            return result
        if self.interrupted:
            raise ValueError("Service is shutting down")
        if action == "cancel":
            return self._cancel_command()
        if action == "stop":
            return self._stop_command(data, focus_gate)
        return self._idle_command(action)

    def _cancel_command(self) -> dict[str, object]:
        self.learning.cancel()
        if self.phase == "delivering" and self.job and self.job.send_pending:
            if self.job.send_dispatched:
                raise ValueError("Send already dispatched; cannot withdraw the key")
            self.cancelled = True
            self.job.cancelled = True
            self.job.cancel_event.set()
            self.message = "Send cancelled; text already inserted"
            return self.status()
        if self.phase == "delivering":
            raise ValueError("Delivery already started; waiting for clipboard/paste to finish")
        self.preview_text = ""
        self.cancelled = True
        if self.job:
            self.job.cancelled = True
            self.job.cancel_event.set()
        if self.phase in ("recording", "switching", "capture_failed"):
            self._stop(None)
        elif self.phase in ("starting", "stopping", "transcribing", "queued", "loading"):
            self.message = "Cancelling; waiting for capture/model to finish"
        return self.status()

    def _stop_command(self, data: Mapping[str, object], focus_gate: FocusGate) -> dict[str, object]:
        if data.get("id") is not None and (not self.job or data["id"] != self.job.item_id):
            return self.status()
        if self.phase == "starting":
            self.job.cancelled = True
            self.job.cancel_event.set()
            self.message = "Cancelling microphone start"
            return self.status()
        if self.phase not in ("recording", "switching", "capture_failed"):
            raise ValueError("Not recording")
        self._request_capture_stop(
            focus_gate, automatic=bool(data.get("automatic")), spoken=bool(data.get("spoken"))
        )
        return self.status()

    def _request_capture_stop(
        self, focus_gate: FocusGate, *, automatic: bool, spoken: bool
    ) -> None:
        target = (
            "capture-focus"
            if self.config.delivery in ("x11", "wayland")
            and self.phase != "capture_failed"
            and (not automatic or spoken)
            else None
        )
        self._stop(target, focus_gate)

    def _idle_command(self, action: str) -> dict[str, object]:
        if self.phase != "idle":
            raise ValueError(f"Busy ({self.phase}); wait or cancel before {action}")
        if self.pending_jobs():
            raise ValueError(f"Busy (queued); wait or cancel before {action}")
        if action == "warmup":
            self.config = Config.load()
            self.cancelled = False
            self.phase, self.message = "loading", "Loading local model"
            self.worker = threading.Thread(target=self._warmup, daemon=True)
            self.worker.start()
            return self.status()
        if action == "clear":
            self.learning.clear()
            self.preview_text = ""
            self.last_text = ""
            self.assistant_proposal = None
            self.history.clear()
            self.item_id = None
            self.message = "History and recordings cleared"
            return self.status()
        raise ValueError(f"Unknown command: {action}")

    def _select_microphone(self, data: Mapping[str, object]) -> dict[str, object]:
        source, persist = selection(data)
        with self.lock:
            if self.interrupted:
                raise ValueError("Service is shutting down")
            if self.phase in ("starting", "switching", "stopping"):
                raise ValueError(f"Busy ({self.phase}); wait before choosing microphone")
            saved = Config.load()
            requested = choose_source(replace(saved, audio_source=source))
            recorder, job = self.recorder, self.job
            active = self.phase == "recording"
            if not active or requested == self.config.audio_source:
                self.microphone.remember(source, saved, persist=persist)
                if not active:
                    self.message = "Microphone selected for next recording"
                return self.status()
            switch_source = getattr(recorder, "switch_source", None)
            if not callable(switch_source):
                raise ValueError("Capture adapter does not support microphone switching")  # noqa: TRY004 - command error
            # The recording phase can be reached only after creating its job.
            job = cast("DictationJob", job)
            fallback = replace(saved, audio_source=self.config.audio_source)
            self.phase, self.message = "switching", "Switching microphone; capture briefly pauses"
            job.phase = "switching"
            self.live_preview.suspend(self.preview_text)

        def switching_current() -> bool:
            with self.lock:
                return (
                    self.recorder is recorder
                    and not job.cancel_event.is_set()
                    and not self.interrupted
                )

        outcome = self.microphone.switch(requested, fallback, switch_source, switching_current)
        with self.lock:
            if outcome is not None and switching_current():
                self._capture_switched(job, outcome, requested)
                if outcome.source == requested:
                    self._remember_microphone(source, saved, persist=persist)
            return self.status()

    def _capture_switched(self, job: DictationJob, outcome: SwitchOutcome, requested: str) -> None:
        current = outcome.source
        if current is None:
            # Recover existing audio without selecting a new delivery destination.
            self.report_error(
                f"Microphone switch failed; recovering recorded audio. {outcome.error}"
            )
            self._stop(None)
            return
        self.config = replace(self.config, audio_source=current)
        job.config = replace(job.config, audio_source=current)
        self.microphone.mark_ready(current)
        if outcome.failed_previous:
            failed, error = outcome.failed_previous
            self.microphone.mark_failed(failed)
            self.report_error(
                f"Previous microphone {failed} failed; switched to {current}. {error}"
            )
        self.live_preview.reset_meter()
        self.phase = "recording"
        job.phase = "recording"
        self.message = f"Listening on {current}; microphone switch has a capture gap"
        if current != requested:
            self.report_error(f"Selected microphone failed; resumed {current}. {outcome.error}")

    def _remember_microphone(self, source: str, saved: Config, *, persist: bool) -> None:
        try:
            self.microphone.remember(source, Config.load() if persist else saved, persist=persist)
        except (OSError, ValueError, TypeError) as exc:
            self.microphone.remember(source, saved, persist=False)
            self.report_error(f"Microphone switched for this session; preference not saved. {exc}")

    def _start_recording(self, data: Mapping[str, object]) -> dict[str, object]:
        with self.lock:
            if self.interrupted:
                raise ValueError("Service is shutting down")
            if self.phase not in ("idle", "transcribing", "queued", "delivering"):
                raise ValueError(f"Busy ({self.phase}); wait or cancel before start")
            pending = self.pending_jobs()
            setup = prepare_recording(data, Config.load(), pending, self.microphone, self.delivery)
            job = setup.job
            self.recording_assistant.bind_session(job, setup.context)
            if job.config.save_history == "on":
                self.history.begin(job.item_id)

            # Preparation and history can fail; do not supersede the old job until both succeed.
            if self.job and self.job.edit_consumed:
                self.job.cancel_event.set()
            for previous in pending:
                previous.suppress_send = True
            self.item_id, self.job = job.item_id, job
            self.jobs = [*pending, job]
            self.config = setup.capture_config
            self.preview_text = ""
            self.live_preview.reset()
            self.learning.cancel()
            self.assistant_proposal = None
            self.recording_assistant.clear_stream()
            self.cancelled = False
            self.phase, self.message = "starting", "Opening microphone"
        return self._open_recording(job, setup.capture_config)

    def _open_recording(self, job: DictationJob, config: Config) -> dict[str, object]:
        def start(chosen: Config, timeout: float) -> Recording:
            factory = self.recorder_factory
            capture: Recording
            if factory is CaptureSession:
                capture = factory(
                    chosen, job.audio_path, cancel_event=job.cancel_event, timeout=timeout
                )
            else:
                capture = factory(chosen, job.audio_path)
            return capture

        failure: Exception | None = None
        try:
            self.media.pause(config)
            capture, chosen = self.microphone.open(
                job.config, start, lambda: job.cancel_event.is_set() or self.interrupted
            )
            with self.lock:
                if not job.cancel_event.is_set() and not self.interrupted:
                    self.recorder = capture
                    self.config = chosen
                    job.config = replace(job.config, audio_source=chosen.audio_source)
                    self.notice = None
                    self.phase, self.message = "recording", "Listening — toggle again to finish"
                    job.phase = "recording"
                    if chosen.audio_source != config.audio_source:
                        self.message += f" (fallback microphone: {chosen.audio_source})"
                    self.started = time.monotonic()
                    play_sound(chosen)
                    return self.status()
            capture.stop()
        # The job or IPC response reports the failure; avoid duplicating private data in logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            failure = exc

        with self.lock:
            cancelled = job.cancel_event.is_set()
            if not self.interrupted:
                error = "" if cancelled else str(failure)
                self._update(job, "cancelled" if cancelled else "failed", error=error)
                self.phase = "idle"
                self.message = "Microphone start cancelled" if cancelled else error
            job.audio_path.unlink(missing_ok=True)
            job.done.set()
        self.media.resume()
        if failure is not None and not cancelled and not self.interrupted:
            raise failure
        return self.status()

    def _owns_learning(self, recording_id: str) -> bool:
        # RecordingLearning calls this under the controller state lock.
        job = self.job
        return (
            not self.interrupted
            and job is not None
            and job.item_id == recording_id
            and not job.cancelled
        )

    def _accept_edit(self, data: Mapping[str, object]) -> dict[str, object]:
        return_to_target = reviewed_return_target(data)
        with self.lock:
            job = self.job
            edit = None
            if (
                not self.interrupted
                and self.phase == "idle"
                and job is not None
                and data.get("id") == job.item_id
                and not self.pending_jobs()
            ):
                config = replace(job.config, return_to_start="on" if return_to_target else "off")
                edit = claim_edit(job, self.assistant_proposal, config)
            if edit is None:
                raise ValueError("No current captured Edit selection available for replacement")
            self.phase, self.message = "queued", "Waiting to replace reviewed selection"
        try:
            with self.model_lock:
                with self.lock:
                    if self.interrupted or self.job is not edit.job:
                        return self.status()
                    if edit.job.cancelled:
                        self._job_status(edit.job, "idle", "Edit cancelled")
                        return self.status()
                self._apply_edit(edit)
        finally:
            with self.lock:
                if self.job is edit.job and not self.interrupted:
                    self.phase = "idle"
        return self.status()

    def _apply_edit(self, edit: EditReplacement) -> None:
        result = edit.apply(self.delivery)
        with self.lock:
            if not self.interrupted:
                if self.job is edit.job or result != "cancelled":
                    self._update(edit.job, result, edit.text)
                self._job_status(edit.job, "idle", edit_message(result))

    def pending_jobs(self) -> list[DictationJob]:
        return [job for job in self.jobs if not job.done.is_set()]

    def _update(
        self, job: DictationJob, status: str, text: str = "", error: str = "", **metadata: object
    ) -> None:
        # Explicit privacy opt-out overrides durable transcript-before-delivery.
        if job.config.save_history == "on":
            if status == "cancelled":
                metadata["raw_text"] = ""
                metadata["was_ai_processed"] = False
                metadata["ai_processing_error"] = ""
                metadata["ai_processing_model"] = ""
            self.history.update(job.item_id, status, text, error, **{**job.metadata, **metadata})

    def _job_status(self, job: DictationJob, phase: str, message: str) -> None:
        with self.lock:
            job.phase = phase
            if self.job is job and not self.interrupted:
                self.phase, self.message = phase, message

    def interrupt(self):
        """Stop on logout without interpreting it as the user's discard command."""
        with self.lock:
            self.interrupted = True
            self.recording_assistant.clear_stream()
            self.learning.cancel()
            self.cancelled = True
            recorder, self.recorder = self.recorder, None
            if self.job and self.job.edit_consumed:
                self.job.cancel_event.set()
            for job in self.pending_jobs():
                job.cancelled = True
                job.cancel_event.set()
                rows = self.history.rows(limit=10000)
                row = next((r for r in rows if r["id"] == job.item_id), {})
                saved_text = row.get("text")
                self._update(
                    job,
                    "failed",
                    saved_text if isinstance(saved_text, str) else "",
                    "Interrupted; audio discarded",
                )
            self.phase, self.message = "interrupted", "Interrupted; audio discarded"
        if recorder is not None:
            try:
                recorder.stop()
            # Logout must continue with interrupted audio cleanup.
            except Exception:  # allow: exception-handling  # noqa: BLE001, S110
                pass  # Cleanup below also discards partial audio.
        # An already-stopping worker owns its recorder. Let it finalize the WAV, without
        # waiting for model inference or making a second concurrent stop call.
        self.stop_done.wait(timeout=11)
        self.media.resume()
        for job in self.jobs:
            job.audio_path.unlink(missing_ok=True)
        # Failed capture teardown may still own an open segment; unlink its name too.
        for segment in (runtime_dir() / "private-recordings").glob("*.wav"):
            segment.unlink(missing_ok=True)

    def _stop(self, target: object, focus_gate: FocusGate | None = None) -> None:
        recorder, self.recorder = self.recorder, None
        self.recorded_seconds = (
            time.monotonic() - self.started if self.started else self.job.duration
        )
        self.job.duration = self.recorded_seconds
        self.job.phase = "stopping"
        self.stop_done.clear()
        self.notice = None
        self.phase, self.message, self.started = "stopping", "Stopping microphone", None
        self.worker = threading.Thread(
            target=self._stopped, args=(self.job, recorder, target, focus_gate), daemon=True
        )
        self.worker.start()

    def _stopped(
        self, job: DictationJob, recorder: Recording, target: object, focus_gate: FocusGate | None
    ) -> None:
        try:
            destination = snapshot_for_stop(
                self.delivery, job.config, target, start_target=job.start_target, mode=job.mode
            )
            job.context = destination.context
            if focus_gate:
                focus_gate.resolved()
            try:
                recorder.stop()
            finally:
                self.stop_done.set()
                self.media.resume()
                play_sound(job.config, stop=True)
            metadata = destination.metadata(self.delivery)
            if metadata is not None:
                job.metadata = metadata
        # The job or IPC response reports the failure; avoid duplicating private data in logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            with self.lock:
                if not self.interrupted:
                    if isinstance(exc, CaptureTeardownError):
                        self.recorder = recorder
                        self._update(job, "recording", error=str(exc))
                        self._job_status(
                            job,
                            "capture_failed",
                            f"{exc}. Stop or Cancel retries microphone shutdown; "
                            "new recording blocked",
                        )
                        return  # Capture still owns audio and pending-job completion.
                    self._update(job, "failed", error=str(exc))
                    self._job_status(job, "idle", str(exc))
                    job.audio_path.unlink(missing_ok=True)
                job.done.set()
            return
        with self.lock:
            if self.interrupted:
                job.done.set()
                return
            if job.cancelled:
                self._job_status(job, "queued", "Cancelling; waiting for model cleanup")
            else:
                self._job_status(job, "queued", "Waiting for earlier dictation")
                self._update(job, "transcribing")
        target = None if focus_gate and not focus_gate.valid else destination.target
        self._finish(job, cast("Target", target), auto_paste_expected=destination.automatic)

    def _manual_delivery(self, action, data):
        with self.lock:
            if self.interrupted:
                raise ValueError("Service is shutting down")
            if self.phase != "idle" or self.pending_jobs():
                raise ValueError(f"Busy ({self.phase}); wait or cancel before {action}")
            text = self.last_text or self.history.latest_text()
            if data.get("id"):
                text = self.history.get(data["id"])["text"]
                if not text:
                    raise ValueError("No transcript for this entry")
            config = Config.load()
            self.phase, self.message = "delivering", "Delivering transcript"
        try:
            self.delivery.copy(text)
            if action == "paste" and not self.interrupted:
                time.sleep(0.2)
                if not self.interrupted:
                    self.delivery.paste_key(config)
            with self.lock:
                if not self.interrupted:
                    self.message = "Copied" if action == "copy" else "Paste requested"
        except Exception as exc:
            with self.lock:
                if not self.interrupted:
                    self.message = str(exc)
            raise
        finally:
            with self.lock:
                if not self.interrupted:
                    self.phase = "idle"
        return self.status()

    def _warmup(self):
        try:
            with self.model_lock:
                self.transcriber.load(self.config)
            message = "Model ready"
        # The job or IPC response reports the failure; avoid duplicating private data in logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            message = f"Model unavailable: {exc}. Run mouthkeys download first."
        with self.lock:
            if not self.interrupted:
                self.phase, self.message = "idle", "Cancelled" if self.cancelled else message

    def _automatic_command_started(self, job: DictationJob) -> None:
        self._job_status(job, "transcribing", "Running configured automatic command")

    def _discard_recording(self, job: DictationJob, message: str = "Cancelled") -> None:
        self._update(job, "cancelled")
        self._job_status(job, "idle", message)

    def _may_finalize(self, job: DictationJob) -> bool:
        """Call under the state lock so an interrupted row cannot be overwritten."""
        if self.interrupted:
            return False
        if job.cancelled:
            self._discard_recording(job)
            return False
        return True

    def _commit_recognition(self, job: DictationJob, result: Recognition) -> bool:
        with self.lock:
            if not self._may_finalize(job):
                return False
            if result.skipped_silence:
                self._discard_recording(job, "Silent recording skipped")
                return False
            config = job.config
            if not result.text:
                if self.job is job:
                    self.preview_text = ""
                self._update(job, "failed", error="No speech detected")
                retained = (
                    "audio discarded"
                    if config.save_history == "on"
                    else "history disabled; audio discarded"
                )
                self._job_status(job, "idle", f"No speech detected; {retained}")
                return False
            # Raw text survives a crash or timeout during optional cleanup.
            self._update(
                job,
                "transcribing",
                result.text,
                raw_text=result.text,
                duration=job.duration,
                model=config.model,
                language=config.language,
            )
            return True

    def _publish_proposal(
        self, job: DictationJob, config: Config, proposal: dict[str, Any], ai_warning: str
    ) -> None:
        with self.lock:
            if not self._may_finalize(job):
                return
            text = proposal["text"]
            assistant_requested = job.mode in ("command", "edit")
            self._update(
                job,
                "transcribed",
                text,
                was_ai_processed=assistant_requested,
                ai_processing_error=ai_warning,
                ai_processing_model=config.ai_model if assistant_requested else "",
            )
            edit = None
            if self.job is job:
                if config.edit_delivery == "replace":
                    edit = claim_edit(job, proposal, config)
                self.assistant_proposal = proposal
                self.recording_assistant.clear_stream()
                self.preview_text = text
            if edit is not None:
                self._job_status(job, "delivering", "Replacing captured selection")
            else:
                message = (
                    "Command finished; execution receipts retained"
                    if proposal.get("execution_count")
                    else "Recording ready for review; nothing inserted"
                )
                self._job_status(job, "idle", message)
        if edit is not None:
            self._apply_edit(edit)

    def _finish(
        self, job: DictationJob, target: Target, *, auto_paste_expected: bool = False
    ) -> None:
        text = ""
        assistant_error: str | None = None
        config = job.config
        path = job.audio_path
        try:
            with self.lock:
                if not self._may_finalize(job):
                    return
            if not wait_for_predecessors(job, lambda: self.interrupted):
                return
            with self.lock:
                if not self._may_finalize(job):
                    return
                self._job_status(job, "transcribing", "Transcribing")
            recognition = recognize_audio(
                job, self.transcriber, self.model_lock, lambda: job.cancelled or self.interrupted
            )
            text = recognition.text
            if not self._commit_recognition(job, recognition):
                return
            config, ai_warning = routed_config(job)
            if ai_warning:
                with self.lock:
                    if self.job is job and not self.interrupted:
                        self.report_error(ai_warning, kind="ai")
            if job.mode != "dictation":
                if job.mode == "training":
                    proposal = {"mode": "training", "text": text}
                elif job.mode == "pronunciation":
                    with self.lock:
                        if self.interrupted or job.cancelled:
                            return
                        # Serialize this small feature-store update with cancellation.
                        enrollment = PronunciationStore.enroll(job.label, path)
                    proposal = {"mode": "pronunciation", "text": job.label, **enrollment}
                else:
                    try:
                        proposal = self.recording_assistant.propose(job, config, text)
                    except Exception as exc:
                        assistant_error = str(exc)
                        raise
                text = proposal["text"]
                self._publish_proposal(job, config, proposal, ai_warning)
                return
            formatted = format_dictation(
                job, config, text, recognition.alignment, has_target=target is not None
            )
            text = formatted.text
            cleanup = cleanup_transcript(
                text, config, job.cancel_event, lambda partial: self._publish_cleanup(job, partial)
            )
            text = cleanup.text
            ai_warning = cleanup.warning or ai_warning
            with self.lock:
                if not self._may_finalize(job):
                    return
                if self.job is job:
                    if cleanup.warning:
                        self.report_error(ai_warning, kind="ai")
                    self.preview_text = text
                    self.dictation_delivery = True
                if text:
                    self.last_text = text
                self._update(
                    job,
                    "transcribed",
                    text,
                    was_ai_processed=cleanup.applied,
                    ai_processing_error=ai_warning,
                    ai_processing_model=config.ai_model if cleanup.applied else "",
                )
                self._job_status(job, "delivering", "Delivering transcript")
            request = DeliveryRequest(
                text, config, target, auto_paste_expected, formatted.should_send
            )
            outcome = self.recording_delivery.deliver(job, request)
            with self.lock:
                if self.interrupted:
                    return
                self._update(job, outcome.status, text, error=outcome.send_warning)
                message = delivery_message(request, outcome)
                if outcome.paste_warning and self.job is job and not ai_warning:
                    self.report_error(message, kind="paste")
                notices = (
                    message,
                    ai_warning,
                    formatted.pronunciation_warning,
                    outcome.send_warning,
                )
                self._job_status(job, "idle", ". ".join(filter(None, notices)))
        # The job or IPC response reports the failure; avoid duplicating private data in logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            with self.lock:
                if not self._may_finalize(job):
                    return
                if self.job is job:
                    self.preview_text = text
                    if assistant_error is not None:
                        self.report_error(f"AI {job.mode} request failed: {exc}", kind="ai")
                self._update(
                    job,
                    "failed",
                    text,
                    str(exc),
                    **({"ai_processing_error": assistant_error} if assistant_error else {}),
                )
                retained = (
                    "Saved in history; audio discarded."
                    if config.save_history == "on"
                    else "History disabled; audio discarded."
                )
                self._job_status(job, "idle", f"{exc}. {retained}")
        finally:
            try:
                reset_cancelled_stream(job, self.transcriber, self.model_lock)
            finally:
                with self.lock:
                    job.audio_path.unlink(missing_ok=True)
                    if not self.interrupted and self.job is job:
                        self.recording_assistant.clear_stream()
                        self.dictation_delivery = False
                        self.phase = "idle"
                    job.done.set()

    def _publish_cleanup(self, job: DictationJob, text: str) -> None:
        with self.lock:
            if self.job is job and not job.cancelled and not self.interrupted:
                self.preview_text = text

    def _publish_preview(self, text: str) -> None:
        self.preview_text = text

    def tick(self) -> None:
        with self.lock:
            if self.phase != "recording":
                return
            job = cast("DictationJob", self.job)  # Recording always has a job.
            self.live_preview.sample(job.audio_path, self.preview_text)
            if self.live_preview.due(self.config):
                self.live_preview.schedule(
                    PreviewWork(
                        job.audio_path,
                        self.config,
                        self.transcriber,
                        job.predecessors,
                        is_recording=lambda: self.phase == "recording" and self.job is job,
                        cancelled=lambda: job.cancelled or self.interrupted,
                        publish=self._publish_preview,
                    )
                )
            spoken = self.live_preview.should_stop(self.config, held=job.held_activation)
            if not (
                spoken
                or self.recorder.finished()
                or time.monotonic() - self.started >= self.config.max_seconds
            ):
                return
            # The daemon must keep accepting commands while focus is resolved.
            gate = FocusGate(deadline=time.monotonic() + FOCUS_TIMEOUT)
            try:
                self._request_capture_stop(gate, automatic=True, spoken=spoken)
            # The capture stop worker reports actionable errors.
            except Exception:  # allow: exception-handling
                if spoken:
                    raise
                return
