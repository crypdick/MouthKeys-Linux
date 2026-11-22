import copy
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import replace
from typing import Any, cast

from . import __version__
from .audio import CaptureSession, CaptureTeardownError, Recording
from .config import Config, runtime_dir
from .delivery import Delivery, Target
from .desktop import MediaSession, choose_source, play_sound
from .microphone import MicrophoneSelection, SwitchOutcome, selection
from .operations import handle_action
from .recording_assistant import RecordingAssistant
from .recording_delivery import (
    RecordingDelivery,
    replay_text,
    replay_transcript,
)
from .recording_destination import FocusGate, snapshot_for_stop
from .recording_edit import (
    claim_edit,
    reviewed_return_target,
)
from .recording_finalizer import RecordingFinalizer
from .recording_job import DictationJob
from .recording_learning import LearningCommand, RecordingLearning
from .recording_preview import PreviewWork, RecordingPreview
from .recording_setup import prepare_recording
from .recording_state import RecordingState
from .storage import History
from .transcribe import Transcriber


class Controller:
    def __init__(self, history=None, transcriber=None, delivery=None, recorder=CaptureSession):
        self.history = history or History()
        self.history.recover()
        self.transcriber = transcriber or Transcriber()
        self.delivery = delivery or Delivery()
        self.recorder_factory = recorder
        self.state = RecordingState(self.history)
        self.lock = self.state.lock
        self.microphone = MicrophoneSelection(self.lock)
        self.recorder: Recording | None = None
        self.started: float | None = None
        self.worker = None
        self.model_lock = threading.Lock()
        self.live_preview = RecordingPreview(self.lock, self.model_lock)
        self.stop_done = threading.Event()
        self.stop_done.set()
        self.media = MediaSession()
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
                "microphone": self.config.audio_source if hasattr(self, "config") else "default",
                "session_microphone": self.microphone.session_source,
                "phase": self.state.phase,
                "message": self.state.message,
                "id": self.state.item_id,
                "seconds": round(time.monotonic() - self.started, 1) if self.started else 0,
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
        if action == "toggle":
            with self.lock:
                action = (
                    "cancel"
                    if self.state.phase == "starting"
                    else (
                        "stop"
                        if self.state.phase in ("recording", "switching", "capture_failed")
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
            return self.learning.command(cast("LearningCommand", action), data)
        with self.lock:
            return self._locked_command(action, data, _focus_gate)

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
        if action in ("cancel", "stop"):
            if data.get("id") is not None and (
                not self.state.job or data["id"] != self.state.job.item_id
            ):
                return self.status()
            if action == "cancel":
                return self._cancel_command()
            return self._stop_command(data, focus_gate)
        return self._idle_command(action)

    def _cancel_command(self) -> dict[str, object]:
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
            self.state.job.cancel_event.set()
        if self.state.phase in ("recording", "switching", "capture_failed"):
            self._stop(None)
        elif self.state.phase in ("starting", "stopping", "transcribing", "queued", "loading"):
            self.state.message = "Cancelling; waiting for capture/model to finish"
        return self.status()

    def _stop_command(self, data: Mapping[str, object], focus_gate: FocusGate) -> dict[str, object]:
        if self.state.phase == "starting":
            job = cast("DictationJob", self.state.job)  # Starting always has a job.
            job.cancelled = True
            job.cancel_event.set()
            self.state.message = "Cancelling microphone start"
            return self.status()
        if self.state.phase not in ("recording", "switching", "capture_failed"):
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
            and self.state.phase != "capture_failed"
            and (not automatic or spoken)
            else None
        )
        self._stop(target, focus_gate)

    def _idle_command(self, action: str) -> dict[str, object]:
        if self.state.phase != "idle":
            raise ValueError(f"Busy ({self.state.phase}); wait or cancel before {action}")
        if self.pending_jobs():
            raise ValueError(f"Busy (queued); wait or cancel before {action}")
        if action == "warmup":
            self.config = Config.load()
            self.state.cancelled = False
            self.state.phase, self.state.message = "loading", "Loading local model"
            self.worker = threading.Thread(target=self._warmup, daemon=True)
            self.worker.start()
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

    def _select_microphone(self, data: Mapping[str, object]) -> dict[str, object]:
        source, persist = selection(data)
        with self.lock:
            if self.state.interrupted:
                raise ValueError("Service is shutting down")
            if self.state.phase in ("starting", "switching", "stopping"):
                raise ValueError(f"Busy ({self.state.phase}); wait before choosing microphone")
            saved = Config.load()
            requested = choose_source(replace(saved, audio_source=source))
            recorder, job = self.recorder, self.state.job
            active = self.state.phase == "recording"
            if not active or requested == self.config.audio_source:
                self.microphone.remember(source, saved, persist=persist)
                if not active:
                    self.state.message = "Microphone selected for next recording"
                return self.status()
            switch_source = getattr(recorder, "switch_source", None)
            if not callable(switch_source):
                raise ValueError("Capture adapter does not support microphone switching")  # noqa: TRY004 - command error
            # The recording phase can be reached only after creating its job.
            job = cast("DictationJob", job)
            fallback = replace(saved, audio_source=self.config.audio_source)
            self.state.phase, self.state.message = (
                "switching",
                "Switching microphone; capture briefly pauses",
            )
            job.phase = "switching"
            self.live_preview.suspend(self.state.preview_text)

        def switching_current() -> bool:
            with self.lock:
                return (
                    self.recorder is recorder
                    and not job.cancel_event.is_set()
                    and not self.state.interrupted
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
        self.state.phase = "recording"
        job.phase = "recording"
        self.state.message = f"Listening on {current}; microphone switch has a capture gap"
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
            if self.state.interrupted:
                raise ValueError("Service is shutting down")
            if self.state.phase not in ("idle", "transcribing", "queued", "delivering"):
                raise ValueError(f"Busy ({self.state.phase}); wait or cancel before start")
            pending = self.pending_jobs()
            setup = prepare_recording(data, Config.load(), pending, self.microphone, self.delivery)
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
                job.config, start, lambda: job.cancel_event.is_set() or self.state.interrupted
            )
            with self.lock:
                if not job.cancel_event.is_set() and not self.state.interrupted:
                    self.recorder = capture
                    self.config = chosen
                    job.config = replace(job.config, audio_source=chosen.audio_source)
                    self.state.notice = None
                    self.state.phase, self.state.message = (
                        "recording",
                        "Listening — toggle again to finish",
                    )
                    job.phase = "recording"
                    if chosen.audio_source != config.audio_source:
                        self.state.message += f" (fallback microphone: {chosen.audio_source})"
                    self.started = time.monotonic()
                    play_sound(chosen)
                    return self.status()
            capture.stop()
        # The job or IPC response reports the failure; avoid duplicating private data in logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            failure = exc

        with self.lock:
            cancelled = job.cancel_event.is_set()
            if not self.state.interrupted:
                error = "" if cancelled else str(failure)
                self.state.update_job(job, "cancelled" if cancelled else "failed", error=error)
                self.state.phase = "idle"
                self.state.message = "Microphone start cancelled" if cancelled else error
            job.audio_path.unlink(missing_ok=True)
            job.done.set()
        self.media.resume()
        if failure is not None and not cancelled and not self.state.interrupted:
            raise failure
        return self.status()

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
            recorder, self.recorder = self.recorder, None
            if self.state.job and self.state.job.edit_consumed:
                self.state.job.cancel_event.set()
            for job in self.pending_jobs():
                job.cancelled = True
                job.cancel_event.set()
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
        if recorder is not None:
            # Logout must continue with interrupted audio cleanup.
            with suppress(Exception):
                recorder.stop()
        # An already-stopping worker owns its recorder. Let it finalize the WAV, without
        # waiting for model inference or making a second concurrent stop call.
        self.stop_done.wait(timeout=11)
        self.media.resume()
        for job in self.state.jobs:
            job.audio_path.unlink(missing_ok=True)
        # Failed capture teardown may still own an open segment; unlink its name too.
        for segment in (runtime_dir() / "private-recordings").glob("*.wav"):
            segment.unlink(missing_ok=True)

    def _stop(self, target: object, focus_gate: FocusGate | None = None) -> None:
        # These phases retain both the job and its recorder until stopping starts.
        job = cast("DictationJob", self.state.job)
        recorder, self.recorder = cast("Recording", self.recorder), None
        if self.started:
            job.duration = time.monotonic() - self.started
        job.phase = "stopping"
        self.stop_done.clear()
        self.state.notice = None
        self.state.phase, self.state.message, self.started = "stopping", "Stopping microphone", None
        self.worker = threading.Thread(
            target=self._stopped, args=(job, recorder, target, focus_gate), daemon=True
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
                if not self.state.interrupted:
                    if isinstance(exc, CaptureTeardownError):
                        self.recorder = recorder
                        self.state.update_job(job, "recording", error=str(exc))
                        self.state.mark_job(
                            job,
                            "capture_failed",
                            f"{exc}. Stop or Cancel retries microphone shutdown; "
                            "new recording blocked",
                        )
                        return  # Capture still owns audio and pending-job completion.
                    self.state.update_job(job, "failed", error=str(exc))
                    self.state.mark_job(job, "idle", str(exc))
                    job.audio_path.unlink(missing_ok=True)
                job.done.set()
            return
        with self.lock:
            if self.state.interrupted:
                job.done.set()
                return
            if job.cancelled:
                self.state.mark_job(job, "queued", "Cancelling; waiting for model cleanup")
            else:
                self.state.mark_job(job, "queued", "Waiting for earlier dictation")
                self.state.update_job(job, "transcribing")
        target = None if focus_gate and not focus_gate.valid else destination.target
        self._finish(job, cast("Target", target), auto_paste_expected=destination.automatic)

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
            paste = (lambda: self.delivery.paste_key(config)) if action == "paste" else None
            message = replay_transcript(text, self.delivery.copy, paste, active)
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
                self.transcriber.load(self.config)
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
            job, target, self.transcriber, auto_paste_expected=auto_paste_expected
        )

    def _publish_preview(self, text: str) -> None:
        self.state.preview_text = text

    def tick(self) -> None:
        with self.lock:
            if self.state.phase != "recording":
                return
            job = cast("DictationJob", self.state.job)  # Recording always has a job.
            self.live_preview.sample(job.audio_path, self.state.preview_text)
            if self.live_preview.due(self.config):
                self.live_preview.schedule(
                    PreviewWork(
                        job.audio_path,
                        self.config,
                        self.transcriber,
                        job.predecessors,
                        is_recording=lambda: (
                            self.state.phase == "recording" and self.state.job is job
                        ),
                        cancelled=lambda: job.cancelled or self.state.interrupted,
                        publish=self._publish_preview,
                    )
                )
            spoken = self.live_preview.should_stop(self.config, held=job.held_activation)
            recorder = cast("Recording", self.recorder)  # Recording owns its capture.
            started = cast("float", self.started)
            if not (
                spoken
                or recorder.finished()
                or time.monotonic() - started >= self.config.max_seconds
            ):
                return
            # The daemon must keep accepting commands while focus is resolved.
            gate = FocusGate.automatic()
            try:
                self._request_capture_stop(gate, automatic=True, spoken=spoken)
            # The capture stop worker reports actionable errors.
            except Exception:  # allow: exception-handling
                if spoken:
                    raise
                return
