import copy
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, cast

from . import __version__
from .audio import CaptureSession, CaptureTeardownError, check_audio, is_short_silence
from .config import Config, private_dir, runtime_dir
from .delivery import Delivery, Target
from .desktop import MediaSession, choose_source, play_sound
from .formatting import enabled
from .microphone import MicrophoneSelection, Recording, SwitchOutcome, selection
from .profiles import PromptProfile
from .pronunciation import PronunciationStore
from .recording_assistant import InlineContext, RecordingAssistant, inline_context
from .recording_cleanup import cleanup_transcript
from .recording_delivery import DeliveryRequest, RecordingDelivery, delivery_message
from .recording_destination import snapshot_for_stop
from .recording_edit import (
    EditReplacement,
    captured_selection,
    claim_edit,
    edit_message,
    reviewed_return_target,
)
from .recording_job import DictationJob, recording_input
from .recording_learning import RecordingLearning
from .recording_preview import PreviewWork, RecordingPreview, reset_owned_stream
from .recording_transcript import format_dictation
from .routing import PromptRoute
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
        self.jobs = []
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
            allow_mutation=self.phase == "idle" and not self._pending_jobs(),
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
        if self._pending_jobs():
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
            from .profiles import load_profiles
            from .routing import load_routes

            mode, selected_text, label = recording_input(data)
            config = Config.load()
            context = inline_context(data, mode, config)
            if len(self._pending_jobs()) >= config.queue_limit:
                raise ValueError("Dictation queue is full; wait for earlier recordings")
            config = self.microphone.startup_config(config)
            profiles = load_profiles()
            routes = load_routes(profiles)
            self.preview_text = ""
            self.live_preview.reset()
            self.config = config
            self.learning.cancel()
            if self.job and self.job.edit_consumed:
                # A new recording supersedes an Edit waiting on desktop work.
                self.job.cancel_event.set()
            for previous in self._pending_jobs():
                previous.suppress_send = True
            self.item_id = str(uuid.uuid4())
            job: DictationJob = self._new_job(
                mode,
                selected_text,
                profiles=profiles,
                routes=routes,
                context=context,
            )
            job.held_activation = data.get("held") is True
            job.label = label
            self._capture_start_target(job)
            self._select_recording_prompt(job, data.get("profile"))
            self.assistant_proposal = None
            self.recording_assistant.clear_stream()
            self.cancelled = False
            if self.config.save_history == "on":
                self.history.begin(self.item_id)
            self.phase, self.message = "starting", "Opening microphone"
            job.phase = "starting"
        return self._open_recording(job, config)

    def _capture_start_target(self, job: DictationJob) -> None:
        if not (
            self.config.target_policy == "start"
            or self.config.return_to_start == "on"
            or job.mode == "edit"
        ) or not hasattr(self.delivery, "capture_target"):
            return
        try:
            job.start_target = self.delivery.capture_target(self.config)
        # Unknown focus disables automatic delivery.
        except Exception:  # allow: exception-handling  # noqa: BLE001
            job.start_target = None
        if job.mode == "edit" and not job.selected_text:
            job.selected_text = captured_selection(job.start_target)

    def _select_recording_prompt(self, job: DictationJob, override: object) -> None:
        if override is None:
            return
        from .routing import select_prompt

        try:
            job.config = select_prompt(
                self.config,
                override=override,
                mode=job.mode if job.mode in ("dictation", "edit", "command") else "dictation",
                profiles=job.profiles,
            )
        except Exception:
            job.done.set()
            raise
        job.prompt_override = cast("str", override)  # Validated by select_prompt.

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
        except Exception as exc:  # allow: exception-handling
            with self.lock:
                cancelled = job.cancel_event.is_set() and not self.interrupted
                if not self.interrupted:
                    self._update(
                        job,
                        "cancelled" if cancelled else "failed",
                        error="" if cancelled else str(exc),
                    )
                    self.message = "Microphone start cancelled" if cancelled else str(exc)
                job.audio_path.unlink(missing_ok=True)
                job.done.set()
                if not self.interrupted:
                    self.phase = "idle"
            self.media.resume()
            if not cancelled and not self.interrupted:
                raise
        else:
            with self.lock:
                if not self.interrupted:
                    self._update(job, "cancelled")
                job.audio_path.unlink(missing_ok=True)
                job.done.set()
                if not self.interrupted:
                    self.phase, self.message = "idle", "Microphone start cancelled"
        self.media.resume()
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
                and not self._pending_jobs()
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

    def _pending_jobs(self):
        return [job for job in self.jobs if not job.done.is_set()]

    def _new_job(
        self,
        mode: str,
        selected_text: str,
        profiles: list[PromptProfile],
        routes: list[PromptRoute],
        context: InlineContext | None,
    ) -> DictationJob:
        pending = self._pending_jobs()
        audio_path = private_dir(runtime_dir() / "private-recordings") / f"{self.item_id}.wav"
        job = DictationJob(
            cast("str", self.item_id),
            self.config,
            tuple(job.done for job in pending),
            audio_path,
            mode=mode,
            selected_text=selected_text,
            profiles=profiles,
            routes=routes,
        )
        self.recording_assistant.bind_session(job, context)
        self.job = job
        self.jobs = [*pending, job]
        return job

    def _update(self, job, status, text="", error="", **metadata):
        # Explicit privacy opt-out overrides durable transcript-before-delivery.
        if job.config.save_history == "on":
            if status == "cancelled":
                metadata["raw_text"] = ""
                metadata["was_ai_processed"] = False
                metadata["ai_processing_error"] = ""
                metadata["ai_processing_model"] = ""
            self.history.update(job.item_id, status, text, error, **{**job.metadata, **metadata})

    def _job_status(self, job, phase, message):
        with self.lock:
            job.phase = phase
            if self.job is job and not self.interrupted:
                self.phase, self.message = phase, message

    def _wait_for_predecessors(self, job):
        for previous in job.predecessors:
            while not previous.wait(0.1):
                if self.interrupted:
                    return False
        return not self.interrupted

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
            for job in self._pending_jobs():
                job.cancelled = True
                job.cancel_event.set()
                rows = self.history.rows(limit=10000)
                row = next((r for r in rows if r["id"] == job.item_id), {})
                self._update(job, "failed", row.get("text", ""), "Interrupted; audio discarded")
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
            if self.phase != "idle" or self._pending_jobs():
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

    def _publish_proposal(
        self, job: DictationJob, config: Config, proposal: dict[str, Any], ai_warning: str
    ) -> None:
        with self.lock:
            if self.interrupted:
                return
            if job.cancelled:
                self._update(job, "cancelled")
                job.audio_path.unlink(missing_ok=True)
                self._job_status(job, "idle", "Cancelled")
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
        alignment = None
        ai_warning = ""
        assistant_error: str | None = None
        config = job.config
        path = job.audio_path
        try:
            if job.cancelled:
                self._update(job, "cancelled")
                path.unlink(missing_ok=True)
                self._job_status(job, "idle", "Cancelled")
                return
            if not self._wait_for_predecessors(job):
                return
            with self.lock:
                if not job.cancelled:
                    self._job_status(job, "transcribing", "Transcribing")
            check_audio(path)
            if enabled(config, "skip_silent_recordings") and is_short_silence(path):
                self._update(job, "cancelled")
                path.unlink(missing_ok=True)
                self._job_status(job, "idle", "Silent recording skipped")
                return
            with self.model_lock:
                if not job.cancelled and not self.interrupted:
                    text = self.transcriber.transcribe(path, config)
                    if config.pronunciation_matching == "on":
                        alignment = copy.deepcopy(getattr(self.transcriber, "last_alignment", None))
            with self.lock:
                if self.interrupted:
                    return
                if job.cancelled:
                    self._update(job, "cancelled")
                    path.unlink(missing_ok=True)
                    self._job_status(job, "idle", "Cancelled")
                    return
                if not text:
                    if self.job is job:
                        self.preview_text = ""
                    self._update(job, "failed", error="No speech detected")
                    retained = (
                        "audio discarded"
                        if config.save_history == "on"
                        else "history disabled; audio discarded"
                    )
                    self._job_status(job, "idle", f"No speech detected; {retained}")
                    return
                # Raw text survives a crash or timeout during optional cleanup.
                self._update(
                    job,
                    "transcribing",
                    text,
                    raw_text=text,
                    duration=job.duration,
                    model=config.model,
                    language=config.language,
                )
            if job.prompt_override is None and job.mode in ("dictation", "edit", "command"):
                try:
                    from .routing import select_prompt

                    config = select_prompt(
                        config,
                        job.metadata.get("app", ""),
                        mode=job.mode,
                        profiles=job.profiles,
                        routes=job.routes,
                    )
                # The job or IPC response reports the failure; avoid duplicating private data in logs.
                except Exception as exc:  # allow: exception-handling  # noqa: BLE001
                    ai_warning = f"Prompt routing failed ({exc}); using original transcription"
                    config = replace(config, ai_enabled="off")
                    if self.job is job:
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
                job, config, text, alignment, has_target=target is not None
            )
            text = formatted.text
            cleanup = cleanup_transcript(
                text, config, job.cancel_event, lambda partial: self._publish_cleanup(job, partial)
            )
            text = cleanup.text
            ai_warning = cleanup.warning or ai_warning
            with self.lock:
                if self.interrupted:
                    return
                if job.cancelled:
                    self._update(job, "cancelled")
                    path.unlink(missing_ok=True)
                    self._job_status(job, "idle", "Cancelled")
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
                if self.interrupted:
                    return
                if job.cancelled:
                    self._update(job, "cancelled")
                    path.unlink(missing_ok=True)
                    self._job_status(job, "idle", "Cancelled")
                else:
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
                key = getattr(self.transcriber, "stream_key", None)
                if job.cancelled and key and key[0] == job.audio_path:
                    with self.model_lock:
                        reset_owned_stream(self.transcriber, job.audio_path)
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
