"""Own microphone transitions and release capture before final processing."""

import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import cast

from .audio import (
    SHORT_RECORDING_SECONDS,
    CaptureSession,
    CaptureTeardownError,
    Recording,
    is_subsecond_silence,
)
from .config import Config, runtime_dir
from .delivery import Target
from .desktop import MediaSession, choose_source, play_sound
from .microphone import MicrophoneSelection, SwitchOutcome, selection
from .recording_delivery import TranscriptDelivery
from .recording_destination import FocusGate, StopDestination, snapshot_for_stop
from .recording_job import QUICK_TOGGLE_SECONDS, DictationJob
from .recording_preview import PreviewWork, RecordingPreview
from .recording_state import RecordingState

CaptureFactory = Callable[[Config, Path], Recording]
StatusReader = Callable[[], dict[str, object]]
FinalizeCapture = Callable[[DictationJob, Target, bool], None]


class RecordingCapture:
    def __init__(
        self,
        state: RecordingState,
        delivery: TranscriptDelivery,
        factory: CaptureFactory,
        preview: RecordingPreview,
        finalize: FinalizeCapture,
    ) -> None:
        self.state = state
        self.lock = state.lock
        self.delivery = delivery
        self.factory = factory
        self.live_preview = preview
        self.finalize = finalize
        self.config = Config()
        self.microphone = MicrophoneSelection(self.lock)
        self.recorder: Recording | None = None
        self.activated_at = 0.0
        self.started: float | None = None
        self.stop_done = threading.Event()
        self.stop_done.set()
        self.media = MediaSession()

    def seconds(self) -> float:
        return round(time.monotonic() - self.started, 1) if self.started else 0

    def is_false_start(self) -> bool:
        return self.state.phase == "starting" or (
            self.state.phase == "recording"
            and time.monotonic() < self.activated_at + QUICK_TOGGLE_SECONDS
        )

    def stop_command(self, data: Mapping[str, object], focus_gate: FocusGate) -> None:
        if self.state.phase == "starting":
            job = cast("DictationJob", self.state.job)  # Starting always has a job.
            job.cancelled = True
            job.cancel_event.set()
            self.state.message = "Cancelling microphone start"
            return
        if self.state.phase not in ("recording", "switching", "capture_failed"):
            raise ValueError("Not recording")
        self.request_stop(
            focus_gate, automatic=bool(data.get("automatic")), spoken=bool(data.get("spoken"))
        )

    def request_stop(self, focus_gate: FocusGate, *, automatic: bool, spoken: bool) -> None:
        target = (
            "capture-focus"
            if self.config.delivery in ("x11", "wayland")
            and self.state.phase != "capture_failed"
            and (not automatic or spoken)
            else None
        )
        self.stop(target, focus_gate)

    def select(self, data: Mapping[str, object], status: StatusReader) -> dict[str, object]:
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
                return status()
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
            return status()

    def _capture_switched(self, job: DictationJob, outcome: SwitchOutcome, requested: str) -> None:
        current = outcome.source
        if current is None:
            # Recover existing audio without selecting a new delivery destination.
            self.state.report_error(
                f"Microphone switch failed; recovering recorded audio. {outcome.error}"
            )
            self.stop(None)
            return
        self.config = replace(self.config, audio_source=current)
        job.config = replace(job.config, audio_source=current)
        self.microphone.mark_ready(current)
        if outcome.failed_previous:
            failed, error = outcome.failed_previous
            self.microphone.mark_failed(failed)
            self.state.report_error(
                f"Previous microphone {failed} failed; switched to {current}. {error}"
            )
        self.live_preview.reset_meter()
        self.state.phase = "recording"
        job.phase = "recording"
        self.state.message = f"Listening on {current}; microphone switch has a capture gap"
        if current != requested:
            self.state.report_error(
                f"Selected microphone failed; resumed {current}. {outcome.error}"
            )

    def _remember_microphone(self, source: str, saved: Config, *, persist: bool) -> None:
        try:
            self.microphone.remember(source, Config.load() if persist else saved, persist=persist)
        except (OSError, ValueError, TypeError) as exc:
            self.microphone.remember(source, saved, persist=False)
            self.state.report_error(
                f"Microphone switched for this session; preference not saved. {exc}"
            )

    def open(self, job: DictationJob, config: Config, status: StatusReader) -> dict[str, object]:
        self.activated_at = time.monotonic()

        def start(chosen: Config, timeout: float) -> Recording:
            factory = self.factory
            capture: Recording
            if factory is CaptureSession:
                capture = CaptureSession(
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
                    return status()
            capture.stop()
        # The job or IPC response reports the failure; avoid duplicating private data in logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            failure = exc

        with self.lock:
            cancelled = job.cancel_event.is_set()
            if not self.state.interrupted:
                if cancelled:
                    self.state.discard(job, "Microphone start cancelled")
                else:
                    error = str(failure)
                    self.state.update_job(job, "failed", error=error)
                    self.state.mark_job(job, "idle", error)
            job.audio_path.unlink(missing_ok=True)
            job.done.set()
        self.media.resume()
        if failure is not None and not cancelled and not self.state.interrupted:
            raise failure
        return status()

    def detach(self) -> Recording | None:
        """Give the caller the not-yet-stopping recorder under the state lock."""
        recorder, self.recorder = self.recorder, None
        return recorder

    def discard_after_logout(self, recorder: Recording | None) -> None:
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

    def stop(self, target: object, focus_gate: FocusGate | None = None) -> None:
        # These phases retain both the job and its recorder until stopping starts.
        job = cast("DictationJob", self.state.job)
        recorder, self.recorder = cast("Recording", self.recorder), None
        if self.started:
            job.duration = time.monotonic() - self.started
        job.phase = "stopping"
        self.stop_done.clear()
        self.state.notice = None
        self.state.phase, self.state.message, self.started = "stopping", "Stopping microphone", None
        worker = threading.Thread(
            target=self._stopped, args=(job, recorder, target, focus_gate), daemon=True
        )
        self.state.worker = worker
        worker.start()

    def _stopped(
        self, job: DictationJob, recorder: Recording, target: object, focus_gate: FocusGate | None
    ) -> None:
        try:
            focus_started = time.monotonic()
            destination = self._stop_destination(job, target)
            # A slow desktop probe can keep capture open past the short-take cutoff.
            ignore_late_audio = (
                job.duration
                < SHORT_RECORDING_SECONDS
                <= job.duration + (time.monotonic() - focus_started)
            )
            job.context = destination.context
            if focus_gate:
                focus_gate.resolved()
            try:
                recorder.stop()
            finally:
                self.stop_done.set()
                self.media.resume()
            silent = not job.cancelled and is_subsecond_silence(
                job.audio_path, ignore_late_audio=ignore_late_audio
            )
            if not (job.quiet_discard or silent):
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
            if silent and not job.cancelled:
                job.quiet_discard = job.cancelled = True
                job.cancel_event.set()
            if not job.cancelled:
                self.state.mark_job(job, "queued", "Waiting for earlier dictation")
                self.state.update_job(job, "transcribing")
        target = None if focus_gate and not focus_gate.valid else destination.target
        self.finalize(job, cast("Target", target), destination.automatic)

    def _stop_destination(self, job: DictationJob, target: object) -> StopDestination:
        # An unavailable desktop must never prevent the microphone from stopping.
        if not job.quiet_discard:
            with suppress(Exception):
                return snapshot_for_stop(
                    self.delivery, job.config, target, start_target=job.start_target, mode=job.mode
                )
        return StopDestination(
            None,
            automatic=target == "capture-focus",
            context=None,
            metadata_target=None,
            collect_metadata=False,
        )

    def _publish_preview(self, text: str) -> None:
        self.state.preview_text = text

    def tick(self, transcriber: object) -> None:
        with self.lock:
            if self.state.phase != "recording":
                return
            job = cast("DictationJob", self.state.job)  # Recording always has a job.
            self.live_preview.sample(job.audio_path, self.state.preview_text)
            if not self.is_false_start() and self.live_preview.due(self.config):
                self.live_preview.schedule(
                    PreviewWork(
                        job.audio_path,
                        self.config,
                        transcriber,
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
                self.request_stop(gate, automatic=True, spoken=spoken)
            # The capture stop worker reports actionable errors.
            except Exception:  # allow: exception-handling
                if spoken:
                    raise
                return
