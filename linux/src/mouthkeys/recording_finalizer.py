"""Run a stopped recording through inference, durable commit, and guarded delivery."""

from _thread import LockType
from dataclasses import dataclass
from typing import Any

from .config import Config
from .delivery import Target
from .pronunciation import PronunciationStore
from .recording_assistant import RecordingAssistant
from .recording_cleanup import CleanupResult, cleanup_transcript
from .recording_delivery import DeliveryRequest, RecordingDelivery, delivery_message
from .recording_edit import EditReplacement, claim_edit, edit_message
from .recording_inference import Recognition, SpeechModel, recognize_audio, reset_cancelled_stream
from .recording_job import DictationJob, wait_for_predecessors
from .recording_setup import routed_config
from .recording_state import RecordingState
from .recording_transcript import format_dictation


@dataclass
class _Progress:
    config: Config
    text: str = ""
    routed_warning: str = ""
    assistant_error: str | None = None


class RecordingFinalizer:
    def __init__(
        self,
        state: RecordingState,
        assistant: RecordingAssistant,
        delivery: RecordingDelivery,
        model_lock: LockType,
    ) -> None:
        self.state = state
        self.assistant = assistant
        self.delivery = delivery
        self.model_lock = model_lock

    def finish(
        self,
        job: DictationJob,
        target: Target,
        model: SpeechModel,
        *,
        auto_paste_expected: bool,
    ) -> None:
        progress = _Progress(job.config)
        try:
            recognition = self._recognize(job, model)
            if recognition is None:
                return
            progress.text = recognition.text
            if not self._commit_recognition(job, recognition):
                return
            progress.config, progress.routed_warning = routed_config(job)
            if progress.routed_warning:
                self._warn_current(job, progress.routed_warning)
            if job.mode == "dictation":
                self._finish_dictation(
                    job, recognition, progress, target, auto_paste_expected=auto_paste_expected
                )
            else:
                self._finish_review(job, progress)
        # Report privately through the job. Do not copy transcript/provider payloads into logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            self._failed(job, progress, str(exc))
        finally:
            try:
                reset_cancelled_stream(job, model, self.model_lock)
            finally:
                self._retire(job)

    def _recognize(self, job: DictationJob, model: SpeechModel) -> Recognition | None:
        with self.state.lock:
            if not self.state.may_finalize(job):
                return None
        if not wait_for_predecessors(job, lambda: self.state.interrupted):
            return None
        with self.state.lock:
            if not self.state.may_finalize(job):
                return None
            self.state.mark_job(job, "transcribing", "Transcribing")
        return recognize_audio(
            job, model, self.model_lock, lambda: job.cancelled or self.state.interrupted
        )

    def _commit_recognition(self, job: DictationJob, result: Recognition) -> bool:
        with self.state.lock:
            if not self.state.may_finalize(job):
                return False
            if result.skipped_silence:
                self.state.discard(job, "Silent recording skipped")
                return False
            config = job.config
            if not result.text:
                if self.state.job is job:
                    self.state.preview_text = ""
                self.state.update_job(job, "failed", error="No speech detected")
                retained = (
                    "audio discarded"
                    if config.save_history == "on"
                    else "history disabled; audio discarded"
                )
                self.state.mark_job(job, "idle", f"No speech detected; {retained}")
                return False
            # Raw text survives a crash or timeout during optional cleanup.
            self.state.update_job(
                job,
                "transcribing",
                result.text,
                raw_text=result.text,
                duration=job.duration,
                model=config.model,
                language=config.language,
            )
            return True

    def _warn_current(self, job: DictationJob, warning: str) -> None:
        with self.state.lock:
            if self.state.job is job and not self.state.interrupted:
                self.state.report_error(warning, kind="ai")

    def _finish_review(self, job: DictationJob, progress: _Progress) -> None:
        if job.mode == "training":
            proposal: dict[str, Any] = {"mode": "training", "text": progress.text}
        elif job.mode == "pronunciation":
            with self.state.lock:
                if not self.state.may_finalize(job):
                    return
                # Enrollment writes must not cross the cancellation boundary.
                enrollment = PronunciationStore.enroll(job.label, job.audio_path)
            proposal = {"mode": "pronunciation", "text": job.label, **enrollment}
        else:
            try:
                proposal = self.assistant.propose(job, progress.config, progress.text)
            # Keep assistant failure distinct from capture and delivery failures.
            except Exception as exc:  # allow: exception-handling
                progress.assistant_error = str(exc)
                raise
        progress.text = proposal["text"]
        self._publish_proposal(job, progress.config, proposal, progress.routed_warning)

    def _publish_proposal(
        self, job: DictationJob, config: Config, proposal: dict[str, Any], warning: str
    ) -> None:
        with self.state.lock:
            if not self.state.may_finalize(job):
                return
            text = proposal["text"]
            assistant_requested = job.mode in ("command", "edit")
            self.state.update_job(
                job,
                "transcribed",
                text,
                was_ai_processed=assistant_requested,
                ai_processing_error=warning,
                ai_processing_model=config.ai_model if assistant_requested else "",
            )
            edit = None
            if self.state.job is job:
                if config.edit_delivery == "replace":
                    edit = claim_edit(job, proposal, config)
                self.state.assistant_proposal = proposal
                self.assistant.clear_stream()
                self.state.preview_text = text
            if edit is not None:
                self.state.mark_job(job, "delivering", "Replacing captured selection")
            else:
                message = (
                    "Command finished; execution receipts retained"
                    if proposal.get("execution_count")
                    else "Recording ready for review; nothing inserted"
                )
                self.state.mark_job(job, "idle", message)
        if edit is not None:
            self.apply_edit(edit)

    def apply_edit(self, edit: EditReplacement) -> None:
        result = edit.apply(self.delivery.adapter)
        with self.state.lock:
            if not self.state.interrupted:
                if self.state.job is edit.job or result != "cancelled":
                    self.state.update_job(edit.job, result, edit.text)
                self.state.mark_job(edit.job, "idle", edit_message(result))

    def _finish_dictation(
        self,
        job: DictationJob,
        recognition: Recognition,
        progress: _Progress,
        target: Target,
        *,
        auto_paste_expected: bool,
    ) -> None:
        config = progress.config
        formatted = format_dictation(
            job, config, progress.text, recognition.alignment, has_target=target is not None
        )
        progress.text = formatted.text
        cleanup = cleanup_transcript(
            progress.text,
            config,
            job.cancel_event,
            lambda partial: self._publish_cleanup(job, partial),
        )
        progress.text = cleanup.text
        warning = cleanup.warning or progress.routed_warning
        if not self._commit_dictation(job, config, cleanup, warning):
            return
        request = DeliveryRequest(
            cleanup.text, config, target, auto_paste_expected, formatted.should_send
        )
        outcome = self.delivery.deliver(job, request)
        with self.state.lock:
            if self.state.interrupted:
                return
            self.state.update_job(job, outcome.status, cleanup.text, error=outcome.send_warning)
            message = delivery_message(request, outcome)
            if outcome.paste_warning and self.state.job is job and not warning:
                self.state.report_error(message, kind="paste")
            notices = (message, warning, formatted.pronunciation_warning, outcome.send_warning)
            self.state.mark_job(job, "idle", ". ".join(filter(None, notices)))

    def _commit_dictation(
        self, job: DictationJob, config: Config, cleanup: CleanupResult, warning: str
    ) -> bool:
        with self.state.lock:
            if not self.state.may_finalize(job):
                return False
            if self.state.job is job:
                if cleanup.warning:
                    self.state.report_error(warning, kind="ai")
                self.state.preview_text = cleanup.text
            if cleanup.text:
                self.state.last_text = cleanup.text
            self.state.update_job(
                job,
                "transcribed",
                cleanup.text,
                was_ai_processed=cleanup.applied,
                ai_processing_error=warning,
                ai_processing_model=config.ai_model if cleanup.applied else "",
            )
            self.state.mark_job(job, "delivering", "Delivering transcript")
            return True

    def _publish_cleanup(self, job: DictationJob, text: str) -> None:
        with self.state.lock:
            if self.state.job is job and not job.cancelled and not self.state.interrupted:
                self.state.preview_text = text

    def _failed(self, job: DictationJob, progress: _Progress, error: str) -> None:
        with self.state.lock:
            if not self.state.may_finalize(job):
                return
            if self.state.job is job:
                self.state.preview_text = progress.text
                if progress.assistant_error is not None:
                    self.state.report_error(f"AI {job.mode} request failed: {error}", kind="ai")
            self.state.update_job(
                job,
                "failed",
                progress.text,
                error,
                **(
                    {"ai_processing_error": progress.assistant_error}
                    if progress.assistant_error
                    else {}
                ),
            )
            retained = (
                "Saved in history; audio discarded."
                if progress.config.save_history == "on"
                else "History disabled; audio discarded."
            )
            self.state.mark_job(job, "idle", f"{error}. {retained}")

    def _retire(self, job: DictationJob) -> None:
        with self.state.lock:
            try:
                job.audio_path.unlink(missing_ok=True)
            except OSError:
                if not self.state.interrupted and self.state.job is job:
                    self.state.report_error("Could not discard temporary recording")
            finally:
                if not self.state.interrupted and self.state.job is job:
                    self.assistant.clear_stream()
                    self.state.phase = "idle"
                job.done.set()
