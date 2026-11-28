"""Shared recording and UI state, committed under the controller lock."""

import time
import uuid
from threading import RLock, Thread
from typing import Any

from .recording_job import DictationJob
from .storage import History


class RecordingState:
    def __init__(self, history: History) -> None:
        self.history = history
        self.lock = RLock()
        self.phase: str = "idle"
        self.message: str = "Ready"
        self.item_id: str | None = None
        self.cancelled: bool = False
        self.preview_text: str = ""
        self.interrupted: bool = False
        self.notice: dict[str, object] | None = None
        self.notice_serial: int = 0
        self.notice_token: str = uuid.uuid4().hex
        self.last_text: str = ""
        self.job: DictationJob | None = None
        self.jobs: list[DictationJob] = []
        self.worker: Thread | None = None
        self.assistant_proposal: dict[str, Any] | None = None

    def report_error(self, message: str, kind: str = "error") -> None:
        with self.lock:
            self.notice_serial += 1
            self.notice = {
                "serial": f"{self.notice_token}:{self.notice_serial}",
                "message": message,
                "at": time.time(),
                "kind": kind,
            }

    def pending_jobs(self) -> list[DictationJob]:
        return [job for job in self.jobs if not job.done.is_set()]

    def update_job(
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

    def mark_job(self, job: DictationJob, phase: str, message: str) -> None:
        with self.lock:
            job.phase = phase
            if self.job is job and not self.interrupted:
                self.phase, self.message = phase, message

    def discard(self, job: DictationJob, message: str = "Cancelled") -> None:
        if job.quiet_discard:
            if job.config.save_history == "on":
                self.history.forget(job.item_id)
            self.mark_job(job, "idle", "Cancelled")
            return
        self.update_job(job, "cancelled")
        self.mark_job(job, "idle", message)

    def may_finalize(self, job: DictationJob) -> bool:
        """Call under the state lock so an interrupted row cannot be overwritten."""
        if self.interrupted:
            return False
        if job.cancelled:
            self.discard(job)
            return False
        return True
