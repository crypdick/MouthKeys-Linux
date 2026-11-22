"""Deliver a committed transcript, optionally sending a key after verified insertion."""

from _thread import RLock
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from . import desktop_input
from .config import Config
from .delivery import Target
from .recording_job import DictationJob
from .recording_learning import RecordingLearning

SEND_REVIEW_SECONDS = 0.5


@runtime_checkable
class TranscriptDelivery(Protocol):
    def deliver(self, text: str, config: Config, target: Target) -> str: ...


@dataclass(frozen=True)
class DeliveryRequest:
    text: str
    config: Config
    target: Target
    automatic: bool
    should_send: bool


@dataclass(frozen=True)
class DeliveryOutcome:
    status: str
    paste_warning: str | None
    send_warning: str


def delivery_message(request: DeliveryRequest, outcome: DeliveryOutcome) -> str:
    if outcome.paste_warning:
        return f"Copied — auto-paste skipped. {outcome.paste_warning}"
    if outcome.status == "send_requested":
        return "Send requested — no text inserted"
    if not request.text:
        return "No text to deliver"
    if outcome.status == "copied":
        return "Copied — paste into your app"
    if request.config.save_history == "on":
        return "Insertion requested — transcript saved in history"
    return "Insertion requested — history disabled"


def _paste_warning(request: DeliveryRequest, status: str) -> str | None:
    if status != "copied" or not request.automatic:
        return None
    problem = desktop_input.automatic_delivery_warning(request.config)
    if problem:
        return problem
    if not request.target:
        return "No destination was captured."
    if request.config.destination_guard == "field" and not isinstance(request.target, dict):
        return (
            "Could not identify the focused field. "
            "Choose “Field first, window fallback” in Text delivery."
        )
    return "Could not verify the original destination was unchanged."


class RecordingDelivery:
    def __init__(
        self,
        adapter: TranscriptDelivery,
        lock: RLock,
        learning: RecordingLearning,
        interrupted: Callable[[], bool],
    ) -> None:
        self.adapter = adapter
        self.lock = lock
        self.learning = learning
        self.interrupted = interrupted

    def deliver(self, job: DictationJob, request: DeliveryRequest) -> DeliveryOutcome:
        """Call after publishing the durable transcript and claiming delivery."""
        status = (
            self.adapter.deliver(request.text, request.config, request.target)
            if request.text
            else "transcribed"
        )
        paste_warning = _paste_warning(request, status)
        if status == "pasted" and request.config.automatic_learning == "on":
            self.learning.start(
                job.item_id, job.config, request.text, getattr(self.adapter, "last_verified", None)
            )
        status, send_warning = self._spoken_send(job, request, status)
        return DeliveryOutcome(status, paste_warning, send_warning)

    def _spoken_send(
        self, job: DictationJob, request: DeliveryRequest, status: str
    ) -> tuple[str, str]:
        if not request.should_send:
            return status, ""
        if request.text and status == "copied":
            return status, "Spoken Send skipped: text copied; insertion destination unavailable"
        if request.text and status not in ("paste_requested", "insert_requested", "pasted"):
            return status, ""

        with self.lock:
            job.send_pending = True
        try:
            job.cancel_event.wait(SEND_REVIEW_SECONDS)
            with self.lock:
                dispatch = self._can_send(job)
            if dispatch and self._send_key(job, request):
                return (status if request.text else "send_requested"), ""
            return status, self._skip_reason(job)
        finally:
            with self.lock:
                job.send_pending = False

    def _can_send(self, job: DictationJob) -> bool:
        return not (self.interrupted() or job.cancelled or job.suppress_send or job.send_dispatched)

    def _send_key(self, job: DictationJob, request: DeliveryRequest) -> bool:
        def reserve() -> bool:
            with self.lock:
                if not self._can_send(job):
                    return False
                job.send_dispatched = True
                return True

        send_key = getattr(self.adapter, "send_key", None)
        if not callable(send_key):
            return False
        if getattr(self.adapter, "send_key_guarded", False):
            return bool(
                send_key(
                    request.config,
                    request.target,
                    before_send=reserve,
                    allow_without_insert=not request.text,
                )
            )
        return reserve() and bool(send_key(request.config, request.target))

    def _skip_reason(self, job: DictationJob) -> str:
        with self.lock:
            if job.cancelled or self.interrupted():
                return ""
            if job.suppress_send:
                return "Spoken Send skipped: a newer recording started"
        return "Spoken Send skipped: destination could not be verified"
