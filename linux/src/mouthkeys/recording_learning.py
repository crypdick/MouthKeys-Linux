"""Observe edits after verified delivery and coordinate correction review."""

import threading
import time
from _thread import RLock
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Literal

from .accessibility import read_snapshot
from .config import Config
from .learning import CorrectionLearner, LearningProposal

OBSERVATION_POLL_SECONDS = 0.5
MAX_PENDING = 200
LearningCommand = Literal["learning_pending", "learning_accept", "learning_dismiss"]


class RecordingLearning:
    """Share the controller lock only for observation ownership and proposal claims."""

    def __init__(
        self,
        lock: RLock,
        owns_recording: Callable[[str], bool],
        is_interrupted: Callable[[], bool],
        report_error: Callable[[str], None],
    ) -> None:
        self.lock = lock
        self.owns_recording = owns_recording
        self.is_interrupted = is_interrupted
        self.report_error = report_error
        self.stop = threading.Event()
        self.worker: threading.Thread | None = None
        self.proposals: dict[str, tuple[CorrectionLearner, LearningProposal]] = {}
        self.reviewing: set[str] = set()

    def cancel(self) -> None:
        with self.lock:
            self.stop.set()

    def clear(self) -> None:
        with self.lock:
            self.cancel()
            self.proposals.clear()

    def pending(self) -> list[LearningProposal]:
        with self.lock:
            return [proposal for _, proposal in self.proposals.values()]

    def command(
        self, action: LearningCommand, data: Mapping[str, object]
    ) -> list[LearningProposal] | LearningProposal:
        if action == "learning_pending":
            return self.pending()
        return self.review(data.get("id"), accept=action == "learning_accept")

    def review(self, proposal_id: object, *, accept: bool) -> LearningProposal:
        with self.lock:
            if self.is_interrupted():
                raise ValueError("Service is shutting down")
            if not isinstance(proposal_id, str) or proposal_id not in self.proposals:
                raise ValueError("Learning suggestion no longer exists")
            if proposal_id in self.reviewing:
                raise ValueError("Learning suggestion is already being reviewed")
            learner, proposal = self.proposals[proposal_id]
            self.reviewing.add(proposal_id)
        try:
            if accept:
                learner.accept(proposal_id)
            else:
                learner.dismiss(proposal_id)
            with self.lock:
                self.proposals.pop(proposal_id, None)
            return proposal
        finally:
            with self.lock:
                self.reviewing.discard(proposal_id)

    def start(self, recording_id: str, config: Config, text: str, initial: object) -> None:
        if not isinstance(initial, dict):
            return
        with self.lock:
            if not self.owns_recording(recording_id):
                return
            self.cancel()
            stop = self.stop = threading.Event()
            # Automatic acceptance still has to pass the recording ownership check.
            learner = CorrectionLearner(replace(config, learning_confirmation="on"))
            if not learner.begin(text, initial):
                return
            self.worker = threading.Thread(
                target=self._observe,
                args=(recording_id, config, learner, dict(initial), stop),
                daemon=True,
            )
            self.worker.start()

    def _observe(
        self,
        recording_id: str,
        config: Config,
        learner: CorrectionLearner,
        target: dict[str, object],
        stop: threading.Event,
    ) -> None:
        deadline = time.monotonic() + config.learning_observe_seconds
        try:
            while not stop.wait(OBSERVATION_POLL_SECONDS) and time.monotonic() < deadline:
                if Config.load().automatic_learning != "on":
                    return
                # The helper has its own bounded timeout. Never hold the state lock during AT-SPI.
                snapshot = read_snapshot(target)
                if stop.is_set():
                    return
                if proposal := learner.observe(snapshot):
                    if Config.load().automatic_learning == "on":
                        self._complete(recording_id, config, learner, proposal, stop)
                    return
                if learner.observation is None:
                    return
        # Surface a notice without logging field contents.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            with self.lock:
                if not stop.is_set():
                    self.report_error(f"Correction learning unavailable: {exc}")
        finally:
            learner.cancel()

    def _complete(
        self,
        recording_id: str,
        config: Config,
        learner: CorrectionLearner,
        proposal: LearningProposal,
        stop: threading.Event,
    ) -> None:
        with self.lock:
            if stop.is_set() or not self.owns_recording(recording_id):
                return
            if config.learning_confirmation != "off":
                self.proposals[proposal["id"]] = (learner, proposal)
                if len(self.proposals) > MAX_PENDING:
                    self.proposals.pop(next(iter(self.proposals)))
                return
            # Ownership is checked at this point. A later cancel cannot undo this review.
        learner.accept(proposal["id"])
