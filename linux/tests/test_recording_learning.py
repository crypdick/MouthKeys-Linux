import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mouthkeys import recording_learning
from mouthkeys.config import Config
from mouthkeys.learning import CorrectionLearner, LearningProposal

from .test_learning import snapshot

PROPOSAL: LearningProposal = {
    "id": "proposal",
    "heard": "wurld",
    "corrected": "world",
    "key": "wurld-world",
}


@pytest.fixture
def observed(monkeypatch):
    Config(automatic_learning="on").save()
    accept, dismiss, notice = Mock(), Mock(), Mock()
    proposals = Mock(return_value=PROPOSAL)
    probe = Mock(return_value=snapshot("world"))
    owns = Mock(return_value=True)
    interrupted = Mock(return_value=False)

    class Learner(CorrectionLearner):
        def observe(self, state, now=None):
            if state is None:
                return super().observe(state, now)
            return proposals(state)

        def accept(self, identity):
            return accept(identity)

        def dismiss(self, identity):
            dismiss(identity)

    monkeypatch.setattr(recording_learning, "CorrectionLearner", Learner)
    monkeypatch.setattr(recording_learning, "read_snapshot", probe)
    monkeypatch.setattr(recording_learning, "OBSERVATION_POLL_SECONDS", 0.001)
    session = recording_learning.RecordingLearning(threading.RLock(), owns, interrupted, notice)
    return SimpleNamespace(
        session=session,
        accept=accept,
        dismiss=dismiss,
        proposals=proposals,
        probe=probe,
        owns=owns,
        interrupted=interrupted,
        notice=notice,
    )


def start(observed, confirmation="on"):
    config = Config(automatic_learning="on", learning_confirmation=confirmation)
    observed.session.start("current", config, "wurld", snapshot("wurld"))


def join(observed):
    worker = observed.session.worker
    assert worker is not None
    worker.join(2)
    assert not worker.is_alive()


def test_observation_requires_a_verified_current_insertion(observed):
    session = observed.session
    config = Config(automatic_learning="on")
    session.start("current", config, "wurld", None)
    observed.owns.return_value = False
    session.start("old", config, "wurld", snapshot("wurld"))
    observed.owns.return_value = True
    session.start("current", config, "wurld", snapshot("different"))
    assert session.worker is None
    assert session.pending() == []
    observed.probe.assert_not_called()


def test_active_observation_respects_opt_out_and_deadline(observed, monkeypatch):
    Config(automatic_learning="off").save()
    start(observed)
    join(observed)
    observed.probe.assert_not_called()
    assert observed.session.pending() == []

    Config(automatic_learning="on").save()
    time = Mock(side_effect=[0, 100])
    monkeypatch.setattr(recording_learning, "time", SimpleNamespace(monotonic=time))
    start(observed)
    join(observed)
    observed.probe.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])
def test_observation_stops_when_the_field_is_lost_or_probe_fails(observed, failure):
    if failure:
        observed.probe.side_effect = OSError("field unavailable")
    else:
        observed.probe.return_value = None
    start(observed)
    join(observed)
    assert observed.session.pending() == []
    if failure:
        observed.notice.assert_called_once_with(
            "Correction learning unavailable: field unavailable"
        )
    else:
        observed.notice.assert_not_called()
        observed.probe.assert_called_once()


@pytest.mark.parametrize("cancel", [False, True])
def test_correction_ready_after_cancellation_or_loss_of_ownership_is_dropped(observed, cancel):
    interrupted, release = threading.Event(), threading.Event()

    def observe(_state):
        interrupted.set()
        assert release.wait(2)
        return PROPOSAL

    observed.proposals.side_effect = observe
    start(observed, confirmation="off")
    try:
        assert interrupted.wait(2)
        if cancel:
            observed.session.cancel()
        else:
            observed.owns.return_value = False
    finally:
        release.set()
        join(observed)
    observed.accept.assert_not_called()
    assert observed.session.pending() == []


def test_error_from_cancelled_probe_is_not_reported(observed):
    entered, release = threading.Event(), threading.Event()

    def probe(_target):
        entered.set()
        assert release.wait(2)
        raise OSError("stale error")

    observed.probe.side_effect = probe
    start(observed)
    try:
        assert entered.wait(2)
        observed.session.cancel()
    finally:
        release.set()
        join(observed)
    observed.notice.assert_not_called()


def test_pending_limit_and_dismissal(observed, monkeypatch):
    monkeypatch.setattr(recording_learning, "MAX_PENDING", 1)
    start(observed)
    join(observed)
    newer = {**PROPOSAL, "id": "newer"}
    observed.proposals.return_value = newer
    start(observed)
    join(observed)
    assert observed.session.pending() == [newer]
    with pytest.raises(ValueError, match="no longer exists"):
        observed.session.review(PROPOSAL["id"], accept=True)
    assert observed.session.review("newer", accept=False) == newer
    observed.dismiss.assert_called_once_with("newer")
    assert observed.session.pending() == []


def test_review_error_releases_claim_and_allows_retry(observed):
    start(observed)
    join(observed)
    observed.accept.side_effect = OSError("dictionary unavailable")
    with pytest.raises(OSError, match="dictionary unavailable"):
        observed.session.review(PROPOSAL["id"], accept=True)
    assert observed.session.pending() == [PROPOSAL]
    observed.accept.side_effect = None
    assert observed.session.review(PROPOSAL["id"], accept=True) == PROPOSAL
    assert not observed.session.pending()


def test_two_concurrent_reviews_cannot_mutate_one_proposal(observed):
    start(observed)
    join(observed)
    entered, release = threading.Event(), threading.Event()

    def accept(_identity):
        entered.set()
        assert release.wait(2)

    observed.accept.side_effect = accept
    with ThreadPoolExecutor(max_workers=1) as pool:
        review = pool.submit(observed.session.review, PROPOSAL["id"], accept=True)
        try:
            assert entered.wait(2)
            with pytest.raises(ValueError, match="already being reviewed"):
                observed.session.review(PROPOSAL["id"], accept=False)
        finally:
            release.set()
        assert review.result(timeout=2) == PROPOSAL
    observed.accept.assert_called_once_with(PROPOSAL["id"])
    observed.dismiss.assert_not_called()
    assert observed.session.pending() == []


@pytest.mark.parametrize("identity", [None, [], "missing"])
def test_review_uses_stable_errors_for_missing_ids_and_shutdown(observed, identity):
    start(observed)
    join(observed)
    with pytest.raises(ValueError, match="no longer exists"):
        observed.session.review(identity, accept=True)
    observed.interrupted.return_value = True
    with pytest.raises(ValueError, match="Service is shutting down"):
        observed.session.review(PROPOSAL["id"], accept=True)
    observed.accept.assert_not_called()


def test_opt_out_during_a_probe_prevents_automatic_acceptance(observed):
    def observe(_state):
        Config(automatic_learning="off").save()
        return PROPOSAL

    observed.proposals.side_effect = observe
    start(observed, confirmation="off")
    join(observed)
    observed.probe.assert_called_once()
    observed.accept.assert_not_called()
    assert observed.session.pending() == []
