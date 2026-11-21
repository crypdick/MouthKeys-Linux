from threading import RLock
from types import SimpleNamespace

import pytest

from mouthkeys import microphone
from mouthkeys.config import Config


@pytest.fixture
def selection_with_clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(microphone, "time", SimpleNamespace(monotonic=lambda: now[0]))
    return microphone.MicrophoneSelection(RLock()), now


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"source": 1},
        {"source": " \t"},
        {"source": "x" * 501},
        {"source": "USB\nMonitor"},
        {"source": "usb", "persist": 1},
    ],
)
def test_invalid_switch_request_cannot_reach_a_capture_adapter(data):
    with pytest.raises(ValueError, match="printable source name and boolean persist"):
        microphone.selection(data)


def test_session_pick_and_quarantine_do_not_rewrite_saved_microphone(
    selection_with_clock, monkeypatch
):
    selection, now = selection_with_clock
    config = Config(audio_source="saved")
    config.save()
    selection.remember("session", config, persist=False)
    selection.mark_failed("session")

    def alternatives(request, *, excluded):
        assert request.audio_source == "session"
        assert "session" in excluded
        return ["fallback"]

    monkeypatch.setattr(microphone, "source_candidates", alternatives)
    assert selection.startup_config(config).audio_source == "fallback"
    assert Config.load().audio_source == "saved"
    now[0] = 60
    assert selection.startup_config(config).audio_source == "session"
    assert not selection.quarantine


def test_quarantine_caps_memory_and_does_not_retry_without_candidates(
    selection_with_clock, monkeypatch
):
    selection, _ = selection_with_clock
    for index in range(66):
        selection.mark_failed(f"source-{index}")
    assert len(selection.quarantine) == 64
    assert "source-0" not in selection.quarantine

    monkeypatch.setattr(microphone, "source_candidates", lambda *args, **kwargs: [])
    with pytest.raises(ValueError, match="failed/Never lists"):
        selection.startup_config(Config(audio_source="source-65"))
    selection.mark_ready("source-65")
    assert selection.startup_config(Config(audio_source="source-65")).audio_source == "source-65"


@pytest.mark.parametrize("interrupted", [False, True])
def test_cancelled_open_never_quarantines_a_healthy_source(selection_with_clock, interrupted):
    selection, _ = selection_with_clock
    calls = []

    def start(config, timeout):
        calls.append((config.audio_source, timeout))
        raise OSError("Readiness interrupted")

    def cancelled():
        return interrupted or bool(calls)

    expected = RuntimeError if interrupted else OSError
    message = "Microphone start cancelled" if interrupted else "Readiness interrupted"
    with pytest.raises(expected, match=message):
        selection.open(Config(audio_source="usb"), start, cancelled)
    assert calls == ([] if interrupted else [("usb", 5)])
    assert not selection.quarantine


def test_capture_timeout_is_shared_across_retry_devices(selection_with_clock, monkeypatch):
    selection, now = selection_with_clock
    calls = []

    def start(config, timeout):
        calls.append((config.audio_source, timeout))
        now[0] += 6
        raise OSError(f"{config.audio_source} not ready")

    monkeypatch.setattr(
        microphone,
        "source_candidates",
        lambda *args, excluded: [
            source for source in ("second", "third") if source not in excluded
        ],
    )
    with pytest.raises(RuntimeError, match="second not ready"):
        selection.open(Config(audio_source="first"), start, lambda: False)
    assert calls == [("first", 5), ("second", 2)]
    assert selection.quarantine.keys() == {"first", "second"}


def test_failed_source_discovery_preserves_original_capture_error(
    selection_with_clock, monkeypatch
):
    selection, _ = selection_with_clock
    original = OSError("Device unplugged")

    def start(_config, _timeout):
        raise original

    def discover(*_args, **_kwargs):
        raise OSError("Source manager unavailable")

    monkeypatch.setattr(microphone, "source_candidates", discover)
    with pytest.raises(RuntimeError, match="Device unplugged") as failure:
        selection.open(Config(audio_source="usb"), start, lambda: False)
    assert failure.value.__cause__ is original

    outcome = selection.switch("usb", Config(audio_source="prior"), start_switch, lambda: True)
    assert outcome.source is None
    assert str(outcome.error) == "Switch adapter declined"


def start_switch(_source, *, timeout):
    assert 0 < timeout <= 5
    raise RuntimeError("Switch adapter declined")


def test_cancelled_switch_does_not_attempt_fallback_or_quarantine_source(
    selection_with_clock, monkeypatch
):
    selection, _ = selection_with_clock
    calls = []
    monkeypatch.setattr(
        microphone, "source_candidates", lambda *a, **kw: pytest.fail("Fallback after cancellation")
    )

    def adapter(source, *, timeout):
        calls.append((source, timeout))
        raise RuntimeError("Stopped while switching")

    assert selection.switch("usb", Config(), adapter, lambda: False) is None
    assert not calls
    assert selection.switch("usb", Config(), adapter, lambda: not calls) is None
    assert calls == [("usb", 5)]
    assert not selection.quarantine


def test_switch_keeps_per_attempt_budget_and_returns_last_failure(
    selection_with_clock, monkeypatch
):
    selection, now = selection_with_clock
    calls = []

    def adapter(source, *, timeout):
        calls.append((source, timeout))
        now[0] += 7
        raise RuntimeError(f"{source} unavailable")

    monkeypatch.setattr(microphone, "source_candidates", lambda *a, **kw: ["prior"])
    outcome = selection.switch("usb", Config(audio_source="prior"), adapter, lambda: True)
    assert calls == [("usb", 5), ("prior", 1)]
    assert outcome.source is None
    assert str(outcome.error) == "prior unavailable"


def test_optional_previous_failure_metadata_never_invalidates_success(selection_with_clock):
    selection, _ = selection_with_clock
    result = selection.switch(
        "usb",
        Config(),
        lambda source, *, timeout: {"failed_source": "previous"},
        lambda: True,
    )
    assert result == microphone.SwitchOutcome("usb", None)


@pytest.mark.parametrize("operation", ["open", "switch"])
def test_device_churn_is_limited_even_if_each_attempt_fails_immediately(
    selection_with_clock, monkeypatch, operation
):
    selection, _ = selection_with_clock
    devices = [f"source-{index}" for index in range(30)]
    calls = []

    def available(_config, *, excluded):
        return [source for source in devices if source not in excluded]

    def fail(source):
        calls.append(source)
        raise OSError(f"{source} failed")

    monkeypatch.setattr(microphone, "source_candidates", available)
    config = Config(audio_source=devices[0])
    if operation == "open":
        with pytest.raises(RuntimeError, match="source-15 failed"):
            selection.open(
                config, lambda chosen, _timeout: fail(chosen.audio_source), lambda: False
            )
    else:
        outcome = selection.switch(
            devices[0], config, lambda source, **_: fail(source), lambda: True
        )
        assert outcome.source is None
        assert str(outcome.error) == "source-15 failed"
    assert calls == devices[:16]
    assert list(selection.quarantine) == devices[:16]
