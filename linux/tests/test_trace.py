import pytest

from mouthkeys.trace import Trace


def test_inactive_trace_ignores_audio_before_recording():
    trace = Trace()
    trace.ingest(1, 0.25)
    trace.advance(0.25)
    assert trace.current == trace.shown == [2] * trace.count
    assert trace.pending_peak == 0
    assert trace.last_voice is None


def test_voice_moves_between_audio_updates_and_eases_new_bars():
    trace = Trace()
    trace.begin(1)
    trace.ingest(0.8, 1)
    trace.advance(1.016)
    first = trace.fraction
    trace.advance(1.032)  # No new IPC/audio sample: the frame clock still slides the trace.
    assert 0 < first < trace.fraction < 1
    trace.advance(1.09)
    assert 2 < trace.shown[-1] < trace.current[-1] <= 30
    trace.advance(1.106)
    assert trace.shown[-1] > 2


def test_silence_and_stalled_capture_hold_scroll_after_voice_hangover():
    trace = Trace()
    trace.begin(1)
    trace.advance(1.1)
    assert trace.fraction == 0
    trace.ingest(0.8, 1.1)
    for i in range(1, 51):
        trace.advance(1.1 + i / 100)
    fraction = trace.fraction
    targets = trace.current.copy()
    trace.advance(3)
    assert trace.fraction == pytest.approx(fraction)
    assert trace.current == targets


def test_stop_settles_in_60ms_and_next_recording_resets_everything():
    trace = Trace()
    trace.begin(1)
    trace.ingest(0.8, 1)
    trace.advance(1.09)
    trace.stop(1.1)
    before = max(trace.shown)
    trace.advance(1.13)
    assert 2 < max(trace.shown) < before
    trace.advance(1.17)
    assert trace.shown == [2] * 63
    trace.begin(2)
    assert trace.fraction == 0
    assert trace.current == trace.shown == [2] * 63
    assert trace.last_voice is None


def test_long_frame_stall_has_bounded_catchup():
    trace = Trace()
    trace.begin(1)
    trace.ingest(1, 100)
    trace.advance(100)
    assert sum(height > 2 for height in trace.current) <= 3
    assert 0 <= trace.fraction < 1


def test_peak_release_uses_calibration_time_not_single_frame_duration():
    trace = Trace()
    trace.begin(0)
    trace.loud_peak = 1.0
    for i in range(1, 61):
        trace.ingest(0, i / 60)
        trace.advance(i / 60)
    assert trace.loud_peak == pytest.approx(0.98, abs=0.003)
