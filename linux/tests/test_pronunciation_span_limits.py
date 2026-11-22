from types import SimpleNamespace

import numpy as np
import pytest

from mouthkeys import pronunciation_spans
from mouthkeys.pronunciation import PronunciationStore

from .test_pronunciation import utterance
from .test_pronunciation_spans import surrounding, trained


def test_budget_stops_without_accepting_half_compared_candidates():
    samples, alignment = surrounding(trained())
    result = PronunciationStore.correct_spans(alignment["text"], samples, alignment, budget=1)
    assert result["truncated"]
    assert result["comparisons"] == 1
    assert result["cells"] <= 80 * 80
    assert result["corrections"] == []
    assert result["text"] == alignment["text"]


def test_long_search_has_explicit_comparison_and_cell_bounds():
    trained()
    samples = np.tile(utterance(base=1800, seconds=0.3), 200)
    text = " ".join(["word"] * 200)
    words = [{"word": "word", "start": i * 0.3, "end": (i + 1) * 0.3} for i in range(200)]
    result = PronunciationStore.correct_spans(
        text, samples, {"text": text, "words": words}, budget=8
    )
    assert result["truncated"]
    assert result["comparisons"] <= 8
    assert result["cells"] <= 8 * 80 * 80
    assert result["text"] == text


def test_wall_deadline_bounds_scanning_even_before_any_dtw(monkeypatch):
    trained()
    clock = [0]

    def monotonic():
        clock[0] += 0.1
        return clock[0]

    monkeypatch.setattr(pronunciation_spans, "time", SimpleNamespace(monotonic=monotonic))
    audio = np.tile(utterance(base=1800, seconds=0.3), 200)
    text = " ".join(["word"] * 200)
    words = [{"word": "word", "start": i * 0.3, "end": (i + 1) * 0.3} for i in range(200)]
    result = PronunciationStore.correct_spans(
        text, audio, {"text": text, "words": words}, max_seconds=0.15
    )
    assert result["truncated"]
    assert result["comparisons"] == 0
    assert result["text"] == text


def test_late_final_comparison_cannot_correct_transcript(monkeypatch):
    samples = trained()
    clock, comparisons = [0.0], []

    def compare(_candidate, _taken):
        comparisons.append(True)
        if len(comparisons) == 2:
            clock[0] = 2.0
        return 0.0

    monkeypatch.setattr(pronunciation_spans, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(pronunciation_spans, "_span_distance", compare)
    alignment = {
        "text": "misspelling",
        "words": [{"word": "misspelling", "start": 0, "end": len(samples) / 16000}],
    }
    result = PronunciationStore.correct_spans(alignment["text"], samples, alignment, max_seconds=1)
    assert result["comparisons"] == 2
    assert result["truncated"]
    assert result["corrections"] == []
    assert result["text"] == alignment["text"]


def test_unrepresentable_time_limit_is_a_validation_error():
    with pytest.raises(ValueError, match="span time limit"):
        PronunciationStore.correct_spans("", np.zeros(16000), {}, max_seconds=10**400)


@pytest.mark.parametrize("budget", [0, True, 20001])
def test_comparison_budget_requires_bounded_integer(budget):
    with pytest.raises(ValueError, match="span budget"):
        PronunciationStore.correct_spans("", np.zeros(1), {}, budget=budget)


def test_cell_cap_stops_before_first_comparison(monkeypatch):
    audio = trained()
    monkeypatch.setattr(pronunciation_spans, "MAX_DISTANCE_CELLS", 1)
    text = "misspelling"
    words = [{"word": text, "start": 0, "end": len(audio) / 16000}]
    result = PronunciationStore.correct_spans(text, audio, {"text": text, "words": words})
    assert result["truncated"]
    assert result["candidates"] == 1
    assert result["comparisons"] == result["cells"] == 0
    assert result["text"] == text


def test_deadline_can_expire_after_selecting_window_but_before_reading_samples(monkeypatch):
    audio = trained()
    clock = iter([0, 0, 2])
    monkeypatch.setattr(
        pronunciation_spans, "time", SimpleNamespace(monotonic=lambda: next(clock, 2))
    )
    text = "word"
    words = [{"word": text, "start": 0, "end": len(audio) / 16000}]
    result = PronunciationStore.correct_spans(
        text, audio, {"text": text, "words": words}, max_seconds=1
    )
    assert result["truncated"]
    assert result["candidates"] == result["comparisons"] == 0
