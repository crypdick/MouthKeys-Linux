import copy
import wave
from pathlib import Path

import numpy as np
import pytest

from mouthkeys import pronunciation_spans
from mouthkeys.pronunciation import PronunciationStore, acoustic_distance, acoustic_features
from mouthkeys.word_alignment import align_words

from .test_pronunciation import utterance


def fixture():
    path = Path(__file__).parent / "resources" / "dictation_fixture.wav"
    with wave.open(str(path)) as source:
        return (
            np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32)
            / 32768
        )


def trained(label="MouthKeys"):
    voice = fixture()
    PronunciationStore.enroll(label, voice)
    PronunciationStore.enroll(label, voice * 0.8)
    return voice


def surrounding(voice):
    before = utterance(base=1100, seconds=0.6)
    after = utterance(base=1800, seconds=0.7)
    samples = np.concatenate([before, voice, after])
    first, end = len(before) / 16000, (len(before) + len(voice)) / 16000
    words = [
        {"word": "Please", "start": 0.0, "end": first / 2},
        {"word": " use", "start": first / 2, "end": first},
        {"word": " mouth", "start": first, "end": (first + end) / 2},
        {"word": " keys,", "start": (first + end) / 2, "end": end},
        {"word": " tomorrow!", "start": end, "end": len(samples) / 16000},
    ]
    return samples, {"text": "Please use mouth keys, tomorrow!", "source": "memory", "words": words}


def test_real_recorded_clip_inside_distractors_replaces_only_true_timed_word_span():
    samples, alignment = surrounding(trained())
    original = copy.deepcopy(alignment)
    result = PronunciationStore.correct_spans(alignment["text"], samples, alignment)
    assert result["text"] == "Please use MouthKeys, tomorrow!"
    assert not result["truncated"]
    assert len(result["corrections"]) == 1
    correction = result["corrections"][0]
    assert (correction["word_start"], correction["word_end"]) == (2, 4)
    assert correction["heard"] == "mouth keys"
    corrected = result["words"][2]
    assert corrected["word"] == "MouthKeys,"
    assert corrected["start"] == alignment["words"][2]["start"]
    assert corrected["end"] == alignment["words"][3]["end"]
    assert corrected["kind"] == "pronunciation_span"
    assert len(corrected["original_words"]) == 2
    assert result["words"][0]["start"] == alignment["words"][0]["start"]
    assert result["words"][-1]["end"] == alignment["words"][-1]["end"]
    assert alignment == original
    # Real waveform/template math + fake timed-ASR metadata, NOT an ASR accuracy claim.


def test_span_correction_preserves_quotes_punctuation_and_unrelated_text_exactly():
    samples, alignment = surrounding(trained("Correct Name"))
    alignment["text"] = "Please use “mouth keys,” tomorrow!"
    alignment["words"][2]["word"] = " “mouth"
    alignment["words"][3]["word"] = " keys,”"
    result = PronunciationStore.correct_spans(alignment["text"], samples, alignment)
    assert result["text"] == "Please use “Correct Name,” tomorrow!"
    assert result["words"][2]["word"] == "“Correct Name,”"
    # Multiword label has one observed phrase interval, not fabricated token times.
    assert len(result["words"]) == 4
    assert len(result["words"][2]["original_words"]) == 2


def test_two_disjoint_matches_apply_without_offset_drift_or_losing_timestamps():
    voice = trained()
    separator = utterance(base=1800, seconds=0.7)
    samples = np.concatenate([voice, separator, voice])
    duration, gap = len(voice) / 16000, len(separator) / 16000
    alignment = {
        "text": "mouth keys, then mouth keys.",
        "source": "memory",
        "words": [
            {"word": "mouth", "start": 0.0, "end": duration / 2},
            {"word": " keys,", "start": duration / 2, "end": duration},
            {"word": " then", "start": duration, "end": duration + gap},
            {"word": " mouth", "start": duration + gap, "end": duration + gap + duration / 2},
            {"word": " keys.", "start": duration + gap + duration / 2, "end": duration * 2 + gap},
        ],
    }
    result = PronunciationStore.correct_spans(alignment["text"], samples, alignment)
    assert result["text"] == "MouthKeys, then MouthKeys."
    assert len(result["corrections"]) == 2
    assert [(word["start"], word["end"]) for word in result["words"]] == [
        (0.0, duration),
        (duration, duration + gap),
        (duration + gap, duration * 2 + gap),
    ]


def test_similar_competing_enrolled_labels_abstain_instead_of_overwriting():
    voice = trained("First Name")
    PronunciationStore.enroll("Other Name", voice)
    PronunciationStore.enroll("Other Name", voice * 0.9)
    samples, alignment = surrounding(voice)
    result = PronunciationStore.correct_spans(alignment["text"], samples, alignment)
    assert result["text"] == alignment["text"]
    assert result["corrections"] == []


def test_unrelated_audio_spans_are_not_changed_despite_recognized_wrong_spelling():
    trained()
    samples = utterance(base=1800, seconds=1.2)
    text = "mouth keys"
    alignment = {
        "text": text,
        "source": "memory",
        "words": [
            {"word": "mouth", "start": 0.0, "end": 0.6},
            {"word": "keys", "start": 0.6, "end": 1.2},
        ],
    }
    result = PronunciationStore.correct_spans(text, samples, alignment)
    assert result["text"] == text
    assert not result["corrections"]


def test_wrong_recording_source_and_misaligned_text_fail_closed(tmp_path):
    samples, alignment = surrounding(trained())
    alignment["source"] = "another-recording"
    path = tmp_path / "does-not-exist.wav"
    result = PronunciationStore.correct_spans(alignment["text"], path, alignment)
    assert result["corrections"] == []
    assert "another recording" in result["skipped_reason"]
    alignment["words"][0]["word"] = "Different"
    result = PronunciationStore.correct_spans(alignment["text"], samples, alignment)
    assert not result["corrections"]
    assert result["skipped_reason"]


@pytest.mark.parametrize(
    "start,end", [(float("nan"), 1), (0, float("inf")), (-1, 1), (1, 0), (True, 1), (0, 100)]
)
def test_invalid_actual_timing_is_never_guessed_or_clamped(start, end):
    samples, alignment = surrounding(trained())
    alignment["words"][0].update(start=start, end=end)
    result = PronunciationStore.correct_spans(alignment["text"], samples, alignment)
    assert result["skipped_reason"]
    assert not result["corrections"]


def test_anchoring_requires_full_text_and_keeps_exact_character_offsets():
    words = [
        {"word": " Hello,", "start": 0.0, "end": 1.0},
        {"word": " world!", "start": 1.0, "end": 2.0},
    ]
    aligned = align_words("Hello,  world!", words, duration=2)
    assert [(word["char_start"], word["char_end"]) for word in aligned] == [(0, 6), (8, 14)]
    with pytest.raises(ValueError):
        align_words("Hello, world! Missing", words)


def test_vectorized_dtw_matches_original_dynamic_programming_recurrence():
    first = acoustic_features(utterance())
    second = acoustic_features(utterance(speed=1.1, noise=0.0001))
    left, right = np.asarray(first["features"]), np.asarray(second["features"])
    costs = np.clip(1 - left @ right.T, 0, 2)
    table = np.full((len(left) + 1, len(right) + 1), np.inf)
    table[0, 0] = 0
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            if abs(i / len(left) - j / len(right)) <= 0.35:
                table[i, j] = costs[i - 1, j - 1] + min(
                    table[i - 1, j], table[i, j - 1], table[i - 1, j - 1]
                )
    assert acoustic_distance(first, second) == pytest.approx(
        table[-1, -1] / max(len(left), len(right)), abs=1e-12
    )


def test_shared_word_intervals_are_replaced_as_one_range_not_duplicated():
    voice = trained()
    duration = len(voice) / 16000
    alignment = {
        "text": "mouth keys",
        "words": [
            {"word": "mouth", "start": 0, "end": duration},
            {"word": "keys", "start": 0, "end": duration},
        ],
    }
    result = PronunciationStore.correct_spans(alignment["text"], voice, alignment)
    assert result["text"] == "MouthKeys"
    assert len(result["corrections"]) == 1
    assert (result["corrections"][0]["word_start"], result["corrections"][0]["word_end"]) == (0, 2)


@pytest.mark.parametrize("bad_edge", ["before", "after"])
def test_acoustic_endpoint_guard_rejects_swallowing_unrelated_neighbors(bad_edge):
    voice = trained()
    unrelated = utterance(base=1800, seconds=0.35)
    samples = np.concatenate([unrelated, voice] if bad_edge == "before" else [voice, unrelated])

    def correction(audio):
        alignment = {
            "text": "mouthkeys",
            "words": [{"word": "mouthkeys", "start": 0, "end": len(audio) / 16000}],
        }
        return PronunciationStore.correct_spans(alignment["text"], audio, alignment)

    assert correction(voice)["text"] == "MouthKeys"
    result = correction(samples)
    assert result["comparisons"] == 2
    assert result["text"] == "mouthkeys"
    assert result["corrections"] == []


def test_actual_wav_recording_reads_only_requested_audio_windows(tmp_path):
    samples, alignment = surrounding(trained())
    path = tmp_path / "recording.wav"
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes((samples * 32767).astype("<i2").tobytes())
    alignment["source"] = str(path)
    result = PronunciationStore.correct_spans(alignment["text"], path, alignment)
    assert result["text"] == "Please use MouthKeys, tomorrow!"
    assert not result["skipped_reason"]


def test_no_ready_enrollments_preserves_original_word_timing_metadata():
    alignment = {
        "text": "Original words",
        "words": [
            {"word": "Original", "start": 0, "end": 0.5},
            {"word": "words", "start": 0.5, "end": 1},
        ],
    }
    result = PronunciationStore.correct_spans(alignment["text"], np.zeros(16000), alignment)
    assert result["text"] == alignment["text"]
    assert [(word["start"], word["end"]) for word in result["words"]] == [(0, 0.5), (0.5, 1)]
    assert not result["truncated"]


@pytest.mark.parametrize("timestamp", [True, float("nan"), 10**400])
def test_alignment_rejects_invalid_timestamps(timestamp):
    with pytest.raises(ValueError, match="Invalid aligned word timing"):
        align_words("one", [{"word": "one", "start": 0, "end": timestamp}])


def test_alignment_rejects_a_nonlist_word_payload():
    with pytest.raises(ValueError, match="Invalid word alignment"):
        align_words("one", {"word": "one", "start": 0, "end": 1})


def test_alignment_rejects_negative_recording_duration():
    with pytest.raises(ValueError, match="Invalid alignment duration"):
        align_words("", [], duration=-1)


def test_alignment_rejects_entries_without_a_word_string():
    with pytest.raises(ValueError, match="Invalid aligned word"):
        align_words("one", [{"start": 0, "end": 1}])


def test_mismatched_transcript_does_not_load_profiles_or_read_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(PronunciationStore, "load", lambda: pytest.fail("loaded profiles"))
    result = PronunciationStore.correct_spans(
        "new text", tmp_path / "missing.wav", {"text": "old text"}
    )
    assert result["text"] == "new text"
    assert result["skipped_reason"] == "Alignment does not match final transcript"


def test_no_enrolled_profiles_still_rejects_malformed_timing():
    result = PronunciationStore.correct_spans(
        "text", np.zeros(16000), {"text": "text", "words": []}
    )
    assert result["words"] == []
    assert "Word timing metadata" in result["skipped_reason"]


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        (np.empty(0), "one hour"),
        (np.zeros((4000, 2)), "finite mono"),
        (np.full(4000, np.nan), "finite mono"),
    ],
)
def test_span_search_rejects_invalid_sample_arrays(audio, message):
    trained()
    with pytest.raises(ValueError, match=message):
        PronunciationStore.correct_spans("", audio, {"text": "", "words": []})


def test_span_search_rejects_stereo_wav(tmp_path):
    trained()
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as recording:
        recording.setparams((2, 2, 16000, 0, "NONE", ""))
        recording.writeframes(bytes(16000))
    with pytest.raises(ValueError, match="mono 16-bit PCM"):
        PronunciationStore.correct_spans("", path, {"text": "", "source": str(path), "words": []})


@pytest.mark.parametrize("duration", [0.1, 10.1])
def test_spans_outside_supported_duration_are_not_candidates(duration):
    trained()
    text = "word"
    words = [{"word": text, "start": 0, "end": duration}]
    result = PronunciationStore.correct_spans(
        text, np.zeros(round(16000 * duration)), {"text": text, "words": words}
    )
    assert result["text"] == text
    assert result["candidates"] == 0
    assert not result["truncated"]


@pytest.mark.parametrize(
    ("text", "silence", "candidates", "comparisons"),
    [("...", False, 0, 0), ("word", True, 1, 0), ("MouthKeys", False, 1, 2)],
)
def test_nonlexical_silent_and_already_correct_spans_are_unchanged(
    text, silence, candidates, comparisons
):
    audio = trained()
    if silence:
        audio = np.zeros_like(audio)
    words = [{"word": text, "start": 0, "end": len(audio) / 16000}]
    result = PronunciationStore.correct_spans(text, audio, {"text": text, "words": words})
    assert result["text"] == text
    assert result["corrections"] == []
    assert result["candidates"] == candidates
    assert result["comparisons"] == comparisons


@pytest.mark.parametrize(("voiced_seconds", "comparisons"), [(0.32, 0), (0.50, 2)])
def test_duration_guard_applies_to_speech_not_leading_or_trailing_silence(
    voiced_seconds, comparisons
):
    PronunciationStore.enroll("label", utterance())
    PronunciationStore.enroll("label", utterance())
    audio = np.pad(utterance(seconds=voiced_seconds), (2400, 2400))
    text = "other"
    words = [{"word": text, "start": 0, "end": len(audio) / 16000}]
    result = PronunciationStore.correct_spans(text, audio, {"text": text, "words": words})
    assert result["candidates"] == 1
    assert result["comparisons"] == comparisons
    assert result["text"] == text


def test_overlapping_acceptable_windows_use_the_closest_match(monkeypatch):
    audio = trained()
    duration = len(audio) / 16000
    # Every timed window matches; the two-word window is closest and excludes both singles.
    monkeypatch.setattr(
        pronunciation_spans,
        "_span_distance",
        lambda candidate, enrolled: 0.0 if candidate["duration"] > duration / 2 else 0.1,
    )
    words = [
        {"word": "mouth", "start": 0, "end": len(audio) / 32000},
        {"word": "keys", "start": len(audio) / 32000, "end": len(audio) / 16000},
    ]
    # Use untrimmed timing so the short fixture's single words also qualify for comparison.
    take = acoustic_features(audio)
    monkeypatch.setattr(
        pronunciation_spans,
        "acoustic_features",
        lambda samples: {
            **take,
            "duration": len(samples) / 16000,
        },
    )
    result = PronunciationStore.correct_spans(
        "mouth keys", audio, {"text": "mouth keys", "words": words}
    )
    assert result["comparisons"] == 6
    assert result["text"] == "MouthKeys"
    assert [(match["word_start"], match["word_end"]) for match in result["corrections"]] == [(0, 2)]
