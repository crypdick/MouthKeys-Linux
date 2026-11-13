from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mouthkeys.config import Config
from mouthkeys.forced_alignment import MAX_SEGMENTS, WhisperForcedAligner
from mouthkeys.transcribe import Transcriber


def config(**kwargs):
    values = {"pronunciation_matching": "on", "pronunciation_aligner_model": "cached-aligner"}
    return replace(Config(), **{**values, **kwargs})


@pytest.fixture
def aligner(monkeypatch):
    from faster_whisper import tokenizer

    class Tokenizer:
        sot_sequence = [1, 2, 3]
        eot = 1000

        def __init__(self, *args):
            pass

        def encode(self, text):
            assert text.startswith(" ") and not text.startswith("  ")
            return [10, 11]

    monkeypatch.setattr(tokenizer, "Tokenizer", Tokenizer)

    class Extractor:
        nb_max_frames = 3000

        def __call__(self, audio):
            return np.zeros((80, len(audio) // 160 + 1), dtype=np.float32)

    calls = []
    model = SimpleNamespace(
        feature_extractor=Extractor(),
        model=SimpleNamespace(
            is_multilingual=True, detect_language=lambda encoded: [[("<|en|>", 0.99)]]
        ),
        hf_tokenizer=object(),
        max_length=448,
        encode=lambda features: "encoded",
        find_alignment=lambda *args: (
            calls.append(args)
            or [[{"word": "Hello", "start": 0.1, "end": 0.4, "probability": 0.9}]]
        ),
    )
    value = WhisperForcedAligner()
    monkeypatch.setattr(value, "_load", lambda cfg: model)
    return value, model, calls


def test_forced_alignment_uses_real_segment_bounds_and_preserves_text_and_score(aligner):
    value, model, calls = aligner
    segment = SimpleNamespace(start=0.5, end=1.5, text="Hello", avg_logprob=-0.2)
    result, warning = value.align(np.zeros(32000), [segment], config())
    assert not warning
    assert calls[0][1:] == ([[10, 11]], "encoded", 100)
    assert result[0].text == "Hello" and result[0].avg_logprob == -0.2
    assert vars(result[0].words[0]) == {
        "word": "Hello",
        "start": 0.6,
        "end": 0.9,
        "probability": 0.9,
        "alignment_source": "whisper-forced:cached-aligner",
    }
    assert not hasattr(segment, "words")


def test_native_word_metadata_prevents_extra_model_load(monkeypatch):
    value = WhisperForcedAligner()
    monkeypatch.setattr(value, "_load", lambda config: pytest.fail("Loaded optional second model"))
    segment = SimpleNamespace(start=0, end=1, text="Hello", words=[object()])
    result, warning = value.align(np.zeros(16000), [segment], config())
    assert result == [segment] and not warning


def test_alignment_opt_out_releases_model_and_does_no_work():
    value = WhisperForcedAligner()
    value.model, value.key = object(), "old-model"
    segments = [SimpleNamespace(start=0, end=1, text="Hello")]
    assert value.align(np.zeros(16000), segments, config(pronunciation_matching="off")) == (
        segments,
        "",
    )
    assert value.model is value.key is None


@pytest.mark.parametrize(
    "word",
    [
        {"word": "Changed", "start": 0.1, "end": 0.4, "probability": 0.9},
        {"word": "Hello", "start": 0.1, "end": 0.4, "probability": 0.1},
        {"word": "Hello", "start": float("nan"), "end": 0.4, "probability": 0.9},
        {"word": "Hello", "start": 0.1, "end": 2, "probability": 0.9},
        {"word": "Hello", "start": 0.4, "end": 0.4, "probability": 0.9},
    ],
)
def test_uncertain_alignment_preserves_raw_asr_and_reports_warning(aligner, word):
    value, model, _ = aligner
    model.find_alignment = lambda *a: [[word]]
    segment = SimpleNamespace(start=0, end=1, text="Hello")
    result, warning = value.align(np.zeros(16000), [segment], config())
    assert result == [segment]
    assert "uncertain" in warning and "ASR text retained" in warning


def test_long_segment_is_not_arbitrarily_partitioned(monkeypatch):
    value = WhisperForcedAligner()
    monkeypatch.setattr(value, "_load", lambda config: pytest.fail("Oversized context was encoded"))
    segment = SimpleNamespace(start=0, end=31, text="Hello")
    result, warning = value.align(np.zeros(31 * 16000), [segment], config())
    assert result == [segment]
    assert "30-second context" in warning


def test_alignment_work_and_inter_call_deadline_are_bounded(monkeypatch, aligner):
    from mouthkeys import forced_alignment

    value, _, _ = aligner
    monkeypatch.setattr(forced_alignment, "time", SimpleNamespace(monotonic=lambda: 0))
    calls = []
    monkeypatch.setattr(value, "_words", lambda *a: calls.append(True) or [])
    segments = [SimpleNamespace(start=0, end=1, text="Hello") for _ in range(MAX_SEGMENTS + 1)]
    result, warning = value.align(np.zeros(16000), segments, config())
    assert len(calls) == MAX_SEGMENTS
    assert len(result) == len(segments) and "budget reached" in warning
    times = iter([0, 11])
    monkeypatch.setattr(forced_alignment, "time", SimpleNamespace(monotonic=lambda: next(times)))
    calls.clear()
    result, warning = value.align(np.zeros(16000), segments[:1], config())
    assert not calls and "budget reached" in warning


def test_auto_language_is_measured_or_abstained_not_guessed(aligner):
    value, model, calls = aligner
    words = value._words(np.zeros(16000), "Hello", config(language="auto"))
    assert words[0]["word"] == "Hello" and len(calls) == 1
    model.model.detect_language = lambda encoded: [[("<|en|>", 0.3)]]
    with pytest.raises(ValueError, match="language is uncertain"):
        value._words(np.zeros(16000), "Hello", config(language="auto"))


def test_second_model_is_explicit_cpu_int8_and_local_only(monkeypatch):
    import faster_whisper

    from mouthkeys import engines

    calls = []

    def resolve(selected):
        assert selected.model == "cached-aligner"
        assert selected.engine == "whisper"
        assert selected.model_revision == "main"  # Not the unrelated primary ASR revision.
        return "/existing/local/snapshot"

    monkeypatch.setattr(engines, "resolve_whisper", resolve)
    monkeypatch.setattr(
        faster_whisper, "WhisperModel", lambda *a, **kw: calls.append((a, kw)) or object()
    )
    value = WhisperForcedAligner()
    value._load(config(model_revision="primary-nemo-revision"))
    value._load(config())
    assert calls == [
        (
            ("/existing/local/snapshot",),
            {
                "device": "cpu",
                "compute_type": "int8",
                "cpu_threads": 2,
                "num_workers": 1,
                "local_files_only": True,
            },
        )
    ]


def test_transcriber_propagates_forced_words_and_aligner_choice_invalidates_preview(
    monkeypatch, aligner
):
    import faster_whisper.audio

    value, _, _ = aligner
    transcriber = Transcriber()
    primary = SimpleNamespace(
        recognize=lambda *a, **kw: [SimpleNamespace(start=0.5, end=1.5, text="Hello")]
    )
    transcriber.model = primary
    transcriber.aligner = value
    monkeypatch.setattr(transcriber, "load", lambda config: None)
    monkeypatch.setattr(faster_whisper.audio, "decode_audio", lambda *a, **kw: np.zeros(32000))
    cfg = config(engine="onnx")
    assert transcriber.transcribe("dictation.wav", cfg) == "Hello"
    assert transcriber.last_alignment["source"] == "dictation.wav"
    assert transcriber.words[0]["alignment_source"] == "whisper-forced:cached-aligner"
    assert transcriber.model is primary
    assert transcriber._stream_key("dictation.wav", cfg) != transcriber._stream_key(
        "dictation.wav", replace(cfg, pronunciation_aligner_model="another-local-model")
    )


def test_disabling_matching_releases_secondary_model_without_reloading_primary(monkeypatch):
    import faster_whisper

    from mouthkeys import engines

    primary = object()
    monkeypatch.setattr(engines, "resolve_whisper", lambda config: "existing-primary-model")
    monkeypatch.setattr(faster_whisper, "WhisperModel", lambda *a, **kw: primary)
    transcriber = Transcriber()
    transcriber.aligner = WhisperForcedAligner()
    transcriber.aligner.model, transcriber.aligner.key = object(), "cached-secondary"
    cfg = config()
    transcriber.load(cfg)
    assert transcriber.aligner.model is not None
    transcriber.load(replace(cfg, pronunciation_matching="off"))
    assert transcriber.aligner.model is None
    assert transcriber.model is primary


def test_zero_duration_punctuation_is_preserved_without_inventing_extent(aligner):
    value, model, _ = aligner
    model.find_alignment = lambda *a: [
        [
            {"word": " Hello", "start": 0.1, "end": 0.4, "probability": 0.9},
            {"word": ".", "start": 0.4, "end": 0.4, "probability": 0.01},
        ]
    ]
    words = value._words(np.zeros(16000), "Hello.", config())
    assert words[-1]["word"] == "."
    assert words[-1]["start"] == words[-1]["end"] == 0.4


@pytest.mark.parametrize("tokens", [[1000], list(range(441))])
def test_control_tokens_and_long_transcripts_are_rejected_before_encoder_work(
    monkeypatch, aligner, tokens
):
    from faster_whisper import tokenizer

    value, model, _ = aligner
    monkeypatch.setattr(tokenizer.Tokenizer, "encode", lambda *a: tokens)
    model.encode = lambda *a: pytest.fail("Invalid forced transcript was encoded")
    with pytest.raises(ValueError, match="token budget"):
        value._words(np.zeros(16000), "Hello", config())


def test_optional_library_failure_retains_primary_transcript_without_error_text(aligner):
    value, model, _ = aligner

    def fail(*args):
        raise IndexError("private transcript data must not appear in status")

    model.find_alignment = fail
    segment = SimpleNamespace(start=0, end=1, text="Hello")
    segments, warning = value.align(np.zeros(16000), [segment], config())
    assert segments == [segment]
    assert "ASR text retained" in warning
    assert "private transcript" not in warning


def test_repeated_alignment_warnings_do_not_accumulate_during_preview():
    transcriber = Transcriber()
    warnings = iter(["first", "second", "first", "second"])
    transcriber.aligner = SimpleNamespace(align=lambda *a: ([], next(warnings)))
    for _ in range(4):
        transcriber._align_segments(np.zeros(16000), config(), [])
    assert transcriber.alignment_warning == "first; second"
