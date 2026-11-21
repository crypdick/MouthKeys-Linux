from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mouthkeys import native_asr, nemo_asr, transformers_asr
from mouthkeys.config import Config

from .native_asr_fakes import Tensor, backend_fixture, streaming_backend


def test_native_word_frame_offsets_are_scaled_without_mutating_hypothesis():
    backend = object.__new__(nemo_asr.NemoBackend)
    backend.model = SimpleNamespace(
        encoder=SimpleNamespace(subsampling_factor=8),
        cfg=SimpleNamespace(preprocessor=SimpleNamespace(window_stride=0.01)),
    )
    timestamp = {
        "timestep": [3, 4],
        "word": [{"word": "hello", "start_offset": 3, "end_offset": 5}],
    }
    hypothesis = SimpleNamespace(timestamp=timestamp)
    assert backend._word_times(hypothesis, offset=10) == [
        {"word": "hello", "start": 10.24, "end": 10.4}
    ]
    assert backend._word_times(hypothesis, duration=0.35)[0]["end"] == 0.35
    assert hypothesis.timestamp is timestamp
    assert timestamp["timestep"] == [3, 4]


def test_cache_aware_word_rendering_restores_mergeable_timestep_list(monkeypatch):
    backend, _ = streaming_backend(monkeypatch)
    monkeypatch.setattr(backend, "_vocabulary", lambda config: None)
    backend.model.encoder.subsampling_factor = 8
    backend.model.cfg = SimpleNamespace(preprocessor=SimpleNamespace(window_stride=0.01))
    hypotheses = []
    untouched = SimpleNamespace(timestamp=[7])

    def step(**kwargs):
        previous = kwargs["previous_hypotheses"]
        if previous is not None:
            assert isinstance(previous[0].timestamp, list)
        hypothesis = SimpleNamespace(
            text="hello",
            timestamp={
                "timestep": [0],
                "word": [{"word": "hello", "start_offset": 0, "end_offset": 1}],
            },
        )
        hypotheses.append(hypothesis)
        return None, [hypothesis], "channel", "time", "length", [hypothesis, untouched]

    backend.model.conformer_stream_step = step
    config = replace(Config(), engine="nemo", pronunciation_matching="on")
    backend.stream(np.arange(12), config, "session")
    backend.stream(np.arange(18), config, "session", final=True)
    assert len(hypotheses) > 1
    assert all(hyp.timestamp == [0] for hyp in hypotheses)
    assert untouched.timestamp == [7]
    assert backend.words[0]["word"] == "hello"
    assert backend.stream_key is None


def test_prompted_native_file_strips_decoder_tags_from_text_and_word_times(monkeypatch):
    backend, _ = streaming_backend(monkeypatch)
    backend.model.encoder.subsampling_factor = 8
    backend.model.cfg = SimpleNamespace(preprocessor=SimpleNamespace(window_stride=0.01))
    calls = []
    backend.model.set_inference_prompt = lambda language: calls.append(("prompt", language))
    backend.model.decoding = SimpleNamespace(
        set_strip_lang_tags=lambda enabled: calls.append(("strip_tags", enabled))
    )
    hypothesis = SimpleNamespace(
        text="Hello. <en-US>",
        timestamp={
            "word": [
                {"word": "Hello.", "start_offset": 1, "end_offset": 4},
                {"word": "<en-US>", "start_offset": 4, "end_offset": 5},
            ]
        },
    )
    backend.model.transcribe = lambda **kwargs: [hypothesis]
    monkeypatch.setattr(backend, "_vocabulary", lambda config: calls.append(("configure", True)))
    config = replace(Config(), engine="nemo", pronunciation_matching="on", language="en")
    segments = backend.segments(np.zeros(16000), config, presegmented=True)
    assert calls == [("configure", True), ("prompt", "en-US"), ("strip_tags", True)]
    assert segments[0].text == "Hello."
    assert segments[0].words == [{"word": "Hello.", "start": 0.08, "end": 0.32}]


@pytest.mark.parametrize("family", ["nemotron3_5_asr", "nemotron_asr_streaming"])
def test_transformers_nemotron_preserves_native_durations_and_groups_word_intervals(
    monkeypatch, tmp_path, family
):
    backend, config, processor, model, *_ = backend_fixture(monkeypatch, tmp_path, family)
    durations = object()
    generated = Tensor()
    monkeypatch.setattr(
        model, "generate", lambda **kw: SimpleNamespace(sequences=generated, durations=durations)
    )
    calls = []

    def decode(sequence, **kwargs):
        calls.append((sequence, kwargs))
        return ["Native result."], [
            [
                {"token": "Nat", "start": 0.08, "end": 0.16},
                {"token": "ive", "start": 0.16, "end": 0.24},
                {"token": " result", "start": 0.32, "end": 0.4},
                {"token": ".", "start": 0.4, "end": 0.48},
            ]
        ]

    monkeypatch.setattr(processor, "decode", decode)
    monkeypatch.setattr(native_asr, "speech_spans", lambda audio: [{"start": 8000, "end": 16000}])
    segments = backend.segments(np.zeros(16000), replace(config, pronunciation_matching="on"))
    assert calls == [(generated, {"skip_special_tokens": True, "durations": durations})]
    assert segments[0].text == "Native result."
    assert segments[0].words == [
        {"word": "Native", "start": 0.58, "end": 0.74},
        {"word": "result.", "start": 0.8200000000000001, "end": 0.98},
    ]
    assert backend.words == segments[0].words


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("First second", [{"token": "First second", "start": 0.1, "end": 0.2}]),
        ("First", [{"token": "Changed", "start": 0.1, "end": 0.2}]),
        ("First", [{"token": "First", "start": -0.1, "end": 0.2}]),
        ("First", [{"token": "First", "start": 0.2, "end": 0.1}]),
        ("First", [{"token": "First", "start": float("nan"), "end": 0.2}]),
        ("First", [{"token": "First", "start": True, "end": 0.2}]),
        ("First", [{"token": "First", "start": 1.1, "end": 1.2}]),
        ("First", [{"token": "First", "start": 0.1, "end": 100}]),
        (
            "First second",
            [
                {"token": "First", "start": 0.2, "end": 0.3},
                {"token": " second", "start": 0.1, "end": 0.4},
            ],
        ),
    ],
)
def test_native_token_timing_abstains_on_ambiguous_or_mismatched_metadata(text, tokens):
    assert transformers_asr.native_token_words(text, tokens, duration=1) == []


def test_native_token_timing_preserves_unicode_punctuation_and_clips_only_waveform_edge():
    words = transformers_asr.native_token_words(
        "Café! next",
        [
            {"token": "  Caf", "start": 0.1, "end": 0.2},
            {"token": "é!", "start": 0.2, "end": 0.3},
            {"token": " next", "start": 0.92, "end": 1.04},
            {"token": "  ", "start": 0.92, "end": 1.04},
        ],
        duration=1,
        offset=4,
    )
    assert words == [
        {"word": "Café!", "start": 4.1, "end": 4.3},
        {"word": "next", "start": 4.92, "end": 5},
    ]


def test_transformers_word_timing_opt_out_clears_previous_metadata(monkeypatch, tmp_path):
    backend, config, processor, model, *_ = backend_fixture(
        monkeypatch, tmp_path, "nemotron_asr_streaming"
    )
    monkeypatch.setattr(
        model, "generate", lambda **kw: SimpleNamespace(sequences=Tensor(), durations=object())
    )
    calls = []

    def decode(sequence, **kwargs):
        calls.append(kwargs)
        if "durations" in kwargs:
            return ["Native"], [[{"token": "Native", "start": 0.08, "end": 0.16}]]
        return ["Native"]

    monkeypatch.setattr(processor, "decode", decode)
    backend.segments(np.zeros(16000), replace(config, pronunciation_matching="on"))
    assert backend.words == [{"word": "Native", "start": 0.08, "end": 0.16}]
    segments = backend.segments(np.zeros(16000), replace(config, pronunciation_matching="off"))
    assert calls[-1] == {"skip_special_tokens": True}
    assert backend.words == segments[0].words == []
