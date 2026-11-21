import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mouthkeys import native_asr, nemo_asr, transformers_asr
from mouthkeys.config import Config

from .native_asr_fakes import (
    Batch,
    Features,
    StreamingBuffer,
    backend_fixture,
    streaming_backend,
    torch_fake,
)


@pytest.mark.parametrize(
    ("compute", "latency_api", "expected"),
    [
        ("auto", True, None),
        ("default", True, None),
        ("float16", True, "requires float32"),
        ("auto", False, "does not support configurable streaming latency"),
    ],
)
def test_nemo_loader_resolves_local_archive_and_checks_real_streaming_capabilities(
    monkeypatch, tmp_path, compute, latency_api, expected
):
    archive = tmp_path / "model.nemo"
    archive.touch()
    config = replace(
        Config(), model=str(tmp_path), engine="nemo", compute_type=compute, streaming_latency="80"
    )
    calls = []

    class Model:
        def __init__(self):
            self.encoder = SimpleNamespace(att_context_size=[56, 1])
            if latency_api:
                self.encoder.set_default_att_context_size = lambda **kw: calls.append(kw)

        def conformer_stream_step(self):
            pytest.fail("Loader must not start inference")

        def to(self, *args, **kwargs):
            calls.append((args, kwargs))
            return self

        def eval(self):
            return self

    model = Model()
    monkeypatch.setattr(native_asr, "validate_artifacts", lambda *args: calls.append(args))
    monkeypatch.setitem(sys.modules, "torch", torch_fake())

    def restore(path, *, map_location):
        calls.append((path, map_location))
        return model

    loader = SimpleNamespace(ASRModel=SimpleNamespace(restore_from=restore))
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", loader)
    if expected is not None:
        with pytest.raises(ValueError, match=expected):
            nemo_asr.NemoBackend(config)
    else:
        backend = nemo_asr.NemoBackend(config)
        assert backend.supports_native_streaming
        assert backend.stream_key is None
        assert calls[-1] == {"att_context_size": [56, 0]}
        assert calls[-2] == ((), {"dtype": "float32"})
    assert calls[:3] == [(archive, "nemo"), (str(archive), "cpu"), (("cpu",), {})]


def test_nemo_loader_rejects_ambiguous_artifact_or_missing_package(monkeypatch, tmp_path):
    config = replace(Config(), model=str(tmp_path), engine="nemo")
    for name in ("first.nemo", "second.nemo"):
        (tmp_path / name).touch()
    with pytest.raises(ValueError, match="exactly one"):
        nemo_asr.NemoBackend(config)

    (tmp_path / "second.nemo").unlink()
    monkeypatch.setattr(native_asr, "validate_artifacts", lambda *_: None)
    monkeypatch.setitem(sys.modules, "torch", torch_fake())
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", None)
    with pytest.raises(RuntimeError, match="Install the nemo optional"):
        nemo_asr.NemoBackend(config)


def test_transformers_loader_rejects_bad_architecture_sample_rate_or_missing_extra(
    monkeypatch, tmp_path
):
    _, config, processor, _, calls = backend_fixture(monkeypatch, tmp_path, "qwen3_asr")
    processor.feature_extractor = SimpleNamespace(sampling_rate=48000)
    with pytest.raises(ValueError, match="16 kHz"):
        transformers_asr.TransformersBackend(config)

    (tmp_path / "qwen3_asr" / "config.json").write_text('{"model_type":"not_builtin"}')
    count = len(calls)
    with pytest.raises(ValueError, match="Use a builtin"):
        transformers_asr.TransformersBackend(config)
    assert len(calls) == count
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(RuntimeError, match="Install the transformers optional"):
        transformers_asr.TransformersBackend(config)


def test_transformers_cohere_removes_verified_conditioning_and_drops_empty_segments(
    monkeypatch, tmp_path
):
    backend, config, processor, model, _ = backend_fixture(monkeypatch, tmp_path, "cohere_asr")
    prompt = np.array([[4, 5, 6]])
    monkeypatch.setattr(
        type(processor), "__call__", lambda *a, **kw: Batch(decoder_input_ids=prompt)
    )
    monkeypatch.setattr(model, "generate", lambda **kw: np.array([[99, 4, 5, 6, 10, 11]]))

    def decode(generated, **options):
        assert generated.tolist() == [[10, 11]]
        assert options == {"skip_special_tokens": True}
        return ["   "]

    monkeypatch.setattr(processor, "decode", decode)
    assert backend.segments(np.zeros(16000), config, presegmented=True) == []
    assert backend.words == []


@pytest.mark.parametrize(
    ("tokens", "duration"),
    [
        (None, 1),
        ([42], 1),
        ([{"token": "word", "start": 1.04, "end": 1.07}], 1),
        ([{"token": "word", "start": 0.1, "end": 0.2}], float("nan")),
        ([{"token": "word", "start": 0, "end": True}], 1),
        ([{"token": "word", "start": 0, "end": float("inf")}], 1),
    ],
)
def test_invalid_token_alignment_never_claims_native_word_times(tokens, duration):
    assert transformers_asr.native_token_words("word", tokens, duration) == []


def test_nemo_segments_refuses_active_native_recording_and_handles_empty_tuple_output(monkeypatch):
    backend, _ = streaming_backend(monkeypatch)
    config = replace(Config(), engine="nemo")
    backend.stream(np.arange(8), config, "active")
    with pytest.raises(RuntimeError, match="Native recording is active"):
        backend.segments(np.zeros(16000), config, presegmented=True)
    backend.reset_stream()

    monkeypatch.setattr(native_asr, "speech_spans", lambda _: [{"start": 8000, "end": 16000}])
    backend.model.transcribe = lambda **kw: ([" <EOU> "], "metadata")
    assert backend.segments(np.zeros(16000), config) == []
    backend.model.transcribe = lambda **kw: (["hello"], "metadata")
    segments = backend.segments(np.zeros(16000), config)
    assert [(s.start, s.end, s.text, s.words) for s in segments] == [(0.5, 1.0, "hello", [])]


def test_nemo_timestamps_ignore_empty_words_and_require_finite_frames(monkeypatch):
    backend, _ = streaming_backend(monkeypatch)
    backend.model.encoder.subsampling_factor = 8
    backend.model.cfg = SimpleNamespace(preprocessor=SimpleNamespace(window_stride=0.01))
    assert backend._word_times(SimpleNamespace(timestamp=None)) == []
    hypothesis = SimpleNamespace(
        timestamp={
            "word": [
                {"word": "<EOU>", "start_offset": 0, "end_offset": 1},
                {"word": "NaN", "start_offset": float("nan"), "end_offset": 1},
                {"word": "reversed", "start_offset": 3, "end_offset": 1},
                {"word": "valid", "start_offset": 1, "end_offset": 2},
            ]
        }
    )
    assert backend._word_times(hypothesis) == [{"word": "valid", "start": 0.08, "end": 0.16}]


def test_nemo_timing_config_reenables_base_decoding_when_disabled(monkeypatch):
    backend, _ = streaming_backend(monkeypatch)
    base = {"strategy": "greedy", "compute_timestamps": False}
    backend.model.cfg = SimpleNamespace(decoding=base)
    decodes = []
    backend.model.change_decoding_strategy = decodes.append
    omega = SimpleNamespace(
        to_container=lambda cfg, **kw: cfg,
        create=lambda cfg: cfg,
    )
    monkeypatch.setitem(sys.modules, "omegaconf", SimpleNamespace(OmegaConf=omega))
    config = replace(Config(), engine="nemo", pronunciation_matching="on")
    backend._vocabulary(config)
    backend._vocabulary(config)
    backend._vocabulary(replace(config, pronunciation_matching="off"))
    assert decodes == [
        {
            **base,
            "compute_timestamps": True,
            "preserve_alignments": False,
            "rnnt_timestamp_type": "word",
        },
        base,
    ]
    assert base["compute_timestamps"] is False

    backend._base_decoding = None
    backend.model.cfg.decoding = []
    with pytest.raises(ValueError, match="must be a mapping"):
        backend._vocabulary(config)


def test_nemo_buffer_uses_first_and_later_chunk_sizes_without_reencoding(monkeypatch):
    backend, calls = streaming_backend(monkeypatch)
    selected = []

    class VariableBuffer(StreamingBuffer):
        def __init__(self, model, **options):
            super().__init__(model, **options)
            selected.append(options)
            self.sampling_frames = [2, 8]
            self.streaming_cfg.chunk_size = [4, 2]

        def __iter__(self):
            while self.buffer_idx < self.buffer.size(-1):
                size = self.streaming_cfg.chunk_size[int(self.buffer_idx > 0)]
                chunk = self.buffer.values[self.buffer_idx : self.buffer_idx + size]
                self.buffer_idx += len(chunk)
                self.step += 1
                yield Features(chunk), np.array([len(chunk)])

    module = sys.modules["nemo.collections.asr.parts.utils.streaming_utils"]
    module.CacheAwareStreamingAudioBuffer = VariableBuffer
    backend.model.cfg = SimpleNamespace(preprocessor=SimpleNamespace(normalize="per_feature"))
    config = replace(Config(), engine="nemo")
    backend.stream(np.arange(10), config, "active")
    with pytest.raises(ValueError, match="Recording shrank"):
        backend.stream(np.arange(9), config, "active")
    assert len(calls) == 2
    backend.stream(np.arange(12), config, "active", final=True)
    assert selected == [{"online_normalization": True}]
    assert [call["processed_signal"].values for call in calls] == [
        [0, 1, 2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
        [10, 11],
        [0, 0],
    ]
    assert calls[-1]["keep_all_outputs"] is True
    assert [call["drop_extra_pre_encoded"] for call in calls] == [0, 1, 1, 1, 1, 1]


def test_nemo_empty_iterator_keeps_last_useful_decode(monkeypatch):
    backend, calls = streaming_backend(monkeypatch)

    class EmptyBuffer(StreamingBuffer):
        def __iter__(self):
            return iter(())

    module = sys.modules["nemo.collections.asr.parts.utils.streaming_utils"]
    module.CacheAwareStreamingAudioBuffer = EmptyBuffer
    config = replace(Config(), engine="nemo")
    assert backend.stream(np.array([]), config, "new") == ""
    assert backend.stream(np.arange(8), config, "new") == ""
    assert not calls
    assert backend.stream(np.arange(8), config, "new", final=True) == ""
    assert backend.stream_key is None


def test_transformers_skips_whitespace_only_tokens_without_borrowing_time():
    words = transformers_asr.native_token_words(
        "Word",
        [
            {"token": " ", "start": 0.0, "end": 0.05},
            {"token": "Word", "start": 0.1, "end": 0.2},
        ],
        duration=1,
    )
    assert words == [{"word": "Word", "start": 0.1, "end": 0.2}]
