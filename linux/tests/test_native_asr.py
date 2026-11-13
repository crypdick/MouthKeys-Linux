import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mouthkeys import native_asr
from mouthkeys.config import Config
from mouthkeys.transcribe import Transcriber, download


class Tensor:
    shape = (1, 3)

    def __getitem__(self, index):
        self.index = index
        return self


class Batch(dict):
    def to(self, *args, **kwargs):
        return self


class Processor:
    feature_extractor = SimpleNamespace(sampling_rate=16000)

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def __call__(self, audio, **kwargs):
        self.inputs.append(kwargs)
        return Batch(input_features=Tensor())

    def apply_transcription_request(self, **kwargs):
        self.inputs.append(kwargs)
        return Batch(input_features=Tensor(), input_ids=Tensor())

    def decode(self, output, **kwargs):
        self.outputs.append((output, kwargs))
        return [" Native result."]


class Model:
    device = "cpu"
    dtype = "float32"

    def __init__(self):
        self.generations = []

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generations.append(kwargs)
        return SimpleNamespace(sequences=Tensor())


def torch_fake(cuda=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        inference_mode=nullcontext,
    )


def backend_fixture(monkeypatch, tmp_path, family):
    path = tmp_path / family
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": family}))
    config = replace(Config(), engine="transformers", model=str(path))
    processor, model, calls = Processor(), Model(), []

    def load_processor(path, **kwargs):
        calls.append(("processor", path, kwargs))
        return processor

    def load_model(path, **kwargs):
        calls.append(("model", path, kwargs))
        return model

    monkeypatch.setitem(sys.modules, "torch", torch_fake())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
            **{
                name: SimpleNamespace(from_pretrained=load_model)
                for name in set(native_asr.FAMILIES.values())
            },
        ),
    )
    monkeypatch.setattr(native_asr, "speech_spans", lambda audio: [{"start": 0, "end": len(audio)}])
    return native_asr.TransformersBackend(config), config, processor, model, calls


@pytest.mark.parametrize("family", list(native_asr.FAMILIES))
def test_native_models_are_local_only_builtin_loaders(monkeypatch, tmp_path, family):
    backend, config, processor, model, calls = backend_fixture(monkeypatch, tmp_path, family)
    assert all(call[2]["local_files_only"] is True for call in calls)
    assert all(call[2]["trust_remote_code"] is False for call in calls)
    segments = backend.segments(np.zeros(16000), config)
    assert [(s.start, s.end, s.text) for s in segments] == [(0.0, 1.0, "Native result.")]
    assert model.generations[0]["max_new_tokens"] == config.asr_max_tokens
    if family == "qwen3_asr":
        assert processor.outputs[0][1] == {"return_format": "transcription_only"}
        assert processor.outputs[0][0].index[1].start == 3
    else:
        assert processor.outputs[0][1] == {"skip_special_tokens": True}


def test_qwen_uses_native_hotword_context(monkeypatch, tmp_path):
    from mouthkeys.vocabulary import VocabularyStore

    VocabularyStore.save(["MouthKeys", "Qwen"])
    backend, config, processor, *_ = backend_fixture(monkeypatch, tmp_path, "qwen3_asr")
    backend.segments(np.zeros(16000), replace(config, vocabulary_boosting="on"))
    assert "MouthKeys" in processor.inputs[0]["prompt"]
    assert processor.inputs[0]["language"] == "en"


def test_native_language_contracts_not_silently_ignored(monkeypatch, tmp_path):
    backend, config, processor, *_ = backend_fixture(monkeypatch, tmp_path, "nemotron3_5_asr")
    backend.segments(np.zeros(16000), replace(config, language="fr"))
    assert processor.inputs[0]["language"] == "fr-FR"
    backend, config, processor, *_ = backend_fixture(monkeypatch, tmp_path, "cohere_asr")
    with pytest.raises(ValueError, match="explicit language"):
        backend.segments(np.zeros(16000), replace(config, language="auto"))
    backend, config, processor, *_ = backend_fixture(
        monkeypatch, tmp_path, "nemotron_asr_streaming"
    )
    with pytest.raises(ValueError, match="English only"):
        backend.segments(np.zeros(16000), replace(config, language="fr"))


def test_download_explicit_revision_filters_out_remote_python(monkeypatch):
    import huggingface_hub

    calls = []

    def snapshot(model, **kwargs):
        calls.append((model, kwargs))
        kwargs["local_dir"].mkdir(parents=True)
        (kwargs["local_dir"] / "model.safetensors").write_bytes(b"fake")
        (kwargs["local_dir"] / "config.json").write_text('{"model_type":"cohere_asr"}')
        (kwargs["local_dir"] / "processor_config.json").write_text("{}")
        (kwargs["local_dir"] / "tokenizer.json").write_text("{}")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    result = download("CohereLabs/cohere-transcribe-03-2026", "transformers")
    assert calls[0][1]["revision"] == "refs/pr/6"
    assert "*.py" not in calls[0][1]["allow_patterns"]
    assert "*.nemo" not in calls[0][1]["allow_patterns"]
    assert (native_asr.Path(result) / native_asr.READY).is_file()


def test_missing_model_never_imports_or_downloads_native_packages(monkeypatch):
    config = replace(Config(), engine="transformers", model="Qwen/Qwen3-ASR-0.6B-hf")
    monkeypatch.setattr(
        native_asr.importlib, "import_module", lambda *a: pytest.fail("Native import attempted")
    )
    with pytest.raises(RuntimeError, match="explicitly download"):
        native_asr.TransformersBackend(config)


def test_native_precision_cuda_failures_are_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", torch_fake())
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        native_asr.torch_runtime(replace(Config(), engine="transformers", device="cuda"))
    with pytest.raises(ValueError, match="compute type"):
        native_asr.torch_runtime(replace(Config(), engine="transformers", compute_type="int8"))


class Features:
    def __init__(self, values):
        self.values = list(values)

    def size(self, dimension):
        return len(self.values)

    def to(self, *args, **kwargs):
        return self


class StreamingBuffer:
    def __init__(self, model, **kwargs):
        self.buffer_idx = 0
        self.step = 0
        self.streaming_cfg = SimpleNamespace(chunk_size=4, drop_extra_pre_encoded=1)
        self.sampling_frames = 2

    def preprocess_audio(self, samples):
        return Features(samples), np.array([len(samples)])

    def __iter__(self):
        while self.buffer_idx < self.buffer.size(-1):
            if self.buffer.size(-1) - self.buffer_idx < self.sampling_frames:
                return
            chunk = Features(self.buffer.values[self.buffer_idx : self.buffer_idx + 4])
            self.buffer_idx += 4
            self.step += 1
            yield chunk, np.array([len(chunk.values)])


def streaming_backend(monkeypatch):
    backend = native_asr.NemoBackend.__new__(native_asr.NemoBackend)
    backend._base_decoding = None
    backend._vocabulary_state = None
    calls = []

    def stream_step(**kwargs):
        calls.append(kwargs)
        n = len(calls)
        return (
            f"pred{n}",
            [SimpleNamespace(text=f"Words {n}<EOU>")],
            f"channel{n}",
            f"time{n}",
            f"length{n}",
            f"hyp{n}",
        )

    backend.model = SimpleNamespace(
        dtype="float32",
        encoder=SimpleNamespace(
            get_initial_cache_state=lambda **kwargs: ("initial", "initial", "initial")
        ),
        conformer_stream_step=stream_step,
    )
    backend.torch = torch_fake()
    backend.torch.nn = SimpleNamespace(
        functional=SimpleNamespace(
            pad=lambda features, pad: Features([*features.values, *([0] * pad[1])])
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "nemo.collections.asr.parts.utils.streaming_utils",
        SimpleNamespace(CacheAwareStreamingAudioBuffer=StreamingBuffer),
    )
    backend.reset_stream()
    return backend, calls


def test_cache_aware_stream_encodes_new_chunks_and_final_tail_once(monkeypatch):
    backend, calls = streaming_backend(monkeypatch)
    config = replace(Config(), engine="nemo")
    assert backend.stream(np.arange(12), config, "recording") == "Words 2"
    assert backend.stream(np.arange(18), config, "recording") == "Words 3"
    assert backend.stream(np.arange(18), config, "recording", final=True) == "Words 5"
    assert [value for call in calls for value in call["processed_signal"].values] == list(range(18))
    assert calls[0]["cache_last_channel"] == "initial"
    assert calls[1]["cache_last_channel"] == "channel1"
    assert calls[1]["previous_hypotheses"] == "hyp1"
    assert calls[-1]["keep_all_outputs"] is True
    assert not any(call["keep_all_outputs"] for call in calls[:-1])
    assert backend.stream_key is None


def test_native_cancel_new_session_and_reprocess_start_with_fresh_caches(monkeypatch):
    backend, calls = streaming_backend(monkeypatch)
    config = replace(Config(), engine="nemo")
    backend.stream(np.arange(8), config, "first")
    backend.reset_stream()
    backend.stream(np.arange(100, 108), config, "second")
    assert calls[-1]["cache_last_channel"] == "initial"
    assert calls[-1]["previous_hypotheses"] is None
    assert calls[-1]["processed_signal"].values == [100, 101, 102, 103]
    backend.stream(np.arange(8), config, "reprocess", final=True)
    assert calls[-2]["cache_last_channel"] == "initial"
    assert backend.stream_key is None


def test_native_final_short_tail_is_padded_not_dropped(monkeypatch):
    backend, calls = streaming_backend(monkeypatch)
    backend.stream(np.array([42]), replace(Config(), engine="nemo"), "short", final=True)
    assert calls[0]["processed_signal"].values == [42, 0]
    assert calls[0]["keep_all_outputs"] is True


def test_transcriber_routes_native_live_and_final_to_same_session(monkeypatch, tmp_path):
    from mouthkeys import audio

    calls = []
    model = SimpleNamespace(stream=lambda *a, **kw: calls.append((a, kw)) or "native")
    transcriber = Transcriber()
    monkeypatch.setattr(transcriber, "load", lambda config: setattr(transcriber, "model", model))
    monkeypatch.setattr(audio, "read_audio_from", lambda path: np.zeros(16000))
    config = replace(Config(), engine="nemo", native_streaming="on")
    assert transcriber.preview(tmp_path / "audio.wav", config) == "native"
    assert transcriber.transcribe(tmp_path / "audio.wav", config) == "native"
    assert calls[0][0][2] == calls[1][0][2]
    assert calls[1][1] == {"final": True}


def test_nemo_uses_real_weighted_graph_scores_and_restores_disabled_decoder(monkeypatch, tmp_path):
    from mouthkeys.vocabulary import VocabularyStore

    VocabularyStore.save_entries(
        [
            {"text": "MouthKeys", "weight": 1, "aliases": []},
            {"text": "Linux", "weight": 5, "aliases": []},
        ]
    )
    monkeypatch.setenv("MOUTHKEYS_NATIVE_STATE", str(tmp_path))
    calls = []

    class Graph:
        def __init__(self, **kwargs):
            pass

        def build(self, **kwargs):
            calls.append(("graph", kwargs))

    class Tree:
        @classmethod
        def from_context_graph(cls, **kwargs):
            calls.append(("tree", kwargs))
            return cls()

        def save_to(self, path):
            native_asr.Path(path).write_bytes(b"fake tree")

    for name in (
        "nemo",
        "nemo.collections",
        "nemo.collections.asr",
        "nemo.collections.asr.parts",
        "nemo.collections.asr.parts.context_biasing",
        "nemo.collections.asr.parts.context_biasing.context_graph_universal",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["nemo.collections.asr.parts.context_biasing"].GPUBoostingTreeModel = Tree
    sys.modules[
        "nemo.collections.asr.parts.context_biasing.context_graph_universal"
    ].ContextGraph = Graph
    monkeypatch.setitem(
        sys.modules,
        "omegaconf",
        SimpleNamespace(
            OmegaConf=SimpleNamespace(
                to_container=lambda value, **kw: value,
                create=lambda value: value,
            )
        ),
    )
    backend = native_asr.NemoBackend.__new__(native_asr.NemoBackend)
    backend._base_decoding = None
    backend._vocabulary_state = None
    backend.model = SimpleNamespace(
        cfg=SimpleNamespace(decoding={"strategy": "greedy_batch", "greedy": {"max_symbols": 10}}),
        tokenizer=SimpleNamespace(text_to_ids=lambda word: [len(word)], vocab_size=100),
        change_decoding_strategy=lambda value: calls.append(("strategy", value)),
    )
    config = replace(Config(), engine="nemo", vocabulary_boosting="on")
    backend._vocabulary(config)
    assert dict(zip(calls[0][1]["phrases"], calls[0][1]["scores"], strict=True)) == {
        "MouthKeys": 1.0,
        "Linux": 5.0,
    }
    assert dict(zip(calls[0][1]["phrases"], calls[0][1]["token_ids"], strict=True)) == {
        "MouthKeys": [9],
        "Linux": [5],
    }
    assert calls[1][1]["use_triton"] is False
    assert calls[2][1]["greedy"]["boosting_tree_alpha"] == 1.0
    assert native_asr.Path(calls[2][1]["greedy"]["boosting_tree"]["model_path"]).is_file()
    before = len(calls)
    backend._vocabulary(config)
    assert len(calls) == before
    backend._vocabulary(replace(config, vocabulary_boosting="off"))
    assert calls[-1][1] == {"strategy": "greedy_batch", "greedy": {"max_symbols": 10}}


def test_download_artifact_validation_rejects_missing_companions_before_ready(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"fake")
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_asr"}')
    with pytest.raises(ValueError, match="processor"):
        native_asr.validate_artifacts(tmp_path, "transformers")
    assert not (tmp_path / native_asr.READY).exists()
    (tmp_path / "processor_config.json").write_text("{}")
    with pytest.raises(ValueError, match="tokenizer"):
        native_asr.validate_artifacts(tmp_path, "transformers")


def test_machine_progress_reports_actual_file_counts():
    from mouthkeys.model_progress import progress_tqdm

    events = []
    progress = progress_tqdm(events.append)(total=3, unit="file", desc="Fetching")
    progress.update(1)
    progress.update(2)
    progress.close()
    assert [event["completed"] for event in events] == [0, 1, 3]
    assert all(event["total"] == 3 and event["unit"] == "file" for event in events)


@pytest.mark.parametrize("bos", [[], [99]])
def test_cohere_conditioning_prompt_is_not_transcript(bos):
    prompt = np.array([[4, 5, 6]])
    generated = np.array([[*bos, 4, 5, 6, 10, 11]])
    assert native_asr._strip_decoder_prompt(generated, prompt).tolist() == [[10, 11]]


def test_cohere_does_not_guess_unexpected_prompt_offset():
    with pytest.raises(ValueError, match="conditioning prefix"):
        native_asr._strip_decoder_prompt(np.array([[8, 8, 4, 5, 6, 10]]), np.array([[4, 5, 6]]))


def test_native_word_frame_offsets_are_scaled_without_mutating_hypothesis():
    backend = object.__new__(native_asr.NemoBackend)
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
    backend, calls = streaming_backend(monkeypatch)
    monkeypatch.setattr(backend, "_vocabulary", lambda config: None)
    backend.model.encoder.subsampling_factor = 8
    backend.model.cfg = SimpleNamespace(preprocessor=SimpleNamespace(window_stride=0.01))
    hypotheses = []

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
        return None, [hypothesis], "channel", "time", "length", [hypothesis]

    backend.model.conformer_stream_step = step
    config = replace(Config(), engine="nemo", pronunciation_matching="on")
    backend.stream(np.arange(12), config, "session")
    backend.stream(np.arange(18), config, "session", final=True)
    assert len(hypotheses) > 1
    assert all(hyp.timestamp == [0] for hyp in hypotheses)
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
    "text,tokens",
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
    assert native_asr._native_token_words(text, tokens, duration=1) == []


def test_native_token_timing_preserves_unicode_punctuation_and_clips_only_waveform_edge():
    words = native_asr._native_token_words(
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
