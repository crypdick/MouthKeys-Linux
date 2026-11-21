"""Optional ASR package and cache fakes: no weight/model loading."""

import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from mouthkeys import native_asr
from mouthkeys.config import Config


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


class Features:
    def __init__(self, values):
        self.values = list(values)

    def size(self, _dimension):
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
