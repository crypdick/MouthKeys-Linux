import json
import sys
import wave
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mouthkeys import engines, file_transcription
from mouthkeys.config import Config
from mouthkeys.transcribe import Transcriber, download


def test_onnx_only_download_action_can_fetch(monkeypatch):
    calls = []

    def load_model(model, path, **kwargs):
        calls.append(("model", model, path, kwargs))
        path.mkdir(parents=True, exist_ok=True)

    def load_vad(model, path, **kwargs):
        calls.append(("vad", model, path, kwargs))
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setitem(
        sys.modules, "onnx_asr", SimpleNamespace(load_model=load_model, load_vad=load_vad)
    )
    path = download("nemo-parakeet-tdt-0.6b-v3", "onnx", "int8")
    assert path == str(engines.managed_path("nemo-parakeet-tdt-0.6b-v3"))
    assert calls[0][-1] == {"quantization": "int8", "offline": False}
    assert calls[1][1] == "silero"
    assert calls[1][-1] == {"offline": False}


def test_onnx_runtime_offline_and_bounded_vad(monkeypatch, tmp_path):
    calls = []
    config = replace(Config(), engine="onnx", model="nemo-parakeet-tdt-0.6b-v3")
    engines.managed_path(config.model).mkdir(parents=True)
    engines.managed_path("silero").mkdir(parents=True)
    (engines.managed_path(config.model) / engines.READY).touch()
    (engines.managed_path("silero") / engines.READY).touch()

    def recognize(audio, **kwargs):
        calls.append(("recognize", len(audio), kwargs))
        return iter([SimpleNamespace(start=0.2, end=2.5, text=" Words.")])

    def with_vad(vad, **kwargs):
        calls.append(("vad-options", kwargs))
        return SimpleNamespace(recognize=recognize)

    def load_model(model, path, **kwargs):
        calls.append(("model", kwargs))
        return SimpleNamespace(with_vad=with_vad)

    def load_vad(*args, **kwargs):
        calls.append(("vad", kwargs))
        return object()

    monkeypatch.setitem(
        sys.modules, "onnx_asr", SimpleNamespace(load_model=load_model, load_vad=load_vad)
    )
    path = tmp_path / "speech.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(bytes(16000 * 4 * 2))
    transcriber = Transcriber()
    assert transcriber.preview(path, config) == "Words."
    assert transcriber.transcribe(path, config) == "Words."
    assert calls[0][-1]["offline"] is True
    assert calls[1][-1] == {"offline": True}
    assert calls[2][-1] == {"max_speech_duration_s": 25}
    assert len([call for call in calls if call[0] == "recognize"]) == 1


def test_onnx_missing_model_never_fetches(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnx_asr", SimpleNamespace())
    with pytest.raises(RuntimeError, match="download first"):
        engines.load_onnx(replace(Config(), engine="onnx"))


def test_model_delete_rejects_local_paths_and_links(tmp_path):
    local = tmp_path / "user-model"
    local.mkdir()
    (local / "weights").write_text("important")
    with pytest.raises(ValueError, match="outside MouthKeys"):
        engines.delete_model(str(local), "onnx")
    path = engines.managed_path("test")
    path.parent.mkdir(parents=True)
    path.symlink_to(local)
    with pytest.raises(ValueError, match="linked"):
        engines.delete_model("test", "onnx")
    assert (local / "weights").read_text() == "important"


def test_model_delete_only_selected_managed_model():
    selected = engines.managed_path("selected")
    other = engines.managed_path("other")
    selected.mkdir(parents=True)
    other.mkdir()
    engines.delete_model("selected", "onnx")
    assert not selected.exists()
    assert other.is_dir()


def test_file_transcription_exports_segments_and_unicode(monkeypatch, tmp_path):
    from faster_whisper import audio

    monkeypatch.setattr(audio, "decode_audio", lambda *a, **kw: np.zeros(16000 * 3))
    model = SimpleNamespace(
        _segments=lambda *a: [SimpleNamespace(start=0.5, end=2.0, text=" Café au lait.")]
    )
    result = file_transcription.transcribe_file("example.mp3", Config(), transcriber=model)
    assert result["duration"] == 3
    assert result["segments"] == [{"start": 0.5, "end": 2.0, "text": "Café au lait."}]
    assert result["text"] == "Café au lait."
    assert result["speaker_turns"] == []
    output = tmp_path / "export.json"
    file_transcription.export_result(result, output)
    assert json.loads(output.read_text()) == result
    assert output.stat().st_mode & 0o777 == 0o600
    file_transcription.export_result(result, output, "txt")
    assert output.read_text() == "Café au lait.\n"


def test_speaker_labels_use_overlap_and_preserve_unmatched(monkeypatch):
    from faster_whisper import audio

    monkeypatch.setattr(audio, "decode_audio", lambda *a, **kw: np.zeros(16000 * 5))
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": 0},
        {"start": 1.0, "end": 3.0, "speaker": 1},
    ]
    monkeypatch.setattr(file_transcription, "diarize", lambda *a: turns)
    model = SimpleNamespace(
        _segments=lambda *a: [
            SimpleNamespace(start=0.5, end=2.5, text="Second speaker mostly."),
            SimpleNamespace(start=4, end=5, text="No matching speech turn."),
        ]
    )
    config = replace(Config(), file_speaker_labels="on", file_speaker_count=2)
    result = file_transcription.transcribe_file("input.wav", config, transcriber=model)
    assert [s["speaker"] for s in result["segments"]] == [1, None]
    assert result["speaker_turns"] == turns
    assert "Speaker 2: Second speaker mostly." in file_transcription.result_text(result)
    assert "Speaker ?:" in file_transcription.result_text(result)


def test_diarization_requires_explicit_download():
    with pytest.raises(RuntimeError, match="download-speakers first"):
        file_transcription.diarize(np.zeros(16000))


def test_sherpa_local_paths_and_speaker_count(monkeypatch):
    calls = []
    for path in file_transcription.speaker_paths():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")

    def config(**kwargs):
        return SimpleNamespace(**kwargs)

    def make_model(settings):
        calls.append(settings)
        return SimpleNamespace(
            sample_rate=16000,
            process=lambda samples: SimpleNamespace(
                sort_by_start_time=lambda: [SimpleNamespace(start=0, end=1, speaker=3)]
            ),
        )

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(
            OfflineSpeakerDiarizationConfig=lambda **kwargs: SimpleNamespace(
                **kwargs, validate=lambda: True
            ),
            OfflineSpeakerSegmentationModelConfig=config,
            OfflineSpeakerSegmentationPyannoteModelConfig=config,
            SpeakerEmbeddingExtractorConfig=config,
            FastClusteringConfig=config,
            OfflineSpeakerDiarization=make_model,
        ),
    )
    assert file_transcription.diarize(np.zeros(16000), 4) == [
        {"start": 0.0, "end": 1.0, "speaker": 3}
    ]
    assert calls[0].clustering.num_clusters == 4
    assert calls[0].segmentation.pyannote.model.endswith("segmentation.onnx")


def fake_audio_decoder(monkeypatch, input_frames, resampled):
    import av

    calls = []

    class Container:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, **kwargs):
            yield from input_frames

    def open_audio(source, **kwargs):
        calls.append(("open", kwargs))
        return Container()

    def resample(frame):
        calls.append(("resample", frame))
        if frame is None:
            return []
        return resampled(frame)

    monkeypatch.setattr(av, "open", open_audio)
    monkeypatch.setattr(
        av.audio.resampler,
        "AudioResampler",
        lambda **kwargs: SimpleNamespace(resample=resample),
    )
    return calls


def test_api_audio_limit_checks_input_before_resample_or_allocation(monkeypatch, tmp_path):
    path = tmp_path / "compressed.audio"
    path.write_bytes(b"tiny compressed content")
    large_frame = SimpleNamespace(samples=16000 * 601, sample_rate=16000)
    calls = fake_audio_decoder(
        monkeypatch, [large_frame], lambda frame: pytest.fail("Resampling oversized frame")
    )
    with pytest.raises(ValueError, match="exceeds 600 seconds"):
        file_transcription._decode_bounded(path, 600)
    assert calls == [
        (
            "open",
            {"mode": "r", "metadata_errors": "ignore", "options": {"protocol_whitelist": "none"}},
        )
    ]


def test_api_audio_limit_checks_resampler_output_before_numpy(monkeypatch, tmp_path):
    path = tmp_path / "compressed.audio"
    path.write_bytes(b"fake")
    frame = SimpleNamespace(samples=100, sample_rate=16000)
    output = SimpleNamespace(
        samples=16001, to_ndarray=lambda: pytest.fail("Allocated overflowing samples")
    )
    fake_audio_decoder(monkeypatch, [frame], lambda frame: [output])
    with pytest.raises(ValueError, match="exceeds 1 seconds"):
        file_transcription._decode_bounded(path, 1)


def test_bounded_decoder_normalizes_resampled_pcm_and_stops_at_exact_limit(monkeypatch, tmp_path):
    path = tmp_path / "compressed.audio"
    path.write_bytes(b"fake")
    samples = np.array([0, 32767, -32768, 16384], dtype=np.int16)
    frames = [SimpleNamespace(samples=2, sample_rate=16000) for _ in range(2)]
    count = iter(range(2))

    def resampled(frame):
        offset = next(count) * 2
        return [SimpleNamespace(samples=2, to_ndarray=lambda: samples[offset : offset + 2])]

    fake_audio_decoder(monkeypatch, frames, resampled)
    result = file_transcription._decode_bounded(path, 4 / 16000)
    np.testing.assert_allclose(result, samples.astype(np.float32) / 32768.0)
    assert result.dtype == np.float32


def test_bounded_decoder_real_stereo_resampling_without_desktop(tmp_path):
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(bytes(44100 * 4))
    result = file_transcription._decode_bounded(path, 1)
    assert result.shape == (16000,)
    assert not np.any(result)
    with pytest.raises(ValueError, match="exceeds 0.5 seconds"):
        file_transcription._decode_bounded(path, 0.5)
