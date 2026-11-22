"""Headless file transcription and explicit local speaker-model setup."""

import io
import json
import math
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
from faster_whisper import audio

from mouthkeys import file_speakers, file_transcription
from mouthkeys.config import Config


@pytest.mark.parametrize("mode", ["turns", "overlap"])
def test_nonfinite_and_out_of_file_turns_do_not_create_speaker_labels(monkeypatch, mode):
    decoded = []
    monkeypatch.setattr(audio, "decode_audio", lambda *args, **kwargs: np.zeros(16000))
    monkeypatch.setattr(
        file_transcription,
        "diarize",
        lambda *_: [
            {"start": float("nan"), "end": 0.75, "speaker": 0},
            {"start": 0.4, "end": float("inf"), "speaker": 1},
            {"start": 1, "end": 2, "speaker": 2},
            {"start": 0.7, "end": 0.3, "speaker": 3},
        ],
    )

    def segments(samples, _config):
        decoded.append(len(samples))
        return [SimpleNamespace(start=0, end=1, text="Keep the words")]

    result = file_transcription.transcribe_file(
        "input.wav",
        Config(file_speaker_labels="on", file_diarization_mode=mode),
        transcriber=SimpleNamespace(_segments=segments),
    )
    assert result["speaker_turns"] == []
    assert result["diarization_mode"] is None
    assert result["text"] == "Keep the words"
    assert decoded == [16000]
    json.dumps(result, allow_nan=False)


def test_file_segments_ignore_nonfinite_or_inverted_timings(monkeypatch):
    monkeypatch.setattr(audio, "decode_audio", lambda *args, **kwargs: np.zeros(16000))
    pieces = [
        SimpleNamespace(start=float("nan"), end=0.25, text="Invalid start"),
        SimpleNamespace(start=0.1, end=float("inf"), text="Invalid end"),
        SimpleNamespace(start=0.75, end=0.5, text="Inverted"),
        SimpleNamespace(start=0.4, end=0.5, text="   "),
        SimpleNamespace(start=0.5, end=1, text="Keep this", avg_logprob=float("nan")),
    ]
    result = file_transcription.transcribe_file(
        "input.wav", Config(), transcriber=SimpleNamespace(_segments=lambda *_: pieces)
    )
    assert result["segments"] == [{"start": 0.5, "end": 1.0, "text": "Keep this"}]
    assert result["confidence"] is None
    json.dumps(result, allow_nan=False)


def test_confidence_uses_only_finite_retained_segment_scores(monkeypatch):
    monkeypatch.setattr(audio, "decode_audio", lambda *args, **kwargs: np.zeros(16000))
    pieces = [
        SimpleNamespace(start=0, end=0.4, text="Half", avg_logprob=-math.log(2)),
        SimpleNamespace(start=0.4, end=0.7, text="Full", avg_logprob=0.2),
        SimpleNamespace(start=0.7, end=1, text="Unknown", avg_logprob=float("inf")),
        SimpleNamespace(start=1, end=1, text="", avg_logprob=-10),
    ]
    result = file_transcription.transcribe_file(
        "input.wav", Config(), transcriber=SimpleNamespace(_segments=lambda *_: pieces)
    )
    assert result["confidence"] == pytest.approx(0.75)
    assert result["confidence_kind"] == "mean_segment_token_likelihood"
    assert result["text"] == "Half Full Unknown"


def test_unsupported_file_export_never_creates_output(tmp_path):
    output = tmp_path / "result.xml"
    result = {"text": "Test", "duration": 1.0, "processing_seconds": 0.1}
    with pytest.raises(ValueError, match="Export format must be json or txt"):
        file_transcription.export_result(result, output, "xml")
    assert not output.exists()


def test_empty_file_never_invokes_inference(monkeypatch):
    monkeypatch.setattr(audio, "decode_audio", lambda *args, **kwargs: np.array([]))
    decoder = SimpleNamespace(_segments=lambda *_: pytest.fail("Inference on empty audio"))
    with pytest.raises(ValueError, match="File contains no audio"):
        file_transcription.transcribe_file("empty.wav", Config(), transcriber=decoder)


def test_file_turns_are_bounded_without_losing_overlapping_attribution(monkeypatch):
    monkeypatch.setattr(audio, "decode_audio", lambda *args, **kwargs: np.zeros(16000))
    monkeypatch.setattr(
        file_transcription,
        "diarize",
        lambda *_: [
            {"start": 0.5, "end": 2, "speaker": 1},
            {"start": -0.25, "end": 0.75, "speaker": 0},
        ],
    )

    def decode(samples, _config):
        return [SimpleNamespace(start=0, end=len(samples) / 16000, text="words")]

    result = file_transcription.transcribe_file(
        "input.wav",
        Config(file_speaker_labels="on"),
        transcriber=SimpleNamespace(_segments=decode),
    )
    assert [(s["start"], s["end"], s["speaker"]) for s in result["segments"]] == [
        (0.0, 0.75, 0),
        (0.5, 1.0, 1),
    ]
    assert result["speaker_turns"] == [
        {"start": 0.5, "end": 1.0, "speaker": 1},
        {"start": 0.0, "end": 0.75, "speaker": 0},
    ]


@pytest.mark.parametrize("count", [True, -1, 21, "automatic"])
def test_invalid_speaker_count_fails_before_accessing_models(count):
    with pytest.raises(ValueError, match="Speaker count must be between"):
        file_transcription.diarize(np.zeros(16000), count)


@pytest.fixture
def cached_speakers():
    paths = file_transcription.speaker_paths()
    paths[0].parent.mkdir(parents=True)
    for path in paths:
        path.write_bytes(b"existing model")
    return paths


@pytest.mark.parametrize(
    ("backend", "valid", "sample_rate", "error"),
    [
        (False, False, 16000, "diarization optional extra"),
        (True, False, 16000, "Invalid or incomplete speaker model"),
        (True, True, 48000, "Speaker model must accept 16 kHz"),
    ],
)
def test_speaker_backend_must_be_installed_and_compatible(
    monkeypatch, cached_speakers, backend, valid, sample_rate, error
):
    def settings(**options):
        return SimpleNamespace(**options)

    def diarization_config(**options):
        assert options["clustering"].num_clusters == -1
        return SimpleNamespace(validate=lambda: valid)

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(
            OfflineSpeakerDiarizationConfig=diarization_config,
            OfflineSpeakerSegmentationModelConfig=settings,
            OfflineSpeakerSegmentationPyannoteModelConfig=settings,
            SpeakerEmbeddingExtractorConfig=settings,
            FastClusteringConfig=settings,
            OfflineSpeakerDiarization=lambda _: SimpleNamespace(sample_rate=sample_rate),
        )
        if backend
        else None,
    )
    with pytest.raises(RuntimeError if not backend else ValueError, match=error):
        file_transcription.diarize(np.zeros(16000))


def model_archive(member_type):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:bz2") as package:
        if member_type != "missing":
            member = tarfile.TarInfo(f"{file_speakers.SEGMENTATION}/model.onnx")
            if member_type == "link":
                member.type = tarfile.SYMTYPE
                member.linkname = "different-model.onnx"
                package.addfile(member)
            else:
                data = b"local segmentation" if member_type == "valid" else b""
                member.size = len(data)
                package.addfile(member, io.BytesIO(data))
        unrelated = tarfile.TarInfo("../must-not-extract")
        package.addfile(unrelated, io.BytesIO())
    return output.getvalue()


@pytest.mark.parametrize(
    ("member_type", "error"),
    [
        ("valid", ""),
        ("missing", "Segmentation model missing"),
        ("link", "Invalid segmentation model archive"),
        ("empty", "Invalid segmentation model archive"),
    ],
)
def test_explicit_speaker_download_stages_only_expected_files(
    monkeypatch, cached_speakers, member_type, error
):
    segmentation, embedding = cached_speakers
    urls = []

    def open_url(url, *, timeout):
        assert url.startswith(file_speakers.RELEASES)
        assert timeout == 30
        urls.append(url)
        data = model_archive(member_type) if url.endswith(".tar.bz2") else b"local embedding"
        return io.BytesIO(data)

    monkeypatch.setattr(file_speakers.urllib.request, "urlopen", open_url)
    if error:
        with pytest.raises(ValueError, match=error):
            file_transcription.download_diarization()
        assert len(urls) == 1
        assert [path.read_bytes() for path in cached_speakers] == [b"existing model"] * 2
    else:
        assert file_transcription.download_diarization() == str(segmentation.parent)
        assert len(urls) == 2
        assert segmentation.read_bytes() == b"local segmentation"
        assert embedding.read_bytes() == b"local embedding"
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in cached_speakers)
    assert sorted(segmentation.parent.iterdir()) == sorted(cached_speakers)


def test_missing_archive_stream_preserves_previous_models(monkeypatch, cached_speakers):
    monkeypatch.setattr(
        file_speakers.urllib.request,
        "urlopen",
        lambda *_args, **_options: io.BytesIO(model_archive("valid")),
    )
    monkeypatch.setattr(tarfile.TarFile, "extractfile", lambda *_: None)
    with pytest.raises(ValueError, match="Cannot read segmentation model archive"):
        file_transcription.download_diarization()
    assert [path.read_bytes() for path in cached_speakers] == [b"existing model"] * 2


def test_failed_embedding_download_keeps_previous_models(monkeypatch, cached_speakers):
    def open_url(url, **_options):
        if url.endswith(".tar.bz2"):
            return io.BytesIO(model_archive("valid"))
        raise OSError("Network interrupted")

    monkeypatch.setattr(file_speakers.urllib.request, "urlopen", open_url)
    with pytest.raises(OSError, match="Network interrupted"):
        file_transcription.download_diarization()
    assert [path.read_bytes() for path in cached_speakers] == [b"existing model"] * 2


def test_model_fetch_checks_scheme_and_size_before_publishing(monkeypatch, tmp_path):
    target = tmp_path / "model.onnx"

    def open_url(url, **_options):
        assert url.startswith("https://")
        return io.BytesIO(b"too long")

    monkeypatch.setattr(file_speakers.urllib.request, "urlopen", open_url)
    with pytest.raises(ValueError, match="require HTTPS"):
        file_speakers.fetch_model("http://example.test/model", target)
    with pytest.raises(ValueError, match="exceeds size limit"):
        file_speakers.fetch_model("https://example.test/model", target, max_bytes=2)
    assert list(tmp_path.iterdir()) == []
