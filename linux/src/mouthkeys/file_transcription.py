"""Structured file transcription, local speaker labels, and portable exports."""

import io
import itertools
import json
import math
import os
import shutil
import tarfile
import tempfile
import time
import urllib.request
from fractions import Fraction
from pathlib import Path

from .config import cache_dir, private_dir
from .transcribe import Transcriber

RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION = "sherpa-onnx-pyannote-segmentation-3-0"
EMBEDDING = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"


def speaker_paths():
    root = cache_dir() / "models" / "speakers"
    return root / "segmentation.onnx", root / "embedding.onnx"


def _fetch(url, destination, max_bytes=512 * 1024 * 1024):
    """Download into a staging directory with a bounded size and private permissions."""
    with tempfile.NamedTemporaryFile(dir=destination.parent) as stream:
        with urllib.request.urlopen(url, timeout=30) as response:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("Speaker model download exceeds size limit")
                stream.write(chunk)
        stream.flush()
        shutil.copyfile(stream.name, destination)
    destination.chmod(0o600)


def download_diarization():
    """Explicitly fetch public sherpa-onnx segmentation and embedding artifacts."""
    segmentation, embedding = speaker_paths()
    private_dir(segmentation.parent)
    with tempfile.TemporaryDirectory(dir=segmentation.parent) as directory:
        root = Path(directory)
        archive = root / "segmentation.tar.bz2"
        _fetch(f"{RELEASES}/speaker-segmentation-models/{SEGMENTATION}.tar.bz2", archive)
        with tarfile.open(archive, "r:bz2") as package:
            member = package.getmember(f"{SEGMENTATION}/model.onnx")
            if not member.isfile() or not 0 < member.size <= 512 * 1024 * 1024:
                raise ValueError("Invalid segmentation model archive")
            # Never extract archive paths, links, or other members onto the filesystem.
            with (
                package.extractfile(member) as source,
                (root / "segmentation.onnx").open("wb") as out,
            ):
                shutil.copyfileobj(source, out)
        _fetch(f"{RELEASES}/speaker-recongition-models/{EMBEDDING}", root / "embedding.onnx")
        for name, target in (("segmentation.onnx", segmentation), ("embedding.onnx", embedding)):
            (root / name).chmod(0o600)
            (root / name).replace(target)
    return str(segmentation.parent)


def diarize(samples, speaker_count=0):
    if type(speaker_count) is not int or not 0 <= speaker_count <= 20:
        raise ValueError("Speaker count must be between 0 (automatic) and 20")
    segmentation, embedding = speaker_paths()
    if not segmentation.is_file() or not embedding.is_file():
        raise RuntimeError("Speaker models missing; run mouthkeys download-speakers first")
    try:
        import sherpa_onnx
    except ImportError:
        raise RuntimeError("Speaker labels require the diarization optional extra") from None
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation)
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding)),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=speaker_count or -1, threshold=0.5
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise ValueError("Invalid or incomplete speaker model files")
    model = sherpa_onnx.OfflineSpeakerDiarization(config)
    if model.sample_rate != 16000:
        raise ValueError("Speaker model must accept 16 kHz audio")
    return [
        {"start": float(turn.start), "end": float(turn.end), "speaker": int(turn.speaker)}
        for turn in model.process(samples).sort_by_start_time()
    ]


def label_segments(segments, turns):
    """Segment-level attribution; preserve full turns so overlaps are not concealed."""
    for segment in segments:
        overlaps = [
            (max(0, min(segment["end"], turn["end"]) - max(segment["start"], turn["start"])), turn)
            for turn in turns
        ]
        overlap, turn = max(overlaps, key=lambda item: item[0], default=(0, None))
        segment["speaker"] = turn["speaker"] if overlap > 0 else None


def _decode_bounded(path, max_seconds):
    """Bound decoded samples, not just compressed bytes or untrusted duration metadata."""
    import av
    import numpy as np
    from faster_whisper.audio import _ignore_invalid_frames

    if (
        isinstance(max_seconds, bool)
        or not isinstance(max_seconds, int | float)
        or not math.isfinite(max_seconds)
        or max_seconds <= 0
    ):
        raise ValueError("Audio duration limit must be positive and finite")
    limit = int(max_seconds * 16000)
    duration = Fraction(0)
    count = 0
    raw = io.BytesIO()
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
    # Custom AVIO reads only uploaded bytes. Disallow nested file/network protocol
    # access (for example a playlist referring to another local file or an URL).
    with (
        Path(path).open("rb") as source,
        av.open(
            source,
            mode="r",
            metadata_errors="ignore",
            options={"protocol_whitelist": "none"},
        ) as container,
    ):
        frames = _ignore_invalid_frames(container.decode(audio=0))
        for frame in itertools.chain(frames, [None]):
            if frame is not None:
                if not frame.sample_rate or frame.samples < 0:
                    raise ValueError("Invalid audio frame")
                duration += Fraction(frame.samples, frame.sample_rate)
                if duration > max_seconds:
                    raise ValueError(f"Audio exceeds {max_seconds:g} seconds")
                frame.pts = None  # Count actual samples rather than timestamp gaps.
            for output in resampler.resample(frame):
                count += output.samples
                if count > limit:
                    raise ValueError(f"Audio exceeds {max_seconds:g} seconds")
                # Check before to_ndarray so overflowing output is never accumulated.
                raw.write(output.to_ndarray())
    return np.frombuffer(raw.getbuffer(), dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_file(path, config, *, transcriber=None, max_seconds=None, on_progress=None):
    """Decode files without capture/delivery/history; optionally report stage dictionaries.

    Transcription counters count completed ASR chunks (including uncovered intervals),
    not estimated wall-clock completion. Whole-file backends report only their stage.
    """
    from faster_whisper.audio import decode_audio

    def progress(stage, **details):
        if on_progress is not None:
            on_progress({"stage": stage, **details})

    started = time.monotonic()
    progress("decode")
    samples = (
        decode_audio(str(path), sampling_rate=16000)
        if max_seconds is None
        else _decode_bounded(path, max_seconds)
    )
    if not len(samples):
        raise ValueError("File contains no audio")
    engine = transcriber or Transcriber()
    turns = []
    confidences = []

    def decode(samples, offset=0, speaker=None, end=None):
        decoded = []
        for segment in engine._segments(samples, config):
            start = max(offset, offset + float(segment.start))
            finish = offset + float(segment.end)
            if end is not None:
                finish = min(end, finish)
            text = segment.text.strip()
            if not text or finish < start:
                continue
            item = {"start": start, "end": finish, "text": text}
            if speaker is not None:
                item["speaker"] = speaker
            decoded.append(item)
            logprob = getattr(segment, "avg_logprob", None)
            if isinstance(logprob, int | float) and math.isfinite(logprob):
                confidences.append(math.exp(min(0, logprob)))
        return decoded

    if config.file_speaker_labels == "on":
        progress("diarization")
        turns = diarize(samples, config.file_speaker_count)
    if turns and config.file_diarization_mode == "turns":
        segments = []
        chunks = []
        duration = len(samples) / 16000
        covered = 0.0
        for turn in sorted(turns, key=lambda item: item["start"]):
            start, end = max(0.0, turn["start"]), min(duration, turn["end"])
            if not (math.isfinite(start) and math.isfinite(end) and start < end):
                continue
            # Decode uncovered intervals too: diarization misses must not silently lose words.
            if start > covered:
                chunks.append((covered, start, None))
            chunks.append((start, end, turn["speaker"]))
            covered = max(covered, end)
        if covered < duration:
            chunks.append((covered, duration, None))
        for index, (start, end, speaker) in enumerate(chunks):
            progress(
                "transcribe",
                completed=index,
                total=len(chunks),
                kind="turn" if speaker is not None else "uncovered",
            )
            segments.extend(
                decode(samples[round(start * 16000) : round(end * 16000)], start, speaker, end)
            )
            progress("transcribe", completed=index + 1, total=len(chunks))
        segments.sort(key=lambda item: item["start"])
    else:
        # Backends may block before returning their full segment list. No invented percentage.
        progress("transcribe")
        segments = decode(samples)
        if turns:
            label_segments(segments, turns)
    progress("finalize")
    return {
        "text": " ".join(segment["text"] for segment in segments).strip(),
        "segments": segments,
        "speaker_turns": turns,
        "duration": len(samples) / 16000,
        "processing_seconds": round(time.monotonic() - started, 3),
        "language": config.language,
        "engine": getattr(config, "engine", "whisper"),
        "model": config.model,
        "confidence": sum(confidences) / len(confidences) if confidences else None,
        "confidence_kind": "mean_segment_token_likelihood" if confidences else "unavailable",
        "diarization_mode": config.file_diarization_mode if turns else None,
    }


def result_text(result):
    if not result.get("speaker_turns"):
        return result["text"]
    return "\n".join(
        f"[{segment['start']:.2f}–{segment['end']:.2f}] "
        f"Speaker {segment['speaker'] + 1 if segment.get('speaker') is not None else '?'}: "
        f"{segment['text']}"
        for segment in result["segments"]
    )


def export_result(result, path, format="json"):
    if format not in ("json", "txt"):
        raise ValueError("Export format must be json or txt")
    path = Path(path)
    text = (
        json.dumps(result, ensure_ascii=False, indent=2)
        if format == "json"
        else result_text(result)
    )
    # Audio/transcript exports remain private and never leave half-written output.
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text + "\n")
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)
