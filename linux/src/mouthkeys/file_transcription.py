"""Structured file transcription, local speaker labels, and portable exports."""

import io
import itertools
import json
import math
import os
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .file_history import FileResult, FileSegment, SpeakerTurn
from .file_speakers import (
    SAMPLE_RATE,
    clip_turns,
    diarize,
    download_diarization,
    label_segments,
    speaker_paths,
    turn_chunks,
)
from .transcribe import Transcriber

__all__ = [
    "diarize",
    "download_diarization",
    "export_result",
    "label_segments",
    "result_text",
    "speaker_paths",
    "transcribe_file",
]


def _decode_bounded(path: str | Path, max_seconds: object) -> NDArray[np.float32]:
    """Bound decoded samples, not just compressed bytes or untrusted duration metadata."""
    import av
    from faster_whisper.audio import _ignore_invalid_frames

    if (
        isinstance(max_seconds, bool)
        or not isinstance(max_seconds, int | float)
        or not math.isfinite(max_seconds)
        or max_seconds <= 0
    ):
        raise ValueError("Audio duration limit must be positive and finite")
    limit = int(max_seconds * SAMPLE_RATE)
    duration = Fraction(0)
    count = 0
    raw = io.BytesIO()
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
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
                if not frame.sample_rate or frame.sample_rate < 0 or frame.samples < 0:
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

    def progress(stage: str, **details: object) -> None:
        if on_progress is not None:
            on_progress({"stage": stage, **details})

    started = time.monotonic()
    progress("decode")
    samples = (
        decode_audio(str(path), sampling_rate=SAMPLE_RATE)
        if max_seconds is None
        else _decode_bounded(path, max_seconds)
    )
    if not len(samples):
        raise ValueError("File contains no audio")
    engine = transcriber or Transcriber()
    duration = len(samples) / SAMPLE_RATE
    turns: list[SpeakerTurn] = []
    confidences: list[float] = []

    def decode(
        samples: NDArray[Any],
        offset: float = 0,
        speaker: int | None = None,
        end: float | None = None,
    ) -> list[FileSegment]:
        decoded: list[FileSegment] = []
        for segment in engine._segments(samples, config):
            segment_start, segment_end = float(segment.start), float(segment.end)
            if not (math.isfinite(segment_start) and math.isfinite(segment_end)):
                continue
            start = max(offset, offset + segment_start)
            finish = offset + segment_end
            if end is not None:
                finish = min(end, finish)
            text = segment.text.strip()
            if not text or finish < start:
                continue
            item: FileSegment = {"start": start, "end": finish, "text": text}
            if speaker is not None:
                item["speaker"] = speaker
            decoded.append(item)
            logprob = getattr(segment, "avg_logprob", None)
            if isinstance(logprob, int | float) and math.isfinite(logprob):
                confidences.append(math.exp(min(0, logprob)))
        return decoded

    if config.file_speaker_labels == "on":
        progress("diarization")
        turns = clip_turns(diarize(samples, config.file_speaker_count), duration)
    if turns and config.file_diarization_mode == "turns":
        segments = []
        chunks = turn_chunks(turns, duration)
        for index, (start, end, speaker) in enumerate(chunks):
            progress(
                "transcribe",
                completed=index,
                total=len(chunks),
                kind="turn" if speaker is not None else "uncovered",
            )
            segments.extend(
                decode(
                    samples[round(start * SAMPLE_RATE) : round(end * SAMPLE_RATE)],
                    start,
                    speaker,
                    end,
                )
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
        "duration": duration,
        "processing_seconds": round(time.monotonic() - started, 3),
        "language": config.language,
        "engine": getattr(config, "engine", "whisper"),
        "model": config.model,
        "confidence": sum(confidences) / len(confidences) if confidences else None,
        "confidence_kind": "mean_segment_token_likelihood" if confidences else "unavailable",
        "diarization_mode": config.file_diarization_mode if turns else None,
    }


def result_text(result: FileResult) -> str:
    if not result.get("speaker_turns"):
        return result["text"]
    return "\n".join(
        f"[{segment['start']:.2f}–{segment['end']:.2f}] "
        f"Speaker {speaker + 1 if (speaker := segment.get('speaker')) is not None else '?'}: "
        f"{segment['text']}"
        for segment in result["segments"]
    )


def export_result(result: FileResult, path: str | Path, format: str = "json") -> None:
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
