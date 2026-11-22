"""Keep a stable transcript prefix and decode only its unresolved audio window."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, Protocol, TypedDict, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .audio import read_audio_from

SAMPLE_RATE = 16000


@runtime_checkable
class Segment(Protocol):
    @property
    def start(self) -> float: ...

    @property
    def end(self) -> float: ...

    @property
    def text(self) -> str: ...


class TimedWord(TypedDict):
    word: str
    start: float
    end: float
    probability: NotRequired[float]
    alignment_source: NotRequired[str]


DecodeSegments = Callable[[NDArray[np.generic]], list[Segment]]


@dataclass(frozen=True)
class DecodedTranscript:
    text: str
    words: list[TimedWord]


def timed_words(segments: Sequence[Segment], offset: float = 0) -> list[TimedWord]:
    words = []
    for segment in segments:
        for word in getattr(segment, "words", None) or []:
            start, end = float(word.start), float(word.end)
            if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start <= end:
                continue
            entry: TimedWord = {"word": word.word, "start": start + offset, "end": end + offset}
            probability = getattr(word, "probability", None)
            if probability is not None and math.isfinite(float(probability)):
                entry["probability"] = float(probability)
            if source := getattr(word, "alignment_source", None):
                entry["alignment_source"] = source
            words.append(entry)
    return words


def decode_transcript(segments: Sequence[Segment], offset: float = 0) -> DecodedTranscript:
    return DecodedTranscript(
        " ".join(s.text.strip() for s in segments).strip(), timed_words(segments, offset)
    )


class TranscriptWindow:
    def __init__(self) -> None:
        self._cursor = 0
        self._covered = 0
        self._committed: list[str] = []
        self._committed_words: list[TimedWord] = []
        self._pending = DecodedTranscript("", [])

    @property
    def committed_segments(self) -> int:
        return len(self._committed)

    def _with_prefix(self, tail: str) -> str:
        return " ".join([*self._committed, tail]).strip()

    def preview(self, path: Path, decode: DecodeSegments) -> str:
        # Grow an unresolved window rather than freezing at its original end.
        unresolved = (self._covered - self._cursor) / SAMPLE_RATE
        seconds = unresolved + 30 if unresolved >= 30 else 30
        audio = read_audio_from(path, self._cursor, seconds=seconds)
        end = self._cursor + len(audio)
        if end != self._covered and len(audio) >= SAMPLE_RATE // 2:
            self._update(decode(audio), end)
        return self._with_prefix(self._pending.text)

    def _update(self, segments: list[Segment], end: int) -> None:
        # Keep two seconds of lookahead before treating a segment as stable.
        cutoff = end - self._cursor - 2 * SAMPLE_RATE
        stable: list[Segment] = []
        boundary = 0
        for segment in segments:
            frame = round(segment.end * SAMPLE_RATE)
            if not 0 < frame <= cutoff:
                break
            stable.append(segment)
            boundary = frame
        offset = self._cursor / SAMPLE_RATE
        committed = [segment.text.strip() for segment in stable]
        words = timed_words(stable, offset)
        pending = decode_transcript(segments[len(stable) :], offset)
        # A malformed segment must not leave half a committed prefix behind.
        self._committed.extend(committed)
        self._committed_words.extend(words)
        self._pending = pending
        self._cursor += boundary
        self._covered = end

    def finish(self, path: Path, decode: DecodeSegments) -> DecodedTranscript:
        audio = read_audio_from(path, self._cursor)
        if self._cursor + len(audio) == self._covered:
            tail = self._pending
        elif len(audio):
            tail = decode_transcript(decode(audio), self._cursor / SAMPLE_RATE)
        else:
            tail = DecodedTranscript("", [])
        return DecodedTranscript(
            self._with_prefix(tail.text), [*self._committed_words, *tail.words]
        )
