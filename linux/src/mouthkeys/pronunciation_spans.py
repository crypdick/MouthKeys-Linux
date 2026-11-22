"""Search observed word-timed audio and apply only complete, unambiguous matches."""

import math
import time
import wave
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .acoustic_features import AcousticFeatures, acoustic_distance, acoustic_features
from .word_alignment import AlignedWord, CharacterOffset, align_words, lexical_span

SAMPLE_RATE = 16000
MAX_DISTANCE_CELLS = 12_800_000
TIMING_WARNING = "Word timing metadata is incomplete or inconsistent"
Samples = NDArray[np.float32]


class PronunciationProfile(TypedDict):
    label: str
    version: str
    samples: list[AcousticFeatures]


class SpanCorrection(TypedDict):
    word_start: int
    word_end: int
    text_start: CharacterOffset
    text_end: CharacterOffset
    start: float
    end: float
    label: str
    heard: str
    distance: float
    confidence: float


class SpanResult(TypedDict):
    text: str
    words: list[AlignedWord]
    corrections: list[SpanCorrection]
    comparisons: int
    cells: int
    candidates: int
    truncated: bool
    skipped_reason: str


@dataclass
class _SearchBudget:
    limit: int
    deadline: float
    comparisons: int = 0
    cells: int = 0
    candidates: int = 0
    truncated: bool = False

    def expired(self) -> bool:
        self.truncated = self.truncated or time.monotonic() >= self.deadline
        return self.truncated

    def reserve(self, cells: int) -> bool:
        if (
            self.comparisons >= self.limit
            or self.cells + cells > MAX_DISTANCE_CELLS
            or self.expired()
        ):
            self.truncated = True
            return False
        self.comparisons += 1
        self.cells += cells
        return True


def search_budget(comparisons: object, seconds: object) -> _SearchBudget:
    if type(comparisons) is not int or not 1 <= comparisons <= 20000:
        raise ValueError("Pronunciation span budget must be 1–20000 comparisons")
    if (
        not isinstance(seconds, int | float)
        or type(seconds) not in (int, float)
        or not 0 < seconds <= 10
    ):
        raise ValueError("Pronunciation span time limit must be 0–10 seconds")
    return _SearchBudget(comparisons, time.monotonic() + seconds)


@dataclass(frozen=True)
class _AudioWindow:
    frames: int
    read_frames: Callable[[int, int], Samples]

    @property
    def duration(self) -> float:
        return self.frames / SAMPLE_RATE

    def read(self, start: float, end: float) -> Samples:
        first = max(0, math.floor(start * SAMPLE_RATE))
        last = min(self.frames, math.ceil(end * SAMPLE_RATE))
        return self.read_frames(first, last)


def _checked_window(frames: int, read: Callable[[int, int], Samples]) -> _AudioWindow:
    if not 0 < frames <= 3600 * SAMPLE_RATE:
        raise ValueError("Span matching supports recordings up to one hour")
    return _AudioWindow(frames, read)


@contextmanager
def _audio_window(audio: object) -> Iterator[_AudioWindow]:
    if isinstance(audio, str | Path):
        with wave.open(str(audio), "rb") as source:
            if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (
                1,
                2,
                SAMPLE_RATE,
            ):
                raise ValueError("Span matching requires mono 16-bit PCM at 16 kHz")

            def read(first: int, last: int) -> Samples:
                source.setpos(first)
                raw = source.readframes(last - first)
                return np.frombuffer(raw, dtype="<i2").astype(np.float32) / np.float32(32768)

            yield _checked_window(source.getnframes(), read)
    else:
        samples = np.asarray(cast("ArrayLike", audio), dtype=np.float32)
        if samples.ndim != 1 or not np.all(np.isfinite(samples)):
            raise ValueError("Span matching requires finite mono audio")
        yield _checked_window(len(samples), lambda first, last: samples[first:last])


def _span_distance(candidate: AcousticFeatures, enrolled: AcousticFeatures) -> float:
    # Full-span DTW can stretch an endpoint over an unrelated neighbor.
    if not 0.65 <= candidate["duration"] / enrolled["duration"] <= 1.55:
        return 1.0
    left = np.asarray(candidate["features"])
    right = np.asarray(enrolled["features"])
    edges = []
    for end in (False, True):
        first = left[-max(1, len(left) // 8) :] if end else left[: max(1, len(left) // 8)]
        second = right[-max(1, len(right) // 8) :] if end else right[: max(1, len(right) // 8)]
        first, second = first.mean(axis=0), second.mean(axis=0)
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        edges.append(float(1 - first @ second / max(denominator, 1e-8)))
    if max(edges) > 0.4 or sum(edges) / 2 > 0.25:
        return 1.0
    return acoustic_distance(candidate, enrolled)


def _separates_phrases(previous: AlignedWord, following: AlignedWord) -> bool:
    return following["start"] - previous["end"] > 0.4 or any(
        symbol in ",.;:!?…。“”" for symbol in previous["word"].rstrip()[-1:]
    )


def _word_windows(
    words: list[AlignedWord], budget: _SearchBudget
) -> Iterator[tuple[int, int, float, float]]:
    for first in range(len(words)):
        for last in range(first, min(len(words), first + 5)):
            if budget.expired():
                return
            if last > first and _separates_phrases(words[last - 1], words[last]):
                break
            start, end = words[first]["start"], words[last]["end"]
            # Do not replace a token that shares its observed time with an excluded neighbor.
            if first and words[first - 1]["end"] > start + 0.02:
                continue
            if last + 1 < len(words) and words[last + 1]["start"] < end - 0.02:
                continue
            if end - start > 10:
                break
            if end - start >= 0.25:
                yield first, last, start, end


def _plausible_duration(duration: float, enrolled: float) -> bool:
    return 0.5 <= duration / enrolled <= 2


def _rank(
    features: AcousticFeatures, profiles: list[PronunciationProfile], budget: _SearchBudget
) -> tuple[float, str] | None:
    ranked = []
    for profile in profiles:
        distances = []
        for take in profile["samples"]:
            if not _plausible_duration(features["duration"], take["duration"]):
                distances.append(1.0)
                continue
            cells = len(features["features"]) * len(take["features"])
            if not budget.reserve(cells):
                return None
            distances.append(_span_distance(features, take))
        distances.sort()
        ranked.append((sum(distances[:2]) / 2, profile["label"]))
    if budget.expired():
        return None  # The last comparison itself can finish after the deadline.
    ranked.sort()
    best = ranked[0]
    if best[0] > 0.18 or (len(ranked) > 1 and ranked[1][0] - best[0] < 0.06):
        return None
    return best


def _collect_matches(
    text: str,
    words: list[AlignedWord],
    source: _AudioWindow,
    profiles: list[PronunciationProfile],
    budget: _SearchBudget,
) -> list[SpanCorrection]:
    durations = {take["duration"] for profile in profiles for take in profile["samples"]}
    matches = []
    for first, last, start, end in _word_windows(words, budget):
        if not any(_plausible_duration(end - start, duration) for duration in durations):
            continue
        if budget.expired():
            break
        char_start, char_end = lexical_span(text, words[first], words[last])
        if char_start == char_end:
            continue
        budget.candidates += 1
        try:
            features = acoustic_features(source.read(start, end))
        except ValueError:
            continue
        best = _rank(features, profiles, budget)
        if best is None:
            continue
        distance, label = best
        heard = text[char_start:char_end]
        if heard != label:
            matches.append(
                SpanCorrection(
                    word_start=first,
                    word_end=last + 1,
                    text_start=char_start,
                    text_end=char_end,
                    start=start,
                    end=end,
                    label=label,
                    heard=heard,
                    distance=distance,
                    confidence=round(1 - distance, 4),
                )
            )
    return matches


def _choose_disjoint(matches: list[SpanCorrection]) -> list[SpanCorrection]:
    chosen: list[SpanCorrection] = []
    for match in sorted(
        matches, key=lambda item: (item["distance"], item["word_start"], item["word_end"])
    ):
        if not any(
            match["word_start"] < other["word_end"] and match["word_end"] > other["word_start"]
            for other in chosen
        ):
            chosen.append(match)
    return sorted(chosen, key=lambda item: item["word_start"])


def _rewrite(
    text: str, words: list[AlignedWord], chosen: list[SpanCorrection], duration: float
) -> tuple[str, list[AlignedWord]]:
    if not chosen:
        return text, words
    corrected = text
    for replacement in reversed(chosen):
        start, end = replacement["text_start"], replacement["text_end"]
        corrected = corrected[:start] + replacement["label"] + corrected[end:]

    replacements = {match["word_start"]: match for match in chosen}
    updated: list[dict[str, object]] = []
    index = 0
    while index < len(words):
        match = replacements.get(index)
        if match is None:
            word = dict(words[index])
            word.pop("char_start")
            word.pop("char_end")
            updated.append(word)
            index += 1
            continue
        last = match["word_end"] - 1
        prefix = text[words[index]["char_start"] : match["text_start"]]
        suffix = text[match["text_end"] : words[last]["char_end"]]
        updated.append(
            {
                "word": prefix + match["label"] + suffix,
                "start": match["start"],
                "end": match["end"],
                "kind": "pronunciation_span",
                "original_words": [dict(word) for word in words[index : last + 1]],
            }
        )
        index = last + 1
    return corrected, align_words(corrected, updated, duration=duration)


def correct_spans(
    text: str,
    audio: object,
    alignment: object,
    load_profiles: Callable[[], list[PronunciationProfile]],
    limits: _SearchBudget,
) -> SpanResult:
    result: SpanResult = {
        "text": text,
        "words": [],
        "corrections": [],
        "comparisons": 0,
        "cells": 0,
        "candidates": 0,
        "truncated": False,
        "skipped_reason": "",
    }
    if not isinstance(alignment, dict) or alignment.get("text") != text:
        result["skipped_reason"] = "Alignment does not match final transcript"
        return result
    if isinstance(audio, str | Path) and alignment.get("source") != str(audio):
        result["skipped_reason"] = "Alignment belongs to another recording"
        return result
    profiles = [profile for profile in load_profiles() if len(profile["samples"]) >= 2]
    if not profiles:
        try:
            result["words"] = align_words(text, alignment.get("words"))
        except ValueError:
            result["skipped_reason"] = TIMING_WARNING
        return result
    with _audio_window(audio) as source:
        try:
            words = align_words(text, alignment.get("words"), duration=source.duration)
        except ValueError:
            result["skipped_reason"] = TIMING_WARNING
            return result
        chosen = _choose_disjoint(_collect_matches(text, words, source, profiles, limits))
        result["text"], result["words"] = _rewrite(text, words, chosen, source.duration)
    result["corrections"] = chosen
    result["comparisons"] = limits.comparisons
    result["cells"] = limits.cells
    result["candidates"] = limits.candidates
    result["truncated"] = limits.truncated
    return result
