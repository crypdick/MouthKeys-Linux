"""Local acoustic phrase enrollment using cepstra and dynamic time warping.

Unlike Apple's model-specific embeddings, this CPU-only matcher needs no model
or download. It enrolls isolated phrases (0.25–10 seconds) and matches them either
as whole utterances or inside actual ASR word-timed spans. Two consistent takes
and a clear winner are required before automatic correction.
"""

import json
import math
import threading
import time
import wave
from pathlib import Path

import numpy as np

from .config import config_path, private_dir
from .operations import write_private

FEATURE_VERSION = "cepstral-dtw-16k-v1"
_LOCK = threading.RLock()


def _samples(audio):
    if isinstance(audio, str | Path):
        with wave.open(str(audio), "rb") as source:
            if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (
                1,
                2,
                16000,
            ):
                raise ValueError("Pronunciation audio must be mono 16-bit PCM at 16 kHz")
            if source.getnframes() > 160000:
                raise ValueError("Pronunciation phrase exceeds ten seconds")
            samples = (
                np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(
                    np.float32
                )
                / 32768
            )
    else:
        samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or not 4000 <= len(samples) <= 160000 or not np.all(np.isfinite(samples)):
        raise ValueError("Pronunciation phrase must contain 0.25–10 seconds of finite mono audio")
    peak = float(np.max(np.abs(samples)))
    if peak < 0.001 or float(np.std(samples)) < 0.0001:
        raise ValueError("Pronunciation sample contains no audible speech")
    # Trim only low-energy edges. Interior pauses remain part of pronunciation.
    energy = np.array(
        [
            np.sqrt(np.mean(part * part))
            for part in np.array_split(samples, max(1, len(samples) // 160))
        ]
    )
    active = np.flatnonzero(energy > max(0.0005, float(energy.max()) * 0.06))
    if not len(active):
        raise ValueError("Pronunciation sample contains no audible speech")
    step = len(samples) / len(energy)
    start = max(0, int((active[0] - 2) * step))
    end = min(len(samples), int((active[-1] + 3) * step))
    samples = samples[start:end]
    if len(samples) < 3200:
        raise ValueError("Pronunciation sample contains too little speech")
    return samples / max(peak, 1e-6)


def acoustic_features(audio):
    samples = _samples(audio)
    duration = len(samples) / 16000
    emphasized = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])
    starts = np.arange(0, max(1, len(emphasized) - 399), 160)
    frames = np.stack(
        [
            np.pad(emphasized[start : start + 400], (0, max(0, start + 400 - len(emphasized))))
            for start in starts
        ]
    )
    power = np.abs(np.fft.rfft(frames * np.hamming(400), n=512)) ** 2
    mel_edges = np.linspace(2595 * np.log10(1 + 70 / 700), 2595 * np.log10(1 + 7600 / 700), 28)
    frequencies = 700 * (10 ** (mel_edges / 2595) - 1)
    bins = np.floor(513 * frequencies / 16000).astype(int)
    filters = np.zeros((26, 257))
    for index, (left, center, right) in enumerate(zip(bins, bins[1:], bins[2:], strict=False)):
        filters[index, left:center] = np.arange(center - left) / max(1, center - left)
        filters[index, center:right] = np.arange(right - center, 0, -1) / max(1, right - center)
    mel_power = power @ filters.T
    floor = np.maximum(mel_power.max(axis=1, keepdims=True) * 1e-3, 1e-10)
    log_mel = np.log(np.maximum(mel_power, floor))
    dct = np.cos(np.pi / 26 * (np.arange(26) + 0.5)[:, None] * np.arange(1, 13))
    cepstra = log_mel @ dct
    # Per-frame normalization removes gain while retaining spectral articulation.
    cepstra /= np.maximum(np.linalg.norm(cepstra, axis=1, keepdims=True), 1e-8)
    if len(cepstra) > 80:
        indices = np.linspace(0, len(cepstra) - 1, 80)
        cepstra = np.stack(
            [np.interp(indices, np.arange(len(cepstra)), cepstra[:, i]) for i in range(12)], axis=1
        )
    return {"duration": duration, "features": np.round(cepstra, 6).tolist()}


def acoustic_distance(first, second):
    """Length-normalized DTW cosine distance; constrained to plausible speech rate."""
    ratio = first["duration"] / second["duration"]
    if not 0.5 <= ratio <= 2:
        return 1.0
    left, right = np.asarray(first["features"]), np.asarray(second["features"])
    costs = np.clip(1 - left @ right.T, 0, 2)
    rows, cols = costs.shape
    previous = np.full(cols + 1, np.inf)
    previous[0] = 0
    for i in range(1, rows + 1):
        current = np.full(cols + 1, np.inf)
        indices = np.flatnonzero(np.abs(i / rows - np.arange(1, cols + 1) / cols) <= 0.35) + 1
        if len(indices):
            row = costs[i - 1, indices - 1]
            prefix = np.cumsum(row)
            incoming = np.minimum(previous[indices], previous[indices - 1])
            # Exact left/diagonal/up recurrence, expressed as a prefix minimum.
            current[indices] = prefix + np.minimum.accumulate(incoming + row - prefix)
        previous = current
    return float(previous[-1] / max(rows, cols))


class PronunciationStore:
    @staticmethod
    def path():
        return config_path().with_name("pronunciations.json")

    @staticmethod
    def validate(profiles):
        if not isinstance(profiles, list) or len(profiles) > 50:
            raise ValueError("At most 50 pronunciation phrases can be enrolled")
        seen = set()
        for profile in profiles:
            if not isinstance(profile, dict) or set(profile) != {"label", "version", "samples"}:
                raise ValueError("Invalid pronunciation profile")
            label = profile["label"]
            if (
                not isinstance(label, str)
                or not label.strip()
                or len(label) > 128
                or not all(c.isprintable() for c in label)
            ):
                raise ValueError("Pronunciation labels must contain 1–128 printable characters")
            if label.casefold() in seen or profile["version"] != FEATURE_VERSION:
                raise ValueError("Duplicate or incompatible pronunciation profile")
            seen.add(label.casefold())
            if not isinstance(profile["samples"], list) or not 1 <= len(profile["samples"]) <= 10:
                raise ValueError("Pronunciation requires 1–10 samples")
            for sample in profile["samples"]:
                if not isinstance(sample, dict) or set(sample) != {"duration", "features"}:
                    raise ValueError("Invalid pronunciation sample")
                duration = sample["duration"]
                values = np.asarray(sample["features"], dtype=float)
                if (
                    isinstance(duration, bool)
                    or not isinstance(duration, int | float)
                    or not np.isfinite(duration)
                    or not 0.2 <= duration <= 10
                    or values.ndim != 2
                    or values.shape[1] != 12
                    or not 1 <= len(values) <= 80
                    or not np.all(np.isfinite(values))
                    or np.max(np.abs(values)) > 1.01
                ):
                    raise ValueError("Invalid pronunciation feature dimensions or values")
        return profiles

    @classmethod
    def load(cls):
        with _LOCK:
            path = cls.path()
            if not path.exists():
                return []
            if path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("Pronunciation store exceeds 16 MiB")
            return cls.validate(json.loads(path.read_text()))

    @classmethod
    def save(cls, profiles):
        with _LOCK:
            cls.validate(profiles)
            path = cls.path()
            private_dir(path.parent)
            write_private(path, json.dumps(profiles, ensure_ascii=False).encode())
            return profiles

    @classmethod
    def enroll(cls, label, audio):
        if (
            not isinstance(label, str)
            or not label.strip()
            or len(label) > 128
            or not all(character.isprintable() for character in label)
        ):
            raise ValueError("Pronunciation labels must contain 1–128 printable characters")
        label = label.strip()
        sample = acoustic_features(audio)
        with _LOCK:
            profiles = cls.load()
            profile = next(
                (item for item in profiles if item["label"].casefold() == label.casefold()), None
            )
            if profile is None:
                profile = {"label": label.strip(), "version": FEATURE_VERSION, "samples": []}
                profiles.append(profile)
            elif min(acoustic_distance(sample, previous) for previous in profile["samples"]) > 0.35:
                raise ValueError(
                    "Pronunciation differs from existing takes; repeat the same phrase"
                )
            profile["samples"] = [*profile["samples"], sample][-10:]
            cls.save(profiles)
            return {
                "label": profile["label"],
                "samples": len(profile["samples"]),
                "ready": len(profile["samples"]) >= 2,
            }

    @classmethod
    def delete(cls, label):
        with _LOCK:
            return cls.save(
                [item for item in cls.load() if item["label"].casefold() != label.casefold()]
            )

    @classmethod
    def match(cls, audio, *, threshold=0.18, margin=0.06):
        sample = acoustic_features(audio)
        candidates = []
        for profile in cls.load():
            if len(profile["samples"]) < 2:
                continue
            distances = sorted(acoustic_distance(sample, take) for take in profile["samples"])
            # A lucky match against just one inconsistent take is insufficient.
            distance = sum(distances[:2]) / 2
            candidates.append((distance, profile["label"]))
        candidates.sort()
        if not candidates or candidates[0][0] > threshold:
            return None
        if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < margin:
            return None
        distance, label = candidates[0]
        return {"label": label, "confidence": round(1 - distance, 4), "distance": distance}

    @classmethod
    def correct_spans(cls, text, audio, alignment, *, budget=2000, max_seconds=2.0):
        """Correct bounded, real word-timed spans without rewriting their neighbors.

        `alignment` contains exact `text`, `source` (recording path), and actual
        `words` [{word,start,end}] in recording seconds. Returns corrected text,
        aligned words, original-range correction metadata, and coverage counters.
        Replacements have ONE observed interval for the whole phrase, never
        fabricated timings for its individual new words.

        Search covers 1–5 source words, 0.25–10 s per span. At most `budget` DTW
        comparisons, 12.8M distance cells, and `max_seconds` wall time are spent.
        A partially compared candidate is discarded, not treated as unambiguous.
        `truncated` explicitly reports incomplete coverage. Alignment/source
        mismatches fail closed and leave the transcript unchanged.
        """
        return _correct_spans(cls, text, audio, alignment, budget=budget, max_seconds=max_seconds)

    @classmethod
    def correct(cls, text, audio):
        # Never replace a sentence/paragraph using an isolated-phrase matcher.
        if not text.strip() or len(text.split()) > 5 or len(text) > 128:
            return text
        try:
            match = cls.match(audio)
        except ValueError:
            return text  # Long/no-speech captures are outside this matcher's contract.
        return match["label"] if match else text


class _AudioWindow:
    def __init__(self, audio):
        self.source = None
        self.samples = None
        if isinstance(audio, str | Path):
            self.source = wave.open(str(audio), "rb")
            if (
                self.source.getnchannels(),
                self.source.getsampwidth(),
                self.source.getframerate(),
            ) != (1, 2, 16000):
                self.close()
                raise ValueError("Span matching requires mono 16-bit PCM at 16 kHz")
            self.frames = self.source.getnframes()
        else:
            self.samples = np.asarray(audio, dtype=np.float32)
            if self.samples.ndim != 1 or not np.all(np.isfinite(self.samples)):
                raise ValueError("Span matching requires finite mono audio")
            self.frames = len(self.samples)
        if not 0 < self.frames <= 3600 * 16000:
            self.close()
            raise ValueError("Span matching supports recordings up to one hour")
        self.duration = self.frames / 16000

    def read(self, start, end):
        first, last = max(0, math.floor(start * 16000)), min(self.frames, math.ceil(end * 16000))
        if self.source is not None:
            self.source.setpos(first)
            return (
                np.frombuffer(self.source.readframes(last - first), dtype="<i2").astype(np.float32)
                / 32768
            )
        return self.samples[first:last]

    def close(self):
        if self.source is not None:
            self.source.close()


def _span_distance(candidate, enrolled):
    # Full-span DTW alone can stretch one endpoint over an unrelated neighbor.
    # Require both acoustic edges to resemble the enrolled phrase as well.
    ratio = candidate["duration"] / enrolled["duration"]
    if not 0.65 <= ratio <= 1.55:
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


def _correct_spans(store, text, audio, alignment, *, budget, max_seconds):
    from .word_alignment import align_words, lexical_span

    if type(budget) is not int or not 1 <= budget <= 20000:
        raise ValueError("Pronunciation span budget must be 1–20000 comparisons")
    if (
        type(max_seconds) not in (int, float)
        or not math.isfinite(max_seconds)
        or not 0 < max_seconds <= 10
    ):
        raise ValueError("Pronunciation span time limit must be 0–10 seconds")
    deadline = time.monotonic() + max_seconds
    result = {
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
    profiles = [profile for profile in store.load() if len(profile["samples"]) >= 2]
    if not profiles:
        try:
            result["words"] = align_words(text, alignment.get("words"))
        except ValueError:
            result["skipped_reason"] = "Word timing metadata is incomplete or inconsistent"
        return result
    source = _AudioWindow(audio)
    try:
        try:
            words = align_words(text, alignment.get("words"), duration=source.duration)
        except ValueError:
            result["skipped_reason"] = "Word timing metadata is incomplete or inconsistent"
            return result
        result["words"] = words
        matches = []
        stop = False
        for first in range(len(words)):
            for last in range(first, min(len(words), first + 5)):
                if time.monotonic() >= deadline:
                    result["truncated"] = stop = True
                    break
                if last > first:
                    previous = words[last - 1]
                    if words[last]["start"] - previous["end"] > 0.4:
                        break
                    if any(
                        character in ",.;:!?…。“”" for character in previous["word"].rstrip()[-1:]
                    ):
                        break
                start, end = words[first]["start"], words[last]["end"]
                duration = end - start
                # A token boundary inside another token's observed audio is
                # ambiguous. Only a range covering that overlap can be replaced.
                if first and words[first - 1]["end"] > start + 0.02:
                    continue
                if last + 1 < len(words) and words[last + 1]["start"] < end - 0.02:
                    continue
                if duration > 10:
                    break
                if duration < 0.25:
                    continue
                if not any(
                    0.5 <= duration / take["duration"] <= 2
                    for profile in profiles
                    for take in profile["samples"]
                ):
                    continue
                if time.monotonic() >= deadline:
                    result["truncated"] = stop = True
                    break
                char_start, char_end = lexical_span(text, words[first], words[last])
                if char_start == char_end:
                    continue
                result["candidates"] += 1
                try:
                    features = acoustic_features(source.read(start, end))
                except ValueError:
                    continue
                ranked = []
                for profile in profiles:
                    distances = []
                    for take in profile["samples"]:
                        # Outside this duration ratio the defined metric is exactly 1.
                        if not 0.5 <= features["duration"] / take["duration"] <= 2:
                            distances.append(1.0)
                            continue
                        cells = len(features["features"]) * len(take["features"])
                        if (
                            result["comparisons"] >= budget
                            or result["cells"] + cells > 12800000
                            or time.monotonic() >= deadline
                        ):
                            result["truncated"] = stop = True
                            break
                        result["comparisons"] += 1
                        result["cells"] += cells
                        distances.append(_span_distance(features, take))
                    if stop:
                        break
                    distances.sort()
                    ranked.append((sum(distances[:2]) / 2, profile["label"]))
                if stop:
                    break
                ranked.sort()
                if ranked[0][0] > 0.18 or (len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 0.06):
                    continue
                distance, label = ranked[0]
                if text[char_start:char_end] == label:
                    continue
                matches.append(
                    {
                        "word_start": first,
                        "word_end": last + 1,
                        "text_start": char_start,
                        "text_end": char_end,
                        "start": start,
                        "end": end,
                        "label": label,
                        "heard": text[char_start:char_end],
                        "distance": distance,
                        "confidence": round(1 - distance, 4),
                    }
                )
            if stop:
                break
        chosen = []
        for match in sorted(
            matches, key=lambda item: (item["distance"], item["word_start"], item["word_end"])
        ):
            if not any(
                match["word_start"] < other["word_end"] and match["word_end"] > other["word_start"]
                for other in chosen
            ):
                chosen.append(match)
        chosen.sort(key=lambda item: item["word_start"])
        corrected = text
        for match in reversed(chosen):
            corrected = (
                corrected[: match["text_start"]] + match["label"] + corrected[match["text_end"] :]
            )
        replacements = {match["word_start"]: match for match in chosen}
        updated, index = [], 0
        while index < len(words):
            match = replacements.get(index)
            if match is None:
                updated.append(
                    {
                        key: value
                        for key, value in words[index].items()
                        if key not in ("char_start", "char_end")
                    }
                )
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
        result.update(
            text=corrected,
            words=align_words(corrected, updated, duration=source.duration),
            corrections=chosen,
        )
        return result
    finally:
        source.close()
