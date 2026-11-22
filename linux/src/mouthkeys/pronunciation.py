"""Local acoustic phrase enrollment using cepstra and dynamic time warping.

Unlike Apple's model-specific embeddings, this CPU-only matcher needs no model
or download. It enrolls isolated phrases (0.25–10 seconds) and matches them either
as whole utterances or inside actual ASR word-timed spans. Two consistent takes
and a clear winner are required before automatic correction.
"""

import json
import threading
from pathlib import Path
from typing import TypedDict

import numpy as np

from .acoustic_features import AcousticFeatures, acoustic_distance, acoustic_features
from .config import config_path, private_dir
from .operations import write_private
from .pronunciation_spans import PronunciationProfile, SpanResult, correct_spans, search_budget

FEATURE_VERSION = "cepstral-dtw-16k-v1"
_LOCK = threading.RLock()


class PronunciationMatch(TypedDict):
    label: str
    confidence: float
    distance: float


def _validated_label(label: object) -> str:
    if (
        not isinstance(label, str)
        or not label.strip()
        or len(label) > 128
        or not all(character.isprintable() for character in label)
    ):
        raise ValueError("Pronunciation labels must contain 1–128 printable characters")
    return label


def _validated_take(sample: object) -> AcousticFeatures:
    if not isinstance(sample, dict) or set(sample) != {"duration", "features"}:
        raise ValueError("Invalid pronunciation sample")
    duration = sample["duration"]
    if (
        not isinstance(duration, int | float)
        or isinstance(duration, bool)
        or not 0.2 <= duration <= 10
    ):
        raise ValueError("Invalid pronunciation sample duration")
    try:
        values = np.asarray(sample["features"], dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid pronunciation feature values") from exc
    if (
        values.ndim != 2
        or values.shape[1] != 12
        or not 1 <= len(values) <= 80
        or not np.all(np.isfinite(values))
        or np.max(np.abs(values)) > 1.01
    ):
        raise ValueError("Invalid pronunciation feature dimensions or values")
    return {"duration": duration, "features": values.tolist()}


class PronunciationStore:
    @staticmethod
    def path() -> Path:
        return config_path().with_name("pronunciations.json")

    @staticmethod
    def validate(profiles: object) -> list[PronunciationProfile]:
        if not isinstance(profiles, list) or len(profiles) > 50:
            raise ValueError("At most 50 pronunciation phrases can be enrolled")
        seen = set()
        validated = []
        for profile in profiles:
            if not isinstance(profile, dict) or set(profile) != {"label", "version", "samples"}:
                raise ValueError("Invalid pronunciation profile")
            label = _validated_label(profile["label"])
            if label.casefold() in seen or profile["version"] != FEATURE_VERSION:
                raise ValueError("Duplicate or incompatible pronunciation profile")
            seen.add(label.casefold())
            if not isinstance(profile["samples"], list) or not 1 <= len(profile["samples"]) <= 10:
                raise ValueError("Pronunciation requires 1–10 samples")
            validated.append(
                PronunciationProfile(
                    label=label,
                    version=FEATURE_VERSION,
                    samples=[_validated_take(sample) for sample in profile["samples"]],
                )
            )
        return validated

    @classmethod
    def load(cls) -> list[PronunciationProfile]:
        with _LOCK:
            path = cls.path()
            if not path.exists():
                return []
            if path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("Pronunciation store exceeds 16 MiB")
            return cls.validate(json.loads(path.read_text()))

    @classmethod
    def save(cls, profiles: object) -> list[PronunciationProfile]:
        with _LOCK:
            validated = cls.validate(profiles)
            path = cls.path()
            private_dir(path.parent)
            write_private(path, json.dumps(validated, ensure_ascii=False).encode())
            return validated

    @classmethod
    def enroll(cls, label: object, audio: object) -> dict[str, str | int | bool]:
        label = _validated_label(label).strip()
        sample = acoustic_features(audio)
        with _LOCK:
            profiles = cls.load()
            profile = next(
                (item for item in profiles if item["label"].casefold() == label.casefold()), None
            )
            if profile is None:
                profile = {"label": label, "version": FEATURE_VERSION, "samples": []}
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
    def delete(cls, label: str) -> list[PronunciationProfile]:
        with _LOCK:
            return cls.save(
                [item for item in cls.load() if item["label"].casefold() != label.casefold()]
            )

    @classmethod
    def match(
        cls, audio: object, *, threshold: float = 0.18, margin: float = 0.06
    ) -> PronunciationMatch | None:
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
    def correct_spans(
        cls,
        text: str,
        audio: object,
        alignment: object,
        *,
        budget: object = 2000,
        max_seconds: object = 2.0,
    ) -> SpanResult:
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
        return correct_spans(text, audio, alignment, cls.load, search_budget(budget, max_seconds))

    @classmethod
    def correct(cls, text: str, audio: object) -> str:
        # Never replace a sentence/paragraph using an isolated-phrase matcher.
        if not text.strip() or len(text.split()) > 5 or len(text) > 128:
            return text
        try:
            match = cls.match(audio)
        except ValueError:
            return text  # Long/no-speech captures are outside this matcher's contract.
        corrected: str = match["label"] if match else text
        return corrected
