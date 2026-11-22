"""Explicit local Whisper alignment of another engine's exact transcript.

Uses faster-whisper 1.2.1 WhisperModel.find_alignment (CTranslate2 DTW),
not a second recognition pass or evenly distributed timestamps. Optional model
must already exist locally. Native word metadata always takes precedence.
"""

import math
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Protocol, cast, runtime_checkable

import faster_whisper
import numpy as np
from faster_whisper import audio as whisper_audio
from faster_whisper import tokenizer as whisper_tokenizer
from numpy.typing import NDArray

from . import engines
from .config import Config
from .transcript_window import SAMPLE_RATE, Segment, TimedWord
from .word_alignment import align_words

MAX_SEGMENTS = 32
MAX_TOKENS = 440
BUDGET_SECONDS = 10
MIN_WORD_PROBABILITY = 0.2


@runtime_checkable
class _WhisperModel(Protocol):
    feature_extractor: Any
    model: Any
    hf_tokenizer: Any
    max_length: int

    def encode(self, features: object) -> object: ...

    def find_alignment(
        self, tokenizer: object, tokens: list[list[int]], encoded: object, frames: int
    ) -> list[list[object]]: ...


def _segment_samples(segment: Segment, frames: int) -> tuple[int, int] | None:
    start, end = segment.start, segment.end
    # Bounded comparison also rejects NaN, infinity, and integers too large to cast.
    if (
        type(start) not in (int, float)
        or type(end) not in (int, float)
        or not 0 <= start < end <= frames / SAMPLE_RATE + 0.001
        or end - start > 30
    ):
        return None
    return round(start * SAMPLE_RATE), min(frames, round(end * SAMPLE_RATE))


def _validated_word(item: object, duration: float, source: str) -> TimedWord:
    if not isinstance(item, dict) or not isinstance(word := item.get("word"), str):
        raise TypeError("aligner returned invalid word metadata")
    start, end, probability = (float(item[key]) for key in ("start", "end", "probability"))
    if (
        not all(math.isfinite(value) for value in (start, end, probability))
        or not 0 <= start <= end <= duration + 0.08
        or not 0 <= probability <= 1
        or (
            any(character.isalnum() for character in word)
            and (end <= start or probability < MIN_WORD_PROBABILITY)
        )
    ):
        raise ValueError("alignment probability or boundaries are uncertain")
    return {
        "word": word,
        "start": start,
        "end": min(end, duration),
        "probability": probability,
        "alignment_source": source,
    }


def _language(
    model: _WhisperModel, features: NDArray[np.generic], requested: str
) -> tuple[str, object | None]:
    language = requested.lower().split("-", 1)[0]
    if language == "auto" and model.model.is_multilingual:
        encoded = model.encode(whisper_audio.pad_or_trim(features))
        choices = model.model.detect_language(encoded)[0]
        if not choices or not 0.8 <= float(choices[0][1]) <= 1:
            raise ValueError("alignment language is uncertain")
        token = choices[0][0]
        if not isinstance(token, str) or not token.startswith("<|") or not token.endswith("|>"):
            raise ValueError("alignment language is uncertain")
        return token[2:-2], encoded
    if language == "auto":
        language = "en"
    if not model.model.is_multilingual and language != "en":
        raise ValueError("alignment model does not support selected language")
    return language, None


def _with_timing(segment: Segment, words: list[TimedWord], offset: float) -> Segment:
    return cast(
        "Segment",
        SimpleNamespace(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            words=[
                SimpleNamespace(
                    **{**word, "start": word["start"] + offset, "end": word["end"] + offset}
                )
                for word in words
            ],
            avg_logprob=getattr(segment, "avg_logprob", None),
        ),
    )


class WhisperForcedAligner:
    def __init__(self) -> None:
        self.model: object = None
        self.key: str | None = None

    def clear(self) -> None:
        self.model = self.key = None

    def _load(self, config: Config) -> object:
        selected = replace(
            config,
            model=config.pronunciation_aligner_model,
            engine="whisper",
            device="cpu",
            compute_type="int8",
            model_revision="main",
        )
        path = engines.resolve_whisper(selected)
        if self.key != path:
            self.clear()
            self.model = faster_whisper.WhisperModel(
                path,
                device="cpu",
                compute_type="int8",
                cpu_threads=2,
                num_workers=1,
                local_files_only=True,
            )
            self.key = path
        return self.model

    def _words(self, audio: NDArray[np.generic], text: str, config: Config) -> list[TimedWord]:
        model = cast("_WhisperModel", self._load(config))
        features = model.feature_extractor(audio)
        frames = features.shape[-1] - 1
        if not 0 < frames <= model.feature_extractor.nb_max_frames:
            raise ValueError("segment exceeds Whisper alignment context")
        language, encoded = _language(model, features, config.language)
        tokenizer = whisper_tokenizer.Tokenizer(
            model.hf_tokenizer, model.model.is_multilingual, "transcribe", language
        )
        # Condition like Whisper's generated transcription tokens. An unspaced
        # first-word token can be extremely unlikely even for matching speech.
        tokens = tokenizer.encode(" " + text.strip())
        if (
            not tokens
            or len(tokens) > min(MAX_TOKENS, model.max_length - len(tokenizer.sot_sequence) - 1)
            or any(token >= tokenizer.eot for token in tokens)
        ):
            raise ValueError("transcript exceeds safe alignment token budget")
        if encoded is None:
            encoded = model.encode(whisper_audio.pad_or_trim(features))
        aligned = model.find_alignment(tokenizer, [tokens], encoded, frames)
        if len(aligned) != 1 or not aligned[0]:
            raise ValueError("aligner returned no complete word metadata")
        duration = len(audio) / SAMPLE_RATE
        source = f"whisper-forced:{config.pronunciation_aligner_model}"
        words = [_validated_word(item, duration, source) for item in aligned[0]]
        align_words(text, words, duration=duration)
        return words

    def align(
        self, audio: NDArray[np.generic], segments: list[Segment], config: Config
    ) -> tuple[list[Segment], str]:
        """Align only real ASR segment bounds; keep text/native metadata unchanged.

        Context/token/segment limits bound native work. The inter-call deadline
        cannot interrupt a running CTranslate2 call; cancellation still suppresses
        delivery through the owning Transcriber's existing service contract.
        """
        if config.pronunciation_matching != "on" or not config.pronunciation_aligner_model:
            self.clear()
            return segments, ""
        deadline = time.monotonic() + BUDGET_SECONDS
        output: list[Segment] = []
        warnings: set[str] = set()
        attempted = 0
        for segment in segments:
            aligned = segment
            if getattr(segment, "words", None) or not segment.text.strip():
                output.append(segment)
                continue
            if attempted >= MAX_SEGMENTS or time.monotonic() >= deadline:
                warnings.add("optional Whisper alignment budget reached")
                output.append(segment)
                continue
            attempted += 1
            if (bounds := _segment_samples(segment, len(audio))) is None:
                warnings.add("ASR segment lacks usable bounds within Whisper's 30-second context")
                output.append(segment)
                continue
            first, last = bounds
            try:
                words = self._words(audio[first:last], segment.text.strip(), config)
                aligned = _with_timing(segment, words, first / SAMPLE_RATE)
            # Optional library failures must not discard ASR or expose transcript data.
            except Exception:  # allow: exception-handling  # noqa: BLE001
                warnings.add(
                    "optional Whisper alignment unavailable or uncertain; ASR text retained"
                )
            output.append(aligned)
        return output, "; ".join(sorted(warnings))
