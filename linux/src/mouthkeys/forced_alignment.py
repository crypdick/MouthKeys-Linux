"""Explicit local Whisper alignment of another engine's exact transcript.

Uses faster-whisper 1.2.1 ``WhisperModel.find_alignment`` (CTranslate2 DTW),
not a second recognition pass or evenly distributed timestamps. Optional model
must already exist locally. Native word metadata always takes precedence.
"""

import math
import time
from dataclasses import replace
from types import SimpleNamespace

from .word_alignment import align_words

MAX_SEGMENTS = 32
MAX_TOKENS = 440
BUDGET_SECONDS = 10
MIN_WORD_PROBABILITY = 0.2


class WhisperForcedAligner:
    def __init__(self):
        self.model = None
        self.key = None

    def clear(self):
        self.model = self.key = None

    def _load(self, config):
        from faster_whisper import WhisperModel

        from .engines import resolve_whisper

        selected = replace(
            config,
            model=config.pronunciation_aligner_model,
            engine="whisper",
            device="cpu",
            compute_type="int8",
            model_revision="main",
        )
        path = resolve_whisper(selected)
        if self.key != path:
            self.clear()
            self.model = WhisperModel(
                path,
                device="cpu",
                compute_type="int8",
                cpu_threads=2,
                num_workers=1,
                local_files_only=True,
            )
            self.key = path
        return self.model

    def _words(self, audio, text, config):
        from faster_whisper.audio import pad_or_trim
        from faster_whisper.tokenizer import Tokenizer

        model = self._load(config)
        features = model.feature_extractor(audio)
        frames = features.shape[-1] - 1
        if not 0 < frames <= model.feature_extractor.nb_max_frames:
            raise ValueError("segment exceeds Whisper alignment context")
        encoded = None
        language = config.language.lower().split("-", 1)[0]
        if language == "auto":
            if model.model.is_multilingual:
                encoded = model.encode(pad_or_trim(features))
                choices = model.model.detect_language(encoded)[0]
                if not choices or float(choices[0][1]) < 0.8:
                    raise ValueError("alignment language is uncertain")
                token = choices[0][0]
                language = token[2:-2] if token.startswith("<|") and token.endswith("|>") else ""
            else:
                language = "en"
        if not model.model.is_multilingual and language != "en":
            raise ValueError("alignment model does not support selected language")
        tokenizer = Tokenizer(
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
            encoded = model.encode(pad_or_trim(features))
        aligned = model.find_alignment(tokenizer, [tokens], encoded, frames)
        if len(aligned) != 1 or not aligned[0]:
            raise ValueError("aligner returned no complete word metadata")
        duration = len(audio) / 16000
        words = []
        for item in aligned[0]:
            word = str(item["word"])
            start, end, probability = (
                float(item["start"]),
                float(item["end"]),
                float(item["probability"]),
            )
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
            words.append(
                {
                    "word": word,
                    "start": start,
                    "end": min(end, duration),
                    "probability": probability,
                    "alignment_source": f"whisper-forced:{config.pronunciation_aligner_model}",
                }
            )
        align_words(text, words, duration=duration)
        return words

    def align(self, audio, segments, config):
        """Align only real ASR segment bounds; keep text/native metadata unchanged.

        Context/token/segment limits bound native work. The inter-call deadline
        cannot interrupt a running CTranslate2 call; cancellation still suppresses
        delivery through the owning Transcriber's existing service contract.
        """
        if config.pronunciation_matching != "on" or not config.pronunciation_aligner_model:
            self.clear()
            return segments, ""
        deadline = time.monotonic() + BUDGET_SECONDS
        output, warnings, attempted = [], set(), 0
        for segment in segments:
            if getattr(segment, "words", None) or not segment.text.strip():
                output.append(segment)
                continue
            if attempted >= MAX_SEGMENTS or time.monotonic() >= deadline:
                warnings.add("optional Whisper alignment budget reached")
                output.append(segment)
                continue
            attempted += 1
            start, end = segment.start, segment.end
            if (
                type(start) not in (int, float)
                or type(end) not in (int, float)
                or not math.isfinite(start)
                or not math.isfinite(end)
                or not 0 <= start < end <= len(audio) / 16000 + 0.001
                or end - start > 30
            ):
                warnings.add("ASR segment lacks usable bounds within Whisper's 30-second context")
                output.append(segment)
                continue
            first, last = round(start * 16000), min(len(audio), round(end * 16000))
            try:
                words = self._words(audio[first:last], segment.text.strip(), config)
                aligned = SimpleNamespace(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    words=[
                        SimpleNamespace(
                            **{
                                **word,
                                "start": word["start"] + first / 16000,
                                "end": word["end"] + first / 16000,
                            }
                        )
                        for word in words
                    ],
                    avg_logprob=getattr(segment, "avg_logprob", None),
                )
            except Exception:
                # Optional library/model failure must not discard primary ASR text.
                # Do not expose exceptions that may echo supplied transcript data.
                warnings.add(
                    "optional Whisper alignment unavailable or uncertain; ASR text retained"
                )
                aligned = segment
            output.append(aligned)
        return output, "; ".join(sorted(warnings))
