"""Offline Transformers speech models loaded only from local artifacts."""

import importlib
import math
import os
import re
from types import SimpleNamespace
from typing import Protocol, Self, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from . import native_asr
from .config import Config
from .native_asr import FAMILIES, NativeWord
from .vocabulary import context_prompt
from .word_alignment import align_words


@runtime_checkable
class DecoderTensor(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...
    def __getitem__(self, index: object) -> Self: ...
    def tolist(self) -> object: ...


def strip_decoder_prompt(generated: DecoderTensor, prompt: DecoderTensor) -> DecoderTensor:
    """Discard verified decoder conditioning, including an optional prepended BOS."""
    count = prompt.shape[1]
    expected = prompt[0].tolist()
    for offset in (0, 1):
        if generated[0, offset : offset + count].tolist() == expected:
            return generated[:, offset + count :]
    raise ValueError("Speech decoder returned an unexpected conditioning prefix")


def native_token_words(
    text: str, tokens: object, duration: float, offset: float = 0
) -> list[NativeWord]:
    """Group exact native emission intervals; never split one token's interval.

    Transformers 5.22.2 Nemotron processors decode generated ``durations`` into
    seconds for emitted token chunks (processing_nemotron*_asr*.py ``decode``).
    Source: huggingface/transformers@v5.22.2, models/nemotron_asr_streaming.
    A word spans
    its first/last native token frames, as in NeMo; this is model alignment, not
    measured acoustic word boundaries. Incomplete/ambiguous renderings abstain.
    """
    emission = _token_ranges(tokens, duration)
    if emission is None:
        return []
    raw, ranges = emission
    if raw.strip() != text:
        return []
    words: list[NativeWord] = []
    index = 0
    for word in re.finditer(r"\S+", raw):
        while index < len(ranges) and ranges[index][1] <= word.start():
            index += 1
        first = index
        while index < len(ranges) and ranges[index][0] < word.end():
            index += 1
        start, end = ranges[first][2], ranges[index - 1][3]
        if start > end:
            return []
        words.append({"word": word[0], "start": start, "end": end})
    try:
        align_words(text, words, duration=duration)
    except ValueError:
        return []
    return [
        {"word": word["word"], "start": word["start"] + offset, "end": word["end"] + offset}
        for word in words
    ]


def _token_ranges(
    tokens: object, duration: float
) -> tuple[str, list[tuple[int, int, float, float]]] | None:
    if not isinstance(tokens, list) or len(tokens) > 10000:
        return None
    parts, ranges, cursor, previous_start, previous_end = [], [], 0, 0.0, 0.0
    for token in tokens:
        if not isinstance(token, dict) or not isinstance(token.get("token"), str):
            return None
        chunk = token["token"]
        start, end = token.get("start"), token.get("end")
        if (
            not isinstance(start, int | float)
            or not isinstance(end, int | float)
            or isinstance(start, bool)
            or isinstance(end, bool)
            or not math.isfinite(start)
            or not math.isfinite(end)
            or not 0 <= start <= end
            or start < previous_start
            or end < previous_end
            or end > duration + 0.08  # At most one native 80 ms encoder frame at the waveform edge.
            or len(chunk) > 1000
            or len(re.findall(r"\S+", chunk)) > 1
        ):
            return None
        parts.append(chunk)
        ranges.append((cursor, cursor + len(chunk), float(start), min(float(end), duration)))
        cursor += len(chunk)
        previous_start, previous_end = float(start), float(end)
    return "".join(parts), ranges


class TransformersBackend:
    def __init__(self, config: Config) -> None:
        path = native_asr.local_artifact(config)
        torch, device, dtype = native_asr.torch_runtime(config)
        try:
            module = importlib.import_module("transformers")
        except ImportError:
            raise RuntimeError("Install the transformers optional dependency extra") from None
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        family = native_asr.builtin_family(path / "config.json")
        if family is None:
            raise ValueError("Use a builtin Nemotron, Qwen3-ASR -hf, or Cohere ASR model")
        self.family = family
        loader = getattr(module, FAMILIES[family])
        self.processor = module.AutoProcessor.from_pretrained(
            str(path),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = (
            loader.from_pretrained(
                str(path),
                local_files_only=True,
                trust_remote_code=False,
                dtype=dtype,
            )
            .to(device)
            .eval()
        )
        self.torch = torch
        if self.processor.feature_extractor.sampling_rate != 16000:
            raise ValueError("Speech model must accept 16 kHz audio")

    def segments(
        self, samples: NDArray[np.generic], config: Config, *, presegmented: bool = False
    ) -> list[SimpleNamespace]:
        if self.family == "cohere_asr" and config.language == "auto":
            raise ValueError("Cohere Transcribe requires an explicit language")
        output = []
        self.words: list[NativeWord] = []
        spans = (
            [{"start": 0, "end": len(samples)}]
            if presegmented
            else native_asr.speech_spans(samples)
        )
        for span in spans:
            audio = samples[span["start"] : span["end"]]
            options = {"sampling_rate": 16000, "return_tensors": "pt"}
            language = self._language(config.language)
            if self.family == "qwen3_asr":
                inputs = self.processor.apply_transcription_request(
                    audio=audio,
                    language=language,
                    prompt=context_prompt(config) or None,
                    **options,
                )
            else:
                if language is not None:
                    options["language"] = language
                inputs = self.processor(audio, **options)
            inputs = inputs.to(self.model.device, dtype=self.model.dtype)
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=config.asr_max_tokens)
            durations = getattr(generated, "durations", None)
            generated = getattr(generated, "sequences", generated)
            token_times = None
            if self.family == "qwen3_asr":
                generated = generated[:, inputs["input_ids"].shape[1] :]
                text = self.processor.decode(generated, return_format="transcription_only")
            else:
                if self.family == "cohere_asr" and "decoder_input_ids" in inputs:
                    generated = strip_decoder_prompt(generated, inputs["decoder_input_ids"])
                timed = (
                    self.family in ("nemotron3_5_asr", "nemotron_asr_streaming")
                    and config.pronunciation_matching == "on"
                    and durations is not None
                )
                decoded = self.processor.decode(
                    generated,
                    skip_special_tokens=True,
                    **({"durations": durations} if timed else {}),
                )
                if timed:
                    text, token_times = decoded
                else:
                    text = decoded
            text = native_asr.transcript_text(text)
            words = (
                native_token_words(text, token_times[0], len(audio) / 16000, span["start"] / 16000)
                if isinstance(token_times, list) and len(token_times) == 1
                else []
            )
            self.words.extend(words)
            if text:
                output.append(
                    SimpleNamespace(
                        start=span["start"] / 16000,
                        end=span["end"] / 16000,
                        text=text,
                        words=words,
                    )
                )
        return output

    def _language(self, language: str) -> str | None:
        if self.family == "nemotron3_5_asr":
            return native_asr.nemotron_language(self.processor, language)
        if self.family == "nemotron_asr_streaming":
            if language not in ("en", "auto"):
                raise ValueError("Nemotron Speech English supports English only")
            return None
        return None if language == "auto" else language
