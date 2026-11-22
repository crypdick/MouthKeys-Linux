"""Local discovery for Settings choices; never load or download a model."""

import subprocess
from collections.abc import Collection
from contextlib import suppress

from faster_whisper import available_models
from faster_whisper.tokenizer import _LANGUAGE_CODES
from PySide6.QtCore import QLocale

from . import desktop, desktop_input
from .engines import ONNX_MODELS
from .native_asr import NEMO_MODELS, TRANSFORMERS_MODELS

ENGINE_MODELS: dict[str, Collection[str]] = {
    "onnx": ONNX_MODELS,
    "nemo": NEMO_MODELS,
    "transformers": TRANSFORMERS_MODELS,
}

MODEL_ALIASES = {"large": "large-v3", "turbo": "large-v3-turbo"}
# Approximate accuracy for short English dictation, strongest first. Language and
# audio can change the ranking; .en and distilled models only support English.
# See https://huggingface.co/distil-whisper/distil-large-v3.5 for comparisons.
MODEL_ACCURACY_ORDER = (
    "large-v3",
    "distil-large-v3.5",
    "large-v3-turbo",
    "distil-large-v3",
    "large-v2",
    "distil-large-v2",
    "large-v1",
    "medium.en",
    "medium",
    "distil-medium.en",
    "small.en",
    "small",
    "distil-small.en",
    "base.en",
    "base",
    "tiny.en",
    "tiny",
)


def model_choices(engine: str) -> list[str]:
    if engine != "whisper":
        return list(ENGINE_MODELS[engine])
    available = {MODEL_ALIASES.get(name, name) for name in available_models()}
    ranked = [name for name in MODEL_ACCURACY_ORDER if name in available]
    return ranked + sorted(available.difference(MODEL_ACCURACY_ORDER))


def language_choices() -> list[tuple[str, str]]:
    # Match the tokenizer in our installed, version-bounded transcription engine.
    names = {"haw": "Hawaiian", "ht": "Haitian Creole", "jw": "Javanese", "yue": "Cantonese"}
    choices = []
    for code in _LANGUAGE_CODES:
        locale = QLocale(code)
        name = names.get(code, QLocale.languageToString(locale.language()))
        choices.append((f"{name if name != 'C' else code} ({code})", code))
    return [("Detect automatically", "auto"), *sorted(choices)]


def audio_sources() -> list[str]:
    if desktop_input.session() != "headless":
        with suppress(OSError, subprocess.SubprocessError):
            return ["default", *desktop.source_names()]
    return ["default"]
