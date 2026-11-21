"""Linux native speech backends using installed packages and explicit local artifacts."""

import json
import os
import tarfile
from collections.abc import Callable, Iterable
from pathlib import Path
from types import ModuleType
from typing import TypedDict

from .config import Config
from .engines import READY, managed_path
from .model_progress import DownloadProgress, progress_tqdm


class NativeWord(TypedDict):
    word: str
    start: float
    end: float


class SpeechSpan(TypedDict):
    start: int
    end: int


# These names/classes are documented by stable Transformers 5.22.2 and NVIDIA NeMo 3.
# No remote Python model code is imported from a downloaded model repository.
TRANSFORMERS_MODELS = {
    "nvidia/nemotron-3.5-asr-streaming-0.6b": {
        "name": "Nemotron 3.5 ASR (offline)",
        "languages": "40 language-locales",
        "family": "nemotron3_5_asr",
        "revision": "main",
        "weight_bytes": 2552062944,
    },
    "nvidia/nemotron-speech-streaming-en-0.6b": {
        "name": "Nemotron Speech English",
        "languages": "English",
        "family": "nemotron_asr_streaming",
        "revision": "main",
        "weight_bytes": 2472413604,
    },
    "Qwen/Qwen3-ASR-0.6B-hf": {
        "name": "Qwen3 ASR 0.6B",
        "languages": "30 languages + Chinese dialects",
        "family": "qwen3_asr",
        "revision": "main",
        "weight_bytes": 1564928088,
    },
    "Qwen/Qwen3-ASR-1.7B-hf": {
        "name": "Qwen3 ASR 1.7B",
        "languages": "30 languages + Chinese dialects",
        "family": "qwen3_asr",
        "revision": "main",
        "weight_bytes": 4076193080,
    },
    "CohereLabs/cohere-transcribe-03-2026": {
        "name": "Cohere Transcribe (gated)",
        "languages": "14 languages; select explicitly",
        "family": "cohere_asr",
        "revision": "refs/pr/6",
        "weight_bytes": 4131862976,
    },
}
NEMO_MODELS = {
    # NVIDIA's native TDT archives expose NeMo word timing and weighted graphs.
    # Their full-context encoders are offline, not cache-aware streaming models.
    "nvidia/parakeet-tdt-0.6b-v2": {
        "name": "Parakeet TDT v2 (NeMo, native vocabulary)",
        "languages": "English",
        "filename": "parakeet-tdt-0.6b-v2.nemo",
        "revision": "main",
        "weight_bytes": 2472222720,
        "native_streaming": False,
    },
    "nvidia/parakeet-tdt-0.6b-v3": {
        "name": "Parakeet TDT v3 (NeMo, native vocabulary)",
        "languages": "25 languages (automatic detection)",
        "filename": "parakeet-tdt-0.6b-v3.nemo",
        "revision": "main",
        "weight_bytes": 2509332480,
        "native_streaming": False,
    },
    "nvidia/parakeet_realtime_eou_120m-v1": {
        "name": "Parakeet Flash / Realtime EOU 120M",
        "languages": "English",
        "filename": "parakeet_realtime_eou_120m-v1.nemo",
        "revision": "main",
        "weight_bytes": 460062720,
    },
    "nvidia/nemotron-3.5-asr-streaming-0.6b": {
        "name": "Nemotron 3.5 ASR (NeMo)",
        "languages": "40 language-locales",
        "filename": "nemotron-3.5-asr-streaming-0.6b.nemo",
        "revision": "main",
        "weight_bytes": 2368284501,
    },
}
FAMILIES = {
    "nemotron3_5_asr": "AutoModelForRNNT",
    "nemotron_asr_streaming": "AutoModelForRNNT",
    "qwen3_asr": "AutoModelForMultimodalLM",
    "cohere_asr": "CohereAsrForConditionalGeneration",
}


def uses_native_streaming(config: Config) -> bool:
    """Full-context Parakeet TDT uses provisional windows, never streaming caches."""
    return (
        config.engine == "nemo"
        and getattr(config, "native_streaming", "on") == "on"
        and bool(NEMO_MODELS.get(config.model, {}).get("native_streaming", True))
    )


def supports_native_streaming(model: object) -> bool:
    """Cache methods alone do not make a full-context Conformer streaming-safe."""
    context = getattr(getattr(model, "encoder", None), "att_context_size", None)
    return bool(
        callable(getattr(model, "conformer_stream_step", None))
        and isinstance(context, (list, tuple))
        and len(context) == 2
        and type(context[1]) is int
        and context[1] >= 0
    )


def effective_revision(model: str, revision: str = "main", engine: str = "transformers") -> str:
    catalog = TRANSFORMERS_MODELS if engine == "transformers" else NEMO_MODELS
    # Mainline Cohere weights predate its builtin Transformers integration. The
    # documented compatibility revision is displayed in the model catalog.
    return str(catalog.get(model, {}).get("revision", "main")) if revision == "main" else revision


def artifact_path(model: str, engine: str, revision: str = "main") -> Path:
    return managed_path(f"{model}@{effective_revision(model, revision, engine)}", engine)


def download_native(
    model: str,
    engine: str,
    revision: str = "main",
    progress: Callable[[DownloadProgress], None] | None = None,
) -> str:
    from huggingface_hub import snapshot_download

    if engine not in ("transformers", "nemo"):
        raise ValueError("Unknown native speech engine")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    revision = effective_revision(model, revision, engine)
    path = artifact_path(model, engine, revision)
    (path / READY).unlink(missing_ok=True)
    patterns = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken"]
    if engine == "nemo":
        patterns = [str(NEMO_MODELS.get(model, {}).get("filename", "*.nemo"))]
    tqdm_class = progress_tqdm(progress) if progress else None
    try:
        snapshot_download(
            model, revision=revision, local_dir=path, allow_patterns=patterns, tqdm_class=tqdm_class
        )
    except Exception as exc:
        if type(exc).__name__ in ("GatedRepoError", "RepositoryNotFoundError"):
            raise RuntimeError(
                "Model access unavailable. Accept its Hugging Face conditions and configure "
                "HF_TOKEN "
                "before requesting the download again."
            ) from None
        raise
    if not any(path.glob("*.nemo" if engine == "nemo" else "*.safetensors")):
        raise ValueError("Downloaded repository has no supported model weights")
    if progress:
        progress(
            {"phase": "verifying model artifacts", "completed": 0, "total": 1, "unit": "stage"}
        )
    validate_artifacts(path, engine)
    (path / READY).touch(mode=0o600)
    if progress:
        progress({"phase": "ready", "completed": 1, "total": 1, "unit": "stage"})
    return str(path)


def _validate_nemo_archive(path: Path) -> None:
    import yaml

    artifacts = [path] if path.is_file() else list(path.glob("*.nemo"))
    if len(artifacts) != 1:
        raise ValueError("NeMo repository must contain exactly one model archive")
    with tarfile.open(artifacts[0], "r:*") as archive:
        members = {member.name.removeprefix("./"): member for member in archive}
        member = members.get("model_config.yaml")
        if member is None or not member.isfile() or member.size > 1024 * 1024:
            raise ValueError("NeMo model configuration missing or invalid")
        source = archive.extractfile(member)
        if source is None:
            raise ValueError("NeMo model configuration could not be read")
        with source:
            config = yaml.safe_load(source)
        if not any(name.endswith("model_weights.ckpt") for name in members):
            raise ValueError("NeMo model weights missing")
        _validate_archive_config(config, members)


def _validate_archive_config(config: object, members: dict[str, tarfile.TarInfo]) -> None:
    seen: set[int] = set()
    pending = [config]
    while pending:
        node = pending.pop()
        if isinstance(node, str) and node.startswith("nemo:"):
            name = node[5:]
            if name not in members or not members[name].isfile():
                raise ValueError("NeMo tokenizer or auxiliary artifact is missing")
        if not isinstance(node, dict | list) or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, list):
            pending.extend(node)
            continue
        for key, child in node.items():
            if key in ("target", "_target_") and (
                not isinstance(child, str) or not child.startswith(("nemo.", "torch."))
            ):
                raise ValueError("NeMo archive requests a non-framework Python target")
            pending.append(child)


def validate_artifacts(path: Path, engine: str) -> None:
    """Check companion files before advertising an explicit download as installed."""
    if engine == "nemo":
        _validate_nemo_archive(path)
        return
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError("Native model config.json missing")
    if builtin_family(config_path) is None:
        raise ValueError("Downloaded model requires an unsupported or remote-code architecture")
    if not any(
        (path / name).is_file() for name in ("processor_config.json", "preprocessor_config.json")
    ):
        raise ValueError("Native model audio processor configuration missing")
    if not any(
        (path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")
    ):
        raise ValueError("Native model tokenizer missing")
    for index in path.glob("*.safetensors.index.json"):
        contents = json.loads(index.read_text())
        mapping = contents.get("weight_map") if isinstance(contents, dict) else None
        if (
            not isinstance(mapping, dict)
            or not mapping
            or any(
                not isinstance(name, str) or Path(name).name != name or not (path / name).is_file()
                for name in mapping.values()
            )
        ):
            raise ValueError("Native model weight shard missing or invalid")


def builtin_family(config_path: Path) -> str | None:
    details = json.loads(config_path.read_text())
    family = details.get("model_type") if isinstance(details, dict) else None
    return family if isinstance(family, str) and family in FAMILIES else None


def local_artifact(config: Config) -> Path:
    path = Path(config.model).expanduser()
    if path.is_dir() or (config.engine == "nemo" and path.is_file()):
        return path
    path = artifact_path(config.model, config.engine, getattr(config, "model_revision", "main"))
    if not (path / READY).is_file():
        raise RuntimeError("Speech model missing; explicitly download it with mouthkeys download")
    return path


def torch_runtime(config: Config) -> tuple[ModuleType, str, object]:
    try:
        import torch
    except ImportError:
        raise RuntimeError(f"Install the {config.engine} optional dependency extra") from None
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; choose CPU or install a CUDA-enabled PyTorch build"
        )
    precision = config.compute_type
    if precision not in ("default", "auto", "float32", "float16", "bfloat16"):
        raise ValueError(
            "Native speech compute type must be default, auto, float32, float16, or bfloat16"
        )
    dtype = (
        "auto"
        if precision == "default"
        else getattr(
            torch,
            ("float16" if device == "cuda" else "float32") if precision == "auto" else precision,
        )
    )
    return torch, device, dtype


def speech_spans(samples: object) -> list[SpeechSpan]:
    """Reuse packaged offline VAD to bound model input, preserving speech timestamps."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    spans: list[SpeechSpan] = get_speech_timestamps(
        samples,
        VadOptions(max_speech_duration_s=25, min_silence_duration_ms=500, speech_pad_ms=200),
    )
    return spans


def transcript_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(transcript_text(item) for item in value).strip()
    if not isinstance(value, str):
        raise ValueError("Speech engine returned no text")
    return value.strip()


def nemotron_language(processor: object, language: str) -> str:
    if language == "auto":
        return "auto"
    # Processor validates locale keys. Stable release exposes this map in its
    # feature extractor configuration; fall back only to documented defaults.
    locales = getattr(processor, "language_to_id", None)
    if locales is None:
        locales = getattr(processor, "lang_to_id", None)
    if isinstance(locales, Iterable) and locales:
        if language in locales:
            return language
        matches = [
            locale
            for locale in locales
            if isinstance(locale, str) and locale.lower().startswith(language.lower() + "-")
        ]
        if matches:
            return matches[0]
    default_locales = {
        "en": "en-US",
        "es": "es-ES",
        "fr": "fr-FR",
        "pt": "pt-PT",
        "de": "de-DE",
        "it": "it-IT",
        "ja": "ja-JP",
        "ko": "ko-KR",
        "zh": "zh-CN",
        "ru": "ru-RU",
        "ar": "ar-AR",
        "hi": "hi-IN",
        "vi": "vi-VN",
        "uk": "uk-UA",
        "nl": "nl-NL",
        "tr": "tr-TR",
        "pl": "pl-PL",
        "sv": "sv-SE",
        "cs": "cs-CZ",
        "nb": "nb-NO",
        "da": "da-DK",
        "bg": "bg-BG",
        "fi": "fi-FI",
        "hr": "hr-HR",
        "sk": "sk-SK",
        "hu": "hu-HU",
        "ro": "ro-RO",
        "et": "et-EE",
        "el": "el-GR",
        "lt": "lt-LT",
        "lv": "lv-LV",
        "mt": "mt-MT",
        "sl": "sl-SI",
        "he": "he-IL",
        "th": "th-TH",
        "nn": "nn-NO",
    }
    return default_locales.get(language, language)
