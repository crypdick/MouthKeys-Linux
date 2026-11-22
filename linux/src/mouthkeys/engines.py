"""Optional Linux ONNX engines and explicit, app-owned model management."""

import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

from .benchmark_reports import measurements
from .config import Config, cache_dir
from .model_cache import (
    READY,
    delete_model,
    managed_path,
    onnx_path,
    usable_whisper_snapshot,
    whisper_cache_index,
    whisper_repositories,
    whisper_revision,
)
from .model_progress import DownloadProgress, progress_tqdm

__all__ = [
    "ONNX_MODELS",
    "READY",
    "delete_model",
    "download_onnx",
    "load_onnx",
    "managed_path",
    "model_catalog",
    "onnx_module",
    "onnx_path",
    "onnx_quantization",
    "resolve_whisper",
    "usable_whisper_snapshot",
]


# onnx-asr 0.12 supports these converted NeMo models on Linux CPUs.
ONNX_MODELS = {
    "nemo-parakeet-tdt-0.6b-v3": ("Parakeet TDT v3", "Multilingual"),
    "nemo-parakeet-tdt-0.6b-v2": ("Parakeet TDT v2", "English"),
    "nemo-canary-1b-v2": ("Canary 1B v2", "Multilingual"),
    "istupakov/canary-180m-flash-onnx": ("Canary 180M Flash", "Multilingual"),
}


def onnx_module() -> ModuleType:
    try:
        import onnx_asr
    except ImportError:
        raise RuntimeError("Install the onnx extra: uv tool install './linux[gui,onnx]'") from None
    module: ModuleType = onnx_asr
    return module


def resolve_whisper(config: Config) -> str:
    """Resolve and preflight locally before faster-whisper can fetch a tokenizer."""
    path = Path(config.model).expanduser()
    if not path.is_dir():
        from faster_whisper.utils import download_model

        try:
            path = Path(
                download_model(
                    config.model,
                    cache_dir=str(cache_dir() / "models"),
                    revision=config.model_revision,
                    local_files_only=True,
                )
            )
        except (OSError, ValueError):
            raise RuntimeError(
                "Whisper model missing locally; run mouthkeys download first"
            ) from None
    if not usable_whisper_snapshot(path):
        raise RuntimeError(
            "Whisper snapshot incomplete: model.bin, config.json, tokenizer.json, "
            "and vocabulary files are required; explicitly download a complete model"
        )
    return str(path)


def _download_onnx_revision(
    model: str,
    path: Path,
    quantization: str | None,
    revision: str,
    progress: Callable[[DownloadProgress], None] | None,
) -> None:
    # onnx-asr 0.12's downloader has no revision parameter. Use its pinned
    # resolver's file manifest, but perform the explicit Hub request ourselves.
    from huggingface_hub import hf_hub_download, snapshot_download
    from onnx_asr.loader import create_asr_resolver
    from onnx_asr.resolver import model_repos

    repository = model_repos.get(model, model)
    hf_hub_download(repository, "config.json", revision=revision, local_dir=path)
    resolver = create_asr_resolver(model, path, offline=True)
    files = list(resolver.model_type._get_model_files(quantization).values())
    files += [file.removeprefix("**/") for file in files if file.startswith("**/")]
    patterns = ["config.json", "config.yaml", *files]
    patterns += [
        str(Path(file).with_suffix(".onnx?data")) for file in files if Path(file).suffix == ".onnx"
    ]
    snapshot_download(
        repository,
        revision=revision,
        local_dir=path,
        allow_patterns=patterns,
        tqdm_class=progress_tqdm(progress) if progress else None,
    )


def download_onnx(
    model: str,
    compute_type: str = "default",
    progress: Callable[[DownloadProgress], None] | None = None,
    *,
    revision: str = "main",
) -> str:
    """Only this explicit action is allowed to fetch ONNX artifacts."""
    module = onnx_module()
    path = onnx_path(model, revision)
    quantization = onnx_quantization(compute_type)
    if progress:
        progress({"phase": "speech model", "completed": 0, "total": 2, "unit": "stage"})
    (path / READY).unlink(missing_ok=True)
    if revision != "main":
        _download_onnx_revision(model, path, quantization, revision, progress)
    module.load_model(model, path, quantization=quantization, offline=revision != "main")
    vad_path = managed_path("silero")
    if progress:
        progress({"phase": "voice activity model", "completed": 1, "total": 2, "unit": "stage"})
    module.load_vad("silero", vad_path, offline=False)
    # A cancelled Hub transfer can leave a directory and some model files behind.
    # Only advertise a managed model after both engine and VAD finished loading.
    (vad_path / READY).touch(mode=0o600)
    (path / READY).touch(mode=0o600)
    if progress:
        progress({"phase": "ready", "completed": 2, "total": 2, "unit": "stage"})
    return str(path)


def onnx_quantization(compute_type: str) -> str | None:
    if compute_type not in ("default", "auto", "float32", "int8"):
        raise ValueError("ONNX compute type must be default, auto, float32, or int8")
    return "int8" if compute_type == "int8" else None


def load_onnx(config: Config) -> object:
    module = onnx_module()
    path = Path(config.model).expanduser()
    local = path.is_dir()
    if not local:
        path = onnx_path(config.model, config.model_revision)
    if (not local and not (path / READY).is_file()) or not (
        managed_path("silero") / READY
    ).is_file():
        raise RuntimeError("ONNX model or Silero VAD missing; run mouthkeys download first")
    providers = ["CPUExecutionProvider"]
    if config.device == "cuda":
        import onnxruntime

        if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
            raise RuntimeError("CUDA ONNX requires onnxruntime-gpu; choose CPU or install it")
        providers.insert(0, "CUDAExecutionProvider")
    quantization = onnx_quantization(config.compute_type)
    model = module.load_model(
        None if local else config.model,
        path,
        quantization=quantization,
        providers=providers,
        offline=True,
    )
    vad = module.load_vad("silero", managed_path("silero"), offline=True)
    return model.with_vad(vad, max_speech_duration_s=25)


def _model_catalog(engine: str = "whisper", *, revision: str = "main") -> list[dict[str, Any]]:
    """Local-only catalog; no Hub queries or downloads."""
    if engine == "onnx":
        return [
            {
                "model": name,
                "name": label,
                "languages": languages,
                "engine": engine,
                "revision": revision,
                "installed": (onnx_path(name, revision) / READY).is_file(),
                "path": str(onnx_path(name, revision)),
                "native_streaming": False,
                "vocabulary": False,
                "vocabulary_method": (
                    "unavailable; use NeMo TDT for native boosting"
                    if "parakeet" in name
                    else "unavailable"
                ),
                "word_timing": False,
                "source": "https://github.com/istupakov/onnx-asr",
            }
            for name, (label, languages) in ONNX_MODELS.items()
        ]
    if engine in ("transformers", "nemo"):
        from .native_asr import NEMO_MODELS, TRANSFORMERS_MODELS, artifact_path, effective_revision

        models = TRANSFORMERS_MODELS if engine == "transformers" else NEMO_MODELS
        return [
            {
                "model": name,
                "name": entry["name"],
                "languages": entry["languages"],
                "engine": engine,
                "revision": effective_revision(name, revision, engine),
                "installed": (artifact_path(name, engine, revision) / READY).is_file(),
                "path": str(artifact_path(name, engine, revision)),
                "weight_bytes": entry["weight_bytes"],
                "native_streaming": engine == "nemo" and entry.get("native_streaming", True),
                "vocabulary": engine == "nemo" or entry.get("family") == "qwen3_asr",
                "word_timing": engine == "nemo"
                or entry.get("family") in ("nemotron3_5_asr", "nemotron_asr_streaming"),
                "source": f"https://huggingface.co/{name}",
                "vocabulary_method": (
                    "weighted context graph"
                    if engine == "nemo"
                    else "context prompt"
                    if entry.get("family") == "qwen3_asr"
                    else "unavailable"
                ),
            }
            for name, entry in models.items()
        ]
    if engine != "whisper":
        raise ValueError("Unknown engine")
    installed = whisper_cache_index()
    rows = []
    for name, repository in whisper_repositories().items():
        cached = installed.get(repository)
        snapshot = whisper_revision(cached, revision)
        path = (
            str(cached.repo_path)
            if cached is not None
            and snapshot is not None
            and usable_whisper_snapshot(snapshot.snapshot_path)
            else ""
        )
        rows.append(
            {
                "model": name,
                "name": name,
                "languages": "English"
                if name.endswith(".en") or "distil" in name
                else "Multilingual",
                "engine": engine,
                "revision": revision,
                "installed": bool(path),
                "path": path,
                "source": f"https://huggingface.co/{repository}",
                "native_streaming": False,
                "vocabulary": True,
                "vocabulary_method": "bounded textual hotword hint",
                "word_timing": True,
            }
        )
    return rows


def model_catalog(
    engine: str = "whisper",
    *,
    revision: str = "main",
    measurement_config: Config | None = None,
    measurement_context: Sequence[object] | None = None,
) -> list[dict[str, Any]]:
    """Report capabilities and explicit unknowns, not Apple-hardware ratings.

    Parameter-memory estimates use the nominal model size in publisher names
    and four bytes per float32 parameter. They exclude inference workspace,
    activations and caches; archive size is not a runtime RAM measurement.
    """
    from .benchmark import matching_measurement

    measured = measurements()
    rows = _model_catalog(engine, revision=revision)
    if (
        measurement_config is not None
        and not any(row["model"] == measurement_config.model for row in rows)
        and Path(measurement_config.model).expanduser().exists()
    ):
        rows.append(
            {
                "model": measurement_config.model,
                "name": "Custom model: " + Path(measurement_config.model).name,
                "languages": "Custom checkpoint",
                "engine": engine,
                "revision": revision,
                "installed": True,
                "path": measurement_config.model,
            }
        )
    for row in rows:
        row["speed_note"] = "Not benchmarked on this hardware"
        row["accuracy_note"] = "Not evaluated for your language/audio"
        if engine in ("transformers", "nemo"):
            name = row["model"].lower()
            parameters = next(
                (
                    count
                    for token, count in (("120m", 120), ("0.6b", 600), ("1.7b", 1700))
                    if token in name
                ),
                None,
            )
            row["memory_note"] = (
                f"~{parameters * 4 / 1000:g} GB float32 parameters "
                "+ workspace/caches; RAM unmeasured"
                if parameters
                else "Full-precision CPU weights + workspace/caches; RAM unmeasured"
            )
        else:
            row["memory_note"] = "Precision/model/device dependent; RAM unmeasured"
        report = (
            matching_measurement(
                replace(
                    measurement_config,
                    engine=engine,
                    model=row["model"],
                    model_revision=row.get("revision", revision),
                ),
                measured,
                measurement_context,
            )
            if measurement_config
            else None
        )
        if report:
            _apply_measurement(row, report)
    return rows


def _apply_measurement(row: dict[str, Any], report: dict[str, Any]) -> None:
    settings = f"{report['device']}/{report['compute_type']} requested"
    row["measurement"] = report
    row["speed_note"] = (
        f"{report['real_time_factor']:.3f}× real time; {settings}; "
        f"fixture {report['fixture_sha256'][:8]} ({report['timestamp_utc'][:10]})"
    )
    scores = report.get("accuracy")
    wer = scores.get("wer") if isinstance(scores, dict) else None
    if (
        isinstance(wer, int | float)
        and not isinstance(wer, bool)
        and 0 <= wer <= sys.float_info.max / 100
    ):
        row["accuracy_note"] = f"{wer:.1%} WER on fixture {report['fixture_sha256'][:8]} only"
    rss = report.get("sampled_peak_tree_rss_bytes")
    if type(rss) is int and 0 < rss <= sys.float_info.max:
        row["memory_note"] += f"; sampled process-tree RSS {rss / 2**20:.0f} MiB"
