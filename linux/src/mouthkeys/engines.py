"""Optional Linux ONNX engines and explicit, app-owned model management."""

import hashlib
import shutil
from pathlib import Path

from .config import cache_dir

READY = ".mouthkeys-ready"

# onnx-asr 0.12 supports these converted NeMo models on Linux CPUs.
ONNX_MODELS = {
    "nemo-parakeet-tdt-0.6b-v3": ("Parakeet TDT v3", "Multilingual"),
    "nemo-parakeet-tdt-0.6b-v2": ("Parakeet TDT v2", "English"),
    "nemo-canary-1b-v2": ("Canary 1B v2", "Multilingual"),
    "istupakov/canary-180m-flash-onnx": ("Canary 180M Flash", "Multilingual"),
}


def managed_path(model, engine="onnx"):
    """Never interpret a model ID as a filesystem path when deleting cache data."""
    digest = hashlib.sha256(model.encode()).hexdigest()[:24]
    return cache_dir() / "models" / engine / digest


def onnx_module():
    try:
        import onnx_asr
    except ImportError:
        raise RuntimeError("Install the onnx extra: uv tool install './linux[gui,onnx]'") from None
    return onnx_asr


def onnx_path(model, revision="main"):
    return managed_path(model if revision == "main" else f"{model}@{revision}")


def usable_whisper_snapshot(path):
    """Read-only completeness check, including the tokenizer's offline dependency."""
    path = Path(path)
    required = [path / name for name in ("model.bin", "config.json", "tokenizer.json")]
    vocabulary = [
        item for name in ("vocabulary.json", "vocabulary.txt") if (item := path / name).is_file()
    ]
    return bool(vocabulary) and all(
        item.is_file() and item.stat().st_size > 0 for item in [*required, *vocabulary]
    )


def resolve_whisper(config):
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


def _download_onnx_revision(model, path, quantization, revision, progress):
    # onnx-asr 0.12's downloader has no revision parameter. Use its pinned
    # resolver's file manifest, but perform the explicit Hub request ourselves.
    from huggingface_hub import hf_hub_download, snapshot_download
    from onnx_asr.loader import create_asr_resolver
    from onnx_asr.resolver import model_repos

    from .model_progress import progress_tqdm

    repository = model_repos.get(model, model)
    hf_hub_download(repository, "config.json", revision=revision, local_dir=path)
    resolver = create_asr_resolver(model, path, offline=True)
    files = list(resolver.model_type._get_model_files(quantization).values())
    files += [file.removeprefix("**/") for file in files if file.startswith("**/")]
    patterns = ["config.json", "config.yaml", *files]
    patterns += [
        str(Path(file).with_suffix(".onnx?data")) for file in files if Path(file).suffix == ".onnx"
    ]
    options = {"tqdm_class": progress_tqdm(progress)} if progress else {}
    snapshot_download(
        repository, revision=revision, local_dir=path, allow_patterns=patterns, **options
    )


def download_onnx(model, compute_type="default", progress=None, *, revision="main"):
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


def onnx_quantization(compute_type):
    if compute_type not in ("default", "auto", "float32", "int8"):
        raise ValueError("ONNX compute type must be default, auto, float32, or int8")
    return "int8" if compute_type == "int8" else None


def load_onnx(config):
    module = onnx_module()
    path = Path(config.model).expanduser()
    local = path.is_dir()
    if not local:
        path = onnx_path(config.model, getattr(config, "model_revision", "main"))
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


def model_catalog(engine="whisper", *, revision="main"):
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
                "memory_note": (
                    "Weights plus inference working memory; CPU default uses full precision"
                ),
                "native_streaming": engine == "nemo",
                "vocabulary": engine == "nemo" or entry.get("family") == "qwen3_asr",
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
    from faster_whisper.utils import _MODELS
    from huggingface_hub import scan_cache_dir

    root = cache_dir() / "models"
    installed = {}
    if root.is_dir():
        installed = {repo.repo_id: repo for repo in scan_cache_dir(root).repos}

    def has_revision(repository):
        cached = installed.get(repository)
        return bool(
            cached
            and any(
                (revision == item.commit_hash or revision in item.refs)
                and usable_whisper_snapshot(item.snapshot_path)
                for item in cached.revisions
            )
        )

    return [
        {
            "model": name,
            "name": name,
            "languages": "English" if name.endswith(".en") or "distil" in name else "Multilingual",
            "engine": engine,
            "revision": revision,
            "installed": has_revision(repo),
            "path": str(installed[repo].repo_path) if has_revision(repo) else "",
        }
        for name, repo in _MODELS.items()
    ]


def delete_model(model, engine="whisper", *, revision="main"):
    """Delete only app-owned cache entries, never an entered local model directory."""
    if Path(model).expanduser().exists():
        raise ValueError("Local model paths must be managed outside MouthKeys")
    if engine in ("onnx", "transformers", "nemo"):
        if engine == "onnx":
            path = onnx_path(model, revision)
        else:
            from .native_asr import artifact_path

            path = artifact_path(model, engine, revision)
        if path.is_symlink():
            raise ValueError("Refusing linked model cache")
        if path.is_dir():
            shutil.rmtree(path)
        return
    if engine != "whisper":
        raise ValueError("Unknown engine")
    from faster_whisper.utils import _MODELS

    repo = _MODELS.get(model, model)
    if repo.count("/") != 1 or any(part in ("", ".", "..") for part in repo.split("/")):
        raise ValueError("Expected model alias or Hugging Face owner/model ID")
    path = cache_dir() / "models" / ("models--" + repo.replace("/", "--"))
    if path.is_symlink():
        raise ValueError("Refusing linked model cache")
    if path.is_dir():
        shutil.rmtree(path)
