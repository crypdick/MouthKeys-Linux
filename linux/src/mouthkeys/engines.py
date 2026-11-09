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


def download_onnx(model, compute_type="default"):
    """Only this explicit action is allowed to fetch ONNX artifacts."""
    module = onnx_module()
    path = managed_path(model)
    quantization = onnx_quantization(compute_type)
    module.load_model(model, path, quantization=quantization, offline=False)
    vad_path = managed_path("silero")
    module.load_vad("silero", vad_path, offline=False)
    # A cancelled Hub transfer can leave a directory and some model files behind.
    # Only advertise a managed model after both engine and VAD finished loading.
    (vad_path / READY).touch(mode=0o600)
    (path / READY).touch(mode=0o600)
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
        path = managed_path(config.model)
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


def model_catalog(engine="whisper"):
    """Local-only catalog; no Hub queries or downloads."""
    if engine == "onnx":
        return [
            {
                "model": name,
                "name": label,
                "languages": languages,
                "engine": engine,
                "installed": (managed_path(name) / READY).is_file(),
                "path": str(managed_path(name)),
            }
            for name, (label, languages) in ONNX_MODELS.items()
        ]
    if engine != "whisper":
        raise ValueError("Unknown engine")
    from faster_whisper.utils import _MODELS
    from huggingface_hub import scan_cache_dir

    root = cache_dir() / "models"
    installed = {}
    if root.is_dir():
        installed = {repo.repo_id: repo for repo in scan_cache_dir(root).repos}
    return [
        {
            "model": name,
            "name": name,
            "languages": "English" if name.endswith(".en") or "distil" in name else "Multilingual",
            "engine": engine,
            "installed": repo in installed,
            "path": str(installed[repo].repo_path) if repo in installed else "",
        }
        for name, repo in _MODELS.items()
    ]


def delete_model(model, engine="whisper"):
    """Delete only app-owned cache entries, never an entered local model directory."""
    if Path(model).expanduser().exists():
        raise ValueError("Local model paths must be managed outside MouthKeys")
    if engine == "onnx":
        path = managed_path(model)
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
