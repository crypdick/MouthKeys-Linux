"""Host and configuration fingerprints used to compare benchmark reports."""

import hashlib
import importlib.metadata
import os
import platform
from contextlib import suppress
from pathlib import Path

from .config import Config


def hardware_context() -> dict[str, str | int | list[str] | None]:
    cpu = "unknown"
    with suppress(OSError, StopIteration):
        cpu = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
    adapters = []
    for device in Path("/sys/class/drm").glob("card[0-9]*/device"):
        with suppress(OSError):
            adapters.append(
                ":".join((device / key).read_text().strip() for key in ("vendor", "device"))
            )
    return {
        "architecture": platform.machine(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "display_adapter_pci_ids": sorted(adapters),
    }


def model_identity(model: str) -> str:
    path = Path(model).expanduser()
    return (
        "local:" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()
        if path.exists()
        else model
    )


def software_context() -> dict[str, str | dict[str, str] | None]:
    versions = {}
    for package in (
        "mouthkeys-linux",
        "faster-whisper",
        "ctranslate2",
        "onnx-asr",
        "onnxruntime",
        "torch",
        "transformers",
        "nemo_toolkit",
    ):
        with suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    try:
        driver = Path("/proc/driver/nvidia/version").read_text().strip()
    except OSError:
        driver = None
    return {
        "python": platform.python_version(),
        "kernel": platform.release(),
        "packages": versions,
        "nvidia_driver": driver,
    }


def decode_settings(config: Config) -> dict[str, str | int]:
    return {
        "language": config.language,
        "asr_max_tokens": config.asr_max_tokens,
        "native_streaming": config.native_streaming,
        "streaming_latency": config.streaming_latency,
    }
