"""Test explicit ONNX cache downloads without network or native dependencies."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from mouthkeys import engines


@pytest.mark.parametrize("progress_enabled", [False, True])
def test_explicit_pinned_onnx_download_fetches_only_manifest(monkeypatch, progress_enabled):
    model = "fixture-model"
    repository = "fixture/converted"
    calls, progress = [], []
    loader, resolver = ModuleType("onnx_asr.loader"), ModuleType("onnx_asr.resolver")
    resolver.model_repos = {model: repository}
    assets = {"encoder": "encoder.onnx", "tokens": "**/tokens.txt"}
    loader.create_asr_resolver = lambda *_a, **_opts: SimpleNamespace(
        model_type=SimpleNamespace(_get_model_files=lambda precision: assets)
    )

    def fetch_config(repo, filename, **options):
        calls.append(("config", repo, filename, options))
        options["local_dir"].mkdir(parents=True)

    def fetch_snapshot(repo, **options):
        calls.append(("fetch", repo, options))

    def load_model(name, path, **options):
        calls.append(("load", name, path, options))

    def load_vad(_name, path, **options):
        assert options == {"offline": False}
        path.mkdir(parents=True)

    package = ModuleType("onnx_asr")
    package.load_model, package.load_vad = load_model, load_vad
    monkeypatch.setitem(sys.modules, "onnx_asr", package)
    monkeypatch.setitem(sys.modules, "onnx_asr.loader", loader)
    monkeypatch.setitem(sys.modules, "onnx_asr.resolver", resolver)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fetch_config)
    monkeypatch.setattr("huggingface_hub.snapshot_download", fetch_snapshot)
    callback = progress.append if progress_enabled else None
    path = engines.onnx_path(model, "pinned")
    assert engines.download_onnx(model, "int8", callback, revision="pinned") == str(path)
    assert calls[0] == (
        "config",
        repository,
        "config.json",
        {"revision": "pinned", "local_dir": path},
    )
    assert calls[1][0:2] == ("fetch", repository)
    assert calls[1][2]["allow_patterns"] == [
        "config.json",
        "config.yaml",
        "encoder.onnx",
        "**/tokens.txt",
        "tokens.txt",
        "encoder.onnx?data",
    ]
    assert calls[1][2]["revision"] == "pinned"
    assert bool(calls[1][2]["tqdm_class"]) is progress_enabled
    assert calls[2] == ("load", model, path, {"quantization": "int8", "offline": True})
    assert [event["phase"] for event in progress] == (
        ["speech model", "voice activity model", "ready"] if progress_enabled else []
    )
    assert (path / engines.READY).exists()
