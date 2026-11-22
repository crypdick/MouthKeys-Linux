"""Verify explicit Whisper download dispatch without contacting the model host."""

import json

import pytest
from faster_whisper import utils

from mouthkeys.cli import main
from mouthkeys.config import Config, cache_dir
from mouthkeys.transcribe import Transcriber, download


@pytest.mark.parametrize("model", ["base.en", "fixture/whisper-model"])
def test_download_cli_reports_progress_and_fetches_whisper_artifacts_only(
    monkeypatch, capsys, model
):
    Config(model=model, model_revision="fixed-revision").save()
    calls = []

    def snapshot(repository, **options):
        calls.append((repository, options))
        return "/local/snapshot"

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    assert main(["download"]) == 0
    repository, options = calls[0]
    assert repository == (utils._MODELS["base.en"] if model == "base.en" else model)
    assert options["revision"] == "fixed-revision"
    assert options["cache_dir"] == str(cache_dir() / "models")
    assert options["allow_patterns"] == [
        "config.json",
        "preprocessor_config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.*",
    ]
    streams = capsys.readouterr()
    assert streams.out.strip() == "/local/snapshot"
    assert json.loads(streams.err)["download_progress"]["phase"] == "preparing"


def test_explicit_download_without_progress_uses_engine_helper(monkeypatch):
    calls = []
    monkeypatch.setattr(
        utils,
        "download_model",
        lambda model, **options: calls.append((model, options)) or "/local/weights",
    )
    assert download("base.en", revision="fixed-revision") == "/local/weights"
    assert calls == [
        ("base.en", {"cache_dir": str(cache_dir() / "models"), "revision": "fixed-revision"})
    ]


def test_invalid_whisper_name_and_unknown_engine_never_reach_download(monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *_a, **_kw: pytest.fail("Hub"))
    monkeypatch.setattr(utils, "download_model", lambda *_a, **_kw: pytest.fail("Whisper fetch"))
    with pytest.raises(ValueError, match="Unknown Whisper model"):
        download("unrecognized-fixture-model", progress=lambda _: None)
    with pytest.raises(ValueError, match="Unknown engine"):
        download("base.en", engine="unrecognized")


def test_invalid_runtime_engine_leaves_model_unloaded():
    transcriber = Transcriber()
    with pytest.raises(ValueError, match="Unknown speech engine"):
        transcriber.load(Config(engine="unrecognized"))
    assert transcriber.loaded_model is None
