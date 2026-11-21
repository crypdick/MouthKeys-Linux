import io
import json
import sys
import tarfile
from dataclasses import replace
from types import SimpleNamespace

import huggingface_hub
import pytest

from mouthkeys import native_asr
from mouthkeys.config import Config
from mouthkeys.engines import READY

from .native_asr_fakes import torch_fake

NEMO_ID = "nvidia/parakeet_realtime_eou_120m-v1"
GOOD_CONFIG = b"""
model:
  target: nemo.collections.asr.models.ASRModel
  auxiliary: nemo:tokenizer.model
  shape: [1, 2, 3]
  recursive: &alias [*alias, torch.nn.Identity]
"""


def archive_model(path, config=GOOD_CONFIG, *, weights=True, tokenizer=True):
    with tarfile.open(path, "w") as archive:
        for name, data in (
            ("./model_config.yaml", config),
            ("model_weights.ckpt", b"placeholder" if weights else None),
            ("tokenizer.model", b"placeholder" if tokenizer else None),
        ):
            if data is not None:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))


def test_nemo_archive_validation_checks_real_members_and_shared_configuration(tmp_path):
    archive = tmp_path / "model.nemo"
    archive_model(archive)
    native_asr.validate_artifacts(archive, "nemo")
    native_asr.validate_artifacts(tmp_path, "nemo")
    (tmp_path / "other.nemo").touch()
    with pytest.raises(ValueError, match="exactly one"):
        native_asr.validate_artifacts(tmp_path, "nemo")


@pytest.mark.parametrize(
    ("config", "weights", "tokenizer", "message"),
    [
        (None, True, True, "configuration missing"),
        (b"x" * (1024 * 1024 + 1), True, True, "configuration missing"),
        (GOOD_CONFIG, False, True, "weights missing"),
        (GOOD_CONFIG, True, False, "auxiliary artifact"),
        (b"model: {_target_: unregistered.Decoder}", True, True, "non-framework"),
        (b"postprocess: {target: 7}", True, True, "non-framework"),
    ],
)
def test_invalid_nemo_artifact_never_passes_local_validation(
    tmp_path, config, weights, tokenizer, message
):
    archive = tmp_path / "model.nemo"
    archive_model(archive, config, weights=weights, tokenizer=tokenizer)
    with pytest.raises(ValueError, match=message):
        native_asr.validate_artifacts(archive, "nemo")


def test_non_file_nemo_configuration_and_tokenizer_are_rejected(tmp_path):
    archive = tmp_path / "model.nemo"
    with tarfile.open(archive, "w") as output:
        directory = tarfile.TarInfo("model_config.yaml")
        directory.type = tarfile.DIRTYPE
        output.addfile(directory)
    with pytest.raises(ValueError, match="configuration missing"):
        native_asr.validate_artifacts(archive, "nemo")

    archive_model(archive, tokenizer=False)
    with tarfile.open(archive, "a") as output:
        directory = tarfile.TarInfo("tokenizer.model")
        directory.type = tarfile.DIRTYPE
        output.addfile(directory)
    with pytest.raises(ValueError, match="auxiliary artifact"):
        native_asr.validate_artifacts(archive, "nemo")


def test_transformers_checks_local_metadata_and_weight_shards(tmp_path):
    config = tmp_path / "config.json"
    with pytest.raises(ValueError, match=r"config.json missing"):
        native_asr.validate_artifacts(tmp_path, "transformers")
    for invalid in ([], {"model_type": "different-decoder"}):
        config.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="unsupported"):
            native_asr.validate_artifacts(tmp_path, "transformers")

    config.write_text('{"model_type":"qwen3_asr"}')
    (tmp_path / "preprocessor_config.json").touch()
    (tmp_path / "vocab.json").touch()
    (tmp_path / "weights-1.safetensors").touch()
    index = tmp_path / "model.safetensors.index.json"
    index.write_text('{"weight_map":{"layer":"weights-1.safetensors"}}')
    native_asr.validate_artifacts(tmp_path, "transformers")

    for invalid in (
        [],
        {"weight_map": []},
        {"weight_map": {}},
        {"weight_map": {"layer": None}},
        {"weight_map": {"layer": "other_folder/shard.safetensors"}},
        {"weight_map": {"layer": "no_file.safetensors"}},
    ):
        index.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="shard missing or invalid"):
            native_asr.validate_artifacts(tmp_path, "transformers")


def test_explicit_nemo_download_limits_files_and_sets_ready_only_after_validation(monkeypatch):
    path = native_asr.artifact_path(NEMO_ID, "nemo", "pinned")
    path.mkdir(parents=True)
    marker = path / READY
    marker.touch()
    calls = []
    progress = []

    def download(model, **options):
        assert model == NEMO_ID
        assert not marker.exists()
        calls.append(options)
        archive_model(path / native_asr.NEMO_MODELS[NEMO_ID]["filename"])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert native_asr.download_native(NEMO_ID, "nemo", "pinned", progress.append) == str(path)
    (options,) = calls
    assert options["revision"] == "pinned"
    assert options["allow_patterns"] == [native_asr.NEMO_MODELS[NEMO_ID]["filename"]]
    assert options["local_dir"] == path
    assert options["tqdm_class"] is not None
    assert [item["phase"] for item in progress] == ["verifying model artifacts", "ready"]
    assert marker.is_file()
    assert (
        native_asr.local_artifact(
            replace(Config(), model=NEMO_ID, engine="nemo", model_revision="pinned")
        )
        == path
    )


class GatedRepoError(Exception):
    pass


@pytest.mark.parametrize(
    ("failure", "expected", "message"),
    [
        (GatedRepoError("hidden"), RuntimeError, "Model access unavailable"),
        (OSError("transfer stopped"), OSError, "transfer stopped"),
        (None, ValueError, "no supported model weights"),
    ],
)
def test_failed_native_download_removes_outdated_readiness(monkeypatch, failure, expected, message):
    path = native_asr.artifact_path("publisher/model", "transformers")
    path.mkdir(parents=True)
    marker = path / READY
    marker.touch()

    def download(*_args, **_kwargs):
        if failure:
            raise failure
        assert not marker.exists()

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    with pytest.raises(expected, match=message):
        native_asr.download_native("publisher/model", "transformers")
    assert not marker.exists()


def test_unknown_engine_does_not_start_download(monkeypatch):
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **kw: pytest.fail("Unexpected download")
    )
    with pytest.raises(ValueError, match="Unknown native speech engine"):
        native_asr.download_native("publisher/model", "unknown")


@pytest.mark.parametrize(
    ("cuda", "device", "compute", "expected"),
    [
        (False, "auto", "auto", ("cpu", "float32")),
        (True, "auto", "auto", ("cuda", "float16")),
        (False, "cpu", "default", ("cpu", "auto")),
        (False, "cpu", "bfloat16", ("cpu", "bfloat16")),
    ],
)
def test_native_runtime_resolves_device_and_precision(monkeypatch, cuda, device, compute, expected):
    torch = torch_fake(cuda=cuda)
    monkeypatch.setitem(sys.modules, "torch", torch)
    config = replace(Config(), device=device, compute_type=compute)
    assert native_asr.torch_runtime(config) == (torch, *expected)


def test_native_runtime_reports_uninstalled_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="Install the nemo optional"):
        native_asr.torch_runtime(replace(Config(), engine="nemo"))


def test_shared_vad_supplies_original_sample_offsets(monkeypatch):
    audio = object()
    spans = [{"start": 123, "end": 456}]
    calls = []

    def vad(samples, options):
        calls.append((samples, options))
        return spans

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper.vad",
        SimpleNamespace(VadOptions=dict, get_speech_timestamps=vad),
    )
    assert native_asr.speech_spans(audio) == spans
    assert calls == [
        (
            audio,
            {
                "max_speech_duration_s": 25,
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
        )
    ]


def test_nemo_archive_requires_a_readable_configuration_stream(tmp_path, monkeypatch):
    archive = tmp_path / "model.nemo"
    archive_model(archive)
    monkeypatch.setattr(tarfile.TarFile, "extractfile", lambda *_: None)
    with pytest.raises(ValueError, match="could not be read"):
        native_asr.validate_artifacts(archive, "nemo")


@pytest.mark.parametrize(
    ("processor", "language", "expected"),
    [
        (SimpleNamespace(), "auto", "auto"),
        (SimpleNamespace(language_to_id={"en-GB": 4}), "en-GB", "en-GB"),
        (SimpleNamespace(lang_to_id={"es-MX": 3, "es-ES": 4}), "es", "es-MX"),
        (SimpleNamespace(language_to_id={"es-MX": 3}), "de", "de-DE"),
        (SimpleNamespace(language_to_id=[]), "xx", "xx"),
    ],
)
def test_nemotron_selects_explicit_locales_before_documented_defaults(
    processor, language, expected
):
    assert native_asr.nemotron_language(processor, language) == expected


def test_native_transcript_requires_text_before_normalizing():
    assert native_asr.transcript_text([" hello", ["world ", "again"]]) == "hello world again"
    with pytest.raises(ValueError, match="returned no text"):
        native_asr.transcript_text([None])
