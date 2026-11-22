"""Exercise model cache management with local synthetic Hub snapshots."""

import pytest

from mouthkeys import engines, model_cache, native_asr
from mouthkeys.cli import main
from mouthkeys.config import Config, cache_dir

FIRST = "1" * 40
SECOND = "2" * 40


def hub_repo(repository):
    return cache_dir() / "models" / ("models--" + repository.replace("/", "--"))


def add_revision(repo, commit, *, ref="", blob=""):
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents=True)
    if blob:
        artifact = repo / "blobs" / blob
        artifact.parent.mkdir(exist_ok=True)
        artifact.write_text("synthetic weights")
        (snapshot / "model.bin").symlink_to(artifact)
    else:
        (snapshot / "model.bin").write_text("synthetic weights")
    for name in ("config.json", "tokenizer.json", "vocabulary.txt"):
        (snapshot / name).write_text("fixture")
    if ref:
        pointer = repo / "refs" / ref
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(commit)
    return snapshot


@pytest.mark.parametrize("revision", ["main", FIRST])
def test_delete_model_cli_preserves_other_whisper_revisions_and_shared_blobs(revision):
    repo = hub_repo("example/native-fixture")
    selected = add_revision(repo, FIRST, ref="main", blob="shared")
    kept = add_revision(repo, SECOND, ref="refs/pr/6", blob="shared")
    identical_hash_in_other_repo = add_revision(hub_repo("example/another"), FIRST)
    Config(engine="whisper", model_revision=revision).save()
    assert main(["delete-model", "example/native-fixture"]) == 0
    assert not selected.exists()
    assert not (repo / "refs" / "main").exists()
    assert kept.is_dir()
    assert (kept / "model.bin").read_text() == "synthetic weights"
    assert (repo / "refs" / "refs" / "pr" / "6").read_text() == SECOND
    assert identical_hash_in_other_repo.is_dir()


def test_delete_uncached_whisper_revision_does_not_remove_cached_revision():
    repo = hub_repo("example/native-fixture")
    kept = add_revision(repo, FIRST, ref="main")
    engines.delete_model("example/native-fixture", revision="uncached")
    assert kept.is_dir()


def test_delete_model_rejects_linked_parent_without_touching_target(tmp_path):
    outside = tmp_path / "other-cache"
    outside.mkdir()
    selected = engines.onnx_path("fixture")
    selected.parent.parent.mkdir(parents=True)
    selected.parent.symlink_to(outside, target_is_directory=True)
    uncopied = outside / selected.name
    uncopied.mkdir()
    (uncopied / "weights").write_text("must remain")
    with pytest.raises(ValueError, match="Refusing linked model cache"):
        engines.delete_model("fixture", "onnx")
    assert (uncopied / "weights").read_text() == "must remain"


def test_whisper_catalog_checks_selected_revision_and_complete_artifacts():
    name, repository = next(iter(model_cache.whisper_repositories().items()))
    repo = hub_repo(repository)
    main_revision = add_revision(repo, FIRST, ref="main")
    pinned_revision = add_revision(repo, SECOND, ref="refs/pr/6")
    (pinned_revision / "tokenizer.json").write_text("")
    # Another repository type with the same ID must not replace the speech model.
    dataset = repo.with_name(repo.name.replace("models--", "datasets--", 1))
    add_revision(dataset, SECOND, ref="main")

    def row(revision):
        return next(
            item
            for item in engines.model_catalog("whisper", revision=revision)
            if item["model"] == name
        )

    assert row("main")["installed"]
    assert row(FIRST)["path"] == str(repo)
    assert not row("refs/pr/6")["installed"]
    assert not row("missing")["installed"]
    (pinned_revision / "tokenizer.json").write_text("{}")
    assert row("refs/pr/6")["installed"]
    assert main_revision.is_dir()


def test_delete_last_whisper_revision_stays_within_that_repo():
    repo = hub_repo("example/model")
    selected = add_revision(repo, FIRST, ref="main")
    kept = add_revision(hub_repo("example/another"), FIRST, ref="main")
    engines.delete_model("example/model")
    assert not selected.exists()
    assert kept.exists()


def test_delete_skips_missing_or_unscannable_repo():
    engines.delete_model("example/missing")
    repo = hub_repo("example/incomplete")
    repo.mkdir(parents=True)
    engines.delete_model("example/incomplete")
    assert repo.is_dir()


def test_delete_refuses_whisper_blob_outside_selected_cache(tmp_path):
    repo = hub_repo("example/model")
    selected = add_revision(repo, FIRST, ref="main")
    kept = add_revision(repo, SECOND)
    outside = tmp_path / "outside.bin"
    outside.write_text("unrelated bytes")
    selected.joinpath("model.bin").unlink()
    selected.joinpath("model.bin").symlink_to(outside)
    with pytest.raises(ValueError, match="outside the selected cache"):
        engines.delete_model("example/model")
    assert outside.read_text() == "unrelated bytes"
    assert selected.is_dir()
    assert kept.is_dir()


@pytest.mark.parametrize(
    ("model", "engine", "error"),
    [
        ("../bad", "whisper", "Expected model alias"),
        ("not-a-repo", "whisper", "Expected model alias"),
        ("name", "unknown", "Unknown engine"),
    ],
)
def test_invalid_model_cache_identifiers_are_rejected(model, engine, error):
    with pytest.raises(ValueError, match=error):
        engines.delete_model(model, engine)


def test_delete_native_revision_keeps_other_native_artifacts():
    model = "native-fixture"
    selected = native_asr.artifact_path(model, "nemo", "selected")
    other = native_asr.artifact_path(model, "nemo", "other")
    selected.mkdir(parents=True)
    other.mkdir()
    engines.delete_model(model, "nemo", revision="selected")
    engines.delete_model("missing", "onnx")
    assert not selected.exists()
    assert other.is_dir()


@pytest.mark.parametrize(
    "optional",
    [
        {"accuracy": None, "sampled_peak_tree_rss_bytes": 0},
        {"accuracy": {"wer": True}, "sampled_peak_tree_rss_bytes": False},
        {"accuracy": {"wer": 10**400}, "sampled_peak_tree_rss_bytes": 10**400},
        {"accuracy": {"wer": float("nan")}, "sampled_peak_tree_rss_bytes": "not measured"},
    ],
)
def test_catalog_keeps_defaults_when_optional_measurements_are_invalid(monkeypatch, optional):
    report = {
        "device": "cpu",
        "compute_type": "default",
        "real_time_factor": 0.5,
        "fixture_sha256": "fixture",
        "timestamp_utc": "2026-11-12",
        **optional,
    }
    monkeypatch.setattr("mouthkeys.benchmark.matching_measurement", lambda *_: report)
    first = engines.model_catalog("onnx", measurement_config=Config(engine="onnx"))[0]
    assert "0.500× real time" in first["speed_note"]
    assert first["accuracy_note"].startswith("Not evaluated")
    assert "RSS" not in first["memory_note"]


def test_deleting_last_complete_revision_keeps_partial_download():
    repo = hub_repo("example/model")
    selected = add_revision(repo, FIRST, ref="main", blob="completed")
    partial = repo / "blobs" / "other-revision.incomplete"
    partial.write_text("download remains")
    engines.delete_model("example/model")
    assert partial.read_text() == "download remains"
    assert not selected.exists()
    assert not (repo / "refs" / "main").exists()
    assert not (repo / "blobs" / "completed").exists()
