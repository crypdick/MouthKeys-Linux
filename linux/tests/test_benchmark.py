import hashlib
import json
import subprocess
import sys
import threading
import wave
from dataclasses import replace
from pathlib import Path
from subprocess import Popen
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QComboBox, QLineEdit, QMessageBox, QWidget

from mouthkeys import benchmark, benchmark_reports, engines
from mouthkeys.config import Config, state_dir
from mouthkeys.models_ui import ModelBrowser


def wav(path, samples=16000):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(b"\x01\x02" * samples)
    return path


def report(config=None):
    config = config or Config()
    return {
        "version": 1,
        "timestamp_utc": "2026-11-12T00:00:00+00:00",
        "engine": config.engine,
        "model": benchmark.model_identity(config.model),
        "revision": config.model_revision,
        "device": config.device,
        "compute_type": config.compute_type,
        "decode_settings": benchmark.decode_settings(config),
        "hardware": benchmark.hardware_context(),
        "software": benchmark.software_context(),
        "artifact": {"metadata_sha256": "artifact"},
        "fixture_sha256": "fixture",
        "load_seconds": 2.0,
        "decode_seconds": 0.5,
        "audio_seconds": 1.0,
        "real_time_factor": 0.5,
        "accuracy": None,
        "scope": benchmark.SCOPE,
    }


def test_exact_metrics_normalization_and_bounded_work(monkeypatch):
    result = benchmark.accuracy("Hello, WORLD!", "hello word")
    assert result["wer"] == 0.5
    assert result["cer"] == 0.1
    assert result["wer_errors"] == result["cer_errors"] == 1
    assert result["reference_sha256"] == hashlib.sha256(b"Hello, WORLD!").hexdigest()
    assert benchmark.accuracy("\N{FULLWIDTH LATIN CAPITAL LETTER A} B", "a b")["wer"] == 0
    assert benchmark.accuracy("a", "a b")["wer"] == 1
    with pytest.raises(ValueError, match="letter or digit"):
        benchmark.accuracy("!!!", "hello")
    monkeypatch.setattr(benchmark, "MAX_METRIC_CELLS", 2)
    bounded = benchmark.accuracy("a b", "a b")
    assert bounded["wer"] is None
    assert "bounded" in bounded["wer_unavailable"]


def test_rss_counts_worker_and_children_from_every_thread(tmp_path):
    for pid, rss, children in ((1, 10, "2 3"), (2, 20, "3"), (3, 30, "")):
        directory = tmp_path / str(pid)
        (directory / "task" / "thread").mkdir(parents=True)
        (directory / "status").write_text(f"VmRSS:\t{rss} kB\n")
        (directory / "task" / "thread" / "children").write_text(children)
    assert benchmark._tree_rss(1, tmp_path) == 60 * 1024
    (tmp_path / "1" / "status").write_text("State: Z")
    assert benchmark._tree_rss(1, tmp_path) == 50 * 1024
    assert benchmark._tree_rss(999, tmp_path) is None


def fake_process(monkeypatch, response, *, active=False, exit_code=0):
    observed = {}

    class Process(Popen):
        pid = 12345
        returncode = exit_code

        def __init__(self, args, **options):
            observed.update(args=args, **options)
            observed["request"] = json.load(options["stdin"])
            options["stdout"].write(json.dumps(response).encode())
            self.active = active
            self.waited = False
            observed["process"] = self

        def poll(self):
            return None if self.active else self.returncode

        def wait(self, timeout):
            self.active = False
            self.waited = True
            return 0

    monkeypatch.setattr(benchmark.subprocess, "Popen", Process)
    monkeypatch.setattr(benchmark.os, "killpg", lambda *a: observed.update(killed=True))
    monkeypatch.setattr(benchmark, "_tree_rss", lambda pid: 1024)
    return observed


def test_runner_isolated_launch_private_bounded_store_and_no_raw_text(monkeypatch, tmp_path):
    fixture = wav(tmp_path / "private-source.wav")
    value = report()
    observed = fake_process(monkeypatch, {"result": value})
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    monkeypatch.setattr(benchmark_reports, "MAX_REPORTS", 2)
    for _ in range(3):
        actual = benchmark.run_benchmark(Config(), fixture, reference="private reference")
    assert observed["args"][1:] == ["-I", "-m", "mouthkeys.benchmark", "--worker"]
    assert observed["env"]["HF_HUB_OFFLINE"] == "1"
    assert "PYTHONPATH" not in observed["env"]
    assert observed["start_new_session"]
    assert observed["process"].waited
    saved = list((state_dir() / "benchmarks").glob("*.json"))
    assert len(saved) == 2
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in saved)
    assert "private-source" not in json.dumps(actual)
    assert "private reference" not in json.dumps(actual)
    assert len(benchmark_reports.measurements()) == 2


def test_runner_counts_only_available_memory_samples(monkeypatch, tmp_path):
    observed = fake_process(monkeypatch, {"result": report()}, active=True)
    readings = [None, 100, 500, 300]
    pending = iter(readings)

    def sample(_pid):
        return next(pending)

    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == len(readings):
            observed["process"].active = False

    monkeypatch.setattr(benchmark, "_tree_rss", sample)
    monkeypatch.setattr(benchmark, "time", SimpleNamespace(monotonic=lambda: 0, sleep=sleep))
    result = benchmark.run_benchmark(Config(), wav(tmp_path / "fixture.wav"), save=False)
    assert result["memory_samples"] == 3
    assert result["sampled_peak_tree_rss_bytes"] == 500
    assert not (state_dir() / "benchmarks").exists()


def test_runner_honors_cancellation_after_worker_finishes(monkeypatch, tmp_path):
    observed = fake_process(monkeypatch, {"result": report()})
    event = threading.Event()
    monkeypatch.setattr(event, "is_set", iter([False, True]).__next__)
    with pytest.raises(RuntimeError, match="Benchmark cancelled"):
        benchmark.run_benchmark(Config(), wav(tmp_path / "fixture.wav"), cancel_event=event)
    assert observed["process"].waited
    assert not (state_dir() / "benchmarks").exists()


@pytest.mark.parametrize("cancelled", [True, False])
def test_cancel_and_timeout_reap_worker_without_report(monkeypatch, tmp_path, cancelled):
    observed = fake_process(monkeypatch, {}, active=True)
    event = threading.Event()
    if cancelled:
        monkeypatch.setattr(event, "is_set", iter([False, True]).__next__)
    clock = iter([0, 10])
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: next(clock, 10))
    with pytest.raises(RuntimeError, match="cancelled" if cancelled else "timed out"):
        benchmark.run_benchmark(
            Config(), wav(tmp_path / "fixture.wav"), timeout=1, cancel_event=event
        )
    assert observed["killed"]
    assert observed["process"].waited
    assert not list((state_dir() / "benchmarks").glob("*.json"))


@pytest.mark.parametrize(
    ("options", "error"),
    [({"timeout": 0}, "Benchmark timeout"), ({"reference": ""}, "Reference must contain")],
)
def test_invalid_options_are_rejected_before_start(monkeypatch, tmp_path, options, error):
    monkeypatch.setattr(benchmark.subprocess, "Popen", lambda *a, **kw: pytest.fail("Worker"))
    with pytest.raises(ValueError, match=error):
        benchmark.run_benchmark(Config(), wav(tmp_path / "fixture.wav"), **options)


@pytest.mark.parametrize("invalid", ["directory", "empty", "oversized"])
def test_invalid_fixtures_are_rejected_before_start(monkeypatch, tmp_path, invalid):
    monkeypatch.setattr(benchmark.subprocess, "Popen", lambda *a, **kw: pytest.fail("Worker"))
    fixture = tmp_path if invalid == "directory" else wav(tmp_path / "fixture.wav", samples=0)
    if invalid == "oversized":
        monkeypatch.setattr(benchmark, "MAX_FIXTURE_BYTES", 1)
    with pytest.raises(ValueError, match="ten minutes" if invalid == "empty" else "128 MiB"):
        benchmark.run_benchmark(Config(), fixture)


def test_worker_failure_is_reaped_without_publishing(monkeypatch, tmp_path):
    observed = fake_process(monkeypatch, {}, exit_code=1)
    with pytest.raises(RuntimeError, match="Benchmark worker failed"):
        benchmark.run_benchmark(Config(), wav(tmp_path / "fixture.wav"))
    assert observed["process"].waited
    assert not (state_dir() / "benchmarks").exists()


def test_worker_rejects_oversized_request_before_loading():
    response = subprocess.run(
        [sys.executable, "-I", "-m", "mouthkeys.benchmark", "--worker"],
        input=b" " * (2 * 1024 * 1024 + 1),
        capture_output=True,
        check=True,
        timeout=10,
    )
    assert json.loads(response.stdout) == {"error": "Benchmark request too large"}


def test_missing_model_worker_fails_closed_without_installing_or_saving(tmp_path):
    # Real dedicated worker and seccomp, no model/library inference or network fetch.
    with pytest.raises(RuntimeError, match="Whisper model missing locally"):
        benchmark.run_benchmark(
            Config(model="missing-fixture-model"), wav(tmp_path / "fixture.wav"), timeout=20
        )
    assert not list((state_dir() / "benchmarks").glob("*.json"))


def test_catalog_ignores_report_deleted_before_sort(monkeypatch):
    directory = state_dir() / "benchmarks"
    directory.mkdir()
    saved = report()
    (directory / "valid.json").write_text(json.dumps(saved))
    vanished = directory / "vanished.json"
    vanished.write_text(json.dumps(saved))
    stat = Path.stat

    def stat_after_removal(path, **options):
        if path == vanished:
            raise FileNotFoundError(path)
        return stat(path, **options)

    monkeypatch.setattr(Path, "stat", stat_after_removal)
    assert benchmark_reports.measurements() == [saved]


def test_catalog_ignores_report_with_unrepresentable_duration():
    directory = state_dir() / "benchmarks"
    directory.mkdir()
    invalid = report()
    invalid["decode_seconds"] = 10**400
    (directory / "invalid.json").write_text(json.dumps(invalid))
    assert benchmark_reports.measurements() == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("version", 2),
        ("hardware", []),
        ("accuracy", True),
        ("engine", 1),
        ("real_time_factor", True),
        ("audio_seconds", -1),
        ("decode_seconds", "fast"),
    ],
)
def test_catalog_ignores_invalid_report_fields(key, value):
    directory = state_dir() / "benchmarks"
    directory.mkdir()
    invalid = report()
    invalid[key] = value
    (directory / "invalid.json").write_text(json.dumps(invalid))
    assert benchmark_reports.measurements() == []


def test_catalog_ignores_bad_json_missing_and_oversized_records(monkeypatch):
    directory = state_dir() / "benchmarks"
    directory.mkdir()
    saved = report()
    (directory / "valid.json").write_text(json.dumps(saved))
    (directory / "array.json").write_text("[]")
    (directory / "malformed.json").write_text("{")
    (directory / "nested.json").write_text("[" * 2000 + "]" * 2000)
    (directory / "oversized.json").write_bytes(b" " * (benchmark_reports.MAX_REPORT_BYTES + 1))
    missing = directory / "missing.json"
    missing.write_text(json.dumps(saved))
    open_file = Path.open

    def open_if_present(path, *args, **options):
        if path == missing:
            raise FileNotFoundError(path)
        return open_file(path, *args, **options)

    monkeypatch.setattr(Path, "open", open_if_present)
    assert benchmark_reports.measurements() == [saved]


def test_onnx_worker_requires_local_model_and_vad(tmp_path):
    fixture = wav(tmp_path / "fixture.wav")
    config = Config(engine="onnx", model="fixture-only")
    with pytest.raises(RuntimeError, match="already installed ONNX model"):
        benchmark.run_benchmark(config, fixture, timeout=10)
    artifact = engines.onnx_path(config.model)
    artifact.mkdir(parents=True)
    (artifact / engines.READY).touch()
    with pytest.raises(RuntimeError, match="already installed Silero VAD"):
        benchmark.run_benchmark(config, fixture, timeout=10)
    assert not (state_dir() / "benchmarks").exists()


@pytest.mark.parametrize("engine", ["whisper", "onnx-managed", "onnx-custom", "nemo"])
def test_measurement_provenance_uses_private_snapshot_and_disables_optional_data(
    monkeypatch, tmp_path, engine
):
    config = Config(vocabulary_boosting="on", pronunciation_matching="on", ai_enabled="on")
    fixture = wav(tmp_path / "chosen-private.wav")
    artifact = tmp_path / "snapshots" / "weights"
    if engine.startswith("onnx"):
        vad = engines.managed_path("silero")
        vad.mkdir(parents=True)
        (vad / engines.READY).touch()
        if engine == "onnx-managed":
            artifact = engines.onnx_path("fixture-only")
    if engine == "nemo":
        artifact = artifact.with_suffix(".nemo")
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"fake")
    else:
        (artifact / "empty-directory").mkdir(parents=True)
        if engine == "whisper":
            for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt"):
                (artifact / name).write_bytes(b"fake")
        elif engine == "onnx-managed":
            (artifact / engines.READY).touch()
    config = replace(
        config,
        engine=engine.split("-")[0],
        model="fixture-only" if engine == "onnx-managed" else str(artifact),
    )
    observed = {}

    class Transcriber:
        model = (
            SimpleNamespace(close=lambda: observed.update(closed=True))
            if engine == "nemo"
            else None
        )

        def load(self, cfg):
            observed["config"] = cfg

    def transcribe(path, cfg, **options):
        observed["fixture"] = path
        assert path != fixture
        assert path.read_bytes() == fixture.read_bytes()
        assert path.stat().st_mode & 0o777 == 0o600
        assert options["max_seconds"] == 600
        return {"text": "Hello world", "duration": 1}

    monkeypatch.setattr("mouthkeys.transcribe.Transcriber", Transcriber)
    monkeypatch.setattr("mouthkeys.file_transcription.transcribe_file", transcribe)
    reference = None if engine == "nemo" else "Hello world"
    result = benchmark._measure(
        {"config": config.__dict__, "fixture": str(fixture), "reference": reference}
    )
    if reference is None:
        assert result["accuracy"] is None
    else:
        assert result["accuracy"]["wer"] == result["accuracy"]["cer"] == 0
    assert result["fixture_sha256"] == hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert observed["config"].model == str(artifact)
    assert bool(observed.get("closed")) == (engine == "nemo")
    assert observed["config"].vocabulary_boosting == "off"
    assert observed["config"].pronunciation_matching == "off"
    assert observed["config"].ai_enabled == "off"
    assert not observed["fixture"].exists()
    assert "Hello world" not in json.dumps(result)
    assert str(tmp_path) not in json.dumps(result)


def test_fingerprinting_rejects_more_files_than_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "MAX_ARTIFACT_FILES", 1)
    (tmp_path / "a").touch()
    (tmp_path / "b").touch()
    with pytest.raises(ValueError, match="too many files"):
        benchmark._artifact_identity(tmp_path)


def test_snapshot_rechecks_fixture_size_during_copy(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "MAX_FIXTURE_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds 128 MiB"):
        benchmark._copy_fixture(wav(tmp_path / "fixture.wav"), tmp_path / "snapshot.wav")


@pytest.mark.parametrize("changed", ["language", "compute_type", "device", "asr_max_tokens"])
def test_catalog_measurements_reject_other_settings_and_artifacts(monkeypatch, tmp_path, changed):
    config = Config()
    item = report(config)
    monkeypatch.setattr(benchmark, "_local_config", lambda cfg: (cfg, tmp_path))
    monkeypatch.setattr(benchmark, "_artifact_identity", lambda path: item["artifact"])
    assert benchmark.matching_measurement(config, [item]) == item
    value = 100 if changed == "asr_max_tokens" else "different"
    assert benchmark.matching_measurement(replace(config, **{changed: value}), [item]) is None
    assert benchmark.matching_measurement(config, [item], ("other-fixture", None)) is None
    monkeypatch.setattr(benchmark, "_artifact_identity", lambda path: {"different": True})
    assert benchmark.matching_measurement(config, [item]) is None


def test_catalog_shows_scoped_measurements_and_active_custom_model(monkeypatch, tmp_path):
    artifact = tmp_path / "custom"
    artifact.mkdir()
    config = Config(model=str(artifact))
    item = report(config)
    item["sampled_peak_tree_rss_bytes"] = 20 * 2**20
    item["accuracy"] = {"wer": 0.25, "reference_sha256": "reference"}
    monkeypatch.setattr(engines, "measurements", lambda: [item])
    monkeypatch.setattr(benchmark, "_local_config", lambda cfg: (cfg, artifact))
    monkeypatch.setattr(benchmark, "_artifact_identity", lambda path: item["artifact"])
    custom = next(
        row
        for row in engines.model_catalog(measurement_config=config)
        if row["model"] == str(artifact)
    )
    assert custom["installed"]
    assert "0.500× real time" in custom["speed_note"]
    assert "25.0% WER on fixture" in custom["accuracy_note"]
    assert "sampled process-tree RSS 20 MiB" in custom["memory_note"]
    assert engines.model_catalog()[0]["speed_note"] == "Not benchmarked on this hardware"


def test_cpu_measurement_ignores_unrelated_gpu_context(monkeypatch, tmp_path):
    config = Config(engine="nemo", device="cpu")
    item = report(config)
    item["hardware"]["display_adapter_pci_ids"] = ["different-gpu"]
    item["software"].pop("nvidia_driver")
    monkeypatch.setattr(benchmark, "_local_config", lambda cfg: (cfg, tmp_path))
    monkeypatch.setattr(benchmark, "_artifact_identity", lambda path: item["artifact"])
    assert benchmark.matching_measurement(config, [item]) == item

    def missing(cfg):
        raise FileNotFoundError(cfg.model)

    monkeypatch.setattr(benchmark, "_local_config", missing)
    assert benchmark.matching_measurement(config, [item]) is None


@pytest.mark.parametrize("driver", ["updated driver", None])
def test_gpu_measurement_does_not_match_a_different_driver(monkeypatch, driver):
    config = Config(device="cuda")
    saved = report(config)
    saved["software"]["nvidia_driver"] = "original driver"
    read = Path.read_text

    def read_driver(path, *args, **options):
        if path == Path("/proc/driver/nvidia/version"):
            if driver is None:
                raise FileNotFoundError(path)
            return f"  {driver}\n"
        return read(path, *args, **options)

    monkeypatch.setattr(Path, "read_text", read_driver)
    assert benchmark.software_context()["nvidia_driver"] == driver
    assert benchmark.matching_measurement(config, [saved]) is None


def test_model_browser_missing_weights_does_not_launch_benchmark(monkeypatch):

    window = QWidget()
    engine = QComboBox()
    engine.addItem("NeMo", "nemo")
    window.fields = {"engine": engine, "model_revision": QLineEdit("main")}
    window.loaded_model = None
    monkeypatch.setattr(QMessageBox, "information", lambda *a: None)
    monkeypatch.setattr(benchmark, "run_benchmark", lambda *a, **kw: pytest.fail("Model load"))
    browser = ModelBrowser(window)
    browser.table.selectRow(0)
    browser.benchmark()
    assert browser.benchmark_task is None
    browser.close()
    window.close()
