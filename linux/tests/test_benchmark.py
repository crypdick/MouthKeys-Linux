import hashlib
import json
import threading
import wave
from dataclasses import replace

import pytest

from mouthkeys import benchmark
from mouthkeys.config import Config, state_dir


def wav(path):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(b"\x01\x02" * 16000)
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
    assert benchmark.accuracy("Ａ B", "a b")["wer"] == 0
    with pytest.raises(ValueError, match="letter or digit"):
        benchmark.accuracy("!!!", "hello")
    monkeypatch.setattr(benchmark, "MAX_METRIC_CELLS", 2)
    bounded = benchmark.accuracy("a b", "a b")
    assert bounded["wer"] is None
    assert "bounded" in bounded["wer_unavailable"]


def test_rss_counts_worker_and_children_from_every_thread(tmp_path):
    for pid, rss, children in ((1, 10, "2 3"), (2, 20, ""), (3, 30, "")):
        directory = tmp_path / str(pid)
        (directory / "task" / "thread").mkdir(parents=True)
        (directory / "status").write_text(f"VmRSS:\t{rss} kB\n")
        (directory / "task" / "thread" / "children").write_text(children)
    assert benchmark._tree_rss(1, tmp_path) == 60 * 1024
    assert benchmark._tree_rss(999, tmp_path) is None


def fake_process(monkeypatch, response, *, active=False):
    observed = {}

    class Process:
        pid = 12345
        returncode = 0

        def __init__(self, args, **options):
            observed.update(args=args, **options)
            observed["request"] = json.load(options["stdin"])
            options["stdout"].write(json.dumps(response).encode())
            self.active = active
            self.waited = False
            observed["process"] = self

        def poll(self):
            return None if self.active else 0

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
    monkeypatch.setattr(benchmark, "MAX_REPORTS", 2)
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
    assert len(benchmark.measurements()) == 2


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
    assert observed["killed"] and observed["process"].waited
    assert not list((state_dir() / "benchmarks").glob("*.json"))


def test_missing_model_worker_fails_closed_without_installing_or_saving(tmp_path):
    # Real dedicated worker and seccomp, no model/library inference or network fetch.
    with pytest.raises(RuntimeError, match="Whisper model missing locally"):
        benchmark.run_benchmark(
            Config(model="missing-fixture-model"), wav(tmp_path / "fixture.wav"), timeout=20
        )
    assert not list((state_dir() / "benchmarks").glob("*.json"))


def test_measurement_provenance_uses_private_snapshot_and_disables_optional_data(
    monkeypatch, tmp_path
):
    config = Config(vocabulary_boosting="on", pronunciation_matching="on", ai_enabled="on")
    fixture = wav(tmp_path / "chosen-private.wav")
    artifact = tmp_path / "weights"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"fake")
    monkeypatch.setattr(benchmark, "_local_config", lambda cfg: (cfg, artifact))
    observed = {}

    class Transcriber:
        model = None

        def load(self, cfg):
            observed["config"] = cfg

    def transcribe(path, cfg, **options):
        observed["fixture"] = path
        assert path != fixture and path.read_bytes() == fixture.read_bytes()
        assert path.stat().st_mode & 0o777 == 0o600
        assert options["max_seconds"] == 600
        return {"text": "Hello world", "duration": 1}

    monkeypatch.setattr("mouthkeys.transcribe.Transcriber", Transcriber)
    monkeypatch.setattr("mouthkeys.file_transcription.transcribe_file", transcribe)
    result = benchmark._measure(
        {"config": config.__dict__, "fixture": str(fixture), "reference": "Hello world"}
    )
    assert result["accuracy"]["wer"] == result["accuracy"]["cer"] == 0
    assert result["fixture_sha256"] == hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert observed["config"].vocabulary_boosting == "off"
    assert observed["config"].pronunciation_matching == "off"
    assert observed["config"].ai_enabled == "off"
    assert not observed["fixture"].exists()
    assert "Hello world" not in json.dumps(result)
    assert str(tmp_path) not in json.dumps(result)


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
    from mouthkeys import engines

    artifact = tmp_path / "custom"
    artifact.mkdir()
    config = Config(model=str(artifact))
    item = report(config)
    item["sampled_peak_tree_rss_bytes"] = 20 * 2**20
    item["accuracy"] = {"wer": 0.25, "reference_sha256": "reference"}
    monkeypatch.setattr(benchmark, "measurements", lambda: [item])
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
    config = Config(device="cpu")
    item = report(config)
    item["hardware"]["display_adapter_pci_ids"] = ["different-gpu"]
    item["software"].pop("nvidia_driver")
    monkeypatch.setattr(benchmark, "_local_config", lambda cfg: (cfg, tmp_path))
    monkeypatch.setattr(benchmark, "_artifact_identity", lambda path: item["artifact"])
    assert benchmark.matching_measurement(config, [item]) == item


def test_model_browser_missing_weights_does_not_launch_benchmark(monkeypatch):
    from PySide6.QtWidgets import QComboBox, QLineEdit, QMessageBox, QWidget

    from mouthkeys.models_ui import ModelBrowser

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
